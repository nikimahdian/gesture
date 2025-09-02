# train.py
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml

from src.dataio import GestureDataset, CANONICAL_CLASSES, Sample
from src.features.mediapipe_hands import HandsWrapper
from src.features.landmark_norm import normalize_landmarks
from src.features.sequence_features import SequenceFeatureExtractor
from src.models.classical import ClassicalModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def extract_landmarks_for_samples(samples: List[Sample], det_conf: float) -> List[Tuple[np.ndarray, int]]:
    """
    Returns list of (normalized_landmarks(21,3), label_idx)
    Skips any sample with no detected hand.
    """
    out: List[Tuple[np.ndarray, int]] = []
    with HandsWrapper(static_image_mode=True, max_num_hands=1, detection_confidence=det_conf) as hands:
        for s in samples:
            img_bgr = s.image
            if img_bgr is None:
                # lazy load if dataset was constructed without images
                img_bgr = cv2.imread(str(s.path))
            if img_bgr is None:
                continue
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = hands.process_image(rgb)
            if not results:
                continue
            lmk = results[0].landmarks  # (21,3) normalized [0,1]
            lmk_norm = normalize_landmarks(lmk)  # (21,3)
            out.append((lmk_norm, int(s.label_numeric)))
    return out


def synthesize_sequence(lmk_norm: np.ndarray, label_idx: int, T: int) -> np.ndarray:
    """
    Build a synthetic temporal sequence from a single normalized frame.
    LEFT/RIGHT: linearly shift X by +/- total_shift over T.
    PALM / THUMB_*: repeat frame with tiny noise.
    """
    cls_name = CANONICAL_CLASSES[label_idx]
    seq = np.repeat(lmk_norm[None, :, :], T, axis=0)  # (T,21,3)

    if cls_name in ("LEFT", "RIGHT"):
        # Define sign: RIGHT => +dx, LEFT => -dx  (interpreting class name literally)
        sign = +1.0 if cls_name == "RIGHT" else -1.0
        total_shift = np.random.choice([0.06, 0.08, 0.10])
        per_step = (sign * total_shift) / max(1, T - 1)
        x = seq[:, :, 0]
        for t in range(1, T):
            x[t] = x[t - 1] + per_step  # accumulate
        # small jitter
        seq[:, :, 0] = x + np.random.normal(0, 0.002, size=x.shape).astype(np.float32)
    else:
        # Static classes: tiny Gaussian noise
        seq = seq + np.random.normal(0, 0.005, size=seq.shape).astype(np.float32)

    # clamp X/Y a bit (Z left as-is)
    seq[:, :, 0] = np.clip(seq[:, :, 0], -2.0, 2.0)
    seq[:, :, 1] = np.clip(seq[:, :, 1], -2.0, 2.0)
    return seq.astype(np.float32)


def build_sequences(
    samples: List[Sample], T: int
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns X (N,F), y (N,), feature_names
    """
    if len(samples) == 0:
        raise RuntimeError("No samples to build sequences from.")
    feat_extractor = SequenceFeatureExtractor(window_size=T)

    # First, extract a normalized landmark frame per sample via MediaPipe
    det_conf = 0.7
    norm_frames = extract_landmarks_for_samples(samples, det_conf)
    if len(norm_frames) == 0:
        raise RuntimeError("No training samples produced. Check dataset and MediaPipe installation.")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    feat_names: List[str] = []

    for lmk_norm, label_idx in norm_frames:
        # Make a few synthetic sequences per image to enrich dynamics
        repeats = 3 if CANONICAL_CLASSES[label_idx] in ("LEFT", "RIGHT") else 2
        for _ in range(repeats):
            seq = synthesize_sequence(lmk_norm, label_idx, T)  # (T,21,3)
            feats, names = feat_extractor.compute(seq)
            if not feat_names:
                feat_names = names
            X_list.append(feats.astype(np.float32))
            y_list.append(label_idx)

    X = np.vstack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, feat_names


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))

    dataset_root = Path(cfg["dataset_root"])
    output_dir = Path(cfg.get("output_dir", "outputs"))
    sequence_window = int(cfg.get("sequence_window", 15))

    log.info("Config: dataset_root=%s  output_dir=%s  sequence_window=%d",
             str(dataset_root), str(output_dir), sequence_window)

    # Load dataset (CSV parsing + Windows path resolution already handled in dataio)
    ds = GestureDataset(dataset_root=dataset_root, load_images=True)
    train_samples, val_samples = ds.load_all()

    # Prepare sequences/features
    X_train, y_train, feat_names = build_sequences(train_samples, sequence_window)
    X_val, y_val, _ = build_sequences(val_samples, sequence_window)

    # Train model (RandomForest per spec)
    rf_cfg = cfg.get("model", {}).get("classical", {})
    model = ClassicalModel.train_rf(
        X_train, y_train, feature_names=feat_names, sequence_window=sequence_window,
        n_estimators=int(rf_cfg.get("n_estimators", 300)),
        max_depth=rf_cfg.get("max_depth", None),
        class_weight=rf_cfg.get("class_weight", "balanced_subsample"),
    )

    # Eval + save reports
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model.evaluate(X_train, y_train, "train", output_dir)
    model.evaluate(X_val, y_val, "val", output_dir)

    # Save model + metadata in outputs\model\ to match inference.py expectations
    model.save(model_dir)

    # Also mirror metadata at outputs/ (optional convenience)
    meta_src = model_dir / "metadata.json"
    (output_dir / "metadata.json").write_text(meta_src.read_text(encoding="utf-8"), encoding="utf-8")

    log.info("Training complete.")


if __name__ == "__main__":
    main()
