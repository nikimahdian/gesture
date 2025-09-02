from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
import time
import logging
import numpy as np
import cv2

from ..features import HandsWrapper, normalize_landmarks, SequenceFeatureExtractor
from ..models.classical import ClassicalModel
from .smoothing import RollingBuffer, PredictionSmoother
from ..ui import VideoPlayer, draw_hud

LOGGER = logging.getLogger(__name__)

@dataclass
class LoopConfig:
    camera_index: int = 0
    buffer_size: int = 45
    target_fps: int = 30
    inference_interval: int = 5
    mirror_input: bool = True
    invert_swipe: bool = False
    static_lock: float = 0.70
    static_lock_frames: int = 9
    swipe_dx_thresh: float = 0.08
    swipe_ratio_thresh: float = 0.80
    swipe_min_frames: int = 10
    swipe_min_path: float = 0.15
    smoother_window: int = 7
    smoother_conf_threshold: float = 0.7


class WebcamGestureLoop:
    left_idx=0; right_idx=1; palm_idx=2; thumb_down_idx=3; thumb_up_idx=4

    def __init__(self, model_dir: Path, loop_cfg: LoopConfig, detection_confidence: float = 0.7, media_path: Optional[str] = None):
        self.cfg = loop_cfg
        self.model = ClassicalModel.load(model_dir)
        self.seq_extractor = SequenceFeatureExtractor(window_size=self.model.sequence_window)
        self.hands = HandsWrapper(static_image_mode=False, max_num_hands=1, min_detection_confidence=detection_confidence, min_tracking_confidence=0.5)

        self.frames = RollingBuffer(loop_cfg.buffer_size)
        self.landmarks = RollingBuffer(loop_cfg.buffer_size)  # normalized landmarks per frame (21,3)
        self.centroids_x = RollingBuffer(loop_cfg.buffer_size)
        self.static_lock_count = 0

        self.player = VideoPlayer(media_path=media_path)

        LOGGER.info("mirror_input=%s invert_swipe=%s classes=%s", loop_cfg.mirror_input, loop_cfg.invert_swipe, self.model.classes)

        self.smoother = PredictionSmoother(window_size=loop_cfg.smoother_window, confidence_threshold=loop_cfg.smoother_conf_threshold)

    def _is_static_motion(self, xs: List[float]) -> bool:
        if len(xs) < 3:
            return True
        diffs = np.abs(np.diff(np.array(xs, dtype=np.float32)))
        return float(np.median(diffs)) < 0.003

    def _compute_swipe(self) -> Tuple[Optional[int], dict]:
        xs = self.centroids_x.get()
        if len(xs) < 2:
            return None, {}
        win = min(self.cfg.buffer_size, 30)
        xs = np.array(xs[-win:], dtype=np.float32)

        if self._is_static_motion(xs.tolist()):
            return None, {"reason": "static_motion"}

        net_dx = float(xs[-1] - xs[0])
        path = float(np.sum(np.abs(np.diff(xs))))
        disp_ratio = float(abs(net_dx) / (path + 1e-6))

        metrics = {"net_dx": net_dx, "path": path, "disp_ratio": disp_ratio}
        if (abs(net_dx) >= self.cfg.swipe_dx_thresh and
            disp_ratio >= self.cfg.swipe_ratio_thresh and
            len(xs) >= self.cfg.swipe_min_frames and
            path >= self.cfg.swipe_min_path):
            pred = self.right_idx if (net_dx > 0) else self.left_idx
            if self.cfg.invert_swipe:
                pred = self.left_idx if pred == self.right_idx else self.right_idx
            return pred, metrics
        return None, metrics

    def _fuse(self, model_probs: np.ndarray, swipe_pred: Optional[int]) -> int:
        pred = int(np.argmax(model_probs))
        conf = float(np.max(model_probs))
        # STATIC-FIRST gating
        static_indices = [self.palm_idx, self.thumb_down_idx, self.thumb_up_idx]
        static_max = float(np.max(model_probs[static_indices]))
        if static_max >= self.cfg.static_lock and self.static_lock_count <= 0:
            self.static_lock_count = self.cfg.static_lock_frames
            return int(np.argmax(model_probs[static_indices]) + min(static_indices))

        if self.static_lock_count > 0:
            self.static_lock_count -= 1
            return pred

        if swipe_pred is None:
            return pred

        lr_sum = float(model_probs[self.left_idx] + model_probs[self.right_idx])
        lr_margin = float(abs(model_probs[self.left_idx] - model_probs[self.right_idx]))

        # If model is uncertain about L/R, prefer swipe
        if lr_sum < 0.6 or lr_margin < 0.15:
            return int(swipe_pred)

        # If strong disagreement, only override with very confident swipe (we don't have swipe_conf here, use heuristic)
        return pred if (pred == swipe_pred) else int(swipe_pred)

    def run(self, camera_index: Optional[int] = None, show_debug: bool = True) -> None:
        cam_idx = camera_index if camera_index is not None else self.cfg.camera_index
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            LOGGER.error("Failed to open camera index %s", cam_idx)
            return

        target_delay = 1.0 / max(1, self.cfg.target_fps)
        last_infer = 0
        t0 = time.time()

        try:
            while True:
                t_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                if self.cfg.mirror_input:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands = self.hands.process_image(rgb)
                if hands:
                    lm = normalize_landmarks(hands[0].landmarks)
                    self.landmarks.append(lm)
                    self.centroids_x.append(float(lm[:,0].mean()))
                else:
                    self.landmarks.append(None)
                    self.centroids_x.append(0.0)

                self.frames.append(frame)

                label_idx = None
                conf = 0.0
                one_hot = [0,0,0,0,0]

                if (len(self.frames) % max(1, self.cfg.inference_interval)) == 0 and len(self.landmarks) >= self.model.sequence_window:
                    # Build last sequence_window of valid landmarks
                    valid = [lm for lm in self.landmarks.get() if lm is not None]
                    if len(valid) >= self.model.sequence_window:
                        seq = np.stack(valid[-self.model.sequence_window:], axis=0)
                        feats, _ = self.seq_extractor(seq)
                        X = feats.reshape(1, -1)
                        probs = self.model.predict_proba(X)[0]

                        swipe_pred, swipe_metrics = self._compute_swipe()

                        pred_idx = self._fuse(probs, swipe_pred)

                        label_idx = int(pred_idx)
                        conf = float(probs[label_idx])
                        one_hot = [1 if i == label_idx else 0 for i in range(len(probs))]

                        if LOGGER.isEnabledFor(logging.DEBUG):
                            LOGGER.debug("probs=%s swipe=%s metrics=%s final=%s conf=%.3f",
                                         np.round(probs,3).tolist(), swipe_pred, swipe_metrics, label_idx, conf)

                        self.smoother.add(label_idx, conf)

                stable_idx, stable_conf = self.smoother.stable()
                if stable_idx is not None:
                    label_idx = int(stable_idx)
                    conf = float(stable_conf)
                    one_hot = [1 if i == label_idx else 0 for i in range(5)]

                    # map to actions
                    if label_idx == self.right_idx:
                        self.player.on_right()
                    elif label_idx == self.left_idx:
                        self.player.on_left()
                    elif label_idx == self.palm_idx:
                        self.player.on_palm()
                    elif label_idx == self.thumb_up_idx:
                        self.player.on_thumb_up()
                    elif label_idx == self.thumb_down_idx:
                        self.player.on_thumb_down()

                # HUD
                if label_idx is not None:
                    label_text = self.model.classes[label_idx] if label_idx < len(self.model.classes) else str(label_idx)
                    draw_hud(frame, label_text, conf, one_hot)

                if show_debug:
                    cv2.imshow("Gesture HMI", frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break

                # timing
                t_end = time.time()
                delay = target_delay - (t_end - t_start)
                if delay > 0:
                    time.sleep(delay)
        finally:
            cap.release()
            cv2.destroyAllWindows()
