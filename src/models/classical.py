# src/models/classical.py
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import sklearn

logger = logging.getLogger(__name__)

CANONICAL_CLASSES = ["LEFT", "RIGHT", "PALM", "THUMB_DOWN", "THUMB_UP"]


@dataclass
class ClassicalModel:
    clf: Any
    classes: List[str]
    feature_names: List[str]
    sequence_window: int

    # ---------- Train / Eval ----------
    @staticmethod
    def train_rf(
        X: np.ndarray, y: np.ndarray, feature_names: List[str], sequence_window: int,
        n_estimators: int = 300, max_depth: Optional[int] = None, class_weight: str = "balanced_subsample",
        random_state: int = 42
    ) -> "ClassicalModel":
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_state,
        )
        clf.fit(X, y)
        return ClassicalModel(clf=clf, classes=CANONICAL_CLASSES, feature_names=feature_names, sequence_window=sequence_window)

    def evaluate(self, X: np.ndarray, y: np.ndarray, split_name: str, out_dir: Path) -> Tuple[float, str, np.ndarray]:
        y_pred = self.clf.predict(X)
        acc = float(np.mean(y_pred == y))
        
        # Only use classes that are actually present in this dataset split
        unique_classes = sorted(set(y))
        present_class_names = [self.classes[i] for i in unique_classes if i < len(self.classes)]
        
        logger.info("[%s] Classes present: %s", split_name, present_class_names)
        
        report = classification_report(y, y_pred, target_names=present_class_names, 
                                     labels=unique_classes, digits=4, zero_division=0)
        cm = confusion_matrix(y, y_pred, labels=unique_classes)
        
        # Save
        (out_dir / f"classification_report_{split_name}.txt").write_text(report, encoding="utf-8")
        np.savetxt(out_dir / f"cm_{split_name}.csv", cm.astype(int), fmt="%d", delimiter=",")
        logger.info("[%s] accuracy=%.4f", split_name, acc)
        logger.info("[%s] report:\n%s", split_name, report)
        return acc, report, cm

    # ---------- Save / Load ----------
    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        # Persist model
        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(self.clf, f)
        # Persist metadata
        meta = {
            "classes": self.classes,
            "sklearn_version": sklearn.__version__,
            "feature_names": self.feature_names,
            "sequence_window": int(self.sequence_window),
        }
        with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved model -> %s", str(model_dir / "model.pkl"))
        logger.info("Saved metadata -> %s", str(model_dir / "metadata.json"))

    @staticmethod
    def load(model_dir: Path) -> "ClassicalModel":
        model_dir = Path(model_dir)
        model_path = model_dir / "model.pkl"
        meta_path = model_dir / "metadata.json"
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        classes = meta.get("classes", CANONICAL_CLASSES)
        feature_names = meta.get("feature_names", [])
        sequence_window = int(meta.get("sequence_window", 15))
        # Version warning
        meta_ver = meta.get("sklearn_version")
        if meta_ver and meta_ver != sklearn.__version__:
            logger.warning(
                "sklearn version mismatch: trained with %s, current %s. Proceeding anyway.",
                meta_ver, sklearn.__version__,
            )
        return ClassicalModel(clf=clf, classes=classes, feature_names=feature_names, sequence_window=sequence_window)

    # ---------- Predict ----------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)
