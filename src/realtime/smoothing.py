from __future__ import annotations
from collections import deque
from typing import Deque, Tuple, Optional
import numpy as np

class RollingBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self._buf: Deque = deque(maxlen=maxlen)

    def append(self, item):
        self._buf.append(item)

    def get(self):
        return list(self._buf)

    def __len__(self):
        return len(self._buf)

    def clear(self):
        self._buf.clear()


class PredictionSmoother:
    def __init__(self, window_size: int = 7, confidence_threshold: float = 0.7):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.labels: Deque[int] = deque(maxlen=window_size)
        self.confidences: Deque[float] = deque(maxlen=window_size)

    def add(self, label_idx: int, confidence: float):
        self.labels.append(label_idx)
        self.confidences.append(confidence)

    def stable(self) -> Tuple[Optional[int], float]:
        if len(self.labels) < max(3, self.window_size // 2):
            return None, 0.0
        # Majority vote weighted by confidence
        labels = np.array(self.labels, dtype=np.int32)
        confs = np.array(self.confidences, dtype=np.float32)
        scores = {}
        for l, c in zip(labels, confs):
            scores[l] = scores.get(l, 0.0) + float(c)
        best_label = max(scores.items(), key=lambda kv: kv[1])[0]
        avg_conf = float(np.mean(confs[labels == best_label])) if np.any(labels == best_label) else 0.0
        if avg_conf >= self.confidence_threshold:
            return int(best_label), float(avg_conf)
        return None, float(avg_conf)
