from __future__ import annotations
from typing import List
import cv2
import numpy as np

def draw_hud(frame_bgr: np.ndarray, label_text: str, confidence: float, one_hot: List[int]) -> None:
    h, w = frame_bgr.shape[:2]
    txt = f"{label_text} {confidence:.2f}"
    cv2.rectangle(frame_bgr, (10, 10), (max(200, 10+8*len(txt)), 70), (0,0,0), -1)
    cv2.putText(frame_bgr, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)

    # one-hot bar
    oh = np.array(one_hot, dtype=np.int32)
    for i, v in enumerate(oh):
        x0 = 20 + i*30
        y0 = 80
        cv2.rectangle(frame_bgr, (x0, y0), (x0+25, y0+25), (0,255,0) if v else (0,100,0), -1)
        cv2.putText(frame_bgr, str(i), (x0+5, y0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
