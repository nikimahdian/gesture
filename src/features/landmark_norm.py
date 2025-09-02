# src/features/landmark_norm.py
from __future__ import annotations

import numpy as np

WRIST = 0
MIDDLE_MCP = 9

def normalize_landmarks(lm: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize 21x3 MediaPipe landmarks:
      - translate so wrist (idx 0) is at the origin
      - scale by distance wrist -> middle_mcp (idx 9) in the XY plane
    Keep the sign of x/y (no abs), preserve z scale relative to XY distance.

    Args:
        lm: (21,3) float array with x,y in [0,1] image coords, z normalized by MP
    Returns:
        (21,3) float32 normalized landmarks
    """
    lm = np.asarray(lm, dtype=np.float32)
    if lm.shape != (21, 3):
        raise ValueError(f"Expected (21,3) landmarks, got {lm.shape}")

    origin = lm[WRIST, :3].copy()
    centered = lm - origin  # translate

    ref = centered[MIDDLE_MCP, :3]
    scale = np.sqrt(ref[0] ** 2 + ref[1] ** 2) + eps  # use XY only
    normalized = centered / scale
    return normalized.astype(np.float32)
