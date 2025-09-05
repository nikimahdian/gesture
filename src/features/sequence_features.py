# src/features/sequence_features.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

TIP_IDXS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
MCP_IDXS = [2, 5, 9, 13, 17]   # thumb_mcp-ish, index_mcp, middle_mcp, ring_mcp, pinky_mcp
WRIST = 0

@dataclass
class SequenceFeatureExtractor:
    """
    Extracts temporal + pose features from a window of T landmark frames.
    Input to compute(): seq of shape (T, 21, 3) with normalized coords (see landmark_norm).
    """
    window_size: int = 15

    # Make the extractor callable so code that does `extractor(seq)` still works
    def __call__(self, seq: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        return self.compute(seq)

    def compute(self, seq: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Args:
            seq: (T, 21, 3) float32 normalized landmarks for a single sequence
        Returns:
            features: (F,) float32
            names: list[str] of feature names in the same order
        """
        seq = np.asarray(seq, dtype=np.float32)
        if seq.ndim != 3 or seq.shape[1:] != (21, 3):
            raise ValueError(f"Expected (T,21,3), got {seq.shape}")
        T = seq.shape[0]

        # ---------- Centroid motion ----------
        xs = seq[:, :, 0]       # (T, 21)
        ys = seq[:, :, 1]
        centroid_x = xs.mean(axis=1)  # (T,)
        centroid_y = ys.mean(axis=1)

        # velocity (frame-to-frame)
        vx = np.diff(centroid_x, prepend=centroid_x[0])
        vy = np.diff(centroid_y, prepend=centroid_y[0])

        net_dx = centroid_x[-1] - centroid_x[0]
        net_dy = centroid_y[-1] - centroid_y[0]
        path_len_x = np.sum(np.abs(vx))
        path_len = np.sum(np.sqrt(vx ** 2 + vy ** 2))
        mean_vx = float(np.mean(vx))
        frac_pos = float(np.mean(vx > 0))
        frac_neg = float(np.mean(vx < 0))
        disp_ratio = float(np.abs(net_dx) / (path_len_x + 1e-6))
        
        # Enhanced velocity/speed features for LEFT/RIGHT detection
        max_vx = float(np.max(np.abs(vx)))  # Maximum X velocity magnitude
        max_vy = float(np.max(np.abs(vy)))  # Maximum Y velocity magnitude
        mean_speed = float(np.mean(np.sqrt(vx ** 2 + vy ** 2)))  # Average speed
        max_speed = float(np.max(np.sqrt(vx ** 2 + vy ** 2)))    # Peak speed
        
        # Acceleration features (velocity changes)
        ax = np.diff(vx, prepend=vx[0])  # X acceleration
        ay = np.diff(vy, prepend=vy[0])  # Y acceleration
        mean_ax = float(np.mean(ax))     # Mean X acceleration
        mean_ay = float(np.mean(ay))     # Mean Y acceleration
        max_accel = float(np.max(np.sqrt(ax ** 2 + ay ** 2)))  # Peak acceleration
        
        # Directional consistency features
        vx_consistency = float(np.std(vx))  # Lower std = more consistent direction
        movement_energy = float(np.sum(vx ** 2))  # Total kinetic energy in X direction
        
        # Speed thresholds for movement detection
        high_speed_frames = float(np.mean(np.sqrt(vx ** 2 + vy ** 2) > 0.02))  # Fraction of high-speed frames
        stationary_frames = float(np.mean(np.sqrt(vx ** 2 + vy ** 2) < 0.005))  # Fraction of stationary frames

        # ---------- Pose / shape statistics ----------
        # Per-joint means/std over time (x,y)
        joint_mean_x = xs.mean(axis=0)   # (21,)
        joint_mean_y = ys.mean(axis=0)
        joint_std_x = xs.std(axis=0)
        joint_std_y = ys.std(axis=0)

        # Fingertip spread from wrist (XY distance), averaged over time
        wrist_xy = seq[:, WRIST, :2]  # (T,2)
        tip_xy = seq[:, TIP_IDXS, :2]  # (T,5,2)
        dist_tips_wrist = np.linalg.norm(tip_xy - wrist_xy[:, None, :], axis=2)  # (T,5)
        tip_spread_mean = dist_tips_wrist.mean(axis=0)  # (5,)
        tip_spread_std = dist_tips_wrist.std(axis=0)    # (5,)

        # Simple openness: average pairwise distance between finger tips (normalized)
        openness_frames = []
        for t in range(T):
            tips = tip_xy[t]  # (5,2)
            d = []
            for i in range(5):
                for j in range(i + 1, 5):
                    d.append(np.linalg.norm(tips[i] - tips[j]))
            openness_frames.append(np.mean(d))
        openness_mean = float(np.mean(openness_frames))
        openness_std = float(np.std(openness_frames))

        # Thumb up/down cues relative to wrist (smaller y = higher)
        thumb_y = seq[:, 4, 1]
        wrist_y = seq[:, 0, 1]
        thumb_rel_y = thumb_y - wrist_y  # negative -> thumb above wrist
        thumb_up_score = float(np.mean(thumb_rel_y < -0.2))
        thumb_down_score = float(np.mean(thumb_rel_y > 0.2))
        thumb_rel_y_mean = float(np.mean(thumb_rel_y))

        # ---------- Pack features ----------
        feats: List[float] = [
            float(centroid_x.mean()), float(centroid_y.mean()),
            float(centroid_x.std()), float(centroid_y.std()),
            float(net_dx), float(net_dy),
            float(path_len_x), float(path_len),
            float(mean_vx), float(frac_pos), float(frac_neg),
            float(disp_ratio),
            # Enhanced velocity/speed features
            max_vx, max_vy, mean_speed, max_speed,
            mean_ax, mean_ay, max_accel,
            vx_consistency, movement_energy,
            high_speed_frames, stationary_frames,
            openness_mean, openness_std,
            thumb_up_score, thumb_down_score, thumb_rel_y_mean,
        ]
        names: List[str] = [
            "centroid_x_mean", "centroid_y_mean",
            "centroid_x_std", "centroid_y_std",
            "net_dx", "net_dy",
            "path_len_x", "path_len",
            "mean_vx", "frac_pos", "frac_neg",
            "disp_ratio",
            # Enhanced velocity/speed feature names
            "max_vx", "max_vy", "mean_speed", "max_speed",
            "mean_ax", "mean_ay", "max_accel",
            "vx_consistency", "movement_energy",
            "high_speed_frames", "stationary_frames",
            "openness_mean", "openness_std",
            "thumb_up_score", "thumb_down_score", "thumb_rel_y_mean",
        ]

        # Append per-joint means/std (x,y)
        for j in range(21):
            feats.append(float(joint_mean_x[j])); names.append(f"j{j}_mx")
            feats.append(float(joint_mean_y[j])); names.append(f"j{j}_my")
            feats.append(float(joint_std_x[j]));  names.append(f"j{j}_sx")
            feats.append(float(joint_std_y[j]));  names.append(f"j{j}_sy")

        return np.asarray(feats, dtype=np.float32), names
