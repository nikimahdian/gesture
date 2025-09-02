# src/features/mediapipe_hands.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise RuntimeError(
        "mediapipe is required. Install it in your venv:\n"
        "  pip install mediapipe==0.10.14"
    ) from e

logger = logging.getLogger(__name__)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


@dataclass
class HandResult:
    """Single hand prediction from MediaPipe Hands."""
    landmarks: np.ndarray               # (21, 3) normalized image coords: x,y in [0,1], z relative
    handedness: str                     # 'Left' or 'Right'
    score: float                        # detection/handedness score
    world_landmarks: Optional[np.ndarray] = None  # (21, 3) in meters if available


def _to_rgb_if_needed(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Ensure RGB input for MediaPipe; returns (rgb, converted_flag)."""
    if img is None:
        raise ValueError("image is None")
    if img.ndim == 3 and img.shape[2] == 3:
        # Heuristic: assume BGR if coming from OpenCV capture
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), True
    return img, False


def normalize_landmarks(landmarks: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Zero-center at wrist (idx 0) and scale by palm size so scale is roughly invariant.
    Input: (21, 3) in normalized image coords.
    Output: (21, 3) normalized.
    """
    if landmarks.shape != (21, 3):
        raise ValueError(f"Expected (21,3) landmarks, got {landmarks.shape}")

    lm = landmarks.copy().astype(np.float32)
    wrist = lm[0, :2]
    lm[:, :2] -= wrist  # center x,y

    # Use distance between MCP joints of index (5) and pinky (17) as palm scale
    palm_scale = np.linalg.norm(lm[5, :2] - lm[17, :2]) + eps
    lm[:, :3] /= palm_scale
    return lm


def draw_landmarks(image_bgr: np.ndarray, hand_result: HandResult) -> None:
    """Draw hand landmarks in-place on a BGR image using MediaPipe's drawer."""
    # Build a fake NormalizedLandmarkList to reuse MediaPipe drawer
    from mediapipe.framework.formats import landmark_pb2

    lmk_list = landmark_pb2.NormalizedLandmarkList(
        landmark=[
            landmark_pb2.NormalizedLandmark(x=float(x), y=float(y), z=float(z))
            for (x, y, z) in hand_result.landmarks.tolist()
        ]
    )
    mp_drawing.draw_landmarks(
        image=image_bgr,
        landmark_list=lmk_list,
        connections=mp_hands.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
        connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
    )


class HandsWrapper:
    """
    Thin, stable wrapper around MediaPipe Hands with a consistent API the rest
    of the codebase can rely on.

    Methods you can call:
      - process_image(image_rgb_or_bgr) -> List[HandResult]
      - process(image_rgb_or_bgr)       -> alias of process_image
      - close()                         -> release resources
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        # Back-compat keyword names:
        detection_confidence: Optional[float] = None,
        tracking_confidence: Optional[float] = None,
    ) -> None:
        # Accept both naming styles
        det_conf = float(detection_confidence) if detection_confidence is not None else float(min_detection_confidence)
        trk_conf = float(tracking_confidence) if tracking_confidence is not None else float(min_tracking_confidence)

        self.static_image_mode = bool(static_image_mode)
        self.max_num_hands = int(max_num_hands)
        self.model_complexity = int(model_complexity)
        self.det_conf = float(det_conf)
        self.trk_conf = float(trk_conf)

        self._hands = mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.det_conf,
            min_tracking_confidence=self.trk_conf,
            model_complexity=self.model_complexity,
        )
        logger.info(
            "MediaPipe Hands initialized (static=%s, max=%d, det=%.2f, track=%.2f, complexity=%d)",
            self.static_image_mode, self.max_num_hands, self.det_conf, self.trk_conf, self.model_complexity
        )

    # --- main API ---
    def process_image(self, image_rgb_or_bgr: np.ndarray) -> List[HandResult]:
        """
        Run the hand detector/tracker on an RGB or BGR frame.
        Returns a (possibly empty) list of HandResult.
        """
        rgb, _ = _to_rgb_if_needed(image_rgb_or_bgr)
        results = self._hands.process(rgb)

        out: List[HandResult] = []
        if not results or results.multi_hand_landmarks is None:
            return out

        handed = results.multi_handedness or []
        world = results.multi_hand_world_landmarks or [None] * len(results.multi_hand_landmarks)

        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            # landmarks in normalized image coords
            lm = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)  # (21,3)

            # optional world landmarks
            w = world[i]
            w_np = None
            if w is not None:
                w_np = np.array([[lm.x, lm.y, lm.z] for lm in w.landmark], dtype=np.float32)

            # handedness & score
            if i < len(handed) and handed[i].classification:
                label = handed[i].classification[0].label  # 'Left' or 'Right'
                score = float(handed[i].classification[0].score)
            else:
                label, score = "Unknown", 0.0

            out.append(HandResult(landmarks=lm, world_landmarks=w_np, handedness=label, score=score))

        return out

    # alias for back-compat (some codebases call .process)
    def process(self, image_rgb_or_bgr: np.ndarray) -> List[HandResult]:
        return self.process_image(image_rgb_or_bgr)

    def close(self) -> None:
        try:
            self._hands.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
