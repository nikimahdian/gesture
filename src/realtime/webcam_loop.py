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
    static_lock: float = 0.60
    static_lock_frames: int = 9
    swipe_dx_thresh: float = 0.03
    swipe_ratio_thresh: float = 0.60
    swipe_min_frames: int = 5
    swipe_min_path: float = 0.08
    smoother_window: int = 5
    smoother_conf_threshold: float = 0.50  # static gestures threshold  
    dynamic_conf_threshold: float = 0.50  # threshold for left/right
    dynamic_hold_time: float = 5.0  # seconds to hold left/right prediction
    static_hold_time: float = 3  # seconds to hold static gestures


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

        # Use very low threshold for smoother, we'll apply our own thresholds later
        self.smoother = PredictionSmoother(window_size=loop_cfg.smoother_window, confidence_threshold=0.3)
        
        # Light EMA smoothing for landmarks (minimal smoothing)
        self.ema_alpha = 0.3  # lighter smoothing, more responsive
        self.smoothed_landmarks = None  # previous smoothed landmarks (21,3)
        
        # Dynamic gesture timing
        self.dynamic_gesture_start_time = None
        self.dynamic_gesture_type = None
        
        # Static gesture timing
        self.static_gesture_start_time = None
        self.static_gesture_type = None

    def _is_static_motion(self, xs: List[float]) -> bool:
        if len(xs) < 3:
            return True
        diffs = np.abs(np.diff(np.array(xs, dtype=np.float32)))
        return float(np.median(diffs)) < 0.005  # slightly more lenient

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

        # Use enhanced model with velocity features for better L/R detection
        # If model is uncertain about L/R, prefer swipe
        if lr_sum < 0.6 or lr_margin < 0.15:
            return int(swipe_pred)

        # If strong disagreement, use swipe override
        return pred if (pred == swipe_pred) else int(swipe_pred)
    
    def _smooth_landmarks_ema(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply Exponential Moving Average smoothing to landmarks"""
        if self.smoothed_landmarks is None:
            # First frame - initialize with current landmarks
            self.smoothed_landmarks = landmarks.copy()
            return landmarks
        
        # EMA formula: smoothed = alpha * current + (1 - alpha) * previous
        self.smoothed_landmarks = (self.ema_alpha * landmarks + 
                                 (1 - self.ema_alpha) * self.smoothed_landmarks)
        return self.smoothed_landmarks
    
    def _check_movement_threshold(self, landmarks: np.ndarray, gesture_type: str = "any") -> bool:
        """Check if movement exceeds threshold for dynamic gestures"""
        if self.smoothed_landmarks is None:
            return True
        
        # Calculate movement of wrist (landmark 0) in pixels
        # Assuming image coordinates are normalized [0,1], convert to pixels (approx 640x480)
        current_wrist = landmarks[0, :2] * np.array([640, 480])  # x, y only
        prev_wrist = self.smoothed_landmarks[0, :2] * np.array([640, 480])
        
        # For left/right gestures, focus on horizontal displacement only
        if gesture_type == "horizontal":
            horizontal_movement = abs(current_wrist[0] - prev_wrist[0])  # x-axis only
            return horizontal_movement >= self.movement_threshold
        else:
            # General movement for other gestures
            movement_distance = np.linalg.norm(current_wrist - prev_wrist)
            return movement_distance >= self.movement_threshold
    
    def _get_movement_metrics(self, landmarks: np.ndarray) -> dict:
        """Get detailed movement metrics for debugging"""
        if self.smoothed_landmarks is None:
            return {"horizontal": 0.0, "vertical": 0.0, "total": 0.0}
        
        current_wrist = landmarks[0, :2] * np.array([640, 480])
        prev_wrist = self.smoothed_landmarks[0, :2] * np.array([640, 480])
        
        horizontal = abs(current_wrist[0] - prev_wrist[0])
        vertical = abs(current_wrist[1] - prev_wrist[1])
        total = np.linalg.norm(current_wrist - prev_wrist)
        
        return {
            "horizontal": float(horizontal), 
            "vertical": float(vertical), 
            "total": float(total)
        }
    
    def _majority_vote(self, prediction: int) -> Optional[int]:
        """Apply majority voting over rolling window of predictions"""
        self.prediction_history.append(prediction)
        
        # Keep only last N predictions
        if len(self.prediction_history) > self.voting_window_size:
            self.prediction_history.pop(0)
        
        # Need enough predictions in window
        if len(self.prediction_history) < self.voting_window_size:
            return None
        
        # Find most common prediction
        counts = {}
        for pred in self.prediction_history:
            counts[pred] = counts.get(pred, 0) + 1
        
        # Return majority if it exists (more than half)
        max_count = max(counts.values())
        if max_count > self.voting_window_size // 2:
            return max(counts, key=counts.get)
        
        return None

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
                    
                    # Apply EMA smoothing to reduce shake/noise
                    smoothed_lm = self._smooth_landmarks_ema(lm)
                    
                    self.landmarks.append(smoothed_lm)
                    self.centroids_x.append(float(smoothed_lm[:,0].mean()))
                else:
                    self.landmarks.append(None)
                    self.centroids_x.append(0.0)
                    # Clear smoother when no hand detected to prevent stale predictions
                    self.smoother.labels.clear()
                    self.smoother.confidences.clear()
                    # Reset all gesture timers
                    self.dynamic_gesture_start_time = None
                    self.dynamic_gesture_type = None
                    self.static_gesture_start_time = None
                    self.static_gesture_type = None

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
                            LOGGER.debug("probs=%s swipe=%s final=%s conf=%.3f",
                                         np.round(probs,3).tolist(), swipe_pred, label_idx, conf)

                        self.smoother.add(label_idx, conf)
                        LOGGER.info("Raw prediction: %s conf=%.3f", self.model.classes[label_idx], conf)

                stable_idx, stable_conf = self.smoother.stable()
                if stable_idx is not None:
                    LOGGER.info("Smoother output: %s conf=%.3f", self.model.classes[stable_idx], stable_conf)
                else:
                    LOGGER.info("Smoother output: None")

                current_time = time.time()
                
                # Initialize display values
                label_idx = None
                conf = 0.0
                one_hot = [0,0,0,0,0]
                action_taken = False
                
                if stable_idx is not None:
                    detected_idx = int(stable_idx)
                    detected_conf = float(stable_conf)
                    
                    LOGGER.info("Processing gesture: %s (idx=%d) conf=%.3f", 
                               self.model.classes[detected_idx], detected_idx, detected_conf)
                    
                    # Handle LEFT/RIGHT gestures with hold timer
                    if detected_idx in [self.left_idx, self.right_idx]:
                        LOGGER.info("Taking LEFT/RIGHT path")
                        if detected_conf >= self.cfg.dynamic_conf_threshold:
                            # Start or continue timing
                            if (self.dynamic_gesture_type != detected_idx or 
                                self.dynamic_gesture_start_time is None):
                                self.dynamic_gesture_start_time = current_time
                                self.dynamic_gesture_type = detected_idx
                                LOGGER.info("Starting hold timer for %s (need %.1fs)", 
                                          self.model.classes[detected_idx], self.cfg.dynamic_hold_time)
                            
                            # Check hold duration
                            hold_duration = current_time - self.dynamic_gesture_start_time
                            progress = min(hold_duration / self.cfg.dynamic_hold_time, 1.0)
                            
                            # Show gesture with progress
                            label_idx = detected_idx
                            conf = progress
                            one_hot = [1 if i == detected_idx else 0 for i in range(5)]
                            
                            # Execute action only after full hold time
                            if hold_duration >= self.cfg.dynamic_hold_time:
                                action_taken = True
                                LOGGER.info("EXECUTING %s after %.1fs hold", 
                                          self.model.classes[detected_idx], hold_duration)
                                if detected_idx == self.right_idx:
                                    self.player.on_right()
                                elif detected_idx == self.left_idx:
                                    self.player.on_left()
                                
                                # Reset timer after action
                                self.dynamic_gesture_start_time = None
                                self.dynamic_gesture_type = None
                        else:
                            # Not confident enough - reset timer and show no gesture
                            self.dynamic_gesture_start_time = None
                            self.dynamic_gesture_type = None
                            # label_idx, conf, one_hot already initialized to None/0
                    
                    # Handle static gestures with hold timer
                    elif detected_idx in [self.palm_idx, self.thumb_up_idx, self.thumb_down_idx]:
                        LOGGER.info("Taking STATIC path")
                        # Reset dynamic timer when switching to static
                        self.dynamic_gesture_start_time = None
                        self.dynamic_gesture_type = None
                        
                        LOGGER.info("Static gesture check: %s conf=%.3f (need %.3f)", 
                                   self.model.classes[detected_idx], detected_conf, self.cfg.smoother_conf_threshold)
                        if detected_conf >= self.cfg.smoother_conf_threshold:
                            # Start or continue timing for static gesture
                            if (self.static_gesture_type != detected_idx or 
                                self.static_gesture_start_time is None):
                                self.static_gesture_start_time = current_time
                                self.static_gesture_type = detected_idx
                                LOGGER.info("Starting static hold timer for %s (need %.1fs)", 
                                          self.model.classes[detected_idx], self.cfg.static_hold_time)
                            else:
                                hold_duration = current_time - self.static_gesture_start_time
                                LOGGER.info("Continuing static hold: %s for %.1fs", 
                                          self.model.classes[detected_idx], hold_duration)
                            
                            # Check hold duration
                            hold_duration = current_time - self.static_gesture_start_time
                            progress = min(hold_duration / self.cfg.static_hold_time, 1.0)
                            
                            # Show gesture with progress
                            label_idx = detected_idx
                            conf = progress
                            one_hot = [1 if i == detected_idx else 0 for i in range(5)]
                            LOGGER.info("Setting static display: %s progress=%.3f", 
                                       self.model.classes[detected_idx], progress)
                            
                            # Execute action only after full hold time
                            if hold_duration >= self.cfg.static_hold_time:
                                action_taken = True
                                LOGGER.info("EXECUTING static %s after %.1fs hold", 
                                          self.model.classes[detected_idx], hold_duration)
                                
                                if detected_idx == self.palm_idx:
                                    self.player.on_palm()
                                elif detected_idx == self.thumb_up_idx:
                                    self.player.on_thumb_up()
                                elif detected_idx == self.thumb_down_idx:
                                    self.player.on_thumb_down()
                                
                                # Reset timer after action
                                self.static_gesture_start_time = None
                                self.static_gesture_type = None
                        else:
                            # Not confident enough - reset timer
                            self.static_gesture_start_time = None
                            self.static_gesture_type = None
                else:
                    # No stable prediction - reset both timers
                    self.dynamic_gesture_start_time = None
                    self.dynamic_gesture_type = None
                    self.static_gesture_start_time = None
                    self.static_gesture_type = None

                # HUD
                LOGGER.info("Display values: label_idx=%s conf=%.3f one_hot=%s", 
                           label_idx, conf, one_hot)
                if label_idx is not None:
                    label_text = self.model.classes[label_idx] if label_idx < len(self.model.classes) else str(label_idx)
                    LOGGER.info("Showing HUD: %s", label_text)
                    draw_hud(frame, label_text, conf, one_hot)
                else:
                    # Show "NO GESTURE" when not confident enough
                    LOGGER.info("Showing HUD: NO GESTURE")
                    draw_hud(frame, "NO GESTURE", 0.0, [0,0,0,0,0])

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
