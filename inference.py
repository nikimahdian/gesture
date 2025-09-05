from __future__ import annotations
import argparse, logging, yaml
from pathlib import Path
from src.realtime.webcam_loop import WebcamGestureLoop, LoopConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--dataset-root", type=str, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--media", type=str, default=None, help="Optional video/media path for VLC player")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Loop config
    rt = cfg["realtime"]
    loop_cfg = LoopConfig(
        camera_index=int(rt.get("camera_index", 0)),
        buffer_size=int(rt.get("buffer_size", 45)),
        target_fps=int(rt.get("target_fps", 30)),
        inference_interval=int(rt.get("inference_interval", 5)),
        mirror_input=bool(rt.get("mirror_input", True)),
        invert_swipe=bool(rt.get("invert_swipe", False)),
        static_lock=float(rt.get("static_lock", 0.70)),
        static_lock_frames=int(rt.get("static_lock_frames", 9)),
        swipe_dx_thresh=float(rt.get("swipe_dx_thresh", 0.08)),
        swipe_ratio_thresh=float(rt.get("swipe_ratio_thresh", 0.80)),
        swipe_min_frames=int(rt.get("swipe_min_frames", 10)),
        swipe_min_path=float(rt.get("swipe_min_path", 0.15)),
        # Gesture timing configuration
        dynamic_conf_threshold=float(rt.get("dynamic_conf_threshold", 0.50)),
        dynamic_hold_time=float(rt.get("dynamic_hold_time", 4.0)),
        static_hold_time=float(rt.get("static_hold_time", 3.0)),
        smoother_window=int(rt["smoother"].get("window_size", 7)),
        smoother_conf_threshold=float(rt["smoother"].get("confidence_threshold", 0.50)),
    )

    camera = args.camera if args.camera is not None else loop_cfg.camera_index
    model_dir = Path(cfg.get("output_dir", "outputs")) / "model"
    detection_confidence = float(cfg.get("detection_confidence", 0.7))

    loop = WebcamGestureLoop(model_dir=model_dir, loop_cfg=loop_cfg, detection_confidence=detection_confidence, media_path=args.media)
    loop.run(camera_index=camera, show_debug=True)

if __name__ == "__main__":
    main()
