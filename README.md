# Real-time Hand-Gesture HMI (Webcam → Video Player)

Production-ready Python project for real-time hand-gesture control of a video player. Runs on CPU with MediaPipe Hands, OpenCV, NumPy, and scikit-learn. Clean Windows pathing via `pathlib.Path` (no hardcoded forward slashes).

> Target: **≥ 24 FPS** end-to-end on CPU using inference every K=5 frames and lightweight feature extraction.

## Repo Layout

```
configs/
  default.yaml
src/
  dataio/
    __init__.py
    csv_utils.py
    dataset.py
  features/
    __init__.py
    mediapipe_hands.py
    landmark_norm.py
    sequence_features.py
  models/
    __init__.py
    classical.py
  realtime/
    __init__.py
    smoothing.py
    webcam_loop.py
  ui/
    __init__.py
    player.py
    overlay.py
train.py
inference.py
tests/
  test_csv_utils.py
  test_path_resolution.py
  test_sequence_features.py
  test_action_mapping.py
  test_realtime_logic.py
README.md
```

## Setup

```powershell
# Windows (Python 3.10.11)
python -m venv .venv
.venv\\Scripts\\activate

pip install --upgrade pip
pip install opencv-python mediapipe scikit-learn numpy pyyaml python-vlc matplotlib
```

> If `python-vlc` is not installed, the player falls back to logging-only mode. Install VLC from https://www.videolan.org/vlc/ and ensure it's on PATH for full control.

## Dataset (Windows)

Place your dataset at:

- `C:\Users\nikim\OneDrive\Desktop\archive`
  - `train.csv`, `val.csv`
  - `train\train\...images...`
  - `val\val\...images...`

**CSV Parsing Rules**

- Delimiter is auto-detected (prefers `;` on failure).
- Filename column auto-detected (first column whose values look like stems/paths).
- Labels may be text (`label`, `class`, or equivalent) and/or numeric.
- Filenames may include a numeric prefix like `12_WIN_...` — code strips a single leading `^\d+_` **before** resolving.
- Accepts stems or relative paths; probes extensions in order: `.jpg`, `.jpeg`, `.png`.
- Builds absolute paths as: `Path(DATASET_ROOT) / ("train/train" | "val/val") / <clean_filename>.<ext>`
- Validates each row; missing files are logged and skipped.
- Returns tuples: `(absolute_path: Path, image, label_text, label_numeric, one_hot)`

Canonical classes (order is consistent project-wide):

```
0 = LEFT         (hand moves RIGHT → seek forward)
1 = RIGHT        (hand moves LEFT  → seek backward)
2 = PALM         (open palm → pause/play)
3 = THUMB_DOWN   (volume down)
4 = THUMB_UP     (volume up)
```

## Training

```powershell
python train.py --config configs\\default.yaml
```

- Extracts MediaPipe landmarks (static mode) for each image.
- Synthesizes temporal sequences for dynamic classes (LEFT/RIGHT) by linear x-shifts over `T=15` frames; **no horizontal-flip augmentation** is used to preserve left/right semantics.
- Static classes (PALM/THUMB_*) repeat frames with tiny Gaussian noise.
- Features (`src/features/sequence_features.py`): centroid motion, velocities, displacement, straightness, pose stats, openness/spread, thumb up/down cues.
- Model: RandomForest (n_estimators=300, class_weight="balanced_subsample").
- Saves:
  - `outputs/model/model.pkl`
  - `outputs/model/metadata.json` with `classes`, `sklearn_version`, `feature_names`, `sequence_window`
  - `outputs/metrics/metrics.json` and `confusion_matrix.png`

## Real-time Inference (Webcam + Player)

```powershell
python inference.py --config configs\\default.yaml --camera 0 --media "C:\\path\\to\\video.mp4"
```

- Mirror and process the same frame (so what you see matches the motion sign).
- STATIC-FIRST gating + hysteresis prevents tiny motions from overriding static gestures.
- Swipe fusion uses centroid trajectory to resolve LEFT/RIGHT when the classifier is uncertain.
- Overlay HUD shows label + confidence and a 5-element one-hot command row.
- Player mapping:
  - RIGHT  → seek +5s
  - LEFT   → seek −5s
  - THUMB_UP   → volume +5%
  - THUMB_DOWN → volume −5%
  - PALM   → pause/play

### Troubleshooting

- If LEFT/RIGHT feel inverted due to camera/OS, set `realtime.invert_swipe: true` in YAML.
- If MediaPipe is missing, install `mediapipe`. For NVIDIA GPUs, this project still uses CPU-only.
- If `scikit-learn` version differs from the saved model, a clear WARNING is printed when loading.
- To meet FPS:
  - Increase `realtime.inference_interval` (e.g., 5).
  - Reduce `realtime.buffer_size` (e.g., 30–36).

## Tests

Run the unit tests (they validate CSV parsing, path resolution, features, debounce, and smoothing):

```powershell
pip install pytest
pytest -q
```

## Notes

- All paths use `pathlib.Path` (Windows-safe). No forward slashes are hardcoded.
- No horizontal-flip augmentation during training (left/right semantics preserved).
- Logging: INFO for runtime, DEBUG for detailed diagnostics (set `PYTHONLOGLEVEL=DEBUG` to enable).
