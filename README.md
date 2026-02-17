Video tracking with YOLO + ByteTrack

Quick start

1. Ensure dependencies are installed (preferably in a conda/venv):

```powershell
pip install -r requirements.txt
```

2. Run the tracking script (example):

```powershell
python track.py --source_weights_path yolo12n.pt --source_video_path v1.mp4 --target_video_path out.mp4
```

- `--source_weights_path`: path to the model weights (e.g. `yolo12n.pt`).
- `--source_video_path`: input video (e.g. `v1.mp4`).
- `--target_video_path`: output video file (e.g. `out.mp4`).
- Use `--confidence_threshold` and `--iou_threshold` to tune detection.

Notes

- Press `q` in any display window to quit early if the script shows frames interactively.
- If you run into import errors due to filename conflicts, avoid using script names that shadow stdlib modules (e.g., `code.py`).
