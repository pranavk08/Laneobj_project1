# RGS-LaneNet Project
A predictive lane detection system with road-width division and ground-surface sensing.

## Setup
- Create a virtual environment (Windows PowerShell):
```
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Run
- Default uses the bundled sample video and tries CUDA if available. Force CPU with `--device cpu`.
```
python run_demo.py --source project_video.mp4 --device cpu
```
- Use a webcam:
```
python run_demo.py --source 0 --device cpu
```
- If your CPU is too slow for deep segmentation, switch to fast mode (simple color heuristic):
```
python run_demo.py --source project_video.mp4 --device cpu --segmentation fast
```

## Enforce a 7.5-inch lane width
To measure/enforce a physical width (e.g., 7.5 inches), you must provide a pixel-to-inch calibration at the bottom of the frame.
Pass the scale via `--pixels-per-inch`. Example (7.5-inch target width with 20 px/in):
```
python run_demo.py --source project_video.mp4 --device cpu --pixels-per-inch 20 --target-lane-width-in 7.5 --width-tolerance-in 0.5
```
Notes:
- If only one lane line is detected, the other will be synthesized at the target width.
- If both are detected but their measured width deviates by more than the tolerance, the right line is adjusted to match the target width.
- The current approach assumes a roughly constant pixel scale near the bottom of the frame. For precise measurements under perspective, provide a homography or per-row scale.

## What’s inside (baseline)
- Depth: MiDaS (DPT_Hybrid) via torch.hub with graceful offline fallback
- Segmentation: DeepLabV3-ResNet101 (COCO) to derive a road-like mask (obstacles inverted)
- Lane detection: classical Canny + Hough transform
- Turn predictor: simple curvature proxy integrated for demo overlay
- Ground model: heuristic combining obstacle density and depth stability

## Project Structure
- run_demo.py : Main pipeline loop
- utils/ : Helper functions (preprocess, ground_model, road_model, visualize)
- models/ : Depth and prediction models (depth_model, segmentation, turn_predictor)
- requirements.txt : dependencies
