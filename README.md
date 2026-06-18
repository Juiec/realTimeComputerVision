# Real-Time Object & Pose Detection

A Python/OpenCV project that uses YOLO models to perform real-time webcam object detection and pose detection. It can draw bounding boxes for detected objects, draw body keypoints/skeletons, and show simple hand/forearm movement feedback such as **HAND SIDEWAYS** or **HAND TURNING**.

## Features

- Real-time webcam capture using OpenCV
- Object detection with a YOLO model
- Pose detection with a YOLO pose model
- Body skeleton/keypoint drawing
- Forearm angle calculation for left and right arms
- Simple hand movement feedback:
  - `HAND SIDEWAYS`
  - `HAND TURNING`
- Keyboard controls to switch detection modes while the program is running

## Project Structure

```text
.
├── main.py
├── addons.py
└── model/
    ├── yolo26n.pt
    └── yolo26n-pose.pt
```

### File Overview

- `main.py`  
  Main application file. It opens the webcam, loads YOLO models, processes each frame, draws detection results, and handles keyboard controls.

- `addons.py`  
  Helper functions for pose-based hand/forearm angle detection and movement feedback.

- `model/`  
  Folder expected to contain the YOLO model files used by the application.

## Requirements

Install the required Python packages:

```bash
pip install opencv-python numpy ultralytics
```

You also need the following model files in the `model` folder:

```text
./model/yolo26n.pt
./model/yolo26n-pose.pt
```

> Note: If the pose model fails to load, the app will still run, but pose-related detection and feedback will not be available.

## How to Run

From the project folder, run:

```bash
python main.py
```

The program will open your default webcam and display the detection window.

## Keyboard Controls

While the app is running, press:

| Key | Mode | Description |
|---|---|---|
| `o` | Objects | Object detection only |
| `p` | Pose | Pose detection only |
| `h` | Hands | Hand mode placeholder; MediaPipe hand code is currently commented out |
| `b` | Both | Object + pose detection |
| `a` | All | All supported detection modes |
| `n` | None | No detection overlay |
| `q` | Quit | Close the app |

Default mode: `both`

## How It Works

1. `main.py` loads the object detection YOLO model from:

   ```python
   ./model/yolo26n.pt
   ```

2. It attempts to load the pose detection model from:

   ```python
   ./model/yolo26n-pose.pt
   ```

3. The webcam frame is flipped horizontally for a mirror-like view.

4. Depending on the active mode, the program:
   - detects objects and draws bounding boxes,
   - detects human pose keypoints and draws skeleton lines,
   - calculates forearm angles using elbow and wrist keypoints,
   - displays hand movement feedback when the forearm angle changes enough.

## Hand / Forearm Feedback Logic

The helper function `get_forearm_angle()` calculates the angle between the elbow and wrist.

The helper function `get_hand_turn_feedback()` compares the current forearm angle with the previous angle:

- If the hand is close to horizontal, it may show `HAND SIDEWAYS`.
- If the angle changes more than the threshold, it may show `HAND TURNING`.

Default values:

```python
change_threshold = 20.0
sideways_band = 25.0
```

## Notes

- The MediaPipe hand tracking code is currently commented out, so the `hands` mode may not display hand landmarks unless that section is restored.
- The program uses webcam index `0`. If you have multiple cameras, you may need to change this line in `main.py`:

```python
cap = cv2.VideoCapture(0)
```

- Optional webcam resolution settings are included but commented out.

## Troubleshooting

### Webcam does not open

Check that:

- your webcam is connected,
- no other app is using the webcam,
- the correct camera index is used in `cv2.VideoCapture()`.

### Model file not found

Make sure the model files are in the correct location:

```text
model/yolo26n.pt
model/yolo26n-pose.pt
```

### Pose detection does not work

If you see a message like:

```text
Pose model not loaded: ...
```

check that `yolo26n-pose.pt` exists and is compatible with the `ultralytics` package.

## Possible Improvements

- Re-enable and complete MediaPipe hand tracking
- Add command-line arguments for camera index and model paths
- Save detection output to a video file
- Add confidence threshold settings for object detection
- Improve multi-person hand feedback tracking
- Move repeated state values into a class for cleaner code

## License

No license has been specified yet. Add a license if you plan to share or publish this project.
