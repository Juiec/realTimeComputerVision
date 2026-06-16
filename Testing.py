import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("./yoloe-26n-seg-pf.pt")

pose_model = None
try:
    pose_model = YOLO("./yolo26n-pose.pt")
except Exception as exc:
    print("Pose model not loaded:", exc)

"""
mp_hands = None
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
except Exception as exc:
    print("Hand model not loaded:", exc)
"""
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (5, 11),
    (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
]

"""
if mp_hands is not None:

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
"""
MODE_OBJECTS = "objects"
MODE_POSE = "pose"
MODE_HANDS = "hands"
MODE_BOTH = "both"
MODE_ALL = "all"
MODE_NONE = "none"

def to_numpy(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)

def draw_pose(frame, keypoints, conf_threshold=0.3):
    if keypoints is None:
        return frame

    if hasattr(keypoints, "xy"):
        xy = to_numpy(keypoints.xy)
        conf = to_numpy(getattr(keypoints, "conf", np.ones(xy.shape[:2], dtype=np.float32)))
    else:
        arr = to_numpy(keypoints)
        if arr.ndim == 3:
            xy = arr[..., :2]
            conf = arr[..., 2] if arr.shape[-1] >= 3 else np.ones(arr.shape[:2], dtype=np.float32)
        elif arr.ndim == 2:
            xy = arr[:, :2][None, ...]
            conf = arr[:, 2][None, ...] if arr.shape[-1] >= 3 else np.ones((1, arr.shape[0]))
        else:
            return frame

    for i in range(xy.shape[0]):
        pose = xy[i]
        score = conf[i]
        for j, (x, y) in enumerate(pose):
            if score[j] < conf_threshold:
                continue
            cv2.circle(frame, (int(x), int(y)), 3, (0, 120, 0), -1)

        for a, b in POSE_CONNECTIONS:
            if score[a] < conf_threshold or score[b] < conf_threshold:
                continue
            pt1 = tuple(map(int, pose[a]))
            pt2 = tuple(map(int, pose[b]))
            cv2.line(frame, pt1, pt2, (7, 81, 142), 2)

    return frame

"""
def draw_hands(frame, results):
    if results is None or not getattr(results, "multi_hand_landmarks", None):
        return frame
    for hand_landmarks in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame
"""
def detect_objects(frame, mode=MODE_BOTH):

    if mode in (MODE_OBJECTS, MODE_BOTH, MODE_ALL):
        results = model(frame)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = results.names[int(box.cls[0])] + f" {box.conf[0]:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (219, 95, 41), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (219, 95, 41), 2)

    if mode in (MODE_POSE, MODE_BOTH, MODE_ALL) and pose_model is not None:
        pose_results = pose_model(frame)[0]
        frame = draw_pose(frame, pose_results.keypoints)

    """
    if mode in (MODE_HANDS, MODE_ALL) and hands is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb)
        frame = draw_hands(frame, hand_results)
    """
    return frame

def main():
    cap = cv2.VideoCapture(0)
    """
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    """

    mode = MODE_BOTH
    print("Press o=objects, p=pose, h=hands, b=both, a=all, n=none, q=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = detect_objects(frame, mode)
        cv2.putText(frame, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Press[Q] to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.imshow("Oh Hello There.", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("o"):
            mode = MODE_OBJECTS
        elif key == ord("p"):
            mode = MODE_POSE
        elif key == ord("h"):
            mode = MODE_HANDS
        elif key == ord("b"):
            mode = MODE_BOTH
        elif key == ord("a"):
            mode = MODE_ALL
        elif key == ord("n"):
            mode = MODE_NONE

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
