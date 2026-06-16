import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 object model
model = YOLO("./yoloe-26n-seg-pf.pt")

pose_model = None
try:
    pose_model = YOLO("./yolo26n-pose.pt")
except Exception as exc:
    print("Pose model not loaded:", exc)

POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (5, 11),
    (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
]


def draw_pose(frame, keypoints, conf_threshold=0.3):
    if keypoints is None:
        return

    def to_numpy(x):
        if x is None:
            return None
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            x = x.numpy()
        return np.asarray(x)

    if hasattr(keypoints, "xy"):
        xy = to_numpy(keypoints.xy)
        conf = to_numpy(getattr(keypoints, "conf", None))
    else:
        keypoints = to_numpy(keypoints)
        if keypoints.ndim == 3:
            # batch of poses: [batch, num_keypoints, 2 or 3]
            for person in keypoints:
                draw_pose(frame, person, conf_threshold)
            return
        xy = keypoints[:, :2] if keypoints.ndim == 2 and keypoints.shape[1] >= 2 else keypoints
        conf = keypoints[:, 2] if keypoints.ndim == 2 and keypoints.shape[1] >= 3 else None

    if xy is None or xy.ndim < 2:
        return

    if xy.ndim == 3:
        for i in range(xy.shape[0]):
            sub_xy = xy[i]
            sub_conf = conf[i] if conf is not None and conf.ndim == 2 else conf
            draw_pose(frame, type("K", (), {"xy": sub_xy, "conf": sub_conf})(), conf_threshold)
        return

    if conf is None:
        conf = np.ones((xy.shape[0],), dtype=np.float32)
    elif conf.ndim == 2 and conf.shape[0] == 1:
        conf = conf[0]

    if xy.shape[0] != conf.shape[0]:
        return

    for point, score in zip(xy, conf):
        x, y = int(point[0]), int(point[1])
        if score < conf_threshold:
            continue
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    for a, b in POSE_CONNECTIONS:
        if a < len(xy) and b < len(xy):
            xa, ya = int(xy[a][0]), int(xy[a][1])
            xb, yb = int(xy[b][0]), int(xy[b][1])
            conf_a = float(conf[a])
            conf_b = float(conf[b])
            if conf_a >= conf_threshold and conf_b >= conf_threshold:
                cv2.line(frame, (xa, ya), (xb, yb), (255, 0, 0), 2)


MODE_OBJECTS = "objects"
MODE_POSE = "pose"
MODE_BOTH = "both"
MODE_NONE = "none"

def detect_objects(frame, mode=MODE_BOTH):
    detected_objects = []

    if mode in (MODE_POSE, MODE_BOTH) and pose_model is not None:
        pose_results = pose_model(frame)
        for pr in pose_results:
            if hasattr(pr, "keypoints") and pr.keypoints is not None:
                draw_pose(frame, pr.keypoints)

    if mode in (MODE_OBJECTS, MODE_BOTH):
        results = model(frame)
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])  # Get class ID
                confidence = box.conf[0].item()  # Confidence score

                if confidence > 0.5:
                    label = model.names[class_id]
                    confidence_percent = confidence * 100
                    clabel = f"{label} {confidence_percent:.2f}%"
                    detected_objects.append(clabel)

                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, clabel, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, detected_objects


def main():
    cap = cv2.VideoCapture(0)  # Open webcam
    mode = MODE_BOTH

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Mirror the frame

        frame, detected_objects = detect_objects(frame, mode)
        cv2.putText(frame, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("AI Vision", frame)

        if detected_objects and mode in (MODE_OBJECTS, MODE_BOTH):
            print("Detected objects:", detected_objects)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            mode = MODE_NONE
            print("Switched to no detection mode")
        elif key == ord('o'):
            mode = MODE_OBJECTS
            print("Switched to object detection mode")
        elif key == ord('p'):
            mode = MODE_POSE
            print("Switched to pose detection mode")
        elif key == ord('b'):
            mode = MODE_BOTH
            print("Switched to both object and pose mode")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()