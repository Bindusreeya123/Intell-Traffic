import cv2
from ultralytics import YOLO
from collections import defaultdict, deque

model = YOLO("yolov8n.pt")

# Store last 30 positions per vehicle
track_history = defaultdict(lambda: deque(maxlen=30))

def process_frame(frame, frame_no, fps):
    annotated = frame.copy()
    detections = []

    results = model.track(
        frame,
        persist=True,
        conf=0.2,            # detect more vehicles
        iou=0.7,             # better association
        classes=[2, 3, 5, 7],  # car, bike, bus, truck
        tracker="bytetrack.yaml"
    )

    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()

        for (x, y, w, h), tid in zip(boxes, ids):
            cx, cy = int(x), int(y)

            # Update track history
            track_history[tid].append((cx, cy, frame_no))

            # Bounding box
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"ID {tid}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            detections.append({
                "id": tid,
                "center": (cx, cy)
            })

    return annotated, detections
