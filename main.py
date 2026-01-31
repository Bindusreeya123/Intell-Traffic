import cv2
import csv
import math
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

# =========================
# CONFIG
# =========================
VIDEO_PATH = "editedvideo.mp4"
OUTPUT_VIDEO = "outputs/finaloutputvideo.mp4"
CSV_PATH = "outputs/finalspeedlog.csv"

PIXELS_PER_METER = 15
SMOOTHING_FRAMES = 10
SPEED_LIMIT = 50

VIOLATION_DIR = "violations/overspeed"
os.makedirs("outputs", exist_ok=True)
os.makedirs(VIOLATION_DIR, exist_ok=True)

# =========================
# LOAD MODELS
# =========================
vehicle_model = YOLO("yolov8n.pt")

try:
    plate_model = YOLO("license_plate.pt")
    print("✅ License plate model loaded")
except:
    plate_model = None
    print("⚠️ License plate model not found")

# =========================
# VIDEO SETUP
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError("❌ Video not found")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

QUEUE_REGION = np.array([[0,0],[w,0],[w,h],[0,h]], np.int32)

# =========================
# CSV SETUP
# =========================
csv_file = open(CSV_PATH, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow([
    "Frame",
    "Queue_Count",
    "Queue_Length",
    "Queue_Density",
    "Total_Vehicle_Count"
])

# =========================
# STORAGE
# =========================
track_history = defaultdict(lambda: deque(maxlen=30))
speed_history = defaultdict(lambda: deque(maxlen=SMOOTHING_FRAMES))
vehicle_in_queue = defaultdict(bool)

all_vehicle_ids = set()
overspeed_saved = set()

frame_no = 0
print("🎥 Processing started...")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1
    annotated = frame.copy()
    queue_points = []

    results = vehicle_model.track(
        frame,
        persist=True,
        conf=0.25,
        iou=0.7,
        classes=[2, 3, 5, 7],
        tracker="bytetrack.yaml"
    )

    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for tid in ids:
            all_vehicle_ids.add(tid)

        for (x, y, bw, bh), tid in zip(boxes, ids):
            cx, cy = int(x), int(y)
            track_history[tid].append((cx, cy, frame_no))

            # ================= SPEED =================
            speed_kmh = 0.0
            if len(track_history[tid]) >= 2:
                x1p, y1p, f1 = track_history[tid][-2]
                x2p, y2p, f2 = track_history[tid][-1]

                pixel_dist = math.hypot(x2p - x1p, y2p - y1p)
                meters = pixel_dist / PIXELS_PER_METER
                time_sec = (f2 - f1) / fps

                if time_sec > 0:
                    speed_history[tid].append((meters / time_sec) * 3.6)

                speed_kmh = sum(speed_history[tid]) / len(speed_history[tid])

            overspeed = speed_kmh > SPEED_LIMIT

            # ================= QUEUE =================
            if speed_kmh < 10:
                vehicle_in_queue[tid] = True
                queue_points.append((cx, cy))
            else:
                vehicle_in_queue[tid] = False

            # ================= OVERSPEED SNAPSHOT + PLATE =================
            if overspeed and tid not in overspeed_saved:
                x1s = max(0, int(x - bw/2))
                y1s = max(0, int(y - bh/2))
                x2s = min(w, int(x + bw/2))
                y2s = min(h, int(y + bh/2))

                vehicle_crop = frame[y1s:y2s, x1s:x2s]

                if vehicle_crop.size > 0:
                    v_name = f"vehicle_{tid}_frame_{frame_no}.jpg"
                    cv2.imwrite(os.path.join(VIOLATION_DIR, v_name), vehicle_crop)

                    # -------- NUMBER PLATE DETECTION --------
                    if plate_model:
                        pres = plate_model(vehicle_crop, conf=0.4)
                        if pres and pres[0].boxes is not None:
                            for pb in pres[0].boxes.xyxy:
                                px1, py1, px2, py2 = map(int, pb)
                                plate_crop = vehicle_crop[py1:py2, px1:px2]
                                if plate_crop.size > 0:
                                    p_name = f"plate_{tid}_frame_{frame_no}.jpg"
                                    cv2.imwrite(os.path.join(VIOLATION_DIR, p_name), plate_crop)
                                break

                    overspeed_saved.add(tid)

            # ================= DRAW =================
            x1, y1 = int(x - bw/2), int(y - bh/2)
            x2, y2 = int(x + bw/2), int(y + bh/2)

            color = (0, 0, 255) if overspeed else (0, 255, 0)
            label = f"ID {tid} | {speed_kmh:.1f} km/h"
            if overspeed:
                label += " OVERSPEED"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ================= QUEUE METRICS =================
    queue_count = len(queue_points)
    queue_length = 0.0
    queue_density = 0.0

    if queue_count >= 2:
        queue_points.sort(key=lambda p: p[1])
        fx, fy = queue_points[0]
        lx, ly = queue_points[-1]
        queue_length = math.hypot(lx - fx, ly - fy) / PIXELS_PER_METER
        if queue_length > 0:
            queue_density = queue_count / queue_length

    total_vehicle_count = len(all_vehicle_ids)

    # ================= DISPLAY =================
    cv2.polylines(annotated, [QUEUE_REGION], True, (255,0,0), 2)

    cv2.putText(annotated, f"Total Vehicles: {total_vehicle_count}", (30,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

    cv2.putText(annotated, f"Queue Count: {queue_count}", (30,75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    cv2.putText(annotated, f"Queue Length: {queue_length:.2f} m",
                (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,165,0), 2)

    cv2.putText(annotated, f"Queue Density: {queue_density:.2f} veh/m",
                (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    out.write(annotated)

    # ================= CSV WRITE =================
    writer.writerow([
        frame_no,
        queue_count,
        round(queue_length, 2),
        round(queue_density, 2),
        total_vehicle_count
    ])

# =========================
# CLEANUP
# =========================
cap.release()
out.release()
csv_file.close()

print("✅ COMPLETED SUCCESSFULLY")
print("📁 Video:", OUTPUT_VIDEO)
print("📄 CSV:", CSV_PATH)
print("📸 Violation images saved in:", VIOLATION_DIR)