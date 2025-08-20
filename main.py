# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 17:39:59 2025

@author: rutik
"""

#!/usr/bin/env python3
"""
Optimized Traffic Lane Vehicle Counter
-------------------------------------
- Automatic lane division (splits video width into 3 lanes)
- YOLOv8n (COCO) for vehicle detection
- ByteTrack for tracking
- Supports GPU if available
- Optional frame resizing to speed up processing
"""

import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import csv
import torch

# ---------------- Configuration ----------------


VIDEO_FILE = '/content/input.mp4'  # path in Colab
OUTPUT_DIR = Path('/content/traffic_output')  # Convert string to Path
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)  # Now this works

OUTPUT_VIDEO = OUTPUT_DIR / "traffic_out.mp4"
OUTPUT_CSV = OUTPUT_DIR / "counts.csv"


VEHICLE_CLASSES = {"car", "motorbike", "bus", "truck"}
CONF_THRESH = 0.25  # Detection confidence
NUM_LANES = 3
RESIZE_WIDTH = 640  # Optional: resize frames to speed up
RESIZE_HEIGHT = 360

# ---------------- Helper Functions ----------------
def point_in_polygon(pt, polygon):
    """Check if a point (x, y) is inside a polygon"""
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), pt, False) >= 0

def auto_define_lanes(frame_width, frame_height, num_lanes=3):
    """Split frame width into equal polygons for lanes"""
    lane_width = frame_width // num_lanes
    lanes = []
    for i in range(num_lanes):
        lanes.append([
            (i*lane_width, 0),
            ((i+1)*lane_width, 0),
            ((i+1)*lane_width, frame_height),
            (i*lane_width, frame_height)
        ])
    return lanes

# ---------------- Main Function ----------------
def main():
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_FILE}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Optional resizing
    scale_x = RESIZE_WIDTH / width
    scale_y = RESIZE_HEIGHT / height
    width, height = RESIZE_WIDTH, RESIZE_HEIGHT

    lanes = auto_define_lanes(width, height, NUM_LANES)
    print(f"Detected lane polygons: {lanes}")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))

    # CSV file
    csv_file = open(OUTPUT_CSV, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["vehicle_id", "lane", "frame", "timestamp_sec"])

    # Load YOLOv8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] Loading YOLOv8n (COCO) on {device}...")
    model = YOLO("yolov8n.pt").to(device)

    # Tracking setup
    counted_ids_per_lane = [set() for _ in range(NUM_LANES)]
    per_lane_counts = [0 for _ in range(NUM_LANES)]

    names = model.model.names if hasattr(model, "model") else model.names
    vehicle_class_ids = {i for i, n in names.items() if n in VEHICLE_CLASSES}

    frame_idx = -1
    start_time = time.time()

    results_gen = model.track(
        source=str(VIDEO_FILE),
        stream=True,
        conf=CONF_THRESH,
        iou=0.5,
        verbose=False,
        tracker="bytetrack.yaml"
    )

    for res in results_gen:
        frame_idx += 1
        frame = res.orig_img
        if frame is None:
            continue

        # Resize frame
        frame = cv2.resize(frame, (width, height))

        # Draw lane polygons and counts
        for li, poly in enumerate(lanes):
            cv2.polylines(frame, [np.array(poly, dtype=np.int32)], True, (0, 255, 255), 2)
            cv2.putText(frame, f"Lane {li+1}: {per_lane_counts[li]}", (10, 40*(li+1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Process detections
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.array([], dtype=int)
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

            for i in range(len(xyxy)):
                if cls.size == 0 or cls[i] not in vehicle_class_ids:
                    continue
                x1, y1, x2, y2 = xyxy[i].tolist()
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                tid = int(ids[i]) if ids is not None else -1

                lane_idx = None
                for li, poly in enumerate(lanes):
                    if point_in_polygon((cx, cy), poly):
                        lane_idx = li
                        break

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                label = f"id:{tid}" if tid != -1 else "id:?"
                cv2.putText(frame, label, (x1, max(20, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                if lane_idx is not None and tid != -1:
                    if tid not in counted_ids_per_lane[lane_idx]:
                        counted_ids_per_lane[lane_idx].add(tid)
                        per_lane_counts[lane_idx] += 1
                        csv_writer.writerow([tid, lane_idx+1, frame_idx, f"{frame_idx/fps:.2f}"])

        # Write frame
        writer.write(frame)

    # Cleanup
    cap.release()
    writer.release()
    csv_file.close()

    print("\n=== SUMMARY ===")
    for i, c in enumerate(per_lane_counts, start=1):
        print(f"Lane {i}: {c}")
    print(f"Total vehicles: {sum(per_lane_counts)}")
    print(f"Annotated video saved: {OUTPUT_VIDEO}")
    print(f"Counts CSV saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
