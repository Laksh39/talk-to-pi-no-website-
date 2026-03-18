#!/usr/bin/env python3
"""
YOLOv8-nano Object Detection for Raspberry Pi 5
Uses OpenCV for display and Ultralytics for inference.

Install dependencies:
    pip install ultralytics opencv-python-headless --break-system-packages

On first run, yolov8n.pt will be downloaded automatically (~6MB).
"""

import cv2
from ultralytics import YOLO
import time

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "yolov8n.pt"   # downloaded automatically on first run
CAMERA_INDEX = 0              # try 1 if 0 doesn't work
CONF_THRESH  = 0.4            # minimum confidence to show a box (0–1)
DISPLAY_W    = 640
DISPLAY_H    = 480
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette — one colour per class index (cycles if > 80 classes)
PALETTE = [
    (255,  56,  56), (255, 157,  51), (255, 225,  51), (199, 255,  51),
    ( 51, 255,  87), ( 51, 255, 198), ( 51, 198, 255), ( 51,  69, 255),
    (139,  51, 255), (255,  51, 225),
]

def get_colour(class_id: int):
    return PALETTE[class_id % len(PALETTE)]


def draw_box(frame, x1, y1, x2, y2, label: str, conf: float, colour):
    """Draw a bounding box with a filled label badge."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    text    = f"{label}  {conf:.0%}"
    font    = cv2.FONT_HERSHEY_SIMPLEX
    scale   = 0.55
    thick   = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)

    # Badge background
    badge_y = max(y1 - th - baseline - 6, 0)
    cv2.rectangle(frame,
                  (x1, badge_y),
                  (x1 + tw + 8, badge_y + th + baseline + 6),
                  colour, -1)

    # Label text (black for readability)
    cv2.putText(frame, text,
                (x1 + 4, badge_y + th + 2),
                font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def main():
    print("Loading YOLOv8-nano model …")
    model = YOLO(MODEL_PATH)
    names = model.names          # dict: {0: 'person', 1: 'bicycle', …}

    print(f"Opening camera (index {CAMERA_INDEX}) …")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {CAMERA_INDEX}. "
            "Try changing CAMERA_INDEX to 1, or check your connection."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Running — press Q to quit.")
    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame — exiting.")
            break

        # ── Run inference ──────────────────────────────────────────────────
        results = model(frame, conf=CONF_THRESH, verbose=False)[0]

        # ── Draw detections ────────────────────────────────────────────────
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            label   = names.get(cls_id, str(cls_id))
            colour  = get_colour(cls_id)
            draw_box(frame, x1, y1, x2, y2, label, conf, colour)

        # ── FPS counter ────────────────────────────────────────────────────
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("YOLOv8-nano  |  Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
