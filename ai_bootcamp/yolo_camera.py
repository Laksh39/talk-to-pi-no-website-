#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2
import time

MODEL_PATH  = "yolov8n.pt"
CONF_THRESH = 0.4
DISPLAY_W   = 640
DISPLAY_H   = 480

PALETTE = [
    (255,  56,  56), (255, 157,  51), (255, 225,  51), (199, 255,  51),
    ( 51, 255,  87), ( 51, 255, 198), ( 51, 198, 255), ( 51,  69, 255),
    (139,  51, 255), (255,  51, 225),
]

def get_colour(class_id):
    return PALETTE[class_id % len(PALETTE)]

def draw_box(frame, x1, y1, x2, y2, label, conf, colour):
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    text = f"{label}  {conf:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, 0.55, 1)
    badge_y = max(y1 - th - baseline - 6, 0)
    cv2.rectangle(frame, (x1, badge_y), (x1 + tw + 8, badge_y + th + baseline + 6), colour, -1)
    cv2.putText(frame, text, (x1 + 4, badge_y + th + 2), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

def main():
    print("Loading YOLOv8-nano model ...")
    model = YOLO(MODEL_PATH)
    names = model.names

    print("Starting Picamera2 ...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (DISPLAY_W, DISPLAY_H)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    print("Running — press Q to quit.")
    prev_time = time.time()

    while True:
        frame = picam2.capture_array()

        results = model(frame, conf=CONF_THRESH, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = names.get(cls_id, str(cls_id))
            colour = get_colour(cls_id)
            draw_box(frame, x1, y1, x2, y2, label, conf, colour)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("YOLOv8-nano  |  Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
