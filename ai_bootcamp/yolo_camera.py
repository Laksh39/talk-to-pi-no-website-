
#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2
import time
import requests

MODEL_PATH  = "yolov8n.pt"
CONF_THRESH = 0.4
DISPLAY_W   = 640
DISPLAY_H   = 480

REPLIT_URL = "https://remote-pi-console.replit.app/api/message"
ALERT_COOLDOWN = 10  # seconds between alerts
PERSON_THRESHOLD = 3  # seconds person must be visible before alert

last_alert_time = 0
person_first_seen = None  # when we first spotted a person this session

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
    cv2.putText(frame, text, (x1 + 4, badge_y + th + 2), font, 0.55, (0, 0, 0), 1)

def send_alert(message):
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time > ALERT_COOLDOWN:
        try:
            requests.post(REPLIT_URL, json={"message": message}, timeout=3)
            print(f"✅ Alert sent: {message}")
            last_alert_time = current_time
        except Exception as e:
            print(f"❌ Failed to send alert: {e}")

def main():
    global person_first_seen

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

        person_detected = False

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = names.get(cls_id, str(cls_id))
            colour = get_colour(cls_id)

            draw_box(frame, x1, y1, x2, y2, label, conf, colour)

            if label == "person":
                person_detected = True

        now = time.time()

        if person_detected:
            if person_first_seen is None:
                person_first_seen = now
                print("👀 Person spotted, watching...")

            seconds_visible = now - person_first_seen

            # Show countdown on screen
            remaining = max(0, PERSON_THRESHOLD - seconds_visible)
            cv2.putText(frame, f"Person: {seconds_visible:.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if seconds_visible >= PERSON_THRESHOLD:
                send_alert("🚨 Person detected by Pi camera for 3+ seconds!")

        else:
            if person_first_seen is not None:
                print("👋 Person left frame.")
            person_first_seen = None  # reset timer when person leaves

        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("YOLOv8-nano  |  Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
