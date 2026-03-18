import requests
import time

REPLIT_URL = "https://remote-pi-console.replit.app/api/message"

last_message = None

def check_for_message():
    global last_message
    try:
        response = requests.get(REPLIT_URL)
        if response.status_code == 200:
            data = response.json()
            message = data.get("message", "")
            if message and message != last_message:
                print(f"📨 New message: {message}")
                last_message = message
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect. Is your Replit app running?")
    except Exception as e:
        print(f"❌ Error: {e}")

print("🟢 Pi Receiver started. Waiting for new messages...")

while True:
    check_for_message()
    time.sleep(5)

