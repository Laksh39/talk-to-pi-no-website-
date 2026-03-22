import speech_recognition as sr
import requests
import subprocess
import re

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL        = "http://localhost:11434/api/generate"
OLLAMA_MODEL      = "gemma3:1b"
ENERGY_THRESHOLD  = 300
PAUSE_THRESHOLD   = 0.8
PHRASE_TIME_LIMIT = 10
AMBIENT_DURATION  = 1

# TTS settings — change voice to "en-us" if you prefer American accent
TTS_VOICE  = "en-gb"   # british english — sounds great on Pi!
TTS_SPEED  = 150       # words per minute (120 = slower, 180 = faster)
TTS_PITCH  = 50        # 0-99 (50 = default)
# ─────────────────────────────────────────────────────────────────────────────

recognizer = sr.Recognizer()
recognizer.energy_threshold         = ENERGY_THRESHOLD
recognizer.pause_threshold          = PAUSE_THRESHOLD
recognizer.dynamic_energy_threshold = True


def clean_reply(text: str) -> str:
    """Strip markdown formatting so the reply reads naturally."""
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"`{1,3}(.+?)`{1,3}", r"\1", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    text = re.sub(r"\n{2,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def speak(text: str):
    """Say text aloud using espeak directly — no pyttsx3 needed."""
    print(f"🔊 {text}")
    subprocess.run(
        ["espeak", "-v", TTS_VOICE, "-s", str(TTS_SPEED), "-p", str(TTS_PITCH), text],
        stderr=subprocess.DEVNULL
    )


def ask_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return a short, spoken-word response."""
    wrapped = (
        "You are a voice assistant. Answer in 1-2 sentences maximum. "
        "Be direct and conversational. No lists, no markdown, no bullet points. "
        f"Question: {prompt}"
    )
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": wrapped, "stream": False},
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        return "I cannot reach Ollama. Please make sure it is running."
    except requests.exceptions.Timeout:
        return "Ollama took too long to respond. Please try again."
    except Exception as e:
        return f"Something went wrong: {e}"


def listen_once(mic: sr.Microphone) -> str | None:
    """Listen for one phrase and return the transcribed text, or None."""
    try:
        print("Listening …", end="\r")
        audio = recognizer.listen(mic, phrase_time_limit=PHRASE_TIME_LIMIT)
        print("Recognising …", end="\r")
        return recognizer.recognize_google(audio).lower()

    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        print("(Could not understand)        ")
        return None
    except sr.RequestError as e:
        print(f"[Google API error: {e}]")
        return None


def main():
    print("🤖  Pi Voice Assistant  —  Gemma 3:1b + espeak")
    print("─" * 50)
    print(f'Voice : {TTS_VOICE} at {TTS_SPEED} wpm')
    print("Listening to everything — no wake word needed.")
    print("Press Ctrl+C to quit.\n")

    speak("Pi assistant is ready. Just speak and I will respond.")

    with sr.Microphone() as mic:
        print(f"Calibrating for background noise ({AMBIENT_DURATION}s) …")
        recognizer.adjust_for_ambient_noise(mic, duration=AMBIENT_DURATION)
        print(f"Ready! (energy threshold: {recognizer.energy_threshold:.0f})\n")

        while True:
            try:
                # ── Listen for anything ───────────────────────────────────
                text = listen_once(mic)
                if not text:
                    continue

                print(f"\n💬 You : {text}")

                # ── Send straight to Ollama ───────────────────────────────
                print("🧠 Thinking …")
                reply = ask_ollama(text)

                # ── Clean, print and speak the reply ─────────────────────
                reply = clean_reply(reply)
                print(f"🤖 Pi  : {reply}\n")
                speak(reply)
                print("─" * 50)

            except KeyboardInterrupt:
                print("\n\nStopped. Goodbye!")
                speak("Goodbye!")
                break


if __name__ == "__main__":
    main()
