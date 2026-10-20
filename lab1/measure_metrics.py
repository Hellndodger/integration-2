import io
import os
import subprocess
import sys
import time

import numpy as np
import requests
import soundfile as sf

ROOT = r"d:\2026-4-1\integration_project\lab1"


def generate_wav(label: str):
    sample_rate = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    freqs = {"yes": 440.0, "no": 550.0, "up": 660.0, "down": 220.0}
    wave = np.sin(2 * np.pi * freqs[label] * t).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, wave, sample_rate, format="WAV")
    return buffer.getvalue()


def main():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8002"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(30):
            try:
                r = requests.get("http://127.0.0.1:8002/health", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)

        latencies = []
        for label in ["yes", "no", "up", "down"]:
            payload = requests.post(
                "http://127.0.0.1:8002/predict",
                files={"file": (f"{label}.wav", generate_wav(label), "audio/wav")},
                timeout=30,
            )
            item = payload.json()
            latencies.append(item["latency_ms"])
            print(label, item)

        model_path = os.path.join(ROOT, "artifacts", "speech_model.pth")
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print("MODEL_SIZE_MB", round(size_mb, 4))
        print("AVG_LATENCY_MS", round(sum(latencies) / len(latencies), 2))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
