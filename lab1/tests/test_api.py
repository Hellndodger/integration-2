import io
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8001"
ALLOWED_LABELS = ["yes", "no", "up", "down"]


def log_test(name: str, description: str):
    print(f"\n=== {name}: {description} ===")


def make_wav_bytes(label: str, sample_rate: int = 16000, duration: float = 0.5):
    freqs = {"yes": 440.0, "no": 550.0, "up": 660.0, "down": 220.0}
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freqs[label] * t).astype(np.float32)
    wave = wave + 0.02 * np.sin(2 * np.pi * (freqs[label] * 2) * t).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, wave, sample_rate, format="WAV")
    return buffer.getvalue()


@pytest.fixture(scope="session", autouse=True)
def start_api_server():
    print("\n=== starting test environment: training model and launching API ===")
    subprocess.run([sys.executable, "-m", "app.train"], cwd=str(ROOT), check=True, capture_output=True, text=True)

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8001"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    else:
        stdout, stderr = proc.communicate(timeout=10)
        raise RuntimeError(f"API server failed to start. stdout={stdout}; stderr={stderr}")

    yield

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_health_endpoint():
    log_test("test_health_endpoint", "checking /health returns OK and model status")
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_loaded" in payload


def test_model_info_endpoint():
    log_test("test_model_info_endpoint", "checking /model-info returns dataset and model metadata")
    response = requests.get(f"{BASE_URL}/model-info", timeout=10)
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"]
    assert payload["labels"] == ALLOWED_LABELS
    assert payload["model_size_mb"] > 0


def test_predict_yes_audio():
    log_test("test_predict_yes_audio", "sending synthetic yes.wav and validating a valid prediction")
    response = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("yes.wav", make_wav_bytes("yes"), "audio/wav")},
        timeout=20,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] in ALLOWED_LABELS
    assert 0.0 <= payload["confidence"] <= 1.0


def test_predict_no_audio():
    log_test("test_predict_no_audio", "sending synthetic no.wav and validating a valid prediction")
    response = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("no.wav", make_wav_bytes("no"), "audio/wav")},
        timeout=20,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] in ALLOWED_LABELS


def test_predict_up_audio():
    log_test("test_predict_up_audio", "sending synthetic up.wav and validating a valid prediction")
    response = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("up.wav", make_wav_bytes("up"), "audio/wav")},
        timeout=20,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] in ALLOWED_LABELS


def test_predict_down_audio():
    log_test("test_predict_down_audio", "sending synthetic down.wav and validating a valid prediction")
    response = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("down.wav", make_wav_bytes("down"), "audio/wav")},
        timeout=20,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] in ALLOWED_LABELS


def test_predict_latency_is_positive():
    log_test("test_predict_latency_is_positive", "checking latency_ms is returned and greater than zero")
    response = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("latency.wav", make_wav_bytes("yes"), "audio/wav")},
        timeout=20,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["latency_ms"] > 0


def test_invalid_extension_is_rejected():
    log_test("test_invalid_extension_is_rejected", "checking that non-WAV files are rejected with 400")
    response = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("invalid.txt", b"not audio", "text/plain")},
        timeout=20,
    )
    assert response.status_code == 400
    assert "Only .wav files are supported" in response.json()["detail"]


def test_empty_upload_is_rejected():
    log_test("test_empty_upload_is_rejected", "checking that empty files are rejected with a clear validation error")
    response = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("empty.wav", b"", "audio/wav")},
        timeout=20,
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_prediction_response_has_valid_schema():
    log_test("test_prediction_response_has_valid_schema", "checking the JSON schema contains all required fields")
    response = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("schema.wav", make_wav_bytes("no"), "audio/wav")},
        timeout=20,
    )
    assert response.status_code == 200
    payload = response.json()
    for required_field in ("label", "confidence", "latency_ms", "dataset", "model_name"):
        assert required_field in payload


def test_model_metadata_file_exists_after_training():
    log_test("test_model_metadata_file_exists_after_training", "checking that trained artifacts exist on disk")
    assert (ROOT / "artifacts" / "model_metadata.json").exists()
    assert (ROOT / "artifacts" / "speech_model.pth").exists()
