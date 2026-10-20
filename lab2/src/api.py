import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.data import LABELS, to_mel_image
from src.model import SpeechCNN

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/artifacts/model.pt"))
METRICS_PATH = Path(
    os.environ.get("METRICS_PATH", str(MODEL_PATH.with_name("metrics.json")))
)
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")
torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))


def load_artifacts() -> tuple[SpeechCNN, dict[str, Any]]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {MODEL_PATH}")
    if not METRICS_PATH.is_file():
        raise FileNotFoundError(f"Model metrics do not exist: {METRICS_PATH}")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    labels = metrics.get("labels", LABELS)
    model = SpeechCNN(num_classes=len(labels))
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.metrics = load_artifacts()
    app.state.labels = app.state.metrics.get("labels", LABELS)
    yield


app = FastAPI(title="Speech Commands API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    metrics = app.state.metrics
    return {
        "dataset": metrics["dataset"],
        "architecture": metrics["architecture"],
        "labels": app.state.labels,
        "model_size_kb": round(MODEL_PATH.stat().st_size / 1024, 2),
        "accuracy": metrics["accuracy"],
        "model_version": os.environ.get(
            "MODEL_VERSION", metrics.get("model_version", MODEL_VERSION)
        ),
    }


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
    if not file.filename or Path(file.filename).suffix.lower() != ".wav":
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    started_at = time.perf_counter()
    try:
        audio, sample_rate = sf.read(
            io.BytesIO(contents), dtype="float32", always_2d=False
        )
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        feature = to_mel_image(torch.as_tensor(audio), sample_rate)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to decode audio: {exc}") from exc

    try:
        with torch.inference_mode():
            logits = app.state.model(feature)
            probabilities = torch.softmax(logits, dim=1)[0]
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Unable to process audio: {exc}") from exc

    prediction_index = int(probabilities.argmax().item())
    return {
        "label": app.state.labels[prediction_index],
        "confidence": round(float(probabilities[prediction_index].item()), 4),
        "probabilities": {
            label: round(float(probabilities[index].item()), 4)
            for index, label in enumerate(app.state.labels)
        },
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "model_version": os.environ.get(
            "MODEL_VERSION", app.state.metrics.get("model_version", MODEL_VERSION)
        ),
    }


if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
    )
