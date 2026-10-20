import json
import os
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import ALLOWED_LABELS, DATASET_NAME, MODEL_PATH
from app.model import load_model, predict_label, preprocess_bytes_to_tensor

app = FastAPI(title="Speech Commands API", version="1.0.0")
app.state.model = None
app.state.labels = ALLOWED_LABELS
app.state.metadata = {}


@app.on_event("startup")
def startup_event():
    if MODEL_PATH.exists():
        model, labels, metadata = load_model(MODEL_PATH)
        app.state.model = model
        app.state.labels = labels
        app.state.metadata = metadata
    else:
        app.state.model = None
        app.state.metadata = {"status": "not_trained"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.get("/model-info")
def model_info():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifacts are not available. Please run training first.")

    size_bytes = MODEL_PATH.stat().st_size
    return {
        "dataset": DATASET_NAME,
        "model_name": app.state.metadata.get("model_name", "SpeechCNN"),
        "architecture": app.state.metadata.get("architecture", "SpeechCNN"),
        "labels": app.state.labels,
        "model_size_mb": round(size_bytes / (1024 * 1024), 4),
        "test_accuracy": app.state.metadata.get("metadata", {}).get("test_accuracy", None),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Run training before prediction.")

    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        feature = preprocess_bytes_to_tensor(contents)
    except Exception as exc:  # pragma: no cover - defensive handling
        raise HTTPException(status_code=400, detail=f"Unable to decode audio: {str(exc)}") from exc

    start = time.perf_counter()
    with torch.no_grad():
        logits = app.state.model(feature)
        probabilities = torch.softmax(logits, dim=1)
        prediction_index = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, prediction_index].item())
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "label": app.state.labels[prediction_index],
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 2),
        "dataset": DATASET_NAME,
        "model_name": app.state.metadata.get("model_name", "SpeechCNN"),
    }


if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=False)
