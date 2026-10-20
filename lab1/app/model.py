import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from app.config import ALLOWED_LABELS, ARTIFACT_DIR, DATASET_NAME, MODEL_NAME, MODEL_PATH
from app.dataset import to_mel_image


class SpeechCNN(nn.Module):
    def __init__(self, num_classes: int = len(ALLOWED_LABELS)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def save_model(model: nn.Module, labels: list[str], metadata: dict | None = None):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    payload = {
        "model_name": MODEL_NAME,
        "dataset": DATASET_NAME,
        "labels": labels,
        "architecture": str(model),
        "metadata": metadata or {},
    }
    with open(ARTIFACT_DIR / "model_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_model(model_path: str | Path = MODEL_PATH):
    model_path = Path(model_path)
    with open(ARTIFACT_DIR / "model_metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    labels = metadata.get("labels", ALLOWED_LABELS)
    model = SpeechCNN(num_classes=len(labels))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, labels, metadata


def preprocess_bytes_to_tensor(audio_bytes: bytes):
    import io
    import soundfile as sf

    buffer = io.BytesIO(audio_bytes)
    audio_array, samplerate = sf.read(buffer, dtype="float32", always_2d=False)
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)
    waveform = torch.tensor(audio_array, dtype=torch.float32)
    waveform = waveform.unsqueeze(0)
    feature = to_mel_image(waveform, samplerate)
    return feature


def predict_label(model, feature_tensor, labels):
    with torch.no_grad():
        logits = model(feature_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_idx].item())
    return labels[pred_idx], confidence
