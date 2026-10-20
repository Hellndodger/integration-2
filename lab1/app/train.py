import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from app.config import ALLOWED_LABELS, ARTIFACT_DIR, MODEL_PATH, SPEECHCOMMANDS_ROOT
from app.dataset import create_loaders
from app.model import SpeechCNN, save_model


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            predictions = torch.argmax(logits, dim=1)
            total += targets.size(0)
            correct += (predictions == targets).sum().item()
    return correct / total if total > 0 else 0.0


def train_model():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(os.environ.get("SPEECHCOMMANDS_ROOT", str(SPEECHCOMMANDS_ROOT)))
    data_root.mkdir(parents=True, exist_ok=True)

    train_loader, validation_loader, test_loader, label_to_idx = create_loaders(data_root, limit_per_class=50)

    model = SpeechCNN(num_classes=len(ALLOWED_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_accuracy = 0.0
    for epoch in range(4):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_accuracy = evaluate(model, validation_loader, device)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
        print(f"Epoch {epoch + 1}: validation_accuracy={val_accuracy:.4f}, loss={epoch_loss / len(train_loader):.4f}")

    test_accuracy = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    model_metadata = {
        "dataset": "Google Speech Commands",
        "subset": "yes,no,up,down",
        "epochs": 4,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "test_accuracy": test_accuracy,
        "label_to_idx": label_to_idx,
    }

    save_model(model, ALLOWED_LABELS, metadata=model_metadata)
    return model, test_accuracy


if __name__ == "__main__":
    start = time.perf_counter()
    train_model()
    elapsed = (time.perf_counter() - start) * 1000
    print(f"Training completed in {elapsed:.2f} ms")
