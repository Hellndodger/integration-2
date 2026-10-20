import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from src.data import LABELS, create_loader
from src.model import SpeechCNN


def evaluate(model: SpeechCNN, loader: torch.utils.data.DataLoader) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.inference_mode():
        for features, targets in loader:
            predictions = model(features).argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            total += int(targets.size(0))
    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset split")
    return correct / total


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))

    data_root = Path(args.data_root)
    train_loader = create_loader(data_root, "training", args.limit_per_class, args.seed)
    validation_loader = create_loader(data_root, "validation", args.limit_per_class, args.seed)
    test_loader = create_loader(data_root, "testing", args.limit_per_class, args.seed)

    model = SpeechCNN(num_classes=len(LABELS))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    started_at = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        batches = 0
        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            batches += 1

        validation_accuracy = evaluate(model, validation_loader)
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"loss={loss_sum / batches:.4f} "
            f"validation_accuracy={validation_accuracy * 100:.2f}%"
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training completed without producing a model checkpoint")
    model.load_state_dict(best_state)
    test_accuracy = evaluate(model, test_loader)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    metrics = {
        "dataset": "Google Speech Commands",
        "modality": "audio",
        "labels": LABELS,
        "architecture": "SpeechCNN",
        "model_version": args.model_version,
        "seed": args.seed,
        "epochs": args.epochs,
        "limit_per_class": args.limit_per_class,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "test_accuracy": test_accuracy,
        "accuracy": round(test_accuracy * 100, 2),
        "model_size_bytes": model_path.stat().st_size,
        "training_seconds": round(time.perf_counter() - started_at, 2),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the Speech Commands CNN")
    parser.add_argument("--data-root", default="/tmp/speechcommands")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--limit-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--model-version", default="1.0.0")
    args = parser.parse_args()
    if args.epochs < 1 or args.limit_per_class < 1 or args.threads < 1:
        parser.error("--epochs, --limit-per-class, and --threads must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
