import argparse
import json
from pathlib import Path

import torch

from src.data import LABELS, create_loader
from src.model import SpeechCNN
from src.train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split")
    parser.add_argument("--data-root", default="/tmp/speechcommands")
    parser.add_argument("--model-path", default="artifacts/model.pt")
    parser.add_argument("--limit-per-class", type=int, default=50)
    args = parser.parse_args()

    loader = create_loader(
        Path(args.data_root), "testing", args.limit_per_class, seed=42
    )
    model = SpeechCNN(num_classes=len(LABELS))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True))
    accuracy = evaluate(model, loader)
    result = {
        "test_accuracy": accuracy,
        "accuracy_percent": round(accuracy * 100, 2),
        "test_samples": len(loader.dataset),
        "model_size_kb": round(Path(args.model_path).stat().st_size / 1024, 2),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
