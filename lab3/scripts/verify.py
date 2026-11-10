import argparse
import json
import math
from pathlib import Path
from typing import Any

import requests


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def verify(args: argparse.Namespace) -> dict[str, Any]:
    expected = read_json(args.samples / "expected.json")
    metrics = read_json(args.metrics)
    latency = read_json(args.latency)

    health_response = requests.get(f"{args.url.rstrip('/')}/health", timeout=10)
    health_response.raise_for_status()
    health = health_response.json()
    if health.get("status") != "ok":
        raise ValueError(f"Unexpected health response: {health}")

    info_response = requests.get(f"{args.url.rstrip('/')}/model-info", timeout=10)
    info_response.raise_for_status()
    model_info = info_response.json()
    if model_info.get("labels") != metrics.get("labels"):
        raise ValueError("API labels do not match the trained metrics artifact")
    if model_info.get("git_commit") != metrics.get("git_commit"):
        raise ValueError("API git_commit does not match the trained metrics artifact")
    if float(model_info.get("accuracy", 0)) < args.min_accuracy:
        raise ValueError(
            f"Accuracy gate failed: {model_info.get('accuracy')}% "
            f"is below {args.min_accuracy}%"
        )

    predictions: list[dict[str, Any]] = []
    labels = set(metrics["labels"])
    if not expected:
        raise ValueError("No golden samples are configured")
    for filename, sample_info in expected.items():
        if not isinstance(sample_info, dict) or "label" not in sample_info:
            raise ValueError(f"Invalid golden sample metadata for {filename}")
        expected_label = sample_info["label"]
        if expected_label not in labels:
            raise ValueError(f"Unknown expected label for {filename}: {expected_label}")
        sample_path = args.samples / filename
        if not sample_path.is_file():
            raise FileNotFoundError(f"Golden sample does not exist: {sample_path}")
        with sample_path.open("rb") as sample:
            response = requests.post(
                f"{args.url.rstrip('/')}/predict",
                files={"file": (sample_path.name, sample, "audio/wav")},
                timeout=30,
            )
        response.raise_for_status()
        prediction = response.json()
        probabilities = prediction.get("probabilities")
        if (
            prediction.get("label") not in labels
            or not isinstance(probabilities, dict)
            or set(probabilities) != labels
            or not 0 <= float(prediction.get("confidence", -1)) <= 1
            or float(prediction.get("latency_ms", 0)) <= 0
        ):
            raise ValueError(f"Prediction response has an invalid schema: {prediction}")
        probability_sum = math.fsum(float(value) for value in probabilities.values())
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=0.02):
            raise ValueError(f"Class probabilities sum to {probability_sum}, not 1")
        if prediction["label"] != expected_label:
            raise ValueError(
                f"Golden check failed for {filename}: expected {expected_label}, "
                f"received {prediction['label']}"
            )
        predictions.append(
            {
                "file": filename,
                "expected": expected_label,
                "predicted": prediction["label"],
                "confidence": prediction["confidence"],
                "latency_ms": prediction["latency_ms"],
            }
        )

    p95_ms = float(latency["client_end_to_end"]["p95_ms"])
    if p95_ms > args.max_p95_ms:
        raise ValueError(
            f"Latency gate failed: client p95={p95_ms} ms exceeds "
            f"{args.max_p95_ms} ms"
        )

    result = {
        "status": "passed",
        "accuracy_percent": model_info["accuracy"],
        "minimum_accuracy_percent": args.min_accuracy,
        "latency_client_p95_ms": p95_ms,
        "maximum_latency_client_p95_ms": args.max_p95_ms,
        "golden_samples": predictions,
        "model_version": model_info["model_version"],
        "git_commit": model_info["git_commit"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the deployed ML API and quality gates")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--samples", type=Path, default=Path("lab3/artifacts/golden"))
    parser.add_argument("--metrics", type=Path, default=Path("lab3/artifacts/metrics.json"))
    parser.add_argument("--latency", type=Path, default=Path("lab3/artifacts/latency.json"))
    parser.add_argument("--output", type=Path, default=Path("lab3/artifacts/verification.json"))
    parser.add_argument("--min-accuracy", type=float, default=45.0)
    parser.add_argument("--max-p95-ms", type=float, default=5000.0)
    args = parser.parse_args()
    if args.min_accuracy < 0 or args.max_p95_ms <= 0:
        parser.error("--min-accuracy must be non-negative and --max-p95-ms must be positive")
    return args


if __name__ == "__main__":
    verify(parse_args())
