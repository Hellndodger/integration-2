import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the CI report and GitHub Job Summary")
    parser.add_argument("--metrics", type=Path, default=Path("lab3/artifacts/metrics.json"))
    parser.add_argument("--latency", type=Path, default=Path("lab3/artifacts/latency.json"))
    parser.add_argument(
        "--verification", type=Path, default=Path("lab3/artifacts/verification.json")
    )
    parser.add_argument(
        "--image-info", type=Path, default=Path("lab3/artifacts/image-info.json")
    )
    parser.add_argument("--output", type=Path, default=Path("lab3/artifacts/ci-report.md"))
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    metrics = load_json(args.metrics)
    latency = load_json(args.latency)
    verification = load_json(args.verification)
    image_info = load_json(args.image_info)
    labels = ", ".join(metrics["labels"])
    report = f"""# Лабораторна робота №3 — CI/CD для аудіокласифікатора

## Результат запуску

| Показник | Значення |
|---|---|
| Статус quality gate | {verification["status"]} |
| Commit | `{metrics["git_commit"]}` |
| Версія моделі | `{metrics["model_version"]}` |
| Датасет / класи | {metrics["dataset"]} / {labels} |
| Архітектура | {metrics["architecture"]} |
| Епохи / seed | {metrics["epochs"]} / {metrics["seed"]} |
| Test accuracy | {metrics["accuracy"]}% |
| Вага моделі | {metrics["model_size_bytes"]} B ({metrics["model_size_bytes"] / 1024:.2f} KiB) |
| Training time | {metrics["training_seconds"]} s |
| API latency, server mean | {latency["server"]["mean_ms"]} ms |
| API latency, client p50 / p95 | {latency["client_end_to_end"]["p50_ms"]} / {latency["client_end_to_end"]["p95_ms"]} ms |
| Runtime image | `{args.image}` |
| Image size (`docker images`) | {image_info["size_display"]} |
| Published to GHCR | {str(image_info["published"]).lower()} |

## Golden-sample перевірка

| Файл | Очікуваний клас | Прогноз | Confidence | API latency |
|---|---|---|---:|---:|
"""
    report += "\n".join(
        f'| `{item["file"]}` | {item["expected"]} | {item["predicted"]} | '
        f'{item["confidence"]} | {item["latency_ms"]} ms |'
        for item in verification["golden_samples"]
    )
    report += "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
