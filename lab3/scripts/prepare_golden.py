import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def find_dataset_directory(root: Path) -> Path:
    candidates = (
        root / "SpeechCommands" / "speech_commands_v0.02",
        root / "speech_commands_v0.02",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Speech Commands v0.02 dataset was not found under {root}")


def prepare(data_root: Path, manifest_path: Path, output: Path) -> None:
    dataset = find_dataset_directory(data_root)
    test_list_path = dataset / "testing_list.txt"
    if not test_list_path.is_file():
        raise FileNotFoundError(f"Official testing split is missing: {test_list_path}")
    test_files = set(test_list_path.read_text(encoding="utf-8").splitlines())
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest:
        raise ValueError(f"Golden sample manifest is empty: {manifest_path}")

    output.mkdir(parents=True, exist_ok=True)
    for filename, sample_info in manifest.items():
        if not isinstance(sample_info, dict):
            raise TypeError(f"Invalid golden sample metadata for {filename}")
        filename_path = Path(filename)
        if filename_path.name != filename or filename_path.suffix.lower() != ".wav":
            raise ValueError(f"Golden sample filename must be a plain WAV filename: {filename}")
        source = sample_info.get("source")
        label = sample_info.get("label")
        if not isinstance(source, str) or not isinstance(label, str):
            raise TypeError(f"Missing source or label for golden sample {filename}")
        source_path = Path(source)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError(f"Golden sample source must be a safe relative path: {source}")
        if source_path.as_posix() not in test_files:
            raise ValueError(f"Golden sample is not in the official test split: {source}")
        if source_path.parts[0] != label:
            raise ValueError(f"Golden sample label does not match its directory: {source}")
        source_file = dataset / source_path
        if not source_file.is_file():
            raise FileNotFoundError(f"Golden sample does not exist: {source_file}")
        shutil.copyfile(source_file, output / filename)

    shutil.copyfile(manifest_path, output / "expected.json")
    print(f"Prepared {len(manifest)} held-out golden samples in {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy held-out WAVs for integration verification")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("lab3/golden/expected.json"))
    parser.add_argument("--output", type=Path, default=Path("lab3/artifacts/golden"))
    args = parser.parse_args()
    prepare(args.data_root, args.manifest, args.output)


if __name__ == "__main__":
    main()
