import random
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

LABELS = ["yes", "no", "up", "down"]
SAMPLE_RATE = 16000


def to_mel_image(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)

    waveform = waveform.squeeze(0)
    if waveform.numel() < SAMPLE_RATE:
        waveform = torch.nn.functional.pad(waveform, (0, SAMPLE_RATE - waveform.numel()))
    else:
        waveform = waveform[:SAMPLE_RATE]
    waveform = waveform.unsqueeze(0)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=400,
        hop_length=160,
        n_mels=64,
        power=2.0,
    )(waveform)
    spectrogram = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)(mel)
    return spectrogram[:, :, :128].unsqueeze(1)


def _dataset_directory(root: Path) -> Path:
    candidates = (
        root / "SpeechCommands" / "speech_commands_v0.02",
        root / "speech_commands_v0.02",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Google Speech Commands data was not found under {root}. "
        "Run training with download enabled or provide SPEECHCOMMANDS_ROOT."
    )


def ensure_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    torchaudio.datasets.SPEECHCOMMANDS(root=str(root), download=True)
    return _dataset_directory(root)


class SpeechFiles(Dataset):
    def __init__(self, files: list[tuple[Path, int]]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.files[index]
        waveform, sample_rate = torchaudio.load(str(path))
        return to_mel_image(waveform, sample_rate).squeeze(0), target


def _files_for_split(
    directory: Path,
    split: str,
    limit_per_class: int | None,
    seed: int,
) -> list[tuple[Path, int]]:
    validation_file = directory / "validation_list.txt"
    testing_file = directory / "testing_list.txt"
    if not validation_file.is_file() or not testing_file.is_file():
        raise FileNotFoundError(f"Official train/validation/test lists are missing in {directory}")

    validation = set(validation_file.read_text(encoding="utf-8").splitlines())
    testing = set(testing_file.read_text(encoding="utf-8").splitlines())
    selected: list[tuple[Path, int]] = []
    counts: defaultdict[str, int] = defaultdict(int)
    label_to_index = {label: index for index, label in enumerate(LABELS)}

    candidates = [
        path
        for label in LABELS
        for path in sorted((directory / label).glob("*.wav"))
    ]
    random.Random(seed).shuffle(candidates)

    for path in candidates:
        relative = path.relative_to(directory).as_posix()
        if split == "validation":
            included = relative in validation
        elif split == "testing":
            included = relative in testing
        elif split == "training":
            included = relative not in validation and relative not in testing
        else:
            raise ValueError(f"Unknown dataset split: {split}")

        label = path.parent.name
        if not included or (limit_per_class is not None and counts[label] >= limit_per_class):
            continue
        selected.append((path, label_to_index[label]))
        counts[label] += 1

    if not selected:
        raise ValueError(f"No audio files found for split '{split}' in {directory}")
    return selected


def create_loader(
    root: Path,
    split: str,
    limit_per_class: int | None = 50,
    seed: int = 42,
    batch_size: int = 32,
) -> DataLoader:
    directory = ensure_dataset(root)
    files = _files_for_split(directory, split, limit_per_class, seed)
    return DataLoader(
        SpeechFiles(files),
        batch_size=batch_size,
        shuffle=(split == "training"),
        num_workers=0,
    )
