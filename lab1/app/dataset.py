from collections import defaultdict
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader

from app.config import ALLOWED_LABELS, SAMPLE_RATE, SPEECHCOMMANDS_ROOT


def normalize_dataset_root(root: Path | str) -> Path:
    root = Path(root)
    if root.name == "speechcommands" and (root / "SpeechCommands").exists():
        return root
    if root.name == "SpeechCommands":
        return root.parent
    if (root / "SpeechCommands").exists():
        return root
    return root


def _split_bucket(filename: str) -> int:
    return sum(ord(ch) for ch in filename) % 10


def load_subset(root: Path | str, subset: str, labels=None, limit_per_class: int | None = None):
    root = normalize_dataset_root(root)
    label_set = labels or ALLOWED_LABELS
    filtered = []
    counts = defaultdict(int)

    for label in label_set:
        label_dir = root / label
        if not label_dir.exists():
            continue
        for wav_file in sorted(label_dir.glob("*.wav")):
            bucket = _split_bucket(wav_file.name)
            in_subset = (
                (subset == "training" and bucket < 8)
                or (subset == "validation" and bucket == 8)
                or (subset == "testing" and bucket == 9)
            )
            if not in_subset:
                continue
            if limit_per_class is not None and counts[label] >= limit_per_class:
                continue
            waveform, sample_rate = torchaudio.load(str(wav_file))
            filtered.append((waveform, sample_rate, label))
            counts[label] += 1
    return filtered


def pad_or_trim(waveform: torch.Tensor, target_length: int = SAMPLE_RATE) -> torch.Tensor:
    waveform = waveform.squeeze(0)
    if waveform.numel() < target_length:
        pad = target_length - waveform.numel()
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:target_length]
    return waveform.reshape(1, -1)


def to_mel_image(waveform: torch.Tensor, sample_rate: int, device: str = "cpu") -> torch.Tensor:
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)

    waveform = pad_or_trim(waveform, SAMPLE_RATE)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=400,
        hop_length=160,
        n_mels=64,
        power=2.0,
    )(waveform.to(device))
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    spectrogram = amp_to_db(mel)
    spectrogram = spectrogram[:, :, :128]
    return spectrogram.unsqueeze(1)


def create_loaders(train_root: Path | str = SPEECHCOMMANDS_ROOT, limit_per_class: int = 50):
    train_data = load_subset(train_root, subset="training", labels=ALLOWED_LABELS, limit_per_class=limit_per_class)
    validation_data = load_subset(train_root, subset="validation", labels=ALLOWED_LABELS, limit_per_class=limit_per_class)
    test_data = load_subset(train_root, subset="testing", labels=ALLOWED_LABELS, limit_per_class=limit_per_class)

    label_to_idx = {label: idx for idx, label in enumerate(ALLOWED_LABELS)}

    def collate(batch):
        features, targets = [], []
        for waveform, sample_rate, label in batch:
            feature = to_mel_image(waveform, sample_rate)
            features.append(feature)
            targets.append(label_to_idx[label])
        features = torch.cat(features, dim=0)
        targets = torch.tensor(targets, dtype=torch.long)
        return features, targets

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(validation_data, batch_size=32, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate)
    return train_loader, validation_loader, test_loader, label_to_idx
