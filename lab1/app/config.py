from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "speech_model.pth"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
SPEECHCOMMANDS_ROOT = Path("/workspace/data/speechcommands") if Path("/workspace").exists() else DATA_DIR / "speechcommands"
ALLOWED_LABELS = ["yes", "no", "up", "down"]
SAMPLE_RATE = 16000
MODEL_NAME = "SpeechCNN"
DATASET_NAME = "Google Speech Commands (subset: yes, no, up, down)"
