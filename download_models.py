
"""Helper script to download YOLO model weights if missing."""

import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

MODELS = {
    "floating_plastic_yolov8.pt": {
        "repo_id": "Manohar150103/floating-plastic-yolov8",
        "filename": "last.pt",
    },
    "taco_yolo11s.pt": {
        "repo_id": "fabiocigaina/TACO-yolo11s",
        "filename": "best_model.pt",
    },
    "marine_seabed_yolov8seg.pt": {
        "repo_id": "Krishna-Jaiswal/yolov8m-marine-trash",
        "filename": "yolov8m_marine_best.pt",
    },
}

def ensure_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for target_name, info in MODELS.items():
        target_path = MODELS_DIR / target_name
        if target_path.exists() and target_path.stat().st_size > 1_000_000:
            continue
        print(f"Downloading {target_name} from HuggingFace ({info['repo_id']})...")
        try:
            cached_file = hf_hub_download(repo_id=info["repo_id"], filename=info["filename"])
            shutil.copy(cached_file, target_path)
            print(f"✓ Saved {target_name} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
        except Exception as e:
            print(f"⚠ Could not download {target_name}: {e}")

if __name__ == "__main__":
    ensure_models()
