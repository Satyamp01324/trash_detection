"""Run YOLO inference on floating trash images using dedicated surface plastic & TACO models."""

import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "floating_trash_yolo"
MODELS_DIR = ROOT / "models"

MODEL_PATH = MODELS_DIR / "floating_plastic_yolov8.pt"
TACO_PATH = MODELS_DIR / "taco_yolo11s.pt"

def main():
    if not MODEL_PATH.exists():
        print(f"Model checkpoint not found at: {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading Floating Plastic YOLO model from: {MODEL_PATH.name}")
    model = YOLO(str(MODEL_PATH))

    test_images_dir = DATASET_DIR / "images" / "test"
    output_dir = ROOT / "predictions"
    output_dir.mkdir(exist_ok=True)

    print(f"Running detection on test images in: {test_images_dir}")
    results = model.predict(
        source=str(test_images_dir),
        conf=0.15,
        save=True,
        project=str(output_dir),
        name="test_results",
        exist_ok=True,
    )

    print(f"\n[DONE] Processed {len(results)} images.")
    save_path = output_dir / "test_results"
    print(f"Annotated images saved to: {save_path.resolve()}")
    for r in results:
        boxes = len(r.boxes) if r.boxes is not None else 0
        img_name = Path(r.path).name
        detected_classes = [model.names[int(b.cls[0])] for b in r.boxes] if r.boxes else []
        print(f"  - {img_name}: {boxes} object(s) -> {detected_classes}")

if __name__ == "__main__":
    main()
