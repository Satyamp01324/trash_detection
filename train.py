"""Train YOLO model on the floating trash dataset.

Note:
The label files in floating_trash_yolo/labels/ currently contain 0 bytes (empty annotations).
To train custom bounding boxes, annotate the images or use auto-labeled predictions.
"""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "floating_trash_yolo" / "data.yaml"

def main():
    print(f"Loading base YOLOv8 model (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")

    print(f"Starting training with data config: {DATA_YAML}")
    model.train(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=640,
        batch=8,
        project="runs/train",
        name="floating_trash_train",
    )

if __name__ == "__main__":
    main()
