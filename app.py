"""Enhanced Multi-Model Gradio Web UI for Floating Trash & Litter Detection."""

import collections
from pathlib import Path
import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO
from download_models import ensure_models

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATASET_DIR = ROOT / "floating_trash_yolo"

# Auto-download models if missing
ensure_models()

# Load models into memory for instant inference
print("Loading detection models...")
MODEL_PATHS = {
    "Floating Plastic & Litter (YOLOv8)": MODELS_DIR / "floating_plastic_yolov8.pt",
    "TACO Waste & Bottles (YOLO11)": MODELS_DIR / "taco_yolo11s.pt",
    "Deep-Sea Seabed Debris (YOLOv8-Seg)": MODELS_DIR / "marine_seabed_yolov8seg.pt",
}

LOADED_MODELS = {}
for name, p in MODEL_PATHS.items():
    if p.exists():
        try:
            print(f"Loading {name} from {p.name}...")
            LOADED_MODELS[name] = YOLO(str(p))
        except Exception as e:
            print(f"Could not load {name}: {e}")

# Color palettes for bounding boxes
PALETTE = [
    (0, 220, 0),     # Bright Green
    (0, 165, 255),   # Orange
    (255, 100, 0),   # Blue
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Magenta
    (255, 200, 0),   # Cyan
    (50, 120, 255),  # Coral
]

def draw_boxes(image_bgr, boxes_data, show_masks=False, result_seg=None):
    vis = image_bgr.copy()
    h, w = vis.shape[:2]

    # Optional segmentation mask
    if show_masks and result_seg is not None and result_seg.masks is not None:
        try:
            vis = result_seg.plot(boxes=False)
        except Exception:
            pass

    for i, item in enumerate(boxes_data):
        x1, y1, x2, y2 = item["box"]
        color = PALETTE[item["cls_id"] % len(PALETTE)]
        label = f"{item['label']} {item['conf']:.2f}"

        # Draw bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Draw text label banner
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bg_y1 = max(y1 - th - 6, 0)
        bg_y2 = y1
        cv2.rectangle(vis, (x1, bg_y1), (min(x1 + tw + 6, w), bg_y2), color, -1)
        cv2.putText(
            vis,
            label,
            (x1 + 3, max(y1 - 4, th + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return vis

def run_detection(image, model_choice, conf, iou_thresh, show_masks):
    if image is None:
        return None, "Please upload or select an image."

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detected_items = []
    seg_result = None

    if model_choice == "Combined Detector (Floating Plastic + TACO Bottles)":
        m_plastic = LOADED_MODELS.get("Floating Plastic & Litter (YOLOv8)")
        m_taco = LOADED_MODELS.get("TACO Waste & Bottles (YOLO11)")

        if m_plastic:
            r1 = m_plastic.predict(img_bgr, conf=float(conf), iou=float(iou_thresh), verbose=False)[0]
            for b in r1.boxes:
                detected_items.append({
                    "box": list(map(int, b.xyxy[0])),
                    "label": m_plastic.names[int(b.cls[0])],
                    "cls_id": int(b.cls[0]),
                    "conf": float(b.conf[0]),
                    "source": "Floating Plastic",
                })

        if m_taco:
            r2 = m_taco.predict(img_bgr, conf=float(conf), iou=float(iou_thresh), verbose=False)[0]
            for b in r2.boxes:
                detected_items.append({
                    "box": list(map(int, b.xyxy[0])),
                    "label": m_taco.names[int(b.cls[0])],
                    "cls_id": int(b.cls[0]) + 10,
                    "conf": float(b.conf[0]),
                    "source": "TACO",
                })
    else:
        model = LOADED_MODELS.get(model_choice)
        if model is None:
            return None, f"Model '{model_choice}' is not available."
        r = model.predict(img_bgr, conf=float(conf), iou=float(iou_thresh), verbose=False)[0]
        seg_result = r
        for b in r.boxes:
            detected_items.append({
                "box": list(map(int, b.xyxy[0])),
                "label": model.names[int(b.cls[0])],
                "cls_id": int(b.cls[0]),
                "conf": float(b.conf[0]),
                "source": model_choice,
            })

    # Render image with clean bounding boxes
    annotated_bgr = draw_boxes(img_bgr, detected_items, show_masks=show_masks, result_seg=seg_result)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Build detailed summary markdown
    if not detected_items:
        summary_md = (
            f"### No trash objects detected at confidence threshold `{conf:.2f}`\n\n"
            f"> **Tip**: Try lowering the **Confidence Threshold** slider (e.g. to `0.10` or `0.08`) "
            f"to capture smaller or partially submerged floating debris."
        )
    else:
        counts = collections.Counter(item["label"] for item in detected_items)
        breakdown = "\n".join([f"- **{label}**: {count} item(s)" for label, count in counts.most_common()])
        summary_md = (
            f"###  Found **{len(detected_items)}** trash object(s)\n\n"
            f"#### Category Breakdown:\n"
            f"{breakdown}\n\n"
            f"| # | Detected Class | Confidence | Model Source |\n"
            f"|---|---|---|---|\n"
        )
        for idx, item in enumerate(detected_items, 1):
            summary_md += f"| {idx} | `{item['label']}` | **{item['conf']*100:.1f}%** | {item['source']} |\n"

    return annotated_rgb, summary_md

def main():
    test_dir = DATASET_DIR / "images" / "test"
    example_files = sorted(test_dir.glob("*.jpg")) if test_dir.exists() else []
    # Also add user crop image if available
    if Path("test_crop.jpg").exists():
        example_files.insert(0, Path("test_crop.jpg"))

    examples = [
        [str(p), "Combined Detector (Floating Plastic + TACO Bottles)", 0.15, 0.45, False]
        for p in example_files[:6]
    ]

    model_options = [
        "Combined Detector (Floating Plastic + TACO Bottles)",
        "Floating Plastic & Litter (YOLOv8)",
        "TACO Waste & Bottles (YOLO11)",
        "Deep-Sea Seabed Debris (YOLOv8-Seg)",
    ]

    with gr.Blocks(title="Floating Trash & River Litter Detection") as demo:
        gr.Markdown(
            "# Floating & Marine Trash Detection (YOLO)\n"
            "Detect floating plastic bottles, bags, cups, cans, and debris in water surfaces. "
            "Upload an image, pick a model, and adjust detection sensitivity."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="numpy", label="Input Photo (Water / River / Shoreline)")
                model_dropdown = gr.Dropdown(
                    choices=model_options,
                    value="Combined Detector (Floating Plastic + TACO Bottles)",
                    label="Detection Model",
                )
                with gr.Row():
                    conf_slider = gr.Slider(0.05, 0.80, value=0.15, step=0.01, label="Confidence Threshold")
                    iou_slider = gr.Slider(0.10, 0.90, value=0.45, step=0.05, label="NMS / IoU Overlap")
                show_masks_chk = gr.Checkbox(
                    value=False,
                    label="Show Segmentation Masks (for Deep-Sea model only; keep off to avoid blue tint)",
                )
                submit_btn = gr.Button("Detect Trash", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(type="numpy", label="Detection Output (Annotated Boxes)")
                summary_output = gr.Markdown(label="Detection Breakdown")

        if examples:
            gr.Examples(
                examples=examples,
                inputs=[input_image, model_dropdown, conf_slider, iou_slider, show_masks_chk],
                outputs=[output_image, summary_output],
                fn=run_detection,
                cache_examples=False,
            )

        submit_btn.click(
            fn=run_detection,
            inputs=[input_image, model_dropdown, conf_slider, iou_slider, show_masks_chk],
            outputs=[output_image, summary_output],
        )

    print("\nStarting Enhanced Gradio App on http://127.0.0.1:7860 ...")
    demo.launch(server_name="127.0.0.1", server_port=7860)

if __name__ == "__main__":
    main()
