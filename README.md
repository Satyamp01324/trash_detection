# Floating & Marine Trash Detection (YOLO)

An intelligent computer vision system powered by YOLOv8 and YOLO11 to detect floating plastic litter, bottles, bags, cans, and marine debris in rivers, lakes, ponds, and shoreline water surfaces.

---

## 🌟 Key Features

- **Multi-Model Support & Combined Ensemble**:
  - **Floating Plastic & River Litter (YOLOv8)**: Detects floating plastic cups, bottles, containers, bags, and cans.
  - **TACO Waste & Bottles (YOLO11)**: Detects real-world litter (bottles, cans, cups, wrappers, cartons).
  - **Combined Ensemble**: Runs multiple specialized detectors in parallel to maximize detection recall on floating litter.
  - **Deep-Sea Debris (YOLOv8-Seg)**: Instance segmentation for seabed and underwater debris.
- **Interactive Gradio Web UI**: Upload images or select sample photos, adjust confidence threshold and NMS IoU sliders, and view real-time detections with detailed category breakdowns.
- **Batch Processing**: Run detections across entire folders of test images and automatically save annotated bounding-box outputs.
- **Auto-Model Download**: Missing weights are automatically downloaded from Hugging Face on the first run.

---

## 📁 Repository Structure

```text
├── app.py                     # Interactive Gradio web application
├── predict.py                 # Batch prediction script for test dataset
├── train.py                   # Model training script
├── download_models.py         # Automatic weight downloader from Hugging Face
├── run.bat                    # Windows 1-click launcher
├── requirements.txt           # Project dependencies
├── floating_trash_yolo/       # Dataset directory
│   ├── classes.txt            # Target class names
│   ├── data.yaml              # Dataset configuration
│   ├── images/                # Train, Val, and Test images
│   └── labels/                # Label annotations
└── models/                    # Model weights directory (auto-downloaded)
```

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/Satyamp01324/trash_detection.git
cd trash_detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Interface
```bash
python app.py
```
Open **[http://127.0.0.1:7860](http://127.0.0.1:7860)** in your web browser.

---

## 📊 Batch Inference

To run batch detection on images in `floating_trash_yolo/images/test/`:

```bash
python predict.py
```
Annotated detection images will be saved to `predictions/test_results/`.

---

## 🏋️ Training Custom Models

To fine-tune or train a model with your own labeled YOLO annotations:

```bash
python train.py
```

---

## 🛠️ Models & Citations

- **Floating Plastic YOLOv8**: Trained on floating riverine and aquatic plastic waste (`Manohar150103/floating-plastic-yolov8`).
- **TACO YOLO11s**: Trained on the Trash Annotations in Context dataset (`fabiocigaina/TACO-yolo11s`).
- **Deep-Sea Marine Trash**: Trained on underwater ROV marine debris (`Krishna-Jaiswal/yolov8m-marine-trash`).
