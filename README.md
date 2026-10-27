# VisionTalk: Real-Time Sign Language Detector

An end-to-end computer vision and deep learning project for detecting American Sign Language (ASL) gestures in real time using transfer learning on **SSD MobileNet V2** and the **TensorFlow Object Detection API**.

---

## Overview & Objective

The objective of this project is to recognize foundational hand sign gestures from a live webcam stream and render localized bounding boxes with predicted class labels and confidence scores. 

The system is configured to detect **5 gesture classes**:
1. `hello` (Class ID 1)
2. `thankyou` (Class ID 2)
3. `yes` (Class ID 3)
4. `no` (Class ID 4)
5. `iloveyou` (Class ID 5)

---

## Architecture & Technologies

- **Language**: Python (recommended Python 3.9–3.11)
- **Deep Learning Framework**: [TensorFlow 2](https://www.tensorflow.org/) (`tensorflow>=2.10.0,<2.16.0`)
- **Model Architecture**: SSD MobileNet V2 FPNLite 320x320 (`ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8`)
- **Object Detection Framework**: [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- **Computer Vision & Video Streaming**: OpenCV (`opencv-python`)
- **Data & Serialization**: Pascal VOC XML annotations, Pandas, Pillow, Protocol Buffers (`protobuf`), and binary TFRecords

---

## Project Workflow

The project implements a complete, modular, and verified object detection lifecycle:

```
[Webcam Capture] ──> [Annotation (LabelImg)] ──> [Validation] ──> [Safe Split] ──> [TFRecords] ──> [Training] ──> [Inference]
1_data_collection.py  workspace/images/collected  dataset_validator.py  split_dataset.py  2_generate_tfrecord  3_setup_and_train  detect.py
```

1. **Data Collection (`1_data_collection.py`)**: Robust webcam capture tool with pre-roll countdown, progress HUD, and non-blocking preview. Captures are kept isolated in `workspace/images/collectedimages/<class>/`.
2. **Annotation in LabelImg**: Images are labeled with Pascal VOC XML bounding boxes enclosing gestures.
3. **Dataset Validation (`dataset_validator.py`)**: Deep inspection of image/XML pairs, coordinate bounds, image dimensions, and class labels against `label_map.pbtxt`.
4. **Safe Dataset Splitting (`split_dataset.py`)**: Stratified train/test splitter with collision prevention, random seeding, and dry-run preview.
5. **TFRecord Generation (`2_generate_tfrecord.py`)**: Self-contained TFRecord serializer with dry-run verification, strict label matching, and overwrite protection.
6. **Model Configuration (`3_setup_and_train.py`)**: Pipeline configuration for SSD MobileNet V2.
7. **Real-Time Inference (`detect.py`)**: Live webcam detection using exported SavedModel.

---

## Repository Structure

```
vision-talk/
├── 1_data_collection.py      # Robust webcam capture tool (isolated staging)
├── 2_generate_tfrecord.py     # Safe TFRecord generator (--dry_run, --overwrite)
├── 3_setup_and_train.py       # Pre-trained model & pipeline configurator
├── 4_realtime_detection.py    # Checkpoint-based detection runner
├── detect.py                  # Primary production inference script (SavedModel)
├── dataset_validator.py       # Full dataset & annotation integrity validator
├── split_dataset.py           # Safe, non-destructive train/test splitter
├── requirements.txt           # Project dependencies
├── .gitignore                 # Excludes caches, checkpoints, and collectedimages
├── scripts/
│   └── generate_tfrecord.py   # Parameterized CLI utility for TFRecord creation

└── workspace/
    ├── annotations/
    │   ├── label_map.pbtxt    # Protobuf class label definitions (IDs 1-5)
    │   ├── test.record        # Serialized evaluation dataset
    │   └── train.record       # Serialized training dataset
    ├── images/
    │   ├── test/              # 24 test images with Pascal VOC XML annotations
    │   └── train/             # 43 train images with Pascal VOC XML annotations
    ├── models/
    │   ├── exported_model/    # Standalone SavedModel for inference (detect.py)
    │   │   ├── saved_model/   # saved_model.pb and variables
    │   │   ├── checkpoint/    # Export checkpoint
    │   │   └── pipeline.config
    │   └── my_ssd_mobnet/     # Training checkpoint (ckpt-9) and training config
    └── pre-trained-models/    # Base SSD MobileNet V2 model directory
```

---

## Prerequisites & Setup

### 1. Environment Setup

It is strongly recommended to use Python 3.9, 3.10, or 3.11 in an isolated virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate on Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Activate on Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies

Install the core verified Python dependencies:

```bash
pip install -r requirements.txt
```

### 3. TensorFlow Object Detection API Setup (Optional - Training Only)

The inference pipeline (`detect.py` and `4_realtime_detection.py`) runs completely **standalone** using pure TensorFlow SavedModel loading and OpenCV. Installing the TensorFlow Models repository is **not required for inference**.

If you wish to retrain or fine-tune models from scratch using `3_setup_and_train.py`, install Google's official TensorFlow Model Garden:

```bash
# Clone the TensorFlow Models repository
git clone https://github.com/tensorflow/models.git

# Install the Object Detection API package
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
cd ../..
```

---

## Step-by-Step Dataset Workflow Guide

### 1. Collect Raw Gesture Images
Run the reliable webcam data collector to stage new samples:

```bash
# Collect 15 images for 'thankyou' (isolated in workspace/images/collectedimages/thankyou/)
python 1_data_collection.py --labels thankyou --num_images 15 --interval 2.0

# Collect for all foundational classes:
python 1_data_collection.py --labels hello thankyou yes no iloveyou --num_images 15
```
- **Live HUD**: Displays pre-roll countdown (`Starting in: 5s`), current counter (`Captured: X/Y`), and interval ticker.
- **Controls**: Press **`q`** or **`ESC`** at any moment to safely exit. Hardware is guaranteed to release in a `finally` block.
- **Isolation**: Images are saved strictly to `workspace/images/collectedimages/<class>/` to keep raw images separate from verified training data.

### 2. Annotate in LabelImg (Pascal VOC XML)
Install and launch LabelImg on your staged raw images:

```bash
# Install LabelImg in your Python environment
pip install labelImg

# Launch LabelImg pointing to your collected images
labelImg workspace/images/collectedimages/thankyou
```

**LabelImg Setup Checklist**:
1. In the left toolbar, click **Change Save Dir** and set it to the same directory as your images (`workspace/images/collectedimages/thankyou/`).
2. Verify that the annotation format button below Save is set to **`PascalVOC`** (generates `.xml` files).
3. Click **View** in the top menu and enable **Auto Save Mode**.
4. Draw bounding boxes around the hand gesture by pressing **`w`**.
5. **Class Naming Rule**: Always type the exact verified class name (e.g. `thankyou`, `hello`, `yes`, `no`, `iloveyou`).
6. Press **`d`** to proceed to the next image.

### 3. Validate Dataset Quality
Audit all image/XML pairs, coordinate bounds, and label mappings before splitting:

```bash
python dataset_validator.py
```
- Performs 8 automated checks (bounding box boundaries, image dimensions, orphaned files, duplicate names across splits).
- Returns **Exit code 0** on success; returns **Exit code 1** with descriptive error logs if any defects exist.

### 4. Split Staged Data into Train and Test Sets
Distribute validated (image + XML) pairs into `train/` (80%) and `test/` (20%) without cross-contamination:

```bash
# Preview split distribution safely with dry run:
python split_dataset.py --source_dir workspace/images/collectedimages/thankyou --dry_run

# Execute safe non-destructive copy into workspace/images/train and test:
python split_dataset.py --source_dir workspace/images/collectedimages/thankyou
```
- **Collision Protection**: Checks if an image basename already exists in `train/` or `test/` and aborts if a duplicate is found.
- **Paired Distribution**: Guarantees each `.jpg` and its `.xml` are always placed together in the exact same split.

### 5. Generate Binary TFRecords
Convert the validated dataset into binary TFRecords for model training:

```bash
# Step A: Validate dataset without writing (Dry-run mode):
python 2_generate_tfrecord.py --dry_run

# Step B: Generate records (Protected: requires --overwrite to replace existing):
python 2_generate_tfrecord.py --overwrite
```
- Uses self-contained TensorFlow serialization (no dependency on external OD API protos).
- Strictly rejects any class not present in `label_map.pbtxt`.

---

## Running the Inference Demo

Launch real-time webcam gesture detection:

```bash
# Primary production detector
python detect.py

# Or via the backwards-compatibility wrapper:
python 4_realtime_detection.py
```

### Advanced Detection Options:
```bash
# Adjust confidence cutoff (default is 0.50):
python detect.py --threshold 0.65

# Switch webcam index if multiple cameras are connected:
python detect.py --camera 1

# Run offline inference on a static test image (no camera required):
python detect.py --image workspace/images/test/hello.3bdc752c-9a47-11f1-8556-1cce51adcd85.jpg

# Save visualized detection output to file:
python detect.py --image workspace/images/test/hello.3bdc752c-9a47-11f1-8556-1cce51adcd85.jpg --output_image demo_out.jpg --headless
```

### Controls & Safeguards:
- **Exit**: Focus on the camera feed window and press **`q`** or **`ESC`**.
- **Resource Protection**: Webcam device capture and GUI display windows are guaranteed to release cleanly via `try/finally`.
- **Diagnostics**: If camera index 0 is inaccessible (in use by Zoom/Teams or permission disabled), the script outputs actionable diagnosis and fallback commands.

---

## Dataset Description & Limitations

- **Dataset Size**: The current dataset contains **67 custom annotated images** (43 in `train/`, 24 in `test/`).
- **Controlled Scope**: Images were recorded in a single indoor environment with consistent lighting.
- **Limitations**:
  - Sample size is small and intended as a proof-of-concept / prototype.
  - Model generalization across varying skin tones, complex background clutter, and diverse camera angles has not been evaluated.
  - Edge cases (fast hand motions, partial occlusions) may yield false positives or reduced detection confidence.

---

## Current Project Status

- **Code & Asset Integrity**: Repository structure, exported model weights, dataset annotations, and configuration files are intact and organized.
- **Verification Status**:
  - Code syntax, paths, and XML annotation schemas have been statically validated.
  - Standalone SavedModel inference and class mapping (1=hello, 2=thankyou, 3=yes, 4=no, 5=iloveyou) have been verified on test images with >95% confidence.
  - Webcam capture handles q/ESC exits, camera unavailability, and try/finally cleanup.

---

## Roadmap & Planned Improvements

- [ ] **Class Label Harmonization**: Align the label name in `1_data_collection.py` and `2_generate_tfrecord.py` (`'thanks'` vs. `'thankyou'`) to ensure seamless retraining.
- [x] **Standalone Inference Engine**: Decouple `detect.py` from `object_detection.utils` by implementing pure OpenCV bounding box drawing, eliminating the need to install the full TensorFlow Models repository for inference.
- [ ] **Dataset Expansion**: Expand the image count per gesture class to at least 100+ multi-user samples under varied illumination.
- [ ] **Quantitative Benchmarking**: Evaluate detection mAP on an expanded test set and measure inference latency across CPU and GPU hardware.

---

## Acknowledgments & References

- [TensorFlow Model Garden / Object Detection API](https://github.com/tensorflow/models) (Google)
- SSD MobileNet V2 Architecture: *Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018*
- Pascal VOC Annotation Format / [LabelImg](https://github.com/HumanSignal/labelImg)
