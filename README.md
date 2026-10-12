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

The project implements a complete, modular object detection lifecycle:

```
[Webcam Capture] ──> [Annotation (Pascal VOC)] ──> [TFRecord Generation] ──> [Transfer Learning] ──> [Export & Inference]
1_data_collection.py  workspace/images/train & test  2_generate_tfrecord.py   3_setup_and_train.py   detect.py
```

1. **Data Collection (`1_data_collection.py`)**: Automates webcam capture of gesture samples with a timed delay to vary hand positions and angles.
2. **Annotation & Formatting**: Images are annotated in Pascal VOC XML format with bounding boxes enclosing hand signs.
3. **TFRecord Generation (`2_generate_tfrecord.py` & `scripts/generate_tfrecord.py`)**: Converts raw images and XML annotations into serialized binary TFRecords (`train.record` and `test.record`) along with a protobuf label map (`label_map.pbtxt`).
4. **Model Configuration (`3_setup_and_train.py`)**: Ingests base pre-trained SSD MobileNet V2 weights, updates pipeline hyperparameters (number of classes, batch size, checkpoint paths), and generates `pipeline.config`.
5. **Real-Time Inference (`detect.py`)**: Loads the exported TensorFlow `SavedModel` and runs real-time gesture recognition directly from a webcam feed.

---

## Repository Structure

```
vision-talk/
├── 1_data_collection.py      # Automated image capture via webcam
├── 2_generate_tfrecord.py     # XML annotation to TFRecord & label map converter
├── 3_setup_and_train.py       # Pre-trained model extraction & pipeline configuration
├── 4_realtime_detection.py    # Checkpoint-based detection runner
├── detect.py                  # Primary production inference script (SavedModel)
├── requirements.txt           # Project dependencies
├── .gitignore                 # Excludes caches, checkpoints, and OS artifacts
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

### 3. TensorFlow Object Detection API Setup

The inference scripts utilize helper utilities (`label_map_util`, `visualization_utils`) from the TensorFlow Object Detection API. To make these utilities available:

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

## Running the Inference Demo

Once dependencies and the Object Detection API utilities are installed, launch the real-time webcam detector:

```bash
python detect.py
```

### Requirements for Inference:
- **Webcam**: A functional webcam accessible at device index `0`.
- **Model Files**: The script automatically loads the pre-packaged exported model located at `workspace/models/exported_model/saved_model` and the label map at `workspace/annotations/label_map.pbtxt`.
- **Exit**: Focus on the camera feed window and press **`q`** to quit the stream.

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
  - End-to-end webcam detection performance, real-time FPS, and quantitative test accuracy metrics (such as mean Average Precision / mAP) have **not been independently benchmarked** in this environment and remain **unverified**.

---

## Roadmap & Planned Improvements

- [ ] **Class Label Harmonization**: Align the label name in `1_data_collection.py` and `2_generate_tfrecord.py` (`'thanks'` vs. `'thankyou'`) to ensure seamless retraining.
- [ ] **Standalone Inference Engine**: Decouple `detect.py` from `object_detection.utils` by implementing pure OpenCV bounding box drawing, eliminating the need to install the full TensorFlow Models repository for inference.
- [ ] **Dataset Expansion**: Expand the image count per gesture class to at least 100+ multi-user samples under varied illumination.
- [ ] **Quantitative Benchmarking**: Evaluate detection mAP on an expanded test set and measure inference latency across CPU and GPU hardware.

---

## Acknowledgments & References

- [TensorFlow Model Garden / Object Detection API](https://github.com/tensorflow/models) (Google)
- SSD MobileNet V2 Architecture: *Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018*
- Pascal VOC Annotation Format / [LabelImg](https://github.com/HumanSignal/labelImg)
