"""
detect.py - Real-Time Sign Language Gesture Detector for VisionTalk

Performs real-time webcam inference or static image inference using the exported
TensorFlow SavedModel and OpenCV, without requiring the external TensorFlow Object
Detection API or protobuf compiler.

Key Architecture & Robustness Features:
1. Resolves model and label map paths relative to the project root directory.
2. Directly loads the TensorFlow SavedModel signature ('serving_default') with uint8 input tensors.
3. Uses the exact class mapping from workspace/annotations/label_map.pbtxt (IDs 1-5, no offset).
4. Pure OpenCV bounding-box drawing, confidence labels, and normalized-to-pixel coordinate conversion.
5. Strict coordinate boundary clamping to prevent out-of-frame drawing errors.
6. Webcam device availability checks, clear diagnostics, q/ESC exit handling,
   and guaranteed resource release via try/finally.
7. Support for single-image inference (--image) for headless testing, CI validation, and debugging.
"""

import argparse
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

# Resolve Project Root (relative to this script's directory)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "workspace", "models", "exported_model", "saved_model"
)
DEFAULT_LABEL_MAP = os.path.join(
    PROJECT_ROOT, "workspace", "annotations", "label_map.pbtxt"
)

# Curated High-Contrast BGR Color Palette for Classes (1 to 5)
CLASS_COLORS = {
    1: (0, 215, 255),    # hello: Bright Yellow / Gold
    2: (255, 140, 0),    # thankyou: Electric Sky Blue
    3: (50, 205, 50),    # yes: Vibrant Lime Green
    4: (30, 30, 240),    # no: Vivid Red
    5: (211, 0, 148),    # iloveyou: Neon Violet / Magenta
}
DEFAULT_BOX_COLOR = (0, 255, 255)


def load_label_map(label_map_path: str) -> Dict[int, str]:
    """
    Parses a Protobuf .pbtxt label map file to extract class IDs and names.
    Does NOT depend on the external TensorFlow Object Detection API or protobuf package.

    Args:
        label_map_path: Path to the label_map.pbtxt file.

    Returns:
        Dict mapping integer class IDs to string class names.
        Example: {1: 'hello', 2: 'thankyou', 3: 'yes', 4: 'no', 5: 'iloveyou'}
    """
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(
            f"Label map file not found at: {label_map_path}\n"
            f"Expected file in: {DEFAULT_LABEL_MAP}"
        )

    with open(label_map_path, "r", encoding="utf-8") as f:
        content = f.read()

    category_index: Dict[int, str] = {}
    items = re.findall(r"item\s*\{([^}]+)\}", content)

    for item in items:
        id_match = re.search(r"id\s*:\s*(\d+)", item)
        name_match = re.search(r"name\s*:\s*['\"]([^'\"]+)['\"]", item)
        if id_match and name_match:
            class_id = int(id_match.group(1))
            class_name = name_match.group(1)
            category_index[class_id] = class_name

    if not category_index:
        raise ValueError(
            f"Failed to parse any valid (id, name) items from label map: {label_map_path}"
        )

    return category_index


class SignLanguageDetector:
    """
    Production-ready wrapper for TensorFlow SavedModel object detection.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        label_map_path: Optional[str] = None,
        min_threshold: float = 0.5,
    ):
        self.model_dir = os.path.abspath(model_dir or DEFAULT_MODEL_DIR)
        self.label_map_path = os.path.abspath(label_map_path or DEFAULT_LABEL_MAP)
        self.min_threshold = float(min_threshold)

        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"SavedModel directory does not exist: {self.model_dir}\n"
                f"Ensure the exported model is located in: {DEFAULT_MODEL_DIR}"
            )

        saved_model_pb = os.path.join(self.model_dir, "saved_model.pb")
        if not os.path.exists(saved_model_pb):
            raise FileNotFoundError(
                f"saved_model.pb not found inside: {self.model_dir}"
            )

        # Load label mappings
        self.category_index = load_label_map(self.label_map_path)

        # Load TensorFlow SavedModel
        print(f"Loading SavedModel from: {self.model_dir} ...")
        load_start = time.time()
        self.model = tf.saved_model.load(self.model_dir)

        # Retrieve serving signature
        if hasattr(self.model, "signatures") and "serving_default" in self.model.signatures:
            self.detect_fn = self.model.signatures["serving_default"]
        else:
            self.detect_fn = self.model

        print(
            f"Model loaded successfully in {time.time() - load_start:.2f}s! "
            f"Classes recognized: {self.category_index}"
        )

    def predict(
        self,
        image_bgr: np.ndarray,
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Runs object detection inference on a single BGR OpenCV frame.

        Args:
            image_bgr: numpy.ndarray representing a BGR image (H, W, 3) uint8.
            threshold: Confidence threshold cutoff (uses self.min_threshold if None).

        Returns:
            List of detected objects sorted by descending confidence.
            Each item is a dict with keys:
                - 'box': [ymin, xmin, ymax, xmax] (normalized floats)
                - 'class_id': int (exact label map ID, e.g. 1..5)
                - 'label': str (e.g. 'hello', 'thankyou', etc.)
                - 'score': float (confidence between 0.0 and 1.0)
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        active_thresh = self.min_threshold if threshold is None else float(threshold)

        # Convert BGR (OpenCV) -> RGB (TensorFlow)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Convert to Tensor (uint8 shape: [1, H, W, 3])
        input_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)[tf.newaxis, ...]

        # Execute SavedModel signature
        raw_outputs = self.detect_fn(input_tensor=input_tensor)

        # Parse outputs
        if "num_detections" in raw_outputs:
            num_detections = int(raw_outputs["num_detections"].numpy().item())
        else:
            num_detections = int(raw_outputs["detection_scores"].shape[1])

        boxes = raw_outputs["detection_boxes"].numpy()[0, :num_detections]
        classes = raw_outputs["detection_classes"].numpy()[0, :num_detections].astype(np.int32)
        scores = raw_outputs["detection_scores"].numpy()[0, :num_detections]

        detections: List[Dict] = []
        for i in range(num_detections):
            score = float(scores[i])
            if score < active_thresh:
                continue

            class_id = int(classes[i])
            ymin, xmin, ymax, xmax = [float(v) for v in boxes[i]]

            # Skip degenerate or non-finite coordinates
            if np.isnan([ymin, xmin, ymax, xmax]).any() or np.isinf([ymin, xmin, ymax, xmax]).any():
                continue

            label = self.category_index.get(class_id, f"class_{class_id}")
            detections.append(
                {
                    "box": [ymin, xmin, ymax, xmax],
                    "class_id": class_id,
                    "label": label,
                    "score": score,
                }
            )

        # Sort by confidence descending
        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    min_threshold: float = 0.5,
    max_boxes: int = 5,
) -> np.ndarray:
    """
    Draws bounding boxes and confidence labels on an image array in-place.
    Clamps normalized coordinates to valid pixel image boundaries.

    Args:
        image: BGR image array (H, W, 3).
        detections: List of detection dictionaries from detector.predict().
        min_threshold: Minimum confidence score to render.
        max_boxes: Maximum number of bounding boxes to draw.

    Returns:
        The annotated image array.
    """
    height, width = image.shape[:2]
    boxes_drawn = 0

    for det in detections:
        if boxes_drawn >= max_boxes:
            break

        score = det["score"]
        if score < min_threshold:
            continue

        ymin, xmin, ymax, xmax = det["box"]

        # Convert normalized coordinates [0.0, 1.0] to pixel coordinates
        top = int(round(ymin * height))
        left = int(round(xmin * width))
        bottom = int(round(ymax * height))
        right = int(round(xmax * width))

        # Boundary clamping to image dimensions
        top = max(0, min(height - 1, top))
        bottom = max(0, min(height, bottom))
        left = max(0, min(width - 1, left))
        right = max(0, min(width, right))

        # Ignore invalid / zero-area detections
        if bottom <= top or right <= left:
            continue

        class_id = det["class_id"]
        color = CLASS_COLORS.get(class_id, DEFAULT_BOX_COLOR)

        # Draw bounding rectangle
        cv2.rectangle(image, (left, top), (right, bottom), color, thickness=2, lineType=cv2.LINE_AA)

        # Prepare label text
        label_text = f"{det['label']}: {score * 100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        margin = 4

        # Position banner above box; if too close to image top, position inside top edge
        if top - text_h - 2 * margin >= 0:
            label_y1 = top - text_h - 2 * margin
            label_y2 = top
        else:
            label_y1 = top
            label_y2 = top + text_h + 2 * margin

        label_x1 = left
        label_x2 = min(width, left + text_w + 2 * margin)

        # Draw filled label background
        cv2.rectangle(
            image,
            (label_x1, label_y1),
            (label_x2, label_y2),
            color,
            thickness=-1,
        )

        # Use dark text for bright backgrounds for optimal readability
        text_color = (0, 0, 0)
        text_y = label_y2 - margin - baseline // 2

        cv2.putText(
            image,
            label_text,
            (label_x1 + margin, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

        boxes_drawn += 1

    return image


def draw_hud(
    image: np.ndarray,
    fps: float,
    threshold: float,
    detections_count: int,
) -> None:
    """Draws a clean top-left status bar for live detection metrics."""
    hud_text = (
        f"FPS: {fps:4.1f} | Detections: {detections_count} | "
        f"Threshold: {threshold:.2f} | [Q/ESC] Quit"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(hud_text, font, font_scale, thickness)

    # Semi-transparent HUD background
    overlay = image.copy()
    cv2.rectangle(overlay, (8, 8), (text_w + 20, text_h + 20), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)

    cv2.putText(
        image,
        hud_text,
        (14, text_h + 14),
        font,
        font_scale,
        (240, 240, 240),
        thickness,
        lineType=cv2.LINE_AA,
    )


def run_webcam_detection(
    detector: SignLanguageDetector,
    camera_id: int = 0,
    threshold: float = 0.5,
    max_boxes: int = 5,
    window_name: str = "VisionTalk - Real-Time Sign Language Detector",
) -> None:
    """
    Executes live webcam detection loop with full error checking and guaranteed resource release.
    """
    print(f"\nInitializing webcam (camera index {camera_id})...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(
            f"ERROR: Could not open camera at device index {camera_id}.\n\n"
            f"Common resolution steps:\n"
            f"  1. Ensure a webcam is plugged in and recognized by Windows.\n"
            f"  2. If using an external USB webcam, try --camera 1 or --camera 2.\n"
            f"  3. Check if another application (Teams, Zoom, Camera app) is holding exclusive access.\n"
            f"  4. Verify Windows Camera privacy settings: Settings > Privacy & security > Camera.\n\n"
            f"To verify detection on a static image instead, run:\n"
            f"  python detect.py --image workspace/images/test/hello.3bdc752c-9a47-11f1-8556-1cce51adcd85.jpg"
        )

    # Validate reading a frame
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        cap.release()
        raise RuntimeError(
            f"ERROR: Webcam opened successfully at index {camera_id}, but failed to read initial frame."
        )

    print("Webcam stream started successfully!")
    print(f"Active confidence threshold: {threshold:.2f}")
    print("Controls: Focus the video window and press 'q' or 'ESC' to exit.\n")

    fps_history = []
    fps = 0.0

    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Failed to capture frame from webcam. Exiting stream.")
                break

            # Predict gestures
            detections = detector.predict(frame, threshold=threshold)

            # Draw visual boxes and labels
            draw_detections(frame, detections, min_threshold=threshold, max_boxes=max_boxes)

            # Compute smoothed FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps_history.append(current_fps)
                if len(fps_history) > 15:
                    fps_history.pop(0)
                fps = sum(fps_history) / len(fps_history)

            # Draw HUD
            draw_hud(frame, fps=fps, threshold=threshold, detections_count=len(detections))

            # Display video window
            cv2.imshow(window_name, frame)

            # Handle user exit (q, Q, or ESC)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("Exit signal received. Shutting down...")
                break

    finally:
        # Guaranteed cleanup
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Webcam released and OpenCV display windows closed.")


def run_image_inference(
    detector: SignLanguageDetector,
    image_path: str,
    threshold: float = 0.5,
    max_boxes: int = 5,
    output_path: Optional[str] = None,
    headless: bool = False,
) -> List[Dict]:
    """
    Runs detection on a static image file, prints detections, and optionally saves/displays visualization.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to decode image from path: {image_path}")

    print(f"\nRunning inference on image: {image_path}")
    start_time = time.time()
    detections = detector.predict(image, threshold=threshold)
    latency_ms = (time.time() - start_time) * 1000.0

    print(f"Inference completed in {latency_ms:.1f}ms. Total detections above {threshold:.2f}: {len(detections)}")
    for idx, det in enumerate(detections[:max_boxes], start=1):
        box_str = ", ".join(f"{v:.4f}" for v in det["box"])
        print(
            f"  [{idx}] Label: '{det['label']}' (ID {det['class_id']}) | "
            f"Confidence: {det['score'] * 100:.2f}% | Box [ymin, xmin, ymax, xmax]: [{box_str}]"
        )

    # Draw boxes
    draw_detections(image, detections, min_threshold=threshold, max_boxes=max_boxes)

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_path, image)
        print(f"Visualized output image saved to: {output_path}")

    if not headless:
        window_name = f"VisionTalk - Detection: {os.path.basename(image_path)}"
        cv2.imshow(window_name, image)
        print("Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detections


def parse_args():
    parser = argparse.ArgumentParser(
        description="VisionTalk: Real-Time Sign Language Gesture Detector"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Path to TensorFlow SavedModel directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--label_map",
        type=str,
        default=DEFAULT_LABEL_MAP,
        help=f"Path to label_map.pbtxt (default: {DEFAULT_LABEL_MAP})",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam camera index (default: 0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for detections (default: 0.5)",
    )
    parser.add_argument(
        "--max_boxes",
        type=int,
        default=5,
        help="Maximum bounding boxes to draw (default: 5)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image file for offline detection instead of live webcam",
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default=None,
        help="Optional path to save visualized image output when --image is used",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable cv2.imshow GUI display (useful for automated testing or headless environments)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        detector = SignLanguageDetector(
            model_dir=args.model_dir,
            label_map_path=args.label_map,
            min_threshold=args.threshold,
        )

        if args.image:
            run_image_inference(
                detector=detector,
                image_path=args.image,
                threshold=args.threshold,
                max_boxes=args.max_boxes,
                output_path=args.output_image,
                headless=args.headless,
            )
        else:
            run_webcam_detection(
                detector=detector,
                camera_id=args.camera,
                threshold=args.threshold,
                max_boxes=args.max_boxes,
            )

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()