import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


# -----------------------------
# Paths
# -----------------------------
PATH_TO_SAVED_MODEL = "workspace/models/exported_model/saved_model"
PATH_TO_LABELS = "workspace/annotations/label_map.pbtxt"


# -----------------------------
# Load model
# -----------------------------
print("Loading model...")

detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

print("Model loaded successfully!")


# -----------------------------
# Load label map
# -----------------------------
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS,
    use_display_name=True
)


# -----------------------------
# Start webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    # Convert BGR -> RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to Tensor
    input_tensor = tf.convert_to_tensor(image_rgb)

    input_tensor = input_tensor[tf.newaxis, ...]

    # Detection
    detections = detect_fn(input_tensor)

    # Convert tensors to numpy
    num_detections = int(detections.pop("num_detections"))

    detections = {
        key: value[0, :num_detections].numpy()
        for key, value in detections.items()
    }

    detections["num_detections"] = num_detections

    detections["detection_classes"] = (
        detections["detection_classes"].astype(np.int64)
    )

    # Draw detections
    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        detections["detection_boxes"],
        detections["detection_classes"],
        detections["detection_scores"],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=0.2,
        agnostic_mode=False
    )

    # Display
    cv2.imshow("Sign Language Detector", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()