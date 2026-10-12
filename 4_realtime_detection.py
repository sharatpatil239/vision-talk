import os
import cv2
import numpy as np
import tensorflow as tf
import os
import sys

RESEARCH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "research"
)

sys.path.insert(0, RESEARCH_PATH)
from object_detection.utils import label_map_util, visualization_utils as viz_utils
from object_detection.builders import model_builder
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

# Paths
WORKSPACE_PATH = 'workspace'
MODEL_PATH = os.path.join(WORKSPACE_PATH, 'models', 'my_ssd_mobnet')
CONFIG_PATH = os.path.join(MODEL_PATH, 'pipeline.config')
LABEL_MAP_PATH = os.path.join(WORKSPACE_PATH, 'annotations', 'label_map.pbtxt')

# Load Pipeline Configuration and Build Detection Model
configs = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
    text_format.Merge(f.read(), configs)

detection_model = model_builder.build(model_config=configs.model, is_training=False)

# Restore Latest Checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(tf.train.latest_checkpoint(MODEL_PATH)).expect_partial()

category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH)

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Start Video Stream
cap = cv2.VideoCapture(0)

print("Starting Real-Time Sign Language Detection (Press 'q' to exit)...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=1,
        min_score_thresh=0.6,
        agnostic_mode=False
    )

    cv2.imshow('Sign Language Real-Time Detection', cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()