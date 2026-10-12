import os
import tarfile
import tensorflow as tf
from google.protobuf import text_format
import os
import sys

RESEARCH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "research"
)

sys.path.insert(0, RESEARCH_PATH)
from object_detection.protos import pipeline_pb2

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = f'https://download.tensorflow.org/models/object_detection/tf2/20200711/{PRETRAINED_MODEL_NAME}.tar.gz'
MODEL_PATH = os.path.join('workspace', 'models', CUSTOM_MODEL_NAME)
PRETRAINED_PATH = os.path.join('workspace', 'pre-trained-models')
ANNOTATION_PATH = os.path.join('workspace', 'annotations')
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(PRETRAINED_PATH, exist_ok=True)

# 1. Download & Extract Pre-trained Weights
# 1. Use manually downloaded & extract Pre-trained Weights
tar_file = os.path.join(
    PRETRAINED_PATH,
    f'{PRETRAINED_MODEL_NAME}.tar'
)

pretrained_model_folder = os.path.join(
    PRETRAINED_PATH,
    PRETRAINED_MODEL_NAME
)

if not os.path.exists(tar_file):
    raise FileNotFoundError(
        f"\nPre-trained model archive not found:\n{tar_file}\n"
        "\nPlease download the model and place the .tar.gz file in "
        "workspace/pre-trained-models."
    )

if not os.path.exists(pretrained_model_folder):
    print("Extracting pre-trained SSD MobileNet V2...")
    with tarfile.open(tar_file, 'r:gz') as tar:
        tar.extractall(PRETRAINED_PATH)
else:
    print("Pre-trained model already extracted.")

# 2. Configure pipeline.config
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
config_path = os.path.join(PRETRAINED_PATH, PRETRAINED_MODEL_NAME, 'pipeline.config')

with tf.io.gfile.GFile(config_path, "r") as f:
    text_format.Merge(f.read(), pipeline_config)

pipeline_config.model.ssd.num_classes = 5
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(PRETRAINED_PATH, PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path = os.path.join(ANNOTATION_PATH, 'label_map.pbtxt')
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(ANNOTATION_PATH, 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = os.path.join(ANNOTATION_PATH, 'label_map.pbtxt')
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(ANNOTATION_PATH, 'test.record')]

dest_config = os.path.join(MODEL_PATH, 'pipeline.config')
with tf.io.gfile.GFile(dest_config, "wb") as f:
    f.write(text_format.MessageToString(pipeline_config).encode('utf-8'))

print(f"\nConfiguration ready at: {dest_config}")
print("\nRun the following command in your terminal to start training:")
print(f"python model_main_tf2.py --model_dir={MODEL_PATH} --pipeline_config_path={dest_config} --num_train_steps=10000")