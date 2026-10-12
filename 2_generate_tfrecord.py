import glob
import io
import os
import xml.etree.ElementTree as ET
import pandas as pd
import tensorflow as tf
from PIL import Image
import sys
RESEARCH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'research')
if os.path.exists(RESEARCH_PATH):
    sys.path.insert(0, RESEARCH_PATH)
from object_detection.utils import dataset_util

LABELS = [
    {'name': 'hello', 'id': 1},
    {'name': 'thanks', 'id': 2},
    {'name': 'yes', 'id': 3},
    {'name': 'no', 'id': 4},
    {'name': 'iloveyou', 'id': 5}
]

ANNOTATION_PATH = os.path.join('workspace', 'annotations')
os.makedirs(ANNOTATION_PATH, exist_ok=True)

# 1. Generate label_map.pbtxt
label_map_file = os.path.join(ANNOTATION_PATH, 'label_map.pbtxt')
with open(label_map_file, 'w') as f:
    for label in LABELS:
        f.write('item {\n')
        f.write(f"\tid: {label['id']}\n")
        f.write(f"\tname: '{label['name']}'\n")
        f.write('}\n')

print(f"Generated {label_map_file}")

# 2. XML to DataFrame Converter
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (
                root.find('filename').text,
                int(root.find('size')[0].text),
                int(root.find('size')[1].text),
                member[0].text,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text)
            )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    return pd.DataFrame(xml_list, columns=column_name)

def class_text_to_int(row_label):
    mapping = {label['name']: label['id'] for label in LABELS}
    return mapping.get(row_label, None)

def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, f"{group.name}"), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.name.encode('utf8')
    image_format = b'jpg'
    xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

# 3. Create records for Train and Test sets
for split in ['train', 'test']:
    image_path = os.path.join('workspace', 'images', split)
    csv_df = xml_to_csv(image_path)
    output_path = os.path.join(ANNOTATION_PATH, f'{split}.record')
    
    writer = tf.io.TFRecordWriter(output_path)
    grouped = csv_df.groupby('filename')
    for filename, group in grouped:
        tf_example = create_tf_example(group, image_path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print(f"Successfully generated {output_path}") 