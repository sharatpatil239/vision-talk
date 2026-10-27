"""
scripts/generate_tfrecord.py - Parameterized TFRecord Generator for VisionTalk

Converts Pascal VOC XML annotations and images from a single directory into a serialized TFRecord.

Features:
- Self-contained pure TensorFlow serialization (no dependency on external OD API protos).
- Strict class validation against the provided label_map.pbtxt.
- Bounding-box coordinate validation and normalization.
- Safety overwrite protection: will not overwrite existing record unless --overwrite is passed.
- Dry-run validation flag (--dry_run) to test dataset health before writing.
"""

import os
import io
import sys
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import tensorflow as tf
from PIL import Image


# --- Pure TensorFlow Serialization Helpers ---
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def load_label_map_dict(labels_path):
    """Loads class-name-to-integer-ID dictionary from label_map.pbtxt."""
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Label map file does not exist: {labels_path}")
        
    mapping = {}
    with open(labels_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    cur_id = None
    cur_name = None
    for line in lines:
        line = line.strip()
        if line.startswith('id:'):
            cur_id = int(line.split(':')[1].strip())
        elif line.startswith('name:'):
            cur_name = line.split(':')[1].strip().strip("'").strip('"')
        elif line.startswith('}') and cur_id is not None and cur_name is not None:
            mapping[cur_name] = cur_id
            cur_id = None
            cur_name = None
            
    if not mapping:
        raise ValueError(f"No valid item entries found in label map: {labels_path}")
    return mapping


def xml_to_df(xml_dir, label_map_dict):
    """Parses XML annotations in xml_dir and validates classes and bounding boxes."""
    xml_list = []
    xml_files = [f for f in os.listdir(xml_dir) if f.lower().endswith(".xml")]

    if not xml_files:
        raise FileNotFoundError(f"No XML files found in: {xml_dir}")

    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        fn_tag = root.find("filename")
        filename = fn_tag.text if fn_tag is not None and fn_tag.text else os.path.splitext(xml_file)[0] + ".jpg"

        size_elem = root.find("size")
        if size_elem is None:
            raise ValueError(f"Missing <size> tag in '{xml_path}'")
        width = int(size_elem.find("width").text)
        height = int(size_elem.find("height").text)

        objects = root.findall("object")
        if not objects:
            raise ValueError(f"Empty annotation (no <object> tags) in '{xml_path}'")

        for obj in objects:
            name_elem = obj.find("name")
            if name_elem is None or not name_elem.text:
                raise ValueError(f"Missing <name> tag in '{xml_path}'")
            cls_name = name_elem.text.strip()

            if cls_name not in label_map_dict:
                valid_classes = sorted(list(label_map_dict.keys()))
                raise ValueError(
                    f"Unknown class '{cls_name}' found in '{xml_path}'. "
                    f"Valid classes in label map: {valid_classes}"
                )

            bndbox = obj.find("bndbox")
            if bndbox is None:
                raise ValueError(f"Missing <bndbox> in '{xml_path}' for class '{cls_name}'")

            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            if xmin >= xmax or ymin >= ymax:
                raise ValueError(f"Invalid bounding box in '{xml_path}': ({xmin},{ymin},{xmax},{ymax})")
            if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                raise ValueError(f"Bounding box out of bounds in '{xml_path}': ({xmin},{ymin},{xmax},{ymax}) vs image ({width}x{height})")

            xml_list.append((filename, width, height, cls_name, xmin, ymin, xmax, ymax))

    columns = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
    return pd.DataFrame(xml_list, columns=columns)


def create_tf_example(group, filename, xml_dir, label_map_dict):
    """Encodes image and bounding boxes into a tf.train.Example."""
    image_path = os.path.join(xml_dir, filename)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found on disk: {image_path}")

    with tf.io.gfile.GFile(image_path, "rb") as fid:
        encoded_jpg = fid.read()

    with Image.open(io.BytesIO(encoded_jpg)) as img:
        width, height = img.size

    filename_bytes = filename.encode("utf8")
    image_format = b"jpg"

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes_int = [], []

    for _, row in group.iterrows():
        norm_xmin = max(0.0, min(1.0, float(row["xmin"]) / width))
        norm_xmax = max(0.0, min(1.0, float(row["xmax"]) / width))
        norm_ymin = max(0.0, min(1.0, float(row["ymin"]) / height))
        norm_ymax = max(0.0, min(1.0, float(row["ymax"]) / height))

        cls_name = row["class"]
        cls_id = label_map_dict[cls_name]

        xmins.append(norm_xmin)
        xmaxs.append(norm_xmax)
        ymins.append(norm_ymin)
        ymaxs.append(norm_ymax)
        classes_text.append(cls_name.encode("utf8"))
        classes_int.append(int(cls_id))

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": int64_feature(height),
                "image/width": int64_feature(width),
                "image/filename": bytes_feature(filename_bytes),
                "image/source_id": bytes_feature(filename_bytes),
                "image/encoded": bytes_feature(encoded_jpg),
                "image/format": bytes_feature(image_format),
                "image/object/bbox/xmin": float_list_feature(xmins),
                "image/object/bbox/xmax": float_list_feature(xmaxs),
                "image/object/bbox/ymin": float_list_feature(ymins),
                "image/object/bbox/ymax": float_list_feature(ymaxs),
                "image/object/class/text": bytes_list_feature(classes_text),
                "image/object/class/label": int64_list_feature(classes_int),
            }
        )
    )
    return tf_example


def main():
    parser = argparse.ArgumentParser(description="Parameterized TFRecord generator for VisionTalk.")
    parser.add_argument("-x", "--xml_dir", required=True, help="Path to folder containing XML annotations and images.")
    parser.add_argument("-l", "--labels_path", required=True, help="Path to label_map.pbtxt file.")
    parser.add_argument("-o", "--output_path", required=True, help="Path to destination .record output file.")
    parser.add_argument("--dry_run", action="store_true", help="Validate XMLs and images without writing output file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_path if it already exists.")

    args = parser.parse_args()

    print(f"Loading label map from: {args.labels_path}")
    label_map_dict = load_label_map_dict(args.labels_path)
    print(f"Loaded {len(label_map_dict)} classes: {label_map_dict}")

    if os.path.exists(args.output_path) and not args.overwrite and not args.dry_run:
        print(f"\n[SAFETY PRESERVATION] Target file '{args.output_path}' already exists.")
        print("To protect verified data, generation was stopped.")
        print("Use --overwrite to overwrite this file.")
        sys.exit(0)

    print(f"Parsing annotations from: {args.xml_dir}...")
    df = xml_to_df(args.xml_dir, label_map_dict)
    grouped = df.groupby("filename")
    print(f"Validated {len(grouped)} images and {len(df)} bounding boxes.")

    if args.dry_run:
        print(f"\n[DRY RUN] All checks passed. Would generate: {args.output_path}")
        sys.exit(0)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    writer = tf.io.TFRecordWriter(args.output_path)
    written_count = 0
    for filename, group in grouped:
        tf_example = create_tf_example(group, filename, args.xml_dir, label_map_dict)
        writer.write(tf_example.SerializeToString())
        written_count += 1
    writer.close()

    file_size = os.path.getsize(args.output_path)
    print(f"\nSuccessfully generated TFRecord: {args.output_path} ({written_count} examples, {file_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()