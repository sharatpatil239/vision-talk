"""
2_generate_tfrecord.py - Robust TFRecord & Label Map Generator for VisionTalk

Features:
- Self-contained TensorFlow serialization (no dependency on external OD API protos).
- Strict class validation against label map (rejects unknown labels with clear error).
- Geometric bounding-box validation (ensures normalized coordinates are strictly valid).
- Safe overwrite protection: preserves existing train.record / test.record unless --overwrite is set.
- Dry-run mode (--dry-run): validates all dataset files without writing to disk.
- Robust XML parsing using semantic tag queries instead of fragile numerical indexing.
"""

import os
import io
import sys
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import tensorflow as tf
from PIL import Image


# --- Pure TensorFlow Serialization Helpers (Eliminates Protobuf 4.x/5.x import crashes) ---
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


def load_or_create_label_map(label_map_path, default_labels=None, dry_run=False, overwrite=False):
    """
    Loads label map if it exists, or creates it if missing.
    Does NOT overwrite an existing label map unless explicitly requested.
    """
    mapping = {}
    if os.path.exists(label_map_path) and not (overwrite and default_labels):
        print(f"Loading existing label map from: {label_map_path}")
        with open(label_map_path, 'r', encoding='utf-8') as f:
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
        print(f"Loaded {len(mapping)} classes from label map: {mapping}")
        return mapping

    if default_labels:
        mapping = {item['name']: item['id'] for item in default_labels}
        if not dry_run:
            os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
            with open(label_map_path, 'w', encoding='utf-8') as f:
                for item in default_labels:
                    f.write("item {\n")
                    f.write(f"\tid: {item['id']}\n")
                    f.write(f"\tname: '{item['name']}'\n")
                    f.write("}\n\n")
            print(f"Generated new label map at: {label_map_path}")
        else:
            print(f"[DRY RUN] Would write label map to: {label_map_path}")
        return mapping

    raise FileNotFoundError(f"Label map not found at {label_map_path} and no default labels provided.")


def parse_xml_to_df(split_dir, label_mapping):
    """Parses all XML files in a split directory into a validated DataFrame."""
    xml_list = []
    xml_files = [f for f in os.listdir(split_dir) if f.lower().endswith('.xml')]
    
    if len(xml_files) == 0:
        raise FileNotFoundError(f"No XML files found in: {split_dir}")

    for xml_file in xml_files:
        xml_path = os.path.join(split_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Resolve image filename
        fn_tag = root.find('filename')
        filename = fn_tag.text if fn_tag is not None and fn_tag.text else os.path.splitext(xml_file)[0] + '.jpg'
        
        # Verify size
        size_elem = root.find('size')
        if size_elem is None:
            raise ValueError(f"Missing <size> tag in: {xml_path}")
        width = int(size_elem.find('width').text)
        height = int(size_elem.find('height').text)
        
        objects = root.findall('object')
        if len(objects) == 0:
            raise ValueError(f"Empty annotation (no <object> tags) in: {xml_path}")
            
        for obj in objects:
            name_elem = obj.find('name')
            if name_elem is None or not name_elem.text:
                raise ValueError(f"Object missing <name> tag in: {xml_path}")
            cls_name = name_elem.text.strip()
            
            # Strict class validation
            if cls_name not in label_mapping:
                valid_classes = sorted(list(label_mapping.keys()))
                raise ValueError(
                    f"\n[CLASS MISMATCH ERROR] Unknown class '{cls_name}' found in '{xml_path}'.\n"
                    f"Valid classes defined in label map: {valid_classes}.\n"
                    f"Please review and align class names before generating TFRecords."
                )
                
            bndbox = obj.find('bndbox')
            if bndbox is None:
                raise ValueError(f"Object '{cls_name}' missing <bndbox> in: {xml_path}")
                
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # Geometric box validation
            if xmin >= xmax or ymin >= ymax:
                raise ValueError(f"Invalid box coordinates in {xml_path}: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
            if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                raise ValueError(f"Box coordinates out of bounds in {xml_path}: box=({xmin},{ymin},{xmax},{ymax}), img=({width}x{height})")
                
            xml_list.append((filename, width, height, cls_name, xmin, ymin, xmax, ymax))

    columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    return pd.DataFrame(xml_list, columns=columns)


def create_tf_example(group, filename, split_dir, label_mapping):
    """Encodes an image and its bounding boxes into a tf.train.Example."""
    image_path = os.path.join(split_dir, filename)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image referenced in XML does not exist: {image_path}")

    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
        
    with Image.open(io.BytesIO(encoded_jpg)) as img:
        width, height = img.size

    filename_bytes = filename.encode('utf8')
    image_format = b'jpg'
    
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes_int = [], []

    for _, row in group.iterrows():
        # Normalized coordinates [0.0, 1.0]
        norm_xmin = float(row['xmin']) / width
        norm_xmax = float(row['xmax']) / width
        norm_ymin = float(row['ymin']) / height
        norm_ymax = float(row['ymax']) / height
        
        # Clamp strictly between 0 and 1
        norm_xmin = max(0.0, min(1.0, norm_xmin))
        norm_xmax = max(0.0, min(1.0, norm_xmax))
        norm_ymin = max(0.0, min(1.0, norm_ymin))
        norm_ymax = max(0.0, min(1.0, norm_ymax))
        
        cls_name = row['class']
        cls_id = label_mapping[cls_name]
        
        xmins.append(norm_xmin)
        xmaxs.append(norm_xmax)
        ymins.append(norm_ymin)
        ymaxs.append(norm_ymax)
        classes_text.append(cls_name.encode('utf8'))
        classes_int.append(int(cls_id))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename_bytes),
        'image/source_id': bytes_feature(filename_bytes),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes_int),
    }))
    
    return tf_example


def generate_records(images_dir="workspace/images", annotations_dir="workspace/annotations",
                     splits=('train', 'test'), dry_run=False, overwrite=False):
    """Generates train and test TFRecord files with strict safety checks."""
    print("================================================================================")
    print("                  VISIONTALK TFRECORD GENERATION UTILITY                        ")
    print("================================================================================")
    print(f"Images Directory:      {os.path.abspath(images_dir)}")
    print(f"Annotations Directory: {os.path.abspath(annotations_dir)}")
    print(f"Target Splits:         {splits}")
    print(f"Dry Run Mode:          {dry_run}")
    print(f"Overwrite Mode:        {overwrite}")
    print("--------------------------------------------------------------------------------")

    label_map_file = os.path.join(annotations_dir, 'label_map.pbtxt')
    label_mapping = load_or_create_label_map(label_map_file, dry_run=dry_run, overwrite=overwrite)

    for split in splits:
        split_dir = os.path.join(images_dir, split)
        output_record = os.path.join(annotations_dir, f"{split}.record")
        
        print(f"\nProcessing split: '{split}'...")
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Directory not found: {split_dir}")

        if os.path.exists(output_record) and not overwrite and not dry_run:
            print(f"[SAFETY PRESERVATION] Output file '{output_record}' already exists.")
            print("To protect verified data, generation was skipped for this split.")
            print("Use --overwrite if you intend to regenerate this record file.")
            continue

        df = parse_xml_to_df(split_dir, label_mapping)
        grouped = df.groupby('filename')
        print(f"Found {len(grouped)} unique images and {len(df)} total bounding boxes in '{split}'.")
        
        # Display class counts
        class_counts = df['class'].value_counts().to_dict()
        print(f"Class distribution: {class_counts}")

        if dry_run:
            print(f"[DRY RUN] Validated all {len(grouped)} examples successfully. Would write: {output_record}")
            continue

        os.makedirs(annotations_dir, exist_ok=True)
        print(f"Writing binary TFRecord to: {output_record}...")
        writer = tf.io.TFRecordWriter(output_record)
        written_count = 0
        for filename, group in grouped:
            tf_example = create_tf_example(group, filename, split_dir, label_mapping)
            writer.write(tf_example.SerializeToString())
            written_count += 1
        writer.close()
        
        file_size = os.path.getsize(output_record)
        print(f"Successfully generated {output_record} ({written_count} examples, {file_size / (1024*1024):.2f} MB).")

    print("\n================================================================================")
    print("                      TFRECORD WORKFLOW COMPLETE                                ")
    print("================================================================================")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate binary TFRecords from Pascal VOC XML annotations.")
    parser.add_argument("--images_dir", default="workspace/images",
                        help="Root directory containing train and test folders (default: workspace/images).")
    parser.add_argument("--annotations_dir", default="workspace/annotations",
                        help="Output directory for label_map.pbtxt and .record files (default: workspace/annotations).")
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                        help="Splits to process (default: train test).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate annotations and images without writing .record files or modifying label map.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing .record files and label_map.pbtxt if they exist.")

    args = parser.parse_args()

    try:
        generate_records(
            images_dir=args.images_dir,
            annotations_dir=args.annotations_dir,
            splits=args.splits,
            dry_run=args.dry_run,
            overwrite=args.overwrite
        )
    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()