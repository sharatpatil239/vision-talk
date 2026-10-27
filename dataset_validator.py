"""
dataset_validator.py - Comprehensive Dataset Validation Utility for VisionTalk

Validates Pascal VOC XML annotations and image files for Object Detection.
Checks:
- Image / XML 1-to-1 correspondence
- Missing and orphaned files
- Duplicate filenames within and across splits
- Empty annotations (<object> tag presence)
- Image dimensions (XML metadata vs actual JPEG header)
- Annotation <filename> tag consistency
- Bounding-box validity (xmin < xmax, ymin < ymax, within image bounds)
- Class name consistency against label map (rejects unknown classes)
- Per-class counts and split distribution

Exits with code 0 on success, code 1 if errors are found.
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from PIL import Image


def parse_label_map(label_map_path):
    """Parses a protobuf label_map.pbtxt file to extract class names and IDs."""
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"Label map not found at: {label_map_path}")
    
    label_to_id = {}
    id_to_label = {}
    with open(label_map_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    current_id = None
    current_name = None
    for line in lines:
        line = line.strip()
        if line.startswith('id:'):
            current_id = int(line.split(':')[1].strip())
        elif line.startswith('name:'):
            # Strip quotes
            current_name = line.split(':')[1].strip().strip("'").strip('"')
        elif line.startswith('}') and current_id is not None and current_name is not None:
            label_to_id[current_name] = current_id
            id_to_label[current_id] = current_name
            current_id = None
            current_name = None
            
    return label_to_id, id_to_label


def validate_dataset(images_dir, label_map_path=None, splits=('train', 'test')):
    """
    Validates all image/XML pairs across specified dataset splits.
    Returns: (is_valid, report_summary)
    """
    errors = []
    warnings = []
    
    label_to_id = {}
    if label_map_path and os.path.exists(label_map_path):
        label_to_id, _ = parse_label_map(label_map_path)
    
    seen_filenames_by_split = defaultdict(set)
    all_seen_basenames = {}
    split_class_counts = defaultdict(Counter)
    total_images_by_split = {}
    total_boxes_by_split = {}
    
    for split in splits:
        split_dir = os.path.join(images_dir, split)
        if not os.path.isdir(split_dir):
            errors.append(f"Split directory does not exist: {split_dir}")
            continue
            
        all_files = os.listdir(split_dir)
        jpg_files = {os.path.splitext(f)[0]: f for f in all_files if f.lower().endswith(('.jpg', '.jpeg'))}
        xml_files = {os.path.splitext(f)[0]: f for f in all_files if f.lower().endswith('.xml')}
        
        total_images_by_split[split] = len(jpg_files)
        total_boxes_by_split[split] = 0
        
        # 1. Check missing pairs
        missing_xml = set(jpg_files.keys()) - set(xml_files.keys())
        missing_jpg = set(xml_files.keys()) - set(jpg_files.keys())
        
        for base in missing_xml:
            errors.append(f"[{split}] Image missing XML annotation: {jpg_files[base]}")
        for base in missing_jpg:
            errors.append(f"[{split}] XML annotation missing image: {xml_files[base]}")
            
        # 2. Check duplicates within and across splits
        for base in jpg_files.keys():
            if base in all_seen_basenames:
                errors.append(f"Duplicate image basename detected: '{base}' exists in both '{all_seen_basenames[base]}' and '{split}'")
            else:
                all_seen_basenames[base] = split
            seen_filenames_by_split[split].add(base)
            
        # 3. Deep inspection of each XML and corresponding JPG
        common_basenames = sorted(list(set(jpg_files.keys()) & set(xml_files.keys())))
        
        for base in common_basenames:
            jpg_name = jpg_files[base]
            xml_name = xml_files[base]
            jpg_path = os.path.join(split_dir, jpg_name)
            xml_path = os.path.join(split_dir, xml_name)
            
            # Read image dimensions
            try:
                with Image.open(jpg_path) as img:
                    img_w, img_h = img.size
                    img_mode = img.mode
            except Exception as e:
                errors.append(f"[{split}] Cannot open image '{jpg_name}': {str(e)}")
                continue
                
            # Parse XML
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except Exception as e:
                errors.append(f"[{split}] XML parse error in '{xml_name}': {str(e)}")
                continue
                
            if root.tag != 'annotation':
                errors.append(f"[{split}] Root tag is '{root.tag}', expected 'annotation' in '{xml_name}'")
                
            # Check XML filename tag
            fn_tag = root.find('filename')
            if fn_tag is None or not fn_tag.text:
                warnings.append(f"[{split}] Missing <filename> tag in '{xml_name}'")
            elif fn_tag.text != jpg_name:
                warnings.append(f"[{split}] <filename> tag '{fn_tag.text}' does not match disk name '{jpg_name}' in '{xml_name}'")
                
            # Check XML size tag
            size_tag = root.find('size')
            if size_tag is None:
                errors.append(f"[{split}] Missing <size> tag in '{xml_name}'")
            else:
                try:
                    xml_w = int(size_tag.find('width').text)
                    xml_h = int(size_tag.find('height').text)
                    if xml_w != img_w or xml_h != img_h:
                        errors.append(f"[{split}] Dimension mismatch in '{xml_name}': XML says ({xml_w}x{xml_h}), actual image is ({img_w}x{img_h})")
                except (ValueError, AttributeError) as e:
                    errors.append(f"[{split}] Invalid <size> values in '{xml_name}': {str(e)}")
                    
            # Check objects and bounding boxes
            objects = root.findall('object')
            if len(objects) == 0:
                errors.append(f"[{split}] Empty annotation: No <object> tags found in '{xml_name}'")
                continue
                
            for obj_idx, obj in enumerate(objects):
                name_tag = obj.find('name')
                if name_tag is None or not name_tag.text:
                    errors.append(f"[{split}] Missing class <name> in '{xml_name}' (object #{obj_idx + 1})")
                    continue
                cls_name = name_tag.text.strip()
                split_class_counts[split][cls_name] += 1
                total_boxes_by_split[split] += 1
                
                # Validate against label map if provided
                if label_to_id and cls_name not in label_to_id:
                    errors.append(f"[{split}] Unknown class '{cls_name}' in '{xml_name}' not defined in label map: {sorted(list(label_to_id.keys()))}")
                    
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    errors.append(f"[{split}] Missing <bndbox> in '{xml_name}' for object '{cls_name}'")
                    continue
                    
                try:
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                except (ValueError, AttributeError) as e:
                    errors.append(f"[{split}] Non-numeric coordinate in '{xml_name}': {str(e)}")
                    continue
                    
                # Geometric box checks
                if xmin >= xmax:
                    errors.append(f"[{split}] Invalid box in '{xml_name}': xmin ({xmin}) >= xmax ({xmax})")
                if ymin >= ymax:
                    errors.append(f"[{split}] Invalid box in '{xml_name}': ymin ({ymin}) >= ymax ({ymax})")
                if xmin < 0 or ymin < 0:
                    errors.append(f"[{split}] Negative coordinates in '{xml_name}': xmin={xmin}, ymin={ymin}")
                if xmax > img_w:
                    errors.append(f"[{split}] Coordinate out of bounds in '{xml_name}': xmax ({xmax}) > image width ({img_w})")
                if ymax > img_h:
                    errors.append(f"[{split}] Coordinate out of bounds in '{xml_name}': ymax ({ymax}) > image height ({img_h})")

    # Generate Report
    is_valid = len(errors) == 0
    return is_valid, errors, warnings, total_images_by_split, total_boxes_by_split, split_class_counts, label_to_id


def print_report(is_valid, errors, warnings, total_images, total_boxes, split_classes, label_to_id):
    """Prints a structured, formatted validation report to stdout."""
    print("================================================================================")
    print("                     VISIONTALK DATASET VALIDATION REPORT                      ")
    print("================================================================================")
    
    print("\n--- DATASET SUMMARY BY SPLIT ---")
    for split in sorted(total_images.keys()):
        imgs = total_images[split]
        boxes = total_boxes.get(split, 0)
        print(f"Split: '{split:<6}' -> Images: {imgs:>3} | Bounding Boxes: {boxes:>3}")
        
    print("\n--- CLASS DISTRIBUTION ---")
    all_classes = set()
    for split, counts in split_classes.items():
        all_classes.update(counts.keys())
    if label_to_id:
        all_classes.update(label_to_id.keys())
        
    sorted_classes = sorted(list(all_classes))
    header = f"{'Class Name':<15} | {'Label ID':<10}"
    for split in sorted(total_images.keys()):
        header += f" | {split:<8}"
    header += " | Total"
    print(header)
    print("-" * len(header))
    
    for cls in sorted_classes:
        cls_id = str(label_to_id.get(cls, "N/A"))
        row = f"{cls:<15} | {cls_id:<10}"
        total_cls = 0
        for split in sorted(total_images.keys()):
            cnt = split_classes[split].get(cls, 0)
            total_cls += cnt
            row += f" | {cnt:<8}"
        row += f" | {total_cls}"
        print(row)
        
    print("\n--- VALIDATION STATUS ---")
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings[:15]:
            print(f"  [WARN] {w}")
        if len(warnings) > 15:
            print(f"  ... and {len(warnings) - 15} more warnings.")
            
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:25]:
            print(f"  [ERROR] {e}")
        if len(errors) > 25:
            print(f"  ... and {len(errors) - 25} more errors.")
        print("\n>>> RESULT: DATASET VALIDATION FAILED! <<<")
    else:
        print("\n>>> RESULT: DATASET VALIDATION PASSED (100% HEALTHY)! <<<")
    print("================================================================================")


def main():
    parser = argparse.ArgumentParser(description="Validate Pascal VOC dataset for VisionTalk sign language detector.")
    parser.add_argument("--images_dir", default="workspace/images", help="Root directory containing train and test folders.")
    parser.add_argument("--label_map", default="workspace/annotations/label_map.pbtxt", help="Path to label_map.pbtxt.")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Dataset splits to validate.")
    parser.add_argument("--fail_on_warning", action="store_true", help="Fail with nonzero exit code even on warnings.")
    
    args = parser.parse_args()
    
    is_valid, errors, warnings, total_images, total_boxes, split_classes, label_to_id = validate_dataset(
        images_dir=args.images_dir,
        label_map_path=args.label_map,
        splits=args.splits
    )
    
    print_report(is_valid, errors, warnings, total_images, total_boxes, split_classes, label_to_id)
    
    if not is_valid:
        sys.exit(1)
    if args.fail_on_warning and warnings:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
