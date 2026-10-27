"""
split_dataset.py - Safe & Reproducible Train/Test Dataset Splitter for VisionTalk

Takes annotated Pascal VOC image/XML pairs from a source/staging directory and
distributes them into train/ and test/ directories without cross-contamination.

Features:
- Ensures image and XML are always kept together in the same split.
- Collision detection: guarantees no image basename is already present in train or test.
- Stratified or deterministic seeded random split (default: 80% train, 20% test).
- Dry-run mode (--dry-run) to preview file operations safely.
- Safe non-destructive copy by default; optional --move.
- Detailed stdout report showing per-class split counts.
"""

import os
import sys
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter


def get_image_xml_pairs(source_dir):
    """Recursively or directly locates all valid (jpg, xml) pairs in source_dir."""
    pairs = []
    orphaned_jpgs = []
    orphaned_xmls = []
    
    for root, _, files in os.walk(source_dir):
        jpgs = {os.path.splitext(f)[0]: os.path.join(root, f) for f in files if f.lower().endswith(('.jpg', '.jpeg'))}
        xmls = {os.path.splitext(f)[0]: os.path.join(root, f) for f in files if f.lower().endswith('.xml')}
        
        common = set(jpgs.keys()) & set(xmls.keys())
        for base in common:
            # Extract class name from XML object tag if possible
            cls_name = "unknown"
            try:
                tree = ET.parse(xmls[base])
                name_elem = tree.getroot().find('object/name')
                if name_elem is not None and name_elem.text:
                    cls_name = name_elem.text.strip()
            except Exception:
                pass
            pairs.append((base, jpgs[base], xmls[base], cls_name))
            
        for base in set(jpgs.keys()) - set(xmls.keys()):
            orphaned_jpgs.append(jpgs[base])
        for base in set(xmls.keys()) - set(jpgs.keys()):
            orphaned_xmls.append(xmls[base])
            
    return pairs, orphaned_jpgs, orphaned_xmls


def split_dataset(source_dir, train_dir="workspace/images/train", test_dir="workspace/images/test",
                  split_ratio=0.8, seed=42, dry_run=False, move_files=False):
    """Splits annotated pairs into train and test directories."""
    print("================================================================================")
    print("                    VISIONTALK DATASET SPLIT UTILITY                            ")
    print("================================================================================")
    print(f"Source Directory:  {os.path.abspath(source_dir)}")
    print(f"Train Directory:   {os.path.abspath(train_dir)}")
    print(f"Test Directory:    {os.path.abspath(test_dir)}")
    print(f"Train Split Ratio: {split_ratio * 100:.1f}%")
    print(f"Random Seed:       {seed}")
    print(f"Dry Run Mode:      {dry_run}")
    print(f"Operation Mode:    {'MOVE' if move_files else 'COPY (Non-destructive)'}")
    print("--------------------------------------------------------------------------------")

    if not os.path.exists(source_dir):
        print(f"[ERROR] Source directory does not exist: {source_dir}")
        return False

    # 1. Check existing basenames in train/test to prevent duplicates
    existing_train_basenames = {os.path.splitext(f)[0] for f in os.listdir(train_dir)} if os.path.exists(train_dir) else set()
    existing_test_basenames = {os.path.splitext(f)[0] for f in os.listdir(test_dir)} if os.path.exists(test_dir) else set()
    all_existing = existing_train_basenames | existing_test_basenames

    # 2. Find pairs in source
    pairs, orphaned_jpgs, orphaned_xmls = get_image_xml_pairs(source_dir)
    print(f"Found {len(pairs)} complete (image + XML) pairs in source.")
    if orphaned_jpgs:
        print(f"[WARN] Found {len(orphaned_jpgs)} unannotated images (ignored).")
    if orphaned_xmls:
        print(f"[WARN] Found {len(orphaned_xmls)} orphaned XML files without images (ignored).")

    if len(pairs) == 0:
        print("[WARNING] No complete (image + XML) pairs found to split.")
        return True

    # 3. Check for collisions
    collisions = [p for p in pairs if p[0] in all_existing]
    if collisions:
        print(f"\n[ERROR] {len(collisions)} file(s) in source have matching basenames already in train/test:")
        for base, _, _, _ in collisions[:5]:
            dest = "train" if base in existing_train_basenames else "test"
            print(f"  Conflict: '{base}' already exists in '{dest}' split.")
        print("Aborting split to prevent accidental data overwrites.")
        return False

    # 4. Group by class for stratified split
    by_class = defaultdict(list)
    for p in pairs:
        by_class[p[3]].append(p)

    random.seed(seed)
    train_plan = []
    test_plan = []

    for cls_name, cls_pairs in by_class.items():
        random.shuffle(cls_pairs)
        n_train = int(round(len(cls_pairs) * split_ratio))
        # Ensure at least 1 in test if more than 1 pair exists
        if len(cls_pairs) > 1 and n_train == len(cls_pairs):
            n_train = len(cls_pairs) - 1
            
        train_plan.extend(cls_pairs[:n_train])
        test_plan.extend(cls_pairs[n_train:])

    # 5. Display Plan
    print("\n--- SPLIT PLAN SUMMARY ---")
    all_classes = sorted(list(by_class.keys()))
    train_counts = Counter(p[3] for p in train_plan)
    test_counts = Counter(p[3] for p in test_plan)

    print(f"{'Class Name':<15} | {'Source':<8} | {'Train':<8} | {'Test':<8}")
    print("-" * 45)
    for cls in all_classes:
        src_c = len(by_class[cls])
        tr_c = train_counts.get(cls, 0)
        te_c = test_counts.get(cls, 0)
        print(f"{cls:<15} | {src_c:<8} | {tr_c:<8} | {te_c:<8}")
    print("-" * 45)
    print(f"{'TOTAL':<15} | {len(pairs):<8} | {len(train_plan):<8} | {len(test_plan):<8}")

    if dry_run:
        print("\n[DRY RUN] No files were moved or copied. Re-run without --dry-run to execute.")
        return True

    # 6. Execute split
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    op_fn = shutil.move if move_files else shutil.copy2

    print(f"\nExecuting {'MOVE' if move_files else 'COPY'} operations...")
    for base, jpg_path, xml_path, _ in train_plan:
        op_fn(jpg_path, os.path.join(train_dir, os.path.basename(jpg_path)))
        op_fn(xml_path, os.path.join(train_dir, os.path.basename(xml_path)))

    for base, jpg_path, xml_path, _ in test_plan:
        op_fn(jpg_path, os.path.join(test_dir, os.path.basename(jpg_path)))
        op_fn(xml_path, os.path.join(test_dir, os.path.basename(xml_path)))

    print("\nDataset successfully split and updated!")
    print(f"  Added to train: {len(train_plan)} pairs")
    print(f"  Added to test:  {len(test_plan)} pairs")
    return True


def main():
    parser = argparse.ArgumentParser(description="Split annotated sign language images into train and test sets.")
    parser.add_argument("--source_dir", required=True,
                        help="Path to folder containing annotated (jpg + xml) image pairs.")
    parser.add_argument("--train_dir", default="workspace/images/train",
                        help="Destination directory for training images (default: workspace/images/train).")
    parser.add_argument("--test_dir", default="workspace/images/test",
                        help="Destination directory for testing images (default: workspace/images/test).")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="Ratio of images to allocate to train (default: 0.8 = 80%% train, 20%% test).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic splitting (default: 42).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview split distribution without moving or copying files.")
    parser.add_argument("--move", action="store_true",
                        help="Move files instead of copying them (default: copy).")
                        
    args = parser.parse_args()
    
    success = split_dataset(
        source_dir=args.source_dir,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        split_ratio=args.ratio,
        seed=args.seed,
        dry_run=args.dry_run,
        move_files=args.move
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
