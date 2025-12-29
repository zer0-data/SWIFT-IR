#!/usr/bin/env python3
"""Prepare C3I dataset for Stage A classifier.

Produces an ImageFolder-style layout:
datasets/classifier_data/{train,val}/{clear,fog,rain}/

Features:
- Accepts raw input path (folders or videos).
- Mapping from input folder names to target labels (JSON, YAML, or inline).
- Extract frames from videos every Nth frame.
- Resize images to 64x64 by default (configurable) or keep original.
- Split into train/val with configurable ratio.
"""
import argparse
import os
import random
import shutil
import sys
import uuid
from pathlib import Path

try:
    import cv2
except Exception as e:
    print("Error: OpenCV (cv2) is required. Install with `pip install opencv-python`.")
    raise

try:
    from PIL import Image
except Exception:
    print("Error: Pillow is required. Install with `pip install pillow`.")
    raise

import json

try:
    import yaml
except Exception:
    yaml = None


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.mpeg'}


def is_video_file(p: Path):
    return p.suffix.lower() in VIDEO_EXTS


def is_image_file(p: Path):
    return p.suffix.lower() in IMAGE_EXTS


def load_mapping(mapping_str: str):
    """Load mapping from a file (json/yaml) or inline string 'A:Fog,B:Clear'."""
    if not mapping_str:
        return {}
    p = Path(mapping_str)
    if p.exists():
        if p.suffix.lower() in ('.json',):
            return json.loads(p.read_text())
        if p.suffix.lower() in ('.yml', '.yaml'):
            if yaml is None:
                raise RuntimeError('PyYAML not installed; cannot read YAML mapping')
            return yaml.safe_load(p.read_text())
        else:
            # try json parse anyway
            try:
                return json.loads(p.read_text())
            except Exception:
                raise RuntimeError('Unsupported mapping file format')
    # parse inline mapping: A:Fog,B:Clear
    mapping = {}
    for pair in mapping_str.split(','):
        if not pair.strip():
            continue
        if ':' not in pair:
            raise ValueError('Inline mapping must be KEY:Label pairs separated by commas')
        k, v = pair.split(':', 1)
        mapping[k.strip()] = v.strip()
    return mapping


def ensure_dirs(root: Path, classes):
    for split in ('train', 'val'):
        for c in classes:
            d = root / split / c
            d.mkdir(parents=True, exist_ok=True)


def save_image_array(img_bgr, out_path: Path, resize_to=None):
    # Convert BGR (cv2) to RGB, save with PIL for format flexibility
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    if resize_to:
        img = img.resize((resize_to, resize_to), Image.LANCZOS)
    img.save(out_path)


def copy_or_resize_image(src: Path, dst: Path, resize_to=None):
    if resize_to is None:
        shutil.copy2(src, dst)
        return
    img = Image.open(src).convert('RGB')
    img = img.resize((resize_to, resize_to), Image.LANCZOS)
    img.save(dst)


def extract_frames_from_video(video_path: Path, out_dir: Path, frame_interval: int = 10, resize_to=None, max_frames_per_video=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'Warning: cannot open video {video_path}')
        return 0
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            name = f'{video_path.stem}_{count}_{uuid.uuid4().hex[:6]}.jpg'
            out_path = out_dir / name
            save_image_array(frame, out_path, resize_to)
            saved += 1
            if max_frames_per_video and saved >= max_frames_per_video:
                break
        count += 1
    cap.release()
    return saved


def gather_and_process(input_path: Path, mapping: dict, out_root: Path, frame_interval: int, resize_to, train_ratio: float, seed: int):
    # mapping: folder_name -> label
    # Find classes from mapping values (ensure lowercase canonicalization)
    mapped_labels = {k: v.lower() for k, v in mapping.items()}
    target_classes = set(mapped_labels.values()) if mapped_labels else set()
    target_classes.update(['clear', 'fog', 'rain'])
    target_classes = sorted(target_classes)
    ensure_dirs(out_root, target_classes)

    # collect temp lists per class
    files_per_class = {c: [] for c in target_classes}

    # Walk the input path: if videos found, extract frames into temp structure then assign label
    for entry in sorted(input_path.iterdir()):
        if entry.is_dir():
            folder_key = entry.name
            label = mapped_labels.get(folder_key)
            if label is None:
                # attempt inference from folder name
                name_l = folder_key.lower()
                if 'fog' in name_l:
                    label = 'fog'
                elif 'rain' in name_l:
                    label = 'rain'
                elif 'clear' in name_l or 'sun' in name_l or 'day' in name_l:
                    label = 'clear'
                else:
                    # default to clear if unknown
                    label = 'clear'
            label = label.lower()
            # gather images and videos under this folder
            for f in sorted(entry.rglob('*')):
                if f.is_file():
                    if is_image_file(f):
                        files_per_class.setdefault(label, []).append(f)
                    elif is_video_file(f):
                        # extract frames to a temp dir inside out_root/.temp/<label>
                        temp_dir = out_root / '.temp' / label
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        n = extract_frames_from_video(f, temp_dir, frame_interval, resize_to)
                        if n > 0:
                            for im in sorted(temp_dir.iterdir()):
                                if is_image_file(im):
                                    files_per_class.setdefault(label, []).append(im)
        elif entry.is_file():
            # files directly under input_path
            if is_image_file(entry):
                # guess label from parent or filename
                label = entry.parent.name
                if label in mapped_labels:
                    label = mapped_labels[label]
                else:
                    lname = entry.name.lower()
                    if 'fog' in lname:
                        label = 'fog'
                    elif 'rain' in lname:
                        label = 'rain'
                    else:
                        label = 'clear'
                files_per_class.setdefault(label, []).append(entry)
            elif is_video_file(entry):
                # extract frames into temp
                temp_dir = out_root / '.temp' / 'videos'
                temp_dir.mkdir(parents=True, exist_ok=True)
                n = extract_frames_from_video(entry, temp_dir, frame_interval, resize_to)
                if n > 0:
                    for im in sorted(temp_dir.iterdir()):
                        if is_image_file(im):
                            # attempt to infer label from filename
                            fname = im.name.lower()
                            label = 'clear'
                            if 'fog' in fname:
                                label = 'fog'
                            elif 'rain' in fname:
                                label = 'rain'
                            files_per_class.setdefault(label, []).append(im)

    # Now split and copy to final folders
    random.seed(seed)
    for label, items in files_per_class.items():
        if not items:
            continue
        # canon label
        lab = label.lower()
        if lab not in files_per_class:
            lab = label
        items = list(items)
        random.shuffle(items)
        n_train = int(len(items) * train_ratio)
        train_items = items[:n_train]
        val_items = items[n_train:]

        for src in train_items:
            dest = out_root / 'train' / lab / f'{uuid.uuid4().hex[:12]}{Path(src).suffix.lower()}'
            try:
                if is_image_file(Path(src)):
                    copy_or_resize_image(Path(src), dest, resize_to)
            except Exception as e:
                print(f'Failed to copy {src} -> {dest}: {e}')

        for src in val_items:
            dest = out_root / 'val' / lab / f'{uuid.uuid4().hex[:12]}{Path(src).suffix.lower()}'
            try:
                if is_image_file(Path(src)):
                    copy_or_resize_image(Path(src), dest, resize_to)
            except Exception as e:
                print(f'Failed to copy {src} -> {dest}: {e}')

    # cleanup temp
    temp_root = out_root / '.temp'
    if temp_root.exists():
        try:
            shutil.rmtree(temp_root)
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(description='Prepare C3I classifier dataset (Clear, Fog, Rain)')
    p.add_argument('input', help='Path to raw C3I data (folders or videos)')
    p.add_argument('--mapping', default=None, help='Mapping file (json/yaml) or inline mapping like "FolderA:Fog,FolderB:Clear"')
    p.add_argument('--out', default='datasets/classifier_data', help='Output root for ImageFolder layout')
    p.add_argument('--frame-interval', type=int, default=10, help='Extract every Nth frame from videos')
    p.add_argument('--resize', type=int, default=64, help='Resize images to NxN; set 0 to keep original')
    p.add_argument('--train-ratio', type=float, default=0.8, help='Train split ratio (0-1)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for splitting')
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f'Input path {input_path} does not exist')
        sys.exit(1)

    mapping = load_mapping(args.mapping) if args.mapping else {}
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    resize_to = None if args.resize == 0 else args.resize

    gather_and_process(input_path, mapping, out_root, args.frame_interval, resize_to, args.train_ratio, args.seed)

    print('Done. Dataset prepared at', out_root)


if __name__ == '__main__':
    main()

