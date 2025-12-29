"""Convert FLIR ADAS Thermal Dataset to YOLOv8 format.

Usage:
    python -m data.flir_processing --input-dir PATH/TO/FLIR --output-dir datasets/flir_yolo

This script:
 - Finds COCO-style JSON annotations (index.json or any .json) in the input dir
 - Maps FLIR classes to YOLO class ids: Person->0, Car->1, Bicycle->2, Dog->3
 - Converts COCO bbox [x,y,w,h] to YOLO format: class x_center y_center width height (normalized)
 - Copies images into `images/train` and `images/val` and writes label .txt files to `labels/*`.
 - Keeps TIFF files as-is (no 14-bit conversion). JPEG/PNG are copied as-is.

The script is defensive about locating image files (searches recursively if needed).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional


FLIR_TO_YOLO = {
    "person": 0,
    "car": 1,
    "bicycle": 2,
    "bike": 2,
    "dog": 3,
}


def find_json_file(input_dir: Path) -> Optional[Path]:
    # Common names
    candidates = [input_dir / "index.json", input_dir / "annotations.json"]
    for c in candidates:
        if c.exists():
            return c
    # fallback: first .json in top-level
    for p in input_dir.glob("*.json"):
        return p
    # fallback: any json anywhere
    files = list(input_dir.rglob("*.json"))
    return files[0] if files else None


def load_coco(json_path: Path) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def map_categories(categories: List[Dict]) -> Dict[int, str]:
    id2name = {}
    for c in categories:
        cid = c.get("id")
        name = c.get("name") or c.get("label")
        if cid is not None and name is not None:
            id2name[int(cid)] = str(name)
    return id2name


def find_image_file_by_name(image_name: str, input_dir: Path) -> Optional[Path]:
    # If image_name is already a path, try it
    p = input_dir / image_name
    if p.exists():
        return p
    # search recursively for filename match
    for f in input_dir.rglob(image_name):
        return f
    # try basename search
    base = Path(image_name).name
    for f in input_dir.rglob(base):
        return f
    return None


def ensure_dirs(output_dir: Path) -> None:
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def coco_to_yolo_box(bbox: List[float], img_w: int, img_h: int) -> List[float]:
    # COCO bbox: [x_min, y_min, width, height]
    x_min, y_min, w, h = bbox
    x_center = x_min + w / 2.0
    y_center = y_min + h / 2.0
    # normalize
    return [x_center / img_w, y_center / img_h, w / img_w, h / img_h]


def determine_split(image_path: Path) -> str:
    p = str(image_path).lower()
    if "train" in p:
        return "train"
    if "val" in p or "validation" in p:
        return "val"
    return "train"


def process(input_dir: Path, output_dir: Path, annotation_file: Optional[Path] = None) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    ensure_dirs(output_dir)

    json_path = annotation_file or find_json_file(input_dir)
    if not json_path:
        raise FileNotFoundError("No annotation JSON found in input directory")

    data = load_coco(json_path)
    images = {img["id"]: img for img in data.get("images", [])}
    annotations_by_image: Dict[int, List[Dict]] = {}
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        annotations_by_image.setdefault(img_id, []).append(ann)

    id2name = map_categories(data.get("categories", []))

    for img_id, img in images.items():
        file_name = img.get("file_name")
        img_w = img.get("width")
        img_h = img.get("height")

        if not file_name:
            continue

        src_path = find_image_file_by_name(file_name, input_dir)
        if src_path is None:
            # try as plain name
            src_path = find_image_file_by_name(Path(file_name).name, input_dir)
        if src_path is None:
            print(f"Warning: image file for {file_name} not found; skipping")
            continue

        if not img_w or not img_h:
            # attempt to get size via PIL if available
            try:
                from PIL import Image

                with Image.open(src_path) as im:
                    img_w, img_h = im.size
            except Exception:
                print(f"Warning: cannot determine size for {src_path}; skipping")
                continue

        split = determine_split(src_path)
        out_img_dir = output_dir / f"images/{split}"
        out_lbl_dir = output_dir / f"labels/{split}"

        out_img_path = out_img_dir / Path(src_path).name
        out_lbl_path = out_lbl_dir / (Path(src_path).stem + ".txt")

        # copy image (preserve tiffs as-is)
        try:
            shutil.copy2(src_path, out_img_path)
        except Exception as e:
            print(f"Error copying {src_path} -> {out_img_path}: {e}")
            continue

        anns = annotations_by_image.get(img_id, [])
        lines: List[str] = []
        for ann in anns:
            coco_bbox = ann.get("bbox")
            if not coco_bbox:
                continue
            cat_id = ann.get("category_id")
            cat_name = None
            if cat_id is not None:
                cat_name = id2name.get(int(cat_id))
            if not cat_name:
                # try if annotation provides category_name
                cat_name = ann.get("category_name") or ann.get("label")
            if not cat_name:
                print(f"Warning: unknown category for annotation {ann.get('id')} on image {file_name}; skipping")
                continue

            cls_name = str(cat_name).lower()
            if cls_name not in FLIR_TO_YOLO:
                print(f"Skipping unsupported class '{cls_name}'")
                continue
            cls_id = FLIR_TO_YOLO[cls_name]

            yolo_box = coco_to_yolo_box(coco_bbox, int(img_w), int(img_h))
            lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(cls_id, *yolo_box))

        # write label file (may be empty)
        try:
            with open(out_lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            print(f"Error writing label file {out_lbl_path}: {e}")

    print(f"Conversion finished. Output at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert FLIR ADAS Thermal Dataset to YOLOv8 format")
    parser.add_argument("--input-dir", required=True, help="Path to unzipped FLIR dataset")
    parser.add_argument("--output-dir", default="datasets/flir_yolo", help="Output YOLO dataset dir")
    parser.add_argument("--annotation", help="Path to specific annotation JSON (optional)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ann = Path(args.annotation) if args.annotation else None

    process(input_dir, output_dir, annotation_file=ann)


if __name__ == "__main__":
    main()

