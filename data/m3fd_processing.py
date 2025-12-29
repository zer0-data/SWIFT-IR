"""Organize M3FD dataset into paired IR/RGB splits for CycleGAN training.

Usage:
	python data/m3fd_processing.py /path/to/M3FD_root \
		--output datasets/m3fd_aligned --split 0.8 --seed 42

The script:
- Verifies each Thermal/IR image has an exact filename match in the Visible/RGB set.
- Builds a paired list, then creates a reproducible train/val split.
- Copies matched files to `output/train/ir`, `output/train/rgb`, `output/val/ir`, `output/val/rgb`.

Filenames are kept identical across IR and RGB folders so CycleGAN can load aligned pairs.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(root: Path) -> Dict[str, Path]:
	"""Recursively find image files and return mapping filename -> path.

	Filenames include extensions and matching requires exact filename match.
	"""
	files: Dict[str, Path] = {}
	for p in root.rglob("*"):
		if p.is_file() and p.suffix.lower() in IMG_EXTS:
			name = p.name  # includes extension
			if name in files:
				# Duplicate filename found in multiple subfolders
				# Keep the first and warn later if needed.
				pass
			else:
				files[name] = p
	return files


def build_pairs(ir_map: Dict[str, Path], vis_map: Dict[str, Path]) -> List[Tuple[Path, Path]]:
	"""Return list of (ir_path, vis_path) for exact filename matches (including extension)."""
	pairs: List[Tuple[Path, Path]] = []
	for name, ir_path in ir_map.items():
		vis_path = vis_map.get(name)
		if vis_path:
			pairs.append((ir_path, vis_path))
	return pairs


def split_pairs(pairs: List[Tuple[Path, Path]], split: float, seed: int) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
	rng = random.Random(seed)
	pairs_copy = list(pairs)
	rng.shuffle(pairs_copy)
	n_train = int(len(pairs_copy) * split)
	return pairs_copy[:n_train], pairs_copy[n_train:]


def ensure_dirs(base: Path):
	for sub in ("train/ir", "train/rgb", "val/ir", "val/rgb"):
		(base / sub).mkdir(parents=True, exist_ok=True)


def copy_pairs(pairs: List[Tuple[Path, Path]], out_ir: Path, out_vis: Path):
	for ir_path, vis_path in pairs:
		shutil.copy2(ir_path, out_ir / ir_path.name)
		shutil.copy2(vis_path, out_vis / vis_path.name)


def main():
	parser = argparse.ArgumentParser(description="Organize M3FD paired IR/RGB dataset for training")
	parser.add_argument("root", type=Path, help="Path to M3FD root containing Visible and Infrared folders")
	parser.add_argument("--vis-name", default="Visible", help="Name of the visible/RGB folder (default: Visible)")
	parser.add_argument("--ir-name", default="Infrared", help="Name of the infrared folder (default: Infrared)")
	parser.add_argument("--output", type=Path, default=Path("datasets/m3fd_aligned"), help="Output base folder")
	parser.add_argument("--split", type=float, default=0.8, help="Train split fraction (default: 0.8)")
	parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible splits")
	args = parser.parse_args()

	root = args.root
	vis_root = root / args.vis_name
	ir_root = root / args.ir_name

	if not vis_root.exists() or not vis_root.is_dir():
		raise SystemExit(f"Visible/RGB folder not found: {vis_root}")
	if not ir_root.exists() or not ir_root.is_dir():
		raise SystemExit(f"Infrared folder not found: {ir_root}")

	print(f"Scanning Visible folder: {vis_root}")
	vis_map = find_images(vis_root)
	print(f"Found {len(vis_map)} visible images")

	print(f"Scanning Infrared folder: {ir_root}")
	ir_map = find_images(ir_root)
	print(f"Found {len(ir_map)} infrared images")

	pairs = build_pairs(ir_map, vis_map)
	print(f"Matched pairs: {len(pairs)}")

	if len(pairs) == 0:
		raise SystemExit("No matched IR-RGB pairs found. Ensure filenames match exactly (including extension).")

	# warn about unmatched counts
	unmatched_ir = set(ir_map.keys()) - set(vis_map.keys())
	unmatched_vis = set(vis_map.keys()) - set(ir_map.keys())
	if unmatched_ir:
		print(f"Warning: {len(unmatched_ir)} IR files have no matching RGB files (examples): {list(unmatched_ir)[:5]}")
	if unmatched_vis:
		print(f"Warning: {len(unmatched_vis)} RGB files have no matching IR files (examples): {list(unmatched_vis)[:5]}")

	train_pairs, val_pairs = split_pairs(pairs, args.split, args.seed)
	print(f"Train pairs: {len(train_pairs)}  Val pairs: {len(val_pairs)}")

	out_base = args.output
	ensure_dirs(out_base)

	copy_pairs(train_pairs, out_base / "train/ir", out_base / "train/rgb")
	copy_pairs(val_pairs, out_base / "val/ir", out_base / "val/rgb")

	print("Done. Output layout:")
	print(f"- {out_base / 'train/ir'} ({len(list((out_base / 'train/ir').iterdir()))} files)")
	print(f"- {out_base / 'train/rgb'} ({len(list((out_base / 'train/rgb').iterdir()))} files)")
	print(f"- {out_base / 'val/ir'} ({len(list((out_base / 'val/ir').iterdir()))} files)")
	print(f"- {out_base / 'val/rgb'} ({len(list((out_base / 'val/rgb').iterdir()))} files)")


if __name__ == "__main__":
	main()


