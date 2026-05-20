"""
Download TrashCAN 1.0 dataset from UMN IRV Lab.

Reference:
  Hong, J. et al. (2020). TrashCan: A Semantically-Segmented Dataset towards
  Visual Detection of Marine Debris. arXiv:2007.08097
  http://irvlab.cs.umn.edu/resources/trash-can-dataset

Usage:
    python datasets/download_trashcan.py [--output datasets/raw/trashcan]

The script downloads the instance-segmentation split, converts COCO-JSON
annotations to flat YOLO bounding-box format, and remaps the 16 original
categories to the 3 project classes: Bio | Rov | Trash.
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ── Dataset URL ───────────────────────────────────────────────────────────────
TRASHCAN_URL = (
    "https://conservancy.umn.edu/bitstream/handle/11299/214366/"
    "TrashCan_1.0.zip?sequence=22&isAllowed=y"
)
TRASHCAN_ZIP = "TrashCan_1.0.zip"

# ── Category mapping: TrashCAN (16 classes) → Project (3 classes) ─────────────
# 0=Bio  1=Rov  2=Trash
CATEGORY_MAP = {
    # Anthropogenic debris → Trash
    "can":            2,
    "carton":         2,
    "metal":          2,
    "paper":          2,
    "plastic":        2,
    "plastic_bag":    2,
    "plastic_bottle": 2,
    "rope":           2,
    "tire":           2,
    "wood":           2,
    "trash":          2,
    # Biological / organic → Bio
    "bio":            0,
    "fish":           0,
    "plant":          0,
    "sea_urchin":     0,
    "eel":            0,
    # Equipment → Rov
    "rov":            1,
    "unknown":        2,   # default to Trash
}

CLASS_NAMES = ["Bio", "Rov", "Trash"]


def _download_file(url: str, dest: Path) -> None:
    """Stream-download a file with a progress bar."""
    print(f"[TrashCAN] Downloading from UMN Conservancy …")
    print(f"  URL : {url}")
    print(f"  Dest: {dest}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0))
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=TRASHCAN_ZIP
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))


def _coco_to_yolo(coco_json: dict, image_dir: Path, label_dir: Path) -> int:
    """
    Convert COCO instance-annotation JSON to per-image YOLO .txt files.
    Returns number of annotations written.
    """
    label_dir.mkdir(parents=True, exist_ok=True)

    # Build lookup structures
    id_to_img  = {img["id"]: img for img in coco_json["images"]}
    cat_name   = {cat["id"]: cat["name"].lower() for cat in coco_json["categories"]}

    # Group annotations by image
    ann_by_img: dict[int, list] = {}
    for ann in coco_json["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    written = 0
    for img_id, anns in ann_by_img.items():
        img_info = id_to_img[img_id]
        W, H = img_info["width"], img_info["height"]
        file_stem = Path(img_info["file_name"]).stem

        lines = []
        for ann in anns:
            raw_name = cat_name.get(ann["category_id"], "unknown")
            # Match by substring to handle minor label variations
            cls_id = next(
                (v for k, v in CATEGORY_MAP.items() if k in raw_name),
                2,  # default → Trash
            )
            x, y, w, h = ann["bbox"]  # COCO format: top-left x, y, w, h
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if lines:
            (label_dir / f"{file_stem}.txt").write_text("\n".join(lines))
            written += len(lines)

    return written


def convert(raw_root: Path, output_dir: Path) -> None:
    """Convert the extracted TrashCAN zip into YOLO layout."""
    # Locate annotation files
    ann_root = raw_root / "dataset_instances"
    if not ann_root.exists():
        ann_root = raw_root  # flat layout fallback

    for split in ("train", "val"):
        json_file = ann_root / f"instances_{split}2019.json"
        if not json_file.exists():
            print(f"  [WARN] Annotation file not found: {json_file} — skipping.")
            continue

        with open(json_file) as fh:
            coco = json.load(fh)

        image_dir = raw_root / split
        label_dir = output_dir / split / "labels"
        # Copy images reference (labels only — images kept in raw)
        n = _coco_to_yolo(coco, image_dir, label_dir)
        print(f"  {split:6s}: {n:>6d} annotations written → {label_dir}")

    # Write data.yaml stub
    yaml_path = output_dir / "data_trashcan.yaml"
    yaml_path.write_text(
        f"# TrashCAN 1.0 — converted to YOLO format\n"
        f"path: {output_dir.resolve()}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    print(f"  data.yaml written → {yaml_path}")


def download(output_dir: str = "datasets/raw/trashcan") -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    zip_path = root / TRASHCAN_ZIP
    if not zip_path.exists():
        _download_file(TRASHCAN_URL, zip_path)
    else:
        print(f"[TrashCAN] ZIP already exists at {zip_path} — skipping download.")

    print(f"[TrashCAN] Extracting …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)

    extracted = next(
        (p for p in root.iterdir() if p.is_dir() and p.name != "__MACOSX"),
        root,
    )
    print(f"[TrashCAN] Extracted to {extracted}")
    convert(extracted, root)
    return root


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & convert TrashCAN 1.0")
    parser.add_argument(
        "--output",
        default="datasets/raw/trashcan",
        help="Output directory (default: datasets/raw/trashcan)",
    )
    args = parser.parse_args()
    download(output_dir=args.output)
    print("[TrashCAN] Done ✓")
