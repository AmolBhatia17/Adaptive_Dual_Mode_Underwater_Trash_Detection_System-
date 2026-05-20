"""
Download a relevant subset of MS COCO 2017 and remap to project classes.

Selected COCO categories (those plausible as underwater debris or equipment):
  bottle, cup, wine glass, fork, knife, spoon, bowl  → Trash
  backpack, handbag, suitcase, umbrella              → Trash
  boat                                                → Rov (vessel)

Only images that contain at least one of these categories are downloaded.

Usage:
    python datasets/download_coco.py [--output datasets/raw/coco] [--max-images 15000]
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

# ── COCO 2017 URLs ────────────────────────────────────────────────────────────
COCO_TRAIN_IMAGES  = "http://images.cocodataset.org/zips/train2017.zip"
COCO_VAL_IMAGES    = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS   = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# ── Category remapping ────────────────────────────────────────────────────────
# COCO category name → project class id
COCO_TO_PROJECT = {
    "bottle":     2,   # Trash
    "cup":        2,
    "wine glass": 2,
    "fork":       2,
    "knife":      2,
    "spoon":      2,
    "bowl":       2,
    "backpack":   2,
    "handbag":    2,
    "suitcase":   2,
    "umbrella":   2,
    "boat":       1,   # Rov (vessel / craft)
}

CLASS_NAMES = ["Bio", "Rov", "Trash"]


def _stream_download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already exists: {dest.name} — skipping.")
        return
    print(f"  Downloading {dest.name} …")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0))
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name[:40]
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)
            bar.update(len(chunk))


def _extract(zip_path: Path, dest: Path) -> None:
    import zipfile
    print(f"  Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def _convert_split(
    ann_file: Path,
    src_img_dir: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    max_images: int,
) -> int:
    """Convert annotations and copy matching images. Returns image count."""
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_file) as fh:
        coco = json.load(fh)

    # Build category lookup
    target_cats = {
        cat["id"]: COCO_TO_PROJECT[cat["name"]]
        for cat in coco["categories"]
        if cat["name"] in COCO_TO_PROJECT
    }
    if not target_cats:
        print("  [WARN] No matching categories found in this split.")
        return 0

    # Group annotations by image
    ann_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] in target_cats:
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

    id_to_img = {img["id"]: img for img in coco["images"]}
    relevant_ids = list(ann_by_img.keys())[:max_images]

    copied = 0
    for img_id in tqdm(relevant_ids, desc="  Converting", unit="img"):
        img_info = id_to_img[img_id]
        W, H = img_info["width"], img_info["height"]
        fname = img_info["file_name"]
        stem  = Path(fname).stem

        src = src_img_dir / fname
        if not src.exists():
            continue
        shutil.copy2(src, out_img_dir / fname)

        lines = []
        for ann in ann_by_img[img_id]:
            cls_id = target_cats[ann["category_id"]]
            x, y, w, h = ann["bbox"]
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            nw, nh  = w / W, h / H
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (out_lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
        copied += 1

    return copied


def download(output_dir: str = "datasets/raw/coco", max_images: int = 15000) -> Path:
    root   = Path(output_dir)
    zips   = root / "_zips"
    raw    = root / "_raw"
    zips.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    # 1. Download annotations
    ann_zip = zips / "annotations_trainval2017.zip"
    _stream_download(COCO_ANNOTATIONS, ann_zip)
    ann_dir = raw / "annotations"
    if not ann_dir.exists():
        _extract(ann_zip, raw)

    # 2. Process train split
    train_zip = zips / "train2017.zip"
    _stream_download(COCO_TRAIN_IMAGES, train_zip)
    if not (raw / "train2017").exists():
        _extract(train_zip, raw)

    print("[COCO] Converting train split …")
    n_train = _convert_split(
        ann_file    = ann_dir / "instances_train2017.json",
        src_img_dir = raw / "train2017",
        out_img_dir = root / "train" / "images",
        out_lbl_dir = root / "train" / "labels",
        max_images  = max_images,
    )
    print(f"  → {n_train} train images")

    # 3. Process val split (smaller — use all)
    val_zip = zips / "val2017.zip"
    _stream_download(COCO_VAL_IMAGES, val_zip)
    if not (raw / "val2017").exists():
        _extract(val_zip, raw)

    print("[COCO] Converting val split …")
    n_val = _convert_split(
        ann_file    = ann_dir / "instances_val2017.json",
        src_img_dir = raw / "val2017",
        out_img_dir = root / "val" / "images",
        out_lbl_dir = root / "val" / "labels",
        max_images  = max_images // 5,
    )
    print(f"  → {n_val} val images")

    # 4. Write data.yaml
    yaml_path = root / "data_coco.yaml"
    yaml_path.write_text(
        f"# MS COCO 2017 subset — relevant classes only\n"
        f"path: {root.resolve()}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    print(f"[COCO] data.yaml → {yaml_path}")
    return root


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download COCO subset")
    parser.add_argument("--output", default="datasets/raw/coco")
    parser.add_argument(
        "--max-images",
        type=int,
        default=15000,
        help="Max train images to copy (default 15000)",
    )
    args = parser.parse_args()
    download(output_dir=args.output, max_images=args.max_images)
    print("[COCO] Done ✓")
