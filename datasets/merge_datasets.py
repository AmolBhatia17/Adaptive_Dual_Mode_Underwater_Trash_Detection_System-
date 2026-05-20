"""
Merge TrashCAN 1.0 + UTD2 (Roboflow) + COCO subset into a single
YOLO-format dataset with deduplication and reproducible train/val/test splits.

Usage:
    python datasets/merge_datasets.py \
        --trashcan  datasets/raw/trashcan \
        --roboflow  datasets/raw/roboflow \
        --coco      datasets/raw/coco \
        --output    datasets/merged \
        [--val-ratio 0.10] [--test-ratio 0.05] [--seed 42]

Outputs:
    datasets/merged/
        data.yaml
        train/images/  train/labels/
        val/images/    val/labels/
        test/images/   test/labels/
        stats.json     ← image & annotation counts per split and class
"""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

CLASS_NAMES = ["Bio", "Rov", "Trash"]
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _md5(path: Path, chunk: int = 65536) -> str:
    """Fast partial MD5 for deduplication (first 256 KB only)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        data = fh.read(chunk * 4)
        h.update(data)
    return h.hexdigest()


def _iter_image_label_pairs(root: Path) -> Iterator[tuple[Path, Path | None]]:
    """
    Yield (image_path, label_path_or_None) for every image under *root*.
    Searches common YOLO directory layouts.
    """
    for img_dir in root.rglob("images"):
        if not img_dir.is_dir():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            lbl_path = img_dir.parent / "labels" / img_path.with_suffix(".txt").name
            yield img_path, lbl_path if lbl_path.exists() else None


def _count_classes(label_path: Path | None) -> dict[int, int]:
    """Return {class_id: count} for a single label file."""
    counts: dict[int, int] = {}
    if label_path is None or not label_path.exists():
        return counts
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            cls = int(parts[0])
            counts[cls] = counts.get(cls, 0) + 1
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Collect all samples from the three source datasets
# ──────────────────────────────────────────────────────────────────────────────

def collect_sources(
    trashcan_dir: Path | None,
    roboflow_dir: Path | None,
    coco_dir: Path | None,
) -> list[tuple[Path, Path | None, str]]:
    """
    Gather (img_path, lbl_path, source_name) from all present source datasets.
    Deduplicates by partial MD5 of the image file.
    """
    all_samples: list[tuple[Path, Path | None, str]] = []
    seen_hashes: set[str] = set()

    sources = [
        (trashcan_dir, "trashcan"),
        (roboflow_dir, "roboflow"),
        (coco_dir,     "coco"),
    ]

    for src_root, src_name in sources:
        if src_root is None or not src_root.exists():
            print(f"  [SKIP] {src_name} — directory not found: {src_root}")
            continue

        before = len(all_samples)
        for img, lbl in _iter_image_label_pairs(src_root):
            h = _md5(img)
            if h in seen_hashes:
                continue           # duplicate
            seen_hashes.add(h)
            all_samples.append((img, lbl, src_name))

        added = len(all_samples) - before
        print(f"  {src_name:10s}: {added:>6d} unique images collected")

    return all_samples


# ──────────────────────────────────────────────────────────────────────────────
# Write merged dataset
# ──────────────────────────────────────────────────────────────────────────────

def write_split(
    samples: list[tuple[Path, Path | None, str]],
    out_root: Path,
    split: str,
) -> dict:
    """Copy images & labels into out_root/{split}/images|labels. Returns stats."""
    img_out = out_root / split / "images"
    lbl_out = out_root / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    class_counts: dict[int, int] = {}
    n_with_label = 0

    for img_path, lbl_path, src in tqdm(samples, desc=f"  {split:5s}", unit="img"):
        # Unique filename: prepend source tag to avoid name collisions
        stem  = f"{src[:3]}_{img_path.stem}"
        out_img = img_out / (stem + img_path.suffix.lower())
        out_lbl = lbl_out / (stem + ".txt")

        shutil.copy2(img_path, out_img)

        if lbl_path and lbl_path.exists():
            shutil.copy2(lbl_path, out_lbl)
            for cls, cnt in _count_classes(lbl_path).items():
                class_counts[cls] = class_counts.get(cls, 0) + cnt
            n_with_label += 1
        else:
            # Write empty label so YOLO doesn't error
            out_lbl.write_text("")

    return {
        "images":         len(samples),
        "labelled":       n_with_label,
        "class_counts":   class_counts,
    }


def merge(
    trashcan_dir: str | None,
    roboflow_dir: str | None,
    coco_dir: str | None,
    output_dir: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    print("\n[Merge] Collecting source datasets …")
    samples = collect_sources(
        Path(trashcan_dir) if trashcan_dir else None,
        Path(roboflow_dir) if roboflow_dir else None,
        Path(coco_dir)     if coco_dir     else None,
    )

    if not samples:
        raise RuntimeError(
            "No samples collected — check that at least one source directory exists "
            "and contains images."
        )

    print(f"\n[Merge] Total unique images: {len(samples)}")

    # Shuffle reproducibly
    random.seed(seed)
    random.shuffle(samples)

    n      = len(samples)
    n_val  = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))
    n_train = n - n_val - n_test

    splits = {
        "train": samples[:n_train],
        "val":   samples[n_train: n_train + n_val],
        "test":  samples[n_train + n_val:],
    }

    out = Path(output_dir)
    stats: dict[str, dict] = {}

    print("\n[Merge] Writing splits …")
    for split_name, split_samples in splits.items():
        stats[split_name] = write_split(split_samples, out, split_name)
        s = stats[split_name]
        cls_str = "  ".join(
            f"{CLASS_NAMES[k]}={v}"
            for k, v in sorted(s["class_counts"].items())
        )
        print(
            f"  {split_name:5s}: {s['images']:>6d} images, "
            f"{s['labelled']:>6d} labelled  [{cls_str}]"
        )

    # Write data.yaml
    yaml_text = (
        f"# DeepClean merged dataset\n"
        f"# Sources: TrashCAN 1.0 + UTD2 (Roboflow) + MS COCO subset\n"
        f"path: {out.resolve()}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"test:  test/images\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    (out / "data.yaml").write_text(yaml_text)

    # Write stats.json
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\n[Merge] data.yaml  → {out / 'data.yaml'}")
    print(f"[Merge] stats.json → {out / 'stats.json'}")
    print("[Merge] Done ✓")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge three datasets into YOLO format")
    parser.add_argument("--trashcan",  default="datasets/raw/trashcan",
                        help="TrashCAN 1.0 root directory")
    parser.add_argument("--roboflow",  default="datasets/raw/roboflow",
                        help="Roboflow UTD2 root directory")
    parser.add_argument("--coco",      default="datasets/raw/coco",
                        help="COCO subset root directory")
    parser.add_argument("--output",    default="datasets/merged",
                        help="Output directory for merged dataset")
    parser.add_argument("--val-ratio",  type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    merge(
        trashcan_dir = args.trashcan,
        roboflow_dir = args.roboflow,
        coco_dir     = args.coco,
        output_dir   = args.output,
        val_ratio    = args.val_ratio,
        test_ratio   = args.test_ratio,
        seed         = args.seed,
    )
