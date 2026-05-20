"""
Download UTD2 dataset from Roboflow Universe.

Usage:
    python datasets/download_roboflow.py --api-key YOUR_KEY [--output datasets/raw/roboflow]

The dataset is available at:
    https://universe.roboflow.com/utd-0dazj/utd2-hyo53
"""

import argparse
import os
import sys
from pathlib import Path


def download(api_key: str, output_dir: str, version: int = 1) -> Path:
    """
    Pull the UTD2 dataset in YOLOv8 format via the Roboflow Python SDK.

    Parameters
    ----------
    api_key   : Roboflow API key (get it from app.roboflow.com → Settings)
    output_dir: Local directory to write images + labels
    version   : Dataset version number (default 1)

    Returns
    -------
    Path to the downloaded dataset root.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit(
            "roboflow package not found.\n"
            "Install with:  pip install roboflow"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[Roboflow] Authenticating …")
    rf = Roboflow(api_key=api_key)

    print(f"[Roboflow] Fetching project  utd-0dazj / utd2-hyo53  v{version} …")
    project = rf.workspace("utd-0dazj").project("utd2-hyo53")
    dataset = project.version(version).download(
        model_format="yolov8",
        location=str(output_path),
        overwrite=True,
    )

    dataset_root = Path(dataset.location)
    print(f"[Roboflow] Dataset saved to  {dataset_root}")
    _print_stats(dataset_root)
    return dataset_root


def _print_stats(root: Path) -> None:
    """Print image counts per split."""
    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        if img_dir.exists():
            n = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
            print(f"  {split:6s}: {n:>5d} images")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download UTD2 from Roboflow")
    parser.add_argument(
        "--api-key",
        required=True,
        help="Your Roboflow API key (find it at app.roboflow.com → Settings)",
    )
    parser.add_argument(
        "--output",
        default="datasets/raw/roboflow",
        help="Directory to download dataset into (default: datasets/raw/roboflow)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Dataset version to download (default: 1)",
    )
    args = parser.parse_args()

    # Allow key via environment variable too
    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        sys.exit(
            "No API key supplied.\n"
            "Pass --api-key or set the ROBOFLOW_API_KEY environment variable."
        )

    download(api_key=api_key, output_dir=args.output, version=args.version)
