"""
Train YOLOv12x — Highest-Capacity Detector
============================================
71.6 M parameters — used as the upper-bound accuracy reference.

Usage:
    python training/train_yolov12x.py [--data datasets/merged/data.yaml]
                                      [--epochs 180] [--device 0]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def train(data: str, epochs: int, device, resume: bool) -> None:
    from ultralytics import YOLO

    try:
        model = YOLO("yolov12x.pt")
    except Exception:
        print("[WARN] yolov12x.pt not found in hub — initialising from yolov8x.pt")
        model = YOLO("yolov8x.pt")

    print("=" * 60)
    print("  DeepClean — Training YOLOv12x (highest-capacity)")
    print(f"  Data   : {data}")
    print(f"  Epochs : {epochs}")
    print(f"  Device : {device}")
    print("  NOTE: ~24 GB VRAM recommended at batch=4")
    print("=" * 60)

    model.train(
        data         = data,
        epochs       = epochs,
        imgsz        = 640,
        batch        = 4,
        device       = device,
        optimizer    = "AdamW",
        lr0          = 0.0005,
        lrf          = 0.01,
        weight_decay = 0.0005,
        warmup_epochs= 5,
        mosaic       = 1.0,
        mixup        = 0.15,
        copy_paste   = 0.1,
        flipud       = 0.1,
        fliplr       = 0.5,
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        project      = "training/results",
        name         = "yolov12x",
        save_period  = 20,
        patience     = 30,
        exist_ok     = True,
        resume       = resume,
        plots        = True,
        val          = True,
        verbose      = True,
    )

    best = Path("training/results/yolov12x/weights/best.pt")
    if best.exists():
        metrics = YOLO(str(best)).val(data=data, imgsz=640, device=device)
        print(f"\n  mAP@0.5        : {metrics.box.map50:.4f}")
        print(f"  mAP@[0.5:0.95] : {metrics.box.map:.4f}")

    print("\n[Done] YOLOv12x training complete.")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv12x")
    parser.add_argument("--data",   default="datasets/merged/data.yaml")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    device = int(args.device) if args.device.isdigit() else args.device
    train(args.data, args.epochs, device, args.resume)
