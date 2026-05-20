"""
Train YOLOv12n — Lightweight next-gen detector
================================================
YOLOv12 introduces attention-centric feature extraction with area attention
modules, enabling better long-range context without sacrificing speed.

Variants used in DeepClean:
  YOLOv12n  — lightweight baseline companion to YOLOv8n
  YOLOv12x  — highest-capacity model in the comparison

Usage:
    python training/train_yolov12n.py [--data datasets/merged/data.yaml]
                                      [--epochs 180] [--device 0]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def train(data: str, epochs: int, device, resume: bool) -> None:
    from ultralytics import YOLO

    # YOLOv12 uses the same Ultralytics API; checkpoint names follow the pattern
    # yolov12n.pt — will be auto-downloaded if available in the Ultralytics hub.
    # Fall back to YOLOv8n architecture weights for fine-tuning if not available.
    try:
        model = YOLO("yolov12n.pt")
    except Exception:
        print("[WARN] yolov12n.pt not found in hub — initialising from yolov8n.pt")
        model = YOLO("yolov8n.pt")

    print("=" * 60)
    print("  DeepClean — Training YOLOv12n (lightweight, attention-based)")
    print(f"  Data   : {data}")
    print(f"  Epochs : {epochs}")
    print(f"  Device : {device}")
    print("=" * 60)

    model.train(
        data         = data,
        epochs       = epochs,
        imgsz        = 640,
        batch        = 32,
        device       = device,
        optimizer    = "AdamW",
        lr0          = 0.001,
        lrf          = 0.01,
        weight_decay = 0.0005,
        warmup_epochs= 3,
        mosaic       = 1.0,
        mixup        = 0.1,
        copy_paste   = 0.1,
        flipud       = 0.1,
        fliplr       = 0.5,
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        project      = "training/results",
        name         = "yolov12n",
        save_period  = 20,
        patience     = 30,
        exist_ok     = True,
        resume       = resume,
        plots        = True,
        val          = True,
        verbose      = True,
    )

    best = Path("training/results/yolov12n/weights/best.pt")
    if best.exists():
        metrics = YOLO(str(best)).val(data=data, imgsz=640, device=device)
        print(f"\n  mAP@0.5        : {metrics.box.map50:.4f}")
        print(f"  mAP@[0.5:0.95] : {metrics.box.map:.4f}")

    print("\n[Done] YOLOv12n training complete.")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv12n")
    parser.add_argument("--data",   default="datasets/merged/data.yaml")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    device = int(args.device) if args.device.isdigit() else args.device
    train(args.data, args.epochs, device, args.resume)
