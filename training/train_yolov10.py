"""
Train YOLOv10 — Mid-range Detector
====================================
YOLOv10 introduces NMS-free training via dual assignments and a consistent
dual-assignment strategy.  We use the medium variant (yolov10m) which offers
a strong accuracy/speed trade-off comparable to YOLOv8x at lower compute.

Parameters: ~16 M (yolov10m)   |   Target latency: ~48 ms/frame

Usage:
    python training/train_yolov10.py [--data datasets/merged/data.yaml]
                                     [--variant m] [--epochs 180]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VARIANTS = {
    "n": ("yolov10n.pt", 32),
    "s": ("yolov10s.pt", 16),
    "m": ("yolov10m.pt", 16),
    "b": ("yolov10b.pt", 12),
    "l": ("yolov10l.pt", 8),
    "x": ("yolov10x.pt", 4),
}


def train(data: str, variant: str, epochs: int, device, resume: bool) -> None:
    from ultralytics import YOLO

    pt, batch = VARIANTS.get(variant, ("yolov10m.pt", 16))
    model = YOLO(pt)

    print("=" * 60)
    print(f"  DeepClean — Training YOLOv10{variant}")
    print(f"  Data    : {data}")
    print(f"  Epochs  : {epochs}")
    print(f"  Device  : {device}")
    print("=" * 60)

    results = model.train(
        data         = data,
        epochs       = epochs,
        imgsz        = 640,
        batch        = batch,
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
        name         = f"yolov10{variant}",
        save_period  = 20,
        patience     = 30,
        exist_ok     = True,
        resume       = resume,
        plots        = True,
        val          = True,
        verbose      = True,
    )

    best = Path(f"training/results/yolov10{variant}/weights/best.pt")
    if best.exists():
        val_model = YOLO(str(best))
        metrics = val_model.val(data=data, imgsz=640, device=device)
        print(f"\n  mAP@0.5        : {metrics.box.map50:.4f}")
        print(f"  mAP@[0.5:0.95] : {metrics.box.map:.4f}")

    print(f"\n[Done] YOLOv10{variant} training complete.")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv10 for underwater detection")
    parser.add_argument("--data",    default="datasets/merged/data.yaml")
    parser.add_argument("--variant", default="m", choices=list(VARIANTS.keys()),
                        help="YOLOv10 variant: n/s/m/b/l/x (default: m)")
    parser.add_argument("--epochs",  type=int, default=180)
    parser.add_argument("--device",  default="0")
    parser.add_argument("--resume",  action="store_true")
    args = parser.parse_args()
    device = int(args.device) if args.device.isdigit() else args.device
    train(args.data, args.variant, args.epochs, device, args.resume)
