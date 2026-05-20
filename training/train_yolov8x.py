"""
Train YOLOv8x — Heavyweight Detector
======================================
Parameters: 66.2 M   |   Target latency: ~56 ms/frame   |   Energy: ~50 W

Usage:
    python training/train_yolov8x.py [--data datasets/merged/data.yaml]
                                     [--epochs 180] [--device 0]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def train(data: str, epochs: int, device: str | int, resume: bool) -> None:
    from ultralytics import YOLO

    model = YOLO("yolov8x.pt")

    print("=" * 60)
    print("  DeepClean — Training YOLOv8x (heavyweight)")
    print(f"  Data   : {data}")
    print(f"  Epochs : {epochs}")
    print(f"  Device : {device}")
    print("  NOTE: Requires ~16 GB VRAM with batch=8")
    print("=" * 60)

    results = model.train(
        data           = data,
        epochs         = epochs,
        imgsz          = 640,
        batch          = 8,            # smaller batch for large model
        device         = device,
        optimizer      = "AdamW",
        lr0            = 0.0005,       # lower LR for large model
        lrf            = 0.01,
        weight_decay   = 0.0005,
        warmup_epochs  = 5,
        mosaic         = 1.0,
        mixup          = 0.15,
        copy_paste     = 0.1,
        flipud         = 0.1,
        fliplr         = 0.5,
        hsv_h          = 0.015,
        hsv_s          = 0.7,
        hsv_v          = 0.4,
        project        = "training/results",
        name           = "yolov8x",
        save_period    = 20,
        patience       = 30,
        exist_ok       = True,
        resume         = resume,
        plots          = True,
        val            = True,
        verbose        = True,
    )

    best_weights = Path("training/results/yolov8x/weights/best.pt")
    if best_weights.exists():
        print(f"\n[Eval] Validating best checkpoint: {best_weights}")
        val_model = YOLO(str(best_weights))
        metrics = val_model.val(data=data, imgsz=640, device=device, verbose=True)
        print(f"\n  mAP@0.5        : {metrics.box.map50:.4f}")
        print(f"  mAP@[0.5:0.95] : {metrics.box.map:.4f}")
        print(f"  Precision      : {metrics.box.mp:.4f}")
        print(f"  Recall         : {metrics.box.mr:.4f}")

    print("\n[Done] YOLOv8x training complete.")
    print(f"  Weights: training/results/yolov8x/weights/best.pt")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8x for underwater detection")
    parser.add_argument("--data",   default="datasets/merged/data.yaml")
    parser.add_argument("--epochs", type=int,   default=180)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = int(args.device) if args.device.isdigit() else args.device
    train(data=args.data, epochs=args.epochs, device=device, resume=args.resume)
