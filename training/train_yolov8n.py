"""
Train YOLOv8n — Lightweight Detector
=====================================
Parameters: 3.2 M   |   Target latency: ~18 ms/frame   |   Energy: ~15 W

Usage:
    python training/train_yolov8n.py [--data datasets/merged/data.yaml]
                                     [--epochs 180] [--device 0]
"""

import argparse
import sys
from pathlib import Path

# ── Allow running from repo root ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def train(data: str, epochs: int, device: str | int, resume: bool) -> None:
    from ultralytics import YOLO

    cfg   = Path(__file__).parent / "configs" / "yolov8n.yaml"
    model = YOLO("yolov8n.pt")       # pretrained COCO weights

    print("=" * 60)
    print("  DeepClean — Training YOLOv8n (lightweight)")
    print(f"  Data   : {data}")
    print(f"  Epochs : {epochs}")
    print(f"  Device : {device}")
    print("=" * 60)

    results = model.train(
        data           = data,
        epochs         = epochs,
        imgsz          = 640,
        batch          = 32,
        device         = device,
        optimizer      = "AdamW",
        lr0            = 0.001,
        lrf            = 0.01,
        weight_decay   = 0.0005,
        warmup_epochs  = 3,
        mosaic         = 1.0,
        mixup          = 0.1,
        copy_paste     = 0.1,
        flipud         = 0.1,
        fliplr         = 0.5,
        hsv_h          = 0.015,
        hsv_s          = 0.7,
        hsv_v          = 0.4,
        project        = "training/results",
        name           = "yolov8n",
        save_period    = 20,
        patience       = 30,
        exist_ok       = True,
        resume         = resume,
        plots          = True,
        val            = True,
        verbose        = True,
    )

    # ── Post-training validation ───────────────────────────────────────────────
    best_weights = Path("training/results/yolov8n/weights/best.pt")
    if best_weights.exists():
        print(f"\n[Eval] Validating best checkpoint: {best_weights}")
        val_model = YOLO(str(best_weights))
        metrics = val_model.val(data=data, imgsz=640, device=device, verbose=True)
        print(f"\n  mAP@0.5        : {metrics.box.map50:.4f}")
        print(f"  mAP@[0.5:0.95] : {metrics.box.map:.4f}")
        print(f"  Precision      : {metrics.box.mp:.4f}")
        print(f"  Recall         : {metrics.box.mr:.4f}")

    print("\n[Done] YOLOv8n training complete.")
    print(f"  Weights: training/results/yolov8n/weights/best.pt")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n for underwater detection")
    parser.add_argument("--data",    default="datasets/merged/data.yaml")
    parser.add_argument("--epochs",  type=int,   default=180)
    parser.add_argument("--device",  default="0", help="GPU id or 'cpu'")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    device = int(args.device) if args.device.isdigit() else args.device
    train(data=args.data, epochs=args.epochs, device=device, resume=args.resume)
