"""
scripts/train_custom.py — Fine-tune YOLOv8n on a custom traffic dataset.

Dataset options:
  A) Roboflow Universe — download via API key (recommended)
  B) Local YOLO-format dataset directory

Usage:
    # With Roboflow API:
    python scripts/train_custom.py --roboflow-key YOUR_KEY \
        --workspace myworkspace --project traffic-detection --version 3

    # With local dataset:
    python scripts/train_custom.py --data data/traffic/data.yaml --epochs 50

After training, the best weights are at runs/detect/train/weights/best.pt
Export to TensorRT with: python scripts/export_tensorrt.py --model runs/detect/train/weights/best.pt
"""

import argparse
import os
from pathlib import Path


def download_roboflow_dataset(api_key: str, workspace: str, project: str, version: int) -> str:
    """Download dataset from Roboflow and return path to data.yaml."""
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("Install roboflow: pip install roboflow")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov8", location="data/roboflow")
    data_yaml = str(Path("data/roboflow") / "data.yaml")
    print(f"[train] Dataset downloaded to: data/roboflow")
    return data_yaml


def train(data_yaml: str, epochs: int, img_size: int, batch: int, base_model: str, resume: bool):
    from ultralytics import YOLO

    model = YOLO(base_model)
    print(f"[train] Starting training: {base_model} on {data_yaml}")
    print(f"[train] Epochs: {epochs}, img_size: {img_size}, batch: {batch}")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        resume=resume,
        project="runs/detect",
        name="traffic_custom",
        patience=20,          # Early stopping
        save=True,
        save_period=10,
        device=0,             # GPU 0; use 'cpu' for Pi
        workers=4,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        hsv_h=0.015,          # Augmentation
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        verbose=True,
    )

    best_weights = "runs/detect/traffic_custom/weights/best.pt"
    print(f"\n[train] Training complete!")
    print(f"[train] Best weights: {best_weights}")
    print(f"[train] mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.3f}")

    return best_weights


def validate(weights_path: str, data_yaml: str, img_size: int):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, imgsz=img_size)
    print(f"\n[val] Validation results:")
    print(f"  mAP50     : {metrics.box.map50:.3f}")
    print(f"  mAP50-95  : {metrics.box.map:.3f}")
    print(f"  Precision  : {metrics.box.mp:.3f}")
    print(f"  Recall     : {metrics.box.mr:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8n on traffic data")
    p.add_argument("--data", default=None, help="Path to data.yaml (local dataset)")
    p.add_argument("--roboflow-key", default=None, help="Roboflow API key")
    p.add_argument("--workspace", default=None)
    p.add_argument("--project", default=None)
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--batch", type=int, default=16,
                   help="Batch size (use 4-8 on Jetson Nano)")
    p.add_argument("--base-model", default="models/yolov8n.pt",
                   help="Starting weights (yolov8n.pt = pretrained on COCO)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last checkpoint")
    p.add_argument("--validate-only", default=None,
                   help="Skip training, just validate these weights")
    args = p.parse_args()

    data_yaml = args.data

    if args.roboflow_key:
        if not all([args.workspace, args.project]):
            print("[train] --workspace and --project required with --roboflow-key")
            exit(1)
        data_yaml = download_roboflow_dataset(
            args.roboflow_key, args.workspace, args.project, args.version
        )

    if not data_yaml:
        print("[train] Provide either --data or --roboflow-key")
        exit(1)

    if args.validate_only:
        validate(args.validate_only, data_yaml, args.img_size)
    else:
        best = train(data_yaml, args.epochs, args.img_size, args.batch, args.base_model, args.resume)
        validate(best, data_yaml, args.img_size)
        print(f"\n[train] Next: python scripts/export_tensorrt.py --model {best} --precision fp16")
