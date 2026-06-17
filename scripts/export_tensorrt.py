"""
scripts/export_tensorrt.py — Export YOLOv8n to TensorRT engine.

Requires: Jetson with JetPack 4.6+ (TensorRT 8.x pre-installed)
          OR a desktop with NVIDIA GPU + TensorRT installed.

Usage:
    python scripts/export_tensorrt.py --precision fp16
    python scripts/export_tensorrt.py --precision int8 --calib-data data/calib/

INT8 calibration images should be representative traffic frames (~100–500 images).
"""

import argparse
import os
from pathlib import Path


def export(
    model_path: str,
    precision: str,
    img_size: int,
    calib_data: str | None,
    workspace_gb: int,
):
    from ultralytics import YOLO

    model = YOLO(model_path)

    kwargs = dict(
        format="engine",
        imgsz=img_size,
        workspace=workspace_gb,
        simplify=True,
        verbose=True,
    )

    if precision == "fp16":
        kwargs["half"] = True
        print("[export] Exporting FP16 TensorRT engine...")
    elif precision == "int8":
        kwargs["int8"] = True
        if calib_data:
            kwargs["data"] = calib_data
            print(f"[export] Exporting INT8 TensorRT engine with calibration data: {calib_data}")
        else:
            print("[export] WARNING: No calibration data provided for INT8. Accuracy may degrade.")
            print("[export] Provide --calib-data path/to/images for best results.")
    else:
        print("[export] Exporting FP32 TensorRT engine...")

    engine_path = model.export(**kwargs)
    print(f"\n[export] Engine saved to: {engine_path}")
    print("[export] Copy the engine path to --trt-engine when running main.py")

    # Quick size comparison
    pt_size = os.path.getsize(model_path) / 1e6
    eng_size = os.path.getsize(engine_path) / 1e6
    print(f"\n[export] Model sizes:")
    print(f"  PyTorch .pt : {pt_size:.1f} MB")
    print(f"  TRT engine  : {eng_size:.1f} MB")

    return engine_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/yolov8n.pt", help="Path to .pt weights")
    p.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--calib-data", default=None,
                   help="Path to calibration images directory (INT8 only)")
    p.add_argument("--workspace", type=int, default=2,
                   help="TensorRT workspace size in GB")
    args = p.parse_args()

    if not Path(args.model).exists():
        print(f"[export] Model not found: {args.model}")
        print("[export] Run: python scripts/download_model.py")
        exit(1)

    export(args.model, args.precision, args.img_size, args.calib_data, args.workspace)
