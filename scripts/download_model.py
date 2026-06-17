"""
scripts/download_model.py — Download YOLOv8n weights from Ultralytics.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model yolov8s  # small (more accurate)
"""

import argparse
import os
from pathlib import Path


def download(model_name: str = "yolov8n"):
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    pt_path = models_dir / f"{model_name}.pt"
    if pt_path.exists():
        print(f"[download] {pt_path} already exists — skipping download.")
        return str(pt_path)

    print(f"[download] Downloading {model_name}.pt from Ultralytics...")
    try:
        from ultralytics import YOLO
        model = YOLO(f"{model_name}.pt")  # auto-downloads to CWD
        # Move to models/
        import shutil
        src = Path(f"{model_name}.pt")
        if src.exists():
            shutil.move(str(src), str(pt_path))
        print(f"[download] Saved to {pt_path}")
    except Exception as e:
        print(f"[download] Failed: {e}")
        print("[download] Manual download:")
        print(f"  wget https://github.com/ultralytics/assets/releases/download/v8.2.0/{model_name}.pt -P models/")
        return None

    return str(pt_path)


def export_onnx(pt_path: str, img_size: int = 640):
    """Export to ONNX (for RPi / CPU inference)."""
    from ultralytics import YOLO
    model = YOLO(pt_path)
    out = model.export(format="onnx", imgsz=img_size, simplify=True)
    print(f"[export] ONNX model saved to: {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n", help="Model variant (yolov8n, yolov8s, ...)")
    p.add_argument("--onnx", action="store_true", help="Also export to ONNX")
    p.add_argument("--img-size", type=int, default=640)
    args = p.parse_args()

    pt = download(args.model)
    if pt and args.onnx:
        export_onnx(pt, args.img_size)
