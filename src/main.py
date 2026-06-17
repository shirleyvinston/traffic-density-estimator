"""
main.py — Traffic Density Estimator entry point.

Usage:
    python src/main.py --source 0               # webcam index
    python src/main.py --source video.mp4       # video file
    python src/main.py --source rtsp://...      # IP camera
    python src/main.py --source 0 --no-trt      # disable TensorRT (RPi / CPU)
    python src/main.py --benchmark              # run FPS benchmark and exit
"""

import argparse
import threading
import time
import cv2
import uvicorn

from detector import Detector
from tracker import Tracker
from analytics import Analytics
from heatmap import HeatmapBuilder
from api import create_app, push_frame_data


def parse_args():
    p = argparse.ArgumentParser(description="Smart Traffic Density Estimator")
    p.add_argument("--source", default="0", help="Camera index, video path, or RTSP URL")
    p.add_argument("--model", default="models/yolov8n.pt", help="Path to YOLOv8 weights")
    p.add_argument("--trt-engine", default="models/yolov8n_fp16.engine", help="TensorRT engine path")
    p.add_argument("--no-trt", action="store_true", help="Disable TensorRT, use ONNX/PyTorch")
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--img-size", type=int, default=640, help="Inference image size")
    p.add_argument("--host", default="0.0.0.0", help="Dashboard server host")
    p.add_argument("--port", type=int, default=8000, help="Dashboard server port")
    p.add_argument("--benchmark", action="store_true", help="Run FPS benchmark and exit")
    p.add_argument("--show", action="store_true", help="Show OpenCV preview window")
    return p.parse_args()


def open_source(source: str):
    """Open video source (webcam index, file path, or RTSP URL)."""
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    return cap


def inference_loop(args, detector, tracker, analytics, heatmap_builder):
    """Main inference loop running in a background thread."""
    cap = open_source(args.source)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"[camera] {frame_w}x{frame_h} @ {fps_cap:.1f} FPS source")

    frame_count = 0
    t_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video files; exit on camera disconnect
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        t0 = time.perf_counter()

        # --- Detection ---
        detections = detector.detect(frame)

        # --- Tracking ---
        tracks = tracker.update(detections, frame)

        # --- Analytics ---
        analytics.update(tracks, frame_w, frame_h)

        # --- Heatmap ---
        heatmap_builder.update(tracks, frame_w, frame_h)

        # --- Metrics ---
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        elapsed = t1 - t_start
        frame_count += 1
        pipeline_fps = frame_count / elapsed if elapsed > 0 else 0

        # --- Push to dashboard ---
        payload = {
            "counts": analytics.get_counts(),
            "speeds": analytics.get_speeds(),
            "dwell": analytics.get_dwell_times(),
            "heatmap": heatmap_builder.get_grid(),
            "fps": round(pipeline_fps, 1),
            "latency_ms": round(latency_ms, 1),
            "total_vehicles": analytics.total_seen,
            "tracks": [
                {
                    "id": t.track_id,
                    "cls": t.cls_name,
                    "bbox": t.bbox,
                    "speed": round(t.speed_kmh, 1),
                }
                for t in tracks
            ],
        }
        push_frame_data(payload)

        # --- Optional preview ---
        if args.show:
            vis = draw_tracks(frame, tracks)
            cv2.imshow("Traffic Estimator", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def draw_tracks(frame, tracks):
    """Draw bounding boxes and track IDs on frame."""
    colors = {
        "car": (0, 200, 0),
        "truck": (0, 100, 255),
        "bus": (255, 100, 0),
        "motorcycle": (255, 0, 200),
        "bicycle": (0, 200, 255),
    }
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        color = colors.get(t.cls_name, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"#{t.track_id} {t.cls_name} {t.speed_kmh:.0f}km/h"
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()

    print("[init] Loading detector...")
    use_trt = not args.no_trt
    detector = Detector(
        model_path=args.model,
        trt_engine_path=args.trt_engine if use_trt else None,
        conf=args.conf,
        iou=args.iou,
        img_size=args.img_size,
    )

    if args.benchmark:
        from scripts.benchmark import run_benchmark
        run_benchmark(detector)
        return

    print("[init] Loading tracker...")
    tracker = Tracker()

    print("[init] Loading analytics engine...")
    analytics = Analytics()

    print("[init] Loading heatmap builder...")
    heatmap_builder = HeatmapBuilder(grid_rows=18, grid_cols=32)

    print("[init] Starting FastAPI dashboard server...")
    app = create_app()

    # Run inference in background thread
    inf_thread = threading.Thread(
        target=inference_loop,
        args=(args, detector, tracker, analytics, heatmap_builder),
        daemon=True,
    )
    inf_thread.start()

    print(f"[dashboard] Open http://localhost:{args.port} in your browser")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
