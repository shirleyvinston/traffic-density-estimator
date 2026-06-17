"""
scripts/benchmark.py — Benchmark detection FPS before and after TensorRT.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --trt-engine models/yolov8n_fp16.engine
    python scripts/benchmark.py --iterations 200 --img-size 640

Run this on your Jetson to generate the numbers you'll show in interviews.
"""

import argparse
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def benchmark_backend(detector, name: str, iterations: int, img_size: int):
    dummy = np.random.randint(0, 255, (img_size, img_size * 16 // 9, 3), dtype=np.uint8)

    # Warmup
    print(f"  Warming up {name}...")
    for _ in range(min(20, iterations // 5)):
        detector.detect(dummy)

    # Timed run
    latencies = []
    t_total = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter()
        detector.detect(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)
    elapsed = time.perf_counter() - t_total

    fps = iterations / elapsed
    avg_lat = np.mean(latencies)
    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)

    print(f"\n  ── {name} ─────────────────────────")
    print(f"  FPS          : {fps:.1f}")
    print(f"  Latency avg  : {avg_lat:.1f} ms")
    print(f"  Latency p50  : {p50_lat:.1f} ms")
    print(f"  Latency p95  : {p95_lat:.1f} ms")
    print(f"  Latency p99  : {p99_lat:.1f} ms")
    print(f"  Total frames : {iterations} in {elapsed:.2f}s")

    return fps, avg_lat


def run_benchmark(detector=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/yolov8n.pt")
    p.add_argument("--trt-engine", default=None)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--compare", action="store_true",
                   help="Benchmark both PyTorch and TRT and print comparison table")
    args = p.parse_args()

    from detector import Detector

    print("\n🔬 Traffic Estimator — FPS Benchmark")
    print("=" * 45)

    results = {}

    if detector:
        # Called from main.py --benchmark
        fps, lat = benchmark_backend(detector, detector.backend.upper(), args.iterations, args.img_size)
        results[detector.backend] = (fps, lat)
    else:
        # Standalone: benchmark ultralytics baseline first
        print("\n[1/2] PyTorch baseline (no TRT):")
        base = Detector(model_path=args.model, trt_engine_path=None)
        fps_base, lat_base = benchmark_backend(base, "PyTorch (FP32)", args.iterations, args.img_size)
        results["pytorch"] = (fps_base, lat_base)

        if args.trt_engine and os.path.exists(args.trt_engine):
            print("\n[2/2] TensorRT engine:")
            trt = Detector(model_path=args.model, trt_engine_path=args.trt_engine)
            fps_trt, lat_trt = benchmark_backend(trt, "TensorRT", args.iterations, args.img_size)
            results["tensorrt"] = (fps_trt, lat_trt)

            print("\n  ── Speedup summary ─────────────────")
            print(f"  FPS  speedup : {fps_trt / max(fps_base, 0.01):.2f}×")
            print(f"  Latency drop : {lat_base:.1f}ms → {lat_trt:.1f}ms "
                  f"({((lat_base - lat_trt) / lat_base * 100):.0f}% faster)")

    print("\n✅ Benchmark complete. Save these numbers for your README!")


if __name__ == "__main__":
    run_benchmark()
