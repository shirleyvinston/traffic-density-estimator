"""
detector.py — YOLOv8-nano inference wrapper.

Supports three backends (auto-selected):
  1. TensorRT engine  (Jetson, fastest)
  2. ONNX Runtime     (RPi / CPU, no CUDA)
  3. PyTorch (native ultralytics)  (dev / Mac)
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import cv2

# COCO vehicle class IDs we care about
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel coords)
    conf: float
    cls_id: int
    cls_name: str


class Detector:
    """
    Multi-backend YOLOv8 detector.

    Priority: TensorRT > ONNX Runtime > Ultralytics PyTorch
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        trt_engine_path: Optional[str] = None,
        conf: float = 0.35,
        iou: float = 0.50,
        img_size: int = 640,
    ):
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.backend = "none"
        self._model = None

        if trt_engine_path and os.path.exists(trt_engine_path):
            self._load_tensorrt(trt_engine_path)
        elif model_path.endswith(".onnx") and os.path.exists(model_path):
            self._load_onnx(model_path)
        elif os.path.exists(model_path):
            self._load_ultralytics(model_path)
        else:
            raise FileNotFoundError(
                f"No model found at {model_path}. "
                "Run: python scripts/download_model.py"
            )

        print(f"[detector] Backend: {self.backend}")

    # ------------------------------------------------------------------
    # Backend loaders
    # ------------------------------------------------------------------

    def _load_tensorrt(self, engine_path: str):
        """Load a serialised TensorRT engine (Jetson only)."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401

            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            with open(engine_path, "rb") as f:
                engine_data = f.read()
            engine = runtime.deserialize_cuda_engine(engine_data)
            self._trt_context = engine.create_execution_context()
            self._trt_engine = engine
            self._cuda = cuda
            self.backend = "tensorrt"
            print(f"[detector] TensorRT engine loaded: {engine_path}")
        except ImportError:
            print("[detector] TensorRT not available — falling back to ultralytics")
            self._load_ultralytics("models/yolov8n.pt")

    def _load_onnx(self, model_path: str):
        """Load YOLOv8 ONNX model via ONNX Runtime (CPU/GPU)."""
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )
        self._ort_session = ort.InferenceSession(model_path, providers=providers)
        self._ort_input_name = self._ort_session.get_inputs()[0].name
        self.backend = "onnx"
        print(f"[detector] ONNX Runtime loaded: {model_path}")

    def _load_ultralytics(self, model_path: str):
        """Load via ultralytics (PyTorch). Works everywhere."""
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self.backend = "ultralytics"
        print(f"[detector] Ultralytics model loaded: {model_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a BGR frame. Returns list of Detection objects."""
        if self.backend == "ultralytics":
            return self._detect_ultralytics(frame)
        elif self.backend == "onnx":
            return self._detect_onnx(frame)
        elif self.backend == "tensorrt":
            return self._detect_tensorrt(frame)
        return []

    # ------------------------------------------------------------------
    # Backend inference
    # ------------------------------------------------------------------

    def _detect_ultralytics(self, frame: np.ndarray) -> List[Detection]:
        results = self._model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            verbose=False,
            classes=list(VEHICLE_CLASS_IDS.keys()),
        )[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASS_IDS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    conf=float(box.conf[0]),
                    cls_id=cls_id,
                    cls_name=VEHICLE_CLASS_IDS[cls_id],
                )
            )
        return detections

    def _detect_onnx(self, frame: np.ndarray) -> List[Detection]:
        """ONNX Runtime inference with manual NMS."""
        blob, scale, (pad_w, pad_h) = self._preprocess(frame)
        outputs = self._ort_session.run(None, {self._ort_input_name: blob})
        return self._postprocess(outputs[0], frame.shape, scale, pad_w, pad_h)

    def _detect_tensorrt(self, frame: np.ndarray) -> List[Detection]:
        """
        TensorRT inference.
        Assumes engine was exported with dynamic batch=1 and
        output shape [1, 84, 8400] (YOLOv8 standard).
        """
        import pycuda.driver as cuda

        blob, scale, (pad_w, pad_h) = self._preprocess(frame)
        blob = blob.astype(np.float32)

        engine = self._trt_engine
        context = self._trt_context

        # Allocate I/O buffers
        bindings = []
        outputs_trt = []
        for binding in engine:
            size = (
                abs(engine.get_binding_volume(engine.get_binding_index(binding)))
                * blob.itemsize
            )
            device_mem = cuda.mem_alloc(size)
            bindings.append(int(device_mem))
            if engine.binding_is_input(engine.get_binding_index(binding)):
                cuda.memcpy_htod(device_mem, blob)
            else:
                host_mem = cuda.pagelocked_empty(
                    engine.get_binding_volume(engine.get_binding_index(binding)),
                    dtype=np.float32,
                )
                outputs_trt.append((host_mem, device_mem))

        stream = cuda.Stream()
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()

        for host_mem, device_mem in outputs_trt:
            cuda.memcpy_dtoh(host_mem, device_mem)

        raw = outputs_trt[0][0].reshape(1, 84, -1)
        return self._postprocess(raw, frame.shape, scale, pad_w, pad_h)

    # ------------------------------------------------------------------
    # Pre / post processing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray):
        """Letterbox resize + normalise to [0,1] float32 NCHW."""
        img_h, img_w = frame.shape[:2]
        s = self.img_size
        scale = min(s / img_h, s / img_w)
        new_h, new_w = int(img_h * scale), int(img_w * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = (s - new_h) // 2
        pad_w = (s - new_w) // 2
        padded = cv2.copyMakeBorder(
            resized, pad_h, s - new_h - pad_h, pad_w, s - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=114,
        )
        blob = padded[:, :, ::-1].transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob)
        return blob, scale, (pad_w, pad_h)

    def _postprocess(
        self, raw: np.ndarray, frame_shape, scale: float, pad_w: int, pad_h: int
    ) -> List[Detection]:
        """
        Parse YOLOv8 output tensor [1, 84, 8400] and apply NMS.
        Output format: [cx, cy, w, h, cls0_conf, ..., cls79_conf]
        """
        preds = raw[0].T  # (8400, 84)
        boxes = preds[:, :4]
        cls_scores = preds[:, 4:]

        cls_ids = cls_scores.argmax(axis=1)
        confs = cls_scores.max(axis=1)

        mask = confs > self.conf
        boxes, cls_ids, confs = boxes[mask], cls_ids[mask], confs[mask]

        if len(boxes) == 0:
            return []

        # cx, cy, w, h → x1, y1, x2, y2 (in letterboxed space)
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        # Remove padding and rescale to original frame
        frame_h, frame_w = frame_shape[:2]
        x1 = ((x1 - pad_w) / scale).clip(0, frame_w)
        y1 = ((y1 - pad_h) / scale).clip(0, frame_h)
        x2 = ((x2 - pad_w) / scale).clip(0, frame_w)
        y2 = ((y2 - pad_h) / scale).clip(0, frame_h)

        # OpenCV NMS
        bboxes_cv = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(bboxes_cv, confs.tolist(), self.conf, self.iou)

        detections = []
        for i in (indices.flatten() if len(indices) else []):
            cls_id = int(cls_ids[i])
            if cls_id not in VEHICLE_CLASS_IDS:
                continue
            detections.append(
                Detection(
                    bbox=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                    conf=float(confs[i]),
                    cls_id=cls_id,
                    cls_name=VEHICLE_CLASS_IDS[cls_id],
                )
            )
        return detections

    def warmup(self, iterations: int = 10):
        """Run N dummy frames to warm up TensorRT / CUDA kernels."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.detect(dummy)
        print(f"[detector] Warmup done ({iterations} iterations)")
