"""
tests/test_detector.py — Unit tests for the Detector class.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detector import Detector, Detection, VEHICLE_CLASS_IDS


class TestDetectorPreprocessing:
    """Test preprocessing logic without requiring model weights."""

    def _make_detector(self):
        """Create a detector instance with a dummy model path for testing preprocessing only."""
        d = object.__new__(Detector)
        d.conf = 0.35
        d.iou = 0.5
        d.img_size = 640
        d.backend = "test"
        d._model = None
        return d

    def test_preprocess_produces_correct_shape(self):
        d = self._make_detector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        blob, scale, (pad_w, pad_h) = d._preprocess(frame)
        assert blob.shape == (1, 3, 640, 640), f"Expected (1,3,640,640), got {blob.shape}"

    def test_preprocess_normalises_range(self):
        d = self._make_detector()
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        blob, _, _ = d._preprocess(frame)
        assert blob.max() <= 1.0 + 1e-6
        assert blob.min() >= 0.0 - 1e-6

    def test_preprocess_portrait_frame(self):
        d = self._make_detector()
        frame = np.zeros((1080, 720, 3), dtype=np.uint8)
        blob, scale, _ = d._preprocess(frame)
        assert blob.shape == (1, 3, 640, 640)
        assert abs(scale - 640 / 1080) < 1e-4

    def test_preprocess_square_frame(self):
        d = self._make_detector()
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        blob, scale, (pw, ph) = d._preprocess(frame)
        assert scale == pytest.approx(1.0, abs=1e-4)
        assert pw == 0
        assert ph == 0


class TestDetection:
    def test_detection_dataclass(self):
        d = Detection(bbox=(10, 20, 100, 150), conf=0.85, cls_id=2, cls_name="car")
        assert d.cls_name == "car"
        assert d.conf == pytest.approx(0.85)

    def test_vehicle_class_ids_coverage(self):
        assert 2 in VEHICLE_CLASS_IDS  # car
        assert 7 in VEHICLE_CLASS_IDS  # truck
        assert 5 in VEHICLE_CLASS_IDS  # bus
