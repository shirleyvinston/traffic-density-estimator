"""
tests/test_tracker.py — Unit tests for ByteTrack tracker.
tests/test_analytics.py — Unit tests for Analytics engine.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detector import Detection
from tracker import Tracker, Track, _iou
from analytics import Analytics
from heatmap import HeatmapBuilder


# ─── Tracker tests ────────────────────────────────────────────────────────────

class TestIoU:
    def test_perfect_overlap(self):
        box = (0, 0, 100, 100)
        assert _iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _iou((0, 0, 50, 50), (60, 60, 110, 110)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        iou = _iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert 0.0 < iou < 1.0

    def test_symmetry(self):
        a = (10, 20, 80, 90)
        b = (40, 30, 120, 100)
        assert _iou(a, b) == pytest.approx(_iou(b, a))


class TestTracker:
    def _make_det(self, x1=100, y1=100, x2=200, y2=200, cls="car", conf=0.9):
        return Detection(bbox=(x1, y1, x2, y2), conf=conf, cls_id=2, cls_name=cls)

    def test_new_track_created(self):
        tracker = Tracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [self._make_det()]
        tracker.update(dets, frame)
        assert len(tracker._tracks) == 1

    def test_track_confirmed_after_hits(self):
        tracker = Tracker(min_hits=3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = self._make_det()
        for _ in range(4):
            tracker.update([det], frame)
        confirmed = [t for t in tracker._tracks if t.is_confirmed]
        assert len(confirmed) >= 1

    def test_track_deleted_after_max_age(self):
        tracker = Tracker(max_age=5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker.update([self._make_det()], frame)
        # No detections for max_age + 1 frames
        for _ in range(7):
            tracker.update([], frame)
        assert len(tracker._tracks) == 0

    def test_unique_track_ids(self):
        tracker = Tracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [
            self._make_det(0, 0, 100, 100),
            self._make_det(200, 200, 300, 300),
            self._make_det(400, 0, 500, 100),
        ]
        tracker.update(dets, frame)
        ids = [t.track_id for t in tracker._tracks]
        assert len(ids) == len(set(ids))

    def test_speed_estimation_zero_for_new_track(self):
        tracker = Tracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker.update([self._make_det()], frame)
        assert tracker._tracks[0].speed_kmh == pytest.approx(0.0, abs=0.1)


# ─── Analytics tests ──────────────────────────────────────────────────────────

class TestAnalytics:
    def _make_track(self, cls="car", tid=1, speed=30.0, dwell=10):
        t = Track(track_id=tid, cls_id=2, cls_name=cls, bbox=(0, 0, 100, 100), conf=0.9)
        t.state = "confirmed"
        t.speed_kmh = speed
        t.dwell_frames = dwell
        return t

    def test_counts_correct(self):
        analytics = Analytics()
        tracks = [self._make_track("car", 1), self._make_track("car", 2), self._make_track("truck", 3)]
        analytics.update(tracks, 640, 480)
        counts = analytics.get_counts()
        assert counts["car"] == 2
        assert counts["truck"] == 1
        assert counts["bus"] == 0

    def test_total_seen_grows(self):
        analytics = Analytics()
        analytics.update([self._make_track("car", 1)], 640, 480)
        analytics.update([self._make_track("car", 1), self._make_track("truck", 2)], 640, 480)
        assert analytics.total_seen == 2

    def test_speeds_averaged(self):
        analytics = Analytics()
        tracks = [self._make_track("car", 1, speed=60.0), self._make_track("car", 2, speed=40.0)]
        analytics.update(tracks, 640, 480)
        # Speed buffer accumulates per-track speeds
        assert analytics.get_speeds()["car"] > 0


# ─── Heatmap tests ────────────────────────────────────────────────────────────

class TestHeatmapBuilder:
    def _make_track(self, cx, cy):
        t = Track(track_id=1, cls_id=2, cls_name="car", bbox=(cx-20, cy-20, cx+20, cy+20), conf=0.9)
        t.state = "confirmed"
        return t

    def test_grid_dimensions(self):
        hm = HeatmapBuilder(grid_rows=10, grid_cols=20)
        grid = hm.get_grid()
        assert len(grid) == 10
        assert len(grid[0]) == 20

    def test_grid_values_in_range(self):
        hm = HeatmapBuilder()
        tracks = [self._make_track(320, 240)]
        hm.update(tracks, 640, 480)
        grid = hm.get_grid()
        flat = [v for row in grid for v in row]
        assert all(0.0 <= v <= 1.0 for v in flat)

    def test_accumulation_increases_density(self):
        hm = HeatmapBuilder(grid_rows=10, grid_cols=10, decay=1.0)
        tracks = [self._make_track(320, 240)]
        for _ in range(10):
            hm.update(tracks, 640, 480)
        grid = hm.get_grid()
        max_val = max(v for row in grid for v in row)
        assert max_val > 0.1

    def test_reset_clears_grid(self):
        hm = HeatmapBuilder()
        tracks = [self._make_track(100, 100)]
        hm.update(tracks, 640, 480)
        hm.reset()
        grid = hm.get_grid()
        assert all(v == 0.0 for row in grid for v in row)
