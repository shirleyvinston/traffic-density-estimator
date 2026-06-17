"""
tracker.py — Lightweight ByteTrack-style multi-object tracker.

ByteTrack reference: Zhang et al. 2022 (https://arxiv.org/abs/2110.06864)

This implementation covers the core ideas:
  - High-confidence detections matched via IoU (Hungarian)
  - Low-confidence detections matched to existing tracks as second pass
  - Kalman filter for motion prediction between frames
  - Track lifecycle: Tentative → Confirmed → Lost → Deleted
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment

from detector import Detection

# Kalman matrices (constant velocity model in image space)
_F = np.eye(8)       # State transition
for i in range(4):
    _F[i, i + 4] = 1  # pos += vel

_H = np.eye(4, 8)    # Measurement (observe position only)

_Q = np.diag([1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01]) * 0.1   # Process noise
_R = np.eye(4) * 1.0                                           # Measurement noise


@dataclass
class Track:
    track_id: int
    cls_id: int
    cls_name: str
    bbox: tuple                  # (x1, y1, x2, y2) — latest measurement
    conf: float

    # Kalman state [cx, cy, w, h, vcx, vcy, vw, vh]
    kf_x: np.ndarray = field(default_factory=lambda: np.zeros((8, 1)))
    kf_P: np.ndarray = field(default_factory=lambda: np.eye(8) * 10)

    age: int = 0                 # frames since creation
    hits: int = 1                # confirmed detections
    time_since_update: int = 0   # frames since last matched
    state: str = "tentative"     # tentative | confirmed | lost

    # Analytics
    center_history: list = field(default_factory=list)
    speed_kmh: float = 0.0
    dwell_frames: int = 0

    @property
    def is_confirmed(self) -> bool:
        return self.state == "confirmed"

    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def to_xyah(self) -> np.ndarray:
        """Convert bbox to [cx, cy, aspect_ratio, height]."""
        x1, y1, x2, y2 = self.bbox
        w, h = x2 - x1, y2 - y1
        return np.array([[x1 + w / 2], [y1 + h / 2], [w / max(h, 1e-6)], [h]])

    def predict(self):
        """Kalman predict step."""
        self.kf_x = _F @ self.kf_x
        self.kf_P = _F @ self.kf_P @ _F.T + _Q
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Detection):
        """Kalman update step with a matched detection."""
        z = np.array([[
            (det.bbox[0] + det.bbox[2]) / 2,
            (det.bbox[1] + det.bbox[3]) / 2,
            (det.bbox[2] - det.bbox[0]) / max(det.bbox[3] - det.bbox[1], 1e-6),
            det.bbox[3] - det.bbox[1],
        ]]).T

        S = _H @ self.kf_P @ _H.T + _R
        K = self.kf_P @ _H.T @ np.linalg.inv(S)
        self.kf_x = self.kf_x + K @ (z - _H @ self.kf_x)
        self.kf_P = (np.eye(8) - K @ _H) @ self.kf_P

        self.bbox = det.bbox
        self.conf = det.conf
        self.hits += 1
        self.time_since_update = 0
        self.dwell_frames += 1

        cx, cy = self.center()
        self.center_history.append((cx, cy))
        if len(self.center_history) > 30:
            self.center_history.pop(0)

        if self.hits >= 3:
            self.state = "confirmed"


def _iou(a: tuple, b: tuple) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(union, 1e-6)


def _iou_matrix(tracks: list, dets: list) -> np.ndarray:
    mat = np.zeros((len(tracks), len(dets)))
    for i, t in enumerate(tracks):
        for j, d in enumerate(dets):
            mat[i, j] = _iou(t.bbox, d.bbox)
    return mat


def _hungarian(iou_mat: np.ndarray, threshold: float):
    """Return matched (ti, di) pairs where IoU >= threshold."""
    if iou_mat.size == 0:
        return [], list(range(iou_mat.shape[0])), list(range(iou_mat.shape[1]))
    cost = 1 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)
    matched, unmatched_t, unmatched_d = [], [], []
    matched_t, matched_d = set(), set()
    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= threshold:
            matched.append((r, c))
            matched_t.add(r)
            matched_d.add(c)
    unmatched_t = [i for i in range(iou_mat.shape[0]) if i not in matched_t]
    unmatched_d = [j for j in range(iou_mat.shape[1]) if j not in matched_d]
    return matched, unmatched_t, unmatched_d


class Tracker:
    """
    ByteTrack-style tracker.

    Parameters
    ----------
    max_age : int
        Frames a track survives without being matched before deletion.
    min_hits : int
        Minimum detections before a track is confirmed.
    high_conf_thresh : float
        Confidence threshold for the high-confidence association pass.
    low_conf_thresh : float
        Confidence threshold for the low-confidence association pass.
    iou_thresh_high : float
        IoU threshold for high-confidence matching.
    iou_thresh_low : float
        IoU threshold for low-confidence matching.
    px_per_meter : float
        Calibration: pixels per real-world meter (used for speed estimation).
    fps : float
        Expected pipeline FPS (used for speed estimation).
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        high_conf_thresh: float = 0.5,
        low_conf_thresh: float = 0.1,
        iou_thresh_high: float = 0.5,
        iou_thresh_low: float = 0.3,
        px_per_meter: float = 20.0,
        fps: float = 15.0,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.high_conf_thresh = high_conf_thresh
        self.low_conf_thresh = low_conf_thresh
        self.iou_thresh_high = iou_thresh_high
        self.iou_thresh_low = iou_thresh_low
        self.px_per_meter = px_per_meter
        self.fps = fps

        self._tracks: List[Track] = []
        self._next_id: int = 1

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """
        Run one tracker step.

        Returns list of currently active confirmed tracks.
        """
        # Predict all existing tracks
        for t in self._tracks:
            t.predict()

        high_dets = [d for d in detections if d.conf >= self.high_conf_thresh]
        low_dets = [d for d in detections if self.low_conf_thresh <= d.conf < self.high_conf_thresh]

        active = [t for t in self._tracks if t.time_since_update <= 1]
        lost = [t for t in self._tracks if t.time_since_update > 1]

        # Pass 1: match high-conf dets to active tracks
        iou_high = _iou_matrix(active, high_dets)
        matched_h, unmatched_t_h, unmatched_d_h = _hungarian(iou_high, self.iou_thresh_high)

        for ti, di in matched_h:
            active[ti].update(high_dets[di])

        # Pass 2: match low-conf dets to unmatched active tracks
        remaining_tracks = [active[i] for i in unmatched_t_h]
        iou_low = _iou_matrix(remaining_tracks, low_dets)
        matched_l, unmatched_t_l, _ = _hungarian(iou_low, self.iou_thresh_low)

        for ti, di in matched_l:
            remaining_tracks[ti].update(low_dets[di])

        # Initiate new tracks for unmatched high-conf dets
        for di in unmatched_d_h:
            d = high_dets[di]
            t = Track(
                track_id=self._next_id,
                cls_id=d.cls_id,
                cls_name=d.cls_name,
                bbox=d.bbox,
                conf=d.conf,
            )
            # Initialise Kalman state
            cx = (d.bbox[0] + d.bbox[2]) / 2
            cy = (d.bbox[1] + d.bbox[3]) / 2
            w = d.bbox[2] - d.bbox[0]
            h = d.bbox[3] - d.bbox[1]
            t.kf_x[:4] = np.array([[cx], [cy], [w / max(h, 1e-6)], [h]])
            t.center_history.append((cx, cy))
            self._tracks.append(t)
            self._next_id += 1

        # Delete stale tracks
        self._tracks = [
            t for t in self._tracks if t.time_since_update <= self.max_age
        ]

        # Update speed estimates
        self._estimate_speeds()

        # Return only confirmed tracks
        return [t for t in self._tracks if t.is_confirmed]

    def _estimate_speeds(self):
        """
        Estimate speed in km/h from recent center history.
        Uses pixel displacement over last N frames.
        """
        for t in self._tracks:
            if len(t.center_history) < 5:
                t.speed_kmh = 0.0
                continue
            pts = t.center_history[-5:]
            total_px = sum(
                np.sqrt((pts[i][0] - pts[i-1][0])**2 + (pts[i][1] - pts[i-1][1])**2)
                for i in range(1, len(pts))
            )
            meters_per_frame = total_px / (len(pts) - 1) / max(self.px_per_meter, 1e-6)
            speed_mps = meters_per_frame * self.fps
            t.speed_kmh = speed_mps * 3.6

    @property
    def tracks(self) -> List[Track]:
        return [t for t in self._tracks if t.is_confirmed]
