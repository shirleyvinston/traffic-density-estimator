"""
analytics.py — Per-class vehicle counts, speed, and dwell time analytics.
"""

from __future__ import annotations
from collections import defaultdict, deque
from typing import Dict, List

from tracker import Track

CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]


class Analytics:
    """
    Accumulates per-frame tracking results into dashboard-ready metrics.

    Metrics:
      - counts         : current confirmed vehicles per class
      - speeds         : rolling average speed per class (km/h)
      - dwell_times    : dwell time histogram (seconds) per class
      - total_seen     : unique vehicle IDs seen since start
    """

    def __init__(self, speed_window: int = 30):
        self._speed_window = speed_window
        self._speed_buffers: Dict[str, deque] = {
            cls: deque(maxlen=speed_window) for cls in CLASSES
        }
        self._seen_ids: set = set()
        self._dwell_times: Dict[str, List[float]] = defaultdict(list)
        self._current_counts: Dict[str, int] = {cls: 0 for cls in CLASSES}
        self._fps: float = 15.0  # updated externally if needed

    def update(self, tracks: List[Track], frame_w: int, frame_h: int):
        """Called once per frame with the current confirmed track list."""
        # Reset current frame counts
        self._current_counts = {cls: 0 for cls in CLASSES}

        for t in tracks:
            if t.cls_name not in self._current_counts:
                continue

            self._current_counts[t.cls_name] += 1
            self._seen_ids.add(t.track_id)

            # Speed
            if t.speed_kmh > 0:
                self._speed_buffers[t.cls_name].append(t.speed_kmh)

            # Dwell time (only record when a track is marked as leaving)
            # We record it as a float (seconds)
            if t.dwell_frames > 0:
                dwell_sec = t.dwell_frames / max(self._fps, 1)
                # Update rolling record — we just track the latest dwell for active tracks
                self._dwell_times[t.cls_name] = self._dwell_times.get(t.cls_name, [])

    def record_leaving_track(self, track: Track):
        """Call when a track is deleted — records its total dwell time."""
        dwell_sec = track.dwell_frames / max(self._fps, 1)
        self._dwell_times[track.cls_name].append(round(dwell_sec, 2))
        # Keep last 200 dwell records per class
        self._dwell_times[track.cls_name] = self._dwell_times[track.cls_name][-200:]

    def get_counts(self) -> Dict[str, int]:
        return dict(self._current_counts)

    def get_speeds(self) -> Dict[str, float]:
        return {
            cls: round(sum(buf) / len(buf), 1) if buf else 0.0
            for cls, buf in self._speed_buffers.items()
        }

    def get_dwell_times(self) -> Dict[str, List[float]]:
        """Return last N dwell time records per class (for histogram)."""
        return {cls: times[-50:] for cls, times in self._dwell_times.items()}

    @property
    def total_seen(self) -> int:
        return len(self._seen_ids)

    def set_fps(self, fps: float):
        self._fps = max(fps, 1.0)

    def summary(self) -> Dict:
        return {
            "counts": self.get_counts(),
            "speeds": self.get_speeds(),
            "total_seen": self.total_seen,
        }
