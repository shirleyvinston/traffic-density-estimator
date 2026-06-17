"""
heatmap.py — Spatial vehicle density heatmap (grid-based).

Divides the frame into a grid and accumulates vehicle centre-point
hits per cell, decaying over time so recent traffic is emphasised.
"""

from __future__ import annotations
import numpy as np
from typing import List

from tracker import Track


class HeatmapBuilder:
    """
    Maintains a time-decayed density grid.

    Parameters
    ----------
    grid_rows : int
        Number of rows in the grid.
    grid_cols : int
        Number of columns in the grid.
    decay : float
        Multiplicative decay applied every frame (0.9 ≈ slow decay).
    max_value : float
        Clamp value used to normalise output to [0, 1].
    """

    def __init__(
        self,
        grid_rows: int = 18,
        grid_cols: int = 32,
        decay: float = 0.97,
        max_value: float = 30.0,
    ):
        self.rows = grid_rows
        self.cols = grid_cols
        self.decay = decay
        self.max_value = max_value
        self._grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)

    def update(self, tracks: List[Track], frame_w: int, frame_h: int):
        """Accumulate track centres into the grid for this frame."""
        # Decay existing values
        self._grid *= self.decay

        for t in tracks:
            cx, cy = t.center()
            col = int(cx / frame_w * self.cols)
            row = int(cy / frame_h * self.rows)
            col = max(0, min(col, self.cols - 1))
            row = max(0, min(row, self.rows - 1))
            self._grid[row, col] += 1.0

        # Clamp to max
        np.clip(self._grid, 0, self.max_value, out=self._grid)

    def get_grid(self) -> List[List[float]]:
        """Return grid normalised to [0, 1] as a JSON-serialisable list."""
        normalised = (self._grid / self.max_value).round(3)
        return normalised.tolist()

    def reset(self):
        self._grid[:] = 0.0

    @property
    def raw(self) -> np.ndarray:
        return self._grid.copy()
