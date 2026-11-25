"""Places in a picture for Omni-Texture's worker: the pixel a place falls on, the four around it, and a picture
read between them. A place is (x, y) with a pixel's middle at its whole number; a picture's pixels are row-major."""

from __future__ import annotations

import numpy as np


def nearest(x: np.ndarray, y: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """The pixel each place falls on: its column and its row."""
    return np.clip(np.round(x), 0, w - 1).astype(np.int64), np.clip(np.round(y), 0, h - 1).astype(np.int64)


def _four(x0: np.ndarray, y0: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The pixel (x0, y0), the one right of it, the one below it and the one below that, kept inside the picture."""
    x0, x1 = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0, y1 = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    return y0 * w + x0, y0 * w + x1, y1 * w + x0, y1 * w + x1


def four_around(x: np.ndarray, y: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The four pixels around each place."""
    return _four(np.floor(x).astype(np.int64), np.floor(y).astype(np.int64), w, h)


def between(flat: np.ndarray, w: int, h: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The picture `flat` ((h * w, channels)) at each place, between the four pixels around it (bilinear)."""
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    a, b, c, d = _four(x0, y0, w, h)
    return (np.take(flat, a, axis=0) * (1 - fx) * (1 - fy) + np.take(flat, b, axis=0) * fx * (1 - fy)
            + np.take(flat, c, axis=0) * (1 - fx) * fy + np.take(flat, d, axis=0) * fx * fy)
