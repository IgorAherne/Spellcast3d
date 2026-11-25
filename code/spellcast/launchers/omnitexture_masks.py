"""Masks for Omni-Texture's worker: grown, shrunk and cleaned; the snaps in a camera's depth; and how much of a
mask's shape a picture still shows."""

from __future__ import annotations

import numpy as np

SPECK = 4                    # px: specks and holes this small leave a mask (clean)
# A snap: a jump in a camera's depth between two neighbouring pixels of the model, one part standing in front of
# another. Qwen-Image's painting sits a pixel or three off the model, so right beside a snap it shows the other
# part's colours: no texel reads a pixel within SNAP_REACH of one.
SNAP = 0.01                  # of the model's diagonal: the jump in depth that is a snap
SNAP_REACH = 2               # px to either side of a snap that no texel reads (one pixel left a faint ghost)
# mask_left: the rings beside the mask's outline that are compared (px from it), the window their colours are
# averaged in along it, and the share of the outline (%) that may show a step: things do end at a mask's edge.
LEFT_NEAR, LEFT_FAR, LEFT_WINDOW, LEFT_SHARE = 3, 9, 12, 10
LEFT_RING_PIXELS = 20        # ring pixels a window needs on each side of the outline to be compared
LEFT_OUTLINE_PIXELS = 30     # outline pixels so compared, under which a mask is too small to tell


def _square(mask: np.ndarray, r: int, op: np.ufunc) -> np.ndarray:
    """`op` (logical or / and) over each (2r + 1)-square window, outside the picture unset: a row pass, then a
    column pass."""
    h, w = mask.shape
    p = np.pad(mask.astype(bool, copy=False), r)
    rows = p[:, :w].copy()
    for d in range(1, 2 * r + 1):
        op(rows, p[:, d:d + w], out=rows)
    out = rows[:h].copy()
    for d in range(1, 2 * r + 1):
        op(out, rows[d:d + h], out=out)
    return out


def grow(mask: np.ndarray, r: int) -> np.ndarray:
    """Set where any pixel of the (2r + 1)-square window around it is set."""
    return _square(mask, r, np.logical_or) if r > 0 else mask.copy()


def shrink(mask: np.ndarray, r: int) -> np.ndarray:
    """Set where every pixel of the (2r + 1)-square window around it is set (outside the picture is not)."""
    return _square(mask, r, np.logical_and) if r > 0 else mask.copy()


def clean(mask: np.ndarray, r: int) -> np.ndarray:
    """The mask without specks or holes smaller than about 2r + 1 pixels (an opening, then a closing)."""
    return shrink(grow(grow(shrink(mask, r), r), r), r)


def closing(mask: np.ndarray, r: int) -> np.ndarray:
    """The mask with its holes and gaps up to 2r pixels wide set (grown, then shrunk; outside the picture counts
    as set): what was set stays set, and a part unset wider than that keeps its outline."""
    return shrink(grow(np.pad(mask, r, constant_values=True), r), r)[r:-r, r:-r] if r > 0 else mask.copy()


def _window_sum(values: np.ndarray, r: int) -> np.ndarray:
    """The sum over each (2r + 1)-square window of `values` ((h, w) or (h, w, n)); outside the picture counts 0."""
    pad = ((r + 1, r), (r + 1, r)) + ((0, 0),) * (values.ndim - 2)
    total = np.cumsum(np.cumsum(np.pad(values.astype(np.float64), pad), axis=0), axis=1)
    k = 2 * r + 1
    return total[k:, k:] - total[:-k, k:] - total[k:, :-k] + total[:-k, :-k]


def mask_left(picture: np.ndarray, where: np.ndarray, model: np.ndarray) -> float:
    """How much of a mask's shape is still in the picture Qwen-Image made of it: the step in colour (levels, the
    largest of the channels) across the outline of `where` ((h, w): the pixels that were masked) between the
    picture ((h, w, 3)) just inside it and just outside it on the `model`, at the LEFT_SHARE of the outline where
    that step is smallest. A picture that has the surface restored shows the outline nowhere, or where things
    really end at it; one with a patch of one colour laid where the mask was shows it all the way round. 0.0 for
    a mask too small to tell."""
    inside = shrink(where, LEFT_NEAR) & ~shrink(where, LEFT_FAR)
    outside = grow(where, LEFT_FAR) & ~grow(where, LEFT_NEAR) & model
    edge = where & ~shrink(where, 1)
    within, without = _window_sum(inside, LEFT_WINDOW), _window_sum(outside, LEFT_WINDOW)
    told = edge & (within >= LEFT_RING_PIXELS) & (without >= LEFT_RING_PIXELS)
    if told.sum() < LEFT_OUTLINE_PIXELS:
        return 0.0
    values = picture.astype(np.float64)
    step = np.abs(_window_sum(values * inside[..., None], LEFT_WINDOW)[told] / within[told][:, None]
                  - _window_sum(values * outside[..., None], LEFT_WINDOW)[told] / without[told][:, None]).max(axis=1)
    return float(np.percentile(step, LEFT_SHARE))


def snaps(depth: np.ndarray, limit: float, reach: int = SNAP_REACH) -> np.ndarray:
    """The pixels within `reach` of a snap: a step in depth larger than `limit` between two neighbouring pixels
    that both see the model (inf: no model - the model's outline against nothing is no snap: there is no surface
    behind it to take its colours), and more than twice the steps next to it on either side: a break, not a
    surface that runs away from the camera steeply (a wall seen nearly edge on steps as far from pixel to pixel
    all along). The pixels on both sides of the step count."""
    out = np.zeros(depth.shape, dtype=bool)
    for line, flags in ((depth, out), (depth.T, out.T)):       # along the rows, then along the columns
        with np.errstate(invalid="ignore"):                     # (inf - inf: no step)
            step = np.abs(line[:, 1:] - line[:, :-1])
        step[~np.isfinite(step)] = 0.0                          # (no model on a side: nothing to compare)
        beside = np.zeros_like(step)
        beside[:, 1:] = step[:, :-1]
        beside[:, :-1] = np.maximum(beside[:, :-1], step[:, 1:])
        snap = (step > limit) & (step > 2.0 * beside)
        flags[:, 1:] |= snap
        flags[:, :-1] |= snap
    return grow(out, reach - 1) if reach > 1 else out
