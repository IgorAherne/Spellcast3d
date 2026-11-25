"""Several views of one model laid out in one square picture, for Omni-Texture's worker.

Qwen-Image paints every view of a sheet in one picture, so the views agree in colour (painted one by one, a pair of
jeans came out greyer in one view and bluer in the next). The views keep one scale - the model is as large in each -
and are never turned; they are laid out by their outlines, not by their boxes, so two figures with their arms out
fit with one's arm above the other's, and stay a gap apart so that Qwen-Image does not draw them as one object.

pack() works on a coarse grid of the picture (GRID cells a side): an outline covers every cell it touches, a placed
one keeps the cells within the gap, and the next goes to the first free place in reading order, or in column order,
whichever lets the views be larger; tidy rows of their boxes are kept where the outlines gain little. Lengths are
in sides of the picture, so one layout serves a picture of any size. fits() only asks whether a set of views can
share a picture at a given scale: the worker's search for which views go together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

GRID = 128                   # cells along a side of the picture the search works in (8 px of a 1024 picture)
STEPS = 14                   # halvings of the scale's range (to 1 part in 16,000)
ORDERLY = 0.95               # the views' boxes in tidy rows unless their outlines fit larger by more than this


@dataclass
class Layout:
    """Where the views lie in a square picture, in sides of it: `scale` (per outline pixel) and per view the place
    of its outline's top left corner, (x, y); `gap`: what the layout keeps between two views and to the picture's
    edge (the asked one or more: the room left over is shared out)."""
    scale: float
    places: list[tuple[float, float]]
    gap: float


def fewest(count: int, share: Callable[[list[int]], bool], kin: Callable[[int, int], float]) -> list[list[int]]:
    """`count` views shared out among the fewest pictures, every combination of them weighed. share(views): whether
    that set fits one picture - asked only once every set one view smaller does (a set fits only if those do).
    Of the ways with as few pictures, the one that keeps the most kin together (the sum of kin(a, b) over the
    pairs sharing a picture: neighbouring cameras, whose paintings meet on the model), then the one keeping the
    first views together -> the sets, each in the views' order, in the order of their first views."""
    fitting: dict[int, bool] = {}
    best: dict[int, tuple[int, float, tuple[int, ...]]] = {0: (0, 0.0, ())}

    def views_of(mask: int) -> list[int]:
        return [k for k in range(count) if mask >> k & 1]

    def fits_together(mask: int) -> bool:
        if mask not in fitting:
            views = views_of(mask)
            fitting[mask] = len(views) == 1 or (all(fits_together(mask & ~(1 << k)) for k in views) and share(views))
        return fitting[mask]

    def split(mask: int) -> tuple[int, float, tuple[int, ...]]:
        """The best way to share out the views of `mask`: (pictures, less the kin kept together, the sets as masks)."""
        if mask not in best:
            first = mask & -mask                        # its first view, with every choice of the others
            rest = mask ^ first
            found = None
            others = rest
            while True:
                group = others | first
                if fits_together(group):
                    pictures, parted, groups = split(mask ^ group)
                    views = views_of(group)
                    together = sum(kin(a, b) for i, a in enumerate(views) for b in views[i + 1:])
                    option = (pictures + 1, round(parted - together, 9), (group, *groups))
                    if found is None or option < found:
                        found = option
                if not others:
                    break
                others = (others - 1) & rest
            best[mask] = found
        return best[mask]

    return [views_of(group) for group in split((1 << count) - 1)[2]]


def _corners(mask: np.ndarray) -> np.ndarray:
    """(h, w) pixels -> (h + 1, w + 1) pixel corners, each set when a pixel around it is."""
    p = np.pad(mask.astype(bool, copy=False), 1)
    return p[1:, 1:] | p[:-1, 1:] | p[1:, :-1] | p[:-1, :-1]


def _cells(corners: np.ndarray, scale: float) -> np.ndarray:
    """The cells an outline touches at `scale` cells per pixel (at most 1: a pixel is no larger than a cell)."""
    out = corners
    for axis in (0, 1):
        at = np.floor(np.arange(out.shape[axis]) * scale).astype(np.int64)
        starts = np.flatnonzero(np.r_[True, at[1:] != at[:-1]])
        out = np.logical_or.reduceat(out, starts, axis=axis)
    return out


def _grown(mask: np.ndarray, ry: int, rx: int) -> np.ndarray:
    """Set where any cell of the (2 ry + 1) by (2 rx + 1) box around it is set."""
    h, w = mask.shape
    p = np.pad(mask, ((ry, ry), (rx, rx)))
    rows = p[:, :w].copy()
    for d in range(1, 2 * rx + 1):
        rows |= p[:, d:d + w]
    out = rows[:h].copy()
    for d in range(1, 2 * ry + 1):
        out |= rows[d:d + h]
    return out


def _place(cells: list[np.ndarray], order: list[int], columns: bool, gap: tuple[int, int],
           n: int = GRID) -> list[tuple[int, int]] | None:
    """Each outline's cells put down in `order` at the first place free of the ones before it and their gap, in
    reading order (or column by column); `gap`: the cells kept between two outlines and to the picture's edge,
    (down, across) -> per outline (row, column) of its top left cell, or None when one does not fit."""
    taken = np.zeros((n, n), dtype=bool)
    spots: list[tuple[int, int] | None] = [None] * len(cells)
    for k in order:
        c = cells[k]
        h, w = c.shape
        ys, xs = n - 2 * gap[0] - h + 1, n - 2 * gap[1] - w + 1
        if ys <= 0 or xs <= 0:
            return None
        y = x = 0
        if taken.any():
            # how many of its cells each place would share with the taken ones (a correlation, by FFT)
            both = np.fft.irfft2(np.fft.rfft2(taken.astype(np.float32)) * np.conj(np.fft.rfft2(c, s=(n, n))),
                                 s=(n, n))
            free = both[gap[0]:gap[0] + ys, gap[1]:gap[1] + xs] < 0.5
            if not free.any():
                return None
            if columns:
                x, y = divmod(int(np.argmax(free.T)), ys)
            else:
                y, x = divmod(int(np.argmax(free)), xs)
        y, x = y + gap[0], x + gap[1]
        spots[k] = (y, x)
        here = np.zeros((n, n), dtype=bool)
        here[y:y + h, x:x + w] = c
        taken |= _grown(here, gap[0], gap[1])
    return spots  # type: ignore[return-value]


def _ways(outlines: list[np.ndarray]) -> list[tuple[list[np.ndarray], list[int], bool]]:
    """The ways a layout is tried, each (the shapes as pixel corners, the order they are put down in, column by
    column?): the views' boxes in rows first (the tidy layout), then their outlines, in their own order and
    largest first, row by row and column by column."""
    corners = [_corners(o) for o in outlines]
    given = list(range(len(outlines)))
    by_size = sorted(given, key=lambda k: -int(outlines[k].sum()))
    ways = [([np.ones_like(c) for c in corners], given, False), (corners, given, False), (corners, given, True)]
    if by_size != given:
        ways += [(corners, by_size, False), (corners, by_size, True)]
    return ways


def fits(outlines: list[np.ndarray], gap: float, scale: float) -> bool:
    """Whether the outlines can be laid out at `scale` (sides of the picture per outline pixel), `gap` apart and
    from the picture's edge: one try a way, where pack() looks for the largest scale."""
    space = max(1, int(np.ceil(gap * GRID)))
    cells = scale * GRID
    if cells > 1.0 or sum(int(o.sum()) for o in outlines) * cells ** 2 > (GRID - 2 * space) ** 2:
        return False                                            # (a pixel larger than a cell; more than the room)
    return any(_place([_cells(c, cells) for c in shapes], order, columns, (space, space)) is not None
               for shapes, order, columns in _ways(outlines))


def pack(outlines: list[np.ndarray], gap: float, limit: float | None = None,
         least: float | None = None) -> Layout | None:
    """The views' outlines (bool pictures at one scale, each cut to its box; at the scale found no pixel of them
    may be larger than a cell of the grid, so draw them finer than the grid) laid out in a square picture as large
    as they fit, `gap` (of the picture's side) apart and from its edge, but no larger than `limit` (sides per
    outline pixel) and, where fits() says they fit at `least`, no smaller than that. Tried: _ways(), the tidy
    rows kept unless the outlines fit larger by more than ORDERLY allows. The room left over is shared out: the
    gap is widened, across and then down, as far as the same layout still fits, and the whole is centred. None:
    nothing to lay out."""
    if not outlines:
        return None
    ways = _ways(outlines)
    space = max(1, int(np.ceil(gap * GRID)))
    longest = max(max(o.shape) for o in outlines)
    top = min((GRID - 2 * space) / longest, 1.0)                # in cells per outline pixel
    if limit is not None:
        top = min(top, limit * GRID)

    def fits(scale: float, way: tuple, room: tuple[int, int] = (space, space)):
        return _place([_cells(c, scale) for c in way[0]], way[1], way[2], room)

    best: list[float] = []
    for way in ways:
        if fits(top, way) is not None:
            best.append(top)
            continue
        lo, hi = 0.0, top
        if least is not None and 0.0 < least * GRID < top and fits(least * GRID, way) is not None:
            lo = least * GRID                                   # (the search below is not sure to come back to it)
        for _ in range(STEPS):
            mid = (lo + hi) / 2.0
            if fits(mid, way) is not None:
                lo = mid
            else:
                hi = mid
        best.append(lo)
    largest = max(best)
    if largest <= 0.0:
        return None
    # the tidy rows unless the outlines win clearly; of those, the first that is as good as the best
    pick = 0 if best[0] >= ORDERLY * largest else next(i for i in range(1, len(ways)) if best[i] >= 0.995 * largest)
    scale, way = best[pick], ways[pick]
    room = [space, space]
    for axis in (1, 0):                                         # the gap widened: across, then down
        lo, hi = space, GRID // 2
        while lo < hi:
            mid = (lo + hi + 1) // 2
            trial = (mid, room[1]) if axis == 0 else (room[0], mid)
            if fits(scale, way, trial) is not None:
                lo = mid
            else:
                hi = mid - 1
        room[axis] = lo
    spots = fits(scale, way, (room[0], room[1]))
    assert spots is not None
    # centred: by the outlines' own extents (a cell's worth of slack at most)
    x0 = min(x for _, x in spots)
    y0 = min(y for y, _ in spots)
    x1 = max(x + o.shape[1] * scale for (_, x), o in zip(spots, outlines))
    y1 = max(y + o.shape[0] * scale for (y, _), o in zip(spots, outlines))
    dx, dy = (GRID - x1 - x0) / 2.0, (GRID - y1 - y0) / 2.0
    return Layout(scale / GRID, [((x + dx) / GRID, (y + dy) / GRID) for y, x in spots], min(room) / GRID)
