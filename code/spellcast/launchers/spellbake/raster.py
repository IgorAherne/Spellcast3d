"""UV-space rasteriser: which low-poly triangle, at which barycentrics, every sample of the
output map lands on (design section 4.4, step L8).

The map is R x R texels; each texel holds S x S stratified samples. Sample (i, j) of
texel (column c, row r) sits at x = c + (i + 1/2) / S, y = r + (j + 1/2) / S in texel
units, i.e. u = x / R, v = 1 - y / R, so row 0 is the top of the image (v = 1). The work
is done on the supersampled grid, where the sample centres are the half-integers
(c * S + i + 1/2, r * S + j + 1/2) and a triangle's corners are X = u * R * S,
Y = (1 - v) * R * S.

Coverage uses edge functions with a top-left style tie rule, evaluated in one canonical
direction per edge so that two triangles sharing an edge compute bit-identical values of
opposite sign: a sample on a shared edge, or on a shared vertex inside a chart, belongs
to exactly one triangle. The rule: a triangle (made counter-clockwise first) owns the
points of its edge a -> b when b - a points into y > 0, or along +x when horizontal.

With ``ring`` > 0 the raster is conservative, with unclamped barycentrics: the surface
continues linearly past the chart edge, so bilinear filtering at a chart border blends two
correctly baked texels instead of a baked one and a fill colour.
- A texel with only some of its S x S samples covered gets the missing ones from the
  nearest triangle that covers the others. Its value is then the mean over the whole
  texel, like its neighbours', instead of leaning towards the covered part (which showed
  as a 1.6 LSB colour step across UV seams at S = 2, against 0.75 at S = 1).
- A texel with no sample within ``ring`` texels of some triangle gets one sample at its
  centre from the nearest triangle: for a planar extrapolation that equals the mean of
  S x S samples.

Everything is produced in bands of ``band_rows`` output rows, so memory stays bounded at
4096 x 4096 with 3 x 3 samples.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .tangents import DEGENERATE_UV_AREA

# Candidate samples tested per vectorised step.
_CHUNK = 1 << 21


@dataclass(frozen=True)
class SampleBand:
    """The samples of output rows [row0, row0 + rows)."""

    row0: int
    rows: int
    tri: np.ndarray      # (n,) int32 triangle index
    bary: np.ndarray     # (n, 3) float64 barycentrics of the triangle's vertices 0, 1, 2
    texel: np.ndarray    # (n,) int32 row-major index into the whole R x R map
    sub: np.ndarray      # (n,) int16 sample index j * S + i inside its texel (0 at a texel centre)
    ring: np.ndarray     # (n,) bool: extrapolated sample (a texel not, or not fully, covered)


def texel_centre_uv(column: np.ndarray, row: np.ndarray, resolution: int) -> np.ndarray:
    """v-up UV of texel centres: u = (c + 1/2) / R, v = 1 - (r + 1/2) / R."""
    column = np.asarray(column, dtype=np.float64)
    row = np.asarray(row, dtype=np.float64)
    return np.stack([(column + 0.5) / resolution, 1.0 - (row + 0.5) / resolution], axis=-1)


class _Triangles:
    """Per-triangle setup in some pixel space (x right, y down), made counter-clockwise.

    A clockwise (mirrored) triangle is set up with its corners 1 and 2 swapped;
    ``bary`` swaps them back, so barycentrics always refer to the input's corner order.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, keep: np.ndarray) -> None:
        self.index = np.flatnonzero(keep)
        x, y = x[self.index], y[self.index]
        area = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (y[:, 1] - y[:, 0]) * (x[:, 2] - x[:, 0])
        self.clockwise = area < 0
        order = np.where(self.clockwise[:, None], [0, 2, 1], [0, 1, 2])
        rows = np.arange(len(self.index))[:, None]
        self.x, self.y = x[rows, order], y[rows, order]
        self.area = np.abs(area)
        # Edge k runs from corner k+1 to k+2 (it is opposite corner k): the edge function is
        # cross(b - a, p - a), positive inside. It is evaluated from the lexicographically
        # smaller endpoint, and negated when the triangle runs the other way, so the two
        # triangles on an edge get bit-identical values of opposite sign.
        ax, ay = self.x[:, [1, 2, 0]], self.y[:, [1, 2, 0]]
        bx, by = self.x[:, [2, 0, 1]], self.y[:, [2, 0, 1]]
        forward = (ax < bx) | ((ax == bx) & (ay < by))
        self.sign = np.where(forward, 1.0, -1.0)
        self.cx = np.where(forward, ax, bx)
        self.cy = np.where(forward, ay, by)
        self.dx = np.where(forward, bx - ax, ax - bx)
        self.dy = np.where(forward, by - ay, ay - by)
        self.owns = ((by - ay) > 0) | (((by - ay) == 0) & ((bx - ax) > 0))

    def edges(self, rows: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """(n, 3) edge functions of triangles ``rows`` at points (px, py), positive inside."""
        return self.sign[rows] * (
            self.dx[rows] * (py[:, None] - self.cy[rows]) - self.dy[rows] * (px[:, None] - self.cx[rows])
        )

    def bary(self, rows: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Barycentrics of the input's corners 0, 1, 2 from the edge values ``e``."""
        b = e / self.area[rows, None]
        b[:, 0] = 1.0 - b[:, 1] - b[:, 2]
        swap = self.clockwise[rows]
        b[swap, 1], b[swap, 2] = b[swap, 2], b[swap, 1]
        return b


def _covered(tris: _Triangles, samples: int, lo: int, hi: int) -> tuple[np.ndarray, ...]:
    """Covered samples of sample rows [lo, hi): (setup row, x index, y index, bary)."""
    y_min = tris.y.min(axis=1)
    y_max = tris.y.max(axis=1)
    first = np.maximum(np.ceil(y_min - 0.5), lo).astype(np.int64)
    last = np.minimum(np.floor(y_max - 0.5), hi - 1).astype(np.int64)
    rows = np.flatnonzero(last >= first)
    span = last[rows] - first[rows] + 1
    # One (triangle, sample row) pair per scanline the triangle crosses.
    pair_tri = np.repeat(rows, span)
    pair_y = np.repeat(first[rows] - np.cumsum(span) + span, span) + np.arange(int(span.sum()))
    yc = pair_y + 0.5

    # Each edge is a linear constraint k * x + m >= 0 on the scanline; intersect them.
    k = -tris.sign[pair_tri] * tris.dy[pair_tri]
    m = tris.sign[pair_tri] * (tris.dx[pair_tri] * (yc[:, None] - tris.cy[pair_tri])
                               + tris.dy[pair_tri] * tris.cx[pair_tri])
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = -m / k
    x_min, x_max = tris.x.min(axis=1)[pair_tri], tris.x.max(axis=1)[pair_tri]
    x_lo = np.clip(np.where(k > 0, bound, -np.inf).max(axis=1), x_min, x_max)
    x_hi = np.clip(np.where(k < 0, bound, np.inf).min(axis=1), x_min, x_max)
    empty = ((k == 0) & (m < 0)).any(axis=1) | (x_lo > x_hi + 1.0)
    # One sample of slack on each side: the exact test below decides.
    c0 = np.maximum(np.floor(x_lo - 0.5) - 1, 0).astype(np.int64)
    c1 = np.minimum(np.ceil(x_hi - 0.5) + 1, samples - 1).astype(np.int64)
    count = np.where(empty, 0, np.maximum(c1 - c0 + 1, 0))

    # The same linear form gives the exact coverage test: k and m of the two triangles on
    # an edge are exact negatives of each other, so their k * x + m are too.
    owns = tris.owns[pair_tri]
    out_row, out_x, out_y, out_bary = [], [], [], []
    ends = np.cumsum(count)
    start = 0
    while start < len(count):
        base = ends[start - 1] if start else 0
        stop = max(int(np.searchsorted(ends, base + _CHUNK, side="right")), start + 1)
        n = count[start:stop]
        pair = np.repeat(np.arange(start, stop), n)
        sx = c0[pair] + (np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n))
        e = np.repeat(k[start:stop], n, axis=0) * (sx + 0.5)[:, None] + np.repeat(m[start:stop], n, axis=0)
        inside = ((e > 0) | ((e == 0) & np.repeat(owns[start:stop], n, axis=0))).all(axis=1)
        pair = pair[inside]
        out_row.append(pair_tri[pair])
        out_x.append(sx[inside])
        out_y.append(pair_y[pair])
        out_bary.append(tris.bary(pair_tri[pair], e[inside]))
        start = stop
    if not out_row:
        none = np.zeros(0, dtype=np.int64)
        return none, none, none, np.zeros((0, 3))
    return (np.concatenate(out_row), np.concatenate(out_x), np.concatenate(out_y),
            np.concatenate(out_bary))


def _complete(
    tris: _Triangles, resolution: int, supersample: int, row0: int, rows: int,
    tri: np.ndarray, sx: np.ndarray, sy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Samples missing from partly covered texels of output rows [row0, row0 + rows), each
    taken from the nearest of the triangles that cover the rest of its texel. ``tri``,
    ``sx``, ``sy`` are the covered samples (setup rows, sample-grid indices).
    Returns (setup row, sx, sy, bary) of the added samples."""
    s = supersample
    grid = np.zeros((rows * s, resolution * s), dtype=bool)
    grid[sy - row0 * s, sx] = True
    count = grid.reshape(rows, s, resolution, s).sum(axis=(1, 3))
    partial = (count > 0) & (count < s * s)
    if not partial.any():
        none = np.zeros(0, dtype=np.int64)
        return none, none, none, np.zeros((0, 3))
    want = ~grid & np.repeat(np.repeat(partial, s, axis=0), s, axis=1)
    msy, msx = np.nonzero(want)
    msy += row0 * s
    miss_texel = (msy // s) * resolution + msx // s
    # Candidates: the distinct triangles covering some sample of the same texel.
    texel = (sy // s) * resolution + sx // s
    sel = partial.reshape(-1)[texel - row0 * resolution]
    width = len(tris.index)
    pairs = np.unique(texel[sel].astype(np.int64) * width + tri[sel])
    pair_texel, pair_tri = pairs // width, pairs % width
    first = np.searchsorted(pair_texel, miss_texel, side="left")
    n = np.searchsorted(pair_texel, miss_texel, side="right") - first
    owner = np.repeat(np.arange(len(miss_texel)), n)
    cand = pair_tri[np.repeat(first - np.cumsum(n) + n, n) + np.arange(int(n.sum()))]
    px, py = msx[owner] + 0.5, msy[owner] + 0.5
    inside = (tris.edges(cand, px, py) >= 0).all(axis=1)
    dist = np.where(inside, 0.0, _distance_to_edges(tris, cand, px, py))
    order = np.lexsort((cand, dist, owner))
    keep = order[np.r_[True, owner[order][1:] != owner[order][:-1]]]
    cand, px, py = cand[keep], px[keep], py[keep]
    return cand, msx, msy, tris.bary(cand, tris.edges(cand, px, py))


def _ring(
    tris: _Triangles, resolution: int, width: float, lo: int, hi: int, covered: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest triangle and its unclamped barycentrics for every uncovered texel of rows
    [lo, hi) whose centre is within ``width`` texels of a triangle (texel pixel space).

    ``covered`` is the (hi - lo, R) mask of texels that already have samples.
    Returns (setup row, texel index, bary)."""
    x0 = np.maximum(np.ceil(tris.x.min(axis=1) - width - 0.5), 0).astype(np.int64)
    x1 = np.minimum(np.floor(tris.x.max(axis=1) + width - 0.5), resolution - 1).astype(np.int64)
    y0 = np.maximum(np.ceil(tris.y.min(axis=1) - width - 0.5), lo).astype(np.int64)
    y1 = np.minimum(np.floor(tris.y.max(axis=1) + width - 0.5), hi - 1).astype(np.int64)
    rows = np.flatnonzero((x1 >= x0) & (y1 >= y0))
    nx = x1[rows] - x0[rows] + 1
    count = nx * (y1[rows] - y0[rows] + 1)

    # Nearest triangle per texel so far (ties: the lower triangle index, for a stable result).
    best_dist = np.full(covered.size, np.inf)
    best_tri = np.full(covered.size, -1, dtype=np.int64)
    ends = np.cumsum(count)
    start = 0
    while start < len(rows):
        base = ends[start - 1] if start else 0
        stop = max(int(np.searchsorted(ends, base + _CHUNK, side="right")), start + 1)
        n = count[start:stop]
        pick = np.repeat(np.arange(start, stop), n)
        offset = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        tri = rows[pick]
        tx = x0[tri] + offset % nx[pick]
        ty = y0[tri] + offset // nx[pick]
        free = ~covered[ty - lo, tx]
        tri, tx, ty = tri[free], tx[free], ty[free]
        px, py = tx + 0.5, ty + 0.5
        inside = (tris.edges(tri, px, py) >= 0).all(axis=1)
        dist = np.zeros(len(tri))
        out = np.flatnonzero(~inside)
        if out.size:
            dist[out] = _distance_to_edges(tris, tri[out], px[out], py[out])
        near = np.flatnonzero(dist <= width)
        cell = (ty[near] - lo) * resolution + tx[near]
        order = np.lexsort((tri[near], dist[near], cell))
        order = order[np.r_[True, cell[order][1:] != cell[order][:-1]]] if order.size else order
        cell, d, t = cell[order], dist[near][order], tri[near][order]
        better = (d < best_dist[cell]) | ((d == best_dist[cell]) & (t < best_tri[cell]))
        best_dist[cell[better]], best_tri[cell[better]] = d[better], t[better]
        start = stop
    cell = np.flatnonzero(best_tri >= 0)
    tri = best_tri[cell]
    texel = lo * resolution + cell
    e = tris.edges(tri, texel % resolution + 0.5, texel // resolution + 0.5)
    return tri, texel, tris.bary(tri, e)


def _distance_to_edges(tris: _Triangles, rows: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Distance from each point to the boundary of its triangle."""
    best = np.full(len(rows), np.inf)
    for k in range(3):
        ax, ay = tris.x[rows, (k + 1) % 3], tris.y[rows, (k + 1) % 3]
        ex, ey = tris.x[rows, (k + 2) % 3] - ax, tris.y[rows, (k + 2) % 3] - ay
        length2 = np.maximum(ex * ex + ey * ey, 1e-300)
        s = np.clip(((px - ax) * ex + (py - ay) * ey) / length2, 0.0, 1.0)
        best = np.minimum(best, np.hypot(px - ax - s * ex, py - ay - s * ey))
    return best


def _boundary_triangles(faces: np.ndarray) -> np.ndarray:
    """Triangles with an edge no other triangle shares: the only ones that can be nearest
    to a texel outside every triangle (the nearest point lies on the union's boundary)."""
    a = faces.reshape(-1)
    b = np.roll(faces, -1, axis=1).reshape(-1)
    key = np.minimum(a, b) * (int(faces.max()) + 1) + np.maximum(a, b)
    _, inverse, count = np.unique(key, return_inverse=True, return_counts=True)
    return (count[inverse.reshape(-1)] == 1).reshape(-1, 3).any(axis=1)


def sample_bands(
    uv: np.ndarray,
    faces: np.ndarray,
    resolution: int,
    supersample: int = 1,
    *,
    ring: float = 0.0,
    band_rows: int = 256,
) -> Iterator[SampleBand]:
    """Every sample of an R x R map (R = ``resolution``, S = ``supersample``), band by band.

    ``uv`` are v-up coordinates indexed by ``faces``. Triangles with no UV area
    (``tangents.DEGENERATE_UV_AREA``) produce no samples. Bands come top row first and
    never overlap; a texel's samples all arrive in the band holding its row.
    """
    uv = np.asarray(uv, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    samples = resolution * supersample
    t = uv[faces]
    t21, t31 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    keep = np.abs(t21[:, 0] * t31[:, 1] - t21[:, 1] * t31[:, 0]) >= DEGENERATE_UV_AREA
    keep &= np.isfinite(t).all(axis=(1, 2))
    fine = _Triangles(t[..., 0] * samples, (1.0 - t[..., 1]) * samples, keep)
    coarse = (_Triangles(t[..., 0] * resolution, (1.0 - t[..., 1]) * resolution, keep & _boundary_triangles(faces))
              if ring > 0 else None)

    for row0 in range(0, resolution, band_rows):
        rows = min(band_rows, resolution - row0)
        tri, sx, sy, bary = _covered(fine, samples, row0 * supersample, (row0 + rows) * supersample)
        is_ring = np.zeros(len(tri), dtype=bool)
        if coarse is not None and supersample > 1 and len(tri):
            c_tri, c_sx, c_sy, c_bary = _complete(fine, resolution, supersample, row0, rows, tri, sx, sy)
            tri, sx, sy = (np.concatenate(pair) for pair in ((tri, c_tri), (sx, c_sx), (sy, c_sy)))
            bary = np.concatenate([bary, c_bary])
            is_ring = np.concatenate([is_ring, np.ones(len(c_tri), dtype=bool)])
        texel = (sy // supersample) * resolution + sx // supersample
        sub = (sy % supersample) * supersample + sx % supersample
        tri = fine.index[tri]
        if coarse is not None:
            covered = np.zeros((rows, resolution), dtype=bool)
            covered.reshape(-1)[texel - row0 * resolution] = True
            r_tri, r_texel, r_bary = _ring(coarse, resolution, ring, row0, row0 + rows, covered)
            tri = np.concatenate([tri, coarse.index[r_tri]])
            bary = np.concatenate([bary, r_bary])
            texel = np.concatenate([texel, r_texel])
            sub = np.concatenate([sub, np.zeros(len(r_tri), dtype=sub.dtype)])
            is_ring = np.concatenate([is_ring, np.ones(len(r_tri), dtype=bool)])
        yield SampleBand(
            row0=row0,
            rows=rows,
            tri=tri.astype(np.int32),
            bary=bary,
            texel=texel.astype(np.int32),
            sub=sub.astype(np.int16),
            ring=is_ring,
        )
