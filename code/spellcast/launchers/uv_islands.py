"""Small UV islands joined onto a neighbouring island (numpy only), so an atlas has fewer crumbs.

xatlas grows a chart only while the chart still lies flat in one projection, so a triangle folded sharply against
its neighbours - a sliver along a crease, common in decimated and retopologized meshes - stays a chart of its own.
Here each small chart is laid onto its largest neighbour instead, triangle by triangle: a triangle is unfolded flat
across an edge it shares with what is laid already, at the neighbour's own texel scale along that edge, and its
third corner takes the island's own place of that vertex when the unfolded one lands near it (closing the notch
between them). A chart is joined whole or not at all: if one of its triangles would overlap the island or turn
over, it stays as the unwrapper made it and may try its next neighbour in the next pass. Every small chart is laid
at once, a ring of triangles per round. The joined islands keep their old places, so the caller packs the atlas
again (xatlas with the input UVs, in packing_units).

Used by the 3D stage's unwrap (pixal3d_glb) and by Instant Meshes' (imesh_server: Retopo's three methods).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

#: Charts under this share of the surface are joined onto a neighbour where they fit.
SMALL_SHARE = 1e-3
#: A corner unfolded this close to where the island has that vertex already (a share of the edge it was
#: unfolded across) takes the island's place instead.
SNAP = 0.25
#: Passes go on while one joins anything (a chart that grew in one pass is a neighbour to try in the next, so
#: chains of crumbs join a link per pass), MAX_PASSES at most.
MAX_PASSES = 16
#: The joined atlas comes back this many units across: xatlas, packing it again from these UVs, leaves a face whose
#: UV area is under FLT_EPSILON out of its island and charts it anew - in a 0..1 atlas of 300k triangles, exactly
#: the slivers just joined.
PACK_SPAN = 4096.0
#: A new triangle whose box covers more grid cells than this is not laid (looking around it would cost too much).
MAX_CELLS = 64


def chart_labels(tris: np.ndarray, count: int) -> np.ndarray:
    """A chart id per triangle, 0..n-1: the triangles that share a split vertex are one chart."""
    parent = np.arange(count)
    while True:
        low = parent[tris].min(axis=1)
        nxt = parent.copy()
        for k in range(3):
            np.minimum.at(nxt, tris[:, k], low)
        while True:
            jumped = nxt[nxt]
            if np.array_equal(jumped, nxt):
                break
            nxt = jumped
        if np.array_equal(nxt, parent):
            break
        parent = nxt
    return np.unique(parent[tris[:, 0]], return_inverse=True)[1].reshape(-1)


def join_small_charts(positions, vmap, tris, uv, small_share: float = SMALL_SHARE):
    """The small charts of an unwrapped mesh joined onto neighbouring charts where they fit.

    positions (V, 3): the mesh's vertices; vmap (S,): the vertex of each split vertex; tris (T, 3): the triangles
    over split vertices; uv (S, 2): the split vertices' coordinates. Returns (vmap, tris, uv, joined): the same
    form, the triangles in their order, split vertices no triangle uses dropped; joined = how many charts joined.
    Hand xatlas packing_units(uv) to pack it again.
    """
    at = _Atlas(positions, vmap, tris, uv)
    joined = 0
    tried: set[tuple[int, int]] = set()
    for _ in range(MAX_PASSES):
        done = _pass(at, small_share, tried)
        joined += done
        if done == 0:
            break
    used = np.zeros(at.count, dtype=bool)
    used[at.tris.reshape(-1)] = True
    remap = np.cumsum(used) - 1
    return at.vmap[:at.count][used], remap[at.tris], at.uv[:at.count][used], joined


def packing_units(uv: np.ndarray) -> np.ndarray:
    """UVs (S, 2) scaled to span PACK_SPAN, as xatlas is to be given them to pack them again."""
    uv = np.asarray(uv, dtype=np.float64)
    span = float((uv.max(axis=0) - uv.min(axis=0)).max()) if len(uv) else 0.0
    return uv * (PACK_SPAN / span) if span > 0 else uv


def _cross2(u: np.ndarray, v: np.ndarray):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _expand(start: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row, start[row] + i) for every i < count[row]: ranges laid out flat."""
    row = np.repeat(np.arange(len(start)), count)
    return row, np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count) + np.repeat(start, count)


def _apart(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per row: are the 2D triangles a[i] and b[i] (n, 3, 2) apart but for a shared edge or corner? (An edge
    normal of one of them has them on its two sides; `a` is shrunk a hair, so touching is not overlapping.)"""
    mid = a.mean(axis=1, keepdims=True)
    a = mid + (a - mid) * (1.0 - 1e-6)
    axes = np.concatenate([np.roll(a, -1, axis=1) - a, np.roll(b, -1, axis=1) - b], axis=1)[..., ::-1] * [-1.0, 1.0]
    pa = np.einsum("nad,npd->nap", axes, a)
    pb = np.einsum("nad,npd->nap", axes, b)
    return ((pa.max(-1) <= pb.min(-1)) | (pb.max(-1) <= pa.min(-1))).any(axis=1)


class _Grid:
    """Triangles by the cells their UV bounding boxes touch, to find the ones near a box. A triangle that moves is
    added again where it is now; it stays listed where it was too, so a caller checks what it finds against the
    triangles' places now."""

    def __init__(self, size: float):
        self.size = size
        self.tri = np.zeros(0, dtype=np.int64)
        self.keys = np.zeros(0, dtype=np.int64)

    def add(self, ids: np.ndarray, tri_uv: np.ndarray) -> None:
        """Triangles `ids` (n,) at their UV places (n, 3, 2)."""
        if len(ids) == 0:
            return
        row, keys, _ = self._cells(tri_uv.min(axis=1), tri_uv.max(axis=1), limit=None)
        order = np.argsort(keys, kind="stable")
        keys, tri = keys[order], np.asarray(ids)[row[order]]
        at = np.searchsorted(self.keys, keys, "right")
        self.keys, self.tri = np.insert(self.keys, at, keys), np.insert(self.tri, at, tri)

    def pairs(self, lo: np.ndarray, hi: np.ndarray):
        """(box, triangle) for the triangles whose cells meet each box (n, 2), and the boxes too large to look up."""
        box, keys, huge = self._cells(lo, hi, MAX_CELLS)
        start = np.searchsorted(self.keys, keys, "left")
        row, j = _expand(start, np.searchsorted(self.keys, keys, "right") - start)
        return box[row], self.tri[j], huge

    def _cells(self, lo, hi, limit):
        x0, y0 = np.floor(lo / self.size).astype(np.int64).T
        x1, y1 = np.floor(hi / self.size).astype(np.int64).T
        sx, sy = x1 - x0 + 1, y1 - y0 + 1
        huge = np.zeros(len(lo), dtype=bool) if limit is None else sx * sy > limit
        row, k = _expand(np.zeros(len(lo), dtype=np.int64), np.where(huge, 0, sx * sy))
        cx, cy = x0[row] + k % sx[row], y0[row] + k // sx[row]
        return row, cx * 1_000_003 + cy, huge            # cells far apart may share a key: more pairs, none lost


class _Atlas:
    """The unwrapped mesh being edited: split vertices (mesh vertex, uv), triangles over them, a chart each."""

    def __init__(self, positions, vmap, tris, uv):
        self.pos = np.asarray(positions, dtype=np.float64)
        self.count = len(vmap)                              # split vertices so far; the buffers below grow
        self.vmap = np.array(vmap, dtype=np.int64)
        self.uv = np.array(uv, dtype=np.float64).reshape(-1, 2)
        self.tris = np.array(tris, dtype=np.int64)
        self.corners = self.vmap[self.tris]                 # (T, 3) mesh vertices
        self.chart = chart_labels(self.tris, self.count)
        p = self.pos[self.corners]
        self.area = np.bincount(self.chart, weights=0.5 * np.linalg.norm(
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1))
        tri_uv = self.uv[self.tris]
        signed = _cross2(tri_uv[:, 1] - tri_uv[:, 0], tri_uv[:, 2] - tri_uv[:, 0])
        self.turn = np.sign(np.bincount(self.chart, weights=signed))     # each chart's winding in UV
        self.cell = max(float(np.median(np.linalg.norm(tri_uv - np.roll(tri_uv, 1, axis=1), axis=-1))) * 2, 1e-12)
        # The triangle across each edge (edge k of triangle t runs from corner k to k + 1); -1 on a border or
        # where more than two triangles meet.
        a, b = self.corners, np.roll(self.corners, -1, axis=1)
        key = (np.minimum(a, b) * len(self.pos) + np.maximum(a, b)).reshape(-1)
        order = np.argsort(key, kind="stable")
        ordered = key[order]
        first = np.searchsorted(ordered, key, "left")
        two = np.searchsorted(ordered, key, "right") - first == 2
        mate = order[np.where(order[first] == np.arange(len(key)), np.minimum(first + 1, len(key) - 1), first)]
        self.partner = np.where(two, mate // 3, -1)
        # The triangle corners at each mesh vertex.
        corner_order = np.argsort(self.corners.reshape(-1), kind="stable")
        self.at_vertex = (corner_order, self.corners.reshape(-1)[corner_order])
        # Every triangle where it lies (a pass's own moves are looked up apart until the pass is settled).
        self.grid = _Grid(self.cell)
        self.grid.add(np.arange(len(self.tris)), tri_uv)

    def touching(self, vertices: np.ndarray) -> np.ndarray:
        """The triangles with a corner at one of the mesh vertices."""
        order, ordered = self.at_vertex
        start = np.searchsorted(ordered, vertices, "left")
        _, j = _expand(start, np.searchsorted(ordered, vertices, "right") - start)
        return np.unique(order[j] // 3)

    def add_vertices(self, vertices: np.ndarray, places: np.ndarray) -> np.ndarray:
        """New split vertices at the end; their indices."""
        while self.count + len(vertices) > len(self.vmap):
            self.vmap = np.concatenate([self.vmap, np.zeros_like(self.vmap)])
            self.uv = np.concatenate([self.uv, np.zeros_like(self.uv)])
        ids = np.arange(self.count, self.count + len(vertices))
        self.vmap[ids], self.uv[ids] = vertices, places
        self.count += len(vertices)
        return ids

    def nearest_place(self, vertex: np.ndarray, usable: Callable, near: np.ndarray):
        """Per row i: of the split vertices of mesh vertex[i] at the triangle corners where usable(rows, triangles)
        holds, the one nearest to the UV point near[i] -> (split vertex or -1, distance)."""
        order, ordered = self.at_vertex
        start = np.searchsorted(ordered, vertex, "left")
        row, j = _expand(start, np.searchsorted(ordered, vertex, "right") - start)
        tri, corner = order[j] // 3, order[j] % 3
        keep = usable(row, tri)
        row, split = row[keep], self.tris[tri[keep], corner[keep]]
        dist = np.linalg.norm(self.uv[split] - near[row], axis=1)
        best = np.lexsort((dist, row))
        first = best[np.unique(row[best], return_index=True)[1]]
        out = np.full(len(vertex), -1, dtype=np.int64)
        out_dist = np.full(len(vertex), np.inf)
        out[row[first]], out_dist[row[first]] = split[first], dist[first]
        return out, out_dist


def _targets(at: _Atlas, small: np.ndarray, tried: set) -> np.ndarray:
    """Per chart: its largest neighbouring chart that is larger and not tried yet (-1: none), for the small ones."""
    t = np.flatnonzero(np.isin(at.chart, small))
    t, k = np.repeat(t, 3), np.tile(np.arange(3), len(t))
    o = at.partner[t * 3 + k]
    x, y = at.chart[t[o >= 0]], at.chart[o[o >= 0]]
    keep = (y != x) & (at.area[y] > at.area[x])
    x, y = x[keep], y[keep]
    if tried:
        done = np.array(sorted(tried), dtype=np.int64)
        keep = ~np.isin(x * len(at.area) + y, done[:, 0] * len(at.area) + done[:, 1])
        x, y = x[keep], y[keep]
    best = np.lexsort((-at.area[y], x))
    first = best[np.unique(x[best], return_index=True)[1]]
    target = np.full(len(at.area), -1, dtype=np.int64)
    target[x[first]] = y[first]
    # A chart whose target is itself being laid elsewhere waits for the next pass: its triangles would hang on
    # corners that are about to move.
    target[(target >= 0) & (target[np.maximum(target, 0)] >= 0)] = -1
    return target


def _pass(at: _Atlas, small_share: float, tried: set) -> int:
    """Every small chart laid onto its best untried neighbour; the charts laid whole are kept. How many."""
    small = np.flatnonzero((at.area > 0) & (at.area < small_share * at.area.sum()))
    target = _targets(at, small, tried)
    if not (target >= 0).any():
        return 0
    size = np.bincount(at.chart, minlength=len(at.area))
    owner = np.full(len(at.tris), -1, dtype=np.int64)       # the chart a triangle laid this pass came from
    before = at.tris.copy()
    laid = _Grid(at.cell)                                   # this pass's laid triangles (at.grid: the others)
    moving = target[at.chart] >= 0                          # the triangles of the charts being laid
    look = np.flatnonzero(moving)                           # which may fit now: first all, then what changed
    while True:
        # Each waiting triangle beside something its chart may lay onto: the target's own triangles or its own
        # chart's triangles laid already.
        t = look[owner[look] < 0]
        t, k = np.repeat(t, 3), np.tile(np.arange(3), len(t))
        o = at.partner[t * 3 + k]
        x = at.chart[t]
        c = target[x]
        o_safe = np.maximum(o, 0)
        keep = (o >= 0) & (((at.chart[o_safe] == c) & (owner[o_safe] < 0)) | (owner[o_safe] == x))
        t, k, o, x, c = t[keep], k[keep], o[keep], x[keep], c[keep]
        if len(t) == 0:
            break
        n = np.arange(len(t))
        k1, k2 = (k + 1) % 3, (k + 2) % 3
        va, vb, vc = at.corners[t, k], at.corners[t, k1], at.corners[t, k2]
        oc = at.corners[o]
        ia, ib = (oc == va[:, None]).argmax(1), (oc == vb[:, None]).argmax(1)
        usable = (oc[n, ia] == va) & (oc[n, ib] == vb) & (ia != ib)
        sa, sb, so = at.tris[o, ia], at.tris[o, ib], at.tris[o, (3 - ia - ib) % 3]
        ua, ub = at.uv[sa], at.uv[sb]
        uc, unfolded = _unfold(at.pos[va], at.pos[vb], at.pos[vc], ua, ub, at.uv[so])
        usable &= unfolded
        snap, dist = at.nearest_place(
            vc, lambda row, tri: (at.chart[tri] == c[row]) & ((owner[tri] < 0) | (owner[tri] == x[row])), uc)
        snap = np.where(dist <= SNAP * np.linalg.norm(ub - ua, axis=1), snap, -1)
        chosen = np.full(len(t), -2, dtype=np.int64)        # -2 does not fit, -1 a new split vertex, else that one
        third = np.zeros((len(t), 2))
        for place, here in ((snap, snap >= 0), (np.full(len(t), -1), True)):     # the island's own corner first
            idx = np.flatnonzero(usable & (chosen == -2) & here)
            if len(idx) == 0:
                continue
            spot = np.where((place[idx] >= 0)[:, None], at.uv[np.maximum(place[idx], 0)], uc[idx])
            tri = _tri(idx, k, k1, k2, ua, ub, spot)
            turn = at.turn[c[idx]]
            fits = (turn == 0) | (_cross2(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) * turn > 0)
            fits &= ~_overlaps(at, tri, c[idx], t[idx], owner, laid)
            chosen[idx[fits]], third[idx[fits]] = place[idx[fits]], spot[fits]
        # One edge per triangle, the island's own corner first; one new split vertex per vertex of a chart in a
        # round (the others take its place in the next); none overlapping another laid this round.
        pick = np.flatnonzero(chosen >= -1)
        pick = pick[np.lexsort((chosen[pick] == -1, t[pick]))]
        pick = pick[np.unique(t[pick], return_index=True)[1]]
        fresh = pick[chosen[pick] == -1]
        pick = np.union1d(pick[chosen[pick] >= 0],
                          fresh[np.unique(x[fresh] * len(at.pos) + vc[fresh], return_index=True)[1]])
        tri = _tri(pick, k, k1, k2, ua, ub, third[pick])
        mine = _Grid(at.cell)
        mine.add(np.arange(len(pick)), tri)
        box, other, _ = mine.pairs(tri.min(axis=1), tri.max(axis=1))
        pair = (other > box) & (c[pick][box] == c[pick][other])
        keep = np.ones(len(pick), dtype=bool)
        keep[other[pair][~_apart(tri[box[pair]], tri[other[pair]])]] = False
        pick, tri = pick[keep], tri[keep]
        if len(pick) == 0:
            break
        # Next round: a triangle that fitted but gave way, and any with a corner where one is laid now (it may lay
        # across it, or take its corner); one that did not fit and has nothing new beside it would not fit now.
        near = at.touching(np.unique(at.corners[t[pick]]))
        look = np.union1d(near[moving[near]], np.setdiff1d(t[chosen >= -1], t[pick]))
        corner3 = chosen[pick].copy()
        new = corner3 == -1
        corner3[new] = at.add_vertices(vc[pick][new], third[pick][new])
        rows = np.zeros((len(pick), 3), dtype=np.int64)
        m = np.arange(len(pick))
        rows[m, k[pick]], rows[m, k1[pick]], rows[m, k2[pick]] = sa[pick], sb[pick], corner3
        at.tris[t[pick]] = rows
        at.chart[t[pick]] = c[pick]
        owner[t[pick]] = x[pick]
        laid.add(t[pick], tri)
    # Settled: a chart laid whole joins its target; any other is put back as it was.
    tried_now = np.unique(owner[owner >= 0])
    laid_count = np.bincount(owner[owner >= 0], minlength=len(at.area))
    whole = tried_now[laid_count[tried_now] == size[tried_now]]
    back = np.isin(owner, tried_now[laid_count[tried_now] < size[tried_now]])
    at.tris[back] = before[back]
    at.chart[back] = owner[back]
    kept = np.flatnonzero(np.isin(owner, whole))
    at.grid.add(kept, at.uv[at.tris[kept]])
    np.add.at(at.area, target[whole], at.area[whole])
    at.area[whole] = 0.0
    tried.update((int(a), int(target[a])) for a in np.flatnonzero(target >= 0))
    return len(whole)


def _unfold(pa, pb, pc, ua, ub, away):
    """Where corner c of each triangle (a, b, c) lands when the triangle is unfolded flat across edge a-b in UV
    (ua, ub), on the side away from the 2D point `away`, its shape kept at the scale of the edge in UV
    -> (uc (n, 2), usable (n,): False for an edge of no length)."""
    e3 = pb - pa
    length2 = (e3 * e3).sum(1)
    eu = ub - ua
    usable = (length2 > 0) & ((eu * eu).sum(1) > 0)
    length2 = np.where(usable, length2, 1.0)
    along = ((pc - pa) * e3).sum(1) / length2
    height = np.linalg.norm((pc - pa) - along[:, None] * e3, axis=1) / np.sqrt(length2)
    side = np.where(_cross2(eu, away - ua) > 0, -1.0, 1.0)
    return ua + along[:, None] * eu + (side * height)[:, None] * np.stack([-eu[:, 1], eu[:, 0]], axis=1), usable


def _tri(idx, k, k1, k2, ua, ub, third):
    """UV triangles (n, 3, 2) in the corner order of the triangles they lay."""
    tri = np.zeros((len(idx), 3, 2))
    n = np.arange(len(idx))
    tri[n, k[idx]], tri[n, k1[idx]], tri[n, k2[idx]] = ua[idx], ub[idx], third
    return tri


def _overlaps(at: _Atlas, tri, chart, own, owner, laid: _Grid) -> np.ndarray:
    """Per new triangle (n, 3, 2): does it overlap a triangle of its chart - one there before this pass, or one
    laid onto it this pass? A triangle too large to look around counts as overlapping."""
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    hit = np.zeros(len(tri), dtype=bool)
    for grid, this_pass in ((at.grid, False), (laid, True)):
        box, other, huge = grid.pairs(lo, hi)
        hit |= huge
        mine = (at.chart[other] == chart[box]) & (other != own[box]) & ((owner[other] >= 0) == this_pass)
        box, other = box[mine], other[mine]
        theirs = at.uv[at.tris[other]]
        near = ((theirs.min(axis=1) < hi[box]) & (theirs.max(axis=1) > lo[box])).all(axis=1)
        box, theirs = box[near], theirs[near]
        hit[box[~_apart(tri[box], theirs)]] = True
    return hit
