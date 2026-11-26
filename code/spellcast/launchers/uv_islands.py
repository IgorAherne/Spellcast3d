"""Small UV islands joined onto a neighbouring island (numpy only), so an atlas has fewer crumbs.

xatlas grows a chart only while the chart still lies flat in one projection, so a triangle folded sharply against
its neighbours - a sliver along a crease, common in decimated and retopologized meshes - stays a chart of its own.
Here each small chart is laid onto its largest neighbour instead, triangle by triangle: a triangle is unfolded flat
across an edge it shares with what is laid already, at the neighbour's own texel scale along that edge, and its
third corner takes the island's own place of that vertex when the unfolded one lands near it (closing the notch
between them). A chart is joined whole or not at all: if one of its triangles would overlap the island or turn
over, it stays as the unwrapper made it and may try its next neighbour in the next pass. Every small chart is laid
at once, a ring of triangles per round. The joined islands keep their old places, so the caller packs the atlas
again (xatlas with the input UVs, in packing_units, as the meshes of chart_groups).

Used by the 3D stage's unwrap (pixal3d_glb) and by Instant Meshes' (imesh_server: Retopo's three methods).
"""
from __future__ import annotations

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
#: The joined atlas is handed to xatlas as this many meshes (chart_groups): it reads the charts of several meshes
#: side by side, and those of one big mesh in several times the time.
PACK_GROUPS = 16

_SAT_ROWS = 16384   # triangle pairs tested at a time (the test's scratch stays in the cache)
_SHORT = 8192       # entries a listing's short part takes before it is poured into the long one
_HASH = (73856093, 19349663, 83492791)      # a grid cell's x, y and the chart, mixed into one key


def chart_labels(tris: np.ndarray, count: int) -> np.ndarray:
    """A chart id per triangle, 0..n-1: the triangles that share a split vertex are one chart."""
    tris = np.asarray(tris)
    corners = tris.reshape(-1)
    parent = np.arange(count)                               # ends as the least split vertex of each chart
    while len(corners):
        p = parent[tris]
        low = np.repeat(np.minimum(np.minimum(p[:, 0], p[:, 1]), p[:, 2]), 3)
        least = parent.copy()
        np.minimum.at(least, corners, low)                  # per vertex: the least its triangles have seen
        nxt = least.copy()
        np.minimum.at(nxt, parent, least)                   # and its root takes it too, so whole trees join
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
    tried = np.zeros(0, dtype=np.int64)                     # (chart, target) pairs tried, as chart * charts + target
    for _ in range(MAX_PASSES):
        done, tried = _pass(at, small_share, tried)
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


def chart_groups(tris: np.ndarray, count: int, groups: int = PACK_GROUPS) -> list[np.ndarray]:
    """The triangles (T, 3) over `count` split vertices dealt into at most `groups` sets of whole charts, the
    biggest charts first, one to each set in turn: each set's triangle indices, ascending. Every set is a mesh of
    its own for xatlas to pack the atlas again from (the triangles keep their order within it)."""
    chart = chart_labels(tris, count)
    sizes = np.bincount(chart)
    groups = max(1, min(int(groups), len(sizes)))
    of_chart = np.empty(len(sizes), dtype=np.int64)
    of_chart[np.argsort(-sizes, kind="stable")] = np.arange(len(sizes)) % groups
    of_tri = of_chart[chart]
    return [np.flatnonzero(of_tri == group) for group in range(groups)]


def _cross2(u: np.ndarray, v: np.ndarray):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _expand(start: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row, start[row] + i) for every i < count[row]: ranges laid out flat."""
    row = np.repeat(np.arange(len(start)), count)
    return row, np.arange(len(row)) + np.repeat(start - (np.cumsum(count) - count), count)


def _bounds(tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The boxes (lo, hi), each (n, 2), of triangles (n, 3, 2)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    return np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c)


def _apart(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per row: are the 2D triangles a[i] and b[i] (n, 3, 2) apart but for a shared edge or corner? (An edge
    normal of one of them has them on its two sides; `a` is shrunk a hair, so touching is not overlapping.)"""
    if len(a) > _SAT_ROWS:
        return np.concatenate([_apart(a[i:i + _SAT_ROWS], b[i:i + _SAT_ROWS]) for i in range(0, len(a), _SAT_ROWS)])
    n = len(a)
    mid = ((a[:, 0] + a[:, 1] + a[:, 2]) / 3.0)[:, None]
    a = mid + (a - mid) * (1.0 - 1e-6)
    ex, ey = np.empty((n, 6)), np.empty((n, 6))             # the six edges, a's then b's
    for i, tri in ((0, a), (3, b)):
        x, y = tri[:, :, 0], tri[:, :, 1]
        ex[:, i], ex[:, i + 1], ex[:, i + 2] = x[:, 1] - x[:, 0], x[:, 2] - x[:, 1], x[:, 0] - x[:, 2]
        ey[:, i], ey[:, i + 1], ey[:, i + 2] = y[:, 1] - y[:, 0], y[:, 2] - y[:, 1], y[:, 0] - y[:, 2]
    nx = ey * -1.0                                          # an edge's normal: (-y, x)
    ends = []
    for tri in (a, b):                                      # each triangle's reach along the six normals
        lo = nx * tri[:, 0, 0:1]
        lo += ex * tri[:, 0, 1:2]
        hi = lo.copy()
        for i in (1, 2):
            along = nx * tri[:, i, 0:1]
            along += ex * tri[:, i, 1:2]
            np.minimum(lo, along, out=lo)
            np.maximum(hi, along, out=hi)
        ends.append((lo, hi))
    split = np.zeros((n, 8), dtype=bool)                    # per normal: the two on its two sides
    np.less_equal(ends[0][1], ends[1][0], out=split[:, :6])
    split[:, :6] |= ends[1][1] <= ends[0][0]
    return split.view(np.uint64).reshape(n) != 0


class _Moved:
    """The triangles that lie elsewhere than the unwrapper left them, by the grid cells their UV boxes touch and
    by their chart, to find a chart's triangles near a box. An entry is a triangle and the box it was listed
    with; the entries' keys are kept sorted, a long run and a short one that takes what is added (poured into the
    long one when it has grown). A triangle that moves again is added again and stays listed where it was too,
    so a caller checks what it finds against the triangles' charts now."""

    def __init__(self):
        self.count = 0                                      # entries so far; the two buffers grow
        self.tri = np.zeros(1024, dtype=np.int64)
        self.box = np.zeros((1024, 4))
        none = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        self.long, self.short = none, none                  # each (sorted keys, their entries)

    @staticmethod
    def _merged(a, b):
        """Two sorted runs as one (a stable sort of two sorted runs is a merge)."""
        if len(a[0]) == 0:
            return b
        keys = np.concatenate([a[0], b[0]])
        order = np.argsort(keys, kind="stable")
        return keys[order], np.concatenate([a[1], b[1]])[order]

    def state(self):
        """What `back` needs to forget everything added from now on."""
        return self.count, self.long, self.short

    def back(self, state) -> None:
        self.count, self.long, self.short = state

    def add(self, keys: np.ndarray, tri: np.ndarray, box: np.ndarray) -> None:
        """Entries: a key each, its triangle, the box (n, 4) it lies in."""
        if len(keys) == 0:
            return
        while self.count + len(keys) > len(self.tri):
            self.tri = np.concatenate([self.tri, np.zeros_like(self.tri)])
            self.box = np.concatenate([self.box, np.zeros_like(self.box)])
        ids = np.arange(self.count, self.count + len(keys))
        self.tri[ids], self.box[ids] = tri, box
        self.count += len(keys)
        order = np.argsort(keys, kind="stable")
        self.short = self._merged(self.short, (keys[order], ids[order]))
        if len(self.short[0]) > max(_SHORT, len(self.long[0]) // 4):
            self.long, self.short = self._merged(self.long, self.short), (self.short[0][:0], self.short[1][:0])

    def find(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(which key, entry) for every entry under each key."""
        found = []
        for listed, entry in (self.long, self.short):
            if len(listed):
                start = np.searchsorted(listed, keys, "left")
                row, j = _expand(start, np.searchsorted(listed, keys, "right") - start)
                found.append((row, entry[j]))
        if len(found) == 1:
            return found[0]
        if not found:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return np.concatenate([found[0][0], found[1][0]]), np.concatenate([found[0][1], found[1][1]])


def _cells(size: float, lo: np.ndarray, hi: np.ndarray, limit: int | None):
    """The grid cells the boxes (n, 2) touch: (box, cell x, cell y) per cell, each box's first cell (x0, y0), and
    the boxes too large to list (more than `limit` cells; they get no cell)."""
    x0, y0 = np.floor(lo / size).astype(np.int64).T
    x1, y1 = np.floor(hi / size).astype(np.int64).T
    sx, sy = x1 - x0 + 1, y1 - y0 + 1
    cells = sx * sy
    huge = np.zeros(len(lo), dtype=bool) if limit is None else cells > limit
    row, k = _expand(np.zeros(len(lo), dtype=np.int64), np.where(huge, 0, cells))
    wide = sx[row]
    return row, x0[row] + k % wide, y0[row] + k // wide, x0, y0, huge


def _key(cx: np.ndarray, cy: np.ndarray, chart: np.ndarray) -> np.ndarray:
    """One number for a cell and a chart (cells far apart may share it: more to look at, none lost)."""
    return cx * _HASH[0] + cy * _HASH[1] + chart * _HASH[2]


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
        first = np.flatnonzero(np.concatenate([[True], ordered[1:] != ordered[:-1]])) if len(key) else order
        two = first[np.diff(np.append(first, len(key))) == 2]
        partner = np.full(len(key), -1, dtype=np.int64)
        partner[order[two]], partner[order[two + 1]] = order[two + 1] // 3, order[two] // 3
        self.partner = partner.reshape(-1, 3)
        # The edges between two charts (a triangle, the one across): where a chart's neighbours are looked for.
        across = self.chart[np.maximum(self.partner, 0)]
        t, k = np.nonzero((self.partner >= 0) & (across != self.chart[:, None]))
        self.border = (t, self.partner[t, k])
        # The triangle corners at each mesh vertex: at_vertex[vertex_start[v]:vertex_start[v + 1]].
        flat = self.corners.reshape(-1)
        self.at_vertex = np.argsort(flat, kind="stable")
        self.vertex_start = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=len(self.pos)))])
        # Every triangle's corners where they lie now (the triangles laid move).
        self.place = tri_uv.reshape(-1, 6).copy()
        # The triangles as the unwrapper left them, by cell and chart: bucket b holds the entries
        # bucket[b]:bucket[b + 1] of listed (the triangle) and listed_box (its box).
        lo, hi = _bounds(tri_uv)
        listed, cx, cy, _, _, _ = _cells(self.cell, lo, hi, None)
        keys, box = _key(cx, cy, self.chart[listed]), np.concatenate([lo, hi], axis=1)[listed]
        self.mask = (1 << max(10, int(np.ceil(np.log2(max(2 * len(keys), 1)))))) - 1
        order = np.argsort(keys & self.mask, kind="stable")
        self.listed, self.listed_box = listed[order], box[order]
        self.bucket = np.concatenate([[0], np.cumsum(np.bincount(keys & self.mask, minlength=self.mask + 1))])
        self.moved = _Moved()                               # the triangles joined in earlier passes, where they lie
        self.flag = np.zeros(len(self.tris), dtype=bool)    # scratch: all False between uses

    def touching(self, vertices: np.ndarray) -> np.ndarray:
        """The triangles with a corner at one of the mesh vertices (each as often as it has such corners)."""
        start = self.vertex_start[vertices]
        _, j = _expand(start, self.vertex_start[vertices + 1] - start)
        return self.at_vertex[j] // 3

    def add_vertices(self, vertices: np.ndarray, places: np.ndarray) -> np.ndarray:
        """New split vertices at the end; their indices."""
        while self.count + len(vertices) > len(self.vmap):
            self.vmap = np.concatenate([self.vmap, np.zeros_like(self.vmap)])
            self.uv = np.concatenate([self.uv, np.zeros_like(self.uv)])
        ids = np.arange(self.count, self.count + len(vertices))
        self.vmap[ids], self.uv[ids] = vertices, places
        self.count += len(vertices)
        return ids

    def nearest_place(self, vertex: np.ndarray, chart: np.ndarray, own: np.ndarray, owner: np.ndarray,
                      near: np.ndarray):
        """Per row i: of the split vertices of mesh vertex[i] at the corners of chart[i]'s triangles - those there
        before this pass, and those chart own[i] laid in it - the one nearest to the UV point near[i]
        -> (split vertex or -1, distance)."""
        out = np.full(len(vertex), -1, dtype=np.int64)
        out_dist = np.full(len(vertex), np.inf)
        start = self.vertex_start[vertex]
        row, j = _expand(start, self.vertex_start[vertex + 1] - start)
        corner = self.at_vertex[j]
        tri = corner // 3
        by = owner[tri]
        keep = np.flatnonzero((self.chart[tri] == chart[row]) & ((by < 0) | (by == own[row])))
        if len(keep) == 0:
            return out, out_dist
        row, split = row[keep], self.tris.reshape(-1)[corner[keep]]
        d = self.uv[split] - near[row]
        dist = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])
        # The first of each row's nearest (the rows run in order)
        begins = np.flatnonzero(np.concatenate([[True], row[1:] != row[:-1]]))
        least = np.fmin.reduceat(dist, begins)
        group = np.repeat(np.arange(len(begins)), np.diff(np.append(begins, len(row))))
        hit = np.flatnonzero(dist == least[group])
        of = group[hit]
        hit = hit[np.concatenate([[True], of[1:] != of[:-1]])[:len(hit)]]
        out[row[hit]], out_dist[row[hit]] = split[hit], dist[hit]
        return out, out_dist


def _targets(at: _Atlas, small: np.ndarray, tried: np.ndarray) -> np.ndarray:
    """Per chart: its largest neighbouring chart that is larger and not tried yet (-1: none), for the small ones
    (`small`: a flag per chart)."""
    charts = len(at.area)
    x, y = at.chart[at.border[0]], at.chart[at.border[1]]
    keep = small[x] & (y != x) & (at.area[y] > at.area[x])
    x, y = x[keep], y[keep]
    if len(tried):
        code = x * charts + y
        keep = tried[np.minimum(np.searchsorted(tried, code), len(tried) - 1)] != code
        x, y = x[keep], y[keep]
    best = np.lexsort((-at.area[y], x))
    first = best[np.unique(x[best], return_index=True)[1]]
    target = np.full(charts, -1, dtype=np.int64)
    target[x[first]] = y[first]
    # A chart whose target is itself being laid elsewhere waits for the next pass: its triangles would hang on
    # corners that are about to move.
    target[(target >= 0) & (target[np.maximum(target, 0)] >= 0)] = -1
    return target


def _pass(at: _Atlas, small_share: float, tried: np.ndarray) -> tuple[int, np.ndarray]:
    """Every small chart laid onto its best untried neighbour; the charts laid whole are kept. How many, and the
    pairs tried so far."""
    charts = len(at.area)
    target = _targets(at, (at.area > 0) & (at.area < small_share * at.area.sum()), tried)
    going = np.flatnonzero(target >= 0)
    if len(going) == 0:
        return 0, tried
    size = np.bincount(at.chart, minlength=charts)
    owner = np.full(len(at.tris), -1, dtype=np.int64)       # the chart a triangle laid this pass came from
    before = at.tris.copy()
    kept_so_far = at.moved.state()                          # this pass's laid triangles are listed there meanwhile
    moving = target[at.chart] >= 0                          # the triangles of the charts being laid
    look = np.flatnonzero(moving)                           # which may fit now: first all, then what changed
    failed_at = np.full(len(at.tris) * 3, -2, dtype=np.int64)   # per edge: the island's corner that did not fit
    failed_new = np.zeros(len(at.tris) * 3, dtype=bool)         # per edge: a corner of its own did not fit
    members = None                                          # the moving triangles by chart, when one is given up
    while True:
        # Each waiting triangle beside something its chart may lay onto: the target's own triangles or its own
        # chart's triangles laid already.
        t = look[owner[look] < 0]
        o = at.partner[t]
        x = at.chart[t]
        c = target[x]
        o_safe = np.maximum(o, 0)
        by = owner[o_safe]
        t_row, k = np.nonzero((o >= 0) & (((at.chart[o_safe] == c[:, None]) & (by < 0)) | (by == x[:, None])))
        if len(t_row) == 0:
            break
        t, o, x, c = t[t_row], o[t_row, k], x[t_row], c[t_row]
        n = np.arange(len(t))
        k1, k2 = (k + 1) % 3, (k + 2) % 3
        mine = at.corners[t]
        va, vb, vc = mine[n, k], mine[n, k1], mine[n, k2]
        oc = at.corners[o]
        ia, ib = (oc == va[:, None]).argmax(1), (oc == vb[:, None]).argmax(1)
        usable = (oc[n, ia] == va) & (oc[n, ib] == vb) & (ia != ib)
        theirs = at.tris[o]
        sa, sb, so = theirs[n, ia], theirs[n, ib], theirs[n, (3 - ia - ib) % 3]
        ua, ub = at.uv[sa], at.uv[sb]
        uc, unfolded = _unfold(at.pos[va], at.pos[vb], at.pos[vc], ua, ub, at.uv[so])
        usable &= unfolded
        snap, dist = at.nearest_place(vc, c, x, owner, uc)
        across = ub - ua
        snap = np.where(dist <= SNAP * np.sqrt(across[:, 0] * across[:, 0] + across[:, 1] * across[:, 1]), snap, -1)
        edge = t * 3 + k
        chosen = np.full(len(t), -2, dtype=np.int64)        # -2 does not fit, -1 a new split vertex, else that one
        third = np.zeros((len(t), 2))
        for own_corner in (True, False):                    # the island's own corner first
            # What did not fit once in this pass does not fit later in it: what is in the way only grows
            if own_corner:
                idx = np.flatnonzero(usable & (snap >= 0) & (snap != failed_at[edge]))
            else:
                idx = np.flatnonzero(usable & (chosen == -2) & ~failed_new[edge])
            if len(idx) == 0:
                continue
            spot = at.uv[snap[idx]] if own_corner else uc[idx]
            tri = _tri(idx, k, k1, k2, ua, ub, spot)
            turn = at.turn[c[idx]]
            fits = (turn == 0) | (_cross2(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) * turn > 0)
            sub = np.flatnonzero(fits)
            if len(sub):
                fits[sub[_overlaps(at, tri[sub], c[idx[sub]], t[idx[sub]])]] = False
            good, bad = idx[fits], idx[~fits]
            if own_corner:
                chosen[good], third[good] = snap[good], spot[fits]
                failed_at[edge[bad]] = snap[bad]
            else:
                chosen[good], third[good] = -1, spot[fits]
                failed_new[edge[bad]] = True
        fitted = chosen >= -1
        if not fitted.all():
            # A triangle that fits nowhere can only be laid across a neighbour of its chart that is laid later. If
            # the unlaid ones beside it fit nowhere either and have no unlaid neighbour beyond them, none of them
            # will ever be laid: their chart goes back as it was at once, and what it laid is in nobody's way.
            ends = np.flatnonzero(np.concatenate([t[1:] != t[:-1], [True]]))
            none = np.logical_or.reduceat(fitted, np.concatenate([[0], ends[:-1] + 1])) == 0
            stuck, home = t[ends[none]], x[ends[none]][:, None]
            beside = at.partner[stuck]
            near = np.maximum(beside, 0)
            waits = (beside >= 0) & (at.chart[near] == home)        # a chart's triangles not laid yet keep its id
            at.flag[t[fitted]] = True
            dead = np.ones(len(stuck), dtype=bool)
            for j in range(3):
                far = at.partner[near[:, j]]
                beyond = ((far >= 0) & (at.chart[np.maximum(far, 0)] == home) & (far != stuck[:, None])
                          & (far != beside[:, 0:1]) & (far != beside[:, 1:2]) & (far != beside[:, 2:3]))
                dead &= ~waits[:, j] | ~(at.flag[near[:, j]] | beyond[:, 0] | beyond[:, 1] | beyond[:, 2])
            at.flag[t[fitted]] = False
            lost = np.unique(home[dead, 0])
            if len(lost):
                if members is None:
                    first_look = np.flatnonzero(moving)
                    chart_of = np.where(owner >= 0, owner, at.chart)[first_look]
                    members = (first_look[np.argsort(chart_of, kind="stable")],
                               np.concatenate([[0], np.cumsum(np.bincount(chart_of, minlength=charts))]))
                _, j = _expand(members[1][lost], members[1][lost + 1] - members[1][lost])
                gone = members[0][j]
                moving[gone] = False
                back = gone[owner[gone] >= 0]
                at.tris[back] = before[back]
                at.chart[back] = owner[back]
                owner[back] = -1
                at.place[back] = at.uv[at.tris[back]].reshape(-1, 6)
                fitted &= moving[t]
                chosen[~fitted] = -2
        # One edge per triangle, the island's own corner first; one new split vertex per vertex of a chart in a
        # round (the others take its place in the next); none overlapping another laid this round.
        pick = np.flatnonzero(fitted)
        pick = pick[np.lexsort((chosen[pick] == -1, t[pick]))]
        pick = pick[np.unique(t[pick], return_index=True)[1]]
        fresh = pick[chosen[pick] == -1]
        pick = np.union1d(pick[chosen[pick] >= 0],
                          fresh[np.unique(x[fresh] * len(at.pos) + vc[fresh], return_index=True)[1]])
        tri = _tri(pick, k, k1, k2, ua, ub, third[pick])
        onto = c[pick]
        lo, hi = _bounds(tri)
        of, cx, cy, x0, y0, _ = _cells(at.cell, lo, hi, None)
        keys = _key(cx, cy, onto[of])
        order = np.argsort(keys, kind="stable")
        keys, of, cx, cy = keys[order], of[order], cx[order], cy[order]
        begins = np.flatnonzero(np.concatenate([[True], keys[1:] != keys[:-1]]))
        count = np.diff(np.append(begins, len(keys)))
        if count.max() > 1:
            # Every two of them under one cell and target whose boxes meet (taken where the boxes' common part
            # begins, so once): the later one gives way if they overlap
            rest = np.repeat(begins + count, count) - np.arange(len(keys)) - 1     # entries after me under my key
            i, j = _expand(np.arange(len(keys)) + 1, rest)
            first, second = np.minimum(of[i], of[j]), np.maximum(of[i], of[j])
            pair = np.flatnonzero((first != second) & (onto[first] == onto[second])
                                  & (lo[first, 0] < hi[second, 0]) & (hi[first, 0] > lo[second, 0])
                                  & (lo[first, 1] < hi[second, 1]) & (hi[first, 1] > lo[second, 1])
                                  & (np.maximum(x0[first], x0[second]) == cx[i])
                                  & (np.maximum(y0[first], y0[second]) == cy[i]))
            first, second = first[pair], second[pair]
            keep = np.ones(len(pick), dtype=bool)
            keep[second[~_apart(tri[first], tri[second])]] = False
            if not keep.all():
                listed = keep[of]
                keys, of = keys[listed], (np.cumsum(keep) - 1)[of[listed]]
                pick, tri, lo, hi = pick[keep], tri[keep], lo[keep], hi[keep]
        if len(pick) == 0:
            break
        # Next round: a triangle that fitted but gave way, and any with a corner where one is laid now (it may lay
        # across it, or take its corner); one that did not fit and has nothing new beside it would not fit now.
        near = at.touching(np.unique(mine[pick]))
        at.flag[near[moving[near]]] = True
        at.flag[t[fitted]] = True
        look = np.flatnonzero(at.flag)
        at.flag[look] = False
        corner3 = chosen[pick].copy()
        new = corner3 == -1
        corner3[new] = at.add_vertices(vc[pick][new], third[pick][new])
        rows = np.zeros((len(pick), 3), dtype=np.int64)
        m = np.arange(len(pick))
        rows[m, k[pick]], rows[m, k1[pick]], rows[m, k2[pick]] = sa[pick], sb[pick], corner3
        at.tris[t[pick]] = rows
        at.chart[t[pick]] = c[pick]
        owner[t[pick]] = x[pick]
        at.place[t[pick]] = tri.reshape(-1, 6)
        at.moved.add(keys, t[pick][of], np.concatenate([lo, hi], axis=1)[of])
    # Settled: a chart laid whole joins its target; any other is put back as it was.
    was_laid = np.flatnonzero(owner >= 0)
    laid_count = np.bincount(owner[was_laid], minlength=charts)
    whole = np.flatnonzero((laid_count > 0) & (laid_count == size))
    is_whole = np.zeros(charts, dtype=bool)
    is_whole[whole] = True
    kept = was_laid[is_whole[owner[was_laid]]]
    back = was_laid[~is_whole[owner[was_laid]]]
    at.tris[back] = before[back]
    at.chart[back] = owner[back]
    at.place[back] = at.uv[at.tris[back]].reshape(-1, 6)
    at.moved.back(kept_so_far)
    lo, hi = _bounds(at.place[kept].reshape(-1, 3, 2))
    of, cx, cy, _, _, _ = _cells(at.cell, lo, hi, None)
    at.moved.add(_key(cx, cy, at.chart[kept[of]]), kept[of], np.concatenate([lo, hi], axis=1)[of])
    np.add.at(at.area, target[whole], at.area[whole])
    at.area[whole] = 0.0
    tried = np.union1d(tried, going * charts + target[going])
    return len(whole), tried


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


def _overlaps(at: _Atlas, tri, chart, own) -> np.ndarray:
    """Per new triangle (n, 3, 2): does it overlap a triangle of its chart - one there before this pass, or one
    laid onto it this pass? A triangle too large to look around counts as overlapping."""
    lo, hi = _bounds(tri)
    row, cx, cy, x0, y0, huge = _cells(at.cell, lo, hi, MAX_CELLS)
    keys = _key(cx, cy, chart[row])
    reach = np.concatenate([hi, lo], axis=1)[row]           # per cell: its triangle's box, the far corner first
    slot = keys & at.mask
    start = at.bucket[slot]
    pairs = []
    for (cell, entry), listed, listed_box in (
            (_expand(start, at.bucket[slot + 1] - start), at.listed, at.listed_box),
            (at.moved.find(keys), at.moved.tri, at.moved.box)):
        if len(cell) == 0:
            continue
        # Those whose boxes meet, each taken once: in the cell where the two boxes' common part begins
        theirs, mine = listed_box[entry], reach[cell]
        near = (theirs[:, :2] < mine[:, :2]) & (theirs[:, 2:] > mine[:, 2:])
        near = np.flatnonzero(near[:, 0] & near[:, 1])
        cell, other = cell[near], listed[entry[near]]
        box = row[cell]
        begins = np.floor(theirs[near, :2] / at.cell).astype(np.int64)
        once = np.flatnonzero((np.maximum(begins[:, 0], x0[box]) == cx[cell])
                              & (np.maximum(begins[:, 1], y0[box]) == cy[cell])
                              & (at.chart[other] == chart[box]) & (other != own[box]))
        pairs.append((box[once], other[once]))
    hit = huge.copy()
    if pairs:
        box, other = (np.concatenate(side) for side in zip(*pairs)) if len(pairs) > 1 else pairs[0]
        hit[box[~_apart(tri[box], at.place[other].reshape(-1, 3, 2))]] = True
    return hit
