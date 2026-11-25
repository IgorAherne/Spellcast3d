"""The seams of an Omni-Texture blended in 3D, on the model's surface across its UV islands (blend).

Where the texture's source changes - one view's picture to another's, or to the model's own texture - the sources'
weights cross-fade over the surface. No colour is blurred, so each picture keeps its own sharp detail (a blur of
the colours that wide smears fine features a seam crosses), and it is the only blending of edges: a feather made in
the picture would be wider on slanted surfaces and miss the lines where one part hides another."""

from __future__ import annotations

import numpy as np

ALIKE = 0.5                  # the cosine between two patches' normals from which they are one surface to blend
REACH = 4                    # voxels to each side a blur reaches along an axis (two sigmas of the wider one)
WHOLE = 1.0 - 1e-4           # a blurred share above this is a source's alone
RING = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]


def within_reach(share: np.ndarray) -> np.ndarray:
    """How far inside what a view can paint a place lies, from the blurred share of its surroundings the view can
    paint: nothing where half of them or fewer can be (the edge of what it sees), whole where all can."""
    return np.clip(2.0 * share - 1.0, 0.0, 1.0)


class _Voxels:
    """The texels gathered in voxels of side `h`, one set per side the surface faces (the main axis of its normal
    and its sign), so that the two sides of something thin never share a voxel. Voxels are in the order of their
    keys: the cell's number times 6, plus the set."""

    def __init__(self, points: np.ndarray, normals: np.ndarray, h: float) -> None:
        size = len(points)
        self.h = h
        self.lo = points.min(axis=0) - REACH * h
        ijk = np.floor((points - self.lo) / h).astype(np.int64)
        self.dims = ijk.max(axis=0) + REACH + 1                     # (padded: no offset wraps round)
        main = np.abs(normals).argmax(axis=1)
        self.side = main * 2 + (normals[np.arange(size), main] < 0)
        key = self.number(ijk[:, 0], ijk[:, 1], ijk[:, 2]) * 6 + self.side
        self.order = np.argsort(key, kind="stable")
        ordered = key[self.order]
        self.starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
        self.key = ordered[self.starts]
        self.cell = self.key // 6
        self.n = len(self.key)
        self.count = np.diff(np.r_[self.starts, len(ordered)]).astype(np.float64)
        facing = np.add.reduceat(normals[self.order], self.starts, axis=0)
        self.facing = facing / np.maximum(np.linalg.norm(facing, axis=1, keepdims=True), 1e-12)
        self.of = np.empty(size, np.int64)                          # per texel, its voxel
        self.of[self.order] = np.repeat(np.arange(self.n), self.count.astype(np.int64))

    def number(self, i, j, k):
        """A cell's number (or an offset's, which adds to it)."""
        return (i * self.dims[1] + j) * self.dims[2] + k

    def summed(self, per_texel: np.ndarray) -> np.ndarray:
        """Per voxel, the sum of its texels' rows."""
        return np.add.reduceat(per_texel[self.order], self.starts, axis=0)

    def cells(self, offset: tuple[int, int, int]) -> np.ndarray:
        """The voxels (one per set there is, up to 6) of the cell at `offset` from each voxel's -> (n, 6), -1: none."""
        want = self.cell + self.number(*offset)
        at = np.searchsorted(self.key, want * 6)[:, None] + np.arange(6)
        inside = np.minimum(at, self.n - 1)
        return np.where((at < self.n) & (self.cell[inside] == want[:, None]), inside, -1)

    def alike(self, u: np.ndarray) -> np.ndarray:
        """How alike each voxel and its (n, k) neighbours `u` face: the cosine, 0 under ALIKE and where none is."""
        cos = np.einsum("ic,ijc->ij", self.facing, self.facing[np.maximum(u, 0)])
        return np.where((u >= 0) & (cos >= ALIKE), cos, 0.0)

    def near(self, keys: np.ndarray, offset: tuple[int, int, int]) -> np.ndarray:
        """The voxel at `offset` from each of `keys` in the same set, -1 where there is none."""
        want = keys + self.number(*offset) * 6
        at = np.minimum(np.searchsorted(self.key, want), self.n - 1)
        return np.where(self.key[at] == want, at, -1)


def _blurred(voxels: _Voxels, fields: np.ndarray, wide: np.ndarray) -> np.ndarray:
    """The voxels' fields blurred over the surface, along each axis in turn, a voxel counting by its texels and by
    how alike it faces: the columns not `wide` by a Gaussian of sigma one voxel, the `wide` ones by one of two."""
    mass = np.stack([voxels.count, voxels.count], axis=1)           # the texels counted, at either width
    for axis in range(3):
        acc, weight = np.zeros_like(fields), np.zeros_like(mass)
        for o in range(-REACH, REACH + 1):
            u = voxels.cells(tuple(o if a == axis else 0 for a in range(3)))
            share = voxels.alike(u) ** 2
            uu = np.maximum(u, 0)
            beside = fields[uu]
            for part, sigma in ((~wide, 1.0), (wide, 2.0)):
                if abs(o) > 2 * sigma:
                    continue
                gauss = share * np.exp(-0.5 * (o / sigma) ** 2)
                weight[:, int(sigma) - 1] += (gauss * mass[uu, int(sigma) - 1]).sum(axis=1)
                acc[:, part] += np.einsum("ij,ijf->if", gauss.astype(np.float32), beside[:, :, part])
        fields, mass = acc, weight
    fields[:, ~wide] /= mass[:, :1].astype(np.float32)
    fields[:, wide] /= mass[:, 1:].astype(np.float32)
    return fields


def _near_a_seam(voxels: _Voxels, fields: np.ndarray, views: int) -> np.ndarray:
    """The voxels whose texels must be worked out one by one: those not one source's alone (and, a view's, all
    within its reach), and the voxels touching them - a texel's 8 voxel centres may be any of those."""
    every = np.arange(voxels.n)
    lead = fields[:, :views + 1].argmax(axis=1)
    whole = fields[every, lead] > WHOLE
    reach = fields[every, views + np.maximum(lead, 1)] > WHOLE
    mixed = ~(whole & (reach | (lead == 0)))
    band = mixed.copy()
    for o in RING:
        u = voxels.near(voxels.key, o)
        band |= (u >= 0) & mixed[np.maximum(u, 0)]
    return band


def _between_centres(voxels: _Voxels, fields: np.ndarray, points: np.ndarray,
                     todo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The fields at the texels `todo`, each between the 8 voxel centres around it (its own set's: their values
    hold the neighbouring sets' already) -> the texels in the order they were looked up in, and their fields."""
    f = (points[todo] - voxels.lo) / voxels.h - 0.5
    base = np.floor(f).astype(np.int64)
    corner_key = voxels.number(base[:, 0], base[:, 1], base[:, 2]) * 6 + voxels.side[todo]
    sort = np.argsort(corner_key, kind="stable")                    # (the look-ups in order)
    todo, f, base, corner_key = todo[sort], f[sort], base[sort], corner_key[sort]
    t = (f - base).astype(np.float32)
    along = [(1.0 - t[:, i], t[:, i]) for i in range(3)]
    total = np.zeros(todo.size, np.float32)
    here = np.zeros((todo.size, fields.shape[1]), np.float32)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                u = voxels.near(corner_key, (a, b, c))
                ok = u >= 0
                weight = along[0][a] * along[1][b] * along[2][c] * ok
                total += weight
                here += weight[:, None] * fields[np.where(ok, u, 0)]
    here /= total[:, None]
    return todo, here


def blend(points: np.ndarray, normals: np.ndarray, owners: np.ndarray, can: list[np.ndarray], radius: float,
          grain: float = 0.0) -> tuple[list[np.ndarray], np.ndarray]:
    """How much of each view's picture each texel takes, the seams blended in 3D. `owners` (n): the view each
    texel goes to (its best camera), -1 where none can paint it (the model's own texture stays); `can`: per view
    the texels it can paint (n, bool). Each source's ownership is blurred over the surface - a Gaussian reaching
    `radius` to each side - so across a seam two views' pictures cross-fade. A view cannot paint past the edge of
    what it sees: its weight falls to nothing inside that edge, over two radii (within_reach of its `can` blurred
    twice as wide), and the others - another view that sees further, else the model's own texture - take its
    share.
    Texels are gathered in voxels of half the radius (never smaller than `grain`: voxels of single texels do not
    touch); voxels are neighbours when their patches face alike (ALIKE), whatever their sets - a view's picture
    ends where the surface turns away from its camera, often just where it turns from one set to the next - and
    never when they face away from each other (the inner sides of two legs). points, normals (unit) per texel ->
    per view its weights (n, float32: exactly 1 away from every seam), and the weights left to the model's own
    texture."""
    views, size = len(can), len(points)
    voxels = _Voxels(points, normals, max(radius / 2.0, grain))
    # Per voxel, the share of its texels each source owns (the model's own texture first), then the share each
    # view can paint: the latter blurred twice as wide.
    fields = np.empty((voxels.n, 2 * views + 1), np.float32)
    fields[:, :views + 1] = np.bincount(voxels.of * (views + 1) + owners + 1,
                                        minlength=voxels.n * (views + 1)).reshape(voxels.n, views + 1)
    fields[:, views + 1:] = voxels.summed(np.stack(can, axis=1).astype(np.float32))
    fields = _blurred(voxels, fields, wide=np.arange(2 * views + 1) > views)
    band = _near_a_seam(voxels, fields, views)
    todo = np.flatnonzero(band[voxels.of])
    weights = [np.zeros(size, np.float32) for _ in range(views)]
    left = np.zeros(size, np.float32)
    plain = ~band[voxels.of]                                        # away from every seam: its owner's, whole
    left[plain & (owners < 0)] = 1.0
    for k in range(views):
        weights[k][plain & (owners == k)] = 1.0
    if todo.size == 0:
        return weights, left
    todo, here = _between_centres(voxels, fields, points, todo)
    # A view's weight: its blurred ownership, times how far inside what it can paint the texel lies; all shared
    # out again, the model's own texture with its ownership.
    able = np.stack([c[todo] for c in can], axis=1)
    part = here[:, 1:views + 1] * within_reach(here[:, views + 1:]) * able
    own = here[:, 0]
    whole = part.sum(axis=1) + own
    none = whole <= 0                                               # (nothing reaches it: it stays as it is)
    whole[none], own[none] = 1.0, 1.0
    left[todo] = own / whole
    for k in range(views):
        weights[k][todo] = part[:, k] / whole
    return weights, left
