"""Push-pull fill of the texels no sample reached (design section 4.4, step L17), after a
short dilation of every island.

Dilate (``dilate``): the covered texels grow outward by DILATE_TEXELS texels, each new
texel taking the mean of its covered 8-neighbours. The texels around a UV seam then hold
the island's own colours, so the bilinear and mip sampling across the seam reads them
and not the push-pull's coarse averages (those showed as thin lines along the seams of
the baked maps, 2026-10-20). It also writes over the ring of extrapolated samples outside
the island, whose rays may have met another surface.

Pull: build a pyramid of weighted sums, halving the size each level. Push: from the
coarsest level down, every texel without weight takes the average one level up. Texels
with weight keep their value exactly. Unlike dilating the charts by N texels alone this
leaves nothing unfilled, so no mip level of the result can pull in a black background.
"""

from __future__ import annotations

from concurrent.futures import Executor
from typing import Callable

import numpy as np

DILATE_TEXELS = 2     # how far every island grows before the push-pull fill (the artist's ask: 2 px)
# The 8 neighbours in the order a new texel's sum adds them (float32: the order is part of the result).
_NEIGHBOURS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)


def dilate(values: np.ndarray, covered: np.ndarray, radius: int = DILATE_TEXELS) -> tuple[np.ndarray, np.ndarray]:
    """(H, W, C) or (H, W) float32 values and the (H, W) bool mask of the covered texels -> the values with every
    texel within ``radius`` of a covered one filled from its covered neighbours (ring by ring: each new texel is
    the mean of its 8-neighbours covered so far), and the grown mask. The covered texels keep their values
    exactly; the rest are left as they were (the push-pull fills them)."""
    values = np.asarray(values)
    squeeze = values.ndim == 2
    vp, mp = _padded(values[..., None] if squeeze else values, covered)
    flat_v, flat_m = vp.reshape(-1, vp.shape[2]), mp.reshape(-1)
    # A new texel's sum is value x covered over its 8 neighbours in order. An uncovered neighbour adds a zero,
    # which changes no sum, unless its value is not finite (NaN x 0 is NaN): only then are those added too.
    skip_uncovered = _finite(values, vp)
    for _ in range(max(0, int(radius))):
        ring = _ring(mp)
        if ring.size == 0:
            break
        count = np.zeros(ring.size, dtype=np.float32)
        total = np.zeros((ring.size, vp.shape[2]), dtype=np.float32)
        for at in _around(ring, mp.shape[1]):
            on = np.take(flat_m, at)
            count += on
            if skip_uncovered:
                some = np.flatnonzero(on)
                total[some] += np.take(flat_v, at[some], axis=0)
            else:
                total += np.take(flat_v, at, axis=0) * on[:, None]
        flat_v[ring] = total / count[:, None]
        flat_m[ring] = True
    v = np.ascontiguousarray(vp[1:-1, 1:-1])
    return (v[..., 0] if squeeze else v), mp[1:-1, 1:-1].copy()


def dilate_at(values: np.ndarray, covered: np.ndarray, wanted: np.ndarray,
              radius: int = DILATE_TEXELS) -> tuple[np.ndarray, np.ndarray]:
    """``dilate(values, covered, radius)`` at the ``wanted`` texels only, for (H, W, C) values: the flat indices of
    the wanted texels the rings reach, in order, and their values, the same bits. Only the ring texels those
    depend on are summed (the Omni-Texture worker's holes are few, the rings around its painted texels long)."""
    values = np.asarray(values)
    h, w = np.shape(covered)
    flat = values.reshape(h * w, -1)
    if not (values.dtype.kind in "biu" or np.isfinite(flat.astype(np.float32)).all()):
        grown, reached = dilate(values, covered, radius)
        at = np.flatnonzero(reached & ~np.asarray(covered, dtype=bool) & np.asarray(wanted, dtype=bool))
        return at, grown.reshape(h * w, -1)[at]
    mp = np.zeros((h + 2, w + 2), dtype=bool)
    mp[1:-1, 1:-1] = covered
    flat_m = mp.reshape(-1)
    level = np.where(flat_m, 0, np.iinfo(np.int32).max).astype(np.int32)    # 0 covered, j + 1 reached by ring j
    rings = []
    for j in range(max(0, int(radius))):
        ring = _ring(mp)
        if ring.size == 0:
            break
        rings.append(ring)
        flat_m[ring] = True
        level[ring] = j + 1
    wp = np.zeros((h + 2, w + 2), dtype=bool)
    wp[1:-1, 1:-1] = wanted
    flat_w = wp.reshape(-1)
    # Backwards: of each ring the texels wanted and those next to the texels needed of the ring after it.
    need: list[np.ndarray] = []
    for ring in reversed(rings):
        keep = flat_w[ring]
        if need:
            near = np.zeros(mp.size, dtype=bool)
            for at in _around(need[-1], w + 2):
                near[at] = True
            keep |= near[ring]
        need.append(ring[keep])
    need.reverse()
    # Forwards, as dilate sums them: a covered neighbour's own value or the ring before's.
    found, mean = [np.zeros(0, dtype=np.int64)], [np.zeros((0, flat.shape[1]), dtype=np.float32)]
    before = before_mean = None
    for j, texels in enumerate(need):
        count = np.zeros(texels.size, dtype=np.float32)
        total = np.zeros((texels.size, flat.shape[1]), dtype=np.float32)
        for at in _around(texels, w + 2):
            round_of = level[at]
            count += round_of <= j
            own = np.flatnonzero(round_of == 0)
            total[own] += flat[_unpadded(at[own], w)].astype(np.float32)
            if j:
                grown = np.flatnonzero(round_of == j)
                total[grown] += before_mean[np.searchsorted(before, at[grown])]
        before, before_mean = texels, total / count[:, None]
        hit = flat_w[texels]
        found.append(_unpadded(texels[hit], w))
        mean.append(before_mean[hit])
    found, mean = np.concatenate(found), np.concatenate(mean)
    order = np.argsort(found)
    return found[order], mean[order]


def _unpadded(at: np.ndarray, w: int) -> np.ndarray:
    """Indices into the padded texture (``w`` + 2 per row) -> into the texture."""
    return (at // (w + 2) - 1) * w + at % (w + 2) - 1


def _padded(values: np.ndarray, covered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(H, W, C) values as float32 and the mask, with one texel of zeros around them (value 0, not covered): every
    texel has its 8 neighbours."""
    h, w = np.shape(covered)
    vp = np.zeros((h + 2, w + 2, values.shape[2]), dtype=np.float32)
    vp[1:-1, 1:-1] = values
    mp = np.zeros((h + 2, w + 2), dtype=bool)
    mp[1:-1, 1:-1] = covered
    return vp, mp


def _finite(values: np.ndarray, vp: np.ndarray) -> bool:
    return values.dtype.kind in "biu" or bool(np.isfinite(vp).all())


def _ring(mp: np.ndarray) -> np.ndarray:
    """The uncovered texels with a covered 8-neighbour (found on the padded mask alone), as indices into it."""
    h, w = mp.shape[0] - 2, mp.shape[1] - 2
    near = np.zeros((h, w), dtype=bool)
    for dy, dx in _NEIGHBOURS:
        near |= mp[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
    ring = np.flatnonzero(near & ~mp[1:-1, 1:-1])
    return ring + (ring // w) * 2 + w + 3


def _around(ring: np.ndarray, width: int) -> list[np.ndarray]:
    """The 8 neighbours of the padded texels ``ring`` (``width`` per row), in order."""
    return [ring + (dy * width + dx) for dy, dx in _NEIGHBOURS]


def _pad_even(a: np.ndarray) -> np.ndarray:
    """Zero-pad the first two axes to even sizes (a padded texel has no weight)."""
    pad_y, pad_x = a.shape[0] % 2, a.shape[1] % 2
    if not (pad_y or pad_x):
        return a
    return np.pad(a, [(0, pad_y), (0, pad_x)] + [(0, 0)] * (a.ndim - 2))


def _levels(a: np.ndarray, pool: Executor | None = None) -> list[np.ndarray]:
    """The pull's pyramid: ``a`` and its sums over 2 x 2 texels, level by level, up to one texel. With a pool,
    values of two or more channels are summed in bands of rows on its threads, the same bits (_sum4)."""
    levels = [a]
    while levels[-1].shape[0] > 1 or levels[-1].shape[1] > 1:
        s = _pad_even(levels[-1])
        h, wd = s.shape[0] // 2, s.shape[1] // 2
        if pool is not None and s.ndim == 3 and s.shape[2] >= 2:
            out = np.empty((h, wd, s.shape[2]), dtype=s.dtype)
            bands(pool, h, lambda lo, hi, s=s, out=out: _sum4(s[2 * lo:2 * hi], out[lo:hi]))
            levels.append(out)
        else:
            levels.append(s.reshape(h, 2, wd, 2, *s.shape[2:]).sum(axis=(1, 3)))
    return levels


def _sum4(s: np.ndarray, out: np.ndarray) -> None:
    """(2h, 2w, C) values, C of 2 or more -> `out` (h, w, C): their sums over 2 x 2 texels in the order numpy's
    reduce adds them there, ((top left + top right) + bottom left) + bottom right: the same bits (with one channel
    numpy pairs them otherwise; tests/bake check the order)."""
    np.add(s[0::2, 0::2], s[0::2, 1::2], out=out)
    np.add(out, s[1::2, 0::2], out=out)
    np.add(out, s[1::2, 1::2], out=out)


BANDS = 8                   # the bands of rows a pool's threads take (elementwise steps: the same bits as one piece)


def bands(pool: Executor, rows: int, work: Callable[[int, int], object], even: bool = False) -> None:
    """work(lo, hi) for bands of `rows` rows on the pool's threads (each band's first row even with `even`), waiting
    for all of them (an error in one is raised here)."""
    step = max(1, -(-rows // BANDS))
    if even:
        step += step % 2
    cuts = list(range(0, rows, step)) + [rows]
    for done in [pool.submit(work, lo, hi) for lo, hi in zip(cuts[:-1], cuts[1:])]:
        done.result()


def push_pull(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(H, W, C) float32 with every texel of zero weight filled from its surroundings.

    ``weights`` (H, W) are >= 0; use the coverage mask (0/1), not sample counts, so the
    fill is an unbiased average of the covered texels. With no weight anywhere the values
    come back unchanged."""
    values = np.asarray(values, dtype=np.float32)
    squeeze = values.ndim == 2
    if squeeze:
        values = values[..., None]
    w = np.asarray(weights, dtype=np.float32)
    if not (w > 0).any():
        return values[..., 0] if squeeze else values.copy()
    out = _push_pull(values, w, _levels(w))
    return out[..., 0] if squeeze else out


def _push_pull(values: np.ndarray, w: np.ndarray, counts: list[np.ndarray],
               pool: Executor | None = None) -> np.ndarray:
    """push_pull with the pyramid of the weights ``w`` given. With a pool (the Omni-Texture worker's, after every
    view) every step runs in bands of rows on its threads: elementwise, and the pull's sums in numpy's own order, so
    the same bits."""
    if pool is not None and values.ndim == 3 and len(counts) > 1:
        return _push_pull_bands(values, w, counts, pool)
    sums = _levels(values * w[..., None])
    current = sums[-1] / np.maximum(counts[-1], 1e-30)[..., None]
    if len(sums) == 1:
        return np.where((w > 0)[..., None], values, current).astype(np.float32)
    for s, c in zip(reversed(sums[1:-1]), reversed(counts[1:-1])):
        up = np.repeat(np.repeat(current, 2, axis=0), 2, axis=1)[: s.shape[0], : s.shape[1]]
        own = s / np.maximum(c, 1e-30)[..., None]
        current = np.where((c > 0)[..., None], own, up)
    # The finest level's weights are ``w``: a texel with weight keeps its value, the others take the level above's.
    h1, w1, channels = current.shape
    up = np.broadcast_to(current[:, None, :, None], (h1, 2, w1, 2, channels)).reshape(2 * h1, 2 * w1, channels)
    return np.where((w > 0)[..., None], values, up[: values.shape[0], : values.shape[1]])


def _push_pull_bands(values: np.ndarray, w: np.ndarray, counts: list[np.ndarray], pool: Executor) -> np.ndarray:
    """_push_pull's steps, each in bands of rows on the pool's threads (a band's rows from the level above's)."""
    weighted = np.empty(values.shape, dtype=np.result_type(values.dtype, w.dtype))

    def weigh(lo: int, hi: int) -> None:
        np.multiply(values[lo:hi], w[lo:hi, :, None], out=weighted[lo:hi])

    bands(pool, values.shape[0], weigh)
    sums = _levels(weighted, pool)
    current = sums[-1] / np.maximum(counts[-1], 1e-30)[..., None]
    for s, c in zip(reversed(sums[1:-1]), reversed(counts[1:-1])):
        above = current
        current = np.empty(s.shape, dtype=np.result_type(s.dtype, above.dtype))

        def push(lo: int, hi: int, s: np.ndarray = s, c: np.ndarray = c, above: np.ndarray = above,
                 out: np.ndarray = current) -> None:
            up = np.repeat(np.repeat(above[lo // 2:(hi + 1) // 2], 2, axis=0), 2, axis=1)[: hi - lo, : s.shape[1]]
            own = s[lo:hi] / np.maximum(c[lo:hi], 1e-30)[..., None]
            out[lo:hi] = np.where((c[lo:hi] > 0)[..., None], own, up)

        bands(pool, s.shape[0], push, even=True)
    result = np.empty(values.shape, dtype=np.result_type(values.dtype, current.dtype))

    def finest(lo: int, hi: int) -> None:
        part = current[lo // 2:(hi + 1) // 2]
        h1, w1, channels = part.shape
        up = np.broadcast_to(part[:, None, :, None], (h1, 2, w1, 2, channels)).reshape(2 * h1, 2 * w1, channels)
        result[lo:hi] = np.where((w[lo:hi] > 0)[..., None], values[lo:hi], up[: hi - lo, : values.shape[1]])

    bands(pool, values.shape[0], finest, even=True)
    return result


class Fill:
    """``push_pull(*dilate(values, covered, radius))`` for many (H, W, C) values over one mask (the Omni-Texture
    worker fills the gutter after every view): the rings, each new texel's covered neighbours and the weights'
    pyramid are found once. The same values, bit for bit."""

    def __init__(self, covered: np.ndarray, radius: int = DILATE_TEXELS) -> None:
        self.covered = np.array(covered, dtype=bool, copy=True)
        self.radius = radius
        h, w = self.covered.shape
        mp = np.zeros((h + 2, w + 2), dtype=bool)
        mp[1:-1, 1:-1] = self.covered
        flat_m = mp.reshape(-1)
        # Per ring: its texels, how many covered neighbours each has, and per rank j the j-th of them (in order),
        # the texels with the most first so that the j-th neighbours of all that have one are a run.
        self.rings: list[tuple[np.ndarray, np.ndarray, list[np.ndarray]]] = []
        for _ in range(max(0, int(radius))):
            ring = _ring(mp)
            if ring.size == 0:
                break
            count = np.zeros(ring.size, dtype=np.int8)
            for at in _around(ring, mp.shape[1]):
                count += np.take(flat_m, at)
            ring, count = ring[np.argsort(-count, kind="stable")], np.sort(count)[::-1]
            ranked = np.zeros((len(_NEIGHBOURS), ring.size), dtype=np.int64)
            rank = np.zeros(ring.size, dtype=np.int64)
            for at in _around(ring, mp.shape[1]):
                some = np.flatnonzero(np.take(flat_m, at))
                ranked[rank[some], some] = at[some]
                rank[some] += 1
            runs = [ranked[j, :np.count_nonzero(count > j)].copy() for j in range(int(count[0]))]
            self.rings.append((ring, count.astype(np.float32), runs))
            flat_m[ring] = True
        self.weights = mp[1:-1, 1:-1].astype(np.float32)
        self.counts = _levels(self.weights)

    def __call__(self, values: np.ndarray, pool: Executor | None = None) -> np.ndarray:
        """pool: each step in bands on its threads (the same bits)."""
        values = np.asarray(values)
        if pool is not None and values.ndim == 3 and values.dtype.kind in "biu":     # whole numbers: all finite
            h, w = self.covered.shape
            vp = np.zeros((h + 2, w + 2, values.shape[2]), dtype=np.float32)

            def pad(lo: int, hi: int) -> None:
                vp[1 + lo:1 + hi, 1:-1] = values[lo:hi]

            bands(pool, h, pad)
        else:
            vp, _ = _padded(values, self.covered)
            if not _finite(values, vp):             # NaN x 0 is NaN: the uncovered neighbours count
                return push_pull(*dilate(values, self.covered, self.radius))
        flat_v = vp.reshape(-1, vp.shape[2])
        for ring, count, runs in self.rings:
            total = np.zeros((ring.size, vp.shape[2]), dtype=np.float32)

            # A ring's texels read only texels covered before it, never each other: its parts are independent.
            def grow(lo: int, hi: int, ring: np.ndarray = ring, count: np.ndarray = count,
                     runs: list[np.ndarray] = runs, total: np.ndarray = total) -> None:
                for run in runs:                    # the longest first
                    if run.size <= lo:
                        break
                    end = min(hi, run.size)
                    total[lo:end] += np.take(flat_v, run[lo:end], axis=0)
                flat_v[ring[lo:hi]] = total[lo:hi] / count[lo:hi, None]

            if pool is None:
                grow(0, ring.size)
            else:
                bands(pool, ring.size, grow)
        grown = vp[1:-1, 1:-1]
        if not (self.weights > 0).any():
            return grown.copy()
        return _push_pull(grown, self.weights, self.counts, pool)


def push_pull_unit(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``push_pull`` for unit vectors (tangent-space normals): the filled vectors are
    averaged as vectors and renormalised; a direction that averages out becomes (0, 0, 1)."""
    out = push_pull(vectors, weights)
    length = np.linalg.norm(out, axis=-1, keepdims=True)
    out = np.divide(out, length, out=np.zeros_like(out), where=length > 0)
    out[length[..., 0] == 0] = (0.0, 0.0, 1.0)
    return out
