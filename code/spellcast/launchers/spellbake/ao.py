"""Ambient occlusion at hit points on the high poly (design section 4.4, step L15).

AO is the fraction of cosine-weighted hemisphere directions around the shading normal
that leave the surface without hitting it within ``max_distance``. Directions come from
a Hammersley set of rays * subsamples points: the S x S samples of one texel take
interleaved subsets of it, so together they cover the hemisphere as one well-spread set
of 32 directions per texel. Each texel rotates the set by its own Cranley-Patterson
offset (hashed from the texel index), which turns banding into fine noise and keeps the
result reproducible. Directions below the geometric surface count as occluded.
"""

from __future__ import annotations

import numpy as np

from .rays import Scene

RAYS_PER_TEXEL = 32
# Rays cast per occlusion query batch (bounds the (n, K, 3) temporaries).
_BATCH_RAYS = 1 << 21


def rays_per_sample(supersample: int) -> int:
    """K = ceil(32 / S^2): the rays each of a texel's S x S samples casts."""
    return -(-RAYS_PER_TEXEL // (supersample * supersample))


def hammersley(count: int) -> np.ndarray:
    """(count, 2) points (i / count, radical inverse of i in base 2)."""
    i = np.arange(count, dtype=np.uint64)
    bits = i.copy()
    bits = ((bits << np.uint64(16)) | (bits >> np.uint64(16))) & np.uint64(0xFFFFFFFF)
    bits = ((bits & np.uint64(0x55555555)) << np.uint64(1)) | ((bits & np.uint64(0xAAAAAAAA)) >> np.uint64(1))
    bits = ((bits & np.uint64(0x33333333)) << np.uint64(2)) | ((bits & np.uint64(0xCCCCCCCC)) >> np.uint64(2))
    bits = ((bits & np.uint64(0x0F0F0F0F)) << np.uint64(4)) | ((bits & np.uint64(0xF0F0F0F0)) >> np.uint64(4))
    bits = ((bits & np.uint64(0x00FF00FF)) << np.uint64(8)) | ((bits & np.uint64(0xFF00FF00)) >> np.uint64(8))
    return np.stack([i / max(count, 1), bits.astype(np.float64) / 2.0 ** 32], axis=1)


def texel_rotation(texel: np.ndarray) -> np.ndarray:
    """(n, 2) offsets in [0, 1) from a 64-bit mix of the texel index (splitmix64)."""
    x = np.asarray(texel, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    low = (x & np.uint64(0xFFFFFFFF)).astype(np.float64) / 2.0 ** 32
    high = (x >> np.uint64(32)).astype(np.float64) / 2.0 ** 32
    return np.stack([low, high], axis=1)


def frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors completing each unit normal to an orthonormal basis
    (Duff et al. 2017, branchless)."""
    x, y, z = normal[:, 0], normal[:, 1], normal[:, 2]
    s = np.where(z >= 0.0, 1.0, -1.0)
    a = -1.0 / (s + z)
    b = x * y * a
    t = np.stack([1.0 + s * x * x * a, s * b, -s * x], axis=1)
    bt = np.stack([b, s + y * y * a, -y], axis=1)
    return t, bt


def cosine_directions(normal: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """(n, K, 3) cosine-weighted directions around each unit normal from (n, K, 2) points."""
    r = np.sqrt(xi[..., 0])
    phi = 2.0 * np.pi * xi[..., 1]
    z = np.sqrt(np.maximum(0.0, 1.0 - xi[..., 0]))
    t, b = frame(normal)
    return ((r * np.cos(phi))[..., None] * t[:, None] + (r * np.sin(phi))[..., None] * b[:, None]
            + z[..., None] * normal[:, None])


def ambient_occlusion(
    scene: Scene,
    points: np.ndarray,
    normals: np.ndarray,
    geometric_normals: np.ndarray,
    texel: np.ndarray,
    sub: np.ndarray,
    *,
    supersample: int,
    max_distance: float,
) -> np.ndarray:
    """(n,) float32 unoccluded fraction at each point.

    ``normals`` are the unit shading normals the hemisphere is built around,
    ``geometric_normals`` the unit normals of the hit triangles (rays start 1e-4 * diag
    off the surface along them, and directions below them count as occluded);
    ``texel`` and ``sub`` are the raster.SampleBand fields of each point."""
    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    rays = rays_per_sample(supersample)
    subsamples = supersample * supersample
    pattern = hammersley(rays * subsamples)
    open_fraction = np.empty(count, dtype=np.float32)
    step = max(1, _BATCH_RAYS // rays)
    for start in range(0, count, step):
        part = slice(start, min(start + step, count))
        n = normals[part]
        g = geometric_normals[part]
        pick = np.asarray(sub[part], dtype=np.int64)[:, None] % subsamples + subsamples * np.arange(rays)
        xi = np.mod(pattern[pick] + texel_rotation(texel[part])[:, None, :], 1.0)
        directions = cosine_directions(n, xi)
        above = np.einsum("nkd,nd->nk", directions, g) > 0
        origin = points[part] + (1e-4 * scene.diag) * g
        hit = np.ones(above.shape, dtype=bool)
        rows, cols = np.nonzero(above)
        if rows.size:
            hit[rows, cols] = scene.occluded(origin[rows], directions[rows, cols], max_distance)
        open_fraction[part] = (~hit).sum(axis=1) / rays
    return open_fraction
