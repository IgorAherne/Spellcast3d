"""Tangent-space normal encoding (design section 4.4, step L13), and the matching decoders.

A baked texel stores the unit vector (x, y, z) that the consumer's shader turns back
into the world normal n_w. Two consumers exist, and each encoder is the exact inverse of
its decoder:

- ``mikk`` (per-fragment MikkTSpace: Blender, Unreal, Unity HDRP, Substance). The shader
  interpolates vN and vT without normalising them and builds vB = w * (vN x vT); the
  world normal is normalize(x vT + y vB + z vN). With M = [vT | vB | vN] the encoder is
  normalize(adj(M) n_w / det M): the rows of adj(M) are vB x vN, vN x vT, vT x vB. Dividing
  by det keeps mirrored charts (w = -1) right, and normalising is allowed because the
  decoder normalises too.
- ``per_vertex`` (three.js r169/r186, Unity URP, xNormal). Per vertex n = normalize(N),
  t = normalize(T), b = normalize(w * (n x t)); these three are interpolated and each
  renormalised per fragment into T', B', N', and the world normal is
  normalize(x T' + y B' + z N'). The frame is not orthogonal, so the encoder solves the
  3 x 3 system (a transpose would cost ~0.3 degrees at p99).

When a frame is (nearly) singular it is orthonormalised first. Stored normals are then
clamped to the outer hemisphere (z >= 0) and quantised as round(255 * (n / 2 + 1 / 2)),
OpenGL +Y.
"""

from __future__ import annotations

import numpy as np

from .tangents import any_perpendicular, normalize

MIKK = "mikk"
PER_VERTEX = "per_vertex"
BASES = (MIKK, PER_VERTEX)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def _orthonormal(tangent: np.ndarray, normal: np.ndarray, sign: np.ndarray) -> tuple[np.ndarray, ...]:
    """Gram-Schmidt frame (T, w * N x T, N) for frames too degenerate to invert."""
    n = normalize(normal)
    n[np.linalg.norm(n, axis=1) == 0] = (0.0, 0.0, 1.0)
    t = normalize(tangent - n * _dot(tangent, n)[:, None])
    lost = np.linalg.norm(t, axis=1) == 0
    if lost.any():
        t[lost] = any_perpendicular(n[lost])
    return t, sign[:, None] * np.cross(n, t), n


def _solve(c0: np.ndarray, c1: np.ndarray, c2: np.ndarray, target: np.ndarray,
           sign: np.ndarray) -> np.ndarray:
    """normalize(x) with [c0 | c1 | c2] x = target, by the adjugate over the determinant."""
    r0, r1, r2 = np.cross(c1, c2), np.cross(c2, c0), np.cross(c0, c1)
    det = _dot(c0, r0)
    scale = np.linalg.norm(c0, axis=1) * np.linalg.norm(c1, axis=1) * np.linalg.norm(c2, axis=1)
    x = np.stack([_dot(r0, target), _dot(r1, target), _dot(r2, target)], axis=1)
    ok = np.abs(det) > 1e-8 * scale
    x[ok] /= det[ok, None]
    bad = np.flatnonzero(~ok)
    if bad.size:
        t, b, n = _orthonormal(c0[bad], c2[bad], sign[bad])
        x[bad] = np.stack([_dot(t, target[bad]), _dot(b, target[bad]), _dot(n, target[bad])], axis=1)
    return normalize(x)


def mikk_bitangent(tangent: np.ndarray, normal: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """vB = w * (vN x vT), from the unnormalised interpolants."""
    return np.asarray(sign, dtype=np.float64)[:, None] * np.cross(normal, tangent)


def mikk_encode(world: np.ndarray, tangent: np.ndarray, normal: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """(n, 3) unit tangent-space normals that ``mikk_decode`` turns back into ``world``.

    ``tangent`` and ``normal`` are the interpolated vT and vN (not normalised), ``sign``
    the triangle's w."""
    sign = np.asarray(sign, dtype=np.float64)
    return _solve(tangent, mikk_bitangent(tangent, normal, sign), normal, world, sign)


def mikk_decode(ts: np.ndarray, tangent: np.ndarray, normal: np.ndarray, sign: np.ndarray) -> np.ndarray:
    b = mikk_bitangent(tangent, normal, sign)
    return normalize(ts[:, 0:1] * tangent + ts[:, 1:2] * b + ts[:, 2:3] * normal)


def per_vertex_basis(normals: np.ndarray, tangents: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-vertex unit (t, b, n) as three.js's normal_vertex chunk builds them from the
    NORMAL and TANGENT attributes; interpolate these, then call ``per_vertex_encode``."""
    tangents = np.asarray(tangents, dtype=np.float64)
    n = normalize(np.asarray(normals, dtype=np.float64))
    t = normalize(tangents[:, :3])
    b = normalize(tangents[:, 3:4] * np.cross(n, t))
    return t, b, n


def per_vertex_encode(world: np.ndarray, t: np.ndarray, b: np.ndarray, n: np.ndarray,
                      sign: np.ndarray) -> np.ndarray:
    """(n, 3) unit tangent-space normals that ``per_vertex_decode`` turns back into
    ``world``. ``t``, ``b``, ``n`` are interpolated ``per_vertex_basis`` vectors; ``sign``
    (the triangle's w) only matters for a degenerate frame."""
    sign = np.asarray(sign, dtype=np.float64)
    return _solve(normalize(t), normalize(b), normalize(n), world, sign)


def per_vertex_decode(ts: np.ndarray, t: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    return normalize(ts[:, 0:1] * normalize(t) + ts[:, 1:2] * normalize(b) + ts[:, 2:3] * normalize(n))


def clamp_hemisphere(ts: np.ndarray) -> np.ndarray:
    """z <- max(z, 0), renormalised; a vector pointing straight inwards becomes (0, 0, 1)."""
    out = np.array(ts, dtype=np.float64, copy=True)
    out[:, 2] = np.maximum(out[:, 2], 0.0)
    out = normalize(out)
    out[np.linalg.norm(out, axis=1) == 0] = (0.0, 0.0, 1.0)
    return out


def quantize(ts: np.ndarray) -> np.ndarray:
    """Unit vectors -> 8-bit RGB, round(255 * (n / 2 + 1 / 2))."""
    return np.round(np.clip(np.asarray(ts) * 0.5 + 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)


def dequantize(rgb: np.ndarray) -> np.ndarray:
    """8-bit RGB -> vectors in [-1, 1] (not renormalised; decoders normalise)."""
    return np.asarray(rgb, dtype=np.float64) / 255.0 * 2.0 - 1.0
