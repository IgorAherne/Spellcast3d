"""MikkTSpace tangents, vectorised (design section 4.4, step L7).

A numpy port of mikktspace.c (genTangSpaceDefault): per triangle the UV-derived
direction vOs = sign(A_uv) * normalize(dt3.v * d1 - dt2.v * d2) (InitTriInfo, the
eq. 18/19 terms), projected at each corner onto the plane of that vertex's normal and
summed with the corner angle between the projected edges as weight (EvalTspace). The
sum is kept apart per UV orientation, which is how MikkTSpace splits a vertex shared by
a mirrored and an unmirrored triangle, and the sign w is +1 for positively oriented
triangles, -1 for mirrored ones.

Run on v-up UVs with w exported unchanged, this equals glTF's (v-down) MikkTSpace with
w negated: checked against the mikktspace 1.1.1 wasm (tests/bake/data/mikk_reference.json).
Where it differs from the C code: triangles around a vertex are summed per orientation
rather than per edge-connected fan, so the two fans of a bow-tie vertex share a tangent.

Triangles with (almost) no UV area add nothing; their corners take the tangent their
vertex gets from its other triangles, or any direction perpendicular to the normal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# |(t2 - t1) x (t3 - t1)| below this (v-up UV units) means the triangle has no UV area.
DEGENERATE_UV_AREA = 1e-12


@dataclass(frozen=True)
class Tangents:
    corners: np.ndarray    # (nF, 3, 4) per face corner: unit tangent xyz, sign w (+-1)
    degenerate: int        # triangles with no UV area (they borrow their vertices' tangents)
    fallback: int          # corners that had no tangent at all: any direction perpendicular to N

    def per_vertex(self, faces: np.ndarray, count: int) -> np.ndarray:
        """(count, 4) tangents for a mesh whose vertices are already split by UV orientation,
        so all corners of a vertex agree. Vertices no face uses get (1, 0, 0, 1)."""
        out = np.tile(np.array([1.0, 0.0, 0.0, 1.0]), (count, 1))
        out[np.asarray(faces).reshape(-1)] = self.corners.reshape(-1, 4)
        return out


def uv_area2(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Signed doubled UV area (t2 - t1) x (t3 - t1) of every triangle; > 0 = not mirrored."""
    t = np.asarray(uv, dtype=np.float64)[faces]
    t21, t31 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    return t21[:, 0] * t31[:, 1] - t21[:, 1] * t31[:, 0]


def normalize(v: np.ndarray) -> np.ndarray:
    """Unit vectors along the last axis; zero vectors stay zero."""
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, length, out=np.zeros_like(v), where=length > 0)


def _project(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """v minus its component along the unit vector n (MikkTSpace's projection onto the
    plane of the vertex normal)."""
    return v - n * np.einsum("ij,ij->i", n, v)[:, None]


def any_perpendicular(n: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to each unit n (the axis least aligned with n, crossed)."""
    axis = np.zeros_like(n)
    axis[np.arange(len(n)), np.argmin(np.abs(n), axis=1)] = 1.0
    return normalize(np.cross(n, axis))


def mikk_tangents(
    positions: np.ndarray, normals: np.ndarray, uv: np.ndarray, faces: np.ndarray
) -> Tangents:
    """Per-corner MikkTSpace tangents of an indexed triangle mesh.

    ``normals`` are the unit vertex normals the mesh is exported and shaded with, ``uv``
    the v-up texture coordinates; both are indexed like ``positions`` by ``faces``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    count = len(positions)

    p = positions[faces]
    t = np.asarray(uv, dtype=np.float64)[faces]
    d1, d2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    t21, t31 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    area = t21[:, 0] * t31[:, 1] - t21[:, 1] * t31[:, 0]
    positive = area > 0
    sign = np.where(positive, 1.0, -1.0)
    v_os = t31[:, 1:2] * d1 - t21[:, 1:2] * d2
    v_os_len = np.linalg.norm(v_os, axis=1)
    valid = (np.abs(area) >= DEGENERATE_UV_AREA) & (v_os_len > 0)
    v_os = np.divide(v_os * sign[:, None], v_os_len[:, None], out=np.zeros_like(v_os),
                     where=valid[:, None])

    # Accumulate angle-weighted projected directions per (vertex, orientation).
    rows = np.flatnonzero(valid)
    acc = np.zeros((2 * count, 3))
    for c in range(3):
        vertex = faces[rows, c]
        n = normals[vertex]
        o = normalize(_project(v_os[rows], n))
        e1 = normalize(_project(p[rows, (c + 1) % 3] - p[rows, c], n))
        e2 = normalize(_project(p[rows, (c + 2) % 3] - p[rows, c], n))
        angle = np.arccos(np.clip(np.einsum("ij,ij->i", e1, e2), -1.0, 1.0))
        key = 2 * vertex + positive[rows]
        for axis in range(3):
            acc[:, axis] += np.bincount(key, weights=o[:, axis] * angle, minlength=2 * count)
    has = np.linalg.norm(acc, axis=1) > 0

    # Every corner reads its own orientation's sum; a corner of a triangle without UV area
    # reads whichever its vertex has (the positive one when both or neither exist).
    vertex = faces.reshape(-1)
    own = np.repeat(positive, 3).astype(np.int64)
    borrow = np.where(has[2 * vertex] & ~has[2 * vertex + 1], 0, 1)
    orient = np.where(np.repeat(valid, 3), own, borrow)
    tangent = normalize(acc[2 * vertex + orient])
    w = np.where(orient == 1, 1.0, -1.0)

    missing = np.linalg.norm(tangent, axis=1) == 0
    if missing.any():
        tangent[missing] = any_perpendicular(normals[vertex[missing]])
    corners = np.concatenate([tangent, w[:, None]], axis=1).reshape(-1, 3, 4)
    return Tangents(corners, int((~valid).sum()), int(missing.sum()))
