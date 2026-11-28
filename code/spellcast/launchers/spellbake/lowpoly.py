"""The low poly a bake writes onto (design section 4.4, steps L1-L7 and L9).

``prepare()`` turns the retopo OBJ (or a GLB) into the triangles the bake samples and
the file exports, in this order:

- L1 read, fan-triangulate each polygon (0, k-1, k) as instant_meshes_brush/uv.py and
  three.js's OBJLoader do, and drop triangles without UVs, without area, or repeated;
- L2 fix the winding: every orientation-consistent component is voted against the high
  poly (Instant Meshes output has ~6% of its triangles wound inwards);
- L3 cage normals C: angle-weighted, welded by position, so continuous everywhere;
- L4 export normals N: smooth, except across edges that are both a UV seam and sharper
  than ``hard_edge_angle``;
- L5 optionally repack the UV islands with xatlas so they sit ``max(4, R // 256)``
  texels apart (the retopo atlas has no padding, which bleeds at small mip levels);
- L6 split render vertices by (position, UV, normal group, UV orientation);
- L7 MikkTSpace tangents on them.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from . import BakeError
from .raster import sample_bands
from .rays import Scene
from .tangents import DEGENERATE_UV_AREA, mikk_tangents, normalize, uv_area2

DEFAULT_HARD_EDGE_ANGLE = 60.0
# Extrapolation ring (raster.sample_bands) for repacked islands, and for the retopo's own
# unpadded atlas, where it has to reach further to keep small mip levels clean.
RING_REPACKED = 4.0
RING_ORIGINAL = 8.0
# A repack may move, turn and scale UV islands but not change their shape.
REPACK_AREA_TOLERANCE = 0.01
# min_chart_gap() stops counting here (the value then means "at least this").
MAX_MEASURED_GAP = 16


@dataclass
class LowInput:
    """Triangles over positions and texture coordinates, as read from a file."""

    positions: np.ndarray               # (nP, 3) float64
    uvs: np.ndarray                     # (nT, 2) float64, v up
    tri_pos: np.ndarray                 # (nF, 3) int64 into positions
    tri_uv: np.ndarray                  # (nF, 3) int64 into uvs; -1 where the file gave none
    normals: np.ndarray | None = None   # (nT, 3) authored unit normals indexed like uvs (GLB)
    polygons: int = 0                   # faces in the file, before triangulation


@dataclass(frozen=True)
class Surface:
    """The low poly at a batch of samples (L9)."""

    position: np.ndarray    # (n, 3) P
    cage: np.ndarray        # (n, 3) unit cage normal C
    normal: np.ndarray      # (n, 3) interpolated export normal vN, not normalised
    tangent: np.ndarray     # (n, 3) interpolated tangent vT, not normalised
    sign: np.ndarray        # (n,) the triangle's tangent sign w
    footprint: np.ndarray   # (n,) world size of one sample, sqrt(A_3D / A_uv) / (R * S)


@dataclass
class RenderMesh:
    """Split vertices as a GLB stores them; the bake samples exactly these triangles."""

    positions: np.ndarray     # (nV, 3) float64
    uv: np.ndarray            # (nV, 2) v up
    normals: np.ndarray       # (nV, 3) unit export normals N
    cage: np.ndarray          # (nV, 3) unit cage normals C
    tangents: np.ndarray      # (nV, 4) MikkTSpace xyz and sign w
    faces: np.ndarray         # (nF, 3) int64
    position_id: np.ndarray   # (nV,) the welded position each vertex sits on

    @cached_property
    def sign(self) -> np.ndarray:
        """(nF,) tangent sign w of each triangle (the same at its three corners)."""
        return self.tangents[self.faces[:, 0], 3]

    @cached_property
    def world_per_uv(self) -> np.ndarray:
        """(nF,) sqrt(A_3D / A_uv): world length per unit of UV (inf without UV area)."""
        p = self.positions[self.faces]
        area_3d = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
        area_uv = np.abs(uv_area2(self.uv, self.faces))
        with np.errstate(divide="ignore"):
            return np.sqrt(area_3d / area_uv)

    def surface(self, tri: np.ndarray, bary: np.ndarray, samples_per_side: int) -> Surface:
        """P, C, vN, vT, w and the sample footprint at barycentrics ``bary`` of triangles
        ``tri``, for a map of ``samples_per_side`` = R * S samples across."""
        f = self.faces[tri]
        return Surface(
            position=np.einsum("nk,nkd->nd", bary, self.positions[f]),
            cage=normalize(np.einsum("nk,nkd->nd", bary, self.cage[f])),
            normal=np.einsum("nk,nkd->nd", bary, self.normals[f]),
            tangent=np.einsum("nk,nkd->nd", bary, self.tangents[f, :3]),
            sign=self.sign[tri],
            footprint=self.world_per_uv[tri] / samples_per_side,
        )

    def probes(self) -> tuple[np.ndarray, np.ndarray]:
        """Points and unit directions that measure how far the high poly is (rays.auto_distance):
        every welded vertex along its cage normal, and every triangle centroid along the
        cage normal interpolated there."""
        _, first = np.unique(self.position_id, return_index=True)
        centroid_dir = normalize(self.cage[self.faces].mean(axis=1))
        points = np.concatenate([self.positions[first], self.positions[self.faces].mean(axis=1)])
        return points, np.concatenate([self.cage[first], centroid_dir])


@dataclass
class LowPoly:
    mesh: RenderMesh
    repacked: bool
    stats: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ring(self) -> float:
        """The extrapolation ring (texels) this layout needs."""
        return RING_REPACKED if self.repacked else RING_ORIGINAL


# ---------------------------------------------------------------------- helpers


def _components(count: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected components of ``count`` nodes joined by the pairs (a[i], b[i]); returns a
    label in [0, number of components) per node. Union by hooking roots, then full path
    compression, so it takes O(log n) numpy rounds."""
    parent = np.arange(count)
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    while a.size:
        ra, rb = parent[a], parent[b]
        apart = ra != rb
        if not apart.any():
            break
        a, b, ra, rb = a[apart], b[apart], ra[apart], rb[apart]
        np.minimum.at(parent, np.maximum(ra, rb), np.minimum(ra, rb))
        while True:
            hop = parent[parent]
            if np.array_equal(hop, parent):
                break
            parent = hop
    return np.unique(parent, return_inverse=True)[1].reshape(-1)


def _face_frames(positions: np.ndarray, tri_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit face normals (nF, 3), doubled 3D areas (nF,) and corner angles (nF, 3)."""
    p = positions[tri_pos]
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    angles = np.empty((len(tri_pos), 3))
    for c in range(3):
        e1 = p[:, (c + 1) % 3] - p[:, c]
        e2 = p[:, (c + 2) % 3] - p[:, c]
        angles[:, c] = np.arctan2(np.linalg.norm(np.cross(e1, e2), axis=1), np.einsum("ij,ij->i", e1, e2))
    return normalize(cross), area2, angles


def _angle_weighted(count: int, corner_key: np.ndarray, face_normals: np.ndarray,
                    angles: np.ndarray) -> np.ndarray:
    """One unit normal per key (count, 3): normalize(sum of angle * face normal) over the
    corners (``corner_key``, (nF, 3)) that share it."""
    acc = np.zeros((count, 3))
    weights = face_normals[:, None, :] * angles[:, :, None]
    key = corner_key.reshape(-1)
    for axis in range(3):
        acc[:, axis] = np.bincount(key, weights=weights[..., axis].reshape(-1), minlength=count)
    out = normalize(acc)
    # A key whose normals cancel out takes the normal of one of its faces.
    lost = np.linalg.norm(out, axis=1) == 0
    if lost.any():
        fallback = np.zeros((count, 3))
        fallback[key] = np.repeat(face_normals, 3, axis=0)
        out[lost] = fallback[lost]
    return out


def _shared_edges(tri_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Consecutive pairs of triangle edges over the same two positions.

    Edge 3 * t + k runs from corner k to corner k + 1 of triangle t. Returns
    (edge_a, edge_b, manifold): manifold marks pairs whose edge has exactly two triangles.
    """
    start = tri_pos.reshape(-1)
    end = np.roll(tri_pos, -1, axis=1).reshape(-1)
    key = np.minimum(start, end) * (int(tri_pos.max()) + 1) + np.maximum(start, end)
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    same = np.flatnonzero(sorted_key[1:] == sorted_key[:-1])
    run_start = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
    run_length = np.diff(np.r_[run_start, len(key)])
    length_of = np.repeat(run_length, run_length)
    return order[same], order[same + 1], length_of[same] == 2


# ---------------------------------------------------------------------- L1: read


def _obj_index(token: str, count: int, line: int, what: str) -> int:
    try:
        value = int(token)
    except ValueError as exc:
        raise BakeError(f"line {line}: '{token}' is not a {what} number") from exc
    if value == 0:
        raise BakeError(f"line {line}: {what} numbers start at 1, not 0")
    return value - 1 if value > 0 else count + value


def read_obj(path: str | Path) -> LowInput:
    """v / vt / f of an OBJ (negative indices, polygons, faces without vt); vn is ignored,
    the bake computes its own normals. Polygons fan out as (0, k-1, k).

    Exact copies of a point are welded, as read_glb does: Instant Meshes can write one point
    twice under two indices, which would split the cage normals there (L3) and hide a
    triangle repeated over them from clean() (L1)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise BakeError(f"Cannot read the retopo mesh {path.name}: {exc}.") from exc
    text = text.replace("\\\r\n", " ").replace("\\\n", " ")

    positions: list[list[str]] = []
    uvs: list[list[str]] = []
    tri_pos: list[tuple[int, int, int]] = []
    tri_uv: list[tuple[int, int, int]] = []
    polygons = 0
    try:
        for number, line in enumerate(text.splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v":
                positions.append(parts[1:4])
            elif tag == "vt":
                uvs.append((parts[1:3] + ["0"])[:2])
            elif tag == "f":
                ring_p, ring_t = [], []
                for token in parts[1:]:
                    fields = token.split("/")
                    ring_p.append(_obj_index(fields[0], len(positions), number, "vertex"))
                    has_uv = len(fields) > 1 and fields[1] != ""
                    ring_t.append(_obj_index(fields[1], len(uvs), number, "texture coordinate")
                                  if has_uv else -1)
                polygons += 1
                for k in range(2, len(ring_p)):
                    tri_pos.append((ring_p[0], ring_p[k - 1], ring_p[k]))
                    tri_uv.append((ring_t[0], ring_t[k - 1], ring_t[k]))
        points = np.array(positions, dtype=np.float64).reshape(-1, 3)
        coords = np.array(uvs, dtype=np.float64).reshape(-1, 2)
    except (BakeError, ValueError) as exc:
        raise BakeError(f"The retopo mesh {path.name} could not be read ({exc}). "
                        f"Run Retopo again to rewrite it.") from exc

    faces = np.array(tri_pos, dtype=np.int64).reshape(-1, 3)
    corners = np.array(tri_uv, dtype=np.int64).reshape(-1, 3)
    if faces.size == 0:
        raise BakeError(f"The retopo mesh {path.name} has no faces. Run Retopo again.")
    if faces.min() < 0 or faces.max() >= len(points) or corners.max() >= len(coords) or (
            (corners < -1).any()):
        raise BakeError(f"The retopo mesh {path.name} refers to vertices it does not contain. "
                        f"Run Retopo again to rewrite it.")
    # Welded points keep the order they first appear in, so a clean file keeps its numbering.
    _, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return LowInput(points[first[order]], coords, rank[inverse.reshape(-1)][faces], corners,
                    polygons=polygons)


_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_COMPONENTS = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32,
               5126: np.float32}
_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_COMPRESSED = ("KHR_draco_mesh_compression", "EXT_meshopt_compression")


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    out = np.eye(4)
    out[:3, :3] = rotation * np.asarray(node.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)
    out[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
    return out


def read_glb(path: str | Path) -> LowInput:
    """The triangles of every mesh in a GLB's default scene, in world space.

    Positions are welded by exact value; every GLB vertex is its own texture coordinate,
    and its NORMAL is kept when all primitives have one, so baking a model onto itself
    reproduces its own shading exactly.
    """
    path = Path(path)
    try:
        data = path.read_bytes()
        magic, version, _ = struct.unpack_from("<III", data, 0)
        if magic != _GLB_MAGIC or version != 2:
            raise ValueError("not a glTF 2.0 binary file")
        gltf, binary, offset = None, b"", 12
        while offset + 8 <= len(data):
            length, kind = struct.unpack_from("<II", data, offset)
            chunk = data[offset + 8:offset + 8 + length]
            if kind == _CHUNK_JSON:
                gltf = json.loads(chunk)
            elif kind == _CHUNK_BIN and not binary:
                binary = chunk
            offset += 8 + length
        if gltf is None:
            raise ValueError("the file has no JSON chunk")
    except (OSError, ValueError, struct.error) as exc:
        raise BakeError(f"Cannot read the low poly {path.name}: {exc}.") from exc
    compressed = [name for name in gltf.get("extensionsRequired", []) if name in _COMPRESSED]
    if compressed:
        raise BakeError(f"The low poly {path.name} uses {compressed[0]}, which the bake cannot "
                        f"read. Export it again without mesh compression.")

    def accessor(index: int) -> np.ndarray:
        acc = gltf["accessors"][index]
        if "sparse" in acc or "bufferView" not in acc:
            raise BakeError(f"The low poly {path.name} uses sparse accessors, which the bake "
                            f"cannot read. Export it again as a plain GLB.")
        view = gltf["bufferViews"][acc["bufferView"]]
        if view.get("buffer", 0) != 0:
            raise BakeError(f"The low poly {path.name} keeps data in external files. "
                            f"Export it as a single self-contained GLB.")
        dtype = np.dtype(_COMPONENTS[acc["componentType"]])
        width = _WIDTHS[acc["type"]]
        stride = view.get("byteStride") or dtype.itemsize * width
        raw = np.ndarray((acc["count"], width), dtype=dtype, buffer=binary,
                         offset=view.get("byteOffset", 0) + acc.get("byteOffset", 0),
                         strides=(stride, dtype.itemsize))
        out = raw.astype(np.float64)
        if acc.get("normalized") and dtype.kind in "iu":
            out = np.maximum(out / np.iinfo(dtype).max, -1.0)
        return out

    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes", [])
    if scenes:
        roots = scenes[gltf.get("scene", 0)].get("nodes", [])
    else:
        children = {c for node in nodes for c in node.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]

    parts_pos, parts_uv, parts_nrm, parts_faces, parts_textured = [], [], [], [], []
    base = 0
    stack = [(i, np.eye(4)) for i in roots]
    while stack:
        index, parent = stack.pop()
        node = nodes[index]
        world = parent @ _node_matrix(node)
        stack.extend((c, world) for c in node.get("children", []))
        if "mesh" not in node:
            continue
        linear = world[:3, :3]
        normal_matrix = np.linalg.inv(linear).T
        for prim in gltf["meshes"][node["mesh"]]["primitives"]:
            if prim.get("mode", 4) != 4 or "POSITION" not in prim["attributes"]:
                continue
            attributes = prim["attributes"]
            pos = accessor(attributes["POSITION"])
            count = len(pos)
            faces = (accessor(prim["indices"]).astype(np.int64).reshape(-1, 3) if "indices" in prim
                     else np.arange(count - count % 3, dtype=np.int64).reshape(-1, 3))
            if np.linalg.det(linear) < 0:   # a mirroring node reverses the winding
                faces = faces[:, [0, 2, 1]]
            parts_pos.append(pos @ linear.T + world[:3, 3])
            if "TEXCOORD_0" in attributes:
                t = accessor(attributes["TEXCOORD_0"])
                parts_uv.append(np.stack([t[:, 0], 1.0 - t[:, 1]], axis=1))
            else:
                parts_uv.append(np.zeros((count, 2)))
            parts_textured.append(np.full(len(faces), "TEXCOORD_0" in attributes))
            parts_nrm.append(normalize(accessor(attributes["NORMAL"]) @ normal_matrix.T)
                             if "NORMAL" in attributes else None)
            parts_faces.append(faces + base)
            base += count
    if not parts_faces or sum(len(f) for f in parts_faces) == 0:
        raise BakeError(f"The low poly {path.name} has no triangles.")

    positions = np.concatenate(parts_pos)
    faces = np.concatenate(parts_faces)
    if faces.max() >= len(positions):
        raise BakeError(f"The low poly {path.name} refers to vertices it does not contain.")
    welded, pos_id = np.unique(positions, axis=0, return_inverse=True)
    tri_uv = np.where(np.concatenate(parts_textured)[:, None], faces, -1)
    normals = None if any(n is None for n in parts_nrm) else np.concatenate(parts_nrm)
    return LowInput(welded, np.concatenate(parts_uv), pos_id.reshape(-1)[faces], tri_uv,
                    normals=normals, polygons=len(faces))


def read_low(path: str | Path) -> LowInput:
    suffix = Path(path).suffix.lower()
    if suffix == ".obj":
        return read_obj(path)
    if suffix == ".glb":
        return read_glb(path)
    raise BakeError(f"The bake reads the low poly from an OBJ or GLB file, not '{suffix}'.")


def clean(low: LowInput, diag: float) -> tuple[LowInput, dict[str, int]]:
    """L1: drop triangles the file gave no texture coordinates, triangles whose 3D area is
    below 1e-12 * diag^2, and repeats of an earlier triangle over the same three positions."""
    tri_pos, tri_uv = low.tri_pos, low.tri_uv
    textured = (tri_uv >= 0).all(axis=1)
    _, area2, _ = _face_frames(low.positions, tri_pos)
    solid = 0.5 * area2 >= 1e-12 * diag * diag
    keep = textured & solid
    unique_rows = np.unique(np.sort(tri_pos[keep], axis=1), axis=0, return_index=True)[1]
    first = np.zeros(int(keep.sum()), dtype=bool)
    first[unique_rows] = True
    rows = np.flatnonzero(keep)[first]
    counts = {
        "untextured_triangles": int((~textured).sum()),
        "dropped_triangles": int((textured & ~solid).sum() + (~first).sum()),
    }
    if rows.size == 0:
        raise BakeError("The retopo mesh has no triangles with both an area and UV coordinates. "
                        "Run Retopo again.")
    cleaned = LowInput(low.positions, low.uvs, tri_pos[rows], tri_uv[rows], low.normals, low.polygons)
    return cleaned, counts


# ---------------------------------------------------------------------- L2: winding


def consistent_components(tri_pos: np.ndarray) -> np.ndarray:
    """Component label per triangle, joining two triangles only across an edge they share
    with no third one and run in opposite directions (so each component has one winding)."""
    a, b, manifold = _shared_edges(tri_pos)
    start = tri_pos.reshape(-1)
    end = np.roll(tri_pos, -1, axis=1).reshape(-1)
    opposite = manifold & (start[a] == end[b])
    return _components(len(tri_pos), a[opposite] // 3, b[opposite] // 3)


def fix_winding(low: LowInput, scene: Scene) -> tuple[LowInput, int]:
    """L2: flip every orientation-consistent component whose area-weighted vote says it
    faces into the high poly. Each triangle votes sign(n_low . n_high) with its area, the
    high triangle being the nearest one along +-its normal within 5% of the diagonal.
    Returns the fixed input and the number of triangles flipped."""
    labels = consistent_components(low.tri_pos)
    normals, area2, _ = _face_frames(low.positions, low.tri_pos)
    centroids = low.positions[low.tri_pos].mean(axis=1)
    hits = scene.nearest_bidirectional(centroids, normals, 0.05 * scene.diag)
    found = hits.hit
    agree = np.einsum("ij,ij->i", scene.face_normals[hits.prim[found]], normals[found])
    vote = np.bincount(labels[found], weights=np.sign(agree) * area2[found], minlength=labels.max() + 1)
    flip = (vote < 0)[labels]
    if not flip.any():
        return low, 0
    swap = [0, 2, 1]
    tri_pos = np.where(flip[:, None], low.tri_pos[:, swap], low.tri_pos)
    tri_uv = np.where(flip[:, None], low.tri_uv[:, swap], low.tri_uv)
    return LowInput(low.positions, low.uvs, tri_pos, tri_uv, low.normals, low.polygons), int(flip.sum())


# ---------------------------------------------------------------------- L3, L4: normals


def cage_normals(positions: np.ndarray, tri_pos: np.ndarray) -> np.ndarray:
    """L3: C_p = normalize(sum over the corners at p of angle * face normal), (nP, 3)."""
    normals, _, angles = _face_frames(positions, tri_pos)
    return _angle_weighted(len(positions), tri_pos, normals, angles)


def export_normals(
    positions: np.ndarray, tri_pos: np.ndarray, tri_uv: np.ndarray, hard_edge_angle: float
) -> tuple[np.ndarray, np.ndarray]:
    """L4: the normal group of every corner (nF, 3) and each group's unit normal.

    The corners at a position are joined across every edge through it that is not hard;
    an edge is hard when it is a UV seam (the texture coordinates differ across it) and its
    dihedral angle exceeds ``hard_edge_angle`` degrees. A negative angle smooths everything
    (the export normals are then the cage normals)."""
    normals, _, angles = _face_frames(positions, tri_pos)
    if hard_edge_angle < 0:
        return tri_pos, _angle_weighted(len(positions), tri_pos, normals, angles)
    a, b, _ = _shared_edges(tri_pos)
    ta, ka, tb, kb = a // 3, a % 3, b // 3, b % 3
    pos = tri_pos.reshape(-1)
    uvs = tri_uv.reshape(-1)
    # Corner of each triangle at the edge's two ends (its start, then its end, for edge a).
    a_start, a_end = 3 * ta + ka, 3 * ta + (ka + 1) % 3
    b_first = 3 * tb + kb
    b_second = 3 * tb + (kb + 1) % 3
    b_at_a_start = np.where(pos[b_first] == pos[a_start], b_first, b_second)
    b_at_a_end = np.where(pos[b_first] == pos[a_start], b_second, b_first)
    seam = (uvs[a_start] != uvs[b_at_a_start]) | (uvs[a_end] != uvs[b_at_a_end])
    cos = np.clip(np.einsum("ij,ij->i", normals[ta], normals[tb]), -1.0, 1.0)
    hard = seam & (np.degrees(np.arccos(cos)) > hard_edge_angle)
    soft = ~hard
    group = _components(3 * len(tri_pos),
                        np.concatenate([a_start[soft], a_end[soft]]),
                        np.concatenate([b_at_a_start[soft], b_at_a_end[soft]])).reshape(-1, 3)
    return group, _angle_weighted(int(group.max()) + 1, group, normals, angles)


# ---------------------------------------------------------------------- L5: repack


class RepackFailed(Exception):
    """The repacked atlas could not be used; the message says why, for a warning."""


def chart_labels(tri_uv: np.ndarray) -> np.ndarray:
    """UV island of every triangle: triangles sharing a texture coordinate are one island."""
    vertex = _components(int(tri_uv.max()) + 1,
                         np.concatenate([tri_uv[:, 0], tri_uv[:, 0]]),
                         np.concatenate([tri_uv[:, 1], tri_uv[:, 2]]))
    return vertex[tri_uv[:, 0]]


def repack(
    positions: np.ndarray, tri_pos: np.ndarray, tri_uv: np.ndarray, uvs: np.ndarray,
    cage: np.ndarray, resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    """L5: the same UV islands, moved apart by xatlas for a ``resolution`` map.

    Returns (uvs, tri_uv) in v-up units of the atlas's longer side, so texels stay square.
    Raises RepackFailed when xatlas fails, drops triangles, or distorts an island: a
    triangle's share of its island's UV area changing by more than 1%."""
    import xatlas

    pair_key = tri_pos.astype(np.int64) * (int(tri_uv.max()) + 1) + tri_uv
    pairs, tri_pair = np.unique(pair_key.reshape(-1), return_inverse=True)
    tri_pair = tri_pair.reshape(-1, 3)
    pair_pos = pairs // (int(tri_uv.max()) + 1)
    pair_uv = pairs % (int(tri_uv.max()) + 1)

    # The islands in texels of the map: packing from given coordinates, xatlas charts anew any face whose UV
    # area is under FLT_EPSILON - in a 0..1 atlas, a sliver of a large island.
    atlas = xatlas.Atlas()
    atlas.add_mesh(
        np.ascontiguousarray(positions[pair_pos], dtype=np.float32),
        np.ascontiguousarray(tri_pair, dtype=np.uint32),
        np.ascontiguousarray(cage[pair_pos], dtype=np.float32),
        np.ascontiguousarray(uvs[pair_uv] * float(resolution), dtype=np.float32),
    )
    chart_options = xatlas.ChartOptions()
    chart_options.use_input_mesh_uvs = True
    pack_options = xatlas.PackOptions()
    pack_options.resolution = int(resolution)
    # 4 texels at least: with 2 (at 512 px) mip level 2 blends neighbouring islands.
    pack_options.padding = max(4, int(resolution) // 256)
    pack_options.bilinear = True
    try:
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
    except Exception as exc:  # xatlas raises bare RuntimeErrors
        raise RepackFailed(f"xatlas: {exc}") from exc
    vmapping, indices, coords = (np.asarray(x) for x in atlas[0])
    indices = indices.astype(np.int64).reshape(-1, 3)
    if indices.shape != tri_pair.shape or not np.array_equal(vmapping[indices], tri_pair):
        raise RepackFailed("xatlas returned a different set of triangles")

    coords = coords.astype(np.float64).reshape(-1, 2)
    width, height = float(atlas.width), float(atlas.height)
    if coords.size and coords.max() <= 1.0 + 1e-4:   # the binding may hand back 0..1 per axis
        coords = coords * np.array([width, height])
    coords /= max(width, height)
    coords[:, 1] = 1.0 - coords[:, 1]

    # xatlas evens out the texel density, so islands may grow or shrink relative to each
    # other (by up to ~25% on Instant Meshes output), and it may rotate or mirror them;
    # what must not change is an island's shape: every triangle keeps its share of it.
    charts = chart_labels(indices)
    old = np.abs(uv_area2(uvs, tri_uv))
    new = np.abs(uv_area2(coords, indices))
    scale = np.bincount(charts, weights=new) / np.maximum(np.bincount(charts, weights=old), 1e-300)
    measured = old >= resolution ** -2.0          # triangles of at least half a texel
    if measured.any():
        change = np.abs(new[measured] / (old[measured] * scale[charts[measured]]) - 1.0).max()
        if change > REPACK_AREA_TOLERANCE:
            raise RepackFailed(f"a UV island would be distorted by {100 * change:.1f}%")
    return coords, indices


def min_chart_gap(uv: np.ndarray, tri_uv: np.ndarray, resolution: int,
                  limit: int = MAX_MEASURED_GAP) -> int:
    """Empty texels between the closest texel centres of two different UV islands in an
    R x R map (their Chebyshev distance - 1), or ``limit`` when none are that close.

    Grows square windows one texel at a time, tracking the lowest and highest island
    label inside each: the first size w at which some window holds two labels puts two
    islands w - 1 texels apart."""
    charts = chart_labels(tri_uv).astype(np.int32)
    label = np.full(resolution * resolution, -1, dtype=np.int32)
    for band in sample_bands(uv, tri_uv, resolution, 1):
        label[band.texel] = charts[band.tri]
    high = label.reshape(resolution, resolution)
    low = np.where(high >= 0, high, np.iinfo(np.int32).max)
    for gap in range(limit):
        high = np.maximum(np.maximum(high[:-1, :-1], high[1:, :-1]), np.maximum(high[:-1, 1:], high[1:, 1:]))
        low = np.minimum(np.minimum(low[:-1, :-1], low[1:, :-1]), np.minimum(low[:-1, 1:], low[1:, 1:]))
        if (high > low).any():
            return gap
    return limit


# ---------------------------------------------------------------------- L6, L7


def build_render_mesh(
    positions: np.ndarray, uvs: np.ndarray, tri_pos: np.ndarray, tri_uv: np.ndarray,
    cage: np.ndarray, group: np.ndarray, group_normals: np.ndarray,
) -> tuple[RenderMesh, dict[str, int]]:
    """L6 + L7: one render vertex per (position, texture coordinate, normal group, UV
    orientation), with MikkTSpace tangents. A corner of a triangle without UV area joins
    the orientation its vertex already has."""
    area = uv_area2(uvs, tri_uv)
    valid = np.repeat(np.abs(area) >= DEGENERATE_UV_AREA, 3)
    positive = np.repeat(area > 0, 3)
    base_key = group.reshape(-1).astype(np.int64) * (int(tri_uv.max()) + 1) + tri_uv.reshape(-1)
    _, base = np.unique(base_key, return_inverse=True)
    base = base.reshape(-1)
    has_pos = np.bincount(base[valid & positive], minlength=base.max() + 1) > 0
    has_neg = np.bincount(base[valid & ~positive], minlength=base.max() + 1) > 0
    orient = np.where(valid, positive, has_pos[base] | ~has_neg[base])
    vertex_key, corner_vertex = np.unique(2 * base + orient, return_inverse=True)
    corner_vertex = corner_vertex.reshape(-1)
    faces = corner_vertex.reshape(-1, 3)
    first = np.zeros(len(vertex_key), dtype=np.int64)
    first[corner_vertex[::-1]] = np.arange(len(corner_vertex))[::-1]
    pos = tri_pos.reshape(-1)[first]

    out_positions = positions[pos]
    out_uv = uvs[tri_uv.reshape(-1)[first]]
    out_normals = group_normals[group.reshape(-1)[first]]
    tangents = mikk_tangents(out_positions, out_normals, out_uv, faces)
    mesh = RenderMesh(
        positions=out_positions,
        uv=out_uv,
        normals=out_normals,
        cage=cage[pos],
        tangents=tangents.per_vertex(faces, len(vertex_key)),
        faces=faces,
        position_id=pos,
    )
    stats = {"uv_degenerate_triangles": tangents.degenerate, "tangent_fallback_corners": tangents.fallback}
    return mesh, stats


# ---------------------------------------------------------------------- all together


def prepare(
    path: str | Path,
    scene: Scene,
    *,
    resolution: int,
    repack_uvs: bool = True,
    hard_edge_angle: float = DEFAULT_HARD_EDGE_ANGLE,
) -> LowPoly:
    """Steps L1-L7 for the low poly at ``path`` (OBJ or GLB) against the high poly in
    ``scene``, for an R = ``resolution`` map. A GLB keeps its own NORMALs as the export
    normals (``hard_edge_angle`` then has no say); an OBJ gets them from L4."""
    low, stats = clean(read_low(path), scene.diag)
    low, flipped = fix_winding(low, scene)
    cage = cage_normals(low.positions, low.tri_pos)
    if low.normals is not None:
        group, group_normals = low.tri_uv, low.normals
    else:
        group, group_normals = export_normals(low.positions, low.tri_pos, low.tri_uv, hard_edge_angle)

    warnings: list[str] = []
    if stats["untextured_triangles"]:
        warnings.append(f"{stats['untextured_triangles']} triangles of the retopo mesh have no UV "
                        f"coordinates and were left out. Run Retopo again to unwrap them.")
    uvs, tri_uv, repacked = low.uvs, low.tri_uv, False
    if repack_uvs:
        try:
            uvs, tri_uv = repack(low.positions, low.tri_pos, low.tri_uv, low.uvs, cage, resolution)
            repacked = True
        except RepackFailed as exc:
            warnings.append(f"The UV islands could not be spread apart ({exc}), so the bake uses "
                            f"the retopo's UV layout as it is; seams may show on distant views.")

    mesh, mesh_stats = build_render_mesh(low.positions, uvs, low.tri_pos, tri_uv, cage, group,
                                         group_normals)
    stats.update(mesh_stats)
    stats.update(
        polygons=low.polygons,
        triangles=len(mesh.faces),
        flipped_triangles=flipped,
        # The clean mesh's own points (what Retopo counts), and the vertices the files hold: a point on a UV seam
        # or a hard edge is one vertex per side there.
        vertices=int(len(np.unique(low.tri_pos))),
        render_vertices=len(mesh.positions),
        min_chart_gap_px=min_chart_gap(uvs, tri_uv, resolution),
    )
    return LowPoly(mesh, repacked, stats, warnings)
