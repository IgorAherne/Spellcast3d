"""Analytic ground truth for the Projection stage's texture baker.

Everything a baker is judged on is computed here from closed-form surfaces, never
from the baker's own data structures:

* scenes     : implicit high-poly surfaces F(X) = 0 (F > 0 outside) with an analytic
               normal and analytic colour / roughness / metalness fields.
* low meshes : the low poly as the Projection stage reads ``retopo.obj``: per-corner
               UVs, polygons fanned (0, k-1, k) like ``instant_meshes_brush.uv``.
* raster     : texel-centre UV rasteriser (with the extrapolation ring) and push-pull.
* frames     : how a tangent-space normal map is decoded -- per-fragment MikkTSpace
               ("mikk", Blender/UE/Unity HDRP) or three.js per-vertex ("per_vertex")
               -- and the exact inverse that encodes it.
* oracle     : a reference baker that writes the analytic answer, with optional
               MUTATIONS (the bugs real bakers have) that prove the gates bite.
* glb io     : a stdlib + numpy GLB 2.0 writer and reader (several primitives, node
               transforms, PNG/JPEG/WebP images, material ``extras.spellcast``).
* evaluate   : the black-box scorer. It reads only the output GLB, samples random
               points on the low surface, casts the baker's ray (the ``cage`` policy
               by default) against the ANALYTIC surface and decodes the normal map in
               the tangent basis the GLB declares.
* gates      : the hard limits of design section 7 (L1/L2) and a PASS/FAIL verdict.

UV convention in numpy: OBJ style, v up, image row 0 = v 1. In a GLB, TEXCOORD_0 is
(u, 1 - v) and images are stored top row first, so rows line up with numpy's.
"""

from __future__ import annotations

import base64
import io
import json
import math
import struct
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

#: The two decoders a baked normal map can be written for (design section 4.4, L13).
TANGENT_BASES = ("mikk", "per_vertex")

#: How the evaluator finds the true surface point of a low-poly sample.
#: ``cage`` is the real baker's L10: from P + d*C along -C (C the welded smooth normal),
#: the first crossing that enters the surface, else the nearest crossing within 4d.
#: ``nearest`` is the nearest crossing along the exported NORMAL within +-d.
POLICIES = ("cage", "nearest")

#: Deliberate oracle bugs; each must make at least one gate FAIL.
MUTATIONS = (
    "flip_green",  # DirectX (Y-down) normal map shipped as glTF
    "ignore_w",  # bitangent sign ignored (mirrored charts come out inverted)
    "vflip",  # image rows written bottom-up
    "gamma_twice",  # sRGB encoding applied to an already-sRGB colour
    "half_texel",  # samples at texel corners instead of centres
    "no_padding",  # no extrapolation ring, background left black
    "face_rays",  # rays along the flat face normal instead of the smooth cage normal
    "transpose",  # TBN^T instead of TBN^-1 (wrong on non-orthonormal frames)
)

#: Ray-search grid points across one window; Illinois regula falsi refines the bracket
#: (superlinear: a smooth crossing is exact to ~1e-15 of the window well within this).
_GRID = 17
_REFINE = 24


# ============================================================================
#  small helpers
# ============================================================================


def normalize(v: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    n = np.sqrt((v * v).sum(-1, keepdims=True))
    return v / np.maximum(n, eps)


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between rows, accurate for tiny angles (atan2, not acos)."""
    a = normalize(a)
    b = normalize(b)
    c = np.clip((a * b).sum(-1), -1.0, 1.0)
    s = np.sqrt((np.cross(a, b) ** 2).sum(-1))
    return np.degrees(np.arctan2(s, c))


def stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
    }


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def to_u8(v: np.ndarray) -> np.ndarray:
    """[0, 1] floats -> bytes, round(255 v) (design L18)."""
    return np.clip(np.round(np.asarray(v) * 255.0), 0, 255).astype(np.uint8)


def scatter_add(index: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    """Per-index row sums, shape (size, k): np.add.at, but at bincount speed."""
    values = np.asarray(values, dtype=np.float64).reshape(len(index), -1)
    out = np.empty((size, values.shape[1]))
    for k in range(values.shape[1]):
        out[:, k] = np.bincount(index, weights=values[:, k], minlength=size)
    return out


# ============================================================================
#  analytic scenes
# ============================================================================


class Scene:
    """High-poly surface as the zero set of F, with F > 0 outside (above)."""

    name = "scene"
    #: Ray window when the baked GLB declares no ``extras.spellcast.ray_distance``.
    t_range = 0.2

    def F(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def normal(self, X: np.ndarray) -> np.ndarray:
        h = 1e-6
        g = np.empty_like(X)
        for k in range(3):
            d = np.zeros(3)
            d[k] = h
            g[:, k] = (self.F(X + d) - self.F(X - d)) / (2 * h)
        return normalize(g)

    def colour(self, X: np.ndarray) -> np.ndarray:
        """Base colour as the sRGB-encoded value in [0, 1] an 8-bit texture stores / 255."""
        raise NotImplementedError

    def metal_rough(self, X: np.ndarray) -> np.ndarray:
        """(n, 3) linear glTF packing: R occlusion (1), G roughness, B metalness."""
        raise NotImplementedError

    def spec(self) -> dict[str, Any]:
        """JSON description that :func:`scene_from_spec` turns back into the scene."""
        raise NotImplementedError

    def escapes(self, X: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """(n, k) True where the ray X + s*dirs[:, j] reaches the sky (AO scenes only)."""
        raise NotImplementedError(f"the {self.name} scene has no occlusion model")

    # --- rays -------------------------------------------------------------
    def hit(self, P: np.ndarray, D: np.ndarray, window: float,
            policy: str = "cage") -> tuple[np.ndarray, np.ndarray]:
        """Surface point of the ray through P along D (unit), and whether one exists.

        ``cage``: walk from P + window*D along -D down to P - window*D and take the first
        crossing that ENTERS the surface (F goes from > 0 to <= 0: a front face seen
        from outside); rows without one fall back to the nearest crossing within
        4*window in either direction. ``nearest``: nearest crossing within +-window.
        Crossings are bracketed on a grid and refined to ~1e-15 window.
        """
        if policy not in POLICIES:
            raise ValueError(f"unknown ray policy {policy!r}; expected one of {POLICIES}")
        n = P.shape[0]
        lo = np.full(n, np.nan)
        hi = np.full(n, np.nan)
        nearest_window = window
        if policy == "cage":
            ts = np.linspace(window, -window, _GRID)
            rows = np.arange(n)
            f_prev = self.F(P + ts[0] * D)
            for k in range(1, _GRID):
                f_k = self.F(P[rows] + ts[k] * D[rows])
                enter = (f_prev > 0) & (f_k <= 0)
                lo[rows[enter]], hi[rows[enter]] = ts[k], ts[k - 1]
                rows, f_prev = rows[~enter], f_k[~enter]
                if rows.size == 0:
                    break
            nearest_window = 4.0 * window
        todo = np.flatnonzero(np.isnan(lo))
        if todo.size:
            lo[todo], hi[todo] = self._nearest_bracket(P[todo], D[todo], nearest_window)
        ok = ~np.isnan(lo)
        t = np.zeros(n)
        if ok.any():
            t[ok] = self._refine(P[ok], D[ok], lo[ok], hi[ok], window)
        return P + t[:, None] * D, ok

    def crossings(self, P: np.ndarray, D: np.ndarray, window: float) -> np.ndarray:
        """Surface crossings of the ray within +-window (1 = the policies agree)."""
        f = np.stack([self.F(P + t * D) for t in np.linspace(-window, window, 2 * _GRID - 1)])
        return ((f[:-1] > 0) != (f[1:] > 0)).sum(0)

    def _nearest_bracket(self, P, D, window):
        """Bracket of the sign change nearest t = 0 (the negative side on a tie), NaN if
        there is none within +-window."""
        ts = np.linspace(0.0, window, _GRID)
        lo = np.full(P.shape[0], np.nan)
        hi = np.full(P.shape[0], np.nan)
        rows = np.arange(P.shape[0])
        f_neg = f_pos = self.F(P) > 0
        for k in range(1, _GRID):
            fn = self.F(P[rows] - ts[k] * D[rows]) > 0
            fp = self.F(P[rows] + ts[k] * D[rows]) > 0
            neg = fn != f_neg
            pos = (fp != f_pos) & ~neg
            lo[rows[neg]], hi[rows[neg]] = -ts[k], -ts[k - 1]
            lo[rows[pos]], hi[rows[pos]] = ts[k - 1], ts[k]
            keep = ~(neg | pos)
            rows, f_neg, f_pos = rows[keep], fn[keep], fp[keep]
            if rows.size == 0:
                break
        return lo, hi

    def _refine(self, P, D, a, b, window):
        """Root of F(P + tD) in the bracket [a, b] (Illinois variant of regula falsi),
        iterating each row until its step falls below 1e-15 of the window."""
        fa = self.F(P + a[:, None] * D)
        fb = self.F(P + b[:, None] * D)
        rows = np.arange(P.shape[0])
        for _ in range(_REFINE):
            A, B, FA, FB = a[rows], b[rows], fa[rows], fb[rows]
            step = FB - FA
            C = np.where(step != 0, B - FB * (B - A) / np.where(step != 0, step, 1.0), B)
            FC = self.F(P[rows] + C[:, None] * D[rows])
            flip = FC * FB < 0
            a[rows], fa[rows] = np.where(flip, B, A), np.where(flip, FB, 0.5 * FA)
            b[rows], fb[rows] = C, FC
            rows = rows[(np.abs(C - B) > 1e-15 * window) & (FC != 0)]
            if rows.size == 0:
                break
        return b


def _wave(X: np.ndarray, dirs: np.ndarray, phases: np.ndarray) -> np.ndarray:
    return 0.5 + 0.5 * np.sin(X @ dirs.T + phases)


class PlaneScene(Scene):
    """Low poly: the unit square z = 0 of a local frame. High poly: z = h(x, y).

    ``R``/``c`` place the local frame in the world (world = R @ local + c), so a baker
    that silently assumes an axis or drops a transform is caught.
    """

    name = "plane"
    t_range = 0.25

    def __init__(self, amp: float = 0.03, bump: float = 0.06,
                 R: Sequence[Sequence[float]] | None = None, c: Sequence[float] | None = None):
        self.amp = amp
        self.bump = bump
        self.R = np.eye(3) if R is None else np.asarray(R, dtype=np.float64)
        self.c = np.zeros(3) if c is None else np.asarray(c, dtype=np.float64)
        self.cdirs = np.array([[9.0, 3.0, 20.0], [-4.0, 10.0, 20.0], [7.0, -8.0, 20.0]])
        self.cphase = np.array([0.3, 1.1, 2.0])

    def to_local(self, X: np.ndarray) -> np.ndarray:
        return (X - self.c) @ self.R

    def to_world(self, L: np.ndarray) -> np.ndarray:
        return L @ self.R.T + self.c

    def h(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        tau = 2 * np.pi
        r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
        return (self.amp * np.sin(tau * 3 * x) * np.cos(tau * 2 * y)
                + self.bump * np.exp(-r2 / (2 * 0.07 ** 2)))

    def F(self, X):
        L = self.to_local(X)
        return L[:, 2] - self.h(L[:, 0], L[:, 1])

    def colour(self, X):
        return _wave(self.to_local(X), self.cdirs, self.cphase)

    def metal_rough(self, X):
        L = self.to_local(X)
        rough = 0.5 + 0.4 * np.sin(2 * np.pi * (2 * L[:, 0] + L[:, 1]))
        metal = 0.5 + 0.5 * np.tanh(6 * np.sin(2 * np.pi * 1.5 * L[:, 1]))
        return np.stack([np.ones_like(rough), rough, metal], -1)

    def spec(self):
        return {"scene": "plane", "amp": self.amp, "bump": self.bump,
                "R": self.R.tolist(), "c": self.c.tolist()}


class SphereScene(Scene):
    """High poly: r(d) = 1 + a g(d) along each direction d (star-shaped; a = 0 is exact)."""

    name = "sphere"
    t_range = 0.2

    def __init__(self, a: float = 0.05):
        self.a = a
        self.cdirs = np.array([[6.0, 2.0, 1.0], [-2.0, 5.0, 3.0], [1.0, -3.0, 6.0]])
        self.cphase = np.array([0.0, 1.0, 2.0])

    def g(self, d: np.ndarray) -> np.ndarray:
        return (0.6 * np.sin(5 * d[:, 0]) * np.sin(4 * d[:, 1])
                + 0.4 * np.cos(6 * d[:, 2] + 1.0)
                + 0.5 * np.exp(-((d - np.array([0.6, 0.6, 0.529])) ** 2).sum(-1) / 0.02))

    def radius(self, d: np.ndarray) -> np.ndarray:
        return 1.0 + self.a * self.g(d) if self.a else np.ones(d.shape[0])

    def F(self, X):
        r = np.sqrt((X * X).sum(-1))
        return r - self.radius(X / np.maximum(r, 1e-30)[:, None])

    def colour(self, X):
        return _wave(X, self.cdirs, self.cphase)

    def metal_rough(self, X):
        rough = 0.5 + 0.4 * np.sin(3 * X[:, 0] + 2 * X[:, 2])
        metal = 0.5 + 0.5 * np.tanh(4 * np.sin(4 * X[:, 1]))
        return np.stack([np.ones_like(rough), rough, metal], -1)

    def spec(self):
        return {"scene": "sphere", "a": self.a}


def scene_from_spec(spec: Mapping[str, Any]) -> Scene:
    """Inverse of ``Scene.spec`` for the scenes the CLI can score."""
    kind = spec.get("scene")
    if kind == "plane":
        return PlaneScene(amp=spec["amp"], bump=spec["bump"], R=spec["R"], c=spec["c"])
    if kind == "sphere":
        return SphereScene(a=spec["a"])
    raise ValueError(f"unknown scene {kind!r}; expected 'plane' or 'sphere'")


# ============================================================================
#  low-poly meshes (split-vertex form, like a GLB primitive)
# ============================================================================


@dataclass
class LowMesh:
    pos: np.ndarray  # (n, 3) float64
    nrm: np.ndarray  # (n, 3) exported normals
    uv: np.ndarray  # (n, 2) v up
    faces: np.ndarray  # (m, 3) int64
    tan: np.ndarray | None = None  # (n, 4) xyz + handedness w
    weld: np.ndarray | None = None  # (n,) position id: one per distinct 3D point

    def face_normals(self) -> np.ndarray:
        p = self.pos[self.faces]
        return normalize(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]))

    @cached_property
    def cage(self) -> np.ndarray:
        """Welded angle-weighted normals per render vertex: the baker's cage normals (L3).

        Continuous across UV seams and hard edges, which is what makes the projection
        seamless; the exported NORMAL may differ where the baker split hard edges.
        """
        if self.weld is None:
            raise ValueError("cage normals need weld ids")
        points = np.zeros((int(self.weld.max()) + 1, 3))
        points[self.weld] = self.pos
        return angle_weighted_normals(points, self.weld[self.faces])[self.weld]


def angle_weighted_normals(P: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Vertex normals = normalize(sum over corners of corner angle * face normal)."""
    p = P[F]
    fn = normalize(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]))
    acc = np.zeros((P.shape[0], 3))
    for k in range(3):
        a = normalize(p[:, (k + 1) % 3] - p[:, k])
        b = normalize(p[:, (k + 2) % 3] - p[:, k])
        ang = np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0))
        acc += scatter_add(F[:, k], fn * ang[:, None], P.shape[0])
    return normalize(acc)


def split_mesh(P: np.ndarray, F: np.ndarray, UV: np.ndarray, FT: np.ndarray,
               N: np.ndarray | None = None) -> LowMesh:
    """Positions indexed by F, UVs by FT -> one render vertex per (position, uv) pair.

    Normals are computed on the WELDED positions unless given, so both sides of a UV
    seam carry the same normal; tangents follow :func:`compute_tangents`.
    """
    if N is None:
        N = angle_weighted_normals(P, F)
    width = int(FT.max()) + 1
    key = F.astype(np.int64) * width + FT
    uniq, inv = np.unique(key.ravel(), return_inverse=True)
    pid = uniq // width
    tid = uniq % width
    m = LowMesh(pos=P[pid].astype(np.float64), nrm=np.asarray(N, np.float64)[pid],
                uv=UV[tid].astype(np.float64), faces=inv.reshape(-1, 3), weld=pid)
    m.tan = compute_tangents(m)
    return m


def compute_tangents(m: LowMesh) -> np.ndarray:
    """Per-vertex tangents in the MikkTSpace way (angle-weighted, projected onto the
    vertex normal plane, handedness in w; bitangent = w * cross(N, T); v-up UVs so +B is
    +v, glTF's "green up"). Matches the mikktspace wasm to ~0.004 deg on clean meshes;
    it has no degenerate-triangle or orientation-split handling.
    """
    p = m.pos[m.faces]
    t = m.uv[m.faces]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    d1, d2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) < 1e-30, 1e-30, r)
    T = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / r[:, None]
    B = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / r[:, None]
    n_v = m.pos.shape[0]
    tacc = np.zeros((n_v, 3))
    bacc = np.zeros((n_v, 3))
    for k in range(3):
        a = normalize(p[:, (k + 1) % 3] - p[:, k])
        b = normalize(p[:, (k + 2) % 3] - p[:, k])
        ang = np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0))[:, None]
        n = m.nrm[m.faces[:, k]]
        tp = normalize(T - n * (T * n).sum(-1, keepdims=True))
        bp = normalize(B - n * (B * n).sum(-1, keepdims=True))
        tacc += scatter_add(m.faces[:, k], tp * ang, n_v)
        bacc += scatter_add(m.faces[:, k], bp * ang, n_v)
    tv = normalize(tacc - m.nrm * (tacc * m.nrm).sum(-1, keepdims=True))
    w = np.where((np.cross(m.nrm, tv) * bacc).sum(-1) < 0, -1.0, 1.0)
    return np.concatenate([tv, w[:, None]], -1)


def _obj_index(token: str, count: int) -> int:
    i = int(token)
    return i - 1 if i > 0 else count + i


def read_obj(path: str | Path) -> LowMesh:
    """The low poly as the Projection stage reads ``retopo.obj``.

    Polygons are fanned (0, k-1, k) exactly like ``instant_meshes_brush.uv._triangulate``
    and three.js's OBJLoader; faces without UVs cannot be baked and are skipped. Render
    vertices are split by (position, uv) with smooth welded normals (no hard edges).
    """
    V: list[list[float]] = []
    VT: list[list[float]] = []
    F: list[tuple[int, int, int]] = []
    FT: list[tuple[int, int, int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            V.append([float(x) for x in parts[1:4]])
        elif parts[0] == "vt":
            VT.append([float(x) for x in parts[1:3]])
        elif parts[0] == "f":
            corners = [c.split("/") for c in parts[1:]]
            if any(len(c) < 2 or not c[1] for c in corners):
                continue
            vi = [_obj_index(c[0], len(V)) for c in corners]
            ti = [_obj_index(c[1], len(VT)) for c in corners]
            for k in range(2, len(vi)):
                F.append((vi[0], vi[k - 1], vi[k]))
                FT.append((ti[0], ti[k - 1], ti[k]))
    if not F:
        raise ValueError(f"{path} has no faces with texture coordinates")
    return split_mesh(np.array(V), np.array(F, np.int64), np.array(VT), np.array(FT, np.int64))


def icosphere(subdiv: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit icosphere, outward winding: (V (n, 3), F (m, 3))."""
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
             (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    V = [np.array(v, np.float64) / np.linalg.norm(v) for v in verts]
    for _ in range(subdiv):
        cache: dict[tuple[int, int], int] = {}

        def mid(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = V[a] + V[b]
                V.append(m / np.linalg.norm(m))
                cache[key] = len(V) - 1
            return cache[key]

        refined = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            refined += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = refined
    return np.array(V), np.array(faces, dtype=np.int64)


# ============================================================================
#  UV rasteriser (texel centres) + push-pull fill
# ============================================================================


def raster_uv(uv: np.ndarray, faces: np.ndarray, res: int, *, ring_px: float = 0.0,
              offset: float = 0.5, chunk: int = 2_000_000
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The triangle and barycentrics at every sample centre of a res x res grid.

    Sample (row r, col c) sits at u = (c + offset)/res, v = 1 - (r + offset)/res, the
    texel centre for the default offset 1/2. Rasterising at R*S reproduces the baker's
    S x S stratified pattern, since sample (c, r, i, j) of design L8 is centre
    (c*S + i, r*S + j) of the finer grid. A centre on several triangles goes to the
    lowest triangle index.

    ``ring_px`` > 0: a centre outside every triangle but within ring_px samples of one
    takes the nearest triangle with UNCLAMPED barycentrics (the surface extended past
    the chart edge).

    Returns tid (res, res) int32 (-1 empty), bary (res, res, 3) and dist (res, res):
    0 inside, else the distance in samples (inf where empty).
    """
    off = offset
    xy = np.stack([uv[:, 0] * res, (1.0 - uv[:, 1]) * res], -1)[faces]
    A, B, C = xy[:, 0], xy[:, 1], xy[:, 2]
    den = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
    lo = np.clip(np.floor(xy.min(1) - off - ring_px), 0, res - 1).astype(np.int64)
    hi = np.clip(np.ceil(xy.max(1) - off + ring_px), 0, res - 1).astype(np.int64)
    nx = hi[:, 0] - lo[:, 0] + 1
    ny = hi[:, 1] - lo[:, 1] + 1
    count = np.where(np.abs(den) > 1e-18, nx * ny, 0)

    tid = np.full(res * res, -1, np.int32)
    bary = np.zeros((res * res, 3))
    dist = np.full(res * res, np.inf)
    order = np.flatnonzero(count > 0)
    csum = np.cumsum(count[order])
    start = 0
    while start < order.size:
        base = csum[start - 1] if start else 0
        stop = max(int(np.searchsorted(csum, base + chunk, side="right")), start + 1)
        tri = order[start:stop]
        c = count[tri]
        t = np.repeat(tri, c)
        k = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        ix = lo[t, 0] + k % nx[t]
        iy = lo[t, 1] + k // nx[t]
        gx = ix + off
        gy = iy + off
        l0 = ((B[t, 1] - C[t, 1]) * (gx - C[t, 0]) + (C[t, 0] - B[t, 0]) * (gy - C[t, 1])) / den[t]
        l1 = ((C[t, 1] - A[t, 1]) * (gx - C[t, 0]) + (A[t, 0] - C[t, 0]) * (gy - C[t, 1])) / den[t]
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        d = np.where(inside, 0.0, np.inf)
        if ring_px > 0:
            for P0, P1 in ((A, B), (B, C), (C, A)):
                e = P1[t] - P0[t]
                qx = gx - P0[t, 0]
                qy = gy - P0[t, 1]
                s = np.clip((qx * e[:, 0] + qy * e[:, 1]) / np.maximum((e * e).sum(1), 1e-30), 0, 1)
                d = np.minimum(d, np.hypot(qx - s * e[:, 0], qy - s * e[:, 1]))
            d = np.where(inside, 0.0, d)
        keep = np.flatnonzero(d <= ring_px)
        texel = iy[keep] * res + ix[keep]
        o = np.lexsort((t[keep], d[keep], texel))
        keep, texel = keep[o], texel[o]
        first = np.ones(texel.size, bool)
        first[1:] = texel[1:] != texel[:-1]
        keep, texel = keep[first], texel[first]
        better = d[keep] < dist[texel]
        keep, texel = keep[better], texel[better]
        tid[texel] = t[keep]
        dist[texel] = d[keep]
        bary[texel] = np.stack([l0[keep], l1[keep], l2[keep]], -1)
        start = stop
    return tid.reshape(res, res), bary.reshape(res, res, 3), dist.reshape(res, res)


def push_pull(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill every texel outside ``mask`` from a mask-weighted pyramid (design L17).

    Covered texels are kept; the rest take the weighted average of the nearest coarser
    level that has data, so no background colour is left at any mip level.
    """
    w = mask.astype(np.float64)
    c = np.asarray(img, np.float64) * w[..., None]
    levels = [(c, w)]
    while max(levels[-1][1].shape) > 1:
        c, w = levels[-1]
        ph, pw = w.shape[0] % 2, w.shape[1] % 2
        if ph or pw:
            c = np.pad(c, ((0, ph), (0, pw), (0, 0)))
            w = np.pad(w, ((0, ph), (0, pw)))
        h2, w2 = w.shape[0] // 2, w.shape[1] // 2
        levels.append((c.reshape(h2, 2, w2, 2, -1).sum((1, 3)), w.reshape(h2, 2, w2, 2).sum((1, 3))))
    c, w = levels[-1]
    cur = c / np.maximum(w, 1e-12)[..., None]
    for c, w in reversed(levels[:-1]):
        up = np.repeat(np.repeat(cur, 2, 0), 2, 1)[: w.shape[0], : w.shape[1]]
        cur = np.where((w > 0)[..., None], c / np.maximum(w, 1e-12)[..., None], up)
    return cur


# ============================================================================
#  tangent frames: how a normal map is decoded, and the exact inverse
# ============================================================================


def frame(m: LowMesh, fid: np.ndarray, bary: np.ndarray, basis: str, *,
          ignore_w: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The (T, B, N) a renderer builds at a surface point; decoded n = normalize(xT + yB + zN).

    ``mikk`` (per-fragment MikkTSpace, mikktspace.h's pixel shader): vT, vN interpolated
    UNnormalised, vB = w * cross(vN, vT), w the triangle's sign.
    ``per_vertex`` (three.js r169/r186 normal_vertex + normal_fragment_begin): per vertex
    n = normalize(N), t = normalize(T), b = normalize(w cross(n, t)); each interpolated
    and renormalised.
    """
    if m.tan is None:
        raise ValueError("the mesh has no tangents")
    f = m.faces[fid]
    bw = bary[..., None]
    nrm = m.nrm[f]
    tan = m.tan[f]
    w = np.ones_like(tan[..., 3:4]) if ignore_w else tan[..., 3:4]
    if basis == "mikk":
        vN = (nrm * bw).sum(1)
        vT = (tan[..., :3] * bw).sum(1)
        return vT, w[:, 0] * np.cross(vN, vT), vN
    if basis == "per_vertex":
        n = normalize(nrm)
        t = normalize(tan[..., :3])
        b = normalize(np.cross(n, t) * w)
        return normalize((t * bw).sum(1)), normalize((b * bw).sum(1)), normalize((n * bw).sum(1))
    raise ValueError(f"unknown tangent basis {basis!r}; expected one of {TANGENT_BASES}")


def frame_derivative(m: LowMesh, fid: np.ndarray, bary: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """three.js without a TANGENT attribute (getTangentFrame from screen derivatives,
    GLTFLoader's normalScale.y flip cancels glTF's v-down UVs). Reported only, to show
    why the TANGENT attribute must be exported."""
    f = m.faces[fid]
    p = m.pos[f]
    t = m.uv[f]
    N = normalize((m.nrm[f] * bary[..., None]).sum(1))
    q0 = p[:, 1] - p[:, 0]
    q1 = p[:, 2] - p[:, 0]
    st0 = t[:, 1] - t[:, 0]
    st1 = t[:, 2] - t[:, 0]
    q1perp = np.cross(q1, N)
    q0perp = np.cross(N, q0)
    T = q1perp * st0[:, 0:1] + q0perp * st1[:, 0:1]
    B = q1perp * st0[:, 1:2] + q0perp * st1[:, 1:2]
    s = np.sign((np.cross(q0, q1) * N).sum(-1))[:, None]
    scale = 1.0 / np.sqrt(np.maximum(np.maximum((T * T).sum(-1), (B * B).sum(-1)), 1e-30))
    return T * s * scale[:, None], B * s * scale[:, None], N


def decode(T: np.ndarray, B: np.ndarray, N: np.ndarray, mapN: np.ndarray) -> np.ndarray:
    return normalize(T * mapN[:, 0:1] + B * mapN[:, 1:2] + N * mapN[:, 2:3])


def encode(T: np.ndarray, B: np.ndarray, N: np.ndarray, n: np.ndarray, *,
           transpose: bool = False) -> np.ndarray:
    """Tangent-space unit normal that :func:`decode` turns back into ``n`` (design L13).

    The exact inverse adj(M) n / det M of M = [T | B | N] (mikktspace.h's sampler, valid
    for both bases and for mirrored frames); a near-singular frame is orthonormalised
    first. ``transpose`` is the mutation M^T n. Then z <- max(z, 0), renormalised.
    """
    if transpose:
        v = np.stack([(n * T).sum(-1), (n * B).sum(-1), (n * N).sum(-1)], -1)
    else:
        r0, r1, r2 = np.cross(B, N), np.cross(N, T), np.cross(T, B)
        det = (T * r0).sum(-1)
        size = np.linalg.norm(T, axis=-1) * np.linalg.norm(B, axis=-1) * np.linalg.norm(N, axis=-1)
        v = np.stack([(n * r0).sum(-1), (n * r1).sum(-1), (n * r2).sum(-1)], -1) * np.sign(det)[:, None]
        tiny = np.abs(det) < 1e-8 * size
        if tiny.any():
            Nn = normalize(N[tiny])
            Tn = normalize(T[tiny] - Nn * (T[tiny] * Nn).sum(-1, keepdims=True))
            hand = np.where((np.cross(Nn, Tn) * B[tiny]).sum(-1) < 0, -1.0, 1.0)[:, None]
            Bn = hand * np.cross(Nn, Tn)
            nt = n[tiny]
            v[tiny] = np.stack([(nt * Tn).sum(-1), (nt * Bn).sum(-1), (nt * Nn).sum(-1)], -1)
    v = normalize(v)
    v[:, 2] = np.maximum(v[:, 2], 0.0)
    return normalize(v)


def sample_rays(m: LowMesh, fid: np.ndarray, bary: np.ndarray, policy: str
                ) -> tuple[np.ndarray, np.ndarray]:
    """Surface point and ray direction of each sample under ``policy``."""
    f = m.faces[fid]
    bw = bary[..., None]
    P = (m.pos[f] * bw).sum(1)
    field_ = m.cage if policy == "cage" else m.nrm
    return P, normalize((field_[f] * bw).sum(1))


# ============================================================================
#  oracle baker (with mutations)
# ============================================================================


@dataclass
class OracleBake:
    mesh: LowMesh
    images: dict[str, np.ndarray]  # baseColor / metallicRoughness (ORM) / normal, uint8
    height: np.ndarray | None  # (res, res) offset of the hit along the cage normal
    mask: np.ndarray  # (res, res) texels that received at least one sample
    window: float
    info: dict[str, Any] = field(default_factory=dict)


def radical_inverse(i: np.ndarray) -> np.ndarray:
    """Van der Corput sequence in base 2 (Hammersley's second coordinate)."""
    b = np.asarray(i, np.uint32)
    b = ((b << 16) | (b >> 16)).astype(np.uint32)
    b = (((b & 0x55555555) << 1) | ((b & 0xAAAAAAAA) >> 1)).astype(np.uint32)
    b = (((b & 0x33333333) << 2) | ((b & 0xCCCCCCCC) >> 2)).astype(np.uint32)
    b = (((b & 0x0F0F0F0F) << 4) | ((b & 0xF0F0F0F0) >> 4)).astype(np.uint32)
    b = (((b & 0x00FF00FF) << 8) | ((b & 0xFF00FF00) >> 8)).astype(np.uint32)
    return b.astype(np.float64) / 2.0 ** 32


def _oracle_ao(scene: Scene, X: np.ndarray, n: np.ndarray, texel: np.ndarray, rays: int,
               per_texel: int, seed: int, chunk: int = 50_000) -> np.ndarray:
    """Unoccluded fraction per sample: cosine-weighted Hammersley directions, one set of
    ceil(rays / S^2) * S^2 points per texel shared out among its samples and rotated per
    texel (Cranley-Patterson) -- the estimator of design L15."""
    k = max(1, math.ceil(rays / per_texel))
    total = k * per_texel
    order = np.argsort(texel, kind="stable")
    rank = np.empty(texel.size, np.int64)
    sorted_texel = texel[order]
    starts = np.flatnonzero(np.r_[True, sorted_texel[1:] != sorted_texel[:-1]])
    rank[order] = np.arange(texel.size) - np.repeat(starts, np.diff(np.r_[starts, texel.size]))
    rng = np.random.default_rng(seed)
    shift = rng.random((int(texel.max()) + 1, 2))
    t1 = normalize(np.cross(n, np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])))
    t2 = np.cross(n, t1)
    out = np.empty(texel.size)
    for s in range(0, texel.size, chunk):
        sl = slice(s, s + chunk)
        idx = (rank[sl, None] % per_texel) + per_texel * np.arange(k)[None, :]
        u1 = (idx / total + shift[texel[sl], 0:1]) % 1.0
        u2 = (radical_inverse(idx) + shift[texel[sl], 1:2]) % 1.0
        r = np.sqrt(u1)
        phi = 2 * np.pi * u2
        local = np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(np.maximum(1 - u1, 0))], -1)
        dirs = (local[..., 0:1] * t1[sl, None] + local[..., 1:2] * t2[sl, None]
                + local[..., 2:3] * n[sl, None])
        out[sl] = scene.escapes(X[sl], dirs).mean(1)
    return out


def oracle_bake(scene: Scene, low: LowMesh, res: int, *, window: float, supersample: int = 1,
                basis: str = "mikk", policy: str = "cage", ring_px: float = 2.0,
                ao_rays: int = 0, height: bool = False, mutations: Iterable[str] = (),
                seed: int = 0) -> OracleBake:
    """Bake the ANALYTIC answer onto ``low``: what a perfect baker writes.

    Samples as design L8: S x S per texel, and a texel with no covered sample but within
    ``ring_px`` texels of a triangle gets one texel-centre sample from the nearest one
    with unclamped barycentrics. Rays under ``policy`` within ``window``, colour averaged
    in linear light, unit normals summed and renormalised, misses dropped, push-pull
    fill, 8-bit encode. ``mutations`` inject the bugs listed in :data:`MUTATIONS`.
    """
    mut = frozenset(mutations)
    unknown = mut - set(MUTATIONS)
    if unknown:
        raise ValueError(f"unknown mutations {sorted(unknown)}")
    if basis not in TANGENT_BASES:
        raise ValueError(f"unknown tangent basis {basis!r}")
    m = low
    S = supersample
    half_texel = "half_texel" in mut  # every sample half an OUTPUT texel off
    tid, bary, _ = raster_uv(m.uv, m.faces, res * S, offset=0.5 - 0.5 * S if half_texel else 0.5)
    rows, cols = np.nonzero(tid >= 0)
    fid = tid[rows, cols]
    bw = bary[rows, cols]
    texel = (rows // S) * res + cols // S
    if "no_padding" not in mut and ring_px > 0:
        ring_tid, ring_bary, _ = raster_uv(m.uv, m.faces, res, ring_px=ring_px,
                                           offset=0.0 if half_texel else 0.5)
        uncovered = np.ones(res * res, bool)
        uncovered[texel] = False
        extra = np.flatnonzero((ring_tid.ravel() >= 0) & uncovered)
        fid = np.concatenate([fid, ring_tid.ravel()[extra]])
        bw = np.concatenate([bw, ring_bary.reshape(-1, 3)[extra]])
        texel = np.concatenate([texel, extra])
    P, D = sample_rays(m, fid, bw, policy)
    if "face_rays" in mut:
        D = m.face_normals()[fid]
    X, ok = scene.hit(P, D, window, policy)
    fid, bw, texel, P, D, X = fid[ok], bw[ok], texel[ok], P[ok], D[ok], X[ok]

    n_true = scene.normal(X)
    T, B, N = frame(m, fid, bw, basis, ignore_w="ignore_w" in mut)
    nts = encode(T, B, N, n_true, transpose="transpose" in mut)
    if "flip_green" in mut:
        nts[:, 1] *= -1
    orm = scene.metal_rough(X).copy()
    if ao_rays:
        orm[:, 0] = _oracle_ao(scene, X, n_true, texel, ao_rays, S * S, seed)

    size = res * res
    count = np.bincount(texel, minlength=size)
    mask = (count > 0).reshape(res, res)

    def mean(values: np.ndarray) -> np.ndarray:
        return (scatter_add(texel, values, size) / np.maximum(count, 1)[:, None]).reshape(res, res, -1)

    colour = mean(srgb_to_linear(scene.colour(X)))
    orm_img = mean(orm)
    normal = normalize(scatter_add(texel, nts, size)).reshape(res, res, 3)
    hmap = mean(((X - P) * D).sum(1)).reshape(res, res) if height else None
    if "no_padding" not in mut:
        colour = push_pull(colour, mask)
        orm_img = push_pull(orm_img, mask)
        normal = normalize(push_pull(normal, mask))
        if hmap is not None:
            hmap = push_pull(hmap[..., None], mask)[..., 0]

    base = linear_to_srgb(colour)
    if "gamma_twice" in mut:
        base = linear_to_srgb(base)
    images = {"baseColor": to_u8(base), "metallicRoughness": to_u8(orm_img),
              "normal": to_u8(0.5 * normal + 0.5)}
    if "no_padding" in mut:
        for img in images.values():
            img[~mask] = 0
    if "vflip" in mut:
        images = {k: np.ascontiguousarray(v[::-1]) for k, v in images.items()}
    return OracleBake(mesh=m, images=images, height=hmap, mask=mask,
                      window=window, info={"samples": int(ok.size), "miss": int((~ok).sum()),
                                           "texels": int(mask.sum())})


# ============================================================================
#  GLB 2.0 writer / reader (stdlib + numpy + Pillow)
# ============================================================================

TEXTURE_SLOTS = ("baseColor", "metallicRoughness", "normal", "occlusion", "emissive")
_GLB_MAGIC, _JSON_CHUNK, _BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942
_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
              5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


@dataclass
class GlbMaterial:
    name: str
    images: dict[str, bytes] = field(default_factory=dict)  # TEXTURE_SLOTS -> PNG/JPEG bytes
    base_color_factor: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic_factor: float = 1.0
    roughness_factor: float = 1.0
    occlusion_from_orm: bool = False  # occlusionTexture -> the metallicRoughness image
    double_sided: bool = False
    extras: dict[str, Any] | None = None


@dataclass
class GlbPrimitive:
    mesh: LowMesh
    material: int = 0


def png_bytes(a: np.ndarray) -> bytes:
    """PNG of a uint8 (H, W[, C]) or uint16 (H, W) array."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(a)).save(buf, "PNG", compress_level=6)
    return buf.getvalue()


def _mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    raise ValueError("images must be PNG or JPEG bytes")


def write_glb(path: str | Path, primitives: Sequence[GlbPrimitive],
              materials: Sequence[GlbMaterial], *, generator: str = "Spellcast3D bakecheck") -> int:
    """Spec-conformant GLB (Khronos validator: 0 errors, 0 warnings): one node, one mesh
    with one primitive per entry, PBR materials with embedded images.

    TEXCOORD_0 is written as (u, 1 - v); TANGENT (when the mesh has it) keeps the w
    computed on v-up UVs -- Mikk on v-up UVs with w as-is equals glTF (v-down) Mikk
    with w negated, which is what three.js's computeMikkTSpaceTangents does.
    Sampler LINEAR / LINEAR_MIPMAP_LINEAR / CLAMP_TO_EDGE: an atlas never tiles.
    """
    blobs: list[bytes] = []
    views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    size = 0

    def add_view(data: bytes, target: int | None = None) -> int:
        nonlocal size
        pad = (-size) % 4
        if pad:
            blobs.append(b"\x00" * pad)
            size += pad
        view: dict[str, Any] = {"buffer": 0, "byteOffset": size, "byteLength": len(data)}
        if target:
            view["target"] = target
        blobs.append(data)
        size += len(data)
        views.append(view)
        return len(views) - 1

    def add_accessor(arr: np.ndarray, kind: str, component: int, target: int,
                     bounds: bool = False) -> int:
        acc: dict[str, Any] = {"bufferView": add_view(arr.tobytes(), target),
                               "componentType": component, "count": int(arr.shape[0]), "type": kind}
        if bounds:
            acc["min"] = arr.min(0).tolist()
            acc["max"] = arr.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    gl_primitives = []
    for prim in primitives:
        m = prim.mesh
        attributes = {
            "POSITION": add_accessor(m.pos.astype(np.float32), "VEC3", 5126, 34962, bounds=True),
            "NORMAL": add_accessor(normalize(m.nrm).astype(np.float32), "VEC3", 5126, 34962),
        }
        if m.tan is not None:
            attributes["TANGENT"] = add_accessor(m.tan.astype(np.float32), "VEC4", 5126, 34962)
        uv = np.stack([m.uv[:, 0], 1.0 - m.uv[:, 1]], -1).astype(np.float32)
        attributes["TEXCOORD_0"] = add_accessor(uv, "VEC2", 5126, 34962)
        indices = add_accessor(m.faces.astype(np.uint32).ravel(), "SCALAR", 5125, 34963)
        gl_primitives.append({"attributes": attributes, "indices": indices,
                              "material": prim.material, "mode": 4})

    images: list[dict[str, Any]] = []
    textures: list[dict[str, int]] = []
    gl_materials = []
    for mat in materials:
        tex: dict[str, int] = {}
        for slot in TEXTURE_SLOTS:
            if slot in mat.images:
                data = mat.images[slot]
                images.append({"bufferView": add_view(data), "mimeType": _mime(data), "name": slot})
                textures.append({"source": len(images) - 1, "sampler": 0})
                tex[slot] = len(textures) - 1
        if mat.occlusion_from_orm and "metallicRoughness" in tex:
            tex["occlusion"] = tex["metallicRoughness"]
        pbr: dict[str, Any] = {"baseColorFactor": list(mat.base_color_factor),
                               "metallicFactor": mat.metallic_factor,
                               "roughnessFactor": mat.roughness_factor}
        if "baseColor" in tex:
            pbr["baseColorTexture"] = {"index": tex["baseColor"]}
        if "metallicRoughness" in tex:
            pbr["metallicRoughnessTexture"] = {"index": tex["metallicRoughness"]}
        gl_mat: dict[str, Any] = {"name": mat.name, "pbrMetallicRoughness": pbr,
                                  "doubleSided": mat.double_sided}
        if "normal" in tex:
            gl_mat["normalTexture"] = {"index": tex["normal"], "scale": 1.0}
        if "occlusion" in tex:
            gl_mat["occlusionTexture"] = {"index": tex["occlusion"], "strength": 1.0}
        if "emissive" in tex:
            gl_mat["emissiveTexture"] = {"index": tex["emissive"]}
            gl_mat["emissiveFactor"] = [1.0, 1.0, 1.0]
        if mat.extras:
            gl_mat["extras"] = mat.extras
        gl_materials.append(gl_mat)

    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "mesh"}],
        "meshes": [{"name": "mesh", "primitives": gl_primitives}],
        "materials": gl_materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": size + (-size) % 4}],
    }
    if images:
        gltf["images"] = images
        gltf["textures"] = textures
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    bin_data = b"".join(blobs) + b"\x00" * ((-size) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_data)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", _GLB_MAGIC, 2, total))
        fh.write(struct.pack("<II", len(js), _JSON_CHUNK))
        fh.write(js)
        fh.write(struct.pack("<II", len(bin_data), _BIN_CHUNK))
        fh.write(bin_data)
    return total


@dataclass
class BakedGlb:
    """A GLB's triangles merged into one :class:`LowMesh`, and its one material."""

    mesh: LowMesh
    images: dict[str, np.ndarray]  # TEXTURE_SLOTS -> uint8 (H, W, 3|4)
    material: dict[str, Any]
    gltf: dict[str, Any]

    @property
    def spellcast(self) -> dict[str, Any]:
        return self.material.get("extras", {}).get("spellcast", {})

    @property
    def tangent_basis(self) -> str:
        """The decoder the bake was encoded for; per-fragment Mikk when undeclared."""
        return str(self.spellcast.get("tangent_basis", "mikk"))

    @property
    def ray_distance(self) -> float | None:
        d = self.spellcast.get("ray_distance")
        return None if d is None else float(d)

    @property
    def normal_scale(self) -> float:
        return float(self.material.get("normalTexture", {}).get("scale", 1.0))

    @property
    def resolution(self) -> int:
        img = self.images.get("normal", self.images.get("baseColor"))
        if img is None:
            raise ValueError("the GLB has neither a normal nor a base colour texture")
        return int(img.shape[0])


def _node_matrix(node: Mapping[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], np.float64).reshape(4, 4).T
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3] = rot * np.asarray(node.get("scale", (1.0, 1.0, 1.0)), np.float64)[None, :]
    M[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
    return M


def _mesh_instances(gltf: Mapping[str, Any]) -> list[tuple[int, np.ndarray]]:
    """(mesh index, world matrix) for every node of the default scene that has a mesh."""
    nodes = gltf.get("nodes", [])
    if "scenes" in gltf:
        roots = gltf["scenes"][gltf.get("scene", 0)].get("nodes", [])
    else:
        children = {c for n in nodes for c in n.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]
    out: list[tuple[int, np.ndarray]] = []
    stack = [(i, np.eye(4)) for i in roots]
    while stack:
        i, parent = stack.pop()
        M = parent @ _node_matrix(nodes[i])
        if "mesh" in nodes[i]:
            out.append((nodes[i]["mesh"], M))
        stack.extend((c, M) for c in nodes[i].get("children", []))
    return out


def _accessor(gltf: Mapping[str, Any], binc: bytes, index: int) -> np.ndarray:
    a = gltf["accessors"][index]
    dtype = np.dtype(_COMPONENT[a["componentType"]])
    width = _WIDTH[a["type"]]
    view = gltf["bufferViews"][a["bufferView"]]
    offset = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = view.get("byteStride") or dtype.itemsize * width
    arr = np.ndarray((a["count"], width), dtype, buffer=binc, offset=offset,
                     strides=(stride, dtype.itemsize)).astype(np.float64 if dtype.kind == "f" else np.int64)
    if a.get("normalized"):
        arr = arr / float(np.iinfo(dtype).max)
    return arr[:, 0] if width == 1 else arr


def _decode_image(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
        return np.asarray(img.convert("RGBA" if alpha else "RGB"))


def _image_bytes(gltf: Mapping[str, Any], binc: bytes, base: Path, texture: int) -> bytes:
    tex = gltf["textures"][texture]
    source = tex.get("source")
    if source is None:
        source = tex.get("extensions", {}).get("EXT_texture_webp", {}).get("source")
    image = gltf["images"][source]
    if "bufferView" in image:
        view = gltf["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        return binc[start:start + view["byteLength"]]
    uri = image["uri"]
    if uri.startswith("data:"):
        return base64.b64decode(uri.split(",", 1)[1])
    return (base / uri).read_bytes()


def read_glb(path: str | Path) -> BakedGlb:
    """All triangle primitives of the default scene, node transforms applied, merged into
    one mesh; the images of their (single) material decoded to uint8 arrays."""
    path = Path(path)
    data = path.read_bytes()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC or version != 2:
        raise ValueError(f"{path} is not a GLB 2.0 file")
    gltf: dict[str, Any] | None = None
    binc = b""
    at = 12
    while at < len(data):
        length, kind = struct.unpack_from("<II", data, at)
        chunk = data[at + 8:at + 8 + length]
        if kind == _JSON_CHUNK:
            gltf = json.loads(chunk)
        elif kind == _BIN_CHUNK:
            binc = chunk
        at += 8 + length
    if gltf is None:
        raise ValueError(f"{path} has no JSON chunk")

    parts: dict[str, list[np.ndarray]] = {"pos": [], "nrm": [], "uv": [], "tan": [], "faces": []}
    material_ids: set[int] = set()
    has_tangents = True
    offset = 0
    for mesh_index, M in _mesh_instances(gltf):
        normal_matrix = np.linalg.inv(M[:3, :3]).T
        for prim in gltf["meshes"][mesh_index]["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            at_ = prim["attributes"]
            missing = {"POSITION", "NORMAL", "TEXCOORD_0"} - set(at_)
            if missing:
                raise ValueError(f"{path}: a primitive has no {', '.join(sorted(missing))}")
            pos = _accessor(gltf, binc, at_["POSITION"])
            count = pos.shape[0]
            faces = (_accessor(gltf, binc, prim["indices"]).reshape(-1, 3) if "indices" in prim
                     else np.arange(count).reshape(-1, 3))
            parts["pos"].append(pos @ M[:3, :3].T + M[:3, 3])
            parts["nrm"].append(normalize(_accessor(gltf, binc, at_["NORMAL"]) @ normal_matrix.T))
            uv = _accessor(gltf, binc, at_["TEXCOORD_0"])
            parts["uv"].append(np.stack([uv[:, 0], 1.0 - uv[:, 1]], -1))
            if "TANGENT" in at_:
                tan = _accessor(gltf, binc, at_["TANGENT"])
                parts["tan"].append(np.concatenate([normalize(tan[:, :3] @ M[:3, :3].T), tan[:, 3:4]], -1))
            else:
                has_tangents = False
            parts["faces"].append(faces + offset)
            offset += count
            material_ids.add(prim.get("material", -1))
    if not parts["faces"]:
        raise ValueError(f"{path} has no triangles")
    if len(material_ids) != 1 or -1 in material_ids:
        raise ValueError(f"{path}: expected every primitive to use one baked material, "
                         f"found {sorted(material_ids)}")
    material = gltf["materials"][material_ids.pop()]
    pos = np.concatenate(parts["pos"])
    mesh = LowMesh(pos=pos, nrm=np.concatenate(parts["nrm"]), uv=np.concatenate(parts["uv"]),
                   faces=np.concatenate(parts["faces"]).astype(np.int64),
                   tan=np.concatenate(parts["tan"]) if has_tangents else None)
    extent = float(np.linalg.norm(pos.max(0) - pos.min(0))) or 1.0
    q = np.round(pos / (1e-7 * extent)).astype(np.int64)
    mesh.weld = np.unique(q, axis=0, return_inverse=True)[1].ravel()

    pbr = material.get("pbrMetallicRoughness", {})
    refs = {"baseColor": pbr.get("baseColorTexture"), "metallicRoughness": pbr.get("metallicRoughnessTexture"),
            "normal": material.get("normalTexture"), "occlusion": material.get("occlusionTexture"),
            "emissive": material.get("emissiveTexture")}
    images = {slot: _decode_image(_image_bytes(gltf, binc, path.parent, ref["index"]))
              for slot, ref in refs.items() if ref}
    return BakedGlb(mesh=mesh, images=images, material=material, gltf=gltf)


# ============================================================================
#  evaluator (black box: output GLB + analytic scene)
# ============================================================================


def bilinear(img: np.ndarray, uv: np.ndarray, level: int = 0) -> np.ndarray:
    """GPU-style bilinear lookup with CLAMP_TO_EDGE; uv v-up, texel centres at (i+1/2)/res.
    ``level`` > 0 reads a box-filtered mip (what generateMipmap approximates)."""
    a = np.asarray(img, np.float64)
    if a.ndim == 2:
        a = a[..., None]
    for _ in range(level):
        h, w = a.shape[:2]
        a = a[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2, -1).mean((1, 3))
    h, w = a.shape[:2]
    x = uv[:, 0] * w - 0.5
    y = (1.0 - uv[:, 1]) * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]

    def at(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        return a[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)]

    return (at(y0, x0) * (1 - fx) * (1 - fy) + at(y0, x0 + 1) * fx * (1 - fy)
            + at(y0 + 1, x0) * (1 - fx) * fy + at(y0 + 1, x0 + 1) * fx * fy)


def sample_surface(m: LowMesh, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n area-uniform random points: (triangle ids, barycentrics)."""
    p = m.pos[m.faces]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=-1)
    fid = rng.choice(len(area), size=n, p=area / area.sum())
    s = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    return fid, np.stack([1 - s, s * (1 - r2), s * r2], -1)


def seam_points(m: LowMesh, per_edge: int = 9
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Pairs (fidA, baryA, fidB, baryB) at the same 3D points on both sides of UV seams."""
    assert m.weld is not None
    edges: dict[tuple[int, int], list[tuple[int, int, bool]]] = {}
    for f, tri in enumerate(m.faces):
        for k in range(3):
            wa, wb = int(m.weld[tri[k]]), int(m.weld[tri[(k + 1) % 3]])
            edges.setdefault((min(wa, wb), max(wa, wb)), []).append((f, k, wa < wb))
    fa, ba, fb, bb = [], [], [], []
    ts = (np.arange(per_edge) + 0.5) / per_edge

    def bary_on_edge(k: int, forward: bool, t: float) -> np.ndarray:
        out = np.zeros(3)
        along = t if forward else 1 - t
        out[k] = 1 - along
        out[(k + 1) % 3] = along
        return out

    for pair in edges.values():
        if len(pair) != 2:
            continue
        (f1, k1, o1), (f2, k2, o2) = pair
        t1, t2 = m.faces[f1], m.faces[f2]
        if t1[k1] == t2[(k2 + 1) % 3] and t1[(k1 + 1) % 3] == t2[k2]:
            continue  # the two triangles share render vertices: not a UV seam
        for t in ts:
            fa.append(f1)
            ba.append(bary_on_edge(k1, o1, t))
            fb.append(f2)
            bb.append(bary_on_edge(k2, o2, t))
    if not fa:
        return None
    return np.array(fa), np.array(ba), np.array(fb), np.array(bb)


def uv_stretch(m: LowMesh, res: int) -> dict[str, float]:
    """Per-face singular values of d(texels)/d(world): density and anisotropy."""
    p = m.pos[m.faces]
    t = m.uv[m.faces] * res
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    n = normalize(np.cross(e1, e2))
    x = normalize(e1)
    y = np.cross(n, x)
    P2 = np.stack([np.stack([(e1 * x).sum(-1), (e1 * y).sum(-1)], -1),
                   np.stack([(e2 * x).sum(-1), (e2 * y).sum(-1)], -1)], -1)
    T2 = np.stack([t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]], -1)
    sv = np.linalg.svd(T2 @ np.linalg.inv(P2), compute_uv=False)
    dens = np.sqrt(sv[:, 0] * sv[:, 1])
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=-1)
    order = np.argsort(dens)
    cum = np.cumsum(area[order])
    med = float(dens[order][np.searchsorted(cum, 0.5 * cum[-1])])
    return {"anisotropy_p95": float(np.percentile(sv[:, 0] / sv[:, 1], 95)),
            "anisotropy_max": float((sv[:, 0] / sv[:, 1]).max()),
            "density_rel_p5": float(np.percentile(dens / med, 5)),
            "density_rel_p95": float(np.percentile(dens / med, 95)),
            "texels_per_unit_median": med}


def near_chart_border(m: LowMesh, fid: np.ndarray, bary: np.ndarray, res: int, px: float) -> np.ndarray:
    """True for samples within ``px`` texels (in UV) of a boundary edge of their chart."""
    edge_use: dict[tuple[int, int], int] = {}
    for tri in m.faces:
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            key = (min(a, b), max(a, b))
            edge_use[key] = edge_use.get(key, 0) + 1
    boundary = np.array([k for k, c in edge_use.items() if c == 1])
    near = np.zeros(fid.shape, bool)
    if boundary.size == 0:
        return near
    uv = (m.uv[m.faces[fid]] * bary[..., None]).sum(1) * res
    A = m.uv[boundary[:, 0]] * res
    AB = m.uv[boundary[:, 1]] * res - A
    for s in range(0, uv.shape[0], 2048):
        q = uv[s:s + 2048, None, :] - A[None]
        t = np.clip((q * AB[None]).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-20), 0, 1)
        d = np.sqrt(((q - t[..., None] * AB[None]) ** 2).sum(-1)).min(1)
        near[s:s + 2048] = d < px
    return near


def _map_normals(g: BakedGlb, values: np.ndarray) -> np.ndarray:
    """8-bit normal texel values -> tangent-space vectors, normalTexture.scale applied."""
    mapN = values / 255.0 * 2 - 1
    mapN[:, :2] *= g.normal_scale
    return mapN


def evaluate(scene: Scene, glb_path: str | Path, *, policy: str = "cage", basis: str | None = None,
             n: int = 120_000, seed: int = 1, mip_levels: Sequence[int] = (1, 2, 3, 4)) -> dict[str, Any]:
    """Score a baked GLB at ``n`` random surface points (design section 7, L1/L2 metrics).

    The ray window is the GLB's ``extras.spellcast.ray_distance`` (else the scene's);
    the normal map is decoded in ``basis`` (default: the basis the GLB declares) and that
    decode is ``normal_err_deg``, the gated one. Both bases, the no-TANGENT decode and
    the error with no normal map at all are reported alongside.
    """
    g = read_glb(glb_path)
    m = g.mesh
    basis = basis or g.tangent_basis
    window = g.ray_distance or scene.t_range
    res = g.resolution
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"glb": str(glb_path), "policy": policy, "tangent_basis": basis,
                           "ray_distance": window, "resolution": res,
                           "has_tangents": m.tan is not None}

    fid, bary = sample_surface(m, n, rng)
    P, D = sample_rays(m, fid, bary, policy)
    uv = (m.uv[m.faces[fid]] * bary[..., None]).sum(1)
    X, ok = scene.hit(P, D, window, policy)
    out["ray_miss_fraction"] = float((~ok).mean())
    out["multi_crossing_fraction"] = float((scene.crossings(P[:20000], D[:20000], window) > 1).mean())
    n_true = scene.normal(X)
    colour_true = scene.colour(X) * 255

    if "baseColor" in g.images:
        err = np.abs(bilinear(g.images["baseColor"][..., :3], uv) - colour_true).max(-1)[ok]
        out["base_err_lsb"] = stats(err)
        out["base_gross_fraction"] = float((err > 24).mean())
    if "metallicRoughness" in g.images:
        mr = bilinear(g.images["metallicRoughness"][..., :3], uv)
        out["mr_err_lsb"] = stats(np.abs(mr - scene.metal_rough(X) * 255)[:, 1:].max(-1)[ok])

    if "normal" in g.images and m.tan is not None:
        mapN = _map_normals(g, bilinear(g.images["normal"][..., :3], uv))
        for label in TANGENT_BASES:
            decoded = decode(*frame(m, fid, bary, label), mapN)
            out[f"normal_err_deg[{label}]"] = stats(angle_deg(decoded, n_true)[ok])
        out["normal_err_deg"] = out[f"normal_err_deg[{basis}]"]
        derivative = decode(*frame_derivative(m, fid, bary), mapN)
        out["normal_err_deg[no_tangent]"] = stats(angle_deg(derivative, n_true)[ok])
        out["normal_err_deg[no_normal_map]"] = stats(
            angle_deg(normalize((m.nrm[m.faces[fid]] * bary[..., None]).sum(1)), n_true)[ok])

        border = near_chart_border(m, fid, bary, res, 3.0)
        T, B, N = frame(m, fid, bary, basis)
        mip: dict[int, dict[str, float | None]] = {}
        for level in mip_levels:
            mapL = _map_normals(g, bilinear(g.images["normal"][..., :3], uv, level))
            ang = angle_deg(decode(T, B, N, mapL), n_true)
            row: dict[str, float | None] = {
                "normal_deg_p99_interior": float(np.percentile(ang[ok & ~border], 99)),
                "normal_deg_p99_border": (float(np.percentile(ang[ok & border], 99))
                                          if (ok & border).any() else None)}
            if "baseColor" in g.images:
                ce = np.abs(bilinear(g.images["baseColor"][..., :3], uv, level) - colour_true).max(-1)
                row["base_lsb_p99_interior"] = float(np.percentile(ce[ok & ~border], 99))
                row["base_lsb_p99_border"] = (float(np.percentile(ce[ok & border], 99))
                                              if (ok & border).any() else None)
            mip[level] = row
        out["mip"] = mip

        seams = seam_points(m)
        if seams is not None:
            fa, ba, fb, bb = seams
            ua = (m.uv[m.faces[fa]] * ba[..., None]).sum(1)
            ub = (m.uv[m.faces[fb]] * bb[..., None]).sum(1)
            na = decode(*frame(m, fa, ba, basis), _map_normals(g, bilinear(g.images["normal"][..., :3], ua)))
            nb = decode(*frame(m, fb, bb, basis), _map_normals(g, bilinear(g.images["normal"][..., :3], ub)))
            out["seam_normal_jump_deg"] = stats(angle_deg(na, nb))
            if "baseColor" in g.images:
                ca = bilinear(g.images["baseColor"][..., :3], ua)
                cb = bilinear(g.images["baseColor"][..., :3], ub)
                out["seam_colour_jump_lsb"] = stats(np.abs(ca - cb).max(-1))
            out["seam_samples"] = int(fa.size)
    out["uv_stretch"] = uv_stretch(m, res)
    return out


def evaluate_texels(scene: Scene, glb_path: str | Path, *, ring_px: float = 1.0,
                    policy: str = "cage", basis: str | None = None) -> dict[str, Any]:
    """Deterministic coverage check: EVERY texel whose centre lies inside a UV triangle
    of the exported mesh (rasterised here, independently of the baker) must hold the
    right value, and so must the ``ring_px`` ring bilinear filtering reads at chart
    edges. Holes and bleeding show up as gross texels (> 10 deg, > 24 LSB)."""
    g = read_glb(glb_path)
    m = g.mesh
    basis = basis or g.tangent_basis
    window = g.ray_distance or scene.t_range
    res = g.resolution
    tid, bary, dist = raster_uv(m.uv, m.faces, res, ring_px=ring_px)
    out: dict[str, Any] = {}
    for label, sel in (("inside", (tid >= 0) & (dist == 0)), ("edge_ring", (tid >= 0) & (dist > 0))):
        fid, bw = tid[sel], bary[sel]
        P, D = sample_rays(m, fid, bw, policy)
        X, ok = scene.hit(P, D, window, policy)
        row: dict[str, Any] = {"texels": int(sel.sum())}
        if "normal" in g.images and m.tan is not None:
            ang = angle_deg(decode(*frame(m, fid, bw, basis),
                                   _map_normals(g, g.images["normal"][..., :3][sel].astype(np.float64))),
                            scene.normal(X))
            row["normal_deg"] = stats(ang[ok])
            row["gross_normal"] = int((ang[ok] > 10).sum())
        if "baseColor" in g.images:
            ce = np.abs(g.images["baseColor"][..., :3][sel].astype(np.float64) - scene.colour(X) * 255).max(-1)
            row["base_lsb"] = stats(ce[ok])
            row["gross_base"] = int((ce[ok] > 24).sum())
        out[label] = row
    return out


#: The 8-bit image of "the float normal is within 0.05 deg of the exact one": such a
#: float moves each encoded component by at most 255/2 * rad(0.05) = 0.11 LSB, so its
#: byte lies within 0.5 + 0.11 LSB of the exact encoded value.
EXACT_NORMAL_LSB = 0.5 + 127.5 * math.radians(0.05)
EXACT_COLOUR_LSB = 1.0


def evaluate_exact(scene: Scene, glb_path: str | Path, supersample: int, *,
                   policy: str = "cage") -> dict[str, Any]:
    """Texel-exact comparison with the analytic answer (the design's analytic sphere).

    Re-creates the baker's own sample pattern on the exported triangles -- S x S samples
    per texel at x = c + (i + 1/2)/S -- casts every sample against the analytic surface,
    averages colour in linear light and sums the exactly-encoded unit normals as the
    baker's accumulation does, and compares every "interior" texel (all S*S samples
    covered and hit) with the 8-bit output.
    """
    g = read_glb(glb_path)
    m = g.mesh
    basis = g.tangent_basis
    window = g.ray_distance or scene.t_range
    res = g.resolution
    S = supersample
    tid, bary, _ = raster_uv(m.uv, m.faces, res * S)
    rows, cols = np.nonzero(tid >= 0)
    fid, bw = tid[rows, cols], bary[rows, cols]
    texel = (rows // S) * res + cols // S
    P, D = sample_rays(m, fid, bw, policy)
    X, ok = scene.hit(P, D, window, policy)
    size = res * res
    interior = (np.bincount(texel, minlength=size) == S * S) & (np.bincount(texel[ok], minlength=size) == S * S)
    fid, bw, texel, X = fid[ok], bw[ok], texel[ok], X[ok]
    count = np.maximum(np.bincount(texel, minlength=size), 1)[:, None]

    lin = scatter_add(texel, srgb_to_linear(scene.colour(X)), size) / count
    expected_colour = linear_to_srgb(lin) * 255
    baked_colour = g.images["baseColor"][..., :3].reshape(-1, 3).astype(np.float64)
    colour_err = np.abs(baked_colour - expected_colour).max(-1)[interior]

    nts = encode(*frame(m, fid, bw, basis), scene.normal(X))
    expected_n = normalize(scatter_add(texel, nts, size))
    baked_n = g.images["normal"][..., :3].reshape(-1, 3).astype(np.float64)
    lsb = (baked_n - 255 * (0.5 * expected_n + 0.5))[interior]
    return {"interior_texels": int(interior.sum()), "tangent_basis": basis,
            "colour_lsb": stats(colour_err),
            "normal_lsb": stats(np.abs(lsb).max(-1)),
            "normal_rms_lsb": float(np.sqrt((lsb ** 2).mean())),
            "normal_deg_8bit": stats(angle_deg(baked_n[interior] / 255 * 2 - 1, expected_n[interior]))}


# ============================================================================
#  gates (design section 7, L1/L2)
# ============================================================================


@dataclass(frozen=True)
class Gate:
    metric: str
    stat: str | None
    hard: float  # FAIL above (or at, when strict)
    warn: float  # WARN above: ~2x the floor a correct 8-bit baker reaches
    strict: bool = False
    optional: bool = False  # absent -> SKIP (seam metrics need a UV seam)

    @property
    def name(self) -> str:
        return self.metric + (f".{self.stat}" if self.stat else "")

    def value(self, ev: Mapping[str, Any]) -> float | None:
        v = ev.get(self.metric)
        if v is None:
            return None
        return float(v[self.stat] if self.stat else v)

    def fails(self, value: float) -> bool:
        return value >= self.hard if self.strict else value > self.hard


GATES: tuple[Gate, ...] = (
    Gate("ray_miss_fraction", None, 1e-3, 0.0),
    Gate("base_err_lsb", "median", 1.0, 0.5),
    Gate("base_err_lsb", "p99", 3.0, 1.0),
    Gate("base_gross_fraction", None, 1e-4, 0.0),
    Gate("mr_err_lsb", "p99", 3.0, 1.0),
    Gate("normal_err_deg", "median", 1.0, 0.3, strict=True),
    Gate("normal_err_deg", "p99", 3.0, 0.8, strict=True),
    Gate("seam_normal_jump_deg", "p99", 1.5, 0.8, optional=True),
    Gate("seam_colour_jump_lsb", "p99", 3.0, 1.5, optional=True),
)
#: Mip levels whose chart-border bleeding FAILs; deeper levels only WARN.
MIP_HARD_LEVELS = 2
MIP_NORMAL_MARGIN_DEG = 2.0
MIP_COLOUR_MARGIN_LSB = 8.0


@dataclass(frozen=True)
class Line:
    tag: str  # "ok", "WARN", "FAIL", "SKIP", "info"
    text: str


@dataclass(frozen=True)
class Verdict:
    lines: tuple[Line, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def failures(self) -> list[str]:
        return [ln.text for ln in self.lines if ln.tag == "FAIL"]

    def __str__(self) -> str:
        body = "\n".join(f"{ln.tag:<4} {ln.text}" for ln in self.lines)
        return f"{body}\nRESULT: {'PASS' if self.passed else 'FAIL'}"


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3g}"


def score(scene: Scene, glb_path: str | Path, *, policy: str = "cage",
          basis: str | None = None) -> tuple[Verdict, dict[str, Any]]:
    """:func:`evaluate` + :func:`evaluate_texels` + :func:`check` in one call; the report
    is the evaluate dict with the texel check under ``"texels"``."""
    ev = evaluate(scene, glb_path, policy=policy, basis=basis)
    ev["texels"] = evaluate_texels(scene, glb_path, policy=policy, basis=basis)
    return check(ev, ev["texels"]), ev


def check(ev: Mapping[str, Any], texels: Mapping[str, Any] | None = None) -> Verdict:
    """Apply :data:`GATES`, the mip border rule and (given ``evaluate_texels`` output)
    the zero-gross-texel rule to an :func:`evaluate` result."""
    lines: list[Line] = []
    if not ev.get("has_tangents", True):
        lines.append(Line("FAIL", "the GLB has no TANGENT attribute (three.js then errs ~13 deg "
                                  "on stretched charts)"))
    for gate in GATES:
        value = gate.value(ev)
        if value is None:
            lines.append(Line("SKIP" if gate.optional else "FAIL", f"{gate.name}: not measured"))
            continue
        tag = "FAIL" if gate.fails(value) else ("WARN" if value > gate.warn else "ok")
        bound = "<" if gate.strict else "<="
        lines.append(Line(tag, f"{gate.name} = {value:.4g} (hard {bound} {gate.hard}, expected <= {gate.warn})"))
    for level, row in ev.get("mip", {}).items():
        nb, ni = row.get("normal_deg_p99_border"), row.get("normal_deg_p99_interior")
        cb, ci = row.get("base_lsb_p99_border"), row.get("base_lsb_p99_interior")
        bad = ((nb is not None and ni is not None and nb > ni + MIP_NORMAL_MARGIN_DEG)
               or (cb is not None and ci is not None and cb > ci + MIP_COLOUR_MARGIN_LSB))
        tag = ("FAIL" if int(level) <= MIP_HARD_LEVELS else "WARN") if bad else "ok"
        lines.append(Line(tag, f"mip{level} border/interior p99: normal {_fmt(nb)}/{_fmt(ni)} deg, "
                               f"colour {_fmt(cb)}/{_fmt(ci)} LSB"))
    if texels is not None:
        for label, row in texels.items():
            gross = int(row.get("gross_normal", 0)) + int(row.get("gross_base", 0))
            lines.append(Line("FAIL" if gross else "ok",
                              f"texels[{label}]: {gross} gross of {row['texels']} (normal > 10 deg: "
                              f"{row.get('gross_normal')}, colour > 24 LSB: {row.get('gross_base')})"))
    if "normal_err_deg[no_tangent]" in ev:
        lines.append(Line("info", f"basis {ev['tangent_basis']}; other decodes p99: "
                          + ", ".join(f"{b} {ev[f'normal_err_deg[{b}]']['p99']:.3g}" for b in TANGENT_BASES)
                          + f", no TANGENT {ev['normal_err_deg[no_tangent]']['p99']:.3g}, "
                            f"no normal map {ev['normal_err_deg[no_normal_map]']['p99']:.3g} deg"))
    stretch = ev.get("uv_stretch", {})
    lines.append(Line("info", f"multi_crossing_fraction={ev.get('multi_crossing_fraction', 0):.4g} "
                              f"policy={ev.get('policy')} ray_distance={_fmt(ev.get('ray_distance'))} "
                              f"uv anisotropy p95={_fmt(stretch.get('anisotropy_p95'))} "
                              f"density p5/p95={_fmt(stretch.get('density_rel_p5'))}/"
                              f"{_fmt(stretch.get('density_rel_p95'))}"))
    return Verdict(tuple(lines))
