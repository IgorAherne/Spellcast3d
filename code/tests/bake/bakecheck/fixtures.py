"""Bake inputs with an analytic truth, and the two bakers the tests drive.

A fixture ("case") is an analytic scene plus the files a baker reads: the high poly as
a textured GLB and the low poly as the retopo stage writes it -- quads with per-corner
v-up UVs and no ``vn``, written by ``instant_meshes_brush.uv.write_obj``. Everything is
generated at test time into a cache folder and never committed:

    python -m bakecheck.fixtures <dir>                 every fixture, to run a baker by hand
    python -m bakecheck.fixtures --normcheck <dir>     plane bakes + manifest for web/normcheck.html

Bakers: :class:`OracleBaker` writes the analytic answer (optionally with mutations);
:class:`SpellbakeBaker` runs the real ``spellbake.bake`` once that package is importable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import bakecheck as bc

#: Bump when a generator changes, so stale cached fixtures are rebuilt.
FIXTURE_VERSION = 1
PLANE_LAYOUTS = ("identity", "rot90", "rot37", "mirror", "aniso", "shear", "split")
LAUNCHERS = Path(__file__).resolve().parents[3] / "spellcast" / "launchers"


def cache_root(config: Any) -> Path:
    """Fixture folder inside pytest's cache (``config`` is ``request.config``), or in the
    system temp folder when the cache plugin is disabled (``-p no:cacheprovider``)."""
    cache = getattr(config, "cache", None)
    base = Path(cache.mkdir("bakecheck")) if cache is not None else Path(tempfile.gettempdir()) / "bakecheck"
    return base / f"v{FIXTURE_VERSION}"


@dataclass(frozen=True)
class Case:
    """One bake input: the analytic truth and the files a baker reads."""

    name: str
    scene: bc.Scene
    high: Path
    low: Path
    diag: float  # bounding-box diagonal of the high poly


# ============================================================================
#  writing files (atomically: a cache hit is always a complete file)
# ============================================================================


def _ensure(path: Path, build: Callable[[Path], object]) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        build(tmp)
        os.replace(tmp, path)
    return path


def _uv() -> Any:
    """``instant_meshes_brush.uv``: tests importorskip it, since the lows depend on it."""
    return importlib.import_module("instant_meshes_brush.uv")


def _write_low(path: Path, V: np.ndarray, faces: np.ndarray, layout: Any = None) -> None:
    uv = _uv()
    if layout is None:
        layout = uv.unwrap(V.astype(np.float32), faces)
    uv.write_obj(path, V, faces, layout)


def _write_packed_low(path: Path, V: np.ndarray, F: np.ndarray, resolution: int,
                      padding: int = 8) -> None:
    """A triangle low unwrapped by xatlas WITH atlas padding (what the Projection stage's
    repack produces), written by uv.write_obj."""
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(V.astype(np.float32), F.astype(np.uint32))
    pack = xatlas.PackOptions()
    pack.resolution = resolution
    pack.padding = padding
    pack.bilinear = True
    atlas.generate(pack_options=pack)
    vmap, ind, uvs = atlas[0]
    uvs = np.asarray(uvs, np.float64)
    if uvs.max() > 1.0 + 1e-4:
        uvs = uvs / max(atlas.width, atlas.height)
    uvs[:, 1] = 1.0 - uvs[:, 1]
    ind = np.asarray(ind, np.int64)
    _uv().write_obj(path, V, np.asarray(vmap, np.int64)[ind], _layout(uvs, ind))


def _layout(uvs: np.ndarray, corner_uv: np.ndarray) -> Any:
    """A hand-made UvLayout, so write_obj emits exactly these per-corner UVs."""
    uv = _uv()
    count = corner_uv.shape[0]
    return uv.UvLayout(uv=uvs.astype(np.float32), faces=corner_uv.astype(np.int32),
                       chart=np.zeros(count, np.int32), cut=np.zeros((0, 3), np.int32),
                       cut_face=np.zeros(0, np.int32), cut_slot=np.zeros((0, 3), np.int32),
                       cut_chart=np.zeros(0, np.int32), chart_count=1, leniency=uv.DEFAULT_LENIENCY)


def _solid_png(colour: tuple[int, int, int]) -> bytes:
    return bc.png_bytes(np.tile(np.array(colour, np.uint8), (8, 8, 1)))


def _flat_mesh(pos: np.ndarray, faces: np.ndarray, normal: Iterable[float],
               uv: np.ndarray | None = None) -> bc.LowMesh:
    """A flat-shaded high-poly part: every vertex carries the face-plane normal."""
    return bc.LowMesh(pos=pos, nrm=np.tile(np.asarray(list(normal), np.float64), (pos.shape[0], 1)),
                      uv=np.zeros((pos.shape[0], 2)) if uv is None else uv, faces=faces)


def glb_diag(path: Path) -> float:
    """Bounding-box diagonal from the POSITION accessors' min/max (single-node files)."""
    data = path.read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + length])
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for mesh in gltf["meshes"]:
        for prim in mesh["primitives"]:
            acc = gltf["accessors"][prim["attributes"]["POSITION"]]
            lo = np.minimum(lo, acc["min"])
            hi = np.maximum(hi, acc["max"])
    return float(np.linalg.norm(hi - lo))


def _write_scene_json(folder: Path, scene: bc.Scene) -> None:
    _ensure(folder / "scene.json",
            lambda p: p.write_text(json.dumps({"scene": scene.spec()}, indent=1), encoding="utf-8"))


def _grid(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """(nx+1)(ny+1) points of the unit square (x fastest) and CCW quads (+z up)."""
    xs, ys = np.linspace(0, 1, nx + 1), np.linspace(0, 1, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    idx = np.arange(X.size).reshape(ny + 1, nx + 1)
    quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    return np.stack([X.ravel(), Y.ravel()], -1), quads


def _tris(quads: np.ndarray) -> np.ndarray:
    return np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])


# ============================================================================
#  high polys of the analytic scenes (tessellated finely, textured at 2048)
# ============================================================================


def _write_high(path: Path, X, N, uv, F, colour, mr) -> None:
    material = bc.GlbMaterial("high", images={"baseColor": bc.png_bytes(bc.to_u8(colour)),
                                              "metallicRoughness": bc.png_bytes(bc.to_u8(mr))})
    bc.write_glb(path, [bc.GlbPrimitive(bc.LowMesh(pos=X, nrm=N, uv=uv, faces=F))], [material])


def _grid_sphere(scene: bc.SphereScene, path: Path, nt: int = 256, nphi: int = 512,
                 tex: tuple[int, int] = (2048, 1024)) -> None:
    """Lat-long tessellation with the analytic fields sampled at texel centres."""
    th = np.linspace(0, np.pi, nt + 1)
    ph = np.linspace(0, 2 * np.pi, nphi + 1)
    T, Pp = np.meshgrid(th, ph, indexing="ij")
    d = np.stack([np.sin(T) * np.cos(Pp), np.sin(T) * np.sin(Pp), np.cos(T)], -1).reshape(-1, 3)
    X = d * scene.radius(d)[:, None]
    uv = np.stack([Pp.ravel() / (2 * np.pi), 1 - T.ravel() / np.pi], -1)
    idx = np.arange(X.shape[0]).reshape(nt + 1, nphi + 1)
    a, b, c, e = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    F = np.concatenate([np.stack([a, e, c], -1).reshape(-1, 3), np.stack([a, c, b], -1).reshape(-1, 3)])
    W, H = tex
    cc, rr = np.meshgrid((np.arange(W) + 0.5) / W, (np.arange(H) + 0.5) / H)
    tt, pp = np.pi * rr.ravel(), 2 * np.pi * cc.ravel()
    dd = np.stack([np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)], -1)
    XX = dd * scene.radius(dd)[:, None]
    _write_high(path, X, scene.normal(X), uv, F, scene.colour(XX).reshape(H, W, 3),
                scene.metal_rough(XX).reshape(H, W, 3))


def _grid_plane(scene: bc.PlaneScene, path: Path, n: int = 384, lo: float = -0.1,
                hi: float = 1.1, tex: int = 2048) -> None:
    """Height field over [lo, hi]^2 (local), high UVs rotated 90 deg against the lows'."""
    xs = np.linspace(lo, hi, n + 1)
    Xg, Yg = np.meshgrid(xs, xs, indexing="xy")
    X = scene.to_world(np.stack([Xg.ravel(), Yg.ravel(), scene.h(Xg.ravel(), Yg.ravel())], -1))
    xn, yn = (Xg.ravel() - lo) / (hi - lo), (Yg.ravel() - lo) / (hi - lo)
    uv = np.stack([1 - yn, xn], -1)
    idx = np.arange(X.shape[0]).reshape(n + 1, n + 1)
    a, b, c, e = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, e], -1).reshape(-1, 3)])
    cc, rr = np.meshgrid((np.arange(tex) + 0.5) / tex, (np.arange(tex) + 0.5) / tex)
    u, v = cc.ravel(), 1 - rr.ravel()
    yl, xl = lo + (1 - u) * (hi - lo), lo + v * (hi - lo)
    XX = scene.to_world(np.stack([xl, yl, scene.h(xl, yl)], -1))
    _write_high(path, X, scene.normal(X), uv, F, scene.colour(XX).reshape(tex, tex, 3),
                scene.metal_rough(XX).reshape(tex, tex, 3))


# ============================================================================
#  L1/L2 cases: planes with 7 UV layouts, displaced spheres
# ============================================================================


def plane_scene(world: bool = True) -> bc.PlaneScene:
    """The plane placed with a rotation and offset (axis bugs), or in its local frame
    (for the orthographic browser check)."""
    if not world:
        return bc.PlaneScene()
    a = 0.5
    R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64) @ np.array(
        [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return bc.PlaneScene(R=R, c=[0.2, 0.1, -0.3])


def plane_layout(layout: str, n: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Local unit-square grid and one of 7 UV layouts, each a way a tangent frame can be set
    up: identity, rot90, rot37, mirror (w = -1), aniso (2:1 texel density), shear
    (non-orthogonal T/B) and split (two charts, the right one rotated and mirrored, seam
    at x = 0.5). Returns (points (k, 3), quads (q, 4), uvs, per-corner uv indices)."""
    xy, quads = _grid(n, n)
    P = np.concatenate([xy, np.zeros((xy.shape[0], 1))], -1)

    def fit(uv: np.ndarray, lo: float = 0.04, hi: float = 0.96) -> np.ndarray:
        mn, mx = uv.min(0), uv.max(0)
        return (uv - mn) * ((hi - lo) / (mx - mn).max()) + lo

    if layout == "split":
        right = xy[quads, 0].min(1) >= 0.5 - 1e-9
        uv_a = xy * 0.9 + [0.03, 0.05]
        uv_b = np.stack([0.52 + 0.45 * xy[:, 1], 0.05 + 0.9 * (xy[:, 0] - 0.5)], -1)
        return P, quads, np.concatenate([uv_a, uv_b]), np.where(right[:, None], quads + xy.shape[0], quads)
    if layout == "identity":
        uv = fit(xy)
    elif layout == "rot90":
        uv = fit(np.stack([1 - xy[:, 1], xy[:, 0]], -1))
    elif layout == "rot37":
        a = math.radians(37)
        uv = fit(xy @ np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]).T)
    elif layout == "mirror":
        uv = fit(np.stack([1 - xy[:, 0], xy[:, 1]], -1))
    elif layout == "aniso":
        uv = np.stack([xy[:, 0], 0.5 * xy[:, 1]], -1) * 0.92 + 0.04
    elif layout == "shear":
        uv = fit(np.stack([xy[:, 0] + 0.45 * xy[:, 1], xy[:, 1]], -1))
    else:
        raise ValueError(f"unknown plane layout {layout!r}; expected one of {PLANE_LAYOUTS}")
    return P, quads, uv, quads


def plane_case(root: Path, layout: str, *, world: bool = True) -> Case:
    scene = plane_scene(world)
    folder = root / ("plane" if world else "plane_local")
    high = _ensure(folder / "high.glb", lambda p: _grid_plane(scene, p))
    _write_scene_json(folder, scene)
    P, quads, uvs, corner_uv = plane_layout(layout)
    low = _ensure(folder / f"low_{layout}.obj",
                  lambda p: _write_low(p, scene.to_world(P), quads, _layout(uvs, corner_uv)))
    return Case(f"plane_{layout}", scene, high, low, glb_diag(high))


def _cube_sphere_quads(n: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Quad-only sphere with non-planar quads, like an Instant Meshes result."""
    index: dict[tuple[float, ...], int] = {}
    V: list[np.ndarray] = []
    Q: list[list[int]] = []

    def vid(p: np.ndarray) -> int:
        key = tuple(np.round(p, 9))
        if key not in index:
            index[key] = len(V)
            V.append(p)
        return index[key]

    g = np.linspace(-1, 1, n + 1)
    for axis in range(3):
        for sign in (-1, 1):
            for i in range(n):
                for j in range(n):
                    corners = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = np.zeros(3)
                        p[axis] = sign
                        p[(axis + 1) % 3] = g[i + di]
                        p[(axis + 2) % 3] = g[j + dj]
                        corners.append(vid(p / np.linalg.norm(p)))
                    Q.append(corners[::-1] if sign < 0 else corners)
    return np.array(V), np.array(Q, dtype=np.int64)


def sphere_case(root: Path) -> Case:
    """Displaced sphere (a = 0.05) over a 216-quad cube sphere unwrapped by uv.py."""
    scene = bc.SphereScene(a=0.05)
    folder = root / "sphere"
    high = _ensure(folder / "high.glb", lambda p: _grid_sphere(scene, p))
    _write_scene_json(folder, scene)
    low = _ensure(folder / "low.obj", lambda p: _write_low(p, *_cube_sphere_quads(6)))
    return Case("sphere", scene, high, low, glb_diag(high))


def coarse_sphere_case(root: Path) -> Case:
    """The displaced sphere over a 20-triangle icosahedron: vertex normals 63 deg apart,
    so interpolated frames are far from orthonormal and face normals far from smooth
    ones (where the ``transpose`` and ``face_rays`` bugs show). Needs a ray window of
    ~0.3 (10% of the diagonal): the faces sink 0.2 below the surface. Its atlas is
    padded for a 512 bake: 20 unpadded charts bleed into each other from mip 2 on."""
    base = sphere_case(root)
    low = _ensure(base.high.parent / "low_coarse.obj",
                  lambda p: _write_packed_low(p, *bc.icosphere(0), resolution=512))
    return Case("coarse_sphere", base.scene, base.high, low, base.diag)


def exact_sphere_case(root: Path) -> Case:
    """The undisplaced unit sphere over a 320-triangle icosphere with vertices ON it: the
    texel-exact analytic check (design section 7, "analytic sphere")."""
    scene = bc.SphereScene(a=0.0)
    folder = root / "sphere_exact"
    high = _ensure(folder / "high.glb", lambda p: _grid_sphere(scene, p))
    _write_scene_json(folder, scene)
    low = _ensure(folder / "low.obj", lambda p: _write_low(p, *bc.icosphere(2)))
    return Case("sphere_exact", scene, high, low, glb_diag(high))


# ============================================================================
#  L5 cases: ray policy (hoop over planks, parallel sheets), AO trench, height
# ============================================================================

PLANKS = (196, 132, 74)  # sRGB bytes
HOOP = (46, 48, 58)
INNER = (200, 64, 60)
OUTER = (60, 70, 200)
GREY = (128, 128, 128)


@dataclass(frozen=True)
class Layer:
    """An upward-facing horizontal rectangle of the high poly (local = world here)."""

    z: float
    x: tuple[float, float]
    y: tuple[float, float]
    colour: tuple[int, int, int]


class LayersScene(bc.Scene):
    """Horizontal layers stacked in z: several surfaces on one baking ray. The low poly
    lies between them, nearer the inner one; only the cage policy picks the outer."""

    name = "layers"

    def __init__(self, layers: Iterable[Layer]):
        self.layers = tuple(layers)

    def _t(self, P: np.ndarray, D: np.ndarray) -> np.ndarray:
        """(layers, n) ray parameter of each layer's plane, NaN off its rectangle."""
        out = np.full((len(self.layers), P.shape[0]), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            for k, layer in enumerate(self.layers):
                t = (layer.z - P[:, 2]) / D[:, 2]
                X = P + t[:, None] * D
                on = ((X[:, 0] >= layer.x[0]) & (X[:, 0] <= layer.x[1])
                      & (X[:, 1] >= layer.y[0]) & (X[:, 1] <= layer.y[1]) & np.isfinite(t))
                out[k, on] = t[on]
        return out

    def hit(self, P, D, window, policy="cage"):
        t = self._t(P, D)
        with np.errstate(invalid="ignore"):
            if policy == "cage":
                enter = np.where((np.abs(t) <= window) & (D[:, 2] > 0), t, -np.inf).max(0)
                near = np.where(np.abs(t) <= 4 * window, np.abs(t), np.inf)
            else:
                enter = np.full(P.shape[0], -np.inf)
                near = np.where(np.abs(t) <= window, np.abs(t), np.inf)
        k = near.argmin(0)
        fallback = t[k, np.arange(P.shape[0])]
        ok = np.isfinite(enter) | np.isfinite(near.min(0))
        chosen = np.where(np.isfinite(enter), enter, np.where(ok, fallback, 0.0))
        return P + chosen[:, None] * D, ok

    def layer_of(self, X: np.ndarray) -> np.ndarray:
        return np.abs(X[:, 2][None] - np.array([L.z for L in self.layers])[:, None]).argmin(0)

    def colour(self, X):
        return np.array([L.colour for L in self.layers], np.float64)[self.layer_of(X)] / 255

    def normal(self, X):
        return np.tile([0.0, 0.0, 1.0], (X.shape[0], 1))

    def metal_rough(self, X):
        return np.tile([1.0, 0.5, 0.0], (X.shape[0], 1))


class TrenchScene(bc.Scene):
    """Floor z = -H between walls x = 0 and x = W, open above, infinitely long in y.

    Cosine-weighted AO on the floor is the view factor to the opening, an infinite strip:
    AO(x) = 1/2 [x / sqrt(x^2 + H^2) + (W - x) / sqrt((W - x)^2 + H^2)].
    """

    name = "trench"
    t_range = 0.1

    def __init__(self, width: float = 1.0, depth: float = 1.0):
        self.W = width
        self.H = depth

    def F(self, X):
        return X[:, 2] + self.H

    def ao(self, x: np.ndarray) -> np.ndarray:
        W, H = self.W, self.H
        return 0.5 * (x / np.sqrt(x * x + H * H) + (W - x) / np.sqrt((W - x) ** 2 + H * H))

    def escapes(self, X, dirs):
        x = X[:, 0:1]
        dx, dz = dirs[..., 0], dirs[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(dx < 0, x / -dx, np.where(dx > 0, (self.W - x) / dx, np.inf))
            return s * dz >= self.H

    def colour(self, X):
        return np.tile(np.array(GREY, np.float64) / 255, (X.shape[0], 1))

    def metal_rough(self, X):
        return np.tile([1.0, 1.0, 0.0], (X.shape[0], 1))


def _factor(colour: tuple[int, int, int]) -> tuple[float, float, float, float]:
    """glTF baseColorFactor (linear) that renders as these sRGB bytes."""
    r, g, b = bc.srgb_to_linear(np.array(colour, np.float64) / 255)
    return (float(r), float(g), float(b), 1.0)


def _flat_patch(x: tuple[float, float], y: tuple[float, float], z: float, n: int = 12) -> bc.LowMesh:
    xy, quads = _grid(n, n)
    pos = np.stack([x[0] + xy[:, 0] * (x[1] - x[0]), y[0] + xy[:, 1] * (y[1] - y[0]), np.full(xy.shape[0], z)], -1)
    return _flat_mesh(pos, _tris(quads), (0.0, 0.0, 1.0), uv=xy.copy())


def _box(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> bc.LowMesh:
    """Closed box, one flat-shaded quad (own vertices, outward normal) per side."""
    pos, nrm, faces = [], [], []
    for axis in range(3):
        for side, value in ((-1, lo[axis]), (1, hi[axis])):
            u, v = (axis + 1) % 3, (axis + 2) % 3
            quad = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = [0.0, 0.0, 0.0]
                p[axis] = value
                p[u] = hi[u] if du else lo[u]
                p[v] = hi[v] if dv else lo[v]
                quad.append(p)
            if side < 0:
                quad = quad[::-1]
            base = len(pos)
            pos += quad
            normal = [0.0, 0.0, 0.0]
            normal[axis] = float(side)
            nrm += [normal] * 4
            faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return bc.LowMesh(pos=np.array(pos), nrm=np.array(nrm), uv=np.zeros((len(pos), 2)),
                      faces=np.array(faces, np.int64))


def _write_unit_low(path: Path) -> None:
    """The flat low poly of the policy scenes: the unit square at z = 0, 8 x 8 quads, one chart."""
    xy, quads = _grid(8, 8)
    _write_low(path, np.concatenate([xy, np.zeros((xy.shape[0], 1))], -1), quads, _layout(xy * 0.92 + 0.04, quads))


def _write_layers_high(path: Path, textured: Layer, solid: bc.LowMesh, solid_colour: tuple[int, int, int]) -> None:
    """Two materials: a textured layer (solid-colour PNG) and a factor-only mesh, so both
    of the baker's colour paths (map x factor, factor alone) are exercised."""
    bc.write_glb(path, [bc.GlbPrimitive(_flat_patch(textured.x, textured.y, textured.z), 0),
                        bc.GlbPrimitive(solid, 1)],
                 [bc.GlbMaterial("textured", images={"baseColor": _solid_png(textured.colour)},
                                 metallic_factor=0.0, roughness_factor=0.5),
                  bc.GlbMaterial("solid", base_color_factor=_factor(solid_colour),
                                 metallic_factor=0.0, roughness_factor=0.5)])


HOOP_X = (0.4, 0.6)


def hoop_case(root: Path) -> Case:
    """Metal hoop over wooden planks: planks at z = -0.004, a hoop box z in [-0.006, 0.012]
    across x in HOOP_X; the low (z = 0) runs inside the hoop, nearer the planks."""
    planks = Layer(-0.004, (-0.1, 1.1), (-0.1, 1.1), PLANKS)
    top = Layer(0.012, HOOP_X, (-0.1, 1.1), HOOP)
    folder = root / "hoop"
    high = _ensure(folder / "high.glb", lambda p: _write_layers_high(
        p, planks, _box((HOOP_X[0], -0.1, -0.006), (HOOP_X[1], 1.1, top.z)), HOOP))
    low = _ensure(folder / "low.obj", _write_unit_low)
    return Case("hoop", LayersScene([planks, top]), high, low, glb_diag(high))


def sheets_case(root: Path) -> Case:
    """Two parallel sheets, inner at z = -0.003 and outer at z = +0.008, low at z = 0."""
    inner = Layer(-0.003, (-0.1, 1.1), (-0.1, 1.1), INNER)
    outer = Layer(0.008, (-0.1, 1.1), (-0.1, 1.1), OUTER)
    folder = root / "sheets"
    high = _ensure(folder / "high.glb", lambda p: _write_layers_high(
        p, inner, _flat_patch(outer.x, outer.y, outer.z), OUTER))
    low = _ensure(folder / "low.obj", _write_unit_low)
    return Case("sheets", LayersScene([inner, outer]), high, low, glb_diag(high))


TRENCH_LENGTH = 50.0  # >= 50 depths, so escapes through the open ends are negligible
TRENCH_LOW_Y = (-2.0, 2.0)


def _write_trench_high(path: Path, scene: TrenchScene) -> None:
    W, H, half = scene.W, scene.H, TRENCH_LENGTH / 2
    parts = []
    for x, z, normal in (((0.0, W), (-H, -H), (0, 0, 1)),  # floor
                         ((-2.0, 0.0), (0.0, 0.0), (0, 0, 1)), ((W, W + 2.0), (0.0, 0.0), (0, 0, 1))):  # rims
        xy, quads = _grid(2, 50)
        pos = np.stack([x[0] + xy[:, 0] * (x[1] - x[0]), -half + xy[:, 1] * 2 * half,
                        np.full(xy.shape[0], z[0])], -1)
        parts.append(_flat_mesh(pos, _tris(quads), normal))
    for x, normal in ((0.0, (1, 0, 0)), (W, (-1, 0, 0))):  # walls
        yz, quads = _grid(50, 1)
        pos = np.stack([np.full(yz.shape[0], x), -half + yz[:, 0] * 2 * half, -H + yz[:, 1] * H], -1)
        faces = _tris(quads) if normal[0] > 0 else _tris(quads)[:, ::-1]  # (y, z) quads face +x
        parts.append(_flat_mesh(pos, faces, normal))
    bc.write_glb(path, [bc.GlbPrimitive(p) for p in parts],
                 [bc.GlbMaterial("trench", base_color_factor=_factor(GREY), metallic_factor=0.0,
                                 roughness_factor=1.0)])


def trench_case(root: Path) -> Case:
    """Trench W = H = 1, 50 long; the low is a 1 x 4 patch of its floor (anisotropic UVs)."""
    scene = TrenchScene()
    folder = root / "trench"
    high = _ensure(folder / "high.glb", lambda p: _write_trench_high(p, scene))

    def write_low(path: Path) -> None:
        xy, quads = _grid(4, 16)
        pos = np.stack([xy[:, 0] * scene.W, TRENCH_LOW_Y[0] + xy[:, 1] * (TRENCH_LOW_Y[1] - TRENCH_LOW_Y[0]),
                        np.full(xy.shape[0], -scene.H)], -1)
        _write_low(path, pos, quads, _layout(xy * 0.92 + 0.04, quads))

    low = _ensure(folder / "low.obj", write_low)
    return Case("trench", scene, high, low, glb_diag(high))


# ============================================================================
#  bakers
# ============================================================================


@dataclass(frozen=True)
class Recipe:
    """The bake settings a test varies (named as spellbake's BakeSettings fields)."""

    texture_size: int = 1024
    quality: int = 2  # samples per texel = quality^2
    tangent_basis: str = "mikk"
    repack_uvs: bool = False
    ray_distance: float | None = None  # percent of the high poly's diagonal; None = auto
    hard_edge_angle: float = 60.0
    ao: bool = False
    ao_rays: int = 32
    height: bool = False

    def settings(self) -> dict[str, Any]:
        """Keyword arguments for ``spellbake.settings.BakeSettings``."""
        return {**asdict(self), "normal_y": "opengl", "export_obj": False, "export_fbx": False}


class Baker(Protocol):
    name: str

    def bake(self, case: Case, recipe: Recipe, out_dir: Path) -> Path:
        """Bake ``case`` into ``out_dir`` (the layout of spellbake: projection.glb,
        maps/*.png, bake.json) and return the GLB."""
        ...


def write_bake(out_dir: Path, result: bc.OracleBake, recipe: Recipe) -> Path:
    """An oracle bake in the Projection stage's output layout (design section 6)."""
    (out_dir / "maps").mkdir(parents=True, exist_ok=True)
    extras: dict[str, Any] = {"tangent_basis": recipe.tangent_basis, "normal_y": "+Y",
                              "ray_distance": result.window, "version": 1}
    if result.height is not None:
        lo, hi = float(result.height[result.mask].min()), float(result.height[result.mask].max())
        scaled = (result.height - lo) / max(hi - lo, 1e-30)
        (out_dir / "maps" / "height.png").write_bytes(
            bc.png_bytes(np.clip(np.round(scaled * 65535), 0, 65535).astype(np.uint16)))
        extras["height_range"] = [lo, hi]
    material = bc.GlbMaterial("projection", images={k: bc.png_bytes(v) for k, v in result.images.items()},
                              occlusion_from_orm=recipe.ao, extras={"spellcast": extras})
    glb = out_dir / "projection.glb"
    bc.write_glb(glb, [bc.GlbPrimitive(result.mesh)], [material], generator="bakecheck oracle")
    stats = {"miss_fraction": result.info["miss"] / max(result.info["samples"], 1),
             "coverage": result.info["texels"] / result.mask.size, "ray_distance": result.window}
    (out_dir / "bake.json").write_text(json.dumps({"settings": asdict(recipe), "stats": stats}, indent=1),
                                       encoding="utf-8")
    return glb


class OracleBaker:
    """The analytic reference: needs no high-poly file, only the case's scene."""

    name = "oracle"

    def __init__(self, mutations: Iterable[str] = (), policy: str = "cage", ring_px: float = 2.0):
        self.mutations = tuple(mutations)
        self.policy = policy
        self.ring_px = ring_px

    def bake(self, case: Case, recipe: Recipe, out_dir: Path) -> Path:
        window = recipe.ray_distance / 100 * case.diag if recipe.ray_distance else case.scene.t_range
        result = bc.oracle_bake(case.scene, bc.read_obj(case.low), recipe.texture_size, window=window,
                                supersample=recipe.quality, basis=recipe.tangent_basis, policy=self.policy,
                                ring_px=self.ring_px, ao_rays=recipe.ao_rays if recipe.ao else 0,
                                height=recipe.height, mutations=self.mutations)
        return write_bake(out_dir, result, recipe)


def _launchers_on_path() -> None:
    if str(LAUNCHERS) not in sys.path:
        sys.path.insert(0, str(LAUNCHERS))


def spellbake_available() -> bool:
    """True once ``spellbake.bake`` (the bake-pipeline package) imports in this venv."""
    _launchers_on_path()
    try:
        importlib.import_module("spellbake.bake")
        importlib.import_module("spellbake.settings")
    except ImportError:
        return False
    return True


class SpellbakeBaker:
    """The real baker, in process: ``spellbake.bake.bake(high, low, out_dir, settings, ...)``."""

    name = "spellbake"

    def __init__(self, threads: int | None = None):
        self.threads = threads or os.cpu_count() or 4

    def bake(self, case: Case, recipe: Recipe, out_dir: Path) -> Path:
        _launchers_on_path()
        from spellbake.bake import bake
        from spellbake.settings import BakeSettings

        out_dir.mkdir(parents=True, exist_ok=True)
        bake(case.high, case.low, out_dir, BakeSettings(**recipe.settings()),
             progress=lambda fraction, message: None, cancelled=lambda: False, threads=self.threads)
        return out_dir / "projection.glb"


BAKERS = ("oracle", "spellbake")


def make_baker(name: str) -> Baker:
    if name == "oracle":
        return OracleBaker()
    if name == "spellbake":
        return SpellbakeBaker()
    raise ValueError(f"unknown baker {name!r}; expected one of {BAKERS}")


_CHILD = """
import json, sys, time
sys.path.insert(0, sys.argv[1])
from spellbake.bake import bake
from spellbake.settings import BakeSettings
job = json.loads(sys.argv[2])
start = time.perf_counter()
bake(job["high"], job["low"], job["out_dir"], BakeSettings(**job["settings"]),
     progress=lambda fraction, message: None, cancelled=lambda: False, threads=job["threads"])
seconds = time.perf_counter() - start
try:
    import psutil
except ImportError:
    peak = None
else:
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak is None:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
print(json.dumps({"seconds": seconds, "peak_rss": peak}))
"""


def bake_in_child(high: Path, low: Path, out_dir: Path, recipe: Recipe, threads: int) -> dict[str, Any]:
    """Run spellbake in a fresh process; returns {"seconds", "peak_rss" (bytes, None
    without psutil)} measured inside it, so the numbers are the bake's alone."""
    out_dir.mkdir(parents=True, exist_ok=True)
    job = {"high": str(high), "low": str(low), "out_dir": str(out_dir), "settings": recipe.settings(),
           "threads": threads}
    done = subprocess.run([sys.executable, "-c", _CHILD, str(LAUNCHERS), json.dumps(job)],
                          capture_output=True, text=True, check=False)
    if done.returncode:
        raise RuntimeError(f"spellbake failed (exit {done.returncode}):\n{done.stderr[-4000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


# ============================================================================
#  command line: write fixtures, or the browser normal-decode check's inputs
# ============================================================================

NORMCHECK_GATES = {"median": 0.15, "p99": 0.3}


#: Oracle bugs the browser check must flag (on the split plane, mikk basis).
NORMCHECK_MUTATIONS = ("flip_green", "ignore_w")


def write_normcheck(out_dir: Path, bakers: Iterable[str]) -> Path:
    """Bakes of the LOCAL plane (orthographic top view == its frame) for every layout and
    both bases, plus two mutated bakes the page must reject, and manifest.json for
    web/normcheck.html."""
    work = out_dir / "_inputs"
    entries = []

    def add(runner: Baker, case: Case, basis: str, name: str, expect: str, **entry: str) -> None:
        glb = runner.bake(case, Recipe(texture_size=512, tangent_basis=basis), work / "bakes" / name)
        shutil.copyfile(glb, out_dir / f"{name}.glb")
        entries.append({"file": f"{name}.glb", "basis": basis, "expect": expect, **entry})

    for baker_name in bakers:
        for layout in PLANE_LAYOUTS:
            for basis in bc.TANGENT_BASES:
                add(make_baker(baker_name), plane_case(work, layout, world=False), basis,
                    f"{baker_name}_{layout}_{basis}", "pass", baker=baker_name, layout=layout)
    for mutation in NORMCHECK_MUTATIONS:
        add(OracleBaker(mutations=[mutation]), plane_case(work, "split", world=False), "mikk",
            f"oracle_split_mikk_{mutation}", "fail", baker="oracle", layout="split")
    manifest = out_dir / "manifest.json"
    scene = plane_scene(world=False)
    manifest.write_text(json.dumps({"scene": {"amp": scene.amp, "bump": scene.bump}, "gates": NORMCHECK_GATES,
                                    "glbs": entries}, indent=1), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bakecheck.fixtures", description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--normcheck", action="store_true",
                        help="write plane bakes and manifest.json for web/normcheck.html")
    parser.add_argument("--baker", action="append", choices=BAKERS,
                        help="normcheck: which baker(s) to run (default: oracle)")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.normcheck:
        print(write_normcheck(args.out_dir, args.baker or ["oracle"]))
        return 0
    root = args.out_dir
    cases = [plane_case(root, layout) for layout in PLANE_LAYOUTS]
    cases += [sphere_case(root), coarse_sphere_case(root), exact_sphere_case(root),
              hoop_case(root), sheets_case(root), trench_case(root)]
    for case in cases:
        print(f"{case.name:<16} high={case.high} low={case.low} diag={case.diag:.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
