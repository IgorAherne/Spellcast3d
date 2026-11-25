"""The 3D model as Omni-Texture's worker reads it: its surface, its base colour texture as stored, smoothed normals,
and per texel of its UV islands the surface point and normal there."""

from __future__ import annotations

import io
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from instant_meshes_brush.assets import AssetError, load_source
from PIL import Image
from spellbake import BakeError, glb
from spellbake.raster import sample_bands

SMOOTHING = 8                # rounds of neighbour averaging of the normals the FACING rule reads: a generated
                             # model's surface is bumpy, and by its own normals the masks come out speckled
RASTER_PARTS = 8             # the UV raster runs on this many parts of the triangle list at once (threads)


@dataclass
class Model:
    """The 3D model's surface as spellbake.highpoly.load reads it (node transforms applied, v-up UVs with the
    material's texture transform), without the mip pyramids of its maps: the texture is read as stored."""
    positions: np.ndarray        # (nV, 3) float64
    normals: np.ndarray          # (nV, 3) float64
    uv: np.ndarray               # (nV, 2) float64, v up
    faces: np.ndarray            # (nF, 3) int64
    materials: int

    def area(self) -> float:
        """The surface's area."""
        corners = self.positions[self.faces]
        return float(np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                                    axis=1).sum() / 2.0)


def read_model(path: Path) -> Model:
    try:
        source = load_source(path)
    except AssetError as exc:
        raise BakeError(f"The high-poly mesh {path.name} could not be read: {exc}.") from exc
    return Model(source.vertices.astype(np.float64), source.normals.astype(np.float64), source.uv.astype(np.float64),
                 source.faces.astype(np.int64), len(source.materials))


def _gltf(data: bytes) -> tuple[dict[str, Any], bytes]:
    n = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + n])
    return gltf, data[20 + n + 8:]


def open_base_colour(data: bytes) -> tuple[Image.Image, str]:
    """The model's base colour texture opened (its size known), not decoded yet, and its format ("webp", "png",
    "jpeg")."""
    gltf, binary = _gltf(data)
    try:
        texture = gltf["textures"][gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
        source = glb.texture_source(texture)
        image = gltf["images"][source]
        view = gltf["bufferViews"][image["bufferView"]]
    except (KeyError, IndexError, TypeError):
        raise BakeError("This 3D model has no colour texture to paint. Make it again in the 3D stage.") from None
    blob = binary[view.get("byteOffset", 0): view.get("byteOffset", 0) + view["byteLength"]]
    picture = Image.open(io.BytesIO(blob))
    return picture, (picture.format or "png").lower()


def pixels(picture: Image.Image) -> np.ndarray:
    """The picture's bytes as stored: sRGB, RGB or RGBA."""
    return np.asarray(picture.convert("RGBA" if "A" in picture.getbands() else "RGB"))


def base_colour(data: bytes) -> tuple[np.ndarray, str]:
    """The model's base colour texture as stored (pixels) and its format."""
    picture, kind = open_base_colour(data)
    return pixels(picture), kind


def smooth_normals(positions: np.ndarray, faces: np.ndarray, normals: np.ndarray, rounds: int) -> np.ndarray:
    """The vertex normals averaged with their neighbours' `rounds` times (unit length)."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a, b = np.concatenate([edges[:, 0], edges[:, 1]]), np.concatenate([edges[:, 1], edges[:, 0]])
    # Per vertex its own normal, then its neighbours' in edge order, as np.add.at adds them. bincount adds in the
    # order given too, but from +0, which differs only where a component is -0: there add.at is used.
    index = np.concatenate([np.arange(len(normals)), a])
    n = normals.copy()
    for _ in range(rounds):
        if n.dtype != np.float64 or np.signbit(n[n == 0]).any():
            total = n.copy()
            np.add.at(total, a, n[b])
        else:
            terms = np.concatenate([n, n[b]])
            total = np.stack([np.bincount(index, terms[:, c], minlength=len(n)) for c in range(3)], axis=1)
        n = total / np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-12)
    return n


def _samples(uv: np.ndarray, faces: np.ndarray, res: int, lo: int, hi: int) -> tuple[np.ndarray, ...]:
    """The texel, triangle and barycentrics of every sample of triangles [lo, hi) (sample_bands: each band in
    triangle order, so a texel's first sample is that of the first triangle covering it)."""
    bands = list(sample_bands(uv, faces[lo:hi], res))
    return (np.concatenate([b.texel for b in bands]), np.concatenate([b.tri for b in bands]).astype(np.int64) + lo,
            np.concatenate([b.bary for b in bands]))


def _surface(pos: np.ndarray, nor: np.ndarray, faces: np.ndarray, tri: np.ndarray,
             bary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The surface point and unit normal (float32) of samples on triangles `tri` at barycentrics `bary`."""
    f = faces[tri]
    b0, b1, b2 = bary[:, 0:1], bary[:, 1:2], bary[:, 2:3]
    point = (b0 * pos[f[:, 0]] + b1 * pos[f[:, 1]]) + b2 * pos[f[:, 2]]
    n = (b0 * nor[f[:, 0]] + b1 * nor[f[:, 1]]) + b2 * nor[f[:, 2]]
    unit = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return point.astype(np.float32), unit.astype(np.float32)


def _first(texel: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """np.unique(texel, return_index=True) for indices below `size`: the texels in order and where each first is."""
    count = np.bincount(texel, minlength=size)
    where = np.empty(size, dtype=np.int64)
    where[texel] = np.arange(texel.size)
    again = np.flatnonzero(count[texel] > 1)                    # a texel two triangles cover: its first sample
    if again.size:
        twice, first = np.unique(texel[again], return_index=True)
        where[twice] = again[first]
    unique = np.flatnonzero(count)
    return unique.astype(texel.dtype), where[unique]


def texels(mesh: Model, res: int,
           pool: ThreadPoolExecutor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every texel of the UV islands of a `res` x `res` texture: its index (row from the top), its surface point
    and its smoothed normal there (float32 values, kept as float64 for the projection's arithmetic) -> those, and
    the smoothed vertex normals. A texel's first sample stands for it: that of the first triangle covering it.
    The raster runs on `pool`, in RASTER_PARTS of the triangle list at once, while the normals are smoothed here."""
    faces = mesh.faces
    cuts = np.unique(np.linspace(0, len(faces), RASTER_PARTS + 1).astype(np.int64))
    parts = [pool.submit(_samples, mesh.uv, faces, res, int(lo), int(hi)) for lo, hi in zip(cuts[:-1], cuts[1:])]
    smooth = smooth_normals(mesh.positions, faces, mesh.normals, SMOOTHING)
    parts = [part.result() for part in parts]
    surfaces = list(pool.map(lambda part: _surface(mesh.positions, smooth, faces, part[1], part[2]), parts))
    texel, first = _first(np.concatenate([part[0] for part in parts]), res * res)
    point = np.concatenate([point for point, _ in surfaces])[first].astype(np.float64)
    normal = np.concatenate([normal for _, normal in surfaces])[first].astype(np.float64)
    return texel, point, normal, smooth
