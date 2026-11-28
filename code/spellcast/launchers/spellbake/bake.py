"""The bake: a textured high poly onto the UVs of a low poly (design section 4.4, L1-L18).

``bake(high, low, out_dir, settings)`` writes the Projection stage's output (design section 6):

    out_dir/projection.glb            the low poly with its baked PBR material (always)
    out_dir/projection.fbx            the same as binary FBX 7.4 (export_fbx)
    out_dir/export/projection.obj     OBJ + MTL, maps referenced as ../maps/*.png (export_obj)
    out_dir/maps/*.png                every map on its own
    out_dir/bake.json                 {"version", "settings", "stats"}

How a texel gets its value:

1. The low poly is prepared (lowpoly.prepare: clean, winding vote, cage and export normals,
   optional repack, MikkTSpace) and rasterised in UV space, R x R texels of S x S samples, in
   bands of at most 256 rows and 4M samples, so memory stays bounded at 4096 px.
2. Each sample shoots the cage ray (rays.cage_hits): from P + d C back along -C, first
   front-facing hit. d is the ray distance, automatic unless the artist set one. A ray that
   would start inside or behind another part of the low poly starts past it instead
   (rays.other_parts), so a finger lying on the thigh does not print onto the thigh, and a hit
   that lies nearer another part of the low poly than to P is passed over
   (rays.claimed_by_other_parts): that part's high poly where it bulges past its low poly.
3. At the hit the high poly's surface is evaluated (highpoly.HighPoly.attributes): colour,
   alpha, roughness, metallic, AO map, emissive and the detail normal, all glTF semantics,
   maps sampled trilinearly in linear light at the level the sample's footprint asks for.
   Ambient occlusion is ray traced where the high poly has no AO map (and ao is on).
4. The detail normal is encoded in the low poly's tangent frame, for the basis the file
   declares (mikk: per-fragment MikkTSpace; per_vertex: three.js); the height (optional) is the
   hit's offset along the cage normal.
5. The samples of a texel are averaged (colours in linear light, normals as vectors), texels
   no sample reached are push-pull filled, and the maps are encoded to 8 bits (16 for height).

The samples of a band are shaded in chunks on a thread pool (numpy and Embree release the GIL),
while the rasteriser prepares the next band on its own thread. Chunks are summed in a fixed
order, so the same inputs always give the same files.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import closing
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from PIL import Image

from . import BakeError, ao, encode, fill, glb, highpoly, lowpoly, objexport, raster, rays
from .fbx import FbxMaterial, write_fbx
from .sampling import linear_to_srgb8, unorm8
from .settings import DEFAULT_AO_DISTANCE_PERCENT, AO_RAYS_STEP, BakeSettings
from .tangents import normalize

LOG = logging.getLogger("spellbake")

Progress = Callable[[float, str], None]

# Progress bands (design section 5.3): load the high poly, prepare the low poly, bake the
# texels, fill and encode, write the files.
_LOAD, _PREPARE, _BAKE, _FILL, _WRITE = (0.0, 0.15), (0.15, 0.22), (0.22, 0.90), (0.90, 0.97), (0.97, 1.0)
BAND_ROWS = 256
# Samples per band at most (4096 px at quality 2 fills a 256-row band with this many): at
# quality 3 bands get fewer rows, so the samples in flight, and the memory, stay the same.
BAND_SAMPLES = 1 << 22
# Samples per chunk of a band: enough to amortise numpy's per-call cost, few enough that every
# thread gets several chunks (and the per-chunk temporaries stay at tens of megabytes).
_CHUNK_MIN, _CHUNK_MAX = 16_384, 262_144
_CHUNKS_PER_THREAD = 3
PNG_COMPRESS_LEVEL = 6
# Above this share of missed samples the result says so (design section 9, risk 2).
MISS_WARNING = 0.005
# Cage rays start past other parts of the low poly (rays.other_parts). Off = the plain cage
# ray of design section 4.4, L10, bit for bit.
CAGE_STOPS_AT_PARTS = True
# Cage hits nearer another part of the low poly than to P are passed over (rays.claimed_by_other_parts):
# a finger's or a drawstring's side that bulges past its low poly no longer prints a streak beside it.
# Off = the rays as before, bit for bit.
CAGE_LEAVES_CLAIMED = True
VERSION = 1

_T = TypeVar("_T")


class BakeCancelled(Exception):
    """``cancelled()`` said stop; nothing usable was written."""


@dataclass(frozen=True)
class MapFile:
    id: str
    label: str
    file: str           # relative to the bake folder, with forward slashes
    colorspace: str     # "srgb" | "linear"


@dataclass
class BakeResult:
    """What a bake wrote. Paths are relative to ``folder``."""

    folder: Path
    glb: str
    fbx: str | None
    obj: str | None
    maps: list[MapFile]
    stats: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        """The worker's result (launchers/imesh_server.py reads it)."""
        return {"glb": self.glb, "fbx": self.fbx, "obj": self.obj,
                "maps": [{"id": m.id, "label": m.label, "file": m.file, "colorspace": m.colorspace}
                         for m in self.maps],
                "stats": self.stats}


# ---------------------------------------------------------------------- progress


class _Reporter:
    """Maps each stage's own 0..1 onto its band of the whole bake, and stops on cancel."""

    def __init__(self, progress: Progress | None, cancelled: Callable[[], bool] | None) -> None:
        self._progress = progress or (lambda fraction, message: None)
        self._cancelled = cancelled or (lambda: False)

    def check(self) -> None:
        if self._cancelled():
            raise BakeCancelled

    def stage(self, band: tuple[float, float]) -> Progress:
        low, high = band

        def report(fraction: float, message: str) -> None:
            self.check()
            self._progress(low + (high - low) * min(1.0, max(0.0, fraction)), message)

        return report


# ---------------------------------------------------------------------- per-sample work


class _Layout:
    """Where each quantity sits in a sample's value row (float32)."""

    def __init__(self, emissive: bool, height: bool) -> None:
        self.base = slice(0, 4)             # linear RGB, alpha
        self.orm = slice(4, 7)              # AO, roughness, metallic
        self.normal = slice(7, 10)          # tangent-space unit vector
        width = 10
        self.emissive = slice(width, width + 3) if emissive else None
        width += 3 if emissive else 0
        self.height = width if height else None
        self.width = width + (1 if height else 0)


@dataclass
class _Chunk:
    texel: np.ndarray       # (k,) texel of every sample that found the high poly
    values: np.ndarray      # (k, layout.width) float32
    samples: int            # samples inside the UV islands (ring samples excluded)
    misses: int             # of those, how many found nothing
    fallbacks: int          # of those, how many took the nearest-hit fallback


@dataclass
class _Job:
    """Everything a chunk needs; read-only while the pool runs."""

    settings: BakeSettings
    mesh: lowpoly.RenderMesh
    high: highpoly.HighPoly
    scene: rays.Scene
    parts: rays.Scene | None    # the low poly's own triangles, for rays.other_parts
    claims: rays.Scene | None   # the same, for rays.claimed_by_other_parts
    distance: float
    ao_distance: float
    ao_output: bool
    layout: _Layout
    per_vertex: tuple[np.ndarray, np.ndarray, np.ndarray] | None

    @property
    def resolution(self) -> int:
        return self.settings.texture_size

    def shade(self, tri: np.ndarray, bary: np.ndarray, texel: np.ndarray, sub: np.ndarray,
              ring: np.ndarray) -> _Chunk:
        settings, layout = self.settings, self.layout
        samples_per_side = self.resolution * settings.supersample
        surface = self.mesh.surface(tri, bary, samples_per_side)
        hits = rays.cage_hits(self.scene, surface.position, surface.cage, self.distance, low=self.parts,
                              claims=self.claims)
        inside = ~ring
        found = np.flatnonzero(hits.hit)
        hit = self.high.attributes(hits.prim[found], hits.u[found], hits.v[found], hits.back[found],
                                   surface.footprint[found])

        values = np.empty((found.size, layout.width), dtype=np.float32)
        values[:, layout.base] = hit.base
        values[:, layout.orm.start + 1] = hit.roughness
        values[:, layout.orm.start + 2] = hit.metallic
        occlusion = hit.ao
        if self.ao_output:
            traced = np.flatnonzero(np.isnan(occlusion))
            if settings.ao and traced.size:
                occlusion[traced] = self._trace_ao(hit, traced, texel[found[traced]], sub[found[traced]])
        values[:, layout.orm.start] = np.nan_to_num(occlusion, nan=1.0)
        values[:, layout.normal] = encode.clamp_hemisphere(self._encode_normals(hit.normal, surface, tri, bary, found))
        if layout.emissive is not None:
            values[:, layout.emissive] = hit.emissive
        if layout.height is not None:
            values[:, layout.height] = hits.height[found]
        return _Chunk(texel=texel[found], values=values, samples=int(inside.sum()),
                      misses=int((inside & ~hits.hit).sum()), fallbacks=int((inside & hits.fallback).sum()))

    def _encode_normals(self, world: np.ndarray, surface: lowpoly.Surface, tri: np.ndarray, bary: np.ndarray,
                        found: np.ndarray) -> np.ndarray:
        if self.per_vertex is None:
            return encode.mikk_encode(world, surface.tangent[found], surface.normal[found], surface.sign[found])
        corners = self.mesh.faces[tri[found]]
        t, b, n = (np.einsum("nk,nkd->nd", bary[found], basis[corners]) for basis in self.per_vertex)
        return encode.per_vertex_encode(world, t, b, n, surface.sign[found])

    def _trace_ao(self, hit: highpoly.HitAttributes, rows: np.ndarray, texel: np.ndarray,
                  sub: np.ndarray) -> np.ndarray:
        """Unoccluded fraction, from ao_rays rays per texel: sets of 32 stratified directions,
        each set rotated differently per texel (ao.ambient_occlusion)."""
        sets = self.settings.ao_rays // AO_RAYS_STEP
        texel = texel.astype(np.int64)
        total = np.zeros(rows.size)
        for index in range(sets):
            total += ao.ambient_occlusion(
                self.scene, hit.position[rows], hit.smooth_normal[rows], hit.geometric_normal[rows],
                texel + index * self.resolution ** 2, sub, supersample=self.settings.supersample,
                max_distance=self.ao_distance)
        return total / sets


# ---------------------------------------------------------------------- the bake loop


def _prefetch(items: Iterator[_T]) -> Iterator[_T]:
    """``items``, produced one ahead on a thread of their own. Closing the iterator stops and
    joins that thread."""
    handoff: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
    stop = threading.Event()

    def put(entry: tuple[str, Any]) -> bool:
        while not stop.is_set():
            try:
                handoff.put(entry, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for item in items:
                if not put(("item", item)):
                    return
            put(("end", None))
        except BaseException as exc:  # handed to the consumer, which raises it
            put(("error", exc))

    thread = threading.Thread(target=produce, name="spellbake-raster", daemon=True)
    thread.start()
    try:
        while True:
            kind, value = handoff.get()
            if kind == "end":
                return
            if kind == "error":
                raise value
            yield value
    finally:
        stop.set()
        thread.join()


@dataclass
class _Maps:
    """The per-texel means of the whole map, (R * R, C) float32, and which texels have any."""

    values: np.ndarray
    weight: np.ndarray      # (R * R,) samples that found the high poly
    covered: np.ndarray     # (R * R,) bool: texels inside a UV island
    samples: int = 0
    misses: int = 0
    fallbacks: int = 0


def _chunk_slices(count: int, threads: int) -> list[slice]:
    size = min(_CHUNK_MAX, max(_CHUNK_MIN, math.ceil(count / (_CHUNKS_PER_THREAD * threads))))
    return [slice(start, min(start + size, count)) for start in range(0, count, size)]


def _accumulate(maps: _Maps, band: raster.SampleBand, chunks: list[_Chunk], resolution: int) -> None:
    """Average the band's samples per texel into ``maps``."""
    if not chunks:      # no UV island reaches these rows
        return
    first = band.row0 * resolution
    length = band.rows * resolution
    maps.covered[band.texel[~band.ring]] = True
    texel = np.concatenate([c.texel for c in chunks]).astype(np.int64) - first
    values = np.concatenate([c.values for c in chunks])
    count = np.bincount(texel, minlength=length)
    has = count > 0
    target = maps.values[first:first + length]
    for channel in range(values.shape[1]):
        sums = np.bincount(texel, weights=values[:, channel], minlength=length)
        target[has, channel] = sums[has] / count[has]
    maps.weight[first:first + length] = count
    for chunk in chunks:
        maps.samples += chunk.samples
        maps.misses += chunk.misses
        maps.fallbacks += chunk.fallbacks


def _bake_texels(job: _Job, ring: float, threads: int, report: Progress) -> _Maps:
    resolution = job.resolution
    maps = _Maps(values=np.zeros((resolution * resolution, job.layout.width), dtype=np.float32),
                 weight=np.zeros(resolution * resolution, dtype=np.float32),
                 covered=np.zeros(resolution * resolution, dtype=bool))
    supersample = job.settings.supersample
    band_rows = max(16, min(BAND_ROWS, BAND_SAMPLES // (resolution * supersample * supersample)))
    bands = _prefetch(raster.sample_bands(job.mesh.uv, job.mesh.faces, resolution, supersample, ring=ring,
                                          band_rows=band_rows))
    pending: deque[tuple[raster.SampleBand, list[Future[_Chunk]]]] = deque()
    rows_done = 0

    def finish_oldest() -> None:
        nonlocal rows_done
        band, futures = pending[0]
        chunks = []
        for index, future in enumerate(futures):
            chunks.append(future.result())
            done = (rows_done + band.rows * (index + 1) / len(futures)) / resolution
            report(done, f"Baking the textures… {100 * done:.0f}%")
        _accumulate(maps, band, chunks, resolution)
        rows_done += band.rows
        pending.popleft()

    # On the way out (cancel, error) the rasteriser thread stops first, then the pool waits only
    # for the chunks already running: the queued ones are cancelled.
    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="spellbake") as pool, closing(bands):
        try:
            for band in bands:
                pending.append((band, [pool.submit(job.shade, band.tri[part], band.bary[part], band.texel[part],
                                                   band.sub[part], band.ring[part])
                                       for part in _chunk_slices(len(band.tri), threads)]))
                # Keep the next band's chunks queued while this one is summed.
                while len(pending) > 1:
                    finish_oldest()
            while pending:
                finish_oldest()
        except BaseException:
            for _, futures in pending:
                for future in futures:
                    future.cancel()
            raise
    return maps


# ---------------------------------------------------------------------- fill, encode, write


def _png(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(buffer, "PNG", compress_level=PNG_COMPRESS_LEVEL)
    return buffer.getvalue()


@dataclass
class _Encoded:
    """The 8-bit (and 16-bit height) maps, and what the material needs to know about them."""

    images: dict[str, np.ndarray]
    emissive_strength: float = 1.0      # linear scale of the emissive map
    height_range: tuple[float, float] | None = None


def _strips(image: np.ndarray, convert: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """``convert(image)`` a band of rows at a time: its float64 temporaries stay small at 4096 px."""
    return np.concatenate([convert(image[row:row + BAND_ROWS]) for row in range(0, len(image), BAND_ROWS)])


def _fill_and_encode(maps: _Maps, job: _Job, alpha: bool, report: Progress) -> _Encoded:
    """L17 + L18: every island grown by fill.DILATE_TEXELS texels (the seams of every map read the island's own
    colours), then the push-pull fill in linear light, then 8-bit encoding."""
    resolution = job.resolution
    layout = job.layout
    # The texels inside a UV island whose samples found the high poly: the dilation grows from these, over the ring
    # of extrapolated samples outside the islands and the texels no sample reached.
    inside = maps.covered.reshape(resolution, resolution) & (maps.weight > 0).reshape(resolution, resolution)
    grid = maps.values.reshape(resolution, resolution, layout.width)
    images: dict[str, np.ndarray] = {}
    message = "Filling the gaps between UV islands…"

    def filled(channels: slice) -> np.ndarray:
        grown, weight = fill.dilate(grid[..., channels], inside)
        return fill.push_pull(grown, weight)

    report(0.0, message)
    base = filled(layout.base)
    colour = _strips(base[..., :3], linear_to_srgb8)
    images["basecolor"] = np.concatenate([colour, _strips(base[..., 3:], unorm8)], axis=-1) if alpha else colour
    if alpha:
        images["opacity"] = np.ascontiguousarray(images["basecolor"][..., 3])
    del base, colour

    report(0.3, message)
    orm = _strips(filled(layout.orm), unorm8)
    if not job.ao_output:
        orm[..., 0] = 255
    images["orm"] = orm
    images["roughness"] = np.ascontiguousarray(orm[..., 1])
    images["metallic"] = np.ascontiguousarray(orm[..., 2])
    if job.ao_output:
        images["ao"] = np.ascontiguousarray(orm[..., 0])

    report(0.5, message)
    grown_n, weight_n = fill.dilate(_strips(grid[..., layout.normal], normalize), inside)
    normals = fill.push_pull_unit(grown_n, weight_n)
    del grown_n
    images["normal"] = _strips(normals, encode.quantize)
    if job.settings.normal_y == "directx":
        directx = images["normal"].copy()
        directx[..., 1] = 255 - directx[..., 1]
        images["normal_directx"] = directx
    del normals

    out = _Encoded(images)
    if layout.emissive is not None:
        report(0.7, message)
        emissive = filled(layout.emissive)
        # Brighter than 1 (emissive strength): the map holds colour / peak, the factor the peak.
        out.emissive_strength = max(1.0, float(emissive.max()))
        emissive /= out.emissive_strength
        images["emissive"] = _strips(emissive, linear_to_srgb8)
    if layout.height is not None:
        report(0.85, message)
        height = filled(layout.height)
        known = height[inside] if inside.any() else height.reshape(-1)
        low, high = float(known.min()), float(known.max())
        scaled = (height - low) / (high - low) if high > low else np.zeros_like(height)
        images["height"] = np.round(np.clip(scaled, 0.0, 1.0) * 65535.0).astype(np.uint16)
        out.height_range = (low, high)
    return out


_LABELS = {"basecolor": "Base colour", "normal": "Normal", "normal_directx": "Normal (DirectX)",
           "roughness": "Roughness", "metallic": "Metallic", "ao": "AO", "orm": "ORM", "emissive": "Emissive",
           "opacity": "Opacity", "height": "Height"}
_SRGB = frozenset({"basecolor", "emissive"})


def _alpha_mode(high: highpoly.HighPoly) -> tuple[str, float]:
    """The baked material's alpha: the high poly's most permissive mode."""
    modes = [m.alpha_mode for m in high.materials]
    if "BLEND" in modes:
        return "BLEND", 0.5
    masks = [m.alpha_cutoff for m in high.materials if m.alpha_mode == "MASK"]
    return ("MASK", min(masks)) if masks else ("OPAQUE", 0.5)


def _write(out_dir: Path, job: _Job, encoded: _Encoded, stats: dict[str, Any], threads: int, started: float,
           report: Progress) -> BakeResult:
    settings, mesh, high = job.settings, job.mesh, job.high
    maps_dir = out_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    report(0.0, "Writing the maps…")
    with ThreadPoolExecutor(max_workers=max(1, min(threads, len(encoded.images)))) as pool:
        png = dict(zip(encoded.images, pool.map(_png, encoded.images.values())))
    for key, data in png.items():
        (maps_dir / f"{key}.png").write_bytes(data)
    map_files = [MapFile(key, _LABELS[key], f"maps/{key}.png", "srgb" if key in _SRGB else "linear")
                 for key in _LABELS if key in png]

    report(0.4, "Writing the GLB…")
    alpha_mode, alpha_cutoff = _alpha_mode(high)
    extras: dict[str, Any] = {"tangent_basis": settings.tangent_basis, "normal_y": "+Y",
                              "ray_distance": job.distance, "version": VERSION}
    if encoded.height_range is not None:
        extras["height_range"] = list(encoded.height_range)
    orm = glb.Texture(png["orm"], "orm")
    emits = "emissive" in png
    material = glb.Material(
        name="projection",
        base_color=glb.Texture(png["basecolor"], "basecolor"),
        metallic_roughness=orm,
        occlusion=orm if job.ao_output else None,
        normal=glb.Texture(png["normal"], "normal"),
        emissive=glb.Texture(png["emissive"], "emissive") if emits else None,
        emissive_factor=(encoded.emissive_strength,) * 3 if emits else (0.0, 0.0, 0.0),
        alpha_mode=alpha_mode,
        alpha_cutoff=alpha_cutoff,
        double_sided=any(m.double_sided for m in high.materials),
        extras={"spellcast": extras},
    )
    primitive = glb.Primitive(positions=mesh.positions, normals=mesh.normals, faces=mesh.faces, uv=mesh.uv,
                              tangents=mesh.tangents)
    glb.write_glb(out_dir / "projection.glb", [primitive], [material], name="projection")

    normal_key = "normal_directx" if settings.normal_y == "directx" else "normal"
    fbx_file = None
    if settings.export_fbx:
        report(0.6, "Writing the FBX…")
        fbx_material: FbxMaterial = {
            "name": "projection", "base_color_png": png["basecolor"], "normal_png": png[normal_key],
            "roughness_png": png["roughness"], "metallic_png": png["metallic"],
            "emissive_png": png.get("emissive"), "emissive_factor": material.emissive_factor,
            "alpha_mode": alpha_mode, "double_sided": material.double_sided,
        }
        fbx_file = "projection.fbx"
        write_fbx(out_dir / fbx_file, name="projection", positions=mesh.positions, normals=mesh.normals,
                  tangents=mesh.tangents, uvs=mesh.uv, indices=mesh.faces, material=fbx_material,
                  map_files={"normal": f"maps/{normal_key}.png"})
    obj_file = None
    if settings.export_obj:
        report(0.8, "Writing the OBJ…")
        (out_dir / "export").mkdir(exist_ok=True)
        obj_file = "export/projection.obj"
        objexport.write_obj(
            out_dir / obj_file, positions=mesh.positions, position_id=mesh.position_id, uv=mesh.uv,
            normals=mesh.normals, faces=mesh.faces,
            maps=objexport.MtlMaps(basecolor="../maps/basecolor.png", normal=f"../maps/{normal_key}.png",
                                   roughness="../maps/roughness.png", metallic="../maps/metallic.png",
                                   emissive="../maps/emissive.png" if emits else None,
                                   opacity="../maps/opacity.png" if "opacity" in png else None,
                                   emissive_strength=encoded.emissive_strength))
    if encoded.height_range is not None:
        stats["height_range"] = list(encoded.height_range)
    stats["seconds"] = round(time.perf_counter() - started, 2)
    (out_dir / "bake.json").write_text(
        json.dumps({"version": VERSION, "settings": settings.to_json(), "stats": stats}, indent=1),
        encoding="utf-8")
    report(1.0, "Baked.")
    return BakeResult(out_dir, "projection.glb", fbx_file, obj_file, map_files, stats)


# ---------------------------------------------------------------------- entry point


def bake(high: str | Path, low: str | Path, out_dir: str | Path, settings: BakeSettings, *,
         progress: Progress | None = None, cancelled: Callable[[], bool] | None = None,
         threads: int | None = None) -> BakeResult:
    """Bake ``high`` (any model instant_meshes_brush reads) onto the UVs of ``low`` (the retopo
    OBJ, or a GLB) into ``out_dir`` (created if needed).

    ``progress(fraction, message)`` hears how far it is; ``cancelled()`` is asked often and
    stops the bake with BakeCancelled. Raises BakeError, with a message for the artist, when
    the meshes or settings cannot be baked."""
    started = time.perf_counter()
    reporter = _Reporter(progress, cancelled)
    threads = max(1, int(threads or os.cpu_count() or 4))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolution = settings.texture_size

    high_poly = highpoly.load(high, reporter.stage(_LOAD))
    reporter.stage(_LOAD)(0.9, "Building the ray-tracing structure…")
    # Rays run on the chunk threads, so the scene needs no pool of its own.
    with rays.Scene(high_poly.positions, high_poly.faces, double_sided=high_poly.double_sided) as scene:
        LOG.info("high poly %s: %d triangles, %d materials (%.1f s)", high_poly.name, len(high_poly.faces),
                 len(high_poly.materials), time.perf_counter() - started)
        prepare = reporter.stage(_PREPARE)
        prepare(0.0, "Preparing the low poly…")
        low_poly = lowpoly.prepare(low, scene, resolution=resolution, repack_uvs=settings.repack_uvs,
                                   hard_edge_angle=settings.hard_edge_angle)
        mesh = low_poly.mesh
        prepare(0.7, "Measuring how far the high poly is…")
        if settings.ray_distance is not None:
            distance = settings.ray_distance / 100.0 * scene.diag
        else:
            distance = rays.auto_distance(scene, *mesh.probes())
        ao_percent = settings.ao_distance if settings.ao_distance is not None else DEFAULT_AO_DISTANCE_PERCENT
        # A few thousand triangles, built in milliseconds; the extra ray per sample costs 2-5 % of a
        # bake (0.2-0.6 s of 11-13 s at 2048 px on 8 threads, five Pixal3D props). The claim test (two
        # more rays) runs only for hits more than 0.1 d above P: 0.7-4 % of the samples, 18 % on the
        # Dragon, whose d is small; under 1 % of a bake.
        own = rays.Scene(mesh.positions, mesh.faces) if CAGE_STOPS_AT_PARTS or CAGE_LEAVES_CLAIMED else None
        job = _Job(
            settings=settings, mesh=mesh, high=high_poly, scene=scene,
            parts=own if CAGE_STOPS_AT_PARTS else None, claims=own if CAGE_LEAVES_CLAIMED else None,
            distance=distance,
            ao_distance=ao_percent / 100.0 * scene.diag,
            ao_output=settings.ao or high_poly.any_ao_map,
            layout=_Layout(emissive=high_poly.any_emissive, height=settings.height),
            per_vertex=encode.per_vertex_basis(mesh.normals, mesh.tangents)
            if settings.tangent_basis == encode.PER_VERTEX else None,
        )
        ring = settings.ring if settings.ring is not None else low_poly.ring
        LOG.info("low poly: %s; ray distance %.3f%% of the size; ring %g px (%.1f s)", low_poly.stats,
                 100 * distance / scene.diag, ring, time.perf_counter() - started)
        maps = _bake_texels(job, ring, threads, reporter.stage(_BAKE))
        LOG.info("texels baked: %d samples, %d missed, %d by the fallback (%.1f s)", maps.samples, maps.misses,
                 maps.fallbacks, time.perf_counter() - started)

    if maps.samples == 0:
        raise BakeError("The low poly's UVs cover no texel of the map. Run Retopo again to unwrap it.")
    if maps.misses == maps.samples:
        raise BakeError("No part of the low poly found the high poly within the ray distance. Check that both "
                        "meshes are in the same place and at the same size, or raise Ray distance.")
    encoded = _fill_and_encode(maps, job, high_poly.any_alpha, reporter.stage(_FILL))
    miss_fraction = maps.misses / maps.samples
    warnings = [*high_poly.warnings, *low_poly.warnings]
    if miss_fraction > MISS_WARNING:
        warnings.append(f"{100 * miss_fraction:.1f}% of the low poly found no high-poly surface within the ray "
                        f"distance ({100 * distance / scene.diag:.2f}% of the model's size); those spots were "
                        f"filled from their surroundings. Raise Ray distance and bake again if they show.")
    stats: dict[str, Any] = {
        "resolution": resolution,
        "supersample": settings.supersample,
        "ray_distance": distance,
        "ray_distance_percent": 100.0 * distance / scene.diag,
        "coverage": float(maps.covered.mean()),
        "miss_fraction": miss_fraction,
        "fallback_fraction": maps.fallbacks / maps.samples,
        "high_triangles": int(len(high_poly.faces)),
        **{key: low_poly.stats[key] for key in ("triangles", "polygons", "vertices", "render_vertices",
                                                "flipped_triangles",
                                                "dropped_triangles", "untextured_triangles",
                                                "uv_degenerate_triangles", "min_chart_gap_px")},
        "repacked": low_poly.repacked,
        "ambient_occlusion": job.ao_output,
        "warnings": warnings,
    }
    del maps
    result = _write(out_dir, job, encoded, stats, threads, started, reporter.stage(_WRITE))
    LOG.info("bake written to %s (%.1f s)", out_dir, time.perf_counter() - started)
    return result
