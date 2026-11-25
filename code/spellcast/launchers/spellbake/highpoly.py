"""The high poly a bake reads from, and what its surface looks like at a ray hit (design
section 4.4, steps L11 and L12).

``load()`` reads the model through instant_meshes_brush.assets.load_source, so a bake sees
exactly what the retopo viewer shows: node transforms applied, FBX in Y-up metres, the file's
own normals (area-weighted ones where it has none), v-up UVs with the material's texture
transform already applied, and every material's maps in the canonical layout at full
resolution (basecolor RGBA sRGB, normal OpenGL, ORM linear, emissive sRGB; see
instant_meshes_brush.materials). Each map becomes a trilinear mip pyramid.

``HighPoly.attributes()`` evaluates the surface at hits with glTF's semantics: what renders is
factor x map, a material without a map contributes its factor, the ORM's red channel is
ambient occlusion only where a texture supplied it, and a tangent-space normal map is decoded
through the high poly's own per-fragment MikkTSpace frame.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from instant_meshes_brush.assets import AssetError, load_source
from instant_meshes_brush.materials import ImageError, MaterialSpec, TexRef

from . import BakeError
from .encode import mikk_decode
from .sampling import CLAMP_TO_EDGE, MIRRORED_REPEAT, REPEAT, MipPyramid, mip_level, uv_jacobians
from .tangents import mikk_tangents, normalize

_WRAPS = (REPEAT, CLAMP_TO_EDGE, MIRRORED_REPEAT)
# The ORM channels a texture may supply.
_ORM = ("roughness", "metallic", "ao")


@dataclass(frozen=True)
class MapSampler:
    """One canonical map of one material: its mip pyramid and glTF wrap modes."""

    pyramid: MipPyramid
    wrap_s: int = REPEAT
    wrap_t: int = REPEAT

    def sample(self, uv: np.ndarray, footprint: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
        """(n, C) float32 linear values at v-up ``uv``, trilinear at the level where one bake
        sample (``footprint`` world units on a hit triangle with UV ``jacobian``) spans one texel."""
        lod = mip_level(footprint, jacobian, self.pyramid.width, self.pyramid.height)
        return self.pyramid.sample(uv[:, 0], uv[:, 1], lod, self.wrap_s, self.wrap_t)


@dataclass(frozen=True)
class HighMaterial:
    """A material's glTF factors and canonical maps (None where it has no such map)."""

    name: str
    base_color_factor: np.ndarray       # (4,) linear RGBA
    metallic: float
    roughness: float
    emissive_factor: np.ndarray         # (3,) linear, emissive strength included
    occlusion_strength: float
    normal_scale: float
    alpha_mode: str                     # OPAQUE | MASK | BLEND
    alpha_cutoff: float
    double_sided: bool
    basecolor: MapSampler | None = None     # RGBA: linear colour, alpha
    orm: MapSampler | None = None           # RGB: AO, roughness, metallic
    has_ao_map: bool = False                # the ORM's red channel came from a texture
    normal: MapSampler | None = None        # RGB in [0, 1], OpenGL
    normal_space: str | None = None         # "tangent" | "object" with a normal map
    emissive: MapSampler | None = None      # RGB linear

    @property
    def emits(self) -> bool:
        return self.emissive is not None or bool(np.any(self.emissive_factor > 0))

    @property
    def tangent_normals(self) -> bool:
        return self.normal is not None and self.normal_space == "tangent"


@dataclass(frozen=True)
class HitAttributes:
    """The high poly's surface at a batch of hits (L11, L12)."""

    position: np.ndarray        # (n, 3)
    smooth_normal: np.ndarray   # (n, 3) unit interpolated vertex normal N_h, facing the ray
    geometric_normal: np.ndarray  # (n, 3) unit face normal, facing the ray
    normal: np.ndarray          # (n, 3) unit detail normal n_w (normal map applied), facing the ray
    base: np.ndarray            # (n, 4) linear colour and alpha (1 on opaque materials)
    roughness: np.ndarray       # (n,)
    metallic: np.ndarray        # (n,)
    ao: np.ndarray              # (n,) occlusion from an AO map, NaN where the material has none
    emissive: np.ndarray        # (n, 3) linear


@dataclass
class HighPoly:
    name: str
    positions: np.ndarray       # (nV, 3) float64
    normals: np.ndarray         # (nV, 3) unit
    uv: np.ndarray              # (nV, 2) v up
    faces: np.ndarray           # (nF, 3) int64
    material: np.ndarray        # (nF,) int32 index into materials
    materials: list[HighMaterial]
    jacobians: np.ndarray       # (nF, 2, 2) d(u, v) / d(world) per triangle (sampling.uv_jacobians)
    tangents: np.ndarray | None  # (nF, 3, 4) float32 per-corner MikkTSpace, when a normal map needs it
    warnings: list[str]

    @property
    def double_sided(self) -> np.ndarray:
        """(nF,) bool: the triangle's material is double-sided."""
        flags = np.array([m.double_sided for m in self.materials], dtype=bool)
        return flags[self.material]

    @property
    def any_emissive(self) -> bool:
        return any(m.emits for m in self.materials)

    @property
    def any_alpha(self) -> bool:
        return any(m.alpha_mode != "OPAQUE" for m in self.materials)

    @property
    def any_ao_map(self) -> bool:
        return any(m.has_ao_map for m in self.materials)

    def attributes(self, prim: np.ndarray, u: np.ndarray, v: np.ndarray, back: np.ndarray,
                   footprint: np.ndarray) -> HitAttributes:
        """The surface at hits on triangles ``prim`` (Embree's u, v weight its corners 1 and 2).

        ``back`` marks hits on the back of a double-sided triangle: every normal is turned
        to face the ray. ``footprint`` is the world size of one bake sample (for the mip level).
        """
        count = len(prim)
        bary = np.stack([1.0 - u - v, u, v], axis=1)
        corners = self.faces[prim]
        flip = np.where(back, -1.0, 1.0)[:, None]
        position = np.einsum("nk,nkd->nd", bary, self.positions[corners])
        vertex_normal = np.einsum("nk,nkd->nd", bary, self.normals[corners])
        faces_normal = self._face_normals(prim)
        smooth = normalize(vertex_normal)
        lost = np.linalg.norm(smooth, axis=1) == 0      # opposite normals across a crease
        smooth[lost] = faces_normal[lost]
        uv = np.einsum("nk,nkd->nd", bary, self.uv[corners])
        jacobian = self.jacobians[prim]

        base = np.empty((count, 4))
        roughness = np.empty(count)
        metallic = np.empty(count)
        ao = np.full(count, np.nan)
        emissive = np.empty((count, 3))
        detail = smooth.copy()

        which = self.material[prim]
        order = np.argsort(which, kind="stable")
        bounds = np.searchsorted(which[order], np.arange(len(self.materials) + 1))
        for index, material in enumerate(self.materials):
            rows = order[bounds[index]:bounds[index + 1]]
            if rows.size == 0:
                continue
            at_uv, fp, jac = uv[rows], footprint[rows], jacobian[rows]
            factor = material.base_color_factor
            if material.basecolor is not None:
                texel = material.basecolor.sample(at_uv, fp, jac)
                base[rows, :3] = texel[:, :3] * factor[:3]
                base[rows, 3] = texel[:, 3] * factor[3]
            else:
                base[rows] = factor
            if material.alpha_mode == "OPAQUE":
                base[rows, 3] = 1.0
            if material.orm is not None:
                texel = material.orm.sample(at_uv, fp, jac)
                roughness[rows] = material.roughness * texel[:, 1]
                metallic[rows] = material.metallic * texel[:, 2]
                if material.has_ao_map:
                    ao[rows] = 1.0 + material.occlusion_strength * (texel[:, 0] - 1.0)
            else:
                roughness[rows] = material.roughness
                metallic[rows] = material.metallic
            if material.emissive is not None:
                emissive[rows] = material.emissive.sample(at_uv, fp, jac)[:, :3] * material.emissive_factor
            else:
                emissive[rows] = material.emissive_factor
            if material.normal is not None:
                mapped = 2.0 * material.normal.sample(at_uv, fp, jac)[:, :3].astype(np.float64) - 1.0
                if material.normal_space == "tangent":
                    assert self.tangents is not None, "load() computes them for tangent-space maps"
                    mapped[:, :2] *= material.normal_scale
                    frame = self.tangents[prim[rows]].astype(np.float64)
                    tangent = np.einsum("nk,nkd->nd", bary[rows], frame[:, :, :3])
                    world = mikk_decode(mapped, tangent, vertex_normal[rows], frame[:, 0, 3])
                else:  # object space: node transforms are already in the positions
                    world = normalize(mapped)
                lost = np.linalg.norm(world, axis=1) == 0
                world[lost] = smooth[rows][lost]
                detail[rows] = world
        return HitAttributes(position=position, smooth_normal=smooth * flip, geometric_normal=faces_normal * flip,
                             normal=detail * flip, base=base, roughness=roughness, metallic=metallic, ao=ao,
                             emissive=emissive)

    def _face_normals(self, prim: np.ndarray) -> np.ndarray:
        p = self.positions[self.faces[prim]]
        return normalize(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]))


def _wrap(value: int) -> int:
    return value if value in _WRAPS else REPEAT


def _sampler(image: np.ndarray | None, source: TexRef | None, *, srgb: int = 0) -> MapSampler | None:
    """The pyramid of a canonical map, repeating as the texture that supplied it does."""
    if image is None:
        return None
    if source is None:
        return MapSampler(MipPyramid(image, srgb_channels=srgb))
    return MapSampler(MipPyramid(image, srgb_channels=srgb), _wrap(source.wrap_s), _wrap(source.wrap_t))


def _material(spec: MaterialSpec, warn: Callable[[str], None]) -> HighMaterial:
    """A HighMaterial with the material's maps at full size."""
    factors = dict(
        name=spec.name,
        base_color_factor=np.asarray(spec.base_color_factor, dtype=np.float64),
        metallic=float(spec.metallic),
        roughness=float(spec.roughness),
        emissive_factor=np.asarray(spec.emissive, dtype=np.float64),
        occlusion_strength=float(spec.occlusion_strength),
        normal_scale=float(spec.normal_scale),
        alpha_mode=spec.alpha_mode,
        alpha_cutoff=float(spec.alpha_cutoff),
        double_sided=bool(spec.double_sided),
    )
    if not spec.refs:
        return HighMaterial(**factors)
    try:
        maps = spec.canonical(None)
    except ImageError as exc:
        warn(f"The maps of material '{spec.name}' could not be loaded ({exc}), so the bake uses its "
             f"plain colours there.")
        return HighMaterial(**factors)
    sources = spec.sources
    orm_source = next((sources[q] for q in _ORM if q in sources), None)
    try:
        return HighMaterial(
            **factors,
            basecolor=_sampler(maps.basecolor, sources.get("basecolor") or sources.get("opacity"), srgb=3),
            orm=_sampler(maps.orm, orm_source),
            has_ao_map="ao" in maps.orm_sources,
            normal=_sampler(maps.normal, sources.get("normal")),
            normal_space=maps.normal_space,
            emissive=_sampler(maps.emissive, sources.get("emissive"), srgb=3),
        )
    finally:
        spec.clear_cache()      # the pyramids hold what the bake needs


def load(path: str | Path, progress: Callable[[float, str], None] | None = None) -> HighPoly:
    """Read the high poly at ``path`` (any format instant_meshes_brush.assets reads).

    ``progress(fraction, message)`` hears about the steps. Raises BakeError when the file
    cannot be read."""
    report = progress or (lambda fraction, message: None)
    path = Path(path)
    report(0.0, "Reading the high-poly mesh…")
    try:
        source = load_source(path)
    except AssetError as exc:
        raise BakeError(f"The high-poly mesh {path.name} could not be read: {exc}.") from exc
    warnings = list(source.warnings)
    materials = []
    for index, spec in enumerate(source.materials):
        report(0.3 + 0.5 * index / len(source.materials), f"Loading the textures of {spec.name}…")
        materials.append(_material(spec, warnings.append))

    report(0.8, "Preparing the high-poly surface…")
    positions = source.vertices.astype(np.float64)
    normals = source.normals.astype(np.float64)
    uv = source.uv.astype(np.float64)
    faces = source.faces.astype(np.int64)
    tangents = None
    if any(m.tangent_normals for m in materials):
        # Per corner: the high poly's vertices need not be split by UV orientation.
        tangents = mikk_tangents(positions, normals, uv, faces).corners.astype(np.float32)
    return HighPoly(
        name=path.name,
        positions=positions,
        normals=normals,
        uv=uv,
        faces=faces,
        material=source.face_materials(),
        materials=materials,
        jacobians=uv_jacobians(positions, uv, faces),
        tangents=tangents,
        warnings=warnings,
    )
