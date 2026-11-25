"""The artist's own model -> the canonical GLB every stage reads (design sections 5.2 and 5.5).

``import_model(src, dst)`` reads an FBX, OBJ+MTL, glTF, Collada, PLY, STL or OFF model through
instant_meshes_brush.assets.load_source -- which classifies every texture (base colour, normal,
roughness or gloss, OpenGL or DirectX green...) -- and writes it back as one GLB in glTF's exact
terms: one primitive per material, node transforms applied (FBX in Y-up metres), and every
material's maps in the canonical layout at full resolution with glTF's factors. Retopo, the
viewers and the bake then all read the same, exact glTF slots. One liberty: glTF binds roughness
and metallic as one texture, so where a model has a map for only one of them, the other channel
holds its factor's value (and the factor becomes 1). It renders the same, and a viewer's Metallic
button then shows the metal the model has rather than a white map scaled by a factor of 0.

What glTF has no slot for is left out, with a warning: height maps, object-space normal maps and
maps whose role the classifier could not place (they stay in the report, so the artist can give
them a role and import again).
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from instant_meshes_brush.assets import AssetError, SourceMesh, load_source
from instant_meshes_brush.materials import ImageError, MaterialSpec, TexRef

from . import BakeError, glb
from .sampling import REPEAT, unorm8

JPEG_QUALITY = 95
PNG_COMPRESS_LEVEL = 6
GENERATOR = "Spellcast3D import"
# The fields of each detected map the hub shows (and takes corrections for, by key).
_GUESS_FIELDS = ("key", "file", "role", "label", "confidence", "y_convention", "y_confident", "evidence")


def _encode(image: np.ndarray, *, jpeg: bool) -> bytes:
    buffer = io.BytesIO()
    picture = Image.fromarray(np.ascontiguousarray(image))
    if jpeg:
        # No chroma subsampling: 4:2:0 costs up to 27 LSB at the colour edges of painted maps.
        picture.save(buffer, "JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=True)
    else:
        picture.save(buffer, "PNG", compress_level=PNG_COMPRESS_LEVEL)
    return buffer.getvalue()


def _texture(image: np.ndarray, name: str, source: TexRef | None, *, jpeg: bool = False) -> glb.Texture:
    wrap_s, wrap_t = (source.wrap_s, source.wrap_t) if source is not None else (REPEAT, REPEAT)
    return glb.Texture(_encode(image, jpeg=jpeg), name, wrap_s, wrap_t)


def _factors_into_orm(orm: np.ndarray, material: glb.Material, sources: frozenset[str]) -> np.ndarray:
    """The ORM with the roughness or metallic channel that no texture supplied holding its factor
    (which becomes 1): glTF binds both channels at once, and read back, a channel of 255 times a
    factor of 0 would show as a white, fully metallic map. What renders stays the same."""
    orm = orm.copy()
    if "roughness" not in sources:
        orm[..., 1] = unorm8(material.roughness_factor)
        material.roughness_factor = 1.0
    if "metallic" not in sources:
        orm[..., 2] = unorm8(material.metallic_factor)
        material.metallic_factor = 1.0
    return orm


def _material(spec: MaterialSpec, warn: Callable[[str], None]) -> glb.Material:
    material = glb.Material(
        name=spec.name,
        base_color_factor=tuple(float(v) for v in spec.base_color_factor),
        metallic_factor=float(spec.metallic),
        roughness_factor=float(spec.roughness),
        emissive_factor=tuple(float(v) for v in spec.emissive),
        occlusion_strength=float(spec.occlusion_strength),
        normal_scale=float(spec.normal_scale),
        alpha_mode=spec.alpha_mode,
        alpha_cutoff=float(spec.alpha_cutoff),
        double_sided=bool(spec.double_sided),
    )
    if not spec.refs:
        return material
    try:
        maps = spec.canonical(None)
    except ImageError as exc:
        warn(f"{spec.name}: its maps could not be read ({exc}), so it keeps only its plain colours.")
        return material
    try:
        sources = spec.sources
        if maps.basecolor is not None:
            opaque = int(maps.basecolor[..., 3].min()) == 255
            material.base_color = _texture(maps.basecolor[..., :3] if opaque else maps.basecolor,
                                           f"{spec.name} basecolor",
                                           sources.get("basecolor") or sources.get("opacity"), jpeg=opaque)
        if maps.orm is not None:
            orm_source = next((sources[q] for q in ("roughness", "metallic", "ao") if q in sources), None)
            image = maps.orm
            if maps.orm_sources & {"roughness", "metallic"}:
                image = _factors_into_orm(image, material, maps.orm_sources)
            orm = _texture(image, f"{spec.name} orm", orm_source)
            if maps.orm_sources & {"roughness", "metallic"}:
                material.metallic_roughness = orm
            if "ao" in maps.orm_sources:
                material.occlusion = orm
        if maps.normal is not None:
            if maps.normal_space == "object":
                warn(f"{spec.name}: its object-space normal map has no place in a glTF material, so it was "
                     f"left out.")
            else:
                material.normal = _texture(maps.normal, f"{spec.name} normal", sources.get("normal"))
        if maps.emissive is not None:
            material.emissive = _texture(maps.emissive, f"{spec.name} emissive", sources.get("emissive"))
            if not any(material.emissive_factor):
                material.emissive_factor = (1.0, 1.0, 1.0)
        if maps.height is not None:
            warn(f"{spec.name}: its height map has no place in a glTF material, so it was left out.")
        for name, _ in maps.others:
            warn(f"{spec.name}: {name} has no place in a glTF material, so it was left out. If it is a base "
                 f"colour, normal, roughness, metallic, AO or emissive map, give it that role and import again.")
    finally:
        spec.clear_cache()
    return material


def _report(source: SourceMesh) -> list[dict[str, Any]]:
    described = source.describe()["materials"]
    return [{"name": entry["name"],
             "maps": [{key: guess[key] for key in _GUESS_FIELDS} for guess in entry["guesses"]]}
            for entry in described]


def import_model(src: str | Path, dst: str | Path, overrides: Mapping[str, Mapping[str, Any]] | None = None, *,
                 progress: Callable[[float, str], None] | None = None) -> dict[str, Any]:
    """Convert ``src`` into the GLB ``dst``. ``overrides`` are the artist's corrections of the
    detected maps, {map key: {"role", "flip_green", "invert"}}.

    Returns {"glb", "seconds", "vertices", "faces", "materials": [{"name", "maps": [{"key",
    "file", "role", "label", "confidence", "y_convention", "y_confident", "evidence"}]}],
    "warnings"}. Raises BakeError when the model cannot be read."""
    started = time.perf_counter()
    report = progress or (lambda fraction, message: None)
    src, dst = Path(src), Path(dst)
    report(0.0, f"Reading {src.name}…")
    try:
        source = load_source(src, dict(overrides or {}))
    except AssetError as exc:
        raise BakeError(f"{src.name} could not be read: {exc}.") from exc
    maps_report = _report(source)
    warnings = list(source.warnings)

    primitives: list[glb.Primitive] = []
    materials: list[glb.Material] = []
    vertices = 0
    for index, (material, first, count) in enumerate(source.groups):
        report(0.2 + 0.7 * index / len(source.groups), f"Converting {source.materials[material].name}…")
        used, corners = np.unique(source.faces[first:first + count], return_inverse=True)
        primitives.append(glb.Primitive(positions=source.vertices[used], normals=source.normals[used],
                                        uv=source.uv[used], faces=corners.reshape(-1, 3), material=len(materials)))
        materials.append(_material(source.materials[material], warnings.append))
        vertices += len(used)
    report(0.9, f"Writing {dst.name}…")
    glb.write_glb(dst, primitives, materials, name=src.stem, generator=GENERATOR)
    report(1.0, "Imported.")
    return {"glb": str(dst), "seconds": round(time.perf_counter() - started, 2), "vertices": vertices,
            "faces": int(len(source.faces)), "materials": maps_report, "warnings": warnings}
