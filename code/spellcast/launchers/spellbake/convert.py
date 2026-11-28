"""Saving a 3D model (a GLB: the 3D stage's high poly, or a bake) as FBX, and as OBJ + MTL + PNG maps,
for the artist's export folder. launchers/export_worker.py runs convert() in the imesh venv.

The model is read the way the bake reads its high poly (instant_meshes_brush.assets.load_source): node
transforms applied, Y-up metres, the file's own normals, v-up UVs, and every material's maps in the
canonical layout (basecolor RGBA sRGB, normal OpenGL, ORM linear, emissive sRGB), WebP textures
included. Each glTF map goes where the writers expect it:

    glTF                          FBX (fbx.write_fbx_materials)        OBJ (MTL)
    baseColor, sRGB               DiffuseColor, embedded PNG           map_Kd  <name>_basecolor.png
    its alpha (MASK, BLEND)       TransparentColor                     map_d   <name>_opacity.png
    normal, tangent space         NormalMap + MikkTSpace tangents      map_Bump, norm  <name>_normal.png
    metallicRoughness G           ShininessExponent (roughness)        map_Pr  <name>_roughness.png
    metallicRoughness B           ReflectionFactor (metallic)          map_Pm  <name>_metallic.png
    occlusion R                   - (FBX has no AO slot)               <name>_ao.png beside it (MTL has no keyword)
    emissive, sRGB                EmissiveColor + EmissiveFactor       map_Ke + Ke

A glTF factor multiplies its map; FBX and MTL readers take a map instead of the value, so a factor
that is not 1 is folded into the map (in linear light for the colours, the normal scale into the
normals) and the value written beside the map is 1. A quantity without a map is written as its
value. An object-space normal map has no slot in either format and is left out (a warning says so).
With several materials the map files are <name>_<material>_<map>.png.

Everything runs on numpy arrays (no Python loop over vertices or faces) and the maps are encoded on
a thread pool: a Pixal3D model (300k triangles, two 2048 px maps) takes a few seconds.
"""

from __future__ import annotations

import io
import re
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from instant_meshes_brush.assets import AssetError, load_source
from instant_meshes_brush.materials import ImageError, MaterialSpec

from . import BakeError, glb
from .fbx import FbxMaterial, write_fbx_materials
from .objexport import MtlMaps, MtlMaterial, write_obj_materials
from .sampling import linear_to_srgb8, srgb_to_linear, unorm8
from .tangents import mikk_tangents, normalize

Progress = Callable[[float, str], None]

FORMATS = ("fbx", "obj")
PLACED_FORMATS = ("glb",)       # ... and, for a model with a placement, its own format with the placement in it
PNG_COMPRESS_LEVEL = 6
BAND_ROWS = 256                 # map rows converted at a time: float temporaries stay small at 4096 px
_MTL_UNSAFE = re.compile(r"[^\w\-.]+")


@dataclass
class _Material:
    """One material, ready for both writers: its encoded maps with the factors folded in."""

    name: str                                   # unique, one word (an MTL name)
    pngs: dict[str, bytes] = field(default_factory=dict)    # map -> PNG: basecolor, normal, roughness...
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)   # linear, where no map replaces it
    metallic: float = 1.0
    roughness: float = 1.0
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emissive_strength: float = 1.0              # the emissive map's multiplier (glTF's emissive strength)
    alpha_mode: str = "OPAQUE"
    double_sided: bool = False
    tangent_normals: bool = False


def _png(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(buffer, "PNG", compress_level=PNG_COMPRESS_LEVEL)
    return buffer.getvalue()


def _bands(image: np.ndarray, convert: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """``convert`` a band of rows at a time."""
    return np.concatenate([convert(image[row:row + BAND_ROWS]) for row in range(0, len(image), BAND_ROWS)])


def _scaled_colour(image: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """sRGB codes times a linear factor per channel, back to sRGB codes (the image itself when it is 1)."""
    if np.all(factor == 1.0):
        return image
    return _bands(image, lambda rows: linear_to_srgb8(srgb_to_linear(rows) * factor))


def _scaled_data(image: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return image
    return _bands(image, lambda rows: unorm8(rows / 255.0 * factor))


def _scaled_normals(image: np.ndarray, scale: float) -> np.ndarray:
    """glTF's normalTexture.scale folded into the map: x and y scaled, then made unit length again."""
    if scale == 1.0:
        return image

    def convert(rows: np.ndarray) -> np.ndarray:
        vectors = rows.astype(np.float64) / 127.5 - 1.0
        vectors[..., :2] *= scale
        return np.round((normalize(vectors) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return _bands(image, convert)


def _unique_names(specs: list[MaterialSpec]) -> list[str]:
    names: list[str] = []
    for index, spec in enumerate(specs):
        base = _MTL_UNSAFE.sub("_", spec.name).strip("_.") or f"material_{index + 1}"
        name, number = base, 2
        while name.casefold() in (n.casefold() for n in names):
            name, number = f"{base}_{number}", number + 1
        names.append(name)
    return names


def _material(spec: MaterialSpec, name: str, pool: ThreadPoolExecutor, warn: Callable[[str], None]) -> _Material:
    """The material's maps in the writers' layout, factors folded in, PNGs encoded on `pool`."""
    base = np.asarray(spec.base_color_factor, dtype=np.float64)
    emissive = np.asarray(spec.emissive, dtype=np.float64)
    out = _Material(name=name, base_color=tuple(base.tolist()), metallic=float(spec.metallic),
                    roughness=float(spec.roughness), emissive=tuple(emissive.tolist()),
                    alpha_mode=spec.alpha_mode, double_sided=bool(spec.double_sided))
    if not spec.refs:
        return out
    try:
        maps = spec.canonical(None)
    except ImageError as exc:
        warn(f"The maps of material '{spec.name}' could not be read ({exc}), so it is saved with its plain colours.")
        return out
    images: dict[str, np.ndarray] = {}
    blend = spec.alpha_mode != "OPAQUE"
    if maps.basecolor is not None:
        colour = _scaled_colour(maps.basecolor[..., :3], base[:3])
        if blend:
            alpha = _scaled_data(maps.basecolor[..., 3], float(base[3]))
            images["basecolor"] = np.concatenate([colour, alpha[..., None]], axis=-1)
            images["opacity"] = np.ascontiguousarray(alpha)
        else:
            images["basecolor"] = colour
        out.base_color = (1.0, 1.0, 1.0, 1.0)
    if maps.orm is not None:
        if "roughness" in maps.orm_sources:
            images["roughness"] = _scaled_data(maps.orm[..., 1], out.roughness)
            out.roughness = 1.0
        if "metallic" in maps.orm_sources:
            images["metallic"] = _scaled_data(maps.orm[..., 2], out.metallic)
            out.metallic = 1.0
        if "ao" in maps.orm_sources:
            strength = float(spec.occlusion_strength)
            images["ao"] = _bands(maps.orm[..., 0], lambda rows: unorm8(1.0 + strength * (rows / 255.0 - 1.0)))
    if maps.normal is not None:
        if maps.normal_space == "tangent":
            images["normal"] = _scaled_normals(maps.normal, float(spec.normal_scale))
            out.tangent_normals = True
        else:
            warn(f"The object-space normal map of material '{spec.name}' is left out: FBX and OBJ only take "
                 f"tangent-space normal maps.")
    if maps.emissive is not None:
        # A glow brighter than 1 (emissive strength): the map holds colour / peak, the multiplier the peak.
        out.emissive_strength = max(1.0, float(emissive.max()))
        images["emissive"] = _scaled_colour(maps.emissive, emissive / out.emissive_strength)
        out.emissive = (0.0, 0.0, 0.0)
    spec.clear_cache()      # the decoded images are not needed once they are re-encoded
    keys = list(images)
    for key, data in zip(keys, pool.map(_png, (images[k] for k in keys))):
        out.pngs[key] = data
    return out


def _fbx_material(material: _Material) -> FbxMaterial:
    strength = material.emissive_strength
    return {
        "name": material.name,
        "base_color_png": material.pngs.get("basecolor"), "normal_png": material.pngs.get("normal"),
        "roughness_png": material.pngs.get("roughness"), "metallic_png": material.pngs.get("metallic"),
        "emissive_png": material.pngs.get("emissive"),
        "base_color_factor": material.base_color, "metallic_factor": material.metallic,
        "roughness_factor": material.roughness,
        "emissive_factor": (strength,) * 3 if "emissive" in material.pngs else material.emissive,
        "alpha_mode": material.alpha_mode, "double_sided": material.double_sided,
    }


def _map_files(name: str, materials: list[_Material]) -> list[dict[str, str]]:
    """Per material, the file name of each of its maps (as the OBJ writes them, beside the model)."""
    single = len(materials) == 1
    return [{key: f"{name}_{key}.png" if single else f"{name}_{m.name}_{key}.png" for key in m.pngs}
            for m in materials]


def _tangents(positions: np.ndarray, normals: np.ndarray, uv: np.ndarray,
              faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MikkTSpace tangents for the FBX -> (positions, normals, uv, tangents, faces). They are per vertex
    when every corner of a vertex agrees (a mesh split along its UV seams); otherwise every corner
    becomes a vertex of its own (three times the rows, still welded by position in the FBX)."""
    corners = mikk_tangents(positions, normals, uv, faces).corners.reshape(-1, 4)
    per_vertex = np.tile(np.array([1.0, 0.0, 0.0, 1.0]), (len(positions), 1))
    flat = faces.reshape(-1)
    per_vertex[flat] = corners
    if np.abs(per_vertex[flat] - corners).max(initial=0.0) <= 1e-6:
        return positions, normals, uv, per_vertex, faces
    return positions[flat], normals[flat], uv[flat], corners, np.arange(len(flat)).reshape(-1, 3)


def placement_of(value: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """A prop's placement as the hub sends it ({"move": [x, y, z], "turn": [x, y, z, w], a unit quaternion}; None:
    the model as it is) -> (rotation 3 x 3, translation): a point p of the model goes to R p + t, in the model's
    own axes (Y up, its front along +Z) and units. ValueError when it is not one."""
    if value is None:
        return None
    try:
        move = np.asarray(value["move"], dtype=np.float64).reshape(3)
        x, y, z, w = (float(part) for part in value["turn"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("'placement' must be {'move': [x, y, z], 'turn': [x, y, z, w]}") from None
    length = (x * x + y * y + z * z + w * w) ** 0.5
    if not np.isfinite(move).all() or not np.isfinite(length) or length < 1e-9:
        raise ValueError("'placement' must hold finite numbers and a quaternion that is not zero")
    x, y, z, w = x / length, y / length, z / length, w / length
    rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return rotation, move


def convert(src: str | Path, outputs: Mapping[str, str | Path], *, name: str,
            progress: Progress | None = None, placement: tuple[np.ndarray, np.ndarray] | None = None
            ) -> dict[str, Any]:
    """Save the model ``src`` as each of ``outputs`` ({"fbx" | "obj": folder}), named ``name``:
    <name>.fbx (maps embedded), <name>.obj + <name>.mtl + <name>_<map>.png. The folders are created.

    ``placement`` (placement_of): the model is turned and moved by it in every file, vertices and normals alike -
    and ``outputs`` may then ask for "glb" too, the model's own GLB with the same done to it (glb.placed).

    Returns {"<format>": {"files": [names written in its folder]}, "stats": {"vertices", "triangles",
    "materials", "seconds"}, "warnings": [...]}. Raises BakeError, with a message for the artist, when
    the model cannot be read or written."""
    started = time.perf_counter()
    report = progress or (lambda fraction, message: None)
    unknown = set(outputs) - set(FORMATS if placement is None else FORMATS + PLACED_FORMATS)
    if unknown or not outputs:
        raise ValueError(f"outputs must be some of {FORMATS}, not {sorted(outputs)}")
    src = Path(src)
    result: dict[str, Any] = {}
    if "glb" in outputs:
        report(0.0, "Placing the 3D model…")
        folder = Path(outputs["glb"])
        folder.mkdir(parents=True, exist_ok=True)
        try:
            (folder / f"{name}.glb").write_bytes(glb.placed(src.read_bytes(), *placement))
        except (OSError, ValueError) as exc:
            raise BakeError(f"The 3D model {src.name} could not be read: {exc}.") from exc
        result["glb"] = {"files": [f"{name}.glb"]}
        if set(outputs) == {"glb"}:
            report(1.0, "Saved.")
            result["stats"] = {"seconds": round(time.perf_counter() - started, 2)}
            result["warnings"] = []
            return result
    report(0.0, "Reading the 3D model…")
    try:
        source = load_source(src)
    except AssetError as exc:
        raise BakeError(f"The 3D model {src.name} could not be read: {exc}.") from exc
    warnings = list(source.warnings)
    positions = source.vertices.astype(np.float64)
    normals = source.normals.astype(np.float64)
    if placement is not None:       # the tangents are worked out from these below, so they turn with them
        positions = positions @ placement[0].T + placement[1]
        normals = normals @ placement[0].T
    uv = source.uv.astype(np.float64)
    faces = source.faces.astype(np.int64)
    face_materials = source.face_materials()

    report(0.2, "Preparing the textures…")
    names = _unique_names(source.materials)
    with ThreadPoolExecutor(max_workers=4) as pool:
        materials = [_material(spec, material_name, pool, warnings.append)
                     for spec, material_name in zip(source.materials, names)]
    map_files = _map_files(name, materials)

    if "fbx" in outputs:
        report(0.45, "Writing the FBX…")
        folder = Path(outputs["fbx"])
        folder.mkdir(parents=True, exist_ok=True)
        if any(m.tangent_normals for m in materials):
            mesh = _tangents(positions, normals, uv, faces)
        else:
            mesh = (positions, normals, uv, None, faces)
        try:
            write_fbx_materials(folder / f"{name}.fbx", name=name, positions=mesh[0], normals=mesh[1],
                                tangents=mesh[3], uvs=mesh[2], indices=mesh[4],
                                materials=[_fbx_material(m) for m in materials], map_files=map_files,
                                face_materials=face_materials)
        except ValueError as exc:       # nothing left to write (every triangle degenerate), or over 4 GB
            raise BakeError(f"{src.name} cannot be saved as FBX: {exc}") from exc
        result["fbx"] = {"files": [f"{name}.fbx"]}

    if "obj" in outputs:
        report(0.7, "Writing the OBJ…")
        folder = Path(outputs["obj"])
        folder.mkdir(parents=True, exist_ok=True)
        _, position_id = np.unique(positions, axis=0, return_inverse=True)
        mtl_materials = []
        files = [f"{name}.obj", f"{name}.mtl"]
        for material, files_of in zip(materials, map_files):
            for key, data in material.pngs.items():
                (folder / files_of[key]).write_bytes(data)
                files.append(files_of[key])
            mtl_materials.append(MtlMaterial(
                material.name,
                MtlMaps(basecolor=files_of.get("basecolor"), normal=files_of.get("normal"),
                        roughness=files_of.get("roughness"), metallic=files_of.get("metallic"),
                        emissive=files_of.get("emissive"), opacity=files_of.get("opacity"),
                        emissive_strength=material.emissive_strength),
                color=material.base_color[:3], roughness=material.roughness, metallic=material.metallic,
                emissive=material.emissive,
                opacity=material.base_color[3] if material.alpha_mode != "OPAQUE" else None))
        write_obj_materials(folder / f"{name}.obj", positions=positions, position_id=position_id.reshape(-1),
                            uv=uv, normals=normals, faces=faces, materials=mtl_materials,
                            face_materials=face_materials, title=f"{name}, saved by Spellcast3D",
                            comment="Spellcast3D. Metal/rough PBR maps; the normal map is tangent space "
                                    "(MikkTSpace); the AO map, if any, is beside it.")
        result["obj"] = {"files": files}

    report(1.0, "Saved.")
    result["stats"] = {"vertices": int(len(positions)), "triangles": int(len(faces)),
                       "materials": len(materials), "seconds": round(time.perf_counter() - started, 2)}
    result["warnings"] = warnings
    return result
