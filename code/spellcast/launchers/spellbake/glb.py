"""A glTF 2.0 binary (GLB) writer: stdlib + numpy, for the bake result and imported models.

trimesh cannot write the bake: it stores tangents as ``_TANGENT``. This writer produces files
the Khronos glTF-Validator passes with no errors and no warnings (tests/bake/test_glb.py):

- one node and one mesh holding one triangle primitive per ``Primitive``, uint32 indices;
- POSITION (with min/max), NORMAL, TANGENT (xyz and w exactly as given: MikkTSpace on v-up UVs
  with w unchanged is glTF's MikkTSpace, see spellbake/__init__) and TEXCOORD_0 = (u, 1 - v);
- metal/rough materials with every factor glTF has, and PNG or JPEG images embedded in the
  binary chunk. Identical images are stored once, so an ORM map that is both the
  metallicRoughness and the occlusion texture costs one image. Samplers filter
  LINEAR / LINEAR_MIPMAP_LINEAR and wrap as each texture says (a baked atlas clamps; an
  imported map may tile). Emissive factors above 1 go through KHR_materials_emissive_strength.

set_base_color_image swaps the base colour image of a written GLB for another (the artist's painting
on a bake) and leaves every other byte of meaning as it was.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import replace_files
from .sampling import CLAMP_TO_EDGE
from .tangents import normalize

GENERATOR = "Spellcast3D projection"
ALPHA_MODES = ("OPAQUE", "MASK", "BLEND")

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_FLOAT, _UINT32 = 5126, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963
_LINEAR, _LINEAR_MIPMAP_LINEAR = 9729, 9987
_EMISSIVE_STRENGTH = "KHR_materials_emissive_strength"


@dataclass(frozen=True)
class Texture:
    """An encoded image (PNG or JPEG bytes) and how it wraps."""

    data: bytes
    name: str = ""
    wrap_s: int = CLAMP_TO_EDGE
    wrap_t: int = CLAMP_TO_EDGE


@dataclass
class Material:
    """A glTF metal/rough material; factors multiply their maps, as glTF defines."""

    name: str
    base_color_factor: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic_factor: float = 1.0
    roughness_factor: float = 1.0
    emissive_factor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_color: Texture | None = None
    metallic_roughness: Texture | None = None   # G roughness, B metallic
    occlusion: Texture | None = None            # R; usually the metallicRoughness image (ORM)
    occlusion_strength: float = 1.0
    normal: Texture | None = None               # tangent space, OpenGL (+Y)
    normal_scale: float = 1.0
    emissive: Texture | None = None
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    extras: dict[str, Any] | None = None


@dataclass
class Primitive:
    """Triangles with their attributes; ``uv`` is v-up (as OBJ and the bake hold it)."""

    positions: np.ndarray                   # (n, 3)
    normals: np.ndarray                     # (n, 3)
    faces: np.ndarray                       # (m, 3) indices into the n vertices
    uv: np.ndarray | None = None            # (n, 2) v up
    tangents: np.ndarray | None = None      # (n, 4) xyz and w = +-1
    material: int = 0


@dataclass
class _Buffer:
    """The binary chunk and the bufferViews / accessors / images that point into it."""

    blobs: list[bytes] = field(default_factory=list)
    size: int = 0
    views: list[dict[str, Any]] = field(default_factory=list)
    accessors: list[dict[str, Any]] = field(default_factory=list)

    def view(self, data: bytes, target: int | None = None) -> int:
        pad = -self.size % 4
        if pad:
            self.blobs.append(bytes(pad))
            self.size += pad
        entry: dict[str, Any] = {"buffer": 0, "byteOffset": self.size, "byteLength": len(data)}
        if target is not None:
            entry["target"] = target
        self.blobs.append(data)
        self.size += len(data)
        self.views.append(entry)
        return len(self.views) - 1

    def accessor(self, array: np.ndarray, kind: str, component: int, target: int, *,
                 bounds: bool = False) -> int:
        entry: dict[str, Any] = {"bufferView": self.view(array.tobytes(), target), "componentType": component,
                                 "count": int(array.shape[0]), "type": kind}
        if bounds:
            entry["min"] = array.min(axis=0).tolist()
            entry["max"] = array.max(axis=0).tolist()
        self.accessors.append(entry)
        return len(self.accessors) - 1

    def binary(self) -> bytes:
        return b"".join(self.blobs) + bytes(-self.size % 4)


def image_mime(data: bytes, webp: bool = False) -> str:
    """The image's glTF mimeType; WebP only where asked (a texture of EXT_texture_webp: set_base_color_image)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if webp and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("glTF images must be PNG or JPEG files" + (" (or WebP)" if webp else ""))


_WEBP = "EXT_texture_webp"


def texture_source(texture: dict[str, Any]) -> Any:
    """The image a glTF texture shows: its own source, or EXT_texture_webp's (Pixal3D's models have only that)."""
    if "source" in texture:
        return texture["source"]
    return ((texture.get("extensions") or {}).get(_WEBP) or {}).get("source")


def _set_source(texture: dict[str, Any], image: int) -> None:
    if "source" in texture or _WEBP not in (texture.get("extensions") or {}):
        texture["source"] = image
    else:
        texture["extensions"][_WEBP]["source"] = image


def _check(primitive: Primitive) -> tuple[np.ndarray, ...]:
    positions = np.asarray(primitive.positions, dtype=np.float64).reshape(-1, 3)
    count = len(positions)
    normals = np.asarray(primitive.normals, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(primitive.faces).reshape(-1, 3)
    if len(normals) != count:
        raise ValueError("a primitive needs one normal per vertex")
    if faces.size == 0 or faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= count:
        raise ValueError("a primitive's faces must be triangles of its own vertices")
    if not (np.isfinite(positions).all() and np.isfinite(normals).all()):
        raise ValueError("vertex positions and normals must be finite")
    return positions, normals, faces


def write_glb(path: str | Path, primitives: list[Primitive], materials: list[Material], *,
              name: str = "mesh", generator: str = GENERATOR) -> int:
    """Write the GLB; returns its size in bytes."""
    if not primitives:
        raise ValueError("a GLB needs at least one primitive")
    buffer = _Buffer()
    gl_primitives = []
    for primitive in primitives:
        if not 0 <= primitive.material < len(materials):
            raise ValueError(f"primitive material {primitive.material} does not exist")
        positions, normals, faces = _check(primitive)
        attributes = {
            "POSITION": buffer.accessor(positions.astype(np.float32), "VEC3", _FLOAT, _ARRAY_BUFFER, bounds=True),
            "NORMAL": buffer.accessor(normalize(normals).astype(np.float32), "VEC3", _FLOAT, _ARRAY_BUFFER),
        }
        if primitive.tangents is not None:
            tangents = np.asarray(primitive.tangents, dtype=np.float64).reshape(-1, 4)
            packed = np.concatenate([normalize(tangents[:, :3]), np.where(tangents[:, 3:] < 0, -1.0, 1.0)], axis=1)
            attributes["TANGENT"] = buffer.accessor(packed.astype(np.float32), "VEC4", _FLOAT, _ARRAY_BUFFER)
        if primitive.uv is not None:
            uv = np.asarray(primitive.uv, dtype=np.float64).reshape(-1, 2)
            flipped = np.stack([uv[:, 0], 1.0 - uv[:, 1]], axis=1).astype(np.float32)
            attributes["TEXCOORD_0"] = buffer.accessor(flipped, "VEC2", _FLOAT, _ARRAY_BUFFER)
        indices = buffer.accessor(faces.astype(np.uint32).reshape(-1), "SCALAR", _UINT32, _ELEMENT_ARRAY_BUFFER)
        gl_primitives.append({"attributes": attributes, "indices": indices, "material": primitive.material,
                              "mode": 4})

    images: list[dict[str, Any]] = []
    image_of: dict[str, int] = {}
    samplers: list[dict[str, int]] = []
    sampler_of: dict[tuple[int, int], int] = {}
    textures: list[dict[str, int]] = []
    texture_of: dict[tuple[int, int], int] = {}

    def texture(tex: Texture) -> int:
        digest = hashlib.sha1(tex.data).hexdigest()
        if digest not in image_of:
            entry = {"bufferView": buffer.view(tex.data), "mimeType": image_mime(tex.data)}
            if tex.name:
                entry["name"] = tex.name
            images.append(entry)
            image_of[digest] = len(images) - 1
        wraps = (int(tex.wrap_s), int(tex.wrap_t))
        if wraps not in sampler_of:
            samplers.append({"magFilter": _LINEAR, "minFilter": _LINEAR_MIPMAP_LINEAR,
                             "wrapS": wraps[0], "wrapT": wraps[1]})
            sampler_of[wraps] = len(samplers) - 1
        key = (image_of[digest], sampler_of[wraps])
        if key not in texture_of:
            textures.append({"source": key[0], "sampler": key[1]})
            texture_of[key] = len(textures) - 1
        return texture_of[key]

    extensions_used: set[str] = set()
    gl_materials = [_material(material, texture, extensions_used) for material in materials]

    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": gl_primitives}],
        "materials": gl_materials,
        "accessors": buffer.accessors,
        "bufferViews": buffer.views,
        "buffers": [{"byteLength": buffer.size + (-buffer.size % 4)}],
    }
    if extensions_used:
        gltf["extensionsUsed"] = sorted(extensions_used)
    if images:
        gltf.update(images=images, samplers=samplers, textures=textures)
    document = json.dumps(gltf, separators=(",", ":"), allow_nan=False).encode("utf-8")
    document += b" " * (-len(document) % 4)
    binary = buffer.binary()
    total = 12 + 8 + len(document) + 8 + len(binary)
    with open(path, "wb") as out:
        out.write(struct.pack("<III", _GLB_MAGIC, 2, total))
        out.write(struct.pack("<II", len(document), _CHUNK_JSON))
        out.write(document)
        out.write(struct.pack("<II", len(binary), _CHUNK_BIN))
        out.write(binary)
    return total


def _material(material: Material, texture: Callable[[Texture], int], extensions_used: set[str]) -> dict[str, Any]:
    if material.alpha_mode not in ALPHA_MODES:
        raise ValueError(f"alpha mode must be one of {ALPHA_MODES}, not {material.alpha_mode!r}")
    pbr: dict[str, Any] = {"baseColorFactor": [float(v) for v in material.base_color_factor],
                           "metallicFactor": float(material.metallic_factor),
                           "roughnessFactor": float(material.roughness_factor)}
    if material.base_color is not None:
        pbr["baseColorTexture"] = {"index": texture(material.base_color)}
    if material.metallic_roughness is not None:
        pbr["metallicRoughnessTexture"] = {"index": texture(material.metallic_roughness)}
    out: dict[str, Any] = {"name": material.name, "pbrMetallicRoughness": pbr}
    if material.normal is not None:
        out["normalTexture"] = {"index": texture(material.normal), "scale": float(material.normal_scale)}
    if material.occlusion is not None:
        out["occlusionTexture"] = {"index": texture(material.occlusion),
                                   "strength": float(material.occlusion_strength)}
    emissive = [float(v) for v in material.emissive_factor]
    strength = max(emissive)
    if strength > 1.0:     # core glTF caps the factor at 1
        emissive = [v / strength for v in emissive]
        out["extensions"] = {_EMISSIVE_STRENGTH: {"emissiveStrength": strength}}
        extensions_used.add(_EMISSIVE_STRENGTH)
    if material.emissive is not None:
        out["emissiveTexture"] = {"index": texture(material.emissive)}
    if any(emissive):
        out["emissiveFactor"] = emissive
    if material.alpha_mode != "OPAQUE":
        out["alphaMode"] = material.alpha_mode
        if material.alpha_mode == "MASK":
            out["alphaCutoff"] = float(material.alpha_cutoff)
    if material.double_sided:
        out["doubleSided"] = True
    if material.extras:
        out["extras"] = material.extras
    return out


# ---------------------------------------------------------------------- the base colour image of a GLB


def replace_base_color_image(path: str | os.PathLike[str], image: bytes, material: int = 0) -> int:
    """set_base_color_image on the file at ``path``, written back atomically; returns its new size."""
    path = Path(path)
    data = set_base_color_image(path.read_bytes(), image, material)
    replace_files({path: data})
    return len(data)


def set_base_color_image(data: bytes, image: bytes, material: int = 0) -> bytes:
    """The GLB ``data`` with the image that ``material``'s baseColorTexture shows replaced by ``image``
    (PNG or JPEG bytes).

    Every base colour slot that shows that image gets the new one. When a slot of another kind shows it
    too (write_glb stores identical images once), that slot keeps it and the base colour gets an image,
    bufferView and texture of its own. The image's bytes are swapped in place when nothing else uses
    them (the bufferViews after it move by a multiple of 4), else the new image goes at the end of the
    binary chunk. Accessors, other images, extras and the JSON's order stay as they were.
    """
    gltf, binary, rest = _read_glb(data)
    mime = image_mime(image, webp=_WEBP in (gltf.get("extensionsUsed") or []))
    textures, images, views = gltf.get("textures") or [], gltf.get("images") or [], gltf.get("bufferViews") or []
    try:
        slot = gltf["materials"][_index(material, gltf["materials"])]["pbrMetallicRoughness"]["baseColorTexture"]
        source = _index(texture_source(textures[_index(slot["index"], textures)]), images)
        view_index = _index(images[source]["bufferView"], views)
    except (KeyError, TypeError, IndexError):
        raise ValueError("the material has no base colour image stored in the GLB") from None
    buffers = gltf.get("buffers") or []
    if views[view_index].get("buffer") != 0 or not buffers or "uri" in buffers[0]:
        raise ValueError("the base colour image is not stored in the GLB's binary chunk")

    def shows(slot: dict[str, Any]) -> bool:
        index = slot["index"]
        return 0 <= index < len(textures) and texture_source(textures[index]) == source

    slots = [(key, slot) for key, slot in _texture_slots(gltf) if shows(slot)]
    base = [slot for key, slot in slots if key == "baseColorTexture"]
    others = [slot for key, slot in slots if key != "baseColorTexture"]
    if others:
        binary, new_view = _append_view(binary, image, views)
        entry: dict[str, Any] = {"bufferView": new_view, "mimeType": mime}
        if "name" in images[source]:
            entry["name"] = images[source]["name"]
        images.append(entry)
        shared = {slot["index"] for slot in others}
        copies: dict[int, int] = {}
        for slot in base:
            index = slot["index"]
            if index not in shared:
                _set_source(textures[index], len(images) - 1)
                continue
            if index not in copies:
                textures.append(copy.deepcopy(textures[index]))
                _set_source(textures[-1], len(images) - 1)
                copies[index] = len(textures) - 1
            slot["index"] = copies[index]
    elif _splicable(gltf, view_index, source):
        binary = _splice_view(binary, image, views, view_index)
        images[source]["mimeType"] = mime
    else:
        binary, images[source]["bufferView"] = _append_view(binary, image, views)
        images[source]["mimeType"] = mime
    binary += bytes(-len(binary) % 4)
    buffers[0]["byteLength"] = len(binary)
    document = json.dumps(gltf, separators=(",", ":"), allow_nan=False).encode("utf-8")
    document += b" " * (-len(document) % 4)
    total = 12 + 8 + len(document) + 8 + len(binary) + len(rest)
    return b"".join([struct.pack("<III", _GLB_MAGIC, 2, total), struct.pack("<II", len(document), _CHUNK_JSON),
                     document, struct.pack("<II", len(binary), _CHUNK_BIN), binary, rest])


def _index(value: Any, items: list[Any]) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < len(items):
        raise IndexError(value)
    return value


def _read_glb(data: bytes) -> tuple[dict[str, Any], bytes, bytes]:
    """(the JSON document, the binary chunk, the chunks after it as they are) of a GLB with one binary chunk."""
    if len(data) < 20 or struct.unpack_from("<III", data, 0) != (_GLB_MAGIC, 2, len(data)):
        raise ValueError("this is not a glTF 2.0 binary (GLB) file")
    chunks: list[tuple[int, bytes]] = []
    at = 12
    while at < len(data):
        if at + 8 > len(data):
            raise ValueError("the GLB file is damaged (a chunk is cut off)")
        length, kind = struct.unpack_from("<II", data, at)
        if at + 8 + length > len(data):
            raise ValueError("the GLB file is damaged (a chunk is cut off)")
        chunks.append((kind, data[at + 8:at + 8 + length]))
        at += 8 + length
    if len(chunks) < 2 or chunks[0][0] != _CHUNK_JSON or chunks[1][0] != _CHUNK_BIN \
            or any(kind in (_CHUNK_JSON, _CHUNK_BIN) for kind, _ in chunks[2:]):
        raise ValueError("the GLB needs its JSON chunk followed by one binary chunk")
    gltf = json.loads(chunks[0][1].rstrip(b"\x00"))
    if not isinstance(gltf, dict):
        raise ValueError("the GLB's JSON chunk is not a glTF document")
    rest = b"".join(struct.pack("<II", len(chunk), kind) + chunk for kind, chunk in chunks[2:])
    return gltf, chunks[1][1], rest


def _texture_slots(node: Any, key: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    """Every textureInfo in the document (materials and their extensions): (its key, the dict)."""
    if isinstance(node, dict):
        if key.endswith("Texture") and isinstance(node.get("index"), int):
            yield key, node
        for child_key, child in node.items():
            yield from _texture_slots(child, child_key)
    elif isinstance(node, list):
        for child in node:
            yield from _texture_slots(child)


def _splicable(gltf: dict[str, Any], view_index: int, source: int) -> bool:
    """True when the image's bufferView is its own: no other image or accessor reads it, no other view
    overlaps it, and no view keeps offsets of its own in an extension (those could not be moved)."""
    views = gltf["bufferViews"]
    start = views[view_index].get("byteOffset", 0)
    end = start + views[view_index]["byteLength"]
    for index, view in enumerate(views):
        if "extensions" in view:
            return False
        if index != view_index and view.get("buffer") == 0 \
                and view.get("byteOffset", 0) < end and view.get("byteOffset", 0) + view["byteLength"] > start:
            return False
    readers = [entry.get("bufferView") for entry in gltf.get("accessors") or []]
    readers += [entry.get("bufferView") for index, entry in enumerate(gltf["images"]) if index != source]
    return view_index not in readers


def _splice_view(binary: bytes, image: bytes, views: list[dict[str, Any]], view_index: int) -> bytes:
    """The binary chunk with the view's bytes (and the padding after them) replaced by ``image``; the views
    after it move by a multiple of 4, so every alignment holds."""
    start = views[view_index].get("byteOffset", 0)
    end = start + views[view_index]["byteLength"]
    following = [view.get("byteOffset", 0) for view in views
                 if view.get("buffer") == 0 and view.get("byteOffset", 0) >= end]
    region_end = min(following, default=len(binary))
    pad = (region_end - start - len(image)) % 4
    shift = start + len(image) + pad - region_end
    for view in views:
        if view.get("buffer") == 0 and view.get("byteOffset", 0) >= end:
            view["byteOffset"] = view.get("byteOffset", 0) + shift
    views[view_index]["byteLength"] = len(image)
    return binary[:start] + image + bytes(pad) + binary[region_end:]


def _append_view(binary: bytes, image: bytes, views: list[dict[str, Any]]) -> tuple[bytes, int]:
    """The binary chunk with ``image`` added at its end (4-byte aligned), and the new view's index."""
    binary += bytes(-len(binary) % 4)
    views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(image)})
    return binary + image, len(views) - 1
