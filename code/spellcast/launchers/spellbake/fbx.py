"""Binary FBX 7.4 writer: one mesh with PBR materials, for the Projection stage's baked low poly
(write_fbx, one material) and for saving a 3D model as FBX (write_fbx_materials, spellbake.convert).

Runs inside the imesh venv (stdlib + numpy + zlib, no FBX SDK) and never imports the hub package.
Written from the file layout that Blender's io_scene_fbx (encode_bin.py) documents:

- header: ``b"Kaydara FBX Binary  \\x00\\x1a\\x00"`` followed by the uint32 version (7400).
- record: end offset (absolute, uint32 before 7500), property count and property byte length,
  then a uint8 name length, the name, the properties and the nested records. When a record has
  children, or has no properties at all, a 13-byte null record closes its nested list.
- property: one type byte followed by the payload. C = bool byte, I = int32, L = int64,
  D = float64, S = string, R = raw bytes. The arrays d and i are uint32 count, uint32 encoding
  (0 = raw, 1 = zlib) and uint32 byte length, then the data. Arrays larger than 128 bytes are
  deflated, as the SDK does.
- footer: the root's null record, a 16-byte footer id, 4 zero bytes, zero padding to a 16-byte
  boundary (16 bytes when the offset is already aligned), the version, 120 zero bytes and a 16-byte
  magic. The FBX SDK checks FileId and the footer id against CreationTime, so all three use the
  fixed values that other non-SDK writers (Blender, Assimp) also use.

Conventions. The input is in the glTF frame (metres, +Y up, +Z front, right-handed); the FBX that
comes out is described below.

- Axes: Y up, Z front, X coord, all signs +1 (the SDK's "MayaYUp"). The vertex data is not rotated.
  Importers that work Z-up (Blender, Unreal) rotate it the way their glTF importers do.
- Units: centimetres. ``UnitScaleFactor`` is 1, the positions are multiplied by 100 and every
  Lcl transform is identity. FBX's native unit is the centimetre, so the object comes out at its
  true size both in importers that honour ``UnitScaleFactor`` (Blender, Unity with Convert Units,
  Maya, Max) and in importers that ignore it (Unreal's legacy importer leaves Convert Scene Unit
  off by default). A metre file with ``UnitScaleFactor`` 100 would import 100 times too small
  there. In Blender the object gets scale 0.01 and rotation X 90 deg, as any Maya file does, and
  its dimensions are the true metres; test_fbx.py checks this with Blender's own importer.
- Topology: triangles only, because the bake's tangent space assumes these triangles. Control
  points are welded by exact position, so UV seams stay connected in DCC tools. Two kinds of
  triangle are left out: those whose corners weld to fewer than three points (they have no area),
  and repeats of the same three points (a mesh in Blender cannot hold both). Blender 5.1's importer
  crashes on either kind while it sets custom normals. Edges and
  ByEdge smoothing are derived from the normals (an edge is hard where the corner normals of its
  two faces differ). Normals, tangents and binormals are per polygon vertex (Direct), with
  binormal = w * (N x T), so importers that read tangents keep the MikkTSpace sign; without
  tangents (a model with no normal map) neither is written. UVs are v-up (as FBX expects) and
  IndexToDirect.
- Materials: FbxSurfacePhong, connected to the model in order; with several, LayerElementMaterial
  gives each polygon its material's index (ByPolygon). The PBR values go where Blender's importer
  reads them (its exporter writes them there too). Unity and Unreal read the colour, normal and
  emissive slots the same way.

      base colour -> DiffuseColor, and also TransparentColor when alpha_mode is MASK or BLEND
      normal      -> NormalMap (tangent space; its Y convention is whatever PNG the caller passes)
      roughness   -> ShininessExponent. This is Blender's convention: the map holds roughness, not
                     gloss. The Shininess value is (10 * (1 - roughness))^2, the inverse of
                     Blender's import formula.
      metallic    -> ReflectionFactor (Blender's convention)
      emissive    -> EmissiveColor. FBX colours stay within 0..1, so an emissive factor above 1 (glTF's
                     emissive strength) is split: its colour over its largest component goes to
                     EmissiveColor and that component to EmissiveFactor, which Blender reads as the
                     emission strength and Assimp multiplies in.

  AO has no FBX slot that Blender, Unity or Unreal read, so it is not written. The baked AO stays
  in maps/ao.png and in the GLB's ORM. Neither the alpha cutoff nor double_sided has an FBX field
  that these importers read.

Large arrays are deflated on a thread pool while the records are built (zlib releases the GIL), which
makes a million-triangle mesh several times faster to write; the bytes are the same either way.

parse_fbx reads such a file back into records whose properties stay encoded, so set_embedded_map can
swap one embedded map (the artist's painting on a bake) and write the rest back byte for byte.
"""

from __future__ import annotations

import os
import struct
import zlib
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from itertools import count
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from . import replace_files

FBX_VERSION = 7400
#: FBX files are written in centimetres (UnitScaleFactor 1); positions are metres times this.
CM_PER_METRE = 100.0

_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"
_NULL_RECORD = bytes(13)          # end offset, property count, property length (uint32) + empty name
_RAW_ARRAY_LIMIT = 128            # arrays up to this many bytes are stored uncompressed
_CREATION_TIME = "1970-01-01 10:00:00:000"
_FILE_ID = bytes.fromhex("28b32aebb624ccc2bfc8b02aa92bfcf1")
_FOOTER_ID = bytes.fromhex("fabcab09d0c8d466b176fb831cf7267e")
_FOOTER_MAGIC = bytes.fromhex("f85a8c6adef5d97eece90ce3758f290b")
_KTIME_PER_SECOND = 46186158000   # FBX 7.x time unit
_APPLICATION = "Spellcast3D"
_CREATOR = "Spellcast3D projection"
_UV_SET = "UVMap"
_NORMAL_TOLERANCE = 1e-6          # corner normals closer than this count as one smooth normal
_POOLED_ARRAY_BYTES = 1 << 16     # arrays at least this big are deflated on the writer's thread pool
_SCALAR_BYTES = {ord("Y"): 2, ord("C"): 1, ord("I"): 4, ord("F"): 4, ord("D"): 8, ord("L"): 8}
_ARRAY_CODES = frozenset(b"fdlib")
_BLOB_CODES = frozenset(b"SR")
# The thread pool of the write_fbx_materials call running in this thread (None: deflate in place).
_POOL: ContextVar[ThreadPoolExecutor | None] = ContextVar("_POOL", default=None)

#: (map, material key, the FBX material property its texture is bound to)
TEXTURE_SLOTS: tuple[tuple[str, str, str], ...] = (
    ("basecolor", "base_color_png", "DiffuseColor"),
    ("normal", "normal_png", "NormalMap"),
    ("roughness", "roughness_png", "ShininessExponent"),
    ("metallic", "metallic_png", "ReflectionFactor"),
    ("emissive", "emissive_png", "EmissiveColor"),
)


class FbxMaterial(TypedDict, total=False):
    """The material of ``write_fbx``. Missing keys take glTF's defaults; PNGs are encoded files."""

    name: str
    base_color_png: bytes | None     # sRGB; RGB when OPAQUE: Blender blends any RGBA base colour
    normal_png: bytes | None         # tangent space (MikkTSpace)
    roughness_png: bytes | None      # grey (L)
    metallic_png: bytes | None       # grey (L)
    ao_png: bytes | None             # accepted but not written: FBX has no AO slot (module doc)
    emissive_png: bytes | None       # sRGB
    base_color_factor: Sequence[float]   # linear RGBA, default (1, 1, 1, 1)
    metallic_factor: float               # default 1
    roughness_factor: float              # default 1
    emissive_factor: Sequence[float]     # linear RGB, default (0, 0, 0); above 1 is fine (module doc)
    alpha_mode: str                      # "OPAQUE" (default) | "MASK" | "BLEND"
    double_sided: bool                   # accepted but not written (module doc)


class _Int64(int):
    """An int written as an FBX int64 (object ids, times) rather than an int32."""


Prop = bool | int | float | str | bytes | np.ndarray


def _array(code: bytes, data: bytes, size: int) -> bytes:
    encoding = int(len(data) > _RAW_ARRAY_LIMIT)
    if encoding:
        data = zlib.compress(data)
    return code + struct.pack("<3I", size, encoding, len(data)) + data


def _encode(value: Prop) -> bytes | Future[bytes]:
    """One property: its type byte and payload (a large array's still being deflated on the pool)."""
    if isinstance(value, bool):
        return b"C" + bytes((value,))
    if isinstance(value, _Int64):
        return b"L" + struct.pack("<q", value)
    if isinstance(value, int):
        return b"I" + struct.pack("<i", value)
    if isinstance(value, float):
        return b"D" + struct.pack("<d", value)
    if isinstance(value, (str, bytes)):
        data = value.encode("utf-8") if isinstance(value, str) else value
        return (b"S" if isinstance(value, str) else b"R") + struct.pack("<I", len(data)) + data
    if isinstance(value, np.ndarray):
        code, dtype = {np.dtype(np.float64): (b"d", "<f8"), np.dtype(np.int32): (b"i", "<i4")}[value.dtype]
        data = np.ascontiguousarray(value, dtype).tobytes()
        pool = _POOL.get()
        if pool is not None and len(data) >= _POOLED_ARRAY_BYTES:
            return pool.submit(_array, code, data, value.size)
        return _array(code, data, value.size)
    raise TypeError(f"no FBX property type for {type(value).__name__}")


class _Node:
    """One FBX record: a name, its properties and its nested records."""

    __slots__ = ("name", "props", "children")

    def __init__(self, name: str, props: Iterable[Prop] = ()) -> None:
        self.name = name.encode("ascii")
        self.props = [_encode(p) for p in props]
        self.children: list[_Node] = []

    @classmethod
    def raw(cls, name: bytes, props: list[bytes]) -> _Node:
        """A record read from a file: its properties stay as they are encoded."""
        node = cls.__new__(cls)
        node.name, node.props, node.children = name, list(props), []
        return node

    def add(self, name: str, *props: Prop) -> _Node:
        child = _Node(name, props)
        self.children.append(child)
        return child

    def write(self, out: list[bytes], offset: int) -> int:
        """Append the record at absolute file ``offset`` to ``out``; return the offset after it."""
        props = b"".join(p if isinstance(p, bytes) else p.result() for p in self.props)
        head = len(out)
        out.append(b"")                       # the header needs the end offset: filled in below
        end = offset + 13 + len(self.name) + len(props)
        for child in self.children:
            end = child.write(out, end)
        if self.children or not self.props:
            out.append(_NULL_RECORD)
            end += len(_NULL_RECORD)
        if end >= 1 << 32:
            raise ValueError("The model is too large for an FBX 7.4 file (4 GB). Use the GLB instead.")
        out[head] = struct.pack("<3IB", end, len(self.props), len(props), len(self.name)) + self.name + props
        return end


#: Properties70 kinds: FBX type name, label, flags, and the value cast (None: no value).
_P_KINDS: dict[str, tuple[str, str, str, type | None]] = {
    "int": ("int", "Integer", "", int),
    "enum": ("enum", "", "", int),
    "bool": ("bool", "", "", int),        # Properties70 stores bools as int32
    "double": ("double", "Number", "", float),
    "number": ("Number", "", "A", float),
    "color": ("Color", "", "A", float),
    "color_rgb": ("ColorRGB", "Color", "", float),
    "vector": ("Vector3D", "Vector", "", float),
    "translation": ("Lcl Translation", "", "A", float),
    "rotation": ("Lcl Rotation", "", "A", float),
    "scaling": ("Lcl Scaling", "", "A", float),
    "string": ("KString", "", "", str),
    "url": ("KString", "Url", "", str),
    "xref_url": ("KString", "XRefUrl", "", str),
    "time": ("KTime", "Time", "", _Int64),
    "datetime": ("DateTime", "", "", str),
    "compound": ("Compound", "", "", None),
    "object": ("object", "", "", None),
}


def _properties(parent: _Node, entries: Iterable[tuple[str, str, Any]]) -> None:
    """A Properties70 block of ``(name, kind, value)``; a vector value is a 3-sequence."""
    block = parent.add("Properties70")
    for name, kind, value in entries:
        type_name, label, flags, cast = _P_KINDS[kind]
        values = () if cast is None else tuple(map(cast, value if isinstance(value, (tuple, list)) else (value,)))
        block.add("P", name, type_name, label, flags, *values)


def _name(name: str, cls: str) -> str:
    """An object name as the binary format spells it: ``name\\x00\\x01Class``."""
    return "".join(ch for ch in name if ch >= " ") + "\x00\x01" + cls


def _material_properties(material: FbxMaterial) -> list[tuple[str, str, Any]]:
    """The Phong properties carrying the PBR factors (see the module doc for the mapping)."""
    base = tuple(material.get("base_color_factor", (1.0, 1.0, 1.0, 1.0)))
    alpha = base[3] if material.get("alpha_mode", "OPAQUE") != "OPAQUE" else 1.0
    shininess = (10.0 * (1.0 - material.get("roughness_factor", 1.0))) ** 2
    emissive = tuple(float(c) for c in material.get("emissive_factor", (0.0, 0.0, 0.0)))
    strength = max(1.0, *emissive)
    return [
        ("ShadingModel", "string", "Phong"),
        ("DiffuseColor", "color", base[:3]),
        ("DiffuseFactor", "number", 1.0),
        # Unity and Blender read alpha as 1 - TransparencyFactor, and fall back to
        # 1 - TransparentColor.r when that is exactly 0 or 1.
        ("TransparentColor", "color", (1.0 - alpha,) * 3),
        ("TransparencyFactor", "number", 1.0 - alpha),
        ("Opacity", "number", alpha),
        ("EmissiveColor", "color", tuple(c / strength for c in emissive)),
        ("EmissiveFactor", "number", strength),
        ("AmbientColor", "color", (0.0, 0.0, 0.0)),
        ("AmbientFactor", "number", 0.0),
        ("SpecularColor", "color", (1.0, 1.0, 1.0)),
        ("SpecularFactor", "number", 0.25),       # Blender: specular = 2 * factor = 0.5 (F0 0.04)
        ("Shininess", "number", shininess),
        ("ShininessExponent", "number", shininess),
        ("ReflectionColor", "color", (1.0, 1.0, 1.0)),
        ("ReflectionFactor", "number", material.get("metallic_factor", 1.0)),
        ("NormalMap", "vector", (0.0, 0.0, 0.0)),
        ("BumpFactor", "double", 1.0),
    ]


def _checked(array: Any, what: str, width: int, rows: int | None = None) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 2 or arr.shape[1] != width or (rows is not None and arr.shape[0] != rows):
        raise ValueError(f"{what} must have shape ({rows if rows is not None else 'n'}, {width}), got {arr.shape}")
    if arr.dtype.kind == "f" and not np.isfinite(arr).all():
        raise ValueError(f"{what} contains NaN or infinite values")
    return arr


def _edge_smoothing(polygon: np.ndarray, corner_normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FBX ``Edges`` (the polygon vertex starting each unique edge) and their ByEdge smoothing.

    ``polygon`` is (m, 3) control-point indices; an edge is smooth (1) when both faces have the
    same normals at both of its ends, or when it has only one face; hard (0) otherwise.
    """
    start = polygon.reshape(-1).astype(np.int64)
    following = np.arange(start.size).reshape(-1, 3)[:, [1, 2, 0]].reshape(-1)   # next corner of the triangle
    end = start[following]
    key = np.minimum(start, end) * (int(start.max()) + 1) + np.maximum(start, end)
    at_low_end = np.where((start <= end)[:, None], corner_normals, corner_normals[following])
    at_high_end = np.where((start <= end)[:, None], corner_normals[following], corner_normals)

    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    first = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
    faces = np.diff(np.r_[first, key.size])
    smooth = faces == 1
    pair = faces == 2
    a, b = order[first[pair]], order[first[pair] + 1]
    smooth[pair] = ((np.abs(at_low_end[a] - at_low_end[b]).max(1) <= _NORMAL_TOLERANCE)
                    & (np.abs(at_high_end[a] - at_high_end[b]).max(1) <= _NORMAL_TOLERANCE))
    return order[first].astype(np.int32), smooth.astype(np.int32)


def _layer_element(geometry: _Node, kind: str, mapping: str, reference: str, name: str = "",
                   version: int = 101, **arrays: np.ndarray) -> str:
    element = geometry.add(kind, 0)
    element.add("Version", version)
    element.add("Name", name)
    element.add("MappingInformationType", mapping)
    element.add("ReferenceInformationType", reference)
    for key, values in arrays.items():
        element.add(key, values)
    return kind


def _geometry(objects: _Node, geometry_id: _Int64, name: str, positions: np.ndarray, normals: np.ndarray,
              tangents: np.ndarray | None, uvs: np.ndarray, indices: np.ndarray,
              face_materials: np.ndarray | None) -> None:
    # Welded over the vertices rather than the corners (a third of the rows to sort); points no kept
    # triangle uses are dropped below either way.
    points, control = np.unique(positions, axis=0, return_inverse=True)
    polygon = control.reshape(-1)[indices]
    distinct = (polygon[:, 0] != polygon[:, 1]) & (polygon[:, 1] != polygon[:, 2]) & (polygon[:, 2] != polygon[:, 0])
    first = np.zeros(len(polygon), bool)
    first[np.unique(np.sort(polygon, axis=1), axis=0, return_index=True)[1]] = True
    keep = distinct & first
    if not keep.any():
        raise ValueError("The mesh has no triangle with three distinct corners to export.")
    used, polygon = np.unique(polygon[keep], return_inverse=True)   # points only dropped triangles used go too
    welded, polygon = points[used], polygon.reshape(-1, 3).astype(np.int32)
    corners = indices[keep].reshape(-1)
    corner_normals = normals[corners].astype(np.float64)
    edges, smoothing = _edge_smoothing(polygon, corner_normals)
    polygon[:, 2] = ~polygon[:, 2]            # a negative index (-i - 1) closes each polygon

    geometry = objects.add("Geometry", geometry_id, _name(name, "Geometry"), "Mesh")
    geometry.add("GeometryVersion", 124)
    geometry.add("Vertices", (welded.astype(np.float64) * CM_PER_METRE).reshape(-1))
    geometry.add("PolygonVertexIndex", polygon.reshape(-1))
    geometry.add("Edges", edges)
    layers = [_layer_element(geometry, "LayerElementNormal", "ByPolygonVertex", "Direct",
                             Normals=corner_normals.reshape(-1))]
    if tangents is not None:
        corner_tangents = tangents[corners, :3].astype(np.float64)
        binormals = tangents[corners, 3:4].astype(np.float64) * np.cross(corner_normals, corner_tangents)
        layers += [_layer_element(geometry, "LayerElementBinormal", "ByPolygonVertex", "Direct", _UV_SET,
                                  Binormals=binormals.reshape(-1)),
                   _layer_element(geometry, "LayerElementTangent", "ByPolygonVertex", "Direct", _UV_SET,
                                  Tangents=corner_tangents.reshape(-1))]
    if face_materials is None:
        by_material = ("AllSame", np.zeros(1, np.int32))
    else:
        by_material = ("ByPolygon", np.ascontiguousarray(face_materials[keep], np.int32))
    layers += [
        _layer_element(geometry, "LayerElementSmoothing", "ByEdge", "Direct", version=102,
                       Smoothing=smoothing),
        _layer_element(geometry, "LayerElementUV", "ByPolygonVertex", "IndexToDirect", _UV_SET,
                       UV=uvs.astype(np.float64).reshape(-1), UVIndex=corners.astype(np.int32)),
        _layer_element(geometry, "LayerElementMaterial", by_material[0], "IndexToDirect",
                       Materials=by_material[1]),
    ]
    layer = geometry.add("Layer", 0)
    layer.add("Version", 100)
    for kind in layers:
        element = layer.add("LayerElement")
        element.add("Type", kind)
        element.add("TypedIndex", 0)


def _header(root: _Node, document: str) -> None:
    extension = root.add("FBXHeaderExtension")
    extension.add("FBXHeaderVersion", 1003)
    extension.add("FBXVersion", FBX_VERSION)
    extension.add("EncryptionType", 0)
    stamp = extension.add("CreationTimeStamp")    # the same fixed instant as CreationTime
    for field, value in (("Version", 1000), ("Year", 1970), ("Month", 1), ("Day", 1), ("Hour", 10),
                         ("Minute", 0), ("Second", 0), ("Millisecond", 0)):
        stamp.add(field, value)
    extension.add("Creator", _CREATOR)
    info = extension.add("SceneInfo", _name("GlobalInfo", "SceneInfo"), "UserData")
    info.add("Type", "UserData")
    info.add("Version", 100)
    meta = info.add("MetaData")
    meta.add("Version", 100)
    for field in ("Title", "Subject", "Author", "Keywords", "Revision", "Comment"):
        meta.add(field, "")
    application = [("ApplicationVendor", "string", _APPLICATION), ("ApplicationName", "string", _APPLICATION),
                   ("ApplicationVersion", "string", "1"), ("DateTime_GMT", "datetime", "01/01/1970 10:00:00.000")]
    _properties(info, [("DocumentUrl", "url", document), ("SrcDocumentUrl", "url", document),
                       ("Original", "compound", None),
                       *((f"Original|{key}", kind, value) for key, kind, value in application),
                       ("Original|FileName", "string", document),
                       ("LastSaved", "compound", None),
                       *((f"LastSaved|{key}", kind, value) for key, kind, value in application)])
    root.add("FileId", _FILE_ID)
    root.add("CreationTime", _CREATION_TIME)
    root.add("Creator", _CREATOR)

    settings = root.add("GlobalSettings")
    settings.add("Version", 1000)
    _properties(settings, [
        ("UpAxis", "int", 1), ("UpAxisSign", "int", 1), ("FrontAxis", "int", 2), ("FrontAxisSign", "int", 1),
        ("CoordAxis", "int", 0), ("CoordAxisSign", "int", 1), ("OriginalUpAxis", "int", 1),
        ("OriginalUpAxisSign", "int", 1), ("UnitScaleFactor", "double", 1.0),
        ("OriginalUnitScaleFactor", "double", 1.0), ("AmbientColor", "color_rgb", (0.0, 0.0, 0.0)),
        ("DefaultCamera", "string", "Producer Perspective"), ("TimeMode", "enum", 11),   # 24 fps
        ("TimeSpanStart", "time", 0), ("TimeSpanStop", "time", _KTIME_PER_SECOND),
        ("CustomFrameRate", "double", 24.0),
    ])


def _definitions(root: _Node, materials: int, textures: int, videos: int) -> None:
    """Object counts and the property templates the SDK falls back to."""
    types: list[tuple[str, int, str, list[tuple[str, str, Any]]]] = [
        ("GlobalSettings", 1, "", []),
        ("Model", 1, "FbxNode", [("Lcl Translation", "translation", (0.0, 0.0, 0.0)),
                                 ("Lcl Rotation", "rotation", (0.0, 0.0, 0.0)),
                                 ("Lcl Scaling", "scaling", (1.0, 1.0, 1.0))]),
        ("Geometry", 1, "FbxMesh", [("Color", "color_rgb", (0.8, 0.8, 0.8)), ("Primary Visibility", "bool", 1),
                                    ("Casts Shadows", "bool", 1), ("Receive Shadows", "bool", 1)]),
        ("Material", materials, "FbxSurfacePhong", _material_properties({})),
        ("Texture", textures, "FbxFileTexture", [("UVSet", "string", "default"), ("UseMaterial", "bool", 0),
                                                 ("WrapModeU", "enum", 0), ("WrapModeV", "enum", 0)]),
        ("Video", videos, "FbxVideo", [("Path", "xref_url", ""), ("Width", "int", 0), ("Height", "int", 0)]),
    ]
    definitions = root.add("Definitions")
    definitions.add("Version", 100)
    definitions.add("Count", sum(number for _, number, _, _ in types))
    for type_name, number, template, entries in types:
        if not number:
            continue
        object_type = definitions.add("ObjectType", type_name)
        object_type.add("Count", number)
        if template:
            _properties(object_type.add("PropertyTemplate", template), entries)


def write_fbx(path: str | os.PathLike[str], *, name: str, positions: np.ndarray, normals: np.ndarray,
              tangents: np.ndarray, uvs: np.ndarray, indices: np.ndarray, material: FbxMaterial,
              embed_textures: bool = True, map_files: Mapping[str, str] | None = None) -> None:
    """Write one triangle mesh with one material as a binary FBX 7.4 file.

    positions (n, 3) metres in the glTF frame (+Y up); normals (n, 3); tangents (n, 4): MikkTSpace
    xyz and w = +-1; uvs (n, 2) v-up; indices (m, 3) triangles into those n vertices.
    ``material``: see ``FbxMaterial``. Each map is embedded in the file when ``embed_textures``
    is true. Either way, FileName and RelativeFilename point at ``map_files[map]``, relative to the
    FBX, with a default of ``maps/<map>.png`` (maps: basecolor, normal, roughness, metallic, emissive).
    """
    files = {key: f"maps/{key}.png" for key, _, _ in TEXTURE_SLOTS} | dict(map_files or {})
    write_fbx_materials(path, name=name, positions=positions, normals=normals, tangents=tangents, uvs=uvs,
                        indices=indices, materials=[material], map_files=[files], embed_textures=embed_textures)


def write_fbx_materials(path: str | os.PathLike[str], *, name: str, positions: np.ndarray, normals: np.ndarray,
                        tangents: np.ndarray | None, uvs: np.ndarray, indices: np.ndarray,
                        materials: Sequence[FbxMaterial], map_files: Sequence[Mapping[str, str]],
                        face_materials: np.ndarray | None = None, embed_textures: bool = True) -> None:
    """Write one triangle mesh with one or more materials as a binary FBX 7.4 file.

    As write_fbx, except: ``tangents`` may be None (a model without a normal map: no tangent space is
    written); ``materials`` lists the materials and ``face_materials`` (m,) gives each triangle's index
    into it (None: every triangle has the first); ``map_files[i][map]`` is the file name of material i's
    map, relative to the FBX, for every map it has. With several materials, a map is called
    "<material name>_<map>" in the file.
    """
    positions = _checked(positions, "positions", 3).astype(np.float32)
    rows = positions.shape[0]
    normals = _checked(normals, "normals", 3, rows)
    tangents = None if tangents is None else _checked(tangents, "tangents", 4, rows)
    uvs = _checked(uvs, "uvs", 2, rows)
    indices = _checked(indices, "indices", 3)
    if indices.dtype.kind not in "iu" or not indices.size or indices.min() < 0 or indices.max() >= rows:
        raise ValueError(f"indices must be integer triangles into the {rows} vertices")
    if not materials or len(map_files) != len(materials):
        raise ValueError("every material needs its map file names")
    if len(materials) == 1:
        face_materials = None
    elif face_materials is not None:
        face_materials = np.asarray(face_materials)
        if face_materials.shape != (len(indices),) or face_materials.dtype.kind not in "iu" \
                or face_materials.min() < 0 or face_materials.max() >= len(materials):
            raise ValueError(f"face_materials must give each of the {len(indices)} triangles one of the "
                             f"{len(materials)} materials")
    else:
        face_materials = np.zeros(len(indices), np.int32)
    ids = map(_Int64, count(1_000_001))

    with ThreadPoolExecutor(max_workers=max(1, min(8, os.cpu_count() or 1))) as pool:
        token = _POOL.set(pool)
        try:
            root = _Node("")
            _header(root, Path(path).name)
            documents = root.add("Documents")
            documents.add("Count", 1)
            document = documents.add("Document", next(ids), "Scene", "Scene")
            _properties(document, [("SourceObject", "object", None), ("ActiveAnimStackName", "string", "")])
            document.add("RootNode", _Int64(0))
            root.add("References")

            objects = _Node("Objects")
            geometry_id, model_id = next(ids), next(ids)
            _geometry(objects, geometry_id, name, positions, normals, tangents, uvs, indices.astype(np.int64),
                      face_materials)
            model = objects.add("Model", model_id, _name(name, "Model"), "Mesh")
            model.add("Version", 232)
            _properties(model, [("Lcl Translation", "translation", (0.0, 0.0, 0.0)),
                                ("Lcl Rotation", "rotation", (0.0, 0.0, 0.0)),
                                ("Lcl Scaling", "scaling", (1.0, 1.0, 1.0)),
                                ("DefaultAttributeIndex", "int", 0), ("InheritType", "enum", 1)])
            model.add("MultiLayer", 0)
            model.add("MultiTake", 0)
            model.add("Shading", True)
            model.add("Culling", "CullingOff")

            # The order of the materials' connections to the model is what LayerElementMaterial indexes.
            material_links: list[tuple[Prop, ...]] = []
            texture_links: list[tuple[Prop, ...]] = []
            textures = videos = 0
            for material, files in zip(materials, map_files):
                material_id = next(ids)
                material_name = material.get("name", name)
                fbx_material = objects.add("Material", material_id, _name(material_name, "Material"), "")
                fbx_material.add("Version", 102)
                fbx_material.add("ShadingModel", "Phong")
                fbx_material.add("MultiLayer", 0)
                _properties(fbx_material, _material_properties(material))
                material_links.append(("OO", material_id, model_id))
                prefix = "" if len(materials) == 1 else f"{material_name}_"
                uses_alpha = material.get("alpha_mode", "OPAQUE") != "OPAQUE"
                for key, material_key, binding in TEXTURE_SLOTS:
                    png = material.get(material_key)
                    if not png:
                        continue
                    file = files[key]
                    video_id = next(ids)
                    video = objects.add("Video", video_id, _name(prefix + key, "Video"), "Clip")
                    video.add("Type", "Clip")
                    _properties(video, [("Path", "xref_url", file)])
                    video.add("UseMipMap", 0)
                    video.add("Filename", file)
                    video.add("RelativeFilename", file)
                    if embed_textures:
                        video.add("Content", bytes(png))
                    videos += 1
                    bindings = [(prefix + key, binding)]
                    if key == "basecolor" and uses_alpha:
                        bindings.append((f"{prefix}{key}_alpha", "TransparentColor"))
                    for texture_name, prop in bindings:
                        texture_id = next(ids)
                        texture = objects.add("Texture", texture_id, _name(texture_name, "Texture"), "")
                        texture.add("Type", "TextureVideoClip")
                        texture.add("Version", 202)
                        texture.add("TextureName", _name(texture_name, "Texture"))
                        _properties(texture, [("UseMaterial", "bool", 1)])
                        texture.add("Media", _name(prefix + key, "Video"))
                        texture.add("FileName", file)
                        texture.add("RelativeFilename", file)
                        texture_links += [("OO", video_id, texture_id), ("OP", texture_id, material_id, prop)]
                        textures += 1

            _definitions(root, len(materials), textures, videos)
            root.children.append(objects)
            connections = root.add("Connections")
            for link in [("OO", model_id, _Int64(0)), ("OO", geometry_id, model_id), *material_links,
                         *texture_links]:
                connections.add("C", *link)
            root.add("Takes").add("Current", "")
            data = _file_bytes(root)
        finally:
            _POOL.reset(token)
    Path(path).write_bytes(data)


def _file_bytes(root: _Node) -> bytes:
    """The whole file: the header, the root's records, then the footer (see the module doc)."""
    out = [_MAGIC, struct.pack("<I", FBX_VERSION)]
    offset = len(_MAGIC) + 4
    for node in root.children:
        offset = node.write(out, offset)
    offset += len(_NULL_RECORD) + len(_FOOTER_ID) + 4
    out += [_NULL_RECORD, _FOOTER_ID, bytes(4), bytes(16 - offset % 16),
            struct.pack("<I", FBX_VERSION), bytes(120), _FOOTER_MAGIC]
    return b"".join(out)


# ---------------------------------------------------------------------- reading a file back


def parse_fbx(data: bytes) -> _Node:
    """A binary FBX 7.4 file (32-bit record offsets, as this module writes them) as a root record whose
    children are the file's top-level records; every property stays encoded, so _file_bytes gives back
    the same bytes for a file of ours. Raises ValueError for anything else."""
    if data[:len(_MAGIC)] != _MAGIC or len(data) < len(_MAGIC) + 4 \
            or struct.unpack_from("<I", data, len(_MAGIC))[0] != FBX_VERSION:
        raise ValueError("this is not a binary FBX 7.4 file")
    root = _Node.raw(b"", [])
    root.children, _ = _parse_records(data, len(_MAGIC) + 4, len(data))
    return root


def _parse_records(data: bytes, at: int, stop: int) -> tuple[list[_Node], int]:
    """The records from ``at`` up to the null record that closes their list (before ``stop``), and the
    offset after it."""
    nodes: list[_Node] = []
    while True:
        if at + len(_NULL_RECORD) > stop:
            raise ValueError("the FBX file is damaged (a record list is not closed)")
        end, count, length, name_length = struct.unpack_from("<3IB", data, at)
        if end == 0:
            if data[at:at + len(_NULL_RECORD)] != _NULL_RECORD:
                raise ValueError("the FBX file is damaged (a broken null record)")
            return nodes, at + len(_NULL_RECORD)
        at += 13
        name = data[at:at + name_length]
        at += name_length
        props_end = at + length
        if end > stop or props_end > end:
            raise ValueError("the FBX file is damaged (a record runs past its end)")
        props = []
        for _ in range(count):
            size = _property_size(data, at, props_end)
            props.append(data[at:at + size])
            at += size
        if at != props_end:
            raise ValueError("the FBX file is damaged (the property bytes do not add up)")
        node = _Node.raw(name, props)
        if at < end:
            node.children, at = _parse_records(data, at, end)
        if at != end:
            raise ValueError("the FBX file is damaged (a record's end offset is wrong)")
        nodes.append(node)


def _property_size(data: bytes, at: int, stop: int) -> int:
    """The bytes of the property at ``at`` (its type byte and its payload), which must end by ``stop``."""
    code = data[at] if at < stop else -1
    if code in _SCALAR_BYTES:
        size = 1 + _SCALAR_BYTES[code]
    elif code in _BLOB_CODES and at + 5 <= stop:
        size = 5 + struct.unpack_from("<I", data, at + 1)[0]
    elif code in _ARRAY_CODES and at + 13 <= stop:
        size = 13 + struct.unpack_from("<I", data, at + 9)[0]
    else:
        raise ValueError("the FBX file is damaged (a property cannot be read)")
    if at + size > stop:
        raise ValueError("the FBX file is damaged (a property runs past its record)")
    return size


def set_embedded_map(data: bytes, image: bytes, key: str = "basecolor") -> bytes | None:
    """The FBX ``data`` (written here) with the file embedded for its map ``key`` replaced by ``image``:
    the Content of the Video called ``key`` ("basecolor"; with several materials "<material>_basecolor").
    Every texture showing that Video shows the new file. None when the map is not embedded (written with
    embed_textures=False); ValueError when the file has no such map."""
    root = parse_fbx(data)
    wanted = _encode(_name(key, "Video"))
    videos = [node for objects in root.children if objects.name == b"Objects"
              for node in objects.children if node.name == b"Video" and node.props[1:2] == [wanted]]
    if not videos:
        raise ValueError(f"the FBX has no map called {key}")
    content = next((child for child in videos[0].children if child.name == b"Content"), None)
    if content is None or content.props in ([], [b"R" + bytes(4)]):
        return None
    content.props = [_encode(bytes(image))]
    return _file_bytes(root)


def replace_embedded_map(path: str | os.PathLike[str], image: bytes, key: str = "basecolor") -> bool:
    """set_embedded_map on the file at ``path``, written back atomically -> whether it was (False: the map
    is not embedded, the file is left alone)."""
    path = Path(path)
    data = set_embedded_map(path.read_bytes(), image, key)
    if data is None:
        return False
    replace_files({path: data})
    return True
