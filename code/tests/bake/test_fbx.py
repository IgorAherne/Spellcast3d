"""FBX export of the baked low poly (spellbake.fbx).

Four readers judge the file. A strict record walker checks the binary layout. assimp checks
geometry, attributes and material slots. instant_meshes_brush.fbx_media (Spellcast3D's own
"use my own model" import) checks the embedded maps. Blender's importer checks true size, axes,
UVs, normals, hard edges, the Principled BSDF images and re-export; it is a dev test that skips
when Blender is not installed.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import shutil
import struct
import subprocess
import zlib
from collections.abc import Mapping
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from spellbake.fbx import (CM_PER_METRE, FBX_VERSION, TEXTURE_SLOTS, _file_bytes, parse_fbx, replace_embedded_map,
                           set_embedded_map, write_fbx, write_fbx_materials)

BLENDER_CHECK = Path(__file__).parent / "blender" / "check_fbx.py"
HARD_BOX_EDGES = 12
Node = tuple[str, list[Any], list["Node"]]


def _png(mode: str, size: tuple[int, int], seed: int) -> bytes:
    pixels = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], len(mode)), np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels.squeeze(-1) if mode == "L" else pixels, mode).save(buffer, "PNG")
    return buffer.getvalue()


def _box(size: tuple[float, float, float], centre: tuple[float, float, float]) -> tuple[np.ndarray, ...]:
    """Six UV islands with face normals (every box edge hard); the three negative faces have mirrored UVs (w = -1)."""
    positions, normals, tangents, uvs, faces = [], [], [], [], []
    for island, (axis_u, axis_v, axis_n, sign) in enumerate(
            (u, v, n, s) for u, v, n in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) for s in (1.0, -1.0)):
        base = len(positions)
        for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):   # counter-clockwise seen from outside
            p, n, t = np.zeros(3), np.zeros(3), np.zeros(4)
            p[axis_n], p[axis_u], p[axis_v] = 0.5 * sign, (u - 0.5) * sign, v - 0.5
            n[axis_n], t[axis_u], t[3] = sign, 1.0, sign   # +axis_u is where the (possibly mirrored) u grows
            positions.append(p)
            normals.append(n)
            tangents.append(t)
            u_image = u if sign > 0 else 1 - u
            uvs.append((0.5 + 0.25 * (island % 2) + 0.2 * u_image, 0.3 * (island // 2) + 0.25 * v))
        faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return (np.array(positions) * size + centre, np.array(normals), np.array(tangents), np.array(uvs),
            np.array(faces))


def _sphere(radii: tuple[float, float, float], centre: tuple[float, float, float]) -> tuple[np.ndarray, ...]:
    """A smooth lat-long ellipsoid: a UV seam whose two sides share positions and normals, welded poles.

    u grows eastwards and v northwards, which seen from outside is mirrored, hence w = -1.
    """
    stacks, slices = 6, 10
    theta = np.linspace(0.0, np.pi, stacks + 1)[:, None]
    phi = np.linspace(0.0, 2.0 * np.pi, slices + 1)[None, :]
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_t[[0, -1]] = 0.0                              # exact poles, so they weld
    unit = np.stack(np.broadcast_arrays(sin_t * np.cos(phi), cos_t, sin_t * np.sin(phi)), -1)
    unit[:, -1] = unit[:, 0]                          # exact seam
    tangent = np.stack(np.broadcast_arrays(-np.sin(phi), 0.0 * theta, np.cos(phi), -1.0 + 0.0 * theta), -1)
    tangent[:, -1] = tangent[:, 0]
    uv = np.stack(np.broadcast_arrays(0.05 + 0.4 * phi / (2 * np.pi), 0.45 * (1.0 - theta / np.pi)), -1)
    grid = np.arange(unit.shape[0] * unit.shape[1]).reshape(unit.shape[:2])
    a, b, c, d = grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]
    first, second = np.stack([a, c, d], -1)[:-1], np.stack([a, b, c], -1)[1:]   # drop the pole-degenerate halves
    faces = np.concatenate([first.reshape(-1, 3), second.reshape(-1, 3)])
    return (unit.reshape(-1, 3) * radii + centre, unit.reshape(-1, 3), tangent.reshape(-1, 4), uv.reshape(-1, 2),
            faces)


@pytest.fixture(scope="module")
def mesh() -> dict[str, np.ndarray]:
    sphere = _sphere((0.3, 0.2, 0.25), (-0.4, 0.5, 0.1))
    box = _box((0.2, 0.6, 0.35), (0.5, 0.3, -0.6))
    offset = len(sphere[0])
    parts = zip(sphere, box)
    positions, normals, tangents, uvs, faces = (np.concatenate(pair) for pair in parts)
    faces[len(sphere[4]):] += offset
    return {"positions": positions.astype(np.float32), "normals": normals.astype(np.float32),
            "tangents": tangents.astype(np.float32), "uvs": uvs.astype(np.float32),
            "indices": faces.astype(np.uint32)}


@pytest.fixture(scope="module")
def material() -> dict[str, Any]:
    return {"name": "baked_material", "base_color_png": _png("RGBA", (32, 16), 1),
            "normal_png": _png("RGB", (16, 16), 2), "roughness_png": _png("L", (8, 16), 3),
            "metallic_png": _png("L", (16, 8), 4),
            "emissive_png": _png("RGB", (8, 8), 5), "ao_png": _png("L", (4, 4), 6),
            "base_color_factor": (0.8, 0.6, 0.4, 0.5), "metallic_factor": 0.25, "roughness_factor": 0.75,
            "emissive_factor": (1.0, 1.0, 1.0), "alpha_mode": "BLEND", "double_sided": False}


@pytest.fixture(scope="module")
def fbx(tmp_path_factory: pytest.TempPathFactory, mesh: dict[str, np.ndarray], material: dict[str, Any]) -> Path:
    """The mesh plus two triangles the writer must leave out, so every reader has to see exactly ``mesh``.

    One triangle's corners weld to two points (vertices 0 and 1 are both the sphere's north pole). The
    other is the first triangle again, reversed. With either kind left in, Blender 5.1 crashed importing a
    real Instant Meshes low poly; with them in this file, its FBX re-export fails.
    """
    path = tmp_path_factory.mktemp("fbx") / "projection.fbx"
    dropped = np.array([[0, 1, 20], mesh["indices"][0][[1, 0, 2]]], np.uint32)
    write_fbx(path, name="p_test", material=material, **{**mesh, "indices": np.concatenate([mesh["indices"], dropped])})
    return path


# --- a strict reader of the binary record layout ---------------------------------------------

def _property(data: bytes, at: int) -> tuple[Any, int]:
    code = chr(data[at])
    at += 1
    scalars = {"C": "<B", "I": "<i", "L": "<q", "D": "<d"}
    if code in scalars:
        return struct.unpack_from(scalars[code], data, at)[0], at + struct.calcsize(scalars[code])
    if code in "SR":
        (length,) = struct.unpack_from("<I", data, at)
        raw = data[at + 4:at + 4 + length]
        return (raw.decode("utf-8") if code == "S" else raw), at + 4 + length
    assert code in "di", f"unexpected property type {code!r}"
    count, encoding, length = struct.unpack_from("<3I", data, at)
    payload = data[at + 12:at + 12 + length]
    raw_size = count * (8 if code == "d" else 4)
    assert encoding == int(raw_size > 128), "arrays over 128 bytes are deflated, smaller ones raw"
    values = np.frombuffer(zlib.decompress(payload) if encoding else payload, "<f8" if code == "d" else "<i4").copy()
    assert values.size == count and values.nbytes == raw_size
    return values, at + 12 + length


def _records(data: bytes, at: int, stop: int) -> tuple[list[Node], int]:
    """Records up to and including the null record that must close the list before ``stop``."""
    nodes: list[Node] = []
    while True:
        assert at + 13 <= stop, "a record list is missing its null record"
        end, count, length, name_length = struct.unpack_from("<3IB", data, at)
        if end == 0:
            assert data[at:at + 13] == bytes(13)
            return nodes, at + 13
        name = data[at + 13:at + 13 + name_length].decode("ascii")
        at += 13 + name_length
        props_end, props = at + length, []
        for _ in range(count):
            value, at = _property(data, at)
            props.append(value)
        assert at == props_end, f"{name}: property bytes"
        children: list[Node] = []
        if at < end:
            children, at = _records(data, at, end)
        else:
            assert count, f"{name}: an empty record needs a null record"
        assert at == end, f"{name}: end offset"
        nodes.append((name, props, children))


def _parse(path: Path) -> list[Node]:
    data = path.read_bytes()
    assert data[:27] == b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", FBX_VERSION)
    nodes, at = _records(data, 27, len(data))
    footer = data[at:]
    padding = 16 - (at + 20) % 16
    assert footer[:16] == bytes.fromhex("fabcab09d0c8d466b176fb831cf7267e")
    assert footer[16:20 + padding] == bytes(4 + padding)
    assert footer[20 + padding:] == struct.pack("<I", FBX_VERSION) + bytes(120) + bytes.fromhex(
        "f85a8c6adef5d97eece90ce3758f290b")
    return nodes


def _child(node: Node, name: str) -> Node:
    return next(child for child in node[2] if child[0] == name)


def _props70(node: Node) -> dict[str, list[Any]]:
    return {p[1][0]: p[1][4:] for p in _child(node, "Properties70")[2]}


# --- tests ----------------------------------------------------------------------------------

def test_binary_layout_and_scene(fbx: Path, mesh: dict[str, np.ndarray], material: dict[str, Any]) -> None:
    nodes = {node[0]: node for node in _parse(fbx)}
    assert list(nodes) == ["FBXHeaderExtension", "FileId", "CreationTime", "Creator", "GlobalSettings", "Documents",
                           "References", "Definitions", "Objects", "Connections", "Takes"]
    settings = _props70(nodes["GlobalSettings"])
    assert {key: settings[key][0] for key in ("UpAxis", "UpAxisSign", "FrontAxis", "FrontAxisSign", "CoordAxis",
                                              "CoordAxisSign", "UnitScaleFactor")} == {
        "UpAxis": 1, "UpAxisSign": 1, "FrontAxis": 2, "FrontAxisSign": 1, "CoordAxis": 0, "CoordAxisSign": 1,
        "UnitScaleFactor": 1.0}

    objects = nodes["Objects"][2]
    kinds = [obj[0] for obj in objects]
    defined = {d[1][0]: _child(d, "Count")[1][0] for d in nodes["Definitions"][2] if d[0] == "ObjectType"}
    assert defined == {"GlobalSettings": 1, **{kind: kinds.count(kind) for kind in set(kinds)}}
    assert kinds.count("Video") == 5 and kinds.count("Texture") == 6       # + the base colour's TransparentColor

    geometry = next(obj for obj in objects if obj[0] == "Geometry")
    vertices = _child(geometry, "Vertices")[1][0].reshape(-1, 3) / CM_PER_METRE
    polygon = _child(geometry, "PolygonVertexIndex")[1][0].reshape(-1, 3)
    assert len(polygon) == len(mesh["indices"]) and (polygon[:, :2] >= 0).all() and (polygon[:, 2] < 0).all()
    polygon[:, 2] = ~polygon[:, 2]
    corners = mesh["indices"].reshape(-1)
    assert len(vertices) == len(np.unique(mesh["positions"][corners], axis=0))
    np.testing.assert_allclose(vertices[polygon.reshape(-1)], mesh["positions"][corners], atol=1e-7)
    smoothing = _child(_child(geometry, "LayerElementSmoothing"), "Smoothing")[1][0]
    assert len(smoothing) == len(_child(geometry, "Edges")[1][0]) and (smoothing == 0).sum() == HARD_BOX_EDGES

    ids = {obj[1][0]: obj for obj in objects}
    bindings: dict[str, bytes] = {}
    for link in nodes["Connections"][2]:
        kind, child, parent, *prop = link[1]
        if kind == "OP":
            video = next(ids[c[1][1]] for c in nodes["Connections"][2] if c[1][2] == child)
            bindings[prop[0]] = _child(video, "Content")[1][0]
    expected = {binding: material[key] for _, key, binding in TEXTURE_SLOTS}
    assert bindings == {**expected, "TransparentColor": material["base_color_png"]}
    phong = _props70(next(obj for obj in objects if obj[0] == "Material"))
    assert phong["DiffuseColor"] == pytest.approx([0.8, 0.6, 0.4]) and phong["Opacity"] == pytest.approx([0.5])
    assert phong["ReflectionFactor"] == pytest.approx([0.25]) and phong["Shininess"] == pytest.approx([6.25])


def test_assimp_round_trip(fbx: Path, mesh: dict[str, np.ndarray]) -> None:
    assimp = pytest.importorskip("assimp_py")
    scene = assimp.import_file(str(fbx), assimp.Process_Triangulate)
    assert scene.num_meshes == 1 and scene.num_materials == 1
    read = scene.meshes[0]
    faces = np.array(read.indices, np.int64).reshape(-1, 3)
    corners, loaded = mesh["indices"].reshape(-1), faces.reshape(-1)
    assert len(faces) == len(mesh["indices"])

    def per_corner(values: Any, width: int) -> np.ndarray:
        return np.array(values, np.float64).reshape(-1, width)[loaded]

    np.testing.assert_allclose(per_corner(read.vertices, 3) / CM_PER_METRE, mesh["positions"][corners], atol=1e-6)
    np.testing.assert_allclose(per_corner(read.normals, 3), mesh["normals"][corners], atol=1e-5)
    np.testing.assert_allclose(per_corner(read.texcoords[0], read.num_uv_components[0])[:, :2], mesh["uvs"][corners],
                               atol=1e-5)
    tangents = mesh["tangents"][corners]
    np.testing.assert_allclose(per_corner(read.tangents, 3), tangents[:, :3], atol=1e-5)
    np.testing.assert_allclose(per_corner(read.bitangents, 3),
                               tangents[:, 3:] * np.cross(mesh["normals"][corners], tangents[:, :3]), atol=1e-5)
    joined = assimp.import_file(str(fbx), assimp.Process_Triangulate | assimp.Process_JoinIdenticalVertices)
    assert joined.meshes[0].num_vertices == len(np.unique(corners))

    textures = scene.materials[0]["TEXTURES"]
    # aiTextureType: 1 diffuse, 4 emissive, 6 normals, 7 shininess, 8 opacity, 15 metalness
    assert textures == {1: ["maps/basecolor.png"], 4: ["maps/emissive.png"], 6: ["maps/normal.png"],
                        7: ["maps/roughness.png"], 8: ["maps/basecolor.png"], 15: ["maps/metallic.png"]}


def _all_bytes(value: Any) -> list[bytes]:
    """Every bytes value inside a reader's result, whatever containers it uses."""
    if isinstance(value, (bytes, bytearray)):
        return [bytes(value)]
    if isinstance(value, Mapping):
        items = list(value.values())
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    elif dataclasses.is_dataclass(value):
        items = [getattr(value, field.name) for field in dataclasses.fields(value)]
    elif hasattr(value, "__dict__"):
        items = list(vars(value).values())
    else:
        return []
    return [found for item in items for found in _all_bytes(item)]


def test_spellcast_fbx_reader_finds_the_embedded_maps(fbx: Path, material: dict[str, Any]) -> None:
    fbx_media = pytest.importorskip("instant_meshes_brush.fbx_media")
    found = _all_bytes(fbx_media.read(fbx))
    for _, key, _ in TEXTURE_SLOTS:
        assert material[key] in found, key


def test_external_maps_options_and_determinism(tmp_path: Path, mesh: dict[str, np.ndarray],
                                               material: dict[str, Any]) -> None:
    opaque = {**material, "alpha_mode": "OPAQUE"}
    first, second, external = (tmp_path / folder / "projection.fbx" for folder in ("a", "b", "external"))
    for path in (first, second, external):
        path.parent.mkdir()
    write_fbx(first, name="p", material=opaque, **mesh)
    write_fbx(second, name="p", material=opaque, **mesh)
    assert first.read_bytes() == second.read_bytes()
    write_fbx(external, name="p", material=opaque, embed_textures=False,
              map_files={"normal": "maps/normal_directx.png"}, **mesh)
    objects = next(node for node in _parse(external) if node[0] == "Objects")[2]
    videos = [obj for obj in objects if obj[0] == "Video"]
    assert all(not any(child[0] == "Content" for child in video[2]) for video in videos)
    assert sorted(_child(video, "RelativeFilename")[1][0] for video in videos) == [
        "maps/basecolor.png", "maps/emissive.png", "maps/metallic.png", "maps/normal_directx.png",
        "maps/roughness.png"]
    phong = _props70(next(obj for obj in objects if obj[0] == "Material"))
    assert phong["Opacity"] == [1.0] and phong["TransparencyFactor"] == [0.0]
    assert sum(obj[0] == "Texture" for obj in objects) == 5                 # no TransparentColor binding


# --- the base colour map swapped (the Projection stage's painting) -----------------------------

def _contents(path: Path) -> dict[str, bytes]:
    """{map name: its embedded file} of an FBX, read by the strict walker."""
    objects = next(node for node in _parse(path) if node[0] == "Objects")[2]
    return {video[1][1].split("\x00")[0]: _child(video, "Content")[1][0] for video in objects
            if video[0] == "Video" and any(child[0] == "Content" for child in video[2])}


def test_a_file_reads_back_byte_for_byte(fbx: Path, tmp_path: Path, mesh: dict[str, np.ndarray],
                                         material: dict[str, Any]) -> None:
    data = fbx.read_bytes()
    assert _file_bytes(parse_fbx(data)) == data
    external = tmp_path / "external.fbx"
    write_fbx(external, name="p", material=material, embed_textures=False, **mesh)
    assert _file_bytes(parse_fbx(external.read_bytes())) == external.read_bytes()
    for broken in (data[:-200], data[:5000], data[:23] + struct.pack("<I", 7500) + data[27:], b"glTF"):
        with pytest.raises(ValueError):
            parse_fbx(broken)


def test_the_base_colour_map_is_swapped(fbx: Path, tmp_path: Path, material: dict[str, Any]) -> None:
    path = tmp_path / "projection.fbx"
    shutil.copyfile(fbx, path)
    painted = _png("RGBA", (64, 32), 9)
    assert replace_embedded_map(path, painted) is True
    assert [p.name for p in tmp_path.iterdir()] == ["projection.fbx"]          # no temporary file left
    assert _contents(path) == {"basecolor": painted, **{key: material[slot] for key, slot, _ in TEXTURE_SLOTS
                                                        if key != "basecolor"}}
    nodes = {node[0]: node for node in _parse(path)}
    ids = {obj[1][0]: obj for obj in nodes["Objects"][2]}
    links = [link[1] for link in nodes["Connections"][2]]
    shown = {prop[0]: _child(ids[next(c[1] for c in links if c[2] == texture)], "Content")[1][0]
             for kind, texture, _material, *prop in links if kind == "OP"}
    assert shown["DiffuseColor"] == shown["TransparentColor"] == painted     # both textures show the one map
    data = path.read_bytes()
    assert _file_bytes(parse_fbx(data)) == data
    assert set_embedded_map(data, material["base_color_png"]) == fbx.read_bytes()     # the first picture back


def test_maps_that_are_not_embedded_are_left_alone(tmp_path: Path, mesh: dict[str, np.ndarray],
                                                   material: dict[str, Any]) -> None:
    path = tmp_path / "external.fbx"
    write_fbx(path, name="p", material=material, embed_textures=False, **mesh)
    before = path.stat().st_mtime_ns, path.read_bytes()
    assert replace_embedded_map(path, _png("RGB", (4, 4), 10)) is False
    assert (path.stat().st_mtime_ns, path.read_bytes()) == before
    with pytest.raises(ValueError, match="no map called glow"):
        set_embedded_map(before[1], _png("RGB", (4, 4), 10), "glow")


def test_a_map_of_one_of_several_materials(tmp_path: Path, mesh: dict[str, np.ndarray],
                                           material: dict[str, Any]) -> None:
    path = tmp_path / "two.fbx"
    write_fbx_materials(path, name="p", positions=mesh["positions"], normals=mesh["normals"],
                        tangents=mesh["tangents"], uvs=mesh["uvs"], indices=mesh["indices"],
                        materials=[{**material, "name": "wood"}, {**material, "name": "iron"}],
                        map_files=[{key: f"{name}_{key}.png" for key, _, _ in TEXTURE_SLOTS} for name in ("wood", "iron")],
                        face_materials=np.arange(len(mesh["indices"])) % 2)
    painted = _png("RGB", (16, 16), 11)
    path.write_bytes(set_embedded_map(path.read_bytes(), painted, "iron_basecolor"))
    contents = _contents(path)
    assert contents["iron_basecolor"] == painted and contents["wood_basecolor"] == material["base_color_png"]
    with pytest.raises(ValueError, match="no map called basecolor"):
        set_embedded_map(path.read_bytes(), painted)


@pytest.mark.parametrize(("factor", "colour", "strength"), [
    ((1.0, 1.0, 1.0), [1.0, 1.0, 1.0], 1.0),
    ((0.5, 0.25, 0.0), [0.5, 0.25, 0.0], 1.0),
    ((4.0, 2.0, 1.0), [1.0, 0.5, 0.25], 4.0),       # glTF emissive strength 4: the colour stays within 0..1
])
def test_emissive_factor_above_one_goes_to_emissive_factor(tmp_path: Path, mesh: dict[str, np.ndarray],
                                                           factor: tuple[float, ...], colour: list[float],
                                                           strength: float) -> None:
    path = tmp_path / "glow.fbx"
    write_fbx(path, name="p", material={"emissive_factor": factor}, **mesh)
    objects = next(node for node in _parse(path) if node[0] == "Objects")[2]
    phong = _props70(next(obj for obj in objects if obj[0] == "Material"))
    assert phong["EmissiveColor"] == pytest.approx(colour) and phong["EmissiveFactor"] == pytest.approx([strength])


def test_bad_input_is_refused(tmp_path: Path, mesh: dict[str, np.ndarray]) -> None:
    path = tmp_path / "bad.fbx"
    with pytest.raises(ValueError, match="tangents"):
        write_fbx(path, name="p", material={}, **{**mesh, "tangents": mesh["tangents"][:, :3]})
    with pytest.raises(ValueError, match="indices"):
        write_fbx(path, name="p", material={}, **{**mesh, "indices": mesh["indices"] + len(mesh["positions"])})
    with pytest.raises(ValueError, match="NaN"):
        write_fbx(path, name="p", material={}, **{**mesh, "normals": mesh["normals"] * np.nan})
    with pytest.raises(ValueError, match="three distinct corners"):
        write_fbx(path, name="p", material={}, **{**mesh, "indices": np.array([[0, 1, 2]], np.uint32)})  # pole
    assert not path.exists()


def _blender() -> str | None:
    installed = sorted(glob(r"C:\Program Files\Blender Foundation\Blender*\blender.exe"), reverse=True)
    candidates = [os.environ.get("SPELLCAST_BLENDER"), shutil.which("blender"), *installed]
    return next((c for c in candidates if c and Path(c).is_file()), None)


def test_blender_imports_true_size_uvs_normals_and_pbr_maps(fbx: Path, mesh: dict[str, np.ndarray],
                                                           material: dict[str, Any]) -> None:
    """Dev check: Blender's importer reads the FBX at its size in metres with every map wired."""
    blender = _blender()
    if blender is None:
        pytest.skip("Blender is not installed (set SPELLCAST_BLENDER to its blender.exe)")
    source, report = fbx.with_suffix(".npz"), fbx.with_suffix(".json")
    np.savez(source, **{key: mesh[key] for key in ("positions", "normals", "uvs", "indices")})
    run = subprocess.run([blender, "--background", "--factory-startup", "--python", str(BLENDER_CHECK), "--",
                          str(fbx), str(source), str(report)], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0 and report.exists(), run.stdout[-2000:] + run.stderr[-2000:]
    result = json.loads(report.read_text(encoding="utf-8"))

    used = mesh["positions"][mesh["indices"].reshape(-1)]
    assert result["mesh_objects"] == 1
    assert (result["vertices"], result["polygons"]) == (len(np.unique(used, axis=0)), len(mesh["indices"]))
    assert result["dimensions"] == pytest.approx(used.max(0) - used.min(0), abs=1e-5)      # metres
    assert result["world_position_error"] < 1e-5            # lands where Blender's glTF importer puts it
    assert result["uv_error"] < 1e-6 and result["normal_error_deg"] < 0.05
    assert result["sharp_edges"] == HARD_BOX_EDGES
    sizes = {name: Image.open(io.BytesIO(material[key])).size for name, key in (
        ("Base Color", "base_color_png"), ("Normal", "normal_png"), ("Roughness", "roughness_png"),
        ("Metallic", "metallic_png"), ("Emission Color", "emissive_png"), ("Alpha", "base_color_png"))}
    for socket, size in sizes.items():
        image = result["images"][socket]
        assert image is not None and image["packed"] and tuple(image["size"]) == size, socket
    assert {result["images"][s]["colorspace"] for s in ("Normal", "Roughness", "Metallic")} == {"Non-Color"}
    assert result["values"] == pytest.approx({"Roughness": 0.75, "Metallic": 0.25, "Alpha": 0.5})   # the factors
    assert result["reexport"] == ["FINISHED"]
