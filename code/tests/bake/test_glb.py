"""The GLB writer (spellbake.glb): what it stores reads back exactly, and the Khronos validator
passes its files with no errors and no warnings (a dev check that skips without Node)."""

from __future__ import annotations

import io
import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from spellbake import glb
from spellbake.sampling import CLAMP_TO_EDGE, MIRRORED_REPEAT, REPEAT

NODE_DIR = Path(__file__).resolve().parent / "node"
_TYPES = {5125: np.uint32, 5126: np.float32}
_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def encoded(image: np.ndarray, kind: str) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, kind)
    return buffer.getvalue()


def parse(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    magic, version, total = struct.unpack_from("<III", data, 0)
    assert (magic, version, total) == (0x46546C67, 2, len(data))
    json_length, json_kind = struct.unpack_from("<II", data, 12)
    assert json_kind == 0x4E4F534A and json_length % 4 == 0
    document = json.loads(data[20:20 + json_length])
    bin_length, bin_kind = struct.unpack_from("<II", data, 20 + json_length)
    assert bin_kind == 0x004E4942 and bin_length % 4 == 0
    return document, data[28 + json_length:28 + json_length + bin_length]


def accessor(document: dict[str, Any], binary: bytes, index: int) -> np.ndarray:
    entry = document["accessors"][index]
    view = document["bufferViews"][entry["bufferView"]]
    dtype = np.dtype(_TYPES[entry["componentType"]])
    assert view["byteOffset"] % dtype.itemsize == 0
    count, width = entry["count"], _WIDTHS[entry["type"]]
    start = view["byteOffset"] + entry.get("byteOffset", 0)
    return np.frombuffer(binary, dtype, count * width, start).reshape(count, width)


def image_bytes(document: dict[str, Any], binary: bytes, texture: int) -> bytes:
    view = document["bufferViews"][document["images"][document["textures"][texture]["source"]]["bufferView"]]
    return binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]


def two_materials() -> tuple[list[glb.Primitive], list[glb.Material], dict[str, bytes]]:
    """A quad in two primitives: a baked-style material (ORM as both metallicRoughness and
    occlusion, clamped) and an imported-style one (JPEG colour that tiles, emissive above 1)."""
    rng = np.random.default_rng(3)
    images = {"png": encoded(rng.integers(0, 256, (8, 8, 3), np.uint8), "PNG"),
              "orm": encoded(rng.integers(0, 256, (8, 8, 3), np.uint8), "PNG"),
              "jpeg": encoded(rng.integers(0, 256, (16, 16, 3), np.uint8), "JPEG")}
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    normals = np.tile([0.0, 0.0, 2.0], (4, 1))        # written unit length
    uv = positions[:, :2] * 0.8 + 0.1
    tangents = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, -1.0], [1.0, 0.0, 0.0, -1.0]])
    orm = glb.Texture(images["orm"], "orm")
    primitives = [glb.Primitive(positions, normals, np.array([[0, 1, 2]]), uv=uv, tangents=tangents, material=0),
                  glb.Primitive(positions, normals, np.array([[0, 2, 3]]), uv=uv, material=1)]
    materials = [
        glb.Material("baked", base_color=glb.Texture(images["png"], "basecolor"), metallic_roughness=orm,
                     occlusion=orm, normal=glb.Texture(images["png"], "normal"), alpha_mode="MASK",
                     alpha_cutoff=0.25, extras={"spellcast": {"tangent_basis": "mikk", "version": 1}}),
        glb.Material("imported", base_color_factor=(0.5, 0.25, 1.0, 0.75), metallic_factor=0.2,
                     roughness_factor=0.6, emissive_factor=(4.0, 2.0, 0.0), occlusion_strength=0.5,
                     base_color=glb.Texture(images["jpeg"], "colour", REPEAT, MIRRORED_REPEAT),
                     emissive=glb.Texture(images["png"], "glow", REPEAT, MIRRORED_REPEAT),
                     alpha_mode="BLEND", double_sided=True),
    ]
    return primitives, materials, images


def test_round_trip(tmp_path: Path) -> None:
    primitives, materials, images = two_materials()
    path = tmp_path / "two.glb"
    size = glb.write_glb(path, primitives, materials, name="quad")
    assert size == path.stat().st_size
    document, binary = parse(path)
    assert document["asset"]["generator"] == glb.GENERATOR
    assert document["nodes"] == [{"mesh": 0, "name": "quad"}]
    first, second = document["meshes"][0]["primitives"]
    assert (first["material"], second["material"]) == (0, 1)

    position = accessor(document, binary, first["attributes"]["POSITION"])
    np.testing.assert_array_equal(position, primitives[0].positions.astype(np.float32))
    assert document["accessors"][first["attributes"]["POSITION"]]["min"] == [0.0, 0.0, 0.0]
    np.testing.assert_array_equal(accessor(document, binary, first["attributes"]["NORMAL"]),
                                  np.tile(np.float32([0, 0, 1]), (4, 1)))
    np.testing.assert_array_equal(accessor(document, binary, first["attributes"]["TANGENT"]),
                                  primitives[0].tangents.astype(np.float32))
    texcoord = accessor(document, binary, first["attributes"]["TEXCOORD_0"])
    np.testing.assert_allclose(texcoord, np.stack([primitives[0].uv[:, 0], 1 - primitives[0].uv[:, 1]], 1),
                               atol=1e-7)
    assert "TANGENT" not in second["attributes"]
    np.testing.assert_array_equal(accessor(document, binary, first["indices"]).reshape(-1), [0, 1, 2])
    assert document["accessors"][second["indices"]]["componentType"] == 5125

    baked, imported = document["materials"]
    pbr = baked["pbrMetallicRoughness"]
    # One ORM image serves metallicRoughness and occlusion; identical PNGs are stored once.
    assert pbr["metallicRoughnessTexture"]["index"] == baked["occlusionTexture"]["index"]
    assert len(document["images"]) == 3
    assert image_bytes(document, binary, pbr["baseColorTexture"]["index"]) == images["png"]
    assert image_bytes(document, binary, baked["occlusionTexture"]["index"]) == images["orm"]
    assert (baked["alphaMode"], baked["alphaCutoff"]) == ("MASK", 0.25)
    assert baked["extras"] == {"spellcast": {"tangent_basis": "mikk", "version": 1}}
    assert "doubleSided" not in baked and "emissiveFactor" not in baked

    colour = imported["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    assert image_bytes(document, binary, colour) == images["jpeg"]
    assert document["images"][document["textures"][colour]["source"]]["mimeType"] == "image/jpeg"
    sampler = document["samplers"][document["textures"][colour]["sampler"]]
    assert sampler == {"magFilter": 9729, "minFilter": 9987, "wrapS": REPEAT, "wrapT": MIRRORED_REPEAT}
    clamped = document["samplers"][document["textures"][pbr["baseColorTexture"]["index"]]["sampler"]]
    assert (clamped["wrapS"], clamped["wrapT"]) == (CLAMP_TO_EDGE, CLAMP_TO_EDGE)
    assert imported["pbrMetallicRoughness"]["baseColorFactor"] == [0.5, 0.25, 1.0, 0.75]
    assert imported["emissiveFactor"] == [1.0, 0.5, 0.0]
    assert imported["extensions"] == {"KHR_materials_emissive_strength": {"emissiveStrength": 4.0}}
    assert document["extensionsUsed"] == ["KHR_materials_emissive_strength"]
    assert (imported["alphaMode"], imported["doubleSided"]) == ("BLEND", True)
    assert "alphaCutoff" not in imported and "occlusionTexture" not in imported


def test_bad_input_is_refused(tmp_path: Path) -> None:
    primitives, materials, _ = two_materials()
    path = tmp_path / "bad.glb"
    with pytest.raises(ValueError, match="faces"):
        glb.write_glb(path, [glb.Primitive(primitives[0].positions, primitives[0].normals, np.array([[0, 1, 4]]))],
                      materials)
    with pytest.raises(ValueError, match="material"):
        glb.write_glb(path, [glb.Primitive(primitives[0].positions, primitives[0].normals, np.array([[0, 1, 2]]),
                                           material=2)], materials)
    with pytest.raises(ValueError, match="PNG or JPEG"):
        glb.write_glb(path, primitives[:1], [glb.Material("x", base_color=glb.Texture(b"GIF89a"))])
    assert not path.exists()


def test_trimesh_reads_it(tmp_path: Path) -> None:
    """An independent reader agrees on the geometry and the v flip."""
    trimesh = pytest.importorskip("trimesh")
    primitives, materials, _ = two_materials()
    path = tmp_path / "two.glb"
    glb.write_glb(path, primitives, materials)
    scene = trimesh.load_scene(path, process=False)
    meshes = list(scene.geometry.values())
    assert sorted(len(m.faces) for m in meshes) == [1, 1]
    for mesh in meshes:
        np.testing.assert_allclose(mesh.visual.uv, primitives[0].uv, atol=1e-6)   # trimesh flips v back


@pytest.mark.skipif(shutil.which("node") is None or not (NODE_DIR / "node_modules").is_dir(),
                    reason="dev-only: needs Node and `npm ci` in code/tests/bake/node")
def test_khronos_validator(tmp_path: Path) -> None:
    primitives, materials, _ = two_materials()
    path = tmp_path / "two.glb"
    glb.write_glb(path, primitives, materials)
    done = subprocess.run(["node", str(NODE_DIR / "validate.mjs"), str(path)], capture_output=True, text=True,
                          cwd=NODE_DIR, check=False)
    assert done.returncode == 0, done.stdout + done.stderr


# ---- the base colour image swapped (the Projection stage's painting) ------------------------------

def baked_like(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    """A GLB as a bake writes it: the base colour image between the accessors and the ORM, normal and
    emissive images; its PNG is 8 x 8, so its length is unlikely to be a multiple of 4."""
    rng = np.random.default_rng(11)
    images = {key: encoded(rng.integers(0, 256, (8, 8, 3), np.uint8), "PNG")
              for key in ("basecolor", "orm", "normal", "emissive")}
    primitive = two_materials()[0][0]
    orm = glb.Texture(images["orm"], "orm")
    material = glb.Material("projection", base_color=glb.Texture(images["basecolor"], "basecolor"),
                            metallic_roughness=orm, occlusion=orm, normal=glb.Texture(images["normal"], "normal"),
                            emissive=glb.Texture(images["emissive"], "emissive"), emissive_factor=(2.0, 2.0, 2.0),
                            extras={"spellcast": {"tangent_basis": "mikk", "version": 1}})
    path = tmp_path / "projection.glb"
    glb.write_glb(path, [primitive], [material], name="projection")
    return path, images


def views_aligned(document: dict[str, Any], binary: bytes) -> None:
    assert len(binary) % 4 == 0 and document["buffers"] == [{"byteLength": len(binary)}]
    for view in document["bufferViews"]:
        offset = view.get("byteOffset", 0)
        assert offset % 4 == 0 and offset + view["byteLength"] <= len(binary)


def validate(*paths: Path) -> None:
    """The Khronos validator's verdict, where Node and its node_modules are installed (a dev check)."""
    if shutil.which("node") is None or not (NODE_DIR / "node_modules").is_dir():
        return
    done = subprocess.run(["node", str(NODE_DIR / "validate.mjs"), *map(str, paths)], capture_output=True, text=True,
                          cwd=NODE_DIR, check=False)
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_base_colour_is_swapped_in_place(tmp_path: Path) -> None:
    path, images = baked_like(tmp_path)
    original = path.read_bytes()
    before, old_binary = parse(path)
    painted = encoded(np.random.default_rng(12).integers(0, 256, (16, 16, 3), np.uint8), "PNG")
    assert glb.replace_base_color_image(path, painted) == path.stat().st_size
    assert sorted(p.name for p in tmp_path.iterdir()) == ["projection.glb"]        # no temporary file left
    after, binary = parse(path)
    views_aligned(after, binary)

    # The JSON is the same but for where the views after the image lie.
    material = after["materials"][0]
    base = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    assert {k: v for k, v in after.items() if k not in ("bufferViews", "buffers")} == \
        {k: v for k, v in before.items() if k not in ("bufferViews", "buffers")}
    image_view = after["images"][after["textures"][base]["source"]]["bufferView"]
    start = before["bufferViews"][image_view]["byteOffset"]
    shift = {view["byteOffset"] - old["byteOffset"] for old, view in zip(before["bufferViews"], after["bufferViews"])
             if old["byteOffset"] > start}
    assert len(shift) == 1 and shift.pop() % 4 == 0
    assert all(view == old for old, view in zip(before["bufferViews"], after["bufferViews"]) if old["byteOffset"] < start)

    # The new picture where the old one was; every other image and every accessor reads as before.
    assert image_bytes(after, binary, base) == painted
    for slot in ("metallicRoughnessTexture", "normalTexture", "occlusionTexture", "emissiveTexture"):
        info = material.get(slot) or material["pbrMetallicRoughness"][slot]
        assert image_bytes(after, binary, info["index"]) == image_bytes(before, old_binary, info["index"])
    for index in range(len(before["accessors"])):
        assert accessor(after, binary, index).tobytes() == accessor(before, old_binary, index).tobytes()
    validate(path)

    # Painting again swaps in place again: putting the first picture back gives the very first file.
    glb.replace_base_color_image(path, images["basecolor"])
    assert path.read_bytes() == original


def test_a_base_colour_that_shares_its_image_gets_its_own(tmp_path: Path) -> None:
    """two_materials' PNG is the baked base colour, its normal map (the same texture) and the imported
    material's glow (another sampler): the base colour alone changes."""
    primitives, materials, images = two_materials()
    path = tmp_path / "two.glb"
    glb.write_glb(path, primitives, materials)
    before, old_binary = parse(path)
    painted = encoded(np.random.default_rng(13).integers(0, 256, (8, 8, 3), np.uint8), "PNG")
    glb.replace_base_color_image(path, painted)
    after, binary = parse(path)
    views_aligned(after, binary)
    assert after["bufferViews"][:len(before["bufferViews"])] == before["bufferViews"]   # nothing moved
    assert binary[:len(old_binary)] == old_binary

    baked, imported = after["materials"]
    base = baked["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    assert image_bytes(after, binary, base) == painted
    assert image_bytes(after, binary, baked["normalTexture"]["index"]) == images["png"]
    assert image_bytes(after, binary, imported["emissiveTexture"]["index"]) == images["png"]
    assert image_bytes(after, binary, imported["pbrMetallicRoughness"]["baseColorTexture"]["index"]) == images["jpeg"]
    assert len(after["images"]) == 4 and len(after["textures"]) == len(before["textures"]) + 1
    old_texture = before["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    assert after["textures"][base]["sampler"] == before["textures"][old_texture]["sampler"]
    assert after["images"][-1] == {"bufferView": len(after["bufferViews"]) - 1, "mimeType": "image/png",
                                   "name": "basecolor"}
    validate(path)

    # Its own image from now on: the next painting goes in place.
    again = encoded(np.random.default_rng(14).integers(0, 256, (4, 4, 3), np.uint8), "PNG")
    glb.replace_base_color_image(path, again)
    document, binary = parse(path)
    assert len(document["images"]) == 4 and image_bytes(document, binary, base) == again
    assert image_bytes(document, binary, baked["normalTexture"]["index"]) == images["png"]
    validate(path)


def test_another_writers_glb(tmp_path: Path) -> None:
    """Pretty-printed JSON, a JPEG between two accessors, an unaligned image and a buffer shorter than its
    chunk (uint16 indices last): the accessors read as before and the image becomes a PNG."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32).tobytes()
    uv = np.array([[0, 0], [1, 0], [0, 1]], np.float32).tobytes()
    jpeg = encoded(np.full((4, 4, 3), 99, np.uint8), "JPEG")
    jpeg += bytes((1 - len(jpeg)) % 4)                  # its length one past a multiple of 4
    indices = np.array([0, 1, 2], np.uint16).tobytes()
    image_at = len(positions) + len(uv)
    indices_at = image_at + len(jpeg) + (-len(jpeg) % 4)
    binary = positions + uv + jpeg + bytes(-len(jpeg) % 4) + indices + bytes(2)
    document = {
        "asset": {"version": "2.0", "generator": "someone else"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "textures": [{"source": 0}], "images": [{"bufferView": 3, "mimeType": "image/jpeg"}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0],
                       "max": [1, 1, 0]},
                      {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
                      {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}],
        "bufferViews": [{"buffer": 0, "byteLength": len(positions)},
                        {"buffer": 0, "byteOffset": len(positions), "byteLength": len(uv)},
                        {"buffer": 0, "byteOffset": indices_at, "byteLength": len(indices)},
                        {"buffer": 0, "byteOffset": image_at, "byteLength": len(jpeg)}],
        "buffers": [{"byteLength": len(binary) - 2}],
    }
    text = json.dumps(document, indent=2).encode("utf-8")
    text += b" " * (-len(text) % 4)
    path = tmp_path / "other.glb"
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 28 + len(text) + len(binary))
                     + struct.pack("<II", len(text), 0x4E4F534A) + text
                     + struct.pack("<II", len(binary), 0x004E4942) + binary)
    painted = encoded(np.random.default_rng(15).integers(0, 256, (8, 8, 3), np.uint8), "PNG")
    glb.replace_base_color_image(path, painted)
    after, binary = parse(path)
    views_aligned(after, binary)
    assert image_bytes(after, binary, 0) == painted and after["images"] == [{"bufferView": 3, "mimeType": "image/png"}]

    def read(view: int) -> bytes:
        at = after["bufferViews"][view].get("byteOffset", 0)
        return binary[at:at + after["bufferViews"][view]["byteLength"]]

    assert (read(0), read(1), read(2)) == (positions, uv, indices)
    assert after["bufferViews"][2]["byteOffset"] != indices_at and "byteOffset" not in after["bufferViews"][0]
    assert after["asset"] == {"version": "2.0", "generator": "someone else"}


def test_a_glb_without_a_base_colour_image_is_refused(tmp_path: Path) -> None:
    primitives, _, images = two_materials()
    path = tmp_path / "plain.glb"
    glb.write_glb(path, primitives[:1], [glb.Material("plain", normal=glb.Texture(images["png"]))])
    original = path.read_bytes()
    with pytest.raises(ValueError, match="no base colour image"):
        glb.replace_base_color_image(path, images["png"])
    with pytest.raises(ValueError, match="no base colour image"):
        glb.replace_base_color_image(path, images["png"], material=1)
    with pytest.raises(ValueError, match="PNG or JPEG"):
        glb.set_base_color_image(original, b"GIF89a")
    with pytest.raises(ValueError, match="not a glTF 2.0 binary"):
        glb.set_base_color_image(original[:-4], images["png"])
    assert path.read_bytes() == original


def webp_glb(shared: bool) -> bytes:
    """A GLB as Pixal3D writes it: its textures only through EXT_texture_webp (no `source` of their own)."""
    picture = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 200, 30)).save(picture, "WEBP")
    image = picture.getvalue()
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32).tobytes()
    binary = positions + image + bytes(-len(image) % 4)
    textures = [{"extensions": {"EXT_texture_webp": {"source": 0}}}]
    pbr: dict[str, Any] = {"baseColorTexture": {"index": 0}}
    if shared:
        pbr["metallicRoughnessTexture"] = {"index": 0}
    document = {
        "asset": {"version": "2.0"}, "extensionsUsed": ["EXT_texture_webp"],
        "extensionsRequired": ["EXT_texture_webp"], "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}], "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": pbr}], "textures": textures,
        "images": [{"bufferView": 1, "mimeType": "image/webp"}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                       "min": [0, 0, 0], "max": [1, 1, 0]}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36},
                        {"buffer": 0, "byteOffset": 36, "byteLength": len(image)}],
        "buffers": [{"byteLength": len(binary)}],
    }
    text = json.dumps(document).encode()
    text += b" " * (-len(text) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)
    return b"".join([struct.pack("<III", 0x46546C67, 2, total), struct.pack("<II", len(text), 0x4E4F534A), text,
                     struct.pack("<II", len(binary), 0x004E4942), binary])


@pytest.mark.parametrize("shared", [False, True])
def test_a_webp_base_colour_is_swapped(shared: bool) -> None:
    picture = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 20, 20)).save(picture, "WEBP")
    new = picture.getvalue()
    out = glb.set_base_color_image(webp_glb(shared), new)
    document, binary = glb._read_glb(out)[:2]
    base = document["textures"][document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
    assert "source" not in base, "the texture keeps showing its image through the extension"
    source = glb.texture_source(base)
    view = document["bufferViews"][document["images"][source]["bufferView"]]
    assert binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]] == new
    assert document["images"][source]["mimeType"] == "image/webp"
    if shared:   # the metallic-roughness texture keeps the old picture
        other = document["textures"][document["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"]]
        assert glb.texture_source(other) != source
    with pytest.raises(ValueError):
        glb.image_mime(new)                              # outside a WebP GLB, WebP is no glTF image


# ---- a GLB moved and turned (a prop's placement, for the export folder) -----------------------------

QUARTER = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])     # about Z: +X to +Y, +Y to -X
MOVE = np.array([0.5, 2.0, -1.0])


def test_a_glb_is_placed_in_its_vertices(tmp_path: Path) -> None:
    """glb.placed: positions, normals and tangents hold R p + t (the normals and tangents R alone, a tangent's w
    kept), the positions' min / max follow, and every other byte of meaning stays - the images, the UVs, the indices."""
    path, images = baked_like(tmp_path)
    before, binary_before = parse(path)
    placed = tmp_path / "placed.glb"
    placed.write_bytes(glb.placed(path.read_bytes(), QUARTER, MOVE))
    document, binary = parse(placed)
    views_aligned(document, binary)
    validate(placed)
    attributes = document["meshes"][0]["primitives"][0]["attributes"]
    old = {key: accessor(before, binary_before, index) for key, index in attributes.items()}
    new = {key: accessor(document, binary, index) for key, index in attributes.items()}
    np.testing.assert_allclose(new["POSITION"], old["POSITION"] @ QUARTER.T + MOVE, atol=1e-6)
    np.testing.assert_allclose(new["NORMAL"], old["NORMAL"] @ QUARTER.T, atol=1e-6)
    np.testing.assert_allclose(new["TANGENT"][:, :3], old["TANGENT"][:, :3] @ QUARTER.T, atol=1e-6)
    assert np.array_equal(new["TANGENT"][:, 3], old["TANGENT"][:, 3]) and np.array_equal(new["TEXCOORD_0"], old["TEXCOORD_0"])
    entry = document["accessors"][attributes["POSITION"]]
    np.testing.assert_allclose([entry["min"], entry["max"]], [new["POSITION"].min(0), new["POSITION"].max(0)])
    assert all("matrix" not in node and "translation" not in node for node in document["nodes"])
    for texture in range(len(document["textures"])):
        assert image_bytes(document, binary, texture) == image_bytes(before, binary_before, texture)
    # Nothing but the three accessors' numbers and the positions' bounds changed.
    entry.update(min=before["accessors"][attributes["POSITION"]]["min"], max=before["accessors"][attributes["POSITION"]]["max"])
    assert document == before
    # A placement that moves nothing gives the model back, number for number.
    same, same_binary = parse_bytes(glb.placed(path.read_bytes(), np.eye(3), np.zeros(3)))
    assert same == before and same_binary == binary_before


def parse_bytes(data: bytes) -> tuple[dict[str, Any], bytes]:
    json_length = struct.unpack_from("<I", data, 12)[0]
    bin_length = struct.unpack_from("<I", data, 20 + json_length)[0]
    return json.loads(data[20:20 + json_length]), data[28 + json_length:28 + json_length + bin_length]


def test_a_glb_that_is_not_plain_carries_its_placement_in_a_node(tmp_path: Path) -> None:
    """A GLB whose nodes have transforms of their own (another writer's) keeps its vertices: a node over the scene's
    nodes holds the placement, which every reader applies - the hub's own (the FBX and OBJ made from it) too."""
    path, _images = baked_like(tmp_path)
    document, binary = parse(path)
    document["nodes"][0]["translation"] = [0.0, 0.0, 3.0]
    text = json.dumps(document).encode()
    text += b" " * (-len(text) % 4)
    other = struct.pack("<III", 0x46546C67, 2, 28 + len(text) + len(binary)) + struct.pack("<II", len(text), 0x4E4F534A) \
        + text + struct.pack("<II", len(binary), 0x004E4942) + binary
    wrapped, wrapped_binary = parse_bytes(glb.placed(other, QUARTER, MOVE))
    assert wrapped_binary == binary                                           # not a vertex touched
    top = wrapped["nodes"][wrapped["scenes"][0]["nodes"][0]]
    assert top["children"] == [0] and wrapped["nodes"][0]["translation"] == [0.0, 0.0, 3.0]
    matrix = np.array(top["matrix"]).reshape(4, 4).T                          # glTF writes it column by column
    np.testing.assert_allclose(matrix[:3, :3], QUARTER)
    np.testing.assert_allclose(matrix[:3, 3], MOVE)
    out = tmp_path / "wrapped.glb"
    out.write_bytes(glb.placed(other, QUARTER, MOVE))
    validate(out)


def test_a_model_is_converted_where_it_was_placed(tmp_path: Path) -> None:
    """convert(placement=...): the FBX and the OBJ hold the model turned and moved, and a GLB of it may be asked for
    beside them; placement_of reads the hub's quaternion."""
    from instant_meshes_brush.assets import load_source
    from spellbake.convert import convert, placement_of

    rotation, move = placement_of({"move": MOVE.tolist(), "turn": [0.0, 0.0, 0.70710678, 0.70710678]})
    np.testing.assert_allclose(rotation, QUARTER, atol=1e-7)
    np.testing.assert_allclose(move, MOVE)
    assert placement_of(None) is None
    for bad in ({"move": [0, 0, 0]}, {"move": [0, 0, 0], "turn": [0, 0, 0, 0]}, {"move": [0, "x", 0], "turn": [0, 0, 0, 1]}):
        with pytest.raises(ValueError):
            placement_of(bad)
    path, _images = baked_like(tmp_path)
    made = load_source(path)
    # The triangle's own three points (an FBX leaves out a point no triangle uses).
    used = np.asarray(made.vertices, np.float64)[np.unique(np.asarray(made.faces))]
    expected = np.unique((used @ QUARTER.T + MOVE).round(5), axis=0)
    result = convert(path, {fmt: tmp_path / fmt for fmt in ("glb", "fbx", "obj")}, name="Crate", placement=(rotation, move))
    assert result["glb"]["files"] == ["Crate.glb"] and result["fbx"]["files"] == ["Crate.fbx"]
    for file in (tmp_path / "glb" / "Crate.glb", tmp_path / "fbx" / "Crate.fbx", tmp_path / "obj" / "Crate.obj"):
        source = load_source(file)
        points = np.asarray(source.vertices, np.float64)[np.unique(np.asarray(source.faces))]
        np.testing.assert_allclose(np.unique(points.round(5), axis=0), expected, atol=2e-5, err_msg=file.name)
        np.testing.assert_allclose(np.asarray(load_source(file).normals, np.float64)[0], [0.0, 0.0, 1.0], atol=1e-6,
                                   err_msg=file.name)                         # turned about Z: it still looks along +Z
    with pytest.raises(ValueError):
        convert(path, {"glb": tmp_path / "plain"}, name="Crate")              # as made, the GLB is the model's own file
