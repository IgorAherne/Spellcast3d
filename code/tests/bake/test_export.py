"""Saving a 3D model as FBX and OBJ for the artist (spellbake.convert, launchers/export_worker.py),
and the artist's painting on a bake (export_worker.py's repaint).

The model is a small GLB written here with two materials: one with every glTF map (base colour with
alpha, normal, ORM with occlusion, a glow brighter than 1) and factors that are not 1, one with plain
factors. Readers judge the saved files: Spellcast3D's own importer (instant_meshes_brush, which reads
FBX through Assimp and its embedded maps itself), trimesh for the OBJ, and Blender's importers (a dev
test that skips when Blender is not installed). A real Pixal3D model from this PC checks the speed.
The paintings go into real bakes of a box onto itself (512 px, under a second each).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import time
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import export_worker
from spellbake import BakeError, glb
from spellbake.bake import bake
from spellbake.convert import convert
from spellbake.sampling import CLAMP_TO_EDGE, linear_to_srgb8, srgb_to_linear, unorm8
from spellbake.settings import BakeSettings
from test_fbx import _contents
from test_glb import image_bytes, parse, validate

BLENDER_CHECK = Path(__file__).parent / "blender" / "check_export.py"
E2E_MESH = Path(os.environ.get("SPELLCAST_E2E_PROP", r"C:\sc3d\spellcast3d\projects\E2E Props\props\p_025d38")) \
    / "mesh" / "mesh.glb"
SIZE = 32


def _png(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, "PNG")
    return buffer.getvalue()


def _read(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def _box() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A unit box, a UV island per side: (positions, normals, v-up uv, faces). Faces 0-5 are the +x/+y/+z
    sides, 6-11 the others."""
    positions, normals, uvs, faces = [], [], [], []
    for side, (axis, sign) in enumerate((a, s) for s in (1.0, -1.0) for a in range(3)):
        u_axis, v_axis = (axis + 1) % 3, (axis + 2) % 3
        base = len(positions)
        for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
            p = np.zeros(3)
            p[axis], p[u_axis], p[v_axis] = 0.5 * sign, (u - 0.5) * sign, v - 0.5
            n = np.zeros(3)
            n[axis] = sign
            positions.append(p + (0.2, 1.0, -0.3))
            normals.append(n)
            uvs.append(((side % 3) / 3 + 0.3 * u, (side // 3) / 2 + 0.45 * v))
        faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return np.array(positions), np.array(normals), np.array(uvs), np.array(faces)


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    rng = np.random.default_rng(7)
    folder = tmp_path_factory.mktemp("model")
    maps = {
        "basecolor": rng.integers(0, 256, (SIZE, SIZE, 4), np.uint8),
        "orm": rng.integers(0, 256, (SIZE, SIZE, 3), np.uint8),
        "emissive": rng.integers(0, 256, (SIZE, SIZE, 3), np.uint8),
    }
    tilt = rng.uniform(-0.4, 0.4, (SIZE, SIZE, 2))
    normal = np.concatenate([tilt, np.sqrt(1.0 - (tilt ** 2).sum(-1, keepdims=True))], -1)
    maps["normal"] = np.round((normal + 1.0) * 127.5).astype(np.uint8)
    orm = glb.Texture(_png(maps["orm"]), "orm", CLAMP_TO_EDGE, CLAMP_TO_EDGE)
    textured = glb.Material(
        name="Painted wood", base_color_factor=(0.5, 1.0, 1.0, 0.8), metallic_factor=0.5, roughness_factor=1.0,
        emissive_factor=(2.0, 2.0, 2.0), base_color=glb.Texture(_png(maps["basecolor"]), "basecolor"),
        metallic_roughness=orm, occlusion=orm, normal=glb.Texture(_png(maps["normal"]), "normal"),
        emissive=glb.Texture(_png(maps["emissive"]), "emissive"), alpha_mode="BLEND")
    plain = glb.Material(name="Iron", base_color_factor=(0.2, 0.4, 0.6, 1.0), metallic_factor=0.3,
                         roughness_factor=0.7)
    positions, normals, uvs, faces = _box()
    glb.write_glb(folder / "model.glb", [glb.Primitive(positions, normals, faces[:6], uvs, material=0),
                                         glb.Primitive(positions, normals, faces[6:], uvs, material=1)],
                  [textured, plain], name="box")
    out = folder / "out"
    result = convert(folder / "model.glb", {"fbx": out / "fbx", "obj": out / "obj"}, name="Crate_high")
    return {"folder": folder, "out": out, "result": result, "maps": maps, "positions": positions,
            "faces": faces}


def test_the_files_and_the_answer(model: dict[str, Any]) -> None:
    result, out = model["result"], model["out"]
    assert result["fbx"] == {"files": ["Crate_high.fbx"]}
    maps = ["basecolor", "opacity", "roughness", "metallic", "ao", "normal", "emissive"]
    assert sorted(result["obj"]["files"]) == sorted(["Crate_high.obj", "Crate_high.mtl",
                                                     *(f"Crate_high_Painted_wood_{m}.png" for m in maps)])
    assert result["stats"]["triangles"] == 12 and result["stats"]["materials"] == 2
    assert sorted(p.name for p in (out / "obj").iterdir()) == sorted(result["obj"]["files"])


def test_glTF_factors_are_folded_into_the_maps(model: dict[str, Any]) -> None:
    obj, source = model["out"] / "obj", model["maps"]
    colour = _read(obj / "Crate_high_Painted_wood_basecolor.png")
    assert colour.shape == (SIZE, SIZE, 4)
    expected = linear_to_srgb8(srgb_to_linear(source["basecolor"][..., :3]) * np.array([0.5, 1.0, 1.0]))
    assert np.array_equal(colour[..., :3], expected)                        # in linear light
    assert np.array_equal(colour[..., 3], unorm8(source["basecolor"][..., 3] / 255.0 * 0.8))
    assert np.array_equal(_read(obj / "Crate_high_Painted_wood_opacity.png"), colour[..., 3])
    assert np.array_equal(_read(obj / "Crate_high_Painted_wood_roughness.png"), source["orm"][..., 1])
    assert np.array_equal(_read(obj / "Crate_high_Painted_wood_metallic.png"), unorm8(source["orm"][..., 2] / 510.0))
    assert np.array_equal(_read(obj / "Crate_high_Painted_wood_ao.png"), source["orm"][..., 0])
    assert np.array_equal(_read(obj / "Crate_high_Painted_wood_normal.png"), source["normal"])
    # A glow of 2: the map keeps its colour, the MTL (and the FBX) multiply it by 2.
    assert np.array_equal(_read(obj / "Crate_high_Painted_wood_emissive.png"), source["emissive"])


def test_the_mtl_names_every_map_and_the_plain_values(model: dict[str, Any]) -> None:
    mtl = (model["out"] / "obj" / "Crate_high.mtl").read_text(encoding="utf-8")
    textured, plain = mtl.split("newmtl ")[1:]
    assert textured.splitlines()[0] == "Painted_wood" and plain.splitlines()[0] == "Iron"
    for line in ("Kd 1 1 1", "map_Kd Crate_high_Painted_wood_basecolor.png",
                 "map_Bump -bm 1 Crate_high_Painted_wood_normal.png", "norm Crate_high_Painted_wood_normal.png",
                 "map_Pr Crate_high_Painted_wood_roughness.png", "map_Pm Crate_high_Painted_wood_metallic.png",
                 "Ke 2 2 2", "map_Ke Crate_high_Painted_wood_emissive.png", "map_d Crate_high_Painted_wood_opacity.png"):
        assert line in textured.splitlines(), line
    assert plain.splitlines()[1:] == ["Kd 0.2 0.4 0.6", "Ks 0 0 0", "illum 2", "Pr 0.7", "Pm 0.3"]
    text = (model["out"] / "obj" / "Crate_high.obj").read_text(encoding="utf-8")
    assert text.splitlines()[1] == "mtllib Crate_high.mtl"
    assert text.count("usemtl Painted_wood") == 1 and text.count("usemtl Iron") == 1


def test_the_obj_loads_in_trimesh(model: dict[str, Any]) -> None:
    import trimesh

    scene = trimesh.load(model["out"] / "obj" / "Crate_high.obj", force="scene", process=False)
    meshes = list(scene.geometry.values())
    assert sum(len(m.faces) for m in meshes) == 12 and len(meshes) == 2
    bounds = np.array([m.bounds for m in meshes])
    assert np.allclose([bounds[:, 0].min(0), bounds[:, 1].max(0)],
                       [model["positions"].min(0), model["positions"].max(0)], atol=1e-6)
    textured = next(m for m in meshes if getattr(m.visual.material, "name", "") == "Painted_wood")
    assert textured.visual.uv is not None and len(textured.visual.uv) == len(textured.vertices)
    assert textured.visual.material.image is not None and textured.visual.material.image.size == (SIZE, SIZE)


@pytest.mark.parametrize("fmt", ["fbx", "obj"])
def test_spellcast3ds_own_importer_reads_it_back(model: dict[str, Any], fmt: str) -> None:
    from instant_meshes_brush.assets import load_source

    source = load_source(model["out"] / fmt / f"Crate_high.{fmt}")
    assert len(source.faces) == 12 and len(source.materials) == 2
    assert np.allclose(source.vertices.min(0), model["positions"].min(0), atol=1e-5)
    assert np.allclose(source.vertices.max(0), model["positions"].max(0), atol=1e-5)
    textured = next(m for m in source.materials if m.name == "Painted_wood")
    canonical = textured.canonical(None)
    assert canonical.basecolor is not None and canonical.normal is not None and canonical.emissive is not None
    assert {"roughness", "metallic"} <= canonical.orm_sources
    assert np.array_equal(canonical.normal, model["maps"]["normal"])     # tangent space, OpenGL, unchanged
    plain = next(m for m in source.materials if m.name == "Iron")
    assert plain.base_color_factor[:3] == pytest.approx((0.2, 0.4, 0.6), abs=1e-3)


def _blender() -> str | None:
    installed = sorted(glob(r"C:\Program Files\Blender Foundation\Blender*\blender.exe"), reverse=True)
    candidates = [os.environ.get("SPELLCAST_BLENDER"), shutil.which("blender"), *installed]
    return next((c for c in candidates if c and Path(c).is_file()), None)


@pytest.mark.parametrize("fmt", ["fbx", "obj"])
def test_blender_imports_it_at_its_place_with_every_map_wired(model: dict[str, Any], fmt: str) -> None:
    """Dev check: Blender's importers put the saved model where its glTF importer puts the GLB, with the
    maps on the Principled BSDF (packed in the FBX, beside the OBJ)."""
    blender = _blender()
    if blender is None:
        pytest.skip("Blender is not installed (set SPELLCAST_BLENDER to its blender.exe)")
    report = model["out"] / f"blender_{fmt}.json"
    run = subprocess.run([blender, "--background", "--factory-startup", "--python", str(BLENDER_CHECK), "--",
                          str(model["out"] / fmt / f"Crate_high.{fmt}"), str(model["folder"] / "model.glb"),
                          str(report)], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0 and report.exists(), run.stdout[-2000:] + run.stderr[-2000:]
    result = json.loads(report.read_text(encoding="utf-8"))
    source, saved = result["source"], result["model"]
    assert saved["triangles"] == source["triangles"] == 12
    assert saved["min"] == pytest.approx(source["min"], abs=1e-5)
    assert saved["max"] == pytest.approx(source["max"], abs=1e-5)
    assert all(layers == 1 for layers in saved["uv_layers"])
    textured = saved["materials"]["Painted_wood"]
    for socket in ("Base Color", "Normal", "Roughness", "Metallic", "Emission Color"):
        image = textured[socket]
        assert image is not None and tuple(image["size"]) == (SIZE, SIZE), socket
        assert image["packed"] is (fmt == "fbx"), socket
    assert "Iron" in saved["materials"]


def _request(tmp_path: Path, src: Path, **changes: Any) -> Path:
    request = {"cmd": "convert", "src": str(src), "name": "Crate_high",
               "outputs": {"fbx": str(tmp_path / "cache" / "fbx"), "obj": str(tmp_path / "cache" / "obj")},
               **changes}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def _lines(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_the_worker_replaces_each_folder_whole(model: dict[str, Any], tmp_path: Path) -> None:
    old = tmp_path / "cache" / "fbx"
    old.mkdir(parents=True)
    (old / "Old_name.fbx").write_bytes(b"from an earlier save")
    stdout = io.StringIO()
    assert export_worker.main([str(_request(tmp_path, model["folder"] / "model.glb"))], stdout) == 0
    lines = _lines(stdout)
    assert lines[-1]["result"]["fbx"]["files"] == ["Crate_high.fbx"] and any("progress" in line for line in lines)
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == ["fbx", "obj"]      # no .tmp folders left
    assert [p.name for p in old.iterdir()] == ["Crate_high.fbx"]


def test_the_worker_reports_what_went_wrong(tmp_path: Path) -> None:
    (tmp_path / "broken.glb").write_bytes(b"glTF but not really")
    stdout = io.StringIO()
    assert export_worker.main([str(_request(tmp_path, tmp_path / "broken.glb"))], stdout) == 1
    answer = _lines(stdout)[-1]
    assert answer["kind"] == "user" and answer["error"].startswith("The 3D model broken.glb could not be read")
    assert not (tmp_path / "cache").exists() or not any((tmp_path / "cache").iterdir())
    stdout = io.StringIO()
    assert export_worker.main([str(_request(tmp_path, tmp_path / "broken.glb", outputs={"zip": "x"}))], stdout) == 1
    assert _lines(stdout)[-1]["kind"] == "internal"
    with pytest.raises(BakeError, match="could not be read"):
        convert(tmp_path / "missing.glb", {"obj": tmp_path / "o"}, name="x")


# ---- the artist's painting saved into a bake ---------------------------------------------------------

def _bake_box(folder: Path, alpha: bool) -> Path:
    """A real 512 px bake of _box onto itself: an opaque colour (GLB, FBX, OBJ), or one with alpha (BLEND:
    an RGBA base colour and an opacity map; GLB and OBJ only) -> the bake folder."""
    positions, normals, uvs, faces = _box()
    colour = np.random.default_rng(21 + alpha).integers(0, 256, (16, 16, 4 if alpha else 3), np.uint8)
    material = glb.Material("paint", base_color=glb.Texture(_png(colour)), alpha_mode="BLEND" if alpha else "OPAQUE")
    glb.write_glb(folder / "high.glb", [glb.Primitive(positions, normals, faces, uvs)], [material])
    lines = [f"v {x} {y} {z}" for x, y, z in positions] + [f"vt {u} {v}" for u, v in uvs]
    lines += ["f " + " ".join(f"{i + 1}/{i + 1}" for i in face) for face in faces]
    (folder / "low.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
    settings = BakeSettings(texture_size=512, quality=1, ao=False, repack_uvs=False, export_obj=True,
                            export_fbx=not alpha)
    return bake(folder / "high.glb", folder / "low.obj", folder / "bake_001", settings, threads=2).folder


def _repaint(bake_dir: Path, pixels: np.ndarray, **changes: Any) -> tuple[int, list[dict[str, Any]]]:
    raw = bake_dir.parent / "paint.rgba"
    pixels.astype(np.uint8).tofile(raw)
    request = {"cmd": "repaint", "dir": str(bake_dir), "rgba": str(raw), "width": pixels.shape[1],
               "height": pixels.shape[0], **changes}
    path = bake_dir.parent / "repaint.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    stdout = io.StringIO()
    code = export_worker.main([str(path)], stdout)
    return code, _lines(stdout)


def _files(folder: Path) -> dict[str, bytes]:
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def _read_map(path: Path) -> tuple[str, np.ndarray]:
    with Image.open(path) as image:
        return image.mode, np.asarray(image)


def test_the_worker_paints_a_bake(tmp_path: Path) -> None:
    bake_dir = _bake_box(tmp_path, alpha=False)
    before = _files(bake_dir)
    painted = np.random.default_rng(5).integers(0, 256, (512, 512, 4), np.uint8)
    painted[..., 3] = 255                                    # the page holds an RGB map opaque
    code, lines = _repaint(bake_dir, painted)
    assert code == 0, lines
    assert lines[-1]["result"] == {"files": ["maps/basecolor.png", "projection.glb", "projection.fbx"]}
    after = _files(bake_dir)
    assert after.keys() == before.keys()                     # no temporary file left
    assert {name for name in after if after[name] != before[name]} == {"maps/basecolor.png", "projection.glb",
                                                                        "projection.fbx"}
    mode, pixels = _read_map(bake_dir / "maps" / "basecolor.png")
    assert mode == "RGB" and np.array_equal(pixels, painted[..., :3])
    png = after["maps/basecolor.png"]
    document, binary = parse(bake_dir / "projection.glb")
    assert image_bytes(document, binary, document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"]) \
        == png
    assert _contents(bake_dir / "projection.fbx")["basecolor"] == png
    validate(bake_dir / "projection.glb")


def test_the_worker_keeps_the_alpha(tmp_path: Path) -> None:
    bake_dir = _bake_box(tmp_path, alpha=True)
    before = _files(bake_dir)
    old_mode, old = _read_map(bake_dir / "maps" / "basecolor.png")
    painted = np.random.default_rng(6).integers(0, 256, (512, 512, 4), np.uint8)
    painted[..., 3] = old[..., 3]                            # a painting never changes the alpha
    code, lines = _repaint(bake_dir, painted)
    assert code == 0, lines
    assert lines[-1]["result"] == {"files": ["maps/basecolor.png", "projection.glb"]}       # baked without FBX
    after = _files(bake_dir)
    assert {name for name in after if after[name] != before[name]} == {"maps/basecolor.png", "projection.glb"}
    mode, pixels = _read_map(bake_dir / "maps" / "basecolor.png")
    assert old_mode == mode == "RGBA" and np.array_equal(pixels, painted)
    document, binary = parse(bake_dir / "projection.glb")
    assert image_bytes(document, binary, document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"]) \
        == after["maps/basecolor.png"]
    validate(bake_dir / "projection.glb")


def test_the_worker_refuses_a_painting_not_made_on_the_bakes_colours(tmp_path: Path) -> None:
    # What a page sends after the graphics card was reset under it: strokes on transparent black.
    lost = np.zeros((512, 512, 4), np.uint8)
    lost[100:140, 100:300, :3] = (32, 19, 20)
    for alpha in (False, True):
        (tmp_path / f"alpha{alpha}").mkdir()
        bake_dir = _bake_box(tmp_path / f"alpha{alpha}", alpha=alpha)
        before = _files(bake_dir)
        code, lines = _repaint(bake_dir, lost, unpainted=str(bake_dir / "maps" / "basecolor.unpainted.png"))
        assert code == 1 and lines[-1]["kind"] == "user" and "not painted on this bake's colours" in lines[-1]["error"]
        assert _files(bake_dir) == before                    # nothing written, the bake's own map not kept either
    # An RGB map with a colour key: the page holds its keyed texels transparent, and so must a painting of it.
    _, old = _read_map(bake_dir / "maps" / "basecolor.png")
    old = old[..., :3]
    Image.fromarray(old).save(bake_dir / "maps" / "basecolor.png", transparency=tuple(int(c) for c in old[0, 0]))
    painted = np.concatenate([old, np.full((512, 512, 1), 255, np.uint8)], axis=-1)
    assert _repaint(bake_dir, painted)[0] == 1
    painted[..., 3] = np.where((old == old[0, 0]).all(axis=-1), 0, 255)
    assert _repaint(bake_dir, painted)[0] == 0


def test_the_first_painting_keeps_the_bakes_own_map_once(tmp_path: Path) -> None:
    bake_dir = _bake_box(tmp_path, alpha=False)
    own = (bake_dir / "maps" / "basecolor.png").read_bytes()
    unpainted = bake_dir / "maps" / "basecolor.unpainted.png"
    painted = np.full((512, 512, 4), 255, np.uint8)
    painted[..., :3] = (200, 40, 40)
    code, lines = _repaint(bake_dir, painted, unpainted=str(unpainted))
    assert code == 0, lines
    assert lines[-1]["result"]["files"] == ["maps/basecolor.unpainted.png", "maps/basecolor.png", "projection.glb",
                                            "projection.fbx"]
    assert unpainted.read_bytes() == own                     # byte for byte, put back to take the painting off
    painted[..., :3] = (40, 200, 40)
    code, lines = _repaint(bake_dir, painted, unpainted=str(unpainted))
    assert code == 0 and "maps/basecolor.unpainted.png" not in lines[-1]["result"]["files"]
    assert unpainted.read_bytes() == own                     # one copy, never replaced
    assert not [p for p in bake_dir.rglob("*.tmp")]


def test_the_worker_refuses_a_painting_that_does_not_fit(tmp_path: Path) -> None:
    bake_dir = _bake_box(tmp_path, alpha=False)
    before = _files(bake_dir)
    code, lines = _repaint(bake_dir, np.zeros((256, 512, 4)))
    assert code == 1 and lines[-1]["kind"] == "user"
    assert "512 x 256 pixels" in lines[-1]["error"] and "512 x 512" in lines[-1]["error"]
    code, lines = _repaint(bake_dir, np.zeros((511, 512, 4)), height=512)
    assert code == 1 and lines[-1] == {"error": "The painting arrived incomplete, so it was not saved. Save it again.",
                                       "kind": "user"}
    code, lines = _repaint(bake_dir, np.zeros((512, 512, 4)), dir=str(tmp_path / "nowhere"))
    assert code == 1 and lines[-1]["kind"] == "user" and "missing" in lines[-1]["error"]
    code, lines = _repaint(bake_dir, np.zeros((512, 512, 4)), width=True)
    assert code == 1 and lines[-1]["kind"] == "internal" and "Saving the painting failed" in lines[-1]["error"]
    assert _files(bake_dir) == before


@pytest.mark.realasset
@pytest.mark.slow
def test_a_pixal3d_model_is_saved_in_seconds(tmp_path: Path) -> None:
    if not E2E_MESH.is_file():
        pytest.skip(f"real asset not on this PC: {E2E_MESH}")
    src = tmp_path / "mesh.glb"
    shutil.copyfile(E2E_MESH, src)
    started = time.perf_counter()
    result = convert(src, {"fbx": tmp_path / "fbx", "obj": tmp_path / "obj"}, name="Wooden_barrel_high")
    seconds = time.perf_counter() - started
    assert result["stats"]["triangles"] > 200_000 and result["warnings"] == []
    assert seconds < 60, seconds            # about 3 s on the development PC
    assert "map_Kd Wooden_barrel_high_basecolor.png" in (tmp_path / "obj" / "Wooden_barrel_high.mtl").read_text()
