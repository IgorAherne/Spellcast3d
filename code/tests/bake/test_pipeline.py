"""The bake pipeline end to end (spellbake.bake, spellbake.importer, launchers/bake_worker.py).

test_analytic.py and test_policy_ao.py score the real baker against the analytic gates (planes
in 7 UV layouts, the displaced sphere with and without repacking, the texel-exact sphere, cage
policy, AO, height); this file adds what they do not reach:

* a repacked plane in both bases, through the same gates;
* glTF factor semantics on a two-material high poly (factors alone, factor x map, AO maps,
  alpha, emissive above 1), texel by texel;
* a high poly's own tangent-space normal map, decoded through ITS frame and re-encoded in the
  low poly's, and the DirectX copy;
* the worker's JSON-lines contract: a bake with GLB + FBX + OBJ, errors, cancelling, an import;
* the import of an OBJ + MTL into the canonical GLB;
* (slow, real asset) the speed and memory of the prop p_025d38 at 2048 px, quality 2.
"""

from __future__ import annotations

import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("instant_meshes_brush.uv", reason="the fixtures' low OBJs are written by instant_meshes_brush.uv")

from bakecheck import bakecheck as bc  # noqa: E402
from bakecheck import fixtures as fx  # noqa: E402
from spellbake import BakeError, glb  # noqa: E402
from spellbake.bake import BakeCancelled, bake  # noqa: E402
from spellbake.highpoly import load as load_high  # noqa: E402
from spellbake.sampling import linear_to_srgb8, srgb_to_linear, unorm8  # noqa: E402
from spellbake.settings import BakeSettings  # noqa: E402

WORKER = fx.LAUNCHERS / "bake_worker.py"
E2E_PROP = Path(os.environ.get("SPELLCAST_E2E_PROP", r"C:\sc3d\spellcast3d\projects\E2E Props\props\p_025d38"))
GIB = 1024 ** 3


@pytest.fixture(scope="module")
def root(request: pytest.FixtureRequest) -> Path:
    return fx.cache_root(request.config)


def png(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, "PNG")
    return buffer.getvalue()


def solid(colour: tuple[int, ...]) -> bytes:
    return png(np.tile(np.array(colour, np.uint8), (8, 8, 1)))


def read_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def write_low_plane(path: Path, lo: float = 0.05, hi: float = 0.95, n: int = 4, uv_span: float = 0.92) -> None:
    """A flat quad grid at z = 0 over [lo, hi]^2 whose UVs are (x, y) squeezed into
    [0.04, 0.04 + uv_span]^2."""
    lines = []
    ticks = np.linspace(lo, hi, n + 1)
    for y in ticks:
        for x in ticks:
            lines.append(f"v {x} {y} 0")
    for y in ticks:
        for x in ticks:
            lines.append(f"vt {0.04 + uv_span * (x - lo) / (hi - lo)} {0.04 + uv_span * (y - lo) / (hi - lo)}")
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i + 1
            quad = (a, a + 1, a + n + 2, a + n + 1)
            lines.append("f " + " ".join(f"{k}/{k}" for k in quad))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plane_part(x0: float, x1: float, uv_of: Any, material: int) -> glb.Primitive:
    """A flat 8 x 8 grid over [x0, x1] x [-0.1, 1.1] at z = 0 facing +z."""
    xs, ys = np.meshgrid(np.linspace(x0, x1, 9), np.linspace(-0.1, 1.1, 9))
    positions = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1)
    index = np.arange(xs.size).reshape(9, 9)
    a, b, c, d = index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    return glb.Primitive(positions, np.tile([0.0, 0.0, 1.0], (xs.size, 1)), faces, uv=uv_of(positions),
                         material=material)


def plain(**changes: Any) -> BakeSettings:
    """Settings for a quick bake of a hand-made layout: its UVs as they are, no AO, GLB only."""
    return BakeSettings(**{"texture_size": 512, "quality": 1, "ao": False, "repack_uvs": False, "export_obj": False,
                           "export_fbx": False, **changes})


def texel_mask(resolution: int, x0: float, x1: float, lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    """Texels of write_low_plane's map whose centre lies over world x in [x0, x1] (y well inside)."""
    centre = (np.arange(resolution) + 0.5) / resolution
    world = lo + (centre - 0.04) / 0.92 * (hi - lo)
    columns = (world >= x0) & (world <= x1)
    rows = (world[::-1] >= 0.2) & (world[::-1] <= 0.8)       # row 0 is v = 1
    return rows[:, None] & columns[None, :]


# ---------------------------------------------------------------------- settings


@pytest.mark.parametrize(("change", "words"), [
    ({"texture_size": 300}, "Texture size"),
    ({"quality": 4}, "Quality"),
    ({"quality": 2.5}, "whole number"),
    ({"ray_distance": 40}, "Ray distance"),
    ({"ray_distance": "auto"}, "number"),
    ({"hard_edge_angle": 200}, "Hard edge angle"),
    ({"tangent_basis": "blender"}, "Tangent basis"),
    ({"normal_y": "up"}, "OpenGL"),
    ({"ao": 1}, "on or off"),
    ({"ao_rays": 48}, "multiple of 32"),
    ({"ring": -1}, "ring"),
    ({"brightness": 2}, "Unknown bake setting"),
])
def test_settings_are_checked(change: dict[str, Any], words: str) -> None:
    with pytest.raises(BakeError, match=words):
        BakeSettings.from_json({"texture_size": 1024, **change})


def test_settings_from_json() -> None:
    settings = BakeSettings.from_json({"texture_size": 2048.0, "quality": 3, "ray_distance": 1, "hard_edge_angle": -1})
    assert (settings.texture_size, settings.supersample, settings.ray_distance) == (2048, 3, 1.0)
    assert isinstance(settings.texture_size, int) and settings.hard_edge_angle == -1.0
    assert BakeSettings.from_json(settings.to_json()) == settings
    assert (BakeSettings().export_obj, BakeSettings().export_fbx) == (True, True)


# ---------------------------------------------------------------------- the analytic gates, repacked


# lowpoly.repack pads islands by max(4, R // 256) texels. With 2 at 512 px, mip level 2 blended
# neighbouring islands (measured 33.5 LSB at borders against 0.9 inside); 4 measured 6.1 LSB.


@pytest.mark.slow
@pytest.mark.parametrize("size", [512, 1024])
@pytest.mark.parametrize("basis", bc.TANGENT_BASES)
def test_repacked_plane_passes_the_gates(root: Path, tmp_path: Path, basis: str, size: int) -> None:
    case = fx.plane_case(root, "split")
    recipe = fx.Recipe(texture_size=size, quality=2, tangent_basis=basis, repack_uvs=True)
    glb_path = fx.SpellbakeBaker().bake(case, recipe, tmp_path)
    verdict, report = bc.score(case.scene, glb_path)
    assert report["tangent_basis"] == basis
    assert verdict.passed, str(verdict)
    stats = json.loads((tmp_path / "bake.json").read_text(encoding="utf-8"))["stats"]
    assert stats["repacked"] and stats["min_chart_gap_px"] >= 2


# ---------------------------------------------------------------------- glTF semantics


def test_gltf_factor_semantics(tmp_path: Path) -> None:
    """Left half: factors only (opaque). Right half: colour map x factor, alpha map (BLEND), an
    ORM map with AO, and an emissive factor of 2 (above what 8 bits hold)."""
    high = tmp_path / "high.glb"
    orm = glb.Texture(solid((100, 200, 50)))
    materials = [
        glb.Material("factors", base_color_factor=(0.5, 0.2, 0.1, 1.0), metallic_factor=0.3, roughness_factor=0.7),
        glb.Material("maps", base_color_factor=(0.5, 1.0, 1.0, 1.0), base_color=glb.Texture(solid((200, 100, 50, 128))),
                     metallic_roughness=orm, occlusion=orm, emissive_factor=(2.0, 0.0, 0.0), alpha_mode="BLEND"),
    ]
    uv_of = lambda p: (p[:, :2] + 0.1) / 1.2  # noqa: E731
    glb.write_glb(high, [plane_part(-0.1, 0.5, uv_of, 0), plane_part(0.5, 1.1, uv_of, 1)], materials)
    low = tmp_path / "low.obj"
    write_low_plane(low)
    result = bake(high, low, tmp_path / "out", plain(quality=2), threads=2)

    maps = {m.id: read_png(tmp_path / "out" / m.file) for m in result.maps}
    assert {"basecolor", "normal", "roughness", "metallic", "ao", "orm", "emissive", "opacity"} <= set(maps)
    left, right = texel_mask(512, 0.1, 0.45), texel_mask(512, 0.55, 0.9)
    expected = {
        "left": (linear_to_srgb8(np.array([0.5, 0.2, 0.1])), 255, (255, unorm8(0.7), unorm8(0.3)), (0, 0, 0)),
        "right": (linear_to_srgb8(srgb_to_linear(np.array([200, 100, 50])) * [0.5, 1.0, 1.0]), 128, (100, 200, 50),
                  (255, 0, 0)),
    }
    for side, mask in (("left", left), ("right", right)):
        colour, alpha, orm_bytes, emissive = expected[side]
        for image, want in ((maps["basecolor"][..., :3], colour), (maps["basecolor"][..., 3], alpha),
                            (maps["orm"], orm_bytes), (maps["emissive"], emissive)):
            error = np.abs(image[mask].astype(int) - np.asarray(want, int)).max()
            assert error <= 1, f"{side}: {image[mask][0]} instead of {want}"
    assert (maps["roughness"] == maps["orm"][..., 1]).all() and (maps["ao"] == maps["orm"][..., 0]).all()

    material = bc.read_glb(result.folder / result.glb).material
    assert material["alphaMode"] == "BLEND"
    orm_index = material["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"]
    assert material["occlusionTexture"]["index"] == orm_index
    assert material["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"] == pytest.approx(2.0)
    assert material["extras"]["spellcast"]["tangent_basis"] == "mikk"
    assert result.stats["ambient_occlusion"] and result.stats["miss_fraction"] == 0


def test_emissive_above_one_reaches_every_export(tmp_path: Path) -> None:
    """An emissive factor of (2, 1, 0.5): the map holds colour / 2 in all three files, and each file
    carries the 2 the way its readers apply it (GLB emissive strength, FBX EmissiveFactor, MTL Ke),
    so the FBX and OBJ glow as brightly as the GLB. Assimp reads both back as emissive colour 2."""
    assimp = pytest.importorskip("assimp_py")
    high = tmp_path / "high.glb"
    glow = glb.Material("glow", base_color_factor=(0.5, 0.5, 0.5, 1.0), emissive_factor=(2.0, 1.0, 0.5))
    glb.write_glb(high, [plane_part(-0.1, 1.1, lambda p: (p[:, :2] + 0.1) / 1.2, 0)], [glow])
    low = tmp_path / "low.obj"
    write_low_plane(low)
    result = bake(high, low, tmp_path / "out", plain(export_obj=True, export_fbx=True), threads=2)

    emissive = read_png(tmp_path / "out" / "maps" / "emissive.png")[texel_mask(512, 0.2, 0.8)]
    assert np.abs(emissive.astype(int) - linear_to_srgb8(np.array([1.0, 0.5, 0.25])).astype(int)).max() <= 1
    material = bc.read_glb(result.folder / result.glb).material
    assert material["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"] == pytest.approx(2.0)
    for exported in (result.fbx, result.obj):
        read = assimp.import_file(str(result.folder / exported), assimp.Process_Triangulate).materials
        assert [m["COLOR_EMISSIVE"] for m in read if m["NAME"] == "projection"] == [pytest.approx([2.0, 2.0, 2.0])]


def test_high_normal_map_goes_through_both_frames(tmp_path: Path) -> None:
    """The high's UVs are the low's turned 90 degrees, so its tangent is the low's -y and its
    bitangent the low's +x: a map value (x, y, z) on the high is (y, -x, z) on the low."""
    wanted = np.array([0.3, 0.2, 0.93]) / np.linalg.norm([0.3, 0.2, 0.93])
    stored = np.round((wanted * 0.5 + 0.5) * 255).astype(np.uint8)
    high = tmp_path / "high.glb"
    uv_of = lambda p: np.stack([(1.1 - p[:, 1]) / 1.2, (p[:, 0] + 0.1) / 1.2], 1)  # noqa: E731
    glb.write_glb(high, [plane_part(-0.1, 1.1, uv_of, 0)],
                  [glb.Material("bumpy", normal=glb.Texture(solid(tuple(stored))), metallic_factor=0.0)])
    assert load_high(high).materials[0].normal_space == "tangent"
    low = tmp_path / "low.obj"
    write_low_plane(low)
    result = bake(high, low, tmp_path / "out", plain(normal_y="directx", export_obj=True, export_fbx=True),
                  threads=2)

    decoded = stored / 255.0 * 2 - 1
    turned = np.array([decoded[1], -decoded[0], decoded[2]]) / np.linalg.norm(decoded)
    normal = read_png(tmp_path / "out" / "maps" / "normal.png")
    mask = texel_mask(512, 0.1, 0.9)
    error = np.abs(normal[mask].astype(int) - np.round((turned * 0.5 + 0.5) * 255).astype(int)).max()
    assert error <= 1, normal[mask][0]
    directx = read_png(tmp_path / "out" / "maps" / "normal_directx.png")
    assert (directx[..., 1] == 255 - normal[..., 1]).all() and (directx[..., ::2] == normal[..., ::2]).all()
    assert "normal_directx" in [m.id for m in result.maps]
    assert (tmp_path / "out" / "export" / "projection.mtl").read_text().count("../maps/normal_directx.png") == 2


def test_rows_without_islands(tmp_path: Path) -> None:
    """UV islands in the map's bottom-left corner only: whole bands of rows have no sample."""
    high = tmp_path / "high.glb"
    glb.write_glb(high, [plane_part(-0.1, 1.1, lambda p: (p[:, :2] + 0.1) / 1.2, 0)],
                  [glb.Material("grey", base_color_factor=(0.2, 0.2, 0.2, 1.0))])
    low = tmp_path / "low.obj"
    write_low_plane(low, uv_span=0.2)
    result = bake(high, low, tmp_path / "out", plain(texture_size=1024), threads=2)
    colour = read_png(tmp_path / "out" / "maps" / "basecolor.png")
    assert (np.abs(colour.astype(int) - linear_to_srgb8(np.full(3, 0.2)).astype(int)) <= 1).all()   # filled
    assert 0.03 < result.stats["coverage"] < 0.05


@pytest.mark.slow
def test_round_trip_where_the_projection_is_invertible(root: Path, tmp_path: Path) -> None:
    """Design section 7, L4 round trip, on geometry where it is well posed: the displaced sphere
    baked onto its cube-sphere low poly, then that bake baked back onto the sphere's own
    latitude-longitude UVs, gives the sphere's texture back (measured 48.8 dB, p99 3 LSB).

    On the Pixal3D prop the same round trip measures 20.7 dB, and Blender Cycles 18.7 dB: 57%
    of that high poly's texels lie on an inner shell no ray from outside reaches (26.5 dB and
    Blender 26.0 dB on the rest)."""
    case = fx.sphere_case(root)
    settings = dict(quality=2, repack_uvs=False, ao=False, export_obj=False, export_fbx=False)
    bake(case.high, case.low, tmp_path / "forward", BakeSettings(texture_size=1024, **settings))
    back = bake(tmp_path / "forward" / "projection.glb", case.high, tmp_path / "back",
                BakeSettings(texture_size=2048, **settings))
    assert back.stats["miss_fraction"] == 0 and back.stats["flipped_triangles"] == 0
    source = bc.read_glb(case.high).images["baseColor"][..., :3].astype(float)      # 1024 rows x 2048
    rows = read_png(tmp_path / "back" / "maps" / "basecolor.png")                     # 2048 x 2048
    # Two baked rows cover one source row: average them in linear light.
    merged = linear_to_srgb8(srgb_to_linear(rows).reshape(1024, 2, 2048, 3).mean(1)).astype(float)
    psnr = 10 * np.log10(255 ** 2 / ((merged - source) ** 2).mean())
    assert psnr > 40, f"round trip {psnr:.1f} dB"


def test_same_files_on_any_thread_count(root: Path, tmp_path: Path) -> None:
    """Samples are summed in raster order whatever the chunking, and AO is seeded per texel."""
    case = fx.sphere_case(root)
    settings = BakeSettings(texture_size=512, quality=2, export_obj=False, export_fbx=False)
    one = bake(case.high, case.low, tmp_path / "one", settings, threads=1)
    three = bake(case.high, case.low, tmp_path / "three", settings, threads=3)
    for entry in one.maps:
        assert (one.folder / entry.file).read_bytes() == (three.folder / entry.file).read_bytes(), entry.id
    assert (one.folder / one.glb).read_bytes() == (three.folder / three.glb).read_bytes()


def test_cancel_in_process(root: Path, tmp_path: Path) -> None:
    case = fx.plane_case(root, "identity")
    calls = []

    def progress(fraction: float, message: str) -> None:
        calls.append(fraction)

    with pytest.raises(BakeCancelled):
        bake(case.high, case.low, tmp_path, BakeSettings(texture_size=512, quality=1), progress=progress,
             cancelled=lambda: len(calls) > 3, threads=2)
    assert not (tmp_path / "projection.glb").exists()


# ---------------------------------------------------------------------- the worker


def run_worker(request: dict[str, Any], request_file: Path) -> tuple[int, list[dict[str, Any]]]:
    request_file.write_text(json.dumps(request), encoding="utf-8")
    done = subprocess.run([sys.executable, str(WORKER), str(request_file)], capture_output=True, text=True,
                          cwd=WORKER.parent, check=False, timeout=600)
    return done.returncode, [json.loads(line) for line in done.stdout.splitlines() if line.strip()]


def bake_request(case: fx.Case, out_root: Path, **settings: Any) -> dict[str, Any]:
    tmp, final = out_root / "bake_001.tmp", out_root / "bake_001"
    tmp.mkdir(parents=True)
    return {"cmd": "bake", "high": str(case.high), "low": str(case.low), "out_dir": str(tmp),
            "final_dir": str(final), "settings": {"texture_size": 512, "quality": 1, **settings}, "threads": 2}


def test_worker_bake_contract(root: Path, tmp_path: Path) -> None:
    case = fx.sphere_case(root)
    request = bake_request(case, tmp_path / "projection", normal_y="directx")
    code, lines = run_worker(request, Path(request["out_dir"]) / "request.json")
    assert code == 0, lines
    progress = [line["progress"] for line in lines[:-1]]
    assert all("progress" in line and isinstance(line["message"], str) for line in lines[:-1])
    assert progress == sorted(progress) and 0 <= progress[0] and progress[-1] == 1.0
    assert not Path(request["out_dir"]).exists()
    folder = Path(request["final_dir"])
    result = lines[-1]["result"]
    assert (result["glb"], result["fbx"], result["obj"]) == ("projection.glb", "projection.fbx",
                                                             "export/projection.obj")
    ids = [m["id"] for m in result["maps"]]
    assert ids == ["basecolor", "normal", "normal_directx", "roughness", "metallic", "ao", "orm"]
    for entry in result["maps"]:
        assert (folder / entry["file"]).is_file() and entry["label"]
        assert entry["colorspace"] == ("srgb" if entry["id"] == "basecolor" else "linear")
    stats = result["stats"]
    for key in ("seconds", "resolution", "supersample", "ray_distance", "ray_distance_percent", "coverage",
                "miss_fraction", "triangles", "flipped_triangles", "dropped_triangles", "min_chart_gap_px",
                "warnings"):
        assert key in stats, key
    assert stats["resolution"] == 512 and 0.3 < stats["coverage"] < 1 and stats["miss_fraction"] < 1e-3
    saved = json.loads((folder / "bake.json").read_text(encoding="utf-8"))
    assert saved["settings"]["normal_y"] == "directx" and saved["stats"] == stats

    assert (folder / "projection.fbx").read_bytes()[:21] == b"Kaydara FBX Binary  \x00"
    obj = (folder / "export" / "projection.obj").read_text().splitlines()
    mtl = (folder / "export" / "projection.mtl").read_text()
    faces = [line for line in obj if line.startswith("f ")]
    assert len(faces) == stats["triangles"] and obj[1] == "mtllib projection.mtl"
    counts = {tag: sum(line.startswith(tag + " ") for line in obj) for tag in ("v", "vt", "vn")}
    corners = np.array([[int(k) for part in line[2:].split() for k in part.split("/")] for line in faces])
    assert corners.min() == 1 and all(corners[:, k::3].max() == counts[tag] for k, tag in enumerate(counts))
    for rel in ("../maps/basecolor.png", "../maps/normal_directx.png", "../maps/roughness.png",
                "../maps/metallic.png"):
        assert rel in mtl and (folder / "export" / rel).resolve().is_file()


def test_worker_reports_errors(root: Path, tmp_path: Path) -> None:
    request = bake_request(fx.plane_case(root, "identity"), tmp_path / "projection", texture_size=100)
    code, lines = run_worker(request, Path(request["out_dir"]) / "request.json")
    assert code == 1 and lines[-1]["kind"] == "user" and "Texture size" in lines[-1]["error"]
    assert not Path(request["out_dir"]).exists() and not Path(request["final_dir"]).exists()

    missing = bake_request(fx.plane_case(root, "identity"), tmp_path / "other")
    missing["low"] = str(tmp_path / "nowhere.obj")
    code, lines = run_worker(missing, tmp_path / "missing.json")
    assert code == 1 and lines[-1]["kind"] == "user" and "nowhere.obj" in lines[-1]["error"]

    code, lines = run_worker({"cmd": "paint"}, tmp_path / "bad.json")
    assert code == 1 and lines == [{"error": lines[0]["error"], "kind": "internal"}]


def test_worker_cancel_leaves_no_folder(root: Path, tmp_path: Path) -> None:
    request = bake_request(fx.sphere_case(root), tmp_path / "projection", texture_size=2048, quality=3)
    out_dir, final_dir = Path(request["out_dir"]), Path(request["final_dir"])
    request_file = out_dir / "request.json"
    request_file.write_text(json.dumps(request), encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(WORKER), str(request_file)], cwd=WORKER.parent,
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    lines = []
    for line in proc.stdout:
        lines.append(json.loads(line))
        if lines[-1].get("progress", 0) > 0.3 and not (out_dir / "cancel").exists():
            (out_dir / "cancel").write_text("", encoding="utf-8")
    assert proc.wait(60) == 1
    assert lines[-1] == {"error": "The bake was cancelled.", "kind": "cancelled"}
    assert not out_dir.exists() and not final_dir.exists()


# ---------------------------------------------------------------------- import


def normal_map(size: int) -> np.ndarray:
    """A tangent-space (OpenGL) normal map of smooth bumps."""
    y, x = np.mgrid[0:size, 0:size] / size * 2 * math.pi
    slope_x = 0.5 * np.cos(2 * x) * np.cos(y)
    slope_up = 0.25 * np.sin(2 * x) * np.sin(y)        # d height / d(up) = -d height / d row
    n = np.stack([-slope_x, -slope_up, np.ones_like(x)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.round((n * 0.5 + 0.5) * 255).astype(np.uint8)


def test_import_obj_mtl(tmp_path: Path) -> None:
    folder = tmp_path / "model"
    folder.mkdir()
    size = 64
    ramp = np.linspace(0, 255, size).astype(np.uint8)
    colour = np.stack([np.tile(ramp, (size, 1)), np.tile(ramp[:, None], (1, size)),
                       np.full((size, size), 90, np.uint8)], -1)
    roughness = np.tile(ramp[::-1], (size, 1))
    metallic = np.where(np.add.outer(np.arange(size) // 16, np.arange(size) // 16) % 2 == 0, 255, 0).astype(np.uint8)
    normal = normal_map(size)
    for name, image in (("Crate_BaseColor", colour), ("Crate_Normal", normal), ("Crate_Roughness", roughness),
                        ("Crate_Metallic", metallic)):
        Image.fromarray(image).save(folder / f"{name}.png")
    (folder / "crate.mtl").write_text("newmtl crate\nKd 1 1 1\nmap_Kd Crate_BaseColor.png\nnorm Crate_Normal.png\n"
                                      "map_Pr Crate_Roughness.png\nmap_Pm Crate_Metallic.png\n", encoding="utf-8")
    (folder / "crate.obj").write_text("mtllib crate.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\n"
                                      "vt 1 1\nvt 0 1\nvn 0 0 1\nusemtl crate\nf 1/1/1 2/2/1 3/3/1 4/4/1\n",
                                      encoding="utf-8")
    dst = tmp_path / "imported.glb"
    code, lines = run_worker({"cmd": "import", "src": str(folder / "crate.obj"), "dst": str(dst), "overrides": {}},
                             tmp_path / "import.json")
    assert code == 0, lines
    result = lines[-1]["result"]
    assert (result["glb"], result["vertices"], result["faces"]) == (str(dst), 4, 2)
    [material] = result["materials"]
    roles = {entry["file"]: entry["role"] for entry in material["maps"]}
    assert roles == {"Crate_BaseColor.png": "basecolor", "Crate_Normal.png": "normal",
                     "Crate_Roughness.png": "roughness", "Crate_Metallic.png": "metallic"}
    assert all(set(entry) == {"key", "file", "role", "label", "confidence", "y_convention", "y_confident",
                              "evidence"} for entry in material["maps"])

    g = bc.read_glb(dst)
    np.testing.assert_allclose(g.mesh.pos[g.mesh.faces].reshape(-1, 3).min(0), [0, 0, 0])
    assert np.abs(g.images["baseColor"].astype(int) - colour).mean() < 1.0     # JPEG, quality 95
    assert np.abs(g.images["baseColor"].astype(int) - colour).max() <= 12
    assert (g.images["normal"] == normal).all()
    orm = g.images["metallicRoughness"]
    assert (orm[..., 0] == 255).all() and (orm[..., 1] == roughness).all() and (orm[..., 2] == metallic).all()
    assert "occlusion" not in g.images
    assert g.material["pbrMetallicRoughness"]["metallicFactor"] == 1.0

    # The import bakes like any other high poly.
    low = tmp_path / "low.obj"
    write_low_plane(low, 0.05, 0.95)
    baked = bake(dst, low, tmp_path / "bake", plain(), threads=2)
    assert baked.stats["miss_fraction"] == 0


# ---------------------------------------------------------------------- speed (real asset)


@pytest.mark.slow
@pytest.mark.realasset
def test_prop_bake_speed_and_memory(tmp_path: Path) -> None:
    """Design section 4.5: the p_025d38 prop at 2048 px, quality 2, with the product's defaults
    (repacked UVs, ambient occlusion), within 35 s on 4 threads and 2.5 GB."""
    high, low = E2E_PROP / "mesh" / "mesh.glb", E2E_PROP / "retopo" / "retopo.obj"
    if not (high.is_file() and low.is_file()):
        pytest.skip(f"real asset not on this PC: {E2E_PROP}")
    high = Path(shutil.copyfile(high, tmp_path / "mesh.glb"))
    low = Path(shutil.copyfile(low, tmp_path / "retopo.obj"))
    started = time.perf_counter()
    run = fx.bake_in_child(high, low, tmp_path / "bake",
                           fx.Recipe(texture_size=2048, quality=2, repack_uvs=True, ao=True), threads=4)
    assert run["seconds"] <= 35, f"{run['seconds']:.1f} s ({time.perf_counter() - started:.1f} s with start-up)"
    if run["peak_rss"] is not None:
        assert run["peak_rss"] <= 2.5 * GIB, f"peak {run['peak_rss'] / GIB:.2f} GiB"
    stats = json.loads((tmp_path / "bake" / "bake.json").read_text(encoding="utf-8"))["stats"]
    assert stats["miss_fraction"] <= 0.005 and stats["min_chart_gap_px"] >= 8
