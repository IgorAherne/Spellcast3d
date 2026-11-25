"""L1/L2 (design section 7): bakes of analytic scenes, scored black-box against the hard gates.

* planes with 7 UV layouts (identity, rot90, rot37, mirror, aniso 2:1, shear, split)
  placed with a world rotation and offset, and a displaced sphere, in both tangent bases;
* the texel-exact analytic sphere at 1024^2 S = 2: interior colour within 1/255 and
  normals within the 8-bit image of 0.05 deg;
* the dev-only Node toolchain (glTF-Validator, reference MikkTSpace) when installed.

Every test runs the oracle -- which proves the harness and its gates -- and, once
``spellbake.bake`` imports (the bake-pipeline package), the real baker. The oracle runs
at S = 1 (its calibration; S = 2 gives the same floors, measured); the real baker at the
product default S = 2.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("instant_meshes_brush.uv", reason="the low OBJs are written by instant_meshes_brush.uv")

from bakecheck import bakecheck as bc  # noqa: E402
from bakecheck import fixtures as fx  # noqa: E402

NODE_DIR = Path(__file__).resolve().parent / "node"
QUALITY = {"oracle": 1, "spellbake": 2}


@pytest.fixture(scope="module")
def root(request: pytest.FixtureRequest) -> Path:
    return fx.cache_root(request.config)


def baker(name: str) -> fx.Baker:
    if name == "spellbake" and not fx.spellbake_available():
        pytest.skip("spellbake.bake is not importable yet (bake-pipeline)")
    return fx.make_baker(name)


def assert_passes(case: fx.Case, glb: Path, basis: str) -> dict:
    verdict, report = bc.score(case.scene, glb)
    assert report["tangent_basis"] == basis, "the GLB must declare the basis it was baked for"
    assert verdict.passed, f"{case.name} / {basis}:\n{verdict}"
    return report


@pytest.mark.parametrize("baker_name", fx.BAKERS)
@pytest.mark.parametrize("basis", bc.TANGENT_BASES)
@pytest.mark.parametrize("layout", fx.PLANE_LAYOUTS)
def test_plane_layouts(root: Path, tmp_path: Path, baker_name: str, basis: str, layout: str) -> None:
    case = fx.plane_case(root, layout)
    recipe = fx.Recipe(texture_size=512, quality=QUALITY[baker_name], tangent_basis=basis)
    assert_passes(case, baker(baker_name).bake(case, recipe, tmp_path), basis)


SPHERE_RUNS = [("oracle", False), ("spellbake", False), ("spellbake", True)]


@pytest.mark.parametrize(("baker_name", "repack"), SPHERE_RUNS)
@pytest.mark.parametrize("basis", bc.TANGENT_BASES)
def test_displaced_sphere(root: Path, tmp_path: Path, baker_name: str, repack: bool, basis: str) -> None:
    case = fx.sphere_case(root)
    recipe = fx.Recipe(texture_size=1024, quality=QUALITY[baker_name], tangent_basis=basis, repack_uvs=repack)
    report = assert_passes(case, baker(baker_name).bake(case, recipe, tmp_path), basis)
    assert report["seam_samples"] > 0, "the sphere's charts must meet at UV seams"


@pytest.mark.parametrize("baker_name", fx.BAKERS)
@pytest.mark.parametrize("basis", bc.TANGENT_BASES)
def test_analytic_sphere_is_exact(root: Path, tmp_path: Path, baker_name: str, basis: str) -> None:
    """No distortion: every interior texel equals the analytic answer up to 8-bit rounding."""
    case = fx.exact_sphere_case(root)
    recipe = fx.Recipe(texture_size=1024, quality=2, tangent_basis=basis, repack_uvs=True)
    exact = bc.evaluate_exact(case.scene, baker(baker_name).bake(case, recipe, tmp_path), recipe.quality)
    assert exact["interior_texels"] > 300_000
    assert exact["colour_lsb"]["max"] <= bc.EXACT_COLOUR_LSB, exact
    assert exact["normal_lsb"]["max"] <= bc.EXACT_NORMAL_LSB, exact


def test_reader_merges_primitives_and_applies_node_transforms(root: Path, tmp_path: Path) -> None:
    """The scorer reads every triangle primitive of the default scene in world space."""
    source = fx.trench_case(root).high  # five primitives, one material
    flat = bc.read_glb(source)
    data = source.read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + length])
    gltf["nodes"] = [{"children": [1], "translation": [1.0, 2.0, 3.0]},
                     {"mesh": 0, "rotation": [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)],
                      "scale": [2.0, 2.0, 2.0]}]
    gltf["scenes"] = [{"nodes": [0]}]
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    rest = data[20 + length:]
    moved = tmp_path / "moved.glb"
    moved.write_bytes(struct.pack("<III", 0x46546C67, 2, 20 + len(js) + len(rest))
                      + struct.pack("<II", len(js), 0x4E4F534A) + js + rest)
    g = bc.read_glb(moved)
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # +90 deg about z
    primitives = gltf["meshes"][0]["primitives"]
    assert len(primitives) == 5
    triangles = sum(gltf["accessors"][prim["indices"]]["count"] for prim in primitives) // 3
    assert g.mesh.faces.shape == flat.mesh.faces.shape == (triangles, 3)
    np.testing.assert_allclose(g.mesh.pos, 2.0 * flat.mesh.pos @ quarter_turn.T + [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(g.mesh.nrm, flat.mesh.nrm @ quarter_turn.T, atol=1e-6)


def run_node(script: str, glb: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["node", str(NODE_DIR / script), str(glb)], capture_output=True, text=True,
                          cwd=NODE_DIR, check=False)


@pytest.mark.skipif(shutil.which("node") is None or not (NODE_DIR / "node_modules").is_dir(),
                    reason="dev-only: needs Node and `npm ci` in code/tests/bake/node")
@pytest.mark.parametrize("baker_name", fx.BAKERS)
def test_node_toolchain(root: Path, tmp_path: Path, baker_name: str) -> None:
    """The GLB passes the Khronos validator with no warnings, and its TANGENTs are the
    reference MikkTSpace ones (the oracle's Mikk-style tangents agree on this clean mesh)."""
    case = fx.sphere_case(root)
    glb = baker(baker_name).bake(case, fx.Recipe(texture_size=512, quality=QUALITY[baker_name]), tmp_path)
    for script in ("validate.mjs", "mikkcheck.cjs"):
        done = run_node(script, glb)
        assert done.returncode == 0, f"{script}:\n{done.stdout}\n{done.stderr}"
