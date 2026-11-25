"""The gates bite: the oracle with each classic baker bug must FAIL at least one gate,
and the clean oracle must PASS on the same fixtures (design section 7, mutation tests).

Most bugs show on the split plane (a mirrored, rotated second chart); ``face_rays`` and
``transpose`` need curvature the tessellation does not follow, so they run on the
icosahedron over the displaced sphere, where the vertex normals are 63 deg apart.
The CLI's PASS/FAIL exit code is checked on the same bakes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("instant_meshes_brush.uv", reason="the low OBJs are written by instant_meshes_brush.uv")

from bakecheck import bakecheck as bc  # noqa: E402
from bakecheck import cli  # noqa: E402
from bakecheck import fixtures as fx  # noqa: E402

#: mutation -> the fixture that exposes it
WHERE = {"flip_green": "plane_split", "ignore_w": "plane_split", "vflip": "plane_split",
         "gamma_twice": "plane_split", "half_texel": "plane_split", "no_padding": "plane_split",
         "face_rays": "coarse_sphere", "transpose": "coarse_sphere"}


@pytest.fixture(scope="module")
def root(request: pytest.FixtureRequest) -> Path:
    return fx.cache_root(request.config)


def setup(root: Path, where: str, basis: str) -> tuple[fx.Case, fx.Recipe]:
    if where == "plane_split":
        return fx.plane_case(root, "split"), fx.Recipe(texture_size=512, quality=1, tangent_basis=basis)
    # the icosahedron's faces sink 0.2 below the surface: a 10%-of-diagonal ray window
    return fx.coarse_sphere_case(root), fx.Recipe(texture_size=512, quality=1, tangent_basis=basis,
                                                  ray_distance=10.0)


def test_every_mutation_has_a_fixture() -> None:
    assert set(WHERE) == set(bc.MUTATIONS)


@pytest.mark.parametrize("where", sorted(set(WHERE.values())))
@pytest.mark.parametrize("basis", bc.TANGENT_BASES)
def test_clean_oracle_passes(root: Path, tmp_path: Path, where: str, basis: str) -> None:
    case, recipe = setup(root, where, basis)
    verdict, _ = bc.score(case.scene, fx.OracleBaker().bake(case, recipe, tmp_path))
    assert verdict.passed, f"clean oracle on {where}:\n{verdict}"


@pytest.mark.parametrize("basis", bc.TANGENT_BASES)
@pytest.mark.parametrize("mutation", bc.MUTATIONS)
def test_mutation_fails_a_gate(root: Path, tmp_path: Path, mutation: str, basis: str) -> None:
    case, recipe = setup(root, WHERE[mutation], basis)
    verdict, _ = bc.score(case.scene, fx.OracleBaker(mutations=[mutation]).bake(case, recipe, tmp_path))
    assert not verdict.passed, f"{mutation} slipped through every gate:\n{verdict}"


def test_cli_exit_codes(root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    case, recipe = setup(root, "plane_split", "mikk")
    scene_json = case.high.parent / "scene.json"
    clean = fx.OracleBaker().bake(case, recipe, tmp_path / "clean")
    report = tmp_path / "report.json"
    assert cli.main([str(scene_json), str(clean), "--json", str(report)]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out
    assert report.is_file()
    flipped = fx.OracleBaker(mutations=["flip_green"]).bake(case, recipe, tmp_path / "flipped")
    assert cli.main([str(scene_json), str(flipped)]) == 1
    assert "RESULT: FAIL" in capsys.readouterr().out
