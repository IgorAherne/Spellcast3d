"""L5 (design section 7): what a single smooth surface cannot tell apart.

(a) hoop over planks, (b) parallel sheets: several surfaces on one baking ray, the low
    poly between them and nearer the inner one. The cage policy (first front face from
    outside) must bake the OUTER surface on 100% of its footprint minus a 1-texel border.
    The oracle run with the "nearest hit" policy bakes the planks under the hoop, which
    proves the check sees the difference.
(c) trench AO: on the floor of a trench W wide and H deep, cosine-weighted AO is the
    view factor of the opening, AO(x) = 1/2 [x/sqrt(x^2+H^2) + (W-x)/sqrt((W-x)^2+H^2)].
(d) height: the height bake equals the analytic h(x, y) of the plane to 1e-3 of the
    diagonal.

The ray distance is set (2% of the diagonal) where the low runs inside a raised
feature: the automatic estimate measures the NEAREST surface, which is the inner one.
Every test runs the oracle and, once ``spellbake.bake`` imports, the real baker.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("instant_meshes_brush.uv", reason="the low OBJs are written by instant_meshes_brush.uv")

from bakecheck import bakecheck as bc  # noqa: E402
from bakecheck import fixtures as fx  # noqa: E402

pytestmark = pytest.mark.slow

QUALITY = {"oracle": 1, "spellbake": 2}
COLOUR_TOLERANCE_LSB = 2
AO_RAYS = 64
AO_MEAN_ERROR = 0.01
AO_SIGMAS = 4.0
AO_WITHIN_SIGMAS = 0.99
HEIGHT_TOLERANCE = 1e-3  # of the high poly's diagonal


@pytest.fixture(scope="module")
def root(request: pytest.FixtureRequest) -> Path:
    return fx.cache_root(request.config)


def baker(name: str) -> fx.Baker:
    if name == "spellbake" and not fx.spellbake_available():
        pytest.skip("spellbake.bake is not importable yet (bake-pipeline)")
    return fx.make_baker(name)


def interior_texels(g: bc.BakedGlb) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Texels whose centre and 8 neighbours' centres are covered (1-texel border off):
    (rows, cols, surface point at the texel centre)."""
    res = g.resolution
    tid, bary, _ = bc.raster_uv(g.mesh.uv, g.mesh.faces, res)
    covered = np.pad(tid >= 0, 1)
    inner = np.ones((res, res), bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            inner &= covered[dy:dy + res, dx:dx + res]
    rows, cols = np.nonzero(inner)
    f = g.mesh.faces[tid[rows, cols]]
    return rows, cols, (g.mesh.pos[f] * bary[rows, cols][..., None]).sum(1)


def texel_size(g: bc.BakedGlb) -> float:
    """World size of one texel (the policy lows have uniform texel density)."""
    p = g.mesh.pos[g.mesh.faces]
    e1, e2 = (g.mesh.uv[g.mesh.faces[:, k]] - g.mesh.uv[g.mesh.faces[:, 0]] for k in (1, 2))
    area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=-1).sum()
    uv_area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
    return float(np.sqrt(area / uv_area)) / g.resolution


def outer_surface_check(case: fx.Case, glb: Path) -> tuple[int, int]:
    """(wrong texels, texels checked) of the outer-surface rule."""
    g = bc.read_glb(glb)
    rows, cols, P = interior_texels(g)
    colour = g.images["baseColor"][rows, cols, :3].astype(np.int64)
    if case.name == "hoop":
        h = texel_size(g)
        band = (P[:, 0] >= fx.HOOP_X[0] + h) & (P[:, 0] <= fx.HOOP_X[1] - h)
        planks = (P[:, 0] <= fx.HOOP_X[0] - h) | (P[:, 0] >= fx.HOOP_X[1] + h)
        expected = np.where(band[:, None], fx.HOOP, fx.PLANKS)
        checked = band | planks
        assert band.sum() > 1000, "the hoop's footprint must cover texels"
    else:
        expected = np.tile(fx.OUTER, (rows.size, 1))
        checked = np.ones(rows.size, bool)
    wrong = np.abs(colour - expected).max(-1) > COLOUR_TOLERANCE_LSB
    return int((wrong & checked).sum()), int(checked.sum())


POLICY_CASES: dict[str, Callable[[Path], fx.Case]] = {"hoop": fx.hoop_case, "sheets": fx.sheets_case}


@pytest.mark.parametrize("baker_name", fx.BAKERS)
@pytest.mark.parametrize("scene_name", sorted(POLICY_CASES))
def test_outer_surface_wins(root: Path, tmp_path: Path, baker_name: str, scene_name: str) -> None:
    case = POLICY_CASES[scene_name](root)
    recipe = fx.Recipe(texture_size=512, quality=QUALITY[baker_name], ray_distance=2.0)
    wrong, checked = outer_surface_check(case, baker(baker_name).bake(case, recipe, tmp_path))
    assert checked > 20_000
    assert wrong == 0, f"{wrong} of {checked} texels show an inner surface"


@pytest.mark.parametrize("scene_name", sorted(POLICY_CASES))
def test_nearest_hit_policy_is_caught(root: Path, tmp_path: Path, scene_name: str) -> None:
    case = POLICY_CASES[scene_name](root)
    recipe = fx.Recipe(texture_size=512, quality=1, ray_distance=2.0)
    wrong, _ = outer_surface_check(case, fx.OracleBaker(policy="nearest").bake(case, recipe, tmp_path))
    assert wrong > 1000


@pytest.mark.parametrize("baker_name", fx.BAKERS)
def test_trench_ambient_occlusion(root: Path, tmp_path: Path, baker_name: str) -> None:
    """Mean error <= 0.01 on the floor, and per texel within 4 sigma of binomial noise.

    The mean is taken per column (fixed x) along the trench, so the 64-ray noise of single
    texels averages out and what remains is the estimator's bias: cosine weighting, the
    ray origin, the ray length, the normal used. (A per-texel mean |error| cannot reach
    0.01 at 64 rays: a correct stratified estimator measures 0.018.) The default AO ray
    length, 10% of the diagonal = 5 trench depths, cuts off only 0.0009 of the AO."""
    case = fx.trench_case(root)
    assert isinstance(case.scene, fx.TrenchScene)
    recipe = fx.Recipe(texture_size=512, quality=QUALITY[baker_name], ao=True, ao_rays=AO_RAYS)
    g = bc.read_glb(baker(baker_name).bake(case, recipe, tmp_path))
    assert "occlusion" in g.images, "an AO bake must set occlusionTexture"
    rows, cols, P = interior_texels(g)
    expected = case.scene.ao(P[:, 0])
    err = g.images["occlusion"][rows, cols, 0] / 255.0 - expected

    column = np.unique(cols, return_inverse=True)[1]
    profile = np.bincount(column, err) / np.bincount(column)
    assert abs(err.mean()) <= AO_MEAN_ERROR, f"AO bias {err.mean():+.4f}"
    assert np.abs(profile).mean() <= AO_MEAN_ERROR, f"AO(x) off by {np.abs(profile).mean():.4f} on average"

    sigma = np.sqrt(expected * (1 - expected) / AO_RAYS) + 0.5 / 255
    within = (np.abs(err) <= AO_SIGMAS * sigma).mean()
    assert within >= AO_WITHIN_SIGMAS, f"only {within:.2%} of texels within {AO_SIGMAS} sigma"


@pytest.mark.parametrize("baker_name", fx.BAKERS)
def test_height_bake(root: Path, tmp_path: Path, baker_name: str) -> None:
    case = fx.plane_case(root, "identity")
    assert isinstance(case.scene, bc.PlaneScene)
    recipe = fx.Recipe(texture_size=512, quality=QUALITY[baker_name], height=True)
    g = bc.read_glb(baker(baker_name).bake(case, recipe, tmp_path))
    lo, hi = g.spellcast["height_range"]
    with Image.open(tmp_path / "maps" / "height.png") as img:
        height = np.asarray(img)
    assert height.dtype == np.uint16, "the height map is a 16-bit PNG"
    rows, cols, P = interior_texels(g)
    baked = lo + (hi - lo) * height[rows, cols] / 65535.0
    local = case.scene.to_local(P)
    err = np.abs(baked - case.scene.h(local[:, 0], local[:, 1]))
    assert err.max() <= HEIGHT_TOLERANCE * case.diag, f"height off by {err.max():.3g} (diag {case.diag:.3g})"
