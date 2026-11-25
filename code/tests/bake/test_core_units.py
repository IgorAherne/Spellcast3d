"""Unit gates for the baker's building blocks (design section 7, "Baker L0").

Every expected value here is exact or analytic: texel centres, barycentrics, coverage
ownership, colour look-up tables, mip means, fill, the tangent-space encoders and their
decoders, MikkTSpace against the reference wasm, the winding vote, the rays and the AO.
"""

from __future__ import annotations

import json
import math
import os
import struct
import time
from pathlib import Path

import numpy as np
import pytest

from spellbake import BakeError, ao, encode, fill, lowpoly, rays, raster, sampling, tangents

DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------------- helpers


def _icosphere(subdiv: int) -> tuple[np.ndarray, np.ndarray]:
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
             (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    points = [np.array(v, float) / np.linalg.norm(v) for v in verts]
    for _ in range(subdiv):
        cache: dict[tuple[int, int], int] = {}

        def mid(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = points[a] + points[b]
                points.append(m / np.linalg.norm(m))
                cache[key] = len(points) - 1
            return cache[key]

        faces = [f for a, b, c in faces
                 for f in ((a, mid(a, b), mid(c, a)), (b, mid(b, c), mid(a, b)),
                           (c, mid(c, a), mid(b, c)), (mid(a, b), mid(b, c), mid(c, a)))]
    return np.array(points), np.array(faces, dtype=np.int64)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1)))


def _all_samples(uv: np.ndarray, faces: np.ndarray, resolution: int, supersample: int,
                 **kwargs) -> dict[str, np.ndarray]:
    bands = list(raster.sample_bands(uv, faces, resolution, supersample, **kwargs))
    return {name: np.concatenate([getattr(b, name) for b in bands])
            for name in ("tri", "bary", "texel", "sub", "ring")}


def _sample_xy(texel: np.ndarray, sub: np.ndarray, resolution: int, supersample: int) -> np.ndarray:
    """Sample centres in texel units (x right, y down)."""
    x = texel % resolution + (sub % supersample + 0.5) / supersample
    y = texel // resolution + (sub // supersample + 0.5) / supersample
    return np.stack([x, y], axis=1)


def _grid(n: int, size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """An n x n quad grid in the z = 0 plane, two triangles per quad (counter-clockwise from +z)."""
    xs = np.linspace(0.0, size, n + 1)
    x, y = np.meshgrid(xs, xs)
    points = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    return points, np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])


def _write_obj(path: Path, positions: np.ndarray, uvs: np.ndarray, faces: list[list[tuple[int, int]]]) -> None:
    lines = [f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in positions]
    lines += [f"vt {u:.9f} {v:.9f}" for u, v in uvs]
    lines += ["f " + " ".join(f"{p + 1}/{t + 1}" for p, t in face) for face in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------- raster


def test_texel_centre_convention() -> None:
    resolution = 4
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    s = _all_samples(uv, faces, resolution, 1)
    assert np.array_equal(np.sort(s["texel"]), np.arange(resolution * resolution))
    got = np.einsum("nk,nkd->nd", s["bary"], uv[faces[s["tri"]]])
    column, row = s["texel"] % resolution, s["texel"] // resolution
    np.testing.assert_allclose(got, raster.texel_centre_uv(column, row, resolution), atol=1e-12)
    np.testing.assert_allclose(raster.texel_centre_uv(0, 0, resolution), [0.125, 0.875])

    # 2 x 2 supersampling: sample (i, j) of texel (c, r) at u = (c + (i + 1/2) / 2) / R.
    s = _all_samples(uv, faces, resolution, 2)
    got = np.einsum("nk,nkd->nd", s["bary"], uv[faces[s["tri"]]])
    xy = _sample_xy(s["texel"], s["sub"], resolution, 2)
    np.testing.assert_allclose(got, np.stack([xy[:, 0] / resolution, 1 - xy[:, 1] / resolution], 1),
                               atol=1e-12)
    assert np.array_equal(np.bincount(s["texel"]), np.full(resolution * resolution, 4))


def test_barycentric_round_trip() -> None:
    rng = np.random.default_rng(3)
    uv = rng.random((60, 2))
    faces = rng.integers(0, 60, (40, 3))
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    resolution, supersample = 64, 3
    s = _all_samples(uv, faces, resolution, supersample)
    assert len(s["tri"]) > 10_000
    got = np.einsum("nk,nkd->nd", s["bary"], uv[faces[s["tri"]]]) * resolution
    xy = _sample_xy(s["texel"], s["sub"], resolution, supersample)
    err = np.abs(got - np.stack([xy[:, 0], resolution - xy[:, 1]], 1)).max() * supersample
    assert err < 1e-6, f"barycentric round trip off by {err} samples"
    assert np.allclose(s["bary"].sum(axis=1), 1.0)
    assert (s["bary"] >= -1e-12).all()


def _lattice_chart() -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """A closed chart whose vertices sit exactly on sample centres of a 64-texel map with
    2 x 2 samples (128 across), so samples land on shared edges and vertices: quads split
    along alternating diagonals, plus a fan of 12 triangles around one vertex with edge
    slopes 0, 1/2, 1, 2 and vertical."""
    samples = 128
    points, faces = [], []
    xs, ys = np.arange(9) * 7 + 10.5, np.arange(9) * 5 + 20.5
    idx = np.arange(81).reshape(9, 9)
    for y in ys:
        for x in xs:
            points.append((x, y))
    for j in range(8):
        for i in range(8):
            a, b, c, d = idx[j, i], idx[j, i + 1], idx[j + 1, i + 1], idx[j + 1, i]
            faces += [(a, b, c), (a, c, d)] if (i + j) % 2 else [(a, b, d), (b, c, d)]
    centre = len(points)
    points.append((100.5, 100.5))
    ring = [(4, 0), (4, 2), (4, 4), (2, 4), (0, 4), (-2, 4), (-4, 4), (-4, 0), (-4, -4), (0, -4),
            (4, -4), (4, -2)]
    first = len(points)
    points += [(100.5 + dx, 100.5 + dy) for dx, dy in ring]
    faces += [(centre, first + k, first + (k + 1) % len(ring)) for k in range(len(ring))]
    xy = np.array(points)
    uv = np.stack([xy[:, 0] / samples, 1 - xy[:, 1] / samples], axis=1)
    return uv, np.array(faces), (10.5, 66.5, 20.5, 60.5)


@pytest.mark.parametrize("mirror", [False, True])
def test_top_left_rule_gives_each_sample_one_owner(mirror: bool) -> None:
    uv, faces, (x0, x1, y0, y1) = _lattice_chart()
    if mirror:
        faces = faces[:, ::-1]
    resolution, supersample = 64, 2
    s = _all_samples(uv, faces, resolution, supersample, band_rows=16)
    xy = _sample_xy(s["texel"], s["sub"], resolution, supersample) * supersample
    key = np.round(xy[:, 1] - 0.5).astype(int) * 128 + np.round(xy[:, 0] - 0.5).astype(int)
    count = np.bincount(key, minlength=128 * 128).reshape(128, 128)
    ys, xs = np.mgrid[0:128, 0:128] + 0.5
    grid_inside = (xs > x0) & (xs < x1) & (ys > y0) & (ys < y1)
    fan_inside = (np.abs(xs - 100.5) < 4) & (np.abs(ys - 100.5) < 4)   # inside the fan's octagon
    assert count.max() == 1, "a sample was claimed by two triangles"
    assert (count[grid_inside] == 1).all() and (count[fan_inside] == 1).all(), "a sample inside the chart was lost"
    assert count[100, 100] == 1, "the fan's centre vertex needs exactly one owner"


def test_ring_extrapolates_the_nearest_triangle() -> None:
    resolution, supersample, width = 32, 2, 2.5
    uv = np.array([[0.30, 0.30], [0.70, 0.35], [0.45, 0.72]])
    faces = np.array([[0, 1, 2]])
    s = _all_samples(uv, faces, resolution, supersample, ring=width, band_rows=5)
    slots = s["texel"].astype(np.int64) * 4 + s["sub"]
    per_texel = np.bincount(s["texel"], minlength=resolution ** 2)
    touched = np.unique(s["texel"][~s["ring"]])
    assert len(np.unique(slots[np.isin(s["texel"], touched)])) == len(slots[np.isin(s["texel"], touched)])
    # A texel the triangle touches gets all 4 samples (the missing ones extrapolated);
    # a texel it misses but that lies within the ring gets one sample at its centre.
    assert (per_texel[touched] == supersample ** 2).all()
    centre_only = np.setdiff1d(np.unique(s["texel"]), touched)
    assert (per_texel[centre_only] == 1).all() and (s["sub"][np.isin(s["texel"], centre_only)] == 0).all()
    assert (s["bary"][s["ring"]].min(axis=1) < 0).any(), "ring barycentrics are not clamped"
    # Every extrapolated sample sits exactly where it should (texel centres for the ring).
    got = np.einsum("nk,nkd->nd", s["bary"], uv[faces[s["tri"]]])
    xy = _sample_xy(s["texel"], s["sub"], resolution, supersample)
    xy[np.isin(s["texel"], centre_only)] = np.stack(
        [s["texel"] % resolution + 0.5, s["texel"] // resolution + 0.5], 1)[np.isin(s["texel"], centre_only)]
    np.testing.assert_allclose(got * [resolution, -resolution] + [0, resolution], xy, atol=1e-9)
    # Ring texels are the uncovered ones within `width` texels of the triangle, all of them.
    px = uv * [resolution, -resolution] + [0, resolution]
    index = np.arange(resolution ** 2)
    centres = np.stack([index % resolution + 0.5, index // resolution + 0.5], 1)
    dist = np.min([_segment_distance(centres, px[a], px[b]) for a, b in ((0, 1), (1, 2), (2, 0))], axis=0)
    assert dist[centre_only].max() <= width + 1e-9
    reached = per_texel > 0
    assert reached[dist < width - 1e-6].all()


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    e = b - a
    t = np.clip(((p - a) @ e) / (e @ e), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * e), axis=1)


def test_raster_skips_triangles_without_uv_area() -> None:
    uv = np.array([[0.1, 0.1], [0.9, 0.1], [0.5, 0.1], [0.5, 0.9]])
    s = _all_samples(uv, np.array([[0, 1, 2], [0, 1, 3]]), 16, 1, ring=2)
    assert set(np.unique(s["tri"])) == {1}


# ---------------------------------------------------------------------- sampling


def test_srgb_lut_round_trip_is_exact() -> None:
    codes = np.arange(256, dtype=np.uint8)
    assert np.array_equal(sampling.linear_to_srgb8(sampling.srgb_to_linear(codes)), codes)
    assert np.array_equal(sampling.linear_to_srgb8(sampling.SRGB_TO_LINEAR), codes)
    assert np.array_equal(sampling.unorm8(codes / 255.0), codes)


@pytest.mark.parametrize("srgb", [0, 3])
@pytest.mark.parametrize("shape", [(256, 256, 4), (64, 256, 3)])
def test_mip_pyramid_preserves_the_mean(srgb: int, shape: tuple[int, int, int]) -> None:
    image = np.random.default_rng(5).integers(0, 256, shape, dtype=np.uint8)
    pyramid = sampling.MipPyramid(image, srgb_channels=srgb)
    assert pyramid.levels[-1].shape[:2] == (1, 1)
    means = [level.reshape(-1, shape[2]).mean(axis=0) * scale
             for level, scale in zip(pyramid.levels, pyramid.scales)]
    assert np.abs(np.array(means) - means[0]).max() < 1e-3
    if srgb:   # the colour channels are filtered in linear light
        linear = sampling.srgb_to_linear(image[..., :srgb]).reshape(-1, srgb).mean(axis=0)
        np.testing.assert_allclose(means[-1][:srgb], linear, atol=1e-3)


def test_trilinear_sampling_and_wrap_modes() -> None:
    image = np.arange(16 * 8, dtype=np.uint8).reshape(8, 16) * 2
    pyramid = sampling.MipPyramid(image)
    col, row = np.meshgrid(np.arange(16), np.arange(8))
    u, v = (col.ravel() + 0.5) / 16, 1 - (row.ravel() + 0.5) / 8
    np.testing.assert_allclose(pyramid.sample(u, v)[:, 0] * 255, image.ravel(), atol=1e-3)

    u = np.array([0.2, 0.37])
    v = np.array([0.6, 0.15])
    for mode in (sampling.REPEAT, sampling.CLAMP_TO_EDGE, sampling.MIRRORED_REPEAT):
        base = pyramid.sample(u, v, 0.0, mode, mode)
        if mode == sampling.REPEAT:
            np.testing.assert_allclose(pyramid.sample(u + 3, v - 2, 0.0, mode, mode), base, atol=1e-5)
        elif mode == sampling.MIRRORED_REPEAT:
            np.testing.assert_allclose(pyramid.sample(2 - u, -v, 0.0, mode, mode), base, atol=1e-5)
        else:
            np.testing.assert_allclose(pyramid.sample(np.array([-5.0]), np.array([0.5 - 0.5 / 8]), 0.0, mode, mode),
                                       pyramid.sample(np.array([0.5 / 16]), np.array([0.5 - 0.5 / 8]), 0.0, mode, mode))
    # A fractional level blends the two levels around it linearly.
    lo, hi = pyramid.sample(u, v, 1.0), pyramid.sample(u, v, 2.0)
    np.testing.assert_allclose(pyramid.sample(u, v, 1.25), lo + 0.25 * (hi - lo), atol=1e-5)


def test_mip_level_matches_texel_footprint() -> None:
    positions = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 1, 2]])
    uv = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]])          # 0.5 UV per world unit
    jacobian = sampling.uv_jacobians(positions, uv, faces)
    np.testing.assert_allclose(np.abs(np.linalg.det(jacobian)), 0.25)
    # 1024 texels across the map: one world unit covers 512 texels.
    np.testing.assert_allclose(sampling.mip_level(np.array([1 / 512, 4 / 512]), jacobian, 1024, 1024),
                               [0.0, 2.0], atol=1e-6)
    # Stretched 10:1: the level follows the direction with the fewest texels (51.2 per unit).
    squashed = sampling.uv_jacobians(positions, np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.05]]), faces)
    np.testing.assert_allclose(sampling.mip_level(np.array([1 / 51.2, 4 / 51.2]), squashed, 1024, 1024),
                               [0.0, 2.0], atol=1e-6)
    # Non-square maps count texels per axis: 1024 x 256 gives 128 texels per unit along v.
    np.testing.assert_allclose(sampling.mip_level(np.array([2 / 128]), jacobian[:1], 1024, 256), [1.0], atol=1e-6)


# ---------------------------------------------------------------------- fill


def test_push_pull_leaves_no_empty_texel() -> None:
    rng = np.random.default_rng(7)
    h, w = 100, 70
    mask = rng.random((h, w)) < 0.01
    values = np.zeros((h, w, 3), np.float32)
    values[mask] = rng.random((int(mask.sum()), 3)) * 0.8 + 0.1
    out = fill.push_pull(values, mask)
    assert (out > 0).all(axis=-1).all(), "a texel was left empty"
    np.testing.assert_array_equal(out[mask], values[mask])
    assert out.min() >= 0.1 - 1e-6 and out.max() <= 0.9 + 1e-6    # only averages of real texels

    vectors = np.zeros((h, w, 3), np.float32)
    vectors[mask] = rng.normal(size=(int(mask.sum()), 3))
    vectors[mask] /= np.linalg.norm(vectors[mask], axis=1, keepdims=True)
    unit = fill.push_pull_unit(vectors, mask)
    np.testing.assert_allclose(np.linalg.norm(unit, axis=-1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------- encode


def _random_frames(count: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Skewed, unnormalised interpolants and mixed signs, like real interpolated frames."""
    normal = rng.normal(size=(count, 3))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    tangent = rng.normal(size=(count, 3))
    tangent -= normal * np.sum(tangent * normal, axis=1, keepdims=True) * rng.uniform(0.2, 1.0, (count, 1))
    normal *= rng.uniform(0.4, 1.0, (count, 1))
    tangent *= rng.uniform(0.3, 1.5, (count, 1))
    sign = np.where(rng.random(count) < 0.5, -1.0, 1.0)
    world = normal / np.linalg.norm(normal, axis=1, keepdims=True) + rng.normal(scale=0.4, size=(count, 3))
    return tangent, normal, sign, world


def test_mikk_encode_decode_identity() -> None:
    tangent, normal, sign, world = _random_frames(20_000, np.random.default_rng(11))
    ts = encode.mikk_encode(world, tangent, normal, sign)
    np.testing.assert_allclose(np.linalg.norm(ts, axis=1), 1.0, atol=1e-12)
    err = _angle_deg(encode.mikk_decode(ts, tangent, normal, sign), world)
    assert err.max() < 0.01, f"mikk round trip max {err.max()} deg"
    assert (sign < 0).any() and (sign > 0).any()


def test_per_vertex_encode_decode_identity() -> None:
    rng = np.random.default_rng(12)
    count = 20_000
    # Per triangle one frame, bent differently at each of its three vertices.
    normal, tangent = rng.normal(size=(count, 1, 3)), rng.normal(size=(count, 1, 3))
    normals = normal + 0.4 * rng.normal(size=(count, 3, 3)) * np.linalg.norm(normal, axis=2, keepdims=True)
    tangents = np.concatenate([tangent + 0.4 * rng.normal(size=(count, 3, 3)), np.ones((count, 3, 1))], axis=2)
    sign = np.where(rng.random(count) < 0.5, -1.0, 1.0)
    tangents[..., 3] = sign[:, None]
    t, b, n = encode.per_vertex_basis(normals.reshape(-1, 3), tangents.reshape(-1, 4))
    bary = rng.dirichlet((1.0, 1.0, 1.0), count)
    ti, bi, ni = (np.einsum("nk,nkd->nd", bary, x.reshape(count, 3, 3)) for x in (t, b, n))
    world = rng.normal(size=(count, 3))
    ts = encode.per_vertex_encode(world, ti, bi, ni, sign)
    err = _angle_deg(encode.per_vertex_decode(ts, ti, bi, ni), world)
    assert err.max() < 0.01, f"per_vertex round trip max {err.max()} deg"


def test_degenerate_frames_are_orthonormalised() -> None:
    tangent = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 0.0]])
    normal = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])   # tangent along the normal
    world = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    ts = encode.mikk_encode(world, tangent, normal, np.array([1.0, -1.0]))
    np.testing.assert_allclose(ts, [[0, 0, 1], [0, 0, 1]], atol=1e-12)


def test_clamp_and_quantize() -> None:
    ts = np.array([[0.0, 0.0, 1.0], [0.6, 0.0, -0.8], [0.0, 0.0, -1.0]])
    clamped = encode.clamp_hemisphere(ts)
    np.testing.assert_allclose(clamped, [[0, 0, 1], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    assert encode.quantize(clamped[:1]).tolist() == [[128, 128, 255]]
    np.testing.assert_allclose(encode.dequantize(np.array([[128, 128, 255]])), [[1 / 255, 1 / 255, 1.0]])


# ---------------------------------------------------------------------- tangents


def test_mikk_matches_reference_wasm() -> None:
    ref = json.loads((DATA / "mikk_reference.json").read_text(encoding="utf-8"))
    faces = np.array(ref["faces"])
    result = tangents.mikk_tangents(np.array(ref["positions"]), np.array(ref["normals"]),
                                    np.array(ref["uv"]), faces)
    expected = np.array(ref["tangents"]).reshape(-1, 3, 4)
    err = _angle_deg(result.corners[..., :3].reshape(-1, 3), expected[..., :3].reshape(-1, 3))
    assert err.max() < 0.01, f"tangent angle max {err.max()} deg"
    assert np.array_equal(np.sign(result.corners[..., 3]), np.sign(expected[..., 3]))
    assert (expected[..., 3] < 0).any() and (expected[..., 3] > 0).any()
    assert result.degenerate == 0 and result.fallback == 0


def test_mikk_uv_degenerate_triangle_borrows_its_vertex_tangent() -> None:
    positions = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    normals = np.tile([0.0, 0.0, 1.0], (4, 1))
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])   # vertex 3 on top of vertex 2
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    result = tangents.mikk_tangents(positions, normals, uv, faces)
    assert result.degenerate == 1
    np.testing.assert_allclose(result.corners[0, :, :3], [[1, 0, 0]] * 3, atol=1e-12)
    np.testing.assert_allclose(result.corners[1, [0, 2], :3], [[1, 0, 0]] * 2, atol=1e-12)
    assert result.fallback == 1      # vertex 3 has no other triangle: any perpendicular vector
    assert abs(result.corners[1, 1, :3] @ [0, 0, 1]) < 1e-12


# ---------------------------------------------------------------------- low poly


def test_read_obj_fans_polygons_like_uv_py(tmp_path: Path) -> None:
    obj = tmp_path / "low.obj"
    obj.write_text(
        "# comment\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0.5 1.5 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvt 0.5 1\n"
        "f 1/1 2/2 3/3 5/5 4/4\n"          # pentagon
        "f -5/-5 -4/-4 -3/-3\n"             # negative indices = vertices 1 2 3
        "f 1//1 2//1 4//1\n",               # no texture coordinates
        encoding="utf-8")
    low = lowpoly.read_obj(obj)
    assert low.polygons == 3
    assert low.tri_pos.tolist() == [[0, 1, 2], [0, 2, 4], [0, 4, 3], [0, 1, 2], [0, 1, 3]]
    assert low.tri_uv.tolist() == [[0, 1, 2], [0, 2, 4], [0, 4, 3], [0, 1, 2], [-1, -1, -1]]

    uv = pytest.importorskip("instant_meshes_brush.uv")
    quads = np.array([[0, 1, 2, 3], [4, 5, 6, 6]])
    fan, _, _ = uv._triangulate(quads)
    faces = [[(int(c), 0) for c in (q[:3] if q[3] == q[2] else q)] for q in quads]   # as uv.write_obj
    _write_obj(obj, np.arange(21.0).reshape(7, 3), np.zeros((1, 2)), faces)
    assert sorted(map(tuple, lowpoly.read_obj(obj).tri_pos.tolist())) == sorted(map(tuple, fan.tolist()))


def test_read_obj_welds_copies_of_a_point(tmp_path: Path) -> None:
    """One point written twice (Instant Meshes does this) is one position: the cage normal is
    not split there, and a triangle repeated over the copy is a duplicate for clean()."""
    obj = tmp_path / "low.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 0 0\nv 1 1 0\n"
                   "vt 0 0\nvt 1 0\nvt 0 1\nvt 1 1\n"
                   "f 1/1 2/2 3/3\nf 4/2 5/4 3/3\nf 1/1 4/2 3/3\n", encoding="utf-8")
    low = lowpoly.read_obj(obj)
    assert low.positions.tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    assert low.tri_pos.tolist() == [[0, 1, 2], [1, 3, 2], [0, 1, 2]]
    cleaned, counts = lowpoly.clean(low, diag=1.0)
    assert cleaned.tri_pos.tolist() == [[0, 1, 2], [1, 3, 2]] and counts["dropped_triangles"] == 1


def test_read_obj_reports_broken_files(tmp_path: Path) -> None:
    obj = tmp_path / "low.obj"
    obj.write_text("v 0 0 0\nf 1 2 3\n", encoding="utf-8")
    with pytest.raises(BakeError, match="Run Retopo again"):
        lowpoly.read_obj(obj)
    obj.write_text("v 0 0 0\n", encoding="utf-8")
    with pytest.raises(BakeError, match="no faces"):
        lowpoly.read_obj(obj)


def test_clean_drops_untextured_flat_and_repeated_triangles() -> None:
    positions = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]])
    tri_pos = np.array([[0, 1, 2], [2, 0, 1], [0, 1, 3], [0, 2, 1], [1, 2, 0]])
    tri_uv = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, -1, 2]])
    low = lowpoly.LowInput(positions, np.zeros((3, 2)), tri_pos, tri_uv)
    cleaned, counts = lowpoly.clean(low, diag=3.0)
    assert cleaned.tri_pos.tolist() == [[0, 1, 2]]
    assert counts == {"untextured_triangles": 1, "dropped_triangles": 3}


def _box_scene(half: float, threads: int = 1) -> rays.Scene:
    """A closed axis-aligned box, outward-facing triangles."""
    corners = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = np.array([f for a, b, c, d in quads for f in ((a, b, c), (a, c, d))])
    return rays.Scene(corners, faces, threads=threads)


def test_winding_vote_flips_an_inward_component() -> None:
    points, faces = _icosphere(3)
    sphere = rays.Scene(points * 1.02, faces)
    region = points[faces].mean(axis=1)[:, 0] > 0.4          # one cap, wound inwards
    bad = np.where(region[:, None], faces[:, [0, 2, 1]], faces)
    low = lowpoly.LowInput(points, np.zeros((len(points), 2)), bad, bad.copy())
    labels = lowpoly.consistent_components(bad)
    assert len(np.unique(labels)) == 2
    fixed, flipped = lowpoly.fix_winding(low, sphere)
    assert flipped == int(region.sum())
    p = points[fixed.tri_pos]
    outward = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p.mean(axis=1))
    assert (outward > 0).all()
    assert np.array_equal(fixed.tri_uv, fixed.tri_pos)


def test_export_normals_split_only_sharp_uv_seams() -> None:
    # A unit cube: every face its own UV island, so every edge is a UV seam at 90 degrees.
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tri_pos = np.array([f for a, b, c, d in quads for f in ((a, b, c), (a, c, d))])
    tri_uv = np.array([[4 * q + k for k in f] for q in range(6) for f in ((0, 1, 2), (0, 2, 3))])
    group, normals = lowpoly.export_normals(corners, tri_pos, tri_uv, 60.0)
    face_n = np.cross(corners[tri_pos[:, 1]] - corners[tri_pos[:, 0]], corners[tri_pos[:, 2]] - corners[tri_pos[:, 0]])
    face_n /= np.linalg.norm(face_n, axis=1, keepdims=True)
    np.testing.assert_allclose(normals[group], np.repeat(face_n[:, None], 3, axis=1), atol=1e-12)
    assert len(np.unique(group)) == 24                           # 8 corners x 3 faces

    group, normals = lowpoly.export_normals(corners, tri_pos, tri_uv, 95.0)   # nothing that sharp
    assert len(np.unique(group)) == 8
    group, normals = lowpoly.export_normals(corners, tri_pos, tri_uv, -1.0)   # all smooth: C
    np.testing.assert_allclose(np.abs(normals[group]), 1 / math.sqrt(3), atol=1e-12)

    # A sharp edge that is not a UV seam stays smooth.
    same_uv = tri_pos.copy()
    group, _ = lowpoly.export_normals(corners, tri_pos, same_uv, 60.0)
    assert len(np.unique(group)) == 8


def test_render_vertices_split_by_uv_orientation() -> None:
    points, faces = _grid(2)
    uvs = points[:, :2] * 0.5
    uvs[[2, 5, 8], 0] = 0.0                                          # fold the right column back: mirrored
    group = faces.copy()
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    mesh, stats = lowpoly.build_render_mesh(points, uvs, faces, faces, normals, group, normals)
    area = tangents.uv_area2(mesh.uv, mesh.faces)
    assert (area > 0).any() and (area < 0).any()
    assert np.array_equal(mesh.sign, np.where(area > 0, 1.0, -1.0))
    for v in range(len(mesh.positions)):                              # each vertex has one orientation
        users = np.flatnonzero((mesh.faces == v).any(axis=1))
        assert len(np.unique(np.sign(area[users]))) == 1
    assert len(mesh.positions) > len(points)
    assert stats["tangent_fallback_corners"] == 0


def test_repack_keeps_island_shapes_and_opens_gaps() -> None:
    import xatlas

    points, faces = _icosphere(2)
    atlas = xatlas.Atlas()
    atlas.add_mesh(points.astype(np.float32), faces.astype(np.uint32))
    atlas.generate()                                                  # no padding, as uv.py does
    vmap, tri_uv, uvs = (np.asarray(x) for x in atlas[0])
    tri_uv = tri_uv.astype(np.int64)
    uvs = uvs.astype(np.float64)
    uvs[:, 1] = 1 - uvs[:, 1]
    tri_pos = np.asarray(vmap, np.int64)[tri_uv]
    before = lowpoly.min_chart_gap(uvs, tri_uv, 256)
    new_uv, new_tri = lowpoly.repack(points, tri_pos, tri_uv, uvs, points, 256)
    after = lowpoly.min_chart_gap(new_uv, new_tri, 256)
    assert after >= 2 and after > before
    assert new_uv.min() >= 0 and new_uv.max() <= 1
    charts = lowpoly.chart_labels(new_tri)
    old_area = np.abs(tangents.uv_area2(uvs, tri_uv))
    new_area = np.abs(tangents.uv_area2(new_uv, new_tri))
    scale = np.bincount(charts, new_area) / np.bincount(charts, old_area)
    np.testing.assert_allclose(new_area / (old_area * scale[charts]), 1.0, atol=2e-3)


def _glb(path: Path, gltf: dict, binary: bytes) -> None:
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    binary += b"\0" * (-len(binary) % 4)
    with path.open("wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, 28 + len(js) + len(binary)))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(binary), 0x004E4942) + binary)


def test_read_glb_applies_node_transforms(tmp_path: Path) -> None:
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    nrm = np.tile(np.array([0, 0, 1], np.float32), (3, 1))
    uv = np.array([[0, 1], [1, 1], [0, 0]], np.float32)      # glTF t runs down
    idx = np.array([0, 1, 2], np.uint16)
    binary = pos.tobytes() + nrm.tobytes() + uv.tobytes() + idx.tobytes()
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": 36}, {"buffer": 0, "byteOffset": 36, "byteLength": 36},
             {"buffer": 0, "byteOffset": 72, "byteLength": 24}, {"buffer": 0, "byteOffset": 96, "byteLength": 6}]
    accessors = [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                 {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
                 {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
                 {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"}]
    prim = {"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3}
    gltf = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"children": [1], "translation": [0, 0, 5]},
                      {"mesh": 0, "scale": [2, 2, -1]},                  # mirrors: winding flips
                      {"mesh": 0}],                                      # not in the scene
            "meshes": [{"primitives": [prim]}], "accessors": accessors, "bufferViews": views,
            "buffers": [{"byteLength": len(binary)}]}
    path = tmp_path / "low.glb"
    _glb(path, gltf, binary)
    low = lowpoly.read_glb(path)
    world = low.positions[low.tri_pos[0]]
    np.testing.assert_allclose(world, [[0, 0, 5], [0, 2, 5], [2, 0, 5]])
    np.testing.assert_allclose(low.uvs[low.tri_uv[0]], [[0, 0], [0, 1], [1, 0]])
    np.testing.assert_allclose(low.normals, [[0, 0, -1]] * 3)
    assert low.polygons == 1


def test_prepare_a_retopo_obj(tmp_path: Path) -> None:
    import xatlas

    points, faces = _icosphere(2)
    atlas = xatlas.Atlas()
    atlas.add_mesh(points.astype(np.float32), faces.astype(np.uint32))
    atlas.generate()
    vmap, tri_uv, uvs = (np.asarray(x) for x in atlas[0])
    uvs = uvs.astype(np.float64)
    uvs[:, 1] = 1 - uvs[:, 1]
    obj = tmp_path / "retopo.obj"
    corners = [[(int(vmap[t]), int(t)) for t in tri] for tri in tri_uv]
    corners[5] = corners[5][::-1]                                     # one inward triangle
    _write_obj(obj, points, uvs, corners + [corners[0]])              # and one repeated
    high, high_faces = _icosphere(4)
    with rays.Scene(high * 1.01, high_faces) as scene:
        low = lowpoly.prepare(obj, scene, resolution=256)
        assert low.repacked and low.ring == lowpoly.RING_REPACKED
        assert low.stats["flipped_triangles"] == 1 and low.stats["dropped_triangles"] == 1
        assert low.stats["triangles"] == len(faces) and low.stats["min_chart_gap_px"] >= 2
        mesh = low.mesh
        np.testing.assert_allclose(np.linalg.norm(mesh.tangents[:, :3], axis=1), 1.0)
        assert np.all(np.abs(np.einsum("ij,ij->i", mesh.tangents[:, :3], mesh.normals)) < 1e-9)
        # The cage normal is continuous: every copy of a position carries the same one.
        for pid in np.unique(mesh.position_id)[:50]:
            same = mesh.cage[mesh.position_id == pid]
            assert np.allclose(same, same[0])
        surface = mesh.surface(np.array([0, 1]), np.full((2, 3), 1 / 3), 512)
        assert surface.position.shape == (2, 3) and (surface.footprint > 0).all()
        d = rays.auto_distance(scene, *mesh.probes())
        assert 0.002 * scene.diag <= d <= 0.05 * scene.diag


# ---------------------------------------------------------------------- rays


def test_first_front_hit_skips_back_faces() -> None:
    # Two parallel squares: z = 1 faces down (-z), z = 0 faces up (+z).
    square = np.array([[-1.0, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    up = np.array([[0, 1, 2], [0, 2, 3]])
    positions = np.concatenate([square, square + [0, 0, 1]])
    faces = np.concatenate([up, up[:, ::-1] + 4])
    origin, down = np.array([[0.1, 0.2, 2.0]]), np.array([[0.0, 0.0, -1.0]])
    with rays.Scene(positions, faces) as scene:
        hit = scene.first_front_hit(origin, down, 5.0)
        assert hit.prim[0] in (0, 1) and hit.t[0] == pytest.approx(2.0, abs=1e-6) and not hit.back[0]
        assert not scene.first_front_hit(origin, down, 1.5).hit[0]
        near = scene.nearest_bidirectional(np.array([[0.1, 0.2, 0.7]]), down, 5.0)
        assert near.prim[0] in (2, 3) and near.t[0] == pytest.approx(-0.3, abs=1e-6)
        assert scene.occluded(origin, down, 1.5).tolist() == [True]
        assert scene.occluded(origin, -down, 5.0).tolist() == [False]
    double = np.array([False, False, True, True])
    with rays.Scene(positions, faces, double_sided=double) as scene:
        hit = scene.first_front_hit(origin, down, 5.0)
        assert hit.prim[0] in (2, 3) and hit.t[0] == pytest.approx(1.0, abs=1e-6) and hit.back[0]


def test_cage_prefers_the_outer_surface() -> None:
    """A raised band (a hoop) over a base plank: under the band the cage ray must land on the
    band, not on the plank below it, and elsewhere on the plank."""
    plank, plank_faces = _grid(8)
    band = np.array([[0.4, -0.1, 0.05], [0.6, -0.1, 0.05], [0.6, 1.1, 0.05], [0.4, 1.1, 0.05]])
    positions = np.concatenate([plank, band])
    faces = np.concatenate([plank_faces, np.array([[0, 1, 2], [0, 2, 3]]) + len(plank)])
    x = np.arange(180) * 0.005 + 0.0525                              # never on the band's edges
    points = np.stack([x, np.full_like(x, 0.5), np.zeros_like(x)], axis=1)
    cage = np.tile([0.0, 0.0, 1.0], (len(x), 1))
    with rays.Scene(positions, faces) as scene:
        hits = rays.cage_hits(scene, points, cage, 0.1)
    on_band = hits.prim >= len(plank_faces)
    under = (x > 0.4) & (x < 0.6)
    assert np.array_equal(on_band, under)
    np.testing.assert_allclose(hits.height[under], 0.05, atol=1e-6)
    np.testing.assert_allclose(hits.height[~under], 0.0, atol=1e-6)
    assert not hits.fallback.any()


def _closed_box(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """An axis-aligned box, 12 triangles wound outwards."""
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    p = corners[faces]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    inward = np.einsum("ij,ij->i", normal, p.mean(axis=1) - corners.mean(axis=0)) < 0
    faces[inward] = faces[inward][:, ::-1]
    return corners, faces


@pytest.mark.parametrize("bulge", [0.005, -0.005])
def test_cage_starts_past_other_parts_of_the_low_poly(bulge: float) -> None:
    """A finger (a box over the plank, z 0.03..0.07) lies over a plank, both in the low poly
    and in the high poly, whose finger is a little fatter (bulge > 0) or thinner than the
    low's. Under the finger the plain cage ray takes the finger's top; with the low poly
    given, the plank is baked there, and the hoop over the planks (high relief with no low
    part of its own) still takes the hoop."""
    plank, plank_faces = _grid(8)
    finger, finger_faces = _closed_box((0.4, -0.1, 0.03), (0.6, 1.1, 0.07))
    high_finger, _ = _closed_box((0.4 - bulge, -0.1, 0.03 - bulge), (0.6 + bulge, 1.1, 0.07 + bulge))
    low_positions = np.concatenate([plank, finger])
    low_faces = np.concatenate([plank_faces, finger_faces + len(plank)])
    high_positions = np.concatenate([plank, high_finger])
    x = np.arange(180) * 0.005 + 0.0525                              # never on the finger's edges
    points = np.stack([x, np.full_like(x, 0.5), np.zeros_like(x)], axis=1)
    cage = np.tile([0.0, 0.0, 1.0], (len(x), 1))
    with rays.Scene(high_positions, low_faces) as scene, rays.Scene(low_positions, low_faces) as low:
        plain = rays.cage_hits(scene, points, cage, 0.1)
        parted = rays.cage_hits(scene, points, cage, 0.1, low=low)
    covered = (x > 0.4) & (x < 0.6)                                   # under the low poly's finger
    assert (plain.prim[(x > 0.41) & (x < 0.59)] >= len(plank_faces)).all()         # the ghost
    assert (parted.prim[covered] < len(plank_faces)).all()                          # the plank instead
    np.testing.assert_allclose(parted.height[covered], 0.0, atol=1e-6)
    assert parted.hit.all() and not parted.fallback.any()
    # Beyond the low finger (where a fatter high finger pokes out) nothing tells the parts
    # apart: the outermost rule holds there as before.
    np.testing.assert_array_equal(parted.prim[~covered], plain.prim[~covered])

    # The hoop: relief of the high poly over a low poly that is the plank alone.
    band = np.array([[0.4, -0.1, 0.05], [0.6, -0.1, 0.05], [0.6, 1.1, 0.05], [0.4, 1.1, 0.05]])
    hoop_faces = np.concatenate([plank_faces, np.array([[0, 1, 2], [0, 2, 3]]) + len(plank)])
    with rays.Scene(np.concatenate([plank, band]), hoop_faces) as scene, rays.Scene(plank, plank_faces) as low:
        hoop = rays.cage_hits(scene, points, cage, 0.1, low=low)
    assert np.array_equal(hoop.prim >= len(plank_faces), (x > 0.4) & (x < 0.6))


def _closed_prism(profile: list[tuple[float, float]], y0: float, y1: float) -> tuple[np.ndarray, np.ndarray]:
    """A prism along y over a convex (x, z) cross-section, its triangles wound outwards."""
    n = len(profile)
    ring = np.array([[x, 0.0, z] for x, z in profile])
    corners = np.concatenate([ring + [0, y0, 0], ring + [0, y1, 0]])
    faces = [(k, (k + 1) % n, n + (k + 1) % n) for k in range(n)] + [(k, n + (k + 1) % n, n + k) for k in range(n)]
    faces += [(0, k, k + 1) for k in range(1, n - 1)] + [(n, n + k + 1, n + k) for k in range(1, n - 1)]
    faces = np.array(faces)
    p = corners[faces]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    inward = np.einsum("ij,ij->i", normal, p.mean(axis=1) - corners.mean(axis=0)) < 0
    faces[inward] = faces[inward][:, ::-1]
    return corners, faces


def test_cage_leaves_hits_claimed_by_other_parts() -> None:
    """The low finger of the test above (a box over the plank, x 0.4..0.6); its high poly is a
    little fatter, with chamfered top edges that overhang the plank beside it (x 0.39..0.4 and
    0.6..0.61). There the cage ray passes beside the low finger and meets the chamfer: the
    finger's side, a streak on the plank. With ``claims`` the low finger, 0.004-0.011 from the
    chamfer, claims it and the plank is baked; the hoop over the planks keeps the hoop."""
    plank, plank_faces = _grid(8)
    finger, finger_faces = _closed_box((0.4, -0.1, 0.03), (0.6, 1.1, 0.07))
    chamfered, chamfered_faces = _closed_prism(
        [(0.39, 0.025), (0.61, 0.025), (0.61, 0.06), (0.59, 0.08), (0.41, 0.08), (0.39, 0.06)], -0.1, 1.1)
    low_positions = np.concatenate([plank, finger])
    low_faces = np.concatenate([plank_faces, finger_faces + len(plank)])
    high_positions = np.concatenate([plank, chamfered])
    high_faces = np.concatenate([plank_faces, chamfered_faces + len(plank)])
    x = np.arange(180) * 0.005 + 0.0525
    points = np.stack([x, np.full_like(x, 0.5), np.zeros_like(x)], axis=1)
    cage = np.tile([0.0, 0.0, 1.0], (len(x), 1))
    with rays.Scene(high_positions, high_faces) as scene, rays.Scene(low_positions, low_faces) as low:
        parted = rays.cage_hits(scene, points, cage, 0.1, low=low)
        claimed = rays.cage_hits(scene, points, cage, 0.1, low=low, claims=low)
        tall = parted.height > rays.PART_MARGIN * 0.1                               # the only ones tested
        flags = rays.claimed_by_other_parts(low, scene, points[tall], cage[tall], parted.prim[tall],
                                            parted.height[tall])
    beside = ((x > 0.39) & (x < 0.4)) | ((x > 0.6) & (x < 0.61))
    assert beside.sum() == 4 and np.array_equal(tall, beside)
    assert (parted.prim[beside] >= len(plank_faces)).all()                          # the streak
    assert (parted.height[beside] > 0.06).all() and flags.all()
    assert (claimed.prim < len(plank_faces)).all() and claimed.hit.all()             # the plank everywhere
    np.testing.assert_allclose(claimed.height, 0.0, atol=1e-6)
    assert not claimed.fallback.any()
    np.testing.assert_array_equal(claimed.prim[~beside], parted.prim[~beside])

    # The hoop: relief with no low part of its own; only the plank lies under it, 0.05 away.
    band = np.array([[0.4, -0.1, 0.05], [0.6, -0.1, 0.05], [0.6, 1.1, 0.05], [0.4, 1.1, 0.05]])
    hoop_faces = np.concatenate([plank_faces, np.array([[0, 1, 2], [0, 2, 3]]) + len(plank)])
    with rays.Scene(np.concatenate([plank, band]), hoop_faces) as scene, rays.Scene(plank, plank_faces) as low:
        plain = rays.cage_hits(scene, points, cage, 0.1)
        hoop = rays.cage_hits(scene, points, cage, 0.1, low=low, claims=low)
    assert np.array_equal(hoop.prim >= len(plank_faces), (x > 0.4) & (x < 0.6))
    np.testing.assert_array_equal(hoop.prim, plain.prim)
    np.testing.assert_array_equal(hoop.height, plain.height)


def test_cage_falls_back_to_the_nearest_hit() -> None:
    with _box_scene(1.0) as scene:
        points = np.array([[0.0, 0.0, 0.9]])                         # inside, cage pointing down
        hits = rays.cage_hits(scene, points, np.array([[0.0, 0.0, -1.0]]), 0.05)
        assert hits.hit[0] and hits.fallback[0]
        assert hits.height[0] == pytest.approx(-0.1, abs=1e-6)       # the top face, behind P


def test_auto_distance_follows_the_deviation() -> None:
    points, faces = _icosphere(3)
    with rays.Scene(points * 1.03, faces) as scene:
        normals = points / np.linalg.norm(points, axis=1, keepdims=True)
        d = rays.auto_distance(scene, points, normals)
        assert d == pytest.approx(1.25 * 0.03, rel=1e-4)
        far = rays.auto_distance(scene, points * 3, normals)
        assert far == pytest.approx(0.05 * scene.diag)


def test_ray_threads_agree_and_scale() -> None:
    points, faces = _icosphere(6)                                     # 81920 triangles
    faces = faces[:, ::-1]                                            # facing inwards, at the rays
    rng = np.random.default_rng(1)
    origins = rng.uniform(-0.3, 0.3, size=(1_200_000, 3))
    directions = rng.normal(size=origins.shape)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    timings = {}
    results = {}
    for threads in (1, 4):
        with rays.Scene(points, faces, threads=threads) as scene:
            best = math.inf
            for _ in range(3):
                start = time.perf_counter()
                results[threads] = scene.first_front_hit(origins, directions, 3.0)
                best = min(best, time.perf_counter() - start)
            timings[threads] = best
    assert np.array_equal(results[1].prim, results[4].prim)
    np.testing.assert_allclose(results[1].t, results[4].t)
    assert results[1].hit.all()
    if (os.cpu_count() or 1) >= 4:
        assert timings[1] / timings[4] > 1.3, f"4 threads: {timings[1] / timings[4]:.2f}x"


# ---------------------------------------------------------------------- AO


def test_hammersley_and_cosine_directions() -> None:
    xi = ao.hammersley(64)
    assert np.array_equal(np.sort(xi[:, 1]), np.arange(64) / 64)
    normal = np.array([[0.0, 0.6, 0.8]])
    directions = ao.cosine_directions(normal, ao.hammersley(4096)[None])
    np.testing.assert_allclose(np.linalg.norm(directions, axis=-1), 1.0, atol=1e-12)
    cos = directions[0] @ normal[0]
    assert cos.min() >= 0 and cos.mean() == pytest.approx(2 / 3, abs=2e-3)
    assert ao.rays_per_sample(1) == 32 and ao.rays_per_sample(2) == 8 and ao.rays_per_sample(3) == 4


def test_ao_of_an_open_plane_and_beside_a_wall() -> None:
    ground, ground_faces = _grid(1, 100.0)
    ground -= [50.0, 50.0, 0.0]
    x = np.array([0.25, 0.5, 1.0, 2.0])
    count = 4000
    points = np.repeat(np.stack([-x, np.zeros_like(x), np.zeros_like(x)], axis=1), count, axis=0)
    up = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    texel = np.arange(len(points))
    with rays.Scene(ground, ground_faces) as scene:
        open_plane = ao.ambient_occlusion(scene, points, up, up, texel, texel % 4, supersample=2,
                                          max_distance=1e9)
    assert (open_plane == 1.0).all()

    height = 1.0                                                      # a wall 100 H long at x = 0
    wall = np.array([[0.0, -50, 0], [0.0, 50, 0], [0.0, 50, height], [0.0, -50, height]])
    positions = np.concatenate([ground, wall])
    faces = np.concatenate([ground_faces, np.array([[0, 1, 2], [0, 2, 3]]) + 4])
    with rays.Scene(positions, faces, threads=2) as scene:
        beside = ao.ambient_occlusion(scene, points, up, up, texel, texel % 4, supersample=2,
                                      max_distance=1e9).reshape(len(x), count).mean(axis=1)
        lifted = height - 1e-4 * scene.diag                          # rays start that far up
    # Cosine-weighted visibility beside an infinite wall of height H: 1/2 + x / (2 sqrt(x^2 + H^2)).
    expected = 0.5 + x / (2 * np.sqrt(x * x + lifted * lifted))
    np.testing.assert_allclose(beside, expected, atol=0.01)


def test_dilate_grows_every_island_by_two_texels_from_its_own_colours() -> None:
    """fill.dilate: the two texels around a covered square take the mean of their covered neighbours, ring by ring;
    the covered texels keep their values; the far texels are left to the push-pull."""
    values = np.zeros((9, 9, 2), dtype=np.float32)
    covered = np.zeros((9, 9), dtype=bool)
    covered[3:6, 3:6] = True
    values[3:6, 3:6, 0] = 0.5
    values[3:6, 3:6, 1] = np.arange(9, dtype=np.float32).reshape(3, 3) / 8
    values[0, 0] = (7.0, 7.0)                                   # an outside value: left as it is
    out, grown = fill.dilate(values, covered)
    assert grown[1:8, 1:8].all() and not grown[0].any() and not grown[:, 0].any()   # two rings grew
    np.testing.assert_array_equal(out[3:6, 3:6], values[3:6, 3:6])                   # the island untouched
    np.testing.assert_allclose(out[2, 2:7, 0], 0.5)                                   # the first ring: the island's colour
    np.testing.assert_allclose(out[1, 1:8, 0], 0.5)                                   # the second too
    assert abs(out[2, 4, 1] - np.mean([0, 1, 2]) / 8) < 1e-6                          # the mean of its covered neighbours
    np.testing.assert_array_equal(out[0, 0], (7.0, 7.0))
    same, mask = fill.dilate(values, covered, radius=0)
    np.testing.assert_array_equal(same, values) and np.testing.assert_array_equal(mask, covered)
    flat, _ = fill.dilate(values[..., 0], covered)                                   # (H, W) works too
    np.testing.assert_allclose(flat[1:8, 1:8], 0.5)
