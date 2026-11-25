"""The Pixal3D launcher's own logic: multi-view requests and their validation, the camera rig, the size
a small picture or slice is enlarged to (1024 px on its largest side), the framing of the views (one scale
per picture), stray bits of a slice, the multi-view weights check, the model-mode switch, the
sparse-structure guard, the inner-copy and streak checks, the GLB orientation and the /spellcast/* answers:
the batches (their record, their order, cancel, failures, what the engine is asked for and what its hooks
do) around a stand-in for the fork's staged engine, and the reference route with its models loaded on first use.

Runs in the hub venv (fastapi + pydantic, no torch / numpy / PIL): launchers/pixal3d_server.py,
pixal3d_batch.py, pixal3d_mv.py and pixal3d_guard.py keep those imports inside functions. What needs numpy,
PIL, OpenCV, torch or safetensors is skipped there and runs in the pixal3d venv (the weight swap on stand-in
models, the input size, the framing and the stray bits on made-up pictures, the ring growth, the hooks'
picture, camera and mesh work on the CPU); the GPU paths (the real engine and pipeline, the rasterized views,
cuBVH) run only on the GPU (profiling/p3d_checks/check_launcher.py).
"""

from __future__ import annotations

import ast
import concurrent.futures
import dataclasses
import json
import shutil
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spellcast import paths

if str(paths.LAUNCHERS) not in sys.path:     # the launchers import their siblings as top-level modules
    sys.path.insert(0, str(paths.LAUNCHERS))

import pixal3d_batch as batch   # noqa: E402
import pixal3d_guard as guard   # noqa: E402
import pixal3d_mv as mv         # noqa: E402
import pixal3d_server as server  # noqa: E402

EXAMPLE_RIG = paths.PIXAL3D_DIR / "assets" / "mv_images" / "example" / "transforms.json"


def view(image_path: Path | str, azimuth: int = 0, rect: list[int] | None = None) -> server.ViewSpec:
    return server.ViewSpec(image_path=str(image_path), azimuth=azimuth, rect=rect)


def request(root: Path, views: list[server.ViewSpec] | None = None, **extra: Any) -> server.GenerateRequest:
    return server.GenerateRequest(image_path=str(root / "images" / "0001.png"),
                                  out_path=str(root / "mesh" / "mesh.partial.glb"), views=views, **extra)


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "images").mkdir()
    for n in range(1, 5):
        (tmp_path / "images" / f"000{n}.png").write_bytes(b"png")
    return tmp_path


# ---- requests ----------------------------------------------------------------------------------------

def test_views_ride_along_in_the_generate_request() -> None:
    req = server.GenerateRequest.model_validate({
        "image_path": "a.png", "out_path": "b.glb", "views_dir": "v",
        "views": [{"image_path": "a.png", "rect": [1, 2, 30, 40], "azimuth": 90}, {"image_path": "b.png"}]})
    assert req.views is not None and req.views[0].rect == [1, 2, 30, 40] and req.views[0].azimuth == 90
    assert req.views[1].rect is None and req.views[1].azimuth == 0
    assert server.GenerateRequest(image_path="a.png", out_path="b.glb").views is None   # the old body still works


def test_multi_view_requests_are_checked_in_artist_words(project: Path) -> None:
    root = project.resolve()
    img = [root / "images" / f"000{n}.png" for n in range(1, 5)]
    ok = [view(img[0], 0, [0, 0, 64, 64]), view(img[1], 90), view(img[2], 180), view(img[3], 270)]
    assert server.validate(request(root, ok, views_dir=str(root / "mesh" / "views")), root) is None
    assert server.validate(request(root, [view(img[1], 90)]), root) is None          # one slice: any direction
    assert server.validate(request(root, [view(img[1], 90), view(img[2], 180)]), root) == mv.FRONT_REQUIRED
    assert "marked Front" in server.validate(request(root, [view(img[0], 0), view(img[1], 0)]), root)
    assert "direction" in server.validate(request(root, [view(img[0], 45)]), root)
    for rect in ([0, 0, 10], [-1, 0, 10, 10], [0, 0, 0, 10]):
        assert "rectangle" in server.validate(request(root, [view(img[0], 0, rect)]), root)
    assert "not found" in server.validate(request(root, [view(root / "images" / "gone.png", 0)]), root)
    outside = project.parent / "elsewhere.png"
    assert "own projects folder" in server.validate(request(root, [view(outside, 0)]), root)
    assert "own projects folder" in server.validate(request(root, ok, views_dir=str(project.parent)), root)
    # the picture itself is not needed when slices are given
    req = request(root, ok)
    req.image_path = str(root / "images" / "missing.png")
    assert server.validate(req, root) is None


def test_the_front_view_goes_first() -> None:
    views = [types.SimpleNamespace(azimuth=a) for a in (270, 180, 0, 90)]
    assert [v.azimuth for v in mv.ordered(views)] == [0, 90, 180, 270]


# ---- the camera rig and the orientation --------------------------------------------------------------

def test_camera_poses_reproduce_the_example_rig() -> None:
    if not EXAMPLE_RIG.is_file():
        pytest.skip("Pixal3D is not checked out")
    rig = json.loads(EXAMPLE_RIG.read_text(encoding="utf-8"))
    assert mv.MV_CAMERA_ANGLE_X == pytest.approx(rig["camera_angle_x"], abs=1e-12)
    for frame in rig["frames"]:
        ours = mv.camera_to_world(int(frame["name"][len("azim"):]))
        for row, expected in zip(ours, frame["transform_matrix"]):
            assert row == pytest.approx(expected, abs=1e-6)


def test_the_front_view_camera_is_pixal3ds_fixed_front_camera() -> None:
    d = mv.MV_CAMERA_DISTANCE
    assert mv.camera_to_world(0) == [[1, 0, 0, 0], [0, 0, -1, -d], [0, 1, 0, 0], [0, 0, 0, 1]]   # ProjGrid's F
    assert mv.camera_to_world(90)[0][3] == pytest.approx(d)       # the prop's left side is seen from +X


def test_glb_rotation_puts_the_picture_side_on_plus_z() -> None:
    r = [row[:3] for row in server.GLB_ROTATION[:3]]

    def apply(v: tuple[int, int, int]) -> list[int]:
        return [sum(a * b for a, b in zip(row, v)) for row in r]

    # to_glb hands over (x, z_latent, -y_latent): the camera sits on its +Y, up is its -Z
    assert apply((0, 1, 0)) == [0, 0, 1]       # camera side -> glTF front (+Z)
    assert apply((0, 0, -1)) == [0, 1, 0]      # up stays +Y
    assert apply((1, 0, 0)) == [1, 0, 0]       # the picture's right stays +X: not mirrored
    det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
           + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    assert det == 1


# ---- the input size: 1024 px on the largest side --------------------------------------------------------

def noise(image_mod: Any, size: tuple[int, int], mode: str = "RGB") -> Any:
    """A picture of noise: every resampling filter gives other pixels from it."""
    bands = [image_mod.effect_noise(size, 60 + 10 * n) for n in range(3)]
    if mode == "RGBA":
        bands.append(image_mod.new("L", size, 255))
    return image_mod.merge(mode, bands)


def test_a_small_picture_or_slice_is_enlarged_to_1024_px(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Small ones are enlarged, the aspect kept, by bilinear interpolation of the slice alone; the unit says by
    how much. Larger ones keep their pixels up to 2048 px (reducing them would lose detail the framing uses)."""
    image_mod = pytest.importorskip("PIL.Image")
    chops = pytest.importorskip("PIL.ImageChops")
    bilinear = image_mod.Resampling.BILINEAR
    assert (mv.INPUT_SIZE, mv.MAX_SLICE) == (1024, 2048)
    small, large, exact = tmp_path / "small.png", tmp_path / "large.png", tmp_path / "exact.png"
    huge = tmp_path / "huge.png"
    noise(image_mod, (512, 384)).save(small)
    noise(image_mod, (1800, 1200)).save(large)
    noise(image_mod, (2400, 1200)).save(huge)
    noise(image_mod, (600, 1024)).save(exact)

    def same(a: Any, b: Any) -> bool:
        return a.mode == b.mode and a.size == b.size and chops.difference(a, b).getbbox() is None

    with image_mod.open(small) as source:
        whole = mv.open_crop(str(small), None)
        assert whole.size == (1024, 768) and whole.info["spellcast_unit"] == 2.0
        assert same(whole, source.resize((1024, 768), bilinear))
        assert not same(whole, source.resize((1024, 768), image_mod.Resampling.LANCZOS))
        tall = mv.open_crop(str(small), [100, 50, 200, 300], 90)                 # a slice: its own largest side
        assert tall.size == (683, 1024) and tall.info["spellcast_unit"] == pytest.approx(1024 / 300)
        assert same(tall, source.crop((100, 50, 300, 350)).resize((683, 1024), bilinear))
        clamped = mv.open_crop(str(small), [-10, -10, 100, 2000])                # what lies inside the picture
        assert clamped.size == (240, 1024) and clamped.info["spellcast_unit"] == pytest.approx(1024 / 384)
    for path in (exact, large):                                                   # 1024 to 2048 px: as it is
        with image_mod.open(path) as source:
            kept = mv.open_crop(str(path), None)
            assert kept.info["spellcast_unit"] == 1.0 and same(kept, source)
    with image_mod.open(huge) as source:                                          # beyond: shrunk, as before
        reduced = mv.open_crop(str(huge), None)
        assert reduced.size == (2048, 1024) and reduced.info["spellcast_unit"] == pytest.approx(2048 / 2400)
        assert same(reduced, source.resize((2048, 1024), image_mod.Resampling.LANCZOS))
    out = capsys.readouterr().out
    assert "front view: picture of 512x384 px brought to 1024x768 px" in out
    assert "left view: slice of 200x300 px brought to 683x1024 px" in out
    assert "picture of 1800x1200 px" not in out and "picture of 2400x1200 px brought to 2048x1024 px" in out
    with pytest.raises(mv.ViewProblem):
        mv.open_crop(str(small), [600, 0, 50, 50])


def test_transparency_survives_the_resize(tmp_path: Path) -> None:
    """RGBA is resized premultiplied: the colour under the transparent pixels does not bleed into the prop's
    rim, an opaque picture stays opaque (its background is still removed), and a palette's or an RGB
    picture's transparent colour becomes an alpha before any pixel is blended."""
    image_mod = pytest.importorskip("PIL.Image")
    np = pytest.importorskip("numpy")
    cut_out = image_mod.new("RGBA", (200, 150), (0, 255, 0, 0))                  # green under the transparency
    cut_out.paste((255, 0, 0, 255), (40, 30, 160, 120))
    cut_out.save(tmp_path / "cut.png")
    for rect in (None, [10, 5, 180, 140]):
        pixels = np.asarray(mv.open_crop(str(tmp_path / "cut.png"), rect))
        assert max(pixels.shape[:2]) == 1024 and pixels.shape[2] == 4
        rim = (pixels[..., 3] > 0) & (pixels[..., 3] < 255)
        assert rim.any() and pixels[..., 3].min() == 0 and pixels[..., 3].max() == 255
        assert (pixels[rim][:, 0] == 255).all() and (pixels[rim][:, 1] == 0).all()   # red, no green
    assert server.has_own_transparency(mv.open_crop(str(tmp_path / "cut.png"), None))

    for size in ((300, 200), (1300, 867), (513, 377)):
        noise(image_mod, size, "RGBA").save(tmp_path / "opaque.png")
        opaque = mv.open_crop(str(tmp_path / "opaque.png"), None)
        assert opaque.mode == "RGBA" and opaque.getchannel("A").getextrema() == (255, 255)
        assert not server.has_own_transparency(opaque) and mv.real_alpha(opaque) is None

    keyed = image_mod.new("RGB", (64, 64), (255, 0, 255))                         # magenta: the transparent colour
    keyed.paste((10, 200, 30), (16, 16, 48, 48))
    keyed.save(tmp_path / "keyed.png", transparency=(255, 0, 255))
    palette = image_mod.new("P", (64, 64), 0)
    palette.putpalette([255, 0, 255, 10, 200, 30])
    palette.paste(1, (16, 16, 48, 48))
    palette.save(tmp_path / "palette.png", transparency=0)
    for name in ("keyed.png", "palette.png"):
        with image_mod.open(tmp_path / name) as source:
            assert source.mode != "RGBA" and "transparency" in source.info
        pixels = np.asarray(mv.open_crop(str(tmp_path / name), None)).astype(int)
        assert pixels.shape == (1024, 1024, 4) and pixels[0, 0, 3] == 0 and pixels[512, 512, 3] == 255, name
        seen = pixels[pixels[..., 3] > 0]
        assert (seen[:, 0] < 60).all() and (seen[:, 2] < 90).all(), name            # no magenta shows anywhere


def test_slices_of_a_sheet_keep_their_proportions_at_1024_px(tmp_path: Path) -> None:
    """Each small slice is enlarged to 1024 px by its own factor; frame_views undoes it (its unit), so the views of a
    sheet stay at one scale: framed as they would be from the slices as drawn."""
    pytest.importorskip("numpy")
    image_mod = pytest.importorskip("PIL.Image")
    sheet = image_mod.new("RGBA", (1200, 600), (0, 0, 0, 0))
    sheet.paste((200, 100, 50, 255), (100, 100, 300, 500))                        # the front: 200 x 400
    sheet.paste((200, 100, 50, 255), (700, 150, 850, 570))                        # the back: 150 x 420, 5% taller
    sheet.save(tmp_path / "sheet.png")
    rects = [[0, 0, 500, 600], [600, 100, 300, 500]]
    crops = [mv.open_crop(str(tmp_path / "sheet.png"), rect, azimuth) for rect, azimuth in zip(rects, (0, 180))]
    units = [crop.info["spellcast_unit"] for crop in crops]
    assert [max(crop.size) for crop in crops] == [1024, 1024]
    assert units == pytest.approx([1024 / 600, 1024 / 500])
    framed = mv.frame_views(crops, [0, 180], ["sheet", "sheet"], units)
    boxes = [mv.object_box(v) for v in framed]
    heights = [b[3] - b[1] for b in boxes]
    assert heights[1] / heights[0] == pytest.approx(420 / 400, abs=0.01)
    assert max(heights) == pytest.approx(mv.VIEW_SIZE / mv.FRAME_MARGIN, abs=2)
    drawn = [sheet.crop((x, y, x + w, y + h)) for x, y, w, h in rects]
    for box, plain in zip(boxes, mv.frame_views(drawn, [0, 180], ["sheet", "sheet"])):
        assert list(box) == pytest.approx(list(mv.object_box(plain)), abs=2)


# ---- framing the views ---------------------------------------------------------------------------------

def test_views_of_one_sheet_share_one_scale() -> None:
    """A turnaround sheet's proportions are exact: its views keep them (a back drawn taller stays taller), and
    the largest extent over them fills 1/1.1 of the square."""
    sheet = [(400.0, 800.0), (300.0, 820.0), (400.0, 840.0), (310.0, 820.0)]
    scales = mv.frame_scales(sheet, ["sheet.png"] * 4)
    assert len(set(scales)) == 1
    assert max(max(w, h) * scales[0] for w, h in sheet) == pytest.approx(mv.VIEW_SIZE / mv.FRAME_MARGIN)
    # A wide sheet: its widest view fills it instead.
    wide = [(900.0, 300.0), (200.0, 310.0)]
    scale = mv.frame_scales(wide, ["w.png", "w.png"])[0]
    assert 900 * scale == pytest.approx(mv.VIEW_SIZE / mv.FRAME_MARGIN)


def test_views_of_different_pictures_are_made_equally_tall() -> None:
    """Photos from each side, at any distance: the prop is as tall from every side (elevation 0)."""
    sizes = [(400.0, 800.0), (150.0, 300.0), (1000.0, 2000.0)]
    scales = mv.frame_scales(sizes, ["a.png", "b.png", "c.png"])
    heights = [h * s for (_, h), s in zip(sizes, scales)]
    assert heights == pytest.approx([mv.VIEW_SIZE / mv.FRAME_MARGIN] * 3)
    # Without pictures to tell them apart (the old call), each view is its own picture: the same.
    assert mv.frame_scales(sizes, [0, 1, 2]) == pytest.approx(scales)


def test_a_sheet_and_a_separate_picture_meet_at_the_sheets_mean_height() -> None:
    sizes = [(400.0, 800.0), (300.0, 900.0), (500.0, 1000.0)]        # front and left on a sheet, back apart
    scales = mv.frame_scales(sizes, ["sheet", "sheet", "back.png"])
    assert scales[0] == scales[1]                                      # one scale on the sheet
    sheet_mean = (800 + 900) / 2 * scales[0]
    assert 1000 * scales[2] == pytest.approx(sheet_mean)               # the back as tall as the sheet's mean
    assert max(max(w, h) * s for (w, h), s in zip(sizes, scales)) == pytest.approx(mv.VIEW_SIZE / mv.FRAME_MARGIN)


def test_frame_views_keeps_a_sheets_proportions_in_pixels() -> None:
    pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")

    def prop(w: int, h: int, size: tuple[int, int] = (600, 1000)) -> Any:
        image = Image.new("RGBA", size, (0, 0, 0, 0))
        image.paste((200, 100, 50, 255), (50, 60, 50 + w, 60 + h))
        return image

    views = [prop(300, 800), prop(200, 840)]                           # the back view drawn 5% taller
    framed = mv.frame_views(views, [0, 180], ["s", "s"])
    boxes = [mv.object_box(v) for v in framed]
    heights = [b[3] - b[1] for b in boxes]
    assert heights[1] / heights[0] == pytest.approx(840 / 800, abs=0.01)    # the sheet's own proportions
    assert max(heights) == pytest.approx(mv.VIEW_SIZE / mv.FRAME_MARGIN, abs=2)
    for (x0, y0, x1, y1) in boxes:                                     # each centred on its prop
        assert (x0 + x1) / 2 == pytest.approx(mv.VIEW_SIZE / 2, abs=1)
        assert (y0 + y1) / 2 == pytest.approx(mv.VIEW_SIZE / 2, abs=1)
    # A slice open_crop brought to half its size (units 0.5) counts in its picture's pixels.
    half = views[1].resize((300, 500))
    again = mv.frame_views([views[0], half], [0, 180], ["s", "s"], units=[1.0, 0.5])
    assert [mv.object_box(v)[3] - mv.object_box(v)[1] for v in again] == pytest.approx(heights, abs=2)
    # Different pictures: equally tall.
    apart = mv.frame_views(views, [0, 180], ["a", "b"])
    assert [mv.object_box(v)[3] - mv.object_box(v)[1] for v in apart] == pytest.approx(
        [mv.VIEW_SIZE / mv.FRAME_MARGIN] * 2, abs=2)


# ---- stray bits of a slice -------------------------------------------------------------------------------

def test_only_stray_bits_of_a_slice_are_cleared() -> None:
    prop = (100, 100, 300, 500, 90_000)                   # (x, y, w, h, area) in a 500 x 700 slice
    orb = (180, 50, 40, 40, 1_200)                        # a floating orb just above the head: stays
    sword = (420, 200, 50, 300, 8_000)                    # held apart, big: stays
    speck = (20, 650, 10, 10, 60)                         # a small caption far below: goes
    edge = (480, 300, 20, 40, 500)                        # a bit of the next view, at the slice's edge: goes
    assert mv.strays([prop, orb, sword, speck, edge], 500, 700) == [3, 4]
    assert mv.strays([orb, prop], 500, 700) == []         # the largest is the prop, wherever it is listed
    assert mv.strays([prop], 500, 700) == [] and mv.strays([], 500, 700) == []


def test_clear_strays_on_a_picture() -> None:
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    image.paste((255, 255, 255, 255), (100, 100, 300, 350))         # the prop
    image.paste((255, 255, 255, 255), (180, 60, 220, 90))           # an orb over it: stays
    image.paste((255, 255, 255, 255), (390, 10, 400, 30))           # cut from the next view: goes
    image.paste((255, 255, 255, 255), (20, 380, 26, 386))           # a speck far away: goes
    assert mv.clear_strays(image) == (4, 2)
    alpha = np.asarray(image.getchannel("A"))
    assert alpha[200, 200] == 255 and alpha[75, 200] == 255
    assert alpha[20, 395] == 0 and alpha[383, 23] == 0


# ---- the inner copy ----------------------------------------------------------------------------------------

def test_the_separate_pieces_go_even_when_the_joined_copy_would_be_too_much() -> None:
    assert guard.inner_plan(1_426_916, 0, 2_968_640) == "whole"             # the sheep
    assert guard.inner_plan(16_316, 1_870_455, 4_400_696) == "both"         # the bear
    assert guard.inner_plan(500, 8_000, 10_000) == "whole"                  # joined: 84% of what is left
    assert guard.inner_plan(8_500, 500, 10_000) == "none"                   # the separate pieces alone 85%
    assert guard.inner_plan(0, 0, 10_000) == "none"


def test_the_joined_copy_is_judged_against_what_the_separate_pieces_leave() -> None:
    # multi-view heads: from 4 files (a solid decoded surface: one hidden wall per visible one) ...
    assert guard.inner_plan(44_248, 2_336_714, 4_537_760) == "both"
    # ... and from a 2x2 sheet (the decoded surface a thin closed shell around a hollow: three hidden walls,
    # 71% of the faces), split as the pieces joined: 43% separate + 49% of the rest, or 3% + 70.5%
    assert guard.inner_plan(3_161_734, 2_060_000, 7_369_340) == "both"
    assert guard.inner_plan(203_896, 5_084_655, 7_421_128) == "both"
    assert guard.inner_plan(0, 5_288_551, 7_421_128) == "both"              # all joined
    assert guard.inner_plan(5_288_551, 0, 7_421_128) == "whole"             # all separate
    assert guard.inner_plan(5_000, 4_000, 10_000) == "both"                 # joined: 80% of what is left
    assert guard.inner_plan(5_000, 4_100, 10_000) == "whole"                # 82%: not a hidden copy
    assert guard.inner_plan(0, 8_100, 10_000) == "none"                     # no separate pieces: of all faces
    assert guard.inner_plan(8_100, 100, 10_000) == "none"                   # the sure part too much: nothing


def test_a_piece_that_sees_out_is_a_hollows_wall() -> None:
    assert guard.is_cavity(0.81) and guard.is_cavity(0.99)                  # jars with an 8-15 degree opening
    assert not guard.is_cavity(0.46) and not guard.is_cavity(0.08)          # the bear's and the head's copy


def test_cut_away_keeps_what_the_mask_does_not_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    """cumesh's remove_faces KEEPS the faces where its mask is True (verified on the GPU): cut_away passes
    it the faces to keep."""
    class Mesh:
        def init(self, vertices: Any, faces: Any) -> None:
            self.vertices, self.faces = vertices, faces

        def remove_faces(self, keep: Any) -> None:
            self.faces = [f for f, k in zip(self.faces, keep) if k]

        def remove_unreferenced_vertices(self) -> None:
            pass

        def read(self) -> tuple[Any, Any]:
            return self.vertices, self.faces

    class Mask(list):
        def __invert__(self) -> "Mask":
            return Mask(not x for x in self)

        def contiguous(self) -> "Mask":
            return self

    cumesh = types.SimpleNamespace(CuMesh=Mesh)
    _, faces = server.cut_away(cumesh, "v", ["outer", "inner", "outer2"], Mask([False, True, False]))
    assert faces == ["outer", "outer2"]


def test_faces_near_a_seen_one_are_grown_ring_by_ring() -> None:
    torch = pytest.importorskip("torch")
    # a strip of 6 triangles, each sharing an edge with the next
    faces = torch.tensor([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4], [4, 5, 6], [5, 7, 6]])
    seen = torch.tensor([True, False, False, False, False, False])
    assert server.grow(seen, faces, 8, 0).tolist() == seen.tolist()
    assert server.grow(seen, faces, 8, 1).tolist() == [True, True, True, False, False, False]
    assert server.grow(seen, faces, 8, 2).tolist() == [True, True, True, True, True, False]


# ---- the multi-view weights ----------------------------------------------------------------------------

def fake_snapshot(root: Path, with_mv: bool = True, skip: str = "") -> Path:
    cache = root / "hub" / "models--TencentARC--Pixal3D"
    (cache / "refs").mkdir(parents=True, exist_ok=True)
    (cache / "refs" / "main").write_text("abc123", encoding="utf-8")
    snap = cache / "snapshots" / "abc123"
    (snap / "ckpts").mkdir(parents=True, exist_ok=True)
    for config, suffix in (("pipeline.json", ""), ("pipeline_mv.json", "_mv")):
        if suffix and not with_mv:
            continue
        models = {name: f"ckpts/{name}{suffix}" for name in mv.FLOW_MODELS}
        models["shape_slat_decoder"] = "ckpts/decoder"
        (snap / config).write_text(json.dumps({"args": {"models": models}}), encoding="utf-8")
        for name in mv.FLOW_MODELS:
            for ext in (".json", ".safetensors"):
                if f"{name}{suffix}{ext}" != skip:
                    (snap / "ckpts" / f"{name}{suffix}{ext}").write_bytes(b"{}")
    return snap


def test_multi_view_weights_count_only_when_complete(tmp_path: Path) -> None:
    assert not mv.multiview_available(tmp_path)                                  # nothing at all
    fake_snapshot(tmp_path / "a", with_mv=False)
    assert mv.flow_checkpoints(tmp_path / "a", "sv") is not None
    assert not mv.multiview_available(tmp_path / "a")                            # no pipeline_mv.json
    fake_snapshot(tmp_path / "b", skip="tex_slat_flow_model_1024_mv.safetensors")
    assert not mv.multiview_available(tmp_path / "b")                            # one file still downloading
    snap = fake_snapshot(tmp_path / "c")
    assert mv.multiview_available(tmp_path / "c")
    found = mv.flow_checkpoints(tmp_path / "c", "mv")
    assert found is not None and found["sparse_structure_flow_model"] == snap / "ckpts" / "sparse_structure_flow_model_mv"


def test_a_missing_multi_view_set_is_reported_before_anything_changes(tmp_path: Path) -> None:
    fake_snapshot(tmp_path, with_mv=False)
    px = types.SimpleNamespace(pipeline=types.SimpleNamespace(models={}))
    modes = mv.ModelModes(px, tmp_path, lambda p: None)
    modes.ensure("sv")                                   # already there: nothing to load
    with pytest.raises(mv.MultiviewMissing):
        modes.ensure("mv")
    assert modes.mode == "sv" and modes.pipeline() is px.pipeline and modes.extractors() == []


class Checkpoints:
    """Stands in for the pixal3d.models module: from_pretrained notes what it was asked to load."""

    def __init__(self) -> None:
        self.asked: list[str] = []
        self.from_pretrained = self.load

    def load(self, path: str, **kwargs: Any) -> str:
        self.asked.append(Path(path).name)
        return f"model of {Path(path).name}"


def start(modes: mv.ModelModes, mode: str, loader: Checkpoints, names: tuple[str, ...] = mv.FLOW_MODELS) -> dict[str, str]:
    """What init_models() does with the checkpoints: every model of pipeline.json through loader.from_pretrained
    (the module attribute, looked up at each call)."""
    with modes.loading_first(mode, loader):
        models = {name: loader.from_pretrained(f"TencentARC/Pixal3D/ckpts/{name}") for name in (*names, "decoder")}
    assert loader.from_pretrained == loader.load             # put back once the models are loaded
    return models


def test_a_start_for_a_multi_view_prop_loads_the_multi_view_models_right_away(tmp_path: Path) -> None:
    """Loading the single-view denoisers first read 22 GB only to replace them before the first mesh."""
    fake_snapshot(tmp_path)
    loader = Checkpoints()
    px = types.SimpleNamespace(pipeline=types.SimpleNamespace(models={}), _update_progress=lambda *a: pytest.fail(
        "nothing is swapped when the start loaded what the first job needs"))
    modes = mv.ModelModes(px, tmp_path, lambda p: None)
    modes._build_mv_pipeline = lambda: "the multi-view pipeline"     # type: ignore[method-assign]
    models = start(modes, "mv", loader)
    assert loader.asked == [*(f"{name}_mv" for name in mv.FLOW_MODELS), "decoder"]   # each read once; the rest shared
    assert models["tex_slat_flow_model_1024"] == "model of tex_slat_flow_model_1024_mv"
    assert modes.mode == "mv"
    modes.ensure("mv")                                               # the first job: no weights to load
    assert modes.pipeline() == "the multi-view pipeline"


def test_a_start_loads_the_single_view_models_unless_told_and_able(tmp_path: Path) -> None:
    plain = [*mv.FLOW_MODELS, "decoder"]
    fake_snapshot(tmp_path / "both")
    px = types.SimpleNamespace(pipeline=types.SimpleNamespace(models={}))
    loader = Checkpoints()
    modes = mv.ModelModes(px, tmp_path / "both", lambda p: None)
    start(modes, "sv", loader)                                       # a prop made from one picture, or a start by hand
    assert loader.asked == plain and modes.mode == "sv"

    fake_snapshot(tmp_path / "half", skip="tex_slat_flow_model_1024_mv.safetensors")   # still downloading
    fake_snapshot(tmp_path / "none", with_mv=False)
    other = fake_snapshot(tmp_path / "other")                        # a multi-view model of another make
    (other / "ckpts" / "shape_slat_flow_model_512_mv.json").write_text('{"name": "Other"}', encoding="utf-8")
    for folder in ("half", "none", "other"):
        loader = Checkpoints()
        modes = mv.ModelModes(px, tmp_path / folder, lambda p: None)
        assert mv.startup_checkpoints(tmp_path / folder) is None
        start(modes, "mv", loader)
        assert loader.asked == plain and modes.mode == "sv", folder


def test_a_start_that_loaded_only_some_multi_view_models_loads_the_set_again(tmp_path: Path) -> None:
    """Should Pixal3D load its checkpoints another way one day, the models are not taken for a set they are not."""
    fake_snapshot(tmp_path)
    loader = Checkpoints()
    modes = mv.ModelModes(types.SimpleNamespace(pipeline=None), tmp_path, lambda p: None)
    start(modes, "mv", loader, mv.FLOW_MODELS[:2])
    assert modes.mode is None                                        # the next ensure() loads all four

    class Broken(Checkpoints):
        def load(self, path: str, **kwargs: Any) -> str:
            raise FileNotFoundError(path)

    broken = Broken()
    with pytest.raises(FileNotFoundError):
        start(mv.ModelModes(types.SimpleNamespace(pipeline=None), tmp_path, lambda p: None), "mv", broken)
    assert broken.from_pretrained == broken.load                     # put back after a failed start too


def test_the_launcher_is_told_what_its_first_job_is_made_from() -> None:
    base = ["--repo", "r", "--port", "7772"]
    assert server.parse_args(base).first_job == "single"             # started by hand, or by an older hub
    assert server.parse_args([*base, "--first-job", "multi"]).first_job == "multi"
    assert server.FIRST_JOBS == {"single": "sv", "multi": "mv"} and set(server.FIRST_JOBS.values()) == set(mv.CONFIGS)
    with pytest.raises(SystemExit):
        server.parse_args([*base, "--first-job", "several"])


def test_a_swap_of_the_models_is_called_a_swap() -> None:
    """Not "Loading ...": after "Pixal3D is starting" that read as the same models loaded twice."""
    for mode, words in (("mv", "single-view models for the multi-view ones"),
                        ("sv", "multi-view models for the single-view ones")):
        label, fraction = server.describe(mv.SWITCHING[mode], 1, 4)
        assert label == f"Swapping the {words}" and 0 < fraction < 0.03


def weighted_snapshot(root: Path) -> tuple[dict[str, Any], dict[tuple[str, str], float]]:
    """A snapshot whose flow checkpoints are real .safetensors of a tiny model (fp32 on disk, bf16 blocks in
    memory, like Pixal3D's), single-view filled with 1, 2, 3, 4 and multi-view with 2, 3, 4, 5; and the
    four modules as init_models() leaves them."""
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file

    class Flow(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = torch.nn.Linear(4, 6)
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(6, 6) for _ in range(2)])
            for param in self.blocks.parameters():
                param.data = param.data.to(torch.bfloat16)

    snap = fake_snapshot(root)
    values = {}
    for mode, suffix in (("sv", ""), ("mv", "_mv")):
        for n, name in enumerate(mv.FLOW_MODELS):
            (snap / "ckpts" / f"{name}{suffix}.json").write_text(json.dumps({"name": "Flow"}), encoding="utf-8")
            fill = (1.0 if mode == "sv" else 2.0) + n
            state = {k: torch.full(v.shape, fill, dtype=torch.float32) for k, v in Flow().state_dict().items()}
            save_file(state, str(snap / "ckpts" / f"{name}{suffix}.safetensors"))
            values[(mode, name)] = fill
    models = {name: Flow() for name in mv.FLOW_MODELS}
    for name, module in models.items():
        mv.copy_weights(module, snap / "ckpts" / f"{name}.safetensors")    # what init_models loaded
    return models, values


def test_the_models_switch_to_multi_view_and_back_one_tensor_at_a_time(tmp_path: Path) -> None:
    models, values = weighted_snapshot(tmp_path)
    progress: list[tuple[str, int, int]] = []
    px = types.SimpleNamespace(pipeline=types.SimpleNamespace(models=models),
                               _update_progress=lambda stage, i, n: progress.append((stage, i, n)))
    modes = mv.ModelModes(px, tmp_path, lambda p: None)
    built: list[str] = []
    modes._build_mv_pipeline = lambda: built.append("mv") or "the multi-view pipeline"   # type: ignore[method-assign]

    def holds(mode: str) -> bool:
        return all(float(models[n].inp.weight.float().mean()) == values[(mode, n)]
                   and float(models[n].blocks[1].bias.float().mean()) == values[(mode, n)] for n in mv.FLOW_MODELS)

    assert holds("sv")
    modes.ensure("mv")
    assert modes.mode == "mv" and holds("mv") and modes.pipeline() == "the multi-view pipeline"
    assert str(models["tex_slat_flow_model_1024"].blocks[0].weight.dtype) == "torch.bfloat16"   # kept in bf16
    assert progress[-1] == (mv.SWITCHING["mv"], 4, 4)
    count = len(progress)
    modes.ensure("mv")                                               # already there: nothing read
    assert len(progress) == count and built == ["mv"]
    modes.ensure("sv")
    assert modes.mode == "sv" and holds("sv") and modes.pipeline() is px.pipeline
    modes.ensure("mv")
    assert holds("mv") and built == ["mv"]                           # the pipeline is built once


def test_a_switch_that_fails_half_way_reloads_all_four_next_time(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    models, values = weighted_snapshot(tmp_path)
    px = types.SimpleNamespace(pipeline=types.SimpleNamespace(models=models), _update_progress=lambda *a: None)
    modes = mv.ModelModes(px, tmp_path, lambda p: None)
    modes._build_mv_pipeline = lambda: "mv pipeline"                 # type: ignore[method-assign]
    real, calls = mv.copy_weights, []

    def flaky(module: Any, path: Path) -> int:
        calls.append(path)
        if len(calls) == 3:
            raise OSError("disk read error")
        return real(module, path)

    monkeypatch.setattr(mv, "copy_weights", flaky)
    with pytest.raises(OSError):
        modes.ensure("mv")
    assert modes.mode is None                                        # half-loaded: neither set
    monkeypatch.setattr(mv, "copy_weights", real)
    modes.ensure("sv")
    assert modes.mode == "sv"
    assert all(float(models[n].inp.weight.float().mean()) == values[("sv", n)] for n in mv.FLOW_MODELS)


def test_copying_weights_gives_what_load_state_dict_gives(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    source = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Linear(5, 2))
    path = tmp_path / "w.safetensors"
    save_file({**{k: v.float() for k, v in source.state_dict().items()}, "unused": torch.ones(2)}, str(path))
    a, b = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Linear(5, 2)), None
    b = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Linear(5, 2))
    a[1].to(torch.bfloat16)
    b[1].to(torch.bfloat16)
    read = mv.copy_weights(a, path)
    b.load_state_dict({k: v for k, v in source.state_dict().items()}, strict=False)
    assert read == sum(t.numel() * 4 for t in source.state_dict().values())      # the unused tensor is not read
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and va.dtype == vb.dtype and torch.equal(va, vb)


def test_the_multi_view_pipeline_does_not_inherit_the_single_view_wrappers() -> None:
    class Pipeline:
        def get_proj_cond_ss(self) -> str:
            return "single-view"

        def decode_latent(self) -> str:
            return "decode"

    sv = Pipeline()
    sv.models = {"flow": object()}
    sv.low_vram = True
    sv.get_proj_cond_ss = lambda: "wrapped single-view"    # what trim_cuda_cache leaves behind
    sv.decode_latent = lambda: "labelled"
    state = mv.shared_state(sv)
    assert set(state) == {"models", "low_vram"} and state["models"] is sv.models


# ---- checks on what the model makes ------------------------------------------------------------------

def column(i: int, j: int, ks: range | tuple[int, ...]) -> list[list[int]]:
    return [[0, i, j, k] for k in ks]


def test_the_edge_score_spots_a_block_running_to_the_grid_face() -> None:
    shell = [row for i in range(8, 24) for j in range(8, 24) for row in column(i, j, (10, 20))]
    near = [row for i in range(8, 24) for j in range(8, 24) for row in column(i, j, range(12, 32))]
    far = [row for i in range(8, 24) for j in range(8, 24) for row in column(i, j, range(0, 18))]
    half = [row for i in range(8, 24) for j in range(8, 24) for row in column(i, j, range(12, 32) if i < 16 else (15,))]
    assert guard.ss_edge_score(shell) == 0.0
    assert guard.ss_edge_score(near) == 1.0
    assert guard.ss_edge_score(far) == 1.0
    assert guard.ss_edge_score(half) == 0.5
    assert guard.ss_edge_score([]) == 0.0
    assert guard.ss_edge_score(near, resolution=64) == 0.0      # the band follows the resolution


def test_the_deep_share_counts_only_columns_solid_from_the_face() -> None:
    block = [row for i in range(8, 24) for j in range(8, 24) for row in column(i, j, range(20, 32))]
    far = [row for i in range(8, 24) for j in range(8, 24) for row in column(i, j, range(0, 8))]
    # a hollow prop that is deep along the camera axis: front and back walls touch both faces
    hollow = [row for i in range(8, 24) for j in range(8, 24) for row in column(i, j, (0, 1, 2, 29, 30, 31))]
    gap = [row for i in range(8, 24) for j in range(8, 24) for row in column(i, j, (31, 30, 29, 28, 27, 26, 25, 23))]
    half = [row for i in range(8, 24) for j in range(8, 24) for row in column(i, j, range(20, 32) if i < 16 else (15,))]
    assert guard.ss_deep_share(block) == 1.0 and guard.ss_deep_share(far) == 1.0
    assert guard.ss_edge_score(hollow) == 1.0 and guard.ss_deep_share(hollow) == 0.0
    assert guard.ss_deep_share(gap) == 0.0                      # 7 in a row, then a hole
    assert guard.ss_deep_share(half) == 0.5 and guard.ss_deep_share([]) == 0.0
    assert guard.ss_blocked(guard.ss_edge_score(block), guard.ss_deep_share(block))
    assert not guard.ss_blocked(guard.ss_edge_score(hollow), guard.ss_deep_share(hollow))
    # the calibration's extremes: healthy props at most 0.109 / 0.017, the failed bear 0.938 / 0.922
    assert not guard.ss_blocked(0.109, 0.017) and guard.ss_blocked(0.938, 0.922)
    assert not guard.ss_blocked(0.3, 0.9) and not guard.ss_blocked(0.9, 0.29)


def guarded(draws: list[list[list[int]]]) -> tuple[Callable[..., Any], server.RunNotes, list[Any]]:
    calls: list[Any] = []

    class Pipeline:
        def sample_sparse_structure(self, cond: Any, resolution: int, num_samples: int = 1,
                                    sampler_params: Any = None) -> list[list[int]]:
            calls.append(sampler_params)
            return draws[len(calls) - 1]

    p = Pipeline()
    notes = server.RunNotes()
    server.guard_sparse_structure(p, notes)
    return p.sample_sparse_structure, notes, calls


def test_the_guard_draws_a_block_again_and_keeps_the_best(capsys: pytest.CaptureFixture[str]) -> None:
    assert (guard.GUARD_THRESHOLD, guard.DEEP_THRESHOLD, guard.GUARD_ATTEMPTS) == (0.3, 0.3, 3)
    params = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}
    good = column(5, 5, (10, 11))
    block = column(5, 5, range(10, 32))
    worse = column(5, 5, range(0, 32))
    hollow = column(5, 5, (0, 1, 30, 31))                                     # reaches both faces, not solid

    sample, notes, calls = guarded([good])
    assert sample({}, 32, 1, params) == good and calls == [params] and calls[0] is params   # untouched
    assert notes.ss_retries == 0 and not notes.ss_blocked

    sample, notes, calls = guarded([hollow])
    assert sample({}, 32, 1, params) == hollow and len(calls) == 1

    sample, notes, calls = guarded([block, good, block])
    assert sample({}, 32, 1, params) == good and len(calls) == 2 and notes.ss_retries == 1
    assert calls[1] is params                                                 # attempt 2: new noise only
    assert "drawing it again" in capsys.readouterr().out and not notes.ss_blocked

    half = column(5, 5, range(10, 32)) + column(6, 6, (12,))                  # 0.5: over the limit, best of all
    sample, notes, calls = guarded([worse, half, block])
    assert sample({}, 32, 1, params) == half and len(calls) == 3 and notes.ss_retries == 2 and notes.ss_blocked
    assert calls[2] == {**params, "steps": 25, "guidance_strength": 5.0} and params["steps"] == 12
    out = capsys.readouterr().out
    assert "keeping the best draw (50%)" in out and "3/3 (steps 25, guidance_strength 5.0" in out

    again = list(half)
    sample, notes, calls = guarded([half, block, again])                      # a tie: the earlier draw
    assert sample({}, 32, 1, params) is half and len(calls) == 3
    notes.reset()
    assert notes.ss_retries == 0 and not notes.ss_blocked


def test_the_guard_keeps_the_first_draw_when_the_check_fails(capsys: pytest.CaptureFixture[str]) -> None:
    sample, notes, calls = guarded([[["not", "a", "row"]]])
    assert sample({}, 32) == [["not", "a", "row"]] and calls == [{}]
    assert "could not check" in capsys.readouterr().out


def test_the_inward_facing_copy_goes_only_when_it_looks_like_one() -> None:
    assert guard.drop_inner_copy(1_426_916, 2_968_640)               # the sheep: its whole inner shell (48%)
    assert guard.drop_inner_copy(1, 100)
    assert not guard.drop_inner_copy(0, 100)                          # nothing inside
    assert not guard.drop_inner_copy(100, 100)                        # everything "inside": the test says nothing
    assert guard.drop_inner_copy(71, 100)                             # three hidden walls around a hollow
    assert not guard.drop_inner_copy(81, 100)                         # more than 80%: not a hidden copy
    assert guard.drop_inner_copy(80, 100)
    assert not guard.drop_inner_copy(0, 0)


def test_only_large_pieces_of_inside_faces_are_the_copy() -> None:
    # the bear: its copy is one piece of 1.89M faces; visible faces in crevices make pieces of up to 789
    assert guard.inner_pieces([1_888_025, 789, 446, 3]) == [True, False, False, False]
    assert guard.inner_pieces([500_000, 20_000, 9_000]) == [True, True, False]   # a second part's copy (2%)
    assert guard.inner_pieces([1999]) == [False] and guard.inner_pieces([2000]) == [True]
    assert guard.inner_pieces([]) == []


def test_hole_centres_are_put_back_where_cumesh_meant_them(monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    torch = pytest.importorskip("torch")

    class Mesh:   # stands in for pixal3d.representations.mesh.base.Mesh with cumesh's uncleared z sum
        def __init__(self) -> None:
            self.vertices = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=torch.float32)
            self.faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)

        def fill_holes(self) -> None:   # closes the rim 0-2-3... with a fan around a new vertex 4
            self.vertices = torch.cat([self.vertices, torch.tensor([[0.5, 0.5, 1e20]])])
            self.faces = torch.cat([self.faces, torch.tensor([[0, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=torch.int32)])

    module = types.ModuleType("pixal3d.representations.mesh.base")
    module.Mesh = Mesh
    monkeypatch.setitem(sys.modules, "pixal3d.representations.mesh.base", module)
    server.exact_hole_centres()
    mesh = Mesh()
    mesh.fill_holes()
    assert mesh.vertices[4].tolist() == pytest.approx([1 / 3, 2 / 3, 0.0])   # the mean of the rim 0, 2, 3
    assert mesh.vertices[:4].tolist() == [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    assert "Put 1 of the 1 vertices" in capsys.readouterr().out
    server.exact_hole_centres()                                        # installed once
    assert getattr(Mesh.fill_holes, "_spellcast_exact", False)


def test_the_streak_share_measures_thin_faces_along_the_camera_axis() -> None:
    pytest.importorskip("numpy")
    # a unit cube: no streaks; a bundle of long thin slivers along Z: all streak
    cube_v = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    cube_f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
              (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    assert guard.streak_share(cube_v, cube_f) == 0.0
    fins_v, fins_f = [], []
    for n in range(10):
        b = len(fins_v)
        fins_v += [(n * 0.1, 0, 0), (n * 0.1 + 0.02, 0, 0), (n * 0.1, 0, 1)]
        fins_f.append((b, b + 1, b + 2))
    assert guard.streak_share(fins_v, fins_f) == pytest.approx(1.0)
    assert guard.streak_share(fins_v, fins_f, axis=0) == 0.0


# ---- the routes ------------------------------------------------------------------------------------------

class InlineGpu:
    """Stands in for server.GpuWorker: everything runs at once on the calling thread, so a batch is over when
    its request is answered."""

    def run(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> concurrent.futures.Future[Any]:
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


class Staged:
    """Stands in for the fork's pixal3d.staged: its parts, types and exceptions as the engine's specification
    has them (pixal3d/staged/types.py)."""

    PARTS = ("cutout", "camera", "features", "ss", "cond_lr", "lr", "upsample", "cond_hr", "hr", "tex", "finish")

    @dataclasses.dataclass(frozen=True)
    class PropRequest:
        id: str
        seed: int
        resolution: int
        multi: bool
        sampler_params: dict[str, Any]
        max_num_tokens: int = 49152
        request_hash: str = ""
        payload: Any = None
        geometry: str = "pixal3d"

    @dataclasses.dataclass
    class Prepared:
        images: list[Any]
        meta: dict[str, Any]
        cutout: Any = None

    @dataclasses.dataclass
    class DecodedMesh:
        vertices: Any
        faces: Any
        coords: Any
        attrs: Any
        resolution: int
        attr_layout: dict[str, Any]

    @dataclasses.dataclass
    class Event:
        kind: str
        part: str = ""
        prop_id: str | None = None
        step: int = 0
        total: int = 0
        detail: dict[str, Any] = dataclasses.field(default_factory=dict)

    @dataclasses.dataclass
    class FinishSteps:
        steps: list[Any]
        state: dict[str, Any]
        result: Callable[[dict[str, Any]], dict[str, Any]]
        native_mib: int = 0
        safety_mib: int | None = None

    class PropFailed(Exception):
        def __init__(self, detail: str, kind: str = "error") -> None:
            super().__init__(detail)
            self.detail, self.kind = detail, kind

    class Cancelled(Exception):
        pass

    class BatchBroken(Exception):
        def __init__(self, detail: str, kind: str = "restart") -> None:
            super().__init__(detail)
            self.detail, self.kind = detail, kind


# generate_3d's keyword defaults, as far as the launcher reads them (server.ui_defaults)
DEFAULTS = {f"{stage}_{name}": value for stage in ("ss", "shape_slat", "tex_slat")
            for name, value in (("sampling_steps", 12), ("guidance_strength", 7.5), ("guidance_rescale", 0.7),
                                ("rescale_t", 5.0))}
GOOD_DRAW = [[0, 5, 5, 10], [0, 5, 5, 11]]              # a thin shell well inside the grid: the guard lets it pass
SAMPLER_STEPS = {"ss": 2, "lr": 2, "hr": 3, "tex": 2}   # the fake engine's sampling parts
NETWORK_OF = {"cutout": "rembg", "camera": "moge", "features": "dino", "ss": "ss_flow", "cond_lr": "naf",
              "lr": "lr_flow", "upsample": "shape_dec", "cond_hr": "naf", "hr": "hr_flow", "tex": "tex_flow",
              "finish": "tex_dec"}
# to_glb's bar of six steps under the names it gives it on the way (o_voxel/postprocess.py)
GLB_BARS = ("Extracting GLB", "Building BVH", "Cleaning mesh", "Parameterizing new mesh", "Sampling attributes",
            "Finalizing mesh")


class FakeGlb:
    vertices: list[Any] = []
    faces: list[Any] = []

    def export(self, path: str, extension_webp: bool = False) -> None:
        assert extension_webp
        Path(path).write_bytes(b"glTF")


class FakeEngine:
    """Stands in for pixal3d.staged.StagedEngine, without a card. It goes through the parts as the engine's
    runner does - each part for every prop of a wave, the props made from several views first - says so in
    events, and calls the launcher's hooks where the engine does (prop_scope around its guarded
    sample_sparse_structure in "ss", finish at the end).

    script[prop id] makes a prop special: {"fail": (part, kind, detail)} fails it there, {"raise": (part,
    exception)} lets the exception escape run(), {"spill": (part, ...)} lifts the memory limit for those parts,
    {"draws": [...]} are the sparse structures its draws give. at[(part, prop id)] is called when that prop
    begins that part (a test waits there, or cancels); watch() after every event. checkpoints: the props a
    stopped process had sampled, which a resumed run only finishes."""

    def __init__(self, options: Any, hooks: Any) -> None:
        self.options, self.hooks = options, hooks
        self.multiview = True
        self.script: dict[str, dict[str, Any]] = {}
        self.at: dict[tuple[str, str], Callable[[], None]] = {}
        self.watch: Callable[[], None] | None = None
        self.checkpoints: set[str] = set()
        self.runs: list[dict[str, Any]] = []
        self.released = 0
        self.closed = False
        self.draws: list[Any] = []

    def sample_sparse_structure(self, cond: Any, resolution: int, num_samples: int = 1,
                                sampler_params: Any = None) -> Any:
        return self.draws.pop(0)

    def release(self) -> None:
        self.released += 1

    def close(self) -> None:
        self.closed = True

    def run(self, props: list[Any], work_dir: Path, *, on_event: Callable[[Any], None],
            cancelled: Callable[[], bool], wave_size: int | None = None, resume: bool = False) -> dict[str, Any]:
        self.runs.append({"props": list(props), "work_dir": Path(work_dir), "wave_size": wave_size, "resume": resume})

        def tell(kind: str, part: str = "", prop_id: str | None = None, step: int = 0, total: int = 0, /,
                 **detail: Any) -> None:
            on_event(Staged.Event(kind, part, prop_id, step, total, detail))
            if self.watch is not None:
                self.watch()

        order = [p for p in props if p.multi] + [p for p in props if not p.multi]
        resumed = {p.id for p in props if resume and p.id in self.checkpoints}
        notes: dict[str, dict[str, Any]] = {p.id: {"slow": []} for p in props}
        results: dict[str, dict[str, Any]] = {}
        for prop in order:
            if prop.id in resumed:
                tell("prop_resumed", "finish", prop.id)
        size = wave_size or len(order)
        try:
            for first in range(0, len(order), size):
                for part in Staged.PARTS:
                    alive = [p for p in order[first:first + size] if p.id not in results
                             and (part != "camera" or not p.multi) and (part == "finish" or p.id not in resumed)]
                    tell("part_start", part, props=[p.id for p in alive])
                    if alive:
                        kind = ".mv" if alive[0].multi else ".sv"
                        name = NETWORK_OF[part] + (kind if NETWORK_OF[part].endswith("_flow") else "")
                        tell("net_load", part, name=name, bytes_read=0, bytes=2_000_000_000)
                        tell("net_ready", part, name=name, seconds=1.2, bytes=2_000_000_000)
                    for position, prop in enumerate(alive, 1):
                        if cancelled():
                            raise Staged.Cancelled()
                        tell("prop_start", part, prop.id, position=position, of=len(alive))
                        self.at.get((part, prop.id), lambda: None)()
                        ended = self._turn(part, prop, notes[prop.id], tell, cancelled, prop.id in resumed)
                        if ended is not None:
                            results[prop.id] = ended
                    tell("part_end", part)
        except Staged.Cancelled:
            pass
        return {p.id: results.get(p.id, {"state": "cancelled"}) for p in props}

    def _turn(self, part: str, prop: Any, notes: dict[str, Any], tell: Callable[..., None],
              cancelled: Callable[[], bool], resumed: bool) -> dict[str, Any] | None:
        """One prop's turn in one part -> how the prop ended, or None while it goes on."""
        plan = self.script.get(prop.id, {})
        if part in plan.get("spill", ()):
            notes["slow"].append(part)
            tell("prop_spilled", part, prop.id)
        if plan.get("raise", ("",))[0] == part:
            raise plan["raise"][1]
        if plan.get("fail", ("",))[0] == part:
            error = {"kind": plan["fail"][1], "detail": plan["fail"][2]}
            tell("prop_failed", part, prop.id, **error)
            return {"state": "failed", "error": error}
        try:
            return self._hooked(part, prop, notes, tell, cancelled, resumed, plan)
        except Staged.PropFailed as exc:            # a hook's: that prop alone fails, as in the engine's runner
            error = {"kind": exc.kind, "detail": exc.detail}
            tell("prop_failed", part, prop.id, **error)
            return {"state": "failed", "error": error}

    def _hooked(self, part: str, prop: Any, notes: dict[str, Any], tell: Callable[..., None],
                cancelled: Callable[[], bool], resumed: bool, plan: dict[str, Any]) -> dict[str, Any] | None:
        if part == "ss":
            self.draws = list(plan.get("draws", [GOOD_DRAW]))
            with self.hooks.prop_scope(prop, "ss") as found:
                self.sample_sparse_structure({}, 32, 1, prop.sampler_params["ss"])
            notes.update(found)
        for step in range(1, SAMPLER_STEPS.get(part, 0) + 1):
            tell("step", part, prop.id, step, SAMPLER_STEPS[part])
            if cancelled():
                raise Staged.Cancelled()
        if part != "finish":
            tell("prop_rest", part, prop.id)
            return None
        for step in (1, 2, 3):      # shape decode, texture decode, holes
            tell("step", part, prop.id, step, 3)
        mesh = Staged.DecodedMesh(None, None, None, None, plan.get("resolution", prop.resolution), {})
        outcome = self.hooks.finish(prop, mesh, notes)
        if isinstance(outcome, Staged.FinishSteps):     # the engine's runner: every step, then the result
            for _kind, fn in outcome.steps:
                fn(outcome.state)
            outcome = outcome.result(outcome.state)
        result = {**outcome, "slow": list(notes["slow"]), "seconds": 1.5,
                  "resolution": mesh.resolution, "resumed": resumed}
        tell("prop_done", part, prop.id, result=result, seconds=1.5)
        return {"state": "done", "result": result}


class Rig:
    """The launcher as main() wires it - the engine with its guard, the hooks, the reference path, the routes -
    around a stand-in for app.py (`px`) and a FakeEngine. Project P lies in `root`, the hub's projects folder."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gpu: Any = None,
                 multiview: bool = True) -> None:
        self.root = (tmp_path / "projects").resolve()
        self.models = tmp_path / "models"
        fake_snapshot(self.models, with_mv=multiview)
        self.engines: list[FakeEngine] = []
        self.prepared: list[Any] = []       # the pipelines the launcher's wrappers were put on
        self.on_bar: Callable[[], None] | None = None
        self.work_folders: list[Any] = []   # what each prop's GLB steps were given as its folder in the work folder
        self.gpu = gpu if gpu is not None else InlineGpu()
        px = self.px = types.SimpleNamespace(
            app=FastAPI(), pipeline=None, moge_model=None, envmap=None, LOW_VRAM=True, MAX_SEED=2 ** 31 - 1,
            CASCADE_MAX_NUM_TOKENS=49152, TMP_DIR=str(tmp_path), loads=0, progress=[],
            _progress_file=lambda session: str(tmp_path / f"{session}.json"))
        px._update_progress = lambda stage, step, total: px.progress.append((stage, step, total))

        def init_models() -> None:          # app.py's: everything into RAM, once
            if px.pipeline is None:
                px.loads += 1
                px.pipeline, px.moge_model, px.envmap = types.SimpleNamespace(models={}), object(), {}

        px.init_models = init_models
        monkeypatch.setattr(server, "staged_module", lambda: Staged)
        monkeypatch.setattr(server, "engine_options",
                            lambda models_root, spill=False: types.SimpleNamespace(models=models_root, spill=spill))
        monkeypatch.setattr(server, "build_engine", self._engine)
        monkeypatch.setattr(server, "glb_steps", self._glb_steps)
        monkeypatch.setattr(server, "checkpoint_loader", Checkpoints)
        monkeypatch.setattr(server, "prepare_pipeline", lambda px_, pipeline, notes: self.prepared.append(pipeline))
        self.notes = server.RunNotes()
        self.shared = server.SharedModels()
        self.modes = mv.ModelModes(px, self.models, lambda p: None)
        self.reference = server.Reference(px, self.modes, self.notes, "sv", self.shared,
                                          release=lambda: self.runner.release())
        self.runner = server.BatchRunner(px, self.gpu, server.BatchHooks(px, self.notes), self.notes, DEFAULTS,
                                         self.models, self.root, unload=self.reference.unload)
        self.gpu.run(self.runner.build)
        server.route_progress(px, self.runner.batches)
        server.add_hub_routes(px, self.gpu, DEFAULTS, self.modes, self.notes, self.root, self.reference, self.runner)
        self.client = TestClient(px.app)

    def _engine(self, options: Any, hooks: Any) -> FakeEngine:
        engine = FakeEngine(options, hooks)
        engine.multiview = mv.multiview_available(options.models)
        self.engines.append(engine)
        return engine

    def _glb_steps(self, px: Any, mesh: Any, attr_layout: Any, resolution: int, req: Any, multi: bool,
                   ss_retries: int, blocked: bool, inline: bool = False, work_folder: Any = None) -> Any:
        """pixal3d_glb.finish_steps without a card (test_pixal3d_glb.py has the real ones): one step with
        to_glb's bar, the streak check, "Saving GLB" and the file, and the answer pixal3d_glb gives."""
        self.work_folders.append(work_folder)
        def finish(state: dict[str, Any]) -> None:
            for n, name in enumerate(GLB_BARS, 1):
                px._update_progress(name, n, len(GLB_BARS))
                if self.on_bar is not None:
                    self.on_bar()
            glb = FakeGlb()
            warning = server.streak_warning(glb) or (guard.STREAK_WARNING if blocked else None)
            px._update_progress(server.SAVING, 0, 1)
            try:
                mv.make_folder(Path(req.out_path).parent)
            except mv.PropDeleted as exc:
                raise Staged.PropFailed(str(exc), "input") from None
            glb.export(req.out_path, extension_webp=True)
            views = req.views or []
            state["result"] = {"mode": "multi" if multi else "single", "views": len(views) if multi else 1,
                               "warning": warning, "ss_retries": int(ss_retries), "resolution": resolution}

        return Staged.FinishSteps([("gpu", finish)], {}, lambda state: dict(state["result"]))

    @property
    def engine(self) -> FakeEngine:
        return self.engines[-1]

    def prop(self, prop_id: str, views: tuple[int, ...] = (), **extra: Any) -> dict[str, Any]:
        """A prop of project P with its picture(s) on disk -> its item of a batch request, as the hub sends it."""
        folder = self.root / "P" / "props" / prop_id
        (folder / "images").mkdir(parents=True, exist_ok=True)
        pictures = [folder / "images" / f"{n:04d}.png" for n in range(1, max(len(views), 1) + 1)]
        for picture in pictures:
            picture.write_bytes(b"png of " + prop_id.encode())
        return {"id": prop_id, "image_path": str(pictures[0]),
                "out_path": str(folder / "mesh" / "mesh.1a2b3c4d.partial.glb"), "seed": 7, "resolution": 1024,
                "decimation_target": 300_000, "texture_size": 2048,
                "views": [{"image_path": str(p), "rect": None, "azimuth": a} for p, a in zip(pictures, views)] or None,
                "views_dir": str(folder / "mesh" / "views") if views else None, **extra}

    def body(self, items: list[dict[str, Any]], batch_id: str = "j1", **extra: Any) -> dict[str, Any]:
        return {"batch_id": batch_id, "work_dir": str(self.root / "P" / "temp" / f"mesh-{batch_id}"),
                "items": items, **extra}

    def post(self, items: list[dict[str, Any]], batch_id: str = "j1", **extra: Any) -> Any:
        return self.client.post("/spellcast/batch", json=self.body(items, batch_id, **extra))

    def told(self, batch_id: str = "j1") -> dict[str, Any]:
        r = self.client.get(f"/spellcast/batch/{batch_id}")
        assert r.status_code == 200, r.text
        return r.json()

    def ended(self, batch_id: str = "j1") -> dict[str, Any]:
        """The batch's record once it is over (a batch on a GPU thread of its own takes a moment)."""
        deadline = time.monotonic() + 10
        while True:
            told = self.told(batch_id)
            if told["state"] != "running":
                return told
            assert time.monotonic() < deadline, told
            time.sleep(0.01)


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    return Rig(tmp_path, monkeypatch)


@pytest.fixture()
def threaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    """A rig whose engine runs on the launcher's real GPU thread: the routes answer while a batch is at work."""
    gpu = server.GpuWorker()
    try:
        yield Rig(tmp_path, monkeypatch, gpu=gpu)
    finally:
        gpu._pool.shutdown(wait=True)


class Gate:
    """A place where the fake engine waits until the test lets it go on."""

    def __init__(self) -> None:
        self.reached, self.opened = threading.Event(), threading.Event()

    def __call__(self) -> None:
        self.reached.set()
        if not self.opened.wait(10):
            raise TimeoutError("the test did not let the engine go on")

    def wait(self) -> None:
        assert self.reached.wait(10), "the engine never came here"

    def open(self) -> None:
        self.opened.set()


def test_generate_answers_with_the_mode_views_warning_and_retries(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """The reference route answers as it did when its models were loaded at the start; they are loaded when it
    is first called, once, after the staged engine let go of the card."""
    rig = Rig(tmp_path, monkeypatch, multiview=False)
    px, root = rig.px, rig.root
    (root / "images").mkdir(parents=True)
    for n in range(1, 5):
        (root / "images" / f"000{n}.png").write_bytes(b"png")
    seen: list[server.GenerateRequest] = []

    def fake_generate(px_: Any, modes_: Any, notes_: Any, req: server.GenerateRequest, seed: int,
                      defaults: Any) -> dict[str, Any]:
        seen.append(req)
        assert px_.pipeline is not None and px_.loads == 1 and defaults is DEFAULTS
        assert rig.engine.released == len(seen)        # before every reference run the engine lets go of the card
        Path(req.out_path).write_bytes(b"glb")
        multi = len(req.views or []) >= 2
        return {"mode": "multi" if multi else "single", "views": len(req.views or []) if multi else 1,
                "warning": guard.STREAK_WARNING, "ss_retries": 2}

    monkeypatch.setattr(server, "generate", fake_generate)
    client = rig.client
    assert client.get("/spellcast/health").json() == {"ready": True, "low_vram": True, "multiview": False,
                                                      "batch": True}

    img = [str(root / "images" / f"000{n}.png") for n in range(1, 5)]
    body = {"image_path": img[0], "out_path": str(root / "mesh" / "mesh.partial.glb"), "seed": 7}
    two = [{"image_path": img[0], "azimuth": 0}, {"image_path": img[1], "azimuth": 90, "rect": [0, 0, 32, 32]}]

    r = client.post("/spellcast/generate", json={**body, "views": two})
    assert r.status_code == 409 and r.json()["kind"] == "mv_missing" and not seen

    r = client.post("/spellcast/generate", json={**body, "views": [two[1], {"image_path": img[2], "azimuth": 180}]})
    assert r.status_code == 400 and r.json() == {"detail": mv.FRONT_REQUIRED, "kind": "input"}
    assert px.pipeline is None and px.loads == 0                       # a refused request loads nothing

    r = client.post("/spellcast/generate", json=body)
    assert r.status_code == 200
    assert r.json() == {"glb_path": body["out_path"], "seconds": r.json()["seconds"], "seed": 7, "mode": "single",
                        "views": 1, "warning": guard.STREAK_WARNING, "ss_retries": 2}
    assert rig.prepared == [px.pipeline] and rig.reference.loaded      # the launcher's wrappers, once

    fake_snapshot(rig.models)
    assert client.get("/spellcast/health").json()["multiview"] is True
    r = client.post("/spellcast/generate", json={**body, "views": two, "views_dir": str(root / "mesh" / "views")})
    assert r.status_code == 200 and r.json()["mode"] == "multi" and r.json()["views"] == 2
    assert seen[-1].views_dir == str(root / "mesh" / "views")
    assert px.loads == 1 and len(rig.prepared) == 1                    # the second mesh loads nothing


def test_view_problems_and_missing_weights_become_artist_errors() -> None:
    r = server.failure_response(mv.ViewProblem("No prop was found in the Back slice."))
    assert r.status_code == 400 and json.loads(bytes(r.body)) == {"detail": "No prop was found in the Back slice.",
                                                                   "kind": "input"}
    r = server.failure_response(mv.MultiviewMissing(mv.MV_MISSING))
    assert r.status_code == 409 and json.loads(bytes(r.body))["kind"] == "mv_missing"
    # the engine's own text only says the networks are missing; the hub's says what to do about it
    assert server.failure_words("mv_missing", "The multi-view networks are not installed.") == mv.MV_MISSING


def test_the_hub_and_the_launcher_say_the_same_of_a_prop_that_does_not_fit_the_card() -> None:
    """The hub's words win for a prop out of video memory (it knows the prop's quality and offers a lower one only
    while there is one); the launcher's own, for its route and log, are the same sentence."""
    from spellcast.adapters import pixal3d as hub_side

    assert server.OUT_OF_GPU_MEMORY == ("This prop needs more video memory than the card has free. Close other "
                                        "programs that use the graphics card, or lower Quality.")
    assert server.failure_words("oom", "CUDA out of memory") == server.OUT_OF_GPU_MEMORY == hub_side._oom_message(1536)
    assert hub_side._oom_message(1024) == server.OUT_OF_GPU_MEMORY.replace(", or lower Quality", "")


def test_a_deleted_props_folder_is_never_made_again(tmp_path: Path) -> None:
    """The launcher makes a prop's mesh folder (or its views folder) when it went, but never the prop's own folder:
    the artist deleted the prop meanwhile (PropDeleted, an input problem). Any other folder - a check's scratch
    space - is made with its parents and belongs to no prop."""
    class Picture:
        def save(self, path: Path) -> None:
            Path(path).write_bytes(b"png")

        def copy(self) -> "Picture":
            return self

    prop = tmp_path / "P" / "props" / "p_a"
    prop.mkdir(parents=True)
    mv.make_folder(prop / "mesh" / "views")
    assert (prop / "mesh" / "views").is_dir() and not mv.prop_gone(prop / "mesh" / "mesh.1.partial.glb")
    shutil.rmtree(prop)
    assert mv.prop_gone(prop / "mesh" / "mesh.1.partial.glb")
    for folder in (prop / "mesh", prop / "mesh" / "views"):
        with pytest.raises(mv.PropDeleted, match="deleted"):
            mv.make_folder(folder)
    with pytest.raises(mv.PropDeleted):
        mv.save_views(str(prop / "mesh" / "views"), [("front", Picture())])
    assert not prop.exists() and batch.classify(mv.PropDeleted(mv.PROP_DELETED)) == "input"
    mv.make_folder(tmp_path / "scratch" / "out" / "views")
    assert (tmp_path / "scratch" / "out" / "views").is_dir() and not mv.prop_gone(tmp_path / "elsewhere" / "x.glb")
    mv.save_views(str(tmp_path / "scratch" / "views"), [("front", Picture())])
    mv.wait_for_views()
    assert (tmp_path / "scratch" / "views" / "0_front.png").is_file()


def test_a_picture_with_nothing_in_it_is_the_artists_to_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Background removal that finds nothing: the artist is told what to do about it, as for a slice - also when
    a box of nothing comes out as a bare ValueError (numpy or Pillow with no image to make)."""
    class Picture:
        mode, width, height = "RGB", 1024, 768

    def empty(pipeline: Any, picture: Any, crop_first: bool) -> Any:
        raise ValueError("zero-size array to reduction operation minimum which has no identity")

    monkeypatch.setattr(mv, "cut_out_single", empty)
    px = types.SimpleNamespace(_update_progress=lambda stage, n, total: None, pipeline=object())
    with pytest.raises(mv.ViewProblem) as problem:
        server.preprocess(px, Picture())                                        # type: ignore[arg-type]
    assert str(problem.value) == mv.NO_PROP and str(problem.value).startswith("No prop was found in the picture")
    assert batch.classify(problem.value) == "input"


def test_a_transparent_picture_is_cut_out_by_its_own_alpha(capsys: pytest.CaptureFixture[str],
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen-Image's transparent pictures: the log and the progress say no background is removed, when the
    background is transparent (pixal3d_mv.transparent_background); an opaque RGBA picture and a plain one get it
    removed. Both paths go through the launcher's own cut-out: the reference path with Pixal3D's pipeline, a batch
    with the stand-in around the engine's model."""
    image_mod = pytest.importorskip("PIL.Image")
    stages: list[str] = []
    seen: list[tuple[Any, Any, bool]] = []
    monkeypatch.setattr(mv, "cut_out_single", lambda pipeline, p, crop_first: seen.append((pipeline, p, crop_first))
                        or "framed")
    px = types.SimpleNamespace(_update_progress=lambda stage, n, total: stages.append(stage), pipeline=object())
    cut_out = image_mod.new("RGBA", (1024, 768), (0, 0, 0, 0))
    cut_out.paste((200, 100, 50, 255), (300, 100, 700, 700))
    pictures = [cut_out, image_mod.new("RGBA", (1024, 768), (9, 9, 9, 255)), image_mod.new("RGB", (1024, 768))]
    assert [server.preprocess(px, p) for p in pictures] == ["framed"] * 3          # type: ignore[arg-type]
    assert seen == [(px.pipeline, p, True) for p in pictures]
    assert stages == [server.OWN_TRANSPARENCY, server.REMOVING_BACKGROUND, server.REMOVING_BACKGROUND]
    out = capsys.readouterr().out
    assert "Picture: 1024x768 px, own transparency (no background removal)" in out
    assert out.count("px, background removed") == 2
    cutter = server.matte_shim(lambda rgb: rgb)
    server.preprocess(px, pictures[2], cutter)                                      # type: ignore[arg-type]
    assert seen[-1][0] is cutter and cutter.low_vram is False                       # the engine's model stays put
    assert server.describe(server.OWN_TRANSPARENCY, 0, 1) == ("Using the picture's own transparency", 0.0)
    assert server.describe(server.REMOVING_BACKGROUND, 0, 1) == ("Removing the background", 0.0)


def test_one_picture_is_opened_like_a_slice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Single view: a picture without views and one slice both come from open_crop (1024 px on the largest side)
    before the cut-out sees them; what generate() opened already is not opened again."""
    class Picture:
        mode, width, height = "RGB", 1024, 768

        def save(self, path: Path) -> None:
            Path(path).write_bytes(b"png")

        def copy(self) -> "Picture":
            return self

    opened: list[tuple[str, list[int] | None, int]] = []
    crops: list[Picture] = []

    def open_crop(path: str, rect: list[int] | None, azimuth: int = 0) -> Picture:
        opened.append((path, rect, azimuth))
        crops.append(Picture())
        return crops[-1]

    monkeypatch.setattr(mv, "open_crop", open_crop)
    cut: list[Picture] = []
    ran: list[dict[str, Any]] = []
    pipeline = types.SimpleNamespace(run=lambda image, **kwargs: ran.append(kwargs) or (["mesh"], (None, None, 1024)))

    def cut_out_single(by: Any, picture: Picture, crop_first: bool) -> Picture:
        assert by is pipeline and crop_first is True            # the reference path: Pixal3D's own background removal
        cut.append(picture)
        return Picture()

    monkeypatch.setattr(mv, "cut_out_single", cut_out_single)
    px = types.SimpleNamespace(
        TMP_DIR=str(tmp_path), pipeline=pipeline, _update_progress=lambda stage, n, total: None,
        get_camera_params_wild_moge=lambda path, **kwargs: {"camera_angle_x": 0.5, "png": Path(path).is_file()},
        WILD_MESH_SCALE=1.0, WILD_EXTEND_PIXEL=0, WILD_IMAGE_RESOLUTION=512, CASCADE_MAX_NUM_TOKENS=49152)
    defaults = {f"{stage}_{name}": 1 for stage in ("ss", "shape_slat", "tex_slat")
                for name in ("sampling_steps", "guidance_strength", "guidance_rescale", "rescale_t")}

    req = request(tmp_path)
    assert server.run_single_view(px, req, None, 3, defaults) == (["mesh"], 1024)   # type: ignore[arg-type]
    assert opened == [(req.image_path, None, 0)] and cut == [crops[0]]
    assert ran[0]["camera_params"]["png"] and ran[0]["preprocess_image"] is False
    side = view(tmp_path / "images" / "0002.png", 90, [4, 8, 300, 200])
    server.run_single_view(px, request(tmp_path, [side]), side, 3, defaults)        # type: ignore[arg-type]
    assert opened[1] == (side.image_path, [4, 8, 300, 200], 90) and cut[1] is crops[1]
    ready = Picture()
    server.run_single_view(px, req, None, 3, defaults, ready)                       # type: ignore[arg-type]
    assert len(opened) == 2 and cut[2] is ready
    assert not list(tmp_path.glob("spellcast_*.png"))                               # MoGe's file is gone again


def test_a_picture_without_views_reaches_pixal3d_at_1024_px(tmp_path: Path) -> None:
    """On real pictures (Pillow: pixal3d venv): the picture of a prop without views, as the page shows it
    (EXIF rotation applied), and the one slice of a prop with one view."""
    image_mod = pytest.importorskip("PIL.Image")
    (tmp_path / "images").mkdir()
    photo = image_mod.new("RGB", (400, 200), (90, 60, 30))
    exif = photo.getexif()
    exif[0x0112] = 6                                                                # shown turned: 200 x 400
    photo.save(tmp_path / "images" / "0001.png", exif=exif)
    req = request(tmp_path)
    whole = server.open_picture(req, None)
    assert whole.size == (512, 1024) and whole.info["spellcast_unit"] == pytest.approx(1024 / 400)
    part = server.open_picture(req, view(req.image_path, 180, [0, 100, 200, 100]))
    assert part.size == (1024, 512) and part.info["spellcast_unit"] == pytest.approx(1024 / 200)


def test_the_views_say_whether_their_own_transparency_cuts_them_out(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-view: the progress of each view says what cut_out does with it - its own transparency (a transparent
    Qwen-Image picture) or background removal - as the log already did."""
    stages: list[tuple[str, int, int]] = []
    px = types.SimpleNamespace(_update_progress=lambda stage, n, total: stages.append((stage, n, total)))
    crops = {0: "see-through", 90: "opaque"}
    monkeypatch.setattr(mv, "open_crop",
                        lambda path, rect, azimuth: types.SimpleNamespace(info={}, kind=crops[azimuth]))
    monkeypatch.setattr(mv, "real_alpha", lambda crop: crop if crop.kind == "see-through" else None)
    monkeypatch.setattr(mv, "cut_out", lambda pipeline, crop, azimuth: crop)
    monkeypatch.setattr(mv, "frame_views", lambda images, azimuths, sources, units: images)
    monkeypatch.setattr(mv, "save_views", lambda folder, views: None)
    views = [view(tmp_path / "front.png", 0), view(tmp_path / "side.png", 90)]
    server.prepare_views(px, None, request(tmp_path, views), views)   # type: ignore[arg-type]
    assert stages == [(server.OWN_TRANSPARENCY, 0, 2), (server.REMOVING_BACKGROUND, 1, 2)]


def test_the_own_transparency_test_wants_a_transparent_background(tmp_path: Path) -> None:
    """has_own_transparency (pixal3d_mv.transparent_background) agrees with pixal3d_image_to_3d.preprocess_image's
    test (an RGBA picture with any alpha below 255) on a cut-out, an opaque picture and a plain one, and parts
    from it where that test misled: a picture with a speck of transparency is opaque (its background is
    removed; a real picture had 0.05 % of such pixels and came out as a whole frame of "prop"), a
    transparent hole inside a prop on an opaque background is not a transparent background, and a prop cut at
    the picture's bottom edge still has one (Pillow + numpy: pixal3d venv)."""
    image_mod = pytest.importorskip("PIL.Image")
    np = pytest.importorskip("numpy")

    def pixal3ds_test(im: Any) -> bool:   # preprocess_image's has_alpha, as written there
        return im.mode == "RGBA" and not np.all(np.array(im)[:, :, 3] == 255)

    def saved(im: Any) -> Any:
        im.save(tmp_path / "p.png")
        with image_mod.open(tmp_path / "p.png") as again:
            again.load()
            return again

    cut_out = image_mod.new("RGBA", (8, 8), (0, 0, 0, 0))
    cut_out.paste((200, 100, 50, 255), (2, 2, 6, 6))
    for im in (cut_out, image_mod.new("RGBA", (8, 8), (9, 9, 9, 255)), image_mod.new("RGB", (8, 8))):
        again = saved(im)
        assert server.has_own_transparency(again) == pixal3ds_test(again) == mv.transparent_background(again), im.mode
    # Grey with an alpha: judged by its alpha (open_crop makes it RGBA before the cut-out sees it anyway);
    # preprocess_image's test looks at the mode alone and would remove the background of an all-transparent one.
    again = saved(image_mod.new("LA", (8, 8), (5, 0)))
    assert mv.transparent_background(again) and not pixal3ds_test(again)
    speck = image_mod.new("RGBA", (400, 300), (9, 9, 9, 255))                  # 0.05 % of its pixels see-through
    speck.paste((0, 0, 0, 0), (200, 150, 210, 156))
    again = saved(speck)
    assert pixal3ds_test(again) and not server.has_own_transparency(again) and mv.real_alpha(again) is None
    hole = image_mod.new("RGBA", (400, 300), (9, 9, 9, 255))                   # a see-through hole, opaque around
    hole.paste((0, 0, 0, 0), (150, 100, 250, 200))
    assert not mv.transparent_background(saved(hole))
    bottom = image_mod.new("RGBA", (400, 300), (0, 0, 0, 0))                   # a prop cut at the bottom edge
    bottom.paste((200, 100, 50, 255), (100, 60, 300, 300))
    assert mv.transparent_background(saved(bottom))
    faint = image_mod.new("RGBA", (400, 300), (0, 0, 0, 6))                    # nearly, not quite, transparent
    faint.paste((200, 100, 50, 255), (100, 60, 300, 240))
    assert mv.transparent_background(saved(faint))


def test_pixal3ds_own_page_gets_the_single_view_models_back_first() -> None:
    order: list[str] = []

    def generate_3d(image: str, seed: int = 0) -> str:
        order.append("generate_3d")
        return f"{image}:{seed}"

    app = types.SimpleNamespace(_deferred_apis=[(generate_3d, {})])
    server.serialise_ui_calls(types.SimpleNamespace(app=app), InlineGpu(),   # type: ignore[arg-type]
                              before=lambda: order.append("ensure sv"))
    wrapped, _ = app._deferred_apis[0]
    assert wrapped("img", seed=3) == "img:3" and order == ["ensure sv", "generate_3d"]
    assert wrapped.__name__ == "generate_3d"                         # gradio reads the signature


def test_the_views_the_model_saw_replace_the_last_runs(tmp_path: Path) -> None:
    class Picture:
        def save(self, path: Path) -> None:
            Path(path).write_bytes(b"png")

        def copy(self) -> "Picture":
            return self

    folder = tmp_path / "views"
    folder.mkdir()
    for name in ("0_front.png", "3_right.png", "notes.png"):
        (folder / name).write_bytes(b"old")
    mv.save_views(str(folder), [("front", Picture()), ("left", Picture())])
    mv.wait_for_views()
    assert sorted(p.name for p in folder.iterdir()) == ["0_front.png", "1_left.png", "notes.png"]
    assert (folder / "0_front.png").read_bytes() == b"png"
    mv.save_views(None, [("front", Picture())])                      # no folder: nothing to do


def test_the_views_are_written_by_a_thread_of_their_own(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """save_views makes the folder and deletes the earlier views before it returns - a deleted prop still fails at
    once (PropDeleted) - and leaves the PNGs to its writer thread: copies of the pictures, so the caller goes on with
    its own. A second save of the same folder deletes only once the first one's files are written, so no view of
    the first is left behind; a file that cannot be written is only noted, as before."""
    release = threading.Event()
    written: list[tuple[str, str]] = []

    class Picture:
        def __init__(self, data: bytes) -> None:
            self.data = data

        def copy(self) -> "Picture":
            return Picture(self.data)

        def save(self, path: Path) -> None:
            written.append((Path(path).name, threading.current_thread().name))
            if not release.wait(10):
                raise TimeoutError("the test did not let the writer go on")
            if self.data == b"locked":
                raise PermissionError(13, "Permission denied", str(path))
            Path(path).write_bytes(self.data)

    folder = tmp_path / "P" / "props" / "p_a" / "mesh" / "views"
    folder.mkdir(parents=True)
    (folder / "2_back.png").write_bytes(b"old")
    try:
        front = Picture(b"front")
        write = mv.save_views(str(folder), [("front", front), ("left", Picture(b"left"))])
        front.data = b"changed afterwards"
        assert write is not None and not write.done()                # the writer waits; save_views did not
        assert list(folder.iterdir()) == []                          # the earlier views went at once
        release.set()
        mv.wait_for_views()
        assert {p.name: p.read_bytes() for p in folder.iterdir()} == {"0_front.png": b"front", "1_left.png": b"left"}
        assert {thread for _, thread in written} == {"spellcast-views_0"}

        release.clear()
        mv.save_views(str(folder), [("front", Picture(b"a")), ("back", Picture(b"b")), ("right", Picture(b"c"))])
        threading.Timer(0.2, release.set).start()
        mv.save_views(str(folder), [("front", Picture(b"again"))])   # waits for the three files, then deletes them
        mv.wait_for_views()
        assert {p.name: p.read_bytes() for p in folder.iterdir()} == {"0_front.png": b"again"}

        mv.save_views(str(folder), [("front", Picture(b"locked")), ("left", Picture(b"left"))])
        mv.wait_for_views()
        assert "Note: could not save the views the model saw in" in capsys.readouterr().out
        assert list(folder.iterdir()) == []                          # the write stopped there, as it did

        shutil.rmtree(folder.parents[1])
        with pytest.raises(mv.PropDeleted):
            mv.save_views(str(folder), [("front", Picture(b"front"))])
        assert not folder.parents[1].exists()
    finally:
        release.set()                                                # never leave the writer waiting
        mv.wait_for_views()


class Opening:
    """Stand-ins for the opening of a prop's pictures and their cut-out (pixal3d_mv), with where and in which order
    each ran. Opening `fail[name]` raises that exception."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, fail: dict[str, BaseException] | None = None) -> None:
        self.opened: list[tuple[str, str]] = []         # (picture, thread)
        self.cut: list[tuple[str, str]] = []            # (picture, thread), in the order of the mattes
        self.fail = fail or {}
        self.lock = threading.Lock()
        monkeypatch.setattr(mv, "open_crop", self.open_crop)
        monkeypatch.setattr(mv, "real_alpha", lambda crop: None)
        monkeypatch.setattr(mv, "cut_out", self.cut_out)
        monkeypatch.setattr(mv, "frame_views", lambda images, azimuths, sources, units: images)
        monkeypatch.setattr(mv, "cut_out_single", lambda cutter, crop, crop_first: mv.cut_out(cutter, crop))

    def open_crop(self, path: str, rect: list[int] | None, azimuth: int = 0) -> Any:
        name = Path(path).name
        with self.lock:
            self.opened.append((name, threading.current_thread().name))
        if name in self.fail:
            raise self.fail[name]
        return types.SimpleNamespace(name=name, info={}, mode="RGB", width=1024, height=1024)

    def cut_out(self, pipeline: Any, crop: Any, azimuth: int = 0) -> Any:
        pipeline.rembg_model(crop)                      # the matte, on the GPU thread
        self.cut.append((crop.name, threading.current_thread().name))
        return crop


def lined_up(tmp_path: Path) -> list[Any]:
    """Four props as the engine's cut-out takes them, the ones made from several views first; none saves views."""
    props = []
    for prop_id, pictures in (("p_a", ("a0.png", "a1.png")), ("p_c", ("c0.png", "c1.png", "c2.png")),
                              ("p_b", ("b.png",)), ("p_d", ("d.png",))):
        folder = tmp_path / "props" / prop_id
        folder.mkdir(parents=True, exist_ok=True)
        views = [view(folder / name, azimuth) for name, azimuth in zip(pictures, (0, 180, 270))]
        payload = server.BatchItem(id=prop_id, image_path=str(folder / pictures[0]),
                                   out_path=str(folder / "mesh" / "mesh.partial.glb"), views=views)
        props.append(Staged.PropRequest(prop_id, 7, 1024, len(pictures) > 1, {}, payload=payload))
    return props


def cut_out_in_turn(hooks: Any, props: list[Any], ahead: bool) -> dict[str, Any]:
    """The engine's cut-out on `props`: each prop's prepare in turn, its outcome by prop - "prepared", or the
    exception's type, words and kind. With `ahead` the hooks are told the part's start and end, as BatchRunner
    tells them."""
    outcomes: dict[str, Any] = {}
    if ahead:
        hooks.ahead.begin(props)
        hooks.ahead.on_event(Staged.Event("part_start", "cutout", detail={"props": [p.id for p in props]}))
    for prop in props:
        try:
            hooks.prepare(prop, lambda rgb: rgb)
            outcomes[prop.id] = "prepared"
        except Exception as exc:
            outcomes[prop.id] = (type(exc).__name__, str(exc), getattr(exc, "kind", None))
    if ahead:
        hooks.ahead.on_event(Staged.Event("part_end", "cutout"))
        hooks.ahead.end()
    return outcomes


def test_a_batch_opens_the_next_props_pictures_while_the_card_cuts_out_one(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """The cut-out of a batch: the pictures of the next two props in line are opened on the opening threads while
    the GPU thread cuts out the one before them - no more than two props' at a time, each picture once - and the
    mattes come in the order they came in without it."""
    monkeypatch.setattr(server, "staged_module", lambda: Staged)
    px = types.SimpleNamespace(_update_progress=lambda stage, n, total: None)
    props = lined_up(tmp_path)
    plain = Opening(monkeypatch)
    assert cut_out_in_turn(server.BatchHooks(px, server.RunNotes()), props, ahead=False) == dict.fromkeys(
        ["p_a", "p_c", "p_b", "p_d"], "prepared")
    here = threading.current_thread().name
    assert {thread for _, thread in plain.opened + plain.cut} == {here}

    opening = Opening(monkeypatch)
    hooks = server.BatchHooks(px, server.RunNotes())
    waiting: list[list[str]] = []
    real_cut_out = opening.cut_out

    def cut_out(pipeline: Any, crop: Any, azimuth: int = 0) -> Any:
        waiting.append(sorted(hooks.ahead._opened))     # the props whose pictures wait, or are being opened
        return real_cut_out(pipeline, crop, azimuth)

    monkeypatch.setattr(mv, "cut_out", cut_out)
    hooks.ahead.begin(props)
    hooks.ahead.on_event(Staged.Event("part_start", "cutout", detail={"props": [p.id for p in props]}))
    assert sorted(hooks.ahead._opened) == ["p_a", "p_c"]
    for prop in props:
        hooks.prepare(prop, lambda rgb: rgb)
    hooks.ahead.on_event(Staged.Event("part_end", "cutout"))
    assert [name for name, _ in opening.cut] == [name for name, _ in plain.cut] == [
        "a0.png", "a1.png", "c0.png", "c1.png", "c2.png", "b.png", "d.png"]
    assert {thread for _, thread in opening.cut} == {here}
    assert all(thread.startswith("pixal3d-open") for _, thread in opening.opened)
    assert sorted(name for name, _ in opening.opened) == sorted(name for name, _ in opening.cut)
    assert waiting == [["p_b", "p_c"]] * 2 + [["p_b", "p_d"]] * 3 + [["p_d"], []]
    assert not hooks.ahead._opened


def test_a_picture_that_cannot_be_opened_ahead_fails_its_prop_alone_with_the_same_words(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Opened ahead, a slice to fix or a missing picture raises in its own prop's prepare, at that view's turn, as
    it did when the GPU thread opened it: the same exception and words, the views before it cut out, the other
    props prepared."""
    monkeypatch.setattr(server, "staged_module", lambda: Staged)
    px = types.SimpleNamespace(_update_progress=lambda stage, n, total: None)
    fail: dict[str, BaseException] = {
        "a1.png": mv.ViewProblem("The Back slice lies outside its picture: slice it again."),
        "c2.png": FileNotFoundError(2, "No such file or directory", "c2.png")}
    told = {}
    for ahead in (False, True):
        opening = Opening(monkeypatch, fail)
        told[ahead] = (cut_out_in_turn(server.BatchHooks(px, server.RunNotes()), lined_up(tmp_path), ahead),
                       [name for name, _ in opening.cut])
    assert told[True] == told[False]
    outcomes, cut = told[True]
    assert outcomes == {"p_a": ("PropFailed", "The Back slice lies outside its picture: slice it again.", "input"),
                        "p_c": ("FileNotFoundError", "[Errno 2] No such file or directory: 'c2.png'", None),
                        "p_b": "prepared", "p_d": "prepared"}
    assert cut == ["a0.png", "c0.png", "c1.png", "b.png", "d.png"]


def test_a_prop_deleted_or_a_batch_stopped_lets_go_of_the_pictures_opened_ahead(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A prop deleted while its pictures were opened ahead fails as before (PropFailed, kind "input"), nothing of it
    is cut out, and the next props go on. A batch that stops (cancelled, failed) drops what waits; a prop that is
    prepared again (the engine's second try after the card ran out of memory) or outside a batch opens its own."""
    monkeypatch.setattr(server, "staged_module", lambda: Staged)
    px = types.SimpleNamespace(_update_progress=lambda stage, n, total: None)
    props = lined_up(tmp_path)
    opening = Opening(monkeypatch)
    hooks = server.BatchHooks(px, server.RunNotes())
    hooks.ahead.begin(props)
    hooks.ahead.on_event(Staged.Event("part_start", "cutout", detail={"props": [p.id for p in props]}))
    shutil.rmtree(tmp_path / "props" / "p_a")
    with pytest.raises(Staged.PropFailed) as failed:
        hooks.prepare(props[0], lambda rgb: rgb)
    assert (failed.value.detail, failed.value.kind) == (mv.PROP_DELETED, "input") and opening.cut == []
    assert sorted(hooks.ahead._opened) == ["p_b", "p_c"]
    hooks.prepare(props[1], lambda rgb: rgb)
    assert [name for name, _ in opening.cut] == ["c0.png", "c1.png", "c2.png"]
    waiting = [future for futures in hooks.ahead._opened.values() for future in futures]
    hooks.ahead.end()                                   # the batch was cancelled: BatchRunner._run ends it
    assert not hooks.ahead._opened and waiting
    concurrent.futures.wait(waiting)                    # cancelled, or let go once opened
    here, before = threading.current_thread().name, len(opening.opened)
    for prop in (props[1], props[2]):                   # again, and outside a batch: opened on the GPU thread
        hooks.prepare(prop, lambda rgb: rgb)
    assert opening.opened[before:] == [(name, here) for name in ("c0.png", "c1.png", "c2.png", "b.png")]
    assert [name for name, _ in opening.cut[-4:]] == ["c0.png", "c1.png", "c2.png", "b.png"]


# ---- batches: the routes and the record ----------------------------------------------------------------------

def test_the_parts_and_their_shares_of_a_props_bar() -> None:
    """What the hub, its page and its stand-in tool (tests/fake_tool.py) take the parts to be."""
    assert [(p.key, p.label, p.start, p.end) for p in batch.PARTS] == [
        ("cutout", "Removing the background", 0.00, 0.03), ("camera", "Estimating the camera angle", 0.03, 0.05),
        ("features", "Reading the picture", 0.05, 0.06), ("ss", "Blocking out the shape", 0.06, 0.15),
        ("cond_lr", "Preparing the shape", 0.15, 0.16), ("lr", "Sculpting the shape", 0.16, 0.29),
        ("upsample", "Sculpting the shape", 0.29, 0.30), ("cond_hr", "Preparing the details", 0.30, 0.32),
        ("hr", "Refining shape details", 0.32, 0.55), ("tex", "Painting the texture", 0.55, 0.72),
        ("finish", "Building the 3D mesh", 0.72, 1.00)]
    assert batch.PART_KEYS == Staged.PARTS and batch.DECODED == 0.82 == server.describe("Extracting GLB", 0, 6)[1]
    assert all(a.end == b.start for a, b in zip(batch.PARTS, batch.PARTS[1:]))      # a bar without gaps
    import fake_tool                                                                # tests/: the hub's stand-in

    assert [tuple(p) for p in batch.PARTS] == [tuple(p) for p in fake_tool.PARTS]
    assert batch.RESTING_MESSAGE == fake_tool.RESTING == "Waiting for the next step"
    assert batch.ON_CARD == fake_tool.ON_CARD and mv.PROP_DELETED == fake_tool.PROP_DELETED
    # The reference path's bars (PHASES) take their words and shares from the parts: the same table as before
    # they did, and a step that spans several parts spans their shares.
    assert [(label, start, end) for _, label, start, end in server.PHASES] == [
        ("Swapping the single-view models for the multi-view ones", 0.00, 0.03),
        ("Swapping the multi-view models for the single-view ones", 0.00, 0.03),
        ("Removing the background", 0.00, 0.03), ("Using the picture's own transparency", 0.00, 0.03),
        ("Estimating the camera angle", 0.03, 0.06), ("Blocking out the shape", 0.06, 0.15),
        ("Sculpting the shape", 0.15, 0.30), ("Refining shape details", 0.30, 0.55),
        ("Painting the texture", 0.55, 0.72), ("Building the 3D mesh (this step can take a few minutes)", 0.72, 0.82),
        ("Cleaning up the mesh and baking textures", 0.82, 0.98), ("Saving the GLB file", 0.98, 1.00)]
    for stage, first, last in (("Sampling sparse structure", "ss", "ss"),
                               ("Sampling shape SLat", "cond_lr", "upsample"),
                               ("Sampling HR shape SLat", "cond_hr", "hr"), ("Sampling texture SLat", "tex", "tex")):
        assert server.describe(stage, 0, 4) == (batch.PART[last].label, batch.PART[first].start)
        assert server.describe(stage, 4, 4) == (batch.PART[last].label, pytest.approx(batch.PART[last].end))
    assert batch.network_label("hr_flow.mv") == "detail model (multi-view)"
    assert batch.network_label("lr_flow.sv") == "shape model"
    assert batch.network_label("rembg") == "background removal model"


def test_health_is_ready_once_the_engine_exists_and_no_model_was_read(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    # "ready" is always true over HTTP: main() stops before serving when the engine cannot be made.
    rig = Rig(tmp_path, monkeypatch, multiview=False)
    health = lambda: rig.client.get("/spellcast/health").json()                     # noqa: E731
    assert health() == {"ready": True, "low_vram": True, "multiview": False, "batch": True}
    assert rig.px.pipeline is None and rig.px.loads == 0 and not rig.reference.loaded
    assert rig.engine.hooks is rig.runner.hooks and rig.engine.options.models == rig.models
    assert "sample_sparse_structure" in vars(rig.engine)                            # the guard is on the engine
    fake_snapshot(rig.models)                                                       # the multi-view download ended
    assert health()["multiview"] is True


def test_a_batch_is_taken_and_stays_readable_until_the_next_one(rig: Rig) -> None:
    item = rig.prop("p_a")
    r = rig.post([item])
    assert r.status_code == 202
    assert r.json() == {"batch_id": "j1", "order": ["p_a"], "parts": [
        {"key": p.key, "label": p.label, "start": p.start, "end": p.end} for p in batch.PARTS]}
    told = rig.told()
    assert told == {
        "batch_id": "j1", "state": "done", "cancelling": False, "seq": told["seq"], "seconds": told["seconds"],
        "part": None, "current": None, "loading": None, "error": None,
        "items": {"p_a": {"state": "done", "part": "finish", "parts_done": 11, "fraction": 1.0, "message": "",
                          "step": 0, "total": 0,
                          "result": {"glb_path": item["out_path"], "seconds": 1.5, "seed": 7, "mode": "single",
                                     "views": 1, "warning": None, "ss_retries": 0, "resolution": 1024, "slow": [],
                                     "resumed": False}}}}
    assert told["seq"] > 40 and told["seconds"] >= 0
    assert Path(item["out_path"]).read_bytes() == b"glTF"
    run = rig.engine.runs[0]
    assert run["work_dir"] == rig.root / "P" / "temp" / "mesh-j1" and run["work_dir"].is_dir()
    assert run["wave_size"] is None and run["resume"] is False
    # the prop's own folder in the work folder, where its texture volume waits for the bake; none between batches
    assert rig.work_folders == [run["work_dir"] / "p_a"] and rig.runner.hooks.work_dir is None

    assert rig.client.get("/spellcast/batch/j0").status_code == 404                 # never heard of
    r = rig.client.post("/spellcast/batch/j0/cancel")
    assert r.status_code == 404 and r.json()["kind"] == "input"
    assert rig.post([rig.prop("p_b")], "j2").status_code == 202
    assert rig.client.get("/spellcast/batch/j1").status_code == 404                 # its place was taken
    assert rig.told("j2")["items"]["p_b"]["state"] == "done"


def test_a_batch_request_that_is_wrong_as_a_whole_is_refused(rig: Rig, tmp_path: Path) -> None:
    good = rig.prop("p_a")

    def refused(**sent: Any) -> str:
        r = rig.client.post("/spellcast/batch", **sent)
        assert r.status_code == 400 and r.json()["kind"] == "input" and r.json()["detail"], r.text
        return r.json()["detail"]

    assert refused(content=b"{no json") == "The batch request is not JSON."
    assert "items" in refused(json={"batch_id": "j1", "work_dir": rig.body([])["work_dir"]})
    assert "seed" in refused(json=rig.body([{**good, "seed": "seven"}]))
    assert refused(json=rig.body([])) == "The batch has no props."
    assert refused(json=rig.body([good, good])) == "A prop is in the batch twice."
    # A seed drawn here would differ in the job's second attempt, and nothing sampled could be resumed: the hub
    # draws it (adapters/pixal3d.py), and a batch with a negative one is refused.
    assert refused(json=rig.body([good, {**rig.prop("p_b"), "seed": -1}])) == (
        "The seed of prop 'p_b' is -1: a batch takes seeds of 0 or more (the hub draws a random one itself).")
    assert "'../p_a'" in refused(json=rig.body([{**good, "id": "../p_a"}]))            # it names a folder
    assert "'j/1'" in refused(json=rig.body([good], "j/1"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "0001.png").write_bytes(b"png")
    for key, path in (("image_path", elsewhere / "0001.png"), ("out_path", elsewhere / "mesh.glb"),
                      ("views_dir", elsewhere / "views")):
        assert refused(json=rig.body([{**good, key: str(path)}])) == server.OUTSIDE_PROJECTS, key
    view = {"image_path": str(elsewhere / "0001.png"), "rect": None, "azimuth": 0}
    assert refused(json=rig.body([{**good, "views": [view]}])) == server.OUTSIDE_PROJECTS
    # The engine empties the work folder: only <projects>/<project>/temp/mesh-<job id> will do.
    project = rig.root / "P"
    assert refused(json={**rig.body([good]), "work_dir": str(elsewhere / "temp" / "mesh-j1")}) == server.OUTSIDE_PROJECTS
    for work in (project / "mesh-j1", project / "temp" / "bake-j1", project / "temp", project / "props" / "p_a",
                 project / "temp" / "mesh-j1" / "p_a", rig.root / "temp" / "mesh-j1",
                 project / "props" / "temp" / "mesh-j1"):
        assert refused(json={**rig.body([good]), "work_dir": str(work)}).startswith("This is not the work folder"), work
        assert not work.exists() or work == project / "props" / "p_a"
    assert not rig.engine.runs and rig.client.get("/spellcast/batch/j1").status_code == 404


def test_a_prop_whose_own_request_is_wrong_fails_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = Rig(tmp_path, monkeypatch, multiview=False)
    good = rig.prop("p_good")
    lost = rig.prop("p_lost")
    Path(lost["image_path"]).unlink()
    coarse = rig.prop("p_coarse", resolution=512)
    frontless = rig.prop("p_frontless", views=(90, 180))
    several = rig.prop("p_several", views=(0, 90))                # its multi-view models are not downloaded
    named = rig.prop("p_named", out_path=str(rig.root / "P" / "props" / "p_named" / "mesh" / "mesh.obj"))
    r = rig.post([lost, coarse, good, frontless, several, named])
    assert r.status_code == 202
    assert r.json()["order"] == ["p_frontless", "p_several", "p_lost", "p_coarse", "p_good", "p_named"]
    told = rig.told()
    assert told["state"] == "done" and told["error"] is None
    failed = {prop_id: item.get("error") for prop_id, item in told["items"].items() if item["state"] == "failed"}
    assert failed == {
        "p_lost": {"detail": f"The input image was not found: {lost['image_path']}. Choose the image again and "
                             "retry.", "kind": "input"},
        "p_coarse": {"detail": "Quality must be 1024 or 1536 (got 512).", "kind": "input"},
        "p_frontless": {"detail": mv.FRONT_REQUIRED, "kind": "input"},
        "p_several": {"detail": mv.MV_MISSING, "kind": "mv_missing"},
        "p_named": {"detail": f"The output file must end in .glb (got {named['out_path']}).", "kind": "input"}}
    assert told["items"]["p_lost"] == {"state": "failed", "part": None, "parts_done": 0, "fraction": 0.0,
                                       "message": "", "step": 0, "total": 0, "error": failed["p_lost"]}
    assert [p.id for p in rig.engine.runs[0]["props"]] == ["p_good"]          # the engine never sees the others
    assert told["items"]["p_good"]["state"] == "done"
    assert not Path(named["out_path"]).parent.exists()                        # nothing was made for a wrong prop

    assert rig.post([lost, coarse], "j2").status_code == 202                  # nothing to make: over at once
    assert rig.told("j2")["state"] == "failed" and len(rig.engine.runs) == 1  # not one prop was made


def test_the_engine_is_asked_for_each_prop_as_the_reference_path_would_make_it(rig: Rig) -> None:
    """Props made from several views go first inside every part (the engine's own order: the answer says it);
    each request carries the seed used, the hub's sampler settings and the prop's item."""
    items = [rig.prop("p_a", seed=123), rig.prop("p_b", views=(0, 90)), rig.prop("p_c", resolution=1536),
             rig.prop("p_d", views=(180, 0, 270), seed=11)]
    r = rig.post(items, wave_size=0)
    assert r.json()["order"] == ["p_b", "p_d", "p_a", "p_c"]
    run = rig.engine.runs[0]
    assert run["wave_size"] == 0                                              # 0: no waves, said in so many words
    props = {p.id: p for p in run["props"]}
    assert list(props) == ["p_a", "p_b", "p_c", "p_d"]                        # as sent: the engine orders them
    assert [p.multi for p in props.values()] == [False, True, False, True]
    assert [p.resolution for p in props.values()] == [1024, 1024, 1536, 1024]
    assert props["p_a"].seed == 123 and props["p_b"].seed == 7 and props["p_d"].seed == 11
    sampler = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}
    for prop in props.values():
        assert prop.sampler_params == {"ss": sampler, "shape": sampler, "tex": sampler}
        assert prop.max_num_tokens == 49152 and isinstance(prop.payload, server.BatchItem)
    assert props["p_d"].payload.views_dir == items[3]["views_dir"] and props["p_d"].payload.texture_size == 2048
    assert len({p.request_hash for p in props.values()}) == 4 and all(len(p.request_hash) == 64 for p in props.values())
    results = {prop_id: item["result"] for prop_id, item in rig.told()["items"].items()}
    assert [(results[i]["mode"], results[i]["views"]) for i in props] == [("single", 1), ("multi", 2), ("single", 1),
                                                                         ("multi", 3)]
    assert results["p_a"]["seed"] == props["p_a"].seed and results["p_c"]["resolution"] == 1536


def test_a_props_bar_rests_at_the_end_of_each_part_and_never_goes_back(rig: Rig) -> None:
    seen: list[dict[str, Any]] = []
    rig.engine.watch = lambda: seen.append(rig.runner.batches.get("j1").snapshot())
    rig.post([rig.prop("p_a"), rig.prop("p_b", views=(0, 90))])
    assert [told["seq"] for told in seen] == sorted({told["seq"] for told in seen})      # every event is a change
    for prop_id in ("p_a", "p_b"):
        entries = [told["items"][prop_id] for told in seen]
        fractions = [entry["fraction"] for entry in entries]
        assert fractions == sorted(fractions) and fractions[0] == 0.0 and fractions[-1] == 1.0
        assert entries[0]["state"] == "waiting" and entries[-1]["state"] == "done"
        for entry in entries:
            if entry["state"] == "resting":
                part = batch.PART[entry["part"]]
                assert entry["fraction"] == part.end and entry["message"] == "Waiting for the next step"
                assert entry["parts_done"] == batch.PART_KEYS.index(part.key) + 1
                assert entry["step"] == entry["total"] == 0
            elif entry["state"] == "working":
                assert batch.PART[entry["part"]].start <= entry["fraction"] <= batch.PART[entry["part"]].end
    by_part = lambda prop_id: {told["items"][prop_id]["part"] for told in seen}           # noqa: E731
    assert "camera" in by_part("p_a") and "camera" not in by_part("p_b")      # views bring their own cameras
    # one answer in the middle: the multi-view prop went first and rests; the other is at step 2 of 3
    told = next(t for t in seen if t["items"]["p_a"]["part"] == "hr" and t["items"]["p_a"]["step"] == 2)
    assert told["state"] == "running" and told["current"] == "p_a" and told["loading"] is None
    assert told["part"] == {"key": "hr", "index": 8, "count": 11, "label": "Refining shape details", "position": 2,
                            "of": 2}
    assert told["items"]["p_a"] == {"state": "working", "part": "hr", "parts_done": 8,
                                    "fraction": round(0.32 + 0.23 * 2 / 3, 4),
                                    "message": "Refining shape details · step 3 of 3", "step": 2, "total": 3}
    assert told["items"]["p_b"] == {"state": "resting", "part": "hr", "parts_done": 9, "fraction": 0.55,
                                    "message": "Waiting for the next step", "step": 0, "total": 0}
    # a network being read from disk is named, and the name goes when it is on the card
    assert {t["loading"] for t in seen} >= {None, "background removal model", "detail model (multi-view)",
                                            "texture decoder"}
    reading = next(t for t in seen if t["loading"] == "camera model")
    assert reading["part"]["key"] == "camera" and reading["part"]["of"] == 1 and reading["current"] is None
    # the engine's three steps of "finish" take the bar to 0.82
    # (the three decode steps done: the message names the hook's first step, of 3 + pixal3d_glb.STEPS)
    decoded = [t["items"]["p_b"] for t in seen if t["items"]["p_b"]["message"] == "Building the 3D mesh · step 4 of 8"]
    assert decoded and decoded[0]["fraction"] == 0.82


def test_every_bar_of_the_mesh_clean_up_counts_as_a_change(rig: Rig) -> None:
    """The hub takes a batch whose `seq` stands still for long as hung; saving a mesh takes a minute and more."""
    during: list[dict[str, Any]] = []
    rig.on_bar = lambda: during.append(rig.runner.batches.get("j1").snapshot())
    rig.post([rig.prop("p_a")])
    assert len(during) == 6 and [d["seq"] for d in during] == sorted({d["seq"] for d in during})
    assert all(d["current"] == "p_a" and d["part"]["key"] == "finish" for d in during)
    assert [(d["items"]["p_a"]["message"], d["items"]["p_a"]["fraction"]) for d in during] == [
        (f"Cleaning up the mesh and baking textures · step {n} of 6", round(0.82 + 0.16 * n / 6, 4)) for n in range(1, 7)]
    assert during[2]["items"]["p_a"] == {"state": "working", "part": "finish", "parts_done": 10, "fraction": 0.9,
                                         "message": "Cleaning up the mesh and baking textures · step 3 of 6", "step": 3,
                                         "total": 6}
    assert rig.px.progress == []                    # nothing of a batch goes to the reference path's progress file
    rig.px._update_progress("Sampling shape SLat (proj)", 3, 12)                # no batch at work: app.py's own
    assert rig.px.progress == [("Sampling shape SLat (proj)", 3, 12)]


def test_a_bar_keeps_inside_its_part_and_an_unknown_one_only_counts() -> None:
    record = batch.Batch("j1", [batch.Item("p_a")], server.failure_words)
    record.on_event(Staged.Event("part_start", "cutout", detail={"props": ["p_a"]}))
    record.on_event(Staged.Event("prop_start", "cutout", "p_a"))
    seq = record.snapshot()["seq"]
    record.bar(*server.describe(server.OWN_TRANSPARENCY, 0, 1), 0, 1)
    told = record.snapshot()
    assert told["items"]["p_a"]["message"] == "Using the picture's own transparency" and told["seq"] == seq + 1
    record.bar("Cleaning up the mesh and baking textures", 0.9, 2, 6)           # a bar from another part's range
    assert record.snapshot()["items"]["p_a"]["fraction"] == 0.03
    record.bar(None, None, 1, 4)                                                 # a bar the launcher has no words for
    told = record.snapshot()
    assert told["seq"] == seq + 3 and told["items"]["p_a"]["message"] == "Cleaning up the mesh and baking textures · step 2 of 6"
    record.on_event(Staged.Event("prop_rest", "cutout", "p_a"))
    record.bar("Saving the GLB file", 0.98, 0, 1)                                # nobody is on the card: no prop's
    assert record.snapshot()["items"]["p_a"]["message"] == "Waiting for the next step"
    record.on_event(Staged.Event("part_start", "camera", detail={"props": []}))  # a part no prop goes through
    assert record.snapshot()["part"]["key"] == "cutout"
    record.on_event(Staged.Event("net_load", "features", detail={"name": "dino", "bytes_read": 0, "bytes": 9}))
    assert record.snapshot()["loading"] == "picture model"
    record.on_event(Staged.Event("prop_start", "features", "p_a"))               # no "ready" came: a prop works
    assert record.snapshot()["loading"] is None and record.snapshot()["part"]["key"] == "features"


def test_a_record_lets_go_of_the_loading_words_and_counts_the_waves(capsys: pytest.CaptureFixture[str]) -> None:
    """A checkpoint that cannot be read fails the props of its group without ever starting one: the page must
    stop saying the model is being read. With waves on, the parts are counted over all the waves together."""
    record = batch.Batch("j1", [batch.Item("p_a"), batch.Item("p_b")], server.failure_words)
    assert record.snapshot()["items"]["p_a"]["message"] == "Waiting for its turn"
    record.on_event(Staged.Event("part_start", "hr", detail={"props": ["p_a", "p_b"], "wave": 2, "waves": 3}))
    told = record.snapshot()
    assert told["part"]["index"] == 11 + 8 and told["part"]["count"] == 33       # never back to the first part
    record.on_event(Staged.Event("net_load", "hr", detail={"name": "hr_flow.sv"}))
    assert record.snapshot()["loading"] == "detail model"
    for prop_id in ("p_a", "p_b"):
        record.on_event(Staged.Event("prop_failed", "hr", prop_id,
                                     detail={"kind": "missing_file", "detail": "hr_flow.safetensors"}))
    assert record.snapshot()["loading"] is None
    record.on_event(Staged.Event("net_load", "tex", detail={"name": "tex_flow.sv"}))
    record.on_event(Staged.Event("part_start", "tex", detail={"props": ["p_b"], "wave": 3, "waves": 3}))
    assert record.snapshot()["loading"] is None                                  # the group was entered
    record.on_event(Staged.Event("store_warning", "tex", detail={"what": "write", "key": "slat",
                                                                 "message": "the disk is full"}))
    assert "Pixal3D's temporary files: the disk is full." in capsys.readouterr().out


def test_the_prop_on_the_card_is_written_to_the_work_folder_whenever_it_changes(rig: Rig, tmp_path: Path,
                                                                                capsys: pytest.CaptureFixture[str]
                                                                                ) -> None:
    """No poll is answered while native code saves a mesh, so the hub's last answer can be older than the prop that
    was on the card when Pixal3D died: the launcher keeps that prop, and the last one it made, in a file of the
    work folder (pixal3d_batch.ON_CARD), which the hub reads then (adapters/pixal3d.py, the same name)."""
    from spellcast.adapters import pixal3d as hub_side

    assert batch.ON_CARD == hub_side.ON_CARD
    note = rig.root / "P" / "temp" / "mesh-j1" / batch.ON_CARD
    note.parent.mkdir(parents=True)
    stale = {"batch_id": "j1", "current": "p_x", "finished": None}         # what the job's first attempt left
    note.write_text(json.dumps(stale), encoding="utf-8")
    seen: list[tuple[dict[str, Any], dict[str, Any]]] = []
    rig.engine.watch = lambda: seen.append((rig.runner.batches.get("j1").snapshot(),
                                            json.loads(note.read_text(encoding="utf-8"))))
    rig.post([rig.prop("p_a"), rig.prop("p_b")])
    assert seen[0][1] == {"batch_id": "j1", "current": None, "finished": None}      # written afresh at the start
    assert all(told["current"] == written["current"] for told, written in seen)     # after every event
    assert {written["current"] for _, written in seen} == {None, "p_a", "p_b"}
    made = [written["finished"]["id"] for _, written in seen if written["finished"]]
    assert made[0] == "p_a" and made[-1] == "p_b"
    final = json.loads(note.read_text(encoding="utf-8"))
    assert final == {"batch_id": "j1", "current": None,
                     "finished": {"id": "p_b", "result": rig.told()["items"]["p_b"]["result"]}}
    assert not list(note.parent.glob("*.tmp"))

    # A batch the engine never sees (every prop failed at once) has its note too; a note that cannot be written
    # costs the batch nothing.
    rig.post([{**rig.prop("p_c"), "resolution": 512}], "j2")
    second = rig.root / "P" / "temp" / "mesh-j2" / batch.ON_CARD
    assert json.loads(second.read_text(encoding="utf-8"))["current"] is None
    record = batch.Batch("j9", [batch.Item("p_a")], server.failure_words, marker=tmp_path / "nowhere" / batch.ON_CARD)
    record.on_event(Staged.Event("part_start", "cutout", detail={"props": ["p_a"]}))
    record.on_event(Staged.Event("prop_start", "cutout", "p_a"))
    assert record.snapshot()["current"] == "p_a"
    assert "could not write which prop is on the card" in capsys.readouterr().out


def test_a_prop_the_engine_never_reached_is_not_blamed() -> None:
    """A batch that made nothing says so, and a prop that never had its turn is not told it failed."""
    record = batch.Batch("j1", [batch.Item("p_a")], server.failure_words)
    record.finish({})
    told = record.snapshot()
    assert told["state"] == "failed"
    assert told["items"]["p_a"]["error"] == {"detail": "Pixal3D did not get to this prop. Try making it again.",
                                             "kind": "stopped"}


def test_one_batch_at_a_time(threaded: Rig) -> None:
    rig, gate = threaded, Gate()
    rig.engine.at[("lr", "p_a")] = gate
    assert rig.post([rig.prop("p_a")]).status_code == 202                        # answered before the work starts
    gate.wait()
    told = rig.told()
    assert told["state"] == "running" and told["current"] == "p_a" and not told["cancelling"]
    assert told["part"] == {"key": "lr", "index": 5, "count": 11, "label": "Sculpting the shape", "position": 1,
                            "of": 1}
    assert told["items"]["p_a"] == {"state": "working", "part": "lr", "parts_done": 5, "fraction": 0.16,
                                    "message": "Sculpting the shape", "step": 0, "total": 0}
    other = rig.prop("p_b")
    r = rig.post([other], "j2")
    assert r.status_code == 409
    assert r.json() == {"detail": "Pixal3D is still working on another job.", "kind": "busy", "batch_id": "j1"}
    assert not Path(other["out_path"]).parent.exists() and rig.told()["state"] == "running"
    gate.open()
    assert rig.ended()["items"]["p_a"]["state"] == "done"
    assert rig.post([other], "j2").status_code == 202 and rig.ended("j2")["state"] == "done"


def test_a_cancel_stops_the_batch_at_the_next_check_point(threaded: Rig) -> None:
    rig, gate = threaded, Gate()
    rig.engine.at[("lr", "p_b")] = gate
    a, b = rig.prop("p_a"), rig.prop("p_b")
    rig.post([a, b])
    gate.wait()                                             # p_a rests after "lr", p_b is in it
    seq = rig.told()["seq"]
    r = rig.client.post("/spellcast/batch/j1/cancel")
    assert r.status_code == 200 and r.json() == {"cancelling": True}
    told = rig.told()
    assert told["state"] == "running" and told["cancelling"] is True and told["seq"] == seq + 1
    assert rig.client.post("/spellcast/batch/j1/cancel").json() == {"cancelling": True}     # asked again: the same
    assert rig.told()["seq"] == seq + 1
    gate.open()
    told = rig.ended()
    assert told["state"] == "cancelled" and told["cancelling"] is True and told["error"] is None
    assert told["part"] is None and told["current"] is None
    assert {prop_id: (item["state"], item["message"]) for prop_id, item in told["items"].items()} == {
        "p_a": ("cancelled", ""), "p_b": ("cancelled", "")}
    assert told["items"]["p_a"]["fraction"] == 0.29 and "result" not in told["items"]["p_a"]
    assert not Path(a["out_path"]).exists()
    assert rig.client.post("/spellcast/batch/j1/cancel").status_code == 200      # after the end: still no error
    assert rig.post([a], "j2").status_code == 202 and rig.ended("j2")["state"] == "done"


def test_a_mesh_that_is_being_saved_is_saved_before_a_cancel_takes_hold(threaded: Rig) -> None:
    rig, gate = threaded, Gate()
    rig.engine.at[("finish", "p_a")] = gate
    a, b = rig.prop("p_a"), rig.prop("p_b")
    rig.post([a, b])
    gate.wait()
    rig.client.post("/spellcast/batch/j1/cancel")
    gate.open()
    told = rig.ended()
    assert told["state"] == "cancelled"
    assert told["items"]["p_a"]["state"] == "done" and Path(a["out_path"]).read_bytes() == b"glTF"
    assert told["items"]["p_b"]["state"] == "cancelled" and told["items"]["p_b"]["fraction"] == 0.72
    assert not Path(b["out_path"]).exists()


def test_a_cancel_that_comes_before_the_engine_starts_costs_nothing(threaded: Rig) -> None:
    rig, held = threaded, threading.Event()
    rig.reference.loaded = True                              # the reference path's models are in memory
    rig.px.pipeline = object()
    rig.gpu.submit(lambda: held.wait(10))                    # the GPU thread is busy with something else
    rig.post([rig.prop("p_a")])
    assert rig.told()["items"]["p_a"]["state"] == "waiting"
    rig.client.post("/spellcast/batch/j1/cancel")
    held.set()
    told = rig.ended()
    assert told["state"] == "cancelled" and told["items"]["p_a"]["state"] == "cancelled"
    assert not rig.engine.runs and rig.px.pipeline is not None           # nothing was unloaded for it


def test_a_prop_that_fails_does_not_stop_the_others(rig: Rig) -> None:
    rig.engine.script = {"p_b": {"fail": ("hr", "oom", "CUDA out of memory. Tried to allocate 2.00 GiB")},
                         "p_c": {"fail": ("cutout", "input", "No prop was found in the Front slice: draw the "
                                                               "slice around the prop.")},
                         "p_d": {"fail": ("finish", "error", "RuntimeError: xatlas gave up")},
                         "p_e": {"fail": ("upsample", "disk", "Not enough free disk space in the project folder "
                                                                "for Pixal3D's temporary files.")},
                         "p_f": {"fail": ("lr", "ram", "DefaultCPUAllocator: not enough memory")}}
    items = [rig.prop(f"p_{letter}") for letter in "abcdef"]
    rig.post(items)
    told = rig.told()
    assert told["state"] == "done" and told["error"] is None
    assert told["items"]["p_a"]["state"] == "done"
    assert {prop_id: item["error"] for prop_id, item in told["items"].items() if item["state"] == "failed"} == {
        "p_b": {"detail": server.OUT_OF_GPU_MEMORY, "kind": "oom"},
        "p_c": {"detail": "No prop was found in the Front slice: draw the slice around the prop.", "kind": "input"},
        "p_d": {"detail": "Pixal3D failed while making the mesh (RuntimeError: xatlas gave up). Try again; if it "
                          "keeps failing, restart Pixal3D and check its log.", "kind": "error"},
        "p_e": {"detail": "Not enough free disk space in the project folder for Pixal3D's temporary files.",
                "kind": "disk"},
        "p_f": {"detail": server.OUT_OF_RAM, "kind": "ram"}}
    assert told["items"]["p_b"]["fraction"] == 0.32 and told["items"]["p_b"]["part"] == "hr"   # where it stood
    assert "page file" in server.OUT_OF_RAM and "25" not in server.OUT_OF_RAM
    assert Path(items[0]["out_path"]).is_file() and not Path(items[1]["out_path"]).exists()


def test_a_prop_that_needed_the_memory_limit_lifted_says_so(rig: Rig, capsys: pytest.CaptureFixture[str]) -> None:
    rig.engine.script = {"p_b": {"spill": ("upsample", "finish"), "resolution": 1408}}
    rig.post([rig.prop("p_a"), rig.prop("p_b", resolution=1536)])
    results = {prop_id: item["result"] for prop_id, item in rig.told()["items"].items()}
    assert results["p_a"]["slow"] == [] and results["p_b"]["slow"] == ["upsample", "finish"]
    assert results["p_b"]["resolution"] == 1408 and results["p_a"]["resolution"] == 1024   # the resolution used
    assert "p_b: the part 'upsample' does not fit into the free graphics memory" in capsys.readouterr().out


def test_a_batch_whose_card_broke_asks_for_a_restart(rig: Rig) -> None:
    rig.engine.script = {"p_b": {"raise": ("lr", Staged.BatchBroken("CUDA error: unspecified launch failure"))}}
    a, b = rig.prop("p_a"), rig.prop("p_b")
    rig.post([a, b])
    told = rig.told()
    assert told["state"] == "failed" and told["part"] is None
    assert told["error"] == {"detail": "Pixal3D lost the graphics card and has to start again (CUDA error: "
                                       "unspecified launch failure).", "kind": "restart"}
    # The prop that was on the card is named, and no prop is failed: the hub sends the unfinished ones again.
    assert told["current"] == "p_b"
    assert {prop_id: item["state"] for prop_id, item in told["items"].items()} == {"p_a": "resting", "p_b": "working"}
    assert rig.engine.released == 0                         # the card cannot be used: nothing is tidied on it
    assert rig.post([a], "j2").status_code == 202           # the record does not block the launcher


def test_any_other_failure_of_the_engine_fails_the_props_that_were_not_made(rig: Rig) -> None:
    rig.engine.script = {"p_b": {"raise": ("hr", RuntimeError("the store lost a file"))}}
    items = [rig.prop("p_a"), rig.prop("p_b"), rig.prop("p_c")]
    rig.post(items, wave_size=1)                            # one prop at a time: p_a is made before p_b starts
    told = rig.told()
    words = ("Pixal3D failed while making the mesh (RuntimeError: the store lost a file). Try again; if it keeps "
             "failing, restart Pixal3D and check its log.")
    assert told["state"] == "failed" and told["error"] == {"detail": words, "kind": "error"}
    assert told["current"] is None and rig.engine.released == 1
    assert told["items"]["p_a"]["state"] == "done" and Path(items[0]["out_path"]).is_file()
    for prop_id in ("p_b", "p_c"):
        assert told["items"][prop_id]["state"] == "failed"
        assert told["items"][prop_id]["error"] == {"detail": words, "kind": "error"}

    rig.engine.script = {"p_a": {"raise": ("features", MemoryError())}}
    rig.post([items[0]], "j2")
    told = rig.told("j2")
    assert told["state"] == "failed" and told["error"] == {"detail": server.OUT_OF_RAM, "kind": "ram"}
    assert told["items"]["p_a"]["error"] == told["error"]

    rig.engine.script = {"p_a": {"raise": ("hr", RuntimeError("CUDA out of memory. Tried to allocate 3.00 GiB"))}}
    rig.post([items[0]], "j3")                              # the batch knows "restart", "ram" and "error" only
    told = rig.told("j3")
    assert told["error"] == {"detail": server.OUT_OF_GPU_MEMORY, "kind": "error"}
    assert told["items"]["p_a"]["error"] == {"detail": server.OUT_OF_GPU_MEMORY, "kind": "oom"}


def test_a_batch_never_stays_running_when_its_work_is_torn_down(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing may leave a batch running: Pixal3D would then be called busy for the life of the process and
    the hub would restart it at every later job. The engine re-raises KeyboardInterrupt by name, and the GPU
    thread can refuse the work."""
    rig.engine.script = {"p_a": {"raise": ("ss", KeyboardInterrupt())}}
    rig.post([rig.prop("p_a")])
    told = rig.told()
    assert told["state"] == "failed" and told["error"]["kind"] == "error"
    assert told["items"]["p_a"]["error"] == {"detail": server.BATCH_STOPPED, "kind": "stopped"}

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the GPU thread is gone")

    monkeypatch.setattr(rig.gpu, "submit", refuse)
    r = rig.post([rig.prop("p_b")], "j2")
    assert r.status_code == 500 and "the GPU thread is gone" in r.json()["detail"]
    assert rig.told("j2")["state"] == "failed" and rig.runner.batches.running() is None
    monkeypatch.undo()
    assert rig.post([rig.prop("p_c")], "j3").status_code == 202      # never 409: no batch is left running


def test_props_of_one_batch_keep_their_own_guard_notes(rig: Rig, capsys: pytest.CaptureFixture[str]) -> None:
    """One RunNotes object serves the guard: it is cleared and read around each prop's turn in "ss"."""
    block = column(5, 5, range(10, 32))
    worse = column(5, 5, range(0, 32))
    half = column(5, 5, range(10, 32)) + column(6, 6, (12,))
    rig.engine.script = {"p_a": {"draws": [block, GOOD_DRAW]},                 # drawn again once
                         "p_c": {"draws": [worse, half, block]}}               # every draw a block: the best is kept
    rig.post([rig.prop("p_a"), rig.prop("p_b"), rig.prop("p_c"), rig.prop("p_d")])
    results = {prop_id: item["result"] for prop_id, item in rig.told()["items"].items()}
    assert [results[i]["ss_retries"] for i in ("p_a", "p_b", "p_c", "p_d")] == [1, 0, 2, 0]
    assert [results[i]["warning"] for i in ("p_a", "p_b", "p_c", "p_d")] == [None, None, guard.STREAK_WARNING, None]
    assert rig.engine.draws == []                                              # every scripted draw was taken
    assert capsys.readouterr().out.count("drawing it again") == 3


def test_a_batch_sent_again_after_a_restart_only_finishes_what_was_sampled(rig: Rig) -> None:
    rig.engine.checkpoints = {"p_b"}
    seen: list[dict[str, Any]] = []
    rig.engine.watch = lambda: seen.append(rig.runner.batches.get("j1").snapshot())
    rig.post([rig.prop("p_a"), rig.prop("p_b")], resume=True, wave_size=8)
    run = rig.engine.runs[0]
    assert run["resume"] is True and run["wave_size"] == 8
    first = seen[0]["items"]["p_b"]                          # it starts where its sampling had ended
    assert first == {"state": "resting", "part": "tex", "parts_done": 10, "fraction": 0.72,
                     "message": "Waiting for the next step", "step": 0, "total": 0}
    assert {t["items"]["p_b"]["part"] for t in seen} == {"tex", "finish"}
    results = {prop_id: item["result"] for prop_id, item in rig.told()["items"].items()}
    assert results["p_b"]["resumed"] is True and results["p_a"]["resumed"] is False


def test_the_fingerprint_of_a_request_changes_with_what_decides_the_latents(rig: Rig) -> None:
    sampler = {"ss": {"steps": 12}, "shape": {"steps": 12}, "tex": {"steps": 12}}

    def of(item: dict[str, Any], seed: int = 7, samplers: dict[str, Any] = sampler, tokens: int = 49152) -> str:
        payload = server.BatchItem.model_validate(item)
        return server.fingerprint(payload, seed, len(payload.views or []) >= 2, samplers, tokens)

    plain = rig.prop("p_a")
    base = of(plain)
    assert base == of(dict(plain)) and len(base) == 64
    assert base == of({**plain, "decimation_target": 5000, "texture_size": 512, "seed": 99,
                       "out_path": plain["out_path"].replace("1a2b", "ffff")})          # the clean-up's alone
    assert base != of(plain, seed=8) and base != of({**plain, "resolution": 1536}) and base != of(plain, tokens=1000)
    assert base != of(plain, samplers={**sampler, "tex": {"steps": 13}})
    Path(plain["image_path"]).write_bytes(b"another picture under the same name")
    assert base != of(plain)
    views = rig.prop("p_b", views=(0, 90))
    both = of(views)
    assert both != of({**views, "views": views["views"][:1]})
    assert both == of({**views, "views": views["views"][::-1]})                         # the front goes first anyway
    cut = [dict(views["views"][0], rect=[0, 0, 10, 10]), views["views"][1]]
    turned = [views["views"][0], dict(views["views"][1], azimuth=270)]
    assert len({both, of({**views, "views": cut}), of({**views, "views": turned})}) == 3
    assert batch.request_hash(a=1, b=[1.5, None]) == batch.request_hash(b=[1.5, None], a=1)


def test_the_engine_is_made_again_when_the_multi_view_models_arrive(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    rig = Rig(tmp_path, monkeypatch, multiview=False)
    first = rig.engine
    several = rig.prop("p_views", views=(0, 90))
    rig.post([several, rig.prop("p_a")])
    assert rig.told()["items"]["p_views"]["error"]["kind"] == "mv_missing" and rig.engine is first
    fake_snapshot(rig.models)                                # downloaded while Pixal3D runs
    rig.post([rig.prop("p_b")], "j2")
    assert rig.engine is first                               # nothing that needs them: nothing changes
    rig.post([several], "j3")
    assert rig.engine is not first and first.closed and rig.engine.multiview
    assert "sample_sparse_structure" in vars(rig.engine)     # the new one is guarded too
    assert rig.told("j3")["items"]["p_views"]["result"]["mode"] == "multi"


def test_the_engine_is_made_again_when_hi3ds_networks_arrive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An engine made before Hi3D's networks were installed (hi3d_scheduler None) finds them only when it is made
    again: a batch with a prop set to Hi3D makes it again, as several views do for the multi-view models."""
    rig = Rig(tmp_path, monkeypatch)
    first = rig.engine
    first.hi3d_scheduler = None                              # made without them
    rig.post([rig.prop("p_a")])
    assert rig.ended()["state"] == "done" and rig.engine is first      # a prop of Pixal3D's own: nothing changes
    rig.post([rig.prop("p_h", geometry="hi3d")], "j2")
    assert rig.ended("j2")["state"] == "done"
    assert rig.engine is not first and first.closed
    assert [p.geometry for p in rig.engine.runs[-1]["props"]] == ["hi3d"]
    assert "sample_sparse_structure" in vars(rig.engine)     # the new one is guarded too


def test_a_batch_says_whether_a_prop_may_spill_and_the_engine_is_made_with_that(rig: Rig) -> None:
    """The hub's pixal3d.allow_spill comes with every batch (off unless the artist allowed it) and goes into the
    engine's options, which are read when the engine is made: a batch that asks otherwise makes it again (no
    weights are read)."""
    first = rig.engine
    assert first.options.spill is False and rig.runner.spill is False
    assert server.BatchRequest.model_validate(rig.body([rig.prop("p_a")])).spill is False
    rig.post([rig.prop("p_a")])
    assert rig.engine is first
    rig.post([rig.prop("p_b")], "j2", spill=True)
    assert rig.told("j2")["items"]["p_b"]["state"] == "done"
    second = rig.engine
    assert second is not first and first.closed and second.options.spill is True and rig.runner.spill is True
    assert "sample_sparse_structure" in vars(second)                    # the new one is guarded too
    rig.post([rig.prop("p_c")], "j3", spill=True)
    assert rig.engine is second
    rig.post([rig.prop("p_d")], "j4")
    assert rig.engine is not second and rig.engine.options.spill is False and rig.runner.spill is False
    assert rig.told("j4")["items"]["p_d"]["state"] == "done"

    # Until the fork is pinned at an engine with the field, the launcher passes it only when it is there.
    @dataclasses.dataclass
    class WithSpill:
        pipeline_dir: str
        spill: bool = False

    @dataclasses.dataclass
    class Before:
        pipeline_dir: str

    assert server.spill_option(WithSpill, True) == {"spill": True} and server.spill_option(WithSpill, False) == {
        "spill": False}
    assert server.spill_option(Before, True) == {}


def test_a_prop_deleted_while_its_batch_runs_fails_at_its_next_hook_and_its_folder_is_not_made_again(
        rig: Rig) -> None:
    """The engine cannot be told to drop one prop; the hooks look at the prop's folder (prepare, camera, the sparse
    structure's prop_scope, finish) and fail a deleted prop there, alone, with the kind "input". Nothing is written
    for it and its folder is not made again."""
    a, b, c = rig.prop("p_a"), rig.prop("p_b"), rig.prop("p_c")
    folder = lambda item: Path(item["out_path"]).parents[1]                         # noqa: E731
    rig.engine.at[("cutout", "p_b")] = lambda: shutil.rmtree(folder(b))   # failed at its sparse structure
    rig.engine.at[("hr", "p_c")] = lambda: shutil.rmtree(folder(c))       # failed before its mesh is built
    rig.post([a, b, c])
    told = rig.told()
    assert told["state"] == "done" and told["items"]["p_a"]["state"] == "done"
    for prop_id, part in (("p_b", "ss"), ("p_c", "finish")):
        item = told["items"][prop_id]
        assert (item["state"], item["part"], item["error"]) == ("failed", part, {"detail": mv.PROP_DELETED,
                                                                                 "kind": "input"}), prop_id
    assert not folder(b).exists() and not folder(c).exists() and Path(a["out_path"]).is_file()

    # Gone before its batch began: it starts as failed, and nothing is made for it.
    gone = rig.root / "P" / "props" / "p_gone"
    d = rig.prop("p_d", out_path=str(gone / "mesh" / "mesh.1a2b3c4d.partial.glb"))
    rig.post([d, rig.prop("p_e")], "j2")
    assert rig.told("j2")["items"]["p_d"]["error"] == {"detail": mv.PROP_DELETED, "kind": "input"}
    assert rig.told("j2")["items"]["p_e"]["state"] == "done" and not gone.exists()


def test_a_batch_lines_its_props_up_for_opening_ahead_and_lets_go_after(rig: Rig, monkeypatch: pytest.MonkeyPatch,
                                                                        capsys: pytest.CaptureFixture[str]) -> None:
    """BatchRunner hands the hooks the batch's props and the engine's events: when the cut-out starts, the first two
    props in its order (the multi-view ones first) are being opened; when the batch is over nothing waits. Opening
    ahead that breaks costs no prop: it is noted and the batch is made."""
    opening = Opening(monkeypatch)
    ahead = rig.runner.hooks.ahead
    seen: list[list[str]] = []
    rig.engine.at[("cutout", "p_a")] = lambda: seen.append(sorted(ahead._opened))
    rig.post([rig.prop("p_a"), rig.prop("p_b"), rig.prop("p_c", views=(0, 90))])
    assert rig.told()["state"] == "done" and seen == [["p_a", "p_c"]]
    assert not ahead._opened and not ahead._props and not ahead._line

    def broken(req: Any, multi: bool) -> list[Any]:
        raise RuntimeError("opening ahead broke")

    monkeypatch.setattr(server, "picture_openers", broken)
    rig.post([rig.prop("p_d"), rig.prop("p_e")], "j2")
    told = rig.told("j2")
    assert told["state"] == "done" and {item["state"] for item in told["items"].values()} == {"done"}
    assert "RuntimeError: opening ahead broke" in capsys.readouterr().err


def test_the_reference_models_and_the_engine_take_turns(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pixal3D's own pipeline keeps every model in RAM: it is loaded when first used and dropped before a batch,
    with the DINOv3 share_dinov3 holds and the multi-view pipeline built from it."""
    px, reference = rig.px, rig.reference
    reference.ensure_single_view()                           # Pixal3D's own page is used
    assert px.loads == 1 and reference.loaded and rig.engine.released == 1 and rig.modes.mode == "sv"
    rig.shared.loaded["dinov3"] = object()
    rig.modes.mode, rig.modes.mv_pipeline = "mv", object()   # a multi-view mesh was made on the reference path
    rig.post([rig.prop("p_a")])
    assert rig.told()["items"]["p_a"]["state"] == "done"
    assert px.pipeline is None and px.moge_model is None and px.envmap is None and not reference.loaded
    assert rig.shared.loaded == {} and rig.modes.mode == "sv" and rig.modes.mv_pipeline is None
    rig.post([rig.prop("p_b")], "j2")                        # nothing is loaded: nothing to drop
    reference.ensure()
    assert px.loads == 2 and rig.engine.released == 2 and len(rig.prepared) == 2
    reference.ensure()
    assert px.loads == 2 and rig.engine.released == 3        # loaded already; the engine still lets go first

    px.pipeline = None                                       # a load that failed after MoGe: the rest goes too
    reference.loaded = False
    reference.ensure()
    assert px.loads == 3 and reference.loaded


def test_share_dinov3_lets_go_of_the_model_it_shares(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[str] = []

    class DINOv3ViTModel:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> object:
            loads.append(path)
            return object()

    icp = types.ModuleType("image_conditioned_proj")
    icp.DINOv3ViTModel = DINOv3ViTModel                      # type: ignore[attr-defined]
    mixins = types.ModuleType("pixal3d.trainers.flow_matching.mixins")
    mixins.image_conditioned_proj = icp                      # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pixal3d.trainers.flow_matching.mixins", mixins)
    shared = server.share_dinov3()
    one, two = icp.DINOv3ViTModel.from_pretrained("dinov3"), icp.DINOv3ViTModel.from_pretrained("dinov3")
    assert one is two and loads == ["dinov3"] and list(shared.loaded.values()) == [one]
    shared.clear()
    assert shared.loaded == {} and icp.DINOv3ViTModel.from_pretrained("dinov3") is not one and len(loads) == 2


def test_the_gpu_thread_takes_work_without_waiting_for_it() -> None:
    gpu = server.GpuWorker()
    try:
        held, order = threading.Event(), []

        def first() -> str:
            assert held.wait(10)
            order.append(threading.current_thread().name)
            return "first"

        asked = time.monotonic()
        future = gpu.submit(first)                           # returns while the work waits
        assert not future.done() and time.monotonic() - asked < 1
        second = gpu.submit(lambda: order.append("second") or threading.current_thread().name)
        held.set()
        assert future.result(10) == "first" and second.result(10).startswith("pixal3d-gpu")
        assert order == [second.result(), "second"]          # one thread, in arrival order
        assert gpu.run(lambda a, b=0: a + b, 2, b=3) == 5
        with pytest.raises(ZeroDivisionError):
            gpu.run(lambda: 1 // 0)
    finally:
        gpu._pool.shutdown(wait=True)


def test_failures_are_told_apart_by_what_ran_out() -> None:
    class OutOfMemoryError(RuntimeError):                    # torch's, by its name
        pass

    kinds = {
        "oom": [OutOfMemoryError("Allocation on device"),
                RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB"),
                RuntimeError("Cuda error: 2[cudaMalloc(&m_gpuPtr, bytes);]"),              # nvdiffrast 0.4.0
                RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"),
                RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`")],
        "ram": [MemoryError(), OSError("[WinError 1455] The paging file is too small for this operation to complete"),
                RuntimeError("[enforce fail at alloc_cpu.cpp:116] data. DefaultCPUAllocator: not enough memory: "
                             "you tried to allocate 5368709120 bytes.")],
        "missing_file": [FileNotFoundError(2, "No such file or directory", "ckpts/x.safetensors")],
        "input": [mv.ViewProblem("The Back slice lies outside its picture: slice it again."),
                  Staged.PropFailed("No prop was found in the Left slice.", "input")],
        "mv_missing": [mv.MultiviewMissing(mv.MV_MISSING)],
        "disk": [Staged.PropFailed("Not enough free disk space in the project folder.", "disk")],
        "restart": [Staged.BatchBroken("CUDA error: device-side assert triggered")],
        "error": [ValueError("zero-size array to reduction operation minimum which has no identity"),
                  RuntimeError("Cuda error: 709[cudaGraphicsMapResources(1, &res, stream);]")]}
    for kind, failures in kinds.items():
        for failure in failures:
            assert batch.classify(failure) == kind, failure
    # The engine names the same failures the same way; otherwise one cause gives two messages in one job.
    runner = paths.PIXAL3D_DIR / "pixal3d" / "staged" / "runner.py"
    assert (batch.OOM_TEXTS, batch.RAM_TEXTS) == (forks_tuple(runner, "OOM_TEXTS"), forks_tuple(runner, "RAM_TEXTS"))
    engine = forks_function(runner, "classify", {
        "PropFailed": Staged.PropFailed, "torch": types.SimpleNamespace(OutOfMemoryError=OutOfMemoryError),
        "OOM_TEXTS": batch.OOM_TEXTS, "RAM_TEXTS": batch.RAM_TEXTS})
    for kind, failures in kinds.items():
        for failure in failures:                   # bar the three the launcher's own hooks raise
            if kind not in ("mv_missing", "restart") and not isinstance(failure, mv.ViewProblem):
                assert engine(failure) == kind, failure
    answer = lambda exc: (server.failure_response(exc).status_code,                        # noqa: E731
                          json.loads(bytes(server.failure_response(exc).body)))
    assert answer(kinds["oom"][2]) == (500, {"detail": server.OUT_OF_GPU_MEMORY, "kind": "oom"})
    assert answer(kinds["ram"][1]) == (500, {"detail": server.OUT_OF_RAM, "kind": "ram"})
    words = answer(kinds["missing_file"][0])[1]["detail"]
    assert words.startswith("A file Pixal3D needs is missing: ckpts/x.safetensors.")
    words = answer(kinds["error"][0])[1]["detail"]
    assert words.startswith("Pixal3D failed while making the mesh (ValueError: zero-size")
    for exc in (MemoryError(), RuntimeError("CUDA out of memory"), OSError("The paging file is too small")):
        hint = server.startup_hint(exc, Path("C:/spellcast3d/code/pixal3d"))
        assert "25" not in hint and "30 GB" not in hint and "low-VRAM" not in hint and "start Pixal3D again" in hint
    assert "page file" in server.startup_hint(MemoryError(), Path("C:/spellcast3d/code/pixal3d"))


def test_a_work_folder_must_be_one_the_hub_made_for_a_job(tmp_path: Path) -> None:
    root = (tmp_path / "projects").resolve()
    assert server.work_dir_problem(str(root / "P" / "temp" / "mesh-j1"), root) is None
    assert server.work_dir_problem(str(root / "P" / "Temp" / "MESH-j1"), root) is None      # Windows' own spelling
    assert server.work_dir_problem(str(root / "P" / "temp" / "mesh-j1" / ".." / "mesh-j2"), root) is None
    assert server.work_dir_problem(str(tmp_path / "anywhere" / "temp" / "mesh-x")) is None  # no projects folder given
    assert server.work_dir_problem(str(tmp_path / "temp" / "mesh-j1"), root) == server.OUTSIDE_PROJECTS
    for wrong in (root / "P" / "temp" / "mesh-j1" / ".." / ".." / "props", root / "P" / "temp" / "meshes",
                  root / "P", root / "temp" / "mesh-j1", root / "P" / "deleted" / "temp" / "mesh-j1"):
        assert "not the work folder of a job" in (server.work_dir_problem(str(wrong), root) or ""), wrong
    assert "not the work folder of a job" in (server.work_dir_problem(str(tmp_path / "temp" / "bake-1")) or "")
    (root / "P" / "images").mkdir(parents=True)
    (root / "P" / "images" / "0001.png").write_bytes(b"png")
    req = request(root / "P")
    assert server.validate(req, root, str(root / "P" / "temp" / "mesh-j1")) is None
    assert server.validate(req, root, str(tmp_path / "temp" / "mesh-j1")) == server.OUTSIDE_PROJECTS
    assert "not the work folder" in (server.validate(req, root, str(root / "P" / "mesh")) or "")


# ---- batches: what the hooks do (numpy, PIL, OpenCV, torch: pixal3d venv, on the CPU) -------------------------

def test_fan_centres_are_found_among_the_fans_alone() -> None:
    """recentre_fans looks at the faces of the fans only, not at every face of the mesh as 64-bit indices: the
    same vertices to the bit as the plain form over all faces, which this test keeps."""
    torch = pytest.importorskip("torch")

    def over_all_faces(vertices: Any, faces: Any, before: int) -> int:
        added = int(vertices.shape[0]) - before
        faces = faces.long()
        total = torch.zeros((added, 3), dtype=torch.float64)
        count = torch.zeros(added, dtype=torch.float64)
        for corner in range(3):
            fan = faces[:, corner] >= before
            centre = faces[fan, corner] - before
            for other in range(3):
                if other != corner:
                    rim = faces[fan, other]
                    old = rim < before
                    total.index_add_(0, centre[old], vertices[rim[old]].double())
                    count.index_add_(0, centre[old], torch.ones(int(old.sum()), dtype=torch.float64))
        placed = count > 0
        mean = total[placed] / count[placed, None]
        moved = vertices[before:]
        wrong = int(((moved[placed].double() - mean).abs() > 1e-4).any(dim=1).sum())
        moved[placed] = mean.to(vertices.dtype)
        return wrong

    generator = torch.Generator().manual_seed(5)
    before, holes = 4000, 300
    old = torch.rand(before, 3, generator=generator)
    body = torch.randint(0, before, (20_000, 3), generator=generator, dtype=torch.int32)
    fans, centres = [], []
    for hole in range(holes):
        rim = torch.randint(0, before, (int(torch.randint(3, 9, (1,), generator=generator)),), generator=generator)
        centre = before + hole
        for n in range(len(rim)):                            # the centre sits at any corner of its fan's faces
            face = [int(rim[n]), int(rim[(n + 1) % len(rim)]), centre]
            fans.append(face[-(hole % 3):] + face[:-(hole % 3)])
        exact = old[rim.unique()].double().mean(dim=0) if hole % 4 else None    # cumesh's own, where it is right
        centres.append(exact.float() if exact is not None else torch.tensor([0.5, 0.5, 1e20]))
    stray = torch.tensor([[7.0, 7.0, 7.0]])                  # a new vertex no face uses stays where it is
    vertices = torch.cat([old, torch.stack(centres), stray])
    faces = torch.cat([body, torch.tensor(fans, dtype=torch.int32)])[torch.randperm(20_000 + len(fans),
                                                                                    generator=generator)]
    ours, plain = vertices.clone(), vertices.clone()
    assert server.recentre_fans(ours, faces, before) == over_all_faces(plain, faces, before) > holes // 5
    assert torch.equal(ours, plain) and torch.equal(ours[:before], old) and ours[-1].tolist() == [7.0, 7.0, 7.0]
    assert float(ours[before:-1].abs().max()) <= 1.0         # every centre is back among the rims
    assert faces.dtype == torch.int32                        # the caller's faces are not touched
    assert server.recentre_fans(vertices[:before].clone(), body, before) == 0


def forks_function(file: Path, name: str, namespace: dict[str, Any], of_class: str | None = None) -> Callable[..., Any]:
    """A function of the fork (a method of `of_class`), compiled from its source alone: importing the package
    needs the GPU libraries. `namespace`: the names it uses."""
    if not file.is_file():
        pytest.skip(f"the Pixal3D checkout is not here ({file})")
    body = ast.parse(file.read_text(encoding="utf-8")).body
    if of_class is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == of_class).body
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(file), "exec"), namespace)
    return namespace[name]


def forks_tuple(file: Path, name: str) -> tuple[Any, ...]:
    """A tuple the fork keeps at module level, read from its source (as forks_function reads a function)."""
    if not file.is_file():
        pytest.skip(f"the Pixal3D checkout is not here ({file})")
    for node in ast.parse(file.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"{file} has no {name}")


def hooks_px(tmp_path: Path, **more: Any) -> Any:
    told: list[tuple[str, int, int]] = []
    return types.SimpleNamespace(TMP_DIR=str(tmp_path), told=told,
                                 _update_progress=lambda stage, step, total: told.append((stage, step, total)), **more)


def test_a_batchs_picture_is_cut_out_as_pixal3ds_own_method_does_with_the_engines_model(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """hooks.prepare for one picture of up to 1024 px: the launcher's cut-out gives what
    Pixal3DImageTo3DPipeline.preprocess_image itself (read from the fork) gives, to the byte, with the stand-in
    that carries only the engine's background removal, and moves no model."""
    image_mod = pytest.importorskip("PIL.Image")
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(server, "staged_module", lambda: Staged)
    preprocess_image = forks_function(paths.PIXAL3D_DIR / "pixal3d" / "pipelines" / "pixal3d_image_to_3d.py",
                                      "preprocess_image", {"np": np, "Image": image_mod},
                                      of_class="Pixal3DImageTo3DPipeline")
    px = hooks_px(tmp_path)
    matted: list[Any] = []

    def matte(rgb: Any) -> Any:                              # the engine's: the picture with its alpha set
        matted.append(rgb.copy())
        alpha = (np.asarray(rgb)[:, :, 0] > 128).astype(np.uint8) * 255
        rgb.putalpha(image_mod.fromarray(alpha))
        return rgb

    (tmp_path / "images").mkdir()
    picture = image_mod.new("RGB", (600, 400), (20, 20, 20))
    picture.paste((200, 60, 30), (100, 100, 300, 300))       # the prop: 200 x 200 px, 341 px once enlarged
    picture.save(tmp_path / "images" / "0001.png")
    hooks = server.BatchHooks(px, server.RunNotes())
    req = Staged.PropRequest("p_a", 7, 1024, False, {}, payload=server.BatchItem(
        id="p_a", image_path=str(tmp_path / "images" / "0001.png"), out_path=str(tmp_path / "mesh.glb"),
        views_dir=str(tmp_path / "views")))
    prepared = hooks.prepare(req, matte)
    assert len(matted) == 1 and matted[0].mode == "RGB" and matted[0].size == (1024, 683)    # at the input size
    assert prepared.meta == {} and len(prepared.images) == 1
    image = prepared.images[0]
    assert image.mode == "RGB" and image.size[0] == image.size[1] == int(341 * 1.1) // 2 * 2
    assert image.getpixel((image.width // 2, image.height // 2)) == (200, 60, 30) and image.getpixel((2, 2)) == (0, 0, 0)
    assert px.told == [(server.REMOVING_BACKGROUND, 0, 1)]
    mv.wait_for_views()
    assert [p.name for p in (tmp_path / "views").iterdir()] == ["0_front.png"]
    theirs = preprocess_image(server.matte_shim(matte), server.open_picture(req.payload, None))
    assert theirs.size == image.size and theirs.tobytes() == image.tobytes()

    side = server.ViewSpec(image_path=req.payload.image_path, rect=[900, 0, 50, 50], azimuth=180)
    off = Staged.PropRequest("p_b", 7, 1024, False, {}, payload=req.payload.model_copy(update={"views": [side]}))
    with pytest.raises(Staged.PropFailed) as failed:         # a slice the artist has to fix: this prop alone
        hooks.prepare(off, matte)
    assert failed.value.kind == "input" and "Back slice lies outside its picture" in failed.value.detail


def cut_out_scene(image_mod: Any, np: Any, size: tuple[int, int], box: tuple[float, float, float, float],
                  alpha: bool = False) -> Any:
    """A prop with fine detail (stripes 3 px wide and a soft rim) at `box` (fractions of the picture) on a dark
    background; with `alpha`, an RGBA cut-out of it (transparent around the prop, a soft edge)."""
    width, height = size
    x0, y0, x1, y1 = int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)
    pixels = np.full((height, width, 4), (20, 20, 20, 0), dtype=np.uint8)
    stripes = (np.arange(x1 - x0) // 3 % 2 * 90 + 120).astype(np.uint8)
    pixels[y0:y1, x0:x1, 0] = 210
    pixels[y0:y1, x0:x1, 1] = stripes[None, :]
    pixels[y0:y1, x0:x1, 2] = np.linspace(30, 220, y1 - y0, dtype=np.uint8)[:, None]
    pixels[y0:y1, x0:x1, 3] = 255
    pixels[y0:y1, x0:x0 + 3, 3] = 140                                     # a soft left rim
    return image_mod.fromarray(pixels if alpha else np.ascontiguousarray(pixels[:, :, :3]))


def test_the_single_view_cut_out_is_pixal3ds_up_to_1024_px_and_crops_before_it_reduces_above(
        capsys: pytest.CaptureFixture[str]) -> None:
    """pixal3d_mv.cut_out_single against Pixal3DImageTo3DPipeline.preprocess_image itself (read from the fork) on
    made-up pictures, with a stand-in background removal that, like BiRefNet, makes its mask at 1024 x 1024 and
    brings it to the picture's size. Up to 1024 px: the same bytes. Larger: the prop is cropped at full size and
    only its square is reduced to 1024 px (preprocess_image reduces the whole picture first, and the prop is then
    enlarged again by the stages) - and in Pixal3D's own order (crop_first False) the same bytes again."""
    image_mod = pytest.importorskip("PIL.Image")
    np = pytest.importorskip("numpy")
    preprocess_image = forks_function(paths.PIXAL3D_DIR / "pixal3d" / "pipelines" / "pixal3d_image_to_3d.py",
                                      "preprocess_image", {"np": np, "Image": image_mod},
                                      of_class="Pixal3DImageTo3DPipeline")
    matted: list[tuple[int, int]] = []

    def matte(rgb: Any) -> Any:
        matted.append(rgb.size)
        small = np.asarray(rgb.resize((1024, 1024), image_mod.Resampling.BILINEAR))
        mask = image_mod.fromarray(((small[:, :, 0] > 128) * 255).astype(np.uint8)).resize(rgb.size)
        rgb.putalpha(mask)
        return rgb

    shim = server.matte_shim(matte)

    def same(a: Any, b: Any) -> bool:
        return a.mode == b.mode and a.size == b.size and a.tobytes() == b.tobytes()

    box = (0.30, 0.20, 0.75, 0.90)
    for size in ((600, 400), (1024, 1024), (700, 1024)):                # Pixal3D's own result, to the byte
        for alpha in (False, True):
            picture = cut_out_scene(image_mod, np, size, box, alpha)
            matted.clear()
            ours = mv.cut_out_single(shim, picture.copy())
            assert same(ours, preprocess_image(shim, picture.copy())), (size, alpha)
            assert ours.mode == "RGB" and matted == ([] if alpha else [size, size])    # the alpha of its own: kept
    assert "brought to" not in capsys.readouterr().out

    for alpha in (False, True):
        picture = cut_out_scene(image_mod, np, (2000, 1500), box, alpha)
        old = preprocess_image(shim, picture.copy())
        assert same(mv.cut_out_single(shim, picture.copy(), crop_first=False), old)    # Pixal3D's own order
        matted.clear()
        ours = mv.cut_out_single(shim, picture.copy())
        assert matted == ([] if alpha else [(2000, 1500)])                  # cut out at the picture's own size
        # What it is: the picture's square around the prop, cropped at full size, reduced to 1024 px, on black.
        cut = picture.copy() if alpha else matte(picture.convert("RGB"))
        ys, xs = np.nonzero(np.asarray(cut.getchannel("A")) > 0.8 * 255)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()                # preprocess_image's own box
        side = int(max(x1 - x0, y1 - y0) * 1.1)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        square = cut.crop((cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2))
        reduced = np.asarray(square.resize((1024, 1024), image_mod.Resampling.LANCZOS)).astype(np.float32) / 255
        expected = image_mod.fromarray((np.clip(reduced[:, :, :3] * reduced[:, :, 3:4], 0, 1) * 255).astype(np.uint8))
        assert ours.size == (1024, 1024) and same(ours, expected), alpha
        assert max(old.size) < 1024 * 0.6        # the same prop, as Pixal3D crops it: well under the stages' 1024 px
    assert capsys.readouterr().out.count("Cut out at 2000x1500 px; the prop's square of 115") == 2

    # A small prop in a large picture: its square is smaller than 1024 px and stays at full size (the stages
    # enlarge it, as they do any square).
    picture = cut_out_scene(image_mod, np, (2000, 1500), (0.40, 0.40, 0.60, 0.60))
    ours = mv.cut_out_single(shim, picture.copy())
    assert ours.size[0] == ours.size[1] and 300 < ours.width < 1024
    assert ours.width > 1.8 * preprocess_image(shim, picture.copy()).width          # about twice the pixels across

    empty = image_mod.new("RGB", (1500, 1200), (20, 20, 20))                        # nothing to cut out
    with pytest.raises(mv.ViewProblem, match="No prop was found"):
        mv.cut_out_single(shim, empty)


def test_a_batchs_views_are_cut_out_and_framed_as_on_the_reference_path(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    image_mod = pytest.importorskip("PIL.Image")
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(server, "staged_module", lambda: Staged)
    px = hooks_px(tmp_path)
    matted: list[tuple[int, int]] = []

    def matte(rgb: Any) -> Any:
        matted.append(rgb.size)
        rgb.putalpha(image_mod.fromarray((np.asarray(rgb)[:, :, 0] > 128).astype(np.uint8) * 255))
        return rgb

    (tmp_path / "images").mkdir()
    sent = []
    for name, azimuth, size in (("left", 90, (120, 300)), ("front", 0, (200, 300))):         # as the hub may send them
        picture = image_mod.new("RGB", (1100, 1100), (20, 20, 20))
        picture.paste((220, 180, 40), (400, 300, 400 + size[0], 300 + size[1]))
        picture.save(tmp_path / "images" / f"{name}.png")
        sent.append(server.ViewSpec(image_path=str(tmp_path / "images" / f"{name}.png"), azimuth=azimuth))
    hooks = server.BatchHooks(px, server.RunNotes())
    req = Staged.PropRequest("p_a", 7, 1024, True, {}, payload=server.BatchItem(
        id="p_a", image_path=sent[1].image_path, out_path=str(tmp_path / "mesh.glb"), views=sent,
        views_dir=str(tmp_path / "views")))
    prepared = hooks.prepare(req, matte)
    assert prepared.meta == {"azimuths": [0, 90]} and matted == [(1100, 1100)] * 2            # the front one first
    assert [(im.mode, im.size) for im in prepared.images] == [("RGBA", (1024, 1024))] * 2
    boxes = [mv.object_box(im) for im in prepared.images]
    heights = [box[3] - box[1] for box in boxes]             # different pictures: one common height, 1/1.1 of the frame
    assert heights[0] == pytest.approx(1024 / 1.1, abs=2) and heights[1] == pytest.approx(heights[0], abs=2)
    mv.wait_for_views()
    assert sorted(p.name for p in (tmp_path / "views").iterdir()) == ["0_front.png", "1_left.png"]
    assert px.told == [(server.REMOVING_BACKGROUND, 0, 2), (server.REMOVING_BACKGROUND, 1, 2)]

    views = hooks.bundle(req, prepared.images, prepared.meta)                                 # what the models take
    assert {size: tuple(t.shape) for size, t in views["images"].items()} == {512: (1, 2, 3, 512, 512),
                                                                            1024: (1, 2, 3, 1024, 1024)}
    assert views["images"][512].dtype == torch.float32 and views["mesh_scale"] == 1.0
    assert tuple(views["transform_matrix"].shape) == (1, 2, 4, 4) and tuple(views["camera_angle_x"].shape) == (1, 2)
    assert views["transform_matrix"][0, 1].tolist() == mv.camera_to_world(90)


def test_the_camera_is_estimated_by_app_pys_own_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """estimate_camera against app.py's get_camera_params_wild_moge itself (read from the fork) for the same
    picture and the same MoGe answer: the same three numbers to the bit. On the CPU, with a stand-in for MoGe."""
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    image_mod = pytest.importorskip("PIL.Image")
    import math

    asked: list[Any] = []

    class MoGe:
        def infer(self, image: Any) -> dict[str, Any]:
            asked.append(image)
            fx = 0.5 + float(image.mean())                   # an answer that depends on the picture it was given
            return {"intrinsics": torch.tensor([[[fx, 0.0, 0.5], [0.0, fx * 1.5, 0.5], [0.0, 0.0, 1.0]]])}

    app = {"torch": torch, "np": np, "math": math, "Image": image_mod, "LOW_VRAM": False, "moge_model": MoGe()}
    for name in ("compute_f_pixels", "distance_from_fov", "get_camera_params_wild_moge"):
        forks_function(paths.PIXAL3D_DIR / "app.py", name, app)
    px = hooks_px(tmp_path, distance_from_fov=app["distance_from_fov"], WILD_MESH_SCALE=1.0, WILD_EXTEND_PIXEL=0,
                  WILD_IMAGE_RESOLUTION=512)
    picture = image_mod.effect_noise((640, 512), 60).convert("RGB")
    picture.save(tmp_path / "prepared.png")
    reference = app["get_camera_params_wild_moge"](str(tmp_path / "prepared.png"), device="cpu", mesh_scale=1.0,
                                                   extend_pixel=0, image_resolution=512)
    ours = server.estimate_camera(px, MoGe(), str(tmp_path / "prepared.png"), device="cpu")
    assert ours == reference and list(ours) == ["camera_angle_x", "distance", "mesh_scale"]
    assert all(type(value) is float for value in ours.values())
    assert torch.equal(asked[0], asked[1]) and tuple(asked[1].shape) == (3, 512, 640)

    seen: list[tuple[bool, str]] = []                        # the hook: from the prepared picture saved to a file

    def estimate(px_: Any, moge: Any, path: str) -> dict[str, float]:
        with image_mod.open(path) as saved:
            seen.append((Path(path).parent == tmp_path, saved.mode))
            assert saved.tobytes() == picture.tobytes()
        return ours

    monkeypatch.setattr(server, "estimate_camera", estimate)
    hooks = server.BatchHooks(px, server.RunNotes())
    assert hooks.camera(Staged.PropRequest("p_a", 7, 1024, False, {}), picture, MoGe()) == ours
    assert seen == [(True, "RGB")] and px.told == [(server.ESTIMATING_CAMERA, 0, 1)]
    assert not list(tmp_path.glob("spellcast_*.png"))        # the file is gone again


def test_both_paths_hand_the_decoded_mesh_to_pixal3d_glb_with_the_launchers_settings(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """glb_steps: the mesh, its layout and resolution, the request's detail, texture size and file, the GLB's turn
    and the streak check go to pixal3d_glb.finish_steps (which test_pixal3d_glb.py checks against o_voxel's
    to_glb); the reference path asks for the bar of every step (inline)."""
    taken: list[tuple[Any, ...]] = []
    monkeypatch.setattr(server.pixal3d_glb, "finish_steps", lambda *args, **kwargs: taken.append((args, kwargs))
                        or "steps")
    px, mesh, layout = object(), object(), {"base_color": slice(0, 3)}
    views = [server.ViewSpec(image_path="f.png"), server.ViewSpec(image_path="b.png", azimuth=180)]
    req = server.BatchItem(id="p_a", image_path="f.png", out_path="o.glb", decimation_target=250_000,
                           texture_size=4096, views=views)
    assert server.glb_steps(px, mesh, layout, 1408, req, True, 2, True) == "steps"
    assert server.glb_steps(px, mesh, layout, 1024, req.model_copy(update={"views": None}), False, 0, False,
                            inline=True) == "steps"
    assert taken == [
        ((px, mesh, layout, 1408, 250_000, 4096, "o.glb", True, 2, True, server.GLB_ROTATION, server.streak_warning),
         {"views": 2, "inline": False, "work_folder": None}),
        ((px, mesh, layout, 1024, 250_000, 4096, "o.glb", False, 0, False, server.GLB_ROTATION,
          server.streak_warning), {"views": 0, "inline": True, "work_folder": None})]


def test_the_finish_hook_hands_the_engine_the_steps_that_write_the_glb(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """The hook gives the engine glb_steps for the prop: its layout and the resolution used, the request, how
    it was made (views, redraws, blocked). A prop deleted before fails at once; one deleted during the clean-up
    fails in the steps' last one (test_pixal3d_glb.py)."""
    px = hooks_px(tmp_path)
    made: list[tuple[Any, ...]] = []
    monkeypatch.setattr(server, "glb_steps", lambda *args, **kwargs: made.append((*args, kwargs)) or "steps")
    monkeypatch.setattr(server, "staged_module", lambda: Staged)
    hooks = server.BatchHooks(px, server.RunNotes())
    out = tmp_path / "props" / "p_a" / "mesh" / "mesh.partial.glb"
    out.parent.parent.mkdir(parents=True)
    views = [server.ViewSpec(image_path="f.png"), server.ViewSpec(image_path="b.png", azimuth=180)]
    payload = server.BatchItem(id="p_a", image_path="f.png", out_path=str(out), decimation_target=5000,
                               texture_size=1024, views=views)
    layout = {"base_color": slice(0, 3)}
    mesh = Staged.DecodedMesh(None, None, None, None, 1280, layout)
    assert hooks.finish(Staged.PropRequest("p_a", 7, 1536, True, {}, payload=payload), mesh,
                        {"ss_retries": 2, "ss_blocked": True, "slow": ["finish"]}) == "steps"
    assert made == [(px, mesh, layout, 1280, payload, True, 2, True, {"work_folder": None})]
    made.clear()
    hooks.work_dir = tmp_path / "temp" / "mesh-j1"          # in a batch: the prop's folder in the work folder
    assert hooks.finish(Staged.PropRequest("p_a", 7, 1024, False, {}, payload=payload), mesh, None) == "steps"
    assert made == [(px, mesh, layout, 1280, payload, False, 0, False, {"work_folder": hooks.work_dir / "p_a"})]
    assert px.told == []
    made.clear()
    monkeypatch.setenv("PIXAL3D_VOXELS_ON_DISK", "off")    # the volume kept in RAM
    hooks.finish(Staged.PropRequest("p_a", 7, 1024, False, {}, payload=payload), mesh, None)
    assert made[0][-1] == {"work_folder": None}
    monkeypatch.delenv("PIXAL3D_VOXELS_ON_DISK")

    # The artist deleted the prop: it fails alone ("input": nothing to retry), before the minutes of the clean-up.
    out.parent.parent.rmdir()
    made.clear()
    with pytest.raises(Staged.PropFailed) as failed:
        hooks.finish(Staged.PropRequest("p_a", 7, 1024, False, {}, payload=payload), mesh, {})
    assert (failed.value.kind, failed.value.detail) == ("input", mv.PROP_DELETED) and made == []
    assert not out.parent.parent.exists()


def test_a_prop_made_at_less_detail_to_fit_the_memory_says_so_with_its_model(tmp_path: Path,
                                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine's "memory_cap" note (the resolution it chose, the one the token cap alone would have given, and
    what was short: the card or the RAM) becomes a warning in the steps' result, beside any other."""
    px = hooks_px(tmp_path)
    monkeypatch.setattr(server, "staged_module", lambda: Staged)
    monkeypatch.setattr(server, "glb_steps", lambda *args, **kwargs: Staged.FinishSteps(
        [], {}, lambda state: {"mode": "single", "views": 1, "warning": kwargs.get("warning"), "ss_retries": 0,
                              "resolution": 1152}))
    hooks = server.BatchHooks(px, server.RunNotes())
    out = tmp_path / "props" / "p_a" / "mesh" / "mesh.partial.glb"
    out.parent.parent.mkdir(parents=True)
    payload = server.BatchItem(id="p_a", image_path="f.png", out_path=str(out), decimation_target=5000,
                               texture_size=1024)
    mesh = Staged.DecodedMesh(None, None, None, None, 1152, {})
    req = Staged.PropRequest("p_a", 7, 1536, False, {}, payload=payload)
    steps = hooks.finish(req, mesh, {"memory_cap": {"by": "card", "resolution": 1152, "without": 1536, "tokens": 9}})
    assert steps.result(steps.state)["warning"] == \
        "Made at quality 1152 instead of 1536: the graphics card's free memory could not hold more."
    steps = hooks.finish(req, mesh, {"memory_cap": {"by": "ram", "resolution": 1280, "without": 1536, "tokens": 9}})
    assert steps.result(steps.state)["warning"].endswith("the PC's free memory could not hold more.")
    # a note that says the memory changed nothing, or none: the result is the steps' own
    for notes in ({"memory_cap": {"by": "card", "resolution": 1536, "without": 1536, "tokens": 9}}, {}, None):
        steps = hooks.finish(req, mesh, notes)
        assert steps.result(steps.state)["warning"] is None
    # a card without a usable decode-in-boxes record (below sm_80): the words say so, and promise the next start's
    # measurement only where the hub said at the launch that it runs then (CALIBRATION_DUE_ENV, adapters/pixal3d.py)
    from spellcast.install import components

    assert server.MEASURED_AGAIN == {True: components.MEASURED_AT_NEXT_START, False: components.MEASURED_ON_REQUEST}
    monkeypatch.setattr(server, "decode_unmeasured", lambda: True)
    capped = {"memory_cap": {"by": "card", "resolution": 1024, "without": 1536, "tokens": 9}}
    for due, again in (("1", components.MEASURED_AT_NEXT_START), ("0", components.MEASURED_ON_REQUEST),
                       (None, components.MEASURED_ON_REQUEST)):
        if due is None:
            monkeypatch.delenv(server.CALIBRATION_DUE_ENV, raising=False)
        else:
            monkeypatch.setenv(server.CALIBRATION_DUE_ENV, due)
        steps = hooks.finish(req, mesh, capped)
        assert steps.result(steps.state)["warning"] == (
            "Made at quality 1024 instead of 1536: this graphics card's one-time measurement did not finish, and "
            f"without it the card's memory cannot hold more. {again}"), due
    steps = hooks.finish(req, mesh, {"memory_cap": {"by": "ram", "resolution": 1280, "without": 1536, "tokens": 9}})
    assert steps.result(steps.state)["warning"].endswith("the PC's free memory could not hold more.")   # RAM: as before


def test_the_finish_hooks_steps_take_the_bar_from_the_decode_to_the_end() -> None:
    """The engine's three steps of "finish" end at DECODED; the hook's steps after them (4..n of n) share the rest
    up to 1.0, so the bar keeps moving while a prop is unwrapped and written."""
    record = batch.Batch("b1", [batch.Item("p_a")], lambda error, item: "")
    record.on_event(types.SimpleNamespace(kind="part_start", part="finish", prop_id=None, step=0, total=0,
                                          detail={"props": ["p_a"]}))
    record.on_event(types.SimpleNamespace(kind="prop_start", part="finish", prop_id="p_a", step=0, total=0,
                                          detail={"position": 1, "of": 1}))
    seen = []
    for step, total in ((1, 3), (2, 3), (3, 3), (4, 7), (5, 7), (6, 7), (7, 7)):
        record.on_event(types.SimpleNamespace(kind="step", part="finish", prop_id="p_a", step=step, total=total,
                                              detail={}))
        seen.append(round(record.items["p_a"].fraction, 3))
    start, decoded = batch.FINISH.start, batch.DECODED
    assert seen == [round(start + (decoded - start) * k / 3, 3) for k in (1, 2, 3)] + \
        [round(decoded + (1.0 - decoded) * k / 4, 3) for k in (1, 2, 3, 4)], seen
    assert record.items["p_a"].message == "Building the 3D mesh · step 7 of 7"


def test_the_read_ahead_takes_the_first_networks_in_their_order_and_holds_while_the_engine_reads(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The launcher's read-ahead starts with the networks a batch loads first (background removal, MoGe-2 for a
    single picture's job, DINOv3), and it waits while the engine reads a network (net_load .. net_ready)."""
    models = tmp_path / "models"
    for relative in ("RMBG-2.0/model.safetensors", "dinov3/model.safetensors",
                     "hub/models--Ruicheng--moge-2-vitl/snapshots/abc/model.pt"):
        (models / relative).parent.mkdir(parents=True, exist_ok=True)
        (models / relative).write_bytes(b"x" * 10)
    (models / "hub/models--Ruicheng--moge-2-vitl/refs").mkdir(parents=True)
    (models / "hub/models--Ruicheng--moge-2-vitl/refs/main").write_text("abc", encoding="utf-8")
    single = server.first_networks(models, "single")
    assert [[path.relative_to(models).as_posix() for path in step] for step in single] == [
        ["RMBG-2.0/model.safetensors"], ["hub/models--Ruicheng--moge-2-vitl/snapshots/abc/model.pt"],
        ["dinov3/model.safetensors"]]
    assert server.first_networks(models, "multi")[1] == []
    assert server.first_networks(tmp_path / "nowhere", "single") == [[], [], []]

    gate = server.ReadAheadGate()
    gate.wait()                                                     # open: no wait
    gate.on_event(types.SimpleNamespace(kind="net_load", detail={"bytes_read": 0}))
    waited = threading.Event()
    reader = threading.Thread(target=lambda: (gate.wait(), waited.set()))
    reader.start()
    time.sleep(0.1)
    assert not waited.is_set()                                      # held while the engine reads
    gate.on_event(types.SimpleNamespace(kind="net_load", detail={"bytes_read": 5}))   # progress: still held
    time.sleep(0.05)
    assert not waited.is_set()
    gate.on_event(types.SimpleNamespace(kind="net_ready", detail={}))
    reader.join(5)
    assert waited.is_set()
    monkeypatch.setattr(server, "READ_AHEAD_HOLD_S", 0.05)          # and never longer than READ_AHEAD_HOLD_S
    gate.on_event(types.SimpleNamespace(kind="net_load", detail={}))
    began = time.monotonic()
    gate.wait()
    assert time.monotonic() - began < 2


def test_the_gpu_worker_says_when_it_is_idle() -> None:
    """exit_at_once_when_idle ends Pixal3D at the hub's stop only when nothing is queued or running."""
    gpu = server.GpuWorker()
    assert gpu.idle()
    go = threading.Event()
    started = threading.Event()
    future = gpu.submit(lambda: (started.set(), go.wait(5)))
    queued = gpu.submit(lambda: None)
    started.wait(5)
    assert not gpu.idle()
    go.set()
    future.result(5)
    queued.result(5)
    end = time.monotonic() + 5
    while not gpu.idle() and time.monotonic() < end:
        time.sleep(0.01)
    assert gpu.idle()
