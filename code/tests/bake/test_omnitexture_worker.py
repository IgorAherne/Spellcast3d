"""Omni-Texture's CPU half (launchers/omnitexture_worker.py) on a box whose six faces each have their own UV island:
what Qwen-Image is shown (the depth map of all the model, no mask), which texels take a picture (only those facing
the camera at 0.5 or more, every view alike, each from the camera facing it best, the seams between views blended
in 3D), and the GLB.

    code\\venvs\\imesh\\Scripts\\python -m pytest code/tests/bake/test_omnitexture_worker.py
"""

from __future__ import annotations

import io
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import omnitexture_masks as ot_masks
import omnitexture_model as ot_model
import omnitexture_repaint as ot_repaint
import omnitexture_sheets as ot_sheets
import omnitexture_tone as ot_tone
import omnitexture_views as ot_views
import omnitexture_worker as w
from spellbake import BakeError, glb
from omnitexture_blend import blend

RES = 128          # the box's texture
SIZE = 128         # the views' pictures
GREY = (128, 128, 128)
# (normal, a, b): a x b = normal, so (-a-b, a-b, a+b, -a+b) winds counter-clockwise seen from outside
FACES = {
    "+Z": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    "-Z": ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
    "+X": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    "-X": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "+Y": ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    "-Y": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
}
PAD = 0.03
# The box stands turned a little about its upright axis (its front towards the cameras' right): seen through the
# orthographic cameras, no face then faces one of them at exactly FACING (a face 60 degrees off does: 0.5).
TURN = np.radians(10.0)
SPIN = np.array([[np.cos(TURN), 0.0, np.sin(TURN)], [0.0, 1.0, 0.0], [-np.sin(TURN), 0.0, np.cos(TURN)]])
TURNED_TO_FRONT_RIGHT = 20.0    # degrees about the upright axis that turn a +z face towards the front right camera


def island(i: int) -> tuple[float, float, float, float]:
    col, row = i % 3, i // 3
    return col / 3 + PAD, row / 2 + PAD, (col + 1) / 3 - PAD, (row + 1) / 2 - PAD


def box(path: Path, size: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Path:
    """The box (half sides `size` along x, y, z before its turn), written as a GLB."""
    positions, normals, uvs, faces = [], [], [], []
    for i, (n, a, b) in enumerate(FACES.values()):
        n, a, b = (np.array(v, float) for v in (n, a, b))
        u0, v0, u1, v1 = island(i)
        base = len(positions)
        for sa, sb, uv in ((-1, -1, (u0, v0)), (1, -1, (u1, v0)), (1, 1, (u1, v1)), (-1, 1, (u0, v1))):
            positions.append(SPIN @ ((n + sa * a + sb * b) * np.array(size)))
            normals.append(SPIN @ n)
            uvs.append(uv)
        faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    image = Image.new("RGB", (RES, RES), GREY)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    material = glb.Material(name="box", base_color=glb.Texture(buffer.getvalue(), "basecolor"),
                            metallic_factor=0.0)
    primitive = glb.Primitive(positions=np.array(positions), normals=np.array(normals), faces=np.array(faces),
                              uv=np.array(uvs))
    glb.write_glb(path, [primitive], [material])
    return path


def centre(texture: np.ndarray, face: str) -> np.ndarray:
    """The colours of the middle of a face's island (rows from the top: v up)."""
    u0, v0, u1, v1 = island(list(FACES).index(face))
    um, vm = (u0 + u1) / 2, (v0 + v1) / 2
    du, dv = (u1 - u0) / 4, (v1 - v0) / 4
    rows = slice(int((1 - vm - dv) * RES), int((1 - vm + dv) * RES))
    cols = slice(int((um - du) * RES), int((um + du) * RES))
    return texture.reshape(RES, RES, 3)[rows, cols].reshape(-1, 3)


def centre_texels(prop: w.Prop, face: str) -> np.ndarray:
    """The texels (indices into the texture) of the middle of a face's island, as `centre` reads them."""
    u0, v0, u1, v1 = island(list(FACES).index(face))
    um, vm = (u0 + u1) / 2, (v0 + v1) / 2
    du, dv = (u1 - u0) / 4, (v1 - v0) / 4
    rows = np.arange(int((1 - vm - dv) * RES), int((1 - vm + dv) * RES))
    cols = np.arange(int((um - du) * RES), int((um + du) * RES))
    return (rows[:, None] * RES + cols[None, :]).reshape(-1)


def expected_colours(prop: w.Prop, face: str, painted: list[tuple[int, tuple[int, int, int]]]) -> np.ndarray:
    """Per middle texel of a face, the colour of the view ([(view, colour)]) of the camera that faces it best of all
    the prop's cameras, at FACING or more, when that view was painted, else the model's own grey (the box is convex:
    nothing hides a face; its faces are flat and no two cameras face one within BAND of each other)."""
    texels = centre_texels(prop, face)
    where = np.searchsorted(prop.texel, texels)
    normal = prop.normal[where].astype(float)
    facing = np.stack([normal @ -cam.forward for cam in prop.cameras])     # (orthographic: one direction a camera)
    best = facing.argmax(axis=0)
    out = np.tile(np.array(GREY, np.uint8), (len(texels), 1))
    for k, colour in painted:
        out[(best == k) & (facing[k] >= ot_views.FACING)] = colour
    return out


def agree(got: np.ndarray, want: np.ndarray) -> float:
    """The share of texels as expected (the mask's cleaning and the hole closing move a texel or two)."""
    return float((got == want).all(axis=1).mean())


def solid(path: Path, colour: tuple[int, int, int]) -> Path:
    Image.new("RGB", (SIZE, SIZE), colour).save(path)
    return path


@pytest.fixture
def prop(tmp_path: Path) -> w.Prop:
    work = tmp_path / "work"
    work.mkdir()
    p = w.Prop(box(tmp_path / "box.glb"), work, SIZE)
    yield p
    p.close()


def test_the_first_views_mask_is_the_facing_rule_too_and_it_may_show_the_lit_clay(prop: w.Prop) -> None:
    view = prop.view(0, "clay")
    assert view["name"] == "front" and view["side"] == "front"
    render = np.asarray(Image.open(view["render"]).convert("RGB"))
    model = render.any(axis=-1)
    weight = np.asarray(Image.open(view["mask"]).convert("L")) / 255.0
    assert model.mean() > 0.3 and ot_views.FACING == 0.5 and model.mean() < 0.95, "the background is black"
    # Nothing is painted yet, but the mask is still only what faces the camera at FACING or more (the front and
    # the back as every other view): the front face (0.98), not the side seen edge on
    # (0.17); all of the picture or none of it (a box's faces meet at creases: no seam to blend), nothing
    # outside the model.
    side = (render == ot_views.lit_clay(np.array([SPIN @ np.array((-1.0, 0.0, 0.0))]), prop.cameras[0])[0]).all(axis=-1)
    assert side.sum() > 50 and not weight[side].any() and not weight[~model].any()
    assert (weight[model] == 1.0).mean() > 0.7 and set(np.unique(weight)) == {0.0, 1.0}
    # The clay, lit from slightly right of the camera: each flat face one shade whatever its depth (the box's front
    # is turned 10 degrees towards the light and is lit; its left side, seen edge on, faces away: ambient only),
    # bluish grey like the page's clay.
    lit = ot_views.lit_clay(np.array([SPIN @ np.array(n) for n in ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0))]),
                            prop.cameras[0])
    colours, counts = np.unique(render[model], axis=0, return_counts=True)
    main = colours[np.argsort(counts)[::-1][:2]]
    assert {tuple(c) for c in main} == {tuple(c) for c in lit}
    assert lit[0].sum() > 2 * lit[1].sum() and (lit[:, 2] > lit[:, 0]).all()
    assert counts.max() > 0.8 * model.sum(), "a flat face: one shade, no depth gradient"


def test_picture_1_is_the_depth_map_and_picture_2_the_model_with_its_own_texture(prop: w.Prop) -> None:
    view = prop.view(0)
    render = np.asarray(Image.open(view["render"]).convert("RGB"))
    model = render.any(axis=-1)
    textured = np.asarray(Image.open(view["model"]).convert("RGB"))
    assert (textured[model] == GREY).all() and not textured[~model].any(), "the box's own texture, on black"
    # The depth map: grey, nearer is lighter, many shades (the box's own texture is one flat grey), from white for the
    # nearest surface down to a tenth of it for the farthest; the background stays black.
    shades = render[model]
    assert (shades[:, 0] == shades[:, 1]).all() and (shades[:, 1] == shades[:, 2]).all()
    assert len(np.unique(shades[:, 0])) > 20
    assert ot_views.DEPTH_FAR == 0.1 and int(shades.max()) == 255 and int(shades.min()) in (25, 26)      # 0.1 x 255
    assert model.mean() < 0.95, "some background is in the picture, and it is black (model: what is not)"


def test_the_settings_test_may_show_the_models_own_texture_in_place_of_the_depth_map(prop: w.Prop) -> None:
    prop.view(0)
    prop.project(0, solid(prop.work / "red.png", (255, 0, 0)))          # the front
    view = prop.view(3, "texture")                                      # from the front left
    render = np.asarray(Image.open(view["render"]).convert("RGB"))
    model = render.any(axis=-1)
    assert model.sum() > 1000 and (render[model] == GREY).all(), "all the model: its own texture, nothing painted"


def test_a_picture_goes_only_onto_surfaces_that_face_the_camera(prop: w.Prop) -> None:
    prop.view(0)
    prop.project(0, solid(prop.work / "red.png", (255, 0, 0)))
    assert (centre(prop.paint, "+Z") == (255, 0, 0)).all()
    # The camera is level with the box: its top, bottom and sides are seen edge on, the back not at all.
    for face in ("+Y", "+X", "-X", "-Z", "-Y"):
        assert (centre(prop.paint, face) == GREY).all(), face


def test_a_views_layer_holds_its_share_of_the_texture_alone(prop: w.Prop) -> None:
    """project's "layer": the texture with this view's texels alone, the bare clay's grey everywhere else, and the
    gutter beside its texels grown from them; the model's own texture is painted as without it."""
    prop.view(0)
    layer = prop.work / "front.webp"
    prop.project(0, solid(prop.work / "red.png", (255, 0, 0)), layer=layer)
    prop.finish(prop.work / "out.glb", tone=False)                      # waits for the layers
    with Image.open(layer) as opened:
        assert opened.size == (RES, RES)
        pixels = np.asarray(opened.convert("RGB")).astype(int).reshape(-1, 3)
    red, grey = np.array((255, 0, 0)), np.array(ot_tone.LAYER_GREY)

    def close(got: np.ndarray, want: np.ndarray) -> bool:               # (WebP, not lossless)
        return bool((np.abs(got - want) <= 12).all())

    assert close(pixels[centre_texels(prop, "+Z")], red)
    for face in ("+Y", "+X", "-X", "-Z", "-Y"):                         # what the front view did not paint
        assert close(pixels[centre_texels(prop, face)], grey), face
    assert (centre(prop.paint, "+Z") == (255, 0, 0)).all() and (centre(prop.paint, "-Z") == GREY).all()
    # Just outside the painted island (its edge texels are the view's): the gutter took their colour, not the grey.
    u0, v0, u1, v1 = island(list(FACES).index("+Z"))
    row, col = int((1 - (v0 + v1) / 2) * RES), int(u0 * RES)
    while not prop.inside[row * RES + col]:
        col += 1
    edge, gutter = pixels[row * RES + col], pixels[row * RES + col - 2]
    assert not prop.inside[row * RES + col - 2] and np.abs(gutter - edge).max() <= 30 and gutter[0] - gutter[1] > 100


def test_a_load_may_ask_for_the_first_views_only(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    two = w.Prop(box(tmp_path / "box.glb"), work, SIZE, 2)
    try:
        assert [side[0] for side in two.sides] == ["front", "back"] and len(two.cameras) == 2
        for k in (0, 1):
            two.view(k)
            two.project(k, solid(work / f"{k}.png", (10 + k, 200, 30)))
        assert two._sights == {}, "no camera past the last one is looked through"
        result = two.finish(tmp_path / "out" / "omnitexture.glb", tone=False)
        assert [view["name"] for view in result["stats"]["views"]] == ["front", "back"]
        assert (centre(two.paint, "+Z") == (10, 200, 30)).all() and (centre(two.paint, "-Z") == (11, 200, 30)).all()
        assert (centre(two.paint, "+X") == GREY).all()
    finally:
        two.close()


def test_a_view_skipped_paints_nothing_and_leaves_its_surfaces_to_the_next_best_camera(tmp_path: Path) -> None:
    """A load's "skip" (a view the artist took off: the result composed again from the others): the views are laid
    out as ever, but the one skipped is not looked through and owns nothing, so what it painted goes to the camera
    that faces it next best, as if it had never been there."""
    work = tmp_path / "work"
    work.mkdir()
    model = box(tmp_path / "box.glb")
    colours = {k: (10 + k, 200, 30) for k in range(3)}
    for skip in ([], [0]):
        three = w.Prop(model, work, SIZE, 3, skip=skip)
        try:
            assert three.loaded()["windows"] == [[0, 0, SIZE, SIZE]] * 3
            assert sorted(three._sights) == [k for k in range(3) if k not in skip]
            for k in range(3):
                three.project(k, solid(work / f"{k}.png", colours[k]))
            result = three.finish(tmp_path / "out" / "omnitexture.glb", tone=False)
            assert [view["name"] for view in result["stats"]["views"]] == ["front", "back", "front_right"][len(skip):]
            # the front faces the box's front best; without it the front right camera (60 degrees off) paints it
            assert agree(centre(three.paint, "+Z"), np.array(colours[2 if skip else 0], np.uint8)) > 0.99
        finally:
            three.close()


def shielded_wall(path: Path, left: float, right: float, low: float, high: float, turn: float = 0.0,
                  texture: int = RES) -> Path:
    """A wall (2 x 2, facing +z) with a smaller shield 0.6 in front of it (x from `left` to `right`, y from `low`
    to `high`), each with its own UV island, both turned `turn` degrees about the upright axis; its texture
    `texture` px square."""
    spin = np.radians(turn)
    turned = np.array([[np.cos(spin), 0.0, np.sin(spin)], [0.0, 1.0, 0.0], [-np.sin(spin), 0.0, np.cos(spin)]])
    quads = (((-1.0, 1.0), (-1.0, 1.0), 0.0, (0.03, 0.47)), ((left, right), (low, high), 0.6, (0.53, 0.97)))
    positions, normals, uvs, faces = [], [], [], []
    for (x0, x1), (y0, y1), z, (u0, u1) in quads:
        base = len(positions)
        for x, y, uv in ((x0, y0, (u0, 0.03)), (x1, y0, (u1, 0.03)), (x1, y1, (u1, 0.97)), (x0, y1, (u0, 0.97))):
            positions.append(turned @ np.array((x, y, z)))
            normals.append(turned @ np.array((0.0, 0.0, 1.0)))
            uvs.append(uv)
        faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    buffer = io.BytesIO()
    Image.new("RGB", (texture, texture), GREY).save(buffer, "PNG")
    material = glb.Material(name="wall", base_color=glb.Texture(buffer.getvalue(), "basecolor"), metallic_factor=0.0)
    glb.write_glb(path, [glb.Primitive(positions=np.array(positions), normals=np.array(normals),
                                       faces=np.array(faces), uv=np.array(uvs))], [material])
    return path


def test_a_picture_does_not_shine_through_onto_what_lies_behind(tmp_path: Path) -> None:
    """A wall with a smaller shield in front of it, seen from the front: the wall behind the shield takes nothing of
    the picture, right up to the shield's outline. Along the outline lie texels of the wall that are hidden while
    the pixel nearest to them looks past the shield: they must not take the picture either (a texel takes it only
    when none of the four pixels around its place sees something nearer). And the shield's outline is a snap in
    depth (a ghost of the Boy's hand on his jeans): on either side of it, the wall beside
    the shield and the shield's own rim, nothing listens to this camera - Qwen-Image's painting sits a pixel or
    three off the model there. The shield's inside and the wall away from it take the picture."""
    # The shield's right edge is put between the centre of one column of the wall's texels and the pixel nearest to
    # it, which lies further right: that column is hidden, and its pixel looks past the shield.
    texels = ((np.arange(RES) + 0.5) / RES - 0.03) / 0.44 * 2.0 - 1.0     # the wall's texel centres, across
    pixels = ((np.arange(SIZE) + 0.5) / SIZE * 2.0 - 1.0) * ot_views.MARGIN   # the pixel centres (the wall: +-1)
    beside = pixels[np.abs(pixels[None, :] - texels[:, None]).argmin(axis=1)] - texels
    column = next(c for c in range(RES) if 0.2 < texels[c] < 0.8 and beside[c] > 0.003)
    left, right, low, high = -0.5, float(texels[column] + beside[column] / 2.0), -0.37, 0.5
    shielded_wall(tmp_path / "wall.glb", left, right, low, high)
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(tmp_path / "wall.glb", work, SIZE, 1)
    try:
        prop.view(0)
        x, y, z = prop.point[:, 0], prop.point[:, 1], prop.point[:, 2]
        wall, shield = z < 0.3, z > 0.3
        hidden = wall & (x > left) & (x < right) & (y > low) & (y < high)
        # The hidden texels whose own pixel sees the wall, not the shield: the ones that shone through.
        cam = prop.cameras[0]
        rel = prop.point - cam.eye
        px = (rel @ cam.right - cam.left) * cam.ppu - 0.5
        py = (cam.top - rel @ cam.up) * cam.ppu - 0.5
        own = prop.depths[0][np.clip(np.round(py), 0, SIZE - 1).astype(int), np.clip(np.round(px), 0, SIZE - 1).astype(int)]
        past = hidden & (rel @ cam.forward <= own + 1e-6)
        assert past.sum() > 10, "the shield's outline passes between such texels and their pixels"
        chosen, _px, _py, _facing = prop._targets_of(0, prop.masks[0], prop.depths[0])
        taken = np.zeros(len(prop.point), dtype=bool)
        taken[chosen] = True
        assert not (taken & hidden).any(), "nothing the shield hides takes the picture"
        # How far (px of the picture) each texel lies from the shield's outline: inside it for the shield's texels,
        # outside it for the wall's that the shield does not hide.
        step = 1.0 / cam.ppu
        inside = np.minimum(np.minimum(x - left, right - x), np.minimum(y - low, high - y)) / step
        outside = np.maximum(np.maximum(left - x, x - right), np.maximum(low - y, y - high)) / step
        assert ot_masks.SNAP_REACH == 2 and ot_masks.SNAP == 0.01
        assert not taken[shield & (inside < 1.5)].any(), "the shield's rim: beside the snap"
        assert not taken[wall & ~hidden & (outside < 1.5)].any(), "the wall next to the shield: beside the snap"
        assert taken[shield & (inside > 3.5)].all() and (shield & (inside > 3.5)).sum() > 1000
        assert taken[wall & (outside > 3.5)].mean() > 0.95, "the wall away from the shield (but for its own outline)"

        prop.project(0, solid(work / "red.png", (255, 0, 0)))
        colour = prop.paint[prop.texel]
        red = (colour == (255, 0, 0)).all(axis=1)
        edge = 0.08                                   # two of the wall's texels: the rim the hole closing may fill
        shadow = wall & (x > left + edge) & (x < right - edge) & (y > low + edge) & (y < high - edge)
        # (the wall's picture fades in from the shadow's line, where the camera stops seeing it: the 3D blend)
        clear = 0.25
        lit = wall & ((x < left - clear) | (x > right + clear) | (y < low - clear) | (y > high + clear))
        rim = wall & ~hidden & (x > left - 0.03) & (x < right + 0.03) & (y > low - 0.03) & (y < high + 0.03)
        assert shadow.sum() > 100 and lit.sum() > 500 and shield.sum() > 500 and rim.sum() > 50
        # (the shield's picture fades in from its rim, where the camera stops painting it: the 3D blend)
        middle = shield & (inside > 14.0)
        assert middle.sum() > 200 and red[middle].all() and red[lit].mean() > 0.99 and not red[rim].any()
        assert not red[shield & (inside < 1.5)].any(), "the shield's rim keeps the model's own"
        assert not red[shadow].any() and (colour[shadow] == GREY).all(), "the wall behind the shield keeps its own"
        assert not prop.done[prop.texel[shadow]].any()
    finally:
        prop.close()


def test_beside_a_snap_the_surface_goes_to_another_camera_that_sees_it_clear(tmp_path: Path) -> None:
    """The wall and its shield turned towards the front right camera, so that both cameras face the wall and the
    front one faces it better. Beside the shield's outline as the front camera sees it - a snap - the wall does
    not listen to the front camera; the front right camera sees the shield elsewhere over the wall (it floats in
    front of it), so where that strip lies clear of its own snaps and of the shield, it takes it."""
    turn = TURNED_TO_FRONT_RIGHT
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(shielded_wall(tmp_path / "wall.glb", -0.5, 0.45, -0.37, 0.5, turn), work, SIZE, 3)
    try:
        prop._plan()
        spin = np.radians(turn)
        back = np.array([[np.cos(spin), 0.0, -np.sin(spin)], [0.0, 1.0, 0.0], [np.sin(spin), 0.0, np.cos(spin)]])
        own = prop.point @ back.T                                   # the texels before the turn
        wall = own[:, 2] < 0.3
        front, right = prop.cameras[0], prop.cameras[2]
        normal = prop.normal[wall][0]
        assert normal @ -front.forward > normal @ -right.forward > 0.6, "both face the wall, the front one better"
        can = {k: np.zeros(len(prop.point), dtype=bool) for k in (0, 2)}
        weight = {k: np.zeros(len(prop.point), np.float32) for k in (0, 2)}
        for k in (0, 2):
            can[k][prop.reach[k][0]] = True
            weight[k][prop.plan[k][0]] = prop.plan[k][3]
        # The wall next to the shield's outline in the front camera's picture: its places there against the
        # shield's corners' (an upright rectangle in an orthographic picture).
        def places(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            rel = points - front.eye
            return (rel @ front.right - front.left) * front.ppu, (front.top - rel @ front.up) * front.ppu

        corners = np.array([(x, y, 0.6) for x in (-0.5, 0.45) for y in (-0.37, 0.5)]) @ np.linalg.inv(back).T
        cx, cy = places(corners)
        px, py = places(prop.point)
        outside = np.maximum(np.maximum(cx.min() - px, px - cx.max()), np.maximum(cy.min() - py, py - cy.max()))
        strip = wall & (outside > 0.0) & (outside < 1.5)
        assert strip.sum() > 100 and not can[0][strip].any(), "beside its snap the front camera paints nothing"
        clear = strip & can[2]
        assert clear.sum() > 40, "the front right camera sees part of that strip clear"
        assert (weight[2][clear] > 0).mean() > 0.9 and not weight[0][clear].any(), "and takes it"
        far = wall & (outside > 12.0) & can[0] & can[2]
        assert far.sum() > 500 and (weight[0][far] > weight[2][far]).mean() > 0.9, "away from it: the front camera's"
    finally:
        prop.close()


def test_a_snap_is_a_break_in_depth_between_two_parts_of_the_model() -> None:
    """snaps(): a floor at depth 5 with a block at depth 2 in front of it, a ramp running steeply away, and nothing
    (inf) to the right. The block's outline against the floor is a snap, both sides of it, and with a reach of two
    the pixels next to those too; the ramp - as large a step from pixel to pixel all along, a surface seen nearly
    edge on - is none, nor where the ramp begins, nor the model's outline against nothing."""
    depth = np.full((12, 24), 5.0)
    depth[3:9, 3:8] = 2.0                                     # the block
    depth[:, 12:18] = 5.0 + 1.5 * np.arange(1, 7)             # the ramp: 1.5 a pixel, well over the limit
    depth[:, 18:] = np.inf
    one, two = ot_masks.snaps(depth, 1.0, 1), ot_masks.snaps(depth, 1.0, 2)
    block = np.zeros(depth.shape, dtype=bool)
    block[3:9, 3:8] = True
    ring = ot_masks.grow(block, 1) & ~ot_masks.shrink(block, 1)     # the pixels on either side of the block's outline
    corners = np.zeros(depth.shape, dtype=bool)
    corners[[2, 2, 9, 9], [2, 8, 2, 8]] = True                # (they touch the block only across a corner)
    assert np.array_equal(one, ring & ~corners), "both sides of the break, nothing else"
    assert np.array_equal(two, ot_masks.grow(one, 1)) and two.sum() > one.sum()
    assert not one[:, 10:].any(), "a steep surface, where it begins, and the outline against nothing: no snaps"
    assert not ot_masks.snaps(depth, 4.0, 1).any(), "a step under the limit"


def test_keep_colour_keeps_the_paintings_colour_at_the_albedos_brightness() -> None:
    painting = np.array([[[200.0, 120.0, 40.0]], [[100.0, 60.0, 20.0]]], np.float32)    # orange, the second shaded
    albedo = np.full((2, 1, 3), 150.0, np.float32)            # flat, but the add-on took the colour off too
    lin = (ot_tone.keep_colour(albedo, painting) / 255.0) ** ot_tone.GAMMA
    a, p = (albedo / 255.0) ** ot_tone.GAMMA, (painting / 255.0) ** ot_tone.GAMMA
    assert np.allclose(lin @ ot_tone.LUMA, a @ ot_tone.LUMA, rtol=1e-4)             # both as bright as the albedo
    assert np.allclose(lin / (lin @ ot_tone.LUMA)[..., None], p / (p @ ot_tone.LUMA)[..., None], rtol=1e-4)   # orange
    # a black painted pixel has no colour to keep: the albedo's own is used
    green = np.array([[[60.0, 160.0, 60.0]]], np.float32)
    assert np.allclose(ot_tone.keep_colour(green, np.zeros_like(green)), green, atol=1e-3)


def test_with_the_albedo_the_paintings_colour_goes_on_and_is_kept(prop: w.Prop) -> None:
    prop.view(0)
    keep = prop.work / "projected.png"
    prop.project(0, solid(prop.work / "grey.png", (150, 150, 150)),
                 colour=solid(prop.work / "orange.png", (200, 120, 40)), keep=keep)
    mixed = ot_tone.keep_colour(np.full((1, 1, 3), 150.0, np.float32), np.array([[[200.0, 120.0, 40.0]]], np.float32))
    want = np.clip(np.round(mixed[0, 0]), 0, 255)
    assert (centre(prop.paint, "+Z") == want).all()
    with Image.open(keep) as kept:
        assert kept.getpixel((SIZE // 2, SIZE // 2)) == tuple(int(v) for v in want)


def test_later_views_show_the_depth_alone_and_repaint_what_earlier_ones_painted(prop: w.Prop) -> None:
    prop.view(0)
    prop.project(0, solid(prop.work / "red.png", (255, 0, 0)))          # the front
    view = prop.view(3)                                                 # from the front left: the front in sight,
    assert view["name"] == "front_left"                                 # 70 degrees off (the box's turn)
    render = np.asarray(Image.open(view["render"]).convert("RGB"))
    model = render.any(axis=-1)
    grey = render[model]
    assert (grey[:, 0] == grey[:, 1]).all() and (grey[:, 1] == grey[:, 2]).all(), "the depth map on all the model"
    assert len(np.unique(grey[:, 0])) > 20, "(no earlier picture shows: Qwen-Image is shown no mask)"
    textured = np.asarray(Image.open(view["model"]).convert("RGB"))
    assert (textured[model] == GREY).all(), "picture 2: the model's own texture, nothing painted over it"
    # Only the surfaces facing this camera at 0.5 or more take its picture.
    prop.project(3, solid(prop.work / "blue.png", (0, 0, 255)))
    for face in FACES:
        want = expected_colours(prop, face, [(0, (255, 0, 0)), (3, (0, 0, 255))])
        assert agree(prop.paint[centre_texels(prop, face)], want) >= 0.99, face
    assert (centre(prop.paint, "-X") == (0, 0, 255)).all() and (centre(prop.paint, "+Y") == GREY).all()
    assert (centre(prop.paint, "+Z") == (255, 0, 0)).all(), "seen too slanted: it keeps the front's picture"


def test_the_eight_views_paint_every_face_and_the_glb_carries_it(prop: w.Prop, tmp_path: Path) -> None:
    colours = {}
    for k, (name, *_rest) in enumerate(ot_views.VIEWS):
        colour = (20 + 25 * k, 200 - 20 * k, (60 * k) % 255)
        colours[name] = colour
        prop.view(k)
        prop.project(k, solid(prop.work / f"{name}.png", colour))
    # Each texel takes the view whose camera faces it best, at FACING or more.
    painted = [(k, colours[name]) for k, (name, *_rest) in enumerate(ot_views.VIEWS)]
    for face in FACES:
        assert agree(prop.paint[centre_texels(prop, face)], expected_colours(prop, face, painted)) >= 0.99, face
    assert (centre(prop.paint, "+Y") == colours["top"]).all() and (centre(prop.paint, "-Y") == colours["bottom"]).all()
    # From above and below the cameras look at a slant: the front they see beside it is too slanted to take them.
    assert ot_views.SLANT == 65.0 and not (centre(prop.paint, "+Z") == colours["top"]).all(axis=1).any()
    # The right side faces both cameras on the right (0.77 and 0.94): the one facing it best painted it, alone.
    right = (centre(prop.paint, "+X") == colours["back_right"]).all(axis=1).mean()
    assert right > 0.9, right
    expected = {"+Y": "top", "-Y": "bottom"}
    result = prop.finish(tmp_path / "out" / "omnitexture.glb")
    assert result["stats"]["painted"] > 0.97
    texture, kind = ot_model.base_colour((tmp_path / "out" / "omnitexture.glb").read_bytes())
    assert kind == "png" and texture.shape == (RES, RES, 3)
    for face, name in expected.items():
        assert (np.abs(centre(texture, face).astype(int) - colours[name]) <= 1).all(), face
    # The geometry and the other material values are the model's own.
    before, _ = ot_model._gltf((tmp_path / "box.glb").read_bytes())
    after, _ = ot_model._gltf((tmp_path / "out" / "omnitexture.glb").read_bytes())
    assert before["accessors"] == after["accessors"] and before["meshes"] == after["meshes"]
    assert before["materials"] == after["materials"]


def test_the_work_done_ahead_while_qwen_paints_changes_no_byte(tmp_path: Path) -> None:
    """What a view leaves for its projection and for the next view (the texels its picture goes onto, what the next
    camera sees) is made on the worker's own thread while Qwen-Image paints: the same pictures, texture and GLB as
    a worker that makes all of it at the command, skipped views and views asked twice included."""
    class Inline:                                           # nothing made ahead: every command makes its own
        def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> None:
            return None

        def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
            pass

    def run(ahead: bool, out: Path) -> list[bytes]:
        (out / "work").mkdir(parents=True)
        prop = w.Prop(box(out / "box.glb"), out / "work", SIZE)
        if not ahead:
            prop.ahead.shutdown()
            prop.ahead = Inline()                           # type: ignore[assignment]
            prop._sights.clear()                            # the first view's too, made from the load on
        kept: list[bytes] = []
        try:
            for k, (name, *_rest) in enumerate(ot_views.VIEWS):
                view = prop.view(k)
                if k == 3:                                  # asked again, as after a retry
                    view = prop.view(k)
                kept += [Path(view[key]).read_bytes() for key in ("render", "mask")]
                if k == 5:                                  # a view with nothing left to paint is skipped
                    continue
                colour = solid(out / f"{name}.png", (20 + 25 * k, 200 - 20 * k, (60 * k) % 255))
                keep = out / f"{name}_kept.png"
                prop.project(k, colour, colour=solid(out / f"{name}_c.png", (200, 120, 40)), keep=keep)
                kept += [keep.read_bytes(), prop.paint.tobytes()]
            prop.finish(out / "omnitexture.glb")
            kept.append((out / "omnitexture.glb").read_bytes())
        finally:
            prop.close()
        return kept

    made_ahead, made_inline = run(True, tmp_path / "ahead"), run(False, tmp_path / "inline")
    assert len(made_ahead) == len(made_inline) == 8 * 2 + 1 + 7 * 2 and made_ahead == made_inline


def test_the_gutter_fill_in_bands_gives_the_same_bits() -> None:
    """fill.Fill with a pool (the worker's settle after every view): every step in bands of rows on threads, the
    pull's 2 x 2 sums in the order numpy's reduce adds them for two or more channels. Checked on values whose sums
    show any other order (a numpy whose reduce changed its order fails here), odd sizes included."""
    from concurrent.futures import ThreadPoolExecutor

    from spellbake import fill

    rng = np.random.default_rng(7)
    for shape in ((2, 2, 3), (6, 10, 3), (64, 64, 2), (128, 96, 4), (1024, 1024, 3)):
        s = (rng.standard_normal(shape) * 10.0 ** rng.integers(-6, 8, shape)).astype(np.float32)
        out = np.empty((shape[0] // 2, shape[1] // 2, shape[2]), np.float32)
        fill._sum4(s, out)
        h, w = shape[0] // 2, shape[1] // 2
        assert np.array_equal(out, s.reshape(h, 2, w, 2, shape[2]).sum(axis=(1, 3))), shape
    with ThreadPoolExecutor(fill.BANDS) as pool:
        for size in (2, 37, 128, 515):
            covered = rng.random((size, size)) < 0.4
            covered[: size // 3, : size // 4] = False            # a gutter wider than the rings reach
            plan = fill.Fill(covered)
            paint = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            assert np.array_equal(plan(paint, pool), plan(paint)), size
            assert np.array_equal(plan(paint), fill.push_pull(*fill.dilate(paint.astype(np.float32), covered))), size


def test_specks_and_holes_leave_the_mask() -> None:
    mask = np.zeros((40, 40), bool)
    mask[5:35, 5:35] = True
    mask[20, 20] = False                                  # a hole
    mask[1, 1] = True                                     # a speck
    cleaned = ot_masks.clean(mask, 2)
    assert cleaned[20, 20] and not cleaned[1, 1] and cleaned[8:32, 8:32].all()


def test_no_facets_blurs_the_depth_within_the_model_and_keeps_what_stands_in_front() -> None:
    """smooth_depth: a slope made of flat faces loses their edges; the background takes no depth and does not pull
    the model's rim towards it; no pixel moves further than `most`, so a part standing in front of another still
    does."""
    size, most = 96, 0.01
    depth = np.full((size, size), np.inf)
    depth[16:80, 8:88] = (1.0 + (np.arange(size) // 8) * 0.008)[8:88]     # faces 8 px wide, 0.008 apart
    depth[40:60, 30:60] = 0.5                                              # a part well in front of the slope
    smoothed = ot_views.smooth_depth(depth, 6.0, most)
    model = np.isfinite(depth)
    assert (np.isfinite(smoothed) == model).all(), "nothing seeps out of the model, nothing of it is lost"
    assert np.abs(smoothed[model] - depth[model]).max() <= most + 1e-12
    across = smoothed[20, 16:80]                    # along the slope, clear of the part in front
    assert np.abs(np.diff(depth[20, 16:80])).max() == pytest.approx(0.008)
    assert (np.diff(across) > 0).all() and np.diff(across).max() < 0.003, "the slope stays, its faces' edges are gone"
    assert abs(smoothed[16, 40] - depth[16, 40]) < 0.004, "the rim keeps its depth"
    assert smoothed[50, 29] - smoothed[50, 30] > 0.4, "the step between the two stays"


def test_the_gutter_the_brush_marks_beside_a_masked_island_shows_on_no_island_next_to_it(tmp_path: Path) -> None:
    """The page's brush marks a few texels of the gutter round what it masks. A pixel that reads those alone - the
    edge of another island that lies beside the masked one in the texture - shows no mask; one on the masked
    island's own edge does (no unmasked line along a seam)."""
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(box(tmp_path / "box.glb"), work, SIZE, 6, ahead=False)
    try:
        u0, v0, u1, v1 = island(list(FACES).index("+Z"))
        marked = np.zeros((RES, RES), bool)       # the island, and three texels of gutter all round it
        marked[int((1 - v1) * RES) - 3:int((1 - v0) * RES) + 4, max(0, int(u0 * RES) - 3):int(u1 * RES) + 4] = True
        assert (marked & ~prop.inside.reshape(RES, RES)).sum() > 100
        v = (v0 + v1) / 2
        shows = lambda u: bool(prop.repaints._masked_pixels(np.array([[u, v]]), marked)[0])   # noqa: E731
        assert not shows(u1 + 2.0 / RES), "the marked gutter alone: no masked surface"
        assert shows(u1 - 0.2 / RES) and shows((u0 + u1) / 2), "the island's edge and its middle"
    finally:
        prop.close()


def test_no_facets_changes_the_depth_map_a_look_writes_inside_the_model_and_nothing_else(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(box(tmp_path / "box.glb"), work, SIZE, 6, ahead=False)
    try:
        ask = {**lens_at_front(prop, centre_texels(prop, "+Z"), work)[0], "size": 512}
        read = lambda made: {key: np.asarray(Image.open(made[key]).convert("L")).astype(int)   # noqa: E731
                             for key in ("depth", "viewport", "redrawn")}
        plain, smooth = read(prop.look(ask)), read(prop.look({**ask, "smooth": True}))
        assert ((smooth["depth"] > 0) == (plain["depth"] > 0)).all(), "the model's outline as it was"
        moved = np.abs(smooth["depth"] - plain["depth"])
        assert 0 < moved.max() < 30 and (moved > 0).mean() > 0.01, "the box's edges rounded, no more"
        assert (smooth["viewport"] == plain["viewport"]).all() and (smooth["redrawn"] == plain["redrawn"]).all()
    finally:
        prop.close()


def test_a_picture_whose_copy_could_not_be_kept_is_projected_again(prop: w.Prop) -> None:
    """The kept copy of what goes onto the model cannot be written: the command fails and leaves the prop as it
    was, so the same view is projected again."""
    prop.view(0)
    picture = solid(prop.work / "painted.png", (40, 160, 90))
    before = prop.paint.copy()
    with pytest.raises(OSError):
        prop.project(0, picture, keep=prop.work / "no" / "such" / "kept.png")
    assert (prop.paint == before).all()
    assert prop.project(0, picture)["painted"] > 0
    assert agree(centre(prop.paint, "+Z"), expected_colours(prop, "+Z", [(0, (40, 160, 90))])) > 0.9


def test_a_side_camera_keeps_off_what_a_better_camera_paints_earlier_or_later(prop: w.Prop) -> None:
    """A side view painted the jacket's back over the back view's picture. The box's
    right side faces the front right camera at 0.77 and the back right one at 0.94; its left side the front left one
    at 0.94 and the back left one at 0.77: each side takes the better camera's picture alone, whichever comes first."""
    colours = {k: (40 * k, 255 - 40 * k, 90) for k in range(6)}
    for k in range(4):                                  # front, back, front right, front left
        prop.view(k)
        prop.project(k, solid(prop.work / f"{k}.png", colours[k]))
    assert (centre(prop.paint, "+X") == GREY).all(), "the front right camera left it to the back right one"
    assert (centre(prop.paint, "-X") == colours[3]).all()
    for k in (4, 5):                                    # back right, back left
        prop.view(k)
        prop.project(k, solid(prop.work / f"{k}.png", colours[k]))
    assert (centre(prop.paint, "+X") == colours[4]).all()
    assert (centre(prop.paint, "-X") == colours[3]).all(), "the back left camera, later and worse, kept off it"


def sheet(spacing: float = 0.004) -> tuple[np.ndarray, np.ndarray]:
    """A flat sheet facing +z, 1 x 0.4, as points and normals."""
    xs, ys = np.meshgrid(np.arange(0.0, 1.0, spacing), np.arange(0.0, 0.4, spacing))
    points = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    return points, np.tile((0.0, 0.0, 1.0), (len(points), 1))


def test_two_views_cross_fade_across_their_seam_in_3d_and_nowhere_else() -> None:
    """blend: a sheet both views can paint, view 0's left of x = 0.5 and view 1's right of it: across the seam
    their weights cross-fade over about the radius to each side, always adding up to all of the picture (nothing
    is left to the model's own texture); away from it each texel is one view's, exactly."""
    points, normals = sheet()
    owners = (points[:, 0] >= 0.5).astype(np.int16)
    can = [np.ones(len(points), bool), np.ones(len(points), bool)]
    radius = 0.03
    (w0, w1), left = blend(points, normals, owners, can, radius)
    gap = points[:, 0] - 0.5
    assert not left.any() and np.allclose(w0 + w1, 1.0, atol=1e-5)
    far = np.abs(gap) > 3 * radius
    assert np.array_equal(w0[far], (gap[far] < 0).astype(np.float32)) and np.array_equal(w1[far], 1 - w0[far])
    seam = np.abs(gap) < 0.003
    assert (np.abs(w0[seam] - 0.5) < 0.15).all(), "half and half on the seam"
    across = np.argsort(points[points[:, 1] == points[50, 1], 0])       # one row of the sheet, left to right
    row = w0[points[:, 1] == points[50, 1]][across]
    assert (np.diff(row) <= 1e-6).all(), "falling steadily from all of view 0 to none"
    inside = (np.abs(gap) < radius)[points[:, 1] == points[50, 1]][across]
    assert 0.02 < row[inside].min() < 0.2 and 0.8 < row[inside].max() < 0.98, "most of the fade within the radius"


def test_a_views_weight_falls_to_nothing_inside_the_edge_of_what_it_can_paint() -> None:
    """blend: view 0 can paint only left of x = 0.5 (it sees no further) and nothing else can: its weight falls
    to nothing at that edge, inside it, and the model's own texture takes the rest; past the edge it has none."""
    points, normals = sheet()
    able = points[:, 0] < 0.5
    owners = np.where(able, 0, -1).astype(np.int16)
    radius = 0.03
    (w0,), left = blend(points, normals, owners, [able], radius)
    gap = points[:, 0] - 0.5
    assert np.allclose(w0 + left, 1.0, atol=1e-5)
    assert not w0[~able].any() and (left[~able] == 1.0).all(), "past the edge: the model's own texture alone"
    assert (w0[gap < -3 * radius] == 1.0).all() and not left[gap < -3 * radius].any()
    edge = able & (gap > -0.005)
    assert w0[edge].max() < 0.15, "next to nothing at the edge"
    half = able & (np.abs(gap + radius / 2) < 0.003)
    assert (w0[half] > 0.4).all() and (w0[half] < 0.95).all(), "rising inside it"


def test_surfaces_facing_away_from_each_other_never_blend() -> None:
    """blend: a second sheet 4 mm behind the first, facing the other way, another view's: nearer than the radius,
    but the first sheet's weights are what they are without it, and the second is all its own view's."""
    points, normals = sheet()
    n = len(points)
    owners = (points[:, 0] >= 0.5).astype(np.int16)
    alone, _ = blend(points, normals, owners, [np.ones(n, bool), np.ones(n, bool)], 0.03)
    both_points = np.concatenate([points, points - (0.0, 0.0, 0.004)])
    both_normals = np.concatenate([normals, -normals])
    front = np.arange(2 * n) < n
    can = [front, front, ~front]
    weights, left = blend(both_points, both_normals, np.concatenate([owners, np.full(n, 2, np.int16)]), can, 0.03)
    assert np.allclose(weights[0][:n], alone[0], atol=1e-4) and np.allclose(weights[1][:n], alone[1], atol=1e-4)
    assert (weights[2][n:] == 1.0).all() and not weights[2][:n].any() and not left.any()


def test_a_seam_where_the_surface_turns_from_one_side_to_the_next_is_blended_too() -> None:
    """A view's picture ends where the surface turns away from its camera, which is
    often just where its normal's main axis changes - and blend keeps a set of voxels per such side. On a quarter
    of a cylinder, view 0's where it faces +z more than +x and view 1's past 45 degrees, the seam lies on that
    very line: the two cross-fade there all the same, patches facing alike being neighbours whatever their sets."""
    angle, along = np.meshgrid(np.radians(np.arange(0.0, 90.0, 0.25)), np.arange(0.0, 0.3, 0.004))
    angle, along = angle.ravel(), along.ravel()
    normals = np.stack([np.sin(angle), np.zeros(angle.size), np.cos(angle)], axis=1)
    points = normals * 0.5 + np.stack([np.zeros(angle.size), along, np.zeros(angle.size)], axis=1)
    owners = (angle >= np.radians(45.0)).astype(np.int16)
    (w0, w1), left = blend(points, normals, owners, [np.ones(angle.size, bool)] * 2, 0.03)
    seam = np.abs(np.degrees(angle) - 45.0) < 0.5
    assert (np.abs(w0[seam] - 0.5) < 0.2).all() and (np.abs(w1[seam] - 0.5) < 0.2).all()
    far = np.abs(np.degrees(angle) - 45.0) > 15.0              # (0.03 is 3.4 degrees of this arc)
    assert np.array_equal(w0[far], (owners[far] == 0).astype(np.float32)) and not left.any()


def arc(path: Path, res: int) -> Path:
    """Half a cylinder standing upright, facing +z in its middle and +-x at its edges, its texture one grey island."""
    steps = 96
    angles = np.radians(np.linspace(-90.0, 90.0, steps + 1))
    positions, normals, uvs, faces = [], [], [], []
    for i, a in enumerate(angles):
        for y, v in ((-1.0, 0.05), (1.0, 0.95)):
            positions.append((np.sin(a), y, np.cos(a)))
            normals.append((np.sin(a), 0.0, np.cos(a)))
            uvs.append((0.05 + 0.9 * i / steps, v))
    for i in range(steps):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        faces += [(a, c, b), (b, c, d)]
    buffer = io.BytesIO()
    Image.new("RGB", (res, res), GREY).save(buffer, "PNG")
    material = glb.Material(name="arc", base_color=glb.Texture(buffer.getvalue(), "basecolor"), metallic_factor=0.0)
    glb.write_glb(path, [glb.Primitive(positions=np.array(positions), normals=np.array(normals),
                                       faces=np.array(faces), uv=np.array(uvs))], [material])
    return path


def test_a_picture_fades_in_inside_the_edge_of_what_its_camera_sees_and_stays_sharp(tmp_path: Path) -> None:
    """The front camera's red on the half cylinder: it can paint as far as the surface faces it at 0.5 (60 degrees
    to each side). Inside that edge its picture fades in over the model's own grey within a few texels (SEAM of the
    texture's width across: 16 texels at 2048); past it nothing is painted; away from it the
    picture is on whole, and finish blurs nothing."""
    res = 512
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(arc(tmp_path / "arc.glb", res), work, 512, 1)
    try:
        view = prop.view(0)
        weight = np.asarray(Image.open(view["mask"]).convert("L")) / 255.0
        assert w.SEAM == 0.008 and ((weight > 0.05) & (weight < 0.95)).sum() > 200, "the mask shows the fade"
        prop.project(0, solid(work / "red.png", (255, 0, 0)))
        row = prop.paint.reshape(res, res, 3)[res // 2].astype(int)       # across the arc, half way up
        red = np.flatnonzero((row == (255, 0, 0)).all(axis=1))
        grey = (row == GREY).all(axis=1)
        first, last = int(red[0]), int(red[-1])
        assert (row[first:last + 1] == (255, 0, 0)).all(), "the picture whole between its two edges"
        assert grey[40:first - 12].all() and grey[last + 12:-40].all(), "the model's own past them"
        fade = row[first - 12:first]                                      # the left edge: grey -> red
        mixed = (fade[:, 0] > GREY[0]) & (fade[:, 0] < 255)
        assert 3 <= mixed.sum() <= 11 and (np.diff(fade[:, 0]) >= 0).all(), fade[:, 0].tolist()
        # each a mix of the model's grey and the red, nothing blurred: grey * (1 - a) + red * a
        share = (fade[mixed, 0] - GREY[0]) / (255.0 - GREY[0])
        assert np.allclose(share, 1.0 - fade[mixed, 1] / GREY[1], atol=0.02)
        before = prop.paint[prop.texel].copy()
        prop.finish(tmp_path / "out.glb", tone=False)
        assert np.array_equal(prop.paint[prop.texel], before), "finish blurs no colour"
    finally:
        prop.close()


def windows_picture(made: dict, colours: dict[str, tuple[int, int, int]], path: Path) -> Path:
    """A picture of a sheet's size with each view's window filled with that view's colour."""
    picture = np.zeros((made["size"], made["size"], 3), np.uint8)
    for view in made["views"]:
        x, y, width, height = view["window"]
        picture[y:y + height, x:x + width] = colours[view["name"]]
    Image.fromarray(picture).save(path)
    return path


def test_two_views_painted_together_lie_in_one_sheet_and_each_takes_its_own_part(tmp_path: Path) -> None:
    """The front and the back together (one picture for several views, so they agree in
    colour): one sheet of the smallest side that shows the box as large as a view alone does, each view in a
    window of it, the gap apart; the sheet's picture goes onto the model window by window."""
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(box(tmp_path / "box.glb"), work, SIZE, 2, together=[SIZE, 2 * SIZE, 4 * SIZE])
    try:
        assert len(prop.sheets) == 1 and prop.sheets[0].views == [0, 1]
        side = prop.sheets[0].side
        assert side == 2 * SIZE and ot_sheets.FIT <= prop.sheets[0].scale < 1.2, "no larger a sheet than the views need"
        alone = ot_views.camera_alone(prop.mesh.positions, 0.0, 0.0, SIZE)
        assert abs(prop.cameras[0].ppu / alone.ppu - prop.sheets[0].scale) < 1e-6
        assert prop.cameras[0].ppu == prop.cameras[1].ppu, "one scale for all of a sheet's views"
        made = prop.sheet(0)
        assert made["size"] == side and [view["name"] for view in made["views"]] == ["front", "back"]
        (fx, fy, fw, fh), (bx, by, bw, bh) = (view["window"] for view in made["views"])
        room = ot_sheets.GAP * side - 2 * ot_sheets.PAD - side / 128     # (the windows' padding, a cell of the layout's grid)
        assert max(bx - (fx + fw), by - (fy + fh)) >= room, "the gap apart: beside each other, or one above the other"
        assert min(fx, fy, bx, by, side - fx - fw, side - fy - fh, side - bx - bw, side - by - bh) >= room
        for key in ("render", "model", "mask"):
            picture = np.asarray(Image.open(made[key]).convert("L"))
            assert picture.shape == (side, side)
            lit = picture > 0
            within = np.zeros_like(lit)
            within[fy:fy + fh, fx:fx + fw] = within[by:by + bh, bx:bx + bw] = True
            assert lit[fy:fy + fh, fx:fx + fw].any() and lit[by:by + bh, bx:bx + bw].any(), key
            assert not (lit & ~within).any(), f"{key}: the model only in the views' windows"
        colours = {"front": (255, 0, 0), "back": (0, 0, 255)}
        cuts = {name: work / f"{name}.png" for name in colours}
        layers = {name: work / f"{name}.webp" for name in colours}
        done = prop.paint_sheet(0, windows_picture(made, colours, work / "both.png"), keep=work / "whole.png",
                                parts=[{"index": k, "keep": str(cuts[name]), "layer": str(layers[name])}
                                       for k, name in enumerate(colours)])
        assert [view["name"] for view in done["views"]] == ["front", "back"] and done["painted"] > 0
        assert (centre(prop.paint, "+Z") == (255, 0, 0)).all() and (centre(prop.paint, "-Z") == (0, 0, 255)).all()
        assert (centre(prop.paint, "+Y") == GREY).all() and (centre(prop.paint, "+X") == GREY).all()
        result = prop.finish(tmp_path / "out.glb")
        assert result["stats"]["sheets"] == [{"side": side, "views": ["front", "back"],
                                              "scale": round(prop.sheets[0].scale, 3), "matched": 0}]
        assert [view["name"] for view in result["stats"]["views"]] == ["front", "back"]
        with Image.open(work / "whole.png") as whole:
            assert whole.size == (side, side)
        for name, (x, y, width, height) in (("front", (fx, fy, fw, fh)), ("back", (bx, by, bw, bh))):
            cut = np.asarray(Image.open(cuts[name]).convert("RGB"))
            assert cut.shape == (height, width, 3), "a view's own cut of the sheet: its window"
            assert (cut == colours[name]).all(axis=-1).mean() > 0.5
            layer = np.asarray(Image.open(layers[name]).convert("RGB")).reshape(-1, 3).astype(int)
            face = "+Z" if name == "front" else "-Z"
            assert np.abs(layer[centre_texels(prop, face)] - colours[name]).max() <= 24
    finally:
        prop.close()


def test_views_that_would_be_too_small_together_are_painted_alone(tmp_path: Path) -> None:
    """With no larger sheet allowed than a single view's picture, the box's two views would be under half the
    size (below FLOOR): each keeps its own picture, exactly as without sheets."""
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(box(tmp_path / "box.glb"), work, SIZE, 2, together=[SIZE])
    try:
        assert [(sheet.side, sheet.views, sheet.scale) for sheet in prop.sheets] == [(SIZE, [0], 1.0), (SIZE, [1], 1.0)]
        for k, (_, _, azimuth, elevation) in enumerate(ot_views.VIEWS[:2]):
            alone = ot_views.camera_alone(prop.mesh.positions, azimuth, elevation, SIZE)
            cam = prop.cameras[k]
            assert (cam.ppu, cam.left, cam.top, cam.width, cam.height) == (alone.ppu, alone.left, alone.top, SIZE,
                                                                             SIZE)
        prop.view(0)
        prop.project(0, solid(work / "red.png", (255, 0, 0)))
        assert (centre(prop.paint, "+Z") == (255, 0, 0)).all()
    finally:
        prop.close()


def test_six_views_together_paint_each_face_from_its_best_camera_as_alone(tmp_path: Path) -> None:
    """All six cameras around the box in one sheet: every face ends as it does with the views painted one by
    one - from the camera that faces it best - and the views are named in reading order."""
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(box(tmp_path / "box.glb"), work, SIZE, 6, together=[2 * SIZE, 4 * SIZE])
    try:
        assert len(prop.sheets) == 1 and sorted(prop.sheets[0].views) == list(range(6))
        made = prop.sheet(0)
        boxes = {view["index"]: view["window"] for view in made["views"]}
        order = [view["index"] for view in made["views"]]
        for a, b in zip(order, order[1:]):                              # left to right, then the next row
            (ax, ay, _, ah), (bx, by, _, _) = boxes[a], boxes[b]
            assert (abs(ay - by) < ah / 2 and bx > ax) or by > ay + ah / 2
        colours = {name: (20 + 25 * k, 200 - 20 * k, (60 * k) % 255)
                   for k, (name, *_rest) in enumerate(ot_views.VIEWS[:6])}
        prop.paint_sheet(0, windows_picture(made, colours, work / "six.png"))
        painted = [(k, colours[name]) for k, (name, *_rest) in enumerate(ot_views.VIEWS[:6])]
        for face in ("+Z", "-Z", "+X", "-X"):
            assert agree(prop.paint[centre_texels(prop, face)], expected_colours(prop, face, painted)) >= 0.99, face
        assert (centre(prop.paint, "+Y") == GREY).all(), "no camera of the six faces the top"
    finally:
        prop.close()


def test_every_view_of_a_sheet_keeps_its_own_size_and_the_sheets_come_in_the_views_order(tmp_path: Path) -> None:
    """A pillar from all eight cameras, in sheets no larger than a single view's picture: from above and below it
    is far larger alone than from the side, so those two never share a sheet with the cameras around it (there
    they would be shrunk to the others' scale); every view keeps FLOOR of its own size alone; and the sheets are
    painted in the views' order - the front's first - whatever order the layout looked through them in."""
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(box(tmp_path / "pillar.glb", (1.0, 4.0, 1.0)), work, SIZE, 8, together=[SIZE])
    try:
        assert sorted(k for sheet in prop.sheets for k in sheet.views) == list(range(8)), "each view once"
        firsts = [min(sheet.views) for sheet in prop.sheets]
        assert firsts == sorted(firsts) and firsts[0] == 0
        assert any(len(sheet.views) > 1 for sheet in prop.sheets), "the slim views around it do fit together"
        for sheet in prop.sheets:
            kept = []
            for k in sheet.views:
                _, _, azimuth, elevation = ot_views.VIEWS[k]
                alone = ot_views.camera_alone(prop.mesh.positions, azimuth, elevation, SIZE)
                kept.append(prop.cameras[k].ppu / alone.ppu)
            assert min(kept) >= ot_sheets.FLOOR - 1e-9 and abs(sheet.scale - min(kept)) < 1e-6, (sheet.views, kept)
            assert len(sheet.views) == 1 or not {6, 7} & set(sheet.views) or set(sheet.views) <= {6, 7}
        assert ot_sheets.FLOOR == 0.9
    finally:
        prop.close()


def test_the_views_that_fit_together_are_found_whatever_their_order(tmp_path: Path) -> None:
    """A beam pointing at the front camera (turned a little, as the box is), from the six cameras around it: end
    on (front, back) it is short, and alone it is shown far larger than from the sides, so an end view never
    shares a sheet with a side view - but the two end views share one, and each side view shares one with the
    side view that shows the beam as long (the turn makes two of them longer). Taken in the cameras' order
    (front, the sides, back) nearly every view would start a sheet of its own; the search over the combinations
    finds three sheets."""
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(box(tmp_path / "beam.glb", (0.3, 0.3, 3.0)), work, SIZE, 6, together=[SIZE])
    try:
        assert sorted(sorted(sheet.views) for sheet in prop.sheets) == [[0, 1], [2, 5], [3, 4]]
        assert all(sheet.scale >= ot_sheets.FLOOR - 1e-9 for sheet in prop.sheets)
        assert [min(sheet.views) for sheet in prop.sheets] == [0, 2, 3], "the sheets in the views' order"
    finally:
        prop.close()


def test_a_part_thinner_than_a_pixel_of_the_outline_is_still_in_it(tmp_path: Path) -> None:
    """A wall with a needle sticking out of its side, a thousandth of the model's width thick: the rays the outline
    is found with pass above and below it, but its edges are drawn into the outline, all along - or the layout
    would put another view over it."""
    positions, normals, uvs, faces = [], [], [], []
    for x0, x1, y0, y1, u0, u1 in ((-1.0, 0.0, -1.0, 1.0, 0.03, 0.47), (0.0, 1.0, 0.1002, 0.1012, 0.53, 0.97)):
        base = len(positions)
        for x, y, uv in ((x0, y0, (u0, 0.03)), (x1, y0, (u1, 0.03)), (x1, y1, (u1, 0.97)), (x0, y1, (u0, 0.97))):
            positions.append((x, y, 0.0))
            normals.append((0.0, 0.0, 1.0))
            uvs.append(uv)
        faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    buffer = io.BytesIO()
    Image.new("RGB", (RES, RES), GREY).save(buffer, "PNG")
    material = glb.Material(name="needle", base_color=glb.Texture(buffer.getvalue(), "basecolor"), metallic_factor=0.0)
    glb.write_glb(tmp_path / "needle.glb", [glb.Primitive(positions=np.array(positions), normals=np.array(normals),
                                                          faces=np.array(faces), uv=np.array(uvs))], [material])
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(tmp_path / "needle.glb", work, SIZE, 1)
    try:
        frame = ot_views.frame(prop.mesh.positions, 0.0, 0.0)
        unit = ot_sheets.OUTLINE / 2.0                                  # the model is 2 wide: 192 px a unit
        outline = ot_sheets.outline(prop.scene, prop.mesh, frame, unit)
        row = int((1.0 - 0.1007) * unit)                        # the needle's row: thinner (0.19 px) than a pixel
        half = ot_sheets.OUTLINE // 2
        assert outline.shape == (ot_sheets.OUTLINE, ot_sheets.OUTLINE) and outline[:, : half - 1].all(), "the wall"
        assert outline[row, ot_sheets.OUTLINE // 2:].all(), "the needle, to its tip"
        assert not outline[row - 3, half + 2:].any() and not outline[row + 3, half + 2:].any()
    finally:
        prop.close()


def test_the_own_texture_takes_the_tone_of_the_paintings_near_it() -> None:
    """match_tone on a strip of surface, the model's own red all along it: its left end painted darker, its right
    end lighter, nothing between them. The own texture beside the dark painting takes that painting's change,
    beside the light one that one's, and between them something between the two (never, as when the change was
    learnt per colour over the whole model, one figure for every red); a half-painted texel has its own half
    retinted; the paintings and the own texture's detail (a texel a little lighter than those around it) stay."""
    x, y = np.meshgrid(np.arange(300.0), np.arange(20.0))
    points = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    along = points[:, 0]
    red, dark, light = (200, 60, 60), (140, 42, 42), (220, 66, 66)
    own = np.tile(np.array(red, np.uint8), (len(points), 1))
    detail = int(np.flatnonzero((along == 110) & (points[:, 1] == 10))[0])
    own[detail] = (210, 63, 63)
    paint = own.copy()
    paint[along < 100], paint[along >= 200] = dark, light
    share = ((along >= 100) & (along < 200)).astype(np.float32)
    half = along == 99
    share[half], paint[half] = 0.5, (170, 51, 51)                         # half the dark painting, half the own red
    out = ot_tone.match_tone(points, own, paint, share, 4.0).astype(int)
    whole = share == 0
    assert np.array_equal(out[whole], paint[whole]), "what the paintings hold whole stays"
    assert np.abs(out[half] - dark).max() <= 2, "its own half of a half-painted texel as the painting beside it"
    assert np.abs(out[(along >= 100) & (along < 104)] - dark).max() <= 3, "beside the dark painting: its change"
    assert np.abs(out[(along >= 196) & (along < 200)] - light).max() <= 3, "beside the light one: that one's"
    between = out[(share == 1) & (np.arange(len(points)) != detail)]
    assert (between >= np.array(dark) - 1).all() and (between <= np.array(light) + 1).all()
    middle = out[(along >= 140) & (along < 160)][:, 0]
    assert middle.min() > dark[0] + 10 and middle.max() < light[0] - 10, "half way: neither painting's alone"
    lighter, plain = out[detail], out[detail + 1]
    assert lighter[0] > plain[0] and abs(lighter[0] / plain[0] - 210 / 200) < 0.03, "detail kept, in the new tone"
    assert ot_tone.match_tone(points, own, own, np.ones(len(own), np.float32), 4.0) is own, "nothing painted: as it was"


def test_only_small_holes_take_their_neighbours_colour_however_often_the_paint_settles(tmp_path: Path) -> None:
    """The half cylinder's island painted red in its left half, by hand, with a hole of 2 x 2 texels in the red:
    settle closes the hole and leaves the unpainted half the model's own to the last texel, however often it runs
    (once per sheet and once more after the tone match). It used to give every unpainted texel next to a painted
    one that one's colour and count it as painted: the red crept a texel further into the grey at every call, a
    dilation beside where a picture ended."""
    res = 512
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(arc(tmp_path / "arc.glb", res), work, 512, 1)
    try:
        prop._plan()
        inside = prop.inside.reshape(res, res)
        columns = np.flatnonzero(inside.any(axis=0))
        middle = int(columns[0] + columns[-1]) // 2
        left = inside & (np.arange(res)[None, :] < middle)
        hole = np.zeros((res, res), bool)
        hole[res // 2:res // 2 + 2, middle - 40:middle - 38] = True
        painted = (left & ~hole).reshape(-1)
        prop.paint[painted] = (255, 0, 0)
        prop.done[painted] = True
        for _ in range(4):
            prop.settle()
        paint = prop.paint.reshape(res, res, 3)
        assert (paint[hole] == (255, 0, 0)).all(), "the hole takes the red around it"
        assert (paint[inside & ~left] == GREY).all(), "the unpainted half stays the model's own"
        assert int(prop.filled.sum()) == int(hole.sum()) and int(prop.done.sum()) == int(left.sum())
    finally:
        prop.close()


def test_a_pictures_tone_is_read_per_colour_from_texels_two_pictures_share() -> None:
    """tone_table / recolour: a picture whose blue is a quarter brighter than another's and whose grey nearly the
    same (the Boy's back sheet against his front sheet) - read from the colours the two show of the same texels,
    its blue is brought down and its grey hardly moved: one change per colour, where one curve a channel would
    give both the same. Texels that show another thing in the two (an edge a pixel off) are left out; a colour
    the pairs never held takes the change of all of them; pictures that are not two paintings of one thing, or
    share too little, are not matched at all."""
    rng = np.random.default_rng(3)
    blue, grey, skin = (61, 112, 165), (77, 84, 89), (225, 176, 147)
    ours_blue, ours_grey = (58, 100, 144), (75, 80, 86)
    n = 6000

    def many(colour: tuple[int, int, int]) -> np.ndarray:
        return np.clip(np.array(colour) + rng.integers(-3, 4, (n, 3)), 0, 255).astype(np.uint8)

    theirs, ours = np.concatenate([many(blue), many(grey)]), np.concatenate([many(ours_blue), many(ours_grey)])
    odd = rng.choice(2 * n, n // 10, replace=False)
    ours[odd] = np.array([(250, 250, 250), (10, 10, 10)], np.uint8)[rng.integers(0, 2, len(odd))]
    table = ot_tone.tone_table(theirs, ours)
    assert table is not None and table.shape == (ot_tone.AGREE_BINS,) * 3 + (3,)
    got = ot_tone.recolour(np.array([blue, grey, skin], np.uint8), table).astype(int)
    assert np.abs(got[0] - ours_blue).max() <= 3, got[0]
    assert np.abs(got[1] - ours_grey).max() <= 3, got[1]
    assert got[0][2] - blue[2] < 3 * (got[1][2] - grey[2]) < 0, "the blue changed far more than the grey"
    alike = np.setdiff1d(np.arange(2 * n), odd)
    overall = ot_tone._linear(ours[alike]).sum(axis=0) / ot_tone._linear(theirs[alike]).sum(axis=0)
    wanted = np.clip(ot_tone._linear(np.array(skin, np.uint8)) * overall, 0, 1) ** (1 / ot_tone.GAMMA) * 255
    assert np.abs(got[2] - wanted).max() <= 2, "a colour never met: the change of all together"
    red, green = np.full((5000, 3), (200, 40, 40), np.uint8), np.full((5000, 3), (40, 200, 40), np.uint8)
    assert ot_tone.tone_table(red, green) is None, "not the same thing in the two pictures"
    assert ot_tone.tone_table(theirs[:500], ours[:500]) is None, "too little in common"


def test_a_channel_without_any_light_leaves_the_tone_table_whole() -> None:
    """Two pictures of a surface that has no green and no blue at all (a saturated red): those channels have no
    ratio between the pictures. They are left as they are, and the red is still matched."""
    theirs = np.tile(np.array([[200, 0, 0]], np.uint8), (3000, 1))
    ours = np.tile(np.array([[190, 0, 0]], np.uint8), (3000, 1))
    table = ot_tone.tone_table(theirs, ours)
    assert table is not None and np.isfinite(table).all()
    assert (table[-1, -1, -1, 1:] == 1.0).all(), "a colour no pair was near: green and blue as they are"
    assert ot_tone.recolour(np.array([[200, 0, 0]], np.uint8), table)[0].tolist() == [190, 0, 0]


def test_a_later_picture_goes_on_in_the_earlier_ones_tone_whatever_its_background(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The box's front and back painted in one colour, then the front right view in a lighter one. The front right
    camera sees the box's front too, slanted: where both pictures show that face the worker reads how they differ
    and brings the later picture to the earlier one's tone, so the side face - the front right picture's alone -
    comes out in the first pictures' colour (and in its own lighter one when the worker is told to leave pictures
    as they come). Only the model's surface counts: the picture's empty part white, black or the colour itself,
    the matching and the face are the same (the black around the views must have no
    effect)."""
    monkeypatch.setattr(ot_tone, "AGREE_PAIRS", 500)                # (the box's faces are small: 128 px of texture)
    first, later = (120, 130, 150), (140, 150, 165)
    seen = {}
    for label in ("solid", "white", "black", "apart"):
        work = tmp_path / label
        work.mkdir()
        prop = w.Prop(box(tmp_path / f"{label}.glb"), work, SIZE, 3)
        try:
            prop.agree = label != "apart"
            for k in (0, 1):
                prop.view(k)
                prop.project(k, solid(work / f"{k}.png", first))
            made = prop.view(2)
            model = np.asarray(Image.open(made["render"]).convert("L")) > 0
            picture = np.empty((SIZE, SIZE, 3), np.uint8)
            picture[:] = {"white": (255, 255, 255), "black": (0, 0, 0)}.get(label, later)
            picture[ot_masks.grow(model, 2)] = later                # (the model, and the pixels its edge texels read)
            Image.fromarray(picture).save(work / "2.png")
            prop.project(2, work / "2.png")
            side = prop.paint[centre_texels(prop, "+X")].astype(int)
            seen[label] = (side.copy(), prop.matched[prop.sheet_of[2]], prop.matched[prop.sheet_of[1]])
            assert (prop.paint[centre_texels(prop, "+Z")] == first).all(), "the front: the first picture's"
        finally:
            prop.close()
    side, pairs, back = seen["solid"]
    assert pairs > 500 and back == 0, "the back shares no surface with the front: left as it came"
    assert np.abs(side - first).max() <= 2, "the side face in the first pictures' tone"
    for label in ("white", "black"):
        assert seen[label][1] == pairs and np.array_equal(seen[label][0], side), label
    assert seen["apart"][1] == 0 and (seen["apart"][0] == later).all(), "told to leave them: its own colour"


def test_what_no_camera_painted_takes_the_paintings_tone_in_the_finished_model(tmp_path: Path) -> None:
    """The box painted from the front and the back alone, bluer and darker than its grey: its other faces keep the
    model's own texture - retinted like the painted ones in the finished GLB, unless the finish is told not to."""
    painted = (100, 110, 140)
    textures = {}
    for tone in (True, False):
        work = tmp_path / f"work{tone}"
        work.mkdir()
        two = w.Prop(box(tmp_path / f"box{tone}.glb"), work, SIZE, 2)
        try:
            for k in (0, 1):
                two.view(k)
                two.project(k, solid(work / f"{k}.png", painted))
            result = two.finish(tmp_path / f"out{tone}.glb", tone=tone)
            textures[tone], _ = ot_model.base_colour((tmp_path / f"out{tone}.glb").read_bytes())
            assert (result["stats"]["own"] is not None) == tone
            if tone:
                assert 0.5 < result["stats"]["own"] < 0.8, "four of the six faces are still the model's own"
        finally:
            two.close()
    for face in ("+Y", "-Y", "+X", "-X"):
        assert (centre(textures[False], face) == GREY).all(), face
        assert np.abs(centre(textures[True], face).astype(int) - painted).max() <= 2, face
    for face in ("+Z", "-Z"):
        assert (centre(textures[True], face) == painted).all() and (centre(textures[False], face) == painted).all()


def test_the_settings_test_may_show_the_models_normal_map(prop: w.Prop) -> None:
    """Picture 1 as a normal map seen by the camera: red towards its right, green up, blue towards it, on black.
    The box's front faces the front camera but for its 10-degree turn; its left side is seen at a slant."""
    view = prop.view(0, "normal")
    render = np.asarray(Image.open(view["render"]).convert("RGB")).reshape(-1, 3)
    colours, counts = np.unique(render, axis=0, return_counts=True)
    seen = {tuple(int(c) for c in colour): int(count) for colour, count in zip(colours, counts)}
    turn = np.degrees(TURN)
    front = tuple(round(127.5 + 127.5 * v) for v in (np.sin(np.radians(turn)), 0.0, np.cos(np.radians(turn))))
    side = tuple(round(127.5 + 127.5 * v) for v in (-np.cos(np.radians(turn)), 0.0, np.sin(np.radians(turn))))
    assert front == (150, 128, 253) and side == (2, 128, 150)
    ranked = sorted(seen, key=seen.get, reverse=True)[:3]
    assert set(ranked) == {(0, 0, 0), front, side}, ranked            # the background, the front, the side


def test_a_lens_puts_a_point_where_its_pixels_ray_goes() -> None:
    """Lens (the page's viewport camera): a point along the ray of pixel (row, column) lies at that pixel - whole
    numbers are pixel middles - at the depth along `forward`, and a surface is faced by how its normal meets the
    line to the eye, not the picture's axis."""
    forward = np.array([0.2, -0.1, -1.0])
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0.0, 1.0, 0.0))
    right /= np.linalg.norm(right)
    lens = ot_views.Lens(np.array([0.3, 0.2, 3.0]), forward, right, np.cross(right, forward), 40.0, 32, 32)
    rays = lens.rays().reshape(32, 32, 3)
    assert np.allclose(np.linalg.norm(rays, axis=2), 1.0)
    for row, column, far in ((0, 0, 2.0), (31, 5, 0.7), (16, 16, 5.0), (9, 30, 1.3)):
        point = lens.eye + rays[row, column] * far
        px, py, z = lens.place(point[None])
        assert np.allclose([px[0], py[0]], [column, row], atol=1e-9) and np.isclose(z[0], far * (rays[row, column] @ forward))
    off = lens.eye + rays[2, 29] * 2.0                        # far from the picture's middle: seen at a slant
    assert np.isclose(lens.facing(off[None], -rays[2, 29][None])[0], 1.0)
    assert lens.facing(off[None], -forward[None])[0] < 0.97
    assert np.isclose(lens.step(np.array([2.0]))[0], 2.0 / 40.0)


def lens_at_front(prop: w.Prop, mask_texels: np.ndarray, work: Path) -> tuple[dict, dict]:
    """A camera in front of the box, a little to the right and above, looking at its front face, and a mask of
    `mask_texels` (indices into the texture) -> (the look's arguments, the mask's path)."""
    mask = np.zeros(RES * RES, np.uint8)
    mask[mask_texels] = 255
    Image.fromarray(mask.reshape(RES, RES)).save(work / "mask.png")
    target = SPIN @ np.array((0.0, 0.0, 1.0))
    eye = SPIN @ np.array((0.9, 0.5, 5.0))
    return {"eye": eye.tolist(), "forward": (target - eye).tolist(), "up": [0.0, 1.0, 0.0], "tan": 0.36,
            "mask": work / "mask.png", "size": SIZE}, {}


def test_a_bent_lens_places_what_its_own_lines_of_sight_meet_and_faces_a_shapes_sides() -> None:
    """The page's "Inverse perspective" (BentLens): half the picture's side at depth z is wide + tan * z. Every
    pixel's line starts on the lens's own plane and a point on it is placed back in that pixel, at its depth; below
    tan 0 the lines meet behind the model, what is farther shows larger, a side that turns away from a camera
    faces this lens, and it sees no deeper than LENS_REACH of the way to that meeting point. tan 0: no
    perspective at all. A plain Lens when nothing is bent; none when the picture would span nothing."""
    eye, forward, up = [0.0, 0.0, 3.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]
    lens = ot_views.page_lens(eye, forward, up, -1.0, 64, wide=4.5)
    assert isinstance(lens, ot_views.BentLens) and lens.reach == pytest.approx(ot_views.LENS_REACH * 4.5)
    origins, rays = lens.origins(), lens.rays()
    cols, rows = (grid.reshape(-1) for grid in np.meshgrid(np.arange(64), np.arange(64)))
    for depth in (0.5, 2.0, 3.5):
        px, py, z = lens.place(origins + rays * (depth / (rays @ lens.forward))[:, None])
        assert np.allclose(px, cols, atol=1e-6) and np.allclose(py, rows, atol=1e-6) and np.allclose(z, depth)
    assert np.allclose(origins + rays * (4.5 / (rays @ lens.forward))[:, None], [0.0, 0.0, -1.5]), "where they meet"
    assert (lens.place(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 3.5]]))[2] == -1).all(), "too deep, and behind it"
    (near,), (far,) = lens.place(np.array([[1.0, 0.0, 2.0]]))[0], lens.place(np.array([[1.0, 0.0, 0.0]]))[0]
    assert far > near > 31.5, "the farther of two points beside the axis lies further out in the picture"
    assert lens.step(np.array([2.0]))[0] == pytest.approx((4.5 - 2.0) * 2 / 64)
    side, outwards = np.array([[1.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])      # a wall at x = 1, facing +x
    assert lens.facing(side, outwards)[0] == pytest.approx((1 / 1.5) / np.hypot(1.0, 1 / 1.5)) and lens.facing(side, outwards)[0] > ot_views.FACING
    plain = ot_views.page_lens(eye, forward, up, 0.5, 64)
    assert type(plain) is ot_views.Lens and plain.facing(side, outwards)[0] < 0, "a camera sees its back"
    assert plain.reach == np.inf and np.allclose(plain.origins(), eye)
    flat = ot_views.page_lens(eye, forward, up, 0.0, 64, wide=2.0)
    assert np.allclose(flat.rays(), forward) and flat.reach == np.inf
    assert flat.place(np.array([[1.0, 0.0, 2.0]]))[0] == flat.place(np.array([[1.0, 0.0, -2.0]]))[0]
    for tan, wide in ((0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), (float("nan"), 1.0)):
        assert ot_views.page_lens(eye, forward, up, tan, 64, wide=wide) is None
    # The inverse lens alone shows the deeper larger - than at its own plane -, and there a surface may be that
    # much more slanted (faced); through the other lenses, by its cosine alone.
    depths = np.array([0.0, 2.0, 3.5])
    assert lens.larger(depths) == pytest.approx([1.0, 1.8, 4.5])
    less = ot_views.page_lens(eye, forward, up, 0.2, 64, wide=1.0)
    assert plain.larger(depths) == 1.0 and (flat.larger(depths) == 1.0).all() and (less.larger(depths) == 1.0).all()
    cosines, deep = np.array([0.45, 0.3, 0.12, 0.1]), np.array([0.0, 2.0, 3.5, 3.5])
    assert ot_views.faced(lens, cosines, deep).tolist() == [False, True, True, False]
    for other in (plain, flat, less):
        assert not ot_views.faced(other, cosines, deep).any()
        assert ot_views.faced(other, np.array([ot_views.FACING]), np.array([3.5])).all()


def test_through_an_inverse_lens_a_repaint_reaches_a_side_no_camera_in_front_sees(tmp_path: Path) -> None:
    """A box looked at straight from the front, its right side masked: a camera sees nothing of that side; the
    bent lens, its lines of sight meeting just behind the box, does - the look shows it, and the picture goes
    onto its texels."""
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(box(tmp_path / "box.glb"), work, SIZE, 6, ahead=False)
    try:
        side = centre_texels(prop, "+X")
        mask = np.zeros(RES * RES, np.uint8)
        mask[side] = 255
        Image.fromarray(mask.reshape(RES, RES)).save(work / "mask.png")
        front = {"eye": (SPIN @ np.array((0.0, 0.0, 3.0))).tolist(), "forward": (SPIN @ np.array((0.0, 0.0, -1.0))).tolist(),
                 "up": [0.0, 1.0, 0.0], "mask": work / "mask.png", "size": SIZE}
        with pytest.raises(w.BakeError, match="in sight from here"):
            prop.look({**front, "tan": 0.5}, pictures=False)
        seen = prop.look({**front, "tan": -1.0, "wide": 4.5})
        assert 0 < seen["seen"] <= seen["masked"] == len(side)
        shown = np.asarray(Image.open(seen["viewport"]).convert("RGB")).reshape(-1, 3)
        assert (shown == ot_repaint.MASK_COLOUR).all(axis=1).any(), "the masked side shows in the look's picture"
        assert prop.repaint(solid(work / "answer.png", (40, 160, 90)))["painted"] > 0
        assert (prop.paint[side] == (40, 160, 90)).all(axis=1).any(), "and takes the picture"
    finally:
        prop.close()


def through_an_inverse_lens(tmp_path: Path, size: tuple[float, float, float], masked: str | None,
                            tan: float = -1.0) -> tuple[w.Prop, dict]:
    """A box of half sides `size` looked at straight from the front, 3 away from its middle, through an inverse
    lens (wide 4.5), a face's middle masked (or every texel) -> (the prop: close it; the look's answer)."""
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(box(tmp_path / "box.glb", size), work, SIZE, 6, ahead=False)
    mask = np.zeros(RES * RES, np.uint8)
    mask[centre_texels(prop, masked) if masked else slice(None)] = 255
    Image.fromarray(mask.reshape(RES, RES)).save(work / "mask.png")
    return prop, prop.look({"eye": (SPIN @ np.array((0.0, 0.0, 3.0))).tolist(), "forward": (SPIN @ np.array((0.0, 0.0, -1.0))).tolist(),
                            "up": [0.0, 1.0, 0.0], "tan": tan, "wide": 4.5, "mask": work / "mask.png", "size": SIZE},
                           pictures=False)


@pytest.mark.parametrize("half_width, least", [(1.0, 0.4), (0.5, 0.2)])
def test_what_an_inverse_lens_shows_larger_takes_its_picture_at_more_of_a_slant(tmp_path: Path, half_width: float,
                                                                                least: float) -> None:
    """The box's right side through the inverse lens, its middle masked: the lens shows the side larger the deeper
    it runs, and that makes up for its slant (faced). So the near end of the masked part, which the lens faces at
    less than FACING, takes the picture too: nearly all that is masked does (by the cosine alone, some six tenths
    of it). The narrow box's side is seen at half the cosine: from pixel to pixel it runs four pixels' sides
    deeper, which the test that nothing hides a texel has to allow (Prop.targets), or half of them go unpainted."""
    prop, seen = through_an_inverse_lens(tmp_path, (half_width, 1.0, 1.0), "+X")
    try:
        look = prop.repaints.at
        mine = look.chosen[look.masked[look.chosen]]
        cosine = look.lens.facing(prop.point[mine], prop.normal[mine])
        depth = look.lens.place(prop.point[mine])[2]
        assert depth.min() < 2.55 and depth.max() > 3.3, "the masked middle of the side, some 2.5 to 3.5 deep"
        slanted = cosine < ot_views.FACING
        assert slanted.any() and depth[slanted].min() < 2.55, "its near end is slanted"
        assert (cosine * look.lens.larger(depth) >= ot_views.FACING).all() and least < cosine.min() < least + 0.05
        assert seen["seen"] == len(mine) > 0.9 * seen["masked"] and (~slanted).sum() < 0.7 * seen["masked"]
    finally:
        prop.close()


def test_an_inverse_lens_sees_no_deeper_than_its_reach_and_paints_nothing_beside_the_models_outline(
        tmp_path: Path) -> None:
    """A thin box whose sides run on beyond the lens's reach, all of it masked: the look's picture ends where the
    reach cuts the box, though the rays would meet it deeper. That cut is an outline the surface is seen squarely
    at, and beside every outline a slanted surface takes the picture too: no texel reads a pixel within SNAP_REACH
    of the model's outline, where Qwen-Image's painting may be the background's."""
    prop, _ = through_an_inverse_lens(tmp_path, (0.3, 0.3, 1.0), None, tan=-1.4)
    try:
        look = prop.repaints.at
        lens = look.lens
        rays = lens.rays()
        met = ot_views.cast(prop.scene, prop.mesh.faces, lens.origins(), rays)        # however deep
        depth = np.full(SIZE * SIZE, np.inf)
        depth[met.hit] = met.t * (rays @ lens.forward)[met.hit]
        assert 2.0 < lens.reach < 3.0 and (look.model.reshape(-1) == (depth <= lens.reach)).all()
        assert (np.isfinite(depth) & (depth > lens.reach)).sum() > 100, "the box runs on behind the cut"
        away = ot_masks.shrink(look.model, ot_masks.SNAP_REACH + 1)
        col, row = w.nearest(look.px, look.py, SIZE, SIZE)
        assert len(look.chosen) > 100 and away[row, col].all()
        # (not for want of texels there: the lens faces the surface beside the cut well enough)
        px, py, z = lens.place(prop.point)
        at = np.flatnonzero((z > 0) & (px >= 4) & (px <= SIZE - 5) & (py >= 4) & (py <= SIZE - 5))
        col, row = w.nearest(px[at], py[at], SIZE, SIZE)
        beside = look.model[row, col] & ~away[row, col]
        assert (beside & ot_views.faced(lens, lens.facing(prop.point[at], prop.normal[at]), z[at])).sum() > 20
    finally:
        prop.close()


def test_a_repaint_through_a_bent_lens_is_laid_again_through_that_lens(tmp_path: Path) -> None:
    """ "fades" reads a repaint's lens as "look" does. A lens turned down but not inside out (tan above 0 with
    "wide") is the one that would go wrong in silence - without its "wide" it is a narrow camera, no error: laid
    again, the repaint goes onto the texels it was painted on, and the same camera and files without "wide", or
    with another, are other repaints kept."""
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(box(tmp_path / "box.glb"), work, SIZE, 6, ahead=False)
    try:
        ask, _ = lens_at_front(prop, centre_texels(prop, "+Z"), work)
        bent = {**ask, "mask": str(ask["mask"]), "tan": 0.1, "wide": 1.1}
        prop.look(bent, pictures=False)
        painted = prop.repaint(solid(work / "projected.png", (40, 160, 90)))["painted"]
        again = {**bent, "picture": str(work / "projected.png"), "fade": 1.0}
        assert prop.fades([again], [None])["painted"] == [painted]
        narrow = {key: value for key, value in again.items() if key != "wide"}
        assert 0 < prop.fades([narrow], [None])["painted"][0] < painted and len(prop.repaints.kept) == 2
        assert prop.fades([{**again, "wide": 1.2}], [None])["painted"][0] > 0 and len(prop.repaints.kept) == 3
    finally:
        prop.close()


def test_what_is_masked_is_painted_again_through_the_pages_camera_and_nothing_else(tmp_path: Path) -> None:
    """A repaint: the middle of the box's front face is masked; the look gives Qwen-Image the
    depth map from the page's camera and the model as it is textured with the masked surface in magenta - no
    picture of the model without the mask, from here or from elsewhere: Qwen-Image painted back what it showed
    under the mask. Its picture
    then goes onto the masked texels alone - the rest of the front face, and every other face, keep their bytes -
    with the mask's edge blended in 3D (no texel far from it changes, those at it are a mix)."""
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(box(tmp_path / "box.glb"), work, SIZE, 6, ahead=False)
    try:
        assert prop.plan is None and not prop._sights, "nothing is looked through ahead"
        masked = centre_texels(prop, "+Z")
        ask, _ = lens_at_front(prop, masked, work)
        made = prop.look(ask)
        depth = np.asarray(Image.open(made["depth"]).convert("L"))
        seen = np.asarray(Image.open(made["viewport"]).convert("RGB"))
        model = depth > 0
        assert 0.2 < model.mean() < 0.9 and depth[model].max() == 255
        assert depth[model].min() == round(255 * ot_views.DEPTH_FAR)
        magenta = (seen == ot_repaint.MASK_COLOUR).all(axis=2)
        rows, cols = np.nonzero(magenta)
        assert magenta.sum() > 200 and magenta[SIZE // 2, SIZE // 2] and not magenta[~model].any()
        assert abs(rows.mean() - SIZE / 2) < 12 and abs(cols.mean() - SIZE / 2) < 16, "the mask is the face's middle"
        assert (seen[model & ~magenta] == GREY).all(), "the rest as the model is textured now"
        assert set(made) == {"depth", "viewport", "redrawn", "masked", "seen"} and not (work / "look_example.png").exists()
        # What the image generator draws anew, the rest held to the view: the masked pixels and a few around them,
        # white on black - on the model only.
        redrawn = np.asarray(Image.open(made["redrawn"]).convert("L"))
        assert set(np.unique(redrawn)) == {0, 255} and (redrawn[magenta] == 255).all() and not redrawn[~model].any()
        room = ot_repaint.HELD_ROOM
        grown = np.zeros_like(magenta)
        for dy in range(-room, room + 1):
            for dx in range(-room, room + 1):
                grown |= np.roll(magenta, (dy, dx), axis=(0, 1))
        assert ((redrawn == 255) == (grown & model)).all() and (redrawn == 255).sum() > magenta.sum()
        assert made["masked"] == len(masked) and 0.9 * len(masked) <= made["seen"] <= len(masked)
        # Qwen-Image's picture: one colour all over.
        colour = (40, 160, 90)
        before = prop.paint.copy()
        done = prop.repaint(solid(work / "answer.png", colour), keep=work / "projected.png")
        assert done == {"painted": done["painted"]} and done["painted"] >= made["seen"]
        inside = np.zeros(RES * RES, dtype=bool)
        inside[masked] = True
        changed = (prop.paint != before).any(axis=1) & prop.inside
        whole = (prop.paint[masked] == colour).all(axis=1)
        middle = masked[len(masked) // 2 + int(np.sqrt(len(masked))) // 2]      # (the mask is a block of texels)
        assert whole.mean() > 0.3 and tuple(prop.paint[middle]) == colour, "the mask's own texels: the picture"
        rim = changed & ~inside
        assert 0 < rim.sum() < len(masked), "a blended edge around the mask, in 3D"
        mixed = prop.paint[rim].astype(int)
        assert ((mixed[:, 1] > GREY[1]) & (mixed[:, 1] < colour[1])).all(), "between the two"
        for face in ("-Z", "+X", "-X", "+Y", "-Y"):
            assert (prop.paint[centre_texels(prop, face)] == GREY).all(), face
        u0, _v0, _u1, v1 = island(list(FACES).index("+Z"))
        corner = int((1 - v1 + 0.02) * RES) * RES + int((u0 + 0.02) * RES)       # the front face's corner: far from the mask
        assert tuple(prop.paint[corner]) == GREY
        result = prop.finish(tmp_path / "out.glb", tone=False)
        texture, _ = ot_model.base_colour((tmp_path / "out.glb").read_bytes())
        assert (centre(texture, "+Z") == colour).all(axis=1).mean() > 0.3 and (centre(texture, "+X") == GREY).all()
        assert result["stats"]["texture"] == RES
    finally:
        prop.close()


def test_a_repaint_goes_on_as_it_was_painted_whatever_its_tone(tmp_path: Path) -> None:
    """Render Masked on a pale model: the picture comes back much brighter than the texture around the mask, with
    a pane of black glass darker than anything on the model. It goes on as it was painted - no lighting taken
    off, no tone matched to the texture: the masked texels take the picture's own colours, and what was kept of
    it is the picture to the byte."""
    work = tmp_path / "work"
    work.mkdir()
    res = size = 512
    prop = w.Prop(shielded_wall(tmp_path / "wall.glb", 0.86, 0.96, 0.86, 0.96, texture=res), work, size, 3,
                  ahead=False)
    try:
        x, y = prop.point[:, 0], prop.point[:, 1]
        on_wall = prop.point[:, 2] < 0.3
        prop.paint[prop.texel] = np.clip(40.0 + 60.0 * (x[:, None] + 1.0), 0, 255).astype(np.uint8)   # a grey ramp
        prop.resettle()
        masked = on_wall & (np.abs(x) < 0.4) & (np.abs(y) < 0.4)
        mask = np.zeros(res * res, np.uint8)
        mask[prop.texel[masked]] = 255
        Image.fromarray(mask.reshape(res, res)).save(work / "mask.png")
        made = prop.look({"eye": [0.0, 0.0, 4.0], "forward": [0.0, 0.0, -1.0], "up": [0.0, 1.0, 0.0], "tan": 0.3,
                          "mask": work / "mask.png", "size": size})
        view = np.asarray(Image.open(made["viewport"]).convert("RGB")).astype(int)
        model = np.asarray(Image.open(made["depth"]).convert("L")) > 0
        picture = np.clip(view * 1.6, 0, 255).astype(np.uint8)          # the view, far brighter
        middle = np.zeros((size, size), dtype=bool)
        middle[size // 2 - 40:size // 2 + 40, size // 2 - 40:size // 2 + 40] = True
        picture[middle] = (5, 5, 6)                                     # the glass
        picture[~model] = 0
        Image.fromarray(picture).save(work / "answer.png")
        done = prop.repaint(work / "answer.png", keep=work / "projected.png")
        assert done["painted"] > 1000
        assert (np.asarray(Image.open(work / "projected.png").convert("RGB")) == picture).all(), "kept as painted"
        glass = masked & (np.abs(x) < 0.08) & (np.abs(y) < 0.08)
        painted = prop.paint[prop.texel].astype(int)
        assert glass.sum() > 100 and (np.abs(painted[glass] - (5, 5, 6)).max(axis=1) <= 1).all(), "as black as painted"
        surface = masked & (np.abs(x) > 0.25) & (np.abs(x) < 0.33) & (np.abs(y) < 0.33)
        own = np.clip(40.0 + 60.0 * (x[surface] + 1.0), 0, 255)
        assert surface.sum() > 300 and (painted[surface, 0] - own).mean() > 0.4 * own.mean(), "as bright as painted"
    finally:
        prop.close()


def test_a_repaints_edge_fades_out_where_the_camera_sees_on_and_in_where_it_does_not(tmp_path: Path) -> None:
    """The page's fade slider on a repaint: the repaint's kept picture is put on again - no pictures made for
    Qwen-Image, no tone read again - with its edge faded `fade` times as far. Where the camera can paint beyond the
    mask the picture fades outwards and what was masked keeps it; where the mask meets the end of what the camera
    can paint (here the shadow of a shield in front of the wall; on the Boy, the neck turning away from a camera
    under the chin) there is nothing beyond, and the repaint thins out inwards, over a width that grows with the
    fade. Its layer shows its share alone over the page's clay grey. (The wall's texture is 512 px: in the box's
    128 px a seam's width is under the blend's two texels, and every fade looks alike.)"""
    work = tmp_path / "work"
    work.mkdir()
    colour = (40, 160, 90)
    picture = solid(work / "kept.png", colour)
    wall = shielded_wall(tmp_path / "wall.glb", -0.3, 0.3, -0.3, 0.3, texture=512)
    shadow = 0.3 * 4.0 / 3.4                                    # the shield's shadow on the wall, seen from z = 4

    def painted(fade: float, patch: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """-> per texel the picture's share, whether it was masked, its distance from the shadow (reaches), the
        shield's texels."""
        prop = w.Prop(wall, work, 512, 3, ahead=False)
        try:
            x, y = prop.point[:, 0], prop.point[:, 1]
            on_wall = prop.point[:, 2] < 0.3
            masked = on_wall & ((x > 0.45) & (x < 0.85) & (np.abs(y) < 0.4) if patch else True)
            mask = np.zeros(prop.res * prop.res, np.uint8)
            mask[prop.texel[masked]] = 255
            Image.fromarray(mask.reshape(prop.res, prop.res)).save(work / "mask.png")
            seen = prop.look({"eye": [0.0, 0.0, 4.0], "forward": [0.0, 0.0, -1.0], "up": [0.0, 1.0, 0.0], "tan": 0.3,
                              "mask": work / "mask.png", "size": 512}, pictures=False)
            assert set(seen) == {"masked", "seen"} and not list(work.glob("look_*.png"))
            prop.paint[:] = GREY
            prop.repaint(picture, layer=work / f"layer{fade}.webp", fade=fade)
            share = (prop.paint[prop.texel, 1].astype(float) - GREY[1]) / (colour[1] - GREY[1])
            apart = np.hypot(np.maximum(np.abs(x) - shadow, 0.0), np.maximum(np.abs(y) - shadow, 0.0))
            apart = apart / prop.seam_reach()
            return share, masked, apart, ~on_wall
        finally:
            prop.close()

    # A patch of the wall in the open: further out, and what was masked keeps the picture.
    near, masked, _, shield = painted(1.0, patch=True)
    far, _, _, _ = painted(4.0, patch=True)
    around = ~masked & ~shield
    assert (far[around] > 0.02).sum() > 2 * (near[around] > 0.02).sum(), "further out"
    assert far[masked].mean() > near[masked].mean() and far[masked].min() > 0.45 and (far[masked] > 0.95).mean() > 0.9
    assert (far[shield] == 0).all() and (far <= 1.0).all()
    layer = np.asarray(Image.open(work / "layer4.0.webp").convert("RGB")).reshape(-1, 3).astype(int)
    prop = w.Prop(wall, work, 512, 3, ahead=False)
    texel = prop.texel
    prop.close()
    assert np.abs(layer[texel[masked]] - colour).max(axis=1).mean() < 12, "the picture where it went on whole"
    assert np.abs(layer[texel[shield]] - ot_tone.LAYER_GREY).max(axis=1).mean() < 12, "clay where it did not"
    # The whole wall masked: beside the shadow the camera sees no further, and the repaint thins out inwards, the
    # further the longer the fade; away from it the wall is repainted whole whatever the fade.
    shares = {fade: painted(fade, patch=False) for fade in (1.0, 4.0)}
    _, wall_masked, apart, _ = shares[1.0]
    ring = wall_masked & (apart > 2) & (apart < 6)
    clear = wall_masked & (apart > 20) & (np.abs(prop.point[:, 0]) < 0.9) & (np.abs(prop.point[:, 1]) < 0.9)
    assert shares[1.0][0][ring].mean() > 0.9 and shares[4.0][0][ring].mean() < 0.75, "inwards, wider"
    assert shares[1.0][0][clear].min() > 0.98 and shares[4.0][0][clear].min() > 0.98


def test_fades_lay_kept_repaints_on_again_as_a_repaint_would_and_work_the_other_fades_out_ahead(
        tmp_path: Path) -> None:
    """The edge fade's command: the repaints' kept pictures onto the model as loaded, each as "look" + "repaint"
    (tone off) with its fade would put it - byte for byte -, the model after each written where asked. Each call
    starts again from the model as loaded, so a fade changed and changed back gives the first texture again; the
    slider's other fades of the repaint asked are worked out on a thread meanwhile."""
    work = tmp_path / "work"
    work.mkdir()
    picture = solid(work / "projected.png", (40, 160, 90))
    glb_path = box(tmp_path / "box.glb")
    prop = w.Prop(glb_path, work, SIZE, 6, ahead=False)
    try:
        ask, _ = lens_at_front(prop, centre_texels(prop, "+Z"), work)
        prop.look(ask, pictures=False)
        prop.repaint(picture, fade=2.0)
        expected = prop.paint.copy()
    finally:
        prop.close()
    asked = [{**{key: ask[key] for key in ("eye", "forward", "up", "tan", "size")}, "mask": str(ask["mask"]),
              "picture": str(picture), "fade": 2.0}]
    prop = w.Prop(glb_path, work, SIZE, 6, ahead=False)
    try:
        made = prop.fades(asked, [str(tmp_path / "two.glb")], ahead=0)
        assert made["painted"][0] > 0 and (prop.paint == expected).all(), "as look + repaint put it"
        texture, _ = ot_model.base_colour((tmp_path / "two.glb").read_bytes())
        assert (texture.reshape(-1, 3)[prop.inside] == prop.paint[prop.inside]).all(), "the model written"
        prop.fades([{**asked[0], "fade": 4.0}], [None])
        assert not (prop.paint == expected).all(), "another fade"
        prop.fades(asked, [None])
        assert (prop.paint == expected).all(), "from the model as loaded each time"
        (kept,) = prop.repaints.kept.values()
        assert set(kept["fades"]) < set(ot_repaint.FADE_STEPS), "a call without `ahead` stops the thread"
        prop.fades(asked, None, ahead=0)                 # (picked again: it starts again from where it was)
        fades = kept["fades"]
        deadline = time.time() + 30
        while not all(fade in fades and fades[fade].done() for fade in ot_repaint.FADE_STEPS) \
                and time.time() < deadline:
            time.sleep(0.05)
        assert set(fades) == set(ot_repaint.FADE_STEPS), "every fade of the slider worked out ahead"
        assert all(done.result()[0].size for done in fades.values())
    finally:
        prop.close()


def paint_layer(path: Path, texels: np.ndarray, colour: tuple[int, int, int]) -> Path:
    """A paint layer of the box's texture: `colour` at the `texels`, nothing painted elsewhere."""
    layer = np.zeros((RES * RES, 4), np.uint8)
    layer[texels] = (*colour, 255)
    Image.fromarray(layer.reshape(RES, RES, 4), "RGBA").save(path)
    return path


def test_a_hand_painting_goes_over_the_step_as_made_and_a_later_one_over_it(tmp_path: Path) -> None:
    """The worker's "hand_paint" (the page's marker and clone stamp, saved): the patch's changed texels go into the
    step's paint layer, a later patch over an earlier one, and the step's model becomes the model as made with the
    layer over its texture, its alpha the model's own. The painted model is never read (its WebP would lose a
    generation a save): whatever it holds, a save gives the model as made plus the layer. The model as made is
    the painted one's hard link at first; the painted one written under another name and moved over it leaves it
    as it was (NTFS keeps the link's other name)."""
    step = tmp_path / "ot_002"
    step.mkdir()
    made, out, layer = step / "unpainted.glb", step / "omnitexture.glb", step / "hand-paint.png"
    data = box(tmp_path / "box.glb").read_bytes()
    texture = np.full((RES, RES, 4), (*GREY, 255), np.uint8)
    texture[:8] = (*GREY, 100)                                   # some alpha of its own
    buffer = io.BytesIO()
    Image.fromarray(texture, "RGBA").save(buffer, "PNG")
    made.write_bytes(glb.set_base_color_image(data, buffer.getvalue()))
    os.link(made, out)                                           # as the hub keeps the model as made
    original = made.read_bytes()

    def save(texels: np.ndarray, colour: tuple[int, int, int], stray: tuple[int, int] | None = None) -> None:
        patch = np.zeros((RES * RES, 4), np.uint8)
        patch[texels] = (*colour, 255)
        if stray is not None:
            patch[stray[0] * RES + stray[1]] = (1, 2, 3, 0)    # alpha 0: not changed, whatever its colour
        (tmp_path / "patch.rgba").write_bytes(patch.tobytes())
        answer = w.paint_by_hand(made, layer, tmp_path / "patch.rgba", RES, RES, out)
        assert answer["texture"] == [RES, RES]

    first, second = np.arange(10 * RES, 10 * RES + 40), np.arange(10 * RES + 20, 10 * RES + 60)
    save(first, (200, 30, 40), stray=(70, 70))
    assert made.read_bytes() == original and os.stat(made).st_nlink == 1, "the model as made, untouched"
    # A model holding something else entirely: the next save starts from the model as made all the same.
    black = io.BytesIO()
    Image.new("RGB", (RES, RES), (0, 0, 0)).save(black, "PNG")
    out.write_bytes(glb.set_base_color_image(original, black.getvalue()))
    save(second, (5, 250, 9))
    held = np.asarray(Image.open(layer)).reshape(-1, 4)
    assert (held[first[:20], :3] == (200, 30, 40)).all() and (held[second, :3] == (5, 250, 9)).all()
    assert np.count_nonzero(held[:, 3]) == 60 and (held[held[:, 3] > 0, 3] == 255).all()
    painted, kind = ot_model.base_colour(out.read_bytes())
    want = texture.copy().reshape(-1, 4)
    want[held[:, 3] > 0, :3] = held[held[:, 3] > 0, :3]
    assert kind == "png" and (painted.reshape(-1, 4) == want).all(), "the model as made with the layer, alpha its own"
    assert made.read_bytes() == original
    with pytest.raises(BakeError, match="pixels"):
        w.paint_by_hand(made, layer, tmp_path / "patch.rgba", RES * 2, RES * 2, out)


def test_a_steps_hand_paint_goes_on_again_after_its_repaint_and_stays_under_the_next(tmp_path: Path) -> None:
    """Put on again ("fades": a fade, or the repaints after a picture taken off), a step's hand paint goes over
    its repaint: the step's model holds it, its model as made does not, and the next repaint goes onto the
    painted texture. A paint layer changed since is read again; the result's "finish" lays one on too."""
    prop, asked = two_repaints(tmp_path)
    try:
        side = centre_texels(prop, "+X")
        layer = paint_layer(tmp_path / "hand-paint.png", side, (250, 5, 5))
        asked[0] = {**asked[0], "hand_paint": str(layer), "unpainted": str(tmp_path / "one-made.glb")}
        outs = [str(tmp_path / "one.glb"), str(tmp_path / "two.glb")]
        prop.fades(asked, outs)

        def at(path: str, texels: np.ndarray) -> np.ndarray:
            return ot_model.base_colour(Path(path).read_bytes())[0].reshape(-1, 3)[texels]

        assert (at(outs[0], side) == (250, 5, 5)).all() and (at(outs[1], side) == (250, 5, 5)).all()
        assert (at(str(tmp_path / "one-made.glb"), side) == GREY).all(), "the model as made: no hand paint"
        front = centre_texels(prop, "+Z")
        assert (at(outs[1], front[[len(front) // 2]]) == (40, 160, 90)).all(), "the next repaint on top"
        time.sleep(0.02)
        paint_layer(layer, side, (9, 9, 250))
        prop.fades(asked, [None, outs[1]])
        assert (at(outs[1], side) == (9, 9, 250)).all(), "a layer changed since is read again"
    finally:
        prop.close()
    one = w.Prop(box(tmp_path / "plain.glb"), tmp_path / "work", SIZE, 1)
    try:
        one.project(0, solid(tmp_path / "front.png", (40, 160, 90)))
        one.finish(tmp_path / "result.glb", tone=False, hand_paint=layer, unpainted=tmp_path / "result-made.glb")
    finally:
        one.close()
    assert (ot_model.base_colour((tmp_path / "result.glb").read_bytes())[0].reshape(-1, 3)[side] == (9, 9, 250)).all()
    assert (ot_model.base_colour((tmp_path / "result-made.glb").read_bytes())[0].reshape(-1, 3)[side] == GREY).all()


def two_repaints(tmp_path: Path) -> tuple[w.Prop, list[dict]]:
    """A box and two repaints of its front (what "fades" takes): one mask and one picture, the second through a
    slightly wider lens."""
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(box(tmp_path / "box.glb"), work, SIZE, 6, ahead=False)
    ask, _ = lens_at_front(prop, centre_texels(prop, "+Z"), work)
    asked = {**{key: ask[key] for key in ("eye", "forward", "up", "tan", "size")}, "mask": str(ask["mask"]),
             "picture": str(solid(work / "projected.png", (40, 160, 90))), "fade": 2.0}
    return prop, [asked, {**asked, "tan": asked["tan"] * 1.05}]


def test_a_hidden_repaint_puts_nothing_on_and_keeps_its_place_and_its_steps_hand_paint(tmp_path: Path) -> None:
    """The page's eye on a repaint: asked "hidden", it puts nothing onto the model - its step is the model before
    it with what was painted by hand on that step - and the repaints after it go on as ever. Its layer is still
    written where asked (the page shows where it would lie). Shown again, it is on."""
    prop, asked = two_repaints(tmp_path)
    try:
        front, side = centre_texels(prop, "+Z"), centre_texels(prop, "+X")
        middle = front[[len(front) // 2]]
        layer = paint_layer(tmp_path / "hand-paint.png", side, (250, 5, 5))
        outs = [str(tmp_path / "one.glb"), str(tmp_path / "two.glb")]
        hidden = {**asked[0], "hidden": True, "hand_paint": str(layer), "unpainted": str(tmp_path / "one-made.glb"),
                  "layer": str(tmp_path / "one.webp")}

        def at(path: str, texels: np.ndarray) -> np.ndarray:
            return ot_model.base_colour(Path(path).read_bytes())[0].reshape(-1, 3)[texels]

        made = prop.fades([hidden, asked[1]], outs)
        assert made["painted"][0] == 0 and made["painted"][1] > 0
        assert (at(outs[0], middle) == GREY).all(), "nothing of it on its own step"
        assert (at(outs[0], side) == (250, 5, 5)).all() and (at(str(tmp_path / "one-made.glb"), side) == GREY).all()
        assert (at(outs[1], middle) == (40, 160, 90)).all() and (at(outs[1], side) == (250, 5, 5)).all()
        assert (tmp_path / "one.webp").is_file(), "where it would lie"
        # The last one hidden: its step is the one before it.
        prop.fades([asked[0], {**asked[1], "hidden": True}], outs)
        assert (at(outs[1], front) == at(outs[0], front)).all()
        prop.fades([{**hidden, "hidden": False}, asked[1]], outs)
        assert (at(outs[0], middle) == (40, 160, 90)).all(), "shown again"
        # Hidden and only made ready (the page picked it): nothing is worked out for it.
        prop.repaints.kept.clear()
        assert prop.fades([hidden], None)["painted"] == [0] and not prop.repaints.kept
    finally:
        prop.close()


def test_a_kept_repaint_is_found_wherever_it_stands_and_looked_at_anew_when_its_files_changed(
        tmp_path: Path) -> None:
    """What "fades" keeps of a repaint goes by its camera and its mask's and picture's files, not by its place
    among those asked: with the one before it taken off or hidden, nothing is worked out again - also when the
    hub names the same files in another folder (linked or copied there). A mask written anew is looked at anew."""
    prop, asked = two_repaints(tmp_path)
    looks: list[Path] = []
    look = prop.repaints.look
    prop.repaints.look = lambda lens, mask, pictures=True: looks.append(mask) or look(lens, mask, pictures)
    try:
        expected = prop.fades(asked, [None, None])["painted"]
        assert len(looks) == 2 and len(prop.repaints.kept) == 2
        assert prop.fades(asked[1:], [None])["painted"] == expected[1:], "the one before it taken off"
        assert prop.fades(asked[::-1], [None, None])["painted"] == expected[::-1]
        moved = tmp_path / "moved"
        moved.mkdir()
        os.link(asked[0]["mask"], moved / "mask.png")
        shutil.copy2(asked[0]["picture"], moved / "projected.png")
        elsewhere = {**asked[0], "mask": str(moved / "mask.png"), "picture": str(moved / "projected.png")}
        assert prop.fades([elsewhere], [None])["painted"] == expected[:1]
        assert len(looks) == 2 and len(prop.repaints.kept) == 2, "nothing looked at or worked out again"
        time.sleep(0.02)
        front = centre_texels(prop, "+Z")
        lens_at_front(prop, front[:len(front) // 4], tmp_path / "work")               # the mask, written anew
        assert prop.fades(asked[:1], [None])["painted"][0] < expected[0]
        assert len(looks) == 3 and len(prop.repaints.kept) == 3
        # No more is kept than the repaints asked and a few of those asked before (one taken off may come back).
        for tan in (0.31, 0.32, 0.33, 0.34):
            prop.fades([{**asked[0], "tan": tan}], [None])
        assert len(prop.repaints.kept) == 1 + ot_repaint.KEPT_SPARE and len(looks) == 7
        assert prop.fades([{**asked[0], "tan": 0.33}], [None]) and len(looks) == 7, "the one before the last is still kept"
    finally:
        prop.close()


def test_another_model_of_the_prop_takes_the_loaded_ones_place_and_the_repaints_go_onto_it(tmp_path: Path) -> None:
    """The worker's "texture" (the painted views hidden, or shown again): the same surface with another texture.
    "fades" then puts the repaints onto that one and writes its file's models; nothing kept of the repaints is
    worked out again. A model with a texture of another size is refused."""
    prop, asked = two_repaints(tmp_path)
    looks: list[Path] = []
    look = prop.repaints.look
    prop.repaints.look = lambda lens, mask, pictures=True: looks.append(mask) or look(lens, mask, pictures)
    try:
        front, side = centre_texels(prop, "+Z"), centre_texels(prop, "+X")
        prop.fades(asked[:1], [str(tmp_path / "grey.glb")])
        blue = io.BytesIO()
        Image.new("RGB", (RES, RES), (10, 20, 200)).save(blue, "PNG")
        other = tmp_path / "blue.glb"
        other.write_bytes(glb.set_base_color_image((tmp_path / "box.glb").read_bytes(), blue.getvalue()))
        assert w.COMMANDS["texture"](prop, {"glb": str(other)}) == {"texture": [RES, RES]}
        prop.fades(asked[:1], [str(tmp_path / "on-blue.glb")])
        painted = ot_model.base_colour((tmp_path / "on-blue.glb").read_bytes())[0].reshape(-1, 3)
        assert (painted[side] == (10, 20, 200)).all() and (painted[front[[len(front) // 2]]] == (40, 160, 90)).all()
        assert len(looks) == 1, "the repaint's look is the surface's: kept"
        small = io.BytesIO()
        Image.new("RGB", (RES // 2, RES // 2), GREY).save(small, "PNG")
        other.write_bytes(glb.set_base_color_image((tmp_path / "box.glb").read_bytes(), small.getvalue()))
        with pytest.raises(BakeError, match="pixels"):
            prop.retexture(other)
    finally:
        prop.close()


def test_a_model_written_quickly_holds_every_pixel_as_given(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model the artist waits for (a repaint put on again) is written without loss, in its texture's own
    format: every pixel as given, also under the texture's see-through texels, whose alpha stays. Written at
    leisure, a WebP texture is the smaller, lossy one."""
    monkeypatch.setattr(glb, "set_base_color_image", lambda data, image: image)       # (the texture as stored)
    rng = np.random.default_rng(3)
    texture = np.dstack([rng.integers(0, 256, (RES, RES, 3), np.uint8), np.full((RES, RES, 1), 255, np.uint8)])
    texture[:8, :, 3] = 0
    pixels = np.repeat(np.repeat(rng.integers(0, 256, (RES // 8, RES // 8, 3), np.uint8), 8, axis=0), 8, axis=1)
    given = np.dstack([pixels, texture[..., 3:]])

    def stored(kind: str, quick: bool) -> tuple[np.ndarray, str, int]:
        data = w.model_with(b"", kind, texture, pixels, quick)
        picture = Image.open(io.BytesIO(data))
        return np.asarray(picture.convert("RGBA")), picture.format.lower(), len(data)

    for kind in ("webp", "png"):
        held, kept_as, _ = stored(kind, True)
        assert kept_as == kind and (held == given).all()
    lossy, kept_as, size = stored("webp", False)
    assert kept_as == "webp" and size < stored("webp", True)[2] and not (lossy == given).all()


def test_a_steps_hand_paint_goes_onto_another_model_of_the_prop(tmp_path: Path) -> None:
    """The worker's "hand_paint" without a patch (the painted views hidden, or shown again): the paint layer as it
    is over another model's texture, that model and the layer left as they are."""
    made, out = box(tmp_path / "box.glb"), tmp_path / "painted.glb"
    texels = np.arange(10 * RES, 10 * RES + 40)
    layer = paint_layer(tmp_path / "hand-paint.png", texels, (200, 30, 40))
    before = made.read_bytes(), layer.read_bytes()
    answer = w.WITHOUT_LOAD["hand_paint"]({"model": str(made), "layer": str(layer), "out": str(out)})
    assert answer == {"texture": [RES, RES]} and (made.read_bytes(), layer.read_bytes()) == before
    painted = ot_model.base_colour(out.read_bytes())[0].reshape(-1, 3)
    assert (painted[texels] == (200, 30, 40)).all() and (np.delete(painted, texels, axis=0) == GREY).all()


def test_working_ahead_stops_for_good_when_another_repaint_is_picked(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The thread works the picked repaint's other fades out; another repaint picked, it lets go after the fade
    it is at and the new one's are worked out - not the first one's to the end before them."""
    entered, go = threading.Event(), threading.Event()
    weight = ot_repaint.Repaints.weight

    def slow(self: ot_repaint.Repaints, look: ot_repaint.Look, fade: float) -> np.ndarray:
        if threading.current_thread() is not threading.main_thread() and not entered.is_set():
            entered.set()
            go.wait(30)
        return weight(self, look, fade)

    monkeypatch.setattr(ot_repaint.Repaints, "weight", slow)
    prop, asked = two_repaints(tmp_path)
    try:
        prop.fades(asked, None, ahead=0)
        assert entered.wait(30), "the first repaint's other fades are being worked out"
        prop.fades(asked, None, ahead=1)
        go.set()
        first, second = (kept["fades"] for kept in prop.repaints.kept.values())
        deadline = time.time() + 30
        while not all(fade in second and second[fade].done() for fade in ot_repaint.FADE_STEPS) \
                and time.time() < deadline:
            time.sleep(0.05)
        assert set(second) == set(ot_repaint.FADE_STEPS), "the repaint picked now: every fade worked out"
        assert len(first) <= 2, "the one picked before: its own fade and the one it was at"
    finally:
        go.set()
        prop.close()


def test_a_fade_that_failed_is_worked_out_anew_the_next_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fade's weights that could not be worked out once (the PC short of memory for a moment) are not kept as
    failed: asked again, they are worked out."""
    weight, failed = ot_repaint.Repaints.weight, []

    def once(self: ot_repaint.Repaints, look: ot_repaint.Look, fade: float) -> np.ndarray:
        if not failed:
            failed.append(fade)
            raise MemoryError
        return weight(self, look, fade)

    monkeypatch.setattr(ot_repaint.Repaints, "weight", once)
    prop, asked = two_repaints(tmp_path)
    try:
        with pytest.raises(MemoryError):
            prop.fades(asked[:1], [None])
        assert prop.fades(asked[:1], [None])["painted"][0] > 0
    finally:
        prop.close()


def test_no_model_is_still_being_written_when_a_fade_fails(tmp_path: Path) -> None:
    """One of a fade's models cannot be written: the error goes out only once the others are written (the hub
    then removes their folders, and nothing may still write into them)."""
    prop, asked = two_repaints(tmp_path)
    written: list[str] = []

    def write(out: Path, pixels: np.ndarray, quick: bool = False) -> None:
        if out.name == "one.glb":
            raise OSError("the drive is full")
        time.sleep(0.4)
        written.append(out.name)

    prop.write_model = write
    try:
        with pytest.raises(OSError):
            prop.fades(asked, [str(tmp_path / "one.glb"), str(tmp_path / "two.glb")])
        assert written == ["two.glb"]
    finally:
        prop.close()


def test_a_look_that_fails_leaves_the_look_before_it_whole(tmp_path: Path) -> None:
    """A look at a mask the camera cannot see fails; the look before it stays as it was, mask and all, so its
    picture still goes onto what it showed."""
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(box(tmp_path / "box.glb"), work, SIZE, 6, ahead=False)
    try:
        masked = centre_texels(prop, "+Z")
        ask, _ = lens_at_front(prop, masked, work)
        prop.look(ask, pictures=False)
        hidden, _ = lens_at_front(prop, centre_texels(prop, "-Z"), work)
        with pytest.raises(w.BakeError, match="in sight from here"):
            prop.look(hidden, pictures=False)
        assert prop.repaint(solid(work / "answer.png", (40, 160, 90)))["painted"] > 0
        middle = masked[len(masked) // 2 + int(np.sqrt(len(masked))) // 2]      # (the mask is a block of texels)
        assert tuple(prop.paint[middle]) == (40, 160, 90)
    finally:
        prop.close()


def test_a_repaint_needs_something_masked_in_sight(tmp_path: Path) -> None:
    """Nothing masked, or only what the page's camera cannot see (the box's back): the artist is told so in plain
    words, and nothing is painted."""
    work = tmp_path / "work"
    work.mkdir()
    prop = w.Prop(box(tmp_path / "box.glb"), work, SIZE, 6, ahead=False)
    try:
        ask, _ = lens_at_front(prop, np.zeros(0, np.int64), work)
        with pytest.raises(w.BakeError, match="Nothing is masked"):
            prop.look(ask)
        ask, _ = lens_at_front(prop, centre_texels(prop, "-Z"), work)
        with pytest.raises(w.BakeError, match="in sight from here"):
            prop.look(ask)
        with pytest.raises(w.RequestError):
            prop.repaint(solid(work / "answer.png", (1, 2, 3)))
    finally:
        prop.close()


def test_a_patch_where_the_mask_was_reads_high_and_a_restored_surface_low() -> None:
    """mask_left (Qwen-Image is to restore what is masked in magenta; now and then it lays a patch
    of one colour there instead): the mask's outline shows all the way round a patch, nowhere on a surface that
    goes on under it, and only along a part of it where a thing really ends at the mask. The model's own outline
    is no step: outside the model nothing is compared."""
    side = 200
    yy, xx = np.mgrid[:side, :side]
    surface = np.repeat((90.0 + 30.0 * np.sin(xx / 17.0) + 10.0 * np.cos(yy / 9.0))[..., None], 3, axis=2)
    where = (xx - 100) ** 2 + (yy - 100) ** 2 < 40 ** 2
    model = np.ones((side, side), dtype=bool)
    assert ot_masks.mask_left(surface, where, model) < 5.0
    patch = surface.copy()
    patch[where] = (150, 40, 60)
    assert ot_masks.mask_left(patch, where, model) > 25.0
    faint = surface.copy()
    faint[where] = surface[where] * 0.7 + np.array([10.0, 45.0, 10.0])     # a greenish patch, as one trial made
    assert ot_masks.mask_left(faint, where, model) > 12.0
    ends = surface.copy()
    ends[(xx > 100) & ~where] = 220.0                           # another thing begins along half of the outline
    assert ot_masks.mask_left(ends, where, model) < 10.0
    part = xx < 120                                             # the model ends inside the mask's circle
    cut = surface.copy()
    cut[~part] = 0.0
    assert ot_masks.mask_left(cut, where & part, part) < 5.0
    assert ot_masks.mask_left(patch, (xx - 100) ** 2 + (yy - 100) ** 2 < 3 ** 2, model) == 0.0, "too small to tell"


def test_the_repaint_commands_answer_as_json_lines(tmp_path: Path) -> None:
    """load with "plan": false, look, left, repaint, finish with "tone": false - the hub's repaint job, line by
    line. A picture of one colour all over has no outline of the mask in it."""
    work = tmp_path / "work"
    work.mkdir()
    glb_path = box(tmp_path / "box.glb")
    probe = w.Prop(glb_path, work, SIZE, 6, ahead=False)
    ask, _ = lens_at_front(probe, centre_texels(probe, "+Z"), work)
    probe.close()
    look = {"cmd": "look", **{key: ask[key] for key in ("eye", "forward", "up", "tan")}, "mask": str(ask["mask"]),
            "size": SIZE}
    answer = str(solid(work / "answer.png", (40, 160, 90)))
    lines = [{"cmd": "load", "glb": str(glb_path), "work": str(work), "size": SIZE, "views": 6, "plan": False}, look,
             {"cmd": "left", "picture": answer},
             {"cmd": "repaint", "picture": answer, "keep": str(work / "kept.png")},
             {"cmd": "finish", "out": str(tmp_path / "out.glb"), "tone": False}, {"cmd": "quit"}]
    out = io.StringIO()
    assert w.main(io.StringIO("\n".join(json.dumps(line) for line in lines) + "\n"), out) == 0
    answers = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [a.get("ok") for a in answers] == [True] * 5, answers
    assert set(answers[1]) >= {"depth", "viewport", "redrawn", "masked", "seen"} and answers[2]["left"] == 0.0
    assert answers[3]["painted"] > 0 and (work / "kept.png").is_file()
    assert (tmp_path / "out.glb").is_file()
    # the viewport picture itself, the magenta still in it, shows the whole of the mask's outline
    out = io.StringIO()
    w.main(io.StringIO("\n".join(json.dumps(line) for line in (lines[0], look, {"cmd": "left", "picture": str(
        work / "look_viewport.png")}, {"cmd": "quit"})) + "\n"), out)
    assert json.loads(out.getvalue().splitlines()[2])["left"] > 25.0


def test_a_load_that_fails_leaves_no_prop_behind(tmp_path: Path) -> None:
    """A second load that fails: the commands after it are told there is no prop, never answered from the first
    one, which was closed for it."""
    work = tmp_path / "work"
    load = {"cmd": "load", "glb": str(box(tmp_path / "box.glb")), "work": str(work), "size": SIZE, "views": 1}
    lines = [load, {**load, "glb": str(tmp_path / "missing.glb")}, {"cmd": "view", "index": 0}]
    out = io.StringIO()
    assert w.main(io.StringIO("\n".join(json.dumps(line) for line in lines) + "\n"), out) == 0
    first, second, third = (json.loads(line) for line in out.getvalue().splitlines())
    assert first["ok"] and "error" in second
    assert "'view' before 'load'" in third["error"]


def test_the_sheet_commands_answer_as_json_lines(tmp_path: Path) -> None:
    work = tmp_path / "work"
    glb_path = box(tmp_path / "box.glb")
    stdout = io.StringIO()
    load = {"cmd": "load", "glb": str(glb_path), "work": str(work), "size": SIZE, "views": 2,
            "together": [SIZE, 2 * SIZE]}
    assert w.main(io.StringIO(json.dumps(load) + "\n" + json.dumps({"cmd": "sheet", "index": 0}) + "\n"), stdout) == 0
    loaded, made = (json.loads(line) for line in stdout.getvalue().splitlines())
    assert loaded["ok"] and loaded["sheets"] == [{"size": 2 * SIZE, "views": [0, 1], "scale": loaded["sheets"][0]["scale"]}]
    assert made["ok"] and made["size"] == 2 * SIZE and Path(made["render"]).name == "sheet0_render.png"
    assert [(view["index"], view["name"], view["side"]) for view in made["views"]] == [(0, "front", "front"),
                                                                                       (1, "back", "back")]
    # a view of a sheet of several is not to be had alone
    stdout = io.StringIO()
    w.main(io.StringIO(json.dumps(load) + "\n" + json.dumps({"cmd": "view", "index": 0}) + "\n"), stdout)
    assert "error" in json.loads(stdout.getvalue().splitlines()[1])


def test_a_picture_is_made_ready_for_the_lighting_pass_without_a_model(tmp_path: Path) -> None:
    """The worker's "fit", for a repaint's example before its lighting is taken off: the pass takes pictures whose
    sides are multiples of 32, so the picture is laid, undistorted, in the middle of the smallest such canvas
    (the rest in its corner's colour), on white where it is see-through, and made smaller when its longer side is
    over `most`. The picture itself stays as it is."""
    cut_out = np.zeros((500, 300, 4), np.uint8)
    cut_out[100:400, 50:250] = (200, 40, 40, 255)                 # a red figure on nothing
    Image.fromarray(cut_out, "RGBA").save(tmp_path / "front.png")
    before = (tmp_path / "front.png").read_bytes()
    out = tmp_path / "examples" / "front.fit.png"
    stdout = io.StringIO()
    asked = {"cmd": "fit", "picture": str(tmp_path / "front.png"), "out": str(out), "most": 1024, "step": 32}
    assert w.main(io.StringIO(json.dumps(asked) + "\n"), stdout) == 0
    answer = json.loads(stdout.getvalue().splitlines()[-1])
    assert answer["ok"] and answer["picture"] == str(out) and answer["size"] == [320, 512]
    made = np.asarray(Image.open(out))
    assert made.shape == (512, 320, 3), "RGB, sides of multiples of 32"
    assert (made[0, 0] == 255).all() and (made[6 + 250, 10 + 150] == (200, 40, 40)).all(), "on white, in the middle"
    assert (made[6 + 100:6 + 400, 10 + 50:10 + 250] == (200, 40, 40)).all() and (made[:6] == 255).all()
    assert (tmp_path / "front.png").read_bytes() == before
    # A large picture: its longer side `most` px, the other as it follows, then the canvas.
    wide = np.full((1000, 2000, 3), (10, 90, 160), np.uint8)
    Image.fromarray(wide).save(tmp_path / "wide.png")
    answer = w.fit(tmp_path / "wide.png", tmp_path / "wide.fit.png", 1024, 32)
    assert answer["size"] == [1024, 512] and (np.asarray(Image.open(tmp_path / "wide.fit.png")) == (10, 90, 160)).all()
    # One that fits already goes through as it is.
    exact = np.random.default_rng(1).integers(0, 255, (64, 96, 3), dtype=np.uint8)
    Image.fromarray(exact).save(tmp_path / "exact.png")
    assert w.fit(tmp_path / "exact.png", tmp_path / "exact.fit.png", 1024, 32)["size"] == [96, 64]
    assert (np.asarray(Image.open(tmp_path / "exact.fit.png")) == exact).all()


def test_a_picture_is_mirrored_without_a_model(tmp_path: Path) -> None:
    """A repaint masked from behind the side the prop's picture shows gives Qwen-Image that picture flipped left to
    right: every pixel in its row, mirrored, the transparency with it; the picture itself stays as it is."""
    pixels = np.zeros((40, 60, 4), np.uint8)
    pixels[..., 3] = 255
    pixels[5, 10] = (255, 0, 0, 255)            # a red dot left of the middle ...
    pixels[:, 50:, 3] = 0                       # ... and a see-through band at the right edge
    Image.fromarray(pixels, "RGBA").save(tmp_path / "front.png")
    before = (tmp_path / "front.png").read_bytes()
    out = tmp_path / "retouch" / "3-example-mirrored.png"
    stdout = io.StringIO()
    assert w.main(io.StringIO(json.dumps({"cmd": "mirror", "picture": str(tmp_path / "front.png"), "out": str(out)}) + "\n"), stdout) == 0
    answer = json.loads(stdout.getvalue())
    assert answer["ok"] and answer["picture"] == str(out)
    with Image.open(out) as made:
        assert (made.mode, made.size) == ("RGBA", (60, 40))
        flipped = np.asarray(made)
    assert np.array_equal(flipped, pixels[:, ::-1])
    assert tuple(flipped[5, 49]) == (255, 0, 0, 255) and flipped[:, :10, 3].max() == 0
    assert (tmp_path / "front.png").read_bytes() == before


def test_the_settings_test_blurs_the_props_pictures_without_a_model(tmp_path: Path) -> None:
    """The settings' test may blur the prop's own pictures: shrunk to the side asked, enlarged back smoothly to
    Qwen-Image's size (sent small, the engine would enlarge them into pixel blocks), transparency kept, numbered
    after the render and the mask. A picture no larger than the side goes as it is."""
    noise = np.random.default_rng(0).integers(0, 256, (400, 600, 3), dtype=np.uint8)    # fine detail everywhere
    cut_out = np.dstack([noise, np.full((400, 600), 255, np.uint8)])
    cut_out[:, :300, 3] = 0                                     # the left half see-through, as around a cut-out
    Image.fromarray(cut_out, "RGBA").save(tmp_path / "front.png")
    Image.fromarray(noise[:100, :100]).save(tmp_path / "small.png")
    out = tmp_path / "0-front"
    request = {"cmd": "blur", "pictures": [str(tmp_path / "front.png"), str(tmp_path / "small.png")], "side": 128,
               "out": str(out), "size": 1024}
    stdout = io.StringIO()
    assert w.main(io.StringIO(json.dumps(request) + "\n"), stdout) == 0          # before any "load"
    answer = json.loads(stdout.getvalue())
    assert answer["ok"] and answer["pictures"] == [str(out / "3-front.png"), str(tmp_path / "small.png")]
    with Image.open(out / "3-front.png") as made:
        assert (made.mode, made.size) == ("RGBA", (1024, 683))
        pixels = np.asarray(made).astype(np.float64)
    # the see-through half stays see-through and the rest opaque, but for a smeared band along the seam (x = 512)
    assert pixels[:, :440, 3].max() <= 1 and pixels[:, 590:, 3].min() >= 254
    # the noise is gone: neighbours nearly alike (the picture's own differ by ~85 on average)
    before = np.abs(np.diff(noise.astype(np.float64), axis=1)).mean()
    after = np.abs(np.diff(pixels[:, 590:, :3], axis=1)).mean()
    assert before > 80 and after < 0.1 * before
