"""Omni-Texture's sheets (launchers/omnitexture_pack.py): several views' outlines laid out in one square picture at
one scale, by their outlines and not their boxes, a gap apart.

    code\\venvs\\imesh\\Scripts\\python -m pytest code/tests/bake/test_omnitexture_pack.py
"""

from __future__ import annotations

import numpy as np

import omnitexture_pack as p

SIDE = 1024
GAP = 0.03


def figure(width: int = 150, height: int = 360) -> np.ndarray:
    """A standing figure, arms down: taller than wide, narrower at the head and between the legs."""
    m = np.zeros((height, width), dtype=bool)
    mid = width // 2
    m[: height // 6, mid - width // 5: mid + width // 5] = True          # head
    m[height // 6: height // 2] = True                                    # shoulders, arms, torso
    m[height // 2:, mid - width // 3: mid - width // 12] = True           # legs
    m[height // 2:, mid + width // 12: mid + width // 3] = True
    return m


def t_pose(width: int = 300, height: int = 384) -> np.ndarray:
    """A figure with its arms straight out: as wide as it is tall, mostly empty box."""
    m = np.zeros((height, width), dtype=bool)
    mid = width // 2
    m[:60, mid - 25: mid + 25] = True                                     # head
    m[60:90] = True                                                       # arms
    m[60:220, mid - 40: mid + 40] = True                                  # torso
    m[220:, mid - 40: mid - 8] = True                                     # legs
    m[220:, mid + 8: mid + 40] = True
    return m


def drawn(outlines: list[np.ndarray], layout: p.Layout, side: int = SIDE) -> list[np.ndarray]:
    """Each outline as it lies in the picture (bool, side x side), at the layout's scale and place."""
    pictures = []
    for outline, (fx, fy) in zip(outlines, layout.places):
        h, w = outline.shape
        ys = np.clip(np.floor((np.arange(side) + 0.5 - fy * side) / (layout.scale * side)).astype(int), -1, h)
        xs = np.clip(np.floor((np.arange(side) + 0.5 - fx * side) / (layout.scale * side)).astype(int), -1, w)
        ok = ((ys >= 0) & (ys < h))[:, None] & ((xs >= 0) & (xs < w))[None, :]
        pictures.append(ok & outline[np.clip(ys, 0, h - 1)][:, np.clip(xs, 0, w - 1)])
    return pictures


def apart(a: np.ndarray, b: np.ndarray, need: int) -> bool:
    """Whether two drawn outlines lie `need` pixels apart or more (one grown by that much stays clear of the other)."""
    return not (p._grown(a, need, need) & b).any()


def box(picture: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.flatnonzero(picture.any(axis=1)), np.flatnonzero(picture.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def test_one_view_fills_the_picture_but_the_gap_and_lies_in_its_middle() -> None:
    outline = figure()
    layout = p.pack([outline], GAP)
    x0, y0, x1, y1 = box(drawn([outline], layout)[0])
    assert abs((y1 - y0) - SIDE * (1 - 2 * GAP)) <= 10, "as tall as the picture less the gap above and below"
    assert abs(y0 - (SIDE - y1)) <= 2 and abs(x0 - (SIDE - x1)) <= 2, "centred"


def test_two_figures_stand_side_by_side_as_large_as_one_and_evenly_spread() -> None:
    outlines = [figure(), figure()]
    alone, two = p.pack(outlines[:1], GAP), p.pack(outlines, GAP)
    assert two.scale == alone.scale, "two tall figures fit beside each other at the size of one"
    (ax0, ay0, ax1, _), (bx0, by0, bx1, _) = (box(d) for d in drawn(outlines, two))
    assert ay0 == by0 and ax1 < bx0, "in one row, in their order"
    left, between, right = ax0, bx0 - ax1, SIDE - bx1
    assert max(left, between, right) - min(left, between, right) <= 3 * SIDE // p.GRID, "the room shared out"
    assert two.gap >= GAP - 1e-9


def test_the_views_keep_the_gap_from_each_other_and_from_the_edge() -> None:
    outlines = [figure(), figure(), figure(110), figure(110), figure(110), figure(110), t_pose(150, 200)]
    layout = p.pack(outlines, GAP)
    pictures = drawn(outlines, layout)
    need = int(GAP * SIDE) - 2 * SIDE // p.GRID                # (the search's cells round the outlines up)
    for i, a in enumerate(pictures):
        x0, y0, x1, y1 = box(a)
        assert min(x0, y0, SIDE - x1, SIDE - y1) >= need, f"view {i} keeps off the edge"
        for j in range(i + 1, len(pictures)):
            assert apart(a, pictures[j], need), f"views {i} and {j}"
    assert sum(int(d.sum()) for d in pictures) == int(np.logical_or.reduce(pictures).sum()), "none over another"


def test_figures_with_their_arms_out_fit_larger_by_their_outlines_than_by_their_boxes() -> None:
    """Two T poses: by their boxes they must stand side by side, half the picture's width each; by their outlines
    one's arm lies above the other's (as in a real prop), and both are larger."""
    outlines = [t_pose(), t_pose()]
    by_outline = p.pack(outlines, GAP)
    by_box = p.pack([np.ones_like(o) for o in outlines], GAP)
    assert by_outline.scale > 1.15 * by_box.scale
    a, b = drawn(outlines, by_outline)
    (_, ay0, ax1, _), (bx0, by0, _, _) = box(a), box(b)
    assert bx0 < ax1 and ay0 != by0, "their boxes overlap across, one stands lower than the other"
    assert not (a & b).any() and apart(a, b, int(GAP * SIDE) - 2 * SIDE // p.GRID)


def test_tidy_rows_are_kept_where_the_outlines_gain_little() -> None:
    """Six stocky figures: their outlines would fit a few per cent larger staggered; they stay in two rows of
    three, in their order."""
    stocky = np.ones((360, 150), dtype=bool)
    stocky[:40, :20] = stocky[:40, -20:] = False                 # only the shoulders' corners are free
    outlines = [stocky.copy() for _ in range(6)]
    layout = p.pack(outlines, GAP)
    boxes = [box(d) for d in drawn(outlines, layout)]
    tops = [b[1] for b in boxes]
    assert tops[0] == tops[1] == tops[2] and tops[3] == tops[4] == tops[5] and tops[3] > boxes[0][3]
    assert boxes[0][0] < boxes[1][0] < boxes[2][0] and boxes[3][0] < boxes[4][0] < boxes[5][0]


def test_whether_views_can_share_a_picture_at_a_scale_is_one_quick_question() -> None:
    """fits(): two standing figures side by side at the scale of one alone - yes; three - no; a little smaller -
    yes; and pack() never comes back with less than a scale fits() agreed to."""
    one = p.pack([figure()], GAP).scale
    assert p.fits([figure(), figure()], GAP, one) and not p.fits([figure(), figure(), figure()], GAP, one)
    assert p.fits([figure(), figure(), figure()], GAP, 0.6 * one)
    assert not p.fits([figure()], GAP, 1.05 * one), "larger than the picture holds"
    three = [figure(), figure(), figure()]
    least = max(s for s in np.linspace(0.6, 1.0, 21) * one if p.fits(three, GAP, s))
    assert p.pack(three, GAP, least=least).scale >= least - 1e-12


def test_the_fewest_pictures_are_found_among_every_combination_of_the_views() -> None:
    """fewest(): two slim views and two wide ones, a slim one first. A wide view fits no picture with another; the
    slim ones fit one together. Going through the views in their order - slim, wide, slim, wide - would start a
    new picture each time; the search puts the slim ones together: three pictures. Among ways with as few
    pictures the kin stay together, and a set is asked about only once its smaller sets fit."""
    slim, asked = {0, 2}, []

    def share(views: list[int]) -> bool:
        asked.append(tuple(views))
        return set(views) <= slim

    assert p.fewest(4, share, lambda a, b: 0.0) == [[0, 2], [1], [3]]
    assert all(len(views) == 2 for views in asked), "pairs only: no set of three had all its pairs fitting"
    # six views around a model, any three of which fit a picture: two pictures of three, neighbours together
    ring = [0.0, 180.0, 60.0, -60.0, 120.0, -120.0]                      # front, back, the four between

    def kin(a: int, b: int) -> float:
        return max(0.0, float(np.cos(np.radians(ring[a] - ring[b]))))

    assert p.fewest(6, lambda views: len(views) <= 3, kin) == [[0, 2, 3], [1, 4, 5]]
    assert p.fewest(3, lambda views: False, kin) == [[0], [1], [2]] and p.fewest(1, lambda views: True, kin) == [[0]]


def test_a_limit_keeps_the_views_from_growing_and_nothing_is_no_layout() -> None:
    outline = figure()
    free = p.pack([outline], GAP)
    held = p.pack([outline], GAP, limit=free.scale / 2)
    assert abs(held.scale - free.scale / 2) < 1e-9
    x0, y0, x1, y1 = box(drawn([outline], held)[0])
    assert abs(y0 - (SIDE - y1)) <= 2 and abs(x0 - (SIDE - x1)) <= 2, "still centred"
    assert p.pack([], GAP) is None
