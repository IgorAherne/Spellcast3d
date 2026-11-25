"""The views' cameras and the pictures they are painted in, for Omni-Texture's worker.

Painted one by one, the views differ in colour. So the views that fit are painted together in one square picture,
a sheet: laid out by their outlines (omnitexture_pack), all at one scale. Views share a sheet only while every
one of them keeps FLOOR of the size it has alone in a picture (each its own: from above a model is larger alone
than from the side); the fewest such sheets are looked for among every combination of the views, down to a view a
picture. A view's camera then looks at its own window of the sheet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from omnitexture_masks import SPECK
from omnitexture_model import Model
from omnitexture_pack import fewest, fits, pack
from omnitexture_views import Camera, camera_alone, frame

OUTLINE = 384                # px along the model's longest extent in the outlines the layout is found from
GAP = 0.03                   # of a sheet's side: the room between two views and to its edge, at the least
FIT = 0.95                   # a sheet's side is the smallest allowed at which its views keep this of their size alone
FLOOR = 0.9                  # of its size alone: what every view of a sheet keeps, at the least
PAD = 2                      # px around a view's box in its part of a sheet (a texel reads the four around it)


@dataclass
class Sheet:
    """A picture Qwen-Image paints: its side, the views in it (in the order the words name them) and each one's
    window in it (x, y of its top left pixel; the camera holds its size), and the least any of its views keeps of
    the size it has alone in a picture of the load's size (1: the same)."""
    side: int
    views: list[int]
    at: dict[int, tuple[int, int]]
    scale: float


def _along_edges(x: np.ndarray, y: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The vertices (x, y: in pixels) and, on every edge longer than a pixel, a point a pixel."""
    a = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    b = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    steps = np.ceil(np.hypot(x[a] - x[b], y[a] - y[b])).astype(np.int64)
    long = np.flatnonzero(steps > 1)
    if long.size:
        edge = np.repeat(long, steps[long] - 1)
        first = np.cumsum(steps[long] - 1) - (steps[long] - 1)
        t = (np.arange(edge.size) - np.repeat(first, steps[long] - 1) + 1) / steps[edge]
        x = np.concatenate([x, x[a[edge]] + (x[b[edge]] - x[a[edge]]) * t])
        y = np.concatenate([y, y[a[edge]] + (y[b[edge]] - y[a[edge]]) * t])
    return x, y


def outline(scene: Any, mesh: Model, framed: tuple, unit: float) -> np.ndarray:
    """The model's outline from a view (`framed`: frame()'s) at `unit` px per unit of the model, cut to its box
    (bool, rows from the top): where a ray through a pixel's middle meets the model, and where any of its
    vertices or edges falls - a part thinner than a pixel may lie between the rays, and the layout would put
    another view over it."""
    (eye, forward, right, up), (x0, x1, y0, y1) = framed
    w, h = max(1, int(np.ceil((x1 - x0) * unit))), max(1, int(np.ceil((y1 - y0) * unit)))
    across = eye + (x0 + (np.arange(w) + 0.5) / unit)[:, None] * right
    down = (y1 - (np.arange(h) + 0.5) / unit)[:, None] * up
    origins = (across[None, :, :] + down[:, None, :]).reshape(-1, 3)
    hits = scene.first_front_hit(origins, np.broadcast_to(forward, origins.shape), np.inf)
    seen = (hits.prim >= 0).reshape(h, w)
    rel = mesh.positions - eye
    x, y = _along_edges((rel @ right - x0) * unit, (y1 - rel @ up) * unit, mesh.faces)
    seen[np.clip(y.astype(np.int64), 0, h - 1), np.clip(x.astype(np.int64), 0, w - 1)] = True
    return seen


def _reading_order(views: list[int], at: dict[int, tuple[int, int]], cameras: list[Camera]) -> list[int]:
    """A sheet's views as one reads them: row by row from the top, each row from the left (a view is in the row
    of the topmost one left when they share more than half the height of the shorter of the two)."""
    span = {k: (at[k][1], at[k][1] + cameras[k].height) for k in views}
    left = sorted(views, key=lambda k: (span[k][0], at[k][0]))
    ordered: list[int] = []
    while left:
        top, bottom = span[left[0]]
        row = [k for k in left if min(bottom, span[k][1]) - max(top, span[k][0])
               > 0.5 * min(bottom - top, span[k][1] - span[k][0])]
        ordered += sorted(row, key=lambda k: at[k][0])
        left = [k for k in left if k not in row]
    return ordered


def _window(framed: tuple, corner: tuple[float, float], ppu: float, side: int) -> tuple[Camera, tuple[int, int]]:
    """A view's camera in a sheet of `side` px, its box's top left `corner` laid there at `ppu` pixels per unit:
    the camera looks at the box and PAD around it -> the camera, and its window's top left pixel."""
    axes, (x0, x1, y0, y1) = framed
    ox, oy = corner
    wx, wy = max(int(np.floor(ox)) - PAD, 0), max(int(np.floor(oy)) - PAD, 0)
    wx1 = min(int(np.ceil(ox + (x1 - x0) * ppu)) + PAD, side)
    wy1 = min(int(np.ceil(oy + (y1 - y0) * ppu)) + PAD, side)
    return Camera(*axes, x0 - (ox - wx) / ppu, y1 + (oy - wy) / ppu, ppu, wx1 - wx, wy1 - wy), (wx, wy)


def lay_out(scene: Any, mesh: Model, sides: tuple, size: int,
            together: list[int] | None) -> tuple[list[Camera], list[int], list[Sheet]]:
    """The cameras of the views `sides` (rows of VIEWS) and the sheets they are painted in: every view alone in a
    picture of `size` px (no `together`, or one view), else together in sheets of the sides `together` allows ->
    per view its camera and the size of a speck its mask is cleaned of, and the sheets in the views' own order."""
    pos = mesh.positions
    count = len(sides)
    alone = [camera_alone(pos, az, el, size) for _, _, az, el in sides]
    cameras, speck = list(alone), [SPECK] * count
    groups: list[tuple[list[int], tuple | None]] = [([k], None) for k in range(count)]
    allowed = sorted({int(side) for side in together or []})
    if count > 1 and allowed:
        frames = [frame(pos, az, el) for _, _, az, el in sides]
        unit = OUTLINE / max(max(x1 - x0, y1 - y0) for _, (x0, x1, y0, y1) in frames)
        outlines = [outline(scene, mesh, framed, unit) for framed in frames]

        def floor(views: list[int]) -> float:
            """The layout scale at which each of the views keeps FLOOR of its size alone, in the largest sheet."""
            return FLOOR * max(alone[k].ppu for k in views) / (allowed[-1] * unit)

        def fit(views: list[int]) -> tuple:
            """The views' layout, their sheet's side, and the least any of them keeps there of its size alone."""
            layout = pack([outlines[k] for k in views], GAP, least=floor(views))
            kept = [min(layout.scale * side * unit / alone[k].ppu for k in views) for side in allowed]
            pick = next((i for i, keep in enumerate(kept) if keep >= FIT), len(allowed) - 1)
            return layout, allowed[pick], kept[pick]

        def kin(a: int, b: int) -> float:
            """How near two cameras stand (0: a right angle apart or more, or one around the model and the other
            slanted from above or below): their paintings meet on the model, so they belong in one sheet."""
            to_a, to_b = -np.asarray(frames[a][0][1]), -np.asarray(frames[b][0][1])
            return max(0.0, float(to_a @ to_b)) if (abs(to_a[1]) > 0.5) == (abs(to_b[1]) > 0.5) else 0.0

        # Where several ways need as few sheets, the one whose sheets hold neighbouring cameras: the seams
        # between sheets then fall on few lines of the model.
        groups = [(views, fit(views) if len(views) > 1 else None) for views in fewest(
            count, lambda views: fits([outlines[k] for k in views], GAP, floor(views)), kin)]
    sheets = []
    for views, plan in sorted(groups, key=lambda group: group[0][0]):    # in the views' own order
        if plan is None:
            sheets.append(Sheet(size, views, {views[0]: (0, 0)}, 1.0))
            continue
        layout, side, kept = plan
        ppu = layout.scale * side * unit
        at = {}
        for k, (fx, fy) in zip(views, layout.places):
            cameras[k], at[k] = _window(frames[k], (fx * side, fy * side), ppu, side)
            speck[k] = max(1, round(SPECK * ppu / alone[k].ppu))
        sheets.append(Sheet(side, _reading_order(views, at, cameras), at, float(kept)))
    return cameras, speck, sheets
