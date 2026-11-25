"""Omni-Texture's CPU half, one prop at a time, in the imesh venv (adapters/omnitexture.py drives it, Qwen-Image paints).

    python omnitexture_worker.py          (JSON lines: a command on stdin, its answer on stdout)

It holds the prop's 3D model (omnitexture_model: read as spellbake.highpoly reads it, without the maps), an Embree
scene of it, its base colour texture as the PNG/WebP stores it (sRGB bytes) and, per texel of the UV islands, the
surface point and a smoothed normal there. The model is seen from the first "views" of VIEWS (omnitexture_views;
all of them when the load names no count), each through an orthographic camera: alone in its picture, framed
tightly, or - the load's "together" - several in one picture, a sheet (omnitexture_sheets), so that Qwen-Image
paints them in one go and they agree in colour. A view's camera then looks at its own window of the sheet.

    {"cmd": "load", "glb": <the 3D model>, "work": <folder for the pictures>, "size": 1024, "views": <optional count>,
     "together": <optional: the sides a sheet may have, e.g. [1024, 1152, ... 2048]>,
     "plan": <optional: false for a repaint ("look"): the views are not looked through ahead>,
     "skip": <optional: indices of views laid out as the others but painting nothing, as if their cameras were
              not there (a view the artist took off: the result composed again from the pictures kept)>}
        -> {"views": [names], "texture": [w, h], "faces": n,
            "sheets": [{"size": its side, "views": [indices, as the words name them], "scale": the model's size in
                        it against a view alone}],
            "windows": [per view its window in its sheet: [x, y, width, height]]}: the pictures to paint, a view
           each without "together"
    {"cmd": "sheet", "index": j, "show": <as "view">}
        -> sheet j's pictures, as "view"'s but of all its views, each in its window: sheet<j>_render.png,
           sheet<j>_model.png, sheet<j>_mask.png -> {"render", "model", "mask", "size", "scale",
           "views": [{"index", "name", "side", "object", "paint", "window": [x, y, width, height]}]}
    {"cmd": "paint", "index": j, "picture": <Qwen-Image's picture of sheet j, or its albedo>, "colour", "keep": as
     "project", "painting": <Qwen-Image's picture>, "views": [{"index", "layer": <.webp>, "answer": <.png>,
     "keep": <.png>}]}
        -> each view's window of the picture onto that view's texels, as "project"; per view its layer, and its own
           cut of the painting ("answer") and of what went onto the model ("keep"): its window, black away from
           the model -> {"painted", "done", "views": [{"index", "name", "painted"}], "matched": the texels over
           which the picture was brought to the tone of the pictures before it (0: left as it came)}
    {"cmd": "view", "index": k, "show": <optional: "depth" | "clay" | "texture" | "normal">}
        -> the pictures for view k, alone in its picture, in work/: <k>_<name>_render.png (picture 1: the model's
           depth map, white near, on black; also the picture Qwen-Image starts from; the settings' test may ask
           for the model in plain clay grey lit from slightly to the side, its own texture, or its normal map as
           the camera sees it, instead of the depth), <k>_<name>_model.png (picture 2: the model with its own
           texture, the 3D stage's, nothing painted over it) and <k>_<name>_mask.png (where the painting goes onto
           the model, and how much of it across the seams: white fully; never shown to Qwen-Image)
           -> {"render", "model", "mask", "name", "side", "object": fraction of the picture,
               "paint": fraction of the model it paints}
    {"cmd": "project", "index": k, "picture": <Qwen-Image's picture of view k, or its albedo>,
     "colour": <optional: Qwen-Image's picture when "picture" is its albedo>, "keep": <optional .png path>,
     "layer": <optional .webp path>}
        -> the picture onto every texel of this camera's in the plan, at its weight -> {"painted": texels,
           "done": fraction of the islands painted so far}. With "colour", what is projected is that picture's
           colour at the albedo's brightness (keep_colour); "keep" saves what was projected; "layer" saves this
           view's share alone as a texture of the model (over LAYER_GREY, which is wherever it painted nothing),
           which the page shows on the model.
    {"cmd": "finish", "out": <glb to write>, "tone": <optional: false leaves the model's own texture as it is>,
     "hand_paint": <optional: a paint layer (below) to lay over the texture>, "unpainted": <optional: the glb
                   without it>}
        -> the 3D model with the new texture (its other maps and the geometry as they were) -> {"glb", "stats"}.
           Where no camera painted, the model's own texture stays, retinted to the paintings' tone (match_tone);
           stats "own": the share of the texture that is mostly its own
    {"cmd": "look", "eye": [x, y, z], "forward": [..], "up": [..], "tan": <the tangent of half the picture's angle>,
     "mask": <png: the texture's texels the artist masked, white>, "size": 1024}
        -> a repaint of what the artist masked, through the page's own camera (omnitexture_repaint; a load with
           "plan": false: nothing is shared out between the stage's cameras): look_depth.png (the model's depth
           from there, as a view's), look_viewport.png (the model as it is textured now, the masked surface in
           magenta), look_redrawn.png (white where that view is masked, and a few pixels around: what the image
           generator draws anew, held to the view elsewhere) -> {"depth", "viewport", "redrawn", "masked": texels,
               "seen": those of them this camera can paint}; with "pictures": false only the last two, and no
           picture is written (a repaint put on again: its picture is there already)
    {"cmd": "left", "picture": <Qwen-Image's picture of what "look" showed it>}
        -> {"left": how much of the mask's shape is still in the picture, in levels (mask_left): a picture that
            shows a patch of one colour where the magenta was, instead of the surface restored, reads high}
    {"cmd": "repaint", "picture": <Qwen-Image's picture, or its albedo>, "colour", "keep", "layer": as "project",
     "fade": <optional: how many times as far its edge fades into what is around it; 1>,
     "tone": <optional: false for a picture that is in the texture's tone already, a "keep" of before>}
        -> the picture onto the masked texels this camera sees, brought to the tone of the texture around them
           first; the edge of the repainted surface is blended in 3D as a seam between views is
           -> {"painted": texels, "matched": texels the tone was read from}; then "finish" with "tone": false
    {"cmd": "fades", "repaints": [{"mask", "picture", "eye", "forward", "up", "tan", "size", "fade", "layer",
                                   "hand_paint", "unpainted"}, ..], "outs": [<glb or null per repaint>] or null}
        -> the repaints put onto the model as loaded, one after the other, each as "look" ("pictures": false) and
           "repaint" (its picture, "tone": false: in the texture's tone already) with its fade, then what was
           painted by hand on its step over it ("hand_paint": its paint layer, below); the model after each
           written to its "out" where there is one (and, before its hand paint, to its "unpainted"), the "layer"s
           where asked. What a fade does not change is kept from call to call (each repaint's look, picture and
           weights per fade, by its place in the list; a paint layer while its file stays the same): a change of
           one repaint's fade costs its blend, laying the pictures on and the models (the page's edge fade).
           Without "outs" only that is made ready (the page picked a repaint). "ahead": the repaint whose weights
           at the slider's other fades are worked out meanwhile -> {"painted": [..]}
    {"cmd": "blur", "pictures": [paths], "side": px, "out": <folder>, "size": 1024}   (needs no "load")
        -> the pictures with their details blurred away (blur(): the settings' test) -> {"pictures": [paths]}
    {"cmd": "mirror", "picture": <path>, "out": <path>}   (needs no "load")
        -> the picture flipped left to right, saved as "out" (a repaint's example seen from behind) -> {"picture"}
    {"cmd": "hand_paint", "model": <the step's glb as it was made>, "layer": <its paint layer .png>,
     "patch": <raw RGBA of the texture's size>, "width", "height", "out": <the step's glb>}   (needs no "load")
        -> what the page's hand painted since its last save (the patch: alpha 0 where nothing changed) into the
           step's paint layer - an RGBA picture of the texture's size, alpha 255 where painted by hand, a later
           paint over an earlier one -, and "out" the model with that layer over its texture (its alpha the
           model's own), encoded once from the model as made -> {"painted": texels, "texture": [w, h]}

The mask is ours alone: Qwen-Image paints the whole model from picture 1 and the start, and the mask picks what of
it goes onto the model - every surface this camera sees facing it at FACING or more (smoothed normals), the same
rule for every view - but for what lies beside a snap in its depth, one part of the model in front of another
(omnitexture_masks). A surface two cameras see goes to the one facing it best, earlier or later: so before the
first view the worker looks through every camera once and shares the texels out (Prop._share_out). The seams - one
view's picture to another's, or to the model's own texture where no view paints - are blended in 3D, on the model's
surface across the UV islands, SEAM of the texture's width across (omnitexture_blend). Each view's picture then
goes onto its texels at its weight, mixed over what the paint holds so that every texel ends as that mix whatever
the order.

Pictures painted apart differ in tone, and they meet on the model: every picture after the first goes on in the tone
of those before it (Prop._agree; omnitexture_tone).

Answers are {"ok": true, ...} or {"error": "<words for the artist>", "kind": "user" | "internal"}.

From the load on, the worker works ahead on a thread of its own: what every camera sees of the model (rays, hits,
texels, depths, facing), then the plan from it, which the first view waits for.
"""

from __future__ import annotations

import io
import json
import sys
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spellbake import BakeError, fill, glb, rays, replace_files
from omnitexture_blend import blend
from omnitexture_masks import SNAP, clean, closing, grow, snaps
from omnitexture_model import RASTER_PARTS, open_base_colour, pixels, read_model, texels
from omnitexture_pixels import between, four_around, nearest
from omnitexture_repaint import Look, Repaints
from omnitexture_sheets import Sheet, lay_out
from omnitexture_tone import (LAYER_GREY, keep_colour, laid_over, match_tone, on_clay, recolour, to_bytes,
                               tone_table)
from omnitexture_views import FACING, VIEWS, Camera, Lens, Sight, see

WEBP_QUALITY = 95
SEAM = 0.008                 # of the texture's width: the texels a seam's blend is across (16 at 2048, 8 at 1024)
DEPTH_TOLERANCE = 0.003      # of the model's diagonal: a texel this close behind the surface a pixel sees is seen
                             # (an exact ray test caught on the bumps of a generated surface: texels went unpainted)
HOLE_TEXELS = 1 / 1000       # of the texture's side: a hole in what the views painted up to twice this wide takes
                             # the painted colours around it (settle)
LAYER_REACH = 4              # texels of the gutter beside a view's texels that take their colour in its layer
LAYER_QUALITY = 90           # the layer's WebP quality (it is looked at, never baked)
CUT_REACH = 6                # px around the model kept in a view's own cut of its sheet's picture


class RequestError(Exception):
    """The request itself is malformed: a bug in whoever wrote it."""


def _lens(eye: list[float], forward: list[float], up: list[float], tan: float, size: int) -> Lens:
    lens = Lens.looking(eye, forward, up, tan, size)
    if lens is None:
        raise RequestError("the camera's forward and up are one line, or its angle is none")
    return lens


class Prop:
    """One prop while it is painted: its model's surface per texel, the cameras and their sheets, the plan (which
    camera's picture goes onto which texel, at what weight) and the texture as painted so far."""

    def __init__(self, glb_path: Path, work: Path, size: int, views: int | None = None,
                 together: list[int] | None = None, ahead: bool = True, skip: list[int] | None = None) -> None:
        """size: the side of a picture with one view alone in it; together: the sides a sheet of several views
        may have (None: every view alone in its picture); ahead False: the views are not looked through from the
        load on (a repaint through the page's camera needs none of them shared out); skip: the views laid out
        with the others that paint nothing."""
        self.work = work
        self.size = size
        self.sides = VIEWS if views is None else VIEWS[:max(1, int(views))]      # the cameras used, in order
        self.skip = frozenset(int(k) for k in skip or ())
        self.data = glb_path.read_bytes()
        picture, self.kind = open_base_colour(self.data)
        if picture.width != picture.height:
            raise BakeError("Only a square colour texture can be painted. Make the 3D model again in the 3D stage.")
        self.res = res = picture.width
        # The steps that need not wait for each other run side by side (numpy, Embree and the decoders let go of
        # the GIL): the texture, the scene, the UV raster, the smoothed normals, the gutter's fill plan.
        with ThreadPoolExecutor(RASTER_PARTS + 2) as pool:
            texture = pool.submit(pixels, picture)
            self.mesh = read_model(glb_path)
            if self.mesh.materials != 1:
                raise BakeError("Omni-Texture paints 3D models with one material; this one has "
                                f"{self.mesh.materials}.")
            self.area = self.mesh.area()
            scene = pool.submit(rays.Scene, self.mesh.positions, self.mesh.faces, threads=16)
            self.texel, self.point, self.normal, self.smooth = texels(self.mesh, res, pool)
            self.inside = np.zeros(res * res, dtype=bool)
            self.inside[self.texel] = True
            gutter_fill = pool.submit(fill.Fill, self.inside.reshape(res, res))   # grown, then push-pull (settle)
            self.texture = texture.result()
            self.scene = scene.result()
            self.cameras, self.speck, self.sheets = lay_out(self.scene, self.mesh, self.sides, size, together)
            self.sheet_of = {k: j for j, sheet in enumerate(self.sheets) for k in sheet.views}
            self.gutter_fill = gutter_fill.result()
        self.own = self.texture[..., :3].reshape(-1, 3)             # the model's own colours, as loaded
        self.paint = self.own.copy()                                # the same, until a view paints
        self.done = np.zeros(res * res, dtype=bool)                 # texels painted, or closed by settle
        self.filled = np.zeros(res * res, dtype=bool)               # ... those settle gave their neighbours' colour
        self.seen_done = self.done.copy()            # done, and the gutter beside done texels (settle)
        self.settled = False                         # the paint's gutter is the islands' fill (settle)
        self.views: list[dict[str, Any]] = []
        self.matched: dict[int, int] = {}            # per sheet painted: the texels its tone was matched over (_agree)
        self.masks: dict[int, np.ndarray] = {}      # per view: the pixels whose surfaces may take its picture
        self.depths: dict[int, np.ndarray] = {}     # per view: the depth each pixel sees (inf: no model)
        # Per view the texels its picture goes onto, their places in it and their weights (_share_out: made once,
        # from every camera), and per texel the weight what the paint holds now has (the model's own texture's at
        # first).
        self.plan: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None
        self.held = np.zeros(0)
        self.own_weight = np.zeros(0, np.float32)    # per texel the plan's weight of the model's own texture
        # Per view every texel it can paint, its place in the view's picture and how squarely it is faced; and
        # what the pictures painted so far show there, each texel from the first picture that can paint it: what
        # a later picture's tone is brought to (_agree). `agree` False leaves every picture as it comes.
        self.reach: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self.shown = np.zeros((0, 3), np.uint8)
        self.shown_at = np.zeros(0, dtype=bool)
        self.agree = True
        self.writer = ThreadPoolExecutor(1)          # the kept pictures, a view's render
        self.layers = ThreadPoolExecutor(1)          # the views' layers, written while the next view is painted
        self._layers: list[Future[None]] = []
        self.bands = ThreadPoolExecutor(fill.BANDS)  # settle's steps, in bands of rows (the same bits)
        # What every camera sees, then the plan from it, made on this thread from the load on (a worker started
        # for a job's next prop has them made by its turn); the first view waits for them.
        self.ahead = ThreadPoolExecutor(1)
        self._sights: dict[int, Future[Sight] | None] = {k: self.ahead.submit(self.sight, k)
                                                          for k in range(len(self.sides) if ahead else 0)
                                                          if k not in self.skip}
        self._seen: dict[int, Sight] = {}           # those taken, until their view's pictures are made
        self._planned: Future[None] | None = self.ahead.submit(self._share_out) if ahead else None
        self.repaints = Repaints(self)
        self._hand_paints: dict[Path, tuple[tuple[int, int], np.ndarray]] = {}  # paint layers read (hand_paint)

    def close(self) -> None:
        self.ahead.shutdown(wait=True, cancel_futures=True)      # before the scene its Sights cast rays into
        self.scene.close()
        self.writer.shutdown(wait=True)
        self.layers.shutdown(wait=True)
        self.repaints.close()
        self.bands.shutdown(wait=True)

    def loaded(self) -> dict[str, Any]:
        """The answer to "load": the pictures to paint."""
        return {"views": [side[0] for side in self.sides], "texture": [self.res, self.res],
                "faces": int(len(self.mesh.faces)),
                "sheets": [{"size": sheet.side, "views": sheet.views, "scale": round(sheet.scale, 4)}
                           for sheet in self.sheets],
                "windows": [[*map(int, self.sheets[self.sheet_of[k]].at[k]), cam.width, cam.height]
                            for k, cam in enumerate(self.cameras)]}

    # ---- what each camera sees, and the plan -----------------------------------------------------------------------
    def sight(self, k: int) -> Sight:
        return see(self.cameras[k], self.scene, self.mesh, self.smooth, self.res)

    def _sight_of(self, k: int) -> Sight:
        """View k's Sight: the one made ahead (waited for), else made now; kept until view(k) has used it."""
        if k not in self._seen:
            early = self._sights.pop(k, None)
            self._seen[k] = early.result() if early is not None else self.sight(k)
        return self._seen[k]

    def _mask(self, k: int) -> np.ndarray:
        """View k's mask (made once): what its camera sees facing it at FACING or more, every view alike, without
        specks or holes. Its edges are hard: the seams are blended in 3D."""
        if k not in self.masks:
            sight = self._sight_of(k)
            shape = sight.depth.shape
            raw = np.zeros(sight.hit.size, dtype=bool)
            raw[sight.ids[sight.facing]] = True
            self.masks[k] = clean(raw.reshape(shape), self.speck[k]) & sight.hit.reshape(shape)
            self.depths[k] = sight.depth
        return self.masks[k]

    def _near_model(self, k: int) -> np.ndarray:
        """In view k's window: the model and CUT_REACH around it (another view's arm may reach into the window)."""
        return grow(np.isfinite(self.depths[k]), CUT_REACH)

    def targets(self, cam: Camera | Lens, mask: np.ndarray,
                depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The texels a camera's picture may go onto, given its mask and depths -> their indices into self.texel,
        their places in the picture (px, py) and how squarely the camera faces them. Every texel is kept test by
        test, each test computing per texel what it would on all of them: in front of the camera, in the picture,
        inside the mask, seen, away from every snap in depth, facing it."""
        px, py, z = cam.place(self.point)
        todo = np.flatnonzero(z > 1e-9)
        if todo.size < z.size:
            px, py, z = px[todo], py[todo], z[todo]
        w, h = cam.width, cam.height
        at = np.flatnonzero((px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1))
        col, row = nearest(px[at], py[at], w, h)
        ok = mask.reshape(-1)[row * w + col]
        # Seen, not behind: no farther than the nearest surface any of the four pixels around its place sees (the
        # four its colour is read between). Against its own pixel alone, the surface hidden just behind the
        # outline of a nearer one took that one's colours: the picture shone through along the outline. The four
        # lie up to a pixel apart: on a surface slanted to the FACING limit that is most of two pixels' sides in
        # depth, more than DEPTH_TOLERANCE in a small picture.
        step = cam.step(z[at])
        tolerance = np.maximum(DEPTH_TOLERANCE * self.scene.diag, 2.0 * step)
        a, b, c, d = four_around(px[at], py[at], w, h)
        flat = depth.reshape(-1)
        ok &= z[at] <= np.minimum(np.minimum(flat[a], flat[b]), np.minimum(flat[c], flat[d])) + tolerance
        # Away from every snap: none of the four is beside one. A surface slanted to the FACING limit steps most
        # of two pixels' sides in depth from pixel to pixel: never a snap, however small the picture.
        pixel = float(np.median(step)) if np.ndim(step) and np.size(step) else float(np.max(step, initial=0.0))
        beside = snaps(depth, max(SNAP * self.scene.diag, 2.0 * pixel)).reshape(-1)
        ok &= ~(beside[a] | beside[b] | beside[c] | beside[d])
        at = at[ok]
        facing = cam.facing(self.point[todo[at]], self.normal[todo[at]])
        squarely = facing >= FACING
        at = at[squarely]
        return todo[at], px[at], py[at], facing[squarely]

    def _targets_of(self, k: int, mask: np.ndarray,
                    depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The texels view k's picture may go onto, given its mask and depths (targets)."""
        return self.targets(self.cameras[k], mask, depth)

    def seam_texels(self) -> float:
        """How far to each side of a seam the blend reaches, in texels of the texture: half of SEAM."""
        return SEAM * self.res / 2.0

    def seam_reach(self) -> float:
        """That reach on the model's surface, by a texel's mean side there."""
        return self.seam_texels() * float(np.sqrt(self.area / max(len(self.texel), 1)))

    def _plan(self) -> None:
        """The plan, made once before the first view is painted (ahead, from the load on; else now)."""
        if self.plan is None:
            early, self._planned = self._planned, None
            if early is not None:
                early.result()
            if self.plan is None:
                self._share_out()

    def _share_out(self) -> None:
        """Shares the texels out between the cameras: from what each camera sees of the model alone, every texel to
        the camera that faces it best (so a later camera keeps an earlier, worse one off it), the seams between
        them blended in 3D over SEAM of the texture (blend). A view skipped can paint none of them."""
        size = len(self.texel)
        owners, best = np.full(size, -1, np.int16), np.full(size, -np.inf)
        found, can = [], []
        none = np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0)
        for k in range(len(self.sides)):
            chosen, px, py, facing = none if k in self.skip else self._targets_of(k, self._mask(k), self.depths[k])
            found.append((chosen, px, py))
            self.reach[k] = (chosen.astype(np.int32), px.astype(np.float32), py.astype(np.float32),
                             facing.astype(np.float32))
            able = np.zeros(size, dtype=bool)
            able[chosen] = True
            can.append(able)
            better = facing > best[chosen]
            owners[chosen[better]], best[chosen[better]] = k, facing[better]
        texel = np.sqrt(self.area / max(size, 1))                # a texel's side on the model's surface
        weights, left = blend(self.point, self.normal, owners, can, self.seam_reach(), grain=2.0 * texel)
        self.held = left.astype(np.float64)
        self.own_weight = left
        self.shown, self.shown_at = np.zeros((size, 3), np.uint8), np.zeros(size, dtype=bool)
        plan = {}
        for k, ((chosen, px, py), weight) in enumerate(zip(found, weights)):
            mine = weight[chosen].astype(np.float64)
            some = mine > 0
            plan[k] = (chosen[some], px[some], py[some], mine[some])
        self.plan = plan

    # ---- a view's pictures -----------------------------------------------------------------------------------------
    def sheet(self, j: int, show: str = "depth") -> dict[str, Any]:
        """Sheet j's pictures (the module doc's "sheet"): each of its views drawn into its window."""
        self._plan()
        sheet = self.sheets[j]
        s = sheet.side
        render = np.zeros((s, s, 3), np.uint8)                  # picture 1 and the start, on black
        textured = np.zeros((s, s, 3), np.uint8)                # picture 2: the model as the 3D stage textured it
        painted = np.zeros((s, s), np.float32)
        views = []
        for k in sheet.views:
            name, side, _, _ = self.sides[k]
            cam = self.cameras[k]
            sight = self._seen.pop(k) if k in self._seen else self.sight(k)   # (a view asked again: made anew)
            x, y = sheet.at[k]
            window = (slice(y, y + cam.height), slice(x, x + cam.width))
            hit = sight.hit.reshape(cam.height, cam.width)      # only its own pixels: windows may overlap
            colours = self.own[sight.texel]
            render[window][hit] = (colours if show == "texture" else sight.grey[:, None] if show == "depth"
                                   else sight.normal if show == "normal" else sight.clay)
            textured[window][hit] = colours
            # How much of its picture each pixel's surface takes (the plan: its mask and the better cameras)
            chosen, _, _, weight = self.plan[k]
            taken = np.zeros(self.res * self.res, np.float32)
            taken[self.texel[chosen]] = weight
            mine = taken[sight.texel]
            painted[window][hit] = mine
            objects = int(sight.ids.size)
            views.append({"index": k, "name": name, "side": side, "object": objects / (s * s),
                          "paint": float(mine.sum()) / max(objects, 1), "window": [x, y, cam.width, cam.height]})
        stem = f"{sheet.views[0]}_{self.sides[sheet.views[0]][0]}" if len(sheet.views) == 1 else f"sheet{j}"
        out = {key: self.work / f"{stem}_{key}.png" for key in ("render", "model", "mask")}
        written = [self.writer.submit(Image.fromarray(picture).save, out[key], compress_level=1)
                   for key, picture in (("render", render), ("model", textured))]
        Image.fromarray(np.round(painted * 255.0).astype(np.uint8)).save(out["mask"], compress_level=1)
        for done in written:
            done.result()
        return {**{key: str(path) for key, path in out.items()}, "size": s, "scale": round(sheet.scale, 4),
                "views": views}

    def view(self, k: int, show: str = "depth") -> dict[str, Any]:
        """View k's pictures, the view alone in its picture."""
        made = self.sheet(self._alone(k), show)
        return {**{key: made[key] for key in ("render", "model", "mask")},
                **{key: made["views"][0][key] for key in ("name", "side", "object", "paint")}}

    def _alone(self, k: int) -> int:
        """The sheet of view k, which must be alone in it."""
        j = self.sheet_of[k]
        if len(self.sheets[j].views) != 1:
            raise RequestError(f"view {k} is painted in a sheet with others: ask for the sheet")
        return j

    # ---- a picture onto the texture --------------------------------------------------------------------------------
    @staticmethod
    def picture(path: Path, side: int) -> np.ndarray:
        """The picture's RGB bytes at `side` px a side."""
        with Image.open(path) as opened:
            return np.asarray(opened.convert("RGB").resize((side, side), Image.Resampling.LANCZOS))

    def to_project(self, picture: Path, colour: Path | None,
                   side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """What of a picture goes onto the model: the picture, or - with `colour` - that picture's colour at this
        one's brightness (keep_colour; levels, not rounded) -> it, the picture as given, and the colour picture."""
        given = self.picture(picture, side)
        if colour is None:
            return given, given, None
        tint = self.picture(colour, side)
        return keep_colour(given, tint), given, tint

    def project(self, k: int, picture: Path, colour: Path | None = None, keep: Path | None = None,
                layer: Path | None = None) -> dict[str, Any]:
        """View k's picture, the view alone in it, onto the model (the module doc's "project")."""
        made = self.paint_sheet(self._alone(k), picture, colour, keep, [{"index": k, "layer": layer}])
        return {"painted": made["painted"], "done": made["done"]}

    def paint_sheet(self, j: int, picture: Path, colour: Path | None = None, keep: Path | None = None,
                    parts: list[dict[str, Any]] | None = None, painting: Path | None = None) -> dict[str, Any]:
        """Sheet j's picture onto the model, each view's window of it onto that view's texels (the module doc's
        "paint"). parts: per view {"index", "layer", "keep", "answer"} (paths, each optional)."""
        sheet = self.sheets[j]
        image, given, tint = self.to_project(picture, colour, sheet.side)
        whole = to_bytes(image)
        self._plan()
        matched, pairs, shows = self._agree(sheet, whole)
        if matched is not whole:                     # (a picture left as it came goes on with its own, unrounded bits)
            image = whole = matched
        asked = {int(part["index"]): part for part in parts or []}
        answer = None                                # Qwen-Image's painting, for the views' own cuts of it
        if painting is not None and any(part.get("answer") for part in asked.values()):
            answer = (tint if tint is not None and painting == colour else given if painting == picture
                      else self.picture(painting, sheet.side))
        # The kept pictures are written while the texels are taken (PIL lets go of the GIL).
        saved = [self.writer.submit(Image.fromarray(whole).save, keep)] if keep is not None else []
        taken = []
        for k in self._painting(sheet):
            chosen, px, py, weight = self.plan[k]
            taken.append((k, chosen, self._in_window(sheet, k, image, px, py), weight))
            saved += self._save_cuts(sheet, k, asked.get(k, {}), whole, answer)
        for done in saved:
            done.result()                            # (a failed write leaves the prop as it was)
        for k in self._painting(sheet):              # painted from here on: what was planned for it is used up
            del self.plan[k], self.reach[k]
        self.shown[shows[0]], self.shown_at[shows[0]] = shows[1], True
        self.settled = False
        views = [self._lay_view(k, chosen, sampled, weight, asked.get(k, {}).get("layer"))
                 for k, chosen, sampled, weight in taken]
        self.views += [{"name": view["name"], "painted": view["painted"]} for view in views]
        self.settle()
        self.matched[j] = pairs
        return {"painted": sum(view["painted"] for view in views), "done": self._done(), "views": views,
                "matched": pairs}

    def _painting(self, sheet: Sheet) -> list[int]:
        """The sheet's views that paint (all but those skipped)."""
        return [k for k in sheet.views if k not in self.skip]

    def _window(self, sheet: Sheet, k: int) -> tuple[slice, slice]:
        """View k's window of its sheet's picture."""
        (x, y), cam = sheet.at[k], self.cameras[k]
        return slice(y, y + cam.height), slice(x, x + cam.width)

    def _in_window(self, sheet: Sheet, k: int, image: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """The sheet's picture at the places (px, py) of view k's window, between its pixels."""
        cam = self.cameras[k]
        return between(image[self._window(sheet, k)].reshape(cam.height * cam.width, -1), cam.width, cam.height,
                       px, py)

    def _save_cuts(self, sheet: Sheet, k: int, asked: dict[str, Any], whole: np.ndarray,
                   answer: np.ndarray | None) -> list[Future[None]]:
        """View k's own cuts of the sheet, where asked: of what goes onto the model ("keep") and of the painting
        ("answer"), its window, black away from the model -> the writes under way."""
        cuts = [(asked.get(key), source) for key, source in (("keep", whole), ("answer", answer))]
        cuts = [(Path(path), source) for path, source in cuts if path and source is not None]
        if not cuts:
            return []
        near = self._near_model(k)[..., None]
        window = self._window(sheet, k)
        return [self.writer.submit(Image.fromarray(np.where(near, source[window], 0)).save, path)
                for path, source in cuts]

    def _lay_view(self, k: int, chosen: np.ndarray, sampled: np.ndarray, weight: np.ndarray,
                  layer: Any) -> dict[str, Any]:
        """View k's picture (`sampled` at its texels `chosen`) over what the paint holds, by its weight against
        the weight held so far: every texel ends as the plan's weights' mix of the views' pictures and the
        model's own texture, whatever the order (and with a view skipped, the others' mix). Its layer, where
        asked: its share alone over the clay grey."""
        texel = self.texel[chosen]
        colours = to_bytes(sampled)
        held = self.held[chosen]
        mix = weight / (held + weight)
        self.held[chosen] = held + weight
        fade = np.flatnonzero(weight < 1.0)
        part = np.flatnonzero(mix < 1.0)
        alone = colours.copy() if fade.size or part.size else colours
        if part.size:
            colours[part] = laid_over(self.paint[texel[part]], sampled[part], mix[part][:, None])
        if fade.size:
            alone[fade] = on_clay(sampled[fade], weight[fade][:, None])
        self.paint[texel] = colours
        self.done[texel] = True
        self.filled[texel] = False                   # (a hole closed after an earlier view: this view's now)
        if layer:                                    # from its own copies: the next picture may be painted meanwhile
            self._layers.append(self.layers.submit(self.write_layer, Path(layer), texel, alone))
        return {"index": k, "name": self.sides[k][0], "painted": int(chosen.size)}

    def _agree(self, sheet: Sheet, image: np.ndarray) -> tuple[np.ndarray, int, tuple[np.ndarray, np.ndarray]]:
        """The sheet's picture (side, side, 3; bytes: what is to go onto the model) in the tone of the pictures
        painted before it -> the picture (the very one given when nothing was matched), the texels it was
        matched over, and what it shows where the earlier pictures showed nothing (texels, colours: they join
        `shown` for the pictures after it, once this one is painted). What this picture shows of each texel its
        cameras can paint (the camera facing it best) is set against what the earlier pictures show there,
        wherever both do: only surface both have painted, never a picture's empty black. tone_table reads the
        change from those pairs, recolour makes it on the model's part of the picture."""
        side = sheet.side
        best = np.full(len(self.texel), -1.0, np.float32)
        place = np.zeros(len(self.texel), np.int64)                 # the pixel of the picture a texel is read at
        near = np.zeros((side, side), dtype=bool)                   # the model in the picture, and a little around
        for k in self._painting(sheet):
            cam = self.cameras[k]
            x, y = sheet.at[k]
            chosen, px, py, facing = self.reach[k]
            col, row = nearest(px, py, cam.width, cam.height)
            better = facing > best[chosen]
            best[chosen[better]], place[chosen[better]] = facing[better], ((y + row) * side + x + col)[better]
            near[self._window(sheet, k)] |= self._near_model(k)
        mine = np.flatnonzero(best >= 0)
        colours = image.reshape(-1, 3)[place[mine]]
        both = self.shown_at[mine]
        pairs = 0
        table = tone_table(colours[both], self.shown[mine[both]]) if self.agree and both.any() else None
        if table is not None:
            image = image.copy()
            image[near] = recolour(image[near], table)
            colours = image.reshape(-1, 3)[place[mine]]
            pairs = int(np.count_nonzero(both))
        return image, pairs, (mine[~both], colours[~both])

    # ---- a repaint of what the artist masked, through the page's camera (omnitexture_repaint) ---------------------
    def look(self, eye: list[float], forward: list[float], up: list[float], tan: float, mask: Path,
             size: int = 1024, pictures: bool = True) -> dict[str, Any]:
        return self.repaints.look(_lens(eye, forward, up, tan, size), mask, pictures)

    def left(self, picture: Path) -> dict[str, Any]:
        return self.repaints.left(self._looked("left"), picture)

    def repaint(self, picture: Path, colour: Path | None = None, keep: Path | None = None,
                layer: Path | None = None, fade: float = 1.0, tone: bool = True) -> dict[str, Any]:
        return self.repaints.repaint(self._looked("repaint"), picture, colour, keep, layer, fade, tone)

    def fades(self, repaints: list[dict[str, Any]], outs: list[str] | None, ahead: int | None = None) -> dict[str, Any]:
        lenses = [_lens(asked["eye"], asked["forward"], asked["up"], float(asked["tan"]), int(asked.get("size", 1024)))
                  for asked in repaints]
        return self.repaints.fades(repaints, lenses, outs, ahead)

    def _looked(self, command: str) -> Look:
        if self.repaints.at is None:
            raise RequestError(f"'{command}' before 'look'")
        return self.repaints.at

    # ---- the texture as painted so far -----------------------------------------------------------------------------
    def pixels(self) -> np.ndarray:
        """The paint as a picture (res x res x 3; a view of it)."""
        return self.paint.reshape(self.res, self.res, 3)

    def mix_in(self, to: np.ndarray, colours: np.ndarray, share: np.ndarray) -> None:
        """`colours` onto the texels `to` (places in the texture) at `share` ((n, 1), 0..1) over what is there."""
        self.paint[to] = laid_over(self.paint[to], colours, share)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """The texture as it is, to restore later."""
        return self.paint.copy(), self.done.copy(), self.filled.copy(), self.settled

    def restore(self, state: tuple[np.ndarray, np.ndarray, np.ndarray, bool]) -> None:
        self.paint[:], self.done[:], self.filled[:], self.settled = state

    def resettle(self) -> None:
        """After colours laid on an already whole texture: every texel of the islands counts as painted, so settle
        closes no hole and only fills the gutter from the new colours."""
        self.done[:] = self.inside
        self.settled = False
        self.settle()

    def _done(self) -> float:
        """The share of the islands' texels painted (only they are ever done)."""
        return np.count_nonzero(self.done) / len(self.texel) if len(self.texel) else float("nan")

    def settle(self) -> None:
        """After a picture: the small holes in what the views painted take the painted colours around them, and
        the gutter around the islands is filled from the islands again (as the bake fills it: grown, then
        push-pull), counting as painted next to painted texels. The next view's pictures read the gutter wherever
        a pixel falls on an island's edge."""
        reach = max(1, round(self.res * HOLE_TEXELS))
        self._close_holes(reach)
        self._fill_gutter()
        res = self.res
        inside, done = self.inside.reshape(res, res), self.done.reshape(res, res)
        self.seen_done = (done | (grow(done, reach + 1) & ~inside)).reshape(-1)
        self.settled = True

    def _close_holes(self, reach: int) -> None:
        """The holes in what the views painted (single texels a view missed, mostly: no wider than twice `reach`)
        take the painted neighbours' colour; larger unpainted parts keep the model's own. Only holes, found from
        the texels the views painted themselves and never from those closed here before: taking any unpainted
        texel near a painted one would creep further over the unpainted surface at every call."""
        res = self.res
        inside = self.inside.reshape(res, res)
        painted = (self.done & ~self.filled).reshape(res, res)
        holes = closing(painted | ~inside, reach) & inside & ~painted
        take, grown = fill.dilate_at(self.pixels(), painted, holes, radius=reach)
        self.paint[take] = to_bytes(grown)
        self.done[take] = True
        self.filled[take] = True

    def _fill_gutter(self) -> None:
        """The gutter around the islands filled from them. The fill keeps the islands' texels as they are:
        rounding all of it rewrites only the gutter."""
        filled = self.gutter_fill(self.pixels(), self.bands)
        paint = np.empty(filled.shape, dtype=np.uint8)

        def rounded(lo: int, hi: int) -> None:
            part = filled[lo:hi]
            paint[lo:hi] = np.clip(np.round(part, out=part), 0, 255, out=part)

        fill.bands(self.bands, self.res, rounded)
        self.paint = paint.reshape(-1, 3)

    def _retint(self) -> float:
        """The model's own texture, where it still shows, takes the tone of the paintings near it (match_tone) ->
        the share of the texture that is mostly its own."""
        if len(self.own_weight) != len(self.texel) or not self.views:
            return float("nan")
        share = np.where(self.held > 0, self.own_weight / np.maximum(self.held, 1e-12), 1.0).astype(np.float32)
        # The holes settle closed hold their painted neighbours' colours: they teach nothing and are not retinted,
        # settle gives them their neighbours' new colours below.
        some = np.flatnonzero(~self.filled[self.texel])
        texel = self.texel[some]
        paint = self.paint[texel]
        retinted = match_tone(self.point[some], self.own[texel], paint, share[some], self.seam_reach() / 2.0)
        if retinted is not paint:
            self.paint[texel] = retinted
            self.settled = False
            self.settle()                            # the holes and the gutter after the islands' new colours
        return float((share[some] > 0.5).sum() / max(len(share), 1))

    # ---- the result ------------------------------------------------------------------------------------------------
    def write_layer(self, path: Path, texel: np.ndarray, colours: np.ndarray) -> None:
        """One picture's share of the texture, alone: its texels' colours over LAYER_GREY, the gutter beside them
        grown from them (no grey lines along the UV seams it painted across)."""
        res = self.res
        layer = np.empty((res * res, 3), dtype=np.uint8)
        layer[:] = LAYER_GREY
        layer[texel] = colours
        mine = np.zeros(res * res, dtype=bool)
        mine[texel] = True
        take, grown = fill.dilate_at(layer.reshape(res, res, 3), mine.reshape(res, res),
                                     ~self.inside.reshape(res, res), radius=LAYER_REACH)
        layer[take] = to_bytes(grown)
        Image.fromarray(layer.reshape(res, res, 3)).save(path, "WEBP", quality=LAYER_QUALITY, method=2)

    def write_model(self, out: Path, pixels: np.ndarray, compress: int = 6) -> None:
        """The 3D model with `pixels` (res x res x 3, bytes) as its base colour, its other maps and the geometry as
        they were. compress: the PNG's zlib level."""
        out.parent.mkdir(parents=True, exist_ok=True)
        replace_files({out: model_with(self.data, self.kind, self.texture, pixels, compress)})

    def hand_paint(self, layer: Path) -> np.ndarray:
        """A step's paint layer (read_layer), read again only when its file changed."""
        stat = layer.stat()
        seen = (stat.st_mtime_ns, stat.st_size)
        if self._hand_paints.get(layer, (None,))[0] != seen:
            self._hand_paints[layer] = (seen, read_layer(layer, self.res, self.res))
        return self._hand_paints[layer][1]

    def step_models(self, out: Any, hand_paint: Any = None, unpainted: Any = None) -> list[tuple[Path, np.ndarray]]:
        """A step's repaint is on the paint (put on again: "fades"): what was painted by hand on the step goes over
        it -> the step's models to write, (path, texture): its model where it has an `out`, and before the hand
        paint its model as made (`unpainted`)."""
        models = []
        if out:
            self.resettle()
            if hand_paint and unpainted:
                models.append((Path(unpainted), self.pixels().copy()))
        if hand_paint:
            lay_hand_paint(self.pixels(), self.hand_paint(Path(hand_paint)))
        if out:
            models.append((Path(out), self.pixels().copy()))
        return models

    def finish(self, out: Path, debug: bool = False, tone: bool = True, hand_paint: Path | None = None,
               unpainted: Path | None = None) -> dict[str, Any]:
        """The module doc's "finish"."""
        res = self.res
        for written in self._layers:                 # (a layer that could not be written fails the prop)
            written.result()
        self._layers.clear()
        own = self._retint() if tone else float("nan")
        if debug:                                    # white: painted; grey: the model's own; black: no island
            done = np.where(self.seen_done, 255, np.where(self.inside, 90, 0)).astype(np.uint8)
            Image.fromarray(done.reshape(res, res)).save(self.work / "done.png")
        pixels = self.pixels() if self.settled else self._with_gutter()
        if hand_paint is not None:
            if unpainted is not None:
                self.write_model(unpainted, pixels)
            pixels = lay_hand_paint(pixels.copy(), self.hand_paint(hand_paint))
        self.write_model(out, pixels)
        sheets = [{"side": sheet.side, "views": [self.sides[k][0] for k in sheet.views],
                   "scale": round(sheet.scale, 3), "matched": self.matched.get(j, 0)}
                  for j, sheet in enumerate(self.sheets)]
        return {"glb": str(out), "stats": {"painted": round(self._done(), 4), "views": self.views, "texture": res,
                                           "own": None if np.isnan(own) else round(own, 4),
                                           "sheets": sheets}}

    def _with_gutter(self) -> np.ndarray:
        """The paint as a picture, its gutter filled from the islands (a texture nothing settled: settle leaves
        the same fill, rounded alike)."""
        res = self.res
        colour = self.pixels().astype(np.float32)
        inside = self.inside.reshape(res, res)
        grown, weight = fill.dilate(colour, inside)
        return to_bytes(np.where(inside[..., None], colour, fill.push_pull(grown, weight)))


def model_with(data: bytes, kind: str, texture: np.ndarray, pixels: np.ndarray, compress: int = 6) -> bytes:
    """The 3D model (`data`, a GLB whose base colour is `texture` as stored, in the format `kind`) with `pixels`
    (h x w x 3, bytes) as its base colour, in that format (WebP at WEBP_QUALITY; PNG at zlib level `compress`) and
    with that texture's alpha; its other maps and the geometry as they were."""
    if texture.shape[2] == 4:
        pixels = np.concatenate([pixels, texture[..., 3:]], axis=2)
    buffer = io.BytesIO()
    picture = Image.fromarray(pixels)
    if kind == "webp":
        picture.save(buffer, "WEBP", quality=WEBP_QUALITY, method=4)
    else:
        picture.save(buffer, "PNG", compress_level=compress)
    return glb.set_base_color_image(data, buffer.getvalue())


# ---- what was painted by hand on a step (the page's marker and clone stamp) -------------------------------------
# A step painted by hand keeps the model as it was made and a paint layer: an RGBA picture of the texture's size,
# alpha 255 where painted. Its model is the one as made with the layer's texels over its texture; put on again (a
# fade, a picture taken off), the layer goes on after the step's own repaint, and the next one onto that.

def read_layer(path: Path, width: int, height: int) -> np.ndarray:
    """A paint layer's RGBA bytes (h x w x 4, writable)."""
    with Image.open(path) as opened:
        layer = np.array(opened.convert("RGBA"))
    if layer.shape[:2] != (height, width):
        raise BakeError(f"The hand painting kept for this step is {layer.shape[1]} x {layer.shape[0]} pixels, not "
                        f"the texture's {width} x {height}.")
    return layer


def lay_hand_paint(pixels: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """What a paint layer holds put over `pixels` (h x w x 3, bytes), in place -> `pixels`."""
    painted = layer[..., 3] != 0
    pixels[painted] = layer[..., :3][painted]
    return pixels


def paint_by_hand(model: Path, layer: Path, patch: Path, width: int, height: int, out: Path) -> dict[str, Any]:
    """The module doc's "hand_paint": the patch into the paint layer, and `out` the model as made (`model`) with
    the layer over its texture, both written together (replace_files). The painted model is never read: its
    WebP would lose a generation at every save."""
    data = model.read_bytes()
    picture, kind = open_base_colour(data)
    if picture.size != (width, height):
        raise BakeError(f"The painting is {width} x {height} pixels, but this model's texture is "
                        f"{picture.width} x {picture.height}. Reload the page and paint again.")
    changed = np.fromfile(patch, np.uint8)
    if changed.size != width * height * 4:
        raise RequestError(f"the patch holds {changed.size} bytes, not {width} x {height} x 4")
    changed = changed.reshape(height, width, 4)
    held = read_layer(layer, width, height) if layer.is_file() else np.zeros((height, width, 4), np.uint8)
    new = changed[..., 3] != 0
    held[new] = changed[new]
    held[new, 3] = 255
    texture = pixels(picture)
    painted = lay_hand_paint(np.array(texture[..., :3]), held)
    kept = io.BytesIO()
    Image.fromarray(held, "RGBA").save(kept, "PNG", compress_level=1)
    replace_files({layer: kept.getvalue(), out: model_with(data, kind, texture, painted)})
    return {"painted": int(np.count_nonzero(held[..., 3])), "texture": [width, height]}


def blur(pictures: list[Path], side: int, out: Path, size: int = 1024, first: int = 3) -> list[str]:
    """The prop's own pictures with their details blurred away, so that Qwen-Image leans on picture 1 (the
    settings' test): each shrunk to `side` px on its longest side, then enlarged back smoothly to its own size (at
    least `size` on its longest side), saved in `out` as <n>-<name>.png (n from `first`: after pictures 1 and 2).
    Never sent small: the engine enlarges an example picture by nearest neighbour, which would give Qwen-Image hard
    pixel blocks instead. A picture no larger than `side` stays as it is (its own path)."""
    from PIL import ImageOps

    out.mkdir(parents=True, exist_ok=True)
    made = []
    for n, src in enumerate(pictures, first):
        with Image.open(src) as opened:
            picture = ImageOps.exif_transpose(opened)       # as the engine's launcher reads it
            picture.load()
        if max(picture.size) <= side:
            made.append(str(src))
            continue
        alpha = picture.mode in ("RGBA", "LA", "PA") or (picture.mode == "P" and "transparency" in picture.info)
        picture = picture.convert("RGBA" if alpha else "RGB")    # (Pillow resizes RGBA premultiplied: no fringes)
        w, h = picture.size
        shrink, enlarge = side / max(w, h), max(1.0, size / max(w, h))
        small = picture.resize((max(1, round(w * shrink)), max(1, round(h * shrink))), Image.Resampling.LANCZOS)
        path = out / f"{n}-{src.stem}.png"
        small.resize((round(w * enlarge), round(h * enlarge)), Image.Resampling.BICUBIC).save(path, compress_level=3)
        made.append(str(path))
    return made


# ---- the commands --------------------------------------------------------------------------------------------------

def _path(request: dict[str, Any], key: str) -> Path | None:
    return Path(request[key]) if request.get(key) else None


def _load(request: dict[str, Any]) -> Prop:
    work = Path(request["work"])
    work.mkdir(parents=True, exist_ok=True)
    return Prop(Path(request["glb"]), work, int(request.get("size", 1024)), request.get("views"),
                request.get("together"), request.get("plan") is not False, request.get("skip"))


def mirror(picture: Path, out: Path) -> str:
    """The picture flipped left to right, saved as `out`, its transparency kept: the prop as its other side shows it."""
    from PIL import ImageOps

    with Image.open(picture) as opened:
        flipped = ImageOps.mirror(ImageOps.exif_transpose(opened))      # as the engine's launcher reads it
    out.parent.mkdir(parents=True, exist_ok=True)
    flipped.save(out)
    return str(out)


def _blur(request: dict[str, Any]) -> dict[str, Any]:
    return {"pictures": blur([Path(p) for p in request["pictures"]], int(request["side"]), Path(request["out"]),
                             int(request.get("size", 1024)))}


def _hand_paint(request: dict[str, Any]) -> dict[str, Any]:
    return paint_by_hand(Path(request["model"]), Path(request["layer"]), Path(request["patch"]),
                         int(request["width"]), int(request["height"]), Path(request["out"]))


# What is asked without a loaded prop: command -> its answer.
WITHOUT_LOAD = {"blur": _blur, "hand_paint": _hand_paint,
                "mirror": lambda request: {"picture": mirror(Path(request["picture"]), Path(request["out"]))}}


# What a loaded prop is asked (the module doc): command -> its answer.
COMMANDS: dict[str, Callable[[Prop, dict[str, Any]], dict[str, Any]]] = {
    "view": lambda prop, r: prop.view(int(r["index"]), str(r.get("show") or "depth")),
    "sheet": lambda prop, r: prop.sheet(int(r["index"]), str(r.get("show") or "depth")),
    "paint": lambda prop, r: prop.paint_sheet(int(r["index"]), Path(r["picture"]), _path(r, "colour"),
                                              _path(r, "keep"), r.get("views"), _path(r, "painting")),
    "project": lambda prop, r: prop.project(int(r["index"]), Path(r["picture"]), _path(r, "colour"),
                                            _path(r, "keep"), _path(r, "layer")),
    "look": lambda prop, r: prop.look(r["eye"], r["forward"], r["up"], float(r["tan"]), Path(r["mask"]),
                                      int(r.get("size", 1024)), r.get("pictures") is not False),
    "left": lambda prop, r: prop.left(Path(r["picture"])),
    "repaint": lambda prop, r: prop.repaint(Path(r["picture"]), _path(r, "colour"), _path(r, "keep"),
                                            _path(r, "layer"), float(r.get("fade") or 1.0),
                                            r.get("tone") is not False),
    "fades": lambda prop, r: prop.fades(list(r["repaints"]), r.get("outs"), r.get("ahead")),
    "finish": lambda prop, r: prop.finish(Path(r["out"]), bool(r.get("debug")), r.get("tone") is not False,
                                          _path(r, "hand_paint"), _path(r, "unpainted")),
}


def _answer(stream, message: dict[str, Any]) -> None:
    stream.write(json.dumps(message) + "\n")
    stream.flush()


def main(stdin=sys.stdin, stdout=sys.stdout) -> int:
    prop: Prop | None = None
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        started = time.perf_counter()
        try:
            request = json.loads(line)
            cmd = request.get("cmd") if isinstance(request, dict) else None
            if cmd == "quit":
                break
            if cmd == "load":
                if prop is not None:
                    prop.close()
                    prop = None                      # (a load that fails leaves no prop: never the closed one)
                prop = _load(request)
                answer = prop.loaded()
            elif cmd in WITHOUT_LOAD:
                answer = WITHOUT_LOAD[cmd](request)
            elif prop is None:
                raise RequestError(f"'{cmd}' before 'load'")
            elif cmd in COMMANDS:
                answer = COMMANDS[cmd](prop, request)
            else:
                raise RequestError(f"unknown command {cmd!r}")
            _answer(stdout, {"ok": True, "seconds": round(time.perf_counter() - started, 2), **answer})
        except BakeError as exc:
            _answer(stdout, {"error": str(exc), "kind": "user"})
        except MemoryError:
            _answer(stdout, {"error": "The computer ran out of memory while painting the texture. Close other "
                                      "programs and try again.", "kind": "user"})
        except Exception as exc:  # a bug: the artist gets one line, the log the traceback
            traceback.print_exc(file=sys.stderr)
            _answer(stdout, {"error": f"Omni-Texture failed ({type(exc).__name__}: {exc}). The details are in "
                                      "code/logs/hub.log.", "kind": "internal"})
    if prop is not None:
        prop.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
