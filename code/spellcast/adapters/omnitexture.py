"""The Omni-Texture stage (stage 3): the 3D model's colour texture improved by Qwen-Image.

Run by the Qwen adapter (QwenAdapter.run_stage for ctx.stage == "omnitexture") with Qwen-Image's server up. Per prop,
launchers/omnitexture_worker.py (imesh venv, CPU) holds the 3D model and its texture. Of its 8 views (the front, the
back, the four diagonals between them, then from above and below) the first VIEW_COUNT are painted - all of them -;
a job's "views" asks for another count.

The views are painted together, so that they agree with each other: the worker lays them out by their outlines in
one square picture, a sheet, all at one scale, and Qwen-Image paints that picture once. A sheet is up to SHEET_MOST
a side and holds the views that each keep most of the size they have alone in a SIZE picture (the worker's FLOOR):
usually four or five, so a prop's eight views take two pictures. A job's "sheet_side" asks for another largest
side and its "together": false has every view painted alone (both the settings' test's).

For each picture - a view, or a sheet of several - Qwen-Image is given:

- picture 1, the model's depth map from there (white near) on black, also the picture a view alone starts from
  (strength STRENGTH over all of it);
- picture 2, the example: the prop's own picture of its front, one of those its 3D model was made from
  (example_of; another side's when it has no front). Not the model with the texture the 3D stage gave it: shown
  that, Qwen-Image copies it instead of painting a better one. A prop without pictures of its own, and a job's "example":
  "model" (the settings' test), still get that one;
- no mask: its job is a good render of the model. The mask is the worker's, which picks what of the painting
  goes onto the model;
- the words (words()): PROMPT, which names the view. A job's "prompt" replaces them (the settings' Omni-Texture
  test): "{view}" in it becomes that phrase, "{side}" the view's side. That test may also have picture 1 show the
  model in lit clay grey, with its own texture or as a normal map (a job's "show"), and send the prop's other
  pictures after picture 2, as they are or blurred (a job's "pictures").
A sheet of several views starts from the model with its own texture, one step in (SHEET_STRENGTH): that picture
is the worker's model.png, kept with the others, whether or not Qwen-Image is shown it.

Each answer goes onto the model as albedo: the albedo add-on takes its lighting and shading off first (Qwen-Image's
server, POST /spellcast/albedo), so the next views are painted next to flat colour, not next to an earlier view's
shadows. What is projected is ALBEDO_COLOUR. Without the add-on, and from the first view it fails on (stats
"warning", which the page shows), the answers go on as they are; a Qwen-Image that has to start again fails the
prop, as for a painting.

The worker projects that onto the surfaces each camera sees facing it, unless another of the job's cameras faces
them better; the seams are blended in 3D on the model's surface. Surfaces no camera saw squarely keep the 3D
model's own texture, retinted at the end to the paintings' tone. A view a better camera leaves next to nothing of
reports a "paint" under MIN_PAINT and is skipped. The result is a copy of the 3D model with the new texture,
omnitexture/ot_NNN/omnitexture.glb, and beside it:

- sheets/<n>/ for a sheet of several views: 1-render.png (picture 1), model.png (the start),
  2-mask.png (where the painting went onto the model, its weight in grey; never sent), request.json (the prompt,
  the settings, the pictures in the order Qwen-Image got them, each view's window), answer.png, and with the
  add-on albedo.png and projected.png (what went onto the model);
- views/<k>-<side>/ per view: its own cut of the painting (answer.png) and of what went onto the model
  (projected.png), and its layer.webp - its share of the texture alone, which the page puts on the model while
  the pointer is over the view's picture. A view painted alone has the sheet's files here too, and the test's
  blurred pictures 3-*.png, 4-*.png...

Between two props of a job the card does not wait for the CPU: while the last view is painted, a second worker reads
the job's next prop's 3D model (a Spare, when the PC has the RAM free), and once the last answer is in the prop lets
go of Qwen-Image (StageContext.release) so the next prop's first view is drawn while this one's last projection and
its GLB are written.

On the result the artist may mask a part and have it painted again through the page's camera (retouch), fade
such a repaint's edge further into what is around it or hide it for a while (refade), hide the painted views
(show_views), take a painted view or a repaint off the model again (take_off), and paint on a step's model by
hand (hand_paint); all but the first without Qwen-Image. What was painted by hand on a step is kept as a layer
beside its model, and goes on again over the step's repaint wherever the step is made again (a fade, a picture
hidden or taken off).

All of that also without an Omni-Texture job: begin_own makes a copy of the 3D model the first step of a result
without views (projects.omnitexture_own), for a prop the artist only wants to touch up. And whatever the artist
painted or had repainted stays on top when a job paints the prop (again): run lays the hand paint over its new
views and puts the repaints on again over that (_own_work, _carry_repaints) - or fails, leaving the prop as it
was, when they cannot be put on: a new result saved without them would have their folders tidied away.
"""

from __future__ import annotations

import asyncio
import ctypes
import dataclasses
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping

from .. import paths, procman, projects
from .base import StageContext, StageError, ToolNeedsRestart

log = logging.getLogger(__name__)

FOLDER = projects.OMNITEXTURE_FOLDER
SIZE = 1024                     # the pictures' side (Qwen-Image's own)
MAX_VIEWS = 8                   # the worker's VIEWS
VIEW_COUNT = MAX_VIEWS          # the views painted, of the worker's in their order: all of them
# How much of the start picture is drawn anew. With the Turbo's 8 fixed steps anything from 0.875 up is ALL of it:
# the drawing starts from pure noise and the start picture does nothing; from 0.87 down it starts one step in and
# the start shows.
STRENGTH = 0.9
# A sheet of several views starts from the model with its own texture, one step in (with the Turbo anything from
# 0.87 down to 0.75 is that step). Qwen-Image's own copies of its pictures are too small in a large sheet to
# place the figures by: from noise a 2048 sheet of four painted 4.5 % of its figures outside the model's outline
# and the head came out larger, from the textured model 1.8 % (measured on a robot prop).
SHEET_STRENGTH = 0.85
MIN_PAINT = 0.01                # a view whose mask covers less of the model than this is skipped
MAX_PICTURES = 8                # the prop's own pictures the test may send along (Qwen-Image takes 10 with pictures 1-2)
CALL_TIMEOUT = 10 * 60.0        # one worker command: loading a big model takes the longest
# what goes onto the model with the albedo add-on: "painting" = Qwen-Image's colour at the albedo's brightness,
# "albedo" = the add-on's answer as it is
ALBEDO_COLOUR = "painting"
# The stage's words. They name the view: without it Qwen-Image paints the back as a front (the depth map cannot
# tell them apart). "Masked", never "the white mask": told that, it draws a white face mask on a character.
PROMPT = ("Make a high-resolution and detailed final render of the object ({view}), add details visible without blur. "
          "Use the {map} as reference{masked}.")
# {masked}: said only while the mask is one of the pictures Qwen-Image sees, which the stage never sends: the
# words end "Use the depthmap as reference."
MASKED = " and generate only where masked"
# {map}: what picture 1 is, by what it shows (SHOWN): the words name the picture Qwen-Image really gets. Told of a
# normal map, it tends to paint a normal map back.
MAP = {"depth": "depthmap", "normal": "normal map", "clay": "lit model", "texture": "textured model"}
# For every view after the front and the back, while Qwen-Image sees a mask with the earlier views' pictures
# around it (as MASKED: never in the stage).
PROMPT_MATCH = "Seamlessly match the surroundings that are outside the masked area."
# What picture 1 shows of the model (a job's "show"; the stage: depth). "normal", its normal map as the camera sees
# it, shows more of the shape (an eye socket is a grey level or three of the depth map), but Qwen-Image follows it
# no better.
SHOWN = ("depth", "clay", "texture", "normal")
PICTURE_SIZES = (512, 256, 128)  # the longest side the test may blur the prop's pictures to (a job's "pictures")
ALONE = ("front", "back")       # the views painted first, with nothing painted in their sight (the worker's VIEWS)
# Which of the prop's own pictures is the example, by the side it shows, in order of choice; a picture of any other
# side (from above, from below) only when it has none of these.
EXAMPLE_SIDES = ("front", "left", "right", "back")
# How large Qwen-Image's own copies of a sheet's pictures are (the launcher's ref_sides). Left to the launcher they
# are alike, 427 px for a 2048 sheet: a figure of 200 px in the depth map's, too small to paint detail from, while
# the example can be small.
# So the depth map's copy shows every view at VIEW_COPY or more (a figure fills 0.95 of its view: ~547 px) and
# every other picture's is EXAMPLE_COPY. The robot's four-view 2048 sheet sat on the model's outline 0.935 with
# the launcher's copies and 0.957 with a 1152 px depth map (painted outside it 2.4 -> 1.6 %). The price: on an
# 8 GB card's budget the prefix cache no longer fits and the sheet takes 118 s instead of 85, in the same video
# memory.
VIEW_COPY = 576
EXAMPLE_COPY = 512
COPY_STEP = 32                  # a copy's sides are multiples of this (the launcher's REFERENCE_STEP), and so are
                                # the sides the lighting pass takes of an example
# A sheet's side: SIZE and up in SHEET_STEP steps (a cell of the worker's layout grid at SIZE, and the engine's 32),
# to SHEET_MOST or what the card draws. Several views are painted in one picture so that they agree with each
# other: painted one by one, the same surface comes out differently from view to view.
# The price is known: Qwen-Image's own copies of pictures 1 and 2 are what the video memory allows whatever the
# sheet (427 px a side for a 2048 one), so each figure in them is small and it follows the model's own texture
# less closely than in a picture of one view.
SHEET_STEP = 128
SHEET_MOST = 2048
NOT_THERE = "The 3D model of this prop is missing. Make it again in the 3D stage, then Omni-Texture it."
# A repaint of what the artist masked on the OmniTextured model (the page's mask marker and "Render Masked"):
# Qwen-Image gets the model as the page's camera sees it with the masked surface in magenta, the model's depth map
# from there, and - the example - the prop's own picture of the side that camera looks from most (example_from),
# its lighting taken off (_example_delit);
# it is told retouch_prompt. Nothing else shows it the model's texture: of the same view unmasked, or of a camera near it, it
# painted back what was under the mask. Its picture goes onto the masked surface as it was painted - no lighting
# taken off, no tone matched: it restores a view of the textured model, lit and toned as the model is already -,
# the edge blended in 3D. A job's "retouch": {"mask": the mask's file in the prop's folder (the texture's texels, white: masked),
# "eye", "forward", "up": the camera in the model's own space, "tan": the tangent of half the angle its square
# picture spans, "wide": only when the page's lens is bent (its "Inverse perspective": half the picture's side at
# depth z is wide + tan * z, tan then 0 or below as well - the worker's BentLens), "words": the artist's own about
# what is masked, only when they gave some (the page's box beside the mask), "smooth": true when that box's "No
# Facets" is on - the depth map is then blurred until the model's facets are gone from it (the worker's
# smooth_depth), so that Qwen-Image paints rounded what the model has in flat faces}. The result is a new ot_NNN
# folder beside the one it was made from, with what Qwen-Image was
# given and answered in its retouch folder, and joins the prop's "steps": the results Ctrl+Z and Ctrl+Y walk
# through (ProjectStore.omnitexture_step).
# The words (retouch_prompt): what to do, then the artist's own about it, then what the pictures are for.
RETOUCH_ASK = "remove the magenta-masked blob."
RETOUCH_REFERENCE = "Use the depthmap and the example image as reference"
WORDS_MOST = 400
# The worker's pictures in the order Qwen-Image gets them, as its "look" names them -> their files in the result's
# retouch folder; the example follows them. The viewport first, the picture the words speak of: with the depth map
# first (the stage's own order) a patch of one colour is left where the mask was far more often.
RETOUCH_PICTURES = (("viewport", "1-viewport.png"), ("depth", "2-depth.png"))
# A repaint is drawn anew where the view is masked, from noise alone, and held to the view everywhere else (the
# engine's own mask: the worker's look_redrawn.png, kept as REDRAWN). Drawn free from the view, the model may give
# back its example picture in place of the view.
REDRAWN = "redrawn.png"
RETOUCH_STRENGTH = 1.0
EXAMPLE_MIRRORED = "3-example-mirrored.png"     # the example as given when the camera looks from behind it (behind)
# The example is the prop's own picture, lit as it was drawn, while the view it helps to restore is flat. So its
# lighting is taken off first, once per picture: kept in the prop's omnitexture/EXAMPLES for the repaints to come.
EXAMPLES = f"{FOLDER}/examples"
# A retouch's camera, as the page sends it and the worker takes it ("wide": a bent lens alone has it).
CAMERA = ("eye", "forward", "up", "tan", "wide")
# Where the camera of each of the Images stage's views looks, in the 3D model's own space: the model faces +z, and
# "left" is the prop's own left side, seen from +x (the worker's cameras: front right, at +60 degrees, is on +x).
SIDE_FORWARD = {"front": (0.0, 0.0, -1.0), "back": (0.0, 0.0, 1.0), "left": (-1.0, 0.0, 0.0), "right": (1.0, 0.0, 0.0)}
# Such a patch is made again with the next seed, up to RETOUCH_TRIES pictures in all: the worker's "left" says how
# much of the mask's shape a picture still has (levels: restored pictures read up to 10, patches from 25), and the
# picture with the least goes onto the model.
RETOUCH_TRIES = 3
MASK_LEFT = 12.0
MASK_LEFT_WARNING = ("Qwen-Image kept the shape of your mask in its picture, so that part may look like a patch. "
                     "Press Ctrl+Z and render it again, or mask a little more around it.")
# The prop keeps a record of each repaint its models hold (omnitexture.repaints: projects.repaint_record): the
# model it made, the model it went onto ("under") and how many times as far as a seam its edge fades (the page's
# slider: refade), and "hidden" while the artist hides it (the page's eye on its picture). A model is the result
# its repaints lie on - the job's, or one made again from it (omnitexture.bases): composed from fewer of its views
# (a view taken off), or the 3D model's own texture while the artist hides the views (show_views) - and those
# repaints (parts_of). The records the kept steps' models are made of are kept (records_after), the others go.
MAX_FADE = 8.0                  # the page's slider ends there, and the worker's fades (tests hold the three together)
FADE_IDLE = 120.0               # s the kept worker stays loaded after a change, for the next one (RAM: ~1 GB)
NO_FADE = ("That repaint cannot be faded again: the model it was painted onto is no longer kept. Mask the place "
           "and render it again.")
NO_HIDE = "That repaint cannot be hidden or shown again: the model it was painted onto is no longer kept."
NO_VIEWS = "The painted views cannot be hidden or shown again: {why}."
HIDDEN_VIEWS = "The painted views are hidden. Show them again first (click the middle of one), then take one off."
# A picture taken off the model shown (the page's × on a painted view or a repaint: take_off).
GONE = "That picture is no longer on the model shown. Reload the page and try again."
ONE_VIEW = ("This is the only view on the model, so it stays. To keep the 3D model's own texture, leave the prop "
            "unchosen.")
NOT_UNDER = "That repaint cannot be taken off: the model it was painted onto is no longer kept."
LATER_GONE = "That picture cannot be taken off: a repaint on the model can no longer be put on again."
NO_RECOMPOSE = "That view cannot be taken off: {why}. Omni-Texture the prop again instead."
MAX_STEPS = 12                  # the results kept for Ctrl+Z (each a 3D model's file)
# Less than this share of the masked texels in the camera's sight (the rest hidden, turned away or off the picture):
# the artist is told, for those stay as they were.
OUT_OF_SIGHT = 0.5
OUT_OF_SIGHT_WARNING = ("Most of what you masked was not in sight from there, so it stayed as it was. Turn the view "
                        "to it, mask it and render again.")
NO_RESULT = "This prop has nothing to repaint yet. Reload the page and mask it again."
NOT_KEPT = ("The new Omni-Texture was painted, but what you had painted again on this prop could not be put onto "
            "it ({why}). The prop is as it was, repaints and all: Omni-Texture it again.")
LEFT_BEHIND = ("{count} of the repaints made on this prop before are not on its new Omni-Texture: their pictures "
               "were no longer kept.")
BAD_RETOUCH = "The repaint's request is incomplete (its mask or its camera is missing). Reload the page and try again."
FINISHING = "Writing the textured 3D model…"
SPARE_SECONDS = 180.0           # a worker started for the job's next prop, not taken this long after its prop's
                                # turn came, is stopped
SPARE_MIN_RAM_MB = 2560         # free RAM it takes to start one (a loaded worker holds ~0.7 GB for a 2048 texture)


class Worker:
    """omnitexture_worker.py in the imesh venv, one command at a time (blocking: call it in a thread)."""

    def __init__(self) -> None:
        argv = [str(paths.venv_python(paths.IMESH)), str(paths.LAUNCHERS / "omnitexture_worker.py")]
        try:
            self.proc = procman.spawn_in_job(argv, paths.LAUNCHERS, paths.clean_env(paths.IMESH),
                                             getattr(subprocess, "CREATE_NO_WINDOW", 0), stdin=subprocess.PIPE)
        except OSError as e:
            raise StageError(f"Spellcast3D could not start the program that paints the texture ({e}). Run "
                             "tools/update.bat, or download Spellcast3D again.") from e

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        timer = threading.Timer(CALL_TIMEOUT, self.stop)
        timer.start()
        try:
            self.proc.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
            for raw in self.proc.stdout:                 # its answer, and its log (stderr) before it
                line = raw.decode("utf-8", "replace").strip()
                try:
                    answer = json.loads(line) if line.startswith("{") else None
                except ValueError:
                    answer = None
                if not isinstance(answer, dict) or not ({"ok", "error"} & answer.keys()):
                    if line:
                        log.info("omnitexture worker: %s", line)
                    continue
                if "error" in answer:
                    log.log(logging.INFO if answer.get("kind") == "user" else logging.ERROR,
                            "omnitexture worker: %s", answer["error"])
                    raise StageError(str(answer["error"]))
                return answer
        except OSError:
            pass
        finally:
            timer.cancel()
        raise StageError("The program that paints the texture stopped unexpectedly. Try again; the details are in "
                         "code/logs/hub.log.")

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        try:
            if self.alive():
                procman.kill_tree(self.proc.pid)
        except OSError:
            pass


@dataclasses.dataclass
class Spare:
    """A worker started for the job's next prop while this one's last view is painted, its load command under way
    (`loading`). That prop's run takes it when it asks for the same load of the same 3D model (signature)."""
    worker: Worker
    request: dict[str, Any]
    signature: tuple[int, int]
    loading: asyncio.Future[dict[str, Any]]
    timer: asyncio.TimerHandle | None = None


_spare: Spare | None = None                 # one job runs at a time: at most one worker waits for its prop
_stopping: set[asyncio.Future[Any]] = set()   # the stops under way (a reference keeps each running)


def _take_spare() -> Spare | None:
    global _spare
    spare, _spare = _spare, None
    if spare is not None and spare.timer is not None:
        spare.timer.cancel()
    return spare


def _stop_spare(spare: Spare | None) -> None:
    """Stop a spare's worker in the background (its load call then ends with an error nobody reads)."""
    if spare is None:
        return
    if spare.timer is not None:
        spare.timer.cancel()
    stop = asyncio.ensure_future(asyncio.to_thread(spare.worker.stop))
    _stopping.add(stop)
    stop.add_done_callback(_stopping.discard)


def _arm_spare(seconds: float | None = None) -> None:
    """The waiting spare, if any, is stopped `seconds` (SPARE_SECONDS) from now unless a run takes it first."""
    if _spare is None:
        return
    if _spare.timer is not None:
        _spare.timer.cancel()
    _spare.timer = asyncio.get_running_loop().call_later(SPARE_SECONDS if seconds is None else seconds, _expire,
                                                         _spare)


def _expire(spare: Spare) -> None:
    global _spare
    if _spare is spare:
        _spare = None
        log.info("Omni-Texture: the worker started for the next prop was not needed; stopped")
        _stop_spare(spare)


def _quiet(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()                  # read: an error shows to the run that takes the spare


def free_ram_mb() -> int | None:
    """The RAM the system has free for a new process (Windows' available physical memory, Linux's MemAvailable),
    or None when it cannot tell."""
    try:
        if os.name == "nt":
            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                            ("page_available", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                            ("virtual_available", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]

            status = Status()
            status.length = ctypes.sizeof(Status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return int(status.available // 2 ** 20)
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, AttributeError):
        return None
    return None


def view_count(params: Mapping[str, Any] | None) -> int:
    """The views a job paints: its "views" (the settings' Omni-Texture test), else VIEW_COUNT."""
    try:
        asked = int((params or {}).get("views") or VIEW_COUNT)
    except (TypeError, ValueError):
        asked = VIEW_COUNT
    return max(1, min(MAX_VIEWS, asked))


def sheet_sides(params: Mapping[str, Any] | None, most: int | None) -> list[int] | None:
    """The sides a sheet of several views may have: SIZE and up in SHEET_STEP steps, to SHEET_MOST or a job's
    "sheet_side", and no larger than the largest picture the card draws (`most`: Qwen-Image's own word, its health
    "sheet_side"; unknown: SIZE alone). A job's "together": false has every view painted alone in its picture
    (None). Both params are the settings' Omni-Texture test's."""
    if (params or {}).get("together") is False:
        return None
    try:
        top = min(int((params or {}).get("sheet_side") or SHEET_MOST), int(most or SIZE), SHEET_MOST)
    except (TypeError, ValueError):
        top = SIZE
    return list(range(SIZE, max(SIZE, top) + 1, SHEET_STEP))


def _load(src: Path, work: Path, views: int, together: list[int] | None = None) -> dict[str, Any]:
    load = {"cmd": "load", "glb": str(src), "work": str(work / "views"), "size": SIZE, "views": views}
    if together:
        load["together"] = together
    return load


async def _start_spare(ctx: StageContext, views: int, together: list[int] | None = None) -> None:
    """A worker for the job's next prop (ctx.upcoming), loading its 3D model while this prop's last view is painted:
    its run takes it (_take_spare). Not without the RAM free for two workers at once; never raises."""
    global _spare
    if ctx.upcoming is None or ctx.cancelled():
        return
    try:
        upcoming = await ctx.upcoming()
        if upcoming is None:
            return
        prop, prop_dir = upcoming
        mesh_rel = (prop.get("mesh") or {}).get("glb")
        src = prop_dir / mesh_rel if mesh_rel else None
        if src is None or not src.is_file():
            return
        free = free_ram_mb()
        if free is not None and free < SPARE_MIN_RAM_MB:
            log.info("Omni-Texture: the next prop's 3D model is read at its turn (%d MB of RAM free)", free)
            return
        work = _next_folder(prop_dir / FOLDER)
        work = work.with_name(work.name + ".tmp")

        def prepare() -> None:              # as run() would: what an interrupted job left goes
            shutil.rmtree(work, ignore_errors=True)
            (work / "views").mkdir(parents=True)

        await asyncio.to_thread(prepare)
        if ctx.cancelled():                 # the job ends: nobody would take it
            return
        worker = await asyncio.to_thread(Worker)
    except Exception as exc:                # (the next prop's run reads its 3D model itself)
        log.info("Omni-Texture: the next prop's 3D model is read at its turn (%s)", exc)
        return
    request = _load(src, work, views, together)     # (one job: its props are painted from as many views)
    log.info("Omni-Texture: the next prop's 3D model is read now, while this prop's last view is painted (%s MB of "
             "RAM free)", free)
    loading = asyncio.ensure_future(asyncio.to_thread(worker.call, request))
    loading.add_done_callback(_quiet)
    old, _spare = _spare, Spare(worker, request, _signature(src), loading)
    _stop_spare(old)
    # Until this prop lets go of Qwen-Image (run's release() starts the SPARE_SECONDS), as long as its last picture
    # and albedo may take: a run that never gets that far still stops it.
    from .qwen import ALBEDO_TIMEOUT, IMAGE_TIMEOUT
    _arm_spare(IMAGE_TIMEOUT + ALBEDO_TIMEOUT + SPARE_SECONDS)


def own_views(ctx: StageContext, prop: Mapping[str, Any]) -> list[tuple[str, Path]]:
    """The prop's own pictures, each with the side it shows: the one picture its 3D model was made from, its
    front; else the views it was made from (mesh/views/<n>_<side>.png - never those an earlier model left there);
    else the pictures of its views (a model of the artist's own has no mesh/views), else its chosen picture."""
    images = prop.get("images") or {}
    alone = projects.made_from_one_picture(dict(prop))
    views = [] if alone else sorted((ctx.prop_dir / "mesh" / "views").glob("*.png"))
    if views:
        return [(path.stem.partition("_")[2], path) for path in views]
    chosen = alone or images.get("chosen")
    listed = [] if alone else [(str(v.get("view") or ""), v.get("src")) for v in images.get("views") or []
                               if isinstance(v, dict)]
    if chosen not in [src for _side, src in listed]:     # the views count only while the chosen picture is one of theirs
        listed = [("front", chosen)]
    first: dict[str, str] = {}              # a picture sliced into several views counts once, as its first
    for side, src in listed:
        if isinstance(src, str) and src:
            first.setdefault(src, side)
    return [(side, path) for src, side in first.items() if (path := ctx.abs(src)).is_file()]


def references(ctx: StageContext, prop: Mapping[str, Any]) -> list[Path]:
    """The prop's own pictures (own_views), as many as Qwen-Image takes."""
    return [path for _side, path in own_views(ctx, prop)][:MAX_PICTURES]


def example_of(ctx: StageContext, prop: Mapping[str, Any]) -> tuple[str, Path] | None:
    """The prop's own picture Qwen-Image is shown as picture 2, with its side: of its front, else by
    EXAMPLE_SIDES; None for a prop without pictures."""
    def choice(view: tuple[str, Path]) -> int:
        return EXAMPLE_SIDES.index(view[0]) if view[0] in EXAMPLE_SIDES else len(EXAMPLE_SIDES)

    return min(own_views(ctx, prop), key=choice, default=None)


def example_from(ctx: StageContext, prop: Mapping[str, Any], forward: list[float]) -> tuple[str, Path] | None:
    """The prop's own picture a repaint is shown, with its side: of the side the page's camera (looking along
    `forward`, in the model's space) looks from most, among those the prop has pictures of (the largest dot product
    with the front, back, left and right directions): the front says little about a part masked from behind. The
    stage's choice (example_of) when none of its pictures shows one of these sides."""
    views = [view for view in own_views(ctx, prop) if view[0] in SIDE_FORWARD]
    if not views:
        return example_of(ctx, prop)
    return max(views, key=lambda view: sum(a * b for a, b in zip(SIDE_FORWARD[view[0]], forward)))


def behind(side: str | None, forward: list[float]) -> bool:
    """The camera, looking along `forward`, is on the far side of the prop from where its picture of `side` was
    taken: it sees the prop facing the other way, as that picture mirrored shows it."""
    towards = SIDE_FORWARD.get(side or "")
    return towards is not None and sum(a * b for a, b in zip(towards, forward)) < 0


def _next_folder(root: Path) -> Path:
    numbers = [int(m.group(1)) for p in root.glob("ot_*") if (m := re.fullmatch(r"ot_(\d+)", p.name))]
    return root / f"ot_{max(numbers, default=0) + 1:03d}"


def model_of(prop: Mapping[str, Any]) -> str:
    """The prop's Qwen-Image model when it is one that draws from a start picture here (else the Turbo)."""
    chosen = ((prop.get("images") or {}).get("settings") or {}).get("qwen_model")
    return chosen if chosen == "full" else "turbo"


def _record(body: Mapping[str, Any], folder: Path, prop_dir: Path) -> dict[str, Any]:
    """request.json: the request as sent, its pictures named as they lie in `folder` (the render and the mask)
    or in the prop's folder (its own pictures), in the order the prompt numbers them."""
    def name(path: str) -> str:
        p = Path(path)
        for root, prefix in ((folder, ""), (prop_dir, "prop: ")):
            try:
                return prefix + p.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                continue
        return p.name

    record = {key: value for key, value in body.items() if key not in ("ref_images", "init_image", "mask_image",
                                                                          "out_dir", "preview")}
    record["pictures"] = [f"picture {i}: {name(p)}" for i, p in enumerate(body["ref_images"], 1)]
    record["start_image"] = name(body["init_image"])
    if body.get("mask_image"):
        record["redrawn"] = name(body["mask_image"])
    record["answer"] = "answer.png"
    return record


def viewed_from(name: str) -> str:
    """How the words name a view: the user's three phrases - the front, the back, and the side for the cameras
    between them - and from above / below for the two slanted ones."""
    if name in ALONE:
        return f"viewed from the {name}"
    return {"top": "viewed from above", "bottom": "viewed from below"}.get(name, "viewed from the side")


def viewed_from_all(names: list[str]) -> str:
    """How the words name a sheet's views (in reading order): "viewed from the front, from the back and from the
    sides" - each phrase once, the side cameras together as "the sides"."""
    sides = sum(viewed_from(name) == "viewed from the side" for name in names)
    said = list(dict.fromkeys(viewed_from(name)[len("viewed "):] for name in names))
    if sides > 1:
        said = ["from the sides" if phrase == "from the side" else phrase for phrase in said]
    return "viewed " + (said[0] if len(said) == 1 else f"{', '.join(said[:-1])} and {said[-1]}")


def words(view: Mapping[str, Any] | list[Mapping[str, Any]], custom: str | None = None,
          mask_seen: bool = False, show: str = SHOWN[0]) -> str:
    """What Qwen-Image is told for a view, or for a sheet's views (a list, in reading order): the stage's words
    (PROMPT), or the job's own (`custom`), with the view named and picture 1 by what it shows (`show`: MAP);
    MASKED, and PROMPT_MATCH but for the front and the back, only when `mask_seen` (the mask is one of the pictures
    it gets)."""
    views = [view] if isinstance(view, Mapping) else list(view)
    names = [str(v.get("name") or "") for v in views]
    text = custom or (f"{PROMPT} {PROMPT_MATCH}" if mask_seen and not set(names) <= set(ALONE) else PROMPT)
    text = text.replace("{masked}", MASKED if mask_seen else "").replace("{map}", MAP.get(show, MAP[SHOWN[0]]))
    sides = [str(v.get("side") or "") for v in views]
    return text.replace("{view}", viewed_from_all(names)).replace("{side}", ", ".join(sides))


def shown(params: Mapping[str, Any] | None) -> str:
    """What picture 1 shows inside the mask: a job's "show" (the settings' Omni-Texture test), else the depth map."""
    asked = (params or {}).get("show")
    return asked if asked in SHOWN else SHOWN[0]


def own_pictures(params: Mapping[str, Any] | None) -> tuple[bool, int | None]:
    """Whether a job sends the prop's other pictures after picture 2, and the longest side they are blurred to: its
    "pictures" (the settings' Omni-Texture test) - "full" (as they are) or one of PICTURE_SIZES; else none are sent
    (the stage)."""
    asked = (params or {}).get("pictures")
    if asked == "full":
        return True, None
    try:
        side = int(asked or 0)
    except (TypeError, ValueError):
        return False, None
    return (True, side) if side in PICTURE_SIZES else (False, None)


def copy_sides(view: Mapping[str, Any], pictures: int) -> list[int] | None:
    """The side of Qwen-Image's own copy of each of a sheet's `pictures`: the depth map's so that every view shows
    at VIEW_COPY or more, the others' EXAMPLE_COPY. None for a view alone: the launcher's own copies are larger."""
    if len(view.get("views") or []) < 2:
        return None
    size = int(view["size"])
    depth = math.ceil(size * VIEW_COPY / (SIZE * float(view.get("scale") or 1.0)) / COPY_STEP) * COPY_STEP
    return [min(depth, size), *[EXAMPLE_COPY] * (pictures - 1)]


def _body(prompt: str, pictures: list[str], start: str, size: int, out_dir: Path, model: str, seed: int,
          preview: bool, strength: float, sides: list[int] | None = None, redrawn: str | None = None) -> dict[str, Any]:
    """What Qwen-Image is asked for one picture of `size` px a side: the words, the pictures in the order it gets
    them (sides: how large its own copy of each is; None: the launcher's choice), and the one it starts from.
    redrawn: a picture, white where it draws anew - held to the start picture elsewhere; None: it paints the whole
    picture, the worker picks what goes onto the model (a view, a sheet)."""
    return {**({"mask_image": redrawn} if redrawn else {}),
            "prompt": prompt, "negative_prompt": "",
            "width": size, "height": size,
            "model": model, "steps": 25 if model == "full" else 8, "cfg_scale": 1.0, "sampler": "euler",
            "seed": seed, "batch_count": 1, "transparent": False, "first_block_cache": None,
            "ref_images": pictures, "ref_sides": sides,
            "init_image": start, "strength": strength,
            "out_dir": str(out_dir.resolve()), "preview": preview,
            # every picture's start is new, so its prefix never comes back: the engine frees it after the
            # drawing instead of keeping it (the decode and the albedo pass then have its room)
            "keep_prefix": False}


def _request(view: Mapping[str, Any], refs: list[Path], out_dir: Path, model: str, seed: int,
             preview: bool, strength: float | None = None, prompt: str | None = None,
             show: str = SHOWN[0], example: Path | None = None) -> dict[str, Any]:
    """The request for one picture: `view` holds its render (showing `show`), model and mask, and for a sheet its
    "size" and the "views" in it. example: picture 2 (None: the view's textured model); refs: the pictures after
    it. strength: a job's own; None: the stage's - a view alone from its render at STRENGTH, a sheet of several
    from the textured model at SHEET_STRENGTH."""
    pictures = [view["render"], *(str(p.resolve()) for p in (example, *refs) if p is not None)]
    if example is None:
        pictures.insert(1, view["model"])
    several = len(view.get("views") or []) > 1
    return _body(words(view.get("views") or view, prompt, view["mask"] in pictures, show), pictures,
                 view["model"] if several else view["render"], int(view.get("size") or SIZE), out_dir, model, seed,
                 preview, strength if strength is not None else SHEET_STRENGTH if several else STRENGTH,
                 copy_sides(view, len(pictures)) if example is not None else None)


@dataclasses.dataclass
class _Asked:
    """What a job asks of every picture of a prop."""
    model: str
    seed: int
    prompt: str | None                  # the job's own words (the settings' test), else the stage's
    show: str                           # what picture 1 shows (SHOWN)
    strength: float | None              # the job's own (the settings' test), else the stage's (_request)
    example: tuple[str, Path] | None    # picture 2: the prop's own picture and its side (None: the textured model)
    refs: list[Path]                    # the prop's other pictures sent after it (the settings' test), else none
    blur: int | None                    # ... blurred to this longest side
    preview: bool


@dataclasses.dataclass
class _Painted:
    """What a run has kept so far (paths in its work folder), and whether the lighting still comes off."""
    albedo: bool
    pictures: list[Path] = dataclasses.field(default_factory=list)
    projected: list[Path] = dataclasses.field(default_factory=list)
    layers: list[Path] = dataclasses.field(default_factory=list)
    sheets: list[tuple[Path, list[Path]]] = dataclasses.field(default_factory=list)    # a sheet's folder, its views'
    warning: str | None = None


async def _pictures_for(worker: Worker, j: int, sheet: Mapping[str, Any], show: str) -> dict[str, Any]:
    """The worker's pictures for the job's j-th picture, a view alone or a sheet of several, as a sheet's answer:
    "render", "model", "mask", "size", "views"."""
    if len(sheet["views"]) > 1:
        return await asyncio.to_thread(worker.call, {"cmd": "sheet", "index": j, "show": show})
    k = int(sheet["views"][0])
    made = await asyncio.to_thread(worker.call, {"cmd": "view", "index": k, "show": show})
    return {**made, "size": SIZE, "views": [{**made, "index": k}]}


async def _given(worker: Worker, asked: _Asked, size: int, inputs: Path) -> list[Path]:
    """The prop's own pictures a picture is sent with: as they are, or the test's blurred copies, kept with the
    picture's other inputs."""
    if asked.blur is None or not asked.refs:
        return asked.refs
    blurred = await asyncio.to_thread(worker.call, {
        "cmd": "blur", "pictures": [str(p) for p in asked.refs], "side": asked.blur, "size": size,
        "out": str(inputs)})
    return [Path(p) for p in blurred["pictures"]]


async def _lighting_off(ctx: StageContext, keep: Path, inputs: Path, folders: dict[int, Path], model: str,
                        project: dict[str, Any], painted: _Painted, told: tuple[str, str, str]) -> None:
    """The albedo add-on takes the lighting off the painting `keep`, and `project` then puts that on. When it
    fails, this painting and the ones after it go on with their shading rather than the painted views be lost
    (an 8 GB card short of memory for the add-on's pass, say); told: how the warning names the picture."""
    from .qwen import albedo

    try:
        light_off = await albedo(ctx, keep, inputs / "albedo.png", model)
    except ToolNeedsRestart:
        raise
    except StageError as exc:
        what, it, them = told
        painted.warning = (f"Taking the lighting off stopped working at {what}, so {it} and the views after {them} "
                           f"went onto the model with their shading. {exc}")
        log.warning("Omni-Texture: %s", painted.warning)
        if ctx.warn is not None:
            ctx.warn(painted.warning)
        painted.albedo = False
        return
    project.update(picture=str(light_off), keep=str(inputs / "projected.png"),
                   colour=str(keep) if ALBEDO_COLOUR == "painting" else None)
    for part, own in zip(project.get("views") or [], folders.values()):
        part["keep"] = str(own / "projected.png")
    painted.projected += [own / "projected.png" for own in folders.values()]


async def _paint(ctx: StageContext, worker: Worker, work: Path, j: int, total: int, made: dict[str, Any],
                 asked: _Asked, painted: _Painted, before_last: Callable[[], None] | None) -> None:
    """The job's j-th picture of `total` drawn by Qwen-Image and put onto the model: `made` holds the worker's
    pictures for it (_pictures_for). What Qwen-Image is given is kept beside its answer (views/<k>-<side>/,
    sheets/<n>/: the module doc). before_last: called once the answer is in, before the worker's last work on
    it (the prop's last picture: the next prop may draw meanwhile)."""
    from .qwen import generate

    views = made["views"]
    alone = len(views) == 1
    k = int(views[0]["index"])
    side = views[0]["side"]
    turn = f" ({j + 1} of {total})" if alone or total > 1 else ""
    them = f"the {side}" if alone else f"the {len(views)} views"
    folders = {int(v["index"]): work / "views" / f"{v['index']}-{v['name']}" for v in views}
    inputs = folders[k] if alone else work / "sheets" / str(j + 1)
    out_dir = inputs / "drawn"
    out_dir.mkdir(parents=True, exist_ok=True)
    for own in folders.values():
        own.mkdir(parents=True, exist_ok=True)
    for key, kept in (("render", "1-render.png"), ("model", "model.png"), ("mask", "2-mask.png")):
        os.replace(made[key], inputs / kept)
        made[key] = str(inputs / kept)
    given = await _given(worker, asked, made["size"], inputs)
    # Qwen-Image's own progress, as this picture's share of the prop's.
    start = f"From {them}" if alone else f"{len(views)} views in one picture"
    sub = dataclasses.replace(ctx, report=lambda f, m: ctx.report(
        (j + 0.9 * max(0.0, min(1.0, f))) / total, f"{start}{turn}: {m}"))
    example = asked.example or (None, None)
    body = _request(made, given, out_dir, asked.model, asked.seed + k, asked.preview, asked.strength, asked.prompt,
                    asked.show, example[1])
    # (the page's captions follow "shown", "example" - the side picture 2 shows - and "pictures_size")
    record = {**_record(body, inputs, ctx.prop_dir), "shown": asked.show, "example": example[0],
              "pictures_size": asked.blur, "painted_where": "2-mask.png"}
    if not alone:                           # where each view lies in the sheet
        record["views"] = [{"name": v["name"], "side": v["side"], "window": v["window"]} for v in views]
        record["scale"] = made.get("scale")
    (inputs / "request.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    result = await generate(sub, body)
    drawn = [Path(p) for p in result.get("images") or [] if Path(p).is_file()]
    if not drawn:
        raise StageError(f"Qwen-Image made no picture of {them}. Try again.")
    keep = inputs / "answer.png"
    os.replace(drawn[0], keep)
    shutil.rmtree(out_dir, ignore_errors=True)
    if alone:
        project = {"cmd": "project", "index": k, "picture": str(keep), "layer": str(inputs / "layer.webp")}
    else:                                   # each view's own cut of the painting, and its layer
        project = {"cmd": "paint", "index": j, "picture": str(keep), "painting": str(keep),
                   "views": [{"index": i, "layer": str(own / "layer.webp"),
                              "answer": str(own / "answer.png")} for i, own in folders.items()]}
        painted.sheets.append((inputs, list(folders.values())))
    painted.pictures += [own / "answer.png" for own in folders.values()]
    painted.layers += [own / "layer.webp" for own in folders.values()]
    if painted.albedo:
        ctx.report((j + 0.92) / total, f"Taking the lighting off {them}{turn}…")
        await _lighting_off(ctx, keep, inputs, folders, asked.model, project, painted,
                            (f"the {side} view", "it", "it") if alone else (them, "they", "them"))
    ctx.report((j + 0.95) / total, f"Painting {them} onto the model{turn}…")
    if before_last is not None:
        before_last()
    await asyncio.to_thread(worker.call, project)


async def run(ctx: StageContext, prop: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """Omni-Texture one prop -> the patch for prop["omnitexture"] (paths relative to the prop folder):
    {"glb", "source_mesh", "views": [the pictures Qwen-Image painted], "projected": [what went onto the model with
    the albedo add-on], "layers": [each view's share of the texture, alone], "stats"}."""
    from .qwen import albedo_ready, sheet_side      # the Qwen adapter imports this module

    if (params or {}).get("retouch") is not None:
        return await retouch(ctx, prop, params)
    spare = _take_spare()               # the worker the job's prop before started for this one, if any
    try:
        mesh_rel = (prop.get("mesh") or {}).get("glb")
        src = ctx.abs(mesh_rel) if mesh_rel else None
        if src is None or not src.is_file():
            raise StageError(NOT_THERE)
        send, blur = own_pictures(params)
        example = None if (params or {}).get("example") == "model" else example_of(ctx, prop)
        others = [path for path in references(ctx, prop) if example is None or path != example[1]]
        asked = _Asked(model=model_of(prop), seed=int((params or {}).get("seed") or 0) or 1234,
                       prompt=str((params or {}).get("prompt") or "").strip() or None, show=shown(params),
                       strength=float((params or {}).get("strength") or 0) or None, example=example,
                       refs=others if send else [], blur=blur, preview=ctx.preview is not None)
        started = time.monotonic()
        before = _signature(src)
        folder = _next_folder(ctx.prop_dir / FOLDER)
        work = folder.with_name(folder.name + ".tmp")
        count = view_count(params)
        together = sheet_sides(params, await sheet_side(ctx))
        load = _load(src, work, count, together)
        if spare is not None and (spare.request != load or spare.signature != before):
            _stop_spare(spare)              # made for another model, or this one was made again since
            spare = None
        if spare is None:
            shutil.rmtree(work, ignore_errors=True)
            (work / "views").mkdir(parents=True)
        painted = _Painted(albedo=await albedo_ready(ctx))
        if not painted.albedo:              # the installer brings it: its files or the engine's pass are missing
            log.warning("Omni-Texture: without the albedo add-on, the painted pictures go onto the model with their "
                        "lighting")
        ctx.report(0.0, "Reading the 3D model…")
        worker = spare.worker if spare is not None else await asyncio.to_thread(Worker)
    except BaseException:
        _stop_spare(spare)              # (None when it was not taken)
        raise
    let_go = False

    def release() -> None:              # the last answer is in: the next prop may draw meanwhile
        nonlocal let_go
        if let_go:
            return
        let_go = True
        _arm_spare()                    # the next prop's turn comes now: its worker waits for it from here
        if ctx.release is not None:
            ctx.release(FINISHING)

    try:
        loaded = await (spare.loading if spare is not None else asyncio.to_thread(worker.call, load))
        # The pictures Qwen-Image paints: a view each, or several views together in a sheet (the worker's layout).
        sheets = loaded.get("sheets") or [{"size": SIZE, "views": [k]} for k in range(len(loaded["views"]))]
        total = len(sheets)
        for j, sheet in enumerate(sheets):
            if ctx.cancelled():
                raise StageError("Cancelled.")
            last = j == total - 1
            made = await _pictures_for(worker, j, sheet, asked.show)
            if last:
                await _start_spare(ctx, count, together)    # the next prop's 3D model is read while this is painted
            if len(made["views"]) == 1 and made["views"][0]["paint"] < MIN_PAINT:
                continue
            await _paint(ctx, worker, work, j, total, made, asked, painted, release if last else None)
        ctx.report(0.98, FINISHING)
        release()
        finish = {"cmd": "finish", "out": str(work / "omnitexture.glb"), "debug": bool((params or {}).get("debug"))}
        hand, repaints, left_behind = _own_work(ctx.prop_dir, prop)
        if hand is not None:            # what the artist painted by hand before goes over the new views
            _link(hand, work / projects.HAND_PAINT)
            finish.update(hand_paint=str(hand), unpainted=str(work / projects.UNPAINTED_MODEL))
        finished = await asyncio.to_thread(worker.call, finish)
    except BaseException as exc:
        await asyncio.to_thread(worker.stop)
        shutil.rmtree(work, ignore_errors=True)
        if ctx.cancelled() or isinstance(exc, asyncio.CancelledError):
            _stop_spare(_take_spare())  # the job ends (or the hub closes): nobody takes the next prop's worker
        else:
            _arm_spare()                # the next prop's turn comes now
        raise
    await asyncio.to_thread(worker.stop)
    for skipped in (work / "views").glob("*_*.png"):         # the pictures of views with nothing left to paint
        skipped.unlink(missing_ok=True)
    os.replace(work, folder)
    result = ctx.rel(folder / "omnitexture.glb")
    carried: dict[str, Any] = {"glb": result, "steps": [], "repaints": []}
    if repaints:                        # ... and the repaints go on again over that, each a step after the result
        try:
            carried = await asyncio.to_thread(_carry_repaints, ctx.prop_dir, result, repaints)
        except BaseException as exc:
            # Saved without them, the new result would be the prop's only one, and the repaints' folders - the
            # artist's work - tidied away with the old one: the prop keeps what it had instead.
            shutil.rmtree(folder, ignore_errors=True)
            if isinstance(exc, StageError):
                raise StageError(NOT_KEPT.format(why=str(exc).rstrip("."))) from exc
            raise
    lost = LEFT_BEHIND.format(count=left_behind) if left_behind else None
    if lost:
        log.warning("Omni-Texture: %s", lost)
        if ctx.warn is not None:
            ctx.warn(" ".join(words for words in (painted.warning, lost) if words))
    ctx.report(1.0, "OmniTextured.")
    if ctx.cancelled():                 # a cancel after the last view: made in full all the same, and saved
        ctx.completed = True
        _stop_spare(_take_spare())      # the job ends: nobody takes the next prop's worker

    def kept(path: Path) -> str:        # (what lay in the work folder lies in the result's now)
        return ctx.rel(folder / path.relative_to(work))

    return {
        **carried,                      # the model shown, the steps Ctrl+Z walks and the repaints on them
        "source_mesh": mesh_rel,
        "views": [kept(p) for p in painted.pictures],
        "projected": [kept(p) for p in painted.projected],      # with the albedo add-on
        "layers": [kept(p) for p in painted.layers],
        # the views painted together: the sheet's folder (what Qwen-Image was given and answered), its views'
        "sheets": [{"folder": kept(sheet), "views": [kept(view) for view in views]}
                   for sheet, views in painted.sheets],
        "stats": {**finished.get("stats", {}), "model": asked.model, "seed": asked.seed,
                  "sheet_side": together[-1] if together else None,      # the largest side the layout could take
                  "albedo": ALBEDO_COLOUR if painted.projected else None,
                  "warning": " ".join(words for words in (painted.warning, lost) if words) or None,
                  "seconds": round(time.monotonic() - started, 1)},
        # The 3D model may have been made again while it was painted: then this is already out of date.
        "stale": _signature(src) != before,
        "bases": [],                    # an Omni-Texture made anew: none of its views taken off
    }


def begin_own(prop_dir: Path, prop: Mapping[str, Any]) -> dict[str, Any]:
    """The prop's 3D model as the first step of a result of its own texture, for the artist to paint on and to
    mask without an Omni-Texture job (the page asks when a tool is taken up on such a prop) -> the patch for
    prop["omnitexture"]: a result without views (projects.omnitexture_own). The model is copied, not linked:
    the 3D stage writes its file anew in place."""
    mesh_rel = (prop.get("mesh") or {}).get("glb")
    src = prop_dir / mesh_rel if isinstance(mesh_rel, str) else None
    if src is None or not src.is_file():
        raise StageError(NOT_THERE)
    folder = _next_folder(prop_dir / FOLDER)
    work = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        shutil.copyfile(src, work / "omnitexture.glb")
        os.replace(work, folder)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return {"glb": (folder / "omnitexture.glb").relative_to(prop_dir).as_posix(), "source_mesh": mesh_rel,
            "views": [], "projected": [], "layers": [], "sheets": [], "stats": None, "stale": False, "painted": False,
            "steps": [], "repaints": [], "bases": []}


def _own_work(prop_dir: Path, prop: Mapping[str, Any]) -> tuple[Path | None, list[dict[str, Any]], int]:
    """What the artist did on the prop's result before a job paints the prop anew, to stay on top of the job's
    views: the hand paint on the result the repaints lie on (its layer's file, or None), those repaints'
    records, oldest first, as far as their pictures are kept, and how many repaints that leaves behind. Nothing
    when that result was made from another 3D model than the prop has now (another shape, another texture
    layout), or is no longer known."""
    block = prop.get("omnitexture") or {}
    root, chain = parts_of(block, block.get("glb"))
    if root is None or not projects.omnitexture_fresh(dict(prop)):
        return None, [], 0
    kept: list[dict[str, Any]] = []
    for record in chain:
        if not _kept_pictures(prop_dir, record["glb"]):
            break                       # (the ones after it went onto a model with it)
        kept.append(record)
    hand = (prop_dir / root).with_name(projects.HAND_PAINT)
    return (hand if hand.is_file() else None), kept, len(chain) - len(kept)


def _carry_repaints(prop_dir: Path, result: str, repaints: list[dict[str, Any]]) -> dict[str, Any]:
    """The repaints a prop had before a job (_own_work) put on again over the job's new `result`, each with its
    fade and its step's hand paint (_put_on_again: the worker alone) -> the patch's "glb", "steps" and
    "repaints": the prop shows the last of them, and Ctrl+Z steps back through them to the result as the job
    made it. The worker is let go at once: a job's next prop needs the memory."""
    line = [result, *(record["glb"] for record in repaints)]
    how = {record["glb"]: _laid(record) for record in repaints}
    try:
        renamed = _put_on_again(prop_dir, line, 1, how, None)
    finally:
        with _fade_lock:
            _drop_session()
    return {"glb": renamed[line[-1]], "steps": [result, *(renamed[old] for old in line[1:])],
            "repaints": _remade(line, renamed, how)}


def retouch_camera(ask: Any) -> dict[str, Any]:
    """A job's "retouch" checked -> {"mask", "eye", "forward", "up", "tan"}, "wide" when the lens is bent, "words"
    when the artist gave some and "smooth" when they asked for no facets (BAD_RETOUCH when it is not one: a
    camera's picture spans an angle, a bent lens's a width at least)."""
    try:
        vectors = {key: [float(v) for v in ask[key]] for key in ("eye", "forward", "up")}
        tan, wide = float(ask["tan"]), float(ask.get("wide") or 0.0)
        mask, words, smooth = ask["mask"], ask.get("words") or "", ask.get("smooth") or False
    except (KeyError, TypeError, ValueError):
        raise StageError(BAD_RETOUCH) from None
    numbers = [v for vector in vectors.values() for v in vector] + [tan, wide]
    if any(len(vector) != 3 for vector in vectors.values()) or not all(math.isfinite(v) for v in numbers) \
            or not -10.0 < tan < 10.0 or wide < 0.0 or (tan <= 0.0 and not wide) \
            or not isinstance(mask, str) or not mask or not isinstance(words, str) or not isinstance(smooth, bool):
        raise StageError(BAD_RETOUCH)
    return {"mask": mask, **vectors, "tan": tan, **({"wide": wide} if wide else {}),
            **({"words": words} if words.strip() else {}), **({"smooth": True} if smooth else {})}


def retouch_prompt(words: str = "") -> str:
    """What Qwen-Image is told for a repaint: RETOUCH_ASK, the artist's own words as a sentence of their own (one
    line, WORDS_MOST characters at most), RETOUCH_REFERENCE."""
    said = " ".join(words.split())[:WORDS_MOST]
    if said and said[-1] not in ".!?":
        said += "."
    return " ".join(part for part in (RETOUCH_ASK, said, RETOUCH_REFERENCE) if part)


def _repaints(block: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The prop's repaint records by the model each made."""
    return {r["glb"]: r for r in block.get("repaints") or [] if isinstance(r, dict) and isinstance(r.get("glb"), str)}


def parts_of(block: Mapping[str, Any], glb: Any) -> tuple[str | None, list[dict[str, Any]]]:
    """What the model `glb` is made of: the result its repaints lie on (None: no longer known), and the records of
    those repaints, oldest first - each went onto the model of the one before it. rules.js madeOf reads the same."""
    records = _repaints(block)
    chain: list[dict[str, Any]] = []
    model = glb
    while model in records and len(chain) < len(records):
        chain.insert(0, records[model])
        model = records[model].get("under")
    return (model if isinstance(model, str) else None), chain


def _base_of(block: Mapping[str, Any], base: str | None) -> dict[str, Any] | None:
    """The record of the result `base` when it was made again from the job's (omnitexture.bases); None: the job's
    own."""
    return next((dict(made) for made in block.get("bases") or []
                 if isinstance(made, dict) and made.get("glb") == base), None)


def views_of(block: Mapping[str, Any], base: str | None) -> list[dict[str, Any]]:
    """The painted views of the result `base`, {"folder", "picture": what went onto the model, "layer"} each:
    those its record lists when it was made again (_base_of: the ones it holds, or - views_hidden - the ones
    hidden on it), else all of the job's."""
    made = _base_of(block, base)
    return _job_views(block) if made is None else [dict(view) for view in made.get("views") or []]


def views_hidden(block: Mapping[str, Any], base: str | None) -> bool:
    """Whether the result `base` is the 3D model's own texture, its painted views hidden (show_views)."""
    return (_base_of(block, base) or {}).get("hidden") is True


def _job_views(block: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The views the job painted, as views_of gives them."""
    def folder(path: str) -> str:
        return path.rsplit("/", 1)[0]

    projected = {folder(p): p for p in block.get("projected") or [] if isinstance(p, str)}
    layers = {folder(p): p for p in block.get("layers") or [] if isinstance(p, str)}
    return [{"folder": folder(v), "picture": projected.get(folder(v), v), "layer": layers.get(folder(v))}
            for v in block.get("views") or [] if isinstance(v, str)]


def records_after(steps: list[str], repaints: list[dict[str, Any]],
                  bases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Of the records given, those the models of `steps` are made of - each step's repaints and the result they
    lie on -, in their order: the patch's "repaints" and "bases"."""
    held: set[str] = set()
    roots: set[str | None] = set()
    for step in steps:
        root, chain = parts_of({"repaints": repaints}, step)
        held.update(r["glb"] for r in chain)
        roots.add(root)
    return {"repaints": [dict(r) for r in repaints if r.get("glb") in held],
            "bases": [dict(b) for b in bases if isinstance(b, dict) and b.get("glb") in roots]}


def line_of(block: Mapping[str, Any]) -> list[str | None]:
    """What a fade of one of the shown model's repaints puts on again: the models it is made of, oldest first -
    the result its repaints lie on (parts_of), then each repaint's -, and after them the steps Ctrl+Y leads on
    to, as long as each is a repaint of the one before."""
    shown = block.get("glb")
    root, chain = parts_of(block, shown)
    line = [root, *(r["glb"] for r in chain)]
    records = _repaints(block)
    steps = [s for s in block.get("steps") or [] if isinstance(s, str)]
    for later in steps[steps.index(shown) + 1:] if shown in steps else []:
        if (records.get(later) or {}).get("under") != line[-1]:
            break
        line.append(later)
    return line


class FadeSession:
    """A worker that keeps one of a prop's models loaded (`model`), for the repaints after it to be put on again
    (_put_on_again: the page's edge fade, its eye on a picture, a picture taken off): loaded once, each repaint's
    look and picture read once, each weight per fade worked out once (the worker's "fades"), so that a change
    costs the blend of the one repaint changed, laying the pictures on and writing the models. It holds memory
    of the PC alone, none of the graphics card's. Stopped after FADE_IDLE seconds unused."""

    def __init__(self, prop_dir: Path, model: str) -> None:
        self.prop_dir, self.model = prop_dir, model
        self.timer: threading.Timer | None = None
        self.scratch = prop_dir / FOLDER / "refade.tmp"
        shutil.rmtree(self.scratch, ignore_errors=True)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.worker = Worker()
        try:
            self.worker.call(_repaint_load(prop_dir / model, self.scratch))
        except BaseException:
            self.close()
            raise

    def serves(self, prop_dir: Path, line: list[str | None], first: int) -> bool:
        """Whether this session's model lies before the repaint at line[first], every repaint between the two
        can be put on again (it puts them on from its model), and its worker still runs."""
        if prop_dir != self.prop_dir or self.model not in line[:first] or not self.worker.alive():
            return False
        return all(_kept_pictures(prop_dir, old) for old in line[line.index(self.model) + 1:first])

    def retexture(self, model: str) -> None:
        """Another model of the prop takes the loaded one's place: the same surface with another texture (the
        painted views hidden, or shown again), so only that is read (the worker's "texture")."""
        self.worker.call({"cmd": "texture", "glb": str(self.prop_dir / model)})
        self.model = model

    def busy(self) -> None:
        """In use: not stopped meanwhile."""
        if self.timer is not None:
            self.timer.cancel()

    def idle(self) -> None:
        """Unused from now: stopped after FADE_IDLE seconds, unless used again meanwhile."""
        self.busy()
        self.timer = threading.Timer(FADE_IDLE, _expire, args=(self,))
        self.timer.daemon = True
        self.timer.start()

    def close(self) -> None:
        self.busy()
        self.worker.stop()
        shutil.rmtree(self.scratch, ignore_errors=True)


# One session is kept, and one refade runs at a time: under this lock the session is chosen, used and left, so
# none is stopped under a call (another prop's fade, or the page readying another repaint, waits its turn).
_fading: FadeSession | None = None
_fade_lock = threading.Lock()


def _session_for(prop_dir: Path, line: list[str | None], first: int, was: str | None = None) -> FadeSession:
    """The session for the repaints from line[first] on, under _fade_lock: the one kept when it serves, else one
    made anew (the model before that repaint loaded). `was`: the model line[0] takes the place of, the same
    surface with another texture - a session kept on that one is given this one's texture."""
    global _fading
    if _fading is not None and was is not None and (_fading.prop_dir, _fading.model) == (prop_dir, was):
        try:
            _fading.retexture(line[0])
        except StageError:
            _drop_session()                     # (loaded anew below)
    if _fading is not None and not _fading.serves(prop_dir, line, first):
        _drop_session()
    if _fading is None:
        _fading = FadeSession(prop_dir, line[first - 1])
    _fading.busy()
    return _fading


def _drop_session() -> None:
    """The kept session stopped, under _fade_lock."""
    global _fading
    session, _fading = _fading, None
    if session is not None:
        session.close()


def _expire(session: FadeSession) -> None:
    """A session's idle time is over: stopped, unless a refade is at work (it leaves the session idle again)."""
    if _fading is session and _fade_lock.acquire(blocking=False):
        try:
            if _fading is session:
                _drop_session()
        finally:
            _fade_lock.release()


def close_fades() -> None:
    """The edge fade's kept worker stopped (the hub shuts down): at once, whatever it is at."""
    session = _fading
    if session is not None:
        session.worker.stop()
    with _fade_lock:
        _drop_session()


def _link(source: Path, target: Path) -> None:
    """`source` at `target` too: a hard link where the drive allows (nothing is copied), else a copy."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _link_folder(source: Path, target: Path, leave: tuple[str, ...]) -> None:
    """`source`'s files in `target` (_link); the names in `leave` are left out (they are written anew - a link
    would write the source's file too)."""
    target.mkdir(parents=True)
    for file in source.iterdir():
        if file.is_file() and file.name not in leave:
            _link(file, target / file.name)


def _fade_of(value: Any) -> float:
    """A repaint's fade, within what the slider and the worker take."""
    return min(max(float(value or 1.0), 1.0), MAX_FADE)


def _laid(record: Mapping[str, Any]) -> dict[str, Any]:
    """How a repaint lies on its model, as its record says and the worker's "fades" takes it: its fade, and
    whether the artist hid it."""
    return {"fade": _fade_of(record.get("fade")), "hidden": record.get("hidden") is True}


def _kept_pictures(prop_dir: Path, step: str) -> bool:
    """Whether the repaint that made `step` still has what puts it on again: its picture, mask and camera."""
    folder = prop_dir / projects.retouch_folder(step)
    return all((folder / name).is_file() for name in (projects.REPAINT_PICTURE, projects.REPAINT_MASK, "request.json"))


def _kept_model(prop_dir: Path, model: Any) -> bool:
    return isinstance(model, str) and (prop_dir / model).is_file()


@dataclasses.dataclass
class _Faded:
    """A step made again (put on again): its folder while it is written, the folder it becomes, and its
    request.json."""
    work: Path
    target: Path
    record: dict[str, Any]


def _ask_fades(prop_dir: Path, line: list[str], base: int, first: int, how: Mapping[str, Mapping[str, Any]],
               picked: str | None, made: dict[str, _Faded] | None) -> tuple[list[dict[str, Any]], list[str | None]]:
    """What the worker's "fades" is asked: every repaint after line[base] as it lies (`how`: _laid) and what was
    painted by hand on its step, and - unless only made ready (no `made`) - a model per repaint from line[first]
    on, each in a new folder numbered after the prop's last, the repaint's pictures and its step's hand paint
    linked into it (`made`, filled as the folders are; the layer of the repaint `picked`, whose fade changes,
    written anew)."""
    asked: list[dict[str, Any]] = []
    outs: list[str | None] = []
    root = prop_dir / FOLDER
    number = int(_next_folder(root).name[3:])
    for old in line[base + 1:]:
        folder = prop_dir / projects.retouch_folder(old)
        record = json.loads((folder / "request.json").read_text(encoding="utf-8"))
        camera = retouch_camera({**(record.get("retouch") or {}), "mask": projects.REPAINT_MASK})
        ask = {"mask": str(folder / projects.REPAINT_MASK), "picture": str(folder / projects.REPAINT_PICTURE),
               "size": SIZE, **how[old], **{key: camera[key] for key in CAMERA if key in camera}}
        hand = (prop_dir / old).with_name(projects.HAND_PAINT)
        if hand.is_file():
            ask["hand_paint"] = str(hand)
        asked.append(ask)
        if made is None or line.index(old) < first:
            outs.append(None)
            continue
        target = root / f"ot_{number:03d}"
        number += 1
        work = target.with_name(target.name + ".tmp")
        made[old] = _Faded(work, target, {**record, "retouch": {**(record.get("retouch") or {}),
                                                                "fade": how[old]["fade"]}})
        shutil.rmtree(work, ignore_errors=True)
        anew = old == picked or not (folder / projects.REPAINT_LAYER).is_file()
        written_anew = ("request.json", projects.REPAINT_LAYER) if anew else ("request.json",)
        _link_folder(folder, work / projects.RETOUCH_FOLDER, written_anew)
        if anew:
            ask["layer"] = str(work / projects.RETOUCH_FOLDER / projects.REPAINT_LAYER)
        if hand.is_file():
            _link(hand, work / projects.HAND_PAINT)
            ask["unpainted"] = str(work / projects.UNPAINTED_MODEL)
        outs.append(str(work / "omnitexture.glb"))
    return asked, outs


def _put_on_again(prop_dir: Path, line: list[str], first: int, how: Mapping[str, Mapping[str, Any]],
                  picked: str | None, write: bool = True, was: str | None = None) -> dict[str, str]:
    """The repaints of `line` - the model they lie on, then the model each made (line_of) - put onto the model
    again from line[first] on, each as it lies (`how`: _laid - its fade; nothing of a hidden one) and its step's
    hand paint over it, the ones before it as they are: by the worker alone, in a session kept for the next
    change (FadeSession) -> {model as it was: model as it is now}. Qwen-Image is not asked: what went onto the
    model is kept with each repaint (in the texture's tone, with its mask and camera).
    Every model from line[first] on becomes a new folder (a model's file is never written twice: the page keeps
    the ones it has shown); its pictures are hard links. `picked`: the repaint whose fade the page's slider sets,
    its other fades worked out ahead. Without `write` only the session is made ready -> {}. `was`: the model
    line[0] takes the place of (_session_for)."""
    made: dict[str, _Faded] = {}
    renamed: dict[str, str] = {}
    with _fade_lock:
        session: FadeSession | None = None
        try:
            session = _session_for(prop_dir, line, first, was)
            base = line.index(session.model)
            asked, outs = _ask_fades(prop_dir, line, base, first, how, picked, made if write else None)
            session.worker.call({"cmd": "fades", "repaints": asked, "outs": outs if write else None,
                                 "ahead": None if picked is None else first - base - 1})
            for old, new in made.items():
                (new.work / projects.RETOUCH_FOLDER / "request.json").write_text(json.dumps(new.record, indent=1),
                                                                        encoding="utf-8")
                os.replace(new.work, new.target)
                renamed[old] = (new.target / "omnitexture.glb").relative_to(prop_dir).as_posix()
        except BaseException as exc:
            for new in made.values():
                shutil.rmtree(new.work, ignore_errors=True)
            if _fading is session:
                # What the worker holds is no longer known - nor, after a failed rename, which model: the caller
                # deletes the one it wrote for this change, and the next change writes another under that name.
                _drop_session()
            if isinstance(exc, (OSError, ValueError, KeyError)):
                raise StageError(f"That repaint could not be put on again ({type(exc).__name__}: {exc}).") from exc
            raise
        finally:
            if session is not None and _fading is session:
                session.idle()
    return renamed


def _remade(line: list[str], renamed: Mapping[str, str],
            how: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The records of the repaints put on again (`renamed`), each onto the model before it in `line` as that is
    now."""
    return [projects.repaint_record(renamed[old], under=renamed.get(before, before), **how[old])
            for before, old in zip(line, line[1:]) if old in renamed]


def _made_again(block: Mapping[str, Any], renamed: Mapping[str, str], repaints: list[dict[str, Any]],
                bases: list[dict[str, Any]]) -> dict[str, Any]:
    """The patch for prop["omnitexture"] once models of its steps were made again under new names (`renamed`),
    with those models' records: "glb", "steps", "repaints", "bases", "stale" as it was. No step is added: Ctrl+Z
    walks the same changes as before."""
    steps = [renamed.get(s, s) for s in block.get("steps") or [] if isinstance(s, str)]
    return {"glb": renamed.get(block.get("glb"), block.get("glb")), "steps": steps,
            **records_after(steps, [*_repaints(block).values(), *repaints], [*(block.get("bases") or []), *bases]),
            "stale": block.get("stale") is True}


def refade(prop_dir: Path, block: Mapping[str, Any], glb: Any, fade: float | None,
           hidden: bool | None = None) -> tuple[dict[str, Any], dict[str, str]] | None:
    """The repaint that made the model `glb` goes onto the model again with its edge faded `fade` times as far
    (the page's slider on a repaint of the model shown) or - `hidden`, the page's eye on it - hidden or shown
    again, and the repaints after it onto that, each as it was (_put_on_again, from the model it went onto) ->
    (the patch for prop["omnitexture"]: _made_again; {model as it was: model as it is now}). A model of another
    step that holds the repaint as it was keeps it so. With neither (the page picked the repaint) only the worker
    is made ready -> None; None too when it is hidden or shown as asked already. NO_FADE / NO_HIDE when the model
    it went onto, or a repaint's kept pictures, are gone."""
    line = line_of(block)
    records = _repaints(block)
    first = line.index(glb) if isinstance(glb, str) and glb in line else 0
    if first == 0 or not _kept_model(prop_dir, line[first - 1]) \
            or not all(old in records and _kept_pictures(prop_dir, old) for old in line[first:]):
        raise StageError(NO_FADE if hidden is None else NO_HIDE)
    how = {old: _laid(records[old]) for old in line[1:]}
    if fade is None and hidden is None:
        # Made ready: a repaint on the model for its fade (the slider's other values worked out ahead), a hidden
        # one to be shown again.
        ahead = None if how[glb]["hidden"] else glb
        how[glb]["hidden"] = False
        _put_on_again(prop_dir, line, first, how, ahead, write=False)
        return None
    if fade is None and hidden == how[glb]["hidden"]:
        return None                             # (it is so already)
    if fade is not None:
        how[glb]["fade"] = _fade_of(fade)
    if hidden is not None:
        how[glb]["hidden"] = hidden
    renamed = _put_on_again(prop_dir, line, first, how, glb if fade is not None else None)
    return _made_again(block, renamed, _remade(line, renamed, how), []), renamed


def _base_from(prop_dir: Path, made: Path, hand: Path | None, link: bool) -> str:
    """A result in a folder of its own: the model `made` - linked, or copied when its file may be written anew
    in place - with the paint layer `hand` over its texture (the worker's "hand_paint" without a patch; `made`
    is then kept beside it as the model as made) -> its path."""
    folder = _next_folder(prop_dir / FOLDER)
    work = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        model = work / (projects.UNPAINTED_MODEL if hand else "omnitexture.glb")
        if link:
            _link(made, model)
        else:
            shutil.copyfile(made, model)
        if hand:
            _link(hand, work / projects.HAND_PAINT)
            worker = Worker()
            try:
                worker.call({"cmd": "hand_paint", "model": str(model), "layer": str(hand),
                             "out": str(work / "omnitexture.glb")})
            finally:
                worker.stop()
        os.replace(work, folder)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return (folder / "omnitexture.glb").relative_to(prop_dir).as_posix()


def show_views(prop_dir: Path, prop: Mapping[str, Any],
               hidden: bool) -> tuple[dict[str, Any], dict[str, str]] | None:
    """The page's eye on the painted views of the model shown - all of them at once: they were painted to go
    together. Hidden, the result they lie on gives way to the 3D model's own texture; shown again, to the result
    as it was made. What was painted by hand on it, and the repaints after it - each as it lies, with its step's
    hand paint (_put_on_again) - go onto the other. As a fade, it is no step of its own -> (the patch for
    prop["omnitexture"]: _made_again; {model as it was: model as it is now}); None when the views are so already.
    Qwen-Image is not asked. NO_VIEWS when the 3D model was made again since, or what it takes is no longer kept."""
    block = prop.get("omnitexture") or {}
    line = line_of(block)
    root, base = line[0], _base_of(block, line[0]) or {}
    views = views_of(block, root)
    if root is None or not views:
        raise StageError(GONE)
    if hidden == (base.get("hidden") is True):
        return None
    records = _repaints(block)
    if not all(old in records and _kept_pictures(prop_dir, old) for old in line[1:]):
        raise StageError(NO_VIEWS.format(why="a repaint on the model can no longer be put on again"))
    hand = (prop_dir / root).with_name(projects.HAND_PAINT)
    record: dict[str, Any] = {"views": views}
    if hidden:
        if not projects.omnitexture_fresh(dict(prop)):
            raise StageError(NO_VIEWS.format(why="the 3D model was made again after this Omni-Texture"))
        # What comes back: the result as it was made (what is painted by hand goes over whichever is shown).
        as_made = (prop_dir / root).with_name(projects.UNPAINTED_MODEL)
        record.update(hidden=True, shown=(as_made if as_made.is_file() else prop_dir / root)
                      .relative_to(prop_dir).as_posix())
        source = prop_dir / str((prop.get("mesh") or {}).get("glb"))
    else:
        source = prop_dir / str(base.get("shown"))
    if not source.is_file():
        raise StageError(NO_VIEWS.format(why="the model that takes their place is no longer kept"))
    # (the 3D stage writes its model anew in place: copied; a result's is never written twice: linked)
    record["glb"] = _base_from(prop_dir, source, hand if hand.is_file() else None, link=not hidden)
    renamed = {root: record["glb"]}
    how = {old: _laid(records[old]) for old in line[1:]}
    over = [record["glb"], *line[1:]]
    try:
        if len(over) > 1:
            renamed.update(_put_on_again(prop_dir, over, 1, how, None, was=root))
    except BaseException:
        shutil.rmtree((prop_dir / record["glb"]).parent, ignore_errors=True)
        raise
    return _made_again(block, renamed, _remade(over, renamed, how), [record]), renamed


def _index_of(view: str) -> int:
    """A painted view's camera, by its folder's name (views/<k>-<name>)."""
    return int(view.rsplit("/", 1)[-1].split("-", 1)[0])


def _painted_with(folder: Path, albedo: Any) -> dict[str, str]:
    """What the job put onto the model of the picture kept in `folder`: the albedo add-on's answer, in the
    painting's colour when the job's ALBEDO_COLOUR was "painting" (stats "albedo"), else the painting itself."""
    if albedo and (folder / "albedo.png").is_file():
        return {"picture": str(folder / "albedo.png"),
                **({"colour": str(folder / "answer.png")} if albedo == "painting" else {})}
    return {"picture": str(folder / "answer.png")}


def _laid_out_otherwise(prop_dir: Path, block: Mapping[str, Any], loaded: Mapping[str, Any]) -> bool:
    """Whether the worker laid the views out otherwise than the job did (its sheets, their sides and each view's
    window in a sheet of several, as the job's stats and the sheets' request.json have them), or the texture is
    another size: the kept pictures would go onto the wrong surfaces."""
    stats = block.get("stats") or {}
    names = loaded.get("views") or []
    laid = [{"side": s["size"], "views": [names[k] for k in s["views"]]} for s in loaded.get("sheets") or []]
    if laid != [{"side": s.get("side"), "views": s.get("views")} for s in stats.get("sheets") or []] \
            or (loaded.get("texture") or [None])[0] != stats.get("texture"):
        return True
    folders = {frozenset(_index_of(v) for v in s["views"]): s["folder"] for s in block.get("sheets") or []}
    for sheet in loaded["sheets"]:
        if len(sheet["views"]) == 1:
            continue
        try:
            record = json.loads((prop_dir / folders[frozenset(sheet["views"])] / "request.json").read_text("utf-8"))
        except (KeyError, OSError, ValueError):
            return True
        if [view.get("window") for view in record.get("views") or []] != [loaded["windows"][k] for k in sheet["views"]]:
            return True
    return False


def _recompose(prop_dir: Path, prop: Mapping[str, Any], views: list[dict[str, Any]], base: str) -> dict[str, Any]:
    """The result `base` composed again from `views` alone (of the painted views it holds, all but one): the
    worker loads the 3D model as the job did - the job's cameras and sheets - and puts each of their pictures on
    as the job did, in the tone of the pictures before it; the views taken off paint nothing, as if their cameras
    were not there, so what only they painted goes to the next best camera, else keeps the model's own texture,
    retinted. What was painted by hand on `base` goes on over that. Qwen-Image is not asked -> the new result's
    record (omnitexture.bases): its model, and per view its folder, what of it went on now and its share of the
    texture now (layer)."""
    block = prop.get("omnitexture") or {}
    stats = block.get("stats") or {}
    sheets = stats.get("sheets")
    if not projects.omnitexture_fresh(prop):
        raise StageError(NO_RECOMPOSE.format(why="the 3D model was made again after this Omni-Texture"))
    if not isinstance(sheets, list) or not sheets:
        raise StageError(NO_RECOMPOSE.format(why="it was made by an older Spellcast3D"))
    job = {_index_of(view["folder"]): view["folder"] for view in _job_views(block)}
    keep = {_index_of(view["folder"]): view["folder"] for view in views}
    # the sides the job's layout could take (a job from before that was kept: its largest sheet's)
    most = int(stats.get("sheet_side") or max(int(sheet.get("side") or SIZE) for sheet in sheets))
    together = None if all(len(sheet.get("views") or []) == 1 for sheet in sheets) \
        else sheet_sides({"sheet_side": most}, most)
    painted_in = {_index_of(view): sheet["folder"] for sheet in block.get("sheets") or [] for view in sheet["views"]}
    if not all((prop_dir / painted_in.get(k, keep[k]) / "answer.png").is_file() for k in keep):
        raise StageError(NO_RECOMPOSE.format(why="its paintings are no longer kept"))
    folder = _next_folder(prop_dir / FOLDER)
    work = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(work, ignore_errors=True)
    (work / "views").mkdir(parents=True)
    load = _load(prop_dir / str((prop.get("mesh") or {}).get("glb")), work,
                 sum(len(sheet.get("views") or []) for sheet in sheets), together)
    worker = Worker()
    try:
        loaded = worker.call({**load, "skip": sorted(set(job) - set(keep))})
        if _laid_out_otherwise(prop_dir, block, loaded):
            raise StageError(NO_RECOMPOSE.format(why="its views were laid out otherwise when it was painted"))
        own = {k: work / "views" / path.rsplit("/", 1)[-1] for k, path in keep.items()}
        for j, sheet in enumerate(loaded["sheets"]):
            mine = [k for k in sheet["views"] if k in keep]
            if not mine:
                continue
            for k in mine:
                own[k].mkdir()
            painting = _painted_with(prop_dir / painted_in.get(mine[0], keep[mine[0]]), stats.get("albedo"))
            parts = [{"index": k, "layer": str(own[k] / "layer.webp"), "keep": str(own[k] / "projected.png")}
                     for k in mine]
            if len(sheet["views"]) > 1:
                worker.call({"cmd": "paint", "index": j, **painting, "views": parts})
            else:
                worker.call({"cmd": "project", **parts[0], **painting})
        finish = {"cmd": "finish", "out": str(work / "omnitexture.glb")}
        hand = (prop_dir / base).with_name(projects.HAND_PAINT)
        if hand.is_file():
            _link(hand, work / projects.HAND_PAINT)
            finish.update(hand_paint=str(hand), unpainted=str(work / projects.UNPAINTED_MODEL))
        worker.call(finish)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    finally:
        worker.stop()
    os.replace(work, folder)

    def kept(path: Path) -> str:
        return (folder / path.relative_to(work)).relative_to(prop_dir).as_posix()

    return {"glb": kept(work / "omnitexture.glb"),
            "views": [{"folder": path, "picture": kept(own[k] / "projected.png"), "layer": kept(own[k] / "layer.webp")}
                      for k, path in keep.items()]}


def _copy(prop_dir: Path, block: Mapping[str, Any],
          model: str) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    """The model `model` once more, in a folder of its own (its file, what was painted by hand on it and a
    repaint's pictures, linked) -> its path and its record: a repaint's like the one it copies, or a result's
    with the views of that one, held or hidden as there."""
    folder = _next_folder(prop_dir / FOLDER)
    work = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for name in ("omnitexture.glb", projects.UNPAINTED_MODEL, projects.HAND_PAINT):
        if (prop_dir / model).with_name(name).is_file():
            _link((prop_dir / model).with_name(name), work / name)
    record = _repaints(block).get(model)
    if record is not None:
        _link_folder(prop_dir / projects.retouch_folder(model), work / projects.RETOUCH_FOLDER, ())
    os.replace(work, folder)
    glb = (folder / "omnitexture.glb").relative_to(prop_dir).as_posix()
    if record is None:
        return glb, None, {**(_base_of(block, model) or {}), "glb": glb, "views": views_of(block, model)}
    layer = str(record.get("layer") or projects.REPAINT_LAYER).rsplit("/", 1)[-1]
    return glb, projects.repaint_record(glb, layer=layer, under=record.get("under"), **_laid(record)), None


def take_off(prop_dir: Path, prop: Mapping[str, Any], view: Any = None, repaint: Any = None,
             whole: bool = False) -> dict[str, Any]:
    """The page's × on a painted view (`view`: its folder) or on a repaint (`repaint`: the model it made) of the
    model shown: a model without that picture becomes one more step after the one shown (steps_after), so Ctrl+Z
    shows the picture again and nothing of it goes while that step is kept -> the patch for prop["omnitexture"]:
    "glb", "steps", "repaints", "bases", "stale" as it was. Qwen-Image is not asked.
    A repaint taken off: the repaints made after it are put on again (as a fade puts them on, each as it lies)
    onto the model it went onto. A view: the result composed again from the others (_recompose), and the
    repaints put on again onto that. With nothing to put on again, the model under the picture taken off is the
    new step, a copy of it in a folder of its own (_copy). GONE when the picture is not on the model shown;
    ONE_VIEW for the only view; HIDDEN_VIEWS for a view while the artist hides them.
    `whole`: the result itself is taken off (the x on its painted views as a whole). Nothing is made: the step
    that follows is projects.TAKEN_OFF, and the steps before it keep their models for Ctrl+Z."""
    block = prop.get("omnitexture") or {}
    records = _repaints(block)
    if whole:
        steps = steps_after(block, projects.TAKEN_OFF)
        return {"glb": projects.TAKEN_OFF, "steps": steps,
                **records_after(steps, [*records.values()], block.get("bases") or []),
                "stale": block.get("stale") is True}
    root, chain = parts_of(block, block.get("glb"))
    repaints = [r["glb"] for r in chain]
    held = views_of(block, root)
    # line: the model without the picture - the result its repaints lie on, then those; from line[first] on they
    # are put on again (the ones before are named too: a worker kept on a model before them serves).
    of_view = repaint is None
    if not of_view and repaint in repaints:
        first = repaints.index(repaint) + 1
        line = [root, *repaints[:first - 1], *repaints[first:]]
        if not _kept_model(prop_dir, line[first - 1]):
            raise StageError(NOT_UNDER)
    elif of_view and isinstance(view, str) and view in [v["folder"] for v in held]:
        if views_hidden(block, root):
            raise StageError(HIDDEN_VIEWS)
        if len(held) == 1:
            raise StageError(ONE_VIEW)
        if root is None:                        # (the repaints under the oldest one known would be lost)
            raise StageError(NO_RECOMPOSE.format(why="the model's oldest repaints are no longer kept"))
        line, first = [root, *repaints], 1
    else:
        raise StageError(GONE)
    if not all(_kept_pictures(prop_dir, old) for old in line[first:]):
        raise StageError(LATER_GONE)
    bases: list[dict[str, Any]] = []
    if of_view:
        bases.append(_recompose(prop_dir, prop, [v for v in held if v["folder"] != view], root))
        line[0] = bases[0]["glb"]
    new: list[dict[str, Any]] = []
    try:
        if len(line) > first:
            how = {old: _laid(records[old]) for old in line[1:]}
            renamed = _put_on_again(prop_dir, line, first, how, None, was=root if of_view else None)
            new, glb = _remade(line, renamed, how), renamed[line[-1]]
        elif bases:
            glb = bases[0]["glb"]
        else:
            glb, record, base = _copy(prop_dir, block, line[-1])
            new, bases = [record] if record else [], [base] if base else []
    except BaseException:
        for composed in bases:                  # (a result composed again for nothing: no step names it)
            shutil.rmtree((prop_dir / composed["glb"]).parent, ignore_errors=True)
        raise
    steps = steps_after(block, glb)
    return {"glb": glb, "steps": steps,
            **records_after(steps, [*records.values(), *new], [*(block.get("bases") or []), *bases]),
            "stale": block.get("stale") is True}


def hand_paint(prop_dir: Path, glb: str, patch: Path, width: int, height: int) -> None:
    """A hand painting of the step `glb` saved (the page's marker and clone stamp: PUT .../omnitexture/paint):
    `patch` - raw RGBA of the texture's size, alpha 0 where nothing changed since the page last saved - goes over
    the step's paint layer (projects.HAND_PAINT), and its model becomes the one as it was made
    (projects.UNPAINTED_MODEL, linked at the first save) with that layer over its texture: one decode, one
    composite, one model written, by the worker alone (its "hand_paint"). The model is rewritten in its own place
    (the page shows it); the kept worker (FadeSession) is let go when that is the model it loaded."""
    model = prop_dir / glb
    unpainted = model.with_name(projects.UNPAINTED_MODEL)
    with _fade_lock:
        if _fading is not None and _fading.prop_dir == prop_dir and _fading.model == glb:
            _drop_session()
    if not unpainted.is_file():
        _link(model, unpainted)                 # (the painted model is written under a new name, then moved)
    worker = Worker()
    try:
        worker.call({"cmd": "hand_paint", "model": str(unpainted), "patch": str(patch), "width": width,
                     "height": height, "layer": str(model.with_name(projects.HAND_PAINT)), "out": str(model)})
    finally:
        worker.stop()


def steps_after(block: Mapping[str, Any], glb: str) -> list[str]:
    """The prop's results to step through once `glb` follows the one it shows now: those up to that one (what was
    taken back with Ctrl+Z before is gone), then `glb`; the newest MAX_STEPS."""
    now = block.get("glb")
    steps = [s for s in block.get("steps") or [] if isinstance(s, str)]
    steps = steps[:steps.index(now) + 1] if now in steps else [now] if isinstance(now, str) else []
    return [*steps, glb][-MAX_STEPS:]


def _repaint_load(src: Path, work: Path) -> dict[str, Any]:
    """The worker's load for a repaint: it looks through the page's camera alone, so nothing is shared out
    between the stage's (one is named: a load has at least one)."""
    return {"cmd": "load", "glb": str(src), "work": str(work), "size": SIZE, "views": 1, "plan": False}


async def _restored(ctx: StageContext, worker: Worker, body: dict[str, Any], seed: int,
                    inputs: Path) -> list[dict[str, Any]]:
    """Qwen-Image's pictures of what the artist masked, a new seed each, until one no longer shows the mask's
    shape (the worker's "left" within MASK_LEFT) or RETOUCH_TRIES are made: kept as rejected-<n>.png in
    `inputs` -> per picture its seed and what it has left of the mask."""
    from .qwen import generate

    tries: list[dict[str, Any]] = []
    for attempt in range(RETOUCH_TRIES):
        if ctx.cancelled():
            raise StageError("Cancelled.")
        body["seed"] = seed + attempt
        words = (f"Painting what you masked, try {attempt + 1} of {RETOUCH_TRIES}\n(the picture before kept the "
                 "mask's shape)") if attempt else "Painting what you masked"
        sub = dataclasses.replace(ctx, report=lambda f, m, words=words: ctx.report(
            0.9 * max(0.0, min(1.0, f)), f"{words}\n{m}"))
        result = await generate(sub, body)
        drawn = [Path(p) for p in result.get("images") or [] if Path(p).is_file()]
        if not drawn:
            raise StageError("Qwen-Image made no picture of what you masked. Try again.")
        kept = inputs / f"rejected-{attempt + 1}.png"
        os.replace(drawn[0], kept)
        judged = await asyncio.to_thread(worker.call, {"cmd": "left", "picture": str(kept)})
        tries.append({"seed": body["seed"], "left": judged["left"]})
        if judged["left"] <= MASK_LEFT:
            break
    return tries


def _example_kept(prop_dir: Path, picture: Path) -> Path:
    """Where a prop's own picture is kept with its lighting taken off: one file per picture as it is now (its
    name, size and time of writing), so a picture drawn anew gets a file of its own - beside the one of the
    picture as it was, which the repaints made with it still name (what Qwen-Image was given)."""
    stat = picture.stat()
    return prop_dir / EXAMPLES / f"{picture.stem}-{stat.st_size:x}-{stat.st_mtime_ns:x}.png"


async def _example_delit(ctx: StageContext, worker: Worker, picture: Path, model: str) -> tuple[Path, bool]:
    """A repaint's example - the prop's own picture `picture` - with its lighting taken off, so that it shows
    the prop as flat as the texture Qwen-Image restores -> (the picture to show, whether its lighting is off).
    Made once per picture and kept (EXAMPLES): the next repaint from that side finds it, one from another side
    makes that side's. Without the add-on, or when the pass fails, the picture goes as it is (the artist is told
    of a failure; nothing is kept, so the next repaint tries again)."""
    from .qwen import albedo, albedo_ready      # the Qwen adapter imports this module

    kept = _example_kept(ctx.prop_dir, picture)
    if kept.is_file():
        return kept, True
    if not await albedo_ready(ctx):
        return picture, False
    ctx.report(0.02, "Taking the lighting off the example picture…")
    ready = kept.with_name(kept.stem + ".fit.png")
    try:
        await asyncio.to_thread(worker.call, {"cmd": "fit", "picture": str(picture.resolve()), "out": str(ready),
                                              "most": SIZE, "step": COPY_STEP})
        made = kept.with_name(kept.stem + ".tmp.png")
        await albedo(ctx, ready, made, model)
        os.replace(made, kept)
    except ToolNeedsRestart:
        raise
    except (StageError, OSError) as exc:
        words = f"The example picture's lighting could not be taken off, so it was shown as it is. {exc}"
        log.warning("Omni-Texture: %s", words)
        if ctx.warn is not None:
            ctx.warn(words)
        return picture, False
    finally:
        for scratch in (ready, kept.with_name(kept.stem + ".tmp.png")):
            scratch.unlink(missing_ok=True)
    return kept, True


async def retouch(ctx: StageContext, prop: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """What the artist masked on the model the prop shows (an Omni-Texture, or the 3D model's own texture:
    begin_own), painted again through the page's camera -> the patch for prop["omnitexture"]: the new "glb", "steps", "repaints" and "bases" (records_after), "stats" (the
    result's own, with "retouch": what was done), "stale" as it was."""
    block = prop.get("omnitexture") or {}
    glb_rel = block.get("glb")
    src = ctx.abs(glb_rel) if glb_rel else None
    if src is None or not src.is_file():
        raise StageError(NO_RESULT)
    ask = retouch_camera(params.get("retouch"))
    camera = {key: ask[key] for key in CAMERA if key in ask}
    mask = ctx.abs(projects.OMNITEXTURE_MASK)
    if ask["mask"] != projects.OMNITEXTURE_MASK or mask is None or not mask.is_file():
        raise StageError(BAD_RETOUCH)
    started = time.monotonic()
    folder = _next_folder(ctx.prop_dir / FOLDER)
    work = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(work, ignore_errors=True)
    inputs = work / projects.RETOUCH_FOLDER
    out_dir = inputs / "drawn"
    out_dir.mkdir(parents=True)
    model = model_of(prop)
    # A seed of its own each time: masked and rendered again, the same place comes out another way.
    seed = int((params or {}).get("seed") or 0) or random.randrange(1, 2 ** 31)
    ctx.report(0.0, "Reading the 3D model…")
    worker = await asyncio.to_thread(Worker)
    try:
        await asyncio.to_thread(worker.call, _repaint_load(src, inputs))
        shutil.copyfile(mask, inputs / projects.REPAINT_MASK)
        look = await asyncio.to_thread(worker.call, {"cmd": "look", "mask": str(inputs / projects.REPAINT_MASK),
                                                     "size": SIZE, **camera, "smooth": "smooth" in ask})
        given = {}
        for key, kept in RETOUCH_PICTURES:
            os.replace(look[key], inputs / kept)
            given[key] = str((inputs / kept).resolve())
        os.replace(look["redrawn"], inputs / REDRAWN)
        # (a prop without pictures of its own: no example)
        example = example_from(ctx, prop, camera["forward"]) or (None, None)
        shown = example[1]
        delit = False
        if shown is not None:
            shown, delit = await _example_delit(ctx, worker, shown, model)
        mirrored = shown is not None and behind(example[0], camera["forward"])
        if mirrored:
            flipped = await asyncio.to_thread(worker.call, {"cmd": "mirror", "picture": str(shown.resolve()),
                                                            "out": str(inputs / EXAMPLE_MIRRORED)})
            shown = Path(flipped["picture"])
        body = _body(str((params or {}).get("prompt") or "").strip() or retouch_prompt(ask.get("words", "")),
                     [*given.values(), *([str(shown.resolve())] if shown else [])],
                     given["viewport"], SIZE, out_dir, model, seed,
                     ctx.preview is not None, RETOUCH_STRENGTH, redrawn=str((inputs / REDRAWN).resolve()))
        tries = await _restored(ctx, worker, body, seed, inputs)
        best = min(range(len(tries)), key=lambda i: tries[i]["left"])
        body["seed"] = seed = tries[best]["seed"]
        answer = inputs / "answer.png"
        os.replace(inputs / f"rejected-{best + 1}.png", answer)
        shutil.rmtree(out_dir, ignore_errors=True)
        record = {**_record(body, inputs, ctx.prop_dir),
                  "retouch": {**camera, "masked": look["masked"], "seen": look["seen"], "from": glb_rel,
                              "example": example[0], "mirrored": mirrored, "delit": delit, "tries": tries,
                              **({"smooth": True} if "smooth" in ask else {})}}
        (inputs / "request.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        project = {"cmd": "repaint", "picture": str(answer), "keep": str(inputs / projects.REPAINT_PICTURE),
                   "layer": str(inputs / projects.REPAINT_LAYER)}
        warnings = [words for told, words in ((look["seen"] < OUT_OF_SIGHT * look["masked"], OUT_OF_SIGHT_WARNING),
                                              (tries[best]["left"] > MASK_LEFT, MASK_LEFT_WARNING)) if told]
        for words in warnings:
            if ctx.warn is not None:
                ctx.warn(words)
        warning = " ".join(warnings) or None
        ctx.report(0.95, "Painting it onto the model…")
        if ctx.release is not None:
            ctx.release(FINISHING)          # the last answer is in: the next job may draw meanwhile
        painted = await asyncio.to_thread(worker.call, project)
        ctx.report(0.98, FINISHING)
        await asyncio.to_thread(worker.call, {"cmd": "finish", "out": str(work / "omnitexture.glb"), "tone": False})
    except BaseException:
        await asyncio.to_thread(worker.stop)
        shutil.rmtree(work, ignore_errors=True)
        raise
    await asyncio.to_thread(worker.stop)
    os.replace(work, folder)
    mask.unlink(missing_ok=True)
    ctx.report(1.0, "Painted again.")
    if ctx.cancelled():                 # a cancel after the picture: made in full all the same, and saved
        ctx.completed = True
    new_glb = ctx.rel(folder / "omnitexture.glb")
    stats = dict(block.get("stats") or {})
    stats["retouch"] = {"folder": ctx.rel(folder / projects.RETOUCH_FOLDER), "painted": painted.get("painted"),
                        "masked": look["masked"], "seen": look["seen"], "seed": seed, "tries": len(tries),
                        "seconds": round(time.monotonic() - started, 1),
                        "warning": warning}
    steps = steps_after(block, new_glb)
    held = [*_repaints(block).values(), projects.repaint_record(new_glb, under=glb_rel)]
    return {"glb": new_glb, "steps": steps, **records_after(steps, held, block.get("bases") or []),
            "stats": stats, "stale": block.get("stale") is True}


def _signature(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return 0, -1
