"""Projects and props on disk: projects/<project>/project.json + projects/<project>/props/<prop_id>/.

A project's folder name is its id in URLs. A named project's folder name IS its name (display names
are sanitised into safe folder names); an untitled one lives in untitled-YYYYMMDD-HHMMSS/ and is shown
as "Untitled" / "Untitled 2" until it is given a name (rename_project). Every change is a
read-modify-write of project.json under one lock, written atomically. projects/.state.json remembers
the project opened last. See ARCHITECTURE.md "Project / prop schema".

A project has at most one DRAFT prop: the Images workbench before it has a picture. Jobs may run on it;
the page's props list shows only the other props. The draft becomes a normal prop (at the end of the
list) the moment it gets its first chosen picture: an import, a picked candidate, or the first picture
of a generation job (_pack).

A prop's views (images.views) are the pictures, or parts of one picture, that show it from its sides: with
two or more, Pixal3D makes one 3D model from all of them (adapters/pixal3d.py). They count only while the
chosen picture is one of theirs (effective_views): another chosen picture is made into 3D alone, and the
views wait for the artist to pick one of theirs again. mesh.source records what the model was made from
(mesh_source), so a 3D model is out of date as soon as that changes.

Saved models go to exports/<project name>/ (spellcast/exports.py); renaming a project renames that
folder too.

projects/<project>/temp/ is scratch space for the tools: a job that needs some gets a folder of its own there
(work_dir) and removes it when it ends; what a stopped tool or a closed hub left behind is swept away later
(clear_work_dirs). Nothing in it belongs to the project: its files are never served (resolve_file).
"""

from __future__ import annotations

import copy
import datetime
import itertools
import json
import logging
import os
import re
import secrets
import shutil
import struct
import threading
import time
import unicodedata
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import IO, Any, Callable, Collection, Iterable, Iterator

from . import paths
from .settings import atomic_write_json

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_THUMB_BYTES = 2 * 1024 * 1024
MAX_PROJECT_NAME = 60
MAX_PROP_NAME = 80
UNTITLED = "Untitled"
NO_IMAGE_YET = "No image yet: import or generate one first."
# The stages, in pipeline order.
STAGE_IDS: tuple[str, ...] = tuple(stage for stage, _tool in paths.STAGES)
# Every stage can have a thumbnail of the prop in the form that stage gave it.
THUMB_STAGES = STAGE_IDS
# The result of a stage that its thumbnail shows: a new one makes the thumbnail out of date.
_THUMB_SOURCE: dict[str, str] = {"images": "chosen", "mesh": "glb", "omnitexture": "glb", "retopo": "obj",
                                 "projection": "glb"}
_STATE_FILE = ".state.json"     # {"last_opened": project id}
_TRASH = ".trash"               # empty untitled projects are moved here, never erased
_TEMP = "temp"                  # projects/<project>/temp/: the tools' scratch space (ProjectStore.work_dir)
# Words that end a prop name taken from a prompt ("wooden barrel with iron bands" -> "Wooden barrel"), and
# prompt parts that describe the picture rather than the thing in it.
_NAME_STOP_WORDS = frozenset({"with", "on", "in", "for", "and", "at", "from", "made", "that", "wearing",
                              "holding", "against", "under", "over", "by", "of"})
_QUALITY_TAGS = frozenset({"masterpiece", "best quality", "high quality", "highest quality", "top quality",
                           "ultra detailed", "highly detailed", "detailed", "highres", "absurdres", "4k", "8k",
                           "hd", "uhd", "hdr", "photorealistic", "realistic", "3d", "3d render", "render",
                           "game asset", "concept art", "white background", "simple background"})
# The UV unwrap of the clean meshes (xatlas's chart cost, 0..1 as Instant Meshes' viewer shows it): the top, its
# largest islands, a little more stretch accepted for them. 0.5 was the default before: in a project that does not
# say it was written since (_UV_DEFAULT_KEY), a prop holding 0.5 takes this one - once, so a 0.5 picked later in
# Instant Meshes' viewer stays.
UV_LENIENCY = 1.0
_OLD_UV_LENIENCY = 0.5
_UV_DEFAULT_KEY = "uv_leniency_default"
# Per editable stage (a stage with a tool): the settings a new prop starts with, and its result keys.
# A new stage needs one entry here (and a tool in paths.STAGES); the stages without a tool get {}.
STAGE_DEFAULTS: dict[str, dict[str, dict[str, Any]]] = {
    "images": {
        # Shared by the image generators: prompt (sent as typed), width, height (the Shape, at 1K), seed,
        # batch_count ("Variations").
        # Forge Neo's own: negative_prompt ("Avoid"), checkpoint, sampler, scheduler, steps,
        # cfg_scale, denoising_strength. Qwen-Image's own: transparent (a picture with its own alpha; off by default,
        # the 3D stage removes an opaque picture's background itself) and qwen_steps
        # (IMAGES_NUMBERS), namespaced so Forge's steps never leak in. Qwen-Image is sampled without guidance, so it
        # has no CFG and no Avoid (adapters/qwen.py). qwen_steps = that adapter's DEFAULT_STEPS, the hub's one place
        # for it (the adapters import this module, so it cannot be imported here; tests/test_qwen_hub.py compares).
        "settings": {"prompt": "", "negative_prompt": "", "checkpoint": None, "sampler": None, "scheduler": None,
                     "steps": 25, "cfg_scale": 6.0, "width": 1024, "height": 1024, "seed": -1,
                     "batch_count": 1, "denoising_strength": 0.65,
                     "transparent": False, "tileable": False, "qwen_steps": 25,
                     # resolution: Qwen-Image's pictures are the Shape's sides times this (1, 1.5 or 2: the panel's
                     # 1K / 1.5K / 2K; adapters/qwen.py RESOLUTIONS)
                     "resolution": 1,
                     # first_block_cache: Qwen-Image's First-Block Cache (adapters/qwen.py): on half of the
                     # refinement steps the engine reuses its middle layers' work, ~35 % faster, the same picture
                     # with slightly less fine detail. A tick, not a number: the share it skips is the engine's.
                     # None = the card decides (the launcher's FIRST_BLOCK_CACHE_AUTO_BELOW: on for the GTX 10xx,
                     # whose pictures take ~2.5 minutes without it, off for newer cards); True / False = the
                     # artist's tick.
                     "first_block_cache": None,
                     # qwen_model: which Qwen-Image 2.1 draws the prop's pictures (QWEN_MODELS): "turbo", the
                     # default for every card - Qwen-Image-2.1-Turbo, its own fixed 8 steps, so qwen_steps and
                     # first_block_cache are not read for it - or "full", the full model at qwen_steps, which the
                     # artist downloads first from the Images page (spellcast/model_downloads.py) - or "flux3",
                     # FLUX.3 Image Klein (not Qwen-Image: the same launcher serves it), its own 16 steps and opaque
                     # pictures (transparent is not read for it), also a download of its own. A prop saved before
                     # there was a choice gets "turbo".
                     "qwen_model": "turbo"},
        # references: the example pictures of "Describe + example" (reference/reference_NNN.<ext>, at most
        # MAX_REFERENCES; Forge uses the first). views: the pictures (or parts of one) that show the prop from each
        # side, for a multi-view 3D model (see set_views). Results like chosen, never settings, so they never pass
        # to the next prop.
        "outputs": {"references": [], "candidates": [], "chosen": None, "views": []},
    },
    "mesh": {
        # resolution: the voxel grid, 1536 ("High") by default, 1024 ("Standard"); multi-view props too.
        # geometry: whose networks make the shape (GEOMETRIES; the page's "Shape"): "pixal3d", Pixal3D's own, or
        # "hi3d", Hi3D v1.0 refining Pixal3D's coarse shape from the front picture (finer carved detail, slower). The
        # texture is Pixal3D's either way. A new prop takes its project's Shape, the one picked last in it (the
        # project's "geometry", Pixal3D until one is: ProjectStore._geometry); a prop saved before there was a choice
        # keeps "pixal3d".
        "settings": {"seed": 0, "resolution": 1536, "decimation_target": 300000, "texture_size": 2048,
                     "geometry": "pixal3d"},
        # imported: {"materials", "warnings", "skipped", "overrides"} when the mesh is the artist's own model.
        # geometry: the one that made the mesh (None: imported, or none yet).
        "outputs": {"glb": None, "source": None, "imported": None, "geometry": None},
    },
    "omnitexture": {
        # chosen: the artist wants this prop's colour texture painted anew (a click on its tile in the Omni-Texture
        # stage). Never inherited by a new prop (_new_prop). Only a chosen prop's fresh result is used (high_poly).
        "settings": {"chosen": False},
        # glb: a copy of the 3D model with the new texture (omnitexture/ot_NNN/...), made from source_mesh; views:
        # the pictures the image generator painted; projected: what went onto the model with the lighting taken off
        # (the albedo add-on; empty without it); layers: per view, its share of the texture alone (the page shows
        # one on the model while the pointer is over its picture); sheets: the views painted together in one
        # picture - per sheet its folder (what the image generator was given and answered) and its views' folders.
        # steps: the result, the repaints of what the artist masked on it and the pictures taken off it again
        # (each a glb in a folder of its own), oldest first: Ctrl+Z and Ctrl+Y on the page walk them
        # (omnitexture_step), "glb" is the one shown. A new mesh.glb makes it stale (apply_stage_patch).
        # repaints: what the steps' models hold of them (adapters/omnitexture.py: each the model it made, the model
        # it went onto - "under" -, picture, layer, fade, "hidden" when the artist hid it); bases: the results made
        # again from the job's - composed from fewer of the views (a view taken off), each its model and the views
        # it holds; or, "hidden", the 3D model's own texture while the artist hides the painted views: the views
        # it lists are not on it, "shown" is the model that holds them.
        # A result without views is the 3D model's own texture (omnitexture_own): the artist masked or painted
        # the prop without an Omni-Texture job. painted: a hand painting was saved into one of the result's models.
        "outputs": {"glb": None, "source_mesh": None, "views": [], "projected": [], "layers": [], "sheets": [],
                    "steps": [], "repaints": [], "bases": [], "stale": False, "stats": None, "painted": False},
    },
    "retopo": {
        # method: the tool that makes the prop's clean meshes (RETOPO_METHODS; TriFlow until the artist picks
        # another: DEFAULT_RETOPO_METHOD). sizes: how many thousand vertices each of them gets, one clean mesh per
        # size (RETOPO_MAX_SIZES at most, the boxes on the prop's tile). The rest of the first two lines
        # are Instant Meshes' viewer's settings: its polygon count (target_vertices), rosy/posy (the Output mesh:
        # 4/4 quads, 6/3 triangles, 2/4 quad-dominant), extrinsic (Smooth flow) and align_to_boundaries (Follow
        # borders), and an Accept there stores the viewer's values of all of them (adapters/imesh.py
        # VIEWER_SETTINGS). lato2_* and triflow_* are LATO.2's and TriFlow's (RETOPO_NUMBERS).
        "settings": {"method": paths.TRIFLOW, "sizes": [2], "target_vertices": 5000, "pure_quad": False,
                     "crease_angle": -1, "smooth_iter": 2, "uv_leniency": UV_LENIENCY, "rosy": 4, "posy": 4,
                     "extrinsic": True, "align_to_boundaries": False,
                     "lato2_seed": 0, "triflow_quad_ratio": 0.95, "triflow_seed": 0},
        # variants: the clean meshes `method` made from the 3D model `source`, [{"size", "obj", "stats"}], one per
        # size (retopo/<size>k/<the tool's file>; _variants_patch). obj and stats: the variant the rest of the
        # app uses - the one picked, else the first - and accepted: the artist picked it (or it is the only size).
        # kept (only while there are some): {method: the clean meshes that method made, set aside while the prop is
        # set to another method} (switch_retopo_method).
        "outputs": {"obj": None, "method": None, "source": None, "session_id": None, "stale": False,
                    "accepted": False, "variants": []},
    },
    "projection": {
        # quality = supersampling 1/2/3 (Fast/Good/Best); ray_distance None = automatic, else % of the diagonal.
        # Checked by adapters/projection.py; keep in sync with launchers/spellbake/settings.py.
        "settings": {"texture_size": 2048, "quality": 2, "ray_distance": None, "ao": True, "height": False,
                     "hard_edge_angle": 60, "tangent_basis": "mikk", "normal_y": "opengl",
                     "repack_uvs": True, "export_obj": True, "export_fbx": True},
        "outputs": {"glb": None, "fbx": None, "obj": None, "maps": [], "source_mesh": None,
                    "source_retopo": None, "stale": False, "stats": None},
    },
}
EDITABLE_STAGES: tuple[str, ...] = tuple(STAGE_DEFAULTS)
DEFAULT_STAGE_SETTINGS: dict[str, dict[str, Any]] = {k: v["settings"] for k, v in STAGE_DEFAULTS.items()}
_STAGE_OUTPUTS: dict[str, dict[str, Any]] = {k: v["outputs"] for k, v in STAGE_DEFAULTS.items()}
# The result key of a stage whose change makes an existing projection (bake) out of date.
_PROJECTION_INPUTS: dict[str, str] = {"mesh": "glb", "omnitexture": "glb", "retopo": "obj"}
# omnitexture/ot_NNN folders kept per prop after a new Omni-Texture: the newest and the one before it (the bakes
# keep two the same way: launchers/imesh_server.py KEEP_BAKES).
KEEP_OMNITEXTURES = 2
OMNITEXTURE_FOLDER = "omnitexture"            # in a prop's folder: its results, each in a folder ot_NNN
OMNITEXTURE_MASK = f"{OMNITEXTURE_FOLDER}/mask.png"     # what the artist masked, until its repaint job takes it
# A step that is no model: the result taken off its prop as a whole (the page's x on its painted views). While it is
# the step shown the prop has no result - it shows, bakes and saves its 3D model - and Ctrl+Z steps back to it.
TAKEN_OFF = ""
_OMNITEXTURE_DIR = re.compile(r"^ot_(\d+)$")
# A repaint keeps its pictures beside the model it made, in ot_NNN/retouch/. The prop's record of it names two.
RETOUCH_FOLDER = "retouch"
REPAINT_PICTURE = "projected.png"               # what went onto the model, as the camera saw it
REPAINT_LAYER = "layer.webp"                    # its share of the texture alone: the page shows it on the model
REPAINT_MASK = "mask.png"                       # what the artist masked
# A step painted by hand on the page (PUT .../omnitexture/paint) keeps beside its model the model as it was made
# and what the hand painted: its model is the one as made with the paint over its texture.
UNPAINTED_MODEL = "unpainted.glb"
HAND_PAINT = "hand-paint.png"                   # RGBA of the texture's size, alpha 255 where painted by hand

# The retopology methods (paths.STAGE_TOOLS): a prop's retopo.settings.method picks the tool that makes its clean
# meshes, and retopo.method says which one made the clean meshes it has.
RETOPO_METHODS: tuple[str, ...] = paths.STAGE_TOOLS["retopo"]
# The method every new prop starts with (_new_prop: it is not inherited from the prop before), and the one of a prop
# that names none (rules.js retopoMethod says the same). A prop that picked keeps its own.
DEFAULT_RETOPO_METHOD = paths.TRIFLOW
# The method that made a clean mesh which does not say (made before there were other methods).
FIRST_RETOPO_METHOD = paths.IMESH
# A prop's sizes (retopo.settings.sizes): thousands of vertices, one clean mesh each, at most RETOPO_MAX_SIZES of
# them, each from RETOPO_MIN_SIZE to RETOPO_MAX_SIZE with up to RETOPO_SIZE_DECIMALS decimals (0.25 = 250 vertices).
# rules.js RETOPO_SIZES says the same (tests/test_web.py compares them).
RETOPO_MAX_SIZES = 3
RETOPO_MIN_SIZE = 0.1
RETOPO_MAX_SIZE = 99
RETOPO_SIZE_DECIMALS = 2
# The clean meshes stay with the prop while it is set to another method (retopo.kept[method], see
# switch_retopo_method), so going back to its method shows them again without making them again: only those whose
# every variant has fewer vertices than this (their stats say how many). The same number as rules.js
# RETOPO_KEEP_VERTICES.
RETOPO_KEEP_VERTICES = 50_000
# What the clean meshes are in prop.retopo: these keys move to retopo.kept[method] and back.
_RETOPO_RESULT_KEYS: tuple[str, ...] = ("obj", "method", "source", "stale", "accepted", "stats", "saved_from",
                                        "variants")
# LATO.2's and TriFlow's numbers in retopo.settings: (whole number?, lowest, highest, the page's name for it).
# Keep in sync with adapters/lato2.py and adapters/triflow.py, which check them again.
RETOPO_NUMBERS: dict[str, tuple[bool, float, float, str]] = {
    "lato2_seed": (True, 0, 2 ** 31 - 1, "LATO.2's Seed"),
    "triflow_quad_ratio": (False, 0.0, 1.0, "TriFlow's Quads"),
    "triflow_seed": (True, 0, 2 ** 31 - 1, "TriFlow's Seed"),
}
# Retopo settings that are none any more, dropped from a project when it is read and from what a page sends: LATO.2's
# vertices and TriFlow's triangles, which the sizes replace.
_RETIRED_RETOPO_SETTINGS: tuple[str, ...] = ("lato2_vertices", "triflow_faces")
# Qwen-Image's numbers in images.settings, the same way. Keep in sync with adapters/qwen.py, which checks them again.
IMAGES_NUMBERS: dict[str, tuple[bool, float, float, str]] = {
    "qwen_steps": (True, 1, 100, "Steps"),
    "resolution": (False, 1, 2, "Resolution"),
}
# The built-in image generator's models (images.settings.qwen_model; adapters/qwen.py QWEN_MODELS checks it again),
# the default first: Qwen-Image 2.1's two (install/models.py QWEN_MODELS) and FLUX.3 Image Klein.
QWEN_MODELS: tuple[str, ...] = ("turbo", "full", "flux3")
# Images settings that are none any more, dropped from a project when it is read and from what a page sends.
# qwen_cfg: Qwen-Image's guidance (6.0 in every project: no page ever showed it) until it was sampled Qwen's own
# way, without any. A prop that still has it was saved under that recipe, whose default was 20 steps: see
# _normalize_images_settings. object_mode: "3D-ready framing", which added words about one centred object to every
# prompt; prompts go to the image generators as the artist typed them now. from_sketch: the Sketch tab (until
# 2026-10-31), whose painting was the one picture a generation was shown; the artist paints on Render Images' viewport
# now ("Re-Render Viewport": the job's from_viewport).
_RETIRED_IMAGES_SETTINGS: tuple[str, ...] = ("qwen_cfg", "object_mode", "from_sketch")
# A job's own, never saved with the prop: the picture "Re-Render Viewport" starts from (saved, it would be re-rendered
# in every later generation), how far the page showed it slid (adapters/qwen.py viewport_shift), and the picture a
# Smart Mask was brushed over (adapters/images_common.py masked_change).
_JOB_IMAGES_SETTINGS: tuple[str, ...] = ("from_viewport", "viewport_shift", "viewport_mask_of")
# The 3D stage's geometries (mesh.settings.geometry; adapters/pixal3d.py sends it, launchers/pixal3d_server.py
# GEOMETRIES checks it again), in the page's order. rules.js GEOMETRIES says the same.
GEOMETRIES: tuple[str, ...] = ("hi3d", "pixal3d")
DEFAULT_GEOMETRY = "pixal3d"            # a new project's Shape for its props (a pick changes the project's: default_geometry)
# The geometry of a prop that names none (or none there is): saved before there was a choice, so Pixal3D made it.
FIRST_GEOMETRY = "pixal3d"
_QWEN_STEPS_WITH_GUIDANCE = 20
MAX_REFERENCES = 10             # example pictures of a prop (Qwen-Image takes up to 10)
# The picture the page's "Re-Render Viewport" sends when the artist painted marks over it, or painted on the empty
# viewport (set_viewport): one file, replaced; a generation's from_viewport names it (or one of the prop's own
# pictures when there are no marks).
VIEWPORT_FILE = "viewport/viewport.png"
# Smart Mask's upload beside it: where the artist brushed, white on black, the picture's size; one file, replaced.
# Read by a job that names the picture it was brushed over (viewport_mask_of).
VIEWPORT_MASK = "viewport/mask.png"
MAX_VIEWPORT_BYTES = 16 * 1024 * 1024
# The artist's painting on a bake (PUT .../projection/paint): the base colour map as raw RGBA, 4096 px at most.
MAX_PAINT_SIZE = 4096
MAX_PAINT_BYTES = MAX_PAINT_SIZE * MAX_PAINT_SIZE * 4 + 1024
BAKED_AGAIN = "This prop was baked again meanwhile, so the painting on the old textures was not saved."
NOT_A_STEP = "That step of this prop's Omni-Texture is no longer kept, so the painting was not saved on it."
# A bake's own base colour map, kept beside it (maps/) before the first painting is saved over it: one copy, never
# replaced, gone with the bake's folder. Putting it back as maps/basecolor.png takes the painting off.
UNPAINTED_BASECOLOR = "basecolor.unpainted.png"


def retopo_method(settings: Any) -> str:
    """The retopology method retopo.settings choose (DEFAULT_RETOPO_METHOD when they name none that exists)."""
    method = settings.get("method") if isinstance(settings, dict) else None
    return method if method in RETOPO_METHODS else DEFAULT_RETOPO_METHOD


def chosen_tool(stage: str, settings: Any) -> str | None:
    """The tool a prop's `stage` settings choose to run it (retopo: its method), or None when the stage's tools
    are not chosen per prop."""
    return retopo_method(settings) if stage == "retopo" else None


def _setting_number(key: str, value: Any) -> int | float:
    """One of RETOPO_NUMBERS or IMAGES_NUMBERS checked -> the value (a whole number as int). ProjectError in artist
    words."""
    whole, low, high, name = RETOPO_NUMBERS.get(key) or IMAGES_NUMBERS[key]
    if whole:
        problem = f"{name} must be a whole number from {low:,} to {high:,}."
        ok = isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    else:
        problem = f"{name} must be a number from {low:g} to {high:g}."
        ok = isinstance(value, (int, float)) and value == value      # not NaN
    if isinstance(value, bool) or not ok or not low <= value <= high:
        raise ProjectError(problem)
    return int(value) if whole else float(value)


def _retopo_size(value: Any) -> int | float | None:
    """A size as the hub keeps it - rounded to RETOPO_SIZE_DECIMALS decimals, a whole one an int (2, never 2.0: the
    folders of older projects stay as they are) - or None when `value` is no number from RETOPO_MIN_SIZE to
    RETOPO_MAX_SIZE. Every size the hub takes in goes through here, so sizes compare as numbers everywhere after."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    size = round(float(value), RETOPO_SIZE_DECIMALS)
    if not RETOPO_MIN_SIZE <= size <= RETOPO_MAX_SIZE:        # NaN too
        return None
    return int(size) if size.is_integer() else size


def check_retopo_sizes(value: Any) -> list[int | float]:
    """retopo.settings.sizes checked -> the sizes (_retopo_size), each once, in the order given (an empty list too:
    the artist erased every box). ProjectError (400) in the artist's words."""
    sizes = [_retopo_size(v) for v in value] if isinstance(value, list) else None
    if sizes is None or len(sizes) > RETOPO_MAX_SIZES or None in sizes:
        raise ProjectError(f"A prop has up to {RETOPO_MAX_SIZES} sizes, each a number of thousands of vertices from "
                           f"{RETOPO_MIN_SIZE:g} to {RETOPO_MAX_SIZE:g}. Type them again in the boxes on its tile.")
    return list(dict.fromkeys(sizes))


def retopo_sizes(settings: Any) -> list[int | float]:
    """The sizes retopo.settings ask for, in the order of the prop's boxes ([] when they are not valid)."""
    try:
        return check_retopo_sizes(settings.get("sizes") if isinstance(settings, dict) else None)
    except ProjectError:
        return []


def retopo_size_name(size: float) -> str:
    """A size as the page, the progress words and the folders write it: 0.25 -> "0.25k", 2 -> "2k", 12.5 -> "12.5k".
    rules.js retopoSizeName writes the same (tests/test_web.py compares them)."""
    return f"{size:g}k"


def retopo_size_of(vertices: Any) -> int | float:
    """The size a clean mesh of `vertices` vertices stands for - its thousands to two significant digits, within the
    sizes (740 -> 0.74, 5,234 -> 5.2, 22,053 -> 22, 96 -> 0.1): one made before there were sizes, or shaped in
    Instant Meshes' viewer. One that does not say: the default size."""
    if isinstance(vertices, bool) or not isinstance(vertices, (int, float)) or vertices != vertices:
        return DEFAULT_STAGE_SETTINGS["retopo"]["sizes"][0]
    thousands = float(f"{vertices / 1000:.2g}")
    size = _retopo_size(min(max(thousands, RETOPO_MIN_SIZE), RETOPO_MAX_SIZE))
    assert size is not None       # within the sizes
    return size


def check_retopo_settings(changes: dict[str, Any]) -> dict[str, Any]:
    """Retopo settings a page sends (PATCH), checked -> a copy with the numbers as the hub stores them. Only the
    method, the sizes and LATO.2's and TriFlow's numbers are checked here; Instant Meshes checks its own when it
    runs. A setting that is none any more (_RETIRED_RETOPO_SETTINGS, from a page loaded before the update) is left
    out. ProjectError (400) in the artist's words."""
    out = {key: value for key, value in changes.items() if key not in _RETIRED_RETOPO_SETTINGS}
    if "method" in out and out["method"] not in RETOPO_METHODS:
        raise ProjectError(f"The retopology method must be one of: {', '.join(RETOPO_METHODS)}. "
                           f"Reload the page and pick it again.")
    if "sizes" in out:
        out["sizes"] = check_retopo_sizes(out["sizes"])
    for key in [key for key in RETOPO_NUMBERS if key in out]:
        out[key] = _setting_number(key, out[key])
    return out


def _normalize_retopo_settings(settings: dict[str, Any], old_uv_default: bool = False) -> None:
    """An older or hand-edited project's retopo settings, repaired in place: a method, sizes or LATO.2 / TriFlow
    number that is not valid becomes the default; the settings that are none any more go. `old_uv_default`: the
    project is from before UV_LENIENCY was the default."""
    defaults = DEFAULT_STAGE_SETTINGS["retopo"]
    for key in _RETIRED_RETOPO_SETTINGS:
        settings.pop(key, None)
    if settings.get("method") not in RETOPO_METHODS:
        settings["method"] = defaults["method"]
    try:
        settings["sizes"] = check_retopo_sizes(settings.get("sizes"))
    except ProjectError:
        settings["sizes"] = list(defaults["sizes"])
    for key in RETOPO_NUMBERS:
        try:
            settings[key] = _setting_number(key, settings.get(key))
        except ProjectError:
            settings[key] = defaults[key]
    # The page never showed the leniency, and a new prop copies the last one's settings, so the old default
    # would stay in a project for good.
    if old_uv_default and settings.get("uv_leniency") == _OLD_UV_LENIENCY:
        settings["uv_leniency"] = UV_LENIENCY


def check_images_settings(changes: dict[str, Any]) -> dict[str, Any]:
    """Images settings a page sends (PATCH), checked -> a copy with the numbers as the hub stores them. Only
    Qwen-Image's own settings are checked here (transparent, first_block_cache, qwen_model, IMAGES_NUMBERS); the
    generators check the rest when they run. A setting that is none any more (_RETIRED_IMAGES_SETTINGS, from a page
    loaded before the update) is left out, whatever its value, and so is a job's own (_JOB_IMAGES_SETTINGS).
    ProjectError (400) in the artist's words."""
    out = {key: value for key, value in changes.items()
           if key not in _RETIRED_IMAGES_SETTINGS and key not in _JOB_IMAGES_SETTINGS}
    if "transparent" in out and not isinstance(out["transparent"], bool):
        raise ProjectError('"Transparent background" must be on or off. Reload the page and set it again.')
    if "tileable" in out and not isinstance(out["tileable"], bool):
        raise ProjectError('"Tileable" must be on or off. Reload the page and set it again.')
    cache = out.get("first_block_cache")
    if cache is not None and not isinstance(cache, bool):
        raise ProjectError('"Faster drawing" must be on or off. Reload the page and set it again.')
    if "qwen_model" in out and out["qwen_model"] not in QWEN_MODELS:
        raise ProjectError("The model must be Turbo, Full or FLUX.3 Klein. Reload the page and pick it again.")
    if "use_references" in out and not isinstance(out["use_references"], bool):
        raise ProjectError("Describe or Describe + example must be picked. Reload the page and pick it again.")
    for key in [key for key in IMAGES_NUMBERS if key in out]:
        out[key] = _setting_number(key, out[key])
    return out


def _normalize_images_settings(settings: dict[str, Any]) -> None:
    """An older or hand-edited project's images settings, repaired in place like _normalize_retopo_settings; the
    settings that are none any more (_RETIRED_IMAGES_SETTINGS) are dropped, so a new prop does not inherit them.
    negative_prompt stays whichever generator runs: it is Forge Neo's "Avoid", and Qwen-Image does not read it."""
    defaults = DEFAULT_STAGE_SETTINGS["images"]
    # A prop saved while Qwen-Image was sampled with guidance (it still has qwen_cfg) at that recipe's default steps
    # follows the default of today's recipe: every prop stored the default, and a new prop takes the last one's
    # settings, so the old number would otherwise stay in the project for good. Any other number is the artist's
    # and stays. Once the project is saved, qwen_cfg is gone and its steps are whatever the artist sets.
    if "qwen_cfg" in settings and settings.get("qwen_steps") == _QWEN_STEPS_WITH_GUIDANCE:
        settings["qwen_steps"] = defaults["qwen_steps"]
    for key in _RETIRED_IMAGES_SETTINGS:
        settings.pop(key, None)
    for key in ("transparent", "tileable"):
        if not isinstance(settings.get(key), bool):
            settings[key] = defaults[key]
    if not isinstance(settings.get("first_block_cache"), bool):
        settings["first_block_cache"] = defaults["first_block_cache"]       # None: the card decides
    # qwen_model: a prop from before there was a choice (or a value that is none) draws with the Turbo, the default
    if settings.get("qwen_model") not in QWEN_MODELS:
        settings["qwen_model"] = defaults["qwen_model"]
    # use_references: false = the page's "Describe" mode, which keeps the example pictures but makes the pictures
    # from the words alone (images_common.reference_files); missing = they go along ("Describe + example").
    if "use_references" in settings and not isinstance(settings["use_references"], bool):
        del settings["use_references"]
    for key in IMAGES_NUMBERS:
        try:
            settings[key] = _setting_number(key, settings.get(key))
        except ProjectError:
            settings[key] = defaults[key]


def default_geometry(project_geometry: Any = None) -> str:
    """A new prop's geometry: its project's (`project_geometry`: the project's "geometry", the Shape picked last in
    it), Hi3D only where its code is on this PC; else DEFAULT_GEOMETRY, Pixal3D's, which this Spellcast3D can always
    make. So Hi3D sticks within a project once picked there (until Pixal3D is picked again) and never beyond it: another
    project starts on Pixal3D (before 2026-11-03 the Shape picked last stuck for the whole installation)."""
    if project_geometry == "hi3d" and paths.hi3d_code_here():
        return project_geometry
    return DEFAULT_GEOMETRY


def mesh_geometry(settings: Any) -> str:
    """The geometry mesh.settings choose (GEOMETRIES): FIRST_GEOMETRY when they name none that exists (a prop saved
    before there was a choice; rules.js meshGeometry says the same)."""
    geometry = settings.get("geometry") if isinstance(settings, dict) else None
    return geometry if geometry in GEOMETRIES else FIRST_GEOMETRY


def check_mesh_settings(changes: dict[str, Any]) -> dict[str, Any]:
    """Mesh settings a page sends (PATCH), checked -> a copy. Only the geometry is checked here; the 3D generator
    checks the numbers when it runs. ProjectError (400) in the artist's words."""
    out = dict(changes)
    if "geometry" in out and out["geometry"] not in GEOMETRIES:
        raise ProjectError("The shape must be made by Hi3D or Pixal3D. Reload the page and pick it again.")
    return out


def retouch_folder(glb: str) -> str:
    """The folder the repaint that made the step `glb` keeps its pictures in (relative to the prop's, as `glb`)."""
    return f"{glb.rsplit('/', 1)[0]}/{RETOUCH_FOLDER}"


def repaint_record(glb: str, fade: float = 1.0, layer: str = REPAINT_LAYER, under: str | None = None,
                   hidden: bool = False) -> dict[str, Any]:
    """What a prop keeps of the repaint that made the model `glb` (omnitexture.repaints): that model, the model
    it went onto (`under`; None: no longer known), the picture that went on, the `layer` the page shows on the
    model while the pointer is over that picture, and how many times as far as a seam the repaint's edge fades
    into what is around it. `hidden`: the artist hid it (the eye on its picture) - it keeps its place among the
    repaints and puts nothing on: its model is the one it went onto, with what was painted by hand on its step.
    The model's FILE is there for a step and for the model the prop shows; a repaint put on again for a later
    one's sake has the record and its pictures alone (adapters/omnitexture.py _put_on_again)."""
    folder = retouch_folder(glb)
    return {"glb": glb, "under": under, "picture": f"{folder}/{REPAINT_PICTURE}", "layer": f"{folder}/{layer}",
            "fade": fade, **({"hidden": True} if hidden else {})}


def _step_before(steps: list[str], step: str) -> str | None:
    """The step before `step`: what a repaint recorded before "under" was went onto (they followed each other)."""
    return steps[steps.index(step) - 1] if step in steps[1:] else None


def check_omnitexture_settings(changes: dict[str, Any]) -> dict[str, Any]:
    """Omni-Texture settings a page sends (PATCH), checked -> a copy. ProjectError (400) in the artist's words."""
    out = dict(changes)
    if "chosen" in out and not isinstance(out["chosen"], bool):
        raise ProjectError("A prop is either chosen for Omni-Texture or not. Reload the page and click it again.")
    return out


def omnitexture_fresh(prop: dict[str, Any]) -> bool:
    """The prop's Omni-Texture was made from the 3D model it has now (a new mesh.glb makes it stale; Pixal3D always
    writes the same path, so the flag says what the path cannot). rules.js omnitextureFresh gives the same."""
    block = prop.get("omnitexture") or {}
    if not block.get("glb") or block.get("stale"):
        return False
    source = block.get("source_mesh")
    return not source or source == (prop.get("mesh") or {}).get("glb")


def omnitexture_own(prop: dict[str, Any]) -> bool:
    """The prop's result is its 3D model's own texture: no Omni-Texture job painted a view onto it, only the artist
    did - by hand, or by masking a part and having that painted again (adapters/omnitexture.py begin_own).
    rules.js omniOwn gives the same."""
    block = prop.get("omnitexture") or {}
    return bool(block.get("glb")) and not block.get("views")


def omnitexture_used(prop: dict[str, Any]) -> bool:
    """The Projection stage bakes the prop's result and Export all saves it in place of the 3D model: a fresh
    Omni-Texture of a prop chosen for one (unchosen, it is kept but not used), or the 3D model's own texture once
    the artist painted or repainted something on it, chosen or not. rules.js omnitextureUsed gives the same."""
    block = prop.get("omnitexture") or {}
    if not omnitexture_fresh(prop):
        return False
    if omnitexture_own(prop):
        return block.get("painted") is True or bool(block.get("repaints"))
    return (block.get("settings") or {}).get("chosen") is True


def high_poly(prop: dict[str, Any]) -> str | None:
    """The detailed model the prop's bake reads (relative to its folder): its Omni-Texture while used
    (omnitexture_used), else its 3D model. Retopo always reads the 3D model. rules.js highPoly gives the same."""
    return prop["omnitexture"]["glb"] if omnitexture_used(prop) else (prop.get("mesh") or {}).get("glb")


def _normalize_mesh(block: dict[str, Any], saved: dict[str, Any], says_geometry: bool) -> None:
    """An older or hand-edited project's mesh block, repaired in place: settings that name no geometry (`saved`, as
    the file has them) keep Pixal3D's (mesh_geometry), and a model whose block did not say which made it
    (`says_geometry` false: made before there was a choice) was Pixal3D's."""
    block["settings"]["geometry"] = mesh_geometry(saved)
    if not says_geometry:
        block["geometry"] = FIRST_GEOMETRY if block.get("glb") and not block.get("imported") else None
    elif block["geometry"] not in GEOMETRIES:
        block["geometry"] = None


def _normalize_references(block: dict[str, Any]) -> None:
    """images.references as project.json has it, in place: the prop's own example pictures (reference/...), each
    once, at most MAX_REFERENCES. The one "reference" of older projects (Forge's img2img picture) becomes the
    first."""
    old = block.pop("reference", None)
    raw = block.get("references") if isinstance(block.get("references"), list) else []
    refs = [rel for rel in ([old] if old else []) + raw if _is_reference_path(rel)]
    block["references"] = list(dict.fromkeys(refs))[:MAX_REFERENCES]


def _is_reference_path(rel: Any) -> bool:
    """reference/<file name>: where add_references stores the example pictures."""
    if not isinstance(rel, str) or ":" in rel or "\\" in rel:
        return False
    parts = rel.split("/")
    return len(parts) == 2 and parts[0] == "reference" and parts[1] not in ("", ".", "..")


def retopo_fresh(prop: dict[str, Any]) -> bool:
    """The prop's clean mesh (retopo.obj) was made from the 3D model it has now, by the method it is set to: a
    new 3D model, or another method picked, makes it "to do" again (another method: the clean mesh waits in
    retopo.kept, see switch_retopo_method; one too big for that stays until it is replaced).
    rules.js retopoFresh must give the same (tests/test_web.py compares them)."""
    retopo = prop.get("retopo") or {}
    if not retopo.get("obj") or retopo.get("stale"):
        return False
    source = retopo.get("source")
    if source and source != (prop.get("mesh") or {}).get("glb"):
        return False
    made_by = retopo.get("method") or FIRST_RETOPO_METHOD
    return made_by == ((retopo.get("settings") or {}).get("method") or DEFAULT_RETOPO_METHOD)


def retopo_missing_sizes(prop: dict[str, Any], sizes: list[int | float] | None = None) -> list[int | float]:
    """The sizes (the prop's own, unless given) it has no clean mesh of yet: those without a variant while its
    clean meshes are up to date (retopo_fresh), else all of them. rules.js retopoMissing gives the same
    (tests/test_web.py compares them)."""
    block = prop.get("retopo") or {}
    wanted = retopo_sizes(block.get("settings")) if sizes is None else sizes
    if not retopo_fresh(prop):
        return list(wanted)
    made = {variant.get("size") for variant in block.get("variants") or [] if isinstance(variant, dict)}
    return [size for size in wanted if size not in made]


def _variants_patch(block: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """New clean meshes (patch "variants": [{"size", "obj", "stats"}], a job's or the viewer's) as the patch that
    makes them the prop's. They join the variants the prop has when those are up to date and of the same method and
    3D model (one of the same size or file is replaced), else they replace them. The variant the prop then uses
    (obj, stats): the one `patch["picked"]` names (the viewer's Accept), else the new one of a prop with one size,
    else the one picked before when it was not made again, else the first of the prop's sizes; accepted for the
    first three. Only "variants" changes while the prop keeps using the same file: a bake made from it stays fresh."""
    method = patch.get("method") if patch.get("method") in RETOPO_METHODS else FIRST_RETOPO_METHOD
    made = [dict(variant) for variant in patch["variants"]]
    if not made:
        return {}
    remade = {variant["size"] for variant in made}
    joins = bool(block.get("obj")) and not block.get("stale") and block.get("source") == patch.get("source") \
        and (block.get("method") or FIRST_RETOPO_METHOD) == method
    files = {variant["obj"] for variant in made}
    old = [variant for variant in (block.get("variants") or [] if joins else [])
           if variant["size"] not in remade and variant["obj"] not in files]
    variants = sorted(old + made, key=lambda variant: variant["size"])
    by_size = {variant["size"]: variant for variant in variants}
    sizes = retopo_sizes(block.get("settings"))
    picked = patch.get("picked")
    if picked is None and len(sizes) == 1 and sizes[0] in remade:
        picked = sizes[0]
    before = next((variant for variant in old if variant["obj"] == block.get("obj")), None)
    if picked in by_size:
        used, accepted = by_size[picked], True
    elif before is not None and block.get("accepted"):
        used, accepted = before, True
    else:
        used = next((by_size[size] for size in sizes if size in by_size), variants[0])
        accepted = False
    if used is before and accepted == bool(block.get("accepted")):
        return {"variants": variants}
    rest = {key: value for key, value in patch.items() if key not in ("variants", "picked")}
    return {**rest, "variants": variants, "obj": used["obj"], "stats": used.get("stats"), "accepted": accepted}


def _use_the_only_size(prop: dict[str, Any]) -> None:
    """A prop that asks for one size and has its clean mesh, up to date, uses that one, picked: there is nothing for
    the artist to choose (a job's result says so itself, _variants_patch; this is for sizes or a method saved since,
    and a project from before there were picks). In place; a bake made from another clean mesh is out of date, and the
    thumbnail of another one goes."""
    block = prop["retopo"]
    sizes = retopo_sizes(block.get("settings"))
    variant = next((v for v in block.get("variants") or [] if v["size"] == sizes[0]), None) \
        if len(sizes) == 1 and retopo_fresh(prop) else None
    if variant is None or (block.get("obj") == variant["obj"] and block.get("accepted") is True):
        return
    if block.get("obj") != variant["obj"]:
        prop["thumbs"].pop("retopo", None)
        if (prop.get("projection") or {}).get("glb"):
            prop["projection"]["stale"] = True
    block.update(obj=variant["obj"], stats=variant.get("stats"), accepted=True)


def retopo_keepable(result: dict[str, Any]) -> bool:
    """The clean meshes are small enough to stay with the prop while it is set to another method: the stats of every
    variant say fewer than RETOPO_KEEP_VERTICES vertices (one that does not say how many is not kept)."""
    def small(variant: Any) -> bool:
        vertices = (variant.get("stats") or {}).get("vertices") if isinstance(variant, dict) else None
        return isinstance(vertices, (int, float)) and not isinstance(vertices, bool) and vertices < RETOPO_KEEP_VERTICES

    variants = result.get("variants") or []
    return bool(result.get("obj")) and bool(variants) and all(small(variant) for variant in variants)


def _take_retopo_result(block: dict[str, Any]) -> dict[str, Any]:
    """The prop's clean mesh, taken out of prop.retopo (its _RETOPO_RESULT_KEYS): the block is left without one."""
    result = {key: block.pop(key) for key in _RETOPO_RESULT_KEYS if key in block}
    block.update({key: copy.deepcopy(default) for key, default in _STAGE_OUTPUTS["retopo"].items()
                  if key in _RETOPO_RESULT_KEYS})
    return result


def _kept_retopos(block: dict[str, Any]) -> dict[str, Any]:
    """retopo.kept: {method: the clean mesh that method made, set aside while the prop is set to another}."""
    return block["kept"] if isinstance(block.get("kept"), dict) else {}


def _set_kept_retopos(block: dict[str, Any], kept: dict[str, Any]) -> None:
    if kept:
        block["kept"] = kept
    else:
        block.pop("kept", None)


def _normalize_variants(result: dict[str, Any]) -> None:
    """An older or hand-edited project's clean meshes (prop.retopo, or one of retopo.kept), repaired in place: the
    variants are {"size", "obj", "stats"}, one per size, smallest first, and only while there is a clean mesh (obj).
    One from before there were sizes (its obj in no variant) becomes the variant of the size its vertex count stands
    for (retopo_size_of)."""
    variants: dict[int | float, dict[str, Any]] = {}
    for variant in result.get("variants") if isinstance(result.get("variants"), list) else []:
        size = _retopo_size(variant.get("size")) if isinstance(variant, dict) else None
        if size is not None and isinstance(variant.get("obj"), str) and variant["obj"] and size not in variants:
            stats = variant.get("stats")
            variants[size] = {"size": size, "obj": variant["obj"], "stats": stats if isinstance(stats, dict) else None}
    obj = result.get("obj")
    if obj and all(variant["obj"] != obj for variant in variants.values()):
        stats = result.get("stats") if isinstance(result.get("stats"), dict) else None
        size = retopo_size_of((stats or {}).get("vertices"))
        variants[size] = {"size": size, "obj": obj, "stats": stats}
    result["variants"] = sorted(variants.values(), key=lambda variant: variant["size"]) if obj else []


def switch_retopo_method(block: dict[str, Any], method: str) -> bool:
    """prop.retopo once the prop is set to `method`, in place -> True when the prop's clean mesh is another one now.
    Each method keeps its own clean mesh: the one `method` made before (retopo.kept) is the prop's again, and the one
    the prop had waits there under its own method when it is small enough (retopo_keepable). One that is too big
    stays the prop's clean mesh, "to do" again, until another method's replaces it. Nothing is deleted on disk (each
    method writes its own file). rules.js retopoSwitched must give the same (tests/test_web.py compares them)."""
    kept = _kept_retopos(block)
    back = kept.get(method)
    back = back if isinstance(back, dict) and back.get("obj") else None
    made_by = (block.get("method") or FIRST_RETOPO_METHOD) if block.get("obj") else None
    if made_by == method or (not back and not retopo_keepable(block)):
        return False
    keep = retopo_keepable(block)
    result = _take_retopo_result(block)
    if keep:
        kept[made_by] = result
    kept.pop(method, None)
    block.update(back or {})
    _set_kept_retopos(block, kept)
    return True


def _make_room_for_retopo(block: dict[str, Any], method: str) -> None:
    """A new clean mesh made by `method` is about to become the prop's: the one that method made before is replaced
    (also a kept one), one of another method is set aside when it is small enough (else it is replaced too)."""
    kept = _kept_retopos(block)
    kept.pop(method, None)
    if block.get("obj") and (block.get("method") or FIRST_RETOPO_METHOD) != method and retopo_keepable(block):
        made_by = block.get("method") or FIRST_RETOPO_METHOD
        kept[made_by] = _take_retopo_result(block)
    _set_kept_retopos(block, kept)


def _switch_retopo(prop: dict[str, Any]) -> None:
    """The prop was set to another retopology method (switch_retopo_method). When its clean mesh is another one
    now, the retopo thumbnail goes (it shows the other one), and the bake is out of date unless it was made from
    the clean mesh that is back: a kept clean mesh remembers whether its bake was up to date (bake_stale)."""
    block, bake = prop["retopo"], prop.get("projection") or {}
    had, made_by = block.get("obj"), block.get("method") or FIRST_RETOPO_METHOD
    if not switch_retopo_method(block, retopo_method(block.get("settings"))):
        return
    prop["thumbs"].pop("retopo", None)
    bake_stale = block.pop("bake_stale", True)
    if not bake.get("glb"):
        return
    aside = _kept_retopos(block).get(made_by) if had else None
    if aside and bake.get("source_retopo") == had:
        aside["bake_stale"] = bool(bake.get("stale"))
    if block.get("obj"):
        bake["stale"] = bool(bake_stale) if bake.get("source_retopo") == block["obj"] else True

# The artist's own 3D model ("+" in the 3D stage, or a drop there): uploaded files, or .zip files
# that are unpacked, land in mesh/import_NNN/ and Instant Meshes converts the model into
# mesh/imported_NNN.glb. Models in order of preference when several arrive at once; keep in sync
# with instant_meshes_brush.assets.SUFFIXES. Every other kind of file is left out.
MAX_MODEL_UPLOAD_BYTES = 1024 ** 3
MODEL_SUFFIXES: tuple[str, ...] = (".fbx", ".glb", ".gltf", ".obj", ".dae", ".ply", ".stl", ".off")
MODEL_COMPANION_SUFFIXES = frozenset({
    ".mtl", ".bin", ".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".bmp", ".webp", ".dds", ".psd",
    ".exr"})    # EXR is not read, but the import then names the texture it could not use
MAX_IMPORT_FILES = 5000
_MAX_IMPORT_PATH = 180          # characters of a saved file's path inside the import folder
_MAX_IMPORT_NAME = 100          # characters of one folder or file name in it
_COPY_CHUNK = 1024 * 1024
_IMPORT_ENTRY = re.compile(r"^import(?:ed)?_(\d+)(?:\.glb)?$")
_PATH_UNSAFE_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')

# The sides a view shows, in the order a multi-view 3D model takes them (the front first). "left": the prop's
# own left side, so it faces image-left; "right": it faces image-right. Keep in sync with static/js/stages/rules.js.
VIEW_ORDER: tuple[str, ...] = ("front", "left", "back", "right")
VIEW_NAMES = {"front": "Front", "left": "← Left", "back": "Back", "right": "Right →"}
MIN_VIEW_SIZE = 16              # pixels: a smaller part of a picture is no view of anything
MAX_IMAGE_UPLOADS = 16          # pictures in one upload (each read into memory)

# The page's state of a project: the stage the artist had open. The rows-of-cards version kept the active
# row's index and one card position per row instead (_OLD_UI_KEYS); a project saved by it opens at that stage,
# read with the stages of its time (_OLD_UI_ROWS: before Omni-Texture), never with today's list.
DEFAULT_UI: dict[str, Any] = {"stage": "images"}
_OLD_UI_KEYS = ("unlocked_row", "active_row", "positions")
_OLD_UI_ROWS: tuple[str, ...] = ("images", "mesh", "retopo", "projection", "rig")

PROP_ID_RE = re.compile(r"^p_[0-9a-f]{6}$")
# A deleted prop's folder in its project's deleted/ (delete_prop): <prop id>_<time>, and in it the prop's entry and
# place in the list, which restore_prop puts back.
_DELETED_RE = re.compile(r"^(p_[0-9a-f]{6})_\d{8}-\d{6}(?:-\d+)?$")
DELETED_RECORD = "deleted-prop.json"
# \w is Unicode-aware, so accented and non-Latin letters survive.
_UNSAFE_CHARS = re.compile(r"[^\w \-.()\[\],+!']")
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                   *(f"LPT{i}" for i in range(1, 10))}


class ProjectError(Exception):
    """A user-facing problem with a project request. `status_code` becomes the HTTP status."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def safe_name(name: str) -> str:
    """A display name turned into a safe Windows folder name ('' if nothing usable is left)."""
    name = _UNSAFE_CHARS.sub(" ", unicodedata.normalize("NFC", name))
    name = " ".join(name.split())[:MAX_PROJECT_NAME].strip(" .")
    if name.split(".")[0].strip().upper() in _RESERVED_NAMES:
        name = f"{name} project"
    return name


def sniff_image(data: bytes) -> str | None:
    """File extension from the magic bytes: ".png", ".jpg", ".webp", or None."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return None


def box_name_from_prompt(prompt: str) -> str:
    """A short prop name from the first words of a prompt that name a thing: "a wooden barrel with iron bands,
    game asset" -> "Wooden barrel"; "masterpiece, best quality, (wooden barrel:1.3)" -> "Wooden barrel".
    "" when no part of the prompt does."""
    text = re.sub(r"<[^>]*>", " ", prompt)                     # <lora:name:0.8>
    text = re.sub(r":\s*-?\d+(?:\.\d+)?|[()\[\]{}]", " ", text)  # (words:1.2)
    for part in re.split(r"[,.;!?|\n]", text):
        words = part.split()
        while words and words[0].lower() in ("a", "an", "the"):
            words = words[1:]
        if not words or " ".join(words).lower() in _QUALITY_TAGS:
            continue
        cut = next((i for i, word in enumerate(words) if i and word.lower() in _NAME_STOP_WORDS), len(words))
        name = " ".join(words[:min(cut, 4)])[:40].strip()
        return name[:1].upper() + name[1:]
    return ""


def box_name_from_file(file_name: str) -> str:
    """A prop name from a file's name, the way the Images workbench names an imported picture:
    "old_barrel-front.png" -> "Old barrel front"; "Chair_01.obj" -> "Chair 01"."""
    stem = re.sub(r"\.[^.]+$", "", file_name)
    name = re.sub(r"\s+", " ", re.sub(r"[_\-.]+", " ", stem)).strip()[:40]
    return name[:1].upper() + name[1:]


# ---- views: the prop seen from each side ---------------------------------------------------------------

def views_signature(views: list[dict[str, Any]]) -> str:
    """The views as one string ("front:images/0003.png:0,0,512,512|left:images/0003.png:*"), in VIEW_ORDER:
    a 3D model made from them is fresh while the prop's views give the same string, whatever order they are
    saved in and however their numbers are written (512.0 is 512, as in JavaScript). rules.js viewsSignature
    must give the same (tests/test_web.py compares them)."""
    rank = {view: i for i, view in enumerate(VIEW_ORDER)}
    ordered = sorted(views, key=lambda v: rank.get(v.get("view"), -1))    # -1: like indexOf in rules.js
    return "|".join(f"{v.get('view')}:{v.get('src')}:"
                    + ("*" if v.get("rect") is None else ",".join(_plain(n) for n in v["rect"])) for v in ordered)


def _plain(number: Any) -> str:
    """A number of a view's frame as JavaScript writes it: 512.0 is "512" there."""
    return str(int(number)) if isinstance(number, float) and number.is_integer() else str(number)


def effective_views(images: dict[str, Any]) -> list[dict[str, Any]]:
    """The views a 3D model is made from now: all of images.views while the chosen picture is one of theirs,
    else none (the chosen picture alone; the views are kept for when the artist picks one of theirs again).
    rules.js effectiveViews must give the same (tests/test_web.py compares them)."""
    views = [v for v in images.get("views") or [] if isinstance(v, dict)]
    chosen = images.get("chosen")
    return views if chosen and any(v.get("src") == chosen for v in views) else []


def thumb_crop(images: dict[str, Any]) -> list[int] | None:
    """The part of the chosen picture its thumbnail shows: the Front slice [x, y, w, h] while the chosen
    picture is a sheet sliced into views, else None (all of it). rules.js thumbCrop gives the same."""
    front = next((v for v in effective_views(images) if v.get("view") == "front"), None)
    rect = front.get("rect") if front and front.get("src") == images.get("chosen") else None
    return list(rect) if isinstance(rect, list) else None


def thumb_signature(images: dict[str, Any]) -> str | None:
    """What the thumbnail of a prop's picture must show: the chosen picture, and the part of it when not all
    (thumb_crop): "images/0003.png" or "images/0003.png#0,0,512,512"; None without a chosen picture. It is kept
    with the thumbnail (thumbs["images"]["of"]). store.js thumbSignature gives the same."""
    chosen = images.get("chosen")
    if not chosen:
        return None
    crop = thumb_crop(images)
    return f"{chosen}#{','.join(str(n) for n in crop)}" if crop else chosen


def _drop_stale_thumb(prop: dict[str, Any]) -> None:
    """Take the picture's thumbnail off a prop when it is not of what the prop shows now (thumb_signature; one
    from before thumbnails said what they show counts as stale): whatever changed the chosen picture or its
    views, a thumbnail of something else is never given out as the prop's. The page draws a new one."""
    thumbs = prop.get("thumbs")
    thumb = thumbs.get("images") if isinstance(thumbs, dict) else None
    images = prop.get("images") if isinstance(prop.get("images"), dict) else {}
    if thumb is not None and thumb.get("of") != thumb_signature(images):
        del thumbs["images"]


VIEWS_SOURCE = "views:"            # how mesh.source begins for a model made from views


def mesh_source(images: dict[str, Any]) -> str | None:
    """What a 3D model is made from (mesh.source): the effective views when there are any, else the chosen
    picture. Nothing else of the prop is in it: not the words its pictures were made from, not the Images tab it
    is on, not the 3D settings (a new seed is asked for with "Make 3D again")."""
    views = effective_views(images)
    return VIEWS_SOURCE + views_signature(views) if views else images.get("chosen")


def made_from_one_picture(prop: dict[str, Any]) -> str | None:
    """The picture the prop's 3D model was made from, when it was made from one alone (mesh.source names it); None
    for a model made from views and for one that says nothing of it (the artist's own)."""
    source = (prop.get("mesh") or {}).get("source")
    return source if isinstance(source, str) and not source.startswith(VIEWS_SOURCE) else None


def tileable_prop(prop: dict[str, Any]) -> bool:
    """The prop is a tileable texture: its Tileable tick is on, for a model that draws such pictures (not FLUX.3: the
    image adapter draws none with it). A texture is no object: it gets no 3D model (adapters/pixal3d.py), and the
    stages after Images pass it by. rules.js tileable gives the same (tests/test_web.py compares them)."""
    settings = (prop.get("images") or {}).get("settings") or {}
    return settings.get("tileable") is True and settings.get("qwen_model") != "flux3"


def front_only(prop: dict[str, Any]) -> bool:
    """The prop's 3D model was made from its front view alone although its views are several: Pixal3D's
    multi-view models were not downloaded then (adapters/pixal3d.py). rules.js frontOnly gives the same."""
    mesh, images = prop.get("mesh") or {}, prop.get("images") or {}
    return bool(mesh.get("glb")) and mesh.get("imported") in (None, False) \
        and (mesh.get("stats") or {}).get("mode") == "single" \
        and len(effective_views(images)) >= 2 and mesh.get("source") == mesh_source(images)


def mesh_fresh(prop: dict[str, Any], multiview: bool = False) -> bool:
    """The prop's 3D model (mesh.glb) was made from the picture, or the views, the prop has now (mesh.source ==
    mesh_source), or is the artist's own model. Only that counts: choosing the same picture again, another one
    and then the first again, or saving the same views again leaves it fresh, and nothing but a new model
    replaces mesh.glb or mesh.source. multiview: Pixal3D's multi-view models are there now, so a model made from
    the front view alone (front_only) is out of date. rules.js meshFresh must give the same (tests/test_web.py
    compares them)."""
    mesh, images = prop.get("mesh") or {}, prop.get("images") or {}
    if not mesh.get("glb"):
        return False
    same = mesh.get("imported") not in (None, False) or not images.get("chosen") or not mesh.get("source") \
        or mesh["source"] == mesh_source(images)
    return same and not (multiview and front_only(prop))


def _check_view(entry: Any, candidates: Collection[str]) -> dict[str, Any]:
    """One entry of images.views, checked -> a clean copy {"view", "src", "rect"}. ProjectError otherwise."""
    if not isinstance(entry, dict) or entry.get("view") not in VIEW_ORDER:
        raise ProjectError(f"Each view must say which side it shows: {', '.join(VIEW_ORDER)}. "
                           f"Reload the page and try again.")
    name = VIEW_NAMES[entry["view"]]
    if not isinstance(entry.get("src"), str) or entry["src"] not in candidates:
        raise ProjectError(f"The {name} view must be one of this prop's own pictures. Pick it again.")
    rect = entry.get("rect")
    if rect is not None:
        if not isinstance(rect, list) or len(rect) != 4 \
                or not all(isinstance(n, int) and not isinstance(n, bool) for n in rect):
            raise ProjectError(f"The {name} view's frame must be four whole numbers (x, y, width, height). "
                               f"Reload the page and try again.")
        x, y, w, h = rect
        if x < 0 or y < 0 or w < MIN_VIEW_SIZE or h < MIN_VIEW_SIZE:
            raise ProjectError(f"The {name} view's frame is too small or starts outside the picture. "
                               f"Draw it again, at least {MIN_VIEW_SIZE} pixels wide and high.")
        rect = [x, y, w, h]
    return {"view": entry["view"], "src": entry["src"], "rect": rect}


def check_views(raw: Any, candidates: Collection[str]) -> list[dict[str, Any]]:
    """The views a page sends, checked (ProjectError otherwise) -> clean copies in VIEW_ORDER.
    At most one view per side; a set without a front view may be saved (the artist is still marking it)."""
    if not isinstance(raw, list):
        raise ProjectError('"views" must be a list of views. Reload the page and try again.')
    views = [_check_view(entry, candidates) for entry in raw]
    seen: set[str] = set()
    for view in views:
        if view["view"] in seen:
            raise ProjectError(f"Only one part of the pictures can be the {VIEW_NAMES[view['view']]} view. "
                               f"Clear the other one first.")
        seen.add(view["view"])
    return sorted(views, key=lambda v: VIEW_ORDER.index(v["view"]))


def _normalize_views(raw: Any, candidates: Collection[str]) -> list[dict[str, Any]]:
    """images.views as project.json has it, without the entries that are not valid views (an older or
    hand-edited file): the first view of each side is kept."""
    views: dict[str, dict[str, Any]] = {}
    for entry in raw if isinstance(raw, list) else []:
        try:
            view = _check_view(entry, candidates)
        except ProjectError:
            continue
        views.setdefault(view["view"], view)
    return [views[side] for side in VIEW_ORDER if side in views]


def check_sides(raw: Any, count: int) -> list[str | None] | None:
    """The sides `count` pictures uploaded together show, as the page sends them (one entry per file, in
    upload order: "front" | "left" | "back" | "right", or None for a picture that is no view), checked
    (ProjectError otherwise). None when the page sent none."""
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != count \
            or any(side is not None and side not in VIEW_ORDER for side in raw):
        raise ProjectError("The sides of these pictures are not right. Reload the page and import them again.")
    named = [side for side in raw if side is not None]
    if len(named) != len(set(named)):
        raise ProjectError("Two of these pictures are said to show the same side. Reload the page and import "
                           "them again.")
    return list(raw)


def _auto_views(sources: list[str], sides: list[str | None] | None = None) -> list[dict[str, Any]]:
    """Several pictures uploaded together show the prop from its sides: the ones `sides` gives (check_sides;
    the page reads them from the file names), else VIEW_ORDER in upload order (further ones are just
    pictures)."""
    if sides is None:
        sides = list(VIEW_ORDER[:len(sources)]) + [None] * max(0, len(sources) - len(VIEW_ORDER))
    views = [{"view": side, "src": src, "rect": None} for side, src in zip(sides, sources) if side]
    return sorted(views, key=lambda v: VIEW_ORDER.index(v["view"]))


def _new_prop(prop_id: str, name: str, previous: dict[str, Any] | None, geometry: str = DEFAULT_GEOMETRY
              ) -> dict[str, Any]:
    """A fresh prop. It inherits the previous prop's stage settings (model, CFG, sizes...), not its prompt, not its
    retopology method or clean-mesh sizes and not its geometry: every prop starts with DEFAULT_RETOPO_METHOD, the
    default size and `geometry` (the project's Shape: ProjectStore._geometry), whatever the one before it has. Nor
    its Tileable tick: a prop after a texture is an object again (a tileable one gets no 3D model: tileable_prop)."""
    prop: dict[str, Any] = {"id": prop_id, "name": name, "created": now_iso(), "draft": False, "thumbs": {}}
    for stage in EDITABLE_STAGES:
        settings = copy.deepcopy(DEFAULT_STAGE_SETTINGS[stage])
        if previous:
            settings.update(copy.deepcopy((previous.get(stage) or {}).get("settings") or {}))
        prop[stage] = {"settings": settings, **copy.deepcopy(_STAGE_OUTPUTS[stage])}
    prop["images"]["settings"]["prompt"] = ""
    prop["images"]["settings"]["tileable"] = False
    prop["omnitexture"]["settings"]["chosen"] = False
    prop["retopo"]["settings"]["method"] = DEFAULT_RETOPO_METHOD
    prop["retopo"]["settings"]["sizes"] = list(DEFAULT_STAGE_SETTINGS["retopo"]["sizes"])
    prop["mesh"]["settings"]["geometry"] = geometry
    for stage, _tool in paths.STAGES:
        prop.setdefault(stage, {})   # stages without a tool yet (rig)
    return prop


def _pack(data: dict[str, Any], prop: dict[str, Any], file_name: str | None = None) -> None:
    """The draft becomes a prop at the end of the list. Unless the artist named it, it is named after its
    prompt, else after the picture's file (`file_name`, an import), else "Prop N"."""
    data["props"].remove(prop)
    data["props"].append(prop)
    prop["draft"] = False
    if not prop["name"].strip():
        count = sum(1 for p in data["props"] if not p["draft"])
        prop["name"] = (box_name_from_prompt(str(prop["images"]["settings"].get("prompt") or ""))
                        or box_name_from_file(file_name or "") or f"Prop {count}")


def _normalize(data: dict[str, Any], folder_name: str) -> dict[str, Any]:
    """Fill keys missing from older or hand-edited project.json files. Adds "id" (the folder name); "geometry", the
    project's Shape for new props, is Pixal3D's where the file names none."""
    data["id"] = folder_name
    data["untitled"] = data.get("untitled") is True
    data["geometry"] = data["geometry"] if data.get("geometry") in GEOMETRIES else DEFAULT_GEOMETRY
    name = data.get("name")
    if not data["untitled"] or not isinstance(name, str) or not name.strip():
        data["name"] = UNTITLED if data["untitled"] else folder_name
    data.setdefault("created", now_iso())
    data.setdefault("updated", data["created"])
    ui = data.get("ui") if isinstance(data.get("ui"), dict) else {}
    stage = ui.get("stage")
    if stage not in STAGE_IDS:
        row = ui.get("active_row")
        old = isinstance(row, int) and not isinstance(row, bool) and 0 <= row < len(_OLD_UI_ROWS)
        stage = _OLD_UI_ROWS[row] if old else DEFAULT_UI["stage"]
    data["ui"] = {**{k: v for k, v in ui.items() if k not in _OLD_UI_KEYS}, "stage": stage}
    props = data.get("props") if isinstance(data.get("props"), list) else []
    data["props"] = [p for p in props if isinstance(p, dict) and PROP_ID_RE.match(str(p.get("id", "")))]
    old_uv_default = data.get(_UV_DEFAULT_KEY) != UV_LENIENCY
    data[_UV_DEFAULT_KEY] = UV_LENIENCY                 # kept by the project's next write
    has_draft = False
    for prop in data["props"]:
        prop.setdefault("name", prop["id"])
        prop.setdefault("created", data["created"])
        prop["draft"] = prop.get("draft") is True and not has_draft     # at most one
        has_draft = has_draft or prop["draft"]
        thumbs = prop.get("thumbs") if isinstance(prop.get("thumbs"), dict) else {}
        prop["thumbs"] = {stage: thumb for stage, thumb in thumbs.items()
                          if stage in THUMB_STAGES and isinstance(thumb, dict) and isinstance(thumb.get("path"), str)}
        for stage in EDITABLE_STAGES:
            block = prop.get(stage) if isinstance(prop.get(stage), dict) else {}
            if stage == "retopo" and "accepted" not in block:
                # Saved before Accept existed: "Save viewer result" was the artist's way of keeping a result.
                block["accepted"] = block.get("saved_from") == "viewer" and bool(block.get("obj"))
            if stage == "retopo" and block.get("method") not in RETOPO_METHODS:
                # Made before there were other methods (or no clean mesh yet).
                block["method"] = FIRST_RETOPO_METHOD if block.get("obj") else None
            says_geometry = "geometry" in block          # (mesh) a model of before the choice says nothing of it
            for key, default in _STAGE_OUTPUTS[stage].items():
                block.setdefault(key, copy.deepcopy(default))
            saved = block.get("settings") if isinstance(block.get("settings"), dict) else {}
            if stage == "retopo" and block.get("obj") and saved.get("method") not in RETOPO_METHODS:
                # A clean mesh from before the prop could pick: its method stays the one that made it (the
                # default would make the mesh "to do" again).
                saved = {**saved, "method": block["method"]}
            block["settings"] = {**DEFAULT_STAGE_SETTINGS[stage], **saved}
            if stage == "mesh":
                _normalize_mesh(block, saved, says_geometry)
            if stage == "omnitexture":
                block["settings"]["chosen"] = block["settings"].get("chosen") is True
                block["painted"] = block.get("painted") is True
                for key in ("views", "projected", "layers", "steps"):
                    listed = block[key] if isinstance(block[key], list) else []
                    block[key] = [v for v in listed if isinstance(v, str)]
                sheets = block["sheets"] if isinstance(block["sheets"], list) else []
                block["sheets"] = [{"folder": s["folder"], "views": [v for v in s["views"] if isinstance(v, str)]}
                                   for s in sheets if isinstance(s, dict) and isinstance(s.get("folder"), str)
                                   and isinstance(s.get("views"), list)]
                steps = block["steps"]
                repaints = block["repaints"] if isinstance(block["repaints"], list) else []
                block["repaints"] = [
                    {"glb": r["glb"], "under": r["under"] if isinstance(r.get("under"), str)
                     else None if "under" in r else _step_before(steps, r["glb"]),
                     "picture": r["picture"],
                     "layer": r["layer"] if isinstance(r.get("layer"), str) else None,
                     "fade": float(r["fade"]) if isinstance(r.get("fade"), (int, float)) else 1.0,
                     **({"hidden": True} if r.get("hidden") is True else {})}
                    for r in repaints if isinstance(r, dict) and isinstance(r.get("glb"), str)
                    and isinstance(r.get("picture"), str)]
                bases = block["bases"] if isinstance(block["bases"], list) else []
                block["bases"] = [
                    {"glb": b["glb"], "views": [
                        {"folder": v["folder"], **{key: v[key] if isinstance(v.get(key), str) else None
                                                   for key in ("picture", "layer")}}
                        for v in b["views"] if isinstance(v, dict) and isinstance(v.get("folder"), str)],
                     **({"hidden": True, "shown": b["shown"]}
                        if b.get("hidden") is True and isinstance(b.get("shown"), str) else {})}
                    for b in bases if isinstance(b, dict) and isinstance(b.get("glb"), str)
                    and isinstance(b.get("views"), list)]
                if not block["repaints"] and not block["bases"]:
                    # Repaints from before they were recorded: every step after the first is one; the page
                    # shows where each is by its mask until it is made again with a layer (a fade).
                    block["repaints"] = [repaint_record(step, layer=REPAINT_MASK, under=_step_before(steps, step))
                                         for step in steps[1:] if step != TAKEN_OFF]
            if stage == "retopo":
                _normalize_retopo_settings(block["settings"], old_uv_default)
                _normalize_variants(block)
                if "sizes" not in saved and block["variants"]:
                    # A clean mesh from before there were sizes: the prop asks for the size it has.
                    block["settings"]["sizes"] = [next(variant["size"] for variant in block["variants"]
                                                       if variant["obj"] == block["obj"])]
                # The clean meshes kept for the prop's other methods (projects before them have none).
                kept = {method: result for method, result in _kept_retopos(block).items()
                        if method in RETOPO_METHODS and isinstance(result, dict) and result.get("obj")}
                for result in kept.values():
                    _normalize_variants(result)
                _set_kept_retopos(block, kept)
            if stage == "images":
                candidates = block["candidates"] if isinstance(block["candidates"], list) else []
                block["views"] = _normalize_views(block["views"], candidates)
                _normalize_references(block)
                block.pop("sketch", None)       # the Sketch tab's painting (retired: from_sketch); its file stays
                _normalize_images_settings(block["settings"])
            prop[stage] = block
        for stage, _tool in paths.STAGES:
            prop.setdefault(stage, {})
        _use_the_only_size(prop)
        _drop_stale_thumb(prop)
    # A draft that has its picture is a prop: the page before auto-pack left that step to the artist.
    for prop in [p for p in data["props"] if p["draft"] and p["images"].get("chosen")]:
        _pack(data, prop)
    return data


def _is_empty(data: dict[str, Any]) -> bool:
    """No props, and a draft (if any) without an image, a prompt or a model: nothing to lose."""
    for prop in data["props"]:
        images = prop["images"]
        if not prop["draft"] or images.get("candidates") or images.get("chosen") or images.get("references") \
                or str(images["settings"].get("prompt") or "").strip() or prop["mesh"].get("glb"):
            return False
    return True


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    """Like paths.write_text_atomic, for bytes: the page may be reading the old file at that moment."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_bytes(data)
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.1)
    finally:
        tmp.unlink(missing_ok=True)


class ProjectStore:
    def __init__(self, root: Path = paths.PROJECTS, exports: Path | None = None):
        """root: the projects folder. exports: where the saved models go (default: the "exports" folder
        beside `root`, which for the real projects folder is paths.EXPORTS)."""
        self.root = root
        self.exports = exports if exports is not None else root.parent / paths.EXPORTS.name
        self._lock = threading.RLock()

    # ---- projects -----------------------------------------------------------------------
    def list_projects(self) -> list[dict[str, Any]]:
        """[{"id", "name", "untitled", "prop_count" (props, not the draft), "updated"}], most recently
        changed first."""
        with self._lock:
            out = [{"id": data["id"], "name": data["name"], "untitled": data["untitled"],
                    "prop_count": sum(1 for p in data["props"] if not p["draft"]), "updated": data["updated"]}
                   for data in self._all_projects()]
        out.sort(key=lambda p: p["updated"], reverse=True)
        return out

    def create_project(self, name: str | None = None) -> dict[str, Any]:
        """A named project, or (no name) a new untitled one."""
        if name is None or (isinstance(name, str) and not name.strip()):
            return self._create_untitled()
        folder = safe_name(name if isinstance(name, str) else "")
        if not folder:
            raise ProjectError("Please give the project a name (letters or numbers).")
        with self._lock:
            d = self.root / folder
            if d.exists():
                raise ProjectError(f'A project called "{folder}" already exists. Pick it from the '
                                   f"project list or choose another name.", 409)
            return self._create(folder, {"name": folder})

    def new_project(self, name: str | None = None, busy: Collection[str] = ()) -> dict[str, Any]:
        """create_project, then open_project: the page comes back to the new project, and an empty untitled
        project left behind is tidied away. Asked for a new untitled project while an empty untitled one
        is open, that one is returned: another would be just the same."""
        with self._lock:
            current = self._last_opened()
            if current and (name is None or (isinstance(name, str) and not name.strip())):
                try:
                    data = self._read(current)
                except ProjectError:
                    data = None
                if data and data["untitled"] and _is_empty(data):
                    return data
            return self.open_project(self.create_project(name)["id"], busy)

    def get_project(self, project: str) -> dict[str, Any]:
        with self._lock:
            return self._read(project)

    def open_project(self, project: str, busy: Collection[str] = ()) -> dict[str, Any]:
        """get_project, remembered as the project opened last. The project open before it is moved to
        projects/.trash when it is an empty untitled one (and no job of it is waiting or running)."""
        with self._lock:
            data = self._read(project)
            previous = self._last_opened()
            if previous != project:
                self._set_last_opened(project)
                if previous:
                    self._trash_if_empty_untitled(previous, busy)
            return data

    def open_last(self, busy: Collection[str] = ()) -> dict[str, Any]:
        """The project to start with: the one opened last, else the one changed last, else a new untitled
        one. Every other empty untitled project is moved to projects/.trash."""
        with self._lock:
            known = self.list_projects()
            last = self._last_opened()
            if any(p["id"] == last for p in known):
                chosen = last
            else:
                chosen = known[0]["id"] if known else self._create_untitled()["id"]
            self._set_last_opened(chosen)
            self.tidy_untitled(keep=chosen, busy=busy)
            return self._read(chosen)

    def tidy_untitled(self, keep: str | None = None, busy: Collection[str] = ()) -> None:
        """Move every empty untitled project but `keep` (default: the one opened last) to projects/.trash."""
        with self._lock:
            keep = keep if keep is not None else self._last_opened()
            for data in self._all_projects():
                if data["id"] != keep:
                    self._trash_if_empty_untitled(data["id"], busy)

    def rename_project(self, project: str, name: Any) -> dict[str, Any]:
        """Give a project a (new) name: its folder is renamed to the name, and so is its folder of saved
        models (exports/<name>) when there is one. The caller makes sure no job or export of the project is
        waiting or running (their files would move away under them)."""
        folder = safe_name(name) if isinstance(name, str) else ""
        if not folder:
            raise ProjectError("Please give the project a name (letters or numbers).")
        with self._lock:
            src = self.project_dir(project)
            data = self._read(project)
            if folder == project and not data["untitled"]:
                return data
            old_exports = self.export_dir(data)
            dst = self.root / folder
            if folder != project:
                if dst.exists() and not os.path.samefile(src, dst):
                    raise ProjectError(f'A project called "{folder}" already exists. Choose another name.', 409)
                _rename_folder(src, dst)
            data["untitled"], data["name"] = False, folder
            self._write(folder, data)
            if self._last_opened() == project:
                self._set_last_opened(folder)
            renamed = self._read(folder)
            _rename_exports(old_exports, self.export_dir(renamed))
            return renamed

    def export_dir(self, data: dict[str, Any]) -> Path:
        """exports/<the project's name, made safe>: where its models are saved (it may not exist yet)."""
        return self.exports / (safe_name(data["name"]) or data["id"])

    def project_dir(self, project: str) -> Path:
        """The folder of an existing project (404 ProjectError otherwise)."""
        if not isinstance(project, str) or not project or safe_name(project) != project:
            raise ProjectError("That project name is not valid.", 404)
        d = self.root / project
        if not (d / "project.json").is_file():
            raise ProjectError(f'There is no project called "{project}". Pick one from the project list.', 404)
        return d

    def patch_ui(self, project: str, changes: dict[str, Any]) -> dict[str, Any]:
        """PATCH /api/projects/{project}: only {"ui": {...}} is accepted; nested dicts are merged."""
        if not isinstance(changes, dict) or set(changes) - {"ui"} or not isinstance(changes.get("ui", {}), dict):
            raise ProjectError('Only the project\'s "ui" state can be changed here.')
        with self._editing(project) as data:
            ui = data["ui"]
            for key, value in (changes.get("ui") or {}).items():
                if isinstance(value, dict) and isinstance(ui.get(key), dict):
                    ui[key] = {**ui[key], **value}
                else:
                    ui[key] = value
            return data

    # ---- props -------------------------------------------------------------------------------
    def prop_dir(self, project: str, prop_id: str) -> Path:
        if not isinstance(prop_id, str) or not PROP_ID_RE.match(prop_id):
            raise ProjectError("That prop id is not valid.", 404)
        return self.project_dir(project) / "props" / prop_id

    def get_prop(self, project: str, prop_id: str) -> dict[str, Any]:
        with self._lock:
            return _find_prop(self._read(project), prop_id)

    def add_prop(self, project: str, name: str | None = None) -> dict[str, Any]:
        """A new prop at the end of the list (its settings are the last prop's)."""
        with self._editing(project) as data:
            props = [p for p in data["props"] if not p["draft"]]
            name = name.strip()[:MAX_PROP_NAME] if isinstance(name, str) else ""
            prop_id = self._new_prop_id(project, data["props"])
            prop = _new_prop(prop_id, name or f"Prop {len(props) + 1}", props[-1] if props else None, self._geometry(data))
            self.prop_dir(project, prop_id).mkdir(parents=True, exist_ok=True)
            data["props"].append(prop)
            return copy.deepcopy(prop)

    def ensure_draft(self, project: str, busy: Collection[str] = ()) -> dict[str, Any]:
        """The project's draft, created when there is none: the last prop's settings, no prompt, no image,
        and no name yet (it is named after its prompt when it becomes a prop, unless the artist names it).

        busy: ids of props with a job waiting or running. A draft among them is still waiting for its first
        picture: it becomes a prop now (the pictures fill it when they arrive) and a new, empty draft is
        made, so the artist can start the next prop meanwhile."""
        with self._lock:
            data = self._read(project)
            draft = next((p for p in data["props"] if p["draft"]), None)
            if draft is not None and draft["id"] not in busy:
                geometry = self._geometry(data)     # never picked for the draft itself (the 3D stage shows no draft)
                if draft["mesh"]["settings"].get("geometry") != geometry:
                    draft["mesh"]["settings"]["geometry"] = geometry
                    self._write(project, data)
                return copy.deepcopy(draft)
            if draft is not None:
                _pack(data, draft)
            props = [p for p in data["props"] if not p["draft"]]
            prop_id = self._new_prop_id(project, data["props"])
            draft = {**_new_prop(prop_id, "", props[-1] if props else None, self._geometry(data)), "draft": True}
            self.prop_dir(project, prop_id).mkdir(parents=True, exist_ok=True)
            data["props"].append(draft)
            self._write(project, data)
            return copy.deepcopy(draft)

    def pack_prop(self, project: str, prop_id: str) -> dict[str, Any]:
        """The draft becomes a prop at the end of the list (a prop stays as it is). 409 without a chosen
        image. The draft is packed by itself when it gets its first picture; this is for older pages."""
        with self._lock:
            data = self._read(project)
            prop = _find_prop(data, prop_id, copy_=False)
            if not prop["draft"]:
                return copy.deepcopy(prop)
            if not prop["images"].get("chosen"):
                raise ProjectError(NO_IMAGE_YET, 409)
            _pack(data, prop)
            self._write(project, data)
            return copy.deepcopy(prop)

    def discard_prop(self, project: str, prop_id: str) -> None:
        """Remove a prop that was only just created for an import that failed, folder and all."""
        with self._editing(project) as data:
            data["props"].remove(_find_prop(data, prop_id, copy_=False))
            shutil.rmtree(self.prop_dir(project, prop_id), ignore_errors=True)

    def patch_prop(self, project: str, prop_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        """{"name"?, "<stage>"?: {"settings": {...}}}; settings are shallow-merged. Only the draft may
        have an empty name (it is named after its prompt when it becomes a prop). The retopo method, sizes and
        LATO.2's and TriFlow's numbers are checked (check_retopo_settings), and so are Qwen-Image's settings
        (check_images_settings) and the 3D geometry (check_mesh_settings)."""
        if not isinstance(changes, dict):
            raise ProjectError("Prop changes must be sent as a JSON object.")
        settings: dict[str, dict[str, Any]] = {}
        for key, value in changes.items():
            if key == "name":
                if not isinstance(value, str):
                    raise ProjectError("Please give the prop a name.")
            elif key in EDITABLE_STAGES:
                if not isinstance(value, dict) or set(value) - {"settings"} \
                        or not isinstance(value.get("settings", {}), dict):
                    raise ProjectError(f'Only the "{key}" settings of a prop can be changed here.')
                settings[key] = copy.deepcopy(value.get("settings") or {})
                if key == "retopo":
                    settings[key] = check_retopo_settings(settings[key])
                elif key == "images":
                    settings[key] = check_images_settings(settings[key])
                elif key == "mesh":
                    settings[key] = check_mesh_settings(settings[key])
                elif key == "omnitexture":
                    settings[key] = check_omnitexture_settings(settings[key])
            else:
                raise ProjectError(f'"{key}" cannot be changed on a prop.')
        with self._editing(project) as data:
            prop = _find_prop(data, prop_id, copy_=False)
            for key, value in changes.items():
                if key == "name":
                    if not value.strip() and not prop["draft"]:
                        raise ProjectError("Please give the prop a name.")
                    prop["name"] = value.strip()[:MAX_PROP_NAME]
                else:
                    prop[key]["settings"].update(settings[key])
                    if key == "retopo" and "method" in settings[key]:
                        _switch_retopo(prop)    # each method has its own clean mesh
                    if key == "retopo" and {"method", "sizes"} & set(settings[key]):
                        _use_the_only_size(prop)
            geometry = settings.get("mesh", {}).get("geometry")
            if geometry is not None and not prop["draft"]:   # the project's Shape: its new props (and the draft) take it
                data["geometry"] = geometry
                for draft in (p for p in data["props"] if p["draft"]):
                    draft["mesh"]["settings"]["geometry"] = self._geometry(data)
            return copy.deepcopy(prop)

    def delete_prop(self, project: str, prop_id: str) -> str:
        """Remove a prop. Its folder is moved to projects/<project>/deleted/, not destroyed, with the prop's entry
        and place in the list (DELETED_RECORD) -> the folder's name there, which restore_prop takes."""
        with self._editing(project) as data:
            prop = _find_prop(data, prop_id, copy_=False)
            index = data["props"].index(prop)
            data["props"].remove(prop)
            src = self.prop_dir(project, prop_id)
            name = f"{prop_id}_{datetime.datetime.now():%Y%m%d-%H%M%S}"
            dst = self.project_dir(project) / "deleted" / name
            for n in itertools.count(1):        # deleted, restored and deleted again within a second
                if not dst.exists():
                    break
                dst = dst.with_name(f"{name}-{n}")
            dst.parent.mkdir(exist_ok=True)
            try:
                if src.exists():
                    shutil.move(str(src), str(dst))
                else:
                    dst.mkdir()
            except OSError as e:
                raise ProjectError(f"Could not remove the prop's folder ({e}). Close any program that "
                                   f"has one of its files open, then try again.", 409) from e
            try:
                atomic_write_json(dst / DELETED_RECORD, {"prop": prop, "index": index})
            except OSError as e:
                log.warning("The deleted prop %s cannot be brought back: %s", prop_id, e)
            return dst.name

    def restore_prop(self, project: str, name: str) -> dict[str, Any]:
        """Bring back a prop delete_prop removed (`name`: its folder in deleted/) at its place in the list -> it."""
        match = _DELETED_RE.match(name) if isinstance(name, str) else None
        if not match:
            raise ProjectError("That deleted prop is not known.", 404)
        with self._editing(project) as data:
            folder = self.project_dir(project) / "deleted" / name
            record_file = folder / DELETED_RECORD
            try:
                record = json.loads(record_file.read_text(encoding="utf-8"))
                prop, index = record["prop"], int(record["index"])
            except (OSError, ValueError, KeyError, TypeError) as e:
                raise ProjectError("That deleted prop cannot be brought back: its folder in \"deleted\" has "
                                   "changed.", 404) from e
            prop_id = match.group(1)
            dst = self.prop_dir(project, prop_id)
            if not isinstance(prop, dict) or prop.get("id") != prop_id:
                raise ProjectError("That deleted prop cannot be brought back: its folder in \"deleted\" has "
                                   "changed.", 404)
            if any(p["id"] == prop_id for p in data["props"]) or dst.exists():
                raise ProjectError("That prop is back already.", 409)
            record_file.unlink()
            try:
                if any(folder.iterdir()):
                    shutil.move(str(folder), str(dst))
                else:
                    folder.rmdir()
            except OSError as e:
                atomic_write_json(record_file, record)
                raise ProjectError(f"Could not bring the prop's folder back ({e}). Close any program that has one "
                                   f"of its files open, then try again.", 409) from e
            data["props"].insert(min(max(index, 0), len(data["props"])), prop)
            return copy.deepcopy(prop)

    def add_image(self, project: str, prop_id: str, data: bytes, file_name: str | None = None) -> dict[str, Any]:
        """Store an uploaded image as a new candidate and select it. A draft becomes a prop (named after its
        prompt, else after `file_name`, the uploaded file's name, unless the artist named it)."""
        return self.add_images(project, prop_id, [(data, file_name)])

    def add_images(self, project: str, prop_id: str, files: list[tuple[bytes, str | None]],
                   sides: list[str | None] | None = None) -> dict[str, Any]:
        """Store uploaded images (data, file name) as new candidates, in order. Two or more are the prop seen
        from its sides: they become its views, instead of the ones it had - the sides `sides` gives (one per
        file, None: no view; check_sides), else front, left side, back, right side in upload order (any
        further ones are just pictures). The front one is selected, else the first. A draft becomes a
        prop, named like add_image does after the first file. Every file is checked before any is stored."""
        if not files:
            raise ProjectError("Choose a picture (PNG, JPG or WEBP) to import.")
        if len(files) > MAX_IMAGE_UPLOADS:
            raise ProjectError(f"These are too many pictures at once. Import at most {MAX_IMAGE_UPLOADS}.", 413)
        sides = check_sides(sides, len(files))
        exts: list[str] = []
        for data, name in files:
            try:
                exts.append(_check_image(data))
            except ProjectError as e:
                if len(files) == 1 or not name:
                    raise
                raise ProjectError(f"{name}: {e}", e.status_code) from e
        with self._editing(project) as proj:
            prop = _find_prop(proj, prop_id, copy_=False)
            rels = [self._write_numbered(project, prop_id, "images", "upload", ext, data)
                    for ext, (data, _name) in zip(exts, files)]
            images = prop["images"]
            images["candidates"] = [*images.get("candidates", []), *rels]
            images["chosen"] = rels[0]
            views = _auto_views(rels, sides) if len(rels) > 1 else []
            if views:
                images["views"] = views
                images["chosen"] = next((v["src"] for v in views if v["view"] == "front"), rels[0])
            prop["thumbs"].pop("images", None)
            if prop["draft"]:
                _pack(proj, prop, files[0][1])
            return copy.deepcopy(prop)

    def set_views(self, project: str, prop_id: str, views: Any) -> dict[str, Any]:
        """The prop's views: [{"view": "front" | "left" | "back" | "right", "src": one of its pictures,
        "rect": [x, y, w, h] in that picture's pixels, or None for all of it}], at most one per side
        (check_views). [] clears them: the 3D model is then made from the chosen picture again."""
        with self._editing(project) as proj:
            prop = _find_prop(proj, prop_id, copy_=False)
            prop["images"]["views"] = check_views(views, prop["images"].get("candidates") or [])
            _drop_stale_thumb(prop)   # it shows another part of the picture now
            return copy.deepcopy(prop)

    def select_image(self, project: str, prop_id: str, rel: str) -> dict[str, Any]:
        with self._editing(project) as proj:
            prop = _find_prop(proj, prop_id, copy_=False)
            if rel not in prop["images"].get("candidates", []):
                raise ProjectError("That image does not belong to this prop. Pick one of its own images.")
            if prop["images"].get("chosen") != rel:
                prop["images"]["chosen"] = rel
                prop["thumbs"].pop("images", None)
            if prop["draft"]:
                _pack(proj, prop)
            return copy.deepcopy(prop)

    def remove_image(self, project: str, prop_id: str, rel: str) -> dict[str, Any]:
        """Take the picture `rel` out of the prop's pictures (images.candidates) and out of its views -> the prop.
        Its file stays in the prop's folder, so restore_image can bring it back (the page's Ctrl+Z). When it was
        the chosen one, its neighbour is chosen: the picture made before it, else the one after it, else none
        (it was the last one). A picture the prop does not have (any more) changes nothing."""
        with self._editing(project) as proj:
            prop = _find_prop(proj, prop_id, copy_=False)
            images = prop["images"]
            before = list(images.get("candidates") or [])
            if rel not in before:
                return copy.deepcopy(prop)
            at = before.index(rel)
            rest = [c for c in before if c != rel]
            images["candidates"] = rest
            images["views"] = [v for v in images.get("views") or [] if v.get("src") != rel]
            if images.get("chosen") == rel:
                images["chosen"] = rest[max(at - 1, 0)] if rest else None
            _drop_stale_thumb(prop)     # of the picture that left, or of a part of one its views made count
            return copy.deepcopy(prop)

    def restore_image(self, project: str, prop_id: str, rel: Any, index: Any = None, views: Any = None,
                      choose: bool = False) -> dict[str, Any]:
        """Bring back a picture remove_image took out -> the prop. `rel` (a picture still in the prop's images
        folder) joins images.candidates again at `index` (its old place; at the end when there is none or the
        list is shorter by now), with the `views` it had (each goes back to its side unless another picture
        has that side by now), and is chosen again when `choose`. A picture the prop has already only gets its
        views and the choice."""
        gone = ProjectError("That picture cannot be brought back: its file is no longer in this prop's folder.")
        if not isinstance(rel, str) or not re.fullmatch(r"images/[^/\\]+\.(png|jpg|jpeg|webp)", rel):
            raise gone
        with self._editing(project) as proj:
            prop = _find_prop(proj, prop_id, copy_=False)
            images = prop["images"]
            candidates = list(images.get("candidates") or [])
            if rel not in candidates:
                try:
                    self.resolve_file(project, f"props/{prop_id}/{rel}")
                except ProjectError:
                    raise gone from None
                at = index if isinstance(index, int) and not isinstance(index, bool) else len(candidates)
                candidates.insert(max(0, min(at, len(candidates))), rel)
                images["candidates"] = candidates
            had = [v for v in check_views(views or [], candidates) if v["src"] == rel]
            taken = {v.get("view") for v in images.get("views") or []}
            images["views"] = sorted([*(images.get("views") or []), *(v for v in had if v["view"] not in taken)],
                                     key=lambda v: VIEW_ORDER.index(v["view"]))
            if choose:
                images["chosen"] = rel
            _drop_stale_thumb(prop)
            return copy.deepcopy(prop)

    def add_references(self, project: str, prop_id: str, files: list[tuple[bytes, str | None]]) -> dict[str, Any]:
        """Store uploaded example pictures (data, file name) for "Describe + example" after the prop's own, as
        reference/reference_NNN.<ext> (a new name each: no stale browser cache). At most MAX_REFERENCES in all (400
        beyond, and nothing is stored); every file is checked before any is stored."""
        if not files:
            raise ProjectError("Choose an example picture (PNG, JPG or WEBP) to add.")
        exts: list[str] = []
        for data, name in files:
            try:
                exts.append(_check_image(data))
            except ProjectError as e:
                if len(files) == 1 or not name:
                    raise
                raise ProjectError(f"{name}: {e}", e.status_code) from e
        with self._editing(project) as proj:
            prop = _find_prop(proj, prop_id, copy_=False)
            refs = prop["images"]["references"]
            if len(refs) + len(files) > MAX_REFERENCES:
                raise ProjectError(too_many_references(len(refs)))
            refs += [self._write_numbered(project, prop_id, "reference", "reference", ext, data)
                     for ext, (data, _name) in zip(exts, files)]
            return copy.deepcopy(prop)

    def remove_references(self, project: str, prop_id: str, rel: str | None = None) -> dict[str, Any]:
        """Remove one example picture of the prop (`rel`, as images.references has it), or all of them (None); their
        files are deleted. A picture the prop does not have (any more) changes nothing."""
        with self._editing(project) as proj:
            prop = _find_prop(proj, prop_id, copy_=False)
            refs = prop["images"]["references"]
            gone = list(refs) if rel is None else [r for r in refs if r == rel]
            prop["images"]["references"] = [r for r in refs if r not in gone]
            for old in gone:
                self._remove_file(project, prop_id, old)
            return copy.deepcopy(prop)

    def set_viewport(self, project: str, prop_id: str, data: bytes, mask: bytes | None = None) -> dict[str, Any]:
        """Store the picture "Re-Render Viewport" sends (the viewport's picture with the artist's marks painted over
        it, or the empty viewport painted on) as VIEWPORT_FILE, in place of the one before -> {"path": VIEWPORT_FILE},
        for the job's from_viewport. Smart Mask sends its `mask` along (where the artist brushed): stored as
        VIEWPORT_MASK -> "mask" too."""
        sent = {VIEWPORT_FILE: data} | ({VIEWPORT_MASK: mask} if mask is not None else {})
        for picture in sent.values():
            if not picture:
                raise ProjectError("The picture did not arrive. Press Re-Render Viewport again.")
            if len(picture) > MAX_VIEWPORT_BYTES:
                raise ProjectError(f"This picture is too big. It can be {MAX_VIEWPORT_BYTES // (1024 * 1024)} MB at "
                                   f"most.", 413)
            if sniff_image(picture) != ".png":
                raise ProjectError("The picture must be a PNG. Reload the page and press Re-Render Viewport again.", 415)
        with self._lock:                # the project itself does not change
            _find_prop(self._read(project), prop_id, copy_=False)
            folder = self.prop_dir(project, prop_id)
            for rel, picture in sent.items():
                (folder / rel).parent.mkdir(parents=True, exist_ok=True)
                _write_bytes_atomic(folder / rel, picture)
            return {"path": VIEWPORT_FILE} | ({"mask": VIEWPORT_MASK} if mask is not None else {})

    def apply_stage_patch(self, project: str, prop_id: str, stage: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Shallow-merge a stage result (from an adapter or the imesh viewer) into prop[stage].

        A new mesh or low poly makes the prop's projection out of date ("stale") until it is baked
        again; a projection result says itself whether it is (its inputs may change while it bakes), and
        drops "painted" (set_painted): the new bake's textures are unpainted.
        A new mesh also makes an existing retopo and Omni-Texture stale (Pixal3D always writes mesh/mesh.glb, so
        their source path alone cannot tell) until a new one arrives. A new Omni-Texture makes the bake stale too
        while the prop uses it (omnitexture_used), says itself whether it is stale (like a bake), and its older
        ot_NNN folders go (_tidy_omnitextures).
        New clean meshes (retopo "variants": a job's, or the viewer's Accept) join or replace the prop's, and the
        patch then names the one the prop uses and whether it is accepted (_variants_patch); a clean mesh that is
        none of its variants is the only one of a new set, accepted only when the patch says so. Their "method" is
        the tool that made them (Instant Meshes when the patch does not say).
        A mesh that was not imported (Pixal3D made it) drops the details of an earlier import, and one that does not
        say which geometry made it (an import) drops the geometry of the one before.
        A new result drops the stage's thumbnail, so the page renders it again.
        A draft whose images get a chosen picture (a generation job's first result) becomes a prop.
        New images candidates join the prop's current ones (pictures imported while the job ran stay).
        """
        if not isinstance(patch, dict):
            raise TypeError(f"stage patch must be a dict, got {type(patch).__name__}")
        with self._editing(project) as proj:
            prop = _find_prop(proj, prop_id, copy_=False)
            block = prop.get(stage) if isinstance(prop.get(stage), dict) else {}
            patch = copy.deepcopy(patch)
            if stage == "images" and isinstance(patch.get("candidates"), list):
                # The pictures a job made join those the prop has now: pictures imported while it ran (and
                # their views) stay, although the job's list was made from the prop as it was at its start.
                now = [c for c in block.get("candidates") or [] if c not in patch["candidates"]]
                patch["candidates"] = [*patch["candidates"], *now]
            if stage == "retopo" and "variants" in patch:
                patch = _variants_patch(block, patch)
            elif stage == "retopo" and patch.get("obj") \
                    and all(variant.get("obj") != patch["obj"] for variant in block.get("variants") or []):
                # A clean mesh that is none of the prop's variants: the only one of its new set.
                stats = patch.get("stats") if isinstance(patch.get("stats"), dict) else {}
                patch["variants"] = [{"size": retopo_size_of(stats.get("vertices")), "obj": patch["obj"],
                                      "stats": patch.get("stats")}]
            if stage == "retopo" and "obj" in patch:
                _make_room_for_retopo(block, patch.get("method") if patch.get("method") in RETOPO_METHODS
                                      else FIRST_RETOPO_METHOD)
            was_stale = block.get("stale") is True
            was_used = omnitexture_used(prop)
            block.update(patch)
            prop[stage] = block
            if stage == "mesh" and "glb" in patch and "imported" not in patch:
                block["imported"] = None
            if stage == "mesh" and "glb" in patch and "geometry" not in patch:
                block["geometry"] = None
            if stage == "projection" or (stage == "omnitexture" and "glb" in patch):
                # A new result says itself whether its inputs changed while it was made. A repaint of a
                # Omni-Texture, or the fade of one, is no new result: it never makes a stale one fresh (the 3D
                # model may have been made again while it was put on).
                block["stale"] = bool(patch.get("stale", False)) \
                    or (stage == "omnitexture" and was_stale and bool(patch.get("steps")))
            if stage == "projection" and "glb" in patch:
                block.pop("painted", None)
            if stage != "projection" and _PROJECTION_INPUTS.get(stage) in patch and prop["projection"].get("glb") \
                    and (stage != "omnitexture" or was_used or omnitexture_used(prop)):   # one not used changes no bake
                prop["projection"]["stale"] = True
            if stage == "retopo" and "obj" in patch:
                block["stale"] = False
                block["accepted"] = patch.get("accepted") is True
                block["method"] = patch.get("method") if patch.get("method") in RETOPO_METHODS \
                    else FIRST_RETOPO_METHOD
            elif stage == "mesh" and "glb" in patch:
                # Every clean mesh of the prop, the ones kept for its other methods too.
                for result in (prop["retopo"], *_kept_retopos(prop["retopo"]).values()):
                    if result.get("obj"):
                        result["stale"] = True
                    result.pop("bake_stale", None)
                if prop["omnitexture"].get("glb"):
                    prop["omnitexture"]["stale"] = True
            if _THUMB_SOURCE.get(stage) in patch:
                prop["thumbs"].pop(stage, None)
            if stage == "images" and prop["draft"] and block.get("chosen"):
                _pack(proj, prop)
            if stage == "omnitexture" and patch.get("glb"):
                self._tidy_omnitextures(project, prop_id, block)
            return copy.deepcopy(prop)

    def _tidy_omnitextures(self, project: str, prop_id: str, block: dict[str, Any]) -> None:
        """A new Omni-Texture arrived: the prop's older omnitexture/ot_NNN folders go, but for the newest
        KEEP_OMNITEXTURES and those the prop still shows something of - its model, its painted views (a repaint
        keeps the views of the result it was made from), the steps Ctrl+Z walks back through and what their models
        are made of (the repaints' and the results made again - with the one that holds the painted views while
        they are hidden). Best effort: a file another program holds open stays."""
        folder = self.prop_dir(project, prop_id) / OMNITEXTURE_FOLDER
        bases = [made for made in block.get("bases") or [] if isinstance(made, dict)]
        shown = [block.get("glb"), *(block.get("steps") or []), *(block.get("views") or []),
                 *[sheet.get("folder") for sheet in block.get("sheets") or [] if isinstance(sheet, dict)],
                 *[made.get(key) for made in block.get("repaints") or [] if isinstance(made, dict)
                   for key in ("picture", "under")],
                 *[made.get(key) for made in bases for key in ("glb", "shown")],
                 *[view.get(key) for made in bases for view in made.get("views") or [] if isinstance(view, dict)
                   for key in ("picture", "layer")]]
        own = {PurePosixPath(path).parts[1] for path in shown
               if isinstance(path, str) and len(PurePosixPath(path).parts) > 1}
        try:
            numbered = sorted((int(m.group(1)), entry) for entry in folder.iterdir()
                              if entry.is_dir() and (m := _OMNITEXTURE_DIR.match(entry.name)))
        except OSError:
            return
        for _number, entry in numbered[:-KEEP_OMNITEXTURES]:
            if entry.name not in own:
                shutil.rmtree(entry, ignore_errors=True)

    def omnitexture_mask(self, project: str, prop_id: str, glb: Any, data: bytes) -> dict[str, Any]:
        """What the artist masked on the prop's OmniTextured model `glb`, a PNG of its texture's texels (white:
        masked), kept for the repaint job that follows (adapters/omnitexture.py retouch) -> {"mask": its path in
        the prop's folder}. 404 without an Omni-Texture, 409 when `glb` is no longer the one shown, 422 no PNG."""
        with self._lock:
            prop = self.get_prop(project, prop_id)
            shown = prop["omnitexture"].get("glb")
            if not shown:
                raise ProjectError("This prop has nothing to repaint yet. Reload the page and mask it again.", 404)
            if glb != shown:
                raise ProjectError("This prop's Omni-Texture changed meanwhile. Reload the page and mask it again.", 409)
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ProjectError("The mask did not arrive as a PNG picture. Reload the page and try again.", 422)
            path = self.prop_dir(project, prop_id) / OMNITEXTURE_MASK
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_bytes_atomic(path, data)
            return {"mask": OMNITEXTURE_MASK}

    def omnitexture_step(self, project: str, prop_id: str, glb: Any) -> dict[str, Any]:
        """Ctrl+Z / Ctrl+Y on the page: the prop shows another of its Omni-Texture's steps (`glb`: one of
        omnitexture.steps - the result, a repaint of it, or TAKEN_OFF: none of them) -> the prop. Its thumbnail is
        drawn again, and a bake made from the step it showed, or to be made from this one, is out of date. 404 when
        `glb` is not a step or its file is gone."""
        with self._editing(project) as proj:
            prop = _find_prop(proj, prop_id, copy_=False)
            block = prop["omnitexture"]
            if not isinstance(glb, str) or glb not in (block.get("steps") or []) \
                    or not (glb == TAKEN_OFF or (self.prop_dir(project, prop_id) / glb).is_file()):
                raise ProjectError("That step of the Omni-Texture is no longer there.", 404)
            if glb != block.get("glb"):
                was_used = omnitexture_used(prop)
                block["glb"] = glb
                prop["thumbs"].pop("omnitexture", None)
                if prop["projection"].get("glb") and (was_used or omnitexture_used(prop)):
                    prop["projection"]["stale"] = True
            return copy.deepcopy(prop)

    def omnitexture_paint_target(self, project: str, prop_id: str, glb: Any, width: int, height: int) -> Path:
        """The model a hand painting of the prop's Omni-Texture goes into (PUT .../omnitexture/paint), checked ->
        its file: 404 without an Omni-Texture, 409 when `glb` is none of its kept steps (taken back and gone, or
        made again by a fade) or its file is gone, 422 when width x height is not its texture's size."""
        block = self.get_prop(project, prop_id)["omnitexture"]
        if not block.get("glb"):
            raise ProjectError("This prop has nothing to paint on yet. Reload the page and paint again.", 404)
        if not isinstance(glb, str) or glb not in (block["glb"], *(block.get("steps") or [])):
            raise ProjectError(NOT_A_STEP, 409)
        try:
            model = self.resolve_file(project, f"props/{prop_id}/{glb}")
        except ProjectError as e:
            raise ProjectError(NOT_A_STEP, 409) from e
        side = (block.get("stats") or {}).get("texture")     # every step's: they are copies of one 3D model
        if isinstance(side, int) and (width, height) != (side, side):
            raise ProjectError(f"The painting is {width} x {height} pixels, but this Omni-Texture's texture is "
                               f"{side} x {side}. Reload the page and paint again.", 422)
        if not (0 < width <= MAX_PAINT_SIZE and 0 < height <= MAX_PAINT_SIZE):
            raise ProjectError(f"The painting is {width} x {height} pixels; a texture is {MAX_PAINT_SIZE} x "
                               f"{MAX_PAINT_SIZE} at most.", 422)
        return model

    def omnitexture_painted(self, project: str, prop_id: str, glb: str) -> dict[str, Any]:
        """A hand painting was saved into the model of the step `glb` -> the prop. As when a repaint changes the
        model shown: its thumbnail is drawn again, and a bake made from it is out of date (while the prop uses its
        Omni-Texture, or whenever the bake was made from that very file)."""
        with self._editing(project) as proj:
            prop = _find_prop(proj, prop_id, copy_=False)
            prop["omnitexture"]["painted"] = True
            shown = prop["omnitexture"].get("glb")
            bake = prop["projection"]
            if glb == shown:
                prop["thumbs"].pop("omnitexture", None)
            if bake.get("glb") and (glb == shown and omnitexture_used(prop) or bake.get("source_mesh") == glb):
                bake["stale"] = True
            return copy.deepcopy(prop)

    def pick_retopo(self, project: str, prop_id: str, size: Any) -> dict[str, Any]:
        """The artist picks the prop's clean mesh of `size` (one of retopo.variants): the prop uses it from now on,
        accepted -> the prop. Another one than it used makes a bake out of date, as a new clean mesh does. 409 while
        it has no such clean mesh up to date, or its file is gone."""
        missing = ProjectError("This clean mesh is not there to pick (it is not made yet, or was made from an older "
                               "3D model). Make it first.", 409)
        size = _retopo_size(size)
        with self._lock:
            prop = self.get_prop(project, prop_id)
            block = prop["retopo"]
            variant = next((variant for variant in block["variants"] if variant["size"] == size), None) \
                if retopo_fresh(prop) and size is not None else None
            if variant is None:
                raise missing
            try:
                self.resolve_file(project, f"props/{prop_id}/{variant['obj']}")
            except ProjectError:
                raise missing from None
            patch: dict[str, Any] = {"accepted": True}
            if variant["obj"] != block["obj"]:
                patch.update(obj=variant["obj"], stats=variant["stats"], method=block["method"])
            return self.apply_stage_patch(project, prop_id, "retopo", patch)

    # ---- thumbnails ------------------------------------------------------------------------------
    def set_thumb(self, project: str, prop_id: str, stage: str, data: bytes,
                  crop: list[int] | None = None, of: str | None = None) -> dict[str, Any]:
        """Store the page's snapshot of a prop in the form `stage` gave it (props/<id>/thumbs/<stage>.png
        or .webp) -> the prop, whose thumbs[stage] = {"path", "v"} (v changes with every new snapshot), and
        "crop": the part of the chosen picture an images thumbnail shows (thumb_crop), when not all of it.
        An images thumbnail says what it shows (`of`, kept as "of"): it is stored only while that is what the prop
        shows now (thumb_signature). One of another picture, or of another part of it (drawn before the artist
        chose again, a generation finished or the views changed) is not stored: the prop comes back as it is."""
        if stage not in THUMB_STAGES:
            raise ProjectError(f'There is no stage called "{stage}".', 404)
        if not data:
            raise ProjectError("The thumbnail is empty.")
        if len(data) > MAX_THUMB_BYTES:
            raise ProjectError(f"A thumbnail can be {MAX_THUMB_BYTES // (1024 * 1024)} MB at most.", 413)
        ext = sniff_image(data)
        if ext not in (".png", ".webp"):
            raise ProjectError("A thumbnail must be a PNG or WEBP image.", 415)
        with self._editing(project) as proj:
            prop = _find_prop(proj, prop_id, copy_=False)
            if stage == "images" and of != thumb_signature(prop["images"]):
                return copy.deepcopy(prop)
            folder = self.prop_dir(project, prop_id) / "thumbs"
            folder.mkdir(parents=True, exist_ok=True)
            _write_bytes_atomic(folder / f"{stage}{ext}", data)
            for other in {".png", ".webp"} - {ext}:
                (folder / f"{stage}{other}").unlink(missing_ok=True)
            previous = prop["thumbs"].get(stage, {}).get("v")
            stamp = int(time.time() * 1000)
            prop["thumbs"][stage] = {"path": f"thumbs/{stage}{ext}",
                                     "v": max(stamp, previous + 1) if isinstance(previous, int) else stamp}
            if stage == "images" and of:
                prop["thumbs"][stage]["of"] = of
                crop = thumb_crop(prop["images"])   # the part `of` names
            if crop:
                prop["thumbs"][stage]["crop"] = list(crop)
            return copy.deepcopy(prop)

    # ---- the artist's painting on a bake --------------------------------------------------------------
    def paint_target(self, project: str, prop_id: str, glb: Any, width: int, height: int) -> Path:
        """The bake folder a painting of the prop's base colour map goes into (PUT .../projection/paint),
        checked: 404 when the prop is not baked or its files are gone, 409 when `glb` is no longer its bake's
        GLB (baked again meanwhile), 422 when width x height is not the size of its maps/basecolor.png."""
        prop = self.get_prop(project, prop_id)
        baked = prop["projection"].get("glb")
        if not baked:
            raise ProjectError("This prop has not been baked yet, so there is nothing to paint on. Project its "
                               "textures in the Projection stage first.", 404)
        if glb != baked:
            raise ProjectError(BAKED_AGAIN, 409)
        try:
            model = self.resolve_file(project, f"props/{prop_id}/{baked}")
            basecolor = self.resolve_file(project, f"props/{prop_id}/{PurePosixPath(baked).parent}/maps/basecolor.png")
        except ProjectError as e:
            raise ProjectError("The baked files of this prop are missing, so the painting could not be saved. "
                               "Bake it again.", 404) from e
        size = _png_size(basecolor)
        if size is None:
            raise ProjectError("This bake's base colour map is not a PNG picture, so it cannot be painted. Bake "
                               "it again.", 409)
        if size != (width, height):
            raise ProjectError(f"The painting is {width} x {height} pixels, but this bake's base colour map is "
                               f"{size[0]} x {size[1]}. Reload the page and paint again.", 422)
        return model.parent

    def unpainted_copy(self, project: str, prop_id: str) -> Path | None:
        """Where the prop's bake keeps its own base colour map before a painting is first saved over it
        (maps/UNPAINTED_BASECOLOR), or None when there is nothing to keep: no bake, or a painted one
        (projection.painted: its map is a painting by now, never to be kept as the bake's own). 404 when the
        bake's map is gone."""
        block = self.get_prop(project, prop_id)["projection"]
        if not block.get("glb") or "painted" in block:
            return None
        bake = PurePosixPath(block["glb"]).parent
        return self.resolve_file(project, f"props/{prop_id}/{bake}/maps/basecolor.png").with_name(UNPAINTED_BASECOLOR)

    def set_painted(self, project: str, prop_id: str, glb: str) -> dict[str, Any]:
        """The painting is saved into the bake `glb` -> the prop: projection.painted is a new stamp (milliseconds,
        it only ever grows) and the projection thumbnail is dropped, so the page renders it again. "stale" stays
        as it is. 409 when the prop was baked again meanwhile."""
        with self._editing(project) as proj:
            prop = _find_prop(proj, prop_id, copy_=False)
            block = prop["projection"]
            if not block.get("glb") or block["glb"] != glb:
                raise ProjectError(BAKED_AGAIN, 409)
            previous = block.get("painted")
            stamp = int(time.time() * 1000)
            block["painted"] = max(stamp, previous + 1) if isinstance(previous, int) else stamp
            prop["thumbs"].pop("projection", None)
            return copy.deepcopy(prop)

    # ---- results, for saving them (exports.py) ----------------------------------------------------
    def projection_files(self, project: str, prop: dict[str, Any]) -> dict[str, Any]:
        """A baked prop's files: {"dir": the bake folder, "glb", "fbx" (or None), "obj" (or None), "maps":
        [(map id, file)]}, every one checked to exist inside the project. 404 when it is not baked."""
        block = prop["projection"]
        if not block.get("glb"):
            raise ProjectError("This prop has not been baked yet. Project its textures in the Projection "
                               "stage first.", 404)

        def file(rel: str) -> Path:
            return self.resolve_file(project, f"props/{prop['id']}/{rel}")

        glb = file(block["glb"])
        return {"dir": glb.parent, "glb": glb, "fbx": file(block["fbx"]) if block.get("fbx") else None,
                "obj": file(block["obj"]) if block.get("obj") else None,
                "maps": [(str(m.get("id") or Path(m["path"]).stem), file(m["path"])) for m in block.get("maps") or []]}

    def result_file(self, project: str, prop: dict[str, Any], stage: str, key: str) -> Path | None:
        """The file of prop[stage][key] (e.g. "mesh", "glb"), or None when there is none or it is missing."""
        rel = (prop.get(stage) or {}).get(key)
        if not rel:
            return None
        try:
            return self.resolve_file(project, f"props/{prop['id']}/{rel}")
        except ProjectError:
            return None

    # ---- the artist's own 3D model ------------------------------------------------------------
    def new_model_import(self, project: str, prop_id: str) -> tuple[Path, Path]:
        """Create mesh/import_NNN/ for uploaded model files -> (that folder, mesh/imported_NNN.glb)."""
        with self._lock:
            mesh_dir = self.prop_dir(project, prop_id) / "mesh"
            number = _next_import_number(mesh_dir)
            folder = mesh_dir / f"import_{number:03d}"
            folder.mkdir(parents=True)
            return folder, mesh_dir / f"imported_{number:03d}.glb"

    def next_imported_glb(self, project: str, prop_id: str) -> Path:
        """A new mesh/imported_NNN.glb name, for converting an imported model again."""
        with self._lock:
            mesh_dir = self.prop_dir(project, prop_id) / "mesh"
            return mesh_dir / f"imported_{_next_import_number(mesh_dir):03d}.glb"

    def remove_other_imports(self, project: str, prop_id: str, keep: Iterable[str]) -> None:
        """Delete the mesh/import_NNN/ folders and mesh/imported_NNN.glb files that neither are nor hold one of
        `keep` (paths relative to the prop): copies of the artist's files, and meshes converted from them, that
        a newer import replaced. Best effort: a file another program holds open stays."""
        mesh_dir = self.prop_dir(project, prop_id) / "mesh"
        kept = [(self.prop_dir(project, prop_id) / rel).resolve() for rel in keep]
        if not mesh_dir.is_dir():
            return
        for entry in mesh_dir.iterdir():
            if not _IMPORT_ENTRY.match(entry.name) or any(path.is_relative_to(entry.resolve()) for path in kept):
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                try:
                    entry.unlink(missing_ok=True)
                except OSError as e:
                    log.info("Could not delete the replaced import %s (%s)", entry, e)

    # ---- files --------------------------------------------------------------------------------
    def resolve_file(self, project: str, rel_path: str) -> Path:
        """An existing file inside projects/<project>/ (never outside it), else a 404 ProjectError.

        Nothing outside the project is touched while checking: a rooted or UNC path
        ("//host/share/x", "\\\\host\\share") is refused before resolve(), which would otherwise
        open an SMB connection to that host (and hand it the user's Windows credentials).
        ':' is refused too (drives, and alternate data streams such as "a.png:stream").
        """
        base = self.project_dir(project).resolve()
        if not rel_path or ":" in rel_path or "\x00" in rel_path:
            raise ProjectError("File not found.", 404)
        rel = PureWindowsPath(rel_path)
        if rel.drive or rel.root or not Path(os.path.normpath(base / rel)).is_relative_to(base):
            raise ProjectError("File not found.", 404)
        target = (base / rel).resolve()   # follows junctions / links, hence the second check
        if not target.is_relative_to(base) or not target.is_file():
            raise ProjectError("File not found.", 404)
        # A tool's scratch files, whatever their names (Windows reads "Temp" and "temp." as the same folder).
        if target.relative_to(base).parts[0].rstrip(" .").casefold() == _TEMP:
            raise ProjectError("File not found.", 404)
        return target

    # ---- the tools' scratch space ---------------------------------------------------------------
    def work_dir(self, project: str, name: str) -> Path:
        """projects/<project>/temp/<name>/, made when it is not there: scratch space for one job
        ("<stage>-<job id>"). Whoever asked for it removes it (remove_work_dir)."""
        folder = self.project_dir(project) / _TEMP / _work_name(name)
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError:       # a sweep removed the empty temp/ at this very moment
            folder.mkdir(parents=True, exist_ok=True)
        return folder

    def remove_work_dir(self, project: str, name: str) -> bool:
        """Remove a work_dir with all that is in it, and temp/ itself once it is empty -> whether it is gone
        (a program that still has one of its files open keeps it: try again, or leave it to clear_work_dirs)."""
        name = _work_name(name)
        try:
            temp = self.project_dir(project) / _TEMP
        except ProjectError:
            return True
        shutil.rmtree(temp / name, ignore_errors=True)
        _remove_if_empty(temp)
        return not (temp / name).exists()

    def clear_work_dirs(self, project: str | None = None,
                        keep: str | Callable[[str], str | None] | None = None) -> int:
        """Remove what is left in temp/ of `project`, or of every project (None), but the work folder `keep`
        (a running job's) -> how many entries went. Nothing else of a project is touched. What cannot be
        removed now (a file still open in a tool that is stopping) stays for the next time.

        keep: the folder's name, or a function that gives it for a project. The function is asked once the
        leftovers are listed, so the folder of a job that starts at this very moment is never among them: it
        is either named by then, or was made after the listing."""
        if project is not None:
            folders = [self.project_dir(project)]
        else:
            try:
                folders = [d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")]
            except OSError:
                folders = []    # no projects folder yet
        removed = 0
        for folder in folders:
            temp = folder / _TEMP
            try:
                entries = list(temp.iterdir())
            except OSError:
                continue        # no temp folder: nothing was left
            spare = keep(folder.name) if callable(keep) else keep
            for entry in entries:
                if entry.name == spare:
                    continue
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    try:
                        entry.unlink()
                    except OSError:
                        pass
                if not entry.exists():
                    removed += 1
                else:
                    log.info("Could not remove the leftover work folder %s yet", entry)
            _remove_if_empty(temp)
        if removed:
            log.info("Removed %d leftover work folder(s) of %s", removed, project or "all projects")
        return removed

    def clear_partial_meshes(self) -> int:
        """Remove the half-written 3D models in every prop's mesh folder (props/*/mesh/mesh.*.partial.glb: Pixal3D
        writes a new model there and the hub renames it to mesh.glb once it is complete) -> how many went. Only
        for the hub's start, when no job can be writing one: a hub that was closed or crashed in the middle of a
        3D job leaves them (a prop's next 3D model clears its own, but a prop never made again kept one)."""
        try:
            folders = [d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        except OSError:
            return 0    # no projects folder yet
        removed = 0
        for folder in folders:
            for partial in folder.glob("props/*/mesh/mesh.*.partial.glb"):
                try:
                    partial.unlink()
                    removed += 1
                except OSError:
                    log.info("Could not remove the half-written model %s yet", partial)
        if removed:
            log.info("Removed %d half-written 3D model(s) of earlier runs", removed)
        return removed

    # ---- internals ----------------------------------------------------------------------------
    def _read(self, project: str) -> dict[str, Any]:
        file = self.project_dir(project) / "project.json"
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ProjectError(f'The project file of "{project}" cannot be read ({e}). '
                               f"Restore {file} from a backup or create a new project.", 500) from e
        if not isinstance(data, dict):
            raise ProjectError(f'The project file of "{project}" is damaged. '
                               f"Restore {file} from a backup or create a new project.", 500)
        return _normalize(data, project)

    @contextmanager
    def _editing(self, project: str) -> Iterator[dict[str, Any]]:
        """Load, let the caller modify, save atomically - all under the lock. Nothing is saved on error."""
        with self._lock:
            data = self._read(project)
            yield data
            self._write(project, data)

    def _write(self, project: str, data: dict[str, Any]) -> None:
        """Save a project read by _read (its "id" is the folder name, not stored). No prop is saved with the
        thumbnail of a picture it no longer shows, whatever changed it (_drop_stale_thumb)."""
        for prop in data.get("props") or []:
            _drop_stale_thumb(prop)
        data["updated"] = now_iso()
        atomic_write_json(self.project_dir(project) / "project.json", {k: v for k, v in data.items() if k != "id"})

    def _all_projects(self) -> Iterator[dict[str, Any]]:
        """Every readable project (folders starting with "." hold Spellcast3D's own files)."""
        if not self.root.is_dir():
            return
        for d in self.root.iterdir():
            if d.name.startswith(".") or not (d / "project.json").is_file():
                continue
            try:
                yield self._read(d.name)
            except ProjectError as e:
                log.warning("Skipping project %s: %s", d.name, e)

    def _create(self, folder: str, fields: dict[str, Any]) -> dict[str, Any]:
        d = self.root / folder
        (d / "props").mkdir(parents=True)
        now = now_iso()
        atomic_write_json(d / "project.json", {**fields, "created": now, "updated": now, "geometry": DEFAULT_GEOMETRY,
                                               "ui": copy.deepcopy(DEFAULT_UI), "props": []})
        return self._read(folder)

    def _create_untitled(self) -> dict[str, Any]:
        """projects/untitled-YYYYMMDD-HHMMSS/, shown as "Untitled" (or "Untitled 2"... when that is taken)."""
        with self._lock:
            stamp = f"untitled-{datetime.datetime.now():%Y%m%d-%H%M%S}"
            folder, number = stamp, 2
            while (self.root / folder).exists():
                folder, number = f"{stamp}-{number}", number + 1
            taken = {p["name"].casefold() for p in self.list_projects()}
            name, number = UNTITLED, 2
            while name.casefold() in taken:
                name, number = f"{UNTITLED} {number}", number + 1
            return self._create(folder, {"name": name, "untitled": True})

    def _state(self) -> dict[str, Any]:
        """The installation's own state (_STATE_FILE): the project opened last."""
        try:
            state = json.loads((self.root / _STATE_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return state if isinstance(state, dict) else {}

    def _save_state(self, **changes: Any) -> None:
        state = self._state()
        if any(state.get(k) != v for k, v in changes.items()):
            atomic_write_json(self.root / _STATE_FILE, {**state, **changes})

    def _last_opened(self) -> str | None:
        value = self._state().get("last_opened")
        return value if isinstance(value, str) else None

    def _set_last_opened(self, project: str) -> None:
        self._save_state(last_opened=project)

    def _geometry(self, data: dict[str, Any]) -> str:
        """A new prop's geometry in this project (`data`): the Shape picked last in it (default_geometry)."""
        return default_geometry(data.get("geometry"))

    def _trash_if_empty_untitled(self, project: str, busy: Collection[str]) -> None:
        """Move an untitled project without content (and without a job) to projects/.trash/."""
        try:
            data = self._read(project)
        except ProjectError:
            return
        if not data["untitled"] or project in busy or not _is_empty(data):
            return
        shutil.rmtree(self.root / project / _TEMP, ignore_errors=True)     # scratch files are no content
        dst = self.root / _TRASH / f"{project}_{datetime.datetime.now():%Y%m%d-%H%M%S}"
        try:
            dst.parent.mkdir(exist_ok=True)
            shutil.move(str(self.root / project), str(dst))
        except OSError as e:
            log.info("Could not tidy away the empty project %s (%s)", project, e)
            return
        log.info("Moved the empty untitled project %s to %s", project, dst)

    def _new_prop_id(self, project: str, props: list[dict[str, Any]]) -> str:
        taken = {p["id"] for p in props}
        while True:
            prop_id = "p_" + secrets.token_hex(3)
            if prop_id not in taken and not self.prop_dir(project, prop_id).exists():
                return prop_id

    def _write_numbered(self, project: str, prop_id: str, folder: str, prefix: str, ext: str, data: bytes) -> str:
        """Write data to <prop>/<folder>/<prefix>_NNN<ext> (first free number); return the relative path."""
        d = self.prop_dir(project, prop_id) / folder
        d.mkdir(parents=True, exist_ok=True)
        used = {int(m.group(1)) for f in d.iterdir() if (m := re.match(rf"{prefix}_(\d+)\.", f.name))}
        n = max(used, default=0) + 1
        path = d / f"{prefix}_{n:03d}{ext}"
        path.write_bytes(data)
        return f"{folder}/{path.name}"

    def _remove_file(self, project: str, prop_id: str, rel: str | None) -> None:
        if not rel:
            return
        try:
            self.resolve_file(project, f"props/{prop_id}/{rel}").unlink()
        except (ProjectError, OSError):
            pass


def _work_name(name: str) -> str:
    """A work folder's name, checked: one plain folder name, never a path that could lead out of temp/."""
    if not isinstance(name, str) or not name or name in (".", "..") or PureWindowsPath(name).name != name:
        raise ValueError(f"not a folder name: {name!r}")
    return name


def _remove_if_empty(folder: Path) -> None:
    try:
        folder.rmdir()
    except OSError:
        pass        # not empty, or not there


def _rename_folder(src: Path, dst: Path) -> None:
    """os.rename, retried for a moment: an antivirus or indexer can hold a file inside briefly."""
    for attempt in range(10):
        try:
            os.rename(src, dst)
            return
        except PermissionError as e:
            if attempt == 9:
                raise ProjectError(f"The project's folder cannot be renamed right now ({e.strerror}). Close any "
                                   f"program or folder window that uses one of its files, then try again.",
                                   409) from e
            time.sleep(0.1)
        except OSError as e:
            raise ProjectError(f"The project's folder cannot be renamed ({e.strerror or e}). "
                               f"Try another name.", 409) from e


def _rename_exports(src: Path, dst: Path) -> None:
    """The saved models follow their renamed project, when there are any. Best effort: a folder already
    called `dst`, or one that another program holds open, leaves them where they are (logged)."""
    if src == dst or not src.is_dir():
        return
    try:
        if dst.exists() and not os.path.samefile(src, dst):
            log.info("Not renaming %s: %s exists already", src, dst)
            return
        for attempt in range(10):
            try:
                os.rename(src, dst)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.1)
    except OSError as e:
        log.info("Could not rename the saved models' folder %s to %s (%s)", src, dst, e)


def too_many_references(have: int) -> str:
    """Why the example pictures being added do not fit a prop that has `have` (at most MAX_REFERENCES)."""
    left = MAX_REFERENCES - have
    return (f"A prop can have at most {MAX_REFERENCES} example pictures"
            + (f", so {left} more can be added. Choose fewer." if left > 0 else ". Remove one before adding another."))


def _png_size(path: Path) -> tuple[int, int] | None:
    """(width, height) from a PNG file's header, or None when it is not a PNG."""
    try:
        with open(path, "rb") as file:
            head = file.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", head[16:24])


def _find_prop(data: dict[str, Any], prop_id: str, copy_: bool = True) -> dict[str, Any]:
    for prop in data["props"]:
        if prop["id"] == prop_id:
            return copy.deepcopy(prop) if copy_ else prop
    raise ProjectError("This prop no longer exists. It may have been deleted; reload the page.", 404)


def _check_image(data: bytes) -> str:
    if not data:
        raise ProjectError("The uploaded file is empty. Please choose an image file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ProjectError(f"This image is too big ({len(data) / 1e6:.0f} MB). "
                           f"The limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", 413)
    ext = sniff_image(data)
    if ext is None:
        raise ProjectError("This file is not a PNG, JPG or WEBP image. Please choose one of those.", 415)
    return ext


# ---- the artist's own 3D model: saving the uploaded files ------------------------------------------

@dataclass
class ModelFiles:
    """What save_model_files stored in an import folder."""
    files: list[str] = field(default_factory=list)     # paths inside the folder (forward slashes), arrival order
    skipped: list[str] = field(default_factory=list)   # names left out: neither a 3D model nor a texture
    model: str | None = None                           # the model to convert (one of `files`)


def save_model_files(folder: Path, uploads: Iterable[tuple[str, IO[bytes]]]) -> ModelFiles:
    """Store uploaded (name, stream) files in `folder`, unpacking .zip files into it, and pick the model.

    Names keep their sub-folders (a dropped folder or a .zip keeps "textures/wood.png" where the model
    looks for it) but are made safe for Windows. A name that climbs out of the folder ("../x", zip slip)
    refuses the whole upload. Everything is copied in chunks and counts against MAX_MODEL_UPLOAD_BYTES,
    unpacked bytes included, so a .zip bomb stops at the limit. The model is the first file chosen if
    that is a model, else the first by MODEL_SUFFIXES order, then the fewest sub-folders.

    Instant Meshes looks for a model's textures only inside the model's own folder, and an asset pack
    keeps them in a sibling one (Meshes/chair.fbx beside Textures/), so every file outside the model's
    folder is also linked into it by name (a name already there is left alone). `files` lists only
    what arrived.
    """
    writer = _ModelFileWriter(folder)
    first: str | None = None
    for index, (name, stream) in enumerate(uploads):
        if PurePosixPath(name.replace("\\", "/")).suffix.lower() == ".zip":
            writer.add_zip(name, stream)
            continue
        rel = writer.add(name, stream)
        if index == 0 and rel is not None and PurePosixPath(rel).suffix.lower() in MODEL_SUFFIXES:
            first = rel
    writer.found.model = first or _preferred_model(writer.found.files)
    if writer.found.model is not None:
        _gather_companions(writer.folder, writer.found)
    return writer.found


def _gather_companions(folder: Path, found: ModelFiles) -> None:
    """Link (or copy, where the drive cannot link) each file outside the model's folder into it."""
    assert found.model is not None
    model_folder = (folder / found.model).parent
    for rel in found.files:
        source = folder / rel
        if source.is_relative_to(model_folder):
            continue
        target = model_folder / source.name
        if target.exists():
            continue
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)


class _ModelFileWriter:
    def __init__(self, folder: Path):
        self.folder = folder.resolve()
        self.left = MAX_MODEL_UPLOAD_BYTES
        self.found = ModelFiles()

    def add(self, name: str, stream: IO[bytes]) -> str | None:
        """Copy one file into the folder -> its path there, or None when the import leaves it out."""
        rel = _import_path(name)
        if rel is None:
            self.found.skipped.append(name)
            return None
        if len(self.found.files) >= MAX_IMPORT_FILES:
            raise ProjectError(f"These are too many files (over {MAX_IMPORT_FILES:,}). Choose only the model "
                               f"and its textures.", 413)
        target = self.folder / rel
        if not target.resolve().is_relative_to(self.folder):
            raise _leads_outside(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as out:
            while chunk := stream.read(_COPY_CHUNK):
                self.left -= len(chunk)
                if self.left < 0:
                    raise ProjectError(f"These files are too big together (the limit is "
                                       f"{MAX_MODEL_UPLOAD_BYTES // 1024 ** 3} GB). Leave out what the model "
                                       f"does not need, or use smaller textures.", 413)
                out.write(chunk)
        if rel not in self.found.files:
            self.found.files.append(rel)
        return rel

    def add_zip(self, name: str, stream: IO[bytes]) -> None:
        try:
            archive = zipfile.ZipFile(stream)
        except (zipfile.BadZipFile, OSError) as e:
            raise ProjectError(f'"{name}" is not a .zip file that can be opened (it may be damaged).') from e
        with archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if member.flag_bits & 0x1:
                    raise ProjectError(f'"{name}" is password-protected. Unpack it yourself and choose '
                                       f"the files instead.")
                with archive.open(member) as src:
                    self.add(member.filename, src)


def _import_path(name: str) -> str | None:
    """An uploaded file's (or .zip member's) name as a safe path inside the import folder, or None
    for a file the import leaves out. Raises ProjectError when the name climbs out of the folder."""
    parts = [part for part in name.replace("\\", "/").split("/") if part not in ("", ".")]
    if parts and re.fullmatch(r"[A-Za-z]:", parts[0]):
        parts = parts[1:]           # "C:/models/x.fbx": kept as models/x.fbx, like zipfile does
    if ".." in parts:
        raise _leads_outside(name)
    if not parts or parts[0] == "__MACOSX" or parts[-1].startswith("._"):
        return None                 # macOS resource forks: "._wood.png" is not an image
    suffix = PurePosixPath(parts[-1]).suffix.lower()
    if suffix not in MODEL_SUFFIXES and suffix not in MODEL_COMPANION_SUFFIXES:
        return None
    parts = [_safe_path_part(part) for part in parts]
    if len("/".join(parts)) > _MAX_IMPORT_PATH:
        parts = parts[-1:]          # textures are also found by their bare name
    return "/".join(parts)


def _safe_path_part(part: str) -> str:
    part = _PATH_UNSAFE_CHARS.sub("_", unicodedata.normalize("NFC", part)).strip().rstrip(". ")
    if len(part) > _MAX_IMPORT_NAME:
        suffix = PurePosixPath(part).suffix[:16]
        part = part[:_MAX_IMPORT_NAME - len(suffix)] + suffix
    if part.split(".")[0].strip().upper() in _RESERVED_NAMES:
        part = f"_{part}"
    return part or "_"


def _leads_outside(name: str) -> ProjectError:
    return ProjectError(f'"{name}" points outside its folder, so nothing was imported. Unpack the files '
                        f"yourself and choose them instead.")


def _preferred_model(files: list[str]) -> str | None:
    ranked = [(MODEL_SUFFIXES.index(suffix), rel.count("/"), index, rel) for index, rel in enumerate(files)
              if (suffix := PurePosixPath(rel).suffix.lower()) in MODEL_SUFFIXES]
    return min(ranked)[3] if ranked else None


def _next_import_number(mesh_dir: Path) -> int:
    used = [int(m.group(1)) for entry in mesh_dir.iterdir()
            if (m := _IMPORT_ENTRY.match(entry.name))] if mesh_dir.is_dir() else []
    return max(used, default=0) + 1
