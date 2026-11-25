"""Pixal3D tool server for Spellcast3D. Runs INSIDE the pixal3d venv (Python 3.11).

    python pixal3d_server.py --repo <code/pixal3d> --port P [--host 127.0.0.1]
                             [--low-vram on|off] [--attn-backend flash_attn|sdpa|xformers] [--sage auto|on|off] [--int8 auto|on|off]
                             [--projects-root <projects>] [--first-job single|multi]

Imports Pixal3D's app.py as a module instead of running it, because its
__main__ block pip-reinstalls utils3d from GitHub on every start and opens a
public share link. Then it adds the /spellcast/* routes the hub drives
(adapters/pixal3d.py) and serves Pixal3D's own UI at "/" on 127.0.0.1 only.
Multi-view generation lives in pixal3d_mv.py, the checks on what the model makes in
pixal3d_guard.py, the record of a batch in pixal3d_batch.py (all next to this file).

A start reads no model: it imports the libraries, makes the staged engine (the fork's
pixal3d/staged; build_engine) and answers. The hub sends all the props of a job in one
request (POST /spellcast/batch, BatchRunner); the engine takes them through its parts, each
part for every prop, with one network on the card at a time, and asks this file for what is
the launcher's own (BatchHooks): opening, cutting out and framing the pictures, the camera
estimate, the notes of the sparse-structure guard, and the mesh clean-up that writes the GLB. While the card cuts
out one prop, the next props' pictures are opened on CPU threads (OpenAhead).

The path Pixal3D itself ships - app.py's pipeline with every model in RAM - stays as the
reference (POST /spellcast/generate, Pixal3D's own page): its models are loaded the first
time one of them is called (Reference), and dropped again before the next batch.

Only the stdlib and what the pixal3d venv already has (torch, gradio 6.14,
fastapi, pydantic, numpy, PIL, OpenCV); never the hub-only packages.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import ctypes
import dataclasses
import functools
import gc
import importlib
import inspect
import json
import math
import os
import random
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Iterator
from urllib.parse import unquote, urlparse

from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

import pixal3d_batch                                        # launchers/ (this script's folder)
import pixal3d_glb
import pixal3d_guard
import pixal3d_mv
from _loopback_guard import LoopbackGuard, is_loopback

ATTN_BACKENDS = ("flash_attn", "sdpa", "xformers")
RESOLUTIONS = (1024, 1536)
GEOMETRIES = ("pixal3d", "hi3d")    # whose networks make a prop's shape; the texture is Pixal3D's either way
FIRST_JOBS = {"single": "sv", "multi": "mv"}   # --first-job (adapters/pixal3d.py MODES) -> pixal3d_mv's mode

# Progress session id of hub generations in app.py's file-based progress store.
SESSION = "spellcast"

# Stage names this module writes itself; the others come from the tqdm bars of
# pipeline.run (pixal3d_image_to_3d.py) and o_voxel.postprocess.to_glb, which
# app.py routes into its progress store.
REMOVING_BACKGROUND = "Removing background"
OWN_TRANSPARENCY = "Own transparency"
ESTIMATING_CAMERA = "Estimating camera"
DECODING = "Decoding the latents"
SAVING = "Saving GLB"

# (tqdm description prefixes, label for the artist, overall fraction at start, at end). The steps both paths go
# through take their words and their share of a prop's bar from the batch's parts (pixal3d_batch.PARTS), so the
# reference path and a batch cannot say different things; a reference step that spans several parts (its shape
# sampling conditions and upsamples inside the bar) spans their shares. Only the bars of the mesh clean-up, which
# a batch reports through this table too (route_progress), are this table's own.
_PART = pixal3d_batch.PART
SAVING_FROM = 0.98                # the clean-up's bars end here; writing the file takes the rest of "finish"
PHASES: tuple[tuple[tuple[str, ...], str, float, float], ...] = (
    # Only when this Pixal3D last held the other set (a start loads the set its first job needs).
    ((pixal3d_mv.SWITCHING["mv"],), "Swapping the single-view models for the multi-view ones", 0.00, 0.03),
    ((pixal3d_mv.SWITCHING["sv"],), "Swapping the multi-view models for the single-view ones", 0.00, 0.03),
    ((REMOVING_BACKGROUND,), _PART["cutout"].label, _PART["cutout"].start, _PART["cutout"].end),
    ((OWN_TRANSPARENCY,), "Using the picture's own transparency", _PART["cutout"].start, _PART["cutout"].end),
    ((ESTIMATING_CAMERA,), _PART["camera"].label, _PART["camera"].start, _PART["features"].end),
    (("Sampling sparse structure",), _PART["ss"].label, _PART["ss"].start, _PART["ss"].end),
    (("Sampling shape SLat",), _PART["lr"].label, _PART["cond_lr"].start, _PART["upsample"].end),
    (("Sampling HR shape SLat",), _PART["hr"].label, _PART["cond_hr"].start, _PART["hr"].end),
    (("Sampling texture SLat",), _PART["tex"].label, _PART["tex"].start, _PART["tex"].end),
    ((DECODING,), f"{_PART['finish'].label} (this step can take a few minutes)", _PART["finish"].start,
     pixal3d_batch.DECODED),
    (("Extracting GLB", "Building BVH", "Cleaning mesh", "Parameterizing new mesh",
      "Sampling attributes", "Finalizing mesh"), "Cleaning up the mesh and baking textures", pixal3d_batch.DECODED,
     SAVING_FROM),
    ((SAVING,), "Saving the GLB file", SAVING_FROM, _PART["finish"].end),
)

MODELS_MISSING = ("Some of Pixal3D's AI model files are missing (they belong in the models\\pixal3d folder of "
                  "Spellcast3D). Close Spellcast3D and start it again: it downloads what is missing in the "
                  "background, then make the 3D model again.")
OUTSIDE_PROJECTS = "Spellcast3D only makes meshes from and into its own projects folder."
# The hub shows its own words for a prop that ran out of video memory (adapters/pixal3d.py OUT_OF_VIDEO_MEMORY, the
# same sentence): it knows the prop's quality, and leaves out "or lower Quality" at the lowest one. This one is for
# the launcher's own route, its log and the record of a batch that failed as a whole.
OUT_OF_GPU_MEMORY = ("This prop needs more video memory than the card has free. Close other programs that use the "
                     "graphics card, or lower Quality.")
OUT_OF_RAM = ("Pixal3D ran out of system memory (RAM and page file). Close other programs and try again. If it "
              "keeps failing, let Windows manage the page file size (search Windows for 'Adjust the performance "
              "of Windows' > Advanced > Virtual memory).")
BATCH_STOPPED = "Pixal3D stopped before this job was done. Try making these 3D models again."
# A prop the engine made at less detail than its token cap allows, so that it fits the card or the RAM (the
# "memory_cap" note of pixal3d.staged, see docs/staged_engine.md): told with the model, like a streak.
LOWER_DETAIL = "Made at quality {made} instead of {asked}: {room} could not hold more."
LOWER_DETAIL_ROOM = {"card": "the graphics card's free memory", "ram": "the PC's free memory"}


# The same where the decoders cannot decode in boxes because the install's check of them did not pass (below sm_80:
# no usable decode_products section, the fork's products_floor.problem). The next start measures again only where
# the hub says so at the launch (CALIBRATION_DUE_ENV); else the words name the Settings button alone.
LOWER_DETAIL_UNMEASURED = ("Made at quality {made} instead of {asked}: this graphics card's one-time measurement did "
                           "not finish, and without it the card's memory cannot hold more. {again}")
CALIBRATION_DUE_ENV = "SPELLCAST_PIXAL3D_CALIBRATION_DUE"
MEASURED_AGAIN = {      # the hub's components.MEASURED_AT_NEXT_START / MEASURED_ON_REQUEST
    True: ("Spellcast3D measures it again at its next start, or close heavy programs and press \"Measure the graphics "
           "card again\" in Settings > Debugging."),
    False: ("To measure it again, close heavy programs and press \"Measure the graphics card again\" in Settings > "
            "Debugging.")}


def decode_unmeasured() -> bool:
    """Whether this process's decoders run on spconv without a usable decode-in-boxes record (the 1024 rule)."""
    try:
        from pixal3d.modules import precision
        from pixal3d.modules.sparse.conv import products_floor

        return precision.profile().conv == "spconv" and bool(products_floor.problem())
    except Exception:       # noqa: BLE001 - a fork without these: the plain words
        return False


def lower_detail_words(cap: dict[str, Any]) -> str:
    if cap.get("by") == "card" and decode_unmeasured():
        again = MEASURED_AGAIN[os.environ.get(CALIBRATION_DUE_ENV) == "1"]
        return LOWER_DETAIL_UNMEASURED.format(made=cap.get("resolution"), asked=cap.get("without"), again=again)
    room = LOWER_DETAIL_ROOM.get(str(cap.get("by")), "the free memory")
    return LOWER_DETAIL.format(made=cap.get("resolution"), asked=cap.get("without"), room=room)
# A prop's id names its folder inside the batch's work folder.
PROP_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
# A batch's cut-out opens the pictures of the next OPEN_AHEAD props while the card cuts out the one before them, on
# OPEN_THREADS threads (OpenAhead). A view's matte takes 0.6-0.7 s; opening its picture 5-15 ms from the file cache,
# more from the disk or for a large photo (decoded, then reduced to pixal3d_mv.MAX_SLICE px).
OPEN_AHEAD = 2
OPEN_THREADS = 2

# app.py extract_glb_api's orientation fix (+Y up) turned 180 degrees about +Y, so the side the picture
# shows (Pixal3D's camera) faces +Z, glTF's front: viewers and thumbnails show the front first. With
# to_glb's own axis swap this keeps the generator's axes: x to the picture's right, y up, z toward the
# camera. Pixal3D's own page still exports its GLBs facing -Z.
GLB_ROTATION = ((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1))


class ViewSpec(BaseModel):
    image_path: str
    rect: list[int] | None = None     # [x, y, w, h] in the picture's pixels; None = all of it
    azimuth: int = 0                  # 0 front | 90 left side (faces left) | 180 back | 270 right side


class GenerateRequest(BaseModel):
    image_path: str
    out_path: str
    seed: int = 0                     # < 0 = random (/spellcast/generate; a batch refuses it: BatchItem)
    resolution: int = 1024            # 1024 | 1536
    decimation_target: int = 300_000
    texture_size: int = 2048
    views: list[ViewSpec] | None = None   # 1: that slice alone; 2+: multi-view (a front one required)
    views_dir: str | None = None          # where to save the views the model saw (<n>_<view>.png)
    geometry: str = "pixal3d"             # GEOMETRIES (a batch only; /spellcast/generate is Pixal3D's own)


class BatchItem(GenerateRequest):
    # The seed must be 0 or more here (BatchRunner refuses a batch with a negative one): a seed drawn per request
    # would give the job's second attempt another fingerprint, and nothing sampled could be resumed. The hub
    # draws a random one itself, once per job (adapters/pixal3d.py _request).
    id: str                           # the hub's prop id


class BatchRequest(BaseModel):
    batch_id: str                     # the hub's job id
    work_dir: str                     # <projects>/<project>/temp/mesh-<job id>: the engine's temporary files
    wave_size: int | None = None      # props taken through the parts together; None / 0: all of them
    resume: bool = False              # the same job again after a restart: keep the props whose sampling was done
    # True: a prop that does not fit into the card's free memory may spill into shared memory, slowly; False: it
    # fails with the kind "oom" once the engine's memory ladder is through (the hub's pixal3d.allow_spill)
    spill: bool = False
    items: list[BatchItem]


class RunNotes:
    """What a generation noticed on its way, for the hub's answer. Written on the GPU worker."""

    def __init__(self) -> None:
        self.ss_retries = 0
        self.ss_blocked = False     # every draw of the sparse structure was a block: the best one was kept

    def reset(self) -> None:
        self.ss_retries, self.ss_blocked = 0, False


class HubJob:
    """What GET /spellcast/progress reports. Written by the generate worker thread."""

    def __init__(self) -> None:
        self.active = False
        self.waiting = False
        self.label = ""
        self.fraction = 0.0

    def start(self) -> None:
        self.active, self.waiting, self.label, self.fraction = True, True, "Starting", 0.0


# ---- environment (must happen before app.py and torch are imported) -------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pixal3D server for Spellcast3D")
    p.add_argument("--repo", required=True, help="Pixal3D checkout (code/pixal3d)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--low-vram", choices=("on", "off"), default="on",
                   help="the reference path only (Pixal3D's own page, /spellcast/generate): keep its models in "
                        "RAM and move each one to the GPU only while it runs")
    p.add_argument("--attn-backend", choices=ATTN_BACKENDS, default="flash_attn")
    p.add_argument("--sage", choices=("auto", "on", "off"), default="auto",
                   help="SageAttention for the flows' self-attention (the fork's PIXAL3D_SAGE): auto = the fork's "
                        "default (on, after its self-check, on sm_80+ cards), on, off (flash_attn, bit for bit)")
    p.add_argument("--int8", choices=("auto", "on", "off"), default="auto",
                   help="the flows' linear layers as int8 tensor-core GEMMs (the fork's PIXAL3D_INT8): auto = the "
                        "fork's default (on, after its self-check, on sm_80+ cards), on, off (bf16, bit for bit)")
    p.add_argument("--coarse-exact", choices=("on", "off"), default="on",
                   help="the conditioning and the coarse stages computed as the original code does, so that a seed "
                        "gives its model (the fork's PIXAL3D_COARSE_EXACT; sm_80+ only): on, or off (their fast "
                        "paths too, a few seconds a prop less)")
    p.add_argument("--projects-root", default=None,
                   help="the hub's projects folder: batches and /spellcast/generate read and write only inside it")
    p.add_argument("--first-job", choices=tuple(FIRST_JOBS), default="single",
                   help="the reference path only: which models its first use loads, the single-view ones or "
                        "the multi-view ones (the other set replaces them when a job needs it)")
    return p.parse_args(argv)


def configure_env(repo: Path, low_vram: bool, attn_backend: str, sage: str = "auto", int8: str = "auto",
                  coarse_exact: str = "on") -> Path:
    """Point every cache at the install's own folders. Returns TORCH_HOME."""
    models = models_folder(repo)
    torch_home = models / "torch"
    os.environ["HF_HOME"] = str(models)
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["LOW_VRAM"] = "1" if low_vram else "0"   # read once by app.py at import
    os.environ["ATTN_BACKEND"] = attn_backend
    if sage in ("on", "off"):                            # "auto": the fork's own default (modules/sage.py)
        os.environ["PIXAL3D_SAGE"] = sage
    if int8 in ("on", "off"):                            # "auto": the fork's own default (modules/int8.py)
        os.environ["PIXAL3D_INT8"] = int8
    os.environ["PIXAL3D_COARSE_EXACT"] = coarse_exact     # read by the fork's modules/exact.py
    # app.py sets this only after its `import cv2`, and OpenCV 4.12 reads it when cv2 loads:
    # init_models() would then fail on the .exr environment maps ("OpenEXR codec is disabled").
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    # torch's CPU allocator (mimalloc 2.2) hands freed memory back to Windows only after a delay, and
    # only when it is next called: the fp32 copies made while each model loads then stay committed.
    # Measured on one 1.3B model: 8.5 GB private without this, 5.1 GB with it (the model is 2.8 GB).
    # Must be set before torch is imported.
    os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
    # gradio copies every file the full UI sends or returns (uploads, 48 renders, GLBs) into its cache,
    # %TEMP%\gradio by default, and never deletes it: keep it in the repo's tmp (cleared at start).
    os.environ.setdefault("GRADIO_TEMP_DIR", str(repo / "tmp" / "gradio"))
    # Triton kernels compiled on first use; keep them with the venv, not in %USERPROFILE%.
    os.environ["TRITON_CACHE_DIR"] = str(Path(sys.prefix) / "triton-cache")
    # install.py already downloaded every weight: skip ~20 HEAD requests per start so
    # startup does not depend on huggingface.co (rate limits, proxies, offline PCs).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    use_ansi_safe_temp()
    if not ansi_safe(sys.prefix):
        print(f"[spellcast] WARNING: the Spellcast3D folder path ({sys.prefix}) has letters that Windows' "
              "'language for non-Unicode programs' cannot represent. Pixal3D's GPU kernels cannot be built "
              "there, so making a mesh will fail. Move the Spellcast3D folder to a simple path such as "
              "C:\\spellcast3d.", flush=True)
    return torch_home


def fp16acc_note() -> str:
    """The start-up line's word on the fork's opt-in PIXAL3D_FP16ACC (modules/fp16acc.py) when it is set: the ss and
    lr flows' linears in fp16 with fp16 accumulation, a precision change that makes other props; nothing when off."""
    if os.environ.get("PIXAL3D_FP16ACC", "").strip().lower() not in ("1", "on", "true", "yes"):
        return ""
    return (" PIXAL3D_FP16ACC=on: the ss and lr flows' linear layers run in fp16 with fp16 accumulation (a precision "
            "change, on GeForce RTX 30xx and newer only).")


def models_folder(repo: Path) -> Path:
    """Where Pixal3D's install.py put the weights: its MODELS folder (NTFS ignores the case), which the installer
    makes a junction to Spellcast3D's models/pixal3d (install/models.py). Every path goes through the junction.
    A Linux test pod's file system minds the case: there the link's own name, MODELS."""
    return repo / ("models" if os.name == "nt" else "MODELS")


# ---- paths with non-English letters -----------------------------------------------------------

def ansi_safe(path: str) -> bool:
    """True when `path` survives Windows' ANSI code page ("language for non-Unicode programs").

    Programs that take narrow (char) paths - triton's bundled tcc.exe and ptxas.exe, OpenCV's
    file reader - see every other letter replaced by '?' or a look-alike ('Łukasz' -> 'Lukasz' on
    English Windows) and then cannot find the file. The strict codec refuses such look-alikes.
    """
    if os.name != "nt":
        return True
    try:
        path.encode("mbcs", "strict")
    except UnicodeEncodeError:
        return False
    return True


def short_path(path: str) -> str | None:
    """The 8.3 short form of an existing path (ASCII when the drive keeps 8.3 names), or None."""
    if os.name != "nt":
        return None
    buf = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf))
    return buf.value if 0 < n < len(buf) else None


def use_ansi_safe_temp() -> None:
    """Give this process a TEMP that triton's compilers can open.

    triton builds its kernels (at the first mesh) in %TEMP% with tcc.exe and ptxas.exe, which read
    their command line in the ANSI code page: a user folder such as C:\\Users\\Łukasz on English
    Windows made every build fail minutes into the first mesh. Windows usually hands out TEMP as an
    8.3 short path already; a TEMP that works is left alone.
    """
    temp = os.environ.get("TEMP") or tempfile.gettempdir()
    if ansi_safe(temp):
        return
    own = Path(sys.prefix) / "tmp"
    for candidate in (short_path(temp), str(own)):
        if not candidate or not ansi_safe(candidate):
            continue
        if candidate == str(own):
            try:
                own.mkdir(exist_ok=True)
            except OSError:
                continue
            for leftover in own.iterdir():   # earlier runs' build folders (a killed run keeps them)
                if leftover.is_dir():
                    shutil.rmtree(leftover, ignore_errors=True)
                else:
                    try:
                        leftover.unlink()
                    except OSError:
                        pass   # in use; next start
        os.environ["TEMP"] = os.environ["TMP"] = candidate
        tempfile.tempdir = None   # tempfile caches the folder it found first
        return
    print(f"[spellcast] WARNING: the temporary folder {temp} has letters that Windows' 'language for "
          "non-Unicode programs' cannot represent, and no other temporary folder was found. Making a mesh "
          "may fail; if it does, move the Spellcast3D folder to a simple path such as C:\\spellcast3d.",
          flush=True)


def open_non_ascii_paths_in_opencv() -> None:
    """Let cv2.imread open files in folders with non-English letters (C:\\Users\\José\\...).

    OpenCV opens files through the narrow (ANSI) API, so any non-ASCII letter in the path makes
    imread return None - and init_models() then crashes on its .exr environment maps after minutes
    of loading. Relative paths are resolved by Windows against the (Unicode) working folder, which is
    the Pixal3D checkout (main() chdirs there), so a path inside it is handed over relative. EXR files
    must go this way: decoding them from memory goes through an OpenCV temp file in %TEMP%, which
    fails the same way. Must run after configure_env() (OPENCV_IO_ENABLE_OPENEXR is read at import).
    app.py calls cv2.imread through the module attribute, so replacing it is enough.
    """
    import cv2
    import numpy as np

    imread = cv2.imread

    def openable(path: Any) -> Any:
        if isinstance(path, str) and not path.isascii():
            try:
                rel = os.path.relpath(path)
            except ValueError:   # on another drive
                return path
            if rel.isascii():
                return rel
        return path

    @functools.wraps(imread)
    def read(filename: Any, *args: Any, **kwargs: Any) -> Any:
        image = imread(openable(filename), *args, **kwargs)
        if image is None and isinstance(filename, str) and not filename.isascii():
            flags = kwargs.get("flags", args[-1] if args and isinstance(args[-1], int) else cv2.IMREAD_COLOR)
            try:   # outside the working folder: decode from memory (fine for PNG / JPG, not for EXR)
                image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
            except (OSError, ValueError):
                return None
        return image

    cv2.imread = read


def load_naf_locally(torch_home: Path) -> None:
    """Load valeoai/NAF from the checkout install.py made instead of asking GitHub first.

    torch.hub.load("valeoai/NAF") contacts github.com before looking at its cache,
    three times per start, and fails when a proxy answers 403/429 even though
    everything is on disk. Pixal3D calls it through the torch.hub module attribute
    (image_conditioned_proj.py _load_naf), so replacing that attribute is enough.
    """
    import torch.hub

    local = torch_home / "hub" / "valeoai_NAF_main"
    if not (local / "hubconf.py").is_file():
        return  # not installed yet: let Pixal3D download it as usual
    github_load = torch.hub.load

    def load(repo_or_dir: str, model: str, *args: Any, **kwargs: Any) -> Any:
        if repo_or_dir == "valeoai/NAF" and kwargs.get("source", "github") == "github":
            kwargs["source"] = "local"
            return github_load(str(local), model, *args, **kwargs)
        return github_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = load


def allow_own_file_urls(host: str, port: int, folders: tuple[Path, ...]) -> None:
    """Let the full Pixal3D UI hand this server's own files back to it.

    index.html passes the preprocessed image to generate_3d as a URL of this server
    (http://127.0.0.1:P/gradio_api/file=...). gradio 6.14 downloads URL inputs with SSRF
    protection that rejects loopback addresses (after asking dns.google about them), so every
    generation started from the full UI failed with "An error occurred during synthesis"
    ("Hostname 127.0.0.1 failed validation"). Such a URL of THIS server's port that names a file
    in its own cache (`folders`) is answered with that file, with no HTTP request at all; every
    other URL still goes through gradio's check. (Adding 127.0.0.1 to gradio's host allow-list
    would have let any request to this server fetch from every other local service.)
    blocks.py calls the download through the module attribute, so replacing it is enough.
    """
    try:
        from gradio import processing_utils
    except Exception:
        return
    original = getattr(processing_utils, "async_ssrf_protected_download", None)
    if original is None:
        print("[spellcast] Note: generations from the full Pixal3D UI may fail with this gradio "
              "version (its URL download function was not found).", flush=True)
        return
    # the hub shows the page under the host the artist opened the hub with (127.0.0.1 or localhost)
    names = {host, "localhost"} if host in ("127.0.0.1", "localhost") else {host}
    roots = tuple(f.resolve() for f in folders)
    prefix = "/gradio_api/file="

    async def download(url: str, cache_dir: str) -> str:
        u = urlparse(url)
        if (u.hostname or "").lower() in names and u.port == port and u.path.startswith(prefix):
            local = Path(unquote(u.path[len(prefix):])).resolve()
            if local.is_file() and any(local.is_relative_to(root) for root in roots):
                return str(local)
        return await original(url, cache_dir)

    processing_utils.async_ssrf_protected_download = download


def clear_old_tmp(tmp_dir: Path) -> None:
    """Delete what earlier runs left in app.py's tmp folder: per generation the full Pixal3D UI
    writes 48 preview renders, a latent .npz and the GLB there and never deletes them, and a hub
    run that was cut short leaves its camera image. Nothing in it is used after a restart."""
    count, size = 0, 0
    gradio_cache = Path(os.environ.get("GRADIO_TEMP_DIR", tmp_dir / "gradio"))
    files = list(tmp_dir.glob("*"))
    if gradio_cache.parent == tmp_dir and gradio_cache.is_dir():   # only a cache this launcher placed there
        files += list(gradio_cache.rglob("*"))
    for f in files:
        try:
            if f.is_file():
                n = f.stat().st_size
                f.unlink()
                count, size = count + 1, size + n
        except OSError:
            pass   # in use by another program; next start
    if count:
        print(f"[spellcast] Removed {count} temporary files of earlier runs ({size / 1e6:.0f} MB).", flush=True)


def quiet_expandable_segments() -> None:
    """app.py sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, which Windows builds of torch do
    not support: they ignore it and put a UserWarning into the log the artist sees at the first
    generation. Turn the setting off again (no change in behaviour on Windows). Linux builds do support
    it: off there too, so that a Linux test pod's card memory behaves as the product's does."""
    if "expandable_segments:True" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        return
    try:
        import torch
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")
    except Exception:
        pass   # private torch API; the warning is harmless


def trim_cuda_cache(pipeline: Any) -> None:
    """Give back the CUDA memory of each conditioning step as soon as the step is done.

    get_proj_cond_shape() builds dense feature grids and keeps only a sparse slice of them (for
    the texture at quality 1024: DINOv3 features upsampled to 1024x1024 by NAF). PyTorch keeps
    the freed blocks reserved until the pipeline's next empty_cache(), which comes only after
    the following sampling stage. On a 12 GB card shared with other programs they spilled into
    shared system memory for the whole texture stage (measured: 10.3 GB dedicated + 8.0 GB
    shared), which is slower and costs the same amount of RAM / page file.
    """
    import torch

    def trimmed(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            finally:
                torch.cuda.empty_cache()
        return wrapper

    # The two decoders (upsampling to the full 1024 grid) are the largest single steps of all.
    for name in ("get_proj_cond_ss", "get_proj_cond_shape", "decode_shape_slat", "decode_tex_slat"):
        fn = getattr(pipeline, name, None)
        if callable(fn):
            setattr(pipeline, name, trimmed(fn))   # instance attribute: shadows the method


def label_decoding(px: ModuleType, pipeline: Any) -> None:
    """Report a progress stage while the latents are decoded into a mesh.

    decode_latent() has no progress bar, and it takes minutes on a 12 GB card (plus a one-time
    kernel autotune on the very first run): without this the artist sees "Painting the texture
    12/12" frozen for 3-4 minutes. It goes to the progress file of whichever session runs it, so
    Pixal3D's own page shows it too.
    """
    fn = getattr(pipeline, "decode_latent", None)
    if not callable(fn):
        return

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        px._update_progress(DECODING, 0, 1)
        return fn(*args, **kwargs)

    pipeline.decode_latent = wrapper


def guard_sparse_structure(pipeline: Any, notes: RunNotes) -> None:
    """Draw the sparse structure again when it came out as a solid block running to the grid's edge.

    If the first stage ("Blocking out the shape") filled the silhouette from the body to the grid face
    on the camera side, every later stage could only refine inside that block and the front of the prop
    would become a bundle of thin fins (pixal3d_guard.ss_edge_score and ss_deep_share). That was first
    blamed for a bear statue's fins, which turned out to come from misplaced hole-fill vertices
    (exact_hole_centres); no healthy sample ever tripped it, so it stays as a cheap safety net. A new
    draw costs one stage (about 20 s, 38 s for the last one) against a ruined 5-minute mesh. The CPU
    random generator has moved on, so the redraw differs and the result still depends only on the seed;
    a prop that passes the first time comes out exactly as before (all 101 healthy samples of the
    calibration did). The last attempt also changes the sampler settings
    (pixal3d_guard.RETRY_SAMPLER). When no draw passes, the one with the lowest edge score is kept (the
    earlier one on a tie). run() calls this positionally: (cond, resolution, num_samples,
    sampler_params); the multi-view pipeline inherits the same stage. `pipeline` is also the staged
    engine: its part "ss" calls its sample_sparse_structure the same way, once per prop, between the
    notes' reset and their reading (BatchHooks.prop_scope), so each prop of a batch gets its own count.
    """
    fn = getattr(pipeline, "sample_sparse_structure", None)
    if not callable(fn):
        return
    guard = pixal3d_guard

    @functools.wraps(fn)
    def wrapper(cond: Any, resolution: int, num_samples: int = 1, sampler_params: dict[str, Any] | None = None
                ) -> Any:
        attempts = guard.GUARD_ATTEMPTS
        base = sampler_params if sampler_params is not None else {}
        best, best_score = None, 0.0
        for attempt in range(1, attempts + 1):
            override = guard.RETRY_SAMPLER.get(attempt) if attempt > 1 else None
            params = {**base, **override} if override else base
            started = time.monotonic()
            coords = fn(cond, resolution, num_samples, params)
            try:
                score = guard.ss_edge_score(coords, resolution)
                deep = guard.ss_deep_share(coords, resolution)
            except Exception as exc:   # a check must never cost the mesh
                print(f"[spellcast] Note: could not check the blocked-out shape ({exc}).", flush=True)
                return coords
            if best is None or score < best_score:
                best, best_score = coords, score
            blocked = guard.ss_blocked(score, deep)
            settings = ", ".join(f"{k} {v}" for k, v in override.items()) if override else "default settings"
            seconds = time.monotonic() - started
            print(f"[spellcast] Blocked-out shape {attempt}/{attempts} ({settings}, {seconds:.0f} s): "
                  f"{score:.0%} of its columns reach the edge of the grid, {deep:.0%} solid from it "
                  f"(limits {guard.GUARD_THRESHOLD:.0%} and {guard.DEEP_THRESHOLD:.0%})"
                  + ("." if not blocked else "; drawing it again." if attempt < attempts
                     else f"; keeping the best draw ({best_score:.0%})."), flush=True)
            if not blocked:
                return coords
            if attempt < attempts:
                notes.ss_retries += 1
        notes.ss_blocked = True
        return best

    pipeline.sample_sparse_structure = wrapper   # instance attribute: shadows the method


def recentre_fans(vertices: Any, faces: Any, before: int) -> int:
    """Put every vertex from index `before` on (the centres cumesh's fill_holes added, one per hole, each the
    tip of a fan over the hole's rim) at the mean of its fan's rim corners, in place. Returns how many were
    out of place (more than 1e-4 away). See exact_hole_centres."""
    import torch

    added = int(vertices.shape[0]) - before
    if added <= 0:
        return 0
    # Only the fans: a few thousand of the mesh's millions of faces. As 64-bit indices all of them would take
    # 24 bytes a face on the card (225 MB for 9.4M faces) right when the decoders' memory is still in use.
    faces = faces[(faces >= before).any(dim=1)].long()
    total = torch.zeros((added, 3), dtype=torch.float64, device=vertices.device)
    count = torch.zeros(added, dtype=torch.float64, device=vertices.device)
    for corner in range(3):                     # each new vertex takes the rim corners of its fan
        fan = faces[:, corner] >= before
        centre = faces[fan, corner] - before
        for other in range(3):
            if other != corner:
                rim = faces[fan, other]
                old = rim < before
                total.index_add_(0, centre[old], vertices[rim[old]].double())
                count.index_add_(0, centre[old], torch.ones(int(old.sum()), dtype=torch.float64,
                                                            device=vertices.device))
    placed = count > 0
    mean = total[placed] / count[placed, None]
    moved = vertices[before:]                   # a view: writing it updates the mesh
    wrong = int(((moved[placed].double() - mean).abs() > 1e-4).any(dim=1).sum())
    moved[placed] = mean.to(vertices.dtype)
    return wrong


def exact_hole_centres() -> None:
    """Put the vertices that fill small holes of the decoded mesh where they belong.

    decode_latent() runs Mesh.fill_holes() (pixal3d/representations/mesh/base.py) on every mesh, and
    cumesh's fill_holes closes each small hole with a fan around one new vertex, meant to be the mean of
    the hole's rim (in a fresh process it is, to 6e-8). But it never clears the z sum of that mean: it adds
    to whatever the GPU memory last held. The first mesh after a start gets fresh, zeroed memory; later
    ones get leftovers of the previous mesh, and the centres land anywhere along z - measured: the sheep made
    right after the bear had 8,769 of them at z = 1e20. Every such fan becomes a sliver running to the grid
    face, which the remesh turns into fins stretched toward the camera, outside the blocked-out shape (most
    likely what made a bear, second and third in one Pixal3D process, look like a block extruded to the
    camera). The fault reproduces on demand with cumesh alone: fill, remesh, fill again - every centre of
    the second fill lands at z ~ 2e9. Recomputing every new vertex as the mean of its fan's rim gives the
    fresh-process result every time.
    """
    try:
        from pixal3d.representations.mesh.base import Mesh
    except Exception as exc:
        print(f"[spellcast] Note: could not check the filled holes of the meshes ({exc}).", flush=True)
        return

    original = Mesh.fill_holes
    if getattr(original, "_spellcast_exact", False):
        return

    @functools.wraps(original)
    def fill_holes(self: Any, *args: Any, **kwargs: Any) -> Any:
        before = int(self.vertices.shape[0])
        result = original(self, *args, **kwargs)
        try:
            added = int(self.vertices.shape[0]) - before
            wrong = recentre_fans(self.vertices, self.faces, before)
            if wrong:
                print(f"[spellcast] Put {wrong:,} of the {added:,} vertices that fill small holes back in "
                      "place (the GPU had left them out of place).", flush=True)
        except Exception as exc:   # a repair must never cost the mesh
            print(f"[spellcast] Note: could not check the filled holes of the mesh ({type(exc).__name__}: {exc}).",
                  flush=True)
        return result

    fill_holes._spellcast_exact = True   # type: ignore[attr-defined]
    Mesh.fill_holes = fill_holes


def _pieces(cumesh: Any, vertices: Any, faces: Any) -> tuple[int, Any]:
    """(count, the piece of each face): the connected pieces of a mesh, by cumesh."""
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    mesh.get_connected_components()                 # read_connected_components() only reads its result
    count, component = mesh.read_connected_components()
    return int(count), component.long()


def _sphere_directions(count: int, device: Any) -> Any:
    """`count` unit vectors spread evenly over the sphere (a Fibonacci spiral)."""
    import torch

    i = torch.arange(count, dtype=torch.float64, device=device) + 0.5
    z = 1 - 2 * i / count
    ring = torch.sqrt((1 - z * z).clamp(min=0))
    angle = i * math.pi * (3 - math.sqrt(5))
    return torch.stack([ring * torch.cos(angle), ring * torch.sin(angle), z], dim=1).float()


def seen_from_outside(vertices: Any, faces: Any) -> Any:
    """Which faces of a mesh can be seen from outside it: the nearest faces in pixal3d_guard.SEEN_DIRECTIONS
    orthographic pictures of it taken all around, SEEN_GRID x SEEN_GRID pixels each (nvdiffrast's
    rasterizer: a z-buffer, exact whichever way a face faces - where the prop is thin, the remesh joins its
    two sides into one sheet that can face either way). cuBVH's ray_trace is no use here: on meshes of
    millions of faces it returns a far hit instead of the nearest one for many rays (measured: 16% of the
    rays at 3.9M faces). A bool mask per face."""
    import nvdiffrast.torch as dr
    import torch

    device = vertices.device
    context = dr.RasterizeCudaContext(device=device)
    lo, hi = vertices.min(dim=0).values, vertices.max(dim=0).values
    centre = (lo + hi) / 2
    radius = float((hi - lo).norm()) / 2 * 1.02
    shifted = (vertices - centre) / radius
    triangles = faces.int().contiguous()
    seen = torch.zeros(faces.shape[0] + 1, dtype=torch.bool, device=device)     # [0]: no face
    directions = _sphere_directions(pixal3d_guard.SEEN_DIRECTIONS, device)
    batch = 8
    for start in range(0, directions.shape[0], batch):
        axes = []
        for direction in directions[start:start + batch]:
            helper = torch.tensor([1.0, 0.0, 0.0] if abs(float(direction[0])) < 0.9 else [0.0, 1.0, 0.0], device=device)
            u = torch.linalg.cross(direction, helper)
            u = u / u.norm()
            axes.append(torch.stack([u, torch.linalg.cross(direction, u), direction * 0.99]))   # x, y, depth
        clip = torch.einsum("vk,bjk->bvj", shifted, torch.stack(axes))                        # [B, V, 3]
        clip = torch.cat([clip, torch.ones_like(clip[..., :1])], dim=-1).contiguous()
        grid = pixal3d_guard.SEEN_GRID
        rast, _ = dr.rasterize(context, clip, triangles, resolution=[grid, grid])
        seen[rast[..., 3].long().reshape(-1)] = True    # triangle id + 1, 0 where no face is
        del clip, rast
    return seen[1:]


def grow(mask: Any, faces: Any, vertex_count: int, rings: int) -> Any:
    """`mask` (per face) and the faces within `rings` rings of it (sharing a corner, ring after ring)."""
    import torch

    corners = faces.long()
    for _ in range(rings):
        touched = torch.zeros(vertex_count, dtype=torch.bool, device=mask.device)
        touched[corners[mask].reshape(-1)] = True
        mask = mask | touched[corners].any(dim=1)
    return mask


def escape_share(bvh: Any, points: Any, rays: int, generator: Any = None) -> Any:
    """Per point, the share of `rays` rays in random directions from it that leave the decoded mesh (its
    cuBVH `bvh`) without hitting it: 0 inside a closed solid; more in a hollow with an opening."""
    import torch

    directions = torch.randn(rays, 3, generator=generator).to(points.device)
    directions = directions / directions.norm(dim=1, keepdim=True)
    origins = points[:, None, :].expand(-1, rays, -1).reshape(-1, 3).contiguous()
    towards = directions[None, :, :].expand(points.shape[0], -1, -1).reshape(-1, 3).contiguous()
    if pixal3d_glb.rays_by_hit(bvh):        # hit or miss alone: the same answers without the search for the nearest
        escaped = ~bvh.ray_hits(origins, towards)
    else:
        _, hit, _ = bvh.ray_trace(origins, towards)
        escaped = hit < 0
    return escaped.reshape(points.shape[0], rays).float().mean(dim=1)


def find_inner_copy(vertices: Any, faces: Any, source: tuple[Any, Any], bvh: Any = None) -> tuple[Any, dict[str, Any]]:
    """The faces of a remeshed surface (vertices, faces: remesh_narrow_band_dc's result) that are the
    hidden inward-facing copy the remesh adds, as a bool mask per face, and what was found (for the log).
    source: the decoded mesh it was made from (vertices, faces); bvh: its cuBVH, when there is one.

    (1) Whole pieces with negative signed volume (the sheep's copy is one).
    (2) Where the copy is joined to the visible shell (a furry bear: wherever the prop is thinner than a
    few voxels), the faces whose centre is inside the decoded mesh by cuBVH's ray test - but never a face
    that can be seen from outside, nor the faces a few rings around one (seen_from_outside, grow): removing
    them would open a hole the artist sees (a fold in the fur is "inside" by the ray test too). Of those,
    the large connected pieces (pixal3d_guard.inner_pieces), unless a piece sees out through an opening:
    the ray test only asks whether a few rays escape, so the inside wall of a jar or a room behind a small
    window looks "inside" too, while the copy, enclosed in the solid, never sees out (escape_share,
    pixal3d_guard.is_cavity).
    pixal3d_guard.inner_plan says which of the two steps are kept (never too much of the surface).
    """
    import cumesh
    import torch

    device = vertices.device
    total = int(faces.shape[0])
    info: dict[str, Any] = {"faces": total}
    none = torch.zeros(total, dtype=torch.bool, device=device)
    corners = faces.long()
    # (1) whole pieces facing inward
    count, component = _pieces(cumesh, vertices, faces)
    tri = vertices[corners]
    volume = (tri[:, 0] * torch.cross(tri[:, 1], tri[:, 2], dim=1)).sum(1).double()   # 6x signed volume
    centres = tri.mean(dim=1)
    del tri
    volumes = torch.zeros(count, dtype=torch.float64, device=device).index_add_(0, component, volume)
    negative = volumes < 0
    whole = negative[component] if bool((~negative).any()) else none.clone()
    info["whole"] = int(whole.sum())
    # (2) the copy where it is joined to the visible shell
    if bvh is None:
        bvh = cumesh.cuBVH(source[0], source[1])
    rest = (~whole).nonzero().squeeze(1)
    candidates = none.clone()
    joined = none.clone()
    near = None
    look = True
    # info "inside": the faces of all of `rest` inside by the ray test; "inside_unseen": only those away from every
    # face seen from outside (what a cumesh that stabs subsets counts once it looks; "stabbed": the points it asked).
    # The two are not comparable.
    if pixal3d_glb.stab_by_index(bvh):
        # Each face is stabbed with the rays the query of all of `rest` would give it (pixal3d_glb.stab_inside), so
        # the same copy is found while fewer faces are asked: first slices of `rest` until enough are inside for a
        # piece of the copy (else step (2) ends there, as _joined_copy would end it), then, once the faces seen from
        # outside are known, the faces left that are away from them - the only ones step (2) can take.
        probed = pixal3d_glb.stab_until(bvh, centres, rest, pixal3d_guard.INNER_PIECE_MIN_FACES)
        count = int(probed.shape[0])
        info["stabbed"] = count
        if count == rest.shape[0] and int(probed.sum()) < pixal3d_guard.INNER_PIECE_MIN_FACES:
            candidates[rest[probed]] = True
            info["inside"] = int(candidates.sum())
            look = False
        else:
            try:
                seen = seen_from_outside(vertices, faces)
                info["seen"] = int(seen.sum())
                near = grow(seen, faces, int(vertices.shape[0]), pixal3d_guard.SEEN_RINGS)
                del seen
            except Exception as exc:   # the separate pieces still go
                info["joined_error"] = f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
                look = False
            if near is not None:
                away = ~near[rest]
                candidates[rest[:count][probed & away[:count]]] = True
                asked = away[count:].nonzero().squeeze(1) + count
                candidates[rest[asked[pixal3d_glb.stab_inside(bvh, centres, rest, asked)]]] = True
                info["stabbed"] += int(asked.shape[0])
                info["inside_unseen"] = int(candidates.sum())
    else:
        inside = torch.empty(rest.shape[0], dtype=torch.bool, device=device)
        chunk = pixal3d_glb.STAB_QUERY      # its seeds: points per ray-test query, as the remesh queries its BVH
        for i in range(0, rest.shape[0], chunk):
            inside[i:i + chunk] = bvh.signed_distance(centres[rest[i:i + chunk]], mode="raystab")[0] < 0   # all hit
        candidates[rest[inside]] = True
        info["inside"] = int(candidates.sum())
    if look:
        try:
            _joined_copy(cumesh, vertices, faces, centres, candidates, bvh, joined, info, near)
        except Exception as exc:   # the separate pieces still go
            joined = none.clone()
            info["joined_error"] = f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
    info["joined"] = int(joined.sum())
    plan = pixal3d_guard.inner_plan(info["whole"], info["joined"], total)
    info["plan"] = plan
    if plan == "both":
        return whole | joined, info
    if plan == "whole":
        return whole, info
    return none, info


def _joined_copy(cumesh: Any, vertices: Any, faces: Any, centres: Any, candidates: Any, bvh: Any, joined: Any,
                 info: dict[str, Any], near: Any = None) -> None:
    """find_inner_copy's step (2): marks in `joined` the faces of `candidates` (inside by the ray test) that
    are the copy joined to the visible shell; notes what it found in `info`. `near`: the faces seen from
    outside and those within SEEN_RINGS of one, when find_inner_copy found them already."""
    import torch

    device = vertices.device
    if int(candidates.sum()) < pixal3d_guard.INNER_PIECE_MIN_FACES:     # no piece of them is large enough
        return
    if near is None:
        seen = seen_from_outside(vertices, faces)
        info["seen"] = int(seen.sum())
        near = grow(seen, faces, int(vertices.shape[0]), pixal3d_guard.SEEN_RINGS)
    candidates = candidates & ~near
    chosen = candidates.nonzero().squeeze(1)
    if chosen.shape[0] < pixal3d_guard.INNER_PIECE_MIN_FACES:
        return
    count, piece = _pieces(cumesh, vertices, faces[chosen].contiguous())
    sizes = torch.bincount(piece, minlength=count).tolist()
    large = pixal3d_guard.inner_pieces(sizes)
    generator = torch.Generator().manual_seed(0)
    hollows = []
    for n in (n for n, keep in enumerate(large) if keep):
        members = chosen[piece == n]
        sample = members[torch.randperm(members.shape[0], generator=generator)[:pixal3d_guard.CAVITY_SAMPLES]
                         .to(device)]
        share = float((escape_share(bvh, centres[sample], pixal3d_guard.CAVITY_RAYS, generator)
                       >= pixal3d_guard.CAVITY_POINT_ESCAPE).float().mean())
        info.setdefault("seeing_out", []).append(round(share, 3))
        if pixal3d_guard.is_cavity(share):
            large[n] = False
            hollows.append((sizes[n], round(share, 3)))
    info["pieces"] = sum(large)
    info["hollows"] = hollows
    mask = torch.tensor(large, dtype=torch.bool, device=device)
    joined[chosen[mask[piece]]] = True


def cut_away(cumesh: Any, vertices: Any, faces: Any, drop: Any) -> tuple[Any, Any]:
    """The mesh without the faces `drop` marks (a bool mask per face) -> (vertices, faces)."""
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    mesh.remove_faces((~drop).contiguous())          # NB: KEEPS the faces where the mask is True (verified)
    mesh.remove_unreferenced_vertices()
    return mesh.read()


def drop_inner_shells() -> None:
    """Remove the hidden inward-facing copy of the surface that to_glb's remeshing adds to every mesh.

    to_glb(remesh=True) rebuilds the surface by dual contouring where the UNSIGNED distance to the
    decoded mesh equals a small eps (about 0.002 of the prop). Around every closed part that gives one
    shell eps outside and one eps inside, whose faces point inward. The inner one is invisible but took
    half of the sheep's faces and texture, doubled retopo, and led the bake's winding fix to turn the
    visible shell inside out. Right after the remesh, before simplify spends the face budget, the faces
    find_inner_copy finds go. Where the copy was joined to the visible shell its rim is left open inside
    the prop (the simplify after can still open a small pit seen only through a narrow gap: the bear kept
    one such hole in its shoulder; pixal3d_guard has the numbers). o_voxel.postprocess looks the function
    up on the module at call time, so replacing the module attribute covers Pixal3D's own page too.
    """
    try:
        import cumesh
        from cumesh import remeshing
    except Exception as exc:
        print(f"[spellcast] Note: meshes keep a hidden inner copy of their surface ({exc}).", flush=True)
        return

    original = remeshing.remesh_narrow_band_dc
    if getattr(original, "_spellcast_outer_only", False):
        return

    def outer_only(vertices: Any, faces: Any, source: tuple[Any, Any], bvh: Any) -> tuple[Any, Any]:
        started = time.monotonic()
        drop, info = find_inner_copy(vertices, faces, source, bvh)
        total = info["faces"]
        removed = int(drop.sum())
        hollow = sum(n for n, _ in info.get("hollows", []))
        spared = f"; kept {hollow:,} faces of a hollow that opens to the outside" if hollow else ""
        if info.get("joined_error"):
            print(f"[spellcast] Note: could not look for the inner copy joined to the visible surface "
                  f"({info['joined_error']}); only its separate pieces go.", flush=True)
        if info["plan"] == "none" and info["whole"]:
            print(f"[spellcast] Kept the separate inward-facing pieces: they are {info['whole'] / total:.0%} of the "
                  "surface, too much to be a hidden copy.", flush=True)
        elif info["plan"] != "both" and info["joined"]:
            print(f"[spellcast] Kept the inward-facing faces joined to the visible surface: they are "
                  f"{info['joined'] / (total - info['whole']):.0%} of what the separate pieces leave, too much "
                  "to be a hidden copy.", flush=True)
        if not removed:
            print(f"[spellcast] No hidden inner copy of the surface to remove{spared} "
                  f"({time.monotonic() - started:.0f} s).", flush=True)
            return vertices, faces
        kept_vertices, kept_faces = cut_away(cumesh, vertices, faces, drop)
        print(f"[spellcast] Removed the hidden inner copy of the surface: {removed:,} of {total:,} faces "
              f"({removed / total:.0%}; {info['whole']:,} as separate pieces; none that can be seen from outside"
              f"{spared}; {time.monotonic() - started:.0f} s).", flush=True)
        return kept_vertices.contiguous(), kept_faces.int().contiguous()

    @functools.wraps(original)
    def wrapper(vertices: Any, faces: Any, *args: Any, **kwargs: Any) -> Any:
        new_vertices, new_faces = original(vertices, faces, *args, **kwargs)
        try:
            return outer_only(new_vertices, new_faces, (vertices, faces), kwargs.get("bvh"))
        except Exception as exc:   # a clean-up must never cost the mesh
            print(f"[spellcast] Note: could not look for the hidden inner copy of the surface "
                  f"({type(exc).__name__}: {exc}); the mesh keeps it.", flush=True)
            return new_vertices, new_faces

    wrapper._spellcast_outer_only = True   # type: ignore[attr-defined]
    remeshing.remesh_narrow_band_dc = wrapper


def prepare_pipeline(px: ModuleType, pipeline: Any, notes: RunNotes) -> None:
    """The launcher's wrappers on one pipeline object: px.pipeline once init_models() made it, and the
    multi-view pipeline when pixal3d_mv.ModelModes builds it."""
    trim_cuda_cache(pipeline)
    label_decoding(px, pipeline)
    guard_sparse_structure(pipeline, notes)


class SharedModels:
    """The models share_dinov3 hands to every caller. clear() lets go of them when the reference path is
    unloaded (Reference.unload): kept here, DINOv3's 1.2 GB would stay in RAM for the life of the process."""

    def __init__(self) -> None:
        self.loaded: dict[str, Any] = {}

    def clear(self) -> None:
        self.loaded.clear()


def share_dinov3() -> SharedModels:
    """Give app.py's four image-conditioning models (IMAGE_COND_CONFIGS) one shared DINOv3.

    Each DinoV3ProjFeatureExtractor loads its own copy of the same frozen DINOv3 ViT-L
    (1.2 GB in fp32), so three copies are 3.6 GB of RAM in low-VRAM mode (VRAM otherwise)
    for nothing. The pipeline uses one extractor at a time and moves it to the GPU and
    back (low-VRAM mode) or keeps all of them there, so a shared module behaves the same.
    Only the reference path loads its conditioning models this way; the staged engine builds
    its own DINOv3 (pixal3d/staged/builders.py) and drops it after the part that needs it.
    """
    shared = SharedModels()
    try:
        from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj as icp
    except Exception as exc:  # layout changed in a Pixal3D update: just load the copies
        print(f"[spellcast] Note: DINOv3 is loaded once per image model ({exc}).", flush=True)
        return shared
    original = getattr(icp, "DINOv3ViTModel", None)
    if original is None or not hasattr(original, "from_pretrained"):
        return shared

    class SharedDINOv3:
        @staticmethod
        def from_pretrained(path: Any, *args: Any, **kwargs: Any) -> Any:
            key = repr((str(path), args, sorted(kwargs.items())))
            if key not in shared.loaded:
                shared.loaded[key] = original.from_pretrained(path, *args, **kwargs)
            return shared.loaded[key]

    icp.DINOv3ViTModel = SharedDINOv3
    return shared


# ---- hub routes -------------------------------------------------------------

def ui_defaults(px: ModuleType) -> dict[str, Any]:
    """generate_3d's keyword defaults (the values Pixal3D's own page sends), so hub runs match it."""
    return {name: p.default for name, p in inspect.signature(px.generate_3d).parameters.items()
            if p.default is not inspect.Parameter.empty}


def sampler_params(defaults: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {"steps": defaults[f"{prefix}_sampling_steps"],
            "guidance_strength": defaults[f"{prefix}_guidance_strength"],
            "guidance_rescale": defaults[f"{prefix}_guidance_rescale"],
            "rescale_t": defaults[f"{prefix}_rescale_t"]}


class GpuWorker:
    """The one thread that runs everything that loads or moves Pixal3D's models: the staged engine
    (its making, every batch, each of its hooks), init_models(), the reference generations and the
    full Pixal3D UI's calls, one at a time in arrival order.

    torch's CPU allocator (mimalloc) gives memory back to Windows only when the thread that
    allocated it frees it. Low-VRAM mode moves every model to the GPU and back in each
    generation: with the models loaded on the main thread and generations on the web server's
    worker threads, each move left the model's old CPU copy committed for good (measured:
    +11 to +14 GB of RAM / page file during the first generation). One thread also keeps two
    generations from fighting over VRAM and moving shared models under each other's feet.
    """

    def __init__(self) -> None:
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pixal3d-gpu")
        self._lock = threading.Lock()
        self._jobs = 0          # queued or running

    def run(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return self.submit(fn, *args, **kwargs).result()

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> concurrent.futures.Future[Any]:
        """Queue fn behind what runs now and return at once (a batch: its request is answered before it starts)."""
        with self._lock:
            self._jobs += 1
        try:
            future = self._pool.submit(self._call, fn, args, kwargs)
        except BaseException:
            self._done()
            raise
        future.add_done_callback(lambda _: self._done())
        return future

    def idle(self) -> bool:
        """Nothing queued or running."""
        with self._lock:
            return self._jobs == 0

    def _done(self) -> None:
        with self._lock:
            self._jobs -= 1

    @staticmethod
    def _call(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        try:
            return fn(*args, **kwargs)
        finally:
            # Give cached VRAM back after every job: other programs share the card, and an idle
            # Pixal3D kept 1.8 GB of freed blocks after a mesh otherwise.
            torch = sys.modules.get("torch")
            try:
                if torch is not None and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
            except Exception:
                pass   # cleanup only; never turn a finished job into a failure


def serialise_ui_calls(px: ModuleType, gpu: GpuWorker, before: Callable[[], None] | None = None) -> None:
    """Run the full Pixal3D UI's calls (its own page) on the GPU worker too, each after `before`
    (which puts the single-view models back after a multi-view mesh: the page runs app.pipeline).

    gradio.Server keeps the @app.api() functions in _deferred_apis until launch()
    (gradio 6.14 server.py:171,211,284); functools.wraps keeps the signature gradio reads.
    """
    def locked(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            def call() -> Any:
                if before is not None:
                    before()
                return fn(*args, **kwargs)
            return gpu.run(call)
        return wrapper

    deferred = getattr(px.app, "_deferred_apis", None)
    if deferred is None:
        print("[spellcast] WARNING: this gradio version hides Pixal3D's API list; generations started "
              "from the full Pixal3D UI are not queued behind Spellcast3D's.", flush=True)
        return
    deferred[:] = [(locked(fn), kwargs) for fn, kwargs in deferred]


def describe(stage: str, step: int, total: int) -> tuple[str, float] | None:
    for prefixes, label, start, end in PHASES:
        if stage.startswith(prefixes):
            done = min(step / total, 1.0) if total > 0 else 0.0
            return label, start + (end - start) * done
    return None


def error_response(status: int, detail: str, kind: str) -> JSONResponse:
    # kind: "input" | "mv_missing" | "oom" | "ram" | "disk" | "missing_file" | "stopped" | "busy" | "error"
    # (adapters/pixal3d.py reads it)
    return JSONResponse({"detail": detail, "kind": kind}, status_code=status)


def failure_words(kind: str, detail: str) -> str:
    """The artist's words for a failure of that kind (pixal3d_batch.classify). `detail`: the exception's
    text; for the kinds "input", "disk" and "stopped" it is already written for the artist. The engine's own
    text for a missing multi-view set only says it is missing, so the hub's words take its place: they say
    what to do about it."""
    if kind in ("input", "disk", "stopped") and detail:
        return detail
    if kind == "mv_missing":
        return pixal3d_mv.MV_MISSING
    if kind == "oom":
        return OUT_OF_GPU_MEMORY
    if kind == "ram":
        return OUT_OF_RAM
    short = detail if len(detail) <= 300 else detail[:300] + "..."
    if kind == "missing_file":
        return (f"A file Pixal3D needs is missing: {short}. If it is one of Pixal3D's model files, close "
                "Spellcast3D and double-click run-spellcast3d.bat again (it downloads what is missing).")
    if kind == "restart":
        return f"Pixal3D lost the graphics card and has to start again ({short})."
    low = detail.lower()
    if ("failed to compile" in low or "ptxas" in low) and not ansi_safe(sys.prefix):
        return ("Pixal3D could not build its GPU code because the Spellcast3D folder path "
                f"({sys.prefix}) has letters Windows' 'language for non-Unicode programs' "
                "cannot represent. Move the Spellcast3D folder to a simple path such as "
                "C:\\spellcast3d and start it again.")
    return (f"Pixal3D failed while making the mesh ({short}). Try again; if it keeps failing, restart Pixal3D "
            "and check its log.")


def failure_text(exc: BaseException, kind: str) -> str:
    """An exception's text as failure_words takes it for that kind: the missing file's name, and for a
    failure of no known kind the exception's type before its text."""
    text = str(exc)
    if kind == "missing_file":
        return str(getattr(exc, "filename", None) or text)
    if kind == "error":
        return f"{type(exc).__name__}: {text}"
    return text


def failure_response(exc: BaseException) -> JSONResponse:
    kind = pixal3d_batch.classify(exc)
    status = {"input": 400, "mv_missing": 409}.get(kind, 500)
    return error_response(status, failure_words(kind, failure_text(exc, kind)), kind)


def outside_projects(req: GenerateRequest, projects_root: Path | None, work_dir: str | None = None) -> bool:
    """Whether a path of the request - its pictures, its output file, its views folder, the work folder of its
    batch - lies outside the hub's projects folder (never, when no folder was given at the start)."""
    views = req.views or []
    paths = [req.image_path, req.out_path, *(v.image_path for v in views), *filter(None, [req.views_dir, work_dir])]
    return projects_root is not None and not all(Path(p).resolve().is_relative_to(projects_root) for p in paths)


def work_dir_problem(work_dir: str, projects_root: Path | None = None) -> str | None:
    """What is wrong with the work folder a batch names, or None. The engine empties that folder, so it must
    be one the hub made for a job: <projects>/<project>/temp/mesh-<job id> (without a projects folder given at
    the start: any .../temp/mesh-<name>)."""
    folder = Path(work_dir).resolve()
    shaped = folder.parent.name.lower() == "temp" and folder.name.lower().startswith("mesh-")
    if projects_root is not None:
        if not folder.is_relative_to(projects_root):
            return OUTSIDE_PROJECTS
        shaped = shaped and len(folder.relative_to(projects_root).parts) == 3
    if not shaped:
        return f"This is not the work folder of a job: {work_dir}."
    return None


def validate(req: GenerateRequest, projects_root: Path | None = None, work_dir: str | None = None) -> str | None:
    """What is wrong with a request (one of a batch: `work_dir` is the batch's work folder), or None."""
    views = req.views or []
    if outside_projects(req, projects_root, work_dir):
        return OUTSIDE_PROJECTS
    if work_dir is not None:
        problem = work_dir_problem(work_dir, projects_root)
        if problem is not None:
            return problem
    if not views and not Path(req.image_path).is_file():
        return f"The input image was not found: {req.image_path}. Choose the image again and retry."
    if not req.out_path.lower().endswith(".glb"):
        return f"The output file must end in .glb (got {req.out_path})."
    if req.resolution not in RESOLUTIONS:
        return f"Quality must be 1024 or 1536 (got {req.resolution})."
    if req.geometry not in GEOMETRIES:
        return f"The geometry must be Pixal3D or Hi3D (got {req.geometry})."
    if req.decimation_target < 1000:
        return f"Detail must be at least 1000 faces (got {req.decimation_target})."
    if not 256 <= req.texture_size <= 8192:
        return f"Texture size must be between 256 and 8192 pixels (got {req.texture_size})."
    return pixal3d_mv.validate_views(views)


def release_gpu(px: ModuleType, extra: tuple[Any, ...] | list[Any] = ()) -> None:
    """After a failed run, put the models back where init_models() left them and free cached VRAM.

    Low-VRAM mode moves each stage's model to the GPU and back; an exception in
    between leaves it there, and every later run would then run out of memory.
    `extra`: the multi-view conditioning models (the other models are shared).
    """
    import torch

    try:
        if px.LOW_VRAM:
            p = px.pipeline
            for m in (*p.models.values(), p.image_cond_model_ss, p.image_cond_model_shape_512,
                      p.image_cond_model_shape_1024, p.image_cond_model_tex_1024,
                      p.rembg_model, px.moge_model, *extra):
                if m is not None:
                    m.cpu()
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        traceback.print_exc()
        print("[spellcast] Could not reset the GPU after the error; restart Pixal3D if the next "
              "generation fails too.", flush=True)


def generate(px: ModuleType, modes: pixal3d_mv.ModelModes, notes: RunNotes, req: GenerateRequest,
             seed: int, defaults: dict[str, Any]) -> dict[str, Any]:
    """The reference path: picture(s) -> GLB with the objects init_models() loaded: app.py's /preprocess +
    generate_3d + extract_glb_api, minus the 8x6 preview renders; with 2+ views the multi-view pipeline instead.
    Call on the GPU worker. Returns what the hub keeps with the mesh (mode, views, warning, ss_retries)."""
    import torch

    views = pixal3d_mv.ordered(req.views or [])
    multi = len(views) >= 2
    px._reset_progress(SESSION)   # routes this thread's tqdm bars into SESSION's progress file
    notes.reset()
    try:
        # The picture or its slices are opened (and, for several, cut out and framed) before the models switch:
        # a slice to fix is said at once, not after a 30-110 s switch. Background removal is the same in both modes.
        if multi:
            framed = prepare_views(px, px.pipeline, req, views)
        else:
            crop = open_picture(req, views[0] if views else None)
        modes.ensure("mv" if multi else "sv")
        pipeline = modes.pipeline()
        torch.manual_seed(seed)
        if multi:
            meshes, res = run_multi_view(px, pipeline, req, views, framed, seed, defaults)
            del framed
        else:
            meshes, res = run_single_view(px, req, views[0] if views else None, seed, defaults, crop)
        # run() already returns decode_latent(latents), exactly what extract_glb_api rebuilds
        # from its saved state; res can be lower than asked when the shape has too many tokens.
        mesh = meshes[0]
        del meshes
        steps = glb_steps(px, mesh, pipeline.pbr_attr_layout, res, req, multi, notes.ss_retries, notes.ss_blocked,
                          inline=True)
        del mesh                   # the steps hold CPU copies: nothing of the decoded mesh stays on the card
        torch.cuda.empty_cache()   # the decoders' freed blocks; the clean-up needs the room
        result = pixal3d_glb.run_all(steps)
    finally:
        px._thread_local.active_session = ""   # this pool thread may serve other requests later
    return result


def has_own_transparency(picture: Any) -> bool:
    """Whether the picture's background is transparent, so that it keeps its own transparency and no background is
    removed (pixal3d_mv.transparent_background: a sizeable share of the picture and of its edge under the prop's
    alpha; preprocess_image's own test took any alpha below 255, a speck included)."""
    return pixal3d_mv.transparent_background(picture)


def preprocess(px: ModuleType, picture: Any, cutter: Any = None, keep: list[Any] | None = None) -> Any:
    """The single-view cut-out on `picture` (pixal3d_mv.cut_out_single: cut out, cropped to the prop, on black;
    what Pixal3D's preprocess_image makes, except that a picture larger than 1024 px is cropped before it is
    reduced), saying first which way it cuts the prop out: the picture's own transparency (a transparent
    Qwen-Image picture, a cut-out PNG) or background removal. `picture` comes from open_picture, at least
    pixal3d_mv.INPUT_SIZE px on its largest side. `cutter`: what removes the background when the reference pipeline
    (px.pipeline) is not loaded - matte_shim around the staged engine's model. Both paths come here, so a batch and
    the reference path cut a picture out alike.

    A picture in which nothing is left opaque is one the artist has to fix (ViewProblem), as for a slice
    (pixal3d_mv.cut_out)."""
    own = has_own_transparency(picture)
    px._update_progress(OWN_TRANSPARENCY if own else REMOVING_BACKGROUND, 0, 1)
    print(f"[spellcast] Picture: {picture.width}x{picture.height} px, "
          f"{'own transparency (no background removal)' if own else 'background removed'}", flush=True)
    try:
        extra = {"keep": keep} if keep is not None else {}
        return pixal3d_mv.cut_out_single(px.pipeline if cutter is None else cutter, picture, pixal3d_mv.CROP_FIRST,
                                         **extra)
    except ValueError:      # a box of nothing: Pillow or numpy have no image to make of it
        raise pixal3d_mv.ViewProblem(pixal3d_mv.NO_PROP) from None


def matte_shim(matte: Callable[[Any], Any]) -> Any:
    """What pixal3d_mv.matte needs of a pipeline to remove a background, around `matte` (RGB picture -> the
    picture with its alpha set; the staged engine keeps its model on the card): with low_vram off no model is
    moved."""
    return SimpleNamespace(low_vram=False, device="cuda", rembg_model=matte)


def open_picture(req: GenerateRequest, view: ViewSpec | None) -> Any:
    """What a single-view run is made from: the request's picture, or its one slice `view`. Both come from
    pixal3d_mv.open_crop, as the views of a multi-view run do: every picture and slice reaches background
    removal at pixal3d_mv.INPUT_SIZE px on its largest side or more."""
    if view is None:
        return pixal3d_mv.open_crop(req.image_path, None)
    return open_view(view)


def open_view(view: ViewSpec) -> Any:
    """One slice of a picture, opened as background removal takes it (pixal3d_mv.open_crop)."""
    return pixal3d_mv.open_crop(view.image_path, view.rect, view.azimuth)


def picture_openers(req: GenerateRequest, multi: bool) -> list[Callable[[], Any]]:
    """What BatchHooks.prepare opens for a prop, picture by picture in the order it cuts them out: the views of a
    prop made from several, the front one first (prepare_views), or its one picture or slice (open_picture)."""
    views = pixal3d_mv.ordered(req.views or [])
    if multi:
        return [functools.partial(open_view, view) for view in views]
    return [functools.partial(open_picture, req, views[0] if views else None)]


def run_single_view(px: ModuleType, req: GenerateRequest, view: ViewSpec | None, seed: int,
                    defaults: dict[str, Any], crop: Any = None) -> tuple[list[Any], int]:
    """The single-view cascade on the picture, or on the one slice `view` (`crop`: either, already opened
    by open_picture): (meshes, resolution)."""
    camera_png = Path(px.TMP_DIR) / f"spellcast_{uuid.uuid4().hex}.png"
    try:
        image = preprocess(px, crop if crop is not None else open_picture(req, view))
        image.save(camera_png)   # MoGe reads the preprocessed image from a file
        pixal3d_mv.save_views(req.views_dir, [(pixal3d_mv.VIEW_NAMES.get(view.azimuth if view else 0, ""), image)])

        px._update_progress(ESTIMATING_CAMERA, 0, 1)
        camera = px.get_camera_params_wild_moge(
            str(camera_png), device="cuda", mesh_scale=px.WILD_MESH_SCALE,
            extend_pixel=px.WILD_EXTEND_PIXEL, image_resolution=px.WILD_IMAGE_RESOLUTION)

        meshes, (_, _, res) = px.pipeline.run(
            image, camera_params=camera, seed=seed,
            sparse_structure_sampler_params=sampler_params(defaults, "ss"),
            shape_slat_sampler_params=sampler_params(defaults, "shape_slat"),
            tex_slat_sampler_params=sampler_params(defaults, "tex_slat"),
            preprocess_image=False, return_latent=True,
            pipeline_type=f"{req.resolution}_cascade", max_num_tokens=px.CASCADE_MAX_NUM_TOKENS)
        return meshes, res
    finally:
        camera_png.unlink(missing_ok=True)


def prepare_views(px: ModuleType, pipeline: Any, req: GenerateRequest, views: list[ViewSpec],
                  opened: list[concurrent.futures.Future[Any]] | None = None) -> list[Any]:
    """2+ slices, the front one first -> the framed views the multi-view models take: each opened (a small one
    enlarged to pixal3d_mv.INPUT_SIZE px on its largest side, the factor kept as its unit), cut out (background removal:
    `pipeline`'s, shared by both modes) and framed like the training renders, slices of one picture at one
    scale whatever their units (pixal3d_mv.frame_views). Saved to req.views_dir. Raises ViewProblem for a
    slice the artist has to fix. `opened`: the views opened ahead in a batch (OpenAhead), one future per view in the
    order of `views`; each is waited for when its view's turn comes, and raises what opening it raised."""
    images, units = [], []
    for n, view in enumerate(views):
        crop = opened[n].result() if opened else open_view(view)
        # as cut_out decides: a slice with its own transparency (a transparent Qwen-Image picture) keeps it
        own = pixal3d_mv.real_alpha(crop) is not None
        px._update_progress(OWN_TRANSPARENCY if own else REMOVING_BACKGROUND, n, len(views))
        units.append(float(crop.info.get("spellcast_unit", 1.0)))
        images.append(pixal3d_mv.cut_out(pipeline, crop, view.azimuth))
    azimuths = [v.azimuth for v in views]
    sources = [os.path.normcase(os.path.abspath(v.image_path)) for v in views]
    framed = pixal3d_mv.frame_views(images, azimuths, sources, units)
    pixal3d_mv.save_views(req.views_dir, [(pixal3d_mv.VIEW_NAMES[a], im) for a, im in zip(azimuths, framed)])
    return framed


def run_multi_view(px: ModuleType, pipeline: Any, req: GenerateRequest, views: list[ViewSpec], framed: list[Any],
                   seed: int, defaults: dict[str, Any]) -> tuple[list[Any], int]:
    """The multi-view cascade on the `framed` views (prepare_views), the front one first, posed on the
    example rig (pixal3d_mv); no camera estimate. Same sampler settings as the single-view path. Returns
    (meshes, resolution)."""
    azimuths = [v.azimuth for v in views]
    bundle = pixal3d_mv.view_bundle(framed, azimuths)
    print(f"[spellcast] Multi-view: {', '.join(bundle['view_names'])} on the example rig "
          f"({math.degrees(pixal3d_mv.MV_CAMERA_ANGLE_X):.0f} degree field of view, distance "
          f"{pixal3d_mv.MV_CAMERA_DISTANCE:.2f}).", flush=True)
    meshes, (_, _, res) = pipeline.run_mv(   # run_mv sets preprocess_image=False itself
        bundle, seed=seed,
        sparse_structure_sampler_params=sampler_params(defaults, "ss"),
        shape_slat_sampler_params=sampler_params(defaults, "shape_slat"),
        tex_slat_sampler_params=sampler_params(defaults, "tex_slat"),
        return_latent=True,
        pipeline_type=f"{req.resolution}_cascade", max_num_tokens=px.CASCADE_MAX_NUM_TOKENS)
    return meshes, res


def streak_warning(glb: Any) -> str | None:
    """The artist's warning when the final mesh has the long fins along the camera axis (GLB Z) that a
    failed guess of the hidden side leaves (pixal3d_guard.streak_share), else None."""
    try:
        share = pixal3d_guard.streak_share(glb.vertices, glb.faces)
    except Exception as exc:   # a check must never cost the mesh
        print(f"[spellcast] Note: could not check the mesh for streaks ({exc}).", flush=True)
        return None
    stretched = share > pixal3d_guard.STREAK_THRESHOLD
    print(f"[spellcast] Streak check: {share:.1%} of the surface runs along the camera axis "
          f"(limit {pixal3d_guard.STREAK_THRESHOLD:.0%})" + ("; warning the artist." if stretched else "."),
          flush=True)
    return pixal3d_guard.STREAK_WARNING if stretched else None


def glb_steps(px: ModuleType, mesh: Any, attr_layout: Any, resolution: int, req: GenerateRequest, multi: bool,
              ss_retries: int, blocked: bool, inline: bool = False, work_folder: Path | None = None) -> Any:
    """A decoded mesh - the reference pipeline's or the staged engine's (pixal3d.staged DecodedMesh) -> the
    steps that clean, unwrap and bake it into the GLB at the request's out_path, turned to face +Z (GLB_ROTATION),
    and say what the hub keeps with the mesh: {"mode", "views", "warning", "ss_retries", "resolution"}
    (pixal3d_glb: o_voxel's to_glb with the launcher's arguments, cut where its work changes between the card
    and the CPU). Both paths come here, so the file and the answer are the same whichever made it: a batch hands
    the steps to the engine, which runs their CPU half while the next prop is decoded; the reference path runs
    them in a row (`inline`, pixal3d_glb.run_all). `resolution` can be lower than asked when the shape has too
    many tokens; `blocked`: every draw of the sparse structure was a block, so the mesh is the best of them and
    the artist is warned as for a streaked one. A prop the artist deleted meanwhile fails (kind "input") before
    its file is written, and its folder is not made again. `work_folder`: the prop's folder in a batch's work folder,
    where its texture volume waits for the bake (pixal3d_glb.TextureVoxels); None keeps it in RAM."""
    views = req.views or []
    return pixal3d_glb.finish_steps(px, mesh, attr_layout, resolution, req.decimation_target, req.texture_size,
                                    req.out_path, multi, ss_retries, blocked, GLB_ROTATION, streak_warning,
                                    views=len(views), inline=inline, work_folder=work_folder)


def announce(req: GenerateRequest, seed: int, prop: str = "") -> None:
    """One log line per mesh that is asked for: what it is made from and with which settings."""
    views = req.views or []
    source = req.image_path if not views else "; ".join(
        f"{pixal3d_mv.VIEW_NAMES.get(v.azimuth, v.azimuth)} {v.image_path}" + (f" {v.rect}" if v.rect else "")
        for v in pixal3d_mv.ordered(views))
    print(f"[spellcast] Mesh for {prop + ': ' if prop else ''}{source} (quality {req.resolution}, seed {seed}, "
          f"detail {req.decimation_target}, texture {req.texture_size})", flush=True)


# ---- batches on the staged engine ---------------------------------------------------------------

def staged_module() -> Any:
    """The fork's staged engine (pixal3d.staged): the engine, its options, and the types its hooks hand back.
    Imported when it is first needed: the fork is on the import path only once main() has put it there."""
    import pixal3d.staged as module

    return module


def spill_option(options_type: Any, spill: bool) -> dict[str, bool]:
    """{"spill": spill} for the engine's options when they have that field, else {}: a fork from before it was
    added lets a prop spill as it always did. The guard goes once the fork is pinned at a version with the field."""
    return {"spill": bool(spill)} if "spill" in {f.name for f in dataclasses.fields(options_type)} else {}


def hi3d_paths(models_root: Path) -> Any:
    """Where Hi3D's code and networks lie, or None when any of them is missing: models/hi3d beside Pixal3D's models
    folder, its code in code/hi3d (SPELLCAST_HI3D_CODE names a checkout elsewhere)."""
    from pixal3d.staged import hi3d

    root = Path(models_root).resolve().parent / "hi3d"
    code = Path(os.environ.get("SPELLCAST_HI3D_CODE") or Path(__file__).resolve().parents[2] / "hi3d")
    paths = hi3d.Hi3DPaths(code=code, weights=root / "Hi3Dv1.0", dino=root / "dinov2-large",
                           clip=root / "clip-vit-large-patch14",
                           encoder=root / "trellis2-shape-encoder" / "ckpts" / "shape_enc_next_dc_f16c32_fp16")
    needed = [code / "sparc3d" / "__init__.py", *[root / rel for rel in HI3D_NEEDED]]
    return paths if all(path.is_file() for path in needed) else None


# Every network file and config of Hi3D the installer pins (install/models.py HI3D_FILES, its licence texts aside;
# test_hi3d.py holds the two together): a download that stopped between them must not stop the engine, since
# Pixal3D's own props are made without Hi3D.
HI3D_NEEDED = (
    "Hi3Dv1.0/model_index.json", "Hi3Dv1.0/scheduler/scheduler_config.json", "Hi3Dv1.0/vae/config.json",
    "Hi3Dv1.0/vae/model.safetensors", "Hi3Dv1.0/transformer_512/config.json",
    "Hi3Dv1.0/transformer_512/model.safetensors", "Hi3Dv1.0/transformer_1024/config.json",
    "Hi3Dv1.0/transformer_1024/model.safetensors", "dinov2-large/config.json", "dinov2-large/model.safetensors",
    "dinov2-large/preprocessor_config.json", "clip-vit-large-patch14/config.json",
    "clip-vit-large-patch14/model.safetensors", "clip-vit-large-patch14/preprocessor_config.json",
    "trellis2-shape-encoder/ckpts/shape_enc_next_dc_f16c32_fp16.json",
    "trellis2-shape-encoder/ckpts/shape_enc_next_dc_f16c32_fp16.safetensors")


def engine_options(models_root: Path, spill: bool = False) -> Any:
    """The engine as the hub uses it: Pixal3D's networks from the snapshot folder its installer downloaded,
    the four small models (background removal, camera, DINOv3, NAF) by the fork's own builders, the rest as
    the engine's defaults have it - every part for all the props of a batch unless a request names a wave.
    `spill`: a prop that does not fit into the card's free memory may spill into shared memory (slowly); without
    it such a prop fails with the kind "oom" once the engine's memory ladder is through (the hub's
    pixal3d.allow_spill, sent with every batch)."""
    from pixal3d.staged import builders

    snapshot = pixal3d_mv.snapshot_dir(models_root)
    if snapshot is None:
        raise FileNotFoundError(f"Pixal3D's model files in {Path(models_root) / pixal3d_mv.REPO_CACHE}")
    small = builders.default_builders(builders.default_paths(Path(models_root)))
    options = staged_module().EngineOptions
    extra = spill_option(options, spill)
    if "hi3d" in {f.name for f in dataclasses.fields(options)}:
        extra["hi3d"] = hi3d_paths(models_root)
    return options(pipeline_dir=snapshot, builders=small, **extra)


def build_engine(options: Any, hooks: Any) -> Any:
    """The staged engine the batches run on. Making it reads the pipeline's settings and the headers of the
    network files: no weights, and nothing touches the card. BatchRunner looks this function up on the module
    every time, so a tool that measures or checks the launcher replaces it from outside to reach the engine."""
    return staged_module().StagedEngine(options, hooks)


def estimate_camera(px: ModuleType, moge: Any, image_path: str, device: str = "cuda") -> dict[str, float]:
    """app.py's get_camera_params_wild_moge with the hub's settings (px.WILD_*) for a MoGe-2 model that is on
    the card already: its lines without the model's moves to the card and back (the staged engine loads the
    model before a batch's camera estimates and drops it after them). Python floats, as they are computed."""
    import numpy as np
    import torch
    from PIL import Image

    mesh_scale, extend_pixel, image_resolution = px.WILD_MESH_SCALE, px.WILD_EXTEND_PIXEL, px.WILD_IMAGE_RESOLUTION
    with Image.open(image_path) as source:
        pil_image = source.convert("RGB")
    width, _ = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    with torch.no_grad():
        output = moge.infer(image_tensor)
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx_normalized = intrinsics[0, 0]
    fx = fx_normalized * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))

    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = px.distance_from_fov(
        camera_angle_x, grid_point,
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale, image_resolution)["distance_from_x"]
    return {"camera_angle_x": camera_angle_x, "distance": distance, "mesh_scale": mesh_scale}


class OpenAhead:
    """Opens the pictures of the props next in line for a batch's cut-out on CPU threads while the GPU thread cuts out
    the prop before them. Opening a picture (pixal3d_mv.open_crop: decoding, the EXIF turn, the crop, the resize to the
    input size) is CPU work the card would otherwise wait for, picture after picture. Only the opening moves: the matte
    on the card, the stray bits, the framing and the views saved stay on the GPU thread in the order they had, and
    open_crop makes the same picture on any thread, so every prop is prepared as before.

    BatchRunner hands it a batch's props (begin) and the engine's events. When the cut-out starts ("part_start" of
    "cutout": the props of a wave in the engine's order, the multi-view ones first) the pictures of the first `ahead`
    props start opening, and each prop that takes its own (take, in BatchHooks.prepare) starts the next prop's: at
    most `ahead` props' pictures wait in RAM (3 MB a 1024 px RGB picture, 16 MB at most), and none once the part, or
    the batch, ends. A prop the engine passes over (failed, cancelled) has its pictures dropped when a later prop takes
    its own. A prop that is not in line - the engine tries a part again after the card ran out of memory, a check
    calls prepare alone - opens its pictures itself. A picture that cannot be opened raises in its prop's prepare,
    when its turn comes, the exception open_crop raised: that prop fails as it did, the others go on.

    Used on the GPU thread alone (the engine's hooks and events, BatchRunner._run); the threads only open."""

    def __init__(self, ahead: int = OPEN_AHEAD, threads: int = OPEN_THREADS) -> None:
        self.ahead = ahead
        self.threads = threads
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None       # made when first needed
        self._props: dict[str, Any] = {}        # the batch's PropRequests by id
        self._line: list[str] = []              # the props the cut-out takes next, in its order
        self._opened: dict[str, list[concurrent.futures.Future[Any]]] = {}

    def begin(self, props: list[Any]) -> None:
        """A batch starts with these props (PropRequest, the payload a BatchItem). Nothing is opened before its
        cut-out does."""
        self.end()
        self._props = {prop.id: prop for prop in props}

    def end(self) -> None:
        """The batch is over, done, failed or cancelled: the pictures still waiting are dropped."""
        self._drop(list(self._opened))
        self._line = []
        self._props = {}

    def on_event(self, event: Any) -> None:
        """The engine's event: the start of the cut-out lines its props up, its end drops what is left."""
        if event.part != "cutout" or event.kind not in ("part_start", "part_end"):
            return
        self._drop(list(self._opened))
        self._line = []
        if event.kind == "part_start":
            self._line = [prop_id for prop_id in (event.detail or {}).get("props", ()) if prop_id in self._props]
            self._fill()

    def take(self, prop_id: str) -> list[concurrent.futures.Future[Any]] | None:
        """The prop's pictures, one future per picture in the order prepare cuts them out (picture_openers), or None:
        it opens them itself. The props after it in line start opening theirs."""
        if prop_id not in self._line:
            return None
        position = self._line.index(prop_id)
        self._drop(self._line[:position])
        self._line = self._line[position + 1:]
        opened = self._opened.pop(prop_id, None)
        self._fill()
        return opened

    def _fill(self) -> None:
        """Start opening the pictures of the first `ahead` props in line. Opening ahead only saves time: when it
        cannot start, that is noted, and the props open their own pictures."""
        try:
            for prop_id in self._line[:self.ahead]:
                if prop_id not in self._opened:
                    prop = self._props[prop_id]
                    if self._pool is None:
                        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.threads,
                                                                           thread_name_prefix="pixal3d-open")
                    openers = picture_openers(prop.payload, prop.multi)
                    self._opened[prop_id] = [self._pool.submit(fn) for fn in openers]
        except Exception:
            traceback.print_exc()
            print("[spellcast] Note: the next pictures could not be opened ahead; each prop opens its own.",
                  flush=True)

    def _drop(self, prop_ids: list[str]) -> None:
        for prop_id in prop_ids:
            for future in self._opened.pop(prop_id, ()):
                future.cancel()                 # one under way finishes, and is let go


class BatchHooks:
    """What the staged engine asks the launcher for while it makes a prop (pixal3d.staged Hooks): everything
    that is the launcher's own, done by the functions the reference path uses - opening, cutting out and
    framing the pictures, the camera estimate, the notes of the sparse-structure guard, the mesh clean-up and
    the GLB. The engine calls all of it on the GPU worker, with `req.payload` the prop's BatchItem. A picture
    the artist has to fix fails that prop alone (PropFailed, kind "input"): after the first part of a batch
    every prop's input problem is known.

    A prop the artist deleted while the batch runs (its folder went to deleted/) fails at the next hook it comes
    to - prepare, camera, the sparse structure's prop_scope, finish - with the kind "input" too: the engine cannot
    be told to drop one prop, and nothing is written for it, nor its folder made again (the hub has seen it go and
    calls it cancelled).

    `ahead` opens the pictures of the props next in line while the card cuts out the one before them (OpenAhead);
    BatchRunner tells it which props a batch has and what the engine does."""

    def __init__(self, px: ModuleType, notes: RunNotes) -> None:
        self.px = px
        self.notes = notes
        self.ahead = OpenAhead()
        self.work_dir: Path | None = None       # the running batch's work folder (BatchRunner sets it): <it>/<prop id>
        #                                         is the prop's own folder, where its texture volume waits for the bake

    @staticmethod
    def still_there(req: Any) -> None:
        """PropFailed (kind "input") when the prop's folder is gone."""
        payload = getattr(req, "payload", None)
        if payload is not None and pixal3d_mv.prop_gone(payload.out_path):
            raise staged_module().PropFailed(pixal3d_mv.PROP_DELETED, "input")

    def prepare(self, req: Any, matte: Callable[[Any], Any]) -> Any:
        """The prop's picture, cut out and framed as the models take it: one RGB picture on black, or for a
        prop made from several views its framed RGBA views, the front one first. `matte` removes a
        background with the engine's model. The views are saved to the request's views_dir. The pictures come from
        `ahead` when they were opened there, else they are opened here."""
        types = staged_module()
        opened = self.ahead.take(req.id)        # and the next props' pictures start opening
        self.still_there(req)
        payload = req.payload
        views = pixal3d_mv.ordered(payload.views or [])
        cutter = matte_shim(matte)
        try:
            if req.multi:
                framed = prepare_views(self.px, cutter, payload, views, opened)
                azimuths = [v.azimuth for v in views]
                print(f"[spellcast] Multi-view: {', '.join(pixal3d_mv.VIEW_NAMES.get(a, str(a)) for a in azimuths)} "
                      f"on the example rig ({math.degrees(pixal3d_mv.MV_CAMERA_ANGLE_X):.0f} degree field of view, "
                      f"distance {pixal3d_mv.MV_CAMERA_DISTANCE:.2f}).", flush=True)
                return types.Prepared(images=framed, meta={"azimuths": azimuths})
            view = views[0] if views else None
            picture = opened[0].result() if opened else open_picture(payload, view)
            cutout: list[Any] = []
            image = preprocess(self.px, picture, cutter, keep=cutout if payload.geometry == "hi3d" else None)
            pixal3d_mv.save_views(payload.views_dir,
                                  [(pixal3d_mv.VIEW_NAMES.get(view.azimuth if view else 0, ""), image)])
        except pixal3d_mv.ViewProblem as exc:       # PropDeleted too
            raise types.PropFailed(str(exc), "input") from None
        if cutout:
            return types.Prepared(images=[image], meta={}, cutout=cutout[0])
        return types.Prepared(images=[image], meta={})

    def camera(self, req: Any, image: Any, moge: Any) -> dict[str, float]:
        """The camera of a prop made from one picture, estimated by `moge` (on the card) from the prepared
        picture, read back from a file as the reference path reads it."""
        self.still_there(req)
        camera_png = Path(self.px.TMP_DIR) / f"spellcast_{uuid.uuid4().hex}.png"
        try:
            image.save(camera_png)
            self.px._update_progress(ESTIMATING_CAMERA, 0, 1)
            return estimate_camera(self.px, moge, str(camera_png))
        finally:
            camera_png.unlink(missing_ok=True)

    def bundle(self, req: Any, images: list[Any], meta: dict[str, Any]) -> dict[str, Any]:
        """The framed views of a multi-view prop as the multi-view models take them (pixal3d_mv.view_bundle)."""
        return pixal3d_mv.view_bundle(images, meta["azimuths"])

    @contextlib.contextmanager
    def prop_scope(self, req: Any, part: str) -> Iterator[dict[str, Any]]:
        """Around one prop's turn in a part. In "ss" the guard on engine.sample_sparse_structure counts its
        redraws in the one RunNotes object: it is cleared when the prop's turn starts and read when it ends,
        into the dict the engine keeps with the prop (and hands to finish)."""
        self.still_there(req)       # the last hook before the long sampling parts
        found: dict[str, Any] = {}
        if part != "ss":
            yield found
            return
        self.notes.reset()
        try:
            yield found
        finally:
            found.update(ss_retries=self.notes.ss_retries, ss_blocked=self.notes.ss_blocked)

    @staticmethod
    def gpu_torch_mib(req: Any) -> int:
        """MiB the GLB GPU steps reserve through PyTorch for this prop, by its texture size (parts.pair_keep asks:
        the engine keeps the texture network and the decoders on a big card only where that fits beside them).
        Measured on the golden meshes: geometry 1,034-1,444 MiB and bake 1,068-1,234 at 2048; bake 2,454-2,498 at
        4096. 500 + 500 per 1024 px of texture width: 1,500 at 2048, 2,500 at 4096, 4,500 at 8192."""
        size = int(getattr(req.payload, "texture_size", 2048) or 2048)
        return max(1_500, 500 + 500 * size // 1024)

    def finish(self, req: Any, mesh: Any, notes: dict[str, Any]) -> Any:
        """The decoded mesh -> the steps that write the GLB at the request's out_path (glb_steps), handed to the
        engine as a FinishSteps: it runs their CPU half on its worker thread while the next prop is decoded. Their
        result is what the hub keeps with the mesh. A prop deleted during the clean-up fails in the last step,
        before its file is written (kind "input"). The texture volume waits for the bake in the prop's folder of the
        batch's work folder (the engine's store made it, and clears it when the prop is over); the environment's
        PIXAL3D_VOXELS_ON_DISK=off keeps it in RAM (a measurement's other arm, or a PC whose temp disk is slow)."""
        self.still_there(req)       # before the minutes of the clean-up
        payload = req.payload
        notes = notes or {}
        keep_in_ram = os.environ.get("PIXAL3D_VOXELS_ON_DISK", "").lower() in ("0", "off", "false", "no")
        folder = self.work_dir / req.id if self.work_dir is not None and not keep_in_ram else None
        steps = glb_steps(self.px, mesh, mesh.attr_layout, int(mesh.resolution), payload, req.multi,
                          int(notes.get("ss_retries") or 0), bool(notes.get("ss_blocked")), work_folder=folder)
        cap = notes.get("memory_cap")
        if isinstance(cap, dict) and cap.get("resolution") != cap.get("without"):
            # The engine made the prop at less detail than the token cap allows, to fit the card or the RAM
            # (parts.resolution_fits): the artist is told with the model, as for a streaked one.
            inner = steps.result

            def result(state: dict[str, Any]) -> dict[str, Any]:
                made = dict(inner(state))
                made["warning"] = " ".join(text for text in (made.get("warning"), lower_detail_words(cap)) if text)
                return made

            steps.result = result
        return steps


def checkpoint_loader() -> Any:
    """The module init_models() loads every checkpoint through (ModelModes.loading_first swaps its loader)."""
    from pixal3d import models

    return models


class Reference:
    """The reference path's models: everything Pixal3D's init_models() loads, kept in RAM (about 25 GB), for
    POST /spellcast/generate and Pixal3D's own page.

    A start loads none of it. ensure() loads it when one of the two is first used - the set --first-job names,
    as a start used to - and unload() drops it again before a batch: the staged engine needs the memory, and
    DINOv3 would otherwise be held twice. `release` makes the staged engine let go of its networks and of its
    limit on PyTorch's memory first. Call everything on the GPU worker."""

    def __init__(self, px: ModuleType, modes: pixal3d_mv.ModelModes, notes: RunNotes, first: str,
                 shared: SharedModels, release: Callable[[], None]) -> None:
        self.px = px
        self.modes = modes
        self.notes = notes
        self.first = first
        self.shared = shared
        self.release = release
        self.loaded = False

    def ensure(self) -> None:
        self.release()
        if self.loaded:
            return
        self.unload()   # what a load that failed half-way left
        print("[spellcast] Loading Pixal3D's own pipeline with all its models (about 25 GB of RAM and page "
              "file; this can take several minutes)...", flush=True)
        started = time.monotonic()
        with self.modes.loading_first(self.first, checkpoint_loader()):
            self.px.init_models()
        prepare_pipeline(self.px, self.px.pipeline, self.notes)
        self.loaded = True
        print(f"[spellcast] Pixal3D's own pipeline is loaded ({time.monotonic() - started:.0f} s).", flush=True)

    def ensure_single_view(self) -> None:
        """Before a call of Pixal3D's own page: it runs app.pipeline, which needs the single-view models."""
        self.ensure()
        self.modes.ensure("sv")

    def unload(self) -> None:
        px = self.px
        if not self.loaded and getattr(px, "pipeline", None) is None and getattr(px, "moge_model", None) is None:
            return
        px.pipeline = None
        px.moge_model = None
        px.envmap = None
        self.modes.reset()
        self.shared.clear()
        self.loaded = False
        gc.collect()   # the launcher's wrappers and the pipeline refer to each other
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        print("[spellcast] Pixal3D's own pipeline was unloaded.", flush=True)


_frozen = False


def freeze_startup_objects() -> None:
    """Once, before the first batch: everything Python holds since the start (app.py, gradio, torch, transformers:
    about half a million objects) goes to the collector's permanent generation, so the engine's `gc.collect()`
    before each of its 70-odd `empty_cache()` calls per batch (a network loaded or dropped, a part trimmed) walks
    the batch's own objects alone: about 0 ms instead of 106 ms each (measured, round G: 7.6 s of a 1,309 s batch,
    almost all with the card idle). What a batch makes is collected as before; what was frozen lives as long as
    the process, as it did."""
    global _frozen
    if not _frozen:
        gc.collect()
        gc.freeze()
        _frozen = True


class BatchRunner:
    """The staged engine and the batches it runs for the hub.

    build() makes the engine, at the start. start() takes a batch request and answers at once: 202 with the
    run order, or what is wrong with the request as a whole; the work is queued on the GPU worker. A prop whose
    own request is wrong starts as failed and the others are made. The record of the batch (`batches`,
    pixal3d_batch.Batch) follows the engine's events, and so does the file in the work folder that says which
    prop is on the card (pixal3d_batch.ON_CARD); `unload` drops the reference path's models first."""

    def __init__(self, px: ModuleType, gpu: GpuWorker, hooks: Any, notes: RunNotes, defaults: dict[str, Any],
                 models_root: Path, projects_root: Path | None = None,
                 unload: Callable[[], None] | None = None) -> None:
        self.px = px
        self.gpu = gpu
        self.hooks = hooks
        self.notes = notes
        self.defaults = defaults
        self.models_root = Path(models_root)
        self.projects_root = projects_root
        self.unload = unload if unload is not None else lambda: None
        self.batches = pixal3d_batch.BatchRegistry()
        self.engine: Any = None
        # What the engine was made with (EngineOptions.spill); a batch that asks otherwise makes it again.
        self.spill = False

    def build(self) -> None:
        """Make the engine and put the sparse-structure guard on it. Call on the GPU worker."""
        engine = build_engine(engine_options(self.models_root, spill=self.spill), self.hooks)
        guard_sparse_structure(engine, self.notes)
        parts = tuple(getattr(staged_module(), "PARTS", pixal3d_batch.PART_KEYS))
        if parts not in (pixal3d_batch.PART_KEYS, pixal3d_batch.ALL_KEYS):
            print(f"[spellcast] Note: this Pixal3D works in the parts {', '.join(parts)}; the progress of a 3D "
                  f"model is laid out for {', '.join(pixal3d_batch.ALL_KEYS)}.", flush=True)
        self.engine = engine

    def release(self) -> None:
        """Before the reference path runs: the engine drops its networks and its limit on PyTorch's memory."""
        if self.engine is not None:
            self.engine.release()

    def start(self, body: BatchRequest) -> tuple[int, dict[str, Any]]:
        """POST /spellcast/batch -> (HTTP status, answer)."""
        problem = self._refusal(body)
        if problem is not None:
            return 400, {"detail": problem, "kind": "input"}
        running = self.batches.running()
        if running is not None:
            return self._busy(running)
        work_dir = Path(body.work_dir)
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return 400, {"detail": f"Cannot create the work folder {work_dir}: {exc.strerror}.", "kind": "input"}
        several = pixal3d_mv.multiview_available(self.models_root)
        items: list[pixal3d_batch.Item] = []
        plans: dict[str, tuple[BatchItem, int, bool]] = {}
        for payload in body.items:
            views = payload.views or []
            multi = len(views) >= 2
            error = self._item_problem(payload, multi, several)
            items.append(pixal3d_batch.Item(payload.id, payload.out_path, payload.seed, multi,
                                            len(views) if multi else 1, payload.resolution, error,
                                            geometry=payload.geometry))
            if error is None:
                plans[payload.id] = (payload, payload.seed, multi)
        batch = pixal3d_batch.Batch(body.batch_id, items, failure_words, marker=work_dir / pixal3d_batch.ON_CARD,
                                    finish_steps=len(pixal3d_glb.STEPS))
        running = self.batches.begin(batch)
        if running is not None:
            return self._busy(running)
        batch.mark()
        print(f"[spellcast] Batch {batch.batch_id}: {len(plans)} of {len(items)} props to make"
              + (" (after a restart: props whose texture was sampled are only built)" if body.resume else ""),
              flush=True)
        for prop_id, (payload, seed, _) in plans.items():
            announce(payload, seed, prop_id)
        if plans:
            # A number goes to the engine as it is (0: no waves); without one the engine's own setting holds.
            wave = body.wave_size if body.wave_size is not None and body.wave_size >= 0 else None
            try:
                self.gpu.submit(self._work, batch, plans, work_dir, wave, body.resume, body.spill)
            except Exception as exc:
                # No thread will ever finish it, and a batch that stays running makes Pixal3D busy for ever.
                traceback.print_exc()
                text = failure_text(exc, "error")
                batch.fail("error", text)
                return 500, {"detail": failure_words("error", text), "kind": "error"}
        else:
            batch.finish()
        return 202, {"batch_id": batch.batch_id, "order": batch.order,
                     "parts": [{"key": part.key, "label": part.label, "start": part.start, "end": part.end}
                               for part in pixal3d_batch.PARTS]}

    @staticmethod
    def _busy(running: pixal3d_batch.Batch) -> tuple[int, dict[str, Any]]:
        return 409, {"detail": "Pixal3D is still working on another job.", "kind": "busy",
                     "batch_id": running.batch_id}

    def _refusal(self, body: BatchRequest) -> str | None:
        """What is wrong with a batch request as a whole, or None."""
        ids = [item.id for item in body.items]
        if not ids:
            return "The batch has no props."
        if len(set(ids)) != len(ids):
            return "A prop is in the batch twice."
        unusable = [name for name in (body.batch_id, *ids) if not PROP_ID.fullmatch(name)]
        if unusable:
            return f"A job or prop cannot be named '{unusable[0]}': letters, digits, _ and - only."
        if any(outside_projects(item, self.projects_root, body.work_dir) for item in body.items):
            return OUTSIDE_PROJECTS
        negative = next((item for item in body.items if item.seed < 0), None)
        if negative is not None:
            # Drawn here, a random seed would differ in the job's second attempt, and so would the fingerprint by
            # which a prop's sampled work is resumed: the hub draws it, once per job.
            return (f"The seed of prop '{negative.id}' is {negative.seed}: a batch takes seeds of 0 or more (the hub "
                    "draws a random one itself).")
        return work_dir_problem(body.work_dir, self.projects_root)

    def _item_problem(self, payload: BatchItem, multi: bool, several: bool) -> dict[str, str] | None:
        """What is wrong with one prop's request - then the prop starts as failed -, or None."""
        problem = validate(payload)
        if problem is not None:
            return {"detail": problem, "kind": "input"}
        if multi and not several:
            return {"detail": pixal3d_mv.MV_MISSING, "kind": "mv_missing"}
        folder = Path(payload.out_path).parent
        try:
            pixal3d_mv.make_folder(folder)
        except pixal3d_mv.PropDeleted as exc:
            return {"detail": str(exc), "kind": "input"}
        except OSError as exc:
            return {"detail": f"Cannot create the output folder {folder}: {exc.strerror}.", "kind": "input"}
        return None

    def _work(self, batch: pixal3d_batch.Batch, plans: dict[str, tuple[BatchItem, int, bool]], work_dir: Path,
              wave_size: int | None, resume: bool, spill: bool = False) -> None:
        """One batch, on the GPU worker. Whatever goes wrong ends in its record: the engine's BatchBroken as
        the kind "restart" (the card cannot be used any more in this process), anything else that escapes the
        engine as an error of the props that were not made yet. This never returns with the batch still
        running - a KeyboardInterrupt out of the engine, or a throw of finish() itself, would otherwise leave
        the registry calling Pixal3D busy for the life of the process."""
        started = time.monotonic()
        results: dict[str, dict[str, Any]] | None = None
        failure: tuple[str, str] | None = None
        try:
            try:
                if not batch.cancel.is_set():
                    results = self._run(batch, plans, work_dir, wave_size, resume, spill)
            except Exception as exc:
                traceback.print_exc()
                kind = pixal3d_batch.classify(exc)
                failure = (kind, failure_text(exc, kind))
            # Outside the except block: a traceback pins the engine's frames and every tensor in them.
            if failure is None:
                batch.finish(results)
            else:
                batch.fail(*failure)
                if failure[0] != "restart":
                    try:
                        self.engine.release()
                    except Exception:
                        traceback.print_exc()
        finally:
            if batch.running():
                batch.fail("stopped", BATCH_STOPPED)
            told = batch.snapshot()
            made = sum(1 for item in told["items"].values() if item["state"] == pixal3d_batch.DONE)
            print(f"[spellcast] Batch {batch.batch_id} {told['state']}: {made} of {len(told['items'])} props made "
                  f"in {time.monotonic() - started:.0f} s", flush=True)

    def _run(self, batch: pixal3d_batch.Batch, plans: dict[str, tuple[BatchItem, int, bool]], work_dir: Path,
             wave_size: int | None, resume: bool, spill: bool = False) -> dict[str, dict[str, Any]]:
        types = staged_module()
        self.unload()
        needs_views = any(multi for _, _, multi in plans.values()) and not getattr(self.engine, "multiview", True)
        # an engine made before Hi3D's networks were installed finds them only when it is made again
        needs_hi3d = any(payload.geometry == "hi3d" for payload, _, _ in plans.values()) \
            and getattr(self.engine, "hi3d_scheduler", True) is None
        if spill != self.spill or needs_views or needs_hi3d:
            self._rebuild(spill)
        tokens = self.px.CASCADE_MAX_NUM_TOKENS
        samplers = {"ss": sampler_params(self.defaults, "ss"), "shape": sampler_params(self.defaults, "shape_slat"),
                    "tex": sampler_params(self.defaults, "tex_slat")}
        props = []
        for prop_id, (payload, seed, multi) in plans.items():
            geometry = {"geometry": payload.geometry} if payload.geometry != "pixal3d" else {}
            props.append(types.PropRequest(
                id=prop_id, seed=seed, resolution=payload.resolution, multi=multi,
                sampler_params=samplers, max_num_tokens=tokens,
                request_hash=fingerprint(payload, seed, multi, samplers, tokens), payload=payload, **geometry))
        ahead = getattr(self.hooks, "ahead", None)

        def on_event(event: Any) -> None:
            READ_AHEAD_GATE.on_event(event)
            if ahead is not None:
                try:
                    ahead.on_event(event)
                except Exception:   # opening ahead only saves time: it never costs a prop, nor the batch's record
                    traceback.print_exc()
            batch.on_event(event)

        if ahead is not None:
            ahead.begin(props)
        if hasattr(self.hooks, "work_dir"):
            self.hooks.work_dir = Path(work_dir)
        freeze_startup_objects()
        try:
            with batch.at_work():
                return self.engine.run(props, work_dir, on_event=on_event, cancelled=batch.cancel.is_set,
                                       wave_size=wave_size, resume=resume)
        finally:
            READ_AHEAD_GATE.release()
            if ahead is not None:
                ahead.end()
            if hasattr(self.hooks, "work_dir"):
                self.hooks.work_dir = None

    def _rebuild(self, spill: bool) -> None:
        """Make the engine again (no weights are read): its options hold `spill`, which a batch may ask otherwise
        than the last one did, and it lists its networks when it is made, so multi-view models (or Hi3D's networks)
        that were downloaded since are found only by a new one. If that fails the old one stays, and says what is
        missing."""
        old, before, had = self.engine, self.spill, getattr(self.engine, "multiview", True)
        self.spill = spill
        try:
            self.build()
        except Exception as exc:
            self.spill = before
            print(f"[spellcast] Note: could not make Pixal3D's engine again ({type(exc).__name__}: {exc}).",
                  flush=True)
            return
        old.close()
        if spill != before:
            print(f"[spellcast] A prop that does not fit into the graphics card's free memory "
                  f"{'may now use shared memory (slowly)' if spill else 'now fails instead of using shared memory'}.",
                  flush=True)
        if not had and getattr(self.engine, "multiview", True):
            print("[spellcast] Pixal3D found its multi-view models.", flush=True)


def fingerprint(payload: BatchItem, seed: int, multi: bool, samplers: dict[str, Any], max_num_tokens: int) -> str:
    """pixal3d_batch.request_hash of a prop's request: the seed used, the quality, whether it is made from
    several views, the sampler settings, the token limit, and each picture it is made from - its size and
    change time on disk, its rectangle and its direction. Detail and texture size are not part of it: they
    only shape the mesh clean-up, which a resumed prop goes through again."""
    views = pixal3d_mv.ordered(payload.views or [])
    sources = [(v.image_path, v.rect, v.azimuth) for v in views] or [(payload.image_path, None, 0)]
    pictures = []
    for path, rect, azimuth in sources:
        try:
            stat = os.stat(path)
            size, changed = stat.st_size, stat.st_mtime_ns
        except OSError:
            size = changed = None
        pictures.append({"size": size, "changed": changed, "rect": rect, "azimuth": azimuth})
    geometry = {"geometry": payload.geometry} if payload.geometry != "pixal3d" else {}
    return pixal3d_batch.request_hash(seed=seed, resolution=payload.resolution, multi=multi, sampler_params=samplers,
                                      max_num_tokens=max_num_tokens, pictures=pictures, **geometry)


def route_progress(px: ModuleType, batches: pixal3d_batch.BatchRegistry) -> None:
    """While the engine runs a batch, send what is reported through px._update_progress - by the launcher's
    hooks ("Removing the background", "Saving GLB") and by every tqdm bar of the mesh clean-up, which app.py
    routes there - to the prop of the batch that is being worked on, each report counted as a change of the
    batch. At any other time it goes to the progress file of the session that runs, as app.py has it. app.py's
    bar class and this file call the function through the module, so replacing the attribute is enough."""
    write = px._update_progress

    @functools.wraps(write)
    def update(stage: str, step: int, total: int) -> None:
        batch = batches.here()
        if batch is None:
            write(stage, step, total)
            return
        label, fraction = describe(str(stage), step, total) or (None, None)
        batch.bar(label, fraction, step, total)

    px._update_progress = update


def add_hub_routes(px: ModuleType, gpu: GpuWorker, defaults: dict[str, Any], modes: pixal3d_mv.ModelModes,
                   notes: RunNotes, projects_root: Path | None, reference: Reference, runner: BatchRunner) -> None:
    job = HubJob()
    progress_file = Path(px._progress_file(SESSION))

    @px.app.get("/spellcast/health")
    async def spellcast_health() -> dict[str, Any]:
        # ready: always true, because main() only serves once the engine is made (no model is read before a
        # job needs it); the hub waits for this answer. multiview: the multi-view weights are on disk.
        return {"ready": runner.engine is not None, "low_vram": bool(px.LOW_VRAM),
                "multiview": pixal3d_mv.multiview_available(modes.models_root), "batch": True}

    @px.app.post("/spellcast/batch")
    async def spellcast_batch(request: Request) -> JSONResponse:
        try:
            body = BatchRequest.model_validate(json.loads(await request.body()))
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(part) for part in first.get("loc", ()))
            return error_response(400, f"The batch request cannot be read ({where}: {first.get('msg', '')}).", "input")
        except ValueError:
            return error_response(400, "The batch request is not JSON.", "input")
        # In a worker thread: it looks at every prop's files, and the event loop keeps answering meanwhile.
        status, answer = await run_in_threadpool(runner.start, body)
        return JSONResponse(answer, status_code=status)

    @px.app.get("/spellcast/batch/{batch_id}")
    async def spellcast_batch_state(batch_id: str) -> JSONResponse:
        batch = runner.batches.get(batch_id)
        if batch is None:
            return error_response(404, "Pixal3D does not know this job (it started again, or a newer job "
                                       "took its place).", "input")
        return JSONResponse(batch.snapshot())

    @px.app.post("/spellcast/batch/{batch_id}/cancel")
    async def spellcast_batch_cancel(batch_id: str) -> JSONResponse:
        batch = runner.batches.get(batch_id)
        if batch is None:
            return error_response(404, "Pixal3D does not know this job (it started again, or a newer job "
                                       "took its place).", "input")
        batch.request_cancel()
        return JSONResponse({"cancelling": True})

    @px.app.get("/spellcast/progress")
    async def spellcast_progress() -> dict[str, Any]:
        if not job.active:
            return {"stage": "Idle", "step": 0, "total": 0, "fraction": 0.0}
        if job.waiting:
            return {"stage": "Waiting for a generation started in the full Pixal3D UI to finish",
                    "step": 0, "total": 0, "fraction": 0.0}
        try:
            raw = json.loads(progress_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}   # being replaced right now: report the last known state
        stage, step, total = str(raw.get("stage") or ""), int(raw.get("step") or 0), int(raw.get("total") or 0)
        described = describe(stage, step, total)
        if described is not None:
            job.label, fraction = described
            job.fraction = max(job.fraction, fraction)
        elif stage:
            job.label = stage
        return {"stage": job.label, "step": step, "total": total, "fraction": round(job.fraction, 3)}

    # A plain def route: FastAPI runs it in a worker thread, so the event loop keeps
    # answering health/progress and Pixal3D's page while a mesh is generated.
    @px.app.post("/spellcast/generate")
    def spellcast_generate(req: GenerateRequest) -> JSONResponse:
        problem = validate(req, projects_root)
        if problem is not None:
            return error_response(400, problem, "input")
        views = req.views or []
        if len(views) >= 2 and not pixal3d_mv.multiview_available(modes.models_root):
            return error_response(409, pixal3d_mv.MV_MISSING, "mv_missing")
        out = Path(req.out_path)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return error_response(400, f"Cannot create the output folder {out.parent}: {exc.strerror}.", "input")
        seed = req.seed if req.seed >= 0 else random.randint(0, px.MAX_SEED)
        announce(req, seed)
        job.start()

        def work() -> tuple[JSONResponse | None, dict[str, Any], float]:   # runs on the GPU worker
            job.waiting = False
            started = time.monotonic()
            failure: JSONResponse | None = None
            result: dict[str, Any] = {}
            try:
                reference.ensure()            # the first call loads the models: its seconds are the mesh's alone
                started = time.monotonic()
                result = generate(px, modes, notes, req, seed, defaults)
            except Exception as exc:
                traceback.print_exc()
                out.unlink(missing_ok=True)
                failure = failure_response(exc)
            if failure is not None and px.pipeline is not None:
                release_gpu(px, modes.extractors())   # outside the except block: the traceback pins GPU tensors
            return failure, result, time.monotonic() - started

        try:
            failure, result, seconds = gpu.run(work)
        finally:
            job.active = False
            try:
                progress_file.unlink(missing_ok=True)
            except OSError:
                pass   # a progress poll has it open right now (Windows); it is rewritten next run
        if failure is not None:
            return failure
        print(f"[spellcast] Mesh saved to {out} in {seconds:.0f} s", flush=True)
        return JSONResponse({"glb_path": str(out), "seconds": round(seconds, 1), "seed": seed, **result})


# ---- main -----------------------------------------------------------------------

def startup_hint(exc: BaseException, repo: Path) -> str:
    low = str(exc).lower()
    if isinstance(exc, ImportError):
        return ("Pixal3D's Python environment is incomplete (" + str(exc) + "). "
                "Close Spellcast3D and double-click run-spellcast3d.bat again to repair it.")
    if type(exc).__name__ == "error" and type(exc).__module__.startswith("cv2") and not str(repo).isascii():
        return (f"Pixal3D could not read its own files because the Spellcast3D folder path ({repo.parents[1]}) "
                "has letters other than English ones. Move the Spellcast3D folder to a path with only English "
                "letters, such as C:\\spellcast3d, and start it again.")
    if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in low:
        return ("Pixal3D ran out of GPU memory while starting. Close other programs that use the graphics "
                "card (games, video tools, other AI apps), then start Pixal3D again.")
    if isinstance(exc, MemoryError) or "not enough memory" in low or "paging file" in low:
        return ("Pixal3D ran out of system memory (RAM and page file) while starting. "
                "Close other programs and stop the other tools, then start Pixal3D again. If it still "
                "fails, let Windows manage the page file size (search Windows for 'Adjust the "
                "performance of Windows' > Advanced > Virtual memory).")
    if isinstance(exc, FileNotFoundError):
        return MODELS_MISSING
    return ("Pixal3D could not start (the error is above). Start it again; if it keeps failing, "
            "close Spellcast3D and double-click run-spellcast3d.bat again to repair it.")


WARM_IMPORTS = ("kornia.filters", "timm")


def warm_imports() -> None:
    """Import, on a thread of its own, the libraries the background-removal model's code pulls in when it is
    first built (RMBG-2.0's birefnet.py imports kornia and timm: 3-5 s from a cold disk, 0.3 s warm, paid on the
    GPU thread at the first cut-out of a batch otherwise). Python serialises imports, so a build that comes first
    waits for the same work and no more; a library that cannot be imported only prints a note, and the build fails
    or succeeds as it would have. Started once the server is ready, so a Pixal3D started early (the hub starts it
    as the artist arrives at the 3D stage) has them by the time a job comes."""
    def run() -> None:
        for name in WARM_IMPORTS:
            try:
                importlib.import_module(name)
            except Exception as exc:    # noqa: BLE001 - whatever the library raises: the model's build says it again
                print(f"[spellcast] Note: {name} could not be imported ahead ({type(exc).__name__}: {exc}).",
                      flush=True)

    threading.Thread(target=run, name="spellcast-warm-imports", daemon=True).start()


# The files the first 3D model of a fresh start reads from the drive when a part first needs them, in the order the
# parts need them: the background-removal network and cuDNN's engines at the first cut-out (0.9 + ~1 GB), MoGe-2 for
# a single-view prop's camera (1.3 GB), DINOv3 and cuBLAS at the first picture network (1.2 + ~0.75 GB), Triton's
# compiled kernels (thousands of small files) at the first flow pass, natten and cumesh at the shape stages. Read
# from a cold drive they cost the first batch 15-30 s (the cut-out alone 2.5 -> 17 s measured on this PC's drive);
# read ahead into the system's file cache from the moment the process starts - beside its own imports, at a low I/O
# priority so that those come first - the same batch finds them in RAM. The networks the engine itself reads ahead
# (staged loader: the next part's while a part runs) are not repeated here. Nothing is kept in the process and the
# card is not touched.
READ_AHEAD_FIRST = ("torch/lib/cudnn*.dll",)
READ_AHEAD_THEN = ("torch/lib/cublas*.dll", "torch/lib/nvrtc*.dll")
READ_AHEAD_LAST = ("natten/*.pyd", "cumesh/*.pyd", "flex_gemm/**/*.pyd")
# The same on a Linux test pod: torch's CUDA libraries come in the nvidia-* wheels, the extensions are .so files
READ_AHEAD_FIRST_LINUX = ("nvidia/cudnn/lib/libcudnn*.so*",)
READ_AHEAD_THEN_LINUX = ("nvidia/cublas/lib/libcublas*.so*", "nvidia/cuda_nvrtc/lib/libnvrtc*.so*")
READ_AHEAD_LAST_LINUX = ("natten/*.so", "cumesh/*.so", "flex_gemm/**/*.so")
READ_AHEAD_CACHE_FILES = 20_000     # at most this many files of the Triton cache
READ_AHEAD_SMALL_THREADS = 8        # the Triton cache's small files are opened by this many threads at once (cold, 6,933
#                                     files: 4.5 s on one thread, 1.8 s on 4, 1.1 s on 8, measured)
READ_AHEAD_SPARE_BYTES = 1536 << 20  # RAM that must stay available beside a network's file before it is read ahead
                                     # (the staged loader's rule)


def first_networks(models: Path, first_job: str) -> list[list[Path]]:
    """The weights files of the networks a batch loads first, as staged/builders.py finds them under the models
    folder, each in a list of its own in the order the read ahead takes them beside the libraries: background
    removal (with cuDNN after it), MoGe-2 when the first job is a single picture's, DINOv3 (cuBLAS after it).
    Files that are not there are left out."""
    rembg, moge, dino = models / "RMBG-2.0" / "model.safetensors", None, models / "dinov3" / "model.safetensors"
    if first_job == "single":
        repo = models / "hub" / "models--Ruicheng--moge-2-vitl"
        try:
            revision = (repo / "refs" / "main").read_text(encoding="utf-8").strip()
        except OSError:
            revision = ""
        moge = repo / "snapshots" / (revision or "main") / "model.pt"
        if not moge.is_file():
            found = sorted((repo / "snapshots").glob("*/model.pt"))
            moge = found[-1] if found else None
    return [[path] if path is not None and path.is_file() else [] for path in (rembg, moge, dino)]


def _low_io_priority(handle: Any) -> None:
    """Windows: the reads of this open file at the lowest I/O priority but one (IoPriorityHintLow), so that the
    process's own reads (its imports, a load the engine waits for) go first; the pages read keep their normal
    memory priority (a background thread mode would lower that too, and the file cache would let them go first)."""
    if os.name != "nt":
        return
    try:
        import msvcrt

        class _Hint(ctypes.Structure):
            _fields_ = [("PriorityHint", ctypes.c_int)]

        hint = _Hint(1)                                             # IoPriorityHintLow
        ctypes.windll.kernel32.SetFileInformationByHandle(ctypes.c_void_p(msvcrt.get_osfhandle(handle.fileno())),
                                                          12, ctypes.byref(hint), ctypes.sizeof(hint))
    except (OSError, AttributeError, ValueError):
        pass


class ReadAheadGate:
    """Holds the launcher's read-ahead while the staged engine reads a network from the drive (its net_load event
    until its net_ready or net_drop, READ_AHEAD_HOLD_S at most), so that the read the GPU thread waits for has the
    drive to itself: on a cold start (a slow QLC drive) the read-ahead ran 61 s at ~38 MB/s beside the first
    batch's own loads. The read-ahead looks at it before each piece it reads."""

    def __init__(self) -> None:
        self._open = threading.Event()
        self._open.set()
        self._held_at = 0.0

    def on_event(self, event: Any) -> None:
        kind = getattr(event, "kind", "")
        if kind == "net_load" and not (getattr(event, "detail", None) or {}).get("bytes_read"):
            self._held_at = time.monotonic()
            self._open.clear()
        elif kind in ("net_ready", "net_drop", "prop_failed", "part_end"):
            self._open.set()

    def release(self) -> None:
        self._open.set()

    def wait(self) -> None:
        if not self._open.is_set():
            self._open.wait(max(0.0, READ_AHEAD_HOLD_S - (time.monotonic() - self._held_at)))


READ_AHEAD_GATE = ReadAheadGate()
READ_AHEAD_HOLD_S = 30.0            # a network's read holds the read-ahead this long at most


def _read_through(path: Path, buffer: bytearray) -> int:
    """Bytes of `path` read once, at a low I/O priority, never while the engine reads a network (READ_AHEAD_GATE);
    0 when it cannot be read."""
    read = 0
    try:
        with open(path, "rb", buffering=0) as handle:
            _low_io_priority(handle)
            while True:
                READ_AHEAD_GATE.wait()
                count = handle.readinto(buffer)
                if not count:
                    break
                read += count
    except OSError:
        pass
    return read


def read_ahead_libraries(networks: list[list[Path]] | None = None) -> None:
    """On a thread of its own, from the start of the process: the first networks of a batch (`networks`:
    first_networks' lists), the GPU libraries of the READ_AHEAD globs under the venv's site-packages and the Triton cache
    (TRITON_CACHE_DIR), each file read once in the order a batch needs them, the cache's small files by
    READ_AHEAD_SMALL_THREADS threads, so that the system's file cache holds them when the first batch loads them. A
    network's file is left out while RAM is short for it; a file that cannot be read is skipped."""
    if os.environ.get("SPELLCAST_READ_AHEAD_LIBS", "").strip().lower() in ("0", "off", "false", "no"):
        return          # developers: the cold first batch as it was, for measurements

    def globbed(patterns: tuple[str, ...]) -> list[Path]:
        import glob

        if os.name == "nt":
            sites = [Path(sys.prefix) / "Lib" / "site-packages"]
        else:       # the venv's own and those it sees (a pod's venv may use the base Python's torch)
            sites = list(dict.fromkeys(Path(p) for p in sys.path if p.endswith("site-packages")))
        return [Path(found) for site in sites for pattern in patterns
                for found in glob.glob(str(site / pattern), recursive=True)]

    def roomy(path: Path) -> bool:
        try:
            import psutil

            return psutil.virtual_memory().available >= path.stat().st_size + READ_AHEAD_SPARE_BYTES
        except Exception:       # noqa: BLE001 - no psutil, a file gone: read it as before
            return True

    def run() -> None:
        linux = os.name != "nt"
        began, read = time.perf_counter(), 0
        buffer = bytearray(8 << 20)
        rembg, moge, dino = (networks or []) + [[]] * (3 - len(networks or []))
        steps = [rembg, READ_AHEAD_FIRST_LINUX if linux else READ_AHEAD_FIRST, moge, dino,
                 READ_AHEAD_THEN_LINUX if linux else READ_AHEAD_THEN]
        for step in steps:
            paths = globbed(step) if step and isinstance(step[0], str) else [path for path in step if roomy(path)]
            for path in paths:
                read += _read_through(path, buffer)
        cache = os.environ.get("TRITON_CACHE_DIR")
        if cache and Path(cache).is_dir():
            small = [found for found in sorted(Path(cache).rglob("*"))[:READ_AHEAD_CACHE_FILES] if found.is_file()]
            buffers = threading.local()

            def one(path: Path) -> int:
                if not hasattr(buffers, "data"):
                    buffers.data = bytearray(1 << 20)
                return _read_through(path, buffers.data)

            with concurrent.futures.ThreadPoolExecutor(READ_AHEAD_SMALL_THREADS,
                                                       thread_name_prefix="spellcast-read-ahead") as pool:
                read += sum(pool.map(one, small, chunksize=64))
        for path in globbed(READ_AHEAD_LAST_LINUX if linux else READ_AHEAD_LAST):
            read += _read_through(path, buffer)
        seconds = time.perf_counter() - began
        if seconds >= 1.0:
            print(f"[spellcast] Read {read / 2**20:,.0f} MB of GPU libraries, compiled kernels and the first AI "
                  f"models into the file cache in {seconds:.1f} s, ahead of the first 3D model.", flush=True)

    threading.Thread(target=run, name="spellcast-read-ahead-libraries", daemon=True).start()


def exit_at_once_when_idle(idle: Callable[[], bool]) -> None:
    """The hub's stop (CTRL_BREAK) ends an idle Pixal3D at once: the process is terminated as the hub's kill would
    terminate it, without the seconds Windows' own exit spends letting go of the CUDA context and the page-locked
    buffers (hub logs: 1.3 s a stop at the median, 5.5 s at most, and 8 of 41 stops killed after the hub's 5 s), on
    the path to the next tool's start. Idle means nothing on the GPU worker and no batch: every GLB is written and
    nothing is left to save. A stop while it works ends it as before. Windows only (Linux pods stop by SIGTERM).
    SPELLCAST_FAST_EXIT=off keeps Windows' own exit (developers, for measurements)."""
    if os.name != "nt" or not hasattr(signal, "SIGBREAK"):
        return
    if os.environ.get("SPELLCAST_FAST_EXIT", "").strip().lower() in ("0", "off", "false", "no"):
        return
    previous = signal.getsignal(signal.SIGBREAK)

    def on_break(signum: int, frame: Any) -> None:
        try:
            quiet = idle()
        except Exception:       # noqa: BLE001 - when it cannot be told: as before
            quiet = False
        if quiet:
            print("[spellcast] Pixal3D stops (it was idle).", flush=True)
            for stream in (sys.stdout, sys.stderr):
                with contextlib.suppress(Exception):
                    stream.flush()
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.TerminateProcess(ctypes.c_void_p(kernel32.GetCurrentProcess()), 0)
        if callable(previous):
            previous(signum, frame)
            return
        signal.signal(signal.SIGBREAK, signal.SIG_DFL)      # Windows' own exit, as without this handler
        signal.raise_signal(signal.SIGBREAK)

    signal.signal(signal.SIGBREAK, on_break)


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).resolve()
    if not (repo / "app.py").is_file():
        print(f"[spellcast] Pixal3D was not found in {repo}. Close Spellcast3D and double-click "
              f"run-spellcast3d.bat again.", flush=True)
        return 2
    low_vram = args.low_vram == "on"
    models = models_folder(repo)
    if not models.is_dir():   # e.g. models/pixal3d was renamed: the junction points nowhere
        print(f"[spellcast] Pixal3D's AI models folder is missing ({models}). " + MODELS_MISSING, flush=True)
        return 2
    torch_home = configure_env(repo, low_vram, args.attn_backend, args.sage, args.int8, args.coarse_exact)
    read_ahead_libraries(first_networks(models, args.first_job))     # beside the imports below
    print(f"[spellcast] Starting Pixal3D from {repo} on {args.host}:{args.port} (attention {args.attn_backend}). "
          "Its models are read from disk when a 3D model needs them." + fp16acc_note(), flush=True)
    os.chdir(repo)   # app.py mounts StaticFiles("assets") relative to the working directory
    sys.path.insert(0, str(repo))
    try:
        open_non_ascii_paths_in_opencv()   # after the chdir: paths inside the repo are passed relative
        load_naf_locally(torch_home)
        import app as px   # module import only: app.py's __main__ block never runs

        clear_old_tmp(Path(px.TMP_DIR))
        allow_own_file_urls(args.host, args.port,
                            (Path(px.TMP_DIR), Path(os.environ.get("GRADIO_TEMP_DIR", px.TMP_DIR))))
        if is_loopback(args.host):
            px.app.add_middleware(LoopbackGuard)   # every HTTP / WebSocket request: refuses DNS rebinding
        quiet_expandable_segments()
        shared = share_dinov3()
        drop_inner_shells()
        exact_hole_centres()
        gpu = GpuWorker()
        notes = RunNotes()
        defaults = ui_defaults(px)
        projects_root = Path(args.projects_root).resolve() if args.projects_root else None
        modes = pixal3d_mv.ModelModes(px, models, lambda p: prepare_pipeline(px, p, notes))
        # The two paths do not fit in memory together: each makes the other let go before it runs.
        reference = Reference(px, modes, notes, FIRST_JOBS[args.first_job], shared, release=lambda: runner.release())
        runner = BatchRunner(px, gpu, BatchHooks(px, notes), notes, defaults, models, projects_root,
                             unload=reference.unload)
        gpu.run(runner.build)   # no weights are read and the card is not touched: that waits for the first job
        route_progress(px, runner.batches)
        serialise_ui_calls(px, gpu, before=reference.ensure_single_view)
        add_hub_routes(px, gpu, defaults, modes, notes, projects_root, reference, runner)
    except Exception as exc:
        traceback.print_exc()
        print("[spellcast] " + startup_hint(exc, repo), flush=True)
        return 1
    print("[spellcast] Pixal3D is ready.", flush=True)
    warm_imports()
    exit_at_once_when_idle(lambda: gpu.idle() and runner.batches.running() is None)
    try:
        px.app.launch(server_name=args.host, server_port=args.port, share=False,
                      show_error=True, inbrowser=False)
    except OSError as exc:   # gradio: "Cannot find empty port in range: P-P"
        print(f"[spellcast] Pixal3D could not open port {args.port} ({exc}). Another program (or an "
              "older Pixal3D that is still closing) uses it: wait a moment and start Pixal3D again, "
              "or change the Pixal3D port in Settings.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
