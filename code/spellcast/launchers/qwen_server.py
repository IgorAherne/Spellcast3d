"""The built-in image generator's tool server for Spellcast3D: Qwen-Image 2.1's two models and FLUX.3 Image Klein.
Runs INSIDE the qwen venv (Python 3.11: fastapi, uvicorn, pillow; no torch).

    python qwen_server.py --sdcpp <code/venvs/qwen/sdcpp> --models <models/qwen-image> --port P [--host 127.0.0.1]
                          [--flux3-models <models/flux3-klein>] [--flux3-engine auto|on|off]
                          [--projects-root <projects>] [--quality Q4_K_M] [--placement auto|gpu|offload]
                          [--max-vram GiB] [--card-gb GB] [--pin auto|on|off] [--prefix-cache auto|f16|q8_0]
                          [--prefix-keep auto|on|off] [--albedo <models/qwen-image/albedo>]
                          [--attention auto|flash|sage] [--eager-load on|off] [--verbose]

Qwen-Image 2.1 (Qwen; research licence for the weights, the pictures are the artist's) makes the Images stage's
pictures from a description, optionally with up to 10 example pictures, and with a transparent background when
asked. Two models of it (QWEN_MODELS), the same architecture with their own denoiser weights and everything else
shared (the VAE, the Qwen3-VL text encoder, its vision part): Qwen-Image-2.1-Turbo ("turbo", the default for every
card, the one the installer downloads) with its own FIXED schedule of TURBO_STEPS steps (TURBO_SIGMAS) and no
First-Block Cache, and the full Qwen-Image 2.1 ("full": STEPS steps by default, the First-Block Cache tick), an
optional download from the Images page that may be missing. A third model is no Qwen-Image: FLUX.3 Image Klein
("flux3": Black Forest Labs, the FLUX Kommunity License; also an optional download, into --flux3-models) with its own
files - its denoiser, FLUX.2's VAE, the Qwen3-VL-4B text encoder, no vision part (its example pictures go through the
VAE) - and its own recipe (sampling: FLUX3_STEPS steps, no guidance, euler, the engine's own schedule; opaque pictures,
none of Qwen's RGBA wording, model args, First-Block Cache or prefix-cache budget), and only with an engine build that
can draw it (flux3_engine_choice: the ONE gate, install/sdcpp.py ENGINE_HAS_FLUX3, with the pinned build). One
stable-diffusion.cpp context holds one of the three (MODELS); a request for another frees it and makes a new one
(QwenEngine.use_model: never two at once - 8 GB cards, 16 GB of RAM). The process starts with the model --first-model
names (the one the artist has picked: the hub's early start) when its files are there, else with the Turbo, or with
the full model on an install that has only that one (Qwen-Image's own files are the process's must). It runs in
stable-diffusion.cpp (MIT) - Spellcast3D's own build of it, pinned (qwen_sdcpp.PINNED_TAG; the installer unpacks it:
install/sdcpp.py) -, called in this process through ctypes (qwen_sdcpp.py): the models are
read once per stage, there is real per-step progress, and a picture stops within one step when cancelled (sd-server,
the alternative, cannot stop a picture it has started). Routes for the hub (adapters/qwen.py):

    GET  /spellcast/health     -> {"ready", "loading", "gpu", "vram_mb", "placement", "quality", "references",
                                   "sheet_side" (the largest square picture Omni-Texture may ask for: sheet_side),
                                   "sdcpp", "load_seconds", "model" (the one loaded: turbo | full | flux3), "models"
                                   ({turbo, full, flux3: it can be drawn now - its files are there, and for flux3 the
                                   engine draws it}), "flux3_engine" (the engine draws FLUX.3), "albedo" (the albedo
                                   add-on is there and the engine build runs it)}; ready only once the models are
                                   loaded, false again once CUDA broke
    GET  /spellcast/progress   -> {"phase", "step", "steps", "fraction", "message", "eta"}; phase: idle | loading |
                                   references | prompt | drawing | finishing | saving; eta in seconds or null;
                                   plus "preview": {"seq", "step", "image"} while there is a preview picture
    GET  /spellcast/preview    -> the picture being drawn as it looks so far, a small PNG (its number and step in the
                                   X-Preview-Seq / X-Preview-Step headers); 404 when there is none
    POST /spellcast/generate   {prompt, negative_prompt, width, height, model, steps, cfg_scale, sampler, seed,
                                batch_count, ref_images: [absolute paths, <= 10], ref_sides (GenerateRequest),
                                transparent, first_block_cache, out_dir (absolute), preview}
                               -> {"images": [absolute .png paths], "seeds", "seconds", "warning", "timings"}
                               errors: {"detail": <for an artist>, "kind": input|oom|cancelled|busy|restart|error}
                               (input, 400: also the asked model's files missing, or FLUX.3 with an engine that
                               cannot draw it);
                               "restart": an error left CUDA unusable, so the process stops being ready and exits a
                               second later (the hub starts it again); "busy": a request the hub gave up on still runs
    POST /spellcast/albedo     {picture, out (absolute .png), model} -> {"image", "seconds"}: the picture with its
                               lighting and shading taken off, by our albedo add-on (--albedo: a LoRA, a decoder and
                               a fixed prompt, one pass of the loaded Qwen-Image 2.1 model, a picture larger than a
                               view's at 3/4 of its sides: lighting_off); health says "albedo"
                               when the add-on is there and the engine build runs it; errors as for generate, never
                               "cancelled": the pass runs to its end (seconds) and reports no progress
    POST /spellcast/warm_up    {albedo} -> {"queued"}: once per process, before any request, a small picture and its
                               albedo on the GPU thread, so an Omni-Texture job's first view does not pay the first use
                               (the hub asks when it started this process early for Omni-Texture); a request that
                               comes meanwhile waits for it
    POST /spellcast/cancel     {} stop the running picture within one step (kind "cancelled")
    POST /spellcast/shutdown   {} graceful stop, as CTRL_BREAK
Paths (out_dir, ref_images, picture, out) must lie inside --projects-root when it is given.

The settings: Qwen's own way of sampling Qwen-Image 2.1, WITHOUT guidance - CFG 1.0: a step is one pass through the
model, and there is no negative prompt (at a CFG of exactly 1.0 sd.cpp skips the unconditional pass and never encodes
one) - with euler, STEPS steps (see STEPS), 1024x1024 (sides multiples of 32) and fused attention in the
diffusion model (SageAttention, or flash attention: see ATTENTIONS), as the request defaults. The adapter sends its
own, always without guidance. cfg_scale and negative_prompt are still taken, for the bench's comparisons of recipes
(spellcast/profiling/qwen_bench.py): any other CFG is guidance, two passes per step, and only then is the negative
prompt read (with CFG 1.0 one is simply unused).
Everything else only makes it faster without changing the recipe:
- the models stay loaded, and the variations of one request are one generate_image call (batch_count): the prompt is
  encoded once for all of them; the conditioning cache (CONDITIONING_CACHE entries) also skips the text encoder when
  the same prompt comes again with another seed;
- weight placement (choose_placement, PLACEMENTS): the weights live in RAM as the model files' mapping, within a
  budget of the free memory at start minus ~1 GiB (sd.cpp's documented "biggest model my card can host":
  params_backend *=cpu, max_vram -1, plus mmap). sd.cpp copies each model's weights to the card for the phase that
  uses it and frees them after it (seen in its log: "staged compute params" / "released compute params"): the text
  encoder for the prompt (skipped when the conditioning cache has it), the diffusion model once per request (1-2 s
  from the mapping), the VAE for the decode. The model files are read ahead in the background while the engine
  loads (read_ahead), so every read and copy is made from Windows' file cache, not the disk;
- the diffusion weights in page-locked RAM (--pin, set_pinning; nothing to do with the pinned BUILD above): a
  stable-diffusion.cpp build that knows SD_PIN_PARAMS - Spellcast3D's own engine build, the one the installer
  unpacks; the official release ignores the variable - keeps them there instead of in the file's mapping, so their
  copy to the card takes ~0.3 s every time (the mapping: 1-2 s, and up to ~60 s once Windows has dropped its pages),
  for RAM that Windows cannot take back while Qwen-Image is loaded (Q4_K_M: 4.65 GB; commit peak ~14.9 GB instead of
  ~10.9 GB). auto: on PCs with 24 GB of RAM or more;
- the prefix KV cache (the prompt's and the example pictures' keys and values between steps), one per condition:
  sd.cpp's "auto" on every card, which is f16 under flash attention (--prefix-cache q8_0: about half the memory, at
  8 bits). Several example pictures are made smaller where the cache would not fit at their full size
  (reference_pixels, by the request's number of conditions); running short anyway, sd.cpp drops the cache for that
  picture and retries (much slower, and the picture can differ slightly): the answer's warning says so;
- the VAE decode with direct convolutions, in one piece, where its usual im2col form would not fit in one piece but
  that does (VAE_DIRECT_MB_PER_MP, vae_direct: 8 GB cards; ~2.2 s for 1024x1024 instead of ~3.2 s in tiles), and
  always on a GTX 10xx; in 512-pixel tiles where neither fits (needs_vae_tiling; sd.cpp would otherwise run out of
  memory first and retry in tiles of its own choosing - 256 pixels before the fork's fix, a grid of seams with this
  16x VAE);
- the diffusion model's attention through SageAttention (ATTENTIONS): INT8 queries and keys, FP16 values - not bit
  for bit (like the engine build's int8 matmul since 2026-10-21, whose min terms are summed on the tensor cores); the
  pictures are visually the same;
- no prompt rewriter (Qwen's are 9B models).
Preview pictures (a request with "preview": true; off by default): after every step stable-diffusion.cpp turns its
estimate of the finished picture, the latent it already holds in RAM, into a small picture with a fixed linear
projection (PREVIEW_PROJ: no model, no decode, nothing on the card; Qwen-Image: 1/16 of the picture's sides, 64x64
for 1024x1024; FLUX.3: 1/8, 128x128) and hands it to QwenEngine._on_preview, which keeps the newest as a PNG in
memory (Progress.preview) for
GET /spellcast/preview. Well under a millisecond per step; nothing is written to disk. A request without it leaves
sd.cpp without a preview callback.
Transparent pictures: the prompt is wrapped exactly as Qwen documents ("This is an RGBA image with transparency.
<prompt>. The image has alpha channel and the background is transparent."); the model always works in 4 channels,
and a picture is saved as an RGBA PNG when transparency was asked for and it has see-through pixels, else as RGB:
the model's alpha on an opaque picture is 253-255 rather than exactly 255, and Pixal3D takes any alpha below 255 as
the picture's own cut-out (it then skips its background removal). Example pictures go in as RGBA (their
transparency is kept; RGB ones are opaque). FLUX.3 Image Klein has no alpha channel (draws_alpha): its prompt is never
wrapped, whatever "transparent" says, and its pictures are saved as RGB, so Pixal3D removes their background (as for
Forge Neo's pictures).

Windows: stable-diffusion.dll and its CUDA backend load from the sdcpp folder (it becomes the current folder; see
qwen_sdcpp.py). A CUDA error inside ggml ends the process at once (ggml aborts); the C runtime is told not to show
an error dialog for that, so the hub sees the exit and starts the tool again.

Only the stdlib, fastapi, pydantic and three stdlib-only modules of the hub (spellcast/config.py for the model sizes,
spellcast/install/models.py for the model files' names, spellcast/install/sdcpp.py for the engine-support gate) at
module level: the hub's tests import this file without Pillow or the DLL.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import ctypes
import faulthandler
import hashlib
import io
import itertools
import json
import math
import os
import random
import re
import signal
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, MutableMapping, NamedTuple

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

import qwen_sdcpp as sdcpp                                 # launchers/ (this script's folder)
from _loopback_guard import LoopbackGuard, is_loopback

# ---- the model files (models/qwen-image, flat) ---------------------------------------------------------------------
# Named in ONE place, the installer's manifest (spellcast/install/models.py: stdlib only, so it imports here too); the
# adapter passes --models and the configured --quality. code/ goes at the END of sys.path: the venv's own packages win.
_CODE = str(Path(__file__).resolve().parents[2])
if _CODE not in sys.path:
    sys.path.append(_CODE)
from spellcast import config  # noqa: E402
from spellcast.install import models as manifest  # noqa: E402
from spellcast.install import sdcpp as engine_pin  # noqa: E402

QUALITIES: tuple[str, ...] = config.QUALITIES             # = the manifest's denoiser sizes
DEFAULT_QUALITY = config.DEFAULT_QUALITY
VAE_FILE = manifest.QWEN_VAE
LLM_FILE = manifest.QWEN_TEXT_ENCODER
VISION_FILE = manifest.QWEN_VISION                        # only for example pictures
FLUX3_VAE_FILE = manifest.FLUX3_VAE
FLUX3_LLM_FILE = manifest.FLUX3_TEXT_ENCODER
FLUX3_FOLDER = "flux3-klein"                              # --flux3-models' default: beside --models (paths.MODELS_FLUX3)
# Qwen-Image 2.1's two models (install/models.py QWEN_MODELS), the default first; then FLUX.3 Image Klein. How the log
# and the messages name them.
QWEN_MODELS: tuple[str, ...] = tuple(manifest.QWEN_MODELS)
MODELS: tuple[str, ...] = (*QWEN_MODELS, "flux3")
DEFAULT_MODEL = MODELS[0]
MODEL_NAMES = {"turbo": "Qwen-Image 2.1 Turbo", "full": "the full Qwen-Image 2.1", "flux3": "FLUX.3 Image Klein"}


def diffusion_file(quality: str, model: str = DEFAULT_MODEL) -> str:
    """The denoiser's file name of `model` in size `quality`."""
    if model == "flux3":
        return manifest.flux3_denoiser(quality)[0]
    return manifest.qwen_denoiser(model, quality)[0]


def draws_alpha(model: str) -> bool:
    """Whether `model` draws pictures with an alpha channel (a transparent background when asked): Qwen-Image's two,
    not FLUX.3."""
    return model in QWEN_MODELS


def model_name(model: str, start: bool = False) -> str:
    """MODEL_NAMES[model], with a capital at the start of a sentence."""
    name = MODEL_NAMES.get(model, model)
    return name[:1].upper() + name[1:] if start else name


def _ram_words() -> str:
    ram = free_ram_mb()
    return f"{ram:,} MB" if ram is not None else "unknown"


# ---- the recipe (Qwen's own: no guidance) and the limits ----------------------------------------------------------
# Qwen documents Qwen-Image 2.1 as sampled WITHOUT guidance: diffusers' pipeline, Qwen's reference, defaults to 40
# steps and a guidance of 1.0 because the model is meant to be sampled that way
# (https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21); ComfyUI's official template samples with 25
# steps, cfg 1, euler (https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1). Until October 2026 the defaults
# here were Unsloth's stable-diffusion.cpp example recipe, "the former recipe" below: 20 steps at CFG 6 with a
# negative prompt, 40 model passes per picture.
# STEPS: the launcher's ONE place for the default number of steps (the adapter and the page have theirs). 25 is
# ComfyUI's official template value, checked against Qwen's own reference of 40 on 2026-10-03 (RTX 3060 12 GB, 4
# prompts at the same seed each, Spellcast3D's own engine build): 25 and 40 steps without guidance were both sharp
# and natural and close to each other, where the former recipe gave over-saturated, soft pictures. So 25 stays the
# default, and 40 stays available on the page's Steps slider (its seconds: below).
STEPS = 25
CFG_SCALE = 1.0                    # exactly 1.0 = no guidance: one condition per step (condition_count)
SAMPLER = "euler"
# Qwen-Image-2.1-Turbo (the "turbo" model) is distilled for ONE fixed schedule: these 8 steps' sigmas, euler, no
# guidance - its model card's recipe ("Setting num_inference_steps alone does not override it"), upstream
# stable-diffusion.cpp's `--sigmas qwen-image-2.1-turbo`. A Turbo request is drawn so whatever its steps, cfg_scale,
# sampler and first_block_cache say (sampling): stable-diffusion.cpp takes the sigmas as sample_params.custom_sigmas
# (len - 1 steps, every one of them used for a new picture; sample_steps says as many: src/pipeline/request.cpp). The
# First-Block Cache is off for it: its eight steps have none to spare (SD_QWEN_FIRST_BLOCK_CACHE=0; the engine itself
# has no guard for short schedules). Checked on 2026-10-25 with sd-cli on the RTX 3060 (Spellcast3D's engine,
# unchanged): 1.8 s a step, the transparent background as with the full model.
TURBO_SIGMAS: tuple[float, ...] = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0)
TURBO_STEPS = len(TURBO_SIGMAS) - 1
# FLUX.3 Image Klein (the "flux3" model) is distilled for FLUX3_STEPS steps at CFG 1.0 with euler (upstream
# stable-diffusion.cpp's docs/flux3.md, Black Forest Labs' recipe): no guidance, so no negative prompt. The schedule is
# the engine's own for FLUX.3 - its default discrete scheduler with FLUX.3's time shift (6.93) and its flow-map
# sampling -, so no custom sigmas and no flow shift are given. A FLUX.3 request is drawn so whatever its steps,
# cfg_scale, sampler and first_block_cache say (sampling), as the Turbo's.
FLUX3_STEPS = 16
# Its attention: flash attention unless --attention sage asks for the sage op explicitly (developers). The engine's
# SageAttention (INT8 queries and keys) was tuned and judged in pictures for Qwen-Image 2.1 only (ATTENTIONS). On a
# GTX 10xx "flash" is the engine's INT8 dp4a kernel for every model, as are its int8 VAE convolutions: judged for
# FLUX.3 under the RTX 3060's emulation of a GTX 10xx (2026-10-26: 32 dB from the F32 path, where the RTX 3060's own
# path is 30 dB from it; the pictures look the same).
FLUX3_ATTENTION_AUTO = "flash"
SIZE = 1024
SIDE_MIN, SIDE_MAX, SIDE_STEP = 256, 2432, 32      # 2432: the long side of a 2K portrait (1664 x 2432)
MAX_REFERENCES = 10
MAX_BATCH = 16
# The First-Block Cache's skip share when a request has it on (GenerateRequest.first_block_cache): the engine then
# reuses its middle layers' work on this share of the sampling steps (never the first third nor the last step),
# ~35 % faster (a GTX 1070 picture 141 -> 91 s) at a small loss of fine detail (ARCHITECTURE "qwen", the fork's
# docs/qwen_image_2.1.md). One number for every picture: the page offers a tick, not a slider.
FIRST_BLOCK_CACHE_SKIP = "0.5"
# The engine's switch for GenerateRequest.keep_prefix (Spellcast3D's build from release spellcast-perf-<round P> on):
# "1" frees the run's prefix at its end instead of keeping it for the next request.
PREFIX_DROP_ENV = "SD_QWEN_PREFIX_DROP"
# A request that leaves the tick to the card (first_block_cache None): on below this compute capability - the
# GTX 10xx (6.x), where a picture takes ~2.5 minutes without it (~1.5 with it) - and off on newer cards, which
# draw the full computation in 20-45 s. images.js shows the same default (FIRST_BLOCK_CACHE_AUTO_BELOW there).
FIRST_BLOCK_CACHE_AUTO_BELOW = (7, 0)


def first_block_cache_on(asked: bool | None, capability: tuple[int, int]) -> bool:
    """Whether a picture of the full model is drawn with the First-Block Cache: the artist's tick, or by the card when
    it is None (the Turbo never: sampling)."""
    return asked if asked is not None else tuple(capability) < FIRST_BLOCK_CACHE_AUTO_BELOW


def sampling(model: str, steps: int, cfg_scale: float, sampler: str, first_block_cache: bool | None,
             capability: tuple[int, int]) -> dict[str, Any]:
    """How a request of `model` is drawn -> {"steps", "sigmas" (the schedule, or None: the sampler's own),
    "cfg_scale", "sampler", "first_block_cache" (on?)}. The Turbo: its fixed TURBO_SIGMAS, euler, no guidance, no
    First-Block Cache, whatever the request said; FLUX.3: FLUX3_STEPS on the engine's own schedule, euler, no
    guidance, no First-Block Cache, whatever the request said; the full model: as asked, the cache by
    first_block_cache_on."""
    if model == "turbo":
        return {"steps": TURBO_STEPS, "sigmas": TURBO_SIGMAS, "cfg_scale": CFG_SCALE, "sampler": SAMPLER,
                "first_block_cache": False}
    if model == "flux3":
        return {"steps": FLUX3_STEPS, "sigmas": None, "cfg_scale": CFG_SCALE, "sampler": SAMPLER,
                "first_block_cache": False}
    return {"steps": steps, "sigmas": None, "cfg_scale": cfg_scale, "sampler": sampler,
            "first_block_cache": first_block_cache_on(first_block_cache, capability)}
MAX_SEED = 2 ** 32 - 1             # Stable Diffusion's usual seed range (Forge Neo's too)
REFERENCE_MAX_SIDE = 2048          # larger example pictures are shrunk first (sd.cpp resizes them to ~1 MP anyway)
REFERENCE_STEP = 32                # an example picture's VAE copy has sides of multiples of this (Qwen-Image's VAE)
REFERENCE_MIN_SIDE = 256           # the smallest copy a request may ask for (ref_sides)
REFERENCE_CACHE_MB = 256           # example pictures kept decoded between requests (Pictures)
# The read-ahead (read_ahead). The engine reads its weights many tensors at once, all over a file, and a drive serves
# that unevenly: one NVMe gave it 580 MB/s on some starts and 95 MB/s on others (loads of 20 or 90 s), while
# sequential reads kept 0.5-1.6 GB/s. So the files are read sequentially just ahead of the engine, which then finds
# them in the file cache. It starts once the engine has taken its page-locked memory (PIN_MARK, a line of its log):
# earlier, that memory is carved out of the cache just filled, and the engine reads the disk all the same.
PIN_MARK = "in page-locked host memory"
READ_CHUNK = 16 << 20
# The components SD_PIN_PARAMS names and the files (model_sets' keys) they are read from, in the engine's own order
# (its model_loader's: the denoiser, the text encoder, its vision part, the VAE).
PINNED_FILES = (("diffusion", "diffusion"), ("text_encoder", "llm"), ("text_encoder", "vision"))
# The warm-up (QwenEngine.warm_omnitexture): a picture this small, with two example pictures, a start picture and a
# mask, and the albedo of one, so the first use of every part of an Omni-Texture view is paid before its first view
WARM_SIDE = 256
WARM_PROMPT = "A plain grey test card."
FLUX3_REFERENCES_FULL_SIZE = 2    # FLUX.3 example pictures at the engine's full size; more share that many's pixels
OPAQUE_ALPHA = 250                 # the model's alpha on an opaque picture is 253-255, not exactly 255
# A picture has a see-through part when more than CUTOUT_SHARE of its pixels are below SEE_THROUGH_ALPHA. Ten
# 1024 x 1024 pictures of 2026-11-21 (no transparency asked): the opaque ones have 0-14 such pixels of a million, the
# ones the model cut out all the same 72-80 % of them.
SEE_THROUGH_ALPHA = 128
CUTOUT_SHARE = 0.001
CONDITIONING_CACHE = 8             # prompts (with their example pictures) whose encoding is kept
RGBA_PREFIX = "This is an RGBA image with transparency."
RGBA_SUFFIX = "The image has alpha channel and the background is transparent."

# ---- weight placement and video memory -----------------------------------------------------------------------------
# "offload" (the default): the weights' home is RAM - the model files' own mapping, so nothing is copied while loading
# and Windows can drop them under pressure - and sd.cpp copies a model's weights to the card for the phase that uses
# it, releasing them after that phase (even when everything would fit: its per-phase policy, not memory pressure), so
# the card holds only what the current phase needs, plus the prefix cache of the example pictures and the VAE decode.
# "gpu" keeps the weights on the card instead (sd.cpp's auto-fit). MEASURED WITH THE FORMER RECIPE (20 steps at CFG 6:
# 40 model passes, two conditions, a q8_0 prefix cache) on an RTX 3060 12 GB (sd.cpp 3f8527a, Q4_K_M, 1024x1024),
# offload / gpu: ready after 2 s (17 s with a private RAM copy) / 66 s; a picture 120 s / 131 s (the VAE decode no
# longer fit in one piece: tiled, 13 s instead of 3 s); with one example picture 160 s / 297 s (the prefix cache no
# longer fit). With sd.cpp's budget at 6 GiB (an 8 GB card) the weights stream in per step: a picture 126 s. The times
# until ready and the VAE decode's do not depend on the recipe.
# MEASURED WITHOUT GUIDANCE on 2026-10-03 (today's recipe: STEPS passes, one condition, an f16 prefix cache; the same
# card, Q4_K_M, 1024x1024, transparent background on; other programs used the card at times, hence the ranges):
# - Spellcast3D's own engine build (branch spellcast-perf of github.com/IgorAherne/stable-diffusion.cpp with
#   github.com/IgorAherne/ggml), its release spellcast-perf-36f92d4 (pinned until 2026-10-21): a picture 50.9-55.3 s
#   (drawing 49.2-53.7 s; in the profile a median step of 1.94 s with the GPU 99.7 % busy; the decode 1.5-1.6 s); 40
#   steps without guidance 80.6-82.6 s; the former recipe 82.8-85.3 s, ~80 s of it drawing. Its all-GPU build
#   (sm_61..sm_120) gave pictures bit-identical to the official release's on this card. Its release
#   spellcast-perf-a10772b (pinned since 2026-10-21: SageAttention, see ATTENTIONS, and the int8 matmul's min terms
#   on the tensor cores), measured that day through this bench path against spellcast-perf-36f92d4: a new description
#   41.1 s instead of 54.4 s, a re-roll 38.8 s instead of 52.1 s (drawing 37.1-37.4 s / 50.2-50.5 s; --attention
#   flash on the new build: 48.9 / 47.1 s); with one 1024x1024 example picture a step 1.64-1.65 s instead of
#   2.47-2.56 s (sd-cli with the launcher's settings);
# - the official release it is based on (master-929-3f8527a, what the installer pinned before): a picture 81.0 s for
#   a new description, 76.3 s for a re-roll (drawing 74.4 / 73.1 s, ~2.9-3.0 s per step; the decode 3.1 s). The
#   former recipe on it, the night before: 121.8 / 119.1 s;
# - memory (sd.cpp's verbose log, the own engine build): drawing took 4,005 MB of weights and a 746 MB compute buffer;
#   a text-only prompt's prefix cache was 56 tokens in f16. --card-gb 8 (an 8 GB card emulated): the picture was made,
#   its VAE decode tiled (3 x 3 tiles of 512 px, a 1,944 MB compute buffer; 7.1 s on a card other programs were
#   using). One example picture at 1024x1024: the picture was made on the 12 GB card;
# - the whole card's peak, read with nvidia-smi and so INCLUDING the other programs' ~1-2 GB: ~11.0-11.4 GB for text
#   to image with the decode in one piece, 7.2 GB in the 8 GB emulation, 11.6 GB with the example picture.
# NOT measured: timings on other cards, a real 8 GB card, PCs with 16 GB of RAM.
PLACEMENTS = ("auto", "gpu", "offload")
VAE_DECODE_MB_PER_MP = 8_400       # the VAE decode in one piece per million pixels (seen: 8,772 MB at 1024x1024)
# The same with direct convolutions (sd_ctx_params.vae_conv_direct; seen 3,889 MB at 1024x1024, the im2col form 7,777
# MB that day, RTX 3060, 2026-10-21): half the memory, ~0.4 s slower (2.23 s against 1.83 s at 1024x1024, both in one
# piece), faster than the im2col form in 512-pixel tiles (3.20 s, 1,944 MB); the pictures the same to 56 dB PSNR. It
# is the VAE's setting for the whole context, so the example pictures' VAE encode runs with direct convolutions too:
# 1.43 s instead of 1.17 s for a 924x1342 picture (sd-cli, the same day), less than the decode saves.
VAE_DIRECT_MB_PER_MP = 3_800
# The direct convolutions take a padded copy of their input from the engine's memory pool, beside the buffers the
# engine's own check counts: this many bytes a pixel at the decoder's widest layer (1.3 GB for 1536x1536), where
# that check allows ENGINE_SCRATCH_MB. A decode it lets through can then fail in the pool, which ends the launcher
# (an 11 GB card, a 1536x1536 sheet): needs_vae_tiling counts the rest.
VAE_DIRECT_SCRATCH = 576
ENGINE_SCRATCH_MB = 512
# The tiles of a decode that does not fit in one piece, pixels a side: the largest whose own decode does fit
# (vae_tile). Fewer, larger tiles are faster and no worse: a 2048x2048 decode in 1024-pixel tiles took 8.8 s against
# 12.3 s in 512-pixel ones on an RTX 3060 (15.7 against 23.0 s on a GTX 1080 Ti), the pictures 0.6 of 255 apart and
# no seam at either's tile borders. (sd.cpp's own retry used 256 before the fork's fix: a grid of seams.)
VAE_TILES = (1024, 512)
# The prefix KV cache keeps the prompt's and the example pictures' keys and values of all 32 layers between steps,
# once per CONDITION: 32 x 2 x 4,096 values per token of the prefix. A request without guidance (CFG exactly 1.0) has
# one condition, one with guidance two (the positive and the negative): condition_count. An example picture adds one
# token per 16x16 pixels of its VAE copy (4,096 at 1024x1024) plus the vision encoder's tokens.
# Its type is sd.cpp's "auto" on every card (PREFIX_CACHE): f16 under flash attention, which casts its keys and
# values to half precision anyway, so the cache holds exactly what it would be given without one. With guidance,
# cards of 16 GB and less used q8_0 for its half of the memory, f16 only above; without guidance there is one
# condition, so one f16 cache takes about what the two q8_0 ones took (0.5 against 0.544 MiB per token) and the
# 8-bit rounding of the keys and values is gone on every card. --prefix-cache q8_0 still asks for it.
# When the cache does not fit, sd.cpp computes the whole sequence at every step instead: three 1024x1024 example
# pictures then took 11 minutes for one picture on the RTX 3060 (the former recipe; q8_0 or f16 alike). So with
# several example pictures their VAE copies are made smaller, to what the card holds (reference_pixels).
PREFIX_CACHE = "auto"
# Whether the engine keeps the prefix cache (and the example pictures' VAE latents) from one picture to the next with
# the same prompt and example pictures - a re-roll, or the variations of one request -, so their first step costs what
# the other steps cost (the engine's model arg qwen_image_2_1_prefix_cache_keep; Spellcast3D's engine build of
# 2026-10-21 or newer, an older one ignores it). MEASURED (RTX 3060, sd-cli, one 1024x1024 example picture,
# SageAttention): the first step 4.4 s -> 1.9 s and the picture's encode 0.6 -> 0.1 s on every re-roll and variation;
# text only ~0.3 s. "on" also makes the first step's own attention take the keys in F16 under SageAttention, like
# every other step (the engine's "auto" keeps the cache under flash attention only, where that holds anyway): a
# one-time change of the pictures by rounding (52-55 dB PSNR with an example picture at 4 steps; judged on the
# six-prompt suite at 25 steps on 2026-10-21), after which a seed gives the same picture with or without a kept
# cache. The cache stays on the card between pictures (0.5 MiB per prefix token, ~2 GB with a 1024x1024 example)
# and the engine drops it when the VAE or the text encoder needs the room, when the prompt or the pictures change,
# and when it was not used.
PREFIX_KEEP = "on"
PREFIX_KEEPS = ("auto", "on", "off")
# MiB per token of the prefix and per condition. DERIVED from the cache's layout, not measured: f16 is 262,144 values
# of 2 bytes; q8_0 is half of the 0.544 the estimate used for the two conditions of the former recipe. (sd.cpp's logs
# of 2026-10-03 gave the caches in tokens, both f16: 56 for a text-only prompt, 4,155 with one 1024x1024 example
# picture. Their megabytes are still this derivation.)
PREFIX_MB_PER_TOKEN = {"q8_0": 0.272, "auto": 0.5, "f16": 0.5}
# What a request with example pictures needs beside its cache, as the engine works when the card is short: the
# weights streamed in blocks, the cache filled by a step over the whole sequence (every card's way; the newer cards'
# prefix-only pass needs less). MEASURED from sd.cpp's verbose log (the pinned engine, the Turbo Q4_K_M, one example
# picture, no guidance, sage and flash attention alike; an RTX 3060 at its own budget of 10,224 MB and held to
# 6,144 MB by --card-gb 8; reports/copy-size-2026-11-24):
# - that step's compute buffer: 2,507 MB for a 1536x1536 picture with a 1024x1024 copy, 3,128 MB for 2048x2048 with
#   608x608, 3,746 MB with 1024x1024 - 0.173 MB per token of the picture and 0.233 per token of the prefix (so too
#   2,025 MB for 1024x1024 with two 864x864 copies); every later step the picture's 0.173 alone;
# - the hidden state kept between blocks: 0.016 MB per token of both;
# - the cache was kept where this estimate left 195 MB of the budget (6,144 MB: 1536x1536 with a 1024x1024 copy)
#   and lost from 9 % over it (10,224 MB: 2048x2048 with 1536x1536) - the picture then takes 1.3 to 2 times as long.
# The vision tower reads every copy at its full size too, its attention at 64 bytes per pair of tokens: 1.2 GB for
# a 1024x1024 copy, 5.5 GB for 1536x1536, 17 GB for 2048x2048 (on a 12 GB card the request fails). That stays
# under what this estimate asks of a budget, so it is no limit of its own.
STEP_MB_PER_TOKEN = 0.19           # of the picture: a step's compute buffer and hidden state
FILL_MB_PER_TOKEN = 0.25           # of the prefix: what the step that fills the cache holds for it beside the cache
SEGMENT_MB = 1_100                 # the block of weights streamed in next to them (seen: 988-1,082 MB)
TEXT_TOKENS = 256                  # the prompt with Qwen's template (seen: 54 for a short one)
VISION_TOKENS = 64                 # what else an example picture adds to the prefix (seen: ~15)
MIN_REFERENCE_TOKENS = 256         # 256x256 pixels: the smallest VAE copy of an example picture
MEMORY_SAFETY = 0.85               # of the budget, for what the estimate misses
# A SMALLER CARD GIVES THE EXAMPLE PICTURES THE SAME COPIES AS FAR AS IT HOLDS THEM (an 8 GB
# card must paint as well as a 12 GB one): the copies are those of a budget of REFERENCE_BUDGET_MB (a 12 GB card's,
# where the stages' pictures were tuned) with MEMORY_SAFETY of it, or what the card's own budget holds without that
# margin where it is less. Cut by a cautious estimate on an 8 GB card, Omni-Texture's two pictures had 464 px a
# side instead of 864 and its painting sat 3 px off the model (0.958 on the model's outline instead of 0.978).
# MEASURED 2026-11-17 (the RTX 3060 held to 6,144 MB, a 1024x1024 sheet with two pictures, the copies set by hand;
# devtools/omnitexture-checks/sheet_trial.py): copies of 864 px cached, with the prefix pass and without (19.5 and
# 27 s; the painting byte-identical to the 12 GB card's), of 960 px with the pass too (22 s); of 1024 px no longer:
# the engine was 96 MB short and drew without the cache, 57 s.
# NOT measured: a real 8 GB card at this limit, two conditions (guidance) - where the cache does not fit after all
# the engine draws without it (slower, the same picture) and the answer's warning says so.
REFERENCE_BUDGET_MB = 10 * 1024

# ---- the attention -------------------------------------------------------------------------------------------------
# --attention auto (the default): the diffusion model's attention through the sage op of Spellcast3D's engine build
# (sd_ctx_params.sage_attn) - INT8 queries (per 16 rows) and keys (per 64 keys, less their mean), FP16 values, P.V
# summed in FP16 per 64 keys and then in FP32, its tuned kernel - on cards of compute capability 8.0 and newer
# (SAGE_MIN_CAPABILITY: the op's kernels are sm_80+; the RTX 30 series and newer). Older cards (RTX 20 / GTX 10
# series) keep flash attention, as before: the launcher must not ask the engine for sage attention there - the engine
# checks the op once at new_sd_ctx and REFUSES TO START when the card has no kernel, it does not fall back. A layer
# with a mask (the prompt's own, in the first step) always goes to flash attention. Only with the pinned engine build
# (an older one, e.g. an install not updated yet, would run upstream's slower sage kernel with an F32 prefix cache,
# twice the memory reference_pixels plans for): flash attention with any other build. --attention flash: flash
# attention always (the former path, bit for bit; for comparisons). --attention sage (developers): the sage op with
# any build that has it, e.g. a build under test that is not the pinned one (qwen_bench --sdcpp), still only on cards
# of SAGE_MIN_CAPABILITY or newer.
# MEASURED on 2026-10-21 (RTX 3060 12 GB, Q4_K_M, 1024x1024, 25 steps, transparent background on, the engine build of
# github.com/IgorAherne/stable-diffusion.cpp of that day, sd-cli): a step 1.97-2.00 s with flash attention, 1.61 s
# with the sage op; with one 1024x1024 example picture 2.47-2.56 s -> 1.78-1.83 s. Six prompts (four text-only, two
# with an example picture) against flash attention's pictures: PSNR 27-42 dB, LPIPS 0.004-0.034 - the same
# compositions, colours and sharpness, small details placed differently (the sampler amplifies any rounding change
# over its steps); two runs give the same bits. The prefix cache stays F16 and the step's working memory the same
# (741.5 MB at 1024x1024 without example pictures).
ATTENTIONS = ("auto", "flash", "sage")
SAGE_MIN_CAPABILITY = (8, 0)       # the sage op's kernels: sm_80 and newer (ggml-cuda/sage-attn.cu)

# ---- the diffusion weights in page-locked RAM ----------------------------------------------------------------------
# SD_PIN_PARAMS names the models ("diffusion", "text_encoder", "vae", "all") whose weights a stable-diffusion.cpp build
# that knows the variable - Spellcast3D's own engine build, which the installer pins (install/sdcpp.py); the official
# release never reads it - keeps in the GPU driver's page-locked RAM instead of the model file's mapping, when they
# live in RAM and are copied to the card for computing (the offload placement). Their copy to the card then takes
# ~0.3 s every time; from the mapping it takes 1-2 s, and up to ~60 s once Windows has dropped the mapping's pages.
# The price is RAM that Windows cannot take back while Qwen-Image is loaded: 4.65 GB with the Q4_K_M diffusion
# weights (a file of 4.2 GB = 4,005 MiB, the installer's manifest), a commit peak of ~14.9 GB instead of ~10.9 GB; a
# bigger model size locks as much more. (The 4.65 GB and the two commit peaks are the earlier measurement's: they
# were not measured again on 2026-10-03, and whether they are GB or thousands of MiB is still to be re-read. That day
# --pin auto set SD_PIN_PARAMS=diffusion on the 32 GB PC and the own engine build's log confirmed it: "Diffusion
# model params: 4005.00 MB in page-locked host memory".)
# Hence --pin auto asks for it only on PCs with 24 GB of RAM or more (the installed RAM, not what is free at the
# start). Windows counts the installed RAM less what the hardware keeps for itself (31,909 MB on a 32 GB PC; an
# integrated graphics chip takes more), so the bar is 22 GiB: a 24 GB PC passes, a 20 GB one does not.
PIN_ENV = "SD_PIN_PARAMS"
PIN_PARAMS = "diffusion"
PINS = ("auto", "on", "off")

# ---- the matmuls ---------------------------------------------------------------------------------------------------
# The engine has two kinds of kernels for the diffusion model's quantized products. mmq (the default): the weights as
# the file has them, the activations in 8 bits with a float scale per 32 values. int8: weights and activations
# requantized for one plain int8 GEMM (ggml-cuda/w8a8.cuh: on the tensor cores from the RTX 20 series on, through
# dp4a on the GTX 10xx; where the engine has no such kernel it stays on mmq). The int8 path is faster (a re-roll 10.1
# s instead of 13.0 on an RTX 4060, 15.8 instead of 25.0 on an RTX 2070 SUPER, 35.6 instead of 52.7 on a GTX 1080)
# and not the same picture (most of its error is the weights' requantization to one int8 scale per output
# channel): a picture moves about twice as far as under a change of rounding, and small features are redrawn (a
# lock's shackle, a roof's ridge, an eye). In blind pairs on those three cards three quarters showed no
# difference; in the rest its picture was judged the worse one 116 times and the better one 65 times, where a
# change of rounding came out level. Hence no card draws with it unless the artist asks: the Images panel's "Faster
# image generation" has the hub start this launcher with --matmul int8 (adapters/qwen.py matmul_arg).
# The engine reads the variable at its first product through its C runtime's getenv, so it goes into the environment
# BEFORE the DLL is loaded, like --pin; a variable already set is never changed.
MATMULS = ("mmq", "int8")
MATMUL_ENV = "GGML_CUDA_W8A8_KERNEL"
# Cards without BF16 tensor cores (below compute capability 8.0: GTX 10xx, RTX 20xx) run a BF16 LoRA's products
# through cuBLAS in F32. SD_LORA_WEIGHT_TYPE=q8_0 has the engine store a LoRA's matrices as Q8_0 for its int8 kernels,
# at half the memory (the albedo add-on 49-56 dB from the BF16 one's, as close to PyTorch's). Set before the DLL is
# loaded; a value already set is kept.
LORA_TYPE_ENV = "SD_LORA_WEIGHT_TYPE"
LORA_Q8_BELOW = (8, 0)
PIN_MIN_RAM_MB = 22 * 1024
# On PCs with 32 GB of RAM or more (30 GiB, as Windows counts it) --pin auto keeps the text encoder's weights there too
# (SD_PIN_PARAMS=diffusion,text_encoder): every new description copies them to the card before the drawing starts,
# and from page-locked RAM that took 0.44 s instead of 1.36 s from the file's mapping (RTX 3060, 4,418 MB, of which
# 4,084 MB are kept page-locked, 2026-10-21) - and instead of 1.9 s once Windows has dropped the file's pages, which
# a 32 GB PC with other programs open does between two descriptions (seen 2026-10-29): ~1-1.5 s less per picture of
# a new prompt, for ~4.1 GB more RAM that Windows cannot take back while Qwen-Image is loaded (8.75 GB with the
# diffusion weights; the bar was 48 GB until 2026-10-29).
PIN_TEXT_PARAMS = "diffusion,text_encoder"
PIN_TEXT_MIN_RAM_MB = 30 * 1024

# ---- the weights kept on the card between runs ---------------------------------------------------------------------
# With the offload placement sd.cpp copies a model's weights to the card for each run and frees them at its end (the
# diffusion weights for every picture: 0.3 s from page-locked RAM, 1-2 s from the file's mapping, a disk read once
# Windows dropped its pages; the text encoder's 4.4 GB for every new description). SD_KEEP_STAGED_PARAMS=1 makes
# Spellcast3D's engine build (since the release after spellcast-perf-52bc254; the official release ignores it) keep
# them there after the run instead, evicting the least recently used model only when another run's weights or working
# memory need the room. --keep-params auto asks for it where the budget holds the diffusion weights beside everything a
# picture needs after them - the decode in one piece with direct convolutions (vae_direct then takes them, whatever
# the usual decode would fit: the im2col form would evict the weights every picture) and the drawing's working memory:
# 12 GB cards. On 8 GB cards the decode needs the weights' room every picture, so there the engine frees them at the
# run's end as before (off); on: always (developers, for comparisons).
KEEP_ENV = "SD_KEEP_STAGED_PARAMS"
KEEPS = ("auto", "on", "off")
VAE_WEIGHTS_MB = 483               # the Wan VAE's weights on the card (seen: 482.81 MB)

MODELS_MISSING = ("Qwen-Image's model files are missing or incomplete (they belong in the models\\qwen-image folder "
                  "of Spellcast3D). Close Spellcast3D and start it again: it puts back what is missing.")
NO_VISION = ("Example pictures need Qwen-Image's vision file (models\\qwen-image\\" + VISION_FILE + "), which is "
             "missing. Close Spellcast3D and start it again: it puts back what is missing. Without example pictures "
             "Qwen-Image works as it is.")
# A request for a model whose denoiser is not in the models folder (said with its file's name). The full model is an
# optional download (the Images page); the Turbo is the installer's, so it is missing only on an install that has not
# been updated since, or after its file was deleted. FLUX.3 is an optional download too (its first missing file named).
MODEL_MISSING = {
    "full": ("The full Qwen-Image 2.1 is not downloaded yet (models\\qwen-image\\{name}). Download it with its button "
             "under Model on the Images page, or pick Turbo."),
    "turbo": ("Qwen-Image 2.1 Turbo's model file is missing (models\\qwen-image\\{name}). Close Spellcast3D and start "
              "it again: it downloads it. Meanwhile, pick Full."),
    "flux3": ("FLUX.3 Image Klein is not downloaded yet (models\\" + FLUX3_FOLDER + "\\{name}). Download it with its "
              "button under Model on the Images page, or pick Turbo."),
}
# A FLUX.3 request while the engine cannot draw it (flux3_engine_choice), and one it failed to load.
FLUX3_NO_ENGINE = ("FLUX.3 Image Klein needs a newer image engine (stable-diffusion.cpp) than this one: it comes with "
                   "the next update of Spellcast3D. Pick Turbo or Full meanwhile.")
FLUX3_NOT_LOADED = "FLUX.3 Image Klein could not be loaded by this image engine"

# (phase, message, fraction at its start, at its end); drawing and finishing are split between the pictures. The
# loading's words while a request switches the model: Progress.loading's.
PHASES = {
    "loading": ("Loading Qwen-Image 2.1", 0.0, 1.0),
    "references": ("Looking at the example pictures", 0.00, 0.04),
    "prompt": ("Reading the description", 0.04, 0.07),
    "drawing": ("Drawing", 0.07, 0.95),
    "finishing": ("Finishing", 0.95, 0.99),
    "saving": ("Saving", 0.99, 1.00),
}
DECODE_GUESS = 2.0                 # seconds per picture's decode until one was timed
LOG_LEVELS = {sdcpp.SD_LOG_DEBUG: " debug", sdcpp.SD_LOG_VERBOSE: " debug", sdcpp.SD_LOG_WARN: " WARNING",
              sdcpp.SD_LOG_ERROR: " ERROR"}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=10_000)
    negative_prompt: str = Field("", max_length=10_000)
    width: int = SIZE
    height: int = SIZE
    # which model draws (MODELS): the Turbo (its own 8 steps: steps, cfg_scale, sampler and first_block_cache are not
    # read for it, see sampling), the full model, or FLUX.3 Image Klein (its own 16 steps, no alpha: transparent is
    # not read for it either); the launcher switches to it first when another one is loaded (QwenEngine.use_model)
    model: Literal["turbo", "full", "flux3"] = DEFAULT_MODEL
    steps: int = Field(STEPS, ge=1, le=150)
    cfg_scale: float = Field(CFG_SCALE, ge=0.0, le=30.0)
    sampler: str = SAMPLER
    # < 0 = random; picture i gets seed + i. Bounded: sd.cpp's seed is an int64 (ctypes would wrap a bigger one
    # silently, and a negative result is random again), and the page's numbers are exact only below 2**53.
    seed: int = Field(-1, ge=-1, le=MAX_SEED)
    batch_count: int = Field(1, ge=1, le=MAX_BATCH)
    ref_images: list[str] = Field(default_factory=list)
    # Qwen-Image 2.1 only: the longer side (px) of its own copy of each example picture, in their order - Omni-Texture
    # has it see the depth map large and the example small. Absent: each copy has the pixels of the picture asked
    # for, or fewer when the prefix cache's budget says so (reference_pixels). See sized_reference.
    ref_sides: list[int] | None = None
    transparent: bool = True
    # Qwen-Image 2.1 only: a picture that repeats without a seam, side by side and one above the other. The engine
    # then pads its convolutions round the picture's edges and makes the transformer's positions periodic over the
    # picture (sd.cpp's circular axes); it keeps no example pictures' latents from one such request to the next.
    # A masked change of such a picture (changed_image) is drawn without them: see to_the_middle.
    tileable: bool = False
    # The picture these are made from was shown slid this far right and down (a share of its size; the page's offset
    # slider for tileable pictures): they are slid back as far, so that they lie where it lies (slid).
    slide_back: float = Field(0.0, ge=-0.5, le=0.5)
    # Tileable pictures made from the first example picture - that picture, changed: each is laid onto it to the pixel
    # (in_place). No border holds a tileable picture, and the model lays what it makes some pixels off.
    keep_place: bool = False
    # A change asked for in one part of a picture (the page's Smart Mask): the picture it is made from, as it is,
    # and what was brushed over it (white); init_image is the copy the model is shown, what was brushed covered in
    # magenta, mask_image the brush.
    # Drawn WHOLE, or HELD when tileable (the notes above with_room say what each is); the answer says when
    # nothing clearly changed where it was brushed. A picture of another shape than the request's (same_shape) is
    # drawn anew as without the two, and where it would have been held the answer warns.
    changed_image: str | None = None
    changed_mask: str | None = None
    # the First-Block Cache: the engine reuses its middle layers on FIRST_BLOCK_CACHE_SKIP of the steps (the
    # first third and the last step always computed), ~35 % faster; off = every step computed in full; None = the
    # card decides (first_block_cache_on: on for the GTX 10xx)
    first_block_cache: bool | None = None
    out_dir: str
    # A folder (inside the projects root) for what the model was given for this request and what it answered, kept
    # for the artist to look at: given.json and the pictures it names (keep_given).
    given_dir: str | None = None
    preview: bool = False       # a small picture of each step's result for GET /spellcast/preview (the module doc)
    # img2img: the picture the drawing starts from (the page's viewport), noised by `strength` (1: drawn anew, as
    # without it; lower keeps more of its layout and colours: sd.cpp starts that far into the schedule), and a mask of
    # it (white: drawn anew, black: kept as it is; none: all of it)
    init_image: str | None = None
    strength: float = Field(0.75, gt=0.0, le=1.0)
    mask_image: str | None = None
    # whether the engine keeps the prefix it computed (the words' and example pictures' keys and values, ~3 GB on the
    # card) for the next request, which draws faster when it has the same words and pictures (a re-roll); False for
    # requests whose prefix never comes back (Omni-Texture: every view's start picture and mask are new), so that
    # its room is free for the decode and the albedo pass after them (PREFIX_DROP_ENV). The pictures are the same.
    keep_prefix: bool = True


# Our albedo add-on (the folder --albedo, models/qwen-image/albedo): its LoRA for the denoiser, its own VAE decoder and
# the fixed prompt it was trained with, in the engine's format (scripts/convert_albedo_addon.py in our
# stable-diffusion.cpp fork).
ALBEDO_FILES = manifest.ALBEDO_FILES
ALBEDO_MISSING = ("The albedo add-on is not installed (models/qwen-image/albedo), or this image engine build cannot "
                  "run it.")


class AlbedoRequest(BaseModel):
    picture: str                # absolute, sides multiples of SIDE_STEP
    out: str                    # absolute .png to write the albedo to
    # on the model loaded (the add-on works on both Qwen-Image 2.1 models); the launcher switches first only when
    # FLUX.3 is the one loaded
    model: Literal["turbo", "full"] = DEFAULT_MODEL


def albedo_files(folder: Path | None) -> dict[str, Path] | None:
    """The add-on's files in `folder`, or None when any is missing."""
    if folder is None:
        return None
    files = {key: folder / name for key, name in ALBEDO_FILES.items()}
    return files if all(path.is_file() for path in files.values()) else None


class Cancelled(Exception):
    """The hub asked to stop (POST /spellcast/cancel)."""


class InputProblem(Exception):
    """Something about the request the artist has to fix (kind "input")."""


class OutOfMemory(Exception):
    """sd.cpp ran out of video memory or RAM (kind "oom")."""


class GenerationFailed(Exception):
    """generate_image returned no pictures for another reason (the log has sd.cpp's own lines)."""


# ---- prompt and placement (pure, tested) ---------------------------------------------------------------------------

def wrap_prompt(prompt: str, transparent: bool) -> str:
    """The prompt as sent to the model: for a transparent background, Qwen's documented RGBA wording around it."""
    text = " ".join(prompt.split())
    if not transparent or text.startswith(RGBA_PREFIX):
        return text
    body = text.rstrip(" .")
    return f"{RGBA_PREFIX} {body}. {RGBA_SUFFIX}" if body else f"{RGBA_PREFIX} {RGBA_SUFFIX}"


def condition_count(cfg_scale: float) -> int:
    """How many conditions sd.cpp computes at every step of a request: 1 without guidance - a CFG of exactly 1.0, as
    the float32 it gets: it skips the unconditional pass and never encodes the negative prompt - else 2, below 1.0
    too (src/pipeline/request.cpp: use_uncond = img_cfg != txt_cfg, and img_cfg is 1 for Qwen-Image)."""
    return 1 if ctypes.c_float(cfg_scale).value == 1.0 else 2


def choose_placement(free_mb: int, weights_mb: int, asked: str = "auto", max_vram: str | None = None,
                     prefix_cache: str = PREFIX_CACHE, prefix_keep: str = PREFIX_KEEP) -> dict[str, Any]:
    """Where the weights live and the memory settings (see PLACEMENTS; "auto" is offload) ->
    {"name": "gpu"|"offload", "params_backend", "max_vram", "prefix_cache", "prefix_keep", "budget_mb"}. free_mb:
    the card's free memory at start; budget_mb: the video memory sd.cpp has for everything but weights kept on the
    card (the prefix cache, the VAE decode, the working memory, the offloaded weights' copies). prefix_cache: the
    cache's type, the same on every card (PREFIX_CACHE unless --prefix-cache says another); prefix_keep: whether the
    engine keeps it between pictures (PREFIX_KEEP unless --prefix-keep says another); the reports name both."""
    name = "gpu" if asked == "gpu" else "offload"
    if name == "gpu":
        # sd.cpp's auto-fit with the whole card: it keeps every weight on the card when they fit
        placement = {"name": "gpu", "params_backend": None, "max_vram": max_vram,
                     "budget_mb": budget_mb(free_mb, max_vram, 512) - weights_mb}
    else:
        # sd.cpp's "biggest model my card can host": weights in RAM, a budget of the free memory at start minus 1 GiB
        placement = {"name": "offload", "params_backend": "*=cpu", "max_vram": max_vram or "-1",
                     "budget_mb": budget_mb(free_mb, max_vram or "-1", 512)}
    placement["prefix_cache"] = prefix_cache
    placement["prefix_keep"] = prefix_keep
    return placement


def budget_mb(free_mb: int, max_vram: str | None, margin_mb: int) -> int:
    """sd.cpp's video memory budget: max_vram in GiB (negative: the free memory at start minus that much), else the
    free memory minus its margin."""
    try:
        value = float(max_vram) if max_vram not in (None, "") else None
    except ValueError:
        value = None
    if value is None or value == 0:
        return free_mb - margin_mb
    if value < 0:
        return free_mb - int(-value * 1024)
    return min(free_mb - margin_mb, int(value * 1024))


def vae_tile(budget: int, direct: bool = False) -> int:
    """The side of the tiles a decode is cut into where it does not fit the budget in one piece: the largest of
    VAE_TILES whose own decode fits (needs_vae_tiling), the smallest where none does."""
    return next((tile for tile in VAE_TILES if not needs_vae_tiling(tile, tile, budget, direct)), VAE_TILES[-1])


def needs_vae_tiling(width: int, height: int, budget: int, direct: bool = False) -> bool:
    """Whether the picture's VAE decode does not fit the budget in one piece (sd.cpp would retry it in tiles after
    running out of memory - 256-pixel ones before the fork's fix, a grid of seams; 512-pixel tiles from the start are
    faster and seam-free). direct: the decode with direct
    convolutions (vae_direct), about half the memory, and their scratch beyond what the engine allows for it."""
    megapixels = width * height / 1e6
    if not direct:
        return megapixels * VAE_DECODE_MB_PER_MP > budget - 512
    scratch = max(0.0, (width + 2) * (height + 2) * VAE_DIRECT_SCRATCH / 2 ** 20 - ENGINE_SCRATCH_MB)
    return megapixels * VAE_DIRECT_MB_PER_MP + scratch > budget - 512


# ggml's direct convolution is fast where its kernel for it is tiled: on the tensor cores (Turing, RTX 20xx, and newer:
# conv2d_implicit_gemm_f16 with mma) and, since the engine release spellcast-perf-838b2f2, on the GTX 10xx (Pascal)
# through the same kernel's F32 SIMT path: a 1024x1024 decode in one piece took 7.1 s on a GTX 1070 Ti (8 GB, peak 5.1
# GB) against 14.2 s for the usual convolutions in 512-px tiles (2026-10-24, a rented pod; the engines before it fell to
# a naive kernel there, one thread per output: 139 s). Volta (7.0) has no such measurement and keeps the usual ones.
VAE_DIRECT_MIN_CAPABILITY = (7, 5)
VAE_DIRECT_PASCAL = ((6, 1), (7, 0))   # [first, last) capability of the GTX 10xx


def vae_direct(budget: int, width: int = SIZE, height: int = SIZE,
               capability: tuple[int, int] = VAE_DIRECT_MIN_CAPABILITY, kept_mb: int = 0) -> bool:
    """Whether the VAE runs with direct convolutions (sd.cpp's vae_conv_direct, chosen once for the context): where the
    usual decode of a picture of the default size would need tiles but the direct one fits in one piece (8 GB cards:
    ~2.2 s instead of ~3.2 s in tiles), on a card of VAE_DIRECT_MIN_CAPABILITY or newer. Not where the usual one fits
    (it is ~0.4 s faster) or neither does. A GTX 10xx (VAE_DIRECT_PASCAL) always takes them, in tiles where one piece
    does not fit: its usual form in 512-pixel tiles is 3-4x slower there (a GTX 1080 held to 6 GB, 2026-11-07: a
    1024x1024 decode 18.7 s against 4.7 s; a painting 60 -> 44 s, an albedo view 27 -> 12 s). capability: the card's
    (major, minor). kept_mb: the diffusion weights the engine keeps on the card between runs (keep_params): the decode
    must fit beside them, so the direct one is taken wherever it does, the usual one (its im2col buffers would push
    the weights off the card for every picture) only where the direct one does not."""
    cap = tuple(capability)
    if VAE_DIRECT_PASCAL[0] <= cap < VAE_DIRECT_PASCAL[1]:
        return True
    if cap < VAE_DIRECT_MIN_CAPABILITY:
        return False
    if kept_mb > 0:
        room = budget - kept_mb
        if not needs_vae_tiling(width, height, room):
            return False                    # the usual decode fits beside the kept weights (16 GB cards)
        if not needs_vae_tiling(width, height, room, direct=True):
            return True                     # the direct one does (12 GB cards)
    return needs_vae_tiling(width, height, budget) and not needs_vae_tiling(width, height, budget, direct=True)


def keep_params(asked: str, budget: int, diffusion_mb: int, width: int = SIZE, height: int = SIZE,
                engine_knows: bool = True) -> bool:
    """--keep-params: whether the engine keeps a model's weights on the card after its run (KEEP_ENV). auto: where the
    budget holds the diffusion weights beside the decode in one piece with direct convolutions and the VAE's weights
    (the drawing's working memory is freed before the decode; the engine's own margin on top): 12 GB cards, with
    ~1.2 GB to spare on an RTX 3060; an 8 GB card's decode would evict them every picture anyway. engine_knows: the
    engine build reads the variable (the pinned build when install/sdcpp.py ENGINE_KEEPS_PARAMS says so; a build
    under test is taken to): auto stays off otherwise, since the direct decode auto pairs with it would only cost."""
    if asked == "on":
        return True
    if asked != "auto" or diffusion_mb <= 0 or not engine_knows:
        return False
    need = diffusion_mb + VAE_WEIGHTS_MB + width * height / 1e6 * VAE_DIRECT_MB_PER_MP + 512
    return budget >= need


# ---- Omni-Texture's sheets ------------------------------------------------------------------------------------------
# Omni-Texture paints several views of a model in one square picture, a sheet (adapters/omnitexture.py), up to
# 2048 a side, and reads from health how large one may be on this card ("sheet_side"). What bounds it is the
# painting's first step. The albedo add-on's pass does not: it reads the picture beside the one it draws, twice
# the picture's tokens in every block (5,917 MB for a whole 2048x2048 sheet, which an 8 GB card does not have), so
# a larger picture goes through it smaller (albedo_size, lighting_off).
# MEASURED 2026-11-16 (RTX 3060, the Turbo Q4_K_M, sm_86; devtools/omnitexture-checks/sheet_trial.py):
# - the budget held to 6,144 MB (--card-gb 8): a 1536x1536 sheet painted in 34 s (first step 1,704 MB of compute
#   buffer, later steps 1,595 MB), its albedo in 13 s (the pass 3,045 MB); a 2048x2048 sheet in 73 s (first step
#   3,195 MB, later steps 2,832 MB, encode and decode in 49 tiles); the card's peak, all programs, 7.6-8.0 GB with
#   ~1.1-1.8 GB of them the other programs';
# - the card's own budget (10,224 MB): 2048x2048 painted in 72-74 s; 1536x1536 in 37 s and its albedo in 9 s;
#   1024x1024 in 19-20 s and 4 s.
# NOT measured: a real 8 GB card, the GTX 10xx / RTX 20xx (no prefix pass, flash attention).
SHEET_SIDE_MIN, SHEET_SIDE_MAX, SHEET_SIDE_STEP = 1024, 2048, 128
SHEET_PAINT_MB_PER_TOKEN = 0.2     # a sheet's first step per token of the picture (seen: 3,195 MB for 2048x2048)
SHEET_WEIGHTS_MB = 950             # the block of weights it needs beside that (seen: 657-914 MB)


def sheet_side(budget: int) -> int:
    """The side of the largest square picture Omni-Texture may paint within the budget (SHEET_SIDE_MIN at the
    least: a single view's picture, as ever)."""
    room = budget * MEMORY_SAFETY - SHEET_WEIGHTS_MB
    side = SHEET_SIDE_MIN
    while side + SHEET_SIDE_STEP <= SHEET_SIDE_MAX \
            and (side + SHEET_SIDE_STEP) ** 2 / 256 * SHEET_PAINT_MB_PER_TOKEN <= room:
        side += SHEET_SIDE_STEP
    return side


# ---- the first step as a prefix-only pass ---------------------------------------------------------------------------
# A request with example pictures computes the prefix (the words' and pictures' keys and values) in its first step,
# over the whole sequence (one graph cut into 34 segments on a 12 GB card). SD_QWEN_PREFIX_PASS=1 makes Spellcast3D's
# engine build compute it in a pass of its own and then the step as every later step, bit for bit the same picture
# (the engine takes it only where it is: a kept F16 prefix under flash / sage attention, a prefix at least as long
# as the picture, and products that give a row the same bits at the three row counts involved - the int8 path, or
# MMQ where ggml-cuda says so, ggml_backend_cuda_mmq_rows_alike: not at 256x256, for one). Measured on an RTX 3060:
# an Omni-Texture view's first step 4.45 -> 3.92 s on the int8 path, its sampling 17.0 -> 16.5 s under MMQ, every
# picture and GLB byte-identical. On by default from the card generation where it was measured (sm_80+, with an
# engine that has the pass's gates: install/sdcpp.py ENGINE_PREFIX_PASS); RTX 20xx / GTX 10xx keep the one-graph
# step (on a GTX 1080 the pass changed the picture). A variable already set is never changed.
PASS_ENV = "SD_QWEN_PREFIX_PASS"
PASS_MIN_CAPABILITY = (8, 0)


def set_prefix_pass(capability: tuple[int, int], engine_knows: bool,
                    environ: MutableMapping[str, str] | None = None) -> str:
    """SD_QWEN_PREFIX_PASS for this card and engine build (see PASS_ENV) -> the log's line."""
    environ = os.environ if environ is None else environ
    current = environ.get(PASS_ENV)
    if current is not None:
        return f"first step: {PASS_ENV}={current} was already set, kept"
    if engine_knows and tuple(capability) >= PASS_MIN_CAPABILITY:
        environ[PASS_ENV] = "1"
        return f"first step: the prefix in a pass of its own ({PASS_ENV}=1), the same pictures"
    environ[PASS_ENV] = "0"
    return "first step: one graph over the prefix and the picture"


def set_keep(keep: bool, environ: MutableMapping[str, str] | None = None) -> str:
    """keep_params's answer applied to this process's environment (a variable already set is never changed) -> the
    log's line. Read by the engine at its first run's end, so before the first picture is enough."""
    environ = os.environ if environ is None else environ
    current = environ.get(KEEP_ENV)
    if current is not None:
        return f"weights between runs: {KEEP_ENV}={current} was already set, kept"
    if keep:
        environ[KEEP_ENV] = "1"
        return (f"weights between runs: kept on the card ({KEEP_ENV}=1): the diffusion weights stay from picture to "
                "picture, the text encoder's until the decode needs their room")
    return "weights between runs: freed at each run's end (the card has no room to keep them beside the decode)"


def attention_choice(asked: str, pinned: bool, capability: tuple[int, int]) -> str:
    """The diffusion model's attention: "sage" (the engine's SageAttention) for --attention auto with the pinned engine
    build, or for --attention sage with any build (developers: a build under test), on a card of compute capability
    SAGE_MIN_CAPABILITY or newer; else "flash" (see ATTENTIONS: asked for sage attention on an older card, the engine
    refuses to start instead of falling back). capability: the card's (major, minor), always given - an assumed one
    is how an old card would be asked for sage attention."""
    wanted = asked == "sage" or (asked == "auto" and pinned)
    return "sage" if wanted and tuple(capability) >= SAGE_MIN_CAPABILITY else "flash"


def flux3_attention_choice(asked: str, capability: tuple[int, int]) -> str:
    """FLUX.3 Image Klein's attention: FLUX3_ATTENTION_AUTO (flash) unless --attention sage asks for the sage op (on a
    card that has its kernels, as attention_choice)."""
    if asked == "sage" and tuple(capability) >= SAGE_MIN_CAPABILITY:
        return "sage"
    return FLUX3_ATTENTION_AUTO


FLUX3_ENGINES = ("auto", "on", "off")


def flux3_engine_choice(asked: str, pinned: bool, gate: bool | None = None) -> tuple[bool, str]:
    """Whether this engine build draws FLUX.3 Image Klein -> (yes?, the log's line). auto: the pinned build when the
    ONE gate says it can (install/sdcpp.py ENGINE_HAS_FLUX3; `gate` overrides it for the tests), never another build;
    on (developers: a build under test, e.g. qwen_bench --sdcpp): taken to draw it - a build that cannot fails to load
    it, with FLUX3_NOT_LOADED; off: never."""
    gate = engine_pin.ENGINE_HAS_FLUX3 if gate is None else gate
    if asked == "off":
        return False, "FLUX.3 Image Klein: off (--flux3-engine off)."
    if asked == "on":
        return True, "FLUX.3 Image Klein: on (--flux3-engine on: this engine build is taken to draw it)."
    if not pinned:
        return False, ("FLUX.3 Image Klein: off - this engine build is not the pinned one (--flux3-engine on tries it "
                       "anyway).")
    if not gate:
        return False, (f"FLUX.3 Image Klein: off - the pinned engine {sdcpp.PINNED_TAG} cannot draw it "
                       f"(install/sdcpp.py ENGINE_HAS_FLUX3).")
    return True, f"FLUX.3 Image Klein: the pinned engine {sdcpp.PINNED_TAG} draws it."


def flux3_reference_pixels(count: int, target_pixels: int) -> int | None:
    """The largest VAE copy (pixels) of each of `count` FLUX.3 example pictures: the engine's own size (None) for up to
    FLUX3_REFERENCES_FULL_SIZE of them, smaller for more, so that together they never add more than that many pictures'
    worth of tokens to every block (FLUX.3 has no prefix cache: a 1 MP copy is 4,096 tokens in every step, ten of them
    far past an 8 GB card). Not measured on an 8 GB card."""
    if count <= FLUX3_REFERENCES_FULL_SIZE:
        return None
    return max(MIN_REFERENCE_TOKENS * 256, FLUX3_REFERENCES_FULL_SIZE * target_pixels // count)


def copy_cap(placement: Mapping[str, Any], req: GenerateRequest, count: int) -> int | None:
    """The most pixels this card's copy of each of a request's `count` example pictures gets (None: the engine's own
    size - the request's picture's for Qwen-Image, 1024x1024 for FLUX.3): Qwen-Image's by the prefix cache's budget
    (reference_pixels), FLUX.3's so that together they stay within a few pictures' worth of tokens."""
    if req.model not in QWEN_MODELS:
        return flux3_reference_pixels(count, req.width * req.height)
    plan = sampling(req.model, req.steps, req.cfg_scale, req.sampler, req.first_block_cache,
                    placement.get("capability", (99, 0)))
    return reference_pixels(placement["budget_mb"], placement["prefix_cache"], count, req.width * req.height,
                            condition_count(plan["cfg_scale"]))


def prefix_room(megabytes: float, target_tokens: int, cache_mb_per_token: float) -> float:
    """The tokens of prefix (the words, the vision encoder's, the VAE copies) whose cache and the step that fills
    it fit `megabytes` beside a picture of `target_tokens` and a block of weights."""
    return (megabytes - SEGMENT_MB - STEP_MB_PER_TOKEN * target_tokens) / (FILL_MB_PER_TOKEN + cache_mb_per_token)


def reference_pixels(budget: int, cache: str, count: int, target_pixels: int, conditions: int = 1) -> int | None:
    """The largest VAE copy (pixels) of each of `count` example pictures for which the prefix cache - one per
    condition of the request (condition_count: 1 without guidance, 2 with) - and the step that fills it fit the
    budget, or None when they fit at sd.cpp's own size (the target picture's area). cache: the cache's type
    (PREFIX_MB_PER_TOKEN). A budget under REFERENCE_BUDGET_MB gives the copies that one would, as far as it holds
    them itself."""
    if count <= 0:
        return None
    per_token = PREFIX_MB_PER_TOKEN.get(cache, PREFIX_MB_PER_TOKEN["f16"]) * conditions
    target_tokens = target_pixels // 256
    prefix_tokens = prefix_room(max(budget, REFERENCE_BUDGET_MB) * MEMORY_SAFETY, target_tokens, per_token)
    if budget < REFERENCE_BUDGET_MB:
        prefix_tokens = min(prefix_tokens, prefix_room(budget, target_tokens, per_token))
    tokens = int((prefix_tokens - TEXT_TOKENS) / count - VISION_TOKENS)
    if tokens >= target_tokens:
        return None
    return max(tokens, MIN_REFERENCE_TOKENS) * 256


def model_args(placement: dict[str, Any]) -> str | None:
    """sd.cpp's model_args for the placement: the prefix cache's type and whether it is kept between pictures, each
    when it is not sd.cpp's own choice ("auto"); None when both are."""
    args = []
    if placement.get("prefix_cache") not in (None, "", "auto"):
        args.append(f"qwen_image_2_1_prefix_cache_type={placement['prefix_cache']}")
    if placement.get("prefix_keep") not in (None, "", "auto"):
        args.append(f"qwen_image_2_1_prefix_cache_keep={placement['prefix_keep']}")
    return ",".join(args) or None


def pin_choice(asked: str, total_mb: int | None, already: str | None, in_ram: bool = True) -> tuple[str | None, str]:
    """What --pin means on this PC (see PIN_ENV) -> (SD_PIN_PARAMS' value to set, or None: the environment stays as
    it is; the log's one line saying what was done and why). asked: auto | on | off. total_mb: the PC's RAM
    (total_ram_mb; None: unknown). already: the variable as it is set outside, None when it is not - the user's own
    setting always stays, whatever --pin says. in_ram: the weights live in RAM (the offload placement); kept on the
    card (--placement gpu) there is nothing to copy, so auto leaves it."""
    off = "The diffusion weights are not kept in page-locked RAM"
    if already is not None:
        return None, f"{PIN_ENV} is already set ({already!r}): left as it is (--pin {asked} never changes a set one)."
    if asked == "off":
        return None, f"{off} (--pin off)."
    if asked == "on":
        why = "--pin on"
    elif not in_ram:
        return None, f"{off}: --placement gpu keeps them on the card."
    elif total_mb is None:
        return None, f"{off}: how much RAM this PC has is unknown (--pin on does it anyway)."
    elif total_mb < PIN_MIN_RAM_MB:
        return None, (f"{off}: this PC has {total_mb:,} MB of RAM, and --pin auto does it from 24 GB on (--pin on "
                      f"does it anyway).")
    elif total_mb >= PIN_TEXT_MIN_RAM_MB:
        return PIN_TEXT_PARAMS, (f"{PIN_ENV}={PIN_TEXT_PARAMS} (--pin auto: this PC has {total_mb:,} MB of RAM, 32 GB "
                                 f"or more): Spellcast3D's engine build keeps the diffusion weights and the text "
                                 f"encoder's in page-locked RAM (they reach the graphics card faster; that much RAM "
                                 f"stays taken while Qwen-Image is loaded). The official stable-diffusion.cpp release "
                                 f"ignores it.")
    else:
        why = f"--pin auto: this PC has {total_mb:,} MB of RAM, 24 GB or more"
    return PIN_PARAMS, (f"{PIN_ENV}={PIN_PARAMS} ({why}): Spellcast3D's engine build keeps the diffusion weights in "
                        f"page-locked RAM (they reach the graphics card faster; that much RAM stays taken while "
                        f"Qwen-Image is loaded). The official stable-diffusion.cpp release ignores it.")


def noise_seed(seed: int, pictures: list[Any]) -> int:
    """The seed the engine draws a request's starting noise from: the request's own for a picture drawn from words
    alone; for one drawn from other pictures, that seed moved on by a number taken from those pictures. Redrawn from
    the very noise it once came from - a picture changed with the seed that made it - a picture comes out harsh and
    speckled all over. Moved on by its pictures, a seed never gives that noise, gives the same picture every time,
    and picture i of a batch is still what seed + i alone gives."""
    if not pictures:
        return seed
    digest = hashlib.blake2b(digest_size=8)
    for picture in pictures:
        digest.update(picture.tobytes())
    return (seed + 1 + int.from_bytes(digest.digest(), "big") % MAX_SEED) % (MAX_SEED + 1)


def drawn_from(references: list[Any], init: Any) -> list[Any]:
    """The pictures a request's noise seed moves on by (noise_seed): its example pictures as read, then its start
    picture, if any."""
    return [*references, *([init] if init is not None else [])]


def batch_seeds(seed: int, count: int) -> list[int]:
    """sd.cpp's seeds for a batch: picture i gets seed + i (src/pipeline/image.cpp)."""
    return [seed + i for i in range(count)]


def is_multiple(value: int, step: int = SIDE_STEP) -> bool:
    return value % step == 0


# ---- progress ------------------------------------------------------------------------------------------------------

_GENERATING = re.compile(r"generating image: (\d+)/(\d+) - seed (-?\d+)")
_TAKING = re.compile(r"taking ([0-9.]+)s")
_DECODED = re.compile(r"latent (\d+) decoded, taking ([0-9.]+)s")
# sd.cpp's words when memory runs out, e.g. "model manager cannot make enough memory available on CUDA0 ... need
# 789.85 MB / available 585.85 MB" (its offload manager), "CUDA error: out of memory", "cudaMalloc failed".
_OOM = re.compile(r"out of memory|cudaMalloc failed|failed to allocate|insufficient memory|not enough memory|"
                  r"enough memory available|alloc.*failed", re.IGNORECASE)


def ran_out_of_memory(lines: list[str]) -> bool:
    """Whether sd.cpp's warning and error lines of a failed run say memory ran out (the prefix cache's own shortage is
    no failure: sd.cpp retries without it)."""
    return any(_OOM.search(line) and "prefix cach" not in line for line in lines)


class Progress:
    """What GET /spellcast/progress reports. Written by sd.cpp's log and progress callbacks (on the generating thread),
    read by the routes."""

    def __init__(self) -> None:
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.previews = 0                   # preview pictures so far: each one's number (never starts again)
        self._reset()

    def _reset(self) -> None:
        self.phase, self.message, self.fraction = "idle", "Idle", 0.0
        # the picture being drawn as it looks so far: (its number, step, which picture of the request, PNG bytes)
        self.preview: tuple[int, int, int, bytes] | None = None
        self.step = self.steps = 0
        self.image = self.images = 0
        self.step_times: list[float] = []
        self.decode_seconds = DECODE_GUESS
        self.decoded = 0
        self.moved_at = time.monotonic()    # the last step (or phase) reported: eta counts down from it
        self.lines: list[str] = []          # this run's warning and error lines (for the failure's kind)
        self.timings: dict[str, float] = {}
        self.cache_dropped = False
        self.switching: str | None = None   # a request's model switch: its words (loading)
        self.size = ""                      # the pictures' size, said in every phase of a request
        self.box: list[float] | None = None     # a window's request: the part of the picture its previews show (window_place)

    def loading(self, words: str | None = None) -> None:
        """The models are being loaded: at the start (the fraction follows sd.cpp's loading), or for a request that
        switches the model (words: what it waits for, e.g. "Loading the full Qwen-Image 2.1"; the fraction stays where
        the request begins, 0)."""
        with self.lock:
            self._reset()
            self.switching = words
            self._enter("loading")

    def start(self, images: int, steps: int, references: int, size: tuple[int, int] | None = None) -> None:
        """A request for pictures of `size` (width, height) begins (the route clears .cancel before, so a cancel
        that comes meanwhile still counts)."""
        with self.lock:
            self._reset()
            self.images, self.steps = images, steps
            self.size = f"{size[0]} × {size[1]} px" if size else ""
            self._enter("references" if references else "prompt")

    def stop(self) -> None:
        with self.lock:
            self._reset()

    def saving(self) -> None:
        with self.lock:
            self._enter("saving")

    def _enter(self, phase: str, done: float = 0.0, picture: str = "", detail: str = "") -> None:
        """The phase and its words: "Drawing picture 2 of 4 · 1536 × 1536 px · step 7 of 25". The size is in every
        phase of a request: the artist is reminded at what resolution it renders."""
        label, start, end = PHASES[phase]
        self.phase = phase
        self.moved_at = time.monotonic()
        if phase == "loading" and self.switching:
            label, self.fraction = self.switching, 0.0
        elif phase == "loading":
            self.fraction = start + (end - start) * done
        else:
            self.fraction = max(self.fraction, start + (end - start) * min(max(done, 0.0), 1.0))
        size = "" if phase == "loading" else self.size
        self.message = " · ".join(part for part in (f"{label} {picture}".rstrip(), size, detail) if part)

    # -- sd.cpp's callbacks
    def on_log(self, level: int, text: str) -> None:
        with self.lock:
            if level >= sdcpp.SD_LOG_WARN:
                self.lines = (self.lines + [text])[-40:]
                if "insufficient memory for prefix caching" in text:
                    self.cache_dropped = True
            if self.phase in ("idle", "loading"):
                return
            if "encode_first_stage completed" in text:
                self._time("references", text)
                self._enter("prompt")
            elif "get_learned_condition completed" in text:
                self._time("prompt", text)
            elif (m := _GENERATING.search(text)) is not None:
                self.image, self.images = int(m.group(1)), int(m.group(2))
                self.step = 0
                self.preview = None          # the picture before this one is not the one being drawn any more
                self._drawing(0)
            elif "sampling completed" in text:
                found = _TAKING.search(text)
                if found:
                    self.timings[f"drawing_{self.image}"] = float(found.group(1))
                self._drawing(self.steps)
            elif "decoding" in text and "latent" in text and self.phase == "drawing":
                self._finishing()
            elif (m := _DECODED.search(text)) is not None:
                self.decoded = int(m.group(1))
                self.decode_seconds = float(m.group(2))
                self.timings[f"finishing_{self.decoded}"] = self.decode_seconds
                self._finishing()

    def _time(self, key: str, text: str) -> None:
        found = _TAKING.search(text)
        if found:
            self.timings[key] = float(found.group(1))

    def on_step(self, step: int, steps: int, seconds: float) -> None:
        with self.lock:
            if self.phase == "loading":
                if steps > 0:
                    self._enter("loading", step / steps)
                return
            if self.phase in ("references", "prompt"):
                return                       # the vision encoder's or a tiled encode's own steps
            if self.phase == "drawing":
                if steps > 0:
                    self.steps = steps
                self.step = step
                if step > 0 and seconds > 0:
                    self.step_times = (self.step_times + [seconds])[-6:]
                self._drawing(step)
            elif self.phase == "finishing" and steps > 0:
                self._finishing(step / steps)

    def on_preview(self, step: int, picture: bytes) -> None:
        """The picture being drawn as it looks after `step` steps (a small PNG: QwenEngine._on_preview). Only the
        newest is kept, and only while pictures are drawn."""
        with self.lock:
            if self.phase == "drawing":
                self.previews += 1
                self.preview = (self.previews, abs(step), self.image, picture)

    def newest_preview(self) -> tuple[int, int, int, bytes] | None:
        with self.lock:
            return self.preview

    def _drawing(self, step: int) -> None:
        """`step` steps are done: the words say the one being drawn ("step 1 of 25" from the start, not "step 0").
        The number of steps is the request's own (start), then sd.cpp's (on_step): no default is named here."""
        n, s = max(self.images, 1), max(self.steps, 1)
        done = (max(self.image - 1, 0) + min(step, s) / s) / n
        now = min(step + 1, s)
        self._enter("drawing", done, "" if n == 1 else f"picture {self.image} of {n}", f"step {now} of {s}")

    def _finishing(self, part: float = 0.0) -> None:
        n = max(self.images, 1)
        self._enter("finishing", (self.decoded + part) / n, "" if n == 1 else f"picture {min(self.decoded + 1, n)} of {n}")

    def eta(self) -> float | None:
        """Seconds left: the steps and decodes still to come at the pace measured, less the time the one under way
        has taken so far (so the estimate counts down between two steps; never past the end of that one)."""
        since = time.monotonic() - self.moved_at
        if self.phase == "drawing" and self.step_times:
            per_step = sum(self.step_times) / len(self.step_times)
            left = (self.steps - self.step) + (self.images - self.image) * self.steps
            return round(left * per_step + self.images * self.decode_seconds - min(since, 0.95 * per_step), 1)
        if self.phase == "finishing":
            return round((self.images - self.decoded) * self.decode_seconds - min(since, 0.95 * self.decode_seconds), 1)
        return None

    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            state = {"phase": self.phase, "step": self.step, "steps": self.steps,
                     "fraction": round(self.fraction, 3), "message": self.message, "eta": self.eta()}
            if self.preview is not None:        # GET /spellcast/preview has its picture
                seq, step, image, _picture = self.preview
                state["preview"] = {"seq": seq, "step": step, "image": image, **({"box": self.box} if self.box else {})}
            return state


# ---- the GPU side --------------------------------------------------------------------------------------------------

class Cuda:
    """The CUDA runtime next to stable-diffusion.dll (the very DLL ggml-cuda uses): the card's memory before loading,
    and whether CUDA still works after an error."""

    def __init__(self) -> None:
        self.rt = ctypes.CDLL("cudart64_12.dll" if os.name == "nt" else "libcudart.so.12")
        self.rt.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self.rt.cudaDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]

    def release_others(self, keep: int, devices: list[dict[str, Any]]) -> None:
        """Measuring a card (devices) gives this process a context on it, a few hundred MB of its memory: give them
        back on every card but `keep`, the one sd.cpp computes on, and make that one current again."""
        for device in devices:
            if device["index"] != keep and self.rt.cudaSetDevice(device["index"]) == 0:
                self.rt.cudaDeviceReset()
        self.rt.cudaSetDevice(keep)

    def devices(self) -> list[dict[str, Any]]:
        """[{"index", "name", "total_mb", "free_mb", "capability"}] of every CUDA device."""
        count = ctypes.c_int(0)
        if self.rt.cudaGetDeviceCount(ctypes.byref(count)) != 0:
            return []
        found = []
        for index in range(count.value):
            free, total = ctypes.c_size_t(0), ctypes.c_size_t(0)
            major, minor = ctypes.c_int(0), ctypes.c_int(0)
            if self.rt.cudaSetDevice(index) != 0 or self.rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) \
                    or self.rt.cudaDeviceGetAttribute(ctypes.byref(major), 75, index) \
                    or self.rt.cudaDeviceGetAttribute(ctypes.byref(minor), 76, index):
                continue            # 75, 76: cudaDevAttrComputeCapabilityMajor / Minor; a card they fail on is no use
            found.append({"index": index, "name": device_name(index), "total_mb": total.value // 2 ** 20,
                          "free_mb": free.value // 2 ** 20, "capability": (major.value, minor.value)})
        return found

    def works(self) -> bool:
        """False when CUDA is broken for this process (a sticky error: an illegal access, a launch failure, a Windows
        GPU timeout). Running out of memory for the test allocation is no such error."""
        err = self.rt.cudaDeviceSynchronize()
        if err == 0:
            pointer = ctypes.c_void_p()
            err = self.rt.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(256))
            if err == 0:
                self.rt.cudaFree(pointer)
        if err not in (0, 2):                    # 2: cudaErrorMemoryAllocation
            print(f"[spellcast] CUDA check failed: error {err}", flush=True)
            return False
        return True


def device_name(index: int) -> str:
    try:
        cuda = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        device = ctypes.c_int(0)
        name = ctypes.create_string_buffer(256)
        if cuda.cuInit(0) == 0 and cuda.cuDeviceGet(ctypes.byref(device), index) == 0 and \
                cuda.cuDeviceGetName(name, 256, device) == 0:
            return name.value.decode("utf-8", "replace")
    except OSError:
        pass
    return f"CUDA device {index}"


class MemoryStatus(ctypes.Structure):
    """Windows' MEMORYSTATUSEX."""
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memory_status() -> MemoryStatus | None:
    """GlobalMemoryStatusEx's answer, or None (the call failed). Elsewhere (a Linux test pod) the same two numbers from
    /proc/meminfo, capped by the container's cgroup memory limit: linux_memory_status."""
    if os.name != "nt":
        return linux_memory_status()
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status


def linux_memory_status() -> MemoryStatus | None:
    """MemTotal / MemAvailable of /proc/meminfo as ullTotalPhys / ullAvailPhys, each no more than the cgroup's limit
    (v2 memory.max, v1 memory.limit_in_bytes) and its headroom (the limit less what the cgroup uses besides its file
    cache, which the kernel gives back on demand - as Windows counts its standby list as available): what a container
    on a big host may really use. None when /proc/meminfo cannot be read."""
    try:
        info = {k: int(v.split()[0]) * 1024 for k, v in
                (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)}
    except (OSError, ValueError):
        return None
    total, avail = info.get("MemTotal", 0), info.get("MemAvailable", info.get("MemFree", 0))
    for limit_file, usage_file in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                                   ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                                    "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            limit = Path(limit_file).read_text().strip()
        except OSError:
            continue
        if limit.isdigit() and int(limit) < total:
            total = int(limit)
            try:
                used = int(Path(usage_file).read_text().strip())
                stat = dict(line.split() for line in Path(Path(limit_file).parent, "memory.stat").read_text().splitlines()
                            if len(line.split()) == 2)
                used -= int(stat.get("file", stat.get("total_cache", 0)))       # v2 "file", v1 "total_cache"
                avail = min(avail, max(0, total - used))
            except (OSError, ValueError):
                avail = min(avail, total)
        break
    status = MemoryStatus()
    status.ullTotalPhys, status.ullAvailPhys = total, avail
    return status


def free_ram_mb() -> int | None:
    status = memory_status()
    return None if status is None else int(status.ullAvailPhys // 2 ** 20)


def total_ram_mb() -> int | None:
    """The PC's RAM as Windows counts it (ullTotalPhys): what is installed less what the hardware keeps for itself,
    so a little under the 16 / 24 / 32 GB on the box. None when unknown."""
    status = memory_status()
    return None if status is None else int(status.ullTotalPhys // 2 ** 20)


def set_pinning(asked: str, in_ram: bool = True, environ: MutableMapping[str, str] | None = None) -> str:
    """--pin, applied to this process's environment (pin_choice decides) -> the log's line. To be called BEFORE
    stable-diffusion.dll is loaded: the engine reads SD_PIN_PARAMS once, through its C runtime's getenv, and a C
    runtime of the DLL's own takes its copy of the environment when the DLL is loaded."""
    environ = os.environ if environ is None else environ
    value, said = pin_choice(asked, total_ram_mb(), environ.get(PIN_ENV), in_ram)
    if value is not None:
        environ[PIN_ENV] = value
    return said


def matmul_choice(asked: str, current: str | None) -> tuple[str | None, str]:
    """--matmul -> (the value MATMUL_ENV gets, or None to leave it; the log's line). See MATMULS."""
    if current is not None:
        return None, f"matmuls: {MATMUL_ENV}={current} was already set, kept"
    if asked == "int8":
        return None, "matmuls: the engine's int8 path where it has one (--matmul int8)"
    return "off", f"matmuls: the MMQ kernels ({MATMUL_ENV}=off)"


def set_matmul(asked: str, environ: MutableMapping[str, str] | None = None) -> str:
    """--matmul, applied to this process's environment (matmul_choice decides) -> the log's line. Before the DLL is
    loaded, as set_pinning."""
    environ = os.environ if environ is None else environ
    value, said = matmul_choice(asked, environ.get(MATMUL_ENV))
    if value is not None:
        environ[MATMUL_ENV] = value
    return said


def load_runtime(sdcpp_dir: Path, pin: str = "auto", in_ram: bool = True, matmul: str = "mmq",
                 capability: tuple[int, int] | None = None) -> Any:
    """stable-diffusion.dll: loaded, its build named in the log and its layout checked (OSError, sdcpp.LayoutError) -
    after --pin went into the environment, where it has to be before the DLL is (set_pinning). capability: the card
    the engine computes on (LORA_TYPE_ENV)."""
    print("[spellcast] " + set_pinning(pin, in_ram), flush=True)
    set_matmul(matmul)   # said in the placement line (placement["matmul"]), no line of its own
    if capability is not None and tuple(capability) < LORA_Q8_BELOW:
        os.environ.setdefault(LORA_TYPE_ENV, "q8_0")
    lib = sdcpp.load(sdcpp_dir)
    print(f"[spellcast] stable-diffusion.cpp commit {sdcpp.commit(lib)} (pinned {sdcpp.PINNED_TAG}"
          + ("" if sdcpp.is_pinned(lib) else ": ANOTHER BUILD, its layout is checked") + ")", flush=True)
    sdcpp.check_layout(lib)
    return lib


def compute_device(devices: list[dict[str, Any]], placement: str) -> dict[str, Any]:
    """The card sd.cpp computes on (Cuda.devices' entries, at least one): the first one (offload), or the one with the
    most free memory (gpu: its auto-fit)."""
    return max(devices, key=lambda d: d["free_mb"]) if placement == "gpu" else devices[0]


def read_order(files: Mapping[str, Path | None], pinned: str | None) -> list[Path]:
    """A model's files (a model_sets entry) in the order they are wanted. First what the engine reads while it
    loads, in its own order: the page-locked components' files (`pinned`: SD_PIN_PARAMS' value, None for none), then
    the VAE, which it always reads. Then what it only maps, in the order a first picture uses it: the text encoder,
    the denoiser, the vision part (example pictures alone need it). Read in another order the engine waits for a
    small file behind a large one that nothing needs yet: the VAE's 0.6 GB took 10-18 s instead of 1-2."""
    parts = set((pinned or "").split(","))
    locked = [key for part, key in PINNED_FILES if part in parts or "all" in parts]
    keys = [*locked, "vae", *(key for key in ("llm", "diffusion", "vision") if key not in locked)]
    return [path for key in keys if (path := files.get(key)) is not None]


def read_ahead(paths: list[Path], stop: threading.Event | None = None, what: str = "Model files") -> None:
    """Read the files once, one after another, so Windows has them in its file cache before the engine's own reads
    get there (see PIN_MARK). Nothing waits for it. `stop`: set to give it up (another model is being loaded).
    `what`: the files' name in the log's line."""
    started, total = time.monotonic(), 0
    try:
        for path in paths:
            with open(path, "rb", buffering=0) as f:
                while chunk := f.read(READ_CHUNK):
                    if stop is not None and stop.is_set():
                        return
                    total += len(chunk)
    except OSError as exc:
        print(f"[spellcast] Reading the {what.lower()} ahead stopped: {exc}", flush=True)
        return
    print(f"[spellcast] {what} read ahead ({total / 2 ** 30:.1f} GB in {time.monotonic() - started:.0f} s).",
          flush=True)


class PathProblem(Exception):
    """A model file's path that stable-diffusion.cpp cannot open (see c_path); its text is for the artist."""


def c_path(path: Path) -> bytes:
    """A model file's path for stable-diffusion.cpp's char*. It opens files with the ANSI code page in some places
    (CreateFileA, GetFileAttributesA) and as UTF-8 in others (ggml's fopen, std::filesystem::u8path), so only a plain
    ASCII path works everywhere: the path as it is; else relative to the current folder (the sdcpp folder, in the same
    Spellcast3D folder as the models, so letters outside plain English are in the part both share); else its 8.3 short
    form. PathProblem when none of them is plain ASCII."""
    text = str(path)
    if text.isascii() or os.name != "nt":
        return text.encode("utf-8")
    try:
        rel = os.path.relpath(text)
    except ValueError:                  # on another drive
        rel = ""
    if rel and rel.isascii():
        return rel.encode("ascii")
    buf = ctypes.create_unicode_buffer(32_768)
    if ctypes.windll.kernel32.GetShortPathNameW(text, buf, len(buf)) and buf.value.isascii():
        return buf.value.encode("ascii")
    raise PathProblem(f"Qwen-Image cannot open its model files in {path.parent}: the folder's path has letters "
                      f"outside plain English, which its engine (stable-diffusion.cpp) cannot read there. Move the "
                      f"Spellcast3D folder (with its models) to a folder whose path has only plain letters and "
                      f"numbers, for example C:\\Spellcast3D, then start it again.")


def quiet_crashes() -> None:
    """A CUDA error inside ggml calls abort(): exit at once with the message in the log, no Windows error dialog
    that would keep the process (and the hub's restart) waiting."""
    faulthandler.enable()
    if os.name != "nt":
        return
    try:
        ucrt = ctypes.CDLL("ucrtbase")
        ucrt._set_abort_behavior(0, 0x1 | 0x2)              # _WRITE_ABORT_MSG | _CALL_REPORTFAULT off
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    except (OSError, AttributeError):
        pass


def model_sets(files: dict[str, Path | None]) -> dict[str, dict[str, Path | None]]:
    """Each model's context files from model_files' answer: {model: {"diffusion", "vae", "llm", "vision"}} - Qwen-Image's
    two share their VAE, text encoder and vision part; FLUX.3 has its own VAE and text encoder and no vision part."""
    sets: dict[str, dict[str, Path | None]] = {
        name: {"diffusion": files.get(name), "vae": files.get("vae"), "llm": files.get("llm"),
               "vision": files.get("vision")} for name in QWEN_MODELS}
    sets["flux3"] = {"diffusion": files.get("flux3"), "vae": files.get("flux3_vae"), "llm": files.get("flux3_llm"),
                     "vision": None}
    return sets


class QwenEngine:
    """stable-diffusion.cpp with one of the models loaded (MODELS: Qwen-Image 2.1's Turbo or full one, or FLUX.3 Image
    Klein), kept loaded while the stage runs; use_model() switches to another one (frees this context, then makes the
    new one), and generate() makes a request's pictures. load(), use_model() and generate() run on the GPU worker
    thread; cancel() comes from the HTTP side.

    files: {"turbo", "full", "flux3" (the denoisers' paths, whether the files are there or not), "vae", "llm",
    "vision" (Qwen-Image's; vision None when missing: no example pictures for it), "flux3_vae", "flux3_llm"}
    (model_files); model: the one load() loads first (start_model); flux3_engine: whether this engine build draws
    FLUX.3 (flux3_engine_choice: else its requests are refused, model_problem)."""

    # What an engine made without __init__ has (the profiler's tests subclass it so): no models known, so no switch
    # and no file check (use_model, model_problem). __init__ sets both.
    model: str | None = None
    denoisers: dict[str, Path | None] = {}
    sets: dict[str, dict[str, Path | None]] = {}
    flux3_engine = False
    albedo_files: dict[str, Path] | None = None      # the albedo add-on, when it is there and the engine runs it
    _ahead: list[Path] | None = None                 # the files to read ahead once the engine says PIN_MARK
    _reading: threading.Event | None = None          # gives up the read-ahead under way

    def __init__(self, lib: Any, files: dict[str, Path | None], placement: dict[str, Any], progress: Progress,
                 verbose: bool = False, model: str = DEFAULT_MODEL, flux3_engine: bool = False) -> None:
        self.lib, self.placement, self.progress = lib, placement, progress
        self.sets = model_sets(files)
        self.denoisers: dict[str, Path | None] = {name: self.sets[name]["diffusion"] for name in MODELS}
        self.model = model                       # the one loaded (or being loaded first)
        self.flux3_engine = flux3_engine
        # the context's files: the loaded model's set (its denoiser as "diffusion": context_params)
        self.files: dict[str, Path | None] = dict(self.sets[model])
        self.verbose = verbose
        self.ctx: int | None = None
        self.ready = False
        self.loading = False
        self.load_seconds: float | None = None
        self.has_vision = files.get("vision") is not None
        self._ctx_lock = threading.Lock()            # self.ctx between a switch (free) and cancel() (HTTP side)
        self._ahead_lock = threading.Lock()          # self._ahead between load() and the log callback
        self._keep: list[Any] = []                   # the context's strings: sd.cpp keeps pointers to them
        self._log_cb = sdcpp.LOG_CALLBACK(self._on_log)
        self._progress_cb = sdcpp.PROGRESS_CALLBACK(self._on_progress)
        self._preview_cb = sdcpp.PREVIEW_CALLBACK(self._on_preview)     # kept here: sd.cpp only holds its address
        lib.sd_set_log_callback(self._log_cb, None)
        lib.sd_set_progress_callback(self._progress_cb, None)

    # -- callbacks (sd.cpp's threads; never raise into C)
    def _on_log(self, level: int, text: bytes | None, _data: Any) -> None:
        try:
            line = (text or b"").decode("utf-8", "replace").rstrip()
            if level >= sdcpp.SD_LOG_INFO or self.verbose:
                print(f"[sd.cpp{LOG_LEVELS.get(level, '')}] {line}", flush=True)
            if self._ahead and PIN_MARK in line:
                self._read_ahead()
            self.progress.on_log(level, line)
            if self.progress.cancel.is_set() and self.ctx:
                self.lib.sd_cancel_generation(self.ctx, sdcpp.SD_CANCEL_ALL)
        except Exception:
            pass

    def _on_progress(self, step: int, steps: int, seconds: float, _data: Any) -> None:
        try:
            self.progress.on_step(step, steps, seconds)
            if self.progress.cancel.is_set() and self.ctx:
                self.lib.sd_cancel_generation(self.ctx, sdcpp.SD_CANCEL_ALL)
        except Exception:
            pass

    def _on_preview(self, step: int, count: int, frames: Any, noisy: bool, _data: Any) -> None:
        """sd.cpp's small picture of a step's result (the generating thread, between two steps): copied out, since
        sd.cpp frees it right after, and kept as a PNG."""
        try:
            if count < 1 or noisy or not frames[0].data or frames[0].channel not in (3, 4):
                return
            frame = frames[0]
            raw = ctypes.string_at(frame.data, frame.width * frame.height * frame.channel)
            self.progress.on_preview(step, encode_preview(raw, frame.width, frame.height, self._preview_alpha))
        except Exception:
            pass

    _previewing = False             # whether sd.cpp has the preview callback now
    _preview_alpha = False          # whether the request being drawn asked for a transparent background

    def _preview(self, wanted: bool) -> None:
        """Have sd.cpp hand over a preview picture after every step (PREVIEW_PROJ of its denoised estimate: the
        module doc), or none at all. It reads this when a picture's sampling starts."""
        if wanted == self._previewing:
            return
        if wanted:
            self.lib.sd_set_preview_callback(self._preview_cb, sdcpp.PREVIEW_PROJ, 1, True, False, None)
        else:
            self.lib.sd_set_preview_callback(sdcpp.PREVIEW_CALLBACK(), sdcpp.PREVIEW_NONE, 1, True, False, None)
        self._previewing = wanted

    # -- loading
    def context_params(self) -> Any:
        p, _ = sdcpp.init_struct(self.lib.sd_ctx_params_init, sdcpp.SdCtxParams)

        def keep(value: bytes | None) -> bytes | None:
            if value is not None:
                self._keep.append(value)
            return value

        p.diffusion_model_path = keep(c_path(self.files["diffusion"]))
        p.vae_path = keep(c_path(self.files["vae"]))
        p.llm_path = keep(c_path(self.files["llm"]))
        if self.files.get("vision") is not None:
            p.llm_vision_path = keep(c_path(self.files["vision"]))
        p.diffusion_flash_attn = True
        p.sage_attn = self.attention() == "sage"                       # see ATTENTIONS
        p.vae_conv_direct = bool(self.placement.get("vae_direct"))     # see vae_direct
        p.conditioning_cache_size = CONDITIONING_CACHE
        # the prefix cache's model args are Qwen-Image 2.1's: none for FLUX.3
        args = model_args(self.placement) if self.model in QWEN_MODELS else None
        p.model_args = keep(args.encode("ascii") if args else None)
        if self.placement.get("params_backend"):
            p.params_backend = keep(self.placement["params_backend"].encode("ascii"))
        if self.placement.get("max_vram"):
            p.max_vram = keep(str(self.placement["max_vram"]).encode("ascii"))
        p.eager_load = bool(self.placement.get("eager_load", True))
        p.enable_mmap = bool(self.placement.get("mmap", True))
        return p

    def attention(self) -> str:
        """The loaded model's attention: placement "attention" for Qwen-Image's models, "flux3_attention" for FLUX.3
        (flux3_attention_choice; flash attention when the placement does not say)."""
        if self.model == "flux3":
            return self.placement.get("flux3_attention") or FLUX3_ATTENTION_AUTO
        return self.placement.get("attention") or "flash"

    def _read_ahead(self) -> None:
        """Start reading the files that wait for it ahead (once per load)."""
        with self._ahead_lock:
            paths, self._ahead = self._ahead, None
        if paths:
            self._reading = stop = threading.Event()
            threading.Thread(target=read_ahead, args=(paths, stop), name="qwen-read-ahead", daemon=True).start()

    def load(self, model: str | None = None, words: str | None = None) -> None:
        """A context with `model`'s files (None: the one set, as at the start): its denoiser, and its VAE and text
        encoder (Qwen-Image's two share theirs). Its files are read ahead in the background meanwhile (PIN_MARK:
        from the moment the engine has its page-locked memory; at once when nothing is page-locked; after the load
        with an engine that never says so). words: the progress's loading words of a request's switch
        (Progress.loading)."""
        if model is not None:
            self.model = model
            self.files = dict(self.sets[model]) if self.sets else {**self.files, "diffusion": self.denoisers[model]}
        self.loading = True
        self.progress.loading(words)
        started = time.monotonic()
        if self._reading is not None:
            self._reading.set()                     # the model before's: its files are no longer wanted
        if self.placement.get("mmap", True):
            self._ahead = read_order(self.files, self.placement.get("pin_params"))
            if not self.placement.get("pin_params"):
                self._read_ahead()
        params = self.context_params()
        self._keep.append(params)
        try:
            ctx = self.lib.new_sd_ctx(ctypes.byref(params))
            lines = list(self.progress.lines)
        finally:
            self.loading = False
            self.progress.stop()
        if not ctx:
            reason = " ".join(lines[-3:]) or "no reason given"
            raise (OutOfMemory if ran_out_of_memory(lines) else GenerationFailed)(f"new_sd_ctx failed: {reason}")
        with self._ctx_lock:
            self.ctx = ctx
        self.load_seconds = round(time.monotonic() - started, 1)
        version = (self.lib.sd_get_model_version_name(ctx) or b"").decode("ascii", "replace")
        diffusion = self.files["diffusion"]
        print(f"[spellcast] {model_name(self.model, start=True)} loaded in {self.load_seconds:.0f} s ({version}; "
              f"{diffusion.name if diffusion is not None else '?'}; weights "
              f"{'on the GPU' if self.placement['name'] == 'gpu' else 'in RAM, copied to the GPU as needed'}).",
              flush=True)
        self.ready = True
        self._read_ahead()

    def model_on_disk(self, model: str) -> bool:
        """Whether `model`'s files are in the models folders now: its denoiser, and FLUX.3's own VAE and text encoder
        (the full model and FLUX.3 may arrive while this runs: the Images page's downloads)."""
        return not self.missing_files(model)

    def missing_files(self, model: str) -> list[str]:
        """The names of `model`'s files that are not there now: its denoiser (Qwen-Image's shared files were checked
        at the start), or any of FLUX.3's three."""
        files = self.sets.get(model) or {"diffusion": self.denoisers.get(model)}
        wanted = ("diffusion", "vae", "llm") if model == "flux3" else ("diffusion",)
        return [path.name if path is not None else model for key in wanted
                if (path := files.get(key)) is None or not path.is_file()]

    def can_draw(self, model: str) -> bool:
        """Whether a request for `model` can be drawn now: its files are there, and FLUX.3 only with an engine that
        draws it (health's "models")."""
        return self.model_on_disk(model) and (model != "flux3" or bool(self.flux3_engine))

    def free(self) -> None:
        """Free the context (its video memory, its page-locked weights, the prefix and conditioning caches). The
        pointer goes first, under the lock cancel() reads it with, so no cancel reaches a freed context."""
        with self._ctx_lock:
            ctx, self.ctx = self.ctx, None
        if ctx:
            self.lib.free_sd_ctx(ctx)
        self._keep.clear()                      # the freed context's strings (sd.cpp kept pointers to them)

    def use_model(self, model: str) -> float | None:
        """Make `model` the loaded one, on the GPU thread at the start of a request -> the switch's seconds, or None
        when it is loaded already. A switch frees the loaded context FIRST and then makes the new one: never two at
        once (an 8 GB card and a PC with 16 GB of RAM hold one); the prefix and conditioning caches go with it. When
        the new one cannot be loaded, the one before is loaded again and the error raised all the same; when even
        that fails there is no context any more (ready false), and the request's answer has the hub start this
        process again (build_app)."""
        if self.model is None or (model == self.model and self.ctx):
            return None
        before = self.model
        started = time.monotonic()
        name = model_name(model)
        print(f"[spellcast] Switching to {name} ({self.denoisers[model].name if self.denoisers[model] else '?'}) "
              f"from {model_name(before)}: freeing it first (RAM free {_ram_words()}).", flush=True)
        self.free()
        print(f"[spellcast] Freed in {time.monotonic() - started:.1f} s (RAM free {_ram_words()}).", flush=True)
        try:
            self.load(model, words=f"Loading {name}")
        except Exception as exc:
            traceback.print_exc()
            print(f"[spellcast] {model_name(model, start=True)} could not be loaded: loading {model_name(before)} "
                  f"again.", flush=True)
            try:
                self.load(before, words=f"Loading {model_name(before)} again")
            except Exception:
                traceback.print_exc()
                self.ready = False
                print("[spellcast] No model could be loaded again: Qwen-Image stops, and the hub starts it again for "
                      "the next pictures.", flush=True)
            if model == "flux3" and isinstance(exc, GenerationFailed):
                # e.g. an engine build taken to draw it (--flux3-engine on) that cannot: said as such
                raise GenerationFailed(f"{FLUX3_NOT_LOADED} ({exc})") from exc
            raise
        seconds = round(time.monotonic() - started, 1)
        print(f"[spellcast] Switched to {name} in {seconds:.1f} s (RAM free {_ram_words()}).", flush=True)
        return seconds

    def cancel(self) -> None:
        self.progress.cancel.set()
        with self._ctx_lock:
            ctx = self.ctx
            if ctx:
                self.lib.sd_cancel_generation(ctx, sdcpp.SD_CANCEL_ALL)

    # -- one request
    def generate(self, req: GenerateRequest, seed: int, references: list[Any], init: Any = None,
                 mask: Any = None) -> list[Any]:
        """The request's pictures as PIL images (RGBA), in seed order. references: PIL RGBA images, in order; init
        (RGB) and mask (L): img2img's start picture and what of it is drawn anew (start_pictures).
        Cancelled: the pictures sd.cpp finished before it stopped (it decodes all of them at the end, so only a
        cancel during that last phase leaves some), or Cancelled when there are none."""
        from PIL import Image

        lib = self.lib
        # how the request's model is drawn (the Turbo: its fixed sigmas, whatever the request says)
        plan = sampling(req.model, req.steps, req.cfg_scale, req.sampler, req.first_block_cache,
                        self.placement.get("capability", (99, 0)))
        method = lib.str_to_sample_method(plan["sampler"].encode("ascii", "replace"))
        g, _ = sdcpp.init_struct(lib.sd_img_gen_params_init, sdcpp.SdImgGenParams)
        keep: list[Any] = []
        transparent = req.transparent and draws_alpha(req.model)            # never FLUX.3's: no alpha channel
        g.prompt = wrap_prompt(req.prompt, transparent).encode("utf-8")
        # sd.cpp reads it only with guidance (the Turbo and FLUX.3 have none: never sent)
        g.negative_prompt = (req.negative_prompt.strip() if req.model == "full" else "").encode("utf-8")
        g.width, g.height = req.width, req.height
        # never while a mask holds the drawing to a picture: wrapped round, the model draws a poor change and lays
        # the picture some pixels off (to_the_middle is what a tileable picture gets then)
        g.circular_x = g.circular_y = req.tileable and req.model in QWEN_MODELS and mask is None
        g.sample_params.sample_method = method
        g.sample_params.sample_steps = plan["steps"]
        g.sample_params.guidance.txt_cfg = plan["cfg_scale"]                # exactly 1.0: one condition per step
        if plan["sigmas"] is not None:
            # the schedule as given (sd.cpp: len - 1 steps, and sample_steps must say as many): kept alive in `keep`
            # until generate_image has returned
            sigmas = (ctypes.c_float * len(plan["sigmas"]))(*plan["sigmas"])
            keep.append(sigmas)
            g.sample_params.custom_sigmas = ctypes.cast(sigmas, ctypes.POINTER(ctypes.c_float))
            g.sample_params.custom_sigmas_count = len(plan["sigmas"])
        g.seed = noise_seed(seed, drawn_from(references, init))
        if g.seed != seed:
            print(f"[spellcast] its noise from seed {g.seed}: the request's seed moved on by its pictures.", flush=True)
        g.batch_count = req.batch_count
        if references:
            # Qwen-Image's models take the pictures as they are, RGBA (load_reference); FLUX.3 takes RGB: a cut-out
            # laid on white (opaque_reference), since the engine would just drop the alpha channel
            pictures = references if draws_alpha(req.model) else [opaque_reference(p) for p in references]
            # Qwen-Image's example pictures are sized for the prefix cache's budget (its layout), FLUX.3's so that
            # together they stay within a few pictures' worth of tokens (flux3_reference_pixels)
            cap = copy_cap(self.placement, req, len(references))
            if req.model in QWEN_MODELS:
                pictures = qwen_copies(req, pictures, cap)
                g.ref_image_args = b"resize_before_vae=0"       # each as it was sized here
                print(f"[spellcast] {len(pictures)} example picture(s), Qwen-Image's own copies of them: "
                      + ", ".join(f"{p.width}x{p.height}" for p in pictures) + " pixels"
                      + (" (as asked)." if req.ref_sides else " (so the prefix cache fits)." if cap else "."), flush=True)
            elif cap is not None:
                g.ref_image_args = f"vae_input_max_pixels={cap}".encode("ascii")
                print(f"[spellcast] {len(references)} example pictures: each one's VAE copy at most "
                      f"{int(cap ** 0.5)}x{int(cap ** 0.5)} pixels.", flush=True)
            array = (sdcpp.SdImage * len(pictures))()
            for slot, picture in zip(array, pictures):
                data = picture.tobytes()
                buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
                keep.append(buf)
                slot.width, slot.height, slot.channel = picture.width, picture.height, len(picture.getbands())
                slot.data = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))
            keep.append(array)
            g.ref_images = ctypes.cast(array, ctypes.POINTER(sdcpp.SdImage))
            g.ref_images_count = len(pictures)
        for field, picture in (("init_image", init), ("mask_image", mask)):
            if picture is None:
                continue
            data = picture.tobytes()
            buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
            keep.append(buf)
            slot = getattr(g, field)
            slot.width, slot.height, slot.channel = picture.width, picture.height, len(picture.getbands())
            slot.data = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))
        if init is not None:
            g.strength = req.strength
        budget, direct = self.placement["budget_mb"], bool(self.placement.get("vae_direct"))
        if needs_vae_tiling(req.width, req.height, budget, direct):
            g.vae_tiling_params.enabled = True
            g.vae_tiling_params.tile_size_w = g.vae_tiling_params.tile_size_h = vae_tile(budget, direct)

        out = ctypes.POINTER(sdcpp.SdImage)()
        count = ctypes.c_int(0)
        if self.progress.cancel.is_set():
            raise Cancelled()
        self._preview_alpha = transparent
        self._preview(req.preview)
        # the engine reads the cache's skip share at the start of every sampling run (its model arg
        # qwen_image_2_1_first_block_cache sets the default; the variable wins when set): "0" for the Turbo and FLUX.3,
        # always (so a value of a full-model picture before cannot reach them)
        os.environ["SD_QWEN_FIRST_BLOCK_CACHE"] = FIRST_BLOCK_CACHE_SKIP if plan["first_block_cache"] else "0"
        # read by the engine at the start of every sampling run too (an engine build without it ignores it)
        os.environ[PREFIX_DROP_ENV] = "0" if req.keep_prefix else "1"
        try:
            ok = lib.generate_image(self.ctx, ctypes.byref(g), ctypes.byref(out), ctypes.byref(count))
        finally:
            self._preview(False)
        pictures = []
        try:
            for i in range(count.value if out else 0):
                img = out[i]
                if not img.data or img.channel not in (3, 4):
                    continue
                size = img.width * img.height * img.channel
                mode = "RGBA" if img.channel == 4 else "RGB"
                pictures.append(Image.frombytes(mode, (img.width, img.height),
                                                ctypes.string_at(img.data, size)).convert("RGBA"))
        finally:
            if out:
                lib.free_sd_images(out, count.value)
        del keep
        self.served = True
        if self.progress.cancel.is_set() and not pictures:
            raise Cancelled()       # pictures finished before the cancel (during the decode at the end) are kept
        if not pictures or (not ok and not self.progress.cancel.is_set()):
            lines = list(self.progress.lines)
            if ran_out_of_memory(lines):
                raise OutOfMemory(" ".join(lines[-4:]))
            raise GenerationFailed(" ".join(lines[-4:]) or "stable-diffusion.cpp made no picture")
        return pictures

    def albedo(self, picture: Any) -> Any:
        """A picture (PIL, sides multiples of SIDE_STEP) -> its albedo (PIL RGB, gamma 2.2) through the add-on: one
        pass of the loaded denoiser with the add-on's LoRA, prompt and decoder (sd_albedo). The engine keeps the
        add-on's files after the first call and leaves the drawing's caches as they were."""
        from PIL import Image

        if not self.albedo_files:
            raise InputProblem(ALBEDO_MISSING)
        lib = self.lib
        p, _ = sdcpp.init_struct(lib.sd_albedo_params_init, sdcpp.SdAlbedoParams)
        rgb = picture.convert("RGB")
        data = rgb.tobytes()
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        p.image.width, p.image.height, p.image.channel = rgb.width, rgb.height, 3
        p.image.data = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))
        paths = [c_path(self.albedo_files[key]) for key in ("lora", "condition", "vae")]
        p.lora_path, p.condition_path, p.decoder_path = paths
        out = lib.sd_albedo(self.ctx, ctypes.byref(p))
        del buf, paths
        if not out or not out[0].data or out[0].channel not in (3, 4):
            if out:
                lib.free_sd_images(out, 1)
            lines = list(self.progress.lines)
            if ran_out_of_memory(lines):
                raise OutOfMemory(" ".join(lines[-4:]))
            raise GenerationFailed(" ".join(lines[-4:]) or "stable-diffusion.cpp made no albedo")
        try:
            img = out[0]
            mode = "RGBA" if img.channel == 4 else "RGB"
            self.served = True
            return Image.frombytes(mode, (img.width, img.height),
                                   ctypes.string_at(img.data, img.width * img.height * img.channel)).convert("RGB")
        finally:
            lib.free_sd_images(out, 1)

    served = False                  # a picture or an albedo was made: the first use is paid (warm_omnitexture)
    warm_seconds: float | None = None

    def warm_omnitexture(self, albedo: bool = True) -> float | None:
        """Pay the first use of an Omni-Texture view now, while nobody waits (the hub asks when it started this process
        early for Omni-Texture): a WARM_SIDE picture drawn from two example pictures, a start picture and a mask as a
        view's, then an albedo (the add-on's files read, its weights staged), so the first view of the job runs as
        fast as the others. What it leaves in the engine's caches is keyed by its own pictures and words, which no
        request has; the albedo's decode remembers the tiles of the last size only, and a view's is another.
        -> its seconds, or None when a request came first (it paid that) or FLUX.3 is the model loaded."""
        from PIL import Image

        if self.served or self.warm_seconds is not None or not self.ctx or self.model not in QWEN_MODELS:
            return None
        self.warm_seconds = 0.0
        started = time.monotonic()
        side, middle = WARM_SIDE, (WARM_SIDE // 4, WARM_SIDE // 4, 3 * WARM_SIDE // 4, 3 * WARM_SIDE // 4)
        grey = Image.new("RGBA", (side, side), (128, 128, 128, 255))
        marked = Image.new("RGBA", (side, side), (0, 0, 0, 255))
        marked.paste((255, 255, 255, 255), middle)
        start = Image.new("RGB", (side, side), (96, 96, 96))
        mask = Image.new("L", (side, side), 0)
        mask.paste(255, middle)
        req = GenerateRequest(prompt=WARM_PROMPT, width=side, height=side, model=self.model, steps=1,
                              cfg_scale=CFG_SCALE, seed=0, transparent=False, first_block_cache=False, out_dir=".",
                              strength=0.9)
        try:
            self.generate(req, 0, [grey, marked], init=start, mask=mask)
            if albedo and self.albedo_files:
                self.albedo(start)
        finally:
            self.served = False
            self.warm_seconds = round(time.monotonic() - started, 1)
        return self.warm_seconds


# ---- pictures in and out (Pillow) ----------------------------------------------------------------------------------

def open_picture(path: Path) -> Any:
    """A picture file decoded (PIL; the file is closed again)."""
    from PIL import Image

    picture = Image.open(path)
    try:
        picture.load()
    except BaseException:
        picture.close()
        raise
    return picture


class Pictures:
    """The pictures requests read: each file of a request decoded once (Omni-Texture's start picture and mask are
    two of its example pictures as well), and the example pictures as load_reference makes them kept between
    requests while the file stays the same (its path, file id, modification time and size: Omni-Texture sends a
    prop's own pictures with each of its views), the most recently used up to REFERENCE_CACHE_MB. A request's start
    picture and mask are not kept: they are new each time."""

    def __init__(self, limit_mb: int = REFERENCE_CACHE_MB) -> None:
        self.limit = limit_mb * 2 ** 20
        self._kept: collections.OrderedDict[tuple[str, int, int, int], Any] = collections.OrderedDict()
        self._bytes = 0

    def read(self, req: GenerateRequest) -> tuple[list[Any], Any, Any]:
        """-> the request's example pictures (load_reference, in order), its start picture and mask
        (start_pictures)."""
        files: dict[str, Any] = {}

        def decoded(path: Path) -> Any:
            if str(path) not in files:
                files[str(path)] = open_picture(path)
            return files[str(path)]

        own = {str(Path(p)) for p in (req.init_image, req.mask_image) if p}
        references = []
        for ref in req.ref_images:
            path = Path(ref)
            try:
                stat = path.stat()
                key = (str(path), stat.st_ino, stat.st_mtime_ns, stat.st_size)
            except OSError:
                key = None
            if key is not None and key in self._kept:
                self._kept.move_to_end(key)
                references.append(self._kept[key])
                continue
            picture = load_reference(path, decoded)
            references.append(picture)
            if key is not None and str(path) not in own:
                self._keep(key, picture)
        return references, *start_pictures(req, decoded)

    def _keep(self, key: tuple[str, int, int, int], picture: Any) -> None:
        size = picture.width * picture.height * len(picture.getbands())
        if size > self.limit:
            return
        self._kept[key] = picture
        self._bytes += size
        while self._bytes > self.limit:
            _key, old = self._kept.popitem(last=False)
            self._bytes -= old.width * old.height * len(old.getbands())


def load_reference(path: Path, decoded: Callable[[Path], Any] | None = None) -> Any:
    """An example picture as RGBA (its own transparency kept, an RGB one opaque), at most REFERENCE_MAX_SIDE.
    decoded(path): the file as open_picture gives it (Pictures: a request's files decoded once)."""
    from PIL import Image, ImageOps

    try:
        if decoded is not None:
            picture = ImageOps.exif_transpose(decoded(path))
            picture.load()
        else:
            with Image.open(path) as opened:
                picture = ImageOps.exif_transpose(opened)
                picture.load()
    except Exception as exc:
        raise InputProblem(f"The example picture {path.name} could not be read ({type(exc).__name__}). Use a PNG, "
                           f"JPG or WEBP picture.") from exc
    has_alpha = picture.mode in ("RGBA", "LA", "PA") or (picture.mode == "P" and "transparency" in picture.info)
    picture = picture.convert("RGBA" if has_alpha else "RGB").convert("RGBA")
    if max(picture.size) > REFERENCE_MAX_SIDE:
        scale = REFERENCE_MAX_SIDE / max(picture.size)
        picture = picture.resize((max(1, round(picture.width * scale)), max(1, round(picture.height * scale))),
                                 Image.Resampling.LANCZOS)
    return picture


def copy_size(size: tuple[int, int], side: int | None, pixels: int) -> tuple[int, int]:
    """The size of Qwen-Image's own copy of a picture of `size`: `side` px on its longer side or, without one,
    `pixels` in all - its shape kept, both sides multiples of REFERENCE_STEP, rounded as the engine rounds."""
    scale = side / max(size) if side else (pixels / (size[0] * size[1])) ** 0.5
    return tuple(max(REFERENCE_STEP, int(length * scale / REFERENCE_STEP + 0.5) * REFERENCE_STEP) for length in size)


def qwen_copies(req: GenerateRequest, pictures: list[Any], cap: int | None) -> list[Any]:
    """Qwen-Image's own copies of a request's example pictures (sized_reference): each of the side the request
    names for it, else with the pixels of the picture asked for, or `cap` where that is fewer."""
    pixels = min(cap or req.width * req.height, req.width * req.height)
    return [sized_reference(p, side, pixels) for p, side in zip(pictures, req.ref_sides or [None] * len(pictures))]


def sized_reference(picture: Any, side: int | None, pixels: int) -> Any:
    """Qwen-Image's own copy of an example picture (copy_size). The engine then takes it as it is (ref_image_args
    resize_before_vae=0): left to itself it brings every picture to `pixels` by nearest neighbour, which makes a
    coarse copy of a larger picture, lying a pixel or so off."""
    size = copy_size(picture.size, side, pixels)
    if size == picture.size:
        return picture
    from PIL import Image

    return picture.resize(size, Image.Resampling.LANCZOS)


def start_pictures(req: GenerateRequest, decoded: Callable[[Path], Any] | None = None) -> tuple[Any, Any]:
    """img2img's start picture (RGB, the request's size: transparent parts black, as the viewport shows them) and its
    mask (L: 255 drawn anew, 0 kept; anything in between rounded) - (None, None) without an init_image.
    A masked change (changed_image) by a model that draws see-through pictures starts from the picture as it is, RGBA:
    held to black where the picture is see-through, the model draws a black ground around the new thing.
    decoded(path): a file as open_picture gives it (Pictures)."""
    if not req.init_image:
        return None, None
    from PIL import Image

    read = decoded or open_picture
    size = (req.width, req.height)
    try:
        picture = read(Path(req.init_image)).convert("RGBA")
        if req.changed_image and draws_alpha(req.model):
            start = picture
        else:
            start = Image.new("RGB", picture.size, (0, 0, 0))
            start.paste(picture, mask=picture.getchannel("A"))
        start = start.resize(size, Image.Resampling.LANCZOS) if start.size != size else start
        mask = None
        if req.mask_image:
            mask = read(Path(req.mask_image)).convert("L")
            mask = mask.resize(size, Image.Resampling.NEAREST) if mask.size != size else mask
            mask = mask.point(lambda v: 255 if v >= 128 else 0)
    except Exception as exc:
        raise InputProblem(f"The viewport's picture could not be read ({type(exc).__name__}).") from exc
    return start, mask


def opaque_reference(picture: Any) -> Any:
    """An example picture (RGBA, load_reference) as RGB, for a model without an alpha channel (FLUX.3): a transparent
    one is laid on white, as the page shows it. The engine would otherwise drop the alpha channel and show the model
    whatever colour the see-through pixels hold (a Qwen-Image cut-out: black, or a purple cast)."""
    return picture.convert("RGB") if picture.getextrema()[3][0] == 255 else on_white(picture)


def on_white(picture: Any) -> Any:
    """An RGBA picture laid on white -> RGB."""
    from PIL import Image

    return Image.alpha_composite(Image.new("RGBA", picture.size, (255, 255, 255, 255)), picture).convert("RGB")


def has_cutout(picture: Any) -> bool:
    """Whether part of the picture is see-through (CUTOUT_SHARE). Qwen-Image's models cut the object out of its
    background for some seeds without being asked: of a sheet of four views, 2 seeds of 6."""
    if picture.mode != "RGBA":
        return False
    below = sum(picture.getchannel("A").histogram()[:SEE_THROUGH_ALPHA])
    return below > CUTOUT_SHARE * picture.width * picture.height


def slid(picture: Any, shift: float) -> Any:
    """The picture moved by `shift` of its size to the right and down (negative: left and up), what leaves one side
    coming back at the other: the page's shiftedCanvas, to the pixel (its rounding)."""
    from PIL import ImageChops
    part = round(((shift % 1) + 1) % 1 * 1000) / 1000
    return ImageChops.offset(picture, int(part * picture.width + 0.5), int(part * picture.height + 0.5))


# A tileable picture made from another lies some pixels off it, another way each time: its positions are periodic,
# so nothing says where it begins.
PLACE_REACH = 48                   # px either way that are searched, at a quarter of the size, then to the pixel
PLACE_SURE = 0.8                   # the best place differs at most this much of the usual one: clearly its place,
#                                    not another picture's, nor a plain ground's


def drift(source: Any, picture: Any) -> tuple[int, int]:
    """How far `picture` lies from `source`, the picture it was made from: (dx, dy) whole pixels to the right and
    down, wrapping round, where their grey levels differ least; (0, 0) when no place within PLACE_REACH fits clearly
    better than the others (another picture altogether, or one moved further)."""
    from PIL import Image, ImageChops, ImageStat

    def apart(a: Any, b: Any, dx: int, dy: int) -> float:
        return ImageStat.Stat(ImageChops.difference(a, ImageChops.offset(b, -dx, -dy))).mean[0]

    now = picture.convert("L")
    was = source.convert("L")
    if was.size != now.size:
        was = was.resize(now.size, Image.Resampling.LANCZOS)
    small = (max(1, now.width // 4), max(1, now.height // 4))
    a, b = was.resize(small, Image.Resampling.BOX), now.resize(small, Image.Resampling.BOX)
    reach = range(-(PLACE_REACH // 4), PLACE_REACH // 4 + 1)
    coarse = {(dx, dy): apart(a, b, dx, dy) for dy in reach for dx in reach}
    (cx, cy), best = min(coarse.items(), key=lambda place: place[1])
    if best > PLACE_SURE * sorted(coarse.values())[len(coarse) // 2]:
        return 0, 0
    fine = {(dx, dy): apart(was, now, dx, dy) for dy in range(4 * cy - 3, 4 * cy + 4) for dx in range(4 * cx - 3, 4 * cx + 4)}
    return min(fine.items(), key=lambda place: place[1])[0]


def in_place(picture: Any, source: Any) -> Any:
    """A tileable picture laid where `source`, the picture it was made from, lies (drift): moved back by whole
    pixels, what leaves one side coming back at the other, so nothing of it is lost."""
    from PIL import ImageChops
    dx, dy = drift(source, picture)
    if not (dx or dy):
        return picture
    print(f"[spellcast] the picture lay {dx:+d}, {dy:+d} px off the one it was made from: laid back onto it.", flush=True)
    return ImageChops.offset(picture, -dx, -dy)


# A masked change (the page's Smart Mask, changed_image): a part of a picture is brushed over and words say what
# should change there. Drawn one of two ways:
# WHOLE: the model draws all of the picture anew, so the new thing has the whole scene to fit into (shown a window
#   of the picture alone it has no scale for it, and shown the whole picture beside the window it draws that
#   picture into the mask). Its tone is brought back to the picture's. Of a picture the model sees at its own
#   size all of that is kept; of a larger one a PATCH round the brush (patch_of), the rest being the picture's own
#   pixels: redrawn from a copy smaller than itself, its finest detail is another drawing's, softer, and it does
#   not lie on its picture - some pixels off and stretched a little, another amount each time. So it is brought
#   onto the picture where the two meet (fitted) and joined to it along the patch's sides by a noise threshold
#   (joined). redrawn_whole.
# HELD, a tileable picture, which drawn anew whole would no longer repeat: drawn in a ROOM around what was brushed, the
#   drawing held to the picture everywhere else. Of what the model drew there, what lies under the brush is kept,
#   and with it whatever it clearly CHANGED beside the brush - the new thing where it is larger than the stroke,
#   its shadow, its reflection, as far as the room goes. The picture itself is laid back everywhere else, its own
#   pixels (kept_outside). A tileable picture is moved for this so that what was brushed lies in its middle
#   (to_the_middle).
# Lengths are shares of the picture's longer side - the whole picture's, where a window of it is drawn; but
# CHANGE_BLUR of what the model draws, SIZE at the most: a ground drawn anew differs at the scale of the model's
# own pixels, and blurred by more a thing a few hundred pixels large goes all but untold.
KEEP_ROOM = 0.12                   # the room: room enough for a thing's own shape and for what it casts
CHANGE_REACH = 0.06                # a change that begins this near what was brushed belongs to it (reached_from)
CHANGE_BLUR = 0.02                 # clearly changed: the two pictures, blurred this far (ground drawn anew does not
CHANGE_LEVELS = 24                 # count), are further apart than this, of 255, in some channel
CHANGE_GROW = 0.02                 # what is kept is taken this much wider (a shadow's faint end) ...
KEEP_EDGE = 0.006                  # ... and meets the picture in a soft edge, as wide as this blur
# The model draws a little off the picture's tone. A curve per channel brings what it drew back, read on the ground
# it drew anew without changing it - where both pictures show the same thing, and neither is see-through - when
# that is enough of all it could be read on (toned).
TONE_KNOTS = (1, 5, 10, 25, 50, 75, 90, 95, 99)       # percent: the ranks matched, straight between them
TONE_GROUND = 0.4
KEPT_NOTHING = ("The picture was asked for in another shape than the one you brushed over, so all of it was drawn "
                "anew. Give it that picture's Shape to keep everything outside the mask.")
# Said when nothing changed clearly where the artist brushed: ground drawn anew and a slight change (a little
# darker, fine cracks) cannot be told apart, so what the model drew there is in the picture either way.
KEPT_ALL = ("Under the mask the picture changed only slightly, if at all. If you wanted more, render again, name "
            "the thing more plainly, or brush a larger place for it.")


def same_shape(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """A picture of size `a`, resized to the width of `b`, is as high as it to a pixel or two."""
    return abs(a[0] * b[1] - b[0] * a[1]) <= 2 * a[0]


def brushed(mask: Any, size: tuple[int, int]) -> Any:
    """What was brushed (`mask`: white) at `size`, black and white (L)."""
    from PIL import Image

    where = mask.convert("L")
    if where.size != size:
        where = where.resize(size, Image.Resampling.BILINEAR)
    return where.point(lambda v: 255 if v >= 128 else 0)


def spread(mask: Any, reach: int) -> Any:
    """`mask` (L, black and white), its white `reach` px wider every way - the square one MaxFilter of 2 * reach + 1
    gives, in steps of two: a rank filter that wide costs its whole area for every pixel."""
    from PIL import ImageFilter

    for _ in range(reach // 2):
        mask = mask.filter(ImageFilter.MaxFilter(5))
    return mask.filter(ImageFilter.MaxFilter(3)) if reach % 2 else mask


def with_room(brush: Any, size: tuple[int, int], side: int | None = None) -> Any:
    """What was brushed with KEEP_ROOM around it, at `size`: where a masked change may be drawn, black and white (L).
    side: the longer side of the picture the lengths are shares of (a window's whole picture; without one, the
    longer of `size`). Grown at an eighth of the size: a wide rim costs little there."""
    from PIL import Image

    small = brush.convert("L").resize((max(1, size[0] // 8), max(1, size[1] // 8)), Image.Resampling.BOX)
    grown = spread(small.point(lambda v: 255 if v else 0), max(1, round(KEEP_ROOM * (side or max(size)) / 8)))
    return grown.resize(size, Image.Resampling.BILINEAR).point(lambda v: 255 if v >= 128 else 0)


def changed_in(room: Any, picture: Any, rest: Any) -> Any:
    """White where `picture` shows something else than `rest` (both RGBA), from afar (CHANGE_BLUR), inside `room`
    (L). As far as each shows (RGBa): the colour left under a see-through part is no change, a thing drawn onto one
    is."""
    from PIL import ImageChops, ImageFilter

    blur = ImageFilter.GaussianBlur(CHANGE_BLUR * min(max(picture.size), SIZE))
    bands = ImageChops.difference(*(p.convert("RGBa").filter(blur) for p in (picture, rest))).split()
    most = bands[0]
    for band in bands[1:]:
        most = ImageChops.lighter(most, band)
    return ImageChops.multiply(most.point(lambda v: 255 if v > CHANGE_LEVELS else 0), room)


def reached_from(brush: Any, changed: Any, room: Any, side: int | None = None) -> Any:
    """What was brushed and the part of `changed` it reaches, or that begins within CHANGE_REACH of it - the new
    thing, also where the model set it beside a thin stroke, not another thing that happened to change further
    off -, a little wider, inside `room` (L). Spread at a quarter of the size, where it takes few rounds."""
    from PIL import Image, ImageChops, ImageFilter

    side = side or max(changed.size)
    small = (max(1, changed.width // 4), max(1, changed.height // 4))
    through, reached = (m.resize(small, Image.Resampling.BOX).point(lambda v: 255 if v else 0) for m in (changed, brush))
    near = spread(reached, max(1, round(CHANGE_REACH * side / 4)))
    reached = ImageChops.lighter(reached, ImageChops.multiply(near, through))
    while True:
        wider = ImageChops.lighter(reached, ImageChops.multiply(reached.filter(ImageFilter.MaxFilter(3)), through))
        if ImageChops.difference(wider, reached).getbbox() is None:
            break
        reached = wider
    grown = spread(reached, max(1, round(CHANGE_GROW * side / 4)))
    return ImageChops.multiply(grown.resize(changed.size, Image.Resampling.BILINEAR).point(lambda v: 255 if v >= 128 else 0), room)


def in_tone(picture: Any, rest: Any, ground: Any) -> Any:
    """`picture` with the tone `rest` has over `ground` (white, L): per colour channel each level goes to the level
    of the same rank there, straight between TONE_KNOTS. A level darker or lighter than any read on the ground - a
    new thing's own colours - moves only as far as the nearest knot does: the ground says no more about it."""
    from PIL import Image

    def ranked(histogram: list[int]) -> list[int]:
        levels, run, level = [], 0, 0
        for knot in TONE_KNOTS:
            while level < 255 and run + histogram[level] <= knot / 100 * (sum(histogram) - 1):
                run += histogram[level]
                level += 1
            levels.append(level)
        return levels

    def table(mine: list[int], theirs: list[int]) -> list[int]:
        out = []
        for level in range(256):
            if level <= mine[0]:
                to = level + theirs[0] - mine[0]
            elif level >= mine[-1]:
                to = level + theirs[-1] - mine[-1]
            else:
                i = max(k for k, knot in enumerate(mine) if knot <= level)
                to = theirs[i] + (level - mine[i]) * (theirs[i + 1] - theirs[i]) / (mine[i + 1] - mine[i])
            out.append(round(min(255.0, max(0.0, to))))
        return out

    bands = picture.split()
    colours = [band.point(table(ranked(band.histogram(ground)), ranked(other.histogram(ground))))
               for band, other in zip(bands[:3], rest.split()[:3])]
    return Image.merge(picture.mode, [*colours, *bands[3:]])


def tone_ground(picture: Any, rest: Any, within: Any, apart: Any) -> tuple[Any, float]:
    """Where the tone of `picture` can be read against `rest` (both RGBA): within `within` but for `apart` (L,
    white), where neither is see-through -> that ground (L, white) and its share of all within that is not
    see-through."""
    from PIL import ImageChops, ImageStat

    solid = ImageChops.multiply(*(p.getchannel("A").point(lambda v: 255 if v >= SEE_THROUGH_ALPHA else 0)
                                  for p in (picture, rest)))
    ground = ImageChops.multiply(ImageChops.subtract(within, apart), solid)
    of = ImageStat.Stat(ImageChops.multiply(within, solid)).sum[0]
    return ground, ImageStat.Stat(ground).sum[0] / of if of else 0.0


def toned(picture: Any, rest: Any, within: Any, apart: Any) -> Any:
    """`picture` in the tone of `rest` (in_tone), read on tone_ground when that is at least TONE_GROUND of all it
    could be read on; else as it is."""
    ground, share = tone_ground(picture, rest, within, apart)
    return in_tone(picture, rest, ground) if share >= TONE_GROUND else picture


def change_of(image: str, mask: str, size: tuple[int, int]) -> tuple[Any, Any] | None:
    """A masked change's picture (RGBA) and what was brushed over it (L, white), both at `size`; None for a picture
    of another shape (same_shape: nothing of it fits)."""
    from PIL import Image

    rest = open_picture(Path(image)).convert("RGBA")
    if not same_shape(rest.size, size):
        return None
    if rest.size != size:
        rest = rest.resize(size, Image.Resampling.LANCZOS)
    return rest, brushed(open_picture(Path(mask)), size)


# ---- a masked change drawn WHOLE ----------------------------------------------------------------------------------
PATCH_SIDE = 1024                  # a patch's sides at the least; a picture of up to that squared is all its patch's
PATCH_MARGIN = 0.05                # a patch stays this share of the picture inside the picture's sides, from afar ...
PATCH_SHY = 1.5                    # ... and half of it where what was brushed lies this many margins from a side
JOIN_BAND = 0.04                   # the width of a patch's join, a share of the picture's longer side
FIT_REACH = 12                     # px either way a redraw is looked for beside its picture (place_in)
WHOLE_GROUND = 0.02                # the least ground a whole redraw's tone is read on (whole_toned)


def patch_of(brushed: tuple[int, int, int, int] | None, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """The part of a whole redraw that is laid onto its picture - (left, top, right, bottom): what was brushed with
    its room each way, PATCH_SIDE a side at the least, round the middle of what was brushed. All of a picture of up
    to PATCH_SIDE squared pixels, which the model sees at its own size, and all of one along a side no longer than
    PATCH_SIDE. It stays off the picture's sides: PATCH_MARGIN where what was brushed lies far from a side, less and
    less the nearer it lies (kept_off) - it nears the side without a jump and is on it only where what was brushed
    is. The page shows it while the artist brushes (mask_window.js patchOf: the two change together, in the same
    order of sums)."""
    room = round(KEEP_ROOM * max(size))

    def kept_off(gap: int, margin: float) -> int:
        """How far a patch stays from a side of the picture that what was brushed lies `gap` from, whole pixels: all
        but `margin` from afar, never more than a third of the way, nothing where the two touch."""
        half = PATCH_SHY * margin
        return int(margin * gap * gap / (gap * gap + half * half))

    def along(low: int, high: int, full: int) -> tuple[int, int]:
        if full <= PATCH_SIDE:
            return 0, full
        margin = PATCH_MARGIN * full
        near, far = kept_off(low, margin), kept_off(full - high, margin)
        side = max(PATCH_SIDE, high - low + 2 * room)
        if side >= full - near - far:                   # no room for all of it: as much as stays off both sides
            return near, full - far
        left = min(max((low + high - side) // 2, near), full - far - side)
        return left, left + side

    if brushed is None or size[0] * size[1] <= PATCH_SIDE ** 2:
        return 0, 0, *size
    (left, right), (top, bottom) = (along(brushed[i], brushed[i + 2], size[i]) for i in (0, 1))
    return left, top, right, bottom


def join_band(size: tuple[int, int]) -> int:
    """How wide a patch's join is in a picture of `size`, px."""
    return max(4, round(JOIN_BAND * max(size)))


def joined(patch: tuple[int, int, int, int], brushed: tuple[int, int, int, int] | None, size: tuple[int, int],
           seed: int) -> Any:
    """Where a patch is laid onto its picture (L, white): all of it, but along its sides inside the picture, where
    over a band each pixel is the patch's or the picture's by a noise threshold - a ragged line, the same for a
    seed. No fade: two drawings of one thing, their finest detail apart, show double where they are mixed. The band
    is join_band wide and no wider than the way from that side to what was `brushed`: the new thing is all the
    patch's."""
    from PIL import Image, ImageChops, ImageFilter

    if brushed is None or patch == (0, 0, *size):
        return Image.new("L", size, 255)
    band = join_band(size)

    def rising(low: int, high: int, full: int, first: int, last: int) -> bytes:
        """Along one axis: black outside the patch (low .. high), white inside it past the band at each of its
        sides that is not the picture's own - what was brushed lies first .. last."""
        up, down = (min(band, first - low) if low > 0 else 0), (min(band, high - last) if high < full else 0)
        return bytes(round(255 * min(1, (at - low + 1) / up if up > 0 else 1, (high - at) / down if down > 0 else 1))
                     if low <= at < high else 0 for at in range(full))

    across, down = (Image.frombytes("L", line, rising(patch[i], patch[i + 2], size[i], brushed[i], brushed[i + 2]))
                    .resize(size, Image.Resampling.NEAREST) for i, line in ((0, (size[0], 1)), (1, (1, size[1]))))
    slope = ImageChops.darker(across, down)
    dice = random.Random(seed)

    def grain(cell: int) -> Any:
        small = (size[0] // cell + 2, size[1] // cell + 2)
        return Image.frombytes("L", small, dice.randbytes(small[0] * small[1])).resize(size, Image.Resampling.BICUBIC)

    # coarse grain and fine, below white: past the band the patch is solid whatever the dice say
    noise = ImageChops.add(grain(band // 4), grain(max(1, band // 16)), scale=2).point(lambda v: min(v, 254))
    return ImageChops.subtract(slope, noise).point(lambda v: 255 if v else 0).filter(ImageFilter.GaussianBlur(0.6))


def place_in(rest: Any, picture: Any, box: tuple[int, int, int, int], apart: Any) -> tuple[float, float] | None:
    """How far `picture` lies from `rest` inside `box` (both L) - (dx, dy) px to the right and down, to a fraction
    of a pixel, where their grey levels differ least, `apart` (L, white) left out. None when no place within
    FIT_REACH fits clearly better than the others (PLACE_SURE: a plain ground, or another drawing), or when most
    of the box is left out."""
    from PIL import Image, ImageChops, ImageStat

    left, top = max(box[0], FIT_REACH), max(box[1], FIT_REACH)
    width, height = ((min(high, full - FIT_REACH) - low) // 4 * 4
                     for low, high, full in ((left, box[2], rest.width), (top, box[3], rest.height)))
    if width <= 0 or height <= 0:
        return None
    was = rest.crop((left, top, left + width, top + height))
    now = picture.crop((left - FIT_REACH, top - FIT_REACH, left + width + FIT_REACH, top + height + FIT_REACH))
    told = apart.crop((left, top, left + width, top + height)).point(lambda v: 0 if v else 255)

    def quarter(part: Any) -> Any:
        return part.resize((part.width // 4, part.height // 4), Image.Resampling.BOX)

    def costs(a: Any, b: Any, where: Any, places: list[tuple[int, int]], reach: int) -> dict[tuple[int, int], float]:
        return {(dx, dy): ImageStat.Stat(ImageChops.difference(a, b.crop((reach + dx, reach + dy, reach + dx + a.width,
                                                                          reach + dy + a.height))), where).mean[0]
                for dx, dy in places}

    told_small = quarter(told).point(lambda v: 255 if v == 255 else 0)
    if ImageStat.Stat(told_small).mean[0] < 128:
        return None
    reach = FIT_REACH // 4
    coarse = costs(quarter(was), quarter(now), told_small,
                   [(dx, dy) for dy in range(-reach, reach + 1) for dx in range(-reach, reach + 1)], reach)
    (cx, cy), best = min(coarse.items(), key=lambda place: place[1])
    if best > PLACE_SURE * sorted(coarse.values())[len(coarse) // 2]:
        return None
    near = [(dx, dy) for dy in range(4 * cy - 4, 4 * cy + 5) for dx in range(4 * cx - 4, 4 * cx + 5)
            if max(abs(dx), abs(dy)) <= FIT_REACH]
    fine = costs(was, now, told, near, FIT_REACH)
    dx, dy = min(fine, key=lambda place: fine[place])
    if max(abs(dx), abs(dy)) >= FIT_REACH - 1:       # on the rim of what was searched: a slope of shade, no place
        return None

    def between(low: float | None, mid: float, high: float | None) -> float:
        """Where between its neighbours a lowest value lies, by the parabola through the three: -0.5 .. 0.5."""
        if low is None or high is None or low - 2 * mid + high <= 0:
            return 0.0
        return 0.5 * (low - high) / (low - 2 * mid + high)

    return (dx + between(fine.get((dx - 1, dy)), fine[dx, dy], fine.get((dx + 1, dy))),
            dy + between(fine.get((dx, dy - 1)), fine[dx, dy], fine.get((dx, dy + 1))))


def line_through(points: list[tuple[float, float]]) -> tuple[float, float]:
    """The straight line most of `points` ((at, value), ...) lie along -> (its value at 0, its slope): the middle
    one of the slopes between points well apart, then the middle of what that leaves - a few wrong points change
    nothing. Level where all lie at one place."""
    span = max(at for at, _ in points) - min(at for at, _ in points)
    slopes = [(other - value) / (there - at) for (at, value), (there, other) in itertools.combinations(points, 2)
              if abs(there - at) >= span / 4 > 0]
    slope = statistics.median(slopes) if slopes else 0.0
    return statistics.median(value - slope * at for at, value in points), slope


def along_the_sides(patch: tuple[int, int, int, int], side: int, size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """Boxes `side` px square, one after another along the four sides of `patch`, each half inside it and half
    outside, as far as they lie in a picture of `size`."""
    left, top, right, bottom = patch
    boxes = [(x, y - side // 2, x + side, y + side // 2) for y in (top, bottom) for x in range(left, right - side + 1, side)]
    boxes += [(x - side // 2, y, x + side // 2, y + side) for x in (left, right) for y in range(top, bottom - side + 1, side)]
    return [(max(0, box[0]), max(0, box[1]), min(size[0], box[2]), min(size[1], box[3])) for box in boxes]


def beyond_its_sides(picture: Any, by: int) -> Any:
    """`picture` with `by` px more on every side, its own last pixels repeated there."""
    from PIL import Image

    width, height = picture.size
    rows = Image.new(picture.mode, (width + 2 * by, height))
    rows.paste(picture.resize((by, height), Image.Resampling.NEAREST, (0, 0, 1, height)), (0, 0))
    rows.paste(picture, (by, 0))
    rows.paste(picture.resize((by, height), Image.Resampling.NEAREST, (width - 1, 0, width, height)), (width + by, 0))
    wide = Image.new(picture.mode, (width + 2 * by, height + 2 * by))
    wide.paste(rows.resize((rows.width, by), Image.Resampling.NEAREST, (0, 0, rows.width, 1)), (0, 0))
    wide.paste(rows, (0, by))
    wide.paste(rows.resize((rows.width, by), Image.Resampling.NEAREST, (0, height - 1, rows.width, height)), (0, height + by))
    return wide


def fitted(picture: Any, rest: Any, patch: tuple[int, int, int, int], apart: Any) -> Any:
    """`picture`, a whole redraw of `rest` (both RGBA), brought onto it where the two are joined: in boxes along the
    sides of `patch` it is told how far the redraw lies off its picture (place_in, `apart` left out), and the
    redraw is moved and stretched along each axis by the line through what they tell (line_through); its own last
    pixels stand where it then ends short of the picture's side. As it is where nothing is joined (the patch is
    the whole picture) or no box tells its place."""
    from PIL import Image

    if patch == (0, 0, *rest.size):
        return picture
    was, now = rest.convert("L"), picture.convert("L")
    told = [(box, place) for box in along_the_sides(patch, 2 * join_band(rest.size), rest.size)
            if (place := place_in(was, now, box, apart)) is not None]
    if not told:
        return picture
    (dx, wider), (dy, taller) = (line_through([((box[axis] + box[axis + 2]) / 2, place[axis]) for box, place in told])
                                 for axis in (0, 1))
    print(f"[spellcast] the redraw lay {dx + wider * rest.width / 2:+.1f}, {dy + taller * rest.height / 2:+.1f} px off "
          f"its picture, and {wider * rest.width:+.1f}, {taller * rest.height:+.1f} px wider and taller: brought onto it.",
          flush=True)
    pad = 2 + math.ceil(max(abs(dx), abs(dx + wider * rest.width), abs(dy), abs(dy + taller * rest.height)))
    # the redraw's (1 + wider) * x + dx is the picture's x; told of pixels' middles, which lie half a pixel in
    onto = (1 + wider, 0, dx - wider / 2 + pad, 0, 1 + taller, dy - taller / 2 + pad)
    # band by band: resampled as RGBA, the colour left under a see-through part would be lost
    return Image.merge("RGBA", [band.transform(rest.size, Image.Transform.AFFINE, onto, Image.Resampling.BICUBIC)
                                for band in beyond_its_sides(picture, pad).split()])


def whole_toned(picture: Any, rest: Any, *aparts: Any) -> Any:
    """`picture`, a whole redraw of `rest`, in its tone (in_tone), read on all of it but for one of `aparts` (L,
    white): the first that leaves TONE_GROUND of the picture to read on, else the last, if that leaves
    WHOLE_GROUND - under a large mask little is left, and that is what the patch is joined to. Else as it is."""
    from PIL import Image

    everywhere = Image.new("L", picture.size, 255)
    for apart in aparts:
        ground, share = tone_ground(picture, rest, everywhere, apart)
        if share >= TONE_GROUND:
            break
    return in_tone(picture, rest, ground) if share >= WHOLE_GROUND else picture


def redrawn_whole(picture: Any, rest: Any, brush: Any, patch: tuple[int, int, int, int], seed: int) -> tuple[Any, bool]:
    """A picture the model drew anew, all of it, from `rest` with a change where `brush` is white -> what is kept of
    it: in the tone of `rest`, brought onto `rest` (fitted), and of it the `patch` (joined) - and beyond the patch
    whatever it clearly changed from what was brushed on (reached_from): a thing larger than its mask where the
    patch's side is close, near the picture's own, a shadow longer than the patch. Where the picture shows through
    an answer that came back see-through unasked, the picture stays. And whether the model clearly changed
    anything in the room - free to draw anywhere, it often sets a thing beside the stroke."""
    from PIL import Image, ImageChops, ImageFilter

    picture = picture.convert("RGBA")
    everywhere = Image.new("L", picture.size, 255)
    near = with_room(brush, picture.size)
    # What changed, roughly: told with the tone read away from the brush's room (as it comes, a redraw lies so far
    # off its picture's tone that ground it did not change all but counts as changed) and left out where the tone
    # is read for good - away from the room too, where a faint shadow would bend it - and where its place is told.
    rough = changed_in(everywhere, whole_toned(picture, rest, near, brush), rest)
    picture = whole_toned(picture, rest, ImageChops.lighter(near, rough), rough)
    if not has_transparency(rest):
        picture = Image.alpha_composite(rest, picture)
    picture = fitted(picture, rest, patch, ImageChops.lighter(brush, rough))
    changed = changed_in(everywhere, picture, rest)
    clearly = ImageChops.multiply(changed, near).getbbox() is not None
    beyond = reached_from(brush, changed, everywhere).filter(ImageFilter.GaussianBlur(KEEP_EDGE * max(picture.size)))
    kept = ImageChops.lighter(joined(patch, brush.getbbox(), rest.size, seed), beyond)
    return Image.composite(picture, rest, kept), clearly


# ---- a masked change HELD to its picture (a tileable one) -----------------------------------------------------------
# It is drawn in a WINDOW of its picture (window_of), cut out at the picture's own size, and laid back: the model
# sees what it changes pixel for pixel, where a whole 2K picture reaches it as a smaller copy, and draws a 1K
# picture's worth. The window is ALL it is shown: given the whole picture too, small, it draws that whole picture
# into what was brushed, the more so the larger the mask.
WINDOW_PIXELS = 1024 * 1024        # the most a window holds: what every card shows the model whole
WINDOW_SIDE = 768                  # its sides at the least, where WINDOW_PIXELS allows both
WINDOW_RIM = 32                    # held to the picture along the window's sides inside it: no drawing ends there
WINDOW_BUFFER_MIN = 96             # how far it reaches beyond what was brushed at the least (else: the room, the rim)


def laid_onto(rest: Any, picture: Any, where: Any) -> Any:
    """`picture` where `where` (L) is white, `rest` where it is black - each one's own pixels, to the bit - and in
    between the two mixed as far as each shows (RGBa): mixed as they are, a see-through pixel would add the colour
    left under it, a dark fringe round a cut-out. Both RGBA."""
    from PIL import Image

    out = Image.composite(picture.convert("RGBa"), rest.convert("RGBa"), where).convert("RGBA")
    out.paste(rest, mask=where.point(lambda v: 255 if v == 0 else 0))
    out.paste(picture, mask=where.point(lambda v: 255 if v == 255 else 0))
    return out


def to_the_middle(brush: Any) -> tuple[int, int]:
    """How far a tileable picture is moved for a masked change - px right and down, what leaves one side coming
    back at the other - so that what was brushed (L, white) lies in its middle: across and down, the longest stretch
    nothing was brushed in, round the picture's edge too, comes to lie half at either side. The room then never ends
    at the picture's edge, and what is drawn across that edge goes on at the other side, as the picture does."""
    def move(brushed_at: list[bool]) -> int:
        length = len(brushed_at)
        longest = begins = run = 0
        for at in range(2 * length):                # twice round: a stretch across the edge is one stretch
            run = 0 if brushed_at[at % length] else run + 1
            if longest < run <= length:
                longest, begins = run, at - run + 1
        return -(begins + longest // 2) % length if 0 < longest < length else 0

    width, height = brush.size
    data = brush.tobytes()
    return (move([255 in data[x::width] for x in range(width)]),
            move([255 in data[y * width:(y + 1) * width] for y in range(height)]))


def window_of(brushed: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Where in a picture of `size` a held masked change is drawn: a box round `brushed` (the box of what was
    brushed) that reaches the room and the rim beyond it each way (KEEP_ROOM, WINDOW_RIM), its sides WINDOW_SIDE at
    the least - while that holds no more than WINDOW_PIXELS; a longer one is thinner, down to WINDOW_BUFFER_MIN
    beyond what was brushed. Its sides in steps of REFERENCE_STEP, inside the picture. None: what was brushed is
    too large for a window. The page shows the artist the same box while they brush
    (static/js/stages/mask_window.js)."""
    def stepped(length: int) -> int:
        return -(-length // REFERENCE_STEP) * REFERENCE_STEP

    spans = (brushed[2] - brushed[0], brushed[3] - brushed[1])
    least = [min(full, stepped(span + 2 * WINDOW_BUFFER_MIN)) for span, full in zip(spans, size)]
    if least[0] * least[1] > WINDOW_PIXELS:
        return None
    buffer = round(KEEP_ROOM * max(size)) + WINDOW_RIM
    sides = [min(full, max(stepped(span + 2 * buffer), WINDOW_SIDE)) for span, full in zip(spans, size)]
    while sides[0] * sides[1] > WINDOW_PIXELS:
        sides[0 if sides[0] - least[0] >= sides[1] - least[1] else 1] -= REFERENCE_STEP      # the side with more to give
    left, top = (min(max((low + high - side) // 2, 0), full - side)
                 for low, high, side, full in zip(brushed[:2], brushed[2:], sides, size))
    return left, top, left + sides[0], top + sides[1]


class Kept(NamedTuple):
    """A masked change as it is drawn: the window of its picture it is drawn in (window_of), to be kept (RGBA), what
    was brushed there and its room (L, white); and the whole picture that window is laid back into (RGBA), with
    the window's corner in it. The picture slid as the page showed it and, a tileable one, moved `moved` px right
    and down, wrapping round (to_the_middle)."""
    rest: Any
    brush: Any
    room: Any
    moved: tuple[int, int]
    whole: Any
    at: tuple[int, int]


def kept_from(req: GenerateRequest, change: tuple[Any, Any]) -> Kept:
    """A held masked change, of the picture and the brush `change` (change_of) holds."""
    from PIL import ImageChops

    rest, brush = change
    moved = to_the_middle(brush) if req.tileable else (0, 0)
    whole, brush = (ImageChops.offset(p, *moved) for p in (slid(rest, req.slide_back), brush))
    brushed_box = brush.getbbox()
    # the whole picture where nothing was brushed, or more than a window holds
    box = (window_of(brushed_box, brush.size) if brushed_box else None) or (0, 0, *brush.size)
    rest, brush = whole.crop(box), brush.crop(box)
    room = with_room(brush, brush.size, max(whole.size))
    return Kept(rest, brush, ImageChops.multiply(room, within_rim(box, whole.size)), moved, whole, box[:2])


def fractions(box: list[int] | None, size: tuple[int, int]) -> list[float] | None:
    """A box in a picture of `size` as shares of its sides (the page lays a window's preview there)."""
    return box and [round(v / side, 5) for v, side in zip(box, (*size, *size))]


def within_rim(box: tuple[int, int, int, int], size: tuple[int, int]) -> Any:
    """A window (`box` in a picture of `size`) but for WINDOW_RIM along its sides that lie inside the picture: white
    (L). A drawing that reached such a side would end there, cut."""
    from PIL import Image

    width, height = box[2] - box[0], box[3] - box[1]
    inner = Image.new("L", (width, height), 0)
    inner.paste(255, (WINDOW_RIM if box[0] > 0 else 0, WINDOW_RIM if box[1] > 0 else 0,
                      width - (WINDOW_RIM if box[2] < size[0] else 0), height - (WINDOW_RIM if box[3] < size[1] else 0)))
    return inner


def held_to(keep: Kept, req: GenerateRequest, references: list[Any],
            init: Any) -> tuple[GenerateRequest, list[Any], Any, Any]:
    """A masked change as it is drawn: the request, its example pictures, start picture and mask. The start picture
    (start_pictures) is laid as the change is (keep.moved) and cut to its window, and so is the example picture of
    the same file - shown the brush's magenta at one place and given room at another, the model draws nothing new.
    A window smaller than its picture is asked for as a picture of its own size. The mask is the room: the drawing
    is held to the picture everywhere outside it."""
    from PIL import ImageChops

    full, size = keep.whole.size, keep.rest.size

    def laid(picture: Any) -> Any:
        across, down = (round(by * side / asked) for by, side, asked in zip(keep.moved, picture.size, full))
        return ImageChops.offset(picture, across, down)

    def window(picture: Any) -> Any:                # of a picture laid as the whole one, whatever its size
        across, down = (side / asked for side, asked in zip(picture.size, full))
        left, top = keep.at
        return picture.crop((round(left * across), round(top * down),
                             round((left + size[0]) * across), round((top + size[1]) * down)))

    start = Path(req.init_image)
    own = [Path(ref) == start for ref in req.ref_images]
    shown = [laid(picture) if same else picture for same, picture in zip(own, references)]
    if size == full:
        return req, shown, laid(init), keep.room
    return (req.model_copy(update={"width": size[0], "height": size[1]}),
            [window(picture) if same else picture for same, picture in zip(own, shown)], window(laid(init)), keep.room)


def kept_outside(picture: Any, keep: Kept) -> tuple[Any, bool]:
    """Of `picture`, drawn for `keep`'s window, what the model drew for what was brushed (reached_from), in the tone
    of the picture it was made from (in_tone); and everywhere else that picture, to the pixel - the whole of it,
    laid as the page showed it (RGBA); and whether the model clearly changed anything at the brush."""
    from PIL import ImageChops, ImageFilter

    rest, brush, room, moved, whole, at = keep
    picture = picture.convert("RGBA")
    side = max(whole.size)
    clearly = False
    if brush.getbbox() is not None:
        changed = changed_in(room, picture, rest)
        part = reached_from(brush, changed, room, side)
        picture = toned(picture, rest, room, ImageChops.lighter(part, changed))
        clearly = ImageChops.multiply(changed, part).getbbox() is not None
        whole = whole.copy()
        whole.paste(laid_onto(rest, picture, part.filter(ImageFilter.GaussianBlur(KEEP_EDGE * side))), at)
    return ImageChops.offset(whole, -moved[0], -moved[1]), clearly


def window_place(keep: Kept) -> list[int] | None:
    """Where a held change's window lies in its picture as the page showed it - [left, top, right, bottom], px;
    on a tileable picture, moved for the change, right and bottom may lie past the picture's sides (what does comes
    back at the other side). None: the whole picture."""
    full, size = keep.whole.size, keep.rest.size
    if size == full:
        return None
    left, top = ((at - by) % side for at, by, side in zip(keep.at, keep.moved, full))
    return [left, top, left + size[0], top + size[1]]


GIVEN_QUALITY = 92                 # of the record's pictures (WebP): to look at, not to work from


def keep_given(folder: Path, req: GenerateRequest, drawn: GenerateRequest, seed: int, noise: int, steps: int,
               copies: list[tuple[Any, tuple[int, int] | None]], start: Mapping[str, Any], answers: list[Any],
               keep: Kept | None, patch: tuple[int, int, int, int] | None) -> None:
    """What the model was given for a request and what it answered, into `folder` for the artist to look at (the
    page's Alt + click on the picture): given.json and the pictures it names. `drawn`: the request as the model got
    it (a window's differs from `req`); `noise`: the seed its noise came from (noise_seed); `copies`: its example
    pictures with the size the model sees each at (None:
    the engine's own); `start`: generate's start picture and mask, if any; `answers`: the model's own pictures of a
    masked change, before anything was kept or laid back (else none: the saved picture is the answer); `keep`: a
    held change; `patch`: of a masked change drawn whole, the part that is laid onto the picture (patch_of)."""
    folder.mkdir(parents=True, exist_ok=True)
    own = [Path(ref) == Path(req.init_image) for ref in req.ref_images] if req.init_image else []
    window = window_place(keep) if keep is not None else None
    way = ("window" if window else "held" if keep is not None else "whole" if patch
           else "start" if start else "examples" if copies else "words")

    def kind(index: int) -> str:
        if not (index < len(own) and own[index]):
            return "example"
        return {"window": "window", "held": "brushed", "whole": "brushed"}.get(way, "start")

    def saved(picture: Any, name: str) -> str:
        picture.save(folder / name, quality=GIVEN_QUALITY)
        return name

    record: dict[str, Any] = {
        "words": drawn.prompt, "model": req.model, "steps": steps, "seed": seed, **({"noise_seed": noise} if noise != seed else {}),
        "size": [drawn.width, drawn.height], "picture": [req.width, req.height], "way": way,
        **({"window": window} if window else {}),
        **({"patch": list(patch)} if patch and patch != (0, 0, req.width, req.height) else {}),
        "pictures": [{"file": saved(picture, f"picture-{n + 1}.webp"), "seen": list(seen) if seen else None, "is": kind(n)}
                     for n, (picture, seen) in enumerate(copies)],
        "start": saved(start["init"], "start.webp") if start else None,
        **({"strength": req.strength} if start else {}),
        "mask": saved(start["mask"], "mask.png") if start and start.get("mask") is not None else None,
        "answers": [saved(picture, f"answer-{n + 1}.webp") for n, picture in enumerate(answers)],
    }
    (folder / "given.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


def opaque(picture: Any) -> Any:
    """The picture as RGB, for a request without a transparent background: its alpha channel is dropped, bit for bit
    the model's colours - unless the model left a part see-through all the same (has_cutout). What lies under that
    part is no background but whatever colour the model left there (a flat magenta on the user's sheet of 2026-11-21),
    so such a picture is laid on white instead."""
    return on_white(picture) if has_cutout(picture) else picture.convert("RGB")


def encode_preview(raw: bytes, width: int, height: int, transparent: bool) -> bytes:
    """sd.cpp's preview frame (width x height RGB or RGBA bytes) as a PNG: RGBA only when a transparent background
    was asked for, as the finished picture (save_picture). A 64x64 frame takes a fraction of a millisecond."""
    from PIL import Image

    picture = Image.frombytes("RGBA" if len(raw) == 4 * width * height else "RGB", (width, height), raw)
    if not transparent:
        picture = opaque(picture)
    out = io.BytesIO()
    picture.save(out, format="PNG", compress_level=1)
    return out.getvalue()


def has_transparency(picture: Any) -> bool:
    """Whether any pixel is see-through. Alpha of OPAQUE_ALPHA and up is the model's noise on an opaque picture."""
    return picture.mode == "RGBA" and picture.getchannel("A").getextrema()[0] < OPAQUE_ALPHA


SAVE_TRIES = 5                     # a fresh file another program (an antivirus, the search indexer) holds for a moment


def save_picture(picture: Any, out_dir: Path, seed: int, transparent: bool = True) -> Path:
    """picture -> <out_dir>/qwen-<seed>.png: RGBA when a transparent background was asked for and some pixels are
    see-through, else RGB (Pixal3D takes any alpha below 255 as the picture's own cut-out and then skips its
    background removal, so an opaque picture must carry no alpha at all; a cut-out nobody asked for is laid on white:
    opaque). Written straight to its name: out_dir is the
    job's own folder, which the hub reads only after the answer, so no temporary name has to be renamed (a rename is
    what an antivirus holding the new file makes fail). Tried again a few times on a Windows sharing error."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"qwen-{seed}.png"
    n = 1
    while path.exists():
        path = out_dir / f"qwen-{seed}-{n}.png"
        n += 1
    keep_alpha = transparent and has_transparency(picture)
    if not keep_alpha and has_cutout(picture):
        print(f"[spellcast] {path.name}: the model left part of the picture see-through without being asked; it is "
              "laid on white.", flush=True)
    image = picture if keep_alpha else opaque(picture)
    for attempt in range(SAVE_TRIES):
        try:
            # zlib level 3: a 1024x1024 RGBA picture in ~60 ms instead of ~220 ms at Pillow's default 6, the same size
            image.save(path, format="PNG", compress_level=3)
            return path
        except PermissionError:
            if attempt == SAVE_TRIES - 1:
                raise
            time.sleep(0.2 * (attempt + 1))
    return path


def save_pictures(pictures: list[Any], out_dir: Path, seeds: list[int],
                  transparent: bool) -> tuple[list[tuple[Path, int, Any]], str | None]:
    """Each picture saved on its own (save_picture) -> ([(path, seed, picture)] of those saved, in order, and what went
    wrong with the others, or None). OSError when none could be saved (a full disk, a folder that went away)."""
    saved: list[tuple[Path, int, Any]] = []
    problem: OSError | None = None
    for picture, seed in zip(pictures, seeds):
        try:
            saved.append((save_picture(picture, out_dir, seed, transparent), seed, picture))
        except OSError as exc:
            traceback.print_exc()
            problem = exc
    if problem is not None and not saved:
        raise problem
    lost = len(pictures) - len(saved)
    return saved, (f"{lost} of {len(pictures)} pictures could not be saved ({problem.strerror or problem})."
                   if problem is not None else None)


# ---- the HTTP side -------------------------------------------------------------------------------------------------

class GpuWorker:
    """One thread for everything that touches the GPU: the loading and every request, one at a time."""

    def __init__(self) -> None:
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-gpu")

    def submit(self, fn: Callable[..., Any], /, *args: Any) -> concurrent.futures.Future[Any]:
        return self._pool.submit(fn, *args)

    def run(self, fn: Callable[..., Any], /, *args: Any) -> Any:
        return self._pool.submit(fn, *args).result()


def error_response(status: int, detail: str, kind: str) -> JSONResponse:
    # kind: "input" | "oom" | "cancelled" | "busy" | "restart" | "error" (adapters/qwen.py reads it)
    return JSONResponse({"detail": detail, "kind": kind}, status_code=status)


def failure_response(exc: BaseException) -> JSONResponse:
    if isinstance(exc, Cancelled):
        return error_response(409, "Stopped: the pictures were cancelled.", "cancelled")
    if isinstance(exc, InputProblem):
        return error_response(422, str(exc), "input")
    if isinstance(exc, (OutOfMemory, MemoryError)):
        return error_response(500, "Qwen-Image ran out of memory. Close other programs that use the graphics card "
                                   "(games, video tools, other AI apps) and try again; fewer example pictures or a "
                                   "smaller shape also need less.", "oom")
    text = str(exc)
    short = text if len(text) <= 300 else text[:300] + "..."
    return error_response(500, f"Qwen-Image failed while making the pictures ({type(exc).__name__}: {short}). Try "
                               "again; if it keeps failing, restart Qwen-Image (Settings > Debugging) and check its "
                               "log.", "error")


BUSY = ("Qwen-Image is still working on earlier pictures, which the hub stopped waiting for. Try again in a moment.")
BROKEN = ("Qwen-Image stopped after an error that left the graphics card unusable for it. It starts again for the "
          "next pictures.")
CACHE_DROPPED = ("Qwen-Image ran short of video memory for its speed-up with these example pictures, so this took "
                 "longer. Fewer example pictures are faster.")
STOPPED_EARLY = "Stopped after {made} of {asked} pictures."


def inside(path: Path, root: Path | None) -> bool:
    return root is None or path.resolve().is_relative_to(root)


def validate(req: GenerateRequest, projects_root: Path | None, has_vision: bool, lib: Any = None) -> str | None:
    """The problem with a request, in an artist's words, or None. A negative prompt without guidance (cfg_scale 1.0)
    is no problem: it is simply not used."""
    if not req.prompt.strip():
        return "Describe the picture first."
    for name, side in (("width", req.width), ("height", req.height)):
        if not SIDE_MIN <= side <= SIDE_MAX or not is_multiple(side):
            return (f"The picture's {name} must be a multiple of {SIDE_STEP} between {SIDE_MIN} and {SIDE_MAX} pixels "
                    f"(got {side}).")
    out = Path(req.out_dir)
    if not out.is_absolute():
        return "The output folder must be an absolute path."
    if not inside(out, projects_root):
        return "Spellcast3D only saves pictures into its own projects folder."
    if req.given_dir is not None and not (Path(req.given_dir).is_absolute() and inside(Path(req.given_dir), projects_root)):
        return "Spellcast3D only keeps a request's record in its own projects folder."
    if len(req.ref_images) > MAX_REFERENCES:
        return f"Qwen-Image takes up to {MAX_REFERENCES} example pictures (got {len(req.ref_images)})."
    if req.ref_images and not has_vision and req.model in QWEN_MODELS:     # FLUX.3's go through its VAE
        return NO_VISION
    if req.ref_sides is not None and (len(req.ref_sides) != len(req.ref_images) or not all(
            REFERENCE_MIN_SIDE <= side <= REFERENCE_MAX_SIDE for side in req.ref_sides)):
        return (f"ref_sides names a side of {REFERENCE_MIN_SIDE} to {REFERENCE_MAX_SIDE} pixels for each example "
                "picture.")
    for ref in req.ref_images:
        path = Path(ref)
        if not path.is_absolute():
            return "The example pictures' paths must be absolute."
        if not inside(path, projects_root):
            return "Spellcast3D only reads example pictures from its own projects folder."
        if not path.is_file():
            return f"The example picture {path.name} was not found. Add it again."
    for what, given in (("start picture", req.init_image), ("mask", req.mask_image),
                        ("picture to change", req.changed_image), ("mask of the change", req.changed_mask)):
        path = Path(given) if given else None
        if path is not None and (not path.is_absolute() or not inside(path, projects_root) or not path.is_file()):
            return f"The {what} must be a picture in Spellcast3D's own projects folder."
    if req.mask_image and not req.init_image:
        return "A mask needs the picture it belongs to (init_image)."
    if bool(req.changed_image) != bool(req.changed_mask):
        return "The picture to change and the mask of the change go together (changed_image, changed_mask)."
    if req.changed_image and not req.mask_image:
        return ("A picture to change needs the start picture and mask that can hold the drawing to it (init_image, "
                "mask_image).")
    if lib is not None and lib.str_to_sample_method(req.sampler.encode("ascii", "replace")) >= \
            sdcpp.SAMPLE_METHOD_COUNT:
        return f"Qwen-Image does not know the sampler {req.sampler!r} (use {SAMPLER})."
    return None


def albedo_problem(req: AlbedoRequest, projects_root: Path | None, engine: Any) -> str | None:
    """The problem with an albedo request, in an artist's words, or None."""
    if not getattr(engine, "albedo_files", None):
        return ALBEDO_MISSING
    picture, out = Path(req.picture), Path(req.out)
    if not (picture.is_absolute() and inside(picture, projects_root) and picture.is_file()):
        return "The picture to take the lighting off must be in Spellcast3D's own projects folder."
    if not (out.is_absolute() and inside(out, projects_root) and out.suffix.lower() == ".png"):
        return "Spellcast3D only saves the albedo as a .png into its own projects folder."
    from PIL import Image

    try:
        with Image.open(picture) as opened:
            width, height = opened.size
    except OSError:
        return f"{picture.name} is not a picture Qwen-Image can read."
    if width % SIDE_STEP or height % SIDE_STEP:
        return f"The picture's sides must be multiples of {SIDE_STEP} pixels (got {width}x{height})."
    return None


# The albedo pass holds twice its picture's tokens, so a picture larger than a single view's goes through it at
# ALBEDO_SHARE of its sides, on every card alike: it then takes what the painting of the whole picture took (the
# 6,144 MB budget: 3,045 MB for the pass at 1536x1536, 3,195 MB for the first step of a 2048x2048 painting).
# Never fewer pixels than a view's picture, which every card's pass holds.
ALBEDO_WHOLE = SIZE * SIZE
ALBEDO_SHARE = 0.75


def albedo_size(size: tuple[int, int]) -> tuple[int, int]:
    """The size a picture of `size` goes through the albedo pass at (sides multiples of SIDE_STEP)."""
    pixels = size[0] * size[1]
    if pixels <= ALBEDO_WHOLE:
        return size
    share = max(ALBEDO_SHARE, (ALBEDO_WHOLE / pixels) ** 0.5)
    return (max(SIDE_STEP, round(size[0] * share) // SIDE_STEP * SIDE_STEP),
            max(SIDE_STEP, round(size[1] * share) // SIDE_STEP * SIDE_STEP))


def lighting_off(albedo: Callable[[Any], Any], picture: Any) -> Any:
    """`picture` (PIL RGB) with its lighting taken off (`albedo`: the pass, a picture -> its albedo), always in
    one pass over all of it: the figures of an Omni-Texture sheet are judged together, as they were painted (a pass
    for each made a model's grey underside black beside its white sides). A picture larger than the pass takes
    (albedo_size) goes through it smaller; what comes back is scaled up and given the fine detail the painting
    lost on its way down."""
    from PIL import Image, ImageChops

    size = albedo_size(picture.size)
    if size == picture.size:
        return albedo(picture)
    small = picture.resize(size, Image.LANCZOS)
    detail = ImageChops.subtract(picture, small.resize(picture.size, Image.BICUBIC), 1, 128)
    return ImageChops.add(albedo(small).resize(picture.size, Image.BICUBIC), detail, 1, -128)


def model_problem(engine: Any, model: str) -> str | None:
    """Why a request for `model` cannot be drawn now, in words, else None (also for an engine that cannot tell: a
    test's): FLUX.3 with an engine build that does not draw it (FLUX3_NO_ENGINE), or the model's files not in the
    models folders now (MODEL_MISSING, naming the first missing file)."""
    denoisers = getattr(engine, "denoisers", None)
    if not denoisers:
        return None
    if model == "flux3" and not getattr(engine, "flux3_engine", False):
        return FLUX3_NO_ENGINE
    missing = engine.missing_files(model)
    return MODEL_MISSING[model].format(name=missing[0]) if missing else None


def request_words(req: GenerateRequest, seed: int, capability: tuple[int, int]) -> str:
    """The log's line of a request: its pictures, the model and how it is drawn (sampling)."""
    plan = sampling(req.model, req.steps, req.cfg_scale, req.sampler, req.first_block_cache, capability)
    guided = condition_count(plan["cfg_scale"]) > 1
    if req.model in ("turbo", "flux3"):
        given = req.model_fields_set        # what the request said (not its defaults, which are the full model's)
        ignored = [f"{req.steps} steps" if "steps" in given and req.steps != plan["steps"] else "",
                   f"CFG {req.cfg_scale:g}" if condition_count(req.cfg_scale) > 1 else "",
                   req.sampler if req.sampler != SAMPLER else ""]
        ignored = [word for word in ignored if word]
        steps = (f"{plan['steps']} steps (its {'fixed' if req.model == 'turbo' else 'own'} schedule"
                 + (f"; the request's {', '.join(ignored)} not used)" if ignored else ")"))
    else:
        steps = f"{plan['steps']} steps"
    unused = " (its negative prompt is not used)" if req.negative_prompt.strip() and not guided else ""
    cache = ("on (skip " + FIRST_BLOCK_CACHE_SKIP + ")" if plan["first_block_cache"] else "off") + \
        (" (by the card)" if req.model == "full" and req.first_block_cache is None else "")
    transparent = "on" if req.transparent and draws_alpha(req.model) else \
        "off (no alpha channel)" if req.transparent else "off"
    return (f"{req.batch_count} picture(s) {req.width}x{req.height} with {model_name(req.model)}, {steps}, "
            + (f"CFG {plan['cfg_scale']:g}" if guided else f"no guidance{unused}")
            + f", {plan['sampler']}, seed {seed}, {len(req.ref_images)} example picture(s), transparent "
            f"{transparent}, first-block cache {cache}")


def build_app(engine: Any, gpu: GpuWorker, progress: Progress, projects_root: Path | None,
              info: dict[str, Any], cuda_works: Callable[[], bool] | None = None,
              exit_later: Callable[[], None] | None = None) -> FastAPI:
    """The /spellcast/* routes around `engine` (a QwenEngine, or anything with .ready, .loading, .has_vision, .lib,
    .generate(req, seed, references) and .cancel()). cuda_works(): whether CUDA still works after a failure (default:
    yes). exit_later(): what ends this process a moment after an answer that says CUDA broke (default: os._exit(3) a
    second later)."""
    app = FastAPI(title="Qwen-Image 2.1 for Spellcast3D", docs_url=None, redoc_url=None, openapi_url=None)
    working = threading.Lock()      # one request at a time: a second one is refused, never queued behind the first
    state = {"broken": False}       # CUDA broke: no more pictures; the process exits
    pictures = Pictures()           # read on the GPU thread, one request at a time
    cuda_works = cuda_works or (lambda: True)
    exit_later = exit_later or (lambda: threading.Timer(1.0, os._exit, (3,)).start())

    @app.exception_handler(RequestValidationError)
    async def readable_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', ())[1:]) or 'request'}: {e.get('msg', '')}"
                             for e in exc.errors())
        return error_response(422, f"Qwen-Image rejected the settings ({problems}).", "input")

    @app.get("/spellcast/health")
    async def health() -> dict[str, Any]:
        answer = {"ready": bool(engine.ready) and not state["broken"], "loading": bool(engine.loading),
                  "references": bool(engine.has_vision), "load_seconds": getattr(engine, "load_seconds", None),
                  "albedo": bool(getattr(engine, "albedo_files", None)), **info}
        if getattr(engine, "model", None) is not None:      # which model is loaded, and which can be drawn now
            answer["model"] = engine.model
            answer["models"] = {name: bool(engine.can_draw(name)) for name in MODELS}
            answer["flux3_engine"] = bool(getattr(engine, "flux3_engine", False))
        return answer

    @app.get("/spellcast/progress")
    async def spellcast_progress() -> dict[str, Any]:
        return progress.to_dict()

    @app.get("/spellcast/preview")
    async def spellcast_preview() -> Response:
        """The picture being drawn as it looks so far (progress says when there is a new one: its "preview")."""
        newest = progress.newest_preview()
        if newest is None:
            return error_response(404, "There is no preview picture right now.", "input")
        seq, step, _image, picture = newest
        return Response(picture, media_type="image/png", headers={
            "Cache-Control": "no-store", "X-Preview-Seq": str(seq), "X-Preview-Step": str(step)})

    @app.post("/spellcast/cancel")
    async def cancel(_: dict[str, Any] = Body(...)) -> dict[str, bool]:
        # the JSON body keeps other web pages out (a CORS preflight, which is refused), as for /shutdown
        busy = working.locked()
        if busy:
            engine.cancel()
        return {"cancelling": busy}

    @app.post("/spellcast/shutdown")
    async def shutdown(_: dict[str, Any] = Body(...)) -> dict[str, bool]:
        """What CTRL_BREAK does, for a parent whose console events cannot reach us (imesh_server.py)."""
        signal.raise_signal(getattr(signal, "SIGBREAK", signal.SIGTERM))
        return {"stopping": True}

    # A plain def route: FastAPI runs it in a worker thread, so health and progress keep answering meanwhile.
    @app.post("/spellcast/generate")
    def generate(req: GenerateRequest) -> JSONResponse:
        if state["broken"]:
            return error_response(503, BROKEN, "restart")
        if not engine.ready:
            return error_response(503, "Qwen-Image is still loading its models. Try again in a moment.", "error")
        problem = validate(req, projects_root, engine.has_vision, getattr(engine, "lib", None)) \
            or model_problem(engine, req.model)
        if problem is not None:
            return error_response(400, problem, "input")
        if not working.acquire(blocking=False):
            # the hub never sends two at once: the first is one it gave up on (a timeout), still running
            return error_response(409, BUSY, "busy")
        try:
            progress.cancel.clear()
            return run(req)
        finally:
            working.release()

    def run(req: GenerateRequest) -> JSONResponse:
        seed = req.seed if req.seed >= 0 else random.randint(0, 2 ** 31 - 1)
        seeds = batch_seeds(seed, req.batch_count)
        refs = [Path(p) for p in req.ref_images]
        placement = getattr(engine, "placement", None) or {}
        capability = placement.get("capability", (99, 0))
        print(f"[spellcast] {request_words(req, seed, capability)}", flush=True)
        steps = sampling(req.model, req.steps, req.cfg_scale, req.sampler, req.first_block_cache, capability)["steps"]
        use_model = getattr(engine, "use_model", None)

        def given(drawn: GenerateRequest, references: list[Any], start: dict[str, Any], answers: list[Any],
                  keep: Kept | None, patch: tuple[int, int, int, int] | None) -> None:
            """keep_given for this request, the copies as generate makes them. Never the job's failure."""
            try:
                noise = noise_seed(seed, drawn_from(references, start.get("init")))      # of the pictures as read
                seen: list[Any] = [None] * len(references)
                if placement and req.model in QWEN_MODELS:
                    references = qwen_copies(drawn, references, copy_cap(placement, drawn, len(references)))
                    seen = [picture.size for picture in references]
                keep_given(Path(req.given_dir), req, drawn, seed, noise, steps, list(zip(references, seen)), start,
                           answers, keep, patch)
            except Exception:
                traceback.print_exc()

        def work() -> tuple[JSONResponse | None, list[Path], dict[str, Any], float]:
            started = time.monotonic()
            failure: JSONResponse | None = None
            paths: list[Path] = []
            extra: dict[str, Any] = {}
            try:
                # the other model first, when it is the one loaded (its loading words are the progress meanwhile)
                switched = use_model(req.model) if use_model is not None else None
                progress.start(req.batch_count, steps, len(refs), (req.width, req.height))
                if switched is not None:
                    progress.timings["switch"] = switched
                references, init, mask = pictures.read(req)
                keep = change = patch = None
                reshaped, drawn = False, req
                if req.changed_image:
                    change = change_of(req.changed_image, req.changed_mask, (req.width, req.height))
                    reshaped = req.tileable and change is None
                    if req.tileable and change is not None:
                        keep = kept_from(req, change)
                        drawn, references, init, mask = held_to(keep, req, references, init)
                    else:                       # drawn anew whole: nothing holds the drawing
                        init = mask = None
                        patch = change and patch_of(change[1].getbbox(), change[1].size)
                start = {"init": init, "mask": mask} if init is not None else {}     # img2img's, when asked
                progress.box = fractions(window_place(keep), (req.width, req.height)) if keep is not None else None
                made = engine.generate(drawn, seed, references, **start)
                if req.given_dir:
                    given(drawn, references, start, made if change is not None else [], keep, patch)
                if req.keep_place and req.tileable and references:
                    made = [in_place(picture, references[0]) for picture in made]
                unchanged = False
                if made and change is not None:
                    finished = [kept_outside(picture, keep) if keep is not None
                                else redrawn_whole(picture, *change, patch, seed) for picture in made]
                    made, unchanged = [picture for picture, _ in finished], not any(changed for _, changed in finished)
                if req.slide_back:
                    made = [slid(picture, -req.slide_back) for picture in made]
                progress.saving()
                transparent = req.transparent and draws_alpha(req.model)     # FLUX.3's: RGB, always
                saved, unsaved = save_pictures(made, Path(req.out_dir), seeds, transparent)
                paths = [path for path, _seed, _picture in saved]
                extra = {"timings": dict(progress.timings), "cache_dropped": progress.cache_dropped,
                         "seeds": [seed for _path, seed, _picture in saved], "unsaved": unsaved,
                         "made": len(made), "reshaped": reshaped, "unchanged": unchanged,
                         "transparent": [transparent and has_transparency(p) for _path, _seed, p in saved]}
            except Exception as exc:
                if not isinstance(exc, (Cancelled, InputProblem)):
                    traceback.print_exc()
                failure = failure_response(exc)
            finally:
                progress.stop()
            return failure, paths, extra, time.monotonic() - started

        failure, paths, extra, seconds = gpu.run(work)
        if failure is not None:
            return failed(failure, "Make these pictures again.")
        timings = extra.get("timings") or {}
        print(f"[spellcast] {len(paths)} picture(s) saved in {seconds:.1f} s ({timings}); transparent: "
              f"{extra.get('transparent')}", flush=True)
        warnings = []
        made = extra.get("made", len(paths))
        if made < req.batch_count:                  # cancelled while the last pictures were being finished
            warnings.append(STOPPED_EARLY.format(made=made, asked=req.batch_count))
        if extra.get("unsaved"):
            warnings.append(extra["unsaved"])
        if extra.get("cache_dropped"):
            warnings.append(CACHE_DROPPED)
        for told, words in (("reshaped", KEPT_NOTHING), ("unchanged", KEPT_ALL)):
            if extra.get(told):
                warnings.append(words)
        return JSONResponse({"images": [str(p) for p in paths], "seeds": extra.get("seeds", seeds[:len(paths)]),
                             "seconds": round(seconds, 1), "warning": " ".join(warnings) or None,
                             "timings": timings})

    @app.post("/spellcast/albedo")
    def albedo(req: AlbedoRequest) -> JSONResponse:
        """A painted picture's albedo (Omni-Texture takes the lighting off each view before projecting it)."""
        if state["broken"]:
            return error_response(503, BROKEN, "restart")
        if not engine.ready:
            return error_response(503, "Qwen-Image is still loading its models. Try again in a moment.", "error")
        problem = albedo_problem(req, projects_root, engine)
        if problem is not None:
            return error_response(400, problem, "input")
        if not working.acquire(blocking=False):
            return error_response(409, BUSY, "busy")
        try:
            progress.cancel.clear()
            return run_albedo(req)
        finally:
            working.release()

    def run_albedo(req: AlbedoRequest) -> JSONResponse:
        use_model = getattr(engine, "use_model", None)

        def work() -> tuple[JSONResponse | None, float]:
            from PIL import Image

            started = time.monotonic()
            progress.stop()                 # no phases to report; the ring of warning lines starts empty
            try:
                # either Qwen-Image 2.1 model runs the add-on: a switch only away from FLUX.3
                if use_model is not None and getattr(engine, "model", None) not in QWEN_MODELS:
                    use_model(req.model)
                with Image.open(req.picture) as opened:
                    picture = opened.convert("RGB")
                out = Path(req.out)
                out.parent.mkdir(parents=True, exist_ok=True)
                lighting_off(engine.albedo, picture).save(out)
                return None, time.monotonic() - started
            except Exception as exc:
                if not isinstance(exc, (Cancelled, InputProblem)):
                    traceback.print_exc()
                return failure_response(exc), time.monotonic() - started
            finally:
                progress.stop()             # a model switch's loading phase ends here

        failure, seconds = gpu.run(work)
        if failure is not None:
            return failed(failure, "Try again.")
        print(f"[spellcast] The albedo of {Path(req.picture).name} in {seconds:.1f} s.", flush=True)
        return JSONResponse({"image": str(Path(req.out)), "seconds": round(seconds, 1)})

    @app.post("/spellcast/warm_up")
    def warm_up(body: dict[str, Any] = Body(...)) -> JSONResponse:
        """The first use of an Omni-Texture view paid now (QwenEngine.warm_omnitexture), on the GPU thread: a request
        that comes meanwhile waits for it there (never "busy"); one under way already makes it needless.
        -> {"queued"}."""
        warm = getattr(engine, "warm_omnitexture", None)
        if warm is None or state["broken"] or not engine.ready or working.locked():
            return JSONResponse({"queued": False})
        albedo = body.get("albedo") is not False

        def work() -> None:
            try:
                seconds = warm(albedo)
            except Cancelled:       # the cancel of a request waiting behind it (which stops that one too)
                print("[spellcast] The warm-up stopped: the pictures waiting behind it were cancelled.", flush=True)
                return
            except Exception:
                traceback.print_exc()
                # As failed() does for a request (on the GPU thread already: cuda_works directly)
                if not (engine.ready and cuda_works()):
                    state["broken"] = True
                    engine.ready = False
                    print("[spellcast] The graphics card does not answer Qwen-Image any more after the warm-up's "
                          "error: it stops now, and the hub starts it again for the next pictures.", flush=True)
                    exit_later()
                    return
                print("[spellcast] The warm-up failed (above); the first view pays its first use instead.",
                      flush=True)
                return
            finally:
                progress.stop()
            if seconds is not None:
                print(f"[spellcast] Warmed up for Omni-Texture in {seconds:.1f} s (a {WARM_SIDE}x{WARM_SIDE} picture "
                      "and its albedo): the first view starts as fast as the others.", flush=True)

        gpu.submit(work)
        return JSONResponse({"queued": True})

    def failed(failure: JSONResponse, again: str) -> JSONResponse:
        """A request's failure answer. A CUDA error can leave the GPU context unusable, and a model switch that could
        load neither model leaves none (use_model: ready false): then not ready any more (health, and any request
        still to come, say so), "restart" with `again` (what to do), then exit; the hub starts Qwen-Image again."""
        kind = json.loads(failure.body).get("kind")
        if kind in ("input", "cancelled") or (engine.ready and gpu.run(cuda_works)):
            return failure
        state["broken"] = True
        engine.ready = False
        print("[spellcast] The graphics card does not answer Qwen-Image any more after that error: it stops now, "
              "and the hub starts it again for the next pictures.", flush=True)
        exit_later()
        return error_response(500, BROKEN + " " + again, "restart")

    return app


# ---- main ----------------------------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen-Image 2.1 server for Spellcast3D")
    p.add_argument("--sdcpp", required=True, help="stable-diffusion.cpp's Windows CUDA build (code/venvs/qwen/sdcpp)")
    p.add_argument("--models", required=True, help="the model files (models/qwen-image)")
    p.add_argument("--albedo", default=None, help="the albedo add-on's folder (models/qwen-image/albedo; ALBEDO_FILES)")
    p.add_argument("--flux3-models", default=None,
                   help=f"FLUX.3 Image Klein's model files (default: {FLUX3_FOLDER} beside --models)")
    p.add_argument("--flux3-engine", choices=FLUX3_ENGINES, default="auto",
                   help="whether this engine build draws FLUX.3 Image Klein: auto = the pinned build when "
                        "install/sdcpp.py ENGINE_HAS_FLUX3 says so; on (developers) = take this build to draw it (a "
                        "build under test); off = never")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--projects-root", default=None,
                   help="the hub's projects folder: /spellcast/generate reads and writes only inside it")
    p.add_argument("--quality", default=DEFAULT_QUALITY, help=f"the diffusion model's size: {', '.join(QUALITIES)}")
    p.add_argument("--first-model", choices=MODELS, default=None,
                   help="the model loaded first, when its files are there and this engine draws it: the one the artist "
                        "has picked, so no other is loaded before it (default: the Turbo)")
    p.add_argument("--text-encoder", default=manifest.DEFAULT_TEXT_ENCODER,
                   help=f"the text encoder's size: {', '.join(manifest.QWEN_TEXT_ENCODERS)} (a bigger one in --models "
                        "is read in its place when this one is not there)")
    p.add_argument("--placement", choices=PLACEMENTS, default="auto",
                   help="offload (auto): weights in RAM, their copies on the card while they fit; gpu: weights kept "
                        "on the card")
    p.add_argument("--max-vram", default=None,
                   help="sd.cpp's per-card budget in GiB (negative: free memory at start minus that much)")
    p.add_argument("--card-gb", type=float, default=None,
                   help="treat the card as this size for the placement (testing a smaller card)")
    p.add_argument("--pin", choices=PINS, default="auto",
                   help=f"the diffusion weights in page-locked RAM ({PIN_ENV}={PIN_PARAMS}, for an engine build that "
                        f"knows it): faster to the card, that RAM stays taken. auto: on PCs with 24 GB of RAM or "
                        f"more; a {PIN_ENV} that is already set is never changed")
    p.add_argument("--prefix-cache", choices=("auto", "f16", "q8_0"), default=PREFIX_CACHE,
                   help="how the prompt's and the example pictures' keys and values are kept between steps: auto = "
                        "sd.cpp's choice (f16 under flash attention), q8_0 = about half the memory, at 8 bits")
    p.add_argument("--prefix-keep", choices=PREFIX_KEEPS, default=PREFIX_KEEP,
                   help="whether the engine keeps them (and the example pictures' latents) from one picture to the "
                        "next with the same prompt and pictures - re-rolls, variations - (on: the default, a first "
                        "step as fast as the others; auto: the engine's own rule, under flash attention only; off: "
                        "never; see PREFIX_KEEP)")
    p.add_argument("--matmul", choices=MATMULS, default="mmq",
                   help=f"the kernels of the diffusion model's quantized products. mmq: the MMQ kernels ({MATMUL_ENV}"
                        "=off), the pictures of the model's file; int8: the engine's int8 path where it has one, "
                        "faster and not the same pictures (see MATMULS)")
    p.add_argument("--keep-params", choices=KEEPS, default="auto",
                   help=f"the weights kept on the card between runs ({KEEP_ENV}=1, for an engine build that knows it): "
                        "the diffusion weights stay from picture to picture, the text encoder's until the decode needs "
                        "their room. auto: where the card holds them beside the decode (12 GB cards); on / off: always / "
                        f"never (comparisons); a {KEEP_ENV} that is already set is never changed")
    p.add_argument("--attention", choices=ATTENTIONS, default="auto",
                   help="auto: the diffusion model's attention through the engine's SageAttention (cards of compute "
                        "capability 8.0 and newer: RTX 30 series ..; flash attention on older cards and with an engine "
                        "build other than the pinned one); flash: flash attention always (the former path, for "
                        "comparisons); sage: the sage op with any build that has it (developers: a build under test; "
                        "still flash attention on older cards)")
    p.add_argument("--mmap", choices=("on", "off"), default="on",
                   help="weights kept in RAM are the model files' own mapping (on) instead of a private copy")
    p.add_argument("--eager-load", choices=("on", "off"), default="on",
                   help="read every weight while starting (on), or at its first use (off: the first picture waits)")
    p.add_argument("--verbose", action="store_true", help="sd.cpp's debug lines too")
    return p.parse_args(argv)


def model_files(models: Path, quality: str, flux3_models: Path | None = None,
                text_encoder: str = manifest.DEFAULT_TEXT_ENCODER) -> tuple[dict[str, Path | None], list[str]]:
    """{"turbo", "full" (the two denoisers, whether there or not: a request checks again, the full one may arrive
    while this runs), "vae", "llm", "vision" (None when missing: no example pictures), "flux3", "flux3_vae",
    "flux3_llm" (FLUX.3 Image Klein's, in flux3_models - FLUX3_FOLDER beside `models` unless given -, whether there or
    not: an optional download)} and the missing required names: Qwen-Image's VAE, its text encoder, and a denoiser -
    the Turbo's, the installer's, when neither is there. FLUX.3's never count as missing."""
    named = manifest.qwen_model_files(quality, models, text_encoder)
    flux3 = manifest.flux3_model_files(quality, flux3_models or models.parent / FLUX3_FOLDER)
    files: dict[str, Path | None] = {"turbo": named["turbo"], "full": named["full"], "vae": named["vae"],
                                     "llm": named["text_encoder"], "vision": named["vision"],
                                     "flux3": flux3["diffusion"], "flux3_vae": flux3["vae"],
                                     "flux3_llm": flux3["text_encoder"]}
    missing = [p.name for key, p in files.items() if key in ("vae", "llm") and p is not None and not p.is_file()]
    if start_model(files) is None:
        missing.insert(0, named["turbo"].name)
    if files["vision"] is not None and not files["vision"].is_file():
        files["vision"] = None
    return files, missing


def start_model(files: dict[str, Path | None], wanted: str | None = None) -> str | None:
    """The model the process loads first: `wanted` (--first-model) when all its files are there; else the Turbo, the
    default, when its file is there; else the full model (an install from before the Turbo that was not updated
    since). None when neither of Qwen-Image's two is there: its files are the process's must, whichever is first."""
    def there(model: str) -> bool:
        keys = (model, "flux3_vae", "flux3_llm") if model == "flux3" else (model,)
        return all((path := files.get(key)) is not None and path.is_file() for key in keys)

    default = next((model for model in QWEN_MODELS if there(model)), None)
    return wanted if default is not None and wanted in MODELS and there(wanted) else default


def startup_hint(exc: BaseException) -> str:
    if isinstance(exc, PathProblem):
        return str(exc)
    if isinstance(exc, (sdcpp.LayoutError, OSError)):
        return sdcpp.describe_failure(exc)
    if isinstance(exc, (OutOfMemory, MemoryError)):
        return ("Qwen-Image ran out of memory while loading (it needs about 11 GB of RAM and video memory together). "
                "Close other programs, then start Qwen-Image again.")
    return ("Qwen-Image could not start (the error is above). Start it again; if it keeps failing, close Spellcast3D "
            "and double-click run-spellcast3d.bat again to repair it.")


def main() -> int:
    args = parse_args()
    sdcpp_dir, models = Path(args.sdcpp).resolve(), Path(args.models).resolve()
    projects_root = Path(args.projects_root).resolve() if args.projects_root else None
    quality = args.quality if args.quality in QUALITIES else DEFAULT_QUALITY
    if quality != args.quality:
        print(f"[spellcast] Unknown quality {args.quality!r}: using {DEFAULT_QUALITY}.", flush=True)
    missing_runtime = sdcpp.missing_files(sdcpp_dir)
    if missing_runtime:
        print(f"[spellcast] Missing in {sdcpp_dir}: {', '.join(missing_runtime)}. Close Spellcast3D and double-click "
              "run-spellcast3d.bat again to repair Qwen-Image.", flush=True)
        return 2
    # The engine's libraries (1.1 GB with the CUDA runtime) are read ahead like the weights: from a cold cache
    # Windows pages them in piece by piece as the first context is made (14 s where the drive answered slowly).
    libraries = [sdcpp_dir / name for name in (sdcpp.DLL_NAME, *sdcpp.GGML_DLLS, *sdcpp.CUDA_DLLS)]
    threading.Thread(target=read_ahead, args=(libraries, None, "Engine libraries"), name="qwen-libraries",
                     daemon=True).start()
    flux3_models = Path(args.flux3_models).resolve() if args.flux3_models else None
    text_encoder = args.text_encoder if args.text_encoder in manifest.QWEN_TEXT_ENCODERS else manifest.DEFAULT_TEXT_ENCODER
    if text_encoder != args.text_encoder:
        print(f"[spellcast] Unknown text encoder size {args.text_encoder!r}: using {text_encoder}.", flush=True)
    files, missing = model_files(models, quality, flux3_models, text_encoder)
    print(f"[spellcast] Reads: {manifest.qwen_files_words(quality, text_encoder, models)}.", flush=True)
    if missing:
        print(f"[spellcast] Missing in {models}: {', '.join(missing)}. " + MODELS_MISSING, flush=True)
        return 2
    if files["vision"] is None:
        print(f"[spellcast] {VISION_FILE} is missing: Qwen-Image works without example pictures.", flush=True)
    default = start_model(files) or DEFAULT_MODEL
    model = start_model(files, args.first_model) or DEFAULT_MODEL
    quiet_crashes()
    progress = Progress()
    try:
        sdcpp.prepare_dll_search(sdcpp_dir)     # the CUDA runtime beside the engine: the card is known before the DLL
        cuda = Cuda()
        devices = cuda.devices()
        device = compute_device(devices, args.placement) if devices else None
        lib = load_runtime(sdcpp_dir, args.pin, in_ram=args.placement != "gpu", matmul=args.matmul,
                           capability=device["capability"] if device else None)
    except Exception as exc:
        traceback.print_exc()
        print("[spellcast] " + startup_hint(exc), flush=True)
        return 2
    flux3_engine, said = flux3_engine_choice(args.flux3_engine, sdcpp.is_pinned(lib))
    if model == "flux3" and not flux3_engine:
        model = default
    for other in QWEN_MODELS:
        path = files[other]
        print(f"[spellcast] {model_name(other, start=True)}: {path.name if path else '?'} "
              + ("loaded first." if other == model else "there: a request for it switches." if path and path.is_file()
                 else "not there (a request for it is refused until it is)."), flush=True)
    flux3_set = model_sets(files)["flux3"]
    flux3_there = [p for p in flux3_set.values() if p is not None and p.is_file()]
    print(f"[spellcast] {said} Its files ({flux3_set['diffusion'].parent if flux3_set['diffusion'] else '?'}): "
          + ("loaded first." if model == "flux3" else "there: a request for it switches." if len(flux3_there) == 3 else
             f"{len(flux3_there)} of 3 there (a request for it is refused until all are).")
          + ("" if flux3_engine else " Its requests are refused."), flush=True)
    if device is None:
        print("[spellcast] Qwen-Image needs an NVIDIA graphics card, and CUDA sees none (or its driver is too old for "
              "CUDA 12.8).", flush=True)
        return 2
    if len(devices) > 1:
        cuda.release_others(device["index"], devices)
    card_mb = int(args.card_gb * 1024) if args.card_gb else device["total_mb"]
    free_mb = min(device["free_mb"], card_mb)
    max_vram = args.max_vram
    if args.card_gb and max_vram is None:
        max_vram = f"{max(args.card_gb - 2.0, 1.0):g}"     # what such a card has free, less sd.cpp's own reserve
    # one model at a time: Qwen-Image's (the bigger of its two denoisers that are there, and the files they share) or
    # FLUX.3's three when they are there and weigh more (--placement gpu's budget fits whichever is loaded)
    denoiser = max((p.stat().st_size for name in QWEN_MODELS if (p := files[name]) is not None and p.is_file()),
                   default=0)
    qwen_mb = (denoiser + sum(p.stat().st_size for key in ("vae", "llm", "vision") if (p := files[key]) is not None)) \
        // 2 ** 20
    weights_mb = max(qwen_mb, sum(p.stat().st_size for p in flux3_there) // 2 ** 20)
    placement = choose_placement(free_mb, weights_mb, args.placement, max_vram, args.prefix_cache, args.prefix_keep)
    placement["eager_load"] = args.eager_load == "on"
    placement["mmap"] = args.mmap == "on"
    placement["pin_params"] = os.environ.get(PIN_ENV) or None     # as asked of the engine (the reports name it)
    placement["attention"] = attention_choice(args.attention, sdcpp.is_pinned(lib), device["capability"])
    placement["flux3_attention"] = flux3_attention_choice(args.attention, device["capability"])
    placement["matmul"] = os.environ.get(MATMUL_ENV) or "auto"     # as asked of the engine (the reports name it)
    # the default start model's own denoiser (the one that would stay on the card), not the biggest one there - also
    # when another model is loaded first (--first-model): a switch keeps these choices, so a process that starts with
    # a model runs as one that switched to it
    kept = files[default]
    diffusion_mb = kept.stat().st_size // 2 ** 20 if kept is not None and kept.is_file() else 0
    placement["keep_params"] = keep_params(args.keep_params, placement["budget_mb"], diffusion_mb,
                                           engine_knows=engine_pin.ENGINE_KEEPS_PARAMS or not sdcpp.is_pinned(lib))
    print("[spellcast] " + set_keep(placement["keep_params"]), flush=True)
    print("[spellcast] " + set_prefix_pass(device["capability"],
                                           engine_pin.ENGINE_PREFIX_PASS or not sdcpp.is_pinned(lib)), flush=True)
    placement["prefix_pass"] = os.environ.get(PASS_ENV) == "1"     # as the engine sees it
    placement["keep_params"] = os.environ.get(KEEP_ENV) == "1"      # as the engine sees it
    placement["vae_direct"] = vae_direct(placement["budget_mb"], capability=device["capability"],
                                         kept_mb=diffusion_mb if placement["keep_params"] else 0)
    placement["capability"] = tuple(device["capability"])     # first_block_cache_on's default per card
    ram = free_ram_mb()
    print(f"[spellcast] Starting Qwen-Image 2.1 ({quality}) on {args.host}:{args.port}: {device['name']} (sm_"
          f"{device['capability'][0]}{device['capability'][1]}, {device['total_mb']:,} MB, {device['free_mb']:,} MB "
          f"free" + (f", treated as {card_mb:,} MB" if args.card_gb else "") + f"); weights {weights_mb:,} MB; RAM "
          f"free " + (f"{ram:,} MB" if ram is not None else "unknown") + f"; placement {placement}.", flush=True)
    if placement["name"] == "offload" and ram is not None and ram < weights_mb:
        print("[spellcast] Less RAM is free than Qwen-Image's weights need: Windows will read them from the disk "
              "again and again, which is slow. Close other programs.", flush=True)

    engine = QwenEngine(lib, files, placement, progress, verbose=args.verbose, model=model, flux3_engine=flux3_engine)
    if args.albedo:
        found = albedo_files(Path(args.albedo).resolve())
        engine.albedo_files = found if found and sdcpp.has_albedo(lib) else None
        print("[spellcast] " + ("The albedo add-on is ready (Omni-Texture takes the lighting off its paintings)."
                                if engine.albedo_files else ALBEDO_MISSING), flush=True)
    gpu = GpuWorker()
    loaded = gpu.submit(engine.load)

    def after_load(future: concurrent.futures.Future[Any]) -> None:
        exc = future.exception()
        if exc is not None:
            traceback.print_exception(exc)
            print("[spellcast] " + startup_hint(exc), flush=True)
            sys.stdout.flush()
            os._exit(1)

    loaded.add_done_callback(after_load)

    import uvicorn

    info = {"gpu": device["name"], "vram_mb": device["total_mb"], "placement": placement["name"], "quality": quality,
            "sheet_side": sheet_side(placement["budget_mb"]),
            "sdcpp": sdcpp.PINNED_TAG if sdcpp.is_pinned(lib) else sdcpp.commit(lib)}
    app = build_app(engine, gpu, progress, projects_root, info, cuda_works=cuda.works)
    code = 0
    try:
        uvicorn.run(LoopbackGuard(app) if is_loopback(args.host) else app, host=args.host, port=args.port,
                    log_level="warning")
    except OSError as exc:
        print(f"[spellcast] Qwen-Image could not open port {args.port} ({exc}). Another program uses it: wait a "
              "moment and start Qwen-Image again, or change the Qwen-Image port in Settings.", flush=True)
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)      # without waiting for a picture or a load still running on the GPU thread


if __name__ == "__main__":
    sys.exit(main())
