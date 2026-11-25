"""Qwen-Image 2.1 adapter: the "images" stage, made by Qwen-Image 2.1 (GGUF) in stable-diffusion.cpp.

Qwen-Image is Spellcast3D's built-in image generator, and Forge Neo the alternative: spellcast3d-config.ini (the
root config file, spellcast/config.py) picks one, and only that one is registered in the hub (build_adapters).

The tool process is launchers/qwen_server.py in the qwen venv: it calls stable-diffusion.dll (code/venvs/qwen/sdcpp)
in-process and keeps the models loaded while it runs. It has no page of its own, only the /spellcast/* routes used
here: GET health, GET progress, GET preview (the picture being drawn as it looks so far, small: asked for with each
request of a job the page follows, and passed on to that job for the page, ctx.preview), POST
generate (blocks until the pictures are written), POST albedo (a picture's lighting taken off by our albedo add-on,
for Omni-Texture: albedo()), POST cancel (stops within one step), POST shutdown. Errors come
back as {"detail", "kind"}: "oom", "input", "cancelled", "restart" (its GPU context broke: the process exits, the job
runner starts it afresh), "error". While a [profiling] switch of the config file is on when it starts, the script is
spellcast/profiling/qwen_profiled.py instead: the same server with the developers' profilers attached from outside.

The prop's images settings it reads: prompt (sent as the artist typed it), width and height (the Shape, at 1K) and
resolution (RESOLUTIONS: the Shape's sides times 1, 1.5 or 2), seed, batch_count ("Variations": one call, so the
prompt is encoded once; the engine draws its pictures one after the other), transparent (a picture with its own alpha: Pixal3D then needs
no background removal; the launcher wraps the prompt in Qwen's RGBA wording), qwen_model and Qwen's own qwen_steps
(namespaced, so Forge's steps never leak in). Its example pictures (images.references, up to MAX_REFERENCES) go along
as references.

The model (qwen_model, QWEN_MODELS): "turbo", Qwen-Image-2.1-Turbo, the default for every card - its own fixed
schedule of TURBO_STEPS steps (the launcher gives it its sigmas), so qwen_steps and first_block_cache are not sent for
it - or "full", the full Qwen-Image 2.1 at qwen_steps with the First-Block Cache tick, which the artist downloads
first from the Images page (spellcast/model_downloads.py): a "full" prop while that file is not on disk is refused in
the artist's words (FULL_MISSING), never drawn with the Turbo instead. A third choice, "flux3", is not Qwen-Image at
all: FLUX.3 Image Klein (Black Forest Labs), which the same launcher serves with its own files (models/flux3-klein,
also a download from the Images page) and its own recipe - FLUX3_STEPS steps, no guidance, euler, opaque pictures (it
has no alpha channel: transparent is sent false, and Pixal3D removes the background as for Forge Neo's pictures) - and
only with an image engine that can draw it (install/sdcpp.py engine_draws, the one gate: FLUX3_NEEDS_ENGINE until
then). One launcher holds one of the three at a time and switches when a request asks for another
(launchers/qwen_server.py).

The recipe: NO guidance (CFG_SCALE 1.0), which is how Qwen documents Qwen-Image 2.1 to be sampled, with euler and
DEFAULT_STEPS steps (ComfyUI's official template; Qwen's own reference is 40 steps, see DEFAULT_STEPS).
Without guidance a step is one pass through the model instead of two, and there is no negative prompt: at a CFG of
exactly 1.0 stable-diffusion.cpp skips the unconditional pass and never encodes one. So every request says CFG_SCALE
and an empty negative prompt, whatever the prop holds: its negative_prompt ("Avoid") is Forge Neo's setting, and the
qwen_cfg of projects saved before this recipe is no setting any more (neither is read here; projects.py drops the
latter when it reads a project).

The model is under Qwen's research licence; Qwen stated that the pictures it makes are the user's (`license`).
FLUX.3 Image Klein is under the FLUX Kommunity License (the page's licence chip for it: static/js/licenses.js).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import httpx

from .. import config, gpu, paths
from ..install import models as model_files
from ..install import sdcpp as engine_pin
from .base import LaunchSpec, StageContext, StageError, ToolAdapter, ToolNeedsRestart
from .images_common import masked_change, reference_files, results_patch, save_candidates

log = logging.getLogger(__name__)

# Qwen's statement of 21 Sep 2026 (Qwen Developers on X): "Outputs are not part of the licensed Materials. Users
# retain the rights to images and other content they generate using the model."
STATEMENT_URL = "https://x.com/QwenDevs/status/2101917379785838660"
LICENSE = {"name": "Qwen Research License (model)",
           "url": "https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE",
           "noncommercial": False, "research": True,
           "note": "The pictures you make are yours, also for commercial work (Qwen, 21 Sep 2026).",
           "links": [{"label": "Qwen’s statement", "url": STATEMENT_URL}]}

READY_TIMEOUT = 900.0              # ~11 GB of weights read before it reports ready: minutes from a slow disk
IMAGE_TIMEOUT = 20 * 60.0          # one picture; older cards, or weights offloaded to the CPU, take minutes
POLL_INTERVAL = 1.0
INCOMING = "images/.incoming"      # where the launcher writes a job's pictures before they become candidates
# What Qwen-Image was given for a picture and what it answered (the launcher's given_dir: given.json and the pictures
# it names), in a folder named as the picture: images/0012.png -> images/.given/0012/. The page shows it on
# Alt + click (stages/given.js).
GIVEN = "images/.given"
PROFILED_ENTRY = paths.PACKAGE / "profiling" / "qwen_profiled.py"     # qwen_server.py with the profilers attached
# The model the launcher holds or is loading, as far as the hub knows: the one it was started with (launch_spec: the
# artist's pick when an early start named one, else the Turbo), then the one its last answered picture request drew
# with. QwenAdapter.prop_order puts that model's props first.
_holding: str = "turbo"

# Keep in sync with projects.py (MAX_REFERENCES, IMAGES_NUMBERS, STAGE_DEFAULTS' qwen_steps): tests/test_qwen_hub.py
# compares them, and the launcher's own limits and request defaults too.
MAX_REFERENCES = 10
# No guidance, Qwen's own way of sampling Qwen-Image 2.1: diffusers' pipeline (Qwen's reference) defaults to a
# guidance of 1.0 because the model is meant to be sampled without it
# (https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21), and ComfyUI's official template uses cfg 1
# (https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1). Not settings: the adapter sends these two for every
# prop. At exactly 1.0 stable-diffusion.cpp's use_uncond stays false (src/pipeline/request.cpp): a step is ONE model
# pass, and a negative prompt is never encoded, so one would change nothing even if it were sent.
CFG_SCALE = 1.0
NEGATIVE_PROMPT = ""
# The hub's ONE place for the default number of steps (projects.STAGE_DEFAULTS repeats it for new props, and that
# test compares the two; the launcher's request default and the page's slider are the other layers' places).
# 25 is ComfyUI's official template value, checked against Qwen's own reference (diffusers) of 40 steps on 2026-10-03:
# 4 prompts at the same seed each (RTX 3060 12 GB, Q4_K_M, 1024x1024, transparent, Spellcast3D's own engine build).
# 25 and 40 steps without guidance were both sharp and natural and close to each other, where the former recipe (20
# steps at CFG 6) gave over-saturated, soft pictures. So 25 stays the default and 40 stays available on the page's
# Steps slider; a picture's seconds at both are in launchers/qwen_server.py (PLACEMENTS).
DEFAULT_STEPS, MIN_STEPS, MAX_STEPS = 25, 1, 100
# The three models a prop can pick (= projects.QWEN_MODELS, a test compares them; install/models.py's QWEN_MODELS are
# Qwen-Image's own two), the default first. The Turbo is sampled with its own fixed schedule of TURBO_STEPS steps (the
# launcher's TURBO_SIGMAS: the model card says the number of steps alone does not change it), so the request says
# TURBO_STEPS for it whatever qwen_steps holds. The third, "flux3" (FLUX.3 Image Klein, not Qwen-Image), is distilled
# for FLUX3_STEPS steps without guidance (Black Forest Labs' recipe, upstream stable-diffusion.cpp's docs/flux3.md):
# the request says FLUX3_STEPS for it, and the launcher draws it so whatever the request says.
QWEN_MODELS = ("turbo", "full", "flux3")
DEFAULT_MODEL = "turbo"
TURBO_STEPS = 8
FLUX3_STEPS = 16
MODEL_NAMES = {"turbo": "Qwen-Image 2.1 Turbo", "full": "the full Qwen-Image 2.1", "flux3": "FLUX.3 Image Klein"}
MAX_BATCH = 16
MAX_SEED = 2 ** 32 - 1                         # the launcher's limit (Stable Diffusion's usual seed range)
MIN_SIDE, MAX_SIDE, GRID = 256, 2432, 32       # stable-diffusion.cpp wants sides that divide by 32
# The Images panel's Resolution: what the Shape's sides (1024 x 1024, 832 x 1216...) are multiplied by.
RESOLUTIONS = {1.0: "1K", 1.5: "1.5K", 2.0: "2K"}
SAMPLER = "euler"

# The prompt of a prop that has example pictures and no words of its own.
FROM_EXAMPLES = ("the object from the example picture", "the object from the example pictures")

UNREACHABLE = ("Qwen-Image 2.1 is not reachable (it may have stopped; Settings > Debugging shows its log). Press "
               "Generate again: it starts by itself.")
CRASHED = ("Qwen-Image 2.1 stopped while making the picture: the graphics card reported an error (Settings > "
           "Debugging shows its log). It starts again for the next prop; press Generate again for this one. If it "
           "keeps happening, close other programs that use the graphics card.")
OUT_OF_MEMORY = ("The graphics card ran out of memory (VRAM) while making the picture. Try a smaller shape, fewer "
                 "variations or example pictures, or close other programs that use the graphics card (games, "
                 "browsers with 3D tabs, other AI tools), then try again.")
# A prop set to the full model while its file is not in models\qwen-image (not downloaded yet, or still downloading).
FULL_MISSING = ("The full Qwen-Image 2.1 is not downloaded yet. Press “Download” on its button under Model (it "
                "downloads once, in the background; a download under way shows its progress there), or pick Turbo, "
                "then press Generate again.")
# The same for FLUX.3 Image Klein (its files in models\flux3-klein), and for an engine that cannot draw it yet.
FLUX3_MISSING = ("FLUX.3 Image Klein is not downloaded yet. Press “Download” on its button under Model (it downloads "
                 "once, in the background; a download under way shows its progress there), or pick Turbo, then press "
                 "Generate again.")
FLUX3_NEEDS_ENGINE = ("FLUX.3 Image Klein needs the next update of Spellcast3D's image engine (stable-diffusion.cpp), "
                      "which this version does not have yet. Pick Turbo or Full meanwhile.")


def matmul_arg(settings: Mapping[str, Any]) -> str:
    """The launcher's --matmul from the hub settings: "int8" where the artist turned on the Images panel's "Faster
    image generation" (qwen.int8_matmul: faster, small details drawn differently; launchers/qwen_server.py
    MATMULS), else "mmq", the model file's own pictures. The launcher reads it when it starts."""
    return "int8" if (settings.get("qwen") or {}).get("int8_matmul") is True else "mmq"


def qwen_settings(params: Mapping[str, Any]) -> dict[str, Any]:
    """The prop's images settings Qwen-Image uses, checked -> {"model", "steps", "width", "height" (the Shape's at
    the Resolution picked), "seed", "batch_count", "transparent", "tileable", "first_block_cache"}. StageError in the artist's
    words. Guidance is no setting
    (CFG_SCALE): a qwen_cfg, cfg_scale or negative_prompt among `params` (an older project's, or Forge Neo's) is neither
    read nor checked. The Turbo (the default model) gets TURBO_STEPS and no First-Block Cache, whatever qwen_steps and
    first_block_cache hold: they are the full model's settings (still checked, so a wrong one is said either way).
    FLUX.3 the same with FLUX3_STEPS, and never a transparent background (it draws opaque pictures) nor a tileable
    picture (the engine wraps Qwen-Image's positions round, not FLUX.3's)."""
    model = params.get("qwen_model", DEFAULT_MODEL)
    if model is None:
        model = DEFAULT_MODEL
    if model not in QWEN_MODELS:
        raise StageError("The model must be Turbo, Full or FLUX.3 Klein. Pick it again on the Images page.")
    steps = _whole(params, "qwen_steps", DEFAULT_STEPS, "Steps")
    if not MIN_STEPS <= steps <= MAX_STEPS:
        raise StageError(f"Steps must be a whole number from {MIN_STEPS} to {MAX_STEPS}.")
    batch = _whole(params, "batch_count", 1, "Variations")
    if not 1 <= batch <= MAX_BATCH:
        raise StageError(f"Variations must be a whole number from 1 to {MAX_BATCH}.")
    seed = _whole(params, "seed", -1, "Seed")
    if seed > MAX_SEED:
        raise StageError(f"Seed must be a whole number from 0 to {MAX_SEED:,} (or empty: a new surprise each time).")
    resolution = _number(params, "resolution", 1.0, "Resolution")
    if resolution not in RESOLUTIONS:
        raise StageError(f"Resolution must be {', '.join(RESOLUTIONS.values())}. Pick it again on the Images page.")
    full = model == "full"
    cache = params.get("first_block_cache") if isinstance(params.get("first_block_cache"), bool) else None
    return {"model": model,
            "steps": steps if full else FLUX3_STEPS if model == "flux3" else TURBO_STEPS,
            "width": _side(_number(params, "width", 1024, "Width") * resolution),
            "height": _side(_number(params, "height", 1024, "Height") * resolution),
            "seed": seed if seed >= 0 else -1, "batch_count": batch,
            # FLUX.3 has no alpha channel: never asked of it (the launcher would not wrap its prompt either)
            "transparent": params.get("transparent") is True and model != "flux3",
            "tileable": params.get("tileable") is True and model != "flux3",
            # the First-Block Cache (projects.py), the full model's: the artist's tick, or None = the card decides
            # (the launcher: on for the GTX 10xx, off for newer cards); the launcher turns it into the engine's skip
            # share. Never for the Turbo (its 8 steps have none to spare) nor FLUX.3 (an option of Qwen-Image's
            # engine; the launcher keeps it off for both anyway).
            "first_block_cache": cache if full else False}


def qwen_prompt(params: Mapping[str, Any], references: int) -> str:
    """The artist's prompt as typed (nothing is added to it); with example pictures and no prompt, it points at
    them."""
    text = str(params.get("prompt") or "").strip()
    return text or (FROM_EXAMPLES[references > 1] if references else "")


# Where the launcher holds a masked change (Smart Mask) to its picture, it is drawn anew under its mask, from noise
# alone, and held to the page's copy of the picture everywhere else (the launcher gives the mask some room first: its
# KEEP_ROOM).
MASKED_STRENGTH = 1.0


def viewport_shift(params: Mapping[str, Any]) -> float | None:
    """How far the page showed the viewport's picture slid right and down when the job was started (Tileable's offset
    slider: -0.5 .. 0.5 of its size, 0 in its middle); None without "from_viewport" or without such a number. The
    page sends it for a tileable picture that the job changes (Re-Render Viewport, Smart Mask): what is made is laid
    where that picture lies - slid back as far (the launcher's slide_back) and onto it to the pixel (keep_place)."""
    shift = params.get("viewport_shift")
    if params.get("from_viewport") is None or isinstance(shift, bool) or not isinstance(shift, (int, float)):
        return None
    return float(shift) if -0.5 <= shift <= 0.5 else None


def qwen_request(params: Mapping[str, Any], references: list[Path], out_dir: Path,
                 kept: tuple[Path, Path] | None = None) -> dict[str, Any]:
    """The body of POST /spellcast/generate for one prop: its settings, its example pictures (absolute paths) and
    out_dir, where the launcher writes the pictures. Always without guidance (CFG_SCALE, NEGATIVE_PROMPT): the
    launcher takes both for the bench's comparisons, the hub never sends others.
    `kept` (a Smart Mask job: images_common.masked_change): the picture a part of which changes, and the mask of
    that part; the first of `references` is the page's copy of that picture, the mask's magenta over that part,
    which the request also names as what the drawing may be held to (MASKED_STRENGTH). The launcher draws the change
    whole - of a large picture's redraw it keeps the patch round what was brushed - or, a tileable picture's, held
    in a window of it; either way the picture lies where it lay, so nothing is laid onto it afterwards (keep_place)."""
    if len(references) > MAX_REFERENCES:
        raise StageError(f"A prop takes at most {MAX_REFERENCES} example pictures. Remove some and press Generate "
                         f"again.")
    if not str(params.get("prompt") or "").strip() and not references:
        raise StageError("Type a prompt that describes the object (or add an example picture), then press Generate.")
    settings = qwen_settings(params)
    shift = viewport_shift(params)
    body = {"prompt": qwen_prompt(params, len(references)),
            "negative_prompt": NEGATIVE_PROMPT,
            "width": settings["width"], "height": settings["height"], "model": settings["model"],
            "steps": settings["steps"], "cfg_scale": CFG_SCALE, "sampler": SAMPLER,
            "seed": settings["seed"], "batch_count": settings["batch_count"],
            "ref_images": [str(p.resolve()) for p in references],
            "transparent": settings["transparent"], "tileable": settings["tileable"],
            "slide_back": shift or 0.0, "keep_place": shift is not None and settings["tileable"] and kept is None,
            "first_block_cache": settings["first_block_cache"],
            "out_dir": str(out_dir.resolve())}
    if kept is not None:
        picture, mask = (str(p.resolve()) for p in kept)
        body |= {"init_image": body["ref_images"][0], "mask_image": mask, "strength": MASKED_STRENGTH,
                 "changed_image": picture, "changed_mask": mask}
    return body


def _number(params: Mapping[str, Any], key: str, default: float, label: str) -> float:
    value = params.get(key)
    if value is None or value == "":
        return float(default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float("nan")
    if isinstance(value, bool) or number != number or number in (float("inf"), float("-inf")):
        raise StageError(f"\"{label}\" must be a number, not \"{value}\". Fix it and press Generate again.")
    return number


def _whole(params: Mapping[str, Any], key: str, default: int, label: str) -> int:
    number = _number(params, key, default, label)
    if not number.is_integer():
        raise StageError(f"\"{label}\" must be a whole number, not \"{params.get(key)}\". Fix it and press Generate "
                         f"again.")
    return int(number)


def _side(value: float) -> int:
    """A picture side on stable-diffusion.cpp's 32-pixel grid (the nearest), within MIN_SIDE..MAX_SIDE."""
    return min(MAX_SIDE, max(MIN_SIDE, round(value / GRID) * GRID))


class QwenAdapter(ToolAdapter):
    tool_id = paths.QWEN
    display_name = "Qwen-Image 2.1"
    role = "image generator"
    stage = "images"
    extra_stages = ("omnitexture",)   # the 3D model's colour texture painted anew (adapters/omnitexture.py)
    uses_gpu = True
    ui_path = "/spellcast/health"      # no page of its own
    license = LICENSE

    def __init__(self, quality: str | None = None, models_dir: Path | None = None, flux3_dir: Path | None = None,
                 text_encoder: str | None = None):
        # The model and text encoder sizes the root config picks when the hub starts (what the installer fetched); a
        # change of the file applies at the next start, like the choice of image generator.
        self.quality = quality if quality is not None else config.qwen_quality()
        self.text_encoder = text_encoder if text_encoder is not None else config.qwen_text_encoder()
        self.models_dir = models_dir or paths.MODELS_QWEN     # where the launcher reads them (--models)
        self.flux3_dir = flux3_dir or paths.MODELS_FLUX3      # FLUX.3 Image Klein's (--flux3-models)
        self.first_model: str | None = None                   # the next start's (aim); None: the Turbo

    def full_model_on_disk(self) -> bool:
        """Whether the full model's denoiser of the hub's size is in the models folder, whole (a few stat calls)."""
        return model_files.qwen_model_on_disk("full", self.quality, self.models_dir)

    def flux3_on_disk(self) -> bool:
        """Whether all of FLUX.3 Image Klein's files of the hub's size are in its folder, whole (a few stat calls)."""
        return model_files.flux3_model_on_disk(self.quality, self.flux3_dir)

    def can_draw(self, model: str) -> bool:
        """Whether `model` can draw now: the optional ones once their files are there, FLUX.3 only with an image
        engine that draws it."""
        if model == "full":
            return self.full_model_on_disk()
        if model == "flux3":
            return engine_pin.engine_draws("flux3") and self.flux3_on_disk()
        return model == DEFAULT_MODEL

    def _wished(self, wish: Mapping[str, Any]) -> str | None:
        """The model an early start's wish names, when it can draw now; else None (the start's default)."""
        model = wish.get("model")
        return model if model in QWEN_MODELS and self.can_draw(model) else None

    def aim(self, wish: Mapping[str, Any]) -> None:
        self.first_model = self._wished(wish)

    def serves(self, wish: Mapping[str, Any]) -> bool:
        return self._wished(wish) in (None, _holding)

    def check_model(self, settings: Mapping[str, Any]) -> None:
        """StageError for a prop set to a model that cannot draw now: the full model or FLUX.3 while its files are
        not there (FULL_MISSING, FLUX3_MISSING), FLUX.3 while the image engine cannot draw it (FLUX3_NEEDS_ENGINE).
        Said, never drawn with the Turbo instead."""
        model = qwen_settings(settings)["model"]
        if model == "full" and not self.full_model_on_disk():
            raise StageError(FULL_MISSING)
        if model == "flux3" and not engine_pin.engine_draws("flux3"):
            raise StageError(FLUX3_NEEDS_ENGINE)
        if model == "flux3" and not self.flux3_on_disk():
            raise StageError(FLUX3_MISSING)

    @functools.cached_property
    def _gpu(self) -> gpu.GpuInfo | None:
        return gpu.primary_gpu()       # nvidia-smi takes a moment and the card does not change while the hub runs

    def supported(self) -> tuple[bool, str]:
        return gpu.qwen_supported(self._gpu)

    # ---- process lifecycle ----------------------------------------------
    def launch_spec(self, port: int, settings: dict[str, Any]) -> LaunchSpec:
        global _holding
        # this start alone loads the model an early start asked for: a later one (a job's, a hand-off's) the Turbo
        _holding, self.first_model = self.first_model or DEFAULT_MODEL, None
        sdcpp = paths.QWEN_SDCPP
        if not sdcpp.is_dir():
            raise StageError("Qwen-Image 2.1's image engine (stable-diffusion.cpp) is missing. Close Spellcast3D and "
                             "start it again with run-spellcast3d.bat to finish the installation.")
        env = paths.clean_env(self.tool_id)
        # stable-diffusion.dll loads ggml-cuda.dll and the CUDA runtime DLLs from its own folder
        env["PATH"] = os.pathsep.join(part for part in (str(sdcpp), env.get("PATH")) if part)
        # A [profiling] switch on (developers): the same server, started through the entry that attaches the profilers
        script = PROFILED_ENTRY if any(config.profiling().values()) else paths.LAUNCHERS / "qwen_server.py"
        log.info("Qwen-Image 2.1 reads: %s", model_files.qwen_files_words(self.quality, self.text_encoder,
                                                                           self.models_dir))
        return LaunchSpec(
            argv=[str(paths.venv_python(self.tool_id)), str(script),
                  "--port", str(port),
                  "--projects-root", str(paths.PROJECTS),
                  "--models", str(self.models_dir),
                  "--albedo", str(paths.MODELS_ALBEDO),
                  "--flux3-models", str(self.flux3_dir),
                  "--sdcpp", str(sdcpp),
                  "--quality", self.quality,
                  "--text-encoder", self.text_encoder,
                  "--first-model", _holding,
                  "--matmul", matmul_arg(settings)],
            cwd=sdcpp,
            env=env,
            ready_timeout=READY_TIMEOUT,
        )

    async def health(self, client: httpx.AsyncClient, base_url: str) -> bool:
        try:
            r = await client.get(f"{base_url}/spellcast/health", timeout=5.0)
            return r.status_code == 200 and r.json().get("ready") is True
        except (httpx.HTTPError, ValueError, AttributeError):
            return False

    async def on_stop(self, client: httpx.AsyncClient, base_url: str) -> None:
        # CTRL_BREAK_EVENT does not reach a process spawned with CREATE_NO_WINDOW (imesh adapter)
        try:
            await client.post(f"{base_url}/spellcast/shutdown", json={}, timeout=5.0)
        except httpx.HTTPError:
            pass

    async def warm_up(self, client: httpx.AsyncClient, base_url: str, stage: str) -> None:
        """Started early for `stage` (the job runner's hand-off, once ready): for Omni-Texture, have Qwen-Image draw
        a small picture and its albedo now (POST /spellcast/warm_up), so that the first view of the job does not pay
        the add-on's loading and the first use of the engine's graphs. Never raises: it only saves time."""
        if stage != "omnitexture":
            return
        try:
            await client.post(f"{base_url}/spellcast/warm_up", json={"albedo": True}, timeout=10.0)
        except httpx.HTTPError:
            pass

    def prop_order(self, stage: str, props: list[dict[str, Any]], running: bool = True) -> list[str]:
        """The order a job takes its props in: Omni-Texture's grouped by the Qwen-Image model each one draws with
        (the launcher holds one model and switches at every change), the one the launcher holds first, so the model
        is switched at most once: while it runs (`running`) the one it holds (_holding), else the Turbo, which a
        job's start loads and the hand-off's warm-up warms. The artist's order within a group, and for the other
        stages."""
        ids = [str(prop["id"]) for prop in props]
        if stage != "omnitexture":
            return ids
        from . import omnitexture
        held = _holding if running and _holding in ("turbo", "full") else "turbo"
        models = [omnitexture.model_of(prop) for prop in props]
        return [prop_id for _model, prop_id in sorted(zip(models, ids), key=lambda item: item[0] != held)]

    # ---- work -------------------------------------------------------------
    def check_settings(self, stage: str, settings: dict[str, Any]) -> None:
        if stage == "omnitexture":
            return                      # its only setting is the artist's choice of props
        qwen_settings(settings)
        self.check_model(settings)      # a job for a "full" / "flux3" prop is refused at once while it cannot draw

    async def run_stage(self, ctx: StageContext, prop: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """The prop's prompt (and example pictures) -> new candidates images/NNNN.png, saved like Forge's. In the
        Omni-Texture stage: the prop's 3D model painted anew (omnitexture.run)."""
        if ctx.stage == "omnitexture":
            from . import omnitexture  # here, not at the top: that module may use this one's helpers
            return await omnitexture.run(ctx, prop, params)
        images: dict[str, Any] = prop.get("images") or {}
        out_dir = ctx.prop_dir / INCOMING
        self.check_model(params)        # again: its file may have gone since the job was queued
        references = reference_files(ctx, images, params)
        if len(references) > MAX_REFERENCES and params.get("from_viewport") is not None:
            # "Re-Render Viewport via examples" with 10 examples: the viewport's picture and the first 9
            references = references[:MAX_REFERENCES]
            if ctx.warn is not None:
                ctx.warn(f"The last example picture was left out: a picture is made from at most {MAX_REFERENCES} "
                         "pictures, the viewport's one included.")
        body = qwen_request(params, references, out_dir, masked_change(ctx, images, params))
        # The picture as it forms, for the page (ctx.preview): always, when the job has a place to show it
        body["preview"] = ctx.preview is not None
        body["given_dir"] = str((out_dir / "given").resolve())
        _remove_dir(out_dir)            # what an earlier, interrupted job left
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            ctx.report(0.0, f"Sending the prompt to {MODEL_NAMES[body['model']]} · {body['width']} × "
                            f"{body['height']} px…")
            result = await self._generate(ctx, body)
            pictures, seeds = _pictures(result, out_dir)
            if not pictures:
                raise StageError("Qwen-Image 2.1 finished without a picture. Check its log (Settings > Debugging) "
                                 "and press Generate again.")
            new_rel = save_candidates(ctx, images, pictures)
            keep_given(ctx, out_dir / "given", new_rel)
        finally:
            _remove_dir(out_dir)
        warning = result.get("warning")
        if isinstance(warning, str) and warning:
            log.info("Qwen-Image 2.1, prop %s: %s", ctx.prop_id, warning)
            if ctx.warn is not None:
                ctx.warn(warning)       # the job's own message ends as "Done.": the page says it on its own
        asked = body["batch_count"]
        done = (f"{len(new_rel)} of {asked} pictures done." if len(new_rel) != asked else
                "1 picture done." if asked == 1 else f"{asked} pictures done.")
        ctx.report(1.0, done)
        return results_patch(images, new_rel, seeds)

    async def _generate(self, ctx: StageContext, body: dict[str, Any]) -> dict[str, Any]:
        """POST /spellcast/generate and report its progress until it answers. A cancelled job asks Qwen-Image to stop
        (again at every poll until it says it stops: a cancel that arrives before the pictures started is not
        kept). stable-diffusion.cpp draws every picture of the request first and decodes them all at the end, so a
        cancel while drawing keeps none ("cancelled"); only one during that last decode answers with the pictures
        decoded so far (and a warning), and those are kept.

        The connection dropping without an answer means the process died while drawing: a CUDA error inside ggml
        aborts it (exit code 3). ToolNeedsRestart then has the job runner start it afresh for the next prop."""
        global _holding
        # (IMAGE_TIMEOUT is a 1024 x 1024 picture's: a bigger one takes as much longer as it has pixels)
        pixels = body.get("width", 1024) * body.get("height", 1024)
        timeout = IMAGE_TIMEOUT * body["batch_count"] * max(1.0, pixels / 1024 ** 2)
        post = asyncio.create_task(ctx.client.post(f"{ctx.base_url}/spellcast/generate", json=body,
                                                   timeout=httpx.Timeout(timeout, connect=10.0)))
        stopping = False
        shown = 0           # the number of the launcher's preview picture the job shows (0: none)
        try:
            while not (await asyncio.wait({post}, timeout=POLL_INTERVAL))[0]:
                if ctx.cancelled() and not stopping:
                    stopping = await _ask_to_stop(ctx)
                shown = await _show_preview(ctx, await _report_progress(ctx), shown)
            response = post.result()
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise StageError(UNREACHABLE) from None
        except httpx.TimeoutException:
            # It may still be at it: the next prop gets a fresh Qwen-Image (the job runner restarts it).
            await _ask_to_stop(ctx)
            raise ToolNeedsRestart(f"Qwen-Image 2.1 did not finish these pictures within {timeout / 60:.0f} minutes, "
                                   "so it was stopped (it starts again for the next prop). Try a smaller shape or "
                                   "fewer variations, or close other programs that use the graphics card.") from None
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError, httpx.CloseError):
            raise ToolNeedsRestart(CRASHED) from None
        except httpx.HTTPError:
            raise StageError("Qwen-Image 2.1 stopped responding while making the picture (it may have crashed or run "
                             "out of memory). Check its log in Settings > Debugging, close other programs and try "
                             "again.") from None
        finally:
            post.cancel()   # no-op once answered; drops the request if the job itself was cancelled

        try:
            data = response.json()
        except ValueError:
            data = None
        if not isinstance(data, dict):
            data = {}
        if response.status_code == 200 and isinstance(data.get("images"), list):
            _holding = body.get("model")
            return data
        kind, detail = data.get("kind"), data.get("detail")
        if kind == "cancelled":
            raise StageError("Cancelled.")
        if kind in ("restart", "busy"):
            # "restart": the graphics card stopped answering after an error (the process exits); "busy": it still
            # works on pictures the hub gave up on. Either way the next prop needs a fresh Qwen-Image.
            raise ToolNeedsRestart(detail if isinstance(detail, str) and detail else
                                   "Qwen-Image 2.1 has to start again after an error. Press Generate again.")
        if kind == "oom":
            raise StageError(OUT_OF_MEMORY)
        if isinstance(detail, str) and detail:
            raise StageError(detail)
        raise StageError(f"Qwen-Image 2.1 failed (HTTP {response.status_code}). Check its log (Settings > "
                         "Debugging) and try again.")


async def _ask_to_stop(ctx: StageContext) -> bool:
    """POST /spellcast/cancel -> whether Qwen-Image is stopping its pictures now."""
    try:
        r = await ctx.client.post(f"{ctx.base_url}/spellcast/cancel", json={}, timeout=5.0)
        return r.status_code == 200 and r.json().get("cancelling", True) is True
    except (httpx.HTTPError, ValueError, AttributeError):
        return False


async def _report_progress(ctx: StageContext) -> dict[str, Any] | None:
    """The launcher's step in its words, with the time left once it knows it ("Drawing picture 2 of 4 · step 7 of 25
    · about 5 min left"): a picture takes minutes. -> what the launcher said, or None when it said nothing to
    report."""
    try:
        r = await ctx.client.get(f"{ctx.base_url}/spellcast/progress", timeout=5.0)
        p = r.json()
        if p.get("phase") == "idle":
            return None     # not started yet, or just finished (its answer is on its way)
        fraction = float(p.get("fraction") or 0.0)
        message = str(p.get("message") or "Working").rstrip(".…")
        eta = p.get("eta")
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        return None         # progress is cosmetic; the generate request decides success
    left = time_left(eta)
    ctx.report(min(max(fraction, 0.0), 0.99), "Stopping…" if ctx.cancelled() else
               f"{message} · {left}" if left else f"{message}…")
    return p


async def _show_preview(ctx: StageContext, progress: dict[str, Any] | None, shown: int) -> int:
    """The launcher's preview picture onto the job (ctx.preview) when its progress names another one than the job
    shows (`shown`: that one's number, 0 for none) -> the number shown now. The launcher has none before the first
    step and between two pictures of a request: the job then shows none either. Cosmetic, like the progress."""
    if ctx.preview is None or progress is None:
        return shown
    info = progress.get("preview")
    seq = info.get("seq") if isinstance(info, dict) else None
    if isinstance(seq, bool) or not isinstance(seq, int):
        if shown:
            ctx.preview(None, 0)
        return 0
    if seq == shown:
        return shown
    try:
        r = await ctx.client.get(f"{ctx.base_url}/spellcast/preview", timeout=5.0)
        if r.status_code != 200 or not r.content:
            return shown
        step = int(r.headers.get("X-Preview-Step") or info.get("step") or 0)
        seq = int(r.headers.get("X-Preview-Seq") or seq)        # a newer one may have come meanwhile
    except (httpx.HTTPError, ValueError, TypeError):
        return shown
    box = info.get("box")               # a window's request: the part of the picture its previews are of
    whole = not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box))
    ctx.preview(r.content, step, None if whole else [float(v) for v in box])
    return seq


def time_left(eta: Any) -> str:
    """The launcher's estimate (seconds, or None) in words: "about 3 min left", "42 s left" (to the second under a
    minute: the launcher counts it down between steps), or ""."""
    if isinstance(eta, bool) or not isinstance(eta, (int, float)) or eta != eta or eta < 0:
        return ""
    return f"{max(1, round(eta))} s left" if eta < 59.5 else f"about {max(1, round(eta / 60))} min left"


def keep_given(ctx: StageContext, record: Path, new_rel: list[str]) -> None:
    """The launcher's record of a request (`record`: its given_dir) beside each picture the request made (GIVEN), each
    with the number of its own answer among the request's. Never the job's failure: the record is there to look at."""
    if not (record / "given.json").is_file():
        return
    try:
        given = json.loads((record / "given.json").read_text(encoding="utf-8"))
        for index, rel in enumerate(new_rel):
            folder = ctx.prop_dir / GIVEN / Path(rel).stem
            _remove_dir(folder)
            shutil.copytree(record, folder)
            (folder / "given.json").write_text(json.dumps({**given, "answer": index}, indent=1), encoding="utf-8")
    except (OSError, ValueError) as e:
        log.warning("Qwen-Image 2.1, prop %s: the record of what it was given could not be kept: %s", ctx.prop_id, e)


def _pictures(result: dict[str, Any], out_dir: Path) -> tuple[list[bytes], list[int]]:
    """The pictures the launcher wrote (PNGs in out_dir, nothing else) -> their bytes, and their seeds ([] unless
    every picture has one)."""
    folder = out_dir.resolve()
    seeds_in = result.get("seeds") if isinstance(result.get("seeds"), list) else []
    pictures: list[bytes] = []
    seeds: list[int | None] = []
    for index, name in enumerate(result.get("images") or []):
        path = Path(str(name))
        try:
            if not path.resolve().is_relative_to(folder) or not path.is_file():
                raise OSError("not in the job's folder")
            raw = path.read_bytes()
        except OSError as e:
            log.warning("Qwen-Image 2.1 reported the picture %s, which cannot be read: %s", name, e)
            continue
        if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            log.warning("Qwen-Image 2.1 reported the picture %s, which is not a PNG", name)
            continue
        pictures.append(raw)
        seed = seeds_in[index] if index < len(seeds_in) else None
        seeds.append(seed if isinstance(seed, int) and not isinstance(seed, bool) else None)
    return pictures, [] if None in seeds else [s for s in seeds if s is not None]


def _remove_dir(folder: Path) -> None:
    shutil.rmtree(folder, ignore_errors=True)   # a file another program holds open stays (the next job tries again)


async def generate(ctx: StageContext, body: dict[str, Any]) -> dict[str, Any]:
    """POST /spellcast/generate with the Images stage's progress, cancelling and errors (QwenAdapter._generate,
    which uses nothing of the adapter), for the other stages that draw with Qwen-Image (adapters/omnitexture.py)."""
    return await QwenAdapter._generate(None, ctx, body)  # type: ignore[arg-type]


ALBEDO_TIMEOUT = 5 * 60.0       # one picture; the first one also reads the add-on's files
ALBEDO_OUT_OF_MEMORY = ("The graphics card ran out of memory (VRAM) while taking the lighting off a picture: close "
                        "other programs that use it (games, browsers with 3D tabs, other AI tools).")


async def albedo_ready(ctx: StageContext) -> bool:
    """Whether Qwen-Image's server has the albedo add-on and an engine build that runs it (its health "albedo")."""
    try:
        response = await ctx.client.get(f"{ctx.base_url}/spellcast/health", timeout=10.0)
        return response.status_code == 200 and response.json().get("albedo") is True
    except (httpx.HTTPError, ValueError, AttributeError):
        return False


async def sheet_side(ctx: StageContext) -> int | None:
    """The side of the largest square picture Qwen-Image's server paints and takes the lighting off on this card
    (its health "sheet_side": Omni-Texture's sheets of several views); None when it does not say."""
    try:
        response = await ctx.client.get(f"{ctx.base_url}/spellcast/health", timeout=10.0)
        side = response.json().get("sheet_side") if response.status_code == 200 else None
        return int(side) if isinstance(side, (int, float)) and side > 0 else None
    except (httpx.HTTPError, ValueError, AttributeError):
        return None


async def albedo(ctx: StageContext, picture: Path, out: Path, model: str) -> Path:
    """POST /spellcast/albedo: `picture` with its lighting and shading taken off by our albedo add-on, written to
    `out` (Omni-Texture, for every picture it paints: a view's, or a sheet of several). Errors as StageError /
    ToolNeedsRestart, as for generate; the pass takes seconds and cannot be stopped (no cancel, no progress)."""
    body = {"picture": str(picture.resolve()), "out": str(out.resolve()), "model": model}
    try:
        response = await ctx.client.post(f"{ctx.base_url}/spellcast/albedo", json=body,
                                         timeout=httpx.Timeout(ALBEDO_TIMEOUT, connect=10.0))
    except (httpx.ConnectError, httpx.ConnectTimeout):
        raise StageError(UNREACHABLE) from None
    except httpx.TimeoutException:
        raise ToolNeedsRestart("Qwen-Image 2.1 did not take the lighting off a picture within "
                               f"{ALBEDO_TIMEOUT / 60:.0f} minutes (it starts again for the next prop). Try again, "
                               "and close other programs that use the graphics card.") from None
    except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError, httpx.CloseError):
        raise ToolNeedsRestart(CRASHED) from None
    except httpx.HTTPError:
        raise StageError("Qwen-Image 2.1 stopped responding while taking the lighting off a picture. Check its log "
                         "in Settings > Debugging and try again.") from None
    try:
        data = response.json()
    except ValueError:
        data = None
    data = data if isinstance(data, dict) else {}
    if response.status_code == 200 and isinstance(data.get("image"), str) and Path(data["image"]).is_file():
        return Path(data["image"])
    kind, detail = data.get("kind"), data.get("detail")
    if kind in ("restart", "busy"):
        raise ToolNeedsRestart(detail if isinstance(detail, str) and detail else
                               "Qwen-Image 2.1 has to start again after an error. Try again.")
    if kind == "oom":
        raise StageError(ALBEDO_OUT_OF_MEMORY)
    if isinstance(detail, str) and detail:
        raise StageError(detail)
    raise StageError(f"Qwen-Image 2.1 could not take the lighting off a picture (HTTP {response.status_code}). Check "
                     "its log (Settings > Debugging) and try again.")
