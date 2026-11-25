"""Forge Neo adapter: the "images" stage, through Forge's A1111-style REST API (/sdapi/v1).

Facts about the pinned neo fork (code/forge @ 7e79011) this module relies on:
- `--ckpt-dirs`, `--lora-dirs`, `--vae-dirs` and `--text-encoder-dirs` are argparse "append" options
  taking ONE directory per flag (modules/cmd_args.py:80-83), scanned besides Forge's own models/ folders
  (modules/sd_models.py:135, extensions-builtin/sd_forge_lora/networks.py:209, modules/sd_vae.py:78,
  modules_forge/main_entry.py:91-101). Paths go in as separate argv items: COMMANDLINE_ARGS
  is split with POSIX shlex, which eats Windows backslashes (modules/paths_internal.py:12-13).
- launch.py always runs verify_version() (modules/launch_utils.py:567-606), which
  blocks on input() when config.json exists without VERSION_UID "PY313".
- /sdapi/v1/txt2img and /img2img run synchronously under Forge's queue lock
  (modules/api/api.py:474, :547); /sdapi/v1/progress does not take the lock, so it
  can be polled while a request is running.
- An unknown override_settings.sd_model_checkpoint is silently dropped
  (modules/processing.py:807-808), so the checkpoint is checked here first.
- Errors come back as JSON {"error", "detail", "body", "message"} (webui.py:31-39),
  or {"detail"} for HTTPException / request validation.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import httpx

from .. import forge_config, paths
from .base import LaunchSpec, StageContext, StageError, ToolAdapter
from .images_common import reference_files, results_patch, save_candidates

_API = "/sdapi/v1"
_POLL_INTERVAL = 0.7        # seconds between progress polls
_CANCEL_GRACE = 60.0        # seconds Forge gets to wind down after an interrupt

# Spellcast3D's models folder (paths.MODELS): Forge's folder flag -> folder. Forge reads VAE and
# text-encoder files only when it starts; checkpoints and LoRAs also on Refresh.
MODEL_FOLDERS = (("--ckpt-dirs", paths.MODELS_CHECKPOINTS), ("--vae-dirs", paths.MODELS_VAE),
                 ("--text-encoder-dirs", paths.MODELS_TEXT_ENCODERS), ("--lora-dirs", paths.MODELS_LORAS))

NO_MODEL = ("No image model found. Put a .safetensors model into the models\\checkpoints folder of Spellcast3D "
            "and press Refresh, or add the folder where your models already are in Settings > Advanced > "
            "Extra model folders.")
UNREACHABLE = ("Could not reach Forge Neo, the image generator. It may still be starting, or it stopped "
               "(Settings > Debugging shows its log): try again in a moment.")


class ForgeAdapter(ToolAdapter):
    tool_id = paths.FORGE
    display_name = "Forge Neo"
    role = "image generator"
    stage = "images"
    uses_gpu = True
    ui_path = "/"

    # ---- process lifecycle ----------------------------------------------
    def launch_spec(self, port: int, settings: dict[str, Any]) -> LaunchSpec:
        forge = settings.get("forge") or {}
        argv = [str(paths.venv_python(paths.FORGE)), "launch.py",
                "--skip-prepare-environment", "--skip-python-version-check", "--skip-version-check",
                "--api", "--port", str(port)]
        for flag, folder in MODEL_FOLDERS:
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue    # e.g. a read-only copy: Forge still has its own models folders
            argv += [flag, str(folder)]
        folders = forge.get("ckpt_dirs") or []    # the artist's extra checkpoint folders (Settings)
        for folder in [folders] if isinstance(folders, str) else folders:
            if str(folder).strip():
                argv += ["--ckpt-dirs", str(paths.user_path(folder))]
        extra = forge.get("extra_args") or []
        argv += extra.split() if isinstance(extra, str) else [str(a) for a in extra]
        try:
            # launch.py would otherwise stop at a "Press Enter" prompt (see forge_config.py)
            forge_config.ensure_forge_config()
        except OSError:
            pass    # unreadable / unwritable: Forge's own log will say what is wrong with its folder
        return LaunchSpec(
            argv=argv,
            cwd=paths.FORGE_DIR,
            # SD_WEBUI_RESTARTING=1 stops Forge from opening a browser tab (webui.py:98-103).
            env=paths.clean_env(paths.FORGE, {"SD_WEBUI_RESTARTING": "1"}),
            ready_timeout=600.0,
        )

    async def health(self, client: httpx.AsyncClient, base_url: str) -> bool:
        """Ready only once Forge has finished starting, not merely when its API answers.

        webui.py serves HTTP while it is still adding routes: /sdapi/v1/* answers as soon as
        Api.__init__ has registered it, but Api.__init__ then builds the default script args, and
        the refiner script's ui() rescans the checkpoint folders (list_models() clears the list and
        refills it) four times - so a model list read right after /progress answers can come back
        partial. /sd_extra_networks/metadata is added after create_api() returns (webui.py:141-143);
        /sdapi/v1/loras comes from the built-in LoRA extension's app_started callback, the last
        startup step in both the UI and the --nowebui (API-only) mode.
        """
        try:
            r = await client.get(f"{base_url}{_API}/progress", params={"skip_current_image": "true"}, timeout=5.0)
            if r.status_code != 200:
                return False
            for route in ("/sd_extra_networks/metadata", f"{_API}/loras"):
                if (await client.get(f"{base_url}{route}", timeout=5.0)).status_code == 200:
                    return True
        except httpx.HTTPError:
            pass
        return False

    async def on_stop(self, client: httpx.AsyncClient, base_url: str) -> None:
        await _interrupt(client, base_url)

    # ---- models (used by server.py) -----------------------------------------
    async def list_models(self, client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
        """Checkpoints, current checkpoint, samplers and schedulers for the Images stage.

        "name" is the stable id to store in settings.checkpoint ("title" gains a hash suffix
        once Forge has hashed the file). "current" is given in the same form as "name".
        Schedulers are Forge's labels ("Karras"); the API accepts label or id
        (modules/sd_schedulers.py:289). "modules" lists the VAE / text-encoder files that
        params["modules"] may name.
        """
        models, options, samplers, schedulers, modules = await asyncio.gather(
            *(_get_json(client, f"{base_url}{_API}/{route}")
              for route in ("sd-models", "options", "samplers", "schedulers", "sd-modules")))
        current_id = options.get("sd_model_checkpoint")
        current = _find_checkpoint(models, current_id) if current_id else None
        checkpoints = sorted(({"title": m["title"], "name": _checkpoint_name(m)} for m in models),
                             key=lambda c: c["name"].lower())   # Forge lists them in folder-scan order
        return {
            "checkpoints": checkpoints,
            "current": _checkpoint_name(current) if current else None,
            "samplers": [s["name"] for s in samplers],
            "schedulers": [s["label"] for s in schedulers],
            "modules": [m["model_name"] for m in modules],
        }

    async def refresh_models(self, client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
        """Rescan the checkpoint folders, then list_models().

        Forge rescans under its queue lock (modules/api/api.py:745-747), so this waits for a
        running generation to finish. New VAE / text-encoder files need a Forge restart.
        """
        try:
            r = await client.post(f"{base_url}{_API}/refresh-checkpoints", timeout=300.0)
        except httpx.HTTPError as e:
            raise StageError(UNREACHABLE) from e
        if r.status_code != 200:
            raise StageError(f"Forge could not rescan its model folders: {_error_text(r)}")
        return await self.list_models(client, base_url)

    # ---- work ---------------------------------------------------------------
    async def run_stage(self, ctx: StageContext, prop: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        images: dict[str, Any] = prop.get("images") or {}
        # img2img takes one picture: the first example picture of the prop (Qwen-Image takes them all)
        references = reference_files(ctx, images, params)
        reference = references[0] if references else None
        prompt = str(params.get("prompt") or "").strip()
        if not prompt and reference is None:
            raise StageError("Type a prompt that describes the object (or add an example picture), then press "
                             "Generate.")

        init_image: str | None = None
        if reference is not None:
            try:
                init_image = base64.b64encode(reference.read_bytes()).decode("ascii")
            except OSError as e:
                raise StageError(f"Could not read this prop's example picture ({e.strerror}). "
                                 "Close any program that has it open, or add it again.") from e
        payload = _build_payload(params, prompt, init_image)
        count = payload["n_iter"]
        overrides = payload["override_settings"]
        checkpoint = await self._checkpoint_id(ctx, params.get("checkpoint"))
        if checkpoint is not None:
            overrides["sd_model_checkpoint"] = checkpoint
        # Flux / Z-Image / Qwen-style models need separate VAE and text-encoder files. Forge keeps
        # them in the "forge_additional_modules" option; set_config() applies it from
        # override_settings like a checkpoint change (modules/sysinfo.py:229-232,
        # modules_forge/main_entry.py:157-177), matching file names against /sdapi/v1/sd-modules.
        # Unknown names would be dropped silently, so they are checked first. params["modules"]
        # absent/None leaves Forge's current modules alone (what Forge's own checkpoint
        # dropdown does); [] clears them, which single-file SDXL checkpoints want.
        modules = params.get("modules")
        if modules is not None:
            overrides["forge_additional_modules"] = await self._module_names(ctx, modules)

        ctx.report(0.0, "Sending the job to Forge...")
        data, keep = await _generate(ctx, "img2img" if init_image else "txt2img", payload, count)
        new_rel, seeds = _save_images(ctx, images, data, keep)
        if not new_rel and keep is None:
            raise StageError("Forge finished without returning an image (was it stopped from the full Forge UI?). "
                             "Press Generate again.")
        ctx.report(1.0, f"{len(new_rel)} image(s) done." if keep is None else "Cancelled.")
        return results_patch(images, new_rel, seeds)

    async def _checkpoint_id(self, ctx: StageContext, wanted: Any) -> str | None:
        """The Forge alias to request for params["checkpoint"], or None = keep Forge's current model."""
        url = f"{ctx.base_url}{_API}/sd-models"
        models = await _get_json(ctx.client, url)
        match = _find_checkpoint(models, str(wanted)) if wanted else None
        if not models or (wanted and match is None):
            # Forge rescans its model folders whenever its own page loads (the full Forge UI
            # overlay) by clearing the list and refilling it; a read in that moment is partial.
            await asyncio.sleep(1.5)
            models = await _get_json(ctx.client, url)
            match = _find_checkpoint(models, str(wanted)) if wanted else None
        if not models:
            raise StageError(NO_MODEL)
        if not wanted:
            return None
        if match is None:
            raise StageError(f"The image model \"{wanted}\" is not in Forge's model list. Pick another model, "
                             "or press Refresh if you just added or renamed the file.")
        return _checkpoint_name(match)

    async def _module_names(self, ctx: StageContext, wanted: Any) -> list[str]:
        names = [wanted] if isinstance(wanted, str) else [str(m) for m in wanted]
        if not names:
            return []
        known = {m["model_name"] for m in await _get_json(ctx.client, f"{ctx.base_url}{_API}/sd-modules")}
        missing = [n for n in names if Path(n).name not in known]
        if missing:
            raise StageError(f"Forge cannot find the VAE / text encoder file(s): {', '.join(missing)}. Put them into "
                             "the models\\vae or models\\text-encoders folder of Spellcast3D, then restart Forge "
                             "(Settings > Debugging: Stop; the next image starts it again).")
        return [Path(n).name for n in names]


# ---- helpers ------------------------------------------------------------------

def _build_payload(params: dict[str, Any], prompt: str, init_image: str | None) -> dict[str, Any]:
    """txt2img body, or img2img when init_image (base64) is given. Field names: StableDiffusionProcessing
    and ...Img2Img (modules/processing.py:142-190, :1657-1671) plus the API extras
    init_images / send_images / save_images (modules/api/models.py:83-114). The prompt and the negative prompt
    go as the artist typed them."""
    negative = str(params.get("negative_prompt") or "").strip()
    payload: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative,
        "steps": max(1, _number(params, "steps", 25)),
        "cfg_scale": _number(params, "cfg_scale", 6.0, float),
        "width": _dimension(_number(params, "width", 1024)),
        "height": _dimension(_number(params, "height", 1024)),
        "seed": _number(params, "seed", -1),
        "batch_size": 1,
        "n_iter": max(1, _number(params, "batch_count", 1)),
        "send_images": True,
        "save_images": False,       # also disables the grid image (api.py:451-452, processing.py:1146)
        "override_settings": {},
        # keep the checkpoint loaded between props instead of switching back after every request
        "override_settings_restore_afterwards": False,
    }
    if init_image is not None:
        payload["init_images"] = [init_image]
        payload["denoising_strength"] = min(1.0, max(0.0, _number(params, "denoising_strength", 0.65, float)))
        # 2 = "Resize and fill": keeps the whole reference visible instead of stretching it
        # to the output size (modules/images.py:257-325).
        payload["resize_mode"] = 2
    # Unset sampler / scheduler: Forge's defaults ("Euler", "Automatic"; api.py:458).
    if params.get("sampler"):
        payload["sampler_name"] = str(params["sampler"])
    if params.get("scheduler"):
        payload["scheduler"] = str(params["scheduler"])
    return payload


_LABELS = {"steps": "Steps", "cfg_scale": "CFG", "width": "Width", "height": "Height", "seed": "Seed",
           "batch_count": "Variations", "denoising_strength": "Reference strength"}


def _number(params: dict[str, Any], key: str, default: Any, kind: type = int) -> Any:
    value = params.get(key)
    if value is None or value == "":
        return default
    try:
        return kind(float(value))
    except (TypeError, ValueError):
        raise StageError(f"\"{_LABELS[key]}\" must be a number, not \"{value}\". "
                         "Fix it and press Generate again.") from None


def _dimension(value: int) -> int:
    """Forge works on a latent grid of 8 px; keep sizes on it."""
    return max(64, value // 8 * 8)


async def _generate(ctx: StageContext, route: str, payload: dict[str, Any],
                    count: int) -> tuple[dict[str, Any], int | None]:
    """POST the job and follow its progress.

    Returns (response JSON, keep): keep is None normally, or - after the user cancelled and Forge
    was interrupted - how many leading images were fully finished and are worth keeping.
    """
    loop = asyncio.get_running_loop()
    request = asyncio.create_task(ctx.client.post(f"{ctx.base_url}{_API}/{route}", json=payload, timeout=None))
    finished = 0
    sampling = False
    cancel_started: float | None = None
    try:
        while True:
            done, _ = await asyncio.wait({request}, timeout=_POLL_INTERVAL)
            if done:
                break
            if ctx.cancelled():
                if cancel_started is None:
                    cancel_started = loop.time()
                    ctx.report(finished / count, "Cancelling...")
                elif loop.time() - cancel_started > _CANCEL_GRACE:
                    return {}, 0
                # Re-sent on every poll: state.begin() clears the flag if the job had not started yet
                # (modules/shared_state.py:121-137).
                await _interrupt(ctx.client, ctx.base_url)
                continue
            state = await _progress_state(ctx)
            if state is None:
                continue
            job_count, job_no = int(state.get("job_count") or 0), int(state.get("job_no") or 0)
            if job_count <= 0:
                # -1 between state.begin() and the first image (model loading, prompt encoding);
                # 0 before the job starts and again after it ended, while the images are sent back.
                if not sampling:
                    ctx.report(0.0, "Preparing (loading the image model can take a minute)...")
                continue
            sampling = True
            finished = min(job_no, count)   # job_no counts fully decoded images (processing.py:1024)
            steps = int(state.get("sampling_steps") or 0)
            step = int(state.get("sampling_step") or 0)
            fraction = (job_no + (step / steps if steps else 0.0)) / job_count
            ctx.report(min(fraction, 0.99), f"Image {min(job_no + 1, count)} of {count}...")
        response = request.result()
    except httpx.HTTPError as e:
        if cancel_started is not None:
            return {}, 0
        raise StageError("Lost the connection to Forge while it was making the image; it may have crashed "
                         "(often: not enough memory). Check the Forge log, start Forge again and retry.") from e
    finally:
        if not request.done():
            request.cancel()

    keep = None if cancel_started is None else finished
    if response.status_code != 200:
        if keep is not None:
            return {}, 0
        raise _generation_error(response)
    try:
        data = response.json()
    except ValueError as e:
        raise StageError("Forge sent back an answer that is not an image result. Check the Forge log.") from e
    return data, keep


def _save_images(ctx: StageContext, images: dict[str, Any], data: dict[str, Any],
                 keep: int | None) -> tuple[list[str], list[int]]:
    """Write the returned images as images/NNNN.png; returns (relative paths, their seeds)."""
    encoded: list[str] = data.get("images") or []
    try:
        info = json.loads(data.get("info") or "{}")
    except ValueError:
        info = {}
    # Response images = [grid?] + one per generated image + script extras (api.py:485, processing.py:1146-1172);
    # infotexts has one entry per generated image (+ the grid).
    first = int(info.get("index_of_first_image") or 0)
    real = encoded[first:len(info.get("infotexts") or encoded)]
    if keep is not None:
        real = real[:keep]          # the image being sampled at the interrupt is only half done
    seeds = [int(s) for s in (info.get("all_seeds") or [])][:len(real)]
    # In Forge's "samples_format" option (api.py:101-129): png unless the user changed it.
    return save_candidates(ctx, images, (base64.b64decode(b64) for b64 in real)), seeds


def _generation_error(response: httpx.Response) -> StageError:
    text = _error_text(response)
    if "out of memory" in text.lower() or "OutOfMemory" in text:
        return StageError("The GPU ran out of memory (VRAM) while making the image. Try a smaller image size, "
                          "fewer variations, or close other programs that use the GPU (games, browsers with "
                          "3D tabs, other AI tools), then try again.")
    return StageError(f"Forge could not make the image: {text}")


def _error_text(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code} {response.text[:300]}".strip()
    if not isinstance(body, dict):
        return f"HTTP {response.status_code} {body}"
    message = body.get("message") or body.get("detail") or body.get("errors") or body.get("error")
    if isinstance(message, list):       # FastAPI request-validation errors
        message = "; ".join(str(m.get("msg", m)) if isinstance(m, dict) else str(m) for m in message)
    kind = body.get("error")
    if kind and message and kind != message:
        return f"{kind}: {message}"
    return str(message or f"HTTP {response.status_code}")


async def _get_json(client: httpx.AsyncClient, url: str) -> Any:
    try:
        r = await client.get(url, timeout=30.0)
    except httpx.HTTPError as e:
        raise StageError(UNREACHABLE) from e
    if r.status_code != 200:
        raise StageError(f"Forge answered with an error: {_error_text(r)}")
    return r.json()


async def _progress_state(ctx: StageContext) -> dict[str, Any] | None:
    try:
        r = await ctx.client.get(f"{ctx.base_url}{_API}/progress", params={"skip_current_image": "true"}, timeout=5.0)
        return r.json().get("state") if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


async def _interrupt(client: httpx.AsyncClient, base_url: str) -> None:
    try:
        await client.post(f"{base_url}{_API}/interrupt", timeout=3.0)
    except httpx.HTTPError:
        pass


def _checkpoint_name(model: dict[str, Any]) -> str:
    """The checkpoint's file name relative to its models folder: its title without the " [hash]" suffix
    (modules/sd_models.py:83). Always one of Forge's checkpoint aliases."""
    title, short = model.get("title") or "", model.get("hash")
    suffix = f" [{short}]"
    return title[:-len(suffix)] if short and title.endswith(suffix) else title


def _find_checkpoint(models: list[dict[str, Any]], wanted: str) -> dict[str, Any] | None:
    for m in models:
        if wanted in (m.get("title"), _checkpoint_name(m), m.get("model_name"), Path(m.get("filename") or "").name):
            return m
    return None
