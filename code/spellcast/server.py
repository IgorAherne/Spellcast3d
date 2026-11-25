"""The hub's FastAPI app - the web UI's only backend. Routes: ARCHITECTURE.md "Hub HTTP API".

Every error reaches the browser as {"detail": "<message for the artist>"}.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import mimetypes
import os
import secrets
import shutil
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO, Any, AsyncIterator, Callable, Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit

import httpx
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile as FormFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__, config, dialogs, exports, gpu, model_downloads, multiview, paths, promptlib, propfiles
from .adapters.base import StageError, ToolAdapter
from .adapters.qwen import FLUX3_NEEDS_ENGINE
from .install import sdcpp as engine_pin
from .installer_bg import IDLE, BackgroundInstall, tail_lines
from .jobs import JobError, JobQueue
from .procman import INSTALLING, NOT_INSTALLED, READY, UNSUPPORTED, ProcessManager, ToolError, ToolProcess
from .projects import (MAX_IMAGE_UPLOADS, MAX_IMPORT_FILES, MAX_MODEL_UPLOAD_BYTES, MAX_PAINT_BYTES, MAX_PAINT_SIZE,
                       MAX_REFERENCES, MAX_THUMB_BYTES, MAX_UPLOAD_BYTES, MAX_VIEWPORT_BYTES, ProjectError,
                       ProjectStore, box_name_from_file, save_model_files, omnitexture_fresh, omnitexture_own,
                       retopo_size_of, too_many_references)
from .settings import SettingsError, SettingsStore

log = logging.getLogger(__name__)

# nvidia-smi takes ~50 ms per call; the header's VRAM meter is informational and the page polls every 2.5 s.
GPU_CACHE_SECONDS = 5.0
LOOPBACK_NAMES = {"127.0.0.1", "localhost", "::1"}
# Sec-Fetch-Site values of requests the hub's own page (or the address bar) makes.
OWN_SITE = {"same-origin", "none"}
MEDIA_TYPES = {
    ".glb": "model/gltf-binary", ".gltf": "model/gltf+json", ".obj": "text/plain", ".mtl": "text/plain",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".json": "application/json", ".txt": "text/plain",
}
INTERNAL_ERROR = "Something went wrong inside Spellcast3D. The details are in code/logs/hub.log."
# Multipart framing around the model files of one import (names, boundaries): far below this.
UPLOAD_OVERHEAD = 8 * 1024 * 1024
MAX_MASK_BYTES = 32 * 1024 * 1024      # what the artist masked for an Omni-Texture repaint: a PNG of the texture's size
OVERRIDE_KEYS = {"role", "flip_green", "invert"}
PROJECT_BUSY = ("Spellcast3D is still working on this project (a job, an import or a save is not done yet). "
                "Wait for it to finish, or cancel the job, then try again.")
LOCAL_CLIENTS = ("127.0.0.1", "::1")
# The Instant Meshes viewer as the Retopo stage embeds it (instant-meshes-py's /viewer options): its panel
# on the right, and an "Accept" button that asks the page (host_origin, added per request) instead of
# downloading the mesh.
VIEWER_OPTIONS = {"panel": "right", "export_label": "Accept", "export_action": "host"}
NO_IMAGE_GENERATOR = f"There is no image generator, because {paths.CONFIG_FILE.name} is not right."
FIX_CONFIG = "Fix it, save the file, then close Spellcast3D and start it again."
# The Omni-Texture stage paints with Qwen-Image 2.1 alone: refused while the config file picks Forge Neo (or nothing).
NO_OMNITEXTURE = (f"Omni-Texture paints with Qwen-Image 2.1, which {paths.CONFIG_FILE.name} does not pick as the "
                   "image generator. To use it, switch on the line backend = qwen there, save the file, then close "
                   "Spellcast3D and start it again.")
BEING_BAKED = ("This prop is being baked again, so the painting was not saved: the new bake replaces these textures. "
               "Paint once the bake is done.")
# The painting's raw pixels wait in the prop's folder under this name while the worker reads them.
PAINT_TEMP = ".paint-"


def config_changed(started: config.Config, now: config.Config) -> str:
    """What the hub says while the root config file picks something else than it started with: the new choice, and
    that it applies at the next start (it installs what a new choice needs)."""
    name = paths.CONFIG_FILE.name
    if now.backend is None:
        return f"{name} was changed and has a mistake now: {config.short(now.problems[0]) if now.problems else ''}" \
               f" {FIX_CONFIG}".replace("  ", " ")
    running = config.BACKEND_NAMES[started.backend] if started.backend is not None else ""
    if now.backend == started.backend:          # Qwen-Image's model size
        return (f"{name} now picks Qwen-Image 2.1's model size {now.quality}. Close Spellcast3D and start it again to "
                f"use it (it downloads that size first; {started.quality} runs until then).")
    return (f"{name} now picks {config.describe(now)}. Close Spellcast3D and start it again to switch (it installs "
            f"what that needs" + (f"; {running} runs until then)." if running else ")."))


def config_status(cfg: config.Config) -> dict[str, Any]:
    """The root config file (spellcast3d-config.ini, spellcast/config.py) for /api/status: {"image_backend": "qwen" |
    "forge" | None (the file has a mistake: no image generator is registered), "quality": Qwen-Image's model size,
    "problems": [what is wrong with the file, in the artist's words: without the sentence that no image generator is
    installed until it is fixed, which the page says in its own place], "changed": None (the routes add what
    ConfigWatch says)}."""
    return {"image_backend": cfg.backend, "quality": cfg.quality, "problems": [config.short(p) for p in cfg.problems],
            "changed": None}


def _config_choice(cfg: config.Config) -> tuple[str | None, str | None]:
    """What the config file switches: the image generator, and Qwen-Image's model size when it is the one."""
    return cfg.backend, cfg.quality if cfg.backend == paths.QWEN else None


class ConfigWatch:
    """Notices the root config file picking something else while the hub runs. Nothing switches over until the next
    start (the installer fetches what a new choice needs, the hub registers the image generator it picks), so
    /api/status settings_warnings then says to start again. Saving it unchanged, or switching back, says nothing."""

    def __init__(self, started_with: config.Config):
        self._started = started_with
        self._started_with = _config_choice(started_with)

    def warnings(self) -> list[str]:
        try:
            now = config.load()         # read again only when the file was saved (one stat call otherwise)
        except Exception:
            log.exception("Could not read the config file")
            return []
        return [config_changed(self._started, now)] if _config_choice(now) != self._started_with else []


class _NoStoreStatic(StaticFiles):
    """The UI must never come from the browser cache after an update."""

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-store"
        return response


@dataclass
class Hub:
    """Everything the routes share. procman / jobs / client exist while the app is running."""
    adapters: dict[str, ToolAdapter]
    settings: SettingsStore
    store: ProjectStore
    logs_dir: Path
    gpu_probe: Callable[[], gpu.GpuInfo | None]
    hub_port: int | None
    show_in_explorer: Callable[[Path, bool], bool]    # exports.show_in_explorer, or a stand-in in tests
    repaint: exports.Repainter                        # exports.run_repaint_worker, or a stand-in in tests
    install: BackgroundInstall | None = None          # the rest of the installation, run in the background
    multiview_download: multiview.BackgroundDownload | None = None     # Pixal3D's multi-view models, on request
    qwen_full_download: model_downloads.BackgroundDownload | None = None    # the full Qwen-Image 2.1, on request
    flux3_download: model_downloads.BackgroundDownload | None = None        # FLUX.3 Image Klein, on request
    hi3d_download: model_downloads.BackgroundDownload | None = None         # Hi3D's AI models, on request
    config: dict[str, Any] = field(default_factory=dict)              # config_status() when the hub started
    config_watch: ConfigWatch | None = None
    # stage -> the registered tools that run it (paths.STAGE_TOOLS order): the Images stage has the one image
    # generator the config file picks (none when the file is not valid)
    stage_tools: dict[str, list[str]] = field(default_factory=dict)
    fading: set[tuple[str, str]] = field(default_factory=set)      # (project, prop): its Omni-Texture being made
                                                                    # again without the image generator
    # (project, prop) -> hand paintings of its Omni-Texture being saved
    painting: collections.Counter[tuple[str, str]] = field(default_factory=collections.Counter)
    procman: ProcessManager = field(init=False)
    jobs: JobQueue = field(init=False)
    client: httpx.AsyncClient = field(init=False)
    exporter: exports.Exporter = field(init=False)
    gpu_lock: asyncio.Lock = field(init=False)
    imesh_lock: asyncio.Lock = field(init=False)
    import_lock: asyncio.Lock = field(init=False)    # one model import at a time (RAM, and its cleanup)
    export_lock: asyncio.Lock = field(init=False)    # one save into the export folders at a time (its caches)
    # project -> model imports or saves in progress: its files must stay where they are
    importing: collections.Counter[str] = field(default_factory=collections.Counter)
    _gpu_cache: dict[str, Any] | None = None
    _gpu_time: float = -GPU_CACHE_SECONDS

    async def gpu_info(self) -> dict[str, Any] | None:
        async with self.gpu_lock:
            if time.monotonic() - self._gpu_time >= GPU_CACHE_SECONDS:
                try:
                    info = await asyncio.to_thread(self.gpu_probe)
                except Exception:
                    log.exception("GPU query failed")
                    info = None
                self._gpu_cache = info.to_dict() if info else None
                self._gpu_time = time.monotonic()
            return self._gpu_cache

    def adapter(self, tool_id: str) -> ToolAdapter:
        adapter = self.adapters.get(tool_id)
        if adapter is None:
            raise ToolError(f'There is no tool called "{tool_id}".', 404)
        return adapter

    def running(self, tool_id: str, not_running: str) -> ToolProcess:
        """The tool's process if it is ready, else a 409 with `not_running` as the message."""
        tp = self.procman.get(tool_id)
        if tp.state != READY or not tp.alive():
            raise ToolError(not_running, 409)
        return tp

    def busy_projects(self) -> set[str]:
        """Projects with a job waiting or running, or a model import or a save in progress: their files must
        stay where they are (no rename, no tidying away)."""
        return self.jobs.active_projects() | {p for p, n in self.importing.items() if n}

    def check_omnitexture_idle(self, project: str, prop_id: str) -> None:
        """409 while the prop's steps may change (check_steps_stay) or a hand painting of it is being saved: before
        its steps are made again (fade_omnitexture, take_off_omnitexture)."""
        self.check_steps_stay(project, prop_id)
        self.check_not_painting(project, [prop_id])

    def check_steps_stay(self, project: str, prop_id: str) -> None:
        """409 while an Omni-Texture job of the prop waits or runs (its result replaces the one shown), or its
        Omni-Texture is being made again without the image generator (fade_omnitexture, take_off_omnitexture):
        the steps Ctrl+Z walks, masks and paints on may go."""
        if prop_id in self.jobs.active_props(project, "omnitexture"):
            raise ProjectError("This prop is being OmniTextured. Wait until it is done.", 409)
        self.check_not_fading(project, [prop_id])

    def check_not_fading(self, project: Any, prop_ids: Any) -> None:
        """409 while the Omni-Texture of one of the props is being made again without the image generator (a
        repaint's fade, a picture taken off): that and an Omni-Texture job would write the prop's next result
        folder both."""
        if any((project, prop_id) in self.fading for prop_id in prop_ids if isinstance(prop_id, str)):
            raise ProjectError("This prop's Omni-Texture is being changed (a fade, or a picture taken off). Wait "
                               "until it is done.", 409)

    def check_not_painting(self, project: Any, prop_ids: Any) -> None:
        """409 while a hand painting of one of the props' Omni-Texture is being saved: a job, a fade or a picture
        taken off would read that model before the painting is in it."""
        if any(self.painting[(project, prop_id)] for prop_id in prop_ids if isinstance(prop_id, str)):
            raise ProjectError("This prop's Omni-Texture is being painted on: your painting is being saved. Try "
                               "again in a moment.", 409)

    async def fade_omnitexture(self, project: str, prop_id: str, glb: Any, fade: Any,
                                prepare: bool = False) -> dict[str, Any]:
        """A repaint of the prop's Omni-Texture with its edge faded `fade` times as far (adapters/omnitexture.py
        refade: the worker alone, no image generator) -> {"prop", "renamed": {step as it was: step as it is}}.
        prepare: the page picked that repaint - the worker is made ready for its fade -> {"ready": True}."""
        from .adapters import omnitexture

        self.check_omnitexture_idle(project, prop_id)
        if prepare:
            block = self.store.get_prop(project, prop_id)["omnitexture"]
            try:
                await asyncio.to_thread(omnitexture.refade, self.store.prop_dir(project, prop_id), block, glb, None)
            except StageError as e:
                raise ProjectError(str(e), 409) from e
            return {"ready": True}
        if not isinstance(fade, (int, float)) or isinstance(fade, bool) or not 1.0 <= fade <= omnitexture.MAX_FADE:
            raise ProjectError(f"A repaint's fade is a number from 1 to {omnitexture.MAX_FADE:g}.", 422)
        self.fading.add((project, prop_id))
        try:
            block = self.store.get_prop(project, prop_id)["omnitexture"]
            try:
                patch, renamed = await asyncio.to_thread(omnitexture.refade, self.store.prop_dir(project, prop_id),
                                                         block, glb, float(fade))
            except StageError as e:
                raise ProjectError(str(e), 409) from e
            prop = await asyncio.to_thread(self.store.apply_stage_patch, project, prop_id, "omnitexture", patch)
            return {"prop": prop, "renamed": renamed}
        finally:
            self.fading.discard((project, prop_id))

    async def own_omnitexture(self, project: str, prop_id: str) -> dict[str, Any]:
        """The prop's 3D model becomes the first step of a result of its own texture, so that the artist may paint
        on it and mask it without an Omni-Texture job (adapters/omnitexture.py begin_own) -> {"prop"}. A prop that
        has a result keeps it - but for a result of its own texture on a 3D model made again since: that is
        begun anew, on the model the prop has now."""
        from .adapters import omnitexture

        self.check_omnitexture_idle(project, prop_id)
        self.fading.add((project, prop_id))
        try:
            prop = self.store.get_prop(project, prop_id)
            if prop["omnitexture"].get("glb") and (omnitexture_fresh(prop) or not omnitexture_own(prop)):
                return {"prop": prop}
            try:
                patch = await asyncio.to_thread(omnitexture.begin_own, self.store.prop_dir(project, prop_id), prop)
            except StageError as e:
                raise ProjectError(str(e), 409) from e
            return {"prop": await asyncio.to_thread(self.store.apply_stage_patch, project, prop_id, "omnitexture",
                                                    patch)}
        finally:
            self.fading.discard((project, prop_id))

    async def take_off_omnitexture(self, project: str, prop_id: str, glb: Any, view: Any, repaint: Any,
                                    whole: bool = False) -> dict[str, Any]:
        """A painted view (`view`: its folder) or a repaint (`repaint`: the model it made) taken off the model
        shown, `glb` (adapters/omnitexture.py take_off: the worker alone, no image generator) -> {"prop"}: it
        shows a new step without that picture. `whole`: the result itself is taken off - the step that follows is
        none of its models, and the prop shows its 3D model."""
        from .adapters import omnitexture

        self.check_omnitexture_idle(project, prop_id)
        self.fading.add((project, prop_id))
        try:
            prop = self.store.get_prop(project, prop_id)
            if glb != prop["omnitexture"].get("glb"):
                raise ProjectError("This prop's Omni-Texture changed meanwhile. Reload the page and try again.", 409)
            try:
                patch = await asyncio.to_thread(omnitexture.take_off, self.store.prop_dir(project, prop_id), prop,
                                                view, repaint, whole)
            except StageError as e:
                raise ProjectError(str(e), 409) from e
            return {"prop": await asyncio.to_thread(self.store.apply_stage_patch, project, prop_id, "omnitexture",
                                                    patch)}
        finally:
            self.fading.discard((project, prop_id))

    async def paint_omnitexture(self, project: str, prop_id: str, glb: str, width: int, height: int,
                                 data: bytes) -> dict[str, Any]:
        """A hand painting of the prop's Omni-Texture step `glb` (`data`: the page's patch, raw RGBA) saved into its
        model (adapters/omnitexture.py hand_paint: the worker alone, no image generator) -> the prop, whose
        thumbnail and bake follow it as they follow a repaint (ProjectStore.omnitexture_painted). Meanwhile no
        job, fade or picture taken off of the prop starts (check_not_painting)."""
        from .adapters import omnitexture

        self.painting[(project, prop_id)] += 1
        try:
            # One save at a time (an export never copies a model being written; one prop's .paint-* files never
            # meet), and the project is not renamed meanwhile.
            async with self.export_lock:
                with self.importing_into(project):
                    self.check_steps_stay(project, prop_id)
                    await asyncio.to_thread(self.store.omnitexture_paint_target, project, prop_id, glb, width, height)
                    prop_dir = self.store.prop_dir(project, prop_id)
                    raw = prop_dir / f"{PAINT_TEMP}{secrets.token_hex(6)}.rgba"
                    try:
                        await asyncio.to_thread(_write_paint, prop_dir, raw, data)
                        del data
                        await asyncio.to_thread(omnitexture.hand_paint, prop_dir, glb, raw, width, height)
                    except StageError as e:
                        raise ProjectError(str(e), 409) from e
                    finally:
                        await asyncio.to_thread(raw.unlink, missing_ok=True)
                    return await asyncio.to_thread(self.store.omnitexture_painted, project, prop_id, glb)
        finally:
            self.painting[(project, prop_id)] -= 1

    def check_paint(self, project: str, prop_id: str) -> None:
        """409 while a projection job of the prop waits or runs (its bake replaces the textures), or while the
        program that saves the painting (in the Instant Meshes venv) cannot run."""
        if prop_id in self.jobs.active_props(project, "projection"):
            raise ProjectError(BEING_BAKED, 409)
        try:
            self.procman.check_available(paths.IMESH, installing_ok=False)
        except ToolError as e:
            raise ProjectError(f"Saving the painting needs the retopology tool. {e}", 409) from e

    def can_convert(self, fmt: str) -> None:
        """Raise ProjectError when an FBX or OBJ cannot be made now: its worker runs in the Instant Meshes
        venv, which may not be installed yet."""
        try:
            self.procman.check_available(paths.IMESH, installing_ok=False)
        except ToolError as e:
            raise ProjectError(f"Saving as .{fmt} needs the retopology tool. {e}", 409) from e

    def explorer(self, request: Request, target: Path, select: bool) -> bool:
        """Open File Explorer on `target` for a page on this computer (never with SPELLCAST_NO_EXPLORER=1)
        -> whether it was opened. Never waits for it."""
        if request.client is None or request.client.host not in LOCAL_CLIENTS or not exports.explorer_allowed():
            return False
        return self.show_in_explorer(target, select)

    @contextmanager
    def importing_into(self, project: str) -> Iterator[None]:
        self.importing[project] += 1
        try:
            yield
        finally:
            self.importing[project] -= 1


def _host_name(value: str) -> str:
    """'127.0.0.1:7770' -> '127.0.0.1', '[::1]:7770' -> '::1', 'http://localhost:7770' -> 'localhost'."""
    value = value.split("://", 1)[-1].split("/", 1)[0].strip().lower()
    if value.startswith("["):
        return value[1:value.find("]")] if "]" in value else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def _import_overrides(raw: Any) -> dict[str, dict[str, Any]]:
    """The artist's corrections of an imported model's textures, checked:
    {texture key: {"role"?: str, "flip_green"?: bool, "invert"?: bool}} (the tool checks the role names)."""
    if raw is None:
        return {}
    problem = ProjectError('"overrides" must map each texture to {"role", "flip_green", "invert"}. '
                           'Reload the page and try again.')
    if not isinstance(raw, dict) or len(raw) > MAX_IMPORT_FILES:
        raise problem
    out: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict) or set(value) - OVERRIDE_KEYS:
            raise problem
        role = value.get("role")
        if role is not None and not (isinstance(role, str) and 0 < len(role) <= 32):
            raise problem
        if any(value.get(flag) is not None and not isinstance(value[flag], bool) for flag in ("flip_green", "invert")):
            raise problem
        out[key] = {k: v for k, v in value.items() if v is not None}
    return out


def _check_upload_size(request: Request, limit: int, too_big: str) -> None:
    """413 `too_big` before an upload is read when it says it is bigger than `limit` bytes."""
    try:
        length = int(request.headers.get("content-length") or 0)
    except ValueError:
        length = 0
    if length > limit:
        raise ProjectError(too_big, 413)


def _check_model_upload_size(request: Request) -> None:
    """413 before a model upload is read when it says it is too big."""
    _check_upload_size(request, MAX_MODEL_UPLOAD_BYTES + UPLOAD_OVERHEAD,
                       f"These files are too big together (the limit is {MAX_MODEL_UPLOAD_BYTES // 1024 ** 3} GB). "
                       f"Leave out what the model does not need, or use smaller textures.")


def _model_uploads(form: Any) -> list[tuple[str, IO[bytes]]]:
    """(name, stream) of the files of a model upload: the form fields "files" or "files[]", repeated."""
    uploads = [(item.filename or "", item.file) for key in ("files", "files[]") for item in form.getlist(key)
               if isinstance(item, FormFile)]
    if not uploads:
        raise ProjectError("Choose your 3D model (.fbx, .obj, .glb or .gltf) together with its textures, "
                           "or one .zip that contains them.")
    return uploads


async def _read_body(request: Request, limit: int, too_big: str) -> bytes:
    """The raw request body, refused (413 `too_big`) as soon as it passes `limit` bytes."""
    try:
        declared = int(request.headers.get("content-length") or 0)
    except ValueError:
        declared = 0
    if declared > limit:
        raise ProjectError(too_big, 413)
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > limit:
            raise ProjectError(too_big, 413)
    return bytes(data)


async def _read_painting(request: Request, width: int, height: int) -> bytes:
    """A painting's raw RGBA body (PUT .../paint): 413 past MAX_PAINT_BYTES, 422 when it is not width x height x 4
    bytes."""
    data = await _read_body(request, MAX_PAINT_BYTES, f"The painting is too big. A texture can be {MAX_PAINT_SIZE} x "
                                                      f"{MAX_PAINT_SIZE} pixels at most.")
    if len(data) != width * height * 4:
        raise ProjectError(f"The painting arrived incomplete ({len(data)} of {width * height * 4} bytes), so it was "
                           f"not saved. Save it again.", 422)
    return data


def _write_paint(prop_dir: Path, raw: Path, data: bytes) -> None:
    """The painting's raw pixels into `raw` in the prop's folder; what an interrupted save left there goes first."""
    for old in prop_dir.glob(f"{PAINT_TEMP}*"):
        old.unlink(missing_ok=True)
    raw.write_bytes(data)


def _page_origin(request: Request, hub_port: int | None, allow_remote: bool) -> str:
    """The origin of the hub's page that sent `request` ("http://127.0.0.1:7770"), for the viewer's
    host_origin: the browser's Origin header when it is a plain http(s) origin of this computer (of any
    host when the hub listens on the network), else the hub's own address."""
    origin = request.headers.get("origin") or ""
    try:
        parts = urlsplit(origin)
        port = parts.port
    except ValueError:
        parts, port = None, None
    if parts is not None and parts.scheme in ("http", "https") and parts.hostname and not parts.username \
            and parts.path in ("", "/") and not parts.query and not parts.fragment \
            and (allow_remote or parts.hostname in LOOPBACK_NAMES):
        host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
        default = 80 if parts.scheme == "http" else 443
        return f"{parts.scheme}://{host}" + (f":{port}" if port is not None and port != default else "")
    port = hub_port or request.url.port
    return f"http://127.0.0.1:{port}" if port else "http://127.0.0.1"


def _viewer_url(base_url: str, viewer_path: str, host_origin: str) -> str:
    """The launcher's viewer path on `base_url`, with the viewer options of the Retopo stage appended."""
    parts = urlsplit(viewer_path)
    query = [*parse_qsl(parts.query, keep_blank_values=True),
             *VIEWER_OPTIONS.items(), ("host_origin", host_origin)]
    return f"{base_url}{parts.path}?{urlencode(query)}"


def create_app(adapters: dict[str, ToolAdapter] | None = None, *,
               settings: SettingsStore | None = None,
               store: ProjectStore | None = None,
               static_dir: Path = paths.STATIC,
               logs_dir: Path = paths.LOGS,
               hub_port: int | None = None,
               allow_remote: bool = False,
               gpu_probe: Callable[[], gpu.GpuInfo | None] = gpu.primary_gpu,
               installer: BackgroundInstall | None = None,
               multiview_download: multiview.BackgroundDownload | None = None,
               qwen_full_download: model_downloads.BackgroundDownload | None = None,
               flux3_download: model_downloads.BackgroundDownload | None = None,
               hi3d_download: model_downloads.BackgroundDownload | None = None,
               show_in_explorer: Callable[[Path, bool], bool] = exports.show_in_explorer,
               converter: exports.Converter = exports.run_export_worker,
               repainter: exports.Repainter = exports.run_repaint_worker,
               prompt_library: Path | None = None) -> FastAPI:
    """The hub app. Tests pass fake adapters / temp stores; `python -m spellcast` uses the real ones, with the
    one image generator the root config file picks (spellcast3d-config.ini, read once here: a change applies at the
    next start, and /api/status says so meanwhile).

    allow_remote: the server listens on a non-loopback address, so the Host-header check
    (protection against DNS rebinding) is switched off.
    installer: installs what is missing or changed in the background while the hub runs (None: nothing
    is installed, the tools are used as they are).
    multiview_download: downloads Pixal3D's multi-view models when the page asks for them (default: into the
    models folder, waiting for `installer` to finish with Pixal3D; tests pass one of their own).
    qwen_full_download: the same for the full Qwen-Image 2.1 of the size the config file picks (default: into
    models/qwen-image, waiting for `installer` to finish with Qwen-Image; tests pass one of their own).
    flux3_download: the same for FLUX.3 Image Klein of that size (default: into models/flux3-klein; tests pass one).
    show_in_explorer(path, select) opens File Explorer after a save; converter makes the FBX and OBJ
    files, repainter saves a painting into a bake (exports.py). Tests pass stand-ins. prompt_library: the prompt
    library's file (paths.PROMPT_LIBRARY).
    """
    library_file = prompt_library or paths.PROMPT_LIBRARY
    settings = settings or SettingsStore()
    # What spellcast3d-config.ini picked as the hub started, one snapshot for all: the image generator registered
    # here, Forge's folder warnings (SettingsStore.config) and the one "start again" warning (ConfigWatch).
    started_with = settings.config if settings.config is not None else config.load()
    for problem in started_with.problems:      # in hub.log too, not only on the page
        log.warning("%s", problem)
    if adapters is None:
        from .adapters import build_adapters
        # "" (the file has a mistake) registers no image generator: a typo must not start one the artist did not pick
        adapters = build_adapters(image_backend=started_with.backend or "")
    if multiview_download is None:
        multiview_download = multiview.BackgroundDownload(install=installer, log_file=logs_dir / multiview.LOG_NAME)
    # The optional models' downloads, in the size and the folders the registered Qwen-Image runs with (as the hub
    # started: the config file's; FLUX.3 in Qwen-Image's size too, in its own folder, since its launcher serves it)
    qwen = adapters.get(paths.QWEN)
    quality = getattr(qwen, "quality", None)
    quality = quality if quality in config.QUALITIES else started_with.quality

    def optional_download(model: str, folder_attr: str) -> model_downloads.BackgroundDownload:
        folder = getattr(qwen, folder_attr, None)
        return model_downloads.BackgroundDownload(model, quality, folder if isinstance(folder, Path) else None,
                                                  install=installer,
                                                  log_file=logs_dir / model_downloads.MODELS[model].log_name)

    if qwen_full_download is None:
        qwen_full_download = optional_download("full", "models_dir")
    if flux3_download is None:
        flux3_download = optional_download("flux3", "flux3_dir")
    if hi3d_download is None:       # the 3D stage's: into models/hi3d, then Hi3D's convolutions measured
        hi3d_download = model_downloads.BackgroundDownload(
            "hi3d", quality, paths.MODELS_HI3D, install=installer,
            log_file=logs_dir / model_downloads.MODELS["hi3d"].log_name, then=model_downloads.hi3d_calibration)
    hub = Hub(adapters=adapters, settings=settings, store=store or ProjectStore(),
              logs_dir=logs_dir, gpu_probe=gpu_probe, hub_port=hub_port, show_in_explorer=show_in_explorer,
              repaint=repainter, install=installer, multiview_download=multiview_download,
              qwen_full_download=qwen_full_download, flux3_download=flux3_download, hi3d_download=hi3d_download,
              config=config_status(started_with),
              config_watch=ConfigWatch(started_with),
              stage_tools={stage: [t for t in tools if t in adapters and stage in adapters[t].stages]
                           for stage, tools in paths.STAGE_TOOLS.items()})

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        hub.gpu_lock, hub.imesh_lock, hub.import_lock = asyncio.Lock(), asyncio.Lock(), asyncio.Lock()
        hub.export_lock = asyncio.Lock()
        if hub.install is not None:
            hub.install.start()
        # trust_env=False: every call goes to 127.0.0.1, a system proxy must never see it.
        hub.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0), trust_env=False)
        hub.procman = ProcessManager(hub.adapters, hub.settings, hub.client, hub_port=hub_port, logs_dir=logs_dir,
                                     install=hub.install)
        await hub.procman.startup()
        hub.jobs = JobQueue(hub.procman, hub.store, hub.settings, hub.adapters, hub.client)
        hub.jobs.start()
        hub.exporter = exports.Exporter(hub.store, converter, hub.can_convert)
        await asyncio.to_thread(hub.store.tidy_untitled)     # empty untitled projects of earlier sessions
        await asyncio.to_thread(hub.store.clear_work_dirs)   # what the tools' jobs left in projects/*/temp/
        await asyncio.to_thread(hub.store.clear_partial_meshes)  # and the 3D models they were writing
        try:
            yield
        finally:
            log.info("Shutting down: stopping all tools…")
            await hub.jobs.close()
            await hub.procman.stop_all()
            from .adapters import omnitexture
            await asyncio.to_thread(omnitexture.close_fades)         # the edge fade's kept worker
            if hub.install is not None:
                await hub.install.close()
            if hub.multiview_download is not None:
                await hub.multiview_download.close()
            for download in (hub.qwen_full_download, hub.flux3_download, hub.hi3d_download):
                if download is not None:
                    await download.close()
            await hub.client.aclose()

    app = FastAPI(title="Spellcast3D", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.hub = hub

    # ---- errors ----------------------------------------------------------------------------
    async def user_error(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=getattr(exc, "status_code", 400))

    for error_type in (ProjectError, ToolError, JobError, SettingsError):
        app.add_exception_handler(error_type, user_error)

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        first = errors[0] if errors else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body") or "request"
        return JSONResponse({"detail": f"The request was not understood ({where}: {first.get('msg', 'invalid')}). "
                                       f"Reload the page and try again."}, status_code=422)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        log.error("Unhandled error in %s %s", request.method, request.url.path, exc_info=exc)
        return JSONResponse({"detail": INTERNAL_ERROR}, status_code=500)

    @app.middleware("http")
    async def local_only(request: Request, call_next: Callable[[Request], Any]) -> Response:
        """Refuse requests addressed to another host name (DNS rebinding) or sent by other websites.

        A web page's <img>/<script> requests carry no Origin, but browsers mark every request with
        Sec-Fetch-Site: anything but the hub's own page (or the address bar) is refused, except
        opening the hub itself from a link (a top-level page load).
        """
        if not allow_remote:
            origin = request.headers.get("origin")
            if _host_name(request.headers.get("host", "")) not in LOOPBACK_NAMES or \
                    (origin is not None and _host_name(origin) not in LOOPBACK_NAMES):
                return JSONResponse({"detail": "Spellcast3D only accepts requests from this computer. "
                                               "Open it at http://127.0.0.1 instead."}, status_code=403)
            site = request.headers.get("sec-fetch-site")
            opened_from_link = (request.method == "GET" and request.headers.get("sec-fetch-mode") == "navigate"
                                and request.headers.get("sec-fetch-dest") == "document")
            if site is not None and site not in OWN_SITE and not opened_from_link:
                return JSONResponse({"detail": "Spellcast3D only accepts requests from its own page."},
                                    status_code=403)
        return await call_next(request)

    # ---- UI and files ----------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index() -> Response:
        page = static_dir / "index.html"
        if not page.is_file():
            return HTMLResponse("<h1>Spellcast3D</h1><p>The web page files are missing "
                                "(code/spellcast/static/index.html). Run tools/update.bat, or download "
                                "Spellcast3D again.</p>", status_code=500)
        return FileResponse(page, media_type="text/html", headers={"Cache-Control": "no-store"})

    app.mount("/static", _NoStoreStatic(directory=static_dir, check_dir=False), name="static")

    @app.get("/files/{project}/{file_path:path}", include_in_schema=False)
    def files(project: str, file_path: str, request: Request) -> Response:
        target = hub.store.resolve_file(project, file_path)
        media = MEDIA_TYPES.get(target.suffix.lower()) or mimetypes.guess_type(target.name)[0] \
            or "application/octet-stream"
        response = FileResponse(target, media_type=media, stat_result=target.stat(),
                                headers={"Cache-Control": "no-cache"})
        etag = response.headers.get("etag")
        if etag and request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "no-cache"})
        return response

    # ---- status and tools ------------------------------------------------------------------
    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        """{"version", "gpu", "settings_warnings", "tools", "stage_tools": {stage: [the registered tools that run
        it]}, "config": config_status() as the hub started, "install", "multiview", "qwen_full", "flux3", "hi3d"}."""
        config_warnings = hub.config_watch.warnings() if hub.config_watch is not None else []
        return {"version": __version__, "gpu": await hub.gpu_info(),
                "settings_warnings": hub.settings.warnings + config_warnings, "tools": hub.procman.all_status(),
                "stage_tools": hub.stage_tools,
                "config": {**hub.config, "changed": config_warnings[0] if config_warnings else None},
                "install": hub.install.snapshot() if hub.install is not None else IDLE,
                "multiview": await asyncio.to_thread(multiview_status),
                "qwen_full": await asyncio.to_thread(qwen_full_status),
                "flux3": await asyncio.to_thread(flux3_status),
                "hi3d": await asyncio.to_thread(hi3d_status)}

    def multiview_status() -> dict[str, Any]:
        """{"available", "downloading", "done_bytes", "total_bytes", "error"} of Pixal3D's multi-view models."""
        assert hub.multiview_download is not None
        return hub.multiview_download.snapshot()

    def qwen_full_status() -> dict[str, Any]:
        """{"available", "downloading", "done_bytes", "total_bytes", "error"} of the full Qwen-Image 2.1."""
        assert hub.qwen_full_download is not None
        return hub.qwen_full_download.snapshot()

    def hi3d_status() -> dict[str, Any]:
        """Hi3D's AI models, the 3D stage's own download (model_downloads "hi3d"): its snapshot ({"available",
        "downloading", "waiting", "done_bytes", "total_bytes", "error", "finishing": the files are there and Hi3D's
        convolutions are being measured}), "available" only with Hi3D's code here as well, and "code": whether it is
        (without it no download makes Hi3D work: the page then says Hi3D is not part of this Spellcast3D)."""
        assert hub.hi3d_download is not None
        code = paths.hi3d_code_here()
        state = hub.hi3d_download.snapshot()
        return {**state, "available": state["available"] and code, "code": code}

    def flux3_status() -> dict[str, Any]:
        """The same of FLUX.3 Image Klein, and "engine": whether Spellcast3D's image engine can draw it
        (install/sdcpp.py engine_draws: the page offers it only then)."""
        assert hub.flux3_download is not None
        return {**hub.flux3_download.snapshot(), "engine": engine_pin.engine_draws("flux3")}

    async def start_optional_download(request: Request, download: model_downloads.BackgroundDownload | None
                                      ) -> dict[str, Any]:
        """Start the download of an optional image model (`download`, of the size the config file picks) in the
        background, and open its folder in File Explorer, where it arrives (a page on this computer only, never with
        SPELLCAST_NO_EXPLORER=1). Nothing more happens while it runs or once it is there (the folder still opens) ->
        its /api/status object and "opened". 409 when Qwen-Image is not the configured image generator, cannot run
        here or is not installed, when the image engine cannot draw the model, or when the drive lacks the space."""
        assert download is not None
        if paths.QWEN not in hub.adapters:
            raise ToolError(f"Qwen-Image 2.1 is not the image generator {paths.CONFIG_FILE.name} picks, so "
                            f"{download.spec.says()} is not needed.", 409)
        hub.procman.check_available(paths.QWEN)    # 409 when it cannot run here, or is not installed
        if not engine_pin.engine_draws(download.model):
            raise ToolError(FLUX3_NEEDS_ENGINE, 409)
        download.start()
        folder = download.folder
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass                                    # the download says what is wrong with it
        opened = hub.explorer(request, folder, False)
        snapshot = qwen_full_status if download is hub.qwen_full_download else flux3_status
        return {**await asyncio.to_thread(snapshot), "opened": opened}

    @app.post("/api/qwen/full/download")
    async def qwen_full_download_start(request: Request) -> dict[str, Any]:
        """Start the download of the full Qwen-Image 2.1 (into models/qwen-image) and open that folder:
        start_optional_download -> /api/status "qwen_full" and "opened"."""
        return await start_optional_download(request, hub.qwen_full_download)

    @app.post("/api/flux3/download")
    async def flux3_download_start(request: Request) -> dict[str, Any]:
        """Start the download of FLUX.3 Image Klein (into models/flux3-klein) and open that folder:
        start_optional_download -> /api/status "flux3" and "opened"; 409 while the image engine cannot draw it."""
        return await start_optional_download(request, hub.flux3_download)

    @app.post("/api/hi3d/download")
    async def hi3d_download_start() -> dict[str, Any]:
        """Start the one-time download of Hi3D's AI models (11.4 GB, into models/hi3d) in the background, which the
        3D stage asks for when a prop is set to Hi3D; once they are there Hi3D's convolutions are measured. Nothing
        happens while it runs or once they are there -> /api/status "hi3d". 409 when Pixal3D cannot run here or is
        not installed, or when Hi3D's code is not part of this Spellcast3D."""
        hub.procman.check_available(paths.PIXAL3D)     # 409 when Pixal3D cannot run here, or is not installed
        if not paths.hi3d_code_here():
            raise ToolError("Hi3D is not part of this Spellcast3D yet, so its AI models are not needed. Pick Pixal3D "
                            "under Shape.", 409)
        assert hub.hi3d_download is not None
        hub.hi3d_download.start()
        return await asyncio.to_thread(hi3d_status)

    @app.post("/api/pixal3d/multiview/download")
    async def multiview_download_start() -> dict[str, Any]:
        """Start the optional one-time download of Pixal3D's multi-view models (22 GB) in the background.
        Nothing happens while it runs or once they are there -> /api/status "multiview"."""
        hub.procman.check_available(paths.PIXAL3D)     # 409 when Pixal3D cannot run here, or is not installed
        assert hub.multiview_download is not None
        hub.multiview_download.start()
        return await asyncio.to_thread(multiview_status)

    # ---- the background installation ---------------------------------------------------------
    @app.post("/api/install/start")
    async def install_start() -> dict[str, Any]:
        """Run the installer again (after a failed download, say): it installs only what is missing."""
        if hub.install is None:
            raise ToolError("Spellcast3D was started without its installer. Close it and start it with "
                            "run-spellcast3d.bat to install what is missing.")
        hub.install.restart()
        return hub.install.snapshot()

    @app.post("/api/install/calibrate")
    async def install_calibrate() -> dict[str, Any]:
        """Measure the graphics card for Pixal3D again (Settings > Debugging): the record of the install's measurement
        is dropped and the background installer runs, which measures the card as the one thing still due. GPU jobs
        wait while it measures (procman.install_wait). 409 while the installer runs."""
        if hub.install is None:
            raise ToolError("Spellcast3D was started without its installer, which does the measuring. Close it and "
                            "start it with run-spellcast3d.bat.")
        from .install.components import drop_pixal3d_calibration

        hub.install.restart()       # (409 first, before the record goes, while an installer already runs)
        await asyncio.to_thread(drop_pixal3d_calibration)
        return hub.install.snapshot()

    @app.get("/api/install/log")
    async def install_log(tail: int = 200) -> dict[str, Any]:
        return {"lines": await asyncio.to_thread(tail_lines, hub.logs_dir / "install.log", tail)}

    @app.post("/api/tools/{tool}/start")
    async def tool_start(tool: str) -> dict[str, Any]:
        hub.procman.start(tool)
        return hub.procman.status(tool)

    @app.post("/api/tools/{tool}/warm")
    async def tool_warm(tool: str, body: dict[str, Any] | None = Body(None)) -> dict[str, Any]:
        """Have the tool ready, as the artist arrives at its stage, so its start is over by the time they ask for
        its work (procman.warm). The body, when there is one, is the wish: {"model"} names the image generator's
        model the artist has picked (the start loads it first; a start that loads another is dropped for it), and
        {"displace": true} lets the start take the card from a tool that idles on it (Images asks so; else only
        when that displaces nothing). A tool that takes the graphics card at start is left alone while jobs wait or
        run: their own tools come first. Never refuses a tool that cannot start now (404 only for one that does not
        exist) and does not wait for it: the tool's status, with "warmed" (whether a start began)."""
        adapter = hub.procman.get(tool).adapter
        wish = body or {}
        jobs_first = adapter.uses_gpu and adapter.gpu_at_start and bool(hub.jobs.active_projects())
        warmed = not jobs_first and hub.procman.warm(tool, wish, displace=wish.get("displace") is True)
        return {**hub.procman.status(tool), "warmed": warmed}

    @app.post("/api/tools/{tool}/stop")
    async def tool_stop(tool: str) -> dict[str, Any]:
        hub.procman.get(tool)       # 404 for a tool that does not exist
        hub.jobs.cancel_tool_jobs(tool)
        hub.procman.request_stop(tool)
        return hub.procman.status(tool)

    @app.get("/api/tools/{tool}/log")
    async def tool_log(tool: str, tail: int = 200) -> dict[str, Any]:
        return {"lines": hub.procman.log_tail(tool, tail)}

    # ---- projects and props ----------------------------------------------------------------
    @app.get("/api/projects")
    def list_projects() -> list[dict[str, Any]]:
        return hub.store.list_projects()

    @app.post("/api/projects")
    async def create_project(body: dict[str, Any] | None = Body(None)) -> dict[str, Any]:
        """{"name": ...} for a named project; no name: a new untitled one (named later with rename). The new
        project is the one opened last from now on."""
        return await asyncio.to_thread(hub.store.new_project, (body or {}).get("name"), hub.busy_projects())

    async def opened(data: dict[str, Any]) -> dict[str, Any]:
        """A project that was just opened: what a job left in its temp/ folder is removed (a tool that was
        stopped, or a file that was still open then), but never the folder a running job works in."""
        try:
            await asyncio.to_thread(hub.store.clear_work_dirs, data["id"], hub.jobs.work_dir_in_use)
        except ProjectError:
            pass    # renamed at this very moment: its next opening sweeps it
        return data

    @app.post("/api/projects/open-last")
    async def open_last_project() -> dict[str, Any]:
        """The project to start with (opened last, else changed last, else a new untitled one)."""
        return await opened(await asyncio.to_thread(hub.store.open_last, hub.busy_projects()))

    @app.get("/api/projects/{project}")
    async def get_project(project: str) -> dict[str, Any]:
        return await opened(await asyncio.to_thread(hub.store.open_project, project, hub.busy_projects()))

    @app.patch("/api/projects/{project}")
    def patch_project(project: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return hub.store.patch_ui(project, body)

    @app.post("/api/projects/{project}/rename")
    async def rename_project(project: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """{"name"}: name (or rename) the project; its folder gets the new name."""
        if project in hub.busy_projects():
            raise ProjectError(PROJECT_BUSY, 409)
        return await asyncio.to_thread(hub.store.rename_project, project, body.get("name"))

    # ---- saving models into exports/<project>/<prop>/ (exports.py) --------------------------
    async def saving(project: str, save: Callable[[], exports.Saved]) -> exports.Saved:
        """Run a save off the event loop, one at a time; the project cannot be renamed meanwhile."""
        async with hub.export_lock:
            with hub.importing_into(project):
                return await asyncio.to_thread(save)

    @app.post("/api/projects/{project}/props/{prop_id}/export")
    async def export_prop(project: str, prop_id: str, request: Request,
                          body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """{"stage": "mesh" | "projection", "format": "glb" | "fbx" | "obj"}: save that file of the prop into
        its own folder in the project's export folder, then show it in File Explorer (for a page on this
        computer) -> {"folder", "folder_path", "files", "shown", "opened"}; the folder is the prop's."""
        saved = await saving(project, lambda: hub.exporter.save_prop(project, prop_id, body.get("stage"),
                                                                     body.get("format")))
        opened = await asyncio.to_thread(hub.explorer, request, saved.folder / str(saved.shown), True)
        return {**saved.answer(opened), "warnings": saved.warnings}

    # ---- the prompt library (promptlib.py): one list for every project --------------------------------------
    @app.get("/api/prompt-library")
    def get_prompt_library() -> dict[str, Any]:
        return {"prompts": promptlib.read(library_file)}

    @app.put("/api/prompt-library")
    def put_prompt_library(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """{"prompts": [{"id", "name", "text", "shown"}]}: the whole library, saved -> the same, as saved."""
        return {"prompts": promptlib.write(library_file, body.get("prompts"))}

    @app.post("/api/projects/{project}/props/{prop_id}/images/save")
    async def save_picture(project: str, prop_id: str, request: Request,
                           body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """{"path"}: the picture the Images viewport shows (one of the prop's pictures, or viewport/viewport.png with the
        marks painted on it) into the prop's export folder, then shown in File Explorer -> like the export above."""
        saved = await saving(project, lambda: hub.exporter.save_picture(project, prop_id, body.get("path")))
        opened = await asyncio.to_thread(hub.explorer, request, saved.folder / str(saved.shown), True)
        return saved.answer(opened)

    @app.post("/api/projects/{project}/export-all")
    async def export_all(project: str, request: Request) -> dict[str, Any]:
        """Every prop's best result (the bake, else the retopo, else the 3D model) into the project's export
        folder, each prop in its own folder there, then open the project's folder
        -> {"folder", "folder_path", "files", "props", "opened", "warnings"}."""
        saved = await saving(project, lambda: hub.exporter.save_all(project))
        opened = await asyncio.to_thread(hub.explorer, request, saved.folder, False)
        return {**saved.answer(opened), "props": saved.props, "warnings": saved.warnings}

    @app.post("/api/pick-folder")
    async def pick_folder(request: Request) -> dict[str, Any]:
        """Opens the hub's folder picker (a native dialog, on this computer only) -> {"folder": path | null} (null:
        cancelled, or no dialog can be shown here)."""
        if request.client is None or request.client.host not in LOCAL_CLIENTS:
            raise HTTPException(403, "A folder can only be picked on the computer Spellcast3D runs on.")
        folder = await asyncio.to_thread(dialogs.pick_folder)
        return {"folder": folder}

    @app.post("/api/projects/{project}/import-folder")
    async def import_folder(project: str, request: Request, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """{"folder": path}: the pictures of that folder (and its subfolders) become props, one per name, their sides
        as the names say (propfiles.import_prop_folder) -> {"props": [{"id", "name", "sides"}], "skipped": [...]}.
        On this computer only: the hub reads the folder itself."""
        if request.client is None or request.client.host not in LOCAL_CLIENTS:
            raise HTTPException(403, "A folder can only be imported on the computer Spellcast3D runs on.")
        folder = body.get("folder")
        if not isinstance(folder, str) or not folder.strip():
            raise ProjectError("Choose a folder to import.")
        await asyncio.to_thread(hub.store.get_project, project)       # 404 first
        return await asyncio.to_thread(propfiles.import_prop_folder, hub.store, project, Path(folder.strip()))

    @app.post("/api/projects/{project}/draft")
    async def ensure_draft(project: str) -> dict[str, Any]:
        """The Images workbench's draft prop (created when there is none). A draft still waiting for the
        first picture of its generation becomes a prop first, and a new empty draft is made."""
        busy = hub.jobs.active_props(project)
        return await asyncio.to_thread(hub.store.ensure_draft, project, busy)

    @app.post("/api/projects/{project}/props")
    def add_prop(project: str, body: dict[str, Any] | None = Body(None)) -> dict[str, Any]:
        return hub.store.add_prop(project, (body or {}).get("name"))

    @app.post("/api/projects/{project}/props/{prop_id}/pack")
    def pack_prop(project: str, prop_id: str) -> dict[str, Any]:
        """The draft becomes a prop (409 while it has no image). It also does by itself with its first picture."""
        return hub.store.pack_prop(project, prop_id)

    @app.put("/api/projects/{project}/props/{prop_id}/thumbs/{stage}")
    async def put_thumb(project: str, prop_id: str, stage: str, request: Request, crop: str | None = None,
                        of: str | None = None) -> dict[str, Any]:
        """Body: a PNG or WEBP snapshot of the prop in the form `stage` gave it, for the props list.
        ?of=: what a picture's thumbnail shows (projects.thumb_signature: the chosen picture and the part of it); the
        hub keeps it only while the prop shows that, else the prop comes back without it (ProjectStore.set_thumb).
        ?crop=x,y,w,h: the part of the chosen picture it shows (projects.thumb_crop), from pages before ?of=."""
        try:
            part = [int(n) for n in crop.split(",")] if crop else None
        except ValueError:
            part = None
        if part is not None and len(part) != 4:
            part = None
        data = await _read_body(request, MAX_THUMB_BYTES,
                                f"A thumbnail can be {MAX_THUMB_BYTES // (1024 * 1024)} MB at most.")
        return await asyncio.to_thread(hub.store.set_thumb, project, prop_id, stage, data, part, of or None)

    @app.put("/api/projects/{project}/props/{prop_id}/projection/paint")
    async def put_paint(project: str, prop_id: str, request: Request, width: int, height: int,
                        glb: str) -> dict[str, Any]:
        """Body: the bake's base colour map as the artist painted it, raw RGBA8 (width x height x 4 bytes, row 0 =
        the first row of maps/basecolor.png); its alpha is the map's own (painting never changes it). ?glb=: the bake
        it was painted on (the prop's projection.glb). The colour goes into the bake's basecolor.png, GLB and FBX
        (exports.run_repaint_worker; the first painting of a bake keeps the bake's own map beside it,
        projects.UNPAINTED_BASECOLOR) -> the prop, with projection.painted a new stamp and no projection thumbnail.
        404 not baked, 409 baked again meanwhile or a bake of the prop waiting or running, 413 too big, 422 not the
        map's size, or an alpha that is not the map's (not painted on this bake's colours: the page lost them)."""
        await asyncio.to_thread(hub.store.paint_target, project, prop_id, glb, width, height)
        hub.check_paint(project, prop_id)           # both before the body is read
        data = await _read_painting(request, width, height)
        async with hub.export_lock:                 # an export never copies half-written bake files
            with hub.importing_into(project):       # nor is the project renamed meanwhile
                folder = await asyncio.to_thread(hub.store.paint_target, project, prop_id, glb, width, height)
                hub.check_paint(project, prop_id)
                unpainted = await asyncio.to_thread(hub.store.unpainted_copy, project, prop_id)
                prop_dir = hub.store.prop_dir(project, prop_id)
                raw = prop_dir / f"{PAINT_TEMP}{secrets.token_hex(6)}.rgba"
                try:
                    await asyncio.to_thread(_write_paint, prop_dir, raw, data)
                    del data
                    await asyncio.to_thread(hub.repaint, folder, raw, width, height, unpainted)
                finally:
                    await asyncio.to_thread(raw.unlink, missing_ok=True)
                return await asyncio.to_thread(hub.store.set_painted, project, prop_id, glb)

    @app.put("/api/projects/{project}/props/{prop_id}/omnitexture/mask")
    async def put_omnitexture_mask(project: str, prop_id: str, request: Request, glb: str) -> dict[str, Any]:
        """Body: what the artist masked on the prop's OmniTextured model (?glb=: the one on the page), a PNG of
        its texture's size, white where masked -> {"mask": its path}: the "mask" of the repaint job's "retouch"
        (POST /api/jobs, adapters/omnitexture.py). 404 no Omni-Texture, 409 another one meanwhile or a
        Omni-Texture job of the prop waiting or running, 413 too big, 422 no PNG."""
        hub.check_steps_stay(project, prop_id)
        data = await _read_body(request, MAX_MASK_BYTES, "The mask is too big to send.")
        return await asyncio.to_thread(hub.store.omnitexture_mask, project, prop_id, glb, data)

    @app.put("/api/projects/{project}/props/{prop_id}/omnitexture/paint")
    async def put_omnitexture_paint(project: str, prop_id: str, request: Request, width: int, height: int,
                                     glb: str) -> dict[str, Any]:
        """Body: what was painted by hand on a step of the prop's Omni-Texture (?glb=: one of its kept steps, the
        one shown or one the page stepped away from since) since the page last saved it, raw RGBA8 (width x height
        x 4 bytes, row 0 = the texture's first row): the colour of each texel changed, alpha 255 there, alpha 0
        everywhere else. It goes into the step's model (Hub.paint_omnitexture) -> the prop. 404 no Omni-Texture,
        409 `glb` no kept step or an Omni-Texture job of the prop waiting or running or a fade or a picture taken
        off under way, 413 too big, 422 not the texture's size or incomplete."""
        await asyncio.to_thread(hub.store.omnitexture_paint_target, project, prop_id, glb, width, height)
        hub.check_steps_stay(project, prop_id)      # both before the body is read
        return await hub.paint_omnitexture(project, prop_id, glb, width, height,
                                            await _read_painting(request, width, height))

    @app.post("/api/projects/{project}/props/{prop_id}/omnitexture/step")
    async def step_omnitexture(project: str, prop_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """{"glb"}: the prop shows that step of its Omni-Texture - the result or one of the repaints of it
        (omnitexture.steps): the page's Ctrl+Z and Ctrl+Y -> the prop. 404 no such step, 409 while an Omni-Texture
        job of the prop waits or runs."""
        hub.check_steps_stay(project, prop_id)
        return await asyncio.to_thread(hub.store.omnitexture_step, project, prop_id, body.get("glb"))

    @app.post("/api/projects/{project}/props/{prop_id}/omnitexture/fade")
    async def fade_omnitexture(project: str, prop_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """{"glb": the step a repaint made (omnitexture.repaints), "fade": 1-8}: that repaint goes onto the
        model again with its edge faded that many times as far into what is around it, the later repaints onto
        that; the image generator is not needed -> {"prop", "renamed": {step as it was: step as it is now}} (the
        steps from that one on are new files). {"glb", "prepare": true}: the page picked that repaint - the
        worker is made ready, so that the slider's first change is quick -> {"ready": true}. 409 while a
        Omni-Texture job of the prop waits or runs, another fade is being made, or what the repaint was made from
        is no longer kept; 422 another fade."""
        return await hub.fade_omnitexture(project, prop_id, body.get("glb"), body.get("fade"),
                                           prepare=body.get("prepare") is True)

    @app.post("/api/projects/{project}/props/{prop_id}/omnitexture/own")
    async def own_omnitexture(project: str, prop_id: str) -> dict[str, Any]:
        """The artist takes up a paint tool on a prop without an Omni-Texture: its 3D model becomes the first step
        of a result of its own texture, which the tools then paint and mask like any result -> {"prop"} (as it
        is, when it has a result already). 409 while an Omni-Texture job of the prop waits or runs or another
        change of it is being made, or when its 3D model is missing."""
        return await hub.own_omnitexture(project, prop_id)

    @app.post("/api/projects/{project}/props/{prop_id}/omnitexture/remove")
    async def remove_from_omnitexture(project: str, prop_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """{"glb": the model the page shows, "view": a painted view's folder | "repaint": the model a repaint
        made}: that picture is taken off the model - the result composed again from the other views, or the
        repaints after it put on again; the image generator is not needed -> {"prop"}, which shows a new step
        without it (Ctrl+Z steps back to the one with it). {"glb", "whole": true}: the result as a whole is taken
        off - the prop shows its 3D model, and Ctrl+Z brings the result back. 409 while an Omni-Texture job of the
        prop waits or runs or another change of it is being made, when the page shows another model, the picture is
        not on it, it is the only view, or what it takes is no longer kept."""
        return await hub.take_off_omnitexture(project, prop_id, body.get("glb"), body.get("view"),
                                               body.get("repaint"), whole=body.get("whole") is True)

    @app.patch("/api/projects/{project}/props/{prop_id}")
    def patch_prop(project: str, prop_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return hub.store.patch_prop(project, prop_id, body)

    @app.delete("/api/projects/{project}/props/{prop_id}")
    def delete_prop(project: str, prop_id: str) -> dict[str, Any]:
        """-> {"ok", "deleted": the prop's folder in the project's deleted/, for the restore below}"""
        return {"ok": True, "deleted": hub.store.delete_prop(project, prop_id)}

    @app.post("/api/projects/{project}/deleted/{name}/restore")
    def restore_prop(project: str, name: str) -> dict[str, Any]:
        """The prop a DELETE above removed (`name`: what it answered) back at its place in the list -> the prop."""
        return hub.store.restore_prop(project, name)

    @app.post("/api/projects/{project}/props/{prop_id}/images/upload")
    async def upload_image(project: str, prop_id: str, request: Request) -> dict[str, Any]:
        """multipart "file" (one picture) or "files" (repeated): pictures of the prop's own, and optionally
        "sides": a JSON list of the side each file shows ("front" | "left" | "back" | "right" | null). Two or
        more become its views: those sides, else front, left side, back, right side in upload order; the
        front one is chosen, else the first (projects.add_images). The draft becomes a prop (named after the
        first file unless it has a name or a prompt)."""
        await asyncio.to_thread(hub.store.get_prop, project, prop_id)       # 404 before the upload is read
        too_many = f"These are too many pictures at once. Import at most {MAX_IMAGE_UPLOADS}."
        _check_upload_size(request, MAX_IMAGE_UPLOADS * MAX_UPLOAD_BYTES + UPLOAD_OVERHEAD,
                           f"These pictures are too big together. Import at most {MAX_IMAGE_UPLOADS} at once, each "
                           f"up to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        try:
            async with request.form(max_files=MAX_IMAGE_UPLOADS * 4, max_fields=16) as form:
                items = [item for key in ("file", "files", "files[]") for item in form.getlist(key)
                         if isinstance(item, FormFile)]
                if len(items) > MAX_IMAGE_UPLOADS:          # said before any of them is read into memory
                    raise ProjectError(too_many, 413)
                raw_sides = form.get("sides")
                files = [(await item.read(MAX_UPLOAD_BYTES + 1),
                          PurePosixPath((item.filename or "").replace("\\", "/")).name) for item in items]
        except StarletteHTTPException as e:
            if e.status_code == 400 and "Too many files" in str(e.detail):     # Starlette's own words
                raise ProjectError(too_many, 413) from None
            raise
        sides = None
        if isinstance(raw_sides, str) and raw_sides.strip():
            try:
                sides = json.loads(raw_sides)
            except ValueError:
                raise ProjectError("The sides of these pictures are not right. Reload the page and import them "
                                   "again.") from None
        return await asyncio.to_thread(hub.store.add_images, project, prop_id, files, sides)

    @app.put("/api/projects/{project}/props/{prop_id}/images/views")
    def put_views(project: str, prop_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """{"views": [{"view": "front" | "left" | "back" | "right", "src": one of the prop's pictures,
        "rect": [x, y, w, h] (pixels of that picture) | null (all of it)}]}: the prop's views, at most one per
        side (projects.check_views); [] clears them -> the prop."""
        return hub.store.set_views(project, prop_id, body.get("views"))

    @app.post("/api/projects/{project}/props/{prop_id}/images/select")
    def select_image(project: str, prop_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return hub.store.select_image(project, prop_id, body.get("path"))

    @app.post("/api/projects/{project}/props/{prop_id}/images/remove")
    def remove_image(project: str, prop_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """{"path": one of the prop's pictures}: it leaves the prop's pictures and views (its file stays, for
        images/restore); when it was the chosen one, its neighbour is chosen (projects.remove_image) -> the prop."""
        return hub.store.remove_image(project, prop_id, body.get("path"))

    @app.post("/api/projects/{project}/props/{prop_id}/images/restore")
    def restore_image(project: str, prop_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """{"path", "index"?, "views"?, "chosen"?}: the page's Ctrl+Z of images/remove. The picture is one of the
        prop's again, at its old place, with the views it had, chosen again when "chosen" is true
        (projects.restore_image) -> the prop."""
        return hub.store.restore_image(project, prop_id, body.get("path"), body.get("index"), body.get("views"),
                                       body.get("chosen") is True)

    async def add_references(project: str, prop_id: str, request: Request) -> dict[str, Any]:
        """The example pictures of a multipart upload ("files" repeated, or "file"), after the prop's own -> the
        prop. More than MAX_REFERENCES in all: 400, said before any file is read into memory."""
        prop = await asyncio.to_thread(hub.store.get_prop, project, prop_id)       # 404 before the upload is read
        have = len(prop["images"]["references"])
        _check_upload_size(request, MAX_REFERENCES * MAX_UPLOAD_BYTES + UPLOAD_OVERHEAD,
                           f"These pictures are too big together. Each example picture can be "
                           f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB at most.")
        try:
            async with request.form(max_files=MAX_REFERENCES + 1, max_fields=16) as form:
                items = [item for key in ("file", "files", "files[]") for item in form.getlist(key)
                         if isinstance(item, FormFile)]
                if have + len(items) > MAX_REFERENCES:
                    raise ProjectError(too_many_references(have))
                files = [(await item.read(MAX_UPLOAD_BYTES + 1),
                          PurePosixPath((item.filename or "").replace("\\", "/")).name) for item in items]
        except StarletteHTTPException as e:
            if e.status_code == 400 and "Too many files" in str(e.detail):     # Starlette's own words
                raise ProjectError(too_many_references(have)) from None
            raise
        return await asyncio.to_thread(hub.store.add_references, project, prop_id, files)

    @app.post("/api/projects/{project}/props/{prop_id}/images/references")
    async def post_references(project: str, prop_id: str, request: Request) -> dict[str, Any]:
        """multipart "files" (repeated): example pictures for "Describe + example", added to the prop's own
        (images.references, at most MAX_REFERENCES in all: more is a 400 and nothing is added) -> the prop."""
        return await add_references(project, prop_id, request)

    @app.delete("/api/projects/{project}/props/{prop_id}/images/references")
    def delete_references(project: str, prop_id: str, path: str | None = None) -> dict[str, Any]:
        """?path=<one of images.references>: remove that example picture; without path: all of them -> the prop."""
        return hub.store.remove_references(project, prop_id, path)

    @app.post("/api/projects/{project}/props/{prop_id}/images/viewport")
    async def post_viewport(project: str, prop_id: str, request: Request) -> dict[str, Any]:
        """multipart "file": the viewport's picture with the artist's marks painted over it, or the empty viewport
        painted on (the page's "Re-Render Viewport"), in place of the one before -> {"path"}, the job's from_viewport
        (adapters/images_common.py). With a "mask" part (Smart Mask: where the artist brushed) that is stored too."""
        await asyncio.to_thread(hub.store.get_prop, project, prop_id)       # 404 before the upload is read
        _check_upload_size(request, MAX_VIEWPORT_BYTES + UPLOAD_OVERHEAD,
                           f"This picture is too big. It can be {MAX_VIEWPORT_BYTES // (1024 * 1024)} MB at most.")
        async with request.form(max_files=2, max_fields=4) as form:
            item, where = form.get("file"), form.get("mask")
            data = await item.read(MAX_VIEWPORT_BYTES + 1) if isinstance(item, FormFile) else b""
            mask = await where.read(MAX_VIEWPORT_BYTES + 1) if isinstance(where, FormFile) else None
        return await asyncio.to_thread(hub.store.set_viewport, project, prop_id, data, mask)

    # The routes of the one reference picture before there were several: add one, remove all.
    @app.post("/api/projects/{project}/props/{prop_id}/reference")
    async def post_reference(project: str, prop_id: str, request: Request) -> dict[str, Any]:
        return await add_references(project, prop_id, request)

    @app.delete("/api/projects/{project}/props/{prop_id}/reference")
    def clear_references(project: str, prop_id: str) -> dict[str, Any]:
        return hub.store.remove_references(project, prop_id)

    # ---- the artist's own 3D model (3D stage) ----------------------------------------------
    async def convert_model(src: Path, dst: Path, overrides: dict[str, Any]) -> dict[str, Any]:
        """Instant Meshes converts `src` into the canonical GLB `dst` (started first if needed)."""
        adapter = hub.adapter(paths.IMESH)
        tp = await hub.procman.ensure_running(paths.IMESH)
        hub.procman.set_busy(paths.IMESH, True)
        try:
            return await adapter.import_mesh(hub.client, tp.base_url, src, dst, overrides)  # type: ignore[attr-defined]
        except StageError as e:
            raise ToolError(str(e), 422) from e
        finally:
            hub.procman.set_busy(paths.IMESH, False)

    def imported_patch(prop_dir: Path, dst: Path, result: dict[str, Any], imported: dict[str, Any]) -> dict[str, Any]:
        return {"glb": dst.relative_to(prop_dir).as_posix(),
                "imported": {"materials": result.get("materials") or [], "warnings": result.get("warnings") or [],
                             **imported},
                "stats": {key: result.get(key) for key in ("seconds", "vertices", "faces")}}

    async def import_uploads(project: str, prop_id: str,
                             uploads: list[tuple[str, IO[bytes]]]) -> tuple[dict[str, Any], str]:
        """Save the uploaded files in the prop's mesh/import_NNN/ and convert the model among them into
        mesh/imported_NNN.glb -> (the prop, the model's path in the import folder). Hold hub.import_lock."""
        folder, dst = await asyncio.to_thread(hub.store.new_model_import, project, prop_id)
        try:
            saved = await asyncio.to_thread(save_model_files, folder, uploads)
            if saved.model is None:
                raise ProjectError("There is no 3D model among these files. Add the .fbx, .obj, .glb or "
                                   ".gltf file (its textures can come along), or a .zip that contains one.")
            result = await convert_model(folder / saved.model, dst, {})
        except BaseException:
            await asyncio.to_thread(shutil.rmtree, folder, True)
            raise
        prop_dir = hub.store.prop_dir(project, prop_id)
        patch = {**imported_patch(prop_dir, dst, result, {"skipped": saved.skipped, "overrides": {}}),
                 "source": (folder / saved.model).relative_to(prop_dir).as_posix()}
        prop = await asyncio.to_thread(hub.store.apply_stage_patch, project, prop_id, "mesh", patch)
        await asyncio.to_thread(hub.store.remove_other_imports, project, prop_id, (patch["glb"], patch["source"]))
        return prop, saved.model

    @app.post("/api/projects/{project}/props/{prop_id}/mesh/import")
    async def import_model(project: str, prop_id: str, request: Request) -> dict[str, Any]:
        """multipart "files" (repeated): the model and its textures (names may carry sub-folders), or .zip files.
        The mesh becomes mesh/imported_NNN.glb; the uploaded files stay in mesh/import_NNN/."""
        await asyncio.to_thread(hub.store.get_prop, project, prop_id)       # 404 before the upload is read
        _check_model_upload_size(request)
        async with request.form(max_files=MAX_IMPORT_FILES, max_fields=16) as form:
            uploads = _model_uploads(form)
            async with hub.import_lock:
                with hub.importing_into(project):
                    prop, _model = await import_uploads(project, prop_id, uploads)
        return prop

    @app.post("/api/projects/{project}/props/import-model")
    async def import_model_as_prop(project: str, request: Request) -> dict[str, Any]:
        """Like .../mesh/import, into a new prop (named after the model) that starts as a 3D model. Nothing
        is kept when the import fails."""
        await asyncio.to_thread(hub.store.project_dir, project)             # 404 before the upload is read
        _check_model_upload_size(request)
        async with request.form(max_files=MAX_IMPORT_FILES, max_fields=16) as form:
            uploads = _model_uploads(form)
            # Named after the first file until the model among them is known.
            first_name = box_name_from_file(PurePosixPath(uploads[0][0].replace("\\", "/")).name)
            async with hub.import_lock:
                with hub.importing_into(project):
                    prop = await asyncio.to_thread(hub.store.add_prop, project, first_name or "Imported model")
                    try:
                        _prop, model = await import_uploads(project, prop["id"], uploads)
                    except BaseException:
                        await asyncio.to_thread(hub.store.discard_prop, project, prop["id"])
                        raise
                    return await asyncio.to_thread(hub.store.patch_prop, project, prop["id"],
                                                   {"name": box_name_from_file(PurePosixPath(model).name)
                                                    or "Imported model"})

    @app.post("/api/projects/{project}/props/{prop_id}/mesh/reimport")
    async def reimport_model(project: str, prop_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """{"overrides": {texture key: {"role"?, "flip_green"?, "invert"?}}}: convert the imported model
        again with the artist's corrections (the full set: it replaces the previous one)."""
        overrides = _import_overrides(body.get("overrides"))
        async with hub.import_lock:
            with hub.importing_into(project):
                return await reconvert_model(project, prop_id, overrides)

    async def reconvert_model(project: str, prop_id: str, overrides: dict[str, Any]) -> dict[str, Any]:
        prop = await asyncio.to_thread(hub.store.get_prop, project, prop_id)
        mesh = prop["mesh"]
        if not mesh.get("imported") or not mesh.get("source"):
            raise ProjectError("This prop's 3D model is not your own imported model, so there is nothing to "
                               "convert again.", 409)
        try:
            src = hub.store.resolve_file(project, f"props/{prop_id}/{mesh['source']}")
        except ProjectError as e:
            raise ProjectError(f"The files of your imported model ({mesh['source']}) are missing. Import it "
                               f"again.", 409) from e
        dst = await asyncio.to_thread(hub.store.next_imported_glb, project, prop_id)
        try:
            result = await convert_model(src, dst, overrides)
        except BaseException:
            dst.unlink(missing_ok=True)
            raise
        prop_dir = hub.store.prop_dir(project, prop_id)
        patch = imported_patch(prop_dir, dst, result, {"skipped": mesh["imported"].get("skipped") or [],
                                                       "overrides": overrides})
        prop = await asyncio.to_thread(hub.store.apply_stage_patch, project, prop_id, "mesh", patch)
        await asyncio.to_thread(hub.store.remove_other_imports, project, prop_id, (patch["glb"], mesh["source"]))
        return prop

    # ---- jobs (async: the queue lives on the event loop) ------------------------------------
    @app.post("/api/jobs")
    async def submit_job(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """{"stage", "project", "prop_ids", "params"?, "tool"?}: queue a job (jobs.JobQueue.submit). In Retopo each
        prop picks its tool, so the page sends one job per method, with its "tool". Images runs on the image
        generator the config file picks: 409 with the file's problems when it has a mistake (no generator then).
        Omni-Texture runs on Qwen-Image 2.1 alone: 409 while the file picks another one."""
        if body.get("stage") == "images" and not hub.stage_tools.get("images") \
                and hub.config.get("image_backend") is None:
            raise ToolError(" ".join([NO_IMAGE_GENERATOR, *hub.config.get("problems", []), FIX_CONFIG]), 409)
        if body.get("stage") == "omnitexture":
            if not hub.stage_tools.get("omnitexture"):
                raise ToolError(NO_OMNITEXTURE, 409)
            hub.check_not_fading(body.get("project"), body.get("prop_ids") or [])
            hub.check_not_painting(body.get("project"), body.get("prop_ids") or [])
        return await hub.jobs.submit(body.get("stage"), body.get("project"), body.get("prop_ids"),
                                     body.get("params"), body.get("tool"))

    @app.get("/api/jobs")
    async def list_jobs(project: str | None = None) -> list[dict[str, Any]]:
        return hub.jobs.list(project)

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        return hub.jobs.get(job_id)

    @app.get("/api/jobs/{job_id}/preview")
    async def job_preview(job_id: str) -> Response:
        """The picture the job's tool is making, as it looks so far: small, from memory (jobs.Job.preview). The job's
        progress.preview.url names it, with a number that changes with every new picture. 404 when there is none
        (most of the time: only Qwen-Image shows one, while it draws)."""
        picture = hub.jobs.preview(job_id)
        if picture is None:
            raise HTTPException(404, "This job has no preview picture right now.")
        media = "image/png" if picture.startswith(b"\x89PNG") else "image/jpeg"
        return Response(picture, media_type=media, headers={"Cache-Control": "no-store"})

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        return hub.jobs.cancel(job_id)

    # ---- Forge models ----------------------------------------------------------------------
    async def forge_models(refresh: bool) -> dict[str, Any]:
        if paths.FORGE not in hub.adapters:
            raise ToolError(f"Forge Neo is not the image generator {paths.CONFIG_FILE.name} picks, so it has no model "
                            f"list here.", 409)
        adapter = hub.adapter(paths.FORGE)
        forge = hub.procman.status(paths.FORGE)
        if forge["state"] in (INSTALLING, NOT_INSTALLED, UNSUPPORTED):
            raise ToolError(forge["message"], 409)
        tp = hub.running(paths.FORGE, f"{adapter.display_name} is not running yet, so its model list is not "
                                      f"available. It starts by itself when you generate, or from Settings > "
                                      f"Debugging.")
        call = adapter.refresh_models if refresh else adapter.list_models  # type: ignore[attr-defined]
        try:
            return await call(hub.client, tp.base_url)
        except StageError as e:
            raise ToolError(str(e), 502) from e
        except httpx.HTTPError as e:
            raise ToolError(f"Could not get the model list from {adapter.display_name} ({type(e).__name__}). "
                            f"Try again in a moment.", 502) from e

    @app.get("/api/forge/models")
    async def list_models() -> dict[str, Any]:
        return await forge_models(refresh=False)

    @app.post("/api/forge/refresh")
    async def refresh_models() -> dict[str, Any]:
        return await forge_models(refresh=True)

    # ---- Instant Meshes viewer sessions ----------------------------------------------------
    @app.post("/api/imesh/session")
    async def imesh_session(request: Request, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """{"project", "prop_id"}: the prop's viewer session -> {"session_id", "viewer_url", "reused",
        "replaced", "lost_strokes"}. viewer_url opens the viewer as the Retopo stage embeds it (VIEWER_OPTIONS,
        and host_origin: the origin of the page that asked)."""
        project, prop_id = body.get("project"), body.get("prop_id")
        adapter = hub.adapter(paths.IMESH)
        prop = await asyncio.to_thread(hub.store.get_prop, project, prop_id)
        glb_rel = prop["mesh"].get("glb")
        try:
            glb = hub.store.resolve_file(project, f"props/{prop_id}/{glb_rel}") if glb_rel else None
        except ProjectError:
            glb = None
        if glb is None:
            raise ProjectError("This prop has no 3D model yet. Make one in the 3D stage first.", 409)
        tp = await hub.procman.ensure_running(paths.IMESH)
        async with hub.imesh_lock:      # one session per prop, even on a double click
            prop = await asyncio.to_thread(hub.store.get_prop, project, prop_id)
            previous = prop["retopo"].get("session_id")
            try:
                # Instant Meshes keeps `previous` (brush strokes intact) only while it is alive and
                # was loaded from this very GLB version; otherwise it closes it and opens a new one.
                result = await adapter.open_session(  # type: ignore[attr-defined]
                    hub.client, tp.base_url, glb, settings=prop["retopo"].get("settings"), reuse=previous)
            except StageError as e:
                raise ToolError(str(e), 502) from e
            except httpx.HTTPError as e:
                raise ToolError(f"Could not open the mesh in {adapter.display_name} ({type(e).__name__}). "
                                f"Try again; if it keeps failing, stop and restart it.", 502) from e
            session_id, viewer_path = result["session_id"], result["viewer_path"]
            if session_id != previous or prop["retopo"].get("session_source") != glb_rel:
                await asyncio.to_thread(hub.store.apply_stage_patch, project, prop_id, "retopo",
                                        {"session_id": session_id, "session_source": glb_rel})
        # lost_strokes: Instant Meshes had closed this prop's earlier viewer to make room for other
        # props while it held guide strokes that were never accepted (the page says so).
        origin = _page_origin(request, hub.hub_port, allow_remote)
        return {"session_id": session_id, "viewer_url": _viewer_url(f"http://127.0.0.1:{tp.port}", viewer_path, origin),
                "reused": bool(result.get("reused")), "replaced": result.get("replaced"),
                "lost_strokes": bool(result.get("lost_strokes"))}

    @app.post("/api/imesh/session/{session_id}/save")
    async def imesh_save(session_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """{"project", "prop_id"}: the viewer's Accept. Its current mesh becomes the prop's clean mesh of the size its
        vertex count stands for, in place of one made of that size, and it is picked; the viewer's settings, as the
        session reports them, become the prop's retopo settings -> the prop. Other body keys are ignored."""
        project, prop_id = body.get("project"), body.get("prop_id")
        adapter = hub.adapter(paths.IMESH)
        prop = await asyncio.to_thread(hub.store.get_prop, project, prop_id)
        if prop["retopo"].get("session_id") != session_id:
            raise ProjectError("This viewer does not belong to this prop any more. "
                               "Select the prop in the Retopo stage again, then accept.", 409)
        tp = hub.running(paths.IMESH, f"{adapter.display_name} is not running, so the viewer's result cannot be "
                                      f"saved. Select the prop in the Retopo stage again to start it.")
        dst = hub.store.prop_dir(project, prop_id) / "retopo" / "retopo.obj"
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = await adapter.save_session(hub.client, tp.base_url, session_id, dst)  # type: ignore[attr-defined]
        except StageError as e:
            raise ToolError(str(e), 502) from e
        except httpx.HTTPError as e:
            raise ToolError(f"Could not save from {adapter.display_name} ({type(e).__name__}). "
                            f"Try again in a moment.", 502) from e
        source = prop["retopo"].get("session_source") or prop["mesh"].get("glb")
        # Same stats keys as a job's clean mesh, so the page shows e.g. a "saved without UVs" warning.
        stats = {k: result[k] for k in ("vertices", "faces", "has_uv", "seconds", "warning") if k in result}
        size = retopo_size_of(stats.get("vertices"))
        patch: dict[str, Any] = {"variants": [{"size": size, "obj": "retopo/retopo.obj", "stats": stats}],
                                 "picked": size, "method": paths.IMESH, "source": source, "saved_from": "viewer"}
        if result.get("settings"):
            patch["settings"] = {**prop["retopo"]["settings"], **result["settings"]}
        return await asyncio.to_thread(hub.store.apply_stage_patch, project, prop_id, "retopo", patch)

    @app.post("/api/projects/{project}/props/{prop_id}/retopo/pick")
    async def pick_retopo(project: str, prop_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """{"size"}: the artist picks the prop's clean mesh of that size (one of retopo.variants), which it uses from
        now on -> the prop. 409 while it has none of that size made from its 3D model."""
        return await asyncio.to_thread(hub.store.pick_retopo, project, prop_id, body.get("size"))

    # ---- settings and folders --------------------------------------------------------------
    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return hub.settings.get()

    @app.patch("/api/settings")
    def patch_settings(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return hub.settings.patch(body)

    @app.post("/api/open-folder")
    async def open_folder(request: Request, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """{"which": "project" (with "project") | "exports" (with "project": its saved models) | "models" |
        "forge_models" | "logs" | "config" (the Spellcast3D folder with spellcast3d-config.ini selected, made first
        when it is missing)} -> {"ok": true, "opened": bool}. opened is false with SPELLCAST_NO_EXPLORER=1."""
        if request.client is None or request.client.host not in LOCAL_CLIENTS:
            raise HTTPException(403, "Folders can only be opened on the computer Spellcast3D runs on.")
        which = body.get("which")
        if which == "config":
            target = config.config_file()
            made = await asyncio.to_thread(config.ensure_config_file)
            if made:
                log.info("%s", made)
            if not target.is_file():
                raise HTTPException(500, f"Could not find or make {target}. Open the Spellcast3D folder yourself in "
                                         f"File Explorer.")
            return {"ok": True, "opened": hub.explorer(request, target, True)}
        if which == "project":
            folder = hub.store.project_dir(body.get("project"))
        elif which == "exports":
            folder = hub.store.export_dir(await asyncio.to_thread(hub.store.get_project, body.get("project")))
        elif which == "logs":
            folder = hub.logs_dir
        elif which == "models":
            folder = paths.MODELS                   # the artist's models folder (its README.txt says what goes where)
        elif which == "forge_models":
            folder = paths.MODELS_CHECKPOINTS       # where the artist drops image models (Forge reads it)
        else:
            raise HTTPException(400, 'Choose which folder to open: "project", "exports", "models", "forge_models", '
                                     '"logs" or "config".')
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if exports.explorer_allowed():
                await asyncio.to_thread(os.startfile, folder)  # type: ignore[attr-defined]
        except OSError as e:
            raise HTTPException(500, f"Could not open the folder {folder} ({e}). Open it yourself in "
                                     f"File Explorer.") from e
        return {"ok": True, "opened": exports.explorer_allowed()}

    return app
