"""The optional models (two of the Images stage's, and Hi3D's): whether they are on disk, and their download when the
artist asks for one.

Three of them (install/models.py OPTIONAL_MODELS; MODELS here says how the hub names and places each):
- "full", the full Qwen-Image 2.1 (4.2 GB at the default size Q4_K_M).
  Qwen-Image-2.1-Turbo is the default model on every card and the installer downloads it, not the full model:
  nobody sits through a download of a model they may never use. The full model - 25 steps, the "Faster drawing"
  tick, a little finer detail - is one press away: its button under Model (POST /api/qwen/full/download), into
  models\\qwen-image, its progress in /api/status "qwen_full". A full model an older install downloaded is simply
  there (available()), and is used.
- "flux3", FLUX.3 Image Klein (Black Forest Labs; 6.8 GB at Q4_K_M: its denoiser of the configured size, FLUX.2's
  VAE, the Qwen3-VL-4B text encoder), the third choice under Model (POST /api/flux3/download), into
  models\\flux3-klein, its progress in /api/status "flux3". Only with an engine that can draw it (the ONE gate,
  install/sdcpp.py ENGINE_HAS_FLUX3 / engine_draws): without, the page shows it disabled and the hub refuses its
  download. A copy already on this PC is linked instead of downloaded (the download script: the installer's sources,
  and by name a file already in models\\qwen-image, where an artist may have put FLUX.3's GGUF first).
- "hi3d", Hi3D's AI models (11.4 GB in four folders: install/models.py HI3D_FILES), the 3D stage's second Shape, into
  models\\hi3d, its progress in /api/status "hi3d" (POST /api/hi3d/download). They
  are no part of Pixal3D's install: the 3D stage asks before a prop is set to Hi3D while they are not on this PC
  (mesh.js agreeToHi3d: the download, slower, the front picture alone, the project's Shape from then on), then starts
  it. Once the files are there, the download runs Pixal3D's calibration for Hi3D's convolutions alone
  (hi3d_calibration: `pixal3d.calibrate --only spconv`, a few minutes with the card quiet; "finishing" in the status
  meanwhile), the step the installer leaves out while Hi3D's models are not here.
A denoiser of a bigger size than the configured one, already in the model's folder, counts as the model (install/models.py
size_on_disk): nothing of it is downloaded then but what is missing besides (FLUX.3's VAE, say).

Each button opens the model's folder in File Explorer and starts its download; the page shows its progress from
/api/status ({"available", "downloading", "waiting", "done_bytes", "total_bytes", "error"}, snapshot()) until the
model can be picked. total_bytes counts the files still missing when the download started (when nothing runs: those
missing now), done_bytes what is on disk of them (whole files and their .part); once all are there both are the model's
whole size. waiting: it has not started yet, an installer being at work on its folder (below).

Like Pixal3D's multi-view models (multiview.py) a download is neither an installer component nor a repair of
Qwen-Image: those would mark Qwen-Image as being installed, and every picture with the Turbo would wait for it. The hub
runs launchers/qwen_download.py --model <name> - the installer's own downloader, install/models.py download_file: the
pinned HuggingFace commit, continued from <name>.part after an interruption, the SHA-256 checked before the file gets
its name - with the hub's own Python (stdlib only), as a child in its Job Object, so it stops with the hub; pressed
again (or after the next start), it continues where it stopped. Each model has its own BackgroundDownload and so its
own process, writing only into its own folder (never two writers on one file): the full model's waits while an
installer works on Qwen-Image (the hub's own background installer, or another one holding code/venvs/.install.lock:
install.py in a window, tools\\update.bat), as both write into models\\qwen-image; FLUX.3's folder is no installer's,
so it never waits. Progress is the bytes of the files, or of their .part, on disk; the output goes to code/logs/
(MODELS' log names).

The size is the one spellcast3d-config.ini picked when the hub started (the size the launcher runs with): a new size
applies at the next start, as everything in that file.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import paths
from .install import models
from .installer_bg import INSTALL_LOCK, BackgroundInstall, lock_held
from .multiview import pump_output
from .procman import ToolError, hub_has_console, kill_tree, spawn_in_job

log = logging.getLogger(__name__)

SCRIPT = paths.LAUNCHERS / "qwen_download.py"
DISK_MARGIN = 1_000_000_000             # free space left over after the download, for the drive's own sake
WAIT_POLL = 5.0                         # seconds between looks at an installer that works on Qwen-Image
EXIT_POLL = 0.5
# qwen_download.py's exit codes (its EXIT_*; a test compares them): 2 is argparse's own, a bad argv
EXIT_OK, EXIT_FAILED, EXIT_CHECKSUM, EXIT_DISK_FULL, EXIT_DENIED = 0, 1, 3, 4, 5


@dataclass(frozen=True)
class OptionalModel:
    """How the hub names and places an optional model (install/models.py OPTIONAL_MODELS)."""
    key: str                # "full" | "flux3": the script's --model, the prop's qwen_model
    status: str             # its key in /api/status: "qwen_full" | "flux3"
    name: str               # in a sentence: "the full Qwen-Image 2.1"
    folder: Path            # where it lives (paths.MODELS_QWEN | paths.MODELS_FLUX3)
    log_name: str           # its download's output, in code/logs
    waits_for: str | None   # the installer component whose work in that folder the download waits for

    def says(self, start: bool = False) -> str:
        """`name`, with a capital at the start of a sentence."""
        return self.name[:1].upper() + self.name[1:] if start else self.name


MODELS: dict[str, OptionalModel] = {
    "full": OptionalModel("full", "qwen_full", "the full Qwen-Image 2.1", paths.MODELS_QWEN, "qwen-full-download.log",
                          paths.QWEN),
    "flux3": OptionalModel("flux3", "flux3", "FLUX.3 Image Klein", paths.MODELS_FLUX3, "flux3-download.log", None),
    "hi3d": OptionalModel("hi3d", "hi3d", "Hi3D's AI models", paths.MODELS_HI3D, "hi3d-download.log", None),
}
assert tuple(MODELS) == models.OPTIONAL_MODELS


# ---- on disk ---------------------------------------------------------------------------------------------

def manifest(model: str, quality: str, folder: Path | None = None) -> models.Manifest:
    """The files of `model` for size `quality` in its folder (`folder`: another one): what its download fetches (a
    bigger denoiser already there stands in for the configured size)."""
    return models.optional_manifest(model, quality, folder or MODELS[model].folder)


def file_names(model: str, quality: str, folder: Path | None = None) -> list[str]:
    return list(manifest(model, quality, folder))


def total_bytes(model: str, quality: str, folder: Path | None = None) -> int:
    """Its whole size: every file."""
    return sum(size for size, _sha in manifest(model, quality, folder).values())


def available(model: str, quality: str, folder: Path | None = None) -> bool:
    """Every file of it in the folder at its size (finished downloads: an unfinished one is .part)."""
    return not models.missing_files(folder or MODELS[model].folder, manifest(model, quality, folder))


def missing(model: str, quality: str, folder: Path | None = None) -> list[str]:
    """Its files not in the folder at their size yet."""
    return models.missing_files(folder or MODELS[model].folder, manifest(model, quality, folder))


def downloaded_bytes(model: str, quality: str, folder: Path | None = None, names: list[str] | None = None) -> int:
    """Bytes of it on disk (of `names` only, when given): the whole files, or what their downloads have written into
    <name>.part so far."""
    files = manifest(model, quality, folder)
    if names is not None:
        files = {name: files[name] for name in names}
    return models.present_bytes(folder or MODELS[model].folder, files)


def _free_space(folder: Path) -> int:
    return shutil.disk_usage(folder).free


def _existing(path: Path) -> Path:
    """`path`, or its nearest folder that exists (for the free space of its drive)."""
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


# ---- the download ------------------------------------------------------------------------------------------

class BackgroundDownload:
    """Runs one optional model's download as a child of the hub (start / close) and says how far it got (snapshot)."""

    def __init__(self, model: str, quality: str, folder: Path | None = None, *, python: Path | None = None,
                 script: Path = SCRIPT, install: BackgroundInstall | None = None, install_lock: Path = INSTALL_LOCK,
                 log_file: Path | None = None, wait_poll: float = WAIT_POLL,
                 then: Callable[[Path], Awaitable[None]] | None = None):
        """model: "full" | "flux3" | "hi3d" (MODELS). quality: the model size the hub runs with. folder: where it goes
        (its MODELS folder unless given). python: the hub's own (the script needs only the stdlib). install: the hub's
        background installer, whose work on Qwen-Image the full model's download waits for (as it does for another
        installer holding install_lock). then: a step once the files are there (hi3d_calibration), given the log file;
        "finishing" in the snapshot while it runs, its failure its own affair."""
        self.spec = MODELS[model]
        self.model = model
        self.quality = quality
        self.folder = folder or self.spec.folder
        self.python = python or Path(sys.executable)
        self.script = script
        self.install = install
        self.install_lock = install_lock
        self.log_file = log_file or paths.LOGS / self.spec.log_name
        self.wait_poll = wait_poll
        self.proc: subprocess.Popen[bytes] | None = None
        self.error: str | None = None
        self.waiting = False               # started, but not yet running: an installer works on its folder
        self.then = then
        self.finishing = False             # the files are there; `then` runs
        self._task: asyncio.Task[None] | None = None
        self._wanted: list[str] = []       # the files missing when the running download started (its total)

    # ---- lifecycle (on the event loop) --------------------------------------------------------
    def start(self) -> None:
        """Begin the download in the background. Nothing happens while it runs, or once its files are there.
        ToolError (409) when the drive has too little free space for what is left of it."""
        if self.downloading() or self.available():
            return
        wanted = missing(self.model, self.quality, self.folder)
        files = manifest(self.model, self.quality)
        left = sum(files[name][0] for name in wanted) - downloaded_bytes(self.model, self.quality, self.folder, wanted)
        drive = _existing(self.folder)
        free = _free_space(drive)
        if free < left + DISK_MARGIN:
            raise ToolError(f"There is not enough free space on drive {drive.drive or drive} for "
                            f"{self.spec.says()}: it needs {(left + DISK_MARGIN) / 1e9:.1f} GB, and {free / 1e9:.1f} "
                            f"GB are free. Free some space, then try again.")
        self.error, self.proc, self._wanted = None, None, wanted
        self._task = asyncio.create_task(self._run(), name=f"{self.model}-download")

    async def close(self) -> None:
        """Stop the download with the hub (the Job Object would kill it anyway); starting it again continues it."""
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        proc = self.proc
        if proc is not None and proc.poll() is None:
            log.info("Stopping the download of %s; it continues when it is started again.", self.spec.says())
            await asyncio.to_thread(kill_tree, proc.pid)
            await asyncio.to_thread(proc.wait, 10)

    async def _run(self) -> None:
        self.waiting = self._installer_busy()
        while self._installer_busy():
            await asyncio.sleep(self.wait_poll)
        self.waiting = False
        proc = self._spawn()
        if proc is None:
            return
        while proc.poll() is None:
            await asyncio.sleep(EXIT_POLL)
        self.error = self._exit_error(proc.returncode)
        log.log(logging.INFO if self.error is None else logging.WARNING,
                "The download of %s finished (exit code %s).", self.spec.says(), proc.returncode)
        if self.error is None and self.then is not None:
            self.finishing = True
            try:
                await self.then(self.log_file)
            except asyncio.CancelledError:
                raise
            except Exception:       # noqa: BLE001 - the step's failure is its own affair (it says so in the log)
                log.exception("The step after the download of %s failed.", self.spec.says())
            finally:
                self.finishing = False

    def _installer_busy(self) -> bool:
        """An installer works on the component whose folder this download shares (Qwen-Image's, for the full model),
        or may be about to: the hub's own one (while it runs, until it has said what it installs), or another one
        (install.py in a window, tools\\update.bat) holding the lock. Never for a folder no installer writes
        (FLUX.3's)."""
        component = self.spec.waits_for
        if component is None:
            return False
        install = self.install
        if install is not None and install.proc is not None and install.proc.poll() is None:
            return install.component(component) is not None or not install.settled()
        return lock_held(self.install_lock)

    def _spawn(self) -> subprocess.Popen[bytes] | None:
        argv = [str(self.python), str(self.script), "--model", self.model, "--models", str(self.folder),
                "--quality", self.quality]
        env = paths.clean_env(None, models.hf_caches_env())    # the artist's HF caches, which clean_env scrubs
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | (0 if hub_has_console() else subprocess.CREATE_NO_WINDOW)
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.proc = spawn_in_job(argv, paths.CODE, env, flags)
        except OSError as e:
            self.error = (f"The download of {self.spec.says()} could not be started ({e}). Close Spellcast3D and "
                          f"start it again.")
            log.error("%s Command: %s", self.error, subprocess.list2cmdline(argv))
            return None
        assert self.proc.stdout is not None
        threading.Thread(target=pump_output, args=(self.proc.stdout, self.log_file),
                         name=f"{self.model}-download-output", daemon=True).start()
        log.info("Downloading %s (%s, %s) in the background (pid %s).", self.spec.says(),
                 ", ".join(self._wanted or file_names(self.model, self.quality, self.folder)),
                 f"{self._total(self._wanted) / 1e9:.1f} GB", self.proc.pid)
        return self.proc

    def _exit_error(self, code: int | None) -> str | None:
        name, title = self.spec.says(), self.spec.says(start=True)
        hint = f"The details are in code/logs/{self.log_file.name}."
        if code == EXIT_OK:
            if self.available():
                return None
            gone = "its file is not there" if len(manifest(self.model, self.quality, self.folder)) == 1 else \
                "not all of its files are there"
            return f"The download of {name} ended, but {gone}. Start it again. {hint}"
        if code == EXIT_FAILED:
            return (f"The download of {name} stopped, most often because the internet connection dropped. Start it "
                    f"again to continue: what is already downloaded is kept. {hint}")
        if code == EXIT_CHECKSUM:
            return (f"{title} was downloaded, but it is not the expected file, so it was not kept. Start the download "
                    f"again; if that keeps happening, a proxy or an antivirus may be changing it.")
        if code == EXIT_DISK_FULL:
            return (f"The drive ran out of space while downloading {name}. Free some space, then start the download "
                    f"again to continue: what is already downloaded is kept.")
        if code == EXIT_DENIED:
            return (f"{title} could not be written into {self.folder} (access denied: an antivirus, Windows' "
                    f"Controlled folder access or a read-only folder may be blocking it). Allow it, then start the "
                    f"download again. {hint}")
        return f"The download of {name} stopped unexpectedly (exit code {code}). {hint}"

    # ---- queries ----------------------------------------------------------------------------
    def available(self) -> bool:
        return available(self.model, self.quality, self.folder)

    def downloading(self) -> bool:
        """It runs, or waits for an installer to finish with Qwen-Image."""
        return self._task is not None and not self._task.done()

    def _total(self, names: list[str]) -> int:
        files = manifest(self.model, self.quality, self.folder)
        return sum(files[name][0] for name in names) if names else total_bytes(self.model, self.quality, self.folder)

    def snapshot(self) -> dict[str, Any]:
        """/api/status "qwen_full" | "flux3": {"available", "downloading", "waiting", "done_bytes", "total_bytes",
        "error"}: total_bytes the files missing when the running download started (none running: those missing now;
        all there: every file), done_bytes what is on disk of them; waiting while it has not started (an installer
        at work on its folder)."""
        ready = self.available()
        downloading = not ready and self.downloading()      # its last moments, once the files are there
        if ready:
            total = done = total_bytes(self.model, self.quality, self.folder)
        else:
            names = self._wanted if downloading and self._wanted else missing(self.model, self.quality, self.folder)
            total = self._total(names)
            done = downloaded_bytes(self.model, self.quality, self.folder, names)
        state = {"available": ready, "downloading": downloading, "waiting": downloading and self.waiting,
                 "done_bytes": done, "total_bytes": total, "error": None if downloading or ready else self.error}
        if self.then is not None:
            state["finishing"] = ready and self.finishing and self.downloading()
        return state


# ---- after Hi3D's download: its convolutions measured ----------------------------------------------------

HI3D_CALIBRATION_HINT = "The details are in code/logs/hi3d-download.log."


async def hi3d_calibration(log_file: Path) -> None:
    """Once Hi3D's AI models are there: Pixal3D's calibration for Hi3D's convolutions alone (install/components.py
    pixal3d_calibration_command: `pixal3d.calibrate --only spconv`, the record's other sections standing; it waits for a
    quiet card first), in Pixal3D's venv, its output into the download's log. A failure is logged only: Hi3D then tunes
    its convolutions at its first use, and the installer's next run measures them (pixal3d_calibration_work)."""
    from .install import components

    argv = components.pixal3d_calibration_command(only="spconv")
    if argv is None:
        return
    env = paths.clean_env(paths.PIXAL3D, components.pixal3d_calibration_env())
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | (0 if hub_has_console() else subprocess.CREATE_NO_WINDOW)
    try:
        proc = spawn_in_job(argv, paths.PIXAL3D_DIR, env, flags)
    except OSError as e:
        log.warning("Hi3D's convolutions could not be measured (%s); Hi3D tunes them at its first use. Command: %s",
                    e, subprocess.list2cmdline(argv))
        return
    assert proc.stdout is not None
    threading.Thread(target=pump_output, args=(proc.stdout, log_file), name="hi3d-calibration-output",
                     daemon=True).start()
    log.info("Measuring Hi3D's convolutions on the graphics card (pid %s).", proc.pid)
    try:
        while proc.poll() is None:
            await asyncio.sleep(EXIT_POLL)
    except asyncio.CancelledError:
        await asyncio.to_thread(kill_tree, proc.pid)
        raise
    if proc.returncode == 0:
        log.info("Hi3D's convolutions are measured.")
    else:
        log.warning("Measuring Hi3D's convolutions did not finish (exit code %s); Hi3D tunes them at its first use, "
                    "and Spellcast3D's next start measures them. %s", proc.returncode, HI3D_CALIBRATION_HINT)
