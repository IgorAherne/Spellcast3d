"""Pixal3D's multi-view models: whether they are on disk, and their optional one-time download (22 GB).

With two or more views of a prop (projects.py images.views), Pixal3D makes one 3D model from all of them
with a second set of four flow models: the "_mv" checkpoints of TencentARC/Pixal3D, which the installation
leaves out. The artist starts their download from the page (POST /api/pixal3d/multiview/download); until they
are there, a prop with views is made from its front view alone (adapters/pixal3d.py).

The download is neither an installer component nor a repair of Pixal3D: those mark Pixal3D as being
installed, and every 3D job would wait hours for it. The hub runs launchers/mv_download.py with the Pixal3D
venv's Python (it has huggingface_hub) as a child in its Job Object, so it stops with the hub; starting it
again continues it (huggingface_hub keeps the unfinished files). It waits while an installer works on Pixal3D.
Its progress is how many bytes of the files are on disk; its output goes to code/logs/multiview-download.log.

Once every file is there, the same venv makes the faster-loading copies of the four flow models (the fork's
`python -m pixal3d.staged.weights`, install/models.py PIXAL3D_COPIES_MODULE; 14 GB more), as the installer does
for the single-view ones, for the card's precision profile as the installer makes them (gpu.pixal3d_card_env). The models count as available before: until the copies exist, Pixal3D reads the
originals, only more slowly, and a failure is only logged.

The revision is the commit Pixal3D's cache already uses (refs/main), never "main": downloading "main" would
move refs/main to a newer commit, whose snapshot lacks the single-view files, and Pixal3D (which starts
offline) would no longer find them.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import IO, Any, Callable

from . import gpu, paths
from .install import models
from .installer_bg import INSTALL_LOCK, BackgroundInstall, lock_held
from .procman import ToolError, hub_has_console, kill_tree, spawn_in_job

log = logging.getLogger(__name__)

REPO = "TencentARC/Pixal3D"
PINNED_REVISION = "b0cb2e1b794cab9aa0ac38a95d794a4d9337437f"   # when the cache has no refs/main yet
# The files of the multi-view models at that commit and their sizes (install/models.py, which the installer
# can import too: it leaves them out of Pixal3D's download progress).
FILES: dict[str, int] = models.PIXAL3D_MULTI_VIEW_FILES
TOTAL_BYTES = sum(FILES.values())       # 22,000,319,480
SCRIPT = paths.LAUNCHERS / "mv_download.py"
LOG_NAME = "multiview-download.log"
# An unfinished file this big is continued over plain HTTPS (install/components.py RESUME_OVER_HTTPS_BYTES).
RESUME_OVER_HTTPS_BYTES = 200_000_000
DISK_MARGIN = 1_000_000_000             # free space left over after the download, for the drive's own sake
WAIT_POLL = 5.0                         # seconds between looks at an installer that works on Pixal3D
EXIT_POLL = 0.5
EXIT_OK, EXIT_FAILED, EXIT_DISK_FULL, EXIT_NO_HF = 0, 1, 3, 4      # mv_download.py's exit codes
LOG_HINT = "The details are in code/logs/multiview-download.log."
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


# ---- on disk ---------------------------------------------------------------------------------------------

def hf_cache(models_root: Path = paths.MODELS) -> Path:
    """Pixal3D's HuggingFace cache (HF_HOME/hub): models/pixal3d/hub, which code/pixal3d/MODELS links to."""
    return models_root / "pixal3d" / "hub"


def revision(models_root: Path = paths.MODELS) -> str:
    """The commit whose multi-view files Pixal3D loads: its cache's refs/main, else the pinned one."""
    try:
        commit = (models.hf_repo_folder(hf_cache(models_root), REPO) / "refs" / "main").read_text(
            encoding="utf-8").strip()
    except OSError:
        return PINNED_REVISION
    return commit if _COMMIT.match(commit) else PINNED_REVISION


def available(models_root: Path = paths.MODELS) -> bool:
    """Every multi-view file is in the snapshot Pixal3D loads (refs/main), and none is empty."""
    snapshot = models.hf_snapshot(hf_cache(models_root), REPO)
    return snapshot is not None and all(_size(snapshot / rel) for rel in FILES)


def downloaded_bytes(models_root: Path = paths.MODELS) -> int:
    """Bytes of the multi-view files on disk: the finished ones, and the unfinished downloads of the repo
    (<blob>.incomplete: while the single-view files are all there, those are multi-view files)."""
    folder = models.hf_repo_folder(hf_cache(models_root), REPO)
    snapshot = folder / "snapshots" / revision(models_root)
    done = sum(_size(snapshot / rel) for rel in FILES) + _partial_bytes(folder)
    return min(done, TOTAL_BYTES)


def _size(path: Path) -> int:
    """A file's size (a snapshot entry is a link to its blob: the blob's), 0 when it is not there."""
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


def _partial_bytes(repo_folder: Path) -> int:
    blobs = repo_folder / "blobs"
    return sum(_size(p) for p in blobs.glob("*.incomplete")) if blobs.is_dir() else 0


def _free_space(folder: Path) -> int:
    return shutil.disk_usage(folder).free


def _existing(path: Path) -> Path:
    """`path`, or its nearest folder that exists (for the free space of its drive)."""
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def pump_output(stream: IO[bytes], log_file: Path) -> None:
    """The download's output into its log file (a full pipe would stop it)."""
    with stream:
        try:
            out = open(log_file, "a", encoding="utf-8")
        except OSError:
            out = None
        try:
            for raw in iter(stream.readline, b""):
                if out is not None:
                    out.write(raw.decode("utf-8", "replace").rstrip("\r\n") + "\n")
                    out.flush()
        finally:
            if out is not None:
                out.close()


# ---- the download ------------------------------------------------------------------------------------------

class BackgroundDownload:
    """Runs the download as a child of the hub (start / close) and says how far it got (snapshot)."""

    def __init__(self, models_root: Path = paths.MODELS, *, python: Path | None = None, script: Path = SCRIPT,
                 install: BackgroundInstall | None = None, install_lock: Path = INSTALL_LOCK,
                 log_file: Path | None = None, wait_poll: float = WAIT_POLL, repo: Path = paths.PIXAL3D_DIR,
                 card: Callable[[], gpu.GpuInfo | None] | None = None):
        """python: the Pixal3D venv's (it has huggingface_hub). install: the hub's background installer, whose
        work on Pixal3D the download waits for (as it does for another installer holding install_lock). repo:
        the Pixal3D checkout, where the copies' maker runs. card: tells the PC's graphics card (nvidia-smi by
        default), whose precision profile the copies are made for."""
        self.models_root = models_root
        self.python = python or paths.venv_python(paths.PIXAL3D)
        self.script = script
        self.install = install
        self.install_lock = install_lock
        self.log_file = log_file or paths.LOGS / LOG_NAME
        self.wait_poll = wait_poll
        self.repo = repo
        self.card = card or gpu.primary_gpu
        self.proc: subprocess.Popen[bytes] | None = None        # the download
        self.copier: subprocess.Popen[bytes] | None = None      # then the copies' maker
        self.error: str | None = None
        self._task: asyncio.Task[None] | None = None

    # ---- lifecycle (on the event loop) --------------------------------------------------------
    def start(self) -> None:
        """Begin the download in the background. Nothing happens while it runs, or once the models are there.
        ToolError (409) when the drive has too little free space for what is left of it."""
        if self.downloading() or available(self.models_root):
            return
        left = TOTAL_BYTES - downloaded_bytes(self.models_root)
        drive = _existing(hf_cache(self.models_root))
        free = _free_space(drive)
        if free < left + DISK_MARGIN:
            raise ToolError(f"There is not enough free space on drive {drive.drive or drive} for the multi-view "
                            f"models: they need {(left + DISK_MARGIN) / 1e9:.1f} GB, and {free / 1e9:.1f} GB "
                            f"are free. Free some space, then try again.")
        self.error, self.proc = None, None
        self._task = asyncio.create_task(self._run(), name="multiview-download")

    async def close(self) -> None:
        """Stop the download with the hub (the Job Object would kill it anyway); starting it again continues it.
        The copies' maker is stopped too (an unfinished copy is no copy: Pixal3D reads the originals)."""
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        for proc, words in ((self.proc, "Stopping the download of the multi-view models; it continues when it is "
                                        "started again."),
                            (self.copier, "Stopping the making of the multi-view models' faster-loading copies.")):
            if proc is not None and proc.poll() is None:
                log.info(words)
                await asyncio.to_thread(kill_tree, proc.pid)
                await asyncio.to_thread(proc.wait, 10)

    async def _run(self) -> None:
        while self._installer_busy():
            await asyncio.sleep(self.wait_poll)
        if not self.python.is_file():
            self.error = ("Pixal3D is not installed yet, so its multi-view models cannot be downloaded. Wait "
                          "for its installation to finish (or start Spellcast3D again), then try again.")
            return
        proc = self._spawn()
        if proc is None:
            return
        while proc.poll() is None:
            await asyncio.sleep(EXIT_POLL)
        self.error = self._exit_error(proc.returncode)
        log.log(logging.INFO if self.error is None else logging.WARNING,
                "The download of the multi-view models finished (exit code %s).", proc.returncode)
        if self.error is None:
            await self._make_copies()

    async def _make_copies(self) -> None:
        """The faster-loading copies of the downloaded flow models, by the fork's own maker in the Pixal3D venv
        (it also looks at the single-view ones, and leaves those that are up to date alone). Its output goes to
        the download's log; a failure is logged and changes nothing else."""
        snapshot = models.hf_snapshot(hf_cache(self.models_root), REPO)
        if snapshot is None:
            return
        argv = [str(self.python), "-m", models.PIXAL3D_COPIES_MODULE, str(snapshot)]
        # Cast on the CPU: with the card hidden it takes nothing from a 3D job that runs meanwhile, and is told the
        # card through PIXAL3D_CARD_SM, as the installer's maker is: the copies' dtypes follow the card's precision
        # profile, and the maker removes copies of another profile (the single-view ones too). Offline: it
        # downloads nothing.
        card = await asyncio.to_thread(self.card)
        env = paths.clean_env(paths.PIXAL3D, {"HF_HOME": str(hf_cache(self.models_root).parent),
                                              "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "-1",
                                              **gpu.pixal3d_card_env(card)})
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | (0 if hub_has_console() else subprocess.CREATE_NO_WINDOW)
        try:
            self.copier = spawn_in_job(argv, self.repo, env, flags)
        except OSError as e:
            log.warning("The faster-loading copies of the multi-view models could not be made (%s); Pixal3D reads "
                        "the original files, which takes longer. Command: %s", e, subprocess.list2cmdline(argv))
            return
        assert self.copier.stdout is not None
        threading.Thread(target=pump_output, args=(self.copier.stdout, self.log_file), name="multiview-copies-output",
                         daemon=True).start()
        log.info("Making the faster-loading copies of the multi-view models (pid %s).", self.copier.pid)
        while self.copier.poll() is None:
            await asyncio.sleep(EXIT_POLL)
        if self.copier.returncode == 0:
            log.info("The faster-loading copies of the multi-view models are made.")
        else:
            log.warning("The faster-loading copies of the multi-view models were not all made (exit code %s); "
                        "Pixal3D reads the original files, which takes longer. %s", self.copier.returncode,
                        LOG_HINT)

    def _installer_busy(self) -> bool:
        """An installer works on Pixal3D, or may be about to: the hub's own one (while it runs, until it has
        said what it installs), or another one (install.py in a window, tools\\update.bat) holding the lock."""
        install = self.install
        if install is not None and install.proc is not None and install.proc.poll() is None:
            return install.component(paths.PIXAL3D) is not None or not install.settled()
        return lock_held(self.install_lock)

    def _spawn(self) -> subprocess.Popen[bytes] | None:
        cache = hf_cache(self.models_root)
        argv = [str(self.python), str(self.script), "--cache-dir", str(cache), "--revision", revision(self.models_root),
                "--resume-over-https", str(RESUME_OVER_HTTPS_BYTES), *FILES]
        # Every HuggingFace file (the Xet cache, a token) stays in models/pixal3d, none in the artist's profile.
        env = paths.clean_env(paths.PIXAL3D, {"HF_HOME": str(cache.parent), "HF_HUB_DOWNLOAD_TIMEOUT": "60",
                                              "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                                              "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"})
        env.pop("HF_HUB_OFFLINE", None)
        if _partial_bytes(models.hf_repo_folder(cache, REPO)) >= RESUME_OVER_HTTPS_BYTES:
            env["HF_HUB_DISABLE_XET"] = "1"     # continued where it stopped (Xet would start the file again)
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | (0 if hub_has_console() else subprocess.CREATE_NO_WINDOW)
        try:
            cache.mkdir(parents=True, exist_ok=True)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.proc = spawn_in_job(argv, paths.CODE, env, flags)
        except OSError as e:
            self.error = (f"The download of the multi-view models could not be started ({e}). Close Spellcast3D "
                          f"and start it again.")
            log.error("%s Command: %s", self.error, subprocess.list2cmdline(argv))
            return None
        assert self.proc.stdout is not None
        threading.Thread(target=pump_output, args=(self.proc.stdout, self.log_file), name="multiview-download-output",
                         daemon=True).start()
        log.info("Downloading Pixal3D's multi-view models in the background (pid %s).", self.proc.pid)
        return self.proc

    def _exit_error(self, code: int | None) -> str | None:
        if code == EXIT_OK:
            if available(self.models_root):
                return None
            return ("The download of the multi-view models ended, but some of their files are still missing. "
                    f"Start it again to finish it. {LOG_HINT}")
        if code == EXIT_FAILED:
            return ("The download of the multi-view models stopped, most often because the internet connection "
                    "dropped. Start it again to continue: what is already downloaded is kept.")
        if code == EXIT_DISK_FULL:
            return ("The drive ran out of space while downloading the multi-view models. Free some space, then "
                    "start the download again to continue: what is already downloaded is kept.")
        if code == EXIT_NO_HF:
            return ("Pixal3D's installation is incomplete, so its multi-view models cannot be downloaded. Start "
                    f"Spellcast3D again to repair it. {LOG_HINT}")
        return f"The download of the multi-view models stopped unexpectedly (exit code {code}). {LOG_HINT}"

    # ---- queries ----------------------------------------------------------------------------
    def downloading(self) -> bool:
        """It runs, or waits for an installer to finish with Pixal3D."""
        return self._task is not None and not self._task.done()

    def snapshot(self) -> dict[str, Any]:
        """/api/status "multiview": {"available", "downloading", "done_bytes", "total_bytes", "error"}."""
        ready = available(self.models_root)
        downloading = not ready and self.downloading()      # its last moments, once the files are all there
        return {"available": ready, "downloading": downloading,
                "done_bytes": TOTAL_BYTES if ready else downloaded_bytes(self.models_root),
                "total_bytes": TOTAL_BYTES, "error": None if downloading or ready else self.error}
