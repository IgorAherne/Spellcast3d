"""The rest of the installation, run by the hub in the background (ARCHITECTURE.md "Startup / install chain").

run-spellcast3d.bat installs only the hub (seconds) and starts it. The hub then runs
`install.py --if-needed --skip hub --progress-file code/venvs/install-progress.json` as a child in its Job
Object and shows the installer's progress (GET /api/status "install"): the installer rewrites that file
at least every 2 s. A tool the installer is installing has the state "installing" (procman.py) and jobs
for it wait (jobs.py). The child dies with the hub; the next start runs it again, and the installer then
redoes only what is unfinished (pip's cache and the partial model downloads are kept).

The installer is never started while another one (a window of install.py or tools\\update.bat) holds
code/venvs/.install.lock: the hub waits for that one to finish first.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any, Callable

from . import paths
from .install import models
from .procman import ToolError, hub_has_console, kill_tree, spawn_in_job

log = logging.getLogger(__name__)

PROGRESS_FILE = paths.INSTALL_PROGRESS
INSTALL_LOCK = paths.VENVS / ".install.lock"
STATUSES = frozenset({"pending", "running", "done", "failed", "unsupported", "skipped"})
ACTIVE = frozenset({"pending", "running"})
EXIT_OK, EXIT_FAILED, EXIT_BUSY = 0, 1, 3     # install.py's exit codes
LOCK_POLL = 5.0         # seconds between looks at another installer's lock
EXIT_POLL = 0.5
# The "ready" banner (done_banner) is printed once the console has had nothing to say for this long after the
# installer ended - the tools a waiting job starts print their lines first - so that it is the last thing an artist
# coming back to the window reads; at most READY_AT_MOST seconds after the installer, whatever the console does.
READY_QUIET = 8.0
READY_AT_MOST = 120.0
MAX_TAIL_BYTES = 512 * 1024
LOG_HINT = "The details are in code/logs/install.log."
STOPPED_EARLY = "The installation stopped before this part was finished. Start Spellcast3D again to continue."
# /api/status "install" when the hub has no background installer (nothing is being installed).
IDLE: dict[str, Any] = {"running": False, "finished": True, "error": None, "current": None, "components": {},
                        "message": None}

_IDLE, _WAITING, _RUNNING, _DONE = "idle", "waiting", "running", "done"
_NO_PROGRESS: dict[str, Any] = {"error": None, "current": None, "components": {}}


def base_python() -> Path:
    """The Python the installer runs on, as tools\\environment.bat picks it: SPELLCAST_BASE_PYTHON (developers),
    else the portable tools\\python, else the Python this hub's venv was made from."""
    override = os.environ.get("SPELLCAST_BASE_PYTHON", "").strip().strip('"')
    if override:
        return Path(override)
    if paths.PORTABLE_PYTHON.is_file():
        return paths.PORTABLE_PYTHON
    return Path(getattr(sys, "_base_executable", sys.executable))


def installer_argv(python: Path | None = None) -> list[str]:
    argv = [str(python or base_python()), str(paths.CODE / "install.py"), "--if-needed", "--skip", paths.HUB,
            "--progress-file", str(PROGRESS_FILE)]
    if os.environ.get("SPELLCAST_VERBOSE_INSTALL") == "1":    # every line of the tools' installers on the console
        argv.append("--verbose")
    return argv


def lock_held(lock_file: Path) -> bool:
    """True while another process holds the installer's lock file (see install/main.py)."""
    import msvcrt

    try:
        f = open(lock_file, "a+b")
    except OSError:
        return False        # no venvs folder yet: no installer can be running from it
    with f:
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return True
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        return False


def tail_lines(path: Path, n: int) -> list[str]:
    """The last n (at most 2000) lines of a text file that may be big; [] when it cannot be read."""
    n = max(0, min(n, 2000))
    try:
        with open(path, "rb") as f:
            size = f.seek(0, os.SEEK_END)
            f.seek(max(0, size - MAX_TAIL_BYTES))
            data = f.read()
    except OSError:
        return []
    lines = data.decode("utf-8", "replace").splitlines()
    if size > MAX_TAIL_BYTES:
        lines = lines[1:]           # the first one was cut
    return lines[-n:] if n else []


def done_banner(page_url: str, ok: bool) -> list[str]:
    """What the console says once the installer is through and the console has gone quiet: the artist who went for a
    coffee finds how to open the page (the hub printed it an hour earlier, above a screen of installer output)."""
    address = page_url.replace("http://", "")
    line = "=" * 78
    return ["", line,
            " Spellcast3D is ready to use." if ok else " Spellcast3D is open; a part of the installation needs another try (see above).",
            "",
            f" You can now open your web browser and enter {address} into its address bar,",
            " like a website.",
            line, ""]


# When the console last showed a line (console_echo, and the hub's own log lines through console_touched, which
# __main__ hangs on the console log handler): the "ready" banner waits for a quiet console.
_console_last = 0.0


def console_touched() -> None:
    global _console_last
    _console_last = time.monotonic()


def console_quiet_for() -> float:
    """Seconds since the console last showed a line."""
    return time.monotonic() - _console_last


def console_echo(line: str) -> None:
    """Show one line of the installer's output in the hub's console window (where it used to appear)."""
    console_touched()
    if sys.stdout is not None:
        try:
            print(line, flush=True)
        except (OSError, ValueError):
            pass


def _fraction(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return min(1.0, max(0.0, float(value)))


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _clean_progress(raw: Any) -> dict[str, Any]:
    """The progress file's content with every field checked (it comes from another program)."""
    if not isinstance(raw, dict):
        return _NO_PROGRESS
    components: dict[str, dict[str, Any]] = {}
    for component, entry in (raw.get("components") if isinstance(raw.get("components"), dict) else {}).items():
        if not isinstance(entry, dict):
            continue
        components[str(component)] = {
            "status": entry["status"] if entry.get("status") in STATUSES else "pending",
            "step": str(entry.get("step") or ""), "fraction": _fraction(entry.get("fraction")),
            "step_fraction": _fraction(entry.get("step_fraction")),
            "bytes_done": _count(entry.get("bytes_done")), "bytes_total": _count(entry.get("bytes_total")),
            "message": str(entry.get("message") or "")}
    error, current = raw.get("error"), raw.get("current")
    return {"error": error if isinstance(error, str) and error else None,
            "current": current if isinstance(current, str) else None, "components": components}


def _pump(stream: IO[bytes], echo: Callable[[str], None] | None) -> None:
    """Drain the installer's output (a full pipe would stop it); it writes its own log (install.log)."""
    with stream:
        for raw in iter(stream.readline, b""):
            if echo is not None:
                echo(raw.decode("utf-8", "replace").rstrip("\r\n"))


class BackgroundInstall:
    """Runs the installer as a child of the hub (start / restart / close) and reads its progress file."""

    def __init__(self, argv: list[str], *, cwd: Path = paths.CODE, progress_file: Path = PROGRESS_FILE,
                 lock_file: Path = INSTALL_LOCK, echo: Callable[[str], None] | None = None,
                 lock_poll: float = LOCK_POLL, page_url: str | None = None, ready_quiet: float = READY_QUIET):
        self.argv = argv
        self.cwd = cwd
        self.progress_file = progress_file
        self.lock_file = lock_file
        self.echo = echo
        self.lock_poll = lock_poll
        self.page_url = page_url        # where the page is ("http://127.0.0.1:7770"): said when the installer is through
        self.ready_quiet = ready_quiet  # ... once the console has been quiet this long (the tools' start lines first)
        self.proc: subprocess.Popen[bytes] | None = None
        self.spawn_error: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._spawned_at = 0.0         # time.time(): an older progress file belongs to an earlier run
        self._progress: dict[str, Any] = _NO_PROGRESS
        self._progress_sig: tuple[int, int] | None = None

    # ---- lifecycle (on the event loop) --------------------------------------------------------
    def start(self) -> None:
        """Begin in the background: wait while another installer runs, then run the installer."""
        if self._phase() in (_WAITING, _RUNNING):
            return
        self.proc, self.spawn_error = None, None
        self._progress, self._progress_sig = _NO_PROGRESS, None
        self._task = asyncio.create_task(self._run(), name="background-install")

    def restart(self) -> None:
        """Run the installer once more (it installs only what is still missing). 409 while it runs."""
        phase = self._phase()
        if phase == _WAITING:
            raise ToolError(self.settle_message(), 409)
        if phase == _RUNNING:
            raise ToolError("Spellcast3D is already installing. Wait for it to finish.", 409)
        self.start()

    async def close(self) -> None:
        """Stop the installer with the hub (the Job Object would kill it anyway); the next start resumes."""
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        proc = self.proc
        if proc is not None and proc.poll() is None:
            log.info("Stopping the installation; it continues when Spellcast3D starts again.")
            await asyncio.to_thread(kill_tree, proc.pid)
            await asyncio.to_thread(proc.wait, 10)

    async def _run(self) -> None:
        while lock_held(self.lock_file):
            await asyncio.sleep(self.lock_poll)
        self._spawn()
        proc = self.proc
        if proc is None:
            return
        while proc.poll() is None:
            await asyncio.sleep(EXIT_POLL)
        log.log(logging.INFO if proc.returncode == EXIT_OK else logging.WARNING,
                "The installer finished (exit code %s).", proc.returncode)
        if self.echo is not None and self.page_url:
            await self._say_ready(proc.returncode == EXIT_OK)

    async def _say_ready(self, ok: bool) -> None:
        """The "ready" banner, last: after READY_QUIET seconds without a console line (the hub starts the tools a
        waiting job needs right after the installer, with a line each), at most READY_AT_MOST seconds later."""
        began = time.monotonic()
        while console_quiet_for() < self.ready_quiet and time.monotonic() - began < READY_AT_MOST:
            await asyncio.sleep(min(1.0, max(0.05, self.ready_quiet - console_quiet_for())))
        for line in done_banner(self.page_url or "", ok):
            self.echo(line)         # type: ignore[misc]

    def _spawn(self) -> None:
        try:
            self.progress_file.unlink(missing_ok=True)
        except OSError:
            pass                    # ignored by its age instead
        self._spawned_at = time.time()
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | (0 if hub_has_console() else subprocess.CREATE_NO_WINDOW)
        try:
            # the artist's HuggingFace caches, which clean_env scrubs, for the installer's linking of models from them
            self.proc = spawn_in_job(self.argv, self.cwd, paths.clean_env(None, models.hf_caches_env()), flags)
        except OSError as e:
            self.spawn_error = f"The installer could not be started ({e}). Close Spellcast3D and start it again."
            log.error("%s Command: %s", self.spawn_error, subprocess.list2cmdline(self.argv))
            return
        assert self.proc.stdout is not None
        threading.Thread(target=_pump, args=(self.proc.stdout, self.echo), name="installer-output",
                         daemon=True).start()
        log.info("Checking in the background what still needs installing (installer pid %s).", self.proc.pid)

    # ---- queries ----------------------------------------------------------------------------
    def _phase(self) -> str:
        if self._task is None:
            return _IDLE
        if self.proc is None:
            return _DONE if self._task.done() else _WAITING
        return _RUNNING if self.proc.poll() is None else _DONE

    def settled(self) -> bool:
        """True once it is known what the installer installs (or it has finished, or never ran): until
        then no tool may start, since the installer may replace its files."""
        phase = self._phase()
        if phase == _RUNNING:
            return bool(self._read_progress()["components"])
        return phase != _WAITING

    def settle_message(self) -> str:
        """Why nothing can start yet while not settled(), in words."""
        if self._phase() == _WAITING:
            return "Spellcast3D is being installed in another window; waiting for that to finish…"
        return "Checking which parts of Spellcast3D still need installing…"

    def component(self, component: str) -> dict[str, Any] | None:
        """The installer's entry for a component while it is going to install it or installing it."""
        if self._phase() != _RUNNING:
            return None
        entry = self._read_progress()["components"].get(component)
        return entry if entry is not None and entry["status"] in ACTIVE else None

    def snapshot(self) -> dict[str, Any]:
        """/api/status "install": {"running", "finished", "error", "current", "components": {id: {"status",
        "step", "fraction", "bytes_done", "bytes_total", "message"}}, "message"}."""
        phase = self._phase()
        if phase == _IDLE:
            return {**IDLE, "components": {}}
        progress = self._read_progress() if phase in (_RUNNING, _DONE) else _NO_PROGRESS
        components = {cid: dict(entry) for cid, entry in progress["components"].items()}
        code = self.proc.poll() if self.proc is not None else None
        if phase == _DONE:
            for entry in components.values():
                if entry["status"] in ACTIVE:     # the installer ended without saying so
                    if code == EXIT_OK:
                        entry["status"] = "done"
                    else:
                        entry.update(status="failed", message=entry["message"] or STOPPED_EARLY)
        return {"running": phase == _RUNNING, "finished": phase == _DONE,
                "error": self._error(progress, code) if phase == _DONE else progress.get("error"),
                "current": progress.get("current") if phase == _RUNNING else None,
                "components": components,
                "message": None if self.settled() else self.settle_message()}

    def _error(self, progress: dict[str, Any], code: int | None) -> str | None:
        if self.spawn_error:
            return self.spawn_error
        if progress.get("error"):
            return str(progress["error"])
        if code in (None, EXIT_OK):
            return None
        if code == EXIT_BUSY:
            return ("Spellcast3D could not install its missing parts because another Spellcast3D installer "
                    "was busy. Close that window, then start Spellcast3D again.")
        if code == EXIT_FAILED:
            return f"Some parts of Spellcast3D could not be installed. {LOG_HINT}"
        return f"The installer stopped unexpectedly (exit code {code}). {LOG_HINT}"

    def _read_progress(self) -> dict[str, Any]:
        """The progress file of this run (the last good read while it is being replaced). Not changed by callers."""
        try:
            st = self.progress_file.stat()
        except OSError:
            return self._progress
        signature = (st.st_mtime_ns, st.st_size)
        if signature != self._progress_sig and st.st_mtime >= self._spawned_at - 2:   # older: an earlier run's
            try:
                raw = json.loads(self.progress_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return self._progress
            self._progress, self._progress_sig = _clean_progress(raw), signature
        return self._progress
