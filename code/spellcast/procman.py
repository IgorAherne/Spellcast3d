"""Tool processes: spawn, health polling, log capture, stop, GPU exclusivity.

One ToolProcess per adapter. Every tool process is put into ONE Windows Job Object with
KILL_ON_JOB_CLOSE: when the hub exits for any reason (even killed from Task Manager) Windows
closes the job handle and kills every tool, so no orphan keeps holding VRAM or RAM.

Graceful stop uses CTRL_BREAK_EVENT, which Windows only delivers to processes attached to
the sender's console. So when the hub has a console (the normal .bat launch) tools share it
(CREATE_NEW_PROCESS_GROUP, output piped, no extra window). Only a console-less hub uses
CREATE_NO_WINDOW, and then stopping skips straight from adapter.on_stop() to killing the tree.

While the background installer (installer_bg.py) installs a tool, the tool's state is "installing" and
it cannot be started; a start waits until the installer has said what it installs.

GPU exclusivity concerns the uses_gpu tools only: Instant Meshes (CPU) starts and runs beside any of them, which
is how it unwraps the UVs of LATO.2's and TriFlow's clean meshes while they hold the graphics card.

Exclusivity is about the graphics card, not the processes. A tool that loads its models onto the card when it starts
(ToolAdapter.gpu_at_start: Qwen-Image, Forge, LATO.2, TriFlow) holds the card from its start, and starting one stops
every other tool that holds it. A tool that touches the card only when a job begins (Pixal3D, whose start is only its
imports) holds nothing until then: it may start early, beside the tool on the card and even while that one is busy,
so its 10-30 s of start are over by the time the artist asks for its work (warm), and a job takes the card for it
just before it runs (claim_gpu), which stops the other tool then.
"""

from __future__ import annotations

import asyncio
import codecs
import collections
import ctypes
import json
import logging
import os
import re
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Collection, Mapping

import httpx

from . import paths
from .adapters.base import StageError, ToolAdapter
from .install import progress as install_progress
from .install import state as install_state
from .settings import SettingsStore

if TYPE_CHECKING:
    from .installer_bg import BackgroundInstall

log = logging.getLogger(__name__)

NOT_INSTALLED = "not_installed"
UNSUPPORTED = "unsupported"
INSTALLING = "installing"
STOPPED = "stopped"
STARTING = "starting"
READY = "ready"
BUSY = "busy"
STOPPING = "stopping"
ERROR = "error"

LOG_RING_LINES = 2000
HEALTH_INTERVAL = 1.0       # seconds between health probes while starting
HEALTH_TIMEOUT = 5.0        # one probe
EXIT_POLL_INTERVAL = 0.5
ON_STOP_TIMEOUT = 3.0
BREAK_WAIT = 5.0            # after CTRL_BREAK_EVENT, before killing the tree
KILL_WAIT = 5.0
MAX_LOG_BYTES = 10 * 1024 * 1024
PORT_SEARCH_SPAN = 100
# Why no GPU tool starts while the installer measures the card for Pixal3D (install_wait, start).
CALIBRATION_WAIT = "Waiting for the graphics-card calibration of the 3D step to finish (a few minutes)…"
# How long a start waits for the background installer to say what it installs (it knows within a second
# or two; much longer only while an installer runs in another window).
INSTALL_SETTLE_TIMEOUT = 120.0
# Log lines worth quoting when a tool dies (ToolProcess.last_words); the launchers' own hints
# ("[spellcast] Pixal3D could not start ...") match too, their progress lines do not.
_MEANINGFUL_LINE = re.compile(r"error|exception|failed|fatal|ran out of|out of memory|not found|missing|"
                              r"incomplete|cannot|could not|denied", re.IGNORECASE)


# How a stopped tool is started again, in words that fit everywhere the message is shown (the status
# line, Settings > Debugging, a prop's job error, a toast).
RESTART_HOW = "it starts by itself when you generate, or from Settings > Debugging."


class ToolError(Exception):
    """A user-facing tool problem. `status_code` becomes the HTTP status."""

    def __init__(self, message: str, status_code: int = 409):
        super().__init__(message)
        self.status_code = status_code


# ---- Windows plumbing ------------------------------------------------------------------------

_CREATE_SUSPENDED = 0x00000004
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_SUSPEND_RESUME = 0x0800

if os.name == "nt":
    from ctypes import wintypes

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),          # ULONG_PTR
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimitInformation),
                    ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    # Own WinDLL instances, so these argtypes never affect ctypes.windll users elsewhere.
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _k32.SetInformationJobObject.restype = wintypes.BOOL
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.AssignProcessToJobObject.restype = wintypes.BOOL
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.CloseHandle.restype = wintypes.BOOL
    _k32.GetConsoleProcessList.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
    _k32.GetConsoleProcessList.restype = wintypes.DWORD
    _k32.SetErrorMode.argtypes = [wintypes.UINT]
    _k32.SetErrorMode.restype = wintypes.UINT
    _ntdll = ctypes.WinDLL("ntdll")
    _ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    _ntdll.NtResumeProcess.restype = ctypes.c_long

# SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX: Windows reports a hard error to the process
# instead of drawing a dialog for it. A process inherits the mode from the one that starts it.
_QUIET_HARD_ERRORS = 0x0001 | 0x0002 | 0x8000


class _KillOnCloseJob:
    """The hub's Job Object. The handle is deliberately never closed: the OS closes it when the
    hub process ends, and that is what kills the tools."""

    def __init__(self) -> None:
        self.handle = _k32.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _k32.SetInformationJobObject(self.handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                                            ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())


_job: _KillOnCloseJob | None = None
_job_lock = threading.Lock()


def spawn_in_job(argv: list[str], cwd: Path, env: dict[str, str], creationflags: int,
                 stdin: int = subprocess.DEVNULL) -> subprocess.Popen[bytes]:
    """Popen with stdout+stderr piped, started suspended, put into the kill-on-close job, then resumed. stdin:
    subprocess.PIPE for a worker that takes commands (omnitexture_worker.py).

    Starting suspended guarantees that even processes the tool spawns immediately (the venv
    launcher starts the real python.exe right away) are inside the job.
    """
    global _job
    if os.name != "nt":
        return subprocess.Popen(argv, cwd=cwd, env=env, stdin=stdin,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with _job_lock:
        if _job is None:
            _job = _KillOnCloseJob()
    # A tool that cannot start (a DLL that fails to initialise: exit code 0xc0000142) or crashes then ends at once,
    # with its exit code in the log, instead of standing behind a Windows dialog until the hub kills it.
    _k32.SetErrorMode(_QUIET_HARD_ERRORS)
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=stdin, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, creationflags=creationflags | _CREATE_SUSPENDED)
    handle = _k32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE | _PROCESS_SUSPEND_RESUME, False, proc.pid)
    if not handle:
        proc.kill()
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not _k32.AssignProcessToJobObject(_job.handle, handle):
            log.warning("Could not put process %s into the hub's job object (error %s); it may outlive "
                        "the hub if the hub crashes.", proc.pid, ctypes.get_last_error())
        status = _ntdll.NtResumeProcess(handle)
        if status != 0:
            proc.kill()
            raise OSError(f"could not resume the new process (NTSTATUS {status & 0xFFFFFFFF:#010x})")
    finally:
        _k32.CloseHandle(handle)
    return proc


def hub_has_console() -> bool:
    if os.name != "nt":
        return False
    return _k32.GetConsoleProcessList((wintypes.DWORD * 1)(), 1) > 0


def kill_tree(pid: int) -> None:
    """taskkill /T /F: the venv launcher, the real python.exe and anything they started."""
    taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
    subprocess.run([str(taskkill), "/T", "/F", "/PID", str(pid)], stdin=subprocess.DEVNULL,
                   capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def port_is_free(port: int) -> bool:
    # Both binds are needed on Windows: a 127.0.0.1 bind succeeds even while 0.0.0.0:port is listening.
    for host in ("127.0.0.1", "0.0.0.0"):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
            except OSError:
                return False
    return True


def find_free_port(start: int, avoid: set[int] | frozenset[int] = frozenset()) -> int:
    for port in range(start, min(start + PORT_SEARCH_SPAN, 65536)):
        if port not in avoid and port_is_free(port):
            return port
    raise ToolError(f"No free network port found between {start} and {start + PORT_SEARCH_SPAN - 1}. "
                    f"Close other programs using them, or change the ports in code/settings.json.", 503)


# ---- tool output capture ------------------------------------------------------------------------

class _OutputPump(threading.Thread):
    """Reads a tool's merged stdout/stderr into the ring buffer and the log file.

    Like a terminal, a line ended by a bare \\r (tqdm progress bars) is overwritten by the next
    one, so progress does not flood the buffer; only finished lines go to the log file.
    """

    _SPLIT = re.compile(r"(\r\n|\n|\r)")

    def __init__(self, tp: ToolProcess, stream: IO[bytes], log_file: IO[str]):
        super().__init__(name=f"{tp.tool_id}-output", daemon=True)
        self.tp, self.stream, self.log_file = tp, stream, log_file

    def run(self) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending, overwrite = "", False
        try:
            while chunk := self.stream.read1(65536):  # type: ignore[attr-defined]
                parts = self._SPLIT.split(pending + decoder.decode(chunk))
                pending = parts[-1]
                finished = []
                with self.tp.lines_lock:
                    for text, sep in zip(parts[0:-1:2], parts[1::2]):
                        if overwrite and self.tp.lines:
                            if text or sep == "\r":
                                self.tp.lines[-1] = text
                        else:
                            self.tp.lines.append(text)
                        overwrite = sep == "\r"
                        if not overwrite:
                            finished.append(self.tp.lines[-1])
                if finished:
                    self.log_file.write("".join(line + "\n" for line in finished))
                    self.log_file.flush()
            if pending:
                self.tp.add_line(pending)
                self.log_file.write(pending + "\n")
        except (OSError, ValueError) as e:
            log.debug("output pump of %s ended: %s", self.tp.tool_id, e)
        finally:
            try:
                self.log_file.close()
            except OSError:
                pass


# ---- one tool ----------------------------------------------------------------------------------

class ToolProcess:
    def __init__(self, adapter: ToolAdapter):
        self.adapter = adapter
        self.tool_id: str = adapter.tool_id
        self.proc: subprocess.Popen[bytes] | None = None
        self.port: int | None = None
        self.state: str = STOPPED
        self.message: str = ""
        self.busy_count = 0
        self.started_at = 0.0
        self.lines: collections.deque[str] = collections.deque(maxlen=LOG_RING_LINES)
        self.lines_lock = threading.Lock()
        self.supported_cache: tuple[bool, str] | None = None
        self.start_task: asyncio.Task[None] | None = None
        self.stop_task: asyncio.Task[None] | None = None
        self.monitor_task: asyncio.Task[None] | None = None
        self.anew_task: asyncio.Task[None] | None = None    # stopping to start anew for another wish (warm)
        # A job has taken the graphics card for this process (ProcessManager.claim_gpu); a new process starts without.
        self.gpu_claimed = False

    @property
    def name(self) -> str:
        return self.adapter.display_name

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def holds_gpu(self) -> bool:
        """Whether the process has the graphics card, or is starting and takes it at once: a tool that loads its
        models at start (ToolAdapter.gpu_at_start) from its start on, any other GPU tool once a job has claimed the
        card for it (ProcessManager.claim_gpu), until it stops."""
        if not self.adapter.uses_gpu:
            return False
        if self.adapter.gpu_at_start:
            return self.alive() or self.state == STARTING
        return self.gpu_claimed and self.alive()

    def add_line(self, line: str) -> None:
        with self.lines_lock:
            self.lines.append(line)

    def tail(self, n: int) -> list[str]:
        with self.lines_lock:
            return list(self.lines)[-n:] if n > 0 else []

    def last_words(self, n: int = 3) -> str:
        """The last few log lines of the current run that say what went wrong, for error messages:
        error / exception lines and the launchers' own hints (without their "[spellcast] " tag).
        Warnings, progress output and traceback frames are left out; "" when nothing qualifies
        (e.g. the process was killed)."""
        with self.lines_lock:
            lines = [line.strip() for line in self.lines]
        start = max((i for i, line in enumerate(lines) if line.startswith("==== ") and " starting " in line),
                    default=-1)
        picked = [line.removeprefix("[spellcast] ") for line in lines[start + 1:]
                  if _MEANINGFUL_LINE.search(line) and "Warning" not in line
                  and not line.startswith(("Traceback", 'File "'))]
        return " | ".join(picked[-n:])


def _not_installed(tp: ToolProcess, reason: str) -> str:
    """The installer's reasons are sentences for the artist that say what to do; short ones are not."""
    if reason.endswith("."):
        return f"{tp.name} is not installed. {reason}"
    return (f"{tp.name} is not installed" + (f" ({reason})" if reason else "") +
            ". Close Spellcast3D and double-click run-spellcast3d.bat again to finish the installation.")


_EXIT_CODE_MEANINGS = {
    0xC0000142: "Windows could not initialise one of its DLLs when it started: usually too little free memory, or a "
                "security program",
}


def _exit_code_text(code: int) -> str:
    if 0 <= code <= 255:
        return str(code)
    text = f"{code & 0xFFFFFFFF:#010x}"
    meaning = _EXIT_CODE_MEANINGS.get(code & 0xFFFFFFFF)
    return f"{text}: {meaning}" if meaning else text


class ProcessManager:
    def __init__(self, adapters: dict[str, ToolAdapter], settings: SettingsStore, client: httpx.AsyncClient,
                 *, hub_port: int | None = None, logs_dir: Path = paths.LOGS,
                 install: BackgroundInstall | None = None):
        self.tools: dict[str, ToolProcess] = {tid: ToolProcess(a) for tid, a in adapters.items()}
        self.settings = settings
        self.client = client
        self.hub_port = hub_port
        self.logs_dir = logs_dir
        self.install = install
        self._card_word: dict[str, Any] | None = None       # what free_card_for_calibration last told the installer
        self.console = hub_has_console()
        self._spawn_lock = asyncio.Lock()

    async def startup(self) -> None:
        """Ask every adapter whether this machine can run it (may query the GPU) once, off the event loop."""
        for tp in self.tools.values():
            try:
                tp.supported_cache = await asyncio.to_thread(tp.adapter.supported)
            except Exception:
                log.exception("%s: supported() failed; assuming it is supported", tp.tool_id)
                tp.supported_cache = (True, "")

    # ---- queries ---------------------------------------------------------------------------
    def get(self, tool_id: str) -> ToolProcess:
        tp = self.tools.get(tool_id)
        if tp is None:
            raise ToolError(f'There is no tool called "{tool_id}".', 404)
        return tp

    def _availability(self, tp: ToolProcess) -> tuple[str | None, str, bool, bool]:
        """(blocking state or None, reason, installed, supported). While the tool is being installed, the
        reason is the installer's progress in words."""
        supported, why = tp.supported_cache if tp.supported_cache is not None else (True, "")
        try:
            entry = tp.adapter.install_status()
        except Exception as e:
            entry = {"status": install_state.NOT_INSTALLED, "reason": str(e)}
        status, reason = entry.get("status"), str(entry.get("reason") or "")
        if status == install_state.UNSUPPORTED:
            return UNSUPPORTED, reason or why, False, False
        if not supported:
            return UNSUPPORTED, why, status == install_state.INSTALLED, False
        installing = self._installing(tp, status)
        if installing is not None:
            return INSTALLING, installing, False, True
        if status != install_state.INSTALLED:
            return NOT_INSTALLED, reason, False, True
        return None, "", True, True

    def _installing(self, tp: ToolProcess, install_status: Any) -> str | None:
        """What the background installer does to the tool, in words, or None when it is not installing it.
        A tool that is not installed counts as installing until the installer has said what it installs."""
        if self.install is None:
            return None
        entry = self.install.component(tp.tool_id)
        if entry is not None:
            if entry["status"] == "pending":
                return f"{tp.name} is waiting to be installed."
            fraction = entry.get("fraction")
            percent = f" ({round(fraction * 100)}%)" if fraction is not None else ""
            return f"{tp.name} is being installed{percent}" + (f": {entry['step']}" if entry.get("step") else ".")
        if install_status != install_state.INSTALLED and not self.install.settled():
            return f"{tp.name} is about to be installed."
        return None

    def install_wait(self, tool_id: str) -> str | None:
        """Why a job for the tool has to wait before it starts the tool (the background installer installs
        it, or has not said yet what it installs; a GPU tool also while the installer measures the graphics card
        for Pixal3D, so that the measurement is of a quiet card), or None when it can go ahead."""
        if self.install is None:
            return None
        if self.get(tool_id).adapter.uses_gpu and self.calibrating():
            return CALIBRATION_WAIT
        entry = self.install.component(tool_id)
        if entry is not None:
            fraction = entry.get("fraction")
            adapter = self.get(tool_id).adapter
            who = f"the {adapter.role}" if adapter.role else adapter.display_name
            return (f"Waiting for {who} to finish installing"
                    + (f" ({round(fraction * 100)}%)" if fraction is not None else "…"))
        return None if self.install.settled() else self.install.settle_message()

    def calibrating(self) -> bool:
        """Is the background installer measuring the graphics card for Pixal3D right now (its step says so)?"""
        if self.install is None:
            return False
        entry = self.install.component(paths.PIXAL3D)
        return bool(entry) and str(entry.get("step") or "").startswith(install_progress.CALIBRATING_STEP)

    def free_card_for_calibration(self, at_work: Collection[str] = (), card_file: Path = paths.CALIBRATION_CARD) -> None:
        """Called every moment while the installer runs. While it measures the card for Pixal3D (calibrating), none
        of our own tools may keep video memory: an image generator left loaded after a picture is idle, so the
        calibration's look at the card's load passes, yet it would measure in what is left of an 8 GB card. Every
        GPU tool that runs without a job is stopped (`at_work`: the tools a job starts or runs on - they finish it
        first; no new job or early start takes the card meanwhile: install_wait, start), and `card_file` tells the
        installer, which waits for it, whether the card is free of them yet."""
        if not self.calibrating():
            self._card_word = None      # the next calibration's installer is told anew
            return
        holding = [tp for tp in self.tools.values() if tp.adapter.uses_gpu and tp.alive()]
        for tp in holding:
            idle = tp.state == READY and not tp.busy_count and tp.tool_id not in at_work
            if idle and (tp.stop_task is None or tp.stop_task.done()):
                log.info("Stopping %s: the graphics card is measured for the 3D step, with nothing else on it", tp.name)
                self.request_stop(tp.tool_id)
        word = {"free": not holding, "waiting_for": [tp.name for tp in holding]}
        if word != self._card_word:
            try:
                paths.write_text_atomic(card_file, json.dumps(word))
                self._card_word = word
            except OSError as e:
                log.debug("Could not write %s: %s", card_file, e)

    def status(self, tool_id: str) -> dict[str, Any]:
        tp = self.get(tool_id)
        blocked, reason, installed, supported = self._availability(tp)
        state, message = tp.state, tp.message
        if blocked and state in (STOPPED, ERROR) and not tp.alive():
            state = blocked
            if blocked == UNSUPPORTED:
                message = reason or f"{tp.name} cannot run on this computer."
            elif blocked == INSTALLING:
                message = reason
            else:
                message = _not_installed(tp, reason)
        elif state == READY and tp.busy_count:
            state, message = BUSY, "Working…"
        starting_seconds = int(time.monotonic() - tp.started_at) if state == STARTING else None
        if state == STARTING:
            message = f"Starting {tp.name}… ({starting_seconds} s)"
        running = state in (READY, BUSY)
        return {
            "state": state, "message": message,
            # how long the start has taken so far: the page's "Loading…" lines count on from it, wherever they show
            "starting_seconds": starting_seconds,
            "port": tp.port if tp.alive() else None,
            "ui_url": f"http://127.0.0.1:{tp.port}{tp.adapter.ui_path}" if running else None,
            "display_name": tp.name, "stage": tp.adapter.stage, "uses_gpu": tp.adapter.uses_gpu,
            # whether a start puts the tool on the graphics card, and whether it is on it now (ToolProcess.holds_gpu)
            "gpu_at_start": tp.adapter.uses_gpu and tp.adapter.gpu_at_start, "holds_gpu": tp.holds_gpu(),
            "installed": installed, "supported": supported, "reason": reason,
            # {"name", "url", "noncommercial"} of a license the artist must know about (TriFlow's), else None
            "license": dict(tp.adapter.license) if tp.adapter.license else None,
        }

    def all_status(self) -> dict[str, dict[str, Any]]:
        return {tid: self.status(tid) for tid in self.tools}

    def log_tail(self, tool_id: str, n: int = 200) -> list[str]:
        return self.get(tool_id).tail(max(0, min(n, LOG_RING_LINES)))

    def set_busy(self, tool_id: str, busy: bool) -> None:
        tp = self.get(tool_id)
        tp.busy_count = max(0, tp.busy_count + (1 if busy else -1))

    # ---- start ------------------------------------------------------------------------------
    def check_available(self, tool_id: str, *, installing_ok: bool = True) -> ToolProcess:
        """Raise ToolError when the tool is not installed or cannot run on this computer; also while it is
        being installed unless `installing_ok` (a job waits for that)."""
        tp = self.get(tool_id)
        blocked, reason, _, _ = self._availability(tp)
        if blocked == UNSUPPORTED:
            raise ToolError(reason or f"{tp.name} cannot run on this computer.")
        if blocked == NOT_INSTALLED:
            raise ToolError(_not_installed(tp, reason))
        if blocked == INSTALLING and not installing_ok:
            raise ToolError(f"{reason} It can start once that is done.")
        return tp

    def start(self, tool_id: str) -> ToolProcess:
        """Begin starting a tool (non-blocking, idempotent). Raises ToolError if it cannot start. A tool that takes
        the graphics card at start stops the GPU tool that holds it (and refuses while that one is busy); one that
        takes it only at its first job starts beside it (see the module docstring)."""
        tp = self.check_available(tool_id, installing_ok=False)
        if tp.start_task and not tp.start_task.done():
            return tp
        if tp.state == READY and tp.alive():
            return tp
        if tp.adapter.uses_gpu and self.calibrating():
            raise ToolError(CALIBRATION_WAIT, 409)
        rival = self._busy_gpu_rival(tp)
        if rival:
            raise ToolError(f"{rival.name} is busy with a job and shares the graphics card with {tp.name}. "
                            f"Wait for the job to finish (or cancel it), then try again.")
        if tp.stop_task is None or tp.stop_task.done():
            # While a stop is still running the state stays "stopping"; _start switches it later,
            # otherwise the exit watcher would report the old process's exit as a failed start.
            tp.state, tp.message, tp.started_at = STARTING, "", time.monotonic()
        tp.start_task = asyncio.create_task(self._start(tp), name=f"start-{tool_id}")
        return tp

    async def ensure_running(self, tool_id: str, wish: Mapping[str, Any] | None = None) -> ToolProcess:
        """Start the tool if needed and wait until it is ready. Safe to call concurrently. With a `wish` (a job's:
        {"stage"}), a tool that runs, or loads, for something else (ToolAdapter.serves: the image generator in
        another stage's arithmetic) and has no job on it is stopped first, and the start is for the wish - once
        more when a start already under way turns out to have been for something else."""
        tp = await self._ready(tool_id, wish)
        if wish is not None and not tp.busy_count and not tp.adapter.serves(wish, self.settings.get()):
            tp = await self._ready(tool_id, wish)
        return tp

    async def _serve(self, tool_id: str, wish: Mapping[str, Any]) -> None:
        """ensure_running's wish: a start anew already under way (warm) is waited for; a process that does not serve
        the wish is stopped; the next launch is for it."""
        tp = self.get(tool_id)
        if tp.anew_task is not None and not tp.anew_task.done():
            await asyncio.shield(tp.anew_task)
        if tp.alive() and not tp.busy_count and not tp.adapter.serves(wish, self.settings.get()):
            log.info("Starting %s anew: it was started for something else than %s", tp.name, dict(wish))
            await self.stop(tool_id)
        if not tp.alive() and not (tp.start_task and not tp.start_task.done()):
            tp.adapter.aim(wish)

    async def _ready(self, tool_id: str, wish: Mapping[str, Any] | None) -> ToolProcess:
        """The tool started (for `wish`, see _serve) and ready. Raises ToolError."""
        if wish is not None:
            await self._serve(tool_id, wish)
        tp = self.start(tool_id)
        task = tp.start_task
        if task is not None and not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                raise ToolError(f"{tp.name} was stopped before it finished starting.") from None
        if tp.state != READY or not tp.alive():
            raise ToolError(tp.message or f"{tp.name} could not be started. See its log for details.", 503)
        return tp

    def warm(self, tool_id: str, wish: Mapping[str, Any] | None = None, *, displace: bool = False) -> bool:
        """Have a tool ready before a job asks for it (the page asks as the artist arrives at the tool's stage, and
        again when they pick something else there): a stopped one starts early (start_early: when that displaces
        nothing, unless `displace`). `wish` is what the artist is about to ask of it (ToolAdapter.aim: the image
        generator's model, and the stage whose arithmetic it takes): a tool that runs without a job but was started
        for something else (ToolAdapter.serves)
        is stopped and started anew for the wish, also in the middle of loading, so that nothing is loaded the
        artist no longer wants; a wish that comes before the process is launched replaces the one before. ->
        whether a start began (or is under way for the wish).
        Never raises for a tool that cannot start now (not installed, unsupported, being installed, in error,
        another tool on the card); a tool in error is started again by a job or from Settings > Debugging, so a
        start that fails is not repeated at every arrival. ToolError (404) only for a tool that does not exist."""
        tp = self.get(tool_id)
        wish = wish or {}
        unlaunched = tp.state == STARTING and not tp.alive()
        if unlaunched or (tp.anew_task is not None and not tp.anew_task.done()):
            tp.adapter.aim(wish)        # the launch to come takes the newest wish
            return True
        if tp.state in (STARTING, READY) and tp.alive():
            if tp.busy_count or tp.adapter.serves(wish, self.settings.get()):
                return False
            tp.adapter.aim(wish)
            log.info("Starting %s anew: it was started for something else than %s", tp.name, dict(wish))
            tp.anew_task = asyncio.create_task(self._start_anew(tp), name=f"anew-{tool_id}")
            return True
        return self.start_early(tool_id, wish=wish, displace=displace)

    async def _start_anew(self, tp: ToolProcess) -> None:
        """Stop the tool, then start it (the start cancelled by the stop is over by then)."""
        await self.stop(tp.tool_id)
        try:
            self.start(tp.tool_id)
        except ToolError as e:
            log.info("%s was not started anew: %s", tp.name, e)

    def start_early(self, tool_id: str, *, wish: Mapping[str, Any] | None = None, displace: bool = False) -> bool:
        """Start a stopped tool before a job asks for it, when that displaces nothing: a tool that takes the graphics
        card only at its first job always, one that takes it at start only while no other GPU tool holds the card or
        is busy (whatever gpu_exclusive says: such a start never puts a second tool on the card). With `displace`
        also while another GPU tool holds the card without a job on it: that one is stopped for this one as a start
        would (the job runner's hand-off between stages, jobs.NEXT_STAGE; the page's arrival at Images). Never while
        a GPU tool is busy with a job. The start is for `wish` (ToolAdapter.aim; none: the tool's default)."""
        tp = self.get(tool_id)
        if tp.state != STOPPED or tp.alive() or self._availability(tp)[0] is not None:
            return False
        if tp.adapter.uses_gpu and tp.adapter.gpu_at_start:
            others = [o for o in self.tools.values() if o is not tp and o.adapter.uses_gpu]
            if any(o.busy_count and o.alive() for o in others):
                return False
            if not displace and any(o.holds_gpu() for o in others):
                return False
        tp.adapter.aim(wish or {})
        try:
            self.start(tool_id)
        except ToolError as e:
            log.info("%s was not started early: %s", tp.name, e)
            return False
        log.info("Starting %s early, before a job asks for it", tp.name)
        return True

    async def claim_gpu(self, tool_id: str) -> ToolProcess:
        """Give the graphics card to a running tool just before a job runs on it: every other tool that holds the
        card is stopped first (when gpu_exclusive), then the tool holds it until it stops. A tool that took the
        card at start finds nothing left to stop; one started without it (gpu_at_start False) takes it here, at
        its first job. Idempotent; a tool that does not use the card is left as it is. Raises ToolError when the
        tool is no longer running (then nothing else is stopped), or stopped while the others were being stopped."""
        tp = self.get(tool_id)
        if not tp.adapter.uses_gpu:
            return tp
        async with self._spawn_lock:      # no start stops or spawns a tool in the middle of the claim
            if tp.state == READY and tp.alive():
                await self._free_gpu(tp)
            if tp.state != READY or not tp.alive():
                crashed = tp.state == ERROR and tp.message
                raise ToolError(tp.message if crashed else f"{tp.name} was stopped before its job could begin. "
                                                           f"Try again.", 503)
            tp.gpu_claimed = True
        return tp

    async def _free_gpu(self, tp: ToolProcess) -> None:
        """Stop every other tool that holds the graphics card (ToolProcess.holds_gpu), so `tp` has it to itself;
        nothing when gpu_exclusive is off. A tool that is running without the card (started early, before its
        first job) is left running. Called with _spawn_lock held."""
        if not tp.adapter.uses_gpu or not self.settings.get().get("gpu_exclusive", True):
            return
        for other in self.tools.values():
            if other is not tp and other.holds_gpu():
                log.info("Stopping %s to free the graphics card for %s", other.name, tp.name)
                await self.stop(other.tool_id)

    def _busy_gpu_rival(self, tp: ToolProcess) -> ToolProcess | None:
        """The GPU tool, busy with a job, that starting `tp` would have to stop, or None. A tool that takes the
        card only at its first job stops nothing when it starts, so nothing is in its way."""
        if (not tp.adapter.uses_gpu or not tp.adapter.gpu_at_start
                or not self.settings.get().get("gpu_exclusive", True)):
            return None
        return next((o for o in self.tools.values()
                     if o is not tp and o.adapter.uses_gpu and o.busy_count and o.alive()), None)

    async def _start(self, tp: ToolProcess) -> None:
        """Stop the GPU tool that holds the card (only for a tool that takes it at start), spawn, wait for health.
        Never raises: the outcome is tp.state / tp.message."""
        try:
            await self._install_settled(tp)
            if tp.stop_task and not tp.stop_task.done():
                await asyncio.shield(tp.stop_task)
                tp.state, tp.message, tp.started_at = STARTING, "", time.monotonic()
            async with self._spawn_lock:
                if tp.adapter.gpu_at_start:
                    await self._free_gpu(tp)
                settings = self.settings.get()
                tp.state, tp.started_at = STARTING, time.monotonic()
                ready_timeout = self._spawn(tp, settings)
            await self._wait_ready(tp, ready_timeout)
        except asyncio.CancelledError:
            raise
        except ToolError as e:
            self._fail(tp, str(e))
        except Exception as e:
            log.exception("Starting %s failed", tp.tool_id)
            self._fail(tp, f"{tp.name} could not be started: {e}. If this keeps happening, close Spellcast3D "
                           f"and run run-spellcast3d.bat again to repair the installation.")

    async def _install_settled(self, tp: ToolProcess) -> None:
        """Wait until the background installer has said what it installs (a tool must not start while its
        files are being replaced); ToolError when that includes this tool."""
        if self.install is None or self.install.settled():
            return
        deadline = time.monotonic() + INSTALL_SETTLE_TIMEOUT
        while not self.install.settled():
            if time.monotonic() > deadline:
                raise ToolError(f"{tp.name} cannot start yet. {self.install.settle_message()}")
            await asyncio.sleep(0.2)
        self.check_available(tp.tool_id, installing_ok=False)

    def _spawn(self, tp: ToolProcess, settings: dict[str, Any]) -> float:
        """Launch the process; returns the adapter's ready_timeout."""
        avoid = {o.port for o in self.tools.values() if o is not tp and o.port and o.alive()}
        if self.hub_port:
            avoid.add(self.hub_port)
        port = find_free_port(int(settings["ports"][tp.tool_id]), avoid)
        try:
            spec = tp.adapter.launch_spec(port, settings)
        except StageError as e:
            raise ToolError(str(e)) from e
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"{tp.tool_id}.log"
        try:
            if log_path.stat().st_size > MAX_LOG_BYTES:
                os.replace(log_path, log_path.with_name(log_path.name + ".1"))
        except OSError:
            pass
        log_file = open(log_path, "a", encoding="utf-8", errors="replace")
        header = f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} starting {tp.name} on port {port} ===="
        log_file.write(f"\n{header}\n{subprocess.list2cmdline(spec.argv)}\n")
        log_file.flush()
        tp.add_line(header)
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | (0 if self.console else subprocess.CREATE_NO_WINDOW)
        try:
            proc = spawn_in_job(spec.argv, spec.cwd, spec.env, flags)
        except OSError as e:
            log_file.close()
            raise ToolError(f"{tp.name} could not be started ({e}). Close Spellcast3D and run "
                            f"run-spellcast3d.bat again to repair the installation.", 503) from e
        assert proc.stdout is not None
        tp.proc, tp.port, tp.gpu_claimed = proc, port, False
        _OutputPump(tp, proc.stdout, log_file).start()
        tp.monitor_task = asyncio.create_task(self._monitor(tp, proc), name=f"monitor-{tp.tool_id}")
        log.info("Started %s (pid %s) on port %s", tp.name, proc.pid, port)
        return float(spec.ready_timeout)

    async def _wait_ready(self, tp: ToolProcess, ready_timeout: float) -> None:
        proc = tp.proc
        deadline = time.monotonic() + ready_timeout
        while tp.proc is proc and tp.state == STARTING:
            try:
                ok = await asyncio.wait_for(tp.adapter.health(self.client, tp.base_url), HEALTH_TIMEOUT)
            except asyncio.CancelledError:
                raise
            except Exception:
                ok = False
            if ok and tp.proc is proc and tp.state == STARTING and tp.alive():
                tp.state, tp.message = READY, "Ready"
                log.info("%s is ready at %s", tp.name, tp.base_url)
                return
            if time.monotonic() > deadline:
                minutes = max(1, round(ready_timeout / 60))
                await self.stop(tp.tool_id)
                self._fail(tp, f"{tp.name} did not become ready within {minutes} minute(s), so it was stopped. "
                               f"Check its log for what went wrong, then try again: {RESTART_HOW}")
                return
            await asyncio.sleep(HEALTH_INTERVAL)

    async def _monitor(self, tp: ToolProcess, proc: subprocess.Popen[bytes]) -> None:
        """Turn an unexpected exit into the error state, with the last log lines."""
        while tp.proc is proc:
            code = proc.poll()
            if code is not None:
                if tp.state in (STOPPING, STOPPED):
                    return
                await asyncio.sleep(0.3)   # let the output pump catch the final lines
                words = tp.last_words()
                when = "while starting" if tp.state == STARTING else "unexpectedly"
                self._fail(tp, f"{tp.name} stopped {when} (exit code {_exit_code_text(code)})."
                               + (f" Last log lines: {words}." if words else "")
                               + f" Open its log for details, then try again: {RESTART_HOW}")
                return
            await asyncio.sleep(EXIT_POLL_INTERVAL)

    def _fail(self, tp: ToolProcess, message: str) -> None:
        tp.state, tp.message = ERROR, message
        log.error("%s: %s", tp.tool_id, message)
        tp.add_line(f"[Spellcast3D] {message}")

    # ---- stop -------------------------------------------------------------------------------
    def request_stop(self, tool_id: str) -> asyncio.Task[None]:
        """Begin stopping a tool (non-blocking). Concurrent requests share one stop task.

        An unfinished start is cancelled, unless the caller IS that start (ready timeout).
        """
        tp = self.get(tool_id)
        if tp.stop_task is None or tp.stop_task.done():
            start_task = tp.start_task
            if start_task and not start_task.done() and start_task is not asyncio.current_task():
                start_task.cancel()
            call_on_stop = tp.state == READY
            if tp.alive():
                tp.state, tp.message = STOPPING, f"Stopping {tp.name}…"
            tp.stop_task = asyncio.create_task(self._terminate(tp, call_on_stop), name=f"stop-{tool_id}")
        return tp.stop_task

    async def stop(self, tool_id: str) -> ToolProcess:
        """Stop a tool (graceful first, then kill the tree) and wait until it is gone."""
        await asyncio.shield(self.request_stop(tool_id))
        return self.get(tool_id)

    async def stop_all(self) -> None:
        await asyncio.gather(*(self.stop(tid) for tid in self.tools), return_exceptions=True)
        for tp in self.tools.values():
            if tp.monitor_task and not tp.monitor_task.done():
                tp.monitor_task.cancel()

    async def _terminate(self, tp: ToolProcess, call_on_stop: bool) -> None:
        proc = tp.proc
        if proc is None or proc.poll() is not None:
            tp.proc, tp.state, tp.message, tp.gpu_claimed = None, STOPPED, "", False
            return
        tp.state, tp.message = STOPPING, f"Stopping {tp.name}…"
        log.info("Stopping %s (pid %s)", tp.name, proc.pid)
        if call_on_stop:
            try:
                await asyncio.wait_for(tp.adapter.on_stop(self.client, tp.base_url), ON_STOP_TIMEOUT)
            except Exception as e:
                log.debug("%s on_stop failed: %s", tp.tool_id, e)
        if self.console and proc.poll() is None:
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            except OSError as e:
                log.debug("CTRL_BREAK to %s failed: %s", tp.tool_id, e)
            await _wait_exit(proc, BREAK_WAIT)
        if proc.poll() is None:
            log.info("%s did not exit by itself; killing its process tree", tp.name)
            await asyncio.to_thread(kill_tree, proc.pid)
            await _wait_exit(proc, KILL_WAIT)
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
            await _wait_exit(proc, KILL_WAIT)
        if proc.poll() is None:
            self._fail(tp, f"{tp.name} could not be stopped. Close Spellcast3D (this ends every tool) "
                           f"and start it again.")
            return
        # Gone, so the card is free: the next job claims it afresh (until then the stopping tool still held it).
        tp.proc, tp.state, tp.message, tp.gpu_claimed = None, STOPPED, "", False
        tp.add_line(f"[Spellcast3D] {tp.name} stopped.")
        log.info("%s stopped", tp.name)


async def _wait_exit(proc: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while proc.poll() is None and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
