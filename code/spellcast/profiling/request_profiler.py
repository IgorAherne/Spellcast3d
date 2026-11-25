"""One profiled request of a tool server (Qwen-Image 2.1's picture requests: qwen_profiled.py attaches it to the
unchanged launchers/qwen_server.py from outside), with the [profiling] switches of spellcast3d-config.ini named as
AURA-V5's training config names them.

    profiler = RequestProfiler.begin(config.profiling(), "qwen", info)     # on the request's own thread
    profiler.mark("prompt"); profiler.mark("drawing 1/1", steps=20); profiler.step(1, 20); ...
    profiler.note("seconds", 120.4)
    profiler.finish("ok")                                                   # never raises

The switches (enable ONE at a time, as in AURA):
  use_python_profiler            PyInstrument on the request's thread (profiling_harness.start/stop_python_profiler):
                                 profile_raw.json + performance_summary.txt - the launcher's own Python around the
                                 ctypes call into stable-diffusion.cpp.
  use_torch_profiler             torch.profiler with CPU + CUDA activities, record_shapes, no schedule, no Python
                                 stacks (all of Qwen-Image's work is stable-diffusion.cpp's C++: with_stack only added
                                 wrapper and event-loop rows at the top of AURA's summary, and ~40% to the trace):
                                 the whole request (begin..finish). CUPTI records stable-diffusion.cpp's kernels
                                 although PyTorch launches none -> trace.json + trace_summary.txt (AURA's
                                 process_torch_trace_json, then cuda_trace_report's phase and kernel-class tables).
  diagnostic_enable_sync_hunter  the CUDA runtime's blocking calls and the GPU's idle gaps (cuda_trace_report's
                                 sync_report; torch's sync debug mode sees nothing here, see sync_hunter.py) ->
                                 sync_report.txt, from torch's trace when use_torch_profiler runs too, else from a lean
                                 CUPTI recording of its own (no shapes, no stacks) kept as sync_trace.json.
python + torch both on: torch is skipped with a warning (AURA's rule, one at a time: there PyInstrument and
with_stack's Python tracer conflict; here PyInstrument's sampling would also slow the request the trace times).
sync runs beside either: its lean recording has no Python tracer.

The request's parts are record_function annotations on its thread (they appear as user_annotation events, and as
gpu_user_annotation around the kernels they launched - only when the CPU activity is recorded):
<label>:request around it all, <label>:<phase> (mark) one after the other, <label>:step k/n (mark with steps, then
step) inside a drawing phase. Without torch (python alone) there are none; the wall-clock phases and steps go into
request.txt either way.

Reports: <root>/<YYYYmmdd-HHMMSS>-<label>[-N]/ (root: code/logs/profiling) with request.txt (which profilers ran,
which were skipped and why, the server's info, the request's notes, the phases' wall-clock times, the files) and the
profilers' files. The folder is made at begin(), with a request.txt that says the request is still running: one left
like that means the process ended inside the request.

Stdlib only at module level: torch (~35 s to import cold - preload() does it in the background at start) and
pyinstrument are imported only when a switch needs them, and a request imports them only after a preload that still
runs is done (wait_for_preload: two threads importing torch at once both fail). With every switch off, or every
switched-on profiler unavailable, begin() returns an inactive profiler whose methods do nothing: no folder, no import.
mark / step / note from another thread than begin()'s are ignored (sd.cpp's callbacks run on the request's thread).
Any failure inside the profiling is one log line and the request goes on.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import logging
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

PYTHON = "use_python_profiler"
TORCH = "use_torch_profiler"
SYNC = "diagnostic_enable_sync_hunter"
SWITCHES = (PYTHON, TORCH, SYNC)
PACKAGES = {PYTHON: "pyinstrument", TORCH: "torch", SYNC: "torch"}
PREFIX = "[profiling] "
REQUEST = "request"                      # <label>:request, the root annotation (cuda_trace_report's window)
# torch.profiler's activities, by name (a test may record the CPU alone). CPU is needed for the annotations.
TORCH_ACTIVITIES: tuple[str, ...] = ("CPU", "CUDA")
LOGGER = "spellcast.profiling"           # the copied modules' loggers are its children
CAVEAT = ("Wall times include the profilers' own overhead (CUPTI records every kernel launch, PyInstrument samples "
          "every millisecond): compare parts of this request with each other, and "
          "the totals with an unprofiled picture's seconds.")
FIX_MISSING = ("close Spellcast3D and start it again: the installer adds it to this tool's Python environment while "
               "the switch is on")
PRELOAD_WAIT = 600.0                     # s a request waits for preload() (a cold torch beside Qwen's read-ahead)

_said: set[str] = set()                  # skip reasons already logged by this process
_current: RequestProfiler | None = None  # the active profiler, so a stale one is finished before the next begins
_preloading: threading.Thread | None = None   # preload()'s import thread (the latest), see wait_for_preload


def _print(line: str) -> None:
    print(line, flush=True)


def _available(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False


def missing_packages(switches: Mapping[str, bool]) -> list[str]:
    """The packages the switched-on profilers need and this Python lacks: ["pyinstrument"], ["torch"], both, or none.
    find_spec only: nothing is imported."""
    wanted = [PACKAGES[key] for key in SWITCHES if switches.get(key)]
    return [package for package in dict.fromkeys(wanted) if not _available(package)]


def plan(switches: Mapping[str, bool]) -> tuple[dict[str, bool], dict[str, str]]:
    """({switch: runs}, {switch: why it is skipped}) for the switched-on ones."""
    runs = {key: False for key in SWITCHES}
    skipped: dict[str, str] = {}
    if switches.get(PYTHON):
        if _available("pyinstrument"):
            runs[PYTHON] = True
        else:
            skipped[PYTHON] = f"pyinstrument is not installed here ({FIX_MISSING})"
    for key in (TORCH, SYNC):
        if not switches.get(key):
            continue
        if not _available("torch"):
            skipped[key] = f"PyTorch is not installed here ({FIX_MISSING})"
        elif key == TORCH and runs[PYTHON]:
            skipped[key] = ("use_python_profiler is on too: PyInstrument's sampling would slow down the request the "
                            "trace times (AURA's rule: one at a time); switch one of them off")
        else:
            runs[key] = True
    return runs, skipped


def preload(switches: Mapping[str, bool], log: Callable[[str], None] = _print) -> None:
    """Imports what the switched-on profilers will use in a daemon thread, at the server's start: torch takes ~35 s
    cold, which the first profiled request would otherwise wait for. A request that comes while this still runs
    waits for it (wait_for_preload) before importing anything itself."""
    global _preloading
    runs, _ = plan(switches)
    modules = (["pyinstrument", "pyinstrument.renderers"] if runs[PYTHON] else []) + \
        (["torch", "torch.profiler", "torch.autograd.profiler"] if runs[TORCH] or runs[SYNC] else [])
    if not modules:
        return
    previous = _preloading

    def work() -> None:
        if previous is not None:                   # a second preload: never two importing threads
            previous.join()
        started = time.monotonic()
        try:
            for name in modules:
                __import__(name)
            log(f"{PREFIX}{', '.join(m for m in modules if '.' not in m)} imported in "
                f"{time.monotonic() - started:.1f} s, ready for the first request.")
        except Exception as exc:                   # noqa: BLE001 - the request reports it again, plainly
            log(f"{PREFIX}Importing the profilers failed ({type(exc).__name__}: {exc}).")

    thread = threading.Thread(target=work, name="profiling-preload", daemon=True)
    thread.start()
    _preloading = thread


def wait_for_preload(timeout: float | None = None) -> float:
    """Waits until preload()'s imports are done; returns the seconds waited (0.0 when none runs). Every import of the
    profilers' packages on a request's path comes after this. Python does not make a thread that imports a SUBMODULE
    of a package another thread is still initialising wait for the package: `from torch.autograd.profiler import ...`
    finds the half-made torch in sys.modules and loads torch.autograd itself; when torch's own __init__ then imports
    torch.autograd, CPython breaks that deadlock by handing it the half-loaded module, and both imports fail - torch
    cannot be imported again in that process (CPU-only reproduction: a request 0.02-0.3 s after preload, every time).
    TimeoutError after `timeout` s (PRELOAD_WAIT by default); begin() then leaves that request unprofiled, without
    importing anything."""
    thread = _preloading
    if thread is None or thread is threading.current_thread() or not thread.is_alive():
        return 0.0
    started = time.monotonic()
    thread.join(PRELOAD_WAIT if timeout is None else timeout)
    if thread.is_alive():
        raise TimeoutError(f"the profilers' background import (preload) has not finished in "
                           f"{time.monotonic() - started:.0f} s")
    return time.monotonic() - started


class _LogHandler(logging.Handler):
    """The copied modules' log records -> the profiler's log callable, every line prefixed."""

    def __init__(self, log: Callable[[str], None]) -> None:
        super().__init__(logging.INFO)
        self._log = log

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = record.getMessage()
            if record.levelno >= logging.WARNING:
                text = f"{record.levelname}: {text}"
            for line in text.splitlines() or [""]:
                self._log(PREFIX + line)
        except Exception:                          # noqa: BLE001 - logging must never break a request
            pass


class RequestProfiler:
    """See the module's docstring. Built by begin()."""

    def __init__(self, label: str, log: Callable[[str], None]) -> None:
        self.label = label
        self.active = False
        self.folder: Path | None = None
        self._log = log
        self._thread = threading.get_ident()
        self._runs: dict[str, bool] = {key: False for key in SWITCHES}
        self._skipped: dict[str, str] = {}
        self._info: dict[str, Any] = {}
        self._notes: dict[str, Any] = {}
        self._stack: list[tuple[str, Any]] = []   # (kind, record_function), innermost last
        self._record_function: Any = None        # torch.autograd.profiler.record_function while torch records
        self._torch: Any = None                   # the running torch.profiler.profile
        self._python: Any = None                  # the running pyinstrument Profiler
        self._handler: _LogHandler | None = None
        self._logger_state: tuple[int, bool] | None = None
        self._started_at = datetime.datetime.now()
        self._t0 = time.perf_counter()
        self._timeline: list[tuple[str, float, int | None]] = []     # (phase, start, steps)
        self._step_marks: list[tuple[str, int, float]] = []          # (phase, step done, time)
        self._finished = False

    # -- building
    @classmethod
    def begin(cls, switches: Mapping[str, bool], label: str, info: dict[str, Any], root: Path | None = None,
              log: Callable[[str], None] = _print) -> RequestProfiler:
        global _current
        profiler = cls(label, log)
        if not any(switches.get(key) for key in SWITCHES):
            return profiler
        try:
            if _current is not None and _current.active and not _current._finished:
                _current._say("The previous request's profiling was never finished: finishing it now.")
                _current.finish("abandoned (never finished)")
            runs, skipped = plan(switches)
            profiler._runs, profiler._skipped = runs, skipped
            for key, why in skipped.items():
                if f"{key}: {why}" not in _said:
                    _said.add(f"{key}: {why}")
                    profiler._say(f"{key} skipped: {why}.")
            if not any(runs.values()):
                return profiler
            if root is None:
                from spellcast import paths
                root = paths.LOGS / "profiling"
            profiler.folder = _new_folder(Path(root), label, profiler._started_at)
            profiler._info = dict(info)
            profiler.active = True
            _current = profiler
            profiler._write_request_txt(running=True)
            profiler._attach_logging()
            profiler._start()
        except Exception as exc:                   # noqa: BLE001
            profiler._say(f"Profiling could not start ({type(exc).__name__}: {exc}); the request runs unprofiled.")
            profiler._stop_everything()
            profiler.active = False
            profiler._detach_logging()
            if profiler.folder is not None:            # not "STILL RUNNING": nothing was recorded
                try:
                    profiler._write_request_txt(running=False, outcome=f"profiling could not start "
                                                f"({type(exc).__name__}: {exc}); the request ran unprofiled", wall=0.0)
                except Exception:                      # noqa: BLE001
                    pass
        return profiler

    def _start(self) -> None:
        waited = wait_for_preload()                # before any import of torch or pyinstrument (see there)
        if waited >= 0.1:
            self._say(f"Waited {waited:.1f} s for the profilers' background import (preload) to finish.")
        if self._runs[TORCH] or self._runs[SYNC]:
            started = time.perf_counter()
            from torch.autograd.profiler import record_function
            from torch.profiler import ProfilerActivity, profile

            activities = [getattr(ProfilerActivity, name) for name in TORCH_ACTIVITIES]
            full = self._runs[TORCH]
            self._torch = profile(activities=activities, record_shapes=full, with_stack=False, profile_memory=False)
            self._torch.start()
            self._record_function = record_function
            kind = "full: kernels, CUDA calls and shapes" if full else "lean, for the sync report"
            self._say(f"torch.profiler recording ({kind}; started in {time.perf_counter() - started:.1f} s).")
        if self._runs[PYTHON]:
            from spellcast.profiling.profiling_harness import start_python_profiler

            self._python = start_python_profiler({PYTHON: True})
        self._open("request", REQUEST)
        self._t0 = time.perf_counter()

    # -- the request's parts
    def mark(self, phase: str, steps: int | None = None) -> None:
        """A new phase begins: closes the open step and phase, opens <label>:<phase>; with steps, also
        <label>:step 1/<steps> inside it."""
        if not self._mine():
            return
        try:
            now = time.perf_counter()
            self._close_kinds(("step", "phase"))
            self._timeline.append((phase, now, steps))
            self._open("phase", phase)
            if steps:
                self._open("step", f"step 1/{steps}")
        except Exception as exc:                   # noqa: BLE001
            self._fail("mark", exc)

    def step(self, step: int, steps: int) -> None:
        """Step `step` of `steps` is done (sd.cpp says so after it): closes its annotation and opens the next one."""
        if not self._mine() or step <= 0:
            return
        try:
            self._step_marks.append((self._timeline[-1][0] if self._timeline else "", step, time.perf_counter()))
            self._close_kinds(("step",))
            if step < steps:
                self._open("step", f"step {step + 1}/{steps}")
        except Exception as exc:                   # noqa: BLE001
            self._fail("step", exc)

    def note(self, key: str, value: Any) -> None:
        """A line for request.txt (the request's settings, its result)."""
        if self._mine():
            self._notes[str(key)] = value

    def finish(self, outcome: str = "ok") -> None:
        """Closes the annotations, stops the profilers, writes the reports and echoes their heads into the log. Never
        raises."""
        global _current
        if not self.active or self._finished:
            return
        self._finished = True
        wall = time.perf_counter() - self._t0
        timings: dict[str, float] = {}
        try:
            if threading.get_ident() != self._thread:
                self._say("finish() came from another thread than begin(): the annotations are left open.")
                self._stack.clear()
            self._close_kinds(("step", "phase", "request"))
            self._timeline.append(("(end)", time.perf_counter(), None))
            if self._python is not None:
                from spellcast.profiling.profiling_harness import stop_python_profiler

                started = time.perf_counter()
                python, self._python = self._python, None
                stop_python_profiler(python, str(self.folder))
                timings["python profile written"] = time.perf_counter() - started
            if self._torch is not None:
                self._torch_reports(timings)
        except Exception as exc:                   # noqa: BLE001
            self._fail("finish", exc)
        finally:
            self._stop_everything()
            try:
                self._write_request_txt(running=False, outcome=outcome, wall=wall, timings=timings)
            except Exception as exc:               # noqa: BLE001
                self._fail("request.txt", exc)
            self._say(f"{self.label} request profiled ({outcome}, {wall:.1f} s); reports in {self.folder} "
                      f"(written in {sum(timings.values()):.1f} s).")
            self._detach_logging()
            if _current is self:
                _current = None

    # -- torch's reports
    def _torch_reports(self, timings: dict[str, float]) -> None:
        from spellcast.profiling import cuda_trace_report as report
        from spellcast.profiling.profiler_python_utils import process_torch_trace_json
        from spellcast.profiling.profiling_harness import _echo_summary

        started = time.perf_counter()
        torch_profile, self._torch = self._torch, None
        torch_profile.stop()
        timings["torch.profiler stopped"] = time.perf_counter() - started
        assert self.folder is not None
        trace = self.folder / ("trace.json" if self._runs[TORCH] else "sync_trace.json")
        started = time.perf_counter()
        torch_profile.export_chrome_trace(str(trace))
        del torch_profile
        timings["trace exported"] = time.perf_counter() - started
        started = time.perf_counter()
        summary = self.folder / "trace_summary.txt"
        if self._runs[TORCH]:
            process_torch_trace_json(str(trace), str(summary))
        analysis = report.TraceAnalysis(report.load_trace(trace), f"{self.label}:")
        if self._runs[TORCH]:
            with open(summary, "a", encoding="utf-8") as f:
                f.write("\n" + report.phase_report(analysis) + "\n" + report.kernel_class_report(analysis))
            _echo_summary(str(summary))
        if self._runs[SYNC]:
            sync = self.folder / "sync_report.txt"
            sync.write_text(report.sync_report(analysis), encoding="utf-8")
            _echo_summary(str(sync))
        share = 100 * analysis.busy_total / analysis.wall if analysis.wall else 0.0
        waiting = (f", waiting for host memory (pageable copies) {analysis.waiting_total / 1e6:.1f} s"
                   if analysis.waiting_total >= 50_000 else "")
        steps = (f"; {len(analysis.steps)} steps, median {statistics.median(s.wall for s in analysis.steps) / 1e3:.0f} "
                 f"ms" if analysis.steps else "")
        self._say(f"GPU busy {analysis.busy_total / 1e6:.1f} s of the request's {analysis.wall / 1e6:.1f} s "
                  f"({share:.0f}%){waiting}{steps}.")
        timings["reports written"] = time.perf_counter() - started

    # -- annotations
    def _open(self, kind: str, name: str) -> None:
        if self._record_function is None:
            return
        annotation = self._record_function(f"{self.label}:{name}")
        annotation.__enter__()
        self._stack.append((kind, annotation))

    def _close_kinds(self, kinds: tuple[str, ...]) -> None:
        """Closes the innermost annotations while they are of these kinds (in reverse order of opening)."""
        while self._stack and self._stack[-1][0] in kinds:
            _kind, annotation = self._stack.pop()
            annotation.__exit__(None, None, None)

    # -- helpers
    def _mine(self) -> bool:
        return self.active and not self._finished and threading.get_ident() == self._thread

    def _say(self, text: str) -> None:
        try:
            self._log(PREFIX + text)
        except Exception:                          # noqa: BLE001
            pass

    def _fail(self, where: str, exc: BaseException) -> None:
        self._say(f"{where} failed ({type(exc).__name__}: {exc}); the request goes on.")

    def _stop_everything(self) -> None:
        """After a failure: whatever still runs is stopped without reports."""
        for stop in (self._stop_python_quietly, self._stop_torch_quietly):
            try:
                stop()
            except Exception:                      # noqa: BLE001
                pass
        self._stack.clear()

    def _stop_python_quietly(self) -> None:
        python, self._python = self._python, None
        if python is not None and python.is_running:
            python.stop()

    def _stop_torch_quietly(self) -> None:
        torch_profile, self._torch = self._torch, None
        if torch_profile is not None:
            torch_profile.stop()

    def _attach_logging(self) -> None:
        logger = logging.getLogger(LOGGER)
        self._logger_state = (logger.level, logger.propagate)
        self._handler = _LogHandler(self._log)
        logger.addHandler(self._handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False                   # the launcher configures no logging: no second copy on stderr

    def _detach_logging(self) -> None:
        if self._handler is None:
            return
        logger = logging.getLogger(LOGGER)
        logger.removeHandler(self._handler)
        if self._logger_state is not None:
            logger.setLevel(self._logger_state[0])
            logger.propagate = self._logger_state[1]
        self._handler = None

    # -- request.txt
    def _write_request_txt(self, running: bool, outcome: str = "", wall: float | None = None,
                           timings: dict[str, float] | None = None) -> None:
        assert self.folder is not None
        lines = [f"Spellcast3D profiling: one {self.label} request",
                 f"started {self._started_at:%Y-%m-%d %H:%M:%S}"]
        if running:
            lines.append("STILL RUNNING - if this stays so, the process ended inside the request (see the tool's log).")
        else:
            lines.append(f"outcome: {outcome}")
            lines.append(f"wall: {wall:.2f} s from begin to finish (the reports' writing not included)")
        lines.append(CAVEAT)
        lines.append("")
        lines.append("profilers")
        for key in SWITCHES:
            if self._runs.get(key):
                lines.append(f"  ran: {key} - {_WHAT[key]}")
            elif key in self._skipped:
                lines.append(f"  skipped: {key} - {self._skipped[key]}")
        if self._info:
            lines += ["", "server"] + [f"  {k}: {_text(v)}" for k, v in self._info.items()]
        if self._notes:
            lines += ["", "request"] + [f"  {k}: {_text(v)}" for k, v in self._notes.items()]
        phases = self._phase_lines()
        if phases:
            lines += ["", "phases (wall clock, from the server's progress)"] + phases
        if not running:
            lines += ["", "files"] + [f"  {name} - {what}" for name, what in _FILES.items()
                                      if (self.folder / name).is_file()]
            if timings:
                lines += ["", "writing the reports: " + ", ".join(f"{k} {v:.1f} s" for k, v in timings.items())]
        (self.folder / "request.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _phase_lines(self) -> list[str]:
        out = []
        for (phase, start, steps), (_next, end, _) in zip(self._timeline, self._timeline[1:]):
            line = f"  {phase}: {end - start:.2f} s"
            if steps:
                marks = [t for p, _s, t in self._step_marks if p == phase and start <= t <= end]
                durations = [b - a for a, b in zip([start] + marks, marks)]
                if durations:
                    line += f" - {len(durations)} of {steps} steps: the first {durations[0]:.2f} s"
                    if len(durations) > 1:
                        rest = durations[1:]
                        line += (f", then median {statistics.median(rest):.2f} s (min {min(rest):.2f}, "
                                 f"max {max(rest):.2f})")
            out.append(line)
        return out


_WHAT = {
    PYTHON: "PyInstrument on the request's thread (the launcher's Python around stable-diffusion.cpp)",
    TORCH: "torch.profiler, CPU + CUDA activities with shapes (no Python stacks), over the whole request",
    SYNC: "the CUDA runtime's blocking calls and the GPU's idle gaps (cuda_trace_report.sync_report)",
}
_FILES = {
    "profile_raw.json": "PyInstrument's raw profile (JSON); performance_summary.txt is made from it",
    "performance_summary.txt": "the launcher's own functions by time (AURA's summary)",
    "trace.json": "torch.profiler's Chrome trace: open it in https://ui.perfetto.dev; "
                  "python -m spellcast.profiling.trace_module_share trace.json gives GPU ms per step and phase",
    "trace_summary.txt": "kernels by time (AURA's summary), then the phases and steps, then the kernel classes",
    "sync_trace.json": "the lean CUPTI recording the sync report was made from (Perfetto opens it too)",
    "sync_report.txt": "blocking CUDA calls, copies, allocations and the GPU's idle gaps",
}


def _text(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _new_folder(root: Path, label: str, when: datetime.datetime) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = f"{when:%Y%m%d-%H%M%S}-{label}"
    for n in range(1, 1000):
        folder = root / (base if n == 1 else f"{base}-{n}")
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            continue
    raise FileExistsError(f"no free report folder for {base} in {root}")
