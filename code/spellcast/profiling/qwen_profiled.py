"""Qwen-Image 2.1's launcher with the per-request profilers attached FROM OUTSIDE. Runs in the qwen venv:

    python qwen_profiled.py <exactly qwen_server.py's arguments>

adapters/qwen.py starts this script instead of launchers/qwen_server.py while a [profiling] switch of
spellcast3d-config.ini is on when Qwen-Image starts (config.profiling()); with every switch off the hub starts the
server itself and nothing here runs. Why a script of its own: the profiling harness stays out of the server's code
(qwen_server.py has not one line of it, as AURA-V5's trainer keeps its harness in files of their own). This imports
qwen_server as a module, wraps a few of its functions and methods, then runs qwen_server.main(): the same server,
the same routes, the same answers.

What it wraps and why (the original is always called; a hook that fails says so once in the log and does nothing
more, so nothing ever raises into the server or into stable-diffusion.cpp's callbacks):
- build_app(engine, gpu, progress, projects_root, info, ...): keeps `info` (card, placement, model size, sd.cpp
  build) and the engine (its placement details and load time) for each report's request.txt.
- GpuWorker.run(fn, *args): the picture request is the closure named "work" that build_app's run() hands to the GPU
  thread. It runs inside one RequestProfiler (request_profiler.py), begun and finished ON the GPU thread: pyinstrument
  and torch.profiler record the thread that starts them, and sd.cpp calls its log and progress callbacks on that
  thread. The switches are read from the config file at every request, so switching one off while Qwen-Image runs
  stops the profiling from the next picture. The reports are written after work() returned, so the answer's
  "seconds" (timed inside work()) never include them. Anything else (the CUDA check after a failure) passes through.
- QwenEngine.generate(req, seed, references[, init, mask]): the request's settings, for request.txt.
- Progress._enter / on_log / on_step: the server already turns sd.cpp's log lines and progress callbacks into the
  phases of its progress bar; these follow them to name the parts of the profile: "references" (the example
  pictures), "prompt" (the text encoder), "prepare" (after it, until drawing starts: the diffusion model's weights
  staged onto the card), "drawing i/n" with "step k/n" inside it, "finishing" (the VAE decode), "saving".

qwen_server.py may change: main() checks every name it wraps, and the Progress fields the hooks read, before it
wraps anything. One gone -> a loud line in the log, and the server runs unprofiled rather than not at all.
tests/test_qwen_profiled.py checks the same names on the current qwen_server.py, so a rename breaks a test first.
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
LAUNCHERS = CODE / "spellcast" / "launchers"
# As qwen_server.py has it when started itself: its folder first (it imports its siblings as top-level modules), code/
# at the end (the venv's own packages win). Not this folder (sys.path[0] when started as a script): its modules
# belong to the spellcast package, imported once under that name.
if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == HERE:
    del sys.path[0]
if str(LAUNCHERS) not in sys.path:
    sys.path.insert(0, str(LAUNCHERS))
if str(CODE) not in sys.path:
    sys.path.append(str(CODE))

import qwen_server as server  # noqa: E402
from spellcast import config, paths  # noqa: E402
from spellcast.profiling import request_profiler  # noqa: E402

LABEL = "qwen"                     # the report folders' <time>-qwen and the annotations' "qwen:..."
WORK = "work"                      # the name of build_app's per-request closure, which GpuWorker.run runs
# What main() wraps (checked first): qwen_server's names, and the Progress fields the hooks read.
TARGETS: tuple[str, ...] = ("main", "build_app", "GpuWorker.run", "QwenEngine.generate", "Progress._enter",
                            "Progress.on_log", "Progress.on_step")
PROGRESS_FIELDS: tuple[str, ...] = ("phase", "image", "images", "steps")
PREPARED = "get_learned_condition completed"    # sd.cpp's line when the text encoder is done
NOTED = ("model", "width", "height", "batch_count", "transparent")     # the request's own
# ... and how the model is drawn (qwen_server.sampling: the Turbo's and FLUX.3's own recipes whatever the request
# says, the First-Block Cache as the artist or the card decided), not what the request asked for
DRAWN = ("steps", "cfg_scale", "sampler", "first_block_cache")

current: Any = None                # the RequestProfiler of the picture request running on the GPU thread, else None
_seen: dict[str, Any] = {}         # what build_app was given: "info", "engine"
_request: dict[str, Any] = {}      # this request's last progress phase ("key") and whether "prepare" was marked
_failed: set[str] = set()          # the hooks that failed (each said once)


def say(text: str) -> None:
    print(text if text.startswith("[") else f"[profiling] {text}", flush=True)


def _safely(name: str, hook: Callable[..., Any], *args: Any) -> Any:
    """hook(*args), or None after one line in the log (the first time this hook fails)."""
    try:
        return hook(*args)
    except Exception as exc:
        if name not in _failed:
            _failed.add(name)
            say(f"The profiling's {name} hook failed ({type(exc).__name__}: {exc}); Qwen-Image goes on, that part of "
                f"the profile is missing.")
        return None


def _active() -> Any:
    profiler = current
    return profiler if profiler is not None and getattr(profiler, "active", False) else None


# ---- the checks and the wraps --------------------------------------------------------------------------------------

def missing(srv: Any = server) -> list[str]:
    """The TARGETS and Progress fields that `srv` (qwen_server) lacks, e.g. ["Progress.on_step", "Progress().steps"]."""
    gone = []
    for name in TARGETS:
        owner: Any = srv
        for part in name.split("."):
            owner = getattr(owner, part, None)
        if not callable(owner):
            gone.append(name)
    try:
        probe = srv.Progress()
        gone += [f"Progress().{field}" for field in PROGRESS_FIELDS if not hasattr(probe, field)]
    except Exception as exc:
        gone.append(f"Progress() ({type(exc).__name__}: {exc})")
    return gone


def attach(srv: Any = server) -> None:
    """Wrap srv's TARGETS (missing(srv) must be empty)."""
    build_app, run, generate = srv.build_app, srv.GpuWorker.run, srv.QwenEngine.generate
    enter, on_log, on_step = srv.Progress._enter, srv.Progress.on_log, srv.Progress.on_step

    @functools.wraps(build_app)
    def build_app_seen(engine: Any, gpu: Any, progress: Any, projects_root: Any, info: Any, *args: Any,
                       **kwargs: Any) -> Any:
        _seen.update(engine=engine, info=info)
        return build_app(engine, gpu, progress, projects_root, info, *args, **kwargs)

    @functools.wraps(run)
    def run_profiled(self: Any, fn: Callable[..., Any], /, *args: Any) -> Any:
        return run(self, profiled(fn) if getattr(fn, "__name__", None) == WORK else fn, *args)

    @functools.wraps(generate)
    def generate_noted(self: Any, req: Any, seed: int, references: list[Any], **start: Any) -> Any:
        _safely("request", _note_request, req, seed, references)
        return generate(self, req, seed, references, **start)

    @functools.wraps(enter)
    def enter_marked(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = enter(self, *args, **kwargs)
        _safely("phase", _on_phase, self)
        return result

    @functools.wraps(on_log)
    def on_log_marked(self: Any, level: int, text: str) -> Any:
        result = on_log(self, level, text)
        _safely("log", _on_log, self, text)
        return result

    @functools.wraps(on_step)
    def on_step_marked(self: Any, step: int, steps: int, seconds: float) -> Any:
        result = on_step(self, step, steps, seconds)
        _safely("step", _on_step, self, step)
        return result

    srv.build_app = build_app_seen
    srv.GpuWorker.run = run_profiled
    srv.QwenEngine.generate = generate_noted
    srv.Progress._enter, srv.Progress.on_log, srv.Progress.on_step = enter_marked, on_log_marked, on_step_marked


# ---- one picture request -------------------------------------------------------------------------------------------

def profiled(work: Callable[..., Any]) -> Callable[..., Any]:
    """work() inside a RequestProfiler, both on the thread that runs it (the GPU thread); the reports after it."""
    @functools.wraps(work)
    def profiled_work(*args: Any) -> Any:
        global current
        current = _safely("begin", _begin)
        result = None
        try:
            result = work(*args)
            return result
        finally:
            profiler, current = current, None
            if profiler is not None:
                _safely("report", _finish, profiler, result)
    return profiled_work


def _begin() -> Any:
    _request.clear()
    return request_profiler.RequestProfiler.begin(config.profiling(), LABEL, _request_info(), log=say)


def _request_info() -> dict[str, Any]:
    """build_app's info with the engine's placement details (budget, prefix cache, weights' home) and load time."""
    info = dict(_seen.get("info") or {})
    engine = _seen.get("engine")
    if isinstance(getattr(engine, "placement", None), dict):
        info["placement_details"] = dict(engine.placement)
    if getattr(engine, "load_seconds", None) is not None:
        info["load_seconds"] = engine.load_seconds
    return info


def _finish(profiler: Any, result: Any) -> None:
    """What work() returned - (failure, paths, extra, seconds) - into request.txt, then the reports. The outcome: "ok",
    the failure's kind ("cancelled", "oom", "input", "error"), or "error" when work() raised (it catches everything
    itself)."""
    outcome = "error"
    try:
        if result is not None:
            failure, saved, extra, seconds = result
            if failure is not None:
                answer = json.loads(failure.body)
                outcome = str(answer.get("kind") or "error")
                profiler.note("failure", answer.get("detail"))
            else:
                outcome = "ok"
            profiler.note("pictures", len(saved))
            profiler.note("seconds", round(float(seconds), 2))
            profiler.note("timings", (extra or {}).get("timings"))
    finally:
        profiler.finish(outcome)


def _note_request(req: Any, seed: int, references: list[Any]) -> None:
    profiler = _active()
    if profiler is None:
        return
    for key in NOTED:
        profiler.note(key, getattr(req, key, None))
    engine = _seen.get("engine")
    capability = (getattr(engine, "placement", None) or {}).get("capability", (99, 0))
    plan = server.sampling(req.model, req.steps, req.cfg_scale, req.sampler, req.first_block_cache, capability)
    for key in DRAWN:
        profiler.note(key, plan[key])
    profiler.note("own_schedule", plan["sigmas"] is not None)
    profiler.note("seed", seed)
    profiler.note("references", len(references))
    profiler.note("prompt_characters", len(getattr(req, "prompt", "") or ""))


def _on_phase(progress: Any) -> None:
    """A new progress phase (or, while drawing, the next picture) -> its own part of the profile."""
    profiler = _active()
    if profiler is None:
        return
    drawing = progress.phase == "drawing"
    key = (progress.phase, progress.image if drawing else 0)
    if _request.get("key") == key:
        return
    _request["key"] = key
    if drawing:
        profiler.mark(f"drawing {progress.image}/{progress.images}", steps=progress.steps)
    else:
        profiler.mark(progress.phase)


def _on_log(progress: Any, text: str) -> None:
    profiler = _active()
    if profiler is None or PREPARED not in text or _request.get("prepare") or progress.phase in ("idle", "loading"):
        return
    _request["prepare"] = True
    profiler.mark("prepare")


def _on_step(progress: Any, step: int) -> None:
    """sd.cpp reports step k after it is done; only the drawing's steps (not the vision encoder's or the VAE tiles')."""
    profiler = _active()
    if profiler is not None and progress.phase == "drawing" and step > 0:
        profiler.step(step, progress.steps)


# ---- main ----------------------------------------------------------------------------------------------------------

def main() -> int:
    # No logging setup here: RequestProfiler routes the harness's log lines (logger spellcast.profiling) through `say`
    # into Qwen-Image's log while it profiles a request; a handler of our own on that logger would print each twice.
    gone = missing()
    if gone:
        say(f"PROFILING IS OFF: qwen_server.py no longer has {', '.join(gone)}, which spellcast/profiling/"
            f"qwen_profiled.py attaches to. Qwen-Image runs without profiling; update qwen_profiled.py (its test, "
            f"tests/test_qwen_profiled.py, shows what changed).")
        return server.main()
    attach()
    switches = config.profiling()
    on = [key for key, value in switches.items() if value]
    say(f"Qwen-Image runs with the profilers attached; switched on in {config.config_file().name}: "
        f"{', '.join(on) or 'none'} (read again at every picture). Reports go to {paths.LOGS / 'profiling'}.")
    lacking = request_profiler.missing_packages(switches)
    if lacking:
        say(f"{' and '.join(lacking)} is not installed in Qwen-Image's Python environment, so that profiler cannot "
            f"run. Leave the switch on, close Spellcast3D and start it again: its installer adds it.")
    request_profiler.preload(switches)
    return server.main()


if __name__ == "__main__":
    sys.exit(main())
