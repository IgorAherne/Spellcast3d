# From AURA-V5 src/world_model/utils/profiling_harness.py (commit 41007db), adapted for Spellcast3D:
#   - torch, torch.profiler and pyinstrument are imported inside the functions that use them (module level is stdlib
#     only: the Qwen launcher's venv has neither unless a [profiling] switch made the installer add them, and torch
#     takes ~35 s to import cold); "from __future__ import annotations" keeps the Profiler type hints lazy;
#   - a module logger (log = logging.getLogger(__name__)) instead of the root logging.* calls, so request_profiler.py
#     can route the lines into the tool's log with a "[profiling] " prefix;
#   - imports point at spellcast.profiling.*;
#   - stop_python_profiler renders the JSON with every frame kept (processor_options filter_threshold 0, same
#     signature): pyinstrument's default drops each frame under 1% of the profile (~1.5 s of a Qwen-Image request:
#     saving the pictures, the progress callbacks) WITHOUT giving its time back, so process_profile_json's SELF (time
#     minus the kept children) showed it as the parent's own time. Not show_all: that would also keep importlib's
#     <frozen ...> frames, which the summary counts as user code; those are still replaced by their children;
#   - dropped as AURA-trainer specific: phase_profiler (its CALIB_PHASES calibrations) and its _noop_step,
#     torch_profiler_wait_after and PROFILER_SETTLE_STEPS (a training loop's optimizer-step schedule).
#   Kept as they are (but for stop_python_profiler's renderer options, above): start_python_profiler /
#   stop_python_profiler (request_profiler.py uses them for each Qwen-Image request), torch_profiler_context
#   (scheduled, for the torch-based tools' step loops: Pixal3D, LATO.2, TriFlow; it reads AURA's torch_profiler_wait /
#   _warmup / _active / _repeat keys), _echo_summary.
"""Shared profiling harness for AURA single-process trainers (Stage-1 encoders, Stage-2 WM).

Three config-gated instruments (enable ONE at a time):
  - use_python_profiler          -> PyInstrument: profile_raw.json + performance_summary.txt
  - use_torch_profiler           -> torch.profiler: trace_{step}_{ts}.json + trace_summary_{step}_{ts}.txt
                                    (GPU ms per module: python -m spellcast.profiling.trace_module_share)
  - diagnostic_enable_sync_hunter-> sync_hunter (wired at call site): sync_report.txt

All artifacts land in the caller's log_dir (runs/<run_name>). Only one trainer runs at a
time, so filenames are deliberately identical across trainers.
The Ray actors (main_train) keep their own AuraProfiler wrapper: multiple concurrent
processes need per-actor filenames, which this single-process harness does not provide.
"""
from __future__ import annotations

import os
import contextlib
import datetime
import logging
from typing import TYPE_CHECKING, Dict, Any, Optional

from spellcast.profiling.profiler_python_utils import process_profile_json, process_torch_trace_json

if TYPE_CHECKING:
    from pyinstrument import Profiler

log = logging.getLogger(__name__)

PROFILE_ECHO_LINES = 30
# [SPELLCAST3D] pyinstrument's remove_irrelevant_nodes threshold: 0 keeps every frame (see the header).
PYINSTRUMENT_OPTIONS: Dict[str, Any] = {"filter_threshold": 0}


def _echo_summary(path: str, lines: int = PROFILE_ECHO_LINES) -> None:
    """Mirrors the head of a summary into the process log. On a cloud pod ONLY the console log
    is shipped home (podlog/<run>/<boot>.log); profiler artifacts stay on the box and are lost
    when the instance is deleted. A summary nobody echoes is a summary nobody reads."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            head = [next(f, '') for _ in range(lines)]
    except Exception as e:
        log.error(f"Could not echo profiler summary {path}: {e}")
        return
    log.info(f">>> PROFILE SUMMARY HEAD ({os.path.basename(path)}):\n" + "".join(head).rstrip())


def start_python_profiler(config: Dict[str, Any]) -> Optional[Profiler]:
    """PyInstrument, full-session capture. Pair with stop_python_profiler in `finally`."""
    if config['use_python_profiler']:
        from pyinstrument import Profiler

        py_profiler = Profiler(interval=0.001)
        py_profiler.start()
        log.info(">>> PYINSTRUMENT: Started capturing Python performance...")
        return py_profiler
    return None


def stop_python_profiler(py_profiler: Optional[Profiler], log_dir: str) -> None:
    if py_profiler and py_profiler.is_running:
        from pyinstrument.renderers import JSONRenderer

        py_profiler.stop()
        log.info(">>> PYINSTRUMENT: Stopped capturing.")
        json_path = os.path.join(log_dir, "profile_raw.json")
        try:
            with open(json_path, "w", encoding='utf-8') as f:
                f.write(py_profiler.output(renderer=JSONRenderer(processor_options=dict(PYINSTRUMENT_OPTIONS))))
            log.info(f">>> PYINSTRUMENT: Raw JSON saved to {json_path}")
            summary_path = os.path.join(log_dir, "performance_summary.txt")
            process_profile_json(json_path, summary_path)
            _echo_summary(summary_path)
        except Exception as e:
            log.error(f"Failed to save profile: {e}")


def torch_profiler_context(config: Dict[str, Any], log_dir: str):
    """torch.profiler context (or nullcontext when disabled). The caller must invoke
    .step() once per optimizer step or the wait/warmup/active schedule never advances
    and no trace is emitted."""
    use_torch = config['use_torch_profiler']
    use_python = config['use_python_profiler']

    if use_torch and use_python:
        log.warning(">>> CONFIG CONFLICT: Both profilers enabled. Disabling Torch to prevent crash.")
        return contextlib.nullcontext()

    if use_torch:
        import torch
        from torch.profiler import profile, ProfilerActivity

        wait_steps = config['torch_profiler_wait']
        warmup_steps = config['torch_profiler_warmup']
        active_steps = config['torch_profiler_active']
        repeat_cycles = config['torch_profiler_repeat']

        trigger_step = wait_steps + warmup_steps + 1
        log.info(f">>> TORCH PROFILER: Enabled. Recording will trigger at Step {trigger_step} "
                 f"(Wait={wait_steps}, Warmup={warmup_steps}, Active={active_steps})")

        def trace_handler(p):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_path = os.path.join(log_dir, f"trace_{p.step_num}_{timestamp}.json")
            log.info(f"Profiler: Exporting raw trace to {trace_path}...")
            p.export_chrome_trace(trace_path)
            summary_path = os.path.join(log_dir, f"trace_summary_{p.step_num}_{timestamp}.txt")
            try:
                process_torch_trace_json(trace_path, output_txt_path=summary_path)
                log.info(f">>> TORCH PROFILER: Summary saved to {summary_path}")
            except Exception as e:
                log.error(f"Failed to process torch trace: {e}")

        return profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=wait_steps, warmup=warmup_steps,
                active=active_steps, repeat=repeat_cycles),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
    return contextlib.nullcontext()
