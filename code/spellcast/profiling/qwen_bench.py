"""Qwen-Image 2.1 bench: the Images stage's default generator on its own, the same request a few times, for
before/after numbers and for profiles of one picture. For developers; from code\\ with the hub's venv:

    venvs\\hub\\Scripts\\python.exe -m spellcast.profiling.qwen_bench [--profile none|python|torch|sync|torch+sync|
        python+sync] [--runs 2] [--model turbo|full|flux3 ...] [--prompt "..."] [--seed 1234] [--size 1024]
        [--steps N] [--cfg 1]
        [--negative "..."] [--batch 1] [--references a.png ...] [--transparent on|off] [--first-block-cache on|off] [--port 0]
        [--pin auto|on|off] [--prefix-cache auto|f16|q8_0] [--prefix-keep auto|on|off] [--attention auto|flash|sage]
        [--card-gb GB] [--flux3-engine auto|on|off]
        [--prompts "..." "..." ...] [--verbose] [--sdcpp DIR] [--models-dir DIR] [--flux3-dir DIR]

The server is started EXACTLY as the hub starts it, by the adapter itself (adapters/qwen.py QwenAdapter().launch_spec:
the same python, script, arguments, environment and working folder), with two differences, and a third when asked:
- SPELLCAST_CONFIG names a copy of the config file, code\\logs\\profiling\\bench-<time>\\spellcast3d-config.ini: as
  the artist's file says, but the [profiling] switches are set from --profile (with_switches). It is set in this
  process's environment before the launch is built, so the adapter picks the script from those switches - the
  profiled entry (spellcast/profiling/qwen_profiled.py), or the plain server for --profile none - and the server
  reads them too: its reports go to code\\logs\\profiling\\<time>-qwen\\, as from the hub.
- --projects-root is the bench's folder: the pictures land there (run-N\\), never in the artist's projects, and the
  example pictures are copied in (the server reads only inside it).
- --pin, --prefix-cache, --attention and --card-gb, only when given, are added to the server's arguments
  (launcher_switches): the launcher's own switches for the diffusion weights in page-locked RAM, the prefix cache's
  type, the attention (SageAttention or flash attention) and the card's size it plans for (a smaller card emulated:
  --card-gb 8), which the hub leaves to the launcher. For measuring one against the other
  (--pin off / on; --prefix-cache q8_0, what cards of 16 GB and less used with the former recipe; --attention flash,
  the attention before 2026-10-21). With --pin, an SD_PIN_PARAMS set in the bench's own shell is NOT passed on to
  the server (bench_launch): the launcher never changes a set one, so --pin off and --pin on would both run pinned
  as the shell says. Without --pin it goes along, as from the hub, and it decides instead of the launcher's auto.
  Either way the bench says so before it starts (shell_pin_words).
- --verbose adds the launcher's --verbose: sd.cpp's debug lines (the weights' staging times, the compute buffers)
  in bench.log and on the console.
- --sdcpp DIR (developers) starts the server with that engine folder (stable-diffusion.dll and its DLLs: a build under
  test) instead of the installed one: the launch's --sdcpp, its working folder and the head of its PATH
  (bench_launch). The launcher sees a build that is not the pinned one and so chooses flash attention unless
  --attention sage is given too; the installed engine is not touched.
- --models-dir DIR (developers) gives the launch's --models that folder instead of models\\qwen-image (its files
  named as there: a model under test, or a folder of hard links that holds both models at one size); --flux3-dir DIR
  the same for FLUX.3 Image Klein's --flux3-models instead of models\\flux3-klein.
- --flux3-engine auto|on|off, only when given, is the launcher's own switch: on takes the engine build (e.g. a build
  under test with --sdcpp) to draw FLUX.3 although install/sdcpp.py ENGINE_HAS_FLUX3 does not say so yet.
python and torch together leave torch out (the profilers' rule), so there is no such profile.

--model turbo|full|flux3 (default turbo, the hub's default for a new prop) is the request's model (qwen_model): the
Turbo draws its own fixed 8 steps and FLUX.3 Image Klein its own 16 (--steps, --cfg and --first-block-cache are the
full model's and not sent for them; FLUX.3 draws opaque pictures, whatever --transparent says), the full model draws
--steps. Several (--model turbo flux3 turbo) give run N the Nth (the list repeated), as --prompts do: the launcher
holds one model at a time, so a run whose model differs from the run before's first switches - frees the loaded one
and loads the other (the table's "switch" seconds) - and is a new description again (the conditioning cache goes with
the old context). The bench refuses a model whose files are not in its models folder.

It waits until the server is ready ("ready in X s": the loading), and until it has read the model files ahead (it
does that in the background after ready; a run beside it would measure the disk too). Then it sends the same request
--runs times, built as the hub builds it (adapters/qwen.py qwen_request, from a new prop's images settings: the
prompt as typed), with the seeds seed, seed+1, ...: run 1 is a new description (the text encoder runs), runs 2+
are re-rolls (sd.cpp's conditioning cache skips the text encoder, as when the artist presses Generate again). With
--prompts, run N takes the Nth prompt (the list repeated), and a run whose prompt differs from the run before is a
new description again: the text encoder after the diffusion weights, as when the artist describes the next prop
(on a PC with little RAM, where Windows has dropped the one's pages for the other's). The
recipe is the hub's unless told otherwise: Qwen's own, without guidance (--cfg 1: one model pass per step, no negative
prompt) at the hub's default steps (--steps: adapters/qwen.py DEFAULT_STEPS; Qwen's own reference is
--steps 40). Any other --cfg is guidance, which the hub never asks for and the launcher still takes: two passes per
step, with --negative (default: none) as the negative prompt. So a recipe with guidance can be measured next to
Qwen's own, e.g. the former one: --steps 20 --cfg 6 (request_body).

Per run: the answer's seconds, sd.cpp's timings and the profiling report's folder; a table at the end; then
/spellcast/shutdown (ended when it does not stop). The server's output goes to bench.log in the bench's folder and,
prefixed, to this console. Ctrl+C cancels the picture and stops the server; the table shows that run as stopped.
Exit code: 0 when every run was answered without a failure, 1 when one failed, 2 when the bench refused to start,
130 after Ctrl+C (also on the last run).

Refuses to start when the installer has not put a profile's packages into the qwen venv (pyinstrument for python,
PyTorch for torch and sync). Warns when the card already has more than 1.5 GB in use (the placement and the timings
depend on the free memory) and when a Qwen-Image server answers on the hub's Qwen port (it holds RAM and video
memory). stdlib + httpx and the hub's own modules; no torch.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Mapping, Sequence

import httpx

from .. import config, gpu, paths, procman, projects
from ..adapters.base import LaunchSpec, StageError
from ..adapters.qwen import (CFG_SCALE, DEFAULT_MODEL, DEFAULT_STEPS, FLUX3_STEPS, IMAGE_TIMEOUT, QWEN_MODELS,
                             TURBO_STEPS, QwenAdapter, qwen_request)
from ..install import models as model_files
from ..settings import SettingsStore

PROFILING_ROOT: Path = paths.LOGS / "profiling"       # the reports (request_profiler.py) and the benches' folders
PROFILES: dict[str, tuple[str, ...]] = {
    "none": (),
    "python": ("use_python_profiler",),
    "torch": ("use_torch_profiler",),
    "sync": ("diagnostic_enable_sync_hunter",),
    "torch+sync": ("use_torch_profiler", "diagnostic_enable_sync_hunter"),
    "python+sync": ("use_python_profiler", "diagnostic_enable_sync_hunter"),
}
# What the installer puts into the qwen venv per switch (install/components.py), as pip names its dist-info folder.
PACKAGES = {"use_python_profiler": "pyinstrument", "use_torch_profiler": "torch",
            "diagnostic_enable_sync_hunter": "torch"}
DEFAULT_PROMPT = "a weathered wooden treasure chest with iron bands and a brass lock"
# The launcher's PIN_ENV (launchers/qwen_server.py, the qwen venv's script, which the hub's modules do not import;
# tests/test_qwen_bench.py compares the two): the engine's variable for weights in page-locked RAM, which the
# launcher's --pin sets only where it is not set yet.
PIN_ENV = "SD_PIN_PARAMS"
BUSY_GPU_MB = 1536            # more in use before the start: the placement and the timings are not the usual ones
HEALTH_INTERVAL = 0.5
READ_AHEAD_WAIT = 600.0       # the model files read ahead (while the launcher loads; a slow disk takes longer)
POLL_INTERVAL = 2.0           # /spellcast/progress while a picture is made
SHUTDOWN_WAIT = 20.0
STOPPED = "stopped (Ctrl+C)"  # the failure of the run Ctrl+C interrupted
INTERRUPTED = 130             # the exit code after Ctrl+C (as a shell gives a process ended by it)
READ_AHEAD_DONE = re.compile(r"Model files read ahead|Reading the model files ahead stopped")


class BenchError(Exception):
    """Why the bench cannot go on, in plain words."""


# ---- arguments -----------------------------------------------------------------------------------------------------

def _side(text: str) -> int:
    value = int(text)
    if not 256 <= value <= 2048 or value % 32:
        raise argparse.ArgumentTypeError("a multiple of 32 from 256 to 2048")
    return value


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("1 or more")
    return value


def _cfg(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 30.0:                      # the launcher's limits (GenerateRequest.cfg_scale)
        raise argparse.ArgumentTypeError("a number from 0 to 30 (1: no guidance)")
    return value


def guided(cfg: float) -> bool:
    """Whether a CFG is guidance for stable-diffusion.cpp: anything but exactly 1.0, at which it skips the second
    model pass of every step and never reads the negative prompt (the launcher's condition_count)."""
    return cfg != 1.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m spellcast.profiling.qwen_bench",
                                description="Run Qwen-Image 2.1 on its own with the same request a few times, "
                                            "optionally profiled (reports in code\\logs\\profiling).")
    p.add_argument("--profile", choices=tuple(PROFILES), default="none",
                   help="which [profiling] switches the server gets (python and torch never together)")
    p.add_argument("--runs", type=_positive, default=2, help="how often the request is sent (run 2+: re-rolls)")
    p.add_argument("--model", nargs="+", choices=QWEN_MODELS, default=[DEFAULT_MODEL], metavar="turbo|full|flux3",
                   help="the request's model: turbo (the default: its own fixed 8 steps), full (--steps) or flux3 "
                        "(FLUX.3 Image Klein: its own 16 steps); several give run N the Nth (repeated): a run whose "
                        "model differs from the run before's switches first")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--prompts", nargs="+", default=None, metavar="TEXT",
                   help="one prompt per run (repeated when there are more runs); a run whose prompt differs from "
                        "the run before is a new description again (the text encoder runs)")
    p.add_argument("--seed", type=int, default=1234, help="run N uses seed + N - 1")
    p.add_argument("--size", type=_side, default=1024, help="width and height in pixels")
    p.add_argument("--steps", type=_positive, default=DEFAULT_STEPS, help="the hub's default unless given")
    p.add_argument("--cfg", type=_cfg, default=CFG_SCALE,
                   help="1 (the hub's, Qwen's own recipe): no guidance, one model pass per step; any other value: "
                        "guidance, two passes per step (e.g. the former recipe: --steps 20 --cfg 6)")
    p.add_argument("--negative", default="", metavar="TEXT",
                   help="the negative prompt of a request with guidance (not sent with --cfg 1: it would not be read)")
    p.add_argument("--batch", type=_positive, default=1, help="variations per request (one generate call)")
    p.add_argument("--references", nargs="+", default=[], metavar="PICTURE", help="example pictures (up to 10)")
    p.add_argument("--transparent", choices=("on", "off"), default="on")
    p.add_argument("--first-block-cache", choices=("auto", "on", "off"), default="auto",
                   help="the First-Block Cache (the page's 'Faster drawing' tick): auto = the card decides as for a new "
                        "prop (on for the GTX 10xx), on / off = the artist's tick")
    p.add_argument("--port", type=int, default=0, help="the server's port (0: any free one)")
    p.add_argument("--pin", choices=("auto", "on", "off"), default=None,
                   help="the launcher's --pin (the diffusion weights in page-locked RAM); not given: as the hub "
                        "starts it (the launcher's auto)")
    p.add_argument("--prefix-cache", choices=("auto", "f16", "q8_0"), default=None,
                   help="the launcher's --prefix-cache; not given: as the hub starts it (the launcher's auto)")
    p.add_argument("--prefix-keep", choices=("auto", "on", "off"), default=None,
                   help="the launcher's --prefix-keep (the prefix cache kept between pictures); not given: as the "
                        "hub starts it (the launcher's default)")
    p.add_argument("--attention", choices=("auto", "flash", "sage"), default=None,
                   help="the launcher's --attention (auto: SageAttention with the pinned engine; flash: flash "
                        "attention; sage: SageAttention with any build that has it, e.g. one under test with "
                        "--sdcpp); not given: as the hub starts it (the launcher's auto)")
    p.add_argument("--keep-params", choices=("auto", "on", "off"), default=None,
                   help="the launcher's --keep-params (the weights kept on the card between runs with an engine build "
                        "that knows SD_KEEP_STAGED_PARAMS; auto: where the decode fits beside them); not given: the "
                        "launcher's auto")
    p.add_argument("--card-gb", type=float, default=None, metavar="GB",
                   help="the launcher's --card-gb: plan for a card of this size (e.g. 8: the 8 GB path on a bigger "
                        "card); not given: the card as it is")
    p.add_argument("--verbose", action="store_true",
                   help="the launcher's --verbose: sd.cpp's debug lines (staging times, compute buffers) too")
    p.add_argument("--sdcpp", default=None, metavar="DIR",
                   help="an engine folder (stable-diffusion.dll and its DLLs: a build under test) instead of the "
                        "installed one (developers; the launcher then sees an unpinned build: add --attention sage "
                        "for SageAttention)")
    p.add_argument("--models-dir", default=None, metavar="DIR",
                   help="a models folder instead of models\\qwen-image (developers: its files named as there)")
    p.add_argument("--flux3-dir", default=None, metavar="DIR",
                   help="a folder instead of models\\flux3-klein for FLUX.3 Image Klein (developers: its files named "
                        "as there)")
    p.add_argument("--flux3-engine", choices=("auto", "on", "off"), default=None,
                   help="the launcher's --flux3-engine (on: take the engine build to draw FLUX.3, e.g. one under test "
                        "with --sdcpp); not given: as the hub starts it (the launcher's auto)")
    args = p.parse_args(argv)
    args.transparent = args.transparent == "on"
    args.first_block_cache = {"auto": None, "on": True, "off": False}[args.first_block_cache]
    return args


# ---- the config copy -----------------------------------------------------------------------------------------------

def switches_for(profile: str) -> dict[str, bool]:
    """{switch: on?} for every [profiling] switch the config file knows."""
    return {key: key in PROFILES[profile] for key in config.PROFILING_KEYS}


_SETTING = re.compile(r"([^=:]*?)\s*[=:]")             # as config.py finds a setting's name on a line
_VALUE = re.compile(r"^([^=:]*[=:]\s*)([^;#]*?)(\s*(?:[;#].*)?)$")    # name =, the value, the spaces and comment
_PROFILING_SECTION = re.compile(r"\[\s*profiling\s*\]", re.IGNORECASE)


def with_switches(text: str, switches: Mapping[str, bool]) -> str:
    """The config file's text with each switch's line set to on/off (only its value replaced). A further line of the
    same switch is switched off with a ; (two lines would leave that profiler off). A switch without a line is added
    after the [profiling] line, or in a new [profiling] section at the end. Every other line stays as it is, line
    endings included."""
    newline = "\r\n" if "\r\n" in text else "\n"
    out: list[str] = []
    done: set[str] = set()
    for line in text.splitlines():
        stripped = line.lstrip()
        m = _SETTING.match(stripped) if stripped and stripped[0] not in ";#[" else None
        key = m.group(1).strip().lower() if m else ""
        if key not in switches:
            out.append(line)
        elif key in done:
            out.append(f"; {stripped}")
        else:
            done.add(key)
            out.append(_VALUE.sub(lambda v: v.group(1) + ("on" if switches[key] else "off") + v.group(3), line))
    added = [f"{key} = {'on' if on else 'off'}" for key, on in switches.items() if key not in done]
    if added:
        section = next((i for i, line in enumerate(out) if _PROFILING_SECTION.fullmatch(line.strip())), None)
        if section is not None:
            out[section + 1:section + 1] = added
        else:
            out += ["[profiling]", *added]
    ends = not text or text.endswith(("\n", "\r")) or bool(added)
    return newline.join(out) + (newline if ends else "")


def write_config_copy(bench_dir: Path, switches: Mapping[str, bool]) -> tuple[Path, config.Config]:
    """The config file in use (config.config_file(); the defaults without one) with the switches, as
    bench_dir/spellcast3d-config.ini, and what the server will read from it. BenchError when that is not the switches
    asked for."""
    source = config.config_file()
    try:
        raw = source.read_bytes()
    except FileNotFoundError:
        raw = b""
    except OSError as e:
        raise BenchError(f"Cannot read {source} ({e}).") from e
    text = config._decode(raw) if raw else ""           # Notepad's encodings, as the hub reads them
    if text is None:
        raise BenchError(f"{source} is not readable text (save it as UTF-8).")
    copy = bench_dir / "spellcast3d-config.ini"
    copy.write_text(with_switches(text.lstrip("﻿"), switches), encoding="utf-8", newline="")
    seen = config.read(copy)
    wanted = tuple(key for key in config.PROFILING_KEYS if switches.get(key))
    if tuple(seen.profiling) != wanted:
        raise BenchError(f"The config copy {copy} switches on {list(seen.profiling)} instead of {list(wanted)}: "
                         + (" ".join(seen.problems) or "check the file."))
    return copy, seen


# ---- the launch ----------------------------------------------------------------------------------------------------

def launcher_switches(args: argparse.Namespace) -> list[str]:
    """The launcher's own switches the bench was asked to add to the hub's launch (--pin, --prefix-cache,
    --prefix-keep, --attention, --card-gb, --flux3-engine, --verbose), as its arguments; [] when none was given: the
    launch is then the hub's, and the launcher decides."""
    extra: list[str] = []
    if args.pin is not None:
        extra += ["--pin", args.pin]
    if args.prefix_cache is not None:
        extra += ["--prefix-cache", args.prefix_cache]
    if args.prefix_keep is not None:
        extra += ["--prefix-keep", args.prefix_keep]
    if args.attention is not None:
        extra += ["--attention", args.attention]
    if getattr(args, "keep_params", None) is not None:
        extra += ["--keep-params", args.keep_params]
    if args.card_gb is not None:
        extra += ["--card-gb", f"{args.card_gb:g}"]
    if getattr(args, "flux3_engine", None) is not None:
        extra += ["--flux3-engine", args.flux3_engine]
    if args.verbose:
        extra += ["--verbose"]
    return extra


def server_launch(port: int, settings: dict[str, Any], projects_root: Path, config_copy: Path,
                  extra: Sequence[str] = (), sdcpp: Path | None = None, models_dir: Path | None = None,
                  flux3_dir: Path | None = None) -> LaunchSpec:
    """The hub's launch of Qwen-Image (procman: the registered QwenAdapter(), launch_spec(port, merged settings)),
    built with SPELLCAST_CONFIG naming the config copy - set in this process's environment first, so the adapter
    picks the script from the copy's switches and the server inherits it - and with --projects-root projects_root
    and the launcher's switches `extra` (launcher_switches) at the end (bench_launch: with --pin among them, without
    the shell's SD_PIN_PARAMS); sdcpp: an engine folder instead of the installed one (--sdcpp); models_dir: a models
    folder instead of models\\qwen-image (--models-dir); flux3_dir: one instead of models\\flux3-klein
    (--flux3-dir)."""
    os.environ["SPELLCAST_CONFIG"] = str(config_copy)
    spec = QwenAdapter().launch_spec(port, settings)
    if spec.env.get("SPELLCAST_CONFIG") != str(config_copy):
        raise BenchError("The server would not read the bench's config copy (SPELLCAST_CONFIG is not passed on).")
    return bench_launch(spec, projects_root, extra, sdcpp, models_dir, flux3_dir)


def bench_launch(spec: LaunchSpec, projects_root: Path, extra: Sequence[str] = (),
                 sdcpp: Path | None = None, models_dir: Path | None = None,
                 flux3_dir: Path | None = None) -> LaunchSpec:
    """`spec` with --projects-root projects_root and `extra` added at the end. With --pin in `extra`, its environment
    is also without SD_PIN_PARAMS: the hub's launch passes on one that is set in the shell (paths.clean_env), and the
    launcher leaves a set one as it is whatever --pin says (its rule for the artist's PC), so the switch the bench was
    asked to measure would do nothing. With `sdcpp`, the launch's --sdcpp, its working folder and the head of its
    PATH (where stable-diffusion.dll finds ggml-cuda.dll and the CUDA runtime) name that folder instead of the
    installed engine's (adapters/qwen.py). With `models_dir`, its --models; with `flux3_dir`, its --flux3-models.
    Nothing else changed."""
    argv = list(spec.argv)
    if "--projects-root" not in argv[:-1]:
        raise BenchError("The hub's launch of Qwen-Image has no --projects-root: update qwen_bench.py.")
    argv[argv.index("--projects-root") + 1] = str(projects_root)
    if models_dir is not None:
        if "--models" not in argv[:-1]:
            raise BenchError("The hub's launch of Qwen-Image has no --models: update qwen_bench.py.")
        argv[argv.index("--models") + 1] = str(models_dir)
    if flux3_dir is not None:
        if "--flux3-models" not in argv[:-1]:
            raise BenchError("The hub's launch of Qwen-Image has no --flux3-models: update qwen_bench.py.")
        argv[argv.index("--flux3-models") + 1] = str(flux3_dir)
    env = spec.env
    if "--pin" in extra and PIN_ENV in env:
        env = {name: value for name, value in env.items() if name != PIN_ENV}
    cwd = spec.cwd
    if sdcpp is not None:
        if "--sdcpp" not in argv[:-1]:
            raise BenchError("The hub's launch of Qwen-Image has no --sdcpp: update qwen_bench.py.")
        installed = argv[argv.index("--sdcpp") + 1]
        argv[argv.index("--sdcpp") + 1] = str(sdcpp)
        path = env.get("PATH", "")
        if path.split(os.pathsep)[0] != installed:
            raise BenchError("The hub's launch of Qwen-Image no longer puts the engine folder first on PATH: update "
                             "qwen_bench.py.")
        env = {**env, "PATH": os.pathsep.join([str(sdcpp), *path.split(os.pathsep)[1:]])}
        if os.name != "nt":
            # the engine's shared libraries resolve their dependencies through LD_LIBRARY_PATH first (a pod's profile
            # names the installed engine there): the folder under test goes first
            env["LD_LIBRARY_PATH"] = os.pathsep.join([str(sdcpp), *filter(None, [env.get("LD_LIBRARY_PATH")])])
        cwd = sdcpp
    return LaunchSpec(argv=[*argv, *extra], cwd=cwd, env=env, ready_timeout=spec.ready_timeout)


def shell_pin_words(pin: str | None, shell: str | None) -> str | None:
    """What the bench says about an SD_PIN_PARAMS of its own shell (`shell`: its value there; None: not set, and
    nothing to say) before it starts the server. pin: the bench's --pin, None when it was not given. With --pin the
    variable stays out of the server's environment and the switch decides (bench_launch); without, it goes along as
    from the hub and the launcher keeps it: every run is then pinned as the shell says, not as the launcher's auto
    would decide on this PC - printed as a warning, since neither the table nor "the hub's" switches show it."""
    if shell is None:
        return None
    if pin is None:
        return (f"{PIN_ENV}={shell!r} is set in this shell and goes to the server, as from the hub: the launcher "
                f"leaves a set one as it is, so in every run the engine keeps in page-locked RAM what IT names (a "
                f"build that knows the variable; the official release ignores it), whatever the launcher's --pin "
                f"auto would decide. Give --pin auto, on or off to measure the switch: the bench then leaves the "
                f"variable out.")
    return (f"{PIN_ENV}={shell!r} of this shell is not passed on to the server: --pin {pin} decides (the launcher "
            f"never changes a set one, so the switch would do nothing).")


def qwen_site_packages() -> Path:
    return paths.venv_dir(paths.QWEN) / "Lib" / "site-packages"


def missing_for(profile: str, site_packages: Path) -> list[str]:
    """The packages --profile needs that pip has not finished installing there (a dist-info folder with its RECORD,
    which pip writes last)."""
    needed = sorted({PACKAGES[key] for key in PROFILES[profile]})
    return [name for name in needed
            if not any((info / "RECORD").is_file() for info in site_packages.glob(f"{name}-*.dist-info"))]


def refusal(profile: str, missing: list[str]) -> str:
    keys = " and ".join(f"{key} = on" for key in PROFILES[profile])
    return (f"--profile {profile} needs {' and '.join(missing)} in Qwen-Image's Python environment "
            f"({paths.venv_dir(paths.QWEN)}), which the installer has not put there. Switch the profiler on in "
            f"{config.config_file()} ([profiling]: {keys}), then close Spellcast3D and start it again (its installer "
            f"adds {'them' if len(missing) > 1 else 'it'}) - or, with Spellcast3D closed, run "
            f"tools\\python\\python.exe code\\install.py --only qwen --if-needed - and run the bench again.")


def run_prompt(args: argparse.Namespace, number: int) -> str:
    """Run `number`'s prompt: --prompts' (the list repeated), else --prompt."""
    if args.prompts:
        return args.prompts[(number - 1) % len(args.prompts)]
    return args.prompt


def run_model(args: argparse.Namespace, number: int) -> str:
    """Run `number`'s model: --model's Nth (the list repeated)."""
    return args.model[(number - 1) % len(args.model)]


def missing_models(args: argparse.Namespace, folder: Path, quality: str, flux3_folder: Path | None = None) -> list[str]:
    """The files of the models --model names that are not there (as the launcher looks: a file of that name): the
    Qwen-Image denoisers in `folder`, FLUX.3's three files in flux3_folder (models\\flux3-klein unless given),
    named with that folder."""
    names = {model_files.qwen_denoiser(model, quality)[0] for model in args.model if model != "flux3"}
    gone = sorted(name for name in names if not (folder / name).is_file())
    if "flux3" in args.model:
        flux3 = flux3_folder or paths.MODELS_FLUX3
        gone += [str(flux3 / name) for name in model_files.flux3_manifest(quality) if not (flux3 / name).is_file()]
    return gone


def request_body(args: argparse.Namespace, seed: int, references: list[Path], out_dir: Path,
                 prompt: str | None = None, model: str | None = None) -> dict[str, Any]:
    """POST /spellcast/generate's body as the hub makes it for a new prop with these settings - always without
    guidance (adapters/qwen.py) - and then the bench's own guidance in it: --cfg, and the negative prompt --negative
    when that CFG is guidance (else none: the launcher would not use it). With --cfg 1 it is the hub's very body.
    prompt: the run's (run_prompt), else --prompt; model: the run's (run_model), else --model's first. The Turbo's
    body says its own 8 steps (the adapter's), FLUX.3's its own 16, and no guidance: the bench's --cfg is the full
    model's."""
    model = model or args.model[0]
    params = {**projects.STAGE_DEFAULTS["images"]["settings"], "prompt": args.prompt if prompt is None else prompt,
              "width": args.size,
              "height": args.size, "seed": seed, "batch_count": args.batch, "transparent": args.transparent,
              "first_block_cache": args.first_block_cache, "qwen_steps": args.steps, "qwen_model": model}
    body = qwen_request(params, references, out_dir)
    if model == "full":
        body["cfg_scale"] = args.cfg
        body["negative_prompt"] = args.negative.strip() if guided(args.cfg) else ""
    return body


def model_words(args: argparse.Namespace) -> str:
    """The runs' models and their steps for the bench's first lines: "turbo (its own 8 steps)", "full (25 steps)",
    "flux3 (its own 16 steps)"."""
    said = {"turbo": f"turbo (its own {TURBO_STEPS} steps)", "full": f"full ({args.steps} steps)",
            "flux3": f"flux3 (its own {FLUX3_STEPS} steps)"}
    return ", then ".join(said[model] for model in args.model) + (" (repeated)" if len(args.model) > 1 else "")


def guidance_words(args: argparse.Namespace) -> str:
    """The request's guidance for the bench's first lines: "no guidance", or the CFG and its negative prompt."""
    if not guided(args.cfg):
        return "no guidance"
    negative = args.negative.strip()
    return f"CFG {args.cfg:g} with " + (f"the negative prompt {negative!r}" if negative else "no negative prompt")


# ---- what is around ------------------------------------------------------------------------------------------------

def busy_gpu_warning() -> str | None:
    gpus = gpu.query_gpus()
    if not gpus or gpus[0].vram_used_mb <= BUSY_GPU_MB:
        return None
    g = gpus[0]
    return (f"{g.name} already has {g.vram_used_mb:,} of {g.vram_total_mb:,} MB in use (other programs, or the hub's "
            f"own tools): Qwen-Image places its weights by the free memory, so the timings are not the usual ones. "
            f"Close what uses the card for comparable numbers.")


def hub_qwen_warning(port: int) -> str | None:
    try:
        with httpx.Client(trust_env=False) as client:
            r = client.get(f"http://127.0.0.1:{port}/spellcast/health", timeout=2.0)
            answering = r.status_code == 200 and "ready" in r.json()
    except (httpx.HTTPError, ValueError, TypeError):
        return None
    return (f"A Qwen-Image server already answers on port {port} (the hub's): it holds RAM and video memory while the "
            f"bench runs a second one. Stop it (Settings > Debugging) for comparable numbers.") if answering else None


def free_port() -> int:
    for _ in range(20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if procman.port_is_free(port):
            return port
    raise BenchError("No free network port found.")


def report_folders(root: Path) -> set[Path]:
    try:
        return {p for p in root.iterdir() if p.is_dir() and not p.name.startswith("bench-")}
    except OSError:
        return set()


def new_report(root: Path, before: set[Path]) -> Path | None:
    """The newest profiling report folder made since `before` was taken, or None."""
    made = report_folders(root) - before
    return max(made, key=lambda p: (p.stat().st_mtime, p.name)) if made else None


def bench_folder(root: Path = PROFILING_ROOT) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder, n = root / f"bench-{stamp}", 1
    while folder.exists():
        n += 1
        folder = root / f"bench-{stamp}-{n}"
    folder.mkdir(parents=True)
    return folder


# ---- the server's process ------------------------------------------------------------------------------------------

class ServerOutput(threading.Thread):
    """The server's merged output into bench.log and, prefixed, this console; remembers the last lines and whether
    the model files were read ahead."""

    def __init__(self, stream: IO[bytes], log_path: Path, echo: bool = True) -> None:
        super().__init__(name="qwen-output", daemon=True)
        self.stream, self.log_path, self.echo = stream, log_path, echo
        self.last: collections.deque[str] = collections.deque(maxlen=15)
        self.read_ahead = threading.Event()

    def run(self) -> None:
        with open(self.log_path, "a", encoding="utf-8", errors="replace") as log:
            for raw in iter(self.stream.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                log.write(line + "\n")
                log.flush()
                self.last.append(line)
                if READ_AHEAD_DONE.search(line):
                    self.read_ahead.set()
                if self.echo:
                    print(f"  qwen | {line}", flush=True)
        self.read_ahead.set()


@dataclass
class Run:
    number: int
    seed: int
    new_description: bool | None = None    # the prompt is not the run before's: the text encoder runs (None: run 1)
    model: str = DEFAULT_MODEL             # the request's model (turbo | full | flux3)
    new_model: bool = False                # not the run before's: the launcher switches first (a new context)
    status: int = 0
    seconds: float | None = None           # the answer's: the server's own timing, without the profiling reports
    wall: float = 0.0                      # the bench's, POST to answer (the reports' writing included)
    timings: dict[str, float] = field(default_factory=dict)
    images: int = 0
    warning: str | None = None
    failure: str | None = None
    report: Path | None = None

    @property
    def conditioning(self) -> str:
        if self.new_model:
            return "new model"              # the switch made a new context: no conditioning cache either
        new = self.number == 1 if self.new_description is None else self.new_description
        return "new description" if new else "re-roll"


def wait_ready(base_url: str, proc: subprocess.Popen[bytes], output: ServerOutput, timeout: float) -> float:
    started = time.monotonic()
    with httpx.Client(trust_env=False) as client:
        while True:
            if proc.poll() is not None:
                raise BenchError(f"Qwen-Image stopped while starting (exit code {proc.returncode}). Last lines:\n  "
                                 + "\n  ".join(output.last))
            try:
                r = client.get(f"{base_url}/spellcast/health", timeout=5.0)
                if r.status_code == 200 and r.json().get("ready") is True:
                    return time.monotonic() - started
            except (httpx.HTTPError, ValueError):
                pass
            if time.monotonic() - started > timeout:
                raise BenchError(f"Qwen-Image was not ready within {timeout / 60:.0f} minutes.")
            time.sleep(HEALTH_INTERVAL)


def one_run(base_url: str, body: dict[str, Any], run: Run, proc: subprocess.Popen[bytes]) -> None:
    """POST /spellcast/generate on a thread (Ctrl+C stays responsive here: it cancels the picture), its progress
    printed when the phase changes and every 15 s."""
    def post() -> httpx.Response:
        with httpx.Client(trust_env=False) as client:
            return client.post(f"{base_url}/spellcast/generate", json=body,
                               timeout=httpx.Timeout(IMAGE_TIMEOUT * body["batch_count"], connect=10.0))

    started = time.monotonic()
    shown, shown_at = "", 0.0
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="bench-post")
    future = pool.submit(post)
    try:
        with httpx.Client(trust_env=False) as poll:
            while True:
                try:
                    response = future.result(timeout=POLL_INTERVAL)
                    break
                except concurrent.futures.TimeoutError:
                    pass
                try:
                    p = poll.get(f"{base_url}/spellcast/progress", timeout=5.0).json()
                except (httpx.HTTPError, ValueError):
                    continue
                message = str(p.get("message") or "")
                new_phase = message.split(" ·")[0] != shown.split(" ·")[0]
                if p.get("phase") not in (None, "idle") and message != shown and \
                        (new_phase or time.monotonic() - shown_at >= 15):
                    eta = p.get("eta")
                    print(f"  run {run.number}: {message}" + (f" (about {eta:.0f} s left)"
                                                              if isinstance(eta, (int, float)) else ""), flush=True)
                    shown, shown_at = message, time.monotonic()
    except httpx.HTTPError as e:
        run.wall = time.monotonic() - started
        died = proc.poll()
        run.failure = (f"no answer: Qwen-Image stopped (exit code {died})" if died is not None else
                       f"no answer ({type(e).__name__}: {e})")
        return
    except KeyboardInterrupt:
        run.wall = time.monotonic() - started
        run.failure = STOPPED
        cancel_picture(base_url)
        raise
    finally:
        pool.shutdown(wait=False)
    run.wall = time.monotonic() - started
    run.status = response.status_code
    try:
        data = response.json()
    except ValueError:
        data = None
    if response.status_code == 200 and isinstance(data, dict):
        run.seconds = data.get("seconds")
        run.timings = data.get("timings") or {}
        run.images = len(data.get("images") or [])
        run.warning = data.get("warning")
    else:
        run.failure = (f"{data.get('kind')}: {data.get('detail')}" if isinstance(data, dict) else
                       f"HTTP {response.status_code}: {response.text[:300]}")


def stop_server(base_url: str, proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        with httpx.Client(trust_env=False) as client:
            client.post(f"{base_url}/spellcast/shutdown", json={}, timeout=5.0)
    except httpx.HTTPError:
        pass
    try:
        proc.wait(SHUTDOWN_WAIT)
    except subprocess.TimeoutExpired:
        print(f"Qwen-Image did not stop within {SHUTDOWN_WAIT:.0f} s: ending it.", flush=True)
        procman.kill_tree(proc.pid)
        try:
            proc.wait(5.0)
        except subprocess.TimeoutExpired:
            pass


def cancel_picture(base_url: str) -> None:
    try:
        with httpx.Client(trust_env=False) as client:
            client.post(f"{base_url}/spellcast/cancel", json={}, timeout=5.0)
    except httpx.HTTPError:
        pass


# ---- the report on the console -------------------------------------------------------------------------------------

def _part(timings: Mapping[str, float], prefix: str) -> float | None:
    values = [v for k, v in timings.items() if k == prefix or k.startswith(prefix + "_")]
    return round(sum(values), 1) if values else None


def summary(runs: list[Run], ready_seconds: float | None) -> str:
    def cell(value: Any) -> str:
        return "-" if value is None else f"{value:.1f}" if isinstance(value, float) else str(value)

    rows = [("run", "seed", "model", "conditioning", "seconds", "wall", "switch", "examples", "prompt", "drawing",
             "finishing", "report / failure")]
    for r in runs:
        rows.append((str(r.number), str(r.seed), r.model, r.conditioning, cell(r.seconds), cell(round(r.wall, 1)),
                     cell(_part(r.timings, "switch")),
                     cell(_part(r.timings, "references")), cell(_part(r.timings, "prompt")),
                     cell(_part(r.timings, "drawing")), cell(_part(r.timings, "finishing")),
                     r.failure or (r.report.name if r.report else "-")))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]
    lines = ["  ".join(text.ljust(w) for text, w in zip(row[:-1], widths)) + "  " + row[-1] for row in rows]
    head = f"Ready in {ready_seconds:.1f} s. " if ready_seconds is not None else ""
    return (head + "seconds: the server's own timing (without writing profiling reports); wall: the request as the "
            "hub waits for it; switch: the launcher's change of model before the run (freeing one, loading the other); "
            "examples / prompt / drawing / finishing: sd.cpp's timings (seconds).\n"
            + "\n".join(lines))


# ---- main ----------------------------------------------------------------------------------------------------------

def run_bench(args: argparse.Namespace) -> int:
    if not paths.venv_python(paths.QWEN).exists():
        raise BenchError(f"Qwen-Image is not installed ({paths.venv_python(paths.QWEN)} is missing). Pick it in "
                         f"spellcast3d-config.ini and start Spellcast3D once.")
    missing = missing_for(args.profile, qwen_site_packages())
    if missing:
        raise BenchError(refusal(args.profile, missing))
    for picture in args.references:
        if not Path(picture).is_file():
            raise BenchError(f"The example picture {picture} was not found.")
    settings = SettingsStore().get()
    warnings = [w for w in (busy_gpu_warning(), hub_qwen_warning(int(settings["ports"][paths.QWEN]))) if w]

    models_dir = Path(args.models_dir).resolve() if args.models_dir else None
    flux3_dir = Path(args.flux3_dir).resolve() if getattr(args, "flux3_dir", None) else None
    quality = config.qwen_quality()          # the config file's (the bench's copy says the same)
    gone = missing_models(args, models_dir or paths.MODELS_QWEN, quality, flux3_dir)
    if gone:
        raise BenchError(f"--model {' '.join(args.model)} needs {', '.join(gone)} in "
                         f"{models_dir or paths.MODELS_QWEN}, which is not there (the full model and FLUX.3 are "
                         f"downloaded from the Images page; the Turbo by the installer).")
    bench_dir = bench_folder()
    config_copy, seen = write_config_copy(bench_dir, switches_for(args.profile))
    references = []
    for i, picture in enumerate(args.references, 1):
        copy = bench_dir / "references" / f"{i:02d}-{Path(picture).name}"
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(picture, copy)
        references.append(copy)
    try:
        request_body(args, args.seed, references, bench_dir / "run-1")       # the hub's own checks, before starting
    except StageError as e:
        raise BenchError(str(e)) from e
    port = args.port or free_port()
    if not procman.port_is_free(port):
        raise BenchError(f"Port {port} is in use: pick another with --port (0: any free one).")
    extra = launcher_switches(args)
    sdcpp = Path(args.sdcpp).resolve() if args.sdcpp else None
    engine_lib = "stable-diffusion.dll" if os.name == "nt" else "libstable-diffusion.so"
    if sdcpp is not None and not (sdcpp / engine_lib).is_file():
        raise BenchError(f"--sdcpp {sdcpp} has no {engine_lib}.")
    try:
        spec = server_launch(port, settings, bench_dir, config_copy, extra, sdcpp, models_dir, flux3_dir)
    except StageError as e:
        raise BenchError(str(e)) from e

    full = "full" in args.model
    cache = {None: "by the card", True: "on", False: "off"}[args.first_block_cache]
    if not full:
        cache = "off (the Turbo, FLUX.3)"
    elif len(set(args.model)) > 1:
        cache += " for the full model's runs, off for the others'"
    print(f"Bench folder: {bench_dir}")
    print(f"Profile: {args.profile} (switched on: {', '.join(seen.profiling) or 'none'}); model size {seen.quality}; "
          f"model {model_words(args)}; {args.runs} run(s) of {args.batch} picture(s) {args.size}x{args.size}, "
          f"{guidance_words(args) if full else 'no guidance'}, {len(references)} example picture(s), transparent "
          f"{'on' if args.transparent else 'off'}" + (" (not for FLUX.3: no alpha)" if args.transparent and "flux3" in
                                                      args.model else "") + f", first-block cache {cache}.")
    if args.negative.strip() and not guided(args.cfg):
        print("--negative is not sent: without guidance (--cfg 1) Qwen-Image reads no negative prompt.")
    if not full and (guided(args.cfg) or args.steps != DEFAULT_STEPS or args.first_block_cache is not None):
        print("--steps, --cfg and --first-block-cache are the full model's: the Turbo draws its own fixed 8 steps, "
              "FLUX.3 its own 16.")
    print("The server's switches: " + (f"the hub's, and {' '.join(extra)}." if extra else
                                       "the hub's (the launcher decides --pin and --prefix-cache).")
          + (f" Engine folder: {sdcpp} (--sdcpp; not the installed one)." if sdcpp is not None else "")
          + (f" Models folder: {models_dir} (--models-dir)." if models_dir is not None else "")
          + (f" FLUX.3's folder: {flux3_dir} (--flux3-dir)." if flux3_dir is not None else ""))
    pin_words = shell_pin_words(args.pin, os.environ.get(PIN_ENV))     # the shell's variable: left out, or deciding
    if pin_words:
        print(pin_words if args.pin is not None else f"WARNING: {pin_words}")
    print("Run 1 is a new description (the text encoder runs); runs 2+ are re-rolls with the next seeds (sd.cpp's "
          "conditioning cache skips the text encoder), as when the artist presses Generate again"
          + (" - except a run whose --prompts entry differs from the run before's: a new description again." if
             args.prompts else "."))
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Starting: {subprocess.list2cmdline(spec.argv)}\n  in {spec.cwd}, SPELLCAST_CONFIG={config_copy}",
          flush=True)

    log_path = bench_dir / "bench.log"
    log_path.write_text(f"{subprocess.list2cmdline(spec.argv)}\nSPELLCAST_CONFIG={config_copy}\n", encoding="utf-8")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0     # Ctrl+C reaches the bench only
    proc = procman.spawn_in_job(spec.argv, spec.cwd, spec.env, flags)
    assert proc.stdout is not None
    output = ServerOutput(proc.stdout, log_path)
    output.start()
    base_url = f"http://127.0.0.1:{port}"
    runs: list[Run] = []
    ready: float | None = None
    interrupted = False
    try:
        ready = wait_ready(base_url, proc, output, spec.ready_timeout)
        print(f"Ready in {ready:.1f} s. Waiting until the model files are read ahead...", flush=True)
        deadline = time.monotonic() + READ_AHEAD_WAIT
        while not output.read_ahead.wait(0.5):          # short waits: Ctrl+C is seen between them
            if time.monotonic() > deadline:
                print("The model files are still being read ahead: starting anyway (run 1 shares the disk).",
                      flush=True)
                break
        loaded = None                                   # the launcher's model (health), then each run's
        try:
            with httpx.Client(trust_env=False) as client:
                loaded = client.get(f"{base_url}/spellcast/health", timeout=5.0).json().get("model")
        except (httpx.HTTPError, ValueError, AttributeError):
            pass
        for n in range(1, args.runs + 1):
            prompt, model = run_prompt(args, n), run_model(args, n)
            run = Run(number=n, seed=args.seed + n - 1, new_description=n == 1 or prompt != run_prompt(args, n - 1),
                      model=model, new_model=loaded is not None and model != loaded)
            loaded = model
            runs.append(run)
            out_dir = bench_dir / f"run-{n}"
            out_dir.mkdir(parents=True, exist_ok=True)
            body = request_body(args, run.seed, references, out_dir, prompt, model)
            print(f"Run {n} of {args.runs} (seed {run.seed}, {model}, {run.conditioning})...", flush=True)
            before = report_folders(PROFILING_ROOT)
            one_run(base_url, body, run, proc)
            run.report = new_report(PROFILING_ROOT, before)
            print(f"Run {n}: " + (f"failed ({run.failure})" if run.failure else
                                  f"{run.seconds} s (wall {run.wall:.1f} s), {run.images} picture(s), sd.cpp timings "
                                  f"{run.timings}" + (f", warning: {run.warning}" if run.warning else ""))
                  + (f"\n  report: {run.report}" if run.report else ""), flush=True)
            if run.failure and (run.failure.startswith("restart") or proc.poll() is not None):
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped (Ctrl+C): cancelling the picture and stopping Qwen-Image.", flush=True)
        cancel_picture(base_url)
        if runs and runs[-1].status == 0 and runs[-1].failure is None:     # interrupted before its answer came
            runs[-1].failure = STOPPED
    finally:
        stop_server(base_url, proc)
        output.join(5.0)
    if runs:
        print("\n" + summary(runs, ready))
    print(f"\nPictures, the config copy and bench.log: {bench_dir}")
    if interrupted:                             # even on the last run: a stopped bench is never a successful one
        return INTERRUPTED
    return 0 if len(runs) == args.runs and not any(r.failure for r in runs) else 1


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    try:
        return run_bench(args)
    except BenchError as e:
        print(f"qwen_bench: {e}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        return INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
