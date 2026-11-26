"""Command line and orchestration of the Spellcast3D installer. STDLIB ONLY.

    tools\\python\\python.exe code\\install.py [--if-needed] [--only IDS] [--skip IDS]
        [--reinstall IDS] [--check] [--dry-run] [--force-pixal3d] [--yes] [--busy-check]
        [--progress-file PATH] [--models-from DIR ...]

Components (IDS, comma-separated): hub, imesh, qwen, forge, pixal3d, lato2, triflow. They are installed
in that order, each into code\\venvs\\<id>, and one failing does not stop the others.
A component whose fingerprint still matches code\\venvs\\install-state.json is skipped.
Of the image generators (qwen = Qwen-Image 2.1, forge = Forge Neo) only the one spellcast3d-config.ini picks is
installed (none while that file has a mistake), unless --only or --reinstall names the other; the other one's
files are kept. Every run but --check, --dry-run and --busy-check first makes that file when it is missing.

tools\\spellcast-internal.bat installs only the hub (--only hub --if-needed: seconds when nothing changed)
and starts it; the hub then runs `--if-needed --skip hub --progress-file code\\venvs\\install-progress.json`
in the background for the tools and shows that file's progress (install/progress.py).
Last (also when nothing needed work), best effort and never a component's problem: the package a [profiling]
switch of spellcast3d-config.ini needs, into Qwen-Image's venv (components.install_profilers, for developers).

Exit codes: 0 ok, 1 something failed (or a check failed), 2 (--check only) something
needs installing, 3 another installer is already running (--busy-check: or Spellcast3D is open).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import io
import os
import shutil
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Iterator, NamedTuple, NoReturn

from .. import config, gpu, paths
from . import layout, models, preflight, progress, sdcpp, state, venvs
from .components import (COMPONENTS, LOG_HINT, Component, Qwen, git_exe, install_profilers, missing_profilers,
                         pixal3d_calibration_work, profilers_pending, profilers_work)
from .progress import Progress
from .runner import Ctx, InstallError, Output

EXIT_OK, EXIT_FAILED, EXIT_PENDING, EXIT_BUSY = 0, 1, 2, 3
UP_TO_DATE = "up to date"  # summary status of a component that needed no work
STOPPED = "The installation was stopped before it finished. Start Spellcast3D again to continue."
INSTALL_LOCK = paths.VENVS / ".install.lock"
HUB_LOCK = paths.VENVS / ".hub.lock"     # held by a running hub (spellcast/__main__.py)
WIDTH = 78
LIGHT_GB = 1.0      # an install this small takes about a minute
IMAGE_BACKENDS = paths.STAGE_TOOLS["images"]     # spellcast3d-config.ini picks one of them (spellcast/config.py)
# install-state status (or UP_TO_DATE) -> the status the progress file reports
_PROGRESS_STATUS = {state.INSTALLED: progress.DONE, UP_TO_DATE: progress.DONE, state.FAILED: progress.FAILED,
                    state.UNSUPPORTED: progress.UNSUPPORTED}


class Pending(NamedTuple):
    """Why a component is installed in this run; repair = it is installed and unchanged, only
    Component.repair() runs (e.g. to fetch missing AI models)."""
    reason: str
    repair: bool = False


class _Busy(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        # argparse would exit with 2, which --check uses for "something needs installing".
        self.print_usage(sys.stderr)
        self.exit(EXIT_FAILED, f"{self.prog}: error: {message}\n")


@functools.cache
def _gpu_info() -> gpu.GpuInfo | None:
    return gpu.primary_gpu()


def _ids(text: str) -> tuple[str, ...]:
    ids = tuple(part.strip().lower() for part in text.split(",") if part.strip())
    unknown = [i for i in ids if i not in paths.ALL_VENV_IDS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown component {', '.join(unknown)}; choose from {', '.join(c.id for c in COMPONENTS)}")
    return ids


def _root_relative(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else paths.ROOT / path


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = _Parser(
        prog="install.py",
        description=f"Installs the parts of Spellcast3D ({', '.join(c.id for c in COMPONENTS)}) into code\\venvs. "
                    f"Of the image generators ({' and '.join(IMAGE_BACKENDS)}) only the one spellcast3d-config.ini "
                    f"picks, unless --only or --reinstall names the other.",
        epilog="Exit codes: 0 ok, 1 failed, 2 (--check) something needs installing, 3 already running.")
    p.add_argument("--if-needed", action="store_true",
                   help="exit at once when everything is installed and unchanged (used at every start)")
    p.add_argument("--only", type=_ids, default=(), metavar="IDS", help="only these components, e.g. hub,imesh")
    p.add_argument("--skip", type=_ids, default=(), metavar="IDS", help="leave these components alone")
    p.add_argument("--reinstall", type=_ids, default=(), metavar="IDS",
                   help="delete these components' venvs and install them from scratch")
    p.add_argument("--check", action="store_true", help="run the system checks and report the state; change nothing")
    p.add_argument("--dry-run", action="store_true", help="print the commands that would run, run nothing")
    p.add_argument("--force-pixal3d", action="store_true",
                   help="install Pixal3D even on a GPU older than GTX 1060 (it will most likely not run)")
    p.add_argument("--yes", action="store_true",
                   help="never wait for a key press (the calibration notice's Enter; SPELLCAST_NO_PAUSE=1 does the "
                        "same); the installer asks no other questions")
    p.add_argument("--verbose", action="store_true",
                   help="show every line of the tools' installers, pip's \"Requirement already satisfied\" ones too "
                        "(for a bug report; SPELLCAST_VERBOSE_INSTALL=1 does the same for the background installer)")
    p.add_argument("--models-from", action="append", type=_root_relative, default=[], metavar="DIR",
                   help="take the AI models of Qwen-Image 2.1, Hi3D, LATO.2 and TriFlow (and stable-diffusion.cpp's "
                        "zips) from DIR when they are there, e.g. another Spellcast3D's models folder: hard links on "
                        "the same drive instead of downloads (repeatable; SPELLCAST_MODELS_FROM, separated by ';', "
                        "does the same)")
    p.add_argument("--busy-check", action="store_true",
                   help="only report whether Spellcast3D or an installer is running (exit 3) or not (exit 0); "
                        "tools\\update.bat uses it")
    p.add_argument("--progress-file", type=_root_relative, default=None, metavar="PATH",
                   help="write the progress as JSON to PATH (relative to the Spellcast3D folder) for the hub, "
                        "which runs the installer this way while it is open: the open hub does not stop it, and "
                        "its own part (hub) is left alone then")
    args = p.parse_args(argv)
    selected = _selected(args)
    outside = [i for i in args.reinstall if i not in {c.id for c in selected}]
    if outside:
        p.error(f"--reinstall {','.join(outside)}: not selected (check --only / --skip)")
    return args


def _selected(args: argparse.Namespace) -> list[Component]:
    """The components of this run, in install order: --only / --skip, and of the image generators only the one
    spellcast3d-config.ini picks (none while it has a mistake), unless --only or --reinstall names another."""
    image = config.image_backend()
    named = set(args.only) | set(getattr(args, "reinstall", ()))
    return [c for c in COMPONENTS if (not args.only or c.id in args.only) and c.id not in args.skip
            and (c.id not in IMAGE_BACKENDS or c.id == image or c.id in named)]


def _config_problems(args: argparse.Namespace) -> list[str]:
    """What is wrong in spellcast3d-config.ini, when this run would install an image generator."""
    concerned = [i for i in IMAGE_BACKENDS if (not args.only or i in args.only) and i not in args.skip]
    return config.problems() if concerned else []


def _pending(c: Component, entry: dict | None, args: argparse.Namespace) -> Pending | None:
    """Why c must be (re)installed or repaired now, or None when it is up to date (or unsupported here)."""
    if c.id in args.reinstall:
        return Pending("a fresh reinstall was requested")
    status = (entry or {}).get("status")
    if status == state.UNSUPPORTED:
        if args.force_pixal3d and c.id == paths.PIXAL3D:
            return Pending("--force-pixal3d was given")
        return Pending("this PC can run it now") if c.supported(_gpu_info())[0] else None
    if status == state.INSTALLED:
        if not venvs.usable(c.id):
            return Pending("its Python environment is missing or broken")
        if entry.get("fingerprint") != c.fingerprint():
            return Pending("an update changed it")
        work = c.repair_work()
        return Pending(work, repair=True) if work else None
    if status == state.FAILED:
        return Pending("the last attempt did not finish")
    return Pending("not installed yet")


def _plan(selected: list[Component], args: argparse.Namespace) -> dict[str, Pending]:
    entries = state.load()
    return {c.id: why for c in selected if (why := _pending(c, entries.get(c.id), args))}


def main(argv: list[str] | None = None) -> int:
    if sys.version_info[:2] != (3, 11):
        print(f"Spellcast3D's installer needs Python 3.11, but this is Python {sys.version.split()[0]}.\n"
              f"Start Spellcast3D with run-browser\\run-spellcast3d.bat, which uses tools\\python.")
        return EXIT_FAILED
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="replace")  # a legacy console code page must not crash on tool output
    args = _parse(argv)
    if args.busy_check:
        return EXIT_BUSY if any(locked(f) for f in (INSTALL_LOCK, HUB_LOCK)) else EXIT_OK
    models.EXTRA_SOURCES[:] = args.models_from
    base_python = Path(getattr(sys, "_base_executable", sys.executable))
    background = args.progress_file is not None and not args.dry_run and not args.check
    # In the background its output shares the hub's console, and the hub shows its progress.
    out = Output(transient=not background)
    tracker = Progress(args.progress_file if background else None)
    # Made at the first start (by the hub-only run already), so the artist finds it next to run-browser\.
    made = None if args.dry_run or args.check else config.ensure_config_file()
    if made:
        out.line(made)
    # The model size of this run: the file edited while it downloads must not change what it fetches.
    for c in COMPONENTS:
        if isinstance(c, Qwen):
            c.pinned_quality = config.qwen_quality()
            c.pinned_text_encoder = config.qwen_text_encoder()
    if not args.check:      # (--check reports them itself)
        for problem in _config_problems(args):
            _wrapped(out, "NOTE: ", problem)
    selected = _selected(args)
    try:
        if args.check:
            return _check(out, selected, args)
        hub_open = not args.dry_run and locked(HUB_LOCK)
        if hub_open and background:
            selected = [c for c in selected if c.id != paths.HUB]   # the open hub runs from its venv
        if not background and not args.dry_run and not (hub_open or locked(INSTALL_LOCK)):
            _calibration_notice(out, wait=not args.yes)     # not before a run that only says "open / installing in
                                                            # another window"
        if not args.dry_run:
            models.ensure_folders()
            for venv_id in paths.ALL_VENV_IDS:
                if venvs.repair_if_moved(venv_id, base_python):
                    out.line(f"Repaired code\\venvs\\{venv_id}: the Spellcast3D folder was moved.")
        plan = _plan(selected, args)
        if args.if_needed and not plan:
            tracker.begin(_progress_start(selected, plan))
            if not args.dry_run:
                with contextlib.suppress(_Busy), _install_lock():    # (another installer at work: its turn)
                    install_profilers(Ctx(out, dry_run=False, base_python=base_python, progress=tracker), selected,
                                      log_header=f"Spellcast3D installer {' '.join(sys.argv[1:])}".rstrip())
                    _tidy(out)
            tracker.finish()
            return EXIT_OK
        if plan and hub_open and not background:
            out.line("Spellcast3D is open in another window, and its parts cannot be installed or updated "
                     "while they run. Close that window, then start again.")
            return EXIT_BUSY
        ctx = Ctx(out, dry_run=args.dry_run, base_python=base_python, progress=tracker,
                  verbose=args.verbose or os.environ.get("SPELLCAST_VERBOSE_INSTALL") == "1")
        if args.dry_run:
            return _run(ctx, selected, plan, args)
        with _install_lock():
            return _run(ctx, selected, plan, args)
    except _Busy:
        # The installer that holds the lock is the one writing the progress file: leave it alone.
        out.line("Spellcast3D is already being installed in another window. Wait for it to finish "
                 "(or close that window), then start again.")
        return EXIT_BUSY
    except KeyboardInterrupt:
        tracker.finish(error=STOPPED)
        out.line("\nInstallation stopped. Start Spellcast3D again to continue where it stopped.")
        return EXIT_FAILED
    except BaseException as e:
        tracker.finish(error=f"The installer stopped with an unexpected error: {e}. {LOG_HINT}")
        raise
    finally:
        for c in COMPONENTS:
            if isinstance(c, Qwen):
                c.pinned_quality = None
        out.close()


def _tidy(out: Output) -> None:
    """After the components (also when none needed work): NVIDIA's CUDA DLLs that Qwen-Image took from the cudart zip
    become links to PyTorch's once a venv has them (sdcpp.share_cuda_dlls), and a copy that came from an earlier
    release shows in its folder what today's release does (layout.tidy: tools\\update.bat runs this installer after
    its git steps). A few stat calls when there is nothing to do. Never fails the run."""
    try:
        sdcpp.share_cuda_dlls(out.line)
    except OSError as e:
        out.log_only(f"Could not link the CUDA files of stable-diffusion.cpp to PyTorch's ({e}); next time.")
    gone, problem = layout.tidy(git_exe())
    if gone:
        out.line(f"Tidied the Spellcast3D folder: it no longer shows {', '.join(gone)}, which only developers need. "
                 f"The license text is in code\\LICENSE.")
    if problem:
        out.log_only(f"Tidying the Spellcast3D folder: {problem}")


def locked(lock_file: Path) -> bool:
    """True while another process holds lock_file (the installer's or the running hub's lock)."""
    import msvcrt

    try:
        f = open(lock_file, "a+b")
    except OSError:
        return False   # no folder yet, or not ours to open: nothing is running from it
    with f:
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return True
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        return False


@contextlib.contextmanager
def _install_lock() -> Iterator[None]:
    """Held for the whole install. Windows releases it when the process dies, so it never goes stale."""
    import msvcrt

    paths.VENVS.mkdir(parents=True, exist_ok=True)
    with open(INSTALL_LOCK, "a+b") as f:
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise _Busy() from None
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def _progress_start(selected: list[Component], plan: dict[str, Pending]) -> dict[str, tuple[str, str]]:
    """The progress file's first picture: every component, pending / done / unsupported / skipped. The image generator
    spellcast3d-config.ini does not pick is not in it at all (not even as skipped)."""
    entries = state.load()
    ids = {c.id for c in selected}
    image = config.image_backend()
    start: dict[str, tuple[str, str]] = {}
    for c in COMPONENTS:
        entry = entries.get(c.id) or {}
        if c.id not in ids and c.id in IMAGE_BACKENDS and c.id != image:
            continue
        if c.id not in ids:
            start[c.id] = (progress.SKIPPED, "")
        elif c.id in plan:
            start[c.id] = (progress.PENDING, plan[c.id].reason)
        elif entry.get("status") == state.UNSUPPORTED:
            start[c.id] = (progress.UNSUPPORTED, entry.get("reason", ""))
        else:
            start[c.id] = (progress.DONE, "")
    return start


def _run(ctx: Ctx, selected: list[Component], plan: dict[str, Pending], args: argparse.Namespace) -> int:
    out = ctx.out
    todo = [c for c in selected if c.id in plan]
    start = _progress_start(selected, plan)
    # Qwen-Image waits while install_profilers, the run's last step, will add a [profiling] switch's package to its
    # venv: the hub must start none that gets PyTorch while it runs (the components before can take many minutes).
    if start.get(paths.QWEN, ("",))[0] == progress.DONE and (profilers := profilers_pending(selected)):
        start[paths.QWEN] = (progress.PENDING, profilers)
    ctx.progress.begin(start)   # first: the hub starts no tool until it knows this
    if ctx.dry_run:
        out.line("Dry run: the commands below are what the installer would run. Nothing is changed.")
    else:
        out.open_log(paths.log_file("install"), f"Spellcast3D installer {' '.join(sys.argv[1:])}".rstrip())
        for problem in _config_problems(args):     # shown on the console already
            out.log_only(f"NOTE: {problem}")
        _clear_install_tmp()   # what a killed run left there can be GBs (pip's half-finished downloads)
        if args.if_needed:
            _start_banner(out, todo, background=args.progress_file is not None)
    # Components this PC cannot run are only recorded as such: no GPU / disk / download checks for them.
    report = _preflight(out, [c for c in todo if _unsupported(c, args) is None], args.reinstall)
    if report.fatal and not ctx.dry_run:
        out.line(f"Nothing was installed. {LOG_HINT}")
        ctx.progress.finish(error=" ".join(report.fatal))
        return EXIT_FAILED

    entries = state.load()
    results: dict[str, tuple[str, str]] = {}
    for index, c in enumerate(selected, 1):
        if c.id not in plan:
            entry = entries.get(c.id) or {}
            results[c.id] = ((state.UNSUPPORTED, entry.get("reason", "")) if entry.get("status") == state.UNSUPPORTED
                             else (UP_TO_DATE, ""))
            continue
        _banner(out, f"[{index}/{len(selected)}] {c.title}", *textwrap.wrap(c.about, WIDTH - 2),
                f"Reason: {plan[c.id].reason}")
        ctx.progress.start(c.id)
        results[c.id] = _install_one(ctx, c, report, args, plan[c.id])
        ctx.progress.end(c.id, _PROGRESS_STATUS[results[c.id][0]], results[c.id][1])
        if c.id == paths.QWEN and results[c.id][0] == state.INSTALLED and (profilers := profilers_pending(selected)):
            ctx.progress.end(c.id, progress.PENDING, profilers)     # (it waits for install_profilers, as above)

    failed = any(status == state.FAILED for status, _ in results.values())
    # Last, so a developer's [profiling] switch never holds up a component nor takes the disk promised to one.
    profiled = install_profilers(ctx, [c for c in selected if results[c.id][0] in (state.INSTALLED, UP_TO_DATE)])
    if not ctx.dry_run:
        _tidy(out)
        _clear_install_tmp()
        _summary(out, selected, results)
    elif not plan and not profiled:
        out.line("Everything is installed and up to date; nothing would run.")
    ctx.progress.finish()
    return EXIT_FAILED if failed or report.fatal else EXIT_OK


def _start_banner(out: Output, todo: list[Component], background: bool) -> None:
    names = "Spellcast3D is installing: " + ", ".join(c.title for c in todo)
    if background:
        _banner(out, names + " (in the background)",
                "You can already work in the browser; each tool is ready as soon as its part is done.",
                "Keep this window open. The very first time takes an hour or more, mostly downloads.")
    elif sum(c.size_gb for c in todo) < LIGHT_GB:
        _banner(out, names, "This takes about a minute. Then Spellcast3D opens in your browser and installs",
                "its tools (the image and 3D generators) in the background while you work.")
    else:
        _banner(out, names, "This can take a long time (the very first start: an hour or more, mostly downloads).",
                "Keep this window open; Spellcast3D opens in your browser when it is done.")


CALIBRATION_NOTICE = (
    "CALIBRATING FOR PERFORMANCE.",
    "",
    "PLEASE CLOSE ANY HEAVY PROGRAMS (BLENDER, PHOTOSHOP, GAMES, VIDEOS,",
    "A BROWSER WITH MANY TABS ...) UNTIL THE INSTALLATION IS DONE.",
    "",
    "At the end of the installation, Spellcast3D measures your graphics card once,",
    "so that its 3D step runs as fast as it can on it. A program that uses the",
    "graphics card while it measures makes the measurement wrong (if the card is",
    "busy, it waits up to 15 minutes for it to be free).",
)
CALIBRATION_PROMPT = "   Left click here, then press Enter to continue: "


def _calibration_notice(out: Output, wait: bool = True) -> None:
    """The big notice at the start of a run, while Pixal3D is still to be calibrated on this PC's card (the first
    install, a new card, a new version of the calibration), before anything is installed: the calibration is the
    last step of the tools' install, an hour later in the background, when the artist may have gone for a coffee.
    Waits for Enter in an interactive console (not with `wait` False - install.py --yes - nor SPELLCAST_NO_PAUSE=1);
    quiet when nothing is due."""
    if pixal3d_calibration_work(_gpu_info()) is None:
        return
    out.line()
    out.line()
    out.line("#" * WIDTH)
    out.line()
    for line in CALIBRATION_NOTICE:
        out.line(f"   {line}" if line else "")
    out.line()
    out.line("#" * WIDTH)
    out.line()
    if not wait or os.environ.get("SPELLCAST_NO_PAUSE") == "1" or not sys.stdin or not sys.stdin.isatty():
        return
    _wait_for_enter(CALIBRATION_PROMPT)
    out.line()


def _wait_for_enter(prompt: str) -> None:
    """input(prompt), with the console's QuickEdit mode off while it waits (Windows): there a click into the window
    starts a text selection, and the Enter that follows would only end the selection, never reach the prompt."""
    handle = mode = None
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.windll.kernel32
            handle = k32.GetStdHandle(-10)                  # STD_INPUT_HANDLE
            current = wintypes.DWORD()
            if k32.GetConsoleMode(handle, ctypes.byref(current)):
                mode = current.value
                k32.SetConsoleMode(handle, (mode & ~0x0040) | 0x0080)   # no ENABLE_QUICK_EDIT_MODE (EXTENDED_FLAGS)
        except Exception:       # noqa: BLE001 - not a console: input() as it is
            mode = None
    try:
        input(prompt)
    except EOFError:
        pass
    finally:
        if mode is not None:
            import ctypes

            ctypes.windll.kernel32.SetConsoleMode(handle, mode)


def _clear_install_tmp() -> None:
    """Empty paths.INSTALL_TMP, the child processes' TEMP (see runner.Ctx.env). Best effort."""
    shutil.rmtree(paths.INSTALL_TMP, ignore_errors=True)
    try:
        paths.INSTALL_TMP.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass   # the first step that needs it reports the problem


def _unsupported(c: Component, args: argparse.Namespace) -> str | None:
    """Why c can never run on this PC (it is then recorded as 'unsupported', not as a failure), or None."""
    ok, why = c.supported(_gpu_info())
    return None if ok or (args.force_pixal3d and c.id == paths.PIXAL3D) else why


def _install_one(ctx: Ctx, c: Component, report: preflight.Report, args: argparse.Namespace,
                 pending: Pending) -> tuple[str, str]:
    out = ctx.out
    # Unsupported first: it is final (not retried at every start), a blocked component is not.
    unsupported = _unsupported(c, args)
    if unsupported:
        _wrapped(out, "Skipped: ", unsupported)
        return _record(ctx, c, state.UNSUPPORTED, unsupported)
    if not c.supported(_gpu_info())[0]:
        out.line(f"Installing anyway because of --force-pixal3d: {c.supported(_gpu_info())[1]}")
    blocked = report.blocked.get(c.id)
    if blocked:
        _wrapped(out, "Skipped: ", blocked)
        return _record(ctx, c, state.FAILED, blocked)
    started = time.monotonic()
    try:
        if pending.repair:
            c.repair(ctx)  # venv installed and unchanged: only fetch what is missing
        else:
            # Recorded first: a closed window or a killed process runs no handler, and an older
            # "installed" entry with the same fingerprint would otherwise hide a half-installed venv
            # for good. (An interrupted repair() needs no marker: its work is still to be done.)
            if not ctx.dry_run:
                state.record(c.id, state.FAILED, STOPPED, c.fingerprint())
            c.prepare(ctx)
            ctx.step("Preparing its Python environment", until=0.0)
            venvs.ensure(ctx, c.id, c.title, fresh=c.id in args.reinstall)
            c.install(ctx)
    except InstallError as e:
        out.line(f"{'Would fail' if ctx.dry_run else 'FAILED'}: {e.reason}")
        return _record(ctx, c, state.FAILED, e.reason)
    except KeyboardInterrupt:
        _record(ctx, c, state.FAILED, STOPPED)
        raise
    except Exception as e:  # a bug or an unexpected OS error: keep going with the other components
        out.line(traceback.format_exc().rstrip())
        reason = f"Unexpected error while installing {c.title}: {e}. {LOG_HINT}"
        out.line(f"FAILED: {reason}")
        return _record(ctx, c, state.FAILED, reason)
    if not ctx.dry_run:
        out.line(f"{c.title} is ready ({_duration(time.monotonic() - started)}).")
    return _record(ctx, c, state.INSTALLED, "")


def _record(ctx: Ctx, c: Component, status: str, reason: str) -> tuple[str, str]:
    # Fingerprint taken now, after the install, so a submodule it just checked out is included.
    if not ctx.dry_run:
        state.record(c.id, status, reason, c.fingerprint())
    return status, reason


def _preflight(out: Output, todo: list[Component], reinstall: tuple[str, ...] = ()) -> preflight.Report:
    report = preflight.run(todo, _gpu_info(), network=True, reinstall=reinstall)
    out.line("System check:")
    for line in report.info:
        out.line(f"  {line}")
    for line in report.warnings:
        _wrapped(out, "  WARNING: ", line)
    titles = {c.id: c.title for c in todo}
    by_reason: dict[str, list[str]] = {}
    for tool_id, reason in report.blocked.items():
        by_reason.setdefault(reason, []).append(titles[tool_id])
    for reason, names in by_reason.items():
        _wrapped(out, f"  PROBLEM for {', '.join(names)}: ", reason)
    for line in report.fatal:
        _wrapped(out, "  ERROR: ", line)
    return report


def _check(out: Output, selected: list[Component], args: argparse.Namespace) -> int:
    out.line("Spellcast3D system check (nothing is changed).")
    _wrapped(out, f"Image generator ({config.config_file().name}): ", config.describe())
    for problem in config.problems():
        _wrapped(out, "  PROBLEM: ", problem)
    entries = state.load()
    plan = _plan(selected, args)
    report = _preflight(out, [c for c in selected if c.id in plan and _unsupported(c, args) is None], args.reinstall)
    out.line("Components:")
    for c in selected:
        entry = entries.get(c.id) or {}
        unsupported = _unsupported(c, args) if c.id in plan else None
        if unsupported:
            text = f"not supported on this PC (not recorded yet): {unsupported}"
        elif c.id in plan:
            text = f"needs {'repairing' if plan[c.id].repair else 'installing'} ({plan[c.id].reason})"
            if c.id in report.blocked:
                text += " - but cannot be installed right now, see PROBLEM above"
        elif entry.get("status") == state.UNSUPPORTED:
            text = f"not supported on this PC: {entry.get('reason', '')}"
        else:
            text = "installed and up to date"
        _wrapped(out, f"  {c.id:<8} ", text)
    profilers = missing_profilers(selected)     # (not a component: never blocked, see install_profilers)
    if profilers:
        _wrapped(out, "  For developers: ", f"{profilers_work(profilers)}, added at the next start.")
    if report.fatal or (_config_problems(args) and config.image_backend() is None):
        return EXIT_FAILED      # (a mistake in spellcast3d-config.ini: no image generator can be installed)
    return EXIT_PENDING if plan or profilers else EXIT_OK


def _summary(out: Output, selected: list[Component], results: dict[str, tuple[str, str]]) -> None:
    labels = {state.INSTALLED: "installed", state.FAILED: "FAILED", state.UNSUPPORTED: "not supported on this PC",
              UP_TO_DATE: UP_TO_DATE}
    _banner(out, "Summary")
    for c in selected:
        status, reason = results[c.id]
        _wrapped(out, f"  {c.title:<16} ", labels[status] + (f": {reason}" if reason else ""))
    out.line(f"  Full log: {paths.log_file('install')}")
    if any(status == state.FAILED for status, _ in results.values()):
        out.line("  Start Spellcast3D again to retry the parts that failed.")
    out.line("=" * WIDTH)


def _banner(out: Output, *lines: str) -> None:
    out.line()
    out.line("=" * WIDTH)
    for line in lines:
        out.line(f" {line}")
    out.line("=" * WIDTH)


def _wrapped(out: Output, prefix: str, text: str) -> None:
    indent = " " * (len(prefix) if len(prefix) <= 20 else 4)
    for line in textwrap.wrap(text, WIDTH, initial_indent=prefix, subsequent_indent=indent):
        out.line(line)


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes} min {secs} s" if minutes else f"{secs} s"
