"""Console + install.log output, and child processes streamed through both. STDLIB ONLY."""

from __future__ import annotations

import codecs
import collections
import datetime
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TextIO

from .. import paths
from .progress import Measure, Progress

LOG_ROTATE_BYTES = 20_000_000

# `pip install --progress-bar raw` prints "Progress <done> of <total>" about once a second.
_PIP_PROGRESS = re.compile(r"^Progress (\d+) of (\d+)$")
# What a full disk looks like in pip / git / Python output. pip often reports it only as an interrupted
# download instead ("Connection interrupted while downloading"), so the failure texts mention free space too.
_DISK_FULL = re.compile(r"No space left on device|\[Errno 28\]|\[WinError 112\]|not enough space on the disk",
                        re.IGNORECASE)


def pip_cache_dir() -> Path:
    """Where pip keeps its downloads for every venv (PIP_CACHE_DIR if the user set one)."""
    if os.environ.get("PIP_CACHE_DIR"):
        return Path(os.environ["PIP_CACHE_DIR"])
    local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(local) / "pip" / "cache"


def disk_full_reason() -> str:
    drives = sorted({p.drive.upper() or str(p) for p in (paths.ROOT, pip_cache_dir())})
    return (f"A disk is full (drive {' or '.join(drives)}). Free up some space, then start Spellcast3D again "
            f"(finished downloads are kept). Details are in code\\logs\\install.log.")


class InstallError(Exception):
    """A step failed. `reason` is shown to the artist (console summary and the hub UI)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Output:
    """Prints to the console and appends to code/logs/install.log once open_log() was called.

    transient() shows a progress line that the next output overwrites; it only reaches
    an interactive console and never the log. transient=False turns those lines off (the installer
    the hub runs in the background: it shares the hub's console, and its progress goes to the hub).
    """

    def __init__(self, transient: bool = True) -> None:
        self._log: TextIO | None = None
        self._transient_len = 0
        self._tty = transient and sys.stdout.isatty()

    def open_log(self, path: Path, header: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > LOG_ROTATE_BYTES:
            os.replace(path, path.with_name(path.stem + ".previous.log"))
        self._log = open(path, "a", encoding="utf-8", errors="replace")
        stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        self._log.write(f"\n{'=' * 78}\n{header} - {stamp}\n{'=' * 78}\n")
        self._log.flush()

    def line(self, text: str = "") -> None:
        self._clear_transient()
        print(text, flush=True)
        if self._log is not None:
            self._log.write(text + "\n")
            self._log.flush()

    def log_only(self, text: str) -> None:
        if self._log is not None:
            self._log.write(text + "\n")
            self._log.flush()

    def transient(self, text: str) -> None:
        if not self._tty:
            return
        width = max(20, shutil.get_terminal_size((100, 20)).columns - 1)
        text = text.replace("\t", " ")[:width]
        sys.stdout.write("\r" + text + " " * max(0, self._transient_len - len(text)))
        sys.stdout.flush()
        self._transient_len = len(text)

    def _clear_transient(self) -> None:
        if self._transient_len:
            sys.stdout.write("\r" + " " * self._transient_len + "\r")
            self._transient_len = 0

    def close(self) -> None:
        self._clear_transient()
        if self._log is not None:
            self._log.close()
            self._log = None


def _human_bytes(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


SATISFIED = "Requirement already satisfied:"      # pip's line for a package the venv has (one per package)


@dataclass
class Ctx:
    """What every install step needs: where to print, whether to only print the plan, and where to report
    progress (install.py --progress-file; a Progress without a file keeps it to itself)."""
    out: Output
    dry_run: bool
    base_python: Path
    progress: Progress = field(default_factory=lambda: Progress(None))
    # Every line of the tools' installers on the console, pip's "Requirement already satisfied:" ones too (install.py
    # --verbose, SPELLCAST_VERBOSE_INSTALL=1); without it those are logged, counted and said as one transient line.
    verbose: bool = False
    _satisfied: int = field(default=0, init=False, repr=False)
    # The last lines of the running command's output (to tell a full disk from a network failure).
    _tail: collections.deque[str] = field(default_factory=lambda: collections.deque(maxlen=50), init=False,
                                          repr=False)
    # Output texts a note is printed after (see run()), and whether the running command printed them.
    _seen: dict[str, bool] = field(default_factory=dict, init=False, repr=False)
    # run(on_line=...): told every complete line of the running command (the calibration's progress lines).
    _on_line: Callable[[str], None] | None = field(default=None, init=False, repr=False)

    def step(self, label: str, until: float, measure: Measure | None = None) -> None:
        """The current component starts a step (artist words) that ends at `until` of its install (0..1)."""
        self.progress.step(label, until, measure)

    def env(self, venv_id: str | None, extra: dict[str, str] | None = None) -> dict[str, str]:
        # pip's normal per-user cache (%LOCALAPPDATA%\pip\cache) on purpose: the 3.5 GB torch wheel is
        # downloaded once for all venvs, wheels the user already has from other AI tools are reused, and
        # a reinstall or an update never downloads them again. A user-level PIP_NO_CACHE_DIR would
        # switch that cache off.
        env = paths.clean_env(venv_id, extra)
        env.pop("PIP_NO_CACHE_DIR", None)
        # Temporary files on the install drive (paths.INSTALL_TMP), which preflight budgets: pip keeps
        # two temporary copies of each download there (6.9 GB for torch), too much for a nearly full
        # C: when Spellcast3D is on D:. TMPDIR first: Python's tempfile looks at it first.
        tmp = str(paths.INSTALL_TMP)
        env.update(TMPDIR=tmp, TEMP=tmp, TMP=tmp)
        return env

    def run(self, argv: list[str], *, cwd: Path, env: dict[str, str], fail: str,
            notes: tuple[tuple[str, str], ...] = (), on_line: Callable[[str], None] | None = None) -> None:
        """Run a child to completion, streaming its output; raise InstallError(fail) if it fails.

        on_line(text) is told every complete line of its output (not the transient progress-bar updates).

        notes: (text, note) pairs; the note is printed after the command's output when a line of it
        contained the text (to comment on something a tool prints).
        """
        cmdline = subprocess.list2cmdline(argv)
        if self.dry_run:
            self.out.line(f"  [in {cwd}]")
            self.out.line(f"  > {cmdline}")
            return
        self.out.line(f"> {cmdline}")
        self.out.log_only(f"  (working folder: {cwd})")
        self._tail.clear()
        self._seen = {text: False for text, _note in notes}
        self._on_line = on_line
        self._satisfied = 0
        try:
            code = self._stream(argv, cwd, env)
        except OSError as e:
            self.out.line(f"Could not start the command: {e}")
            raise InstallError(fail) from e
        finally:
            for text, note in notes:
                if self._seen.get(text):
                    self.out.line(note)
            self._seen = {}
        if code != 0:
            self.out.line(f"The command above failed with exit code {code}.")
            if any(_DISK_FULL.search(line) for line in self._tail):
                raise InstallError(disk_full_reason())
            raise InstallError(fail)

    def probe(self, argv: list[str], *, cwd: Path, env: dict[str, str]) -> bool:
        """True when the command exits 0. Output goes to the log only. Always False in dry-run."""
        if self.dry_run:
            self.out.line(f"  > {subprocess.list2cmdline(argv)}   (a check; the next step runs only if it fails)")
            return False
        try:
            res = subprocess.run(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", timeout=600)
        except (OSError, subprocess.TimeoutExpired) as e:
            self.out.log_only(f"check {subprocess.list2cmdline(argv)} -> {e}")
            return False
        self.out.log_only(f"check {subprocess.list2cmdline(argv)} -> exit {res.returncode}\n"
                          f"{(res.stdout + res.stderr).strip()}")
        return res.returncode == 0

    def pip(self, venv_id: str, *args: str, fail: str) -> None:
        py = paths.venv_python(venv_id)
        self.run([str(py), "-m", "pip", "install", "--progress-bar", "raw", *args],
                 cwd=paths.venv_dir(venv_id), env=self.env(venv_id), fail=fail)

    def _stream(self, argv: list[str], cwd: Path, env: dict[str, str]) -> int:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        try:
            while True:
                chunk = proc.stdout.read1(65536)
                pending = self._emit(pending + decoder.decode(chunk, final=not chunk), final=not chunk)
                if not chunk:
                    break
            return proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise

    def _emit(self, buf: str, final: bool) -> str:
        """Print every complete line of buf and return the unfinished rest.

        "\\r\\n" and "\\n" end a line. A lone "\\r" (progress bars of pip, tqdm, git) ends a
        transient console update that the next one overwrites.
        """
        start = 0
        while True:
            nl = buf.find("\n", start)
            cr = buf.find("\r", start)
            if cr != -1 and (nl == -1 or cr < nl):
                if cr + 1 == len(buf) and not final:
                    break  # maybe the first half of "\r\n": wait for more output
                if buf.startswith("\r\n", cr):
                    self._line(buf[start:cr])
                    start = cr + 2
                else:
                    if buf[start:cr].strip():
                        self.out.transient(buf[start:cr].strip())
                    start = cr + 1
            elif nl != -1:
                self._line(buf[start:nl])
                start = nl + 1
            else:
                break
        rest = buf[start:]
        if final and rest:
            self._line(rest)
            return ""
        return rest

    def _line(self, text: str) -> None:
        if not self.verbose and text.startswith(SATISFIED):
            # pip checking every pinned package of a venv that has them (an update of one tool runs its install
            # again): hundreds of these at a start spooked artists. Logged, counted, one transient line instead.
            self._satisfied += 1
            self.out.log_only(text.rstrip())
            self.out.transient(f"    verifying the installation... ({self._satisfied} packages already there)")
            self.progress.note(f"Verifying the installation ({self._satisfied} packages already there)")
            return
        m = _PIP_PROGRESS.match(text.strip())
        if m is None:
            self.out.line(text.rstrip())
            self._tail.append(text)
            self.progress.note(text)
            for wanted in self._seen:
                self._seen[wanted] = self._seen[wanted] or wanted in text
            if self._on_line is not None:
                try:
                    self._on_line(text)
                except Exception:       # noqa: BLE001 - a hook's slip must not stop the install
                    pass
            return
        done, total = int(m.group(1)), int(m.group(2))
        self.progress.download(done, total)
        if total > 0:
            self.out.transient(f"    downloaded {_human_bytes(done)} of {_human_bytes(total)} "
                               f"({done * 100 // total}%)")
