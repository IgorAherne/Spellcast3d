"""One bake or one model import, in a child process of the Instant Meshes launcher.

    python bake_worker.py <request.json>

It runs in the imesh venv and imports only its sibling package spellbake and instant_meshes_brush,
never the hub. The request is one of

    {"cmd": "bake", "high": <model file>, "low": <low-poly OBJ or GLB>, "out_dir": <.../bake_NNN.tmp>,
     "final_dir": <.../bake_NNN>, "settings": {...}, "threads": N}
    {"cmd": "import", "src": <model file>, "dst": <.glb to write>, "overrides": {...}}

It prints JSON lines on stdout: {"progress": 0..1, "message": str} while it works, then
{"result": {...}} and exit code 0, or {"error": "<message for an artist>", "kind": ...} and exit code 1.
kind is "user" (the input's fault), "cancelled" or "internal" (a bug; the traceback goes to stderr,
which the launcher logs).

A bake writes into out_dir and renames it to final_dir once every file is written, so a folder
named bake_NNN is always complete. Its result names the files relative to that folder (see
spellbake.bake.BakeResult.to_json). A bake that fails or is cancelled deletes out_dir. It is
cancelled when a file named ``cancel`` appears in out_dir; the launcher usually just kills the
process tree instead.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, TextIO

from spellbake import BakeError
from spellbake.bake import BakeCancelled, bake
from spellbake.importer import import_model
from spellbake.settings import BakeSettings

CANCEL_FILE = "cancel"
# Progress lines at most this often (a bake reports thousands of steps).
PROGRESS_INTERVAL = 0.25
# A just-written folder can be held briefly by a virus scanner or the search indexer.
RENAME_ATTEMPTS = 40


class RequestError(Exception):
    """The request itself is malformed: a bug in whoever wrote it."""


class Output:
    """The JSON-lines channel to the launcher (thread-safe, progress throttled)."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self._last = 0.0

    def send(self, message: dict[str, Any]) -> None:
        with self._lock:
            self._stream.write(json.dumps(message) + "\n")
            self._stream.flush()

    def progress(self, fraction: float, message: str) -> None:
        now = time.monotonic()
        if now - self._last >= PROGRESS_INTERVAL or fraction >= 1.0:
            self._last = now
            self.send({"progress": round(fraction, 4), "message": message})


def _path(request: dict[str, Any], key: str) -> Path:
    value = request.get(key)
    if not isinstance(value, str) or not value:
        raise RequestError(f"the request has no '{key}' path")
    return Path(value)


def _rename(source: Path, target: Path) -> None:
    for attempt in range(RENAME_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == RENAME_ATTEMPTS - 1:
                raise
            time.sleep(0.05)


def run_bake(request: dict[str, Any], out: Output) -> dict[str, Any]:
    out_dir, final_dir = _path(request, "out_dir"), _path(request, "final_dir")
    threads = request.get("threads")
    if threads is not None and (not isinstance(threads, int) or isinstance(threads, bool) or threads < 1):
        raise RequestError("'threads' must be a positive whole number")
    cancel_file = out_dir / CANCEL_FILE
    try:
        settings = BakeSettings.from_json(request.get("settings") or {})
        result = bake(_path(request, "high"), _path(request, "low"), out_dir, settings, progress=out.progress,
                      cancelled=cancel_file.exists, threads=threads)
        if final_dir.exists():
            raise RequestError(f"{final_dir} exists already")
        _rename(out_dir, final_dir)
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return result.to_json()


def run_import(request: dict[str, Any], out: Output) -> dict[str, Any]:
    overrides = request.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise RequestError("'overrides' must be a JSON object")
    return import_model(_path(request, "src"), _path(request, "dst"), overrides, progress=out.progress)


COMMANDS = {"bake": run_bake, "import": run_import}
FAILED = {"bake": "The bake failed", "import": "Converting the model failed"}


def main(argv: list[str], stdout: TextIO = sys.stdout) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s: [%(name)s] %(message)s")
    out = Output(stdout)
    command = "bake"
    try:
        if len(argv) != 1:
            raise RequestError("usage: python bake_worker.py <request.json>")
        request = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
        if not isinstance(request, dict) or request.get("cmd") not in COMMANDS:
            raise RequestError(f"'cmd' must be one of {', '.join(COMMANDS)}")
        command = request["cmd"]
        result = COMMANDS[command](request, out)
    except BakeCancelled:
        out.send({"error": "The bake was cancelled.", "kind": "cancelled"})
        return 1
    except BakeError as exc:
        out.send({"error": str(exc), "kind": "user"})
        return 1
    except MemoryError:
        out.send({"error": "The computer ran out of memory. Close other programs, or choose a smaller texture "
                           "size or the Fast quality, and try again.", "kind": "user"})
        return 1
    except Exception as exc:  # a bug: the artist gets one line, the log the traceback
        traceback.print_exc(file=sys.stderr)
        out.send({"error": f"{FAILED[command]} ({type(exc).__name__}: {exc}). The Instant Meshes log has the "
                           f"details.", "kind": "internal"})
        return 1
    out.send({"result": result})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
