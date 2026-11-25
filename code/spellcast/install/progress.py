"""install.py --progress-file: what the running installer is doing, as JSON for the hub. STDLIB ONLY.

    {"running": bool, "finished": bool, "error": str|null, "current": component id|null,
     "pid": int, "updated": unix time,
     "components": {id: {"status": "pending"|"running"|"done"|"failed"|"unsupported"|"skipped",
                         "step": str, "fraction": float|null, "step_fraction": float|null,
                         "bytes_done": int|null, "bytes_total": int|null, "message": str}}}

`fraction` is the component's whole install (0..1, never going back); `step` says in artist words what it
does now (with " · what the step does right now" when the step tells: Progress.detail), and `step_fraction` how far
that step is (0..1) when it tells; `bytes_done` / `bytes_total` are the current download when it is known: pip's raw progress, or
how far Pixal3D's AI models folder has grown towards its size. `message` is the latest output line while it
runs, and the reason when it failed, is unsupported or is pending. `error` is a problem that stopped the
whole run (e.g. no internet); per-component failures are in their `message`.

Written atomically (paths.write_text_atomic) when the run begins and ends and every second in between
(`updated` moves on); a file whose `running` is true but whose `updated` stands still belongs to an installer
that was killed.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .. import paths

PENDING, RUNNING, DONE, FAILED, UNSUPPORTED, SKIPPED = "pending", "running", "done", "failed", "unsupported", "skipped"
# The step text of Pixal3D's install while it measures the graphics card (components.Pixal3D._calibrate): the hub
# (procman.install_wait) lets no job take the card meanwhile, so the measurement is of a quiet card.
CALIBRATING_STEP = "Calibrating Pixal3D for your graphics card"
TICK = 1.0            # seconds between two writes while the installer runs
MESSAGE_CHARS = 300

Measure = Callable[[], tuple[int, int]]   # () -> (bytes done, bytes expected)


class Progress:
    """The progress file of one installer run. With path None it only keeps the state (nothing is written)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()      # the state
        self._io = threading.Lock()        # one write at a time, so an older state never lands last
        self._data: dict[str, Any] = {"running": False, "finished": False, "error": None, "current": None,
                                      "pid": os.getpid(), "updated": 0.0, "components": {}}
        self._step_from = self._step_until = 0.0
        self._step_label = ""
        self._measure: Measure | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- the run --------------------------------------------------------------------------------------
    def begin(self, components: dict[str, tuple[str, str]]) -> None:
        """components: {id: (status, message)} for every component, in install order."""
        with self._lock:
            self._data["running"] = True
            self._data["components"] = {cid: _entry(status, message) for cid, (status, message) in components.items()}
        self._write()
        if self.path is not None and self._thread is None:
            self._thread = threading.Thread(target=self._heartbeat, name="install-progress", daemon=True)
            self._thread.start()

    def finish(self, error: str | None = None) -> None:
        """The run is over: components still pending or running are marked failed (with `error` when given)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._lock:
            for entry in self._data["components"].values():
                if entry["status"] in (PENDING, RUNNING):
                    entry.update(status=FAILED, message=error or entry["message"])
            self._data.update(running=False, finished=True, error=error, current=None)
            self._measure = None
        self._write()

    # ---- one component --------------------------------------------------------------------------------
    def start(self, cid: str) -> None:
        with self._lock:
            self._data["current"] = cid
            self._data["components"][cid] = {**_entry(RUNNING, ""), "fraction": 0.0}
            self._step_from = self._step_until = 0.0
            self._measure = None

    def end(self, cid: str, status: str, message: str = "") -> None:
        with self._lock:
            entry = self._data["components"].setdefault(cid, _entry(status, message))
            entry.update(status=status, message=message, step="", step_fraction=None, bytes_done=None,
                         bytes_total=None, fraction=1.0 if status == DONE else entry.get("fraction"))
            if self._data["current"] == cid:
                self._data["current"] = None
            self._measure = None

    def step(self, label: str, until: float, measure: Measure | None = None) -> None:
        """The current component starts a step that ends at `until` of its whole install (0..1).

        measure, when given, is called about once a second for (bytes done, bytes expected) of the step.
        """
        with self._lock:
            entry = self._current()
            if entry is None:
                return
            self._step_from = max(self._step_until, entry["fraction"] or 0.0)
            self._step_until = max(until, self._step_from)
            self._step_label = label
            self._measure = measure
            entry.update(step=label, step_fraction=None, bytes_done=None, bytes_total=None, fraction=self._step_from)

    def detail(self, text: str, fraction: float | None = None) -> None:
        """What the current step does now (shown after the step's label: "Calibrating ... · timing the attention
        kernels") and, when given, how far the step is (0..1 of the step; the component's fraction never goes back)."""
        with self._lock:
            entry = self._current()
            if entry is None:
                return
            label = self._step_label or entry.get("step") or ""
            entry["step"] = f"{label} · {text.strip()}"[:MESSAGE_CHARS] if text.strip() else label
            if fraction is not None:
                entry["step_fraction"] = round(max(entry.get("step_fraction") or 0.0, min(1.0, max(0.0, fraction))), 4)
                share = self._step_from + (self._step_until - self._step_from) * entry["step_fraction"]
                entry["fraction"] = round(max(entry["fraction"] or 0.0, share), 4)

    def download(self, done: int, total: int) -> None:
        """Bytes of the step's current download (pip's raw progress, or a measure)."""
        with self._lock:
            entry = self._current()
            if entry is None or total <= 0:
                return
            done = max(0, min(done, total))
            share = self._step_from + (self._step_until - self._step_from) * done / total
            entry.update(bytes_done=done, bytes_total=total, fraction=round(max(entry["fraction"] or 0.0, share), 4))

    def note(self, line: str) -> None:
        """The latest output line of the current component."""
        line = line.strip()
        if not line:
            return
        with self._lock:
            entry = self._current()
            if entry is not None:
                entry["message"] = line[:MESSAGE_CHARS]

    # ---- writing --------------------------------------------------------------------------------------
    def _current(self) -> dict[str, Any] | None:
        cid = self._data["current"]
        return self._data["components"].get(cid) if cid else None

    def _heartbeat(self) -> None:
        while not self._stop.wait(TICK):
            measure = self._measure
            if measure is not None:
                try:
                    self.download(*measure())
                except Exception:   # cosmetic: a failed measurement must not stop the heartbeat
                    pass
            self._write()

    def _write(self) -> None:
        if self.path is None:
            return
        with self._io:
            with self._lock:
                self._data["updated"] = round(time.time(), 3)
                text = json.dumps(self._data, indent=1) + "\n"
            try:
                paths.write_text_atomic(self.path, text)
            except OSError:
                pass    # the hub held the file a moment too long: the next tick writes it again


def _entry(status: str, message: str) -> dict[str, Any]:
    return {"status": status, "step": "", "fraction": 1.0 if status == DONE else None, "step_fraction": None,
            "bytes_done": None, "bytes_total": None, "message": message}
