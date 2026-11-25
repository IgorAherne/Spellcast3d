"""Batches for pixal3d_server.py: all the props of a hub job in one request, each part of the work for every prop.

Pixal3D's staged engine (pixal3d/staged in the fork) takes the props of a batch through its parts - cut-out,
camera, picture features, rough shape, shape, details, texture, mesh - and reads each part's network from disk
when the part starts, so one network is on the card at a time. This file keeps what the hub sees of that: the
record of a batch (Batch: where every prop is, written from the engine's events and from the bars of the mesh
clean-up), the one batch that may run and the last one that ended (BatchRegistry), the parts with their words
and their share of a prop's bar (PARTS), the words for a network being read (network_label), the kind of a
failure (classify) and the fingerprint of a prop's request (request_hash), by which a launcher that was started
again picks up a prop whose sampling was already done.

A record is written on the GPU thread and read by the web server's threads, always under its lock. `seq` grows
with every change, down to each bar update: the hub takes a `seq` that stands still for long as "Pixal3D hangs".
The prop on the card and the last one made also go to a small file in the batch's work folder (ON_CARD), for the
hub to read when Pixal3D stopped without a last answer.

Only the stdlib at module level: the hub's tests import this file without torch, numpy or PIL.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple

import pixal3d_mv                                           # launchers/ (this script's folder); stdlib only


class Part(NamedTuple):
    key: str          # the engine's name of the part (pixal3d.staged PARTS)
    label: str        # the artist's words while a prop is in it
    start: float      # where the prop's own bar stands when the part begins ...
    end: float        # ... and when it is done; a prop that waits for its next part stays here


PARTS: tuple[Part, ...] = (
    Part("cutout", "Removing the background", 0.00, 0.03),
    Part("camera", "Estimating the camera angle", 0.03, 0.05),
    Part("features", "Reading the picture", 0.05, 0.06),
    Part("ss", "Blocking out the shape", 0.06, 0.15),
    Part("cond_lr", "Preparing the shape", 0.15, 0.16),
    Part("lr", "Sculpting the shape", 0.16, 0.29),
    Part("upsample", "Sculpting the shape", 0.29, 0.30),
    Part("cond_hr", "Preparing the details", 0.30, 0.32),
    Part("hr", "Refining shape details", 0.32, 0.55),
    Part("tex", "Painting the texture", 0.55, 0.72),
    Part("finish", "Building the 3D mesh", 0.72, 1.00),
)
PART = {part.key: part for part in PARTS}
PART_KEYS = tuple(part.key for part in PARTS)
FINISH = PART["finish"]

# A prop whose shape Hi3D makes goes through its own parts in place of cond_lr, lr, upsample and hr. Their words,
# and each part's share of the prop's time on an RTX 3060 (the boy at 1536 after the fifth speed round: Hi3D's
# passes are four fifths of it).
HI3D_ORDER = ("cutout", "camera", "features", "ss", "hi3d_cond", "hi3d_p1", "hi3d_d1", "hi3d_p2", "hi3d_d2",
              "hi3d_p3", "hi3d_d3", "hi3d_encode", "cond_hr", "tex", "finish")
HI3D_LABELS = {"hi3d_cond": "Reading the picture", "hi3d_p1": "Sculpting the shape", "hi3d_d1": "Building the shape",
               "hi3d_p2": "Refining shape details", "hi3d_d2": "Building the details",
               "hi3d_p3": "Refining fine details", "hi3d_d3": "Building the fine details",
               "hi3d_encode": "Preparing the texture"}
HI3D_SECONDS = {"cutout": 3, "camera": 2, "features": 1, "ss": 12, "hi3d_cond": 1, "hi3d_p1": 7, "hi3d_d1": 1,
                "hi3d_p2": 44, "hi3d_d2": 3, "hi3d_p3": 145, "hi3d_d3": 8, "hi3d_encode": 2, "cond_hr": 3,
                "tex": 30, "finish": 45}
# Every part of either geometry, in the engine's order (pixal3d.staged PARTS).
ALL_KEYS = ("cutout", "camera", "features", "ss", "cond_lr", "lr", "upsample", *HI3D_ORDER[4:12], "cond_hr", "hr",
            "tex", "finish")


def hi3d_parts(passes: int) -> tuple[Part, ...]:
    """The parts of a Hi3D prop with `passes` passes (2 for 1024, 3 for 1536), shares laid out by HI3D_SECONDS."""
    keys = [key for key in HI3D_ORDER if not (key[:-1] in ("hi3d_p", "hi3d_d") and int(key[-1]) > passes)]
    total = float(sum(HI3D_SECONDS[key] for key in keys))
    parts, start = [], 0.0
    for key in keys:
        end = 1.0 if key == keys[-1] else start + HI3D_SECONDS[key] / total
        parts.append(Part(key, HI3D_LABELS.get(key) or PART[key].label, round(start, 4), round(end, 4)))
        start = end
    return tuple(parts)


def parts_of(geometry: str, resolution: int) -> tuple[Part, ...]:
    return hi3d_parts(2 if resolution <= 1024 else 3) if geometry == "hi3d" else PARTS


def _moved(fraction: float, part: Part) -> float:
    """A fraction of a Pixal3D prop's bar inside `part` (what the hooks' bars and DECODED say) at the same place
    inside the share `part` has in its own prop's table."""
    base = PART.get(part.key)
    if base is None or base == part or base.end <= base.start:
        return fraction
    return part.start + (fraction - base.start) / (base.end - base.start) * (part.end - part.start)
# Inside "finish": the engine's own steps (shape decode, texture decode, holes) end here; the bars of the mesh
# clean-up (pixal3d_server.PHASES) and the finish hook's own steps take the rest.
DECODED = 0.82
DECODE_STEPS = 3

WAITING, WORKING, RESTING, DONE, FAILED, CANCELLED = "waiting", "working", "resting", "done", "failed", "cancelled"
ENDED = (DONE, FAILED, CANCELLED)
RUNNING = "running"
WAITING_MESSAGE = "Waiting for its turn"
RESTING_MESSAGE = "Waiting for the next step"
NOT_MADE = "Pixal3D did not get to this prop. Try making it again."

# The engine's network names (before the ".sv" / ".mv" of the four that differ) -> the artist's words.
NETWORKS = {"rembg": "background removal model", "moge": "camera model", "dino": "picture model",
            "naf": "picture detail model", "ss_flow": "rough shape model", "ss_dec": "rough shape decoder",
            "lr_flow": "shape model", "hr_flow": "detail model", "tex_flow": "texture model",
            "shape_dec": "mesh decoder", "tex_dec": "texture decoder", "hi3d_dino": "Hi3D picture model",
            "hi3d_clip": "Hi3D picture model (CLIP)",
            "hi3d_t512": "Hi3D shape model", "hi3d_t1024": "Hi3D detail model", "hi3d_vae": "Hi3D mesh decoder",
            "shape_enc": "shape encoder"}

OOM_TEXTS = ("out of memory", "alloc_failed", "cuda error: 2[", "cudamalloc")
RAM_TEXTS = ("paging file", "defaultcpuallocator")

# {"batch_id", "current": the prop on the card | null, "finished": {"id", "result"} of the last prop made | null},
# in the batch's work folder, replaced whole whenever one of the two changes (Batch.marker). No poll is answered
# while native code saves a mesh, so the hub's last answer can be older than the prop that was on the card when
# Pixal3D died; this file is not. adapters/pixal3d.py ON_CARD: keep the name in sync.
ON_CARD = "on_card.json"


def network_label(name: str) -> str:
    """ "hr_flow.mv" -> "detail model (multi-view)": what the hub puts after "Loading the"."""
    base, _, kind = str(name).partition(".")
    label = NETWORKS.get(base, "model")
    return f"{label} (multi-view)" if kind == "mv" else label


def classify(exc: BaseException) -> str:
    """The kind of a failure: "input" | "mv_missing" | "oom" | "ram" | "disk" | "missing_file" | "restart" |
    "stopped" | "error". The staged engine's own exceptions (PropFailed, BatchBroken) say theirs. Out of video
    memory is told by PyTorch's exception and by the texts of the native libraries that allocate on the card
    themselves: cuDNN and cuBLAS ("..._ALLOC_FAILED") and nvdiffrast, whose check prints the CUDA error number
    and the call ("Cuda error: 2[cudaMalloc(&m_gpuPtr, bytes);]", read in the installed 0.4.0). Out of RAM comes
    as MemoryError, as torch's CPU allocator's text, or as Windows' "The paging file is too small".

    The ladder below the engine's own exceptions is the engine's (pixal3d/staged/runner.py classify), step for
    step, so one cause cannot come out as two different messages in one job. Two differences on purpose: the
    failures of the launcher's own hooks come first, and torch's OutOfMemoryError is told by its name, because
    this file must import nothing but the stdlib.
    """
    if isinstance(exc, pixal3d_mv.ViewProblem):
        return "input"
    if isinstance(exc, pixal3d_mv.MultiviewMissing):
        return "mv_missing"
    own = getattr(exc, "kind", None)
    if type(exc).__name__ in ("PropFailed", "BatchBroken") and isinstance(own, str) and own:
        return own
    if type(exc).__name__ == "OutOfMemoryError":
        return "oom"
    if isinstance(exc, MemoryError):
        return "ram"
    if isinstance(exc, FileNotFoundError):
        return "missing_file"
    low = str(exc).lower()
    if any(text in low for text in RAM_TEXTS):
        return "ram"
    if isinstance(exc, RuntimeError) and any(text in low for text in OOM_TEXTS):
        return "oom"
    return "error"


def request_hash(**defining: Any) -> str:
    """The fingerprint of everything that decides a prop's latents (seed, quality, sampler settings, the
    pictures' sizes and change times, the views' rectangles and directions): SHA-256 of their canonical JSON.
    The engine keeps it with a prop's checkpoint; a batch sent again after a restart resumes the prop only
    when it is the same."""
    text = json.dumps(defining, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_order(items: list[tuple[str, bool]]) -> list[str]:
    """Prop ids in the order the engine takes them inside every part: (id, made from several views) in the
    request's order -> the multi-view props first, then the others, each group in the request's order."""
    return [prop_id for prop_id, multi in items if multi] + [prop_id for prop_id, multi in items if not multi]


class Item:
    """One prop of a batch."""

    def __init__(self, prop_id: str, out_path: str = "", seed: int = 0, multi: bool = False, views: int = 1,
                 resolution: int = 1024, error: dict[str, str] | None = None, geometry: str = "pixal3d") -> None:
        self.id = prop_id
        self.parts = parts_of(geometry, resolution)     # its parts, with their words and shares of its bar
        self.part_of = {part.key: part for part in self.parts}
        self.state = FAILED if error else WAITING
        self.part: str | None = None        # the last part begun
        self.parts_done = 0
        self.fraction = 0.0
        self.message = "" if error else WAITING_MESSAGE
        self.step = 0
        self.total = 0
        self.result: dict[str, Any] | None = None
        self.error = error                  # {"detail", "kind"}; given at the start: the request itself is wrong
        self.out_path = out_path
        self.seed = seed
        self.multi = multi
        self.views = views
        self.resolution = resolution
        self.resumed = False
        self.slow: list[str] = []           # parts that ran with the memory limit lifted
        self.seconds = 0.0                  # on the GPU thread, summed over its parts
        self.began: float | None = None

    def told(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"state": self.state, "part": self.part, "parts_done": self.parts_done,
                                 "fraction": round(self.fraction, 4), "message": self.message,
                                 "step": self.step, "total": self.total}
        if self.result is not None:
            entry["result"] = dict(self.result)
        if self.error is not None:
            entry["error"] = dict(self.error)
        return entry

    def clock(self, running: bool) -> None:
        now = time.monotonic()
        if self.began is not None:
            self.seconds += now - self.began
        self.began = now if running else None


class Batch:
    """The record of one batch: what GET /spellcast/batch/{id} answers.

    `words(kind, detail)` turns a failure into the artist's words (pixal3d_server.failure_words). `marker`: the
    ON_CARD file in the batch's work folder (None: no file is kept). `finish_steps`: how many steps the finish
    hook adds after the engine's three decode steps (pixal3d_glb.STEPS), so the finish part's message counts them
    from its first step on (0: only what the events say)."""

    def __init__(self, batch_id: str, items: list[Item], words: Callable[[str, str], str],
                 marker: str | Path | None = None, finish_steps: int = 0) -> None:
        self.batch_id = batch_id
        self.marker = Path(marker) if marker is not None else None
        self.finish_steps = int(finish_steps)
        self.finished: dict[str, Any] | None = None     # the last prop made: {"id", "result"}
        self._marked: tuple[Any, ...] | None = None     # what the marker file says now
        self.items = {item.id: item for item in items}
        self.order = run_order([(item.id, item.multi) for item in items])
        # The parts this batch's props go through, in the engine's order: what "part N of M" counts.
        used = {part.key for item in items for part in item.parts}
        self.keys = tuple(key for key in ALL_KEYS if key in used) or PART_KEYS
        self.labels = {part.key: part.label for item in items for part in item.parts}
        self.words = words
        self.lock = threading.Lock()
        self.cancel = threading.Event()     # set by the web server's threads, read by the engine
        self.state = RUNNING
        self.cancelling = False
        self.seq = 1
        self.started = time.monotonic()
        self.ended: float | None = None
        self.part: dict[str, Any] | None = None
        self.wave = (1, 1)                  # which wave of the engine's run this is, and how many there are
        self.current: str | None = None
        self.loading: str | None = None
        self.error: dict[str, str] | None = None
        self.thread: int | None = None      # the thread that runs the engine for this batch right now
        self._begun: set[str] = set()       # props that began the current part

    # ---- read by the web server's threads ----
    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            seconds = (self.ended if self.ended is not None else time.monotonic()) - self.started
            return {"batch_id": self.batch_id, "state": self.state, "cancelling": self.cancelling, "seq": self.seq,
                    "seconds": round(seconds, 1), "part": dict(self.part) if self.part else None,
                    "current": self.current, "loading": self.loading,
                    "items": {prop_id: item.told() for prop_id, item in self.items.items()},
                    "error": dict(self.error) if self.error else None}

    def request_cancel(self) -> None:
        """Ask the engine to stop at its next check point. Asking again changes nothing."""
        with self.lock:
            if self.state == RUNNING and not self.cancelling:
                self.cancelling = True
                self.seq += 1
        self.cancel.set()

    def running(self) -> bool:
        with self.lock:
            return self.state == RUNNING

    # ---- written on the GPU thread ----
    @contextlib.contextmanager
    def at_work(self) -> Iterator[None]:
        """While the engine runs this batch on the calling thread: BatchRegistry.here() then finds it, so the
        progress the hooks report goes to its props."""
        self.thread = threading.get_ident()
        try:
            yield
        finally:
            self.thread = None

    def on_event(self, event: Any) -> None:
        """One event of the staged engine (pixal3d.staged Event: kind, part, prop_id, step, total, detail). When
        it changed the prop on the card or the last one made (prop_start, prop_rest, prop_done, prop_failed,
        part_start), the marker file says so before the engine goes on."""
        handler = getattr(self, f"_on_{getattr(event, 'kind', '')}", None)
        with self.lock:
            self.seq += 1
            line = handler(event) if handler is not None else None
            note = self._note()
        if note is not None:
            self._write_marker(note)
        if line:
            print(f"[spellcast] {line}", flush=True)

    def mark(self) -> None:
        """Write the marker file as the batch starts: nothing on the card yet, nothing made. The work folder of a
        second attempt still holds the first one's."""
        with self.lock:
            self._marked = None
            note = self._note()
        if note is not None:
            self._write_marker(note)

    def _note(self) -> dict[str, Any] | None:
        """What the marker file must say now, when that is not what it says (the lock is held)."""
        key = (self.current, (self.finished or {}).get("id"))
        if self.marker is None or key == self._marked:
            return None
        self._marked = key
        return {"batch_id": self.batch_id, "current": self.current, "finished": self.finished}

    def _write_marker(self, note: dict[str, Any]) -> None:
        """Replace the marker file whole (a temporary file, then os.replace): the hub never reads half of it. Only
        the GPU thread writes it once the batch runs. Never fails the batch."""
        path = self.marker
        if path is None:
            return
        spare = path.with_name(path.name + ".tmp")       # a leftover is cleared by the engine's next start
        try:
            spare.write_text(json.dumps(note), encoding="utf-8")
            for attempt in range(10):
                try:
                    os.replace(spare, path)
                    return
                except PermissionError:     # Windows: the hub has the file open this very moment
                    if attempt == 9:
                        raise
                    time.sleep(0.02)
        except OSError as exc:
            print(f"[spellcast] Note: could not write which prop is on the card to {path} ({exc}).", flush=True)

    def bar(self, label: str | None, fraction: float | None, step: int, total: int) -> None:
        """What a hook reports while it works on the current prop ("Removing the background", the bars of the
        mesh clean-up): its words, and where the prop's bar stands, kept inside the share of the prop's part.
        A report that says nothing new still counts as a change: the hub sees that Pixal3D is alive."""
        with self.lock:
            self.seq += 1
            item = self.items.get(self.current or "")
            if item is None or item.state != WORKING or label is None:
                return
            item.step, item.total = int(step), int(total)
            # "Building the 3D mesh · step 1 of 2": the hook's own steps, said so that they are not read as props.
            item.message = f"{label} · step {item.step} of {item.total}" if item.total > 1 and item.step > 0 else label
            part = item.part_of.get(item.part or "")
            if part is not None and fraction is not None:
                item.fraction = max(item.fraction, min(max(_moved(fraction, part), part.start), part.end))

    def finish(self, results: dict[str, dict[str, Any]] | None = None) -> None:
        """The engine is through (`results`: its verdict per prop id). A prop it says nothing about was
        cancelled when a cancel was asked for, else it never had its turn (kind "stopped": nothing the artist
        did). The batch is done unless a prop was cancelled, or unless not one of them was made."""
        with self.lock:
            for prop_id, item in self.items.items():
                if item.state in ENDED:
                    continue
                verdict = (results or {}).get(prop_id) or {}
                if verdict.get("state") == DONE:
                    self._done(item, verdict.get("result") or {}, None)
                elif verdict.get("state") == FAILED:
                    error = verdict.get("error") or {}
                    self._failed(item, str(error.get("kind") or "error"), str(error.get("detail") or NOT_MADE))
                elif verdict.get("state") == CANCELLED or self.cancel.is_set():
                    item.clock(False)
                    item.state, item.message, item.step, item.total = CANCELLED, "", 0, 0
                else:
                    self._failed(item, "stopped", NOT_MADE)
            states = [item.state for item in self.items.values()]
            self._end(CANCELLED if CANCELLED in states else FAILED if set(states) == {FAILED} else DONE)

    def fail(self, kind: str, detail: str) -> None:
        """The batch as a whole failed (`kind` as classify gives it; the record says "restart", "ram" or
        "error"). Kind "restart": the process must start afresh; the unfinished props and the prop that was on
        the card stay as they are, for the hub to send again or to blame. Any other kind: the unfinished
        props fail with the batch's error."""
        with self.lock:
            self.error = {"detail": self.words(kind, detail), "kind": kind if kind in ("restart", "ram") else "error"}
            current = self.current
            if kind != "restart":
                for item in self.items.values():
                    if item.state not in ENDED:
                        self._failed(item, kind, detail)
            self._end(FAILED)
            if kind == "restart":
                self.current = current

    def _end(self, state: str) -> None:
        self.state = state
        self.ended = time.monotonic()
        self.part = self.current = self.loading = None
        self.seq += 1

    # ---- the engine's events (the lock is held) ----
    def _item(self, event: Any) -> Item | None:
        return self.items.get(getattr(event, "prop_id", None) or self.current or "")

    def _on_part_start(self, event: Any) -> None:
        detail = getattr(event, "detail", None) or {}
        props = detail.get("props") or []
        if not props:
            return      # no prop goes through it (the camera of a batch made from views only)
        wave, waves = detail.get("wave"), detail.get("waves")
        if isinstance(wave, int) and isinstance(waves, int) and 0 < wave <= waves:
            self.wave = (wave, waves)
        self._enter(str(getattr(event, "part", "")), len(props))

    def _enter(self, key: str, of: int) -> None:
        known = key in self.keys
        # The parts are counted over all the waves together: with waves on, the second wave begins at the part
        # after the first wave's last one, so the step the hub shows never walks back.
        wave, waves = self.wave
        self.part = {"key": key, "index": (wave - 1) * len(self.keys) + (self.keys.index(key) if known else 0),
                     "count": waves * len(self.keys),
                     "label": self.labels.get(key, key), "position": 1, "of": of}
        self._begun = set()
        self.current = self.loading = None

    def _on_prop_start(self, event: Any) -> None:
        item = self._item(event)
        if item is None:
            return
        key = str(getattr(event, "part", ""))
        part = item.part_of.get(key)
        if self.part is None or self.part["key"] != key:
            self._enter(key, 1)
        self._begun.add(item.id)
        detail = getattr(event, "detail", None) or {}
        position, of = detail.get("position"), detail.get("of")
        self.part["position"] = position if isinstance(position, int) and position > 0 else len(self._begun)
        self.part["of"] = max(of if isinstance(of, int) else self.part["of"], self.part["position"])
        item.state, item.part, item.step, item.total = WORKING, key, 0, 0
        item.message = part.label if part else key
        if part is not None:
            item.fraction = max(item.fraction, part.start)
        item.clock(True)
        self.current = item.id
        self.loading = None     # also after a read that was given up: a prop is worked on

    def _on_step(self, event: Any) -> None:
        item = self._item(event)
        if item is None or item.state != WORKING:
            return
        part = item.part_of.get(item.part or "")
        item.step, item.total = int(getattr(event, "step", 0)), int(getattr(event, "total", 0))
        if part is None:
            return
        # A step event says that step is done, so the message names the next one, the one in progress (the last
        # stays the last until the prop is done). The engine's decode events count 3; the hook's steps follow.
        shown_total = item.total
        if part.key == FINISH.key and item.total == DECODE_STEPS and self.finish_steps > 0:
            shown_total = DECODE_STEPS + self.finish_steps
        item.message = f"{part.label} · step {min(item.step + 1, shown_total)} of {shown_total}" if shown_total > 0 \
            else part.label
        start, end = part.start, part.end
        step, total = item.step, item.total
        if part.key == FINISH.key:
            # The engine's three steps (shape, texture, holes) end at DECODED; the steps of the finish hook that
            # follow (the clean-up and the file, which the engine counts on after them) take the rest.
            decoded = _moved(DECODED, part)
            if step <= DECODE_STEPS:
                end, total = decoded, DECODE_STEPS
            else:
                start, step, total = decoded, step - DECODE_STEPS, total - DECODE_STEPS
        share = min(step / total, 1.0) if total > 0 else 0.0
        item.fraction = max(item.fraction, start + (end - start) * share)

    def _on_prop_rest(self, event: Any) -> None:
        item = self._item(event)
        if item is None:
            return
        part = item.part_of.get(str(getattr(event, "part", "")) or item.part or "")
        item.clock(False)
        item.state, item.message, item.step, item.total = RESTING, RESTING_MESSAGE, 0, 0
        if part is not None:
            item.parts_done = max(item.parts_done, item.parts.index(part) + 1)
            item.fraction = max(item.fraction, part.end)
        if self.current == item.id:
            self.current = None

    def _on_prop_done(self, event: Any) -> str | None:
        item = self._item(event)
        if item is None:
            return None
        detail = getattr(event, "detail", None) or {}
        self._done(item, detail.get("result") or {}, detail.get("seconds"))
        return f"Mesh saved to {item.out_path} in {item.result['seconds']:.0f} s"

    def _on_prop_failed(self, event: Any) -> str | None:
        item = self._item(event)
        if item is None:
            return None
        detail = getattr(event, "detail", None) or {}
        kind, text = str(detail.get("kind") or "error"), str(detail.get("detail") or "")
        self._failed(item, kind, text)
        short = text if len(text) <= 300 else text[:300] + "..."
        return f"Prop {item.id} failed in the part '{getattr(event, 'part', '') or item.part}' ({kind}): {short}"

    def _on_prop_spilled(self, event: Any) -> str | None:
        item = self._item(event)
        if item is None:
            return None
        part = str(getattr(event, "part", "") or item.part or "")
        if part and part not in item.slow:
            item.slow.append(part)
        return (f"Prop {item.id}: the part '{part}' does not fit into the free graphics memory; it runs without "
                "the limit, slowly.")

    def _on_prop_resumed(self, event: Any) -> str | None:
        item = self._item(event)
        if item is None:
            return None
        item.resumed = True
        item.state, item.part, item.message = RESTING, "tex", RESTING_MESSAGE
        item.parts_done = max(item.parts_done, len(item.parts) - 1)
        item.fraction = max(item.fraction, item.part_of["finish"].start)
        return (f"Prop {item.id}: its shape and texture were sampled before Pixal3D started again; only the "
                "mesh is built.")

    def _on_net_load(self, event: Any) -> None:
        self.loading = network_label((getattr(event, "detail", None) or {}).get("name", ""))

    def _on_net_ready(self, event: Any) -> str | None:
        detail = getattr(event, "detail", None) or {}
        self.loading = None
        seconds, size = detail.get("seconds"), detail.get("bytes")
        if not isinstance(seconds, (int, float)) or not isinstance(size, (int, float)):
            return None
        return f"Read the {network_label(detail.get('name', ''))} from disk in {seconds:.1f} s ({size / 1e9:.2f} GB)"

    def _on_store_warning(self, event: Any) -> str:
        detail = getattr(event, "detail", None) or {}
        return f"Note: Pixal3D's temporary files: {detail.get('message') or 'a file could not be written'}."

    # ---- endings of a prop (the lock is held) ----
    def _done(self, item: Item, result: dict[str, Any], seconds: Any) -> None:
        item.clock(False)
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
            seconds = item.seconds
        item.state, item.message, item.step, item.total = DONE, "", 0, 0
        item.parts_done, item.fraction = len(item.parts), 1.0
        item.result = {
            "glb_path": item.out_path, "seconds": round(float(seconds), 1), "seed": item.seed,
            "mode": result.get("mode") or ("multi" if item.multi else "single"),
            "views": result.get("views") or item.views, "warning": result.get("warning"),
            "ss_retries": int(result.get("ss_retries") or 0),
            "resolution": result.get("resolution") or item.resolution,
            "slow": list(result.get("slow") or item.slow), "resumed": bool(result.get("resumed") or item.resumed)}
        self.finished = {"id": item.id, "result": dict(item.result)}
        if self.current == item.id:
            self.current = None

    def _failed(self, item: Item, kind: str, detail: str) -> None:
        item.clock(False)
        item.state, item.message, item.step, item.total = FAILED, "", 0, 0
        item.error = {"detail": self.words(kind, detail), "kind": kind}
        self.loading = None     # a network that could not be read fails its props without starting one
        if self.current == item.id:
            self.current = None


class BatchRegistry:
    """The batch that runs, or the last one that ended: Pixal3D works on one at a time, and the record of
    the last one stays readable until the next one starts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._batch: Batch | None = None

    def running(self) -> Batch | None:
        """The batch that still runs, if any."""
        with self._lock:
            return self._batch if self._batch is not None and self._batch.running() else None

    def begin(self, batch: Batch) -> Batch | None:
        """Make `batch` the current one -> None; or, while another one still runs, that one (nothing changes)."""
        with self._lock:
            if self._batch is not None and self._batch.running():
                return self._batch
            self._batch = batch
            return None

    def get(self, batch_id: str) -> Batch | None:
        with self._lock:
            return self._batch if self._batch is not None and self._batch.batch_id == batch_id else None

    def here(self) -> Batch | None:
        """The batch the engine is running on the calling thread right now, if any."""
        with self._lock:
            batch = self._batch
        return batch if batch is not None and batch.thread == threading.get_ident() else None
