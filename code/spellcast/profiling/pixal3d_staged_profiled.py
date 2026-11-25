"""Pixal3D's launcher with probes on its STAGED engine, attached from outside. Runs in the pixal3d venv:

    python pixal3d_staged_profiled.py <exactly pixal3d_server.py's arguments> --profile-out <folder>
                                      [--profile-name staged] [--profile-hz 20] [--profile-external]
                                      [--profile-ballast-leave MIB] [--profile-mesh-hashes]

For developers, for the low-memory work on the 3D stage (design note 10, WP10): what each part of a batch needs on
the card and in RAM, prop by prop, on the full card or on a smaller one emulated inside the process. The batch path
(POST /spellcast/batch) builds its engine through pixal3d_server.build_engine, which the launcher looks up on its
module at every call; this file replaces that function with one that calls the original and then wraps, on the
objects it returns, the engine's run() (to see every event the engine hands to on_event), the loader's set_cap()
and clear_cap() (the limits it gave PyTorch) and the hooks' finish() (the GLB phase of the last part). Nothing in
the launcher or in the fork has a line of profiling code; the server, its routes and its answers are the same.
Start it yourself (never from the hub), or with pixal3d_membench.py --batch, which starts it, sends the batches and
stops it. pixal3d_memreport.py turns the files into the report.

What it writes into --profile-out (all files start with --profile-name):
  <name>.samples.csv (+ .util.csv)   the sampler's rows for the whole process (mem_watch.py, 20 a second): torch
        allocated / reserved, the card's memory and utilisation, this process's dedicated and SHARED GPU memory,
        private bytes, working set, PRIVATE working set, the system's available RAM, commit charge and commit
        limit, and `pinned`: the page-locked RAM the loader and the store report (see below). `run` is the batch
        (batch1, after-batch1, ...), `stage` the part and the prop on the card. About 160 KB a minute; the .util.csv
        twin (the driver's 200 ms utilisation samples) about 4 KB a minute.
  <name>.samples-outside.csv   with --profile-external: the same rows taken by a process of its own (no gaps where
        native code keeps the GIL; no torch and no pinned column). Another 150 KB a minute.
  <name>.events.jsonl   one line per thing that happened: every engine event (part_start / part_end with the
        loader's and the store's stats(), net_load / net_ready / net_drop with seconds and bytes, prop_spilled,
        store_warning, prop_resumed, prop_done, prop_failed), one "prop_part" line when a prop leaves a part (seconds,
        PyTorch's allocated and reserved peak since the prop entered the part, every cap the loader set meanwhile,
        whether the cap was lifted and what the prop's turn cost before and after, the step times, and for `finish`
        its decode and GLB phases), "batch_start" / "batch_end". A few KB per prop-part: about 30 KB per prop.
  <name>.env.json       the process, the card, the PC, the engine's options and the launcher's arguments.
  <name>.ballast.json   with --profile-ballast-leave: the block held to emulate a smaller card (launcher_probes.py).

How a prop-part is measured. At the prop_start event (on the GPU thread, before the part function runs, after its
networks are loaded) PyTorch's peak counters are reset; at prop_rest / prop_done / prop_failed they are read. No
synchronisation is added: an allocation's peak is counted when it is made, on the CPU side, so the numbers are exact
without waiting for the card, and the timing is the engine's own. Inside `finish` the counters are read and reset
again where the hook starts (end of the decode phase) and where it returns (end of the GLB phase).

Page-locked RAM. Windows counts page-locked host memory as SHARED GPU memory, one for one (measured in round A), so
"shared GPU memory never rises above idle" has to be read minus the pinned bytes the engine itself holds. `pinned`
is the loader's stats()["pinned_bytes"] (its one read buffer) + the running store's stats()["pinned_bytes"] (its
slots) + the slot buffers closed stores left in the fork's pool for the next batch (store._spare; they stay locked
for the life of the process). It is refreshed at every engine event except the sampler steps. Windows has no
per-process counter for pages the driver locks, so this known figure stands in for "locked pages".

The decoded meshes against the golden ones (--profile-mesh-hashes): where `finish` hands the decoded mesh to the
hook, the SHA-256 of its vertices, faces, voxel coords and attributes (each tensor's bytes in C order, the hash
pixal3d_golden.py's manifests carry) go into the decode phase's line; pixal3d_memreport.py holds them against
golden/<case>/manifest.json for props named after a golden case (membench's sv1024, mv1024, ...). It costs about a
second per prop (hashing a few hundred MB on the CPU), between the two phases: inside the prop's seconds, outside
both phases' times. Off by default.

A smaller card, emulated (--profile-ballast-leave MIB): launcher_probes.hold_ballast's block, held inside this process
(a ballast in another process does not emulate a smaller card on Windows: the driver evicts an idle process's
memory), taken at the first part_start of the first batch, on the GPU thread, before any network is on the card.
"""

from __future__ import annotations

import argparse
import atexit
import functools
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
LAUNCHERS = CODE / "spellcast" / "launchers"
# As pixal3d_server.py has it when started itself: its folder first (it imports its siblings as top-level modules),
# code/ at the end (the venv's own packages win). Not this folder (sys.path[0] when started as a script).
if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == HERE:
    del sys.path[0]
if str(LAUNCHERS) not in sys.path:
    sys.path.insert(0, str(LAUNCHERS))
if str(CODE) not in sys.path:
    sys.path.append(str(CODE))

import pixal3d_server as server  # noqa: E402
from spellcast.profiling import launcher_probes as probes  # noqa: E402
from spellcast.profiling import mem_watch, toolkit  # noqa: E402

MIB = mem_watch.MIB
SERVER_TARGETS = ("main", "build_engine", "add_hub_routes")
STEPS_KEPT = 600                     # step times kept per prop-part (a sampler has 12-50)
GLB_RESERVE_FLOOR = 2300             # MiB: runner.glb_reserve(faces) is this plus 60 per million faces

REC: "Recorder | None" = None
say = probes.say
_safely = probes.Guard("probe")         # a probe that fails is said once, then does nothing: Pixal3D goes on
_mib = probes.mib


def _plain(value: Any) -> Any:
    """What of an event's detail goes into a JSON line: numbers, strings, short lists; no tensors."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in list(value.items())[:40]}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in list(value)[:40]]
    return str(value)[:200]


def spare_pinned_bytes() -> int:
    """The page-locked slot buffers closed stores left in the fork's pool (pixal3d.staged.store._spare): kept
    for the next batch, locked for the life of the process. 0 before the store module is imported."""
    store = sys.modules.get("pixal3d.staged.store")
    pool = getattr(store, "_spare", None) or {}
    total = 0
    for (size, card), buffers in list(pool.items()):
        if card:                         # buffers for the card are the page-locked ones (when the system allowed)
            total += int(size) * len(buffers)
    return total


def summary_of_loader(stats: dict[str, Any]) -> dict[str, Any]:
    """The loader's stats() without its list of every load (those are in the net_ready lines)."""
    out = {k: _plain(v) for k, v in stats.items() if k != "loads"}
    out["loads_so_far"] = len(stats.get("loads") or [])
    return out


class Recorder:
    """The records of this process. Every method is called on the GPU thread (the engine's events, the wrapped
    loader and hooks), except pinned_total(), which the sampler thread reads."""

    def __init__(self, folder: Path, name: str, watch: Any, ballast_leave: int = 0,
                 mesh_hashes: bool = False) -> None:
        self.folder, self.name, self.watch = folder, name, watch
        self.ballast_leave = ballast_leave
        self.mesh_hashes = mesh_hashes
        self.ballast_done = False
        self.env_done = False
        self.batches = 0
        self.run = "startup"
        self.engine: Any = None
        self.part = ""
        self.open: dict[str, Any] | None = None      # the prop-part being measured
        self.pinned = {"loader": 0, "store": 0, "spare": 0}
        self.options: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._events = open(self.file("events.jsonl"), "w", encoding="utf-8", newline="\n", buffering=1)

    # ---- files (launcher_probes.hold_ballast uses file(), event() and watch) ----
    def file(self, suffix: str) -> Path:
        return self.folder / f"{self.name}.{suffix}"

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            if not self._events.closed:              # an event at exit, after close()
                self._events.write(json.dumps(record, default=str) + "\n")

    def event(self, kind: str, **fields: Any) -> None:
        self.write({"kind": kind, "run": self.run, "t": round(self.watch.now(), 3), **fields})

    def close(self) -> None:
        with self._lock:
            self._events.close()

    # ---- PyTorch and the engine's own figures ----
    @staticmethod
    def torch() -> Any:
        return probes.cuda_torch()

    def peaks(self, reset: bool) -> dict[str, Any]:
        """PyTorch's peaks since the last reset and its memory now (MiB); {} before CUDA runs."""
        torch = self.torch()
        if torch is None:
            return {}
        found = {"peak_alloc": _mib(torch.cuda.max_memory_allocated()),
                 "peak_reserved": _mib(torch.cuda.max_memory_reserved()),
                 "alloc": _mib(torch.cuda.memory_allocated()), "reserved": _mib(torch.cuda.memory_reserved())}
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return found

    def pinned_total(self) -> int:
        return sum(self.pinned.values())

    def stats(self) -> dict[str, Any]:
        """The loader's and the store's stats(), and the pinned figures refreshed from them."""
        engine = self.engine
        loader = getattr(engine, "loader", None)
        store = getattr(engine, "_store", None)
        out: dict[str, Any] = {}
        if loader is not None and callable(getattr(loader, "stats", None)):
            raw = loader.stats()
            out["loader"] = summary_of_loader(raw)
            self.pinned["loader"] = int(raw.get("pinned_bytes") or 0)
        if store is not None and callable(getattr(store, "stats", None)):
            raw = store.stats()
            out["store"] = _plain(raw)
            self.pinned["store"] = int(raw.get("pinned_bytes") or 0)
        else:
            self.pinned["store"] = 0
        self.pinned["spare"] = spare_pinned_bytes()
        out["pinned_bytes"] = dict(self.pinned, total=self.pinned_total())
        return out

    def refresh_pinned(self) -> None:
        _safely("pinned bytes", self.stats)

    # ---- a batch ----
    def batch_start(self, engine: Any, props: list[Any], work_dir: Any, kwargs: dict[str, Any]) -> None:
        self.engine = engine
        self.batches += 1
        self.run = f"batch{self.batches}"
        self.watch.segment(self.run)
        options = getattr(engine, "options", None)
        self.options = {key: _plain(getattr(options, key, None)) for key in (
            "device", "max_vram_mib", "store_ram_mib", "prefetch", "wave_size", "check_rng")}
        self.refresh_pinned()
        self.event("batch_start", work_dir=str(work_dir), wave_size=kwargs.get("wave_size"),
                   resume=bool(kwargs.get("resume")), options=self.options,
                   props=[{"id": p.id, "resolution": p.resolution, "multi": p.multi, "seed": p.seed,
                           "views": len(getattr(p.payload, "views", None) or []) or 1} for p in props],
                   pinned_bytes=self.pinned_total())

    def batch_end(self, results: dict[str, Any] | None, error: BaseException | None) -> None:
        self.close_open("batch end")
        told: dict[str, Any] = {}
        for prop_id, verdict in (results or {}).items():
            result = verdict.get("result") or {}
            told[prop_id] = {"state": verdict.get("state"), "seconds": _plain(result.get("seconds")),
                             "slow": _plain(result.get("slow")), "resolution": result.get("resolution"),
                             "resumed": result.get("resumed"), "error": _plain(verdict.get("error"))}
        self.refresh_pinned()
        self.event("batch_end", results=told,
                   error=None if error is None else f"{type(error).__name__}: {error}"[:300],
                   pinned_bytes=self.pinned_total(), **self.peaks(reset=False))
        self.part = ""
        self.watch.stage = ""
        self.watch.segment(f"after-{self.run}")
        self.run = f"after-{self.run}"
        _safely("flush", self.watch.flush)

    # ---- the engine's events ----
    def on_event(self, event: Any) -> None:
        kind = str(getattr(event, "kind", ""))
        part = str(getattr(event, "part", "") or "")
        prop = getattr(event, "prop_id", None)
        detail = getattr(event, "detail", None) or {}
        now = round(self.watch.now(), 3)
        if kind == "step":
            current = self.open
            room = {key: detail[key] for key in ("room_mib", "card_mib") if key in detail}    # a finish step's
            if current is not None and current["prop"] == prop and len(current["steps"]) < STEPS_KEPT:
                entry = [now, int(getattr(event, "step", 0)), int(getattr(event, "total", 0))]
                current["steps"].append(entry + [room] if room else entry)
            elif part == "finish":
                # A finish step of a prop whose turn is over (its hook's steps run between the other props' turns):
                # its own line, so that the steps' times are kept past the hand-off.
                self.event("step", part=part, prop=prop, step=int(getattr(event, "step", 0)),
                           total=int(getattr(event, "total", 0)), **room)
            return
        if kind == "part_start":
            if self.ballast_leave and not self.ballast_done:
                self.ballast_done = True
                _safely("ballast", probes.hold_ballast, int(self.ballast_leave), self)
            if not self.env_done:
                self.env_done = True
                _safely("env", write_env, self)
            self.part = part
            self.watch.stage = part
            self.refresh_pinned()
            self.event("part_start", part=part, props=_plain(detail.get("props")), wave=detail.get("wave"),
                       waves=detail.get("waves"), pinned_bytes=self.pinned_total())
        elif kind == "prop_start":
            self.close_open("next prop")
            self.refresh_pinned()
            start = self.peaks(reset=True)
            self.open = {"part": part, "prop": prop, "t0": now, "alloc0": start.get("alloc"),
                         "reserved0": start.get("reserved"), "caps": [], "steps": [], "phases": [],
                         "spilled_at": None, "position": detail.get("position"), "of": detail.get("of"),
                         "pinned0": self.pinned_total()}
            self.watch.stage = f"{part} {prop}"
        elif kind == "prop_spilled":
            current = self.open
            before = None
            if current is not None and current["prop"] == prop:
                current["spilled_at"] = now
                before = round(now - current["t0"], 3)
            self.event("prop_spilled", part=part, prop=prop, seconds_before_lift=before)
        elif kind in ("prop_rest", "prop_done", "prop_failed"):
            self.close_open(kind, prop)
            if kind == "prop_done":
                self.event("prop_done", part=part, prop=prop, seconds=_plain(detail.get("seconds")),
                           result=_plain(detail.get("result")))
            elif kind == "prop_failed":
                self.event("prop_failed", part=part, prop=prop, error=_plain(detail))
            self.watch.stage = self.part
        elif kind == "part_end":
            self.close_open("part end")
            stats = self.stats()
            self.event("part_end", part=part, **stats, **self.peaks(reset=False))
            self.watch.stage = ""
        else:                                  # net_load / net_ready / net_drop / store_warning / prop_resumed / ...
            if kind in ("net_ready", "net_drop"):
                self.refresh_pinned()
            self.event(kind, part=part, prop=prop, detail=_plain(detail),
                       **({"pinned_bytes": self.pinned_total()} if kind in ("net_ready", "net_drop") else {}))

    def close_open(self, why: str, prop: Any = None) -> None:
        """The prop-part being measured ends: its line."""
        current, self.open = self.open, None
        if current is None:
            return
        now = round(self.watch.now(), 3)
        end = self.peaks(reset=False)
        peak_alloc = max([v for v in [end.get("peak_alloc")] + [p.get("peak_alloc") for p in current["phases"]]
                          if v is not None], default=None)
        peak_reserved = max([v for v in [end.get("peak_reserved")] + [p.get("peak_reserved")
                                                                       for p in current["phases"]]
                             if v is not None], default=None)
        loader = getattr(self.engine, "loader", None)
        capped = _safely("capped", loader.capped) if loader is not None and hasattr(loader, "capped") else None
        spilled = current["spilled_at"]
        record = {"part": current["part"], "prop": current["prop"], "end": why, "t0": current["t0"], "t1": now,
                  "seconds": round(now - current["t0"], 3), "position": current["position"], "of": current["of"],
                  "peak_alloc": peak_alloc, "peak_reserved": peak_reserved,
                  "alloc0": current["alloc0"], "reserved0": current["reserved0"],
                  "alloc1": end.get("alloc"), "reserved1": end.get("reserved"),
                  "caps": current["caps"], "capped_at_end": capped, "spilled": spilled is not None,
                  "seconds_before_lift": None if spilled is None else round(spilled - current["t0"], 3),
                  "seconds_uncapped": None if spilled is None else round(now - spilled, 3),
                  "pinned0": current["pinned0"], "pinned1": self.pinned_total()}
        if current["phases"]:
            record["phases"] = current["phases"]
        if current["steps"]:
            record["steps"] = current["steps"]
        self.event("prop_part", **record)

    # ---- the wrapped loader and hooks ----
    def cap_set(self, cap_mib: Any, native_reserve_mib: Any, safety_mib: Any) -> None:
        entry = {"t": round(self.watch.now(), 3), "cap_mib": cap_mib, "native_reserve_mib": native_reserve_mib,
                 "safety_mib": safety_mib}
        if self.open is not None:
            self.open["caps"].append(entry)
        else:
            self.event("cap", part=self.part, **entry)

    def cap_cleared(self) -> None:
        entry = {"t": round(self.watch.now(), 3), "cap_mib": None, "lifted": True}
        if self.open is not None:
            self.open["caps"].append(entry)
        else:
            self.event("cap", part=self.part, **entry)

    def phase(self, name: str, t0: float, info: dict[str, Any] | None = None, error: str | None = None,
              cap: Any = None) -> None:
        """A phase of the prop-part that is open ends now: PyTorch's peaks inside it, then a fresh start."""
        current = self.open
        if current is None:
            return
        found = self.peaks(reset=True)
        entry = {"name": name, "t0": round(t0, 3), "t1": round(self.watch.now(), 3),
                 "peak_alloc": found.get("peak_alloc"), "peak_reserved": found.get("peak_reserved"),
                 "alloc1": found.get("alloc"), "reserved1": found.get("reserved"), "cap_mib": cap}
        if info:
            entry.update(info)
        if error:
            entry["error"] = error
        current["phases"].append(entry)


# ---- attaching -------------------------------------------------------------------------------------------------------

def _rows(x: Any) -> int | None:
    shape = getattr(x, "shape", None)
    return int(shape[0]) if shape is not None and len(shape) else None


def mesh_hashes(mesh: Any) -> dict[str, Any]:
    """{shape, dtype, sha256} of the decoded mesh's tensors, by the hash the golden manifests carry."""
    return {name: toolkit.tensor_record(getattr(mesh, name)) for name in ("vertices", "faces", "coords", "attrs")
            if getattr(mesh, name, None) is not None}


def finish_caps(caps: list[dict[str, Any]]) -> tuple[Any, Any]:
    """(decode cap, GLB cap) of `finish` from the caps set so far, read where the hook starts. parts.finish sets
    the decode phase's cap first and the GLB phase's (native reserve glb_reserve(): 2,300 MiB and more) right
    before the hook; a try with the cap lifted sets neither (None, None)."""
    if not caps or caps[-1].get("lifted"):
        return None, None
    if (caps[-1].get("native_reserve_mib") or 0) >= GLB_RESERVE_FLOOR:
        before = caps[-2] if len(caps) > 1 else {}
        return (None if before.get("lifted") else before.get("cap_mib")), caps[-1].get("cap_mib")
    return caps[-1].get("cap_mib"), None


def wrap_hooks(hooks: Any) -> None:
    """hooks.finish (the GLB phase of `finish`): the decode phase ends where it starts, the GLB phase where it
    returns. On the instance; the same hooks object is handed to every engine the launcher builds."""
    if getattr(hooks, "_spellcast_measured", False) or not callable(getattr(hooks, "finish", None)):
        return
    finish = hooks.finish

    @functools.wraps(finish)
    def finish_measured(req: Any, mesh: Any, notes: Any) -> Any:
        rec = REC
        if rec is None or rec.open is None:
            return finish(req, mesh, notes)
        counts = {"faces": _rows(getattr(mesh, "faces", None)), "vertices": _rows(getattr(mesh, "vertices", None)),
                  "voxels": _rows(getattr(mesh, "coords", None)), "resolution": getattr(mesh, "resolution", None)}
        decode_cap, glb_cap = finish_caps(rec.open["caps"])
        _safely("decode phase", rec.phase, "decode", rec.open["t0"], counts, None, decode_cap)
        if rec.mesh_hashes and rec.open is not None and rec.open["phases"]:
            began = rec.watch.now()
            found = _safely("mesh hashes", mesh_hashes, mesh)
            rec.open["phases"][-1].update(mesh=found, hash_seconds=round(rec.watch.now() - began, 3))
        started = rec.watch.now()
        try:
            result = finish(req, mesh, notes)
        except BaseException as exc:
            _safely("glb phase", rec.phase, "glb", started, None, type(exc).__name__, glb_cap)
            raise
        _safely("glb phase", rec.phase, "glb", started, None, None, glb_cap)
        return result

    object.__setattr__(hooks, "finish", finish_measured)
    object.__setattr__(hooks, "_spellcast_measured", True)


def attach_engine(engine: Any) -> None:
    """engine.run (every event, the batch's start and end), loader.set_cap / clear_cap (the limits given)."""
    run = engine.run

    @functools.wraps(run)
    def run_measured(props: Any, work_dir: Any, *args: Any, on_event: Callable[[Any], None], **kwargs: Any) -> Any:
        rec = REC
        if rec is None:
            return run(props, work_dir, *args, on_event=on_event, **kwargs)

        def relay(event: Any) -> None:
            _safely("event", rec.on_event, event)
            on_event(event)

        _safely("batch start", rec.batch_start, engine, list(props), work_dir, kwargs)
        try:
            results = run(props, work_dir, *args, on_event=relay, **kwargs)
        except BaseException as exc:
            _safely("batch end", rec.batch_end, None, exc)
            raise
        _safely("batch end", rec.batch_end, results, None)
        return results

    object.__setattr__(engine, "run", run_measured)
    loader = getattr(engine, "loader", None)
    if loader is None:
        say("PROFILING: the engine has no loader; its caps and statistics are not measured.")
        return
    set_cap, clear_cap = loader.set_cap, loader.clear_cap

    @functools.wraps(set_cap)
    def set_cap_noted(native_reserve_mib: int = 200, safety_mib: int | None = None) -> int:
        cap = set_cap(native_reserve_mib, safety_mib)
        if REC is not None:
            _safely("cap", REC.cap_set, cap, native_reserve_mib, safety_mib)
        return cap

    @functools.wraps(clear_cap)
    def clear_cap_noted() -> None:
        clear_cap()
        if REC is not None:
            _safely("cap", REC.cap_cleared)

    object.__setattr__(loader, "set_cap", set_cap_noted)
    object.__setattr__(loader, "clear_cap", clear_cap_noted)


def missing(srv: Any = server) -> list[str]:
    return [name for name in SERVER_TARGETS if not callable(getattr(srv, name, None))]


def attach_server(srv: Any = server) -> None:
    """pixal3d_server.build_engine (looked up by the launcher at every call) and add_hub_routes (the last step
    before the server answers: the moment it is ready)."""
    build = srv.build_engine

    @functools.wraps(build)
    def build_engine_measured(options: Any, hooks: Any) -> Any:
        _safely("hooks", wrap_hooks, hooks)
        engine = build(options, hooks)
        _safely("attaching to the engine", attach_engine, engine)
        if REC is not None:
            REC.event("note", text="engine built", multiview=getattr(engine, "multiview", None))
        return engine

    srv.build_engine = build_engine_measured
    routes = srv.add_hub_routes

    @functools.wraps(routes)
    def add_hub_routes_noted(*args: Any, **kwargs: Any) -> Any:
        result = routes(*args, **kwargs)
        if REC is not None:
            REC.event("note", text="ready: the routes are added, the server starts answering")
        return result

    srv.add_hub_routes = add_hub_routes_noted


def write_env(rec: Recorder) -> None:
    store_defaults = None
    store = sys.modules.get("pixal3d.staged.store")
    if store is not None and callable(getattr(store, "ram_defaults", None)):
        store_defaults = store.ram_defaults()
    env = {"tool": "pixal3d_staged_profiled", **probes.process_env(mem_watch.HostProbe(), rec.watch),
           "ballast_leave_mib": rec.ballast_leave or None, "engine_options": rec.options,
           "store_ram_defaults_mib": store_defaults,
           "environment": {key: os.environ.get(key) for key in (
               "ATTN_BACKEND", "SPARSE_ATTN_BACKEND", "SPARSE_CONV_BACKEND", "PYTORCH_CUDA_ALLOC_CONF",
               "MIMALLOC_PURGE_DELAY", "HF_HUB_OFFLINE", "FLEX_GEMM_AUTOTUNE_CACHE_PATH")}}
    rec.file("env.json").write_text(json.dumps(env, indent=1, default=str), encoding="utf-8")


# ---- main ------------------------------------------------------------------------------------------------------------

def split_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """(this file's own arguments, the rest: pixal3d_server.py's)."""
    p = probes.profile_arguments("staged")
    p.add_argument("--profile-mesh-hashes", action="store_true",
                   help="SHA-256 of every decoded mesh, to hold against the golden ones (about 1 s per prop)")
    return p.parse_known_args(argv)


def main() -> int:
    global REC
    own, rest = split_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *rest]
    folder = Path(own.profile_out).resolve()             # absolute: the launcher changes the working folder
    folder.mkdir(parents=True, exist_ok=True)
    gone = missing()
    if gone:
        say(f"PROFILING IS OFF: pixal3d_server.py no longer has {', '.join(gone)}, which spellcast/profiling/"
            f"pixal3d_staged_profiled.py attaches to. Pixal3D runs unmeasured; update this file.")
        return server.main()
    watch = mem_watch.MemWatch(folder / f"{own.profile_name}.samples.csv", hz=own.profile_hz,
                               extra={"pinned": lambda: REC.pinned_total() if REC is not None else -1})
    REC = Recorder(folder, own.profile_name, watch, own.profile_ballast_leave, own.profile_mesh_hashes)
    watch.start()
    atexit.register(watch.stop)
    atexit.register(REC.close)
    if own.profile_external:
        probes.watch_from_outside(folder, own.profile_name, watch, own.profile_hz)
    attach_server()
    REC.event("note", text="process started; imports begin")
    say(f"Pixal3D runs with probes on its staged engine ({own.profile_hz:g} rows a second"
        + (f", a smaller card emulated: {own.profile_ballast_leave:,} MiB" if own.profile_ballast_leave else "")
        + f"). The measurements go to {folder} ({own.profile_name}.*).")
    return server.main()


if __name__ == "__main__":
    sys.exit(main())
