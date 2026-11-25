"""Tables (Markdown) from a pixal3d_staged_profiled.py measurement: the staged batch path the hub uses. For
developers; stdlib only, any Python:

    python pixal3d_memreport.py <folder> --name staged > staged-<case>.md

The measurement is found by its <name>.samples.csv (--staged is still taken and changes nothing). A short report
against design note 10. Its first lines say which card size and which RAM size were EMULATED (the in-process block
of --profile-ballast-leave, <name>.ballast.json; the page-locked RAM of ram_ballast.py, <name>.ramballast.json) and
that neither a real 8 GB card nor a real 16 GB PC was used. Then per batch: PyTorch's allocated and reserved peaks
per part (the largest over its props; `finish` split into decode and GLB) against the reserved limits of 2.2 and
its "after, typical" estimates, the caps the loader set; times against 2.5; per part the process's shared GPU
memory MINUS the page-locked bytes the loader and the store report (Windows counts those as shared memory one for
one) against its level before the batch's first part, and dedicated + that against the emulated card; the
launcher's private working set + page-locked bytes against 12 GiB, its private bytes, the system's least available
RAM and least commit headroom; every prop that needed the cap lifted (its part, its seconds before and after the
lift); the network loads; the card's idle seconds per part, the idle gaps between two props of a part next to the
part's longest network load, and the idle seconds inside `finish` per prop; with --mesh-hashes measurements, the
decoded meshes against the golden ones (--golden). Memory columns from the outside sampler's rows when there are
any (no gaps), else the thread's; the pinned bytes are a step function of the engine's events, so they apply to
either. All memory in MiB (1 GiB = 1024 MiB). Windows' counters trail torch's by a few milliseconds (by up to 3 s
while memory spills), so a value shown is the largest one held over three rows in a row, not a single row's.

The round-A report (a pixal3d_profiled.py measurement of the reference path: stage trees, the models, the spill
episodes) is code/devtools/pixal3d-lowmem/pixal3d_profiled_report.py; it reads the files with this module's
Samples, load_events and table helpers.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

NUMERIC = ("torch_alloc", "torch_reserved", "nvml_used", "nvml_util", "proc_ded", "proc_shared", "adapter_ded",
           "adapter_shared", "priv", "ws", "avail", "commit", "late_ms", "ws_private", "commit_limit", "pinned")
IDLE_BELOW = 10            # percent: a 200 ms utilisation sample under it counts as idle
GIB = 1024


class Samples:
    """One samples file: columns as lists, `t` ascending."""

    def __init__(self, path: Path, ballast: float = 0.0, ballast_t: float = 0.0) -> None:
        """ballast: MiB held in this process from t = ballast_t on to emulate a smaller card (taken off proc_ded)."""
        self.t: list[float] = []
        self.stage: list[str] = []
        self.run: list[str] = []
        self.col: dict[str, list[float | None]] = {name: [] for name in NUMERIC}
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    self.t.append(float(row["t"]))
                except (TypeError, ValueError):
                    continue
                self.stage.append(row.get("stage") or "")
                self.run.append(row.get("run") or "")
                for name in NUMERIC:
                    cell = row.get(name)
                    self.col[name].append(float(cell) if cell not in (None, "") else None)
        if ballast:
            self.col["proc_ded"] = [None if v is None else (max(0.0, v - ballast) if t >= ballast_t else v)
                                    for t, v in zip(self.t, self.col["proc_ded"])]
            self.col["priv"] = [None if v is None else (v - ballast if t >= ballast_t else v)
                                for t, v in zip(self.t, self.col["priv"])]
        both: list[float | None] = []
        for ded, shared in zip(self.col["proc_ded"], self.col["proc_shared"]):
            both.append(None if ded is None or shared is None else ded + shared)
        self.col["proc_both"] = both
        # Under WDDM every MiB CUDA allocates - on the card or spilled into shared memory - is also charged to the
        # process's private bytes and the system's commit (verified: a 2 GiB and a 4 GiB tensor raise private
        # bytes by 2,052 and 4,104 MiB with the working set unchanged; at the texture conditioning's spill, private
        # = CPU side + dedicated + shared to within 1%). What is left is the CPU side.
        self.col["cpu_priv"] = [None if priv is None else priv - (ded or 0.0) - (shared or 0.0)
                                for priv, ded, shared in zip(self.col["priv"], self.col["proc_ded"],
                                                             self.col["proc_shared"])]
        # What is not PyTorch's: the CUDA context and the native packages' own allocations (thread rows only).
        self.col["outside_torch"] = [None if both_ is None or reserved is None else both_ - reserved
                                     for both_, reserved in zip(both, self.col["torch_reserved"])]
        self.util_t: list[float] = []
        self.util: list[float] = []
        util_path = path.with_suffix(".util.csv")
        if util_path.is_file():
            with open(util_path, newline="", encoding="utf-8") as f:
                rows = sorted((float(r["t"]), float(r["util"])) for r in csv.DictReader(f) if r.get("t"))
            self.util_t = [t for t, _ in rows]
            self.util = [u for _, u in rows]

    def span(self, t0: float, t1: float) -> range:
        lo, hi = bisect.bisect_left(self.t, t0), bisect.bisect_right(self.t, t1)
        if lo < hi:
            return range(lo, hi)
        near = [i for i in (lo - 1, lo) if 0 <= i < len(self.t) and min(abs(self.t[i] - t0),
                                                                        abs(self.t[i] - t1)) <= 0.15]
        if not near:
            return range(0)
        best = min(near, key=lambda i: abs(self.t[i] - (t0 + t1) / 2))
        return range(best, best + 1)

    def peak(self, name: str, t0: float, t1: float) -> float | None:
        values = [v for v in (self.col[name][i] for i in self.span(t0, t1)) if v is not None]
        return max(values) if values else None

    def held_peak(self, name: str, t0: float, t1: float, rows: int = 3) -> float | None:
        """The largest value that `rows` rows in a row all reach (0.15 s at 20 rows a second): a level that was
        held, not the one-row jump two counters make when they are read a few milliseconds apart."""
        values = [v for v in (self.col[name][i] for i in self.span(t0, t1)) if v is not None]
        if not values:
            return None
        if len(values) < rows:
            return min(values)
        return max(min(values[i:i + rows]) for i in range(len(values) - rows + 1))

    def low(self, name: str, t0: float, t1: float) -> float | None:
        values = [v for v in (self.col[name][i] for i in self.span(t0, t1)) if v is not None]
        return min(values) if values else None

    def idle_seconds(self, t0: float, t1: float) -> float | None:
        """The seconds of [t0, t1] the driver's samples call idle (each sample: the 200 ms before its stamp)."""
        if not self.util_t:
            return None
        total = 0.0
        lo = bisect.bisect_left(self.util_t, t0)
        for i in range(lo, len(self.util_t)):
            end = self.util_t[i]
            start = end - 0.2
            if start >= t1:
                break
            if self.util[i] < IDLE_BELOW:
                total += max(0.0, min(end, t1) - max(start, t0))
        return total


def load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass                      # the last line of a killed process
    return events


def read_json(path: Path) -> dict[str, Any]:
    """The JSON object in `path`, {} without the file."""
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def num(value: Any, digits: int = 0) -> str:
    if value is None:
        return "-"
    return f"{value:,.{digits}f}"


def table(header: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    header = list(header)
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def idle_stretches(samples: Samples, t0: float, t1: float, min_seconds: float = 1.0) -> list[tuple[float, float]]:
    found = []
    start = None
    end = 0.0
    for stamp, value in zip(samples.util_t, samples.util):
        if stamp < t0 or stamp - 0.2 > t1:
            continue
        if value < IDLE_BELOW:
            if start is None:
                start = stamp - 0.2
            end = stamp
        elif start is not None:
            if end - start >= min_seconds:
                found.append((max(start, t0), min(end, t1)))
            start = None
    if start is not None and end - start >= min_seconds:
        found.append((max(start, t0), min(end, t1)))
    return found


# ---- the staged path (pixal3d_staged_profiled.py) --------------------------------------------------------------------

STAGED_PARTS = ("cutout", "camera", "features", "ss", "cond_lr", "lr", "upsample", "cond_hr", "hr", "tex", "finish")
# Design note 10, 2.2: what PyTorch may RESERVE with max_vram_mib = 7168 (the 7 GiB target), per phase.
RESERVED_LIMIT = {"part": 6368, "finish: decode": 6068, "finish: glb": 3546}
# 2.2, column "after, typical" (PyTorch's peak ALLOCATED need, weights included; [est] until measured). finish's
# decode: the larger of the two decoders (first set of savings); multi-view features: one batch of V views.
TYPICAL_ALLOC = {"cutout": 3219, "camera": 2921, "features": 1600, "ss": 3000, "cond_lr": 1500, "lr": 2880,
                 "upsample": 1250, "cond_hr": 3900, "hr": 3370, "tex": 3310, "finish: decode": 4260,
                 "finish: glb": 1000}
CONDITIONING = ("features", "cond_lr", "cond_hr")
SAMPLERS = ("ss", "lr", "hr", "tex")
GOLDEN = Path(__file__).resolve().parents[2] / "logs" / "profiling" / "pixal3d-lowmem" / "golden"
# A decoded mesh's tensor -> where a golden manifest keeps its hash (pixal3d_golden.py: the final mesh after
# fill_holes only as vertices_sha / faces_sha; the texture decoder's voxels under tensors/11-decode).
GOLDEN_KEYS = {"vertices": ("vertices_sha",), "faces": ("faces_sha",),
               "coords": ("tensors", "11-decode", "tex_coords", "sha256"),
               "attrs": ("tensors", "11-decode", "tex_attrs", "sha256")}
SHARED_TOLERANCE = 64          # MiB: shared - pinned this far above its idle level still counts as "at idle"
RAM_TARGET = 12 * GIB          # 2.4: the launcher's private working set + locked pages on a 16 GiB PC
GAP_SLACK = 0.3                # s: an idle stretch this close to the boundary between two props belongs to it


def staged_lines(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [e for e in events if e.get("kind") == kind]


def pinned_steps(events: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """(t, MiB) of every pinned figure the events carry, in time order: the page-locked bytes the loader, the
    running store and the slot pool held from then on."""
    steps = []
    for e in events:
        t = e.get("t")
        if t is None:
            continue
        value = e.get("pinned_bytes")
        if isinstance(value, dict):
            value = value.get("total")
        if isinstance(value, (int, float)):
            steps.append((float(t), value / (1024 * 1024)))
        for key, at in (("pinned0", "t0"), ("pinned1", "t1")):
            if isinstance(e.get(key), (int, float)) and e.get(at) is not None:
                steps.append((float(e[at]), e[key] / (1024 * 1024)))
    return sorted(steps)


def pinned_at(steps: list[tuple[float, float]], t: float) -> float:
    """The pinned MiB at time t: the last figure at or before t (0 before the first one)."""
    i = bisect.bisect_right(steps, (t, float("inf"))) - 1
    return steps[i][1] if i >= 0 else 0.0


def net_shared(samples: Samples, steps: list[tuple[float, float]]) -> list[float | None]:
    """Per row: this process's shared GPU memory minus the page-locked bytes it held then."""
    return [None if shared is None else shared - pinned_at(steps, t)
            for t, shared in zip(samples.t, samples.col["proc_shared"])]


def held_max(values: list[float], rows: int = 3) -> float | None:
    """The largest level `rows` rows in a row all reach (as Samples.held_peak)."""
    if not values:
        return None
    if len(values) < rows:
        return min(values)
    return max(min(values[i:i + rows]) for i in range(len(values) - rows + 1))


def span_values(samples: Samples, values: list[float | None], t0: float, t1: float) -> list[float]:
    return [v for v in (values[i] for i in samples.span(t0, t1)) if v is not None]


def idle_level(samples: Samples, values: list[float | None], t_start: float, before: float = 3.0) -> float | None:
    """The level of `values` just before a batch's first part: the median of the rows in the `before` seconds
    ahead of t_start, else the first row from t_start on."""
    found = span_values(samples, values, t_start - before, t_start - 1e-6)
    if found:
        return statistics.median(found)
    later = [v for t, v in zip(samples.t, values) if t >= t_start and v is not None]
    return later[0] if later else None


def shared_rise(samples: Samples, steps: list[tuple[float, float]], t0: float, t1: float,
                idle: float | None) -> dict[str, float | None]:
    """In [t0, t1]: the shared peak, the pinned bytes, shared - pinned held over three rows, and how far that
    rises above `idle` (the level before the batch)."""
    net = net_shared(samples, steps)
    values = span_values(samples, net, t0, t1)
    peak = held_max(values)
    return {"shared_peak": samples.peak("proc_shared", t0, t1),
            "pinned_peak": max([pinned_at(steps, samples.t[i]) for i in samples.span(t0, t1)], default=None),
            "net_peak": peak, "rise": None if peak is None or idle is None else peak - idle}


def windows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every part a batch went through: {"run", "part", "t0", "t1", "props"} from part_start / part_end. The
    texture and the finish overlap (the engine runs them prop by prop, their brackets open together), so a window
    is kept by its run and its part."""
    found, open_ = [], {}
    for e in events:
        key = (e.get("run"), e.get("part"))
        if e.get("kind") == "part_start":
            open_[key] = {"run": e.get("run"), "part": e.get("part"), "t0": e["t"], "t1": None,
                          "props": e.get("props") or []}
        elif e.get("kind") == "part_end" and open_.get(key):
            w = open_.pop(key)
            w["t1"] = e["t"]
            found.append(w)
        elif e.get("kind") == "batch_end":
            for key in [key for key in open_ if key[0] == e.get("run")]:
                w = open_.pop(key)          # a batch that broke inside a part
                w["t1"] = e["t"]
                found.append(w)
    return found


def part_peaks(prop_parts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per part (finish split into its decode and GLB phases): the largest allocated and reserved peak over its
    props, which prop that was, the caps the loader set, the props that needed the cap lifted, the seconds."""
    rows: dict[str, dict[str, Any]] = {}

    def add(key: str, alloc: Any, reserved: Any, prop: str, seconds: float, caps: list[Any], spilled: bool) -> None:
        row = rows.setdefault(key, {"alloc": None, "reserved": None, "prop": None, "caps": [], "spilled": [],
                                    "seconds": [], "props": 0})
        row["props"] += 1
        row["seconds"].append(seconds)
        if alloc is not None and (row["alloc"] is None or alloc > row["alloc"]):
            row["alloc"], row["prop"] = alloc, prop
        if reserved is not None and (row["reserved"] is None or reserved > row["reserved"]):
            row["reserved"] = reserved
        row["caps"] += [c for c in caps if c is not None]
        if spilled:
            row["spilled"].append(prop)

    for pp_ in prop_parts:
        caps = [c.get("cap_mib") for c in pp_.get("caps") or []]
        if pp_["part"] == "finish" and pp_.get("phases"):
            for phase in pp_["phases"]:
                add(f"finish: {phase['name']}", phase.get("peak_alloc"), phase.get("peak_reserved"), pp_["prop"],
                    phase["t1"] - phase["t0"], [phase.get("cap_mib")], pp_.get("spilled", False))
        else:
            add(pp_["part"], pp_.get("peak_alloc"), pp_.get("peak_reserved"), pp_["prop"], pp_["seconds"], caps,
                pp_.get("spilled", False))
    return rows


def idle_gaps(samples: Samples, ordered: list[dict[str, Any]], min_seconds: float = 0.2) -> list[float]:
    """For each two props one after the other in a part: the seconds the card idled at their boundary (the idle
    stretches of `min_seconds` or more that reach within GAP_SLACK of it)."""
    if not ordered:
        return []
    stretches = idle_stretches(samples, ordered[0]["t0"], ordered[-1]["t1"], min_seconds)
    gaps = []
    for before, after in zip(ordered, ordered[1:]):
        lo, hi = before["t1"] - GAP_SLACK, after["t0"] + GAP_SLACK
        gaps.append(sum(end - start for start, end in stretches if start <= hi and end >= lo))
    return gaps


def emulation_lines(env: dict[str, Any], held: dict[str, Any], ram: dict[str, Any], measured: float | None
                    ) -> list[str]:
    """The report's first lines: what was emulated, said plainly."""
    card = env.get("card") or {}
    total = card.get("total_mib")
    if held.get("ballast_mib"):
        leave = held.get("leave_mib")
        card_text = (f"an emulated card of about {leave / 1024:.1f} GiB ({leave:,} MiB of dedicated memory for the "
                     f"process, its CUDA context included): a block of {held['ballast_mib']:,} MiB held inside the "
                     f"process on the real {card.get('name')} ({num(total)} MiB)")
        if measured is not None:
            card_text += (f"; the most dedicated memory the process got besides the block: {measured:,.0f} MiB "
                          f"(that is the emulated card's real limit only if memory spilled)")
    else:
        card_text = (f"the full card, nothing emulated: {card.get('name')} ({num(total)} MiB = "
                     f"{(total or 0) / 1024:.1f} GiB)")
    if ram.get("locked_mib"):
        before = ram.get("before") or {}
        what = (f"an emulated {ram['pc_gib']:g} GiB PC" if ram.get("pc_gib") else
                f"{ram.get('leave_mib'):,} MiB left available")
        ram_text = (f"{what}: ram_ballast.py held {ram['locked_mib']:,} MiB page-locked on the real PC's "
                    f"{num(before.get('ram_total_mib'))} MiB (available RAM fell by "
                    f"{num(ram.get('available_drop_mib'))} MiB: {'verified' if ram.get('verified') else 'NOT verified'}"
                    + ("; PARTIAL: not all of it could be locked" if ram.get("partial") else "") + ")")
    else:
        ram_text = (f"the PC's full RAM, nothing emulated: {num(env.get('ram_total_mib'))} MiB = "
                    f"{(env.get('ram_total_mib') or 0) / 1024:.1f} GiB")
    return [f"**Card: {card_text}.**", "",
            f"**RAM: {ram_text}.**", "",
            "**Neither a real 8 GB card nor a real 16 GB PC was used: smaller cards and PCs are emulated on the "
            "developer's PC as said above.**", ""]


def golden_case(prop_id: str) -> str:
    """membench names a prop after its golden case; a second copy in one batch gets "-2"."""
    base, dash, tail = prop_id.rpartition("-")
    return base if dash and tail.isdigit() else prop_id


def against_golden(prop_parts: list[dict[str, Any]], golden: Path) -> list[list[str]]:
    """Per prop with mesh hashes: each tensor equal to the golden manifest's, different, or not in it."""
    rows = []
    for pp_ in sorted((x for x in prop_parts if x["part"] == "finish"), key=lambda x: x["t0"]):
        decode = next((ph for ph in pp_.get("phases") or [] if ph.get("name") == "decode" and ph.get("mesh")), None)
        if decode is None:
            continue
        manifest = read_json(golden / golden_case(pp_["prop"]) / "manifest.json")
        cells = []
        for name in ("vertices", "faces", "coords", "attrs"):
            mine = (decode["mesh"].get(name) or {}).get("sha256")
            theirs: Any = manifest
            for key in GOLDEN_KEYS[name]:
                theirs = theirs.get(key) if isinstance(theirs, dict) else None
            cells.append("-" if mine is None else "no golden" if not theirs else
                         "equal" if mine == theirs else "DIFFERENT")
        rows.append([pp_["prop"], golden_case(pp_["prop"]) if manifest else "(none)", *cells])
    return rows


def verdict_reserved(peak: float | None, limit: float) -> str:
    if peak is None:
        return "-"
    return "inside" if peak <= limit else f"OVER by {peak - limit:,.0f}"


def staged_report(folder: Path, name: str, golden: Path = GOLDEN) -> str:
    out: list[str] = []
    events = load_events(folder / f"{name}.events.jsonl")
    env, held, ram, bench = (read_json(folder / f"{name}.{part}.json")
                             for part in ("env", "ballast", "ramballast", "bench"))
    ballast, ballast_t = float(held.get("ballast_mib") or 0.0), float(held.get("t") or 0.0)
    thread_path = folder / f"{name}.samples.csv"
    thread = Samples(thread_path, ballast, ballast_t) if thread_path.is_file() else None
    outside_path = folder / f"{name}.samples-outside.csv"
    outside = Samples(outside_path, ballast, ballast_t) if outside_path.is_file() else None
    if outside is not None and not outside.t:
        outside = None
    mem = outside if outside is not None else thread
    steps = pinned_steps(events)
    measured = None
    if ballast and thread is not None:
        raw = [v for v in thread.col["proc_ded"] if v is not None]      # the block is already taken off
        measured = max(raw) if raw else None
    emulated_card = float(held.get("leave_mib") or 0) or float((env.get("card") or {}).get("total_mib") or 0)

    out.append(f"# Pixal3D staged batches: {name}\n")
    out += emulation_lines(env, held, ram, measured)
    card = env.get("card") or {}
    out.append(f"- PC: {card.get('name')}, RAM {num(env.get('ram_total_mib'))} MiB, commit limit "
               f"{num(env.get('commit_limit_mib'))} MiB; torch {env.get('torch')} (CUDA {env.get('cuda')}); engine "
               f"options {env.get('engine_options')}; the store's RAM cache / slot sizes for this PC "
               f"{env.get('store_ram_defaults_mib')} MiB; {env.get('hz')} rows a second.")
    if bench:
        out.append(f"- Ready {bench.get('ready_seconds')} s after the spawn (2.5: 15-25 s). Card before the start: "
                   f"{bench.get('card_before')}; after the stop: {bench.get('card_after_stop')}, port open: "
                   f"{bench.get('port_open_after_stop')}.")
    source = "the outside sampler (no gaps)" if outside is not None else "the sampler thread"
    out.append(f"- Memory columns from {source}; MiB throughout (1 GiB = 1,024 MiB); shared GPU memory is read "
               f"minus the page-locked bytes the loader and the store report.\n")

    for start in staged_lines(events, "batch_start"):
        run = start["run"]
        mine = [e for e in events if e.get("run") == run]
        end = next((e for e in mine if e.get("kind") == "batch_end"), None)
        t_start, t_end = start["t"], (end or {}).get("t") or (mine[-1]["t"] if mine else start["t"])
        prop_parts = staged_lines(mine, "prop_part")
        parts = windows(mine)
        results = (end or {}).get("results") or {}
        props = start.get("props") or []
        first_part = parts[0]["t0"] if parts else t_start
        out.append(f"## {run}: {len(props)} props, {t_end - t_start:.1f} s\n")
        out.append("; ".join(f"{p['id']} ({p['resolution']}"
                             + (f", {p['views']} views" if p.get("views", 1) > 1 else "")
                             + f", seed {p['seed']}): {results.get(p['id'], {}).get('state', '?')}"
                             + (f", {results[p['id']]['seconds']:.1f} s" if isinstance(
                                 results.get(p['id'], {}).get('seconds'), (int, float)) else "")
                             + (f", slow in {', '.join(results[p['id']]['slow'])}"
                                if results.get(p['id'], {}).get('slow') else "")
                             for p in props) + f". Wave size {start.get('wave_size')}, resume {start.get('resume')}."
                   + (f" ERROR: {end['error']}" if end and end.get("error") else "") + "\n")

        # ---- peaks per part
        peaks = part_peaks(prop_parts)
        rows = []
        for key in [*STAGED_PARTS[:-1], "finish", "finish: decode", "finish: glb"]:
            row = peaks.get(key)
            if row is None:
                continue
            limit = RESERVED_LIMIT.get(key, RESERVED_LIMIT["part"])
            caps = row["caps"]
            rows.append([key, row["props"], num(row["alloc"]), row["prop"] or "-", num(TYPICAL_ALLOC.get(key)),
                         num(row["reserved"]), num(limit), verdict_reserved(row["reserved"], limit),
                         (num(min(caps)) if min(caps) == max(caps) else f"{num(min(caps))}-{num(max(caps))}")
                         if caps else "-",
                         num(sum(row["seconds"]), 1), num(statistics.median(row["seconds"]), 1),
                         ", ".join(row["spilled"]) or "-"])
        out.append("### PyTorch's peaks per part (largest over the props; limits of 2.2 at max_vram_mib 7168)\n")
        out.append(table(("part", "props", "peak allocated", "of prop", "2.2 typical (est)", "peak reserved",
                          "2.2 reserved limit", "reserved vs limit", "caps set", "seconds, all props",
                          "seconds, median prop", "cap lifted for"), rows))
        out.append("")

        # ---- times
        rows = []
        for p in props:
            mine_pp = [x for x in prop_parts if x["prop"] == p["id"]]
            by = {x["part"]: x for x in mine_pp}
            finish = by.get("finish") or {}
            phases = {ph["name"]: ph["t1"] - ph["t0"] for ph in finish.get("phases") or []}
            rows.append([p["id"], num(sum(by[k]["seconds"] for k in CONDITIONING if k in by), 1),
                         num(sum(by[k]["seconds"] for k in SAMPLERS if k in by), 1),
                         num(phases.get("decode"), 1), num(phases.get("glb"), 1),
                         num(sum(x["seconds"] for x in mine_pp), 1),
                         num(results.get(p["id"], {}).get("seconds"), 1)])
        loads = [e for e in staged_lines(mine, "net_ready")]
        load_s = sum((e.get("detail") or {}).get("seconds") or 0 for e in loads)
        load_gb = sum((e.get("detail") or {}).get("bytes") or 0 for e in loads) / 1e9
        out.append("### Times (s) against 2.5 (one prop at 1024: conditioning 3-5, samplers 130.6, decoders 6, "
                   "GLB 42.5 + 8.2; a prop of a batch about 190, the first after a start 235-250, one prop on a "
                   "running tool, 16 GiB, cold cache about 225; network loads 30-40 per wave, cold)\n")
        out.append(table(("prop", "conditioning (features, cond_lr, cond_hr)", "samplers (ss, lr, hr, tex)",
                          "finish: decode", "finish: GLB", "its prop-parts", "the engine's result seconds"), rows))
        out.append(f"\nNetwork loads: {len(loads)} in {load_s:.1f} s ({load_gb:.1f} GB); the batch's wall time "
                   f"{t_end - t_start:.1f} s, of which {first_part - t_start:.1f} s before its first part.\n")
        rows = [[e.get("part"), (e.get("detail") or {}).get("name"), num((e.get("detail") or {}).get("seconds"), 2),
                 num(((e.get("detail") or {}).get("bytes") or 0) / 1e9, 2),
                 num(((e.get("detail") or {}).get("bytes") or 0) / 1e9 / (e["detail"]["seconds"] or 1e9), 2)
                 if (e.get("detail") or {}).get("seconds") else "-"] for e in loads]
        out.append(table(("part", "network", "seconds", "GB", "GB/s"), rows) if rows else "No network loads.")
        out.append("")

        # ---- shared GPU memory and the emulated card, per part
        if mem is not None and mem.t:
            net = net_shared(mem, steps)
            idle = idle_level(mem, net, first_part)
            # The CUDA context takes its own share of shared memory at the first allocation, after the level above
            # was read, and keeps it: the lowest level while the batch runs is then the idle one, and a spill
            # shows as a rise above that.
            during = [v for v in span_values(mem, net, first_part, mem.t[-1]) if v is not None]
            if during and (idle is None or min(during) > idle):
                idle = min(during)
            both = [None if d is None or n is None else d + n for d, n in zip(mem.col["proc_ded"], net)]
            rows = []
            for w in parts:
                rise = shared_rise(mem, steps, w["t0"], w["t1"], idle)
                ded_net = held_max(span_values(mem, both, w["t0"], w["t1"]))
                rows.append([w["part"], num(w["t1"] - w["t0"], 1), num(rise["shared_peak"]), num(rise["pinned_peak"]),
                             num(rise["net_peak"]), num(rise["rise"]),
                             "at idle" if rise["rise"] is not None and rise["rise"] <= SHARED_TOLERANCE
                             else ("ABOVE idle" if rise["rise"] is not None else "-"),
                             num(mem.peak("proc_ded", w["t0"], w["t1"])), num(ded_net),
                             ("under" if ded_net <= emulated_card else f"OVER by {ded_net - emulated_card:,.0f}")
                             if ded_net is not None and emulated_card else "-",
                             num(mem.low("avail", w["t0"], w["t1"]))])
            out.append(f"### Shared GPU memory minus page-locked bytes, per part (idle level, with the CUDA context: "
                       f"{num(idle)} MiB; 'at idle' = within {SHARED_TOLERANCE} MiB of it; held over three rows)\n")
            out.append(table(("part", "seconds", "shared peak", "page-locked (loader + store)", "shared - locked peak",
                              "rise above idle", "verdict", "dedicated peak", "dedicated + (shared - locked) peak",
                              f"vs the card ({num(emulated_card)} MiB)", "least RAM available"), rows))
            out.append("")

            # ---- RAM
            t0, t1 = t_start, t_end
            ws_locked = [None if w is None else w + pinned_at(steps, t)
                         for t, w in zip(mem.t, mem.col["ws_private"])]
            peak_ws_locked = held_max(span_values(mem, ws_locked, t0, t1))
            limits = span_values(mem, mem.col["commit_limit"], t0, t1)
            headroom = [None if c is None or lim is None else lim - c
                        for c, lim in zip(mem.col["commit"], mem.col["commit_limit"])]
            least_headroom = min(span_values(mem, headroom, t0, t1), default=None)
            out.append("### RAM (2.4: the launcher's private working set + page-locked bytes under 12 GiB on a 16 GiB "
                       "PC; its commit and the system's headroom stated)\n")
            out.append(table(("measure", "MiB"), [
                ["launcher: private working set peak", num(mem.peak("ws_private", t0, t1))],
                ["launcher: page-locked bytes peak (loader + store + slot pool)",
                 num(max([s for t, s in steps if t0 <= t <= t1], default=pinned_at(steps, t0)))],
                ["**launcher: private working set + page-locked, peak (held)**",
                 f"{num(peak_ws_locked)} ({verdict_reserved(peak_ws_locked, RAM_TARGET)} {num(RAM_TARGET)})"],
                ["launcher: working set peak (shared pages included)", num(mem.peak("ws", t0, t1))],
                ["launcher: private bytes (its commit) peak", num(mem.peak("priv", t0, t1))],
                ["launcher: CPU side of the private bytes (private - dedicated - shared) peak",
                 num(max(span_values(mem, mem.col["cpu_priv"], t0, t1), default=None))],
                ["system: least RAM available", num(mem.low("avail", t0, t1))],
                ["system: commit charge peak", num(mem.peak("commit", t0, t1))],
                ["system: commit limit", num(max(limits) if limits else None)],
                ["system: least commit headroom (limit - charge)", num(least_headroom)],
            ]))
            if ram.get("locked_mib") and ram.get("pc_gib"):
                pc = ram["pc_gib"] * 1024
                out.append(f"\nOn the emulated {ram['pc_gib']:g} GiB PC ({pc:,.0f} MiB): the launcher took at most "
                           f"{num(peak_ws_locked)} MiB of it (private working set + page-locked); the least RAM "
                           f"left available to everything else (Windows, the hub, the browser) was "
                           f"{num(mem.low('avail', t0, t1))} MiB.")
            out.append("")

        # ---- the cap lifted
        lifted = [x for x in prop_parts if x.get("spilled")]
        out.append("### Props that needed the cap lifted (made slowly)\n")
        out.append(table(("prop", "part", "seconds in the part", "before the lift (capped tries)",
                          "after the lift (uncapped)", "peak allocated", "peak reserved"),
                         [[x["prop"], x["part"], num(x["seconds"], 1), num(x.get("seconds_before_lift"), 1),
                           num(x.get("seconds_uncapped"), 1), num(x.get("peak_alloc")), num(x.get("peak_reserved"))]
                          for x in lifted]) if lifted else "None: every prop-part stayed inside its cap.")
        out.append("")

        # ---- the card's idle time
        if thread is not None and thread.util_t:
            rows = []
            for w in parts:
                ordered = sorted((x for x in prop_parts if x["part"] == w["part"] and w["t0"] <= x["t0"] <= w["t1"]),
                                 key=lambda x: x["t0"])
                gaps = idle_gaps(thread, ordered)
                in_part = [e for e in loads if w["t0"] <= e["t"] <= w["t1"]]
                longest_load = max(((e.get("detail") or {}).get("seconds") or 0 for e in in_part), default=None)
                idle = thread.idle_seconds(w["t0"], w["t1"]) or 0.0
                before = thread.idle_seconds(w["t0"], ordered[0]["t0"]) if ordered else None
                rows.append([w["part"], num(w["t1"] - w["t0"], 1), num(idle, 1),
                             f"{idle / (w['t1'] - w['t0']):.0%}" if w["t1"] > w["t0"] else "-",
                             num(before, 1), len(gaps), num(max(gaps), 1) if gaps else "-",
                             num(sum(gaps), 1) if gaps else "-", num(longest_load, 1),
                             ("longer than a load" if gaps and longest_load is not None and max(gaps) > longest_load
                              else "-")])
            out.append("### The card's idle time per part (driver samples under 10%)\n")
            out.append(table(("part", "seconds", "card idle s", "idle share", "idle before its first prop (loads)",
                              "prop boundaries", "longest idle gap between props", "idle at all boundaries",
                              "longest network load in the part", "note"), rows))
            rows = []
            for x in sorted((x for x in prop_parts if x["part"] == "finish"), key=lambda x: x["t0"]):
                glb = next((ph for ph in x.get("phases") or [] if ph["name"] == "glb"), None)
                rows.append([x["prop"], num(x["seconds"], 1), num(thread.idle_seconds(x["t0"], x["t1"]), 1),
                             num(glb["t1"] - glb["t0"], 1) if glb else "-",
                             num(thread.idle_seconds(glb["t0"], glb["t1"]), 1) if glb else "-"])
            out.append("\nInside `finish` (to_glb's CPU-only work leaves the card idle; 7.1):\n")
            out.append(table(("prop", "finish s", "card idle s", "GLB phase s", "card idle in the GLB phase s"), rows)
                       if rows else "No prop reached `finish`.")
            out.append("")

        # ---- the decoded meshes against the golden ones
        compared = against_golden(prop_parts, golden)
        if compared:
            out.append(f"### Decoded meshes against the golden ones ({golden})\n")
            out.append(table(("prop", "golden case", "vertices (after fill_holes)", "faces", "voxel coords",
                              "voxel attributes"), compared))
            out.append("")

        # ---- the store and the loader at each part's end
        rows = []
        for e in staged_lines(mine, "part_end"):
            store, loader = e.get("store") or {}, e.get("loader") or {}
            pinned = e.get("pinned_bytes") or {}
            rows.append([e.get("part"), num((store.get("ram_bytes") or 0) / 1048576),
                         num((store.get("peak_ram_bytes") or 0) / 1048576),
                         num((store.get("disk_bytes") or 0) / 1048576), store.get("spills", "-"),
                         num((store.get("spilled_bytes") or 0) / 1048576), store.get("uploads_ahead", "-"),
                         store.get("uploads_at_take", "-"), num((pinned.get("loader") or 0) / 1048576),
                         num((pinned.get("store") or 0) / 1048576), num((pinned.get("spare") or 0) / 1048576),
                         store.get("warnings", "-"), loader.get("cap_mib", "-")])
        out.append("### The store and the loader at the end of each part\n")
        out.append(table(("part", "RAM cache", "RAM cache peak", "on disk", "spills", "spilled MiB",
                          "uploads ahead", "uploads at take", "pinned: loader", "pinned: store", "pinned: pool",
                          "warnings", "cap"), rows) if rows else "No part ended.")
        warnings = staged_lines(mine, "store_warning")
        if warnings:
            out.append("\nStore warnings: " + "; ".join(
                f"{w.get('part')}/{w.get('prop') or '-'}: {(w.get('detail') or {}).get('message')}" for w in warnings))
        out.append("")
    if not staged_lines(events, "batch_start"):
        out.append("No batch in the records.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Markdown tables from a pixal3d_staged_profiled.py measurement.")
    p.add_argument("folder")
    p.add_argument("--name", default="pixal3d")
    p.add_argument("--staged", action="store_true", help="a pixal3d_staged_profiled.py measurement (found by itself "
                                                          "when <name>.samples.csv is there)")
    p.add_argument("--golden", default=str(GOLDEN), help="the golden captures the mesh hashes are held against")
    args = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
    folder = Path(args.folder)
    if args.staged or (folder / f"{args.name}.samples.csv").is_file():
        print(staged_report(folder, args.name, Path(args.golden)))
        return 0
    print(f"{folder} has no {args.name}.samples.csv, so it is not a pixal3d_staged_profiled.py measurement named "
          f"{args.name!r}. A round-A one (pixal3d_profiled.py) is read by "
          f"code/devtools/pixal3d-lowmem/pixal3d_profiled_report.py.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
