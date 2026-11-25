"""What the GPU did during one profiled request, read from a torch.profiler (Kineto / CUPTI) Chrome trace: per phase and
step, per kernel class, and where the CPU waited for the GPU or the GPU sat idle. New for Spellcast3D (not from
AURA-V5); for Qwen-Image 2.1, whose GPU work is all stable-diffusion.cpp's (ggml's CUDA kernels).

    python -m spellcast.profiling.cuda_trace_report <trace.json> [--prefix qwen:] [--report all|phases|kernels|sync]

The trace is request_profiler.py's trace.json (use_torch_profiler) or sync_trace.json (diagnostic_enable_sync_hunter
on its own). CUPTI records every kernel, copy and memset of the process although PyTorch launches none of them; each
carries args.correlation, the id of the CUDA runtime (or driver) call that launched it, which is recorded on the
launching thread. The request's annotations, made on its thread with record_function (the trace's user_annotation
events), tell the parts apart: <prefix>request around the whole request (the window: everything outside it - Kineto's
own synchronisations at start and stop included - is left out), <prefix><phase> one after the other inside it
(references, prompt, prepare, drawing i/n, finishing, saving) and <prefix>step k/n inside a drawing phase. Without a
<prefix>request annotation the window is the whole trace.

Every GPU event is charged to the phase and step whose annotation holds the CUDA call that LAUNCHED it (the GPU runs
behind the CPU, so its own start time can fall into the next phase), or to its own start time when the launch is not
in the trace or came from another thread. GPU busy time is the union of the intervals of the kernels, memsets and
copies from or to pinned or device memory: overlapping ones (two streams) are never counted twice. A copy from or to
PAGEABLE memory is not GPU work: CUPTI stretches its interval over the CPU's staging of the memory through a pinned
buffer (page faults from disk included: 64 MB in 4 s when the weights come from disk), so it counts as GPU idle, and
the part of the idle time in which one is in flight is reported on its own as "waiting for host memory (pageable
copies)". Built for a whole 2-3 minute request (~150k-400k events, 50-150 MB of JSON): one sort, then bisect; the
trace in memory takes ~5x the file's size.

    phase_report(events, prefix)         wall, GPU busy, waiting for host memory, kernels, copies and CPU time in CUDA
                                         calls per phase; steps
    kernel_class_report(events, prefix)  GPU ms per kind of kernel (matmul, attention, ...), the top kernels of each
    sync_report(events, prefix)          blocking CUDA calls, copies, allocations, the GPU's idle gaps (each labelled by
                                         the phase and step holding most of it)
Each takes the 'X' events (load_trace) or a TraceAnalysis of them (built once when several reports are wanted).
Stdlib only.
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, Union

from spellcast.profiling.profiler_python_utils import _clean_kernel_name, readable_kernel_name

log = logging.getLogger(__name__)

GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
LAUNCH_CATS = ("cuda_runtime", "cuda_driver")
REQUEST = "request"
OUTSIDE = "(outside the phases)"
HOST_WAIT = "waiting for host memory"   # the GPU idle while a pageable copy is in flight (the CPU stages it)
GAP_MIN_US = 100.0                  # idle gaps shorter than 0.1 ms are a kernel boundary, not a wait
SHOWN_US = 50.0                     # a figure smaller than this would print as 0.0 ms: left out where optional
TOP_GAPS = 10
TOP_KERNELS = 3
TOP_CLASSES_PER_PHASE = 4

# Calls after which the CPU has waited for the GPU (or for the driver, which waits for it): explicit synchronisations,
# synchronous copies and memsets, and the frees and pinned-memory calls that synchronise the device implicitly.
BLOCKING_CALLS = frozenset({
    "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaThreadSynchronize",
    "cudaMemcpy", "cudaMemcpy2D", "cudaMemcpy3D", "cudaMemcpyToSymbol", "cudaMemcpyFromSymbol", "cudaMemcpyPeer",
    "cudaMemset", "cudaMemset2D", "cudaMemset3D", "cudaFree", "cudaFreeHost", "cudaFreeArray", "cudaMallocHost",
    "cudaHostAlloc", "cudaHostRegister", "cudaHostUnregister",
    "cuCtxSynchronize", "cuStreamSynchronize", "cuEventSynchronize", "cuMemcpy", "cuMemcpyHtoD", "cuMemcpyDtoH",
    "cuMemcpyPeer", "cuMemsetD8", "cuMemsetD16", "cuMemsetD32", "cuMemFree", "cuMemFreeHost", "cuMemAllocHost",
    "cuMemHostAlloc", "cuMemHostRegister", "cuMemHostUnregister",
})
# Memory management: ggml's pool grows with cudaMalloc, or with the virtual-memory calls (cuMem*) on cards that have
# them. Kineto records only the driver calls it knows, so cuMem* can be missing although they were made.
ALLOCATION_CALLS = frozenset({
    "cudaMalloc", "cudaFree", "cudaMallocAsync", "cudaFreeAsync", "cudaMallocManaged", "cudaMallocPitch",
    "cudaMallocHost", "cudaFreeHost", "cudaHostAlloc", "cudaHostRegister", "cudaHostUnregister",
    "cuMemAlloc", "cuMemFree", "cuMemAllocAsync", "cuMemFreeAsync", "cuMemCreate", "cuMemMap", "cuMemSetAccess",
    "cuMemUnmap", "cuMemRelease", "cuMemAddressReserve", "cuMemAddressFree",
})

_STEP = re.compile(r"step (\d+)/(\d+)$")
_API_SUFFIX = re.compile(r"(_v\d+)?(_ptsz|_ptds)?$")
_DIRECTION = re.compile(r"\b(HtoD|DtoH|DtoD|HtoH|PtoP)\b")

# Kernel classes, tested in this order on the kernel's function name (readable, lower case, without a leading k_).
_CLASS_PREFIXES = (
    ("matmul", ("mul_mat", "dequantize_mul_mat", "gemm", "gemv", "splitkreduce")),
    ("quantize", ("quantize",)),
    ("attention", ("flash_attn", "soft_max", "softmax", "fattn", "flash_fwd", "fmha")),
    ("rope", ("rope",)),
    ("norm", ("rms_norm", "norm", "group_norm", "l2_norm", "layer_norm", "layernorm")),
    ("conv", ("im2col", "col2im", "conv2d", "conv1d", "conv_2d", "conv_transpose", "conv_", "pool")),
    ("copy/convert", ("cpy", "convert", "dup", "cont", "get_rows", "set_rows", "concat", "pad", "upscale",
                      "permute", "transpose", "repeat", "dequantize", "copy")),
    ("elementwise", ("bin_bcast", "unary", "scale", "add", "sub", "mul", "div", "silu", "gelu", "swiglu", "geglu",
                     "reglu", "glu", "clamp", "sqr", "sqrt", "sin", "cos", "tanh", "sigmoid", "relu", "leaky_relu",
                     "neg", "step", "exp", "log", "abs", "sgn", "hardswish", "hardsigmoid", "elu", "softplus",
                     "floor", "ceil", "round", "trunc", "op_", "acc", "fill", "set")),
)
# Anywhere in the raw name (cuBLAS's and cuDNN's kernels, PyTorch's): AURA's own keywords.
_CLASS_KEYWORDS = (
    ("matmul", ("gemm", "gemv", "splitkreduce", "cublas")),
    ("attention", ("softmax", "fmha", "flash_fwd")),
    ("norm", ("layer_norm", "layernorm")),
    ("conv", ("convolution",)),
    ("copy/convert", ("copy",)),
    ("elementwise", ("elementwise",)),
)

Events = Union[Sequence[dict], "TraceAnalysis"]


# ---- loading ---------------------------------------------------------------------------------------------------------

def load_trace(path: str | Path) -> list[dict[str, Any]]:
    """The trace's complete ('X') events. Tolerates the '\"\"\"' corruption torch's exporter can write (AURA's
    process_torch_trace_json repairs it the same way)."""
    raw = Path(path).read_text(encoding="utf-8")
    if '"""' in raw:
        log.warning(f"Sanitizing corrupt Torch trace: {path}")
        raw = raw.replace('"""', "'''")
    data = json.loads(raw)
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    return [e for e in events if isinstance(e, dict) and e.get("ph") == "X" and "ts" in e and "dur" in e]


def detect_prefix(events: Iterable[dict]) -> str:
    """The annotation prefix of the trace's request ("qwen:" for <prefix>request), or "" when there is none."""
    for e in events:
        name = e.get("name")
        if e.get("cat") == "user_annotation" and isinstance(name, str) and name.endswith(":" + REQUEST):
            return name[:-len(REQUEST)]
    return ""


def api_name(name: str) -> str:
    """A CUDA API call's name without its version and per-thread-stream suffixes (cuMemcpyHtoD_v2 -> cuMemcpyHtoD)."""
    return _API_SUFFIX.sub("", name or "")


def is_pageable(event: dict) -> bool:
    """A copy from or to pageable (not pinned) host memory: CUPTI names it "Memcpy HtoD (Pageable -> Device)"."""
    return event.get("cat") == "gpu_memcpy" and "pageable" in str(event.get("name", "")).lower()


def kernel_class(event: dict) -> str:
    """matmul | quantize | attention | rope | norm | conv | copy/convert | elementwise | memcpy HtoD | memcpy DtoH |
    memcpy DtoD | memcpy | memset | other; a pageable copy's class ends with " (pageable)"."""
    cat, name = event.get("cat"), str(event.get("name", ""))
    if cat == "gpu_memcpy":
        m = _DIRECTION.search(name)
        return (f"memcpy {m.group(1)}" if m else "memcpy") + (" (pageable)" if is_pageable(event) else "")
    if cat == "gpu_memset":
        return "memset"
    base = (readable_kernel_name(name) or name).split("<", 1)[0].rsplit("::", 1)[-1].lower()
    if base.startswith("k_"):
        base = base[2:]
    for kind, prefixes in _CLASS_PREFIXES:
        if base.startswith(prefixes):
            return kind
    low = name.lower()
    for kind, words in _CLASS_KEYWORDS:
        if any(w in low for w in words):
            return kind
    return "other"


# ---- the analysis ----------------------------------------------------------------------------------------------------

@dataclass
class Span:
    name: str                       # the annotation without its prefix: "prompt", "drawing 1/2", "step 3/20"
    start: float                    # microseconds, the trace's clock
    end: float

    @property
    def wall(self) -> float:
        return self.end - self.start


def _end(e: dict) -> float:
    return e["ts"] + e["dur"]


def _thread(e: dict) -> tuple[Any, Any]:
    return e.get("pid"), e.get("tid")


def _merge(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _subtract(intervals: list[tuple[float, float]], minus: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The parts of `intervals` outside `minus` (both merged: sorted, not overlapping), in one sweep."""
    out: list[tuple[float, float]] = []
    j = 0
    for start, end in intervals:
        while j < len(minus) and minus[j][1] <= start:
            j += 1
        at, k = start, j
        while k < len(minus) and minus[k][0] < end:
            if minus[k][0] > at:
                out.append((at, minus[k][0]))
            at = max(at, minus[k][1])
            k += 1
        if at < end:
            out.append((at, end))
    return out


class _Timeline:
    """Merged intervals (sorted, not overlapping) with running totals: their length inside any [a, b] by bisect."""

    def __init__(self, merged: list[tuple[float, float]]) -> None:
        self.spans = merged
        self._starts = [a for a, _ in merged]
        self._ends = [b for _, b in merged]
        self._cum = [0.0]
        for a, b in merged:
            self._cum.append(self._cum[-1] + (b - a))

    @property
    def total(self) -> float:
        return self._cum[-1]

    def between(self, a: float, b: float) -> float:
        if b <= a or not self.spans:
            return 0.0
        i = bisect.bisect_right(self._ends, a)
        j = bisect.bisect_left(self._starts, b)
        if i >= j:
            return 0.0
        total = self._cum[j] - self._cum[i]
        total -= max(0.0, a - self._starts[i])
        total -= max(0.0, self._ends[j - 1] - b)
        return total


def _most(spans: Sequence[Span], starts: list[float], a: float, b: float,
          keep: Callable[[int], bool] | None = None) -> int:
    """The index of the span (sorted, not overlapping) that holds most of [a, b] - at least as much as the part of
    [a, b] outside every span - else -1. `keep` limits the candidates (their overlaps still count as inside)."""
    best, best_overlap, inside = -1, 0.0, 0.0
    i = max(0, bisect.bisect_right(starts, a) - 1)
    while i < len(spans) and spans[i].start < b:
        overlap = min(spans[i].end, b) - max(spans[i].start, a)
        if overlap > 0:
            inside += overlap
            if overlap > best_overlap and (keep is None or keep(i)):
                best, best_overlap = i, overlap
        i += 1
    return best if best >= 0 and best_overlap >= (b - a) - inside else -1


def _union_by(spans: Iterable[tuple[float, float]], buckets: Iterable[int]) -> dict[int, float]:
    """Total length of the union of the intervals of each bucket; the intervals come sorted by start."""
    total: dict[int, float] = defaultdict(float)
    current: dict[int, list[float]] = {}
    for (start, end), bucket in zip(spans, buckets):
        cur = current.get(bucket)
        if cur is None or start > cur[1]:
            if cur is not None:
                total[bucket] += cur[1] - cur[0]
            current[bucket] = [start, end]
        elif end > cur[1]:
            cur[1] = end
    for bucket, cur in current.items():
        total[bucket] += cur[1] - cur[0]
    return total


class TraceAnalysis:
    """The trace sorted once: the window, the phases and steps, every GPU event's phase and step (by its launch), the
    merged GPU-busy intervals (pageable copies left out) and the merged waits for host memory (a pageable copy in
    flight, the GPU otherwise idle). The three reports read it."""

    def __init__(self, events: Sequence[dict], prefix: str | None = None) -> None:
        self.prefix = detect_prefix(events) if prefix is None else prefix
        gpu: list[dict] = []
        calls: list[dict] = []
        notes: list[dict] = []
        lo, hi = math.inf, -math.inf
        for e in events:
            cat = e.get("cat")
            if cat in GPU_CATS:
                gpu.append(e)
            elif cat in LAUNCH_CATS:
                calls.append(e)
            elif cat == "user_annotation" and self.prefix and str(e.get("name", "")).startswith(self.prefix):
                notes.append(e)
            lo, hi = min(lo, e["ts"]), max(hi, _end(e))
        requests = [a for a in notes if a["name"] == self.prefix + REQUEST]
        request = max(requests, key=lambda a: a["dur"]) if requests else None
        if request is not None:
            self.window = (float(request["ts"]), float(_end(request)))
            self.thread: tuple[Any, Any] | None = _thread(request)
            self.window_name = request["name"]
        else:
            self.window = (float(lo), float(hi)) if events else (0.0, 0.0)
            self.thread = None
            self.window_name = "the whole trace"
        phases: list[Span] = []
        steps: list[Span] = []
        for a in notes:
            if a is request or (self.thread is not None and _thread(a) != self.thread):
                continue
            label = a["name"][len(self.prefix):]
            if label == REQUEST:
                continue
            (steps if _STEP.match(label) else phases).append(Span(label, float(a["ts"]), float(_end(a))))
        self.phases = sorted(phases, key=lambda s: s.start)
        self.steps = sorted(steps, key=lambda s: s.start)
        self._phase_starts = [s.start for s in self.phases]
        self._step_starts = [s.start for s in self.steps]
        self.step_phase = [self.phase_at(s.start) for s in self.steps]

        w0, w1 = self.window
        self.gpu = sorted(gpu, key=lambda e: e["ts"])
        self.launch: dict[Any, dict] = {}
        for c in calls:
            corr = (c.get("args") or {}).get("correlation")
            if corr is not None:
                self.launch[corr] = c
        self.calls = sorted((c for c in calls if w0 <= c["ts"] <= w1), key=lambda e: e["ts"])
        self.thread_calls = [c for c in self.calls if self.thread is None or _thread(c) == self.thread]
        self.gpu_phase: list[int] = []
        self.gpu_step: list[int] = []
        for g in self.gpu:
            t = g["ts"]
            launch = self.launch.get((g.get("args") or {}).get("correlation"))
            if launch is not None and (self.thread is None or _thread(launch) == self.thread):
                t = launch["ts"]
            self.gpu_phase.append(self.phase_at(t))
            self.gpu_step.append(self.step_at(t))
        self.gpu_pageable = [is_pageable(g) for g in self.gpu]
        inside = [(max(g["ts"], w0), min(_end(g), w1), pageable)
                  for g, pageable in zip(self.gpu, self.gpu_pageable) if _end(g) > w0 and g["ts"] < w1]
        self.busy = _merge((a, b) for a, b, pageable in inside if not pageable)
        self._busy = _Timeline(self.busy)
        # the pageable copies' time in which nothing else runs on the GPU (with a kernel on another stream it is busy)
        self.waiting = _subtract(_merge((a, b) for a, b, pageable in inside if pageable), self.busy)
        self._waiting = _Timeline(self.waiting)
        self._call_starts = [c["ts"] for c in self.thread_calls]
        self._call_longest = max((c["dur"] for c in self.thread_calls), default=0.0)

    # -- lookups
    @property
    def wall(self) -> float:
        return self.window[1] - self.window[0]

    @property
    def busy_total(self) -> float:
        return self._busy.total

    @property
    def waiting_total(self) -> float:
        """Microseconds of the window in which the GPU only waited for host memory (pageable copies)."""
        return self._waiting.total

    def phase_at(self, t: float) -> int:
        i = bisect.bisect_right(self._phase_starts, t) - 1
        return i if i >= 0 and t < self.phases[i].end else -1

    def step_at(self, t: float) -> int:
        i = bisect.bisect_right(self._step_starts, t) - 1
        return i if i >= 0 and t < self.steps[i].end else -1

    def phase_name(self, index: int) -> str:
        return self.phases[index].name if index >= 0 else OUTSIDE

    def where_most(self, a: float, b: float) -> str:
        """The phase (and step) holding most of [a, b], or OUTSIDE when most of it is outside every phase: an idle gap
        that opens a moment before the first phase (the request's annotation opens first) is that phase's."""
        phase = _most(self.phases, self._phase_starts, a, b)
        if phase < 0:
            return OUTSIDE
        step = _most(self.steps, self._step_starts, a, b, lambda i: self.step_phase[i] == phase)
        return self.phase_name(phase) + (f", {self.steps[step].name}" if step >= 0 else "")

    def busy_between(self, a: float, b: float) -> float:
        """GPU-busy microseconds inside [a, b] (the merged intervals, clipped)."""
        return self._busy.between(a, b)

    def waiting_between(self, a: float, b: float) -> float:
        """Microseconds inside [a, b] in which the GPU only waited for host memory (pageable copies)."""
        return self._waiting.between(a, b)

    def gaps(self) -> list[tuple[float, float]]:
        """The GPU's idle intervals inside the window (a pageable copy may be in flight in them: waiting_between)."""
        w0, w1 = self.window
        out, at = [], w0
        for a, b in self.busy:
            if a > at:
                out.append((at, a))
            at = max(at, b)
        if w1 > at:
            out.append((at, w1))
        return out

    def call_during(self, a: float, b: float) -> tuple[str, float, int] | None:
        """The request thread's CUDA call that overlaps [a, b] the most, summed by name (a long gap can hold hundreds
        of the same call): (name, overlap in microseconds, calls)."""
        by_name: dict[str, list[float]] = {}
        i = bisect.bisect_left(self._call_starts, b) - 1
        while i >= 0 and self._call_starts[i] >= a - self._call_longest:
            c = self.thread_calls[i]
            overlap = min(_end(c), b) - max(c["ts"], a)
            if overlap > 0:
                row = by_name.setdefault(str(c.get("name", "?")), [0.0, 0])
                row[0] += overlap
                row[1] += 1
            i -= 1
        if not by_name:
            return None
        name, (overlap, calls) = max(by_name.items(), key=lambda kv: kv[1][0])
        return name, overlap, int(calls)

    def busy_by_phase(self) -> dict[int, float]:
        return self._busy_by(self.gpu_phase)

    def busy_by_step(self) -> dict[int, float]:
        return self._busy_by(self.gpu_step)

    def waiting_by_phase(self) -> dict[int, float]:
        return self._waiting_by(self.gpu_phase)

    def waiting_by_step(self) -> dict[int, float]:
        return self._waiting_by(self.gpu_step)

    def _busy_by(self, buckets: list[int]) -> dict[int, float]:
        """GPU busy per bucket (phase or step, by launch): the union of its events' intervals, pageable copies left
        out."""
        return _union_by(((g["ts"], _end(g)) for g, pageable in zip(self.gpu, self.gpu_pageable) if not pageable),
                         (bucket for bucket, pageable in zip(buckets, self.gpu_pageable) if not pageable))

    def _waiting_by(self, buckets: list[int]) -> dict[int, float]:
        """Waiting for host memory per bucket (by launch): the union of its pageable copies inside the window, less
        the GPU-busy time inside them."""
        w0, w1 = self.window
        spans: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for g, bucket, pageable in zip(self.gpu, buckets, self.gpu_pageable):
            if pageable and _end(g) > w0 and g["ts"] < w1:
                spans[bucket].append((max(g["ts"], w0), min(_end(g), w1)))
        return {bucket: sum((b - a) - self.busy_between(a, b) for a, b in _merge(found))
                for bucket, found in spans.items()}


def _analysis(events: Events, prefix: str | None) -> TraceAnalysis:
    return events if isinstance(events, TraceAnalysis) else TraceAnalysis(events, prefix)


def _ms(us: float) -> str:
    return f"{us / 1000.0:.1f}"


def _pct(part: float, whole: float) -> str:
    return f"{100.0 * part / whole:.1f}" if whole > 0 else "0.0"


def _calls(n: int) -> str:
    return f"{n} call" if n == 1 else f"{n} calls"


def _spread(values: list[float]) -> str:
    return (f"min {_ms(min(values))} / median {_ms(statistics.median(values))} / max {_ms(max(values))}"
            if values else "-")


# ---- a) phases and steps ---------------------------------------------------------------------------------------------

def phase_report(events: Events, prefix: str | None = None) -> str:
    """Per phase, in time order: wall ms, GPU busy ms, GPU % of wall, ms waiting for host memory (pageable copies),
    kernels, MB copied, CPU ms inside CUDA calls (the request thread's); then the steps (the first step of each picture
    apart: it fills the caches and stages the weights) and the total."""
    a = _analysis(events, prefix)
    busy = a.busy_by_phase()
    waiting = a.waiting_by_phase()
    kernels: Counter[int] = Counter()
    copied: dict[int, float] = defaultdict(float)
    for g, phase in zip(a.gpu, a.gpu_phase):
        if g.get("cat") == "kernel":
            kernels[phase] += 1
        elif g.get("cat") == "gpu_memcpy":
            copied[phase] += float((g.get("args") or {}).get("bytes", 0) or 0)
    call_phase = [a.phase_at(c["ts"]) for c in a.thread_calls]
    in_calls = _union_by(((c["ts"], _end(c)) for c in a.thread_calls), call_phase)

    lines = ["--- PHASES (cuda_trace_report) ---",
             f"Window: {a.window_name}, {_ms(a.wall)} ms. GPU work is charged to the phase and step whose CUDA call "
             f"launched it; GPU busy = the union of kernel, memset and pinned / device memcpy intervals. A pageable "
             f"memcpy is not GPU work (CUPTI stretches it over the CPU's staging of the memory, page faults from disk "
             f"included): the GPU idle while one is in flight is {HOST_WAIT.upper()} (AURA's kernel summary's GPU "
             f"BUSY counts it as busy).",
             f"PHASE | WALL (ms) | GPU BUSY (ms) | GPU % | {HOST_WAIT.upper()} (ms) | KERNELS | MB COPIED | "
             f"CPU IN CUDA CALLS (ms)"]
    w0, w1 = a.window
    for i, p in enumerate(a.phases):
        lines.append(f"{p.name} | {_ms(p.wall)} | {_ms(busy.get(i, 0.0))} | {_pct(busy.get(i, 0.0), p.wall)} | "
                     f"{_ms(waiting.get(i, 0.0))} | {kernels[i]} | {copied[i] / 1e6:.1f} | {_ms(in_calls.get(i, 0.0))}")
    if busy.get(-1) or waiting.get(-1) or kernels[-1] or in_calls.get(-1):
        inside = sum(max(0.0, min(p.end, w1) - max(p.start, w0)) for p in a.phases)
        wall = max(0.0, a.wall - inside)
        lines.append(f"{OUTSIDE} | {_ms(wall)} | {_ms(busy.get(-1, 0.0))} | {_pct(busy.get(-1, 0.0), wall)} | "
                     f"{_ms(waiting.get(-1, 0.0))} | {kernels[-1]} | {copied[-1] / 1e6:.1f} | "
                     f"{_ms(in_calls.get(-1, 0.0))}")

    if a.steps:
        step_busy, step_waiting = a.busy_by_step(), a.waiting_by_step()
        walls = [s.wall for s in a.steps]
        busies = [step_busy.get(i, 0.0) for i in range(len(a.steps))]
        waits = [step_waiting.get(i, 0.0) for i in range(len(a.steps))]
        busy_median, wall_median = statistics.median(busies), statistics.median(walls)
        lines.append(f"STEPS: {len(a.steps)} | wall ms {_spread(walls)} | GPU busy median {_ms(busy_median)} ms "
                     f"({_pct(busy_median, wall_median)}%)")
        first = [i for i, s in enumerate(a.steps) if int(_STEP.match(s.name).group(1)) == 1]
        for i in first:
            s = a.steps[i]
            lines.append(f"  first step of {a.phase_name(a.step_phase[i])}: wall {_ms(s.wall)} ms, GPU busy "
                         f"{_ms(busies[i])} ms ({_pct(busies[i], s.wall)}%), {HOST_WAIT} {_ms(waits[i])} ms "
                         f"({_pct(waits[i], s.wall)}%)")
        firsts = set(first)
        rest = [i for i in range(len(a.steps)) if i not in firsts]
        if rest:
            rest_busy = [busies[i] for i in rest]
            lines.append(f"  the other {len(rest)} steps: wall ms {_spread([walls[i] for i in rest])} | GPU busy "
                         f"median {_ms(statistics.median(rest_busy))} ms | {HOST_WAIT} median "
                         f"{_ms(statistics.median([waits[i] for i in rest]))} ms")
    idle = a.wall - a.busy_total
    lines.append(f"TOTAL: wall {_ms(a.wall)} ms | GPU busy {_ms(a.busy_total)} ms ({_pct(a.busy_total, a.wall)}%) | "
                 f"GPU idle {_ms(idle)} ms ({_pct(idle, a.wall)}%), of it {HOST_WAIT} (pageable copies) "
                 f"{_ms(a.waiting_total)} ms ({_pct(a.waiting_total, a.wall)}%)")
    return "\n".join(lines) + "\n"


# ---- b) kernel classes -----------------------------------------------------------------------------------------------

def _readable_rows(raw_names: Iterable[str]) -> dict[str, str]:
    """Raw name -> the name shown (the S2 cleaner's); two raw names that read alike get #2, #3, ... so distinct
    kernels stay distinct rows."""
    shown: dict[str, str] = {}
    seen: Counter[str] = Counter()
    for raw in raw_names:
        text = _clean_kernel_name(raw)
        seen[text] += 1
        shown[raw] = text if seen[text] == 1 else f"{text} #{seen[text]}"
    return shown


def kernel_class_report(events: Events, prefix: str | None = None) -> str:
    """GPU ms (sum of durations), share and calls per kernel class, the TOP_KERNELS kernels of each; then the classes
    per phase."""
    a = _analysis(events, prefix)
    ms: Counter[str] = Counter()
    calls: Counter[str] = Counter()
    per_kernel: dict[str, Counter[str]] = defaultdict(Counter)
    per_kernel_calls: Counter[str] = Counter()
    per_phase: dict[int, Counter[str]] = defaultdict(Counter)
    classes: dict[str, str] = {}
    for g, phase in zip(a.gpu, a.gpu_phase):
        raw = str(g.get("name", "?"))
        kind = classes.get(raw)
        if kind is None:
            kind = classes[raw] = kernel_class(g)
        ms[kind] += g["dur"]
        calls[kind] += 1
        per_kernel[kind][raw] += g["dur"]
        per_kernel_calls[raw] += 1
        per_phase[phase][kind] += g["dur"]
    total = sum(ms.values())
    pageable = sum(t for kind, t in ms.items() if kind.endswith("(pageable)"))
    lines = ["--- GPU TIME BY KERNEL CLASS (cuda_trace_report) ---",
             f"GPU total {_ms(total)} ms (sum of kernel, memcpy and memset durations; {sum(calls.values())} events"
             + (f"; {_ms(pageable)} ms of it pageable copies, mostly the CPU staging them: see {HOST_WAIT} in the "
                f"phases" if pageable else "") + ")",
             "CLASS | GPU (ms) | SHARE % | CALLS"]
    for kind, t in sorted(ms.items(), key=lambda kv: -kv[1]):
        lines.append(f"{kind} | {_ms(t)} | {_pct(t, total)} | {calls[kind]}")
        top = per_kernel[kind].most_common(TOP_KERNELS)
        shown = _readable_rows(raw for raw, _ in per_kernel[kind].most_common())
        for raw, kt in top:
            lines.append(f" {shown[raw]} ({_ms(kt)} ms, {_calls(per_kernel_calls[raw])})")
    if a.phases and total > 0:
        lines.append("PER PHASE (GPU ms by class, largest first)")
        order = [i for i in range(len(a.phases))] + ([-1] if per_phase.get(-1) else [])
        for i in order:
            row = per_phase.get(i)
            if not row:
                continue
            phase_total = sum(row.values())
            parts = [f"{kind} {_ms(t)} ({_pct(t, phase_total)}%)"
                     for kind, t in row.most_common(TOP_CLASSES_PER_PHASE)]
            lines.append(f" {a.phase_name(i)}: " + "; ".join(parts))
    return "\n".join(lines) + "\n"


# ---- c) synchronisation and idle time --------------------------------------------------------------------------------

def sync_report(events: Events, prefix: str | None = None) -> str:
    """Where the CPU waited for the GPU and where the GPU sat idle: blocking CUDA calls, copies by direction and
    memory kind, allocation calls, and the GPU's idle gaps inside the request window (a pageable copy's time is idle
    time, shown apart as waiting for host memory), each labelled by the phase and step holding most of it."""
    a = _analysis(events, prefix)
    w0, w1 = a.window
    idle = a.wall - a.busy_total
    lines = ["--- CPU-GPU SYNCHRONISATION AND GPU IDLE TIME (cuda_trace_report) ---",
             "PyTorch's sync debug mode (sync_hunter.py) only sees synchronisations that PyTorch operations cause, "
             "and Qwen-Image runs none: stable-diffusion.cpp drives the GPU through its own CUDA runtime. So this "
             "reads the CUDA runtime's own blocking calls and the GPU's idle gaps from the CUPTI trace instead.",
             f"Window: {a.window_name}, {_ms(a.wall)} ms | GPU busy {_ms(a.busy_total)} ms "
             f"({_pct(a.busy_total, a.wall)}%; kernels, memsets, pinned / device copies) | GPU idle {_ms(idle)} ms, "
             f"of it {HOST_WAIT} (pageable copies) {_ms(a.waiting_total)} ms ({_pct(a.waiting_total, a.wall)}%)"]

    # -- blocking calls
    rows: dict[str, list[float]] = {}                # name -> [calls, total us, max us]
    by_phase: dict[int, list[float]] = defaultdict(lambda: [0, 0.0])
    in_steps = [0, 0.0]
    other_threads = 0

    def count(name: str, c: dict) -> None:
        nonlocal other_threads
        row = rows.setdefault(name, [0, 0.0, 0.0])
        row[0] += 1
        row[1] += c["dur"]
        row[2] = max(row[2], c["dur"])
        phase = by_phase[a.phase_at(c["ts"])]
        phase[0] += 1
        phase[1] += c["dur"]
        if a.step_at(c["ts"]) >= 0:
            in_steps[0] += 1
            in_steps[1] += c["dur"]
        if a.thread is not None and _thread(c) != a.thread:
            other_threads += 1

    for c in a.calls:
        if api_name(str(c.get("name", ""))) in BLOCKING_CALLS:
            count(api_name(str(c["name"])), c)
    for g in a.gpu:                                  # an async copy from or to pageable memory waits like a sync one
        if g.get("cat") != "gpu_memcpy" or "pageable" not in str(g.get("name", "")).lower():
            continue
        launch = a.launch.get((g.get("args") or {}).get("correlation"))
        if launch is None or not w0 <= launch["ts"] <= w1:
            continue
        name = api_name(str(launch.get("name", "")))
        if name not in BLOCKING_CALLS:                # cudaMemcpy itself is counted above already
            count(f"{name} (pageable: synchronous)", launch)
    lines.append("BLOCKING CALLS (the CPU waits for the GPU) | CALLS | TOTAL (ms) | MAX (ms)")
    if rows:
        for name, (n, total, longest) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
            lines.append(f"{name} | {n} | {_ms(total)} | {_ms(longest)}")
        lines.append(" per phase: " + "; ".join(
            f"{a.phase_name(i)} {_calls(int(by_phase[i][0]))} {_ms(by_phase[i][1])} ms"
            for i in list(range(len(a.phases))) + [-1] if by_phase.get(i) and by_phase[i][0]))
        if a.steps:
            lines.append(f" per step: {in_steps[0] / len(a.steps):.1f} calls, {_ms(in_steps[1] / len(a.steps))} ms "
                         f"on average ({len(a.steps)} steps)")
        if other_threads:
            lines.append(f" {other_threads} of them on other threads than the request's")
    else:
        lines.append(" none inside the window")

    # -- copies
    copies: dict[str, list[float]] = {}               # name -> [count, bytes, us]
    copy_phase: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    pageable = False
    for g, phase in zip(a.gpu, a.gpu_phase):
        if g.get("cat") != "gpu_memcpy" or not (w0 <= g["ts"] <= w1):
            continue
        name = str(g.get("name", "Memcpy"))
        size = float((g.get("args") or {}).get("bytes", 0) or 0)
        row = copies.setdefault(name, [0, 0.0, 0.0])
        row[0] += 1
        row[1] += size
        row[2] += g["dur"]
        copy_phase[phase][0] += size
        copy_phase[phase][1] += g["dur"]
        pageable = pageable or "pageable" in name.lower()
    lines.append("COPIES | CALLS | MB | ms | GB/s")
    if copies:
        for name, (n, size, us) in sorted(copies.items(), key=lambda kv: -kv[1][2]):
            rate = size / us / 1e3 if us > 0 else 0.0
            lines.append(f"{name} | {int(n)} | {size / 1e6:.1f} | {_ms(us)} | {rate:.1f}")
        lines.append(" per phase: " + "; ".join(
            f"{a.phase_name(i)} {copy_phase[i][0] / 1e6:.1f} MB {_ms(copy_phase[i][1])} ms"
            for i in list(range(len(a.phases))) + [-1] if i in copy_phase))
        if pageable:
            lines.append(" Pageable: from or to ordinary (not pinned) memory. Such an async copy is effectively "
                         "synchronous: the CPU stages it and waits (the BLOCKING CALLS above count it). Its ms here "
                         f"are mostly that staging, so they are GPU idle time ({HOST_WAIT}), not GPU busy.")
    else:
        lines.append(" none inside the window")

    # -- allocations
    allocs: dict[str, list[float]] = {}
    for c in a.calls:
        name = api_name(str(c.get("name", "")))
        if name in ALLOCATION_CALLS:
            row = allocs.setdefault(name, [0, 0.0])
            row[0] += 1
            row[1] += c["dur"]
    lines.append("ALLOCATION / VMM CALLS | CALLS | TOTAL (ms)")
    if allocs:
        for name, (n, us) in sorted(allocs.items(), key=lambda kv: -kv[1][1]):
            lines.append(f"{name} | {int(n)} | {_ms(us)}")
    else:
        lines.append(" none inside the window (Kineto records only the driver calls it knows: cuMem* can be missing)")

    # -- idle gaps
    def waited(us: float, text: str) -> str:
        return text.format(f"{_ms(us)} ms {HOST_WAIT}") if us >= SHOWN_US else ""

    gaps = a.gaps()
    longer = [(g0, g1) for g0, g1 in gaps if g1 - g0 > GAP_MIN_US]
    lines.append(f"IDLE GAPS (no kernel, memset or pinned / device copy runs; {HOST_WAIT.upper()}: a pageable copy is "
                 f"in flight meanwhile)")
    lines.append(f" GPU idle inside the window: {_ms(idle)} ms, of it {HOST_WAIT} {_ms(a.waiting_total)} ms; gaps "
                 f"over {GAP_MIN_US / 1000:g} ms: {len(longer)} ({_ms(sum(g1 - g0 for g0, g1 in longer))} ms)")
    if a.phases:
        parts = []
        for p in a.phases:
            free = p.wall - a.busy_between(p.start, p.end)
            parts.append(f"{p.name} {_ms(free)} ms ({_pct(free, p.wall)}%"
                         + waited(a.waiting_between(p.start, p.end), ", {}") + ")")
        lines.append(" per phase: " + "; ".join(parts))
    if a.steps:
        inside = [s.wall - a.busy_between(s.start, s.end) for s in a.steps]
        lines.append(f" inside the steps: {_ms(sum(inside))} ms ({_ms(sum(inside) / len(inside))} ms per step on "
                     f"average)" + waited(sum(a.waiting_between(s.start, s.end) for s in a.steps), ", of it {}"))
        # A step's annotation closes when sd.cpp reports it done (after its graph's synchronisation): the GPU idle
        # gap holding that moment is what one step boundary costs.
        gap_starts = [g0 for g0, _ in gaps]

        def gap_at(t: float) -> int:
            i = bisect.bisect_right(gap_starts, t) - 1
            return i if i >= 0 and t <= gaps[i][1] else -1

        # a boundary inside GPU work costs 0; one gap holding several boundaries (idle across whole steps) counts once
        boundaries: list[float] = []
        held: set[int] = set()
        for i in range(len(a.steps) - 1):
            if a.step_phase[i] != a.step_phase[i + 1]:
                continue
            gap = gap_at(a.steps[i].end)
            if gap < 0:
                boundaries.append(0.0)
            elif gap not in held:
                held.add(gap)
                boundaries.append(gaps[gap][1] - gaps[gap][0])
        if boundaries:
            lines.append(f" at the boundaries between steps: {len(boundaries)}, the GPU idle there median "
                         f"{_ms(statistics.median(boundaries))} ms, max {_ms(max(boundaries))} ms, total "
                         f"{_ms(sum(boundaries))} ms")
    if gaps:
        lines.append(f" the {min(TOP_GAPS, len(gaps))} largest (WHERE: the phase and step holding most of the gap):")
        lines.append(f" OFFSET (ms) | IDLE (ms) | OF IT {HOST_WAIT.upper()} (ms) | WHERE | THE REQUEST THREAD")
        for g0, g1 in sorted(gaps, key=lambda g: g[0] - g[1])[:TOP_GAPS]:
            doing = a.call_during(g0, g1)
            what = (f"in {doing[0]} for {_ms(doing[1])} ms of it" + (f" ({_calls(doing[2])})" if doing[2] > 1 else "")
                    if doing else "no CUDA call: CPU work in sd.cpp or Python")
            lines.append(f" {_ms(g0 - w0)} | {_ms(g1 - g0)} | {_ms(a.waiting_between(g0, g1))} | "
                         f"{a.where_most(g0, g1)} | {what}")
    return "\n".join(lines) + "\n"


# ---- the command line ------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="Chrome trace JSON written by torch.profiler (request_profiler.py's trace.json)")
    ap.add_argument("--prefix", default=None, help='the request\'s annotation prefix, e.g. "qwen:" (default: found)')
    ap.add_argument("--report", choices=("all", "phases", "kernels", "sync"), default="all")
    args = ap.parse_args(argv)
    analysis = TraceAnalysis(load_trace(args.trace), args.prefix)
    parts = {"phases": phase_report, "kernels": kernel_class_report, "sync": sync_report}
    for key, report in parts.items():
        if args.report in ("all", key):
            sys.stdout.write(report(analysis) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
