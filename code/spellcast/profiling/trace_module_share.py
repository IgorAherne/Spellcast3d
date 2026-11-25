# From AURA-V5 src/world_model/utils/trace_module_share.py (commit 41007db), adapted for Spellcast3D:
#   - REPO_MARKERS name Spellcast3D's folders, with / and with \ (torch's python_function names carry Windows paths
#     here), instead of AURA's;
#   - the usage line runs it as spellcast.profiling.trace_module_share;
#   - main()'s note for a trace without Python frames: for request_profiler's trace (it has a <label>:request
#     annotation) it says the launches go to the request's annotations, instead of AURA's "use a training-window
#     trace", which Spellcast3D does not make; the trace argument's help names both kinds of trace.
#   Otherwise verbatim. The trace is the trace.json request_profiler.py writes with use_torch_profiler for a Qwen-Image
#   request - recorded with_stack=False, so it has no Python frames and no nn.Module - or a torch tool's
#   trace_{step}_{ts}.json (with_stack=True). Qwen-Image's CUDA kernels are stable-diffusion.cpp's, launched from
#   inside the ctypes call: the innermost-user-annotation fallback below charges them to the request's own
#   annotations, i.e. to <qwen:step k/n> while drawing and to <qwen:prompt>, <qwen:finishing>, ... otherwise: GPU ms
#   per step and phase. The module attribution itself is for the torch tools.
"""GPU time per top-level nn.Module, read from a torch.profiler Chrome trace.

    python -m spellcast.profiling.trace_module_share <trace.json> [--depth 1] [--top 30]

The trace is the trace_{step}_{ts}.json that profiling_harness writes with use_torch_profiler
(with_stack=True, which the harness sets). Every kernel / memcpy / memset is charged to the code
that launched it: its correlation id names the CUDA runtime (or driver) call, and that call's
enclosing CPU events on its thread hold the nn.Module frames, outermost first. The module path is
cut at --depth (0 = the outermost module, 1 = its children, ...). Backward kernels run on the
autograd thread, where no module frame exists: they are charged to the forward op that created the
autograd node (same sequence number), i.e. to that op's module. A launch outside every module goes
to the innermost user annotation (Optimizer.step, ...) or else to the innermost repo Python frame,
marked <no module>. Prints the total GPU ms, then per bucket: ms, share of the total, forward /
backward split. The whole trace is held in memory: ~5x the file size in RAM.
"""
import argparse
import collections
import json
from typing import Dict, List, Tuple

GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
LAUNCH_CATS = ("cuda_runtime", "cuda_driver")      # cuBLASLt / cutlass launch through cuLaunchKernel
CPU_CATS = ("cpu_op", "python_function", "user_annotation") + LAUNCH_CATS
MODULE_PREFIX = "nn.Module: "
BACKWARD_PREFIX = "autograd::engine::evaluate_function: "
STEP_PREFIX = "ProfilerStep#"
REPO_MARKERS = ("spellcast/", "launchers/", "profiling/", "spellcast\\", "launchers\\", "profiling\\")
EPS_US = 0.05          # Kineto timestamps are float microseconds; nesting tolerates this rounding


class ThreadTree:
    """Nesting of one thread's CPU events: parent[i] is the innermost event containing event i."""

    def __init__(self, events: List[dict]):
        self.ev = sorted(events, key=lambda e: (e["ts"], -e["dur"]))
        self.parent = [-1] * len(self.ev)
        stack = []
        for i, e in enumerate(self.ev):
            end = e["ts"] + e["dur"]
            while stack and self.ev[stack[-1]]["ts"] + self.ev[stack[-1]]["dur"] + EPS_US < end:
                stack.pop()
            self.parent[i] = stack[-1] if stack else -1
            stack.append(i)
        self._modules: Dict[int, tuple] = {}

    def ancestors(self, i: int):
        i = self.parent[i]
        while i >= 0:
            yield self.ev[i]
            i = self.parent[i]

    def module_path(self, i: int) -> tuple:
        """nn.Module names enclosing event i, outermost first (memoised along the parent chain)."""
        chain = []
        j = i
        while j >= 0 and j not in self._modules:
            chain.append(j)
            j = self.parent[j]
        path = self._modules[j] if j >= 0 else ()
        for k in reversed(chain):
            name = self.ev[k]["name"]
            if self.ev[k]["cat"] == "python_function" and name.startswith(MODULE_PREFIX):
                path = path + (name[len(MODULE_PREFIX):],)
            self._modules[k] = path
        return path


class Attributor:
    def __init__(self, events: List[dict], depth: int):
        self.depth = depth
        by_thread = collections.defaultdict(list)
        for e in events:
            if e.get("cat") in CPU_CATS:
                by_thread[(e["pid"], e["tid"])].append(e)
        self.launch = {}          # correlation -> (tree, index) of the launching runtime/driver call
        # autograd sequence number -> (tree, index) of the forward op that created the node. The
        # node's own backward op carries the same number on the autograd thread, always later.
        self.fwd_by_seq = {}
        first_ts = {}
        for events_t in by_thread.values():
            tree = ThreadTree(events_t)
            for i, e in enumerate(tree.ev):
                args = e.get("args", {})
                if e["cat"] in LAUNCH_CATS and "correlation" in args:
                    self.launch[args["correlation"]] = (tree, i)
                elif e["cat"] == "cpu_op" and "Sequence number" in args:
                    seq = args["Sequence number"]
                    if seq not in first_ts or e["ts"] < first_ts[seq]:
                        first_ts[seq] = e["ts"]
                        self.fwd_by_seq[seq] = (tree, i)

    def _site(self, tree: ThreadTree, i: int) -> str:
        """Bucket of a forward-side CPU event: module path, else annotation, else repo frame."""
        path = tree.module_path(i)
        if path:
            return " > ".join(path[:self.depth + 1])
        repo_frame = None
        for a in tree.ancestors(i):
            if a["cat"] == "user_annotation" and not a["name"].startswith(STEP_PREFIX):
                return f"<{a['name']}>"
            if repo_frame is None and a["cat"] == "python_function" and any(m in a["name"] for m in REPO_MARKERS):
                repo_frame = a["name"]
        return f"<no module> {repo_frame}" if repo_frame else "<no module>"

    def charge(self, gpu_event: dict) -> Tuple[str, bool]:
        """(bucket, is_backward) of one GPU event."""
        hit = self.launch.get(gpu_event.get("args", {}).get("correlation"))
        if hit is None:
            return "<launch call not in the trace>", False
        tree, i = hit
        if not tree.module_path(i):
            for a in tree.ancestors(i):
                args = a.get("args", {})
                if a["cat"] == "cpu_op" and a["name"].startswith(BACKWARD_PREFIX) and "Sequence number" in args:
                    fwd = self.fwd_by_seq.get(args["Sequence number"])
                    return (self._site(*fwd) if fwd else "<backward, forward op not in the trace>"), True
        return self._site(tree, i), False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="Chrome trace JSON written by torch.profiler (with_stack=True for the module "
                                  "attribution; a Qwen-Image request's trace.json has annotations only)")
    ap.add_argument("--depth", type=int, default=1,
                    help="module-path levels kept: 0 = outermost module, 1 = its children (default)")
    ap.add_argument("--top", type=int, default=30, help="rows printed (default 30)")
    a = ap.parse_args()

    with open(a.trace, encoding="utf-8") as f:
        events = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X"]
    gpu = [e for e in events if e.get("cat") in GPU_CATS]
    steps = sum(1 for e in events if e.get("cat") == "user_annotation" and e["name"].startswith(STEP_PREFIX))
    att = Attributor(events, a.depth)

    total = sum(g["dur"] for g in gpu)
    by_cat = collections.Counter()
    fwd = collections.Counter()
    bwd = collections.Counter()
    for g in gpu:
        by_cat[g["cat"]] += g["dur"]
        bucket, backward = att.charge(g)
        (bwd if backward else fwd)[bucket] += g["dur"]

    rows = sorted(set(fwd) | set(bwd), key=lambda k: -(fwd[k] + bwd[k]))
    cats = ", ".join(f"{c.replace('gpu_', '')} {by_cat[c] / 1e3:.1f}" for c in GPU_CATS)
    print(f"GPU total {total / 1e3:.1f} ms ({cats}) over {steps} profiler step(s)")
    if not any(e.get("cat") == "python_function" for e in events):
        # [SPELLCAST3D] request_profiler's trace (its root annotation is <label>:request): with_stack=False on
        # purpose, and its annotations name the request's parts
        request = next((e["name"] for e in events
                        if e.get("cat") == "user_annotation" and e["name"].endswith(":request")), None)
        if request is not None:
            label = request[:-len("request")]
            print(f"NOTE: no Python frames in this trace (request_profiler records a request with_stack=False), so "
                  f"nothing is charged to an nn.Module: each launch goes to its innermost user annotation "
                  f"(<{label}step k/n>, <{label}prompt>, ...), as intended for a Qwen-Image request.")
        else:
            print("NOTE: no Python frames in this trace (recorded with_stack=False, like the calib_* phase "
                  "traces), so nothing can be charged to a module; use a training-window trace.")
    print(f"{'ms':>10} {'share':>6} {'fwd ms':>9} {'bwd ms':>9}  module path (depth {a.depth})")
    for k in rows[:a.top]:
        ms = fwd[k] + bwd[k]
        print(f"{ms / 1e3:10.1f} {100 * ms / max(total, 1e-9):5.1f}% {fwd[k] / 1e3:9.1f} {bwd[k] / 1e3:9.1f}  {k}")
    if len(rows) > a.top:
        rest = sum(fwd[k] + bwd[k] for k in rows[a.top:])
        print(f"{rest / 1e3:10.1f} {100 * rest / max(total, 1e-9):5.1f}%  ({len(rows) - a.top} more rows)")


if __name__ == "__main__":
    main()
