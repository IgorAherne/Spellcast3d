"""Check of the staged engine's per-prop store (pixal3d/staged/store.py of a fork worktree): every item type
through RAM, through a slot and through disk, the RAM budget and what the process's memory does after a run that
spills, the page-locked slots and their uploads, and what happens when files, the disk or a checkpoint misbehave.

    <pixal3d python> check_store.py --repo <worktree>                    # on the CPU: no card, no lock, no GOLD
    <pixal3d python> check_store.py --repo <worktree> --device cuda      # + the card (takes the GPU lock, 1-2 min)
    --only a,b        run only these checks (--list names them)
    --stfile PATH     use this stfile.py instead of the worktree's own (--types PATH: the same for types.py)
    --keep            leave the scratch folder (<worktree>/tmp/check-store-<pid>) in place
    --only allocator  not a check of the store: what torch's CPU allocator does with large blocks that are freed
                      (by the thread that made them, by another one), each case in a process of its own
    --wrong-thread    a test of this check itself: the spilled tensors are freed by another thread than the caller's.
                      `--only ram_budget --wrong-thread` must FAIL in two lines (private bytes; who let go)

The store is loaded alone, without the rest of the pixal3d package (it needs only torch, stfile and the
exceptions of types.py). A worktree that has no pixal3d/staged/stfile.py or types.py yet gets the stand-ins at the
end of this file, written to the same contract; the first lines of the output say which were used.
"""

from __future__ import annotations

import sys

import common                                           # first: before torch and before anything of Pixal3D

import gc
import importlib
import importlib.util
import json
import os
import random
import shutil
import struct
import subprocess
import threading
import time
import traceback
import types
import weakref
from pathlib import Path

MIB = 1 << 20
CPU_ONLY = {"small_items", "discard_during_read", "held_file", "full_disk"}       # file logic: the same on a card
CARD_ONLY = {"pinned", "pump_timing", "cap_rule", "overwrite_control"}
ASKED_ONLY = {"allocator"}                              # a measurement, run with --only allocator


def arguments():
    parser = common.parser("the per-prop store: items, RAM budget, slots, uploads, files, checkpoints")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--only", default="", help="comma-separated names of the checks to run")
    parser.add_argument("--list", action="store_true", help="print the names of the checks and stop")
    parser.add_argument("--stfile", default="", help="another stfile.py than the worktree's own")
    parser.add_argument("--types", default="", help="another types.py than the worktree's own")
    parser.add_argument("--keep", action="store_true", help="keep the scratch folder")
    parser.add_argument("--wrong-thread", action="store_true",
                        help="test of the check: free the spilled tensors on another thread; ram_budget must fail")
    return parser.parse_args()


# =====================================================================================================================
# Loading the store alone
# =====================================================================================================================

def load_store(repo: Path, stfile_path: str, types_path: str) -> tuple[types.ModuleType, list[str]]:
    """pixal3d/staged/store.py of the worktree as `pixal3d.staged.store`, with empty stand-in packages above it so
    that neither pixal3d/__init__.py nor staged/__init__.py runs. Returns the module and what was stood in."""
    staged = repo / "pixal3d" / "staged"
    notes: list[str] = []
    for name, folder in (("pixal3d", repo / "pixal3d"), ("pixal3d.staged", staged)):
        package = types.ModuleType(name)
        package.__path__ = [str(folder)]
        sys.modules[name] = package
    sys.modules["pixal3d"].staged = sys.modules["pixal3d.staged"]

    def install(short: str, module: types.ModuleType) -> None:
        sys.modules[f"pixal3d.staged.{short}"] = module
        setattr(sys.modules["pixal3d.staged"], short, module)

    def from_file(short: str, path: str) -> None:
        spec = importlib.util.spec_from_file_location(f"pixal3d.staged.{short}", path)
        module = importlib.util.module_from_spec(spec)
        install(short, module)
        spec.loader.exec_module(module)

    for short, path, stand_in in (("stfile", stfile_path, stand_in_stfile), ("types", types_path, stand_in_types)):
        if path:
            from_file(short, path)
            notes.append(f"{short}: {path}")
        elif (staged / f"{short}.py").is_file():
            notes.append(f"{short}: the worktree's own")
        else:
            install(short, stand_in())
            notes.append(f"{short}: the STAND-IN of check_store.py (the worktree has no pixal3d/staged/{short}.py)")
    return importlib.import_module("pixal3d.staged.store"), notes


def free_on_wrong_thread(module) -> None:
    """Break the store's one-thread rule on purpose: when an item has reached its file, its tensors' last
    reference is dropped by another thread. Everything else, the bookkeeping included, stays as it is."""
    real = module.PropStore._spilled

    def spilled(self, job) -> None:
        last = dict(job.item.ram or {})
        real(self, job)
        thread = threading.Thread(target=last.clear, name="wrong-thread")
        thread.start()
        thread.join()

    module.PropStore._spilled = spilled


# =====================================================================================================================
# Helpers
# =====================================================================================================================

class Context:
    def __init__(self, module, device: str, scratch: Path):
        self.module = module
        self.device = device
        self.cuda = device == "cuda"
        self.scratch = scratch
        self.results: list[common.Result] = []
        self.current = ""

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append(common.Result(f"{self.current}/{name}", bool(ok), "store", detail))
        print(self.results[-1], flush=True)
        return bool(ok)


class Bench:
    """A store in a folder of its own, with its events collected."""

    def __init__(self, ctx: Context, name: str, device: str | None = None, **options):
        self.ctx = ctx
        self.folder = ctx.scratch / name
        self.events: list[dict] = []
        self.store = ctx.module.PropStore(self.folder, device or ctx.device, on_event=self.events.append, **options)

    def warnings(self, what: str | None = None) -> list[dict]:
        return [event for event in self.events if what is None or event["what"] == what]

    def files(self) -> list[str]:
        """Every file under the work folder, relative, sorted."""
        return sorted(str(path.relative_to(self.folder)).replace("\\", "/")
                      for path in self.folder.rglob("*") if path.is_file())

    def slot_states(self) -> list[str]:
        return [slot.state for slot in self.store._slots or []]

    def close(self) -> None:
        self.store.close()


def noise(seed: int, dtype, shape) -> "torch.Tensor":
    """A tensor of every bit pattern (NaN payloads, denormals, negative zeros included)."""
    generator = torch.Generator().manual_seed(seed)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=generator, dtype=torch.uint8).to(torch.bool)
    count = 1
    for side in shape:
        count *= side
    raw = torch.randint(0, 256, (count * torch.empty((), dtype=dtype).element_size(),), dtype=torch.uint8,
                        generator=generator)
    return raw.view(dtype).view(shape)


def big(seed: int, mib: int, base: "torch.Tensor | None" = None) -> dict:
    """A conditioning of about `mib` MiB: {"global": bf16 [1,5,1024], "proj": bf16 [rows,2048]}."""
    rows = mib * MIB // 4096
    generator = torch.Generator().manual_seed(seed)
    proj = torch.empty(rows, 2048, dtype=torch.int16).random_(-32768, 32767, generator=generator)
    return {"global": noise(seed + 1, torch.bfloat16, (1, 5, 1024)), "proj": proj.view(torch.bfloat16)}


def raw_bytes(tensor) -> "torch.Tensor":
    tensor = tensor.detach().cpu().contiguous().reshape(-1)
    return tensor.view(torch.uint8) if tensor.numel() else tensor.new_empty(0, dtype=torch.uint8)


def differs(got: dict, want: dict, device: str | None = None) -> str | None:
    """None when `got` holds the tensors of `want` bit for bit (names, order, dtype, shape, bytes; and the device)."""
    if list(got) != list(want):
        return f"names {list(got)} != {list(want)}"
    for name, expected in want.items():
        tensor = got[name]
        if device is not None and tensor.device.type != device:
            return f"{name} is on {tensor.device}, not on {device}"
        if tensor.dtype != expected.dtype or tensor.shape != expected.shape:
            return f"{name} is {tensor.dtype} {list(tensor.shape)}, put was {expected.dtype} {list(expected.shape)}"
        a, b = raw_bytes(tensor), raw_bytes(expected)
        if not torch.equal(a, b):
            unequal = a != b
            return f"{name}: {int(unequal.sum())} of {a.numel()} bytes differ, first at {int(unequal.int().argmax())}"
    return None


def on_device(ctx: Context, tensors: dict) -> dict:
    return {name: tensor.to(ctx.device) for name, tensor in tensors.items()} if ctx.cuda else tensors


def staged(store, count: int, seconds: float = 10.0) -> list[str]:
    """Wait until `count` slots hold a staged item (their fills are done and the copies before them have run)."""
    deadline = time.monotonic() + seconds
    while True:
        store.flush()
        states = [slot.state for slot in store._slots or []]
        if states.count("staged") >= count or time.monotonic() > deadline:
            return states
        time.sleep(0.001)


def tensors_fit(store, tensors: dict) -> bool:
    """Does an item of these tensors fit one slot of the store."""
    need = sum(-(-tensor.numel() * tensor.element_size() // 64) * 64 for tensor in tensors.values())
    return bool(store._slot_count) and need <= store._slot_bytes


def settle(store) -> None:
    """File work done and every upload's copies run."""
    store.flush()
    if store._slots:
        for slot in store._slots:
            store._synchronise(slot)


class Freed:
    """Which thread let go of each tensor the store kept in RAM: `with Freed(module) as freed:` wraps the store's
    one place where a tensor enters its RAM cache or a checkpoint job, and notes the thread on which the tensor
    object dies. The store's rule is that only the thread that calls it (the GPU thread) frees tensor memory."""

    def __init__(self, module):
        self.module = module
        self.threads: dict[str, int] = {}
        self.kept = 0

    def _note(self) -> None:
        name = threading.current_thread().name
        self.threads[name] = self.threads.get(name, 0) + 1

    def __enter__(self) -> "Freed":
        self.real = self.module._resting

        def resting(tensor):
            value = self.real(tensor)
            self.kept += 1
            weakref.finalize(value, self._note)
            return value

        self.module._resting = resting
        return self

    def __exit__(self, *exc) -> None:
        self.module._resting = self.real

    def verdict(self) -> tuple[bool, str]:
        gc.collect()
        mine = threading.current_thread().name
        return (self.threads == {mine: self.kept} if self.kept else not self.threads,
                f"{self.kept} tensors kept by the store, let go of by: {self.threads or 'nobody yet'}")


def private_mib() -> float:
    info = psutil.Process().memory_info()
    return getattr(info, "private", info.rss) / MIB


def table_items() -> dict[str, dict]:
    """The large items of the specification's table (1.3) at sizes of a typical prop, and one of every dtype."""
    return {
        "dino512": {"z": noise(1, torch.float32, (2, 1029, 1024))},
        "dino1024": {"z": noise(2, torch.float32, (1, 4101, 1024))},
        "cond_ss": {"global": noise(3, torch.bfloat16, (1, 5, 1024)),
                    "proj": noise(4, torch.bfloat16, (1, 4096, 1024))},
        "cond_lr": {"global": noise(5, torch.bfloat16, (1, 5, 1024)), "proj": noise(6, torch.bfloat16, (3273, 2048))},
        "cond_hr": {"global": noise(7, torch.bfloat16, (1, 5, 1024)), "proj": noise(8, torch.bfloat16, (14041, 2048))},
        "cond_tex": {"global": noise(9, torch.bfloat16, (1, 5, 1024)),
                     "proj": noise(10, torch.bfloat16, (14041, 2048))},
        "zoo": {"i32": noise(11, torch.int32, (7, 4)), "u8": noise(12, torch.uint8, (5,)),
                "flag": noise(13, torch.bool, (9,)), "f16": noise(14, torch.float16, (3, 3)),
                "i64": noise(15, torch.int64, (4,)), "scalar": noise(16, torch.float32, ()),
                "none": noise(17, torch.float32, (0, 32)), "odd": noise(18, torch.uint8, (3,)),
                "bf16": noise(19, torch.bfloat16, (33,))},
    }


# =====================================================================================================================
# The checks
# =====================================================================================================================

def check_small_items(ctx: Context) -> None:
    """Small items: the object put is the object got; discard and close_prop let go of it."""
    from PIL import Image

    bench = Bench(ctx, "small", ram_mib=64)
    store = bench.store
    store.begin({})
    store.open_prop("a")
    store.open_prop("b")
    request = {"seed": 11, "resolution": 1024, "multi": False, "sampler_params": {"ss": {"steps": 12}}}
    picture = Image.new("RGBA", (1024, 1024), (10, 20, 30, 255))
    coords = {"coords": noise(1, torch.int32, (3273, 4))}
    rng = {"state": torch.get_rng_state()}
    for key, value in (("request", request), ("image_0", picture), ("coords_ss", coords), ("rng_ss", rng)):
        store.put_small("a", key, value)
    store.put_small("b", "request", {"seed": 12})
    ctx.record("the object itself comes back",
               store.get_small("a", "request") is request and store.get_small("a", "image_0") is picture
               and store.get_small("a", "coords_ss") is coords and store.get_small("a", "rng_ss") is rng)
    ctx.record("props do not see each other's items", store.get_small("b", "request") == {"seed": 12})
    expected = 1024 * 1024 * 4 + 3273 * 16 + rng["state"].numel()
    ctx.record("stats count their bytes", store.stats()["small_bytes"] == expected,
               f"{store.stats()['small_bytes']} bytes, expected {expected}")
    store.put_small("a", "rng_ss", {"state": torch.zeros(4, dtype=torch.uint8)})
    ctx.record("a second put replaces", store.get_small("a", "rng_ss")["state"].numel() == 4)
    store.discard("a", "request", "image_0", "not-there")
    missing = []
    for key in ("request", "image_0"):
        try:
            store.get_small("a", key)
        except KeyError:
            missing.append(key)
    ctx.record("discard lets go", missing == ["request", "image_0"] and store.get_small("a", "coords_ss") is coords)
    store.close_prop("a")
    try:
        store.get_small("a", "coords_ss")
        closed = False
    except KeyError:
        closed = True
    ctx.record("close_prop lets go of all", closed and store.stats()["small_bytes"] == 0
               and store.get_small("b", "request") == {"seed": 12}, f"{store.stats()['small_bytes']} bytes left")
    refused = []
    for bad in ("../x", "a/b", "a\\b", "", ".", "..", "c:", "x ", "x."):
        try:
            store.open_prop(bad)
        except ValueError:
            refused.append(bad)
    ctx.record("a prop id that is not a plain name is refused", len(refused) == 9, f"refused {refused}")
    try:
        store.put_small("never-opened", "request", {})
        strict = False
    except KeyError:
        strict = True
    ctx.record("a prop that is not open cannot be put to", strict)
    store.flush()
    ctx.record("small items never reach the disk", bench.files() == [], f"files: {bench.files()}")
    ctx.record("no warning", not bench.events, f"{bench.events[:2]}")
    bench.close()


def check_round_trip(ctx: Context) -> None:
    """Every large item type, bit for bit, by every way from put_device to take."""
    import safetensors.torch

    items = table_items()
    ways = {            # name: (store options, want it first, pump before take)
        "RAM, copied at take": (dict(ram_mib=1024, slots=2, slot_mib=256), False, False),
        "RAM -> slot -> uploaded by pump": (dict(ram_mib=1024, slots=2, slot_mib=256), True, True),
        "RAM -> slot -> copied at take": (dict(ram_mib=1024, slots=2, slot_mib=256), True, False),
        "disk -> slot -> uploaded by pump": (dict(ram_mib=0, slots=2, slot_mib=256), True, True),
        "disk -> slot -> copied at take": (dict(ram_mib=0, slots=2, slot_mib=256), True, False),
        "disk -> slot at take (never wanted)": (dict(ram_mib=0, slots=2, slot_mib=256), False, False),
        "disk -> buffer at take (larger than a slot)": (dict(ram_mib=0, slots=2, slot_mib=1), True, True),
        "RAM, larger than a slot": (dict(ram_mib=1024, slots=2, slot_mib=1), True, True),
        "disk, no slots at all": (dict(ram_mib=0, slots=0, slot_mib=1), True, True),
    }
    for index, (way, (options, want, pump)) in enumerate(ways.items()):
        bench = Bench(ctx, f"round{index}", **options)
        store = bench.store
        store.begin({})
        store.open_prop("p")
        problems: list[str] = []
        for key, tensors in items.items():
            store.put_device("p", key, on_device(ctx, tensors))
        store.flush()
        stats = store.stats()
        if options["ram_mib"] == 0 and (stats["spills"] != len(items) or stats["ram_bytes"] != 0):
            problems.append(f"expected all {len(items)} items on disk: {stats}")
        if options["ram_mib"] and stats["spills"]:
            problems.append(f"nothing should have spilled: {stats}")
        if options["ram_mib"] == 0 and index == 3:
            for key, tensors in items.items():              # the files are safetensors any reader understands
                loaded = safetensors.torch.load_file(str(bench.folder / "p" / f"{key}.safetensors"))
                problem = differs({name: loaded[name] for name in tensors}, tensors)
                if problem:
                    problems.append(f"the file of {key}, read with the safetensors library: {problem}")
        for key, tensors in items.items():
            before = store.stats()
            if index == 5 and store._slots is None:
                store._make_slots()                         # the slots exist, but this item was never wanted
            if want:
                store.want("p", key)
                staged(store, 1 if tensors_fit(store, tensors) else 0)
                if pump:
                    store.pump()
            got = store.take("p", key)
            problem = differs(got, tensors, ctx.device)
            after = store.stats()
            ahead = after["uploads_ahead"] - before["uploads_ahead"]
            if "uploaded by pump" in way and ahead != 1:
                problem = problem or "pump did not upload it ahead"
            if ("larger than a slot" in way and not tensors_fit(store, tensors) or not pump) and ahead:
                problem = problem or "it was uploaded ahead, which this way excludes"
            if not problem and not ctx.cuda:
                for tensor in got.values():                 # the caller owns what it got: writing to it changes
                    if tensor.numel():                      # nothing in the store
                        first = tensor.reshape(-1).view(torch.uint8)[:1]
                        first.copy_(1 - first if tensor.dtype == torch.bool else first ^ 0xFF)
                again = store.take("p", key)
                problem = differs(again, tensors, ctx.device)
                problem = problem and f"second take after the first was written to: {problem}"
            if problem:
                problems.append(f"{key}: {problem}")
            del got
            settle(store)
            store.discard("p", key)
        settle(store)
        store.close_prop("p")
        store.flush()
        end = store.stats()
        if end["ram_bytes"] or end["disk_bytes"] or end["device_bytes"] or bench.files() or bench.folder.joinpath(
                "p").exists():
            problems.append(f"left behind: {end}, files {bench.files()}")
        if bench.events:
            problems.append(f"warnings: {bench.events[:2]}")
        ctx.record(way, not problems, "; ".join(problems) if problems else
                   f"{len(items)} items, {sum(len(v) for v in items.values())} tensors bit-equal "
                   f"(f32, bf16, f16, i32, i64, u8, bool, 0-dim, empty)")
        bench.close()


def check_ram_budget(ctx: Context) -> None:
    """ram_mib=64 and 6 props of 200 MiB: the RAM cache holds its limit plus the one item on its way, and the
    process's private bytes are back where they started when the run is over."""
    limit, size, count = 64, 200, 6
    freed = Freed(ctx.module).__enter__()
    bench = Bench(ctx, "budget", ram_mib=limit, slots=2, slot_mib=256)
    store = bench.store
    store.begin({})
    highest_cache = 0
    highest_private = 0.0
    pumps: dict[str, list[float]] = {"nothing to do": [], "dropped a spilled item": [], "issued an upload": []}
    problems: list[str] = []

    def look() -> None:
        nonlocal highest_cache, highest_private
        highest_cache = max(highest_cache, store.stats()["ram_bytes"])
        highest_private = max(highest_private, private_mib())

    def pump() -> None:
        before = store.stats()
        began = time.perf_counter()
        store.pump()
        took = time.perf_counter() - began
        after = store.stats()
        what = ("issued an upload" if after["uploads_ahead"] != before["uploads_ahead"] else
                "dropped a spilled item" if after["spills"] != before["spills"] else "nothing to do")
        pumps[what].append(took)
        look()

    def rested() -> float:
        settle(store)
        if ctx.cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        time.sleep(0.2)
        return private_mib()

    def run() -> dict:
        """The six props: every conditioning into the store (and so to disk), then back and compared."""
        digests: dict[str, dict] = {}
        for index in range(count):
            prop = f"p{index}"
            store.open_prop(prop)
            tensors = big(index, size)
            digests[prop] = {name: common.sha256_of(tensor) for name, tensor in tensors.items()}
            tensors = on_device(ctx, tensors)
            store.put_device(prop, "cond_hr", tensors)
            del tensors
            look()
            for _ in range(20):
                pump()
                time.sleep(0.002)
        store.flush()
        look()
        resting = store.stats()
        for index in range(count):
            prop = f"p{index}"
            if index + 1 < count:
                store.want(f"p{index + 1}", "cond_hr")
            for _ in range(5):
                pump()
            got = store.take(prop, "cond_hr")
            for name, digest in digests[prop].items():
                if common.sha256_of(got[name]) != digest or got[name].device.type != ctx.device:
                    problems.append(f"{prop}/{name}")
            del got
            look()
            store.discard(prop, "cond_hr")
            store.close_prop(prop)
            pump()
        return resting

    # The first run is not measured: it takes the process past what happens once (torch's CPU allocator keeps the
    # first eight blocks of 60 MiB or more that are given back to it; the driver sets up its buffers for copies
    # to and from pageable memory). The second, identical run must leave the private bytes where they were.
    fresh = rested()
    run()
    start = rested()
    before = store.stats()
    resting = run()
    end = rested()
    left = store.stats()

    spills = resting["spills"] - before["spills"]
    ctx.record("all six went to disk and the cache rests under its limit",
               spills == count and resting["ram_bytes"] <= limit * MIB,
               f"{spills} spills, {(resting['spilled_bytes'] - before['spilled_bytes']) // MIB} MiB written, cache at "
               f"rest {resting['ram_bytes'] // MIB} MiB (limit {limit}), on disk {resting['disk_bytes'] // MIB} MiB")
    ctx.record("the cache never held more than its limit and the item on its way",
               highest_cache <= (limit + size + 1) * MIB and left["peak_ram_bytes"] <= (limit + size + 1) * MIB,
               f"highest {left['peak_ram_bytes'] / MIB:.0f} MiB (limit {limit} + one item of {size})")
    ctx.record("every prop's 200 MiB came back bit-equal from disk", not problems, f"wrong: {problems}" if problems
               else f"SHA-256 of 2 x {count} x 2 tensors; {left['uploads_ahead']} uploaded ahead by pump, "
                    f"{left['uploads_at_take']} at take")
    ctx.record("nothing is left in the store", not (left["ram_bytes"] or left["disk_bytes"] or left["device_bytes"]
                                                    or left["queued"] or bench.files()),
               f"{left}, files {bench.files()}")
    ctx.record("private bytes return to the start value (the slots are fixed)", abs(end - start) <= 32,
               f"second run: start {start:.0f} MiB, highest {highest_private:.0f} MiB, end {end:.0f} MiB: "
               f"{end - start:+.0f} MiB after {count} x {size} MiB through RAM, disk and back. (The first run, with "
               f"what happens once in a process: {start - fresh:+.0f} MiB, of which the slots "
               f"{left['slot_bytes'] // MIB} MiB, page-locked {left['pinned_bytes'] // MIB} MiB.)")
    shown = "; ".join(f"{what}: {len(times)} calls, longest {max(times) * 1e3:.2f} ms"
                      for what, times in pumps.items() if times)
    limit_ms = 5.0 if ctx.cuda else 250.0                   # on the CPU an "upload" is a copy of 200 MiB in RAM;
    #                                                         on a card the call may get the memory from the driver
    ctx.record("pump stays short during the run", max(pumps["nothing to do"]) < 1e-3
               and max(pumps["issued an upload"] or [0.0]) * 1e3 < limit_ms, shown)
    ctx.record("no warning", not bench.events, f"{bench.events[:2]}")
    bench.close()
    freed.__exit__()
    ctx.record("every tensor of the RAM cache was let go of by the calling thread, none by the helper thread",
               *freed.verdict())

    # What the one-thread rule is for, measured on this build: the same amount let go of by ANOTHER thread.
    gc.collect()
    time.sleep(0.2)
    before = private_mib()
    held = [torch.empty(size * MIB, dtype=torch.uint8).fill_(index) for index in range(count)]
    full = private_mib()
    held.clear()
    time.sleep(0.2)
    own = private_mib()
    held = [torch.empty(size * MIB, dtype=torch.uint8).fill_(index) for index in range(count)]
    thread = threading.Thread(target=held.clear)
    thread.start()
    thread.join()
    primer = torch.empty(size * MIB, dtype=torch.uint8)
    del primer
    time.sleep(0.2)
    other = private_mib()
    print(f"      control (not a pass/fail): {count} x {size} MiB made by this thread: {full - before:+.0f} MiB; let "
          f"go of by this thread: {own - before:+.0f} MiB left; made again and let go of by ANOTHER thread: "
          f"{other - before:+.0f} MiB left (MIMALLOC_PURGE_DELAY={os.environ.get('MIMALLOC_PURGE_DELAY')}). When "
          f"the last number is not far above the one before it, the private-bytes line above could not have seen a "
          f"wrong thread in this run; the line about who let go of the tensors always can.", flush=True)


def check_three_demands(ctx: Context) -> None:
    """Two slots, three demands at once (two want and one take of a third item): nobody waits for anybody."""
    cases = {"all three on disk": 0, "all three in RAM": 1024}
    for label, ram in cases.items():
        bench = Bench(ctx, f"three{ram}", ram_mib=ram, slots=2, slot_mib=64)
        store = bench.store
        store.begin({})
        tensors = {}
        for index, prop in enumerate("abc"):
            store.open_prop(prop)
            tensors[prop] = big(40 + index, 24)
            store.put_device(prop, "cond", on_device(ctx, tensors[prop]))
        store.flush()
        began = time.perf_counter()
        store.want("a", "cond")
        store.want("b", "cond")
        got_c = store.take("c", "cond")                     # both slots are taken or being filled
        waited = time.perf_counter() - began
        states = bench.slot_states()
        got_a = store.take("a", "cond")
        got_b = store.take("b", "cond")
        total = time.perf_counter() - began
        problems = [f"{prop}: {problem}" for prop, got in (("a", got_a), ("b", got_b), ("c", got_c))
                    if (problem := differs(got, tensors[prop], ctx.device))]
        ctx.record(f"{label}: want a, want b, take c", not problems and total < 10.0,
                   "; ".join(problems) if problems else
                   f"take c returned after {waited * 1e3:.0f} ms with the slots {states}; all three after "
                   f"{total * 1e3:.0f} ms, bit-equal (limit 10 s)")
        del got_a, got_b, got_c

        # the same with nobody pumping in between and both slots left staged: the third goes around them
        store.want("a", "cond")
        store.want("b", "cond")
        both = staged(store, 2)
        began = time.perf_counter()
        got_c = store.take("c", "cond")
        waited = time.perf_counter() - began
        problem = differs(got_c, tensors["c"], ctx.device)
        ctx.record(f"{label}: both slots staged for others, take c", not problem and waited < 10.0
                   and both == ["staged", "staged"],
                   problem or f"slots {both}; take c in {waited * 1e3:.0f} ms, bit-equal")

        # three wants on two slots: the third gets the first slot that comes free
        store.want("c", "cond")
        got = {prop: store.take(prop, "cond") for prop in "cab"}
        problems = [f"{prop}: {problem}" for prop in "abc"
                    if (problem := differs(got[prop], tensors[prop], ctx.device))]
        ctx.record(f"{label}: three wants, then take c, a, b", not problems, "; ".join(problems) or "bit-equal")
        del got, got_c
        settle(store)
        for prop in "abc":
            store.close_prop(prop)
        store.flush()
        ctx.record(f"{label}: nothing left, no warning", not bench.files() and not bench.events
                   and set(bench.slot_states()) == {"free"},
                   f"{bench.files()} {bench.events[:1]} {bench.slot_states()}")
        bench.close()


def check_stress(ctx: Context) -> None:
    """200 cycles of want / pump / take / discard / close_prop / cancel in every order, with the bytes of every
    taken item checked: a slot refilled while its copies were still running would hand out another item's bytes."""
    cycles = 200
    rng = random.Random(5)
    slot_mib = 64
    bench = Bench(ctx, "stress", ram_mib=256 if ctx.cuda else 96, slots=2, slot_mib=slot_mib)    # on the card the
    store = bench.store                                     # slots are the subject, on the CPU the disk as well
    module = ctx.module
    store.begin({})
    base = torch.randint(0, 256, (96 * MIB,), dtype=torch.uint8, generator=torch.Generator().manual_seed(7))

    def make(seed: int, mib: int) -> dict:
        """{"a": bf16 [rows,1024], "b": i32 [rows,4]} of about `mib` MiB, a function of the seed alone."""
        rows = max(1, mib * MIB // (2048 + 16))
        offset = (seed * 7919) % (4 * MIB) * 4
        a = (base[offset:offset + rows * 2048] ^ (seed % 251)).view(torch.bfloat16).view(rows, 1024)
        b = (base[offset // 4:offset // 4 + rows * 16] ^ (seed % 13)).view(torch.int32).view(rows, 4)
        return {"a": a, "b": b}

    counts = {"waited for a copy in flight": 0, "refills": 0, "uploads in flight when checked": 0}
    synchronise, lend = store._synchronise, store._lend

    def counted_synchronise(slot):
        if slot.state == module.UPLOADING and not slot.event.query():
            counts["waited for a copy in flight"] += 1
        return synchronise(slot)

    def counted_lend(slot, item):
        counts["refills"] += 1
        if slot.event is not None or slot.state != module.FREE:
            raise AssertionError("a slot was lent before its event had completed")
        return lend(slot, item)

    store._synchronise, store._lend = counted_synchronise, counted_lend
    freed = Freed(module).__enter__()

    live: dict[tuple[str, str], tuple[int, int]] = {}       # (prop, key) -> (seed, MiB)
    props: list[str] = []
    problems: list[str] = []
    taken = 0
    seed = 0

    def new_prop() -> str:
        nonlocal seed
        prop = f"p{len(props)}-{seed}"
        store.open_prop(prop)
        props.append(prop)
        for key in ("x", "y"):
            seed += 1
            mib = rng.choice((2, 9, 24, 40, 60)) if ctx.cuda else rng.choice((1, 3, 9, 24, 60))
            if rng.random() < 0.08:
                mib = slot_mib + 6                          # larger than a slot: never staged
            live[(prop, key)] = (seed, mib)
            store.put_device(prop, key, on_device(ctx, make(seed, mib)))
        return prop

    def verify(prop: str, key: str) -> None:
        nonlocal taken
        got = store.take(prop, key)
        if store._slots and any(slot.state == module.UPLOADING and not slot.event.query() for slot in store._slots):
            counts["uploads in flight when checked"] += 1
        problem = differs(got, make(*live[(prop, key)]), ctx.device)
        taken += 1
        if problem:
            problems.append(f"cycle {cycle}, {prop}/{key} ({live[(prop, key)][1]} MiB): {problem}")

    def close(prop: str) -> None:
        store.close_prop(prop)
        props.remove(prop)
        for key in ("x", "y"):
            live.pop((prop, key), None)

    def some(but: str = "") -> tuple[str, str] | None:
        """An item of an older prop that is still there."""
        others = [entry for entry in live if entry[0] != but]
        return rng.choice(others) if others else None

    began = time.perf_counter()
    for cycle in range(cycles):
        prop = new_prop()
        older = some(prop)
        store.want(prop, "x")
        if rng.random() < 0.5:
            store.flush()                                   # staged; else the helper may still be filling
        store.pump(upload=rng.random() < 0.85)              # issues the copies: they are in flight from here on
        action = cycle % 8
        if action == 0:
            verify(prop, "x")                               # taken while its copies run
            if older:
                store.want(*older)                          # the slot is asked for again at once
                store.pump()
                verify(*older)
        elif action == 1:
            close(prop)                                     # closed while its copies run
            if older:
                store.want(*older)
                store.pump()
                verify(*older)
        elif action == 2:
            store.discard(prop, "x")                        # discarded while its copies run
            live.pop((prop, "x"))
            store.want(prop, "y")
            verify(prop, "y")
        elif action == 3:
            store.want(prop, "y")
            if older:
                store.want(*older)                          # three wants on two slots
            store.pump()
            for entry in ((prop, "y"), (prop, "x"), older):
                if entry:
                    verify(*entry)
        elif action == 4:
            for name in list(props):                        # cancel: everything goes while copies run
                close(name)
            store.flush()
        elif action == 5:
            if older:
                verify(*older)
            verify(prop, "x")
            verify(prop, "x")                               # a second take of the same item
        elif action == 6:
            store.pump(upload=False)                        # short of memory: what was uploaded ahead goes back
            store.pump(upload=False)
            verify(prop, "x")
            if older:
                store.want(*older)
                store.discard(*older)                       # discarded while (or before) it is being staged
                live.pop(older)
        else:
            store.put_device(prop, "x", on_device(ctx, make(seed + 1000, 9)))      # replaced while wanted
            live[(prop, "x")] = (seed + 1000, 9)
            verify(prop, "x")
            store.flush()
        while len(props) > 5:
            close(props[0])
        if problems:
            break
    for name in list(props):
        close(name)
    settle(store)
    seconds = time.perf_counter() - began
    store._synchronise, store._lend = synchronise, lend
    stats = store.stats()
    ctx.record(f"{cycles} cycles: every taken item has its own bytes", not problems and cycle == cycles - 1,
               problems[0] if problems else
               f"{taken} takes bit-equal in {seconds:.1f} s; {counts['refills']} slot fills, {stats['uploads_ahead']} "
               f"uploads ahead, {stats['uploads_at_take']} at take, {stats['spills']} spills; a copy was still in "
               f"flight {counts['uploads in flight when checked']} times when its bytes were checked and "
               f"{counts['waited for a copy in flight']} times when its slot or item was wanted gone")
    ctx.record("nothing is left: RAM, card, disk, queue, slots, files",
               not (stats["ram_bytes"] or stats["device_bytes"] or stats["disk_bytes"] or stats["queued"]
                    or stats["small_bytes"] or bench.files()) and set(bench.slot_states()) <= {"free"},
               f"{stats}, slots {bench.slot_states()}, files {bench.files()[:3]}")
    ctx.record("no warning", not bench.events, f"{len(bench.events)}: {bench.events[:2]}")
    if ctx.cuda:
        ctx.record("copies were in flight when slots and items were given up (the case is exercised)",
                   counts["waited for a copy in flight"] + counts["uploads in flight when checked"] > 0, f"{counts}")
    bench.close()
    freed.__exit__()
    ctx.record("every tensor of the RAM cache was let go of by the calling thread, none by the helper thread",
               *freed.verdict())


def check_discard_during_read(ctx: Context) -> None:
    """discard / close_prop while the helper thread reads that very item into a slot, and while the read still
    waits in the queue: returns at once, raises nothing, and the slot is not given on before the read is over."""
    module = ctx.module
    bench = Bench(ctx, "discard", ram_mib=0, slots=1, slot_mib=64)
    store = bench.store
    store.begin({})
    tensors = {name: big(60 + index, 16) for index, name in enumerate("abcd")}
    for name in "abcd":
        store.open_prop(name)
        store.put_device(name, "cond", tensors[name])
    store.flush()

    real = module.stfile.read_into
    started, release = threading.Event(), threading.Event()

    def slow_read(path, names, buffer):
        started.set()
        release.wait(10.0)
        return real(path, names, buffer)

    module.stfile.read_into = slow_read
    try:
        store.want("a", "cond")
        ok = started.wait(5.0)
        began = time.perf_counter()
        store.discard("a", "cond")                          # the helper is inside the read of this item
        returned = time.perf_counter() - began
        store.want("b", "cond")                             # the one slot is still being written to
        store.pump()
        during = bench.slot_states()
        waiting = [item.prop.id for item in store._wants]
        release.set()
        store.flush()
        after = bench.slot_states()
        got = store.take("b", "cond")
        problem = differs(got, tensors["b"])
        ctx.record("discard while the item is being read", ok and returned < 0.05 and not problem
                   and during == ["filling"] and waiting == ["b"] and after == ["staged"],
                   problem or f"returned in {returned * 1e3:.2f} ms; the slot stayed {during} with b waiting, then "
                              f"{after} for b; b bit-equal; file of a gone: "
                              f"{not (bench.folder / 'a' / 'cond.safetensors').exists()}")
        ctx.record("the discarded item's file is deleted after the read", "a/cond.safetensors" not in bench.files(),
                   f"{bench.files()}")

        started.clear()
        release.clear()
        store.want("c", "cond")                             # the helper is held inside this read ...
        started.wait(5.0)
        store.want("d", "cond")                             # ... no slot: d waits
        began = time.perf_counter()
        store.close_prop("c")                               # closed while it is being read
        store.close_prop("d")                               # closed while it waits for a slot
        returned = time.perf_counter() - began
        release.set()
        store.flush()
        ctx.record("close_prop while the item is being read / waits for a slot",
                   returned < 0.05 and bench.slot_states() == ["free"] and not store._wants
                   and not (bench.folder / "c").exists() and not (bench.folder / "d").exists(),
                   f"returned in {returned * 1e3:.2f} ms; slot {bench.slot_states()}; folders left: "
                   f"{sorted(path.name for path in bench.folder.iterdir())}")
    finally:
        release.set()
        module.stfile.read_into = real
    try:
        store.take("a", "cond")
        gone = False
    except KeyError:
        gone = True
    ctx.record("a discarded item cannot be taken", gone)
    store.close_prop("a")
    store.close_prop("b")
    store.flush()
    ctx.record("nothing left, no warning", not bench.files() and not bench.events,
               f"{bench.files()} {bench.events[:2]}")
    bench.close()


def check_held_file(ctx: Context) -> None:
    """A spilled file another program holds open: the delete is tried again three times over two seconds and
    then reported as a warning; nothing is raised into the GPU thread."""
    if os.name != "nt":
        ctx.record("skipped (only Windows refuses to delete an open file)", True)
        return
    bench = Bench(ctx, "held", ram_mib=0, slots=2, slot_mib=64)
    store = bench.store
    store.begin({})
    for prop in ("a", "b"):
        store.open_prop(prop)
        store.put_device(prop, "cond", big(70, 8))
    store.flush()
    path = bench.folder / "a" / "cond.safetensors"
    handle = open(path, "rb")
    try:
        began = time.perf_counter()
        store.discard("a", "cond")
        returned = time.perf_counter() - began
        store.flush()
        waited = time.perf_counter() - began
    finally:
        handle.close()
    warnings = bench.warnings("delete")
    ctx.record("held for good: tried again, then reported", returned < 0.05 and 1.9 <= waited < 4.0
               and len(warnings) == 1 and "4 tries" in warnings[0]["message"] and warnings[0]["prop_id"] == "a"
               and path.exists(),
               f"discard returned in {returned * 1e3:.2f} ms, the helper gave up after {waited:.2f} s; warning: "
               f"{warnings[0]['message'] if warnings else None}")

    path = bench.folder / "b" / "cond.safetensors"
    handle = open(path, "rb")
    timer = threading.Timer(0.7, handle.close)
    timer.start()
    began = time.perf_counter()
    store.discard("b", "cond")
    store.flush()
    waited = time.perf_counter() - began
    timer.join()
    ctx.record("let go after 0.7 s: deleted by a later try, no warning",
               not path.exists() and len(bench.warnings("delete")) == 1 and 0.6 < waited < 2.5,
               f"deleted after {waited:.2f} s")
    store.close_prop("a")                                   # the file of a is free now: the folder goes
    store.close_prop("b")
    store.flush()
    ctx.record("close_prop removes what the failed delete left",
               not bench.files() and not (bench.folder / "a").exists(),
               f"{bench.files()}")

    store.open_prop("c")                                    # a folder held: close_prop reports, does not raise
    store.put_device("c", "cond", big(71, 8))
    store.flush()
    handle = open(bench.folder / "c" / "cond.safetensors", "rb")
    try:
        store.close_prop("c")
        store.flush()
    finally:
        handle.close()
    ctx.record("close_prop with a held file: reported", len(bench.warnings("delete")) == 2,
               f"{bench.warnings('delete')[-1]['message'] if bench.warnings('delete') else None}")
    bench.close()


def check_full_disk(ctx: Context) -> None:
    """A full drive (simulated): the item stays in RAM and a warning says so; with RAM short as well its prop
    fails with kind "disk"; a failed write is the same; a job that fits in RAM never asks the drive."""
    module = ctx.module
    real_free, real_available, real_write = module._free_bytes, module._ram_available, module.stfile.write
    asked: list[Path] = []

    def little(folder):
        asked.append(folder)
        return 500 * MIB

    try:
        module._free_bytes = little
        module._ram_available = lambda: 8 << 30
        bench = Bench(ctx, "full", ram_mib=0, slots=2, slot_mib=64)
        store = bench.store
        store.begin({})
        tensors = big(80, 12)
        store.open_prop("a")
        store.put_device("a", "cond", tensors)
        store.flush()
        stats = store.stats()
        warnings = bench.warnings("spill")
        got = store.take("a", "cond")
        ctx.record("drive under 1 GiB free: kept in RAM, warned, still usable",
                   stats["spills"] == 0 and stats["ram_bytes"] >= 12 * MIB and len(warnings) == 1
                   and "500 MiB are free" in warnings[0]["message"] and not differs(got, tensors)
                   and not bench.files(),
                   f"cache {stats['ram_bytes'] // MIB} MiB, files {bench.files()}, warning: "
                   f"{warnings[0]['message'] if warnings else None}")
        store.pump()
        store.flush()
        ctx.record("the item is not tried again and again", len(bench.warnings("spill")) == 1 and len(asked) == 1,
                   f"{len(bench.warnings('spill'))} warnings, the drive was asked {len(asked)} times")

        module._ram_available = lambda: 1 << 30             # now RAM is short too
        store.open_prop("b")
        store.open_prop("c")
        store.put_small("c", "request", {"seed": 1})
        store.put_device("b", "cond", big(81, 12))
        store.flush()
        failure = None
        try:
            store.take("b", "cond")
        except Exception as error:
            failure = error
        ctx.record('RAM short as well: the prop fails with kind "disk"',
                   type(failure).__name__ == "PropFailed" and getattr(failure, "kind", None) == "disk"
                   and str(failure) == module.DISK_MESSAGE and store.get_small("c", "request") == {"seed": 1}
                   and not differs(store.take("a", "cond"), tensors),
                   f"{type(failure).__name__}({failure}), kind {getattr(failure, 'kind', None)}; cache now "
                   f"{store.stats()['ram_bytes'] // MIB} MiB; the other props go on")
        store.close_prop("b")

        module._free_bytes = real_free
        module._ram_available = lambda: 8 << 30

        def no_space(path, tensors, *, fsync=False):
            raise OSError(28, "No space left on device")

        module.stfile.write = no_space
        store.put_device("c", "cond", tensors)
        store.flush()
        module.stfile.write = real_write
        warnings = bench.warnings("spill")
        ctx.record("a write that fails: kept in RAM, warned, still usable",
                   len(warnings) == 3 and "No space left" in warnings[-1]["message"]
                   and not differs(store.take("c", "cond"), tensors) and not bench.files(),
                   f"warning: {warnings[-1]['message'] if warnings else None}")
        for prop in "ac":
            store.close_prop(prop)
        store.flush()
        bench.close()

        module._free_bytes = lambda folder: 0               # a nearly full drive, a job that fits in RAM
        bench = Bench(ctx, "full-ram", ram_mib=256, slots=2, slot_mib=64)
        store = bench.store
        store.begin({})
        store.open_prop("a")
        store.put_device("a", "cond", tensors)
        store.want("a", "cond")
        store.flush()
        store.pump()
        ctx.record("a job that fits in RAM runs on a full drive", not differs(store.take("a", "cond"), tensors)
                   and not bench.events and not bench.files(), f"{bench.events[:1]} {bench.files()}")
        store.close_prop("a")
        bench.close()
    finally:
        module._free_bytes, module._ram_available, module.stfile.write = real_free, real_available, real_write


def check_checkpoint(ctx: Context) -> None:
    """checkpoint + begin: a valid checkpoint comes back; a wrong hash, a truncated or altered file and a
    leftover .tmp are deleted, as are the folders of props not named; a failed write only warns."""
    module = ctx.module

    def latents(seed: int, rows: int) -> dict:
        return {"coords": noise(seed, torch.int32, (rows, 4)), "res": torch.tensor([1024], dtype=torch.int32),
                "shape_slat": noise(seed + 1, torch.float32, (rows, 32)),
                "tex_slat": noise(seed + 2, torch.float32, (rows, 32))}

    saved = {name: latents(100 + 10 * index, 13000 + index) for index, name in enumerate(
        ("good", "wronghash", "truncated", "altered", "norecord", "emptyhash"))}
    notes = {"ss_retries": 2, "ss_blocked": False, "slow": ["hr"], "views": {"azimuths": [0.0, 90.0]}}

    first = Bench(ctx, "checkpoint", ram_mib=0, slots=2, slot_mib=64)
    store = first.store
    store.begin({})
    for name, tensors in saved.items():
        store.open_prop(name)
        store.put_small(name, "request", {"seed": 1})
        store.put_device(name, "cond_tex", big(5, 4))       # a spilled file beside the checkpoint
        store.checkpoint(name, "" if name == "emptyhash" else f"hash-{name}", on_device(ctx, tensors), notes)
    notes["slow"].append("changed after the call")           # the notes are the ones of the moment of the call
    store.open_prop("failing")

    def refuse(path, tensors, *, fsync=False):
        raise OSError(13, "Permission denied")

    real_write = module.stfile.write
    store.flush()
    module.stfile.write = refuse
    try:
        store.checkpoint("failing", "hash-failing", saved["good"], notes)
        store.flush()
    finally:
        module.stfile.write = real_write
    store.put_small("failing", "request", {"seed": 2})
    ctx.record("a checkpoint write that fails only warns: the prop goes on",
               len(first.warnings("checkpoint")) == 1
               and "Permission denied" in first.warnings("checkpoint")[0]["message"]
               and store.get_small("failing", "request") == {"seed": 2}
               and not (first.folder / "failing" / "final.safetensors").exists(),
               f"{first.warnings('checkpoint')[0]['message'] if first.warnings('checkpoint') else None}")
    store.checkpoint("failing", "hash-failing", saved["good"], {"hook": object()})
    store.checkpoint("failing", "hash-failing", {**saved["good"], "res": 1024}, notes)
    store.flush()
    ctx.record("notes that are not JSON, a latent that is not a tensor: a warning each, nothing raised or written",
               len(first.warnings("checkpoint")) == 3 and not (first.folder / "failing" / "final.safetensors").exists(),
               f"{[event['message'][-60:] for event in first.warnings('checkpoint')[1:]]}")

    record = json.loads((first.folder / "good" / "final.json").read_text(encoding="utf-8"))
    data = first.folder / "good" / "final.safetensors"
    import hashlib
    ctx.record("final.json: request_hash, notes, bytes, sha256 of final.safetensors",
               sorted(record) == ["bytes", "notes", "request_hash", "sha256"] and record["request_hash"] == "hash-good"
               and record["bytes"] == data.stat().st_size
               and record["sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
               and record["notes"]["slow"] == ["hr"],
               f"{ {key: (value if key != 'sha256' else value[:12] + '...') for key, value in record.items()} }")
    ctx.record("no checkpoint for a prop without a request hash",
               not (first.folder / "emptyhash" / "final.safetensors").exists())
    ctx.record("stats count the checkpoints on disk", store.stats()["disk_bytes"] >= 5 * record["bytes"],
               f"{store.stats()['disk_bytes']} bytes")
    first.close()                                           # the process ends here: nothing was closed properly

    whole = (first.folder / "truncated" / "final.safetensors").read_bytes()
    (first.folder / "truncated" / "final.safetensors").write_bytes(whole[:len(whole) // 2])
    changed = bytearray((first.folder / "altered" / "final.safetensors").read_bytes())
    changed[len(changed) // 2] ^= 0x01
    (first.folder / "altered" / "final.safetensors").write_bytes(bytes(changed))
    (first.folder / "norecord" / "final.json").unlink()
    (first.folder / "good" / "cond_hr.safetensors.tmp").write_bytes(b"half a file")
    (first.folder / "good" / "final.json.tmp").write_bytes(b"{")
    (first.folder / "stray.tmp").write_bytes(b"x")
    (first.folder / "foreign").mkdir()
    (first.folder / "foreign" / "cond_lr.safetensors").write_bytes(b"an older batch's")
    (first.folder / "notours").mkdir()
    (first.folder / "notours" / "picture.png").write_bytes(b"somebody's file")
    before = first.files()

    second = Bench(ctx, "checkpoint", ram_mib=64, slots=2, slot_mib=64)
    store = second.store
    resumed = store.begin({"good": "hash-good", "wronghash": "another", "truncated": "hash-truncated",
                           "altered": "hash-altered", "norecord": "hash-norecord", "emptyhash": "",
                           "failing": "hash-failing", "new": "hash-new"})
    problem = None
    if list(resumed) != ["good"]:
        problem = f"resumed {list(resumed)}, expected only 'good'"
    else:
        entry = resumed["good"]
        problem = differs({name: entry["tensors"][name] for name in saved["good"]}, saved["good"], "cpu")
        if not problem and (entry["notes"] != {"ss_retries": 2, "ss_blocked": False, "slow": ["hr"],
                                               "views": {"azimuths": [0.0, 90.0]}}
                            or entry["request_hash"] != "hash-good"):
            problem = f"notes / hash differ: {entry['notes']} {entry['request_hash']}"
    ctx.record("begin returns the valid checkpoint: tensors bit-equal, notes as written", not problem,
               problem or f"{ {name: list(t.shape) for name, t in resumed['good']['tensors'].items()} }")
    after = second.files()
    ctx.record("wrong hash, truncated, altered, no record, .tmp files and other props' folders are deleted",
               after == ["good/final.json", "good/final.safetensors", "notours/picture.png"]
               and sorted(path.name for path in second.folder.iterdir()) == ["good", "notours"],
               f"before: {len(before)} files in {len({name.split('/')[0] for name in before})} folders; after: {after}")
    foreign = [event for event in second.events if "not the store's" in event["message"]]
    ctx.record("a folder with files that are not the store's is left alone and reported",
               len(foreign) == 1 and len(second.events) == 1, f"{second.events}")

    store.open_prop("good")                                 # resumed: only its last part runs
    store.put_small("good", "coords_hr", {"coords": resumed["good"]["tensors"]["coords"]})
    store.flush()
    ctx.record("open_prop keeps the checkpoint of a resumed prop", second.files()[:2] == ["good/final.json",
                                                                                         "good/final.safetensors"])
    store.close_prop("good")
    store.flush()
    ctx.record("close_prop takes the checkpoint with it", not (second.folder / "good").exists(), f"{second.files()}")
    second.close()

    third = Bench(ctx, "checkpoint2", ram_mib=64)           # a checkpoint kept by begin but never opened: cancel
    store = third.store
    store.begin({})
    store.open_prop("p")
    store.checkpoint("p", "h", saved["good"], {})
    store.checkpoint("p", "h", saved["wronghash"], {"second": True})      # a second one replaces the first
    store.flush()
    third.close()
    fourth = Bench(ctx, "checkpoint2", ram_mib=64)
    store = fourth.store
    resumed = store.begin({"p": "h"})
    ok = list(resumed) == ["p"] and resumed["p"]["notes"] == {"second": True} and not differs(
        {name: resumed["p"]["tensors"][name] for name in saved["wronghash"]}, saved["wronghash"])
    store.close_prop("p")                                   # never opened in this batch
    store.flush()
    ctx.record("the later of two checkpoints is the one found; close_prop of a kept, unopened prop removes it",
               ok and not fourth.files() and store.stats()["disk_bytes"] == 0, f"{fourth.files()} {store.stats()}")
    again = store.begin({"p": "h"})
    ctx.record("begin on an emptied folder resumes nothing", again == {} and not fourth.events, f"{fourth.events[:1]}")
    fourth.close()


def check_pinned(ctx: Context) -> None:
    """The slots are page-locked, made at the first want, and stats()["pinned_bytes"] is slots x slot_mib."""
    for slots, slot_mib, device in ((2, 128, "cuda"), (2, 256, "cuda:0")):
        bench = Bench(ctx, f"pinned{slot_mib}", device=device, ram_mib=64, slots=slots, slot_mib=slot_mib)
        store = bench.store
        store.begin({})
        store.open_prop("a")
        tensors = big(1, 4)
        store.put_device("a", "cond", on_device(ctx, tensors))
        before = store.stats()["pinned_bytes"]
        store.want("a", "cond")
        after = store.stats()["pinned_bytes"]
        locked = [slot.buf.is_pinned() for slot in store._slots]
        sizes = [slot.buf.numel() for slot in store._slots]
        store.flush()
        views = all(view.is_pinned() for slot in store._slots if slot.views for view in slot.views.values())
        direct = [slot.direct for slot in store._slots if slot.views]
        store.pump()
        problem = differs(store.take("a", "cond"), tensors, "cuda")
        ctx.record(f"{slots} slots of {slot_mib} MiB (device given as {device!r})",
                   before == 0 and after == slots * slot_mib * MIB and all(locked)
                   and sizes == [slot_mib * MIB] * slots and views and direct == [True] and not problem
                   and store.stats()["uploads_ahead"] == 1,
                   problem or f"pinned_bytes {before} before the first want, {after // MIB} MiB after; is_pinned "
                              f"{locked}; the staged views are seen as page-locked: {views}; uploaded ahead and "
                              f"bit-equal")
        settle(store)
        store.close_prop("a")
        bench.close()
    spare = {key: len(value) for key, value in ctx.module._spare.items()}
    bench = Bench(ctx, "pinned-again", ram_mib=64, slots=2, slot_mib=256)
    bench.store.begin({})
    bench.store.open_prop("a")
    bench.store.put_device("a", "cond", on_device(ctx, big(1, 4)))
    bench.store.want("a", "cond")
    left = {key: len(value) for key, value in ctx.module._spare.items()}
    ctx.record("a later store of the process takes over the buffers of a closed one (nothing new is locked)",
               spare.get((256 * MIB, True)) == 2 and left.get((256 * MIB, True)) == 0, f"spare {spare} -> {left}")
    bench.close()


def check_pump_timing(ctx: Context) -> None:
    """pump() under 1 ms with nothing to do, and under 1 ms while it issues the upload of a 200 MiB item: the
    copy is only queued, which is the proof that torch sees the slot's views as page-locked memory."""
    bench = Bench(ctx, "pump", ram_mib=1024, slots=2, slot_mib=256)
    store = bench.store
    store.begin({})
    tensors = big(3, 200)
    nbytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    store.open_prop("warm")
    store.put_device("warm", "cond", on_device(ctx, big(4, 8)))
    store.want("warm", "cond")
    store.flush()
    store.pump()
    store.take("warm", "cond")
    settle(store)
    store.close_prop("warm")

    idle = []
    for _ in range(2000):
        began = time.perf_counter()
        store.pump()
        idle.append(time.perf_counter() - began)
    idle.sort()
    ctx.record("nothing to do", idle[-1] < 1e-3,
               f"2000 calls: median {idle[1000] * 1e6:.1f} us, 99% {idle[1980] * 1e6:.1f} us, longest "
               f"{idle[-1] * 1e6:.1f} us (limit 1000)")

    issue, copies, problems, fresh = [], [], [], []
    for round_ in range(8):
        prop = f"p{round_}"
        store.open_prop(prop)
        store.put_device(prop, "cond", on_device(ctx, tensors))
        store.want(prop, "cond")
        store.flush()
        torch.cuda.synchronize()
        if round_ >= 5:
            torch.cuda.empty_cache()                        # PyTorch has no block of this size left: it asks the
        ahead = store.stats()["uploads_ahead"]              # driver for one inside the call
        began = time.perf_counter()
        store.pump()
        (issue if round_ < 5 else fresh).append(time.perf_counter() - began)
        in_flight = any(slot.state == "uploading" and not slot.event.query() for slot in store._slots)
        if store.stats()["uploads_ahead"] != ahead + 1:
            problems.append(f"round {round_}: pump did not issue the upload")
        torch.cuda.synchronize()
        copies.append(time.perf_counter() - began)
        if not in_flight:
            problems.append(f"round {round_}: the copy was already over when pump returned")
        began = time.perf_counter()
        got = store.take(prop, "cond")
        took = time.perf_counter() - began
        problem = differs(got, tensors, "cuda")
        if problem:
            problems.append(f"round {round_}: {problem}")
        if took > 1e-3:
            problems.append(f"round {round_}: take of an uploaded item took {took * 1e3:.2f} ms")
        del got
        store.close_prop(prop)
        store.flush()
    ctx.record(f"while it issues the upload of {nbytes / MIB:.0f} MiB", not problems and max(issue) < 1e-3,
               "; ".join(problems) if problems else
               f"5 uploads: pump took {', '.join(f'{value * 1e6:.0f}' for value in issue)} us (limit 1000); the "
               f"copies themselves ran {', '.join(f'{value * 1e3:.0f}' for value in copies[:5])} ms, still in "
               f"flight when pump returned; bit-equal on the card")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    began = time.perf_counter()
    block = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    driver = time.perf_counter() - began
    del block
    ctx.record("the same when PyTorch must first get the card's memory from the driver", max(fresh) < 5e-3,
               f"3 uploads after empty_cache(): pump took {', '.join(f'{value * 1e6:.0f}' for value in fresh)} us "
               f"(limit 5000; the driver's allocation of {nbytes / MIB:.0f} MiB alone: {driver * 1e6:.0f} us)")

    pageable = torch.empty(nbytes, dtype=torch.uint8).fill_(1)
    torch.cuda.synchronize()
    began = time.perf_counter()
    on_card = pageable.to("cuda", non_blocking=True)
    waited = time.perf_counter() - began
    torch.cuda.synchronize()
    del on_card, pageable
    print(f"      control (not a pass/fail): the same {nbytes / MIB:.0f} MiB from memory that is NOT page-locked, "
          f"non_blocking=True: the call returned after {waited * 1e3:.1f} ms", flush=True)
    ctx.record("no warning", not bench.events, f"{bench.events[:2]}")
    bench.close()
    torch.cuda.empty_cache()


def check_cap_rule(ctx: Context) -> None:
    """An upload ahead is issued only while reserved + item + 256 MiB stays under PyTorch's cap; under a tighter
    cap the item waits in its slot for take. pump(upload=False) uploads nothing and gives back what was uploaded."""
    bench = Bench(ctx, "cap", ram_mib=1024, slots=2, slot_mib=128)
    store = bench.store
    store.begin({})
    tensors = big(9, 100)
    nbytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    store.open_prop("a")
    store.put_device("a", "cond", on_device(ctx, tensors))
    gc.collect()
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved()
    before = torch.cuda.get_per_process_memory_fraction()
    try:
        tight = reserved + nbytes + 200 * MIB               # room for the item, not for the 256 MiB of headroom
        torch.cuda.set_per_process_memory_fraction(tight / total)
        store.want("a", "cond")
        store.flush()
        for _ in range(3):
            store.pump()
        waited = store.stats()["uploads_ahead"] == 0 and bench.slot_states().count("staged") == 1
        roomy = reserved + nbytes + 300 * MIB
        torch.cuda.set_per_process_memory_fraction(roomy / total)
        store.pump()
        uploaded = store.stats()["uploads_ahead"] == 1 and store.stats()["device_bytes"] == nbytes
        ctx.record("the cap decides", waited and uploaded,
                   f"reserved {reserved // MIB} MiB + item {nbytes // MIB} MiB: under a cap of {tight // MIB} MiB "
                   f"the item stayed staged ({waited}); under {roomy // MIB} MiB pump uploaded it ({uploaded})")
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        store.pump(upload=False)
        back = store.stats()["device_bytes"] == 0 and torch.cuda.memory_allocated() <= allocated - nbytes + MIB
        store.flush()
        store.pump(upload=False)
        held = store.stats()["uploads_ahead"] == 1 and bench.slot_states().count("staged") == 1
        got = store.take("a", "cond")
        problem = differs(got, tensors, "cuda")
        ctx.record("pump(upload=False) gives the card back what was uploaded ahead and uploads nothing",
                   back and held and not problem,
                   problem or f"the {nbytes // MIB} MiB left the card at once ({back}), the item was staged again "
                              f"and not uploaded ({held}), then taken bit-equal")
        del got
    finally:
        torch.cuda.set_per_process_memory_fraction(before)
    settle(store)
    store.close_prop("a")
    store.flush()
    ctx.record("no warning", not bench.events, f"{bench.events[:2]}")
    bench.close()
    torch.cuda.empty_cache()


def check_overwrite_control(ctx: Context) -> None:
    """Not a check of the store: shows that the stress check's comparison would see a slot refilled too early.
    A page-locked buffer is overwritten right after its non-blocking upload was issued."""
    spare = ctx.module._spare.get((256 * MIB, True)) or []
    if not spare:
        ctx.record("skipped: no spare page-locked buffer (run after `pinned`)", True)
        return
    buffer = spare[0]
    size = 200 * MIB
    seen = []
    for _ in range(3):
        buffer[:size].fill_(1)
        torch.cuda.synchronize()
        on_card = buffer[:size].to("cuda", non_blocking=True)
        buffer[:size].fill_(2)                              # what a refill before the event would do
        torch.cuda.synchronize()
        seen.append(int((on_card != 1).sum()))
        del on_card
    print(f"      control (not a pass/fail): bytes that arrived changed when the buffer was overwritten during its "
          f"upload: {seen} of {size}", flush=True)
    ctx.record("an overwrite during the upload reaches the card (so the stress check can see one)", max(seen) > 0,
               f"{seen} changed bytes in 3 tries")
    torch.cuda.empty_cache()


ALLOCATOR_CHILD = r"""
import os, sys, threading
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import psutil, torch
MIB = 1 << 20
process = psutil.Process()
start = process.memory_info()
shown = []
for step in sys.argv[1].split(","):
    kind, size = step[0], int(step[1:]) * MIB
    if kind == "p":                         # made and freed by this thread, never written to
        block = torch.empty(size, dtype=torch.uint8)
        del block
    elif kind == "a":                       # made, written to and freed by this thread
        block = torch.empty(size, dtype=torch.uint8).fill_(1)
        del block
    else:                                   # made and written to by this thread, freed by another
        box = [torch.empty(size, dtype=torch.uint8).fill_(1)]
        thread = threading.Thread(target=box.clear)
        thread.start()
        thread.join()
    shown.append(f"{step}:{(process.memory_info().private - start.private) / MIB:+.0f}")
now = process.memory_info()
print(" ".join(shown), f"| at the end: private {(now.private - start.private) / MIB:+.0f} MiB, "
      f"working set {(now.rss - start.rss) / MIB:+.0f} MiB")
"""


def check_allocator(ctx: Context) -> None:
    """Not a check of the store: what torch's CPU allocator gives back to Windows when large blocks are freed."""
    cases = {
        "ten blocks of 200 MiB, one after the other (a = made, written, freed by the same thread)":
            ",".join(["a200"] * 10),
        "the same with 512 MiB": ",".join(["a512"] * 10),
        "where it starts: 58 MiB and 60 MiB": "a58,a58,a58,a60,a60,a60",
        "eight of 100 MiB, then 200, 512, 54: the eight are counted over all sizes":
            ",".join(["a100"] * 8) + ",a200,a512,a54",
        "eight of 60 MiB never written to (p), then the same": ",".join(["p60"] * 8) + ",a200,a200,a512,a100",
        "past the eight, blocks freed by ANOTHER thread (x), then by the same again":
            ",".join(["a200"] * 9) + ",x54,x54,x54,x54,x200,x200,a200,a54",
    }
    print("      each step shows the process's private bytes after it, in MiB over the start: <kind><MiB>:<+private>",
          flush=True)
    lines = []
    for label, plan in cases.items():
        done = subprocess.run([sys.executable, "-c", ALLOCATOR_CHILD, plan], capture_output=True, text=True,
                              timeout=300)
        line = done.stdout.strip() or done.stderr.strip().splitlines()[-1]
        lines.append(line)
        print(f"      {label}:\n         {line}", flush=True)
    ctx.record("measured (see the lines above)", len(lines) == len(cases),
               f"MIMALLOC_PURGE_DELAY={os.environ.get('MIMALLOC_PURGE_DELAY')}, torch {torch.__version__}")


CHECKS = {              # ram_budget first: in a young process a wrong thread shows in the private bytes (its control)
    "ram_budget": check_ram_budget,
    "small_items": check_small_items,
    "round_trip": check_round_trip,
    "three_demands": check_three_demands,
    "discard_during_read": check_discard_during_read,
    "held_file": check_held_file,
    "full_disk": check_full_disk,
    "checkpoint": check_checkpoint,
    "stress": check_stress,
    "pinned": check_pinned,
    "pump_timing": check_pump_timing,
    "cap_rule": check_cap_rule,
    "overwrite_control": check_overwrite_control,
    "allocator": check_allocator,
}


# =====================================================================================================================
# Stand-ins for a worktree that has no stfile.py / types.py yet (the contract of the specification's 1.2 and 1.5)
# =====================================================================================================================

def stand_in_stfile() -> types.ModuleType:
    import torch
    from typing import NamedTuple

    codes = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I32": torch.int32,
             "I64": torch.int64, "U8": torch.uint8, "BOOL": torch.bool}
    names_of = {dtype: code for code, dtype in codes.items()}

    class TooLarge(ValueError):
        pass

    class TensorInfo(NamedTuple):
        dtype: str
        shape: tuple
        start: int
        end: int

    def read_header(path):
        with open(path, "rb", buffering=0) as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            entries = json.loads(handle.read(length))
        found = sorted(((name, TensorInfo(entry["dtype"], tuple(entry["shape"]), 8 + length + entry["data_offsets"][0],
                                          8 + length + entry["data_offsets"][1]))
                        for name, entry in entries.items() if name != "__metadata__"), key=lambda pair: pair[1].start)
        return dict(found)

    def read_into(path, names, buffer):
        header = read_header(path)
        chosen = [(name, info) for name, info in header.items() if names is None or name in set(names)]
        places, need = [], 0
        for name, info in chosen:
            size = torch.empty((), dtype=codes[info.dtype]).element_size()
            need += -need % size
            places.append((name, info, need))
            need += info.end - info.start
        if need > buffer.numel():
            raise TooLarge(f"{path}: {need} bytes asked for, the buffer holds {buffer.numel()}")
        raw = memoryview(buffer.numpy())
        tensors = {}
        with open(path, "rb", buffering=0) as handle:
            for name, info, at in places:
                handle.seek(info.start)
                got, size = 0, info.end - info.start
                while got < size:
                    count = handle.readinto(raw[at + got:at + size])
                    if not count:
                        raise EOFError(f"{path} ended early")
                    got += count
                tensors[name] = buffer[at:at + size].view(codes[info.dtype]).view(info.shape)
        return tensors

    def write(path, tensors, *, fsync=False):
        entries = sorted(((name, names_of[tensor.dtype], tensor.contiguous()) for name, tensor in tensors.items()),
                         key=lambda entry: (-entry[2].element_size(), entry[0]))
        header, total = {}, 0
        for name, code, tensor in entries:
            size = tensor.numel() * tensor.element_size()
            header[name] = {"dtype": code, "shape": list(tensor.shape), "data_offsets": [total, total + size]}
            total += size
        text = json.dumps(header, separators=(",", ":")).encode("utf-8")
        text += b" " * (-len(text) % 8)
        partial = f"{os.fspath(path)}.tmp"
        try:
            with open(partial, "wb", buffering=0) as handle:
                handle.write(struct.pack("<Q", len(text)) + text)
                for name, code, tensor in entries:
                    if tensor.numel():
                        data = memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
                        sent = 0
                        while sent < len(data):
                            sent += handle.write(data[sent:])
                if fsync:
                    os.fsync(handle.fileno())
            os.replace(partial, path)
        except BaseException:
            try:
                os.unlink(partial)
            except OSError:
                pass
            raise
        return 8 + len(text) + total

    module = types.ModuleType("pixal3d.staged.stfile")
    module.DTYPES = set(codes) | {"C64"}
    module.TooLarge, module.TensorInfo = TooLarge, TensorInfo
    module.read_header, module.read_into, module.write = read_header, read_into, write
    return module


def stand_in_types() -> types.ModuleType:
    class PropFailed(Exception):
        def __init__(self, detail: str, kind: str = "error"):
            super().__init__(detail)
            self.detail = detail
            self.kind = kind

    module = types.ModuleType("pixal3d.staged.types")
    module.PropFailed = PropFailed
    return module


# =====================================================================================================================

def main() -> int:
    global torch, psutil
    args = arguments()
    if args.list:
        for name, check in CHECKS.items():
            where = ("CPU run only" if name in CPU_ONLY else "card only" if name in CARD_ONLY else
                     "when asked for" if name in ASKED_ONLY else "both")
            print(f"{name:<22} {where:<13} {check.__doc__.strip().splitlines()[0]}")
        return 0
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""             # this run must not be able to touch the card
    common.boot(args.repo, golden=args.device == "cuda")    # on the CPU: no golden case, no model file is read
    import psutil
    import torch

    module, stood_in = load_store(args.repo, args.stfile, args.types)
    module.PATIENCE = 30.0                                  # a wait that never ends fails its check, not the run
    if args.wrong_thread:
        free_on_wrong_thread(module)
        print("[check_store] --wrong-thread: the store is BROKEN on purpose; ram_budget's lines about private bytes "
              "and about who let go of the tensors must fail", flush=True)
    for line in stood_in:
        print(f"[check_store] {line}", flush=True)
    print(f"[check_store] store: {module.__file__}; device {args.device}; torch {torch.__version__}; "
          f"ram_defaults() = {module.ram_defaults()} on {psutil.virtual_memory().total / (1 << 30):.1f} GiB",
          flush=True)

    scratch = args.repo / "tmp" / f"check-store-{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    ctx = Context(module, args.device, scratch)
    only = [name for name in args.only.split(",") if name]
    unknown = [name for name in only if name not in CHECKS]
    if unknown:
        print(f"unknown checks {unknown}; --list names them")
        return 2
    chosen = [name for name in CHECKS if (name in only if only else
                                          name not in (CARD_ONLY if args.device == "cpu" else CPU_ONLY) | ASKED_ONLY)]

    def run() -> None:
        for name in chosen:
            if name in CARD_ONLY and args.device == "cpu":
                print(f"[check_store] {name}: needs the card, skipped", flush=True)
                continue
            ctx.current = name
            print(f"\n== {name}: {CHECKS[name].__doc__.strip().splitlines()[0]}", flush=True)
            began = time.perf_counter()
            try:
                with torch.no_grad():
                    CHECKS[name](ctx)
            except Exception:
                ctx.record("ran to its end", False, traceback.format_exc().strip().splitlines()[-1])
                traceback.print_exc()
            print(f"   ({time.perf_counter() - began:.1f} s)", flush=True)

    try:
        if args.device == "cuda":
            with common.gpu_lock("WP5", "check_store " + (args.only or "all"), expected_minutes=3):
                run()
                torch.cuda.synchronize()
                print(f"[check_store] card at the end: {torch.cuda.memory_allocated() // MIB} MiB allocated, "
                      f"{torch.cuda.memory_reserved() // MIB} MiB reserved by PyTorch", flush=True)
        else:
            run()
            print(f"[check_store] CUDA was initialised in this run: {torch.cuda.is_initialized()}", flush=True)
    finally:
        if not args.keep:
            shutil.rmtree(scratch, ignore_errors=True)
    failed = [result for result in ctx.results if not result.ok]
    print("", flush=True)
    for result in failed:
        print(result, flush=True)
    if failed or not ctx.results:
        print(f"FAILED: {len(failed)} of {len(ctx.results)} checks of the store ({args.device}); first: "
              f"{failed[0].name if failed else 'nothing ran'}", flush=True)
        return 1
    print(f"PASSED: all {len(ctx.results)} checks of the store ({args.device}; {'; '.join(stood_in)}).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
