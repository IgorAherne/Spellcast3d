"""What the launchers with probes share: pixal3d_staged_profiled.py (the staged batch path the hub uses) and the
round-A pixal3d_profiled.py (the reference path, code/devtools/pixal3d-lowmem). Both import launchers/
pixal3d_server.py as a module, wrap parts of it from outside and run its main(); what they have in common is here:

    say / Guard          the console line, and the guard every probe runs under: a probe that raises is said once
                         and then does nothing, so a measurement never breaks the server it measures
    mib                  bytes -> MiB with one decimal (None for "unknown")
    cuda_torch()         torch once CUDA runs in this process, else None (never imports torch, never starts CUDA)
    profile_arguments    the --profile-* arguments both take (each adds its own)
    process_env          the part of <name>.env.json both write: the process, the card, the PC
    watch_from_outside   mem_watch.py as a child process watching this one (--profile-external)
    hold_ballast         a smaller card, emulated inside this process (--profile-ballast-leave)

For developers; stdlib only at import, like the probes' own modules (torch is imported where a probe needs it).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from spellcast.profiling import mem_watch, toolkit

MIB = mem_watch.MIB
WDDM_MARGIN = 128                    # MiB of the card Windows never hands out (in the generations its use topped
#                                      out at 12,157 to 12,187 of 12,288; with 256 here the limit came out 130 to
#                                      180 MiB above the one asked for: constrained-and-loading.md, section 4)
HELD: dict[str, Any] = {}            # the ballast once held: "ptr" (None when none was needed), "lib", "mib"

say = toolkit.teller("profiling", clock=False)


class Guard:
    """guard(name, hook, *args, **kwargs): hook(*args, **kwargs), or None after one line in the log the first time
    the hook named `name` fails. `word` is what the log calls a hook ("probe", "hook")."""

    def __init__(self, word: str = "probe") -> None:
        self.word = word
        self.failed: set[str] = set()

    def __call__(self, name: str, hook: Callable[..., Any] | None, *args: Any, **kwargs: Any) -> Any:
        if hook is None:
            return None
        try:
            return hook(*args, **kwargs)
        except Exception as exc:
            if name not in self.failed:
                self.failed.add(name)
                say(f"The {name} {self.word} failed ({type(exc).__name__}: {exc}); Pixal3D goes on, that part of the "
                    f"measurement is missing.")
            return None


def mib(value: Any) -> float | None:
    return None if value is None or value < 0 else round(value / MIB, 1)


def cuda_torch() -> Any:
    """torch once CUDA runs in this process, else None (nothing here starts CUDA or imports torch)."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            return torch
    except Exception:
        pass
    return None


def profile_arguments(default_name: str) -> argparse.ArgumentParser:
    """The --profile-* arguments of both launchers with probes; parse_known_args() leaves pixal3d_server.py's."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--profile-out", required=True, help="the folder the measurements are written into")
    p.add_argument("--profile-name", default=default_name, help="what every file's name starts with")
    p.add_argument("--profile-hz", type=float, default=20.0, help="the sampler's rows per second (~160 KB/min at 20)")
    p.add_argument("--profile-external", action="store_true",
                   help="also sample from a process of its own (mem_watch.py watch; another ~150 KB/min)")
    p.add_argument("--profile-ballast-leave", type=int, default=0,
                   help="emulate a smaller card: the MiB of dedicated memory this process may hold in all (0: off)")
    return p


def process_env(host: mem_watch.HostProbe, watch: Any, when: str = "") -> dict[str, Any]:
    """The process, the card (torch's view of it) and the PC's memory, for <name>.env.json. `when` goes into the
    names of the figures that change ("_at_start": ram_available_at_start_mib)."""
    import torch

    ram = host.system()
    card = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        card = {"name": properties.name, "total_mib": round(properties.total_memory / MIB)}
    return {"argv": sys.argv, "pid": os.getpid(), "python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "card": card, "ram_total_mib": mib(ram[3]),
            "commit_limit_mib": mib(ram[2]), f"ram_available{when}_mib": mib(ram[0]), f"commit{when}_mib": mib(ram[1]),
            "hz": round(1 / watch.period, 1)}


def watch_from_outside(folder: Path, name: str, watch: mem_watch.MemWatch, hz: float) -> None:
    """mem_watch.py as a child process watching this one, into <name>.samples-outside.csv (same clock: its t = 0 is
    this watch's). It ends when this process does."""
    try:
        subprocess.Popen([sys.executable, mem_watch.__file__, "watch", "--pid", str(os.getpid()),
                          "--out", str(folder / f"{name}.samples-outside.csv"), "--origin", repr(watch.origin),
                          "--hz", str(hz)], stdin=subprocess.DEVNULL)
    except OSError as exc:
        say(f"The outside sampler could not be started ({exc}); only the thread's rows are written.")


def hold_ballast(leave_mib: int, marks: Any) -> None:
    """--profile-ballast-leave: make the CUDA context (the first model move would), then take one block with a raw
    cudaMalloc so that this process can hold `leave_mib` MiB of dedicated memory in all, its context included; what
    Pixal3D needs beyond that the driver puts into shared GPU memory, as on a card that small. Call it on the GPU
    worker thread, before the first network reaches the card. `marks`: whatever has .watch, .file() and .event()
    (the recorder of the launcher with probes): the block's size goes into <name>.ballast.json, which
    pixal3d_memreport.py takes off the process's dedicated memory.

    cudart through ctypes: PyTorch's allocator does not know the block, so torch's allocated / reserved numbers stay
    Pixal3D's own. Why inside the process and not a second one: Windows pushes an IDLE process's video memory out
    to RAM as soon as an active process wants the room (measured: an idle 7.6 GiB ballast in another process shrank
    to 4.2 GiB dedicated while this one took 7 GiB without spilling), so a ballast process emulates nothing; inside
    one process the driver never moves a block once it is placed (measured: blocks allocated after the dedicated
    memory was full stayed 7 times slower, and the idle ballast stayed where it was)."""
    import ctypes

    import torch

    started = time.perf_counter()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    nvml, pdh = mem_watch.NvmlProbe(0), mem_watch.PdhProbe()
    time.sleep(0.5)
    pdh.read()
    time.sleep(0.2)
    used, _, total = nvml.read()
    own = max(0, pdh.read()[0])
    size = (total - used + own) // MIB - WDDM_MARGIN - leave_mib
    size -= size % 2
    if size < 2:
        say(f"PROFILING: no ballast: this process can already hold only "
            f"{(total - used + own) // MIB - WDDM_MARGIN:,} MiB of dedicated memory.")
        HELD["ptr"] = None
        return
    lib = ctypes.WinDLL(str(Path(torch.__file__).parent / "lib" / "cudart64_12.dll"))
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    pointer = ctypes.c_void_p()
    status = lib.cudaMalloc(ctypes.byref(pointer), size * MIB)
    if status != 0:
        raise RuntimeError(f"cudaMalloc of {size} MiB failed (status {status})")
    lib.cudaMemset(pointer, 1, size * MIB)             # written once: the memory really exists
    torch.cuda.synchronize()
    time.sleep(0.5)
    used1 = nvml.read()[0]
    own1, shared1 = pdh.read()[:2]
    HELD.update(ptr=pointer, lib=lib, mib=size)
    note = {"ballast_mib": size, "leave_mib": leave_mib, "t": round(marks.watch.now(), 3),
            "card_total_mib": total // MIB, "card_used_before_mib": used // MIB, "card_used_after_mib": used1 // MIB,
            "own_dedicated_before_mib": own // MIB, "own_dedicated_after_mib": own1 // MIB,
            "own_shared_after_mib": shared1 // MIB, "wddm_margin_mib": WDDM_MARGIN,
            "seconds": round(time.perf_counter() - started, 2)}
    marks.file("ballast.json").write_text(json.dumps(note, indent=1), encoding="utf-8")
    marks.event("note", text="ballast held", **note)
    say(f"A smaller card, emulated: {size:,} MiB held outside PyTorch; this process can hold about {leave_mib:,} MiB "
        f"of dedicated memory in all (the card: {used1 // MIB:,} of {total // MIB:,} MiB in use).")
