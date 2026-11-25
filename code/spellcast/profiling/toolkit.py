"""The small helpers the measuring tools of the 3D stage share: what nvidia-smi says about the card, the hash of a
tensor's values, the console line. For developers; stdlib only at import (torch and numpy only when a tensor is
hashed), and it imports none of its neighbours, so every tool can import it however it was started: as
`spellcast.profiling.toolkit`, or as `toolkit` with this folder on sys.path (gpu_lock.py, p3d_checks/common.py and
the scripts in code/devtools/pixal3d-lowmem).

    card()            {"name", "used_mib", "total_mib", "util", "driver", "processes"}, or "error" when nvidia-smi
                      could not be asked (gpu_lock.py waits on it, pixal3d_membench.py records it)
    sha256_of(x)      SHA-256 of a tensor's (or a numpy array's) values: the hash every golden manifest carries
    tensor_record(x)  {"shape", "dtype", "sha256"} of one tensor (pixal3d_staged_profiled.py's mesh hashes)
    teller("bench")   a print function whose lines start "[bench 12:34:56] "

mem_watch.py keeps its own reading of the card (NVML through ctypes, 20 times a second, no child process): it is the
sampler, this is the one-off question before and after a run.
"""

from __future__ import annotations

import hashlib
import subprocess
import time
from typing import Any, Callable

MIB = 1 << 20
CHUNK = 64 << 20                    # bytes hashed at a time: a tensor of several GB is never copied whole


def teller(tag: str, clock: bool = True) -> Callable[[str], None]:
    """print() for one tool: every line starts with "[tag HH:MM:SS] " (without the time when clock is False)."""
    def say(text: str) -> None:
        print(f"[{tag} {time.strftime('%H:%M:%S')}] {text}" if clock else f"[{tag}] {text}", flush=True)
    return say


def card(timeout: float = 30.0) -> dict[str, Any]:
    """The card as nvidia-smi sees it now: {"name", "used_mib", "total_mib", "util", "driver", "processes"}. When
    it cannot be asked, "error" says why (with the first query's fields when only the list of processes failed).
    The whole card, every process and the desktop included."""
    out: dict[str, Any] = {}
    try:
        text = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,driver_version",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=timeout).stdout
        name, used, total, util, driver = [part.strip() for part in text.strip().splitlines()[0].split(",")]
        out = {"name": name, "used_mib": int(used), "total_mib": int(total), "util": int(util), "driver": driver}
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=timeout).stdout
        out["processes"] = [line.strip() for line in apps.strip().splitlines() if line.strip()]
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _values(x: Any) -> Any:
    """A C-contiguous numpy array over a tensor's values (on the CPU; bfloat16 as its int16 bits, complex numbers as
    pairs of reals: numpy has neither), or the numpy array itself made contiguous."""
    if type(x).__module__.split(".")[0] == "torch":
        import torch

        value = x.detach().cpu().contiguous()
        if value.dtype == torch.bfloat16:
            value = value.view(torch.int16)
        elif value.is_complex():
            value = torch.view_as_real(value).contiguous()
        return value.numpy()
    import numpy as np

    return np.ascontiguousarray(x)


def tensor_bytes(x: Any) -> Any:
    """The values' bytes in row-major (C) order, little-endian, without a copy where the tensor allows it."""
    array = _values(x)
    return memoryview(array.reshape(-1)).cast("B") if array.size else b""


def sha256_of(x: Any) -> str:
    """SHA-256 of a tensor's values: its bytes in row-major order, nothing else (no shape, no dtype; a view hashes
    in its own order). The hash the golden manifests carry, so a tensor of any run can be held against them."""
    data = tensor_bytes(x)
    digest = hashlib.sha256()
    for start in range(0, len(data), CHUNK):
        digest.update(data[start:start + CHUNK])
    return digest.hexdigest()


def tensor_record(x: Any) -> dict[str, Any]:
    """{"shape", "dtype", "sha256"} of a tensor (dtype as torch names it, without "torch.") or a numpy array."""
    dtype = str(x.dtype).replace("torch.", "") if type(x).__module__.split(".")[0] == "torch" else _values(x).dtype.name
    return {"shape": list(x.shape), "dtype": dtype, "sha256": sha256_of(x)}
