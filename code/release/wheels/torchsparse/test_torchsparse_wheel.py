"""Test a built torchsparse wheel WITHOUT installing it: unpack it to a scratch folder, put that (plus a
folder holding rootpath 0.1.1, its only non-standard import dependency) on PYTHONPATH, and run, with the
python of an existing torch 2.8.0+cu128 venv:
  1. `import torchsparse` (+ backend .pyd load) from the unpacked wheel,
  2. smoke_torchsparse.py (GPU checks against dense conv3d).
The GPU is shared: before step 2 it waits (10 min, up to 6 times) while other processes hold > 6 GB of
VRAM or free RAM is < 6 GB.

    python test_torchsparse_wheel.py <wheel> --python <venv python> --deps <dir with rootpath/> [--out result.json]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def free_ram_gb() -> float:
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30


def gpu_used_gb() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split()
    return float(out[0]) / 1024 if out else 0.0


def wait_for_resources(tries: int = 6, wait_s: int = 600) -> dict:
    for i in range(tries + 1):
        used, ram = gpu_used_gb(), free_ram_gb()
        print(f"[gate] GPU used by others {used:.1f} GB, free RAM {ram:.1f} GB", flush=True)
        if used <= 6.0 and ram >= 6.0:
            return {"gpu_used_gb_before": round(used, 2), "free_ram_gb_before": round(ram, 1), "waits": i}
        if i < tries:
            print(f"[gate] busy, waiting {wait_s // 60} min ({i + 1}/{tries})", flush=True)
            time.sleep(wait_s)
    sys.exit("GPU/RAM stayed busy; not running the GPU test")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel", type=Path)
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--deps", type=Path, required=True, help="folder that contains the rootpath package")
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    work = a.work or a.wheel.parent.parent / "test" / "ts"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    with zipfile.ZipFile(a.wheel) as z:
        z.extractall(work)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "VIRTUAL_ENV")}
    env["PYTHONPATH"] = os.pathsep.join([str(work), str(a.deps)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result: dict = {"wheel": a.wheel.name, "python": str(a.python)}

    probe = ("import json, torch, torchsparse, torchsparse.backend as b; "
             "print(json.dumps({'torchsparse': torchsparse.__version__, 'file': torchsparse.__file__, "
             "'backend': b.__file__, 'torch': torch.__version__, 'cuda': torch.version.cuda}))")
    cp = subprocess.run([str(a.python), "-c", probe], env=env, capture_output=True, text=True, cwd=work.parent)
    result["import"] = json.loads(cp.stdout.strip().splitlines()[-1]) if cp.returncode == 0 else {"error": cp.stderr[-2000:]}
    result["import_ok"] = cp.returncode == 0 and str(work) in result["import"].get("backend", "")
    if not result["import_ok"]:
        print(json.dumps(result, indent=1))
        return 1

    result["gate"] = wait_for_resources()
    t = time.perf_counter()
    cp = subprocess.run([str(a.python), str(HERE / "smoke_torchsparse.py")], env=env, capture_output=True,
                        text=True, cwd=work.parent)
    result["smoke_seconds"] = round(time.perf_counter() - t, 1)
    try:
        result["smoke"] = json.loads(cp.stdout)
    except json.JSONDecodeError:
        result["smoke"] = {"stdout": cp.stdout[-3000:], "stderr": cp.stderr[-3000:]}
    result["smoke_stderr_tail"] = cp.stderr[-1500:]
    result["passed"] = result["import_ok"] and cp.returncode == 0
    text = json.dumps(result, indent=1)
    print(text)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
