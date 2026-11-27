"""Build Spellcast3D's xatlas wheel (xatlas-python 0.0.11 with its vendored xatlas made faster, the same atlas: the
chart-merging pass, then the chart growing and the task scheduler; the user's forks, checked out at
temp_code/xatlas-python with extern/xatlas on its spellcast3d branch), cp311 win_amd64, into <checkout>/dist; copy
the wheel into code/whl, name it in install/components.py XATLAS_WHEEL and write its sidecar (code/whl/xatlas.json)
by hand after the checks below. MSVC's environment is taken from vcvars64.bat inside Python (Git Bash cannot pass
ProgramFiles(x86) to a .bat); scikit-build-core, cmake and ninja come through pip's isolated build. About a minute.

The checks, against the wheel before (the first of ours against PyPI's 0.0.11): unpack both with
`pip install --no-deps --target <folder>`, and with each folder first on PYTHONPATH unwrap the same retopology
meshes with instant_meshes_brush.uv.unwrap at leniency 0.5 and 1.0; every array of every layout (uv, faces, chart,
cut, cut_face, cut_slot, cut_chart, the counts) must be equal bit for bit. The sidecar records the cases.

    <imesh venv>/Scripts/python.exe code/release/wheels/xatlas/build_xatlas_win.py [checkout] [dist]
"""
from __future__ import annotations

import glob
import subprocess
import sys
from pathlib import Path

VCVARS = next(iter(sorted(glob.glob(r"C:\Program Files\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvars64.bat"))), None)


def vcvars_env() -> dict[str, str]:
    cp = subprocess.run(f'cmd /s /c ""{VCVARS}" >nul 2>&1 && set"', capture_output=True, text=True, shell=False)
    env = {}
    for line in cp.stdout.splitlines():
        if "=" in line and not line.startswith("="):
            k, v = line.split("=", 1)
            env[k] = v
    if "VCToolsVersion" not in env:
        sys.exit(f"vcvars64 did not set up MSVC: {VCVARS}")
    return env


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\_myDrive\repos\spellcast3d\temp_code\xatlas-python").resolve()
    dist = Path(sys.argv[2] if len(sys.argv) > 2 else root / "dist").resolve()
    dist.mkdir(parents=True, exist_ok=True)
    env = vcvars_env()
    env["CMAKE_GENERATOR"] = "Ninja"
    env["CMAKE_BUILD_TYPE"] = "Release"
    cmd = [sys.executable, "-m", "pip", "wheel", str(root), "--no-deps", "-w", str(dist)]
    print("building:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env, cwd=root).returncode


if __name__ == "__main__":
    sys.exit(main())
