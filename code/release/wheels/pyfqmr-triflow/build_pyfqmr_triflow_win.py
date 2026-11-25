"""Build pyfqmr-triflow 0.5.0+triflow (DerKleineLi/pyfqmr-triflow @ c93a767, the commit TriFlow's
third_party/pyfqmr-triflow gitlink pins) as a Windows cp311 wheel with MSVC 2022, OUT OF TREE, and
write a provenance JSON next to it.

    <any python 3.8+> build_pyfqmr_triflow_win.py [--out DIR] [--work DIR] [--keep-work] ...

 1. Clones the pinned commit into --src (depth 1) if missing; checks HEAD and that the clone is clean.
 2. Exports the tree with `git archive` into <work>/src (so the build's Simplify.cpp, build/ and
    *.egg-info never land in the clone / TriFlow's submodule). No source patches are needed.
 3. Creates a throw-away build venv from --base-python with pinned build requirements from the pip
    cache (setuptools, wheel, Cython, numpy) and runs `pip wheel --no-deps --no-build-isolation`
    inside the vcvars64 environment (DISTUTILS_USE_SDK=1).
 4. Checks the wheel carries LICENSE, reads the .pyd's DLL imports, hashes it, writes pyfqmr_triflow.json.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/DerKleineLi/pyfqmr-triflow.git"
PIN = "c93a7671ba4a2163dd3c7ea6ab6b52ad0167cc72"
BUILD_PINS = ["setuptools==80.9.0", "wheel==0.45.1", "Cython==3.2.4", "numpy==2.2.6"]

ROOT = HERE.parents[3]  # the Spellcast3D folder (this script is in code/release/wheels/<wheel>/)
SCRATCH = ROOT / "release" / "cache" / "wheels"  # clones, build trees and logs (release/ is git-ignored)
DEFAULTS = {
    "src": SCRATCH / "src" / "pyfqmr-triflow",
    "work": SCRATCH / "build" / "fq",
    "out": ROOT / "code" / "whl",
    "log_dir": SCRATCH / "logs",
    "base_python": ROOT / "tools" / "python" / "python.exe",
    "vcvars": Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"),
}


def run(cmd, **kw) -> subprocess.CompletedProcess:
    kw.setdefault("check", True)
    kw.setdefault("text", True)
    kw.setdefault("capture_output", True)
    return subprocess.run([str(c) for c in cmd], **kw)


def git(repo: Path, *args: str) -> str:
    return run(["git", "-C", repo, "-c", "core.autocrlf=false", *args]).stdout


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vcvars_env(vcvars: Path) -> dict[str, str]:
    cp = subprocess.run(f'cmd /s /c ""{vcvars}" >nul 2>&1 && set"', capture_output=True, text=True)
    env = dict(line.split("=", 1) for line in cp.stdout.splitlines() if "=" in line and not line.startswith("="))
    if "VCToolsVersion" not in env:
        sys.exit(f"vcvars64 did not set up MSVC: {vcvars}")
    return env


def pe_imports(data: bytes) -> list[str]:
    """DLL names from a PE file's import table (same logic as spellcast3d's prepare_release.py)."""
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    opt_size = struct.unpack_from("<H", data, pe + 20)[0]
    opt = pe + 24
    dd = opt + (112 if struct.unpack_from("<H", data, opt)[0] == 0x20B else 96)
    import_rva = struct.unpack_from("<I", data, dd + 8)[0]
    secs = [struct.unpack_from("<IIII", data, opt + opt_size + 40 * i + 8) for i in range(nsec)]

    def offset(rva):
        for vsize, vaddr, rsize, raddr in secs:
            if vaddr <= rva < vaddr + max(vsize, rsize):
                return rva - vaddr + raddr
        return None

    names, p = [], offset(import_rva) if import_rva else None
    while p is not None and p + 20 <= len(data):
        name_rva = struct.unpack_from("<I", data, p + 12)[0]
        o = offset(name_rva) if name_rva else None
        if o is None:
            break
        names.append(data[o:data.index(b"\0", o)].decode("ascii", "replace"))
        p += 20
    return names


def ensure_source(src: Path) -> None:
    if not (src / ".git").exists():
        src.mkdir(parents=True, exist_ok=True)
        run(["git", "init", "-q", src])
        git(src, "config", "core.autocrlf", "false")
        git(src, "remote", "add", "origin", REPO_URL)
        run(["git", "-C", src, "fetch", "-q", "--depth", "1", "origin", PIN], capture_output=False)
        git(src, "checkout", "-q", "--detach", "FETCH_HEAD")
    head = git(src, "rev-parse", "HEAD").strip()
    if head != PIN:
        sys.exit(f"{src} is at {head}, expected {PIN}")
    if git(src, "status", "--porcelain").strip():
        sys.exit(f"{src} has local changes; the wheel would not match the pinned commit")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=Path, default=v)
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    for k, v in vars(a).items():   # relative paths would break once cwd moves to the work dir
        if isinstance(v, Path):
            setattr(a, k, v.absolute())

    started = datetime.now(timezone.utc)
    a.out.mkdir(parents=True, exist_ok=True)
    a.log_dir.mkdir(parents=True, exist_ok=True)
    ensure_source(a.src)
    if a.work.exists():
        shutil.rmtree(a.work)
    (a.work / "src").mkdir(parents=True)
    data = run(["git", "-C", a.src, "-c", "core.autocrlf=false", "archive", "--format=tar", PIN], text=False).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        t.extractall(a.work / "src")

    env = vcvars_env(a.vcvars)
    for k in ("PYTHONPATH", "PYTHONHOME", "PIP_REQUIRE_VIRTUALENV", "VIRTUAL_ENV"):
        env.pop(k, None)
    env["DISTUTILS_USE_SDK"] = "1"
    # Reproducible output: /Brepro (cl + link) drops PE timestamps/GUIDs, SOURCE_DATE_EPOCH fixes the
    # wheel's zip timestamps (to the pinned commit's time), so a rebuild gives the same sha256.
    commit_time = git(a.src, "log", "-1", "--format=%ct", PIN).strip()
    env.update({"_CL_": "/Brepro", "_LINK_": "/Brepro", "SOURCE_DATE_EPOCH": commit_time,
                "PYTHONHASHSEED": "0"})   # Cython code generation order
    venv = a.work / "venv"
    run([a.base_python, "-m", "venv", venv], env=env)
    py = venv / "Scripts" / "python.exe"
    run([py, "-m", "pip", "install", "--disable-pip-version-check", "-q", *BUILD_PINS], env=env, capture_output=False)
    versions = json.loads(run([py, "-c", "import json, setuptools, wheel, Cython, numpy; print(json.dumps({"
                                         "'setuptools': setuptools.__version__, 'wheel': wheel.__version__, "
                                         "'Cython': Cython.__version__, 'numpy': numpy.__version__}))"], env=env).stdout)
    cl = shutil.which("cl", path=env["PATH"])
    cl_banner = run([cl], env=env, check=False).stderr.splitlines()[0]
    print(f"build tools {versions} | {cl_banner}")

    wheel_out = a.work / "wheel-out"
    wheel_out.mkdir()
    log = a.log_dir / "pyfqmr_triflow-build.log"
    cmd = [py, "-m", "pip", "wheel", a.work / "src", "--no-deps", "--no-build-isolation",
           "--disable-pip-version-check", "-v", "-w", wheel_out]
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"# {' '.join(map(str, cmd))}\n")
        f.flush()
        rc = subprocess.run([str(c) for c in cmd], env=env, cwd=a.work, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        sys.exit(f"build failed (exit {rc}); log: {log}")
    wheels = list(wheel_out.glob("pyfqmr_triflow-*.whl"))
    if len(wheels) != 1 or not wheels[0].name.endswith("-cp311-cp311-win_amd64.whl"):
        sys.exit(f"expected one cp311 win_amd64 wheel, got {[w.name for w in wheels]}")
    wheel = a.out / wheels[0].name
    if wheel.exists():
        wheel.unlink()
    shutil.move(str(wheels[0]), wheel)

    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        pyds = [n for n in names if n.endswith(".pyd")]
        imports = {n: pe_imports(z.read(n)) for n in pyds}
        lic = [n for n in names if re.search(r"\.dist-info/(licenses/)?LICENSE", n)]
        lic_ok = bool(lic) and z.read(lic[0]) == (a.work / "src" / "LICENSE").read_bytes()
    if not lic_ok:
        sys.exit(f"{wheel.name} does not carry the repo's LICENSE (found {lic})")

    prov = {
        "wheel": wheel.name,
        "sha256": sha256_file(wheel),
        "size_bytes": wheel.stat().st_size,
        "built": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "build_seconds": int((datetime.now(timezone.utc) - started).total_seconds()),
        "source_repo": "https://github.com/DerKleineLi/pyfqmr-triflow (branch triflow)",
        "source_commit": PIN,
        "source_commit_why": "the gitlink of third_party/pyfqmr-triflow in IgorAherne/triflow @ 873a2cdf68b0a1ec243635bbf9170c87556b2f2b",
        "source_license": "MIT (Copyright Kramer84); LICENSE is inside the wheel: " + ", ".join(lic),
        "patches": "none (built out of tree from a `git archive` export; the clone/submodule stays untouched)",
        "python": "CPython 3.11.9 (spellcast3d tools/python), throw-away build venv",
        "msvc": f"{cl_banner} (Visual Studio Community 2022 17.14, toolset {env.get('VCToolsVersion')}), "
                "setuptools default flags, dynamic CRT /MD",
        "windows_sdk": env.get("WindowsSDKVersion", "").strip("\\"),
        "cuda": "none (CPU only)",
        "build_deps": versions,
        "pyd_dll_imports": imports,
        "build_command": "python -m pip wheel <git-archive export> --no-deps --no-build-isolation -v -w <out>  "
                         "(inside vcvars64, DISTUTILS_USE_SDK=1, _CL_=/Brepro, _LINK_=/Brepro, "
                         f"SOURCE_DATE_EPOCH={commit_time}; driver: code/release/wheels/pyfqmr-triflow/build_pyfqmr_triflow_win.py)",
        "reproducible": "bit-identical when rebuilt with the same tools from the same --work path "
                        "(verified: two builds, same sha256); a different work path changes the .pyd layout",
        "runtime_requirements": {"install": "pip install --no-deps <this wheel>", "import_needs": ["numpy"]},
        "wheel_files": names,
        "build_log": str(log.relative_to(ROOT)) if log.is_relative_to(ROOT) else str(log),
    }
    js = a.out / "pyfqmr_triflow.json"
    js.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    print(f"wheel {wheel} ({wheel.stat().st_size} bytes) sha256 {prov['sha256']}\nprovenance {js}")
    if not a.keep_work:
        shutil.rmtree(a.work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
