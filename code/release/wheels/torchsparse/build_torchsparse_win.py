"""Build torchsparse 2.1.0 (mit-han-lab @ 385f5ce, the commit TriFlow's setup.sh pins) as a
Windows cp311 wheel for torch 2.8.0+cu128, with CUDA 12.8 + MSVC 2022, and write a provenance JSON.

    <any python 3.8+> build_torchsparse_win.py [--out DIR] [--work DIR] [--keep-work] ...

What it does (nothing is installed into an existing venv, nothing is written into the source clone):
 1. Clones mit-han-lab/torchsparse at the pin into --src (depth 1) if it is not there; checks HEAD
    and that the clone is clean.
 2. Exports the pinned tree with `git archive` into <work>/src and applies patch_torchsparse_win.py
    (minimal MSVC/nvcc fixes, see that file) plus the wheel's local version label. Clones sparsehash
    2.0.4 (header-only, BSD-3) for the CPU hashmap code and puts sparsehash_msvc/ (a std::hash
    sparseconfig.h) and its src/ on INCLUDE; its COPYING ships as dist-info/licenses/LICENSE-sparsehash.
    setup.cfg gets a [build] build_base on the 8.3 short path (object paths would pass MAX_PATH).
 3. Creates a throw-away build venv from --base-python with pinned build tools from the pip cache
    (setuptools, wheel, ninja) and a .pth that appends --torch-venv's site-packages, so torch
    2.8.0+cu128 is importable without copying or modifying that venv.
 4. Runs `pip wheel --no-deps --no-build-isolation` inside the vcvars64 environment with
    CUDA_HOME=CUDA_PATH=CUDA 12.8, DISTUTILS_USE_SDK=1, FORCE_CUDA=1, TORCH_CUDA_ARCH_LIST, MAX_JOBS.
 5. Lists the SASS/PTX in the .pyd with cuobjdump, reads its DLL imports, hashes the wheel,
    moves it to --out and writes <wheel-stem>.json next to it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import patch_torchsparse_win as patch  # noqa: E402

REPO_URL = "https://github.com/mit-han-lab/torchsparse.git"
PIN = patch.PIN
LOCAL_VERSION = "cu128torch2.8"
ARCH_LIST = "6.1;7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"
BUILD_PINS = ["setuptools==80.9.0", "wheel==0.45.1", "ninja==1.13.2"]
# torchsparse's CPU hashmap/query code includes <google/dense_hash_map> (header-only sparsehash).
SPARSEHASH_URL = "https://github.com/sparsehash/sparsehash.git"
SPARSEHASH_TAG = "sparsehash-2.0.4"
SPARSEHASH_COMMIT = "f93c0c69e959c1c77611d0ba8d107aa971338811"
SPARSEHASH_CFG = HERE / "sparsehash_msvc"   # hub-owned sparseconfig.h (std::hash), see the file

ROOT = HERE.parents[3]  # the Spellcast3D folder (this script is in code/release/wheels/<wheel>/)
SCRATCH = ROOT / "release" / "cache" / "wheels"  # clones, build trees and logs (release/ is git-ignored)
DEFAULTS = {
    "src": SCRATCH / "src" / "torchsparse",
    "sparsehash": SCRATCH / "src" / "sparsehash",
    "work": SCRATCH / "build" / "ts",
    "out": ROOT / "code" / "whl",
    "log_dir": SCRATCH / "logs",
    "base_python": ROOT / "tools" / "python" / "python.exe",
    "torch_venv": ROOT / "code" / "venvs" / "pixal3d",
    "vcvars": Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"),
    "cuda": Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"),
}


def run(cmd, **kw) -> subprocess.CompletedProcess:
    kw.setdefault("check", True)
    kw.setdefault("text", True)
    kw.setdefault("capture_output", True)
    return subprocess.run([str(c) for c in cmd], **kw)


def short_path(p: Path) -> Path:
    """The 8.3 form of an existing path. The scratch folder is ~125 characters deep and nvcc/cl
    object paths would pass MAX_PATH (260) with the long form."""
    import ctypes
    buf = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetShortPathNameW(str(p), buf, len(buf))
    return Path(buf.value) if 0 < n < len(buf) else p


def which(name: str, env: dict) -> str:
    found = shutil.which(name, path=env["PATH"])
    if not found:
        sys.exit(f"{name} not found on the build PATH")
    return found


def git(repo: Path, *args: str) -> str:
    return run(["git", "-C", repo, "-c", "core.autocrlf=false", *args]).stdout


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vcvars_env(vcvars: Path) -> dict[str, str]:
    """The environment after `call vcvars64.bat` (captured with `set`)."""
    cp = subprocess.run(f'cmd /s /c ""{vcvars}" >nul 2>&1 && set"', capture_output=True, text=True, shell=False)
    env = {}
    for line in cp.stdout.splitlines():
        if "=" in line and not line.startswith("="):
            k, v = line.split("=", 1)
            env[k] = v
    if "VCToolsVersion" not in env:
        sys.exit(f"vcvars64 did not set up MSVC: {vcvars}")
    return env


def pe_imports(data: bytes) -> list[str]:
    """DLL names from a PE file's import table (same logic as spellcast3d's prepare_release.py)."""
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    opt_size = struct.unpack_from("<H", data, pe + 20)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    dd = opt + (112 if magic == 0x20B else 96)
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


def cuobjdump_arches(cuobjdump: Path, pyd: Path) -> dict:
    elf = run([cuobjdump, "--list-elf", pyd], check=False).stdout
    ptx = run([cuobjdump, "--list-ptx", pyd], check=False).stdout
    sm_elf = sorted({int(m) for m in re.findall(r"\.sm_(\d+)\.cubin", elf)})
    sm_ptx = sorted({int(m) for m in re.findall(r"\.sm_(\d+)\.ptx", ptx)})
    return {"sass_sm": sm_elf, "ptx_compute": sm_ptx,
            "cubin_images": len(re.findall(r"\.cubin", elf)), "ptx_images": len(re.findall(r"\.ptx", ptx))}


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


def ensure_sparsehash(src: Path) -> None:
    if not (src / ".git").exists():
        src.mkdir(parents=True, exist_ok=True)
        run(["git", "init", "-q", src])
        git(src, "config", "core.autocrlf", "false")
        git(src, "remote", "add", "origin", SPARSEHASH_URL)
        run(["git", "-C", src, "fetch", "-q", "--depth", "1", "origin",
             f"refs/tags/{SPARSEHASH_TAG}:refs/tags/{SPARSEHASH_TAG}"], capture_output=False)
        git(src, "checkout", "-q", "--detach", SPARSEHASH_TAG)
    head = git(src, "rev-parse", "HEAD").strip()
    if head != SPARSEHASH_COMMIT or git(src, "status", "--porcelain").strip():
        sys.exit(f"{src} is not a clean {SPARSEHASH_TAG} ({SPARSEHASH_COMMIT}) checkout")


def export_tree(src: Path, dest: Path) -> None:
    data = run(["git", "-C", src, "-c", "core.autocrlf=false", "archive", "--format=tar", PIN],
               text=False).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        t.extractall(dest)


def make_build_venv(base_python: Path, venv: Path, torch_venv: Path, env: dict) -> Path:
    run([base_python, "-m", "venv", venv], env=env)
    py = venv / "Scripts" / "python.exe"
    run([py, "-m", "pip", "install", "--disable-pip-version-check", "-q", *BUILD_PINS], env=env,
        capture_output=False)
    site = run([py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], env=env).stdout.strip()
    torch_site = torch_venv / "Lib" / "site-packages"
    # .pth entries are APPENDED to sys.path, so this venv's own setuptools/wheel win over torch_venv's.
    (Path(site) / "zz_torch_from_venv.pth").write_text(str(torch_site) + "\n", encoding="utf-8")
    return py


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=Path, default=v)
    ap.add_argument("--arch-list", default=ARCH_LIST)
    ap.add_argument("--max-jobs", default="8")
    ap.add_argument("--keep-work", action="store_true", help="keep the build venv and build tree")
    a = ap.parse_args()
    for k, v in vars(a).items():   # relative paths would break once cwd moves to the work dir
        if isinstance(v, Path):
            setattr(a, k, v.absolute())

    started = datetime.now(timezone.utc)
    a.out.mkdir(parents=True, exist_ok=True)
    a.log_dir.mkdir(parents=True, exist_ok=True)
    ensure_source(a.src)
    ensure_sparsehash(a.sparsehash)
    if a.work.exists():
        shutil.rmtree(a.work)
    (a.work / "src").mkdir(parents=True)
    (a.work / "tmp").mkdir()
    work = short_path(a.work)
    export_tree(a.src, a.work / "src")
    applied = patch.apply(a.work / "src", LOCAL_VERSION)
    for h in applied:
        print(f"patched [{h['id']}] {h['file']}")
    # sparsehash templates are compiled into the .pyd, so its BSD-3 notice ships in the wheel
    # (setuptools' default license_files glob LICEN[CS]E* picks this up into dist-info/licenses/).
    shutil.copyfile(a.sparsehash / "COPYING", a.work / "src" / "LICENSE-sparsehash")
    # Build config only (not a source patch): keep setuptools' build tree on the short path.
    with open(a.work / "src" / "setup.cfg", "a", encoding="utf-8") as f:
        f.write(f"\n[build]\nbuild_base = {work / 'o'}\n")

    env = vcvars_env(a.vcvars)
    for k in ("PYTHONPATH", "PYTHONHOME", "PIP_REQUIRE_VIRTUALENV", "VIRTUAL_ENV"):
        env.pop(k, None)
    venv = a.work / "venv"
    py = make_build_venv(a.base_python, venv, a.torch_venv, env)
    cuda = str(a.cuda)
    env.update({
        "CUDA_HOME": cuda, "CUDA_PATH": cuda,            # the inherited CUDA_HOME points to a missing v12.4
        "DISTUTILS_USE_SDK": "1", "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": a.arch_list, "MAX_JOBS": str(a.max_jobs),
        "PATH": os.pathsep.join([str(venv / "Scripts"), str(a.cuda / "bin"), env["PATH"]]),
        "TEMP": str(work / "tmp"), "TMP": str(work / "tmp"),   # nvcc/cl intermediates stay in the work dir
        # cl searches INCLUDE after the /I dirs: our sparseconfig.h first, then sparsehash's headers.
        "INCLUDE": os.pathsep.join([str(short_path(SPARSEHASH_CFG)), str(short_path(a.sparsehash / "src")),
                                    env.get("INCLUDE", "")]),
    })
    probe = run([py, "-c", "import torch, setuptools, wheel, sys; print(torch.__version__, torch.version.cuda, "
                           "setuptools.__version__, wheel.__version__, torch.__file__)"], env=env).stdout.split()
    torch_ver, torch_cuda, st_ver, wh_ver = probe[:4]
    nvcc_ver = re.search(r"release ([\d.]+), (V[\d.]+)", run([a.cuda / "bin" / "nvcc.exe", "--version"]).stdout)
    cl_banner = run([which("cl", env)], env=env, check=False).stderr.splitlines()[0]
    ninja_ver = run([which("ninja", env), "--version"], env=env).stdout.strip()
    print(f"torch {torch_ver} (CUDA {torch_cuda}) | setuptools {st_ver} | wheel {wh_ver} | ninja {ninja_ver}")
    print(f"nvcc {nvcc_ver.group(1)} {nvcc_ver.group(2)} | {cl_banner}")
    print(f"TORCH_CUDA_ARCH_LIST={a.arch_list} MAX_JOBS={a.max_jobs}")

    wheel_out = a.work / "wheel-out"
    wheel_out.mkdir()
    log = a.log_dir / "torchsparse-build.log"
    cmd = [py, "-m", "pip", "wheel", work / "src", "--no-deps", "--no-build-isolation",
           "--disable-pip-version-check", "-v", "-w", wheel_out]
    print("running:", " ".join(map(str, cmd)))
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"# {' '.join(map(str, cmd))}\n# TORCH_CUDA_ARCH_LIST={a.arch_list} MAX_JOBS={a.max_jobs}\n")
        f.flush()
        rc = subprocess.run([str(c) for c in cmd], env=env, cwd=work, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        sys.exit(f"build failed (exit {rc}); log: {log}")
    wheels = list(wheel_out.glob("torchsparse-*.whl"))
    if len(wheels) != 1 or not wheels[0].name.endswith("-cp311-cp311-win_amd64.whl"):
        sys.exit(f"expected one cp311 win_amd64 wheel, got {[w.name for w in wheels]}")
    wheel = a.out / wheels[0].name
    shutil.move(str(wheels[0]), wheel)

    with zipfile.ZipFile(wheel) as z:
        pyds = [n for n in z.namelist() if n.endswith(".pyd")]
        pyd_path = a.work / Path(pyds[0]).name
        pyd_path.write_bytes(z.read(pyds[0]))
        imports = pe_imports(z.read(pyds[0]))
        files = z.namelist()
    arches = cuobjdump_arches(a.cuda / "bin" / "cuobjdump.exe", pyd_path)
    print(f"SASS sm: {arches['sass_sm']}  PTX compute: {arches['ptx_compute']}")

    ts_lic = (a.work / "src" / "LICENSE")
    prov = {
        "wheel": wheel.name,
        "sha256": sha256_file(wheel),
        "size_bytes": wheel.stat().st_size,
        "built": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "build_seconds": int((datetime.now(timezone.utc) - started).total_seconds()),
        "source_repo": "https://github.com/mit-han-lab/torchsparse",
        "source_commit": PIN,
        "source_commit_why": "the commit TriFlow's setup.sh:11 installs (IgorAherne/triflow @ 873a2cdf68b0a1ec243635bbf9170c87556b2f2b)",
        "source_license": "MIT (LICENSE is inside the wheel's dist-info)" if ts_lic.is_file() else "MIT",
        "patches": {
            "script": "code/release/wheels/torchsparse/patch_torchsparse_win.py",
            "applied_to": "a clean `git archive` export of the pinned commit (the clone itself stays untouched)",
            "equivalent_to": "the Windows edits of Deathdadev/torchsparse (MIT) commit f1787ee "
                             "(in bebadc476e...), except the hash_cpu int->int64_t change (it breaks int32 "
                             "coords, which torchsparse asserts) and the no-op uint64_t->unsigned long long edits",
            "hunks": applied,
        },
        "vendored_headers": {
            "sparsehash": {"repo": "https://github.com/sparsehash/sparsehash", "tag": SPARSEHASH_TAG,
                           "commit": SPARSEHASH_COMMIT, "license": "BSD-3-Clause (Google), shipped as "
                           "dist-info/licenses/LICENSE-sparsehash",
                           "used_by": "torchsparse/backend/hashmap/hashmap_cpu.cpp, others/query_cpu.cpp (CPU tensors only)",
                           "config": "code/release/wheels/torchsparse/sparsehash_msvc/sparsehash/internal/sparseconfig.h "
                                     "(std::hash, as ./configure generates on Linux; sparsehash's own Windows "
                                     "config names stdext::hash_compare, which MSVC 17.14 no longer has)",
                           "how": "INCLUDE=<sparseconfig dir>;<sparsehash>/src;%INCLUDE% (no torchsparse source change)"},
        },
        "python": "CPython 3.11.9 (spellcast3d tools/python), throw-away build venv",
        "torch_at_build": f"{torch_ver} (CUDA {torch_cuda}) from {a.torch_venv.relative_to(ROOT) if a.torch_venv.is_relative_to(ROOT) else a.torch_venv} via a .pth (not installed/modified)",
        "msvc": f"{cl_banner} (Visual Studio Community 2022 17.14, toolset {env.get('VCToolsVersion')}), "
                f"dynamic CRT /MD like torch",
        "windows_sdk": env.get("WindowsSDKVersion", "").strip("\\"),
        "cuda": f"CUDA {nvcc_ver.group(1)} (nvcc {nvcc_ver.group(2)}) at {a.cuda}",
        "torch_cuda_arch_list": a.arch_list,
        "cuobjdump": arches,
        "pyd": pyds[0],
        "pyd_dll_imports": imports,
        "build_tools": {"setuptools": st_ver, "wheel": wh_ver, "ninja": ninja_ver},
        "env": {"CUDA_HOME": cuda, "DISTUTILS_USE_SDK": "1", "FORCE_CUDA": "1",
                "TORCH_CUDA_ARCH_LIST": a.arch_list, "MAX_JOBS": str(a.max_jobs)},
        "build_command": " ".join(["python", "-m", "pip", "wheel", "<patched export>", "--no-deps",
                                   "--no-build-isolation", "-v", "-w", "<out>"]) +
                         "  (inside vcvars64; driver: code/release/wheels/torchsparse/build_torchsparse_win.py)",
        "runtime_requirements": {
            "install": "pip install --no-deps <this wheel>",
            "import_needs": ["torch 2.8.x+cu128", "numpy", "tqdm", "typing-extensions",
                             "rootpath 0.1.1 (--no-deps: at import it needs only six)", "six"],
            "not_needed_on_py311": ["backports.cached_property (functools.cached_property is used)",
                                    "wheel", "torchvision (metadata only)"],
        },
        "wheel_files": len(files),
        "wheel_license_files": [n for n in files if "/licenses/" in n or n.endswith(".dist-info/LICENSE")],
        "build_log": str(log.relative_to(ROOT)) if log.is_relative_to(ROOT) else str(log),
    }
    js = a.out / "torchsparse.json"
    js.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    print(f"wheel {wheel} ({wheel.stat().st_size / 1e6:.1f} MB) sha256 {prov['sha256']}\nprovenance {js}")
    if not a.keep_work:
        shutil.rmtree(a.work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
