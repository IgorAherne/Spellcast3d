"""Build cumesh from the user's fork (https://github.com/IgorAherne/CuMesh, branch spellcast3d, at the commit
patch_cumesh_win.PINS["fork"] names: JeffreyXiang/CuMesh main 12289e1 merged, cubvh ce92267 vendored, eigen e63d9f6cc as
a submodule, Spellcast3D's edits as commits) as a Windows cp311 wheel for torch 2.8.0+cu128, with CUDA 12.8 + MSVC 2022,
CUDA kernels for sm_61..sm_120 + PTX, and write a provenance JSON.

    <any python 3.8+> build_cumesh_win.py [--src CLONE] [--out DIR] [--work DIR] [--keep-work] [--no-check] ...

What it does (nothing is installed into an existing venv; a source repository only gains the pinned commits when
it lacks them):
 1. Finds the pinned fork commit in --src (any clone of the fork - at any branch or HEAD, with any local edits - or a
    bare repository; one is made under release/cache/wheels/src and the commit fetched with depth 1 when it is not
    there) and eigen's pinned commit in --eigen-src (default: the clone's initialised submodule when it has the
    commit, else release/cache/wheels/src/eigen, fetched the same way); the fork's submodule link must name it.
 2. Unless --no-check: `patch_cumesh_win.py check` - the fork's commit file by file against upstream 12289e1 + cubvh
    ce92267 + the hunks listed there (README.md aside), so every change the wheel carries is one of those hunks.
 3. Exports the fork's commit with `git archive` into <work>/src and eigen's into third_party/cubvh/third_party/eigen;
    nothing is patched (the version in pyproject.toml must be patch_cumesh_win.WHEEL_VERSION). setup.cfg gets a
    [build] build_base on the 8.3 short path (object paths would pass MAX_PATH).
 4. Creates a throw-away build venv from --base-python with pinned build tools from the pip cache (setuptools,
    wheel, ninja) and a .pth that appends --torch-venv's site-packages, so torch 2.8.0+cu128 is importable
    without copying or modifying that venv.
 5. Runs `pip wheel --no-deps --no-build-isolation` inside the vcvars64 environment with CUDA_HOME=CUDA_PATH=
    CUDA 12.8, DISTUTILS_USE_SDK=1, FORCE_CUDA=1, TORCH_CUDA_ARCH_LIST, MAX_JOBS.
 6. Lists the SASS/PTX of the two CUDA .pyds with cuobjdump, reads their DLL imports, hashes the wheel, moves it
    to --out (the fork's whl folder, which Pixal3D's install.py installs from) and writes cumesh.json next to it.

About 9-10 minutes with MAX_JOBS 8 on a 16-core PC (eleven .cu files of the mesh library, two of cubvh, nine
architectures). Wheels 0.0.1.post1-post3 were built from JeffreyXiang/CuMesh b84410e + cubvh ff89002 with the hunks
applied at build time (this script and patch_cumesh_win.py before 2026-10-16, in Spellcast3D's history).
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
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import patch_cumesh_win as patch  # noqa: E402

FORK_URL, FORK_PIN = patch.PINS["fork"]
EIGEN_URL, EIGEN_PIN = patch.PINS["eigen"]
# Upstream's version with a post-release number that moves with every build of ours (post1: the cuBVH built in RAM
# and uploaded on demand, the leak in fill_holes' chain; post2: fill_holes' chain in less card memory, the same tables
# bit for bit; post3: the ray-stab's inside test with an any-hit traversal, the same verdicts bit for bit; post4: the
# same edits on upstream's main of 2026-03 - its CUDA stream handling for Blackwell, the M=0 guard, its Windows build,
# cubvh's namespace - built from the user's fork; post5: cuBVH's tree built on several threads, the same tree byte
# for byte; post6: the ray-stab's inside test for any subset of a call's points, the same verdicts bit for bit;
# post7: xatlas's chart growing and task scheduler faster, the same atlas byte for byte)
# and the local label of the torch build: pip installs a wheel over
# an installed one only when its version is greater, and Pixal3D's install.py installs the wheels of its whl folder
# on every run. It lives in the fork's pyproject.toml (hunk local-version).
VERSION = patch.WHEEL_VERSION
ARCH_LIST = "6.1;7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"
BUILD_PINS = ["setuptools==80.9.0", "wheel==0.45.1", "ninja==1.13.2"]

ROOT = HERE.parents[3]  # the Spellcast3D folder (this script is in code/release/wheels/<wheel>/)
SCRATCH = ROOT / "release" / "cache" / "wheels"  # clones, build trees and logs (release/ is git-ignored)
DEFAULTS = {
    "src": SCRATCH / "src" / "cumesh",
    "eigen_src": None,
    "cubvh_src": SCRATCH / "src" / "cubvh",
    "work": SCRATCH / "build" / "cm",
    "out": ROOT / "code" / "pixal3d" / "whl",
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
    """The 8.3 form of an existing path: nvcc/cl object paths would pass MAX_PATH (260) with the long form."""
    import ctypes
    buf = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetShortPathNameW(str(p), buf, len(buf))
    return Path(buf.value) if 0 < n < len(buf) else p


def which(name: str, env: dict) -> str:
    found = shutil.which(name, path=env["PATH"])
    if not found:
        sys.exit(f"{name} not found on the build PATH")
    return found


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
    # cmd's `set` spells it as the parent process does (Path from PowerShell, PATH from Git Bash)
    for key in [k for k in env if k.upper() == "PATH" and k != "PATH"]:
        env["PATH"] = env.pop(key)
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


def ensure_sources(src: Path, eigen_src: Path | None, rev: str = FORK_PIN) -> dict[str, str]:
    """The fork's pinned commit in `src` and eigen's in `eigen_src` (made and fetched when missing) ->
    {"fork": full commit, "eigen": full commit, "eigen_repo": the repository eigen is exported from}."""
    fork = patch.ensure_commit(src, FORK_URL, rev, "the fork")
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head and head != fork:
        print(f"note: {src} is at {head[:9]}; the wheel is built from {fork[:9]} all the same")
    link = patch.gitlinks(src, fork).get(patch.EIGEN_DIR)
    if link != EIGEN_PIN:
        sys.exit(f"the fork's {patch.EIGEN_DIR} points at {link}, not eigen {EIGEN_PIN[:9]}")
    if eigen_src is None:
        submodule = src / patch.EIGEN_DIR
        eigen_src = submodule if (submodule / ".git").exists() and patch.has_commit(submodule, EIGEN_PIN) \
            else SCRATCH / "src" / "eigen"
    return {"fork": fork, "eigen": patch.ensure_commit(eigen_src, EIGEN_URL, EIGEN_PIN, "eigen"),
            "eigen_repo": str(eigen_src)}


def make_build_venv(base_python: Path, venv: Path, torch_venv: Path, env: dict) -> Path:
    run([base_python, "-m", "venv", venv], env=env)
    py = venv / "Scripts" / "python.exe"
    run([py, "-m", "pip", "install", "--disable-pip-version-check", "-q", *BUILD_PINS], env=env, capture_output=False)
    site = run([py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], env=env).stdout.strip()
    torch_site = torch_venv / "Lib" / "site-packages"
    # .pth entries are APPENDED to sys.path, so this venv's own setuptools/wheel win over torch_venv's.
    (Path(site) / "zz_torch_from_venv.pth").write_text(str(torch_site) + "\n", encoding="utf-8")
    return py


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=Path, default=v)
    ap.add_argument("--no-check", action="store_true", help="skip patch_cumesh_win.py check (the fork against "
                    "upstream + cubvh + the hunks; --cubvh-src is only read by it)")
    ap.add_argument("--rev", default=FORK_PIN, help="the fork's commit to build (default: the pin, "
                    "patch_cumesh_win.PINS['fork']); a branch name builds its head (development)")
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
    commits = ensure_sources(a.src, a.eigen_src, a.rev)
    if a.work.exists():
        shutil.rmtree(a.work)
    (a.work / "src").mkdir(parents=True)
    (a.work / "tmp").mkdir()
    checked = None
    if not a.no_check:
        if patch.check(a.src, commits["fork"], a.cubvh_src, work=a.work / "tmp") != 0:
            sys.exit("the fork's tree is not upstream + cubvh + the hunks of patch_cumesh_win.py (see above); "
                     "--no-check builds it anyway")
        checked = (f"equal: upstream {patch.PINS['upstream'][1][:9]} + cubvh {patch.PINS['cubvh'][1][:9]} + "
                   f"{len(patch.HUNKS)} hunks, file by file (README.md aside); eigen's link at {EIGEN_PIN[:9]}")
    work = short_path(a.work)
    patch.export_tree(a.src, commits["fork"], a.work / "src")
    patch.export_tree(Path(commits["eigen_repo"]), commits["eigen"], a.work / "src" / patch.EIGEN_DIR)
    version = re.search(r'^version = "([^"]+)"$', (a.work / "src" / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    if not version or version.group(1) != VERSION:
        sys.exit(f"the fork's pyproject.toml says version {version and version.group(1)}, this script builds {VERSION}")
    print(f"sources: fork {commits['fork'][:9]} ({a.src}), eigen {commits['eigen'][:9]} ({commits['eigen_repo']}); "
          f"version {VERSION}; check: {checked or 'skipped'}", flush=True)
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
        "CUDA_HOME": cuda, "CUDA_PATH": cuda,
        "DISTUTILS_USE_SDK": "1", "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": a.arch_list, "MAX_JOBS": str(a.max_jobs),
        "PATH": os.pathsep.join([str(venv / "Scripts"), str(a.cuda / "bin"), env["PATH"]]),
        "TEMP": str(work / "tmp"), "TMP": str(work / "tmp"),   # nvcc/cl intermediates stay in the work dir
    })
    probe = run([py, "-c", "import torch, setuptools, wheel, sys; print(torch.__version__, torch.version.cuda, "
                           "setuptools.__version__, wheel.__version__, torch.__file__)"], env=env).stdout.split()
    torch_ver, torch_cuda, st_ver, wh_ver = probe[:4]
    nvcc_ver = re.search(r"release ([\d.]+), (V[\d.]+)", run([a.cuda / "bin" / "nvcc.exe", "--version"]).stdout)
    cl_banner = run([which("cl", env)], env=env, check=False).stderr.splitlines()[0]
    ninja_ver = run([which("ninja", env), "--version"], env=env).stdout.strip()
    print(f"torch {torch_ver} (CUDA {torch_cuda}) | setuptools {st_ver} | wheel {wh_ver} | ninja {ninja_ver}")
    print(f"nvcc {nvcc_ver.group(1)} {nvcc_ver.group(2)} | {cl_banner}")
    print(f"TORCH_CUDA_ARCH_LIST={a.arch_list} MAX_JOBS={a.max_jobs}", flush=True)

    wheel_out = a.work / "wheel-out"
    wheel_out.mkdir()
    log = a.log_dir / "cumesh-build.log"
    cmd = [py, "-m", "pip", "wheel", work / "src", "--no-deps", "--no-build-isolation",
           "--disable-pip-version-check", "-v", "-w", wheel_out]
    print("running:", " ".join(map(str, cmd)), flush=True)
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"# {' '.join(map(str, cmd))}\n# TORCH_CUDA_ARCH_LIST={a.arch_list} MAX_JOBS={a.max_jobs}\n")
        f.flush()
        rc = subprocess.run([str(c) for c in cmd], env=env, cwd=work, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        sys.exit(f"build failed (exit {rc}); log: {log}")
    wheels = list(wheel_out.glob("cumesh-*.whl"))
    if len(wheels) != 1 or not wheels[0].name.endswith("-cp311-cp311-win_amd64.whl"):
        sys.exit(f"expected one cp311 win_amd64 wheel, got {[w.name for w in wheels]}")
    wheel = a.out / wheels[0].name
    shutil.move(str(wheels[0]), wheel)

    with zipfile.ZipFile(wheel) as z:
        pyds = [n for n in z.namelist() if n.endswith(".pyd")]
        arches, imports = {}, {}
        for name in pyds:
            pyd_path = a.work / Path(name).name
            pyd_path.write_bytes(z.read(name))
            imports[Path(name).name] = pe_imports(z.read(name))
            if Path(name).name.startswith(("_C", "_cubvh")):
                arches[Path(name).name] = cuobjdump_arches(a.cuda / "bin" / "cuobjdump.exe", pyd_path)
        files = z.namelist()
    for name, found in arches.items():
        print(f"{name}: SASS sm {found['sass_sm']}  PTX compute {found['ptx_compute']}")

    prov = {
        "wheel": wheel.name,
        "sha256": sha256_file(wheel),
        "size_bytes": wheel.stat().st_size,
        "built": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "build_seconds": int((datetime.now(timezone.utc) - started).total_seconds()),
        "sources": {
            "fork": {"repo": FORK_URL, "branch": patch.FORK_BRANCH, "commit": commits["fork"],
                     "license": "MIT (Microsoft TRELLIS.2's CuMesh)"},
            "upstream": {"repo": patch.PINS["upstream"][0], "commit": patch.PINS["upstream"][1],
                         "how": "merged into the fork's branch"},
            "cubvh": {"repo": patch.PINS["cubvh"][0], "commit": patch.PINS["cubvh"][1],
                      "how": "vendored as plain files in third_party/cubvh (the commit upstream's submodule pins)",
                      "license": "MIT (ashawkey/cubvh lineage; NVIDIA instant-ngp's BVH code under its licence, "
                                 "LICENSE_NVIDIA in the tree)"},
            "eigen": {"repo": EIGEN_URL, "commit": commits["eigen"], "how": "the fork's submodule",
                      "license": "MPL-2.0 (headers only)"},
            "xatlas": {"where": "third_party/xatlas in CuMesh's tree", "license": "MIT"},
        },
        "patches": {
            "script": "code/release/wheels/cumesh/patch_cumesh_win.py",
            "applied_to": "nothing at build time: the fork's commit is built as committed; the hunks are its "
                          "commits' contents, and `patch_cumesh_win.py check` holds the commit equal to upstream + "
                          "cubvh + the hunks",
            "check": checked or "skipped (--no-check)",
            "why": f"the version ({VERSION}) and no torch pin; cuBVH's build without Python's lock; "
                   "cuBVH's card copy on demand (upload=False builds on the CPU only; upload(), release_device(), "
                   "on_device; the queries upload first); fill_holes' chain in less card memory with the same "
                   "tables bit for bit (get_edges: a DoubleBuffer sort, outputs of their exact length, its scratch "
                   "freed; get_boundary_info and get_manifold_boundary_adjacency: counting iterators and exact "
                   "outputs; fill_holes: the tables it no longer reads freed before the mesh buffers are extended), "
                   "on upstream's stream handling; the ray-stab's inside test with an any-hit traversal (the same "
                   "verdicts bit for bit); cuBVH's tree built on several threads (the same tree byte for byte); the "
                   "ray-stab's inside test for any subset of a call's points (raystab_inside: each point's rays "
                   "seeded from its index in the full query, the same verdicts bit for bit) and an any-hit ray_hits "
                   "query (equal to ray_trace's hit >= 0); xatlas's chart growing and task scheduler faster (the "
                   "same atlas byte for byte); Pixal3D's memory edits of remeshing.py (chunked BVH "
                   "queries and frees, value-neutral). Upstream's own Windows build (setup.py since f149d59) as it is",
            "hunks": [{"id": h["id"], "file": h["file"], "commit": h["commit"]} for h in patch.HUNKS],
            "structure": patch.STRUCTURE,
            "dropped_since_post3": patch.DROPPED,
        },
        "python": "CPython 3.11.9 (spellcast3d tools/python), throw-away build venv",
        "torch_at_build": f"{torch_ver} (CUDA {torch_cuda}) from {a.torch_venv.relative_to(ROOT) if a.torch_venv.is_relative_to(ROOT) else a.torch_venv} via a .pth (not installed/modified)",
        "msvc": f"{cl_banner} (toolset {env.get('VCToolsVersion')}), dynamic CRT /MD like torch",
        "windows_sdk": env.get("WindowsSDKVersion", "").strip("\\"),
        "cuda": f"CUDA {nvcc_ver.group(1)} (nvcc {nvcc_ver.group(2)}) at {a.cuda}",
        "torch_cuda_arch_list": a.arch_list,
        "cuobjdump": arches,
        "pyds": pyds,
        "pyd_dll_imports": imports,
        "build_tools": {"setuptools": st_ver, "wheel": wh_ver, "ninja": ninja_ver},
        "env": {"CUDA_HOME": cuda, "DISTUTILS_USE_SDK": "1", "FORCE_CUDA": "1",
                "TORCH_CUDA_ARCH_LIST": a.arch_list, "MAX_JOBS": str(a.max_jobs)},
        "build_command": "python -m pip wheel <patched export> --no-deps --no-build-isolation -v -w <out>"
                         "  (inside vcvars64; driver: code/release/wheels/cumesh/build_cumesh_win.py)",
        "runtime_requirements": {
            "install": "pip install <this wheel>  (Pixal3D's install.py installs every wheel in its whl folder)",
            "import_needs": ["torch 2.8.x+cu128", "numpy", "tqdm"],
        },
        "wheel_files": len(files),
        "wheel_license_files": [n for n in files if "/licenses/" in n or n.endswith(".dist-info/LICENSE")],
        "build_log": str(log.relative_to(ROOT)) if log.is_relative_to(ROOT) else str(log),
    }
    js = a.out / "cumesh.json"
    js.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    print(f"wheel {wheel} ({wheel.stat().st_size / 1e6:.1f} MB) sha256 {prov['sha256']}\nprovenance {js}")
    if not a.keep_work:
        shutil.rmtree(a.work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
