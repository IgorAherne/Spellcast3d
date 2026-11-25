"""Build nvdiffrast 0.4.0+spellcast3d.1 from the user's fork (https://github.com/IgorAherne/nvdiffrast, branch
updates-1, at 933724c: its csrc/, nvdiffrast/, setup.py and pyproject.toml are NVlabs/nvdiffrast's tag v0.4.0) as a
Windows cp311 wheel for torch 2.8.0+cu128, with CUDA 12.8 + MSVC 2022 and NO /arch for the host code, CUDA kernels for
sm_61..sm_90 + PTX, and write a provenance JSON.

    <any python 3.9+> build_nvdiffrast_win.py [--src REPO] [--out DIR] [--work DIR] [--keep-work] ...

Why it exists: the third-party wheel it replaces was built with /arch:AVX2, and its host code held BMI2 instructions
(shlx, sarx, bzhi), some run at import: "Illegal instruction" on CPUs with AVX but no AVX2 (Sandy / Ivy Bridge).
Without /arch, MSVC's x64 code needs SSE2 only.

What it does (nothing is installed into an existing venv; a repository given with --src is only read):
 1. Finds the pinned commit in --src (any clone of the fork). The default, release/cache/wheels/src/nvdiffrast, is made
    as a blob-less bare repository with the commit fetched at depth 1: the commit also holds ~600 MB of the fork's old
    build outputs (build/, dist/), and only the files the build reads come down (~80 KB), when the export reads them.
 2. Exports csrc/, nvdiffrast/, setup.py, pyproject.toml, LICENSE.txt, README.md and MANIFEST.in into <work>/src, the
    folders with `git archive`, the files with `git cat-file`, and sets the exported pyproject.toml's version to VERSION
    (the one source change). setup.cfg gets a [build] build_base on the 8.3 short path (build config only).
 3. Creates a throw-away build venv from --base-python with pinned build tools from the pip cache (setuptools, wheel,
    ninja) and a .pth that appends --torch-venv's site-packages, so torch 2.8.0+cu128 is importable without copying or
    modifying that venv.
 4. Runs `pip wheel --no-deps --no-build-isolation` inside the vcvars64 environment without CL, _CL_,
    NVCC_PREPEND_FLAGS and NVCC_APPEND_FLAGS (flags there reach the compilers without showing in the log), with
    CUDA_HOME=CUDA_PATH=CUDA 12.8, DISTUTILS_USE_SDK=1, TORCH_CUDA_ARCH_LIST, MAX_JOBS and CUDA_VISIBLE_DEVICES=-1 (the
    compile needs no card).
 5. Proves the wheel before it leaves the work folder, and stops at the first failure: no /arch in the log's compile
    commands (ninja prints every one), the source's LICENSE.txt in the wheel's dist-info, the SASS and PTX that
    cuobjdump lists exactly what TORCH_CUDA_ARCH_LIST asks for, and `prepare_release.py scan-cpu` on the wheel (exit 0:
    no binary needs AVX2 / FMA / BMI2).
 6. Reads the extension's DLL imports, hashes the wheel, moves it to --out and writes nvdiffrast.json next to it.

--out is release/cache/wheels/out/nvdiffrast by default, not code/pixal3d/whl: the shipped wheel was validated on real
cards, and a wheel of the same version would not even replace an installed copy (pip keeps it), so the script refuses to
write over a wheel of the same name outside release/cache. README.txt says how a new build replaces the shipped one;
smoke_nvdiffrast.py checks a built wheel on the card against the installed one.

About a minute with MAX_JOBS 8 (ten .cpp and five .cu files, seven architectures).
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

FORK_URL = "https://github.com/IgorAherne/nvdiffrast.git"
FORK_BRANCH = "updates-1"
PIN = "933724c7d81c58f0231cb5d5e8bc70024ac01b29"
UPSTREAM_URL = "https://github.com/NVlabs/nvdiffrast.git"
UPSTREAM_TAG = "v0.4.0"
UPSTREAM_COMMIT = "253ac4fcea7de5f396371124af597e6cc957bfae"
# The pinned commit's paths the build reads, each the same git object as at the upstream tag; the rest of the commit
# is the fork's old build outputs and a .gitignore.
EXPORT = ("csrc", "nvdiffrast", "setup.py", "pyproject.toml", "LICENSE.txt", "README.md", "MANIFEST.in")
SOURCE_VERSION = "0.4.0"
# A local version on top of upstream's: pip replaces an installed nvdiffrast only when the version differs (with the
# same version it answers "already installed" and keeps the old binary), and a constraint nvdiffrast==0.4.0 still
# accepts it. A build that changes the binary gets the next +spellcast3d.N.
VERSION = "0.4.0+spellcast3d.1"
TORCH = "2.8.0+cu128"
# The architectures of the wheel this one replaced.
ARCH_LIST = "6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX"
BUILD_PINS = ["setuptools==80.9.0", "wheel==0.45.1", "ninja==1.13.2"]
# Environment variables whose flags reach cl / nvcc without appearing on a command line in the log.
HIDDEN_FLAGS = ("CL", "_CL_", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # the Spellcast3D folder (this script is in code/release/wheels/<wheel>/)
SCRATCH = ROOT / "release" / "cache" / "wheels"  # clones, build trees and logs (release/ is git-ignored)
DEFAULTS = {
    "src": SCRATCH / "src" / "nvdiffrast",
    "work": SCRATCH / "build" / "nvdr",
    "out": SCRATCH / "out" / "nvdiffrast",
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


def git(repo: Path, *args: str, text: bool = True, env: dict | None = None):
    """git on the repository `repo` itself (--git-dir: a folder that is not a repository never falls through to the
    Spellcast3D checkout around release/cache)."""
    git_dir = repo / ".git" if (repo / ".git").exists() else repo
    cp = subprocess.run(["git", f"--git-dir={git_dir}", "-c", "core.autocrlf=false", *args], capture_output=True,
                        text=text, env=env)
    if cp.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} in {repo} failed: {cp.stderr if text else cp.stderr.decode()}")
    return cp.stdout


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


def expected_arches(arch_list: str) -> dict:
    """TORCH_CUDA_ARCH_LIST -> the SASS and PTX architectures torch's extension build compiles for it."""
    archs = [a.strip() for a in arch_list.replace(" ", ";").split(";") if a.strip()]
    number = lambda a: int(a.removesuffix("+PTX").replace(".", ""))  # noqa: E731
    return {"sass_sm": sorted({number(a) for a in archs}),
            "ptx_compute": sorted({number(a) for a in archs if a.endswith("+PTX")})}


def has_commit(repo: Path, rev: str) -> bool:
    """Whether `repo` holds the commit, without asking a promisor remote for it."""
    git_dir = repo / ".git" if (repo / ".git").exists() else repo
    return subprocess.run(["git", f"--git-dir={git_dir}", "cat-file", "-e", rev + "^{commit}"], capture_output=True,
                          env={**os.environ, "GIT_NO_LAZY_FETCH": "1"}).returncode == 0


def ensure_source(src: Path) -> str:
    """The pinned commit in `src` -> its full id. A folder that is not a repository becomes a blob-less bare one with
    the commit fetched at depth 1 (file contents come down when the export reads them); a clone that has the commit
    is only read."""
    if not ((src / "HEAD").is_file() or (src / ".git").exists()):
        src.mkdir(parents=True, exist_ok=True)
        run(["git", "init", "-q", "--bare", src])
        git(src, "remote", "add", "origin", FORK_URL)
        git(src, "config", "remote.origin.promisor", "true")
        git(src, "config", "remote.origin.partialclonefilter", "blob:none")
    if not has_commit(src, PIN):
        if git(src, "config", "--default", "", "--get", "remote.origin.promisor").strip() != "true":
            sys.exit(f"{src} does not have commit {PIN}: fetch it there, or leave --src at its default")
        print(f"the fork: fetching {PIN[:9]} from {FORK_URL} (depth 1, no file contents)", flush=True)
        git(src, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", PIN)
        git(src, "update-ref", "refs/spellcast3d/pin", PIN)    # a reference keeps the commit from being pruned
    full = git(src, "rev-parse", PIN + "^{commit}").strip()
    if full != PIN:
        sys.exit(f"{src} resolves {PIN} to {full}")
    return full


def export_tree(repo: Path, rev: str, dest: Path) -> None:
    """EXPORT of commit `rev` into dest: a folder with `git archive` of its own tree (an archive of the whole commit
    would fetch every blob of a blob-less repository first, the old build outputs included), a file with
    `git cat-file`."""
    entries = {}
    for line in git(repo, "ls-tree", rev, "--", *EXPORT).splitlines():
        meta, path = line.split("\t", 1)
        entries[path] = meta.split()[1]
    missing = [p for p in EXPORT if p not in entries]
    if missing:
        sys.exit(f"commit {rev[:9]} has no {', '.join(missing)}")
    dest.mkdir(parents=True, exist_ok=True)
    for path in EXPORT:
        if entries[path] == "tree":
            data = git(repo, "archive", "--format=tar", f"--prefix={path}/", f"{rev}:{path}", text=False)
            with tarfile.open(fileobj=io.BytesIO(data)) as t:
                t.extractall(dest)
        else:
            (dest / path).write_bytes(git(repo, "cat-file", "blob", f"{rev}:{path}", text=False))


def set_version(pyproject: Path) -> dict:
    """The exported pyproject.toml's version line made VERSION -> the change as the provenance records it."""
    old, new = f'version = "{SOURCE_VERSION}"', f'version = "{VERSION}"'
    line = re.compile(rb"^" + re.escape(old.encode()) + rb"(?=\r?$)", re.M)
    data = pyproject.read_bytes()
    if len(line.findall(data)) != 1:
        sys.exit(f"{pyproject} has no single line {old}")
    pyproject.write_bytes(line.sub(new.encode(), data))
    return {"id": "local-version", "file": "pyproject.toml", "from": old, "to": new}


def make_build_venv(base_python: Path, venv: Path, torch_venv: Path, env: dict) -> Path:
    run([base_python, "-m", "venv", venv], env=env)
    py = venv / "Scripts" / "python.exe"
    run([py, "-m", "pip", "install", "--disable-pip-version-check", "-q", *BUILD_PINS], env=env, capture_output=False)
    site = run([py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], env=env).stdout.strip()
    torch_site = torch_venv / "Lib" / "site-packages"
    # .pth entries are APPENDED to sys.path, so this venv's own setuptools/wheel win over torch_venv's.
    (Path(site) / "zz_torch_from_venv.pth").write_text(str(torch_site) + "\n", encoding="utf-8")
    return py


def host_compiles(log_text: str) -> dict:
    """The compile commands ninja printed into the build log: how many went to cl and to nvcc, the host compiler's
    flags among them (cl's own, and nvcc's -Xcompiler ones), and every /arch the log holds anywhere."""
    lines = [line for line in log_text.splitlines() if re.match(r"\s*\[\d+/\d+\] ", line)]   # pip indents them
    cl = [line for line in lines if re.match(r"\s*\[\d+/\d+\] cl(\.exe)? ", line)]
    nvcc = [line for line in lines if re.search(r"nvcc(\.exe)?\"? ", line)]
    skip = ("/Fo", "/I", "/D", "/wd")
    flags = {f for line in cl for f in re.findall(r"(?<=\s)/[A-Za-z][^\s\"]*", line) if not f.startswith(skip)}
    flags |= {f for line in nvcc for f in re.findall(r"-Xcompiler\s+(\S+)", line) if not f.startswith(skip)}
    return {"cl": len(cl), "nvcc": len(nvcc), "host_flags": sorted(flags),
            "arch_flags": sorted(set(re.findall(r"[/-]arch:\S+", log_text, re.I)))}


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
    wheel = a.out / f"nvdiffrast-{VERSION}-cp311-cp311-win_amd64.whl"
    if wheel.exists() and not wheel.is_relative_to(SCRATCH.parent):
        sys.exit(f"{wheel} exists: a different build under the same version would not replace an installed copy "
                 f"(pip keeps it); raise VERSION's +spellcast3d.N to replace that wheel")
    a.out.mkdir(parents=True, exist_ok=True)
    a.log_dir.mkdir(parents=True, exist_ok=True)
    commit = ensure_source(a.src)
    if a.work.exists():
        shutil.rmtree(a.work)
    (a.work / "src").mkdir(parents=True)
    (a.work / "tmp").mkdir()
    work = short_path(a.work)
    export_tree(a.src, commit, a.work / "src")
    patched = [set_version(a.work / "src" / "pyproject.toml")]
    print(f"sources: the fork {commit[:9]} ({a.src}): {', '.join(EXPORT)}; version {VERSION}", flush=True)
    # Build config only (not a source patch): keep setuptools' build tree on the short path.
    with open(a.work / "src" / "setup.cfg", "a", encoding="utf-8") as f:
        f.write(f"\n[build]\nbuild_base = {work / 'o'}\n")

    env = vcvars_env(a.vcvars)
    for k in ("PYTHONPATH", "PYTHONHOME", "PIP_REQUIRE_VIRTUALENV", "VIRTUAL_ENV", *HIDDEN_FLAGS):
        env.pop(k, None)
    venv = a.work / "venv"
    py = make_build_venv(a.base_python, venv, a.torch_venv, env)
    cuda = str(a.cuda)
    env.update({
        "CUDA_HOME": cuda, "CUDA_PATH": cuda,
        "DISTUTILS_USE_SDK": "1",
        "TORCH_CUDA_ARCH_LIST": a.arch_list, "MAX_JOBS": str(a.max_jobs),
        "CUDA_VISIBLE_DEVICES": "-1",
        "PATH": os.pathsep.join([str(venv / "Scripts"), str(a.cuda / "bin"), env["PATH"]]),
        "TEMP": str(work / "tmp"), "TMP": str(work / "tmp"),   # nvcc/cl intermediates stay in the work dir
    })
    probe = run([py, "-c", "import platform, torch, setuptools, wheel; print(torch.__version__, torch.version.cuda, "
                           "setuptools.__version__, wheel.__version__, platform.python_version())"], env=env)
    torch_ver, torch_cuda, st_ver, wh_ver, py_ver = probe.stdout.split()[:5]
    if torch_ver != TORCH:
        sys.exit(f"{a.torch_venv} has torch {torch_ver}; this wheel is built against {TORCH}")
    nvcc_ver = re.search(r"release ([\d.]+), (V[\d.]+)", run([a.cuda / "bin" / "nvcc.exe", "--version"]).stdout)
    cl_banner = run([which("cl", env)], env=env, check=False).stderr.splitlines()[0]
    ninja_ver = run([which("ninja", env), "--version"], env=env).stdout.strip()
    print(f"python {py_ver} | torch {torch_ver} (CUDA {torch_cuda}) | setuptools {st_ver} | wheel {wh_ver} | "
          f"ninja {ninja_ver}")
    print(f"nvcc {nvcc_ver.group(1)} {nvcc_ver.group(2)} | {cl_banner}")
    print(f"TORCH_CUDA_ARCH_LIST={a.arch_list} MAX_JOBS={a.max_jobs}; unset: {', '.join(HIDDEN_FLAGS)}", flush=True)

    wheel_out = a.work / "wheel-out"
    wheel_out.mkdir()
    log = a.log_dir / "nvdiffrast-build.log"
    cmd = [py, "-m", "pip", "wheel", work / "src", "--no-deps", "--no-build-isolation",
           "--disable-pip-version-check", "-v", "-w", wheel_out]
    print("running:", " ".join(map(str, cmd)), flush=True)
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"# {' '.join(map(str, cmd))}\n# TORCH_CUDA_ARCH_LIST={a.arch_list} MAX_JOBS={a.max_jobs}\n")
        f.flush()
        rc = subprocess.run([str(c) for c in cmd], env=env, cwd=work, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        sys.exit(f"build failed (exit {rc}); log: {log}")
    built = list(wheel_out.glob("nvdiffrast-*.whl"))
    if [w.name for w in built] != [wheel.name]:
        sys.exit(f"expected {wheel.name}, got {[w.name for w in built]}")
    built = built[0]

    compiles = host_compiles(log.read_text(encoding="utf-8", errors="replace"))
    if not compiles["cl"] or not compiles["nvcc"]:
        sys.exit(f"the build log holds {compiles['cl']} cl and {compiles['nvcc']} nvcc commands: without them the "
                 f"host flags cannot be checked; log: {log}")
    if compiles["arch_flags"]:
        sys.exit(f"the build log shows {', '.join(compiles['arch_flags'])}: the host code must be built without "
                 f"/arch; log: {log}")
    print(f"host compiles: {compiles['cl']} cl + {compiles['nvcc']} nvcc, no /arch; "
          f"flags {' '.join(compiles['host_flags'])}")

    licence = (a.work / "src" / "LICENSE.txt").read_bytes()
    with zipfile.ZipFile(built) as z:
        files = z.namelist()
        pyds = [n for n in files if n.endswith(".pyd")]
        licence_files = [n for n in files if re.search(r"\.dist-info/(licenses/)?LICENSE\.txt$", n)]
        licence_ok = any(z.read(n) == licence for n in licence_files)
        arches, imports, pyd_sha = {}, {}, {}
        for name in pyds:
            data = z.read(name)
            pyd_path = a.work / Path(name).name
            pyd_path.write_bytes(data)
            imports[name] = pe_imports(data)
            pyd_sha[name] = hashlib.sha256(data).hexdigest()
            arches[name] = cuobjdump_arches(a.cuda / "bin" / "cuobjdump.exe", pyd_path)
    if not licence_ok:
        sys.exit(f"{built.name} does not carry nvdiffrast's LICENSE.txt (found {licence_files})")
    if len(pyds) != 1:
        sys.exit(f"expected one extension in {built.name}, got {pyds}")
    want = expected_arches(a.arch_list)
    for name, found in arches.items():
        print(f"{name}: SASS sm {found['sass_sm']}  PTX compute {found['ptx_compute']}")
        if {k: found[k] for k in want} != want:
            sys.exit(f"{name}: cuobjdump lists SASS {found['sass_sm']} + PTX {found['ptx_compute']}, "
                     f"TORCH_CUDA_ARCH_LIST={a.arch_list} asks for SASS {want['sass_sm']} + PTX {want['ptx_compute']}")

    scan_cmd = [a.base_python, ROOT / "prepare_release.py", "scan-cpu", "--all", built]
    print("running:", " ".join(map(str, scan_cmd)), flush=True)
    scan = run(scan_cmd, check=False, cwd=ROOT)
    print((scan.stdout + scan.stderr).strip())
    if scan.returncode != 0:
        sys.exit(f"prepare_release.py scan-cpu {built.name} exited {scan.returncode}: the wheel needs more than an "
                 f"x64 baseline CPU (or the scan could not run)")
    # Its verdicts, each binary named inside the wheel (the work folder's path is gone once the build ends).
    verdicts = [re.sub(r"^.*\.whl!", "", line) for line in scan.stdout.splitlines() if line[:1].strip()]

    shutil.move(str(built), wheel)
    pyd = pyds[0]
    prov = {
        "wheel": wheel.name,
        "sha256": sha256_file(wheel),
        "size_bytes": wheel.stat().st_size,
        "built": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "build_seconds": int((datetime.now(timezone.utc) - started).total_seconds()),
        "sources": {
            "nvdiffrast": {
                "repo": FORK_URL, "branch": FORK_BRANCH, "commit": commit,
                "how": f"{', '.join(EXPORT)} exported (git archive / git cat-file), each the same git object as at "
                       f"upstream's tag {UPSTREAM_TAG}; the fork's three commits after it add build outputs and a "
                       f".gitignore only, which are not exported",
                "license": "NVIDIA Source Code License for nvdiffrast (non-commercial; LICENSE.txt in the wheel's "
                           "dist-info)",
            },
            "upstream": {"repo": UPSTREAM_URL, "tag": UPSTREAM_TAG, "commit": UPSTREAM_COMMIT},
        },
        "patches": {
            "script": "code/release/wheels/nvdiffrast/build_nvdiffrast_win.py",
            "applied_to": "the export (the source repository is only read)",
            "hunks": patched,
            "why": "the local version, so that pip replaces an installed 0.4.0; no code change",
            "build_config": "setup.cfg [build] build_base on the work folder's short path (not a source change)",
        },
        "why": "the wheel it replaces (nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl, a third party's, sha256 "
               "153c8c63c88ef7aab6ec5f48597c4c44d799693a1e80d09b9aaed5df495f145d) had host code built with "
               "/arch:AVX2: BMI2 shlx/shrx/sarx/bzhi, some run at import, so on a CPU with AVX but no AVX2 (Sandy / "
               "Ivy Bridge) Pixal3D's server died while starting. This build has no /arch (MSVC x64's SSE2 baseline); "
               "its GPU code is built for the same architectures",
        "version": f"{VERSION}: a new version, so that pip replaces an installed {SOURCE_VERSION} (with the same "
                   f"version it answers 'already installed' and keeps the old binary); a constraint "
                   f"nvdiffrast=={SOURCE_VERSION} still accepts it",
        "python": f"CPython {py_ver} (spellcast3d tools/python), throw-away build venv, pip wheel (setuptools' "
                  f"build_meta)",
        "torch_at_build": f"{torch_ver} (CUDA {torch_cuda}) from "
                          f"{a.torch_venv.relative_to(ROOT) if a.torch_venv.is_relative_to(ROOT) else a.torch_venv} "
                          f"via a .pth (not installed/modified)",
        "msvc": f"{cl_banner} (toolset {env.get('VCToolsVersion')}, vcvars64), host flags "
                f"{' '.join(compiles['host_flags'])}, no /arch; {', '.join(HIDDEN_FLAGS)} unset",
        "windows_sdk": env.get("WindowsSDKVersion", "").strip("\\"),
        "cuda": f"CUDA {nvcc_ver.group(1)} (nvcc {nvcc_ver.group(2)}) at {a.cuda}",
        "torch_cuda_arch_list": a.arch_list,
        "cuobjdump": arches,
        "pyds": pyds,
        "pyd_sha256": pyd_sha[pyd],
        "pyd_dll_imports": imports,
        "cpu_instructions": f"prepare_release.py scan-cpu --all, exit 0: {' '.join(verdicts)}",
        "checks": f"build: no /arch in the log's {compiles['cl']} cl and {compiles['nvcc']} nvcc commands; LICENSE.txt "
                  f"in the dist-info equal to the source's; cuobjdump's SASS and PTX as TORCH_CUDA_ARCH_LIST asks; "
                  f"prepare_release.py scan-cpu exit 0. On the card: code/release/wheels/nvdiffrast/"
                  f"smoke_nvdiffrast.py (not run by the build)",
        "build_tools": {"setuptools": st_ver, "wheel": wh_ver, "ninja": ninja_ver},
        "env": {"CUDA_HOME": cuda, "DISTUTILS_USE_SDK": "1", "TORCH_CUDA_ARCH_LIST": a.arch_list,
                "MAX_JOBS": str(a.max_jobs), "CUDA_VISIBLE_DEVICES": "-1"},
        "build_command": "python -m pip wheel <export> --no-deps --no-build-isolation -v -w <out>"
                         "  (inside vcvars64; driver: code/release/wheels/nvdiffrast/build_nvdiffrast_win.py)",
        "runtime_requirements": {
            "install": "pip install <this wheel>  (Pixal3D's install.py installs every wheel in its whl folder)",
            "import_needs": ["torch 2.8.x+cu128", "numpy"],
        },
        "wheel_files": len(files),
        "wheel_license_files": licence_files,
        "build_log": log.relative_to(ROOT).as_posix() if log.is_relative_to(ROOT) else str(log),
    }
    js = a.out / "nvdiffrast.json"
    js.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    print(f"wheel {wheel} ({wheel.stat().st_size / 1e6:.1f} MB) sha256 {prov['sha256']}\nprovenance {js}")
    if not a.keep_work:
        shutil.rmtree(a.work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
