"""Spellcast3D release helper. Developer-only, STDLIB ONLY, Windows only.

    python prepare_release.py fetch-tools  [--force]
        Download + SHA256-verify the portable Python (NuGet CPython 3.11.9, full layout)
        and MinGit into tools/, and copy the app-local VC++ runtime DLLs next to
        tools/python/python.exe. Pins live in code/release/manifest.json.
    python prepare_release.py build-wheel  [--python EXE] [--no-test]
        Build instant-meshes' cp311 wheel with a static CRT into code/whl/ (+ .json sidecar).
        (TriFlow's torchsparse and pyfqmr-triflow wheels: code/release/wheels/*/build_*_win.bat, see the
        README.txt there; `check` verifies them.)
    python prepare_release.py check
        Is the repo ready to release? Exits non-zero when it is not.
    python prepare_release.py scan-cpu     PATH [PATH ...] [--all]
        Would these binaries run on a CPU without AVX2? PATH: a .pyd / .dll / .exe, a wheel, or a folder of them
        (another project's release folder, a wheel just built). `check` asks the same of everything Spellcast3D
        ships and installs. Exits non-zero when one needs AVX2.
    python prepare_release.py build        [--from-worktree] [--allow-unpushed] [--out DIR] [--no-zip]
        Assemble release/staging/spellcast3d/ and zip it to release/spellcast3d-<version>-windows.zip.
    python prepare_release.py clean        [--yes] [--venvs] [--tools] [--submodules]
        Remove __pycache__, logs and staging from the dev tree. Dry-run unless --yes.

See DEVELOPERS.md ("Making a release").
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import fnmatch
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True      # importing spellcast.paths must not leave __pycache__ in the tree
sys.path.insert(0, str(ROOT / "code"))
from spellcast import paths  # noqa: E402  (stdlib only)
from spellcast.install import models as tool_models  # noqa: E402  (stdlib only)
from spellcast.install import sdcpp  # noqa: E402  (stdlib only)
from spellcast.install import submodules as nested  # noqa: E402  (stdlib only)

RELEASE = ROOT / "release"
CACHE = RELEASE / "cache"
STAGING = RELEASE / "staging" / "spellcast3d"
MANIFEST = paths.CODE / "release" / "manifest.json"
WHEEL_SIDECAR = paths.WHL / "instant_meshes_brush.json"
WHEEL_GLOB = "instant_meshes_brush-*.whl"
# Pinned build dependencies of the instant-meshes wheel (its pyproject only asks for minimum versions).
BUILD_CONSTRAINTS = paths.CODE / "release" / "build-constraints.txt"
IMESH_REQUIREMENTS = paths.PACKAGE / "install" / "requirements-imesh.txt"
WHEEL_TEST_PYTEST = "pytest==9.1.1"

GITHUB_FAIL_BYTES = 95 * 1024 * 1024    # GitHub rejects files over 100 MB; keep a margin
GITHUB_WARN_BYTES = 50 * 1024 * 1024    # ...and warns above 50 MB

# DLL name prefixes a static-CRT (/MT) extension must not import.
DYNAMIC_CRT_PREFIXES = ("msvcp1", "vcruntime1", "concrt1", "vcomp1", "api-ms-win-crt-")

# Untracked files the tools write into their own checkout when run from there.
KNOWN_SUBMODULE_JUNK: dict[str, tuple[str, ...]] = {
    "code/pixal3d": ("_tmp_preprocessed.png", "third_party/", "whl/torch_scatter", "whl/vox2seq",
                     "tmp/", "output.glb", "autotune_cache.json", "calibration.json"),
    "code/lato2": ("ckpt/", "_condtest/", "outputs/"),
}
# TriFlow (ADPNCL: a changed or added file must be documented and sent to AUDI AG) stays byte-identical to its
# commits, nested submodules included, with core.autocrlf=false in each of its repos (spellcast/install/submodules.py).
TRIFLOW = "code/triflow"
TRIFLOW_REPOS = (TRIFLOW, *(f"{TRIFLOW}/{d.relative_to(paths.TRIFLOW_DIR).as_posix()}"
                            for d in nested.triflow_nested_dirs()))
# Wheels built for TriFlow by the scripts in code/release/wheels/<name>/ (provenance sidecar: code/whl/<project>.json,
# as for instant_meshes_brush). They link to torch's DLLs, so unlike instant_meshes_brush they use the dynamic MSVC
# runtime, which torch and tools/python ship.
TRIFLOW_WHEELS = ("torchsparse-2.1.0+cu128torch2.8-cp311-cp311-win_amd64.whl",
                  "pyfqmr_triflow-0.5.0+triflow-cp311-cp311-win_amd64.whl")
# Instant Meshes' UV unwrap: xatlas-python 0.0.11 with xatlas's chart-merging pass fixed (this project's forks; its source
# is not a submodule here, so its sidecar's source_commit is not checked against a pin).
IMESH_WHEELS = ("xatlas-0.0.11+spellcast3d.1-cp311-cp311-win_amd64.whl",)
OWN_WHEELS = TRIFLOW_WHEELS + IMESH_WHEELS
WHEEL_BUILD_SCRIPTS = {"torchsparse": "code/release/wheels/torchsparse/build_torchsparse_win.bat",
                       "pyfqmr_triflow": "code/release/wheels/pyfqmr-triflow/build_pyfqmr_triflow_win.bat",
                       "xatlas": "code/release/wheels/xatlas/build_xatlas_win.py"}
# Tracked folders an artist's copy does without: the tests (code/spellcast/profiling stays: the [profiling] switches
# of the config file start it). The committed build checks them out of its clone with a sparse checkout, which
# update.bat's fetch + reset --keep keeps (git status stays clean, new files in them stay out); --from-worktree does
# not copy them.
RELEASE_LEAVES_OUT = ("code/tests/",)
SPARSE_PATTERNS = ("/*", *(f"!/{folder}" for folder in RELEASE_LEAVES_OUT))      # git sparse-checkout --no-cone
# The developers' own: their hand-over notes, their record of the work and their one-off tools (a folder ends with
# "/"). They lie beside the code and are never tracked - the repository is public, and a release keeps its .git, so
# a tracked file is in every copy whatever the checkout leaves out (check_files, verify_staging).
DEVELOPERS_OWN = ("SUMMARIES.md", "SUMMARIES_detailed.md", "code/ARCHITECTURE.md", "code/devtools/")


def wheel_sidecar(name: str) -> Path:
    """code/whl/<project>.json for a wheel file name."""
    return paths.WHL / f"{name.split('-')[0]}.json"

# Text files for people: a release must not ship a placeholder in them (check_files).
AUTHOR_TEXTS = ("README.md", "README - LICENSE.txt", "HOW TO USE - INSTRUCTIONS.txt",
                "run-browser/support & donations.txt")
# What "README - LICENSE.txt" must name (check_files), case aside: every tool a stage runs on (paths.STAGE_TOOLS;
# a test keeps the two together), and what TriFlow brings: its ADPNCL licence, MeshLib's non-commercial trial,
# TRELLIS (whose Inria files it loads), our wheels for it and their parts (code/whl), DINOv2 (LATO.2), and the
# Spellcast3D files that are MIT, not AGPL, because they run in TriFlow's process. Qwen-Image 2.1: its licence by
# name and the Notice its section 3.c asks for (Hangzhou Tongyi Laboratory's copyright line), its runtime
# stable-diffusion.cpp (MIT) and its Qwen3-VL text encoder (Apache-2.0). Hi3D (the 3D stage's second geometry): its
# licence by name, its owner (the NOTICE of its section 8.2), the credit its section 7 asks for, its Acceptable Use
# Policy (passed on to the artist) and the encoders it loads.
LICENSE_TOOL_NAMES = {"forge": "Forge Neo", "pixal3d": "Pixal3D", "imesh": "Instant Meshes", "lato2": "LATO.2",
                      "triflow": "TriFlow", "qwen": "Qwen-Image 2.1"}
LICENSE_MUST_NAME = (*LICENSE_TOOL_NAMES.values(), "ADPNCL", "MeshLib", "TRELLIS", "DINOv2", "torchsparse",
                     "sparsehash", "pyfqmr-triflow", "triflow_server.py", "triflow_stubs", "attn_shim.py", "spconv_shim.py",
                     "_loopback_guard.py", "Qwen RESEARCH LICENSE AGREEMENT", "Hangzhou Tongyi Laboratory",
                     "stable-diffusion.cpp", "Qwen3-VL", "Qwen-Image-2.1-Turbo", "FLUX Kommunity License",
                     "Black Forest Labs", "Hi3D Open License Agreement Version 1.0", "OCEANO HK HOLDING LIMITED",
                     "Powered by Hi3D 1.0", "Acceptable Use Policy", "CLIP", "dinov2-large")
# The artist's choices (spellcast/config.py): gitignored, so the build puts the template's text into the release
# under this name, ready before the first start downloads anything.
CONFIG_NAME = paths.CONFIG_FILE.name
CONFIG_TEMPLATE = paths.CONFIG_TEMPLATE.relative_to(ROOT).as_posix()

# Files a release must contain (besides the submodules and tools/).
RELEASE_MUST_HAVE = (
    "run-browser/run-spellcast3d.bat", "run-browser/support & donations.txt",
    "tools/environment.bat", "tools/spellcast-internal.bat", "tools/update.bat",
    "HOW TO USE - INSTRUCTIONS.txt", "README - LICENSE.txt", "LICENSE",
    "code/install.py", "code/requirements-hub.txt", "code/spellcast/__init__.py",
    "code/spellcast/defaults.json", "code/spellcast/static/index.html",
    "tools/python/python.exe", "tools/git/cmd/git.exe",
    CONFIG_NAME, CONFIG_TEMPLATE, "code/spellcast/install/requirements-qwen.txt",
)
# Hi3D's code (code/hi3d) is a submodule once .gitmodules names it: then a release must have it too (verify_staging).
HI3D_SUBMODULE = ("code/hi3d", tool_models.HI3D_CODE_LANDMARK)
SUBMODULE_LANDMARKS = {"code/forge": "launch.py", "code/pixal3d": "app.py",
                       "code/instant-meshes": "pyproject.toml", "code/lato2": "scripts/e2e_inference.py",
                       TRIFLOW: "inference.py",
                       **{f"{TRIFLOW}/{rel}": landmark for rel, landmark in nested.TRIFLOW_LANDMARKS.items()}}

# .gitignore sanity: (path, must it be ignored?). The tracked ones would be dropped
# silently by generic patterns such as lib/ or dist/ (git matches them case-insensitively here).
IGNORE_PROBES: tuple[tuple[str, bool], ...] = (
    ("tools/python/Lib/os.py", True), ("tools/git/cmd/git.exe", True),
    ("code/venvs/hub/pyvenv.cfg", True), ("code/logs/install.log", True), ("code/tmp/pip-unpack-1/x.whl", True),
    ("code/settings.json", True), ("projects/demo/project.json", True), ("exports/Demo/Barrel.fbx", True),
    ("release/cache/python.3.11.9.nupkg", True), ("code/spellcast/__pycache__/paths.cpython-311.pyc", True),
    *((name + "x.py" if name.endswith("/") else name, True) for name in DEVELOPERS_OWN),
    ("prepare_release.py", False), ("tools/environment.bat", False), ("tools/update.bat", False),
    ("tools/spellcast-internal.bat", False), ("run-browser/run-spellcast3d.bat", False),
    ("code/install.py", False), ("code/spellcast/paths.py", False),
    ("code/spellcast/install/venv.py", False), ("code/spellcast/static/lib/app.js", False),
    ("code/spellcast/static/vendor/three.module.js", False), ("code/release/manifest.json", False),
    ("code/whl/instant_meshes_brush-0.1.0-cp311-cp311-win_amd64.whl", False),
    ("code/whl/instant_meshes_brush.json", False), ("code/requirements-hub.txt", False),
    ("models/README.txt", False), ("models/checkpoints/.gitkeep", False), ("models/pixal3d/.gitkeep", False),
    ("models/checkpoints/sd_xl_base_1.0.safetensors", True), ("models/pixal3d/hub/weights.safetensors", True),
    ("models/notes.txt", True), ("models/lato2/.gitkeep", False), ("models/triflow/.gitkeep", False),
    ("models/lato2/vflow.pt", True), ("models/lato2/dinov2/checkpoints/dinov2_vitl14_reg4_pretrain.pth", True),
    ("models/triflow/flow_model.safetensors", True), ("models/triflow/triflow-weights.json", True),
    ("models/hi3d/.gitkeep", False), ("models/hi3d/Hi3Dv1.0/vae/model.safetensors", True),
    ("code/spellcast/install/requirements-lato2.txt", False),
    ("code/spellcast/install/requirements-triflow.txt", False),
    *((f"code/whl/{name}", False) for name in OWN_WHEELS),
    *((f"code/whl/{name.split('-')[0]}.json", False) for name in OWN_WHEELS),
    *((script, False) for script in WHEEL_BUILD_SCRIPTS.values()),
    ("code/release/wheels/torchsparse/sparsehash_msvc/sparsehash/internal/sparseconfig.h", False),
    ("release/cache/wheels/build/ts/src/setup.py", True),
    # Qwen-Image 2.1 and the config file that picks the image generator
    (CONFIG_NAME, True), (CONFIG_TEMPLATE, False), ("code/spellcast/config.py", False),
    ("code/spellcast/install/sdcpp.py", False), ("code/spellcast/install/requirements-qwen.txt", False),
    ("code/spellcast/launchers/qwen_server.py", False), ("code/spellcast/adapters/qwen.py", False),
    ("models/qwen-image/.gitkeep", False), ("models/qwen-image/qwen-image-2.1-Q4_K_M.gguf", True),
    ("models/qwen-image/qwen-image-2.1-turbo-Q4_K_M.gguf", True),
    ("models/qwen-image/qwen-image-2.1-Q4_K_M.gguf.part", True),
    ("models/qwen-image/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf.part", True),
    ("models/flux3-klein/flux-3-image-klein-UD-Q4_K_XL.gguf", True),
    ("models/flux3-klein/Qwen3VL-4B-Instruct-Q4_K_M.gguf.part", True),
    (f"code/venvs/downloads/{sdcpp.SDCPP.name}", True), ("code/venvs/qwen/sdcpp/stable-diffusion.dll", True),
)


class ReleaseError(Exception):
    """A failure whose message says what happened and what to do."""


class Report:
    """Prints check results and counts problems."""

    def __init__(self) -> None:
        self.errors = 0
        self.warnings = 0

    def section(self, title: str) -> None:
        print(f"\n{title}")

    def ok(self, msg: str) -> None:
        print(f"  [ OK ] {msg}")

    def info(self, msg: str) -> None:
        print(f"  [info] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings += 1
        print(f"  [WARN] {msg}")

    def fail(self, msg: str) -> None:
        self.errors += 1
        print(f"  [FAIL] {msg}")


# ----------------------------------------------------------------------------- helpers

def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def mb(n: float) -> str:
    return f"{n / 1_048_576:,.1f} MB"


def inside(path: str | Path, root: Path) -> bool:
    p = os.path.normcase(os.path.abspath(path))
    r = os.path.normcase(os.path.abspath(root)).rstrip("\\/")
    return p == r or p.startswith(r + os.sep)


def tool_env() -> dict[str, str]:
    """Environment for running tools/python, pip and git: scrubbed, no .pyc writes."""
    env = paths.clean_env()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run(cmd: list[Any], *, cwd: Path | None = None, env: dict[str, str] | None = None,
        check: bool = True, what: str = "") -> subprocess.CompletedProcess[str]:
    argv = [str(c) for c in cmd]
    try:
        cp = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    except OSError as e:
        raise ReleaseError(f"Could not start {argv[0]} ({e}).") from e
    if check and cp.returncode != 0:
        detail = (cp.stderr.strip() or cp.stdout.strip())[-3000:]
        raise ReleaseError(f"{what or 'A command'} failed (exit {cp.returncode}):\n  {' '.join(argv)}\n{detail}")
    return cp


def run_live(cmd: list[Any], *, what: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run with the output going straight to the console (for long git operations)."""
    argv = [str(c) for c in cmd]
    try:
        code = subprocess.run(argv, cwd=cwd, env=env).returncode
    except OSError as e:
        raise ReleaseError(f"Could not start {argv[0]} ({e}).") from e
    if code != 0:
        raise ReleaseError(f"{what} failed (exit {code}):\n  {' '.join(argv)}\nSee the output above.")


def run_tee(cmd: list[Any], log: Path, *, cwd: Path | None = None,
            env: dict[str, str] | None = None) -> tuple[int, str]:
    """Run, streaming the output to the console and to `log`. Returns (exit code, output)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with open(log, "w", encoding="utf-8") as f, subprocess.Popen(
            [str(c) for c in cmd], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace") as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
            lines.append(line)
    return proc.returncode, "".join(lines)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _retry_writable(func: Callable[..., Any], path: str, _exc: Any) -> None:
    os.chmod(path, stat.S_IWRITE)        # git marks its pack files read-only
    func(path)


def remove_tree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        os.chmod(path, stat.S_IWRITE)
        path.unlink()
    elif path.exists():
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_retry_writable)
        else:
            shutil.rmtree(path, onerror=_retry_writable)


def tree_size(path: Path) -> tuple[int, int]:
    """(bytes, files) under path."""
    if path.is_file():
        return path.stat().st_size, 1
    total = count = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
                count += 1
            except OSError:
                pass
    return total, count


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def read_version(root: Path) -> str:
    text = (root / "code" / "spellcast" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r"""__version__\s*=\s*["']([^"']+)["']""", text)
    if not m:
        raise ReleaseError(f"No __version__ in {root / 'code/spellcast/__init__.py'}.")
    return m.group(1)


# ----------------------------------------------------------------------------- git

def git_exe() -> str:
    found = shutil.which("git")
    if found:
        return found
    if paths.PORTABLE_GIT.exists():
        return str(paths.PORTABLE_GIT)
    raise ReleaseError("git was not found. Install Git for Windows (https://git-scm.com), "
                       "or run 'python prepare_release.py fetch-tools' so tools\\git exists.")


def git(repo: Path, *args: str, check: bool = True) -> str:
    return run([git_exe(), "-C", repo, *args], check=check, what=f"git {args[0]} in {rel(repo)}").stdout


def gitlinks(repo: Path, tree: str | None = None) -> dict[str, str]:
    """{path: pinned commit} for the submodule entries in the index, or in `tree` (e.g. "HEAD")."""
    if tree is None:
        out = git(repo, "ls-files", "-s", "-z")
    else:
        cp = run([git_exe(), "-C", repo, "ls-tree", "-r", "-z", tree], check=False)
        if cp.returncode != 0:
            return {}
        out = cp.stdout
    links: dict[str, str] = {}
    for record in out.split("\0"):
        meta, _, path = record.partition("\t")
        fields = meta.split()
        if fields and fields[0] == "160000":
            links[path] = fields[1] if tree is None else fields[2]
    return links


def submodules() -> list[tuple[str, str]]:
    """(path, url) of every submodule in .gitmodules."""
    out = git(ROOT, "config", "-f", ".gitmodules", "--get-regexp", r"^submodule\..*\.(path|url)$", check=False)
    entries: dict[str, dict[str, str]] = {}
    for line in out.splitlines():
        key, _, value = line.partition(" ")
        name, _, field = key[len("submodule."):].rpartition(".")
        entries.setdefault(name, {})[field] = value
    return [(e["path"], e.get("url", "")) for e in entries.values() if "path" in e]


def left_out(path: str) -> bool:
    """Is this repo-relative path in one of the folders the release leaves out (RELEASE_LEAVES_OUT)."""
    return any(path.startswith(folder) for folder in RELEASE_LEAVES_OUT)


def listed_files(repo: Path) -> list[str]:
    """Tracked + untracked-not-ignored files of `repo` that exist on disk. Submodules (gitlinks) excluded."""
    links = set(gitlinks(repo))
    seen: set[str] = set()
    files: list[str] = []
    for p in git(repo, "ls-files", "-co", "--exclude-standard", "-z").split("\0"):
        if p and p not in links and p not in seen and (repo / p).is_file():
            seen.add(p)
            files.append(p)
    return files


def tracked_files(repo: Path) -> list[str]:
    """The files git tracks in `repo` (no untracked ones, no submodules), as listed_files spells them."""
    links = set(gitlinks(repo))
    return [p for p in git(repo, "ls-files", "-c", "-z").split("\0") if p and p not in links and (repo / p).is_file()]


def on_remote_branch(repo: Path) -> tuple[bool | None, str]:
    """Is HEAD on origin's copy of the current branch (the branch tools\\update.bat follows)?

    Fetches first: the local origin/<branch> may be stale. (True, "") when HEAD is that branch's tip
    or an ancestor of it, (False, why) when it is not, (None, why) when origin cannot be asked.
    """
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    cp = run([git_exe(), "-C", repo, "fetch", "--quiet", "origin", branch], check=False)
    if cp.returncode != 0:
        return None, f"git fetch origin {branch} failed: " + (cp.stderr.strip().splitlines() or ["?"])[-1]
    cp = run([git_exe(), "-C", repo, "merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD"], check=False)
    if cp.returncode == 0:
        return True, ""
    head = git(repo, "rev-parse", "--short", "HEAD").strip()
    return False, (f"commit {head} is not on origin/{branch} on GitHub. Push it first (git push): users "
                   f"update with tools\\update.bat, which can only reach pushed commits. Never amend or "
                   f"force-push a commit you have released")


def remote_has_commit(repo: Path, sha: str) -> tuple[bool | None, str]:
    """(True, where) if origin has sha, (False, why) if it does not, (None, why) if unknown."""
    cp = run([git_exe(), "-C", repo, "ls-remote", "origin"], check=False)
    if cp.returncode != 0:
        return None, "could not reach origin: " + (cp.stderr.strip().splitlines() or ["?"])[-1]
    refs = [ref for ref_sha, _, ref in (line.partition("\t") for line in cp.stdout.splitlines()) if ref_sha == sha]
    if refs:
        return True, next((ref for ref in refs if ref.startswith("refs/heads/")), refs[0])
    cp = run([git_exe(), "-C", repo, "fetch", "--quiet", "origin"], check=False)
    if cp.returncode != 0:
        return None, "git fetch origin failed: " + (cp.stderr.strip().splitlines() or ["?"])[-1]
    cp = run([git_exe(), "-C", repo, "branch", "-r", "--contains", sha, "--list", "origin/*"], check=False)
    branches = cp.stdout.split()
    if branches:
        return True, "contained in " + ", ".join(branches)
    return False, "no branch on origin contains it"


# ----------------------------------------------------------------------------- PE / Windows

def pe_header(data: bytes) -> int:
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError("not a PE file")
    return pe


def pe_machine(data: bytes) -> int:
    return struct.unpack_from("<H", data, pe_header(data) + 4)[0]


def pe_imports(data: bytes) -> list[str]:
    """DLL names a PE file (exe/dll/pyd) imports directly (the same list as dumpbin /dependents)."""
    pe = pe_header(data)
    n_sections, = struct.unpack_from("<H", data, pe + 6)
    opt_size, = struct.unpack_from("<H", data, pe + 20)
    opt = pe + 24
    magic, = struct.unpack_from("<H", data, opt)
    import_rva, = struct.unpack_from("<I", data, opt + (120 if magic == 0x20B else 104))
    sections = []
    for i in range(n_sections):
        vsize, va, raw_size, raw_ptr = struct.unpack_from("<IIII", data, opt + opt_size + i * 40 + 8)
        sections.append((va, max(vsize, raw_size), raw_ptr))

    def offset(rva: int) -> int | None:
        for va, size, raw in sections:
            if va <= rva < va + size:
                return rva - va + raw
        return None

    names: list[str] = []
    p = offset(import_rva) if import_rva else None
    while p is not None and p + 20 <= len(data):
        name_rva, = struct.unpack_from("<I", data, p + 12)
        o = offset(name_rva) if name_rva else None
        if o is None:
            break
        names.append(data[o:data.index(b"\0", o)].decode("ascii", "replace"))
        p += 20
    return names


def wheel_binaries(wheel: Path) -> dict[str, list[str]]:
    """{.pyd/.dll inside the wheel: the DLLs it imports}."""
    with zipfile.ZipFile(wheel) as z:
        return {n: pe_imports(z.read(n)) for n in z.namelist() if n.lower().endswith((".pyd", ".dll"))}


def dynamic_crt(imports: list[str]) -> list[str]:
    return [d for d in imports if d.lower().startswith(DYNAMIC_CRT_PREFIXES)]


def file_version(path: Path) -> tuple[int, int, int, int] | None:
    """The FILEVERSION of a Windows binary, or None if it has none."""
    version = ctypes.WinDLL("version")
    size = version.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return None
    buf = ctypes.create_string_buffer(size)
    if not version.GetFileVersionInfoW(str(path), 0, size, buf):
        return None
    ptr, length = ctypes.c_void_p(), ctypes.c_uint()
    if not version.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)) or not ptr.value:
        return None
    fixed = (ctypes.c_uint32 * 4).from_address(ptr.value)    # dwSignature, dwStrucVersion, MS, LS
    ms, ls = fixed[2], fixed[3]
    return ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF


def fmt_version(v: tuple[int, ...] | None) -> str:
    return ".".join(map(str, v)) if v else "unknown version"


# ----------------------------------------------------------------------------- CPUs without AVX2

# Spellcast3D runs on CPUs that have AVX but no AVX2, FMA or BMI2 (Intel Sandy / Ivy Bridge). A binary built with
# MSVC's /arch:AVX2 stops there with "Illegal instruction", as a prebuilt nvdiffrast wheel did when Pixal3D started.
# Such a build shows in its disassembly by two things together: its everyday float code is VEX-encoded (vmovss,
# vaddsd where an SSE2 build has movss, addsd), and it holds what only that compiler target emits: BMI2 (shlx, sarx,
# bzhi), or FMA / AVX2 beyond the C runtime's own few routines, which ask the CPU before they run. A binary whose
# everyday code is SSE2 with AVX2 code beside it picks its path at run time (numpy, PyTorch, OpenCV): fine.
# The disassembly is Visual Studio's (dumpbin), so this needs its C++ tools.
RUNTIME_OWN = 40      # FMA or AVX2 instructions of the C runtime's CPU-checked routines in one binary, at most
# Binaries built for AVX2 that only ever run on a CPU that has it: path pattern -> who sees to that.
PICKED_BY_CPU = {
    "*/ggml-cpu-*.dll": "the image engine loads the variant its CPU supports (ggml-cpu-sandybridge.dll without AVX2)",
    "*/torch/*/fbgemm.dll": "PyTorch's quantized kernels: it uses them only on a CPU with AVX2",
    "*/numpy/_core/_simd.*.pyd": "numpy's test module of the intrinsics it dispatches at run time: nothing imports it",
    "*/embreex.libs/embree4*.dll": "Embree holds one build per instruction set and picks by the CPU",
    "*/Pillow_SIMD-*.whl!*": "never installed: the hub runs Pixal3D's install with --no-pillow-simd",
}
HOST_CODE_CACHE = CACHE / "host-code.json"        # what each build of a binary holds: disassembling all takes minutes

_DISASM = re.compile(r"^\s+[0-9A-F]{8,16}:\s(?:[0-9A-F]{2} )+\s*([a-z][a-z0-9]*)\s*(.*)$")
_FMA = re.compile(r"^vfn?m(add|sub)(sub|add)?(132|213|231)")
_SCALAR = ("movss movsd addss addsd subss subsd mulss mulsd divss divsd cvtsi2ss cvtsi2sd cvttss2si cvttsd2si cvtss2sd "
           "cvtsd2ss ucomiss ucomisd comiss comisd sqrtss sqrtsd maxss minss maxsd minsd").split()
_KIND = {
    **{op: "legacy" for op in _SCALAR}, **{"v" + op: "vex" for op in _SCALAR},
    **{op: "bmi2" for op in "bzhi mulx pdep pext rorx sarx shlx shrx".split()},
    **{op: "avx2" for op in (
        "vpbroadcastb vpbroadcastw vpbroadcastd vpbroadcastq vbroadcasti128 vinserti128 vextracti128 vperm2i128 "
        "vpermd vpermq vpermps vpermpd vpgatherdd vpgatherqd vpgatherdq vpgatherqq vgatherdps vgatherqps vgatherdpd "
        "vgatherqpd vpmaskmovd vpmaskmovq vpsllvd vpsllvq vpsrlvd vpsrlvq vpsravd vpblendd").split()},
}
_AVX1_ON_YMM = ("vpermilps", "vpermilpd", "vptest")       # "vp..." on ymm that AVX1 already has


def find_dumpbin() -> Path | None:
    """Visual Studio's disassembler (the x64 one of its newest C++ tools), or None without them."""
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None
    found = run([vswhere, "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-find", r"VC\Tools\MSVC\**\bin\Hostx64\x64\dumpbin.exe"], check=False).stdout.split("\n")
    tools = sorted(Path(line.strip()) for line in found if line.strip())
    return tools[-1] if tools else None


def count_host_code(disassembly: Iterable[str]) -> dict[str, int]:
    """What a binary's code holds, from the lines of its disassembly (dumpbin /disasm): {"legacy", "vex": scalar
    float instructions in the SSE2 and in the VEX encoding; "bmi2"; "fma"; "avx2": AVX2 and AVX-512}."""
    counts = dict.fromkeys(("legacy", "vex", "bmi2", "fma", "avx2"), 0)
    for line in disassembly:
        found = _DISASM.match(line)
        if not found:
            continue
        op, args = found.groups()
        kind = _KIND.get(op)
        if op[0] == "v":
            if "zmm" in args or "{k" in args:
                kind = "avx2"
            elif kind is None and _FMA.match(op):
                kind = "fma"
            elif kind is None and (op[1] == "p" and "ymm" in args and op not in _AVX1_ON_YMM
                                   or op == "vbroadcastss" and "ptr" not in args and "xmm" in args.split(",")[-1]):
                kind = "avx2"
        if kind:
            counts[kind] += 1
    return counts


def host_code(dumpbin: str, binary: str) -> dict[str, int]:
    """count_host_code of `binary` (.pyd / .dll / .exe), disassembled by `dumpbin`."""
    with subprocess.Popen([dumpbin, "/nologo", "/disasm", binary], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True, errors="replace", bufsize=1 << 20) as dump:
        return count_host_code(dump.stdout)         # type: ignore[arg-type]


def needs_avx2(counts: dict[str, int]) -> bool:
    """A binary with these counts (host_code) was built for a CPU with AVX2: one without it cannot run it."""
    return counts["vex"] > counts["legacy"] and (counts["bmi2"] > 0 or counts["fma"] > RUNTIME_OWN
                                                 or counts["avx2"] > RUNTIME_OWN)


def picked_by_cpu(name: str) -> str | None:
    """Why a binary built for AVX2 is no problem (PICKED_BY_CPU), or None: it is one."""
    return next((why for pattern, why in PICKED_BY_CPU.items() if fnmatch.fnmatch("/" + name, pattern)), None)


class Binary(NamedTuple):
    """A binary found: `build` tells one build of it from another without reading it whole (its name, size and the
    link time in its PE header; in a wheel, the checksum the wheel keeps of it), and where it is - `file`, or
    `member` of the wheel `file`."""
    build: str
    file: Path
    member: str | None = None


def binaries_in(sources: list[Path]) -> dict[str, Binary]:
    """{name to report: the binary} of every .pyd / .dll / .exe that is one of `sources`, lies in a folder of them,
    or in a wheel there ("<wheel>!<file in it>")."""
    found: dict[str, Binary] = {}
    for source in sources:
        files = [Path(folder, name) for folder, _, names in os.walk(source) for name in names
                 if name.lower().endswith((".pyd", ".dll", ".exe", ".whl"))] if source.is_dir() else [source]
        for file in files:
            kind, name = file.suffix.lower(), rel(file).replace("\\", "/")
            if kind in (".pyd", ".dll", ".exe"):
                with open(file, "rb") as f:
                    head = f.read(4096)
                try:
                    linked, = struct.unpack_from("<I", head, pe_header(head) + 8)
                except (ValueError, struct.error):
                    continue                                    # no Windows binary
                found[name] = Binary(f"{file.name}|{file.stat().st_size}|{linked:08x}", file)
            elif kind == ".whl" and zipfile.is_zipfile(file):       # (numpy leaves an empty file of that name)
                with zipfile.ZipFile(file) as wheel:
                    for info in wheel.infolist():
                        if info.filename.lower().endswith((".pyd", ".dll")):
                            build = f"{Path(info.filename).name}|{info.file_size}|crc {info.CRC:08x}"
                            found[f"{name}!{info.filename}"] = Binary(build, file, info.filename)
    return dict(sorted(found.items()))


def scan_host_code(dumpbin: Path, binaries: dict[str, Binary]) -> dict[str, dict[str, int]]:
    """{name: its counts (count_host_code)} for `binaries` (binaries_in). The same build in several places (PyTorch
    in each venv) is disassembled once, several builds at a time, and remembered from run to run
    (HOST_CODE_CACHE); a wheel is unpacked only for a build not seen before."""
    try:
        known = json.loads(HOST_CODE_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        known = {}
    new = {binary.build: binary for binary in binaries.values() if binary.build not in known}
    if new:
        print(f"         disassembling what was not seen before: {len(new)} (minutes when most are new) ...", flush=True)
        with tempfile.TemporaryDirectory(prefix="sc3d-host-code-") as scratch:
            files = []
            for i, binary in enumerate(new.values()):
                if binary.member:
                    with zipfile.ZipFile(binary.file) as wheel:
                        files.append(wheel.extract(binary.member, Path(scratch, str(i))))
                else:
                    files.append(str(binary.file))
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
                known.update(zip(new, pool.map(host_code, [str(dumpbin)] * len(files), files)))
        HOST_CODE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        HOST_CODE_CACHE.write_text(json.dumps(known, indent=0), encoding="utf-8")
    return {name: known[binary.build] for name, binary in binaries.items()}


def avx2_only(counted: dict[str, dict[str, int]]) -> tuple[dict[str, dict[str, int]], dict[str, str]]:
    """Of the binaries counted (scan_host_code): ({name: counts} of those a CPU without AVX2 would meet and cannot
    run, {name: why} of those built for AVX2 that such a CPU never runs)."""
    built_for = {name: counts for name, counts in counted.items() if needs_avx2(counts)}
    excused = {name: why for name in built_for if (why := picked_by_cpu(name))}
    return {name: counts for name, counts in built_for.items() if name not in excused}, excused


# ----------------------------------------------------------------------------- tool validation

VENV_PROBE = r"""
import ctypes, json, sys
import ssl, sqlite3, lzma, _ssl, _ctypes
k32 = ctypes.WinDLL("kernel32")
k32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
def module_path(handle):
    buf = ctypes.create_unicode_buffer(32768)
    k32.GetModuleFileNameW(handle, buf, len(buf))
    return buf.value
msvcp = ctypes.WinDLL("msvcp140.dll")
print(json.dumps({"sys_path": sys.path, "image": module_path(None), "msvcp140": module_path(msvcp._handle),
                  "_ssl": _ssl.__file__, "_ctypes": _ctypes.__file__}))
"""


def venv_problems(base_python: Path) -> list[str]:
    """Create a throw-away venv from base_python and check it is self-contained."""
    CACHE.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="venvcheck-", dir=CACHE))
    venv = work / "v"
    try:
        cp = run([base_python, "-m", "venv", "--without-pip", venv], check=False, env=tool_env())
        if cp.returncode != 0:
            return [f"'python -m venv' does not work with tools/python: {cp.stderr.strip()[-500:]}"]
        cp = run([venv / "Scripts" / "python.exe", "-c", VENV_PROBE], check=False, env=tool_env(), cwd=work)
        if cp.returncode != 0:
            return ["a venv made from tools/python cannot import ssl/ctypes/sqlite3/lzma or load "
                    f"msvcp140.dll: {cp.stderr.strip()[-500:]}"]
        info = json.loads(cp.stdout)
        problems: list[str] = []
        leaks = [p for p in info["sys_path"]
                 if p and not inside(p, paths.PORTABLE_PYTHON_DIR) and not inside(p, venv)]
        if leaks:
            problems.append("a venv made from tools/python sees folders outside it on sys.path "
                            f"(registry leak?): {', '.join(leaks)}")
        for key in ("image", "msvcp140", "_ssl", "_ctypes"):
            if not inside(info[key], paths.PORTABLE_PYTHON_DIR):
                problems.append(f"in a venv, {key} comes from {info[key]} instead of tools/python")
        return problems
    finally:
        remove_tree(work)


def python_problems(manifest: dict[str, Any], full: bool) -> list[str]:
    """What is wrong with tools/python ([] = nothing). `full` also imports modules and tests a venv."""
    py = paths.PORTABLE_PYTHON
    if not py.is_file():
        return ["tools/python/python.exe is missing - run: python prepare_release.py fetch-tools"]
    problems = [f"tools/python/{f} is missing (not a full CPython layout) - run: python prepare_release.py fetch-tools --force"
                for f in ("include/Python.h", "libs/python311.lib", "Lib/os.py",
                          "Lib/venv/__init__.py", "Lib/ensurepip/__init__.py")
                if not (paths.PORTABLE_PYTHON_DIR / f).is_file()]
    want = f"Python {manifest['python']['version']}"
    cp = run([py, "-I", "-B", "--version"], check=False, env=tool_env())
    got = (cp.stdout + cp.stderr).strip()
    if got != want:
        problems.append(f"tools/python reports {got!r}, expected {want!r} - run: python prepare_release.py fetch-tools --force")
    if not full or problems:
        return problems
    vc = manifest["vc_runtime"]
    problems += [f"tools/python/{dll} is missing (VC++ runtime) - run: python prepare_release.py fetch-tools"
                 for dll in vc["required"] if not (ROOT / vc["dest"] / dll).is_file()]
    cp = run([py, "-I", "-B", "-c", "import venv, ensurepip, ssl, ctypes, sqlite3, lzma"], check=False, env=tool_env())
    if cp.returncode != 0:
        problems.append(f"tools/python cannot import its standard modules: {cp.stderr.strip()[-500:]}")
    return problems or venv_problems(py)


def git_problems(manifest: dict[str, Any]) -> list[str]:
    exe = paths.PORTABLE_GIT
    if not exe.is_file():
        return ["tools/git/cmd/git.exe is missing - run: python prepare_release.py fetch-tools"]
    want = f"git version {manifest['git']['version']}"
    cp = run([exe, "--version"], check=False)
    got = (cp.stdout.strip() or cp.stderr.strip())
    if got != want:
        return [f"tools/git reports {got!r}, expected {want!r} - run: python prepare_release.py fetch-tools --force"]
    return []


# ----------------------------------------------------------------------------- fetch-tools

def download(url: str, dest: Path, size: int | None) -> None:
    """Download url to dest with a progress line; an interrupted download resumes from dest + '.part'."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if size and have > size:
        part.unlink()
        have = 0
    req = urllib.request.Request(url, headers={"User-Agent": "spellcast3d-prepare-release"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and size and have == size:
            os.replace(part, dest)
            return
        raise ReleaseError(f"Download failed: HTTP {e.code} for {url}. Check the URL in "
                           "code/release/manifest.json, or try again later.") from e
    except (urllib.error.URLError, OSError) as e:
        raise ReleaseError(f"Could not reach {url} ({e}). Check the internet connection and run "
                           "the command again; the download resumes where it stopped.") from e
    with resp:
        if have and resp.status != 206:     # the server ignored Range: start over
            have = 0
        total = size or (have + int(resp.headers.get("Content-Length") or 0)) or 0
        done, started, last = have, time.monotonic(), 0.0
        try:
            with open(part, "ab" if have else "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if now - last >= 0.5:
                        last = now
                        speed = (done - have) / max(now - started, 1e-6)
                        pct = f" ({done * 100 // total}%)" if total else ""
                        print(f"\r    {mb(done)} / {mb(total) if total else '?'}{pct}  {mb(speed)}/s   ",
                              end="", flush=True)
        except OSError as e:
            raise ReleaseError(f"The download of {url} was interrupted ({e}). Run the command again "
                               "to resume it.") from e
    print(f"\r    {mb(done)} downloaded{' ' * 30}")
    if size and done != size:
        raise ReleaseError(f"The download of {url} is incomplete ({done:,} of {size:,} bytes). "
                           "Run the command again to resume it.")
    os.replace(part, dest)


def fetch_archive(manifest: dict[str, Any], key: str) -> Path:
    """Download (or reuse from release/cache) and SHA256-verify manifest[key]."""
    entry = manifest[key]
    dest = CACHE / Path(urllib.parse.urlparse(entry["url"]).path).name
    CACHE.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"  downloading {entry['url']}")
        download(entry["url"], dest, entry.get("size"))
    if entry.get("size") and dest.stat().st_size != entry["size"]:
        dest.unlink()
        raise ReleaseError(f"{dest.name} has the wrong size (expected {entry['size']:,} bytes); it was "
                           "deleted. Run fetch-tools again.")
    digest = sha256_file(dest)
    if not entry.get("sha256"):
        entry["sha256"] = digest
        save_manifest(manifest)
        print(f"  [note] recorded sha256 {digest} for {dest.name} in code/release/manifest.json "
              "(first download) - review and commit it")
    elif digest != entry["sha256"]:
        dest.unlink()
        raise ReleaseError(
            f"{dest.name} does not match the SHA256 pinned in code/release/manifest.json "
            f"(got {digest}). The file was deleted; run fetch-tools again. If it fails again the file on "
            "the server has changed: do not update the pin unless you trust the new file.")
    else:
        print(f"  {dest.name}: SHA256 verified")
    return dest


def extract(archive: Path, subdir: str, percent_encoded: bool, dest: Path) -> None:
    """Extract the members of archive under `subdir` into dest, replacing dest only once it succeeded."""
    tmp = dest.with_name(dest.name + ".partial")
    remove_tree(tmp)
    tmp_root = tmp.resolve()
    count = 0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = urllib.parse.unquote(info.filename) if percent_encoded else info.filename
            name = name.replace("\\", "/")
            if not name.startswith(subdir) or name == subdir:
                continue
            target = (tmp / name[len(subdir):]).resolve()
            if not inside(target, tmp_root):
                raise ReleaseError(f"{archive.name} contains an unsafe path ({name}); refusing to extract it.")
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            count += 1
    if count == 0:
        raise ReleaseError(f"{archive.name} has no files under '{subdir}'. Is the manifest entry right?")
    try:
        remove_tree(dest)
    except OSError as e:
        raise ReleaseError(f"Could not replace {rel(dest)} ({e}). Close Spellcast3D and anything else that "
                           "uses it (a terminal inside it, a running venv...), then run again.") from e
    tmp.rename(dest)
    print(f"  extracted {count:,} files into {rel(dest)}")


def copy_vc_runtime(manifest: dict[str, Any]) -> None:
    vc = manifest["vc_runtime"]
    src_dir = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    dest_dir = ROOT / vc["dest"]
    missing: list[str] = []
    for dll in vc["required"] + vc["optional"]:
        src, dst = src_dir / dll, dest_dir / dll
        if not src.is_file():
            if dll in vc["required"]:
                missing.append(dll)
            else:
                print(f"  [WARN] {dll} is not in {src_dir}; skipped (optional)")
            continue
        with open(src, "rb") as f:
            if pe_machine(f.read(4096)) != 0x8664:
                raise ReleaseError(f"{src} is not a 64-bit DLL. Run prepare_release.py with a 64-bit Python.")
        new = file_version(src)
        old = file_version(dst) if dst.exists() else None
        if dst.exists() and (new is None or (old is not None and old >= new)):
            print(f"  {dll}: keeping {fmt_version(old)} (System32 has {fmt_version(new)})")
            continue
        shutil.copy2(src, dst)
        print(f"  {dll}: copied {fmt_version(new)}" + (f" (replaced {fmt_version(old)})" if old else ""))
    if missing:
        raise ReleaseError(
            f"{', '.join(missing)} not found in {src_dir}. Install the Microsoft Visual C++ 2015-2022 "
            "Redistributable (x64) from https://aka.ms/vs/17/release/vc_redist.x64.exe, then run fetch-tools again.")


def cmd_fetch_tools(args: argparse.Namespace) -> int:
    manifest = load_manifest()

    print("Portable Python (tools/python)")
    if not args.force and not python_problems(manifest, full=False):
        print(f"  already installed: Python {manifest['python']['version']} (use --force to reinstall)")
    else:
        entry = manifest["python"]
        archive = fetch_archive(manifest, "python")
        extract(archive, entry["archive_subdir"], entry["percent_encoded_names"], ROOT / entry["dest"])

    print("App-local VC++ runtime (next to tools/python/python.exe)")
    copy_vc_runtime(manifest)

    print("Portable git (tools/git)")
    if not args.force and not git_problems(manifest):
        print(f"  already installed: git {manifest['git']['version']} (use --force to reinstall)")
    else:
        entry = manifest["git"]
        archive = fetch_archive(manifest, "git")
        extract(archive, entry["archive_subdir"], entry["percent_encoded_names"], ROOT / entry["dest"])

    print("Validating (includes a throw-away venv)")
    problems = python_problems(manifest, full=True) + git_problems(manifest)
    if problems:
        raise ReleaseError("tools/ is not usable:\n  " + "\n  ".join(problems))
    py_size, py_files = tree_size(paths.PORTABLE_PYTHON_DIR)
    git_size, git_files = tree_size(paths.PORTABLE_GIT_DIR)
    print(f"  tools/python: Python {manifest['python']['version']}, {py_files:,} files, {mb(py_size)}")
    print(f"  tools/git:    git {manifest['git']['version']}, {git_files:,} files, {mb(git_size)}")
    print("Done: tools/ is ready.")
    return 0


# ----------------------------------------------------------------------------- build-wheel

BUILD_PY_PROBE = r"""
import json, os, struct, sys, sysconfig
v = sys.version_info
print(json.dumps({"version": f"{v[0]}.{v[1]}.{v[2]}", "bits": struct.calcsize("P") * 8,
    "headers": os.path.isfile(os.path.join(sysconfig.get_paths()["include"], "Python.h")),
    "lib": os.path.isfile(os.path.join(sys.base_prefix, "libs", f"python{v[0]}{v[1]}.lib"))}))
"""


def find_build_python(explicit: str | None) -> tuple[Path, str]:
    """A full 64-bit CPython 3.11 (headers + import lib). Returns (python.exe, its version)."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    else:
        candidates.append(paths.PORTABLE_PYTHON)
        if shutil.which("py"):
            cp = run(["py", "-3.11", "-c", "import sys; print(sys.executable)"], check=False)
            if cp.returncode == 0:
                candidates.append(Path(cp.stdout.strip()))
        if os.environ.get("LOCALAPPDATA"):
            candidates.append(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Python" / "Python311" / "python.exe")
    rejected: list[str] = []
    for exe in candidates:
        if not exe.is_file():
            continue
        cp = run([exe, "-I", "-B", "-c", BUILD_PY_PROBE], check=False, env=tool_env())
        if cp.returncode != 0:
            rejected.append(f"{exe}: does not run")
            continue
        info = json.loads(cp.stdout)
        why = []
        if not info["version"].startswith("3.11."):
            why.append(f"is Python {info['version']}")
        if info["bits"] != 64:
            why.append("is 32-bit")
        if not (info["headers"] and info["lib"]):
            why.append("has no include/Python.h or libs/python311.lib (embeddable build?)")
        if why:
            rejected.append(f"{exe}: " + ", ".join(why))
            continue
        return exe, info["version"]
    raise ReleaseError("No usable 64-bit CPython 3.11 with headers was found"
                       + (":\n  " + "\n  ".join(rejected) if rejected else ".")
                       + "\nRun 'python prepare_release.py fetch-tools' (tools/python has everything needed), "
                         "or pass --python <path to a full python 3.11 python.exe>.")


def find_msvc() -> dict[str, Any] | None:
    """The newest Visual Studio with the x64 C++ tools, via vswhere."""
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None
    cp = run([vswhere, "-latest", "-products", "*", "-requires",
              "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-format", "json", "-utf8"], check=False)
    try:
        found = json.loads(cp.stdout or "[]")
    except json.JSONDecodeError:
        return None
    if not found:
        return None
    install = Path(found[0]["installationPath"])
    toolset_file = install / "VC" / "Auxiliary" / "Build" / "Microsoft.VCToolsVersion.default.txt"
    cmake_bin = install / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "CMake" / "bin"
    return {
        "name": found[0].get("displayName", "Visual Studio"),
        "version": found[0].get("installationVersion", "?"),
        "toolset": toolset_file.read_text().strip() if toolset_file.is_file() else "?",
        "cmake_bin": cmake_bin if (cmake_bin / "cmake.exe").is_file() else None,
    }


def test_wheel(python: Path, wheel: Path, env: dict[str, str]) -> None:
    """Install the wheel into a scratch venv and run instant-meshes' core test suite against it."""
    venv = CACHE / "wheel-test-venv"
    remove_tree(venv)
    print(f"\nTesting {wheel.name} in a scratch venv ({rel(venv)})")
    run([python, "-m", "venv", venv], env=env, what="Creating the test venv")
    vpy = venv / "Scripts" / "python.exe"
    # The versions users get (components.Imesh: requirements-imesh.txt, then the wheel with --no-deps).
    print(f"  installing {rel(IMESH_REQUIREMENTS)} + {WHEEL_TEST_PYTEST}, then the wheel ...")
    run([vpy, "-m", "pip", "install", "--isolated", "--disable-pip-version-check", "-q",
         "-r", IMESH_REQUIREMENTS, WHEEL_TEST_PYTEST], env=env, what="Installing the wheel's dependencies for testing")
    run([vpy, "-m", "pip", "install", "--isolated", "--disable-pip-version-check", "-q", "--no-deps",
         "--force-reinstall", wheel], env=env, what="Installing the wheel for testing")
    cp = run([vpy, "-c", "import instant_meshes_brush as m; print(m.__version__, m.__file__)"],
             env=env, cwd=CACHE, what="Importing instant_meshes_brush")
    print(f"  import OK: instant_meshes_brush {cp.stdout.strip()}")
    tests = paths.IMESH_DIR / "tests"
    log = CACHE / "wheel-test.log"
    code, _ = run_tee([vpy, "-m", "pytest", tests, "--ignore", tests / "test_server.py", "-q",
                       "-p", "no:cacheprovider"], log, cwd=CACHE, env=env)
    if code != 0:
        raise ReleaseError(f"The wheel's tests failed (log: {rel(log)}); code/whl was not changed. "
                           "Fix instant-meshes, or use --no-test to keep the wheel for inspection.")
    remove_tree(venv)


def pinned_versions(path: Path) -> dict[str, str]:
    """{normalised name: version} of a requirements / constraints file's name==version lines."""
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, sep, version = line.split("#", 1)[0].strip().partition("==")
        if sep:
            pins[re.sub(r"[-_.]+", "-", name.strip()).lower()] = version.strip()
    return pins


def check_build_deps(output: str) -> dict[str, str]:
    """The versions pip's build environment installed (its "Successfully installed" line) - they must be the
    pinned ones of BUILD_CONSTRAINTS, whatever the pip version made of the constraints."""
    m = re.search(r"^\s*Successfully installed (.+)$", output, re.MULTILINE)
    if not m:
        raise ReleaseError("Could not find which build dependencies pip installed (no 'Successfully installed' "
                           "line in the build log).")
    used = {}
    for item in m.group(1).split():
        name, _, version = item.rpartition("-")
        used[re.sub(r"[-_.]+", "-", name).lower()] = version
    wanted = pinned_versions(BUILD_CONSTRAINTS)
    wrong = {k: v for k, v in used.items() if k in wanted and wanted[k] != v}
    unpinned = sorted(set(used) - set(wanted))
    if wrong or unpinned:
        raise ReleaseError(f"The wheel was built with other build dependencies than {rel(BUILD_CONSTRAINTS)} pins "
                           f"(different: {wrong or 'none'}; not pinned: {unpinned or 'none'}). Update that file "
                           f"on purpose, or find out why pip ignored it.")
    return dict(sorted(used.items()))


def cmd_build_wheel(args: argparse.Namespace) -> int:
    imesh = paths.IMESH_DIR
    if not (imesh / "pyproject.toml").is_file():
        raise ReleaseError("code/instant-meshes is not checked out. Run: git submodule update --init code/instant-meshes")
    for ext in ("dset", "pcg32"):
        if not (imesh / "ext" / ext / "README.md").is_file():
            raise ReleaseError(f"code/instant-meshes/ext/{ext} is empty. Run (never with --recursive):\n"
                               "  git -C code/instant-meshes submodule update --init ext/dset ext/pss ext/pcg32")
    if (imesh / "ext" / "nanogui" / "ext" / "eigen").exists() and not (imesh / "ext" / "eigen").exists():
        raise ReleaseError("code/instant-meshes/ext/nanogui is initialised: CMake would use its Eigen 3.2.9, which "
                           "does not compile with MSVC C++17. Run: git -C code/instant-meshes submodule deinit -f ext/nanogui")
    if git(imesh, "status", "--porcelain", "--untracked-files=no").strip():
        raise ReleaseError("code/instant-meshes has uncommitted changes, so the wheel would not match the commit "
                           "recorded in its sidecar. Commit (and push) them in instant-meshes-py first.")
    source_commit = git(imesh, "rev-parse", "HEAD").strip()

    python, py_version = find_build_python(args.python)
    msvc = find_msvc()
    if msvc is None:
        raise ReleaseError("Visual Studio with the C++ tools was not found. Install Visual Studio 2022 "
                           "Community with the 'Desktop development with C++' workload.")
    env = tool_env()
    if msvc["cmake_bin"]:
        env["PATH"] = f"{msvc['cmake_bin']}{os.pathsep}{env['PATH']}"
    # Pin pip's build environment (scikit-build-core, pybind11...): PIP_CONSTRAINT reaches it in the
    # bundled pip 24; pip 25.3+ uses PIP_BUILD_CONSTRAINT for it. The versions used are checked below.
    env["PIP_CONSTRAINT"] = env["PIP_BUILD_CONSTRAINT"] = str(BUILD_CONSTRAINTS)
    print(f"Python:  {python} ({py_version})")
    print(f"MSVC:    {msvc['name']} {msvc['version']}, toolset {msvc['toolset']}")
    print(f"CMake:   {msvc['cmake_bin'] or 'none from Visual Studio (pip will download one)'}")
    print(f"Source:  code/instant-meshes @ {source_commit[:12]}\n")

    out_dir, build_dir, log = CACHE / "wheel-out", CACHE / "imesh-build", CACHE / "build-wheel.log"
    remove_tree(out_dir)
    remove_tree(build_dir)
    out_dir.mkdir(parents=True)
    # build-dir outside the submodule: its pyproject default (build/{wheel_tag}) would write into code/instant-meshes.
    code, output = run_tee(
        [python, "-m", "pip", "wheel", imesh, "--no-deps", "--isolated", "--disable-pip-version-check", "-v",
         "-w", out_dir, "-C", "cmake.define.CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded", "-C", f"build-dir={build_dir}"],
        log, cwd=CACHE, env=env)
    if code != 0:
        raise ReleaseError(f"The wheel build failed (exit {code}). The full log is {rel(log)}.")
    built = sorted(out_dir.glob(WHEEL_GLOB))
    if len(built) != 1 or "-cp311-cp311-win_amd64" not in built[0].name:
        raise ReleaseError(f"Expected one cp311 win_amd64 wheel in {rel(out_dir)}, found: {[w.name for w in built]}")
    wheel = built[0]
    build_deps = check_build_deps(output)

    binaries = wheel_binaries(wheel)
    if not binaries:
        raise ReleaseError(f"{wheel.name} contains no compiled extension (.pyd); the build is broken.")
    for name, imports in binaries.items():
        if dynamic_crt(imports):
            raise ReleaseError(f"{name} in {wheel.name} imports {dynamic_crt(imports)}: it needs the VC++ "
                               "redistributable. Check that CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded reached CMake "
                               f"(log: {rel(log)}).")
        print(f"\n[ OK ] {name} imports only {', '.join(imports)} (static CRT)")

    if not args.no_test:
        test_wheel(python, wheel, env)

    paths.WHL.mkdir(parents=True, exist_ok=True)
    for old in paths.WHL.glob(WHEEL_GLOB):
        old.unlink()
        print(f"removed old {rel(old)}")
    final = paths.WHL / wheel.name
    shutil.move(str(wheel), final)
    compiler = re.search(r"The CXX compiler identification is (MSVC [\d.]+)", output)
    sidecar = {
        "wheel": final.name,
        "sha256": sha256_file(final),
        "source_commit": source_commit,
        "built": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "python": f"CPython {py_version} ({'tools/python' if inside(python, paths.PORTABLE_PYTHON_DIR) else 'system'})",
        "msvc": f"{compiler.group(1) if compiler else 'MSVC ?'} ({msvc['name']} {msvc['version']}, "
                f"toolset {msvc['toolset']}, static CRT /MT)",
        "build_deps": build_deps,
    }
    WHEEL_SIDECAR.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    remove_tree(build_dir)
    remove_tree(out_dir)

    leftovers = git(imesh, "status", "--porcelain").strip()
    if leftovers:
        print(f"[WARN] the build left files in code/instant-meshes:\n{leftovers}")
    print(f"\nDone: {rel(final)} ({mb(final.stat().st_size)}), sidecar {rel(WHEEL_SIDECAR)}")
    print("Commit both files so users get the new wheel.")
    return 0


# ----------------------------------------------------------------------------- check

def check_tools(r: Report, manifest: dict[str, Any]) -> None:
    r.section("Portable tools")
    problems = python_problems(manifest, full=True)
    for p in problems:
        r.fail(p)
    if not problems:
        r.ok(f"tools/python: Python {manifest['python']['version']}, full layout, venvs are self-contained, "
             "VC++ runtime app-local")
    problems = git_problems(manifest)
    for p in problems:
        r.fail(p)
    if not problems:
        r.ok(f"tools/git: git {manifest['git']['version']}")


def check_submodules(r: Report) -> dict[str, str]:
    r.section("Submodules")
    pinned = gitlinks(ROOT)
    committed = gitlinks(ROOT, "HEAD")
    heads: dict[str, str] = {}
    for path, url in submodules():
        sha = pinned.get(path)
        if not sha:
            r.fail(f"{path} is in .gitmodules but has no pinned commit. Run: git submodule add {url} {path}")
            continue
        if not url.startswith("https://"):
            r.fail(f"{path}: url {url!r} is not https - users have no SSH keys. Use the https:// URL in .gitmodules.")
        if committed.get(path) != sha:
            r.warn(f"{path}: pin {sha[:7]} is staged but not committed yet")
        sub = ROOT / path
        if not (sub / ".git").exists():
            r.fail(f"{path} is not checked out. Run: git submodule update --init {path}")
            continue
        head = git(sub, "rev-parse", "HEAD").strip()
        heads[path] = head
        if head != sha:
            r.fail(f"{path} is at {head[:7]} but the superproject pins {sha[:7]}. Either run "
                   f"'git submodule update {path}', or record the new commit with 'git add {path}' and commit.")
        changed = git(sub, "status", "--porcelain", "--untracked-files=no").strip()
        if changed:
            r.fail(f"{path} has uncommitted changes to tracked files - commit and push them in that repo, or revert:\n"
                   + "\n".join("         " + line for line in changed.splitlines()[:10]))
        untracked = [p for p in git(sub, "ls-files", "-o", "--exclude-standard", "--directory", "-z").split("\0") if p]
        if untracked:
            known = [p for p in untracked if p.startswith(KNOWN_SUBMODULE_JUNK.get(path, ()))]
            unknown = [p for p in untracked if p not in known]
            if known:
                r.warn(f"{path} has run-time leftovers from the tool itself ({', '.join(known[:6])}). They are not "
                       "released (ignore=untracked), but 'build --from-worktree' would copy them: delete them.")
            if unknown:
                r.warn(f"{path} has untracked files ({', '.join(unknown[:6])}{' ...' if len(unknown) > 6 else ''}). "
                       "Not released, but copied by 'build --from-worktree'.")
        if head == sha and not changed:
            found, where = remote_has_commit(sub, sha)
            if found:
                r.ok(f"{path} @ {sha[:7]}, clean, on origin ({where})")
            elif found is None:
                r.warn(f"{path} @ {sha[:7]}: could not verify it is on origin ({where})")
            else:
                r.fail(f"{path} @ {sha[:7]} is not on origin ({where}). Push that commit in the tool's repo, "
                       "otherwise users cannot download it.")
    nanogui = paths.IMESH_DIR / "ext" / "nanogui"
    if nanogui.is_dir() and any(nanogui.iterdir()):
        r.warn("code/instant-meshes/ext/nanogui is initialised; its Eigen 3.2.9 breaks build-wheel. "
               "Run: git -C code/instant-meshes submodule deinit -f ext/nanogui")
    else:
        r.info("code/instant-meshes/ext/nanogui is not initialised (correct: users get the prebuilt wheel)")
    return heads


def triflow_init_commands(triflow: str = TRIFLOW) -> list[str]:
    """The commands that check out TriFlow's nested submodules (spellcast/install/submodules.py), for messages."""
    return [f"git -c core.autocrlf=false -C {triflow}/{level}".rstrip("/") + " submodule update --init --depth 1 -- "
            + " ".join(subs) for level, subs in nested.TRIFLOW_NESTED]


def check_triflow(r: Report) -> None:
    r.section("TriFlow's files (they must stay byte-identical to its commits: ADPNCL licence)")
    root = ROOT / TRIFLOW
    if not (root / ".git").exists():
        r.fail(f"{TRIFLOW} is not checked out. Run: git -c core.autocrlf=false submodule update --init --depth 1 "
               f"-- {TRIFLOW}")
        return
    missing = nested.triflow_missing(root)
    if missing:
        r.fail(f"{TRIFLOW}: nested submodules not checked out ({', '.join(missing)}). Run, in this order (never "
               f"--recursive):\n" + "\n".join("         " + c for c in triflow_init_commands()))
    for repo in TRIFLOW_REPOS:
        if (ROOT / repo / ".git").exists():
            setting = run([git_exe(), "-C", ROOT / repo, "config", "--local", "core.autocrlf"],
                          check=False).stdout.strip()
            if setting != "false":
                r.fail(f"{repo} has core.autocrlf={setting or 'unset'}, so git may rewrite its line endings. Run: "
                       f"git -C {repo} config core.autocrlf false")
    problems = nested.byte_identity_problems(git_exe(), root, nested.TRIFLOW_ALLOWED_EXTRA, prefix=TRIFLOW + "/")
    if problems:
        r.fail("TriFlow's files are not exactly those of its commits (its licence: every change must be documented "
               "and sent to AUDI AG). Restore them (git checkout / git clean in that repo) or explain:\n"
               + "\n".join("         " + p for p in problems[:15]) + ("\n         ..." if len(problems) > 15 else ""))
    elif not missing:
        r.ok(f"{TRIFLOW} and its {len(TRIFLOW_REPOS) - 1} nested checkouts are byte-identical to their commits "
             f"(only the weights are linked into checkpoints/)")


def triflow_wheel_sources() -> dict[str, str | None]:
    """{wheel project: the commit TriFlow pins for its source}: pyfqmr-triflow is its third_party/pyfqmr-triflow
    submodule (the gitlink), torchsparse the commit its setup.sh installs (read from the unmodified file)."""
    try:
        setup = (ROOT / TRIFLOW / "setup.sh").read_text(encoding="utf-8", errors="replace")
    except OSError:
        setup = ""
    found = re.search(r"torchsparse\.git@([0-9a-f]{40})", setup)
    return {"pyfqmr_triflow": nested.pinned_commit(git_exe(), ROOT / TRIFLOW, "third_party/pyfqmr-triflow"),
            "torchsparse": found.group(1) if found else None}


def check_triflow_wheels(r: Report) -> None:
    r.section("Spellcast3D's wheels (code/whl)")
    sources = triflow_wheel_sources() if (ROOT / TRIFLOW / ".git").exists() else {}
    for name in OWN_WHEELS:
        wheel = paths.WHL / name
        sidecar = wheel_sidecar(name)
        script = WHEEL_BUILD_SCRIPTS[name.split("-")[0]]
        if not wheel.is_file():
            r.fail(f"code/whl/{name} is missing. Build it: {script} (see code/release/wheels/README.txt)")
            continue
        if not sidecar.is_file():
            r.fail(f"code/whl/{sidecar.name} (where the wheel was built from, and how) is missing.")
            continue
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if meta.get("wheel") != name:
            r.fail(f"{sidecar.name} describes {meta.get('wheel')}, not {name}: rebuild with {script}.")
        elif meta.get("sha256") != sha256_file(wheel):
            r.fail(f"{name} does not match the sha256 in {sidecar.name}: rebuild it, or record the wheel you meant.")
        elif not wheel_binaries(wheel):
            r.fail(f"{name} contains no compiled extension (.pyd); the build is broken.")
        elif name.split("-")[0] in sources and meta.get("source_commit") != sources.get(name.split("-")[0]):
            r.fail(f"{name} was built from {str(meta.get('source_commit'))[:7]}, but TriFlow now pins "
                   f"{str(sources.get(name.split('-')[0]))[:7]} for it: rebuild it with {script}.")
        else:
            r.ok(f"{name} ({mb(wheel.stat().st_size)}) matches its sidecar"
                 + (" and the source TriFlow pins" if name.split("-")[0] in sources else ""))
        for f in (wheel, sidecar):
            if run([git_exe(), "-C", ROOT, "ls-files", "--error-unmatch", f.relative_to(ROOT).as_posix()],
                   check=False).returncode != 0:
                r.warn(f"{rel(f)} is not committed yet (git add code/whl)")


def check_superproject(r: Report) -> None:
    r.section("Spellcast3D repo")
    status = git(ROOT, "status", "--porcelain").strip()
    if status:
        lines = status.splitlines()
        r.warn(f"{len(lines)} uncommitted change(s) - 'build' refuses until they are committed "
               "(or use 'build --from-worktree' to test):\n"
               + "\n".join("         " + line for line in lines[:15]) + ("\n         ..." if len(lines) > 15 else ""))
    else:
        r.ok("working tree is clean")
    if git(ROOT, "rev-parse", "--abbrev-ref", "HEAD").strip() == "HEAD":
        r.warn("HEAD is detached; 'build' needs the branch you release (e.g. git switch main)")
        return
    found, why = on_remote_branch(ROOT)
    if found:
        r.ok("the current commit is on GitHub")
    elif found is None:
        r.warn(f"could not check that the current commit is on GitHub ({why})")
    else:
        r.fail(why)


def check_wheel(r: Report, imesh_head: str | None) -> None:
    r.section("instant-meshes wheel (code/whl)")
    wheels = sorted(paths.WHL.glob(WHEEL_GLOB))
    if not WHEEL_SIDECAR.is_file():
        r.fail(f"{rel(WHEEL_SIDECAR)} is missing. Run: python prepare_release.py build-wheel")
        return
    meta = json.loads(WHEEL_SIDECAR.read_text(encoding="utf-8"))
    wheel = paths.WHL / meta.get("wheel", "?")
    if not wheel.is_file():
        r.fail(f"the sidecar names {meta.get('wheel')} but that file is not in code/whl. Run: python prepare_release.py build-wheel")
        return
    if len(wheels) > 1:
        r.warn(f"several wheels in code/whl ({', '.join(w.name for w in wheels)}); keep only {wheel.name}")
    if sha256_file(wheel) != meta.get("sha256"):
        r.fail(f"{wheel.name} does not match the sha256 in its sidecar. Run: python prepare_release.py build-wheel")
    pinned = gitlinks(ROOT).get("code/instant-meshes")
    if not (pinned and meta.get("source_commit") == imesh_head == pinned):
        r.fail(f"the wheel was built from instant-meshes {str(meta.get('source_commit'))[:7]}, but the submodule is at "
               f"{str(imesh_head)[:7]} (pinned {str(pinned)[:7]}). Run: python prepare_release.py build-wheel")
    else:
        r.ok(f"{wheel.name} matches instant-meshes {pinned[:7] if pinned else '?'} (built {meta.get('built')})")
    for name, imports in wheel_binaries(wheel).items():
        if dynamic_crt(imports):
            r.fail(f"{name} imports {dynamic_crt(imports)} (needs the VC++ redistributable). Rebuild with build-wheel.")
    if run([git_exe(), "-C", ROOT, "ls-files", "--error-unmatch", wheel.relative_to(ROOT).as_posix()],
           check=False).returncode != 0:
        r.warn(f"{rel(wheel)} is not committed yet (git add code/whl)")


def shipped_wheels() -> list[Path]:
    """The wheels in the release: Spellcast3D's own (code/whl) and those the tools' forks carry (code/<tool>/whl)."""
    return sorted(paths.CODE.glob("whl/*.whl")) + sorted(paths.CODE.glob("*/whl/*.whl"))


def check_old_cpus(r: Report) -> None:
    r.section("CPUs without AVX2 (Sandy / Ivy Bridge): no binary may need AVX2, FMA or BMI2")
    dumpbin = find_dumpbin()
    if not dumpbin:
        r.fail("Visual Studio 2022 with the C++ tools is not installed, so the binaries could not be disassembled "
               "(its dumpbin does that). Install it, or run 'check' on a PC that builds the wheels.")
        return
    # What pip and the downloads bring (PyTorch, flash_attn, the image engine...) is only on disk once installed.
    tools = sorted({paths.HUB, *(tool for stage in paths.STAGE_TOOLS.values() for tool in stage)})
    venvs = [paths.venv_dir(tool) for tool in tools if (paths.venv_dir(tool) / "pyvenv.cfg").is_file()]
    missing = [tool for tool in tools if paths.venv_dir(tool) not in venvs and tool != paths.FORGE]
    if missing:
        r.warn(f"not installed here: {', '.join(missing)}. What their installation brings was not checked: install "
               "them (run.bat), then check again.")
    wheels = shipped_wheels()
    counted = scan_host_code(dumpbin, binaries_in([*wheels, ROOT / "tools" / "python", *venvs]))
    fails, excused = avx2_only(counted)
    for name, counts in fails.items():
        r.fail(f"{name} is built for CPUs with AVX2 (its float code is VEX-encoded; BMI2 {counts['bmi2']}, FMA "
               f"{counts['fma']}, AVX2 {counts['avx2']}): on a CPU with AVX only it stops with \"Illegal "
               "instruction\". One of our wheels: rebuild it without /arch (code/release/wheels/README.txt). "
               "Somebody else's: pin a build without it, or build our own. If it only ever runs after a check of the "
               "CPU, add it to PICKED_BY_CPU and say who checks.")
    if not fails:
        r.ok(f"{len(counted)} binaries ({len(wheels)} shipped wheels, tools/python, installed: "
             f"{', '.join(v.name for v in venvs) or 'nothing'}): none needs AVX2"
             + (f"; {len(excused)} are built for it and picked by the CPU at run time" if excused else ""))


def check_files(r: Report) -> None:
    r.section("Files and .gitignore")
    too_big = False
    for path in listed_files(ROOT):
        size = (ROOT / path).stat().st_size
        if size > GITHUB_FAIL_BYTES:
            too_big = True
            r.fail(f"{path} is {mb(size)}: GitHub rejects files over 100 MB. Keep it out of git "
                   "(download it at install time, or attach it to the GitHub release).")
        elif size > GITHUB_WARN_BYTES:
            r.warn(f"{path} is {mb(size)} (GitHub warns above 50 MB)")
    if not too_big:
        r.ok("no tracked or to-be-added file is over 95 MB")
    if not (ROOT / "LICENSE").is_file():
        r.fail("LICENSE is missing: Spellcast3D's own code needs a license text with a copyright line "
               "(GitHub shows it; 'README - LICENSE.txt' points to it).")
    for name in AUTHOR_TEXTS:
        path = ROOT / name
        if path.is_file() and "to be confirmed" in path.read_text(encoding="utf-8", errors="replace").lower():
            r.fail(f"{name} still says 'to be confirmed': decide it before releasing.")
    check_license_text(r)
    for d in ("tools/python", "tools/git", "code/venvs", "code/logs", "projects", "exports", "release"):
        tracked = git(ROOT, "ls-files", "--", d).split()
        if tracked:
            r.fail(f"{d}/ is tracked by git ({len(tracked)} files) and must not be. Run: git rm -r --cached {d}")
    for name in DEVELOPERS_OWN:
        if git(ROOT, "ls-files", "--", name).strip():
            r.fail(f"{name} is tracked by git and must not be: the developers' own notes and tools are not "
                   f"published, and a release keeps its .git. Run: git rm -r --cached {name}")
    probes = [p for p, _ in IGNORE_PROBES]
    ignored = set(run([git_exe(), "-C", ROOT, "check-ignore", "--no-index", *probes], check=False).stdout.split("\n"))
    bad = [(p, want) for p, want in IGNORE_PROBES if (p in ignored) != want]
    for p, want in bad:
        r.fail(f".gitignore: {p} is {'NOT ' if want else ''}ignored but should {'be' if want else 'not be'} "
               f"(check with: git check-ignore --no-index -v {p})")
    if not bad:
        r.ok(f".gitignore: {len(IGNORE_PROBES)} probe paths behave (tools/python, venvs, projects ignored; "
             "hub code, .bat files and the wheel kept)")


def license_text_gaps(text: str) -> list[str]:
    """The names "README - LICENSE.txt" (its text) must mention and does not (LICENSE_MUST_NAME, case aside)."""
    low = text.lower()
    return [name for name in LICENSE_MUST_NAME if name.lower() not in low]


def check_license_text(r: Report) -> None:
    """A release tells what every tool and its parts are licensed under: TriFlow's licence (ADPNCL 3.1) asks that
    anybody who gets its source is told so, and triflow_server.py points to the MIT terms in that file."""
    path = ROOT / "README - LICENSE.txt"
    if not path.is_file():
        r.fail("'README - LICENSE.txt' is missing.")
        return
    gaps = license_text_gaps(path.read_text(encoding="utf-8", errors="replace"))
    if gaps:
        r.fail(f"'README - LICENSE.txt' does not mention {', '.join(gaps)}. Every tool, TriFlow's licence and parts "
               f"(MeshLib's trial, TRELLIS, the wheels in code/whl), the MIT files that run in TriFlow's process, and "
               f"Qwen-Image 2.1's licence with its Notice, runtime and text encoder must be named there before a "
               f"release.")
    else:
        r.ok(f"'README - LICENSE.txt' names every tool and what TriFlow and Qwen-Image 2.1 bring "
             f"({len(LICENSE_MUST_NAME)} names)")


def _get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "spellcast3d-release-check",
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_qwen_downloads(r: Report) -> None:
    """What the installer downloads for Qwen-Image 2.1 is still there exactly as pinned: stable-diffusion.cpp's two
    zips, each in its own release (GitHub's API: size and SHA-256) - the engine in Spellcast3D's own, on the author's
    fork, and NVIDIA's CUDA runtime in the official one - and every AI model file, all sizes (HuggingFace's API:
    size and LFS SHA-256 at the pinned commits; a small file kept as a plain git file there, such as the albedo
    add-on's LICENSE, by size: the installer checks every SHA-256 anyway). No network: a warning (check again online
    before releasing). A release GitHub does not have (HTTP 404: the engine's not published yet, or one taken down) is
    an error: every install would fail on that download."""
    r.section("Qwen-Image 2.1's pinned downloads")
    for archive in (sdcpp.SDCPP, sdcpp.CUDART):
        release = f"{archive.release.repo} {archive.release.tag}"
        try:
            assets = {a["name"]: a for a in _get_json(archive.release.api).get("assets", [])}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                r.fail(f"GitHub has no release {release}, where the installer downloads {archive.name}: publish it "
                       f"(with exactly that file), or pin another one in code/spellcast/install/sdcpp.py")
            else:
                r.warn(f"could not ask GitHub about {release} ({e}); check again online")
            continue
        except (urllib.error.URLError, OSError, ValueError, AttributeError) as e:
            r.warn(f"could not ask GitHub about {release} ({e}); check again online")
            continue
        asset = assets.get(archive.name)
        if asset is None:
            r.fail(f"{release} has no {archive.name}: upload it there, or pin another release in "
                   f"code/spellcast/install/sdcpp.py")
        elif asset.get("size") != archive.size or asset.get("digest", f"sha256:{archive.sha256}") \
                != f"sha256:{archive.sha256}":
            r.fail(f"{archive.name} of {release} is not the pinned file (size {asset.get('size')}, "
                   f"{asset.get('digest')}): check it and update code/spellcast/install/sdcpp.py")
        else:
            r.ok(f"{release}: {archive.name} as pinned ({mb(archive.size)})")
    # both Qwen-Image models (the Turbo and the full one) in every size, what they share, and FLUX.3 Image Klein's files
    # (the Images page's optional third model) in every size
    pinned = {**tool_models.QWEN_PINS, **tool_models.FLUX3_PINS}
    folders: dict[tuple[str, str, str], dict[str, str]] = {}    # (repo, commit, folder) -> {path there: our name}
    for name, (repo, revision, remote) in {**tool_models.QWEN_SOURCES, **tool_models.FLUX3_SOURCES}.items():
        folders.setdefault((repo, revision, remote.rpartition("/")[0]), {})[remote] = name
    for (repo, revision, folder), files in folders.items():
        url = f"https://huggingface.co/api/models/{repo}/tree/{revision}" + (f"/{folder}" if folder else "")
        try:
            tree = {e["path"]: e for e in _get_json(url) if isinstance(e, dict) and "path" in e}
        except (urllib.error.URLError, OSError, ValueError, TypeError) as e:
            r.warn(f"could not ask HuggingFace about {repo}@{revision[:7]} ({e}); check again online")
            continue
        bad = 0
        for remote, name in files.items():
            entry = tree.get(remote) or {}
            got = (entry.get("size"), (entry.get("lfs") or {}).get("oid"))
            if got[0] != pinned[name][0] or got[1] not in (None, pinned[name][1]):
                bad += 1
                r.fail(f"{repo}@{revision[:7]}: {remote} is not the pinned file (got size {got[0]}, SHA-256 {got[1]}): "
                       f"check code/spellcast/install/models.py")
        if not bad:
            r.ok(f"{repo}@{revision[:7]}: {len(files)} file(s) as pinned")


def check_hi3d_downloads(r: Report) -> None:
    """Hi3D's pinned files (install/models.py HI3D_FILES, installed with Pixal3D) are still on HuggingFace as pinned: the
    LFS files by size and SHA-256, the small ones (plain git files there, whose SHA-256 the API does not give) by size;
    the installer checks every SHA-256 anyway. No network: a warning."""
    r.section("Hi3D's pinned downloads")
    for folder, (repo, revision) in tool_models.HI3D_SOURCES_PINNED.items():
        files = {rel.split("/", 1)[1]: pin for rel, pin in tool_models.HI3D_FILES.items()
                 if rel.split("/", 1)[0] == folder}
        url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true"
        try:
            tree = {e["path"]: e for e in _get_json(url) if isinstance(e, dict) and "path" in e}
        except (urllib.error.URLError, OSError, ValueError, TypeError) as e:
            r.warn(f"could not ask HuggingFace about {repo}@{revision[:7]} ({e}); check again online")
            continue
        bad = 0
        for remote, (size, sha) in files.items():
            entry = tree.get(remote) or {}
            lfs = (entry.get("lfs") or {}).get("oid")
            if entry.get("size") != size or (lfs is not None and lfs != sha):
                bad += 1
                r.fail(f"{repo}@{revision[:7]}: {remote} is not the pinned file (got size {entry.get('size')}, SHA-256 "
                       f"{lfs}): check code/spellcast/install/models.py")
        if not bad:
            r.ok(f"{repo}@{revision[:7]}: {len(files)} file(s) as pinned")


def cmd_check(_args: argparse.Namespace) -> int:
    manifest = load_manifest()
    r = Report()
    check_tools(r, manifest)
    heads = check_submodules(r)
    check_superproject(r)
    check_wheel(r, heads.get("code/instant-meshes"))
    check_triflow(r)
    check_triflow_wheels(r)
    check_old_cpus(r)
    check_qwen_downloads(r)
    check_hi3d_downloads(r)
    check_files(r)
    print(f"\n{r.errors} error(s), {r.warnings} warning(s).")
    if r.errors:
        print("NOT ready to release: fix the [FAIL] lines above.")
        return 1
    if git(ROOT, "status", "--porcelain").strip():
        print("No errors, but commit (and push) your changes first: 'build' only packs committed work.")
    else:
        print("Ready to release." if not r.warnings else "Ready to release; read the [WARN] lines above.")
    return 0


def cmd_scan_cpu(args: argparse.Namespace) -> int:
    dumpbin = find_dumpbin()
    if not dumpbin:
        raise ReleaseError("Visual Studio 2022 with the C++ tools is needed: its dumpbin disassembles the binaries.")
    sources = [Path(p).resolve() for p in args.paths]
    for source in sources:
        if not source.exists():
            raise ReleaseError(f"{source} does not exist.")
    counted = scan_host_code(dumpbin, binaries_in(sources))
    fails, excused = avx2_only(counted)
    for name, counts in counted.items():
        if name in fails or name in excused or args.all:
            verdict = ("NEEDS AVX2" if name in fails else f"built for AVX2, picked by the CPU ({excused[name]})"
                       if name in excused else "runs without AVX2")
            print(f"{name}: {verdict}  [float code SSE2 / VEX {counts['legacy']} / {counts['vex']}, BMI2 "
                  f"{counts['bmi2']}, FMA {counts['fma']}, AVX2 {counts['avx2']}]")
    print(f"Binaries: {len(counted)}. Need AVX2 (a CPU with AVX only stops with \"Illegal instruction\"): {len(fails)}."
          + (f" Built for it and picked by the CPU at run time: {len(excused)}." if excused else ""))
    return 1 if fails else 0


# ----------------------------------------------------------------------------- build

def prepare_staging_dir(staging: Path) -> None:
    if inside(ROOT, staging) or (inside(staging, ROOT) and not inside(staging, RELEASE)):
        raise ReleaseError(f"--out {staging} overlaps the repo. Use a folder outside it, or release/.")
    if staging.exists():
        if any(staging.iterdir()) and not (staging / "code" / "spellcast").is_dir():
            raise ReleaseError(f"{staging} exists and does not look like an old Spellcast3D staging folder; "
                               "refusing to delete it. Choose an empty or new --out folder.")
        print(f"removing the previous staging folder {staging}")
        remove_tree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)


def stage_committed(staging: Path, allow_unpushed: bool) -> None:
    status = git(ROOT, "status", "--porcelain").strip()
    if status:
        lines = status.splitlines()
        raise ReleaseError("The repo has uncommitted changes, and a release is built from what is committed:\n"
                           + "\n".join("  " + line for line in lines[:15]) + ("\n  ..." if len(lines) > 15 else "")
                           + "\nCommit them first, or use --from-worktree to test your uncommitted work.")
    branch = git(ROOT, "rev-parse", "--abbrev-ref", "HEAD").strip()
    if branch == "HEAD":
        raise ReleaseError("HEAD is detached. Check out the branch you want to release (e.g. git switch main).")
    origin = git(ROOT, "remote", "get-url", "origin", check=False).strip()
    if not origin.startswith("https://"):
        raise ReleaseError(f"The repo's origin ({origin or 'none'}) is not an https URL; users update from it. "
                           "Run: git remote set-url origin https://github.com/<you>/spellcast3d.git")
    found, why = on_remote_branch(ROOT)
    if not found:
        if not allow_unpushed:
            raise ReleaseError((why if found is False else f"Could not check that the commit is on GitHub ({why})")
                               + ". (For a test build of an unpushed commit: build --allow-unpushed.)")
        print(f"[WARN] {why}: this zip is for testing only.")
    pinned = gitlinks(ROOT, "HEAD")
    if not pinned:
        raise ReleaseError("The last commit has no submodules (code/forge, code/pixal3d, code/hi3d, "
                           "code/instant-meshes, code/lato2, code/triflow). Commit .gitmodules and the submodule "
                           "pins first.")

    # Line endings: the release is checked out with CRLF text files and keeps core.autocrlf=true in its
    # own .git config (the root's and every submodule's). Otherwise a user whose Git config says
    # autocrlf=false would see every file as edited, and update.bat could never update.
    # Except TriFlow and its nested repos: byte-identical to their commits, with core.autocrlf=false (and so
    # unedited under any user config too).
    print(f"Cloning the committed state of '{branch}' (depth 1) into {staging}")
    run_live([git_exe(), "clone", "--depth", "1", "--config", "core.autocrlf=true", "--branch", branch,
              ROOT.as_uri(), staging], what="git clone")
    print("Fetching the pinned submodule commits from their GitHub repos (depth 1)")
    others = [path for path in pinned if path != TRIFLOW]
    if others:      # (no paths after "--" would mean all of them)
        run_live([git_exe(), "-c", "core.autocrlf=true", "-C", staging, "submodule", "update", "--init", "--depth",
                  "1", "--", *others], what="git submodule update")
    for path in others:
        run([git_exe(), "-C", staging / path, "config", "core.autocrlf", "true"])
    if TRIFLOW in pinned:
        print("Fetching TriFlow and its nested submodules level by level (core.autocrlf=false, depth 1)")
        run_live([git_exe(), "-c", "core.autocrlf=false", "-C", staging, "submodule", "update", "--init", "--depth",
                  "1", "--", TRIFLOW], what="git submodule update (TriFlow)")
        nested.init_triflow(git_exe(), lambda argv, cwd: run_live(argv, what="git (TriFlow's nested submodules)"),
                            staging / TRIFLOW)
    run([git_exe(), "-C", staging, "remote", "set-url", "origin", origin])
    print(f"Leaving {', '.join(RELEASE_LEAVES_OUT)} out of the working tree (sparse checkout; still in git)")
    run([git_exe(), "-C", staging, "sparse-checkout", "set", "--no-cone", *SPARSE_PATTERNS],
        what="Leaving the tests and the developers' tools out (git sparse-checkout)")
    for path, sha in pinned.items():
        got = git(staging / path, "rev-parse", "HEAD").strip()
        if got != sha:
            raise ReleaseError(f"The staged {path} is at {got[:7]}, expected {sha[:7]}. Delete {staging} and try again.")
    print(f"  origin -> {origin}; submodules at their pinned commits")


def stage_worktree(staging: Path) -> None:
    print(f"Copying the working tree (tracked + untracked, not ignored; no .git) into {staging}")
    total = 0
    jobs: list[tuple[Path, str]] = [(ROOT, p) for p in listed_files(ROOT) if not left_out(p)]
    for sub_path in gitlinks(ROOT):
        sub = ROOT / sub_path
        if not (sub / ".git").exists():
            raise ReleaseError(f"{sub_path} is not checked out. Run: git submodule update --init {sub_path}")
        jobs += [(sub, p) for p in listed_files(sub)]
    # The only nested submodules a release has (the committed build checks them out too).
    for sub_path in TRIFLOW_REPOS[1:] if TRIFLOW in gitlinks(ROOT) else ():
        sub = ROOT / sub_path
        if not (sub / ".git").exists():
            raise ReleaseError(f"{sub_path} is not checked out. Run, in this order:\n  "
                               + "\n  ".join(triflow_init_commands()))
        jobs += [(sub, p) for p in listed_files(sub)]
    if TRIFLOW in gitlinks(ROOT):
        # A changed or added file in TriFlow's tree would ship a modified TriFlow (its licence: every change must be
        # documented and sent to AUDI AG). A staging folder has no .git, so this is checked here, and only TriFlow's
        # tracked files are copied (the weights linked into checkpoints/ are not).
        problems = nested.byte_identity_problems(git_exe(), ROOT / TRIFLOW, nested.TRIFLOW_ALLOWED_EXTRA,
                                                 prefix=TRIFLOW + "/")
        if problems:
            raise ReleaseError("TriFlow's files are not exactly those of its commits, so they cannot go into a "
                               "release (not even a test one):\n  " + "\n  ".join(problems[:15])
                               + ("\n  ..." if len(problems) > 15 else ""))
        tracked = {(ROOT / repo).resolve() for repo in TRIFLOW_REPOS}
        jobs = [(repo, p) for repo, p in jobs if repo.resolve() not in tracked]
        jobs += [(ROOT / repo, p) for repo in TRIFLOW_REPOS for p in tracked_files(ROOT / repo)]
    for repo, p in jobs:
        src = repo / p
        dst = staging / src.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        total += src.stat().st_size
    print(f"  {len(jobs):,} files, {mb(total)}")


def stage_config(staging: Path) -> None:
    """spellcast3d-config.ini in the release's root: the staged template's text (git ignores the file itself), so the
    artist can choose the image generator before the first start downloads anything."""
    template = staging / CONFIG_TEMPLATE
    if template.is_file():
        print(f"Writing {CONFIG_NAME} (from {CONFIG_TEMPLATE})")
        shutil.copyfile(template, staging / CONFIG_NAME)


def git_problems_in_release(staging: Path) -> list[str]:
    """A committed staging folder must look unedited to git on any PC: local core.autocrlf=true in every
    repo, and `git status` clean even under a user Git config that says autocrlf=false."""
    problems = []
    repos = [Path(".")] + [Path(p) for p in gitlinks(staging, "HEAD")]
    repos += [Path(p) for p in TRIFLOW_REPOS[1:] if (staging / p / ".git").exists()]
    with tempfile.TemporaryDirectory() as tmp:
        user_config = Path(tmp) / "gitconfig"
        user_config.write_text("[core]\n\tautocrlf = false\n", encoding="utf-8")
        env = {**tool_env(), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": str(user_config)}
        for repo in repos:
            where = staging / repo
            name = "the root" if repo == Path(".") else repo.as_posix()
            want = "false" if name in TRIFLOW_REPOS else "true"
            setting = run([git_exe(), "-C", where, "config", "--local", "core.autocrlf"], check=False).stdout.strip()
            if setting != want:
                problems.append(f"{name} has no local core.autocrlf={want} (it has {setting or 'none'})")
            status = run([git_exe(), "-C", where, "status", "--porcelain", "--ignore-submodules=all"],
                         env=env, check=False).stdout.strip()
            if status:
                problems.append(f"{name} looks edited under autocrlf=false: {status.splitlines()[0]} ...")
    return problems


def verify_staging(staging: Path, strict: bool) -> None:
    missing = [f for f in RELEASE_MUST_HAVE if not (staging / f).is_file()]
    landmarks = {**SUBMODULE_LANDMARKS, **dict([HI3D_SUBMODULE] if HI3D_SUBMODULE[0] in dict(submodules()) else [])}
    missing += [f"{sub}/{landmark}" for sub, landmark in landmarks.items()
                if not (staging / sub / landmark).is_file()]
    sidecar = staging / WHEEL_SIDECAR.relative_to(ROOT)
    if not sidecar.is_file():
        missing.append(rel(WHEEL_SIDECAR))
    elif not (staging / "code" / "whl" / json.loads(sidecar.read_text(encoding="utf-8"))["wheel"]).is_file():
        missing.append("code/whl/<the wheel named in the sidecar>")
    junk = [d for d in ("code/venvs", "code/logs", "code/tmp", "projects", "exports", "code/settings.json",
                        *RELEASE_LEAVES_OUT, *DEVELOPERS_OWN) if (staging / d).exists()]
    if (staging / ".git").is_dir():                 # ... nor in the commit it holds
        junk += [f"{name} (in the release's .git)" for name in DEVELOPERS_OWN
                 if git(staging, "ls-tree", "--name-only", "HEAD", "--", name).strip()]
    junk += [str(Path(dp).relative_to(staging)) for dp, dirs, _ in os.walk(staging) if "__pycache__" in dirs][:5]
    # cmd.exe misreads .bat files with bare LF line endings (labels, blocks).
    lf_bats = [p.relative_to(staging).as_posix() for d in ("run-browser", "tools") for p in (staging / d).glob("*.bat")
               if re.search(rb"(?<!\r)\n", p.read_bytes())]
    for f in missing:
        print(f"  [{'FAIL' if strict else 'WARN'}] missing from the release: {f}")
    for j in junk:
        print(f"  [FAIL] must not be in the release: {j}")
    for b in lf_bats:
        print(f"  [FAIL] {b} has LF line endings; cmd.exe needs CRLF. Add '*.bat text eol=crlf' to a "
              ".gitattributes (or set git config core.autocrlf true), then build again.")
    git_bad = git_problems_in_release(staging) if (staging / ".git").is_dir() else []
    if (staging / TRIFLOW / ".git").exists():
        git_bad += [f"TriFlow is not byte-identical to its commits: {p}"
                    for p in nested.byte_identity_problems(git_exe(), staging / TRIFLOW, prefix=TRIFLOW + "/")][:15]
    for g in git_bad:
        print(f"  [FAIL] {g}")
    if junk or lf_bats or git_bad or (missing and strict):
        raise ReleaseError(f"{staging} is not a valid release (see above).")
    if missing:
        print("  [WARN] this staging folder is incomplete and must not be shipped")


def print_sizes(staging: Path) -> None:
    parts = ["tools/python", "tools/git", "code/forge", "code/pixal3d", "code/hi3d", "code/instant-meshes",
             "code/lato2", "code/triflow", "code/whl"]
    total, files = tree_size(staging)
    print(f"\nStaging folder {staging}: {files:,} files, {mb(total)}")
    counted = 0
    for p in parts:
        size, n = tree_size(staging / p)
        counted += size
        print(f"  {p:<22} {mb(size):>12}  ({n:,} files)")
    if (staging / ".git").is_dir():
        size, n = tree_size(staging / ".git")
        counted += size
        print(f"  {'.git (superproject)':<22} {mb(size):>12}  ({n:,} files, submodule objects included)")
    print(f"  {'everything else':<22} {mb(total - counted):>12}")


def make_zip(src: Path, zip_path: Path) -> None:
    """Zip src's contents (src itself is the zip root) to zip_path."""
    tmp = zip_path.with_name(zip_path.name + ".part")
    print(f"\nZipping into {zip_path}")
    done_bytes = done_files = 0
    last = 0.0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True, compresslevel=6,
                         strict_timestamps=False) as z:
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames.sort()
            here = Path(dirpath)
            arc_dir = here.relative_to(src).as_posix()
            if not dirnames and not filenames and arc_dir != ".":
                z.writestr(zipfile.ZipInfo(arc_dir + "/"), "")     # git needs some empty dirs (refs/tags...)
            for name in sorted(filenames):
                path = here / name
                z.write(path, path.relative_to(src).as_posix())
                done_files += 1
                done_bytes += path.stat().st_size
                if time.monotonic() - last >= 1.0:
                    last = time.monotonic()
                    print(f"\r  {done_files:,} files, {mb(done_bytes)}   ", end="", flush=True)
    os.replace(tmp, zip_path)
    print(f"\r  {done_files:,} files, {mb(done_bytes)} -> {mb(zip_path.stat().st_size)} zipped")


def cmd_build(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    problems = python_problems(manifest, full=False) + git_problems(manifest)
    if problems:
        raise ReleaseError("tools/ is not ready:\n  " + "\n  ".join(problems))
    staging = Path(args.out).resolve() if args.out else STAGING
    prepare_staging_dir(staging)
    if args.from_worktree:
        stage_worktree(staging)
    else:
        stage_committed(staging, args.allow_unpushed)

    print("Copying tools/python and tools/git (without __pycache__)")
    for tool_dir in (paths.PORTABLE_PYTHON_DIR, paths.PORTABLE_GIT_DIR):
        shutil.copytree(tool_dir, staging / tool_dir.relative_to(ROOT),
                        ignore=shutil.ignore_patterns("__pycache__"))
    stage_config(staging)
    verify_staging(staging, strict=not args.from_worktree)
    version = read_version(staging)
    print_sizes(staging)
    if args.no_zip:
        print(f"\nDone (not zipped): {staging}")
        return 0
    # "-windows": GitHub names a release's automatic "Source code (zip)" spellcast3d-<version>.zip.
    suffix = "-worktree" if args.from_worktree else "-unpushed" if args.allow_unpushed else ""
    zip_path = (staging.parent if args.out else RELEASE) / f"spellcast3d-{version}-windows{suffix}.zip"
    make_zip(staging, zip_path)
    print("\nDone. This zip is built from uncommitted files: for testing only." if args.from_worktree else
          "\nDone. This zip may not be able to update itself: for testing only." if args.allow_unpushed else
          "\nDone. Upload the zip as an asset of a GitHub Release (never commit it).")
    return 0


# ----------------------------------------------------------------------------- clean

def cmd_clean(args: argparse.Namespace) -> int:
    submodule_dirs = [ROOT / p for p in gitlinks(ROOT)] or list(paths.TOOL_DIRS.values())
    keep_out = [paths.PROJECTS, paths.EXPORTS, paths.VENVS, paths.PORTABLE_PYTHON_DIR, paths.PORTABLE_GIT_DIR, RELEASE,
                ROOT / ".git", ROOT / ".claude"]
    if not args.submodules:
        keep_out += submodule_dirs
    skip_names = {".git", "venv", ".venv", "models", "node_modules"}

    caches: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        here = Path(dirpath)
        kept = []
        for d in dirnames:
            full = here / d
            if d.lower() in skip_names or any(inside(full, k) for k in keep_out):
                continue
            if d == "__pycache__":
                caches.append(full)
            else:
                kept.append(d)
        dirnames[:] = kept
        caches += [here / f for f in filenames if f.endswith((".pyc", ".pyo"))]

    targets: list[Path] = [p for p in (paths.LOGS, paths.INSTALL_TMP, RELEASE / "staging") if p.exists()]
    if args.venvs and paths.VENVS.exists():
        targets.append(paths.VENVS)
    if args.tools:
        targets += [p for p in (paths.PORTABLE_PYTHON_DIR, paths.PORTABLE_GIT_DIR) if p.exists()]
    # Never user work or downloaded models, whatever the options.
    targets = [p for p in targets if not inside(p, paths.PROJECTS) and not inside(p, paths.EXPORTS)
               and "models" not in (part.lower() for part in p.relative_to(ROOT).parts)]

    cache_bytes = sum(tree_size(p)[0] for p in caches)
    where = "the repo and its submodules" if args.submodules else "the repo (submodules untouched; --submodules to include them)"
    print(f"{len(caches):,} __pycache__ folders / .pyc files in {where}: {mb(cache_bytes)}")
    for p in caches[:5]:
        print(f"    {rel(p)}")
    if len(caches) > 5:
        print(f"    ... and {len(caches) - 5:,} more")
    for p in targets:
        print(f"{rel(p)}: {mb(tree_size(p)[0])}")
    if not caches and not targets:
        print("Nothing to clean.")
        return 0
    if not args.yes:
        print("\nDry run: nothing was deleted. Add --yes to delete the items above.")
        return 0

    failed: list[str] = []
    for p in caches + targets:
        try:
            remove_tree(p)
        except OSError as e:
            failed.append(f"{rel(p)} ({e})")
    if failed:
        print("Could not delete (in use? close Spellcast3D and its tools, then retry):\n  " + "\n  ".join(failed))
        return 1
    print(f"Deleted {len(caches) + len(targets):,} items.")
    return 0


# ----------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")     # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(prog="prepare_release.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    p = sub.add_parser("fetch-tools", help="download + verify tools/python and tools/git, copy the VC++ DLLs")
    p.add_argument("--force", action="store_true", help="re-extract even if tools/ already look valid")
    p = sub.add_parser("build-wheel", help="build instant-meshes' cp311 wheel (static CRT) into code/whl")
    p.add_argument("--python", help="python.exe of a full CPython 3.11 (default: tools/python, then the system one)")
    p.add_argument("--no-test", action="store_true", help="skip installing the wheel into a scratch venv + pytest")
    sub.add_parser("check", help="report whether the repo is ready to release (non-zero exit if not)")
    p = sub.add_parser("scan-cpu", help="would these binaries run on a CPU without AVX2? (non-zero exit if not)")
    p.add_argument("paths", nargs="+", metavar="PATH", help="a .pyd / .dll / .exe, a wheel, or a folder of them")
    p.add_argument("--all", action="store_true", help="list every binary, not only those built for AVX2")
    p = sub.add_parser("build", help="assemble the release folder and zip")
    p.add_argument("--from-worktree", action="store_true",
                   help="copy the working tree (incl. uncommitted files, no .git) instead of cloning the committed state")
    p.add_argument("--allow-unpushed", action="store_true",
                   help="build a commit that is not on GitHub yet (testing only: such a zip cannot update)")
    p.add_argument("--out", help=f"staging folder (default: {rel(STAGING)})")
    p.add_argument("--no-zip", action="store_true", help="stop after assembling the staging folder")
    p = sub.add_parser("clean", help="remove __pycache__, logs, staging (dry-run unless --yes)")
    p.add_argument("--yes", action="store_true", help="really delete")
    p.add_argument("--venvs", action="store_true", help="also delete code/venvs (everything reinstalls on next start)")
    p.add_argument("--tools", action="store_true", help="also delete tools/python and tools/git")
    p.add_argument("--submodules", action="store_true", help="also clean __pycache__ inside the submodules")
    args = parser.parse_args(argv)

    if os.name != "nt":
        print("prepare_release.py builds a Windows release and must run on Windows.")
        return 2
    commands: dict[str, Callable[[argparse.Namespace], int]] = {
        "fetch-tools": cmd_fetch_tools, "build-wheel": cmd_build_wheel, "check": cmd_check,
        "scan-cpu": cmd_scan_cpu, "build": cmd_build, "clean": cmd_clean,
    }
    try:
        return commands[args.command](args)
    except ReleaseError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
