"""Checks before installing: Python, GPU + driver, disk, install path, VC++ runtime, internet. STDLIB ONLY.

A problem either stops everything (fatal) or blocks only the components it concerns
(blocked, e.g. no NVIDIA GPU blocks the image generator and Pixal3D but not the hub), so the hub can
still start and show what is missing. "Uses the GPU" is not "needs PyTorch": Qwen-Image 2.1 runs on
stable-diffusion.cpp, so download.pytorch.org and pip's room for PyTorch do not concern it.
"""

from __future__ import annotations

import math
import os
import platform
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .. import gpu, paths
from . import submodules, venvs
from .components import PROBE_URLS, TORCH_DIST, Component, git_exe
from .runner import pip_cache_dir

# Deepest installed files sit ~170 characters below the root; Windows' limit is 260.
MAX_ROOT_LEN = 60
DISK_MARGIN_GB = 1.0
CACHE_MIN_GB = 5.0  # pip keeps the 3.5 GB PyTorch download in its cache (%LOCALAPPDATA%\pip\cache)
VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
DRIVER_URL = "https://www.nvidia.com/drivers"


@dataclass
class Report:
    info: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fatal: list[str] = field(default_factory=list)
    blocked: dict[str, str] = field(default_factory=dict)   # component id -> reason

    def block(self, components: list[Component], reason: str) -> None:
        for c in components:
            self.blocked.setdefault(c.id, reason)


def run(todo: list[Component], gpu_info: gpu.GpuInfo | None, *, network: bool,
        reinstall: tuple[str, ...] = ()) -> Report:
    """todo = the components that are about to be installed; reinstall = the ids of those whose venvs are deleted
    and made again (--reinstall)."""
    r = Report()
    _check_python(r)
    _check_folder(r, todo)
    _check_vc_runtime(r)
    _check_gpu(r, todo, gpu_info)
    _check_disk(r, todo, reinstall)
    if network and todo:
        _check_internet(r, todo)
    return r


def _check_python(r: Report) -> None:
    r.info.append(f"Python {platform.python_version()} ({sys.executable})")
    if sys.version_info[:2] != (3, 11) or sys.maxsize <= 2**32:
        r.fatal.append(f"Spellcast3D needs 64-bit Python 3.11, but this is Python {platform.python_version()} "
                       f"({platform.architecture()[0]}). Use tools\\python\\python.exe from the Spellcast3D download.")


def _long_paths_enabled() -> bool:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            return winreg.QueryValueEx(key, "LongPathsEnabled")[0] == 1
    except OSError:
        return False


def ansi_safe(path: str) -> bool:
    """True when `path` survives Windows' ANSI code page ("language for non-Unicode programs").

    Programs that take narrow (char) paths - Pixal3D's GPU kernel compilers (triton's tcc.exe and
    ptxas.exe) among them - see other letters replaced by '?' or a look-alike ('Łukasz' becomes
    'Lukasz' on English Windows) and cannot find their files. Accented letters of the user's own
    language (José on English Windows) are fine.
    """
    try:
        path.encode("mbcs", "strict")
    except UnicodeEncodeError:
        return False
    except LookupError:   # not Windows
        return True
    return True


def _check_folder(r: Report, todo: list[Component]) -> None:
    root = str(paths.ROOT)
    r.info.append(f"Install folder: {root}")
    if not ansi_safe(root):
        problem = (f"The install folder path ({root}) contains letters that Windows' 'language for non-Unicode "
                   f"programs' cannot represent. Pixal3D cannot build its GPU code there, so making 3D models "
                   f"would fail. Move the Spellcast3D folder to a simple path such as C:\\spellcast3d and start "
                   f"again.")
        # Fatal before anything is downloaded: the venvs cannot simply be moved afterwards (they record
        # their absolute path), so everything would have to be installed again.
        if any(c.id == paths.PIXAL3D for c in todo):
            r.fatal.append(problem)
        else:
            r.warnings.append(problem)
    odd = sorted({ch for ch in root if ch in "&%!^"})
    if odd:
        r.warnings.append(f"The install folder path contains special characters ({' '.join(odd)}). Some of the "
                          f"tools cannot handle them, even when the installation itself works, and Spellcast3D "
                          f"may not start by double-click. Move the Spellcast3D folder to a simple path such as "
                          f"C:\\spellcast3d and start again.")
    if len(root) > MAX_ROOT_LEN and not _long_paths_enabled():
        r.fatal.append(f"The install folder path is too long ({len(root)} characters; at most {MAX_ROOT_LEN} "
                       f"work). Some files are installed about 170 characters deeper, and Windows cannot handle "
                       f"paths longer than 260 characters. Move the Spellcast3D folder to a short path such as "
                       f"C:\\spellcast3d and start again.")
    _check_triflow_path(r, todo)
    try:
        with tempfile.TemporaryFile(dir=paths.CODE):
            pass
    except OSError:
        r.fatal.append(f"Spellcast3D cannot write into its own folder ({paths.CODE}). Do not put it inside "
                       f"'Program Files' or another protected folder; move it to e.g. C:\\spellcast3d.")


def _check_triflow_path(r: Report, todo: list[Component]) -> None:
    """TriFlow's parts are downloaded with git into folders ~165 characters below the root (submodules.py): in a git
    copy with a longer root path than TRIFLOW_MAX_ROOT_LEN that download fails, so TriFlow alone is blocked."""
    triflow = [c for c in todo if c.id == paths.TRIFLOW]
    root = str(paths.ROOT)
    if (not triflow or len(root) <= submodules.TRIFLOW_MAX_ROOT_LEN or not (paths.ROOT / ".git").exists()
            or not submodules.triflow_checkout_work(git_exe())):
        return
    r.block(triflow, f"TriFlow cannot be downloaded into this folder because its path is too long ({len(root)} "
                     f"characters). Git keeps TriFlow's innermost part about {len(submodules.TRIFLOW_GIT_DEEPEST)} "
                     f"characters deeper, and Windows cannot handle paths longer than 260 characters there. Move the "
                     f"Spellcast3D folder to a shorter path (at most {submodules.TRIFLOW_MAX_ROOT_LEN} characters, "
                     f"such as C:\\spellcast3d) and start again. The other tools are not affected.")


def _check_vc_runtime(r: Report) -> None:
    folders = [paths.PORTABLE_PYTHON_DIR, Path(sys.executable).parent,
               Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"]
    if not any((folder / "msvcp140.dll").exists() for folder in folders):
        r.fatal.append(f"The Microsoft Visual C++ Redistributable is not installed (msvcp140.dll is missing); "
                       f"PyTorch needs it. Install it from {VC_REDIST_URL} and start again.")


def _check_gpu(r: Report, todo: list[Component], gpu_info: gpu.GpuInfo | None) -> None:
    needs_gpu = [c for c in todo if c.uses_gpu]
    if gpu_info is None:
        msg = ("No NVIDIA graphics card was found (or its driver is not installed). Image and 3D generation "
               f"need an NVIDIA GPU. If you have one, install its latest driver from {DRIVER_URL} and start again.")
        if needs_gpu:
            r.block(needs_gpu, msg)
        else:
            r.warnings.append(msg)
        return
    r.info.append(f"GPU: {gpu_info.name}, {gpu_info.vram_total_mb / 1024:.0f} GB, "
                  f"compute capability {gpu_info.compute_cap:.1f}, driver {gpu_info.driver_version}")
    ok, why = gpu.driver_ok(gpu_info)
    if not ok:
        r.block(needs_gpu, f"{why} Download it from {DRIVER_URL}, install it, and start again.")
    if gpu_info.vram_total_mb < 7900:
        r.warnings.append(f"Your GPU has {gpu_info.vram_total_mb / 1024:.0f} GB of video memory. Spellcast3D is "
                          f"made for 8 GB or more; generating may fail or be very slow.")


def _check_disk(r: Report, todo: list[Component], reinstall: tuple[str, ...] = ()) -> None:
    usage = shutil.disk_usage(paths.ROOT)
    free = usage.free / 1e9
    drive = paths.ROOT.drive or str(paths.ROOT.anchor)
    total_need = 0.0
    # A new venv shares its identical files (the same torch) with a venv that has them by hard links: it keeps
    # less disk than it needs while installing. That venv may also be one installed earlier in this run.
    torch_twin = any(venvs.torch_dist(paths.venv_dir(v)) == TORCH_DIST for v in paths.ALL_VENV_IDS)
    for c in todo:
        if c.id in r.blocked:
            continue
        # A venv made again needs as much as a first install: its files other venvs share stay on disk through them.
        fresh = c.id in reinstall
        need = c.disk_needed_gb(fresh)
        if need + DISK_MARGIN_GB > free:
            # Rounded apart (need up, free down; the margin included): never "needs about 10 GB, but only 10 GB".
            r.block([c], f"Not enough free disk space: {c.title} needs about {math.ceil(need + DISK_MARGIN_GB)} GB, "
                         f"but only {math.floor(max(0.0, free))} GB is free on drive {drive}. Free up space (or move "
                         f"the Spellcast3D folder to a bigger drive) and start again.")
        else:
            kept = c.disk_kept_gb(torch_twin, fresh)
            free -= kept
            total_need += kept
            torch_twin = torch_twin or c.needs_torch
    need_text = (f" (this install needs about {total_need:.0f} GB)" if total_need >= 1
                 else " (this install needs less than 1 GB)" if total_need > 0 else "")
    r.info.append(f"Free disk space on drive {drive} {usage.free / 1e9:.0f} GB{need_text}")
    # Temporary files go to code\tmp on this drive (runner.Ctx.env). pip's download cache is elsewhere:
    cache = pip_cache_dir()
    if any(c.needs_torch and c.id not in r.blocked for c in todo) and cache.drive.lower() != paths.ROOT.drive.lower():
        try:
            cache_free = shutil.disk_usage(cache.anchor or cache).free / 1e9
        except OSError:
            return
        if cache_free < CACHE_MIN_GB:
            r.warnings.append(f"Only {cache_free:.1f} GB is free on drive {cache.drive}, where pip keeps its "
                              f"downloads ({cache}). PyTorch (3.5 GB) cannot be kept there, so it is downloaded "
                              f"once for each tool that needs it, and a partial file may stay there. Free up "
                              f"about {CACHE_MIN_GB:.0f} GB on {cache.drive} to download it only once.")


def _reachable(url: str) -> str | None:
    """None when url answers, else a short description of the problem."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=20):
            return None
    except urllib.error.HTTPError as e:
        # 404/405 still prove the server is there; 403/407/429/5xx usually mean a proxy or firewall.
        return None if e.code in (404, 405) else f"the server answered HTTP {e.code}"
    except (urllib.error.URLError, OSError) as e:
        return str(getattr(e, "reason", e))


def _check_internet(r: Report, todo: list[Component]) -> None:
    problem = _reachable("https://pypi.org/simple/pip/")
    if problem:
        r.fatal.append(f"Could not reach pypi.org, where the Python packages come from ({problem}). Installing "
                       f"needs the internet: check your connection, VPN or firewall and start again.")
        return
    reached = ["pypi.org"]
    # Qwen-Image 2.1 uses the GPU without PyTorch: download.pytorch.org must not block it.
    needs_torch = [c for c in todo if c.needs_torch and c.id not in r.blocked]
    if needs_torch:
        problem = _reachable("https://download.pytorch.org/whl/cu128/")
        if problem:
            r.block(needs_torch, f"Could not reach download.pytorch.org, where PyTorch comes from ({problem}). "
                                 f"Check your connection, VPN or firewall and start again.")
        else:
            reached.append("download.pytorch.org")
    # The other servers the components must download from now (e.g. huggingface.co for AI models, github.com for
    # stable-diffusion.cpp): each one checked once, and only for what it is needed for.
    by_host: dict[str, list[Component]] = {}
    for c in todo:
        if c.id not in r.blocked:
            for host in sorted(c.download_hosts()):
                by_host.setdefault(host, []).append(c)
    for host, needing in by_host.items():
        if host not in PROBE_URLS:
            continue
        problem = _reachable(PROBE_URLS[host])
        if problem:
            r.block(needing, f"Could not reach {host}, where {_WHAT_FROM.get(host, 'some of its files')} come from "
                             f"({problem}). Check your connection, VPN or firewall and start again.")
        else:
            reached.append(host)
    r.info.append(f"Internet: {', '.join(reached[:-1]) + ' and ' if len(reached) > 1 else ''}{reached[-1]} reachable")


_WHAT_FROM = {"huggingface.co": "its AI models", "github.com": "its program files"}
