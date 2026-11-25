"""Creating, repairing and deleting the per-tool venvs in code/venvs, and sharing their identical files. STDLIB ONLY."""

from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path

from .. import paths
from .runner import Ctx, InstallError

PIP_VERSION = "26.2.1"
# Packages whose big files are the same bytes in every venv that has the same torch (they come from the same
# wheels): share_identical_files() makes a new venv's copies hard links to an existing venv's. A top-level name
# matches the package folder, its ".libs" folder and its top-level extension (flash_attn_2_cuda.cp311-win_amd64.pyd).
SHARED_PACKAGES: tuple[str, ...] = (
    "torch", "torchvision", "torchaudio", "triton", "xformers", "flash_attn", "flash_attn_2_cuda", "numpy", "scipy",
    "spconv", "cumm", "cv2", "open3d", "llvmlite", "nvdiffrast", "o_voxel",
)
SHARE_MIN_BYTES = 1_000_000     # smaller files are not worth a link


def _read_cfg(venv_id: str) -> dict[str, str] | None:
    try:
        text = (paths.venv_dir(venv_id) / "pyvenv.cfg").read_text(encoding="utf-8")
    except OSError:
        return None
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key.strip().lower()] = value.strip()
    return values


def usable(venv_id: str) -> bool:
    """The venv's python.exe exists and the base Python it points to exists too."""
    cfg = _read_cfg(venv_id)
    return (paths.venv_python(venv_id).exists() and cfg is not None
            and (Path(cfg.get("home", "")) / "python.exe").exists())


def repair_if_moved(venv_id: str, base_python: Path) -> bool:
    """Point the venv at base_python again after the Spellcast3D folder was moved or renamed.

    A venv made by `python -m venv` only records the base Python's absolute folder
    (pyvenv.cfg "home"); nothing we run depends on any other absolute path in it.
    Returns True when the file was rewritten.
    """
    cfg = _read_cfg(venv_id)
    if cfg is None or (Path(cfg.get("home", "")) / "python.exe").exists():
        return False
    if cfg.get("version", "").split(".")[:2] != [str(v) for v in sys.version_info[:2]]:
        return False  # made by another Python version: must be recreated, not repointed
    path = paths.venv_dir(venv_id) / "pyvenv.cfg"
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        key = line.partition("=")[0].strip().lower()
        if key == "home":
            line = f"home = {base_python.parent}"
        elif key == "executable":
            line = f"executable = {base_python}"
        lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def _make_writable_and_retry(func, path, _exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def pip_intact(venv_id: str) -> bool:
    """pip is completely installed in the venv. pip writes pip-*.dist-info/RECORD last, so a pip whose
    install was cut off (window closed, power cut) has no RECORD - even when `pip --version` still works."""
    sp = paths.venv_dir(venv_id) / "Lib" / "site-packages"
    return (sp / "pip" / "__init__.py").exists() and any(sp.glob("pip-*.dist-info/RECORD"))


def delete(ctx: Ctx, venv_id: str, title: str) -> None:
    target = paths.venv_dir(venv_id)
    if not target.exists():
        return
    if ctx.dry_run:
        ctx.out.line(f"  [dry-run] delete {target}")
        return
    ctx.out.line(f"Deleting the old Python environment {target} ...")
    try:
        shutil.rmtree(target, onerror=_make_writable_and_retry)
    except OSError as e:
        ctx.out.line(f"Could not delete it: {e}")
        raise InstallError(
            f"Could not delete the old Python environment of {title} ({target}). Close Spellcast3D and "
            f"any window that still runs {title}, then try again.") from e


def ensure(ctx: Ctx, venv_id: str, title: str, fresh: bool) -> None:
    """Create the venv if needed (fresh=True deletes it first, a broken one is recreated) and pin pip."""
    target = paths.venv_dir(venv_id)
    if target.exists() and not fresh and not usable(venv_id):
        ctx.out.line(f"The Python environment of {title} is incomplete or broken; it is created again.")
        fresh = True
    if fresh:
        delete(ctx, venv_id, title)
    cannot_create = (f"Could not create the Python environment for {title}. Make sure the Spellcast3D folder "
                     f"is not inside 'Program Files' and that your antivirus does not block it, then try again.")
    if fresh or not paths.venv_python(venv_id).exists():
        if not ctx.dry_run:
            paths.VENVS.mkdir(parents=True, exist_ok=True)
        ctx.run([str(ctx.base_python), "-m", "venv", str(target)], cwd=paths.VENVS, env=ctx.env(None),
                fail=cannot_create)
    elif not ctx.dry_run and not pip_intact(venv_id):
        # `python -m venv` or the pip upgrade below was stopped (window closed, Ctrl+C, power cut) while
        # pip's files were being written: pip is missing or half-written and cannot repair itself.
        # Remove it and pip's own stash folders (~ip*), then put back the pip bundled with Python:
        # offline, and nothing else installed so far is lost. (ensurepip does nothing while a
        # pip-*.dist-info is present.)
        ctx.out.line(f"The Python environment of {title} was not finished last time; finishing it.")
        sp = target / "Lib" / "site-packages"
        try:
            for leftover in [sp / "pip", *sp.glob("pip-*.dist-info"), *sp.glob("~ip*")]:
                if leftover.exists():
                    shutil.rmtree(leftover, onerror=_make_writable_and_retry)
        except OSError as e:
            raise InstallError(cannot_create) from e
        ctx.run([str(paths.venv_python(venv_id)), "-m", "ensurepip", "--default-pip"], cwd=target,
                env=ctx.env(venv_id), fail=cannot_create)
    # In a new venv this is the pip bundled with Python (24.0), which lacks "--progress-bar raw".
    ctx.run([str(paths.venv_python(venv_id)), "-m", "pip", "install", f"pip=={PIP_VERSION}"],
            cwd=target, env=ctx.env(venv_id),
            fail=f"Could not set up pip for {title}. Check your internet connection and start again. If it "
                 f"keeps failing, close Spellcast3D, delete the folder code\\venvs\\{venv_id} and start again.")


# ---- identical files shared between venvs ------------------------------------------------------------------

def site_packages(venv: Path) -> Path:
    return venv / "Lib" / "site-packages"


def torch_dist(venv: Path) -> str | None:
    """The folder name of the venv's installed torch metadata (e.g. "torch-2.8.0+cu128.dist-info"), or None."""
    return next((p.name for p in sorted(site_packages(venv).glob("torch-*.dist-info"))), None)


def torch_twins(venv_id: str, dist: str | None = None) -> list[Path]:
    """The other venvs in code/venvs with the same torch as venv_id's (or as `dist`, e.g. before it exists)."""
    dist = dist or torch_dist(paths.venv_dir(venv_id))
    if dist is None:
        return []
    return [paths.venv_dir(other) for other in paths.ALL_VENV_IDS
            if other != venv_id and torch_dist(paths.venv_dir(other)) == dist]


def _shared(name: str, packages: tuple[str, ...]) -> bool:
    return not name.endswith(".dist-info") and any(name == p or name.startswith(p + ".") for p in packages)


def _same_bytes(a: str, b: str) -> bool:
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            x, y = fa.read(1 << 20), fb.read(1 << 20)
            if x != y:
                return False
            if not x:
                return True


def share_identical_files(dst: Path, candidates: list[Path], packages: tuple[str, ...] = SHARED_PACKAGES,
                          min_bytes: int = SHARE_MIN_BYTES, dry_run: bool = False) -> tuple[int, int]:
    """Replace dst venv's big files of `packages` that are byte for byte the same file (same path in
    site-packages, same size, same bytes) in a candidate venv on the same drive by hard links to that copy.
    Returns (files linked, bytes saved).

    Safe because pip never edits an installed file in place: an upgrade or uninstall removes the file and writes
    a new one, which only drops that venv's name for it. Only dst's own files are replaced; the candidates' files
    are just link targets and are never changed. dry_run: only count.
    """
    root = site_packages(dst)
    others = [site_packages(c) for c in candidates if site_packages(c).is_dir()]
    if not root.is_dir() or not others:
        return 0, 0
    linked = saved = 0
    for entry in os.scandir(root):
        if not _shared(entry.name, packages):
            continue
        if entry.is_file(follow_symlinks=False):
            files = [entry.path]
        else:
            files = []
            for dirpath, dirs, names in os.walk(entry.path):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                files += [os.path.join(dirpath, n) for n in names]
        for file in files:
            try:
                st = os.lstat(file)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode) or st.st_size < min_bytes:
                continue
            rel = os.path.relpath(file, root)
            for other_root in others:
                other = os.path.join(other_root, rel)
                try:
                    ost = os.stat(other)
                except OSError:
                    continue
                if ost.st_dev != st.st_dev or ost.st_size != st.st_size:
                    continue
                if ost.st_ino == st.st_ino:
                    break                       # linked already
                try:
                    if not _same_bytes(file, other):
                        continue
                    if not dry_run:
                        tmp = file + ".sc3dlink"
                        if os.path.lexists(tmp):
                            os.unlink(tmp)
                        os.link(other, tmp)
                        try:
                            os.replace(tmp, file)
                        except OSError:
                            os.unlink(tmp)
                            raise
                except OSError:
                    break                       # in use or not allowed: keep this venv's own copy
                linked += 1
                saved += st.st_size
                break
    return linked, saved
