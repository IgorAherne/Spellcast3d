"""stable-diffusion.cpp (MIT), Qwen-Image 2.1's runtime: Spellcast3D's own Windows CUDA 12 build of it, pinned and
unpacked into code/venvs/qwen/sdcpp, with NVIDIA's CUDA runtime DLLs next to it. STDLIB ONLY.

Two zips, each from its own GitHub release (Release, Archive):
- the engine (SDCPP, 0.34 GB) from the author's fork, github.com/IgorAherne/stable-diffusion.cpp, release ENGINE:
  its branch spellcast3d at COMMIT with github.com/IgorAherne/ggml's branch spellcast3d at GGML_COMMIT, based on
  leejet/stable-diffusion.cpp master-929 (3f8527a) and leejet/ggml 89c4413, MIT like them. Built like the official
  cuda12 release (CUDA 12.8, sm_61;70;75;80;86;89;90;100;120: GTX 10xx .. RTX 50xx, one shared stable-diffusion.dll)
  but without WebP / WebM (the launcher reads and writes the pictures itself). What it changes: fewer copies and
  fused kernels in Qwen-Image 2.1's graph, an asynchronous int8 matmul on sm_80 and newer (its min terms summed on
  the tensor cores), SageAttention tuned for Qwen-Image 2.1 (INT8 queries and keys, FP16 values: the launcher's
  --attention) on the RTX 30 series and newer, a faster VAE decode, and the SD_PIN_PARAMS switch (page-locked
  weights: the launcher's --pin). Up to the release of 2026-10-03 (spellcast-perf-36f92d4) its pictures were
  bit-identical to the official release's; the two changes of 2026-10-21 are not bit for bit (the artist allowed
  mixed precision that leaves the pictures visually the same) and were checked in pictures instead: see
  launchers/qwen_server.py ATTENTIONS and code/ARCHITECTURE.md. On the RTX 3060 (sm_86) a step of a 1024x1024
  picture without guidance took 2.9-3.0 s with the official release, 1.94-2.00 s with spellcast-perf-36f92d4 and
  1.45-1.47 s with this build; the other GPU generations compile but were not run;
- NVIDIA's CUDA runtime (CUDART, 0.56 GB) from leejet's official release OFFICIAL, unchanged: the engine's zip holds
  none, like the official engine zip. Its cudart64_12, cublas64_12 and cublasLt64_12 (0.79 GB) are the same bytes as
  PyTorch 2.8.0+cu128's (checked against that zip: the SHA-256 of cudart64_12, and the size and CRC-32 of the other
  two), so they are hard-linked from a venv here that has that PyTorch (Pixal3D's, LATO.2's, ...) when one has them,
  and only otherwise taken from the cudart zip. Copies taken from that zip become links once such a venv is
  installed, and the zip goes then (share_cuda_dlls).

A zip is downloaded into code/venvs/downloads and kept there: a repair or a reinstall unpacks it again without
downloading. An install made with the former pin (the official release's engine) gets this build's files over it
(unpack_engine): the folder is marked with the zip its files came from (ENGINE_MARK), because sizes alone do not tell
the two builds apart, and the four files only the official release had are removed (FORMER_ONLY).
"""

from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .. import paths
from . import models

Log = Callable[[str], None]


@dataclass(frozen=True)
class Release:
    """A GitHub release: its repository ("owner/name") and its tag."""
    repo: str
    tag: str

    @property
    def url(self) -> str:
        """Where its assets are downloaded from (an asset's name follows)."""
        return f"https://github.com/{self.repo}/releases/download/{self.tag}/"

    @property
    def api(self) -> str:
        """GitHub's API for it: its assets with their sizes and SHA-256 (prepare_release.py check)."""
        return f"https://api.github.com/repos/{self.repo}/releases/tags/{self.tag}"


# The engine's release, and the commits it was built from: stable-diffusion.dll's sd_commit() answers COMMIT's first
# 7 characters (launchers/qwen_sdcpp.py pins the same commit and checks the DLL's layout against it).
ENGINE = Release("IgorAherne/stable-diffusion.cpp", "spellcast-perf-25d4127")
COMMIT = "25d412700bd616e4a9118abe6c5b42498de038f4"
GGML_COMMIT = "a7c45a7573f85e7032c1cf5da02dcc2ff2b0230b"
# THE ENGINE-SUPPORT GATE of FLUX.3 Image Klein, the Images stage's optional third model: whether ENGINE can load and
# draw it (upstream stable-diffusion.cpp's FLUX.3 support, PR #2143, which the fork took in 52bc254: release
# spellcast-perf-52bc254 was the first built with it - True since that pin, 2026-10-25).
# While False the hub does not offer FLUX.3 (the Images page shows it disabled: "needs the next engine update"; its
# download and its props are refused) and the launcher refuses its requests. Set it to True IN THE SAME CHANGE that
# moves ENGINE / COMMIT (and launchers/qwen_sdcpp.py's PINNED_TAG / PINNED_COMMIT) to a release built with FLUX.3.
# The one place for it: the hub (engine_draws: spellcast/model_downloads.py, adapters/qwen.py) and the launcher
# (launchers/qwen_server.py flux3_engine_choice: with the pinned build only; its --flux3-engine on is the developers'
# override for a build under test) all read it here.
ENGINE_HAS_FLUX3 = True
# Whether the pinned engine keeps a model's staged weights on the card between runs when the launcher asks
# (SD_KEEP_STAGED_PARAMS; the fork's 1fe8ef6, first in release spellcast-perf-a45dcf4; launchers/qwen_server.py keep_params). While False the
# launcher's --keep-params auto stays off with the pinned build (an older engine ignores the variable, and the direct
# decode the launcher would pair with it is slower there). Set it to True IN THE SAME CHANGE that moves the pin to a
# release built with it.
ENGINE_KEEPS_PARAMS = True
# Whether the pinned engine has the prefix-only first step's gates (the fork's 1834c1a: only where its stand-in
# attention is built and the drawn rows are on W8A8 or, from spellcast-perf-25d4127, on MMQ kernels that compute
# each row alike; earlier builds would abort with a q8_0 prefix cache) and the
# per-request prefix drop (SD_QWEN_PREFIX_DROP) of round P. While False the launcher leaves SD_QWEN_PREFIX_PASS off
# with the pinned build (launchers/qwen_server.py set_prefix_pass). Set it to True IN THE SAME CHANGE that moves the
# pin to a release built with them.
ENGINE_PREFIX_PASS = True


def engine_draws(model: str) -> bool:
    """Whether ENGINE draws the Images stage's model `model`: Qwen-Image 2.1's ("turbo", "full") always, FLUX.3 Image
    Klein ("flux3") only once ENGINE_HAS_FLUX3 says so."""
    return ENGINE_HAS_FLUX3 if model == "flux3" else True


# The official release ENGINE is based on, and the one the CUDA runtime zip still comes from. Qwen-Image 2.1 needs
# master-914 or newer (white blocks on CUDA before); this one has every fix for it (prefix cache, alpha, VAE).
OFFICIAL = Release("leejet/stable-diffusion.cpp", "master-929-3f8527a")


@dataclass(frozen=True)
class Archive:
    """A zip of a GitHub release (pinned: size, SHA-256) and the files it holds (flat: name -> bytes)."""
    release: Release
    name: str
    size: int
    sha256: str
    files: dict[str, int] = field(hash=False)

    @property
    def url(self) -> str:
        return self.release.url + self.name

    @property
    def path(self) -> Path:
        return paths.DOWNLOADS / self.name


SDCPP = Archive(ENGINE, "sd-spellcast-perf-25d4127-bin-win-cuda12-x64.zip", 352_768_144,
                "85e7c7c0bb3d98fd63b227d6cfd69934ff77f3123c7df647f9c3e083596c9549", {
                    "SPELLCAST-BUILD.txt": 919,         # which commits it was built from, and how
                    "ggml-base.dll": 681_472,
                    "ggml-cpu-alderlake.dll": 907_264,
                    "ggml-cpu-cannonlake.dll": 953_344,
                    "ggml-cpu-cascadelake.dll": 949_760,
                    "ggml-cpu-haswell.dll": 908_800,
                    "ggml-cpu-icelake.dll": 949_760,
                    "ggml-cpu-sandybridge.dll": 882_688,
                    "ggml-cpu-skylakex.dll": 953_344,
                    "ggml-cpu-sse42.dll": 869_888,
                    "ggml-cpu-x64.dll": 871_424,
                    "ggml-cuda.dll": 348_747_776,
                    "ggml.dll": 59_392,
                    "ggml.txt": 1_099,
                    "msvcp140.dll": 557_728,
                    "msvcp140_codecvt_ids.dll": 31_872,
                    "sageattention.txt": 11_558,
                    "sd-cli.exe": 665_088,
                    "sd-server.exe": 1_009_152,
                    "stable-diffusion.cpp.txt": 1_082,
                    "stable-diffusion.dll": 35_496_448,
                    "vcomp140.dll": 193_152,
                    "vcruntime140.dll": 124_544,
                    "vcruntime140_1.dll": 49_792,
                })
CUDART = Archive(OFFICIAL, "cudart-sd-bin-win-cu12-x64.zip", 563_452_046,
                 "fe20366827d357c00797eebb58244dddab7fd9a348d70090c3871004c320f38d", {
                     "cublas64_12.dll": 113_716_224,
                     "cublasLt64_12.dll": 674_667_520,
                     "cudart64_12.dll": 573_952,
                 })
# Files of an earlier pin that this build does not have (name -> bytes): the official release's engine zip
# (sd-master-3f8527a-bin-win-cuda12-x64.zip, the pin until October 2026) was built with WebP / WebM. An install made
# with it leaves them in the runtime folder; unpack_engine removes them, each only at exactly this size, and nothing
# else is ever deleted there. (A later pin keeps these and adds what it drops itself.)
FORMER_ONLY = {"libsharpyuv.dll": 24_576, "libwebp.dll": 373_248, "libwebpmux.dll": 41_984, "webm.dll": 328_192}
# Written into the runtime folder once the engine's files are all there: the SHA-256 of the zip they came from. A
# file's size does not say which build it is (ggml.dll has 59,392 bytes in the official release and in this build,
# with other bytes), so a folder without the pinned zip's mark gets every file of the engine again
# (runtime_missing): an install made with the former pin, or an unpacking that was interrupted.
ENGINE_MARK = "spellcast-engine.txt"
# CUDART's files: (bytes, SHA-256), the same as PyTorch 2.8.0+cu128's torch\lib copies.
CUDA_DLLS: models.Manifest = {
    "cudart64_12.dll": (573_952, "c2c9a9c22a9bcba90e261825968836787b331038047a26770cffb7a583c28344"),
    "cublas64_12.dll": (113_716_224, "9513540e4ec4c51ee9e7304138c2cc255c29a8c181f9e80c38efa25738becd99"),
    "cublasLt64_12.dll": (674_667_520, "b199d1ff892a81b7fd3d57ba1781549609b41500b36008fef326038393ad46c7"),
}
# What the runtime takes on disk: the unpacked engine and its kept zip, the CUDA DLLs and their kept zip (when they
# are not links to PyTorch's).
RUNTIME_GB = (sum(SDCPP.files.values()) + SDCPP.size) / 1e9
CUDA_GB = (sum(CUDART.files.values()) + CUDART.size) / 1e9
# Checks that the runtime loads, is the pinned build and sees the GPU: run by the qwen venv's Python in the runtime's
# folder. sd_commit() answers the first characters of the commit the DLL was built from ("25d412700bd616e4a9118abe6c5b42498de038f4"; this build's
# sd_version() is "unknown", so it is not asked). The rule is the launcher's (qwen_sdcpp.is_pinned).
VERIFY_CODE = ("import ctypes, os\n"
               "here = os.getcwd()\n"
               "os.add_dll_directory(here)\n"
               "sd = ctypes.CDLL(os.path.join(here, 'stable-diffusion.dll'))\n"
               "sd.sd_commit.restype = ctypes.c_char_p\n"
               "sd.sd_list_devices.restype = ctypes.c_size_t\n"
               "sd.sd_list_devices.argtypes = [ctypes.c_char_p, ctypes.c_size_t]\n"
               "buf = ctypes.create_string_buffer(1 << 16)\n"
               "sd.sd_list_devices(buf, len(buf))\n"
               "devices = buf.value.decode('utf-8', 'replace').strip()\n"
               "commit = (sd.sd_commit() or b'').decode('ascii', 'replace').strip().lower()\n"
               "print('stable-diffusion.cpp commit', commit or '?')\n"
               "print(devices)\n"
               f"if len(commit) < 7 or not '{COMMIT}'.startswith(commit):\n"
               f"    raise SystemExit('stable-diffusion.dll is not the pinned build ({ENGINE.tag})')\n"
               "if 'CUDA' not in devices: raise SystemExit('stable-diffusion.cpp cannot use the NVIDIA GPU')\n"
               "import fastapi, uvicorn, PIL\n"
               "print('Qwen-Image 2.1 runtime OK')\n")


def _wrong_size(folder: Path, files: dict[str, int]) -> list[str]:
    out = []
    for name, size in files.items():
        try:
            if os.stat(folder / name).st_size == size:
                continue
        except OSError:
            pass
        out.append(name)
    return out


def unpacked_from(folder: Path) -> str:
    """The SHA-256 of the zip the engine's files in `folder` came from (ENGINE_MARK); "" when nothing says."""
    try:
        return (folder / ENGINE_MARK).read_text(encoding="ascii", errors="replace").strip()
    except OSError:
        return ""


def runtime_missing(folder: Path | None = None) -> list[str]:
    """The engine's files that the runtime folder does not hold as pinned (a few stat calls and one small read):
    those not there at their size - and all of them while the folder is not marked as unpacked from the pinned zip
    (ENGINE_MARK: another build's files, or an unpacking that did not finish)."""
    folder = folder or paths.QWEN_SDCPP
    return _wrong_size(folder, SDCPP.files) if unpacked_from(folder) == SDCPP.sha256 else list(SDCPP.files)


def cuda_missing(folder: Path | None = None) -> list[str]:
    """The CUDA DLLs that are not next to the runtime at their size."""
    return _wrong_size(folder or paths.QWEN_SDCPP, CUDART.files)


def missing(folder: Path | None = None) -> list[str]:
    """Short names of what is missing of the runtime, for the installer's repair check."""
    return (["stable-diffusion.cpp"] if runtime_missing(folder) else []) + cuda_missing(folder)


def cached(archive: Archive) -> bool:
    """The zip is in code/venvs/downloads at its size (checked against its SHA-256 when it is unpacked)."""
    try:
        return os.stat(archive.path).st_size == archive.size
    except OSError:
        return False


def drop_partial(archive: Archive) -> None:
    """Delete the interrupted download (.part) of `archive` that nothing needs any more: the CUDA zip's, cut off before
    a venv with PyTorch's copies of its DLLs was installed, would otherwise stay in code/venvs/downloads for good."""
    try:
        archive.path.with_name(archive.name + ".part").unlink()
    except OSError:
        pass


def download_bytes(archive: Archive) -> int:
    """How much of the zip is downloaded (for the progress)."""
    for path in (archive.path, archive.path.with_name(archive.name + ".part")):
        try:
            return min(os.stat(path).st_size, archive.size)
        except OSError:
            continue
    return 0


def torch_libs(name: str) -> list[Path]:
    """`name` in the torch\\lib folder of every other venv here that has it at the pinned size (stat calls)."""
    size = CUDA_DLLS[name][0]
    found = []
    for venv_id in paths.ALL_VENV_IDS:
        lib = paths.venv_dir(venv_id) / "Lib" / "site-packages" / "torch" / "lib" / name
        try:
            if venv_id != paths.QWEN and os.stat(lib).st_size == size:
                found.append(lib)
        except OSError:
            continue
    return found


def source_folders() -> list[Path]:
    """Where another copy of the zips may be: each --models-from / SPELLCAST_MODELS_FROM folder itself, its downloads
    folder, and - for another Spellcast3D's models folder, the documented use - that Spellcast3D's
    code\\venvs\\downloads, where its installer keeps them."""
    found: list[Path] = []
    for folder in models.extra_sources():
        for candidate in (folder, folder / "downloads", folder.parent / "code" / "venvs" / "downloads"):
            if candidate.is_dir() and candidate not in found:
                found.append(candidate)
    return found


def on_this_pc(archive: Archive) -> bool:
    """The zip is in code/venvs/downloads, or in a source_folders() folder on the same drive (linked there by fetch,
    not downloaded): a few stat calls, no hashing."""
    if cached(archive):
        return True
    try:
        drive = os.stat(paths.VENVS if paths.VENVS.exists() else paths.ROOT).st_dev
    except OSError:
        return False
    for folder in source_folders():
        try:
            st = os.stat(folder / archive.name)
        except OSError:
            continue
        if st.st_size == archive.size and st.st_dev == drive:
            return True
    return False


def fetch(archive: Archive, log: Log) -> Path:
    """The zip in code/venvs/downloads, SHA-256 checked: the one there, else one in a source_folders() folder (a hard
    link), else downloaded (resumed). Raises OSError (models.ChecksumError: downloaded in full, but not that file)."""
    path = archive.path
    if cached(archive):
        if models.is_recorded(path, archive.sha256) or models.sha256_file(path) == archive.sha256:
            models.record_verified(path, archive.sha256)
            return path
        log(f"{path.name} in code\\venvs\\downloads is not the expected file (its SHA-256 differs): it is downloaded "
            f"again.")
        path.unlink()
    for folder in source_folders():
        if models.link_from_local_dir(folder, path.parent, [archive.name], {archive.name: (archive.size,
                                                                                          archive.sha256)}, log,
                                      what="zip(s) of the image engine's runtime"):
            return path
    log(f"Downloading {archive.name} ({archive.size / 1e9:.2f} GB) from GitHub ...")
    models.download_file(archive.url, path, archive.size, archive.sha256, log)
    return path


def unpack(archive: Archive, folder: Path, names: list[str] | None = None) -> None:
    """Write `names` of the zip's files (default: those not in `folder` at their size) into `folder`, each through a
    temporary name. Only the pinned names are taken from the zip, so nothing lands outside `folder`. Raises OSError."""
    todo = set(names if names is not None else _wrong_size(folder, archive.files)) & set(archive.files)
    if not todo:
        return
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive.path) as z:
            for info in z.infolist():
                if info.filename not in todo:
                    continue
                target = folder / info.filename
                tmp = target.with_name(target.name + ".partial")
                try:
                    with z.open(info) as src, open(tmp, "wb") as out:
                        shutil.copyfileobj(src, out, 1 << 20)
                    os.replace(tmp, target)
                finally:
                    tmp.unlink(missing_ok=True)
    except zipfile.BadZipFile as e:
        raise OSError(f"{archive.name}: {e}") from e
    wrong = _wrong_size(folder, {n: archive.files[n] for n in todo})
    if wrong:
        raise OSError(f"{archive.name} did not give {', '.join(wrong)}")


def unpack_engine(folder: Path, log: Log) -> None:
    """The engine's zip (fetched before) into `folder`: the files runtime_missing names - all of them over another
    build's -, then the former pin's leftovers are removed (FORMER_ONLY) and, last, the folder is marked
    (ENGINE_MARK). Raises OSError; interrupted before the mark, the next run unpacks everything again."""
    unpack(SDCPP, folder, runtime_missing(folder))
    removed = []
    for name, size in FORMER_ONLY.items():
        try:
            if os.stat(folder / name).st_size == size:
                (folder / name).unlink()
                removed.append(name)
        except OSError:
            continue                    # not there, or it cannot be removed now: it harms nothing where it is
    paths.write_text_atomic(folder / ENGINE_MARK, SDCPP.sha256 + "\n")
    if removed:
        log(f"Removed {len(removed)} files of the stable-diffusion.cpp build used before, which this one does not "
            f"need ({', '.join(removed)}).")


def link_cuda_dlls(folder: Path, log: Log) -> list[str]:
    """Hard-link the CUDA DLLs missing in `folder` from a venv's PyTorch that has them byte for byte (SHA-256, hashed
    once per file and remembered), or copy them where no hard link can be made (a copy of 0.79 GB still beats a
    download). Returns the names still missing."""
    linked: list[tuple[str, Path]] = []
    for name in cuda_missing(folder):
        sha = CUDA_DLLS[name][1]
        for lib in torch_libs(name):
            try:
                if not (models.is_recorded(lib, sha) or models.sha256_file(lib) == sha):
                    continue
            except OSError:
                continue
            models.record_verified(lib, sha)
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / name
            tmp = target.with_name(name + ".link")
            try:
                tmp.unlink(missing_ok=True)
                try:
                    os.link(lib, tmp)
                except OSError:
                    shutil.copy2(lib, tmp)
                os.replace(tmp, target)
            except OSError as e:
                tmp.unlink(missing_ok=True)
                log(f"Could not take {name} from {lib} ({e}).")
                continue
            linked.append((name, lib))
            break
    if linked:
        venvs = ", ".join(sorted({_rel(lib.parents[4]) for _name, lib in linked}))
        size = sum(CUDA_DLLS[name][0] for name, _lib in linked)
        log(f"NVIDIA's CUDA files for stable-diffusion.cpp ({size / 1e9:.2f} GB) are the same as PyTorch's in "
            f"{venvs}: linked from there, which takes no extra disk space.")
    return cuda_missing(folder)


def share_cuda_dlls(log: Log, folder: Path | None = None) -> int:
    """CUDA DLLs that were unpacked from the cudart zip (no venv here had PyTorch then: on a new PC Qwen-Image installs
    before Pixal3D) become hard links to a venv's PyTorch copy once one has them byte for byte, and the cudart zip is
    removed once none of them needs it: ~1.35 GB of disk back. Run after an installer run; a DLL that is in use
    (Qwen-Image is running) stays as it is until the next run. -> the bytes given back."""
    folder = folder or paths.QWEN_SDCPP
    if not folder.is_dir():
        return 0
    freed, all_shared = 0, True
    for name, (size, sha) in CUDA_DLLS.items():
        target = folder / name
        try:
            st = os.stat(target)
        except OSError:
            all_shared = False          # missing: the next repair links or unpacks it
            continue
        twins = torch_libs(name)
        if st.st_size != size or not twins:
            all_shared = False
            continue
        if any(_same_file(target, lib) for lib in twins):
            continue
        shared = False
        for lib in twins:
            try:
                if os.stat(lib).st_dev != st.st_dev:
                    continue            # another drive: no hard link
                if not (models.is_recorded(lib, sha) or models.sha256_file(lib) == sha):
                    continue
                models.record_verified(lib, sha)
                tmp = target.with_name(name + ".link")
                tmp.unlink(missing_ok=True)
                os.link(lib, tmp)
                try:
                    os.replace(tmp, target)
                finally:
                    tmp.unlink(missing_ok=True)
            except OSError:
                continue
            freed += size if st.st_nlink <= 1 else 0
            shared = True
            break
        all_shared = all_shared and shared
    if all_shared and CUDART.path.exists():
        try:
            CUDART.path.unlink()
            freed += CUDART.size
        except OSError:
            pass
    if freed:
        log(f"NVIDIA's CUDA files for stable-diffusion.cpp are hard links to PyTorch's now: {freed / 1e9:.2f} GB of "
            f"disk space given back.")
    return freed


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def dry_run(log: Log, folder: Path | None = None) -> None:
    """What install would do, for install.py --dry-run."""
    folder = folder or paths.QWEN_SDCPP
    rel = _rel(folder)
    log(f"  [dry-run] {rel}: if the {len(SDCPP.files)} files of Spellcast3D's stable-diffusion.cpp build "
        f"({SDCPP.release.tag}) are not all there, unpack {_rel(SDCPP.path)}, downloaded first if it is not there: "
        f"{SDCPP.url} ({SDCPP.size / 1e9:.2f} GB, SHA-256 checked)")
    for name in CUDART.files:
        twins = ", ".join(_rel(lib.parents[4]) for lib in torch_libs(name)) or "no venv here yet"
        log(f"  [dry-run] {rel}\\{name}: if missing, a hard link to PyTorch's when its SHA-256 is the pinned one "
            f"({twins}), else from stable-diffusion.cpp's official release: {CUDART.url} "
            f"({CUDART.size / 1e9:.2f} GB)")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(paths.ROOT))
    except ValueError:
        return str(path)
