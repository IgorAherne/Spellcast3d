"""An optional model downloaded into Spellcast3D's models folder, in a child process of the hub.

    <hub venv>\\Scripts\\python.exe qwen_download.py --model full|flux3|hi3d --models <its folder> --quality Q4_K_M

The hub (spellcast/model_downloads.py) starts it when the artist asks for one on the Images page (install/models.py
OPTIONAL_MODELS, optional_manifest): "full", the full Qwen-Image 2.1's denoiser of the size spellcast3d-config.ini
picks, into models/qwen-image (the one file the installer no longer fetches since Qwen-Image-2.1-Turbo became the
default; the default --model, as before there was a choice), or "flux3", FLUX.3 Image Klein's three files (its
denoiser of that size, its text encoder and its VAE) into models/flux3-klein. A denoiser of a bigger size already in
the folder stands in for the configured one (install/models.py size_on_disk): only what is missing besides is fetched.
Or "hi3d", Hi3D's AI models (the 3D stage's second Shape: install/models.py HI3D_FILES, 11.4 GB in four folders) into
models/hi3d, when a prop is set to Hi3D (--quality is Qwen-Image's then, unused). A file put into the folder by hand is
checked once against its pin before anything is fetched (a wrong one is removed and fetched again), as the installer
does with its own.
It is the installer's own downloader (install/models.py download_file: HTTPS from the pinned HuggingFace commit,
continued from <name>.part after an interruption, the SHA-256 checked before the file gets its name, remembered in
models-verified.json), so a file it finished is exactly what an install would have put there. Before downloading it
looks for a copy on this PC, as the installer does: the SPELLCAST_MODELS_FROM folders (laid out like ours), the models
folder next to --models that holds Qwen-Image's files (models/qwen-image: an artist who put FLUX.3's file there first),
then the artist's HuggingFace caches at the pinned commit (the hub passes them in SPELLCAST_HF_CACHES, as its children's
environment lacks HF_HOME; hard links on the same drive only, after the SHA-256 is checked: nothing is copied). Stdlib
only: it imports the hub's stdlib-only install/models.py, never the hub itself.

An interrupted download is tried again (download_file already retries a few times in a row) until an attempt brings no
new bytes MAX_STALLED_ATTEMPTS times in a row; what arrived stays in <name>.part, and starting it again continues there.
The files go one after the other; one that is there already is kept.

Exit codes (EXIT_*; spellcast/model_downloads.py repeats them): 0 the files are there, 1 the download failed (network;
what arrived is kept), 3 a file arrived in full but is not the pinned one (its SHA-256 differs: a proxy or an antivirus
may be changing it), 4 the drive is full, 5 the folder cannot be written (access denied: an antivirus, Controlled
folder access, a read-only folder); 2 is argparse's, a bad command line. What happened is printed for the hub's log
(code/logs/qwen-full-download.log, flux3-download.log).
"""

from __future__ import annotations

import argparse
import errno
import sys
import time
from pathlib import Path
from typing import Iterator

# code/ at the END of sys.path, as qwen_server.py does: the hub's stdlib-only install/models.py
_CODE = str(Path(__file__).resolve().parents[2])
if _CODE not in sys.path:
    sys.path.append(_CODE)
from spellcast.install import models  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_CHECKSUM, EXIT_DISK_FULL, EXIT_DENIED = 0, 1, 3, 4, 5
RETRY_DELAY = 30.0          # seconds before an interrupted download is tried again
MAX_STALLED_ATTEMPTS = 10   # attempts in a row that brought no new bytes before it gives up
_DISK_FULL_WINERRORS = {112, 39}    # ERROR_DISK_FULL, ERROR_HANDLE_DISK_FULL
QWEN_FOLDER = "qwen-image"          # Qwen-Image's models folder, beside the others (paths.MODELS_QWEN's name)


def say(message: str) -> None:
    print(f"[qwen_download] {message}", flush=True)


def _causes(error: BaseException | None) -> Iterator[BaseException]:
    """The error and those it came from (download_file retries a failed write a few times, then raises an OSError
    of its own from the last one)."""
    while error is not None:
        yield error
        error = error.__cause__ or error.__context__


def disk_full(error: BaseException | None) -> bool:
    """The drive ran full: the error itself, or one it came from."""
    return any(isinstance(e, OSError) and (e.errno == errno.ENOSPC
                                           or getattr(e, "winerror", None) in _DISK_FULL_WINERRORS)
               for e in _causes(error))


def access_denied(error: BaseException | None) -> bool:
    """The folder cannot be written (or the file is held by another program): trying again later changes nothing."""
    return any(isinstance(e, PermissionError) for e in _causes(error))


def part_bytes(dst: Path) -> int:
    try:
        return dst.with_name(dst.name + ".part").stat().st_size
    except OSError:
        return 0


def local_folders(folder: Path) -> list[Path]:
    """Folders on this PC that may hold the files by their names here: each SPELLCAST_MODELS_FROM folder and its
    sub-folders named like `folder` and like Qwen-Image's (another Spellcast3D's models folder), and Qwen-Image's own
    folder beside `folder` (models/qwen-image, for FLUX.3's files)."""
    found: list[Path] = []
    for base in models.extra_sources():
        found += [base, base / folder.name, base / QWEN_FOLDER]
    if folder.name != QWEN_FOLDER:
        found.append(folder.parent / QWEN_FOLDER)
    return list(dict.fromkeys(found))


def link_local_copies(folder: Path, manifest: models.Manifest, sources: dict[str, tuple[str, str, str]]) -> bool:
    """Copies on this PC of the manifest's files that are not in `folder` yet, hard-linked into it (local_folders by
    name, then the artist's HuggingFace caches at each file's pinned commit, under its path in the repo; each checked
    against its SHA-256) -> whether all of them are there now."""
    for src in local_folders(folder):
        missing = models.missing_files(folder, manifest)
        if not missing:
            return True
        models.link_from_local_dir(src, folder, missing, manifest, say)
    for name in models.missing_files(folder, manifest):
        repo, revision, remote = sources[name]
        for snapshot in models.hf_cache_sources(repo, revision):
            if not models.missing_files(folder, {name: manifest[name]}):
                break
            models.link_from_local_dir(snapshot, folder, [name], manifest, say, names={name: remote})
    return not models.missing_files(folder, manifest)


def download_one(folder: Path, name: str, pin: tuple[int, str], source: tuple[str, str, str]) -> int:
    """One file of the manifest from its pinned commit, tried again until attempts stop bringing bytes -> an exit
    code."""
    size, sha = pin
    dst = folder / name
    url = models.hf_url(*source)
    say(f"Downloading {name} ({size / 1e9:.2f} GB) from {url}" + (f", continuing at {part_bytes(dst):,} bytes"
                                                                  if part_bytes(dst) else "") + ".")
    stalled = 0
    while True:
        before = part_bytes(dst)
        try:
            models.download_file(url, dst, size, sha, say)
            say(f"{name} is there ({size:,} bytes, SHA-256 checked).")
            return EXIT_OK
        except models.ChecksumError as e:
            say(f"{e}. A proxy or an antivirus may be changing it.")
            return EXIT_CHECKSUM
        except OSError as e:
            if disk_full(e):
                say(f"The drive is full: {e}")
                return EXIT_DISK_FULL
            if access_denied(e):
                say(f"Writing into {folder} is not allowed: {e}")
                return EXIT_DENIED
            problem = str(e) or type(e).__name__
        stalled = 0 if part_bytes(dst) > before else stalled + 1
        if stalled >= MAX_STALLED_ATTEMPTS:
            say(f"Giving up after {stalled} attempts without progress. Last problem: {problem}")
            return EXIT_FAILED
        say(f"Download interrupted ({problem}). Trying again in {RETRY_DELAY:.0f} s "
            f"(attempt {stalled + 1} of {MAX_STALLED_ATTEMPTS} without progress).")
        time.sleep(RETRY_DELAY)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download an optional model of Spellcast3D's Images stage")
    p.add_argument("--model", choices=models.OPTIONAL_MODELS, default="full",
                   help="full: the full Qwen-Image 2.1 (into models/qwen-image); flux3: FLUX.3 Image Klein (into "
                        "models/flux3-klein); hi3d: Hi3D's AI models (into models/hi3d)")
    p.add_argument("--models", required=True,
                   help="the model's folder (models/qwen-image, models/flux3-klein, models/hi3d)")
    p.add_argument("--quality", required=True, choices=tuple(models.QWEN_DENOISERS), help="the model size")
    args = p.parse_args(argv)
    folder = Path(args.models)
    manifest = models.optional_manifest(args.model, args.quality, folder)
    sources = models.optional_sources(args.model)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        models.check_unverified(folder, manifest, say)      # a file put there by hand: checked once, a wrong one goes
    except OSError as e:
        say(f"Looking at the folder failed ({e}).")
    if not models.missing_files(folder, manifest):
        say(f"{', '.join(manifest)} {'is' if len(manifest) == 1 else 'are'} already there.")
        return EXIT_OK
    try:
        if link_local_copies(folder, manifest, sources):
            say("Everything was on this PC already: linked, nothing downloaded.")
            return EXIT_OK
    except OSError as e:
        say(f"Looking for a copy on this PC failed ({e}); downloading instead.")
    for name in models.missing_files(folder, manifest):
        code = download_one(folder, name, manifest[name], sources[name])
        if code != EXIT_OK:
            return code
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
