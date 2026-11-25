"""Pixal3D's multi-view models downloaded into Spellcast3D's models folder, in a child process of the hub.

    <pixal3d venv>\\Scripts\\python.exe mv_download.py --cache-dir <models/pixal3d/hub> --revision <commit>
        [--resume-over-https BYTES] <file> [<file> ...]

The hub (spellcast/multiview.py) starts it in the Pixal3D venv, which has huggingface_hub (and hf_xet); the
hub's own venv stays light. It never imports the hub. The files are paths inside TencentARC/Pixal3D at that
commit (never "main": that would move the cache's refs/main to a newer commit). Interrupted downloads are
retried, and continued where they stopped: huggingface_hub keeps <blob>.incomplete files, and a big one is
continued over plain HTTPS (an Xet download starts the file again from its first byte).

Exit codes: 0 every file is there, 1 the download failed (network; what arrived is kept), 3 the drive is
full, 4 huggingface_hub is missing from the venv. What happened is printed for the hub's log.
"""

from __future__ import annotations

import argparse
import errno
import sys
import time
from pathlib import Path

REPO = "TencentARC/Pixal3D"
EXIT_OK, EXIT_FAILED, EXIT_DISK_FULL, EXIT_NO_HF = 0, 1, 3, 4
RETRY_DELAY = 30.0          # seconds before an interrupted download is tried again
MAX_STALLED_ATTEMPTS = 10   # attempts in a row that brought no new bytes before it gives up
_DISK_FULL_WINERRORS = {112, 39}    # ERROR_DISK_FULL, ERROR_HANDLE_DISK_FULL


def say(message: str) -> None:
    print(f"[mv_download] {message}", flush=True)


def blob_bytes(cache_dir: Path) -> int:
    """Bytes of the repo's blobs, the unfinished ones too: whether an attempt got anywhere."""
    folder = cache_dir / ("models--" + REPO.replace("/", "--")) / "blobs"
    total = 0
    for entry in folder.glob("*") if folder.is_dir() else ():
        try:
            total += entry.stat().st_size
        except OSError:
            pass
    return total


def partial_bytes(cache_dir: Path) -> int:
    folder = cache_dir / ("models--" + REPO.replace("/", "--")) / "blobs"
    total = 0
    for entry in folder.glob("*.incomplete") if folder.is_dir() else ():
        try:
            total += entry.stat().st_size
        except OSError:
            pass
    return total


def missing(snapshot: Path, files: list[str]) -> list[str]:
    """The files not (or only empty) in the snapshot folder."""
    def size(rel: str) -> int:
        try:
            return (snapshot / rel).stat().st_size
        except OSError:
            return 0
    return [rel for rel in files if not size(rel)]


def disk_full(error: BaseException) -> bool:
    while error is not None:
        if isinstance(error, OSError) and (error.errno == errno.ENOSPC
                                           or getattr(error, "winerror", None) in _DISK_FULL_WINERRORS):
            return True
        error = error.__cause__ or error.__context__
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download Pixal3D's multi-view models")
    p.add_argument("--cache-dir", required=True, help="the HuggingFace cache: models/pixal3d/hub")
    p.add_argument("--revision", required=True, help="the commit to download (the cache's refs/main)")
    p.add_argument("--resume-over-https", type=int, default=200_000_000,
                   help="continue over plain HTTPS once unfinished files are this big (bytes)")
    p.add_argument("files", nargs="+", help="files of the repo to download")
    args = p.parse_args(argv)
    cache_dir = Path(args.cache_dir)

    try:
        import huggingface_hub
        from huggingface_hub import constants, snapshot_download
    except ImportError as e:
        say(f"huggingface_hub is not installed in this Python ({e}).")
        return EXIT_NO_HF

    snapshot = cache_dir / ("models--" + REPO.replace("/", "--")) / "snapshots" / args.revision
    say(f"Downloading {len(args.files)} files of {REPO} at {args.revision} into {cache_dir} "
        f"(huggingface_hub {huggingface_hub.__version__}).")
    stalled = 0
    while True:
        if partial_bytes(cache_dir) >= args.resume_over_https and not constants.HF_HUB_DISABLE_XET:
            say("Continuing the unfinished files over HTTPS, where they stopped.")
            constants.HF_HUB_DISABLE_XET = True    # read at each download, not only at import
        before = blob_bytes(cache_dir)
        try:
            # max_workers=1: one file at a time, so a slow line finishes files instead of starting all four.
            snapshot_download(REPO, revision=args.revision, cache_dir=str(cache_dir), allow_patterns=args.files,
                              max_workers=1)
            # Offline, snapshot_download returns the snapshot folder without downloading anything.
            left = missing(snapshot, args.files)
            if not left:
                say("All files are there.")
                return EXIT_OK
            problem = f"still missing after the download: {', '.join(left)} (is the internet connection down?)"
        except Exception as e:          # noqa: BLE001 - every failure is retried or reported
            if disk_full(e):
                say(f"The drive is full: {e}")
                return EXIT_DISK_FULL
            problem = f"{type(e).__name__}: {e}"
        stalled = 0 if blob_bytes(cache_dir) > before else stalled + 1
        if stalled >= MAX_STALLED_ATTEMPTS:
            say(f"Giving up after {stalled} attempts without progress. Last problem: {problem}")
            return EXIT_FAILED
        say(f"Download interrupted ({problem}). Trying again in {RETRY_DELAY:.0f} s "
            f"(attempt {stalled + 1} of {MAX_STALLED_ATTEMPTS} without progress).")
        time.sleep(RETRY_DELAY)


if __name__ == "__main__":
    sys.exit(main())
