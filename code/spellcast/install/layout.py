"""What an artist's copy of Spellcast3D shows in its folder, and keeping it so. STDLIB ONLY.

A release is a git clone (tools\\update.bat updates it with git), so every file the repository tracks would lie in
the folder the artist opens. What only git, GitHub's pages or developers read is left out of the working tree by a
sparse checkout instead: git then reads its own .gitignore, .gitattributes and .gitmodules from its index, and an
update keeps them out. .git stays where git looks for it and carries Windows' Hidden flag, as in any clone made by
Git for Windows. prepare_release.py builds a release this way; tidy() brings a copy unpacked from an earlier release
to the same. A developer's clone has no sparse checkout, or one of its own, and is left alone.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

from .. import paths

# What the repository puts into the folder an artist opens (the tracked entries of its root that they use).
SHOWN: tuple[str, ...] = ("code", "models", "run-browser", "tools",
                          "HOW TO USE - INSTRUCTIONS.txt", "README - LICENSE.txt")
# What a release's working tree leaves out, repo-relative (a folder ends with "/"): the tests, git's own files, and
# what is written for GitHub's pages and for developers (the licence's text is in code/LICENSE too). Still in git,
# so an update never trips on them. Every tracked entry of the root is named here or in SHOWN.
LEAVES_OUT: tuple[str, ...] = ("code/tests/", ".gitattributes", ".gitignore", ".gitmodules",
                               "DEVELOPERS.md", "LICENSE", "README.md", "prepare_release.py")
SPARSE_PATTERNS: tuple[str, ...] = ("/*", *(f"!/{entry}" for entry in LEAVES_OUT))      # git sparse-checkout --no-cone


def left_out(path: str) -> bool:
    """Does a release's working tree leave this repo-relative path out (LEAVES_OUT)."""
    return any(path.startswith(entry) if entry.endswith("/") else path == entry for entry in LEAVES_OUT)


def hidden(path: Path) -> bool:
    """Does it carry Windows' Hidden flag (File Explorer lists it only when asked to show hidden items)."""
    try:
        return bool(getattr(os.stat(path), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_HIDDEN)
    except OSError:
        return False


def hide(path: Path) -> None:
    """Set Windows' Hidden flag on a file or folder that is there (best effort)."""
    if os.name != "nt":
        return
    import ctypes

    try:
        attributes = os.stat(path).st_file_attributes
        if not attributes & stat.FILE_ATTRIBUTE_HIDDEN:
            ctypes.windll.kernel32.SetFileAttributesW(str(path), attributes | stat.FILE_ATTRIBUTE_HIDDEN)
    except OSError:
        pass


def tracked_text(name: str, git: str | None, root: Path | None = None) -> str | None:
    """The text of a file in the checkout's root, e.g. ".gitmodules": the file itself, else (a release leaves it out
    of its folder) what git's index holds for it. None when there is neither, or no git to ask."""
    root = root or paths.ROOT
    try:
        return (root / name).read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    if git is None or not (root / ".git").exists():
        return None
    try:
        done = _git(git, root, "cat-file", "blob", f":{name}")
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def sparse_patterns(root: Path) -> list[str] | None:
    """The patterns of the clone's sparse checkout, or None when it never had one."""
    try:
        text = (root / ".git" / "info" / "sparse-checkout").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def _a_releases(patterns: list[str]) -> bool:
    """Whether a sparse checkout is one a release made, of any day: everything, less entries it names. (A
    developer's own names what it keeps, or leaves out by a pattern.)"""
    return patterns[:1] == ["/*"] and all(line.startswith("!/") and "*" not in line for line in patterns[1:])


def tidy(git: str | None, root: Path | None = None) -> tuple[list[str], str]:
    """Bring a release copy's folder to today's layout: .git hidden, LEAVES_OUT out of the working tree (a file
    edited here stays; git names it).

    -> (the root's entries that are no longer shown, what went wrong or ""). Nothing happens in a copy without git
    data or in a developer's clone; one small file is read when all is in place; it never raises.
    """
    root = root or paths.ROOT
    current = sparse_patterns(root)
    if current is None or not _a_releases(current):
        return [], ""
    hide(root / ".git")
    if git is None or current == list(SPARSE_PATTERNS):
        return [], ""
    try:
        if _git(git, root, "config", "--get", "--bool", "core.sparseCheckout").stdout.strip() != "true":
            return [], ""
        candidates = [entry for entry in LEAVES_OUT if "/" not in entry and (root / entry).exists()]
        done = _git(git, root, "sparse-checkout", "set", "--no-cone", *SPARSE_PATTERNS)
    except (OSError, subprocess.SubprocessError) as e:
        return [], f"git could not be run: {e}"
    gone = [entry for entry in candidates if not (root / entry).exists()]
    said = " ".join((done.stdout + done.stderr).split())
    return gone, (f"git sparse-checkout: {said or f'exit code {done.returncode}'}"
                  if done.returncode != 0 or said else "")


def _git(git: str, root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([git, "-C", str(root), *args], capture_output=True, text=True, errors="replace",
                          stdin=subprocess.DEVNULL, env=paths.clean_env(None), timeout=300,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
