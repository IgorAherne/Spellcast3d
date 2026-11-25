"""code/venvs/install-state.json (schema in paths.py) and the fingerprints stored in it. STDLIB ONLY.

A fingerprint is a sha256 over everything that decides what ends up in a venv
(requirement files, wheel bytes, submodule commit, pins, installer version / recipe). When
it differs from the stored one, the component is installed again.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from .. import paths

INSTALLED = "installed"
UNSUPPORTED = "unsupported"
FAILED = "failed"
NOT_INSTALLED = "not_installed"


def load() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(paths.INSTALL_STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(data: dict[str, Any]) -> None:
    # The hub may be reading the file at this very moment; write_text_atomic retries the rename then.
    paths.write_text_atomic(paths.INSTALL_STATE, json.dumps(data, indent=2) + "\n")


def record(tool_id: str, status: str, reason: str, fingerprint: str) -> None:
    data = load()
    data[tool_id] = {
        "status": status,
        "reason": reason,
        "fingerprint": fingerprint,
        "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _save(data)


def note(key: str, value: Any) -> None:
    """Keep the installer's own bookkeeping `value` under `key` beside the tools' entries (paths.py), None drops it."""
    data = load()
    if value is None:
        data.pop(key, None)
    else:
        data[key] = value
    _save(data)


def fingerprint(parts: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except OSError:
        return "missing"


def source_version(repo: Path, fallback_files: tuple[str, ...]) -> str:
    """The submodule's checked-out commit, or (copies without git data) a hash of its key files."""
    head = git_head(repo)
    if head:
        return head
    return "files:" + fingerprint({name: file_sha256(repo / name) for name in fallback_files})


def git_head(repo: Path) -> str | None:
    """HEAD commit of a checkout, read straight from its git files (no git executable needed)."""
    dotgit = repo / ".git"
    try:
        if dotgit.is_file():  # submodules and worktrees: "gitdir: <path>"
            text = dotgit.read_text(encoding="utf-8").strip()
            if not text.startswith("gitdir:"):
                return None
            gitdir = (repo / text[len("gitdir:"):].strip()).resolve()
        elif dotgit.is_dir():
            gitdir = dotgit
        else:
            return None
        head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or None
        ref = head[len("ref:"):].strip()
        common = gitdir
        if (gitdir / "commondir").is_file():
            common = (gitdir / (gitdir / "commondir").read_text(encoding="utf-8").strip()).resolve()
        for base in (gitdir, common):
            if (base / ref).is_file():
                return (base / ref).read_text(encoding="utf-8").strip() or None
        packed = common / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                sha, _, name = line.partition(" ")
                if name.strip() == ref:
                    return sha
    except OSError:
        return None
    return None
