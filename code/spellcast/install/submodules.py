"""TriFlow's nested git submodules, and the check that its files are exactly those of its commits. STDLIB ONLY.

TriFlow (code/triflow) is licensed under the ADPNCL 1.0, which asks that any changed or added source file be
documented and sent to AUDI AG. Spellcast3D therefore changes nothing in its tree and adds nothing to it:

- its nested submodules (TRELLIS with its FlexiCubes, Direct3D-S2, pyfqmr-triflow) are checked out one level at a
  time, never with --recursive, at depth 1 (only the pinned commits);
- core.autocrlf=false is used for that checkout (git -c, which the submodule commands pass on) and then set in
  each of these repos, so git writes every file exactly as committed (Git for Windows' own default, true, would
  turn LF into CRLF);
- byte_identity_problems() proves it: every tracked file hashed as it is on disk (git hash-object --no-filters)
  equals its blob, nested checkouts included. Only the three weight files the installer hard-links into
  checkpoints/ (ignored by TriFlow's own checkpoints/.gitignore) may be there besides.

The installer (components.Triflow.prepare) and prepare_release.py check out the same way. tools/update.bat moves
code/triflow itself (core.autocrlf=false) and leaves the nested checkouts to the installer it runs next, whose
preflight says when the folder's path is too long for them (a failure there stops only TriFlow).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Callable

from .. import paths

# (folder inside code/triflow, its submodules to check out), in order: TRELLIS itself is a checkout of level one.
TRIFLOW_NESTED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("", ("third_party/TRELLIS", "third_party/Direct3D-S2", "third_party/pyfqmr-triflow")),
    ("third_party/TRELLIS", ("trellis/representations/mesh/flexicubes",)),
)
# A file every nested checkout has: without it, that checkout is missing or incomplete.
TRIFLOW_LANDMARKS: dict[str, str] = {
    "third_party/TRELLIS": "trellis/__init__.py",
    "third_party/Direct3D-S2": "direct3d_s2/pipeline.py",
    "third_party/pyfqmr-triflow": "setup.py",
    "third_party/TRELLIS/trellis/representations/mesh/flexicubes": "flexicubes.py",
}
# The only files that may be in TriFlow's tree besides its own: the weights, hard links to models/triflow.
TRIFLOW_WEIGHT_FILES: tuple[str, ...] = ("flow_model.safetensors", "sdf_vae.safetensors", "nvv_vae.safetensors")
TRIFLOW_ALLOWED_EXTRA: tuple[str, ...] = tuple(f"checkpoints/{name}" for name in TRIFLOW_WEIGHT_FILES)

# The longest path git writes for TriFlow, below the Spellcast3D folder: the data of its innermost checkout
# (FlexiCubes, inside TRELLIS). Windows cannot write paths of 260 characters or more there: git's index-pack failed
# at 283 even with core.longpaths=true. So a git copy whose folder path is longer than TRIFLOW_MAX_ROOT_LEN cannot
# download TriFlow's parts (preflight blocks TriFlow then, and says why).
TRIFLOW_GIT_DEEPEST = (".git\\modules\\triflow\\modules\\third_party\\TRELLIS\\modules\\trellis\\representations\\mesh"
                       "\\flexicubes\\objects\\pack\\pack-" + "0" * 40 + ".promisor")
TRIFLOW_MAX_ROOT_LEN = 259 - 1 - len(TRIFLOW_GIT_DEEPEST)      # 94

Run = Callable[[list[str], Path], None]      # (argv, working folder): runs a command that changes something


def triflow_nested_dirs(triflow: Path | None = None) -> list[Path]:
    """Every nested checkout of TriFlow, parents first."""
    base = triflow or paths.TRIFLOW_DIR
    return [base / level / sub for level, subs in TRIFLOW_NESTED for sub in subs]


def triflow_missing(triflow: Path | None = None) -> list[str]:
    """TriFlow's nested checkouts (relative to code/triflow) that are missing or incomplete."""
    base = triflow or paths.TRIFLOW_DIR
    return [rel for rel, landmark in TRIFLOW_LANDMARKS.items() if not (base / rel / landmark).is_file()]


def triflow_heads(triflow: Path | None = None) -> dict[str, str]:
    """{nested checkout: its HEAD commit} (read from git's files), for TriFlow's install fingerprint."""
    from .state import git_head

    base = triflow or paths.TRIFLOW_DIR
    return {d.relative_to(base).as_posix(): git_head(d) or "missing" for d in triflow_nested_dirs(base)}


def _checkouts_to_do(git: str, repo: Path, level: str, subs: tuple[str, ...]) -> list[str]:
    """Of `subs` (nested checkouts of `repo`, at `level` inside code/triflow), those missing or not at the commit
    `repo` pins."""
    from .state import git_head

    return [sub for sub in subs
            if not (repo / sub / TRIFLOW_LANDMARKS[(Path(level) / sub).as_posix()]).is_file()
            or git_head(repo / sub) != pinned_commit(git, repo, sub)]


def triflow_checkout_work(git: str | None, triflow: Path | None = None) -> list[str]:
    """What init_triflow would still download with git (relative to code/triflow; "." = TriFlow itself). Without git
    (a release copy has no git data), only what is missing."""
    base = triflow or paths.TRIFLOW_DIR
    if not (base / "inference.py").is_file():
        return ["."]
    if git is None or not (base / ".git").exists():
        return triflow_missing(base)
    work: list[str] = []
    for level, subs in TRIFLOW_NESTED:
        repo = base / level if level else base
        work += [(Path(level) / sub).as_posix() for sub in _checkouts_to_do(git, repo, level, subs)]
    return work


def init_triflow(git: str, run: Run, triflow: Path | None = None, everything: bool = False) -> None:
    """Check out TriFlow's nested submodules byte-identically (see the module docstring).

    Level by level, only the checkouts that are missing or not at the commit their parent pins (everything=True:
    all of them, e.g. for a dry run that only prints the commands). code/triflow itself must be checked out.
    """
    base = triflow or paths.TRIFLOW_DIR
    run([git, "-C", str(base), "config", "core.autocrlf", "false"], base)
    for level, subs in TRIFLOW_NESTED:
        repo = base / level if level else base
        todo = list(subs) if everything else _checkouts_to_do(git, repo, level, subs)
        if todo:
            # Never --recursive: each level is named explicitly. --depth 1: only the pinned commit (TRELLIS'
            # whole history is 1.1 GB).
            run([git, "-c", "core.autocrlf=false", "-C", str(repo), "submodule", "update", "--init", "--depth", "1",
                 "--", *todo], repo)
        for sub in subs:
            if everything or (repo / sub / ".git").exists():
                run([git, "-C", str(repo / sub), "config", "core.autocrlf", "false"], repo)


def pinned_commit(git: str, repo: Path, sub: str) -> str | None:
    """The commit `repo` pins for its submodule at `sub` (its index entry), or None."""
    out = _git(git, repo, "ls-files", "-s", "--", sub).stdout.decode("utf-8", "replace").split()
    return out[1] if len(out) >= 2 and out[0] == "160000" else None


def _git(git: str, repo: Path, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([git, "-C", str(repo), *args],
                          input=None if stdin is None else stdin.encode("utf-8", "surrogateescape"),
                          capture_output=True, stdin=None if stdin is not None else subprocess.DEVNULL,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def git_blob_sha1(data: bytes) -> str:
    """What `git hash-object --no-filters` says for these bytes in a SHA-1 repository."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def byte_identity_problems(git: str, repo: Path, allowed_extra: tuple[str, ...] = (), prefix: str = "",
                           crlf_only: list[Path] | None = None) -> list[str]:
    """Why the checkout at `repo` is not byte-identical to its commit, nested checkouts included ([] = it is).

    - the index must equal HEAD, and every tracked file hashed as it is on disk (git hash-object --no-filters)
      its blob; a nested checkout's HEAD must be the commit its parent pins, and it is checked the same way;
    - no other files, ignored ones included (e.g. __pycache__), except the paths in allowed_extra.
    crlf_only, when given, collects the files that differ only by CRLF line endings (restorable by a checkout
    with core.autocrlf=false).
    """
    name = prefix.rstrip("/") or repo.name
    if not (repo / ".git").exists():
        return [f"{name}: not checked out"]
    problems: list[str] = []
    if _git(git, repo, "diff-index", "--cached", "--quiet", "HEAD", "--").returncode != 0:
        problems.append(f"{name}: git's index differs from its commit (staged changes)")
    listed = _git(git, repo, "ls-files", "-s", "-z")
    if listed.returncode != 0:
        return problems + [f"{name}: git ls-files failed: {listed.stderr.decode('utf-8', 'replace').strip()}"]
    files: list[tuple[str, str]] = []           # (path, blob)
    nested: list[tuple[str, str]] = []          # (path, pinned commit)
    for record in listed.stdout.decode("utf-8", "surrogateescape").split("\0"):
        meta, _, path = record.partition("\t")
        fields = meta.split()
        if not fields:
            continue
        (nested if fields[0] == "160000" else files).append((path, fields[1]))
    present = [(p, sha) for p, sha in files if os.path.lexists(repo / p)]
    problems += [f"{prefix}{p}: missing" for p, _ in files if not os.path.lexists(repo / p)]
    if present:
        hashed = _git(git, repo, "hash-object", "--no-filters", "--stdin-paths",
                      stdin="".join(p + "\n" for p, _ in present))
        got = hashed.stdout.decode("ascii", "replace").split()
        if hashed.returncode != 0 or len(got) != len(present):
            problems.append(f"{name}: git hash-object failed: {hashed.stderr.decode('utf-8', 'replace').strip()}")
        else:
            for (p, want), sha in zip(present, got):
                if sha == want:
                    continue
                problems.append(f"{prefix}{p}: differs from its commit")
                if crlf_only is not None:
                    try:
                        if git_blob_sha1((repo / p).read_bytes().replace(b"\r\n", b"\n")) == want:
                            crlf_only.append(repo / p)
                    except OSError:
                        pass
    extra = _git(git, repo, "ls-files", "-o", "-z", "--directory")        # untracked, ignored ones too
    for p in extra.stdout.decode("utf-8", "surrogateescape").split("\0"):
        if p and p.rstrip("/") not in allowed_extra and p.rstrip("/") not in {n for n, _ in nested}:
            problems.append(f"{prefix}{p}: added (not part of its commit)")
    from .state import git_head
    for p, pinned in nested:
        sub = repo / p
        head = git_head(sub)
        if head is None:
            problems.append(f"{prefix}{p}: not checked out")
            continue
        if head != pinned:
            problems.append(f"{prefix}{p}: at {head[:7]}, but {name} pins {pinned[:7]}")
        inner = tuple(a[len(p) + 1:] for a in allowed_extra if a.startswith(p + "/"))
        problems += byte_identity_problems(git, sub, inner, f"{prefix}{p}/", crlf_only)
    return problems
