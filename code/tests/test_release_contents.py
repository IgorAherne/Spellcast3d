"""What an artist's copy leaves out (prepare_release.py RELEASE_LEAVES_OUT): the tests. The committed build checks
them out of its clone with a sparse checkout; here a real git (the PATH's, else tools/git, as
prepare_release.git_exe finds it) runs that on a small repo made in the test, then update.bat's steps (fetch
--depth 1, reset --keep @{u}): they stay out, the other files update, git status stays clean. And the developers'
own notes, record of the work and one-off tools (DEVELOPERS_OWN) are not tracked at all: the repository is public,
and a release keeps its .git.
Skipped without git. Run with the hub venv:  python -m pytest -p no:cacheprovider code/tests/test_release_contents.py
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from spellcast import paths


@functools.cache
def _prepare_release() -> Any:
    sys.path.insert(0, str(paths.ROOT))
    try:
        import prepare_release
    finally:
        sys.path.remove(str(paths.ROOT))
    return prepare_release


def test_the_tests_are_left_out_and_profiling_is_not() -> None:
    pr = _prepare_release()
    assert pr.RELEASE_LEAVES_OUT == ("code/tests/",)
    assert pr.left_out("code/tests/test_core.py") and pr.left_out("code/tests/web/images.test.mjs")
    assert not pr.left_out("code/spellcast/profiling/qwen_profiled.py")       # the [profiling] switches start it
    assert not pr.left_out("code/testsuite.txt") and not pr.left_out("code/spellcast/tests.py")
    assert pr.SPARSE_PATTERNS == ("/*", "!/code/tests/")
    assert not any(pr.left_out(path) for path in pr.RELEASE_MUST_HAVE)


def test_the_developers_own_notes_and_tools_are_tracked_by_no_commit() -> None:
    """The repository is public and a release is a clone with its .git: whatever is tracked is published and in every
    artist's copy, checked out or not. So the developers' notes, their record of the work and their one-off tools
    are ignored files beside the code, and `check` fails should git track one."""
    pr = _prepare_release()
    try:
        git_exe = pr.git_exe()
    except pr.ReleaseError:
        pytest.skip("no git")
    assert set(pr.DEVELOPERS_OWN) == {"SUMMARIES.md", "SUMMARIES_detailed.md", "code/ARCHITECTURE.md", "code/devtools/"}
    assert not any(pr.left_out(name) for name in pr.DEVELOPERS_OWN), "not left out of a checkout: not there at all"
    probes = ["SUMMARIES.md", "SUMMARIES_detailed.md", "code/ARCHITECTURE.md", "code/devtools/x.py"]
    assert all((probe, True) in pr.IGNORE_PROBES for probe in probes)

    def git(*args: str) -> str:
        return subprocess.run([git_exe, "-C", str(paths.ROOT), *args], capture_output=True, text=True).stdout

    if not git("rev-parse", "--is-inside-work-tree").strip():
        pytest.skip("not a git checkout")
    assert git("check-ignore", "--no-index", *probes, "code/devtools/pod/sync.sh").split() == [*probes, "code/devtools/pod/sync.sh"]
    assert git("ls-files", "--", *pr.DEVELOPERS_OWN).strip() == "", "tracked again"


def test_a_sparse_release_clone_leaves_them_out_and_still_updates(tmp_path: Path) -> None:
    pr = _prepare_release()
    try:
        git_exe = pr.git_exe()
    except pr.ReleaseError:
        pytest.skip("no git")
    config = tmp_path / "gitconfig"                           # a user config that says autocrlf=false, as
    config.write_text("[core]\n\tautocrlf = false\n[user]\n\tname = t\n\temail = t@t\n", encoding="utf-8")
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": str(config)}  # verify_staging does

    def git(cwd: Path, *args: str, check: bool = True) -> str:
        done = subprocess.run([git_exe, "-C", str(cwd), *args], capture_output=True, text=True, env=env)
        assert done.returncode == 0 or not check, f"git {' '.join(args)}: {done.stdout}{done.stderr}"
        return done.stdout

    def files(folder: Path) -> list[str]:
        return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()
                      and ".git" not in p.relative_to(folder).parts)

    origin = tmp_path / "origin"
    for name, text in {"tools/update.bat": "@echo off\r\n", "code/spellcast/z.py": "z1\n",
                       "code/spellcast/profiling/p.py": "p\n", "code/tests/test_x.py": "t1\n",
                       "code/tests/web/y.test.mjs": "y\n"}.items():
        (origin / name).parent.mkdir(parents=True, exist_ok=True)
        (origin / name).write_bytes(text.encode())
    git(tmp_path, "init", "-b", "main", str(origin))
    git(origin, "add", "-A")
    git(origin, "commit", "-m", "one")

    staging = tmp_path / "staging"                            # as stage_committed clones and thins it
    git(tmp_path, "clone", "--depth", "1", "--config", "core.autocrlf=true", "--branch", "main", origin.as_uri(),
        str(staging))
    git(staging, "sparse-checkout", "set", "--no-cone", *pr.SPARSE_PATTERNS)
    assert files(staging) == ["code/spellcast/profiling/p.py", "code/spellcast/z.py", "tools/update.bat"]
    assert git(staging, "status", "--porcelain", "--ignore-submodules=all") == ""

    (origin / "code/spellcast/z.py").write_text("z2\n", encoding="utf-8")
    (origin / "code/tests/test_x.py").write_text("t2\n", encoding="utf-8")
    (origin / "code/tests/new.py").write_text("n\n", encoding="utf-8")
    git(origin, "add", "-A")
    git(origin, "commit", "-m", "two")
    git(staging, "update-index", "-q", "--refresh", check=False)                 # tools/update.bat's steps
    git(staging, "fetch", "--depth", "1", "--no-recurse-submodules", "origin")
    git(staging, "reset", "--keep", "@{u}")
    assert files(staging) == ["code/spellcast/profiling/p.py", "code/spellcast/z.py", "tools/update.bat"]
    assert (staging / "code/spellcast/z.py").read_text(encoding="utf-8") == "z2\n"
    assert git(staging, "status", "--porcelain", "--ignore-submodules=all") == ""
