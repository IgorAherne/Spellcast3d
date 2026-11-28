"""What an artist's copy shows in its folder and what it leaves out (spellcast/install/layout.py, which
prepare_release.py builds a release by): the tests, git's own files in the root, and what is written for GitHub's
pages and for developers. The committed build leaves them out of its clone with a sparse checkout; here a real git
(the PATH's, else tools/git, as prepare_release.git_exe finds it) runs that on a small repo with a tool as a
submodule, then update.bat's steps (fetch --depth 1, reset --keep @{u}, submodule sync / update): they stay out, git
reads its .gitignore, .gitattributes and .gitmodules from its index, the other files update, git status stays clean.
A copy unpacked from an earlier release is brought to the same folder by the installer (layout.tidy), a developer's
clone is left alone, and the zip keeps .git hidden. And the developers' own notes, record of the work and one-off
tools (DEVELOPERS_OWN) are not tracked at all: the repository is public, and a release keeps its .git.
Skipped without git. Run with the hub venv:  python -m pytest -p no:cacheprovider code/tests/test_release_contents.py
"""

from __future__ import annotations

import ctypes
import functools
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable

import pytest

from spellcast import paths
from spellcast.install import components, layout

Git = Callable[..., str]
ROOT_OF_A_RELEASE = [".git", "HOW TO USE - INSTRUCTIONS.txt", "README - LICENSE.txt", "code", "models", "run-browser",
                     "tools"]
OLD_PATTERNS = ("/*", "!/code/tests/")              # the sparse checkout of the first release
ROOT_FILES_LEFT_OUT = [".gitattributes", ".gitignore", ".gitmodules", "DEVELOPERS.md", "LICENSE", "README.md",
                       "prepare_release.py"]


@functools.cache
def _prepare_release() -> Any:
    sys.path.insert(0, str(paths.ROOT))
    try:
        import prepare_release
    finally:
        sys.path.remove(str(paths.ROOT))
    return prepare_release


def _git_exe() -> str:
    pr = _prepare_release()
    try:
        return pr.git_exe()
    except pr.ReleaseError:
        pytest.skip("no git")


def test_the_tests_are_left_out_and_profiling_is_not() -> None:
    pr = _prepare_release()
    assert pr.RELEASE_LEAVES_OUT is layout.LEAVES_OUT and "code/tests/" in pr.RELEASE_LEAVES_OUT
    assert pr.left_out("code/tests/test_core.py") and pr.left_out("code/tests/web/images.test.mjs")
    assert not pr.left_out("code/spellcast/profiling/qwen_profiled.py")       # the [profiling] switches start it
    assert not pr.left_out("code/testsuite.txt") and not pr.left_out("code/spellcast/tests.py")
    assert pr.SPARSE_PATTERNS == ("/*", *(f"!/{entry}" for entry in pr.RELEASE_LEAVES_OUT))
    assert pr.SPARSE_PATTERNS[:2] == OLD_PATTERNS
    assert not any(pr.left_out(path) for path in pr.RELEASE_MUST_HAVE)


def test_the_folder_shows_what_an_artist_uses_and_every_root_entry_is_decided() -> None:
    """The root's files that only git, GitHub or developers read are left out; the same names deeper are not (a
    tool's own README.md and LICENSE stay). A new file in the repository's root must be put into one of the lists."""
    pr = _prepare_release()
    assert sorted(entry for entry in layout.LEAVES_OUT if "/" not in entry) == ROOT_FILES_LEFT_OUT
    assert sorted([".git", *layout.SHOWN]) == ROOT_OF_A_RELEASE and not set(layout.SHOWN) & set(layout.LEAVES_OUT)
    assert all(pr.left_out(name) for name in ROOT_FILES_LEFT_OUT)
    for kept in ("code/LICENSE", "code/forge/LICENSE", "code/forge/README.md", "code/pixal3d/.gitignore",
                 "README - LICENSE.txt", "HOW TO USE - INSTRUCTIONS.txt", "tools/update.bat", "models/README.txt"):
        assert not pr.left_out(kept), kept
    assert pr.undecided_root_entries(["code/a.py", "LICENSE", "NEW.md", "notes/x.txt", "tools/update.bat"]) == ["NEW.md",
                                                                                                             "notes"]
    assert pr.LICENSE_COPY == "code/LICENSE" and pr.LICENSE_COPY in pr.RELEASE_MUST_HAVE
    assert "LICENSE" not in pr.RELEASE_MUST_HAVE
    assert not pr.license_copy_differs(paths.ROOT), "code/LICENSE must be a copy of LICENSE"
    license_note = (paths.ROOT / "README - LICENSE.txt").read_text(encoding="utf-8")
    assert "Full text: code\\LICENSE" in license_note and "(in this folder)" not in license_note

    done = subprocess.run([_git_exe(), "-C", str(paths.ROOT), "ls-files", "-z"], capture_output=True, text=True)
    if done.returncode != 0:
        pytest.skip("not a git checkout")
    assert pr.undecided_root_entries(p for p in done.stdout.split("\0") if p) == []


def test_the_developers_own_notes_and_tools_are_tracked_by_no_commit() -> None:
    """The repository is public and a release is a clone with its .git: whatever is tracked is published and in every
    artist's copy, checked out or not. So the developers' notes, their record of the work and their one-off tools
    are ignored files beside the code, and `check` fails should git track one."""
    pr = _prepare_release()
    git_exe = _git_exe()
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


# ----------------------------------------------------------------------------- a small release, made with a real git

@pytest.fixture
def git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Git:
    """git -C <folder> ... under a user config that says autocrlf=false (as verify_staging checks a release) and lets
    a submodule be fetched from a folder of this PC. layout.tidy's own git calls get the same config."""
    git_exe = _git_exe()
    config = tmp_path / "gitconfig"
    config.write_text("[core]\n\tautocrlf = false\n[user]\n\tname = t\n\temail = t@t\n"
                      "[protocol \"file\"]\n\tallow = always\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))

    def run(cwd: Path, *args: str, check: bool = True) -> str:
        done = subprocess.run([git_exe, "-C", str(cwd), *args], capture_output=True, text=True)
        assert done.returncode == 0 or not check, f"git {' '.join(args)}: {done.stdout}{done.stderr}"
        return done.stdout

    return run


def _write(folder: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(text.encode())


def _origin(tmp_path: Path, git: Git) -> Path:
    """A repository laid out like Spellcast3D's, with one tool as a submodule (code/tool)."""
    tool = tmp_path / "tool"
    _write(tool, {"launch.py": "v1\n", "README.md": "the tool's own\n", "LICENSE": "the tool's own\n"})
    git(tmp_path, "init", "-b", "main", str(tool))
    git(tool, "add", "-A")
    git(tool, "commit", "-m", "tool one")
    origin = tmp_path / "origin"
    _write(origin, {
        ".gitattributes": "*.bat text eol=crlf\n", ".gitignore": "/projects/\n/spellcast3d-config.ini\n",
        "README.md": "for GitHub\n", "DEVELOPERS.md": "d\n", "LICENSE": "agpl\n", "prepare_release.py": "pass\n",
        "HOW TO USE - INSTRUCTIONS.txt": "how\n", "README - LICENSE.txt": "licenses\n",
        "run-browser/run.bat": "@echo off\n", "tools/update.bat": "@echo off\n", "models/README.txt": "m\n",
        "code/LICENSE": "agpl\n", "code/spellcast/z.py": "z1\n", "code/spellcast/profiling/p.py": "p\n",
        "code/tests/test_x.py": "t1\n", "code/tests/web/y.test.mjs": "y\n"})
    git(tmp_path, "init", "-b", "main", str(origin))
    git(origin, "submodule", "add", tool.as_uri(), "code/tool")
    git(origin, "add", "-A")
    git(origin, "commit", "-m", "one")
    return origin


def _release(tmp_path: Path, git: Git, origin: Path, patterns: tuple[str, ...], name: str = "staging") -> Path:
    """As stage_committed clones and thins it."""
    staging = tmp_path / name
    git(tmp_path, "clone", "--depth", "1", "--config", "core.autocrlf=true", "--branch", "main", origin.as_uri(),
        str(staging))
    git(staging, "-c", "core.autocrlf=true", "submodule", "update", "--init", "--depth", "1")
    git(staging / "code/tool", "config", "core.autocrlf", "true")
    git(staging, "sparse-checkout", "set", "--no-cone", *patterns)
    return staging


def _update(git: Git, staging: Path) -> None:
    """tools/update.bat's git steps."""
    git(staging, "update-index", "-q", "--refresh", check=False)
    git(staging, "fetch", "--depth", "1", "--no-recurse-submodules", "origin")
    git(staging, "reset", "--keep", "@{u}")
    git(staging, "submodule", "sync")
    git(staging, "submodule", "update", "--init", "--depth", "1")


def _files(folder: Path) -> list[str]:
    """The superproject's own files (not git's data, not the tools')."""
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()
                  and not {".git", "tool", "second"} & set(p.relative_to(folder).parts))


def _root(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def _status(git: Git, folder: Path) -> str:
    return git(folder, "status", "--porcelain")


FILES_OF_A_RELEASE = ["HOW TO USE - INSTRUCTIONS.txt", "README - LICENSE.txt", "code/LICENSE",
                      "code/spellcast/profiling/p.py", "code/spellcast/z.py", "models/README.txt",
                      "run-browser/run.bat", "tools/update.bat"]


def test_a_sparse_release_clone_leaves_them_out_and_still_updates(tmp_path: Path, git: Git,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    pr = _prepare_release()
    origin = _origin(tmp_path, git)
    staging = _release(tmp_path, git, origin, pr.SPARSE_PATTERNS)
    assert _root(staging) == ROOT_OF_A_RELEASE
    assert _files(staging) == FILES_OF_A_RELEASE
    assert (staging / "code/tool/README.md").is_file() and (staging / "code/tool/LICENSE").is_file()
    assert (staging / "tools/update.bat").read_bytes() == b"@echo off\r\n"       # .gitattributes, read from the index
    _write(staging, {"projects/a/project.json": "{}", "spellcast3d-config.ini": "[image]\n"})
    assert _status(git, staging) == "", "the .gitignore in git's index is not read"
    # The installer asks which tools are submodules: the file is not in the folder, git's index has it.
    assert "path = code/tool" in layout.tracked_text(".gitmodules", _git_exe(), staging)
    assert layout.tracked_text(".gitmodules", None, staging) is None
    assert layout.tracked_text(".gitmodules", _git_exe(), origin) == (origin / ".gitmodules").read_text(encoding="utf-8")
    assert layout.tracked_text("no-such-file", _git_exe(), staging) is None
    assert layout.tracked_text(".gitmodules", _git_exe(), tmp_path / "nowhere") is None
    monkeypatch.setattr(paths, "ROOT", staging)
    monkeypatch.setattr(paths, "HI3D_DIR", staging / "code" / "tool")       # (as Hi3D's code is one of them)
    monkeypatch.setattr(components, "git_exe", _git_exe)
    assert components.hi3d_submodule()
    monkeypatch.setattr(paths, "HI3D_DIR", staging / "code" / "other")
    assert not components.hi3d_submodule()

    # The next version changes git's own files, the left-out ones, a tool's pin and where the tool comes from,
    # and brings a second tool.
    tool = tmp_path / "tool"
    _write(tool, {"launch.py": "v2\n"})
    git(tool, "commit", "-am", "tool two")
    moved = tmp_path / "tool-moved"
    git(tmp_path, "clone", "--bare", str(tool), str(moved))
    git(origin / "code/tool", "pull", "-q", "origin", "main")
    git(origin, "submodule", "add", tool.as_uri(), "code/second")
    _write(origin, {
        ".gitattributes": "*.bat text eol=crlf\n*.cmd text eol=crlf\n", ".gitignore": "/projects/\n/spellcast3d-config.ini\n/exports/\n",
        ".gitmodules": (origin / ".gitmodules").read_text(encoding="utf-8").replace(tool.as_uri(), moved.as_uri()),
        "README.md": "for GitHub, edited\n", "LICENSE": "agpl 2\n", "code/LICENSE": "agpl 2\n", "prepare_release.py": "print()\n",
        "tools/update.bat": "@echo off\nrem two\n", "tools/new.cmd": "@echo off\n",
        "code/spellcast/z.py": "z2\n", "code/tests/test_x.py": "t2\n", "code/tests/new.py": "n\n"})
    git(origin, "add", "-A")
    git(origin, "commit", "-m", "two")
    _update(git, staging)

    assert _root(staging) == sorted([*ROOT_OF_A_RELEASE, "projects", "spellcast3d-config.ini"])
    assert _files(staging) == sorted([*FILES_OF_A_RELEASE, "projects/a/project.json", "spellcast3d-config.ini",
                                      "tools/new.cmd"])
    assert (staging / "code/spellcast/z.py").read_text(encoding="utf-8") == "z2\n"
    assert (staging / "code/LICENSE").read_text(encoding="utf-8") == "agpl 2\n"
    assert (staging / "tools/update.bat").read_bytes() == b"@echo off\r\nrem two\r\n"
    assert (staging / "tools/new.cmd").read_bytes() == b"@echo off\r\n", "the new .gitattributes line is not read"
    assert (staging / "code/tool/launch.py").read_text(encoding="utf-8") == "v2\n", "the tool did not move to its pin"
    assert git(staging, "config", "--get", "submodule.code/tool.url").strip() == moved.as_uri()
    assert (staging / "code/second/launch.py").is_file(), "a tool the update brought was not checked out"
    assert "path = code/second" in layout.tracked_text(".gitmodules", _git_exe(), staging)
    _write(staging, {"exports/P/a.glb": "x"})
    assert _status(git, staging) == "", "the new .gitignore line is not read"


def _show(path: Path) -> None:
    """Take Windows' Hidden flag off, as an unzip program that drops it leaves .git."""
    ctypes.windll.kernel32.SetFileAttributesW(str(path), os.stat(path).st_file_attributes & ~stat.FILE_ATTRIBUTE_HIDDEN)


def test_a_copy_of_an_earlier_release_gets_todays_folder_and_a_developers_clone_is_left_alone(
        tmp_path: Path, git: Git) -> None:
    git_exe = _git_exe()
    origin = _origin(tmp_path, git)
    old = _release(tmp_path, git, origin, OLD_PATTERNS)
    assert _root(old) == sorted([*ROOT_OF_A_RELEASE, *ROOT_FILES_LEFT_OUT])
    if os.name == "nt":
        _show(old / ".git")
        assert not layout.hidden(old / ".git")

    gone, problem = layout.tidy(git_exe, old)
    assert (sorted(gone), problem) == (ROOT_FILES_LEFT_OUT, "")
    assert _root(old) == ROOT_OF_A_RELEASE and _files(old) == FILES_OF_A_RELEASE
    assert layout.sparse_patterns(old) == list(layout.SPARSE_PATTERNS)
    assert layout.hidden(old / ".git") or os.name != "nt"
    assert _status(git, old) == ""
    assert layout.tidy(git_exe, old) == ([], "") and layout.tidy(None, old) == ([], "")
    _write(origin, {"code/spellcast/z.py": "z2\n", "README.md": "edited\n"})        # and it still updates
    git(origin, "commit", "-am", "two")
    _update(git, old)
    assert _root(old) == ROOT_OF_A_RELEASE and (old / "code/spellcast/z.py").read_text(encoding="utf-8") == "z2\n"

    # A file the artist edited is not taken away: git leaves it and says so; nothing fails.
    edited = _release(tmp_path, git, origin, OLD_PATTERNS, "edited")
    (edited / "README.md").write_text("my notes\n", encoding="utf-8")
    gone, problem = layout.tidy(git_exe, edited)
    assert "README.md" not in gone and "README.md" in problem and len(gone) == len(ROOT_FILES_LEFT_OUT) - 1
    assert (edited / "README.md").read_text(encoding="utf-8") == "my notes\n"

    # A developer's clone has no sparse checkout: every file stays, whatever .git's flag is.
    clone = tmp_path / "clone"
    git(tmp_path, "clone", origin.as_uri(), str(clone))
    before = _root(clone)
    if os.name == "nt":
        _show(clone / ".git")
    assert layout.tidy(git_exe, clone) == ([], "") and _root(clone) == before and "README.md" in before
    assert not layout.hidden(clone / ".git")
    git(clone, "sparse-checkout", "set", "--no-cone", "/*", "!/code/tests/")         # ... also one who tried it
    git(clone, "sparse-checkout", "disable")
    assert layout.tidy(git_exe, clone) == ([], "") and _root(clone) == before
    git(clone, "sparse-checkout", "set", "--cone", "code")                           # ... or keeps one of their own
    if os.name == "nt":
        _show(clone / ".git")
    own, kept = layout.sparse_patterns(clone), _root(clone)
    assert "code" in kept and "tools" not in kept
    assert layout.tidy(git_exe, clone) == ([], "") and (layout.sparse_patterns(clone), _root(clone)) == (own, kept)
    assert not layout.hidden(clone / ".git")
    assert layout.tidy(git_exe, tmp_path / "tool") == ([], "") and layout.tidy(git_exe, tmp_path / "nowhere") == ([], "")


@pytest.mark.skipif(os.name != "nt", reason="Windows' Hidden flag")
def test_the_zip_keeps_what_is_hidden_hidden(tmp_path: Path) -> None:
    """File Explorer, 7-Zip and WinRAR hand a zip entry's Hidden flag on to the unpacked file or folder. Everything is
    inside the one folder spellcast3d: unpacked straight onto a drive, the zip leaves one folder there."""
    pr = _prepare_release()
    src = tmp_path / "src"
    _write(src, {".git/HEAD": "ref: refs/heads/main\n", "code/a.py": "a\n", "code/tool/.git": "gitdir: ../../.git/modules/tool\n",
                 "HOW TO USE - INSTRUCTIONS.txt": "how\n"})
    (src / ".git/refs/tags").mkdir(parents=True)
    (src / "models/checkpoints").mkdir(parents=True)
    layout.hide(src / ".git")
    layout.hide(src / "code/tool/.git")
    assert layout.hidden(src / ".git") and not layout.hidden(src / "code")
    pr.make_zip(src, tmp_path / "out.zip")
    with zipfile.ZipFile(tmp_path / "out.zip") as z:
        names = z.namelist()
        assert pr.ZIP_ROOT == "spellcast3d" and all(name.startswith("spellcast3d/") for name in names)
        attributes = {info.filename.removeprefix("spellcast3d/"): info.external_attr & 0xFF for info in z.infolist()}
        assert z.read("spellcast3d/.git/HEAD") == b"ref: refs/heads/main\n"
    assert attributes[".git/"] == pr.ZIP_FOLDER | pr.ZIP_HIDDEN
    assert attributes["code/tool/.git"] & pr.ZIP_HIDDEN
    assert attributes[".git/refs/tags/"] == pr.ZIP_FOLDER and attributes["models/checkpoints/"] == pr.ZIP_FOLDER
    assert not attributes[".git/HEAD"] & pr.ZIP_HIDDEN and not attributes["code/a.py"] & pr.ZIP_HIDDEN
    assert "code/" not in attributes and "HOW TO USE - INSTRUCTIONS.txt" in attributes
