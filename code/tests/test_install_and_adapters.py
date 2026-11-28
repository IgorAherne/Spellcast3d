"""Installer, adapter and launcher logic that needs no tool process: fingerprints, git_head, preflight path
checks, venv repair, the interrupted-install marker, the models folder (Pixal3D's junction and its migration),
the progress file, Forge / Instant Meshes helpers, the launchers' guards.

Fast (no network, no GPU). Run with the hub venv:  python -m pytest -p no:cacheprovider code/tests
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import msvcrt
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from spellcast import config, forge_config, gpu, paths
from spellcast.adapters import forge, imesh, lato2, pixal3d, triflow
from spellcast.adapters.base import StageContext, StageError
from spellcast.install import components, models, preflight, progress, state, venvs
from spellcast.install import main as install_main
from spellcast.install.main import Pending
from spellcast.install.runner import Ctx, InstallError, Output

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Spellcast3D is Windows-only")

if str(paths.LAUNCHERS) not in sys.path:     # the launchers import their guard as a sibling module
    sys.path.insert(0, str(paths.LAUNCHERS))


def _ansi_has(text: str) -> bool:
    try:
        text.encode("mbcs", "strict")
    except UnicodeEncodeError:
        return False
    return True


needs_non_ansi = pytest.mark.skipif(_ansi_has("Łukasz"), reason="this PC's ANSI code page has these letters")


# ---- install/state.py and fingerprints ------------------------------------------------------------

def test_git_head_follows_gitdir_file_and_packed_refs(tmp_path: Path) -> None:
    modules = tmp_path / "super" / ".git" / "modules" / "tool"
    modules.mkdir(parents=True)
    (modules / "HEAD").write_text("ref: refs/heads/main\n")
    (modules / "packed-refs").write_text("# pack-refs\n" + "a" * 40 + " refs/heads/main\n")
    repo = tmp_path / "super" / "code" / "tool"
    repo.mkdir(parents=True)
    (repo / ".git").write_text("gitdir: ../../.git/modules/tool\n")
    assert state.git_head(repo) == "a" * 40
    (modules / "HEAD").write_text("b" * 40 + "\n")              # detached HEAD (what submodules use)
    assert state.git_head(repo) == "b" * 40
    assert state.git_head(tmp_path / "nothing") is None


def test_pending_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = components.Hub()
    args = argparse.Namespace(reinstall=(), force_pixal3d=False)
    monkeypatch.setattr(install_main.venvs, "usable", lambda _id: True)
    ok = {"status": state.INSTALLED, "fingerprint": hub.fingerprint()}
    assert install_main._pending(hub, ok, args) is None
    assert install_main._pending(hub, {**ok, "fingerprint": "old"}, args) == Pending("an update changed it")
    assert install_main._pending(hub, {"status": state.FAILED}, args) == Pending("the last attempt did not finish")
    assert install_main._pending(hub, None, args) == Pending("not installed yet")
    monkeypatch.setattr(hub, "repair_work", lambda: "its AI models move")     # installed, unchanged: repair only
    assert install_main._pending(hub, ok, args) == Pending("its AI models move", repair=True)
    monkeypatch.setattr(install_main.venvs, "usable", lambda _id: False)
    assert "missing or broken" in install_main._pending(hub, ok, args).reason
    args.reinstall = ("hub",)
    assert install_main._pending(hub, ok, args) == Pending("a fresh reinstall was requested")


def test_recipe_changes_only_its_own_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    before = {c.id: c.fingerprint() for c in components.COMPONENTS}
    imesh_c = next(c for c in components.COMPONENTS if c.id == paths.IMESH)
    # An empty recipe adds nothing: the fingerprints of existing installs did not change with the key.
    assert before[paths.IMESH] == state.fingerprint({
        "component": imesh_c.id, "installer": components.INSTALLER_VERSION,
        "python": f"{sys.version_info[0]}.{sys.version_info[1]}", "pip": venvs.PIP_VERSION, **imesh_c.inputs()})
    monkeypatch.setattr(imesh_c, "recipe", "2")
    after = {c.id: c.fingerprint() for c in components.COMPONENTS}
    assert [k for k in before if before[k] != after[k]] == [paths.IMESH]


def test_newest_imesh_wheel_is_picked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for v in ("0.1.0", "0.10.0", "0.9.3"):
        (tmp_path / f"instant_meshes_brush-{v}-cp311-cp311-win_amd64.whl").write_bytes(b"x")
    monkeypatch.setattr(components.paths, "WHL", tmp_path)
    assert components.Imesh.wheel().name.startswith("instant_meshes_brush-0.10.0-")


def test_pixal3d_install_py_is_still_parseable() -> None:
    """A Pixal3D bump that renames these lists makes the installer fall back silently (a stale model list,
    and its --no-cache-dir pip downloads every pinned package again): fail loudly here instead."""
    if not (paths.PIXAL3D_DIR / "install.py").is_file():
        pytest.skip("code/pixal3d is not checked out")
    _, tree = components._pixal3d_install_py()
    assert tree is not None
    assert components._str_list_assigned(tree, "PIXAL3D_SINGLE_VIEW_PATTERNS"), "patterns list not found"
    general, torch_index = components._pixal3d_pip_pins()
    assert general                      # torch_index: empty since install.py installs no xformers


def test_pixal3ds_downloads_from_a_moving_tag_are_pinned() -> None:
    """A mirror's "-latest" release tag may move to another build: every file install.py takes from one names the
    SHA-256 of the build Spellcast3D tested, and its download is checked against it (download_file's sha256)."""
    if not (paths.PIXAL3D_DIR / "install.py").is_file():
        pytest.skip("code/pixal3d is not checked out")
    _, tree = components._pixal3d_install_py()
    moving = [node for node in ast.walk(tree) if isinstance(node, ast.Dict) and "-latest/" in ast.unparse(node)]
    assert len(moving) == 3             # flash_attn, natten, flex_gemm's Triton kernels
    for spec in moving:
        pins = [value.value for key, value in zip(spec.keys, spec.values)
                if isinstance(key, ast.Constant) and key.value == "sha256" and isinstance(value, ast.Constant)]
        assert len(pins) == 1 and len(pins[0]) == 64 and int(pins[0], 16) >= 0, ast.unparse(spec)
    checked = {ast.unparse(node.args[0]) for node in ast.walk(tree)
               if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "download_file"
               and "sha256" in ast.unparse(node)}
    assert checked == {"spec['urls']", "FLEX_GEMM_COMPLETE['urls']"}


class _RecordingCtx(Ctx):
    """Records the commands instead of running them."""

    def __init__(self) -> None:
        super().__init__(Output(), False, Path(sys.executable))
        self.commands: list[list[str]] = []
        self.cwds: list[Path] = []
        self.envs: list[dict[str, str]] = []
        self.notes: list[tuple[tuple[str, str], ...]] = []
        self.fail_first = False

    def run(self, argv: list[str], *, cwd: Path, env: dict[str, str], fail: str,
            notes: tuple[tuple[str, str], ...] = (), on_line: Any = None) -> None:
        self.commands.append([str(a) for a in argv])
        self.cwds.append(cwd)
        self.envs.append(env)
        self.notes.append(notes)
        if self.fail_first:
            self.fail_first = False
            raise InstallError(fail)


def test_pixal3d_constraints_keep_pixal3d_pins_and_fall_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "INSTALL_TMP", tmp_path)
    monkeypatch.setattr(paths, "VENVS", tmp_path / "venvs")
    ctx = _RecordingCtx()
    components.Pixal3D()._pip_tested_versions(ctx, ("scipy", "Pandas==9.9.9", "nest_asyncio"))
    written = (tmp_path / "constraints-pixal3d.txt").read_text(encoding="utf-8").splitlines()
    assert "scipy==1.17.1" in written and "nest-asyncio==1.6.0" in written
    assert not [line for line in written if line.lower().startswith("pandas")]   # install.py's own pin wins
    assert ctx.commands[-1][-2:] == ["-c", str(tmp_path / "constraints-pixal3d.txt")]
    ctx = _RecordingCtx()
    ctx.fail_first = True     # the tested versions no longer fit: installed with Pixal3D's pins alone
    assert components.Pixal3D()._pip_tested_versions(ctx, ("scipy",)) is None
    assert "-c" in ctx.commands[0] and "-c" not in ctx.commands[1]


def test_pixal3ds_own_pip_calls_take_the_tested_versions(layout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """install.py's own pip calls (its unpinned dependencies: utils3d pulled moderngl 5.13.0 / glcontext 3.1.0 where
    the tested venv has 5.12.0 / 3.0.0) take the constraints file - SPELLCAST_PIP_CONSTRAINT, which install.py adds
    as -c (its pip runs --isolated, deaf to PIP_CONSTRAINT) - where the tested versions served install.py's general
    packages; not where those fell back to Pixal3D's own pins."""
    monkeypatch.setattr(paths, "INSTALL_TMP", layout / "tmp")
    monkeypatch.setattr(components.Pixal3D, "missing_files", lambda self: [])
    for name in ("_make_copies", "_link_user_hf_cache", "_calibrate", "_verify", "_hi3d_packages"):
        monkeypatch.setattr(components.Pixal3D, name, lambda self, *args, **kwargs: None)
    monkeypatch.setattr(components, "share_torch_files", lambda ctx, venv_id: None)
    monkeypatch.setattr(components, "_pixal3d_pip_pins", lambda: (("scipy", "trimesh==4.10.1"), ()))

    def install_py_env() -> dict[str, str]:
        ctx = _RecordingCtx()
        components.Pixal3D().install(ctx)
        return next(env for cmd, env in zip(ctx.commands, ctx.envs) if cmd[-2:] == ["install.py", "--no-pillow-simd"])

    written = Path(install_py_env()["SPELLCAST_PIP_CONSTRAINT"]).read_text(encoding="utf-8").splitlines()
    assert "moderngl==5.12.0" in written and "glcontext==3.0.0" in written
    assert not [line for line in written if line.startswith("trimesh")]               # install.py's own pin wins
    # the tested versions did not fit (Pixal3D's pins alone served): install.py goes without them too
    monkeypatch.setattr(components.Pixal3D, "_pip_tested_versions", lambda self, ctx, general: None)
    assert "SPELLCAST_PIP_CONSTRAINT" not in install_py_env()


# ---- runner / main ---------------------------------------------------------------------------------

def test_children_get_the_install_drive_temp_and_a_full_disk_is_named(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "INSTALL_TMP", tmp_path / "tmp")
    ctx = Ctx(Output(), False, Path(sys.executable))
    env = ctx.env(None)
    assert env["TEMP"] == env["TMP"] == env["TMPDIR"] == str(tmp_path / "tmp")
    script = "import sys; print('OSError: [Errno 28] No space left on device'); sys.exit(1)"
    with pytest.raises(InstallError, match="disk is full"):
        ctx.run([sys.executable, "-c", script], cwd=tmp_path, env=env, fail="network?")
    with pytest.raises(InstallError, match="network"):
        ctx.run([sys.executable, "-c", "import sys; sys.exit(1)"], cwd=tmp_path, env=env, fail="network?")


def test_busy_check_sees_the_install_and_hub_locks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(install_main, "INSTALL_LOCK", tmp_path / ".install.lock")
    monkeypatch.setattr(install_main, "HUB_LOCK", tmp_path / ".hub.lock")
    assert install_main.main(["--busy-check"]) == install_main.EXIT_OK
    for lock in (tmp_path / ".install.lock", tmp_path / ".hub.lock"):
        with open(lock, "a+b") as f:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            assert install_main.main(["--busy-check"]) == install_main.EXIT_BUSY
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    assert install_main.main(["--busy-check"]) == install_main.EXIT_OK


def test_an_interrupted_install_is_never_recorded_as_installed(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "INSTALL_STATE", tmp_path / "install-state.json")
    monkeypatch.setattr(install_main.venvs, "ensure", lambda *a, **k: None)
    monkeypatch.setattr(install_main, "_gpu_info", lambda: None)
    hub = components.Hub()
    state.record(hub.id, state.INSTALLED, "", hub.fingerprint())    # e.g. before a --reinstall
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(hub, "prepare", lambda ctx: None)
    monkeypatch.setattr(hub, "install", lambda ctx: seen.append(state.load()[hub.id]))
    args = argparse.Namespace(reinstall=("hub",), force_pixal3d=False)
    report = preflight.Report()
    result = install_main._install_one(Ctx(Output(), False, Path(sys.executable)), hub, report, args,
                                       Pending("a fresh reinstall was requested"))
    # While it installs (a closed window would stop it right here), the entry says "not finished".
    assert seen[0]["status"] == state.FAILED and seen[0]["reason"] == install_main.STOPPED
    assert result[0] == state.INSTALLED and state.load()[hub.id]["status"] == state.INSTALLED


def test_a_half_written_pip_is_repaired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "VENVS", tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    (base / "python.exe").write_bytes(b"")
    venv = tmp_path / "imesh"
    (venv / "Scripts").mkdir(parents=True)
    (venv / "Scripts" / "python.exe").write_bytes(b"")
    (venv / "pyvenv.cfg").write_text(f"home = {base}\nversion = 3.11.9\n", encoding="utf-8")
    sp = venv / "Lib" / "site-packages"
    (sp / "pip").mkdir(parents=True)
    (sp / "pip" / "__init__.py").write_text("")                  # the upgrade was cut off: no RECORD yet
    (sp / "~ip").mkdir()
    (sp / "~ip-24.0.dist-info").mkdir()
    (sp / "numpy").mkdir()
    assert not venvs.pip_intact("imesh")
    ctx = _RecordingCtx()
    venvs.ensure(ctx, "imesh", "Instant Meshes", fresh=False)
    assert not (sp / "pip").exists() and not (sp / "~ip").exists() and not (sp / "~ip-24.0.dist-info").exists()
    assert (sp / "numpy").exists()                                # nothing else is touched
    assert ctx.commands[0][-3:] == ["-m", "ensurepip", "--default-pip"]
    assert ctx.commands[1][-2:] == ["install", f"pip=={venvs.PIP_VERSION}"]
    (sp / "pip").mkdir()
    (sp / "pip" / "__init__.py").write_text("")
    (sp / f"pip-{venvs.PIP_VERSION}.dist-info").mkdir()
    (sp / f"pip-{venvs.PIP_VERSION}.dist-info" / "RECORD").write_text("")
    assert venvs.pip_intact("imesh")
    ctx = _RecordingCtx()
    venvs.ensure(ctx, "imesh", "Instant Meshes", fresh=False)
    assert len(ctx.commands) == 1                                 # a healthy venv only gets the pip pin


def test_main_runs_for_the_open_hub_with_a_progress_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(install_main, "HUB_LOCK", tmp_path / ".hub.lock")
    monkeypatch.setattr(install_main, "INSTALL_LOCK", tmp_path / ".install.lock")
    monkeypatch.setattr(paths, "INSTALL_STATE", tmp_path / "install-state.json")
    monkeypatch.setattr(install_main.models, "ensure_folders", lambda: None)
    monkeypatch.setattr(install_main.venvs, "repair_if_moved", lambda *a: False)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "spellcast3d-config.ini")     # none yet: made, qwen
    monkeypatch.delenv("SPELLCAST_CONFIG", raising=False)
    monkeypatch.setattr(install_main.sdcpp, "share_cuda_dlls", lambda log: 0)       # not this PC's real runtime
    selected: list[list[str]] = []
    todo: dict[str, Pending] = {}

    def plan(chosen: list[components.Component], args: argparse.Namespace) -> dict[str, Pending]:
        selected.append([c.id for c in chosen])
        return dict(todo)

    monkeypatch.setattr(install_main, "_plan", plan)
    progress_file = tmp_path / "install-progress.json"
    with open(tmp_path / ".hub.lock", "a+b") as f:              # Spellcast3D is open
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        # The hub's own run: its part (hub) is left alone, and nothing to do is written as finished.
        assert install_main.main(["--if-needed", "--progress-file", str(progress_file)]) == install_main.EXIT_OK
        assert selected[-1] == ["imesh", "qwen", "pixal3d", "lato2", "triflow"]      # the configured generator
        assert (tmp_path / "spellcast3d-config.ini").is_file()
        written = json.loads(progress_file.read_text(encoding="utf-8"))
        assert written["running"] is False and written["finished"] is True and written["error"] is None
        assert written["components"]["hub"]["status"] == progress.SKIPPED
        assert written["components"]["qwen"] == {**written["components"]["qwen"], "status": progress.DONE,
                                                 "fraction": 1.0}
        assert "forge" not in written["components"]                  # not picked: not on the install card
        # Anyone else still waits for the open hub to close.
        todo["qwen"] = Pending("an update changed it")
        assert install_main.main(["--if-needed"]) == install_main.EXIT_BUSY
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    assert install_main._parse(["--progress-file", "code/venvs/p.json"]).progress_file == \
        paths.ROOT / "code" / "venvs" / "p.json"


# ---- install/progress.py ----------------------------------------------------------------------------------

def test_progress_file_follows_the_run(tmp_path: Path) -> None:
    target = tmp_path / "install-progress.json"
    tracker = progress.Progress(target)
    tracker.begin({"hub": (progress.SKIPPED, ""), "forge": (progress.PENDING, "not installed yet"),
                   "pixal3d": (progress.PENDING, "not installed yet")})
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["running"] is True and written["pid"] == os.getpid()
    assert written["components"]["forge"] == {"status": "pending", "step": "", "fraction": None, "step_fraction": None,
                                              "bytes_done": None, "bytes_total": None, "message": "not installed yet"}
    tracker.start("forge")
    tracker.step("Downloading PyTorch (3.5 GB)", until=0.6)
    tracker.download(50, 100)
    forge_now = tracker._data["components"]["forge"]
    assert forge_now["fraction"] == 0.3 and forge_now["bytes_done"] == 50 and forge_now["status"] == "running"
    tracker.download(10, 100)                    # the next file of the same pip call: the bar never goes back
    assert forge_now["fraction"] == 0.3 and forge_now["bytes_done"] == 10
    tracker.step("Installing Forge Neo's packages", until=0.97)
    assert forge_now["fraction"] == 0.6 and forge_now["bytes_done"] is None
    tracker.note("Collecting gradio==4.40.0\n")
    assert forge_now["message"] == "Collecting gradio==4.40.0"
    # What the step does right now, and how far it is (the calibration's lines): after the label, within the step.
    tracker.detail("timing the attention kernels", 0.5)
    assert forge_now["step"] == "Installing Forge Neo's packages · timing the attention kernels"
    assert forge_now["step_fraction"] == 0.5 and forge_now["fraction"] == round(0.6 + 0.37 * 0.5, 4)
    tracker.detail("the graphics card is busy (46 % in use): waiting")      # no fraction: the text alone changes
    assert forge_now["step"].endswith("· the graphics card is busy (46 % in use): waiting")
    assert forge_now["step_fraction"] == 0.5
    tracker.detail("shape 1 of 9", 0.2)                                    # never back
    assert forge_now["step_fraction"] == 0.5 and forge_now["fraction"] == round(0.6 + 0.37 * 0.5, 4)
    tracker.step("Checking", until=1.0)
    assert forge_now["step"] == "Checking" and forge_now["step_fraction"] is None
    tracker.end("forge", progress.DONE)
    tracker.finish("Could not reach pypi.org")
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["running"] is False and written["finished"] is True and written["current"] is None
    assert written["components"]["forge"]["status"] == "done" and written["components"]["forge"]["fraction"] == 1.0
    assert written["components"]["pixal3d"] == {**written["components"]["pixal3d"], "status": "failed",
                                                "message": "Could not reach pypi.org"}
    assert written["error"] == "Could not reach pypi.org"


def test_progress_heartbeat_measures_and_rewrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(progress, "TICK", 0.05)
    target = tmp_path / "install-progress.json"
    tracker = progress.Progress(target)
    tracker.begin({"pixal3d": (progress.PENDING, "")})
    tracker.start("pixal3d")
    tracker.step("Downloading its AI models (about 27 GB)", until=1.0, measure=lambda: (27, 108))
    deadline = time.monotonic() + 5
    while json.loads(target.read_text(encoding="utf-8"))["components"]["pixal3d"]["bytes_done"] != 27:
        assert time.monotonic() < deadline, "the heartbeat did not write the measured bytes"
        time.sleep(0.05)
    first = json.loads(target.read_text(encoding="utf-8"))["updated"]
    while json.loads(target.read_text(encoding="utf-8"))["updated"] == first:   # written again, unchanged
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert json.loads(target.read_text(encoding="utf-8"))["components"]["pixal3d"]["fraction"] == 0.25
    tracker.finish()


def test_runner_reports_pip_progress_and_adds_notes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tracker = progress.Progress(None)
    tracker.begin({"pixal3d": (progress.PENDING, "")})
    tracker.start("pixal3d")
    tracker.step("Downloading PyTorch (3.5 GB)", until=0.5)
    ctx = Ctx(Output(), False, Path(sys.executable), progress=tracker)
    script = ("print('Progress 3 of 4'); print('Set these before running, so the app finds the weights:'); "
              "print('  set HF_HOME=x'); print('Then:  python inference.py')")
    ctx.run([sys.executable, "-c", script], cwd=tmp_path, env=ctx.env(None), fail="?",
            notes=((components.PIXAL3D_ENV_HINT, components.PIXAL3D_ENV_NOTE),))
    printed = capsys.readouterr().out.splitlines()
    assert printed[-1] == components.PIXAL3D_ENV_NOTE and printed[-2] == "Then:  python inference.py"
    assert "Progress 3 of 4" not in printed
    entry = tracker._data["components"]["pixal3d"]
    assert entry["bytes_done"] == 3 and entry["fraction"] == 0.375 and entry["message"].startswith("Then:")
    ctx.run([sys.executable, "-c", "print('nothing to see')"], cwd=tmp_path, env=ctx.env(None), fail="?",
            notes=((components.PIXAL3D_ENV_HINT, components.PIXAL3D_ENV_NOTE),))
    assert components.PIXAL3D_ENV_NOTE not in capsys.readouterr().out


# ---- install/models.py: the models folder and Pixal3D's junction ------------------------------------------

@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A Spellcast3D folder in tmp_path/sc3d with code/pixal3d and models/pixal3d (+ its .gitkeep)."""
    root = tmp_path / "sc3d"
    repo = root / "code" / "pixal3d"
    repo.mkdir(parents=True)
    (root / "models" / "pixal3d").mkdir(parents=True)
    (root / "models" / "pixal3d" / ".gitkeep").write_bytes(b"")
    for name, value in (("ROOT", root), ("MODELS", root / "models"), ("MODELS_PIXAL3D", root / "models" / "pixal3d"),
                        ("PIXAL3D_DIR", repo), ("PIXAL3D_MODELS_LINK", repo / "MODELS"),
                        ("INSTALL_STATE", root / "code" / "venvs" / "install-state.json")):
        monkeypatch.setattr(paths, name, value)
    monkeypatch.setattr(components, "PIXAL3D_MODELS", repo / "MODELS")
    monkeypatch.setattr(components, "_card", lambda: None)     # no graphics card unless a test gives one
    # no model copies on this PC: the disk budget counts those (e.g. Hi3D's in this PC's HuggingFace cache) as there
    monkeypatch.setattr(models, "user_hf_caches", lambda: [])
    monkeypatch.setattr(models, "EXTRA_SOURCES", [])
    monkeypatch.delenv("SPELLCAST_MODELS_FROM", raising=False)
    return root


OLD_MODELS = {
    "dinov3/model.safetensors": b"dino",
    "RMBG-2.0/model.safetensors": b"rmbg",
    "hub/models--TencentARC--Pixal3D/refs/main": b"abc",
    "hub/models--TencentARC--Pixal3D/snapshots/abc/pipeline.json": b"{}",
    "torch/hub/checkpoints/naf_release.pth": b"naf",
}


def _write_tree(folder: Path, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(data)


def _tree(folder: Path) -> dict[str, bytes]:
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_an_older_models_folder_moves_into_models_pixal3d(layout: Path) -> None:
    link, target = paths.PIXAL3D_MODELS_LINK, paths.MODELS_PIXAL3D
    _write_tree(link, OLD_MODELS)
    assert models.pixal3d_link_work() == "its AI models move into the models\\pixal3d folder"
    before = os.stat(link / "dinov3" / "model.safetensors").st_ino
    lines: list[str] = []
    assert models.link_pixal3d_models(lines.append)
    assert models.is_junction(link) and models.pixal3d_link_work() is None
    assert _tree(target) == {**OLD_MODELS, ".gitkeep": b""}
    assert _tree(link) == _tree(target)                                    # Pixal3D's paths go through the link
    assert os.stat(target / "dinov3" / "model.safetensors").st_ino == before   # moved, not copied
    assert not list(paths.MODELS.glob("pixal3d-old-*")) and lines[0].startswith("Moving Pixal3D's AI models")
    assert components.Pixal3D().models_present_gb() == sum(map(len, OLD_MODELS.values())) / 1e9   # counted once
    assert models.link_pixal3d_models(lines.append)                          # nothing left to do
    assert models.is_junction(link) and _tree(target) == {**OLD_MODELS, ".gitkeep": b""}


def test_moving_into_models_pixal3d_keeps_its_files_and_sets_the_older_copies_aside(layout: Path) -> None:
    link, target = paths.PIXAL3D_MODELS_LINK, paths.MODELS_PIXAL3D
    _write_tree(link, OLD_MODELS)
    mine = {"dinov3/model.safetensors": b"mine", "hub/models--TencentARC--Pixal3D/refs/main": b"zzz"}
    _write_tree(target, mine)
    lines: list[str] = []
    assert models.link_pixal3d_models(lines.append)
    assert _tree(target) == {**OLD_MODELS, **mine, ".gitkeep": b""}          # the artist's own files win
    aside, = paths.MODELS.glob("pixal3d-old-*")
    assert _tree(aside) == {rel: OLD_MODELS[rel] for rel in mine}           # nothing is lost
    assert any("already in models\\pixal3d" in line and aside.name in line for line in lines)


@pytest.mark.parametrize("with_conflicts", [False, True])
def test_a_failed_move_puts_everything_back(layout: Path, monkeypatch: pytest.MonkeyPatch,
                                            with_conflicts: bool) -> None:
    link, target = paths.PIXAL3D_MODELS_LINK, paths.MODELS_PIXAL3D
    _write_tree(link, OLD_MODELS)
    if with_conflicts:
        _write_tree(target, {"dinov3/model.safetensors": b"mine"})
    before_old, before_new = _tree(link), _tree(target)

    def no_junction(link: Path, target: Path) -> None:
        raise OSError("mklink: not supported here")

    monkeypatch.setattr(models, "_make_junction", no_junction)
    lines: list[str] = []
    assert not models.link_pixal3d_models(lines.append)
    assert not models.is_junction(link) and _tree(link) == before_old and _tree(target) == before_new
    assert not list(paths.MODELS.glob("pixal3d-old-*"))
    assert "not supported here" in lines[-1] and "tries again" in lines[-1]


def test_links_to_models_pixal3d_are_made_and_mended(layout: Path) -> None:
    link, target = paths.PIXAL3D_MODELS_LINK, paths.MODELS_PIXAL3D
    assert models.pixal3d_link_work() == "the link from code\\pixal3d\\MODELS to models\\pixal3d is missing"
    assert models.link_pixal3d_models(lambda line: None)                    # a fresh install: just the link
    assert models.is_junction(link) and (link / ".gitkeep").is_file()
    os.rmdir(link)
    gone = layout / "gone"
    gone.mkdir()
    models._make_junction(link, gone)
    gone.rmdir()                                                             # a link that points nowhere
    assert models.pixal3d_link_work() is not None
    assert models.link_pixal3d_models(lambda line: None)
    assert os.path.realpath(link) == os.path.realpath(target)
    os.rmdir(link)
    own = layout / "my weights"
    own.mkdir()
    models._make_junction(link, own)                                         # the artist's own link: left alone
    assert models.pixal3d_link_work() is None


def test_a_whole_models_folder_copied_into_models_pixal3d_is_unwrapped(layout: Path) -> None:
    target = paths.MODELS_PIXAL3D
    _write_tree(target / "MODELS", OLD_MODELS)
    models.unwrap_copied_folder(lambda line: None)
    assert _tree(target) == {**OLD_MODELS, ".gitkeep": b""}
    models.unwrap_copied_folder(lambda line: None)                            # nothing more to do
    assert _tree(target) == {**OLD_MODELS, ".gitkeep": b""}


def test_folder_sizes_count_hard_links_once(tmp_path: Path) -> None:
    (tmp_path / "hub" / "blobs").mkdir(parents=True)
    (tmp_path / "hub" / "blobs" / "a").write_bytes(b"x" * 10)
    os.link(tmp_path / "hub" / "blobs" / "a", tmp_path / "hub" / "a-snapshot")
    (tmp_path / "hub" / "blobs" / "b.incomplete").write_bytes(b"x" * 7)
    assert models.folder_bytes(tmp_path) == 17
    assert models.partial_download_bytes(tmp_path) == 7


def test_pixal3d_download_progress_leaves_out_the_multiview_models(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """The hub downloads them on request (multiview.py): a repair of Pixal3D does not count them as done."""
    monkeypatch.setattr(components, "PIXAL3D_MODELS", tmp_path)
    mv = next(rel for rel in models.PIXAL3D_MULTI_VIEW_FILES if rel.endswith(".json"))
    _hf_cache(tmp_path / "hub", components.PIXAL3D_HF_REPO, {"pipeline.json": b"x" * 30, mv: b"y" * 9}, symlinks=False)
    assert models.folder_bytes(tmp_path) == 39 + len("c1")
    assert components._pixal3d_models_bytes() == 30 + len("c1")
    # nor the faster-loading copies of the flow models (made after the download), finished or not
    ckpts = models.hf_repo_folder(tmp_path / "hub", components.PIXAL3D_HF_REPO) / "snapshots" / "c1" / "ckpts"
    ckpts.mkdir()
    (ckpts / f"flow{models.PIXAL3D_COPY_SUFFIX}").write_bytes(b"c" * 5)
    (ckpts / f"flow_mv{models.PIXAL3D_COPY_SUFFIX}.tmp").write_bytes(b"t" * 3)
    (ckpts / "flow.safetensors").write_bytes(b"w" * 7)
    assert components._pixal3d_models_bytes() == 30 + len("c1") + 7


def test_pixal3d_copies_its_flow_models_after_the_download_and_never_fails_for_them(
        layout: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The fork's `python -m pixal3d.staged.weights <snapshot>` in the Pixal3D venv: it skips copies that are up
    to date (each install runs it), downloads nothing, and a failure leaves Pixal3D reading the originals."""
    models_dir = paths.PIXAL3D_MODELS_LINK
    pixal = components.Pixal3D()
    ctx = _RecordingCtx()
    pixal._make_copies(ctx, until=0.97)                       # no models: nothing to copy, a warning
    assert ctx.commands == [] and "faster-loading copies were not made" in capsys.readouterr().out

    snapshot = _hf_cache(models_dir / "hub", components.PIXAL3D_HF_REPO, {"pipeline.json": b"{}"},
                         symlinks=False) / "snapshots" / "c1"
    pixal._make_copies(ctx, until=0.97)
    assert ctx.commands == [[pixal._python(), "-m", "pixal3d.staged.weights", str(snapshot)]]
    assert ctx.cwds == [paths.PIXAL3D_DIR]
    env = ctx.envs[-1]
    assert env["CUDA_VISIBLE_DEVICES"] == "-1" and env["HF_HUB_OFFLINE"] == "1"     # on the CPU, offline
    assert env["HF_HOME"] == str(models_dir) and env["TORCH_HOME"] == str(models_dir / "torch")
    assert "Warning" not in capsys.readouterr().out
    ctx.fail_first = True
    pixal._make_copies(ctx, until=0.97)                       # it failed: a warning, the install goes on
    assert "Warning: the faster-loading copies" in capsys.readouterr().out

    # where: after the models are downloaded (also by a repair), before the installation is checked
    order: list[str] = []
    for name in ("_download_models", "_make_copies", "_link_models", "_link_user_hf_cache", "_pip_tested_versions",
                 "_verify"):
        monkeypatch.setattr(components.Pixal3D, name,
                            lambda self, *args, _name=name, **kwargs: order.append(_name))
    monkeypatch.setattr(components, "share_torch_files", lambda ctx, venv_id: order.append("share"))
    pixal.install(_RecordingCtx())
    assert order.index("_download_models") < order.index("_make_copies") < order.index("_verify")
    order.clear()
    monkeypatch.setattr(components.Pixal3D, "missing_files", lambda self: ["NAF"])
    pixal.repair(_RecordingCtx())
    assert order == ["_link_models", "_link_user_hf_cache", "_download_models", "_make_copies"]
    order.clear()
    monkeypatch.setattr(components.Pixal3D, "missing_files", lambda self: [])
    pixal.repair(_RecordingCtx())                             # nothing missing: no copies' run either
    assert order == ["_link_models"]


def test_the_hub_names_the_copies_as_the_fork_does() -> None:
    weights = paths.PIXAL3D_DIR / "pixal3d" / "staged" / "weights.py"
    if not weights.is_file():
        pytest.skip("code/pixal3d is not checked out")
    tree = ast.parse(weights.read_text(encoding="utf-8"))
    suffix = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and getattr(node.targets[0], "id", "") == "COPY_SUFFIX")
    assert suffix == models.PIXAL3D_COPY_SUFFIX
    assert models.PIXAL3D_COPIES_MODULE == "pixal3d.staged.weights"
    # a fresh install has room for the copies; an update of one without them needs none (they are optional). Hi3D's
    # packages come with it, its models are the 3D stage's own download (test_hi3d).
    pixal = components.Pixal3D()
    assert pixal.size_gb == pytest.approx(40.0 + models.PIXAL3D_COPIES_BYTES / 1e9 + components.HI3D_PACKAGES_GB)
    assert pixal.models_gb == pytest.approx(components.PIXAL3D_MODELS_BYTES / 1e9)


def _hf_cache(cache: Path, repo_id: str, files: dict[str, bytes], symlinks: bool) -> Path:
    """A HuggingFace cache as huggingface_hub writes it, with symbolic links to blobs or with plain files."""
    folder = models.hf_repo_folder(cache, repo_id)
    (folder / "refs").mkdir(parents=True)
    (folder / "refs" / "main").write_text("c1", encoding="utf-8")
    for rel, data in files.items():
        snapshot_file = folder / "snapshots" / "c1" / rel
        snapshot_file.parent.mkdir(parents=True, exist_ok=True)
        if symlinks:
            blob = folder / "blobs" / f"sha-{len(data)}"
            blob.parent.mkdir(exist_ok=True)
            blob.write_bytes(data)
            os.symlink(os.path.relpath(blob, snapshot_file.parent), snapshot_file)
        else:
            snapshot_file.write_bytes(data)
    return folder


@pytest.mark.parametrize("symlinks", [False, True])
def test_models_in_the_artists_hf_cache_are_hard_linked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        symlinks: bool) -> None:
    cache = tmp_path / "hf" / "hub"
    try:
        source = _hf_cache(cache, components.MOGE_HF_REPO, {"model.pt": b"moge", "README.md": b"#"}, symlinks)
    except OSError:
        pytest.skip("this Windows user may not create symbolic links (Developer Mode is off)")
    monkeypatch.setattr(models, "user_hf_caches", lambda: [cache])
    hub = tmp_path / "sc3d" / "hub"
    lines: list[str] = []
    assert models.link_from_hf_cache(hub, components.MOGE_HF_REPO, None, lines.append)
    snapshot = models.hf_snapshot(hub, components.MOGE_HF_REPO)
    assert snapshot is not None and snapshot.name == "c1"
    assert (snapshot / "model.pt").read_bytes() == b"moge" and (snapshot / "README.md").read_bytes() == b"#"
    real = Path(os.path.realpath(source / "snapshots" / "c1" / "model.pt"))
    assert os.stat(snapshot / "model.pt").st_ino == os.stat(real).st_ino           # a hard link, no copy
    assert (models.hf_repo_folder(hub, components.MOGE_HF_REPO) / "blobs" / real.name).is_file() == symlinks
    assert "linked 2 file(s)" in lines[-1]
    assert not models.link_from_hf_cache(tmp_path / "other", "nobody/nothing", None, lines.append)
    monkeypatch.setattr(models, "_same_volume", lambda a, b: False)
    assert not models.link_from_hf_cache(tmp_path / "third", components.MOGE_HF_REPO, None, lines.append)
    assert "another drive" in lines[-1] and not (tmp_path / "third" / "models--Ruicheng--moge-2-vitl").exists()


def test_pixal3d_downloads_resume_an_interrupted_file_over_https(layout: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    models_dir = paths.PIXAL3D_MODELS_LINK
    (models_dir / "hub").mkdir(parents=True)
    monkeypatch.setattr(components.Pixal3D, "missing_files", lambda self: [])
    monkeypatch.setattr(components, "RESUME_OVER_HTTPS_BYTES", 10)
    ctx = _RecordingCtx()
    pixal = components.Pixal3D()
    pixal._download_models(ctx, "--models-only", until=1.0)
    assert ctx.envs[-1]["HF_HOME"] == str(models_dir) and "HF_HUB_DISABLE_XET" not in ctx.envs[-1]
    assert ctx.notes[-1] == ((components.PIXAL3D_ENV_HINT, components.PIXAL3D_ENV_NOTE),)
    (models_dir / "hub" / "blob.incomplete").write_bytes(b"x" * 10)
    pixal._download_models(ctx, "--models-only", until=1.0)
    assert ctx.envs[-1]["HF_HUB_DISABLE_XET"] == "1"
    assert pixal.inputs()["xet"] == components.HF_XET              # a new hf_xet pin installs Pixal3D again


def test_pixal3d_repair_only_links_when_nothing_is_missing(layout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_tree(paths.PIXAL3D_MODELS_LINK, OLD_MODELS)
    pixal = components.Pixal3D()
    monkeypatch.setattr(components.Pixal3D, "missing_files", lambda self: [])
    assert pixal.repair_work() == "its AI models move into the models\\pixal3d folder"
    ctx = _RecordingCtx()
    pixal.repair(ctx)
    assert ctx.commands == [] and pixal.repair_work() is None
    assert models.is_junction(paths.PIXAL3D_MODELS_LINK)


def test_pixal3d_fingerprint_follows_its_install_inputs_not_the_forks_commit(layout: Path,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """A code-only update of the fork starts without a reinstall; a change of install.py, of a file in whl/ (a
    rebuilt wheel of the same name too) or of the copies' format installs it again."""
    repo = paths.PIXAL3D_DIR
    monkeypatch.setattr(components, "PIXAL3D_WHL", repo / "whl")
    (repo / "install.py").write_text("general_deps = ['tqdm==1']\n", encoding="utf-8")
    (repo / "pixal3d" / "staged").mkdir(parents=True)
    weights = repo / "pixal3d" / "staged" / "weights.py"
    weights.write_text('COPY_SUFFIX = ".x"\nFORMAT = "1"   # bump when\n', encoding="utf-8")
    (repo / "whl").mkdir()
    wheel = repo / "whl" / "flex_gemm-0.0.1-cp311-cp311-win_amd64.whl"
    wheel.write_bytes(b"one")
    pixal = components.Pixal3D()
    assert components.pixal3d_copies_format() == "1"
    seen = [pixal.fingerprint()]
    (repo / ".git").mkdir()
    (repo / ".git" / "HEAD").write_text("a" * 40 + "\n", encoding="utf-8")       # a new commit of the fork ...
    (repo / "pixal3d" / "staged" / "runner.py").write_text("# code\n", encoding="utf-8")   # ... of Python code only
    assert pixal.fingerprint() == seen[-1]
    for change in (lambda: (repo / "install.py").write_text("general_deps = ['tqdm==2']\n", encoding="utf-8"),
                   lambda: wheel.write_bytes(b"two!"),
                   lambda: (repo / "whl" / "flex_gemm.json").write_text("{}", encoding="utf-8"),
                   lambda: weights.write_text('FORMAT = "2"\n', encoding="utf-8"),
                   lambda: weights.write_text('FORMAT = "2"\nINT8_FORMAT = "1"\n', encoding="utf-8"),   # int8 copies
                   lambda: weights.write_text('FORMAT = "2"\nINT8_FORMAT = "2"\n', encoding="utf-8")):
        change()
        seen.append(pixal.fingerprint())
        assert seen[-1] not in seen[:-1]
    weights.unlink()
    assert components.pixal3d_copies_format() == "" and pixal.fingerprint() not in seen


RTX_3060 = gpu.GpuInfo(0, "NVIDIA GeForce RTX 3060", 8.6, 12288, 500, "596.36")
# An old card's floor for the decode in boxes (the fork's products_floor section; its own rule: test below)
PRODUCTS = {"ok": True, "profile": "turing", "products_floor": None, "linear_floor": 4096, "exact": True, "shapes": []}


def _calibration_record(card: str, **values: Any) -> None:
    record = {"version": components.PIXAL3D_CALIBRATION, "complete": True, "card": card, **values}
    components.pixal3d_calibration_file().write_text(json.dumps(record), encoding="utf-8")


def _calibration_checkout() -> None:
    (paths.PIXAL3D_DIR / "install.py").write_text("", encoding="utf-8")
    (paths.PIXAL3D_DIR / "pixal3d").mkdir(exist_ok=True)
    (paths.PIXAL3D_DIR / "pixal3d" / "calibrate.py").write_text("", encoding="utf-8")


def test_pixal3d_is_calibrated_once_for_this_pcs_card(layout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Due on a card Pixal3D runs on until a complete record of this version, made on this card, is there."""
    monkeypatch.setattr(components, "_cards", lambda: ())            # no nvidia-smi: the given card alone
    monkeypatch.setattr(paths, "VENVS", layout / "code" / "venvs")   # Pixal3D's venv: none yet (no Triton, no kernels)
    work = components.pixal3d_calibration_work
    assert work(gpu.GpuInfo(0, "NVIDIA GeForce GTX 980", 5.2, 4096, 500, "596.36")) is None    # Pixal3D never runs
    assert work(RTX_3060) is not None                   # the fork not checked out yet: it comes with it
    (paths.PIXAL3D_DIR / "install.py").write_text("", encoding="utf-8")
    assert work(RTX_3060) is None                       # a checkout of the fork without the calibration
    _calibration_checkout()
    assert "calibrated once" in work(RTX_3060)          # no record
    _calibration_record(RTX_3060.name, complete=False)
    assert work(RTX_3060) is None                       # a run with a reservation stands (Settings measures again)
    _calibration_record(RTX_3060.name, version=components.PIXAL3D_CALIBRATION - 1)
    assert "new version" in work(RTX_3060)
    _calibration_record("NVIDIA GeForce RTX 2080 Ti")
    assert "graphics card changed" in work(RTX_3060)
    _calibration_record(RTX_3060.name)
    assert work(RTX_3060) is None
    # Hi3D's step: left out until its models are here (their download runs it); then due alone, unless it ran
    monkeypatch.setattr(models, "hi3d_on_disk", lambda folder=None: True)
    assert work(RTX_3060) == components.HI3D_CALIBRATION_WORK                      # no "spconv" section
    _calibration_record(RTX_3060.name, spconv={"ok": True, "skipped": "Hi3D's code is not on this PC", "entries": []})
    assert work(RTX_3060) == components.HI3D_CALIBRATION_WORK
    _calibration_record(RTX_3060.name, spconv={"ok": False, "error": "it failed"})
    assert work(RTX_3060) is None                                                  # a step that ran stands
    _calibration_record(RTX_3060.name, spconv={"ok": True, "entries": [{"shape": 1}]})
    assert work(RTX_3060) is None
    monkeypatch.setattr(models, "hi3d_on_disk", lambda folder=None: False)
    _calibration_record(RTX_3060.name)
    assert work(RTX_3060) is None
    # a record made before the Triton step chose the int8 GEMMs' fc1 tile: that step is due alone (sm_80+ cards,
    # where those GEMMs are Triton's), not a new version of the whole calibration
    turing = gpu.GpuInfo(0, "NVIDIA GeForce RTX 2070", 7.5, 8192, 500, "596.36")
    decoders = {"ok": True, "entries": [{"key": 1}]}               # Turing's spconv choices (their own rule, below)
    _calibration_record(RTX_3060.name, triton={"ok": True, "fused": True, "int8": True, "codes": True})
    assert work(RTX_3060) == components.INT8_TILES_CALIBRATION_WORK
    _calibration_record(turing.name, triton={"ok": True, "int8": True}, spconv_decoders=decoders,
                        decode_products=PRODUCTS)
    assert work(turing) is None
    _calibration_record(RTX_3060.name, triton={"ok": True, "int8": "off: no Triton"})
    assert work(RTX_3060) is None                                                  # its int8 GEMMs did not run
    _calibration_record(RTX_3060.name, triton={"ok": True, "int8": True, "int8_tiles": {"ok": False}})
    assert work(RTX_3060) is None                                                  # a timing that ran stands
    _calibration_record(RTX_3060.name, triton={"ok": True, "int8": True, "int8_tiles": {"fc1_group": [64]}})
    assert work(RTX_3060) is None
    # a choice timed with another Triton than the venv's (the fork's int8mm reads it as none): due again, alone. The
    # venv's version is triton/__init__.py's __version__ ("3.4.0"), never the wheel's "3.4.0.post21"
    site = paths.venv_dir(paths.PIXAL3D) / "Lib" / "site-packages" / "triton"
    timed = {"ok": True, "fc1_group": [64, 128, 128, 8, 3, 4], "triton": "0.0"}
    _calibration_record(RTX_3060.name, triton={"ok": True, "int8": True, "int8_tiles": timed})
    assert work(RTX_3060) is None                                                  # the venv's version unknown
    site.mkdir(parents=True)
    (site / "__init__.py").write_text("import os\n__version__ = '3.4.0'\n", encoding="utf-8")
    assert components.pixal3d_triton_version() == "3.4.0"
    assert work(RTX_3060) == components.INT8_TILES_CALIBRATION_WORK
    _calibration_record(turing.name, triton={"ok": True, "int8": True, "int8_tiles": timed}, spconv_decoders=decoders,
                        decode_products=PRODUCTS)
    assert work(turing) is None                                                    # no Triton int8 GEMMs there
    _calibration_record(RTX_3060.name, triton={"ok": True, "int8": True, "int8_tiles": {**timed, "triton": "3.4.0"}})
    assert work(RTX_3060) is None                                                  # this Triton's choice
    _calibration_record(RTX_3060.name, triton={"ok": True, "int8": True, "int8_tiles": {"ok": False, "triton": "0.0"}})
    assert work(RTX_3060) is None                                                  # a failed timing stands
    monkeypatch.setattr(models, "hi3d_on_disk", lambda folder=None: True)          # both due: one run of both
    _calibration_record(RTX_3060.name, triton={"ok": True, "int8": True})
    assert work(RTX_3060) not in (None, components.INT8_TILES_CALIBRATION_WORK, components.HI3D_CALIBRATION_WORK)
    monkeypatch.setattr(models, "hi3d_on_disk", lambda folder=None: False)
    components.pixal3d_calibration_file().write_text("{", encoding="utf-8")
    assert work(RTX_3060) is not None                   # unreadable: measured again
    # the card itself, by UUID, whatever index torch or nvidia-smi give it (a PC with two cards)
    this = gpu.GpuInfo(0, RTX_3060.name, 8.6, 12288, 500, "596.36", uuid="GPU-aaaa-1111")
    other = gpu.GpuInfo(1, "NVIDIA GeForce RTX 4070", 8.9, 12288, 500, "596.36", uuid="GPU-bbbb-2222")
    _calibration_record("NVIDIA GeForce RTX 4070", card_uuid="bbbb-2222")        # torch spells it without GPU-
    assert work(other, cards=(this, other)) is None                              # made on a card this PC has
    assert work(this, cards=(this, other)) is None                               # ... whichever is index 0
    assert "graphics card changed" in work(this, cards=(this,))                  # that card is gone
    _calibration_record("NVIDIA GeForce RTX 4070")                               # an older record: by name
    assert work(this, cards=(this, other)) is None and "changed" in work(this, cards=(this,))
    # a run that ended with a reservation is never tried again by itself, however often: the artist's hours are
    # not for measuring (Settings > Debugging drops the record for a new measurement)
    _calibration_record(RTX_3060.name, complete=False, attempts=1)
    assert work(RTX_3060) is None
    assert components.drop_pixal3d_calibration() is True and "calibrated once" in work(RTX_3060)
    assert components.drop_pixal3d_calibration() is False


def _pixal3d_kernels_dist(version: str) -> None:
    """Pixal3D's venv with pixal3d_kernels of that version (its dist-info, as pip leaves it); "" takes it away."""
    site = paths.venv_dir(paths.PIXAL3D) / "Lib" / "site-packages"
    for old in site.glob("pixal3d_kernels-*.dist-info"):
        shutil.rmtree(old)
    if version:
        (site / f"pixal3d_kernels-{version}.dist-info").mkdir(parents=True)
        (site / f"pixal3d_kernels-{version}.dist-info" / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: pixal3d_kernels\nVersion: {version}\nSummary: x\n", encoding="utf-8")


def test_the_calibration_runs_the_steps_a_standing_record_lacks_alone(layout: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """Four races and choices that came after records were made, each due alone where it serves (never as a new
    version of the whole calibration): the group GEMMs' CUDA kernel against Triton's (sm_80+, in the Triton step's
    int8_tiles), spconv's kernels for the decoders (cards whose decoders run them: Turing and Volta), the element-wise
    kernels of the old cards (below sm_80), the decoders' GEMM-call floor for the decode in boxes (every card whose
    decoders run on spconv: below sm_80). A race made with another pixal3d_kernels build than the venv's is due
    again; one that ran and failed stands; with the venv's build unreadable, or one without those kernels, nothing
    is due."""
    monkeypatch.setattr(components, "_cards", lambda: ())
    monkeypatch.setattr(paths, "VENVS", layout / "code" / "venvs")
    _calibration_checkout()
    work = components.pixal3d_calibration_work
    site = paths.venv_dir(paths.PIXAL3D) / "Lib" / "site-packages"
    (site / "triton").mkdir(parents=True)
    (site / "triton" / "__init__.py").write_text("__version__ = '3.4.0'\n", encoding="utf-8")
    assert components.pixal3d_kernels_version() is None
    _pixal3d_kernels_dist("0.2.0")
    assert components.pixal3d_kernels_version() == (0, 2, 0)
    turing = gpu.GpuInfo(0, "NVIDIA GeForce RTX 2070", 7.5, 8192, 500, "596.36")
    volta = gpu.GpuInfo(0, "NVIDIA TITAN V", 7.0, 12288, 500, "596.36")
    pascal = gpu.GpuInfo(0, "NVIDIA GeForce GTX 1080", 6.1, 8192, 500, "596.36")
    race = {"ok": True, "profile": "turing", "kernels": "0.2.0", "on": {"modulate": True}, "timings": []}
    decoders = {"ok": True, "entries": [{"key": 1}]}
    tiles = {"ok": True, "fc1_group": [64, 128, 128, 8, 3, 4], "triton": "3.4.0"}
    group = {"ok": True, "triton": "3.4.0", "kernels": "0.2.0", "choice": {"4608x1536.bf16": 2}}

    # (a) sm_80+: a tile choice without the group GEMMs' race, or with one of another kernels build: the Triton step
    for raced, due in ((None, True), (group, False), ({**group, "kernels": "0.1.0"}, True),
                       ({"ok": False, "why": "pixal3d_kernels has no sm_80 group GEMM for this card"}, False),
                       ({"ok": False, "error": "RuntimeError: x"}, False)):
        found = {**tiles, **({"group_cuda": raced} if raced is not None else {})}
        _calibration_record(RTX_3060.name, triton={"ok": True, "int8": True, "int8_tiles": found})
        assert work(RTX_3060) == (components.INT8_TILES_CALIBRATION_WORK if due else None), raced
    _calibration_record(RTX_3060.name, triton={"ok": True, "int8": True, "int8_tiles": {**tiles, "ok": False}})
    assert work(RTX_3060) is None                                   # a failed tile timing stands, race or none
    _calibration_record(RTX_3060.name, triton={"ok": True, "int8": True, "int8_tiles": tiles})
    for version, due in (("0.1.0", False), ("", False), ("0.2.0", True), ("0.3.0", True)):
        _pixal3d_kernels_dist(version)                              # a build without the kernel, none, newer ones
        assert work(RTX_3060) == (components.INT8_TILES_CALIBRATION_WORK if due else None), version
    _pixal3d_kernels_dist("0.2.0")
    _calibration_record(RTX_3060.name, triton={"ok": True, "int8": True, "int8_tiles": tiles})   # never the others
    assert components.SPCONV_DECODERS_CALIBRATION_WORK not in work(RTX_3060)
    assert components.OLD_CARDS_CALIBRATION_WORK not in work(RTX_3060)

    # (b) the decoders' spconv choices: Turing and Volta, where the decoders run spconv's own GEMM kernels
    floors = {"decode_products": PRODUCTS}                          # (their own rule: (d))
    for card in (turing, volta):
        _calibration_record(card.name, old_cards=race, **floors)
        assert work(card) == components.SPCONV_DECODERS_CALIBRATION_WORK, card
    for held in (decoders, {"ok": True, "skipped": "spconv is not here", "entries": []},
                 {"ok": False, "error": "RuntimeError: x"}):
        _calibration_record(turing.name, old_cards=race, spconv_decoders=held, **floors)
        assert work(turing) is None, held                           # chosen, skipped or failed: it stands
    _calibration_record(turing.name, old_cards=race, conv_backend={"ok": True, "backend": "flex_gemm"})
    assert work(turing) is None                                     # its decoders run flex_gemm's kernels
    _calibration_record(turing.name, old_cards=race, conv_backend={"ok": True, "backend": "spconv"}, **floors)
    assert work(turing) == components.SPCONV_DECODERS_CALIBRATION_WORK
    _calibration_record(pascal.name, old_cards={**race, "profile": "pascal"}, **floors)
    assert work(pascal) is None                                     # Pascal's decoders take torch products
    _calibration_record(RTX_3060.name)
    assert work(RTX_3060) is None                                   # Ampere: flex_gemm's

    # (c) the old cards' element-wise race, below sm_80
    _calibration_record(pascal.name, **floors)
    assert work(pascal) == components.OLD_CARDS_CALIBRATION_WORK
    for held, due in (({**race, "profile": "pascal"}, False), ({**race, "kernels": "0.1.0"}, True),
                      ({"ok": False, "error": "RuntimeError: x"}, False),
                      ({"ok": False, "error": "the kernels do not give the eager chains' bits here"}, False)):
        _calibration_record(pascal.name, old_cards=held, **floors)
        assert work(pascal) == (components.OLD_CARDS_CALIBRATION_WORK if due else None), held
    _calibration_record(pascal.name, **floors)
    for version in ("0.1.0", ""):
        _pixal3d_kernels_dist(version)                              # no element-wise kernels to race
        assert work(pascal) is None, version
    _pixal3d_kernels_dist("0.2.0")

    # (d) the decoders' GEMM-call floor for the decode in boxes (round O): every card whose decoders run on spconv
    for card in (turing, volta, pascal):
        _calibration_record(card.name, old_cards={**race, "profile": "pascal" if card is pascal else "turing"},
                            spconv_decoders=decoders)
        assert work(card) == components.DECODE_PRODUCTS_CALIBRATION_WORK, card
    for held in (PRODUCTS, {"ok": True, "profile": "pascal", "skipped": "the decoders run on flex_gemm here"},
                 {"ok": False, "error": "RuntimeError: x"}):
        _calibration_record(pascal.name, old_cards={**race, "profile": "pascal"}, decode_products=held)
        assert work(pascal) is None, held                           # measured, skipped or failed: it stands
    # a sweep cut short by a passing cause (the fork then keeps spconv's boxes off: the 1024 safety net) is due
    # again, alone, whatever the record's own "attempts" say (the hub counts the retries: the test below)
    failed = {"ok": False, "error": "OutOfMemoryError: CUDA out of memory", "seconds": 3.1}
    for card in (turing, pascal):
        for attempts in (None, 1, 5):
            _calibration_record(card.name, old_cards={**race, "profile": "pascal" if card is pascal else "turing"},
                                spconv_decoders=decoders, decode_products=failed,
                                **({"attempts": attempts} if attempts is not None else {}))
            assert work(card) == components.CALIBRATION_RETRY_WORK["decode_products"], (card, attempts)
    # a section that measured shapes in an older form than the fork's (round P's review: the 2^20 boundary): once more,
    # alone, the section dropped from the record first (the fork's calibrate keeps a section that passed)
    measured = {"ok": True, "profile": "pascal", "shapes": [{"kind": "products", "ci": 128, "co": 128}]}
    for form, due in ((None, True), (2, True), (components.DECODE_PRODUCTS_FORM, False), ("x", True)):
        held = {**measured, **({"form": form} if form is not None else {})}
        _calibration_record(pascal.name, old_cards={**race, "profile": "pascal"}, decode_products=held)
        assert work(pascal) == (components.DECODE_PRODUCTS_AGAIN_WORK if due else None), form
    for held in ({**measured, "ok": False, "error": "x"}, {**measured, "shapes": []}):
        _calibration_record(pascal.name, old_cards={**race, "profile": "pascal"}, decode_products=held)
        assert work(pascal) is None, held                           # a failed or a shapeless section stands
    _calibration_record(RTX_3060.name, triton={"ok": True, "int8": True, "int8_tiles": {**tiles, "group_cuda": group}},
                        decode_products={**measured, "form": 2})
    assert work(RTX_3060) is None                                   # Ampere: never
    _calibration_record(pascal.name, old_cards={**race, "profile": "pascal"}, decode_products={**measured, "form": 2},
                        spconv_decoders=decoders)
    monkeypatch.setattr(components, "_card", lambda: pascal)
    ctx = _RecordingCtx()
    components.Pixal3D()._calibrate(ctx, until=1.0)
    assert ctx.commands[-1][-2:] == ["--only", "decode_products"]
    kept = json.loads(components.pixal3d_calibration_file().read_text(encoding="utf-8"))
    assert "decode_products" not in kept and kept["spconv_decoders"] == decoders and kept["card"] == pascal.name
    _calibration_record(turing.name, old_cards=race, spconv_decoders=decoders,
                        conv_backend={"ok": True, "backend": "flex_gemm"})
    assert work(turing) is None                                     # its decoders run flex_gemm's kernels
    _calibration_record(RTX_3060.name, triton={"ok": True, "int8": True, "int8_tiles": {**tiles, "group_cuda": group}})
    assert work(RTX_3060) is None                                   # Ampere: never (flex_gemm's boxes)
    monkeypatch.setattr(components, "_card", lambda: pascal)
    _calibration_record(pascal.name, old_cards={**race, "profile": "pascal"})
    ctx = _RecordingCtx()
    components.Pixal3D()._calibrate(ctx, until=1.0)
    assert ctx.commands[-1][-2:] == ["--only", "decode_products"]

    # all three on a Turing record made before them: one run of the three steps alone
    _calibration_record(turing.name, triton={"ok": True, "int8": True})
    both = work(turing)
    assert both == (f"{components.SPCONV_DECODERS_CALIBRATION_WORK}; {components.OLD_CARDS_CALIBRATION_WORK}; "
                    f"{components.DECODE_PRODUCTS_CALIBRATION_WORK}")
    monkeypatch.setattr(components, "_card", lambda: turing)
    ctx = _RecordingCtx()
    components.Pixal3D()._calibrate(ctx, until=1.0)
    assert ctx.commands[-1][-2:] == ["--only", "spconv_decoders,old_cards,decode_products"]
    _calibration_record(turing.name, triton={"ok": True, "int8": True}, spconv_decoders=decoders, old_cards=race,
                        **floors)
    assert work(turing) is None


def test_a_turing_record_without_the_int8_attention_race_times_the_attention_alone(
        layout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Turing SageAttention kernel's int8 P.V twin is raced at install (the fork's sage._old_card_trial records
    "int8pv" in the "sage" section, whatever its result): a record whose Turing kernel ran but has no such race runs
    the attention step alone; a race made, failed or left out stands, and so does a section whose kernel did not run
    (or another card's)."""
    monkeypatch.setattr(components, "_cards", lambda: ())
    monkeypatch.setattr(paths, "VENVS", layout / "code" / "venvs")
    _calibration_checkout()
    work = components.pixal3d_calibration_work
    turing = gpu.GpuInfo(0, "NVIDIA GeForce RTX 2070", 7.5, 8192, 500, "596.36")
    standing = {"spconv_decoders": {"ok": True, "entries": [{"key": 1}]},        # (the venv: no kernels build)
                "decode_products": PRODUCTS}
    ran = {"wheel": None, "capability": [7, 5], "tuned": False, "candidate": "turing", "candidate_ok": True}
    _calibration_record(turing.name, sage=ran, **standing)
    assert work(turing) == components.TURING_INT8PV_CALIBRATION_WORK
    for section in ({**ran, "int8pv": True}, {**ran, "int8pv": False, "int8pv_why": "not 5 % faster"},
                    {**ran, "int8pv": False, "int8pv_why": "the installed wheel has no int8 P.V kernel for sm_75"},
                    {**ran, "candidate_ok": False, "why": "the trial failed"},
                    {"wheel": None, "tuned": False, "why": "PIXAL3D_SAGE is off", "summary": "off"}):
        _calibration_record(turing.name, sage=section, **standing)
        assert work(turing) is None, section
    _calibration_record(RTX_3060.name, sage={"wheel": None, "tuned": True, "timings": []})
    assert work(RTX_3060) is None                                   # Ampere's int8 race is its own section's
    monkeypatch.setattr(components, "_card", lambda: turing)
    _calibration_record(turing.name, sage=ran, **standing)
    ctx = _RecordingCtx()
    components.Pixal3D()._calibrate(ctx, until=1.0)
    assert ctx.commands[-1][-2:] == ["--only", "sage"]
    _calibration_record(turing.name, sage=ran, decode_products=PRODUCTS)   # with the decoders' choices lacking too
    ctx = _RecordingCtx()
    components.Pixal3D()._calibrate(ctx, until=1.0)
    assert ctx.commands[-1][-2:] == ["--only", "spconv_decoders,sage"]


def test_a_new_sage_wheel_times_the_attention_alone(layout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A record whose attention section was made with another SageAttention wheel than the pinned one (Spellcast3D
    round O: post8 brings the Pascal kernel's twins, raced at install) runs the attention step alone; a section of the
    pinned wheel, or one without a wheel (SageAttention off), stands; with another step lacking, both in one run."""
    monkeypatch.setattr(components, "_cards", lambda: ())
    monkeypatch.setattr(paths, "VENVS", layout / "code" / "venvs")
    _calibration_checkout()
    work = components.pixal3d_calibration_work
    pascal = gpu.GpuInfo(0, "NVIDIA GeForce GTX 1080", 6.1, 8192, 500, "596.36")
    monkeypatch.setattr(components, "pixal3d_sage_wheel_version", lambda: "2.2.0+cu128torch2.8.0.post8")
    ran = {"capability": [6, 1], "tuned": False, "candidate": "pascal", "candidate_ok": True, "int8pv": True}
    floors = {"decode_products": {**PRODUCTS, "profile": "pascal"}}
    _calibration_record(pascal.name, sage={**ran, "wheel": "2.2.0+cu128torch2.8.0.post7"}, **floors)
    assert work(pascal) == components.SAGE_WHEEL_CALIBRATION_WORK
    for section in ({**ran, "wheel": "2.2.0+cu128torch2.8.0.post8"}, {"wheel": None, "tuned": False}):
        _calibration_record(pascal.name, sage=section, **floors)
        assert work(pascal) is None, section
    monkeypatch.setattr(components, "_card", lambda: pascal)
    _calibration_record(pascal.name, sage={**ran, "wheel": "2.2.0+cu128torch2.8.0.post7"}, **floors)
    ctx = _RecordingCtx()
    components.Pixal3D()._calibrate(ctx, until=1.0)
    assert ctx.commands[-1][-2:] == ["--only", "sage"]
    turing = gpu.GpuInfo(0, "NVIDIA GeForce RTX 2070", 7.5, 8192, 500, "596.36")
    monkeypatch.setattr(components, "_card", lambda: turing)
    _calibration_record(turing.name, sage={"wheel": "2.2.0+cu128torch2.8.0.post7", "capability": [7, 5],
                                           "tuned": False, "candidate": "turing", "candidate_ok": True},
                        decode_products=PRODUCTS)
    assert work(turing) == f"{components.SPCONV_DECODERS_CALIBRATION_WORK}; {components.SAGE_WHEEL_CALIBRATION_WORK}"
    ctx = _RecordingCtx()
    components.Pixal3D()._calibrate(ctx, until=1.0)
    assert ctx.commands[-1][-2:] == ["--only", "spconv_decoders,sage"]      # (the int8 P.V race: in the sage step)


def test_an_old_cards_step_cut_short_runs_again_twice_at_most(layout: Path, monkeypatch: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    """Below sm_80 the decode-in-boxes check, the attention trial and the decoders' spconv choices have no stand-in at
    first use: one cut short by a passing cause (out of memory, a crashed or timed-out trial) runs again alone at the
    next start, CALIBRATION_RETRIES times at most for the card and the calibration version, counted by the hub before
    each run (the fork's "attempts" restarts with every complete record, and a failed trial leaves the record
    complete). A measured answer stands, an sm_80+ card never retries, and Settings' "Measure the graphics card
    again", another card or a whole calibration count anew."""
    monkeypatch.setattr(components, "_cards", lambda: ())
    monkeypatch.setattr(paths, "VENVS", layout / "code" / "venvs")
    monkeypatch.setattr(components, "_old_cards_race_lacking", lambda section: False)
    _calibration_checkout()
    work = components.pixal3d_calibration_work
    pascal = gpu.GpuInfo(0, "NVIDIA GeForce GTX 1080", 6.1, 8192, 500, "596.36")
    turing = gpu.GpuInfo(0, "NVIDIA GeForce RTX 2070", 7.5, 8192, 500, "596.36")
    ran = {"ok": True, "capability": [6, 1], "candidate": "pascal", "candidate_ok": True, "int8pv": True,
           "pascal_twins": {}}
    floors = {"ok": True, "profile": "pascal", "shapes": [{"kind": "products"}], "form": components.DECODE_PRODUCTS_FORM}
    standing = {"old_cards": {"ok": True, "profile": "pascal", "kernels": "0.2.0", "on": {}}, "decode_products": floors}
    again = components.CALIBRATION_RETRY_WORK["sage"]
    for why, due in (("the trial process ended with code 3221225477: illegal memory access", True),
                     ("the trial did not finish in 600 s", True),
                     ("OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", True),
                     # the PC's own memory, as a 16 GB PC runs short of it: torch's allocator, Python, Windows
                     ("RuntimeError: [enforce fail at alloc_cpu.cpp:121] data. DefaultCPUAllocator: not enough memory: "
                      "you tried to allocate 1073741824 bytes.", True),
                     ("MemoryError: ", True),
                     ("OSError: [WinError 1455] The paging file is too small for this operation to complete", True),
                     ("the trial process ended with code 1: CUDA error: no kernel image is available", False),
                     ("its answer differs from fp32 attention (relative L2 0.21)", False),
                     ("the installed wheel has no pascal kernel", False)):
        _calibration_record(pascal.name, **standing, sage={**ran, "candidate_ok": False, "why": why})
        assert work(pascal) == (again if due else None), why

    def state() -> dict[str, Any]:
        return json.loads(paths.INSTALL_STATE.read_text(encoding="utf-8")) if paths.INSTALL_STATE.is_file() else {}

    # a record complete with its trial timed out (the fork writes "attempts": 1 after every such run), start after
    # start: run again twice, then it stands
    monkeypatch.setattr(components, "_card", lambda: pascal)
    timed_out = {**ran, "candidate_ok": False, "why": "the trial did not finish in 600 s"}
    runs = []
    for _start in range(5):
        _calibration_record(pascal.name, attempts=1, **standing, sage=timed_out)
        if work(pascal) is not None:
            ctx = _RecordingCtx()
            components.Pixal3D()._calibrate(ctx, until=1.0)
            runs.append(ctx.commands[-1][-2:])
    assert runs == [["--only", "sage"]] * components.CALIBRATION_RETRIES
    assert state()[components.CALIBRATION_RETRIES_KEY] == {
        "card": pascal.name, "version": components.PIXAL3D_CALIBRATION, "steps": {"sage": components.CALIBRATION_RETRIES}}
    other = gpu.GpuInfo(0, "NVIDIA GeForce GTX 1070", 6.1, 8192, 500, "596.36")
    _calibration_record(other.name, attempts=1, **standing, sage=timed_out)
    assert work(other) == again                                     # another card's record: its own count
    _calibration_record(pascal.name, attempts=1, **standing, sage=timed_out)
    assert work(pascal) is None
    assert components.drop_pixal3d_calibration() is True            # "Measure the graphics card again": the whole
    assert "calibrated once" in work(pascal)                        # calibration, which counts anew
    components.Pixal3D()._calibrate(_RecordingCtx(), until=1.0)
    assert components.CALIBRATION_RETRIES_KEY not in state()
    _calibration_record(pascal.name, attempts=1, **standing, sage=timed_out)
    assert work(pascal) == again
    components.Pixal3D()._calibrate(_RecordingCtx(), until=1.0)
    assert state()[components.CALIBRATION_RETRIES_KEY]["steps"] == {"sage": 1}
    _calibration_record(pascal.name, version=components.PIXAL3D_CALIBRATION - 1)
    components.Pixal3D()._calibrate(_RecordingCtx(), until=1.0)     # a new version: a whole calibration too
    assert components.CALIBRATION_RETRIES_KEY not in state()

    # several cut short: one run of those steps alone, behind the same notice as the first run
    _calibration_record(turing.name, old_cards={"ok": True, "profile": "turing"},
                        sage={**ran, "capability": [7, 5], "candidate": "turing", "candidate_ok": False,
                              "why": "the trial did not finish in 600 s"},
                        decode_products={"ok": False, "error": "RuntimeError: CUDA error: out of memory"},
                        spconv_decoders={"ok": False, "error": "torch.OutOfMemoryError: CUDA out of memory"})
    assert work(turing) == "; ".join(components.CALIBRATION_RETRY_WORK[s]
                                     for s in ("decode_products", "sage", "spconv_decoders"))
    monkeypatch.setattr(components, "_card", lambda: turing)
    capsys.readouterr()
    ctx = _RecordingCtx()
    components.Pixal3D()._calibrate(ctx, until=1.0)
    assert ctx.commands[-1][-2:] == ["--only", "decode_products,sage,spconv_decoders"]
    assert "keep heavy programs" in capsys.readouterr().out
    # an sm_80+ card's failed steps stand (each tunes itself at its first use)
    _calibration_record(RTX_3060.name, decode_products={"ok": False, "error": "OutOfMemoryError: CUDA out of memory"},
                        sage={"ok": False, "error": "OutOfMemoryError: CUDA out of memory"})
    assert work(RTX_3060) is None


def test_the_install_says_what_an_old_card_does_until_it_is_measured(layout: Path, monkeypatch: pytest.MonkeyPatch,
                                                                       capsys: pytest.CaptureFixture[str]) -> None:
    """After a calibration run that did not finish, below sm_80: what the card does without the steps the record
    lacks, and the next start's measurement promised only where it runs them (else the Settings button); an sm_80+
    card, or an old one that lacks nothing, keeps the general words."""
    monkeypatch.setattr(components, "_cards", lambda: ())
    monkeypatch.setattr(paths, "VENVS", layout / "code" / "venvs")
    monkeypatch.setattr(components, "_old_cards_race_lacking", lambda section: False)
    _calibration_checkout()
    turing = gpu.GpuInfo(0, "NVIDIA GeForce RTX 2070", 7.5, 8192, 500, "596.36")
    monkeypatch.setattr(components, "_card", lambda: turing)
    oom = {"ok": False, "error": "OutOfMemoryError: CUDA out of memory"}
    trial = {"ok": True, "candidate": "turing", "candidate_ok": True, "int8pv": True}
    held = {"sage": trial, "spconv_decoders": {"ok": True, "entries": [{"key": 1}]}}
    lower = "Until it does, High (1536) 3D models may be made at Standard (1024) detail."
    failed = components.PIXAL3D_CALIBRATION_FAILED_OLD_CARD
    _calibration_record(turing.name, **held, decode_products=oom)
    ctx = _RecordingCtx()
    ctx.fail_first = True
    components.Pixal3D()._calibrate(ctx, until=1.0)                 # retry 1 of 2: the next start measures again
    assert f"{failed} {lower} {components.MEASURED_AT_NEXT_START}" in capsys.readouterr().out
    ctx = _RecordingCtx()
    ctx.fail_first = True
    components.Pixal3D()._calibrate(ctx, until=1.0)                 # retry 2 of 2: the Settings button alone
    out = capsys.readouterr().out
    assert f"{failed} {lower} {components.MEASURED_ON_REQUEST}" in out
    assert components.PIXAL3D_CALIBRATION_FAILED not in out

    class MeasuredNow(_RecordingCtx):
        """calibrate.py ends a run of some steps of a record with a reservation with 1, the steps measured or not."""

        def run(self, argv: list[str], **kwargs: Any) -> None:
            super().run(argv, **kwargs)
            _calibration_record(turing.name, complete=False, **held, decode_products=PRODUCTS)
            raise InstallError(kwargs["fail"])

    components._forget_retries()
    _calibration_record(turing.name, complete=False, **held, decode_products=oom)
    components.Pixal3D()._calibrate(MeasuredNow(), until=1.0)       # the step measured now: no warning
    out = capsys.readouterr().out
    assert "keep heavy programs" in out and "Warning" not in out
    words = components._calibration_failed_words
    _calibration_record(turing.name, **{**held, "sage": {**trial, "candidate_ok": False, "why": "its answer differs"}},
                        decode_products=PRODUCTS)
    assert words() == f"{failed} Until it does, 3D models take longer. {components.MEASURED_ON_REQUEST}"
    _calibration_record(turing.name, complete=False, **held, decode_products=PRODUCTS)
    assert words() == components.PIXAL3D_CALIBRATION_FAILED          # a busy card, every step passed: it stands
    components.pixal3d_calibration_file().unlink()                  # no record at all: the whole run again
    assert words() == (f"{failed} Until it does, High (1536) 3D models may be made at Standard (1024) detail and 3D "
                       f"models take longer. {components.MEASURED_AT_NEXT_START}")
    titan = gpu.GpuInfo(0, "NVIDIA TITAN RTX", 7.5, 24576, 500, "596.36")
    monkeypatch.setattr(components, "_card", lambda: titan)
    _calibration_record(titan.name, **held, decode_products={"ok": False, "error": "RuntimeError: x"})
    assert words() == components.PIXAL3D_CALIBRATION_FAILED          # 24 GB: 1536 without the boxes too
    monkeypatch.setattr(components, "_card", lambda: RTX_3060)
    _calibration_record(RTX_3060.name, decode_products=oom)
    assert words() == components.PIXAL3D_CALIBRATION_FAILED


def test_pixal3ds_launcher_is_told_whether_the_next_start_runs_the_decode_check(
        layout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The launcher's words for a prop made at less detail without the decode-in-boxes check promise the next start's
    measurement only where the hub runs that step then (pixal3d_step_due, told at the launch)."""
    from spellcast.adapters.pixal3d import Pixal3DAdapter

    step_due = components.pixal3d_step_due
    turing = gpu.GpuInfo(0, "NVIDIA GeForce RTX 2070", 7.5, 8192, 500, "596.36")
    for due, flag in ((True, "1"), (False, "0")):
        monkeypatch.setattr(components, "pixal3d_step_due", lambda step, info, due=due: due and step == "decode_products")
        adapter = Pixal3DAdapter()
        adapter.__dict__["_gpu"] = turing
        assert adapter.launch_spec(7772, {}).env["SPELLCAST_PIXAL3D_CALIBRATION_DUE"] == flag
    adapter = Pixal3DAdapter()
    adapter.__dict__["_gpu"] = None
    assert "SPELLCAST_PIXAL3D_CALIBRATION_DUE" not in adapter.launch_spec(7772, {}).env
    # the rule: that step alone, or a whole calibration
    for why, due in ((None, False), (components.CALIBRATION_RETRY_WORK["decode_products"], True),
                     (components.DECODE_PRODUCTS_CALIBRATION_WORK, True),
                     (f"{components.OLD_CARDS_CALIBRATION_WORK}; {components.CALIBRATION_RETRY_WORK['sage']}", False),
                     ("it is calibrated once for this PC's graphics card", True)):
        monkeypatch.setattr(components, "pixal3d_calibration_work", lambda info=None, why=why: why)
        assert step_due("decode_products", turing) is due, why


def test_pixal3d_calibrates_last_and_a_failed_calibration_is_only_a_warning(
        layout: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(components, "_card", lambda: RTX_3060)
    pixal = components.Pixal3D()
    ctx = _RecordingCtx()
    pixal._calibrate(ctx, until=1.0)
    assert ctx.commands == []                           # a checkout without it: nothing runs
    _calibration_checkout()
    pixal._calibrate(ctx, until=1.0)
    assert ctx.commands == [[pixal._python(), "-m", "pixal3d.calibrate", "--version",
                             str(components.PIXAL3D_CALIBRATION), "--skip", "spconv"]]   # Hi3D's models not here
    assert ctx.cwds == [paths.PIXAL3D_DIR] and ctx.envs[-1]["HF_HUB_OFFLINE"] == "1"
    assert ctx.envs[-1].get("CUDA_VISIBLE_DEVICES") != "-1"         # it measures the card
    monkeypatch.setattr(models, "hi3d_on_disk", lambda folder=None: True)      # Hi3D's models here: its step too
    pixal._calibrate(ctx, until=1.0)
    assert ctx.commands[-1][-2:] == ["--version", str(components.PIXAL3D_CALIBRATION)]
    monkeypatch.setattr(components, "pixal3d_calibration_work",
                        lambda gpu_info=None: components.HI3D_CALIBRATION_WORK)
    pixal._calibrate(ctx, until=1.0)                    # only Hi3D's step is missing: it runs alone
    assert ctx.commands[-1][-2:] == ["--only", "spconv"]
    monkeypatch.setattr(components, "pixal3d_calibration_work",
                        lambda gpu_info=None: components.INT8_TILES_CALIBRATION_WORK)
    pixal._calibrate(ctx, until=1.0)                    # only the int8 tile choice is missing: the Triton step alone
    assert ctx.commands[-1][-2:] == ["--only", "triton"]
    for why, only in ((components.SPCONV_DECODERS_CALIBRATION_WORK, "spconv_decoders"),
                      (components.OLD_CARDS_CALIBRATION_WORK, "old_cards"),
                      (components.DECODE_PRODUCTS_CALIBRATION_WORK, "decode_products"),
                      (f"{components.INT8_TILES_CALIBRATION_WORK}; {components.HI3D_CALIBRATION_WORK}", "triton,spconv")):
        monkeypatch.setattr(components, "pixal3d_calibration_work", lambda gpu_info=None, why=why: why)
        pixal._calibrate(ctx, until=1.0)                # steps a standing record lacks: those alone, in one run
        assert ctx.commands[-1][-2:] == ["--only", only]
    monkeypatch.setattr(components, "pixal3d_calibration_work",
                        lambda gpu_info=None: f"{components.OLD_CARDS_CALIBRATION_WORK}; a new version")
    pixal._calibrate(ctx, until=1.0)                    # anything else with them: the whole calibration
    assert "--only" not in ctx.commands[-1]
    monkeypatch.setattr(components, "pixal3d_calibration_work", lambda gpu_info=None: "calibrated once")
    assert "keep heavy programs" in capsys.readouterr().out
    ctx.fail_first = True
    pixal._calibrate(ctx, until=1.0)                    # it failed: a warning, the install goes on
    assert components.PIXAL3D_CALIBRATION_FAILED in capsys.readouterr().out

    order: list[str] = []
    for name in ("_download_models", "_make_copies", "_link_models", "_link_user_hf_cache", "_pip_tested_versions",
                 "_verify", "_calibrate"):
        monkeypatch.setattr(components.Pixal3D, name,
                            lambda self, *args, _name=name, **kwargs: order.append(_name))
    monkeypatch.setattr(components, "share_torch_files", lambda ctx, venv_id: order.append("share"))
    monkeypatch.setattr(components, "pixal3d_calibration_work", lambda gpu_info=None: "due")
    pixal.install(_RecordingCtx())
    assert order[-1] == "_calibrate" and order.index("_verify") < order.index("_calibrate")   # last
    monkeypatch.setattr(components, "pixal3d_calibration_work", lambda gpu_info=None: None)
    order.clear()
    pixal.install(_RecordingCtx())                      # an update of the checkout: the card was measured already
    assert "_calibrate" not in order and order[-1] == "_verify"
    monkeypatch.setattr(components.Pixal3D, "missing_files", lambda self: [])
    monkeypatch.setattr(models, "pixal3d_link_work", lambda: None)
    monkeypatch.setattr(components, "pixal3d_calibration_work", lambda gpu_info=None: "due")
    assert pixal.repair_work() == "due"                 # an installed Pixal3D on a new card
    order.clear()
    pixal.repair(_RecordingCtx())
    assert order == ["_link_models", "_calibrate"]      # nothing downloaded, only measured
    monkeypatch.setattr(components, "pixal3d_calibration_work", lambda gpu_info=None: None)
    order.clear()
    pixal.repair(_RecordingCtx())
    assert order == ["_link_models"]                    # measured already: never again


def test_the_calibration_notice_comes_first_and_waits_for_enter(monkeypatch: pytest.MonkeyPatch,
                                                                capsys: pytest.CaptureFixture[str]) -> None:
    class Console:
        def isatty(self) -> bool:
            return True

    waited: list[str] = []
    monkeypatch.setattr(install_main, "_gpu_info", lambda: RTX_3060)
    monkeypatch.setattr(install_main, "pixal3d_calibration_work", lambda info: "due")
    monkeypatch.setattr(install_main, "_wait_for_enter", waited.append)
    monkeypatch.setattr(sys, "stdin", Console())
    monkeypatch.delenv("SPELLCAST_NO_PAUSE", raising=False)
    install_main._calibration_notice(Output(transient=False))
    shown = capsys.readouterr().out
    assert "CALIBRATING FOR PERFORMANCE." in shown and "BLENDER, PHOTOSHOP" in shown
    assert waited == [install_main.CALIBRATION_PROMPT] and "press Enter" in install_main.CALIBRATION_PROMPT
    install_main._calibration_notice(Output(transient=False), wait=False)     # install.py --yes: shown, not waited for
    assert "CALIBRATING" in capsys.readouterr().out and waited == [install_main.CALIBRATION_PROMPT]
    monkeypatch.setenv("SPELLCAST_NO_PAUSE", "1")       # tests and automation: shown, never waited for
    install_main._calibration_notice(Output(transient=False))
    assert "CALIBRATING" in capsys.readouterr().out and waited == [install_main.CALIBRATION_PROMPT]
    monkeypatch.setattr(install_main, "pixal3d_calibration_work", lambda info: None)
    install_main._calibration_notice(Output(transient=False))      # nothing due: nothing shown
    assert capsys.readouterr().out == ""


def test_forge_gets_the_models_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folders = tuple((flag, tmp_path / "models" / folder.name) for flag, folder in forge.MODEL_FOLDERS)
    monkeypatch.setattr(forge, "MODEL_FOLDERS", folders)
    monkeypatch.setattr(forge.forge_config, "ensure_forge_config", lambda: None)
    argv = forge.ForgeAdapter().launch_spec(7781, {"forge": {"ckpt_dirs": ["D:/my models"]}}).argv
    pairs = list(zip(argv, argv[1:]))
    for flag, folder in folders:
        assert (flag, str(folder)) in pairs and folder.is_dir()
    assert ("--ckpt-dirs", str(Path("D:/my models"))) in pairs
    cmd_args = paths.FORGE_DIR / "modules" / "cmd_args.py"
    if cmd_args.is_file():      # a Forge update that renames a flag would make Forge refuse to start
        text = cmd_args.read_text(encoding="utf-8")
        assert all(f'"{flag}"' in text for flag, _ in forge.MODEL_FOLDERS)


def test_pixal3d_launcher_stops_when_its_models_folder_is_gone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    import pixal3d_server

    (tmp_path / "app.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["pixal3d_server.py", "--repo", str(tmp_path), "--port", "7782"])
    assert pixal3d_server.main() == 2
    assert "models\\pixal3d" in capsys.readouterr().out


# ---- preflight: install folder ---------------------------------------------------------------------

@needs_non_ansi
def test_a_folder_outside_the_ansi_code_page_is_fatal_for_pixal3d(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "ROOT", Path(r"C:\Users\Łukasz\spellcast3d"))
    pixal = next(c for c in components.COMPONENTS if c.id == paths.PIXAL3D)
    hub = next(c for c in components.COMPONENTS if c.id == paths.HUB)
    r = preflight.Report()
    preflight._check_folder(r, [hub, pixal])
    assert any("non-Unicode" in f and "C:\\spellcast3d" in f for f in r.fatal)
    r = preflight.Report()
    preflight._check_folder(r, [hub])
    assert not r.fatal and any("non-Unicode" in w for w in r.warnings)


def test_accented_letters_of_the_ansi_code_page_are_fine(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _ansi_has("José"):
        pytest.skip("this PC's ANSI code page has no é")
    monkeypatch.setattr(paths, "ROOT", Path(r"C:\Users\José\spellcast3d"))
    r = preflight.Report()
    preflight._check_folder(r, list(components.COMPONENTS))
    assert not r.fatal and not r.warnings


# ---- shared helpers ------------------------------------------------------------------------------------

def test_write_text_atomic_retries_a_locked_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = os.replace
    fails = [2]

    def flaky(src: Any, dst: Any) -> None:
        if fails[0]:
            fails[0] -= 1
            raise PermissionError("in use")
        real(src, dst)

    monkeypatch.setattr(paths.os, "replace", flaky)
    monkeypatch.setattr(paths.time, "sleep", lambda s: None)
    target = tmp_path / "x.json"
    paths.write_text_atomic(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello" and list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("start", [None, "{broken", "[1, 2]", {"VERSION_UID": "OLD", "outdir": "D:/Übersicht"},
                                   {"auto_launch_browser": "Local"}])
def test_forge_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start: Any) -> None:
    monkeypatch.setattr(forge_config, "forge_version_uid", lambda: "PY313")
    cfg = tmp_path / "config.json"
    if start is not None:
        cfg.write_text(start if isinstance(start, str) else json.dumps(start, ensure_ascii=False), encoding="utf-8")
    note = forge_config.ensure_forge_config(cfg)
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["VERSION_UID"] == "PY313"
    assert (note is not None) == (start == "{broken") == (tmp_path / "config.json.broken").exists()
    if isinstance(start, dict):
        assert {k: v for k, v in data.items() if k != "VERSION_UID"}.items() >= \
            {k: v for k, v in start.items() if k != "VERSION_UID"}.items()     # the user's own settings stay
    if isinstance(start, dict) and "outdir" in start:
        assert "Übersicht" in cfg.read_text(encoding="utf-8")                 # not escaped (Forge's own format)
    assert data["auto_launch_browser"] == (start.get("auto_launch_browser", "Disable") if isinstance(start, dict)
                                           else "Disable")
    stamp = cfg.stat().st_mtime_ns
    forge_config.ensure_forge_config(cfg)
    assert cfg.stat().st_mtime_ns == stamp                                     # already right: not rewritten


# ---- adapters: pure helpers ----------------------------------------------------------------------------

def _ctx(prop_dir: Path) -> StageContext:
    return StageContext(project="p", prop_id="p_000001", prop_dir=prop_dir, base_url="http://x", client=None,  # type: ignore[arg-type]
                        settings={}, report=lambda f, m: None, cancelled=lambda: False)


def test_forge_checkpoint_names() -> None:
    m = {"title": "sdxl/real.safetensors [31e35c80fc]", "hash": "31e35c80fc", "model_name": "sdxl_real",
         "filename": r"D:\models\sdxl\real.safetensors"}
    assert forge._checkpoint_name(m) == "sdxl/real.safetensors"
    for wanted in ("sdxl/real.safetensors", m["title"], "sdxl_real", "real.safetensors"):
        assert forge._find_checkpoint([m], wanted) is m
    assert forge._find_checkpoint([m], "other") is None


def test_forge_save_images_skips_grid_and_half_done(tmp_path: Path) -> None:
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 8).decode()
    info = {"index_of_first_image": 1, "infotexts": ["grid", "a", "b", "c"], "all_seeds": [11, 12, 13]}
    data = {"images": [png] * 5, "info": json.dumps(info)}          # grid + 3 images + 1 script extra
    saved, seeds = forge._save_images(_ctx(tmp_path), {"candidates": ["images/0007.png"]}, data, keep=None)
    assert saved == ["images/0008.png", "images/0009.png", "images/0010.png"] and seeds == [11, 12, 13]
    saved, seeds = forge._save_images(_ctx(tmp_path), {}, data, keep=1)   # cancelled after the 1st image
    assert len(saved) == 1 and seeds == [11]
    assert not list((tmp_path / "images").glob("*.tmp"))


def test_forge_payload() -> None:
    p = forge._build_payload({"object_mode": True, "negative_prompt": " blurry ", "width": 1001, "batch_count": 0},
                             "barrel", "b64")
    # the words as typed: "3D-ready framing" (object_mode) is retired, nothing is added to either prompt
    assert (p["prompt"], p["negative_prompt"]) == ("barrel", "blurry") and p["width"] == 1000 and p["n_iter"] == 1
    assert p["init_images"] == ["b64"] and p["resize_mode"] == 2
    with pytest.raises(StageError, match="Steps"):
        forge._build_payload({"steps": "many"}, "x", None)


def test_imesh_retopo_params() -> None:
    p = imesh._retopo_params({"target_vertices": "5000", "crease_angle": 400, "smooth_iter": 99, "uv_leniency": -1})
    assert p["target_vertices"] == 5000 and p["crease_angle"] == 180.0 and p["smooth_iter"] == 10
    assert p["uv_leniency"] == 0.0
    assert (p["rosy"], p["posy"], p["extrinsic"], p["align_to_boundaries"]) == (4, 4, True, False)
    assert imesh._retopo_params({"rosy": 5, "posy": 6})["rosy"] == 4          # not a symmetry: quads
    with pytest.raises(StageError, match="between"):
        imesh._retopo_params({"target_vertices": 50})


def test_imesh_accepted_settings_are_the_viewers_checked() -> None:
    """What the launcher reports with an accepted mesh becomes prop.retopo.settings; nothing unusable does."""
    reported = {"target_vertices": 3390, "pure_quad": True, "crease_angle": 30.0, "smooth_iter": 3,
                "uv_leniency": 0.25, "rosy": 6, "posy": 3, "extrinsic": False, "align_to_boundaries": True,
                "scale": 0.02, "deterministic": True}
    assert imesh.accepted_settings(reported) == {key: reported[key] for key in imesh.VIEWER_SETTINGS}
    assert imesh.accepted_settings({"target_vertices": 500_000, "rosy": 5, "posy": 3, "smooth_iter": "2",
                                    "crease_angle": None, "uv_leniency": 7}) == {"posy": 3, "uv_leniency": 1.0}
    assert imesh.accepted_settings(None) == {} and imesh.accepted_settings([1]) == {}


def _module_constants(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


@pytest.mark.parametrize("adapter, launcher, names", [
    (imesh, "imesh_server.py", ("MIN_TARGET_VERTICES", "MAX_TARGET_VERTICES", "MAX_SMOOTH_ITER")),
    (pixal3d, "pixal3d_server.py", ("ATTN_BACKENDS",)),
    (lato2, "lato2_server.py", ("MIN_VERTICES", "MAX_VERTICES")),
    (triflow, "triflow_server.py", ("MIN_FACES", "MAX_FACES")),
])
def test_launcher_constants_match_adapters(adapter: Any, launcher: str, names: tuple[str, ...]) -> None:
    """ "Keep in sync" constants between the hub's adapters and the launchers (other venvs)."""
    consts = _module_constants(paths.LAUNCHERS / launcher)
    for name in names:
        assert consts[name] == getattr(adapter, name), name


def test_retopo_numbers_match_the_retopology_adapters() -> None:
    """projects.RETOPO_NUMBERS (what the page may save) and DEFAULT_STAGE_SETTINGS agree with the ranges and defaults
    LATO.2's and TriFlow's adapters check a job against, and every size the page may save is asked of each tool as a
    count it takes: Instant Meshes' vertices in full, TriFlow's triangles from its least (a quarter of a thousand
    vertices), LATO.2's vertices from its least to its models' own bound - outside those, the nearest."""
    from spellcast.projects import DEFAULT_STAGE_SETTINGS, RETOPO_MAX_SIZE, RETOPO_MIN_SIZE, RETOPO_NUMBERS

    assert RETOPO_NUMBERS["lato2_seed"][:3] == (True, 0, lato2.MAX_SEED)
    assert RETOPO_NUMBERS["triflow_quad_ratio"][:3] == (False, 0.0, 1.0)
    assert RETOPO_NUMBERS["triflow_seed"][:3] == (True, 0, triflow.MAX_SEED)
    defaults = DEFAULT_STAGE_SETTINGS["retopo"]
    assert {key: defaults[key] for key in triflow.DEFAULTS} == triflow.DEFAULTS
    steps = round((RETOPO_MAX_SIZE - RETOPO_MIN_SIZE) * 100)
    sizes = [round(RETOPO_MIN_SIZE + step / 100, 2) for step in range(steps + 1)]     # every size with two decimals
    assert (sizes[0], sizes[-1], len(sizes)) == (RETOPO_MIN_SIZE, RETOPO_MAX_SIZE, 9891)
    assert (triflow.faces_for(RETOPO_MIN_SIZE), triflow.faces_for(RETOPO_MAX_SIZE)) == (triflow.MIN_FACES,
                                                                                        triflow.MAX_FACES)
    assert [triflow.faces_for(size) for size in (0.1, 0.24, 0.25, 0.26, 2, 12.25)] == [500, 500, 500, 520, 4000, 24_500]
    assert all(triflow.MIN_FACES <= triflow.faces_for(size) <= triflow.MAX_FACES for size in sizes)
    assert all(lato2.MIN_VERTICES <= lato2.vertices_for(size) <= lato2.MAX_VERTICES for size in sizes)
    assert [lato2.vertices_for(size) for size in (0.1, 0.2, 0.25, 4, 10, 22)] == [200, 200, 250, 4000, 10_000, 10_000]
    assert all(imesh.vertices_for(size) == round(1000 * size) for size in sizes), "every size fits Instant Meshes"
    assert (imesh.vertices_for(0.1), imesh.vertices_for(0.3), imesh.vertices_for(12.25)) == (100, 300, 12_250)
    # both adapters accept exactly the saved range's ends and refuse one step past them
    for key, (whole, low, high, _name) in RETOPO_NUMBERS.items():
        check = lato2.lato2_params if key.startswith("lato2") else triflow.triflow_params
        for value in (low, high):
            check({key: value})
        step = 1 if whole else 0.01
        for value in (low - step, high + step):
            with pytest.raises(StageError):
                check({key: value})


# ---- launchers ------------------------------------------------------------------------------------------

def test_loopback_guard_refuses_other_hosts_and_origins() -> None:
    from _loopback_guard import LoopbackGuard

    app = FastAPI()

    @app.get("/x")
    def x() -> dict[str, bool]:
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        await socket.send_text("hi")
        await socket.close()

    client = TestClient(LoopbackGuard(app))
    assert client.get("/x", headers={"host": "127.0.0.1:7773"}).status_code == 200
    assert client.get("/x", headers={"host": "localhost:7773", "origin": "http://127.0.0.1:7770"}).status_code == 200
    assert client.get("/x", headers={"host": "attacker.example:7773"}).status_code == 403    # DNS rebinding
    assert client.get("/x", headers={"host": "127.0.0.1:7773", "origin": "http://evil.example"}).status_code == 403
    with client.websocket_connect("/ws", headers={"host": "127.0.0.1:7773"}) as socket:
        assert socket.receive_text() == "hi"
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"host": "attacker.example:7773"}) as socket:
            socket.receive_text()


@needs_non_ansi
def test_pixal3d_gets_a_temp_folder_its_compilers_can_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pixal3d_server

    prefix = tmp_path / "venv"
    prefix.mkdir()
    (prefix / "tmp" / "old-build").mkdir(parents=True)          # a killed run's leftovers
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setenv("TEMP", r"C:\Users\Łukasz\AppData\Local\Temp\no-such-folder")
    monkeypatch.setenv("TMP", r"C:\Users\Łukasz\AppData\Local\Temp\no-such-folder")
    pixal3d_server.use_ansi_safe_temp()
    assert os.environ["TEMP"] == os.environ["TMP"] == str(prefix / "tmp")
    assert not (prefix / "tmp" / "old-build").exists()
    monkeypatch.setenv("TEMP", str(tmp_path))                  # a TEMP that works is left alone
    pixal3d_server.use_ansi_safe_temp()
    assert os.environ["TEMP"] == str(tmp_path)


def test_the_calibration_lines_drive_the_step_text_and_bar() -> None:
    """install/components._calibration_line: what pixal3d.calibrate prints ("[CALIBRATE]  42% ...", or a line without
    a percentage) becomes the step's detail and its fraction; other output is left alone."""
    tracker = progress.Progress(None)
    tracker.begin({"pixal3d": (progress.PENDING, "")})
    tracker.start("pixal3d")
    tracker.step(progress.CALIBRATING_STEP, until=1.0)
    ctx = Ctx(out=Output(transient=False), dry_run=False, base_python=Path(sys.executable), progress=tracker)
    entry = tracker._data["components"]["pixal3d"]
    components._calibration_line(ctx, "[CALIBRATE]   0% checking that the graphics card is free")
    assert entry["step"] == f"{progress.CALIBRATING_STEP} · checking that the graphics card is free"
    assert entry["step_fraction"] == 0.0
    components._calibration_line(ctx, "[CALIBRATE] the graphics card is busy (46 % in use): waiting for it")
    assert entry["step"].endswith("· the graphics card is busy (46 % in use): waiting for it") and entry["step_fraction"] == 0.0
    components._calibration_line(ctx, "[CALIBRATE]  42% convolutions 1024x4096: 9 sizes in 136.2 s - shape 3 of 9")
    assert entry["step_fraction"] == 0.42 and entry["step"].endswith("shape 3 of 9")
    components._calibration_line(ctx, "[SPARSE] Conv backend: flex_gemm")        # not the calibration's own line
    assert entry["step"].endswith("shape 3 of 9")
    assert entry["step"].startswith(progress.CALIBRATING_STEP)                   # procman.calibrating() still sees it


def test_the_calibration_waits_for_the_hubs_word_that_its_tools_left_the_card(tmp_path: Path) -> None:
    """install/components._wait_for_the_hubs_tools: the installer a hub runs waits until the hub's card file says free
    (the hub stops its idle GPU tools meanwhile: procman.free_card_for_calibration), saying whom it waits for; with
    no word at all (no hub listening) or tools that never go, it goes on after its waits."""
    tracker = progress.Progress(None)
    tracker.begin({"pixal3d": (progress.PENDING, "")})
    tracker.start("pixal3d")
    tracker.step(progress.CALIBRATING_STEP, until=1.0)
    ctx = Ctx(out=Output(transient=False), dry_run=False, base_python=Path(sys.executable), progress=tracker)
    card = tmp_path / "calibration-card.json"
    wait = lambda **limits: components._wait_for_the_hubs_tools(                                 # noqa: E731
        ctx, card_file=card, poll=0.02, **{"answer_wait": 0.2, "tools_wait": 0.4, **limits})

    began = time.monotonic()
    assert wait() is False and 0.2 <= time.monotonic() - began < 2.0       # nobody answered: on it goes
    card.write_text(json.dumps({"free": True, "waiting_for": []}), encoding="utf-8")
    assert wait() is True
    card.write_text(json.dumps({"free": False, "waiting_for": ["Qwen-Image 2.1"]}), encoding="utf-8")
    threading.Timer(0.25, lambda: card.write_text(json.dumps({"free": True, "waiting_for": []}), encoding="utf-8")).start()
    assert wait(tools_wait=10.0) is True
    step = tracker._data["components"]["pixal3d"]["step"]
    assert step == f"{progress.CALIBRATING_STEP} · waiting for Qwen-Image 2.1 to finish and close"
    card.write_text(json.dumps({"free": False, "waiting_for": ["Qwen-Image 2.1"]}), encoding="utf-8")
    began = time.monotonic()
    assert wait() is False and 0.4 <= time.monotonic() - began < 3.0       # still there: measured beside it
    card.write_text("{half", encoding="utf-8")                              # being replaced: read as no word yet
    assert wait() is False


def test_a_download_that_is_through_no_longer_shows_its_size() -> None:
    """Progress.note: once pip goes on after a download (it unpacks for minutes), the entry's bytes are cleared -
    "583 KB of 583 KB" standing there read as stuck. A download under way, or a step's own measure, keeps them."""
    tracker = progress.Progress(None)
    tracker.begin({"lato2": (progress.PENDING, "")})
    tracker.start("lato2")
    tracker.step("Installing its Python packages", until=0.5)
    entry = tracker._data["components"]["lato2"]
    tracker.download(300, 583)
    tracker.note("Downloading triflow.whl")
    assert (entry["bytes_done"], entry["bytes_total"]) == (300, 583)
    tracker.download(583, 583)
    tracker.note("Installing collected packages: torch")
    assert (entry["bytes_done"], entry["bytes_total"], entry["message"]) == (None, None, "Installing collected packages: torch")
    tracker.step("Downloading its AI models", until=1.0, measure=lambda: (10, 10))
    tracker.download(10, 10)
    tracker.note("still linking")
    assert (entry["bytes_done"], entry["bytes_total"]) == (10, 10)          # a folder measured every second: kept


def test_pips_already_satisfied_lines_stay_off_the_console(capsys: pytest.CaptureFixture[str]) -> None:
    """runner.Ctx._line: "Requirement already satisfied:" lines are logged and counted, one transient line says that
    the installation is being verified; with verbose they show as they are."""
    tracker = progress.Progress(None)
    tracker.begin({"pixal3d": (progress.PENDING, "")})
    tracker.start("pixal3d")
    ctx = Ctx(out=Output(transient=False), dry_run=False, base_python=Path(sys.executable), progress=tracker)
    ctx._line("Requirement already satisfied: numpy==2.1 in c:\\venv (from torch)")
    ctx._line("Requirement already satisfied: pillow in c:\\venv")
    ctx._line("Collecting gradio==6.14.0")
    out = capsys.readouterr().out
    assert "Requirement already satisfied" not in out and "Collecting gradio==6.14.0" in out
    assert ctx._satisfied == 2
    assert tracker._data["components"]["pixal3d"]["message"] == "Collecting gradio==6.14.0"
    ctx.verbose = True
    ctx._line("Requirement already satisfied: pillow in c:\\venv")
    assert "Requirement already satisfied: pillow" in capsys.readouterr().out
