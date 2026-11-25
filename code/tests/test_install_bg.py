"""The hub's background installation (installer_bg.py): progress, tools in the "installing" state, jobs that
wait for their tool, another installer's lock, and the installer dying with the hub.

A scripted stand-in replaces install.py: it follows the steps of a JSON file (write a progress file, record
a component as installed, wait for a file, exit with a code).
Run with the hub venv:  <venv>\\Scripts\\python.exe -m pytest code\\tests -q
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import msvcrt
import os
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from spellcast import installer_bg, paths
from spellcast.gpu import GpuInfo
from spellcast.installer_bg import IDLE, PROGRESS_FILE, STOPPED_EARLY, BackgroundInstall, installer_argv, tail_lines
from spellcast.jobs import WAITING, JobQueue
from spellcast.procman import ProcessManager, ToolError
from spellcast.projects import ProjectStore
from spellcast.server import create_app
from test_core import FakeAdapter, make_settings, pid_alive

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Spellcast3D is Windows-only")

FAKE_INSTALLER = r'''
import json, os, sys, time
steps = json.load(open(sys.argv[1], encoding="utf-8"))
progress = sys.argv[sys.argv.index("--progress-file") + 1]
for kind, *args in steps:
    if kind == "progress":
        with open(progress + ".tmp", "w", encoding="utf-8") as f:
            json.dump(args[0], f)
        os.replace(progress + ".tmp", progress)
    elif kind == "state":
        with open(args[0], "w", encoding="utf-8") as f:
            json.dump(args[1], f)
    elif kind == "wait_for":
        while not os.path.exists(args[0]):
            time.sleep(0.02)
    elif kind == "sleep":
        time.sleep(args[0])
    elif kind == "print":
        print(args[0], flush=True)
    elif kind == "pid":
        with open(args[0], "w") as f:
            f.write(str(os.getpid()))
    elif kind == "exit":
        sys.exit(args[0])
'''


def progress(current: str | None, **components: tuple[str, str, float | None]) -> dict[str, Any]:
    """A progress file as install.py writes it: component=(status, step, fraction)."""
    return {"running": True, "finished": False, "error": None, "current": current,
            "components": {cid: {"status": status, "step": step, "fraction": fraction, "step_fraction": None,
                                 "bytes_done": None, "bytes_total": None, "message": ""}
                           for cid, (status, step, fraction) in components.items()}}


class Installer:
    """A BackgroundInstall running the stand-in installer with a script of steps."""

    def __init__(self, tmp_path: Path, steps: list[list[Any]]):
        self.script = tmp_path / "fake_install.py"
        self.script.write_text(FAKE_INSTALLER, encoding="utf-8")
        self.steps_file = tmp_path / "steps.json"
        self.set_steps(steps)
        self.lines: list[str] = []
        self.progress_file = tmp_path / "install-progress.json"
        self.bg = BackgroundInstall([sys.executable, str(self.script), str(self.steps_file), "--if-needed",
                                     "--skip", "hub", "--progress-file", str(self.progress_file)],
                                    cwd=tmp_path, progress_file=self.progress_file,
                                    lock_file=tmp_path / ".install.lock", echo=self.lines.append, lock_poll=0.1,
                                    page_url="http://127.0.0.1:7770", ready_quiet=0.0)

    def set_steps(self, steps: list[list[Any]]) -> None:
        self.steps_file.write_text(json.dumps(steps), encoding="utf-8")


class InstallingAdapter(FakeAdapter):
    """A fake tool whose install state is read from a file the stand-in installer writes."""

    def __init__(self, state_file: Path, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.state_file = state_file

    def install_status(self) -> dict[str, Any]:
        """The state file maps a tool to its status, or to [status, the installer's reason]."""
        try:
            entry = json.loads(self.state_file.read_text(encoding="utf-8")).get(self.tool_id, "not_installed")
        except (OSError, ValueError):
            entry = "not_installed"
        status, reason = entry if isinstance(entry, list) else (entry, "its download failed")
        return {"status": status, "reason": "" if status == "installed" else reason}


def adapters_with_state(tmp_path: Path, installed: dict[str, str]) -> tuple[Path, dict[str, InstallingAdapter]]:
    state = tmp_path / "state.json"
    state.write_text(json.dumps(installed), encoding="utf-8")
    return state, {"forge": InstallingAdapter(state, "forge", "images", True, tmp_path),
                   "imesh": InstallingAdapter(state, "imesh", "retopo", False, tmp_path)}


@contextlib.asynccontextmanager
async def running(tmp_path: Path, adapters: dict[str, Any], bg: BackgroundInstall) -> AsyncIterator[ProcessManager]:
    async with httpx.AsyncClient(trust_env=False) as client:
        pm = ProcessManager(adapters, make_settings(tmp_path), client, logs_dir=tmp_path / "logs", install=bg)
        await pm.startup()
        bg.start()
        try:
            yield pm
        finally:
            await pm.stop_all()
            await bg.close()


async def until(predicate: Any, timeout: float = 20.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


# ---- tools and jobs while installing ----------------------------------------------------------------

def test_tools_being_installed_wait_and_their_jobs_resume(tmp_path: Path) -> None:
    state, adapters = adapters_with_state(tmp_path, {"imesh": "installed"})
    go = tmp_path / "go"
    installer = Installer(tmp_path, [
        ["progress", progress("forge", forge=("running", "Downloading torch", 0.45), imesh=("skipped", "", None))],
        ["print", "Installing Forge Neo"],
        ["wait_for", str(go)],
        ["state", str(state), {"imesh": "installed", "forge": "installed"}],
        ["progress", {**progress(None, forge=("done", "", 1.0), imesh=("skipped", "", None)), "running": False,
                      "finished": True}],
        ["exit", 0]])
    store = ProjectStore(tmp_path / "projects")
    store.create_project("Demo")
    ids = [store.add_prop("Demo", f"Box {i}")["id"] for i in range(2)]
    order: list[tuple[str, dict[str, Any]]] = []
    adapters["forge"].calls = adapters["imesh"].calls = order

    async def main() -> None:
        async with running(tmp_path, adapters, installer.bg) as pm:
            await until(installer.bg.settled, what="the installer's plan")
            forge = pm.status("forge")
            assert forge["state"] == "installing" and not forge["installed"]
            assert forge["message"] == "Fake forge is being installed (45%): Downloading torch"
            assert pm.status("imesh")["state"] == "stopped"
            with pytest.raises(ToolError, match="being installed") as e:
                pm.start("forge")
            assert e.value.status_code == 409
            snap = installer.bg.snapshot()
            assert (snap["running"], snap["finished"], snap["current"], snap["error"]) == (True, False, "forge", None)
            assert snap["components"]["forge"]["fraction"] == 0.45 and snap["message"] is None

            queue = JobQueue(pm, store, pm.settings, adapters, pm.client)
            queue.start()
            images = await queue.submit("images", "Demo", [ids[0]])        # accepted: it waits
            await until(lambda: queue.get(images["id"])["progress"]["message"]
                        == "Waiting for Fake forge to finish installing (45%)", what="the waiting message")
            shared = await queue.submit("retopo", "Demo", [ids[0]], {"method": "imesh"})   # the same box: after its image
            other = await queue.submit("retopo", "Demo", [ids[1]], {"method": "imesh"})    # another box: goes first
            await until(lambda: queue.get(other["id"])["status"] == "done", what="the other job")
            assert queue.get(images["id"])["status"] == "queued"
            assert queue.get(shared["id"])["status"] == "queued"
            assert queue.get(shared["id"])["progress"]["message"] == WAITING
            assert "Installing Forge Neo" in installer.lines

            go.touch()
            await until(lambda: queue.get(shared["id"])["status"] == "done", what="the waiting jobs")
            assert queue.get(images["id"])["status"] == "done"
            assert [prop for prop, _params in order] == [ids[1], ids[0], ids[0]]
            await until(lambda: installer.bg.snapshot()["finished"], what="the installer to finish")
            snap = installer.bg.snapshot()
            assert snap["running"] is False and snap["error"] is None and snap["current"] is None
            assert snap["components"]["forge"]["status"] == "done"
            assert pm.status("forge")["installed"] is True
            await queue.close()

    asyncio.run(main())


@pytest.mark.parametrize("code, error", [
    (1, "Some parts of Spellcast3D could not be installed. The details are in code/logs/install.log."),
    (3, "another Spellcast3D installer was busy"),
    (7, "The installer stopped unexpectedly (exit code 7)")])
def test_a_failed_installation_is_reported(tmp_path: Path, code: int, error: str) -> None:
    state, adapters = adapters_with_state(tmp_path, {"imesh": "installed"})
    installer = Installer(tmp_path, [["progress", progress("forge", forge=("running", "Installing", 0.2),
                                                           imesh=("pending", "", None))], ["exit", code]])

    async def main() -> None:
        async with running(tmp_path, adapters, installer.bg) as pm:
            await until(lambda: installer.bg.snapshot()["finished"], what="the installer to end")
            snap = installer.bg.snapshot()
            assert error in snap["error"] and snap["running"] is False
            assert snap["components"]["forge"] == {"status": "failed", "step": "Installing", "fraction": 0.2,
                                                   "step_fraction": None, "bytes_done": None, "bytes_total": None,
                                                   "message": STOPPED_EARLY}
            forge = pm.status("forge")
            assert forge["state"] == "not_installed" and "its download failed" in forge["message"]
            assert pm.status("imesh")["state"] == "stopped"         # still installed: usable
            sentence = "A disk is full (drive C:). Free up some space, then start Spellcast3D again."
            state.write_text(json.dumps({"imesh": "installed", "forge": ["failed", sentence]}), encoding="utf-8")
            assert pm.status("forge")["message"] == f"Fake forge is not installed. {sentence}"
            assert pm.install_wait("forge") is None
            queue = JobQueue(pm, ProjectStore(tmp_path / "projects"), pm.settings, adapters, pm.client)
            queue.store.create_project("Demo")
            pid = queue.store.add_prop("Demo", "Box")["id"]
            with pytest.raises(ToolError, match="not installed"):
                await queue.submit("images", "Demo", [pid])

    asyncio.run(main())


def test_nothing_starts_before_the_plan_nor_while_another_installer_runs(tmp_path: Path) -> None:
    state, adapters = adapters_with_state(tmp_path, {"imesh": "installed"})
    installer = Installer(tmp_path, [["sleep", 0.6], ["progress", progress(None, imesh=("skipped", "", None),
                                                                           forge=("pending", "", None))],
                                     ["wait_for", str(tmp_path / "never")]])
    with open(tmp_path / ".install.lock", "a+b") as lock:           # an installer in another window
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)

        async def main() -> None:
            async with running(tmp_path, adapters, installer.bg) as pm:
                await asyncio.sleep(0.3)
                snap = installer.bg.snapshot()
                assert (snap["running"], snap["finished"]) == (False, False) and installer.bg.proc is None
                assert "another window" in snap["message"]
                assert pm.status("forge")["state"] == "installing"
                assert pm.status("forge")["message"] == "Fake forge is about to be installed."
                assert "another window" in pm.install_wait("imesh")
                pm.start("imesh")                                   # installed: it waits for the plan
                await asyncio.sleep(0.3)
                assert pm.status("imesh")["state"] == "starting" and pm.get("imesh").proc is None

                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)  # the other installer finished
                await until(lambda: installer.bg.proc is not None, what="the installer to start")
                assert installer.bg.snapshot()["running"] and not installer.bg.settled()
                assert pm.install_wait("imesh") == "Checking which parts of Spellcast3D still need installing…"
                await until(lambda: pm.status("imesh")["state"] == "ready", what="Instant Meshes to start")
                assert pm.status("forge")["message"] == "Fake forge is waiting to be installed."
                assert pm.install_wait("imesh") is None
                assert pm.install_wait("forge") == "Waiting for Fake forge to finish installing…"

        asyncio.run(main())


def test_gpu_jobs_wait_while_the_card_is_calibrated_for_pixal3d(tmp_path: Path) -> None:
    """While the background installer's Pixal3D step measures the graphics card (its step text says so), a job for
    a GPU tool waits (the measurement must be of a quiet card); a CPU tool's job and any job afterwards go ahead."""
    from spellcast.install import progress as install_progress

    state, adapters = adapters_with_state(tmp_path, {"imesh": "installed", "forge": "installed"})
    calibrating = progress("pixal3d", pixal3d=("running", f"{install_progress.CALIBRATING_STEP} (a few minutes)", 0.98))
    installer = Installer(tmp_path, [["progress", calibrating], ["sleep", 0.8],
                                     ["progress", progress("pixal3d", pixal3d=("running", "Checking", 0.99))],
                                     ["sleep", 0.8], ["done"]])

    async def main() -> None:
        async with running(tmp_path, adapters, installer.bg) as pm:
            await until(lambda: pm.calibrating(), what="the calibration step to show")
            assert "calibration" in (pm.install_wait("forge") or "")        # the GPU tool waits
            assert pm.install_wait("imesh") is None                          # the CPU tool does not
            await until(lambda: not pm.calibrating(), what="the calibration step to end")
            assert pm.install_wait("forge") is None

    asyncio.run(main())


def test_a_progress_file_of_an_earlier_run_is_ignored(tmp_path: Path) -> None:
    installer = Installer(tmp_path, [["exit", 0]])
    installer.progress_file.write_text(json.dumps(progress("forge", forge=("running", "Old", 0.5))), encoding="utf-8")

    async def main() -> None:
        installer.bg.start()
        await until(lambda: installer.bg.snapshot()["finished"], what="the installer to end")
        assert installer.bg.snapshot() == {"running": False, "finished": True, "error": None, "current": None,
                                           "components": {}, "message": None}
        assert not installer.progress_file.exists()
        assert installer.bg.component("forge") is None and installer.bg.settled()
        await installer.bg.close()

    asyncio.run(main())


def test_the_installer_dies_with_the_hub_and_can_run_again(tmp_path: Path) -> None:
    pidfile = tmp_path / "installer.pid"
    installer = Installer(tmp_path, [["pid", str(pidfile)], ["progress", progress("pixal3d", pixal3d=(
        "running", "Downloading the AI models", 0.1))], ["wait_for", str(tmp_path / "never")]])

    async def main() -> None:
        installer.bg.start()
        await until(lambda: pidfile.exists() and pidfile.read_text(), what="the installer")
        pid = int(pidfile.read_text())
        with pytest.raises(ToolError) as e:
            installer.bg.restart()
        assert e.value.status_code == 409
        await installer.bg.close()
        assert not pid_alive(pid)

        installer.set_steps([["exit", 0]])
        installer.bg.restart()                                      # "Try again" after a failure
        await until(lambda: installer.bg.snapshot()["finished"], what="the second run")
        assert installer.bg.snapshot()["error"] is None
        await installer.bg.close()

    asyncio.run(main())


def test_the_installer_command_and_its_python(monkeypatch: pytest.MonkeyPatch) -> None:
    assert installer_argv(Path("C:/py/python.exe")) == [
        str(Path("C:/py/python.exe")), str(paths.CODE / "install.py"), "--if-needed", "--skip", "hub",
        "--progress-file", str(PROGRESS_FILE)]
    assert PROGRESS_FILE == paths.INSTALL_PROGRESS == paths.VENVS / "install-progress.json"
    monkeypatch.setenv("SPELLCAST_BASE_PYTHON", '"D:\\dev\\python.exe"')
    assert installer_bg.base_python() == Path("D:/dev/python.exe")
    monkeypatch.delenv("SPELLCAST_BASE_PYTHON")
    monkeypatch.setattr(paths, "PORTABLE_PYTHON", Path("Z:/missing/python.exe"))
    assert installer_bg.base_python() == Path(sys._base_executable)


def test_tail_lines_of_a_big_log(tmp_path: Path) -> None:
    log = tmp_path / "install.log"
    log.write_text("".join(f"line {i:07d}\n" for i in range(60_000)), encoding="utf-8")     # 780 KB
    assert tail_lines(log, 3) == ["line 0059997", "line 0059998", "line 0059999"]
    assert len(tail_lines(log, 5000)) == 2000
    assert all(line.startswith("line ") and len(line) == 12 for line in tail_lines(log, 2000))
    assert tail_lines(log, 0) == [] and tail_lines(tmp_path / "missing.log", 5) == []


# ---- HTTP API --------------------------------------------------------------------------------------

@pytest.fixture
def app_with(tmp_path: Path) -> Iterator[Any]:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>Spellcast3D</h1>", encoding="utf-8")

    def build(installer: BackgroundInstall | None, adapters: dict[str, Any]) -> TestClient:
        app = create_app(adapters, settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                         static_dir=static, logs_dir=tmp_path / "logs", installer=installer,
                         gpu_probe=lambda: GpuInfo(0, "Fake RTX 3060", 8.6, 12288, 3000, "580.10"))
        return TestClient(app, base_url="http://127.0.0.1")

    yield build


def _poll(client: TestClient, key: Any, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get("/api/status").json()
        if key(status):
            return status
        time.sleep(0.05)
    raise AssertionError(f"/api/status never matched: {status}")


def test_api_install_status_log_and_retry(tmp_path: Path, app_with: Any) -> None:
    state, adapters = adapters_with_state(tmp_path, {"imesh": "installed"})
    go = tmp_path / "go"
    installer = Installer(tmp_path, [["progress", progress("forge", forge=("running", "Installing torch", 0.3),
                                                           imesh=("skipped", "", None))],
                                     ["wait_for", str(go)], ["state", str(state), {"imesh": "installed",
                                                                                   "forge": "installed"}],
                                     ["exit", 0]])
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "install.log").write_text("first\nsecond\nthird\n", encoding="utf-8")
    with app_with(installer.bg, adapters) as client:
        status = _poll(client, lambda s: s["install"]["components"])
        assert status["install"]["running"] and status["install"]["current"] == "forge"
        assert status["tools"]["forge"]["state"] == "installing"
        assert "30%" in status["tools"]["forge"]["message"]
        started = client.post("/api/tools/forge/start")
        assert started.status_code == 409 and "being installed" in started.json()["detail"]
        assert client.post("/api/install/start").status_code == 409
        assert client.get("/api/install/log?tail=2").json() == {"lines": ["second", "third"]}

        go.touch()
        status = _poll(client, lambda s: s["install"]["finished"])
        assert status["install"]["error"] is None and status["tools"]["forge"]["state"] == "stopped"
        again = client.post("/api/install/start").json()           # runs the installer once more
        assert again["finished"] is False
        _poll(client, lambda s: s["install"]["finished"])

    _, plain = adapters_with_state(tmp_path, {"forge": "installed", "imesh": "installed"})
    with app_with(None, plain) as client:
        assert client.get("/api/status").json()["install"] == IDLE
        refused = client.post("/api/install/start")
        assert refused.status_code == 409 and "run-spellcast3d.bat" in refused.json()["detail"]


def test_the_console_says_how_to_open_the_page_when_the_installer_is_through() -> None:
    from spellcast.installer_bg import done_banner

    lines = done_banner("http://127.0.0.1:7770", True)
    assert any("Spellcast3D is ready to use." in line for line in lines)
    assert any("enter 127.0.0.1:7770 into its address bar" in line for line in lines)
    assert not any("already open" in line for line in lines)            # no aside after the one thing to do
    lines = done_banner("http://127.0.0.1:7770", False)
    assert any("needs another try" in line for line in lines)


def test_the_ready_banner_waits_for_a_quiet_console(tmp_path: Path) -> None:
    """installer_bg._say_ready: the banner comes after READY_QUIET seconds without a console line, so that it is
    the last thing in the window (the tools a waiting job starts print right after the installer)."""
    from spellcast import installer_bg
    from spellcast.installer_bg import BackgroundInstall, console_echo, console_touched

    lines: list[str] = []
    bg = BackgroundInstall(["x"], cwd=tmp_path, progress_file=tmp_path / "p.json", lock_file=tmp_path / "l",
                           echo=lines.append, page_url="http://127.0.0.1:7770", ready_quiet=0.3)

    async def main() -> None:
        console_touched()
        task = asyncio.create_task(bg._say_ready(True))
        await asyncio.sleep(0.15)
        console_touched()                       # a tool said something: the clock starts again
        await asyncio.sleep(0.2)
        assert lines == [], "not yet: the console was busy 0.2 s ago"
        await asyncio.sleep(0.25)
        await asyncio.wait_for(task, 2)
        assert any("ready to use" in line for line in lines)
        console_echo("")                                        # the real echo counts as console activity
        assert installer_bg.console_quiet_for() < 0.1

    asyncio.run(main())
