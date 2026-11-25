"""The full Qwen-Image 2.1 as an on-demand download (spellcast/model_downloads.py, launchers/qwen_download.py): whether it
is on disk, its download in the background (one at a time, waiting for an installer that works on Qwen-Image, its
room on the drive, what /api/status "qwen_full" says while it runs and after it stopped), the download script itself
(the installer's own downloader, retried, linked from a copy on this PC), and POST /api/qwen/full/download (it opens
models\\qwen-image in File Explorer).

Run with the hub venv:  <venv>\\Scripts\\python.exe -m pytest code\\tests -q
Nothing here downloads anything: the download runs a stand-in script (or the real one with a stand-in downloader), and
the models folder is a temp one. The pinned sizes are made small (the `small_pin` fixture).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import msvcrt
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from spellcast import model_downloads, paths
from spellcast.install import models
from spellcast.procman import ToolError
from spellcast.projects import ProjectStore
from spellcast.server import create_app
from test_core import FakeAdapter, make_settings
from test_qwen_hub import RTX_3060, FakeQwen, _static

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Spellcast3D is Windows-only")

sys.path.insert(0, str(paths.LAUNCHERS))
import qwen_download  # noqa: E402  (launchers/: runs in the hub venv as a child, stdlib only)

QUALITY = "Q4_K_M"
NAME = "qwen-image-2.1-Q4_K_M.gguf"
CONTENT = b"the full model, small" * 10                      # 210 bytes: the stand-in's pinned file
SHA = hashlib.sha256(CONTENT).hexdigest()
REAL_HF_CACHES = models.user_hf_caches                       # before small_pin replaces it


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


@pytest.fixture(autouse=True)
def small_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """The full model's Q4_K_M pin is CONTENT (the real one is 4.2 GB), the drive has room, and no copy on this PC is
    looked for anywhere but where a test says (what is checked is remembered in conftest's records)."""
    monkeypatch.setitem(models.QWEN_DENOISERS, QUALITY, (len(CONTENT), SHA))
    monkeypatch.setattr(model_downloads, "_free_space", lambda folder: 10 ** 14)
    monkeypatch.setattr(models, "user_hf_caches", lambda: [])
    monkeypatch.setattr(models, "EXTRA_SOURCES", [])
    monkeypatch.delenv("SPELLCAST_MODELS_FROM", raising=False)


FAKE_DOWNLOAD = r'''
import json, os, sys, time
from pathlib import Path
plan = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))
argv = sys.argv[1:]
Path(plan["record"]).write_text(json.dumps({"argv": argv, "cwd": os.getcwd()}), encoding="utf-8")
folder = Path(argv[argv.index("--models") + 1])
name = "qwen-image-2.1-" + argv[argv.index("--quality") + 1] + ".gguf"
part = folder / (name + ".part")
part.write_bytes(plan.get("part", b"".decode()).encode())
print("fake download started", flush=True)
while plan.get("wait_for") and not Path(plan["wait_for"]).exists():
    time.sleep(0.02)
if plan.get("complete"):
    part.write_bytes(plan["content"].encode())
    os.replace(part, folder / name)
print("fake download done", flush=True)
sys.exit(plan.get("exit", 0))
'''


class Download:
    """A model_downloads.BackgroundDownload of the full model running the stand-in script, into a temp models
    folder."""

    def __init__(self, tmp_path: Path, **plan: Any):
        self.folder = tmp_path / "models" / "qwen-image"
        self.script = tmp_path / "fake_qwen_download.py"
        self.script.write_text(FAKE_DOWNLOAD, encoding="utf-8")
        self.record = tmp_path / "record.json"
        self.plan(**plan)
        self.lock = tmp_path / ".install.lock"
        self.log = tmp_path / "logs" / model_downloads.MODELS["full"].log_name
        self.bg = model_downloads.BackgroundDownload("full", QUALITY, self.folder, python=Path(sys.executable),
                                                     script=self.script, install_lock=self.lock, log_file=self.log,
                                                     wait_poll=0.05)

    def plan(self, **plan: Any) -> None:
        self.script.with_suffix(".json").write_text(json.dumps({"record": str(self.record),
                                                                "content": CONTENT.decode(), **plan}),
                                                    encoding="utf-8")

    def recorded(self) -> dict[str, Any]:
        return json.loads(self.record.read_text(encoding="utf-8"))


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


async def _until(predicate: Callable[[], bool], timeout: float = 20.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


# ---- on disk ---------------------------------------------------------------------------------------------

def test_the_full_model_is_there_only_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "qwen-image"
    md = model_downloads
    assert md.file_names("full", QUALITY, folder) == [NAME] and md.total_bytes("full", QUALITY, folder) == len(CONTENT)
    assert not md.available("full", QUALITY, folder) and md.downloaded_bytes("full", QUALITY, folder) == 0
    folder.mkdir()
    (folder / (NAME + ".part")).write_bytes(CONTENT[:100])            # an unfinished download: its bytes count
    assert not md.available("full", QUALITY, folder) and md.downloaded_bytes("full", QUALITY, folder) == 100
    (folder / NAME).write_bytes(CONTENT[:200])                        # not its size: not there
    assert not md.available("full", QUALITY, folder)
    (folder / NAME).write_bytes(CONTENT)
    assert md.available("full", QUALITY, folder) and md.downloaded_bytes("full", QUALITY, folder) == len(CONTENT)
    assert md.MODELS["full"].folder == paths.MODELS_QWEN
    # the Turbo's file is another one: it says nothing about the full model
    assert not models.qwen_model_on_disk("turbo", QUALITY, folder)
    # a bigger size the artist put there stands in for the configured one (models.size_on_disk): nothing to download,
    # and the launcher loads that file
    (folder / NAME).unlink()
    bigger, big = "qwen-image-2.1-Q6_K.gguf", b"the full model, bigger" * 12
    monkeypatch.setitem(models.QWEN_DENOISERS, "Q6_K", (len(big), hashlib.sha256(big).hexdigest()))
    assert not md.available("full", QUALITY, folder)
    (folder / bigger).write_bytes(big)
    assert md.available("full", QUALITY, folder) and md.file_names("full", QUALITY, folder) == [bigger]
    assert md.total_bytes("full", QUALITY, folder) == len(big)
    assert models.qwen_size_on_disk("full", QUALITY, folder) == "Q6_K"
    assert models.qwen_model_files(QUALITY, folder)["full"] == folder / bigger


def test_the_exit_codes_are_the_scripts() -> None:
    md, script = model_downloads, qwen_download
    assert (md.EXIT_OK, md.EXIT_FAILED, md.EXIT_CHECKSUM, md.EXIT_DISK_FULL, md.EXIT_DENIED) == (
        script.EXIT_OK, script.EXIT_FAILED, script.EXIT_CHECKSUM, script.EXIT_DISK_FULL, script.EXIT_DENIED)
    assert 2 not in (script.EXIT_FAILED, script.EXIT_CHECKSUM, script.EXIT_DISK_FULL, script.EXIT_DENIED)  # argparse's


def test_the_hubs_children_get_the_artists_hf_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """paths.clean_env scrubs HF_HOME and HF_HUB_CACHE from every child; the download script (and the background
    installer) get the caches they name through SPELLCAST_HF_CACHES instead, so a copy there is linked."""
    monkeypatch.setattr(models, "user_hf_caches", REAL_HF_CACHES)
    cache = tmp_path / "hf" / "hub"
    cache.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv(models.HF_CACHES_ENV, raising=False)
    assert REAL_HF_CACHES()[0] == cache
    env = paths.clean_env(None, models.hf_caches_env())
    assert "HF_HOME" not in env and env[models.HF_CACHES_ENV].split(";")[0] == str(cache)
    monkeypatch.delenv("HF_HOME")
    assert cache not in REAL_HF_CACHES()
    monkeypatch.setenv(models.HF_CACHES_ENV, env[models.HF_CACHES_ENV])
    assert REAL_HF_CACHES()[0] == cache                                 # the child finds it through the variable


# ---- the download in the background -------------------------------------------------------------------------

def test_the_download_runs_once_and_ends_with_the_file(tmp_path: Path) -> None:
    dl = Download(tmp_path, complete=True, part="half", wait_for=str(tmp_path / "go"))

    async def main() -> None:
        before = dl.bg.snapshot()
        assert before == {"available": False, "downloading": False, "waiting": False, "done_bytes": 0,
                          "total_bytes": len(CONTENT), "error": None}
        dl.bg.start()
        dl.bg.start()                                           # nothing more while it runs
        await _until(lambda: dl.record.exists() and _size(dl.folder / (NAME + ".part")) == 4,
                     what="the download to start")
        proc = dl.bg.proc
        state = dl.bg.snapshot()
        assert state["downloading"] is True and state["done_bytes"] == 4      # the .part's bytes so far
        assert state["waiting"] is False
        assert dl.recorded()["argv"] == ["--model", "full", "--models", str(dl.folder), "--quality", QUALITY]
        assert Path(dl.recorded()["cwd"]) == paths.CODE
        (tmp_path / "go").touch()
        await _until(lambda: not dl.bg.downloading(), what="the download to end")
        assert dl.bg.proc is proc
        assert dl.bg.snapshot() == {"available": True, "downloading": False, "waiting": False,
                                    "done_bytes": len(CONTENT), "total_bytes": len(CONTENT), "error": None}
        await _until(lambda: "fake download done" in _text(dl.log), what="its output in its log")
        dl.record.unlink()
        dl.bg.start()                                           # the file is there: nothing to do
        assert not dl.bg.downloading() and not dl.record.exists()
        await dl.bg.close()

    asyncio.run(main())
    assert (dl.folder / NAME).read_bytes() == CONTENT


@pytest.mark.parametrize("code, message", [
    (1, "most often because the internet connection dropped"), (3, "not the expected file"),
    (4, "ran out of space"), (5, "access denied"), (9, "stopped unexpectedly (exit code 9)"),
    (0, "its file is not there")])
def test_a_failed_download_says_why_and_continues_when_started_again(tmp_path: Path, code: int, message: str) -> None:
    dl = Download(tmp_path, exit=code, part="some")

    async def main() -> None:
        dl.bg.start()
        await _until(lambda: not dl.bg.downloading(), what="the download to end")
        state = dl.bg.snapshot()
        assert state["downloading"] is False and state["available"] is False and message in state["error"]
        assert state["done_bytes"] == 4                         # what arrived is kept (.part)
        dl.plan(complete=True)
        dl.bg.start()                                           # again: it continues
        assert dl.bg.snapshot()["error"] is None and dl.bg.snapshot()["downloading"] is True
        await _until(lambda: not dl.bg.downloading(), what="the download to end")
        assert dl.bg.snapshot()["available"] is True and dl.bg.snapshot()["error"] is None
        await dl.bg.close()

    asyncio.run(main())


def test_the_download_waits_for_an_installer_that_works_on_qwen(tmp_path: Path) -> None:
    dl = Download(tmp_path, complete=True)
    with open(dl.lock, "a+b") as lock:                             # an installer in another window
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)

        async def main() -> None:
            dl.bg.start()
            await asyncio.sleep(0.4)
            state = dl.bg.snapshot()
            assert dl.bg.proc is None and state["downloading"] is True and not dl.record.exists()
            assert state["waiting"] is True                          # the page: "Waiting for the installer…"
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            await _until(lambda: not dl.bg.downloading(), what="the download to end")
            assert dl.bg.snapshot()["available"] is True and dl.bg.snapshot()["waiting"] is False

        asyncio.run(main())

    class Installer:                                                # the hub's own, installing Qwen-Image
        def __init__(self) -> None:
            self.proc = type("P", (), {"poll": lambda self: None})()
            self.working = True

        def component(self, component: str) -> dict[str, Any] | None:
            return {"status": "running"} if self.working and component == paths.QWEN else None

        def settled(self) -> bool:
            return True

    installer = Installer()
    dl.bg.install = installer                                   # type: ignore[assignment]
    assert dl.bg._installer_busy()
    installer.working = False                                   # it works on another tool: no reason to wait
    assert not dl.bg._installer_busy()


def test_the_download_needs_room(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dl = Download(tmp_path, complete=True)
    dl.folder.mkdir(parents=True)
    (dl.folder / (NAME + ".part")).write_bytes(CONTENT[:10])
    monkeypatch.setattr(model_downloads, "_free_space", lambda folder: 500_000_000)

    async def main() -> None:
        with pytest.raises(ToolError, match="not enough free space on drive") as e:
            dl.bg.start()
        assert e.value.status_code == 409 and "1.0 GB" in str(e.value) and "0.5 GB" in str(e.value)
        assert not dl.bg.downloading()

    asyncio.run(main())


# ---- the download script (launchers/qwen_download.py) --------------------------------------------------------

def test_the_script_downloads_with_the_installers_downloader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "qwen-image"
    calls: list[tuple[str, Path, int, str]] = []

    def download(url: str, dst: Path, size: int, sha: str, log: Any) -> None:
        calls.append((url, dst, size, sha))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(CONTENT)

    monkeypatch.setattr(models, "download_file", download)
    assert qwen_download.main(["--models", str(folder), "--quality", QUALITY]) == qwen_download.EXIT_OK
    assert calls == [(models.hf_url(*models.QWEN_SOURCES[NAME]), folder / NAME, len(CONTENT), SHA)]
    assert calls[0][0] == ("https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/resolve/"
                           "2c31ccd392b367a6637841a143813320a02dff55/qwen-image-2.1-Q4_K_M.gguf")
    assert qwen_download.main(["--models", str(folder), "--quality", QUALITY]) == qwen_download.EXIT_OK
    assert len(calls) == 1                                          # there already: nothing downloaded
    with pytest.raises(SystemExit):
        qwen_download.main(["--models", str(folder), "--quality", "Q9"])


def test_the_script_links_a_copy_on_this_pc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    other = tmp_path / "other" / "models" / "qwen-image"            # another Spellcast3D's models folder
    other.mkdir(parents=True)
    (other / NAME).write_bytes(CONTENT)
    monkeypatch.setattr(models, "EXTRA_SOURCES", [other.parent])
    monkeypatch.setattr(models, "download_file", lambda *a: pytest.fail("downloaded"))
    folder = tmp_path / "qwen-image"
    assert qwen_download.main(["--models", str(folder), "--quality", QUALITY]) == qwen_download.EXIT_OK
    assert os.path.samefile(folder / NAME, other / NAME)


def test_the_script_retries_until_it_stops_getting_anywhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "qwen-image"
    part = folder / (NAME + ".part")
    outcomes: list[str] = ["progress", "progress", "nothing", "done"]

    def download(url: str, dst: Path, size: int, sha: str, log: Any) -> None:
        outcome = outcomes.pop(0)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if outcome == "progress":
            with open(part, "ab") as f:
                f.write(b"x" * 10)
            raise OSError(f"{dst.name}: the connection dropped")
        if outcome == "nothing":
            raise OSError(f"{dst.name}: timed out")
        if outcome == "checksum":
            raise models.ChecksumError(f"{dst.name}: the downloaded file is not the expected one")
        if outcome == "full":        # download_file's own OSError, from the write that failed
            raise OSError(f"{dst.name}: the download stopped (10 of 210 bytes)") from OSError(28, "No space left")
        if outcome == "denied":
            raise OSError(f"{dst.name}: Access is denied (0 of 210 bytes)") from PermissionError(13, "Access is denied")
        part.unlink(missing_ok=True)
        dst.write_bytes(CONTENT)

    monkeypatch.setattr(models, "download_file", download)
    monkeypatch.setattr(qwen_download, "RETRY_DELAY", 0.0)
    argv = ["--models", str(folder), "--quality", QUALITY]
    assert qwen_download.main(argv) == qwen_download.EXIT_OK and not outcomes
    (folder / NAME).unlink()
    outcomes[:] = ["nothing"] * qwen_download.MAX_STALLED_ATTEMPTS
    assert qwen_download.main(argv) == qwen_download.EXIT_FAILED and not outcomes      # never any progress
    outcomes[:] = ["checksum"]
    assert qwen_download.main(argv) == qwen_download.EXIT_CHECKSUM
    outcomes[:] = ["full"]
    assert qwen_download.main(argv) == qwen_download.EXIT_DISK_FULL
    outcomes[:] = ["denied"]                                            # at once: no 10 tries 30 s apart
    assert qwen_download.main(argv) == qwen_download.EXIT_DENIED and not outcomes
    assert qwen_download.disk_full(OSError(28, "No space left on device"))
    assert not qwen_download.disk_full(OSError("timed out")) and not qwen_download.access_denied(OSError("timed out"))
    assert qwen_download.access_denied(PermissionError(13, "Access is denied"))


# ---- HTTP API -------------------------------------------------------------------------------------------

@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, Download, list[Any]]]:
    dl = Download(tmp_path, complete=True)
    qwen = FakeQwen(tmp_path)
    qwen.models_dir = dl.folder
    shown: list[tuple[Path, bool]] = []
    app = create_app({paths.QWEN: qwen, "imesh": FakeAdapter("imesh", "retopo", False, tmp_path)},
                     settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=_static(tmp_path), logs_dir=tmp_path / "logs", gpu_probe=lambda: RTX_3060,
                     qwen_full_download=dl.bg,
                     show_in_explorer=lambda path, select: shown.append((path, select)) or True)
    monkeypatch.delenv("SPELLCAST_NO_EXPLORER", raising=False)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
        yield client, dl, shown


def test_api_status_and_the_download_that_opens_the_folder(api: Any) -> None:
    client, dl, shown = api
    assert client.get("/api/status").json()["qwen_full"] == {
        "available": False, "downloading": False, "waiting": False, "done_bytes": 0, "total_bytes": len(CONTENT),
        "error": None}
    started = client.post("/api/qwen/full/download")
    assert started.status_code == 200 and started.json()["total_bytes"] == len(CONTENT)
    assert started.json()["opened"] is True and shown == [(dl.folder, False)]      # the folder it arrives in
    deadline = time.monotonic() + 20
    while not client.get("/api/status").json()["qwen_full"]["available"]:
        assert time.monotonic() < deadline, client.get("/api/status").json()["qwen_full"]
        time.sleep(0.05)
    again = client.post("/api/qwen/full/download").json()            # nothing left to do; the folder opens again
    assert again["downloading"] is False and again["available"] is True and len(shown) == 2


def test_api_download_needs_qwen_as_the_image_generator(tmp_path: Path) -> None:
    dl = Download(tmp_path, complete=True)
    forge = {"forge": FakeAdapter("forge", "images", True, tmp_path)}
    app = create_app(forge, settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=_static(tmp_path), logs_dir=tmp_path / "logs", gpu_probe=lambda: None,
                     qwen_full_download=dl.bg)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        refused = client.post("/api/qwen/full/download")
        assert refused.status_code == 409 and "not the image generator" in refused.json()["detail"]
        assert not dl.bg.downloading() and not dl.record.exists()
    qwen = FakeQwen(tmp_path)
    qwen.install_status = lambda: {"status": "not_installed", "reason": ""}     # type: ignore[method-assign]
    app = create_app({paths.QWEN: qwen}, settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=_static(tmp_path), logs_dir=tmp_path / "logs", gpu_probe=lambda: None,
                     qwen_full_download=dl.bg)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        refused = client.post("/api/qwen/full/download")
        assert refused.status_code == 409 and "not installed" in refused.json()["detail"]
        assert not dl.bg.downloading()


def test_the_hubs_own_download_is_of_its_qwens_size_and_folder(tmp_path: Path) -> None:
    """create_app's default download: the size and the models folder of the registered Qwen-Image (what its
    launcher reads), the config file's otherwise."""
    qwen = FakeQwen(tmp_path)
    qwen.quality = "Q6_K"
    app = create_app({paths.QWEN: qwen}, settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=_static(tmp_path), logs_dir=tmp_path / "logs", gpu_probe=lambda: None)
    download = app.state.hub.qwen_full_download
    assert (download.quality, download.folder) == ("Q6_K", qwen.models_dir)
    assert download.log_file == tmp_path / "logs" / "qwen-full-download.log"
