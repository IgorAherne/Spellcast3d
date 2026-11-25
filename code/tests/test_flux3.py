"""FLUX.3 Image Klein, the Images stage's optional third model: its pinned files (install/models.py), its download
(spellcast/model_downloads.py, launchers/qwen_download.py --model flux3: copies on this PC linked first, also one in
models\\qwen-image by name), /api/status "flux3" and POST /api/flux3/download, the engine-support gate
(install/sdcpp.py ENGINE_HAS_FLUX3), the adapter's request and refusals, a job through the hub, and the launcher's
part (launchers/qwen_server.py: its files, its sampling, the switch between any two models, its requests refused
without an engine that draws it).

Run with the hub venv:  <venv>\\Scripts\\python.exe -m pytest code\\tests -q
Nothing here downloads anything or touches the real models folders: the pins are made small (`small_flux3`), the
download runs a stand-in script (or the real one with a stand-in downloader), the folders are temp ones.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from spellcast import config, model_downloads, paths
from spellcast.adapters import qwen
from spellcast.adapters.base import StageError
from spellcast.install import models
from spellcast.install import sdcpp as engine_pin
from spellcast.projects import ProjectError, ProjectStore
from spellcast.server import create_app
from test_core import FakeAdapter, make_settings
from test_qwen_hub import (RTX_3060, FakeQwen, _color_type, _images, _launcher, _prop, _run, _static,  # noqa: F401
                           root_config)

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Spellcast3D is Windows-only")

if str(paths.LAUNCHERS) not in sys.path:     # the launchers import their siblings as top-level modules
    sys.path.insert(0, str(paths.LAUNCHERS))
import qwen_download  # noqa: E402
import qwen_sdcpp as sdcpp  # noqa: E402
import qwen_server as server  # noqa: E402
from test_qwen_launcher import (GiB, FakePicture, _ContextLib, _fake_engine, _FakeReference,  # noqa: E402,F401
                                _RecordingLib, _request, fake_pictures, root)

QUALITY = "Q4_K_M"
DENOISER, TEXT_ENCODER, VAE = "flux-3-image-klein-UD-Q4_K_XL.gguf", models.FLUX3_TEXT_ENCODER, models.FLUX3_VAE
CONTENT = {DENOISER: b"flux3 denoiser, small" * 4, TEXT_ENCODER: b"qwen3-vl-4b, small" * 3, VAE: b"flux2 vae" * 2}
PINS = {name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in CONTENT.items()}
WHOLE = sum(size for size, _sha in PINS.values())
REAL_GATE = engine_pin.ENGINE_HAS_FLUX3          # as install/sdcpp.py has it (the tests set it as they need)


@pytest.fixture(autouse=True)
def small_flux3(monkeypatch: pytest.MonkeyPatch) -> None:
    """FLUX.3's Q4_K_M files are CONTENT's (the real ones are 6.8 GB), the drive has room, the gate is closed (as it
    was before the pinned engine of today drew FLUX.3) unless a test opens it, and no copy on this PC is looked for
    anywhere but where a test says."""
    monkeypatch.setitem(models.FLUX3_DENOISERS, QUALITY, (DENOISER, PINS[DENOISER]))
    monkeypatch.setitem(models.FLUX3_SHARED_FILES, TEXT_ENCODER, PINS[TEXT_ENCODER])
    monkeypatch.setitem(models.FLUX3_SHARED_FILES, VAE, PINS[VAE])
    monkeypatch.setattr(model_downloads, "_free_space", lambda folder: 10 ** 14)
    monkeypatch.setattr(engine_pin, "ENGINE_HAS_FLUX3", False)
    monkeypatch.setattr(models, "user_hf_caches", lambda: [])
    monkeypatch.setattr(models, "EXTRA_SOURCES", [])
    monkeypatch.delenv("SPELLCAST_MODELS_FROM", raising=False)


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def _put(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(CONTENT[name])


async def _until(predicate: Callable[[], bool], timeout: float = 20.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


# ---- the pins ------------------------------------------------------------------------------------------------

def test_the_pins_follow_the_hubs_sizes_and_name_their_sources(tmp_path: Path) -> None:
    """One recommended file of unsloth's per size of the hub (config.QUALITIES), the text encoder and the VAE for every
    size, each from its pinned commit; the VAE has another path in its repo than its name here."""
    assert tuple(models.FLUX3_DENOISERS) == config.QUALITIES
    assert [name for name, _pin in models.FLUX3_DENOISERS.values()] == [
        "flux-3-image-klein-UD-Q3_K_XL.gguf", DENOISER, "flux-3-image-klein-UD-Q5_K_XL.gguf",
        "flux-3-image-klein-Q6_K.gguf", "flux-3-image-klein-Q8_0.gguf"]
    assert models.flux3_manifest(QUALITY, tmp_path) == PINS == models.optional_manifest("flux3", QUALITY, tmp_path)
    assert models.optional_manifest("full", QUALITY, tmp_path) == models.qwen_full_manifest(QUALITY, tmp_path)
    assert models.OPTIONAL_MODELS == ("full", "flux3", "hi3d") and model_downloads.MODELS["flux3"].folder == paths.MODELS_FLUX3
    assert paths.MODELS_FLUX3 == paths.MODELS / "flux3-klein"
    assert models.FLUX3_SOURCES[VAE] == ("Comfy-Org/Flux-2-Klein-4B", "533c4cbc3d3b89af1743d9466c434e89810391c0",
                                         "split_files/vae/flux2-vae.safetensors")
    assert models.FLUX3_SOURCES[TEXT_ENCODER][0] == "Qwen/Qwen3-VL-4B-Instruct-GGUF"
    assert models.FLUX3_SOURCES[DENOISER] == (models.FLUX3_GGUF_REPO, models.FLUX3_GGUF_REVISION, DENOISER)
    assert set(models.FLUX3_PINS) == set(models.FLUX3_SOURCES)            # every pinned file has its source
    # the real pins (FLUX3_PINS is made at import, before this test's small ones)
    assert models.FLUX3_PINS["flux-3-image-klein-UD-Q4_K_XL.gguf"] == (
        3_929_944_992, "0a057e07550a5b2169e344d686171a7dd3c7d82c7d8d1d1605d8d881c16e9fa7")
    assert models.FLUX3_PINS[VAE][0] == 336_211_292 and models.FLUX3_PINS[TEXT_ENCODER][0] == 2_497_281_664
    named = models.flux3_model_files(QUALITY, Path("F"))
    assert named == {"diffusion": Path("F") / DENOISER, "vae": Path("F") / VAE, "text_encoder": Path("F") / TEXT_ENCODER}
    # the installer never fetches them: Qwen-Image's own manifest has none of them
    assert not set(models.qwen_manifest(QUALITY, tmp_path)) & set(models.FLUX3_PINS)


def test_on_disk_only_with_all_three_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "flux3-klein"
    assert not models.flux3_model_on_disk(QUALITY, folder) and not model_downloads.available("flux3", QUALITY, folder)
    _put(folder, DENOISER, VAE)
    (folder / (TEXT_ENCODER + ".part")).write_bytes(CONTENT[TEXT_ENCODER][:10])
    assert not models.flux3_model_on_disk(QUALITY, folder)
    assert model_downloads.missing("flux3", QUALITY, folder) == [TEXT_ENCODER]
    assert model_downloads.downloaded_bytes("flux3", QUALITY, folder) == PINS[DENOISER][0] + PINS[VAE][0] + 10
    _put(folder, TEXT_ENCODER)
    assert models.flux3_model_on_disk(QUALITY, folder) and model_downloads.total_bytes("flux3", QUALITY, folder) == WHOLE
    # a bigger denoiser the artist put there stands in for the configured one (models.size_on_disk): its download is
    # then only what is missing besides, and the launcher loads that file
    (folder / DENOISER).unlink()
    bigger, big = models.FLUX3_DENOISERS["Q6_K"][0], b"flux3, bigger" * 5
    monkeypatch.setitem(models.FLUX3_DENOISERS, "Q6_K", (bigger, (len(big), hashlib.sha256(big).hexdigest())))
    assert not models.flux3_model_on_disk(QUALITY, folder)
    (folder / bigger).write_bytes(big)
    assert models.flux3_size_on_disk(QUALITY, folder) == "Q6_K" and models.flux3_model_on_disk(QUALITY, folder)
    assert set(models.flux3_manifest(QUALITY, folder)) == {bigger, TEXT_ENCODER, VAE}
    assert models.flux3_model_files(QUALITY, folder)["diffusion"] == folder / bigger
    (folder / VAE).unlink()
    assert model_downloads.missing("flux3", QUALITY, folder) == [VAE] and not models.flux3_model_on_disk(QUALITY, folder)


def test_the_one_engine_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """install/sdcpp.py ENGINE_HAS_FLUX3 is the one switch: the hub's engine_ready and the launcher's auto follow it,
    and it is on with the pinned engine of today (spellcast-perf-25d4127; 52bc254 was the first built with FLUX.3,
    458dccb had none). The same file's ENGINE_KEEPS_PARAMS is on with it too (the first engine that reads the variable)."""
    assert REAL_GATE is True and engine_pin.ENGINE.tag == "spellcast-perf-25d4127"      # the file's own value
    assert engine_pin.ENGINE_KEEPS_PARAMS is True
    monkeypatch.setattr(engine_pin, "ENGINE_HAS_FLUX3", False)
    assert not engine_pin.engine_draws("flux3") and engine_pin.engine_draws("full") and engine_pin.engine_draws("turbo")
    assert server.flux3_engine_choice("auto", pinned=True)[0] is False
    monkeypatch.setattr(engine_pin, "ENGINE_HAS_FLUX3", True)
    assert engine_pin.engine_draws("flux3")
    on, said = server.flux3_engine_choice("auto", pinned=True)
    assert on and said == f"FLUX.3 Image Klein: the pinned engine {sdcpp.PINNED_TAG} draws it."
    assert server.flux3_engine_choice("auto", pinned=False)[0] is False, "another build: only with --flux3-engine on"
    assert server.flux3_engine_choice("off", pinned=True)[0] is False
    monkeypatch.setattr(engine_pin, "ENGINE_HAS_FLUX3", False)
    assert server.flux3_engine_choice("on", pinned=False)[0] is True, "the developers' override (a build under test)"


# ---- the download --------------------------------------------------------------------------------------------

FAKE_DOWNLOAD = r'''
import json, os, sys, time
from pathlib import Path
plan = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))
argv = sys.argv[1:]
Path(plan["record"]).write_text(json.dumps({"argv": argv}), encoding="utf-8")
folder = Path(argv[argv.index("--models") + 1])
for name, data in plan["files"]:
    if plan.get("wait_for") and name == plan.get("wait_before"):
        while not Path(plan["wait_for"]).exists():
            time.sleep(0.02)
    (folder / name).write_text(data, encoding="utf-8")
print("fake flux3 download done", flush=True)
sys.exit(plan.get("exit", 0))
'''


class Download:
    """model_downloads.BackgroundDownload of FLUX.3 running the stand-in script, into a temp models folder."""

    def __init__(self, tmp_path: Path, **plan: Any):
        self.folder = tmp_path / "models" / "flux3-klein"
        self.script = tmp_path / "fake_flux3_download.py"
        self.script.write_text(FAKE_DOWNLOAD, encoding="utf-8")
        self.record = tmp_path / "record.json"
        self.lock = tmp_path / ".install.lock"
        self.log = tmp_path / "logs" / "flux3-download.log"
        files = [[name, CONTENT[name].decode()] for name in (TEXT_ENCODER, DENOISER, VAE)]
        self.script.with_suffix(".json").write_text(json.dumps({"record": str(self.record), "files": files, **plan}),
                                                    encoding="utf-8")
        self.bg = model_downloads.BackgroundDownload("flux3", QUALITY, self.folder, python=Path(sys.executable),
                                                     script=self.script, install_lock=self.lock, log_file=self.log,
                                                     wait_poll=0.05)


def test_the_download_counts_what_is_missing_and_never_waits_for_an_installer(tmp_path: Path) -> None:
    """total_bytes: the files missing when it started (the VAE was there already), done_bytes what is on disk of
    them; once all are there, the whole model. It runs while another installer holds the lock (FLUX.3's folder is no
    installer's), as `--model flux3`, into its own folder, its output in flux3-download.log."""
    import msvcrt

    dl = Download(tmp_path, wait_for=str(tmp_path / "go"), wait_before=DENOISER)
    _put(dl.folder, VAE)
    (dl.folder / (TEXT_ENCODER + ".part")).write_bytes(CONTENT[TEXT_ENCODER][:7])
    left = PINS[DENOISER][0] + PINS[TEXT_ENCODER][0]
    assert dl.bg.spec.status == "flux3" and dl.bg.spec.waits_for is None
    with open(dl.lock, "a+b") as lock:                                  # an installer in another window
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)

        async def main() -> None:
            assert dl.bg.snapshot() == {"available": False, "downloading": False, "waiting": False, "done_bytes": 7,
                                        "total_bytes": left, "error": None}
            dl.bg.start()
            await _until(lambda: _size(dl.folder / TEXT_ENCODER) == PINS[TEXT_ENCODER][0], what="the text encoder")
            state = dl.bg.snapshot()                    # one of the two is whole now: the total stays the same
            assert state["downloading"] and state["total_bytes"] == left and state["done_bytes"] >= PINS[TEXT_ENCODER][0]
            assert json.loads(dl.record.read_text(encoding="utf-8"))["argv"] == [
                "--model", "flux3", "--models", str(dl.folder), "--quality", QUALITY]
            (tmp_path / "go").touch()
            await _until(lambda: not dl.bg.downloading(), what="the download to end")
            assert dl.bg.snapshot() == {"available": True, "downloading": False, "waiting": False,
                                        "done_bytes": WHOLE, "total_bytes": WHOLE, "error": None}
            await _until(lambda: "fake flux3 download done" in dl.log.read_text(encoding="utf-8")
                         if dl.log.exists() else False, what="its log")
            await dl.bg.close()

        asyncio.run(main())
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def test_a_failed_download_names_flux3(tmp_path: Path) -> None:
    dl = Download(tmp_path, exit=1, files=[])

    async def main() -> None:
        dl.bg.start()
        await _until(lambda: not dl.bg.downloading(), what="the download to end")
        error = dl.bg.snapshot()["error"]
        assert error.startswith("The download of FLUX.3 Image Klein stopped, most often because the internet")

    asyncio.run(main())
    (tmp_path / "again").mkdir()
    ok_but_gone = Download(tmp_path / "again", exit=0, files=[])

    async def again() -> None:
        ok_but_gone.bg.start()
        await _until(lambda: not ok_but_gone.bg.downloading(), what="the download to end")
        assert "not all of its files are there" in ok_but_gone.bg.snapshot()["error"]

    asyncio.run(again())


def test_the_script_downloads_the_three_files_from_their_commits(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "models" / "flux3-klein"
    calls: list[tuple[str, str]] = []

    def download(url: str, dst: Path, size: int, sha: str, log: Any) -> None:
        calls.append((url, dst.name))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(CONTENT[dst.name])

    monkeypatch.setattr(models, "download_file", download)
    argv = ["--model", "flux3", "--models", str(folder), "--quality", QUALITY]
    assert qwen_download.main(argv) == qwen_download.EXIT_OK
    assert sorted(calls, key=lambda c: c[1]) == sorted([
        (models.hf_url(*models.FLUX3_SOURCES[name]), name) for name in PINS], key=lambda c: c[1])
    assert (models.hf_url(*models.FLUX3_SOURCES[VAE]), VAE) in calls
    assert models.hf_url(*models.FLUX3_SOURCES[VAE]) == (
        "https://huggingface.co/Comfy-Org/Flux-2-Klein-4B/resolve/533c4cbc3d3b89af1743d9466c434e89810391c0/"
        "split_files/vae/flux2-vae.safetensors")
    assert qwen_download.main(argv) == qwen_download.EXIT_OK and len(calls) == 3      # all there: nothing more


def test_the_script_links_the_gguf_the_artist_put_into_qwen_image(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """A copy already on this PC is linked (same drive, no copy): by name from models\\qwen-image (where the artist put
    FLUX.3's GGUF first), and from the HuggingFace cache at the pinned commit under its path in the repo (the VAE).
    Only the rest is downloaded; a wrong file of that name is not used."""
    models_dir = tmp_path / "models"
    _put(models_dir / "qwen-image", DENOISER)
    cache = tmp_path / "hf-cache"
    repo, revision, remote = models.FLUX3_SOURCES[VAE]
    snapshot = models.hf_repo_folder(cache, repo) / "snapshots" / revision
    (snapshot / remote).parent.mkdir(parents=True)
    (snapshot / remote).write_bytes(CONTENT[VAE])
    monkeypatch.setattr(models, "user_hf_caches", lambda: [cache])
    downloaded: list[str] = []

    def download(url: str, dst: Path, size: int, sha: str, log: Any) -> None:
        downloaded.append(dst.name)
        dst.write_bytes(CONTENT[dst.name])

    monkeypatch.setattr(models, "download_file", download)
    folder = models_dir / "flux3-klein"
    assert qwen_download.main(["--model", "flux3", "--models", str(folder), "--quality", QUALITY]) == 0
    assert downloaded == [TEXT_ENCODER]
    assert os.path.samefile(folder / DENOISER, models_dir / "qwen-image" / DENOISER)      # a hard link
    assert os.path.samefile(folder / VAE, snapshot / remote)
    assert (models_dir / "qwen-image" / DENOISER).read_bytes() == CONTENT[DENOISER]      # left as it was

    other = tmp_path / "other" / "models"                            # a wrong file of that name: downloaded instead
    (other / "qwen-image").mkdir(parents=True)
    (other / "qwen-image" / DENOISER).write_bytes(b"x" * PINS[DENOISER][0])
    downloaded.clear()
    monkeypatch.setattr(models, "user_hf_caches", lambda: [])
    assert qwen_download.main(["--model", "flux3", "--models", str(other / "flux3-klein"), "--quality", QUALITY]) == 0
    assert sorted(downloaded) == sorted(PINS)


# ---- the hub's API -------------------------------------------------------------------------------------------

@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, Download, list[Any]]]:
    dl = Download(tmp_path)
    qwen_adapter = FakeQwen(tmp_path)
    qwen_adapter.flux3_dir = dl.folder
    shown: list[tuple[Path, bool]] = []
    app = create_app({paths.QWEN: qwen_adapter, "imesh": FakeAdapter("imesh", "retopo", False, tmp_path)},
                     settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=_static(tmp_path), logs_dir=tmp_path / "logs", gpu_probe=lambda: RTX_3060,
                     flux3_download=dl.bg,
                     show_in_explorer=lambda path, select: shown.append((path, select)) or True)
    monkeypatch.delenv("SPELLCAST_NO_EXPLORER", raising=False)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
        yield client, dl, shown


def test_api_status_and_the_download_behind_the_engine_gate(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    client, dl, shown = api
    status = client.get("/api/status").json()
    assert status["flux3"] == {"available": False, "downloading": False, "waiting": False, "done_bytes": 0,
                               "total_bytes": WHOLE, "error": None, "engine": False}
    assert "engine" not in status["qwen_full"]                       # the full model's stays as the page knows it
    refused = client.post("/api/flux3/download")
    assert refused.status_code == 409 and refused.json()["detail"] == qwen.FLUX3_NEEDS_ENGINE
    assert not dl.bg.downloading() and not dl.record.exists() and shown == []
    monkeypatch.setattr(engine_pin, "ENGINE_HAS_FLUX3", True)         # the engine's pin moved to one with FLUX.3
    started = client.post("/api/flux3/download")
    assert started.status_code == 200, started.text
    assert started.json()["engine"] is True and started.json()["total_bytes"] == WHOLE
    assert started.json()["opened"] is True and shown == [(dl.folder, False)]       # models\flux3-klein opens
    deadline = time.monotonic() + 20
    while not client.get("/api/status").json()["flux3"]["available"]:
        assert time.monotonic() < deadline, client.get("/api/status").json()["flux3"]
        time.sleep(0.05)
    again = client.post("/api/flux3/download").json()
    assert again["available"] is True and again["downloading"] is False and len(shown) == 2


def test_the_hubs_own_flux3_download_is_of_its_qwens_size_and_folder(tmp_path: Path) -> None:
    qwen_adapter = FakeQwen(tmp_path)
    qwen_adapter.quality = "Q6_K"
    app = create_app({paths.QWEN: qwen_adapter}, settings=make_settings(tmp_path),
                     store=ProjectStore(tmp_path / "projects"), static_dir=_static(tmp_path),
                     logs_dir=tmp_path / "logs", gpu_probe=lambda: None)
    download = app.state.hub.flux3_download
    assert (download.model, download.quality, download.folder) == ("flux3", "Q6_K", tmp_path / "flux3-klein")
    assert download.log_file == tmp_path / "logs" / "flux3-download.log"
    assert app.state.hub.qwen_full_download.folder == tmp_path / "qwen-image"       # each in its own folder
    assert QwenAdapter_default_dirs() == (paths.MODELS_QWEN, paths.MODELS_FLUX3)


def QwenAdapter_default_dirs() -> tuple[Path, Path]:
    adapter = qwen.QwenAdapter(quality=QUALITY)
    return adapter.models_dir, adapter.flux3_dir


# ---- the adapter -----------------------------------------------------------------------------------------------

def test_the_request_of_a_flux3_prop(tmp_path: Path) -> None:
    """Its own 16 steps, no guidance, never a transparent background nor the First-Block Cache, whatever the prop
    holds (those are Qwen-Image's settings); up to 10 example pictures as for Qwen-Image."""
    refs = [tmp_path / "reference" / f"reference_00{i}.png" for i in range(1, 4)]
    params = {"prompt": "a knight in ornate armour, front view, studio light", "qwen_model": "flux3",
              "qwen_steps": 40, "transparent": True, "first_block_cache": True, "seed": 3, "batch_count": 2}
    body = qwen.qwen_request(params, refs, tmp_path / "out")
    assert (body["model"], body["steps"], body["cfg_scale"], body["negative_prompt"], body["sampler"]) == (
        "flux3", qwen.FLUX3_STEPS, 1.0, "", "euler")
    assert body["transparent"] is False and body["first_block_cache"] is False and len(body["ref_images"]) == 3
    assert qwen.FLUX3_STEPS == server.FLUX3_STEPS == 16 and qwen.MODEL_NAMES["flux3"] == "FLUX.3 Image Klein"
    assert qwen.qwen_settings({"qwen_model": "flux3"})["transparent"] is False
    with pytest.raises(StageError, match="Turbo, Full or FLUX.3 Klein"):
        qwen.qwen_settings({"qwen_model": "flux"})


def test_a_flux3_prop_needs_the_engine_and_its_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FakeQwen(tmp_path)
    flux3 = {"prompt": "a lamp", "qwen_model": "flux3"}
    with pytest.raises(StageError) as e:
        adapter.check_settings("images", flux3)
    assert str(e.value) == qwen.FLUX3_NEEDS_ENGINE and "next update" in qwen.FLUX3_NEEDS_ENGINE
    monkeypatch.setattr(engine_pin, "ENGINE_HAS_FLUX3", True)
    with pytest.raises(StageError) as e:
        adapter.check_settings("images", flux3)
    assert str(e.value) == qwen.FLUX3_MISSING and "not downloaded yet" in qwen.FLUX3_MISSING
    _put(adapter.flux3_dir, DENOISER, VAE)
    assert not adapter.flux3_on_disk()
    _put(adapter.flux3_dir, TEXT_ENCODER)
    assert adapter.flux3_on_disk()
    adapter.check_settings("images", flux3)                             # drawable now
    adapter.check_settings("images", {"prompt": "a lamp"})              # the Turbo never needed any of it


def test_projects_take_flux3_as_a_model(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    pid = store.add_prop("P", "Lamp")["id"]
    prop = store.patch_prop("P", pid, {"images": {"settings": {"qwen_model": "flux3"}}})
    assert prop["images"]["settings"]["qwen_model"] == "flux3"
    assert store.add_prop("P", "Chair")["images"]["settings"]["qwen_model"] == "flux3"     # like every setting
    with pytest.raises(ProjectError, match="Turbo, Full or FLUX.3 Klein"):
        store.patch_prop("P", pid, {"images": {"settings": {"qwen_model": "FLUX3"}}})


@pytest.fixture
def hub(tmp_path: Path, root_config: Callable[..., None]) -> Any:      # noqa: F811
    adapters = {paths.QWEN: FakeQwen(tmp_path), "pixal3d": FakeAdapter("pixal3d", "mesh", True, tmp_path),
                "imesh": FakeAdapter("imesh", "retopo", False, tmp_path)}
    app = create_app(adapters, settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=_static(tmp_path), logs_dir=tmp_path / "logs", gpu_probe=lambda: RTX_3060)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield app, client


def test_a_flux3_job_through_the_hub(hub: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refused at once while the engine cannot draw FLUX.3 (the launcher is never asked); with the gate open and its
    files there, the launcher gets "flux3" with its own 16 steps and no transparent background, and the pictures are
    saved without alpha (Pixal3D removes their background)."""
    app, client = hub
    pid = _prop(client)
    assert client.patch(f"/api/projects/Demo/props/{pid}",
                        json={"images": {"settings": {"qwen_model": "flux3", "transparent": True}}}).status_code == 200
    refused = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [pid],
                                             "params": {"prompt": "a barrel"}})
    assert refused.status_code == 400 and refused.json()["detail"] == f"Barrel: {qwen.FLUX3_NEEDS_ENGINE}"
    monkeypatch.setattr(engine_pin, "ENGINE_HAS_FLUX3", True)
    adapter = app.state.hub.adapters[paths.QWEN]
    _put(adapter.flux3_dir, DENOISER, TEXT_ENCODER, VAE)
    assert client.get("/api/status").json()["flux3"]["available"] is True
    job = _run(client, pid, prompt="a wooden barrel with iron bands, side view, soft studio light")
    assert job["status"] == "done", job
    sent = _launcher(client, "/fake/qwen")[-1]
    assert (sent["model"], sent["steps"], sent["transparent"], sent["first_block_cache"]) == ("flux3", 16, False, False)
    images = _images(client, pid)
    prop_dir = app.state.hub.store.prop_dir("Demo", pid)
    assert _color_type((prop_dir / images["candidates"][0]).read_bytes()) == 2               # RGB: no alpha


# ---- the launcher ----------------------------------------------------------------------------------------------

def test_the_launcher_draws_flux3_with_its_own_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    """16 steps on the engine's own schedule (no custom sigmas, no flow shift), euler, no guidance, no negative prompt,
    no First-Block Cache (the variable says 0, so a full-model picture's value cannot leak), the prompt never wrapped in
    Qwen's RGBA words, its example pictures as RGB (a cut-out laid on white: opaque_reference; Qwen-Image's stay RGBA)
    and at the engine's own size up to two of them (flux3_reference_pixels; no prefix-cache budget)."""
    flux3 = {"steps": 16, "sigmas": None, "cfg_scale": 1.0, "sampler": "euler", "first_block_cache": False}
    for asked in ((25, 1.0, "euler", None), (40, 6.0, "dpm++2m", True)):
        for card in ((6, 1), (8, 6)):
            assert server.sampling("flux3", *asked, card) == flux3
    assert not server.draws_alpha("flux3") and server.draws_alpha("turbo") and server.draws_alpha("full")
    assert server.diffusion_file(QUALITY, "flux3") == DENOISER

    class Lib(_RecordingLib):
        def generate_image(self, ctx: Any, params: Any, out: Any, count: Any) -> bool:
            g = params._obj
            self.given.append({"prompt": g.prompt, "steps": g.sample_params.sample_steps,
                               "cfg": g.sample_params.guidance.txt_cfg, "negative": g.negative_prompt,
                               "sigmas": bool(g.sample_params.custom_sigmas),
                               "flow_shift": g.sample_params.flow_shift, "scheduler": g.sample_params.scheduler,
                               "references": g.ref_images_count,
                               "channels": {g.ref_images[i].channel for i in range(g.ref_images_count)},
                               "ref_image_args": g.ref_image_args, "cache": os.environ.get("SD_QWEN_FIRST_BLOCK_CACHE")})
            return False

    lib = Lib()
    engine = _fake_engine(monkeypatch, lib, capability=(6, 1))         # a GTX 10xx: the full model's cache by the card
    for req in (server.GenerateRequest(prompt="a knight", out_dir="C:/x", model="full", transparent=True),
                server.GenerateRequest(prompt="a knight", out_dir="C:/x", model="flux3", transparent=True, steps=40,
                                       cfg_scale=6.0, negative_prompt="blurry", first_block_cache=True)):
        with pytest.raises(server.GenerationFailed):
            engine.generate(req, 7, [_FakeReference()] * 3)
    full, flux3_given = lib.given
    assert full["prompt"].startswith(server.RGBA_PREFIX.encode()) and full["cache"] == server.FIRST_BLOCK_CACHE_SKIP
    assert full["ref_image_args"] is not None and full["channels"] == {4}     # Qwen-Image's prefix-cache budget, RGBA
    assert full["flow_shift"] == float("inf") and full["scheduler"] == sdcpp.SCHEDULER_COUNT   # the DLL's "own"
    assert flux3_given == {"prompt": b"a knight", "steps": 16, "cfg": 1.0, "negative": b"", "sigmas": False,
                           "flow_shift": float("inf"), "scheduler": sdcpp.SCHEDULER_COUNT, "references": 3,
                           "channels": {3}, "ref_image_args": b"vae_input_max_pixels=699050", "cache": "0"}
    with pytest.raises(server.GenerationFailed):                       # up to two: the engine's own size
        engine.generate(server.GenerateRequest(prompt="a knight", out_dir="C:/x", model="flux3"), 7,
                        [_FakeReference()] * 2)
    assert lib.given[-1]["ref_image_args"] is None and lib.given[-1]["channels"] == {3}


def _three_model_engine(folder: Path, flux3_engine: bool = True) -> tuple[Any, Any]:
    """A real QwenEngine on a _ContextLib that also keeps each context's other files and settings."""
    for name in (server.VAE_FILE, server.LLM_FILE, server.VISION_FILE, "qwen-image-2.1-turbo-Q4_K_M.gguf"):
        (folder / "qwen-image" / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / "qwen-image" / name).write_bytes(b"")
    files, missing = server.model_files(folder / "qwen-image", QUALITY)
    assert missing == [] and files["flux3"] == folder / "flux3-klein" / DENOISER    # FLUX3_FOLDER beside --models

    class Lib(_ContextLib):
        def __init__(self) -> None:
            super().__init__()
            self.params: list[dict[str, Any]] = []

        def new_sd_ctx(self, params: Any) -> int | None:
            p = params._obj
            self.params.append({"vae": Path(p.vae_path.decode()).name, "llm": Path(p.llm_path.decode()).name,
                                "vision": p.llm_vision_path, "model_args": p.model_args, "sage": p.sage_attn})
            return super().new_sd_ctx(params)

    lib = Lib()
    placement = {**server.choose_placement(11 * GiB, 10_664), "mmap": False, "capability": (8, 6),
                 "attention": "sage", "flux3_attention": server.flux3_attention_choice("auto", (8, 6))}
    engine = server.QwenEngine(lib, files, placement, server.Progress(), model=server.start_model(files),
                               flux3_engine=flux3_engine)
    engine.load()
    return engine, lib


def test_switching_between_any_two_models_frees_the_one_loaded_first(tmp_path: Path) -> None:
    """Turbo -> FLUX.3 -> Turbo: each switch frees the loaded context before the next is made (never two); FLUX.3's
    context has its own VAE and text encoder, no vision part, none of Qwen-Image's model args, and flash attention
    (the sage op was judged for Qwen-Image only); a FLUX.3 the engine cannot load says so, and the Turbo comes back."""
    engine, lib = _three_model_engine(tmp_path)
    assert not engine.model_on_disk("flux3") and engine.missing_files("flux3") == [DENOISER, VAE, TEXT_ENCODER]
    _put(tmp_path / "flux3-klein", DENOISER, TEXT_ENCODER, VAE)
    assert engine.model_on_disk("flux3") and engine.can_draw("flux3")
    engine.use_model("flux3")
    assert lib.freed == [101] and lib.made[-1] == (102, DENOISER) and engine.model == "flux3"
    assert lib.params[-1] == {"vae": VAE, "llm": TEXT_ENCODER, "vision": None, "model_args": None, "sage": False}
    assert lib.params[0]["vae"] == server.VAE_FILE and lib.params[0]["vision"] is not None
    assert lib.params[0]["model_args"] is not None and lib.params[0]["sage"] is True       # the Turbo's, as before
    engine.use_model("turbo")
    assert lib.freed == [101, 102] and lib.made[-1] == (103, "qwen-image-2.1-turbo-Q4_K_M.gguf")
    assert lib.params[-1]["vae"] == server.VAE_FILE and lib.params[-1]["llm"] == server.LLM_FILE
    assert server.flux3_attention_choice("sage", (8, 6)) == "sage" and server.flux3_attention_choice("sage", (7, 5)) \
        == "flash"
    lib.fail_for = {DENOISER}                                         # e.g. an engine build that cannot load it
    with pytest.raises(server.GenerationFailed, match=server.FLUX3_NOT_LOADED):
        engine.use_model("flux3")
    assert engine.model == "turbo" and engine.ready and lib.made[-1][1] == "qwen-image-2.1-turbo-Q4_K_M.gguf"


def test_the_routes_refuse_flux3_without_the_engine_or_its_files(root: Path, fake_pictures: list[Path],  # noqa: F811
                                                                tmp_path: Path) -> None:
    folder = root / "models"
    engine, lib = _three_model_engine(folder, flux3_engine=False)
    engine.generate = lambda req, seed, refs: [FakePicture(req.transparent)]
    client = TestClient(server.build_app(engine, server.GpuWorker(), engine.progress, root, {"quality": QUALITY}))
    health = client.get("/spellcast/health").json()
    assert health["models"] == {"turbo": True, "full": False, "flux3": False} and health["flux3_engine"] is False
    r = client.post("/spellcast/generate", json=_request(root, model="flux3").model_dump())
    assert r.status_code == 400 and r.json() == {"detail": server.FLUX3_NO_ENGINE, "kind": "input"}
    engine.flux3_engine = True
    r = client.post("/spellcast/generate", json=_request(root, model="flux3").model_dump())
    assert r.status_code == 400 and r.json()["detail"] == (
        f"FLUX.3 Image Klein is not downloaded yet (models\\flux3-klein\\{DENOISER}). Download it with its button "
        "under Model on the Images page, or pick Turbo.")
    _put(folder / "flux3-klein", DENOISER, TEXT_ENCODER, VAE)
    assert client.get("/spellcast/health").json()["models"]["flux3"] is True
    ref = str(root / "p1" / "reference" / "reference_001.png")
    engine.has_vision = False                                          # FLUX.3's example pictures need no vision part
    r = client.post("/spellcast/generate", json=_request(root, model="flux3", transparent=True,
                                                         ref_images=[ref]).model_dump())
    assert r.status_code == 200, r.text
    assert isinstance(r.json()["timings"]["switch"], float) and engine.model == "flux3" and lib.freed == [101]
    assert "vision file" in server.validate(_request(root, ref_images=[ref]), root, False)    # Qwen-Image's still do


def test_the_log_line_of_a_flux3_request(root: Path) -> None:  # noqa: F811
    req = server.GenerateRequest(prompt="x", out_dir=str(root), model="flux3", steps=30, transparent=True)
    words = server.request_words(req, 5, (8, 6))
    assert "with FLUX.3 Image Klein, 16 steps (its own schedule; the request's 30 steps not used), no guidance" in words
    assert words.endswith("transparent off (no alpha channel), first-block cache off")


def test_model_files_find_flux3_beside_qwen_image_or_where_told(tmp_path: Path) -> None:
    files, _missing = server.model_files(tmp_path / "qwen-image", QUALITY)
    assert (files["flux3"], files["flux3_vae"], files["flux3_llm"]) == (
        tmp_path / "flux3-klein" / DENOISER, tmp_path / "flux3-klein" / VAE, tmp_path / "flux3-klein" / TEXT_ENCODER)
    files, _missing = server.model_files(tmp_path / "qwen-image", "Q8_0", tmp_path / "elsewhere")
    assert files["flux3"] == tmp_path / "elsewhere" / "flux-3-image-klein-Q8_0.gguf"
    args = server.parse_args(["--sdcpp", "S", "--models", "M", "--port", "1"])
    assert (args.flux3_models, args.flux3_engine) == (None, "auto")
    sets = server.model_sets(files)
    assert sets["flux3"]["vision"] is None and sets["turbo"]["vae"] == files["vae"]


# ---- the bench -------------------------------------------------------------------------------------------------

def test_the_bench_runs_flux3_and_switches_to_it(tmp_path: Path) -> None:
    """--model turbo flux3 turbo: run N's model (a switch where it changes), FLUX.3's body as the hub makes it (its own
    16 steps, opaque, never the bench's --cfg), its files looked for in models\flux3-klein or --flux3-dir, and the
    launcher's --flux3-engine and --flux3-models passed on only when asked."""
    from spellcast.adapters.base import LaunchSpec
    from spellcast.profiling import qwen_bench as bench

    args = bench.parse_args(["--model", "turbo", "flux3", "turbo", "--runs", "3", "--cfg", "6", "--flux3-engine", "on",
                             "--flux3-dir", str(tmp_path / "f3")])
    assert [bench.run_model(args, n) for n in (1, 2, 3)] == ["turbo", "flux3", "turbo"]
    body = bench.request_body(args, 7, [], tmp_path / "run-2", model="flux3")
    assert (body["model"], body["steps"], body["cfg_scale"], body["negative_prompt"], body["transparent"]) == (
        "flux3", 16, 1.0, "", False)
    assert bench.model_words(args) == ("turbo (its own 8 steps), then flux3 (its own 16 steps), then turbo (its own 8 "
                                       "steps) (repeated)")
    assert bench.launcher_switches(args)[-2:] == ["--flux3-engine", "on"]
    assert "--flux3-engine" not in bench.launcher_switches(bench.parse_args([]))
    qwen_dir = tmp_path / "q"
    qwen_dir.mkdir()
    (qwen_dir / "qwen-image-2.1-turbo-Q4_K_M.gguf").write_bytes(b"")
    flux3_dir = tmp_path / "f3"
    assert bench.missing_models(args, qwen_dir, QUALITY, flux3_dir) == [str(flux3_dir / n) for n in PINS]
    _put(flux3_dir, DENOISER, TEXT_ENCODER, VAE)
    assert bench.missing_models(args, qwen_dir, QUALITY, flux3_dir) == []
    spec = LaunchSpec(argv=["python", "qwen_server.py", "--projects-root", "P", "--models", "M", "--flux3-models", "F",
                            "--sdcpp", "S"], cwd=tmp_path, env={}, ready_timeout=5.0)
    moved = bench.bench_launch(spec, tmp_path / "bench", flux3_dir=flux3_dir)
    assert moved.argv == ["python", "qwen_server.py", "--projects-root", str(tmp_path / "bench"), "--models", "M",
                          "--flux3-models", str(flux3_dir), "--sdcpp", "S"]
