"""Hi3D v1.0, the 3D stage's second geometry (a prop's mesh.settings.geometry: "hi3d" | "pixal3d"): the setting per
prop (new props take their project's Shape, props saved before the choice, PATCH), what the Pixal3D adapter sends and
records, the hub's words when Hi3D's networks are missing, /api/status "hi3d" and the download of Hi3D's AI models on
request (POST /api/hi3d/download: the 3D stage's own, no part of the install since 2026-11-03; its calibration step
after), the installer (pins, packages, the submodule hook; nothing of the models) and the release's checks and licence
texts. Pixal3D is tests/fake_tool.py.

Fast (no network, no GPU). Run with the hub venv:  python -m pytest -p no:cacheprovider code/tests
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from spellcast import model_downloads, paths
from spellcast.adapters import pixal3d
from spellcast.adapters.base import StageError
from spellcast.install import components, models
from spellcast.install.components import InstallError
from spellcast.jobs import JobError
from spellcast.projects import (DEFAULT_GEOMETRY, GEOMETRIES, ProjectError, ProjectStore, check_mesh_settings,
                                default_geometry, mesh_geometry)
from spellcast.server import create_app
from test_core import FakeAdapter, make_settings
from test_install_and_adapters import _RecordingCtx
from test_mesh_batch import asked, mesh_of, pixal3d_hub, project, watch

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Spellcast3D is Windows-only")

if str(paths.LAUNCHERS) not in sys.path:     # the launchers import their siblings as top-level modules
    sys.path.insert(0, str(paths.LAUNCHERS))

import pixal3d_server  # noqa: E402
import qwen_download  # noqa: E402

NOTICE = ("Hi3D is licensed under the Hi3D Open License Agreement Version 1.0. Copyright © 2026 OCEANO HK HOLDING "
          "LIMITED. All rights not expressly granted are reserved. The Hi3D trademarks are owned by OCEANO HK HOLDING "
          "LIMITED or its applicable affiliate(s).")
LICENSE_URL = "https://huggingface.co/hi3d-official/Hi3Dv1.0/blob/d502e938cbedc6e87aba8884614c5ca6d177a4b6/LICENSE"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(autouse=True)
def hi3d_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Hi3D's code on this PC (SPELLCAST_HI3D_CODE), as in a Spellcast3D that ships it; the tests of one that does
    not take it away (_no_hi3d_code)."""
    code = tmp_path / "hi3d-code"
    (code / "sparc3d").mkdir(parents=True)
    (code / "sparc3d" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setenv(paths.HI3D_CODE_ENV, str(code))
    return code


def _no_hi3d_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """This Spellcast3D without Hi3D's code: no code/hi3d, no SPELLCAST_HI3D_CODE, .gitmodules not naming it."""
    monkeypatch.delenv(paths.HI3D_CODE_ENV)
    monkeypatch.setattr(paths, "HI3D_DIR", tmp_path / "no-code")
    monkeypatch.setattr(components, "hi3d_submodule", lambda: False)


def _install_quietly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pixal3D.install without its downloads, links, copies and calibration: its commands alone are recorded."""
    for name in ("_download_models", "_make_copies", "_link_models", "_link_user_hf_cache", "_pip_tested_versions",
                 "_calibrate"):
        monkeypatch.setattr(components.Pixal3D, name, lambda self, *args, **kwargs: None)
    monkeypatch.setattr(components, "share_torch_files", lambda ctx, venv_id: None)
    monkeypatch.setattr(components, "_pixal3d_pip_pins", lambda: ((), ()))
    monkeypatch.setattr(components, "_card", lambda: None)     # this PC's card never decides; a test gives one


# ---- the setting per prop -------------------------------------------------------------------------------------

def test_new_props_take_their_projects_shape_picked_last_and_pixal3d_at_first(tmp_path: Path) -> None:
    """The Shape picked last in a project is that project's for its new props (and the draft; project.json
    "geometry"): one Hi3D pick makes Hi3D the project's Shape until Pixal3D is picked again, and another project
    starts on Pixal3D (before 2026-11-03 the pick stuck for the whole installation). A prop keeps the Shape it was
    given; a pick on the draft itself changes nothing (the 3D stage shows no draft)."""
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    first = store.add_prop("P", "Barrel")
    assert GEOMETRIES == ("hi3d", "pixal3d") and DEFAULT_GEOMETRY == default_geometry() == "pixal3d"
    assert store.get_project("P")["geometry"] == "pixal3d"
    assert first["mesh"]["settings"]["geometry"] == "pixal3d" and first["mesh"]["geometry"] is None
    assert store.ensure_draft("P")["mesh"]["settings"]["geometry"] == "pixal3d"
    changed = store.patch_prop("P", first["id"], {"mesh": {"settings": {"geometry": "hi3d", "seed": 9}}})
    assert changed["mesh"]["settings"]["geometry"] == "hi3d" and store.get_project("P")["geometry"] == "hi3d"
    second = store.add_prop("P", "Crate")
    assert second["mesh"]["settings"]["geometry"] == "hi3d" and second["mesh"]["settings"]["seed"] == 9
    assert store.ensure_draft("P")["mesh"]["settings"]["geometry"] == "hi3d", "the draft follows"
    store.create_project("Q")                                                       # another project: Pixal3D's
    assert store.get_project("Q")["geometry"] == "pixal3d"
    assert store.add_prop("Q", "Lamp")["mesh"]["settings"]["geometry"] == "pixal3d"
    store.patch_prop("P", second["id"], {"mesh": {"settings": {"geometry": "pixal3d"}}})
    assert store.get_project("P")["geometry"] == "pixal3d"
    assert store.get_prop("P", first["id"])["mesh"]["settings"]["geometry"] == "hi3d", "a prop keeps its own"
    assert store.ensure_draft("P")["mesh"]["settings"]["geometry"] == "pixal3d"
    store.patch_prop("P", first["id"], {"mesh": {"settings": {"seed": 4}}})          # no Shape picked: no change
    assert store.get_project("P")["geometry"] == "pixal3d"
    store.patch_prop("P", second["id"], {"mesh": {"settings": {"geometry": "hi3d"}}})
    again = ProjectStore(tmp_path / "projects")                                      # the next start of the hub
    assert again.add_prop("P", "Pot")["mesh"]["settings"]["geometry"] == "hi3d", "kept in the project file"
    assert again.add_prop("Q", "Pan")["mesh"]["settings"]["geometry"] == "pixal3d"
    draft = again.ensure_draft("Q")
    again.patch_prop("Q", draft["id"], {"mesh": {"settings": {"geometry": "hi3d"}}})
    assert again.get_project("Q")["geometry"] == "pixal3d"
    assert default_geometry("hi3d") == "hi3d" and default_geometry("nobody") == "pixal3d"


def test_a_prop_saved_before_the_choice_keeps_pixal3d(tmp_path: Path) -> None:
    """Its settings name no shape and its model says nothing of who made it: Pixal3D did. Reading changes no file."""
    root = tmp_path / "projects"
    folder = root / "Old"
    ids = ["p_00000a", "p_00000b", "p_00000c", "p_00000d"]
    for prop_id in ids:
        (folder / "props" / prop_id).mkdir(parents=True)
    picture = {"settings": {"prompt": "a barrel"}, "candidates": ["images/0001.png"], "chosen": "images/0001.png"}
    old = {"name": "Old", "props": [
        {"id": ids[0], "name": "Made", "images": picture,
         "mesh": {"settings": {"seed": 5}, "glb": "mesh/mesh.glb", "source": "images/0001.png"}},
        {"id": ids[1], "name": "Own", "images": picture,
         "mesh": {"settings": {}, "glb": "mesh/own.glb", "imported": {"materials": []}}},
        {"id": ids[2], "name": "None yet", "images": picture, "mesh": {"settings": {}}},
        {"id": ids[3], "name": "Hand-edited", "images": picture,
         "mesh": {"settings": {"geometry": "x"}, "glb": "mesh/mesh.glb", "geometry": "y"}}]}
    (folder / "project.json").write_text(json.dumps(old), encoding="utf-8")
    before = (folder / "project.json").read_bytes()
    store = ProjectStore(root)
    props = store.open_project("Old")["props"]
    assert [p["mesh"]["settings"]["geometry"] for p in props] == ["pixal3d"] * 4
    assert [p["mesh"]["geometry"] for p in props] == ["pixal3d", None, None, None]
    assert props[0]["mesh"]["settings"]["seed"] == 5
    assert (folder / "project.json").read_bytes() == before
    # a new prop after them is Pixal3D's
    assert store.ensure_draft("Old")["mesh"]["settings"]["geometry"] == "pixal3d"


def test_a_model_says_which_geometry_made_it_and_an_import_says_none(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    pid = store.add_prop("P", "Barrel")["id"]
    made = store.apply_stage_patch("P", pid, "mesh", {"glb": "mesh/mesh.glb", "source": "x", "geometry": "hi3d"})
    assert made["mesh"]["geometry"] == "hi3d"
    own = store.apply_stage_patch("P", pid, "mesh", {"glb": "mesh/own.glb", "imported": {"materials": []}})
    assert own["mesh"]["geometry"] is None
    again = store.apply_stage_patch("P", pid, "mesh", {"glb": "mesh/mesh.glb", "geometry": "pixal3d"})
    assert again["mesh"]["geometry"] == "pixal3d" and again["mesh"]["imported"] is None


# ---- what Pixal3D is sent, and what the hub says ----------------------------------------------------------------

def test_each_prop_is_sent_its_geometry_and_its_model_records_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pixal3d, "POLL_INTERVAL", 0.02)
    store, ids = project(tmp_path, b"", b"")
    store.patch_prop("P", ids[0], {"mesh": {"settings": {"geometry": "hi3d"}}})
    store.patch_prop("P", ids[1], {"mesh": {"settings": {"geometry": "pixal3d"}}})

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store) as (pm, queue, _):
            job = await queue.submit("mesh", "P", ids)
            final, _seen = await watch(queue, job["id"])
            assert final["status"] == "done", final
            sent = {item["id"]: item["geometry"] for item in (await asked(pm))[-1]["items"]}
            assert sent == {ids[0]: "hi3d", ids[1]: "pixal3d"}
            # a job may not ask for a shape there is not
            with pytest.raises(JobError, match="Hi3D or Pixal3D"):
                await queue.submit("mesh", "P", ids[:1], params={"geometry": "trellis"})

    asyncio.run(main())
    assert [mesh_of(store, prop_id)["geometry"] for prop_id in ids] == ["hi3d", "pixal3d"]


def test_a_hi3d_prop_without_hi3d_fails_with_words_that_say_how_to_get_it(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """The launcher fails it with the kind "missing_file"; the other props go on. Hi3D's networks are not on this PC
    (each test's own, empty models\\hi3d), so the hub's words point at the download, or at Pixal3D."""
    monkeypatch.setattr(pixal3d, "POLL_INTERVAL", 0.02)
    store, ids = project(tmp_path, b"", b"")
    store.patch_prop("P", ids[0], {"mesh": {"settings": {"geometry": "hi3d"}}})
    store.patch_prop("P", ids[1], {"mesh": {"settings": {"geometry": "pixal3d"}}})

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--no-hi3d") as (_pm, queue, _):
            job = await queue.submit("mesh", "P", ids)
            final, _seen = await watch(queue, job["id"])
            assert final["status"] == "error" and final["errors"] == {ids[0]: pixal3d.HI3D_MISSING}

    asyncio.run(main())
    assert mesh_of(store, ids[0])["glb"] is None and mesh_of(store, ids[1])["geometry"] == "pixal3d"
    assert "Download" in pixal3d.HI3D_MISSING and "Pixal3D" in pixal3d.HI3D_MISSING
    # the same kind for a Pixal3D prop, or with Hi3D's networks here (the launcher's own words then)
    detail = {"kind": "missing_file", "detail": "Hi3D's networks are not installed."}
    hi3d = pixal3d._Request({"geometry": "hi3d", "resolution": 1536}, Path(), Path(), None)
    own = pixal3d._Request({"geometry": "pixal3d", "resolution": 1536}, Path(), Path(), None)
    assert pixal3d._failure(detail, hi3d) == pixal3d.HI3D_MISSING
    assert pixal3d._failure({**detail, "detail": "A file is missing."}, own) == "A file is missing."
    monkeypatch.setattr(models, "hi3d_on_disk", lambda folder=None: True)
    assert pixal3d._failure(detail, hi3d) == "Hi3D's networks are not installed."
    with pytest.raises(StageError, match="Hi3D or Pixal3D"):
        pixal3d.Pixal3DAdapter().check_settings("mesh", {"geometry": "nobody"})
    pixal3d.Pixal3DAdapter().check_settings("mesh", {"geometry": "hi3d"})


def test_api_status_says_whether_hi3d_is_on_this_pc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The download's snapshot (model_downloads "hi3d": into models\\hi3d, Hi3D's whole size to come), "available"
    only with Hi3D's code here as well, and "code"; the route starts the download (a stand-in script here) and refuses
    it without Hi3D's code."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>Spellcast3D</h1>", encoding="utf-8")
    adapters = {"pixal3d": FakeAdapter("pixal3d", "mesh", True, tmp_path)}
    app = create_app(adapters, settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=static, logs_dir=tmp_path / "logs")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/status").json()["hi3d"] == {
            "available": False, "code": True, "downloading": False, "waiting": False, "finishing": False,
            "done_bytes": 0, "total_bytes": models.HI3D_BYTES, "error": None}
        _hi3d_here(tmp_path, monkeypatch)
        hi3d = client.get("/api/status").json()["hi3d"]
        assert hi3d["available"] is True and hi3d["done_bytes"] == hi3d["total_bytes"] == sum(
            size for size, _sha in models.HI3D_FILES.values())
        started = client.post("/api/hi3d/download")                 # there already: nothing happens
        assert started.status_code == 200 and started.json()["available"] is True
        _no_hi3d_code(tmp_path, monkeypatch)                # the page then says Hi3D is not in this Spellcast3D
        hi3d = client.get("/api/status").json()["hi3d"]
        assert hi3d["available"] is False and hi3d["code"] is False
        refused = client.post("/api/hi3d/download")
        assert refused.status_code == 409 and "not part of this Spellcast3D" in refused.json()["detail"]


FAKE_HI3D_DOWNLOAD = r'''
import json, sys, time
from pathlib import Path
plan = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))
argv = sys.argv[1:]
Path(plan["record"]).write_text(json.dumps({"argv": argv}), encoding="utf-8")
folder = Path(argv[argv.index("--models") + 1])
for name, data in plan["files"]:
    (folder / name).parent.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(data, encoding="utf-8")
print("fake hi3d download done", flush=True)
sys.exit(plan.get("exit", 0))
'''


async def _until(condition: Any, what: str, seconds: float = 20.0) -> None:
    deadline = asyncio.get_running_loop().time() + seconds
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, f"waited too long for {what}"
        await asyncio.sleep(0.02)


def test_the_download_runs_the_stand_in_script_then_measures_hi3ds_convolutions(
        hi3d_small: dict[str, bytes], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """model_downloads "hi3d": into models\\hi3d (no installer waited for: no installer writes there), its total what
    is missing, and - the files there - the `then` step (the hub's: hi3d_calibration) with "finishing" in the
    snapshot; a job started through the hub meanwhile is a job as any other."""
    folder = tmp_path / "models" / "hi3d"
    script = tmp_path / "fake_hi3d_download.py"
    script.write_text(FAKE_HI3D_DOWNLOAD, encoding="utf-8")
    record = tmp_path / "record.json"
    script.with_suffix(".json").write_text(json.dumps({
        "record": str(record), "files": [[rel, data.decode()] for rel, data in hi3d_small.items()]}), encoding="utf-8")
    whole = sum(len(data) for data in hi3d_small.values())
    steps: list[Path] = []
    go = asyncio.Event()

    async def then(log_file: Path) -> None:
        steps.append(log_file)
        await go.wait()

    bg = model_downloads.BackgroundDownload("hi3d", "Q4_K_M", folder, python=Path(sys.executable), script=script,
                                            install_lock=tmp_path / ".install.lock",
                                            log_file=tmp_path / "logs" / "hi3d-download.log", wait_poll=0.05, then=then)
    assert bg.spec.status == "hi3d" and bg.spec.waits_for is None and bg.spec.says(start=True) == "Hi3D's AI models"

    async def main() -> None:
        assert bg.snapshot() == {"available": False, "downloading": False, "waiting": False, "finishing": False,
                                 "done_bytes": 0, "total_bytes": whole, "error": None}
        bg.start()
        await _until(lambda: bg.finishing, what="the download to end")
        state = bg.snapshot()
        assert state["available"] and state["finishing"] and not state["downloading"] and state["error"] is None
        assert json.loads(record.read_text(encoding="utf-8"))["argv"] == [
            "--model", "hi3d", "--models", str(folder), "--quality", "Q4_K_M"]
        assert models.hi3d_on_disk(folder)
        go.set()
        await _until(lambda: not bg.downloading(), what="the step after it")
        assert bg.snapshot() == {"available": True, "downloading": False, "waiting": False, "finishing": False,
                                 "done_bytes": whole, "total_bytes": whole, "error": None}
        assert steps == [tmp_path / "logs" / "hi3d-download.log"]
        await bg.close()

    asyncio.run(main())


def test_hi3ds_calibration_step_runs_pixal3ds_calibration_for_spconv_alone(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """hi3d_calibration: the installer's calibration command with --only spconv (the record's other sections stand),
    its output in the download's log; a failure is a warning (Hi3D tunes at its first use; the next start measures)."""
    commands: list[list[str]] = []

    def command(only: str = "", skip: str = "", check: bool = True) -> list[str] | None:
        commands.append([only, skip])
        return [sys.executable, "-c", "print('calibrated for', 'spconv')"]

    monkeypatch.setattr(components, "pixal3d_calibration_command", command)
    monkeypatch.setattr(components, "pixal3d_calibration_env", lambda: {"HF_HUB_OFFLINE": "1"})
    log_file = tmp_path / "logs" / "hi3d-download.log"
    log_file.parent.mkdir()
    asyncio.run(model_downloads.hi3d_calibration(log_file))
    assert commands == [["spconv", ""]]
    deadline = time.monotonic() + 5
    while "calibrated for spconv" not in log_file.read_text(encoding="utf-8"):
        assert time.monotonic() < deadline, log_file.read_text(encoding="utf-8")
        time.sleep(0.05)
    monkeypatch.setattr(components, "pixal3d_calibration_command", lambda **kw: None)   # nothing to run: nothing
    asyncio.run(model_downloads.hi3d_calibration(log_file))


def _hi3d_here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A small stand-in of Hi3D's networks, each file at its pinned size (its code is the hi3d_code fixture's)."""
    small = {rel: (len(rel), _sha(rel.encode())) for rel in models.HI3D_FILES}
    monkeypatch.setattr(models, "HI3D_FILES", small)
    for rel in small:
        (paths.MODELS_HI3D / rel).parent.mkdir(parents=True, exist_ok=True)
        (paths.MODELS_HI3D / rel).write_bytes(rel.encode())


def test_hi3d_is_on_disk_only_with_its_code_and_every_file_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert not models.hi3d_on_disk()
    _hi3d_here(tmp_path, monkeypatch)
    assert models.hi3d_on_disk() and models.hi3d_code() == tmp_path / "hi3d-code"
    victim = paths.MODELS_HI3D / "clip-vit-large-patch14" / "model.safetensors"
    victim.write_bytes(b"cut")                                      # not whole
    assert not models.hi3d_on_disk()
    victim.write_bytes(b"clip-vit-large-patch14/model.safetensors")
    assert models.hi3d_on_disk()
    _no_hi3d_code(tmp_path, monkeypatch)
    assert models.hi3d_code() == paths.HI3D_DIR and not models.hi3d_on_disk()     # every file, but no code


# ---- the installer ------------------------------------------------------------------------------------------

def test_hi3d_pins_are_the_releases_and_what_the_launcher_loads() -> None:
    files = models.HI3D_FILES
    assert files["Hi3Dv1.0/vae/model.safetensors"] == (
        376_629_712, "605d207c77f8db0b33c567a5861252f0090cf3c3604dfda1cfee09e8f0b53f3e")
    assert files["Hi3Dv1.0/transformer_512/model.safetensors"] == (
        3_675_679_280, "8ba88311dccf23463c6e10dd0d3bd5e0c5c35d1608c6f466413b6428b5ad8e56")
    assert files["Hi3Dv1.0/transformer_1024/model.safetensors"] == (
        3_675_679_280, "a2ca5e44dde51cacecd59b42fceeb5f14e75d55e6dd047337752c4e6f281008c")
    assert files["trellis2-shape-encoder/ckpts/shape_enc_next_dc_f16c32_fp16.safetensors"] == (
        708_797_208, "f37c5ff5b983b68e9946060000f09bc131f3e84318a2c8b7430a81e4b4636c41")
    assert files["dinov2-large/model.safetensors"][0] == 1_217_522_888
    assert files["clip-vit-large-patch14/model.safetensors"][0] == 1_710_540_580
    assert models.HI3D_SOURCES_PINNED == {
        "Hi3Dv1.0": ("hi3d-official/Hi3Dv1.0", "d502e938cbedc6e87aba8884614c5ca6d177a4b6"),
        "dinov2-large": ("facebook/dinov2-large", "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"),
        "clip-vit-large-patch14": ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"),
        "trellis2-shape-encoder": ("microsoft/TRELLIS.2-4B", "af44b45f2e35a493886929c6d786e563ec68364d")}
    # its licence and notices stay beside the weights
    assert {"Hi3Dv1.0/LICENSE", "Hi3Dv1.0/NOTICE", "Hi3Dv1.0/AUTHORS", "Hi3Dv1.0/THIRD_PARTY_LICENSES.md"} <= set(files)
    # the launcher looks for every network file and config before it offers Hi3D (hi3d_paths), the texts aside
    assert set(pixal3d_server.HI3D_NEEDED) == {rel for rel in files if rel.endswith((".json", ".safetensors"))}
    assert models.HI3D_SOURCES["trellis2-shape-encoder/ckpts/shape_enc_next_dc_f16c32_fp16.json"] == (
        "microsoft/TRELLIS.2-4B", "af44b45f2e35a493886929c6d786e563ec68364d", "ckpts/shape_enc_next_dc_f16c32_fp16.json")
    assert models.hf_url(*models.HI3D_SOURCES["Hi3Dv1.0/vae/model.safetensors"]) == (
        "https://huggingface.co/hi3d-official/Hi3Dv1.0/resolve/d502e938cbedc6e87aba8884614c5ca6d177a4b6/"
        "vae/model.safetensors")
    assert models.HI3D_BYTES == sum(size for size, _sha in files.values()) == 11_364_913_260
    assert all(len(sha) == 64 for _size, sha in files.values())
    assert "hi3d" in [folder.name for folder in models.FOLDERS]       # made at every start, like the others


def test_hi3d_packages_are_the_tested_ones_installed_with_pixal3d(monkeypatch: pytest.MonkeyPatch) -> None:
    pins = components.HI3D_PACKAGES
    assert "spconv-cu126==2.3.8" in pins and "cumm-cu126==0.7.11" in pins and "pyfqmr==0.5.0" in pins
    assert not [pin for pin in pins if pin.startswith("spconv-cu120")]          # it crashes under numpy 2
    tested = components.PIXAL3D_CONSTRAINTS.read_text(encoding="utf-8").splitlines()
    assert set(pins) <= set(tested), "constraints-pixal3d.txt has the same versions"
    pixal = components.Pixal3D()
    assert pixal.inputs()["hi3d"] == " ".join(pins)          # a new pin installs Pixal3D again
    # installed --no-deps after install.py and gradio, then imported, before Pixal3D's own check
    _install_quietly(monkeypatch)
    ctx = _RecordingCtx()
    pixal.install(ctx)
    gradio = next(i for i, c in enumerate(ctx.commands) if components.PIXAL3D_GRADIO[0] in c)
    hi3d = next(i for i, c in enumerate(ctx.commands) if pins[0] in c)
    assert gradio < hi3d and ctx.commands[hi3d][-len(pins) - 1:] == ["--no-deps", *pins]
    assert "import spconv.pytorch, pyfqmr" in ctx.commands[hi3d + 1][-1]
    assert "Pixal3D packages OK" in ctx.commands[-1][-1] and "spconv" not in ctx.commands[-1][-1]


def test_spconv_is_pixal3ds_own_check_below_sm80(monkeypatch: pytest.MonkeyPatch) -> None:
    """Below sm_80 the 3D decoders convolve on spconv: Pixal3D's own check imports it there (a failed spconv install
    fails Pixal3D instead of passing as Hi3D's warning), and only there."""
    from spellcast import gpu
    from test_install_and_adapters import _RecordingCtx

    _install_quietly(monkeypatch)
    for card, wanted in ((gpu.GpuInfo(0, "NVIDIA GeForce GTX 1080", 6.1, 8192, 500, "596.36"), True),
                         (gpu.GpuInfo(0, "NVIDIA GeForce RTX 2070", 7.5, 8192, 500, "596.36"), True),
                         (gpu.GpuInfo(0, "NVIDIA GeForce RTX 3060", 8.6, 12288, 500, "596.36"), False)):
        monkeypatch.setattr(components, "_card", lambda card=card: card)
        ctx = _RecordingCtx()
        components.Pixal3D().install(ctx)
        check = ctx.commands[-1][-1]
        assert "Pixal3D packages OK" in check
        assert ("spconv.pytorch" in check.splitlines()[1]) is wanted, (card.name, check)


def test_hi3ds_packages_failing_never_fail_pixal3d(monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    """A pip hiccup on spconv, or a spconv that will not import on some PC, is a warning: Pixal3D is checked and
    installed all the same (its fingerprint names Hi3D's pins, so every existing install runs through here)."""
    pins = components.HI3D_PACKAGES

    class Failing(_RecordingCtx):
        def __init__(self, trips: str) -> None:
            super().__init__()
            self.trips = trips

        def run(self, argv: list[str], **kwargs: Any) -> None:
            super().run(argv, **kwargs)
            if any(self.trips in str(arg) for arg in argv):
                raise InstallError(kwargs["fail"])

    _install_quietly(monkeypatch)
    for trips, words in ((pins[0], "Downloading or installing the packages of Hi3D failed"),
                         ("import spconv", "Hi3D was installed but does not start correctly")):
        ctx = Failing(trips)
        components.Pixal3D().install(ctx)
        said = capsys.readouterr().out
        assert f"Warning: {words}" in said and components.HI3D_LATER in said, said
        assert "Pixal3D packages OK" in ctx.commands[-1][-1]


@pytest.fixture
def hi3d_small(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Hi3D's manifest cut to two small files of its real names (so their sources are the real ones)."""
    data = {"Hi3Dv1.0/LICENSE": b"the licence", "dinov2-large/model.safetensors": b"dino weights"}
    monkeypatch.setattr(models, "HI3D_FILES", {rel: (len(d), _sha(d)) for rel, d in data.items()})
    monkeypatch.setattr(models, "user_hf_caches", lambda: [])
    monkeypatch.setattr(models, "EXTRA_SOURCES", [])
    monkeypatch.delenv("SPELLCAST_MODELS_FROM", raising=False)
    return data


def test_hi3ds_models_are_no_part_of_pixal3ds_install(hi3d_small: dict[str, bytes],
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """Since 2026-11-03 nobody sits through Hi3D's 11.4 GB at the install: Pixal3D's install and repair fetch
    nothing of them, miss nothing of them and need no room for them; its texts say Hi3D is a separate download the
    3D stage offers. Its code and packages still come with Pixal3D (the tests above)."""
    assert not hasattr(components, "HI3D_MODELS") and not hasattr(components.Pixal3D, "_fetch_hi3d")
    pixal = components.Pixal3D()
    assert "Hi3D" not in pixal.missing_files() and components.hi3d_work() is None
    assert pixal.models_gb == pytest.approx(components.PIXAL3D_MODELS_BYTES / 1e9)
    assert pixal.size_gb == pytest.approx(40.0 + models.PIXAL3D_COPIES_BYTES / 1e9 + components.HI3D_PACKAGES_GB)
    assert "separate 11.4 GB download" in pixal.about and "33 GB" in pixal.about
    monkeypatch.setattr(models, "download_file", lambda *a: pytest.fail("downloaded"))
    _install_quietly(monkeypatch)
    ctx = _RecordingCtx()
    pixal.install(ctx)
    assert "Pixal3D packages OK" in ctx.commands[-1][-1]
    monkeypatch.setattr(components.Pixal3D, "missing_files", lambda self: [])
    pixal.repair(_RecordingCtx())
    assert "Hi3D" not in " ".join(components.HI3D_LATER.split()) or "AI models" not in components.HI3D_LATER
    assert models.optional_manifest("hi3d", "Q4_K_M") == models.HI3D_FILES
    assert models.optional_sources("hi3d") == models.HI3D_SOURCES and "hi3d" in models.OPTIONAL_MODELS
    assert model_downloads.MODELS["hi3d"].folder.name == "hi3d"       # models\hi3d (the hub passes the folder)


def test_the_script_fetches_hi3ds_files_from_their_commits_and_links_copies(hi3d_small: dict[str, bytes],
                                                                            tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """launchers/qwen_download.py --model hi3d: the installer's own downloader (resumed, SHA-256 checked), each file
    into its folder (Hi3Dv1.0/, dinov2-large/, ...); a copy in the artist's HuggingFace cache at the pinned commit is
    hard-linked, not downloaded; a file put there by hand is checked once, and a wrong one fetched again."""
    folder = tmp_path / "models" / "hi3d"
    snapshot = tmp_path / "hf" / "snapshots" / "47b73ee"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(hi3d_small["dinov2-large/model.safetensors"])
    monkeypatch.setattr(models, "hf_cache_sources",
                        lambda repo, revision: [snapshot] if repo == "facebook/dinov2-large" else [])
    fetched: list[str] = []

    def download(url: str, dst: Path, size: int, sha: str, log: Any) -> None:
        fetched.append(url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(next(d for rel, d in hi3d_small.items() if url.endswith(rel.split("/", 1)[1])))
        models.record_verified(dst, sha)              # as the real one does once the SHA-256 is right

    monkeypatch.setattr(models, "download_file", download)
    argv = ["--model", "hi3d", "--models", str(folder), "--quality", "Q4_K_M"]
    assert not models.hi3d_on_disk(folder) and not model_downloads.available("hi3d", "Q4_K_M", folder)
    assert qwen_download.main(argv) == qwen_download.EXIT_OK
    assert fetched == [
        "https://huggingface.co/hi3d-official/Hi3Dv1.0/resolve/d502e938cbedc6e87aba8884614c5ca6d177a4b6/LICENSE"]
    linked = folder / "dinov2-large" / "model.safetensors"
    assert os.stat(linked).st_ino == os.stat(snapshot / "model.safetensors").st_ino      # a hard link, no download
    assert models.hi3d_on_disk(folder) and model_downloads.available("hi3d", "Q4_K_M", folder)
    fetched.clear()
    assert qwen_download.main(argv) == qwen_download.EXIT_OK and fetched == []       # there: nothing again
    stray = folder / "Hi3Dv1.0" / "LICENSE"
    stray.unlink()
    stray.write_bytes(b"not licence")                      # the pinned size, other bytes
    os.utime(stray, ns=(os.stat(stray).st_atime_ns, os.stat(stray).st_mtime_ns + 10**9))   # not the checked file
    assert qwen_download.main(argv) == qwen_download.EXIT_OK
    assert [url.rsplit("/", 1)[1] for url in fetched] == ["LICENSE"] and stray.read_bytes() == b"the licence"


def test_without_hi3ds_code_nothing_of_hi3d_is_downloaded_and_new_props_take_pixal3d(
        hi3d_small: dict[str, bytes], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Spellcast3D without Hi3D's code (code/hi3d not shipped yet): its AI models would be 11 GB for nothing, so
    the route refuses their download (the test of /api/status above) and nothing of Hi3D is missed; new props start
    on Pixal3D even in a project whose Shape says Hi3D; a prop set to Hi3D all the same is told Hi3D is not here, not
    to download."""
    _no_hi3d_code(tmp_path, monkeypatch)
    assert not models.hi3d_on_disk()
    pixal = components.Pixal3D()
    stray = paths.MODELS_HI3D / "Hi3Dv1.0" / "LICENSE"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"not licence")                      # the pinned size, other bytes: not even looked at
    assert "Hi3D" not in pixal.missing_files() and components.hi3d_work() is None
    monkeypatch.setattr(models, "download_file", lambda *a: pytest.fail("downloaded"))
    _install_quietly(monkeypatch)
    ctx = _RecordingCtx()
    pixal.install(ctx)
    assert "Pixal3D packages OK" in ctx.commands[-1][-1]
    assert default_geometry() == "pixal3d" and default_geometry("hi3d") == "pixal3d"
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    first = store.add_prop("P", "Barrel")
    store.patch_prop("P", first["id"], {"mesh": {"settings": {"geometry": "hi3d"}}})
    assert store.get_project("P")["geometry"] == "hi3d"
    assert store.add_prop("P", "Crate")["mesh"]["settings"]["geometry"] == "pixal3d"
    assert store.ensure_draft("P")["mesh"]["settings"]["geometry"] == "pixal3d"
    detail = {"kind": "missing_file", "detail": "Hi3D's networks are not installed."}
    hi3d = pixal3d._Request({"geometry": "hi3d", "resolution": 1536}, Path(), Path(), None)
    assert pixal3d._failure(detail, hi3d) == pixal3d.HI3D_NOT_HERE
    assert "Pixal3D" in pixal3d.HI3D_NOT_HERE and "Download" not in pixal3d.HI3D_NOT_HERE


def test_hi3d_code_is_checked_out_once_gitmodules_names_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "sc3d"
    (root / "code").mkdir(parents=True)
    monkeypatch.setattr(paths, "ROOT", root)
    monkeypatch.setattr(paths, "HI3D_DIR", root / "code" / "hi3d")
    checkouts: list[tuple[Path, str, str]] = []
    monkeypatch.setattr(components, "checkout_submodule",
                        lambda ctx, folder, marker, title: checkouts.append((folder, marker, title)))
    pixal = components.Pixal3D()
    assert not components.hi3d_submodule() and components.hi3d_work() is None
    pixal._checkout_hi3d(_RecordingCtx())
    assert checkouts == []                                  # not a submodule (yet): nothing to do
    (root / ".gitmodules").write_text('[submodule "hi3d"]\n\tpath = code/hi3d\n\turl = https://example/hi3d.git\n',
                                      encoding="utf-8")
    assert components.hi3d_submodule() and "not checked out" in components.hi3d_work()
    pixal._checkout_hi3d(_RecordingCtx())
    assert checkouts == [(root / "code" / "hi3d", "sparc3d/__init__.py", "Hi3D")]

    def fails(*args: Any) -> None:
        raise components.InstallError("no git here")

    monkeypatch.setattr(components, "checkout_submodule", fails)
    pixal._checkout_hi3d(_RecordingCtx())                    # a warning: Pixal3D works on
    assert "Warning: Hi3D's code could not be put into code\\hi3d (no git here)" in capsys.readouterr().out
    (root / "code" / "hi3d" / "sparc3d").mkdir(parents=True)
    (root / "code" / "hi3d" / "sparc3d" / "__init__.py").write_text("", encoding="utf-8")
    assert components.hi3d_work() is None


# ---- the release ------------------------------------------------------------------------------------------------

@functools.cache
def _prepare_release() -> Any:
    sys.path.insert(0, str(paths.ROOT))
    try:
        import prepare_release
    finally:
        sys.path.remove(str(paths.ROOT))
    return prepare_release


def test_the_licence_texts_carry_hi3ds_terms_and_notice() -> None:
    pr = _prepare_release()
    text = (paths.ROOT / "README - LICENSE.txt").read_text(encoding="utf-8")
    assert pr.license_text_gaps(text) == []
    for name in ("Hi3D Open License Agreement Version 1.0", "OCEANO HK HOLDING LIMITED", "Powered by Hi3D 1.0",
                 "Acceptable Use Policy"):
        assert name in pr.LICENSE_MUST_NAME
    assert NOTICE in text and LICENSE_URL in text and "200,000 monthly active users" in text and "60 days" in text
    assert NOTICE in (paths.MODELS / "README.txt").read_text(encoding="utf-8")


def test_the_release_check_compares_hi3ds_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    pr = _prepare_release()
    trees: dict[str, list[dict[str, Any]]] = {}
    for rel, (size, sha) in models.HI3D_FILES.items():
        folder, path = rel.split("/", 1)
        repo, revision = models.HI3D_SOURCES_PINNED[folder]
        big = path.endswith(".safetensors")
        trees.setdefault(f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true", []).append(
            {"type": "file", "path": path, "size": size, **({"lfs": {"oid": sha}} if big else {"oid": "git"})})
    monkeypatch.setattr(pr, "_get_json", lambda url: trees[url])
    report = pr.Report()
    pr.check_hi3d_downloads(report)
    assert report.errors == 0
    trees[next(iter(trees))][-1]["size"] += 1
    report = pr.Report()
    pr.check_hi3d_downloads(report)
    assert report.errors == 1
