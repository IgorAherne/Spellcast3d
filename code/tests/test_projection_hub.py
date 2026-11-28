"""The Projection stage and the artist's own 3D models, hub side: settings, the prop schema, jobs, the
HTTP routes and the Instant Meshes launcher's bake task API.

Runs in the hub venv (no numpy, no GPU). Instant Meshes is played by fake_tool.py, which the real
ImeshAdapter talks to. The launcher's bake / import routes are tested in-process with a stand-in bake
worker and stand-ins for instant_meshes_brush, which only the imesh venv has.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import json
import os
import re
import struct
import sys
import time
import types
import zipfile
import zlib
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spellcast import paths
from spellcast import projects as projects_mod
from spellcast import server as server_mod
from spellcast.adapters import build_adapters, imesh, projection
from spellcast.adapters.base import LaunchSpec, StageError
from spellcast.adapters.imesh import ImeshAdapter
from spellcast.jobs import JobError, JobQueue
from spellcast.procman import ToolError
from spellcast.projects import (BAKED_AGAIN, DEFAULT_STAGE_SETTINGS, EDITABLE_STAGES, ProjectError, ProjectStore,
                                save_model_files)
from spellcast.server import BEING_BAKED, create_app
from test_core import FakeAdapter, make_settings, manager, pid_alive

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Spellcast3D is Windows-only")

FAKE_TOOL = Path(__file__).resolve().parent / "fake_tool.py"
PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 64
FINISHED = {"done", "error", "cancelled"}


class FakeImesh(ImeshAdapter):
    """The real Instant Meshes adapter, pointed at fake_tool.py instead of the imesh venv."""

    def __init__(self, cwd: Path, *args: str) -> None:
        self.cwd, self.args = cwd, list(args)

    def install_status(self) -> dict[str, Any]:
        return {"status": "installed", "reason": ""}

    def launch_spec(self, port: int, settings: dict[str, Any]) -> LaunchSpec:
        return LaunchSpec(argv=[sys.executable, str(FAKE_TOOL), "--port", str(port), *self.args],
                          cwd=self.cwd, env=paths.clean_env(None), ready_timeout=30.0)


def ready_to_bake(store: ProjectStore, project: str, prop_id: str, low_poly: str = "v 0 0 0\n") -> Path:
    """Give a prop a high-poly GLB and a low-poly OBJ with UVs -> its folder."""
    prop_dir = store.prop_dir(project, prop_id)
    (prop_dir / "mesh").mkdir(parents=True, exist_ok=True)
    (prop_dir / "mesh" / "mesh.glb").write_bytes(b"glTF")
    (prop_dir / "retopo").mkdir(exist_ok=True)
    (prop_dir / "retopo" / "retopo.obj").write_text(low_poly, encoding="utf-8")
    store.apply_stage_patch(project, prop_id, "mesh", {"glb": "mesh/mesh.glb"})
    store.apply_stage_patch(project, prop_id, "retopo", {"obj": "retopo/retopo.obj", "stats": {"has_uv": True}})
    return prop_dir


class FakeConverter:
    """Stands in for exports.run_export_worker (the hub venv cannot run export_worker.py): writes the files
    the worker would, and remembers each call as (source GLB, formats, name) - and the placements it was given, one
    per call (None: the model as it is)."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, list[str], str]] = []
        self.placements: list[dict[str, Any] | None] = []

    def __call__(self, src: Path, outputs: dict[str, Path], name: str,
                 placement: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((Path(src), sorted(outputs), name))
        self.placements.append(placement)
        result: dict[str, Any] = {"warnings": []}
        for fmt, folder in outputs.items():
            folder.mkdir(parents=True, exist_ok=True)
            if fmt == "glb":
                (folder / f"{name}.glb").write_bytes(b"placed " + Path(src).read_bytes())
                result["glb"] = {"files": [f"{name}.glb"]}
            elif fmt == "fbx":
                (folder / f"{name}.fbx").write_bytes(b"Kaydara FBX Binary " + Path(src).read_bytes())
                result["fbx"] = {"files": [f"{name}.fbx"]}
            else:
                (folder / f"{name}.obj").write_text(f"mtllib {name}.mtl\nv 0 0 0\n", encoding="utf-8")
                (folder / f"{name}.mtl").write_text(f"newmtl m\nmap_Kd {name}_basecolor.png\n", encoding="utf-8")
                (folder / f"{name}_basecolor.png").write_bytes(PNG)
                result["obj"] = {"files": [f"{name}.obj", f"{name}.mtl", f"{name}_basecolor.png"]}
        return result


class FakeRepainter:
    """Stands in for exports.run_repaint_worker: remembers each call as (bake folder, raw file, its bytes, width,
    height, where the bake's own map is to be kept or None) and runs `during` (when set) while it works."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, bytes, int, int, Path | None]] = []
        self.during: Any = None

    def __call__(self, bake_dir: Path, rgba: Path, width: int, height: int,
                 unpainted: Path | None = None) -> dict[str, Any]:
        self.calls.append((Path(bake_dir), Path(rgba), Path(rgba).read_bytes(), width, height, unpainted))
        if self.during is not None:
            self.during()
        return {"files": ["maps/basecolor.png", "projection.glb"]}


def tiny_png(width: int, height: int) -> bytes:
    """A black RGB PNG, written without Pillow."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\0" + bytes(3 * width) for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def give_bake(store: ProjectStore, project: str, prop_id: str, number: int = 1) -> str:
    """Give a prop the files of a bake (a 4 x 2 base colour map) and its projection block -> its GLB's path."""
    folder = f"projection/bake_{number:03d}"
    bake_dir = store.prop_dir(project, prop_id) / folder
    (bake_dir / "maps").mkdir(parents=True)
    (bake_dir / "projection.glb").write_bytes(b"glTF-baked")
    (bake_dir / "maps" / "basecolor.png").write_bytes(tiny_png(4, 2))
    store.apply_stage_patch(project, prop_id, "projection", {
        "glb": f"{folder}/projection.glb", "stats": {},
        "maps": [{"id": "basecolor", "label": "Base colour", "path": f"{folder}/maps/basecolor.png", "colorspace": "srgb"}]})
    return f"{folder}/projection.glb"


def make_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def wait_for(read: Any, done: Any, timeout: float = 30.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = read()
        if done(value):
            return value
        time.sleep(0.05)
    raise AssertionError(f"timed out; last value: {value}")


# ---- settings and stage wiring ---------------------------------------------------------------------

def test_projection_is_a_second_stage_of_instant_meshes() -> None:
    adapters = build_adapters()
    assert paths.STAGE_TOOLS["projection"] == (paths.IMESH,) and paths.IMESH in paths.STAGE_TOOLS["retopo"]
    assert adapters[paths.IMESH].stages == ("retopo", "projection")
    assert set(EDITABLE_STAGES) == {stage for stage, tools in paths.STAGE_TOOLS.items() if tools}


def test_bake_settings_are_checked_in_artist_words() -> None:
    assert projection.bake_settings({}) == {
        "texture_size": 2048, "quality": 2, "ray_distance": None, "hard_edge_angle": 60.0, "tangent_basis": "mikk",
        "normal_y": "opengl", "ao": True, "height": False, "repack_uvs": True, "export_obj": True, "export_fbx": True}
    loose = projection.bake_settings({"texture_size": "1024", "quality": 3.0, "ray_distance": "2.5",
                                      "hard_edge_angle": -1, "ui_only": "dropped"})
    assert (loose["texture_size"], loose["quality"], loose["ray_distance"], loose["hard_edge_angle"]) == (1024, 3, 2.5, -1)
    assert "ui_only" not in loose
    for bad, words in (({"texture_size": 3000}, "Texture size must be 512, 1024, 2048 or 4096 pixels."),
                       ({"quality": 4}, "Quality must be Fast, Good or Best."),
                       ({"ray_distance": 50}, "between 0.05 % and 20 %"),
                       ({"ray_distance": "far"}, "Ray distance must be a number"),
                       ({"hard_edge_angle": 200}, "Hard edge angle must be between 0° and 180°"),
                       ({"tangent_basis": "xnormal"}, "MikkTSpace"),
                       ({"normal_y": "up"}, "OpenGL (Blender, Unity, Godot, glTF) or DirectX"),
                       ({"ao": "yes"}, '"Ambient occlusion" must be on or off.'),
                       ({"export_fbx": 1}, '"Also write FBX" must be on or off.'),
                       ({"texture_size": True}, "Texture size must be a number."),
                       ({"quality": float("nan")}, "Quality must be a number")):
        with pytest.raises(StageError, match=re.escape(words)):
            projection.bake_settings(bad)


# ---- the prop schema ---------------------------------------------------------------------------------

def test_props_get_a_projection_block_that_goes_stale_with_its_inputs(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    prop = store.add_prop("P", "Barrel")
    pid = prop["id"]
    assert prop["projection"] == {"settings": DEFAULT_STAGE_SETTINGS["projection"], "glb": None, "fbx": None,
                                  "obj": None, "maps": [], "source_mesh": None, "source_retopo": None,
                                  "stale": False, "stats": None}
    assert prop["mesh"]["imported"] is None

    def stale_after(stage: str, patch: dict[str, Any]) -> bool:
        return store.apply_stage_patch("P", pid, stage, patch)["projection"]["stale"]

    assert stale_after("retopo", {"obj": "retopo/retopo.obj"}) is False          # nothing baked yet
    assert stale_after("projection", {"glb": "projection/bake_001/projection.glb"}) is False
    assert stale_after("retopo", {"session_id": "s1", "session_source": "mesh/mesh.glb"}) is False   # no new result
    assert stale_after("retopo", {"obj": "retopo/retopo.obj"}) is True
    assert stale_after("projection", {"glb": "projection/bake_002/projection.glb"}) is False
    assert stale_after("mesh", {"glb": "mesh/mesh.glb"}) is True
    # A new 3D model also makes the retopo made from the one before it out of date, until a new retopo.
    assert store.get_prop("P", pid)["retopo"]["stale"] is True
    assert store.apply_stage_patch("P", pid, "retopo", {"session_id": "s2"})["retopo"]["stale"] is True
    assert store.apply_stage_patch("P", pid, "retopo", {"obj": "retopo/retopo.obj"})["retopo"]["stale"] is False
    assert stale_after("projection", {"glb": "projection/bake_003/projection.glb", "stale": True}) is True

    # A mesh that was not imported (Pixal3D made it) drops the details of an earlier import.
    store.apply_stage_patch("P", pid, "mesh", {"glb": "mesh/imported_001.glb", "imported": {"materials": []}})
    assert store.get_prop("P", pid)["mesh"]["imported"] == {"materials": []}
    assert store.apply_stage_patch("P", pid, "mesh", {"glb": "mesh/mesh.glb"})["mesh"]["imported"] is None

    # A project saved before the Projection stage existed gets the block (and keeps its results).
    file = tmp_path / "projects" / "P" / "project.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    data["props"][0]["projection"] = {}
    file.write_text(json.dumps(data), encoding="utf-8")
    old = store.get_prop("P", pid)
    assert old["projection"]["settings"]["texture_size"] == 2048 and old["projection"]["maps"] == []
    assert old["mesh"]["glb"] == "mesh/mesh.glb"


def test_model_files_are_saved_inside_their_folder(tmp_path: Path) -> None:
    folder = tmp_path / "import_001"
    folder.mkdir()
    archive = make_zip({"Chair/chair.fbx": b"fbx", "Chair/textures/chair_N.png": PNG, "__MACOSX/Chair/._chair_N.png": b"x",
                        "Chair/readme.txt": b"hi", "Chair/tex/CON.png": PNG, "Chair/a:b?.png": PNG,
                        "C:/Users/me/chair_D.png": PNG})
    saved = save_model_files(folder, [("Chair.zip", io.BytesIO(archive))])
    assert saved.files == ["Chair/chair.fbx", "Chair/textures/chair_N.png", "Chair/tex/_CON.png", "Chair/a_b_.png",
                           "Users/me/chair_D.png"]
    assert saved.skipped == ["__MACOSX/Chair/._chair_N.png", "Chair/readme.txt"]
    assert saved.model == "Chair/chair.fbx"
    assert (folder / "Chair" / "textures" / "chair_N.png").read_bytes() == PNG
    # Files outside the model's folder are linked into it by name, where Instant Meshes looks for maps.
    assert (folder / "Chair" / "chair_D.png").read_bytes() == PNG
    assert not (folder / "Chair" / "chair_N.png").exists()     # already inside the model's folder
    pack = save_model_files(tmp_path / "pack", [("Pack.zip", io.BytesIO(make_zip({
        "Pack/Meshes/crate.fbx": b"fbx", "Pack/Textures/crate_D.png": PNG})))])
    assert pack.model == "Pack/Meshes/crate.fbx" and pack.files == ["Pack/Meshes/crate.fbx", "Pack/Textures/crate_D.png"]
    assert (tmp_path / "pack" / "Pack" / "Meshes" / "crate_D.png").read_bytes() == PNG

    # The first file chosen wins when it is a model; otherwise FBX > GLB > glTF > OBJ, then the fewest folders.
    first = save_model_files(tmp_path / "a", [("chair.obj", io.BytesIO(b"v")), ("chair.mtl", io.BytesIO(b"")),
                                             ("textures/chair_d.png", io.BytesIO(PNG)), ("other.fbx", io.BytesIO(b""))])
    assert first.model == "chair.obj" and "textures/chair_d.png" in first.files
    ranked = save_model_files(tmp_path / "b", [("t.png", io.BytesIO(PNG)), ("b.glb", io.BytesIO(b"")),
                                              ("sub/a.fbx", io.BytesIO(b"")), ("c.fbx", io.BytesIO(b""))])
    assert ranked.model == "c.fbx"
    assert save_model_files(tmp_path / "c", [("t.png", io.BytesIO(PNG))]).model is None
    long_name = "x" * 300 + ".png"                   # cut to 100 characters, keeping the suffix
    assert save_model_files(tmp_path / "d", [(f"deep/{long_name}", io.BytesIO(PNG))]).files == ["deep/" + "x" * 96 + ".png"]
    too_deep = "/".join(["d" * 99] * 3) + "/wood.png"   # a path too long for Windows keeps its bare name
    assert save_model_files(tmp_path / "e", [(too_deep, io.BytesIO(PNG))]).files == ["wood.png"]


def test_model_files_that_climb_out_or_are_too_big_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for index, name in enumerate(("../evil.png", "a/../../evil.png", "..\\evil.png")):
        with pytest.raises(ProjectError, match="points outside its folder"):
            save_model_files(tmp_path / f"f{index}", [(name, io.BytesIO(PNG))])
    with pytest.raises(ProjectError, match="points outside its folder"):
        save_model_files(tmp_path / "zip", [("m.zip", io.BytesIO(make_zip({"ok.obj": b"v", "../../escaped.png": PNG})))])
    assert not list(tmp_path.rglob("evil.png")) and not list(tmp_path.rglob("escaped.png"))

    with pytest.raises(ProjectError, match="not a .zip file that can be opened"):
        save_model_files(tmp_path / "bad", [("m.zip", io.BytesIO(b"PK\x03\x04 damaged"))])
    locked = bytearray(make_zip({"model.obj": b"v"}))
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):     # the "encrypted" bit
        locked[locked.index(signature) + flag_offset] |= 0x1
    with pytest.raises(ProjectError, match="password-protected"):
        save_model_files(tmp_path / "locked", [("m.zip", io.BytesIO(bytes(locked)))])

    monkeypatch.setattr(projects_mod, "MAX_MODEL_UPLOAD_BYTES", 1000)
    with pytest.raises(ProjectError, match="too big together") as e:
        save_model_files(tmp_path / "big", [("model.obj", io.BytesIO(b"v" * 2000))])
    assert e.value.status_code == 413
    bomb = make_zip({"model.obj": b"v" * 100_000})
    assert len(bomb) < 1000
    with pytest.raises(ProjectError, match="too big together"):
        save_model_files(tmp_path / "bomb", [("m.zip", io.BytesIO(bomb))])


def test_replaced_imports_are_removed(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    pid = store.add_prop("P")["id"]
    folder, glb = store.new_model_import("P", pid)
    assert (folder.name, glb.name) == ("import_001", "imported_001.glb")
    glb.write_bytes(b"1")
    folder2, glb2 = store.new_model_import("P", pid)
    assert (folder2.name, glb2.name) == ("import_002", "imported_002.glb")
    (folder2 / "sub").mkdir()
    (folder2 / "sub" / "model.fbx").write_bytes(b"fbx")
    glb2.write_bytes(b"2")
    assert store.next_imported_glb("P", pid).name == "imported_003.glb"
    (glb.parent / "mesh.glb").write_bytes(b"pixal3d")
    store.remove_other_imports("P", pid, ("mesh/imported_002.glb", "mesh/import_002/sub/model.fbx"))
    assert sorted(p.name for p in glb.parent.iterdir()) == ["import_002", "imported_002.glb", "mesh.glb"]


# ---- jobs (the real adapter against fake_tool.py) ---------------------------------------------------------

def _project(tmp_path: Path, names: tuple[str, ...]) -> tuple[ProjectStore, list[str]]:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("Demo")
    return store, [store.add_prop("Demo", name)["id"] for name in names]


async def _follow_job(queue: JobQueue, job_id: str, pm: Any, timeout: float = 30.0) -> tuple[dict[str, Any], set[str], set[str]]:
    """Wait for a job -> (its end state, the progress messages seen, the tool states seen)."""
    messages: set[str] = set()
    states: set[str] = set()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = queue.get(job_id)
        messages.add(job["progress"]["message"])
        states.add(pm.status(paths.IMESH)["state"])
        if job["status"] in FINISHED:
            return job, messages, states
        await asyncio.sleep(0.05)
    raise AssertionError(f"job stuck: {queue.get(job_id)}")


def test_a_projection_job_bakes_on_instant_meshes_and_saves_the_result(tmp_path: Path) -> None:
    store, ids = _project(tmp_path, ("Barrel", "Crate"))
    first_dir = ready_to_bake(store, "Demo", ids[0])
    ready_to_bake(store, "Demo", ids[1])
    store.patch_prop("Demo", ids[1], {"projection": {"settings": {"texture_size": 1024, "export_obj": False}}})

    async def main() -> None:
        adapters = {paths.IMESH: FakeImesh(tmp_path, "--bake-seconds", "1.5")}
        async with manager(tmp_path, adapters, {"imesh": {"threads": 6}}) as pm:
            queue = JobQueue(pm, store, pm.settings, adapters, pm.client)
            queue.start()
            job = await queue.submit("projection", "Demo", ids)
            final, messages, states = await _follow_job(queue, job["id"], pm)
            assert final["status"] == "done", final
            assert any(m.startswith("Fake baking") for m in messages)          # the bake's own progress
            assert "busy" in states and pm.status(paths.IMESH)["state"] == "ready"

            baked = store.get_prop("Demo", ids[0])["projection"]
            assert baked["glb"] == "projection/bake_001/projection.glb"
            assert baked["fbx"] == "projection/bake_001/projection.fbx"
            assert baked["obj"] == "projection/bake_001/export/projection.obj"
            assert baked["maps"][0] == {"id": "basecolor", "label": "Base colour", "colorspace": "srgb",
                                        "path": "projection/bake_001/maps/basecolor.png"}
            assert [m["id"] for m in baked["maps"]] == ["basecolor", "normal", "orm"]
            assert (baked["source_mesh"], baked["source_retopo"]) == ("mesh/mesh.glb", "retopo/retopo.obj")
            assert baked["stale"] is False and baked["stats"]["resolution"] == 2048
            assert baked["settings"] == DEFAULT_STAGE_SETTINGS["projection"]
            second = store.get_prop("Demo", ids[1])["projection"]
            assert second["obj"] is None and second["stats"]["resolution"] == 1024

            asked = (await pm.client.get(f"{pm.get(paths.IMESH).base_url}/fake/bakes")).json()
            request = asked[0]["request"]
            assert request == {"high": str((first_dir / "mesh" / "mesh.glb").resolve()),
                               "low": str((first_dir / "retopo" / "retopo.obj").resolve()),
                               "out_root": str((first_dir / "projection").resolve()),
                               "settings": projection.bake_settings({}), "threads": 6}
            await queue.close()

    asyncio.run(main())


def test_projection_problems_reach_the_card_and_cancel_stops_the_bake(tmp_path: Path) -> None:
    store, ids = _project(tmp_path, ("No retopo", "No UVs", "Inside out", "Slow", "Bad size"))
    (ready_to_bake(store, "Demo", ids[0]) / "retopo" / "retopo.obj").unlink()
    ready_to_bake(store, "Demo", ids[1])
    store.apply_stage_patch("Demo", ids[1], "retopo", {"stats": {"has_uv": False}})
    ready_to_bake(store, "Demo", ids[2], low_poly="FAKE_ERROR:The low poly is inside out.")
    ready_to_bake(store, "Demo", ids[3])
    store.patch_prop("Demo", ids[4], {"projection": {"settings": {"texture_size": 3000}}})

    async def main() -> None:
        adapters = {paths.IMESH: FakeImesh(tmp_path, "--bake-seconds", "30")}
        async with manager(tmp_path, adapters) as pm:
            queue = JobQueue(pm, store, pm.settings, adapters, pm.client)
            with pytest.raises(JobError) as e:
                await queue.submit("projection", "Demo", ids[4:])
            assert e.value.status_code == 400
            assert str(e.value) == "Bad size: Texture size must be 512, 1024, 2048 or 4096 pixels."
            with pytest.raises(JobError, match="Quality must be Fast, Good or Best"):
                await queue.submit("projection", "Demo", ids[:1], {"quality": 9})
            queue.start()

            adapters[paths.IMESH].args = ["--bake-seconds", "1"]
            failed, _, _ = await _follow_job(queue, (await queue.submit("projection", "Demo", ids[:3]))["id"], pm)
            assert failed["status"] == "error"
            assert "no low-poly model yet" in failed["errors"][ids[0]]
            assert "has no UVs" in failed["errors"][ids[1]]
            assert failed["errors"][ids[2]] == "The low poly is inside out."

            await pm.stop(paths.IMESH)
            adapters[paths.IMESH].args = ["--bake-seconds", "30"]
            slow = await queue.submit("projection", "Demo", ids[3:4])
            while not queue.get(slow["id"])["progress"]["message"].startswith("Fake baking"):
                await asyncio.sleep(0.05)
            queue.cancel(slow["id"])
            final, _, _ = await _follow_job(queue, slow["id"], pm)
            assert final["status"] == "cancelled" and final["errors"] == {}
            asked = (await pm.client.get(f"{pm.get(paths.IMESH).base_url}/fake/bakes")).json()
            assert asked[-1]["state"] == "cancelled"
            assert store.get_prop("Demo", ids[3])["projection"]["glb"] is None
            await queue.close()

    asyncio.run(main())


# ---- HTTP API ----------------------------------------------------------------------------------------

@pytest.fixture
def hub(tmp_path: Path) -> Iterator[tuple[FastAPI, TestClient, FakeImesh]]:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>Spellcast3D</h1>", encoding="utf-8")
    fake = FakeImesh(tmp_path, "--bake-seconds", "1")
    adapters = {"forge": FakeAdapter("forge", "images", True, tmp_path),
                "pixal3d": FakeAdapter("pixal3d", "mesh", True, tmp_path), paths.IMESH: fake}
    app = create_app(adapters, settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=static, logs_dir=tmp_path / "logs", gpu_probe=lambda: None,
                     show_in_explorer=lambda path, select: True, converter=FakeConverter(),
                     repainter=FakeRepainter())
    with TestClient(app, base_url="http://127.0.0.1") as client:
        client.post("/api/projects", json={"name": "Demo"})
        yield app, client, fake


def _new_prop(client: TestClient, name: str) -> str:
    return client.post("/api/projects/Demo/props", json={"name": name}).json()["id"]


def _job(client: TestClient, body: dict[str, Any]) -> dict[str, Any]:
    job = client.post("/api/jobs", json=body).json()
    return wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json(), lambda j: j["status"] in FINISHED)


def _tool_url(client: TestClient) -> str:
    return f"http://127.0.0.1:{client.get('/api/status').json()['tools'][paths.IMESH]['port']}"


def test_api_projection_job_saved_files_and_tool_stop(hub: Any, tmp_path: Path) -> None:
    app, client, fake = hub
    store = app.state.hub.store
    pid = _new_prop(client, "Wooden barrel")
    ready_to_bake(store, "Demo", pid)
    bad = client.post("/api/jobs", json={"stage": "projection", "project": "Demo", "prop_ids": [pid],
                                         "params": {"texture_size": 3000}})
    assert bad.status_code == 400
    assert bad.json()["detail"] == "Wooden barrel: Texture size must be 512, 1024, 2048 or 4096 pixels."
    url = f"/api/projects/Demo/props/{pid}/export"
    unbaked = client.post(url, json={"stage": "projection", "format": "obj"})
    assert unbaked.status_code == 404 and "not been baked" in unbaked.json()["detail"]

    done = _job(client, {"stage": "projection", "project": "Demo", "prop_ids": [pid]})
    assert done["status"] == "done", done
    baked = client.get("/api/projects/Demo").json()["props"][0]["projection"]
    assert baked["glb"] == "projection/bake_001/projection.glb" and baked["stale"] is False
    assert client.get(f"/files/Demo/props/{pid}/{baked['fbx']}").content == b"Kaydara FBX Binary"

    # The bake's files, flat in the prop's folder exports/Demo/Wooden_barrel: the MTL and the OBJ's mtllib name
    # the flat map files.
    folder = tmp_path / "exports" / "Demo" / "Wooden_barrel"
    obj = client.post(url, json={"stage": "projection", "format": "obj"}).json()
    assert obj == {"folder": "exports\\Demo\\Wooden_barrel", "folder_path": str(folder), "shown": "Wooden_barrel.obj",
                   "files": ["Wooden_barrel.obj", "Wooden_barrel.mtl", "Wooden_barrel_basecolor.png",
                             "Wooden_barrel_normal.png", "Wooden_barrel_orm.png"], "opened": False, "warnings": []}
    assert sorted(p.name for p in folder.iterdir()) == sorted(obj["files"])
    assert (folder / "Wooden_barrel.mtl").read_text(encoding="utf-8") == \
        "newmtl projection\nmap_Kd Wooden_barrel_basecolor.png\n"
    assert (folder / "Wooden_barrel.obj").read_text(encoding="utf-8") == "mtllib Wooden_barrel.mtl\nv 0 0 0\n"
    assert (folder / "Wooden_barrel_normal.png").read_bytes().endswith(b"normal")
    fbx = client.post(url, json={"stage": "projection", "format": "fbx"}).json()
    assert fbx["files"] == ["Wooden_barrel.fbx"] and (folder / "Wooden_barrel.fbx").read_bytes() == b"Kaydara FBX Binary"
    assert client.post(url, json={"stage": "projection", "format": "glb"}).json()["shown"] == "Wooden_barrel.glb"
    assert (folder / "Wooden_barrel.glb").read_bytes() == b"glTF-baked"
    for body in ({"stage": "retopo", "format": "obj"}, {"stage": "projection", "format": "zip"}, {}):
        assert client.post(url, json=body).status_code == 400
    assert client.get(f"/api/projects/Demo/props/{pid}/projection/obj.zip").status_code == 404     # no zips any more
    assert client.get("/api/projects/Demo/export.zip").status_code == 404

    # A new low poly makes the bake stale.
    (store.prop_dir("Demo", pid) / "retopo" / "retopo.obj").write_text("v 1 1 1\n", encoding="utf-8")
    store.apply_stage_patch("Demo", pid, "retopo", {"obj": "retopo/retopo.obj"})
    assert client.get("/api/projects/Demo").json()["props"][0]["projection"]["stale"] is True

    # Stopping Instant Meshes cancels its projection job too.
    fake.args = ["--bake-seconds", "30"]
    client.post(f"/api/tools/{paths.IMESH}/stop")
    wait_for(lambda: client.get("/api/status").json()["tools"][paths.IMESH]["state"], lambda s: s == "stopped")
    job = client.post("/api/jobs", json={"stage": "projection", "project": "Demo", "prop_ids": [pid]}).json()
    wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json()["progress"]["message"],
             lambda m: m.startswith("Fake baking"))
    assert client.post("/api/tools/nope/stop").status_code == 404
    client.post(f"/api/tools/{paths.IMESH}/stop")
    final = wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json(), lambda j: j["status"] in FINISHED)
    assert final["status"] == "cancelled"
    assert client.get("/api/projects/Demo").json()["props"][0]["projection"]["glb"] == baked["glb"]


def _paint(client: TestClient, pid: str, body: bytes, glb: str, width: int = 4, height: int = 2) -> httpx.Response:
    return client.put(f"/api/projects/Demo/props/{pid}/projection/paint", content=body,
                      params={"width": width, "height": height, "glb": glb},
                      headers={"content-type": "application/octet-stream"})


def test_api_paint_saves_into_the_bake(hub: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    app, client, _ = hub
    state = app.state.hub
    store, repaint = state.store, state.repaint
    pid = _new_prop(client, "Barrel")
    prop_dir = store.prop_dir("Demo", pid)
    pixels = bytes(range(32))                                   # 4 x 2 RGBA
    unbaked = _paint(client, pid, pixels, "projection/bake_001/projection.glb")
    assert unbaked.status_code == 404 and "not been baked" in unbaked.json()["detail"]
    assert _paint(client, "p_000000", pixels, "projection/bake_001/projection.glb").status_code == 404
    glb = give_bake(store, "Demo", pid)

    old = _paint(client, pid, pixels, "projection/bake_000/projection.glb")
    assert old.status_code == 409 and old.json()["detail"] == BAKED_AGAIN
    wrong = _paint(client, pid, pixels, glb, width=2, height=4)
    assert wrong.status_code == 422 and "is 2 x 4 pixels, but this bake's base colour map is 4 x 2" in \
        wrong.json()["detail"]
    short = _paint(client, pid, pixels[:-1], glb)
    assert short.status_code == 422 and "arrived incomplete (31 of 32 bytes)" in short.json()["detail"]
    assert client.put(f"/api/projects/Demo/props/{pid}/projection/paint", content=pixels).status_code == 422
    monkeypatch.setattr(server_mod, "MAX_PAINT_BYTES", 16)
    big = _paint(client, pid, pixels, glb)
    assert big.status_code == 413 and "too big" in big.json()["detail"]
    monkeypatch.undo()

    def not_installed(*args: Any, **kwargs: Any) -> None:
        raise ToolError("Instant Meshes is not installed yet.")

    monkeypatch.setattr(state.procman, "check_available", not_installed)
    missing = _paint(client, pid, pixels, glb)
    assert missing.status_code == 409 and missing.json()["detail"] == \
        "Saving the painting needs the retopology tool. Instant Meshes is not installed yet."
    monkeypatch.undo()
    assert repaint.calls == []

    # A stale bake with a thumbnail: the painting is saved, the bake stays stale, the thumbnail goes.
    assert client.put(f"/api/projects/Demo/props/{pid}/thumbs/projection", content=PNG).status_code == 200
    (prop_dir / "retopo").mkdir()
    (prop_dir / "retopo" / "retopo.obj").write_text("v 0 0 0\n", encoding="utf-8")
    store.apply_stage_patch("Demo", pid, "retopo", {"obj": "retopo/retopo.obj"})
    busy: list[set[str]] = []
    repaint.during = lambda: busy.append(state.busy_projects())
    answer = _paint(client, pid, pixels, glb)
    assert answer.status_code == 200, answer.text
    prop = answer.json()
    assert isinstance(prop["projection"]["painted"], int) and prop["projection"]["stale"] is True
    assert prop["projection"]["glb"] == glb and "projection" not in prop["thumbs"]
    bake_dir, raw, data, width, height, unpainted = repaint.calls[0]
    assert bake_dir == (prop_dir / "projection" / "bake_001").resolve()
    assert raw.parent == prop_dir and raw.name.startswith(".paint-") and (data, width, height) == (pixels, 4, 2)
    assert unpainted == bake_dir / "maps" / projects_mod.UNPAINTED_BASECOLOR    # the first painting keeps the bake's map
    assert busy == [{"Demo"}]                                   # the project cannot be renamed meanwhile
    assert not list(prop_dir.glob(".paint-*"))
    saved = client.get("/api/projects/Demo").json()["props"][0]["projection"]
    assert saved["painted"] == prop["projection"]["painted"] and saved["stale"] is True
    again = _paint(client, pid, pixels, glb).json()["projection"]["painted"]
    assert again > prop["projection"]["painted"]
    assert repaint.calls[-1][5] is None                         # a painted map is never kept as the bake's own

    # The worker's own words; its raw file is gone and the stamp stays.
    def fails() -> None:
        raise ProjectError("The bake's base colour map is missing.", 422)

    repaint.during = fails
    failed = _paint(client, pid, pixels, glb)
    assert failed.status_code == 422 and failed.json()["detail"] == "The bake's base colour map is missing."
    assert not list(prop_dir.glob(".paint-*")) and store.get_prop("Demo", pid)["projection"]["painted"] == again

    # Baked again while the painting was being saved: refused, and the new bake is unpainted.
    repaint.during = lambda: give_bake(store, "Demo", pid, number=2)
    late = _paint(client, pid, pixels, glb)
    assert late.status_code == 409 and late.json()["detail"] == BAKED_AGAIN
    baked = store.get_prop("Demo", pid)["projection"]
    assert baked["glb"] == "projection/bake_002/projection.glb" and "painted" not in baked
    # ... so its first painting keeps its own map in turn.
    repaint.during = None
    assert _paint(client, pid, pixels, baked["glb"]).status_code == 200
    assert repaint.calls[-1][5] == (prop_dir / "projection" / "bake_002" / "maps" / "basecolor.unpainted.png").resolve()


def test_api_paint_is_refused_while_the_prop_is_baked(hub: Any) -> None:
    app, client, fake = hub
    store = app.state.hub.store
    pid = _new_prop(client, "Crate")
    ready_to_bake(store, "Demo", pid)
    glb = give_bake(store, "Demo", pid)
    fake.args = ["--bake-seconds", "30"]
    job = client.post("/api/jobs", json={"stage": "projection", "project": "Demo", "prop_ids": [pid]}).json()
    refused = _paint(client, pid, bytes(32), glb)
    assert refused.status_code == 409 and refused.json()["detail"] == BEING_BAKED
    other = _new_prop(client, "Lamp")                           # another prop's bake is no reason to wait
    assert _paint(client, other, bytes(32), give_bake(store, "Demo", other)).status_code == 200
    client.post(f"/api/jobs/{job['id']}/cancel")
    wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json()["status"], lambda status: status in FINISHED)
    assert _paint(client, pid, bytes(32), glb).status_code == 200
    assert [call[2] for call in app.state.hub.repaint.calls] == [bytes(32), bytes(32)]


def test_api_mesh_import_reimport_and_cleanup(hub: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    app, client, _ = hub
    store = app.state.hub.store
    pid = _new_prop(client, "Chair")
    prop_dir = store.prop_dir("Demo", pid)
    url = f"/api/projects/Demo/props/{pid}/mesh"
    store.apply_stage_patch("Demo", pid, "projection", {"glb": "projection/bake_001/projection.glb"})

    files = [("files", ("chair.obj", b"mtllib chair.mtl\nv 0 0 0\n", "text/plain")),
             ("files", ("chair.mtl", b"map_Kd textures/chair_d.png\n", "text/plain")),
             ("files", ("textures/chair_d.png", PNG, "image/png")),
             ("files", ("notes.txt", b"hello", "text/plain"))]
    answer = client.post(f"{url}/import", files=files)
    assert answer.status_code == 200, answer.text
    prop = answer.json()
    mesh = prop["mesh"]
    assert mesh["glb"] == "mesh/imported_001.glb" and mesh["source"] == "mesh/import_001/chair.obj"
    assert mesh["imported"]["materials"][0]["maps"][0]["file"] == "chair_d.png"
    assert mesh["imported"]["skipped"] == ["notes.txt"] and mesh["imported"]["overrides"] == {}
    assert mesh["imported"]["warnings"] == [] and mesh["stats"] == {"seconds": 0.2, "vertices": 8, "faces": 12}
    assert prop["projection"]["stale"] is True
    assert (prop_dir / "mesh" / "import_001" / "textures" / "chair_d.png").read_bytes() == PNG
    assert (prop_dir / "mesh" / "imported_001.glb").read_bytes() == b"glTF-imported"
    asked = httpx.get(f"{_tool_url(client)}/fake/imports", trust_env=False).json()
    assert asked[0] == {"src": str((prop_dir / "mesh" / "import_001" / "chair.obj").resolve()),
                        "dst": str((prop_dir / "mesh" / "imported_001.glb").resolve()), "overrides": {}}

    overrides = {"chair_d.png": {"role": "normal", "flip_green": True, "invert": None}}
    again = client.post(f"{url}/reimport", json={"overrides": overrides})
    assert again.status_code == 200, again.text
    mesh = again.json()["mesh"]
    assert mesh["glb"] == "mesh/imported_002.glb" and mesh["source"] == "mesh/import_001/chair.obj"
    assert mesh["imported"]["overrides"] == {"chair_d.png": {"role": "normal", "flip_green": True}}
    assert mesh["imported"]["skipped"] == ["notes.txt"]
    assert not (prop_dir / "mesh" / "imported_001.glb").exists() and (prop_dir / "mesh" / "import_001").is_dir()
    asked = httpx.get(f"{_tool_url(client)}/fake/imports", trust_env=False).json()
    assert asked[1]["overrides"] == {"chair_d.png": {"role": "normal", "flip_green": True}}
    for bad in ({"overrides": [1]}, {"overrides": {"x": {"role": 5}}}, {"overrides": {"x": {"colour": "red"}}},
                {"overrides": {"x": {"invert": "yes"}}}):
        assert client.post(f"{url}/reimport", json=bad).status_code == 400, bad

    # A .zip with the model in a folder replaces the first import; the replaced files go.
    archive = make_zip({"Lamp/lamp.fbx": b"fbx", "Lamp/lamp_BaseColor.png": PNG})
    zipped = client.post(f"{url}/import", files=[("files", ("lamp.zip", archive, "application/zip"))])
    assert zipped.status_code == 200, zipped.text
    assert zipped.json()["mesh"]["source"] == "mesh/import_003/Lamp/lamp.fbx"
    assert sorted(p.name for p in (prop_dir / "mesh").iterdir()) == ["import_003", "imported_003.glb"]

    slip = client.post(f"{url}/import", files=[("files", ("evil.zip", make_zip({"m.obj": b"v", "../../../evil.png": PNG})))])
    assert slip.status_code == 400 and "points outside its folder" in slip.json()["detail"]
    no_model = client.post(f"{url}/import", files=[("files", ("wood.png", PNG, "image/png"))])
    assert no_model.status_code == 400 and "no 3D model among these files" in no_model.json()["detail"]
    assert client.post(f"{url}/import", data={"x": "1"}).status_code == 400
    broken = client.post(f"{url}/import", files=[("files", ("bad.fbx", b"FAKE_ERROR:This FBX has no mesh in it."))])
    assert broken.status_code == 422 and broken.json()["detail"] == "This FBX has no mesh in it."
    assert sorted(p.name for p in (prop_dir / "mesh").iterdir()) == ["import_003", "imported_003.glb"]
    assert not list(Path(prop_dir).parents[3].rglob("evil.png"))
    assert client.get("/api/projects/Demo").json()["props"][0]["mesh"]["source"] == "mesh/import_003/Lamp/lamp.fbx"

    monkeypatch.setattr("spellcast.server.MAX_MODEL_UPLOAD_BYTES", 10)
    monkeypatch.setattr("spellcast.server.UPLOAD_OVERHEAD", 10)
    too_big = client.post(f"{url}/import", files=[("files", ("m.obj", b"v" * 5000))])
    assert too_big.status_code == 413 and "too big together" in too_big.json()["detail"]

    generated = _new_prop(client, "Generated")
    store.apply_stage_patch("Demo", generated, "mesh", {"glb": "mesh/mesh.glb"})
    refused = client.post(f"/api/projects/Demo/props/{generated}/mesh/reimport", json={"overrides": {}})
    assert refused.status_code == 409 and "not your own imported model" in refused.json()["detail"]
    assert client.post("/api/projects/Demo/props/p_ffffff/mesh/import", files=files).status_code == 404


# ---- the launcher's bake task API (imesh_server.py, in-process) ---------------------------------------------

WORKER = r'''
import json, os, sys, time
from pathlib import Path

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
say = lambda **message: print(json.dumps(message), flush=True)
print("a line that is not JSON", flush=True)
if request["cmd"] == "import":
    mode = request["overrides"].get("mode", "ok")
    if mode == "hang":
        Path(request["overrides"]["pidfile"]).write_text(str(os.getpid()))
        time.sleep(60)
    if mode == "error":
        say(error="This FBX has no mesh in it.", kind="user")
        sys.exit(1)
    Path(request["dst"]).write_bytes(b"glTF-imported")
    say(result={"seconds": 0.1, "vertices": 3, "faces": 1, "materials": [], "warnings": ["EXR skipped"]})
    sys.exit(0)
mode = request["settings"].get("mode", "ok")
out = Path(request["out_dir"])
Path(request["high"]).with_name("worker.pid").write_text(str(os.getpid()))    # the real process, not the venv's redirector
for step in range(3):
    say(progress=step / 4, message=f"step {step}")
    time.sleep(0.05)
if mode == "slow":
    time.sleep(60)
if mode == "error":
    say(error="The low poly is inside out.", kind="user")
    sys.exit(1)
if mode == "crash":
    print("Traceback: boom", file=sys.stderr, flush=True)
    sys.exit(3)
(out / "maps").mkdir()
(out / "export").mkdir()
for name in ("projection.glb", "maps/basecolor.png", "export/projection.obj"):
    (out / name).write_bytes(b"x")
os.replace(out, request["final_dir"])
say(result={"glb": "projection.glb", "fbx": None, "obj": "export/projection.obj",
            "maps": [{"id": "basecolor", "label": "Base colour", "file": "maps/basecolor.png", "colorspace": "srgb"}],
            "stats": {"coverage": 0.9}} if mode != "missing" else {"glb": "nothing.glb"})
'''


@pytest.fixture
def launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Any, TestClient, Path, Any]]:
    """imesh_server imported with stand-ins for instant_meshes_brush
    -> (the module, a client of its bake routes, a prop folder, its BakeWorkers)."""
    package = types.ModuleType("instant_meshes_brush")
    session_names = ("BrushSession", "Extraction", "SessionClosedError", "SessionError", "SessionLimitError",
                     "SessionRegistry")
    members: dict[str, dict[str, Any]] = {
        "assets": {"SUFFIXES": frozenset({".glb", ".gltf", ".obj", ".fbx"}), "AssetError": RuntimeError},
        "server": {}, "uv": {},
        "session_manager": {**{name: type(name, (Exception,), {}) for name in session_names},
                            "config_from_mapping": dict}}
    for name, attributes in members.items():
        module = types.ModuleType(f"instant_meshes_brush.{name}")
        module.__dict__.update(attributes)
        setattr(package, name, module)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "instant_meshes_brush", package)
    monkeypatch.syspath_prepend(str(paths.LAUNCHERS))
    sys.modules.pop("imesh_server", None)
    launcher_module = importlib.import_module("imesh_server")
    try:
        projects = (tmp_path / "projects").resolve()
        monkeypatch.setattr(launcher_module, "PROJECTS_ROOT", projects)
        (tmp_path / "worker.py").write_text(WORKER, encoding="utf-8")
        workers = launcher_module.BakeWorkers(tmp_path / "worker.py")
        app = FastAPI()
        launcher_module.add_bake_routes(app, workers)
        prop_dir = projects / "Demo" / "props" / "p_000001"
        for rel in ("mesh/mesh.glb", "retopo/retopo.obj", "mesh/import_001/model.fbx"):
            (prop_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (prop_dir / rel).write_bytes(b"x")
        with TestClient(app) as client:
            yield launcher_module, client, prop_dir, workers
    finally:
        sys.modules.pop("imesh_server", None)


def _bake(client: TestClient, prop_dir: Path, mode: str, **changes: Any) -> Any:
    body = {"high": str(prop_dir / "mesh" / "mesh.glb"), "low": str(prop_dir / "retopo" / "retopo.obj"),
            "out_root": str(prop_dir / "projection"), "settings": {"mode": mode}, **changes}
    return client.post("/spellcast/bake", json=body)


def _bake_status(client: TestClient, task_id: str) -> dict[str, Any]:
    return wait_for(lambda: client.get(f"/spellcast/bake/{task_id}").json(), lambda s: s["state"] != "running")


def test_launcher_bakes_in_a_worker_and_keeps_the_two_newest_bakes(launcher: Any) -> None:
    module, client, prop_dir, _ = launcher
    out = prop_dir / "projection"
    (out / "bake_007.tmp").mkdir(parents=True)         # left behind by a killed Instant Meshes
    (out / "bake_001").mkdir()
    (out / "bake_002").mkdir()
    task = _bake(client, prop_dir, "ok").json()["task_id"]
    status = _bake_status(client, task)
    assert status["state"] == "done" and status["fraction"] == 1.0 and status["message"] == "Baked."
    folder = (out / "bake_008").resolve()
    assert status["result"] == {
        "dir": str(folder), "glb": str(folder / "projection.glb"), "fbx": None,
        "obj": str(folder / "export" / "projection.obj"), "stats": {"coverage": 0.9},
        "maps": [{"id": "basecolor", "label": "Base colour", "colorspace": "srgb", "path": str(folder / "maps" / "basecolor.png")}]}
    assert sorted(p.name for p in out.iterdir()) == ["bake_002", "bake_008"]
    request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    assert request["cmd"] == "bake" and request["threads"] == os.cpu_count()
    assert request["out_dir"] == str(out.resolve() / "bake_008.tmp") and request["final_dir"] == str(folder)
    assert _bake(client, prop_dir, "ok", threads=3).status_code == 200
    assert module.IMPORT_TIMEOUT_SECONDS < imesh.IMPORT_TIMEOUT.read      # the launcher answers first


def _slow_bake(client: TestClient, prop_dir: Path) -> tuple[str, list[int]]:
    """Start a bake that does not end by itself -> (task id, pids of the worker and its venv redirector)."""
    task = _bake(client, prop_dir, "slow").json()["task_id"]
    wait_for(lambda: client.get(f"/spellcast/bake/{task}").json()["message"], lambda m: m == "step 2")
    return task, [int((prop_dir / "mesh" / "worker.pid").read_text())]


def test_launcher_runs_one_bake_at_a_time_and_cancel_kills_it(launcher: Any) -> None:
    module, client, prop_dir, workers = launcher
    task, pids = _slow_bake(client, prop_dir)
    pids.append(workers.tasks[task].proc.pid)
    running = client.get(f"/spellcast/bake/{task}").json()
    assert running["state"] == "running" and running["fraction"] == 0.5
    second = _bake(client, prop_dir, "ok")
    assert second.status_code == 409 and "already baking" in second.json()["detail"]
    assert all(pid_alive(pid) for pid in pids)
    cancelled = client.post(f"/spellcast/bake/{task}/cancel", json={}).json()
    assert cancelled["state"] == "cancelled"
    assert not any(pid_alive(pid) for pid in pids) and not list((prop_dir / "projection").iterdir())
    assert client.post(f"/spellcast/bake/{task}/cancel", json={}).json()["state"] == "cancelled"
    assert _bake_status(client, _bake(client, prop_dir, "ok").json()["task_id"])["state"] == "done"

    # A graceful stop of Instant Meshes (the app shuts down) kills a running bake too.
    app = FastAPI()
    others = module.BakeWorkers(workers.worker)
    module.add_bake_routes(app, others)
    with TestClient(app) as other_client:
        task, pids = _slow_bake(other_client, prop_dir)
        pids.append(others.tasks[task].proc.pid)
    assert not any(pid_alive(pid) for pid in pids)
    assert sorted(p.name for p in (prop_dir / "projection").iterdir()) == ["bake_001"]


def test_launcher_reports_what_went_wrong(launcher: Any) -> None:
    _, client, prop_dir, workers = launcher
    for mode, detail in (("error", "The low poly is inside out."),
                         ("crash", "The bake stopped unexpectedly (exit code 3)."),
                         ("missing", "The bake ended, but nothing.glb is missing.")):
        status = _bake_status(client, _bake(client, prop_dir, mode).json()["task_id"])
        assert status["state"] == "error" and status["detail"].startswith(detail), status
        assert not list((prop_dir / "projection").iterdir())
    outside = _bake(client, prop_dir, "ok", out_root=str(prop_dir.parents[3] / "elsewhere"))
    assert outside.status_code == 400 and "inside its own projects folder" in outside.json()["detail"]
    ply = prop_dir / "retopo" / "retopo.ply"
    ply.write_bytes(b"x")
    wrong = _bake(client, prop_dir, "ok", low=str(ply))
    assert wrong.status_code == 400 and ".glb, .obj" in wrong.json()["detail"]
    gone = _bake(client, prop_dir, "ok", high=str(prop_dir / "mesh" / "gone.glb"))
    assert gone.status_code == 400 and "does not exist any more" in gone.json()["detail"]
    assert client.get("/spellcast/bake/nope").status_code == 404
    workers.worker = prop_dir / "missing_worker.py"
    missing = _bake(client, prop_dir, "ok")
    assert missing.status_code == 500 and "bake program is missing" in missing.json()["detail"]
    assert not list((prop_dir / "projection").iterdir())


def test_launcher_imports_a_model_in_a_worker(launcher: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    module, client, prop_dir, _ = launcher
    mesh = prop_dir / "mesh"
    src = mesh / "import_001" / "model.fbx"

    def convert(dst: str, **overrides: Any) -> Any:
        return client.post("/spellcast/import", json={"src": str(src), "dst": str(mesh / dst), "overrides": overrides})

    done = convert("imported_001.glb")
    assert done.status_code == 200, done.text
    assert done.json() == {"glb": str((mesh / "imported_001.glb").resolve()), "seconds": 0.1, "vertices": 3,
                           "faces": 1, "materials": [], "warnings": ["EXR skipped"]}
    assert (mesh / "imported_001.glb").read_bytes() == b"glTF-imported"
    failed = convert("imported_002.glb", mode="error")
    assert failed.status_code == 422 and failed.json()["detail"] == "This FBX has no mesh in it."
    monkeypatch.setattr(module, "IMPORT_TIMEOUT_SECONDS", 1)
    pidfile = mesh / "worker.pid"
    slow = convert("imported_003.glb", mode="hang", pidfile=str(pidfile))
    assert slow.status_code == 504 and "took longer than" in slow.json()["detail"]
    assert not pid_alive(int(pidfile.read_text()))
    pidfile.unlink()
    assert sorted(p.name for p in mesh.iterdir()) == ["import_001", "imported_001.glb", "mesh.glb"]
    assert convert("imported_004.obj").status_code == 400


def test_launcher_reports_the_viewer_settings_and_retopologizes_with_them(launcher: Any) -> None:
    module = launcher[0]
    # A session's config (instant_meshes_brush.session_manager.BrushSession.config) in the hub's words.
    config = {"rosy": 6, "posy": 3, "scale": 0.012, "face_count": -1, "vertex_count": 3390, "crease_angle": 30.0,
              "extrinsic": False, "align_to_boundaries": True, "deterministic": False, "smooth_iter": 3,
              "pure_quad": False}
    assert module._viewer_settings(config, 0.25) == {
        "target_vertices": 3390, "pure_quad": False, "crease_angle": 30.0, "smooth_iter": 3, "rosy": 6, "posy": 3,
        "extrinsic": False, "align_to_boundaries": True, "uv_leniency": 0.25}
    # An auto retopo takes them all, as the adapter sends them.
    request = module.RetopoRequest(src="a.glb", dst="b.obj", **imesh._retopo_params({"rosy": 6, "posy": 3}))
    assert (request.rosy, request.posy, request.extrinsic, request.align_to_boundaries) == (6, 3, True, False)
    with pytest.raises(ValueError):
        module.RetopoRequest(src="a.glb", dst="b.obj", rosy=5)


class _Rows:
    """Stands in for a numpy array: the launcher only reads .shape of what it passes along."""

    def __init__(self, rows: int) -> None:
        self.shape = (rows, 3)


def test_launcher_unwraps_another_tools_low_poly_as_it_is(launcher: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /spellcast/unwrap: LATO.2's or TriFlow's low poly gets UVs, its faces and positions as they are (no
    solve), written with the Python writer; a failed unwrap still writes the mesh, without UVs, and says so."""
    module, _, prop_dir, _ = launcher
    calls: list[tuple[Any, ...]] = []
    mesh = (_Rows(5), _Rows(4))
    layout = types.SimpleNamespace(cut_count=2)        # two faces a UV seam cuts diagonally: written as 4 triangles

    class UnwrapError(RuntimeError):
        pass

    def unwrap(vertices: Any, faces: Any, leniency: float) -> Any:
        calls.append(("unwrap", (vertices, faces) == mesh, leniency))
        if leniency == 0:
            raise UnwrapError("xatlas gave up")
        return layout

    def write_obj(path: Any, vertices: Any, faces: Any, uv_layout: Any) -> None:
        calls.append(("write", (vertices, faces) == mesh, uv_layout is layout))
        Path(path).write_text("v 0 0 0\n", encoding="utf-8")

    def load_source(src: Any, materials: bool = True) -> str:
        calls.append(("load", Path(src).name, materials))
        if "broken" in Path(src).name:
            raise module.assets.AssetError("no triangles")
        return "source"

    monkeypatch.setattr(module.assets, "load_source", load_source, raising=False)
    monkeypatch.setattr(module.assets, "solver_mesh", lambda source: mesh, raising=False)
    for name, value in (("available", lambda: True), ("unwrap", unwrap), ("UnwrapError", UnwrapError),
                        ("write_obj", write_obj)):
        monkeypatch.setattr(module.uv, name, value, raising=False)
    app = FastAPI()
    module.add_spellcast_routes(app, types.SimpleNamespace())
    client = TestClient(app)
    src, dst = prop_dir / "retopo" / "lato2.raw.obj", prop_dir / "retopo" / "lato2.obj"
    src.write_text("v 0 0 0\n", encoding="utf-8")

    def post(**changes: Any) -> Any:
        return client.post("/spellcast/unwrap", json={"src": str(src), "dst": str(dst), "uv_leniency": 0.3, **changes})

    done = post()
    assert done.status_code == 200, done.text
    result = done.json()
    assert result.pop("seconds") >= 0 and result == {"vertices": 5, "faces": 6, "has_uv": True, "warning": ""}
    assert calls == [("load", "lato2.raw.obj", False), ("unwrap", True, 0.3), ("write", True, True)]
    assert sorted(p.name for p in dst.parent.iterdir()) == ["lato2.obj", "lato2.raw.obj", "retopo.obj"]

    failed = post(uv_leniency=0).json()
    assert (failed["has_uv"], failed["faces"]) == (False, 4)
    assert failed["warning"] == ("The mesh was saved without UVs because unwrapping failed (xatlas gave up). "
                                 "Try another UV leniency.")
    assert calls[-1] == ("write", True, False)
    monkeypatch.setattr(module.uv, "available", lambda: False)
    assert "xatlas package is missing" in post().json()["warning"]

    broken = prop_dir / "retopo" / "broken.obj"
    broken.write_text("x", encoding="utf-8")
    for changes, status, words in (({"src": str(broken)}, 422, "could not use broken.obj"),
                                   ({"src": str(prop_dir / "retopo" / "gone.obj")}, 400, "does not exist"),
                                   ({"dst": str(prop_dir / "retopo" / "lato2.glb")}, 400, "OBJ files only"),
                                   ({"dst": str(prop_dir.parents[3] / "elsewhere.obj")}, 400,
                                    "inside its own projects folder"),
                                   ({"uv_leniency": 2}, 422, "Instant Meshes rejected the settings")):
        answer = post(**changes)
        assert answer.status_code == status and words in answer.json()["detail"], (changes, answer.text)
