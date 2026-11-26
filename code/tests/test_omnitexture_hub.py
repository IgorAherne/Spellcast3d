"""The Omni-Texture stage, hub side: its place among the stages (and the projects saved before it), the prop's
"omnitexture" block and what makes it stale, the job routed to the Qwen adapter, the Projection stage baking the
Omni-Texture's colours, and Export all saving it. The painting itself (adapters/omnitexture.py and its worker) has
tests of its own; here a stand-in replaces omnitexture.run.

Runs in the hub venv. Qwen-Image's launcher is played by fake_tool.py (test_qwen_hub.FakeQwen), Instant Meshes' by
test_projection_hub.FakeImesh.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import threading
import time
import types
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from spellcast import paths, projects
from spellcast.adapters import projection, omnitexture
from spellcast.adapters.base import StageContext, StageError, ToolNeedsRestart
from spellcast.exports import Exporter
from spellcast.jobs import NEXT_STAGE, JobQueue
from spellcast.projects import (KEEP_OMNITEXTURES, ProjectError, ProjectStore, high_poly, repaint_record,
                                omnitexture_fresh,
                                omnitexture_used)
from spellcast.server import NO_OMNITEXTURE, create_app
from test_core import PNG, FakeAdapter, _poll, make_settings, manager
from test_projection_hub import FakeConverter, FakeImesh, _follow_job, ready_to_bake
from test_qwen_hub import FakeQwen, _static, api, root_config  # noqa: F401  (fixtures)

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Spellcast3D is Windows-only")

ST = "omnitexture/ot_001/omnitexture.glb"
FINISHED = {"done", "error", "cancelled"}


def _give_omnitexture(store: ProjectStore, project: str, prop_id: str, number: int = 1, **patch: Any) -> str:
    """Give a prop the file of an Omni-Texture (ot_NNN) and its block, made from its mesh.glb, one view painted
    (a result without views is the prop's own texture) -> the glb's path."""
    rel = f"omnitexture/ot_{number:03d}/omnitexture.glb"
    file = store.prop_dir(project, prop_id) / rel
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(f"glTF-omnitexture {number}".encode())
    store.apply_stage_patch(project, prop_id, "omnitexture", {
        "glb": rel, "source_mesh": "mesh/mesh.glb", "views": [f"omnitexture/ot_{number:03d}/views/0-front/answer.png"],
        "stats": {"seconds": 2.0}, **patch})
    return rel


# ---- the stages -----------------------------------------------------------------------------------------------

def test_omnitexture_is_the_third_stage_and_runs_on_qwen() -> None:
    assert [stage for stage, _tool in paths.STAGES] == ["images", "mesh", "omnitexture", "retopo", "projection", "rig"]
    assert dict(paths.STAGES)["omnitexture"] == paths.QWEN and paths.STAGE_TOOLS["omnitexture"] == (paths.QWEN,)
    assert FakeQwen(Path(".")).stages == ("images", "omnitexture")
    # The 3D stage hands the card on to Qwen for Omni-Texture; an Omni-Texture job done hands it to no one (the
    # artist goes on there, masking and repainting: tests/test_retopo_hub.py).
    assert NEXT_STAGE["mesh"] == "omnitexture" and "omnitexture" not in NEXT_STAGE


def test_projects_saved_before_omnitexture_open_on_the_stage_they_had(tmp_path: Path) -> None:
    """The rows-of-cards version stored the row's number: read with the stages of its time, never today's list."""
    root = tmp_path / "projects"
    for name, ui in (("Row2", {"active_row": 2}), ("Row4", {"active_row": 4, "positions": {}}),
                     ("Bool", {"active_row": True}), ("Far", {"active_row": 9}), ("Named", {"stage": "omnitexture"}),
                     ("Retopo", {"stage": "retopo"})):
        (root / name).mkdir(parents=True)
        (root / name / "project.json").write_text(json.dumps({"name": name, "ui": ui, "props": [
            {"id": "p_025d38", "name": "Barrel", "images": {"settings": {}}, "mesh": {"glb": "mesh/mesh.glb"},
             "omnitexture": {"settings": {"chosen": "yes"}, "views": "nope"}}]}), encoding="utf-8")
    store = ProjectStore(root)
    names = ("Row2", "Row4", "Bool", "Far", "Named", "Retopo")
    stages = {name: store.get_project(name)["ui"]["stage"] for name in names}
    assert stages == {"Row2": "retopo", "Row4": "rig", "Bool": "images", "Far": "images", "Named": "omnitexture",
                      "Retopo": "retopo"}
    # A block that is not right is repaired: chosen is True or False, views a list of paths.
    block = store.get_prop("Row2", "p_025d38")["omnitexture"]
    assert block == {"settings": {"chosen": False}, "views": [], "projected": [], "layers": [], "sheets": [],
                     "steps": [], "repaints": [], "bases": [], "glb": None, "source_mesh": None,
                     "stale": False, "stats": None, "painted": False}


# ---- the prop's block ------------------------------------------------------------------------------------------

def test_the_choice_is_a_setting_a_new_prop_never_inherits(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    prop = store.add_prop("P", "Barrel")
    assert prop["omnitexture"] == {"settings": {"chosen": False}, "glb": None, "source_mesh": None, "views": [],
                                    "projected": [], "layers": [], "sheets": [], "steps": [], "repaints": [],
                                    "bases": [], "stale": False, "stats": None, "painted": False}
    pid = prop["id"]
    assert store.patch_prop("P", pid, {"omnitexture": {"settings": {"chosen": True}}})["omnitexture"]["settings"] == {
        "chosen": True}
    with pytest.raises(ProjectError, match="either chosen for Omni-Texture or not"):
        store.patch_prop("P", pid, {"omnitexture": {"settings": {"chosen": "yes"}}})
    with pytest.raises(ProjectError):
        store.patch_prop("P", pid, {"omnitexture": {"glb": "x.glb"}})          # results are not settings
    assert store.add_prop("P", "Crate")["omnitexture"]["settings"]["chosen"] is False
    assert store.get_prop("P", pid)["omnitexture"]["settings"]["chosen"] is True        # saved, read again


def test_a_new_model_makes_the_omnitexture_stale_and_a_new_omnitexture_the_bake(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    pid = store.add_prop("P", "Barrel")["id"]
    ready_to_bake(store, "P", pid)
    store.apply_stage_patch("P", pid, "projection", {"glb": "projection/bake_001/projection.glb",
                                                     "source_mesh": "mesh/mesh.glb"})
    store.set_thumb("P", pid, "omnitexture", PNG)
    assert store.get_prop("P", pid)["thumbs"]["omnitexture"]["path"] == "thumbs/omnitexture.png"

    # An Omni-Texture the prop does not use (not chosen) changes nothing of its bake.
    _give_omnitexture(store, "P", pid)
    prop = store.get_prop("P", pid)
    assert prop["omnitexture"]["glb"] == ST and prop["omnitexture"]["stale"] is False
    assert omnitexture_fresh(prop) and not omnitexture_used(prop) and high_poly(prop) == "mesh/mesh.glb"
    assert prop["projection"]["stale"] is False
    assert "omnitexture" not in prop["thumbs"], "its snapshot is drawn again"

    store.patch_prop("P", pid, {"omnitexture": {"settings": {"chosen": True}}})
    assert high_poly(store.get_prop("P", pid)) == ST
    second = _give_omnitexture(store, "P", pid, number=2)
    prop = store.get_prop("P", pid)
    assert prop["projection"]["stale"] is True, "the bake reads the Omni-Texture's colours: bake again"
    assert prop["retopo"]["stale"] is False, "Retopo reads the 3D model, which did not change"
    assert high_poly(prop) == second

    # Pixal3D always writes mesh/mesh.glb: the flag says what the path cannot.
    prop = store.apply_stage_patch("P", pid, "mesh", {"glb": "mesh/mesh.glb"})
    assert prop["omnitexture"]["stale"] is True and not omnitexture_fresh(prop)
    assert high_poly(prop) == "mesh/mesh.glb", "a stale Omni-Texture is not baked"
    # A result says itself whether its 3D model changed while it was painted (like a bake).
    stale = store.apply_stage_patch("P", pid, "omnitexture", {"glb": ST, "stale": True})
    assert stale["omnitexture"]["stale"] is True
    assert store.apply_stage_patch("P", pid, "omnitexture", {"glb": ST})["omnitexture"]["stale"] is False
    # A model of another path (an import) shows by source_mesh too.
    prop = store.apply_stage_patch("P", pid, "mesh", {"glb": "mesh/imported_001.glb"})
    prop["omnitexture"]["stale"] = False
    assert not omnitexture_fresh(prop)


def test_a_repaint_or_a_fade_never_makes_a_stale_omnitexture_fresh(tmp_path: Path) -> None:
    """The 3D model is made again while a repaint is faded (or put on): the fade's patch was written from the prop
    as it was before, "stale": false. The Omni-Texture stays stale; one made anew is fresh."""
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    pid = store.add_prop("P", "Barrel")["id"]
    ready_to_bake(store, "P", pid)
    first = _give_omnitexture(store, "P", pid)
    store.apply_stage_patch("P", pid, "mesh", {"glb": "mesh/mesh.glb"})
    assert store.get_prop("P", pid)["omnitexture"]["stale"] is True
    second = "omnitexture/ot_002/omnitexture.glb"
    store.apply_stage_patch("P", pid, "omnitexture", {"glb": second, "steps": [first, second], "stale": False,
                                                       "repaints": [repaint_record(second)]})
    assert store.get_prop("P", pid)["omnitexture"]["stale"] is True
    _give_omnitexture(store, "P", pid, number=3, steps=[], repaints=[], stale=False)
    assert store.get_prop("P", pid)["omnitexture"]["stale"] is False


def test_older_omnitexture_folders_are_tidied_away(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    pid = store.add_prop("P", "Barrel")["id"]
    folder = store.prop_dir("P", pid) / "omnitexture"
    for name in ("ot_001", "ot_002", "ot_003", "ot_010.tmp", "other"):
        (folder / name).mkdir(parents=True)
    _give_omnitexture(store, "P", pid, number=4)
    assert sorted(p.name for p in folder.iterdir()) == ["ot_003", "ot_004", "ot_010.tmp", "other"]
    assert KEEP_OMNITEXTURES == 2
    # A result in an older folder than the newest one there keeps its own folder.
    (folder / "ot_009").mkdir()
    _give_omnitexture(store, "P", pid, number=5)
    assert sorted(p.name for p in folder.iterdir()) == ["ot_005", "ot_009", "ot_010.tmp", "other"]


# ---- the job: the Qwen adapter ------------------------------------------------------------------------------------

class FakePaint:
    """Stands in for omnitexture.run: writes omnitexture/ot_NNN/omnitexture.glb (a copy of mesh.glb) and a view,
    remembers each call; `fail` makes it say a problem instead; `during(ctx)` runs while it works."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = ""
        self.during: Callable[[StageContext], None] | None = None

    async def __call__(self, ctx: StageContext, prop: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"prop": ctx.prop_id, "stage": ctx.stage, "params": dict(params), "base_url": ctx.base_url})
        if self.during is not None:
            self.during(ctx)
        if self.fail:
            raise StageError(self.fail)
        ctx.report(0.5, "Painting view 4 of 8…")
        await asyncio.sleep(0.05)
        out = ctx.prop_dir / "omnitexture" / f"ot_{len(self.calls):03d}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "omnitexture.glb").write_bytes((ctx.prop_dir / prop["mesh"]["glb"]).read_bytes())
        (out / "view_1.png").write_bytes(PNG)
        return {"glb": ctx.rel(out / "omnitexture.glb"), "source_mesh": prop["mesh"]["glb"],
                "views": [ctx.rel(out / "view_1.png")], "stats": {"seconds": 3.5}}


@pytest.fixture
def paint(monkeypatch: pytest.MonkeyPatch) -> FakePaint:
    fake = FakePaint()
    monkeypatch.setattr(omnitexture, "run", fake)
    return fake


def _meshed(client: TestClient, app: Any, name: str, chosen: bool = True) -> str:
    pid = client.post("/api/projects/Demo/props", json={"name": name}).json()["id"]
    store = app.state.hub.store
    glb = store.prop_dir("Demo", pid) / "mesh" / "mesh.glb"
    glb.parent.mkdir(parents=True, exist_ok=True)
    glb.write_bytes(f"glTF of {name}".encode())
    store.apply_stage_patch("Demo", pid, "mesh", {"glb": "mesh/mesh.glb"})
    assert client.patch(f"/api/projects/Demo/props/{pid}",
                        json={"omnitexture": {"settings": {"chosen": chosen}}}).status_code == 200
    return pid


def _job(client: TestClient, body: dict[str, Any]) -> dict[str, Any]:
    job = client.post("/api/jobs", json=body)
    assert job.status_code == 200, job.text
    return _poll(client, f"/api/jobs/{job.json()['id']}", lambda j: j["status"], FINISHED)


def test_a_omnitexture_job_runs_on_qwen_for_the_chosen_props(api: Any, paint: FakePaint,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    from spellcast.adapters import qwen

    monkeypatch.setattr(qwen, "_holding", "turbo")      # (the test's Qwen-Image never tells it: launch_spec)
    app, client, _ = api
    client.post("/api/projects", json={"name": "Demo"})
    barrel, crate = _meshed(client, app, "Barrel"), _meshed(client, app, "Crate")
    # Images settings the Images stage would refuse (the full model is not downloaded): not this stage's business.
    client.patch(f"/api/projects/Demo/props/{barrel}", json={"images": {"settings": {"qwen_model": "full"}}})
    refused = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [barrel]})
    assert refused.status_code == 400

    job = _job(client, {"stage": "omnitexture", "project": "Demo", "prop_ids": [barrel, crate]})
    assert job["status"] == "done" and job["tool"] == paths.QWEN, job
    # The props of the model Qwen-Image holds first (QwenAdapter.prop_order): a fresh one holds the Turbo.
    assert job["prop_ids"] == [crate, barrel]
    assert [(c["prop"], c["stage"]) for c in paint.calls] == [(crate, "omnitexture"), (barrel, "omnitexture")]
    port = client.get("/api/status").json()["tools"][paths.QWEN]["port"]
    assert all(c["base_url"] == f"http://127.0.0.1:{port}" for c in paint.calls), "Qwen-Image's own server"
    assert paint.calls[0]["params"] == {"chosen": True}
    prop = next(p for p in client.get("/api/projects/Demo").json()["props"] if p["id"] == barrel)
    assert prop["omnitexture"]["glb"] == "omnitexture/ot_002/omnitexture.glb"
    assert prop["omnitexture"]["views"] == ["omnitexture/ot_002/view_1.png"]
    assert (prop["omnitexture"]["stale"], prop["omnitexture"]["source_mesh"]) == (False, "mesh/mesh.glb")
    assert prop["omnitexture"]["settings"] == {"chosen": True} and omnitexture_used(prop)
    assert client.get("/api/status").json()["stage_tools"]["omnitexture"] == [paths.QWEN]

    paint.fail = "Qwen-Image 2.1 could not paint this model."
    failed = _job(client, {"stage": "omnitexture", "project": "Demo", "prop_ids": [barrel]})
    assert failed["status"] == "error" and failed["errors"] == {barrel: paint.fail}
    kept = next(p for p in client.get("/api/projects/Demo").json()["props"] if p["id"] == barrel)["omnitexture"]
    assert kept["glb"] == "omnitexture/ot_002/omnitexture.glb", "a failed painting keeps the one before"


def test_omnitexture_and_images_jobs_take_turns_on_the_one_qwen(api: Any, paint: FakePaint) -> None:
    """Both stages run on Qwen-Image's one process: one job after the other, in the order they were queued, and the
    second one finds the tool loaded (not started again)."""
    app, client, _ = api
    client.post("/api/projects", json={"name": "Demo"})
    pid = _meshed(client, app, "Barrel")
    seen: list[tuple[str, int]] = []
    qwen = app.state.hub.procman.get(paths.QWEN)
    client.patch(f"/api/projects/Demo/props/{pid}", json={"images": {"settings": {"prompt": "a red barrel"}}})
    images = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [pid]}).json()
    paint.during = lambda ctx: seen.append((app.state.hub.jobs.get(images["id"])["status"], qwen.proc.pid))
    job = client.post("/api/jobs", json={"stage": "omnitexture", "project": "Demo", "prop_ids": [pid]}).json()
    assert job["status"] == "queued"
    images = _poll(client, f"/api/jobs/{images['id']}", lambda j: j["status"], FINISHED)
    first = qwen.proc.pid
    job = _poll(client, f"/api/jobs/{job['id']}", lambda j: j["status"], FINISHED)
    assert images["status"] == "done" and job["status"] == "done", (images, job)
    assert seen == [("done", first)], "it waited for the Images job, and found Qwen-Image loaded"
    assert client.get("/api/status").json()["tools"][paths.QWEN]["state"] == "ready"


def test_without_qwen_the_stage_is_refused_in_plain_words(tmp_path: Path) -> None:
    adapters = {"forge": FakeAdapter("forge", "images", True, tmp_path),
                "pixal3d": FakeAdapter("pixal3d", "mesh", True, tmp_path)}
    app = create_app(adapters, settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=_static(tmp_path), logs_dir=tmp_path / "logs", gpu_probe=lambda: None)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        client.post("/api/projects", json={"name": "Demo"})
        pid = client.post("/api/projects/Demo/props", json={"name": "Barrel"}).json()["id"]
        assert client.get("/api/status").json()["stage_tools"]["omnitexture"] == []
        answer = client.post("/api/jobs", json={"stage": "omnitexture", "project": "Demo", "prop_ids": [pid]})
        assert answer.status_code == 409 and answer.json()["detail"] == NO_OMNITEXTURE
        assert "backend = qwen" in NO_OMNITEXTURE


# ---- Projection bakes the Omni-Texture's colours -------------------------------------------------------------------

def test_the_bake_reads_the_omnitexture_while_the_prop_is_chosen(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    pid = store.add_prop("P", "Barrel")["id"]
    prop_dir = ready_to_bake(store, "P", pid)
    ctx = types.SimpleNamespace(prop_dir=prop_dir)
    assert projection._inputs(ctx, store.get_prop("P", pid)) == ("mesh/mesh.glb", "retopo/retopo.obj")
    _give_omnitexture(store, "P", pid)
    assert projection._inputs(ctx, store.get_prop("P", pid))[0] == "mesh/mesh.glb", "made, but not chosen"
    store.patch_prop("P", pid, {"omnitexture": {"settings": {"chosen": True}}})
    assert projection._inputs(ctx, store.get_prop("P", pid)) == (ST, "retopo/retopo.obj")
    (prop_dir / ST).unlink()
    with pytest.raises(StageError, match="Omni-Texture of this prop .* is missing"):
        projection._inputs(ctx, store.get_prop("P", pid))
    store.apply_stage_patch("P", pid, "mesh", {"glb": "mesh/mesh.glb"})      # stale now: the 3D model again
    assert projection._inputs(ctx, store.get_prop("P", pid))[0] == "mesh/mesh.glb"


def test_a_projection_job_bakes_the_omnitexture_and_records_it(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("Demo")
    pid = store.add_prop("Demo", "Barrel")["id"]
    prop_dir = ready_to_bake(store, "Demo", pid)
    _give_omnitexture(store, "Demo", pid)
    store.patch_prop("Demo", pid, {"omnitexture": {"settings": {"chosen": True}}})

    async def main() -> None:
        adapters = {paths.IMESH: FakeImesh(tmp_path, "--bake-seconds", "0.5")}
        async with manager(tmp_path, adapters) as pm:
            queue = JobQueue(pm, store, pm.settings, adapters, pm.client)
            queue.start()
            final, _, _ = await _follow_job(queue, (await queue.submit("projection", "Demo", [pid]))["id"], pm)
            assert final["status"] == "done", final
            asked = (await pm.client.get(f"{pm.get(paths.IMESH).base_url}/fake/bakes")).json()
            assert asked[-1]["request"]["high"] == str((prop_dir / ST).resolve())
            await queue.close()

    asyncio.run(main())
    baked = store.get_prop("Demo", pid)["projection"]
    assert (baked["source_mesh"], baked["stale"]) == (ST, False)


# ---- the painting loop with the albedo add-on ----------------------------------------------------------------------

class FakeWorker:
    """Stands in for omnitexture.Worker: two views, every command answered and remembered in `calls`."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.work = Path()

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(request)
        cmd = request["cmd"]
        if cmd == "load":
            self.work = Path(request["work"])
            return {"views": ["front", "back"]}
        if cmd == "view":
            k = request["index"]
            name = ("front", "back")[k]
            files = {key: self.work / f"{k}_{name}_{key}.png" for key in ("render", "model", "mask")}
            for path in files.values():
                path.write_bytes(PNG)
            return {**{key: str(path) for key, path in files.items()}, "name": name, "side": name, "object": 0.5,
                    "paint": 0.5}
        if cmd == "project":
            for kept in ("keep", "layer"):
                if request.get(kept):
                    Path(request[kept]).write_bytes(PNG)
            return {"painted": 10, "done": 0.5}
        if cmd == "mirror":
            Path(request["out"]).write_bytes(b"mirrored " + Path(request["picture"]).read_bytes())
            return {"picture": request["out"]}
        if cmd == "blur":
            made = [Path(request["out"]) / f"{n}-{Path(p).stem}.png" for n, p in enumerate(request["pictures"], 3)]
            for path in made:
                path.write_bytes(PNG)
            return {"pictures": [str(path) for path in made]}
        Path(request["out"]).write_bytes(b"glTF")
        return {"glb": request["out"], "stats": {"painted": 0.9}}

    def alive(self) -> bool:
        return not getattr(self, "stopped", False)

    def stop(self) -> None:
        pass


class FakeAlbedo:
    """Stands in for qwen.albedo (and albedo_ready: `ready`): copies the picture to `out`, remembers each call;
    `fail[k]` makes the k-th call raise that error instead."""

    def __init__(self, ready: bool = True) -> None:
        self.ready = ready
        self.calls: list[tuple[str, str, str]] = []
        self.fail: dict[int, Exception] = {}

    async def __call__(self, ctx: StageContext, picture: Path, out: Path, model: str) -> Path:
        self.calls.append((picture.name, out.name, model))
        if (error := self.fail.get(len(self.calls) - 1)) is not None:
            raise error
        out.write_bytes(picture.read_bytes())
        return out


class FakeSheetWorker(FakeWorker):
    """A worker that lays its views out together: `sheets` (lists of view indices), each a picture of `side`."""

    NAMES = ("front", "back", "front_right", "front_left")

    def __init__(self, sheets: list[list[int]], side: int = 1536) -> None:
        super().__init__()
        self.sheets, self.side = sheets, side

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        cmd = request["cmd"]
        if cmd == "load":
            self.calls.append(request)
            self.work = Path(request["work"])
            count = max(max(views) for views in self.sheets) + 1
            return {"views": list(self.NAMES[:count]),
                    "sheets": [{"size": self.side if len(views) > 1 else 1024, "views": views, "scale": 1.0}
                               for views in self.sheets]}
        if cmd == "sheet":
            self.calls.append(request)
            j = request["index"]
            files = {key: self.work / f"sheet{j}_{key}.png" for key in ("render", "model", "mask")}
            for path in files.values():
                path.write_bytes(PNG)
            return {**{key: str(path) for key, path in files.items()}, "size": self.side, "scale": 1.0,
                    "views": [{"index": k, "name": self.NAMES[k], "side": self.NAMES[k].replace("_", " "),
                               "object": 0.1, "paint": 0.5, "window": [10 + 100 * n, 20, 90, 200]}
                              for n, k in enumerate(self.sheets[j])]}
        if cmd == "paint":
            self.calls.append(request)
            for path in [request.get("keep")] + [part.get(key) for part in request["views"]
                                                 for key in ("layer", "answer", "keep")]:
                if path:
                    Path(path).write_bytes(PNG)
            return {"painted": 10, "done": 0.5, "views": []}
        return super().call(request)


def _paint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, albedo: FakeAlbedo,
           params: dict[str, Any] | None = None,
           asked: list[dict[str, Any]] | None = None,
           own: tuple[str, ...] = (), worker: FakeWorker | None = None,
           card_side: int | None = None, had: Callable[[Path], dict[str, Any]] | None = None,
           workers: Callable[[], FakeWorker] | None = None,
           prop_dir: Path | None = None) -> tuple[dict[str, Any], FakeWorker, list[str]]:
    """omnitexture.run on a prop with a 3D model, Qwen-Image's painting and the worker played by fakes -> (the
    patch it returned, the worker, the warnings it gave the artist). params: the job's; asked: takes what
    Qwen-Image was asked, view by view; own: the prop's own pictures (mesh/views/); worker: another than the
    two-view FakeWorker; card_side: Qwen-Image's word for the largest sheet it draws (None: it does not say);
    had: makes the result the prop has before the job in its folder -> its "omnitexture" block; workers: makes
    each worker asked for (the job's, then the one that puts repaints on again); prop_dir: the prop's folder in a
    store (its 3D model there already), else one made here."""
    from spellcast.adapters import qwen

    async def generate(ctx: StageContext, body: dict[str, Any]) -> dict[str, Any]:
        if asked is not None:
            asked.append(body)
        drawn = Path(body["out_dir"]) / "0001.png"
        drawn.write_bytes(PNG)
        return {"images": [str(drawn)]}

    async def albedo_ready(ctx: StageContext) -> bool:
        return albedo.ready

    async def sheet_side(ctx: StageContext) -> int | None:
        return card_side

    worker = worker or FakeWorker()
    monkeypatch.setattr(omnitexture, "Worker", workers or (lambda: worker))
    monkeypatch.setattr(qwen, "generate", generate)
    monkeypatch.setattr(qwen, "albedo", albedo)
    monkeypatch.setattr(qwen, "albedo_ready", albedo_ready)
    monkeypatch.setattr(qwen, "sheet_side", sheet_side)
    prop_dir = prop_dir or tmp_path / "prop"
    (prop_dir / "mesh" / "views").mkdir(parents=True, exist_ok=True)
    (prop_dir / "mesh" / "mesh.glb").write_bytes(b"glTF")
    for name in own:
        (prop_dir / "mesh" / "views" / name).write_bytes(PNG)
    warnings: list[str] = []
    ctx = StageContext(project="P", prop_id="p1", prop_dir=prop_dir, base_url="http://qwen", client=None,  # type: ignore[arg-type]
                       settings={}, report=lambda f, m: None, cancelled=lambda: False, stage="omnitexture",
                       warn=warnings.append)
    prop = {"mesh": {"glb": "mesh/mesh.glb"}, **({"omnitexture": had(prop_dir)} if had else {})}
    patch = asyncio.run(omnitexture.run(ctx, prop, {"seed": 5, **(params or {})}))
    return patch, worker, warnings


def _touched_up(prop_dir: Path) -> dict[str, Any]:
    """A prop the artist touched up without an Omni-Texture: its own texture (ot_001, painted by hand), a part
    masked and painted again twice (ot_002; ot_003, painted by hand too, faded 3 times as far) -> its block."""
    steps = _repainted(prop_dir, 2)
    _hand_painted(prop_dir, steps[0])
    _hand_painted(prop_dir, steps[2])
    return {"glb": steps[2], "source_mesh": "mesh/mesh.glb", "stale": False, "views": [], "steps": steps,
            "repaints": [repaint_record(steps[1], under=steps[0]), repaint_record(steps[2], fade=3.0, under=steps[1])]}


def test_what_the_artist_painted_and_had_painted_again_stays_on_top_of_a_new_omnitexture(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A job paints a prop that was touched up before (its own texture, or an earlier Omni-Texture): the hand
    paint on the result goes over the job's views as the model is written, and the repaints are put on again
    over that, oldest first, each with its fade and its step's hand paint - the prop shows the last one, and
    Ctrl+Z steps back through them to the Omni-Texture as the job made it."""
    made: list[FakeFades] = []
    prop_dir = tmp_path / "prop"
    try:
        patch, _worker, warnings = _paint(tmp_path, monkeypatch, FakeAlbedo(), had=_touched_up,
                                          workers=lambda: made.append(FakeFades()) or made[-1])
    finally:
        omnitexture.close_fades()
    st = "omnitexture/ot_{:03d}/omnitexture.glb".format
    assert patch["views"] == ["omnitexture/ot_004/views/0-front/answer.png", "omnitexture/ot_004/views/1-back/answer.png"]
    job, again = made
    finish = job.calls[-1]
    assert (finish["cmd"], Path(finish["out"]).name) == ("finish", "omnitexture.glb")
    assert Path(finish["hand_paint"]) == prop_dir / "omnitexture/ot_001/hand-paint.png"
    assert Path(finish["unpainted"]).name == "unpainted.glb" and Path(finish["unpainted"]).parent.name == "ot_004.tmp"
    assert (prop_dir / "omnitexture/ot_004/hand-paint.png").read_bytes() == b"paint of " + st(1).encode(), \
        "kept with the new result: a view taken off lays it on again"
    # The repaints: a worker of their own loads the job's result and puts both on, the second with its hand paint.
    assert [call["cmd"] for call in again.calls] == ["load", "fades"]
    assert Path(again.calls[0]["glb"]) == prop_dir / st(4)
    asked = again.calls[1]["repaints"]
    assert [Path(ask["picture"]) for ask in asked] == [prop_dir / f"omnitexture/ot_00{n}/retouch/projected.png" for n in (2, 3)]
    assert [ask["fade"] for ask in asked] == [1.0, 3.0]
    assert ["hand_paint" in ask for ask in asked] == [False, True]
    assert again.stopped is True, "let go at once: the job's next prop needs the memory"
    assert (patch["glb"], patch["steps"]) == (st(6), [st(4), st(5), st(6)])
    assert patch["repaints"] == [repaint_record(st(5), under=st(4)), repaint_record(st(6), fade=3.0, under=st(5))]
    assert (patch["bases"], warnings, patch["stats"]["warning"]) == ([], [], None)
    assert all((prop_dir / step).is_file() for step in patch["steps"])


def test_a_new_omnitexture_starts_clean_when_the_work_before_was_on_another_3d_model(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """What was painted on a 3D model made again since has another shape under it: nothing of it goes onto the
    new Omni-Texture. A repaint whose kept pictures are gone is left out with the ones after it, and the artist is
    told how many. When the repaints cannot be put on (the worker fails), the prop's job fails and the prop keeps
    all it had: saved without them, the new result would have the old one's folders - the repaints - tidied away."""
    def stale(prop_dir: Path) -> dict[str, Any]:
        return {**_touched_up(prop_dir), "stale": True}

    made: list[FakeFades] = []
    workers = lambda: made.append(FakeFades()) or made[-1]     # noqa: E731
    patch, _worker, warnings = _paint(tmp_path / "a", monkeypatch, FakeAlbedo(), had=stale, workers=workers)
    assert (patch["glb"], patch["steps"], patch["repaints"]) == ("omnitexture/ot_004/omnitexture.glb", [], [])
    assert len(made) == 1 and "hand_paint" not in made[0].calls[-1] and warnings == []

    def first_gone(prop_dir: Path) -> dict[str, Any]:
        block = _touched_up(prop_dir)
        (prop_dir / "omnitexture/ot_002/retouch/request.json").unlink()
        return block

    made.clear()
    patch, _worker, warnings = _paint(tmp_path / "b", monkeypatch, FakeAlbedo(), had=first_gone, workers=workers)
    assert (patch["glb"], patch["steps"], patch["repaints"]) == ("omnitexture/ot_004/omnitexture.glb", [], [])
    assert len(made) == 1 and "hand_paint" in made[0].calls[-1], "the hand paint on the result still goes on"
    assert warnings == [omnitexture.LEFT_BEHIND.format(count=2)] and patch["stats"]["warning"] == warnings[0]

    class Refusing(FakeFades):
        def call(self, request: dict[str, Any]) -> dict[str, Any]:
            if request["cmd"] == "fades":
                raise StageError("The worker stopped.")
            return super().call(request)

    made.clear()
    try:
        with pytest.raises(StageError) as refused:
            _paint(tmp_path / "c", monkeypatch, FakeAlbedo(), had=_touched_up,
                   workers=lambda: made.append(Refusing()) or made[-1])
    finally:
        omnitexture.close_fades()
    assert str(refused.value) == omnitexture.NOT_KEPT.format(why="The worker stopped")
    folder = tmp_path / "c" / "prop" / "omnitexture"
    assert sorted(entry.name for entry in folder.iterdir()) == ["ot_001", "ot_002", "ot_003"], \
        "the new result is gone again, what the prop had is all there"
    assert all((folder / f"ot_00{n}" / "retouch" / "projected.png").is_file() for n in (2, 3))


def test_a_carried_omnitexture_keeps_every_folder_its_steps_need_when_the_old_ones_are_tidied(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The job's patch as the store takes it (apply_stage_patch tidies the prop's older folders away): the new
    result and the repaints put on again stay, each with the pictures a later fade or Ctrl+Z needs; what the
    prop had before - the folders the repaints were carried from - goes."""
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    pid = store.add_prop("P", "Guy")["id"]
    prop_dir = store.prop_dir("P", pid)
    (prop_dir / "mesh").mkdir(parents=True)
    (prop_dir / "mesh" / "mesh.glb").write_bytes(b"glTF")
    store.apply_stage_patch("P", pid, "mesh", {"glb": "mesh/mesh.glb"})
    store.apply_stage_patch("P", pid, "omnitexture", _touched_up(prop_dir))
    made: list[FakeFades] = []
    try:
        patch, _worker, _warnings = _paint(tmp_path, monkeypatch, FakeAlbedo(), prop_dir=prop_dir,
                                           had=lambda _dir: store.get_prop("P", pid)["omnitexture"],
                                           workers=lambda: made.append(FakeFades()) or made[-1])
    finally:
        omnitexture.close_fades()
    prop = store.apply_stage_patch("P", pid, "omnitexture", patch)
    block = prop["omnitexture"]
    st = "omnitexture/ot_{:03d}/omnitexture.glb".format
    assert (block["glb"], block["steps"]) == (st(6), [st(4), st(5), st(6)]) and len(block["views"]) == 2
    assert sorted(entry.name for entry in (prop_dir / "omnitexture").iterdir()) == ["ot_004", "ot_005", "ot_006"]
    for step in block["steps"]:
        assert (prop_dir / step).is_file()
    for record in block["repaints"]:
        assert omnitexture._kept_pictures(prop_dir, record["glb"]) and (prop_dir / record["under"]).is_file()
    assert store.omnitexture_step("P", pid, st(4))["omnitexture"]["glb"] == st(4), "Ctrl+Z to the job's own result"


def test_a_prop_gets_a_result_of_its_own_texture_to_paint_and_mask_without_a_omnitexture(api: Any) -> None:
    """The page's tools on a prop that was never OmniTextured (POST .../omnitexture/own): its 3D model, copied,
    becomes the first step of a result without views. Nothing uses it until something is painted or painted
    again on it - then Projection bakes it, chosen or not. Asked again it stays; made on a 3D model that was
    made again since, it is begun anew; an Omni-Texture a job painted is never replaced by it."""
    from spellcast.projects import high_poly, omnitexture_own, omnitexture_used

    app, client, _ = api
    client.post("/api/projects", json={"name": "Demo"})
    store = app.state.hub.store
    pid = _meshed(client, app, "Guy", chosen=False)
    url = f"/api/projects/Demo/props/{pid}/omnitexture"
    store.apply_stage_patch("Demo", pid, "projection", {"glb": "projection/bake.glb", "stale": False})
    begun = client.post(f"{url}/own")
    assert begun.status_code == 200, begun.text
    prop = begun.json()["prop"]
    block = prop["omnitexture"]
    first = "omnitexture/ot_001/omnitexture.glb"
    assert (block["glb"], block["source_mesh"], block["views"], block["steps"], block["stale"], block["painted"]) == \
        (first, "mesh/mesh.glb", [], [], False, False)
    prop_dir = store.prop_dir("Demo", pid)
    assert (prop_dir / first).read_bytes() == b"glTF of Guy"
    assert os.stat(prop_dir / first).st_nlink == 1, "a copy: the 3D stage writes mesh.glb anew in place"
    assert omnitexture_own(prop) and not omnitexture_used(prop) and high_poly(prop) == "mesh/mesh.glb"
    assert prop["projection"]["stale"] is False, "a tool taken up and nothing painted: the bake stays as it is"
    assert client.post(f"{url}/own").json()["prop"]["omnitexture"]["glb"] == first, "asked again: the one it has"
    # The mask is taken for it like for any result, and a hand painting saved makes it the model the bake reads.
    assert client.put(f"{url}/mask", params={"glb": first}, content=PNG).status_code == 200
    painted = store.omnitexture_painted("Demo", pid, first)
    assert painted["omnitexture"]["painted"] is True and omnitexture_used(painted) and high_poly(painted) == first
    assert painted["projection"]["stale"] is True
    assert omnitexture_used({**painted, "omnitexture": {**painted["omnitexture"], "settings": {"chosen": True}}})
    # A part painted again counts the same (no hand paint).
    repainted = {**prop, "omnitexture": {**block, "repaints": [repaint_record("omnitexture/ot_002/omnitexture.glb", under=first)]}}
    assert omnitexture_used(repainted)
    # The 3D model made again: what was painted lay on the earlier one - not used, and begun anew when asked.
    store.apply_stage_patch("Demo", pid, "mesh", {"glb": "mesh/mesh.glb"})
    stale = store.get_prop("Demo", pid)
    assert stale["omnitexture"]["stale"] is True and high_poly(stale) == "mesh/mesh.glb"
    anew = client.post(f"{url}/own").json()["prop"]["omnitexture"]
    assert (anew["glb"], anew["stale"], anew["painted"], anew["steps"]) == ("omnitexture/ot_002/omnitexture.glb", False, False, [])
    # A job's Omni-Texture stays, fresh or not: the tools paint that.
    other = _meshed(client, app, "Barrel")
    made = _give_omnitexture(store, "Demo", other)
    assert client.post(f"/api/projects/Demo/props/{other}/omnitexture/own").json()["prop"]["omnitexture"]["glb"] == made
    assert not omnitexture_own(store.get_prop("Demo", other))
    # No 3D model: nothing to begin on.
    bare = client.post("/api/projects/Demo/props", json={"name": "Idea"}).json()["id"]
    assert client.post(f"/api/projects/Demo/props/{bare}/omnitexture/own").status_code == 409
    # Never beside a job, a fade or a hand painting of the same prop: each writes the prop's next result folder.
    hub = app.state.hub
    fresh = _meshed(client, app, "Crate", chosen=False)
    own = f"/api/projects/Demo/props/{fresh}/omnitexture/own"
    hub.fading.add(("Demo", fresh))
    assert client.post(own).status_code == 409
    hub.fading.discard(("Demo", fresh))
    hub.painting[("Demo", fresh)] += 1
    assert client.post(own).status_code == 409
    hub.painting[("Demo", fresh)] -= 1
    active = hub.jobs.active_props
    hub.jobs.active_props = lambda project, stage=None: {fresh}
    assert client.post(own).status_code == 409
    hub.jobs.active_props = active
    assert store.get_prop("Demo", fresh)["omnitexture"]["glb"] is None, "nothing was begun meanwhile"
    seen: list[bool] = []
    begin = omnitexture.begin_own

    def watched(prop_dir: Path, prop: dict[str, Any]) -> dict[str, Any]:
        seen.append(("Demo", fresh) in hub.fading)
        return begin(prop_dir, prop)

    omnitexture.begin_own = watched
    try:
        assert client.post(own).status_code == 200
    finally:
        omnitexture.begin_own = begin
    assert seen == [True] and ("Demo", fresh) not in hub.fading, "marked as being changed while it is made"


def _projections(worker: FakeWorker) -> list[dict[str, Any]]:
    return [{k: Path(v).name if isinstance(v, str) and v.endswith((".png", ".webp")) else v for k, v in call.items()}
            for call in worker.calls if call["cmd"] == "project"]


def test_each_painted_view_goes_onto_the_model_as_albedo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    albedo = FakeAlbedo()
    patch, worker, warnings = _paint(tmp_path, monkeypatch, albedo)
    assert albedo.calls == [("answer.png", "albedo.png", "turbo")] * 2
    # the add-on's answer, with the painting's colour, and what went on is kept
    assert _projections(worker) == [{"cmd": "project", "index": k, "picture": "albedo.png", "colour": "answer.png",
                                     "keep": "projected.png", "layer": "layer.webp"} for k in (0, 1)]
    assert patch["views"] == ["omnitexture/ot_001/views/0-front/answer.png", "omnitexture/ot_001/views/1-back/answer.png"]
    assert patch["projected"] == ["omnitexture/ot_001/views/0-front/projected.png",
                                  "omnitexture/ot_001/views/1-back/projected.png"]
    # each view's share alone, for the page to show on the model
    assert patch["layers"] == ["omnitexture/ot_001/views/0-front/layer.webp",
                               "omnitexture/ot_001/views/1-back/layer.webp"]
    assert (patch["stats"]["albedo"], patch["stats"]["warning"], warnings) == ("painting", None, [])
    assert (tmp_path / "prop" / patch["projected"][1]).is_file() and (tmp_path / "prop" / patch["glb"]).is_file()
    assert (tmp_path / "prop" / patch["layers"][1]).is_file()


def _names(value: Any) -> Any:
    """A request with its paths as file names (two folders up for a view's own files)."""
    if isinstance(value, dict):
        return {key: _names(v) for key, v in value.items()}
    if isinstance(value, list):
        return [_names(v) for v in value]
    if isinstance(value, str) and value.endswith((".png", ".webp")):
        return "/".join(Path(value).parts[-3:])
    return value


def test_views_laid_out_together_are_painted_in_one_picture_and_go_on_view_by_view(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker lays the front and the back out in one sheet (painted together, the views
    agree in colour): Qwen-Image is asked once, for a picture of the sheet's size whose words name both views; its
    lighting is taken off once; the worker gets the sheet's picture with, per view, where to keep that view's own
    cut of the painting, of what went on, and its layer. The page finds the sheet by its views' folders."""
    asked: list[dict[str, Any]] = []
    albedo = FakeAlbedo()
    worker = FakeSheetWorker([[0, 1]], side=1536)
    patch, _, warnings = _paint(tmp_path, monkeypatch, albedo, asked=asked, worker=worker, card_side=2048,
                                params={"sheet_side": 2048})
    assert worker.calls[0]["together"] == list(range(1024, 2049, 128)), "the test's: up to what the card draws"
    assert [call["cmd"] for call in worker.calls] == ["load", "sheet", "paint", "finish"]
    assert len(asked) == 1 and (asked[0]["width"], asked[0]["height"]) == (1536, 1536)
    assert asked[0]["prompt"].count("(viewed from the front and from the back)") == 1
    sheet = "omnitexture/ot_001/sheets/1"
    assert [Path(p).name for p in asked[0]["ref_images"]] == ["1-render.png", "model.png"]
    assert Path(asked[0]["init_image"]).parent.name == "1" and "mask_image" not in asked[0]
    # a sheet starts from the textured model, one step in (a view alone: from its depth map, all of it drawn anew)
    assert (Path(asked[0]["init_image"]).name, asked[0]["strength"]) == ("model.png", omnitexture.SHEET_STRENGTH)
    assert albedo.calls == [("answer.png", "albedo.png", "turbo")]
    assert _names(worker.calls[2]) == {
        "cmd": "paint", "index": 0, "picture": "sheets/1/albedo.png", "colour": "sheets/1/answer.png",
        "keep": "sheets/1/projected.png", "painting": "sheets/1/answer.png",
        "views": [{"index": 0, "layer": "views/0-front/layer.webp", "answer": "views/0-front/answer.png",
                   "keep": "views/0-front/projected.png"},
                  {"index": 1, "layer": "views/1-back/layer.webp", "answer": "views/1-back/answer.png",
                   "keep": "views/1-back/projected.png"}]}
    views = ["omnitexture/ot_001/views/0-front", "omnitexture/ot_001/views/1-back"]
    assert patch["views"] == [f"{view}/answer.png" for view in views]
    assert patch["projected"] == [f"{view}/projected.png" for view in views]
    assert patch["layers"] == [f"{view}/layer.webp" for view in views]
    assert patch["sheets"] == [{"folder": sheet, "views": views}] and warnings == []
    prop = tmp_path / "prop"
    for kept in ("1-render.png", "model.png", "2-mask.png", "answer.png", "albedo.png", "projected.png"):
        assert (prop / sheet / kept).is_file(), kept
    record = json.loads((prop / sheet / "request.json").read_text(encoding="utf-8"))
    assert (record["width"], record["pictures"]) == (1536, ["picture 1: 1-render.png", "picture 2: model.png"])
    assert record["views"] == [{"name": "front", "side": "front", "window": [10, 20, 90, 200]},
                               {"name": "back", "side": "back", "window": [110, 20, 90, 200]}]
    assert all((prop / path).is_file() for key in ("views", "projected", "layers") for path in patch[key])


def test_sheets_and_views_alone_mix_and_without_the_add_on_the_painting_goes_on(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Three views in one sheet and a fourth alone: two pictures, the sheet's first; without the albedo add-on a
    sheet's painting goes on as it is, each view keeping its own cut of it."""
    asked: list[dict[str, Any]] = []
    worker = FakeSheetWorker([[0, 2, 3], [1]], side=1280)
    patch, _, _ = _paint(tmp_path, monkeypatch, FakeAlbedo(ready=False), asked=asked, worker=worker,
                         params={"views": 4}, card_side=1280)
    assert [call["cmd"] for call in worker.calls] == ["load", "sheet", "paint", "view", "project", "finish"]
    assert [(body["width"], body["seed"]) for body in asked] == [(1280, 5), (1024, 6)], "a seed per first view"
    assert asked[0]["prompt"].count("(viewed from the front and from the sides)") == 1
    assert asked[1]["prompt"].count("(viewed from the back)") == 1
    assert _names(worker.calls[2]) == {
        "cmd": "paint", "index": 0, "picture": "sheets/1/answer.png", "painting": "sheets/1/answer.png",
        "views": [{"index": k, "layer": f"views/{k}-{name}/layer.webp", "answer": f"views/{k}-{name}/answer.png"}
                  for k, name in ((0, "front"), (2, "front_right"), (3, "front_left"))]}
    assert [Path(p).parent.name for p in patch["views"]] == ["0-front", "2-front_right", "3-front_left", "1-back"]
    assert patch["projected"] == [] and len(patch["layers"]) == 4
    assert patch["sheets"] == [{"folder": "omnitexture/ot_001/sheets/1",
                                "views": [f"omnitexture/ot_001/views/{name}"
                                          for name in ("0-front", "2-front_right", "3-front_left")]}]


def test_the_stages_sheets_go_up_to_2048_or_what_the_card_draws() -> None:
    """A sheet is 1024 to 2048 a side in steps of the layout's grid (several views in one 2k picture,
    each at its 1k size, so that they agree with each other), and no larger than Qwen-Image's own word for this card. The settings'
    test may ask for another largest side, or have every view painted alone."""
    sides = omnitexture.sheet_sides
    assert sides(None, 2048) == sides({}, 4096) == list(range(1024, 2049, 128))
    assert sides(None, 1664) == [1024, 1152, 1280, 1408, 1536, 1664], "what the card draws"
    assert sides(None, None) == [1024], "an older launcher says nothing: a single view's size"
    assert sides({"sheet_side": 2048}, 2048) == sides({"sheet_side": 4096}, 4096) == list(range(1024, 2049, 128))
    assert sides({"sheet_side": 2048}, 1664) == [1024, 1152, 1280, 1408, 1536, 1664]
    assert sides({"sheet_side": 1280}, 2048) == [1024, 1152, 1280]
    assert sides({"sheet_side": 2048}, None) == sides({"sheet_side": "x"}, 2048) == sides({"sheet_side": 512}, 2048) \
        == [1024], "an older launcher, or nonsense: no larger"
    assert sides({"together": False}, 2048) is None
    load = omnitexture._load(Path("m.glb"), Path("w"), 6, [1024, 1152])
    assert load["together"] == [1024, 1152] and "together" not in omnitexture._load(Path("m.glb"), Path("w"), 6)


def test_a_stage_job_asks_the_worker_for_sheets_up_to_2048(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch, worker, _ = _paint(tmp_path, monkeypatch, FakeAlbedo(), card_side=2048)
    assert worker.calls[0]["together"] == list(range(1024, 2049, 128))
    assert patch["stats"]["sheet_side"] == 2048, "kept: a view taken off later is laid out with the same sides"


def test_a_jobs_own_strength_is_every_pictures_else_a_sheet_starts_one_step_in(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stage: a sheet of several views at SHEET_STRENGTH, a view alone at STRENGTH. The settings' test's
    "strength" is every picture's."""
    for params, strengths in ((None, [omnitexture.SHEET_STRENGTH, omnitexture.STRENGTH]), ({"strength": 0.6}, [0.6, 0.6])):
        asked: list[dict[str, Any]] = []
        _paint(tmp_path / ("stage" if params is None else "test"), monkeypatch, FakeAlbedo(), params=params,
               asked=asked, worker=FakeSheetWorker([[0, 2], [1]], side=1536), card_side=2048)
        assert [body["strength"] for body in asked] == strengths, params


def test_the_add_on_failing_on_a_sheet_loses_none_of_its_views(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """Two sheets of two views; the add-on fails on the first: both sheets' paintings go onto the model as they
    are, every view keeps its cut and its layer, the add-on is not asked again, and the warning names the sheet."""
    albedo = FakeAlbedo()
    albedo.fail[0] = StageError("No room on the card.")
    worker = FakeSheetWorker([[0, 2], [1, 3]], side=1024)
    patch, _, warnings = _paint(tmp_path, monkeypatch, albedo, worker=worker, params={"views": 4})
    assert len(albedo.calls) == 1
    paints = [_names(call) for call in worker.calls if call["cmd"] == "paint"]
    assert [(p["picture"], "colour" in p, "keep" in p) for p in paints] == [("sheets/1/answer.png", False, False),
                                                                           ("sheets/2/answer.png", False, False)]
    assert all("keep" not in part for p in paints for part in p["views"])
    assert patch["projected"] == [] and len(patch["views"]) == len(patch["layers"]) == 4
    assert warnings == [patch["stats"]["warning"]]
    assert patch["stats"]["warning"] == ("Taking the lighting off stopped working at the 2 views, so they and the "
                                         "views after them went onto the model with their shading. No room on the "
                                         "card.")


def test_the_words_name_every_view_of_a_sheet_once() -> None:
    def said(*names: str) -> str:
        text = omnitexture.words([{"name": name, "side": name.replace("_", " ")} for name in names])
        return text[text.index("(") + 1:text.index(")")]

    assert said("front") == "viewed from the front"
    assert said("front", "back") == "viewed from the front and from the back"
    assert said("front", "back", "front_right") == "viewed from the front, from the back and from the side"
    assert said("front", "back", "front_right", "front_left", "back_right", "back_left") \
        == "viewed from the front, from the back and from the sides"
    assert said("front", "back", "front_right", "front_left", "top", "bottom") \
        == "viewed from the front, from the back, from the sides, from above and from below"
    own = omnitexture.words([{"name": "front", "side": "front"}, {"name": "back_left", "side": "back left"}],
                             "The {side}: {view}.")
    assert own == "The front, back left: viewed from the front and from the side."


def test_the_views_painted_and_the_words_are_the_stages_unless_the_job_says(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stage paints VIEW_COUNT views with its words, the view named in them; the settings' Omni-Texture test
    sends its own count and words as the job's params, "{view}" and "{side}" in them filled in per view."""
    asked: list[dict[str, Any]] = []
    _patch, worker, _ = _paint(tmp_path / "stage", monkeypatch, FakeAlbedo(), asked=asked)
    assert worker.calls[0]["cmd"] == "load" and worker.calls[0]["views"] == omnitexture.VIEW_COUNT == 8
    # (the mask is not one of the pictures it gets: no "only where masked")
    assert [body["prompt"] for body in asked] == [
        "Make a high-resolution and detailed final render of the object (viewed from the front), add details "
        "visible without blur. Use the depthmap as reference.",
        "Make a high-resolution and detailed final render of the object (viewed from the back), add details "
        "visible without blur. Use the depthmap as reference."]
    views = [call for call in worker.calls if call["cmd"] == "view"]
    assert [call["show"] for call in views] == ["depth", "depth"], "the stage shows the depth map"

    asked.clear()
    _patch, worker, _ = _paint(tmp_path / "test", monkeypatch, FakeAlbedo(), asked=asked,
                               params={"views": 5, "prompt": " The {side} of it, {view}. "})
    assert worker.calls[0]["views"] == 5
    assert [body["prompt"] for body in asked] == ["The front of it, viewed from the front.",
                                                  "The back of it, viewed from the back."]

    # The test may show the lit clay, the model's own texture or its normal map where the depth map would be: the
    # worker is asked for it, the words name the picture Qwen-Image really gets (not "depthmap"
    # whatever is shown), and the view's record says what was shown.
    for show, named in (("texture", "textured model"), ("clay", "lit model"), ("normal", "normal map")):
        asked.clear()
        patch, worker, _ = _paint(tmp_path / show, monkeypatch, FakeAlbedo(), asked=asked, params={"show": show})
        assert [call["show"] for call in worker.calls if call["cmd"] == "view"] == [show, show]
        assert asked[0]["prompt"].endswith(f"add details visible without blur. Use the {named} as reference.")
        kept = tmp_path / show / "prop" / Path(patch["views"][0]).parent / "request.json"
        assert json.loads(kept.read_text(encoding="utf-8"))["shown"] == show
    # the job's own words name it with "{map}"
    asked.clear()
    _paint(tmp_path / "own", monkeypatch, FakeAlbedo(), asked=asked,
           params={"show": "normal", "prompt": "Follow the {map}, {view}."})
    assert asked[0]["prompt"] == "Follow the normal map, viewed from the front."
    assert omnitexture.words({"name": "front", "side": "front"}, "the {map}") == "the depthmap"
    assert set(omnitexture.MAP) == set(omnitexture.SHOWN)
    assert [omnitexture.shown(p) for p in (None, {}, {"show": "nonsense"}, {"show": "texture"}, {"show": "clay"},
                                            {"show": "normal"})] \
        == ["depth", "depth", "depth", "texture", "clay", "normal"]


def test_qwen_image_gets_the_depth_map_and_the_props_own_front_picture_not_the_textured_model(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stage's view: picture 1 the depth map (also the start), picture 2 the example - the prop's own picture
    of its front, for every view alike, and none of its other pictures - and no mask. The model with the texture
    the 3D stage gave it is no picture any more (shown that, Qwen-Image copies it); it is still kept, and is picture 2 for a prop without pictures and for a job that asks for it.
    request.json names the pictures in their order and the example's side."""
    asked: list[dict[str, Any]] = []
    own = ("0_front.png", "1_back.png")
    patch, worker, _ = _paint(tmp_path / "stage", monkeypatch, FakeAlbedo(), asked=asked, own=own)
    assert [c["show"] for c in worker.calls if c["cmd"] == "view"] == ["depth", "depth"]
    assert not [c for c in worker.calls if c["cmd"] == "blur"]
    for body in asked:
        assert [Path(p).name for p in body["ref_images"]] == ["1-render.png", "0_front.png"]
        assert Path(body["init_image"]).name == "1-render.png"
        assert "mask_image" not in body, "no mask: Qwen-Image paints the whole picture"
    folder = tmp_path / "stage" / "prop" / Path(patch["views"][0]).parent
    record = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    assert record["pictures"] == ["picture 1: 1-render.png", "picture 2: prop: mesh/views/0_front.png"]
    assert (record["shown"], record["example"], record["pictures_size"], record["painted_where"]) \
        == ("depth", "front", None, "2-mask.png")
    assert "mask_image" not in record
    assert (folder / "model.png").is_file() and (folder / "2-mask.png").is_file()
    for case, params, pictures in (("none", None, ()), ("asked", {"example": "model"}, own)):
        asked.clear()
        patch, _worker, _ = _paint(tmp_path / case, monkeypatch, FakeAlbedo(), asked=asked, params=params, own=pictures)
        assert [Path(p).name for p in asked[0]["ref_images"]] == ["1-render.png", "model.png"], case
        kept = tmp_path / case / "prop" / Path(patch["views"][0]).parent / "request.json"
        assert json.loads(kept.read_text(encoding="utf-8"))["example"] is None


def test_a_sheets_depth_map_reaches_qwen_image_large_and_its_example_small(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A figure in the depth map's copy must be large enough to paint detail from (512 px at least); the example
    picture can be small. A sheet of several views asks for Qwen-Image's own copies by their sides: the depth map's so that every
    view shows at VIEW_COPY or more - a larger share of a sheet whose views are smaller -, every other picture's
    EXAMPLE_COPY. A view alone, and a job with the textured model as picture 2, leave the copies to the launcher."""
    sides = omnitexture.copy_sides
    four = [{}] * 4
    assert sides({"size": 2048, "scale": 1.0, "views": four}, 2) == [1152, 512]
    assert sides({"size": 2048, "scale": 0.93, "views": four}, 2) == [1248, 512], "smaller views: a larger copy"
    assert sides({"size": 1792, "scale": 1.01, "views": four[:3]}, 2) == [1024, 512]
    assert sides({"size": 1024, "scale": 0.5, "views": four}, 4) == [1024, 512, 512, 512], "no larger than the sheet"
    assert sides({"size": 1024, "views": four[:1]}, 2) is None and sides({"size": 1024}, 2) is None
    assert all(side % omnitexture.COPY_STEP == 0 for scale in (0.9, 0.95, 1.0, 1.06)
               for side in sides({"size": 2048, "scale": scale, "views": four}, 2))
    own = ("0_front.png", "1_back.png")
    for case, params, asked_for in (("stage", None, [[864, 512], None]), ("model", {"example": "model"}, [None, None]),
                                    ("more", {"pictures": "full"}, [[864, 512, 512], None])):
        asked: list[dict[str, Any]] = []
        patch, _worker, _ = _paint(tmp_path / case, monkeypatch, FakeAlbedo(), params=params, asked=asked, own=own,
                                   worker=FakeSheetWorker([[0, 2], [1]], side=1536), card_side=2048)
        assert [body["ref_sides"] for body in asked] == asked_for, case
    kept = tmp_path / "more" / "prop" / "omnitexture" / "ot_001" / "sheets" / "1" / "request.json"
    assert json.loads(kept.read_text(encoding="utf-8"))["ref_sides"] == [864, 512, 512], "kept for the page"


def test_a_repaints_example_is_the_props_picture_of_the_side_the_page_looks_from(tmp_path: Path) -> None:
    """The example is the picture of the side the camera looks from most: the largest dot product with the front,
    back, left and right directions; the only picture when there is one. The model faces +z; its own left side is
    seen from +x."""
    prop_dir = tmp_path / "prop"
    (prop_dir / "mesh" / "views").mkdir(parents=True)
    (prop_dir / "images").mkdir()
    (prop_dir / "images" / "0001.png").write_bytes(PNG)
    ctx = StageContext(project="P", prop_id="p1", prop_dir=prop_dir, base_url="http://qwen", client=None,  # type: ignore[arg-type]
                       settings={}, report=lambda f, m: None, cancelled=lambda: False, stage="omnitexture")
    prop = {"images": {"chosen": "images/0001.png"}}

    def chosen(forward: list[float], *names: str) -> str | None:
        for old in (prop_dir / "mesh" / "views").iterdir():
            old.unlink()
        for name in names:
            (prop_dir / "mesh" / "views" / name).write_bytes(PNG)
        found = omnitexture.example_from(ctx, prop, forward)
        return found and found[0]

    four = ("0_front.png", "1_left.png", "2_back.png", "3_right.png")
    for forward, side in (([0.0, -0.1, -1.0], "front"), ([0.0, 0.2, 1.0], "back"), ([-1.0, 0.0, 0.1], "left"),
                          ([1.0, -0.3, 0.0], "right"), ([-0.8, 0.0, -0.6], "left"), ([0.5, 0.0, 0.9], "back")):
        assert chosen(forward, *four) == side, forward
    assert chosen([0.0, 0.0, -1.0], "0_back.png", "1_right.png") == "right", "from the front, no front: a side"
    assert chosen([0.0, 0.0, 1.0], "0_top.png", "1_bottom.png") == "top", "none of the four sides: the stage's choice"
    assert chosen([0.0, 0.0, 1.0], "0_front.png") == "front" and chosen([0.0, 0.0, 1.0]) == "front", "the only one"
    assert omnitexture.example_from(ctx, {}, [0.0, 0.0, -1.0]) is None, "a prop without pictures"


def test_the_example_is_the_props_front_else_another_side_and_one_from_above_or_below_last(tmp_path: Path) -> None:
    """The example is the front view; without one, another side, a picture from above or below last. A prop whose
    3D model came from one picture has that picture, its front."""
    prop_dir = tmp_path / "prop"
    (prop_dir / "mesh" / "views").mkdir(parents=True)
    (prop_dir / "images").mkdir()
    (prop_dir / "images" / "0001.png").write_bytes(PNG)
    ctx = StageContext(project="P", prop_id="p1", prop_dir=prop_dir, base_url="http://qwen", client=None,  # type: ignore[arg-type]
                       settings={}, report=lambda f, m: None, cancelled=lambda: False, stage="omnitexture")

    def chosen(*names: str) -> tuple[str, str] | None:
        for old in (prop_dir / "mesh" / "views").iterdir():
            old.unlink()
        for name in names:
            (prop_dir / "mesh" / "views" / name).write_bytes(PNG)
        found = omnitexture.example_of(ctx, {"images": {"chosen": "images/0001.png"}})
        return found and (found[0], found[1].name)

    assert chosen("0_front.png", "1_left.png", "2_back.png", "3_right.png") == ("front", "0_front.png")
    assert chosen("0_top.png", "1_back.png", "2_right.png", "3_left.png") == ("left", "3_left.png")
    assert chosen("0_bottom.png", "1_back.png") == ("back", "1_back.png")
    assert chosen("0_top.png", "1_bottom.png") == ("top", "0_top.png"), "from above or below only when nothing else"
    assert chosen() == ("front", "0001.png"), "made from one picture: that picture"
    assert omnitexture.example_of(ctx, {}) is None


def test_the_settings_test_may_send_the_props_own_pictures_as_they_are_or_blurred(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The settings' Omni-Texture test may send the prop's other pictures after picture 2 (a job's "pictures"): as
    they are, or blurred - the worker makes the copies beside each view's other inputs, the depth map and the
    example stay sharp - and request.json says so. With the textured model as picture 2 ("example": "model") all
    of the prop's pictures follow it."""
    asked: list[dict[str, Any]] = []
    own = ("0_front.png", "1_left.png", "2_back.png")
    patch, worker, _ = _paint(tmp_path / "blurred", monkeypatch, FakeAlbedo(), asked=asked,
                              params={"pictures": 256}, own=own)
    blurs = [(c["side"], c["size"], [Path(p).name for p in c["pictures"]]) for c in worker.calls if c["cmd"] == "blur"]
    assert blurs == [(256, omnitexture.SIZE, list(own[1:]))] * 2
    assert [Path(p).name for p in asked[1]["ref_images"]] == ["1-render.png", "0_front.png", "3-1_left.png",
                                                              "4-2_back.png"]
    folder = tmp_path / "blurred" / "prop" / Path(patch["views"][1]).parent
    record = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    assert record["pictures_size"] == 256 and (folder / "4-2_back.png").is_file()
    assert record["pictures"][2:] == ["picture 3: 3-1_left.png", "picture 4: 4-2_back.png"]   # the page shows these

    asked.clear()
    patch, worker, _ = _paint(tmp_path / "full", monkeypatch, FakeAlbedo(), asked=asked, params={"pictures": "full"},
                              own=own)
    assert not [c for c in worker.calls if c["cmd"] == "blur"]
    assert [Path(p).name for p in asked[0]["ref_images"]] == ["1-render.png", *own], "the prop's own files, each once"
    asked.clear()
    _paint(tmp_path / "model", monkeypatch, FakeAlbedo(), asked=asked, params={"pictures": "full", "example": "model"},
           own=own)
    assert [Path(p).name for p in asked[0]["ref_images"]] == ["1-render.png", "model.png", *own]
    asks = (None, {}, {"pictures": ""}, {"pictures": 0}, {"pictures": "full"}, {"pictures": "256"}, {"pictures": 300},
            {"pictures": "x"}, {"pictures": 128}, {"pictures": 512})
    assert [omnitexture.own_pictures(p) for p in asks] == [
        (False, None), (False, None), (False, None), (False, None), (True, None), (True, 256), (False, None),
        (False, None), (True, 128), (True, 512)]


def test_the_words_name_the_view_and_speak_of_a_mask_only_while_qwen_image_sees_one() -> None:
    """The user's words, the view named: the front and the back, "the side" for the cameras between them, from
    above and below for the two slanted ones. "... only where masked" and, for the cameras after the front and the
    back, "match the surroundings outside the masked area" only while Qwen-Image sees a mask - never since it is
    given none."""
    said = {name: omnitexture.words({"name": name, "side": name.replace("_", " ")})
            for name in ("front", "back", "front_right", "back_left", "top", "bottom")}
    match = " Seamlessly match the surroundings that are outside the masked area."
    lead = "Make a high-resolution and detailed final render of the object"
    tail = ", add details visible without blur. Use the depthmap as reference."
    for name, phrase in (("front", "viewed from the front"), ("back", "viewed from the back"),
                         ("front_right", "viewed from the side"), ("back_left", "viewed from the side"),
                         ("top", "viewed from above"), ("bottom", "viewed from below")):
        assert said[name] == f"{lead} ({phrase}){tail}", name
    # With a mask in sight: the words name it, and the later cameras match their surroundings.
    seen = omnitexture.words({"name": "front", "side": "front"}, mask_seen=True)
    assert seen.endswith("Use the depthmap as reference and generate only where masked.")
    later = omnitexture.words({"name": "top", "side": "top"}, mask_seen=True)
    assert later.endswith(f"reference and generate only where masked.{match}")
    # "the white mask" made Qwen-Image draw a face mask on a character: the words never say "mask" as a thing.
    assert not any("mask " in text or "mask." in text for text in said.values())
    counts = [omnitexture.view_count(p) for p in (None, {}, {"views": 0}, {"views": "x"}, {"views": 99}, {"views": 1})]
    assert counts == [8, 8, 8, 8, omnitexture.MAX_VIEWS, 1]      # every view unless the job asks for fewer


def test_a_model_of_the_artists_own_is_shown_with_its_props_pictures(tmp_path: Path) -> None:
    """The prop's own pictures, when the settings' test sends them: the views the 3D model was made from; without
    them (an imported model) the pictures of the prop's views while its chosen picture is one of theirs, else the
    chosen picture alone."""
    prop_dir = tmp_path / "prop"
    (prop_dir / "images").mkdir(parents=True)
    for name in ("front.png", "back.png", "other.png"):
        (prop_dir / "images" / name).write_bytes(PNG)
    ctx = StageContext(project="P", prop_id="p1", prop_dir=prop_dir, base_url="http://qwen", client=None,  # type: ignore[arg-type]
                       settings={}, report=lambda f, m: None, cancelled=lambda: False, stage="omnitexture")
    views = [{"view": "front", "src": "images/front.png", "rect": None},
             {"view": "back", "src": "images/back.png", "rect": None}]

    def names(images: dict[str, Any]) -> list[str]:
        return [p.name for p in omnitexture.references(ctx, {"images": images})]

    assert names({"chosen": "images/front.png", "views": views}) == ["front.png", "back.png"]
    assert names({"chosen": "images/other.png", "views": views}) == ["other.png"]
    assert names({"chosen": "images/other.png"}) == ["other.png"] and names({}) == []
    (prop_dir / "mesh" / "views").mkdir(parents=True)
    (prop_dir / "mesh" / "views" / "0_front.png").write_bytes(PNG)
    assert names({"chosen": "images/other.png", "views": views}) == ["0_front.png"]
    # ... while they are this model's. A model made from one picture since - the artist chose another picture and
    # made 3D again - has that picture for its own, whatever views an earlier model left in the folder (a
    # grass texture was once painted from a pinecone's front); one made from views has them.

    def of_model(source: str) -> list[str]:
        prop = {"images": {"chosen": "images/other.png", "views": views}, "mesh": {"source": source}}
        return [p.name for p in omnitexture.references(ctx, prop)]

    assert of_model("images/other.png") == ["other.png"] and of_model("views:front+back") == ["0_front.png"]
    assert of_model("images/front.png") == ["front.png"], "the picture the model was made from, not the one chosen since"
    assert omnitexture.example_of(ctx, {"images": {"chosen": "images/other.png"}, "mesh": {"source": "images/other.png"}}) == (
        "front", ctx.abs("images/other.png"))


def test_a_views_request_lets_the_engine_free_its_prefix(tmp_path: Path) -> None:
    """Every view's pictures are new, so its prefix never comes back: the request says keep_prefix false (the
    launcher's SD_QWEN_PREFIX_DROP), and request.json records it. No inpaint mask is sent (the worker's alone)."""
    view = {"render": str(tmp_path / "r.png"), "model": str(tmp_path / "t.png"), "mask": str(tmp_path / "m.png")}
    body = omnitexture._request(view, [tmp_path / "own.png"], tmp_path / "out", "turbo", 5, False)
    assert body["keep_prefix"] is False
    assert body["ref_images"][:2] == [view["render"], view["model"]] and body["init_image"] == view["render"]
    assert "mask_image" not in body
    shown = omnitexture._request(view, [tmp_path / "own.png"], tmp_path / "out", "turbo", 5, False,
                                  example=tmp_path / "front.png")
    assert [Path(p).name for p in shown["ref_images"]] == ["r.png", "front.png", "own.png"]
    record = omnitexture._record(body, tmp_path, tmp_path)
    assert record["keep_prefix"] is False and "mask_image" not in record


def test_without_the_add_on_the_paintings_go_on_as_they_are(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    albedo = FakeAlbedo(ready=False)
    patch, worker, warnings = _paint(tmp_path, monkeypatch, albedo)
    assert albedo.calls == [] and warnings == []
    assert _projections(worker) == [{"cmd": "project", "index": k, "picture": "answer.png", "layer": "layer.webp"}
                                    for k in (0, 1)]
    assert patch["projected"] == [] and patch["stats"]["albedo"] is None and patch["stats"]["warning"] is None


def test_from_the_view_the_add_on_fails_on_the_paintings_go_on_with_their_shading(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure of the add-on (short of memory, say) loses no painted view: the page's warning says what happened.
    Only a Qwen-Image that has to start again fails the prop, as a painting's would."""
    albedo = FakeAlbedo()
    albedo.fail[1] = StageError("No room on the card.")
    patch, worker, warnings = _paint(tmp_path, monkeypatch, albedo)
    assert len(albedo.calls) == 2
    assert _projections(worker) == [{"cmd": "project", "index": 0, "picture": "albedo.png", "colour": "answer.png",
                                     "keep": "projected.png", "layer": "layer.webp"},
                                    {"cmd": "project", "index": 1, "picture": "answer.png", "layer": "layer.webp"}]
    assert patch["projected"] == ["omnitexture/ot_001/views/0-front/projected.png"]
    assert warnings == [patch["stats"]["warning"]]
    assert patch["stats"]["warning"] == ("Taking the lighting off stopped working at the back view, so it and the "
                                         "views after it went onto the model with their shading. No room on the card.")

    albedo = FakeAlbedo()
    albedo.fail[0] = ToolNeedsRestart("Qwen-Image 2.1 has to start again after an error. Try again.")
    with pytest.raises(ToolNeedsRestart):
        _paint(tmp_path / "again", monkeypatch, albedo)
    assert not list((tmp_path / "again" / "prop" / "omnitexture").glob("ot_*")), "nothing half done is left"


# ---- between two props of a job: the next one's model read early, the last projection behind -----------------------

def _fakes(monkeypatch: pytest.MonkeyPatch, events: list[tuple[Any, ...]], free_mb: int = 64_000) -> list[FakeWorker]:
    """omnitexture.run's Qwen-Image calls and workers played by fakes that log what they do into `events`
    (("worker", which one, cmd, index), ("generate" | "albedo", prop id)) -> the workers started, in order."""
    from spellcast.adapters import qwen

    made: list[FakeWorker] = []

    class Logged(FakeWorker):
        def call(self, request: dict[str, Any]) -> dict[str, Any]:
            events.append(("worker", made.index(self), request["cmd"], request.get("index")))
            answer = super().call(request)
            if request["cmd"] == "finish":         # a GLB takes its time
                time.sleep(0.3)
                events.append(("finished", made.index(self)))
            return answer

        def stop(self) -> None:
            events.append(("stopped", made.index(self)))

    def spawn() -> FakeWorker:
        made.append(Logged())
        return made[-1]

    async def generate(ctx: StageContext, body: dict[str, Any]) -> dict[str, Any]:
        events.append(("generate", ctx.prop_id))
        await asyncio.sleep(0.05)
        events.append(("generated", ctx.prop_id))
        drawn = Path(body["out_dir"]) / "0001.png"
        drawn.write_bytes(PNG)
        return {"images": [str(drawn)]}

    async def albedo(ctx: StageContext, picture: Path, out: Path, model: str) -> Path:
        events.append(("albedo", ctx.prop_id))
        out.write_bytes(picture.read_bytes())
        return out

    async def albedo_ready(ctx: StageContext) -> bool:
        return True

    monkeypatch.setattr(omnitexture, "Worker", spawn)
    monkeypatch.setattr(omnitexture, "free_ram_mb", lambda: free_mb)
    monkeypatch.setattr(omnitexture, "_spare", None)
    monkeypatch.setattr(qwen, "generate", generate)
    monkeypatch.setattr(qwen, "albedo", albedo)
    monkeypatch.setattr(qwen, "albedo_ready", albedo_ready)
    return made


class FakeLook(FakeWorker):
    """The worker of a repaint: look writes its pictures, left says how much of the mask's shape each of
    Qwen-Image's pictures kept (`lefts`, in turn), repaint what went onto the model."""

    lefts: tuple[float, ...] = (0.4,)
    seen = 800

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        cmd = request["cmd"]
        if cmd == "look":
            self.calls.append(request)
            files = {key: self.work / f"look_{key}.png" for key in ("depth", "viewport", "redrawn")}
            for path in files.values():
                path.write_bytes(PNG)
            return {**{key: str(path) for key, path in files.items()}, "masked": 900, "seen": self.seen}
        if cmd == "left":
            judged = [call for call in self.calls if call["cmd"] == "left"]
            self.calls.append(request)
            return {"left": self.lefts[min(len(judged), len(self.lefts) - 1)]}
        if cmd == "repaint":
            self.calls.append(request)
            for kept in ("keep", "layer"):
                if request.get(kept):
                    Path(request[kept]).write_bytes(PNG)
            return {"painted": 850}
        if cmd == "fit":
            self.calls.append(request)
            Path(request["out"]).parent.mkdir(parents=True, exist_ok=True)
            Path(request["out"]).write_bytes(b"fit " + Path(request["picture"]).read_bytes())
            return {"picture": request["out"], "size": [1024, 1024]}
        return super().call(request)


RETOUCH = {"mask": "omnitexture/mask.png", "eye": [0.1, 0.2, 3.0], "forward": [0.0, -0.1, -1.0],
           "up": [0.0, 1.0, 0.0], "tan": 0.29}


def _retouch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, albedo: FakeAlbedo, block: dict[str, Any] | None = None,
             params: dict[str, Any] | None = None, lefts: tuple[float, ...] = (0.4,), seen: int = 800,
             warnings: list[str] | None = None,
             own: tuple[str, ...] = ()) -> tuple[dict[str, Any], FakeLook, list[dict[str, Any]], Path]:
    """omnitexture.run with a "retouch" on a prop that has an Omni-Texture (ot_001) and a mask sent -> (the patch,
    the worker, what Qwen-Image was asked, the prop's folder). own: the prop's own pictures (mesh/views/)."""
    from spellcast.adapters import qwen

    asked: list[dict[str, Any]] = []

    async def generate(ctx: StageContext, body: dict[str, Any]) -> dict[str, Any]:
        asked.append(dict(body))
        drawn = Path(body["out_dir"]) / "0001.png"
        drawn.write_bytes(PNG)
        return {"images": [str(drawn)]}

    async def albedo_ready(ctx: StageContext) -> bool:
        return albedo.ready

    worker = FakeLook()
    worker.lefts, worker.seen = lefts, seen
    monkeypatch.setattr(omnitexture, "Worker", lambda: worker)
    monkeypatch.setattr(qwen, "generate", generate)
    monkeypatch.setattr(qwen, "albedo", albedo)
    monkeypatch.setattr(qwen, "albedo_ready", albedo_ready)
    prop_dir = tmp_path / "prop"
    (prop_dir / "omnitexture" / "ot_001").mkdir(parents=True)
    (prop_dir / "omnitexture" / "ot_001" / "omnitexture.glb").write_bytes(b"glTF st 1")
    (prop_dir / "omnitexture" / "mask.png").write_bytes(PNG)
    for name in own:
        (prop_dir / "mesh" / "views").mkdir(parents=True, exist_ok=True)
        (prop_dir / "mesh" / "views" / name).write_bytes(PNG)
    ctx = _st_ctx(prop_dir)
    if warnings is not None:
        ctx = dataclasses.replace(ctx, warn=warnings.append)
    block = {"glb": "omnitexture/ot_001/omnitexture.glb", "views": ["omnitexture/ot_001/views/0-front/answer.png"],
             "stats": {"seconds": 60.0, "seed": 7}, "steps": [], **(block or {})}
    patch = asyncio.run(omnitexture.run(ctx, {"mesh": {"glb": "mesh/mesh.glb"}, "omnitexture": block},
                                         {"retouch": RETOUCH, **(params or {})}))
    return patch, worker, asked, prop_dir


def test_a_repaint_shows_qwen_image_the_pages_view_and_makes_a_new_step(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """The artist's "Render Masked": the worker reads the OmniTextured model (not the 3D stage's),
    without the stage's plan, and looks through the page's camera; Qwen-Image gets the viewport with the mask in
    magenta, the depth map and the example - the prop's own picture of the side that camera looks from most, here
    its front, and no other picture of the model -, in that order, with the repaint's words and a seed of its own; its picture is looked at for the
    mask's shape and goes onto the masked surface as it was painted: its lighting is not taken off (the add-on is
    asked for the example picture alone) and the worker matches no tone. The result is a new folder
    beside the one it was made from and one more of the prop's steps; its views stay the first result's, its
    numbers too, with what the repaint did. A prop without pictures of its own gets no example."""
    _patch, _worker, bare, _dir = _retouch(tmp_path / "bare", monkeypatch, FakeAlbedo())
    assert [Path(p).name for p in bare[0]["ref_images"]] == ["1-viewport.png", "2-depth.png"]
    # (Which picture is the example, and whether it is mirrored: told here without the add-on, so the picture
    # goes as it is. With it, its lighting comes off first: the test after this one.)
    _patch, _worker, behind, _dir = _retouch(tmp_path / "behind", monkeypatch, FakeAlbedo(ready=False), own=("0_back.png", "1_front.png"),
                                             params={"retouch": {**RETOUCH, "forward": [0.0, -0.1, 1.0]}})
    assert Path(behind[0]["ref_images"][-1]).name == "0_back.png", "masked from behind: the prop's picture of its back"
    # One picture alone, the front its 3D model was made from, and the camera behind it: the prop faces the other
    # way there, so the picture is given mirrored (the worker flips it into the repaint's own folder). From the
    # front, or a side, it goes as it is.
    for forward, flipped in (([0.0, -0.1, 1.0], True), ([0.6, 0.0, 0.8], True), ([0.0, -0.1, -1.0], False), ([1.0, 0.0, 0.0], False)):
        _patch, turned, alone, alone_dir = _retouch(tmp_path / f"alone{forward[0]}{forward[2]}", monkeypatch, FakeAlbedo(ready=False), own=("0_front.png",),
                                                    params={"retouch": {**RETOUCH, "forward": forward}})
        example = Path(alone[0]["ref_images"][-1])
        record = json.loads((alone_dir / "omnitexture" / "ot_002" / "retouch" / "request.json").read_text(encoding="utf-8"))
        assert (record["retouch"]["example"], record["retouch"]["mirrored"]) == ("front", flipped), forward
        mirrors = [call for call in turned.calls if call["cmd"] == "mirror"]
        if flipped:
            assert (example.name, example.parent.name) == ("3-example-mirrored.png", "retouch"), example
            assert [Path(call["picture"]).name for call in mirrors] == ["0_front.png"]
            assert record["pictures"][-1] == "picture 3: 3-example-mirrored.png"
            assert (alone_dir / "omnitexture" / "ot_002" / "retouch" / "3-example-mirrored.png").read_bytes().startswith(b"mirrored ")
        else:
            assert example.name == "0_front.png" and mirrors == [] and record["pictures"][-1] == "picture 3: prop: mesh/views/0_front.png"
    assert [omnitexture.behind(side, [0.0, 0.0, 1.0]) for side in ("front", "back", "left", "top", None)] == [True, False, False, False, False]
    add_on = FakeAlbedo()
    patch, worker, asked, prop_dir = _retouch(tmp_path, monkeypatch, add_on, own=("0_back.png", "1_front.png"))
    load, look, fit, left, repaint, finish = worker.calls
    # The add-on is asked once, for the example picture - never for the painting.
    (lit_off,) = add_on.calls
    assert lit_off[0].startswith("1_front-") and lit_off[0].endswith(".fit.png") and lit_off[2] == "turbo"
    assert fit["cmd"] == "fit" and Path(fit["picture"]).name == "1_front.png"
    assert left["cmd"] == "left" and Path(left["picture"]).parent.name == "retouch"
    assert load["glb"].endswith("ot_001\\omnitexture.glb") or load["glb"].endswith("ot_001/omnitexture.glb")
    assert load["plan"] is False and load["views"] == 1, "it looks through the page's camera alone"
    assert look["cmd"] == "look" and {key: look[key] for key in ("eye", "forward", "up", "tan")} \
        == {key: RETOUCH[key] for key in ("eye", "forward", "up", "tan")}
    assert Path(look["mask"]).name == "mask.png" and "ot_002" in look["mask"]
    (body,) = asked
    assert body["prompt"] == ("remove the magenta-masked blob. Use the depthmap and the example image as "
                              "reference") == omnitexture.retouch_prompt()
    # The artist's own words from the box beside the mask: a sentence of their own after what to do.
    _patch, _worker, told, _dir = _retouch(tmp_path / "told", monkeypatch, FakeAlbedo(ready=False), own=("1_front.png",),
                                           params={"retouch": {**RETOUCH, "words": "  a red\nstripe  "}})
    assert told[0]["prompt"] == ("remove the magenta-masked blob. a red stripe. Use the depthmap and the example "
                                 "image as reference")
    # "No Facets": the worker is asked for a depth map without the model's facets, and the record says so.
    assert look["smooth"] is False and "smooth" not in json.loads(
        (prop_dir / "omnitexture" / "ot_002" / "retouch" / "request.json").read_text(encoding="utf-8"))["retouch"]
    _patch, smoothly, _asked, smooth_dir = _retouch(tmp_path / "smooth", monkeypatch, FakeAlbedo(ready=False),
                                                    params={"retouch": {**RETOUCH, "smooth": True}})
    assert next(call for call in smoothly.calls if call["cmd"] == "look")["smooth"] is True
    assert json.loads((smooth_dir / "omnitexture" / "ot_002" / "retouch" / "request.json").read_text(
        encoding="utf-8"))["retouch"]["smooth"] is True
    example = Path(body["ref_images"][-1])
    assert [Path(p).name for p in body["ref_images"][:2]] == ["1-viewport.png", "2-depth.png"]
    assert example.name.startswith("1_front-") and example.parent == prop_dir / "omnitexture" / "examples"
    assert Path(body["init_image"]).name == "1-viewport.png" and (body["width"], body["height"]) == (1024, 1024)
    # Drawn anew where the view is masked, and held to the view everywhere else: free, Qwen-Image gave back its
    # example picture instead of the view.
    assert (Path(body["mask_image"]).name, body["strength"]) == ("redrawn.png", 1.0) == (omnitexture.REDRAWN, omnitexture.RETOUCH_STRENGTH)
    # ... whatever strength a job names (the stage's views take one): below 1 the masked part would start from the
    # magenta itself.
    _patch, _worker, weak, _dir = _retouch(tmp_path / "weak", monkeypatch, FakeAlbedo(ready=False), own=("1_front.png",),
                                           params={"retouch": RETOUCH, "strength": 0.6})
    assert weak[0]["strength"] == 1.0
    assert Path(body["mask_image"]).parent == Path(body["init_image"]).parent
    assert body["keep_prefix"] is False and body["cfg_scale"] == 1.0
    assert 1 <= body["seed"] < 2 ** 31
    assert {key: value if key == "cmd" else Path(value).name for key, value in repaint.items()} == {
        "cmd": "repaint", "picture": "answer.png", "keep": "projected.png", "layer": "layer.webp"}
    assert finish == {"cmd": "finish", "out": finish["out"], "tone": False}
    folder = prop_dir / "omnitexture" / "ot_002"
    assert patch["glb"] == "omnitexture/ot_002/omnitexture.glb" and (folder / "omnitexture.glb").is_file()
    assert sorted(p.name for p in (folder / "retouch").iterdir()) == [
        "1-viewport.png", "2-depth.png", "answer.png", "layer.webp", "mask.png",
        "projected.png", "redrawn.png", "request.json"]
    record = json.loads((folder / "retouch" / "request.json").read_text(encoding="utf-8"))
    assert (record["start_image"], record["redrawn"], record["strength"]) == ("1-viewport.png", "redrawn.png", 1.0)
    assert "mask_image" not in record
    assert record["retouch"]["from"].endswith("ot_001/omnitexture.glb")
    assert record["retouch"]["example"] == "front", "the side the example shows, for the page"
    assert record["retouch"]["delit"] is True
    assert record["pictures"] == ["picture 1: 1-viewport.png", "picture 2: 2-depth.png",
                                  f"picture 3: prop: omnitexture/examples/{example.name}"]
    assert record["seed"] == body["seed"] and record["retouch"]["tries"] == [{"seed": body["seed"], "left": 0.4}]
    assert patch["steps"] == ["omnitexture/ot_001/omnitexture.glb", "omnitexture/ot_002/omnitexture.glb"]
    assert patch["repaints"] == [{"glb": "omnitexture/ot_002/omnitexture.glb", "fade": 1.0,
                                  "under": "omnitexture/ot_001/omnitexture.glb",
                                  "picture": "omnitexture/ot_002/retouch/projected.png",
                                  "layer": "omnitexture/ot_002/retouch/layer.webp"}]
    assert "views" not in patch and patch["bases"] == [] and patch["stale"] is False
    assert patch["stats"]["seed"] == 7 and patch["stats"]["retouch"]["painted"] == 850 \
        and patch["stats"]["retouch"]["folder"] == "omnitexture/ot_002/retouch"
    assert patch["stats"]["retouch"]["seed"] == body["seed"] and patch["stats"]["retouch"]["tries"] == 1 \
        and patch["stats"]["retouch"]["warning"] is None
    assert not (prop_dir / "omnitexture" / "mask.png").exists(), "the mask went with its repaint"


def test_a_repaints_example_has_its_lighting_taken_off_once_per_picture(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """The example a repaint is shown is the prop's own picture, lit as it was drawn, while the view it helps to
    restore is flat: its lighting is taken off first (the worker makes it fit the pass, the add-on takes the
    lighting off), and what comes back is kept in the prop's omnitexture/examples - the next repaint from that side
    asks nobody, a repaint from another side makes that side's, and a picture drawn anew gets a new one beside the
    old (the repaints made with that one still name it). Without the add-on, or when the pass fails, the picture
    goes as it is and nothing is kept (a failure is told)."""
    from spellcast.adapters import qwen

    prop_dir = tmp_path / "prop"
    views = prop_dir / "mesh" / "views"
    views.mkdir(parents=True)
    front, back = views / "0_front.png", views / "1_back.png"
    front.write_bytes(b"front as drawn")
    back.write_bytes(b"back as drawn")
    add_on = FakeAlbedo()

    async def albedo_ready(ctx: StageContext) -> bool:
        return add_on.ready

    monkeypatch.setattr(qwen, "albedo", add_on)
    monkeypatch.setattr(qwen, "albedo_ready", albedo_ready)
    told: list[str] = []
    ctx = dataclasses.replace(_st_ctx(prop_dir), warn=told.append)
    worker = FakeLook()

    def shown(picture: Path) -> tuple[Path, bool]:
        return asyncio.run(omnitexture._example_delit(ctx, worker, picture, "turbo"))

    kept = prop_dir / "omnitexture" / "examples"
    first, delit = shown(front)
    assert delit and first.parent == kept and first.name.startswith("0_front-") and first.suffix == ".png"
    assert first.read_bytes() == b"fit front as drawn", "what the add-on gave back, of the picture made to fit"
    (fit,) = worker.calls
    assert (fit["cmd"], fit["most"], fit["step"]) == ("fit", 1024, 32) and Path(fit["picture"]) == front.resolve()
    assert len(add_on.calls) == 1 and [p.name for p in kept.iterdir()] == [first.name], "no scratch file is left"
    assert shown(front) == (first, True) and len(add_on.calls) == 1 and len(worker.calls) == 1, "kept: nobody asked"
    other, delit = shown(back)
    assert delit and other.name.startswith("1_back-") and len(add_on.calls) == 2, "another side: that side's own"
    assert sorted(p.name for p in kept.iterdir()) == sorted([first.name, other.name])
    # The picture drawn anew (another size here): its example is made anew, and the old one stays for its repaints.
    front.write_bytes(b"front, drawn again")
    again, delit = shown(front)
    assert delit and again != first and again.read_bytes() == b"fit front, drawn again"
    assert first.read_bytes() == b"fit front as drawn"
    assert sorted(p.name for p in kept.iterdir()) == sorted([first.name, again.name, other.name])
    # The pass fails: the picture as it is, told, nothing kept - and the next repaint tries again.
    front.write_bytes(b"front, a third time")
    add_on.fail[len(add_on.calls)] = StageError("No room on the card.")
    assert shown(front) == (front, False) and len(told) == 1 and "No room on the card." in told[0]
    assert sorted(p.name for p in kept.iterdir()) == sorted([first.name, again.name, other.name]), "nothing new, no scratch"
    assert shown(front)[1] is True, "tried again, and made"
    # A card that has to start again is not swallowed; without the add-on the picture goes as it is, untold.
    front.write_bytes(b"front, once more ...")
    add_on.fail[len(add_on.calls)] = ToolNeedsRestart("Qwen-Image 2.1 has to start again after an error. Try again.")
    with pytest.raises(ToolNeedsRestart):
        shown(front)
    add_on.ready = False
    calls = len(add_on.calls)
    assert shown(front) == (front, False) and len(add_on.calls) == calls and len(told) == 1


def test_a_picture_that_kept_the_masks_shape_is_made_again_with_the_next_seed(tmp_path: Path,
                                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen-Image now and then lays a patch of one colour where the magenta was, instead of the surface restored:
    the worker's "left" tells, and the picture is made again with the next seed, three in all at the most; the one
    with the least of the mask left goes onto the model, the others stay beside it for the record. When every one
    kept the mask, or most of the mask was out of the camera's sight, the artist is told."""
    patch, worker, asked, prop_dir = _retouch(tmp_path / "second", monkeypatch, FakeAlbedo(), params={"seed": 40},
                                              lefts=(80.0, 3.0))
    assert [body["seed"] for body in asked] == [40, 41]
    retouch = prop_dir / "omnitexture" / "ot_002" / "retouch"
    assert (retouch / "answer.png").is_file() and (retouch / "rejected-1.png").is_file() \
        and not (retouch / "rejected-2.png").exists()
    record = json.loads((retouch / "request.json").read_text(encoding="utf-8"))
    assert record["seed"] == 41 and record["retouch"]["tries"] == [{"seed": 40, "left": 80.0}, {"seed": 41, "left": 3.0}]
    assert patch["stats"]["retouch"]["tries"] == 2 and patch["stats"]["retouch"]["warning"] is None
    told: list[str] = []
    patch, _worker, asked, prop_dir = _retouch(tmp_path / "none", monkeypatch, FakeAlbedo(), params={"seed": 40},
                                              lefts=(80.0, 30.0, 50.0), seen=300, warnings=told)
    assert [body["seed"] for body in asked] == [40, 41, 42]
    retouch = prop_dir / "omnitexture" / "ot_002" / "retouch"
    assert sorted(p.name for p in retouch.glob("rejected-*")) == ["rejected-1.png", "rejected-3.png"]
    assert json.loads((retouch / "request.json").read_text(encoding="utf-8"))["seed"] == 41, "the least of the mask"
    assert told == [omnitexture.OUT_OF_SIGHT_WARNING, omnitexture.MASK_LEFT_WARNING]
    assert patch["stats"]["retouch"]["warning"] == " ".join(told) and patch["stats"]["retouch"]["tries"] == 3


def test_a_repaints_progress_says_what_it_is_at_and_under_it_what_qwen_image_says(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The busy layer shows each line as one: what the repaint is at - a second try says why - and, on a line of
    their own, Qwen-Image's words."""
    from spellcast.adapters import qwen

    async def generate(ctx: StageContext, body: dict[str, Any]) -> dict[str, Any]:
        ctx.report(0.5, "Drawing")
        drawn = tmp_path / f"drawn-{body['seed']}.png"
        drawn.write_bytes(PNG)
        return {"images": [str(drawn)]}

    monkeypatch.setattr(qwen, "generate", generate)
    worker = FakeLook()
    worker.lefts = (80.0, 3.0)                      # the first picture kept the mask's shape
    said: list[str] = []
    ctx = dataclasses.replace(_st_ctx(tmp_path), report=lambda fraction, words: said.append(words))
    asyncio.run(omnitexture._restored(ctx, worker, {"seed": 1}, 1, tmp_path))
    assert said == ["Painting what you masked\nDrawing",
                    f"Painting what you masked, try 2 of {omnitexture.RETOUCH_TRIES}\n(the picture before kept the "
                    "mask's shape)\nDrawing"]


def test_a_repaint_after_ctrl_z_drops_what_was_taken_back_and_keeps_the_newest_steps(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    steps = [f"omnitexture/ot_{n:03d}/omnitexture.glb" for n in (1, 2, 3)]
    assert omnitexture.steps_after({"glb": steps[1], "steps": steps}, "new") == [steps[0], steps[1], "new"]
    assert omnitexture.steps_after({"glb": steps[0], "steps": []}, "new") == [steps[0], "new"]
    many = [f"s{n}" for n in range(omnitexture.MAX_STEPS)]
    assert omnitexture.steps_after({"glb": many[-1], "steps": many}, "new") == [*many[1:], "new"]
    # Without the add-on too the picture goes on as it is; a request without its camera is refused in plain words.
    _patch, worker, _asked, _dir = _retouch(tmp_path, monkeypatch, FakeAlbedo(ready=False))
    assert Path(worker.calls[3]["picture"]).name == "answer.png" and "colour" not in worker.calls[3]
    for bad in ({}, {**RETOUCH, "tan": 0}, {**RETOUCH, "eye": [1, 2]}, {**RETOUCH, "mask": ""}, "nope",
                {**RETOUCH, "words": ["a red stripe"]}, {**RETOUCH, "smooth": "yes"}):
        with pytest.raises(StageError, match="incomplete"):
            omnitexture.retouch_camera(bad)
    # The artist's words, when there are any: one sentence, kept short.
    assert "words" not in omnitexture.retouch_camera({**RETOUCH, "words": "  "})
    assert omnitexture.retouch_camera({**RETOUCH, "words": "rivets!"})["words"] == "rivets!"
    assert omnitexture.retouch_prompt("Rivets!") == ("remove the magenta-masked blob. Rivets! Use the depthmap and the "
                                                     "example image as reference")
    assert len(omnitexture.retouch_prompt("x" * 5000)) == len(omnitexture.retouch_prompt()) + omnitexture.WORDS_MOST + 2
    # An Omni-Texture made anew starts the steps afresh, and holds no repaint.
    anew, _worker, _warnings = _paint(tmp_path / "anew", monkeypatch, FakeAlbedo())
    assert anew["steps"] == [] and anew["repaints"] == [] and anew["bases"] == []
    # The repaints the models hold: those of steps taken back with Ctrl+Z are gone with them.
    held = [repaint_record(step, 2.0, under=before) for before, step in zip(steps, steps[1:])]
    new = "omnitexture/ot_009/omnitexture.glb"
    kept = omnitexture.steps_after({"glb": steps[1], "steps": steps}, new)
    after = omnitexture.records_after(kept, [*held, repaint_record(new, under=steps[1])], [])
    assert [made["glb"] for made in after["repaints"]] == [steps[1], new] and after["bases"] == []
    assert after["repaints"][0]["fade"] == 2.0 and after["repaints"][1]["fade"] == 1.0


def test_a_repaint_through_the_pages_bent_lens_names_it_to_the_worker_and_keeps_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The page's "Inverse perspective": the retouch names the bent lens ("wide", with "tan" 0 or below as well);
    the worker looks through it and the repaint's record keeps it: put on again (its fade, its eye), it goes
    through that lens. A camera sends none; a lens whose picture would span nothing is refused."""
    bent = {**RETOUCH, "tan": -0.3, "wide": 0.8}
    assert omnitexture.retouch_camera(bent) == bent
    assert "wide" not in omnitexture.retouch_camera(RETOUCH) and "wide" not in omnitexture.retouch_camera({**RETOUCH, "wide": 0})
    for bad in ({**RETOUCH, "tan": -0.3}, {**RETOUCH, "tan": 0.0}, {**bent, "wide": -1.0}, {**bent, "wide": "x"},
                {**bent, "tan": -20.0}, {**bent, "wide": float("inf")}):
        with pytest.raises(StageError, match="incomplete"):
            omnitexture.retouch_camera(bad)
    _patch, worker, _asked, prop_dir = _retouch(tmp_path, monkeypatch, FakeAlbedo(ready=False), params={"retouch": bent})
    look = next(call for call in worker.calls if call["cmd"] == "look")
    assert (look["tan"], look["wide"]) == (-0.3, 0.8)
    record = json.loads((prop_dir / "omnitexture" / "ot_002" / "retouch" / "request.json").read_text(encoding="utf-8"))
    assert (record["retouch"]["tan"], record["retouch"]["wide"]) == (-0.3, 0.8)
    fading: list[FakeFades] = []
    monkeypatch.setattr(omnitexture, "Worker", lambda: fading.append(FakeFades()) or fading[-1])
    try:
        omnitexture.refade(prop_dir, {**_patch, "stale": False}, _patch["glb"], 2.0)
        (again,) = fading[0].calls[-1]["repaints"]
        assert (again["tan"], again["wide"], again["fade"]) == (-0.3, 0.8, 2.0)
    finally:
        omnitexture.close_fades()


def test_a_repaints_mask_is_the_one_the_page_sent_and_no_other_file(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A repaint job takes the mask the store keeps for it (PUT .../omnitexture/mask), which it removes once
    painted. A job that names another file is refused, and that file stays."""
    other = tmp_path / "other.png"
    other.write_bytes(PNG)
    with pytest.raises(StageError, match="incomplete"):
        _retouch(tmp_path, monkeypatch, FakeAlbedo(), params={"retouch": {**RETOUCH, "mask": "../other.png"}})
    assert other.is_file()


class FakeFades(FakeWorker):
    """The edge fade's worker: "fades" writes the models asked ("outs") and the layers; `stopped` once stopped."""

    stopped = False

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        if request["cmd"] == "texture":
            self.calls.append(request)
            return {"texture": [64, 64]}
        if request["cmd"] != "fades":
            return super().call(request)
        self.calls.append(request)
        for out in request.get("outs") or []:
            if out:
                Path(out).write_bytes(b"glTF faded")
        for asked in request["repaints"]:
            if request.get("outs") and asked.get("layer"):
                Path(asked["layer"]).write_bytes(PNG)
        return {"painted": [10] * len(request["repaints"])}

    def stop(self) -> None:
        self.stopped = True


def _repainted(prop_dir: Path, count: int) -> list[str]:
    """A prop folder with an Omni-Texture (ot_001) and `count` repaints of it after it (their kept pictures)."""
    steps = [f"omnitexture/ot_{n:03d}/omnitexture.glb" for n in range(1, count + 2)]
    for n, step in enumerate(steps):
        (prop_dir / step).parent.mkdir(parents=True)
        (prop_dir / step).write_bytes(b"glTF")
        if n:
            _kept_repaint(prop_dir, step, n)
    return steps


def _kept_repaint(prop_dir: Path, step: str, seed: int) -> None:
    """What the repaint that made `step` keeps to be put on again: its mask, picture and camera."""
    kept = (prop_dir / step).parent / "retouch"
    kept.mkdir()
    for name in ("mask.png", "projected.png", "answer.png"):
        (kept / name).write_bytes(PNG)
    (kept / "request.json").write_text(json.dumps({"seed": seed, "retouch": {
        key: RETOUCH[key] for key in ("eye", "forward", "up", "tan")}}), encoding="utf-8")


def test_a_change_that_fails_lets_the_kept_worker_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A repaint put on again that fails - here Windows refuses to rename the new folder, a file in it held open -
    leaves nothing of it behind, and the kept worker is let go: what it holds by then is not known (after a
    failed hide of the views, a model that is deleted and whose name the next change gives to another). The next
    change loads its model anew."""
    prop_dir = tmp_path / "prop"
    steps = _repainted(prop_dir, 1)
    block = {"glb": steps[1], "steps": steps, "stale": False, "repaints": [repaint_record(steps[1], 1.0, under=steps[0])]}
    workers: list[FakeFades] = []
    monkeypatch.setattr(omnitexture, "Worker", lambda: workers.append(FakeFades()) or workers[-1])

    def refused(source: Any, target: Any) -> None:
        raise PermissionError(5, "Access is denied")

    try:
        omnitexture.refade(prop_dir, block, steps[1], None)
        with monkeypatch.context() as held:
            held.setattr(os, "replace", refused)
            with pytest.raises(StageError, match="could not be put on again"):
                omnitexture.refade(prop_dir, block, steps[1], 3.0)
        assert omnitexture._fading is None and workers[0].stopped
        assert sorted(p.name for p in (prop_dir / "omnitexture").iterdir()) == ["ot_001", "ot_002"]
        _patch, renamed = omnitexture.refade(prop_dir, block, steps[1], 3.0)
        assert list(renamed) == [steps[1]] and [call["cmd"] for call in workers[1].calls] == ["load", "fades"]
    finally:
        omnitexture.close_fades()


def test_a_repaint_is_put_on_again_with_another_fade_and_the_later_ones_onto_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The page's fade slider on a repaint: the worker alone - no
    Qwen-Image - reads the step before that repaint once and keeps it, with each repaint's sight and picture, for
    the next change (FadeSession): every fade after the first is one "fades" call, the picked repaint with its new
    fade, the later ones with theirs. Every step from that one on is a new folder, the repaint's pictures linked
    into it; the prop's steps and repaints name them, the model shown follows. Picking a repaint makes the worker
    ready and writes nothing. Unused, the worker is stopped (FADE_IDLE)."""
    prop_dir = tmp_path / "prop"
    steps = _repainted(prop_dir, 2)
    block = {"glb": steps[2], "steps": steps, "stale": False,
             "repaints": [repaint_record(steps[1], 1.0, under=steps[0]), repaint_record(steps[2], 2.0, under=steps[1])]}
    workers: list[FakeFades] = []
    monkeypatch.setattr(omnitexture, "Worker", lambda: workers.append(FakeFades()) or workers[-1])
    try:
        # Picked: the model before the repaint loaded, the repaints' sight read - nothing written.
        assert omnitexture.refade(prop_dir, block, steps[1], None) is None
        (worker,) = workers
        load, ready = worker.calls
        assert Path(load["glb"]) == prop_dir / steps[0] and load["plan"] is False and ready["outs"] is None
        assert ready["ahead"] == 0, "the picked repaint's other fades worked out meanwhile"
        assert [asked["fade"] for asked in ready["repaints"]] == [1.0, 2.0]
        assert sorted(p.name for p in (prop_dir / "omnitexture").iterdir()) == ["ot_001", "ot_002", "ot_003", "refade.tmp"]
        # Let go of at x3: one call, the models of the two steps from it on written into new folders.
        patch, renamed = omnitexture.refade(prop_dir, block, steps[1], 3.0)
        new = [f"omnitexture/ot_{n:03d}/omnitexture.glb" for n in (4, 5)]
        assert renamed == {steps[1]: new[0], steps[2]: new[1]}
        assert patch == {"glb": new[1], "steps": [steps[0], *new], "stale": False,
                         "repaints": [repaint_record(new[0], 3.0, under=steps[0]), repaint_record(new[1], 2.0, under=new[0])],
                         "bases": []}
        faded = worker.calls[-1]
        assert faded["cmd"] == "fades" and [asked["fade"] for asked in faded["repaints"]] == [3.0, 2.0]
        first, later = faded["repaints"]
        assert first["eye"] == RETOUCH["eye"] and Path(first["picture"]).name == "projected.png"
        assert "ot_004" in first["layer"] and "ot_005" in later["layer"], "a layer where there was none"
        assert ["ot_004.tmp" in faded["outs"][0], "ot_005.tmp" in faded["outs"][1]] == [True, True]
        for n, fade in ((4, 3.0), (5, 2.0)):
            folder = prop_dir / "omnitexture" / f"ot_{n:03d}"
            assert (folder / "omnitexture.glb").is_file() and not folder.with_name(folder.name + ".tmp").exists()
            assert sorted(p.name for p in (folder / "retouch").iterdir()) == [
                "answer.png", "layer.webp", "mask.png", "projected.png", "request.json"]
            assert json.loads((folder / "retouch" / "request.json").read_text(encoding="utf-8"))["retouch"]["fade"] == fade
            assert os.stat(folder / "retouch" / "mask.png").st_nlink == 2, "linked, not copied"
        assert os.stat(prop_dir / "omnitexture" / "ot_002" / "retouch" / "request.json").st_nlink == 1
        # The next change: the same worker, the model not loaded again; the repaint after it keeps its layer.
        block = {**block, **patch}
        patch, renamed = omnitexture.refade(prop_dir, block, new[0], 5.0)
        assert len(workers) == 1 and [call["cmd"] for call in worker.calls].count("load") == 1
        assert [asked["fade"] for asked in worker.calls[-1]["repaints"]] == [5.0, 2.0]
        assert "layer" not in worker.calls[-1]["repaints"][1], "its layer was there: linked"
        # The newest repaint picked: the same worker serves it (the model before both is loaded); nothing before it
        # is written again.
        block = {**block, **patch}
        patch, _renamed = omnitexture.refade(prop_dir, block, block["steps"][2], 4.0)
        assert len(workers) == 1 and worker.calls[-1]["outs"][0] is None and worker.calls[-1]["outs"][1]
        assert worker.calls[-1]["ahead"] == 1
        # Unused: stopped (here at once).
        monkeypatch.setattr(omnitexture, "FADE_IDLE", 0.05)
        block = {**block, **patch}
        omnitexture.refade(prop_dir, block, block["steps"][2], None)
        time.sleep(0.4)
        assert worker.stopped and omnitexture._fading is None
        assert not (prop_dir / "omnitexture" / "refade.tmp").exists()
        # The fade stays within what the worker takes; the first step is no repaint; one whose pictures are gone,
        # or whose model before it is, cannot be put on again.
        assert omnitexture.refade(prop_dir, block, block["steps"][2], 99.0)[0]["repaints"][1]["fade"] \
            == omnitexture.MAX_FADE
        for bad in (steps[0], "omnitexture/ot_077/omnitexture.glb", None):
            with pytest.raises(StageError, match="cannot be faded again"):
                omnitexture.refade(prop_dir, block, bad, 2.0)
        (prop_dir / block["steps"][2]).with_name("retouch").joinpath("projected.png").unlink()
        with pytest.raises(StageError, match="cannot be faded again"):
            omnitexture.refade(prop_dir, block, block["steps"][1], 2.0)
    finally:
        omnitexture.close_fades()


def test_a_fade_of_another_prop_or_after_another_change_loads_its_model_anew(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One worker is kept at a time: a fade of another prop, or of this prop once its repaints changed otherwise
    (a new repaint, or one taken back), stops it and loads the model it needs."""
    workers: list[FakeFades] = []
    monkeypatch.setattr(omnitexture, "Worker", lambda: workers.append(FakeFades()) or workers[-1])
    try:
        one, two = tmp_path / "one", tmp_path / "two"
        steps_one, steps_two = _repainted(one, 1), _repainted(two, 2)
        block_one = {"glb": steps_one[1], "steps": steps_one,
                     "repaints": [repaint_record(steps_one[1], 1.0, under=steps_one[0])]}
        block_two = {"glb": steps_two[2], "steps": steps_two,
                     "repaints": [repaint_record(step, 1.0, under=before) for before, step in zip(steps_two, steps_two[1:])]}
        omnitexture.refade(one, block_one, steps_one[1], 2.0)
        omnitexture.refade(two, block_two, steps_two[2], 2.0)
        assert len(workers) == 2 and workers[0].stopped and not workers[1].stopped
        # the second prop's first repaint: its model before it is not the one loaded
        omnitexture.refade(two, block_two, steps_two[1], None)
        assert len(workers) == 3 and workers[1].stopped
    finally:
        omnitexture.close_fades()
    assert workers[2].stopped


class MortalFades(FakeFades):
    """A fade's worker that is of no use once stopped, as the real one."""

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.stopped:
            raise StageError("The program that paints the texture stopped unexpectedly.")
        return super().call(request)


def _outcome(function: Callable[..., Any], *args: Any) -> Any:
    try:
        return function(*args)
    except StageError as exc:
        return exc


def test_a_fade_under_way_is_not_stopped_by_the_page_readying_another_repaint(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One worker is kept, for one prop: while it writes a fade's models, a repaint of another prop is pointed at
    (the page readies its worker). That waits; the fade is finished first, and neither fails."""
    entered, go = threading.Event(), threading.Event()

    class Slow(MortalFades):
        def call(self, request: dict[str, Any]) -> dict[str, Any]:
            if request["cmd"] == "fades" and request.get("outs") and not entered.is_set():
                entered.set()
                go.wait(10)
            return super().call(request)

    workers: list[Slow] = []
    monkeypatch.setattr(omnitexture, "Worker", lambda: workers.append(Slow()) or workers[-1])
    one, two = tmp_path / "one", tmp_path / "two"
    steps_one, steps_two = _repainted(one, 1), _repainted(two, 1)
    block_one = {"glb": steps_one[1], "steps": steps_one, "repaints": [repaint_record(steps_one[1], under=steps_one[0])]}
    block_two = {"glb": steps_two[1], "steps": steps_two, "repaints": [repaint_record(steps_two[1], under=steps_two[0])]}
    made: dict[str, Any] = {}
    fade = threading.Thread(target=lambda: made.update(fade=_outcome(omnitexture.refade, one, block_one,
                                                                    steps_one[1], 2.0)))
    ready = threading.Thread(target=lambda: made.update(ready=_outcome(omnitexture.refade, two, block_two,
                                                                      steps_two[1], None)))
    try:
        fade.start()
        assert entered.wait(10)
        ready.start()
        time.sleep(0.3)
        assert not workers[0].stopped, "the worker is left alone while it writes the fade's models"
        go.set()
        fade.join(10)
        ready.join(10)
        assert not isinstance(made["fade"], StageError), made["fade"]
        assert made["ready"] is None and len(workers) == 2 and workers[0].stopped and not workers[1].stopped
    finally:
        go.set()
        omnitexture.close_fades()


def test_a_kept_worker_that_stopped_by_itself_is_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The kept worker crashed, or was stopped over a command that took too long: the next fade loads the model
    anew instead of failing on it."""
    workers: list[MortalFades] = []
    monkeypatch.setattr(omnitexture, "Worker", lambda: workers.append(MortalFades()) or workers[-1])
    prop_dir = tmp_path / "prop"
    steps = _repainted(prop_dir, 1)
    block = {"glb": steps[1], "steps": steps, "repaints": [repaint_record(steps[1], under=steps[0])]}
    try:
        omnitexture.refade(prop_dir, block, steps[1], None)
        workers[0].stopped = True
        patch, _renamed = omnitexture.refade(prop_dir, block, steps[1], 2.0)
        assert len(workers) == 2 and patch["repaints"][0]["fade"] == 2.0
    finally:
        omnitexture.close_fades()


# ---- a picture taken off the model (the page's ×) --------------------------------------------------------------------

def test_a_repaint_taken_off_is_one_more_step_and_the_one_before_still_holds_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The page's × on a repaint: the repaints made after it go on again, each with its own fade, onto the model
    it went onto (the fade's worker, no Qwen-Image), into new folders; the model without it is one more step, so
    the step before still holds it (Ctrl+Z) and nothing of it is gone. A fade there fades the repaint put on
    again, in the worker kept. With nothing after it the model under it is the new step, a copy in a folder of its
    own; with no repaint left, a copy of the result, holding its views. Taken off after a Ctrl+Z, the steps ahead
    go, and the records only they held."""
    prop_dir = tmp_path / "prop"
    steps = _repainted(prop_dir, 3)
    records = [repaint_record(step, fade, under=before) for before, step, fade in zip(steps, steps[1:], (1.0, 2.0, 3.0))]
    views = ["omnitexture/ot_001/views/0-front/answer.png"]
    block = {"glb": steps[3], "steps": steps, "repaints": records, "bases": [], "views": views, "stale": False}
    st = [f"omnitexture/ot_{n:03d}/omnitexture.glb" for n in range(5, 9)]
    workers: list[FakeFades] = []
    monkeypatch.setattr(omnitexture, "Worker", lambda: workers.append(FakeFades()) or workers[-1])
    try:
        patch = omnitexture.take_off(prop_dir, {"omnitexture": block}, repaint=steps[2])
        assert patch == {"glb": st[0], "steps": [*steps, st[0]], "stale": False, "bases": [],
                         "repaints": [*records, repaint_record(st[0], 3.0, under=steps[1])]}
        (worker,) = workers
        load, put_on = worker.calls
        assert Path(load["glb"]) == prop_dir / steps[1] and load["plan"] is False, "the model it went onto"
        (later,) = put_on["repaints"]
        assert later["fade"] == 3.0 and "ot_004" in later["picture"] and "ot_005.tmp" in put_on["outs"][0]
        assert put_on["ahead"] is None, "no repaint picked"
        assert all((prop_dir / step).is_file() for step in [*steps, st[0]])
        assert os.stat(prop_dir / "omnitexture" / "ot_005" / "retouch" / "mask.png").st_nlink == 2
        block = {**block, **patch}
        assert omnitexture.parts_of(block, st[0]) == (steps[0], [records[0], patch["repaints"][-1]])
        assert omnitexture.parts_of(block, steps[3]) == (steps[0], records), "Ctrl+Z: the model with it"
        # A fade of the repaint put on again: the worker kept serves it; the steps before keep theirs.
        faded, renamed = omnitexture.refade(prop_dir, block, st[0], 5.0)
        assert renamed == {st[0]: st[1]} and len(workers) == 1 and [c["cmd"] for c in worker.calls].count("load") == 1
        (out,) = worker.calls[-1]["outs"]
        assert "ot_006.tmp" in out and worker.calls[-1]["ahead"] == 0
        assert faded["steps"] == [*steps, st[1]] and faded["repaints"] == [*records, repaint_record(st[1], 5.0, under=steps[1])]
        block = {**block, **faded}
        # The newest taken off: the model under it, a copy of its own (the first repaint's, linked).
        patch = omnitexture.take_off(prop_dir, {"omnitexture": block}, repaint=st[1])
        copy = repaint_record(st[2], 1.0, under=steps[0])
        assert patch["glb"] == st[2] and patch["steps"] == [*steps, st[1], st[2]] and patch["repaints"][-1] == copy
        assert os.stat(prop_dir / st[2]).st_nlink == 2 and (prop_dir / st[2]).parent.joinpath("retouch", "projected.png").is_file()
        block = {**block, **patch}
        # The last repaint taken off: a copy of the result, which holds every view.
        patch = omnitexture.take_off(prop_dir, {"omnitexture": block}, repaint=st[2])
        assert patch["glb"] == st[3] and patch["bases"] == [{"glb": st[3], "views": [
            {"folder": "omnitexture/ot_001/views/0-front", "picture": views[0], "layer": None}]}]
        assert omnitexture.parts_of({**block, **patch}, st[3]) == (st[3], [])
        assert len(workers) == 1, "no worker for a copy"
        # After Ctrl+Z to the first repaint: the steps ahead go, and what only they held.
        back = {**block, **patch, "glb": steps[1]}
        patch = omnitexture.take_off(prop_dir, {"omnitexture": back}, repaint=steps[1])
        assert patch["steps"] == [steps[0], steps[1], patch["glb"]] and patch["repaints"] == [records[0]]
        assert [made["glb"] for made in patch["bases"]] == [patch["glb"]]
        # What is not on the model shown, or no longer kept, is not taken off.
        for bad in ({"repaint": steps[3]}, {"repaint": "omnitexture/ot_077/omnitexture.glb"}, {}, {"view": "nope"}):
            with pytest.raises(StageError, match="no longer on the model shown"):
                omnitexture.take_off(prop_dir, {"omnitexture": back}, **bad)
        (prop_dir / steps[3]).with_name("retouch").joinpath("mask.png").unlink()
        with pytest.raises(StageError, match="can no longer be put on again"):
            omnitexture.take_off(prop_dir, {"omnitexture": {**block, "glb": steps[3]}}, repaint=steps[1])
        (prop_dir / steps[0]).unlink()
        with pytest.raises(StageError, match="no longer kept"):
            omnitexture.take_off(prop_dir, {"omnitexture": {**block, "glb": steps[1]}}, repaint=steps[1])
    finally:
        omnitexture.close_fades()


class FakeLayout(FakeFades, FakeSheetWorker):
    """A worker that lays the views out in `sheets` and answers what taking a view off reads, as the real one:
    each view's window in its sheet and the texture's size at the load, the sheets in the finished model's stats;
    it also puts repaints on again ("fades"). `shift` moves the windows, as another version's layout would."""

    shift = 0

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        answer = super().call(request)
        alone = [len(views) == 1 for views in self.sheets]
        if request["cmd"] == "load":
            windows = {k: [0, 0, 1024, 1024] if one else [10 + 100 * n + self.shift, 20, 90, 200]
                       for views, one in zip(self.sheets, alone) for n, k in enumerate(views)}
            answer = {**answer, "texture": [64, 64], "windows": [windows[k] for k in sorted(windows)]}
        elif request["cmd"] == "finish":
            answer["stats"] = {**answer["stats"], "texture": 64, "sheets": [
                {"side": 1024 if one else self.side, "views": [self.NAMES[k] for k in views]}
                for views, one in zip(self.sheets, alone)]}
        return answer


def _composed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], Path, list[FakeLayout]]:
    """A prop OmniTextured from three views - the front and the front right in one sheet, the back alone - with
    the albedo add-on -> (the prop, its folder, the workers started from here on)."""
    patch, _worker, _ = _paint(tmp_path, monkeypatch, FakeAlbedo(), worker=FakeLayout([[0, 2], [1]], side=1024))
    workers: list[FakeLayout] = []
    monkeypatch.setattr(omnitexture, "Worker",
                        lambda: workers.append(FakeLayout([[0, 2], [1]], side=1024)) or workers[-1])
    return {"mesh": {"glb": "mesh/mesh.glb"}, "omnitexture": {**patch, "stale": False}}, tmp_path / "prop", workers


def test_a_view_taken_off_composes_the_result_again_from_the_others(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The page's × on a painted view: the worker loads the 3D model as the job did - its views, its sheets - the
    view taken off painting nothing ("skip"), and puts each other view's picture on again as the job did, from
    what the job kept (the add-on's answer in the painting's colour; a sheet's whole picture for the views in it):
    no Qwen-Image. Each view's share of the texture and what of it went on are new, beside the new model, which
    is one more step and the record of the views it holds (omnitexture.bases). The last view stays; an Omni-Texture
    whose 3D model was made again, or whose views come out laid out otherwise, is not composed again."""
    prop, prop_dir, workers = _composed(tmp_path, monkeypatch)
    block = prop["omnitexture"]
    front, right, back = (view.rsplit("/", 1)[0] for view in block["views"])
    assert (front, right, back) == tuple(f"omnitexture/ot_001/views/{k}" for k in ("0-front", "2-front_right", "1-back"))
    patch = omnitexture.take_off(prop_dir, prop, view=front)
    made = "omnitexture/ot_002"
    assert patch["glb"] == f"{made}/omnitexture.glb" and patch["steps"] == [ST, patch["glb"]]
    assert patch["repaints"] == [] and patch["stale"] is False
    assert patch["bases"] == [{"glb": patch["glb"], "views": [
        {"folder": folder, "picture": f"{made}/views/{name}/projected.png", "layer": f"{made}/views/{name}/layer.webp"}
        for folder, name in ((right, "2-front_right"), (back, "1-back"))]}]
    assert all((prop_dir / view[key]).is_file() for view in patch["bases"][0]["views"] for key in ("picture", "layer"))
    (worker,) = workers
    load, sheet, alone, finish = worker.calls
    assert (Path(load["glb"]), load["views"], load["together"], load["skip"]) == (prop_dir / "mesh" / "mesh.glb", 3, [1024], [0])
    assert _names(sheet) == {"cmd": "paint", "index": 0, "picture": "sheets/1/albedo.png",
                             "colour": "sheets/1/answer.png", "views": [
                                 {"index": 2, "layer": "views/2-front_right/layer.webp",
                                  "keep": "views/2-front_right/projected.png"}]}
    assert "ot_001" in sheet["picture"] and "ot_002.tmp" in sheet["views"][0]["layer"]
    assert _names(alone) == {"cmd": "project", "index": 1, "picture": "views/1-back/albedo.png",
                             "colour": "views/1-back/answer.png", "layer": "views/1-back/layer.webp",
                             "keep": "views/1-back/projected.png"}
    assert finish["cmd"] == "finish" and "tone" not in finish, "the own texture retinted, as by the job"
    # The model composed again holds the two views; the step before it, all three.
    block = {**block, **patch}
    assert [v["folder"] for v in omnitexture.views_of(block, omnitexture.parts_of(block, block["glb"])[0])] == [right, back]
    assert len(omnitexture.views_of(block, omnitexture.parts_of(block, ST)[0])) == 3
    # The next: the two taken off are left out, from the job's own pictures again; the last one stays.
    again = omnitexture.take_off(prop_dir, {**prop, "omnitexture": block}, view=back)
    assert workers[-1].calls[0]["skip"] == [0, 1] and [c["cmd"] for c in workers[-1].calls] == ["load", "paint", "finish"]
    assert [v["folder"] for v in again["bases"][-1]["views"]] == [right] and len(again["bases"]) == 2
    block = {**block, **again}
    with pytest.raises(StageError, match="only view"):
        omnitexture.take_off(prop_dir, {**prop, "omnitexture": block}, view=right)
    with pytest.raises(StageError, match="no longer on the model shown"):
        omnitexture.take_off(prop_dir, {**prop, "omnitexture": block}, view=back)
    with pytest.raises(StageError, match="made again"):
        omnitexture.take_off(prop_dir, {**prop, "omnitexture": {**prop["omnitexture"], "stale": True}}, view=front)
    monkeypatch.setattr(FakeLayout, "shift", 3)
    with pytest.raises(StageError, match="laid out otherwise"):
        omnitexture.take_off(prop_dir, prop, view=front)
    assert sorted(p.name for p in (prop_dir / "omnitexture").iterdir()) == ["ot_001", "ot_002", "ot_003"], \
        "nothing left of the one refused"


def test_a_view_taken_off_is_laid_out_with_every_side_the_job_could_take(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The layout follows from the sides it may take, not from the largest sheet it came to: the job's stats keep
    the largest ("sheet_side"); a result from before they did is laid out up to its largest sheet."""
    patch, _worker, _ = _paint(tmp_path, monkeypatch, FakeAlbedo(), worker=FakeLayout([[0, 2], [1]], side=1536),
                               card_side=2048)
    workers: list[FakeLayout] = []
    monkeypatch.setattr(omnitexture, "Worker",
                        lambda: workers.append(FakeLayout([[0, 2], [1]], side=1536)) or workers[-1])
    prop = {"mesh": {"glb": "mesh/mesh.glb"}, "omnitexture": {**patch, "stale": False}}
    front = patch["views"][0].rsplit("/", 1)[0]
    omnitexture.take_off(tmp_path / "prop", prop, view=front)
    assert workers[-1].calls[0]["together"] == list(range(1024, 2049, 128))
    stats = {key: value for key, value in patch["stats"].items() if key != "sheet_side"}
    omnitexture.take_off(tmp_path / "prop", {**prop, "omnitexture": {**prop["omnitexture"], "stats": stats}}, view=front)
    assert workers[-1].calls[0]["together"] == list(range(1024, 1537, 128))


def test_the_repaints_go_on_again_onto_the_result_composed_without_the_view(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A view taken off a repainted model: the result is composed again without it, and the repaint goes on again
    onto that (the fade's worker), with its fade; the new step is that repaint's, and the step before still holds
    the view and the repaint as they were."""
    prop, prop_dir, workers = _composed(tmp_path, monkeypatch)
    repaint = "omnitexture/ot_002/omnitexture.glb"
    (prop_dir / repaint).parent.mkdir()
    (prop_dir / repaint).write_bytes(b"glTF repainted")
    _kept_repaint(prop_dir, repaint, 9)
    record = repaint_record(repaint, 2.0, under=ST)
    block = {**prop["omnitexture"], "glb": repaint, "steps": [ST, repaint], "repaints": [record]}
    front = block["views"][0].rsplit("/", 1)[0]
    try:
        patch = omnitexture.take_off(prop_dir, {**prop, "omnitexture": block}, view=front)
    finally:
        omnitexture.close_fades()
    base, again = "omnitexture/ot_003/omnitexture.glb", "omnitexture/ot_004/omnitexture.glb"
    assert patch["glb"] == again and patch["steps"] == [ST, repaint, again]
    assert [made["glb"] for made in patch["bases"]] == [base]
    assert patch["repaints"] == [record, repaint_record(again, 2.0, under=base)]
    composer, fades = workers
    assert composer.calls[0]["skip"] == [0] and Path(fades.calls[0]["glb"]) == prop_dir / base
    assert fades.calls[1]["repaints"][0]["fade"] == 2.0 and "ot_004.tmp" in fades.calls[1]["outs"][0]
    block = {**block, **patch}
    assert omnitexture.parts_of(block, again) == (base, [patch["repaints"][1]])
    assert omnitexture.parts_of(block, repaint) == (ST, [record]) and len(omnitexture.views_of(block, ST)) == 3


# ---- hand painting on a step (the page's marker and clone stamp) -------------------------------------------------

class FakeHandPaint(FakeFades):
    """The worker's "hand_paint" as the real one writes its files: the patch after the paint layer before it (as
    the layer), and the model as made with it (as the step's model), each under another name first, then moved."""

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        if request["cmd"] != "hand_paint":
            return super().call(request)
        self.calls.append(request)
        layer, out = Path(request["layer"]), Path(request["out"])
        held = layer.read_bytes() if layer.is_file() else b""
        writes = [(out, Path(request["model"]).read_bytes() + b" + " + held)]
        if request.get("patch"):                # (without one: the layer as it is over another model)
            held += Path(request["patch"]).read_bytes()
            writes = [(layer, held), (out, Path(request["model"]).read_bytes() + b" + " + held)]
        for path, data in writes:
            path.with_name("moved.tmp").write_bytes(data)
            os.replace(path.with_name("moved.tmp"), path)
        return {"painted": len(held), "texture": [request.get("width"), request.get("height")]}


def _hand_painted(prop_dir: Path, step: str) -> Path:
    """What the hub keeps of a step painted by hand: its paint layer, and its model as made -> the layer."""
    folder = (prop_dir / step).parent
    (folder / projects.UNPAINTED_MODEL).write_bytes(b"glTF as made")
    (folder / projects.HAND_PAINT).write_bytes(b"paint of " + step.encode())
    return folder / projects.HAND_PAINT


def test_a_hand_painting_is_saved_into_its_steps_model_over_the_model_as_made(
        api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """PUT .../omnitexture/paint: the first save keeps the step's model as made (a hard link) and its paint
    layer; every save sends the worker the patch, the layer and the model as made - never the painted one - and
    the step's model is written anew (under another name, then moved: the model as made keeps its bytes). As when
    a repaint changes the model shown: its thumbnail is drawn again and the bake made from it is out of date. A
    step the page stepped away from since is painted all the same; meanwhile Ctrl+Z goes on, but no job, fade or
    picture taken off starts. Refused: no Omni-Texture (404); no kept step, a job of the prop, a fade under way
    (409); too big (413); not the texture's size, or incomplete (422)."""
    from spellcast import server

    app, client, _ = api
    client.post("/api/projects", json={"name": "Demo"})
    hub = app.state.hub
    store = hub.store
    pid = _meshed(client, app, "Barrel")
    first = _give_omnitexture(store, "Demo", pid, stats={"texture": 4})
    prop_dir = store.prop_dir("Demo", pid)
    made = (prop_dir / first).read_bytes()
    store.set_thumb("Demo", pid, "omnitexture", PNG)
    store.apply_stage_patch("Demo", pid, "projection", {"glb": "projection/bake.glb", "stale": False,
                                                        "source_mesh": first})
    workers: list[FakeHandPaint] = []
    monkeypatch.setattr(omnitexture, "Worker", lambda: workers.append(FakeHandPaint()) or workers[-1])
    url = f"/api/projects/Demo/props/{pid}/omnitexture"

    def paint(glb: str, body: bytes = b"\x01" * 64, side: int = 4) -> Any:
        return client.put(f"{url}/paint", params={"glb": glb, "width": side, "height": side}, content=body)

    saved = paint(first)
    assert saved.status_code == 200, saved.text
    folder = (prop_dir / first).parent
    (call,) = workers[0].calls
    assert (Path(call["model"]), Path(call["out"]), Path(call["layer"])) == (
        folder / "unpainted.glb", prop_dir / first, folder / "hand-paint.png")
    assert (call["width"], call["height"]) == (4, 4) and not list(prop_dir.glob(".paint-*")), "the patch let go of"
    assert (folder / "unpainted.glb").read_bytes() == made
    assert (prop_dir / first).read_bytes() == made + b" + " + b"\x01" * 64
    prop = saved.json()
    assert "omnitexture" not in prop["thumbs"] and prop["projection"]["stale"] is True
    assert prop["omnitexture"]["glb"] == first and prop["omnitexture"]["steps"] == []
    # The next save: over the layer, from the model as made again.
    assert paint(first, b"\x02" * 64).status_code == 200
    assert (folder / "hand-paint.png").read_bytes() == b"\x01" * 64 + b"\x02" * 64
    assert (prop_dir / first).read_bytes() == made + b" + " + b"\x01" * 64 + b"\x02" * 64
    assert (folder / "unpainted.glb").read_bytes() == made
    # A step the page stepped away from since (Ctrl+Z after painting it) keeps its painting; the shown one's
    # thumbnail stays, the bake (made from the first) is out of date only because it was made from that file.
    second = _give_omnitexture(store, "Demo", pid, number=2, steps=[first, "omnitexture/ot_002/omnitexture.glb"],
                                repaints=[repaint_record("omnitexture/ot_002/omnitexture.glb", under=first)],
                                stats={"texture": 4})
    assert client.post(f"{url}/step", json={"glb": first}).status_code == 200
    store.set_thumb("Demo", pid, "omnitexture", PNG)
    store.apply_stage_patch("Demo", pid, "projection", {"glb": "projection/bake.glb", "stale": False,
                                                        "source_mesh": first})
    away = paint(second)
    assert away.status_code == 200 and (prop_dir / second).read_bytes().endswith(b"\x01" * 64)
    assert "omnitexture" in away.json()["thumbs"] and away.json()["projection"]["stale"] is False
    # Refused.
    other = client.post("/api/projects/Demo/props", json={"name": "Chest"}).json()["id"]
    assert client.put(f"/api/projects/Demo/props/{other}/omnitexture/paint",
                      params={"glb": first, "width": 4, "height": 4}, content=b"\x01" * 64).status_code == 404
    gone = paint("omnitexture/ot_077/omnitexture.glb")
    assert gone.status_code == 409 and "no longer kept" in gone.text
    assert paint(first, side=8, body=b"\x01" * 256).status_code == 422, "not the texture's size"
    assert paint(first, body=b"\x01" * 63).status_code == 422, "incomplete"
    monkeypatch.setattr(server, "MAX_PAINT_BYTES", 32)
    assert paint(first).status_code == 413
    monkeypatch.undo()
    monkeypatch.setattr(omnitexture, "Worker", lambda: workers.append(FakeHandPaint()) or workers[-1])
    hub.fading.add(("Demo", pid))
    try:
        assert paint(first).status_code == 409, "a fade or a picture taken off under way"
    finally:
        hub.fading.discard(("Demo", pid))
    with monkeypatch.context() as busy:
        busy.setattr(hub.jobs, "active_props", lambda project, stage=None: {pid})
        assert paint(first).status_code == 409, "an Omni-Texture job of the prop"
    # While a painting is being saved: Ctrl+Z goes on; nothing that would read the model before it does.
    hub.painting[("Demo", pid)] += 1
    try:
        assert client.post(f"{url}/step", json={"glb": second}).status_code == 200
        for refused in (client.post(f"{url}/fade", json={"glb": second, "fade": 2.0}),
                        client.post(f"{url}/remove", json={"glb": second, "repaint": second}),
                        client.post("/api/jobs", json={"stage": "omnitexture", "project": "Demo", "prop_ids": [pid]})):
            assert refused.status_code == 409 and "being painted on" in refused.text, refused.text
    finally:
        hub.painting[("Demo", pid)] -= 1


def test_the_fades_worker_is_let_go_when_the_model_it_loaded_is_painted(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The edge fade's kept worker holds the model under a repaint as it loaded it: a hand painting saved into
    that model lets it go (the next fade loads the painted one); one saved into a later step leaves it (it reads a
    step's paint layer again whenever that file changed)."""
    prop_dir = tmp_path / "prop"
    steps = _repainted(prop_dir, 2)
    block = {"glb": steps[2], "steps": steps,
             "repaints": [repaint_record(step, under=before) for before, step in zip(steps, steps[1:])]}
    workers: list[FakeHandPaint] = []
    monkeypatch.setattr(omnitexture, "Worker", lambda: workers.append(FakeHandPaint()) or workers[-1])
    patch = tmp_path / "patch.rgba"
    patch.write_bytes(b"\x01" * 64)
    try:
        omnitexture.refade(prop_dir, block, steps[2], None)
        kept = omnitexture._fading
        assert kept is not None and kept.model == steps[1]
        omnitexture.hand_paint(prop_dir, steps[2], patch, 4, 4)
        assert omnitexture._fading is kept and not kept.worker.stopped, "a later step painted: still serves"
        omnitexture.hand_paint(prop_dir, steps[1], patch, 4, 4)
        assert omnitexture._fading is None and kept.worker.stopped, "the model it loaded painted: let go"
        omnitexture.refade(prop_dir, block, steps[2], None)
        assert omnitexture._fading is not None and omnitexture._fading is not kept
    finally:
        omnitexture.close_fades()


def test_a_fade_puts_each_steps_hand_paint_on_again_in_its_new_folder(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fade of an earlier repaint makes the steps from it on again: each painted by hand has its paint layer
    put on again over its repaint (the worker's "fades": "hand_paint"), its model as made written beside its
    model ("unpainted"), and its new folder keeps the layer (a hard link)."""
    prop_dir = tmp_path / "prop"
    steps = _repainted(prop_dir, 2)
    block = {"glb": steps[2], "steps": steps,
             "repaints": [repaint_record(step, under=before) for before, step in zip(steps, steps[1:])]}
    layers = [_hand_painted(prop_dir, step) for step in steps[1:]]
    workers: list[FakeFades] = []
    monkeypatch.setattr(omnitexture, "Worker", lambda: workers.append(FakeFades()) or workers[-1])
    try:
        _patch, renamed = omnitexture.refade(prop_dir, block, steps[1], 3.0)
    finally:
        omnitexture.close_fades()
    asked = workers[0].calls[-1]["repaints"]
    assert [Path(ask["hand_paint"]) for ask in asked] == layers
    for old, ask in zip(steps[1:], asked):
        folder = (prop_dir / renamed[old]).parent
        assert Path(ask["unpainted"]) == folder.with_name(folder.name + ".tmp") / "unpainted.glb"
        assert (folder / "hand-paint.png").read_bytes() == b"paint of " + old.encode()
        assert os.stat(folder / "hand-paint.png").st_nlink == 2, "linked, not copied"


def test_a_view_taken_off_lays_the_results_hand_paint_on_the_one_composed_again(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The result composed again without a view gets the hand paint of the result it replaces over it (the
    worker's "finish": "hand_paint", and the model as made beside it), and keeps the layer in its folder."""
    prop, prop_dir, workers = _composed(tmp_path, monkeypatch)
    layer = _hand_painted(prop_dir, ST)
    patch = omnitexture.take_off(prop_dir, prop, view=prop["omnitexture"]["views"][0].rsplit("/", 1)[0])
    finish = workers[0].calls[-1]
    folder = (prop_dir / patch["glb"]).parent
    assert finish["cmd"] == "finish" and Path(finish["hand_paint"]) == layer
    assert Path(finish["unpainted"]) == folder.with_name(folder.name + ".tmp") / "unpainted.glb"
    assert (folder / "hand-paint.png").read_bytes() == layer.read_bytes()


def test_a_repaint_taken_off_takes_its_steps_hand_paint_with_it_and_keeps_the_later_ones(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A repaint taken off: the hand paint of its own step goes with it (its step, kept for Ctrl+Z, still holds
    it), the later steps' goes on again in their new folders; with nothing to put on again, the copy of the model
    under it keeps that model's hand paint and its model as made."""
    prop_dir = tmp_path / "prop"
    steps = _repainted(prop_dir, 3)
    records = [repaint_record(step, under=before) for before, step in zip(steps, steps[1:])]
    block = {"glb": steps[3], "steps": steps, "repaints": records, "bases": [], "stale": False}
    gone, later = _hand_painted(prop_dir, steps[2]), _hand_painted(prop_dir, steps[3])
    workers: list[FakeFades] = []
    monkeypatch.setattr(omnitexture, "Worker", lambda: workers.append(FakeFades()) or workers[-1])
    try:
        patch = omnitexture.take_off(prop_dir, {"omnitexture": block}, repaint=steps[2])
    finally:
        omnitexture.close_fades()
    (asked,) = workers[0].calls[-1]["repaints"]
    assert Path(asked["hand_paint"]) == later and gone.is_file(), "the later step's, not its own"
    assert ((prop_dir / patch["glb"]).parent / "hand-paint.png").read_bytes() == later.read_bytes()
    under = _hand_painted(prop_dir, steps[1])
    copy = omnitexture.take_off(prop_dir, {"omnitexture": {**block, "glb": steps[2]}}, repaint=steps[2])
    folder = (prop_dir / copy["glb"]).parent
    assert (folder / "hand-paint.png").read_bytes() == under.read_bytes()
    assert (folder / "unpainted.glb").read_bytes() == b"glTF as made"


# ---- the eye on a picture (the page's click on its middle) --------------------------------------------------------

def test_a_repaint_hidden_keeps_its_place_and_one_kept_worker_serves_every_change(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The page's eye on a repaint: the worker is asked it "hidden" (nothing of it goes on), its record keeps its
    place and its fade, and the steps from it on are made again in new folders as by a fade - no step is added.
    Shown again the same way. The worker kept on the model before it serves what follows - another repaint's eye,
    a fade, a repaint taken off - without loading again; one that is so already is left alone. A copy of a
    hidden repaint's model (the repaint after it taken off) is hidden too."""
    prop_dir = tmp_path / "prop"
    steps = _repainted(prop_dir, 3)
    records = [repaint_record(step, fade, under=before) for before, step, fade in zip(steps, steps[1:], (1.0, 2.0, 3.0))]
    block = {"glb": steps[3], "steps": steps, "repaints": records, "bases": [], "stale": False}
    workers: list[FakeFades] = []
    monkeypatch.setattr(omnitexture, "Worker", lambda: workers.append(FakeFades()) or workers[-1])

    def hidden(patch: dict[str, Any]) -> list[bool]:
        return [made.get("hidden", False) for made in omnitexture.parts_of(patch, patch["glb"])[1]]

    try:
        patch, renamed = omnitexture.refade(prop_dir, block, steps[1], None, hidden=True)
        new = [f"omnitexture/ot_{n:03d}/omnitexture.glb" for n in (5, 6, 7)]
        assert renamed == dict(zip(steps[1:], new))
        assert patch["glb"] == new[2] and patch["steps"] == [steps[0], *new], "no step added"
        assert patch["repaints"] == [repaint_record(new[0], 1.0, under=steps[0], hidden=True),
                                     repaint_record(new[1], 2.0, under=new[0]), repaint_record(new[2], 3.0, under=new[1])]
        (worker,) = workers
        load, put_on = worker.calls
        assert Path(load["glb"]) == prop_dir / steps[0] and put_on["ahead"] is None and all(put_on["outs"])
        assert [(asked["hidden"], asked["fade"]) for asked in put_on["repaints"]] == [(True, 1.0), (False, 2.0), (False, 3.0)]
        assert "layer" not in put_on["repaints"][0] or "ot_005" in put_on["repaints"][0]["layer"]
        block = {**block, **patch}
        assert omnitexture.refade(prop_dir, block, new[0], None, hidden=True) is None, "hidden already"
        # Pointed at while hidden: made ready to be shown again - asked as shown, none of the slider's fades ahead.
        assert omnitexture.refade(prop_dir, block, new[0], None) is None
        ready = worker.calls[-1]
        assert ready["outs"] is None and ready["ahead"] is None and ready["repaints"][0]["hidden"] is False
        # The newest hidden too: the worker kept on the model before the first writes that step alone.
        patch, renamed = omnitexture.refade(prop_dir, block, new[2], None, hidden=True)
        assert list(renamed) == [new[2]] and hidden({**block, **patch}) == [True, False, True]
        assert [bool(out) for out in worker.calls[-1]["outs"]] == [False, False, True]
        assert [asked["hidden"] for asked in worker.calls[-1]["repaints"]] == [True, False, True]
        block = {**block, **patch}
        # A fade of the one between them leaves both hidden; the first shown again.
        patch, _renamed = omnitexture.refade(prop_dir, block, new[1], 4.0)
        block = {**block, **patch}
        assert hidden(block) == [True, False, True] and block["repaints"][1]["fade"] == 4.0
        patch, _renamed = omnitexture.refade(prop_dir, block, block["steps"][1], None, hidden=False)
        block = {**block, **patch}
        assert hidden(block) == [False, False, True] and worker.calls[-1]["repaints"][0]["hidden"] is False
        assert len(block["steps"]) == 4 and all((prop_dir / step).is_file() for step in block["steps"])
        # The one between taken off: the newest goes on again onto the first, hidden as it is - the same worker.
        first = block["steps"][1]
        taken = omnitexture.take_off(prop_dir, {"omnitexture": block}, repaint=block["steps"][2])
        assert [asked["hidden"] for asked in worker.calls[-1]["repaints"]] == [False, True]
        assert [bool(out) for out in worker.calls[-1]["outs"]] == [False, True]
        assert taken["steps"] == [*block["steps"], taken["glb"]]
        assert taken["repaints"][-1] == repaint_record(taken["glb"], 3.0, under=first, hidden=True)
        assert len(workers) == 1 and [call["cmd"] for call in worker.calls].count("load") == 1
        # The newest taken off with nothing to put on again: the model under it once more, hidden as that one is.
        block = {**block, **taken}
        patch, _renamed = omnitexture.refade(prop_dir, block, first, None, hidden=True)
        block = {**block, **patch}
        copied = omnitexture.take_off(prop_dir, {"omnitexture": block}, repaint=block["glb"])
        assert copied["repaints"][-1] == repaint_record(copied["glb"], 1.0, under=steps[0], hidden=True)
        for bad in (steps[0], "omnitexture/ot_077/omnitexture.glb", None):
            with pytest.raises(StageError, match="cannot be hidden or shown"):
                omnitexture.refade(prop_dir, block, bad, None, hidden=True)
    finally:
        omnitexture.close_fades()


def test_the_painted_views_hidden_give_way_to_the_3d_models_own_texture_and_come_back(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The page's eye on a painted view - on all of them at once: hidden, the result they lie on gives way to a
    copy of the 3D model (a copy: the 3D stage writes its file anew in place), recorded with the views it hides
    and the model that holds them; the repaints go on again onto it, each as it lies. No step is added. What was
    painted by hand goes over the other model (the worker's "hand_paint" without a patch). Shown again, the
    result as it was made is back (linked), under what is painted by hand now; the worker kept on the one model
    is given the other's texture instead of loading it. While they are hidden no view is taken off; an
    Omni-Texture whose 3D model was made again does not hide them; a result of the model's own texture has none."""
    prop_dir = tmp_path / "prop"
    steps = _repainted(prop_dir, 1)
    (prop_dir / "mesh").mkdir()
    (prop_dir / "mesh" / "mesh.glb").write_bytes(b"glTF mesh")
    views = [f"omnitexture/ot_001/views/{name}/answer.png" for name in ("0-front", "1-back")]
    listed = [{"folder": view.rsplit("/", 1)[0], "picture": view, "layer": None} for view in views]
    block = {"glb": steps[1], "steps": steps, "repaints": [repaint_record(steps[1], 2.0, under=steps[0])], "bases": [],
             "views": views, "source_mesh": "mesh/mesh.glb", "stale": False}
    prop = {"mesh": {"glb": "mesh/mesh.glb"}, "omnitexture": block}
    workers: list[FakeHandPaint] = []
    monkeypatch.setattr(omnitexture, "Worker", lambda: workers.append(FakeHandPaint()) or workers[-1])
    try:
        assert omnitexture.show_views(prop_dir, prop, hidden=False) is None, "shown already"
        patch, renamed = omnitexture.show_views(prop_dir, prop, hidden=True)
        bare, onto = (f"omnitexture/ot_{n:03d}/omnitexture.glb" for n in (3, 4))
        assert renamed == {steps[0]: bare, steps[1]: onto}
        assert patch["glb"] == onto and patch["steps"] == [bare, onto], "no step added"
        assert patch["bases"] == [{"glb": bare, "views": listed, "hidden": True, "shown": steps[0]}]
        assert patch["repaints"] == [repaint_record(onto, 2.0, under=bare)]
        assert (prop_dir / bare).read_bytes() == b"glTF mesh" and os.stat(prop_dir / bare).st_nlink == 1, "a copy"
        (worker,) = workers
        assert Path(worker.calls[0]["glb"]) == prop_dir / bare and "ot_004.tmp" in worker.calls[1]["outs"][0]
        hidden = {**prop, "omnitexture": {**block, **patch}}
        assert omnitexture.views_hidden(hidden["omnitexture"], bare) and not omnitexture.views_hidden(block, steps[0])
        assert omnitexture.views_of(hidden["omnitexture"], bare) == listed
        assert omnitexture.show_views(prop_dir, hidden, hidden=True) is None, "hidden already"
        with pytest.raises(StageError, match="Show them again first"):
            omnitexture.take_off(prop_dir, hidden, view=listed[0]["folder"])
        # Painted by hand while they are hidden; shown again: the result as made, linked, with that paint over it.
        layer = _hand_painted(prop_dir, bare)
        patch, renamed = omnitexture.show_views(prop_dir, hidden, hidden=False)
        back, again = (f"omnitexture/ot_{n:03d}/omnitexture.glb" for n in (5, 6))
        assert renamed == {bare: back, onto: again} and patch["steps"] == [back, again]
        assert patch["bases"] == [{"glb": back, "views": listed}] and patch["repaints"] == [repaint_record(again, 2.0, under=back)]
        folder = (prop_dir / back).parent
        assert (folder / "unpainted.glb").read_bytes() == b"glTF" and os.stat(folder / "unpainted.glb").st_nlink == 2
        assert (folder / "hand-paint.png").read_bytes() == layer.read_bytes()
        assert (prop_dir / back).read_bytes() == b"glTF + " + layer.read_bytes()
        (painted,) = [call for made in workers for call in made.calls if call["cmd"] == "hand_paint"]
        assert "patch" not in painted and Path(painted["layer"]) == layer
        # The worker kept on the model without the views is given the other texture: the surface is the same.
        assert [call["cmd"] for call in worker.calls] == ["load", "fades", "texture", "fades"]
        assert Path(worker.calls[2]["glb"]) == prop_dir / back and omnitexture._fading.model == back
        # Hidden again, from a result painted by hand: its model as made is the one named to come back.
        shown = {**prop, "omnitexture": {**block, **patch}}
        patch, _renamed = omnitexture.show_views(prop_dir, shown, hidden=True)
        (record,) = patch["bases"]
        assert record["shown"] == "omnitexture/ot_005/unpainted.glb" and record["hidden"] is True
        folder = (prop_dir / record["glb"]).parent
        assert (folder / "unpainted.glb").read_bytes() == b"glTF mesh"
        assert (prop_dir / record["glb"]).read_bytes() == b"glTF mesh + " + layer.read_bytes()
        with pytest.raises(StageError, match="made again"):
            omnitexture.show_views(prop_dir, {**prop, "omnitexture": {**block, "stale": True}}, hidden=True)
        with pytest.raises(StageError, match="no longer on the model shown"):
            omnitexture.show_views(prop_dir, {**prop, "omnitexture": {**block, "views": []}}, hidden=True)
        (prop_dir / steps[1]).with_name("retouch").joinpath("mask.png").unlink()
        with pytest.raises(StageError, match="can no longer be put on again"):
            omnitexture.show_views(prop_dir, prop, hidden=True)
        assert sorted(p.name for p in (prop_dir / "omnitexture").glob("ot_*")) == [f"ot_{n:03d}" for n in range(1, 9)], \
            "nothing left of the ones refused"
    finally:
        omnitexture.close_fades()


def test_the_pages_eye_hides_a_picture_and_what_is_hidden_stays_so(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST .../omnitexture/hide: a repaint, or the painted views all at once, hidden or shown again -> the prop,
    its steps made again and which is which; the bake made from the model it showed is out of date. What is
    hidden is so when the project is read again, and the result the hidden views lie on stays when older folders
    are tidied away. Refused without "hidden", while the page shows another model, and while another change of
    the prop's Omni-Texture is being made."""
    app, client, _ = api
    client.post("/api/projects", json={"name": "Demo"})
    hub = app.state.hub
    store = hub.store
    pid = _meshed(client, app, "Barrel")
    front = "omnitexture/ot_001/views/0-front"
    first = _give_omnitexture(store, "Demo", pid, views=[f"{front}/answer.png"])
    prop_dir = store.prop_dir("Demo", pid)
    second = "omnitexture/ot_002/omnitexture.glb"
    (prop_dir / second).parent.mkdir()
    (prop_dir / second).write_bytes(b"glTF repainted")
    _kept_repaint(prop_dir, second, 1)
    store.apply_stage_patch("Demo", pid, "omnitexture", {"glb": second, "steps": [first, second],
                                                         "repaints": [repaint_record(second, under=first)]})
    store.apply_stage_patch("Demo", pid, "projection", {"glb": "projection/bake.glb", "stale": False})
    monkeypatch.setattr(omnitexture, "Worker", FakeFades)
    url = f"/api/projects/Demo/props/{pid}/omnitexture/hide"
    try:
        assert client.post(url, json={"glb": second, "repaint": second}).status_code == 422
        assert client.post(url, json={"glb": first, "repaint": second, "hidden": True}).status_code == 409
        hub.fading.add(("Demo", pid))
        try:
            busy = client.post(url, json={"glb": second, "repaint": second, "hidden": True})
            assert busy.status_code == 409 and "being changed" in busy.text
        finally:
            hub.fading.discard(("Demo", pid))
        made = client.post(url, json={"glb": second, "repaint": second, "hidden": True})
        assert made.status_code == 200 and not hub.fading
        third = "omnitexture/ot_003/omnitexture.glb"
        block = made.json()["prop"]["omnitexture"]
        assert made.json()["renamed"] == {second: third} and block["glb"] == third and block["steps"] == [first, third]
        assert block["repaints"] == [repaint_record(third, under=first, hidden=True)]
        assert made.json()["prop"]["projection"]["stale"] is True
        same = client.post(url, json={"glb": third, "repaint": third, "hidden": True})
        assert same.status_code == 200 and same.json()["renamed"] == {} and same.json()["prop"]["omnitexture"] == block
        views = client.post(url, json={"glb": third, "views": True, "hidden": True})
        assert views.status_code == 200, views.text
        bare, onto = (f"omnitexture/ot_{n:03d}/omnitexture.glb" for n in (4, 5))
        block = views.json()["prop"]["omnitexture"]
        assert views.json()["renamed"] == {first: bare, third: onto} and block["steps"] == [bare, onto]
        assert block["bases"] == [{"glb": bare, "hidden": True, "shown": first, "views": [
            {"folder": front, "picture": f"{front}/answer.png", "layer": None}]}]
        assert (prop_dir / bare).read_bytes() == (prop_dir / "mesh" / "mesh.glb").read_bytes()
    finally:
        omnitexture.close_fades()
    # Read again from the project's file: as hidden as before. Tidied: the result the views lie on stays.
    again = ProjectStore(store.root).get_prop("Demo", pid)["omnitexture"]
    assert again["bases"] == block["bases"] and again["repaints"] == [repaint_record(onto, under=bare, hidden=True)]
    for number in (6, 7, 8):
        (prop_dir / "omnitexture" / f"ot_{number:03d}").mkdir()
    store.apply_stage_patch("Demo", pid, "omnitexture", {"glb": onto})
    assert sorted(p.name for p in (prop_dir / "omnitexture").glob("ot_*")) == ["ot_001", "ot_004", "ot_005", "ot_007", "ot_008"]


def test_a_fade_and_a_job_of_one_prop_never_run_together(api: Any, paint: FakePaint,  # noqa: F811
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """Both write the prop's next result folder. While a repaint is being faded an Omni-Texture job of the prop is
    refused; while a job of it waits or runs, no worker is made ready for a fade (the page pointing at a repaint)."""
    app, client, _ = api
    client.post("/api/projects", json={"name": "Demo"})
    pid = _meshed(client, app, "Barrel")
    hub = app.state.hub
    first = _give_omnitexture(hub.store, "Demo", pid)
    hub.fading.add(("Demo", pid))
    try:
        refused = client.post("/api/jobs", json={"stage": "omnitexture", "project": "Demo", "prop_ids": [pid]})
        assert refused.status_code == 409 and "being changed" in refused.text
    finally:
        hub.fading.discard(("Demo", pid))
    readied: list[Any] = []
    monkeypatch.setattr(omnitexture, "refade", lambda *args: readied.append(args))
    monkeypatch.setattr(hub.jobs, "active_props", lambda project, stage=None: {pid})
    refused = client.post(f"/api/projects/Demo/props/{pid}/omnitexture/fade", json={"glb": first, "prepare": True})
    assert refused.status_code == 409 and not readied


def test_the_page_the_hub_and_the_worker_agree_on_the_fade_and_on_the_clay() -> None:
    """Three programs in two languages hold the fade slider's range - the page offers it, the hub checks it, the
    worker works its values out ahead - and two the clay grey of a layer (the worker lays a picture's share over
    it, the page's viewer shows the model in it). Changed in one place alone they would drift apart unnoticed."""
    def read(*parts: str) -> str:
        return Path(omnitexture.__file__).parents[1].joinpath(*parts).read_text(encoding="utf-8")

    def number(text: str, pattern: str) -> float:
        return float(re.search(pattern, text, re.MULTILINE).group(1))

    worker, page = read("launchers", "omnitexture_repaint.py"), read("static", "js", "stages", "rules.js")
    assert number(worker, r"^MAX_FADE = ([\d.]+)$") == number(page, r"^export const MAX_FADE = ([\d.]+);$") \
        == omnitexture.MAX_FADE
    assert number(worker, r"^FADE_STEP = ([\d.]+)$") == number(page, r"^export const FADE_STEP = ([\d.]+);$")
    clay = re.search(r"^CLAY = \((\d+), (\d+), (\d+)\)", read("launchers", "omnitexture_tone.py"), re.MULTILINE)
    viewer = re.search(r"^const CLAY = \{ color: 0x([0-9a-f]{6}),", read("static", "js", "viewer3d.js"), re.MULTILINE)
    assert "{:02x}{:02x}{:02x}".format(*(int(v) for v in clay.groups())) == viewer.group(1)
    # ... and one how deep an inverse lens sees: the page's viewer draws no deeper than the worker looks.
    assert number(read("launchers", "omnitexture_views.py"), r"^LENS_REACH = ([\d.]+)$") \
        == number(read("static", "js", "viewer3d.js"), r"^export const LENS_REACH = ([\d.]+);$")


def test_the_mask_is_kept_for_its_job_and_ctrl_z_steps_between_the_results(api: Any) -> None:
    """The page's routes: the mask of a repaint (a PNG) is kept in the prop's folder for the job that follows, only
    for the Omni-Texture the page shows; Ctrl+Z and Ctrl+Y have the prop show another of its steps - the bake made
    from the one it showed is then out of date - and the folders the steps and the painted views lie in are kept
    when older results are tidied away."""
    app, client, _ = api
    client.post("/api/projects", json={"name": "Demo"})
    store = app.state.hub.store
    pid = _meshed(client, app, "Barrel")
    project = "Demo"
    first = _give_omnitexture(store, project, pid, number=1, steps=[],
                               views=["omnitexture/ot_001/views/0-front/answer.png"])
    url = f"/api/projects/{project}/props/{pid}/omnitexture"
    sent = client.put(f"{url}/mask", params={"glb": first}, content=PNG)
    assert sent.status_code == 200 and sent.json() == {"mask": "omnitexture/mask.png"}
    assert (store.prop_dir(project, pid) / "omnitexture" / "mask.png").read_bytes() == PNG
    assert client.put(f"{url}/mask", params={"glb": "omnitexture/ot_009/omnitexture.glb"}, content=PNG).status_code == 409
    assert client.put(f"{url}/mask", params={"glb": first}, content=b"not a picture").status_code == 422
    # Three repaints: each a new folder; the first result's folder stays (the views are its own), and the steps'.
    glbs = [first]
    for number in (2, 3, 4):
        glbs.append(_give_omnitexture(store, project, pid, number=number, steps=[*glbs, f"omnitexture/ot_{number:03d}/omnitexture.glb"],
                                       views=["omnitexture/ot_001/views/0-front/answer.png"]))
    folder = store.prop_dir(project, pid) / "omnitexture"
    assert sorted(p.name for p in folder.iterdir() if p.is_dir()) == ["ot_001", "ot_002", "ot_003", "ot_004"]
    store.apply_stage_patch(project, pid, "projection", {"glb": "projection/bake.glb", "stale": False})
    back = client.post(f"{url}/step", json={"glb": glbs[2]})
    assert back.status_code == 200 and back.json()["omnitexture"]["glb"] == glbs[2]
    assert back.json()["omnitexture"]["steps"] == glbs and back.json()["projection"]["stale"] is True
    assert client.post(f"{url}/step", json={"glb": glbs[3]}).json()["omnitexture"]["glb"] == glbs[3]
    assert client.post(f"{url}/step", json={"glb": "omnitexture/ot_077/omnitexture.glb"}).status_code == 404
    # A repaint's fade (the page's slider): the worker's part stands in here; the prop then shows the steps as
    # they were made again, and the page is told which is which. Only a number from 1 to 8, never beside a fade
    # of the same prop that is still being made.
    from spellcast.adapters import omnitexture as adapter

    asked: list[tuple[str, float]] = []

    def refade(prop_dir: Path, block: dict[str, Any], glb: str, fade: float) -> tuple[dict[str, Any], dict[str, str]]:
        asked.append((glb, fade))
        assert client.post(f"{url}/step", json={"glb": glbs[2]}).status_code == 409, "not while it is being faded"
        made = "omnitexture/ot_006/omnitexture.glb"
        (prop_dir / made).parent.mkdir(parents=True)
        (prop_dir / made).write_bytes(b"glTF")
        steps = [made if step == glb else step for step in block["steps"]]
        return ({"glb": made, "steps": steps, "stale": False, "repaints": [
            {"glb": made, "picture": "omnitexture/ot_006/retouch/projected.png", "layer": None, "fade": fade}]},
            {glb: made})

    adapter_refade, adapter.refade = adapter.refade, refade
    try:
        faded = client.post(f"{url}/fade", json={"glb": glbs[3], "fade": 2.5})
    finally:
        adapter.refade = adapter_refade
    assert faded.status_code == 200 and asked == [(glbs[3], 2.5)]
    assert faded.json()["renamed"] == {glbs[3]: "omnitexture/ot_006/omnitexture.glb"}
    block = faded.json()["prop"]["omnitexture"]
    assert block["glb"] == "omnitexture/ot_006/omnitexture.glb" and block["steps"][-1] == block["glb"]
    assert block["repaints"] == [{"glb": block["glb"], "picture": "omnitexture/ot_006/retouch/projected.png",
                                  "layer": None, "fade": 2.5}]
    for bad in (0.5, 9, "2", None, True):
        assert client.post(f"{url}/fade", json={"glb": glbs[3], "fade": bad}).status_code == 422, bad
    assert client.post(f"{url}/step", json={"glb": glbs[2]}).status_code == 200, "free again once it is made"
    # Repaints from before they were recorded are told from the steps: every step after the first, shown by its mask.
    store.apply_stage_patch(project, pid, "omnitexture", {"repaints": []})
    told = store.get_prop(project, pid)["omnitexture"]
    assert [made["glb"] for made in told["repaints"]] == told["steps"][1:]
    assert told["repaints"][0]["picture"].endswith("/retouch/projected.png")         and told["repaints"][0]["layer"].endswith("/retouch/mask.png") and told["repaints"][0]["fade"] == 1.0
    # An Omni-Texture made anew starts the steps afresh: the folders of the old ones go, but for the newest two.
    _give_omnitexture(store, project, pid, number=7, steps=[], repaints=[], views=[])
    assert sorted(p.name for p in folder.iterdir() if p.is_dir()) == ["ot_006", "ot_007"]


def test_the_page_takes_a_picture_off_and_ctrl_z_and_ctrl_y_step_over_it(api: Any) -> None:
    """POST .../omnitexture/remove: the model shown without that picture becomes the prop's next step (here its
    only repaint, so a copy of the result, which holds the views: no worker), and the bake made from the model
    it showed is out of date; Ctrl+Z (POST .../step) shows the model with it again, Ctrl+Y the one without.
    Refused while the page shows another model, for a picture not on the model shown, and while another change
    of the prop's Omni-Texture is being made."""
    app, client, _ = api
    client.post("/api/projects", json={"name": "Demo"})
    hub = app.state.hub
    store = hub.store
    pid = _meshed(client, app, "Barrel")
    front = "omnitexture/ot_001/views/0-front"
    first = _give_omnitexture(store, "Demo", pid, views=[f"{front}/answer.png"])
    prop_dir = store.prop_dir("Demo", pid)
    second = "omnitexture/ot_002/omnitexture.glb"
    (prop_dir / second).parent.mkdir()
    (prop_dir / second).write_bytes(b"glTF repainted")
    _kept_repaint(prop_dir, second, 1)
    record = repaint_record(second, under=first)
    store.apply_stage_patch("Demo", pid, "omnitexture", {"glb": second, "steps": [first, second], "repaints": [record]})
    store.apply_stage_patch("Demo", pid, "projection", {"glb": "projection/bake.glb", "stale": False})
    url = f"/api/projects/Demo/props/{pid}/omnitexture"
    shown_before = client.post(f"{url}/remove", json={"glb": first, "repaint": second})
    assert shown_before.status_code == 409 and "changed meanwhile" in shown_before.text
    assert client.post(f"{url}/remove", json={"glb": second, "repaint": first}).status_code == 409
    hub.fading.add(("Demo", pid))
    try:
        busy = client.post(f"{url}/remove", json={"glb": second, "repaint": second})
        assert busy.status_code == 409 and "being changed" in busy.text
    finally:
        hub.fading.discard(("Demo", pid))
    taken = client.post(f"{url}/remove", json={"glb": second, "repaint": second})
    assert taken.status_code == 200 and not hub.fading
    prop = taken.json()["prop"]
    block, third = prop["omnitexture"], "omnitexture/ot_003/omnitexture.glb"
    assert block["glb"] == third and block["steps"] == [first, second, third] and block["repaints"] == [record]
    assert block["bases"] == [{"glb": third, "views": [{"folder": front, "picture": f"{front}/answer.png",
                                                         "layer": None}]}]
    assert (prop_dir / third).read_bytes() == (prop_dir / first).read_bytes() and prop["projection"]["stale"] is True
    back = client.post(f"{url}/step", json={"glb": second}).json()["omnitexture"]
    assert back["glb"] == second and omnitexture.parts_of(back, second) == (first, [record])
    assert client.post(f"{url}/step", json={"glb": third}).json()["omnitexture"]["glb"] == third


def test_older_repaints_went_onto_the_step_before_and_what_the_steps_hold_is_kept(tmp_path: Path) -> None:
    """Repaints recorded before "under" was went onto the step before theirs (one made after another); a result
    composed again is read as written. Without records every step after the first is a repaint (from before they
    were recorded), unless a result was composed again. When older folders are tidied away, those the steps'
    models are made of stay: the repaints', what they went onto, the results composed again and their views'
    pictures."""
    root = tmp_path / "projects"
    steps = [f"omnitexture/ot_{n:03d}/omnitexture.glb" for n in (1, 2, 3)]
    composed = {"glb": "omnitexture/ot_004/omnitexture.glb", "views": [
        {"folder": "omnitexture/ot_001/views/1-back", "picture": "omnitexture/ot_005/views/1-back/projected.png",
         "layer": 7}, {"nope": True}]}
    blocks = {
        "p_000001": {"glb": steps[2], "steps": steps, "repaints": [
            {"glb": steps[1], "picture": "a.png"}, {"glb": steps[2], "picture": "b.png", "under": None},
            {"glb": "omnitexture/ot_009/omnitexture.glb", "picture": "c.png"}], "bases": [composed, "x"]},
        "p_000002": {"glb": steps[1], "steps": steps[:2]},
        "p_000003": {"glb": composed["glb"], "steps": [steps[0], composed["glb"]], "bases": [composed]},
    }
    (root / "P").mkdir(parents=True)
    (root / "P" / "project.json").write_text(json.dumps({"name": "P", "props": [
        {"id": pid, "name": pid, "omnitexture": block} for pid, block in blocks.items()]}), encoding="utf-8")
    store = ProjectStore(root)
    one, two, three = (store.get_prop("P", pid)["omnitexture"] for pid in blocks)
    assert [(r["glb"], r["under"]) for r in one["repaints"]] == [
        (steps[1], steps[0]), (steps[2], None), ("omnitexture/ot_009/omnitexture.glb", None)]
    assert one["bases"] == [{**composed, "views": [{**composed["views"][0], "layer": None}]}]
    assert [(r["glb"], r["under"]) for r in two["repaints"]] == [(steps[1], steps[0])]
    assert three["repaints"] == [] and three["bases"] == one["bases"]
    prop_dir = store.prop_dir("P", "p_000001")
    for n in range(1, 10):
        (prop_dir / "omnitexture" / f"ot_{n:03d}").mkdir(parents=True)
    block = {"glb": "omnitexture/ot_006/omnitexture.glb", "steps": [steps[0], "omnitexture/ot_006/omnitexture.glb"],
             "views": ["omnitexture/ot_001/views/1-back/answer.png"], "bases": [composed],
             "repaints": [repaint_record("omnitexture/ot_006/omnitexture.glb", under=composed["glb"])]}
    store._tidy_omnitextures("P", "p_000001", block)
    assert sorted(p.name for p in (prop_dir / "omnitexture").iterdir()) == [
        "ot_001", "ot_004", "ot_005", "ot_006", "ot_008", "ot_009"]


def _st_prop(root: Path, name: str) -> Path:
    prop_dir = root / name
    (prop_dir / "mesh").mkdir(parents=True)
    (prop_dir / "mesh" / "mesh.glb").write_bytes(b"glTF of " + name.encode())
    return prop_dir


def _st_ctx(prop_dir: Path, **fields: Any) -> StageContext:
    return StageContext(project="P", prop_id=prop_dir.name, prop_dir=prop_dir, base_url="http://qwen",
                        client=None, settings={}, report=lambda f, m: None, cancelled=lambda: False,  # type: ignore[arg-type]
                        stage="omnitexture", **fields)


PROP = {"mesh": {"glb": "mesh/mesh.glb"}}


def test_the_next_prop_is_read_while_the_last_view_is_painted_and_draws_while_this_one_finishes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """At the last view of a prop a second worker reads the job's next prop's 3D model (ctx.upcoming) while
    Qwen-Image paints; once the last albedo is in the prop lets go of Qwen-Image (ctx.release) before its last
    projection and its GLB, and the next prop's run takes the waiting worker: no second start, no second load."""
    events: list[tuple[Any, ...]] = []
    made = _fakes(monkeypatch, events)
    a, b = _st_prop(tmp_path, "a"), _st_prop(tmp_path, "b")

    async def main() -> tuple[dict[str, Any], dict[str, Any]]:
        b_turn = asyncio.Event()

        def release(words: str) -> None:
            events.append(("release", words))
            b_turn.set()

        async def upcoming() -> tuple[dict[str, Any], Path]:
            return PROP, b

        first = asyncio.create_task(omnitexture.run(_st_ctx(a, release=release, upcoming=upcoming), PROP, {}))
        await b_turn.wait()
        second = await omnitexture.run(_st_ctx(b), PROP, {})
        return await first, second

    patch_a, patch_b = asyncio.run(main())
    assert len(made) == 2, "one worker per prop, the second started by the first prop"
    spare = [e for e in events if e[:2] == ("worker", 1)]
    assert [e[2:] for e in spare] == [("load", None), ("view", 0), ("project", 0), ("view", 1), ("project", 1),
                                      ("finish", None)]
    assert made[1].work == b / "omnitexture" / "ot_001.tmp" / "views"
    painted = max(i for i, e in enumerate(events) if e == ("generated", "a"))
    assert events.index(("worker", 1, "load", None)) < painted, "read while the last view is painted"
    last_albedo = max(i for i, e in enumerate(events) if e == ("albedo", "a"))
    assert last_albedo < events.index(("release", omnitexture.FINISHING)) < events.index(("worker", 0, "project", 1))
    for prop_dir, patch in ((a, patch_a), (b, patch_b)):
        assert (prop_dir / patch["glb"]).is_file() and patch["glb"] == "omnitexture/ot_001/omnitexture.glb"
    assert omnitexture._spare is None


def test_no_worker_waits_for_a_prop_that_does_not_take_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The next prop's worker starts only with the RAM free for two (SPARE_MIN_RAM_MB); one made for a 3D model that
    changed before its turn (or for another prop) is stopped and the prop reads its model itself."""
    events: list[tuple[Any, ...]] = []
    made = _fakes(monkeypatch, events, free_mb=omnitexture.SPARE_MIN_RAM_MB - 1)
    a, b = _st_prop(tmp_path, "a"), _st_prop(tmp_path, "b")

    async def upcoming() -> tuple[dict[str, Any], Path]:
        return PROP, b

    asyncio.run(omnitexture.run(_st_ctx(a, upcoming=upcoming), PROP, {}))
    assert len(made) == 1 and omnitexture._spare is None, "short of RAM: nothing started early"

    monkeypatch.setattr(omnitexture, "free_ram_mb", lambda: 64_000)

    async def main() -> None:
        await omnitexture.run(_st_ctx(a, upcoming=upcoming), PROP, {})
        assert len(made) == 3 and omnitexture._spare is not None
        (b / "mesh" / "mesh.glb").write_bytes(b"glTF of b, made again")       # its turn finds another model
        await omnitexture.run(_st_ctx(b), PROP, {})
        await asyncio.sleep(0.1)                                              # the spare's stop, in a thread

    asyncio.run(main())
    assert len(made) == 4 and ("stopped", 2) in events
    assert [e[2] for e in events if e[:2] == ("worker", 3)][0] == "load", "its own worker, its own load"
    assert omnitexture._spare is None


def test_omnitexture_props_take_turns_on_qwen_in_groups_of_their_image_model(tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """An Omni-Texture job draws each prop with its own Images model; the launcher holds one model and switches at
    every change, so the props go in groups, the one it holds first (the model of its last picture while it runs; a
    new one holds the Turbo), the artist's order inside a group: the model is switched once at most."""
    import httpx

    from spellcast.adapters import qwen
    from spellcast.adapters.qwen import QwenAdapter

    def prop(pid: str, model: str | None) -> dict[str, Any]:
        return {"id": pid, "images": {"settings": {"qwen_model": model} if model else {}}}

    monkeypatch.setattr(qwen, "_holding", "turbo")
    adapter = QwenAdapter(quality="Q4_K_M", text_encoder="UD-Q4_K_XL")
    order = adapter.prop_order
    props = [prop("a", "turbo"), prop("b", "full"), prop("c", None), prop("d", "full"), prop("e", "flux3")]
    assert order("omnitexture", props) == ["a", "c", "e", "b", "d"]      # FLUX.3 props paint with the Turbo
    assert order("images", props) == ["a", "b", "c", "d", "e"]
    full_first = [prop("a", "full"), prop("b", "turbo"), prop("c", None)]
    assert order("omnitexture", full_first) == ["b", "c", "a"], "the Turbo, which a new launcher holds, first"

    async def draw(model: str) -> None:        # a picture request answered: the launcher holds its model now
        answer = httpx.MockTransport(lambda request: httpx.Response(200, json={"images": [], "seeds": []}))
        async with httpx.AsyncClient(transport=answer) as client:
            ctx = StageContext(project="P", prop_id="a", prop_dir=tmp_path, base_url="http://qwen", client=client,
                               settings={}, report=lambda f, m: None, cancelled=lambda: False)
            await qwen.generate(ctx, {"model": model, "batch_count": 1})

    asyncio.run(draw("full"))
    assert order("omnitexture", full_first) == ["a", "b", "c"]
    assert order("omnitexture", props) == ["b", "d", "a", "c", "e"]
    assert order("omnitexture", full_first, running=False) == ["b", "c", "a"], "stopped: a new one starts"
    monkeypatch.setattr(paths, "QWEN_SDCPP", tmp_path / "no engine")
    with pytest.raises(StageError):
        adapter.launch_spec(7777, {})
    assert qwen._holding == "turbo" and order("omnitexture", full_first) == ["b", "c", "a"], "started afresh"


def test_qwen_warms_up_only_for_omnitexture(monkeypatch: pytest.MonkeyPatch) -> None:
    """The job runner's hand-off asks the tool started early to warm up for the stage: Qwen-Image does it for
    Omni-Texture (POST /spellcast/warm_up), and a launcher that cannot be reached costs nothing."""
    import httpx

    from spellcast.adapters.qwen import QwenAdapter

    seen: list[tuple[str, Any]] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"queued": True})

    adapter = QwenAdapter(quality="Q4_K_M", text_encoder="UD-Q4_K_XL")

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
            await adapter.warm_up(client, "http://qwen", "images")
            await adapter.warm_up(client, "http://qwen", "omnitexture")
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(
                httpx.ConnectError("refused")))) as client:
            await adapter.warm_up(client, "http://qwen", "omnitexture")

    asyncio.run(main())
    assert seen == [("/spellcast/warm_up", {"albedo": True})]


def test_the_hand_off_warms_qwen_up_for_omnitexture_once_it_is_ready(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """A 3D job done hands the card to Qwen-Image (jobs.NEXT_STAGE); once it is ready the job runner asks it to warm
    up for Omni-Texture (adapter.warm_up), so that the job's first view does not pay the first use."""
    from spellcast import jobs as jobs_module
    from test_core import wait_state
    from test_retopo_hub import props_with_methods, retopo_adapters, wait_jobs

    monkeypatch.setattr(jobs_module, "HANDOFF_SECONDS", 0.2)
    store = ProjectStore(tmp_path / "projects")
    ids = props_with_methods(store, {"A": "lato2"})
    asked: list[tuple[str, str]] = []

    class Warming(FakeAdapter):
        async def warm_up(self, client: Any, base_url: str, stage: str) -> None:
            asked.append((stage, base_url))

    async def main() -> None:
        qwen = Warming(paths.QWEN, "omnitexture", True, tmp_path)
        adapters = retopo_adapters(tmp_path, **{paths.QWEN: qwen})
        async with manager(tmp_path, adapters) as pm:
            queue = JobQueue(pm, store, pm.settings, adapters, pm.client)
            queue.start()
            (job,) = await wait_jobs(queue, [(await queue.submit("mesh", "Demo", [ids["A"]]))["id"]])
            assert job["status"] == "done"
            await wait_state(pm, paths.QWEN, {"ready"}, timeout=15)
            for _ in range(100):
                if asked:
                    break
                await asyncio.sleep(0.05)
            assert asked == [("omnitexture", pm.get(paths.QWEN).base_url)]
            await queue.close()

    asyncio.run(main())


def test_arriving_at_omnitexture_while_the_hand_off_loads_qwen_joins_that_load(api: Any, paint: FakePaint,
                                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """A 3D job done starts Qwen-Image for Omni-Texture (jobs.NEXT_STAGE) while the artist is still in 3D. Whatever
    the page asks for as the artist arrives at Omni-Texture while it loads - the stage's early start (.../warm), a
    start (.../start), the first Omni-Texture job (which also drops the hand-off) - joins that one load: Qwen-Image
    is launched once and its "Starting… (N s)" keeps counting from the hand-off's start."""
    from spellcast import jobs as jobs_module
    from test_retopo_hub import starts

    monkeypatch.setattr(jobs_module, "HANDOFF_SECONDS", 0.2)
    app, client, _ = api
    procman = app.state.hub.procman
    procman.get("pixal3d").adapter.gpu_at_start = False      # as Pixal3D's own: it takes the card at its first job
    qwen = procman.get(paths.QWEN)
    qwen.adapter.args = ["--delay", "4"]                     # a load that lasts while the artist arrives
    client.post("/api/projects", json={"name": "Demo"})
    barrel = _meshed(client, app, "Barrel")
    chest = client.post("/api/projects/Demo/props", json={"name": "Chest"}).json()["id"]
    assert _job(client, {"stage": "mesh", "project": "Demo", "prop_ids": [chest]})["status"] == "done"

    loading = _poll(client, "/api/status", lambda s: s["tools"][paths.QWEN]["port"] is not None, {True})["tools"]
    assert loading[paths.QWEN]["state"] == "starting" and loading["pixal3d"]["state"] == "stopped"
    assert loading[paths.QWEN]["starting_seconds"] >= 0 and loading["pixal3d"]["starting_seconds"] is None
    pid, started, start = qwen.proc.pid, qwen.started_at, qwen.start_task
    warm = client.post(f"/api/tools/{paths.QWEN}/warm").json()
    assert warm["warmed"] is False and warm["state"] == "starting"
    assert client.post(f"/api/tools/{paths.QWEN}/start").json()["state"] == "starting"
    seen: list[int] = []
    paint.during = lambda ctx: seen.append(qwen.proc.pid)
    queued = client.post("/api/jobs", json={"stage": "omnitexture", "project": "Demo", "prop_ids": [barrel]}).json()
    assert client.get("/api/status").json()["tools"][paths.QWEN]["state"] == "starting", "asked while it loads"
    assert qwen.start_task is start

    job = _poll(client, f"/api/jobs/{queued['id']}", lambda j: j["status"], FINISHED)
    assert job["status"] == "done", job
    assert seen == [pid] and starts(procman, paths.QWEN) == 1 and qwen.started_at == started


def test_a_prop_that_lets_go_of_qwen_finishes_behind_the_next_one(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """jobs._run_props with StageContext.release: the next prop goes to Qwen-Image at once while the one before
    writes its GLB, which the page shows with its words; each prop is told the one after it (StageContext.upcoming)
    and both end done."""
    app, client, _ = api
    client.post("/api/projects", json={"name": "Demo"})
    a, b = _meshed(client, app, "Barrel"), _meshed(client, app, "Crate")
    seen: list[tuple[Any, ...]] = []

    async def fake(ctx: StageContext, prop: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        nxt = await ctx.upcoming() if ctx.upcoming is not None else None
        seen.append(("start", ctx.prop_id, nxt[1].name if nxt else None))
        ctx.report(0.95, "Painting the bottom onto the model (8 of 8)…")
        assert ctx.release is not None
        ctx.release(omnitexture.FINISHING)
        if ctx.prop_id == a:
            for _ in range(500):            # until the next prop has begun
                if ("start", b, None) in seen:
                    break
                await asyncio.sleep(0.01)
            seen.append(("states", app.state.hub.jobs.list()[0]["props"]))
        out = ctx.prop_dir / "omnitexture" / "ot_001"
        out.mkdir(parents=True, exist_ok=True)
        (out / "omnitexture.glb").write_bytes(b"glTF")
        return {"glb": ctx.rel(out / "omnitexture.glb"), "source_mesh": prop["mesh"]["glb"], "views": [],
                "stats": {}}

    monkeypatch.setattr(omnitexture, "run", fake)
    job = _job(client, {"stage": "omnitexture", "project": "Demo", "prop_ids": [a, b]})
    assert job["status"] == "done", job
    assert seen[:2] == [("start", a, b), ("start", b, None)]
    states = seen[2][1]
    assert states[a] == {"state": "working", "fraction": 0.95, "message": omnitexture.FINISHING}
    assert states[b]["state"] == "working"
    assert job["props"] == {p: {"state": "done", "fraction": 1.0, "message": ""} for p in (a, b)}


def test_a_job_of_two_props_reads_the_second_model_early_and_paints_it_behind_the_first(
        api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """omnitexture.run itself through the job runner (its worker and Qwen-Image's calls played by fakes): the
    second prop's worker is the one the first prop started at its last view, its first picture is asked for before
    the first prop's GLB is written, and both props end done with their own Omni-Texture."""
    app, client, _ = api
    client.post("/api/projects", json={"name": "Demo"})
    a, b = _meshed(client, app, "Barrel"), _meshed(client, app, "Crate")
    events: list[tuple[Any, ...]] = []
    made = _fakes(monkeypatch, events)
    job = _job(client, {"stage": "omnitexture", "project": "Demo", "prop_ids": [a, b]})
    assert job["status"] == "done" and job["errors"] == {}, job
    assert len(made) == 2
    assert [e[2] for e in events if e[:2] == ("worker", 1)][:2] == ["load", "view"]
    assert events.index(("worker", 1, "load", None)) < max(i for i, e in enumerate(events) if e == ("generated", a))
    assert events.index(("generate", b)) < events.index(("finished", 0)), "the GLB is written behind"
    for pid in (a, b):
        prop = next(p for p in client.get("/api/projects/Demo").json()["props"] if p["id"] == pid)
        assert prop["omnitexture"]["glb"] == "omnitexture/ot_001/omnitexture.glb", pid
    assert omnitexture._spare is None


def test_the_next_prop_takes_its_worker_however_long_the_last_view_takes(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """The next prop's worker waits SPARE_SECONDS from the moment this prop lets go of Qwen-Image, not from its own
    start: a last view slower than that (an old card, the full model) still hands it over."""
    from spellcast.adapters import qwen

    events: list[tuple[Any, ...]] = []
    made = _fakes(monkeypatch, events)
    monkeypatch.setattr(omnitexture, "SPARE_SECONDS", 0.2)
    plain = qwen.generate
    asked: list[str] = []

    async def slow(ctx: StageContext, body: dict[str, Any]) -> dict[str, Any]:
        asked.append(ctx.prop_id)
        if asked.count("a") == 2 and ctx.prop_id == "a":       # its last view
            await asyncio.sleep(0.6)
        return await plain(ctx, body)

    monkeypatch.setattr(qwen, "generate", slow)
    a, b = _st_prop(tmp_path, "a"), _st_prop(tmp_path, "b")

    async def main() -> None:
        b_turn = asyncio.Event()

        async def upcoming() -> tuple[dict[str, Any], Path]:
            return PROP, b

        first = asyncio.create_task(omnitexture.run(_st_ctx(a, release=lambda words: b_turn.set(),
                                                             upcoming=upcoming), PROP, {}))
        await b_turn.wait()
        await omnitexture.run(_st_ctx(b), PROP, {})
        await first
        await asyncio.sleep(0.4)                # past SPARE_SECONDS: no timer is left to fire

    asyncio.run(main())
    assert len(made) == 2 and ("stopped", 1) in events
    assert [e[2] for e in events if e[:2] == ("worker", 1)][:2] == ["load", "view"], "the spare, taken"
    assert events.count(("stopped", 1)) == 1, "stopped by its own run, once"
    assert omnitexture._spare is None


def test_a_cancel_during_the_last_albedo_saves_the_prop_and_stops_the_next_ones_worker(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancel after the last picture: the prop is made in full and saved (completed), the job ends there, so the
    worker started for its next prop is stopped at once rather than after SPARE_SECONDS."""
    import dataclasses

    from spellcast.adapters import qwen

    events: list[tuple[Any, ...]] = []
    made = _fakes(monkeypatch, events)
    plain = qwen.albedo
    cancelled: list[bool] = []

    async def albedo(ctx: StageContext, picture: Path, out: Path, model: str) -> Path:
        if len([e for e in events if e == ("albedo", "a")]) == 1:      # the second (last) one
            cancelled.append(True)
        return await plain(ctx, picture, out, model)

    monkeypatch.setattr(qwen, "albedo", albedo)
    a, b = _st_prop(tmp_path, "a"), _st_prop(tmp_path, "b")

    async def upcoming() -> tuple[dict[str, Any], Path]:
        return PROP, b

    ctx = dataclasses.replace(_st_ctx(a, upcoming=upcoming), cancelled=lambda: bool(cancelled))

    async def main() -> dict[str, Any]:
        patch = await omnitexture.run(ctx, PROP, {})
        await asyncio.sleep(0.1)                # the spare's stop, in a thread
        return patch

    patch = asyncio.run(main())
    assert ctx.completed is True and (a / patch["glb"]).is_file()
    assert len(made) == 2 and ("stopped", 1) in events and omnitexture._spare is None

    # A cancel during the last view's worker call: no spare is started at all.
    events.clear()
    late = dataclasses.replace(_st_ctx(a, upcoming=upcoming), cancelled=lambda: ("worker", 2, "view", 1) in events)
    asyncio.run(omnitexture.run(late, PROP, {}))
    assert late.completed is True and len(made) == 3 and omnitexture._spare is None


def test_the_next_prop_starts_without_the_last_ones_preview(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The job's one preview picture is the prop's on the card: it goes when that prop lets go of Qwen-Image, so the
    next prop's viewport starts without it, and a prop finishing behind never shows a picture over the next one's."""
    app, client, _ = api
    client.post("/api/projects", json={"name": "Demo"})
    a, b = _meshed(client, app, "Barrel"), _meshed(client, app, "Crate")
    seen: list[tuple[str, Any]] = []

    def shown() -> Any:
        return app.state.hub.jobs.list()[0]["progress"].get("preview")

    async def fake(ctx: StageContext, prop: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        assert ctx.preview is not None and ctx.release is not None
        if ctx.prop_id == a:
            ctx.preview(PNG, 7)
            seen.append(("a drew", shown()))
            ctx.release(omnitexture.FINISHING)
            for _ in range(500):            # until the next prop shows a picture of its own
                if any(what == "b drew" for what, _ in seen):
                    break
                await asyncio.sleep(0.01)
            ctx.preview(b"a late picture of a", 8)
            seen.append(("a behind", shown()))
        else:
            seen.append(("b starts", shown()))
            ctx.preview(PNG, 1)
            seen.append(("b drew", shown()))
            for _ in range(500):            # until the one behind has tried
                if any(what == "a behind" for what, _ in seen):
                    break
                await asyncio.sleep(0.01)
        out = ctx.prop_dir / "omnitexture" / "ot_001"
        out.mkdir(parents=True, exist_ok=True)
        (out / "omnitexture.glb").write_bytes(b"glTF")
        return {"glb": ctx.rel(out / "omnitexture.glb"), "source_mesh": prop["mesh"]["glb"], "views": [],
                "stats": {}}

    monkeypatch.setattr(omnitexture, "run", fake)
    job = _job(client, {"stage": "omnitexture", "project": "Demo", "prop_ids": [a, b]})
    assert job["status"] == "done", job
    pictures = dict(seen)
    assert (pictures["a drew"]["step"], pictures["a drew"]["prop"]) == (7, a)
    assert pictures["b starts"] is None, "the prop before's picture went with it"
    assert (pictures["b drew"]["step"], pictures["b drew"]["prop"]) == (1, b)
    assert pictures["a behind"] == pictures["b drew"], "the prop behind shows nothing over the next one"
    assert "preview" not in job["progress"]


def test_a_prop_that_ends_while_the_one_before_finishes_behind_says_how_it_ended() -> None:
    """Job.prop_states with a prop finishing behind: the props after it may end first (a failure at once, a cancel),
    and say so by their own record (Job.ended), not by their place in the order."""
    from spellcast.jobs import Job

    def states(**fields: Any) -> dict[str, str]:
        job = Job(id="j_1", stage="omnitexture", tool=paths.QWEN, project="P", prop_ids=["a", "b", "c"], params={},
                  status="running", behind={"a": (0.95, omnitexture.FINISHING)}, **fields)
        return {prop_id: entry["state"] for prop_id, entry in job.to_dict()["props"].items()}

    assert states(current_prop="b") == {"a": "working", "b": "working", "c": "waiting"}
    assert states(current_prop="c", done=1, ended={"b": "failed"}, errors={"b": "x"}) == {
        "a": "working", "b": "failed", "c": "working"}
    assert states(current_prop="c", done=2, ended={"b": "failed", "c": "failed"}, errors={"b": "x", "c": "y"}) == {
        "a": "working", "b": "failed", "c": "failed"}
    cut = states(current_prop="b", done=1, cut_short=1, ended={"b": "cancelled"}, cancel_requested=True)
    assert (cut["a"], cut["b"]) == ("working", "cancelled")


def test_the_page_sees_a_prop_fail_while_the_one_before_finishes_behind(api: Any,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the job runner: the first prop lets go of Qwen-Image, the second fails at once, the third is drawn,
    all while the first still writes its GLB."""
    app, client, _ = api
    client.post("/api/projects", json={"name": "Demo"})
    a, b, c = (_meshed(client, app, name) for name in ("Barrel", "Crate", "Lamp"))
    seen: list[tuple[str, Any]] = []

    async def wait_for(what: str) -> None:
        for _ in range(500):
            if any(w == what for w, _ in seen):
                return
            await asyncio.sleep(0.01)

    async def fake(ctx: StageContext, prop: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        if ctx.prop_id == b:
            raise StageError("Its 3D model is missing.")
        if ctx.prop_id == a:
            ctx.release(omnitexture.FINISHING)
            await wait_for("c started")
            seen.append(("states", app.state.hub.jobs.list()[0]["props"]))
        else:
            seen.append(("c started", None))
            await wait_for("states")
        out = ctx.prop_dir / "omnitexture" / "ot_001"
        out.mkdir(parents=True, exist_ok=True)
        (out / "omnitexture.glb").write_bytes(b"glTF")
        return {"glb": ctx.rel(out / "omnitexture.glb"), "source_mesh": prop["mesh"]["glb"], "views": [],
                "stats": {}}

    monkeypatch.setattr(omnitexture, "run", fake)
    job = _job(client, {"stage": "omnitexture", "project": "Demo", "prop_ids": [a, b, c]})
    states = {pid: entry["state"] for pid, entry in dict(seen)["states"].items()}
    assert states == {a: "working", b: "failed", c: "working"}
    assert job["status"] == "error" and {pid: e["state"] for pid, e in job["props"].items()} == {
        a: "done", b: "failed", c: "done"}


# ---- Export all ----------------------------------------------------------------------------------------------------

def test_export_all_saves_a_used_omnitexture_in_place_of_the_3d_model(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("Demo")
    ids = {}
    for name in ("Used", "Left out", "Stale"):
        pid = ids[name] = store.add_prop("Demo", name)["id"]
        glb = store.prop_dir("Demo", pid) / "mesh" / "mesh.glb"
        glb.parent.mkdir(parents=True)
        glb.write_bytes(f"glTF of {name}".encode())
        store.apply_stage_patch("Demo", pid, "mesh", {"glb": "mesh/mesh.glb"})
        _give_omnitexture(store, "Demo", pid)
        store.patch_prop("Demo", pid, {"omnitexture": {"settings": {"chosen": name != "Left out"}}})
    store.apply_stage_patch("Demo", ids["Stale"], "mesh", {"glb": "mesh/mesh.glb"})
    converter = FakeConverter()
    exporter = Exporter(store, converter, lambda fmt: None)
    saved = exporter.save_all("Demo")
    folder = tmp_path / "exports" / "Demo"
    assert saved.props == 3 and saved.warnings == []
    assert (folder / "Used" / "Used_high.glb").read_bytes() == b"glTF-omnitexture 1"
    assert (folder / "Left_out" / "Left_out_high.glb").read_bytes() == b"glTF of Left out"
    assert (folder / "Stale" / "Stale_high.glb").read_bytes() == b"glTF of Stale"
    used_dir = store.prop_dir("Demo", ids["Used"])
    assert (used_dir / ST).resolve() in {src.resolve() for src, _formats, _name in converter.calls}
    assert (used_dir / "export_cache" / "omnitexture" / "fbx").is_dir(), "its conversions are cached apart"
    # The 3D stage's own buttons save the 3D model itself.
    one = exporter.save_prop("Demo", ids["Used"], "mesh", "glb")
    assert one.shown == "Used_high.glb" and (folder / "Used" / "Used_high.glb").read_bytes() == b"glTF of Used"


def test_the_whole_omnitexture_taken_off_is_a_step_that_is_no_model_and_ctrl_z_brings_it_back(api: Any) -> None:
    """The page's x beside the painted views: nothing is made - the step that follows is TAKEN_OFF, the prop has no
    result (it bakes its 3D model again; a bake made from the result is out of date) and the result's files and
    record stay for the way back. Ctrl+Z and Ctrl+Y step onto the result and off it; a project read again holds
    the same; a tool taken up on the 3D model then begins a result of its own, the steps before gone."""
    app, client, _ = api
    client.post("/api/projects", json={"name": "Demo"})
    store = app.state.hub.store
    pid = _meshed(client, app, "Barrel")
    project, url = "Demo", f"/api/projects/Demo/props/{pid}/omnitexture"
    result = _give_omnitexture(store, project, pid, number=1, steps=[])
    store.apply_stage_patch(project, pid, "projection", {"glb": "projection/bake.glb", "stale": False})
    assert high_poly(store.get_prop(project, pid)) == result
    off = client.post(f"{url}/remove", json={"glb": result, "whole": True})
    assert off.status_code == 200, off.text
    prop = off.json()["prop"]
    assert projects.TAKEN_OFF == "" and prop["omnitexture"]["glb"] == projects.TAKEN_OFF
    assert prop["omnitexture"]["steps"] == [result, projects.TAKEN_OFF] and prop["omnitexture"]["views"]
    assert not omnitexture_used(prop) and high_poly(prop) == "mesh/mesh.glb" and prop["projection"]["stale"] is True
    assert (store.prop_dir(project, pid) / result).is_file(), "nothing of the result goes"
    assert client.post(f"{url}/remove", json={"glb": result, "whole": True}).status_code == 409, "it is off already"
    # Ctrl+Z, Ctrl+Y: each changes what the bake is made from.
    for glb in (result, projects.TAKEN_OFF):
        store.apply_stage_patch(project, pid, "projection", {"glb": "projection/bake.glb", "stale": False})
        stepped = client.post(f"{url}/step", json={"glb": glb})
        assert stepped.status_code == 200 and stepped.json()["omnitexture"]["glb"] == glb
        assert stepped.json()["omnitexture"]["steps"] == [result, projects.TAKEN_OFF]
        assert stepped.json()["projection"]["stale"] is True, glb
    again = ProjectStore(store.root).get_prop(project, pid)["omnitexture"]
    assert (again["glb"], again["steps"], again["repaints"]) == (projects.TAKEN_OFF, [result, projects.TAKEN_OFF], [])
    own = client.post(f"{url}/own")
    assert own.status_code == 200, own.text
    begun = own.json()["prop"]["omnitexture"]
    assert begun["glb"] and (begun["steps"], begun["views"]) == ([], [])
