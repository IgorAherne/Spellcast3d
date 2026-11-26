"""Retopology with three tools, hub side: the retopo method each prop picks (Instant Meshes, LATO.2, TriFlow),
its settings and sizes, the clean meshes a job makes (one per size, its variants) and the one the prop uses,
which tool made them and when they are fresh, one tool per job ("Retopo All" queues one job per method and each
tool loads once), the UV unwrap Instant Meshes does for the other two, the pick route, the license the page flags,
and the export warning for TriFlow's results.

Runs in the hub venv. The tools are played by fake_tool.py: LATO.2 and TriFlow by FakeRetopo (which does what
their adapters do: the launcher's /spellcast/retopo for each size, then StageContext.unwrap), Instant Meshes by the
real ImeshAdapter (FakeImesh).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from spellcast import exports, paths, projects
from spellcast.adapters import Lato2Adapter, TriflowAdapter, build_adapters, lato2, triflow, variants
from spellcast.adapters.base import StageContext, StageError, ToolNeedsRestart
from spellcast.exports import Exporter
from spellcast.gpu import GpuInfo
from spellcast.jobs import JobError, JobQueue
from spellcast.procman import ProcessManager, ToolError
from spellcast.projects import (DEFAULT_STAGE_SETTINGS, EDITABLE_STAGES, RETOPO_KEEP_VERTICES, RETOPO_METHODS,
                                ProjectError, ProjectStore, retopo_fresh, retopo_missing_sizes, retopo_size_of)
from spellcast.server import create_app
from spellcast import jobs as jobs_module
from test_core import FakeAdapter, make_settings, manager, wait_state
from test_projection_hub import FINISHED, FakeConverter, FakeImesh

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Spellcast3D is Windows-only")

LATO2, TRIFLOW, IMESH = paths.LATO2, paths.TRIFLOW, paths.IMESH
TRIFLOW_LICENSE = {"name": "ADPNCL 1.0", "noncommercial": True,
                   "url": "https://github.com/IgorAherne/triflow/blob/5cf7f00e4f5d5745a1cafbb21ff5d7d28bb4f878/LICENSE"}


class FakeRetopo(FakeAdapter):
    """LATO.2 or TriFlow as their adapters work: for each size the job makes (adapters/variants.py), the launcher
    (fake_tool.py) makes a triangle OBJ without UVs, one after the other; then ctx.unwrap has Instant Meshes give
    each its UVs -> the contract's patch. `watch()` is called right after the unwraps (the tests look at the tools'
    states then); `reports` keeps every progress report of the job, in order."""

    def __init__(self, tool_id: str, cwd: Path, *args: str, license: dict[str, Any] | None = None):
        super().__init__(tool_id, "retopo", True, cwd, args=args)
        self.display_name = {LATO2: "LATO.2", TRIFLOW: "TriFlow"}[tool_id]
        self.license = license
        self.watch: Callable[[], None] = lambda: None
        self.marks: list[tuple[str, str, float]] = []    # ("start" | "unwrapped", prop id, time.monotonic())
        self.bodies: list[dict[str, Any]] = []           # what the launcher was asked for, mesh by mesh
        self.reports: list[tuple[float, str]] = []

    def check_settings(self, stage: str, settings: dict[str, Any]) -> None:
        variants.sizes_of(settings)

    async def run_stage(self, ctx: StageContext, prop: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((ctx.prop_id, dict(params)))
        self.marks.append(("start", ctx.prop_id, time.monotonic()))
        glb = prop["mesh"]["glb"]
        report = ctx.report

        def told(fraction: float, message: str) -> None:
            self.reports.append((fraction, message))
            report(fraction, message)

        ctx.report = told

        async def make(size: int, part: StageContext) -> tuple[str, dict[str, Any]]:
            raw = f"{variants.folder(size)}/{self.tool_id}.raw.obj"
            (ctx.prop_dir / raw).parent.mkdir(parents=True, exist_ok=True)
            body = {"mesh_path": str((ctx.prop_dir / glb).resolve()), "out_path": str((ctx.prop_dir / raw).resolve()),
                    "seed": params.get(f"{self.tool_id}_seed", 0)}
            if self.tool_id == LATO2:
                body["vertices"] = lato2.vertices_for(size)
            else:
                body.update(faces=triflow.faces_for(size), quad_ratio=params["triflow_quad_ratio"], qem_threshold=0.5)
            self.bodies.append(body)
            part.report(0.5, f"{self.display_name} is making the clean mesh…")
            response = await ctx.client.post(f"{ctx.base_url}/spellcast/retopo", json=body)
            if response.status_code >= 400:
                raise StageError(response.json()["detail"])
            return raw, response.json()

        made = await variants.make_each(ctx, variants.to_make(prop, params), 0.8, make)
        ctx.report(0.8, "Unwrapping the UVs…")
        meshes = []
        for size, (raw, answer) in made:
            obj = f"{variants.folder(size)}/{self.tool_id}.obj"
            stats = await ctx.unwrap(raw, obj, params.get("uv_leniency", 1.0))
            meshes.append({"size": size, "obj": obj, "stats": {**stats, "raw_vertices": answer["vertices"]}})
        self.marks.append(("unwrapped", ctx.prop_id, time.monotonic()))
        self.watch()
        return {"variants": meshes, "method": self.tool_id, "source": glb, "saved_from": "auto"}


def retopo_adapters(tmp_path: Path, *lato2_args: str, **overrides: Any) -> dict[str, Any]:
    adapters = {"pixal3d": FakeAdapter("pixal3d", "mesh", True, tmp_path), IMESH: FakeImesh(tmp_path),
                LATO2: FakeRetopo(LATO2, tmp_path, *lato2_args),
                TRIFLOW: FakeRetopo(TRIFLOW, tmp_path, license=TRIFLOW_LICENSE)}
    adapters.update(overrides)
    return adapters


def props_with_methods(store: ProjectStore, methods: dict[str, str], mesh: bytes = b"glTF") -> dict[str, str]:
    """A project "Demo" with a prop per name, each with a 3D model and set to its retopo method -> {name: id}."""
    if not (store.root / "Demo").exists():
        store.create_project("Demo")
    ids = {}
    for name, method in methods.items():
        pid = store.add_prop("Demo", name)["id"]
        glb = store.prop_dir("Demo", pid) / "mesh" / "mesh.glb"
        glb.parent.mkdir(parents=True, exist_ok=True)
        glb.write_bytes(mesh)
        store.apply_stage_patch("Demo", pid, "mesh", {"glb": "mesh/mesh.glb"})
        store.patch_prop("Demo", pid, {"retopo": {"settings": {"method": method}}})
        ids[name] = pid
    return ids


async def wait_jobs(queue: JobQueue, job_ids: list[str], timeout: float = 60.0) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        jobs = [queue.get(job_id) for job_id in job_ids]
        if all(job["status"] in FINISHED for job in jobs):
            return jobs
        await asyncio.sleep(0.05)
    raise AssertionError(f"jobs stuck: {[queue.get(job_id) for job_id in job_ids]}")


def starts(pm: ProcessManager, tool: str) -> int:
    """How often the tool's process was started (its log's header lines)."""
    name = pm.get(tool).name
    return sum(f"starting {name} on port" in line for line in pm.log_tail(tool, 2000))


# ---- which tools run which stage ------------------------------------------------------------------------

def test_stage_tools_are_the_one_source() -> None:
    """paths.STAGE_TOOLS says which tools can run a stage (the default first, as paths.STAGES has it); every
    adapter runs exactly the stages it lists it for, and the retopo methods are its retopo tools."""
    adapters = build_adapters()
    assert set(adapters) == set(paths.TOOL_IDS)
    assert all(adapter.tool_id == tool for tool, adapter in adapters.items())
    assert isinstance(adapters[LATO2], Lato2Adapter) and isinstance(adapters[TRIFLOW], TriflowAdapter)
    by_adapter = {(stage, tool) for tool, adapter in adapters.items() for stage in adapter.stages}
    assert by_adapter == {(stage, tool) for stage, tools in paths.STAGE_TOOLS.items() for tool in tools}
    assert {stage: tool for stage, tool in paths.STAGES} == {stage: tools[0] if tools else None
                                                             for stage, tools in paths.STAGE_TOOLS.items()}
    assert set(EDITABLE_STAGES) == {stage for stage, tools in paths.STAGE_TOOLS.items() if tools}
    assert RETOPO_METHODS == paths.STAGE_TOOLS["retopo"] == (TRIFLOW, LATO2, IMESH)      # TriFlow is the default


# ---- settings, results, freshness ---------------------------------------------------------------------

def test_retopo_settings_are_checked_and_repaired_and_a_new_prop_starts_with_the_default_size(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    first = store.add_prop("P", "Barrel")
    assert {k: first["retopo"]["settings"][k] for k in ("method", "sizes", "lato2_seed", "triflow_quad_ratio",
                                                         "triflow_seed")} == {
        "method": "triflow", "sizes": [2], "lato2_seed": 0, "triflow_quad_ratio": 0.95, "triflow_seed": 0}
    assert "lato2_vertices" not in first["retopo"]["settings"] and "triflow_faces" not in first["retopo"]["settings"]
    assert first["retopo"]["variants"] == []        # one clean mesh of 2,000 vertices to make; TriFlow's at first
    patched = store.patch_prop("P", first["id"], {"retopo": {"settings": {
        "method": "lato2", "sizes": [4, 22.0, 6], "triflow_quad_ratio": 1, "target_vertices": 7000,
        "lato2_vertices": 3000}}})
    settings = patched["retopo"]["settings"]
    assert (settings["method"], settings["sizes"], settings["triflow_quad_ratio"]) == ("lato2", [4, 22, 6], 1.0)
    assert all(isinstance(size, int) for size in settings["sizes"]) and isinstance(settings["triflow_quad_ratio"], float)
    assert "lato2_vertices" not in settings, "a setting that is none any more, from an older page, is left out"
    twice = store.patch_prop("P", first["id"], {"retopo": {"settings": {"sizes": [4, 4, 9]}}})["retopo"]["settings"]
    assert twice["sizes"] == [4, 9], "each size once"
    assert store.patch_prop("P", first["id"], {"retopo": {"settings": {"sizes": []}}})["retopo"]["settings"]["sizes"] == []
    # Under a thousand vertices, and with decimals: rounded to two, a whole one an int, each once after rounding.
    small = store.patch_prop("P", first["id"], {"retopo": {"settings": {"sizes": [0.3, 2.0, 12.254]}}})["retopo"]
    assert small["settings"]["sizes"] == [0.3, 2, 12.25] and [type(size) for size in small["settings"]["sizes"]] == [
        float, int, float]
    rounded = store.patch_prop("P", first["id"], {"retopo": {"settings": {"sizes": [0.1, 0.099, 99]}}})["retopo"]
    assert rounded["settings"]["sizes"] == [0.1, 99], "0.099 is 0.1, once"
    settings = store.patch_prop("P", first["id"], {"retopo": {"settings": {"sizes": [4, 22, 6]}}})["retopo"]["settings"]
    # A new prop inherits the tools' numbers, but starts with TriFlow again and with the default size: the sizes
    # typed on one prop's tile are that prop's (a prop after one of 1k showed 1k, and read as the default).
    store.patch_prop("P", first["id"], {"retopo": {"settings": {"triflow_seed": 7}}})
    second = store.add_prop("P", "Crate")
    assert [second["retopo"]["settings"][key] for key in ("method", "sizes", "triflow_seed")] == ["triflow", [2], 7]
    assert store.ensure_draft("P")["retopo"]["settings"]["sizes"] == [2]
    settings = store.get_prop("P", first["id"])["retopo"]["settings"]
    assert second["retopo"]["obj"] is None and second["retopo"]["method"] is None

    sizes = "A prop has up to 3 sizes, each a number of thousands of vertices from 0.1 to 99."
    for bad, words in (({"method": "blender"}, "The retopology method must be one of: triflow, lato2, imesh."),
                       ({"sizes": [0]}, sizes), ({"sizes": [0.05]}, sizes), ({"sizes": [0.094]}, sizes),
                       ({"sizes": [100]}, sizes), ({"sizes": [99.01]}, sizes), ({"sizes": [1, 2, 3, 4]}, sizes),
                       ({"sizes": [-2]}, sizes), ({"sizes": [float("nan")]}, sizes), ({"sizes": [float("inf")]}, sizes),
                       ({"sizes": [True]}, sizes), ({"sizes": ["4"]}, sizes), ({"sizes": ["0.3"]}, sizes),
                       ({"sizes": 4}, sizes), ({"sizes": None}, sizes),
                       ({"lato2_seed": True}, "LATO.2's Seed must be a whole number from 0 to 2,147,483,647."),
                       ({"triflow_seed": -1}, "TriFlow's Seed must be a whole number"),
                       ({"triflow_quad_ratio": 1.5}, "TriFlow's Quads must be a number from 0 to 1."),
                       ({"triflow_quad_ratio": None}, "TriFlow's Quads must be a number")):
        with pytest.raises(ProjectError, match=re.escape(words)) as e:
            store.patch_prop("P", first["id"], {"retopo": {"settings": bad}})
        assert e.value.status_code == 400
    assert store.get_prop("P", first["id"])["retopo"]["settings"] == settings    # nothing of them was kept

    # A hand-edited project.json: what is not valid becomes the default, the rest stays; LATO.2's and TriFlow's
    # counts of before go.
    file = tmp_path / "projects" / "P" / "project.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    data["props"][0]["retopo"]["settings"].update(method="blender", sizes="lots", lato2_vertices=3000,
                                                  triflow_faces=6000)
    file.write_text(json.dumps(data), encoding="utf-8")
    repaired = store.get_prop("P", first["id"])["retopo"]["settings"]
    assert (repaired["method"], repaired["sizes"], repaired["target_vertices"]) == ("triflow", [2], 7000)
    assert "lato2_vertices" not in repaired and "triflow_faces" not in repaired
    data["props"][1]["retopo"]["settings"]["sizes"] = [0.25, 2.0, 0.2500001]       # sizes the hub keeps as numbers
    file.write_text(json.dumps(data), encoding="utf-8")
    read = store.get_prop("P", second["id"])["retopo"]["settings"]["sizes"]
    assert read == [0.25, 2] and isinstance(read[1], int)
    # The UV leniency's old default (the page never showed it) becomes today's, xatlas's largest islands, in a
    # project from before it; another value stays. Once the project says it is from since, a 0.5 (picked in Instant
    # Meshes' viewer) stays too.
    assert projects.UV_LENIENCY == 1.0 and data.get("uv_leniency_default") == 1.0     # written since
    older = {key: value for key, value in data.items() if key != "uv_leniency_default"}
    for marked, kept, expected in ((older, 0.5, 1.0), (older, 0.3, 0.3), (data, 0.5, 0.5)):
        marked["props"][1]["retopo"]["settings"]["uv_leniency"] = kept
        file.write_text(json.dumps(marked), encoding="utf-8")
        assert store.get_prop("P", second["id"])["retopo"]["settings"]["uv_leniency"] == expected

    # A project from before the choice: a prop with a clean mesh keeps the method that made it (Instant Meshes, so
    # the mesh is not "to do" again); one without gets the default.
    file = tmp_path / "projects" / "P" / "project.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    for prop, obj in zip(data["props"], ("retopo/retopo.obj", None)):
        del prop["retopo"]["settings"]["method"]
        prop["retopo"].update(obj=obj, method=None)
    file.write_text(json.dumps(data), encoding="utf-8")
    old, new = (store.get_prop("P", prop["id"]) for prop in (first, second))
    assert (old["retopo"]["settings"]["method"], old["retopo"]["method"], retopo_fresh(old)) == ("imesh", "imesh", True)
    assert (new["retopo"]["settings"]["method"], new["retopo"]["method"]) == ("triflow", None)


def test_a_clean_mesh_says_which_method_made_it_and_is_fresh_while_that_is_picked(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    pid = props_with_methods(store, {"Barrel": "lato2"})["Barrel"]
    prop = store.apply_stage_patch("Demo", pid, "retopo", {"obj": "retopo/lato2.obj", "method": "lato2",
                                                           "source": "mesh/mesh.glb", "accepted": False})
    assert (prop["retopo"]["method"], prop["retopo"]["stale"], prop["retopo"]["accepted"]) == ("lato2", False, False)
    assert retopo_fresh(prop)
    # Another method picked: the prop is "to do" again, and keeps its clean mesh until a new one replaces it.
    prop = store.patch_prop("Demo", pid, {"retopo": {"settings": {"method": "triflow"}}})
    assert not retopo_fresh(prop) and prop["retopo"]["obj"] == "retopo/lato2.obj"
    assert retopo_fresh(store.patch_prop("Demo", pid, {"retopo": {"settings": {"method": "lato2"}}}))
    # A result that does not say its method is Instant Meshes' (the viewer and older code).
    prop = store.apply_stage_patch("Demo", pid, "retopo", {"obj": "retopo/retopo.obj", "source": "mesh/mesh.glb"})
    assert prop["retopo"]["method"] == "imesh" and not retopo_fresh(prop)
    prop = store.apply_stage_patch("Demo", pid, "retopo", {"obj": "retopo/x.obj", "method": "blender"})
    assert prop["retopo"]["method"] == "imesh"
    prop = store.apply_stage_patch("Demo", pid, "retopo", {"obj": "retopo/lato2.obj", "method": "lato2",
                                                           "source": "mesh/mesh.glb"})
    assert retopo_fresh(prop)
    prop = store.apply_stage_patch("Demo", pid, "retopo", {"stats": {"vertices": 3}})    # not a new clean mesh
    assert prop["retopo"]["method"] == "lato2" and retopo_fresh(prop)
    prop = store.apply_stage_patch("Demo", pid, "mesh", {"glb": "mesh/mesh.glb"})         # a new 3D model
    assert prop["retopo"]["stale"] and not retopo_fresh(prop)


# ---- the clean meshes of a prop: one per size, the one picked ----------------------------------------------

RETOPO_FILES = {"triflow": "triflow.obj", "lato2": "lato2.obj", "imesh": "retopo.obj"}


def kept_setup(tmp_path: Path, method: str = "triflow",
               sizes: tuple[int, ...] = (2, 5)) -> tuple[ProjectStore, str, Any, Any]:
    """A prop set to `method` and `sizes`, with pick(method) (the bar) and made(method, {size: vertices},
    **more) (a job's clean meshes, their files written like the tools do: each size and method its own)."""
    store = ProjectStore(tmp_path / "projects")
    pid = props_with_methods(store, {"Barrel": method})["Barrel"]
    store.patch_prop("Demo", pid, {"retopo": {"settings": {"sizes": list(sizes)}}})

    def pick(method: str) -> dict[str, Any]:
        return store.patch_prop("Demo", pid, {"retopo": {"settings": {"method": method}}})

    def made(method: str, meshes: dict[int, int], **more: Any) -> dict[str, Any]:
        made = []
        for size, vertices in meshes.items():
            rel = f"{variants.folder(size)}/{RETOPO_FILES[method]}"
            file = store.prop_dir("Demo", pid) / rel
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(f"# {method} {vertices}\n", encoding="utf-8")
            made.append({"size": size, "obj": rel, "stats": {"vertices": vertices, "faces": 2 * vertices}})
        return store.apply_stage_patch("Demo", pid, "retopo", {
            "variants": made, "method": method, "source": "mesh/mesh.glb", "saved_from": "auto", **more})

    return store, pid, pick, made


def test_a_job_makes_one_clean_mesh_per_size_and_the_prop_uses_the_one_picked(tmp_path: Path) -> None:
    store, pid, pick, made = kept_setup(tmp_path, sizes=(4, 22, 6))
    retopo = made("triflow", {4: 4100, 22: 21_800, 6: 6020})["retopo"]
    assert [(v["size"], v["obj"], v["stats"]["vertices"]) for v in retopo["variants"]] == [
        (4, "retopo/4k/triflow.obj", 4100), (6, "retopo/6k/triflow.obj", 6020), (22, "retopo/22k/triflow.obj", 21_800)]
    # Several sizes: none is picked yet; the rest of the app reads the first size's (retopo.obj).
    assert (retopo["obj"], retopo["stats"]["vertices"], retopo["accepted"]) == ("retopo/4k/triflow.obj", 4100, False)
    prop = store.pick_retopo("Demo", pid, 22)
    assert (prop["retopo"]["obj"], prop["retopo"]["stats"]["vertices"], prop["retopo"]["accepted"]) == (
        "retopo/22k/triflow.obj", 21_800, True)
    assert retopo_fresh(prop) and "retopo" not in prop["thumbs"]
    # One more size: only it is made, and the pick stays (its file is the same); one made again is picked again.
    store.patch_prop("Demo", pid, {"retopo": {"settings": {"sizes": [4, 22, 8]}}})
    assert retopo_missing_sizes(store.get_prop("Demo", pid)) == [8]
    retopo = made("triflow", {8: 7900})["retopo"]
    assert [v["size"] for v in retopo["variants"]] == [4, 6, 8, 22], "a size erased keeps its clean mesh, unseen"
    assert (retopo["obj"], retopo["accepted"]) == ("retopo/22k/triflow.obj", True)
    retopo = made("triflow", {4: 4000, 22: 22_100, 8: 8000})["retopo"]
    assert (retopo["obj"], retopo["stats"]["vertices"], retopo["accepted"]) == ("retopo/4k/triflow.obj", 4000, False)
    assert retopo_missing_sizes(store.get_prop("Demo", pid)) == []
    # The boxes brought down to one size that is made: it is the prop's, picked, without a click - and a bake of
    # the one it used before is out of date. One that is not made yet changes nothing.
    store.apply_stage_patch("Demo", pid, "projection", {"glb": "projection/baked.glb", "stale": False})
    only = store.patch_prop("Demo", pid, {"retopo": {"settings": {"sizes": [22]}}})
    assert (only["retopo"]["obj"], only["retopo"]["stats"]["vertices"], only["retopo"]["accepted"]) == (
        "retopo/22k/triflow.obj", 22_100, True)
    assert only["projection"]["stale"] is True and "retopo" not in only["thumbs"]
    # A prop with one size: its clean mesh is picked as soon as it is made.
    store.patch_prop("Demo", pid, {"retopo": {"settings": {"sizes": [3]}}})
    assert store.get_prop("Demo", pid)["retopo"]["obj"] == "retopo/22k/triflow.obj"
    retopo = made("triflow", {3: 3050})["retopo"]
    assert (retopo["obj"], retopo["accepted"]) == ("retopo/3k/triflow.obj", True)
    # A new 3D model: every clean mesh is old, all are missing, and the next job's replace them.
    store.apply_stage_patch("Demo", pid, "mesh", {"glb": "mesh/mesh.glb"})
    stale = store.get_prop("Demo", pid)
    assert stale["retopo"]["stale"] and retopo_missing_sizes(stale) == [3]
    # Old clean meshes are no one's pick: another single size typed meanwhile changes nothing.
    assert store.patch_prop("Demo", pid, {"retopo": {"settings": {"sizes": [22]}}})["retopo"]["obj"] ==         "retopo/3k/triflow.obj"
    store.patch_prop("Demo", pid, {"retopo": {"settings": {"sizes": [3]}}})
    retopo = made("triflow", {3: 2990})["retopo"]
    assert [v["size"] for v in retopo["variants"]] == [3] and not retopo["stale"] and retopo["accepted"]


def test_picking_a_clean_mesh_makes_a_bake_of_another_one_old(tmp_path: Path) -> None:
    store, pid, pick, made = kept_setup(tmp_path, "lato2", sizes=(2, 5))
    made("lato2", {2: 2000, 5: 4900})
    store.pick_retopo("Demo", pid, 2)
    bake = {"glb": "projection/baked.glb", "source_mesh": "mesh/mesh.glb", "source_retopo": "retopo/2k/lato2.obj"}
    assert not store.apply_stage_patch("Demo", pid, "projection", bake)["projection"]["stale"]
    assert not store.pick_retopo("Demo", pid, 2)["projection"]["stale"], "the same one again changes nothing"
    assert not made("lato2", {5: 5000})["projection"]["stale"], "another size made: the one used is the same"
    assert store.pick_retopo("Demo", pid, 5)["projection"]["stale"]
    # Not there to pick: a size not made, a file gone, clean meshes from an older 3D model.
    for size in (3, None, "5", True):
        with pytest.raises(ProjectError, match="not there to pick") as e:
            store.pick_retopo("Demo", pid, size)
        assert e.value.status_code == 409
    (store.prop_dir("Demo", pid) / "retopo" / "2k" / "lato2.obj").unlink()
    with pytest.raises(ProjectError, match="not there to pick"):
        store.pick_retopo("Demo", pid, 2)
    store.apply_stage_patch("Demo", pid, "mesh", {"glb": "mesh/mesh.glb"})
    with pytest.raises(ProjectError, match="not there to pick"):
        store.pick_retopo("Demo", pid, 5)


def test_each_method_keeps_its_clean_meshes_while_another_is_picked(tmp_path: Path) -> None:
    store, pid, pick, made = kept_setup(tmp_path)
    made("triflow", {2: 2100, 5: 5200})
    prop = store.pick_retopo("Demo", pid, 2)
    assert retopo_fresh(prop) and prop["retopo"]["accepted"]
    # Another method picked: TriFlow's clean meshes wait, LATO.2 has none yet.
    retopo = pick("lato2")["retopo"]
    assert (retopo["obj"], retopo["method"], retopo["accepted"], retopo["variants"], "stats" in retopo) == (
        None, None, False, [], False)
    assert retopo["kept"]["triflow"]["obj"] == "retopo/2k/triflow.obj"
    assert [v["size"] for v in retopo["kept"]["triflow"]["variants"]] == [2, 5]
    prop = made("lato2", {2: 1900, 5: 5000})
    assert retopo_fresh(prop) and not prop["retopo"]["accepted"] and set(prop["retopo"]["kept"]) == {"triflow"}
    # Back to TriFlow: its files, its numbers and its pick are there again, nothing is made again; LATO.2's wait.
    prop = pick("triflow")
    retopo = prop["retopo"]
    assert (retopo["obj"], retopo["method"], retopo["stats"]["vertices"], retopo["accepted"]) \
        == ("retopo/2k/triflow.obj", "triflow", 2100, True)
    assert [v["size"] for v in retopo["variants"]] == [2, 5]
    assert retopo_fresh(prop) and set(retopo["kept"]) == {"lato2"} and "retopo" not in prop["thumbs"]
    assert (store.prop_dir("Demo", pid) / "retopo" / "2k" / "lato2.obj").read_text(encoding="utf-8") == "# lato2 1900\n"
    # "Make it again" remakes the picked method's only, and a pick picks among the picked method's only.
    prop = made("triflow", {2: 2300, 5: 5100})
    assert prop["retopo"]["stats"]["vertices"] == 2300 and not prop["retopo"]["accepted"]
    assert prop["retopo"]["kept"]["lato2"]["stats"]["vertices"] == 1900
    pick("lato2")
    retopo = store.pick_retopo("Demo", pid, 5)["retopo"]
    assert (retopo["method"], retopo["obj"], retopo["accepted"], retopo["kept"]["triflow"]["accepted"]) == (
        "lato2", "retopo/5k/lato2.obj", True, False)
    # Instant Meshes' clean meshes (a job's, or the viewer's Accept) are kept the same way.
    assert set(pick("imesh")["retopo"]["kept"]) == {"triflow", "lato2"}
    made("imesh", {5: 5000}, saved_from="viewer", picked=5)
    retopo = pick("lato2")["retopo"]
    assert (retopo["saved_from"], retopo["accepted"], retopo["kept"]["imesh"]["saved_from"]) == ("auto", True, "viewer")
    assert pick("imesh")["retopo"]["accepted"] and store.get_prop("Demo", pid)["retopo"]["stats"]["vertices"] == 5000
    # A new 3D model: every clean mesh of the prop is old, the kept ones too.
    prop = store.apply_stage_patch("Demo", pid, "mesh", {"glb": "mesh/mesh.glb"})
    assert prop["retopo"]["stale"] and all(kept["stale"] for kept in prop["retopo"]["kept"].values())
    assert not any(retopo_fresh(pick(method)) for method in ("triflow", "lato2", "imesh"))
    assert retopo_fresh(made("imesh", {5: 5100})) and not retopo_fresh(pick("triflow"))
    # The project file has them: read again, they are still there.
    again = ProjectStore(tmp_path / "projects").get_prop("Demo", pid)["retopo"]
    assert (again["method"], set(again["kept"])) == ("triflow", {"lato2", "imesh"})
    assert [v["size"] for v in again["kept"]["imesh"]["variants"]] == [5]


def test_only_clean_meshes_under_the_limit_are_kept(tmp_path: Path) -> None:
    store, pid, pick, made = kept_setup(tmp_path, sizes=(2, 50))
    assert RETOPO_KEEP_VERTICES == 50_000
    made("triflow", {2: 2000, 50: RETOPO_KEEP_VERTICES})
    # One too big to keep: they stay the prop's clean meshes, "to do" again, as before there were kept ones.
    prop = pick("lato2")
    assert prop["retopo"]["obj"] == "retopo/2k/triflow.obj" and not retopo_fresh(prop) and "kept" not in prop["retopo"]
    assert retopo_fresh(pick("triflow"))
    pick("lato2")
    prop = made("lato2", {2: 1900})                             # another method's replace them: gone
    assert retopo_fresh(prop) and "kept" not in prop["retopo"]
    retopo = pick("triflow")["retopo"]
    assert retopo["obj"] is None and set(retopo["kept"]) == {"lato2"}
    made("triflow", {2: 2000, 50: RETOPO_KEEP_VERTICES - 1})    # one vertex fewer: kept
    retopo = pick("lato2")["retopo"]
    assert (retopo["method"], set(retopo["kept"])) == ("lato2", {"triflow"})
    # Kept clean meshes coming back replace big ones too.
    made("lato2", {50: 60_000})
    retopo = pick("triflow")["retopo"]
    assert retopo["variants"][-1]["stats"]["vertices"] == RETOPO_KEEP_VERTICES - 1 and "kept" not in retopo
    # One that does not say how many vertices it has (older results) is not kept.
    store.apply_stage_patch("Demo", pid, "retopo", {"obj": "retopo/triflow.obj", "method": "triflow",
                                                    "source": "mesh/mesh.glb", "stats": None})
    assert pick("lato2")["retopo"]["obj"] == "retopo/triflow.obj"


def test_a_bake_follows_the_clean_mesh_it_was_made_from(tmp_path: Path) -> None:
    store, pid, pick, made = kept_setup(tmp_path, sizes=(2,))
    made("triflow", {2: 2100})
    bake = {"glb": "projection/baked.glb", "source_mesh": "mesh/mesh.glb", "source_retopo": "retopo/2k/triflow.obj"}
    assert not store.apply_stage_patch("Demo", pid, "projection", bake)["projection"]["stale"]
    assert not pick("lato2")["projection"]["stale"]             # no clean mesh of LATO.2 yet: nothing to bake
    assert made("lato2", {2: 1900})["projection"]["stale"]
    assert not pick("triflow")["projection"]["stale"]           # the bake is of TriFlow's clean mesh, which is back
    assert pick("lato2")["projection"]["stale"]
    pick("triflow")
    store.apply_stage_patch("Demo", pid, "mesh", {"glb": "mesh/mesh.glb"})      # a new 3D model: old for good
    pick("lato2")
    assert pick("triflow")["projection"]["stale"]


def test_a_project_from_before_sizes_opens_with_its_clean_mesh_as_the_one_variant(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    ids = props_with_methods(store, {"Viewer": "imesh", "Auto": "lato2", "None": "triflow", "Bare": "triflow",
                                     "Tiny": "imesh", "Huge": "imesh"})
    file = tmp_path / "projects" / "Demo" / "project.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    old = {"Viewer": {"obj": "retopo/retopo.obj", "method": "imesh", "accepted": True, "saved_from": "viewer",
                      "stats": {"vertices": 5320}},
           "Auto": {"obj": "retopo/lato2.obj", "method": "lato2", "accepted": False, "saved_from": "auto",
                    "stats": {"vertices": 1910, "has_uv": True},
                    "kept": {"triflow": {"obj": "retopo/triflow.obj", "method": "triflow", "stats": {"vertices": 3990}}}},
           "None": {"obj": "retopo/triflow.obj", "method": "triflow", "stats": None},
           "Tiny": {"obj": "retopo/retopo.obj", "method": "imesh", "stats": {"vertices": 96}},
           "Huge": {"obj": "retopo/retopo.obj", "method": "imesh", "stats": {"vertices": 120_000}}}
    for prop in data["props"]:
        prop["retopo"]["settings"].pop("sizes")
        prop["retopo"]["settings"]["lato2_vertices"] = 2000
        prop["retopo"].pop("variants")
        prop["retopo"].update(source="mesh/mesh.glb", **old.get(prop["name"], {}))
        if prop["retopo"].get("obj"):
            obj = store.prop_dir("Demo", prop["id"]) / prop["retopo"]["obj"]
            obj.parent.mkdir(parents=True, exist_ok=True)
            obj.write_text("v 0 0 0\n", encoding="utf-8")
    file.write_text(json.dumps(data), encoding="utf-8")
    props = {name: store.get_prop("Demo", pid) for name, pid in ids.items()}
    viewer, auto, none, bare = (props[name]["retopo"] for name in ("Viewer", "Auto", "None", "Bare"))
    # The size a vertex count stands for: its thousands to two significant digits, within the sizes.
    assert viewer["variants"] == [{"size": 5.3, "obj": "retopo/retopo.obj", "stats": {"vertices": 5320}}]
    assert (viewer["settings"]["sizes"], viewer["accepted"]) == ([5.3], True), "it asks for the size it has, picked"
    assert retopo_fresh(props["Viewer"]) and retopo_missing_sizes(props["Viewer"]) == []
    assert (auto["variants"][0]["size"], auto["settings"]["sizes"], auto["accepted"]) == (1.9, [1.9], True), \
        "one size, and it is made: nothing to choose, so it is picked though nobody accepted it then"
    assert auto["kept"]["triflow"]["variants"] == [{"size": 4, "obj": "retopo/triflow.obj", "stats": {"vertices": 3990}}]
    assert isinstance(auto["kept"]["triflow"]["variants"][0]["size"], int), "3,990 is 4.0 thousand: a whole size"
    assert none["variants"] == [{"size": 2, "obj": "retopo/triflow.obj", "stats": None}], "no count: the default size"
    assert (bare["variants"], bare["settings"]["sizes"]) == ([], [2])
    assert [props[name]["retopo"]["settings"]["sizes"] for name in ("Tiny", "Huge")] == [[0.1], [99]], "within the sizes"
    assert all("lato2_vertices" not in p["retopo"]["settings"] for p in props.values())
    # Picked, and its bake: as before.
    assert store.pick_retopo("Demo", ids["Auto"], 1.9)["retopo"]["obj"] == "retopo/lato2.obj"


@pytest.mark.parametrize("vertices, size", [(740, 0.74), (5234, 5.2), (22_053, 22), (96, 0.1), (0, 0.1), (1, 0.1),
                                            (104, 0.1), (1949, 1.9), (2000, 2), (12_340, 12), (99_400, 99),
                                            (250_000, 99), (3999.6, 4), (None, 2), ("5000", 2), (True, 2)])
def test_the_size_a_clean_mesh_stands_for(vertices: Any, size: Any) -> None:
    """retopo_size_of: the thousands of a vertex count to two significant digits, within the sizes, a whole one an
    int (a clean mesh of an older project, or one shaped in Instant Meshes' viewer); one that does not say a count:
    the default size."""
    assert retopo_size_of(vertices) == size and type(retopo_size_of(vertices)) is type(size)


def test_a_project_from_before_kept_clean_meshes_reads_as_it_was(tmp_path: Path) -> None:
    store, pid, pick, made = kept_setup(tmp_path, "lato2", sizes=(2,))
    made("lato2", {2: 1900})
    file = tmp_path / "projects" / "Demo" / "project.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    assert "kept" not in data["props"][0]["retopo"]             # one method's clean meshes: no "kept" in the file
    data["props"][0]["retopo"]["kept"] = {"blender": {"obj": "x.obj"}, "triflow": "junk", "imesh": {"obj": None}}
    file.write_text(json.dumps(data), encoding="utf-8")
    prop = ProjectStore(tmp_path / "projects").get_prop("Demo", pid)
    assert "kept" not in prop["retopo"] and retopo_fresh(prop) and prop["retopo"]["accepted"]
    assert prop["retopo"]["obj"] == "retopo/2k/lato2.obj"


# ---- one tool per job ------------------------------------------------------------------------------

def test_a_job_has_one_tool_and_the_props_must_fit_it(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    ids = props_with_methods(store, {"Barrel": "lato2", "Crate": "lato2", "Lamp": "triflow", "Chest": "imesh"})
    barrel, crate, lamp, chest = ids.values()

    async def main() -> None:
        adapters = {"forge": FakeAdapter("forge", "images", True, tmp_path), **retopo_adapters(tmp_path)}
        async with manager(tmp_path, adapters) as pm:
            queue = JobQueue(pm, store, pm.settings, adapters, pm.client)      # not started: the jobs wait
            job = await queue.submit("retopo", "Demo", [barrel, crate])
            assert (job["tool"], job["stage"], job["status"]) == ("lato2", "retopo", "queued")
            assert (await queue.submit("retopo", "Demo", [lamp], None, "triflow"))["tool"] == "triflow"
            assert (await queue.submit("retopo", "Demo", [chest]))["tool"] == "imesh"
            assert (await queue.submit("images", "Demo", [chest]))["tool"] == "forge"
            # A job's params can pick the method too (they override the prop's settings).
            assert (await queue.submit("retopo", "Demo", [chest], {"method": "lato2"}))["tool"] == "lato2"
            assert [j["tool"] for j in queue.list("Demo")] == ["lato2", "forge", "imesh", "triflow", "lato2"]

            for args, words in (
                    (("retopo", "Demo", [barrel, lamp]), "These props use different methods (TriFlow, LATO.2). "
                                                         "Start one job per method."),
                    (("retopo", "Demo", [barrel, chest], None, "lato2"),
                     "Chest is set to be made with Instant Meshes, not LATO.2. Reload the page and try again."),
                    (("retopo", "Demo", [chest], None, "blender"), '"blender" cannot run the retopo stage. '
                                                                   'Use one of: triflow, lato2, imesh.'),
                    (("images", "Demo", [chest], None, "imesh"), '"imesh" cannot run the images stage.'),
                    (("retopo", "Demo", [barrel], {"sizes": [0]}), "A prop has up to 3 sizes"),
                    (("retopo", "Demo", [chest], {"method": "blender"}), "The retopology method must be one of")):
                with pytest.raises(JobError, match=re.escape(words)) as e:
                    await queue.submit(*args)
                assert e.value.status_code == 400, args

            # Without the tool's adapter (not part of this Spellcast3D), or with its tool not installed.
            missing = {key: value for key, value in adapters.items() if key != TRIFLOW}
            with pytest.raises(JobError, match="TriFlow|triflow") as e:
                await JobQueue(pm, store, pm.settings, missing, pm.client).submit("retopo", "Demo", [lamp])
            assert e.value.status_code == 400 and "not part of this Spellcast3D" in str(e.value)
            adapters[LATO2].install = "not_installed"
            with pytest.raises(ToolError, match="LATO.2 is not installed"):
                await queue.submit("retopo", "Demo", [barrel])

    asyncio.run(main())


def test_retopo_all_split_per_method_loads_each_tool_once(tmp_path: Path) -> None:
    """The page queues "Retopo All" as one job per method, LATO.2, TriFlow, then Instant Meshes, right after
    Pixal3D: each tool starts once and does all of its props; each GPU tool stops the one before it; Instant
    Meshes (CPU) starts beside LATO.2 to unwrap its UVs, and stays for its own props. A prop of one size has its
    clean mesh picked as soon as it is made."""
    store = ProjectStore(tmp_path / "projects")
    ids = props_with_methods(store, {"A": "lato2", "B": "lato2", "C": "triflow", "D": "imesh", "E": "imesh"})
    store.patch_prop("Demo", ids["C"], {"retopo": {"settings": {"sizes": [4], "uv_leniency": 0.25}}})
    seen: list[tuple[str, str, str]] = []

    async def main() -> None:
        adapters = retopo_adapters(tmp_path)
        async with manager(tmp_path, adapters) as pm:
            for tool in (LATO2, TRIFLOW):
                adapters[tool].watch = lambda tool=tool: seen.append(
                    (tool, pm.status(tool)["state"], pm.status(IMESH)["state"]))
            await pm.ensure_running("pixal3d")                          # the 3D stage's job ran just before
            queue = JobQueue(pm, store, pm.settings, adapters, pm.client)
            queue.start()
            jobs = [await queue.submit("retopo", "Demo", [ids["A"], ids["B"]], None, LATO2),
                    await queue.submit("retopo", "Demo", [ids["C"]], None, TRIFLOW),
                    await queue.submit("retopo", "Demo", [ids["D"], ids["E"]], None, IMESH)]
            done = await wait_jobs(queue, [job["id"] for job in jobs])
            assert [job["status"] for job in done] == ["done", "done", "done"], done
            assert done[0]["finished"] <= done[1]["finished"] <= done[2]["finished"]
            assert {tool: starts(pm, tool) for tool in (LATO2, TRIFLOW, IMESH)} == {LATO2: 1, TRIFLOW: 1, IMESH: 1}
            states = {tool: pm.status(tool)["state"] for tool in ("pixal3d", LATO2, TRIFLOW, IMESH)}
            assert states == {"pixal3d": "stopped", LATO2: "stopped", TRIFLOW: "ready", IMESH: "ready"}
            unwraps = (await pm.client.get(f"{pm.get(IMESH).base_url}/fake/unwraps")).json()
            await queue.close()
        # Instant Meshes unwrapped while LATO.2 (then TriFlow) held the graphics card, busy with its job (and may be
        # at the next prop's unwrap already: the job runner overlaps the two).
        assert [(tool, state) for tool, state, _ in seen] == [(LATO2, "busy")] * 2 + [(TRIFLOW, "busy")]
        assert all(imesh in ("ready", "busy") for _, _, imesh in seen)
        assert [(Path(u["src"]).name, Path(u["dst"]).name, u["uv_leniency"]) for u in unwraps] == [
            ("lato2.raw.obj", "lato2.obj", 1.0), ("lato2.raw.obj", "lato2.obj", 1.0),
            ("triflow.raw.obj", "triflow.obj", 0.25)]
        assert [prop_id for prop_id, _ in adapters[LATO2].calls] == [ids["A"], ids["B"]]
        assert adapters[TRIFLOW].bodies[0]["faces"] == 8000, "two triangles per vertex wanted"
        for name, (obj, method) in {"A": ("retopo/2k/lato2.obj", LATO2), "B": ("retopo/2k/lato2.obj", LATO2),
                                    "C": ("retopo/4k/triflow.obj", TRIFLOW), "D": ("retopo/2k/retopo.obj", IMESH),
                                    "E": ("retopo/2k/retopo.obj", IMESH)}.items():
            prop = store.get_prop("Demo", ids[name])
            retopo = prop["retopo"]
            assert (retopo["obj"], retopo["method"], retopo["source"], retopo["accepted"]) == \
                   (obj, method, "mesh/mesh.glb", True), name
            assert retopo["stats"]["has_uv"] is True and retopo_fresh(prop), name
            assert (store.prop_dir("Demo", ids[name]) / obj).read_text(encoding="utf-8").count("vt ") == 1
        assert store.get_prop("Demo", ids["A"])["retopo"]["stats"]["raw_vertices"] == 3

    asyncio.run(main())


def test_a_job_makes_the_sizes_a_prop_lacks_one_after_another_and_says_which(tmp_path: Path) -> None:
    """One clean mesh per size, in the order of the prop's boxes, each asked for in the tool's own unit (TriFlow:
    two triangles per vertex; LATO.2: vertices, at most its models' 10,000), with words that name the size. A job
    makes the sizes the prop lacks, else all of them again ("Make it again")."""
    store = ProjectStore(tmp_path / "projects")
    ids = props_with_methods(store, {"Lamp": "triflow", "Barrel": "lato2"})
    for pid in ids.values():
        store.patch_prop("Demo", pid, {"retopo": {"settings": {"sizes": [4, 22, 6]}}})

    async def main() -> None:
        adapters = retopo_adapters(tmp_path)
        async with manager(tmp_path, adapters) as pm:
            queue = JobQueue(pm, store, pm.settings, adapters, pm.client)
            queue.start()

            async def run(tool: str, name: str) -> dict[str, Any]:
                return await queue.submit("retopo", "Demo", [ids[name]], None, tool)

            (job,) = await wait_jobs(queue, [(await run(TRIFLOW, "Lamp"))["id"]])
            assert job["status"] == "done", job
            assert [body["faces"] for body in adapters[TRIFLOW].bodies] == [8000, 44_000, 12_000]
            words = [message for _fraction, message in adapters[TRIFLOW].reports]
            assert words[:3] == ["4k vertices (1 of 3) · TriFlow is making the clean mesh…",
                                 "22k vertices (2 of 3) · TriFlow is making the clean mesh…",
                                 "6k vertices (3 of 3) · TriFlow is making the clean mesh…"]
            fractions = [fraction for fraction, _message in adapters[TRIFLOW].reports]
            assert fractions == sorted(fractions) and fractions[2] < 0.8 <= fractions[3], "one share each, then the UVs"
            lamp = store.get_prop("Demo", ids["Lamp"])["retopo"]
            assert [v["obj"] for v in lamp["variants"]] == [f"retopo/{size}k/triflow.obj" for size in (4, 6, 22)]
            assert (lamp["obj"], lamp["accepted"]) == ("retopo/4k/triflow.obj", False)

            (job,) = await wait_jobs(queue, [(await run(LATO2, "Barrel"))["id"]])
            assert job["status"] == "done", job
            assert [body["vertices"] for body in adapters[LATO2].bodies] == [4000, 10_000, 6000]

            # One more size: only it is made; then nothing lacks, and all are made again.
            adapters[TRIFLOW].bodies.clear()
            store.patch_prop("Demo", ids["Lamp"], {"retopo": {"settings": {"sizes": [4, 22, 9]}}})
            await wait_jobs(queue, [(await run(TRIFLOW, "Lamp"))["id"]])
            assert [body["faces"] for body in adapters[TRIFLOW].bodies] == [18_000]
            assert adapters[TRIFLOW].reports[-2][1] == "9k vertices · TriFlow is making the clean mesh…"
            adapters[TRIFLOW].bodies.clear()
            await wait_jobs(queue, [(await run(TRIFLOW, "Lamp"))["id"]])
            assert [body["faces"] for body in adapters[TRIFLOW].bodies] == [8000, 44_000, 18_000]
            # Erased boxes: a prop that asks for no size is refused at once.
            store.patch_prop("Demo", ids["Lamp"], {"retopo": {"settings": {"sizes": []}}})
            with pytest.raises(JobError, match="asks for no clean mesh") as e:
                await run(TRIFLOW, "Lamp")
            assert e.value.status_code == 400
            await queue.close()

    asyncio.run(main())


def test_sizes_under_a_thousand_are_made_in_their_own_folders_with_each_tools_nearest_count(tmp_path: Path) -> None:
    """A size with decimals (0.25 is 250 vertices) is asked of each tool in its unit, or as the nearest count the tool
    takes: TriFlow 500 triangles at least, LATO.2 200 vertices at least, Instant Meshes every size as it is. Its folder
    and its progress words name it ("0.25k vertices (2 of 3) · ..."), and it is picked by its number."""
    store = ProjectStore(tmp_path / "projects")
    ids = props_with_methods(store, {"Cup": "triflow", "Jar": "lato2", "Mug": "imesh"})
    for pid in ids.values():
        store.patch_prop("Demo", pid, {"retopo": {"settings": {"sizes": [0.1, 0.25, 2]}}})

    async def main() -> list[dict[str, Any]]:
        adapters = retopo_adapters(tmp_path)
        async with manager(tmp_path, adapters) as pm:
            queue = JobQueue(pm, store, pm.settings, adapters, pm.client)
            queue.start()
            jobs = [await queue.submit("retopo", "Demo", [ids[name]], None, tool)
                    for name, tool in (("Cup", TRIFLOW), ("Jar", LATO2), ("Mug", IMESH))]
            done = await wait_jobs(queue, [job["id"] for job in jobs])
            assert [job["status"] for job in done] == ["done"] * 3, done
            asked = (await pm.client.get(f"{pm.get(IMESH).base_url}/fake/retopos")).json()
            await queue.close()
        assert [body["faces"] for body in adapters[TRIFLOW].bodies] == [500, 500, 4000]
        assert [body["vertices"] for body in adapters[LATO2].bodies] == [200, 250, 2000]
        assert [message for _fraction, message in adapters[TRIFLOW].reports][:3] == [
            "0.1k vertices (1 of 3) · TriFlow is making the clean mesh…",
            "0.25k vertices (2 of 3) · TriFlow is making the clean mesh…",
            "2k vertices (3 of 3) · TriFlow is making the clean mesh…"]
        return asked

    asked = asyncio.run(main())
    assert [body["target_vertices"] for body in asked] == [100, 250, 2000]
    for name, file in (("Cup", "triflow.obj"), ("Jar", "lato2.obj"), ("Mug", "retopo.obj")):
        retopo = store.get_prop("Demo", ids[name])["retopo"]
        assert [(v["size"], v["obj"]) for v in retopo["variants"]] == [
            (0.1, f"retopo/0.1k/{file}"), (0.25, f"retopo/0.25k/{file}"), (2, f"retopo/2k/{file}")], name
        assert all((store.prop_dir("Demo", ids[name]) / v["obj"]).is_file() for v in retopo["variants"]), name
    prop = store.pick_retopo("Demo", ids["Cup"], 0.25)
    assert (prop["retopo"]["obj"], prop["retopo"]["accepted"]) == ("retopo/0.25k/triflow.obj", True)
    assert store.pick_retopo("Demo", ids["Cup"], 2.0)["retopo"]["obj"] == "retopo/2k/triflow.obj", "2.0 is the size 2"
    assert store.pick_retopo("Demo", ids["Jar"], 0.1 + 0.15000000001)["retopo"]["obj"] == "retopo/0.25k/lato2.obj", \
        "a size as the hub keeps it: rounded to two decimals"
    again = ProjectStore(tmp_path / "projects").get_prop("Demo", ids["Cup"])["retopo"]
    assert again["settings"]["sizes"] == [0.1, 0.25, 2] and [v["size"] for v in again["variants"]] == [0.1, 0.25, 2]
    assert isinstance(again["settings"]["sizes"][2], int) and isinstance(again["variants"][2]["size"], int)


def test_a_size_that_fails_leaves_the_others_and_a_cancel_keeps_what_was_made() -> None:
    """variants.make_each: a size that fails is told with the result and the others are made; none made: its
    failure is the prop's. A cancel stops before the next size; a tool that must start afresh stops at once."""
    told: list[str] = []
    cancel = {"now": False}
    ctx = StageContext(project="Demo", prop_id="p_1", prop_dir=Path("."), base_url="http://x", client=None,  # type: ignore[arg-type]
                       settings={}, report=lambda f, m: None, cancelled=lambda: cancel["now"], warn=told.append)

    async def make(size: int, part: StageContext) -> int:
        if size == 22:
            raise StageError("LATO.2 placed no vertices on this model.")
        if size == 9:
            raise ToolNeedsRestart("LATO.2 has to start again after an error.")
        cancel["now"] = size == 5
        return size * 1000

    assert asyncio.run(variants.make_each(ctx, [4, 22, 6], 0.9, make)) == [(4, 4000), (6, 6000)]
    assert told == ["22k vertices: LATO.2 placed no vertices on this model."]
    with pytest.raises(StageError, match="placed no vertices"):
        asyncio.run(variants.make_each(ctx, [22], 0.9, make))
    with pytest.raises(ToolNeedsRestart):
        asyncio.run(variants.make_each(ctx, [4, 9, 6], 0.9, make))
    assert asyncio.run(variants.make_each(ctx, [5, 6, 7], 0.9, make)) == [(5, 5000)], "cancelled after the first"


def test_the_unwrap_says_what_went_wrong(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    ok = props_with_methods(store, {"Fine": "lato2"})
    no_uv = props_with_methods(store, {"Flat": "lato2"}, b"glTF\n# FAKE_NO_UV\n")
    broken = props_with_methods(store, {"Knot": "lato2"}, b"glTF\n# FAKE_UNWRAP_ERROR: The mesh is not manifold.\n")
    ids = [*ok.values(), *no_uv.values(), *broken.values()]

    async def main() -> None:
        adapters = retopo_adapters(tmp_path)
        async with manager(tmp_path, adapters) as pm:
            queue = JobQueue(pm, store, pm.settings, adapters, pm.client)
            queue.start()
            (job,) = await wait_jobs(queue, [(await queue.submit("retopo", "Demo", ids))["id"]])
            assert job["status"] == "error" and list(job["errors"]) == [ids[2]]
            assert job["errors"][ids[2]] == "The mesh is not manifold."
            flat = store.get_prop("Demo", ids[1])["retopo"]
            assert flat["stats"]["has_uv"] is False and "without UVs" in flat["stats"]["warning"]
            assert store.get_prop("Demo", ids[0])["retopo"]["stats"]["has_uv"] is True

            # Instant Meshes is not installed: a LATO.2 job is refused at once (before LATO.2 spends GPU time)...
            await pm.stop(IMESH)
            adapters[IMESH].install_status = lambda: {"status": "not_installed", "reason": "venv missing"}
            with pytest.raises(ToolError, match="LATO.2's clean meshes get their UVs from Instant Meshes: Instant "
                                                "Meshes is not installed"):
                await queue.submit("retopo", "Demo", ids[:1])
            # ...and one queued before that says so for its prop.
            adapters[IMESH].install_status = lambda: {"status": "installed"}
            waiting = await queue.submit("retopo", "Demo", ids[:1])
            adapters[IMESH].install_status = lambda: {"status": "not_installed", "reason": "venv missing"}
            (job,) = await wait_jobs(queue, [waiting["id"]])
            assert job["errors"][ids[0]].startswith("The UVs are unwrapped by Instant Meshes, which could not "
                                                    "start: Instant Meshes is not installed")
            await queue.close()

            # Without Instant Meshes in this Spellcast3D at all.
            alone = {LATO2: adapters[LATO2]}
            queue = JobQueue(pm, store, pm.settings, alone, pm.client)
            queue.start()
            (job,) = await wait_jobs(queue, [(await queue.submit("retopo", "Demo", ids[:1]))["id"]])
            assert job["errors"][ids[0]] == ("The UVs are unwrapped by Instant Meshes, which is not part of this "
                                             "Spellcast3D.")
            await queue.close()

    asyncio.run(main())
    context = StageContext(project="Demo", prop_id=ids[0], prop_dir=tmp_path, base_url="http://x",
                           client=None, settings={}, report=lambda f, m: None,  # type: ignore[arg-type]
                           cancelled=lambda: False)
    with pytest.raises(StageError, match="cannot unwrap UVs here"):
        asyncio.run(context.unwrap("a.obj", "b.obj", projects.UV_LENIENCY))


def test_waiting_and_stopping_go_by_the_jobs_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A job waits for its own tool's installation, and stopping a tool cancels only the jobs of that tool."""
    store = ProjectStore(tmp_path / "projects")
    ids = props_with_methods(store, {"A": "lato2", "B": "lato2", "D": "imesh"})
    installing = {LATO2: "Waiting for LATO.2 to finish installing (40%)"}

    async def main() -> None:
        adapters = retopo_adapters(tmp_path, "--retopo-seconds", "1.5")
        async with manager(tmp_path, adapters) as pm:
            monkeypatch.setattr(pm, "install_wait", lambda tool: installing.get(tool))
            queue = JobQueue(pm, store, pm.settings, adapters, pm.client)
            queue.start()
            lato2 = await queue.submit("retopo", "Demo", [ids["A"], ids["B"]], None, LATO2)
            imesh = await queue.submit("retopo", "Demo", [ids["D"]], None, IMESH)
            (done,) = await wait_jobs(queue, [imesh["id"]])       # another tool's job goes ahead
            assert done["status"] == "done"
            assert queue.get(lato2["id"])["progress"]["message"] == installing[LATO2]
            installing.clear()
            while queue.get(lato2["id"])["status"] != "running":
                await asyncio.sleep(0.05)
            queue.cancel_tool_jobs(IMESH)                          # Instant Meshes' Stop: not LATO.2's job
            queue.cancel_tool_jobs(TRIFLOW)
            assert queue.get(lato2["id"])["status"] == "running"
            queue.cancel_tool_jobs(LATO2)
            (final,) = await wait_jobs(queue, [lato2["id"]])
            assert final["status"] == "cancelled" and final["tool"] == LATO2
            assert len(adapters[LATO2].calls) == 1
            await queue.close()

    asyncio.run(main())


# ---- HTTP: jobs with a tool, Accept, the license --------------------------------------------------------

@pytest.fixture
def api(tmp_path: Path) -> Any:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>Spellcast3D</h1>", encoding="utf-8")
    adapters = retopo_adapters(tmp_path)
    app = create_app(adapters, settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=static, logs_dir=tmp_path / "logs",
                     gpu_probe=lambda: GpuInfo(0, "Fake RTX 3060", 8.6, 12288, 3000, "580.10"))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield app, client, adapters


def test_api_jobs_with_a_tool_the_pick_and_the_license(api: Any) -> None:
    app, client, _ = api
    store: ProjectStore = app.state.hub.store
    ids = props_with_methods(store, {"Barrel": "lato2", "Lamp": "triflow"})
    store.patch_prop("Demo", ids["Barrel"], {"retopo": {"settings": {"sizes": [2, 4]}}})
    tools = client.get("/api/status").json()["tools"]
    assert tools[TRIFLOW]["license"] == TRIFLOW_LICENSE and tools[TRIFLOW]["display_name"] == "TriFlow"
    assert tools[LATO2]["license"] is None and tools[IMESH]["license"] is None
    assert tools[TRIFLOW]["stage"] == tools[LATO2]["stage"] == "retopo"

    url = f"/api/projects/Demo/props/{ids['Barrel']}/retopo/pick"
    nothing = client.post(url, json={"size": 2})
    assert nothing.status_code == 409 and nothing.json()["detail"] == (
        "This clean mesh is not there to pick (it is not made yet, or was made from an older 3D model). Make it first.")
    split = client.post("/api/jobs", json={"stage": "retopo", "project": "Demo", "prop_ids": list(ids.values())})
    assert split.status_code == 400 and "one job per method" in split.json()["detail"]
    wrong = client.post("/api/jobs", json={"stage": "retopo", "project": "Demo", "prop_ids": [ids["Barrel"]],
                                           "tool": TRIFLOW})
    assert wrong.status_code == 400 and "Barrel is set to be made with LATO.2, not TriFlow" in wrong.json()["detail"]
    job = client.post("/api/jobs", json={"stage": "retopo", "project": "Demo", "prop_ids": [ids["Barrel"]],
                                         "tool": LATO2}).json()
    assert job["tool"] == LATO2
    deadline = time.monotonic() + 60
    while client.get(f"/api/jobs/{job['id']}").json()["status"] not in FINISHED:
        assert time.monotonic() < deadline, client.get(f"/api/jobs/{job['id']}").json()
        time.sleep(0.1)
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "done"
    prop = client.get("/api/projects/Demo").json()["props"][0]
    assert (prop["retopo"]["obj"], prop["retopo"]["method"], prop["retopo"]["accepted"]) == \
           ("retopo/2k/lato2.obj", LATO2, False)
    assert [v["size"] for v in prop["retopo"]["variants"]] == [2, 4]

    picked = client.post(url, json={"size": 4}).json()
    assert picked["retopo"]["accepted"] is True and picked["retopo"]["obj"] == "retopo/4k/lato2.obj"
    assert picked["retopo"]["stale"] is False and picked["retopo"]["method"] == LATO2
    (store.prop_dir("Demo", ids["Barrel"]) / "retopo" / "2k" / "lato2.obj").unlink()
    assert client.post(url, json={"size": 2}).status_code == 409                # its file is gone
    assert client.post(url, json={"size": 9}).status_code == 409                # none of that size
    assert client.post("/api/projects/Demo/props/p_ffffff/retopo/pick", json={"size": 2}).status_code == 404


# ---- exports -------------------------------------------------------------------------------------------

def test_saved_triflow_results_carry_a_noncommercial_warning(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    ids = props_with_methods(store, {"Lamp": "triflow", "Barrel": "lato2", "Crate": "triflow"})
    for name, method in (("Lamp", TRIFLOW), ("Barrel", LATO2), ("Crate", TRIFLOW)):
        obj = store.prop_dir("Demo", ids[name]) / "retopo" / f"{method}.obj"
        obj.parent.mkdir(parents=True, exist_ok=True)
        obj.write_text("v 0 0 0\nvt 0 0\n", encoding="utf-8")
        store.apply_stage_patch("Demo", ids[name], "retopo", {"obj": f"retopo/{method}.obj", "method": method,
                                                              "source": "mesh/mesh.glb"})
    # The crate is baked on TriFlow's low poly.
    baked = store.prop_dir("Demo", ids["Crate"]) / "projection" / "bake_001" / "projection.glb"
    baked.parent.mkdir(parents=True)
    baked.write_bytes(b"glTF-baked")
    store.apply_stage_patch("Demo", ids["Crate"], "projection", {
        "glb": "projection/bake_001/projection.glb", "fbx": None, "obj": None, "maps": [],
        "source_mesh": "mesh/mesh.glb", "source_retopo": "retopo/triflow.obj"})
    exporter = Exporter(store, FakeConverter(), lambda fmt: None)

    saved = exporter.save_all("Demo")
    assert saved.props == 3 and "Lamp\\Lamp_retopo.obj" in saved.files and "Crate\\Crate.glb" in saved.files
    # one sentence for all of them (Export all with many TriFlow props says it once)
    assert saved.warnings == ["Lamp and Crate were made with TriFlow — non-commercial use only."]
    assert exporter.save_prop("Demo", ids["Crate"], "projection", "glb").warnings == [
        "Crate: made with TriFlow — non-commercial use only."]
    assert exporter.save_prop("Demo", ids["Crate"], "mesh", "glb").warnings == []     # the 3D model: Pixal3D's
    # A new clean mesh from Instant Meshes: the bake still is TriFlow's (its low poly's file says so).
    store.apply_stage_patch("Demo", ids["Crate"], "retopo", {"obj": "retopo/retopo.obj", "method": IMESH})
    crate = store.get_prop("Demo", ids["Crate"])
    assert exports.low_poly_method(crate, "retopo/triflow.obj") == TRIFLOW
    assert exports.low_poly_method(crate, "retopo/retopo.obj") == IMESH
    assert exports.noncommercial_note(crate, "retopo/retopo.obj") is None
    assert exports.made_with_note("TriFlow", ["A", "B", "C"]) == \
        "A, B and C were made with TriFlow — non-commercial use only."
    assert exports.made_with_note("TriFlow", list("ABCDEF")) == \
        "A, B and 4 more were made with TriFlow — non-commercial use only."
    assert exporter.save_prop("Demo", ids["Crate"], "projection", "glb").warnings == [
        "Crate: made with TriFlow — non-commercial use only."]


# ---- one prop's unwrap beside the next prop's turn; the next stage's tool started early ------------------------

def test_a_props_uv_unwrap_overlaps_the_next_props_turn_on_the_card(tmp_path: Path) -> None:
    """JobQueue._run_props: once a prop's mesh is made and Instant Meshes unwraps its UVs (CPU), the next prop goes to
    TriFlow at once; the unwraps queue up behind, as many as there are, and the card never waits. The page sees them
    all at work: the ones behind "Unwrapping the UVs", the one on the card with its own words."""
    store = ProjectStore(tmp_path / "projects")
    ids = props_with_methods(store, {"A": "triflow", "B": "triflow", "C": "triflow"},
                             mesh=b"glTF\n# FAKE_UNWRAP_SECONDS:0.8\n")
    seen: list[dict[str, Any]] = []

    async def main() -> None:
        adapters = retopo_adapters(tmp_path)
        async with manager(tmp_path, adapters) as pm:
            queue = JobQueue(pm, store, pm.settings, adapters, pm.client)
            queue.start()
            job_id = (await queue.submit("retopo", "Demo", [ids["A"], ids["B"], ids["C"]], None, TRIFLOW))["id"]
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                snap = queue.get(job_id)
                working = {p: e for p, e in snap["props"].items() if e["state"] == "working"}
                if len(working) >= 2:
                    seen.append(snap)
                if snap["status"] in FINISHED:
                    break
                await asyncio.sleep(0.02)
            (job,) = await wait_jobs(queue, [job_id])
            await queue.close()
        assert job["status"] == "done", job
        marks = {(kind, prop): t for kind, prop, t in adapters[TRIFLOW].marks}
        a, b, c = ids["A"], ids["B"], ids["C"]
        assert marks[("start", b)] < marks[("unwrapped", a)], "B went to the card while A's UVs were unwrapped"
        assert marks[("start", c)] < marks[("unwrapped", b)], "C too: the card waits for no unwrap"
        assert marks[("start", c)] < marks[("unwrapped", c)]
        assert seen, "the page saw two props at work"
        both = seen[0]["props"]
        behind = [p for p, e in both.items() if e["message"] == jobs_module.UNWRAPPING]
        assert behind and all(both[p]["fraction"] == jobs_module.UNWRAP_FRACTION for p in behind)
        assert seen[0]["progress"]["message"] != jobs_module.UNWRAPPING     # the job's words are the lead prop's
        for name in ("A", "B", "C"):
            prop = store.get_prop("Demo", ids[name])
            assert prop["retopo"]["stats"]["has_uv"] is True and retopo_fresh(prop), name

    asyncio.run(main())


def test_a_omnitexture_job_done_keeps_qwen_image_on_the_card(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """The image generator stays loaded after Omni-Texture: the next stage's networks load only when its work is
    asked for. An
    Omni-Texture job - the stage's own, or a repaint of what the artist masked - hands the card to no one; the
    retopology tool takes it when its first job comes."""
    monkeypatch.setattr(jobs_module, "HANDOFF_SECONDS", 0.2)
    store = ProjectStore(tmp_path / "projects")
    ids = props_with_methods(store, {"A": "lato2"})

    async def main() -> None:
        adapters = retopo_adapters(tmp_path, **{paths.QWEN: FakeAdapter(paths.QWEN, "omnitexture", True, tmp_path)})
        async with manager(tmp_path, adapters) as pm:
            queue = JobQueue(pm, store, pm.settings, adapters, pm.client)
            queue.start()
            for params in (None, {"retouch": {"mask": "omnitexture/mask.png"}}):
                (job,) = await wait_jobs(queue, [(await queue.submit("omnitexture", "Demo", [ids["A"]], params))["id"]])
                assert job["status"] == "done"
                await asyncio.sleep(1.0)
                assert pm.status(paths.QWEN)["holds_gpu"] and pm.status(LATO2)["state"] == "stopped"
            assert starts(pm, LATO2) == 0 and starts(pm, paths.QWEN) == 1
            (job,) = await wait_jobs(queue, [(await queue.submit("retopo", "Demo", [ids["A"]], None, LATO2))["id"]])
            assert job["status"] == "done" and pm.status(LATO2)["holds_gpu"]       # "Retopologize": its job
            assert pm.status(paths.QWEN)["state"] == "stopped"
            await queue.close()

    asyncio.run(main())


def test_the_next_stages_tool_starts_early_once_a_job_is_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """jobs.NEXT_STAGE: a 3D job that is done hands the card to Qwen for Omni-Texture after HANDOFF_SECONDS,
    stopping the idle Pixal3D; a job queued within the grace keeps the card for its own tool instead; a retopo
    job done starts nothing new (Instant Meshes, its next stage's tool, runs)."""
    monkeypatch.setattr(jobs_module, "HANDOFF_SECONDS", 0.4)
    store = ProjectStore(tmp_path / "projects")
    ids = props_with_methods(store, {"A": "lato2", "B": "lato2"})

    async def main() -> None:
        adapters = retopo_adapters(tmp_path, **{paths.QWEN: FakeAdapter(paths.QWEN, "omnitexture", True, tmp_path)})
        async with manager(tmp_path, adapters) as pm:
            queue = JobQueue(pm, store, pm.settings, adapters, pm.client)
            queue.start()
            (job,) = await wait_jobs(queue, [(await queue.submit("mesh", "Demo", [ids["A"]]))["id"]])
            assert job["status"] == "done" and pm.status("pixal3d")["holds_gpu"]
            await wait_state(pm, paths.QWEN, {"ready"}, timeout=15)  # the hand-off: Qwen up, Pixal3D gone
            assert pm.status("pixal3d")["state"] == "stopped" and pm.status(paths.QWEN)["holds_gpu"]
            assert starts(pm, paths.QWEN) == 1

            (job,) = await wait_jobs(queue, [(await queue.submit("mesh", "Demo", [ids["B"]]))["id"]])
            assert job["status"] == "done" and pm.status(paths.QWEN)["state"] == "stopped"    # Pixal3D took the card back
            store.patch_prop("Demo", ids["A"], {"retopo": {"settings": {"method": IMESH}}})
            cpu = await queue.submit("retopo", "Demo", [ids["A"]], None, IMESH)          # within the grace
            (job,) = await wait_jobs(queue, [cpu["id"]])
            assert job["status"] == "done"
            await asyncio.sleep(0.8)
            assert pm.status(paths.QWEN)["state"] == "stopped" and pm.status("pixal3d")["holds_gpu"]   # no hand-off
            assert pm.status(IMESH)["state"] == "ready" and starts(pm, LATO2) == 0 and starts(pm, paths.QWEN) == 1
            await queue.close()

    asyncio.run(main())
