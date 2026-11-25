"""A mesh job hands Pixal3D all its props at once (adapters/pixal3d.py run_batch, jobs.py _run_batch): where each
prop is while the tool takes them through its steps, a prop that fails alone, a cancel, a tool that dies, hangs or
stops answering, and what is left on disk afterwards. Pixal3D is tests/fake_tool.py, which answers like its
launcher's batch routes.

Run with the hub venv:  <venv>\\Scripts\\python.exe -m pytest code\\tests -q
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from spellcast import jobs as jobs_mod
from spellcast import multiview, paths
from spellcast.adapters import pixal3d
from spellcast.adapters.base import BatchContext, BatchItem, ToolNeedsRestart
from spellcast.adapters.pixal3d import MADE_SLOWLY, NO_GLB, Pixal3DAdapter
from spellcast.gpu import GpuInfo
from spellcast.jobs import JobQueue
from spellcast.procman import ProcessManager
from spellcast.projects import ProjectStore, tileable_prop
from spellcast.server import create_app
from test_core import PNG, FakeAdapter, make_settings, manager

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Spellcast3D is Windows-only")

FINISHED = ("done", "error", "cancelled")
PART_KEYS = ["cutout", "camera", "features", "ss", "cond_lr", "lr", "upsample", "cond_hr", "hr", "tex", "finish"]
BLAMED = "Pixal3D stopped while working on this prop. Check its log (Settings > Debugging), then make this prop again."
NOT_REACHED = "Pixal3D stopped before this prop's turn. Check its log, then run this prop again."


@pytest.fixture(autouse=True)
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hub asks the fake often, and the multi-view models count as downloaded."""
    monkeypatch.setattr(pixal3d, "POLL_INTERVAL", 0.05)
    monkeypatch.setattr(jobs_mod, "WORK_DIR_RETRY", 0.1)
    monkeypatch.setattr(multiview, "available", lambda *args: True)


class Pixal3DOnFake(Pixal3DAdapter):
    """The real Pixal3D adapter, driving fake_tool.py instead of Pixal3D."""

    def __init__(self, tmp_path: Path, *args: str):
        self.fake = FakeAdapter("pixal3d", "mesh", True, tmp_path, args=args)
        self.starts = 0

    def install_status(self) -> dict[str, Any]:
        return {"status": "installed"}

    def supported(self) -> tuple[bool, str]:
        return True, ""

    def launch_spec(self, port: int, settings: dict[str, Any]) -> Any:
        self.starts += 1
        return self.fake.launch_spec(port, settings)


def project(tmp_path: Path, *pictures: bytes | int) -> tuple[ProjectStore, list[str]]:
    """Project "P" with one prop per entry: the text that follows its picture's PNG header (what the fake
    reads), or the number of pictures a prop made from that many views has."""
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    ids = []
    for index, picture in enumerate(pictures):
        pid = store.add_prop("P", f"Prop {index}")["id"]
        if isinstance(picture, int):
            store.add_images("P", pid, [(PNG, f"view{n}.png") for n in range(picture)])
        else:
            store.add_image("P", pid, PNG + picture)
        ids.append(pid)
    return store, ids


@contextlib.asynccontextmanager
async def pixal3d_hub(tmp_path: Path, store: ProjectStore, *fake_args: str, settings: dict[str, Any] | None = None,
                      ) -> AsyncIterator[tuple[ProcessManager, JobQueue, Pixal3DOnFake]]:
    adapter = Pixal3DOnFake(tmp_path, *fake_args)
    async with manager(tmp_path, {"pixal3d": adapter}, settings) as pm:
        queue = JobQueue(pm, store, pm.settings, {"pixal3d": adapter}, pm.client)
        queue.start()
        try:
            yield pm, queue, adapter
        finally:
            await queue.close()


async def watch(queue: JobQueue, job_id: str, each: Callable[[dict[str, Any]], Any] | None = None,
                timeout: float = 60.0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Follow a job until it ended -> (how it ended, every different state it was seen in on the way)."""
    seen: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout
    while True:
        job = queue.get(job_id)
        if not seen or job != seen[-1]:
            seen.append(job)
            if each is not None:
                each(job)
        if job["status"] in FINISHED:
            return job, seen
        assert time.monotonic() < deadline, job
        await asyncio.sleep(0.01)


async def until(queue: JobQueue, job_id: str, reached: Callable[[dict[str, Any]], bool], timeout: float = 30.0,
                ) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while not reached(job := queue.get(job_id)):
        assert job["status"] not in FINISHED and time.monotonic() < deadline, job
        await asyncio.sleep(0.01)
    return job


async def asked(pm: ProcessManager, what: str = "batches") -> Any:
    return (await pm.client.get(f"{pm.get('pixal3d').base_url}/fake/{what}")).json()


def states(job: dict[str, Any]) -> list[str]:
    return [job["props"][prop_id]["state"] for prop_id in job["prop_ids"]]


def leftovers(store: ProjectStore) -> list[str]:
    """What must be gone when a job has ended: partly written models and the job's scratch folder."""
    root = store.project_dir("P")
    return [str(p.relative_to(root)) for p in [*root.rglob("*.partial.glb"), *root.glob("temp"), *root.glob("temp/*")]]


def mesh_of(store: ProjectStore, prop_id: str) -> dict[str, Any]:
    return store.get_prop("P", prop_id)["mesh"]


# ---- a job from start to end ------------------------------------------------------------------------

def test_props_wait_work_and_rest_step_by_step_and_each_model_is_there_at_once(tmp_path: Path) -> None:
    store, ids = project(tmp_path, b"", 2, b"FAKE_RETRY FAKE_SPILL")
    for prop_id in ids[:2]:                 # the views an earlier model of each was made from
        old = store.prop_dir("P", prop_id) / "mesh" / "views"
        old.mkdir(parents=True)
        (old / "0_front.png").write_bytes(PNG)
    work_dirs: list[list[str]] = []
    early: list[tuple[str, bool]] = []

    def each(job: dict[str, Any]) -> None:
        temp = store.project_dir("P") / "temp"
        if job["status"] == "running" and temp.is_dir():
            work_dirs.append(sorted(p.name for p in temp.iterdir()))
        made = [p for p in job["prop_ids"] if job["props"][p]["state"] == "done"]
        if made and len(made) < 3 and not early:        # the first model, while the others are still worked on
            early.append((mesh_of(store, made[0]).get("glb"), (store.prop_dir("P", made[0]) / "mesh/mesh.glb").is_file()))

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.08", "--load-seconds", "0.15") as (pm, queue, _):
            job = await queue.submit("mesh", "P", ids)
            assert states(job) == ["waiting"] * 3 and job["part"] is None
            final, seen = await watch(queue, job["id"], each)
            assert final["status"] == "done" and final["errors"] == {} and final["warnings"] == {}
            assert final["progress"] == {"done": 3, "total": 3, "fraction": 1.0, "message": "Done."}
            assert states(final) == ["done"] * 3 and final["part"] is None and final["current_prop"] is None
            assert all(s["prop_ids"] == ids for s in seen)              # the order the artist chose, all the way

            running = [s for s in seen if s["status"] == "running"]
            for prop_id in ids:
                steps = [s["props"][prop_id]["state"] for s in running]
                order = [state for i, state in enumerate(steps) if i == 0 or state != steps[i - 1]]
                assert order[0] == "waiting" and order[-1] == "done", order
                assert set(order) == {"waiting", "working", "resting", "done"}, order
                assert order.count("waiting") == 1 and order.count("done") == 1     # neither comes back
                assert order.count("resting") >= 3, order                           # it rests between its steps
                fractions = [s["props"][prop_id]["fraction"] for s in running]
                assert fractions == sorted(fractions) and fractions[-1] == 1.0
            # They interleave: while one is worked on, another one that has begun waits for its next step.
            assert any(sorted(states(s)) == ["resting", "resting", "working"] for s in running)
            assert all(states(s).count("working") <= 1 for s in running)
            # The step all of them are taken through, and the prop on the card.
            parts = [s["part"] for s in running if s["part"]]
            assert all(set(part) == {"key", "label", "position", "of"} for part in parts)
            keys = [part["key"] for part in parts]
            assert [key for i, key in enumerate(keys) if i == 0 or key != keys[i - 1]] == PART_KEYS
            assert {(part["position"], part["of"]) for part in parts} == {(1, 3), (2, 3), (3, 3)}
            assert any(part == {"key": "hr", "label": "Refining shape details", "position": 2, "of": 3}
                       for part in parts)
            currents = [s["current_prop"] for s in running if s["current_prop"]]
            assert currents[0] == ids[1] and set(currents) == set(ids)     # Pixal3D begins with the prop that has views
            said = {s["progress"]["message"] for s in running}
            assert {m for m in said if m.startswith("Refining shape details · step ")} <= {
                f"Refining shape details · step {step} of 4" for step in range(1, 5)}
            assert any(m.startswith("Refining shape details · step ") for m in said)
            assert all(s["progress"]["message"] == s["props"][s["current_prop"]]["message"] for s in running
                       if s["current_prop"] and s["props"][s["current_prop"]]["state"] == "working"
                       and s["props"][s["current_prop"]]["message"])
            assert any(s["progress"]["message"] == "Loading the texture model…" for s in running)
            # The page is told what the tool reads from disk, and that nothing is being stopped, as two keys
            # of their own: it never reads that from the job's words.
            assert any(s["loading"] == "texture model" for s in running)
            assert final["loading"] == "" and not any(s["stopping"] for s in seen)
            assert all(0 <= a["progress"]["fraction"] <= b["progress"]["fraction"] <= 1
                       for a, b in zip(running, running[1:]))
            assert any(0 < s["progress"]["done"] < 3 for s in running)

            # Each model was in its prop, and on disk, while the others were still worked on.
            assert early == [("mesh/mesh.glb", True)]
            # One request for all of them, with the job's own scratch folder, which is gone now.
            (request,) = await asked(pm)
            work = store.project_dir("P") / "temp" / f"mesh-{job['id']}"
            assert request["batch_id"] == job["id"] and Path(request["work_dir"]) == work.resolve()
            assert request["wave_size"] == 0 and request["resume"] is False
            assert request["spill"] is False                # pixal3d.allow_spill is off unless the artist allows it
            assert [item["id"] for item in request["items"]] == ids
            assert [work.name] in work_dirs and all(names in ([work.name], []) for names in work_dirs)
            assert leftovers(store) == []

    asyncio.run(main())
    for prop_id in ids:
        mesh = mesh_of(store, prop_id)
        assert mesh["glb"] == "mesh/mesh.glb" and mesh["stats"]["seconds"] == 0.1
        assert (store.prop_dir("P", prop_id) / "mesh" / "mesh.glb").read_bytes() == b"glTF-generated"
    assert mesh_of(store, ids[1])["stats"]["mode"] == "multi" and mesh_of(store, ids[1])["stats"]["views"] == 2
    # A model made from one picture leaves no earlier model's views behind; one made from views keeps its folder.
    assert not (store.prop_dir("P", ids[0]) / "mesh" / "views").exists()
    assert (store.prop_dir("P", ids[1]) / "mesh" / "views").is_dir()
    # What the artist should know about a result is in its stats only (nothing is toasted when the job ends).
    assert mesh_of(store, ids[2])["stats"]["warning"] == f"The shape needed 3 tries. {MADE_SLOWLY}"
    assert mesh_of(store, ids[2])["stats"]["ss_retries"] == 2 and mesh_of(store, ids[0])["stats"]["warning"] is None


def test_a_prop_that_fails_fails_alone_and_says_so_at_once(tmp_path: Path) -> None:
    store, ids = project(tmp_path, b"", b"FAKE_ERROR:The shape could not be blocked out.", b"FAKE_OOM", b"")
    store.patch_prop("P", ids[2], {"mesh": {"settings": {"resolution": 1536}}})
    store.patch_prop("P", ids[3], {"mesh": {"settings": {"seed": "many"}}})     # the hub's own check: not sent

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.05") as (pm, queue, _):
            job = await queue.submit("mesh", "P", ids)
            final, seen = await watch(queue, job["id"])
            assert final["status"] == "error" and states(final) == ["done", "failed", "failed", "failed"]
            assert final["progress"]["message"] == "3 of 4 props failed. Each prop shows what went wrong."
            assert final["progress"]["done"] == 4 and final["progress"]["fraction"] == 1.0
            # A prop that failed is through, however far it had come (as in a job that runs prop by prop).
            assert [final["props"][p]["fraction"] for p in ids] == [1.0] * 4
            assert final["errors"] == {
                ids[1]: "The shape could not be blocked out.",
                ids[2]: "This prop needs more video memory than the card has free. Close other programs that use the "
                        "graphics card, or lower Quality.",
                ids[3]: "A mesh setting is not a number. Check seed, quality, detail and texture size of this prop "
                        "in 3D."}
            # At the lowest quality there is no lower one to offer.
            assert pixal3d._oom_message(1024) == ("This prop needs more video memory than the card has free. Close "
                                                  "other programs that use the graphics card.")
            # Each error is in the job the moment it happens, while the job goes on with the others.
            for prop_id in ids[1:]:
                first = next(s for s in seen if prop_id in s["errors"])
                assert first["status"] == "running" and first["props"][prop_id]["state"] == "failed"
                assert first["props"][ids[0]]["state"] != "done"
            assert [item["id"] for item in (await asked(pm))[0]["items"]] == ids[:3]
            assert pm.status("pixal3d")["state"] == "ready"

    asyncio.run(main())
    assert mesh_of(store, ids[0])["glb"] == "mesh/mesh.glb" and leftovers(store) == []
    assert all(mesh_of(store, prop_id)["glb"] is None for prop_id in ids[1:])


def test_a_tileable_texture_gets_no_3d_model_and_the_prop_after_it_is_an_object_again(tmp_path: Path) -> None:
    store, ids = project(tmp_path, b"", b"")
    texture = lambda: tileable_prop(store.get_prop("P", ids[1]))    # noqa: E731
    store.patch_prop("P", ids[1], {"images": {"settings": {"tileable": True}}})
    assert [tileable_prop(store.get_prop("P", prop_id)) for prop_id in ids] == [False, True]
    # FLUX.3 draws no tileable picture: with it, the tick makes no texture of the prop.
    store.patch_prop("P", ids[1], {"images": {"settings": {"qwen_model": "flux3"}}})
    assert texture() is False
    store.patch_prop("P", ids[1], {"images": {"settings": {"qwen_model": "turbo"}}})
    assert texture() is True
    # A prop added after a texture is an object: the tick is the texture's own.
    after = store.add_prop("P", "After")
    assert (after["images"]["settings"]["tileable"], after["images"]["settings"]["qwen_model"]) == (False, "turbo")
    assert store.ensure_draft("P")["images"]["settings"]["tileable"] is False

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.05") as (pm, queue, _):
            job = await queue.submit("mesh", "P", ids)
            final, _seen = await watch(queue, job["id"])
            assert states(final) == ["done", "failed"]
            assert final["errors"] == {ids[1]: "This prop is a tileable texture, which gets no 3D model. To make one "
                                               "from its picture, untick Tileable in Images first."}
            assert [item["id"] for item in (await asked(pm))[0]["items"]] == ids[:1], "never sent to Pixal3D"

    asyncio.run(main())
    assert mesh_of(store, ids[0])["glb"] == "mesh/mesh.glb" and mesh_of(store, ids[1])["glb"] is None


# ---- cancel -----------------------------------------------------------------------------------------

def test_a_cancel_keeps_the_finished_props_and_is_repeated_when_it_gets_lost(tmp_path: Path) -> None:
    """With pixal3d.wave_size 1 the props are made one after another; the first cancel request is lost."""
    store, ids = project(tmp_path, b"", b"", b"")

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.1", "--batch-lose-cancels", "1",
                               settings={"pixal3d": {"wave_size": 1, "allow_spill": True}}) as (pm, queue, adapter):
            job = await queue.submit("mesh", "P", ids)
            await until(queue, job["id"], lambda j: j["props"][ids[0]]["state"] == "done")
            assert (await asked(pm))[0]["wave_size"] == 1 and (await asked(pm))[0]["spill"] is True
            stopping = queue.cancel(job["id"])
            assert stopping["stopping"] is True and stopping["progress"]["message"] == jobs_mod.CANCELLING_BATCH
            final, seen = await watch(queue, job["id"])
            assert final["status"] == "cancelled" and final["errors"] == {} and final["stopping"] is False
            assert states(final) == ["done", "cancelled", "cancelled"]
            assert final["progress"]["message"] == "Cancelled after 1 of 3 props. What was already finished is kept."
            assert final["progress"]["done"] == 1 and final["part"] is None and final["current_prop"] is None
            # A prop that was made is through; one that never ran has nothing to show (as in a one-by-one job).
            assert [final["props"][p]["fraction"] for p in ids] == [1.0, 0.0, 0.0]
            # No step's words in between, and the page hears all the way that the job is being stopped.
            assert all(s["progress"]["message"].startswith(("Stopping", "Cancelled")) for s in seen)
            assert all(s["stopping"] is (s["status"] == "running") for s in seen)
            cancels = await asked(pm, "cancels")
            assert cancels["lost"] == 1 and cancels["received"] >= 2
            # Nothing was started or sent again: the same process, one request.
            assert adapter.starts == 1 and len(await asked(pm)) == 1 and pm.status("pixal3d")["state"] == "ready"

    asyncio.run(main())
    assert mesh_of(store, ids[0])["glb"] == "mesh/mesh.glb" and mesh_of(store, ids[1])["glb"] is None
    assert leftovers(store) == []


def test_a_cancel_waits_while_a_mesh_is_built_and_keeps_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After a cancel Pixal3D finishes the mesh it is building: its decode and UV unwrap can keep Python busy for
    minutes without a word (FAKE_SLOW_SAVE: 2 s here, against a CANCEL_STALL of 0.5 s). It is not stopped for that,
    and the model is kept."""
    store, ids = project(tmp_path, b"FAKE_SLOW_SAVE", b"")
    monkeypatch.setattr(pixal3d, "CANCEL_STALL", 0.5)

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.05") as (pm, queue, adapter):
            job = await queue.submit("mesh", "P", ids)
            await until(queue, job["id"], lambda j: (j["part"] or {}).get("key") == "finish"
                        and j["current_prop"] == ids[0])
            assert queue.cancel(job["id"])["progress"]["message"] == jobs_mod.CANCELLING_BATCH
            final, _ = await watch(queue, job["id"])
            assert final["status"] == "cancelled" and states(final) == ["done", "cancelled"] and final["errors"] == {}
            assert final["progress"]["message"] == "Cancelled after 1 of 2 props. What was already finished is kept."
            assert adapter.starts == 1 and pm.status("pixal3d")["state"] == "ready"      # never stopped for it

    asyncio.run(main())
    assert mesh_of(store, ids[0])["glb"] == "mesh/mesh.glb" and leftovers(store) == []


def test_a_cancel_that_pixal3d_does_not_obey_stops_it_and_never_starts_it_again(tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    store, ids = project(tmp_path, b"", b"FAKE_STALL")
    monkeypatch.setattr(pixal3d, "CANCEL_STALL", 0.6)

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.02") as (pm, queue, adapter):
            job = await queue.submit("mesh", "P", ids)
            await until(queue, job["id"], lambda j: (j["part"] or {}).get("key") == "lr" and j["current_prop"] == ids[1])
            await asyncio.sleep(0.3)
            assert queue.get(job["id"])["status"] == "running"      # it stands still, and nobody minds yet
            asked_at = time.monotonic()
            queue.cancel(job["id"])
            final, _ = await watch(queue, job["id"])
            assert 0.5 < time.monotonic() - asked_at < 10
            assert final["status"] == "cancelled" and states(final) == ["cancelled", "cancelled"]
            assert final["errors"] == {} and final["progress"]["message"].startswith("Cancelled after 0 of 2 props.")
            await asyncio.sleep(0.5)
            assert pm.status("pixal3d")["state"] == "stopped" and adapter.starts == 1

    asyncio.run(main())
    assert leftovers(store) == []


def test_closing_spellcast3d_in_the_middle_ends_the_job_and_leaves_nothing_behind(tmp_path: Path) -> None:
    store, ids = project(tmp_path, b"", b"")

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.3") as (pm, queue, _):
            job = await queue.submit("mesh", "P", ids)
            await until(queue, job["id"], lambda j: "resting" in states(j))
            assert (store.project_dir("P") / "temp" / f"mesh-{job['id']}").is_dir()
            await queue.close()
            final = queue.get(job["id"])
            assert final["status"] == "cancelled" and states(final) == ["cancelled", "cancelled"]
            assert final["progress"]["message"] == "Cancelled because Spellcast3D was closed."
            assert final["part"] is None and final["current_prop"] is None and final["errors"] == {}
            assert leftovers(store) == []

    asyncio.run(main())


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, ProjectStore, Pixal3DOnFake]]:
    config_file = tmp_path / "spellcast3d-config.ini"       # the template's choices, never the Spellcast3D folder's
    config_file.write_bytes(paths.CONFIG_TEMPLATE.read_bytes())
    monkeypatch.setenv("SPELLCAST_CONFIG", str(config_file))
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>Spellcast3D</h1>", encoding="utf-8")
    store = ProjectStore(tmp_path / "projects")
    adapter = Pixal3DOnFake(tmp_path, "--part-seconds", "0.2")
    adapters = {"forge": FakeAdapter("forge", "images", True, tmp_path), "pixal3d": adapter,
                "imesh": FakeAdapter("imesh", "retopo", False, tmp_path)}
    app = create_app(adapters, settings=make_settings(tmp_path), store=store, static_dir=static,
                     logs_dir=tmp_path / "logs", gpu_probe=lambda: GpuInfo(0, "Fake RTX 3060", 8.6, 12288, 3000, "580.10"))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client, store, adapter


def test_stopping_pixal3d_in_settings_ends_the_job_cancelled_and_it_stays_stopped(api: Any, tmp_path: Path) -> None:
    client, _, adapter = api
    store, ids = project(tmp_path, b"", b"", b"")
    job = client.post("/api/jobs", json={"stage": "mesh", "project": "P", "prop_ids": ids}).json()
    assert set(job) == {"id", "stage", "tool", "project", "prop_ids", "status", "current_prop", "progress", "props",
                        "part", "loading", "stopping", "errors", "warnings", "created", "finished"}
    assert job["props"] == {prop_id: {"state": "waiting", "fraction": 0.0, "message": ""} for prop_id in ids}
    deadline = time.monotonic() + 30
    while "working" not in states(job := client.get(f"/api/jobs/{job['id']}").json()):
        assert time.monotonic() < deadline and job["status"] not in FINISHED, job
        time.sleep(0.02)
    assert client.get(f"/files/P/temp/mesh-{job['id']}/{ids[0]}/state.bin").status_code == 404     # never served
    assert (store.project_dir("P") / "temp" / f"mesh-{job['id']}" / ids[0] / "state.bin").is_file()

    assert client.post("/api/tools/pixal3d/stop").json()["state"] in ("stopping", "stopped")
    while (job := client.get(f"/api/jobs/{job['id']}").json())["status"] not in FINISHED:
        assert time.monotonic() < deadline, job
        time.sleep(0.02)
    assert job["status"] == "cancelled" and states(job) == ["cancelled"] * 3 and job["errors"] == {}
    assert job["progress"]["message"] == "Cancelled after 0 of 3 props. What was already finished is kept."
    time.sleep(1.0)
    assert client.get("/api/status").json()["tools"]["pixal3d"]["state"] == "stopped" and adapter.starts == 1
    assert leftovers(store) == []


# ---- a tool that has to start afresh ----------------------------------------------------------------

@pytest.mark.parametrize("mark", ["FAKE_DIE", "FAKE_BROKEN"])
def test_the_prop_on_the_card_fails_when_pixal3d_stops_and_the_others_are_sent_again(tmp_path: Path,
                                                                                    mark: str) -> None:
    """FAKE_DIE: the process ends while that prop is worked on. FAKE_BROKEN: it says itself that it has to
    start again (its GPU context broke)."""
    store, ids = project(tmp_path, b"", mark.encode(), b"")

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.03") as (pm, queue, adapter):
            job = await queue.submit("mesh", "P", ids)
            first_pid = None

            def each(now: dict[str, Any]) -> None:
                nonlocal first_pid
                if first_pid is None and now["status"] == "running":
                    first_pid = pm.get("pixal3d").proc.pid

            final, seen = await watch(queue, job["id"], each)
            assert final["status"] == "error" and final["errors"] == {ids[1]: BLAMED}
            assert states(final) == ["done", "failed", "done"]
            assert final["progress"]["message"] == "1 of 3 props failed. Each prop shows what went wrong."
            # Started afresh once; the others went to the new process, into the same scratch folder.
            assert adapter.starts == 2 and pm.get("pixal3d").proc.pid != first_pid
            assert pm.status("pixal3d")["state"] == "ready"
            (again,) = await asked(pm)
            assert again["resume"] is True and [item["id"] for item in again["items"]] == [ids[0], ids[2]]
            assert again["work_dir"] == str(store.project_dir("P") / "temp" / f"mesh-{job['id']}")
            assert again["batch_id"] == job["id"]
            # In between the props that are sent again wait again, and the job says what it does.
            waiting = [s for s in seen if s["props"][ids[1]]["state"] == "failed" and s["status"] == "running"
                       and states(s)[::2] == ["waiting", "waiting"]]
            assert waiting and any(s["progress"]["message"] == "Starting Pixal3D again…" for s in waiting)

    asyncio.run(main())
    assert mesh_of(store, ids[0])["glb"] == mesh_of(store, ids[2])["glb"] == "mesh/mesh.glb"
    assert mesh_of(store, ids[1])["glb"] is None and leftovers(store) == []


def test_the_scratch_folder_outlives_the_restart_so_sampled_props_are_only_saved_again(tmp_path: Path) -> None:
    """Pixal3D keeps a prop's result in the job's scratch folder once its texture is sampled. When it dies
    while it saves the second prop's mesh, the third is sent again, into the same folder: Pixal3D finds its
    result there and has only the mesh left to build (its bar goes on from where meshes begin, 0.72)."""
    store, ids = project(tmp_path, b"", b"FAKE_SAVE_DIE", b"")

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.03") as (pm, queue, adapter):
            final, seen = await watch(queue, (await queue.submit("mesh", "P", ids))["id"])
            assert states(final) == ["done", "failed", "done"] and final["errors"] == {ids[1]: BLAMED}
            assert adapter.starts == 2
            (again,) = await asked(pm)
            assert again["resume"] is True and [item["id"] for item in again["items"]] == [ids[2]]
            # Pixal3D counts only the seconds of the run that finished a prop, so a prop it picked up again
            # tells no time at all rather than a fraction of the truth.
            assert mesh_of(store, ids[0])["stats"]["seconds"] == 0.1
            assert mesh_of(store, ids[2])["stats"]["seconds"] is None
            sent_again = next(i for i, s in enumerate(seen) if s["progress"]["message"] == "Starting Pixal3D again…")
            assert seen[sent_again]["props"][ids[2]] == {"state": "waiting", "fraction": 0.0, "message": ""}
            worked = [s["props"][ids[2]] for s in seen[sent_again:] if s["props"][ids[2]]["state"] == "working"]
            assert worked and all(entry["fraction"] >= 0.72 for entry in worked)
            assert all(entry["message"].startswith("Building the 3D mesh") for entry in worked)

    asyncio.run(main())
    assert mesh_of(store, ids[0])["glb"] == mesh_of(store, ids[2])["glb"] == "mesh/mesh.glb"
    assert leftovers(store) == []


def test_a_model_saved_just_before_pixal3d_died_is_kept_and_the_prop_after_it_is_blamed(tmp_path: Path) -> None:
    """Pixal3D saves the first prop's model, begins the second prop and dies, without a poll answered in between
    (FAKE_CRASH_AFTER_SAVE): the hub's last answer still has the first prop on the card. Pixal3D's own note in the
    work folder says better: the first one's model is saved, and the second one is blamed."""
    store, ids = project(tmp_path, b"FAKE_CRASH_AFTER_SAVE", b"", b"")

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.08") as (pm, queue, adapter):
            final, seen = await watch(queue, (await queue.submit("mesh", "P", ids))["id"])
            again = next(i for i, s in enumerate(seen) if s["progress"]["message"] == "Starting Pixal3D again…")
            before = [s for s in seen[:again] if s["status"] == "running"]
            # The bug this guards against: the last word was "the first prop is on the card, building its mesh".
            assert any(s["current_prop"] == ids[0] and (s["part"] or {}).get("key") == "finish" for s in before)
            assert not any(s["current_prop"] == ids[1] and (s["part"] or {}).get("key") == "finish" for s in before)
            assert states(final) == ["done", "failed", "done"] and final["errors"] == {ids[1]: BLAMED}
            assert adapter.starts == 2
            (resent,) = await asked(pm)
            assert [item["id"] for item in resent["items"]] == [ids[2]]         # the first one is not made again

    asyncio.run(main())
    first = mesh_of(store, ids[0])
    assert first["glb"] == "mesh/mesh.glb" and first["stats"]["seconds"] == 0.1 and first["stats"]["mode"] == "single"
    assert (store.prop_dir("P", ids[0]) / "mesh" / "mesh.glb").read_bytes() == b"glTF-generated"
    assert leftovers(store) == []


def test_a_random_seed_is_drawn_once_per_job(tmp_path: Path) -> None:
    """A seed below 0 asks for a random one: the hub draws it, and sends the same one again after a restart, so
    Pixal3D can resume what it had sampled for that prop (its fingerprint holds the seed)."""
    store, ids = project(tmp_path, b"", b"FAKE_DIE")
    store.patch_prop("P", ids[0], {"mesh": {"settings": {"seed": -1}}})

    async def main() -> int:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.06") as (pm, queue, adapter):
            job = await queue.submit("mesh", "P", ids)
            await until(queue, job["id"], lambda j: "working" in states(j))
            (sent,) = await asked(pm)                           # the first process, before it dies
            final, _ = await watch(queue, job["id"])
            assert states(final) == ["done", "failed"] and adapter.starts == 2
            (resent,) = await asked(pm)
            seed = resent["items"][0]["seed"]
            assert resent["items"][0]["id"] == ids[0] and 0 <= seed <= pixal3d.MAX_SEED
            assert sent["items"][0]["seed"] == seed
            return seed

    seed = asyncio.run(main())
    assert mesh_of(store, ids[0])["stats"]["seed"] == seed
    assert store.get_prop("P", ids[0])["mesh"]["settings"]["seed"] == -1      # the prop still asks for a random one
    # Drawn again for the next job.
    adapter = Pixal3DAdapter()
    prop = store.get_prop("P", ids[0])
    seeds = set()
    for _ in range(5):
        params = {"seed": -1}
        seeds.add(adapter._request(prop, params, store.prop_dir("P", ids[0])).body["seed"])
        assert params["seed"] in seeds                          # kept in the params both attempts share
    assert len(seeds) > 1 and all(0 <= s <= pixal3d.MAX_SEED for s in seeds)


def test_pixal3d_is_started_again_only_once_per_job(tmp_path: Path) -> None:
    store, ids = project(tmp_path, b"", b"", b"")

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.2", "--batch-die-after", "0.5") as (pm, queue, adapter):
            final, _ = await watch(queue, (await queue.submit("mesh", "P", ids))["id"])
            assert final["status"] == "error" and states(final) == ["failed"] * 3 and adapter.starts == 2
            assert final["progress"]["done"] == 3 and final["progress"]["fraction"] == 1.0
            assert set(final["errors"].values()) <= {BLAMED, NOT_REACHED}
            assert list(final["errors"].values()).count(BLAMED) <= 2       # at most the one on the card, each time
            assert NOT_REACHED in final["errors"].values()
            assert pm.status("pixal3d")["state"] in ("stopped", "error")

    asyncio.run(main())
    assert leftovers(store) == []
    assert all(mesh_of(store, prop_id)["glb"] is None for prop_id in ids)


# ---- one prop's trouble with its files ----------------------------------------------------------------

def test_a_deleted_prop_a_locked_file_and_a_missing_file_each_cost_one_prop(tmp_path: Path) -> None:
    store, ids = project(tmp_path, b"", b"", b"", b"FAKE_NO_GLB")
    fine, deleted, locked, missing = ids
    old = store.prop_dir("P", locked) / "mesh" / "mesh.glb"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"glTF-before")
    store.apply_stage_patch("P", locked, "mesh", {"glb": "mesh/mesh.glb"})

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.03") as (pm, queue, _):
            job = await queue.submit("mesh", "P", ids)
            await until(queue, job["id"], lambda j: j["props"][deleted]["state"] == "resting")
            store.delete_prop("P", deleted)
            with open(old, "rb"):                               # open in another program
                final, _ = await watch(queue, job["id"])
            assert final["status"] == "error"
            assert states(final) == ["done", "cancelled", "failed", "failed"]
            assert final["errors"] == {
                locked: "Could not save mesh/mesh.glb: it may be open in another program. Close it and try again.",
                missing: NO_GLB}
            assert final["progress"]["message"] == "2 of 4 props failed. Each prop shows what went wrong."
            assert pm.status("pixal3d")["state"] == "ready"

    asyncio.run(main())
    assert mesh_of(store, fine)["glb"] == "mesh/mesh.glb" and leftovers(store) == []
    assert old.read_bytes() == b"glTF-before"                   # the model it had is still there
    # The deleted prop: its folder went to deleted/ with its picture, and what Pixal3D wrote afterwards is gone.
    assert not store.prop_dir("P", deleted).exists()
    assert [p["id"] for p in store.get_project("P")["props"]] == [fine, locked, missing]
    (kept,) = (store.project_dir("P") / "deleted").iterdir()
    assert kept.name.startswith(deleted) and (kept / "images" / "upload_001.png").is_file()


def test_a_prop_deleted_while_pixal3d_makes_it_is_cancelled_at_once_and_its_folder_stays_gone(tmp_path: Path) -> None:
    """Pixal3D cannot drop one prop of its batch, but it writes nothing more for a prop whose folder is gone and
    fails it at its next step; the hub calls the prop cancelled with its next poll, long before that. A Pixal3D of
    before that check (FAKE_OLD_SAVE) makes the prop's folder again for its model: cleared away when the job ends."""
    store, ids = project(tmp_path, b"", b"", b"FAKE_OLD_SAVE")
    fine, gone, old = ids

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.1") as (pm, queue, _):
            job = await queue.submit("mesh", "P", ids)
            await until(queue, job["id"], lambda j: j["props"][old]["state"] == "resting")
            store.delete_prop("P", gone)
            store.delete_prop("P", old)
            now = await until(queue, job["id"], lambda j: j["props"][gone]["state"] == j["props"][old]["state"]
                              == "cancelled")
            assert now["status"] == "running" and now["props"][fine]["state"] != "done"      # long before the end
            final, _ = await watch(queue, job["id"])
            assert final["status"] == "done" and final["errors"] == {}
            assert states(final) == ["done", "cancelled", "cancelled"]
            told = (await pm.client.get(f"{pm.get('pixal3d').base_url}/spellcast/batch/{job['id']}")).json()
            assert told["items"][gone]["error"] == {"detail": "This prop was deleted while its 3D model was being "
                                                              "made.", "kind": "input"}
            assert told["items"][old]["state"] == "done"        # made all the same, into a folder made again

    asyncio.run(main())
    assert not store.prop_dir("P", gone).exists() and not store.prop_dir("P", old).exists()
    assert [p["id"] for p in store.get_project("P")["props"]] == [fine]
    assert mesh_of(store, fine)["glb"] == "mesh/mesh.glb" and leftovers(store) == []
    for kept in (store.project_dir("P") / "deleted").iterdir():                      # as they were deleted
        assert (kept / "images").is_dir() and not list(kept.rglob("*.glb"))


def test_a_prop_that_has_ended_keeps_what_it_is(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Saving a model takes a moment, in which Pixal3D can still report the prop: the model it saved must win.
    And whatever is said about a prop afterwards - Pixal3D dying right then blames the prop it was on - never
    turns a model that is on disk into a failure."""
    store, ids = project(tmp_path, b"", b"")

    class NoTool:           # _batch_context only asks the tool where it listens and whether it is alive
        base_url = "http://127.0.0.1:1"
        state = "ready"

        def alive(self) -> bool:
            return True

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store) as (_pm, queue, _):
            job = jobs_mod.Job(id="j_x", stage="mesh", tool="pixal3d", project="P", prop_ids=ids, params={},
                               status="running")
            job.props = {ids[0]: {"state": "working", "fraction": 0.9, "message": "Building the 3D mesh · step 9 of 12"},
                         ids[1]: {"state": "resting", "fraction": 0.3, "message": "Waiting for the next step"}}
            ctx = queue._batch_context(job, NoTool(), tmp_path, 1)

            save = store.apply_stage_patch

            def slowly(*args: Any) -> Any:
                time.sleep(0.3)
                return save(*args)

            monkeypatch.setattr(store, "apply_stage_patch", slowly)
            saving = asyncio.create_task(ctx.finish(ids[0], {"glb": "mesh/mesh.glb"}))
            await asyncio.sleep(0.1)
            ctx.update(ids[0], "working", 0.95, "Building the 3D mesh · step 11 of 12")
            await saving
            assert job.props[ids[0]] == {"state": "done", "fraction": 1.0, "message": ""}

            ctx.fail(ids[0], "Pixal3D stopped while working on this prop.")
            ctx.update(ids[0], "working", 0.5, "Painting the texture · step 1 of 4")
            assert job.props[ids[0]] == {"state": "done", "fraction": 1.0, "message": ""} and job.errors == {}
            ctx.fail(ids[1], "Pixal3D stopped while working on this prop.")
            assert job.props[ids[1]] == {"state": "failed", "fraction": 1.0, "message": ""}
            assert set(job.errors) == {ids[1]} and job.to_dict()["progress"]["done"] == 2

    asyncio.run(main())
    assert mesh_of(store, ids[0])["glb"] == "mesh/mesh.glb"


# ---- what the adapter does when Pixal3D does not answer -----------------------------------------------

class Told:
    """What run_batch told about each prop, with a BatchContext that listens and patches the props."""

    def __init__(self, store: ProjectStore, ids: list[str]):
        self.store, self.ids = store, ids
        self.states: dict[str, str] = {}
        self.errors: dict[str, str] = {}
        self.patches: dict[str, dict[str, Any]] = {}
        self.parts: list[tuple[Any, Any, Any]] = []
        self.cancel = False

    def context(self, pm: ProcessManager | None, job_id: str = "j_test", settings: dict[str, Any] | None = None,
                attempt: int = 1, alive: Callable[[], bool] = lambda: True) -> BatchContext:
        """pm None: no tool runs (nothing is asked over HTTP)."""
        async def finish(prop_id: str, patch: dict[str, Any]) -> None:
            await asyncio.to_thread(self.store.apply_stage_patch, "P", prop_id, "mesh", patch)
            self.states[prop_id], self.patches[prop_id] = "done", patch

        def fail(prop_id: str, message: str) -> None:
            self.states[prop_id], self.errors[prop_id] = "failed", message

        return BatchContext(project="P", stage="mesh",
                            base_url=pm.get("pixal3d").base_url if pm else "http://127.0.0.1:1",
                            client=pm.client if pm else None,  # type: ignore[arg-type]
                            settings=settings or {}, job_id=job_id, work_dir=self.store.work_dir("P", f"mesh-{job_id}"),
                            attempt=attempt, cancelled=lambda: self.cancel, tool_alive=alive,
                            update=lambda prop_id, state, fraction, message: self.states.__setitem__(prop_id, state),
                            finish=finish, fail=fail,
                            set_part=lambda part, current, loading: self.parts.append((part, current, loading)))

    def items(self, settings: dict[str, Any] | None = None) -> list[BatchItem]:
        """The props as the job runner hands them over, with their own mesh settings (or `settings`)."""
        props = {p["id"]: p for p in self.store.get_project("P")["props"]}
        return [BatchItem(prop_id, props[prop_id], dict(settings or props[prop_id]["mesh"]["settings"]),
                          self.store.prop_dir("P", prop_id)) for prop_id in self.ids]


def test_the_half_written_model_of_a_deleted_prop_does_not_stay_in_the_deleted_folder(tmp_path: Path) -> None:
    """A prop deleted in the moment between Pixal3D writing its model and the hub saving it: the prop's folder
    moves to deleted/ with the half-written file in it, where it is of no use to anybody."""
    store, ids = project(tmp_path, b"")

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store) as (pm, _, adapter):
            await pm.ensure_running("pixal3d")
            told = Told(store, ids)
            (item,) = told.items()
            request = adapter._request(item.prop, item.params, item.prop_dir)
            request.partial.write_bytes(b"glTF-half")
            store.delete_prop("P", ids[0])
            await adapter._complete(told.context(pm), item, request, {})
            assert told.states == {ids[0]: "cancelled"} and told.errors == {}

    asyncio.run(main())
    (kept,) = (store.project_dir("P") / "deleted").iterdir()
    assert (kept / "images").is_dir() and not list(kept.rglob("*.partial.glb"))


def test_without_a_last_answer_pixal3ds_own_note_says_which_prop_to_blame(tmp_path: Path) -> None:
    """When Pixal3D stopped and its last answer may be old, the note it keeps in the work folder (ON_CARD) says
    which prop was on the card, and which one it made last: that one is saved instead of being made again."""
    store, ids = project(tmp_path, b"", b"", b"")
    a, b, c = ids
    adapter = Pixal3DAdapter()
    told = Told(store, ids)
    ctx = told.context(None)
    items = {item.prop_id: item for item in told.items()}
    requests = {prop_id: adapter._request(item.prop, item.params, item.prop_dir) for prop_id, item in items.items()}
    note = ctx.work_dir / pixal3d.ON_CARD

    def write(current: str | None, made: str | None = None, **result: Any) -> None:
        finished = None if made is None else {"id": made, "result": {
            "glb_path": requests[made].body["out_path"], "seconds": 12.5, "seed": 7, "mode": "single", "views": 1,
            "warning": None, "ss_retries": 0, "resolution": 1024, "slow": [], "resumed": False, **result}}
        note.write_text(json.dumps({"batch_id": "j_test", "current": current, "finished": finished}), encoding="utf-8")

    async def blame(open_ids: set[str], current: str | None, fresh: bool = False) -> str | None:
        return (await adapter._stopped(ctx, items, requests, open_ids, current, "it stopped", fresh)).blame

    async def main() -> None:
        # No note (a Pixal3D that died before its batch began, or an older one): the last answer's prop ...
        assert await blame(set(ids), a) == a
        note.write_text("{half a", encoding="utf-8")
        assert await blame(set(ids), a) == a
        # ... unless it has ended for the hub: a prop whose model was saved is never blamed.
        assert await blame({b, c}, a) is None
        # The note wins over the last answer, also when nothing was on the card (between two steps).
        write(b)
        assert await blame(set(ids), a) == b and await blame({a, c}, a) is None
        write(None)
        assert await blame(set(ids), a) is None
        # An answer that itself said the card broke under a prop is fresh: the note is not needed.
        assert await blame(set(ids), a, fresh=True) == a
        # The model of the note's last prop is not there: nothing to save, and it stays to be made.
        write(b, made=a)
        open_ids = set(ids)
        assert await blame(open_ids, a) == b and open_ids == set(ids) and told.states == {}
        # The note of another request for the same prop (the job's first attempt): not this one's model.
        requests[a].partial.write_bytes(b"glTF-made")
        write(b, made=a, glb_path=str(tmp_path / "elsewhere.partial.glb"))
        assert await blame(open_ids, a) == b and open_ids == set(ids) and told.states == {}
        # This request's model is there: saved now, with what the note says of it.
        write(b, made=a)
        assert await blame(open_ids, a) == b and open_ids == {b, c} and told.states == {a: "done"}
        assert told.patches[a]["stats"]["seconds"] == 12.5 and told.patches[a]["stats"]["seed"] == 7
        assert (items[a].prop_dir / "mesh" / "mesh.glb").read_bytes() == b"glTF-made"

    asyncio.run(main())


def test_polls_without_an_answer_do_not_restart_a_living_pixal3d(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """While a mesh is saved Pixal3D's native code can keep Python busy for many seconds: no poll is answered."""
    store, ids = project(tmp_path, b"", b"")
    monkeypatch.setattr(pixal3d, "POLL_TIMEOUT", httpx.Timeout(0.2, connect=3.0))

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.02", "--batch-mute", "2.0") as (pm, queue, adapter):
            started = time.monotonic()
            final, _ = await watch(queue, (await queue.submit("mesh", "P", ids))["id"])
            assert time.monotonic() - started > 1.8
            assert final["status"] == "done" and states(final) == ["done", "done"]
            assert adapter.starts == 1 and len(await asked(pm)) == 1

    asyncio.run(main())
    assert leftovers(store) == []


def test_a_pixal3d_that_cannot_be_reached_twice_in_a_row_has_to_start_afresh(tmp_path: Path) -> None:
    """Its process is there, nobody listens: said after two polls, not after the 45 minutes of the watchdog."""
    store, ids = project(tmp_path, b"", b"")

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.8", "--batch-deaf-after", "0.3") as (pm, _, adapter):
            await pm.ensure_running("pixal3d")
            told = Told(store, ids)
            started = time.monotonic()
            with pytest.raises(ToolNeedsRestart) as stop:
                await adapter.run_batch(told.context(pm), told.items())
            assert time.monotonic() - started < 20 and pm.get("pixal3d").alive()
            assert stop.value.blame == ids[0]                   # the prop on the card, by Pixal3D's own note
            assert told.errors == {} and "done" not in told.states.values()
            assert not list(store.project_dir("P").rglob("*.partial.glb"))

    asyncio.run(main())


def test_a_batch_that_stands_still_or_ignores_a_cancel_has_to_start_afresh(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    store, ids = project(tmp_path, b"", b"FAKE_STALL")

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.01") as (pm, _, adapter):
            await pm.ensure_running("pixal3d")
            # Nothing moves for STALL_TIMEOUT: Pixal3D is asked to stop, and has to start afresh.
            monkeypatch.setattr(pixal3d, "STALL_TIMEOUT", 0.6)
            told = Told(store, ids)
            with pytest.raises(ToolNeedsRestart, match="no progress") as stop:
                await adapter.run_batch(told.context(pm, "j_one"), told.items())
            assert stop.value.blame == ids[1] and told.states == {ids[0]: "resting", ids[1]: "working"}
            assert (await asked(pm, "cancels"))["received"] == 1

        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.01") as (pm, _, adapter):
            await pm.ensure_running("pixal3d")
            # A cancel that Pixal3D took and does not act on: CANCEL_STALL without any change is the limit.
            monkeypatch.setattr(pixal3d, "STALL_TIMEOUT", 600.0)
            monkeypatch.setattr(pixal3d, "CANCEL_STALL", 0.6)
            told = Told(store, ids)
            ctx = told.context(pm, "j_two")
            run = asyncio.create_task(adapter.run_batch(ctx, told.items()))
            while told.states.get(ids[1]) != "working":
                await asyncio.sleep(0.01)
            await asyncio.sleep(1.0)
            assert not run.done()                               # standing still alone is fine for 45 minutes
            told.cancel = True
            asked_at = time.monotonic()
            with pytest.raises(ToolNeedsRestart, match="did not stop after the cancel"):
                await run
            assert 0.5 < time.monotonic() - asked_at < 10
            assert (await asked(pm, "cancels"))["received"] == 1    # it said it took it: not asked again

    asyncio.run(main())


def test_what_pixal3d_answers_to_the_request_itself(tmp_path: Path) -> None:
    """Busy with another batch (one the hub gave up on): it has to start afresh. A request it refuses fails
    every prop with its words. A batch it does not know (it started afresh by itself): start afresh."""
    store, ids = project(tmp_path, b"", b"")

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.3") as (pm, _, adapter):
            await pm.ensure_running("pixal3d")
            told = Told(store, ids)
            other = store.work_dir("P", "mesh-j_other")
            (other / "p.png").write_bytes(PNG)
            running = {"batch_id": "j_other", "work_dir": str(other), "items": [
                {"id": "p_000001", "image_path": str(other / "p.png"), "out_path": str(other / "x.glb"), "seed": 1,
                 "resolution": 1024, "decimation_target": 1, "texture_size": 1, "views": None, "views_dir": None}]}
            assert (await pm.client.post(f"{pm.get('pixal3d').base_url}/spellcast/batch", json=running)).status_code == 202
            with pytest.raises(ToolNeedsRestart, match="another job") as stop:
                await adapter.run_batch(told.context(pm), told.items())
            assert stop.value.blame is None and told.states == {} and told.errors == {}
            assert not list(store.project_dir("P").rglob("*.partial.glb"))

            # No batch of that name: the 404 of a Pixal3D that was started afresh behind the hub's back.
            requests = {prop_id: adapter._request(item.prop, item.params, item.prop_dir)
                        for prop_id, item in zip(ids, told.items())}
            with pytest.raises(ToolNeedsRestart, match="does not know this job"):
                await adapter._follow(told.context(pm, "j_unknown"), dict(zip(ids, told.items())), requests)

        async with pixal3d_hub(tmp_path, store) as (pm, _, adapter):
            await pm.ensure_running("pixal3d")
            told = Told(store, ids)
            ctx = told.context(pm)
            ctx.work_dir = store.project_dir("P") / "props"         # no scratch folder of a job: refused
            await adapter.run_batch(ctx, told.items())
            assert told.states == {ids[0]: "failed", ids[1]: "failed"}
            assert set(told.errors.values()) == {f"This is no work folder of a job: {ctx.work_dir}."}
            assert not list(store.project_dir("P").rglob("*.partial.glb"))
            # A process manager that says the tool is gone is believed at once.
            told = Told(store, ids)
            with pytest.raises(ToolNeedsRestart, match="Pixal3D stopped"):
                await adapter.run_batch(told.context(pm, alive=lambda: False), told.items())

    asyncio.run(main())
