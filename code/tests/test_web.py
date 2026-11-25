"""The web UI's rules (static/js/stages/rules.js): which work taking the props down to a stage queues, when
a result is fresh, and the prop names the Images workbench suggests and what a 3D model is made from (its
mesh.source), which must be the hub's own. Also the jobs the page's own tests are fed (tests/web/hub_job.mjs):
they say where their props stand as the hub's jobs do.

They are tested with Node.js, a developer tool (Spellcast3D's users never need it): without it these
tests are skipped.

Run with the hub venv:  <venv>\\Scripts\\python.exe -m pytest code\\tests -q
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from spellcast.jobs import Job
from spellcast.projects import (RETOPO_KEEP_VERTICES, box_name_from_prompt, effective_views, front_only, mesh_fresh,
                                mesh_source, retopo_fresh, switch_retopo_method, thumb_crop, views_signature)

WEB_TESTS = Path(__file__).resolve().parent / "web"
HUB_JOB = WEB_TESTS / "hub_job.mjs"
RULES = Path(__file__).resolve().parents[1] / "spellcast" / "static" / "js" / "stages" / "rules.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


def _node(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([NODE, *args], capture_output=True, text=True, encoding="utf-8", timeout=120)


def test_stage_rules() -> None:
    run = _node("--test", *(str(p) for p in sorted(WEB_TESTS.glob("*.test.mjs"))))
    assert run.returncode == 0, run.stdout + run.stderr


def _job(status: str, done: int = 0, **more: Any) -> Job:
    """A job of three props that are made one after another (any tool but the 3D generator), as far as `done`."""
    return Job(id="j1", stage="retopo", tool="lato2", project="Props", prop_ids=["p_1", "p_2", "p_3"], params={},
               status=status, done=done, **more)


_NO_START = "LATO.2 could not start."
# Every way such a job can stand. cut_short: the props a cancel stopped half-way (counted in `done`).
ONE_BY_ONE_JOBS = [
    _job("queued"),
    _job("starting_tool", message="Starting LATO.2…"),
    _job("running", current_prop="p_1", message="Working on Barrel…"),
    _job("running", 1, current_prop="p_2", fraction=1.4 / 3, prop_fraction=0.4, message="Generating the mesh… 40%"),
    _job("running", 2, current_prop="p_3", fraction=2 / 3, errors={"p_1": "It failed."}),
    _job("running", 2, current_prop="p_2", fraction=2 / 3, cut_short=1, cancel_requested=True,
         message="Cancelling… (the current prop may take a moment to stop)"),
    _job("done", 3, fraction=1.0, message="Done."),
    _job("error", 3, fraction=1.0, errors={"p_2": "It failed."}),
    _job("error", errors={"p_1": _NO_START, "p_2": _NO_START, "p_3": _NO_START}, message=_NO_START),
    _job("error", 1, fraction=1 / 3, message="Something went wrong inside Spellcast3D."),
    _job("cancelled", 2, fraction=2 / 3, cut_short=1),
    _job("cancelled", 1, fraction=1 / 3),
    _job("cancelled"),
]


def test_fixture_jobs_are_the_hubs() -> None:
    """The page reads where a job's props stand from job.props alone, and its tests feed it jobs that hub_job.mjs
    completes with `props` and `part`: for a job whose props run one after another, exactly what the hub derives
    (jobs.py Job.prop_states), in every state such a job can be in."""
    sent = []
    for job in ONE_BY_ONE_JOBS:
        told = job.to_dict()
        sent.append([{k: v for k, v in told.items() if k not in ("props", "part")}, job.cut_short])
    script = (f"import {{ hubJob }} from {json.dumps(HUB_JOB.as_uri())};\n"
              f"console.log(JSON.stringify({json.dumps(sent)}.map(([job, cutShort]) => {{\n"
              f"  const {{ props, part }} = hubJob(job, {{ cutShort }});\n  return {{ props, part }};\n}})));")
    run = _node("--input-type=module", "-e", script)
    assert run.returncode == 0, run.stderr
    page = json.loads(run.stdout)
    seen: set[str] = set()
    for job, fixture in zip(ONE_BY_ONE_JOBS, page):
        hub = job.to_dict()
        where = f"{job.status}, done {job.done}"
        assert fixture["part"] is None and hub["part"] is None, where
        assert list(fixture["props"]) == list(hub["props"]) == job.prop_ids, where
        for prop_id, entry in hub["props"].items():
            mine = fixture["props"][prop_id]
            assert (mine["state"], mine["message"]) == (entry["state"], entry["message"]), f"{where}, {prop_id}"
            assert mine["fraction"] == pytest.approx(entry["fraction"], abs=1e-3), f"{where}, {prop_id}"
            seen.add(entry["state"])
    assert seen == {"waiting", "working", "done", "failed", "cancelled"}      # a prop of such a job never rests
    working = ONE_BY_ONE_JOBS[3].to_dict()["props"]["p_2"]
    assert (working["fraction"], working["message"]) == (0.4, "Generating the mesh… 40%")


PROMPTS = [
    "a wooden barrel with iron bands, game asset",
    "An ornate treasure chest (gold trim)",
    "the old brass lantern standing on a table",
    "Sword; low poly",
    "with love",
    "a a a",
    "  ",
    "Ölfass aus Eichenholz mit Eisenbändern und Rost",
    "big red mushroom house wearing a hat",
    "masterpiece, best quality, (wooden barrel:1.3), <lora:props_v2:0.8> iron bands",
    "8k, 3D render, [stone well]; mossy",
    "highly detailed | white background",
]


def test_suggested_names_are_the_hubs() -> None:
    """The draft's name field suggests the name the hub gives it when it becomes a prop without one."""
    script = (f"import {{ nameFromPrompt }} from {json.dumps(RULES.as_uri())};\n"
              f"console.log(JSON.stringify({json.dumps(PROMPTS)}.map(nameFromPrompt)));")
    run = _node("--input-type=module", "-e", script)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == [box_name_from_prompt(p) for p in PROMPTS]


FRONT = {"view": "front", "src": "images/0003.png", "rect": [0, 0, 512, 512]}
SHEET = [{"view": "right", "src": "images/upload_002.jpg", "rect": None},
         {"view": "back", "src": "images/0003.png", "rect": [1024, 16, 500, 498]},
         {"view": "left", "src": "images/Ölfass (2).png", "rect": [512, 0, 512, 512]}, FRONT]
NO_FRONT = [{"view": "left", "src": "images/0003.png", "rect": None},
            {"view": "back", "src": "images/0004.png", "rect": None}]
IMAGES = [
    {"chosen": "images/0001.png", "views": []},
    {"chosen": "images/0001.png"},
    {"chosen": None, "views": []},
    {},
    # The views count while the chosen picture is one of theirs, else the chosen picture alone.
    {"chosen": "images/0001.png", "views": [FRONT]},
    {"chosen": "images/0003.png", "views": [FRONT]},
    {"chosen": "images/0001.png", "views": SHEET},
    {"chosen": "images/0003.png", "views": SHEET},
    {"chosen": "images/Ölfass (2).png", "views": SHEET},
    {"chosen": "images/upload_002.jpg", "views": SHEET},
    {"chosen": None, "views": SHEET},
    {"views": SHEET},
    {"chosen": "images/0001.png", "views": NO_FRONT},
    {"chosen": "images/0004.png", "views": NO_FRONT},
]


def test_mesh_sources_are_the_hubs() -> None:
    """The page's meshFresh compares mesh.source, which the hub stores, with meshSource of the prop's pictures
    now: both must say the same for the same pictures and views, and use the same views (effective_views)."""
    script = (f"import {{ effectiveViews, meshSource, thumbCrop, viewsSignature }} from {json.dumps(RULES.as_uri())};\n"
              f"const images = {json.dumps(IMAGES)};\n"
              f"console.log(JSON.stringify([images.map(meshSource), images.map((i) => viewsSignature(i.views)),"
              f" images.map(effectiveViews), images.map(thumbCrop)]));")
    run = _node("--input-type=module", "-e", script)
    assert run.returncode == 0, run.stderr
    sources, signatures, effective, crops = json.loads(run.stdout)
    assert sources == [mesh_source(i) for i in IMAGES]
    assert signatures == [views_signature(i.get("views") or []) for i in IMAGES]
    assert effective == [effective_views(i) for i in IMAGES]
    assert sum(1 for views in effective if views) == 5         # both answers are compared
    assert crops == [thumb_crop(i) for i in IMAGES]              # the part of the chosen picture a thumbnail shows
    assert any(crops) and not all(crops)


_A, _B = "images/upload_009.png", "images/0004.png"
# A sheet cut into "4 quadrants", and the same sheet cut "4 in a row" (slicer.js presetViews).
QUAD = [{"view": "front", "src": _A, "rect": [0, 0, 512, 512]}, {"view": "left", "src": _A, "rect": [0, 512, 512, 512]},
        {"view": "back", "src": _A, "rect": [512, 0, 512, 512]}, {"view": "right", "src": _A, "rect": [512, 512, 512, 512]}]
ROW = [{"view": view, "src": _A, "rect": [i * 256, 0, 256, 1024]} for i, view in enumerate(("front", "left", "back", "right"))]
PAIR = [{"view": "front", "src": _A, "rect": None}, {"view": "back", "src": _B, "rect": None}]


def _model(made_from: dict[str, Any], now: dict[str, Any], **mesh: Any) -> dict[str, Any]:
    """A prop whose 3D model was made while its pictures were `made_from` (the hub kept mesh_source of them, as
    adapters/pixal3d.py does), and whose pictures are `now`."""
    return {"id": "p_000001", "images": {"candidates": [_A, _B], "settings": {"prompt": "a bear"}, **now},
            "mesh": {"glb": "mesh/mesh.glb", "source": mesh_source(made_from), "imported": None,
                     "settings": {"seed": 1, "resolution": 1024}, **mesh}}


_SHEET = {"chosen": _A, "views": QUAD}
# (the prop, its 3D model is fresh, ... and with Pixal3D's multi-view models there)
MESH_BOXES: list[tuple[dict[str, Any], bool, bool]] = [
    (_model(_SHEET, _SHEET), True, True),                                           # the same picture chosen again
    (_model(_SHEET, {"chosen": _B, "views": QUAD}), False, False),                  # another picture: made alone
    (_model(_SHEET, {"chosen": _A, "views": QUAD[::-1]}), True, True),              # the views in another order
    (_model(_SHEET, {"chosen": _A, "views": [{**v, "rect": [float(n) for n in v["rect"]]} for v in QUAD]}), True, True),
    (_model(_SHEET, {**_SHEET, "settings": {"prompt": "a fox", "use_references": False, "from_sketch": True},
                     "references": ["reference/reference_001.png"], "sketch": "sketch/sketch.png"}), True, True),
    (_model(_SHEET, _SHEET, settings={"seed": 7, "resolution": 1536, "texture_size": 4096}), True, True),
    (_model(_SHEET, {"chosen": _A, "views": [{**v, "rect": [520, 0, 504, 512]} if v["view"] == "back" else v
                                             for v in QUAD]}), False, False),       # a slice moved
    (_model(_SHEET, {"chosen": _A, "views": ROW}), False, False),                   # cut another way
    (_model(_SHEET, {"chosen": _A, "views": []}), False, False),                    # the views cleared
    (_model({"chosen": _A}, {"chosen": _A, "views": []}), True, True),              # one picture alone
    (_model({"chosen": _A}, {"chosen": _B}), False, False),
    (_model({"chosen": _A}, {"chosen": _A, "views": [{"view": "front", "src": _B, "rect": None}]}), True, True),
    (_model({"chosen": _A}, {"chosen": _A, "views": QUAD[:1]}), False, False),      # a part of it marked since
    (_model({"chosen": _A, "views": PAIR}, {"chosen": _B, "views": PAIR}), True, True),   # any picture of the views
    # Made from the front view alone: out of date once the multi-view models are there.
    (_model({"chosen": _A, "views": PAIR}, {"chosen": _A, "views": PAIR}, stats={"mode": "single", "views": 1}), True, False),
    (_model({"chosen": _A, "views": PAIR}, {"chosen": _A, "views": PAIR}, stats={"mode": "multi", "views": 2}), True, True),
    (_model({"chosen": _A, "views": PAIR[:1]}, {"chosen": _A, "views": PAIR[:1]}, stats={"mode": "single"}), True, True),
    (_model(_SHEET, {"chosen": _B}, imported={"materials": []}), True, True),       # the artist's own model
    (_model(_SHEET, {"chosen": _B}, source=None), True, True),                      # made before "source" existed
    (_model(_SHEET, {"chosen": None, "views": QUAD}), True, True),                  # no picture to make it from
    (_model(_SHEET, _SHEET, glb=None), False, False),                               # no model
    ({"id": "p_000002", "images": {"chosen": _A}}, False, False),
    ({"id": "p_000003", "mesh": {"glb": "mesh/imported_001.glb", "imported": {"materials": []}}}, True, True),
]


def test_mesh_freshness_is_the_hubs() -> None:
    """A 3D model is fresh while it was made from the picture, or the views, the prop has now (rules.js meshFresh
    == projects.mesh_fresh, frontOnly == front_only): nothing else of the prop counts, so going back to Images
    and choosing the same picture, saving the same views or switching its tab makes nothing "to do" again."""
    boxes = [box for box, _fresh, _with_models in MESH_BOXES]
    script = (f"import {{ frontOnly, meshFresh }} from {json.dumps(RULES.as_uri())};\n"
              f"const boxes = {json.dumps(boxes)};\n"
              f"console.log(JSON.stringify([boxes.map((b) => meshFresh(b)), boxes.map((b) => meshFresh(b, "
              f"{{ multiview: true }})), boxes.map((b) => !!b.mesh && frontOnly(b))]));")
    run = _node("--input-type=module", "-e", script)
    assert run.returncode == 0, run.stderr
    fresh, with_models, front = json.loads(run.stdout)
    assert fresh == [mesh_fresh(box) for box in boxes] == [is_fresh for _box, is_fresh, _with in MESH_BOXES]
    assert with_models == [mesh_fresh(box, multiview=True) for box in boxes] == [w for _box, _fresh, w in MESH_BOXES]
    assert front == [front_only(box) for box in boxes] and sum(front) == 1



def _box(obj: Any = "retopo/lato2.obj", made_by: Any = "lato2", picked: Any = "lato2", **retopo: Any) -> dict[str, Any]:
    """A prop as the hub sends it, with a clean mesh made by `made_by` while the prop is set to `picked`."""
    settings = {} if picked is None else {"method": picked}
    block = {"obj": obj, "source": "mesh/mesh.glb", "stale": False, "settings": settings, **retopo}
    if made_by is not None:
        block["method"] = made_by
    return {"id": "p_000001", "mesh": {"glb": "mesh/mesh.glb"}, "retopo": block}


RETOPO_BOXES = [
    _box(),                                                 # made by the method it is set to
    _box(made_by="triflow", obj="retopo/triflow.obj", picked="triflow"),
    _box(made_by="imesh", obj="retopo/retopo.obj", picked="imesh"),
    _box(picked="triflow"),                                 # another method picked: to do again
    _box(picked="imesh"),
    _box(made_by="imesh", obj="retopo/retopo.obj", picked="lato2"),
    _box(made_by=None, obj="retopo/retopo.obj", picked="imesh"),     # older projects: Instant Meshes made it
    _box(made_by=None, obj="retopo/retopo.obj", picked="lato2"),
    _box(made_by="imesh", obj="retopo/retopo.obj", picked=None),     # no method picked: Instant Meshes
    _box(made_by="lato2", picked=None),
    _box(made_by="", obj="retopo/retopo.obj", picked=""),
    _box(obj=None),                                         # no clean mesh
    _box(obj=""),
    _box(stale=True),                                       # made from an older 3D model
    _box(source="mesh/imported_002.glb"),
    _box(source=None),                                      # made before "source" existed
    {"id": "p_000002", "mesh": {}, "retopo": {"obj": "retopo/lato2.obj", "method": "lato2", "source": "mesh/mesh.glb",
                                              "settings": {"method": "lato2"}}},
    {"id": "p_000003", "retopo": {"obj": "retopo/retopo.obj", "settings": {"method": "imesh"}}},
    {"id": "p_000004", "mesh": {"glb": "mesh/mesh.glb"}, "retopo": {"settings": {"method": "imesh"}}},
]


def test_a_tileable_texture_is_the_hubs() -> None:
    """A prop is a tileable texture while its Tileable tick is on, for a model that draws such pictures (rules.js
    tileable == projects.tileable_prop): the page passes such a prop by, and the hub refuses its 3D model, by one rule."""
    from spellcast.projects import tileable_prop

    boxes = [{"images": {"settings": settings}} for settings in (
        {"tileable": True}, {"tileable": True, "qwen_model": "turbo"}, {"tileable": True, "qwen_model": "full"},
        {"tileable": True, "qwen_model": "flux3"}, {"tileable": False}, {"tileable": "yes"}, {})] + [{"images": {}}, {}]
    script = (f"import {{ tileable }} from {json.dumps(RULES.as_uri())};\n"
              f"console.log(JSON.stringify({json.dumps(boxes)}.map((b) => tileable(b))));")
    run = _node("--input-type=module", "-e", script)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == [tileable_prop(box) for box in boxes] == [True, True, True] + [False] * 6


def test_retopo_freshness_is_the_hubs() -> None:
    """A clean mesh is fresh while it was made from the prop's 3D model by the method the prop is set to
    (rules.js retopoFresh == projects.retopo_fresh): picking another method makes the prop "to do" again."""
    script = (f"import {{ retopoFresh }} from {json.dumps(RULES.as_uri())};\n"
              f"console.log(JSON.stringify({json.dumps(RETOPO_BOXES)}.map((b) => retopoFresh(b))));")
    run = _node("--input-type=module", "-e", script)
    assert run.returncode == 0, run.stderr
    page = json.loads(run.stdout)
    assert page == [retopo_fresh(box) for box in RETOPO_BOXES]
    assert page[:3] == [True, True, True] and page[3:6] == [False, False, False]
    assert any(page[6:]) and not all(page[6:])


def _made(method: str, vertices: Any = 2000, *more_sizes: int, **more: Any) -> dict[str, Any]:
    """Clean meshes as prop.retopo (or retopo.kept) has them: one of `vertices` (the one used), and one of 1,000
    vertices more per size of `more_sizes`."""
    file = {"triflow": "triflow.obj", "lato2": "lato2.obj", "imesh": "retopo.obj"}[method]
    stats = {"vertices": vertices, "faces": 4000}
    variants = [{"size": 2, "obj": f"retopo/2k/{file}", "stats": stats},
                *({"size": size, "obj": f"retopo/{size}k/{file}", "stats": {"vertices": 1000 * size}} for size in more_sizes)]
    return {"obj": f"retopo/2k/{file}", "method": method, "source": "mesh/mesh.glb", "stale": False, "accepted": False,
            "saved_from": "auto", "stats": stats, "variants": variants, **more}


_NONE = {"obj": None, "method": None, "source": None, "stale": False, "accepted": False, "variants": []}
_REST = {"session_id": "s1", "settings": {"method": "triflow", "sizes": [2], "lato2_seed": 3}}
_OLD = {"obj": "retopo/retopo.obj", "stats": {"vertices": 800},     # older: Instant Meshes', its one variant
        "variants": [{"size": 1, "obj": "retopo/retopo.obj", "stats": {"vertices": 800}}]}
# (prop.retopo, the method picked next)
RETOPO_SWITCHES: list[tuple[dict[str, Any], str]] = [
    ({**_REST, **_made("triflow")}, "lato2"),                                   # set aside, none yet for LATO.2
    ({**_REST, **_made("triflow")}, "triflow"),                                 # the same method: nothing
    ({**_REST, **_made("triflow", accepted=True), "kept": {"lato2": _made("lato2", 1900)}}, "lato2"),     # back
    ({**_REST, **_made("lato2"), "kept": {"triflow": _made("triflow"), "imesh": _made("imesh", 5000)}}, "imesh"),
    ({**_REST, **_NONE, "kept": {"triflow": _made("triflow", stale=True)}}, "triflow"),
    ({**_REST, **_NONE, "kept": {"triflow": _made("triflow")}}, "lato2"),       # none, and none kept for it
    ({**_REST, **_NONE}, "imesh"),
    ({**_REST, **_made("triflow", 50_000)}, "lato2"),                           # too big: they stay
    ({**_REST, **_made("triflow", 49_999, 4, 22)}, "lato2"),                    # every variant under the limit
    ({**_REST, **_made("triflow", 50_000), "kept": {"lato2": _made("lato2")}}, "lato2"),     # too big: dropped
    ({**_REST, **_made("triflow", None)}, "imesh"),                             # no count: not kept
    ({**_REST, **_made("triflow", True)}, "imesh"),
    ({**_REST, **_OLD}, "triflow"),
    ({**_REST, **_OLD}, "imesh"),
    ({**_REST, **_made("lato2"), "kept": {"triflow": {"obj": None}}}, "triflow"),
    ({**_REST, **_made("lato2", bake_stale=False), "kept": {"imesh": _made("imesh", bake_stale=True)}}, "imesh"),
    ({**_REST, **_made("triflow", 2000, 60)}, "lato2"),                         # one variant too big: they stay
]


def test_switching_the_retopo_method_is_the_hubs() -> None:
    """Each method keeps its clean meshes while the prop is set to another, under the same limit on both sides
    (rules.js retopoSwitched == projects.switch_retopo_method, RETOPO_KEEP_VERTICES)."""
    script = (f"import {{ RETOPO_KEEP_VERTICES, retopoSwitched }} from {json.dumps(RULES.as_uri())};\n"
              f"console.log(JSON.stringify([RETOPO_KEEP_VERTICES, {json.dumps(RETOPO_SWITCHES)}"
              f".map(([retopo, method]) => retopoSwitched(retopo, method))]));")
    run = _node("--input-type=module", "-e", script)
    assert run.returncode == 0, run.stderr
    limit, page = json.loads(run.stdout)
    assert limit == RETOPO_KEEP_VERTICES
    hub = []
    for retopo, method in RETOPO_SWITCHES:
        block = copy.deepcopy(retopo)
        assert switch_retopo_method(block, method) == (block != retopo)
        hub.append(block)
    assert page == hub
    assert [block != retopo for block, (retopo, _method) in zip(hub, RETOPO_SWITCHES)] == [
        True, False, True, True, True, False, False, False, True, True, False, False, True, False, True, True, False]
    assert hub[2]["obj"] == "retopo/2k/lato2.obj" and hub[2]["kept"]["triflow"]["accepted"] is True
    assert hub[9]["obj"] == "retopo/2k/lato2.obj" and "kept" not in hub[9]
    assert [v["size"] for v in hub[8]["kept"]["triflow"]["variants"]] == [2, 4, 22]


# (the prop, the sizes it asks for, the sizes it has a clean mesh of)
_SIZED = [
    ({**_REST, **_made("triflow", 2000, 4)}, [2, 4]),
    ({**_REST, **_made("triflow", 2000, 4)}, [4, 22, 6]),
    ({**_REST, **_made("triflow", 2000, 4)}, []),
    ({**_REST, **_made("triflow", 2000, 4, stale=True)}, [2, 4]),             # made from an older 3D model
    ({**_REST, **_made("lato2", 2000, 4)}, [2, 4]),                           # made with another method
    ({**_REST, **_NONE}, [9]),
    ({**_REST, **_made("triflow", 2000, 0.25)}, [0.25, 2, 0.3]),              # under a thousand among whole ones
]
MISSING_BOXES = [{"id": "p_000001", "mesh": {"glb": "mesh/mesh.glb"},
                  "retopo": {**retopo, "settings": {**retopo["settings"], "sizes": sizes}}} for retopo, sizes in _SIZED]


def test_the_sizes_a_prop_lacks_are_the_hubs() -> None:
    """The sizes a prop has no up-to-date clean mesh of (rules.js retopoMissing == projects.retopo_missing_sizes):
    what "Retopo All" counts and what a job then makes."""
    from spellcast.projects import retopo_missing_sizes

    script = (f"import {{ retopoMissing }} from {json.dumps(RULES.as_uri())};\n"
              f"console.log(JSON.stringify({json.dumps(MISSING_BOXES)}.map((b) => retopoMissing(b))));")
    run = _node("--input-type=module", "-e", script)
    assert run.returncode == 0, run.stderr
    page = json.loads(run.stdout)
    assert page == [retopo_missing_sizes(box) for box in MISSING_BOXES]
    assert page == [[], [22, 6], [], [2, 4], [2, 4], [9], [0.3]]


# 2.01 and 4.07: a thousand times them is no whole number in floating point (2009.9999999999998, 4070.0000000000005).
RETOPO_SIZES_SEEN = [0.1, 0.15, 0.2, 0.24, 0.25, 0.3, 0.74, 1, 2, 2.01, 4.07, 5.2, 9.99, 10, 10.01, 12.25, 22, 98.99,
                     99]


def test_the_sizes_and_the_vertices_each_method_is_asked_for_are_the_hubs() -> None:
    """What a size may be, how it is written, and the vertices each method is asked for to make it - its own count,
    or the nearest one the method places (rules.js RETOPO_SIZES, retopoSizeName, RETOPO_VERTICES and retopoVertices
    == projects.RETOPO_MAX_SIZES / RETOPO_MIN_SIZE / RETOPO_MAX_SIZE / RETOPO_SIZE_DECIMALS, retopo_size_name and the
    adapters' faces_for and vertices_for)."""
    from spellcast import projects
    from spellcast.adapters import imesh, lato2, triflow

    script = (f"import {{ RETOPO_SIZES, RETOPO_VERTICES, retopoSizeName, retopoVertices }} from "
              f"{json.dumps(RULES.as_uri())};\nconst sizes = {json.dumps(RETOPO_SIZES_SEEN)};\n"
              "const asked = (m) => [m, sizes.map((s) => retopoVertices(m, s))];\n"
              "console.log(JSON.stringify([RETOPO_SIZES, RETOPO_VERTICES, sizes.map(retopoSizeName),\n"
              "  Object.fromEntries(Object.keys(RETOPO_VERTICES).map(asked))]));")
    run = _node("--input-type=module", "-e", script)
    assert run.returncode == 0, run.stderr
    limits, bounds, names, asked = json.loads(run.stdout)
    assert limits == {"count": projects.RETOPO_MAX_SIZES, "min": projects.RETOPO_MIN_SIZE,
                      "max": projects.RETOPO_MAX_SIZE, "decimals": projects.RETOPO_SIZE_DECIMALS}
    assert names == [projects.retopo_size_name(size) for size in RETOPO_SIZES_SEEN]
    assert names[:2] == ["0.1k", "0.15k"] and names[-3:] == ["22k", "98.99k", "99k"]
    per_vertex = triflow.FACES_PER_VERTEX
    assert bounds == {"triflow": [triflow.MIN_FACES // per_vertex, triflow.MAX_FACES // per_vertex],
                      "lato2": [lato2.MIN_VERTICES, lato2.MAX_VERTICES],
                      "imesh": [imesh.MIN_TARGET_VERTICES, imesh.MAX_TARGET_VERTICES]}
    hub = {"triflow": [triflow.faces_for(size) // per_vertex for size in RETOPO_SIZES_SEEN],
           "lato2": [lato2.vertices_for(size) for size in RETOPO_SIZES_SEEN],
           "imesh": [imesh.vertices_for(size) for size in RETOPO_SIZES_SEEN]}
    assert asked == hub
    assert asked["triflow"][:6] == [250, 250, 250, 250, 250, 300] and asked["lato2"][:3] == [200, 200, 200]
    assert asked["lato2"][-4:] == [10_000] * 4 and asked["imesh"] == [round(1000 * size) for size in RETOPO_SIZES_SEEN]


def test_the_shape_a_prop_makes_is_the_hubs() -> None:
    """The 3D stage's Shape (rules.js GEOMETRIES / meshGeometry == projects.GEOMETRIES / mesh_geometry): the same
    choices, Hi3D first, and Pixal3D's for settings that name none (a prop saved before the choice) or none there is.
    The page's size of Hi3D's download is the installer's."""
    from spellcast.install import models
    from spellcast.projects import GEOMETRIES, mesh_geometry

    settings = [{}, {"geometry": "hi3d"}, {"geometry": "pixal3d"}, {"geometry": "trellis"}, {"geometry": None}]
    script = (f"import {{ GEOMETRIES, meshGeometry }} from {json.dumps(RULES.as_uri())};\n"
              f"console.log(JSON.stringify([GEOMETRIES, {json.dumps(settings)}"
              f".map((s) => meshGeometry({{ mesh: {{ settings: s }} }}))]));")
    run = _node("--input-type=module", "-e", script)
    assert run.returncode == 0, run.stderr
    choices, page = json.loads(run.stdout)
    assert choices == list(GEOMETRIES)
    assert page == [mesh_geometry(s) for s in settings] == ["pixal3d", "hi3d", "pixal3d", "pixal3d", "pixal3d"]
    mesh_js = (RULES.parent / "mesh.js").read_text(encoding="utf-8")
    assert f"const HI3D_BYTES = {models.HI3D_BYTES:_};" in mesh_js


_ST = "omnitexture/ot_002/omnitexture.glb"
def _painted(mesh: str | None, chosen: bool, glb: str | None = _ST, source: str | None = "mesh/mesh.glb",
             stale: bool = False, own: bool = False, by_hand: bool = False, repaints: int = 0) -> dict[str, Any]:
    """A prop as the Omni-Texture rules see it: `glb` a job's Omni-Texture (it has a painted view), or with `own`
    the 3D model's own texture as the artist painted it (`by_hand`: hand paint saved; `repaints`: how many)."""
    return {"mesh": {"glb": mesh}, "omnitexture": {
        "settings": {"chosen": chosen}, "glb": glb, "source_mesh": source, "stale": stale, "painted": by_hand,
        "views": ["omnitexture/ot_001/views/0-front/answer.png"] if glb and not own else [],
        "repaints": [{"glb": f"omnitexture/ot_00{n + 2}/omnitexture.glb"} for n in range(repaints)]}}


OMNITEXTURE_BOXES: list[dict[str, Any]] = [
    _painted("mesh/mesh.glb", True), _painted("mesh/mesh.glb", False), _painted("mesh/mesh.glb", True, stale=True),
    _painted("mesh/imported_002.glb", True), _painted("mesh/mesh.glb", True, source=None),
    _painted("mesh/mesh.glb", True, glb=None, source=None), _painted(None, False, glb=None, source=None),
    # The prop's own texture as a result: used once something is painted or painted again on it, chosen or not.
    _painted("mesh/mesh.glb", False, own=True), _painted("mesh/mesh.glb", True, own=True),
    _painted("mesh/mesh.glb", False, own=True, by_hand=True), _painted("mesh/mesh.glb", False, own=True, repaints=1),
    _painted("mesh/mesh.glb", True, own=True, repaints=2), _painted("mesh/mesh.glb", False, own=True, by_hand=True, stale=True),
]


def test_the_omnitexture_rules_are_the_hubs() -> None:
    """Whether an Omni-Texture is fresh, whether the result is the prop's own texture, whether it is used and which
    model the bake reads (rules.js omnitextureFresh / omniOwn / omnitextureUsed / highPoly ==
    projects.omnitexture_fresh / omnitexture_own / omnitexture_used / high_poly): a chosen prop's fresh
    Omni-Texture replaces its 3D model there, and so does a prop's own texture once the artist painted on it."""
    from spellcast.projects import high_poly, omnitexture_fresh, omnitexture_own, omnitexture_used

    script = (f"import {{ highPoly, omniOwn, omnitextureFresh, omnitextureUsed }} from {json.dumps(RULES.as_uri())};\n"
              f"console.log(JSON.stringify({json.dumps(OMNITEXTURE_BOXES)}"
              f".map((b) => [omnitextureFresh(b), omniOwn(b), omnitextureUsed(b), highPoly(b)])));")
    run = _node("--input-type=module", "-e", script)
    assert run.returncode == 0, run.stderr
    page = [tuple(row) for row in json.loads(run.stdout)]
    assert page == [(omnitexture_fresh(b), omnitexture_own(b), omnitexture_used(b), high_poly(b))
                    for b in OMNITEXTURE_BOXES]
    assert [high == _ST for _fresh, _own, _used, high in page] == [True, False, False, False, True, False, False,
                                                                 False, False, True, True, True, False]


def test_the_page_knows_the_calibration_step_by_the_installer_words() -> None:
    """shell/calibration.js shows its notice while the installer's step text begins with progress.CALIBRATING_STEP: the
    two spellings must be one."""
    from spellcast.install import progress

    js = (Path(__file__).resolve().parents[1] / "spellcast" / "static" / "js" / "shell" / "calibration.js").read_text(encoding="utf-8")
    assert f"export const CALIBRATING_STEP = '{progress.CALIBRATING_STEP}';" in js


def test_the_folder_grouping_rules_are_the_same_on_the_page_and_in_the_hub() -> None:
    """static/js/stages/propfiles.js (a drop of many pictures on the chooser) and spellcast/propfiles.py (the hub's
    "Import folder") group files into props by the same rules."""
    from spellcast import propfiles

    names = ["knight_front.png", "dragon_front.png", "knight_back.png", "dragon-left.png", "Front_Ork.png", "ork back.jpg",
             "barrel.png", "knight_right.png", "knight_front (2).png", "Knight-Front.png", "knight front.jpg",
             "frontier_post.png", "props/knight/front.png", "props/knight/back.png", "front.png", "KNIGHT_left.png",
             "knight.png", "old_barrel-01.webp", "Left-old_barrel.webp", "art/dragon.right.png"]
    script = (
        "import { groupPropFiles } from 'file:///" + str(WEB_TESTS.parents[1] / "spellcast" / "static" / "js" / "stages" / "propfiles.js").replace("\\", "/") + "';\n"
        "const names = " + json.dumps(names) + ";\n"
        "const groups = groupPropFiles(names.map((n) => ({ name: n.split('/').pop(), webkitRelativePath: n })));\n"
        "console.log(JSON.stringify(groups.map((g) => [g.name, g.files.map((f) => [f.file.webkitRelativePath, f.side])])));"
    )
    run = _node("--input-type=module", "-e", script)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout) == [[name, [[path, side] for path, side in files]]
                                      for name, files in propfiles.group_prop_files(names)]
