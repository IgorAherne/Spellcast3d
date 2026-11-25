"""A prop seen from several sides: its views (projects.py images.views), several pictures imported at once,
what the 3D stage sends Pixal3D for 0, 1 and 2+ views (adapters/pixal3d.py, against fake_tool.py), the
order of a job's props, and Pixal3D's optional multi-view models (multiview.py and launchers/mv_download.py).
How a mesh job follows its props through Pixal3D is in test_mesh_batch.py.

Run with the hub venv:  <venv>\\Scripts\\python.exe -m pytest code\\tests -q
Nothing here downloads anything: the download runs a stand-in script, and the models folder is a temp one.
"""

from __future__ import annotations

import ast
import asyncio
import json
import msvcrt
import os
import re
import sys
import time
import types
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from spellcast import multiview, paths
from spellcast.adapters import pixal3d
from spellcast.adapters.base import ToolAdapter
from spellcast.adapters.images_common import results_patch
from spellcast.adapters.pixal3d import (FRONT_ONLY, MADE_SLOWLY, MV_MISSING,
                                        NO_FRONT, Pixal3DAdapter)
from spellcast.gpu import GpuInfo
from spellcast.procman import ProcessManager, ToolError
from spellcast.projects import (MAX_IMAGE_UPLOADS, VIEW_ORDER, ProjectError, ProjectStore, check_views,
                                effective_views, mesh_fresh, mesh_source, retopo_fresh, thumb_crop, thumb_signature,
                                views_signature)
from spellcast.server import create_app
from test_core import JPG, PNG, FakeAdapter, make_settings
from test_mesh_batch import Pixal3DOnFake, Told, asked, pixal3d_hub, watch

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Spellcast3D is Windows-only")

sys.path.insert(0, str(paths.LAUNCHERS))
import mv_download  # noqa: E402  (launchers/: runs in the Pixal3D venv, stdlib only at import)

PIXAL3D_INSTALL = paths.PIXAL3D_DIR / "install.py"


@pytest.fixture(autouse=True)
def roomy_drive(monkeypatch: pytest.MonkeyPatch) -> None:
    """The download's free-space check sees plenty of room (the real drive may be fuller), and Xet is on."""
    monkeypatch.setattr(multiview, "_free_space", lambda folder: 10 ** 14)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)


def _store_with_prop(tmp_path: Path, *pictures: bytes) -> tuple[ProjectStore, str]:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    pid = store.add_prop("P", "Crate")["id"]
    for data in pictures:
        store.add_image("P", pid, data)
    return store, pid


# ---- views ----------------------------------------------------------------------------------------

def test_views_are_checked_and_stored_in_view_order(tmp_path: Path) -> None:
    store, pid = _store_with_prop(tmp_path, PNG, JPG)
    a, b = "images/upload_001.png", "images/upload_002.jpg"
    prop = store.set_views("P", pid, [{"view": "back", "src": b, "rect": None},
                                      {"view": "front", "src": a, "rect": [0, 4, 512, 600]},
                                      {"view": "left", "src": a, "rect": [512, 4, 16, 16], "extra": 1}])
    assert prop["images"]["views"] == [{"view": "front", "src": a, "rect": [0, 4, 512, 600]},
                                       {"view": "left", "src": a, "rect": [512, 4, 16, 16]},
                                       {"view": "back", "src": b, "rect": None}]
    assert store.get_prop("P", pid)["images"]["views"] == prop["images"]["views"]
    assert prop["images"]["chosen"] == b                                   # views leave the chosen picture be
    # A set without a front view can be saved: the artist is still marking it.
    assert store.set_views("P", pid, [{"view": "right", "src": b, "rect": None}])["images"]["views"] == [
        {"view": "right", "src": b, "rect": None}]
    assert store.set_views("P", pid, [])["images"]["views"] == []

    bad = [(None, '"views" must be a list'), ({"view": "front"}, '"views" must be a list'),
           ([{"view": "top", "src": a, "rect": None}], "which side"),
           (["front"], "which side"),
           ([{"view": "front", "src": "images/other.png", "rect": None}], "one of this prop's own pictures"),
           ([{"view": "front", "src": "../../x.png", "rect": None}], "one of this prop's own pictures"),
           ([{"view": "front", "src": a, "rect": [0, 0, 100]}], "four whole numbers"),
           ([{"view": "front", "src": a, "rect": [0, 0, 100.5, 100]}], "four whole numbers"),
           ([{"view": "front", "src": a, "rect": [True, 0, 100, 100]}], "four whole numbers"),
           ([{"view": "front", "src": a, "rect": "0,0,100,100"}], "four whole numbers"),
           ([{"view": "front", "src": a, "rect": [-1, 0, 100, 100]}], "too small or starts outside"),
           ([{"view": "front", "src": a, "rect": [0, 0, 15, 100]}], "too small or starts outside"),
           ([{"view": "front", "src": a, "rect": None}, {"view": "front", "src": b, "rect": None}], "Only one"),
           ]
    for views, message in bad:
        with pytest.raises(ProjectError, match=message) as e:
            store.set_views("P", pid, views)
        assert e.value.status_code == 400, views
    assert store.get_prop("P", pid)["images"]["views"] == []             # nothing was stored
    with pytest.raises(ProjectError) as e:
        store.set_views("P", "p_ffffff", [])
    assert e.value.status_code == 404


def test_views_stay_with_their_prop_and_old_projects_get_none(tmp_path: Path) -> None:
    store, pid = _store_with_prop(tmp_path, PNG, PNG)
    store.set_views("P", pid, [{"view": "front", "src": "images/upload_001.png", "rect": None},
                               {"view": "left", "src": "images/upload_002.png", "rect": None}])
    assert store.add_prop("P", "Next")["images"]["views"] == []           # an output, never inherited
    assert store.ensure_draft("P")["images"]["views"] == []

    # project.json from before views, and a hand-edited one: only the valid views remain.
    file = store.project_dir("P") / "project.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    for prop in data["props"]:
        prop["images"].pop("views")
    data["props"][0]["images"]["views"] = [
        {"view": "left", "src": "images/upload_002.png", "rect": [1, 2, 30, 40]},
        {"view": "left", "src": "images/upload_001.png", "rect": None},           # a second left side
        {"view": "front", "src": "images/gone.png", "rect": None},                # not a picture of the prop
        {"view": "back", "src": "images/upload_001.png", "rect": [0, 0, 2, 2]},   # too small
        "front", {"view": "right", "src": "images/upload_001.png"}]
    data["props"][1]["images"]["views"] = "all"
    file.write_text(json.dumps(data), encoding="utf-8")
    props = store.get_project("P")["props"]
    assert props[0]["images"]["views"] == [{"view": "left", "src": "images/upload_002.png", "rect": [1, 2, 30, 40]},
                                           {"view": "right", "src": "images/upload_001.png", "rect": None}]
    assert all(p["images"]["views"] == [] for p in props[1:])


def test_views_signature_and_mesh_source() -> None:
    views = [{"view": "right", "src": "images/a.png", "rect": None},
             {"view": "front", "src": "images/0003.png", "rect": [0, 0, 512, 512]},
             {"view": "back", "src": "images/0003.png", "rect": [1024, 16, 500, 498]}]
    assert views_signature(views) == ("front:images/0003.png:0,0,512,512|back:images/0003.png:1024,16,500,498|"
                                      "right:images/a.png:*")
    assert views_signature([]) == ""
    assert mesh_source({"chosen": "images/0001.png", "views": []}) == "images/0001.png"
    assert mesh_source({"chosen": "images/0001.png"}) == "images/0001.png"
    assert mesh_source({"chosen": None, "views": []}) is None
    assert mesh_source({"chosen": "images/0003.png", "views": views}) == "views:" + views_signature(views)
    assert mesh_source({"chosen": "images/a.png", "views": views}) == "views:" + views_signature(views)
    # The signature changes with any change of a view.
    moved = [dict(views[1], rect=[0, 0, 512, 511]), views[0], views[2]]
    assert mesh_source({"chosen": "images/a.png", "views": moved}) != mesh_source({"chosen": "images/a.png",
                                                                                  "views": views})
    assert check_views(views, {"images/a.png", "images/0003.png"})[0]["view"] == "front"


def test_the_views_count_while_the_chosen_picture_is_one_of_theirs() -> None:
    """Another chosen picture (generated or imported later) is made into 3D alone; the views are kept for
    when the artist picks one of theirs again."""
    views = [{"view": "front", "src": "images/0003.png", "rect": [0, 0, 512, 512]},
             {"view": "left", "src": "images/upload_002.png", "rect": None}]
    assert effective_views({"chosen": "images/0003.png", "views": views}) == views
    assert effective_views({"chosen": "images/upload_002.png", "views": views}) == views     # any of theirs
    assert effective_views({"chosen": "images/0004.png", "views": views}) == []
    assert effective_views({"chosen": None, "views": views}) == []
    assert effective_views({"views": views}) == [] and effective_views({}) == []
    assert effective_views({"chosen": "images/0003.png", "views": [None, "front", *views]}) == views
    assert mesh_source({"chosen": "images/0004.png", "views": views}) == "images/0004.png"
    assert mesh_source({"chosen": None, "views": views}) is None


def test_a_3d_model_stays_fresh_until_what_it_is_made_from_changes(tmp_path: Path) -> None:
    """Going back to Images and looking around makes no 3D model "to do" again (mesh_fresh): only another chosen
    picture or other views do. Whichever it is, nothing but a new model replaces mesh.glb or mesh.source, or
    makes the clean mesh and the bake made from it old: with the first picture (or views) back, the model is
    fresh again."""
    store, pid = _store_with_prop(tmp_path, PNG, PNG, JPG)
    a, b = "images/upload_001.png", "images/upload_002.png"
    quad = [{"view": "front", "src": a, "rect": [0, 0, 512, 512]}, {"view": "left", "src": a, "rect": [0, 512, 512, 512]},
            {"view": "back", "src": a, "rect": [512, 0, 512, 512]}, {"view": "right", "src": a, "rect": [512, 512, 512, 512]}]
    row = [{"view": view, "src": a, "rect": [i * 256, 0, 256, 1024]} for i, view in enumerate(VIEW_ORDER)]
    store.select_image("P", pid, a)
    made = mesh_source(store.set_views("P", pid, quad)["images"])     # what adapters/pixal3d.py keeps as mesh.source
    assert made == ("views:front:images/upload_001.png:0,0,512,512|left:images/upload_001.png:0,512,512,512|"
                    "back:images/upload_001.png:512,0,512,512|right:images/upload_001.png:512,512,512,512")
    assert mesh_source({"chosen": a, "views": [dict(v, rect=[float(n) for n in v["rect"]]) for v in quad]}) == made
    store.apply_stage_patch("P", pid, "mesh", {"glb": "mesh/mesh.glb", "source": made, "stats": {"mode": "multi"}})
    store.apply_stage_patch("P", pid, "retopo", {"obj": "retopo/triflow.obj", "method": "triflow",
                                                 "source": "mesh/mesh.glb"})
    store.apply_stage_patch("P", pid, "projection", {"glb": "projection/bake_001/projection.glb"})

    def fresh(prop: dict[str, Any]) -> bool:
        assert prop["mesh"]["glb"] == "mesh/mesh.glb" and prop["mesh"]["source"] == made    # never dropped
        assert retopo_fresh(prop) and prop["projection"]["glb"] and not prop["projection"]["stale"]
        return mesh_fresh(prop)

    assert fresh(store.select_image("P", pid, a))                        # the same picture chosen again
    assert not fresh(store.select_image("P", pid, b))                    # another one: it is made into 3D alone
    assert fresh(store.select_image("P", pid, a))                        # ... and the first one again
    assert fresh(store.set_views("P", pid, quad))                        # the same views saved again
    assert fresh(store.set_views("P", pid, quad[::-1]))                  # ... in another order
    assert fresh(store.set_views("P", pid, [dict(v, rect=list(v["rect"]), extra=1) for v in quad]))
    # The tabs of Images are settings; neither they, nor the words, nor the 3D settings are what a model is made from.
    assert fresh(store.patch_prop("P", pid, {"images": {"settings": {"use_references": False, "from_sketch": True,
                                                                     "prompt": "a fox"}}}))
    assert fresh(store.patch_prop("P", pid, {"images": {"settings": {"use_references": True, "from_sketch": False}}}))
    assert fresh(store.patch_prop("P", pid, {"mesh": {"settings": {"seed": 7, "resolution": 1536}}}))
    assert fresh(store.set_thumb("P", pid, "images", PNG, of=thumb_signature(store.get_prop("P", pid)["images"])))
    assert fresh(store.set_thumb("P", pid, "mesh", PNG))
    assert fresh(ProjectStore(tmp_path / "projects").get_project("P")["props"][0])      # the project opened again

    # An imported picture is the chosen one; a removed one takes its views along, and Ctrl+Z brings both back.
    assert not fresh(store.add_image("P", pid, PNG))
    assert fresh(store.select_image("P", pid, a))
    gone = store.remove_image("P", pid, a)
    assert gone["images"]["chosen"] == b and gone["images"]["views"] == [] and not fresh(gone)
    assert fresh(store.restore_image("P", pid, a, index=0, views=quad, choose=True))
    # A finished generation: its first picture is the chosen one. One that made none changes nothing, and says
    # nothing about the chosen picture (the job read the prop before: one imported meanwhile stays chosen).
    images = store.get_prop("P", pid)["images"]
    assert "chosen" not in results_patch({"candidates": [], "chosen": None}, [], [])
    assert fresh(store.apply_stage_patch("P", pid, "images", results_patch(images, [], [])))
    assert not fresh(store.apply_stage_patch("P", pid, "images", results_patch(images, ["images/0001.png"], [5])))
    assert fresh(store.select_image("P", pid, a))

    # What the model is made from changed: a slice moved, two sides swapped, the sheet cut another way, no views.
    moved = [dict(v, rect=[520, 0, 504, 512]) if v["view"] == "back" else v for v in quad]
    swapped = [dict(v, view={"front": "back", "back": "front"}.get(v["view"], v["view"])) for v in quad]
    for views in (moved, swapped, row, []):
        assert not fresh(store.set_views("P", pid, views)), views
    assert fresh(store.set_views("P", pid, quad))                        # as it was made: the model it has is that one


# ---- several pictures at once ------------------------------------------------------------------------

def test_pictures_imported_together_become_the_views(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    draft = store.ensure_draft("P")
    pid = draft["id"]
    store.set_thumb("P", pid, "images", PNG)
    prop = store.add_images("P", pid, [(PNG, "red_dragon-front.png"), (JPG, "side.jpg"), (PNG, "back.png"),
                                       (PNG, "other side.png"), (PNG, "extra.png")])
    images = prop["images"]
    assert images["candidates"] == [f"images/upload_{i:03d}.{ext}"
                                    for i, ext in ((1, "png"), (2, "jpg"), (3, "png"), (4, "png"), (5, "png"))]
    assert images["chosen"] == "images/upload_001.png"
    assert images["views"] == [{"view": "front", "src": "images/upload_001.png", "rect": None},
                               {"view": "left", "src": "images/upload_002.jpg", "rect": None},
                               {"view": "back", "src": "images/upload_003.png", "rect": None},
                               {"view": "right", "src": "images/upload_004.png", "rect": None}]
    assert prop["draft"] is False and prop["name"] == "Red dragon front"        # named after the first file
    assert "images" not in prop["thumbs"]
    assert [p["id"] for p in store.get_project("P")["props"]] == [pid]          # packed once

    two = store.add_images("P", pid, [(PNG, "a.png"), (PNG, "b.png")])["images"]
    assert two["chosen"] == "images/upload_006.png" and len(two["candidates"]) == 7
    assert two["views"] == [{"view": "front", "src": "images/upload_006.png", "rect": None},
                            {"view": "left", "src": "images/upload_007.png", "rect": None}]
    one = store.add_image("P", pid, JPG)["images"]                  # one picture: the views stay as they are
    assert one["chosen"] == "images/upload_008.jpg" and one["views"] == two["views"]
    assert mesh_source(one) == "images/upload_008.jpg"              # ... but it is made into 3D alone

    # One bad file refuses them all, before anything is stored.
    with pytest.raises(ProjectError, match="^notes.txt: This file is not a PNG") as e:
        store.add_images("P", pid, [(PNG, "a.png"), (b"hello", "notes.txt")])
    assert e.value.status_code == 415
    assert len(list((store.prop_dir("P", pid) / "images").iterdir())) == 8
    with pytest.raises(ProjectError, match="Choose a picture"):
        store.add_images("P", pid, [])
    with pytest.raises(ProjectError, match="too many pictures") as e:
        store.add_images("P", pid, [(PNG, "x.png")] * (MAX_IMAGE_UPLOADS + 1))
    assert e.value.status_code == 413


def test_pictures_imported_together_take_the_sides_the_page_read(tmp_path: Path) -> None:
    """The page reads the sides from the file names (slicer.js importSides) and sends them: a front and a back
    are the front and the back, not a front and a side; the front one is chosen."""
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    pid = store.add_prop("P", "Bear")["id"]
    images = store.add_images("P", pid, [(PNG, "bear_back.png"), (PNG, "bear_front.png"), (PNG, "notes.png")],
                              ["back", "front", None])["images"]
    assert images["candidates"] == ["images/upload_001.png", "images/upload_002.png", "images/upload_003.png"]
    assert images["chosen"] == "images/upload_002.png"                   # the front, not the first file
    assert images["views"] == [{"view": "front", "src": "images/upload_002.png", "rect": None},
                               {"view": "back", "src": "images/upload_001.png", "rect": None}]
    # No front among them: the first one is chosen (the page asks for a front).
    left = store.add_images("P", pid, [(PNG, "l.png"), (PNG, "r.png")], ["left", "right"])["images"]
    assert left["chosen"] == "images/upload_004.png" and [v["view"] for v in left["views"]] == ["left", "right"]
    # Sides for no picture at all leave the views as they were.
    kept = store.add_images("P", pid, [(PNG, "x.png"), (PNG, "y.png")], [None, None])["images"]
    assert kept["views"] == left["views"] and kept["chosen"] == "images/upload_006.png"
    for bad in (["front"], ["front", "sideways"], ["front", "front"], "front,back"):
        with pytest.raises(ProjectError, match="Reload the page"):
            store.add_images("P", pid, [(PNG, "a.png"), (PNG, "b.png")], bad)
    assert len(store.get_prop("P", pid)["images"]["candidates"]) == 7    # a refused upload stores nothing


def test_a_sliced_sheets_thumbnail_is_its_front_and_pictures_imported_meanwhile_stay(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    pid = store.add_prop("P", "Ogre")["id"]
    sheet = store.add_image("P", pid, PNG)["images"]["chosen"]
    store.set_thumb("P", pid, "images", PNG, of=sheet)
    front = {"view": "front", "src": sheet, "rect": [0, 0, 64, 64]}
    back = {"view": "back", "src": sheet, "rect": [64, 0, 64, 64]}
    assert thumb_crop({"chosen": sheet, "views": [front, back]}) == [0, 0, 64, 64]
    assert thumb_crop({"chosen": "images/other.png", "views": [front, back]}) is None     # not the chosen one's
    assert thumb_crop({"chosen": sheet, "views": [back]}) is None                          # no front slice
    # The thumbnail shows the Front slice: it is made again when that slice moves, not for other views.
    assert "images" not in store.set_views("P", pid, [front, back])["thumbs"]      # made for all of it
    sliced = f"{sheet}#0,0,64,64"       # what it shows: thumb_signature
    assert store.set_thumb("P", pid, "images", PNG, of=sliced)["thumbs"]["images"]["crop"] == [0, 0, 64, 64]
    assert "images" in store.set_views("P", pid, [front])["thumbs"]
    assert "images" not in store.set_views("P", pid, [{**front, "rect": [8, 0, 64, 64]}])["thumbs"]
    # A thumbnail of a slice the hub does not have (not yet, or not any more) is not kept: the page saves its
    # views before it draws one (thumbs.js).
    assert "images" not in store.set_thumb("P", pid, "images", PNG, of=sliced)["thumbs"]
    assert "images" not in store.set_views("P", pid, [front, back])["thumbs"]
    assert "images" in store.set_thumb("P", pid, "images", PNG, of=sliced)["thumbs"]

    # A generation's result (made from the prop as it was when it started) keeps the pictures imported while
    # it ran, their views and the picture chosen meanwhile.
    started = store.get_prop("P", pid)["images"]
    store.add_images("P", pid, [(PNG, "front.png"), (PNG, "back.png")], ["front", "back"])
    imported = store.get_prop("P", pid)["images"]
    patch = {"candidates": [*started["candidates"], "images/0001.png"], "last_seeds": [1]}
    after = store.apply_stage_patch("P", pid, "images", patch)["images"]
    assert after["candidates"] == [*started["candidates"], "images/0001.png", *imported["candidates"][1:]]
    assert after["views"] == imported["views"] and after["chosen"] == imported["chosen"]


def test_new_props_make_one_variation(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    first = store.add_prop("P", "One")
    assert first["images"]["settings"]["batch_count"] == 1
    store.patch_prop("P", first["id"], {"images": {"settings": {"batch_count": 4}}})
    assert store.add_prop("P", "Two")["images"]["settings"]["batch_count"] == 4   # the last prop's, as before


# ---- the 3D stage -------------------------------------------------------------------------------------

@pytest.fixture
def quick_polls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pixal3d, "POLL_INTERVAL", 0.02)


async def _make(adapter: Pixal3DOnFake, pm: ProcessManager, store: ProjectStore, pid: str,
                settings: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Run the prop through Pixal3DAdapter.run_batch, alone -> (what Pixal3D was sent for it, or None when
    nothing was sent; its patch; its error)."""
    before = len(await asked(pm))
    told = Told(store, [pid])
    await adapter.run_batch(told.context(pm), told.items(settings))
    requests = await asked(pm)
    sent = requests[-1]["items"][0] if len(requests) > before else None
    return sent, told.patches.get(pid), told.errors.get(pid)


def test_what_pixal3d_is_sent_for_0_1_and_more_views(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     quick_polls: None) -> None:
    store, pid = _store_with_prop(tmp_path, PNG, PNG, PNG)
    prop_dir = store.prop_dir("P", pid).resolve()
    a, b, c = "images/upload_001.png", "images/upload_002.png", "images/upload_003.png"
    available = {"value": True}
    monkeypatch.setattr(multiview, "available", lambda *args: available["value"])
    settings = {"seed": 7, "resolution": 1024, "decimation_target": 200_000, "texture_size": 1024}

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store) as (pm, _, adapter):
            await pm.ensure_running("pixal3d")

            async def run(views: list[dict[str, Any]] | None) -> tuple[dict[str, Any], dict[str, Any]]:
                if views is not None:
                    store.set_views("P", pid, views)
                sent, patch, error = await _make(adapter, pm, store, pid, settings)
                assert sent is not None and patch is not None and error is None, error
                return sent, patch

            # No views: the chosen picture. A prop's entry in the batch is one generate request, with its id.
            sent, patch = await run(None)
            assert set(sent) == {"id", "image_path", "out_path", "seed", "resolution", "decimation_target",
                                 "texture_size", "views", "views_dir", "geometry"}
            assert sent["geometry"] == "pixal3d"        # settings that name none: Pixal3D's own shape
            assert sent["id"] == pid and sent["image_path"] == str(prop_dir / c)
            assert sent["views"] is None and sent["views_dir"] is None
            assert (sent["seed"], sent["resolution"], sent["decimation_target"], sent["texture_size"]) == \
                (7, 1024, 200_000, 1024)
            # Pixal3D writes beside mesh.glb, which stays until the new model is complete.
            partial = Path(sent["out_path"])
            assert partial.parent == prop_dir / "mesh" and partial.match("mesh.*.partial.glb")
            assert patch == {"glb": "mesh/mesh.glb", "source": c, "geometry": "pixal3d",
                             "stats": {"seconds": 0.1, "seed": 7, "views": 1, "mode": "single", "warning": None,
                                       "ss_retries": 0}}
            assert (prop_dir / "mesh" / "mesh.glb").read_bytes() == b"glTF-generated"
            assert not list((prop_dir / "mesh").glob("*.partial.glb"))
            request = (await asked(pm))[-1]
            assert request["wave_size"] == 0 and request["resume"] is False and request["batch_id"] == "j_test"

            # One view: that part of its picture alone (any side), while that picture is the chosen one.
            store.select_image("P", pid, a)
            one = [{"view": "left", "src": a, "rect": [10, 20, 300, 400]}]
            sent, patch = await run(one)
            assert sent["image_path"] == str(prop_dir / a)
            assert sent["views"] == [{"image_path": str(prop_dir / a), "rect": [10, 20, 300, 400], "azimuth": 90}]
            assert sent["views_dir"] == str(prop_dir / "mesh" / "views")
            assert patch["source"] == "views:left:images/upload_001.png:10,20,300,400"
            assert patch["stats"]["mode"] == "single" and patch["stats"]["views"] == 1

            # Several: every one with its camera, the front first.
            views = [{"view": "right", "src": b, "rect": None}, {"view": "back", "src": a, "rect": [0, 0, 64, 64]},
                     {"view": "front", "src": a, "rect": [64, 0, 64, 64]}, {"view": "left", "src": c, "rect": None}]
            sent, patch = await run(views)
            assert sent["image_path"] == str(prop_dir / a)
            assert [(v["image_path"], v["rect"], v["azimuth"]) for v in sent["views"]] == [
                (str(prop_dir / a), [64, 0, 64, 64], 0), (str(prop_dir / c), None, 90),
                (str(prop_dir / a), [0, 0, 64, 64], 180), (str(prop_dir / b), None, 270)]
            assert patch["source"] == "views:" + views_signature(views)
            assert patch["stats"] == {"seconds": 0.1, "seed": 7, "views": 4, "mode": "multi", "warning": None,
                                      "ss_retries": 0}

            # Several views are made at the quality asked, like one picture (High, 1536, by default).
            settings["resolution"] = 1536
            assert (await run(None))[0]["resolution"] == 1536
            store.select_image("P", pid, c)            # another view's picture: the views still count
            assert (await run(None))[0]["resolution"] == 1536

            # Another chosen picture, not one of the views': made into 3D alone, the views are kept.
            store.add_image("P", pid, PNG)
            d = "images/upload_004.png"
            sent, patch = await run(None)
            assert sent["image_path"] == str(prop_dir / d) and sent["views"] is None and sent["resolution"] == 1536
            assert patch["source"] == d and patch["stats"]["mode"] == "single"
            assert len(store.get_prop("P", pid)["images"]["views"]) == 4
            store.select_image("P", pid, a)
            settings["resolution"] = 1024

            # Without the multi-view models: the front view alone, and the artist is told.
            available["value"] = False
            sent, patch = await run(None)
            assert sent["views"] == [{"image_path": str(prop_dir / a), "rect": [64, 0, 64, 64], "azimuth": 0}]
            assert patch["source"] == "views:" + views_signature(views)
            assert patch["stats"]["warning"] == FRONT_ONLY and patch["stats"]["mode"] == "single"
            assert patch["stats"]["views"] == 1

            # The quality asked is sent whatever is sent: the front view alone, or every view.
            settings["resolution"] = 1536
            sent, patch = await run(None)
            assert len(sent["views"]) == 1 and sent["resolution"] == 1536
            assert patch["stats"]["warning"] == FRONT_ONLY
            available["value"] = True
            sent, patch = await run(None)
            assert len(sent["views"]) == 4 and sent["resolution"] == 1536
            assert patch["stats"]["warning"] is None
            available["value"], settings["resolution"] = False, 1024

            # Two or more views need a front one: nothing is sent.
            store.set_views("P", pid, [{"view": "left", "src": a, "rect": None}, {"view": "back", "src": b, "rect": None}])
            assert await _make(adapter, pm, store, pid, settings) == (None, None, NO_FRONT)

            # A view's picture that is gone.
            store.set_views("P", pid, [{"view": "front", "src": b, "rect": None}])
            store.select_image("P", pid, b)
            (prop_dir / b).unlink()
            sent, patch, error = await _make(adapter, pm, store, pid, settings)
            assert sent is None and patch is None
            assert re.search(r"picture of the Front view \(images/upload_002.png\) is missing", error or "")
            assert not list((prop_dir / "mesh").glob("*.partial.glb"))

    asyncio.run(main())


def test_pixal3d_without_the_multiview_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              quick_polls: None) -> None:
    """The hub saw the models and the running Pixal3D does not (its health says so): the front view alone,
    like without them. When Pixal3D finds them missing only as it makes the prop, the prop fails."""
    store, pid = _store_with_prop(tmp_path)
    store.add_images("P", pid, [(PNG + b"FAKE_RETRY FAKE_SPILL", "front.png"), (PNG, "side.png")])
    monkeypatch.setattr(multiview, "available", lambda *args: True)

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--no-multiview") as (pm, _, adapter):
            await pm.ensure_running("pixal3d")
            health = (await pm.client.get(f"{pm.get('pixal3d').base_url}/spellcast/health")).json()
            assert health == {"ready": True, "low_vram": True, "multiview": False, "batch": True}
            sent, patch, error = await _make(adapter, pm, store, pid)
            assert error is None and len(sent["views"]) == 1 and sent["views"][0]["azimuth"] == 0
            # Everything the artist should know about the result, in its stats: the hub's words, Pixal3D's, and
            # that a prop which did not fit the card's free memory took longer.
            assert patch["stats"]["warning"] == f"{FRONT_ONLY} The shape needed 3 tries. {MADE_SLOWLY}"
            assert patch["stats"]["ss_retries"] == 2 and patch["stats"]["mode"] == "single"
            assert patch["source"] == mesh_source(store.get_prop("P", pid)["images"])
            assert len(await asked(pm)) == 1

        store.add_images("P", pid, [(PNG + b"FAKE_MV_MISSING", "front.png"), (PNG, "side.png")])
        async with pixal3d_hub(tmp_path, store) as (pm, _, adapter):
            await pm.ensure_running("pixal3d")
            sent, patch, error = await _make(adapter, pm, store, pid)
            assert len(sent["views"]) == 2 and patch is None and error == MV_MISSING
            assert len(await asked(pm)) == 1                    # not made from the front view in a second try

    asyncio.run(main())


def test_a_job_keeps_the_order_the_artist_chose_and_pixal3d_orders_the_props(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch,
                                                                           quick_polls: None) -> None:
    """Pixal3D takes the props made from several views first through each of its steps (its launcher orders
    them); the job lists them as chosen, and says where each one is."""
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    ids = [store.add_prop("P", name)["id"] for name in ("Many A", "One B", "Many C", "One D", "Many E")]
    for pid in ids:
        if "Many" in store.get_prop("P", pid)["name"]:
            store.add_images("P", pid, [(PNG, "front.png"), (PNG, "side.png")])
        else:
            store.add_image("P", pid, PNG)
    monkeypatch.setattr(multiview, "available", lambda *args: True)

    async def main() -> None:
        async with pixal3d_hub(tmp_path, store, "--part-seconds", "0.05") as (pm, queue, adapter):
            job = await queue.submit("mesh", "P", ids)
            final, seen = await watch(queue, job["id"])
            assert final["status"] == "done", final
            assert all(s["prop_ids"] == ids for s in seen)
            currents = [s["current_prop"] for s in seen if s["current_prop"]]
            assert currents[0] in (ids[0], ids[2], ids[4])           # the first prop on the card has views
            (request,) = await asked(pm)
            assert [item["id"] for item in request["items"]] == ids
            assert [item["views"] is None for item in request["items"]] == [False, True, False, True, False]
            assert adapter.starts == 1

    asyncio.run(main())
    many = store.get_prop("P", ids[0])
    assert many["mesh"]["glb"] == "mesh/mesh.glb" and many["mesh"]["source"] == mesh_source(many["images"])
    assert many["mesh"]["source"].startswith("views:front:images/upload_001.png:*|left:")
    assert many["mesh"]["stats"]["mode"] == "multi" and many["mesh"]["stats"]["views"] == 2
    one = store.get_prop("P", ids[1])
    assert one["mesh"]["source"] == one["images"]["chosen"] and one["mesh"]["stats"]["mode"] == "single"


def test_pixal3d_starts_the_same_whatever_the_job() -> None:
    """Pixal3D reads its models when a job needs them, step by step, so a start says nothing about the job or
    the card: no --first-job, no --low-vram (pixal3d.low_vram is read by nothing), and it is ready in minutes."""
    spec = Pixal3DAdapter().launch_spec(7772, {"pixal3d": {"attn_backend": "sdpa", "low_vram": False}})
    assert spec.argv[1:] == [str(paths.LAUNCHERS / "pixal3d_server.py"), "--repo", str(paths.PIXAL3D_DIR),
                             "--port", "7772", "--attn-backend", "sdpa", "--sage", "auto", "--int8", "auto",
                             "--coarse-exact", "on", "--projects-root", str(paths.PROJECTS)]
    assert spec.ready_timeout == pixal3d.READY_TIMEOUT == 300.0
    assert Pixal3DAdapter().launch_spec(7772, {}).argv[spec.argv.index("--attn-backend") + 1] == "flash_attn"
    # pixal3d.sage: SageAttention in the fork (on by default after its self-check); false = flash_attn as before
    at = spec.argv.index("--sage") + 1
    assert Pixal3DAdapter().launch_spec(7772, {"pixal3d": {"sage": False}}).argv[at] == "off"
    assert Pixal3DAdapter().launch_spec(7772, {"pixal3d": {"sage": True}}).argv[at] == "on"
    # pixal3d.int8: the flows' linear layers as int8 GEMMs in the fork (on by default after its self-check)
    at = spec.argv.index("--int8") + 1
    assert Pixal3DAdapter().launch_spec(7772, {"pixal3d": {"int8": False}}).argv[at] == "off"
    assert Pixal3DAdapter().launch_spec(7772, {"pixal3d": {"int8": True}}).argv[at] == "on"
    # pixal3d.coarse_exact: the 3D panel's "Faster 3D generation" on is false; anything else keeps them exact
    at = spec.argv.index("--coarse-exact") + 1
    assert Pixal3DAdapter().launch_spec(7772, {"pixal3d": {"coarse_exact": False}}).argv[at] == "off"
    assert Pixal3DAdapter().launch_spec(7772, {"pixal3d": {"coarse_exact": "no"}}).argv[at] == "on"
    launcher = (paths.LAUNCHERS / "pixal3d_server.py").read_text(encoding="utf-8")
    assert all(f'"{flag}"' in launcher for flag in spec.argv[2::2])      # the launcher knows every one of them
    # The launcher orders the props of a batch; the job runner has no hook for groups or first props any more.
    assert Pixal3DAdapter.batch_stages == ("mesh",)
    assert not hasattr(ToolAdapter, "run_group") and not hasattr(ToolAdapter, "expect")


# ---- the multi-view models on disk ----------------------------------------------------------------------

def _pixal3d_repo(models_root: Path) -> Path:
    return models_root / "pixal3d" / "hub" / "models--TencentARC--Pixal3D"


def _snapshot(models_root: Path, commit: str = multiview.PINNED_REVISION, *, refs: bool = True) -> Path:
    """A HuggingFace cache like Pixal3D's, with its single-view pipeline.json only."""
    repo = _pixal3d_repo(models_root)
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "pipeline.json").write_text("{}", encoding="utf-8")
    if refs:
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(commit, encoding="utf-8")
    return snapshot


def _put_files(snapshot: Path, files: list[str], size: int = 10) -> None:
    for rel in files:
        (snapshot / rel).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / rel).write_bytes(b"x" * size)


def test_the_multiview_files_are_those_of_pixal3d() -> None:
    assert multiview.TOTAL_BYTES == 22_000_319_480
    if not PIXAL3D_INSTALL.is_file():
        pytest.skip("the Pixal3D submodule is not checked out")
    patterns = None
    for node in ast.walk(ast.parse(PIXAL3D_INSTALL.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "PIXAL3D_MULTI_VIEW_PATTERNS":
            patterns = ast.literal_eval(node.value)
    assert patterns is not None
    files = [f for p in patterns for f in ((p[:-2] + ".json", p[:-2] + ".safetensors") if p.endswith(".*") else (p,))]
    assert sorted(files) == sorted(multiview.FILES)


def test_multiview_models_available_and_downloaded_bytes(tmp_path: Path) -> None:
    root = tmp_path / "models"
    assert not multiview.available(root) and multiview.downloaded_bytes(root) == 0
    assert multiview.revision(root) == multiview.PINNED_REVISION
    commit = "0123456789abcdef0123456789abcdef01234567"
    snapshot = _snapshot(root, commit)
    assert multiview.revision(root) == commit
    files = list(multiview.FILES)
    _put_files(snapshot, files[:4], size=1000)
    blobs = _pixal3d_repo(root) / "blobs"
    blobs.mkdir()
    (blobs / "abc.incomplete").write_bytes(b"x" * 500)
    (blobs / "done_blob").write_bytes(b"x" * 7)                      # finished: counted through its snapshot entry
    assert not multiview.available(root) and multiview.downloaded_bytes(root) == 4 * 1000 + 500
    _put_files(snapshot, files[4:])
    (snapshot / files[-1]).write_bytes(b"")                          # an empty file is no file
    assert not multiview.available(root)
    (snapshot / files[-1]).write_bytes(b"x")
    assert multiview.available(root)
    # Only refs/main's snapshot counts (what Pixal3D loads offline); garbage in refs/main: the pinned commit.
    (_pixal3d_repo(root) / "refs" / "main").write_text("main", encoding="utf-8")
    assert not multiview.available(root) and multiview.revision(root) == multiview.PINNED_REVISION
    assert multiview.downloaded_bytes(root) == 500


# ---- the download ---------------------------------------------------------------------------------------

FAKE_DOWNLOAD = r'''
import json, os, sys, time
from pathlib import Path
plan = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))
args = sys.argv[1:]
record = {"argv": args, "env": {k: v for k, v in os.environ.items() if k.startswith("HF_")}}
Path(plan["record"]).write_text(json.dumps(record), encoding="utf-8")
print("fake download starting", flush=True)
while plan.get("wait_for") and not os.path.exists(plan["wait_for"]):
    time.sleep(0.02)
cache, rev = Path(args[args.index("--cache-dir") + 1]), args[args.index("--revision") + 1]
if plan.get("complete"):
    snapshot = cache / "models--TencentARC--Pixal3D" / "snapshots" / rev
    for rel in args[args.index("--resume-over-https") + 2:]:
        (snapshot / rel).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / rel).write_bytes(b"x" * 3)
print("fake download done", flush=True)
sys.exit(plan.get("exit", 0))
'''


# The fork's maker of the faster-loading copies (python -m pixal3d.staged.weights <snapshot>), as a stand-in.
FAKE_COPIES = r'''
import json, os, sys
from pathlib import Path
plan = json.loads(Path(PLAN).read_text(encoding="utf-8"))
wanted = ("HF_HOME", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES", "PIXAL3D_CARD_SM", "DISABLE_ADDMM_CUDA_LT")
record = {"argv": sys.argv[1:], "cwd": os.getcwd(), "env": {k: v for k, v in os.environ.items() if k in wanted}}
Path(plan["copies_record"]).write_text(json.dumps(record), encoding="utf-8")
print("fake copies made", flush=True)
sys.exit(plan.get("copies_exit", 0))
'''


class Download:
    """A BackgroundDownload running the stand-in script, into a temp models folder, and then the stand-in maker of
    the copies in a stand-in Pixal3D checkout. card: the PC's graphics card the copies are made for (none: the
    test does not ask nvidia-smi)."""

    def __init__(self, tmp_path: Path, card: GpuInfo | None = None, **plan: Any):
        self.root = tmp_path / "models"
        self.script = tmp_path / "fake_download.py"
        self.script.write_text(FAKE_DOWNLOAD, encoding="utf-8")
        self.record = tmp_path / "record.json"
        self.copies_record = tmp_path / "copies.json"
        self.fork = tmp_path / "fork"
        (self.fork / "pixal3d" / "staged").mkdir(parents=True)
        (self.fork / "pixal3d" / "__init__.py").write_text("", encoding="utf-8")
        (self.fork / "pixal3d" / "staged" / "__init__.py").write_text("", encoding="utf-8")
        (self.fork / "pixal3d" / "staged" / "weights.py").write_text(
            f"PLAN = {str(self.script.with_suffix('.json'))!r}\n" + FAKE_COPIES, encoding="utf-8")
        self.plan(**plan)
        self.lock = tmp_path / ".install.lock"
        self.log = tmp_path / "logs" / multiview.LOG_NAME
        self.bg = multiview.BackgroundDownload(self.root, python=Path(sys.executable), script=self.script,
                                               install_lock=self.lock, log_file=self.log, wait_poll=0.05,
                                               repo=self.fork, card=lambda: card)

    def plan(self, **plan: Any) -> None:
        self.script.with_suffix(".json").write_text(json.dumps({"record": str(self.record),
                                                                "copies_record": str(self.copies_record), **plan}),
                                                    encoding="utf-8")

    def recorded(self) -> dict[str, Any]:
        return json.loads(self.record.read_text(encoding="utf-8"))

    def copies(self) -> dict[str, Any]:
        return json.loads(self.copies_record.read_text(encoding="utf-8"))


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


async def _until(predicate: Any, timeout: float = 20.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def test_the_download_runs_once_and_ends_with_the_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIXAL3D_CARD_SM", raising=False)
    monkeypatch.delenv("DISABLE_ADDMM_CUDA_LT", raising=False)
    dl = Download(tmp_path, complete=True, wait_for=str(tmp_path / "go"))
    snapshot = _snapshot(dl.root)
    blobs = _pixal3d_repo(dl.root) / "blobs"
    blobs.mkdir()
    (blobs / "big.incomplete").write_bytes(b"\0" * 1000)

    async def main() -> None:
        before = dl.bg.snapshot()
        assert before == {"available": False, "downloading": False, "done_bytes": 1000,
                          "total_bytes": 22_000_319_480, "error": None}
        dl.bg.start()
        dl.bg.start()                                           # nothing more while it runs
        await _until(lambda: dl.record.exists(), what="the download to start")
        proc = dl.bg.proc
        assert dl.bg.snapshot()["downloading"] is True
        argv = dl.recorded()["argv"]
        assert argv[:6] == ["--cache-dir", str(dl.root / "pixal3d" / "hub"), "--revision", multiview.PINNED_REVISION,
                            "--resume-over-https", str(multiview.RESUME_OVER_HTTPS_BYTES)]
        assert argv[6:] == list(multiview.FILES)
        env = dl.recorded()["env"]
        assert env["HF_HOME"] == str(dl.root / "pixal3d") and "HF_HUB_OFFLINE" not in env
        assert "HF_HUB_DISABLE_XET" not in env                   # 1000 bytes: not worth continuing over HTTPS
        assert not dl.copies_record.exists()                     # not before the models are all there
        (tmp_path / "go").touch()
        await _until(lambda: not dl.bg.downloading(), what="the download to end")
        assert dl.bg.proc is proc
        assert dl.bg.snapshot() == {"available": True, "downloading": False, "done_bytes": multiview.TOTAL_BYTES,
                                    "total_bytes": multiview.TOTAL_BYTES, "error": None}
        await _until(lambda: "fake download done" in _text(dl.log), what="the download's output in its log")
        # then the faster-loading copies, made in the Pixal3D checkout by the venv's Python, card hidden, offline
        copies = dl.copies()
        assert copies["argv"] == [str(snapshot)] and Path(copies["cwd"]) == dl.fork
        assert copies["env"] == {"HF_HOME": str(dl.root / "pixal3d"), "HF_HUB_OFFLINE": "1",
                                 "CUDA_VISIBLE_DEVICES": "-1"}
        assert dl.bg.copier is not None and dl.bg.copier.returncode == 0
        await _until(lambda: "fake copies made" in _text(dl.log), what="the copies' output in the log")
        dl.record.unlink()
        dl.copies_record.unlink()
        dl.bg.start()                                           # the models are there: nothing to do
        assert not dl.bg.downloading()
        await dl.bg.close()

    asyncio.run(main())
    assert (snapshot / "pipeline_mv.json").is_file() and not dl.copies_record.exists()


@pytest.mark.parametrize("cc, told", [(7.5, {"PIXAL3D_CARD_SM": "75"}),
                                      (6.1, {"PIXAL3D_CARD_SM": "61", "DISABLE_ADDMM_CUDA_LT": "1"}),
                                      (8.6, {"PIXAL3D_CARD_SM": "86"})])
def test_the_copies_are_made_for_the_cards_precision_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             cc: float, told: dict[str, str]) -> None:
    """The maker runs with the card hidden, so it is told the card as the installer's is (gpu.pixal3d_card_env):
    without it an RTX 20xx or GTX 10xx PC would get copies in the Ampere profile's dtypes, which its loads never
    read, in place of the installer's single-view ones."""
    monkeypatch.delenv("PIXAL3D_CARD_SM", raising=False)
    monkeypatch.delenv("DISABLE_ADDMM_CUDA_LT", raising=False)
    card = GpuInfo(0, "NVIDIA GeForce test card", cc, 8192, 500, "596.36")
    dl = Download(tmp_path, card=card, complete=True)
    _snapshot(dl.root)

    async def main() -> None:
        dl.bg.start()
        await _until(lambda: not dl.bg.downloading(), what="the download and the copies to end")
        assert dl.bg.copier is not None and dl.bg.copier.returncode == 0
        await dl.bg.close()

    asyncio.run(main())
    assert dl.copies()["env"] == {"HF_HOME": str(dl.root / "pixal3d"), "HF_HUB_OFFLINE": "1",
                                  "CUDA_VISIBLE_DEVICES": "-1", **told}


def test_copies_that_fail_leave_the_models_available(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Pixal3D reads the originals then, only more slowly: no error for the artist, a warning in the log."""
    dl = Download(tmp_path, complete=True, copies_exit=1)
    _snapshot(dl.root)

    async def main() -> None:
        dl.bg.start()
        await _until(lambda: not dl.bg.downloading(), what="the download and the copies to end")
        assert dl.bg.copier is not None and dl.bg.copier.returncode == 1
        assert dl.bg.snapshot() == {"available": True, "downloading": False, "done_bytes": multiview.TOTAL_BYTES,
                                    "total_bytes": multiview.TOTAL_BYTES, "error": None}
        await dl.bg.close()

    with caplog.at_level("WARNING", logger=multiview.__name__):
        asyncio.run(main())
    assert any("copies of the multi-view models were not all made (exit code 1)" in r.getMessage()
               for r in caplog.records)


@pytest.mark.parametrize("code, message", [
    (1, "most often because the internet connection dropped"), (3, "ran out of space"),
    (4, "installation is incomplete"), (9, "stopped unexpectedly (exit code 9)"),
    (0, "some of their files are still missing")])
def test_a_failed_download_says_why(tmp_path: Path, code: int, message: str) -> None:
    dl = Download(tmp_path, exit=code)
    _snapshot(dl.root)

    async def main() -> None:
        dl.bg.start()
        await _until(lambda: not dl.bg.downloading(), what="the download to end")
        state = dl.bg.snapshot()
        assert state["downloading"] is False and state["available"] is False and message in state["error"]
        assert dl.bg.copier is None and not dl.copies_record.exists()      # no copies of models not there
        dl.plan(complete=True)
        dl.bg.start()                                           # again: it continues
        assert dl.bg.snapshot()["error"] is None and dl.bg.snapshot()["downloading"] is True
        await _until(lambda: not dl.bg.downloading(), what="the download to end")
        assert dl.bg.snapshot()["available"] is True
        await dl.bg.close()

    asyncio.run(main())


def test_the_download_waits_for_an_installer_and_continues_over_https(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    dl = Download(tmp_path, complete=True)
    _snapshot(dl.root)
    blobs = _pixal3d_repo(dl.root) / "blobs"
    blobs.mkdir()
    monkeypatch.setattr(multiview, "RESUME_OVER_HTTPS_BYTES", 1000)
    (blobs / "big.incomplete").write_bytes(b"x" * 1000)           # an interrupted big file
    with open(dl.lock, "a+b") as lock:                             # an installer in another window
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)

        async def main() -> None:
            dl.bg.start()
            await asyncio.sleep(0.4)
            assert dl.bg.proc is None and dl.bg.snapshot()["downloading"] is True
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            await _until(lambda: not dl.bg.downloading(), what="the download to end")
            assert dl.bg.snapshot()["available"] is True
            assert dl.recorded()["env"]["HF_HUB_DISABLE_XET"] == "1"

        asyncio.run(main())


def test_the_download_needs_room_and_pixal3d(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dl = Download(tmp_path, complete=True)
    _snapshot(dl.root)
    monkeypatch.setattr(multiview, "_free_space", lambda folder: 5_000_000_000)

    async def main() -> None:
        with pytest.raises(ToolError, match="not enough free space on drive") as e:
            dl.bg.start()
        assert e.value.status_code == 409 and "23.0 GB" in str(e.value) and not dl.bg.downloading()
        monkeypatch.setattr(multiview, "_free_space", lambda folder: 10 ** 14)
        dl.bg.python = tmp_path / "missing" / "python.exe"
        dl.bg.start()
        await _until(lambda: not dl.bg.downloading(), what="the download to give up")
        assert "Pixal3D is not installed yet" in dl.bg.snapshot()["error"] and not dl.record.exists()

    asyncio.run(main())


# ---- the download script (launchers/mv_download.py) --------------------------------------------------------

class FakeHub(types.ModuleType):
    """A stand-in huggingface_hub: its snapshot_download follows a script of outcomes."""

    def __init__(self, outcomes: list[str]):
        super().__init__("huggingface_hub")
        self.__version__ = "0.0-fake"
        self.constants = types.SimpleNamespace(HF_HUB_DISABLE_XET=False)
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def snapshot_download(self, repo_id: str, **kwargs: Any) -> str:
        self.calls.append({"repo_id": repo_id, "xet_off": self.constants.HF_HUB_DISABLE_XET, **kwargs})
        cache = Path(kwargs["cache_dir"])
        repo = cache / "models--TencentARC--Pixal3D"
        snapshot = repo / "snapshots" / kwargs["revision"]
        outcome = self.outcomes.pop(0)
        if outcome == "partial":            # an interrupted big file
            (repo / "blobs").mkdir(parents=True, exist_ok=True)
            with open(repo / "blobs" / "x.incomplete", "ab") as f:
                f.write(b"x" * 64)
            raise ConnectionError("the connection dropped")
        if outcome == "full":
            raise OSError(28, "No space left on device")
        if outcome == "done":
            for rel in kwargs["allow_patterns"]:
                (snapshot / rel).parent.mkdir(parents=True, exist_ok=True)
                (snapshot / rel).write_bytes(b"x")
        return str(snapshot)                # "offline": the folder, whatever is in it


def _script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcomes: list[str]) -> tuple[FakeHub, list[str]]:
    hub = FakeHub(outcomes)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(mv_download, "RETRY_DELAY", 0.0)
    return hub, ["--cache-dir", str(tmp_path / "hub"), "--revision", multiview.PINNED_REVISION,
                 "--resume-over-https", "100", "pipeline_mv.json", "ckpts/a_mv.json"]


def test_the_download_script_retries_until_every_file_is_there(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub, argv = _script(monkeypatch, tmp_path, ["partial", "partial", "offline", "done"])
    assert mv_download.main(argv) == mv_download.EXIT_OK
    assert [c["xet_off"] for c in hub.calls] == [False, False, True, True]   # 128 bytes unfinished: over HTTPS
    assert all(c["revision"] == multiview.PINNED_REVISION and c["max_workers"] == 1 and
               c["allow_patterns"] == ["pipeline_mv.json", "ckpts/a_mv.json"] for c in hub.calls)

    hub, argv = _script(monkeypatch, tmp_path / "2", ["offline"] * mv_download.MAX_STALLED_ATTEMPTS)
    assert mv_download.main(argv) == mv_download.EXIT_FAILED            # never any progress: it gives up
    hub, argv = _script(monkeypatch, tmp_path / "3", ["full"])
    assert mv_download.main(argv) == mv_download.EXIT_DISK_FULL
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)          # not installed in that Python
    assert mv_download.main(argv) == mv_download.EXIT_NO_HF


# ---- HTTP API ---------------------------------------------------------------------------------------------

@pytest.fixture
def api(tmp_path: Path) -> Iterator[tuple[TestClient, Download]]:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>Spellcast3D</h1>", encoding="utf-8")
    dl = Download(tmp_path, complete=True)
    adapters = {"forge": FakeAdapter("forge", "images", True, tmp_path),
                "pixal3d": FakeAdapter("pixal3d", "mesh", True, tmp_path),
                "imesh": FakeAdapter("imesh", "retopo", False, tmp_path)}
    app = create_app(adapters, settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=static, logs_dir=tmp_path / "logs", multiview_download=dl.bg,
                     gpu_probe=lambda: GpuInfo(0, "Fake RTX 3060", 8.6, 12288, 3000, "580.10"))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client, dl


def test_api_views_and_uploads(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = api
    client.post("/api/projects", json={"name": "Demo"})
    draft = client.post("/api/projects/Demo/draft").json()
    url = f"/api/projects/Demo/props/{draft['id']}"
    up = client.post(f"{url}/images/upload", files=[("files", ("C:\\art\\knight_front.png", PNG, "image/png")),
                                                   ("files", ("knight_side.jpg", JPG, "image/jpeg")),
                                                   ("files", ("knight_back.png", PNG, "image/png"))])
    assert up.status_code == 200, up.text
    prop = up.json()
    assert prop["draft"] is False and prop["name"] == "Knight front"
    assert prop["images"]["chosen"] == "images/upload_001.png"
    assert [(v["view"], v["src"], v["rect"]) for v in prop["images"]["views"]] == [
        ("front", "images/upload_001.png", None), ("left", "images/upload_002.jpg", None),
        ("back", "images/upload_003.png", None)]
    one = client.post(f"{url}/images/upload", files={"file": ("x.png", PNG, "image/png")}).json()
    assert one["images"]["chosen"] == "images/upload_004.png" and len(one["images"]["views"]) == 3
    assert client.post(f"{url}/images/upload", files={"files[]": ("y.png", PNG, "image/png")}).status_code == 200
    none = client.post(f"{url}/images/upload", data={"note": "no file"})
    assert none.status_code == 400 and "Choose a picture" in none.json()["detail"]
    bad = client.post(f"{url}/images/upload", files=[("files", ("a.png", PNG, "image/png")),
                                                    ("files", ("evil.png", b"MZ\x90\x00", "image/png"))])
    assert bad.status_code == 415 and bad.json()["detail"].startswith("evil.png: ")
    assert client.post("/api/projects/Demo/props/p_ffffff/images/upload",
                       files={"file": ("x.png", PNG, "image/png")}).status_code == 404
    with monkeypatch.context() as mp:
        mp.setattr("spellcast.server.MAX_UPLOAD_BYTES", 1)
        mp.setattr("spellcast.server.UPLOAD_OVERHEAD", 0)
        too_big = client.post(f"{url}/images/upload", files={"file": ("x.png", PNG, "image/png")})
    assert too_big.status_code == 413 and "too big together" in too_big.json()["detail"]

    # The sides the page read from the file names come with the files.
    sided = client.post(f"{url}/images/upload", data={"sides": '["back", "front"]'},
                        files=[("files", ("b.png", PNG, "image/png")), ("files", ("f.png", PNG, "image/png"))])
    assert sided.status_code == 200, sided.text
    assert sided.json()["images"]["chosen"] == "images/upload_007.png"
    assert [(v["view"], v["src"]) for v in sided.json()["images"]["views"]] == [
        ("front", "images/upload_007.png"), ("back", "images/upload_006.png")]
    odd = client.post(f"{url}/images/upload", data={"sides": "[front"},
                      files=[("files", ("b.png", PNG, "image/png")), ("files", ("f.png", PNG, "image/png"))])
    assert odd.status_code == 400 and "Reload the page" in odd.json()["detail"]
    # Too many: said in plain words, before they are read (Starlette's own limit included).
    for count in (MAX_IMAGE_UPLOADS + 1, MAX_IMAGE_UPLOADS * 4 + 1):
        many = client.post(f"{url}/images/upload", files=[("files", (f"{i}.png", PNG, "image/png")) for i in range(count)])
        assert many.status_code == 413 and "too many pictures" in many.json()["detail"], many.text
    assert len(client.get("/api/projects/Demo").json()["props"][0]["images"]["candidates"]) == 7

    views = [{"view": "front", "src": "images/upload_004.png", "rect": [0, 0, 64, 64]}]
    put = client.put(f"{url}/images/views", json={"views": views})
    assert put.status_code == 200 and put.json()["images"]["views"] == views
    assert client.get("/api/projects/Demo").json()["props"][0]["images"]["views"] == views
    refused = client.put(f"{url}/images/views", json={"views": [{"view": "front", "src": "nope.png", "rect": None}]})
    assert refused.status_code == 400 and "own pictures" in refused.json()["detail"]
    assert client.put(f"{url}/images/views", json={}).status_code == 400
    assert client.put(f"{url}/images/views", json={"views": []}).json()["images"]["views"] == []


def test_api_multiview_status_and_download(api: Any) -> None:
    client, dl = api
    _snapshot(dl.root)
    state = client.get("/api/status").json()["multiview"]
    assert state == {"available": False, "downloading": False, "done_bytes": 0, "total_bytes": 22_000_319_480,
                     "error": None}
    started = client.post("/api/pixal3d/multiview/download")
    assert started.status_code == 200 and started.json()["total_bytes"] == 22_000_319_480
    deadline = time.monotonic() + 20
    while not client.get("/api/status").json()["multiview"]["available"]:
        assert time.monotonic() < deadline, client.get("/api/status").json()["multiview"]
        time.sleep(0.05)
    assert client.post("/api/pixal3d/multiview/download").json()["downloading"] is False   # nothing left to do


def test_api_multiview_download_needs_pixal3d(tmp_path: Path) -> None:
    dl = Download(tmp_path, complete=True)
    adapters = {"pixal3d": FakeAdapter("pixal3d", "mesh", True, tmp_path, install="not_installed"),
                "imesh": FakeAdapter("imesh", "retopo", False, tmp_path, supported=(True, ""))}
    app = create_app(adapters, settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=tmp_path, logs_dir=tmp_path / "logs", multiview_download=dl.bg, gpu_probe=lambda: None)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        refused = client.post("/api/pixal3d/multiview/download")
        assert refused.status_code == 409 and not dl.bg.downloading() and not dl.record.exists()
    unsupported = {"pixal3d": FakeAdapter("pixal3d", "mesh", True, tmp_path,
                                          supported=(False, "Pixal3D needs an RTX 30xx or newer."))}
    app = create_app(unsupported, settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=tmp_path, logs_dir=tmp_path / "logs", multiview_download=dl.bg, gpu_probe=lambda: None)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        refused = client.post("/api/pixal3d/multiview/download")
        assert refused.status_code == 409 and "RTX 30xx" in refused.json()["detail"]
