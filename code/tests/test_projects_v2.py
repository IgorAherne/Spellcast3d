"""Projects for the props workbench: untitled projects, the last one opened, renaming, the tools' scratch
folder (temp/), the draft that becomes a prop with its first picture, thumbnails, a model imported as a new
prop, and saving every prop's result into the project's export folder.

Run with the hub venv:  <venv>\\Scripts\\python.exe -m pytest code\\tests -q
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spellcast import exports, paths
from spellcast.jobs import RUNNING, Job
from spellcast.projects import NO_IMAGE_YET, ProjectError, ProjectStore, box_name_from_prompt, thumb_signature
from spellcast.server import create_app
from test_core import FakeAdapter, make_settings
from test_projection_hub import FINISHED, PNG, FakeImesh, FakeConverter, make_zip, ready_to_bake, wait_for

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Spellcast3D is Windows-only")

WEBP = b"RIFF\x24\0\0\0WEBPVP8 " + b"\0" * 64
JPG = b"\xff\xd8\xff\xe0" + b"\0" * 64
UNTITLED_ID = re.compile(r"^untitled-\d{8}-\d{6}(-\d+)?$")


def _state(root: Path) -> dict[str, Any]:
    return json.loads((root / ".state.json").read_text(encoding="utf-8"))


def _trashed(root: Path) -> list[str]:
    trash = root / ".trash"
    return sorted(p.name for p in trash.iterdir()) if trash.is_dir() else []


# ---- projects ------------------------------------------------------------------------------------

def test_untitled_projects_are_numbered_and_named_later(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    store = ProjectStore(root)
    first = store.create_project()
    assert UNTITLED_ID.match(first["id"]) and first["name"] == "Untitled" and first["untitled"] is True
    second = store.create_project("   ")                       # a blank name is no name
    assert second["id"] != first["id"] and UNTITLED_ID.match(second["id"]) and second["name"] == "Untitled 2"
    assert {(p["id"], p["name"], p["untitled"], p["prop_count"]) for p in store.list_projects()} == {
        (first["id"], "Untitled", True, 0), (second["id"], "Untitled 2", True, 0)}
    assert "id" not in json.loads((root / first["id"] / "project.json").read_text(encoding="utf-8"))

    named = store.rename_project(first["id"], "  Castle: props ")
    assert (named["id"], named["name"], named["untitled"]) == ("Castle props", "Castle props", False)
    assert not (root / first["id"]).exists() and store.get_project("Castle props")["name"] == "Castle props"
    assert store.rename_project("Castle props", "Castle props") == store.get_project("Castle props")
    recased = store.rename_project("Castle props", "castle PROPS")          # the same folder on Windows
    assert recased["id"] == "castle PROPS" and [d.name for d in root.iterdir() if d.name.startswith("c")] == \
        ["castle PROPS"]
    for bad in ("???", None, 5):
        with pytest.raises(ProjectError) as e:
            store.rename_project(second["id"], bad)
        assert e.value.status_code == 400
    with pytest.raises(ProjectError) as e:
        store.rename_project(second["id"], "Castle Props")
    assert e.value.status_code == 409 and "already exists" in str(e.value)
    assert store.create_project()["name"] == "Untitled"          # free again: the first one has a name now
    with pytest.raises(ProjectError) as e:
        store.rename_project("Nope", "Other")
    assert e.value.status_code == 404


def test_the_last_project_is_reopened_and_empty_untitled_ones_are_tidied_away(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    store = ProjectStore(root)
    fresh = store.open_last()                                   # nothing yet: a new untitled project
    assert fresh["untitled"] and _state(root) == {"last_opened": fresh["id"]}
    assert store.open_last()["id"] == fresh["id"]
    assert store.open_project(fresh["id"])["id"] == fresh["id"] and _trashed(root) == []

    store.create_project("Props")
    store.add_prop("Props", "Barrel")
    store.open_project("Props")                                 # leaving an empty untitled project...
    assert _state(root) == {"last_opened": "Props"}
    assert [name.startswith(fresh["id"] + "_") for name in _trashed(root)] == [True]    # ...tidies it away
    assert [p["id"] for p in store.list_projects()] == ["Props"]

    typed = store.create_project()
    draft = store.ensure_draft(typed["id"])
    store.patch_prop(typed["id"], draft["id"], {"images": {"settings": {"prompt": "a lantern"}}})
    busy = store.create_project()
    empty = store.create_project()
    store.open_project(typed["id"])
    store.open_project("Props")                                 # a prompt is content: kept
    store.open_project(busy["id"], busy={busy["id"]})
    store.open_project("Props", busy={busy["id"]})              # a job is running for it: kept
    assert {p["id"] for p in store.list_projects()} == {"Props", typed["id"], busy["id"], empty["id"]}

    store.tidy_untitled(keep=empty["id"])                       # at hub start: all but the open one
    assert {p["id"] for p in store.list_projects()} == {"Props", typed["id"], empty["id"]}
    assert len(_trashed(root)) == 2
    store.tidy_untitled()                                       # keep = the one opened last ("Props")
    assert {p["id"] for p in store.list_projects()} == {"Props", typed["id"]}

    (root / ".state.json").write_text('{"last_opened": "Gone"}', encoding="utf-8")
    props_file = root / "Props" / "project.json"
    props_file.write_text(props_file.read_text(encoding="utf-8").replace(
        store.get_project("Props")["updated"], "2026-01-01T00:00:00+01:00"), encoding="utf-8")
    assert store.open_last()["id"] == typed["id"]               # changed last
    (root / ".state.json").write_text("not json", encoding="utf-8")
    assert store.open_last()["id"] == typed["id"]

    store.rename_project(typed["id"], "Lanterns")               # the renamed project stays the last one
    assert _state(root) == {"last_opened": "Lanterns"}


def test_a_new_project_is_the_one_opened_last(tmp_path: Path) -> None:
    """"New project" in the page: a reload comes back to it, and leaving an empty untitled one tidies it."""
    root = tmp_path / "projects"
    store = ProjectStore(root)
    store.create_project("Props")
    store.add_prop("Props", "Barrel")
    store.open_project("Props")
    fresh = store.new_project()
    assert fresh["untitled"] and _state(root) == {"last_opened": fresh["id"]}
    assert store.new_project("  ")["id"] == fresh["id"]          # it is new already: no "Untitled 2"
    assert store.open_last()["id"] == fresh["id"]
    named = store.new_project("Lanterns")
    assert named["id"] == "Lanterns" and _state(root) == {"last_opened": "Lanterns"}
    assert {p["id"] for p in store.list_projects()} == {"Props", "Lanterns"}     # the empty one went to .trash
    assert [name.startswith(fresh["id"] + "_") for name in _trashed(root)] == [True]
    with pytest.raises(ProjectError) as e:
        store.new_project("lanterns")
    assert e.value.status_code == 409 and _state(root) == {"last_opened": "Lanterns"}
    store.open_project("Props")
    made = store.new_project()                                  # Props has a prop: a new one is made
    assert made["untitled"] and _state(root) == {"last_opened": made["id"]} and len(store.list_projects()) == 3


def test_projects_saved_before_props_open_unchanged(tmp_path: Path) -> None:
    """A project.json from the rows-of-cards version: no draft, no thumbnails, no "untitled", no "accepted"."""
    root = tmp_path / "projects"
    folder = root / "E2E Props"
    (folder / "props" / "p_025d38").mkdir(parents=True)
    old = {"name": "E2E Props", "created": "2026-09-23T18:10:52+02:00", "updated": "2026-09-24T04:58:13+02:00",
           "ui": {"unlocked_row": 4, "active_row": 3, "positions": {"images": 2, "mesh": 0, "retopo": 1}},
           "props": [{"id": "p_025d38", "name": "Wooden barrel", "created": "2026-09-23T18:11:00+02:00",
                      "images": {"settings": {"prompt": "a wooden barrel"}, "candidates": ["images/0002.png"],
                                 "chosen": "images/0002.png", "reference": None},
                      "mesh": {"settings": {"seed": 5}, "glb": "mesh/mesh.glb", "source": "images/0002.png"},
                      "retopo": {"settings": {}, "obj": "retopo/retopo.obj"}, "projection": {}, "rig": {}}]}
    (folder / "project.json").write_text(json.dumps(old), encoding="utf-8")
    before = (folder / "project.json").read_bytes()
    store = ProjectStore(root)
    project = store.open_project("E2E Props")
    assert (project["id"], project["name"], project["untitled"]) == ("E2E Props", "E2E Props", False)
    assert project["ui"] == {"stage": "projection"}             # the row it had open (active_row 3)
    prop = project["props"][0]
    assert prop["draft"] is False and prop["thumbs"] == {} and prop["name"] == "Wooden barrel"
    assert prop["images"]["chosen"] == "images/0002.png" and prop["mesh"]["settings"]["seed"] == 5
    assert prop["projection"]["settings"]["texture_size"] == 2048
    assert prop["retopo"]["accepted"] is True       # its one clean mesh: the one size it asks for, so picked
    assert prop["retopo"]["settings"]["target_vertices"] == 5000 and prop["retopo"]["settings"]["rosy"] == 4
    # Made before there were other retopology methods: Instant Meshes, which the prop is set to.
    assert prop["retopo"]["method"] == "imesh" and prop["retopo"]["settings"]["method"] == "imesh"
    assert (folder / "project.json").read_bytes() == before     # reading changes nothing
    assert store.list_projects()[0]["prop_count"] == 1
    store.tidy_untitled()
    assert store.open_last()["id"] == "E2E Props"               # never tidied away: it has a name
    draft = store.ensure_draft("E2E Props")                     # created when the page asks for it
    assert draft["draft"] and draft["mesh"]["settings"]["seed"] == 5
    assert [p["id"] for p in store.get_project("E2E Props")["props"]] == ["p_025d38", draft["id"]]


# ---- the tools' scratch space: projects/<project>/temp/ ---------------------------------------------

def _content(folder: Path) -> dict[str, bytes]:
    """Every file of a project but those in temp/, with what is in it."""
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*")
            if p.is_file() and p.relative_to(folder).parts[0] != "temp"}


def _project_with_everything(store: ProjectStore, name: str) -> Path:
    """A project with a prop that has a picture, a deleted prop, and a file of the artist's own beside them."""
    store.create_project(name)
    kept = store.add_prop(name, "Barrel")["id"]
    store.add_image(name, kept, PNG)
    gone = store.add_prop(name, "Crate")["id"]
    store.add_image(name, gone, PNG)
    store.delete_prop(name, gone)
    folder = store.project_dir(name)
    (folder / "notes.txt").write_text("mine", encoding="utf-8")
    assert {p.name for p in folder.iterdir()} == {"project.json", "props", "deleted", "notes.txt"}
    return folder


def test_a_jobs_work_folder_is_made_in_temp_and_removed_without_a_trace(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    folder = _project_with_everything(store, "Props")
    before = _content(folder)
    work = store.work_dir("Props", "mesh-j_12ab34cd")
    assert work == folder / "temp" / "mesh-j_12ab34cd" and work.is_dir()
    assert store.work_dir("Props", "mesh-j_12ab34cd") == work                # there already: the same one
    (work / "p_000001").mkdir()
    (work / "p_000001" / "cond.bin").write_bytes(b"\0" * 1000)
    # Neither end of the pair takes anything but a plain folder name: nothing in temp/ can reach the props.
    for bad in ("", ".", "..", "a/b", "a\\b", "..\\props", "C:x"):
        with pytest.raises(ValueError):
            store.work_dir("Props", bad)
        with pytest.raises(ValueError):
            store.remove_work_dir("Props", bad)
    assert (folder / "props").is_dir()
    with pytest.raises(ProjectError) as e:
        store.work_dir("Nope", "mesh-j_1")
    assert e.value.status_code == 404

    # A file a stopped tool still holds open keeps the folder for now (the caller tries again, or a sweep does).
    with open(work / "p_000001" / "cond.bin", "rb"):
        assert store.remove_work_dir("Props", "mesh-j_12ab34cd") is False
        assert (work / "p_000001" / "cond.bin").is_file()
    assert store.remove_work_dir("Props", "mesh-j_12ab34cd") is True
    assert not (folder / "temp").exists()                       # temp/ itself goes with its last folder
    assert store.remove_work_dir("Props", "mesh-j_12ab34cd") is True and store.remove_work_dir("Nope", "x") is True
    assert _content(folder) == before
    assert {p.name for p in folder.iterdir()} == {"project.json", "props", "deleted", "notes.txt"}


def test_leftover_work_folders_are_swept_and_nothing_else_is_touched(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    store = ProjectStore(root)
    folders = {name: _project_with_everything(store, name) for name in ("Props", "Other")}
    before = {name: _content(folder) for name, folder in folders.items()}
    for name in folders:
        for job in ("mesh-j_aaaa1111", "mesh-j_bbbb2222"):
            (store.work_dir(name, job) / "p_000001").mkdir()
            (store.work_dir(name, job) / "p_000001" / "latents.bin").write_bytes(b"x" * 100)
        (folders[name] / "temp" / "stray.tmp").write_bytes(b"x")
    (root / ".trash" / "old_20260101-000000" / "temp" / "mesh-j_cccc3333").mkdir(parents=True)

    # One project, but the folder of the job that runs in it now.
    assert store.clear_work_dirs("Props", keep="mesh-j_bbbb2222") == 2
    assert [p.name for p in (folders["Props"] / "temp").iterdir()] == ["mesh-j_bbbb2222"]
    assert len(list((folders["Other"] / "temp").iterdir())) == 3
    # `keep` may be asked for when the leftovers are listed (the server passes the job queue's own answer).
    asked: list[str] = []
    assert store.clear_work_dirs("Props", keep=lambda project: asked.append(project) or "mesh-j_bbbb2222") == 0
    assert asked == ["Props"] and (folders["Props"] / "temp" / "mesh-j_bbbb2222" / "p_000001" / "latents.bin").is_file()
    assert store.clear_work_dirs("Props") == 1 and not (folders["Props"] / "temp").exists()
    assert store.clear_work_dirs("Props") == 0                   # nothing left: nothing to do
    # Every project (the hub's start); what Spellcast3D keeps in its own dot-folders is not looked at.
    assert store.clear_work_dirs() == 3 and not (folders["Other"] / "temp").exists()
    assert (root / ".trash" / "old_20260101-000000" / "temp" / "mesh-j_cccc3333").is_dir()
    with pytest.raises(ProjectError):
        store.clear_work_dirs("Nope")
    assert ProjectStore(tmp_path / "no projects yet").clear_work_dirs() == 0
    for name, folder in folders.items():
        assert _content(folder) == before[name]
        assert {p.name for p in folder.iterdir()} == {"project.json", "props", "deleted", "notes.txt"}


def test_files_in_temp_are_never_served_and_are_no_content(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    store = ProjectStore(root)
    store.create_project("Props")
    work = store.work_dir("Props", "mesh-j_aaaa1111")
    (work / "preview.png").write_bytes(PNG)
    (root / "Props" / "props" / "temp").mkdir()                 # a folder of that name further down is no scratch
    (root / "Props" / "props" / "temp" / "x.png").write_bytes(PNG)
    assert store.resolve_file("Props", "props/temp/x.png").is_file()
    for path in ("temp/mesh-j_aaaa1111/preview.png", "Temp/mesh-j_aaaa1111/preview.png",
                 "TEMP\\mesh-j_aaaa1111\\preview.png", "props/../temp/mesh-j_aaaa1111/preview.png",
                 "temp./mesh-j_aaaa1111/preview.png", "./temp/mesh-j_aaaa1111/preview.png"):
        with pytest.raises(ProjectError) as e:
            store.resolve_file("Props", path)
        assert e.value.status_code == 404, path

    # An untitled project with nothing in it but a job's leftovers is still empty: tidied away, without them.
    empty = store.create_project()
    (store.work_dir(empty["id"], "mesh-j_bbbb2222") / "latents.bin").write_bytes(b"x" * 100)
    store.open_project(empty["id"])
    store.open_project("Props")
    (trashed,) = _trashed(root)
    assert trashed.startswith(empty["id"]) and not (root / ".trash" / trashed / "temp").exists()
    assert (root / ".trash" / trashed / "project.json").is_file()


def test_box_names_from_prompts() -> None:
    assert box_name_from_prompt("a wooden barrel with iron bands, game asset") == "Wooden barrel"
    assert box_name_from_prompt("The knight holding a sword") == "Knight"
    assert box_name_from_prompt("low poly fantasy lantern hanging on a hook") == "Low poly fantasy lantern"
    assert box_name_from_prompt("treasure chest. gold coins") == "Treasure chest"
    assert box_name_from_prompt("masterpiece, best quality, (wooden barrel:1.3), 3d render") == "Wooden barrel"
    assert box_name_from_prompt("(masterpiece:1.2), <lora:crates:0.8> a crate [of apples]") == "Crate"
    assert box_name_from_prompt("  ,  ") == "" and box_name_from_prompt("") == ""
    assert box_name_from_prompt("masterpiece, 8k, white background") == ""
    assert box_name_from_prompt("with love") == "With love"
    assert len(box_name_from_prompt("x" * 100)) == 40


# ---- the draft becomes a prop ------------------------------------------------------------------

def test_the_draft_becomes_a_prop_with_its_first_picture(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    first = store.add_prop("P", "Barrel")
    store.patch_prop("P", first["id"], {"images": {"settings": {"prompt": "barrel", "cfg_scale": 3.5,
                                                                "mode": "describe"}}})
    draft = store.ensure_draft("P")
    assert draft["draft"] is True and draft["name"] == "" and draft["images"]["chosen"] is None
    assert draft["images"]["settings"]["prompt"] == "" and draft["images"]["settings"]["cfg_scale"] == 3.5
    assert draft["images"]["settings"]["mode"] == "describe"
    assert store.ensure_draft("P")["id"] == draft["id"]         # at most one
    assert store.list_projects()[0]["prop_count"] == 1          # the draft is not a prop yet

    with pytest.raises(ProjectError) as e:
        store.pack_prop("P", draft["id"])
    assert e.value.status_code == 409 and str(e.value) == NO_IMAGE_YET
    assert store.get_prop("P", draft["id"])["draft"] is True

    assert store.patch_prop("P", draft["id"], {"name": ""})["name"] == ""        # the draft may stay unnamed
    with pytest.raises(ProjectError, match="Please give the prop a name"):
        store.patch_prop("P", first["id"], {"name": "  "})                       # a prop may not
    store.patch_prop("P", draft["id"], {"images": {"settings": {"prompt": "a wooden barrel with iron bands"}}})
    imported = store.add_prop("P", "Chair")                     # e.g. a model imported meanwhile
    packed = store.add_image("P", draft["id"], PNG)             # its first picture: a prop now
    assert packed["draft"] is False and packed["name"] == "Wooden barrel"
    assert [p["id"] for p in store.get_project("P")["props"]] == [first["id"], imported["id"], draft["id"]]
    assert store.pack_prop("P", draft["id"]) == packed          # the old "pack" changes nothing
    assert store.list_projects()[0]["prop_count"] == 3

    nxt = store.ensure_draft("P")
    assert nxt["id"] != draft["id"] and nxt["images"]["settings"]["prompt"] == ""
    assert store.add_image("P", nxt["id"], PNG, "old_lamp-front.png")["name"] == "Old lamp front"  # no prompt
    anonymous = store.ensure_draft("P")
    assert store.add_image("P", anonymous["id"], PNG)["name"] == "Prop 5"         # no prompt, no file name
    named = store.ensure_draft("P")
    store.patch_prop("P", named["id"], {"name": " Old lamp ", "images": {"settings": {"prompt": "lamp"}}})
    assert store.add_image("P", named["id"], JPG, "x.jpg")["name"] == "Old lamp"   # the artist's name wins
    assert store.add_image("P", named["id"], PNG)["draft"] is False               # a prop stays one

    # A generation's first picture (the job's result, and a picked candidate) makes a prop too.
    generated = store.ensure_draft("P")
    store.patch_prop("P", generated["id"], {"images": {"settings": {"prompt": "iron lantern, game asset"}}})
    nothing = store.apply_stage_patch("P", generated["id"], "images", {"candidates": [], "chosen": None})
    assert nothing["draft"] is True                             # cancelled before its first picture
    made = store.apply_stage_patch("P", generated["id"], "images",
                                   {"candidates": ["images/0001.png"], "chosen": "images/0001.png"})
    assert made["draft"] is False and made["name"] == "Iron lantern"
    picked = store.ensure_draft("P")
    file = tmp_path / "projects" / "P" / "project.json"
    data = json.loads(file.read_text(encoding="utf-8"))         # a draft with pictures but none chosen
    data["props"][-1]["images"]["candidates"] = ["images/0001.png"]
    file.write_text(json.dumps(data), encoding="utf-8")
    assert store.select_image("P", picked["id"], "images/0001.png")["draft"] is False


def test_a_draft_still_generating_becomes_a_prop_when_a_new_one_is_asked_for(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    draft = store.ensure_draft("P")
    store.patch_prop("P", draft["id"], {"images": {"settings": {"prompt": "a copper kettle"}}})
    assert store.ensure_draft("P", busy={"p_000000"})["id"] == draft["id"]      # its own job is not running
    fresh = store.ensure_draft("P", busy={draft["id"]})         # "+ New prop" while its pictures are made
    assert fresh["id"] != draft["id"] and fresh["draft"] is True and fresh["images"]["settings"]["prompt"] == ""
    kettle = store.get_prop("P", draft["id"])
    assert kettle["draft"] is False and kettle["name"] == "Copper kettle"
    assert [p["id"] for p in store.get_project("P")["props"]] == [draft["id"], fresh["id"]]
    # Its pictures fill it when they arrive.
    later = store.apply_stage_patch("P", draft["id"], "images", {"candidates": ["images/0001.png"],
                                                                "chosen": "images/0001.png"})
    assert later["images"]["chosen"] == "images/0001.png" and later["name"] == "Copper kettle"


def test_a_retopo_saved_from_the_viewer_before_accept_existed_counts_as_accepted(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    (root / "Old" / "props").mkdir(parents=True)
    props = [{"id": f"p_00000{i}", "name": name, "retopo": {"obj": obj, "saved_from": saved_from}}
             for i, (name, obj, saved_from) in enumerate((("Viewer", "retopo/retopo.obj", "viewer"),
                                                          ("Auto", "retopo/retopo.obj", "auto"),
                                                          ("None", None, None)))]
    (root / "Old" / "project.json").write_text(json.dumps({"name": "Old", "props": props}), encoding="utf-8")
    store = ProjectStore(root)
    props = store.get_project("Old")["props"]
    # (the auto one too, since props pick their only size by themselves: there is nothing to choose)
    assert [p["retopo"]["accepted"] for p in props] == [True, True, False]
    assert [p["retopo"]["method"] for p in props] == ["imesh", "imesh", None]   # None: no clean mesh yet


def test_a_draft_left_with_its_picture_by_an_older_page_is_a_prop(tmp_path: Path) -> None:
    """The page before auto-pack left packing to the artist: a draft with a chosen picture opens as a prop."""
    root = tmp_path / "projects"
    (root / "Old" / "props").mkdir(parents=True)
    props = [{"id": "p_000001", "name": "Barrel", "images": {"chosen": "images/0001.png"}},
             {"id": "p_000002", "name": "", "draft": True,
              "images": {"settings": {"prompt": "a cute toy rocket spaceship on a gray background"},
                         "candidates": ["images/0001.png"], "chosen": "images/0001.png"}}]
    (root / "Old" / "project.json").write_text(json.dumps({"name": "Old", "props": props}), encoding="utf-8")
    store = ProjectStore(root)
    rocket = store.get_project("Old")["props"][1]
    assert (rocket["id"], rocket["draft"], rocket["name"]) == ("p_000002", False, "Cute toy rocket spaceship")
    draft = store.ensure_draft("Old")                           # the next prop gets a draft of its own
    assert draft["id"] not in ("p_000001", "p_000002") and draft["images"]["chosen"] is None
    assert [p["draft"] for p in store.get_project("Old")["props"]] == [False, False, True]


def test_a_hand_edited_project_keeps_one_draft(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    ids = [store.add_prop("P", name)["id"] for name in ("A", "B")]
    file = tmp_path / "projects" / "P" / "project.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    for prop in data["props"]:
        prop["draft"], prop["thumbs"] = True, {"mesh": "not a thumb", "nope": {"path": "x"}}
    file.write_text(json.dumps(data), encoding="utf-8")
    props = store.get_project("P")["props"]
    assert [(p["id"], p["draft"], p["thumbs"]) for p in props] == [(ids[0], True, {}), (ids[1], False, {})]


# ---- thumbnails ------------------------------------------------------------------------------------

def test_thumbnails_are_stored_and_dropped_with_their_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    pid = store.add_prop("P", "Barrel")["id"]
    thumbs = store.prop_dir("P", pid) / "thumbs"
    prop = store.set_thumb("P", pid, "mesh", PNG)
    first = prop["thumbs"]["mesh"]
    assert first["path"] == "thumbs/mesh.png" and isinstance(first["v"], int)
    assert (thumbs / "mesh.png").read_bytes() == PNG
    again = store.set_thumb("P", pid, "mesh", WEBP)["thumbs"]["mesh"]
    assert again["path"] == "thumbs/mesh.webp" and again["v"] > first["v"]
    assert sorted(p.name for p in thumbs.iterdir()) == ["mesh.webp"]
    for stage in ("images", "retopo", "projection", "rig"):
        store.set_thumb("P", pid, stage, PNG)
    for stage, data, status in (("nope", PNG, 404), ("mesh", JPG, 415), ("mesh", b"", 400), ("mesh", b"GIF89a", 415)):
        with pytest.raises(ProjectError) as e:
            store.set_thumb("P", pid, stage, data)
        assert e.value.status_code == status, stage
    monkeypatch.setattr("spellcast.projects.MAX_THUMB_BYTES", 10)
    with pytest.raises(ProjectError) as e:
        store.set_thumb("P", pid, "mesh", PNG)
    assert e.value.status_code == 413
    monkeypatch.undo()

    # A new result makes its stage's snapshot out of date: the page renders it again.
    store.apply_stage_patch("P", pid, "mesh", {"glb": "mesh/mesh.glb"})
    store.apply_stage_patch("P", pid, "retopo", {"stats": {"vertices": 5}})    # not a new low poly
    assert set(store.get_prop("P", pid)["thumbs"]) == {"images", "retopo", "projection", "rig"}
    store.apply_stage_patch("P", pid, "retopo", {"obj": "retopo/retopo.obj"})
    store.apply_stage_patch("P", pid, "projection", {"glb": "projection/bake_001/projection.glb"})
    prop = store.add_image("P", pid, PNG)
    assert set(prop["thumbs"]) == {"rig"}
    store.set_thumb("P", pid, "images", PNG, of="images/upload_001.png")
    store.add_image("P", pid, JPG)
    store.set_thumb("P", pid, "images", PNG, of="images/upload_002.jpg")
    assert "images" in store.select_image("P", pid, "images/upload_002.jpg")["thumbs"]    # already chosen
    assert "images" not in store.select_image("P", pid, "images/upload_001.png")["thumbs"]


def test_a_picture_thumbnail_is_kept_only_while_the_prop_shows_what_it_was_drawn_from(tmp_path: Path) -> None:
    """thumb_signature: the chosen picture and the part of it. A thumbnail of anything else is never stored, and
    one that is stored is dropped as soon as the prop shows something else - whoever changed it."""
    assert thumb_signature({}) is None and thumb_signature({"chosen": None, "views": []}) is None
    assert thumb_signature({"chosen": "images/0002.png"}) == "images/0002.png"
    sheet = {"chosen": "images/0002.png", "views": [
        {"view": "left", "src": "images/0002.png", "rect": [512, 0, 512, 512]},
        {"view": "front", "src": "images/0002.png", "rect": [0, 0, 512, 512]}]}
    assert thumb_signature(sheet) == "images/0002.png#0,0,512,512"
    assert thumb_signature({**sheet, "chosen": "images/0001.png"}) == "images/0001.png"   # the views wait aside

    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    pid = store.add_prop("P", "Creature")["id"]
    thumbs = store.prop_dir("P", pid) / "thumbs"
    first = store.add_image("P", pid, PNG)["images"]["chosen"]
    assert store.set_thumb("P", pid, "images", WEBP, of=first)["thumbs"]["images"]["of"] == first

    # A generation finishes: the hub chooses its first new picture. The thumbnail of the picture before is gone,
    # and one of it that arrives late (drawn before the page knew) is not stored in its place.
    prop = store.apply_stage_patch("P", pid, "images", {"candidates": [first, "images/0002.png"],
                                                           "chosen": "images/0002.png"})
    assert "images" not in prop["thumbs"]
    (thumbs / "images.webp").unlink()
    late = store.set_thumb("P", pid, "images", WEBP, of=first)
    assert "images" not in late["thumbs"] and not (thumbs / "images.webp").exists()
    assert "images" not in store.set_thumb("P", pid, "images", WEBP)["thumbs"]    # it must say what it shows
    kept = store.set_thumb("P", pid, "images", WEBP, of="images/0002.png")["thumbs"]["images"]
    assert kept["of"] == "images/0002.png" and "crop" not in kept

    # The picture is sliced into views: its thumbnail shows the Front slice from then on.
    views = [{"view": "front", "src": "images/0002.png", "rect": [0, 0, 512, 512]},
             {"view": "back", "src": "images/0002.png", "rect": [512, 0, 512, 512]}]
    assert "images" not in store.set_views("P", pid, views)["thumbs"]
    assert "images" not in store.set_thumb("P", pid, "images", WEBP, of="images/0002.png")["thumbs"]    # all of it
    sliced = store.set_thumb("P", pid, "images", WEBP, of="images/0002.png#0,0,512,512")["thumbs"]["images"]
    assert sliced["of"] == "images/0002.png#0,0,512,512" and sliced["crop"] == [0, 0, 512, 512]
    views[1]["rect"] = [512, 0, 256, 512]    # another slice moves: the Front one is the same
    assert "images" in store.set_views("P", pid, views)["thumbs"]
    views[0]["rect"] = [16, 0, 512, 512]
    assert "images" not in store.set_views("P", pid, views)["thumbs"]

    # Whatever wrote project.json (an older hub, a change that forgot the thumbnail): a thumbnail that is not of
    # what the prop shows - or does not say - is not given out, and not saved again.
    store.set_thumb("P", pid, "images", WEBP, of="images/0002.png#16,0,512,512")
    file = store.project_dir("P") / "project.json"
    for stale in ({"of": first}, {"of": "images/0002.png"}, {}):
        data = json.loads(file.read_text(encoding="utf-8"))
        data["props"][0]["thumbs"]["images"] = {"path": "thumbs/images.webp", "v": 5, **stale}
        data["props"][0]["thumbs"]["mesh"] = {"path": "thumbs/mesh.webp", "v": 5}
        file.write_text(json.dumps(data), encoding="utf-8")
        assert set(store.get_prop("P", pid)["thumbs"]) == {"mesh"}, stale
    data["props"][0]["thumbs"]["images"] = {"path": "thumbs/images.webp", "v": 5, "of": "images/0002.png#16,0,512,512"}
    file.write_text(json.dumps(data), encoding="utf-8")
    assert set(store.get_prop("P", pid)["thumbs"]) == {"images", "mesh"}
    store.patch_prop("P", pid, {"name": "Werewolf"})
    assert set(json.loads(file.read_text(encoding="utf-8"))["props"][0]["thumbs"]) == {"images", "mesh"}


# ---- HTTP API ------------------------------------------------------------------------------------

@pytest.fixture
def hub(tmp_path: Path) -> Iterator[tuple[FastAPI, TestClient]]:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>Spellcast3D</h1>", encoding="utf-8")
    adapters = {"forge": FakeAdapter("forge", "images", True, tmp_path),
                "pixal3d": FakeAdapter("pixal3d", "mesh", True, tmp_path),
                paths.IMESH: FakeImesh(tmp_path, "--bake-seconds", "0.5")}
    shown: list[tuple[Path, bool]] = []
    converter = FakeConverter()
    app = create_app(adapters, settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=static, logs_dir=tmp_path / "logs", gpu_probe=lambda: None,
                     show_in_explorer=lambda path, select: shown.append((path, select)) is None,
                     converter=converter)
    app.state.shown, app.state.converter = shown, converter
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield app, client


def test_api_untitled_start_rename_and_busy_projects(hub: Any, tmp_path: Path) -> None:
    app, client = hub
    root = tmp_path / "projects"
    assert client.get("/api/projects").json() == []
    started = client.post("/api/projects/open-last").json()
    assert started["untitled"] and started["name"] == "Untitled" and UNTITLED_ID.match(started["id"])
    assert client.post("/api/projects/open-last").json()["id"] == started["id"]
    assert client.post("/api/projects").json()["id"] == started["id"]    # no body: new untitled - the open one is
    assert client.post("/api/projects", json=[1]).status_code == 422
    named = client.post("/api/projects", json={"name": "Props"}).json()
    assert named["id"] == "Props" and _state(root) == {"last_opened": "Props"}     # a reload comes back to it
    assert len(_trashed(root)) == 1                             # the empty "Untitled" was left: tidied away
    another = client.post("/api/projects", json={}).json()
    assert another["name"] == "Untitled" and _state(root) == {"last_opened": another["id"]}
    listed = client.get("/api/projects").json()
    assert set(listed[0]) == {"id", "name", "untitled", "prop_count", "updated"}
    assert client.get(f"/api/projects/{another['id']}").json()["id"] == another["id"]

    project = another["id"]
    draft = client.post(f"/api/projects/{project}/draft").json()
    assert draft["draft"] is True
    job = client.post("/api/jobs", json={"stage": "images", "project": project, "prop_ids": [draft["id"]],
                                         "params": {"mode": "slow"}}).json()
    busy = client.post(f"/api/projects/{project}/rename", json={"name": "Dungeon"})
    assert busy.status_code == 409 and "still working on this project" in busy.json()["detail"]
    wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json()["status"], lambda s: s in FINISHED)
    renamed = client.post(f"/api/projects/{project}/rename", json={"name": "Dungeon"}).json()
    assert (renamed["id"], renamed["name"], renamed["untitled"]) == ("Dungeon", "Dungeon", False)
    assert renamed["props"][0]["images"]["result"] == "images/out.txt"     # the job's result moved along
    assert client.get(f"/api/projects/{project}").status_code == 404
    assert client.post("/api/projects/Dungeon/rename", json={"name": "???"}).status_code == 400
    assert client.post("/api/projects/Dungeon/rename", json={}).status_code == 400


def test_api_leftover_work_folders_go_when_the_hub_starts_and_when_a_project_is_opened(tmp_path: Path) -> None:
    """What a job could not remove (the hub was closed, a file was still open): swept when the hub starts, and
    when its project is opened. Never the folder of a job that runs, and never anything else of the project."""
    root = tmp_path / "projects"
    store = ProjectStore(root)
    folders = {name: _project_with_everything(store, name) for name in ("Props", "Other")}
    before = {name: _content(folder) for name, folder in folders.items()}
    for name in folders:
        (store.work_dir(name, "mesh-j_aaaa1111") / "latents.bin").write_bytes(b"x" * 100)
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>Spellcast3D</h1>", encoding="utf-8")
    adapters = {"forge": FakeAdapter("forge", "images", True, tmp_path),
                "imesh": FakeAdapter("imesh", "retopo", False, tmp_path)}
    app = create_app(adapters, settings=make_settings(tmp_path), store=store, static_dir=static,
                     logs_dir=tmp_path / "logs", gpu_probe=lambda: None)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert not any((folder / "temp").exists() for folder in folders.values())          # the start's sweep
        for name in folders:
            (store.work_dir(name, "mesh-j_bbbb2222") / "latents.bin").write_bytes(b"x" * 100)
        assert client.get("/files/Props/temp/mesh-j_bbbb2222/latents.bin").status_code == 404
        assert client.get("/files/Props/notes.txt").status_code == 200
        assert client.get("/api/projects/Props").json()["id"] == "Props"
        assert not (folders["Props"] / "temp").exists() and (folders["Other"] / "temp").is_dir()
        assert client.post("/api/projects/open-last").json()["id"] == "Props"              # opened last: nothing left
        # The folder of a job that runs in the project stays (the queue names it), the others go.
        queue = app.state.hub.jobs
        queue._jobs["j_cccc3333"] = Job(id="j_cccc3333", stage="mesh", tool="pixal3d", project="Other", prop_ids=[],
                                        params={}, status=RUNNING, work="mesh-j_cccc3333")
        (store.work_dir("Other", "mesh-j_cccc3333") / "latents.bin").write_bytes(b"x" * 100)
        assert queue.work_dir_in_use("Other") == "mesh-j_cccc3333" and queue.work_dir_in_use("Props") is None
        assert client.get("/api/projects/Other").json()["id"] == "Other"
        assert [p.name for p in (folders["Other"] / "temp").iterdir()] == ["mesh-j_cccc3333"]
        del queue._jobs["j_cccc3333"]
        assert client.get("/api/projects/Other").status_code == 200 and not (folders["Other"] / "temp").exists()
    for name, folder in folders.items():
        assert _content(folder) == before[name]


def test_half_written_models_go_when_the_hub_starts(tmp_path: Path) -> None:
    """A hub closed (or crashed) in the middle of a 3D job leaves the model Pixal3D was writing beside the prop's
    mesh.glb (props/<prop>/mesh/mesh.<name>.partial.glb): swept when the hub starts, and nothing else."""
    root = tmp_path / "projects"
    store = ProjectStore(root)
    folders = {name: _project_with_everything(store, name) for name in ("Props", "Other")}
    half = []
    for folder in folders.values():
        mesh = next((folder / "props").iterdir()) / "mesh"
        mesh.mkdir()
        (mesh / "mesh.glb").write_bytes(b"glTF-kept")
        for name in ("mesh.1a2b3c4d.partial.glb", "mesh.check.partial.glb"):
            (mesh / name).write_bytes(b"glTF-half")
            half.append(mesh / name)
    # A deleted prop's folder and the trash are the artist's to look into: left as they are.
    kept = next((folders["Props"] / "deleted").iterdir()) / "mesh" / "mesh.5e6f7a8b.partial.glb"
    kept.parent.mkdir()
    kept.write_bytes(b"glTF-half")
    (root / ".trash" / "old" / "props" / "p_x" / "mesh").mkdir(parents=True)
    (root / ".trash" / "old" / "props" / "p_x" / "mesh" / "mesh.9a9a9a9a.partial.glb").write_bytes(b"glTF-half")
    before = {name: {rel: data for rel, data in _content(folder).items() if not rel.endswith(".partial.glb")
                     or rel.startswith("deleted/")} for name, folder in folders.items()}

    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>Spellcast3D</h1>", encoding="utf-8")
    app = create_app({"forge": FakeAdapter("forge", "images", True, tmp_path)}, settings=make_settings(tmp_path),
                     store=store, static_dir=static, logs_dir=tmp_path / "logs", gpu_probe=lambda: None)
    with TestClient(app, base_url="http://127.0.0.1"):
        assert not any(path.exists() for path in half)
    for name, folder in folders.items():
        assert _content(folder) == before[name]
    assert kept.is_file()
    assert (root / ".trash" / "old" / "props" / "p_x" / "mesh" / "mesh.9a9a9a9a.partial.glb").is_file()
    assert store.clear_partial_meshes() == 0 and ProjectStore(tmp_path / "no projects yet").clear_partial_meshes() == 0


def test_api_draft_prop_and_thumbnails(hub: Any) -> None:
    app, client = hub
    client.post("/api/projects", json={"name": "Demo"})
    draft = client.post("/api/projects/Demo/draft").json()
    url = f"/api/projects/Demo/props/{draft['id']}"
    refused = client.post(f"{url}/pack")
    assert refused.status_code == 409 and refused.json()["detail"] == NO_IMAGE_YET
    uploaded = client.post(f"{url}/images/upload", files={"file": ("C:\\art\\small_red-dragon.png", PNG, "image/png")})
    assert uploaded.json()["draft"] is False and uploaded.json()["name"] == "Small red dragon"   # the file's name
    assert client.post(f"{url}/pack").json() == uploaded.json()
    assert client.post("/api/projects/Demo/draft").json()["id"] != draft["id"]
    assert client.post("/api/projects/Demo/props/p_ffffff/pack").status_code == 404

    # "+ New prop" while the draft's first pictures are being made: it becomes a prop, a new draft comes.
    generating = client.post("/api/projects/Demo/draft").json()
    client.patch(f"/api/projects/Demo/props/{generating['id']}",
                 json={"images": {"settings": {"prompt": "a copper kettle"}}})
    job = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [generating["id"]],
                                         "params": {"mode": "slow"}}).json()
    fresh = client.post("/api/projects/Demo/draft").json()
    assert fresh["id"] != generating["id"] and fresh["draft"] is True
    assert client.get("/api/projects/Demo").json()["props"][1]["name"] == "Copper kettle"
    wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json()["status"], lambda s: s in FINISHED)
    assert client.post("/api/projects/Demo/draft").json()["id"] == fresh["id"]

    put = client.put(f"{url}/thumbs/mesh", content=PNG, headers={"content-type": "image/png"})
    assert put.status_code == 200 and put.json()["thumbs"]["mesh"]["path"] == "thumbs/mesh.png"
    # A picture's thumbnail says what it shows (?of=: the chosen picture, and the part of it when not all). One that
    # does not say, or that shows another picture, is not kept: the prop comes back as it is.
    chosen = uploaded.json()["images"]["chosen"]
    assert "images" not in client.put(f"{url}/thumbs/images?crop=0,0,64,64", content=PNG).json()["thumbs"]
    mine = client.put(f"{url}/thumbs/images", params={"of": chosen}, content=PNG, headers={"content-type": "image/png"})
    assert mine.status_code == 200 and mine.json()["thumbs"]["images"]["of"] == chosen
    assert "crop" not in mine.json()["thumbs"]["images"]
    late = client.put(f"{url}/thumbs/images", params={"of": "images/0099.png"}, content=WEBP)
    assert late.status_code == 200 and late.json()["thumbs"]["images"] == mine.json()["thumbs"]["images"]
    assert client.get(f"/files/Demo/props/{draft['id']}/thumbs/images.png").content == PNG
    thumb = client.get(f"/files/Demo/props/{draft['id']}/thumbs/mesh.png")
    assert thumb.content == PNG and thumb.headers["content-type"] == "image/png"
    webp = client.put(f"{url}/thumbs/retopo", content=WEBP, headers={"content-type": "image/webp"}).json()
    assert webp["thumbs"]["retopo"]["path"] == "thumbs/retopo.webp"
    assert client.get(f"/files/Demo/props/{draft['id']}/thumbs/retopo.webp").headers["content-type"] == "image/webp"
    too_big = client.put(f"{url}/thumbs/mesh", content=PNG + b"\0" * (2 * 1024 * 1024))
    assert too_big.status_code == 413 and "2 MB" in too_big.json()["detail"]
    assert client.put(f"{url}/thumbs/nope", content=PNG).status_code == 404
    assert client.put(f"{url}/thumbs/mesh", content=JPG).status_code == 415
    assert client.put("/api/projects/Demo/props/p_ffffff/thumbs/mesh", content=PNG).status_code == 404


def test_api_import_a_model_as_a_new_prop(hub: Any) -> None:
    app, client = hub
    store = app.state.hub.store
    client.post("/api/projects", json={"name": "Demo"})
    first = client.post("/api/projects/Demo/props", json={"name": "Barrel"}).json()
    client.patch(f"/api/projects/Demo/props/{first['id']}", json={"mesh": {"settings": {"resolution": 1536}}})
    draft = client.post("/api/projects/Demo/draft").json()

    files = [("files[]", ("Chair_01.obj", b"mtllib chair.mtl\nv 0 0 0\n", "text/plain")),
             ("files[]", ("chair.mtl", b"map_Kd chair_d.png\n", "text/plain")),
             ("files[]", ("chair_d.png", PNG, "image/png"))]
    answer = client.post("/api/projects/Demo/props/import-model", files=files)
    assert answer.status_code == 200, answer.text
    prop = answer.json()
    assert prop["name"] == "Chair 01" and prop["draft"] is False and prop["images"]["chosen"] is None
    assert prop["mesh"]["glb"] == "mesh/imported_001.glb" and prop["mesh"]["source"] == "mesh/import_001/Chair_01.obj"
    assert prop["mesh"]["imported"]["materials"][0]["maps"][0]["file"] == "chair_d.png"
    assert prop["mesh"]["settings"]["resolution"] == 1536                 # the last prop's settings
    assert [p["id"] for p in client.get("/api/projects/Demo").json()["props"]] == [first["id"], draft["id"], prop["id"]]

    zipped = client.post("/api/projects/Demo/props/import-model",
                         files=[("files", ("lamp.zip", make_zip({"Lamp/lamp.fbx": b"fbx"}), "application/zip"))])
    assert zipped.status_code == 200 and zipped.json()["name"] == "Lamp"

    before = sorted(p.name for p in (store.project_dir("Demo") / "props").iterdir())
    for bad_files, status in (([("files", ("wood.png", PNG, "image/png"))], 400),
                              ([("files", ("bad.fbx", b"FAKE_ERROR:This FBX has no mesh in it."))], 422)):
        failed = client.post("/api/projects/Demo/props/import-model", files=bad_files)
        assert failed.status_code == status, failed.text
    assert client.post("/api/projects/Demo/props/import-model", data={"x": "1"}).status_code == 400
    assert len(client.get("/api/projects/Demo").json()["props"]) == 4          # no half-made props
    assert sorted(p.name for p in (store.project_dir("Demo") / "props").iterdir()) == before
    assert client.post("/api/projects/Nope/props/import-model", files=files).status_code == 404


# ---- saving models into exports/<project>/ -------------------------------------------------------

def test_file_names_are_the_prop_names_and_stay_put() -> None:
    props = [{"id": f"p_{i:06d}", "name": name, "draft": draft} for i, (name, draft) in enumerate((
        ("Wooden barrel", False), ("wooden  BARREL", False), ("Barrel high", False), ("", True),
        ("Wooden barrel high", False), ("???", False), ("Épée d'or (v2)", False), ("Wooden barrel", False)))]
    assert exports.file_stems(props) == {
        "p_000000": "Wooden_barrel", "p_000001": "wooden_BARREL_2", "p_000002": "Barrel_high",
        "p_000004": "Wooden_barrel_high_2",        # not the first barrel's 3D model's name
        "p_000005": "p_000005", "p_000006": "Épée_d'or_(v2)", "p_000007": "Wooden_barrel_3"}
    # A prop added at the end changes no earlier name.
    more = exports.file_stems([*props, {"id": "p_000099", "name": "Wooden barrel"}])
    assert more["p_000099"] == "Wooden_barrel_4" and all(more[k] == v for k, v in exports.file_stems(props).items())


def _local(app: FastAPI) -> TestClient:
    """A client the hub takes for the page on this computer (File Explorer may open for it)."""
    return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))


def test_api_save_one_file_and_show_it(hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, client = hub
    store, converter, shown = app.state.hub.store, app.state.converter, app.state.shown
    client.post("/api/projects", json={"name": "Demo"})
    pid = client.post("/api/projects/Demo/props", json={"name": "Wooden barrel"}).json()["id"]
    url = f"/api/projects/Demo/props/{pid}/export"
    no_mesh = client.post(url, json={"stage": "mesh", "format": "glb"})
    assert no_mesh.status_code == 409 and "no 3D model yet" in no_mesh.json()["detail"]
    glb = store.prop_dir("Demo", pid) / "mesh" / "mesh.glb"
    glb.parent.mkdir(parents=True)
    glb.write_bytes(b"glTF-high")
    store.apply_stage_patch("Demo", pid, "mesh", {"glb": "mesh/mesh.glb"})

    monkeypatch.delenv("SPELLCAST_NO_EXPLORER")
    project = tmp_path / "exports" / "Demo"
    folder = project / "Wooden_barrel"                          # the prop's own folder
    project.mkdir(parents=True, exist_ok=True)
    (project / "Wooden_barrel_high.glb").write_bytes(b"saved before props had their own folders")
    (project / "notes.txt").write_bytes(b"the artist's own")
    answer = _local(app).post(url, json={"stage": "mesh", "format": "glb"}).json()
    assert answer == {"folder": "exports\\Demo\\Wooden_barrel", "folder_path": str(folder),
                      "files": ["Wooden_barrel_high.glb"], "shown": "Wooden_barrel_high.glb", "opened": True,
                      "warnings": []}
    assert shown == [(folder / "Wooden_barrel_high.glb", True)] and converter.calls == []
    assert (folder / "Wooden_barrel_high.glb").read_bytes() == b"glTF-high"
    assert sorted(p.name for p in project.iterdir()) == ["Wooden_barrel", "notes.txt"]    # the old copy is gone
    assert client.post(url, json={"stage": "mesh", "format": "glb"}).json()["opened"] is False   # not this PC
    monkeypatch.setenv("SPELLCAST_NO_EXPLORER", "1")
    assert _local(app).post(url, json={"stage": "mesh", "format": "glb"}).json()["opened"] is False
    assert len(shown) == 1

    # FBX and OBJ are converted from the GLB (once: the conversion is kept until the GLB changes).
    fbx = client.post(url, json={"stage": "mesh", "format": "fbx"}).json()
    assert fbx["files"] == ["Wooden_barrel_high.fbx"] and fbx["shown"] == "Wooden_barrel_high.fbx"
    assert (folder / "Wooden_barrel_high.fbx").read_bytes() == b"Kaydara FBX Binary glTF-high"
    obj = client.post(url, json={"stage": "mesh", "format": "obj"}).json()
    assert obj["files"] == ["Wooden_barrel_high.obj", "Wooden_barrel_high.mtl", "Wooden_barrel_high_basecolor.png"]
    assert converter.calls == [(glb, ["fbx"], "Wooden_barrel_high"), (glb, ["obj"], "Wooden_barrel_high")]
    (folder / "Wooden_barrel_high.fbx").write_bytes(b"edited by hand")
    client.post(url, json={"stage": "mesh", "format": "fbx"})
    assert len(converter.calls) == 2                            # from the cache...
    assert (folder / "Wooden_barrel_high.fbx").read_bytes() == b"Kaydara FBX Binary glTF-high"   # ...overwritten
    glb.write_bytes(b"glTF-high, made again")
    client.post(url, json={"stage": "mesh", "format": "fbx"})
    assert len(converter.calls) == 3 and (folder / "Wooden_barrel_high.fbx").read_bytes().endswith(b"made again")
    client.patch(f"/api/projects/Demo/props/{pid}", json={"name": "Oak barrel"})
    assert client.post(url, json={"stage": "mesh", "format": "fbx"}).json()["shown"] == "Oak_barrel_high.fbx"
    assert converter.calls[-1][2] == "Oak_barrel_high"          # the name is inside the FBX too

    oak = project / "Oak_barrel"                                # a renamed prop: a new folder, the old one stays
    with open(oak / "Oak_barrel_high.fbx", "rb"):               # open in a game engine, say
        locked = client.post(url, json={"stage": "mesh", "format": "fbx"})
    assert locked.status_code == 409 and "is open in another program" in locked.json()["detail"]
    assert not [p for p in oak.iterdir() if p.name.endswith(".tmp")]

    # Without the Instant Meshes venv there is no FBX or OBJ, but the GLB can still be saved.
    imesh = app.state.hub.adapters[paths.IMESH]
    monkeypatch.setattr(imesh, "install_status", lambda: {"status": "not_installed", "reason": "venv missing"})
    glb.write_bytes(b"glTF-high, third")
    missing = client.post(url, json={"stage": "mesh", "format": "obj"})
    assert missing.status_code == 409 and missing.json()["detail"].startswith("Saving as .obj needs the retopology tool.")
    assert client.post(url, json={"stage": "mesh", "format": "glb"}).status_code == 200

    draft = client.post("/api/projects/Demo/draft").json()
    assert client.post(f"/api/projects/Demo/props/{draft['id']}/export",
                       json={"stage": "mesh", "format": "glb"}).status_code == 409
    assert client.post("/api/projects/Demo/props/p_ffffff/export", json={"stage": "mesh", "format": "glb"}) \
        .status_code == 404

    # The saved models follow their project to its new name.
    renamed = client.post("/api/projects/Demo/rename", json={"name": "Tavern props"})
    assert renamed.status_code == 200
    assert not project.exists()
    assert (tmp_path / "exports" / "Tavern props" / "Oak_barrel" / "Oak_barrel_high.glb").is_file()


def test_api_save_every_prop(hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, client = hub
    store, converter, shown = app.state.hub.store, app.state.converter, app.state.shown
    client.post("/api/projects", json={"name": "Demo"})
    nothing = client.post("/api/projects/Demo/export-all")
    assert nothing.status_code == 404 and "nothing to save yet" in nothing.json()["detail"]
    ids = [client.post("/api/projects/Demo/props", json={"name": name}).json()["id"]
           for name in ("Barrel", "Barrel", "Crate", "Lamp", "Sketch")]
    for pid in ids[:3]:
        ready_to_bake(store, "Demo", pid)
    # The second barrel is baked without FBX and OBJ: they are converted from its GLB.
    client.patch(f"/api/projects/Demo/props/{ids[1]}",
                 json={"projection": {"settings": {"export_fbx": False, "export_obj": False}}})
    job = client.post("/api/jobs", json={"stage": "projection", "project": "Demo", "prop_ids": ids[:2]}).json()
    done = wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json(), lambda j: j["status"] in FINISHED)
    assert done["status"] == "done", done
    lamp = store.prop_dir("Demo", ids[3])
    (lamp / "mesh").mkdir(parents=True)
    (lamp / "mesh" / "imported_002.glb").write_bytes(b"glTF-lamp")
    store.apply_stage_patch("Demo", ids[3], "mesh", {"glb": "mesh/imported_002.glb"})
    store.add_image("Demo", ids[4], PNG)                        # a picture only: nothing to save
    draft = client.post("/api/projects/Demo/draft").json()
    ready_to_bake(store, "Demo", draft["id"])                   # not a prop yet: left out

    monkeypatch.delenv("SPELLCAST_NO_EXPLORER")
    folder = tmp_path / "exports" / "Demo"
    answer = _local(app).post("/api/projects/Demo/export-all").json()
    # Each prop in its own folder (the two barrels: Barrel and Barrel_2), its files flat in it.
    baked = [f"Barrel\\{name}" for name in ("Barrel.glb", "Barrel.fbx", "Barrel.obj", "Barrel.mtl",
                                            "Barrel_basecolor.png", "Barrel_normal.png", "Barrel_orm.png")]
    converted = [f"Barrel_2\\{name}" for name in ("Barrel_2.glb", "Barrel_2.fbx", "Barrel_2.obj", "Barrel_2.mtl",
                                                  "Barrel_2_basecolor.png")]
    high = [f"Lamp\\{name}" for name in ("Lamp_high.glb", "Lamp_high.fbx", "Lamp_high.obj", "Lamp_high.mtl",
                                         "Lamp_high_basecolor.png")]
    assert answer == {"folder": "exports\\Demo", "folder_path": str(folder), "props": 4, "opened": True,
                      "files": [*baked, *converted, "Crate\\Crate_retopo.obj", *high], "warnings": []}
    assert sorted(p.name for p in folder.iterdir()) == ["Barrel", "Barrel_2", "Crate", "Lamp"]
    assert sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()) == sorted(answer["files"])
    assert shown == [(folder, False)]
    assert (folder / "Barrel" / "Barrel.mtl").read_text(encoding="utf-8") == \
        "newmtl projection\nmap_Kd Barrel_basecolor.png\n"
    assert (folder / "Lamp" / "Lamp_high.glb").read_bytes() == b"glTF-lamp"
    assert (folder / "Crate" / "Crate_retopo.obj").read_text(encoding="utf-8") == "v 0 0 0\n"
    baked_glb = store.prop_dir("Demo", ids[1]) / store.get_prop("Demo", ids[1])["projection"]["glb"]
    assert converter.calls == [(baked_glb, ["fbx", "obj"], "Barrel_2"), (lamp / "mesh" / "imported_002.glb",
                                                                       ["fbx", "obj"], "Lamp_high")]
    client.post("/api/projects/Demo/export-all")
    assert len(converter.calls) == 2                            # the conversions are kept

    # A bake whose files are gone falls back to the retopo; a conversion that fails leaves the rest saved.
    baked_glb.unlink()

    def broken(*_: Any) -> dict[str, Any]:
        raise ProjectError("The 3D model imported_002.glb could not be read: no mesh.", 422)

    monkeypatch.setattr(app.state.hub.exporter, "convert", broken)
    (lamp / "mesh" / "imported_002.glb").write_bytes(b"glTF-lamp, damaged")
    partly = client.post("/api/projects/Demo/export-all").json()
    assert "Barrel_2\\Barrel_2_retopo.obj" in partly["files"] and "Lamp\\Lamp_high.glb" in partly["files"]
    assert partly["props"] == 4 and partly["warnings"] == ["Lamp: The 3D model imported_002.glb could not be read: no mesh."]


def test_a_picture_leaves_the_prop_and_comes_back(hub: Any) -> None:
    """remove_image takes a picture out of images.candidates and out of the views (its file stays);
    restore_image puts it back at its place with the views it had, chosen again. The thumbnail never stays
    one of a picture that left. The routes images/remove and images/restore are the page's × and Ctrl+Z."""
    app, client = hub
    store = app.state.hub.store
    store.create_project("P")
    pid = store.add_prop("P", "Crate")["id"]
    a, b, c = (store.add_image("P", pid, PNG)["images"]["chosen"] for _ in range(3))
    views = [{"view": "front", "src": b, "rect": [0, 0, 32, 32]}, {"view": "left", "src": a, "rect": None},
             {"view": "back", "src": b, "rect": [32, 0, 32, 32]}]
    mine = [v for v in views if v["src"] == b]
    store.set_views("P", pid, views)
    store.select_image("P", pid, b)
    store.set_thumb("P", pid, "images", PNG, of=thumb_signature(store.get_prop("P", pid)["images"]))
    assert "images" in store.get_prop("P", pid)["thumbs"]

    # not the chosen one: the chosen picture and its thumbnail stay
    prop = store.remove_image("P", pid, c)
    assert prop["images"]["candidates"] == [a, b] and prop["images"]["chosen"] == b and "images" in prop["thumbs"]
    assert store.remove_image("P", pid, c)["images"] == prop["images"]     # not one of its pictures any more
    assert store.restore_image("P", pid, c, 2)["images"]["candidates"] == [a, b, c]

    # the chosen one: the picture made before it is chosen, its views and the thumbnail go, its file stays
    api = f"/api/projects/P/props/{pid}/images"
    prop = client.post(f"{api}/remove", json={"path": b}).json()
    images = prop["images"]
    assert images["candidates"] == [a, c] and images["chosen"] == a and "images" not in prop["thumbs"]
    assert images["views"] == [{"view": "left", "src": a, "rect": None}]
    assert (store.prop_dir("P", pid) / b).is_file()

    # back: at its place, with its views, chosen again
    images = client.post(f"{api}/restore", json={"path": b, "index": 1, "views": mine, "chosen": True}).json()["images"]
    assert images["candidates"] == [a, b, c] and images["chosen"] == b and images["views"] == views
    # a side another picture has by now stays that picture's; a place beyond the list is its end
    store.remove_image("P", pid, b)
    store.set_views("P", pid, [{"view": "front", "src": c, "rect": None}])
    images = store.restore_image("P", pid, b, 99, mine, False)["images"]
    assert images["candidates"] == [a, c, b] and images["chosen"] == a
    assert [(v["view"], v["src"]) for v in images["views"]] == [("front", c), ("back", b)]

    # the first one hands over to the one after it, the last one to none
    store.select_image("P", pid, a)
    assert store.remove_image("P", pid, a)["images"]["chosen"] == c
    store.remove_image("P", pid, c)
    prop = store.remove_image("P", pid, b)
    assert prop["images"]["candidates"] == [] and prop["images"]["chosen"] is None and prop["draft"] is False
    # only a picture still in the prop's images folder comes back
    for bad in ("images/nope.png", "../project.json", "images/../project.json", "reference/reference_001.png", None, 5):
        with pytest.raises(ProjectError, match="cannot be brought back"):
            store.restore_image("P", pid, bad)
    assert client.post(f"{api}/restore", json={"path": "images/nope.png"}).status_code == 400
    assert client.post(f"{api}/restore", json={"path": a, "chosen": True}).json()["images"]["chosen"] == a


def test_3d_ready_framing_is_no_setting_any_more(tmp_path: Path) -> None:
    """object_mode ("3D-ready framing") is retired like qwen_cfg: a new prop has none, a project's is dropped when
    it is read, and one a page still sends is left out."""
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    pid = store.add_prop("P", "Crate")["id"]
    assert "object_mode" not in store.get_prop("P", pid)["images"]["settings"]
    file = store.root / "P" / "project.json"
    project = json.loads(file.read_text(encoding="utf-8"))
    project["props"][0]["images"]["settings"]["object_mode"] = True
    file.write_text(json.dumps(project), encoding="utf-8")
    assert "object_mode" not in store.get_prop("P", pid)["images"]["settings"]
    patched = store.patch_prop("P", pid, {"images": {"settings": {"object_mode": False, "prompt": "a crate"}}})
    assert "object_mode" not in patched["images"]["settings"] and patched["images"]["settings"]["prompt"] == "a crate"


def test_a_folder_of_several_props_pictures_becomes_props(tmp_path: Path) -> None:
    """propfiles.import_prop_folder (the chooser's "Import folder", read by the hub): one prop per name, the sides as
    the names say with the front one chosen, a single picture a prop of its own, bad files and extras said."""
    from spellcast import propfiles

    store = ProjectStore(tmp_path / "projects")
    store.create_project("Demo")
    folder = tmp_path / "art"
    (folder / "sub").mkdir(parents=True)
    png = b"\x89PNG\r\n\x1a\n" + b"\0" * 64
    for name in ("knight_front.png", "knight_back.png", "dragon_front.png", "sub/dragon_left.png", "barrel.png",
                 "notes.txt"):
        (folder / name).write_bytes(png if name.endswith(".png") else b"text")
    (folder / "broken_left.png").write_bytes(b"not a picture")
    (folder / "empty_front.png").write_bytes(b"")
    made = propfiles.import_prop_folder(store, "Demo", folder)
    assert [(p["name"], p["sides"]) for p in made["props"]] == [
        ("Barrel", []), ("Dragon", ["front", "left"]), ("Knight", ["back", "front"])]   # files in name order
    assert sorted(made["skipped"]) == ["broken_left.png: not a PNG, JPG or WebP picture", "empty_front.png: empty"]
    knight = store.get_prop("Demo", made["props"][2]["id"])
    assert [v["view"] for v in knight["images"]["views"]] == ["front", "back"]   # views in their own order
    assert knight["images"]["chosen"] == next(v["src"] for v in knight["images"]["views"] if v["view"] == "front")
    barrel = store.get_prop("Demo", made["props"][0]["id"])
    assert barrel["images"]["chosen"] and not barrel["images"].get("views")
    with pytest.raises(ProjectError):
        propfiles.import_prop_folder(store, "Demo", tmp_path / "nowhere")
