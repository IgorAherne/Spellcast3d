"""spellcast/exports.py pieces the route tests (test_projects_v2.py, test_projection_hub.py) do not reach:
the real export worker run from the hub (a conversion and a painting), the File Explorer command line, the flat
OBJ rewrite, and the folders of props that share a name.

Runs in the hub venv. The worker tests need the Instant Meshes venv (code/venvs/imesh) and skip without it.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest

from spellcast import exports, paths
from spellcast.projects import ProjectError, ProjectStore
from test_projection_hub import FakeConverter, tiny_png

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Spellcast3D is Windows-only")


def _triangle_glb(path: Path) -> None:
    """A GLB with one triangle and no material, written without numpy."""
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    indices = struct.pack("<3I", 0, 1, 2)
    document = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
                "nodes": [{"mesh": 0}], "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
                "buffers": [{"byteLength": 48}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36},
                                {"buffer": 0, "byteOffset": 36, "byteLength": 12}],
                "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                               "min": [0, 0, 0], "max": [1, 1, 0]},
                              {"bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR"}]}
    text = json.dumps(document).encode("utf-8")
    text += b" " * (-len(text) % 4)
    binary = positions + indices
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 28 + len(text) + len(binary))
                     + struct.pack("<II", len(text), 0x4E4F534A) + text
                     + struct.pack("<II", len(binary), 0x004E4942) + binary)


@pytest.mark.skipif(not paths.venv_python(paths.IMESH).is_file(), reason="the Instant Meshes venv is not installed")
def test_the_hub_runs_the_real_export_worker(tmp_path: Path) -> None:
    src = tmp_path / "prop" / "mesh" / "mesh.glb"
    src.parent.mkdir(parents=True)
    _triangle_glb(src)
    cache = tmp_path / "prop" / "export_cache" / "mesh"
    result = exports.run_export_worker(src, {"fbx": cache / "fbx", "obj": cache / "obj"}, "Tri_high")
    assert result["fbx"]["files"] == ["Tri_high.fbx"] and result["obj"]["files"] == ["Tri_high.obj", "Tri_high.mtl"]
    assert result["stats"]["triangles"] == 1
    assert (cache / "fbx" / "Tri_high.fbx").read_bytes().startswith(b"Kaydara FBX Binary")
    lines = (cache / "obj" / "Tri_high.obj").read_text(encoding="utf-8").splitlines()
    assert lines[1] == "mtllib Tri_high.mtl" and sum(line.startswith("v ") for line in lines) == 3
    assert [line for line in lines if line.startswith("f ")] == ["f 1/1/1 3/1/1 2/1/1"]   # points sorted, welded
    assert sorted(p.name for p in cache.iterdir()) == ["fbx", "obj"]     # no request file, no .tmp folders

    src.write_bytes(b"glTF, but broken")
    with pytest.raises(ProjectError) as e:
        exports.run_export_worker(src, {"obj": cache / "obj"}, "Tri_high")
    assert e.value.status_code == 422 and str(e.value).startswith("The 3D model mesh.glb could not be read")
    assert (cache / "obj" / "Tri_high.obj").is_file()                    # the earlier conversion stays whole


def _glb_chunks(data: bytes) -> tuple[dict[str, Any], bytes]:
    length = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20:20 + length]), data[28 + length:]


def _painted_triangle_glb(path: Path, image: bytes) -> None:
    """A GLB with one triangle whose base colour is `image`, stored between its positions and its indices."""
    positions, indices = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0), struct.pack("<3I", 0, 1, 2)
    padded = image + bytes(-len(image) % 4)
    document = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
                "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
                "textures": [{"source": 0}], "images": [{"bufferView": 2, "mimeType": "image/png"}],
                "buffers": [{"byteLength": 48 + len(padded)}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36},
                                {"buffer": 0, "byteOffset": 36 + len(padded), "byteLength": 12},
                                {"buffer": 0, "byteOffset": 36, "byteLength": len(image)}],
                "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                               "min": [0, 0, 0], "max": [1, 1, 0]},
                              {"bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR"}]}
    text = json.dumps(document).encode("utf-8")
    text += b" " * (-len(text) % 4)
    binary = positions + padded + indices
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 28 + len(text) + len(binary))
                     + struct.pack("<II", len(text), 0x4E4F534A) + text
                     + struct.pack("<II", len(binary), 0x004E4942) + binary)


@pytest.mark.skipif(not paths.venv_python(paths.IMESH).is_file(), reason="the Instant Meshes venv is not installed")
def test_the_hub_runs_the_real_repaint_worker(tmp_path: Path) -> None:
    prop_dir = tmp_path / "prop"
    bake_dir = prop_dir / "projection" / "bake_001"
    (bake_dir / "maps").mkdir(parents=True)
    old = tiny_png(4, 2)
    (bake_dir / "maps" / "basecolor.png").write_bytes(old)
    _painted_triangle_glb(bake_dir / "projection.glb", old)
    raw = prop_dir / ".paint-test.rgba"
    # What the page sends of a reset graphics card's texture: transparent black. The map is opaque, so it is refused.
    raw.write_bytes(bytes(32))
    with pytest.raises(ProjectError) as e:
        exports.run_repaint_worker(bake_dir, raw, 4, 2)
    assert e.value.status_code == 422 and "not painted on this bake's colours" in str(e.value)
    assert (bake_dir / "maps" / "basecolor.png").read_bytes() == old
    raw.write_bytes(bytes(255 if i % 4 == 3 else i for i in range(32)))     # opaque, as painted on this map
    unpainted = bake_dir / "maps" / "basecolor.unpainted.png"
    assert exports.run_repaint_worker(bake_dir, raw, 4, 2, unpainted) == \
        {"files": ["maps/basecolor.unpainted.png", "maps/basecolor.png", "projection.glb"]}
    new = (bake_dir / "maps" / "basecolor.png").read_bytes()
    assert new != old and new.startswith(b"\x89PNG\r\n\x1a\n") and new[16:24] == struct.pack(">II", 4, 2)
    assert unpainted.read_bytes() == old                                    # the bake's own map, byte for byte
    document, binary = _glb_chunks((bake_dir / "projection.glb").read_bytes())
    image, indices = document["bufferViews"][2], document["bufferViews"][1]
    assert binary[image["byteOffset"]:image["byteOffset"] + image["byteLength"]] == new
    assert binary[indices["byteOffset"]:indices["byteOffset"] + 12] == struct.pack("<3I", 0, 1, 2)
    assert sorted(p.name for p in prop_dir.iterdir()) == [".paint-test.rgba", "projection"]     # no request file
    # The one kept copy is never replaced, by a painting asked to keep the map again or not.
    raw.write_bytes(bytes(255 if i % 4 == 3 else 7 for i in range(32)))
    files = exports.run_repaint_worker(bake_dir, raw, 4, 2, unpainted)["files"]
    assert files == ["maps/basecolor.png", "projection.glb"]
    assert unpainted.read_bytes() == old
    new = (bake_dir / "maps" / "basecolor.png").read_bytes()

    with pytest.raises(ProjectError) as e:
        exports.run_repaint_worker(bake_dir, raw, 2, 4)
    assert e.value.status_code == 422 and "The painting is 2 x 4 pixels" in str(e.value)
    assert (bake_dir / "maps" / "basecolor.png").read_bytes() == new


def test_file_explorer_selects_the_saved_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    started: list[Any] = []

    class Process:
        def __init__(self, command: Any, **kwargs: Any) -> None:
            started.append(command)

        def poll(self) -> int | None:
            return None

    monkeypatch.setattr(subprocess, "Popen", Process)
    target = tmp_path / "Tavern props, old" / "Wooden_barrel.fbx"
    assert exports.show_in_explorer(target, True) is True
    assert exports.show_in_explorer(target.parent, False) is True
    explorer = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "explorer.exe"
    assert started == [f'"{explorer}" /select,"{target}"', f'"{explorer}" "{target.parent}"']
    monkeypatch.setenv("SPELLCAST_NO_EXPLORER", "1")
    assert exports.explorer_allowed() is False
    monkeypatch.delenv("SPELLCAST_NO_EXPLORER")
    assert exports.explorer_allowed() is True


def test_a_bake_obj_is_rewritten_to_the_flat_names(tmp_path: Path) -> None:
    bake = tmp_path / "bake_001"
    (bake / "export").mkdir(parents=True)
    (bake / "maps").mkdir()
    mtl = ("# comment\nnewmtl projection\nKd 1 1 1\nmap_Kd ../maps/basecolor.png\n"
           "map_Bump -bm 1 ../maps/normal.png\nnorm ../maps/normal.png\nmap_Pr ../maps/other.png\n")
    flat = {os.path.normcase(str((bake / "maps" / name).resolve())): f"Barrel_{Path(name).stem}.png"
            for name in ("basecolor.png", "normal.png")}
    assert exports._flat_mtl(mtl, bake / "export", flat).splitlines() == [
        "# comment", "newmtl projection", "Kd 1 1 1", "map_Kd Barrel_basecolor.png",
        "map_Bump -bm 1 Barrel_normal.png", "norm Barrel_normal.png", "map_Pr ../maps/other.png"]

    obj = bake / "export" / "projection.obj"
    obj.write_bytes(b"# Spellcast3D projection: 1 triangles\nmtllib projection.mtl\nv 0 0 0\nmtllib not.mtl\n")
    exports._copy_obj(obj, tmp_path / "Barrel.obj", "Barrel.mtl")
    assert (tmp_path / "Barrel.obj").read_bytes() == \
        b"# Spellcast3D projection: 1 triangles\nmtllib Barrel.mtl\nv 0 0 0\nmtllib not.mtl\n"
    obj.write_bytes(b"v 0 0 0\nmtllib late.mtl\n")                 # none before the geometry: left as it is
    exports._copy_obj(obj, tmp_path / "Plain.obj", "Barrel.mtl")
    assert (tmp_path / "Plain.obj").read_bytes() == b"v 0 0 0\nmtllib late.mtl\n"


def test_props_of_the_same_name_get_a_folder_each(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("Demo")
    ids = []
    for name in ("Barrel", "barrel", "Barrel"):                    # Windows takes Barrel and barrel for one name
        pid = store.add_prop("Demo", name)["id"]
        glb = store.prop_dir("Demo", pid) / "mesh" / "mesh.glb"
        glb.parent.mkdir(parents=True)
        glb.write_bytes(f"glTF of {pid}".encode())
        store.apply_stage_patch("Demo", pid, "mesh", {"glb": "mesh/mesh.glb"})
        ids.append(pid)
    folder = tmp_path / "exports" / "Demo"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Barrel_high.glb").write_bytes(b"saved before props had their own folders")
    (folder / "notes.txt").write_bytes(b"the artist's own")
    exporter = exports.Exporter(store, FakeConverter(), lambda fmt: None)
    # A folder's name never ends in a dot (Windows would drop it: "Barrel." is the folder "Barrel").
    assert [exports.file_stem(name) for name in ("Wooden barrel", "Barrel._", "_.Barrel", "._")] == [
        "Wooden_barrel", "Barrel", "Barrel", ""]

    saved = exporter.save_all("Demo")
    assert saved.props == 3 and saved.folder == folder and saved.warnings == []
    # A folder per prop; the older copy straight in the project's folder is gone, the artist's file is not.
    assert sorted(p.name for p in folder.iterdir()) == ["Barrel", "Barrel_3", "barrel_2", "notes.txt"]
    for pid, stem in zip(ids, ("Barrel", "barrel_2", "Barrel_3")):
        assert (folder / stem / f"{stem}_high.glb").read_bytes() == f"glTF of {pid}".encode()
        assert sorted(p.name for p in (folder / stem).iterdir()) == sorted(
            f"{stem}_high{end}" for end in (".glb", ".fbx", ".obj", ".mtl", "_basecolor.png"))
        assert f"{stem}\\{stem}_high.fbx" in saved.files
    assert len(saved.files) == 15 and saved.answer(False)["folder"] == "exports\\Demo"

    # One file of one prop goes into the same folder of that prop.
    one = exporter.save_prop("Demo", ids[1], "mesh", "glb")
    assert one.answer(False) == {"folder": "exports\\Demo\\barrel_2", "folder_path": str(folder / "barrel_2"),
                                 "files": ["barrel_2_high.glb"], "shown": "barrel_2_high.glb", "opened": False}
    assert len(list((folder / "barrel_2").iterdir())) == 5


def test_images_save_puts_the_viewports_picture_into_the_props_folder(tmp_path: Path) -> None:
    """Images' Save: one of the prop's pictures as it is, or the viewport with its marks as the next marked_<n>."""
    store = ProjectStore(tmp_path / "projects")
    store.create_project("Demo")
    pid = store.add_prop("Demo", "Wooden barrel")["id"]
    store.add_image("Demo", pid, b"\x89PNG\r\n\x1a\npicture", "barrel.png")
    (rel,) = store.get_prop("Demo", pid)["images"]["candidates"]
    exporter = exports.Exporter(store, FakeConverter(), lambda fmt: None)
    folder = tmp_path / "exports" / "Demo" / "Wooden_barrel"

    saved = exporter.save_picture("Demo", pid, rel)
    name = f"Wooden_barrel_picture_{Path(rel).stem}.png"
    assert saved.answer(False) == {"folder": "exports\\Demo\\Wooden_barrel", "folder_path": str(folder),
                                   "files": [name], "shown": name, "opened": False}
    assert (folder / name).read_bytes() == b"\x89PNG\r\n\x1a\npicture"
    store.set_viewport("Demo", pid, b"\x89PNG\r\n\x1a\nmarked")
    assert [exporter.save_picture("Demo", pid, exports.VIEWPORT_FILE).shown for _ in range(2)] == [
        "Wooden_barrel_marked_1.png", "Wooden_barrel_marked_2.png"]
    for bad in ("../project.json", "images/9999.png", None):
        with pytest.raises(ProjectError):
            exporter.save_picture("Demo", pid, bad)
