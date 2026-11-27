"""Saving finished models for the artist: exports/<project name>/<prop>/, one folder per project with a
folder per prop in it.

The page's .glb / .fbx / .obj buttons (3D and Projection stages), Images' Save and Export all write here, then File
Explorer opens on the result (a saved file selected in its prop's folder; Export all: the project's
folder), so the artist drags the files into a game engine by hand. The page never downloads anything.

A prop's folder and its file names are the prop's name made safe, spaces as underscores ("Wooden barrel"
-> "Wooden_barrel"); props that share a name get _2, _3 in project order (file_stems), so a prop keeps
its names while props are added after it. A file of the same name is replaced, and its older copy
straight in the project's folder (every file was saved there before props had their own folders) is
removed; nothing else in the folders is touched. The files of a prop, flat in its folder:

    the bake (Projection)      Wooden_barrel.glb, .fbx (maps embedded), .obj + .mtl + Wooden_barrel_<map>.png
    the 3D model (3D stage)    Wooden_barrel_high.glb, _high.fbx, _high.obj + _high.mtl + _high_<map>.png
    the retopo alone           Wooden_barrel_retopo.obj  (Export all, for a prop that is not baked)

Export all saves a prop with neither a bake nor a retopo as its 3D model, or as its Omni-Texture while that is used
(the 3D stage's buttons always save the 3D model itself).

The bake's own FBX is copied as it is; its OBJ is copied with the MTL (and the OBJ's mtllib line)
rewritten to the flat map names. A 3D model's FBX and OBJ, and a bake's when it was baked without
them, are converted from the GLB by launchers/export_worker.py in the imesh venv (the hub venv has no
numpy); a conversion is kept in the prop's export_cache/<stage>/<format>/ and made again only when the
GLB or the file name changes.

The same worker saves the artist's painting on a bake into the bake's own files (run_repaint_worker).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths
from .procman import kill_tree, spawn_in_job
from .projects import VIEWPORT_FILE, ProjectError, ProjectStore, safe_name, omnitexture_used

log = logging.getLogger(__name__)

EXPORT_STAGES = ("mesh", "projection")
EXPORT_FORMATS = ("glb", "fbx", "obj")
CONVERTED = ("fbx", "obj")          # the formats export_worker.py makes out of a GLB
CONVERT_TIMEOUT = 15 * 60.0         # a million triangles take seconds; this only catches a hung worker
REPAINT_TIMEOUT = 5 * 60.0          # a 4096 px painting takes seconds
CACHE_VERSION = 1                   # raise it when the worker's files change, so the caches are made again
CACHE_FOLDER = "export_cache"       # props/<id>/export_cache/<stage>/<format>/
_SOURCE_FILE = "source.json"        # in a cache folder: the GLB and name it was made from, and its files
# The suffixes a prop's file stem is also used with: a stem is only free when none of them is taken.
_VARIANTS = ("", "_high", "_retopo")
_MTL_MAP_KEYWORDS = frozenset({"map_kd", "map_ka", "map_ks", "map_ns", "map_d", "map_bump", "bump", "norm", "disp",
                               "decal", "refl", "map_pr", "map_pm", "map_ps", "map_ke"})
NO_EXPLORER_ENV = "SPELLCAST_NO_EXPLORER"

# convert(src GLB, {format: cache folder}, file name without extension) -> export_worker.py's result.
Converter = Callable[[Path, dict[str, Path], str], dict[str, Any]]
# repaint(bake folder, raw RGBA file, width, height, where to keep the bake's own base colour map first or None)
# -> export_worker.py's result: {"files": [...]}.
Repainter = Callable[[Path, Path, int, int, Path | None], dict[str, Any]]


# ---- names ----------------------------------------------------------------------------------------

def file_stem(name: str) -> str:
    """A prop's name as its folder's name and the start of its file names: "Wooden barrel" -> "Wooden_barrel"
    ('' when nothing is left). Never with a dot at an end: Windows drops the last dot of a folder's name, so
    "Barrel." would be the folder of the prop called "Barrel"."""
    return re.sub(r"_+", "_", safe_name(name).replace(" ", "_")).strip("_.")


def file_stems(props: Iterable[dict[str, Any]]) -> dict[str, str]:
    """{prop id: file stem} of the props (not the draft), in project order: the name of the prop's export
    folder and the start of its file names. The second prop called "Barrel" is Barrel_2, the third
    Barrel_3; a stem is never one another prop's files also use (a prop called "Barrel high" does not get
    the 3D model's "Barrel_high")."""
    taken: set[str] = set()
    stems: dict[str, str] = {}
    for prop in props:
        if prop.get("draft"):
            continue
        base = file_stem(str(prop.get("name") or "")) or prop["id"]
        stem, number = base, 2
        while any(f"{stem}{variant}".casefold() in taken for variant in _VARIANTS):
            stem, number = f"{base}_{number}", number + 1
        taken.update(f"{stem}{variant}".casefold() for variant in _VARIANTS)
        stems[prop["id"]] = stem
    return stems


# ---- File Explorer -----------------------------------------------------------------------------------

_explorers: list[subprocess.Popen[bytes]] = []
_explorers_lock = threading.Lock()


def explorer_allowed() -> bool:
    """False when SPELLCAST_NO_EXPLORER=1: tests and automated runs save files but open no windows."""
    return os.environ.get(NO_EXPLORER_ENV) != "1"


def show_in_explorer(target: Path, select: bool) -> bool:
    """Open File Explorer on the folder `target`, or (select) on the folder of the file `target` with it
    selected. Not waited for. False when Explorer could not be started."""
    explorer = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "explorer.exe"
    # One command line, quoted the way explorer.exe parses it: /select,"<file>" (Windows paths have no ").
    command = f'"{explorer}" /select,"{target}"' if select else f'"{explorer}" "{target}"'
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, close_fds=True)
    except OSError as e:
        log.warning("Could not open File Explorer on %s (%s)", target, e)
        return False
    with _explorers_lock:       # kept until it exits, so it is reaped rather than reported as leaked
        _explorers[:] = [p for p in _explorers if p.poll() is None] + [process]
    return True


# ---- the worker ------------------------------------------------------------------------------------

def run_export_worker(src: Path, outputs: dict[str, Path], name: str) -> dict[str, Any]:
    """Convert `src` into each of `outputs` ({format: folder}) with launchers/export_worker.py in the imesh
    venv -> its result. Blocking (run it in a thread). Raises ProjectError with the worker's words."""
    first = next(iter(outputs.values()))
    first.parent.mkdir(parents=True, exist_ok=True)
    request = {"cmd": "convert", "src": str(src), "name": name,
               "outputs": {fmt: str(folder) for fmt, folder in outputs.items()}}
    return _run_worker(request, first.parent / f"request{os.getpid()}.json", f"Saving {name}",
                       "the program that saves FBX and OBJ files", CONVERT_TIMEOUT)


def run_repaint_worker(bake_dir: Path, rgba: Path, width: int, height: int,
                       unpainted: Path | None = None) -> dict[str, Any]:
    """Save the painting in the raw RGBA file `rgba` (width x height) into the bake in `bake_dir` with
    launchers/export_worker.py -> its result. With `unpainted`, the bake's own base colour map is kept there first
    (unless a file is there already). Blocking (run it in a thread). Raises ProjectError."""
    request: dict[str, Any] = {"cmd": "repaint", "dir": str(bake_dir), "rgba": str(rgba), "width": width,
                               "height": height}
    if unpainted is not None:
        request["unpainted"] = str(unpainted)
    return _run_worker(request, rgba.with_name(f"{rgba.name}.json"), "Saving the painting",
                       "the program that saves the painting", REPAINT_TIMEOUT)


def _run_worker(request: dict[str, Any], request_file: Path, task: str, program: str,
                timeout: float) -> dict[str, Any]:
    """Run export_worker.py on `request` (in `request_file` while it runs) -> its result. `task` names the work in
    the messages ("Saving Crate_high"), `program` the worker ("the program that saves FBX and OBJ files")."""
    request_file.write_text(json.dumps(request), encoding="utf-8")
    timed_out = threading.Event()
    result: Any = None
    error: tuple[str, str] | None = None
    try:
        process = spawn_in_job([str(paths.venv_python(paths.IMESH)), str(paths.LAUNCHERS / "export_worker.py"),
                                str(request_file)], cwd=paths.LAUNCHERS, env=paths.clean_env(paths.IMESH),
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

        def stop() -> None:
            timed_out.set()
            kill_tree(process.pid)

        timer = threading.Timer(timeout, stop)
        timer.start()
        try:
            assert process.stdout is not None
            for raw in process.stdout:          # its JSON lines, and its log (stderr) in between
                line = raw.decode("utf-8", "replace").strip()
                try:
                    message = json.loads(line) if line.startswith("{") else None
                except ValueError:
                    message = None
                if not isinstance(message, dict):
                    if line:
                        log.info("export worker: %s", line)
                elif "result" in message:
                    result = message["result"]
                elif "error" in message:
                    error = (str(message["error"]), str(message.get("kind") or "internal"))
            code = process.wait()
        finally:
            timer.cancel()
    except OSError as e:
        raise ProjectError(f"Spellcast3D could not start {program} ({e}). Run tools/update.bat, or download "
                           f"Spellcast3D again.", 500) from e
    finally:
        request_file.unlink(missing_ok=True)
    if timed_out.is_set():
        raise ProjectError(f"{task} took longer than {timeout / 60:.0f} minutes, so it was stopped. Try again; "
                           f"the details are in code/logs/hub.log.", 504)
    if error is not None:
        log.log(logging.INFO if error[1] == "user" else logging.ERROR, "export worker: %s", error[0])
        raise ProjectError(error[0], 422 if error[1] == "user" else 500)
    if code != 0 or not isinstance(result, dict):
        raise ProjectError(f"{task} stopped unexpectedly (exit code {code}). The details are in "
                           f"code/logs/hub.log.", 500)
    return result


# ---- saving -----------------------------------------------------------------------------------------

@dataclass
class Saved:
    """What one save wrote: `files`, named from `folder` (the folder the page names and File Explorer shows)."""

    project: Path                       # the project's export folder: exports/<project name>
    folder: Path                        # one prop's folder in it, or `project` itself (Export all)
    files: list[str] = field(default_factory=list)      # Export all: "Wooden_barrel\\Wooden_barrel.glb"
    into: Path = field(init=False)      # the prop's folder being written now (Export all: prop after prop)
    shown: str | None = None            # the file to select in File Explorer
    props: int = 0                      # props that got at least one file (Export all)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.into = self.folder

    def answer(self, opened: bool) -> dict[str, Any]:
        """The route's answer. folder: as the page names it ("exports\\Tavern props"; one prop's:
        "exports\\Tavern props\\Wooden_barrel")."""
        named = (self.project.parent.name, *self.folder.relative_to(self.project.parent).parts)
        out = {"folder": "\\".join(named), "folder_path": str(self.folder),
               "files": list(dict.fromkeys(self.files)), "opened": opened}
        if self.shown is not None:
            out["shown"] = self.shown
        return out


class Exporter:
    """Writes props' results into their project's export folder, each prop into its own folder there. Not
    thread-safe: the hub runs one save at a time (its export lock)."""

    def __init__(self, store: ProjectStore, convert: Converter, can_convert: Callable[[str], None]):
        """convert: runs export_worker.py (run_export_worker, or a stand-in in tests). can_convert(format)
        raises ProjectError saying why an FBX or OBJ cannot be made right now (the imesh venv missing)."""
        self.store = store
        self.convert = convert
        self.can_convert = can_convert

    def save_prop(self, project: str, prop_id: str, stage: Any, fmt: Any) -> Saved:
        """One file (an OBJ with its MTL and maps) of one prop, into the prop's folder: the 3D model (stage
        "mesh") or the bake ("projection") as "glb", "fbx" or "obj"."""
        if stage not in EXPORT_STAGES:
            raise ProjectError('Choose what to save: "mesh" (the 3D model) or "projection" (the textured '
                               'low poly).')
        if fmt not in EXPORT_FORMATS:
            raise ProjectError('Choose the file type: "glb", "fbx" or "obj".')
        data = self.store.get_project(project)
        prop = self.store.get_prop(project, prop_id)
        if prop["draft"]:
            raise ProjectError("This prop has no picture yet, so there is nothing to save.", 409)
        stem = file_stems(data["props"])[prop_id]
        folder = self.store.export_dir(data)
        saved = Saved(folder, folder / stem)
        if stage == "mesh":
            high = self.store.result_file(project, prop, "mesh", "glb")
            if high is None:
                raise ProjectError("This prop has no 3D model yet. Make one in the 3D stage first.", 409)
            name = f"{stem}_high"
            self._save_model(project, prop, "mesh", high, name, (fmt,), saved)
        else:
            name = stem
            self._save_bake(project, prop, self.store.projection_files(project, prop), stem, (fmt,), saved)
        saved.shown = f"{name}.{fmt}"
        return saved

    def save_picture(self, project: str, prop_id: str, rel: Any) -> Saved:
        """The picture the Images viewport shows, as it is, into the prop's folder: one of its pictures (images.candidates)
        as <stem>_picture_<its number>, or the viewport with the marks painted on it (projects.VIEWPORT_FILE, the
        page uploads it first) as <stem>_marked_<n>.png, the next free n."""
        data = self.store.get_project(project)
        prop = self.store.get_prop(project, prop_id)
        if not isinstance(rel, str) or (rel != VIEWPORT_FILE and rel not in ((prop.get("images") or {}).get("candidates")
                                                                            or [])):
            raise ProjectError("That picture is not one of this prop's. Pick it under the viewport and save again.", 404)
        src = self.store.prop_dir(project, prop_id) / rel
        if not src.is_file():
            raise ProjectError("That picture's file is gone. Pick another picture under the viewport.", 404)
        stem = file_stems(data["props"])[prop_id]
        folder = self.store.export_dir(data)
        saved = Saved(folder, folder / stem)
        if rel == VIEWPORT_FILE:
            n = 1
            while (saved.folder / f"{stem}_marked_{n}.png").exists():
                n += 1
            name = f"{stem}_marked_{n}.png"
        else:
            name = f"{stem}_picture_{Path(rel).stem}{Path(rel).suffix.lower()}"
        self._place(src, saved, name)
        saved.shown = name
        return saved

    def save_all(self, project: str) -> Saved:
        """Every prop's best result, each prop in its own folder: the bake (GLB, FBX, OBJ + MTL + maps), else
        the retopo OBJ, else the 3D model (GLB, FBX, OBJ; its Omni-Texture instead while that is used:
        projects.omnitexture_used). A prop with a picture only has nothing to save. A
        prop whose FBX or OBJ cannot be made keeps what could be saved, and a warning says why."""
        data = self.store.get_project(project)
        stems = file_stems(data["props"])
        folder = self.store.export_dir(data)
        saved = Saved(folder, folder)
        for prop in data["props"]:
            if prop["draft"]:
                continue
            stem, before = stems[prop["id"]], len(saved.files)
            saved.into = folder / stem
            try:
                try:
                    baked = self.store.projection_files(project, prop)
                except ProjectError:
                    baked = None
                if baked is not None:
                    self._save_bake(project, prop, baked, stem, EXPORT_FORMATS, saved)
                elif (low := self.store.result_file(project, prop, "retopo", "obj")) is not None:
                    self._place(low, saved, f"{stem}_retopo.obj")
                elif omnitexture_used(prop) and (high := self.store.result_file(
                        project, prop, "omnitexture", "glb")) is not None:
                    self._save_model(project, prop, "omnitexture", high, f"{stem}_high", EXPORT_FORMATS, saved)
                elif (high := self.store.result_file(project, prop, "mesh", "glb")) is not None:
                    self._save_model(project, prop, "mesh", high, f"{stem}_high", EXPORT_FORMATS, saved)
            except ProjectError as e:
                saved.warnings.append(f"{prop['name']}: {e}")
            if len(saved.files) > before:
                saved.props += 1
        if not saved.props:
            if saved.warnings:
                raise ProjectError(" ".join(saved.warnings), 409)
            raise ProjectError("There is nothing to save yet: none of the props has a 3D model.", 404)
        return saved

    # ---- the parts ----------------------------------------------------------------------------------
    def _save_model(self, project: str, prop: dict[str, Any], stage: str, glb: Path, name: str,
                    formats: Iterable[str], saved: Saved) -> None:
        """A GLB as <name>.glb (copied) and/or converted to <name>.fbx / <name>.obj + .mtl + maps."""
        formats = list(formats)
        if "glb" in formats:
            self._place(glb, saved, f"{name}.glb")
        converted = [fmt for fmt in formats if fmt in CONVERTED]
        if converted:
            for folder, files in self._converted(project, prop, stage, glb, name, converted, saved).values():
                for file in files:
                    self._place(folder / file, saved, file)

    def _save_bake(self, project: str, prop: dict[str, Any], baked: dict[str, Any], stem: str,
                   formats: Iterable[str], saved: Saved) -> None:
        """The bake's own files, flat in the prop's folder; what it was baked without is converted from its GLB."""
        formats = list(formats)
        if "glb" in formats:
            self._place(baked["glb"], saved, f"{stem}.glb")
        if "fbx" in formats and baked["fbx"] is not None:
            self._place(baked["fbx"], saved, f"{stem}.fbx")
        if "obj" in formats and baked["obj"] is not None:
            self._place_baked_obj(baked, stem, saved)
        missing = [fmt for fmt in CONVERTED if fmt in formats and baked[fmt] is None]
        if missing:
            self._save_model(project, prop, "projection", baked["glb"], stem, missing, saved)

    def _place_baked_obj(self, baked: dict[str, Any], stem: str, saved: Saved) -> None:
        """The bake's OBJ, its MTL and every baked map, flat: <stem>.obj, <stem>.mtl, <stem>_<map>.png. The
        MTL's map paths (../maps/basecolor.png) and the OBJ's mtllib line name the new files."""
        obj: Path = baked["obj"]
        mtl = obj.with_suffix(".mtl")
        maps = [(path, f"{stem}_{map_id}{path.suffix.lower()}") for map_id, path in baked["maps"]]
        has_mtl = mtl.is_file()
        self._write(saved, f"{stem}.obj", lambda tmp: _copy_obj(obj, tmp, f"{stem}.mtl" if has_mtl else None))
        if has_mtl:
            flat = {os.path.normcase(str(path.resolve())): name for path, name in maps}
            text = _flat_mtl(mtl.read_text(encoding="utf-8", errors="replace"), mtl.parent, flat)
            self._write(saved, f"{stem}.mtl", lambda tmp: tmp.write_text(text, encoding="utf-8", newline="\n"))
        for path, name in maps:
            self._place(path, saved, name)

    def _converted(self, project: str, prop: dict[str, Any], stage: str, glb: Path, name: str,
                   formats: list[str], saved: Saved) -> dict[str, tuple[Path, list[str]]]:
        """{format: (cache folder, its files)}: made by the worker unless the cache still fits the GLB."""
        prop_dir = self.store.prop_dir(project, prop["id"])
        root = prop_dir / CACHE_FOLDER / stage
        stat = glb.stat()
        source = {"glb": glb.resolve().relative_to(prop_dir.resolve()).as_posix(), "size": stat.st_size,
                  "mtime_ns": stat.st_mtime_ns, "name": name, "version": CACHE_VERSION}
        ready: dict[str, tuple[Path, list[str]]] = {}
        for fmt in formats:
            files = _cached(root / fmt, source)
            if files is not None:
                ready[fmt] = (root / fmt, files)
        missing = [fmt for fmt in formats if fmt not in ready]
        if missing:
            for fmt in missing:
                self.can_convert(fmt)
            log.info("Converting %s to %s", glb, ", ".join(missing))
            result = self.convert(glb, {fmt: root / fmt for fmt in missing}, name)
            for fmt in missing:
                files = (result.get(fmt) or {}).get("files")
                if not isinstance(files, list) or not files or not all(
                        isinstance(f, str) and Path(f).name == f and (root / fmt / f).is_file() for f in files):
                    raise ProjectError(f"Saving {name}.{fmt} went wrong: the files are incomplete. Try again; the "
                                       f"details are in code/logs/hub.log.", 500)
                paths.write_text_atomic(root / fmt / _SOURCE_FILE, json.dumps({"source": source, "files": files}))
                ready[fmt] = (root / fmt, files)
            saved.warnings += [str(w) for w in result.get("warnings") or []]
        return ready

    def _place(self, src: Path, saved: Saved, name: str) -> None:
        self._write(saved, name, lambda tmp: shutil.copyfile(src, tmp))

    def _write(self, saved: Saved, name: str, write: Callable[[Path], Any]) -> None:
        """write(temporary file) beside name in the prop's folder (saved.into), then replace it: a
        half-written file never has the name. Its copy straight in the project's folder (saved there
        before props had their own folders) is removed, so the file is not there twice."""
        folder = saved.into
        target = folder / name
        tmp = folder / f".{name}.{os.getpid()}.tmp"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            write(tmp)
            os.replace(tmp, target)
        except PermissionError as e:
            raise ProjectError(f'{name} is open in another program, so it could not be replaced. Close it '
                               f'there, then save again.', 409) from e
        except OSError as e:
            raise ProjectError(f"Could not save {name} in {folder} ({e.strerror or e}). Check that the "
                               f"disk is not full.", 500) from e
        finally:
            tmp.unlink(missing_ok=True)
        old = saved.project / name
        try:
            if old.is_file():
                old.unlink()
        except OSError as e:            # open in another program, say: it stays
            log.info("Could not remove the older %s (%s)", old, e)
        saved.files.append(str(target.relative_to(saved.folder)))


def _cached(folder: Path, source: dict[str, Any]) -> list[str] | None:
    """The files of a cache folder made from `source`, or None when it was made from something else."""
    try:
        meta = json.loads((folder / _SOURCE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    files = meta.get("files") if isinstance(meta, dict) else None
    if meta.get("source") != source or not isinstance(files, list) or not files:
        return None
    if not all(isinstance(f, str) and Path(f).name == f and (folder / f).is_file() for f in files):
        return None
    return files


def _flat_mtl(text: str, folder: Path, flat: dict[str, str]) -> str:
    """An MTL with every map it names (relative to `folder`) renamed as `flat` says (normcased paths)."""
    lines = []
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0].lower() in _MTL_MAP_KEYWORDS:
            name = flat.get(os.path.normcase(str((folder / tokens[-1]).resolve())))
            if name is not None:
                line = line[:line.rstrip().rfind(tokens[-1])] + name
        lines.append(line)
    return "\n".join(lines) + "\n"


def _copy_obj(src: Path, dst: Path, mtllib: str | None) -> None:
    """Copy an OBJ with its mtllib line (at the top, before any geometry) naming `mtllib` instead."""
    with open(src, "rb") as reader, open(dst, "wb") as writer:
        for line in reader:
            if line.startswith(b"mtllib") and mtllib is not None:
                writer.write(f"mtllib {mtllib}\n".encode("utf-8"))
                break
            writer.write(line)
            if line[:2] in (b"v ", b"vt", b"vn", b"f "):
                break
        shutil.copyfileobj(reader, writer, 1 << 20)
