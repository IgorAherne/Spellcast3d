"""Pictures of several props at once: the Images chooser's "Import folder" (the hub's own folder picker, dialogs.py,
then this module reads the folder) and, on the page, a drop of many pictures (static/js/stages/propfiles.js, the
same rules; tests/test_web.py keeps the two the same). Which prop a file belongs to and which side it shows is read
from its name:

    knight_front.png  knight_back.png  dragon_front.png  dragon-left.png  Front_Ork.png  ork back.jpg

A side word ("front", "back" or "rear", "left", "right") at the start or the end of the name (after the usual
separators: "_", "-", a space, a dot) is the side; the rest, with the separators as spaces, is the prop's name
("Knight", "Dragon", "Ork"). A file that is nothing but a side word ("front.png" in a folder "knight") takes its
folder's name. A name without a side word is a prop of its own, with that one picture ("barrel.png" -> "Barrel").
Two files with the same prop and side are not the same prop: the second keeps what is left of its name.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from .projects import MAX_IMAGE_UPLOADS, MAX_UPLOAD_BYTES, ProjectError, ProjectStore, sniff_image

SIDES = {"front": "front", "back": "back", "rear": "back", "left": "left", "right": "right"}
_SEP = r"[\s_\-.]+"
_AT_START = re.compile(rf"^(front|back|rear|left|right)(?:{_SEP}|$)", re.IGNORECASE)
_AT_END = re.compile(rf"(?:^|{_SEP})(front|back|rear|left|right)$", re.IGNORECASE)
MAX_NAME = 40                   # as the hub shortens a prop's name (projects.box_name_from_file)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
MAX_FOLDER_PROPS = 60           # props one folder import makes at most (the rest is said, not made)


def _parts(path: str) -> tuple[str, str]:
    """"C:\\art\\Knight_Front.png" or "knight/front.png" -> (stem "Knight_Front", folder "knight")."""
    slash = str(path or "").replace("\\", "/")
    file = slash.rsplit("/", 1)[-1]
    folder = slash.rsplit("/", 1)[0].rsplit("/", 1)[-1] if "/" in slash else ""
    return re.sub(r"\.[^.]*$", "", file), folder


def prop_name(stem: str) -> str:
    """The words of a file or folder name as a prop's name: "old_barrel-01" -> "Old barrel 01"."""
    name = re.sub(r"[\s_\-.]+", " ", str(stem or "")).strip()[:MAX_NAME]
    return name[:1].upper() + name[1:] if name else ""


def prop_of(path: str) -> tuple[str, str | None]:
    """One file's place: (the prop's name, the side it shows or None). `path` may carry its folders."""
    stem, folder = _parts(path)
    side = None
    rest = stem
    m = _AT_END.search(stem)
    if m:
        side = SIDES[m.group(1).lower()]
        rest = stem[:m.start()]
    else:
        m = _AT_START.search(stem)
        if m:
            side = SIDES[m.group(1).lower()]
            rest = stem[m.end():]
    name = prop_name(rest) or (prop_name(folder) if side else "") or prop_name(stem)
    return name, side


def group_prop_files(paths: list[str]) -> list[tuple[str, list[tuple[str, str | None]]]]:
    """Files -> the props to make, in the order their first file came: [(name, [(path, side), ...])]. A file's side
    is what its name says; a second file for a side the prop already has becomes its own prop; files without a side
    word are props of their own."""
    groups: dict[str, dict[str, Any]] = {}

    def claim(name: str, path: str, side: str | None) -> None:
        g = groups.setdefault(name.lower(), {"name": name, "files": [], "sides": set()})
        g["files"].append((path, side))
        if side:
            g["sides"].add(side)

    for path in paths:
        prop, side = prop_of(path)
        if not side:
            claim(prop_name(_parts(path)[0]) or prop or "Prop", path, None)
            continue
        if side not in groups.get(prop.lower(), {"sides": set()})["sides"]:
            claim(prop, path, side)
            continue
        name = prop_name(_parts(path)[0])      # "knight_front (2).png" after "knight_front.png": its own prop
        n = 2
        while side in groups.get(name.lower(), {"sides": set()})["sides"]:
            name = f"{prop_name(_parts(path)[0])} {n}"
            n += 1
        claim(name, path, side)
    return [(g["name"], g["files"]) for g in groups.values()]


def folder_pictures(folder: Path) -> list[Path]:
    """The picture files of `folder` and its subfolders (by suffix), in a fixed order."""
    found = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(found, key=lambda p: str(p.relative_to(folder)).lower())


def import_prop_folder(store: ProjectStore, project: str, folder: Path,
                       say: Callable[[str], None] | None = None) -> dict[str, Any]:
    """The pictures of `folder` become props of `project` (ProjectStore.add_prop + add_images, the sides their
    names say, the front one chosen). -> {"props": [{"id", "name", "sides"}], "skipped": [str], "folder": str}.
    ProjectError when the folder is not there or has no pictures."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ProjectError(f"The folder {folder} does not exist (or is not a folder).", 404)
    files = folder_pictures(folder)
    if not files:
        raise ProjectError(f"The folder {folder} has no pictures (PNG, JPG or WebP).", 400)
    skipped: list[str] = []
    readable: dict[str, bytes] = {}
    for path in files:
        rel = str(path.relative_to(folder)).replace("\\", "/")
        try:
            data = path.read_bytes()
        except OSError as e:
            skipped.append(f"{rel}: could not be read ({e.strerror or e})")
            continue
        if not data or len(data) > MAX_UPLOAD_BYTES:
            skipped.append(f"{rel}: {'empty' if not data else f'bigger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB'}")
        elif sniff_image(data) is None:
            skipped.append(f"{rel}: not a PNG, JPG or WebP picture")
        else:
            readable[rel] = data
    groups = group_prop_files(list(readable))
    made: list[dict[str, Any]] = []
    for name, own in groups:
        if len(made) >= MAX_FOLDER_PROPS:
            skipped.append(f"{name}: more than {MAX_FOLDER_PROPS} props in one go")
            continue
        if len(own) > MAX_IMAGE_UPLOADS:
            skipped.append(f"{name}: {len(own) - MAX_IMAGE_UPLOADS} pictures more than the {MAX_IMAGE_UPLOADS} a prop takes at once")
            own = own[:MAX_IMAGE_UPLOADS]
        sides = [side for _, side in own] if any(side for _, side in own) else None
        prop = store.add_prop(project, name)
        try:
            store.add_images(project, prop["id"], [(readable[rel], rel.rsplit("/", 1)[-1]) for rel, _ in own],
                             sides if len(own) > 1 else None)
        except ProjectError as e:
            skipped.append(f"{name}: {e}")
            store.delete_prop(project, prop["id"])
            continue
        made.append({"id": prop["id"], "name": name, "sides": [side for _, side in own if side]})
        if say:
            say(f"imported {name} ({len(own)} picture{'s' if len(own) != 1 else ''})")
    return {"props": made, "skipped": skipped, "folder": str(folder)}
