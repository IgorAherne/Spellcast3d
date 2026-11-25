"""The artist's prompt library: short named snippets for the words of a picture, kept with the installation in one
file (paths.PROMPT_LIBRARY, prompt-library.json beside spellcast3d-config.ini, gitignored) rather than in a project,
so every project offers them. The page edits the list in its "Prompt Library" panel and sends it back whole (PUT):
one artist, one page, so the last save wins.

An entry: {"id", "name" (10 characters at most: the chip at the bottom of the words field), "text", "shown" (the chip
is there)}. A library starts with DEFAULTS; the file remembers which of them it was offered ("offered"), so a default
the artist removed stays removed, and one added later reaches the libraries that exist already - at their end, or
right after the prompt it belongs to (AFTER; "placed": the file remembers that too)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .projects import ProjectError
from .settings import atomic_write_json

log = logging.getLogger(__name__)

MAX_ENTRIES = 500
MAX_NAME = 10
MAX_TEXT = 20_000
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
# The prompts every library is given once, as a new library has them. "4 views": a sheet the Images stage's "4
# quadrants" slices into the object's sides. "T 4 views": that sheet for a character to be rigged, arms out.
# "toon": a look.
DEFAULTS: tuple[dict[str, Any], ...] = (
    {"id": "views4", "name": "4 views", "shown": True,
     "text": "shown in an orthographic sheet of four quadrants: front, back, left, right."},
    {"id": "tpose4", "name": "T 4 views", "shown": True,
     "text": "standing in a t-pose, shown in an orthographic sheet of four quadrants: front, back, left view, "
             "right view.\nAt the top row the arms are spread to the sides."},
    {"id": "toon", "name": "toon", "shown": True, "text": "a stylized hand-painted cartoon style"},
)
# A default that stands right after another one (id -> the other's id) in a library that has both: where it is put
# when it arrives, and where a library that got it at its end before this rule has it moved to, once.
AFTER = {"tpose4": "views4"}


def check(entries: Any) -> list[dict[str, Any]]:
    """The list the page sent, checked -> entries as they are saved (a name cut to MAX_NAME characters)."""
    if not isinstance(entries, list) or len(entries) > MAX_ENTRIES:
        raise ProjectError(f"The prompt library takes a list of at most {MAX_ENTRIES} prompts.")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ProjectError("Each prompt of the library is an object.")
        pid, name, text = entry.get("id"), entry.get("name", ""), entry.get("text", "")
        if not isinstance(pid, str) or not _ID_RE.match(pid) or pid in seen:
            raise ProjectError("Each prompt of the library needs an id of its own.")
        if not isinstance(name, str) or not isinstance(text, str) or len(text) > MAX_TEXT:
            raise ProjectError(f"A prompt's name and text are words, its text {MAX_TEXT} characters at most.")
        seen.add(pid)
        out.append({"id": pid, "name": name.strip()[:MAX_NAME], "text": text, "shown": entry.get("shown") is True})
    return out


def read(path: Path) -> list[dict[str, Any]]:
    """The saved library, with the DEFAULTS it was not offered yet added at its end (and saved so). It starts anew -
    the defaults alone - when there is none yet, or when the file cannot be read: it is set aside then."""
    entries: list[dict[str, Any]] = []
    offered: set[Any] = set()
    placed: set[Any] = set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = check(data.get("prompts") if isinstance(data, dict) else data)
        if isinstance(data, dict):
            offered = {name for name in data.get("offered") or [] if isinstance(name, str)}
            placed = {name for name in data.get("placed") or [] if isinstance(name, str)}
    except FileNotFoundError:
        pass
    except (OSError, ValueError, ProjectError) as e:     # set aside, so the next save does not overwrite it
        aside = path.with_name(f"{path.stem}.unreadable{path.suffix}")
        log.warning("The prompt library %s cannot be read (%s): it starts anew, the file moves to %s", path, e, aside)
        try:
            path.replace(aside)
        except OSError:
            pass
    new = [dict(d) for d in DEFAULTS if d["id"] not in offered and all(e["id"] != d["id"] for e in entries)]
    if not new and placed >= set(AFTER):
        return entries
    entries = [*entries, *new]
    for prompt, other in AFTER.items():
        if prompt not in placed:
            _put_after(entries, prompt, other)
    try:
        return write(path, entries)
    except OSError as e:                                  # (a folder that cannot be written: shown all the same)
        log.warning("The prompt library %s could not be saved with its default prompts (%s)", path, e)
        return entries


def _put_after(entries: list[dict[str, Any]], prompt: str, other: str) -> None:
    """Moves the entry `prompt` right after the entry `other`, when the library has both."""
    ids = [e["id"] for e in entries]
    if prompt in ids and other in ids:
        entry = entries.pop(ids.index(prompt))
        entries.insert([e["id"] for e in entries].index(other) + 1, entry)


def write(path: Path, entries: Any) -> list[dict[str, Any]]:
    """Checks and saves the whole library -> the entries saved. Every default counts as offered and placed from then
    on: the page reads the library (read adds and places them) before it saves it."""
    checked = check(entries)
    atomic_write_json(path, {"prompts": checked, "offered": [d["id"] for d in DEFAULTS], "placed": sorted(AFTER)})
    return checked
