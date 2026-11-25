"""One model saved as FBX and/or OBJ for the artist's export folder, or the artist's painting saved
into a bake, in a child process of the hub.

    <imesh venv>\\Scripts\\python.exe export_worker.py <request.json>

The hub (spellcast/exports.py) starts it in the imesh venv, which has numpy, trimesh, Pillow and
instant_meshes_brush; the hub's own venv stays light. It imports only its sibling package spellbake
and instant_meshes_brush, never the hub. The request is one of

    {"cmd": "convert", "src": <model file (.glb)>, "name": <file name without extension>,
     "outputs": {"fbx": <folder>, "obj": <folder>}}          (either or both formats)
    {"cmd": "repaint", "dir": <bake folder>, "rgba": <raw file>, "width": W, "height": H,
     "unpainted": <file> (optional)}

convert: each output folder is written as <folder>.tmp and renamed to <folder> once every file of it
is written (an older <folder> is replaced), so a folder with that name is always complete.

repaint: the raw file holds W x H RGBA8 pixels, row 0 first as in maps/basecolor.png, which must be
that size. Their colour becomes the bake's base colour; its alpha is not touched (the painting never
changes alpha, so the opacity map stays) and neither is its mode (RGB stays RGB). The painting's alpha
must therefore be the map's own as the page holds it (255 for an RGB map): any other is no painting of
this bake's colours - a page whose copy of them was lost (a reset graphics card hands back transparent
black) - and is refused. With "unpainted", the map as it is now is kept in that file first, unless one
is there already (the hub asks for it before the bake's first painting). The one new PNG goes into
maps/basecolor.png, projection.glb and projection.fbx (when that embeds its maps): every file is
written beside itself first, then they are replaced one right after the other. The result is
{"files": [the files changed, relative to the bake folder]}.

It prints JSON lines on stdout, as bake_worker.py does: {"progress": 0..1, "message": str} while it
works, then {"result": {...}} (convert: spellbake.convert.convert's answer) and exit code 0, or
{"error": "<message for an artist>", "kind": "user" | "internal"} and exit code 1 (the traceback of an
internal error goes to stderr, which the hub logs).
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from PIL import Image

from spellbake import BakeError, fbx, glb, replace_files
from spellbake.bake import PNG_COMPRESS_LEVEL
from spellbake.convert import FORMATS, convert

PROGRESS_INTERVAL = 0.25
# A just-written folder can be held briefly by a virus scanner or the search indexer.
RENAME_ATTEMPTS = 40


class RequestError(Exception):
    """The request itself is malformed: a bug in whoever wrote it."""


class Output:
    """The JSON-lines channel to the hub (thread-safe, progress throttled)."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self._last = 0.0

    def send(self, message: dict[str, Any]) -> None:
        with self._lock:
            self._stream.write(json.dumps(message) + "\n")
            self._stream.flush()

    def progress(self, fraction: float, message: str) -> None:
        now = time.monotonic()
        if now - self._last >= PROGRESS_INTERVAL or fraction >= 1.0:
            self._last = now
            self.send({"progress": round(fraction, 4), "message": message})


def _replace_folder(source: Path, target: Path) -> None:
    for attempt in range(RENAME_ATTEMPTS):
        try:
            if target.exists():
                shutil.rmtree(target)
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == RENAME_ATTEMPTS - 1:
                raise
            time.sleep(0.05)


def run_convert(request: dict[str, Any], out: Output) -> dict[str, Any]:
    src, name, outputs = request.get("src"), request.get("name"), request.get("outputs")
    if not isinstance(src, str) or not src:
        raise RequestError("the request has no 'src' path")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise RequestError("'name' must be a file name")
    if not isinstance(outputs, dict) or not outputs or set(outputs) - set(FORMATS) \
            or not all(isinstance(v, str) and v for v in outputs.values()):
        raise RequestError(f"'outputs' must map some of {', '.join(FORMATS)} to folders")
    finals = {fmt: Path(folder) for fmt, folder in outputs.items()}
    temporary = {fmt: folder.with_name(folder.name + ".tmp") for fmt, folder in finals.items()}
    try:
        for folder in temporary.values():
            shutil.rmtree(folder, ignore_errors=True)
        result = convert(Path(src), temporary, name=name, progress=out.progress)
        for fmt, folder in temporary.items():
            _replace_folder(folder, finals[fmt])
    finally:
        for folder in temporary.values():
            shutil.rmtree(folder, ignore_errors=True)
    return result


def run_repaint(request: dict[str, Any], out: Output) -> dict[str, Any]:
    folder, raw = request.get("dir"), request.get("rgba")
    width, height = request.get("width"), request.get("height")
    unpainted = request.get("unpainted")
    if not isinstance(folder, str) or not folder or not isinstance(raw, str) or not raw:
        raise RequestError("the request needs the 'dir' and 'rgba' paths")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (width, height)):
        raise RequestError("'width' and 'height' must be positive whole numbers")
    if unpainted is not None and (not isinstance(unpainted, str) or not unpainted):
        raise RequestError("'unpainted' must be a path")
    bake_dir = Path(folder)
    basecolor, model, fbx_file = bake_dir / "maps" / "basecolor.png", bake_dir / "projection.glb", \
        bake_dir / "projection.fbx"
    out.progress(0.0, "Saving the painting…")
    try:
        with Image.open(basecolor) as old:
            old.load()
            mode, size = old.mode, old.size
            alpha = np.asarray(old.getchannel("A")) if mode == "RGBA" else None
            # The alpha the page holds: the map's own, an RGB map's colour key (a tRNS chunk), or opaque.
            keyed = mode == "RGB" and "transparency" in old.info
            shown = alpha if alpha is not None else np.asarray(old.convert("RGBA").getchannel("A")) if keyed else 255
    except FileNotFoundError:
        raise BakeError("The bake's base colour map is missing, so the painting could not be saved. Bake the "
                        "prop again.") from None
    except OSError as e:
        raise BakeError(f"The bake's base colour map could not be read ({e}). Bake the prop again.") from e
    if size != (width, height):
        raise BakeError(f"The painting is {width} x {height} pixels, but the bake's base colour map is "
                        f"{size[0]} x {size[1]}. Reload the page and paint again.")
    if mode not in ("RGB", "RGBA"):
        raise BakeError("The bake's base colour map is not an RGB picture, so it cannot be painted. Bake the "
                        "prop again.")
    pixels = np.fromfile(raw, np.uint8)
    if pixels.size != width * height * 4:
        raise BakeError("The painting arrived incomplete, so it was not saved. Save it again.")
    painting = pixels.reshape(height, width, 4)
    if not np.all(painting[..., 3] == shown):
        raise BakeError("The painting was not saved: it was not painted on this bake's colours (the page lost its "
                        "copy of them, as when the graphics card is reset), and saving it would have overwritten them. "
                        "Reload the page (F5) and paint again.")
    colour = painting[..., :3]
    image = colour if alpha is None else np.concatenate([colour, alpha[..., None]], axis=-1)
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(buffer, "PNG", compress_level=PNG_COMPRESS_LEVEL)
    png = buffer.getvalue()

    out.progress(0.4, "Saving the painting…")
    files: dict[Path, bytes] = {}
    if unpainted and not Path(unpainted).exists():
        files[Path(unpainted)] = basecolor.read_bytes()      # put in place before the map it keeps is replaced
    files[basecolor] = png
    files[model] = glb.set_base_color_image(model.read_bytes(), png)
    if fbx_file.is_file():
        out.progress(0.6, "Saving the painting…")
        embedded = fbx.set_embedded_map(fbx_file.read_bytes(), png)
        if embedded is not None:
            files[fbx_file] = embedded
    out.progress(0.9, "Saving the painting…")
    replace_files(files)
    out.progress(1.0, "Saved.")
    return {"files": [path.relative_to(bake_dir).as_posix() for path in files]}


COMMANDS = {"convert": run_convert, "repaint": run_repaint}
WHAT = {"convert": "model", "repaint": "painting"}


def main(argv: list[str], stdout: TextIO = sys.stdout) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s: [%(name)s] %(message)s")
    out = Output(stdout)
    command = "convert"
    try:
        if len(argv) != 1:
            raise RequestError("usage: python export_worker.py <request.json>")
        request = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
        if not isinstance(request, dict) or request.get("cmd") not in COMMANDS:
            raise RequestError(f"'cmd' must be one of {', '.join(COMMANDS)}")
        command = request["cmd"]
        result = COMMANDS[command](request, out)
    except BakeError as exc:
        out.send({"error": str(exc), "kind": "user"})
        return 1
    except MemoryError:
        out.send({"error": f"The computer ran out of memory while saving the {WHAT[command]}. Close other "
                           f"programs and try again.", "kind": "user"})
        return 1
    except Exception as exc:  # a bug: the artist gets one line, the log the traceback
        traceback.print_exc(file=sys.stderr)
        out.send({"error": f"Saving the {WHAT[command]} failed ({type(exc).__name__}: {exc}). The details are in "
                           f"code/logs/hub.log.", "kind": "internal"})
        return 1
    out.send({"result": result})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
