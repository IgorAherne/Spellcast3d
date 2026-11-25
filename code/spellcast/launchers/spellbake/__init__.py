"""spellbake: bakes a textured high poly onto a UV-unwrapped low poly, on the CPU.

It runs inside the Instant Meshes venv (numpy, trimesh, Pillow, xatlas, embreex), in a
child process per bake (launchers/bake_worker.py), and never imports the hub package.

Conventions every module shares:
- UVs are v-up, as OBJ stores them; a glTF file stores (u, 1 - v).
- Image row 0 is the top of the map (v = 1): texel (column c, row r) of an R x R map is
  centred on u = (c + 0.5) / R, v = 1 - (r + 0.5) / R.
- The tangent space is MikkTSpace computed on the v-up UVs, and its sign w is exported
  unchanged; normal maps are OpenGL (+Y along +v).
- Positions are float64 world units; ``diag`` is the diagonal of the high poly's bounding box.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path

# A just-written file can be held briefly by a virus scanner or the search indexer.
REPLACE_ATTEMPTS = 40


class BakeError(Exception):
    """A problem with the artist's meshes or settings. The message is written for an artist."""


def replace_files(files: Mapping[Path, bytes]) -> None:
    """Write each file beside itself under a temporary name, then put them all in place (os.replace,
    retried while Windows holds the old file): no file is ever half-written, and the window in which
    some are new and some old stays as short as it can be."""
    temporary = {path: path.with_name(f".{path.name}.{os.getpid()}.tmp") for path in files}
    try:
        for path, data in files.items():
            temporary[path].write_bytes(data)
        for path, tmp in temporary.items():
            for attempt in range(REPLACE_ATTEMPTS):
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:
                    if attempt == REPLACE_ATTEMPTS - 1:
                        raise
                    time.sleep(0.05)
    finally:
        for tmp in temporary.values():
            tmp.unlink(missing_ok=True)
