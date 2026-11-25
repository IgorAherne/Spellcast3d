"""TriFlow tool server for Spellcast3D. Runs INSIDE the triflow venv (code/venvs/triflow, Python 3.11).

    python triflow_server.py --repo <code/triflow> --port P [--host 127.0.0.1] [--projects-root <projects>]
                             [--attn auto|flash_attn|xformers|sdpa] [--dataflow auto|gather_scatter]

LICENSE OF THIS FILE: this file and launchers/triflow_stubs/ are MIT-licensed (Copyright (c) Spellcast3D's
authors; the MIT terms are in "README - LICENSE.txt"), NOT AGPL like the rest of Spellcast3D, because this process
loads TriFlow's code, which is under the ADPNCL 1.0 (non-commercial; code/triflow/LICENSE). The other Spellcast3D
modules this process loads are MIT too, for the same reason (each says so in its header): launchers/_loopback_guard.py
and launchers/attn_shim.py. TriFlow runs in its own process (this one), never inside the hub.

TriFlow (this project's fork https://github.com/IgorAherne/triflow; weights HF lihcxr/TriFlow) turns a high poly into a clean,
quad-friendly triangle mesh with about the triangle count asked for. This server serves the hub
(adapters/triflow.py):

    GET  /spellcast/health     -> {"ready", "attention", "gpu", "weights_read"} (ready false once CUDA broke)
    GET  /spellcast/progress   -> {"phase": "idle"|"working", "fraction", "message"}
    POST /spellcast/retopo     {mesh_path, out_path, faces, quad_ratio, qem_threshold, seed, name?}
                               -> {obj_path, vertices, faces, seconds, seed, warning, note, target_faces, input_faces,
                                   watertight, winding_consistent, outward, weights_from_disk, gpu_peak_mb,
                                   gpu_tensors_mb, ram_peak_mb}
                               errors: {"detail": <for an artist>, "kind": input|oom|ram|missing_file|busy|restart|
                               error}; "restart": an error left CUDA unusable, so the process stops being ready and
                               exits a second later (the hub starts it again for the next prop); "busy": a mesh the
                               hub gave up on still runs (TriFlow can't be stopped in the middle of one)
    POST /spellcast/shutdown   {} graceful stop, as CTRL_BREAK
One prop at a time; TriFlow cannot be stopped in the middle of a prop.

The output OBJ holds triangles only, no UVs (the hub unwraps them in Instant Meshes), in the SAME frame and scale
as the input mesh (y up, front +Z): TriFlow works in a 512-voxel grid and writes its result there; it is mapped
back here (grid_frame).

TriFlow's checkout is this project's fork (github.com/IgorAherne/triflow), whose changes to TriFlow are
documented in its CHANGES.md as its licence asks (every change dated, and sent to AUDI AG); nothing here edits its
tree at run time, and nothing is written into its folder: bytecode goes to the venv (sys.pycache_prefix). Each prop
calls `inference.run_inference(in_dir, out_dir, 0, 1, faces, qem_threshold, quad_ratio)` with a fresh folder holding
one clean OBJ of the prop, after seeding torch (TriFlow takes no seed). What this file changes lives in third-party
libraries only:
- safetensors.torch.load_file is memoised (WeightCache): run_inference loads its three weight files on every call;
  they are read from disk once per process (at start) and kept in RAM, so a group of props costs one read.
- meshiki's quadrangulation is skipped (skip_unused_quadrangulation): TriFlow runs it on every input only to
  measure a quad ratio it ignores when given one, and this launcher always gives one (a minute per prop).
- attention: flash_attn on RTX 30xx+, torch SDPA through attn_shim.py below that (TRELLIS and Direct3D-S2 accept
  only flash_attn or xformers, and Direct3D-S2 imports flash_attn on every GPU, so below RTX 30xx `flash_attn` is
  the shim: the real one has no kernels there). xformers 0.0.32 is not used by default: it refuses to import next
  to flash_attn 2.8.3 (and next to the shim).
- below RTX 20xx, torchsparse's global convolution config uses its gather-scatter dataflow (its default
  implicit-GEMM kernels need sm_75+ in fp16), through torchsparse's own public API.
- NVIDIA kaolin is not installed: triflow_stubs/kaolin gives FlexiCubes the one function it imports (never run).
- on a card spconv has no prebuilt GEMM kernels for (sm_60, 70, 75, 80, 86, 89 and 90 are built into spconv-cu126
  2.3.8; a GTX 10xx, sm_61, is not), spconv would compile them at run time through NVRTC, which cannot work from the
  wheel (no headers shipped, no include path given: a GTX 1080 fails at TRELLIS' first convolution). There spconv's
  Native algorithm runs with the products on torch.mm (spconv_shim.py, SPCONV_ALGO=native for TRELLIS), as
  Pixal3D's fork does for its own decoders.
SPARSE_BACKEND stays unset: TRELLIS' flow model needs spconv, Direct3D-S2's VAEs need torchsparse (their weights
are stored in each library's own layout).

Only the stdlib, fastapi and pydantic at module level: the hub's tests import this file without torch or numpy.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import importlib.util
import io
import os
import random
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import spconv_shim                                        # launchers/ (this script's folder)
from _loopback_guard import LoopbackGuard, is_loopback

# The triangles the hub can ask for: two per vertex of 0.25 to 99 thousand (it asks a smaller size for the least).
# Keep in sync with adapters/triflow.py.
MIN_FACES = 500
MAX_FACES = 198_000
MESH_SUFFIXES = (".glb", ".gltf", ".obj", ".ply", ".stl", ".off")
ATTENTION = ("auto", "flash_attn", "xformers", "sdpa")

WEIGHTS = ("flow_model.safetensors", "sdf_vae.safetensors", "nvv_vae.safetensors")   # TriFlow reads checkpoints/
STUBS = Path(__file__).resolve().parent / "triflow_stubs"
WORK_PREFIX = "spellcast-triflow-"     # per prop, in TEMP: one clean OBJ for TriFlow to read, and its result
# TriFlow's nested checkouts, put on the path as its setup.sh does (their packages: trellis, direct3d_s2).
THIRD_PARTY = ("third_party/TRELLIS", "third_party/Direct3D-S2")

# The grid TriFlow's inference.py puts every mesh in (process_one_mesh res_fine=512, pad=1.5 coarse voxels of
# 512/64 = 8 fine ones): the longest side of the bounding box spans GRID - 2 * PAD voxels, centred in the grid.
GRID = 512
PAD = 12

# The warnings an artist gets with a mesh (it is still saved).
OVER_TARGET = 1.25             # triangles made / asked, above which the artist hears why
FRAME_LIMIT = 0.05             # how far the result's box may sit from the model's (share of the longest side)
OVER = ("TriFlow kept {made:,} triangles, more than the {asked:,} asked for: fewer would lose too much of this "
        "shape.")
HOLES = "The clean mesh has holes ({count:,} open edges). Try another Seed."
OPEN_INPUT = ("The clean mesh has holes ({count:,} open edges) where the 3D model has some too ({input:,}): TriFlow "
              "follows its surface.")
# Said quietly (the "note" of a result), not as a warning: holes no more than the 3D model's own, and fewer
# triangles than asked - TriFlow only takes triangles away from its own remesh of the model, so a size above that
# remesh's count comes out with all of it (a thin rifle: 96,552 triangles, whatever larger number is asked).
OPEN_LIKE_INPUT = ("Open where the 3D model is open ({count:,} edges; the model has {input:,}): TriFlow follows its "
                   "surface.")
UNDER_TARGET = 0.9
UNDER = "TriFlow made {made:,} triangles where {asked:,} were asked: it has no more for this model's surface."
FLIPPED = "Some faces of the clean mesh point inward. Try another Seed."
OFF_FRAME = ("The clean mesh does not line up with the 3D model ({share:.0%} of its size off). Make it again; if "
             "it happens again, check the TriFlow log (Settings > Debugging).")

MODELS_MISSING = ("TriFlow's AI model files are missing (they belong in the models\\triflow folder of "
                  "Spellcast3D). Close Spellcast3D and start it again: it puts back what is missing.")

# (phase label, fraction at its start, at its end), in order; the fractions were timed on an RTX 3060 (Pixal3D props
# of ~290k faces: 50-90 s each; the voxels 40%, the sampling 20-30%, the mesh 20-35%).
PHASES = {
    "read": ("Reading the 3D model", 0.00, 0.03),
    "setup": ("Setting up TriFlow's models", 0.03, 0.08),
    "voxels": ("Measuring the shape", 0.08, 0.50),
    "sampling": ("Finding the new edge flow", 0.50, 0.72),
    "mesh": ("Building the clean mesh", 0.72, 0.98),
    "save": ("Saving the clean mesh", 0.98, 1.00),
}
# What TriFlow prints on its way (stdout / its tqdm bar on stderr) -> the phase it has reached.
PROCESSING = re.compile(r"^\(\d+/\d+\) Processing ")
SAMPLING = re.compile(r"^Sampling:\s*\d+%.*?\b(\d+)/(\d+)\b")
FLOW2MESH = "[flow2mesh]"
FLOW2MESH_DONE = (("bilateral", 0.4), ("watershed", 0.85), ("qem", 1.0))   # its stage timings -> share of "mesh"


class RetopoRequest(BaseModel):
    mesh_path: str                                   # the high poly (.glb/.obj/...), inside the projects folder
    out_path: str                                    # the clean mesh (.obj), inside the projects folder
    faces: int = Field(4000, ge=MIN_FACES, le=MAX_FACES)    # target triangle count
    quad_ratio: float = Field(0.95, ge=0.0, le=1.0)         # how much of the result pairs up into quads
    qem_threshold: float = Field(12.0, gt=0.0, le=1000.0)   # the largest error the final simplification allows
    seed: int = 0                                    # < 0 = random
    name: str = ""                                   # the file name TriFlow sees (the prop id), for its log


class InputProblem(Exception):
    """Something about the request or the mesh the artist has to fix (kind "input")."""


class Progress:
    """What GET /spellcast/progress reports. Moved on by the launcher and by what TriFlow prints (feed)."""

    def __init__(self) -> None:
        self.phase = "idle"
        self.fraction = 0.0
        self.message = "Idle"
        self._lock = threading.Lock()

    def start(self) -> None:
        self.phase, self.fraction, self.message = "working", 0.0, "Starting"

    def step(self, key: str, done: float = 0.0, detail: str = "") -> None:
        """Enter or advance a phase of PHASES (done: 0..1 of it)."""
        label, start, end = PHASES[key]
        with self._lock:
            self.fraction = max(self.fraction, start + (end - start) * min(max(done, 0.0), 1.0))
            self.message = f"{label} ({detail})" if detail else label

    def feed(self, text: str) -> None:
        """One line (or tqdm redraw) of TriFlow's output."""
        if self.phase != "working":
            return
        line = text.strip()
        if PROCESSING.match(line):
            self.step("voxels")
            return
        sampled = SAMPLING.match(line)
        if sampled:
            step, total = int(sampled.group(1)), int(sampled.group(2))
            if total > 0:
                self.step("sampling", step / total, f"{step}/{total}")
                if step >= total:
                    self.step("mesh")
            return
        if line.startswith(FLOW2MESH):
            # it times its stages: the NVV filter, the watershed, the QEM simplification
            low = line.lower()
            for word, done in FLOW2MESH_DONE:
                if word in low:
                    self.step("mesh", done)
                    return

    def stop(self) -> None:
        self.phase, self.fraction, self.message = "idle", 0.0, "Idle"

    def to_dict(self) -> dict[str, Any]:
        return {"phase": self.phase, "fraction": round(self.fraction, 3), "message": self.message}


class ProgressTee(io.TextIOBase):
    """sys.stdout / sys.stderr: everything still goes to the log, and each finished line or tqdm redraw ('\\r')
    also to Progress.feed."""

    def __init__(self, stream: Any, progress: Progress) -> None:
        super().__init__()
        self._stream = stream
        self._progress = progress
        self._pending = ""

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._pending += text
        *done, self._pending = re.split(r"[\r\n]", self._pending)
        for line in done:
            if line:
                try:
                    self._progress.feed(line)
                except Exception:
                    pass   # progress is cosmetic
        if len(self._pending) > 4096:
            self._pending = self._pending[-4096:]
        return len(text)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:     # type: ignore[override]
        return getattr(self._stream, "encoding", "utf-8")

    def fileno(self) -> int:
        return self._stream.fileno()


class WeightCache:
    """A memoised safetensors.torch.load_file: each file is read from disk once per process into memory of its own
    and its tensors are kept (TriFlow's run_inference loads its three checkpoints on every call; load_state_dict
    copies from them, so they are never changed). A changed file (size or time) is read again."""

    def __init__(self, load: Callable[..., Any], log: Callable[[str], None] = print) -> None:
        self._load = load
        self._log = log
        self._kept: dict[tuple[Any, ...], tuple[tuple[int, int], dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self.reads = 0            # files read from disk so far

    def __call__(self, filename: Any, device: Any = "cpu") -> dict[str, Any]:
        path = os.path.normcase(os.path.abspath(os.fspath(filename)))
        stat = os.stat(path)
        signature = (stat.st_size, stat.st_mtime_ns)
        key = (path, str(device))
        with self._lock:
            kept = self._kept.get(key)
            if kept is None or kept[0] != signature:
                started = time.monotonic()
                mapped = self._load(filename, device=device)
                # safetensors maps the file and reads a page only when it is first touched: copy the tensors into
                # memory of their own now, so the file is read once, here, and not held open afterwards
                tensors = {name: _own_copy(tensor) for name, tensor in mapped.items()}
                del mapped
                self._kept[key] = kept = (signature, tensors)
                self.reads += 1
                self._log(f"[spellcast] Read {os.path.basename(path)} from disk ({stat.st_size / 1e6:,.0f} MB, "
                          f"{time.monotonic() - started:.1f} s); kept in RAM for the next props.")
        return dict(kept[1])      # a new dict each time: the caller may change it, never the tensors


def _own_copy(tensor: Any) -> Any:
    clone = getattr(tensor, "clone", None)
    return clone() if callable(clone) else tensor


def skip_unused_quadrangulation(mesh_class: Any) -> bool:
    """Make meshiki's Mesh.quadrangulate (a third-party library) do nothing. Returns whether it was replaced.

    TriFlow's process_one_mesh ends with compute_quad_ratio(), which quadrangulates the whole decimated input with
    meshiki to measure its quad ratio - single-threaded, 60-90 s of every prop's ~2 minutes (measured on 300k-face
    Pixal3D meshes). run_inference uses that ratio only when it is given no quad ratio, and this launcher always
    gives one (RetopoRequest.quad_ratio), so the value is thrown away: skipping the work changes nothing TriFlow
    makes (the ratio then reads 0). TriFlow uses meshiki for nothing else at inference (fps: augmentation only)."""
    original = getattr(mesh_class, "quadrangulate", None)
    if original is None:
        return False
    if getattr(original, "_spellcast_skipped", False):
        return True

    def quadrangulate(self: Any, *args: Any, **kwargs: Any) -> None:
        return None

    quadrangulate._spellcast_skipped = True   # type: ignore[attr-defined]
    mesh_class.quadrangulate = quadrangulate
    return True


def install_weight_cache(log: Callable[[str], None] = print) -> WeightCache:
    """Replace safetensors.torch.load_file by a WeightCache around it. Call BEFORE `import inference` (it binds
    the name at import). Idempotent."""
    import safetensors.torch

    current = safetensors.torch.load_file
    if isinstance(current, WeightCache):
        return current
    cache = WeightCache(current, log)
    safetensors.torch.load_file = cache
    return cache


# ---- environment (before torch and TriFlow are imported) ---------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TriFlow server for Spellcast3D")
    p.add_argument("--repo", required=True, help="TriFlow checkout (code/triflow) with its third_party checkouts")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--projects-root", default=None,
                   help="the hub's projects folder: /spellcast/retopo reads and writes only inside it")
    p.add_argument("--attn", choices=ATTENTION, default=None,
                   help="default: $ATTN_BACKEND, else auto (flash_attn on RTX 30xx+, torch SDPA below; xformers "
                        "only when asked for)")
    p.add_argument("--dataflow", choices=("auto", "gather_scatter"), default="auto",
                   help="torchsparse's convolutions: auto = gather-scatter below RTX 20xx only (gather_scatter "
                        "tries the older GPUs' path on a newer one)")
    p.add_argument("--spconv", choices=("auto", "torch"), default="auto",
                   help="spconv's convolutions (TRELLIS'): auto = their products on torch.mm only on a card spconv "
                        "has no prebuilt GEMM kernels for (a GTX 10xx); torch tries that path on any card")
    return p.parse_args(argv)


def search_path(repo: Path) -> list[str]:
    """What goes in front of sys.path: TriFlow, its TRELLIS and Direct3D-S2 checkouts, then the kaolin stand-in
    (adapters/triflow.py puts the same on PYTHONPATH)."""
    return [str(repo), *(str(repo / part) for part in THIRD_PARTY), str(STUBS)]


def configure_env(repo: Path) -> None:
    """Environment TriFlow's libraries read when they are imported; the working folder TriFlow expects."""
    for name in ("SPARSE_BACKEND", "SPARSE_ATTN_BACKEND", "SPARSE_CONV_BACKEND"):
        os.environ.pop(name, None)      # both sparse libraries keep their own default (see the module doc)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")        # nothing is downloaded at run time
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # torch's CPU allocator gives freed memory back to Windows only after a delay otherwise (pixal3d_server.py)
    os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
    # torch's default CUDA cache. On a 12 GB card a large prop (2M voxels) makes the cache hold ~17 GB for ~8.5 GB of
    # tensors, the rest in shared system memory (slower, but it finishes); garbage_collection_threshold:0.8 +
    # max_split_size_mb:512 did not lower that and ran out of memory beside another program (expandable_segments,
    # which would, does not exist on Windows).
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ.setdefault("TRITON_CACHE_DIR", str(Path(sys.prefix) / "triton-cache"))
    # numba's cache ignores the pycache prefix below: it would write its .nbi/.nbc files into TriFlow's __pycache__
    # folders (and the bilateral filter's kernel compiles for 4-6 s in every process without a cache)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(sys.prefix) / "numba-cache"))
    # TriFlow's folder stays byte-identical: its bytecode goes into the venv.
    if sys.pycache_prefix is None:
        sys.pycache_prefix = str(Path(sys.prefix) / "pycache")
    for entry in reversed(search_path(repo)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    # inference.py reads checkpoints/ and Hydra's data/ paths relative to the working folder
    os.chdir(repo)


def choose_attention(asked: str | None, capability: tuple[int, int],
                     xformers_installed: bool = False) -> tuple[str, bool, bool]:
    """(ATTN_BACKEND for TRELLIS / Direct3D-S2, whether `flash_attn` is attn_shim.py, whether torchsparse uses its
    gather-scatter dataflow) for the GPU's compute capability.

    flash_attn's kernels need sm_80+. Below that, torch's own attention (SDPA) through the shim, whose module is
    named flash_attn, so the backend is "flash_attn": Direct3D-S2 imports flash_attn on every GPU, and the shim also
    spares loading the real 1 GB library for nothing there. xformers only when asked for and installed (it needs
    the real flash_attn importable: it refuses the shim, and 2.8.3 needs its version check turned off, see main)."""
    asked = asked if asked in ATTENTION else "auto"
    gather_scatter = capability < (7, 5)
    if asked == "xformers" and xformers_installed and capability >= (7, 5):
        return "xformers", False, gather_scatter
    if capability >= (8, 0) and asked != "sdpa":
        return "flash_attn", False, gather_scatter
    return "flash_attn", True, gather_scatter


def use_gather_scatter() -> None:
    """torchsparse's convolutions through its gather-scatter dataflow (its public global config), for GPUs below
    sm_75: the default implicit-GEMM kernels need tensor cores there, and TriFlow's VAEs run in float16."""
    import torchsparse.nn.functional as F
    from torchsparse.nn.functional.conv import conv_config

    config = conv_config.get_default_conv_config()
    config = type(config)(config)       # a copy: the default object is shared
    config.dataflow = F.Dataflow.GatherScatter
    conv_config.set_global_conv_config(config)


# ---- one prop ----------------------------------------------------------------------------------------------------

def grid_frame(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> tuple[tuple[float, ...], float]:
    """(centre, scale) of the grid frame TriFlow gives a mesh whose bounding box is lo..hi:
    grid = (v - centre) * scale + GRID / 2, so v = (grid - GRID / 2) / scale + centre."""
    size = max(h - l for l, h in zip(lo, hi))
    scale = (GRID - 2 * PAD) / size if size > 0 else 1.0
    return tuple((l + h) / 2 for l, h in zip(lo, hi)), scale


def safe_name(name: str) -> str:
    """A file name TriFlow can take (letters, digits, - and _), for its log."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name or "").strip("_")[:64]
    return cleaned or "mesh"


def warnings_for(made: int, asked: int, open_edges: int, winding_ok: bool, outward: bool,
                 frame_error: float, input_open: int = 0) -> str | None:
    """The artist's warning on a finished mesh, or None. input_open: the open edges of the 3D model it was made from
    (a Pixal3D model often has some: TriFlow keeps holes where the model has them; no more of them than the model
    has is only a note, note_for)."""
    notes = []
    if frame_error > FRAME_LIMIT:
        notes.append(OFF_FRAME.format(share=frame_error))
    if made > asked * OVER_TARGET:
        notes.append(OVER.format(made=made, asked=asked))
    if open_edges and not (input_open and open_edges <= input_open):
        notes.append(OPEN_INPUT.format(count=open_edges, input=input_open) if input_open
                     else HOLES.format(count=open_edges))
    if not winding_ok or not outward:
        notes.append(FLIPPED)
    return " ".join(notes) or None


def note_for(open_edges: int, input_open: int, made: int = 0, asked: int = 0) -> str | None:
    """The quiet note on a finished mesh, or None: it has clearly fewer triangles than asked, its holes are no
    more than its 3D model's own."""
    notes = []
    if made < asked * UNDER_TARGET:
        notes.append(UNDER.format(made=made, asked=asked))
    if open_edges and input_open and open_edges <= input_open:
        notes.append(OPEN_LIKE_INPUT.format(count=open_edges, input=input_open))
    return " ".join(notes) or None


def open_edge_count(faces: Any) -> int:
    """Edges of a triangle mesh used by one face only (its holes' rims)."""
    import numpy as np

    edges = np.sort(np.asarray(faces, dtype=np.int64)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, uses = np.unique(edges, axis=0, return_counts=True)
    return int((uses == 1).sum())


def read_mesh(path: Path) -> tuple[Any, Any]:
    """The prop's mesh as welded float64 vertices and int64 triangles: every node of a scene in place, vertices at
    one spot merged (a GLB splits them along its UV seams, which TriFlow would see as open edges), degenerate and
    repeated faces and unused vertices dropped."""
    import numpy as np
    import trimesh

    try:
        loaded = trimesh.load(str(path), force="mesh", process=False)
    except Exception as exc:
        raise InputProblem(f"The 3D model {path.name} could not be read ({type(exc).__name__}: {exc}).") from exc
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise InputProblem(f"The 3D model {path.name} has no faces. Make it again in the 3D stage.")
    mesh = trimesh.Trimesh(vertices=np.asarray(loaded.vertices, dtype=np.float64),
                           faces=np.asarray(loaded.faces, dtype=np.int64), process=False)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) < 16:
        raise InputProblem(f"The 3D model {path.name} has too few faces to clean up ({len(mesh.faces)}).")
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def write_obj(path: Path, vertices: Any, faces: Any) -> None:
    """Positions and triangles only, written through a temporary file (any letters in the path)."""
    import numpy as np

    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.obj")
    try:
        with open(tmp, "w", encoding="ascii", newline="\n") as f:
            f.write("# Spellcast3D TriFlow clean mesh (triangles, no UVs)\n")
            np.savetxt(f, vertices, fmt="v %.9g %.9g %.9g")
            np.savetxt(f, np.asarray(faces, dtype=np.int64) + 1, fmt="f %d %d %d")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class TriFlow:
    """TriFlow's inference module, imported once; retopo() makes one clean mesh. Runs on the GPU worker thread."""

    def __init__(self, repo: Path, weights: WeightCache) -> None:
        self.repo = repo
        self.weights = weights
        self.inference: Any = None
        self.ready = False
        self.count = 0

    def load(self) -> None:
        """Import TriFlow (the models are made by run_inference itself) and read the weights into RAM."""
        import inference   # TriFlow's inference.py, unmodified (from the working folder, see configure_env)

        self.inference = inference
        import meshiki

        if skip_unused_quadrangulation(meshiki.Mesh):
            print("[spellcast] Skipping meshiki's quadrangulation of each input: TriFlow measures the input's quad "
                  "ratio for when none is given, and Spellcast3D always gives one (saves about a minute per prop).",
                  flush=True)
        for name in WEIGHTS:
            self.weights(Path("checkpoints") / name, device="cpu")
        self.ready = True

    def retopo(self, req: RetopoRequest, seed: int, progress: Progress) -> dict[str, Any]:
        import numpy as np
        import torch
        import trimesh

        started = time.monotonic()
        reads_before = self.weights.reads
        progress.step("read")
        vertices, faces = read_mesh(Path(req.mesh_path))
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        centre, scale = grid_frame(tuple(lo), tuple(hi))
        name = safe_name(req.name or Path(req.mesh_path).parent.parent.name)
        work = Path(tempfile.mkdtemp(prefix=WORK_PREFIX))
        try:
            in_dir, out_dir = work / "in", work / "out"
            in_dir.mkdir()
            write_obj(in_dir / f"{name}.obj", vertices, faces)
            progress.step("setup", 0.0, "the first mesh after a start takes longer" if self.count == 0 else "")
            # TriFlow draws its noise from torch's global generator and takes no seed of its own
            random.seed(seed)
            np.random.seed(seed % 2 ** 32)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            # a quad ratio is always given, so the input's own one (meshiki's quadrangulation) is never used
            self.inference.run_inference(str(in_dir), str(out_dir), 0, 1, req.faces, req.qem_threshold,
                                         req.quad_ratio)
            made_file = out_dir / f"{name}.obj"
            if not made_file.is_file():
                raise RuntimeError("TriFlow finished without writing a mesh (its log above says why)")
            progress.step("save")
            made = trimesh.load(str(made_file), force="mesh", process=False)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self.count += 1
        cuda = torch.cuda.is_available()
        gpu_peak_mb = int(torch.cuda.max_memory_reserved() // 2 ** 20) if cuda else 0     # what the card held
        gpu_used_mb = int(torch.cuda.max_memory_allocated() // 2 ** 20) if cuda else 0     # of it, in tensors

        # back from TriFlow's grid to the model's own frame
        out_vertices = (np.asarray(made.vertices, dtype=np.float64) - GRID / 2) / scale + np.asarray(centre)
        out_faces = np.asarray(made.faces, dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=out_vertices, faces=out_faces, process=False)
        extent = float((hi - lo).max()) or 1.0
        frame_error = float(max(np.abs(mesh.bounds[0] - lo).max(), np.abs(mesh.bounds[1] - hi).max())) / extent
        open_edges = open_edge_count(out_faces)
        watertight = bool(mesh.is_watertight)
        winding_ok = bool(mesh.is_winding_consistent)
        # signed volume (holes and all): clearly negative only when the mesh came out inside out
        outward = winding_ok and bool(mesh.volume > -1e-6 * extent ** 3)
        out = Path(req.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_obj(out, out_vertices, out_faces)
        ram_peak_mb = _ram_peak_mb()
        input_open = open_edge_count(faces) if open_edges else 0
        print(f"[spellcast] {name}: {len(out_faces):,} triangles (asked {req.faces:,}), {len(out_vertices):,} "
              f"vertices, {'closed' if watertight else f'{open_edges:,} open edges'}, winding "
              f"{'consistent' if winding_ok else 'mixed'}{', outward' if outward else ', INWARD'}; frame off by "
              f"{frame_error:.2%}; GPU peak {gpu_peak_mb:,} MB ({gpu_used_mb:,} MB in tensors), RAM peak "
              f"{ram_peak_mb:,} MB; weights read from disk "
              f"this time: {self.weights.reads - reads_before}; {time.monotonic() - started:.0f} s.", flush=True)
        return {
            "obj_path": str(out),
            "vertices": int(len(out_vertices)),
            "faces": int(len(out_faces)),
            "seed": seed,
            "warning": warnings_for(len(out_faces), req.faces, open_edges, winding_ok, outward, frame_error,
                                    input_open),
            "note": note_for(open_edges, input_open, len(out_faces), req.faces),
            "target_faces": req.faces,
            "input_faces": int(len(faces)),
            "watertight": watertight,
            "winding_consistent": winding_ok,
            "outward": outward,
            "weights_from_disk": self.weights.reads - reads_before,
            "gpu_peak_mb": gpu_peak_mb,
            "gpu_tensors_mb": gpu_used_mb,
            "ram_peak_mb": ram_peak_mb,
        }


def clear_old_work(root: Path | None = None, older_than: float = 3600.0) -> int:
    """Delete the per-prop work folders (WORK_PREFIX*, in TEMP) a TriFlow that was stopped mid-prop left behind.
    Only folders older than `older_than` seconds: another TriFlow may be working in a newer one. -> how many."""
    root = Path(root or tempfile.gettempdir())
    count = 0
    now = time.time()
    for folder in root.glob(f"{WORK_PREFIX}*"):
        try:
            if folder.is_dir() and now - folder.stat().st_mtime > older_than:
                shutil.rmtree(folder, ignore_errors=True)
                count += 1
        except OSError:
            pass
    return count


def cuda_works() -> bool:
    """False when CUDA fails even a tiny job (after some errors its context stays broken for this process). Running
    out of memory for it is no such error (another program may fill the card for a moment): that counts as working."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return True
    try:
        torch.ones(1, device="cuda").add_(1).item()
        torch.cuda.synchronize()
        return True
    except Exception as exc:
        if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower():
            return True
        print(f"[spellcast] CUDA check failed: {type(exc).__name__}: {exc}", flush=True)
        return False


BUSY = ("TriFlow is still working on an earlier mesh, which the hub stopped waiting for. It starts again for the "
        "next prop.")
BROKEN = ("TriFlow stopped after an error that left the graphics card unusable for it. It starts again for the next "
          "prop.")


def _ram_peak_mb() -> int:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", info.rss) // 2 ** 20)
    except Exception:
        return 0


# ---- the HTTP side ---------------------------------------------------------------------------------------------

class GpuWorker:
    """One thread for everything that touches torch: the loading and every mesh, one at a time. torch's CPU
    allocator gives memory back to Windows only on the thread that took it (pixal3d_server.GpuWorker)."""

    def __init__(self) -> None:
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="triflow-gpu")

    def run(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return self._pool.submit(self._call, fn, args, kwargs).result()

    @staticmethod
    def _call(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        try:
            return fn(*args, **kwargs)
        finally:
            gc.collect()       # run_inference's models go with its frame; free their VRAM for the next prop
            torch = sys.modules.get("torch")
            try:
                if torch is not None and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()     # other programs share the card
            except Exception:
                pass


def error_response(status: int, detail: str, kind: str) -> JSONResponse:
    # kind: "input" | "oom" | "ram" | "missing_file" | "error" (adapters/triflow.py reads it)
    return JSONResponse({"detail": detail, "kind": kind}, status_code=status)


def failure_response(exc: BaseException) -> JSONResponse:
    if isinstance(exc, InputProblem):
        return error_response(422, str(exc), "input")
    text = str(exc)
    low = text.lower()
    if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in low:
        return error_response(500, "TriFlow ran out of GPU memory. Close other programs that use the graphics "
                                   "card and try again.", "oom")
    if isinstance(exc, MemoryError) or "not enough memory" in low or "paging file" in low:
        return error_response(500, "TriFlow ran out of system memory (RAM). Close other programs and try again.",
                              "ram")
    if isinstance(exc, FileNotFoundError):
        return error_response(500, f"A file TriFlow needs is missing: {exc.filename or text}. " + MODELS_MISSING,
                              "missing_file")
    short = text if len(text) <= 300 else text[:300] + "..."
    return error_response(500, f"TriFlow failed while making the clean mesh ({type(exc).__name__}: {short}). "
                               "Try again or try another Seed; if it keeps failing, restart TriFlow (Settings > "
                               "Debugging) and check its log.", "error")


def validate(req: RetopoRequest, projects_root: Path | None) -> str | None:
    """The problem with a request, in an artist's words, or None."""
    mesh, out = Path(req.mesh_path), Path(req.out_path)
    if not mesh.is_absolute() or not out.is_absolute():
        return "The mesh paths must be absolute."
    if projects_root is not None and not (mesh.resolve().is_relative_to(projects_root)
                                          and out.resolve().is_relative_to(projects_root)):
        return "Spellcast3D only makes clean meshes from and into its own projects folder."
    if mesh.suffix.lower() not in MESH_SUFFIXES:
        return f"TriFlow cannot read '{mesh.suffix}' files. Use one of: {', '.join(MESH_SUFFIXES)}."
    if not mesh.is_file():
        return f"The 3D model was not found: {mesh}. Make it again in the 3D stage."
    if out.suffix.lower() != ".obj":
        return f"The clean mesh is saved as an .obj file (got {out.name})."
    return None


def build_app(engine: Any, gpu: GpuWorker, progress: Progress, projects_root: Path | None,
              info: dict[str, Any], exit_later: Callable[[], None] | None = None) -> FastAPI:
    """The /spellcast/* routes around `engine` (a TriFlow, or anything with .ready, .weights.reads and
    .retopo(req, seed, progress)). exit_later(): what ends this process a moment after an answer that says CUDA broke
    (default: os._exit(3) a second later)."""
    app = FastAPI(title="TriFlow for Spellcast3D", docs_url=None, redoc_url=None, openapi_url=None)
    working = threading.Lock()      # one mesh at a time: a second request is refused, never queued behind the first
    state = {"broken": False}       # CUDA broke: no more meshes; the process exits
    exit_later = exit_later or (lambda: threading.Timer(1.0, os._exit, (3,)).start())

    @app.exception_handler(RequestValidationError)
    async def readable_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', ())[1:]) or 'request'}: {e.get('msg', '')}"
                             for e in exc.errors())
        return error_response(422, f"TriFlow rejected the settings ({problems}).", "input")

    @app.get("/spellcast/health")
    async def health() -> dict[str, Any]:
        return {"ready": bool(engine.ready), "weights_read": engine.weights.reads, **info}

    @app.get("/spellcast/progress")
    async def spellcast_progress() -> dict[str, Any]:
        return progress.to_dict()

    @app.post("/spellcast/shutdown")
    async def shutdown(_: dict[str, Any] = Body(...)) -> dict[str, bool]:
        """What CTRL_BREAK does, for a parent whose console events cannot reach us (imesh_server.py). The JSON
        body keeps other web pages out (a CORS preflight, which is refused)."""
        signal.raise_signal(getattr(signal, "SIGBREAK", signal.SIGTERM))
        return {"stopping": True}

    # A plain def route: FastAPI runs it in a worker thread, so health and progress keep answering meanwhile.
    @app.post("/spellcast/retopo")
    def retopo(req: RetopoRequest) -> JSONResponse:
        problem = validate(req, projects_root)
        if problem is not None:
            return error_response(400, problem, "input")
        if state["broken"]:
            return error_response(503, BROKEN, "restart")
        if not engine.ready:
            return error_response(503, "TriFlow is still loading. Try again in a moment.", "error")
        if not working.acquire(blocking=False):
            # the hub never sends two at once: the first is one it gave up on (a timeout), still running
            return error_response(409, BUSY, "busy")
        try:
            return run(req)
        finally:
            working.release()

    def run(req: RetopoRequest) -> JSONResponse:
        seed = req.seed if req.seed >= 0 else random.randint(0, 2 ** 31 - 1)
        print(f"[spellcast] Clean mesh of {req.mesh_path} ({req.faces} triangles, quads {req.quad_ratio:g}, "
              f"QEM threshold {req.qem_threshold:g}, seed {seed})", flush=True)

        def work() -> tuple[JSONResponse | None, dict[str, Any], float]:
            started = time.monotonic()
            progress.start()
            failure: JSONResponse | None = None
            result: dict[str, Any] = {}
            try:
                result = engine.retopo(req, seed, progress)
            except Exception as exc:
                if not isinstance(exc, InputProblem):
                    traceback.print_exc()
                failure = failure_response(exc)
            finally:
                progress.stop()
            return failure, result, time.monotonic() - started

        failure, result, seconds = gpu.run(work)
        if failure is not None:
            if not gpu.run(cuda_works):
                # A CUDA error can leave the GPU context unusable: not ready any more (health, and any request still
                # to come, say so), answer, then exit; the hub starts TriFlow again for the next prop.
                state["broken"] = True
                engine.ready = False
                print("[spellcast] The graphics card does not answer TriFlow any more after that error: TriFlow "
                      "stops now, and the hub starts it again for the next clean mesh.", flush=True)
                exit_later()
                return error_response(500, BROKEN + " Make this clean mesh again.", "restart")
            return failure
        print(f"[spellcast] Clean mesh saved to {result['obj_path']} in {seconds:.0f} s", flush=True)
        return JSONResponse({**result, "seconds": round(seconds, 1)})

    return app


# ---- main -------------------------------------------------------------------------------------------------------

def startup_hint(exc: BaseException) -> str:
    low = str(exc).lower()
    if isinstance(exc, ImportError):
        return ("TriFlow's Python environment is incomplete (" + str(exc) + "). Close Spellcast3D and double-click "
                "run-spellcast3d.bat again to repair it.")
    if isinstance(exc, MemoryError) or "not enough memory" in low or "paging file" in low:
        return ("TriFlow ran out of system memory (RAM) while loading its models (about 1.3 GB). Close other "
                "programs, then start TriFlow again.")
    if isinstance(exc, FileNotFoundError):
        return MODELS_MISSING
    return ("TriFlow could not start (the error is above). Start it again; if it keeps failing, close Spellcast3D "
            "and double-click run-spellcast3d.bat again to repair it.")


def missing_parts(repo: Path) -> list[str]:
    """What is missing from a TriFlow checkout: its nested checkouts, then the weights (checkpoints/)."""
    parts = ["inference.py", "third_party/TRELLIS/trellis/__init__.py",
             "third_party/TRELLIS/trellis/representations/mesh/flexicubes/flexicubes.py",
             "third_party/Direct3D-S2/direct3d_s2/modules/sparse/__init__.py"]
    parts += [f"checkpoints/{name}" for name in WEIGHTS]
    return [part for part in parts if not (repo / part).is_file()]


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).resolve()
    missing = missing_parts(repo)
    if missing:
        weights_only = all(part.startswith("checkpoints/") for part in missing)
        print(f"[spellcast] Missing in {repo}: {', '.join(missing)}. " + (
            MODELS_MISSING if weights_only else "Close Spellcast3D and double-click run-spellcast3d.bat again: "
                                                "it repairs TriFlow's files."), flush=True)
        return 2
    configure_env(repo)
    gpu = GpuWorker()
    progress = Progress()
    sys.stdout = ProgressTee(sys.stdout, progress)
    sys.stderr = ProgressTee(sys.stderr, progress)
    try:
        import torch

        if not torch.cuda.is_available():
            print("[spellcast] TriFlow needs an NVIDIA graphics card, and torch sees none.", flush=True)
            return 2
        capability = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        total_mb = torch.cuda.get_device_properties(0).total_memory // 2 ** 20
        backend, shim, gather_scatter = choose_attention(args.attn or os.environ.get("ATTN_BACKEND"), capability,
                                                         importlib.util.find_spec("xformers") is not None)
        os.environ["ATTN_BACKEND"] = backend
        if backend == "xformers":
            # xformers 0.0.32 accepts flash_attn up to 2.8.2 only and refuses to import otherwise; it uses its own
            # kernels here (below sm_80 flash_attn's are never picked)
            os.environ["XFORMERS_IGNORE_FLASH_VERSION_CHECK"] = "1"
        if shim:
            import attn_shim

            attn_shim.install_flash_attn_shim()
        gather_scatter = gather_scatter or args.dataflow == "gather_scatter"
        if gather_scatter:
            use_gather_scatter()
        torch_products = spconv_shim.torch_products_where_needed(capability, always=args.spconv == "torch")
        attention = "sdpa" if shim and backend == "flash_attn" else backend
        weights = install_weight_cache(lambda line: print(line, flush=True))
        print(f"[spellcast] Starting TriFlow from {repo} on {args.host}:{args.port}: {name} (sm_{capability[0]}"
              f"{capability[1]}, {total_mb:,} MB), attention {attention}"
              + (", torchsparse gather-scatter" if gather_scatter else "")
              + (", spconv's products on torch.mm" if torch_products else "") + ".", flush=True)
        clear_old_work()
        engine = TriFlow(repo, weights)
        gpu.run(engine.load)
    except Exception as exc:
        traceback.print_exc()
        print("[spellcast] " + startup_hint(exc), flush=True)
        return 1
    print("[spellcast] TriFlow is ready.", flush=True)

    import uvicorn

    info = {"attention": attention, "gpu": name}
    app = build_app(engine, gpu, progress, Path(args.projects_root).resolve() if args.projects_root else None, info)
    try:
        uvicorn.run(LoopbackGuard(app) if is_loopback(args.host) else app, host=args.host, port=args.port,
                    log_level="warning")
    except OSError as exc:
        print(f"[spellcast] TriFlow could not open port {args.port} ({exc}). Another program uses it: wait a moment "
              "and start TriFlow again, or change the TriFlow port in Settings.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
