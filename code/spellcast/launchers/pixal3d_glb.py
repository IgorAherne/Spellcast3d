"""A decoded Pixal3D prop -> its GLB, in five steps that alternate between the CPU and the card, so that a batch
overlaps one prop's CPU work with the next prop's decode (the staged engine's FinishSteps: pixal3d/docs/
staged_engine.md, "Finishing in steps").

This is o_voxel.postprocess.to_glb (Microsoft TRELLIS.2, MIT) as the launcher calls it - remesh=True,
remesh_band=1, remesh_project=0, the unit box around the origin, the grid of the resolution the prop was made at -
followed by what the launcher does with the mesh (the turn to face +Z, the streak check, the file). It is cut where
the work changes sides:

    cpu  bvh       the decoded mesh's BVH: cuBVH builds its tree on the CPU for seconds (without Python's lock, in
                   Spellcast3D's build of cumesh) while the GPU thread decodes the next prop. With a cumesh whose
                   cuBVH can be built without the card (`upload=False`, Spellcast3D's wheel from round F on) the
                   tree stays in RAM and takes nothing on the card here; with an older cumesh, which uploads at
                   construction, it is built here only when the engine has room for it on the card meanwhile, else
                   by geometry
    gpu  geometry  the remesh guided by that BVH (uploaded for the step, the card copy let go when it returns),
                   simplify, the clean-up, the degenerate faces, and the GPU half of CuMesh.uv_unwrap (the fast
                   clustering into charts) with the vertex normals
    cpu  unwrap    the rest of CuMesh.uv_unwrap: xatlas on the charts (always CPU code); then, Spellcast3D's own,
                   the atlas's small islands joined onto their neighbours (uv_islands) and the atlas packed again
    gpu  bake      the texels' places on the mesh (nvdiffrast in UV space), moved onto the decoded mesh (its BVH
                   again: the same tree uploaded once more, or built again with an older cumesh), the texture
                   volume sampled there
    cpu  export    the texture clean-up (cv2.inpaint), the trimesh, the turn, the streak check, the file

Each step is to_glb's (and uv_unwrap's) code in their order, with their dtypes, devices, arguments and
empty_cache() points; to_glb's verbose and tqdm lines are left out (the launcher runs it with verbose off), its bar
is reported through px._update_progress instead. What is new, besides the joined islands, is only at the cuts: a
GPU step ends by bringing what the next step reads to the CPU and freeing everything it put on the card, and the
step after takes it back. Those
moves copy values exactly, and the one computation that changed place - the vertex normals, which to_glb reads
after the unwrap - reads a mesh the unwrap does not touch.

Two more differences without a difference in the file. The BVH of the decoded mesh is built by cuBVH from the
mesh's rows in RAM, which the steps have anyway, instead of from a copy brought down from the card (CuMesh's init
and read copy the rows unchanged); and it is built once instead of twice: to_glb builds the same BVH again for the
bake, from the same rows, and both the build and the queries are deterministic (checked on a million queries).
With Spellcast3D's cumesh (`cuBVH(vertices, faces, upload=False)`, `.upload()`, `.release_device()`: see
code/release/wheels/cumesh) the tree is built on the worker thread without the card, each GPU step uploads it for
its own queries (a tenth of a second) and lets the card copy go when it returns, and the tree in RAM serves the
remesh and the bake alike: nothing is held on the card between the steps, the engine's room (state["room_mib"],
pixal3d/docs/staged_engine.md "Finishing in steps") is not needed, and the GPU thread never builds a tree. (Holding
the card copy instead was measured in round H: an upload is 0.07-0.12 s on a warm card, and one made on the worker
costs the card the same time, since cuBVH copies on the default stream every kernel of the engine runs on: at most
0.5 s a batch, for a gigabyte held through another prop's texture turn - not done.) With an older cumesh, whose
cuBVH uploads at construction, the tree is built on the worker only when the engine has room for it on the card
meanwhile and held there until the bake (HELD, state["card_mib"]), else built by geometry and again by bake, as
to_glb does.

The texture volume (the decode's voxel coordinates and attributes, 36 bytes a voxel, 0.1-0.4 GB at 1536) is read
by the bake alone, a texture turn or two after the hand-over, while up to three props are in their steps: in a batch
it waits in the prop's folder of the work folder instead of RAM (TextureVoxels: written by a thread of its own as it
is handed over, read back by another while the bake starts on the card), the same bytes either way.

Both paths use it: a batch hands finish_steps(...) to the engine (BatchHooks.finish), the reference path runs the
same steps one after the other (run_all), so the GLB is the same whichever made it.

Only the stdlib (and the launchers' own stdlib modules) at module level: the hub's tests import pixal3d_server,
which imports this file, without torch, numpy or PIL.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import importlib
import inspect
import io
import json
import os
import struct
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Callable

import pixal3d_guard                                        # launchers/ (this script's folder); stdlib only
import pixal3d_mv

# to_glb's bar, as its tqdm descriptions name the six points (pixal3d_server.PHASES maps them to the artist's words),
# and save_glb's report before it writes the file.
BARS = ("Extracting GLB", "Building BVH", "Cleaning mesh", "Parameterizing new mesh", "Sampling attributes",
        "Finalizing mesh")
SAVING = "Saving GLB"
# The hook's steps in their order (glb_steps makes the FinishSteps from them); the batch counts them behind the
# engine's three decode steps in "Building the 3D mesh · step k of 8".
STEPS = (("cpu", "bvh"), ("gpu", "geometry"), ("cpu", "unwrap"), ("gpu", "bake"), ("cpu", "export"))

# The launcher's arguments of to_glb (decoded_to_glb).
AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
REMESH_BAND = 1
REMESH_PROJECT = 0
# The remesh (narrow-band dual contouring on a 512 grid, three times coarser than a 1536 decode) tears the sheets
# that are thinner than its cells - a wing membrane where it meets a bone - and nothing closed those tears: the
# decoded mesh's hole filling runs before it. So the simplified mesh gets cumesh's hole filling once more, for holes
# up to this perimeter in the mesh's own units (the unit box around the origin; the decode's fill uses 0.03): a fan
# around one new vertex at the rim's mean (round K: a dragon's membrane had a pinhole, a small hole or a gap at the
# same spot in three equally good samples). PIXAL3D_FILL_AFTER_REMESH: a number, or off.
FILL_AFTER_REMESH_PERIMETER = 0.06
RASTER_CHUNK = 100000                 # faces rasterized at a time (to_glb's own figure)

# What each step leaves in `state` for the steps after it. A step never removes its own inputs (a GPU step may run
# twice after an out-of-memory); the step after it throws away what nothing reads any more.
DECODED = ("vertices", "faces")                         # finish_steps: the decoded mesh (CPU); bvh builds from it
VOXELS = ("voxels",)                                    # finish_steps: the texture volume (TextureVoxels: in RAM, or
#                                                         in the prop's work folder until the bake)
CHARTS = ("num_charts", "chart_vertices", "chart_faces", "chart_vertex_offset", "chart_face_offset", "chart_vmap",
          "unwrap_vertices", "normals")                 # geometry
SOURCE = ("source_vertices", "source_faces")            # geometry: the decoded mesh as CuMesh reads it (DECODED's
#                                                         tensors themselves), for the bake
HELD = ("bvh", "card_mib")                              # bvh or geometry, when the engine gave it room: the BVH kept
#                                                         for the steps after, and the MiB it takes on the card (the
#                                                         engine's protocol). With host trees, "bvh" alone: the tree
#                                                         in RAM, nothing on the card
MIB = 1 << 20
UNWRAPPED = ("out_vertices", "out_faces", "out_uvs", "out_normals")      # unwrap
BAKED = ("mask", "texels")                              # bake


BVH_BYTES_PER_FACE = 70         # cuBVH on the card: 48 B a triangle and 36 B a node, about 0.6 nodes a triangle
#                                 (measured: 604 MiB for 9.05 M faces, 153 for 2.30 M - within 1 % of this figure).
#                                 The card's free memory around a build is no measure of it: the engine's GPU thread
#                                 decodes the next prop meanwhile (2,300-2,400 MiB were read for a 415 MiB tree).

_host_trees: bool | None = None     # whether this cumesh builds a cuBVH without the card (decided at first use)


def fill_after_remesh_perimeter() -> float:
    """The bound of the hole filling after the remesh (FILL_AFTER_REMESH_PERIMETER), 0 for off."""
    text = os.environ.get("PIXAL3D_FILL_AFTER_REMESH", "").strip().lower()
    if text in ("off", "0", "false", "no"):
        return 0.0
    try:
        return float(text) if text else FILL_AFTER_REMESH_PERIMETER
    except ValueError:
        return FILL_AFTER_REMESH_PERIMETER


def fill_after_remesh(cumesh: Any, mesh: Any) -> Any:
    """The simplified remesh with its small holes closed (cumesh's fill_holes up to fill_after_remesh_perimeter),
    the fans' centres put at their rims' means (pixal3d_server.recentre_fans: cumesh leaves them anywhere along
    z after the first mesh of a process) and the fans oriented with the faces around them (fill_holes winds them
    as it finds each rim, many against their neighbours). Returns the mesh (a new CuMesh when vertices were
    added)."""
    bound = fill_after_remesh_perimeter()
    if bound <= 0 or not hasattr(mesh, "fill_holes"):      # off, or a test's fake backend without it
        return mesh
    try:
        before = int(mesh.num_vertices)
        mesh.fill_holes(max_hole_perimeter=bound)
        added = int(mesh.num_vertices) - before
        if added <= 0:
            return mesh
        from pixal3d_server import recentre_fans

        vertices, faces = mesh.read()
        recentre_fans(vertices, faces, before)
        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.unify_face_orientations()
        print(f"[spellcast] Closed {added:,} small holes the remesh left (fans up to {bound:g} of the box in "
              f"perimeter).", flush=True)
        return mesh
    except Exception as exc:        # a repair never costs the mesh: the remesh goes on as it was
        print(f"[spellcast] Note: could not close the remesh's small holes ({type(exc).__name__}: {exc}).", flush=True)
        return mesh


def join_islands(cumesh: Any, positions: Any, vmap: Any, faces: Any, uvs: Any) -> tuple[Any, Any, Any]:
    """xatlas's atlas (CPU tensors: vmap into positions, faces over the split vertices, uvs) with its small islands
    joined onto their neighbours (uv_islands) and packed again by xatlas from those UVs: far fewer crumbs, the
    same triangles in the same order. As it was when nothing joins, or PIXAL3D_JOIN_ISLANDS is off; the join never
    costs the mesh: if it fails, xatlas's atlas stays."""
    if os.environ.get("PIXAL3D_JOIN_ISLANDS", "").strip().lower() in ("off", "0", "false", "no"):
        return vmap, faces, uvs
    import numpy as np
    import torch
    import uv_islands

    started = time.perf_counter()
    try:
        joined_vmap, joined_faces, joined_uvs, joined = uv_islands.join_small_charts(
            positions.numpy(), vmap.numpy(), faces.numpy(), uvs.numpy())
        if joined == 0:
            return vmap, faces, uvs
        atlas = cumesh.Atlas()
        atlas.add_mesh(torch.from_numpy(np.ascontiguousarray(positions.numpy()[joined_vmap], dtype=np.float32)),
                       torch.from_numpy(np.ascontiguousarray(joined_faces, dtype=np.int32)),
                       uvs=torch.from_numpy(np.ascontiguousarray(uv_islands.packing_units(joined_uvs),
                                                                 dtype=np.float32)))
        atlas.compute_charts(use_input_mesh_uvs=True)
        atlas.pack_charts()
        xref, packed_faces, packed_uvs = atlas.get_mesh(0)
    except Exception as exc:
        print(f"[spellcast] Note: the small UV islands were left as xatlas made them ({type(exc).__name__}: {exc}).",
              flush=True)
        return vmap, faces, uvs
    print(f"[spellcast] Joined {joined:,} small UV islands onto their neighbours "
          f"({time.perf_counter() - started:.1f} s).", flush=True)
    return torch.from_numpy(joined_vmap)[xref.long()].to(vmap.dtype), packed_faces, packed_uvs


def host_trees() -> bool:
    """Whether the installed cumesh builds a cuBVH in RAM alone (`cuBVH(vertices, faces, upload=False)`, then
    `upload()` and `release_device()` on demand): Spellcast3D's wheel from round F on. False for a cumesh whose
    cuBVH uploads at construction (the one the fork shipped): the steps then hold or rebuild the tree as to_glb does."""
    global _host_trees
    if _host_trees is None:
        try:
            import cumesh

            _host_trees = ("upload" in inspect.signature(cumesh.cuBVH.__init__).parameters
                           and hasattr(cumesh.cuBVH, "upload") and hasattr(cumesh.cuBVH, "release_device"))
        except Exception:       # no cumesh at all: the steps fail where they first need it, with its own error
            _host_trees = False
    return _host_trees


def bvh_card_mib(faces: int) -> int:
    """MiB a cuBVH of `faces` triangles takes on the card (the figure the engine is told as the prop's hold)."""
    return int(faces) * BVH_BYTES_PER_FACE // MIB


# ---- the rays of the inner-copy search (pixal3d_server.find_inner_copy) ---------------------------------------------

# Points per ray-stab query of the inner-copy search (as the remesh itself queries its BVH). cuBVH seeds each point's
# stab rays from its index within its query, so this figure is part of the answer: changing it changes which faces
# count as inside.
STAB_QUERY = 2_000_000


def stab_by_index(bvh: Any) -> bool:
    """Whether this cuBVH answers the ray-stab's inside test for any subset of a query's points with each point's
    own seed (`raystab_inside(points, rng_index=...)`: Spellcast3D's cumesh from 0.0.1.post6). Without it the
    inner-copy search asks every point it might need, in the queries the seeds come from."""
    return callable(getattr(bvh, "raystab_inside", None))


def stab_inside(bvh: Any, centres: Any, rest: Any, asked: Any) -> Any:
    """Whether the ray-stab finds each point centres[rest[asked]] inside the decoded mesh (bvh), as a bool per point:
    the answer `bvh.signed_distance(centres[rest[i:i + STAB_QUERY]], mode="raystab")[0] < 0` gives it when the whole
    of `rest` is asked in queries of STAB_QUERY points, bit for bit, for the points asked alone. `asked`: positions in
    `rest`. Each point's rays are seeded from its index in the query it would be in (its position mod STAB_QUERY), and
    cuBVH.raystab_inside answers a ray by the point, the direction and the tree alone, so any subset in any order
    gets the same rays and the same verdicts (code/release/wheels/cumesh, patch_cumesh_win.py 8). One pass over the 32
    directions: a first pass of a few directions for all, with the points whose rays all hit packed together for the
    rest, was no faster (measured on a 3.6 M-face remesh: the faces asked are 95 % inside, and the stab's time is
    theirs, 64 rays each)."""
    import torch

    inside = torch.empty(asked.shape[0], dtype=torch.bool, device=centres.device)
    for i in range(0, asked.shape[0], STAB_QUERY):
        part = asked[i:i + STAB_QUERY]
        inside[i:i + STAB_QUERY] = bvh.raystab_inside(centres[rest[part]], rng_index=(part % STAB_QUERY).int())
    return inside


# The first slice of `rest` stab_until asks, in points; each slice after is four times as large. On props with a
# joined copy (a third to a half of `rest` inside) the first slice or the next finds INNER_PIECE_MIN_FACES; their faces
# near a seen one are asked for nothing (measured: 0.04 s at most, against ~0.8 s for the look a solid prop is spared).
STAB_PROBE = 8192


def stab_until(bvh: Any, centres: Any, rest: Any, enough: int) -> Any:
    """stab_inside over the first positions of `rest`, in slices of STAB_PROBE points growing four times each, until
    `enough` of the points asked are inside or all of `rest` is asked -> the verdicts of positions 0 .. n-1 of `rest`,
    a bool each (n: the points asked). The inner-copy search asks this before it looks for the faces seen from
    outside: a prop with fewer inside faces than a piece of the copy has needs no look (seen_from_outside takes ~1 s
    on a mesh of millions of faces)."""
    import torch

    verdicts = []
    done = found = 0
    size = STAB_PROBE
    while done < int(rest.shape[0]) and found < enough:
        part = torch.arange(done, min(int(rest.shape[0]), done + size), device=rest.device)
        verdicts.append(stab_inside(bvh, centres, rest, part))
        found += int(verdicts[-1].sum())
        done += int(part.shape[0])
        size *= 4
    return torch.cat(verdicts) if verdicts else torch.zeros(0, dtype=torch.bool, device=centres.device)


def rays_by_hit(bvh: Any) -> bool:
    """Whether this cuBVH tells whether rays hit the mesh by a traversal that stops at the first hit (`ray_hits`,
    ray_trace's face id >= 0 bit for bit: Spellcast3D's cumesh from 0.0.1.post6); else ray_trace is asked."""
    return callable(getattr(bvh, "ray_hits", None))


def staged_types() -> Any:
    """The fork's pixal3d.staged.types (FinishSteps, PropFailed). Imported when first needed: the fork is on the
    import path only once the server's main() has put it there, and its package imports torch."""
    return importlib.import_module("pixal3d.staged.types")


def _drop(state: dict, keys: tuple[str, ...]) -> None:
    for key in keys:
        state.pop(key, None)


# ---- the texture volume, in the prop's work folder until the bake ------------------------------------------------

VOXEL_FILE = "glb-voxels.safetensors"   # in the prop's folder of the batch's work folder. The engine's store deletes
#                                         files of this kind (.safetensors, .tmp) when it closes the prop's folder,
#                                         and the hub removes the whole work folder when the job is over
_DTYPES = {"torch.int32": "I32", "torch.float32": "F32", "torch.float16": "F16", "torch.bfloat16": "BF16",
           "torch.int64": "I64", "torch.uint8": "U8", "torch.bool": "BOOL"}

# The two threads of the volumes' file work (TextureVoxels): one writes them into the work folder as they are handed
# over, one reads them back for the bakes, so that a bake's read never waits behind the write of the prop decoded
# just before it (in a batch the hand-over of one prop comes right before the bake of another). Made when first
# needed, kept for the process.
_io: dict[str, concurrent.futures.ThreadPoolExecutor] = {}
_io_lock = threading.Lock()


def _io_pool(kind: str) -> concurrent.futures.ThreadPoolExecutor:
    """The "write" or the "read" thread."""
    with _io_lock:
        if kind not in _io:
            _io[kind] = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                              thread_name_prefix=f"spellcast-glb-{kind}")
        return _io[kind]


def _remove(*paths: Path) -> None:
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def write_tensors(path: Path, tensors: dict[str, Any]) -> None:
    """CPU tensors -> a safetensors file, written straight from their memory (no copy in RAM)."""
    import torch

    header: dict[str, Any] = {}
    start = 0
    for name, tensor in tensors.items():
        size = tensor.numel() * tensor.element_size()
        header[name] = {"dtype": _DTYPES[str(tensor.dtype)], "shape": list(tensor.shape),
                        "data_offsets": [start, start + size]}
        start += size
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for tensor in tensors.values():
            # the bytes as they lie in memory, whatever the dtype (numpy has no bfloat16)
            f.write(memoryview(tensor.detach().contiguous().view(-1).view(torch.uint8).numpy()))


def read_tensors(path: Path) -> dict[str, Any]:
    """A safetensors file written by write_tensors -> CPU tensors in memory numpy took (C runtime's heap: given back
    to Windows when they are let go of, on any thread - torch's own allocator keeps big blocks)."""
    import numpy as np
    import torch

    by_name = {code: getattr(torch, name.split(".")[1]) for name, code in _DTYPES.items()}
    found = {}
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
        base = 8 + length
        for name, info in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
            begin, end = info["data_offsets"]
            f.seek(base + begin)
            raw = np.fromfile(f, dtype=np.uint8, count=end - begin)
            if raw.size != end - begin:
                raise OSError(f"{path} is shorter than its header says")
            found[name] = torch.from_numpy(raw).view(by_name[info["dtype"]]).reshape(info["shape"])
    return found


class TextureVoxels:
    """A decoded prop's texture volume (coords int32 [N, 3] and attrs [N, C], 36 bytes a voxel: 0.1-0.4 GB at 1536)
    from the hand-over to the bake, its only reader, a texture turn or two later. Given the prop's folder in the
    batch's work folder, a writing thread puts it there as soon as it is handed over and lets go of it in RAM; the
    bake asks for it back as it starts (prefetch, on a reading thread while the card rasterizes and queries the
    BVH) and takes it before it samples (take). Read back in memory the C runtime gives back to Windows when it is
    let go of. Without a folder (the reference path, a check), or when the file cannot be written, it stays in RAM.
    The values are the same bytes either way.

    The file goes when this object does - the export lets go of it, or the engine clears a failed prop's state - and
    with the prop's folder, which the engine's store clears when the prop is over (a .safetensors file is the
    store's kind), and with the work folder, which the hub removes when the job is over."""

    def __init__(self, coords: Any, attrs: Any, folder: Path | str | None = None) -> None:
        self._lock = threading.Lock()
        self._tensors: dict[str, Any] | None = {"coords": coords, "attrs": attrs}
        self._read: concurrent.futures.Future[dict[str, Any]] | None = None
        self._written: concurrent.futures.Future[None] | None = None
        self.path: Path | None = None
        self.seconds = {"write": 0.0, "read": 0.0, "waited": 0.0}    # the file threads', and the bake's wait
        if folder is not None and Path(folder).is_dir():
            self.path = Path(folder) / VOXEL_FILE
            tmp = self.path.with_name(self.path.name + ".tmp")
            weakref.finalize(self, _remove, self.path, tmp)
            self._written = _io_pool("write").submit(self._write, self.path, tmp)

    @property
    def on_disk(self) -> bool:
        with self._lock:
            return self._tensors is None

    def _write(self, path: Path, tmp: Path) -> None:
        with self._lock:
            tensors = self._tensors
        started = time.perf_counter()
        try:
            write_tensors(tmp, tensors)
            os.replace(tmp, path)
            self.seconds["write"] = time.perf_counter() - started
        except Exception as exc:            # a full disk, a folder gone: the volume stays in RAM, as before
            _remove(tmp, path)
            print(f"[spellcast] Note: the texture volume stays in RAM ({type(exc).__name__}: {exc}).", flush=True)
            return
        del tensors
        with self._lock:
            self._tensors = None            # the file holds it now (take waits for this write before it reads)

    def prefetch(self) -> None:
        """Start reading the volume back (the bake, as it starts): the reading thread reads while the card works."""
        with self._lock:
            if self._tensors is None and self._read is None:
                self._read = _io_pool("read").submit(self._load)

    def _load(self) -> dict[str, Any]:
        started = time.perf_counter()
        tensors = read_tensors(self.path)
        self.seconds["read"] = time.perf_counter() - started
        return tensors

    def take(self) -> tuple[Any, Any]:
        """(coords, attrs) on the CPU, read back first when they are in the file; kept until this object goes, so
        a bake run again after an out-of-memory takes them again. A wait of more than half a second is noted in
        the log (a slow or busy disk)."""
        started = time.perf_counter()
        if self._written is not None:
            self._written.result()          # the write is long over by the bake; never ahead of it
        self.prefetch()
        with self._lock:
            read = self._read if self._tensors is None else None
        if read is not None:
            tensors = read.result()
            with self._lock:
                self._tensors = tensors
            waited = time.perf_counter() - started
            self.seconds["waited"] += waited
            if waited > 0.5:
                print(f"[spellcast] Note: the bake waited {waited:.1f} s for the texture volume from the work folder "
                      f"(written in {self.seconds['write']:.1f} s, read back in {self.seconds['read']:.1f} s).",
                      flush=True)
        with self._lock:
            return self._tensors["coords"], self._tensors["attrs"]


# ---- the export's threads ------------------------------------------------------------------------------------------

# The threads the export fills and encodes its textures on beside its own (OpenCV's inpaint and Pillow's WebP encoder
# let go of Python's lock): three channels of the base colour at once where the PC has the threads. Shared by every
# export of the process; the engine runs two CPU steps at most at once (runner.WORKERS), so the pool has room for
# two exports' channels where the PC has the threads (with three, two exports at once took 1.6-2.3 times as long
# each as one alone; with six, as long as one alone: the same bytes), and its number is bounded whatever runs.
EXPORT_THREADS = max(1, min(2 * 3, (os.cpu_count() or 2) - 1))
_export: concurrent.futures.ThreadPoolExecutor | None = None
_export_lock = threading.Lock()


def _export_pool() -> concurrent.futures.ThreadPoolExecutor:
    global _export
    with _export_lock:
        if _export is None:
            _export = concurrent.futures.ThreadPoolExecutor(max_workers=EXPORT_THREADS,
                                                            thread_name_prefix="spellcast-glb-export")
        return _export


def _webp(pixels: Any) -> bytes:
    """The WebP file trimesh's GLB export makes of a texture (gltf._append_image: `img.save(f, format="WEBP")` on
    the PIL picture of these pixels), made here ahead of it."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="WEBP")
    return buffer.getvalue()


def encoded_as_webp(image: Any, webp: concurrent.futures.Future[bytes]) -> Any:
    """`image` (a PIL picture) whose save(f, format="WEBP") - what trimesh's GLB export calls for each texture -
    writes the bytes `webp` encoded ahead from the same pixels (_webp) instead of encoding them again; any other
    save is Pillow's own."""
    from PIL import Image

    def save(fp: Any, format: str | None = None, **params: Any) -> None:
        if str(format).upper() == "WEBP" and not params and hasattr(fp, "write"):
            fp.write(webp.result())
            return
        Image.Image.save(image, fp, format, **params)

    image.save = save
    return image


GRID_SAMPLE_KERNEL = "flex_gemm.kernels.triton.grid_sample.indice_weighed_sum_fwd"


def reuse_grid_sample_tuning(rows: int, channels: int, points: int, corners: int = 8) -> bool:
    """flex_gemm tunes its trilinear grid_sample kernel (indice_weighed_sum_fwd_kernel) for every number of query
    points it meets: its key holds the bake's valid texel count exactly, new for every prop, so every bake ran 12
    configs x 13 timed launches and rewrote flex_gemm's JSON cache, on the GPU thread. Each output row of that
    kernel is its own sum over the 8 corners in fp32, whatever the config's tiling: the 12 configs give the same
    bytes (round J, overlap/autotune_bench.py). So the config tuned for the nearest count of the same table size
    (log2 of `rows`), `channels` and `corners` is given to this count of `points` before the call, which then runs at
    once. True when it gave one; nothing is done where flex_gemm is not laid out so, or nothing of that shape was
    tuned yet (the first bake of an install tunes as before)."""
    import ast
    import math

    try:
        kernel = importlib.import_module(GRID_SAMPLE_KERNEL).indice_weighed_sum_fwd_kernel
        cache = kernel.cache
    except Exception:           # another flex_gemm: it tunes as it does
        return False
    if not isinstance(cache, dict) or rows < 1 or points < 1:
        return False
    logn = int(math.log2(rows))
    best = None
    for key in list(cache):
        try:
            parts = ast.literal_eval(key)
        except (ValueError, SyntaxError):
            continue
        if not (isinstance(parts, tuple) and len(parts) >= 4 and parts[0] == logn and parts[2] == channels
                and parts[3] == corners and isinstance(parts[1], int)):
            continue
        if parts[1] == points:
            return True                                     # tuned (or given) already
        if best is None or abs(parts[1] - points) < abs(best[0][1] - points):
            best = (parts, key)
    if best is None:
        return False
    parts, key = best
    cache[str((parts[0], points, *parts[2:]))] = cache[key]
    return True


class _Glb:
    """One prop's steps and the settings they share. Nothing of the prop's data is kept here: it lives in the
    FinishSteps' state, which the engine clears when the prop is over."""

    def __init__(self, px: Any, attr_layout: dict, resolution: int, decimation_target: int, texture_size: int,
                 out_path: str, multi: bool, views: int, ss_retries: int, blocked: bool, rotation: Any,
                 streak_warning: Callable[[Any], str | None], inline: bool, types: Any, faces: int = 0) -> None:
        self.px = px
        self.faces = int(faces)                                 # of the decoded mesh
        # What its BVH takes on the card between two steps: nothing with host trees (the card copy lives only
        # inside the GPU step that uses it), else the tree the bvh step may be given room for.
        self.holds_mib = 0 if host_trees() else bvh_card_mib(faces)
        self.attr_layout = attr_layout
        self.resolution = int(resolution)
        self.decimation_target = int(decimation_target)
        self.texture_size = int(texture_size)
        self.out_path = str(out_path)
        self.multi = bool(multi)
        self.views = int(views)
        self.ss_retries = int(ss_retries)
        self.blocked = bool(blocked)
        self.rotation = rotation
        self.streak_warning = streak_warning
        self.inline = inline
        self.types = types

    def report(self, stage: str, step: int, total: int, always: bool = False) -> None:
        """to_glb's bar through the launcher's progress (looked up on px at call time: route_progress replaces
        it). Under the engine only the geometry step reports (`always`): it runs right after the prop's decode,
        while the prop is the one on the card, and the launcher gives a report to the prop on the card
        (pixal3d_batch.Batch.bar); the later steps run while other props are on the card, and the engine's own
        step events say where this prop is."""
        if always or self.inline:
            self.px._update_progress(stage, step, total)

    # ---- the steps ------------------------------------------------------------------------------------------------

    def bvh(self, state: dict) -> None:
        """CPU: the BVH of the decoded mesh for the remesh and the bake, built here on the worker thread while the
        GPU thread decodes the next prop. Reads DECODED. With host trees it is built in RAM alone (`upload=False`:
        the card is not touched; geometry and bake upload it for their own queries) and kept under "bvh" for both.
        With an older cumesh it uploads as it is built, so it is built here only when the engine's
        state["room_mib"] has room for it on the card meanwhile (HELD); else nothing, and geometry builds it. The
        hold is then written before the build, since the engine counts it from that key; a release meanwhile (a
        decode ran out of memory) takes the key away, and the tree built for nothing is dropped."""
        import cumesh

        if host_trees():
            state["bvh"] = cumesh.cuBVH(state["vertices"], state["faces"], upload=False)
            return
        room = int(state.get("room_mib") or 0)
        if room <= 0 or self.holds_mib > room:
            return
        state["card_mib"] = self.holds_mib
        made = cumesh.cuBVH(state["vertices"], state["faces"])
        if "card_mib" not in state:
            del made
            return
        state["bvh"] = made

    def geometry(self, state: dict) -> None:
        """GPU: the decoded mesh -> the cleaned, simplified remesh, split into charts. Reads DECODED and the BVH
        under "bvh" when the bvh step built it; writes SOURCE and CHARTS, all on the CPU, and leaves nothing on the
        card. With host trees the tree stays in RAM under "bvh" for the bake (its card copy is let go here); with
        an older cumesh it stays on the card (HELD) only when the engine's state["room_mib"] has room for it."""
        import torch

        if host_trees():
            bvh = state.get("bvh")          # the bvh step's tree in RAM; an attempt that failed uploads it again
            try:
                found = self._geometry(state["vertices"], state["faces"], bvh=bvh, host=True)
            finally:
                if bvh is not None:
                    bvh.release_device()    # the card copy, whatever happened; the tree in RAM serves the bake
            del bvh
        else:
            bvh = state.pop("bvh", None)    # the bvh step's, when it built one (an attempt that failed: used again)
            held = int(state.pop("card_mib", 0) or 0)
            found = self._geometry(state["vertices"], state["faces"], int(state.get("room_mib") or 0), bvh, held)
            del bvh
        state.update(found)                 # the card's copies died with _geometry's frame
        del found
        torch.cuda.empty_cache()

    def _geometry(self, decoded_vertices: Any, decoded_faces: Any, room_mib: int = 0, bvh: Any = None,
                  bvh_mib: int = 0, host: bool = False) -> dict:
        import numpy as np
        import torch
        import cumesh

        aabb = torch.tensor(np.array(AABB), dtype=torch.float32, device='cuda')
        grid_size = torch.tensor([self.resolution] * 3, dtype=torch.int32, device='cuda')

        vertices = decoded_vertices.cuda()
        faces = decoded_faces.cuda()
        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        vertices, faces = mesh.read()
        self.report(BARS[0], 1, 6, always=True)

        # The decoded mesh as CuMesh read it, in RAM: what the bake reads its triangles from and builds its BVH from
        # when it has to (to_glb reads it back from the card there). CuMesh's init and read copy the rows unchanged
        # (the same bytes, dtypes and shapes: checked on real props), so the rows the steps were handed serve: no
        # copy back from the card, and no second copy in RAM beside DECODED's until unwrap lets those go.
        source_vertices = decoded_vertices
        source_faces = decoded_faces
        # Build BVH for the current mesh to guide remeshing - unless the bvh step built it on the worker thread.
        # A host tree is uploaded for this step (a tenth of a second; what the upload takes on the card is inside
        # the hook's reserve, as the build's own upload was); an older cumesh's cuBVH uploads as it is built.
        if bvh is None:
            bvh = cumesh.cuBVH(source_vertices, source_faces, upload=False) if host else \
                cumesh.cuBVH(source_vertices, source_faces)
            bvh_mib = bvh_card_mib(int(faces.shape[0]))
        if host:
            bvh.upload()
        self.report(BARS[1], 2, 6, always=True)

        # --- Remeshing pipeline (to_glb's remesh=True branch) ---
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()
        # DC resolution capped to control peak VRAM: it only sets the intermediate shell's density, the texture is
        # baked from the volume at full voxel resolution regardless.
        dc_resolution = min(int(resolution), 512)
        # Free CuMesh internal buffers before the heavy remesh allocation
        del mesh
        torch.cuda.empty_cache()
        # Through the module attribute: the launcher's drop_inner_shells wraps it there.
        remesh_verts, remesh_faces = cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces,
            center = center,
            scale = (dc_resolution + 3 * REMESH_BAND) / dc_resolution * scale,
            resolution = dc_resolution,
            band = REMESH_BAND,
            project_back = REMESH_PROJECT,
            verbose = False,
            bvh = bvh,
        )
        # The decoded mesh is not needed until the bake. A host tree lets go of its card copy and stays in RAM for
        # the bake (geometry's caller keeps it in the state); an older cumesh's tree is kept on the card for the
        # bake when there is room for it meanwhile (else the bake builds it again), freed otherwise.
        del vertices, faces
        if host:
            bvh.release_device()
            held = {"bvh": bvh}
        else:
            held = {"bvh": bvh, "card_mib": bvh_mib} if 0 < room_mib and bvh_mib <= room_mib else {}
        del bvh
        torch.cuda.empty_cache()

        mesh = cumesh.CuMesh()
        mesh.init(remesh_verts, remesh_faces)
        del remesh_verts, remesh_faces
        mesh.simplify(self.decimation_target, verbose=False)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.unify_face_orientations()
        mesh = fill_after_remesh(cumesh, mesh)
        self.report(BARS[2], 3, 6, always=True)

        # --- UV parameterization ---
        # Remove degenerate (zero-area) faces that cause xatlas normal assertion failures
        verts_tmp, faces_tmp = mesh.read()
        v0 = verts_tmp[faces_tmp[:, 0]]
        v1 = verts_tmp[faces_tmp[:, 1]]
        v2 = verts_tmp[faces_tmp[:, 2]]
        face_areas = torch.cross(v1 - v0, v2 - v0, dim=1).norm(dim=1)
        degenerate = face_areas < 1e-10
        if degenerate.any():
            good_faces = faces_tmp[~degenerate]
            mesh = cumesh.CuMesh()
            mesh.init(verts_tmp, good_faces)
        del verts_tmp, faces_tmp, v0, v1, v2, face_areas, degenerate

        # CuMesh.uv_unwrap up to its xatlas part, with to_glb's compute_charts_kwargs
        mesh.remove_degenerate_faces()
        # 1. Fast mesh clustering
        mesh.compute_charts(
            threshold_cone_half_angle_rad=np.radians(90.0),
            refine_iterations=0,
            global_iterations=1,
            smooth_strength=1,
        )
        new_vertices, new_faces = mesh.read()
        (num_charts, charts_id, chart_vmap, chart_faces, chart_vertex_offset,
         chart_face_offset) = mesh.read_atlas_charts()
        chart_vertices = new_vertices[chart_vmap].cpu()
        chart_faces = chart_faces.cpu()
        chart_vertex_offset = chart_vertex_offset.cpu()
        chart_face_offset = chart_face_offset.cpu()
        chart_vmap = chart_vmap.cpu()
        # to_glb computes and reads the vertex normals after the unwrap; xatlas does not touch this mesh, so they
        # are the same now. The unwrap indexes them with its vertex map (out_normals).
        mesh.compute_vertex_normals()
        normals = mesh.read_vertex_normals().cpu()
        unwrap_vertices = new_vertices.cpu()
        return {"source_vertices": source_vertices, "source_faces": source_faces, "num_charts": int(num_charts),
                "chart_vertices": chart_vertices, "chart_faces": chart_faces,
                "chart_vertex_offset": chart_vertex_offset, "chart_face_offset": chart_face_offset,
                "chart_vmap": chart_vmap, "unwrap_vertices": unwrap_vertices, "normals": normals, **held}

    def unwrap(self, state: dict) -> None:
        """CPU: xatlas on the charts (the rest of CuMesh.uv_unwrap), on the CPU tensors geometry left. Reads
        CHARTS; writes UNWRAPPED."""
        import torch
        import cumesh

        _drop(state, DECODED)               # geometry's input; it is done
        num_charts = state["num_charts"]
        chart_vertices, chart_faces = state["chart_vertices"], state["chart_faces"]
        chart_vertex_offset, chart_face_offset = state["chart_vertex_offset"], state["chart_face_offset"]
        chart_vmap = state["chart_vmap"]

        # 2. Xatlas packing
        xatlas = cumesh.Atlas()
        chart_vmaps = []
        for i in range(num_charts):
            chart_faces_i = chart_faces[chart_face_offset[i]:chart_face_offset[i+1]] - chart_vertex_offset[i]
            chart_vertices_i = chart_vertices[chart_vertex_offset[i]:chart_vertex_offset[i+1]]
            chart_vmap_i = chart_vmap[chart_vertex_offset[i]:chart_vertex_offset[i+1]]
            chart_vmaps.append(chart_vmap_i)
            xatlas.add_mesh(chart_vertices_i, chart_faces_i)
        xatlas.compute_charts(verbose=False)
        xatlas.pack_charts(verbose=False)
        vmaps = []
        faces = []
        uvs = []
        cnt = 0
        for i in range(num_charts):
            vmap, x_faces, x_uvs = xatlas.get_mesh(i)
            vmaps.append(chart_vmaps[i][vmap])
            faces.append(x_faces + cnt)
            uvs.append(x_uvs)
            cnt += vmap.shape[0]
        vmaps = torch.cat(vmaps, dim=0)
        faces = torch.cat(faces, dim=0)
        uvs = torch.cat(uvs, dim=0)
        vmaps, faces, uvs = join_islands(cumesh, state["unwrap_vertices"], vmaps, faces, uvs)
        vertices = state["unwrap_vertices"][vmaps]
        normals = state["normals"][vmaps]   # to_glb: mesh.read_vertex_normals()[out_vmaps]
        state.update(out_vertices=vertices, out_faces=faces, out_uvs=uvs, out_normals=normals)

    def bake(self, state: dict) -> None:
        """GPU: the texture, sampled from the volume where each texel lies on the decoded mesh. Reads UNWRAPPED,
        SOURCE and VOXELS, and the BVH under "bvh" when geometry kept it (a host tree in RAM, or an older cumesh's
        on the card, HELD); writes BAKED on the CPU and leaves nothing on the card (the kept BVH is used up here)."""
        import torch

        _drop(state, CHARTS)                # unwrap's input; it is done
        self.report(BARS[3], 4, 6)          # to_glb's bar reaches this point once the unwrap is over
        state["voxels"].prefetch()          # read back from the work folder meanwhile, by a thread
        if host_trees():
            # The tree in RAM stays in the state until the bake has succeeded: an attempt that runs out of memory
            # (the engine runs the step again with what it kept on the card gone) uploads the same tree again
            # instead of building it anew on the GPU thread (9-13 s for a 15-21 M-face mesh, the card idle; seen
            # twice in round I's 7.2 GiB run). The same tree, so the same answers.
            bvh = state.get("bvh")
        else:
            bvh = state.pop("bvh", None)    # geometry's, when it was kept on the card; a failed attempt let it go
            state.pop("card_mib", None)
        try:
            found = self._bake(state, bvh)
        finally:
            if bvh is not None and host_trees():
                bvh.release_device()        # an attempt that failed: nothing of the tree stays on the card
        del bvh
        _drop(state, HELD)                  # the bake has used the tree up
        state.update(found)
        del found
        torch.cuda.empty_cache()
        self.report(BARS[4], 5, 6)

    @staticmethod
    def release(state: dict) -> None:
        """Let go of what the BVH kept for the bake holds on the card (the engine asks when a decode ran out of
        memory). An older cumesh's tree goes altogether, and the bake then builds it again, as to_glb does; a host
        tree only lets go of its card copy, which the bake uploads again."""
        if host_trees():
            bvh = state.get("bvh")
            if bvh is not None:
                bvh.release_device()
            state.pop("card_mib", None)
            return
        _drop(state, HELD)

    def _bake(self, state: dict, bvh: Any = None) -> dict:
        import numpy as np
        import torch
        import cumesh
        import nvdiffrast.torch as dr
        from flex_gemm.ops.grid_sample import grid_sample_3d

        texture_size = self.texture_size
        aabb = torch.tensor(np.array(AABB), dtype=torch.float32, device='cuda')
        grid_size = torch.tensor([self.resolution] * 3, dtype=torch.int32, device='cuda')
        voxel_size = (aabb[1] - aabb[0]) / grid_size
        out_vertices = state["out_vertices"].cuda()
        out_faces = state["out_faces"].cuda()
        out_uvs = state["out_uvs"].cuda()

        # Setup differentiable rasterizer context
        ctx = dr.RasterizeCudaContext()
        # Prepare UV coordinates for rasterization (rendering in UV space)
        uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])],
                             dim=-1).unsqueeze(0)
        rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)

        # Rasterize in chunks to save memory
        for i in range(0, out_faces.shape[0], RASTER_CHUNK):
            rast_chunk, _ = dr.rasterize(
                ctx, uvs_rast, out_faces[i:i+RASTER_CHUNK],
                resolution=[texture_size, texture_size],
            )
            mask_chunk = rast_chunk[..., 3:4] > 0
            rast_chunk[..., 3:4] += i # Store face ID in alpha channel
            rast = torch.where(mask_chunk, rast_chunk, rast)

        # Mask of valid pixels in texture
        mask = rast[0, ..., 3] > 0

        # Interpolate 3D positions in UV space (finding 3D coord for every texel)
        pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
        valid_pos = pos[mask]

        # Free rasterization intermediates - only mask and valid_pos needed from here
        del rast, pos, uvs_rast, ctx
        torch.cuda.empty_cache()

        # Map these positions back to the *original* high-res mesh to get accurate attributes: geometry's BVH of it
        # when it was kept (a host tree uploaded again for this step), else the same BVH built again from the same
        # rows (in RAM: cuBVH builds on the CPU)
        if bvh is None:
            bvh = cumesh.cuBVH(state["source_vertices"], state["source_faces"])
        elif host_trees():
            bvh.upload()
        vertices = state["source_vertices"].cuda()
        faces = state["source_faces"].cuda()

        _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
        orig_tri_verts = vertices[faces[face_id.long()]] # (N_new, 3, 3)
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

        # Free original high-res mesh data and the BVH's card copy (the caller lets go of the tree itself)
        if host_trees():
            bvh.release_device()
        del vertices, faces, bvh, orig_tri_verts, face_id, uvw
        torch.cuda.empty_cache()

        coords, attr_volume = state["voxels"].take()       # the bytes the decode gave, from RAM or the file
        attr_volume = attr_volume.cuda()
        coords = coords.cuda()

        # Trilinear sampling from the attribute volume (Color, Material props); flex_gemm's kernel is not tuned
        # again for this prop's texel count (the same bytes from any of its configs: reuse_grid_sample_tuning)
        reuse_grid_sample_tuning(int(attr_volume.shape[0]), int(attr_volume.shape[1]), int(valid_pos.shape[0]))
        attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda', dtype=attr_volume.dtype)
        attrs[mask] = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
            grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
            mode='trilinear',
        )

        del attr_volume, coords
        torch.cuda.empty_cache()
        return {"mask": mask.cpu().numpy(), "texels": attrs.cpu()}

    def export(self, state: dict) -> None:
        """CPU: the material, the trimesh turned to face +Z, the streak check and the file at out_path. Reads
        UNWRAPPED and BAKED; writes "result", what the hub keeps with the mesh."""
        import numpy as np
        import cv2
        import trimesh
        import trimesh.visual
        from PIL import Image

        _drop(state, SOURCE + VOXELS)       # bake's input; it is done
        mask = state["mask"]
        attrs = state["texels"]
        attr_layout = self.attr_layout

        # --- Texture post-processing & material construction ---
        # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
        base_color = np.clip(attrs[..., attr_layout['base_color']].numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., attr_layout['metallic']].numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., attr_layout['roughness']].numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., attr_layout['alpha']].numpy() * 255, 0, 255).astype(np.uint8)
        alpha_mode = 'OPAQUE'

        # Inpainting: fill gaps (dilation) to prevent black seams at UV boundaries. cv2.inpaint fills each channel on
        # its own from the same mask (checked: the same pixels as one call per channel), so the three one-channel
        # maps go as one picture (half the time of to_glb's three calls) on this thread, and the base colour's
        # three channels each on an export thread meanwhile where there are three (a third less time than its one
        # call: most of a call is the order it fills the mask in), else in one call on one (OpenCV lets go of
        # Python's lock while it works). Each texture is encoded as WebP on an export thread as soon as it is
        # filled, while this thread builds the mesh; the file gets those bytes (encoded_as_webp).
        pool = _export_pool()
        mask_inv = (~mask).astype(np.uint8)
        packed = np.concatenate([metallic, roughness, alpha], axis=-1)
        if EXPORT_THREADS >= 3:
            base_jobs = [pool.submit(cv2.inpaint, np.ascontiguousarray(base_color[..., c]), mask_inv, 3,
                                     cv2.INPAINT_TELEA) for c in range(3)]
        else:
            base_jobs = [pool.submit(cv2.inpaint, base_color, mask_inv, 3, cv2.INPAINT_TELEA)]
        packed = cv2.inpaint(packed, mask_inv, 1, cv2.INPAINT_TELEA)
        metallic, roughness, alpha = (packed[..., i:i + 1] for i in range(3))
        metallic_roughness = np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)
        metallic_roughness_webp = pool.submit(_webp, metallic_roughness)
        filled = [job.result() for job in base_jobs]
        base_color = np.stack(filled, axis=-1) if len(filled) == 3 else filled[0]
        base_rgba = np.concatenate([base_color, alpha], axis=-1)
        base_webp = pool.submit(_webp, base_rgba)

        # Standard PBR packs Metallic and Roughness into Blue and Green channels
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=encoded_as_webp(Image.fromarray(base_rgba), base_webp),
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=encoded_as_webp(Image.fromarray(metallic_roughness), metallic_roughness_webp),
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode=alpha_mode,
            doubleSided=False,              # to_glb: True if not remesh else False; the launcher remeshes
        )

        # --- Coordinate system conversion & final object ---
        # Copies: to_glb's came from the card; these must not write into the state's tensors.
        vertices_np = state["out_vertices"].numpy().copy()
        faces_np = state["out_faces"].numpy().copy()
        uvs_np = state["out_uvs"].numpy().copy()
        normals_np = state["out_normals"].numpy().copy()

        # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
        normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
        uvs_np[:, 1] = 1 - uvs_np[:, 1] # Flip UV V-coordinate

        glb = trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            vertex_normals=normals_np,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
        )
        self.report(BARS[5], 6, 6)

        # --- The launcher's own: decoded_to_glb's turn, save_glb's check and file ---
        glb.apply_transform(np.array(self.rotation, dtype=np.float64))
        warning = self.streak_warning(glb) or (pixal3d_guard.STREAK_WARNING if self.blocked else None)
        self.report(SAVING, 0, 1)
        # The mesh folder again if it went, but never a prop's folder: a prop the artist deleted while its mesh was
        # made fails (kind "input") instead of coming back as a folder of its own.
        try:
            pixal3d_mv.make_folder(Path(self.out_path).parent)
        except pixal3d_mv.PropDeleted as exc:
            raise self.types.PropFailed(str(exc), "input") from None
        glb.export(self.out_path, extension_webp=True)
        state["result"] = {"mode": "multi" if self.multi else "single", "views": self.views if self.multi else 1,
                           "warning": warning, "ss_retries": self.ss_retries, "resolution": self.resolution}

    @staticmethod
    def result(state: dict) -> dict:
        return dict(state["result"])


def finish_steps(px: Any, mesh: Any, attr_layout: dict, resolution: int, decimation_target: int, texture_size: int,
                 out_path: str, multi: bool, ss_retries: int, blocked: bool, rotation: Any,
                 streak_warning: Callable[[Any], str | None], *, views: int = 1, inline: bool = False,
                 types: Any = None, work_folder: Path | str | None = None) -> Any:
    """The decoded `mesh` (vertices, faces, coords, attrs: the engine's DecodedMesh, or the reference pipeline's
    mesh) -> a FinishSteps whose five steps write the GLB to `out_path` and whose result is what the hub keeps with
    the mesh: {"mode", "views", "warning", "ss_retries", "resolution"} (save_glb's answer and the resolution).
    `work_folder`: the prop's own folder in the batch's work folder, where its texture volume waits for the bake
    (TextureVoxels); None keeps it in RAM.

    `resolution`: the grid the prop was decoded at; `rotation`: the 4x4 turn the GLB gets (pixal3d_server.
    GLB_ROTATION); `streak_warning(glb)`: the artist's warning for a streaked mesh or None (pixal3d_server); `blocked`:
    every draw of the sparse structure was a block, so the mesh is warned about as a streaked one; `views`: how many
    views a multi-view prop was made from. `inline`: the steps run one after the other on the calling thread
    (run_all), so every step reports to_glb's bar and "Saving GLB" as to_glb and save_glb did; under the engine
    only the first step reports (_Glb.report). `types`: what provides FinishSteps and PropFailed (default: the
    fork's pixal3d.staged.types).

    The mesh's tensors are copied to the CPU when they are not there (the reference pipeline's are on the card;
    its caller can let go of them once this returns). `native_mib` is left to the engine (parts.glb_reserve)."""
    types = types if types is not None else staged_types()
    glb = _Glb(px, attr_layout, resolution, decimation_target, texture_size, out_path, multi, views, ss_retries,
               blocked, rotation, streak_warning, inline, types, faces=int(mesh.faces.shape[0]))
    state = {"vertices": mesh.vertices.cpu(), "faces": mesh.faces.cpu(),
             "voxels": TextureVoxels(mesh.coords.cpu(), mesh.attrs.cpu(), work_folder)}
    known = {f.name for f in dataclasses.fields(types.FinishSteps)}
    extra = {name: value for name, value in (("release", glb.release), ("holds_mib", glb.holds_mib)) if name in known}
    return types.FinishSteps(steps=[(kind, getattr(glb, name)) for kind, name in STEPS],
                             state=state, result=glb.result, **extra)


def run_all(steps: Any) -> dict:
    """Every step of a FinishSteps in order on the calling thread, then its result: the reference path's finish,
    and a check's. The state is cleared at the end, as the engine clears it."""
    try:
        for _kind, fn in steps.steps:
            fn(steps.state)
        return steps.result(steps.state)
    finally:
        steps.state.clear()
