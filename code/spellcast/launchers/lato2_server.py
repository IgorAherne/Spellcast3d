"""LATO.2 tool server for Spellcast3D. Runs INSIDE the lato2 venv (Python 3.11).

    python lato2_server.py --repo <code/lato2> --models <models/lato2> --port P [--host 127.0.0.1]
                           [--projects-root <projects>] [--attn auto|flash_attn|sdpa]
                           [--precision auto|bf16|fp16|fp32] [--low-vram auto|on|off]

LATO.2 (https://github.com/IgorAherne/LATO2-stableprojectorz, MIT, weights MIT) turns a high poly into a clean,
artist-like triangle mesh of a chosen vertex count. This server loads its models once and serves the hub
(adapters/lato2.py):

    GET  /spellcast/health     -> {"ready", "attention", "precision", "low_vram", "gpu"} (ready false once CUDA broke)
    GET  /spellcast/progress   -> {"phase": "idle"|"working", "fraction", "message"}
    POST /spellcast/retopo     {mesh_path, out_path, vertices, seed} -> {obj_path, vertices, faces, seconds, seed,
                               warning, non_manifold_edges, boundary_edges, outward, missing, prepared, ...}
                               errors: {"detail": <for an artist>, "kind": input|oom|ram|missing_file|cancelled|busy|
                               restart|error}; "restart": an error left CUDA unusable, so the process stops being ready
                               and exits a second later (the hub starts it again for the next prop); "busy": a mesh
                               the hub gave up on still runs
    POST /spellcast/cancel     {} stop the running mesh at its next step (kind "cancelled")
    POST /spellcast/shutdown   {} graceful stop, as CTRL_BREAK

The output OBJ holds triangles only, no UVs (the hub unwraps them in Instant Meshes), in the SAME frame and scale
as the input mesh (y up, front +Z): LATO.2 works in the input's bounding box scaled to [-0.5, 0.5] and the result
is mapped back, with the half-voxel its export leaves out ((v_int + 0.5) / 1024: the voxel centre of its own
quantisation) put back.

LATO.2's repo is never edited, and its scripts/e2e_inference.py is not run: this file calls LATO.2's own model
classes and dataset functions in the same order, with these differences:
- the conditioning picture is drawn by lato2_render.py (upstream's Open3D renderer cannot run on Windows);
- the faces come from lato2_faces.py (a manifold, outward-wound recovery from the pair logits, then crumbs dropped
  and small holes closed) instead of upstream's edges_to_faces (every 3-clique, unwound);
- a crowded high poly is solidified first (lato2_prepare.py), unless that would lose thin or separate parts;
- the models are loaded once (weights only, the encoders LATO.2 needs only for training dropped) and every mesh is
  seeded on its own, so a prop's result does not depend on the props made before it;
- older GPUs: below RTX 30xx the attention runs on torch's SDPA through attn_shim.py (flash_attn has no kernels
  there) and the autocast dtype is float16 (RTX 20xx) or float32 (GTX 10xx) instead of bfloat16; on a GTX 10xx
  the V-VAE's spconv runs its products on torch.mm (spconv_shim.py: its wheel has no kernels for sm_61). All forced on an
  RTX 3060, the same quality; a Pixal3D bear took 56 s (flash_attn, bfloat16), 62 s (shim, bfloat16), 76 s (shim,
  float16), 206 s (shim, float32). Cards under 10 GB keep the models in RAM and move each stage's to the GPU (peak
  3.2-3.9 GB allocated instead of 5.5 GB);
- torch_scatter (imported by a training-only module, never called here) is stubbed when missing, and o_voxel is
  imported without its package __init__ (which pulls in triton, flex_gemm, cumesh and nvdiffrast and queries the
  GPU): only its mesh_to_flexible_dual_grid is used.
Windows: no DataLoader workers, no /tmp, no expandable_segments, bytecode caches kept in the venv.

Only the stdlib, fastapi and pydantic at module level: the hub's tests import this file without torch or numpy.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import importlib.util
import json
import os
import random
import signal
import sys
import threading
import time
import traceback
import types
from pathlib import Path
from typing import Any, Callable

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import attn_shim                                           # launchers/ (this script's folder)
import spconv_shim
import lato2_faces
import lato2_prepare
import lato2_render
from _loopback_guard import LoopbackGuard, is_loopback

# Keep in sync with adapters/lato2.py.
MIN_VERTICES = 200
MAX_VERTICES = 10_000          # the trained range (vflow max_vertex_num, tflow max_vertices)
MESH_SUFFIXES = (".glb", ".gltf", ".obj", ".ply", ".stl", ".off")
ATTENTION = ("auto", "flash_attn", "sdpa")
PRECISIONS = ("auto", "bf16", "fp16", "fp32")
PREPARE = ("auto", "off", "always")
LOW_VRAM_MB = 10_000           # "auto": swap the two stages' models through the GPU on cards below this

CHECKPOINTS = ("vflow", "vvae", "offset_head", "tflow", "tvae", "voxel_encoder")
DINO_DIR = "dinov2"            # models/lato2/dinov2: checkpoints/dinov2_vitl14_reg4_pretrain.pth + the hub code
DINO_CODE = "facebookresearch_dinov2_main"
DINO_WEIGHTS = "checkpoints/dinov2_vitl14_reg4_pretrain.pth"

# LATO.2's sampling settings (scripts/e2e_inference.py defaults).
VFLOW_STEPS = 24
CFG_STRENGTH = 3.0
RESCALE_T = 1.0
TFLOW_STEPS = 50
INFERENCE_THRESHOLD = 0.5

# The warnings an artist gets with a mesh (it is still saved).
HOLES_LIMIT = 0.02             # share of open (boundary) edges
INWARD_LIMIT = 0.90            # share of faces facing like the high poly's surface next to them
SHORT_LIMIT = 0.5              # vertices made / asked
MISSING_LIMIT = 0.05           # share of the high poly's surface over one vertex spacing from the clean mesh
# Face recovery clean-up (lato2_faces): pieces of fewer faces are dropped as crumbs, loops of up to this many open
# edges are closed.
ISLAND_FACES = 16
HOLE_EDGES = 48
HOLES = ("The clean mesh has holes ({share:.0%} of its edges are open). Try another Seed, or a different "
         "size.")
INWARD = ("Parts of the clean mesh fold over ({share:.0%} of its faces face away from the model's surface). Try "
          "another Seed, or a different size.")
SHORT = "LATO.2 placed {made:,} vertices of the {asked:,} asked for: this shape is simpler at this size."
MISSING = ("Part of the model is missing from the clean mesh ({share:.0%} of its surface is not covered). Try another "
           "Seed.")

MODELS_MISSING = ("LATO.2's AI model files are missing or incomplete (they belong in the models\\lato2 folder of "
                  "Spellcast3D). Close Spellcast3D and start it again: it puts back what is missing.")

# (phase label, fraction at its start, at its end), in order; timed on an RTX 3060 (a Pixal3D bear, 2,000 vertices:
# read 0.4 s, crease closing 4 s, picture + voxels 2 s, vertices 38 s, decode 1.5 s, connections 8 s, edges 1 s,
# faces 1 s).
PHASES = {
    "read": ("Reading the 3D model", 0.00, 0.01),
    "prepare": ("Closing tight creases", 0.01, 0.08),
    "look": ("Looking at the model", 0.08, 0.11),
    "vertices": ("Placing the vertices", 0.11, 0.79),
    "decode": ("Decoding the vertices", 0.79, 0.82),
    "connect": ("Connecting the vertices", 0.82, 0.96),
    "edges": ("Scoring the edges", 0.96, 0.975),
    "faces": ("Building the faces", 0.975, 0.995),
    "save": ("Saving the clean mesh", 0.995, 1.00),
}


class RetopoRequest(BaseModel):
    mesh_path: str                                   # the high poly (.glb/.obj/...), inside the projects folder
    out_path: str                                    # the clean mesh (.obj), inside the projects folder
    vertices: int = Field(2000, ge=MIN_VERTICES, le=MAX_VERTICES)
    seed: int = 0                                    # < 0 = random
    prepare: str = "auto"                            # close tight creases first: auto (when crowded) | off | always


class Cancelled(Exception):
    """The hub asked to stop this mesh (POST /spellcast/cancel)."""


class InputProblem(Exception):
    """Something about the request or the mesh the artist has to fix (kind "input")."""


class Progress:
    """What GET /spellcast/progress reports; written by the GPU worker."""

    def __init__(self) -> None:
        self.phase = "idle"
        self.fraction = 0.0
        self.message = "Idle"
        self.cancel = threading.Event()

    def start(self) -> None:
        self.phase, self.fraction, self.message = "working", 0.0, "Starting"
        self.cancel.clear()

    def step(self, key: str, done: float = 0.0, detail: str = "") -> None:
        """Enter or advance a phase of PHASES (done: 0..1 of it). Raises Cancelled when asked to stop."""
        if self.cancel.is_set():
            raise Cancelled()
        label, start, end = PHASES[key]
        self.fraction = max(self.fraction, start + (end - start) * min(max(done, 0.0), 1.0))
        self.message = f"{label} ({detail})" if detail else label

    def stop(self) -> None:
        self.phase, self.fraction, self.message = "idle", 0.0, "Idle"

    def to_dict(self) -> dict[str, Any]:
        return {"phase": self.phase, "fraction": round(self.fraction, 3), "message": self.message}


# ---- environment (before torch and LATO.2 are imported) ---------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LATO.2 server for Spellcast3D")
    p.add_argument("--repo", required=True, help="LATO.2 checkout (code/lato2)")
    p.add_argument("--models", required=True, help="LATO.2's weights (models/lato2)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--projects-root", default=None,
                   help="the hub's projects folder: /spellcast/retopo reads and writes only inside it")
    p.add_argument("--attn", choices=ATTENTION, default="auto",
                   help="flash_attn on RTX 30xx+ when installed, else torch SDPA through attn_shim.py")
    p.add_argument("--precision", choices=PRECISIONS, default="auto",
                   help="autocast dtype: bf16 on RTX 30xx+, fp16 on RTX 20xx, fp32 below")
    p.add_argument("--low-vram", choices=("auto", "on", "off"), default="auto",
                   help="keep the models in RAM and move each stage's to the GPU only while it runs")
    return p.parse_args(argv)


def configure_env() -> None:
    """Environment LATO.2 and its libraries read when they are imported."""
    os.environ["SPARSE_BACKEND"] = "spconv"
    # LATO.2's sparse attention accepts only flash_attn or xformers; below RTX 30xx `flash_attn` is attn_shim.py
    os.environ["SPARSE_ATTN_BACKEND"] = "flash_attn"
    os.environ["ATTN_BACKEND"] = "flash_attn"
    os.environ.setdefault("SPCONV_ALGO", "native")      # no per-shape tuning: every prop has other shapes
    os.environ["XFORMERS_DISABLED"] = "1"               # DINOv2: torch SDPA on every GPU
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # torch's CPU allocator gives freed memory back to Windows only after a delay otherwise (pixal3d_server.py)
    os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)     # upstream's expandable_segments: not on Windows
    # Bytecode caches of LATO.2's repo and of the DINOv2 hub code (both read-only here) go into the venv.
    if sys.pycache_prefix is None:
        sys.pycache_prefix = str(Path(sys.prefix) / "pycache")


def install_stubs() -> list[str]:
    """Stand-ins for what LATO.2 imports but never runs here. Returns what was stubbed (for the log)."""
    done = []
    if importlib.util.find_spec("torch_scatter") is None:
        # modules/pointnet.py imports scatter_mean for the training-only point encoders
        stub = types.ModuleType("torch_scatter")

        def unused(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("torch_scatter is not installed: LATO.2 needs it only for training")

        for name in ("scatter", "scatter_mean", "scatter_max", "scatter_sum", "scatter_add"):
            setattr(stub, name, unused)
        sys.modules["torch_scatter"] = stub
        done.append("torch_scatter")
    spec = importlib.util.find_spec("o_voxel")
    if spec is not None and spec.submodule_search_locations and "o_voxel" not in sys.modules:
        # an empty package over o_voxel's folder: `o_voxel.convert` imports only its compiled _C
        package = types.ModuleType("o_voxel")
        package.__path__ = list(spec.submodule_search_locations)
        package.__spec__ = spec
        sys.modules["o_voxel"] = package
        done.append("o_voxel (without its __init__)")
    return done


def choose_attention(asked: str, capability: tuple[int, int], flash_installed: bool) -> str:
    """flash_attn (the real one) on RTX 30xx+ when installed, else the SDPA shim."""
    if asked == "flash_attn" and flash_installed and capability >= (8, 0):
        return "flash_attn"
    if asked == "auto" and flash_installed and capability >= (8, 0):
        return "flash_attn"
    return "sdpa"


def choose_precision(asked: str, capability: tuple[int, int]) -> str:
    """bfloat16 needs RTX 30xx+ (sm_80); float16 is fast from sm_70; GTX 10xx computes float16 slowly."""
    if asked == "bf16" and capability < (8, 0):
        return "fp16" if capability >= (7, 0) else "fp32"
    if asked != "auto":
        return asked
    if capability >= (8, 0):
        return "bf16"
    return "fp16" if capability >= (7, 0) else "fp32"


# ---- the models ----------------------------------------------------------------------------------------------

class Lato2:
    """LATO.2's models, loaded once; retopo() makes one clean mesh. Every call runs on the GPU worker thread."""

    def __init__(self, repo: Path, models: Path, attention: str, precision: str, low_vram: bool) -> None:
        self.repo, self.models = repo, models
        self.attention, self.precision, self.low_vram = attention, precision, low_vram
        self.ready = False
        self.last_image: Any = None

    # -- loading
    def load(self) -> None:
        import torch

        from models import (DinoV2Encoder, OffsetHead, TopologySiTFlow, TopologyVAE, VertexSLatFlowModel,
                            VertexVAE, VoxelFieldConditioner)

        self.device = torch.device("cuda")
        park = torch.device("cpu") if self.low_vram else self.device
        started = time.monotonic()
        self.vflow, vflow_cfg = load_model(VertexSLatFlowModel, self.models / "vflow.pt", park)
        self.vvae, vvae_cfg = load_model(VertexVAE, self.models / "vvae.pt", park, drop=("encoder",))
        self.offset_head, _ = load_model(OffsetHead, self.models / "offset_head.pt", park)
        self.tflow, tflow_cfg = load_model(TopologySiTFlow, self.models / "tflow.pt", park)
        # T-VAE casts its tokens to its config's dtype (bfloat16) whenever autocast is off: in float32 it must be
        # float32, or its float32 layers refuse the bfloat16 tokens (modules/utils.py manual_cast)
        self.tvae, _ = load_model(TopologyVAE, self.models / "tvae.pt", park,
                                  drop=("encoder", "point_net", "point_embed", "point_fusion"),
                                  args={"dtype": "float32"} if self.precision == "fp32" else None)
        self.voxel_encoder, venc_cfg = load_model(VoxelFieldConditioner, self.models / "voxel_encoder.pt", park)
        self.res = int(vvae_cfg["resolution"])
        self.min_res = int(vvae_cfg["min_resolution"])
        self.latent_dim = int(vflow_cfg["latent_dim"])
        self.density_max = float(vflow_cfg["max_vertex_num"])
        self.z_dim = int(tflow_cfg["args"]["z_dim"])
        self.num_discrete = int(tflow_cfg["args"]["num_discrete"])
        self.max_vertices = int(tflow_cfg["args"]["max_vertices"])
        self.latent_scale = float(tflow_cfg["latent_scale"])
        self.voxel_res = int(venc_cfg["resolution"])
        if self.num_discrete != self.res or self.voxel_res != self.min_res:
            raise RuntimeError(f"LATO.2's checkpoints do not belong together (T-Flow grid {self.num_discrete} vs "
                               f"V-VAE {self.res}, voxel encoder {self.voxel_res} vs {self.min_res}).")
        self.dino = DinoV2Encoder(model_name=vflow_cfg["dino_version"], hub_dir=str(self.models / DINO_DIR),
                                  img_res=int(vflow_cfg["image_resolution"])).eval().to(park)
        self.stage1 = (self.dino, self.vflow, self.vvae, self.offset_head)
        self.stage2 = (self.tflow, self.tvae, self.voxel_encoder)
        torch.backends.cuda.matmul.allow_tf32 = True     # T-Flow runs in float32: TF32 on RTX 30xx+
        torch.backends.cudnn.allow_tf32 = True
        gc.collect()
        print(f"[spellcast] LATO.2 models loaded in {time.monotonic() - started:.0f} s ("
              f"{'kept in RAM, moved to the GPU per stage' if self.low_vram else 'on the GPU'}).", flush=True)
        self.ready = True

    def _on_gpu(self, stage: tuple[Any, ...]) -> Any:
        """Context: the stage's models on the GPU (low VRAM: only while it runs). Moving them there can run out of
        memory half-way: then the ones moved already go back to the CPU before the error goes on (__exit__ does not
        run when __enter__ fails), so the next prop does not start with them stranded on the card."""
        engine = self

        def park() -> None:
            import torch

            for m in stage:
                m.to("cpu")
            torch.cuda.empty_cache()

        class _Stage:
            def __enter__(self) -> None:
                if engine.low_vram:
                    try:
                        for m in stage:
                            m.to(engine.device)
                    except BaseException:
                        park()
                        raise

            def __exit__(self, *exc: Any) -> None:
                if engine.low_vram:
                    park()

        return _Stage()

    def _autocast(self) -> Any:
        import torch

        if self.precision == "fp32":
            return torch.autocast("cuda", enabled=False)
        return torch.autocast("cuda", dtype=torch.bfloat16 if self.precision == "bf16" else torch.float16)

    # -- one mesh
    def retopo(self, req: RetopoRequest, seed: int, progress: Progress) -> dict[str, Any]:
        """The clean mesh of req.mesh_path at about req.vertices vertices -> req.out_path (OBJ, triangles, no
        UVs, the input's frame). Returns the answer's fields (see the module doc)."""
        import numpy as np
        import torch

        from dataset.utils import dedup_quantized_mesh, extract_active_voxels
        from models import TopoFlowEulerSampler, VertFlowEulerCfgSampler
        from modules.sparse import SparseTensor
        from utils.inference import build_voxel_fields, decode_vertices, pad_verts

        timings: dict[str, float] = {}
        clock = [time.monotonic()]

        def lap(name: str) -> None:
            now = time.monotonic()
            timings[name] = round(now - clock[0], 2)
            clock[0] = now

        reseed(seed)
        progress.step("read")
        vertices, faces = read_mesh(Path(req.mesh_path))
        lap("read")

        progress.step("prepare")
        vertices, faces, prepared, crowd = lato2_prepare.prepare(
            vertices, faces, req.vertices, req.prepare, seed=seed, log=lambda text: print(text, flush=True))
        lap("prepare")

        progress.step("look")
        center, extent = frame_of(vertices)
        normalized = (vertices - center) / extent
        v_int, offsets, qfaces = quantize(vertices, faces, self.res)
        gt_int, _, gt_faces = dedup_quantized_mesh(v_int, offsets, qfaces, self.res)
        if len(gt_int) < 3 or len(gt_faces) < 1:
            raise InputProblem("This 3D model is too small or flat for LATO.2 (it collapses to a few points).")
        quant_v = np.clip(gt_int.astype(np.float64) / (self.res - 1.0) - 0.5, -0.5 + 1e-6, 0.5 - 1e-6)
        active = extract_active_voxels(quant_v.astype(np.float32), gt_faces, self.min_res)      # (M, 3) int
        image = lato2_render.render_condition(v_int.astype(np.float64) / self.res - 0.5, qfaces, self.device)
        self.last_image = image         # the picture LATO.2 saw, for a look when a result puzzles
        lap("look")

        with torch.no_grad(), self._on_gpu(self.stage1):
            cond = self.dino(image[None]).float()
            neg_cond = torch.zeros_like(cond)
            density = torch.tensor([req.vertices / self.density_max * 1000.0], device=self.device)
            coords = torch.cat([torch.zeros((len(active), 1), dtype=torch.int32), active.int()], dim=1)
            with self._autocast():
                coords = coords.to(self.device)
                noise = SparseTensor(coords=coords, feats=torch.randn(len(coords), self.latent_dim,
                                                                     device=self.device))
                calls = 2 * VFLOW_STEPS
                counter = Counter(self.vflow, lambda n: progress.step("vertices", n / calls,
                                                                      f"step {min(n // 2 + 1, VFLOW_STEPS)} of "
                                                                      f"{VFLOW_STEPS}"))
                z = VertFlowEulerCfgSampler().sample(model=counter, noise=noise, cond=cond, neg_cond=neg_cond,
                                                     steps=VFLOW_STEPS, cfg_strength=CFG_STRENGTH,
                                                     rescale_t=RESCALE_T, density=density)
                lap("vertices")
                progress.step("decode")
                pred_coords, pred_offsets = decode_vertices(self.vvae, self.offset_head, z, INFERENCE_THRESHOLD)
            del cond, neg_cond, noise, z
        lap("decode")
        vert_int = pred_coords[pred_coords[:, 0] == 0, 1:].long()
        vert_off = pred_offsets[pred_coords[:, 0] == 0].float()
        made = int(vert_int.shape[0])
        if made < 3:
            raise InputProblem("LATO.2 placed no vertices on this model. Try another Seed or more vertices.")
        if made > self.max_vertices:
            raise InputProblem(f"LATO.2 placed {made:,} vertices, more than the {self.max_vertices:,} it can "
                               f"connect. Ask for fewer vertices and try again.")

        progress.step("connect")
        with torch.no_grad(), self._on_gpu(self.stage2):
            verts_t, mask, _ = pad_verts([vert_int], self.device)
            field = build_voxel_fields([active.long()], self.voxel_res, self.device)
            cond_vox = self.voxel_encoder(field)
            z0 = torch.randn(1, made, self.z_dim, device=self.device)
            counter = Counter(self.tflow, lambda n: progress.step("connect", n / TFLOW_STEPS,
                                                                  f"step {n} of {TFLOW_STEPS}"))
            z_flow = TopoFlowEulerSampler().sample(model=counter, noise=z0, verts=verts_t, mask=mask,
                                                   cond=cond_vox, steps=TFLOW_STEPS)   # float32, as upstream
            z = z_flow.float() / self.latent_scale
            lap("connect")
            progress.step("edges")
            with self._autocast():
                logits = lato2_faces.pair_logits(self.tvae, z, verts_t, mask)[0]
            del verts_t, mask, field, cond_vox, z0, z_flow, z
        lap("edges")

        progress.step("faces")
        points = decode_positions(vert_int.numpy(), vert_off.numpy(), self.res)     # LATO.2's [-0.5, 0.5] frame
        surface = lato2_faces.SurfaceSamples(normalized, faces, seed=seed)
        if surface.inside_out():
            surface.flip()
        normals = surface.normals_at(points)
        tris = lato2_faces.build_faces(points, logits, normals)
        tris, islands = lato2_faces.drop_islands(tris, ISLAND_FACES)
        if len(tris) == 0:
            raise InputProblem("LATO.2 could not connect its vertices into faces on this model. Try another Seed "
                               "or a different size.")
        points, tris, holes = lato2_faces.close_holes(points, tris, HOLE_EDGES)
        points, tris, unused = drop_unused(points, tris)
        report = lato2_faces.edge_report(tris)
        outward = surface.outward_share(points, tris)
        spacing = (surface.area / max(len(points), 1)) ** 0.5       # the clean mesh's typical edge
        missing = surface.missing_share(points, tris, spacing)
        lap("faces")

        progress.step("save")
        out = Path(req.out_path)
        write_obj(out, points * extent + center, tris)
        lap("save")

        warnings = []
        if report["boundary"] > HOLES_LIMIT:
            warnings.append(HOLES.format(share=report["boundary"]))
        if outward < INWARD_LIMIT:
            warnings.append(INWARD.format(share=1.0 - outward))
        if missing > MISSING_LIMIT:
            warnings.append(MISSING.format(share=missing))
        if made < SHORT_LIMIT * req.vertices:
            warnings.append(SHORT.format(made=made, asked=req.vertices))
        print(f"[spellcast] {len(points):,} vertices ({made:,} placed, {unused} unused), {len(tris):,} faces "
              f"({holes} holes closed, {islands} crumb faces dropped); "
              f"non-manifold edges {report['non_manifold']:.2%}, open {report['boundary']:.2%}, outward "
              f"{outward:.1%}, uncovered {missing:.1%}; {prepared}"
              + (f" (crowding {crowd:.0%})" if crowd is not None else "") + f"; {timings}", flush=True)
        return {"obj_path": str(out), "vertices": int(len(points)), "faces": int(len(tris)), "seed": seed,
                "warning": " ".join(warnings) or None,
                "non_manifold_edges": round(report["non_manifold"], 5), "boundary_edges": round(report["boundary"], 5),
                "consistent_winding": report["consistent"], "outward": round(outward, 4), "missing": round(missing, 4),
                "vertices_placed": made, "holes_closed": holes, "crumb_faces_dropped": islands, "prepared": prepared,
                "crowding": None if crowd is None else round(crowd, 3), "timings": timings}


class Counter:
    """Calls a model and reports how many calls were made (the flow samplers take any callable)."""

    def __init__(self, model: Any, report: Callable[[int], None]) -> None:
        self.model, self.report, self.calls = model, report, 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.report(self.calls)
        out = self.model(*args, **kwargs)
        self.calls += 1
        return out


def load_model(cls: type, path: Path, device: Any, drop: tuple[str, ...] = (),
               args: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    """A LATO.2 checkpoint {"config": {"args": ...}, "state_dict": ...} -> (model in eval mode on `device`, config).
    Weights only (no code runs from the file), memory-mapped; `args` override the config's constructor arguments;
    the modules in `drop` (the encoders only training uses) are removed after loading."""
    import torch

    if not path.is_file():
        raise FileNotFoundError(2, "LATO.2 model file missing", str(path))
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    config = ckpt["config"]
    model = cls(**{**config["args"], **(args or {})})
    model.load_state_dict(ckpt["state_dict"], strict=True)
    del ckpt
    for name in drop:
        if hasattr(model, name):
            delattr(model, name)
    model.requires_grad_(False)
    return model.eval().to(device), config


def reseed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def read_mesh(path: Path) -> tuple[Any, Any]:
    """(vertices float64 (V, 3), faces int64 (F, 3)) of a mesh file, scenes flattened with their transforms,
    as LATO.2 reads it (trimesh, force="mesh", no processing)."""
    import numpy as np
    import trimesh

    try:
        mesh = trimesh.load(str(path), process=False, force="mesh")
    except Exception as exc:
        raise InputProblem(f"LATO.2 could not read {path.name} ({type(exc).__name__}: {exc}). Make the 3D model "
                           f"again in the 3D stage.") from exc
    vertices = np.asarray(getattr(mesh, "vertices", ()), dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(getattr(mesh, "faces", ()), dtype=np.int64).reshape(-1, 3)
    if len(vertices) < 3 or len(faces) < 1:
        raise InputProblem(f"{path.name} has no triangles. Make the 3D model again in the 3D stage.")
    if not np.isfinite(vertices).all():
        raise InputProblem(f"{path.name} has broken vertex positions. Make the 3D model again in the 3D stage.")
    return vertices, faces


def frame_of(vertices: Any) -> tuple[Any, float]:
    """LATO.2's normalisation of a mesh (dataset/utils.py quantize_mesh_clustering): (bbox centre, largest extent)."""
    import numpy as np

    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    return (lo + hi) / 2.0, max(float((hi - lo).max()), 1e-7)


def quantize(vertices: Any, faces: Any, resolution: int) -> tuple[Any, Any, Any]:
    """dataset/utils.py quantize_mesh_clustering on arrays instead of a file (same maths, LATO.2 MIT): the
    vertices averaged per voxel of a resolution^3 grid over the normalised mesh -> (voxel ints, sub-voxel offsets
    in (-1, 1), faces without the collapsed ones)."""
    import numpy as np

    center, max_extent = frame_of(vertices)
    normalized_v = (vertices - center) / max_extent + 0.5
    v_grid = np.clip(np.floor(normalized_v * resolution), 0, resolution - 1).astype(np.int64)
    v_hash = v_grid[:, 0] * resolution * resolution + v_grid[:, 1] * resolution + v_grid[:, 2]
    unique_hashes, inverse = np.unique(v_hash, return_inverse=True)
    inverse = inverse.reshape(-1)
    v_sum = np.zeros((len(unique_hashes), 3), dtype=np.float64)
    counts = np.zeros(len(unique_hashes), dtype=np.float64)
    np.add.at(v_sum, inverse, normalized_v)
    np.add.at(counts, inverse, 1)
    v_mean = v_sum / counts[:, None]
    v_int = np.clip(np.floor(v_mean * resolution), 0, resolution - 1).astype(np.int32)
    voxel_center = (v_int.astype(np.float64) + 0.5) / float(resolution)
    offsets = np.clip(((v_mean - voxel_center) * 2.0 * resolution).astype(np.float32), -1.0, 1.0)
    new_faces = inverse[faces]
    valid = ((new_faces[:, 0] != new_faces[:, 1]) & (new_faces[:, 1] != new_faces[:, 2])
             & (new_faces[:, 2] != new_faces[:, 0]))
    return v_int, offsets, new_faces[valid]


def decode_positions(vert_int: Any, vert_off: Any, resolution: int) -> Any:
    """Generated vertices in LATO.2's [-0.5, 0.5] frame: the voxel centre plus the offset head's sub-voxel
    offset. Upstream's export (e2e_inference.py export_mesh) uses the voxel's corner, v_int / res: 1/2048 of the
    size off on every axis against its own quantisation (voxel centre (v_int + 0.5) / res)."""
    import numpy as np

    return ((np.asarray(vert_int, dtype=np.float64) + 0.5 + np.asarray(vert_off, dtype=np.float64) / 2.0)
            / float(resolution) - 0.5)


def drop_unused(points: Any, tris: Any) -> tuple[Any, Any, int]:
    """Remove the vertices no face uses (renumbering the faces) -> (points, faces, how many were removed)."""
    import numpy as np

    used = np.zeros(len(points), dtype=bool)
    used[np.asarray(tris).reshape(-1)] = True
    remap = np.cumsum(used) - 1
    return points[used], remap[tris].astype(np.int64), int((~used).sum())


def write_obj(path: Path, vertices: Any, faces: Any) -> None:
    """A plain triangle OBJ (1-based indices), written next to `path` first and then moved over it."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{os.getpid()}{path.suffix}")
    try:
        with open(tmp, "w", encoding="ascii", newline="\n") as f:
            f.write("# Spellcast3D: clean mesh made by LATO.2 (triangles, no UVs)\n")
            np.savetxt(f, np.asarray(vertices, dtype=np.float64), fmt="v %.7g %.7g %.7g")
            np.savetxt(f, np.asarray(faces, dtype=np.int64) + 1, fmt="f %d %d %d")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ---- the HTTP side ---------------------------------------------------------------------------------------------

class GpuWorker:
    """One thread for everything that touches torch: the loading and every mesh, one at a time. torch's CPU
    allocator gives memory back to Windows only on the thread that took it (pixal3d_server.GpuWorker)."""

    def __init__(self) -> None:
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="lato2-gpu")

    def run(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return self._pool.submit(self._call, fn, args, kwargs).result()

    @staticmethod
    def _call(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        try:
            return fn(*args, **kwargs)
        finally:
            gc.collect()
            torch = sys.modules.get("torch")
            try:
                if torch is not None and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()     # other programs share the card
            except Exception:
                pass


def error_response(status: int, detail: str, kind: str) -> JSONResponse:
    # kind: "input" | "oom" | "ram" | "missing_file" | "cancelled" | "error" (adapters/lato2.py reads it)
    return JSONResponse({"detail": detail, "kind": kind}, status_code=status)


def failure_response(exc: BaseException) -> JSONResponse:
    if isinstance(exc, Cancelled):
        return error_response(409, "Stopped: the clean mesh was cancelled.", "cancelled")
    if isinstance(exc, InputProblem):
        return error_response(422, str(exc), "input")
    text = str(exc)
    low = text.lower()
    if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in low:
        return error_response(500, "LATO.2 ran out of GPU memory. Close other programs that use the graphics "
                                   "card and try again, or ask for fewer vertices.", "oom")
    if isinstance(exc, MemoryError) or "not enough memory" in low or "paging file" in low:
        return error_response(500, "LATO.2 ran out of system memory (RAM). Close other programs and try again.",
                              "ram")
    if isinstance(exc, FileNotFoundError):
        return error_response(500, f"A file LATO.2 needs is missing: {exc.filename or text}. " + MODELS_MISSING,
                              "missing_file")
    short = text if len(text) <= 300 else text[:300] + "..."
    return error_response(500, f"LATO.2 failed while making the clean mesh ({type(exc).__name__}: {short}). "
                               "Try again; if it keeps failing, restart LATO.2 (Settings > Debugging) and check "
                               "its log.", "error")


def cuda_works() -> bool:
    """False when CUDA fails even a tiny job (after some errors - an illegal memory access, a device-side assert, a
    Windows GPU timeout - its context stays broken for this process). Running out of memory for it is no such error
    (another program may fill the card for a moment): that counts as working."""
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


BUSY = ("LATO.2 is still working on an earlier mesh, which the hub stopped waiting for. It starts again for the "
        "next prop.")
BROKEN = ("LATO.2 stopped after an error that left the graphics card unusable for it. It starts again for the next "
          "prop.")


def validate(req: RetopoRequest, projects_root: Path | None) -> str | None:
    """The problem with a request, in an artist's words, or None."""
    mesh, out = Path(req.mesh_path), Path(req.out_path)
    if not mesh.is_absolute() or not out.is_absolute():
        return "The mesh paths must be absolute."
    if projects_root is not None and not (mesh.resolve().is_relative_to(projects_root)
                                          and out.resolve().is_relative_to(projects_root)):
        return "Spellcast3D only makes clean meshes from and into its own projects folder."
    if mesh.suffix.lower() not in MESH_SUFFIXES:
        return f"LATO.2 cannot read '{mesh.suffix}' files. Use one of: {', '.join(MESH_SUFFIXES)}."
    if not mesh.is_file():
        return f"The 3D model was not found: {mesh}. Make it again in the 3D stage."
    if out.suffix.lower() != ".obj":
        return f"The clean mesh is saved as an .obj file (got {out.name})."
    if req.prepare not in PREPARE:
        return f"prepare must be one of {', '.join(PREPARE)} (got {req.prepare!r})."
    return None


def build_app(engine: Any, gpu: GpuWorker, progress: Progress, projects_root: Path | None,
              info: dict[str, Any], exit_later: Callable[[], None] | None = None) -> FastAPI:
    """The /spellcast/* routes around `engine` (a Lato2, or anything with .ready and .retopo(req, seed, progress)).
    exit_later(): what ends this process a moment after an answer that says CUDA broke (default: os._exit(3) a
    second later)."""
    app = FastAPI(title="LATO.2 for Spellcast3D", docs_url=None, redoc_url=None, openapi_url=None)
    working = threading.Lock()      # one mesh at a time: a second request is refused, never queued behind the first
    state = {"broken": False}       # CUDA broke: no more meshes; the process exits
    exit_later = exit_later or (lambda: threading.Timer(1.0, os._exit, (3,)).start())

    @app.exception_handler(RequestValidationError)
    async def readable_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', ())[1:]) or 'request'}: {e.get('msg', '')}"
                             for e in exc.errors())
        return error_response(422, f"LATO.2 rejected the settings ({problems}).", "input")

    @app.get("/spellcast/health")
    async def health() -> dict[str, Any]:
        return {"ready": bool(engine.ready), **info}

    @app.get("/spellcast/progress")
    async def spellcast_progress() -> dict[str, Any]:
        return progress.to_dict()

    @app.post("/spellcast/cancel")
    async def cancel(_: dict[str, Any] = Body(...)) -> dict[str, bool]:
        # the JSON body keeps other web pages out (a CORS preflight, which is refused), as for /shutdown
        working = progress.phase == "working"
        if working:
            progress.cancel.set()
        return {"cancelling": working}

    @app.post("/spellcast/shutdown")
    async def shutdown(_: dict[str, Any] = Body(...)) -> dict[str, bool]:
        """What CTRL_BREAK does, for a parent whose console events cannot reach us (imesh_server.py)."""
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
            return error_response(503, "LATO.2 is still loading its models. Try again in a moment.", "error")
        if not working.acquire(blocking=False):
            # the hub never sends two at once: the first is one it gave up on (a timeout), still running
            return error_response(409, BUSY, "busy")
        try:
            return run(req)
        finally:
            working.release()

    def run(req: RetopoRequest) -> JSONResponse:
        seed = req.seed if req.seed >= 0 else random.randint(0, 2 ** 31 - 1)
        print(f"[spellcast] Clean mesh of {req.mesh_path} ({req.vertices} vertices, seed {seed})", flush=True)

        def work() -> tuple[JSONResponse | None, dict[str, Any], float]:
            started = time.monotonic()
            progress.start()
            failure: JSONResponse | None = None
            result: dict[str, Any] = {}
            try:
                result = engine.retopo(req, seed, progress)
            except Exception as exc:
                if not isinstance(exc, (Cancelled, InputProblem)):
                    traceback.print_exc()
                failure = failure_response(exc)
            finally:
                progress.stop()
            return failure, result, time.monotonic() - started

        failure, result, seconds = gpu.run(work)
        if failure is not None:
            kind = json.loads(failure.body).get("kind")
            if kind not in ("input", "cancelled") and not gpu.run(cuda_works):
                # A CUDA error can leave the GPU context unusable: not ready any more (health, and any request still
                # to come, say so), answer, then exit; the hub starts LATO.2 again for the next prop.
                state["broken"] = True
                engine.ready = False
                print("[spellcast] The graphics card does not answer LATO.2 any more after that error: LATO.2 "
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
        return ("LATO.2's Python environment is incomplete (" + str(exc) + "). Close Spellcast3D and double-click "
                "run-spellcast3d.bat again to repair it.")
    if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in low:
        return ("LATO.2 ran out of GPU memory while loading. Close other programs that use the graphics card "
                "(games, video tools, other AI apps), then start LATO.2 again.")
    if isinstance(exc, MemoryError) or "not enough memory" in low or "paging file" in low:
        return ("LATO.2 ran out of system memory (RAM) while loading its models (about 6 GB). Close other programs, "
                "then start LATO.2 again.")
    if isinstance(exc, FileNotFoundError):
        return MODELS_MISSING
    return ("LATO.2 could not start (the error is above). Start it again; if it keeps failing, close Spellcast3D "
            "and double-click run-spellcast3d.bat again to repair it.")


def missing_weights(models: Path) -> list[str]:
    names = [f"{name}.pt" for name in CHECKPOINTS] + [f"{DINO_DIR}/{DINO_WEIGHTS}",
                                                       f"{DINO_DIR}/{DINO_CODE}/hubconf.py"]
    return [name for name in names if not (models / name).is_file()]


def main() -> int:
    args = parse_args()
    repo, models = Path(args.repo).resolve(), Path(args.models).resolve()
    if not (repo / "models" / "__init__.py").is_file():
        print(f"[spellcast] LATO.2 was not found in {repo}. Close Spellcast3D and double-click run-spellcast3d.bat "
              "again.", flush=True)
        return 2
    missing = missing_weights(models)
    if missing:
        print(f"[spellcast] Missing in {models}: {', '.join(missing)}. " + MODELS_MISSING, flush=True)
        return 2
    configure_env()
    sys.path.insert(0, str(repo))       # LATO.2's top-level packages: models, modules, utils, dataset
    gpu = GpuWorker()
    progress = Progress()
    try:
        import torch

        if not torch.cuda.is_available():
            print("[spellcast] LATO.2 needs an NVIDIA graphics card, and torch sees none.", flush=True)
            return 2
        capability = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        total_mb = torch.cuda.get_device_properties(0).total_memory // 2 ** 20
        attention = choose_attention(args.attn, capability, importlib.util.find_spec("flash_attn") is not None)
        precision = choose_precision(args.precision, capability)
        low_vram = args.low_vram == "on" or (args.low_vram == "auto" and total_mb < LOW_VRAM_MB)
        if attention == "sdpa":
            attn_shim.install_flash_attn_shim()
        stubbed = install_stubs()
        if spconv_shim.torch_products_where_needed(capability):
            stubbed.append("spconv's products on torch.mm")
        print(f"[spellcast] Starting LATO.2 from {repo} on {args.host}:{args.port}: {name} (sm_{capability[0]}"
              f"{capability[1]}, {total_mb:,} MB), attention {attention}, {precision}, low VRAM "
              f"{'on' if low_vram else 'off'}" + (f"; stand-ins: {', '.join(stubbed)}" if stubbed else "") + ".",
              flush=True)
        engine = Lato2(repo, models, attention, precision, low_vram)
        gpu.run(engine.load)
    except Exception as exc:
        traceback.print_exc()
        print("[spellcast] " + startup_hint(exc), flush=True)
        return 1

    import uvicorn

    info = {"attention": attention, "precision": precision, "low_vram": low_vram, "gpu": name}
    app = build_app(engine, gpu, progress, Path(args.projects_root).resolve() if args.projects_root else None, info)
    try:
        uvicorn.run(LoopbackGuard(app) if is_loopback(args.host) else app, host=args.host, port=args.port,
                    log_level="warning")
    except OSError as exc:
        print(f"[spellcast] LATO.2 could not open port {args.port} ({exc}). Another program uses it: wait a moment "
              "and start LATO.2 again, or change the LATO.2 port in Settings.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
