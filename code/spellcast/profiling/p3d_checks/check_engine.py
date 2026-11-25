"""The staged engine (pixal3d/staged: parts, runner, engine): its logic on fakes, and its results on the golden cases.

    python check_engine.py --repo <fork worktree> --fake                   the logic alone: CPU, no card, no lock
    python check_engine.py --repo <fork worktree> --fake --only ladder     the tests whose name has "ladder" in it
    python check_engine.py --repo <fork worktree> --fake --real-store --real-loader
                                                                           the same tests on the worktree's own
                                                                           PropStore and NetworkLoader (still CPU)
    python check_engine.py --repo <fork worktree> --case sv1024            the engine on one golden case (GPU, lock)
    python check_engine.py --repo <fork worktree> --batch mv1024,sv1024,mv1024b,sv1536 --wave-size 2 --reverse
                           --cap-mib 7168                                  a mixed batch, as on a 7 GiB card

--fake runs the REAL engine, runner and part functions (engine.py, runner.py, parts.py) with everything below them
replaced: a loader and a store that keep a log and can be told to fail, networks and samplers that are a few
lines of arithmetic, stand-ins for pipelines/proj_cond.py and pipelines/decode.py, and hooks that make their
pictures from the request. The arithmetic is chosen so that a tensor handed to the wrong prop, a wrong network, a
changed order of the random draws or a lost setting changes that prop's result; a prop's result in any batch must
equal, bit for bit, its result when it is made alone, and what passes between the store and the samplers is
compared with the reference's formulas written out again in the test. What the tests cover: the order (every part
for all props, multi-view first), waves, the ladder normal / tight / cap lifted and the failure classifier, the
swallowed out-of-memory rule, cancel at every check point, failure isolation, resume from checkpoints, the random
generator's hand-over between parts, the events, the networks' time on the card, the two caps of `finish`, the
paired texture and finish (its prop order, what it keeps on the card, its decoders read ahead first), the
proj_linear answers kept per block.
--real-store and --real-loader put the worktree's own PropStore (its files in <repo>/tmp) and NetworkLoader (with
the stand-in networks) under the same tests; they need a worktree that has store.py, stfile.py and loader.py.

The golden mode runs the engine as the launcher will (the worktree's own loader, store, conditioning and decoding
code) with hooks that hand it the golden prepared pictures, cameras and view bundles, and compares what the engine
stores for each prop - DINOv3's tokens, conditionings (cast as the engine stores them), coords, generator states,
latents, the noise of every draw of the sparse structure - and the decoded mesh with the golden tensors
(specification 4.3 rule 5). It prints PyTorch's allocated and reserved peak of every prop-part, every cap and what
was allocated when it was set, and the prop-parts that needed the cap lifted. For the injected failures of the
acceptance list:

    --inject hr:sv1024:3:oom           an out-of-memory error at step 3 of `hr` (":2" twice: the cap is lifted;
                                       ":3": the prop fails); the retried prop must still equal the golden
    --inject hr:sv1024:5:error         any other error: that prop fails, the others must equal the golden
    --inject hr:sv1024:5:cuda          a "CUDA error": one prop fails; with --probe-fails the batch must break
    --inject tex:sv1024:5:cancel       a cancel at that step;  --cancel-at-load hr_flow.sv  while a network loads
    --inject hr:redraw:1:cancel        with --case redraw: stop where its capture stopped (after `lr`)
    --bump-ooms cond_hr:sv1024         PyTorch counts a failed allocation nobody raised: the attempt is repeated
    --die-in sv1024 --work <folder>    the process exits inside that prop's finish hook; run the same batch again
                                       with --resume --work <folder>: props whose `tex` was done only decode

The golden mode needs the merged branch (loader.py, store.py, stfile.py, proj_cond.py, decode.py, the samplers'
step_callback), GOLD (the golden cases, code/logs/profiling/pixal3d-lowmem/golden) and the worktree's MODELS
junction; it runs on the card, under the GPU lock. --fake needs none of them: the fork's code and torch only
(code/tests/test_pixal3d_engine.py runs the same tests under pytest).
"""

from __future__ import annotations

import contextlib
import copy
import functools
import gc
import os
import shutil
import sys
import threading
import time
import traceback
import types
import weakref
import zlib
from pathlib import Path
from typing import Any, Callable, Iterator

import common

TESTS: dict[str, Callable[[], None]] = {}
SCRATCH: Path = Path()
CURRENT: Any = None                     # the fake world whose engine is running
REAL_STORE = False                      # --real-store: the worktree's own PropStore under the fake world
REAL_LOADER = False                     # --real-loader: the worktree's own NetworkLoader, with stand-in networks



def pixal3d_parts() -> tuple:
    """The parts a Pixal3D prop goes through, in order: the engine's PARTS without Hi3D's geometry."""
    from pixal3d.staged import PARTS

    return tuple(part for part in PARTS if not part.startswith("hi3d_"))

def test(fn: Callable[[], None]) -> Callable[[], None]:
    TESTS[fn.__name__] = fn
    return fn


# =====================================================================================================================
# The fake world
# =====================================================================================================================

FEATURES = 16                           # channels of the stand-in DINOv3
PIPELINE = {
    "models": {
        "sparse_structure_decoder": "ckpts/ss_dec", "sparse_structure_flow_model": "ckpts/ss_flow",
        "shape_slat_decoder": "ckpts/shape_dec", "shape_slat_flow_model_512": "ckpts/lr_flow",
        "shape_slat_flow_model_1024": "ckpts/hr_flow", "tex_slat_decoder": "ckpts/tex_dec",
        "tex_slat_flow_model_1024": "ckpts/tex_flow",
    },
    "sparse_structure_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
                                 "params": {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7,
                                            "guidance_interval": [0.6, 1.0], "rescale_t": 5.0}},
    "shape_slat_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
                           "params": {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5,
                                      "guidance_interval": [0.6, 1.0], "rescale_t": 3.0}},
    "tex_slat_sampler": {"name": "FlowEulerGuidanceIntervalSampler", "args": {"sigma_min": 1e-5},
                         "params": {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0,
                                    "guidance_interval": [0.6, 0.9], "rescale_t": 3.0}},
    "shape_slat_normalization": {"mean": [0.1 * i - 1.0 for i in range(32)],
                                 "std": [2.0 + 0.05 * i for i in range(32)]},
    "tex_slat_normalization": {"mean": [0.5 - 0.02 * i for i in range(32)], "std": [1.5 + 0.03 * i for i in range(32)]},
}
SMALL_MODELS = ("rembg", "moge", "dino", "naf")
STORED = ("dino512", "dino1024", "cond_ss", "coords_ss", "rng_ss", "cond_lr", "lr_slat", "rng_lr", "coords_hr",
          "cond_hr", "cond_tex", "shape_slat", "rng_hr", "tex_slat")


def multiview_pipeline() -> dict:
    """The multi-view pipeline file: its own flow checkpoints, and (unlike the real one) its own numbers, so that
    a multi-view prop made with the single-view file's settings shows."""
    multi = copy.deepcopy(PIPELINE)
    for key, value in multi["models"].items():
        if "flow" in key:
            multi["models"][key] = value + "_mv"
    multi["sparse_structure_sampler"]["params"]["guidance_rescale"] = 0.6
    multi["shape_slat_sampler"]["params"]["rescale_t"] = 3.5
    multi["tex_slat_sampler"]["params"]["rescale_t"] = 2.5
    multi["shape_slat_normalization"]["mean"] = [value + 0.25 for value in PIPELINE["shape_slat_normalization"]["mean"]]
    multi["tex_slat_normalization"]["std"] = [value * 1.5 for value in PIPELINE["tex_slat_normalization"]["std"]]
    return multi


def pipeline_dir(folder: Path, multiview: bool) -> Path:
    """A snapshot folder as the engine reads it: the two pipeline files and (empty) checkpoint files."""
    import json

    (folder / "ckpts").mkdir(parents=True, exist_ok=True)
    multi = multiview_pipeline()
    (folder / "pipeline.json").write_text(json.dumps({"name": "x", "args": PIPELINE}), encoding="utf-8")
    (folder / "pipeline_mv.json").write_text(json.dumps({"name": "x", "args": multi}), encoding="utf-8")
    names = set(PIPELINE["models"].values()) | (set(multi["models"].values()) if multiview else set())
    for name in names:
        for extension in (".json", ".safetensors"):
            (folder / (name + extension)).write_bytes(b"")
    return folder


def fake_modules() -> dict[str, types.ModuleType]:
    """Stand-ins for pipelines/proj_cond.py and pipelines/decode.py with the signatures of the specification."""
    import numpy as np
    import torch

    offsets = {"ss": 0.0, "shape_512": 0.125, "shape_1024": 0.25, "tex_1024": 0.5}
    proj_cond = types.ModuleType("pixal3d.pipelines.proj_cond")
    proj_cond.SPECS = {name: types.SimpleNamespace(name=name) for name in offsets}

    class Camera:
        def __init__(self, values: Any, views: int):
            self.values, self.views = values, views

        @staticmethod
        def single(camera: dict, device: Any) -> "Camera":
            assert all(isinstance(camera[name], float) for name in ("camera_angle_x", "distance", "mesh_scale"))
            return Camera(torch.tensor([camera["camera_angle_x"], camera["distance"], camera["mesh_scale"]],
                                       dtype=torch.float32, device=device), 1)

        @staticmethod
        def from_bundle(views: dict, device: Any) -> "Camera":
            values = torch.cat([views["camera_angle_x"].flatten(), views["camera_distance"].flatten(),
                                views["transform_matrix"].flatten(), torch.tensor([views["mesh_scale"]])])
            return Camera(values.to(device), views["camera_angle_x"].shape[1])

    def image_tensor(images: Any, size: int, device: Any) -> Any:
        if isinstance(images, dict):
            return images["images"][size][0].to(device)
        assert all(image.mode == "RGB" for image in images), "a single-view prop's picture is RGB"
        small = [np.array(image.resize((size // 64, size // 64))).astype(np.float32) / 255 for image in images]
        return torch.stack([torch.from_numpy(array).permute(2, 0, 1) for array in small]).to(device)

    def encode(dino: Any, image: Any) -> Any:
        CURRENT.tap("encode", size=image.shape[-1] * 64, views=image.shape[0])
        assert dino.name == "dino"
        tokens = 5 + image.shape[-1]
        ramp = torch.arange(tokens * FEATURES, dtype=torch.float32).reshape(1, tokens, FEATURES) / (tokens * FEATURES)
        return ramp * image.mean(dim=(1, 2, 3)).reshape(-1, 1, 1) + dino.weight + image.shape[-1] * 0.001

    def cond_dense(spec: Any, z: Any, camera: Any) -> dict:
        CURRENT.tap("cond_dense", spec=spec.name, z=z)
        cell = torch.arange(4096, dtype=torch.float32).reshape(1, 4096, 1)
        proj = torch.sin(cell * (1.0 + z.mean() + 0.01 * camera.values.sum()))
        proj = proj + torch.arange(FEATURES, dtype=torch.float32).reshape(1, 1, FEATURES) * 0.01
        return {"global": z.mean(0, keepdim=True)[:, :5].clone(), "proj": proj}

    def cond_tokens(spec: Any, z: Any, image: Any, naf: Any, camera: Any, coords: Any, grid: int) -> dict:
        CURRENT.tap("cond_tokens", spec=spec.name, grid=grid, tokens=coords.shape[0], size=image.shape[-1] * 64,
                    z=z, coords=coords)
        assert naf.name == "naf" and coords.dtype == torch.int32 and coords.shape[1] == 4
        where = (coords[:, 1:].float() * torch.tensor([1.0, 0.5, 0.25])).sum(1, keepdim=True) / grid
        proj = torch.sin(where * (3.0 + z.mean())) + torch.arange(2 * FEATURES, dtype=torch.float32)[None] * 0.01
        proj = proj + image.mean() + 0.01 * camera.values.sum() + naf.weight + offsets[spec.name]
        return {"global": z.mean(0, keepdim=True)[:, :5].clone(), "proj": proj}

    def cond_tokens_pair(spec_a: Any, spec_b: Any, z: Any, image: Any, naf: Any, camera: Any, coords: Any,
                         grid: int, first_dtype: Any = None) -> tuple:
        """The real one makes both conditionings with NAF's encoder run once; here the two calls, so that the
        taps and the arithmetic are the specification's."""
        CURRENT.tap("cond_tokens_pair", specs=(spec_a.name, spec_b.name))
        a = cond_tokens(spec_a, z, image, naf, camera, coords, grid)
        b = cond_tokens(spec_b, z, image, naf, camera, coords, grid)
        if first_dtype is not None:
            a["proj"] = a["proj"].type(first_dtype)
        return a, b

    def negative(cond: dict) -> dict:
        return {name: torch.zeros_like(value) for name, value in cond.items()}

    proj_cond.Camera, proj_cond.image_tensor, proj_cond.encode = Camera, image_tensor, encode
    proj_cond.cond_dense, proj_cond.cond_tokens, proj_cond.negative = cond_dense, cond_tokens, negative
    proj_cond.cond_tokens_pair = cond_tokens_pair
    proj_cond.clear_grids = lambda: None            # the real one frees the projection grids off the card

    decode = types.ModuleType("pixal3d.pipelines.decode")

    def on_card() -> list[str]:
        return sorted(CURRENT.loader.modules)

    def upsample(decoder: Any, feats: Any, coords: Any) -> Any:
        CURRENT.tap("upsample", lean=getattr(decoder, "lean", False), tokens=coords.shape[0])
        assert decoder.name == "shape_dec" and feats.shape == (coords.shape[0], 32)
        base = coords[:, 1:] * 16 + (feats[:, :3] > 0).int()
        children = torch.cat([base + offset for offset in (0, 5, 9, 12, 14)])      # merge more on a coarser grid
        return torch.cat([torch.zeros(children.shape[0], 1, dtype=torch.int32), children.int()], dim=1)

    def decode_shape(decoder: Any, feats: Any, coords: Any, res: int) -> tuple:
        CURRENT.tap("decode_shape", res=res, on_card=on_card(), lean=getattr(decoder, "lean", False),
                    chunk=getattr(decoder, "chunk_voxels", None))
        assert decoder.name == "shape_dec" and coords.dtype == torch.int32
        vertices = torch.cat([coords[:, 1:].float() / (res // 16) - 0.5 + decoder.weight, feats[:, :3] * 0.01])
        faces = torch.arange(coords.shape[0] // 3 * 3, dtype=torch.int32).reshape(-1, 3)
        return types.SimpleNamespace(vertices=vertices, faces=faces), [feats[:, i:i + 8] > 0 for i in range(4)]

    def decode_texture(decoder: Any, feats: Any, coords: Any, subs: list) -> tuple:
        CURRENT.tap("decode_texture", on_card=on_card(), chunk=getattr(decoder, "chunk_voxels", None),
                    lean=getattr(decoder, "lean", False))
        assert decoder.name == "tex_dec" and len(subs) == 4 and subs[0].dtype == torch.bool
        attrs = torch.sigmoid(feats[:, :6]) * (0.5 + 0.5 * subs[0][:, :6].float()) + decoder.weight * 0.001
        return coords[:, 1:].contiguous(), attrs

    def fill_holes(mesh: Any) -> None:
        CURRENT.tap("fill_holes", on_card=on_card())
        mesh.vertices = torch.cat([mesh.vertices, mesh.vertices.mean(0, keepdim=True)])

    decode.upsample, decode.decode_shape, decode.decode_texture = upsample, decode_shape, decode_texture
    decode.fill_holes = fill_holes
    return {"proj_cond": proj_cond, "decode": decode}


def install_fakes() -> None:
    """Put the stand-in modules where the parts import them from; refuse to hide real ones."""
    import importlib.util

    import pixal3d.pipelines as pipelines

    for name, module in fake_modules().items():
        if importlib.util.find_spec(f"pixal3d.pipelines.{name}") is not None:
            print(f"[fake] note: this worktree has its own pipelines/{name}.py; the stand-in is used all the same")
        sys.modules[f"pixal3d.pipelines.{name}"] = module
        setattr(pipelines, name, module)
    install_hi3d_fakes()


HI3D_NETWORKS = ("hi3d_dino", "hi3d_clip", "hi3d_t512", "hi3d_t1024", "hi3d_vae", "shape_enc")


def install_hi3d_fakes() -> None:
    """Stand-ins for Hi3D's model calls (staged/hi3d.py: its registration, conditioning, voxelizer, sampler, decoder
    and TRELLIS.2's encoder), with the arithmetic chosen as the other fakes': the wrong prop's tensor, network or
    generator state gives other numbers. The coarse mesh, the frames and the guide tables stay the real ones."""
    import numpy as np
    import torch
    from pixal3d.staged import hi3d

    def register(loader: Any, paths: Any) -> None:
        for name in HI3D_NETWORKS:
            loader.register_builder(name, lambda device, name=name: make_net(name))

    def condition(dino: Any, clip: Any, picture: Any) -> Any:
        CURRENT.tap("hi3d_condition", mode=picture.mode)
        assert dino.name == "hi3d_dino" and clip.name == "hi3d_clip" and picture.mode == "RGBA"
        return torch.full((1, 5, 4), float(np.asarray(picture)[..., :3].mean()) / 255) + dino.weight + clip.weight

    def voxelize(vertices: Any, faces: Any, structure: int, budget: int, device: Any) -> tuple:
        v = torch.as_tensor(vertices, dtype=torch.float64)
        low, high = v.min(0).values, v.max(0).values
        center, scale = (low + high) / 2, (structure - 6) / float((high - low).max())
        grid = ((v - center) * scale + structure / 2).floor().clamp(0, structure - 1).int()
        return torch.unique(grid, dim=0), {"center": center, "scale": scale, "res": structure}

    def sample(transformer: Any, steps: Any, tokens: Any, coords: Any, index: int, total: int, generator: Any,
               device: Any, tick: Any, kv_cache: bool = False) -> tuple:
        CURRENT.tap("hi3d_sample", index=index, total=total, tokens=coords.shape[0], kv_cache=kv_cache)
        assert transformer.name == hi3d.TRANSFORMERS[index] and steps == "fake hi3d scheduler"
        feats = torch.randn(coords.shape[0], 32, generator=generator) * 0.1 + tokens.float().mean() \
            + transformer.weight + index
        for step in range(3):
            tick(step + 1, 3)
        rows = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords.int()], dim=1)
        return rows, feats.float()

    def decode(vae: Any, coords: Any, feats: Any, index: int, device: Any, budget_mib: float) -> tuple:
        CURRENT.tap("hi3d_decode", index=index, budget_mib=budget_mib)
        assert vae.name == "hi3d_vae" and budget_mib > 0
        corners = coords[:, 1:].double() * 8 + feats[:, :3].double().clamp(-1, 1)
        faces = np.arange(corners.shape[0] // 3 * 3, dtype=np.int64).reshape(-1, 3)
        return (corners / 100).numpy(), faces, torch.diag(torch.tensor([0.01, 0.01, 0.01, 1.0], dtype=torch.float64))

    def dual_grid(vertices: Any, faces: Any, resolution: int) -> tuple:
        # o_voxel and cumesh read the buffers as rows: a strided mesh reaches them as garbage
        assert vertices.is_contiguous() and faces.is_contiguous(), "the mesh is not laid out in rows"
        fine = torch.unique(((vertices.float().clamp(-0.5, 0.4999) + 0.5) * resolution).floor().int(), dim=0)
        return fine, (fine.float() + 0.5) / resolution, torch.zeros(fine.shape[0], 3, dtype=torch.bool)

    def surface_count(vertices: Any, faces: Any, resolution: int, device: Any) -> tuple:
        # the GPU count stands in for the dual grid's figures: here exactly the fake grid's
        CURRENT.tap("hi3d_surface_count", resolution=resolution)
        fine = dual_grid(torch.as_tensor(np.ascontiguousarray(vertices)), torch.as_tensor(np.ascontiguousarray(faces)),
                         resolution)[0]
        return int(fine.shape[0]), hi3d._tokens(fine, resolution)

    def encode_mesh(encoder: Any, fine: Any, dual: Any, intersected: Any, resolution: int, device: Any) -> Any:
        from pixal3d.modules.sparse import SparseTensor

        CURRENT.tap("hi3d_encode_mesh", resolution=resolution, voxels=fine.shape[0])
        assert encoder.name == "shape_enc"
        coarse, inverse = torch.unique(fine.long() >> 4, dim=0, return_inverse=True)
        offsets = (dual * resolution - fine).float().repeat(1, 11)[:, :32]
        feats = torch.zeros(coarse.shape[0], 32).index_add_(0, inverse, offsets) + encoder.weight
        rows = torch.cat([torch.zeros(coarse.shape[0], 1, dtype=torch.int32), coarse.int()], dim=1)
        return SparseTensor(feats=feats, coords=rows)

    hi3d.register, hi3d.scheduler = register, lambda paths: "fake hi3d scheduler"
    hi3d.condition, hi3d.voxelize, hi3d.sample, hi3d.decode = condition, voxelize, sample, decode
    hi3d.dual_grid, hi3d.encode_mesh, hi3d.surface_count = dual_grid, encode_mesh, surface_count


def make_net(name: str) -> Any:
    """A stand-in network: one weight derived from its name, so that the wrong network gives other numbers."""
    import torch

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.name = name
            self.weight = torch.nn.Parameter(torch.full((1,), (zlib.crc32(name.encode()) % 1000) / 1000.0),
                                             requires_grad=False)
            if name.startswith("ss_flow"):
                self.resolution, self.in_channels = 16, 8
            elif "_flow" in name:
                self.in_channels = 64 if name.startswith("tex_flow") else 32

        def forward(self, x: Any) -> Any:
            if name == "ss_dec":            # logits at 64^3 with exactly 40 of the 16^3 cells above zero
                plane = x[:, :1].float()
                threshold = plane.flatten().kthvalue(plane.numel() - 40).values
                return torch.nn.functional.interpolate(plane - threshold, scale_factor=4, mode="nearest")
            if name == "rembg":
                CURRENT.tap("matte")
                assert x.mode == "RGB"
                image = x.convert("RGBA")
                image.putalpha(200)
                return image
            raise AssertionError(f"the network {name} is not called directly")

    return Net()


class FakeSampler:
    """A dozen steps of arithmetic in place of a denoiser. The result depends on the noise, the conditioning,
    the network and the settings."""

    def __init__(self, stage: str):
        self.stage = stage

    def sample(self, model: Any, noise: Any, cond: Any = None, neg_cond: Any = None, steps: int = 50,
               rescale_t: float = 1.0, guidance_strength: float = 3.0, guidance_rescale: float = 0.0,
               guidance_interval: tuple = (0.0, 1.0), verbose: bool = True, tqdm_desc: str = "Sampling",
               keep_trajectory: bool = True, step_callback: Any = None, **kwargs: Any) -> Any:
        import torch

        concat = kwargs.pop("concat_cond", None)
        assert not kwargs, f"arguments that would reach the model: {sorted(kwargs)}"
        sparse = not torch.is_tensor(noise)
        proj = cond["proj"].feats if sparse else cond["proj"]
        CURRENT.log.append(("sample", self.stage, model.name, CURRENT.now, {
            "steps": steps, "guidance_strength": guidance_strength, "guidance_rescale": guidance_rescale,
            "rescale_t": rescale_t, "guidance_interval": tuple(guidance_interval), "negative": neg_cond is not None,
            "verbose": verbose, "keep_trajectory": keep_trajectory, "dtype": str(proj.dtype),
            "global_dtype": str(cond["global"].dtype), "tokens": proj.shape[0] if sparse else proj.shape[1],
            "noise": (noise.feats if sparse else noise).clone(), "coords": noise.coords.clone() if sparse else None,
            "concat": None if concat is None else concat.feats.clone()}))
        made = CURRENT.log[-1][4]
        if neg_cond is not None:
            zero = neg_cond["proj"].feats if sparse else neg_cond["proj"]
            assert zero.shape == proj.shape and zero.dtype == proj.dtype and not zero.any()
        gain = float(model.weight) + 0.01 * guidance_strength + 0.001 * rescale_t + 0.0001 * guidance_rescale
        x = noise
        for index in range(steps):
            if sparse:
                assert cond["proj"].coords.shape == x.coords.shape
                width = x.feats.shape[1]
                push = proj.float()[:, :width] * 0.05 + cond["global"].float().mean() * 0.01
                if concat is not None:
                    push = push + concat.feats[:, :width] * 0.02
                x = x.replace(feats=x.feats * 0.9 + push + gain * 0.01)
            else:
                field = proj.float().reshape(1, 1, 16, 16, 16, -1)[..., 0]
                x = x * 0.9 + field * 0.5 + cond["global"].float().mean() * 0.01 + gain * 0.01
            if step_callback is not None:
                step_callback(index + 1, steps)
        made["samples"] = (x.feats if sparse else x).clone()
        return types.SimpleNamespace(samples=x, pred_x_t=[], pred_x_0=[])


class FakeLoader:
    """NetworkLoader's contract (specification 1.2) with nothing behind it. Refuses what the real one could not
    do or what would waste the card: a load or a drop while an exception is being handled, a network dropped
    while somebody else still refers to it, two flow networks at once."""

    def __init__(self, world: "World", device: str = "cuda", *, prefetch: str = "cache",
                 max_vram_mib: int | None = None, safety_mib: int = 300, on_event: Any = None):
        self.world, self.device, self.on_event = world, device, on_event
        self.made = {"device": device, "prefetch": prefetch, "max_vram_mib": max_vram_mib}
        self.checkpoints: dict[str, Path] = {}
        self.builders: dict[str, tuple] = {}
        self.modules: dict[str, Any] = {}
        self.cap: tuple | None = None
        self.closed = False
        world.loader = self

    def _log(self, *entry: Any) -> None:
        self.world.log.append((*entry, self.world.now))

    def register_checkpoint(self, name: str, ckpt: Path, convert=None) -> None:
        self.checkpoints[name] = Path(ckpt)

    def register_builder(self, name: str, build: Any, file: Path | None = None) -> None:
        self.builders[name] = (build, file)

    def load(self, name: str, cancelled: Any = None) -> Any:
        from pixal3d.staged.types import Cancelled

        assert sys.exc_info()[0] is None, "a network is loaded while an exception is being handled"
        assert name in self.checkpoints or name in self.builders, f"{name} was never registered"
        assert name not in self.modules, f"{name} is loaded twice"
        self._log("load", name)
        if cancelled is not None and cancelled():
            raise Cancelled()
        pending = self.world.fail_load.get(name)
        if pending:
            raise pending.pop(0)
        if self.on_event is not None:
            self.on_event({"kind": "net_load", "name": name, "bytes_read": 0, "bytes": 1})
        module = self.builders[name][0](self.device) if name in self.builders else make_net(name)
        self.modules[name] = module
        flows = [loaded for loaded in self.modules if "_flow." in loaded]
        assert len(flows) <= 1, f"two flow networks on the card: {flows}"
        if self.on_event is not None:
            self.on_event({"kind": "net_ready", "name": name, "seconds": 0.0, "bytes": 1})
        return module

    def get(self, name: str) -> Any:
        return self.modules.get(name)

    def drop(self, *names: str) -> None:
        assert sys.exc_info()[0] is None, "a network is dropped while an exception is being handled"
        for name in names:
            assert name in self.modules, f"{name} is dropped but not loaded"
            module = self.modules.pop(name)
            gc.collect()
            assert sys.getrefcount(module) <= 2, f"{name} is dropped while something else still refers to it"
            self._log("drop", name)
            if self.on_event is not None:
                self.on_event({"kind": "net_drop", "name": name})

    def drop_all(self) -> None:
        self._log("drop_all")
        self.drop(*list(self.modules))

    def trim(self) -> None:
        assert sys.exc_info()[0] is None, "the cache is trimmed while an exception is being handled"
        self._log("trim")

    def set_cap(self, native_reserve_mib: int = 200, safety_mib: int | None = None) -> int:
        self.cap = (native_reserve_mib, safety_mib)
        self._log("set_cap", native_reserve_mib, safety_mib)
        return self.world.limit

    def clear_cap(self) -> None:
        self.cap = None
        self._log("clear_cap")

    def capped(self) -> bool:
        return self.cap is not None

    def headroom_mib(self) -> int | None:
        return self.world.headroom

    def loaded(self) -> list[str]:
        return list(self.modules)

    def size_mib(self, name: str) -> int | None:
        return self.world.sizes.get(name.split(".")[0], 1) if name in self.modules else None

    def stats(self) -> dict:
        return {"pinned_bytes": 0, "cap_mib": self.world.limit if self.cap else None, "loads": []}

    def prefetch(self, name: str | None, *more: str) -> None:
        self._log("prefetch", name)
        for each in more:
            self._log("prefetch", each)

    def close(self) -> None:
        self.closed = True


class RecordedLoader:
    """--real-loader: the worktree's own NetworkLoader (loader.py, on the CPU) behind the fake world's log. Its
    bookkeeping, events and read-ahead are the real ones; the networks are the stand-ins (a checkpoint is
    registered as a builder of one), and the cap is the fake world's, since a CPU has none."""

    def __init__(self, world: "World", device: str = "cuda", **options: Any):
        from pixal3d.staged.loader import NetworkLoader

        self.world, self.device = world, device
        self.made = {"device": device, "prefetch": options.get("prefetch"), "max_vram_mib": options.get("max_vram_mib")}
        self.real = NetworkLoader(device, **options)
        self.checkpoints: dict[str, Path] = {}
        self.builders: dict[str, tuple] = {}
        self.cap: tuple | None = None
        self.closed = False
        world.loader = self

    def _log(self, *entry: Any) -> None:
        self.world.log.append((*entry, self.world.now))

    @property
    def modules(self) -> dict[str, Any]:
        return {name: self.real.get(name) for name in self.real.loaded()}

    def register_checkpoint(self, name: str, ckpt: Path, convert=None) -> None:
        self.checkpoints[name] = Path(ckpt)
        self.real.register_builder(name, lambda device, name=name: make_net(name))

    def register_builder(self, name: str, build: Any, file: Path | None = None) -> None:
        self.builders[name] = (build, file)
        self.real.register_builder(name, build, file)

    def load(self, name: str, cancelled: Any = None) -> Any:
        assert sys.exc_info()[0] is None, "a network is loaded while an exception is being handled"
        assert self.real.get(name) is None, f"{name} is loaded twice"
        self._log("load", name)
        pending = self.world.fail_load.get(name)
        if pending and not (cancelled is not None and cancelled()):
            raise pending.pop(0)
        module = self.real.load(name, cancelled)
        flows = [loaded for loaded in self.real.loaded() if "_flow." in loaded]
        assert len(flows) <= 1, f"two flow networks on the card: {flows}"
        return module

    def get(self, name: str) -> Any:
        return self.real.get(name)

    def drop(self, *names: str) -> None:
        assert sys.exc_info()[0] is None, "a network is dropped while an exception is being handled"
        for name in names:
            assert self.real.get(name) is not None, f"{name} is dropped but not loaded"
            self._log("drop", name)
        self.real.drop(*names)

    def drop_all(self) -> None:
        self._log("drop_all")
        self.drop(*self.real.loaded())

    def trim(self) -> None:
        assert sys.exc_info()[0] is None, "the cache is trimmed while an exception is being handled"
        self._log("trim")
        self.real.trim()

    def set_cap(self, native_reserve_mib: int = 200, safety_mib: int | None = None) -> int:
        self.cap = (native_reserve_mib, safety_mib)
        self._log("set_cap", native_reserve_mib, safety_mib)
        self.real.set_cap(native_reserve_mib, safety_mib)
        return self.world.limit

    def clear_cap(self) -> None:
        self.cap = None
        self._log("clear_cap")
        self.real.clear_cap()

    def capped(self) -> bool:
        return self.cap is not None

    def loaded(self) -> list[str]:
        return self.real.loaded()

    def size_mib(self, name: str) -> int | None:
        return self.real.size_mib(name)

    def stats(self) -> dict:
        return self.real.stats()

    def prefetch(self, name: str | None, *more: str) -> None:
        self._log("prefetch", name)
        for each in more:
            self._log("prefetch", each)
        self.real.prefetch(name, *more)

    def close(self) -> None:
        self.closed = True
        self.real.close()


class FakeStore:
    """PropStore's contract (specification 1.3) in a dict. Stricter than the real one may be: a key that is not
    there cannot be read, wanted or discarded, a large item cannot be put twice, a closed prop cannot be touched.
    Checkpoints lie in world.disk, which outlives the engine as the work folder outlives a process."""

    def __init__(self, world: "World", work_dir: Path, device: str = "cuda", *, ram_mib: int | None = 768,
                 slots: int = 2, slot_mib: int | None = 256, on_event: Any = None):
        self.world, self.device = world, device
        self.made = {"work_dir": Path(work_dir), "device": device, "ram_mib": ram_mib, "slots": slots,
                     "slot_mib": slot_mib}
        self.props: dict[str, dict] = {}
        self.begun = self.closed = False
        world.store = self

    def _log(self, *entry: Any) -> None:
        self.world.log.append(("store", *entry, self.world.now))

    def _prop(self, prop_id: str) -> dict:
        assert self.begun and not self.closed, "the store is used before begin() or after close()"
        assert prop_id in self.props, f"prop {prop_id} is not open"
        return self.props[prop_id]

    def begin(self, keep: dict[str, str]) -> dict[str, dict]:
        assert not self.begun
        self.begun = True
        disk = self.world.disk
        for prop_id in list(disk):
            if keep.get(prop_id) != disk[prop_id]["request_hash"]:
                del disk[prop_id]
        self._log("begin", dict(keep))
        return {prop_id: {"tensors": dict(entry["tensors"]), "notes": copy.deepcopy(entry["notes"])}
                for prop_id, entry in disk.items()}

    def open_prop(self, prop_id: str) -> None:
        assert self.begun and prop_id not in self.props
        self.props[prop_id] = {}

    def put_small(self, prop_id: str, key: str, value: Any) -> None:
        self._prop(prop_id)[key] = value
        self.world.puts.setdefault(prop_id, {})[key] = copy.deepcopy(value)
        self._log("put_small", prop_id, key)

    def get_small(self, prop_id: str, key: str) -> Any:
        return self._prop(prop_id)[key]

    def put_device(self, prop_id: str, key: str, tensors: dict) -> None:
        assert key not in self._prop(prop_id), f"{key} of {prop_id} is put twice without a discard"
        held = {name: value.detach().cpu().clone() for name, value in tensors.items()}
        self.props[prop_id][key] = held
        self.world.puts.setdefault(prop_id, {})[key] = held
        self._log("put_device", prop_id, key)

    def want(self, prop_id: str, key: str) -> None:
        assert key in self._prop(prop_id), f"{key} of {prop_id} is wanted but not there"
        self._log("want", prop_id, key)

    def pump(self, upload: bool = True) -> None:
        self._log("pump", upload)

    def take(self, prop_id: str, key: str) -> dict:
        self._log("take", prop_id, key)
        taken = {name: value.to(self.device).clone() for name, value in self._prop(prop_id)[key].items()}
        self.world.taken.append((self.world.now, key, [weakref.ref(value) for value in taken.values()]))
        return taken

    def get_large(self, prop_id: str, key: str) -> dict:
        self._log("get_large", prop_id, key)
        return dict(self._prop(prop_id)[key])          # the store's own tensors, as the real one hands them out

    def discard(self, prop_id: str, *keys: str) -> None:
        for key in keys:
            assert key in self._prop(prop_id), f"{key} of {prop_id} is discarded but not there"
            del self.props[prop_id][key]
            self._log("discard", prop_id, key)

    def checkpoint(self, prop_id: str, request_hash: str, tensors: dict, notes: dict) -> None:
        self._prop(prop_id)
        assert request_hash and all(value.device.type == "cpu" for value in tensors.values())
        self.world.disk[prop_id] = {"request_hash": request_hash, "notes": copy.deepcopy(notes),
                                    "tensors": {name: value.clone() for name, value in tensors.items()}}
        self._log("checkpoint", prop_id, sorted(tensors))

    def close_prop(self, prop_id: str) -> None:
        assert sys.exc_info()[0] is None, "a prop is closed while an exception is being handled"
        self._prop(prop_id)
        del self.props[prop_id]
        self.world.disk.pop(prop_id, None)
        self._log("close_prop", prop_id)

    def flush(self) -> None:
        self._log("flush")

    def close(self) -> None:
        self.closed = True

    def stats(self) -> dict:
        return {}


class RecordedStore:
    """--real-store: the worktree's own PropStore (store.py; on the CPU, its files in the scratch folder) behind
    the fake world's log, so that the same tests run against it. Checkpoints are then real files, and a "process"
    that died leaves them in the work folder."""

    def __init__(self, world: "World", work_dir: Path, device: str = "cuda", **options: Any):
        from pixal3d.staged.store import PropStore

        self.world, self.device = world, device
        self.made = {"work_dir": Path(work_dir), "device": device,
                     **{name: options.get(name) for name in ("ram_mib", "slots", "slot_mib")}}
        self.real = PropStore(work_dir, device, **options)
        self.props: dict[str, dict] = {}        # prop id -> the keys it holds (a shadow of the store's own)
        self.closed = False
        world.store = self

    def _log(self, *entry: Any) -> None:
        self.world.log.append(("store", *entry, self.world.now))

    def begin(self, keep: dict[str, str]) -> dict[str, dict]:
        found = self.real.begin(keep)
        self._log("begin", dict(keep))
        return found

    def open_prop(self, prop_id: str) -> None:
        self.real.open_prop(prop_id)
        self.props[prop_id] = {}

    def put_small(self, prop_id: str, key: str, value: Any) -> None:
        self.real.put_small(prop_id, key, value)
        self.props[prop_id][key] = value
        self.world.puts.setdefault(prop_id, {})[key] = copy.deepcopy(value)
        self._log("put_small", prop_id, key)

    def get_small(self, prop_id: str, key: str) -> Any:
        return self.real.get_small(prop_id, key)

    def put_device(self, prop_id: str, key: str, tensors: dict) -> None:
        self.real.put_device(prop_id, key, tensors)
        held = {name: value.detach().cpu().clone() for name, value in tensors.items()}
        self.props[prop_id][key] = held
        self.world.puts.setdefault(prop_id, {})[key] = held
        self._log("put_device", prop_id, key)

    def want(self, prop_id: str, key: str) -> None:
        self.real.want(prop_id, key)
        self._log("want", prop_id, key)

    def pump(self, upload: bool = True) -> None:
        self.real.pump(upload)
        self._log("pump", upload)

    def take(self, prop_id: str, key: str) -> dict:
        self._log("take", prop_id, key)
        taken = self.real.take(prop_id, key)
        self.world.taken.append((self.world.now, key, [weakref.ref(value) for value in taken.values()]))
        return taken

    def get_large(self, prop_id: str, key: str) -> dict:
        self._log("get_large", prop_id, key)
        return self.real.get_large(prop_id, key)

    def discard(self, prop_id: str, *keys: str) -> None:
        self.real.discard(prop_id, *keys)
        for key in keys:
            self.props[prop_id].pop(key)
            self._log("discard", prop_id, key)

    def checkpoint(self, prop_id: str, request_hash: str, tensors: dict, notes: dict) -> None:
        self.real.checkpoint(prop_id, request_hash, tensors, notes)
        self._log("checkpoint", prop_id, sorted(tensors))

    def close_prop(self, prop_id: str) -> None:
        self.real.close_prop(prop_id)
        self.props.pop(prop_id)
        self._log("close_prop", prop_id)

    def flush(self) -> None:
        self.real.flush()
        self._log("flush")

    def close(self) -> None:
        self.real.close()
        self.closed = True

    def stats(self) -> dict:
        return self.real.stats()


class FakeHooks:
    """The caller's side: pictures made from the request's payload, a camera read off the picture, a bundle of
    small tensors, and a `finish` that keeps the decoded mesh."""

    def __init__(self, world: "World"):
        self.world = world
        self.retries = 0

    def prepare(self, req: Any, matte: Callable[[Any], Any]) -> Any:
        from PIL import Image
        from pixal3d.staged.types import Prepared, PropFailed

        self.world.tap("prepare", prop=req.id)
        spec = req.payload
        if spec.get("bad"):
            raise PropFailed("This slice is empty.", "input")
        images = []
        for view in range(spec["views"]):
            red, green, blue = spec["color"]
            image = Image.new("RGB", (64, 64), (red, green, (blue + 40 * view) % 256))
            image.putpixel((view, view), (255, 255, 255))
            image = image.convert("RGBA") if spec.get("own_alpha") else matte(image)
            images.append(image if req.multi else image.convert("RGB"))
            if view == 0:
                cutout = image
        if getattr(req, "geometry", "pixal3d") == "hi3d" and not req.multi:
            return Prepared(images, {}, cutout=cutout)
        return Prepared(images, {"azimuths": [0, 90, 180, 270][:spec["views"]]} if req.multi else {})

    def camera(self, req: Any, image: Any, moge: Any) -> dict:
        self.world.tap("camera", prop=req.id)
        assert moge.name == "moge" and not req.multi
        red, green, _ = image.getpixel((5, 5))
        return {"camera_angle_x": 0.5 + red / 1000, "distance": 1.25 + green / 1000, "mesh_scale": 1.0}

    def bundle(self, req: Any, images: list, meta: dict) -> dict:
        import numpy as np
        import torch

        self.world.tap("bundle", prop=req.id)
        assert req.multi and len(images) == len(meta["azimuths"]) and all(i.mode == "RGBA" for i in images)

        def cond(image: Any, size: int) -> Any:
            array = np.array(image.convert("RGB").resize((size // 64, size // 64))).astype(np.float32) / 255
            return torch.from_numpy(array).permute(2, 0, 1)

        count = len(images)
        angles = torch.tensor(meta["azimuths"], dtype=torch.float32)
        return {"images": {size: torch.stack([cond(image, size) for image in images])[None] for size in (512, 1024)},
                "camera_angle_x": torch.full((1, count), 0.349), "camera_distance": torch.full((1, count), 3.119),
                "transform_matrix": (torch.eye(4)[None, None] * (1 + angles / 360).reshape(1, count, 1, 1)),
                "mesh_scale": 1.0}

    @contextlib.contextmanager
    def prop_scope(self, req: Any, part: str) -> Iterator[dict]:
        assert part == "ss"
        self.retries = 0
        notes: dict = {}
        yield notes
        notes.update(ss_retries=self.retries, ss_blocked=False)

    def finish(self, req: Any, mesh: Any, notes: dict) -> dict:
        self.world.tap("finish", prop=req.id, mesh=mesh)
        assert all(getattr(mesh, name).is_contiguous() for name in ("vertices", "faces", "coords", "attrs")),             "the decoded mesh reaches the hook's native code laid out other than in rows"
        self.world.finished[req.id] = (mesh, copy.deepcopy(notes))
        return {"mode": "multi" if req.multi else "single", "views": req.payload["views"], "warning": None,
                "ss_retries": notes.get("ss_retries", 0)}


class World:
    """One fake installation and what happened in it. `run` builds a new engine each time (a new "process");
    `disk` (the checkpoints) and the tables that make things fail stay."""

    count = 0

    def __init__(self, *, multiview: bool = True):
        World.count += 1
        self.work = SCRATCH / f"work-{World.count}"         # the batch's work folder: one per world
        self.pipeline_dir = pipeline_dir(SCRATCH / ("pipeline" if multiview else "pipeline-sv"), multiview)
        self.log: list[tuple] = []
        self.fail_step: dict[tuple, list] = {}      # (part, prop id, step) -> what the next such step events raise
        self.fail_load: dict[str, list] = {}        # network -> what its next loads raise
        self.taps: dict[str, list] = {}             # name -> functions the fakes call (with keyword arguments)
        self.redraws: dict[str, int] = {}           # prop id -> draws of the sparse structure the guard rejects
        self.cancel_at: tuple | None = None         # (part, prop id, step): ask for a cancel at that step event
        self.cancel = False
        self.limit = 6068                           # what the loader's set_cap answers (MiB)
        self.headroom: int | None = None            # what the loader's headroom_mib answers (None: cannot say)
        self.sizes: dict[str, int] = {}             # the fake loader's size_mib of a loaded network, by its name
        #                                             without the kind (".sv" / ".mv"); 1 when not given
        self.disk: dict[str, dict] = {}
        self.puts: dict[str, dict] = {}             # prop id -> {key: the last value put into the store}
        self.finished: dict[str, tuple] = {}        # prop id -> (DecodedMesh, notes) as hooks.finish got them
        self.now: tuple = ("", None)                # (part, prop id) of the running prop-part
        self.taken: list[tuple] = []                # (prop-part, key, weak references) of what store.take handed out
        self.held: list[tuple] = []                 # taken tensors still alive when another prop-part started
        self.loader: Any = None
        self.store: Any = None
        self.engine: Any = None
        self.hooks: Any = None
        self.results: dict[str, dict] = {}

    def tap(self, name: str, **info: Any) -> None:
        for fn in self.taps.get(name, ()):
            fn(**info)

    def on_event(self, event: Any) -> None:
        self.log.append(("event", event.kind, event.part, event.prop_id, event.step, event.total, event.detail))
        if event.kind in ("prop_start", "prop_rest", "prop_done", "prop_failed"):
            self._released()
        if event.kind == "prop_start":
            self.now = (event.part, event.prop_id)
        if event.kind == "step":
            where = (event.part, event.prop_id, event.step)
            if self.cancel_at == where:
                self.cancel = True
            if self.fail_step.get(where):
                raise self.fail_step[where].pop(0)

    def _released(self) -> None:
        """What a part took from the store lives until its prop-part is over, and not longer: when a prop-part
        starts or has ended, nothing taken before may still be referenced."""
        def alive() -> list[tuple]:
            return [(where, key) for where, key, references in self.taken
                    if any(reference() is not None for reference in references)]

        if alive():
            gc.collect()
        self.held += alive()
        self.taken = []

    def build(self, *, check_rng: bool = False, option_waves: int | None = None, guard: bool = True,
              store_ram_mib: int | None = None, spill: bool = False, hooks: type | None = None,
              paired: bool | None = None, hi3d: bool = False) -> Any:
        """`paired`: EngineOptions.paired_finish (each prop's finish right after its texture); None: the engine's
        own default. The tests of the parts' order pass False, since they spell out every part for all props."""
        from pixal3d.staged import EngineOptions, StagedEngine

        global CURRENT
        CURRENT = self
        self.hooks = (hooks or FakeHooks)(self)
        chosen = {} if paired is None else {"paired_finish": paired}
        if hi3d:
            chosen["hi3d"] = "fake Hi3D paths"
        options = EngineOptions(pipeline_dir=self.pipeline_dir, device="cpu", wave_size=option_waves,
                                check_rng=check_rng, store_ram_mib=store_ram_mib, spill=spill,
                                builders={name: (lambda device, name=name: make_net(name)) for name in SMALL_MODELS},
                                **chosen)
        engine = StagedEngine(options, self.hooks,
                              loader_factory=functools.partial(RecordedLoader if REAL_LOADER else FakeLoader, self),
                              store_factory=functools.partial(RecordedStore if REAL_STORE else FakeStore, self))
        for args in engine.pipeline_args.values():
            args.samplers = {stage: FakeSampler(stage) for stage in args.samplers}
        if guard:
            self._guard(engine)
        self.engine = engine
        return engine

    def _guard(self, engine: Any) -> None:
        """What the launcher's guard does: a wrapper under the engine's own attribute that calls the draw again,
        the third time with other sampler settings."""
        fn = engine.sample_sparse_structure

        def wrapper(cond: Any, resolution: int, num_samples: int = 1, sampler_params: dict | None = None) -> Any:
            base = sampler_params if sampler_params is not None else {}
            rejected = self.redraws.get(self.now[1], 0)
            for attempt in range(1, rejected + 2):
                params = {**base, "steps": 25, "guidance_strength": 5.0} if attempt == 3 else base
                coords = fn(cond, resolution, num_samples, params)
                if attempt <= rejected:
                    self.hooks.retries += 1
            return coords

        engine.sample_sparse_structure = wrapper

    def run(self, requests: list, *, wave_size: int | None = None, resume: bool = False, **build: Any) -> dict:
        engine = self.build(**build)
        self.now = ("", None)
        self.results = engine.run(requests, self.work, on_event=self.on_event,
                                  cancelled=lambda: self.cancel, wave_size=wave_size, resume=resume)
        assert not self.held, f"inputs taken from the store outlived their prop-part: {self.held}"
        left = sorted(path.name for path in self.work.iterdir()) if self.work.is_dir() else []
        assert not left, f"the work folder is not empty after the batch: {left}"
        return self.results

    def checkpoints(self) -> list[str]:
        """The props whose checkpoint lies in the work folder."""
        if REAL_STORE:
            return sorted(path.parent.name for path in self.work.glob("*/final.json"))
        return sorted(self.disk)

    # ---- reading the log ------------------------------------------------------------------------------------------

    def events(self, *kinds: str) -> list[tuple]:
        """(kind, part, prop id, step, total, detail) of the engine's events, all or of the kinds given."""
        return [entry[1:] for entry in self.log if entry[0] == "event" and (not kinds or entry[1] in kinds)]

    def starts(self) -> list[tuple[str, str]]:
        return [(part, prop) for _, part, prop, *_ in self.events("prop_start")]

    def entries(self, *head: Any) -> list[tuple]:
        """The log entries that begin with `head`, e.g. entries("store", "discard") or entries("load")."""
        return [entry for entry in self.log if entry[:len(head)] == head]

    def states(self) -> dict[str, str]:
        return {prop_id: result["state"] for prop_id, result in self.results.items()}

    def signature(self, prop_id: str) -> dict[str, Any]:
        """Every tensor the prop left behind: what the parts stored for it and the decoded mesh."""
        puts = self.puts[prop_id]
        tensors = {f"{key}/{name}": value for key in STORED for name, value in puts[key].items()}
        mesh, _ = self.finished[prop_id]
        tensors.update({f"mesh/{name}": getattr(mesh, name) for name in ("vertices", "faces", "coords", "attrs")})
        return tensors


REQUESTS = {
    "A": dict(seed=11, resolution=1024, multi=False, payload={"color": (200, 30, 40), "views": 1}),
    "B": dict(seed=22, resolution=1536, multi=False, payload={"color": (20, 130, 240), "views": 1, "own_alpha": True}),
    "C": dict(seed=33, resolution=1024, multi=True, payload={"color": (90, 200, 10), "views": 4}),
    "D": dict(seed=44, resolution=1024, multi=True, payload={"color": (250, 250, 100), "views": 2}),
    "E": dict(seed=55, resolution=1024, multi=False, payload={"color": (5, 5, 90), "views": 1}),
    "H": dict(seed=66, resolution=1536, multi=False, geometry="hi3d", payload={"color": (180, 90, 60), "views": 1}),
    "I": dict(seed=77, resolution=1024, multi=False, geometry="hi3d", payload={"color": (60, 180, 90), "views": 1}),
    "M": dict(seed=88, resolution=1024, multi=True, geometry="hi3d", payload={"color": (90, 60, 180), "views": 3}),
}
_alone: dict[str, dict] = {}


def request(name: str, **changes: Any) -> Any:
    from pixal3d.staged import PropRequest

    fields = {"sampler_params": {"ss": {}, "shape": {}, "tex": {}}, **REQUESTS[name[0]], **changes}
    return PropRequest(id=name, **fields)


def requests(names: str, **changes: Any) -> list:
    return [request(name, **changes) for name in names]


def alone(name: str) -> dict[str, Any]:
    """The tensors of prop `name` made by itself in a fresh world: what any batch must reproduce."""
    if name not in _alone:
        world = World()
        world.run([request(name)])
        assert world.states() == {name: "done"}, world.results
        _alone[name] = world.signature(name)
    return _alone[name]


def differing(got: dict[str, Any], want: dict[str, Any]) -> list[str]:
    import torch

    if got.keys() != want.keys():
        return sorted(set(got) ^ set(want))
    return [name for name in want
            if got[name].dtype != want[name].dtype or got[name].shape != want[name].shape
            or not torch.equal(got[name], want[name])]


def expect_alone(world: World, names: str) -> None:
    for name in names:
        different = differing(world.signature(name), alone(name))
        assert not different, f"prop {name} differs from the same prop made alone in: {different}"


def oom() -> BaseException:
    import torch

    return torch.OutOfMemoryError("CUDA out of memory (injected)")


# =====================================================================================================================
# The tests (--fake)
# =====================================================================================================================

@test
def order_every_part_for_all_props_multi_view_first() -> None:
    PARTS = pixal3d_parts()

    world = World()
    world.run(requests("ACBD"), paired=False)
    assert world.states() == dict.fromkeys("ACBD", "done"), world.results
    want = [(part, prop) for part in PARTS for prop in "CDAB" if part != "camera" or prop in "AB"]
    assert world.starts() == want, world.starts()
    brackets = [(kind, part) for kind, part, *_ in world.events("part_start", "part_end")]
    assert brackets == [(kind, part) for part in PARTS for kind in ("part_start", "part_end")], brackets
    for kind, part, _, _, _, detail in world.events("part_start"):
        assert detail["props"] == [prop for prop in "CDAB" if part != "camera" or prop in "AB"], (part, detail)
    assert list(world.results) == list("ACBD")          # the answer keeps the request's order
    for index, entry in enumerate(world.log):
        if entry[:2] == ("event", "part_end"):
            before = [earlier for earlier in world.log[:index] if earlier[0] in ("store", "event")][-1]
            assert before[:2] == ("store", "flush"), (entry[2], before)
    places = [(part, prop, detail["position"], detail["of"]) for _, part, prop, _, _, detail
              in world.events("prop_start") if part in ("camera", "hr")]
    assert places == [("camera", "A", 1, 2), ("camera", "B", 2, 2), ("hr", "C", 1, 4), ("hr", "D", 2, 4),
                      ("hr", "A", 3, 4), ("hr", "B", 4, 4)], places
    expect_alone(world, "ABCD")


@test
def paired_finish_decodes_each_prop_right_after_its_texture() -> None:
    """With EngineOptions.paired_finish (the default) the two last parts go prop by prop: a prop is decoded and
    handed to its finish as soon as its texture is sampled, the texture network leaving the card for the decode
    and read again for the next prop; the brackets of the two parts overlap; props resumed from a checkpoint go
    first; the results are those of the props made alone; a process that dies in a prop's finish leaves that
    prop's checkpoint alone, since the props after it have not sampled their texture yet."""
    PARTS = pixal3d_parts()

    world = World()
    world.run(requests("ACBD"))
    assert world.states() == dict.fromkeys("ACBD", "done"), world.results
    earlier = [(part, prop) for part in PARTS[:-2] for prop in "CDAB" if part != "camera" or prop in "AB"]
    # the pair's groups go biggest shape first (parts.pair_order): the 1536 prop B before A
    assert world.starts() == earlier + [(part, prop) for prop in "CDBA" for part in ("tex", "finish")], world.starts()
    brackets = [(kind, part) for kind, part, *_ in world.events("part_start", "part_end")]
    assert brackets[:-4] == [(kind, part) for part in PARTS[:-2] for kind in ("part_start", "part_end")], brackets
    assert brackets[-4:] == [("part_start", "tex"), ("part_start", "finish"), ("part_end", "tex"),
                             ("part_end", "finish")], brackets
    for kind, part, _, _, _, detail in world.events("part_start"):
        assert detail["props"] == [prop for prop in "CDAB" if part != "camera" or prop in "AB"], (part, detail)
    places = [(part, prop, detail["position"], detail["of"]) for _, part, prop, _, _, detail
              in world.events("prop_start") if part in ("tex", "finish")]
    assert places == [("tex", "C", 1, 4), ("finish", "C", 1, 4), ("tex", "D", 2, 4), ("finish", "D", 2, 4),
                      ("tex", "B", 3, 4), ("finish", "B", 3, 4), ("tex", "A", 4, 4), ("finish", "A", 4, 4)], places
    for index, entry in enumerate(world.log):
        if entry[:2] == ("event", "part_end"):
            before = [earlier for earlier in world.log[:index] if earlier[0] in ("store", "event")][-1]
            assert before[:2] == ("store", "flush"), (entry[2], before)
    # the texture network leaves the card before each decode (read again for the next prop of its kind) and
    # nothing of the texture part is on the card when a decode starts
    loads = [name for _, name, _ in world.entries("load")]
    assert loads[loads.index("tex_flow.mv"):] == ["tex_flow.mv", "shape_dec", "tex_dec", "tex_flow.mv", "shape_dec",
                                                  "tex_dec", "tex_flow.sv", "shape_dec", "tex_dec", "tex_flow.sv",
                                                  "shape_dec", "tex_dec"], loads
    held: set[str] = set()
    for entry in world.log:
        if entry[0] == "load":
            held.add(entry[1])
        elif entry[0] == "drop":
            held.discard(entry[1])
        elif entry[0] == "event" and entry[1] == "prop_start" and entry[2] == "finish":
            assert not held, (entry[3], held)
    # read ahead: the finish's decoders and then the next kind's texture network as the group begins (round P: the
    # decoders first, every prop's finish loads them before that network), the group's own again between its pairs
    # (for the next prop), the decoders and the next kind's again after its last pair; in the last group the
    # decoders (asked already: not again as it begins), its own between its pairs, the decoders after its last
    # pair; then the first network of the next batch (background removal), whose read is left to go on when the
    # batch ends; and that one first of all when the engine was made
    ahead = [name for _, name, _ in world.entries("prefetch")]
    assert ahead[0] == "rembg" and ahead[ahead.index("tex_flow.mv"):] == [
        "tex_flow.mv", "shape_dec", "tex_dec", "tex_flow.sv", "tex_flow.mv", "shape_dec", "tex_dec", "tex_flow.sv",
        "tex_flow.sv", "shape_dec", "tex_dec", "rembg"], ahead      # a group's networks are read ahead in turn
    assert not world.loader.modules and not world.loader.capped()
    expect_alone(world, "ABCD")

    # a death in a prop's finish: its own checkpoint stays, the props after it have no texture yet; resumed, it
    # goes first, before the props that still sample their texture
    class Died(SystemExit):
        pass

    def die(prop: str, mesh: Any) -> None:
        if prop == "B":
            raise Died()

    world = World()
    world.taps["finish"] = [die]
    try:
        world.run([request(name, request_hash=f"hash-{name}") for name in "ABC"])     # C, then B (dies), A never
    except Died:
        pass
    else:
        raise AssertionError("the simulated process death was swallowed")
    assert sorted(world.finished) == ["C"] and world.checkpoints() == ["B"], (world.finished, world.checkpoints())
    world.taps.clear()
    world.log.clear()
    world.finished.clear()
    world.run([request("A", request_hash="hash-A"), request("B", request_hash="hash-B")], resume=True)
    assert world.states() == {"A": "done", "B": "done"}, world.results
    assert [prop for _, _, prop, *_ in world.events("prop_resumed")] == ["B"]
    assert [(part, prop) for part, prop in world.starts() if part in ("tex", "finish")] == \
        [("finish", "B"), ("tex", "A"), ("finish", "A")], world.starts()
    assert world.results["B"]["result"]["resumed"] is True and world.results["A"]["result"]["resumed"] is False
    want = {name: value for name, value in alone("B").items() if name.startswith("mesh/")}
    got = {f"mesh/{name}": getattr(world.finished["B"][0], name) for name in ("vertices", "faces", "coords", "attrs")}
    assert not differing(got, want), differing(got, want)      # the resumed prop: its mesh (its coords lack the
    expect_alone(world, "A")                                    # voxel figure a checkpoint never carries)
    assert not world.checkpoints() and not world.store.props


CARD_SIZES = {"tex_flow": 2680, "shape_dec": 905, "tex_dec": 905}   # MiB the networks take on the card (RTX 3060)


@test
def the_pair_keeps_the_texture_network_and_the_decoders_on_the_card_where_there_is_room() -> None:
    """parts.pair_keep: where the caps leave room for everything that runs beside them, the texture network stays
    on the card through a prop's finish (it is not read again for the next prop of the group) and both decoders
    from one prop's finish to the next; where they do not - a card of 7.2 GiB - nothing stays, as before. What is
    kept counts against the finish's cap, against the room a finish step is offered, and goes at any failure."""
    import torch
    from pixal3d.staged import parts
    from pixal3d.staged.runner import PropState

    def batch(limit: int, **build: Any) -> tuple[World, dict[tuple, list[str]]]:
        world = World()
        world.headroom, world.limit, world.sizes = 12_000, limit, dict(CARD_SIZES)
        seen: dict[tuple, list[str]] = {}
        world.taps["decode_shape"] = [lambda on_card, **info: seen.setdefault(("decode", CURRENT.now[1]), on_card)]
        world.taps["fill_holes"] = [lambda on_card: seen.setdefault(("fill", CURRENT.now[1]), on_card)]
        world.run(requests("ACBD"), **build)
        assert world.states() == dict.fromkeys("ACBD", "done"), world.results
        expect_alone(world, "ACBD")
        assert not world.loader.modules and not world.loader.capped()
        return world, seen

    def loads(world: World) -> list[str]:
        names = [name for _, name, _ in world.entries("load")]
        return names[names.index("tex_flow.mv"):]

    both = ["shape_dec", "tex_dec"]
    if REAL_LOADER:
        # the worktree's own loader on the CPU has no card to judge: nothing is kept, as before
        world, seen = batch(20_000)
        assert loads(world) == ["tex_flow.mv", *both] * 2 + ["tex_flow.sv", *both] * 2, loads(world)
        return
    # room for all: each group's texture network read once, the decoders loaded once per group (the last prop of a
    # group keeps nothing: the next group's networks come)
    world, seen = batch(20_000)
    assert loads(world) == ["tex_flow.mv", *both, "tex_flow.sv", *both], loads(world)
    assert seen == {("decode", "C"): [*both, "tex_flow.mv"], ("fill", "C"): [*both, "tex_flow.mv"],
                    ("decode", "D"): both, ("fill", "D"): [],
                    ("decode", "B"): [*both, "tex_flow.sv"], ("fill", "B"): [*both, "tex_flow.sv"],
                    ("decode", "A"): both, ("fill", "A"): []}, seen
    log = world.log
    turn = [index for index, entry in enumerate(log) if entry[:3] == ("event", "prop_start", "tex")]
    assert not [entry for entry in log[turn[0]:turn[1]] if entry[:2] == ("prefetch", "tex_flow.mv")], \
        "the texture network kept on the card is not read ahead again"

    # the caps of a card of 7.2 GiB (the cap of `tex` 6,368 MiB, of the decode 6,068): the texture network stays
    # through the first prop's finish of each group (its steps fit beside it since round M's reserve figures), the
    # decoders are loaded for every prop
    world, seen = batch(6_368)
    assert loads(world) == ["tex_flow.mv", *both, *both, "tex_flow.sv", *both, *both], loads(world)
    assert seen == {("decode", "C"): [*both, "tex_flow.mv"], ("fill", "C"): ["tex_flow.mv"],
                    ("decode", "D"): both, ("fill", "D"): [],
                    ("decode", "B"): [*both, "tex_flow.sv"], ("fill", "B"): ["tex_flow.sv"],
                    ("decode", "A"): both, ("fill", "A"): []}, seen
    # a card of 6 GB (the cap of `tex` 4,664 MiB): nothing is kept
    world, seen = batch(4_664)
    assert loads(world) == ["tex_flow.mv", *both] * 2 + ["tex_flow.sv", *both] * 2, loads(world)
    assert all(on_card == (both if what == "decode" else []) for (what, _), on_card in seen.items()), seen

    # room for the texture network beside the finish, not for the decoders as well
    world, seen = batch(7_500)
    assert loads(world) == ["tex_flow.mv", *both, *both, "tex_flow.sv", *both, *both], loads(world)
    assert seen[("decode", "C")] == [*both, "tex_flow.mv"] and seen[("fill", "C")] == ["tex_flow.mv"], seen

    # no room for the next prop's texture sampler beside the decoders (as the pair judges it, after a texture):
    # the texture network alone stays
    kept = parts.sampler_mib
    parts.sampler_mib = lambda tokens, *rest: 1e9 if CURRENT.now[0] == "tex" else kept(tokens, *rest)
    try:
        world, seen = batch(20_000)
        assert loads(world) == ["tex_flow.mv", *both, *both, "tex_flow.sv", *both, *both], loads(world)
    finally:
        parts.sampler_mib = kept

    # a failure lets go of what was kept: the retry finds the card as it would without it. An out-of-memory in D's
    # texture (the decoders kept beside it) and in C's finish (the texture network kept beside it)
    for where, gone in ((("tex", "D", 3), both), (("finish", "C", 1), ["tex_flow.mv"])):
        world = World()
        world.headroom, world.limit, world.sizes = 12_000, 20_000, dict(CARD_SIZES)
        world.fail_step[where] = [oom()]
        world.run(requests("ACBD"))
        assert world.states() == dict.fromkeys("ACBD", "done"), (where, world.results)
        expect_alone(world, "ACBD")
        log = world.log
        failed = next(index for index, entry in enumerate(log) if entry[:5] == ("event", "step", *where))
        retry = next(index for index, entry in enumerate(log) if index > failed and entry[0] == "set_cap")
        dropped = sorted(entry[1] for entry in log[failed:retry] if entry[0] == "drop")
        assert dropped == sorted(gone), (where, dropped)

    # what is kept is counted out of the room a finish step is offered (parts.tail_room)
    ram = parts.ram_available_mib
    parts.ram_available_mib = lambda: 60_000
    try:
        world = World()
        world.headroom, world.limit, world.sizes = 12_000, 20_000, dict(CARD_SIZES)
        world.run(requests("AEB"), hooks=RoomHooks)
        assert world.states() == dict.fromkeys("AEB", "done"), world.results
        rooms = {(prop, step): room for prop, step, room in world.hooks.rooms}
        first = 12_000 - (parts.NATIVE_RESERVE["finish"] + parts.FINISH_SAFETY_MIB) - parts.HOLD_BESIDE_MIB \
            - parts.HOLD_SLACK_MIB
        # B (1536) first; its first step runs right after its finish, with the texture network and both decoders
        # kept for A; E is the last prop: nothing is kept beside its finish
        assert rooms[("B", "gpu1")] == first - sum(CARD_SIZES.values()), (rooms, first)
        assert rooms[("E", "gpu2")] == first, rooms
        expect_alone(world, "AEB")
    finally:
        parts.ram_available_mib = ram

    # the rule itself, with the figures of the test set and a bf16 texture network: a 1536 prop (9 M voxels; the
    # next one 31 k tokens) on the full RTX 3060 (the cap of `tex` 10,666 MiB) keeps everything, the bulkiest
    # (12.3 M voxels) the texture network alone; under the caps of a card of 7.2 GiB
    # (6,624 / 6,368) nothing, nor a 1024 prop (3 M voxels); a prop in its steps with a large native reserve, or
    # a CPU run, keeps nothing
    class Loader:
        safety_mib = 300

        def __init__(self, cap: int, headroom: int | None = 12_000):
            self.cap, self.room = cap, headroom

        def stats(self) -> dict:
            return {"cap_mib": self.cap}

        def headroom_mib(self) -> int | None:
            return self.room

        def get(self, name: str) -> Any:
            return object() if name == "tex_flow.sv" else None

        def size_mib(self, name: str) -> int | None:
            return CARD_SIZES.get(name.split(".")[0])

    class Store:
        def __init__(self, voxels: int, tokens: int):
            self.voxels, self.tokens = voxels, tokens

        def get_small(self, prop_id: str, key: str) -> dict:
            assert key == "coords_hr"
            return {"coords": torch.zeros(self.tokens, 4, dtype=torch.int32), "res": torch.tensor([1536]),
                    "voxels": torch.tensor([self.voxels])}

    def rule(cap: int, voxels: int, natives: tuple = (), headroom: int | None = 12_000) -> frozenset:
        ctx = types.SimpleNamespace(loader=Loader(cap, headroom), store=Store(voxels, 31_000), part="tex")
        return parts.pair_keep(ctx, PropState(req=request("A")), PropState(req=request("E")), list(natives))

    everything = frozenset({"tex_flow.sv", "shape_dec", "tex_dec"})
    # (the steps of a prop foreseen at 35 M faces reserve ~3.3 GB through PyTorch, steps_torch_mib, beside their
    # native reserve of 3.7 GB, so a bf16 texture network of 2,680 MiB leaves no room for the decoders as well; at
    # 27 M faces everything fits since round M's reserve figures; see also
    # the_pair_counts_what_the_steps_of_a_bulky_prop_reserve_through_pytorch)
    assert rule(10_666, 16_000_000) == frozenset({"tex_flow.sv"})
    assert rule(10_666, 12_300_000) == everything
    assert rule(10_666, 9_000_000) == everything
    assert rule(6_624, 12_300_000) == rule(6_368, 12_300_000) == rule(6_624, 3_000_000) == frozenset()
    assert rule(10_666, 12_300_000, natives=(9_000,)) == frozenset()
    assert rule(10_666, 12_300_000, headroom=None) == frozenset()


@test
def background_removal_is_read_ahead_when_the_engine_is_made_and_after_the_last_network() -> None:
    """The first network a batch loads (parts.FIRST_NETWORK, background removal) is read into the file cache when
    the engine is made, and again after the batch's last network has been read (its last wave), for the next
    batch: that read is left to go on when the batch ends. A cancelled batch stops reading; release() too."""
    world = World()
    engine = world.build()
    assert [entry[1] for entry in world.entries("prefetch")] == ["rembg"] and not world.loader.modules
    world.log.clear()
    world.now = ("", None)
    world.results = engine.run(requests("AC"), world.work, on_event=world.on_event, cancelled=lambda: False)
    ahead = [entry[1] for entry in world.entries("prefetch")]
    assert ahead[-1] == "rembg" and ahead.count("rembg") == 1 and None not in ahead, ahead
    log = world.log
    asked = next(index for index, entry in enumerate(log) if entry[:2] == ("prefetch", "rembg"))
    decoded = [index for index, entry in enumerate(log) if entry[:2] == ("load", "tex_dec")]
    assert decoded[-1] < asked, "asked for once the batch's last network was loaded"
    engine.release()
    assert world.entries("prefetch")[-1][1] is None

    world = World()
    world.run(requests("ABCDE"), wave_size=2)
    ahead = [entry[1] for entry in world.entries("prefetch")]
    assert ahead.count("rembg") == 2 and ahead[-1] == "rembg", ahead       # when made, and after the last wave

    world = World()
    world.cancel_at = ("tex", "C", 5)
    world.run(requests("AC"))
    ahead = [entry[1] for entry in world.entries("prefetch")]
    assert ahead[-1] is None and ahead.count("rembg") == 1, ahead


@test
def the_flows_kernel_libraries_are_imported_ahead_on_a_thread_for_a_card() -> None:
    """An engine for the card imports the libraries the flows' first pass would import on the GPU thread (triton,
    flash_attn, and SageAttention when its switch is on) on a thread of its own when it is made; a library that
    fails is left to the first pass. An engine for the CPU starts no such thread."""
    from pixal3d.modules.attention import config
    from pixal3d.staged import engine as engine_module

    before = {thread.ident for thread in threading.enumerate()}
    World().build()
    assert not [thread for thread in threading.enumerate()
                if thread.name == "p3d-import-ahead" and thread.ident not in before]
    asked: list[str] = []

    def imported(name: str) -> None:
        asked.append(name)
        if name == "flash_attn":
            raise ImportError("not built for this card")

    real, sage = engine_module.importlib.import_module, config.SAGE
    engine_module.importlib.import_module = imported
    try:
        for switch in (True, False):
            config.SAGE = switch
            asked.clear()
            engine_module._import_ahead()
            assert asked == ["triton", "flash_attn", *(["sageattention"] if switch else [])], asked
    finally:
        engine_module.importlib.import_module, config.SAGE = real, sage


@test
def the_decoders_give_their_cache_back_only_where_the_decode_needs_it() -> None:
    """decoder.lean (each level's cache given back, the results moved through RAM) is set for a decode only where
    the decode would not fit the room under the cap without it (parts.lean_needed), and on the ladder's tight rung;
    the up-sampling keeps it. The values are the same either way."""
    from pixal3d.staged import parts

    def leans(**build: Any) -> tuple[World, list[tuple]]:
        seen: list[tuple] = []
        world = World()
        for name in ("upsample", "decode_shape", "decode_texture"):
            world.taps[name] = [lambda name=name, **info: seen.append((name, CURRENT.now[1], info["lean"]))]
        return world, seen

    world, seen = leans()
    world.run(requests("AE"))
    assert seen == [("upsample", "A", True), ("upsample", "E", True), ("decode_shape", "A", False),
                    ("decode_texture", "A", False), ("decode_shape", "E", False), ("decode_texture", "E", False)], seen
    expect_alone(world, "AE")

    kept = parts.UNLEAN_WHOLE, parts.UNLEAN_BOXES
    parts.UNLEAN_WHOLE = parts.UNLEAN_BOXES = 1e6       # no room for a decode without the trims
    try:
        world, seen = leans()
        world.run(requests("AE"))
        assert all(lean for _, _, lean in seen), seen
        expect_alone(world, "AE")
    finally:
        parts.UNLEAN_WHOLE, parts.UNLEAN_BOXES = kept

    world, seen = leans()
    world.fail_step[("finish", "A", 1)] = [oom()]
    world.run(requests("AE"))
    assert [entry for entry in seen if entry[0] == "decode_shape"] == [
        ("decode_shape", "A", False), ("decode_shape", "A", True), ("decode_shape", "E", False)], seen
    expect_alone(world, "AE")

    # the figures: in boxes, a 1536 prop of the test set (12.3 M voxels) needs the trims under the decode cap of a
    # card of 7.2 GiB (6,324 MiB), not under the full RTX 3060's (10,366), nor with the texture network beside it
    # there (2,708 MiB); a shape on whole tables (3 M voxels) needs them on the small card
    assert parts.lean_needed(12_300_000, 2, parts.CHUNK_VOXELS, 6_324)
    assert not parts.lean_needed(12_300_000, 2, parts.CHUNK_VOXELS, 10_366)
    assert not parts.lean_needed(12_300_000, 2, parts.CHUNK_VOXELS, 10_366 - 2_708)
    assert parts.lean_needed(3_000_000, 2, 0, 6_324) and not parts.lean_needed(3_000_000, 2, 0, 10_366)
    assert not parts.lean_needed(12_300_000, 2, parts.CHUNK_VOXELS, 0)       # a CPU run: nothing to judge


@test
def results_do_not_depend_on_the_batch_its_order_or_its_waves() -> None:
    for names in ("CABD", "DBAC", "BEA", "AEB"):
        for wave_size in (None, 1, 2):
            world = World()
            world.run(requests(names), wave_size=wave_size)
            assert world.states() == dict.fromkeys(names, "done"), (names, wave_size, world.results)
            expect_alone(world, names)
            assert not world.store.props and world.store.closed and not world.loader.modules


@test
def waves_finish_their_props_before_the_next_wave_starts() -> None:
    world = World()
    world.run(requests("ABCDE"), wave_size=2)           # run order C D | A B | E
    order = [prop for part, prop in world.starts() if part == "finish"]
    assert order == list("CDBAE"), order           # within a wave the pair goes biggest shape first: B before A
    kinds = [(kind, prop) for kind, _, prop, *_ in world.events("prop_start", "prop_done")]
    assert kinds.index(("prop_done", "D")) < kinds.index(("prop_start", "A"))
    assert kinds.index(("prop_done", "B")) < kinds.index(("prop_start", "E"))
    waves = [(detail["wave"], detail["waves"]) for *_, detail in world.events("part_start")]
    assert waves[0] == (1, 3) and waves[-1] == (3, 3) and waves == sorted(waves), waves
    # the engine's own default applies when run() names no wave size; 0 means "no waves" either way
    for run_waves, option_waves, count in ((None, 2, 3), (0, 2, 1), (None, None, 1), (None, 0, 1), (1, None, 5)):
        world = World()
        world.run(requests("ABCDE"), wave_size=run_waves, option_waves=option_waves)
        assert len({detail["wave"] for *_, detail in world.events("part_start")}) == count, (run_waves, option_waves)
        # waves of one prop have nothing to order; otherwise the sv group's pair goes B (1536) first
        assert [prop for part, prop in world.starts() if part == "finish"] == list("CDABE" if run_waves == 1 else "CDBAE")
        expect_alone(world, "ABCDE")


@test
def networks_are_on_the_card_only_for_their_part() -> None:
    world = World()
    world.run(requests("ACBD"), paired=False)
    loads = [name for _, name, _ in world.entries("load")]
    assert loads == ["rembg", "moge", "dino", "ss_flow.mv", "ss_dec", "ss_flow.sv", "naf", "lr_flow.mv",
                     "lr_flow.sv", "shape_dec", "naf", "hr_flow.mv", "hr_flow.sv", "tex_flow.mv", "tex_flow.sv",
                     *["shape_dec", "tex_dec"] * 4], loads
    # what is on the card when each prop-part starts: the part's own networks, nothing of an earlier part
    held: set[str] = set()
    allowed = {"cutout": {"rembg"}, "camera": {"moge"}, "features": {"dino"}, "ss": {"ss_flow", "ss_dec"},
               "cond_lr": {"naf"}, "lr": {"lr_flow"}, "upsample": {"shape_dec"}, "cond_hr": {"naf"},
               "hr": {"hr_flow"}, "tex": {"tex_flow"}, "finish": {"shape_dec", "tex_dec"}}
    for entry in world.log:
        if entry[0] == "load":
            held.add(entry[1])
        elif entry[0] == "drop":
            held.discard(entry[1])
        elif entry[0] == "event" and entry[1] == "prop_start":
            part, prop = entry[2], entry[3]
            kind = "mv" if prop in "CD" else "sv"
            assert {name.split(".")[0] for name in held} <= allowed[part], (part, prop, held)
            assert all(name.endswith(kind) for name in held if "." in name), (part, prop, held)
    assert not world.loader.modules and not world.loader.capped()
    for index, entry in enumerate(world.log):
        if entry[:2] == ("event", "prop_rest"):
            before = [earlier for earlier in world.log[:index] if earlier[0] != "store"][-1]
            assert (before[0] == "trim") == (entry[2] in ("features", "cond_lr", "upsample", "cond_hr")), entry[:4]
    ahead = [name for _, name, _ in world.entries("prefetch")]
    assert ahead == ["rembg", "moge", "dino", "ss_flow.mv", "ss_dec", "ss_flow.sv", "lr_flow.mv", "lr_flow.sv",
                     "shape_dec", "hr_flow.mv", "hr_flow.sv", "tex_flow.mv", "tex_flow.sv", "shape_dec", "tex_dec",
                     "rembg"], ahead
    # a batch whose pictures carry their own transparency never loads background removal
    world = World()
    world.run([request("B"), request("E", payload={"color": (5, 5, 90), "views": 1, "own_alpha": True})])
    assert "rembg" not in [name for _, name, _ in world.entries("load")]


@test
def ladder_tight_then_cap_lifted_then_failed() -> None:
    # one out-of-memory failure: the second attempt (tight) succeeds, same result, not slow
    world = World()
    world.fail_step[("hr", "A", 3)] = [oom()]
    world.run(requests("AE"))
    assert world.states() == {"A": "done", "E": "done"}
    assert world.results["A"]["result"]["slow"] == [] and not world.events("prop_spilled")
    expect_alone(world, "AE")
    caps = [entry for entry in world.entries("set_cap") if entry[-1] == ("hr", "A")]
    assert [entry[1:3] for entry in caps] == [(200, None), (200, None)], caps
    samples = [entry for entry in world.entries("sample", "shape") if entry[3] == ("hr", "A")]
    assert len(samples) == 2
    assert len(world.entries("store", "take", "A", "cond_hr")) == 1, "the second attempt reuses what the first took"
    # the tight attempt does not bring the next prop's inputs to the card ahead of time
    log = world.log
    second = max(index for index, entry in enumerate(log) if entry[:2] == ("sample", "shape")
                 and entry[3] == ("hr", "A"))
    done = next(index for index, entry in enumerate(log) if entry[:4] == ("event", "prop_rest", "hr", "A"))
    assert [entry[2] for entry in log[second:done] if entry[:2] == ("store", "pump")] == [False] * 12
    first = min(index for index, entry in enumerate(log) if entry[:2] == ("sample", "shape")
                and entry[3] == ("hr", "A"))
    assert [entry[2] for entry in log[first:second] if entry[:2] == ("store", "pump")] == [True, True, False], \
        "the failed attempt gives back what was sent ahead for the next prop"
    assert [entry for entry in log[first:second] if entry[0] == "trim"], "no trim between the attempts"

    # two failures: the prop fails with "oom" after the tight attempt, the cap is never lifted (the default: nothing
    # spills into shared memory), the other prop is untouched
    world = World()
    world.fail_step[("hr", "A", 3)] = [oom(), oom()]
    world.run(requests("AE"))
    assert world.states() == {"A": "failed", "E": "done"}
    assert world.results["A"]["error"]["kind"] == "oom", world.results["A"]
    assert not world.events("prop_spilled") and not world.entries("clear_cap")[:-1], world.entries("clear_cap")
    assert not world.fail_step[("hr", "A", 3)], "both attempts ran"
    expect_alone(world, "E")

    # two failures with a spill allowed: the cap is lifted for this prop-part, the prop is done and marked slow;
    # the cap comes back
    world = World()
    world.fail_step[("hr", "A", 3)] = [oom(), oom()]
    world.run(requests("AE"), spill=True)
    assert world.states() == {"A": "done", "E": "done"}
    assert world.results["A"]["result"]["slow"] == ["hr"] and world.results["E"]["result"]["slow"] == []
    assert [(part, prop) for _, part, prop, *_ in world.events("prop_spilled")] == [("hr", "A")]
    expect_alone(world, "AE")
    lifted = next(index for index, entry in enumerate(world.log) if entry[0] == "clear_cap")
    after = next(entry for entry in world.log[lifted + 1:] if entry[0] in ("set_cap", "clear_cap"))
    assert after[0] == "set_cap" and after[-1] == ("hr", "E"), after
    assert world.finished["A"][1]["slow"] == ["hr"]         # the hook is told as well

    # three failures with a spill allowed: this prop failed, the other is untouched
    world = World()
    world.fail_step[("lr", "A", 12)] = [oom(), oom(), oom()]
    world.run(requests("AE"), spill=True)
    assert world.states() == {"A": "failed", "E": "done"}
    assert world.results["A"]["error"]["kind"] == "oom", world.results["A"]
    expect_alone(world, "E")

    # anything but out of memory is not tried again
    world = World()
    world.fail_step[("tex", "E", 1)] = [ValueError("no"), ValueError("never raised")]
    world.run(requests("AE"))
    assert world.states() == {"A": "done", "E": "failed"}
    assert world.results["E"]["error"] == {"kind": "error", "detail": "ValueError: no"}
    assert len(world.fail_step[("tex", "E", 1)]) == 1


@test
def ladder_of_finish_one_decoder_at_a_time_then_no_cap() -> None:
    from pixal3d.staged.parts import glb_reserve

    from pixal3d.staged.parts import GLB_RESERVE_BASE_MIB as base, GLB_RESERVE_MIB_PER_MFACE as slope

    assert glb_reserve(8_700_000) == base + slope * 87 // 10 and glb_reserve(0) == base
    both = ["shape_dec", "tex_dec"]
    normal = [("decode_shape", both), ("decode_texture", both), ("fill_holes", []), ("hook", [])]
    tight = [("decode_shape", ["shape_dec"]), ("decode_texture", ["tex_dec"]), ("fill_holes", []), ("hook", [])]
    decode_cap = (300, 500)
    hook_cap = {prop: (glb_reserve(alone(prop)["mesh/faces"].shape[0]), 500) for prop in "AE"}

    for failures in (0, 1, 2, 3):
        world = World()
        world.limit = 20000                         # roomy: both decoders may share the card in the normal mode
        seen: list[tuple] = []                      # (what, networks on the card, the loader's cap) in A's finish
        left = [oom() for _ in range(failures)]

        def record(what: str, world: World = world, seen: list = seen) -> Callable[..., None]:
            def tap(**info: Any) -> None:
                if world.now == ("finish", "A"):
                    seen.append((what, sorted(world.loader.modules), world.loader.cap))
            return tap

        def fail(prop: str, mesh: Any, left: list = left) -> None:
            if prop == "A" and left:
                raise left.pop(0)

        for name in ("decode_shape", "decode_texture", "fill_holes"):
            world.taps[name] = [record(name)]
        world.taps["finish"] = [record("hook"), fail]
        world.run(requests("AE"), spill=True)

        attempts = [normal, tight, tight][:min(failures, 2) + 1]
        assert [entry[:2] for entry in seen] == [step for attempt in attempts for step in attempt], (failures, seen)
        caps = [entry[2] for entry in seen]
        capped = [decode_cap, decode_cap, decode_cap, hook_cap["A"]]
        assert caps == capped * min(failures + 1, 2) + [None] * (4 if failures >= 2 else 0), (failures, caps)
        if failures == 3:
            assert world.states() == {"A": "failed", "E": "done"} and world.results["A"]["error"]["kind"] == "oom"
        else:
            assert world.states() == {"A": "done", "E": "done"}
            assert world.results["A"]["result"]["slow"] == (["finish"] if failures == 2 else [])
            expect_alone(world, "A")
        # the next prop gets its caps back, whatever happened to this one
        other = [entry[1:3] for entry in world.entries("set_cap") if entry[-1] == ("finish", "E")]
        assert other == [decode_cap, hook_cap["E"]], (failures, other)
        expect_alone(world, "E")
        assert world.results["E"]["result"]["slow"] == []

    # an attempt that fails while both decoders are on the card leaves the tight attempt one at a time
    world = World()
    world.limit = 20000
    held: list[list[str]] = []
    left = [oom()]

    def shape(on_card: list[str], **info: Any) -> None:
        held.append(on_card)
        if left:
            raise left.pop(0)

    world.taps["decode_shape"] = [shape]
    world.taps["decode_texture"] = [lambda on_card, **info: held.append(on_card)]
    world.run(requests("A"))
    assert held == [both, ["shape_dec"], ["tex_dec"]], held
    assert world.results["A"]["result"]["slow"] == []
    expect_alone(world, "A")

    # a prop too large for both decoders under the cap is decoded with one at a time from the start
    world = World()
    world.limit = 1900
    on_card: list[list[str]] = []
    world.taps["decode_texture"] = [lambda **info: on_card.append(info["on_card"])]
    world.run(requests("A"))
    assert on_card == [["tex_dec"]] and world.results["A"]["result"]["slow"] == []
    expect_alone(world, "A")


@test
def a_decode_that_cannot_fit_under_the_cap_fails_at_once() -> None:
    from pixal3d.staged import parts

    # Every shape's finish needs far more than any card. The decode of a big shape runs in boxes and no longer
    # grows with it (parts.chunk_for): what bounds a finish is the closing of the holes (parts.fill_mib).
    for name in "AE":
        alone(name)                             # made alone with the real figures, whatever ran before
    kept = parts.FILL_MIB_PER_KVOXEL
    parts.FILL_MIB_PER_KVOXEL = 1e6
    try:
        decodes: list[tuple] = []
        world = World()
        world.taps["decode_shape"] = [lambda **info: decodes.append(CURRENT.now)]
        world.run(requests("AE"))
        assert world.states() == {"A": "failed", "E": "failed"}, world.results
        for name in "AE":
            error = world.results[name]["error"]
            assert error["kind"] == "oom" and "needs about" in error["detail"], error
        assert decodes == [], "nothing was decoded: the attempt would have run out of memory after a minute"
        assert not world.events("prop_spilled")

        # with a spill allowed the cap is lifted and the decode runs once, with the cap lifted
        decodes.clear()
        world = World()
        world.taps["decode_shape"] = [lambda **info: decodes.append(CURRENT.now)]
        world.run(requests("AE"), spill=True)
        assert world.states() == {"A": "done", "E": "done"}, world.results
        assert decodes == [("finish", "A"), ("finish", "E")], decodes
        assert all(world.results[name]["result"]["slow"] == ["finish"] for name in "AE")
        expect_alone(world, "AE")
    finally:
        parts.FILL_MIB_PER_KVOXEL = kept


@test
def a_big_shape_is_decoded_in_boxes() -> None:
    """Both decoders are told the box size (decoder.chunk_voxels: parts.chunk_for) when the whole-table decode of
    the shape would take more than parts.CHUNK_ABOVE_MIB, and 0 (whole tables) otherwise."""
    from pixal3d.staged import parts

    chunks: list[tuple] = []
    world = World()
    world.taps["decode_shape"] = [lambda **info: chunks.append(("shape", info["chunk"]))]
    world.taps["decode_texture"] = [lambda **info: chunks.append(("tex", info["chunk"]))]
    world.run(requests("A"))
    assert world.states() == {"A": "done"} and chunks == [("shape", 0), ("tex", 0)], chunks      # a small shape
    kept = parts.CHUNK_ABOVE_MIB
    parts.CHUNK_ABOVE_MIB = -1                  # every shape is a big one
    try:
        chunks.clear()
        world = World()
        world.taps["decode_shape"] = [lambda **info: chunks.append(("shape", info["chunk"]))]
        world.taps["decode_texture"] = [lambda **info: chunks.append(("tex", info["chunk"]))]
        world.run(requests("A"))
        assert world.states() == {"A": "done"}, world.results
        assert chunks == [("shape", parts.CHUNK_VOXELS), ("tex", parts.CHUNK_VOXELS)], chunks
        assert parts.chunk_for(1) == parts.CHUNK_VOXELS and parts.finish_mib(1) >= parts.decode_mib(1, 1, parts.CHUNK_VOXELS)
    finally:
        parts.CHUNK_ABOVE_MIB = kept
    # the figures: a 1536 prop of the test set (12.3 M voxels) in boxes, a 1024 one (3 M) whole; the finish of
    # the former bounded by its decode in boxes (the closing of its holes takes less since cumesh 0.0.1.post2),
    # inside what a 7.2 GiB card leaves it
    assert parts.chunk_for(12_300_000) == parts.CHUNK_VOXELS and parts.chunk_for(3_000_000) == 0
    assert parts.finish_mib(12_300_000) == parts.decode_mib(12_300_000, 1, parts.CHUNK_VOXELS) < 6068 / 1.1
    assert parts.fill_mib(12_300_000) < parts.finish_mib(12_300_000)
    assert parts.decode_mib(12_300_000) > 6068 > parts.decode_mib(12_300_000, 2, parts.CHUNK_VOXELS)


class _Backend:
    """parts.decode_backend stood in for ("flex_gemm" | "spconv" | "spconv_torch"), and chunked.boxes_supported
    (True / False; with `covers`, False for a shape of more fine voxels than that, as products_floor.covers_voxels
    answers for a prop beyond what the install's decode_products measurement verified), for the length of a `with`:
    the decode rules of the cards below sm_80 on the fakes' CPU, whatever record this PC has."""

    def __init__(self, backend: str, boxes: bool, covers: int | None = None):
        self.backend, self.boxes, self.covers = backend, boxes, covers

    def __enter__(self) -> "_Backend":
        from pixal3d.models.sc_vaes import chunked
        from pixal3d.staged import parts

        self.kept = parts.decode_backend, getattr(chunked, "boxes_supported", None)
        parts.decode_backend = lambda: self.backend

        def boxes_supported(voxels: int | None = None) -> bool:
            if voxels is None or self.covers is None:
                return self.boxes
            return self.boxes and voxels <= self.covers

        chunked.boxes_supported = boxes_supported
        return self

    def __exit__(self, *exc: Any) -> None:
        from pixal3d.models.sc_vaes import chunked
        from pixal3d.staged import parts

        parts.decode_backend = self.kept[0]
        if self.kept[1] is None:
            del chunked.boxes_supported
        else:
            chunked.boxes_supported = self.kept[1]


@test
def the_decode_is_counted_with_the_figures_of_the_backend_the_decoders_run_on() -> None:
    """parts.decode_backend / decode_figures (round O): flex_gemm's figures are the module's constants, as before;
    spconv's implicit GEMM (the turing profile) and spconv's products on torch.mm (pascal: no prebuilt kernels for
    sm_61) have their own, which hold the decodes measured on them (the RTX 3060 under the forced profiles, the
    finish replayed from checkpoints: code/logs/profiling/pixal3d-lowmem/round-o/memory); a backend without boxes
    is counted on whole tables (chunk_for 0), and its unlean reserve with its own factor."""
    import torch
    from pixal3d.modules import precision
    from pixal3d.modules.sparse import config as sparse_config
    from pixal3d.staged import parts
    if not hasattr(parts, "decode_figures"):
        print("  (this checkout's parts.py has no backend figures (before round O): nothing to check)")
        return

    # the fakes' profile is ampere's: flex_gemm, the figures as they always were
    assert parts.decode_backend() == "flex_gemm"
    assert parts.decode_figures() == {
        "whole": parts.DECODE_MIB_PER_KVOXEL, "chunked": parts.CHUNKED_MIB_PER_KVOXEL,
        "chunk": parts.CHUNK_MIB_PER_KVOXEL, "unlean_whole": parts.UNLEAN_WHOLE, "unlean_boxes": parts.UNLEAN_BOXES,
        "chunk_above": parts.CHUNK_ABOVE_MIB}
    assert parts.decode_mib(12_300_000, 2, parts.CHUNK_VOXELS) == \
        2 * parts.DECODER_MIB + parts.CHUNKED_MIB_PER_KVOXEL * 12_300_000 / 1000 + parts.CHUNK_MIB_PER_KVOXEL * 2_000
    assert parts.decode_mib(3_000_000, 1) == parts.DECODER_MIB + parts.DECODE_MIB_PER_KVOXEL * 3_000

    # the backend as a card below sm_80 has it: the forced profile, the convolutions on spconv (as
    # precision.conv_backend makes them when the sparse config is read), the products path as conv_spconv finds it
    from pixal3d.modules.sparse.conv import conv_spconv

    names = ("PIXAL3D_PRECISION", "PIXAL3D_CARD_SM")
    saved = {name: os.environ.get(name) for name in names}
    conv, products, indice_conv = sparse_config.CONV, conv_spconv._torch_products, conv_spconv.spconv_ops.indice_conv
    try:
        sparse_config.CONV = "spconv"
        for profile, sm, backend in (("turing", "75", "spconv"), ("pascal", "61", "spconv_torch")):
            os.environ["PIXAL3D_PRECISION"], os.environ["PIXAL3D_CARD_SM"] = profile, sm
            precision.reset()
            conv_spconv._torch_products = None
            assert parts.decode_backend() == backend, (profile, parts.decode_backend())
            assert parts.decode_figures()["whole"] == parts.SPCONV_DECODE_MIB_PER_KVOXEL[backend]
    finally:
        sparse_config.CONV, conv_spconv._torch_products = conv, products
        conv_spconv.spconv_ops.indice_conv = indice_conv
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        precision.reset()
    assert parts.decode_backend() == "flex_gemm" and not torch.cuda.is_initialized()

    # the measured peaks (PyTorch's allocated, MiB) under the figures, whole tables: (voxels, decoders on the card,
    # MiB beside them, the peak measured)
    measured = {
        "spconv": [(3_487_388, 2, 0, 3_957),            # the boy at 1536, both decoders (boy-tur; M1 3,967)
                   (3_487_388, 1, 0, 3_044),            # the same, one decoder (boy-tur one-lean)
                   (3_487_388, 1, 1_566, 4_606),        # one decoder beside the texture network (boy-tur)
                   (10_565_409, 2, 0, 8_241),           # the ork at 1536, both (M1 tur-ork; whole-tur 8,230)
                   (10_565_409, 1, 0, 7_318),           # the same, one decoder (whole-tur)
                   (10_565_409, 1, 1_557, 8_877)],      # one beside the texture network (whole-tur)
        "spconv_torch": [(3_549_866, 2, 0, 4_673),      # the boy at 1536, both decoders (boy-pas; boy-pas2 4,643,
                         #                                M1 4,654)
                         (3_549_866, 1, 0, 3_731),      # one decoder (boy-pas2 one-lean)
                         (3_549_866, 1, 1_926, 5_680),   # one beside the texture network (boy-pas; boy-pas2 5,651)
                         (10_564_282, 1, 0, 9_191)],    # the ork at 1536, one decoder, lean (whole-pas)
    }
    boxed = {"spconv": [(3_487_388, 2, 0, 3_163), (10_565_409, 2, 0, 4_345), (10_565_409, 1, 1_557, 4_992),
                        (12_240_099, 2, 0, 4_962)],
             "spconv_torch": [(3_549_866, 2, 0, 3_378), (10_564_282, 2, 0, 4_359), (10_564_282, 1, 1_918, 5_362),
                              (12_238_837, 2, 0, 4_964)]}            # (boxes-pas-a / -b)
    for backend, cases in measured.items():
        with _Backend(backend, boxes=False):
            for voxels, decoders, beside, peak in cases:
                assert parts.chunk_for(voxels) == 0, (backend, voxels)          # no boxes: whole tables
                assert beside + parts.decode_mib(voxels, decoders) >= peak, (backend, voxels, decoders)
                assert beside + parts.finish_mib(voxels, decoders) >= peak, (backend, voxels, decoders)
                # the margin is small: the figure is the measurement's, not flex_gemm's 0.52 (R3: 3,656 MiB for
                # the boy's pascal decode against 4,654 measured)
                assert beside + parts.decode_mib(voxels, decoders) <= 1.07 * peak, (backend, voxels, decoders)
            # a decode without the trims reserves the backend's factor times what it allocates: the boy with one
            # decoder reserved 5,886 MiB with no cap on spconv (boy-tur one-unlean), 7,530 on the products path
            # (boy-pas2), a room lean_needed must refuse
            unlean = {"spconv": 5_886, "spconv_torch": 7_530}[backend]
            assert parts.decode_figures()["unlean_whole"] * parts.decode_mib(3_487_388, 1) >= unlean
            assert parts.lean_needed(3_487_388, 1, 0, unlean) and not parts.lean_needed(3_487_388, 1, 0, 9_500)
        with _Backend(backend, boxes=True):
            # boxes only where the whole-table decode does not fit the finish decode's cap (parts'
            # SPCONV_BOXES_WHERE_NEEDED: no real old card has run the boxes); whole where there is no cap to judge
            assert parts.chunk_for(12_300_000) == 0 and parts.chunk_for(12_300_000, 5_761) == parts.CHUNK_VOXELS
            assert parts.chunk_for(1_000_000, 5_761) == 0
            assert parts.finish_mib(12_300_000, 1, 5_761) == parts.decode_mib(12_300_000, 1, parts.CHUNK_VOXELS)
            # what was decoded whole before round O stays whole where it fits, under the pessimistic 8 GB decode
            # cap (5,761) and the RTX 2070's (6,813): a 1024 boy (1.65 M voxels), the boy at 1536 (3.5 M), the
            # 2070's ork at 1024 (5.14 M); the ork and the tree at 1536 there go into boxes
            for voxels in (1_652_000, 3_487_388, 3_549_866, 5_136_000):
                assert parts.chunk_for(voxels, 5_761) == parts.chunk_for(voxels, 6_813) == 0, (backend, voxels)
            for voxels in (10_565_409, 12_238_927):
                assert parts.chunk_for(voxels, 5_761) == parts.CHUNK_VOXELS, (backend, voxels)
                assert parts.decoders_fit(voxels, 5_761) == parts.decoders_fit(voxels, 5_761, parts.CHUNK_VOXELS)
            # the box decodes measured (chunked.py's spconv boxes, 2 M voxels, lean: runs/boxes-tur, boxes-pas-a)
            for voxels, decoders, beside, peak in boxed[backend]:
                assert beside + parts.decode_mib(voxels, decoders, parts.CHUNK_VOXELS) >= peak, (backend, voxels)
            # a record with an output-layer boundary band: the shape decoder holds its boxes' tail rows
            # (parts._holds_tail_rows, products_floor.has_boundary); measured with both decoders on the products path
            # (boxes-REPORT 11.5): the boy at 2 M boxes 1,082 MiB above them, the ork 3,077
            holds = getattr(parts, "_holds_tail_rows", None)
            if holds is not None:
                assert not parts.decode_figures().get("hold"), "no record in the fakes: no hold"
                parts._holds_tail_rows = lambda: True
                try:
                    for voxels, peak in ((3_549_866, 1_810 + 1_082), (10_564_282, 1_810 + 3_077)):
                        assert parts.decode_mib(voxels, 2, parts.CHUNK_VOXELS) >= peak, (backend, voxels)
                    assert parts.decode_mib(10_564_282, 2, parts.CHUNK_VOXELS) > \
                        2 * parts.DECODER_MIB + 0.22 * 10_564.282 + parts.CHUNK_VOXELS * 0.42 / 1000
                    assert parts.decode_mib(3_000_000, 1) == parts.DECODER_MIB + parts.decode_figures()["whole"] * 3_000_000 / 1000
                finally:
                    parts._holds_tail_rows = holds
            # boxing every decode above the backend's threshold, once a real card has passed the gate: one switch
            kept_switch = parts.SPCONV_BOXES_WHERE_NEEDED
            parts.SPCONV_BOXES_WHERE_NEEDED = False
            try:
                assert parts.chunk_for(5_136_000) == parts.CHUNK_VOXELS and parts.chunk_for(1_652_000) == 0
            finally:
                parts.SPCONV_BOXES_WHERE_NEEDED = kept_switch
    # on flex_gemm without its masked split-K (no boxes) the whole tables are counted too; with it, as before,
    # whatever the cap (flex_gemm boxes above CHUNK_ABOVE_MIB: the Ampere cards' bits)
    with _Backend("flex_gemm", boxes=False):
        assert parts.chunk_for(12_300_000) == 0
    assert parts.chunk_for(12_300_000) == parts.chunk_for(12_300_000, 50_000) == parts.CHUNK_VOXELS
    assert parts.chunk_for(3_000_000) == parts.chunk_for(3_000_000, 1_000) == 0


@test
def on_spconv_a_shape_is_decoded_in_boxes_only_where_its_whole_tables_do_not_fit() -> None:
    """The decode itself (decoder.chunk_voxels at decode_shape / decode_texture) takes chunk_for's decision under its
    own cap: on spconv a shape whose whole-table decode fits is decoded whole, as every prop was before round O
    (parts.SPCONV_BOXES_WHERE_NEEDED: no real old card has run the boxes), and one that does not fit in boxes; with
    the switch off, the threshold decides, as on flex_gemm. Both decoders together where their decode fits."""
    from pixal3d.staged import parts
    if not hasattr(parts, "SPCONV_BOXES_WHERE_NEEDED"):
        print("  (this checkout's parts.py boxes spconv by a threshold (before round O's review): nothing to check)")
        return
    if REAL_LOADER:
        print("  (the worktree's own loader on the CPU caps nothing: no cap to decide under)")
        return

    def chunks(**figures: Any) -> list:
        seen: list = []
        world = World()
        world.taps["decode_shape"] = [lambda on_card, **info: seen.append(("shape", info["chunk"], len(on_card)))]
        world.taps["decode_texture"] = [lambda on_card, **info: seen.append(("tex", info["chunk"], len(on_card)))]
        kept = dict(parts.SPCONV_DECODE_MIB_PER_KVOXEL)
        parts.SPCONV_DECODE_MIB_PER_KVOXEL.update(figures)
        try:
            world.run([request("A")])
        finally:
            parts.SPCONV_DECODE_MIB_PER_KVOXEL.clear()
            parts.SPCONV_DECODE_MIB_PER_KVOXEL.update(kept)
        assert world.states() == {"A": "done"}, world.results
        expect_alone(world, "A")
        return seen

    with _Backend("spconv", boxes=True):
        assert chunks() == [("shape", 0, 2), ("tex", 0, 2)], chunks()          # fits whole: whole, both decoders
        # a whole-table decode that would not fit the decode cap (the fake world's 5,768 MiB): boxes
        huge = chunks(spconv=1e6)
        assert [chunk for _, chunk, _ in huge] == [parts.CHUNK_VOXELS] * 2, huge
        kept_switch = parts.SPCONV_BOXES_WHERE_NEEDED
        parts.SPCONV_BOXES_WHERE_NEEDED = False                               # boxing by the threshold later
        try:
            assert [chunk for _, chunk, _ in chunks(spconv=1e3)] == [parts.CHUNK_VOXELS] * 2
            assert [chunk for _, chunk, _ in chunks()] == [0, 0]
        finally:
            parts.SPCONV_BOXES_WHERE_NEEDED = kept_switch


@test
def on_an_8_gb_card_without_boxes_a_1536_prop_is_made_at_1024_and_with_them_at_1536() -> None:
    """parts.resolution_fits on an emulated 8 GB card (the pessimistic Windows caps: 6,061 MiB for the parts, 5,761
    for the finish's decode, R3): where the decoders cannot cut their levels into boxes (spconv before round O), a
    1536 prop is made at 1024 whatever its size (WHOLE_LEVEL_1536_MIB, the safety net of round L) and the memory
    note says so; once boxes are supported (chunked.boxes_supported) it is made at 1536 - on spconv and on
    flex_gemm alike - and the finish test counts the decode itself: the ork at 1536 (10.6 M voxels) fits in boxes,
    not on whole tables, and the boy (3.5 M) fits even whole."""
    from pixal3d.staged import parts
    if not hasattr(parts, "decode_figures"):
        print("  (this checkout's parts.py has no backend figures (before round O): nothing to check)")
        return

    kept = parts._card_total_mib
    parts._card_total_mib = lambda ctx: 8_192
    try:
        # (the worktree's own loader on the CPU caps nothing, so the card is not judged there: the rule alone below)
        for backend in (() if REAL_LOADER else ("spconv", "spconv_torch", "flex_gemm")):
            for boxes, made in ((False, 1024), (True, 1536)):
                with _Backend(backend, boxes):
                    world = World()
                    world.limit = 6_061
                    world.run([request("B")])
                    assert world.states() == {"B": "done"}, (backend, boxes, world.results)
                    res = int(world.puts["B"]["coords_hr"]["res"][0])
                    assert res == made, (backend, boxes, res)
                    note = world.finished["B"][1].get("memory_cap")
                    if made == 1024:
                        assert note is not None and note["by"] == "card" and note["without"] == 1536, note
                    else:
                        assert note is None, note
    finally:
        parts._card_total_mib = kept

    # the rule itself with the 8 GB card's figures (the finish test alone: a 12 GB card, which the safety net
    # leaves to it): the ork's whole-table decode would not fit the pessimistic decode cap, the boy's would
    class Loader:
        safety_mib = 300

        def stats(self) -> dict:
            return {"cap_mib": 6_061}

    ctx = types.SimpleNamespace(loader=Loader(), part="upsample", hooks=None, device="cpu")
    weights, ram = parts.flow_weights_mib, parts.ram_available_mib
    parts.flow_weights_mib = lambda ctx, kind: 1_572                     # the turing int8 hr flow (R3)
    parts.ram_available_mib = lambda: 60_000
    parts._card_total_mib = lambda ctx: 12_288
    try:
        for backend in ("spconv", "spconv_torch"):
            with _Backend(backend, boxes=False):
                assert parts.resolution_fits(ctx, 0, 30_500, 1536, voxels=10_565_409) == "card", backend
                assert parts.resolution_fits(ctx, 0, 12_110, 1536, voxels=3_487_388) is None, backend
            with _Backend(backend, boxes=True):
                assert parts.resolution_fits(ctx, 0, 30_500, 1536, voxels=10_565_409) is None, backend
        parts._card_total_mib = lambda ctx: 8_192
        for backend in ("spconv", "spconv_torch"):
            # a shape beyond what the card's decode_products measurement covers (chunked.boxes_supported(voxels)):
            # planned whole, so on an 8 GB card made at 1024 like a card without boxes, and not boxed in finish
            with _Backend(backend, boxes=True, covers=11_000_000):
                assert parts.resolution_fits(ctx, 0, 43_500, 1536, voxels=12_238_927) == "card", backend
                assert parts.resolution_fits(ctx, 0, 30_500, 1536, voxels=10_565_409) is None, backend
                assert parts.chunk_for(12_238_927, 5_761) == 0 and parts.chunk_for(10_565_409, 5_761) > 0
        for backend in ("spconv", "spconv_torch"):
            with _Backend(backend, boxes=False):     # the safety net: the boy as well
                assert parts.resolution_fits(ctx, 0, 12_110, 1536, voxels=3_487_388) == "card", backend
            with _Backend(backend, boxes=True):
                assert parts.resolution_fits(ctx, 0, 12_110, 1536, voxels=3_487_388) is None, backend
                assert parts.resolution_fits(ctx, 0, 12_110, 1024, voxels=1_652_000) is None, backend
    finally:
        parts.flow_weights_mib, parts.ram_available_mib, parts._card_total_mib = weights, ram, kept


@test
def a_resumed_prop_counts_the_voxels_its_checkpoint_noted() -> None:
    """parts.tex notes the voxels upsample foresaw with the checkpoint (parts.CHECKPOINT_VOXELS, in its notes: the
    tensors stay coords, res, shape_slat, tex_slat), and runner._restore puts them back into coords_hr, so a resumed
    prop's finish counts them (parts.decode_voxels) instead of VOXELS_PER_TOKEN a token (round O: on spconv's whole
    tables a resumed 1536 tree, 17.4 M voxels counted for 12.2 M, was refused at once on a 12 GB card). The notes the
    hook is handed do not carry it; a checkpoint without it (before round O) resumes as before."""
    from pixal3d.staged import parts
    if not hasattr(parts, "CHECKPOINT_VOXELS"):
        print("  (this checkout's parts.py notes no voxels with a checkpoint (before round O): nothing to check)")
        return

    class Died(SystemExit):
        pass

    def die(prop: str, mesh: Any) -> None:
        raise Died()

    for old in (False, True):
        world = World()
        world.taps["finish"] = [die]
        try:
            world.run([request("A", request_hash="hash-A")], paired=False)
        except Died:
            pass
        foreseen = int(world.puts["A"]["coords_hr"]["voxels"][0])
        assert foreseen > 0 and world.checkpoints() == ["A"]
        if not REAL_STORE:
            notes = world.disk["A"]["notes"]
            assert notes[parts.CHECKPOINT_VOXELS] == foreseen, notes
            assert sorted(world.disk["A"]["tensors"]) == ["coords", "res", "shape_slat", "tex_slat"]
            if old:
                del notes[parts.CHECKPOINT_VOXELS]
        elif old:
            continue
        world.taps.clear()
        world.finished.clear()
        world.run([request("A", request_hash="hash-A")], resume=True, paired=False)
        assert world.states() == {"A": "done"} and world.results["A"]["result"]["resumed"], world.results
        stored = world.puts["A"]["coords_hr"]
        tokens = stored["coords"].shape[0]
        assert parts.decode_voxels(stored) == (parts.VOXELS_PER_TOKEN * tokens if old else foreseen), stored
        assert parts.CHECKPOINT_VOXELS not in world.finished["A"][1], world.finished["A"][1]
        want = {name: value for name, value in alone("A").items() if name.startswith("mesh/")}
        got = {f"mesh/{name}": getattr(world.finished["A"][0], name) for name in ("vertices", "faces", "coords",
                                                                                 "attrs")}
        assert not differing(got, want), differing(got, want)       # the same mesh either way


@test
def the_samplers_count_an_fp32_torsos_tokens_at_its_own_rate() -> None:
    """parts.sampler_mib (round O): a bf16 or fp16 torso as before, 50 MiB a thousand tokens with the slack; an fp32
    torso (the pascal profile) SAMPLER_FP32_MIB_PER_KTOKEN, above the hr samplings measured on it beyond the weights
    (the boy at 1536, 12.1k tokens: 5,114 - 2,129 MiB of proj tables = 2,985 allocated, M1's pas-a; the tree at
    ~30k tokens under the pessimistic 8 GB cap: 4,482, runs/tree-pas-hr), and above bf16's figure, which let them
    through ~0.6 GB short. Where the old cards' int8 MLP runs in row slabs (parts._mlp_slab_rows: 8,192 rows in fp32)
    the tokens beyond the slab count SAMPLER_FP32_SLABBED_MIB_PER_KTOKEN: the boy at 1536 then allocated 2,863 MiB
    beyond the weights and tables (finish/runs/pas1536-slab), the tree at 43.5k tokens is foreseen to fit a GTX 10xx's
    6,061 MiB cap; without the slabs it is not."""
    import torch
    from pixal3d.modules import precision
    from pixal3d.staged import parts
    if not hasattr(parts, "SAMPLER_FP32_MIB_PER_KTOKEN"):
        print("  (this checkout's parts.py has no fp32 sampler figure (before round O): nothing to check)")
        return

    torso = precision.torso
    try:
        precision.torso = lambda: torch.bfloat16
        bf16 = parts.sampler_mib(30_500, 1557)
        assert bf16 == parts.SAMPLER_SLACK * (1557 + parts.SAMPLER_MIB_PER_KTOKEN * 30.5)
        precision.torso = lambda: torch.float16
        assert parts.sampler_mib(30_500, 1557) == bf16
        assert parts.sampler_mib(30_500, 1572) >= 1572 + 58.2 * 30.5      # turing's measured 57.4-58.2 a thousand
        precision.torso = lambda: torch.float32
        slab_rows = getattr(parts, "_mlp_slab_rows", None)
        if slab_rows is not None:
            parts._mlp_slab_rows = lambda tokens: tokens                    # no slabs: the whole figure
        for tokens, peak in ((12_110, 2_985), (30_200, 4_482)):
            assert parts.sampler_mib(tokens, 1929) >= 1.2 * peak, (tokens, parts.sampler_mib(tokens, 1929))
        assert parts.sampler_mib(43_500, 1929) > 6_061 > parts.sampler_mib(30_200, 1929)
        assert parts.sampler_mib(43_500, 1929) > parts.SAMPLER_SLACK * (1929 + parts.SAMPLER_MIB_PER_KTOKEN * 43.5)
        if slab_rows is not None:
            parts._mlp_slab_rows = lambda tokens: min(tokens, 8_192)        # the fp32 slabs of int8mm
            assert parts.sampler_mib(5_100, 1929) == parts.SAMPLER_SLACK * (1929 + 90 * 5_100 / 1000)   # below: whole
            assert parts.sampler_mib(12_110, 1929) >= 1.2 * 2_863           # the boy at 1536, slabbed
            assert parts.sampler_mib(12_110, 1929) >= 1929 + 76.5 * 12.11 * 1.2
            assert parts.sampler_mib(43_500, 1993) < 6_061                  # the tree, the weights before their load
            tree = parts.SAMPLER_SLACK * (1993 + (90 * 8_192 + 52 * (43_500 - 8_192)) / 1000)
            assert parts.sampler_mib(43_500, 1993) == tree, parts.sampler_mib(43_500, 1993)
            assert parts.sampler_mib(43_500, 1929) >= 1.2 * (1936 + 44 * 43.5 + 578 - 1929) + 1929
    finally:
        precision.torso = torso
        if slab_rows is not None:
            parts._mlp_slab_rows = slab_rows


@test
def the_pair_keeps_nothing_where_the_whole_table_decode_would_not_fit_beside_the_texture_network() -> None:
    """parts.pair_keep with spconv's figures: the RTX 2070's ork at 1024 (5.14 M voxels; the caps 6,813 for the
    decode, 7,113 for the parts; its int8 texture network 1,572 MiB) kept the texture network beside both decoders
    under flex_gemm's box figures and its first decode ran out of memory (round N, 2070-u6: 1,572 + 1,810 + 0.61 a
    thousand voxels = 6,515 MiB of a 6,813 cap). Counted on spconv's whole tables, nothing stays; where the decode
    runs in boxes the texture network may stay again: on spconv that is a shape whose whole-table decode does not
    fit (8.6 M voxels under the 2070's caps; the 2070's ork at 1024 fits whole and is decoded whole, as before round
    O). On the pessimistic 8 GB caps a boy-sized prop keeps nothing either (5,579 MiB counted against 5,237: the
    replay fitted it with 127 MiB to spare, lean). tail_room counts the same decisions: exactly, with the RAM put
    at 60,000 MiB so that this PC's own does not clamp both sides."""
    import torch
    from pixal3d.staged import parts
    from pixal3d.staged.runner import PropState
    if not hasattr(parts, "decode_figures"):
        print("  (this checkout's parts.py has no backend figures (before round O): nothing to check)")
        return

    class Loader:
        safety_mib = 300

        def __init__(self, cap: int, headroom: int = 12_000):
            self.cap, self.room = cap, headroom

        def stats(self) -> dict:
            return {"cap_mib": self.cap}

        def headroom_mib(self) -> int:
            return self.room

        def get(self, name: str) -> Any:
            return object() if name == "tex_flow.sv" else None

        def size_mib(self, name: str) -> int | None:
            return {"tex_flow.sv": 1_572, "shape_dec": 905, "tex_dec": 905}.get(name)

    class Store:
        def __init__(self, voxels: int, tokens: int):
            self.voxels, self.tokens = voxels, tokens

        def get_small(self, prop_id: str, key: str) -> dict:
            return {"coords": torch.zeros(self.tokens, 4, dtype=torch.int32), "res": torch.tensor([1024]),
                    "voxels": torch.tensor([self.voxels])}

    def rule(cap: int, voxels: int, tokens: int = 12_500) -> frozenset:
        ctx = types.SimpleNamespace(loader=Loader(cap), store=Store(voxels, tokens), part="tex", hooks=None)
        return parts.pair_keep(ctx, PropState(req=request("A")), PropState(req=request("E")), [])

    def tail(voxels: int) -> int:
        """The room a finish step may keep with one prop of `voxels` voxels still to decode, on a card that leaves
        the 2070's finish decode cap (headroom 7,613 less the finish's 800 of reserves = 6,813)."""
        ctx = types.SimpleNamespace(loader=Loader(linux_2070, headroom=7_613), store=Store(voxels, 12_500))
        return parts.tail_room(ctx, [PropState(req=request("A"))], 0)

    linux_2070, pessimistic = 7_113, 6_061          # the parts' caps; the decode's are 300 MiB less (card_limits)
    ram = parts.ram_available_mib
    parts.ram_available_mib = lambda: 60_000
    try:
        with _Backend("flex_gemm", boxes=True):     # the figures the 2070 ran with in round N
            assert "tex_flow.sv" in rule(linux_2070, 5_136_000)
        # tail_room, exactly: 7,613 - 800 - 1.1 x finish_mib(8.6 M voxels, both decoders) - 512 (HOLD_SLACK_MIB):
        # in boxes 1,810 + 0.22 x 8,600 + a 2 M box's 660 (spconv) / 840 (the products path); on whole tables nothing
        boxed_room = {"spconv": 1_502, "spconv_torch": 1_304}
        for backend in ("spconv", "spconv_torch"):
            with _Backend(backend, boxes=False):
                assert rule(linux_2070, 5_136_000) == frozenset(), backend
                assert rule(pessimistic, 3_487_388) == frozenset(), backend
                assert rule(linux_2070, 8_600_000) == frozenset(), backend
                assert tail(8_600_000) == 0, (backend, tail(8_600_000))
            with _Backend(backend, boxes=True):
                # the 2070's ork at 1024 fits whole: decoded whole as before, both decoders on the card, nothing
                # kept beside it
                assert parts.chunk_for(5_136_000, 6_813) == 0 and parts.decoders_fit(5_136_000, 6_813), backend
                assert rule(linux_2070, 5_136_000) == frozenset(), backend
                # 8.6 M voxels do not fit whole: in boxes, and the texture network stays beside them
                assert parts.chunk_for(8_600_000, 6_813) == parts.CHUNK_VOXELS, backend
                assert rule(linux_2070, 8_600_000) == frozenset({"tex_flow.sv"}), (backend, rule(linux_2070,
                                                                                                8_600_000))
                assert tail(8_600_000) == boxed_room[backend] > 0, (backend, tail(8_600_000))
    finally:
        parts.ram_available_mib = ram


@test
def the_memory_of_the_card_and_of_the_ram_lower_the_resolution_of_a_1536_prop() -> None:
    from pixal3d.staged import parts

    full = World()
    full.run([request("B")])
    assert full.states() == {"B": "done"}
    stored = full.puts["B"]["coords_hr"]
    assert int(stored["res"][0]) == 1536
    voxels, tokens_1536 = int(stored["voxels"][0]), stored["coords"].shape[0]
    assert voxels > 0 and "memory_cap" not in full.finished["B"][1]

    # the card: the decode phase of `finish` gets 300 MiB less than the cap in force (the real loader on the CPU
    # caps nothing, so the limits are stood in for); make the closing of this shape's holes - what bounds a big
    # shape's finish, its decode running in boxes - need a little more than that at 1536, so that 1408 (0.84 of
    # the voxels) is the first resolution that fits
    limit = 6068
    finish_limit = limit - 300
    kept, kept_limits = parts.FILL_MIB_PER_KVOXEL, parts.card_limits
    parts.FILL_MIB_PER_KVOXEL = finish_limit / 1.1 * 1000 / voxels * 1.01
    parts.card_limits = lambda ctx: (finish_limit, limit)
    try:
        world = World()
        world.run([request("B")])
        assert world.states() == {"B": "done"}, world.results
        stored = world.puts["B"]["coords_hr"]
        assert int(stored["res"][0]) == 1408, stored["res"]
        note = world.finished["B"][1]["memory_cap"]
        assert note == {"by": "card", "resolution": 1408, "without": 1536, "tokens": stored["coords"].shape[0]}, note
        assert world.results["B"]["result"]["resolution"] == 1408
        # the same prop made at that resolution by the token cap alone is the same prop
        same = World()
        same.run([request("B", max_num_tokens=tokens_1536)])
        assert int(same.puts["B"]["coords_hr"]["res"][0]) == 1408
        assert "memory_cap" not in same.finished["B"][1]
        assert not differing(world.signature("B"), same.signature("B"))
    finally:
        parts.FILL_MIB_PER_KVOXEL, parts.card_limits = kept, kept_limits
    if not REAL_LOADER:         # the fake loader answers the cap it was given: the limits come from it
        seen: list = []
        world = World()
        world.taps["upsample"] = [lambda **info: seen.append(parts.card_limits(CURRENT.engine._ctx))]
        world.run([request("A")])
        assert seen == [(world.limit - 300, world.limit)], seen

    # the RAM: the finish hook would leave less than the floor at 1536 (the fake meshes are tiny: the hook's need
    # per voxel is made large so that the resolutions are told apart), not at 1408
    kept_ram, kept_slope = parts.ram_available_mib, parts.HOOK_RAM_MIB_PER_MVOXEL
    parts.HOOK_RAM_MIB_PER_MVOXEL = 1e9
    parts.ram_available_mib = lambda: int(parts.RAM_FLOOR_MIB + parts.hook_ram_mib(int(voxels * 0.9)))
    try:
        world = World()
        world.run([request("B")])
        assert world.states() == {"B": "done"}, world.results
        note = world.finished["B"][1]["memory_cap"]
        assert note["by"] == "ram" and note["resolution"] == 1408 and note["without"] == 1536, note
        # a 1024 prop is never lowered, whatever the memory says: it is made and the note is not written
        world = World()
        world.run([request("A")])
        assert world.states() == {"A": "done"} and "memory_cap" not in world.finished["A"][1]
        assert int(world.puts["A"]["coords_hr"]["res"][0]) == 1024
    finally:
        parts.ram_available_mib, parts.HOOK_RAM_MIB_PER_MVOXEL = kept_ram, kept_slope


class StepHooks(FakeHooks):
    """FakeHooks whose finish hands the rest back as steps - two on the GPU thread, two on the worker - each
    recording where and when it ran; the result and the finished mesh are FakeHooks.finish's."""

    sleep = 0.0                 # seconds a CPU step takes
    cancel_in_cpu1 = False      # the first CPU step asks for a cancel

    def __init__(self, world: "World"):
        super().__init__(world)
        self.trace: list[tuple] = []        # (prop id, step, thread name, started, ended)
        self.fail_in: dict[tuple, list] = {}    # (prop id, step) -> what its next runs raise
        self.alive = 0
        self.most_alive = 0

    def finish(self, req: Any, mesh: Any, notes: dict) -> Any:
        from pixal3d.staged.types import FinishSteps

        self.world.tap("finish", prop=req.id, mesh=mesh)
        self.alive += 1
        self.most_alive = max(self.most_alive, self.alive)
        state = {"mesh": mesh, "notes": copy.deepcopy(notes)}

        def step(name: str) -> Callable[[dict], None]:
            def run(state: dict) -> None:
                started = time.perf_counter()
                pending = self.fail_in.get((req.id, name))
                if pending:
                    raise pending.pop(0)
                if name.startswith("cpu"):
                    if name == "cpu1" and self.cancel_in_cpu1:
                        self.world.cancel = True
                    time.sleep(self.sleep)
                state[name] = True
                self.trace.append((req.id, name, threading.current_thread().name, started, time.perf_counter()))
            return run

        def result(state: dict) -> dict:
            assert all(state.get(name) for name in ("gpu1", "cpu1", "gpu2", "cpu2")), sorted(state)
            self.alive -= 1
            self.world.finished[req.id] = (state["mesh"], state["notes"])
            return {"mode": "multi" if req.multi else "single", "views": req.payload["views"], "warning": None,
                    "ss_retries": state["notes"].get("ss_retries", 0)}

        return FinishSteps(steps=[("gpu", step("gpu1")), ("cpu", step("cpu1")), ("gpu", step("gpu2")),
                                  ("cpu", step("cpu2"))], state=state, result=result)


class RoomHooks(StepHooks):
    """StepHooks whose first GPU step keeps `hold` MiB on the card when the room the runner wrote allows it
    (state["card_mib"]) and whose second lets go; the room written before every step and every release are kept."""
    hold = 0

    def __init__(self, world: "World"):
        super().__init__(world)
        self.rooms: list[tuple] = []        # (prop id, step, state["room_mib"] as the step found it)
        self.released: list[str] = []

    def finish(self, req: Any, mesh: Any, notes: dict) -> Any:
        steps = super().finish(req, mesh, notes)
        names = ("gpu1", "cpu1", "gpu2", "cpu2")
        wrapped = []
        for name, (kind, fn) in zip(names, steps.steps):
            def around(state: dict, fn: Any = fn, name: str = name) -> None:
                self.rooms.append((req.id, name, state.get("room_mib")))
                fn(state)
                if name == "gpu1" and self.hold and int(state.get("room_mib") or 0) >= self.hold:
                    state["card_mib"] = self.hold
                if name == "gpu2":
                    state.pop("card_mib", None)
            wrapped.append((kind, around))
        steps.steps = wrapped

        def release(state: dict) -> None:
            self.released.append(req.id)
            state.pop("card_mib", None)

        steps.release = release
        return steps


class CpuFirstHooks(FakeHooks):
    """FakeHooks whose finish hands back steps that begin on the worker, as the hub's do (the cuBVH build), then
    one on the card, then the worker again: cpu1, gpu1, cpu2. `wait_for_previous` makes finish (called on the
    GPU thread at the decode) wait until the prop before has run its first step, so that the boundary that
    follows finds that prop's GPU step ready to run."""
    wait_for_previous = False
    first_sleep = 0.0           # seconds the first step takes
    last_waits_for = 0          # the last step waits (LAST_WAIT_S at most) until this many props had their finish
    gpu_sleep = 0.0             # seconds the GPU step takes
    LAST_WAIT_S = 10.0

    def __init__(self, world: "World"):
        super().__init__(world)
        self.trace: list[tuple] = []        # (prop id, step, thread name, started, ended)
        self.order: list[str] = []          # props as their finish was called
        self.first_done: dict[str, threading.Event] = {}
        self.alive = 0
        self.most_alive = 0

    def finish(self, req: Any, mesh: Any, notes: dict) -> Any:
        from pixal3d.staged.types import FinishSteps

        self.world.tap("finish", prop=req.id, mesh=mesh)
        if self.wait_for_previous and self.order:
            assert self.first_done[self.order[-1]].wait(5.0), self.order
        self.order.append(req.id)
        self.first_done[req.id] = threading.Event()
        self.alive += 1
        self.most_alive = max(self.most_alive, self.alive)
        state = {"mesh": mesh, "notes": copy.deepcopy(notes)}

        def step(name: str) -> Callable[[dict], None]:
            def run(state: dict) -> None:
                started = time.perf_counter()
                if name == "cpu1":
                    time.sleep(self.first_sleep)
                if name == "cpu2":
                    deadline = time.monotonic() + self.LAST_WAIT_S
                    while len(self.order) < self.last_waits_for and time.monotonic() < deadline:
                        time.sleep(0.005)
                if name == "gpu1":
                    time.sleep(self.gpu_sleep)
                state[name] = True
                self.trace.append((req.id, name, threading.current_thread().name, started, time.perf_counter()))
                if name == "cpu1":
                    self.first_done[req.id].set()
            return run

        def result(state: dict) -> dict:
            assert all(state.get(name) for name in ("cpu1", "gpu1", "cpu2")), sorted(state)
            self.alive -= 1
            self.world.finished[req.id] = (state["mesh"], state["notes"])
            return {"mode": "multi" if req.multi else "single", "views": req.payload["views"], "warning": None,
                    "ss_retries": state["notes"].get("ss_retries", 0)}

        return FinishSteps(steps=[("cpu", step("cpu1")), ("gpu", step("gpu1")), ("cpu", step("cpu2"))],
                           state=state, result=result)


@test
def a_fresh_tails_first_cpu_step_waits_for_the_boundarys_gpu_steps_while_props_remain() -> None:
    """A tail that has run no step yet and begins on the worker is handed over after the GPU steps of the other
    tails at that boundary, not before them (what its first step makes - the hub's cuBVH tree - is needed one
    texture later, while those GPU steps let go of an older prop's: two trees in RAM, not three). The last
    prop's tail, with nothing to wait for, is handed over at once. The first step sleeps so that it is still
    running at the pump before the next prop's texture, and the GPU step after it runs at the next boundary."""
    CpuFirstHooks.wait_for_previous, CpuFirstHooks.gpu_sleep, CpuFirstHooks.first_sleep = True, 0.2, 0.3
    try:
        world = World()
        world.run(requests("AEB"), hooks=CpuFirstHooks)
        assert world.states() == dict.fromkeys("AEB", "done"), world.results
        expect_alone(world, "AEB")
        first, second, third = world.hooks.order
        when = {(prop, name): (started, ended) for prop, name, _, started, ended in world.hooks.trace}
        assert all((prop, name) in when for prop in "AEB" for name in ("cpu1", "gpu1", "cpu2")), sorted(when)
        # the second prop's first step started after the first prop's GPU step had run at that boundary
        assert when[(second, "cpu1")][0] > when[(first, "gpu1")][1], (first, second, when)
        # the last prop's first step was handed over before the second prop's GPU step ran at its boundary (the
        # worker starts it while that step sleeps)
        assert when[(third, "cpu1")][0] < when[(second, "gpu1")][1], (second, third, when)
        main = threading.current_thread().name
        for prop, name, thread, *_ in world.hooks.trace:
            assert (thread == main) == name.startswith("gpu"), (prop, name, thread)
    finally:
        CpuFirstHooks.wait_for_previous, CpuFirstHooks.gpu_sleep, CpuFirstHooks.first_sleep = False, 0.0, 0.0


@test
def a_prop_in_its_last_step_no_longer_counts_as_in_flight() -> None:
    """IN_FLIGHT bounds the props whose steps hold their mesh and textures in RAM; a prop in its last step (the
    hub's export) is not one of them, so a fourth prop is decoded while three write their files. The last steps
    wait for the fourth prop's decode rather than for a fixed time: with the real loader (a garbage collection at
    every load, drop and trim) props came ~0.25 s apart, and a 0.5 s last step ended about when the third was
    decoded, so the count depended on the PC's pace; an engine that counted those props would hold the fourth
    decode until the wait gives up, and find 3."""
    from pixal3d.staged import runner

    assert runner.IN_FLIGHT == 3, runner.IN_FLIGHT
    CpuFirstHooks.last_waits_for = 4
    try:
        world = World()
        world.run(requests("CAEB"), hooks=CpuFirstHooks)
        assert world.states() == dict.fromkeys("CAEB", "done"), world.results
        expect_alone(world, "CAEB")
        assert world.hooks.most_alive == 4, world.hooks.most_alive
    finally:
        CpuFirstHooks.last_waits_for = 0


def _bare_runner(alive: list) -> Any:
    """A StagedRunner on stand-ins with nothing but tails to move: `alive` the props of its wave."""
    import types as pytypes
    from pixal3d.staged import parts
    from pixal3d.staged.runner import PartContext, StagedRunner

    class Loader:
        def set_cap(self, *args: Any, **kwargs: Any) -> int:
            return 0

        def trim(self) -> None:
            pass

    class Store:
        def close_prop(self, prop_id: str) -> None:
            pass

    events: list = []
    ctx = PartContext(engine=None, loader=Loader(), store=Store(), hooks=None,
                      options=pytypes.SimpleNamespace(spill=False), emit=events.append, cancelled=lambda: False,
                      pipeline_args={}, device="cpu")
    runner = StagedRunner(ctx, parts)
    runner._wave_now = alive
    return runner


@test
def after_the_last_decode_the_tail_with_the_most_steps_left_runs_its_gpu_step_first() -> None:
    """Once no prop of the wave is left to decode, the GPU steps run one at a time, the tail with the most steps
    left first (a geometry before a bake: it hands the worker the longer chain), the CPU steps collected and handed
    over between two GPU steps; while props remain to be decoded the tails keep their own order."""
    from pixal3d.staged.runner import PropState, _Tail
    from pixal3d.staged.types import FinishSteps

    def tails(runner: Any, order: list) -> None:
        for name, index in (("X", 3), ("Y", 1), ("Z", 1)):      # X at its bake, Y and Z at their geometry
            def step(label: str, name: str = name) -> Callable[[dict], None]:
                return lambda state: order.append((name, label, threading.current_thread().name))
            steps = FinishSteps(steps=[("cpu", step("bvh")), ("gpu", step("geometry")), ("cpu", step("unwrap")),
                                       ("gpu", step("bake")), ("cpu", step("export"))],
                                state={}, result=lambda state: {})
            prop = PropState(req=request("A" + name))
            prop.part, prop.state = "finish", "working"
            runner._tails.append(_Tail(prop, steps, index=index))

    main = threading.current_thread().name
    order: list = []
    runner = _bare_runner([])
    tails(runner, order)
    runner._pump_tails(until=lambda: not runner._tails)
    runner._worker.shutdown(wait=True)
    gpu = [(name, label) for name, label, thread in order if thread == main]
    assert gpu[:3] == [("Y", "geometry"), ("Z", "geometry"), ("X", "bake")], gpu
    assert sorted(gpu) == sorted([("X", "bake"), ("Y", "geometry"), ("Y", "bake"), ("Z", "geometry"),
                                  ("Z", "bake")]), gpu
    assert not runner._tails

    order = []
    waiting = PropState(req=request("A"))
    waiting.part = "tex"                                        # still to be decoded: the tails' own order
    runner = _bare_runner([waiting])
    tails(runner, order)
    runner._pump_tails(until=lambda: True)
    runner._worker.shutdown(wait=True)
    gpu = [(name, label) for name, label, thread in order if thread == main]
    assert gpu == [("X", "bake"), ("Y", "geometry"), ("Z", "geometry")], gpu


@test
def a_network_kept_for_the_next_prop_and_dropped_at_a_failure_is_read_ahead_again() -> None:
    """runner._drop_extra (a failure: the retry must find the card as it would without what was kept) reads the
    largest network it lets go of into the file cache again, as _set_aside does: the next prop of the group loads
    it soon, and from the disk a flow takes 1.1-1.8 s against 0.4 s from the cache."""
    world = World()
    world.headroom, world.limit, world.sizes = 12_000, 20_000, dict(CARD_SIZES)
    world.fail_step[("finish", "C", 1)] = [oom()]
    world.run(requests("ACBD"))
    assert world.states() == dict.fromkeys("ACBD", "done"), world.results
    expect_alone(world, "ACBD")
    log = world.log
    failed = next(index for index, entry in enumerate(log) if entry[:5] == ("event", "step", "finish", "C", 1))
    retry = next(index for index, entry in enumerate(log) if index > failed and entry[0] == "set_cap")
    between = log[failed:retry]
    dropped = [entry[1] for entry in between if entry[0] == "drop"]
    ahead = [entry[1] for entry in between if entry[0] == "prefetch"]
    if REAL_LOADER:
        return              # the worktree's own loader on the CPU keeps nothing: nothing is dropped
    assert dropped == ["tex_flow.mv"] and ahead == ["tex_flow.mv"], (dropped, ahead)


@test
def the_paired_finish_reads_its_decoders_ahead_before_the_next_kinds_texture_network() -> None:
    """runner._pair: the files read ahead as a texture group begins are the finish's decoders first (every prop of
    the group loads them before the next kind's texture network: in round N's six-prop batch the first finish read
    them from the disk, 1.77 + 1.74 s, while the read ahead had gone to the single-view texture network), then the
    next group's (the order of the whole batch: paired_finish_decodes_each_prop_right_after_its_texture)."""
    world = World()
    world.run(requests("ACBD"))
    assert world.states() == dict.fromkeys("ACBD", "done"), world.results
    ahead = [entry[1] for entry in world.entries("prefetch")]
    begins = ahead.index("tex_flow.mv") + 1                 # hr's last group asked for the multi-view texture flow
    assert ahead[begins:begins + 3] == ["shape_dec", "tex_dec", "tex_flow.sv"], ahead
    log = world.log
    asked = next(index for index, entry in enumerate(log) if entry[:2] == ("prefetch", "shape_dec"))
    first_finish = next(index for index, entry in enumerate(log) if entry[:3] == ("event", "prop_start", "finish"))
    assert asked < first_finish, "the decoders are asked for before the first finish loads them"
    expect_alone(world, "ACBD")


@test
def the_pair_counts_what_the_steps_of_a_bulky_prop_reserve_through_pytorch() -> None:
    """parts.pair_keep counts, for this prop and for each prop in its steps, the native reserve and what the
    GLB steps reserve through PyTorch (steps_torch_mib: the hook's figure up to 15 M faces, more beyond): on a card
    that leaves 7.2 GiB, an int8 texture network (1,557 MiB) stays beside the decode of a 1536 prop only when the
    bake of the 21.5 M-face prop in its steps fits beside it as well - in round I's run it did not, the step ran
    out of memory and ran again with the network gone (re-read from the disk)."""
    import torch
    from pixal3d.staged import parts
    from pixal3d.staged.runner import PropState

    sizes = {"tex_flow": 1557, "shape_dec": 905, "tex_dec": 905}

    class Loader:
        safety_mib = 300

        def __init__(self, cap: int):
            self.cap = cap

        def stats(self) -> dict:
            return {"cap_mib": self.cap}

        def headroom_mib(self) -> int:
            return 12_000

        def get(self, name: str) -> Any:
            return object() if name == "tex_flow.sv" else None

        def size_mib(self, name: str) -> int | None:
            return sizes.get(name.split(".")[0])

    class Store:
        def get_small(self, prop_id: str, key: str) -> dict:
            return {"coords": torch.zeros(31_000, 4, dtype=torch.int32), "res": torch.tensor([1536]),
                    "voxels": torch.tensor([5_000_000])}    # this prop's own steps: 11 M faces foreseen

    def rule(cap: int, natives: tuple) -> frozenset:
        ctx = types.SimpleNamespace(loader=Loader(cap), store=Store(), part="tex", hooks=None)
        return parts.pair_keep(ctx, PropState(req=request("A")), PropState(req=request("E")), list(natives))

    ork, dragon, knight = parts.glb_reserve(21_500_000), parts.glb_reserve(18_300_000), parts.glb_reserve(14_000_000)
    assert parts.reserve_faces(ork) // 100_000 == 215
    assert parts.steps_torch_mib(None, None, 14_000_000) == parts.HOOK_TORCH_MIB
    assert parts.steps_torch_mib(None, None, 21_500_000) > parts.HOOK_TORCH_MIB
    # a card whose cap for `tex` is 5,700 MiB: the steps of round I's two bulkiest props do not fit beside the
    # texture network (the ork's 2,890 + 1,680, the dragon's 2,698 + 1,500, with the network's 1,557), the knight's do
    assert rule(5_700, (ork,)) == frozenset() and rule(5_700, (dragon, knight)) == frozenset()
    assert rule(5_700, (knight,)) == frozenset({"tex_flow.sv"})
    # the full RTX 3060 keeps everything, beside the bulkiest prop's steps too
    assert rule(10_666, (ork, dragon)) == frozenset({"tex_flow.sv", "shape_dec", "tex_dec"})


@test
def the_pair_leaves_the_next_samplings_proj_cache_its_room() -> None:
    """parts.pair_keep keeps the decoders for the next prop's finish only when they fit beside that prop's texture
    sampling as it will run: its sampler and the kept proj_linear answers (one [tokens, 1536] bf16 table a block)
    of the blocks the cap in force has room for beside it (the rule of parts.proj_cache_blocks). On the full RTX
    3060 a next prop of 44 k tokens keeps its cache (4.2 GB with the margin) and the decoders give way; so they do
    to a flow of 60 blocks, which keeps the answers of the 31 that fit; a prop whose sampler leaves no room for a
    single table keeps none and leaves the decoders their room, as before; 36 k tokens fit all three with the int8
    network (the dragon's texture at 29 k: 7,045 MiB allocated with the cache and both decoders beside it)."""
    import torch
    from pixal3d.staged import parts
    from pixal3d.staged.runner import PropState

    sizes = {"tex_flow": 1557, "shape_dec": 905, "tex_dec": 905}

    class Projecting(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._proj_cache = None
            self.proj_linear = torch.nn.Linear(4, 1536, dtype=torch.bfloat16, device="meta")

    def flow(blocks: int) -> torch.nn.Module:
        return torch.nn.Sequential(*[Projecting() for _ in range(blocks)])

    class Loader:
        safety_mib = 300

        def __init__(self, network: Any):
            self.network = network

        def stats(self) -> dict:
            return {"cap_mib": 10_666}

        def headroom_mib(self) -> int:
            return 12_000

        def get(self, name: str) -> Any:
            return self.network if name == "tex_flow.sv" else None

        def size_mib(self, name: str) -> int | None:
            return sizes.get(name.split(".")[0])

    class Store:
        def __init__(self, tokens: int):
            self.tokens = tokens

        def get_small(self, prop_id: str, key: str) -> dict:
            return {"coords": torch.zeros(self.tokens, 4, dtype=torch.int32), "res": torch.tensor([1536]),
                    "voxels": torch.tensor([5_000_000])}

    def rule(network: Any, tokens: int) -> frozenset:
        ctx = types.SimpleNamespace(loader=Loader(network), store=Store(tokens), part="tex", hooks=None)
        return parts.pair_keep(ctx, PropState(req=request("A")), PropState(req=request("E")), [])

    everything = frozenset({"tex_flow.sv", "shape_dec", "tex_dec"})
    margin, parts.PROJ_CACHE_MARGIN_MIB = parts.PROJ_CACHE_MARGIN_MIB, 1_536     # the figures below are for this one
    try:
        assert parts.proj_cache_mib(flow(30), 36_000) > 3_000
        assert rule(object(), 36_000) == everything                     # no proj cache: as before
        assert rule(flow(30), 36_000) == everything                     # the cache, the sampler and the decoders fit
        assert rule(flow(30), 44_000) == frozenset({"tex_flow.sv"})     # the cache is kept; the decoders give way
        assert rule(flow(60), 44_000) == frozenset({"tex_flow.sv"})     # 31 of 60 blocks keep theirs
        assert parts.proj_cache_count(parts.sampler_mib(110_000, 1557),
                                      parts.proj_cache_tables_mib(flow(30), 110_000), 10_666) == 0
        assert rule(flow(30), 110_000) == everything                    # no room for a table: none is kept
        assert parts.next_sampler_mib(None, 36_000, 10_666) == parts.sampler_mib(36_000)
        sampler, tables = parts.sampler_mib(44_000, 1557), parts.proj_cache_tables_mib(flow(60), 44_000)
        assert parts.proj_cache_count(sampler, tables, 10_666) == 31
        assert parts.next_sampler_mib(flow(60), 44_000, 10_666, 1557) == \
            sampler + parts.PROJ_CACHE_SLACK * sum(tables[:31])
    finally:
        parts.PROJ_CACHE_MARGIN_MIB = margin


@test
def the_flows_keep_the_proj_linear_answers_of_as_many_blocks_as_the_cap_has_room_for() -> None:
    """parts.proj_cache_blocks: the first K blocks of a flow keep their proj_linear answer, K as many as fit beside
    the sampler under the cap in force with the margin (not all 30 or none): all of them on the full RTX 3060, some
    on a 7-8 GB card's cap, none on the ladder's tight rung, on the CPU or with the cap lifted. With an fp32 torso
    (the pascal profile) the sampler beside the tables counts its tokens twice as wide (proj_cache_sampler_mib), an
    fp16 one (turing) as bf16, and next_sampler_mib counts the same blocks. proj_cache_limit reads the count off a
    conditioning (True for all, none in a zero pass). In a batch ss, lr, hr and tex mark their conditional pass with
    the count, the zero pass never, and every sampling attempt drops its tables when it ends: after its draws, after
    an out-of-memory (the tight retry runs without them), a cancel or a failure."""
    import torch
    from pixal3d.modules import precision
    from pixal3d.modules.attention.proj_attention import proj_cache_limit
    from pixal3d.staged import parts

    class Projecting(torch.nn.Module):
        def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
            super().__init__()
            self._proj_cache = None
            self.proj_linear = torch.nn.Linear(4, 1536, dtype=dtype, device="meta")

    flow = torch.nn.Sequential(*[Projecting() for _ in range(30)])
    table = 30_000 * 1536 * 2 / 2**20                  # MiB of one block's answer at 30 k tokens (the ork)

    class Loader:
        safety_mib = 300

        def __init__(self, cap: int | None, weights: int):
            self.cap, self.weights = cap, weights

        def stats(self) -> dict:
            return {"cap_mib": self.cap}

        def size_mib(self, name: str) -> int | None:
            return self.weights if name.startswith("hr_flow.") else None

    def blocks(cap: int | None, mode: str = "normal", tokens: int = 30_000, network: Any = flow,
               weights: int = 1557) -> int:
        ctx = types.SimpleNamespace(loader=Loader(cap, weights), part="hr")
        return parts.proj_cache_blocks(ctx, network, tokens, "hr", mode)

    torso = precision.torso
    precision.torso = lambda: torch.bfloat16          # the figures below are the ampere profile's
    try:
        sampler = parts.sampler_mib(30_000, 1557)
        assert parts.proj_cache_sampler_mib(30_000, 1557) == sampler
        assert parts.proj_cache_tables_mib(flow, 30_000) == [table] * 30
        assert blocks(10_666) == 30                                     # the full RTX 3060
        on_8gb = int((6_368 - sampler - parts.PROJ_CACHE_MARGIN_MIB) // (parts.PROJ_CACHE_SLACK * table))
        assert 0 < on_8gb < 30 and blocks(6_368) == on_8gb, (blocks(6_368), on_8gb)     # a 7.2 GiB card's cap
        assert blocks(int(sampler + parts.PROJ_CACHE_MARGIN_MIB)) == 0     # no room for one table
        assert blocks(10_666, "tight") == blocks(10_666, "uncapped") == blocks(None) == 0
        assert blocks(10_666, tokens=0) == 0

        # an fp16 torso (turing) counts as bf16; an fp32 one (pascal) twice the tokens beside its fp32 tables: a
        # GTX 1080 Ti's cap with the ork at 1536 keeps 14 blocks where sampler_mib's figure would leave room for 16
        # (24 before round O, when sampler_mib counted an fp32 torso's tokens as bf16's)
        precision.torso = lambda: torch.float16
        assert parts.proj_cache_sampler_mib(30_000, 1557) == sampler and blocks(6_368) == on_8gb
        precision.torso = lambda: torch.float32
        wide = torch.nn.Sequential(*[Projecting(torch.float32) for _ in range(30)])
        pascal = parts.proj_cache_sampler_mib(30_500, 1917)
        assert pascal == parts.SAMPLER_SLACK * (1917 + 2 * parts.SAMPLER_MIB_PER_KTOKEN * 30.5)
        tables = parts.proj_cache_tables_mib(wide, 30_500)
        assert tables == [30_500 * 1536 * 4 / 2**20] * 30
        assert blocks(9_640, tokens=30_500, network=wide, weights=1917) == 14
        fp32_rate = getattr(parts, "SAMPLER_FP32_MIB_PER_KTOKEN", None)
        assert parts.proj_cache_count(parts.sampler_mib(30_500, 1917), tables, 9_640) == (24 if fp32_rate is None
                                                                                           else 16)
        if fp32_rate is not None:
            assert parts.sampler_mib(30_500, 1917) == parts.SAMPLER_SLACK * (1917 + fp32_rate * 30.5)
        assert parts.next_sampler_mib(wide, 30_500, 9_640, 1917) == \
            pascal + parts.PROJ_CACHE_SLACK * sum(tables[:14])
        assert parts.next_sampler_mib(None, 30_500, 9_640, 1917) == pascal
    finally:
        precision.torso = torso
    assert proj_cache_limit({"proj_cache": 7}, 30) == 7 and proj_cache_limit({"proj_cache": True}, 30) == 30
    assert proj_cache_limit({"proj_cache": 99}, 30) == 30 and proj_cache_limit({"proj": None}, 30) == 0
    assert proj_cache_limit({"proj_cache": 7, "zero": True}, 30) == 0 and proj_cache_limit((1, 2), 30) == 0

    # in a batch: what each sampling attempt marks its conditioning with, and the drop that ends it (the worktree's
    # own loader on the CPU has no cap to judge by: no block keeps its table, every attempt still drops)
    full = None if REAL_LOADER else 30
    asked: list[tuple] = []
    sample, drop, tables_of = FakeSampler.sample, parts._drop_proj_cache, parts.proj_cache_tables_mib

    def spy(self: Any, model: Any, noise: Any, cond: Any = None, neg_cond: Any = None, **kwargs: Any) -> Any:
        asked.append(("sample", CURRENT.now, self.stage, cond.get("proj_cache"), (neg_cond or {}).get("proj_cache")))
        return sample(self, model, noise, cond, neg_cond, **kwargs)

    def dropped(ctx: Any, network: Any, part: str, prop: Any, count: int) -> None:
        asked.append(("drop", (part, prop.id), count))
        drop(ctx, network, part, prop, count)

    def attempts() -> list[tuple]:
        """(prop-part, sampler stage, count) of each sampling attempt in order; each one dropped once, after it."""
        made = []
        assert len(asked) % 2 == 0, asked
        for (kind, now, stage, kept, zero), ended in zip(asked[::2], asked[1::2]):
            assert kind == "sample" and ended[0] == "drop", (kind, now, ended)
            assert ended[1] == now and ended[2] == (kept or 0), (now, kept, ended)
            assert zero is None, (stage, now)               # the zero pass never keeps
            made.append((now, stage, kept))
        return made

    def batch(names: str, setup: Callable[[World], Any]) -> tuple[World, list[tuple]]:
        asked.clear()
        world = World()
        world.limit = 20_000
        setup(world)
        world.run(requests(names))
        return world, attempts()

    parts.proj_cache_tables_mib = lambda network, tokens: [1.0] * 30 if "_flow." in getattr(network, "name", "") \
        else tables_of(network, tokens)
    parts._drop_proj_cache = dropped
    FakeSampler.sample = spy
    try:
        # the tight attempt after an out-of-memory runs with none; the results are those of the props made alone
        world, made = batch("AE", lambda world: world.fail_step.update({("ss", "A", 2): [oom()],
                                                                        ("hr", "E", 3): [oom()]}))
        assert world.states() == {"A": "done", "E": "done"}, world.results
        expect_alone(world, "AE")
        seen: dict[tuple, list] = {}
        for now, stage, kept in made:
            seen.setdefault((stage, now), []).append(kept)
        assert seen[("ss", ("ss", "A"))] == [full, None], seen      # the tight attempt keeps none
        assert seen[("ss", ("ss", "E"))] == [full], seen
        assert seen[("shape", ("hr", "E"))] == [full, None] and seen[("shape", ("hr", "A"))] == [full], seen
        assert seen[("shape", ("lr", "A"))] == [full] and seen[("tex", ("tex", "A"))] == [full], seen

        # a cancel at a texture step, an error at a detail step: the attempt still drops its tables
        world, made = batch("A", lambda world: setattr(world, "cancel_at", ("tex", "A", 5)))
        assert world.states() == {"A": "cancelled"} and made[-1] == (("tex", "A"), "tex", full), (world.results, made)
        world, made = batch("A", lambda world: world.fail_step.update({("hr", "A", 4): [RuntimeError("boom")]}))
        assert world.states() == {"A": "failed"} and made[-1] == (("hr", "A"), "shape", full), (world.results, made)
    finally:
        FakeSampler.sample = sample
        parts._drop_proj_cache = drop
        parts.proj_cache_tables_mib = tables_of


@test
def the_projection_attention_modules_keep_one_answer_until_it_is_dropped() -> None:
    """ProjectAttention and SparseProjectAttention (a tiny fp32 proj_linear, a stand-in global branch): with
    `cache` a conditional pass keeps proj_linear's answer and the next pass on the same tensor gets that very tensor
    back (the same values as without it); without `cache`, in a zero pass, or once the input changed in place
    nothing kept serves; proj_cache_tables gives each module's table in its dtype, drop_proj_caches the bytes it
    freed and leaves nothing kept."""
    import torch
    from pixal3d.modules.attention.proj_attention import ProjectAttention
    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.modules.sparse.attention.proj_attention import (SparseProjectAttention, drop_proj_caches,
                                                                  proj_cache_bytes, proj_cache_tables)

    class Global(torch.nn.Module):
        def forward(self, x: Any, context: Any) -> Any:
            return x * 0.5

    torch.manual_seed(5)
    dense, sparse = ProjectAttention(Global(), 8, 4), SparseProjectAttention(Global(), 8, 4)
    model = torch.nn.Sequential(dense, sparse)
    coords = torch.zeros(5, 4, dtype=torch.int32)
    coords[:, 1] = torch.arange(5)
    x, xs = torch.randn(1, 5, 8), SparseTensor(feats=torch.randn(5, 8), coords=coords)
    context = {"global": torch.randn(1, 3, 8), "proj": torch.randn(1, 5, 4)}
    sparse_context = {"global": torch.randn(1, 3, 8), "proj": SparseTensor(feats=torch.randn(5, 4), coords=coords)}

    plain, plain_sparse = dense(x, context), sparse(xs, sparse_context)
    assert dense._proj_cache is None and sparse._proj_cache is None
    dense(x, context, zero=True, cache=True)
    sparse(xs, sparse_context, zero=True, cache=True)
    assert dense._proj_cache is None and sparse._proj_cache is None     # a zero pass keeps nothing
    assert torch.equal(dense(x, context, cache=True), plain)
    assert torch.equal(sparse(xs, sparse_context, cache=True).feats, plain_sparse.feats)
    kept, kept_sparse = dense._proj_cache[1], sparse._proj_cache[1]
    assert dense.proj_feats(context["proj"], True) is kept
    assert sparse.proj_feats(sparse_context["proj"], True) is kept_sparse
    assert torch.equal(dense(x, context, cache=True), plain) and dense._proj_cache[1] is kept
    assert dense.proj_feats(context["proj"], False) is not kept
    context["proj"].mul_(1.0)                                           # the same values, another version
    assert dense.proj_feats(context["proj"], True) is not kept

    assert proj_cache_tables(model, 7) == [7 * 8 * 4, 7 * 8 * 4] and proj_cache_bytes(model, 7) == 2 * 7 * 8 * 4
    assert proj_cache_tables(object(), 7) == [] and proj_cache_bytes(object(), 7) == 0
    assert drop_proj_caches(model) == 2 * 5 * 8 * 4
    assert dense._proj_cache is None and sparse._proj_cache is None and drop_proj_caches(model) == 0
    assert drop_proj_caches(object()) == 0


@test
def the_flows_hand_the_proj_cache_to_their_first_blocks_only() -> None:
    """SparseStructureFlowModel and SLatFlowModel (4 blocks at a tiny width, each block's forward a pass-through
    that records what it is given): a conditioning with "proj_cache": 2 reaches blocks 0-1 with proj_cache on and
    blocks 2-3 with it off, True reaches every block, a zero pass and a conditioning without the entry none."""
    import torch
    from pixal3d.models.sparse_structure_flow import SparseStructureFlowModel
    from pixal3d.models.structured_latent_flow import SLatFlowModel
    from pixal3d.modules.sparse import SparseTensor

    def recorded(network: Any) -> list:
        seen: list = []
        for block in network.blocks:
            def passing(h: Any, *args: Any, zero: bool = False, proj_cache: bool = False, **kwargs: Any) -> Any:
                seen.append((proj_cache, zero))
                return h
            block.forward = passing
        return seen

    def check(network: Any, x: Any, cond: dict) -> None:
        seen = recorded(network)
        t = torch.tensor([0.5])
        for asked, cached, zero in (({"proj_cache": 2}, [True, True, False, False], False),
                                    ({"proj_cache": True}, [True] * 4, False),
                                    ({"proj_cache": 2, "zero": True}, [False] * 4, True),
                                    ({}, [False] * 4, False)):
            seen.clear()
            network(x, t, {**cond, **asked})
            assert seen == [(keep, zero) for keep in cached], (type(network).__name__, asked, seen)

    sizes = dict(in_channels=2, model_channels=8, cond_channels=8, out_channels=2, num_blocks=4, num_heads=2,
                 image_attn_mode="proj", proj_in_channels=4)
    torch.manual_seed(6)
    check(SparseStructureFlowModel(resolution=2, **sizes), torch.randn(1, 2, 2, 2, 2),
          {"global": torch.randn(1, 3, 8), "proj": torch.randn(1, 8, 4)})
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 1]], dtype=torch.int32)
    check(SLatFlowModel(resolution=8, **sizes), SparseTensor(feats=torch.randn(3, 2), coords=coords),
          {"global": torch.randn(1, 3, 8), "proj": SparseTensor(feats=torch.randn(3, 4), coords=coords)})


@test
def finish_steps_are_told_the_room_they_may_keep_and_let_go_after_an_out_of_memory() -> None:
    """Before every step the runner writes state["room_mib"] (parts.tail_room): 0 on a loader that cannot say what
    the card has; else what the card has beside the reserves and what the props still to decode (or another prop's
    steps) need, less what the other tails hold. A decode that runs out of memory makes every tail let go."""
    from pixal3d.staged import parts

    world = World()
    world.run(requests("AE"), hooks=RoomHooks)
    assert world.states() == {"A": "done", "E": "done"}, world.results
    assert len(world.hooks.rooms) == 8 and all(room == 0 for _, _, room in world.hooks.rooms), world.hooks.rooms
    assert not world.hooks.released
    if REAL_LOADER:
        return          # the worktree's own loader on the CPU has no card to report room on: the rest needs the fake

    ram = parts.ram_available_mib
    parts.ram_available_mib = lambda: 60_000
    RoomHooks.hold = 700
    StepHooks.sleep = 0.3
    try:
        world = World()
        world.headroom = 12_000
        world.sizes["tex_flow"] = 100_000       # the pair keeps nothing on the card (tested by itself)
        world.run(requests("AEB"), hooks=RoomHooks)
        assert world.states() == dict.fromkeys("AEB", "done"), world.results
        expect_alone(world, "AEB")
        rooms = {(prop, step): room for prop, step, room in world.hooks.rooms}
        # the fakes' meshes are tiny: what another prop's steps need (HOLD_BESIDE_MIB) bounds every room alike
        first = 12_000 - (parts.NATIVE_RESERVE["finish"] + parts.FINISH_SAFETY_MIB) - parts.HOLD_BESIDE_MIB \
            - parts.HOLD_SLACK_MIB
        assert rooms[("B", "gpu1")] == first, (rooms, first)     # B (1536) goes first: parts.pair_order
        assert rooms[("A", "gpu1")] == first - 700, rooms        # B holds while A's first step runs
        assert set(rooms.values()) <= {first, first - 700, first - 1400}, rooms     # up to two others hold
        assert not world.hooks.released

        world = World()
        world.headroom = 12_000
        world.sizes["tex_flow"] = 100_000
        world.fail_step[("finish", "E", 1)] = [oom()]
        world.run(requests("AE"), hooks=RoomHooks)
        assert world.states() == {"A": "done", "E": "done"}, world.results
        assert world.hooks.released == ["A"], world.hooks.released     # let go before E's decode was tried again
        expect_alone(world, "AE")
    finally:
        parts.ram_available_mib = ram
        RoomHooks.hold = 0
        StepHooks.sleep = 0.0


@test
def finish_steps_run_their_cpu_steps_on_the_worker_while_the_next_prop_is_decoded() -> None:
    from pixal3d.staged import runner

    StepHooks.sleep = 0.3
    try:
        decodes: list[tuple] = []
        world = World()
        world.taps["decode_shape"] = [lambda **info: decodes.append((CURRENT.now, time.perf_counter()))]
        world.run(requests("AEB"), hooks=StepHooks)
        assert world.states() == {"A": "done", "E": "done", "B": "done"}, world.results
        expect_alone(world, "AEB")
        trace = world.hooks.trace
        assert [(prop, name) for prop, name, *_ in trace if prop == "A"] == \
            [("A", "gpu1"), ("A", "cpu1"), ("A", "gpu2"), ("A", "cpu2")]
        main = threading.current_thread().name
        for prop, name, thread, *_ in trace:
            if name.startswith("gpu"):
                assert thread == main, (prop, name, thread)
            else:
                assert thread.startswith("p3d-finish-cpu"), (prop, name, thread)
        # E was decoded while A's first CPU step ran
        _, _, _, cpu1_start, cpu1_end = next(entry for entry in trace if entry[:2] == ("A", "cpu1"))
        e_decoded = next(when for now, when in decodes if now == ("finish", "E"))
        assert cpu1_start < e_decoded < cpu1_end, (cpu1_start, e_decoded, cpu1_end)
        # the hook's steps count on after the three of the decode, the prop is done when its last step is
        steps = [(step, total) for _, part, prop, step, total, _ in world.events("step") if (part, prop) == ("finish", "A")]
        assert steps == [(1, 3), (2, 3), (3, 3), (4, 7), (5, 7), (6, 7), (7, 7)], steps
        log = world.log
        a_done = next(i for i, entry in enumerate(log) if entry[:4] == ("event", "prop_done", "finish", "A"))
        e_start = next(i for i, entry in enumerate(log) if entry[:4] == ("event", "prop_start", "finish", "E"))
        assert e_start < a_done
        assert world.hooks.most_alive <= runner.IN_FLIGHT, world.hooks.most_alive
    finally:
        StepHooks.sleep = 0.0

    # a failing CPU step fails its prop alone; a GPU step that runs out of memory goes down the ladder
    world = World()
    world.run(requests("AE"), hooks=StepHooks)
    hook_caps = len([entry for entry in world.entries("set_cap") if entry[1] != 300])
    world = World()
    world.build(hooks=StepHooks)
    world.hooks.fail_in[("A", "cpu1")] = [ValueError("boom")]
    world.now = ("", None)
    world.results = world.engine.run(requests("AE"), world.work, on_event=world.on_event, cancelled=lambda: False)
    assert world.states() == {"A": "failed", "E": "done"}, world.results
    assert world.results["A"]["error"] == {"kind": "error", "detail": "ValueError: boom"}
    assert "A" not in world.finished
    expect_alone(world, "E")
    for failures, spill, expected in ((1, False, "done"), (2, False, "failed"), (2, True, "done")):
        world = World()
        world.build(hooks=StepHooks, spill=spill)
        world.hooks.fail_in[("A", "gpu2")] = [oom() for _ in range(failures)]
        world.now = ("", None)
        world.results = world.engine.run(requests("AE"), world.work, on_event=world.on_event, cancelled=lambda: False)
        assert world.states() == {"A": expected, "E": "done"}, (failures, spill, world.results)
        caps = len([entry for entry in world.entries("set_cap") if entry[1] != 300])
        if expected == "done":
            expect_alone(world, "AE")
            assert world.results["A"]["result"]["slow"] == (["finish"] if spill else []), world.results["A"]
            assert caps == hook_caps + 1, (caps, hook_caps)            # one retry under the cap
        else:
            assert world.results["A"]["error"]["kind"] == "oom"
            assert not world.events("prop_spilled")
        assert len(world.events("prop_spilled")) == (1 if spill and failures == 2 else 0)


@test
def a_cancel_lets_the_props_in_their_finish_steps_end() -> None:
    StepHooks.sleep, StepHooks.cancel_in_cpu1 = 0.2, True
    try:
        world = World()
        world.run(requests("AEB"), hooks=StepHooks)
        assert world.states() == {"B": "done", "A": "cancelled", "E": "cancelled"}, world.results   # B goes first
        assert [(prop, name) for prop, name, *_ in world.hooks.trace] == \
            [("B", "gpu1"), ("B", "cpu1"), ("B", "gpu2"), ("B", "cpu2")]
        expect_alone(world, "B")
    finally:
        StepHooks.sleep, StepHooks.cancel_in_cpu1 = 0.0, False


@test
def a_swallowed_out_of_memory_in_a_convolution_part_sends_the_attempt_down_the_ladder() -> None:
    from pixal3d.staged import runner

    counter = {"ooms": 0, "bumped": []}
    real = runner.num_ooms
    runner.num_ooms = lambda device: counter["ooms"]
    try:
        world = World()

        def bump(spec: str, **info: Any) -> None:       # PyTorch counted a failed allocation, nobody raised
            if world.now == ("cond_hr", "A") and spec == "tex_1024" and not counter["bumped"]:
                counter["ooms"] += 1
                counter["bumped"].append(world.now)

        world.taps["cond_tokens"] = [bump]
        world.run(requests("AE"))
        assert counter["bumped"] == [("cond_hr", "A")]
        assert world.states() == {"A": "done", "E": "done"} and world.results["A"]["result"]["slow"] == []
        expect_alone(world, "AE")
        puts = [entry[3] for entry in world.entries("store", "put_device", "A") if entry[-1] == ("cond_hr", "A")]
        assert puts == ["cond_hr", "cond_tex", "cond_hr", "cond_tex"], puts
        thrown = [entry[3] for entry in world.entries("store", "discard", "A") if entry[3] in ("cond_hr", "cond_tex")
                  and entry[-1] == ("cond_hr", "A")]
        assert thrown == ["cond_hr", "cond_tex"], thrown

        # the counter moving in a part without convolutions means nothing
        world = World()
        world.taps["upsample"] = [lambda **info: counter.__setitem__("ooms", counter["ooms"] + 1)]
        world.run(requests("A"))
        assert world.states() == {"A": "done"} and len(world.entries("event", "prop_start", "upsample")) == 1
        # each part the runner caps: once before its group's networks are read, once for the one attempt
        assert len(world.entries("set_cap", 200)) == 20
    finally:
        runner.num_ooms = real


@test
def classifier() -> None:
    import torch
    from pixal3d.staged import PropFailed
    from pixal3d.staged.runner import classify

    table = [
        (torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"), "oom"),
        (RuntimeError("CUDA error: out of memory"), "oom"),
        (RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"), "oom"),
        (RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`"), "oom"),
        (RuntimeError("Cuda error: 2[cudaMalloc(&m_gpuPtr, size);]"), "oom"),
        (MemoryError(), "ram"),
        (OSError("[WinError 1455] The paging file is too small for this operation to complete"), "ram"),
        (RuntimeError("[enforce fail at alloc_cpu.cpp:116] data. DefaultCPUAllocator: not enough memory"), "ram"),
        (FileNotFoundError(2, "No such file or directory", "ckpts/x.safetensors"), "missing_file"),
        (PropFailed("This slice is empty.", "input"), "input"),
        (PropFailed("no"), "error"),
        (ValueError("out of memory"), "error"),             # only a RuntimeError's text counts
        (RuntimeError("CUDA error: device-side assert triggered"), "error"),
        (KeyError("cond_lr"), "error"),
    ]
    for exc, kind in table:
        assert classify(exc) == kind, (exc, classify(exc), kind)


@test
def a_cuda_error_fails_one_prop_or_breaks_the_batch_as_the_probe_says() -> None:
    from pixal3d.staged import BatchBroken, runner

    real = runner.probe
    probes: list[str] = []
    try:
        runner.probe = lambda device: probes.append(device) or True
        world = World()
        world.fail_step[("hr", "A", 2)] = [RuntimeError("CUDA error: unspecified launch failure")]
        world.run(requests("AE"))
        assert probes == ["cpu"] and world.states() == {"A": "failed", "E": "done"}
        assert world.results["A"]["error"]["kind"] == "error"
        expect_alone(world, "E")

        runner.probe = lambda device: False
        world = World()
        world.fail_step[("hr", "A", 2)] = [RuntimeError("CUDA error: unspecified launch failure")]
        try:
            world.run(requests("AE"))
        except BatchBroken as broken:
            assert broken.kind == "restart" and "unspecified launch failure" in broken.detail
        else:
            raise AssertionError("BatchBroken was not raised")
        assert world.store.closed and world.engine._store is None

        # other failures never reach the probe
        runner.probe = lambda device: 1 / 0
        world = World()
        world.fail_step[("hr", "A", 2)] = [RuntimeError("shape mismatch")]
        world.run(requests("AE"))
        assert world.states() == {"A": "failed", "E": "done"}
    finally:
        runner.probe = real


@test
def cancel_at_every_check_point() -> None:
    def cancelled_cleanly(world: World, done: str = "") -> None:
        assert world.states() == {prop: "done" if prop in done else "cancelled" for prop in world.results}
        assert not world.store.props and world.store.closed, world.store.props
        assert not world.loader.modules and not world.loader.capped()
        assert not world.checkpoints(), "a cancelled prop's checkpoint is left in the work folder"
        assert world.entries("store", "flush")
        assert all("result" not in result for prop, result in world.results.items() if prop not in done)

    # at a sampler step: nothing runs after that step
    world = World()
    world.cancel_at = ("tex", "C", 5)
    world.run(requests("ACE", request_hash="h"))
    cancelled_cleanly(world)
    last = [event for event in world.events() if not event[0].startswith("net_")][-1]
    assert last[:4] == ("step", "tex", "C", 5), last

    # before a prop of a part
    world = World()
    world.taps["camera"] = [lambda prop: setattr(world, "cancel", prop == "A")]
    world.run(requests("AE"))
    cancelled_cleanly(world)
    assert world.starts()[-1] == ("camera", "A")

    # during a network load (the loader raises between two tensors)
    world = World()
    world.taps["cond_tokens"] = [lambda spec, **info: setattr(world, "cancel", world.now == ("cond_lr", "E"))]
    world.run(requests("AE"))
    cancelled_cleanly(world)
    assert world.entries("load")[-1][1] == "lr_flow.sv" and world.starts()[-1] == ("cond_lr", "E")

    # between the two decoders of `finish`, and before the hook: the prop is not saved
    for tap in ("decode_shape", "fill_holes"):
        world = World()
        world.taps[tap] = [lambda **info: setattr(world, "cancel", True)]
        world.run(requests("AE"))
        cancelled_cleanly(world)
        assert not world.finished

    # inside the hook: that prop is finished and saved, the rest is cancelled
    world = World()
    world.taps["finish"] = [lambda prop, mesh: setattr(world, "cancel", True)]
    world.run(requests("ACE"))
    cancelled_cleanly(world, done="C")
    expect_alone(world, "C")

    # before anything
    world = World()
    world.cancel = True
    world.run(requests("AC"))
    cancelled_cleanly(world)
    assert not world.starts() and not world.entries("load")


@test
def a_failing_prop_does_not_touch_the_others() -> None:
    world = World()
    world.fail_step[("hr", "C", 7)] = [RuntimeError("boom")]
    world.run(requests("ACBD", request_hash="h"))
    assert world.states() == {"A": "done", "C": "failed", "B": "done", "D": "done"}
    assert world.results["C"]["error"] == {"kind": "error", "detail": "RuntimeError: boom"}
    expect_alone(world, "ABD")
    log = world.log
    closed = log.index(next(entry for entry in log if entry[:3] == ("store", "close_prop", "C")))
    following = next(index for index, entry in enumerate(log) if entry[:4] == ("event", "prop_start", "hr", "D"))
    assert closed < following, "the failed prop's folder is closed before the next prop starts"
    failed = [(part, prop, detail) for _, part, prop, _, _, detail in world.events("prop_failed")]
    assert failed == [("hr", "C", {"kind": "error", "detail": "RuntimeError: boom"})], failed
    assert "C" not in [prop for part, prop in world.starts() if part in ("tex", "finish")]
    for _, part, _, _, _, detail in world.events("part_start"):
        assert ("C" in detail["props"]) == (part not in ("camera", "tex", "finish")), (part, detail)
    assert not world.checkpoints() and "C" not in world.finished

    # a picture the artist has to fix is known after the first part, and says so
    world = World()
    world.run([request("A"), request("E", payload={"color": (1, 2, 3), "views": 1, "bad": True}), request("C")])
    assert world.states() == {"A": "done", "E": "failed", "C": "done"}
    assert world.results["E"]["error"] == {"kind": "input", "detail": "This slice is empty."}
    assert [prop for part, prop in world.starts() if part == "camera"] == ["A"]
    expect_alone(world, "AC")

    # a failure that names its kind keeps it (the store fails a prop it cannot keep with kind "disk")
    from pixal3d.staged import PropFailed

    world = World()

    def full(spec: str, **info: Any) -> None:
        if world.now == ("cond_lr", "A"):
            raise PropFailed("Not enough free disk space.", "disk")

    world.taps["cond_tokens"] = [full]
    world.run(requests("AE"))
    assert world.states() == {"A": "failed", "E": "done"}
    assert world.results["A"]["error"] == {"kind": "disk", "detail": "Not enough free disk space."}

    # everything failed: the batch still ends in order
    world = World()
    world.fail_step[("ss", "A", 1)] = [RuntimeError("x")]
    world.run(requests("A"))
    assert world.states() == {"A": "failed"} and not world.loader.modules and not world.store.props


@test
def a_network_that_cannot_be_loaded_fails_its_group() -> None:
    missing = lambda: FileNotFoundError(2, "No such file or directory", "ckpts/hr_flow_mv.safetensors")  # noqa: E731

    world = World()
    world.fail_load["hr_flow.mv"] = [missing(), missing()]
    world.run(requests("ACBD"))
    assert world.states() == {"A": "done", "C": "failed", "B": "done", "D": "failed"}
    for prop in "CD":
        assert world.results[prop]["error"]["kind"] == "mv_missing", world.results[prop]
    expect_alone(world, "AB")
    tries = [index for index, entry in enumerate(world.log) if entry[:2] == ("load", "hr_flow.mv")]
    assert len(tries) == 2
    between = [entry[0] for entry in world.log[tries[0]:tries[1]]]
    assert "drop_all" in between and "trim" in between, between

    # a load that fails once is simply made again
    world = World()
    world.fail_load["lr_flow.sv"] = [oom()]
    world.run(requests("AC"))
    assert world.states() == {"A": "done", "C": "done"}
    expect_alone(world, "AC")

    # a network both kinds share is not a multi-view problem; the group fails with what happened
    world = World()
    world.fail_load["lr_flow.sv"] = [oom(), oom()]
    world.fail_load["naf"] = []
    world.run(requests("AC"))
    assert world.states() == {"A": "failed", "C": "done"} and world.results["A"]["error"]["kind"] == "oom"

    world = World()
    world.fail_load["dino"] = [missing(), missing(), missing(), missing()]
    world.run(requests("AC"))
    assert {prop: result["error"]["kind"] for prop, result in world.results.items()} == {
        "A": "missing_file", "C": "missing_file"}, world.results

    # a network a part loads itself (background removal, the decoders) fails the prop that needed it
    world = World()
    world.fail_load["rembg"] = [missing()]
    world.run(requests("AB"))                           # B's picture carries its own transparency
    assert world.states() == {"A": "failed", "B": "done"} and world.results["A"]["error"]["kind"] == "missing_file"


@test
def without_the_multi_view_files_only_multi_view_props_fail() -> None:
    world = World(multiview=False)
    world.run(requests("ACD"))
    assert world.engine.multiview is False
    assert world.states() == {"A": "done", "C": "failed", "D": "failed"}
    assert world.results["C"]["error"]["kind"] == "mv_missing"
    assert world.starts()[0] == ("cutout", "A") and all(prop == "A" for _, prop in world.starts())
    assert not [name for name in world.loader.checkpoints if name.endswith(".mv")]
    expect_alone(world, "A")
    assert World().build().multiview is True


@test
def resume_runs_only_finish_for_props_whose_sampling_was_done() -> None:
    class Died(SystemExit):
        pass

    def die(prop: str, mesh: Any) -> None:
        if prop == "A":
            raise Died()

    world = World()
    world.taps["finish"] = [die]
    names = ["A", "B", "C"]
    batch = [request(name, request_hash=f"hash-{name}") for name in names]
    try:
        world.run(batch, paired=False)                  # finish order: C, A (dies), B
    except Died:
        pass
    else:
        raise AssertionError("the simulated process death was swallowed")
    assert sorted(world.finished) == ["C"] and world.checkpoints() == ["A", "B"], (world.finished,
                                                                                   world.checkpoints())
    assert world.store.closed

    world.taps.clear()
    world.log.clear()
    world.finished.clear()
    again = [request("A", request_hash="hash-A"), request("B", request_hash="hash-B"),
             request("E", request_hash="hash-E"), request("D", request_hash="hash-D")]
    world.run(again, resume=True, paired=False)
    assert world.states() == dict.fromkeys("ABED", "done"), world.results
    assert sorted(prop for _, _, prop, *_ in world.events("prop_resumed")) == ["A", "B"]
    assert [part for part, prop in world.starts() if prop in "AB"] == ["finish", "finish"]
    assert len([part for part, prop in world.starts() if prop == "E"]) == 11
    for prop in "AB":
        want = {name: value for name, value in alone(prop).items() if name.startswith("mesh/")}
        got = {f"mesh/{name}": getattr(world.finished[prop][0], name) for name in ("vertices", "faces", "coords",
                                                                                  "attrs")}
        assert not differing(got, want), (prop, differing(got, want))
        assert world.results[prop]["result"]["resumed"] is True
    assert world.results["E"]["result"]["resumed"] is False
    expect_alone(world, "ED")
    assert not world.checkpoints() and not world.store.props
    assert [prop for part, prop in world.starts() if part == "finish"] == ["D", "A", "B", "E"]

    # resumed props alone need the two decoders and nothing else
    world = World()
    world.run([request("A", request_hash="1"), request("C", request_hash="2")], paired=False)
    assert not world.checkpoints(), "a finished prop's checkpoint is removed with its folder"
    world = World()
    world.taps["finish"] = [lambda prop, mesh: (_ for _ in ()).throw(Died())]
    try:
        world.run([request("A", request_hash="1"), request("C", request_hash="2")], paired=False)
    except Died:
        pass
    world.taps.clear()
    world.log.clear()
    world.run([request("A", request_hash="1"), request("C", request_hash="2")], resume=True, paired=False)
    assert sorted({name for _, name, _ in world.entries("load")}) == ["shape_dec", "tex_dec"]
    assert world.states() == {"A": "done", "C": "done"}

    # another request under the same id, or a batch that does not ask for it, starts from the beginning
    for changed, resume in ((True, True), (False, False)):
        world = World()
        world.taps["finish"] = [lambda prop, mesh: (_ for _ in ()).throw(Died())]
        try:
            world.run([request("A", request_hash="1")], paired=False)
        except Died:
            pass
        world.taps.clear()
        world.log.clear()
        world.run([request("A", request_hash="2" if changed else "1")], resume=resume, paired=False)
        assert len(world.starts()) == 11 and not world.events("prop_resumed"), (changed, resume)
        assert world.results["A"]["result"]["resumed"] is False
        expect_alone(world, "A")

    # what was noted before the restart - a part that needed the cap lifted, redraws of the first stage - is
    # still told after it
    world = World()
    world.redraws = {"A": 1}
    world.fail_step[("hr", "A", 4)] = [oom(), oom()]
    world.taps["finish"] = [lambda prop, mesh: (_ for _ in ()).throw(Died())]
    try:
        world.run([request("A", request_hash="1")], spill=True, paired=False)
    except Died:
        pass
    assert world.checkpoints() == ["A"]
    world.taps.clear()
    world.finished.clear()
    world.run([request("A", request_hash="1")], resume=True, spill=True, paired=False)
    result = world.results["A"]["result"]
    assert result["resumed"] is True and result["slow"] == ["hr"] and result["ss_retries"] == 1, result
    assert world.finished["A"][1] == {"ss_retries": 1, "ss_blocked": False, "slow": ["hr"]}, world.finished["A"][1]

    # without a request hash nothing is written for a resume
    world = World()
    world.run(requests("A"))
    assert not world.entries("store", "checkpoint")
    world = World()
    world.run([request("A", request_hash="1")])
    assert [entry[3] for entry in world.entries("store", "checkpoint")] == [["coords", "res", "shape_slat",
                                                                            "tex_slat"]]


HI3D_STORED = ("hi3d_lat", "coords_hr", "shape_slat", "tex_slat", "hi3d_mesh", "hi3d_subs")


def hi3d_signature(world: World, prop_id: str) -> dict[str, Any]:
    """What a Hi3D prop left behind: the last value of each of its own items and the decoded mesh."""
    puts = world.puts[prop_id]
    tensors = {f"{key}/{name}": value for key in HI3D_STORED for name, value in puts[key].items()}
    mesh, _ = world.finished[prop_id]
    tensors.update({f"mesh/{name}": getattr(mesh, name) for name in ("vertices", "faces", "coords", "attrs")})
    return tensors


def hi3d_alone(name: str) -> dict[str, Any]:
    if name not in _alone:
        world = World()
        world.run([request(name)], hi3d=True)
        assert world.states() == {name: "done"}, world.results
        _alone[name] = hi3d_signature(world, name)
    return _alone[name]


@test
def hi3d_props_take_their_own_geometry_parts_and_pixal3d_props_theirs() -> None:
    from pixal3d.staged import PARTS

    world = World()
    world.run(requests("AHCMI"), hi3d=True, paired=False)
    assert world.states() == dict.fromkeys("AHCMI", "done"), world.results
    started = world.starts()
    pixal3d = [part for part in pixal3d_parts()]
    hi3d_parts = [part for part in PARTS if part not in ("cond_lr", "lr", "upsample", "hr")]
    for prop, multi, passes in (("A", False, 0), ("C", True, 0), ("H", False, 3), ("I", False, 2), ("M", True, 2)):
        want = pixal3d if not passes else [part for part in hi3d_parts
                                           if not (part[:-1] in ("hi3d_p", "hi3d_d") and int(part[-1]) > passes)]
        want = [part for part in want if part != "camera" or not multi]
        got = [part for part, who in started if who == prop]
        assert got == want, (prop, got)
    # every part for all its props, the multi-view ones first
    order = [part for part, _ in started]
    assert order == sorted(order, key=PARTS.index), order
    for part in PARTS:
        props = [who for name, who in started if name == part]
        assert props == sorted(props, key=lambda who: who not in "CM"), (part, props)
    assert {name for _, name, *_ in world.entries("load")} >= set(HI3D_NETWORKS) - {"hi3d_t1024"} | {"hi3d_t1024"}


@test
def hi3d_results_do_not_depend_on_the_batch_or_its_order() -> None:
    world = World()
    world.run(requests("IAHE"), hi3d=True)
    assert world.states() == dict.fromkeys("IAHE", "done"), world.results
    for name in "HI":
        different = differing(hi3d_signature(world, name), hi3d_alone(name))
        assert not different, f"Hi3D prop {name} differs from the same prop made alone in: {different}"
    expect_alone(world, "AE")
    mesh, _ = world.finished["H"]
    assert mesh.vertices.dtype == torch_float32() and mesh.faces.dtype == torch_int32()
    assert float(mesh.vertices.abs().max()) <= 0.5, "the mesh came back outside Pixal3D's frame"


@test
def a_hi3d_mesh_past_the_cube_is_scaled_into_it_and_a_huge_one_reduced() -> None:
    import numpy as np
    from pixal3d.staged import hi3d

    inside = np.array([[0.4, -0.2, 0.1], [-0.3, 0.45, 0.0]])
    assert hi3d.into_cube(inside)[1] == 1.0 and hi3d.into_cube(inside)[0] is inside
    outside = np.array([[0.6, -0.2, 0.1], [-0.3, 0.45, 0.0]])
    fitted, scale = hi3d.into_cube(outside)
    assert abs(scale - hi3d.CUBE_FIT / 0.6) < 1e-12 and np.allclose(fitted, outside * scale)
    assert float(np.abs(fitted).max()) <= hi3d.CUBE_FIT
    # a grid of triangles twice over the target comes back under it, within the same bounds
    side = 40
    xs, ys = np.meshgrid(np.linspace(-0.4, 0.4, side), np.linspace(-0.4, 0.4, side), indexing="ij")
    vertices = np.stack([xs.ravel(), ys.ravel(), 0.05 * np.sin(8 * xs.ravel())], axis=1)
    quads = np.array([[i * side + j, (i + 1) * side + j, (i + 1) * side + j + 1, i * side + j + 1]
                      for i in range(side - 1) for j in range(side - 1)])
    faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]).astype(np.int64)
    try:
        fewer_vertices, fewer_faces = hi3d.simplified(vertices, faces, faces.shape[0] // 2)
    except ImportError:
        return                                   # pyfqmr is Hi3D's package: not in every check venv
    assert fewer_faces.shape[0] <= faces.shape[0] // 2 + 2 and fewer_faces.shape[0] > faces.shape[0] // 4
    assert fewer_vertices.dtype == np.float64 and fewer_faces.dtype == np.int64
    assert fewer_vertices.flags["C_CONTIGUOUS"] and fewer_faces.flags["C_CONTIGUOUS"]
    assert np.abs(fewer_vertices).max() <= np.abs(vertices).max() + 1e-6
    assert hi3d.simplified(vertices, faces, faces.shape[0])[1] is faces     # under the target: untouched


def torch_float32() -> Any:
    import torch

    return torch.float32


def torch_int32() -> Any:
    import torch

    return torch.int32


@test
def hi3d_props_resume_at_their_finish_with_their_own_mesh() -> None:
    class Died(SystemExit):
        pass

    def die(prop: str, mesh: Any) -> None:
        if prop == "H":
            raise Died()

    world = World()
    world.taps["finish"] = [die]
    try:
        world.run([request("H", request_hash="h"), request("A", request_hash="a")], hi3d=True, paired=False)
    except Died:
        pass
    else:
        raise AssertionError("the simulated process death was swallowed")
    assert "H" in world.checkpoints(), world.checkpoints()
    world.taps.clear()
    world.log.clear()
    world.finished.clear()
    world.run([request("H", request_hash="h"), request("A", request_hash="a")], resume=True, hi3d=True, paired=False)
    assert world.states() == {"H": "done", "A": "done"}, world.results
    assert [part for part, prop in world.starts() if prop == "H"] == ["finish"]
    want = {name: value for name, value in hi3d_alone("H").items() if name.startswith("mesh/")}
    got = {f"mesh/{name}": getattr(world.finished["H"][0], name) for name in ("vertices", "faces", "coords", "attrs")}
    assert not differing(got, want), differing(got, want)

    # resumed alone, a Hi3D prop needs the texture decoder and nothing else
    world = World()
    world.taps["finish"] = [die]
    try:
        world.run([request("H", request_hash="h")], hi3d=True, paired=False)
    except Died:
        pass
    world.taps.clear()
    world.log.clear()
    world.run([request("H", request_hash="h")], resume=True, hi3d=True, paired=False)
    assert {name for _, name, _ in world.entries("load")} == {"tex_dec"}, world.entries("load")


@test
def a_hi3d_decode_gets_the_cap_less_the_allocators_margin_and_less_again_on_the_tight_rung() -> None:
    from pixal3d.staged import hi3d

    if REAL_LOADER:             # the fake loader answers the cap it was given: the budgets come from it
        return
    budgets: list = []
    world = World()
    world.taps["hi3d_decode"] = [lambda index, budget_mib: budgets.append((index, budget_mib))]
    world.fail_step[("hi3d_d2", "I", 1)] = [oom()]
    world.run(requests("I"), hi3d=True, paired=False)
    assert world.states() == {"I": "done"}, world.results
    normal = world.limit - hi3d.DECODE_MARGIN_MIB
    assert budgets == [(0, normal), (1, normal), (1, normal * hi3d.TIGHT_SHARE)], budgets


@test
def a_hi3d_encode_out_of_memory_lowers_the_resolution_and_the_prop_is_told() -> None:
    """The shape encoder at the request's resolution runs out of memory: the mesh is encoded 128 lower (down to
    1024), the prop's texture made there, and the "memory_cap" note tells the artist, as Pixal3D's upsample does.
    At 1024 an out-of-memory error reaches the ladder."""
    tried: list = []

    def fail_once(resolution: int, voxels: int) -> None:
        tried.append(resolution)
        if len(tried) == 1:
            raise oom()

    world = World()
    world.taps["hi3d_encode_mesh"] = [fail_once]
    world.run([request("H", resolution=1536)], hi3d=True, paired=False)
    assert world.states() == {"H": "done"}, world.results
    assert tried == [1536, 1408], tried
    assert int(world.puts["H"]["coords_hr"]["res"][0]) == 1408
    assert world.puts["H"]["notes"]["memory_cap"] == {"by": "card", "resolution": 1408, "without": 1536,
                                                     "tokens": world.puts["H"]["notes"]["memory_cap"]["tokens"]}
    assert world.finished["H"][0].resolution == 1408
    tried.clear()
    world = World()
    world.taps["hi3d_encode_mesh"] = [lambda resolution, voxels: (_ for _ in ()).throw(oom())]
    world.run([request("H", resolution=1024)], hi3d=True, paired=False, spill=False)
    assert world.states() == {"H": "failed"} and world.results["H"]["error"]["kind"] == "oom", world.results


@test
def without_hi3d_files_only_hi3d_props_fail() -> None:
    world = World()
    world.run(requests("HA"))
    assert world.states() == {"H": "failed", "A": "done"}, world.results
    assert world.results["H"]["error"]["kind"] == "missing_file"
    assert not [part for part, prop in world.starts() if prop == "H"]


@test
def a_hi3d_prop_frees_its_items_as_their_last_reader_ends() -> None:
    world = World()
    world.run(requests("H"), hi3d=True, paired=False)
    discarded = [entry[3] for entry in world.entries("store", "discard") if entry[2] == "H"]
    for key in ("hi3d_ss64", "cutout", "hi3d_cond", "hi3d_coords", "hi3d_lat", "hi3d_frames", "hi3d_rng",
                "rng_ss", "coords_ss", "dino1024", "cond_tex", "rng_hr", "hi3d_mesh", "hi3d_subs"):
        assert key in discarded, (key, discarded)
    assert discarded.count("hi3d_lat") == 3, discarded
    assert "dino512" not in [entry[3] for entry in world.entries("store", "put_device") if entry[2] == "H"]


@test
def random_numbers_belong_to_the_prop() -> None:
    import torch

    # a second attempt after the draw of lr, hr, tex and ss draws the same noise
    for part, stage in (("ss", "ss"), ("lr", "shape"), ("hr", "shape"), ("tex", "tex")):
        world = World()
        world.fail_step[(part, "A", 9)] = [oom()]
        world.fail_step[(part, "C", 2)] = [oom(), oom()]
        world.run(requests("ACE"), spill=True)
        assert world.states() == dict.fromkeys("ACE", "done"), (part, world.results)
        expect_alone(world, "ACE")
        assert world.results["C"]["result"]["slow"] == [part]

    # the stored states are the reference's: seed, then the draws in order, each with the reference's shape
    world = World()
    world.run(requests("AC"))
    for prop in "AC":
        stored = world.puts[prop]
        n0, n = stored["coords_ss"]["coords"].shape[0], stored["coords_hr"]["coords"].shape[0]
        torch.manual_seed(request(prop).seed)
        torch.randn(1, 8, 16, 16, 16)
        assert torch.equal(torch.get_rng_state(), stored["rng_ss"]["state"]), prop
        torch.randn(n0, 32)
        assert torch.equal(torch.get_rng_state(), stored["rng_lr"]["state"]), prop
        torch.randn(n, 32)
        assert torch.equal(torch.get_rng_state(), stored["rng_hr"]["state"]), prop
        assert stored["rng_ss"]["state"].dtype == torch.uint8 and stored["rng_ss"]["state"].shape == (5056,)

    # somebody else drawing from the generator between the parts changes nothing ...
    world = World()
    for name in ("prepare", "camera", "bundle", "encode", "cond_tokens", "upsample", "decode_shape", "finish"):
        world.taps[name] = [lambda **info: torch.randn(3)]
    world.run(requests("ACE"))
    expect_alone(world, "ACE")

    # ... and with check_rng a part that must not draw and does, fails its prop
    world = World()
    world.taps["cond_tokens"] = [lambda spec, **info: torch.randn(3) if world.now == ("cond_hr", "E") else None]
    world.run(requests("ACE"), check_rng=True)
    assert world.states() == {"A": "done", "C": "done", "E": "failed"}, world.results
    assert "random generator" in world.results["E"]["error"]["detail"]
    expect_alone(world, "AC")
    world = World()
    world.run(requests("ACE"), check_rng=True)
    assert world.states() == dict.fromkeys("ACE", "done")


@test
def the_guard_on_sample_sparse_structure_applies_and_each_prop_keeps_its_notes() -> None:
    import torch

    world = World()
    world.redraws = {"A": 2, "C": 1}
    world.run(requests("ACE"))
    assert world.states() == dict.fromkeys("ACE", "done")
    draws = {prop: [entry[4]["steps"] for entry in world.entries("sample", "ss") if entry[3] == ("ss", prop)]
             for prop in "ACE"}
    assert draws == {"A": [12, 12, 25], "C": [12, 12], "E": [12]}, draws
    third = [entry[4] for entry in world.entries("sample", "ss") if entry[3] == ("ss", "A")][2]
    assert third["guidance_strength"] == 5.0 and third["guidance_rescale"] == 0.7 and third["rescale_t"] == 5.0
    assert {prop: world.finished[prop][1]["ss_retries"] for prop in "ACE"} == {"A": 2, "C": 1, "E": 0}
    assert {prop: world.results[prop]["result"]["ss_retries"] for prop in "ACE"} == {"A": 2, "C": 1, "E": 0}
    # the state stored after `ss` is the one after ALL draws
    torch.manual_seed(request("A").seed)
    for _ in range(3):
        torch.randn(1, 8, 16, 16, 16)
    assert torch.equal(torch.get_rng_state(), world.puts["A"]["rng_ss"]["state"])
    expect_alone(world, "E")
    assert differing(world.signature("A"), alone("A")), "two rejected draws must change the prop"

    # outside part `ss` the attribute cannot be called
    engine = World().build(guard=False)
    try:
        engine.sample_sparse_structure({}, 32)
    except RuntimeError:
        pass
    else:
        raise AssertionError("sample_sparse_structure ran outside a batch")


@test
def events_tell_the_whole_story() -> None:
    PARTS = pixal3d_parts()

    world = World()
    world.fail_step[("lr", "B", 4)] = [RuntimeError("boom")]
    world.run(requests("ABC"), wave_size=2, paired=False)       # waves: C A | B
    open_part, open_prop, steps = None, None, 0
    seen: dict[str, list[str]] = {}
    for kind, part, prop, step, total, detail in world.events():
        if kind == "part_start":
            assert open_part is None and open_prop is None
            open_part = part
        elif kind == "part_end":
            assert open_part == part and open_prop is None
            open_part = None
        elif kind == "prop_start":
            assert open_part == part and open_prop is None and 1 <= detail["position"] <= detail["of"]
            open_prop, steps = prop, 0
            seen.setdefault(prop, []).append(part)
        elif kind == "step":
            assert (part, prop) == (open_part, open_prop) and step == steps + 1 and step <= total, (part, step)
            assert total == (3 if part == "finish" else 12)
            steps = step
        elif kind in ("prop_rest", "prop_done", "prop_failed"):
            assert (part, prop) == (open_part, open_prop)
            assert (kind == "prop_done") == (part == "finish" and kind != "prop_failed")
            if part in ("ss", "lr", "hr", "tex") and kind == "prop_rest":
                assert steps == 12
            if kind == "prop_done":
                assert steps == 3 and detail["result"] is world.results[prop]["result"]
                assert set(detail["result"]) >= {"slow", "seconds", "resolution", "resumed", "mode", "views",
                                                 "warning", "ss_retries"}
                assert detail["seconds"] == detail["result"]["seconds"] > 0
            open_prop = None
        elif kind in ("net_load", "net_ready", "net_drop"):
            assert detail["name"] and part == (open_part or part)
        else:
            raise AssertionError(f"an event of an unknown kind: {kind}")
    assert open_part is None and open_prop is None
    assert seen["A"] == list(PARTS) and seen["C"] == [part for part in PARTS if part != "camera"]
    assert seen["B"] == list(PARTS[:PARTS.index("lr") + 1])
    loads = [(part, detail["name"]) for kind, part, _, _, _, detail in world.events("net_ready")]
    assert ("ss", "ss_flow.mv") in loads and ("finish", "tex_dec") in loads and ("cutout", "rembg") in loads


@test
def the_store_holds_only_what_a_later_part_reads() -> None:
    world = World()
    held: dict[tuple, list[str]] = {}
    world.taps["finish"] = [lambda prop, mesh: held.__setitem__(("finish", prop), sorted(world.store.props[prop]))]
    world.taps["cond_tokens"] = [lambda spec, **info: held.setdefault(world.now, sorted(
        world.store.props[world.now[1]]))]
    world.run(requests("AC", request_hash="h"))
    assert held[("finish", "A")] == held[("finish", "C")] == ["coords_hr", "notes", "request", "shape_slat",
                                                              "tex_slat"], held
    assert held[("cond_lr", "A")] == ["coords_ss", "dino1024", "dino512", "image_0", "notes", "prep", "request",
                                      "rng_ss"], held[("cond_lr", "A")]
    assert held[("cond_hr", "C")] == ["bundle", "coords_hr", "dino1024", "image_0", "image_1", "image_2", "image_3",
                                      "notes", "prep", "request", "rng_lr"], held[("cond_hr", "C")]
    assert ("cond_hr", "bundle") in [(entry[-1][0], entry[3]) for entry in world.entries("store", "discard", "C")]
    thrown = [(entry[-1][0], entry[3]) for entry in world.entries("store", "discard", "A")]
    assert thrown == [("ss", "cond_ss"), ("cond_lr", "dino512"), ("lr", "cond_lr"), ("lr", "rng_ss"),
                      ("upsample", "coords_ss"), ("upsample", "lr_slat"), ("cond_hr", "dino1024"),
                      ("cond_hr", "prep"), ("cond_hr", "image_0"), ("hr", "cond_hr"), ("hr", "rng_lr"),
                      ("tex", "cond_tex"), ("tex", "rng_hr")], thrown
    assert world.puts["A"]["request"] == {"id": "A", "seed": 11, "resolution": 1024, "multi": False,
                                          "sampler_params": {"ss": {}, "shape": {}, "tex": {}},
                                          "max_num_tokens": 49152, "request_hash": "h"}
    # every conditioning is stored as the denoisers cast it; latents, coords and features keep their form
    import torch

    for prop in "AC":
        stored = world.puts[prop]
        for key in ("cond_ss", "cond_lr", "cond_hr", "cond_tex"):
            assert {value.dtype for value in stored[key].values()} == {torch.bfloat16}, (prop, key)
            assert sorted(stored[key]) == ["global", "proj"]
        tokens, low = stored["coords_hr"]["coords"].shape[0], stored["coords_ss"]["coords"].shape[0]
        assert stored["cond_lr"]["proj"].shape[0] == low and stored["cond_hr"]["proj"].shape[0] == tokens
        assert stored["cond_tex"]["proj"].shape[0] == tokens and stored["cond_ss"]["proj"].shape[:2] == (1, 4096)
        for key in ("coords_ss", "coords_hr"):
            coords = stored[key]["coords"]
            assert coords.dtype == torch.int32 and coords.shape[1] == 4 and coords.is_contiguous()
            assert coords.device.type == "cpu"
        assert stored["coords_hr"]["res"].dtype == torch.int32 and stored["coords_hr"]["res"].shape == (1,)
        for key, rows in (("lr_slat", low), ("shape_slat", tokens), ("tex_slat", tokens)):
            assert stored[key]["feats"].dtype == torch.float32 and stored[key]["feats"].shape == (rows, 32)
        assert stored["dino512"]["z"].dtype == stored["dino1024"]["z"].dtype == torch.float32
        assert stored["dino512"]["z"].shape[0] == stored["dino1024"]["z"].shape[0] == (4 if prop == "C" else 1)
        mesh = world.finished[prop][0]
        assert {mesh.vertices.device.type, mesh.faces.device.type, mesh.coords.device.type,
                mesh.attrs.device.type} == {"cpu"}
        assert mesh.faces.dtype == torch.int32 and mesh.coords.dtype == torch.int32 and mesh.coords.shape[1] == 3
        assert mesh.attrs.shape[1] == 6 and mesh.attr_layout["alpha"] == slice(5, 6) and mesh.resolution == 1024
    assert sorted(world.puts["A"]["prep"]["camera"]) == ["camera_angle_x", "distance", "mesh_scale"]
    assert world.puts["C"]["prep"] == {"meta": {"azimuths": [0, 90, 180, 270]}, "camera": None}
    # DINOv3 runs once per input size and prop; a multi-view prop's bundle is built once, by `features`, and kept
    # in the store until `cond_hr` has read it
    world = World()
    calls: list[tuple] = []
    world.taps["encode"] = [lambda size, views: calls.append((world.now, size, views))]
    world.taps["bundle"] = [lambda prop: calls.append((world.now, "bundle"))]
    world.run(requests("AC"))
    assert calls == [(("features", "C"), "bundle"), (("features", "C"), 512, 4), (("features", "C"), 1024, 4),
                     (("features", "A"), 512, 1), (("features", "A"), 1024, 1)], calls
    kept = [(entry[-1][0], entry[3]) for entry in world.entries("store", "put_small", "C") if entry[3] == "bundle"]
    assert kept == [("features", "bundle")], kept
    expect_alone(world, "AC")
    # where the RAM has no room for it beside the floor, each part builds it again: the same tensors
    from pixal3d.staged import parts

    available = parts.ram_available_mib
    parts.ram_available_mib = lambda: parts.BUNDLE_RAM_MIB
    try:
        world = World()
        calls.clear()
        world.taps["bundle"] = [lambda prop: calls.append((world.now, "bundle"))]
        world.run(requests("AC"))
        assert calls == [(("features", "C"), "bundle"), (("cond_lr", "C"), "bundle"), (("cond_hr", "C"), "bundle")]
        assert not world.entries("store", "put_small", "C", "bundle") and not [
            entry for entry in world.entries("store", "discard", "C") if entry[3] == "bundle"]
        expect_alone(world, "AC")
    finally:
        parts.ram_available_mib = available


@test
def the_next_props_inputs_are_asked_for_ahead_of_time() -> None:
    world = World()
    world.run(requests("ABE"))
    log = world.log
    for part, key in (("ss", "cond_ss"), ("cond_lr", "dino512"), ("lr", "cond_lr"), ("cond_hr", "dino1024"),
                      ("hr", "cond_hr"), ("tex", "cond_tex")):
        wants = [(index, entry[2]) for index, entry in enumerate(log) if entry[:2] == ("store", "want")
                 and entry[3] == key]
        order = ["B", "A", "E"] if part == "tex" else ["A", "B", "E"]      # the pair: biggest shape (B) first
        assert [prop for _, prop in wants] == order, (part, wants)
        start = {prop: next(index for index, entry in enumerate(log)
                            if entry[:4] == ("event", "prop_start", part, prop)) for prop in "ABE"}
        one, two, _ = order
        assert wants[0][0] < start[one] and wants[1][0] < start[one] and start[one] < wants[2][0] < start[two]
        takes = [entry[2] for entry in world.entries("store", "take") if entry[3] == key]
        assert takes == order, (part, takes)
    # the samplers pump the store at every step, so that the upload rides behind the running prop
    first = next(index for index, entry in enumerate(log) if entry[:4] == ("event", "prop_start", "hr", "A"))
    last = next(index for index, entry in enumerate(log) if entry[:4] == ("event", "prop_rest", "hr", "A"))
    assert [entry[2] for entry in log[first:last] if entry[:2] == ("store", "pump")] == [True] * 12


@test
def the_token_cap_lowers_the_resolution_of_a_1536_prop() -> None:
    def made(max_num_tokens: int, name: str = "B") -> tuple[World, int, int, list]:
        world = World()
        grids: list[tuple] = []
        world.taps["cond_tokens"] = [lambda spec, grid, tokens, size, **more: grids.append((spec, grid, tokens, size))]
        world.run([request(name, max_num_tokens=max_num_tokens)])
        assert world.states() == {name: "done"}, world.results
        stored = world.puts[name]["coords_hr"]
        res, tokens = int(stored["res"][0]), stored["coords"].shape[0]
        assert world.results[name]["result"]["resolution"] == res == world.finished[name][0].resolution
        low = world.puts[name]["coords_ss"]["coords"].shape[0]
        assert grids == [("shape_512", 32, low, 512), ("shape_1024", res // 16, tokens, 1024),
                         ("tex_1024", res // 16, tokens, 1024)], grids
        return world, res, tokens, grids

    full, res, most, _ = made(49152)
    assert res == 1536, res
    # one token fewer than the prop has at 1536: the first lower resolution with fewer tokens is taken
    seen = {1536: most}
    world, res, tokens, _ = made(most)
    assert res < 1536 and tokens < most, (res, tokens, most)
    seen[res] = tokens
    assert differing(world.signature("B"), full.signature("B"))
    # a cap nothing fits under ends at 1024, where the prop is made whatever its token count
    world, res, tokens, _ = made(10)
    assert res == 1024 and tokens >= 10
    seen[res] = tokens
    by_resolution = [seen[key] for key in sorted(seen, reverse=True)]
    assert len(seen) == 3 and by_resolution == sorted(by_resolution, reverse=True), seen
    # a 1024 prop is never lowered
    world, res, tokens, _ = made(10, "A")
    assert res == 1024 and tokens >= 10

    # the up-sampling runs the shape decoder in its memory-saving form; the decode only where it needs it (the
    # fake shapes are tiny: not here; the_decoders_give_their_cache_back_only_where_the_decode_needs_it)
    lean: list[bool] = []
    world = World()
    world.taps["upsample"] = [lambda **info: lean.append(info["lean"])]
    world.taps["decode_shape"] = [lambda **info: lean.append(info["lean"])]
    world.run(requests("A"))
    assert lean == [True, False], lean


@test
def the_arithmetic_between_store_and_samplers_is_the_reference_one() -> None:
    import torch

    world = World()
    given: dict[tuple, dict] = {}
    world.taps["cond_tokens"] = [lambda spec, z, coords, **more: given.__setitem__(
        (world.now[1], spec), {"z": z.clone(), "coords": coords.clone()})]
    world.taps["cond_dense"] = [lambda spec, z: given.__setitem__((world.now[1], spec), {"z": z.clone()})]
    world.run(requests("AC"))
    decode = sys.modules["pixal3d.pipelines.decode"]
    for prop, numbers in (("A", PIPELINE), ("C", multiview_pipeline())):
        stored = world.puts[prop]
        calls = {entry[3][0]: entry[4] for entry in world.log if entry[0] == "sample" and entry[3][1] == prop}
        coords_ss, coords_hr = stored["coords_ss"]["coords"], stored["coords_hr"]["coords"]
        low, tokens = coords_ss.shape[0], coords_hr.shape[0]

        # the noise: the reference's draws, in its order and shapes, from the prop's seed
        torch.manual_seed(request(prop).seed)
        for part, shape in (("ss", (1, 8, 16, 16, 16)), ("lr", (low, 32)), ("hr", (tokens, 32)), ("tex", (tokens, 32))):
            assert torch.equal(calls[part]["noise"], torch.randn(*shape)), (prop, part)

        # p2d:340-346: the decoder's logits above zero, pooled to the 32 grid, as rows (batch, x, y, z)
        decoded = make_net("ss_dec")(calls["ss"]["samples"]) > 0
        decoded = torch.nn.functional.max_pool3d(decoded.float(), 2, 2, 0) > 0.5
        assert torch.equal(coords_ss, torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()), prop
        assert torch.equal(calls["lr"]["coords"], coords_ss)
        assert torch.equal(calls["hr"]["coords"], coords_hr) and torch.equal(calls["tex"]["coords"], coords_hr)

        # the latents: denormalised with the shape numbers (lr, hr) and the texture numbers (tex);
        # the texture stage is given the shape latent normalised again
        mean = torch.tensor(numbers["shape_slat_normalization"]["mean"])[None]
        std = torch.tensor(numbers["shape_slat_normalization"]["std"])[None]
        tex_mean = torch.tensor(numbers["tex_slat_normalization"]["mean"])[None]
        tex_std = torch.tensor(numbers["tex_slat_normalization"]["std"])[None]
        assert torch.equal(stored["lr_slat"]["feats"], calls["lr"]["samples"] * std + mean), prop
        assert torch.equal(stored["shape_slat"]["feats"], calls["hr"]["samples"] * std + mean), prop
        assert torch.equal(calls["tex"]["concat"], (stored["shape_slat"]["feats"] - mean) / std), prop
        assert torch.equal(stored["tex_slat"]["feats"], calls["tex"]["samples"] * tex_std + tex_mean), prop
        assert calls["lr"]["concat"] is None and calls["hr"]["concat"] is None

        # p2d:711-725: the up-sampled coords quantised to the HR grid with (grid - 1), rounded, made unique
        hr_coords = decode.upsample(make_net("shape_dec"), stored["lr_slat"]["feats"], coords_ss)
        grid = int(stored["coords_hr"]["res"][0]) // 16
        quantised = torch.cat([hr_coords[:, :1], ((hr_coords[:, 1:] + 0.5) / 512 * (grid - 1)).round().int()], dim=1)
        assert grid == 64 and torch.equal(coords_hr, quantised.unique(dim=0)), prop

        # the conditioning passes: DINOv3's tokens of the stage's input size, the stage's coords
        assert torch.equal(given[(prop, "ss")]["z"], stored["dino512"]["z"])
        assert torch.equal(given[(prop, "shape_512")]["z"], stored["dino512"]["z"])
        assert torch.equal(given[(prop, "shape_512")]["coords"], coords_ss)
        for spec in ("shape_1024", "tex_1024"):
            assert torch.equal(given[(prop, spec)]["z"], stored["dino1024"]["z"]), (prop, spec)
            assert torch.equal(given[(prop, spec)]["coords"], coords_hr), (prop, spec)


@test
def sampler_settings_and_calls_are_the_reference_ones() -> None:
    world = World()
    world.run(requests("AC"))
    calls = {(entry[3][0], entry[3][1]): (entry[1], entry[2], entry[4]) for entry in world.log
             if entry[0] == "sample"}
    for prop, kind in (("A", "sv"), ("C", "mv")):
        multi = kind == "mv"
        for part, stage, strength, rescale, t, negative in (
                ("ss", "ss", 7.5, 0.6 if multi else 0.7, 5.0, True),
                ("lr", "shape", 7.5, 0.5, 3.5 if multi else 3.0, True),
                ("hr", "shape", 7.5, 0.5, 3.5 if multi else 3.0, True),
                ("tex", "tex", 1.0, 0.0, 2.5 if multi else 3.0, False)):
            got_stage, network, settings = calls[(part, prop)]
            assert got_stage == stage and network == f"{part}_flow.{kind}", (part, prop, got_stage, network)
            assert settings["steps"] == 12 and settings["guidance_strength"] == strength
            assert settings["guidance_rescale"] == rescale and settings["rescale_t"] == t
            assert settings["guidance_interval"] == ((0.6, 0.9) if part == "tex" else (0.6, 1.0))
            assert settings["negative"] is negative, (part, settings)      # no zero conditioning nobody reads
            assert settings["verbose"] is False and settings["keep_trajectory"] is False
            assert settings["dtype"] == settings["global_dtype"] == "torch.bfloat16", settings

    # a request's settings are laid over the pipeline file's, stage by stage
    world = World()
    world.run([request("A", sampler_params={"ss": {"steps": 5}, "shape": {"steps": 7, "guidance_strength": 2.0},
                                            "tex": {"guidance_strength": 3.0, "rescale_t": 4.0}})])
    calls = {entry[3][0]: entry[4] for entry in world.log if entry[0] == "sample"}
    assert calls["ss"]["steps"] == 5 and calls["ss"]["guidance_strength"] == 7.5 and calls["ss"]["rescale_t"] == 5.0
    assert calls["lr"]["steps"] == calls["hr"]["steps"] == 7 and calls["hr"]["guidance_strength"] == 2.0
    assert calls["hr"]["guidance_rescale"] == 0.5
    assert calls["tex"]["steps"] == 12 and calls["tex"]["negative"] is True and calls["tex"]["rescale_t"] == 4.0
    totals = {part: total for kind, part, _, _, total, _ in world.events("step")}
    assert totals == {"ss": 5, "lr": 7, "hr": 7, "tex": 12, "finish": 3}, totals
    # a request may leave a stage out
    world = World()
    world.run([request("A", sampler_params={})])
    expect_alone(world, "A")


@test
def the_engine_tells_loader_and_store_what_the_specification_says() -> None:
    world = World()
    engine = world.build()
    loader = world.loader
    assert loader.made == {"device": "cpu", "prefetch": "cache", "max_vram_mib": None}
    folder = world.pipeline_dir
    assert loader.checkpoints == {
        "ss_flow.sv": folder / "ckpts/ss_flow", "lr_flow.sv": folder / "ckpts/lr_flow",
        "hr_flow.sv": folder / "ckpts/hr_flow", "tex_flow.sv": folder / "ckpts/tex_flow",
        "ss_flow.mv": folder / "ckpts/ss_flow_mv", "lr_flow.mv": folder / "ckpts/lr_flow_mv",
        "hr_flow.mv": folder / "ckpts/hr_flow_mv", "tex_flow.mv": folder / "ckpts/tex_flow_mv",
        "ss_dec": folder / "ckpts/ss_dec", "shape_dec": folder / "ckpts/shape_dec",
        "tex_dec": folder / "ckpts/tex_dec"}, loader.checkpoints
    assert sorted(loader.builders) == ["dino", "moge", "naf", "rembg"]
    assert not loader.modules and world.store is None, "building the engine loads nothing"
    assert sorted(engine.pipeline_args) == ["mv", "sv"]
    assert engine.pipeline_args["sv"].sampler_params["tex"]["guidance_interval"] == [0.6, 0.9]
    assert engine.pipeline_args["sv"].shape_normalization == PIPELINE["shape_slat_normalization"]

    results = engine.run(requests("A"), world.work, on_event=world.on_event, cancelled=lambda: False)
    assert results["A"]["state"] == "done"
    assert world.store.made == {"work_dir": world.work, "device": "cpu", "ram_mib": None, "slots": 2,
                                "slot_mib": None}, world.store.made
    first = world.store
    results = engine.run(requests("C"), world.work, on_event=world.on_event, cancelled=lambda: False)
    assert results["C"]["state"] == "done" and world.store is not first and first.closed and world.store.closed

    try:
        engine.run([request("A"), request("A")], world.work, on_event=world.on_event,
                   cancelled=lambda: False)
    except ValueError:
        pass
    else:
        raise AssertionError("two props with one id were accepted")
    assert engine._store is None, "the engine is free again after a refused batch"
    loader.load("dino")
    loader.set_cap()
    world.log.clear()
    engine.release()
    assert not loader.modules and not loader.capped() and world.entries("prefetch", None)
    engine.close()
    assert loader.closed
    world = World()
    world.build(store_ram_mib=64)
    world.engine.run(requests("A"), world.work, on_event=world.on_event, cancelled=lambda: False)
    assert world.store.made["ram_mib"] == 64

    # the weights files of the real builders are named for the loader's read-ahead
    from pixal3d.staged import builders
    from pixal3d.staged.engine import _builder_file

    paths = builders.ModelPaths(rembg=str(SCRATCH / "m/RMBG"), moge=str(SCRATCH / "m/moge.pt"),
                                dino=str(SCRATCH / "m/dino"), naf_repo=str(SCRATCH),
                                naf_weights=str(SCRATCH / "m/n.pth"))
    for file in ("m/RMBG/model.safetensors", "m/moge.pt", "m/dino/model.safetensors", "m/n.pth"):
        (SCRATCH / file).parent.mkdir(parents=True, exist_ok=True)
        (SCRATCH / file).write_bytes(b"")
    found = {name: _builder_file(name, build) for name, build in builders.default_builders(paths).items()}
    assert found == {"rembg": SCRATCH / "m/RMBG/model.safetensors", "moge": SCRATCH / "m/moge.pt",
                     "dino": SCRATCH / "m/dino/model.safetensors", "naf": SCRATCH / "m/n.pth"}, found
    assert _builder_file("dino", lambda device: None) is None


@test
def fp16_accumulation_is_off_by_default_and_only_on_the_ampere_profile() -> None:
    """PIXAL3D_FP16ACC (modules/fp16acc.py, an opt-in precision change): unset or off, the engine's convert hook
    hands a flow back untouched (the same modules: the bf16 layers' bits); on, it converts nothing on the turing and
    pascal profiles, nothing for the hr and tex flows and nothing off the card. The fp16 layer keeps the bf16 weight
    to 2^-20 of each row, the rows SageAttention's fold puts 2^-8 down shifted back up by a power of two, and answers
    in the layer's dtype."""
    import torch
    from pixal3d.modules import fp16acc, precision
    from pixal3d.modules.attention import MultiHeadAttention
    from pixal3d.modules.transformer.blocks import FeedForwardNet

    class Flow(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = MultiHeadAttention(64, num_heads=4)
            self.mlp = FeedForwardNet(64, mlp_ratio=2.0)

    def layers(model: torch.nn.Module) -> list:
        return [(name, type(module).__name__) for name, module in model.named_modules()]

    names = ("PIXAL3D_FP16ACC", "PIXAL3D_PRECISION", "PIXAL3D_CARD_SM")
    saved = {name: os.environ.get(name) for name in names}
    try:
        flow = Flow().to(torch.bfloat16)
        before = layers(flow)
        for value in (None, "", "off", "0"):
            if value is None:
                os.environ.pop("PIXAL3D_FP16ACC", None)
            else:
                os.environ["PIXAL3D_FP16ACC"] = value
            precision.reset()
            assert not fp16acc.requested() and not fp16acc.allowed()[0]
            for kind in ("ss", "lr"):
                assert fp16acc.convert(flow, kind) is flow and layers(flow) == before
        os.environ["PIXAL3D_FP16ACC"] = "on"
        for name in ("turing", "pascal"):
            os.environ["PIXAL3D_PRECISION"] = name
            precision.reset()
            assert not fp16acc.allowed()[0], name
            assert fp16acc.convert(flow, "ss") is flow and layers(flow) == before
        os.environ["PIXAL3D_PRECISION"] = "ampere"
        precision.reset()
        assert fp16acc.allowed()[0], fp16acc.allowed()
        for kind in ("hr", "tex", "", "ss", "lr"):          # ss / lr: the weights are on the CPU here
            assert fp16acc.convert(flow, kind) is flow and layers(flow) == before

        torch.manual_seed(3)
        source = torch.nn.Linear(64, 96).to(torch.bfloat16)
        with torch.no_grad():
            source.weight[64:] *= 2.0 ** -8                  # the fold's v rows
        layer = fp16acc.Fp16AccLinear(source)
        want = source.weight.float()
        err = (layer.weight_as(torch.float32) - want).norm(dim=1) / want.norm(dim=1)
        assert bool((err <= fp16acc.EXACT).all()) and layer.shift_max > 0 and layer.shifted_rows <= 32
        assert layer.weight16.dtype == torch.float16 and layer.unscale is not None
        x = torch.randn(5, 64).to(torch.bfloat16)
        assert layer(x).dtype == torch.bfloat16 and layer(x).shape == (5, 96)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        precision.reset()


@test
def the_loader_reads_a_checkpoint_with_several_readers_into_the_same_tensors() -> None:
    """(Last of the tests: what it imports - accelerate, safetensors - would slow every later collection,
    and the timed tests with it.) loader._read_parallel (the way onto the card; here forced onto the CPU): READERS threads read the file in
    ranges into slots and the copies are queued, and every tensor gets the bytes, the dtype and the shape the
    one-buffer way gives it - the copy and the original, tensors cast on the device, tensors larger than the
    buffer in pieces, ranges that end inside an element. The cancel is asked once per tensor; a copy that breaks
    while it is read gives way to the original; no reader thread outlives its load; close() frees the slots."""
    import json as json_module

    import torch
    import torch.nn as nn
    from safetensors.torch import save_file
    from pixal3d.staged import loader as L, stfile, weights as W
    from pixal3d.staged.types import Cancelled

    class Block(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.mix = nn.Linear(width, width)
            self.norm = nn.LayerNorm(width)

    class Tiny(nn.Module):
        """A bf16 torso, fp32 ends and a complex buffer, as the flow models have."""

        def __init__(self, width: int = 24, depth: int = 3):
            super().__init__()
            self.first = nn.Linear(8, width)
            angles = torch.arange(24, dtype=torch.float32).reshape(4, 6)
            self.register_buffer("phases", torch.polar(torch.ones_like(angles), angles))
            self.blocks = nn.ModuleList([Block(width) for _ in range(depth)])
            for sub in self.blocks.modules():
                if isinstance(sub, nn.Linear):
                    for parameter in sub.parameters():
                        parameter.data = parameter.data.to(torch.bfloat16)

    folder = SCRATCH / "parallel-loader"
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(5)
    state = {key: (value if value.is_complex() else torch.randn(value.shape) * 0.1 + 1e-3)
             for key, value in Tiny().state_dict().items()}
    state["unused.weight"] = torch.randn(4)
    save_file(state, str(folder / "tiny.safetensors"))
    (folder / "tiny.json").write_text(json_module.dumps({"name": "Tiny", "args": {"width": 24, "depth": 3}}))
    ckpt = folder / "tiny"

    saved = (L._model_class, L.BUFFER_BYTES, L.CHUNK_BYTES, L.POOL_BYTES, L.PARALLEL_ON_CPU, L.READERS, stfile._fill)
    L._model_class = {"Tiny": Tiny}.__getitem__
    L.BUFFER_BYTES, L.CHUNK_BYTES, L.POOL_BYTES, L.PARALLEL_ON_CPU = 2048, 1003, 4 * 1003, True
    L._staging.clear()
    events: list[dict] = []
    loader = L.NetworkLoader("cpu", prefetch="off", on_event=events.append)
    loader.register_checkpoint("tiny", ckpt)

    def load(readers: int, cancelled: Any = None) -> tuple[dict, list[dict]]:
        L.READERS = readers
        events.clear()
        module = loader.load("tiny", cancelled)
        got = {name: value.clone() for name, value in module.state_dict().items()}
        del module
        loader.drop("tiny")
        return got, [event for event in events if event["kind"] != "net_drop"]

    def same(got: dict, want: dict) -> bool:
        return list(got) == list(want) and all(
            got[name].dtype == want[name].dtype and got[name].shape == want[name].shape
            and torch.equal(got[name].reshape(-1).view(torch.uint8), want[name].reshape(-1).view(torch.uint8))
            for name in want)

    def readers_left() -> list:
        return [thread for thread in threading.enumerate() if thread.name == "p3d-net-read" and thread.is_alive()]

    try:
        header = stfile.read_header(folder / "tiny.safetensors")
        assert any(info.nbytes > L.BUFFER_BYTES for info in header.values())        # pieces in a block of their own
        serial, told_serial = load(1)
        parallel, told = load(3)
        assert same(parallel, serial) and parallel["blocks.0.mix.weight"].dtype == torch.bfloat16
        assert parallel["phases"].dtype == torch.complex64 and "unused.weight" not in parallel
        assert [(event["kind"], event["bytes"]) for event in told] == \
            [(event["kind"], event["bytes"]) for event in told_serial], (told, told_serial)
        assert not readers_left() and loader.stats()["pinned_bytes"] == 0

        W.make_copy(ckpt)                                   # the copy: every tensor in the model's dtype already
        copied, told = load(3)
        assert same(copied, serial) and told[0]["bytes"] < told_serial[0]["bytes"]

        real_fill, calls = stfile._fill, []

        def breaking(handle: Any, view: Any, path: Any) -> None:
            if W.COPY_SUFFIX in str(path) and threading.current_thread().name == "p3d-net-read":
                calls.append(1)
                if len(calls) > 2:
                    raise OSError(22, "a bad sector")
            real_fill(handle, view, path)

        stfile._fill = breaking
        broke, told = load(3)
        stfile._fill = real_fill
        assert same(broke, serial) and len(calls) > 2, len(calls)
        assert [event["kind"] for event in told if event["kind"] == "net_ready" or event["bytes_read"] == 0] == \
            ["net_load", "net_load", "net_ready"], told
        assert not readers_left()

        asked: list[int] = []

        def after_three() -> bool:
            asked.append(1)
            return len(asked) > 3

        try:
            load(3, after_three)
        except Cancelled:
            pass
        else:
            raise AssertionError("the load was not cancelled")
        assert len(asked) == 4 and loader.loaded() == [] and not readers_left()
        assert same(load(3)[0], serial)
        loader.close()
        assert loader._pool is None and loader.stats()["pinned_bytes"] == 0
    finally:
        (L._model_class, L.BUFFER_BYTES, L.CHUNK_BYTES, L.POOL_BYTES, L.PARALLEL_ON_CPU, L.READERS,
         stfile._fill) = saved
        L._staging.clear()
        loader = None
        gc.collect()        # this test's models and tensors: not left to the first collection of a timed test


def prepare_fake(repo: Path, scratch: Path, real_store: bool = False, real_loader: bool = False) -> None:
    """This process made ready for the fake tests: the worktree as working folder and first import path, the
    stand-in modules in place, `scratch` an empty folder for the tests' files. Nothing here may touch the card."""
    global SCRATCH, REAL_STORE, REAL_LOADER
    REAL_STORE, REAL_LOADER = real_store, real_loader
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    SCRATCH = scratch
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    install_fakes()
    # The engine's RAM rules read the PC's free memory: on a PC where other programs hold most of it (the review of
    # round J: 2.5 GB free, 9 of 41 tests failed) the fakes would plan around a shortage the tests never meant.
    # Every test sees a PC with plenty, unless it sets its own figure (and puts this one back): run_fake and the
    # hub's pytest wrapper (code/tests/test_pixal3d_engine.py) alike.
    from pixal3d.staged import parts as staged_parts

    staged_parts.ram_available_mib = lambda: 60_000


def run_fake(args: Any) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"           # nothing here may touch the card
    repo = Path(args.repo)
    if not (repo / "pixal3d" / "staged" / "engine.py").is_file():
        print(f"{repo} has no pixal3d/staged/engine.py")
        return 1
    scratch = repo / "tmp" / "p3d-checks" / f"engine-fake-{os.getpid()}"
    try:
        import torch

        prepare_fake(repo, scratch, args.real_store, args.real_loader)       # (with the RAM figure: prepare_fake)
        results = []
        chosen = [name for name in TESTS if not args.only or args.only in name]
        for name in chosen:
            log = contextlib.redirect_stdout(open(os.devnull, "w")) if not args.verbose else contextlib.nullcontext()
            try:
                with log:
                    TESTS[name]()
                results.append(common.Result(name.replace("_", " "), True, "fake"))
            except Exception:
                trace = traceback.format_exc()
                last = [line for line in trace.strip().splitlines() if line.strip()][-1]
                results.append(common.Result(name.replace("_", " "), False, "fake", last))
                print(trace, flush=True)
        assert not torch.cuda.is_initialized(), "the fake tests initialised CUDA"
        for result in results:
            print(f"{'PASS' if result.ok else 'FAIL'}  {result.name}" + (f"\n      {result.detail}" if result.detail
                                                                        else ""), flush=True)
        failed = [result for result in results if not result.ok]
        print(f"{'FAILED' if failed or not results else 'PASSED'}: {len(results) - len(failed)} of {len(results)} "
              f"tests of the engine's logic on fakes" + (", with the worktree's own PropStore" if REAL_STORE else "")
              + (", with the worktree's own NetworkLoader" if REAL_LOADER else "") + ".", flush=True)
        return 1 if failed or not results else 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# =====================================================================================================================
# The golden mode: the engine with the worktree's own loader, store, conditioning and decoding code, on the card
# =====================================================================================================================

# What a part stores for a prop -> the golden tensor it must equal: (store key, tensor, golden file, golden tensor).
# A conditioning is stored as bfloat16; the golden one (float32, what the reference computed) is cast the same way.
COMPARED = (
    ("dino512", "z", "01-dino", "z512_a"), ("dino1024", "z", "01-dino", "z1024_a"),
    ("cond_ss", "global", "02-cond_ss", "global"), ("cond_ss", "proj", "02-cond_ss", "proj"),
    ("coords_ss", "coords", "03-ss", "coords"), ("rng_ss", "state", "03-ss", "rng_after"),
    ("cond_lr", "global", "04-cond_lr", "global"), ("cond_lr", "proj", "04-cond_lr", "proj"),
    ("lr_slat", "feats", "05-lr", "slat"), ("rng_lr", "state", "05-lr", "rng_after"),
    ("coords_hr", "coords", "06-upsample", "coords"), ("coords_hr", "res", "06-upsample", "res"),
    ("cond_hr", "global", "07-cond_hr", "global"), ("cond_hr", "proj", "07-cond_hr", "proj"),
    ("shape_slat", "feats", "08-hr", "slat"), ("rng_hr", "state", "08-hr", "rng_after"),
    ("cond_tex", "global", "09-cond_tex", "global"), ("cond_tex", "proj", "09-cond_tex", "proj"),
    ("tex_slat", "feats", "10-tex", "slat"),
)


class SpyStore:
    """The worktree's PropStore with a copy kept of everything the parts put into it."""

    def __init__(self, store: Any, kept: dict[str, dict]):
        self._store, self._kept = store, kept

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def _keep(self, prop_id: str, key: str, value: Any) -> None:
        if isinstance(value, dict) and key in {entry[0] for entry in COMPARED}:
            self._kept.setdefault(prop_id, {})[key] = {name: tensor.detach().cpu().clone()
                                                       for name, tensor in value.items()}

    def put_small(self, prop_id: str, key: str, value: Any) -> None:
        self._keep(prop_id, key, value)
        self._store.put_small(prop_id, key, value)

    def put_device(self, prop_id: str, key: str, tensors: dict) -> None:
        self._keep(prop_id, key, tensors)
        self._store.put_device(prop_id, key, tensors)


class GoldenHooks:
    """Hooks that hand the engine what the reference path prepared for each golden case - the cut-out picture(s),
    the camera, the view bundle - and compare the decoded mesh instead of writing a file. A prop's id is its case."""

    def __init__(self, cases: dict[str, common.Case], notes: Any, die_in: str | None):
        self.cases, self.notes, self.die_in = cases, notes, die_in
        self.results: list[common.Result] = []
        self.draws = 0                      # draws of the sparse structure of the prop in part `ss`
        self.forced = False                 # the test guard rejects the first two draws of the prop in part `ss`

    def prepare(self, req: Any, matte: Any) -> Any:
        from PIL import Image
        from pixal3d.staged import Prepared

        case = self.cases[req.id]
        if not req.multi:
            return Prepared([Image.fromarray(case.tensor("00-input", "image_u8").numpy())], {})
        count = sum(1 for name in case.names("00-input") if name.startswith("view_") and name.endswith("_u8"))
        views = [Image.fromarray(case.tensor("00-input", f"view_{index}_u8").numpy()) for index in range(count)]
        assert all(view.mode == "RGBA" for view in views)
        return Prepared(views, {"azimuths": case.tensor("00-input", "azimuths").tolist()})

    def camera(self, req: Any, image: Any, moge: Any) -> dict:
        exact = self.cases[req.id].manifest["camera_hex"]       # the three floats bit for bit
        return {name: float.fromhex(exact[name]) for name in ("camera_angle_x", "distance", "mesh_scale")}

    def bundle(self, req: Any, images: list, meta: dict) -> dict:
        case = self.cases[req.id]
        return {"images": {512: case.tensor("00-input", "images_512"), 1024: case.tensor("00-input", "images_1024")},
                "camera_angle_x": case.tensor("00-input", "camera_angle_x"),
                "camera_distance": case.tensor("00-input", "camera_distance"),
                "transform_matrix": case.tensor("00-input", "transform_matrix"),
                "mesh_scale": float(case.manifest["camera"]["mesh_scale"])}

    @contextlib.contextmanager
    def prop_scope(self, req: Any, part: str) -> Iterator[dict]:
        self.notes.reset()
        self.draws = 0
        self.forced = bool(self.cases[req.id].manifest["capture_options"].get("test_guard"))
        found: dict = {}
        try:
            yield found
        finally:
            self.forced = False
        found.update(ss_retries=self.notes.ss_retries, ss_blocked=self.notes.ss_blocked)

    def finish(self, req: Any, mesh: Any, notes: dict) -> dict:
        import torch

        case, results = self.cases[req.id], []

        def compare(name: str, got: Any) -> None:
            try:
                golden = case.tensor("11-decode", name)
            except KeyError:        # the reference never got this far (big1536: out of memory in the texture decode)
                print(f"{req.id}: 11-decode/{name} is not in the reference, not compared", flush=True)
                return
            result = common.compare(f"11-decode/{name}", got, golden, case.spread)
            result.name = f"{req.id}: {result.name}"
            results.append(result)

        if "11-decode" in case.files():
            manifest = case.manifest
            compare("vertices_raw", mesh.vertices[:int(manifest["vertices_raw"])])
            compare("faces_raw", mesh.faces[:int(manifest["faces_raw"])])
            compare("tex_coords", mesh.coords)
            compare("tex_attrs", mesh.attrs)
            for what, tensor in (("vertices", mesh.vertices), ("faces", mesh.faces)):
                if f"{what}_sha" not in manifest:
                    print(f"{req.id}: the reference has no final mesh, {what} not compared", flush=True)
                    continue
                same = tensor.shape[0] == int(manifest[what]) and common.sha256_of(tensor) == manifest[f"{what}_sha"]
                results.append(common.Result(f"{req.id}: {what} after fill_holes", same, "sha256",
                                             f"{tensor.shape[0]} rows, the reference has {manifest[what]}"))
            results.append(common.Result(f"{req.id}: resolution", mesh.resolution == int(manifest["res"]), "integers",
                                         f"{mesh.resolution}, the reference used {manifest['res']}"))
            on_cpu = {tensor.device.type for tensor in (mesh.vertices, mesh.faces, mesh.coords, mesh.attrs)} == {"cpu"}
            kept = common.kept_on_card()
            allocated = torch.cuda.memory_allocated()
            if allocated - kept > common.LIBRARY_TABLES_DURING:
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) and obj.is_cuda:
                            holders = [type(ref).__name__ for ref in gc.get_referrers(obj)][:4]
                            print(f"{req.id}: on the card at the hook: {list(obj.shape)} {obj.dtype} "
                                  f"{obj.untyped_storage().nbytes()} B, held by {holders}", flush=True)
                    except Exception:
                        pass
            empty = 0 <= allocated - kept <= common.LIBRARY_TABLES_DURING
            results.append(common.Result(f"{req.id}: mesh on the CPU, card empty", on_cpu and empty,
                                         "form", f"{allocated} B allocated when the hook starts, {kept} B kept "
                                                 "on purpose (cuBLAS's work area, the projection grids)"
                                                 + ("" if empty else f"; {common.BOUNDS_HINT}")))
        self.results += results
        if self.die_in == req.id:
            print(f"[check_engine] dying inside the finish hook of {req.id}, as asked", flush=True)
            os._exit(3)
        return {"mode": "multi" if req.multi else "single", "views": len(case.request.get("views") or []) or 1,
                "warning": None, "ss_retries": notes.get("ss_retries", 0)}


def run_golden(args: Any) -> int:
    """The engine on golden cases, alone or as one batch. Written against the contracts of loader.py, store.py,
    proj_cond.py and decode.py: it needs the branch that has them all."""
    names = [name for name in (args.batch.split(",") if args.batch else [args.case]) if name]
    if args.reverse:
        names.reverse()
    common.boot(args.repo)
    cases = {name: common.golden(name) for name in names}
    with common.gpu_lock("WP6", f"check_engine {','.join(names)}", expected_minutes=4 * len(names) + 2):
        return common.report(_golden_batch(args, cases))


def _golden_batch(args: Any, cases: dict[str, common.Case]) -> list[common.Result]:
    import time

    import torch
    import pixal3d_guard
    import pixal3d_mv
    from pixal3d.staged import BatchBroken, EngineOptions, PropRequest, StagedEngine, runner
    from pixal3d.staged.store import PropStore

    srv = common.launcher()
    names = list(cases)
    notes = srv.RunNotes()
    hooks = GoldenHooks(cases, notes, args.die_in)
    kept: dict[str, dict] = {}
    options = EngineOptions(pipeline_dir=pixal3d_mv.snapshot_dir(srv.models_folder(args.repo)),
                            builders=common.builders(args.repo), device="cuda", max_vram_mib=args.cap_mib,
                            check_rng=True)
    cold = not torch.cuda.is_initialized()
    engine = StagedEngine(options, hooks, store_factory=lambda *a, **k: SpyStore(PropStore(*a, **k), kept))
    built_cold = not cold or not torch.cuda.is_initialized()
    srv.exact_hole_centres()                # the launcher's patch of Mesh.fill_holes: a batch decodes several meshes

    # The launcher's guard around the engine's patch point. For a case captured with the test guard its verdict
    # is "the first two draws are rejected", as in the capture.
    srv.guard_sparse_structure(engine, notes)
    verdict = pixal3d_guard.ss_blocked

    def ss_blocked(score: float, deep: float) -> bool:
        hooks.draws += 1
        return hooks.draws < 3 if hooks.forced else verdict(score, deep)

    pixal3d_guard.ss_blocked = ss_blocked

    # The noise of every draw of the sparse structure (it is stored nowhere), taken where the sampler gets it.
    noises: dict[str, list] = {}
    now: dict[str, Any] = {"part": "", "prop": None}
    for pipeline_args in engine.pipeline_args.values():
        sampler = pipeline_args.samplers["ss"]

        def recorded(model: Any, noise: Any, *a: Any, _sample: Any = sampler.sample, **k: Any) -> Any:
            noises.setdefault(now["prop"], []).append(noise.detach().cpu().clone())
            return _sample(model, noise, *a, **k)

        sampler.sample = recorded

    requests = []
    for name, case in cases.items():
        manifest = case.manifest
        passed = manifest["pipeline"]["sampler_params_passed"]
        requests.append(PropRequest(
            id=name, seed=int(manifest["seed"]), resolution=int(manifest["resolution"]),
            multi=manifest["kind"] == "multi", max_num_tokens=int(manifest["pipeline"]["max_num_tokens"]),
            sampler_params={"ss": passed["sparse_structure"], "shape": passed["shape_slat"], "tex": passed["tex_slat"]},
            request_hash=f"golden-{name}"))

    # --inject part:prop:step:what[:times] - what = oom | error | cuda | cancel, raised (or asked for) at that step
    # event; "cuda" is an error whose text makes the runner probe the card
    inject: dict[tuple, list] = {}
    cancel: dict[str, Any] = {"asked": False, "at": None}
    expected = {name: ("done",) for name in names}          # the states each prop may end in
    for words in args.inject or []:
        part, prop, step, what, *times = words.split(":")
        where, count = (part, prop, int(step)), int(times[0]) if times else 1
        if what == "cancel":
            cancel["at"] = where
            expected = {name: ("done", "cancelled") for name in names}
        elif what == "oom":                 # three in a row use up the ladder
            inject[where] = [torch.OutOfMemoryError("CUDA out of memory (injected by check_engine)")
                             for _ in range(count)]
            expected[prop] = ("failed",) if count >= 3 else expected[prop]
        else:
            inject[where] = [RuntimeError("CUDA error: unspecified launch failure (injected by check_engine)"
                                          if what == "cuda" else "injected by check_engine")]
            expected[prop] = ("failed",)
    touched = {where[1] for where in inject}
    if args.cancel_at_load:
        expected = {name: ("done", "cancelled") for name in names}
    if args.probe_fails:                    # the card "does not answer" after a CUDA error: the batch must break
        runner.probe = lambda device: False

    # --bump-ooms part:prop - PyTorch's count of failed allocations moves by one during that prop-part's first
    # attempt, as it does when cuDNN swallows an out-of-memory and takes another algorithm
    counted = {"extra": 0, "armed": False, "reads": 0, "at": tuple((args.bump_ooms or ":").split(":"))}
    real_ooms = runner.num_ooms

    def num_ooms(device: str) -> int:
        if counted["armed"]:
            counted["reads"] += 1
            if counted["reads"] == 2:       # the first read is the one before the attempt
                counted["extra"], counted["armed"] = 1, False
        return real_ooms(device) + counted["extra"]

    runner.num_ooms = num_ooms
    if args.bump_ooms:
        touched.add(counted["at"][1])

    peaks: list[tuple] = []                 # (part, prop, allocated MiB, reserved MiB, how it ended)
    top = {"allocated": 0, "reserved": 0}
    spilled: list[tuple] = []
    caps: list[tuple] = []                  # (part, prop, MiB allocated when the cap was set, the cap in MiB)
    set_cap = engine.loader.set_cap

    def watched_cap(*a: Any, **k: Any) -> int:
        limit = set_cap(*a, **k)
        caps.append((now["part"], now["prop"], torch.cuda.memory_allocated() // common.MIB, limit))
        return limit

    engine.loader.set_cap = watched_cap

    def on_event(event: Any) -> None:
        if event.kind == "prop_start":
            now.update(part=event.part, prop=event.prop_id)
            torch.cuda.reset_peak_memory_stats()
            if (event.part, event.prop_id) == counted["at"] and not counted["extra"]:
                counted.update(armed=True, reads=0)
        elif event.kind == "net_load" and event.detail.get("name") == args.cancel_at_load:
            cancel["asked"] = True
        elif event.kind in ("prop_rest", "prop_done", "prop_failed"):
            allocated = torch.cuda.max_memory_allocated() // common.MIB
            reserved = torch.cuda.max_memory_reserved() // common.MIB
            peaks.append((event.part, event.prop_id, allocated, reserved, event.kind))
            top.update(allocated=max(top["allocated"], allocated), reserved=max(top["reserved"], reserved))
        elif event.kind == "prop_spilled":
            spilled.append((event.part, event.prop_id))
        elif event.kind == "step":
            where = (event.part, event.prop_id, event.step)
            if cancel["at"] == where:
                cancel["asked"] = True
            if inject.get(where):
                raise inject[where].pop(0)

    work = Path(args.work) if args.work else args.repo / "tmp" / "p3d-checks" / "engine-work"
    work.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    try:
        answers = engine.run(requests, work, on_event=on_event, cancelled=lambda: cancel["asked"],
                             wave_size=args.wave_size, resume=args.resume)
    except BatchBroken as broken:           # nothing can be asked of the card any more: no clean-up, no comparisons
        return [common.Result("the batch broke, as the failing probe says it must", bool(args.probe_fails), "form",
                              f"{broken.kind}: {broken.detail}")]
    seconds = time.perf_counter() - began
    engine.close()
    if torch.cuda.memory_allocated():
        # Name what is left: every tensor still on the card and what holds it.
        gc.collect()
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    holders = [type(ref).__name__ for ref in gc.get_referrers(obj)][:4]
                    print(f"left on the card: {list(obj.shape)} {obj.dtype} {obj.untyped_storage().nbytes()} B, "
                          f"held by {holders}", flush=True)
            except Exception:
                pass
    left = torch.cuda.memory_allocated()
    ooms = int(torch.cuda.memory_stats().get("num_ooms", 0))

    results = [common.Result("building the engine leaves CUDA alone", built_cold, "form"), *hooks.results]
    for name, case in cases.items():
        answer = answers[name]
        results.append(common.Result(f"{name}: state", answer["state"] in expected[name], "form",
                                     f"{answer['state']} {answer.get('error') or ''}".strip()))
        files = set(case.files())
        resumed = bool(answer.get("result", {}).get("resumed"))
        for key, tensor, file, golden_name in COMPARED:
            if resumed or key not in kept.get(name, {}) or file not in files or golden_name not in case.names(file):
                continue
            want = case.tensor(file, golden_name)
            if key.startswith("cond_"):
                want = want.type(torch.bfloat16)
            result = common.compare(f"{file}/{golden_name}", kept[name][key][tensor], want, case.spread)
            result.name = f"{name}: {result.name}"
            results.append(result)
        if "03-ss" in files and not resumed and name in noises:
            drawn = sorted(int(tensor.split("_")[1]) for tensor in case.names("03-ss") if tensor.startswith("noise_"))
            results.append(common.Result(f"{name}: draws of the sparse structure", len(noises[name]) == len(drawn),
                                         "integers", f"{len(noises[name])}, the reference made {len(drawn)}"))
            for index, noise in zip(drawn, noises[name]):
                result = common.compare(f"03-ss/noise_{index}", noise, case.tensor("03-ss", f"noise_{index}"),
                                        case.spread)
                result.name = f"{name}: {result.name}"
                results.append(result)
        if answer["state"] == "done":
            slow = answer["result"]["slow"]
            may_spill = int(case.manifest["resolution"]) != 1024 or name in touched
            results.append(common.Result(f"{name}: parts that needed the cap lifted", not slow or may_spill, "form",
                                         ", ".join(slow) or "none"))
    results.append(common.Result("card empty after the batch", left <= common.LIBRARY_TABLES_AFTER, "form",
                                 f"{left} B (libraries' own tables only up to {common.LIBRARY_TABLES_AFTER} B)"
                                 + ("" if left <= common.LIBRARY_TABLES_AFTER else f"; {common.BOUNDS_HINT}")))
    leftovers = sorted(path.name for path in work.iterdir())
    results.append(common.Result("work folder empty after the batch", not leftovers, "form", ", ".join(leftovers)))
    if not leftovers:
        shutil.rmtree(work, ignore_errors=True)

    print(f"\nwhole batch: peak {top['allocated']} MiB allocated, {top['reserved']} MiB reserved inside the "
          f"prop-parts, {seconds:.0f} s, {ooms} out-of-memory events counted by PyTorch", flush=True)
    print("peaks per prop-part (MiB allocated / reserved):", flush=True)
    for part, prop, allocated, reserved, ended in peaks:
        print(f"  {part:<9} {prop:<9} {allocated:>6} / {reserved:>6}  {ended}", flush=True)
    print("caps (part, prop, MiB allocated when the cap was set, MiB PyTorch may reserve):", flush=True)
    for entry in caps:
        print(f"  {entry}", flush=True)
    print(f"cap lifted for: {spilled or 'nothing'}", flush=True)
    return results


def main() -> int:
    parser = common.parser(__doc__.split("\n")[0])
    parser.add_argument("--fake", action="store_true", help="the engine's logic on fakes: CPU, no card, no lock")
    parser.add_argument("--only", default="", help="--fake: only the tests whose name contains this")
    parser.add_argument("--verbose", action="store_true", help="--fake: show what the engine prints")
    parser.add_argument("--real-store", action="store_true", help="--fake: use the worktree's own PropStore (CPU)")
    parser.add_argument("--real-loader", action="store_true", help="--fake: use the worktree's own NetworkLoader "
                                                                    "(CPU, stand-in networks)")
    parser.add_argument("--batch", default="", help="golden cases made as one batch, e.g. mv1024,sv1024 (else --case)")
    parser.add_argument("--reverse", action="store_true", help="the batch in reversed request order")
    parser.add_argument("--wave-size", type=int, default=None, help="props that go through the parts together")
    parser.add_argument("--cap-mib", type=int, default=None, help="behave as on a card of this size (7168)")
    parser.add_argument("--inject", action="append", metavar="PART:PROP:STEP:WHAT[:TIMES]",
                        help="at that step event raise an out-of-memory error (oom) or another one (error) TIMES "
                             "times, or ask for a cancel (cancel); e.g. hr:sv1024:3:oom")
    parser.add_argument("--cancel-at-load", default=None, metavar="NETWORK",
                        help="ask for a cancel when that network starts loading, e.g. hr_flow.sv")
    parser.add_argument("--bump-ooms", default=None, metavar="PART:PROP",
                        help="PyTorch's count of failed allocations moves during that prop-part, e.g. cond_hr:sv1024")
    parser.add_argument("--probe-fails", action="store_true",
                        help="the runner's probe of the card fails (with --inject ...:cuda: BatchBroken is expected)")
    parser.add_argument("--die-in", default=None, metavar="PROP", help="exit the process inside that prop's finish "
                                                                       "hook (then run again with --resume)")
    parser.add_argument("--resume", action="store_true", help="use the checkpoints a killed run left in --work")
    parser.add_argument("--work", default="", help="the batch's work folder (default: <repo>/tmp/p3d-checks/"
                                                   "engine-work)")
    args = parser.parse_args()
    return run_fake(args) if args.fake else run_golden(args)


if __name__ == "__main__":
    sys.exit(main())
