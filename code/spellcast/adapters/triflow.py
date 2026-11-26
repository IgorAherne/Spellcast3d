"""TriFlow adapter: the prop's high poly -> a clean, quad-friendly triangle mesh with UVs (stage "retopo", GPU).

TriFlow (code/triflow, a submodule kept byte-identical) is under the ADPNCL 1.0: free for non-commercial use only,
so the page flags it wherever it is picked (`license`). Its weights: models/triflow, hard-linked into
code/triflow/checkpoints by the installer.

The tool process is launchers/triflow_server.py running in the triflow venv, in its own process (TriFlow's code
must never be loaded into the hub). It has no page of its own, only the /spellcast/* routes used here:
GET health, GET progress, POST retopo (blocks until the OBJ is written), POST shutdown.
A prop gets one clean mesh per size it asks for (adapters/variants.py), made one after the other while TriFlow
holds its models; TriFlow is asked for two triangles per vertex wanted (a closed triangle mesh has about twice as
many faces as vertices). Its meshes come without UVs: Instant Meshes then unwraps each (StageContext.unwrap) into
retopo/<size>k/triflow.obj, the files the next stages use.

A prop that failed so that TriFlow must start afresh (its GPU context broke and the process exits, or it went past
RETOPO_TIMEOUT and may still be at it: TriFlow can't be stopped in the middle of a mesh) raises ToolNeedsRestart:
the job runner then starts TriFlow again before the next prop.
"""

from __future__ import annotations

import asyncio
import functools
import os
from typing import Any, Mapping

import httpx

from .. import gpu, paths, projects
from . import variants
from .base import LaunchSpec, StageContext, StageError, ToolAdapter, ToolNeedsRestart

LICENSE_URL = "https://github.com/IgorAherne/triflow/blob/5cf7f00e4f5d5745a1cafbb21ff5d7d28bb4f878/LICENSE"
# TriFlow runs on MeshLib, whose licence is a non-commercial trial: its terms at the version the venv installs
# (install/requirements-triflow.txt pins meshlib==MESHLIB_VERSION; a test keeps the two together).
MESHLIB_VERSION = "3.1.1.211"
MESHLIB_LICENSE_URL = f"https://github.com/MeshInspector/MeshLib/blob/v{MESHLIB_VERSION}/LICENSE"

READY_TIMEOUT = 600.0              # imports (~15 s warm, a minute cold) + reading 1.3 GB of weights
RETOPO_TIMEOUT = 30 * 60.0         # one mesh takes about a minute on an RTX 3060; slower cards far longer
POLL_INTERVAL = 1.0
UNWRAP_SHARE = 0.9                 # the prop's progress bar: TriFlow's meshes up to here, then the UV unwraps

FACES_PER_VERTEX = 2               # TriFlow is asked for triangles: this many per vertex wanted
# The triangles TriFlow is asked for: at least its launcher's least (a smaller size is made with that many), at most
# those of the largest size (projects.RETOPO_MAX_SIZE thousand vertices; TriFlow itself has no such bound: its
# face-count conditioning takes up to a million). Keep in sync with launchers/triflow_server.py and rules.js
# RETOPO_VERTICES.
MIN_FACES = 500
MAX_FACES = FACES_PER_VERTEX * 1000 * projects.RETOPO_MAX_SIZE
MAX_SEED = 2 ** 31 - 1             # projects.RETOPO_NUMBERS
DEFAULTS = {"triflow_quad_ratio": 0.95, "triflow_seed": 0}
QEM_THRESHOLD = 12.0               # TriFlow's own default: the largest error its final simplification allows
# Under a thousand vertices the artist asks for a mesh that small on purpose, so the simplification may go as far as
# it takes (the launcher's largest). At TriFlow's default it stopped a rifle near 470 vertices whether 100, 250 or
# 300 were asked; with this it made 303 for 250 and 347 for 300 (RTX 3060).
QEM_THRESHOLD_SMALL = 1000.0

RAW_OBJ = "triflow.raw.obj"        # TriFlow's mesh, no UVs (removed once unwrapped), in its size's folder
OBJ = "triflow.obj"                # with UVs: what the next stages use
STUBS = paths.LAUNCHERS / "triflow_stubs"
# Its nested checkouts on the path, as TriFlow's setup.sh does (their packages trellis and direct3d_s2).
THIRD_PARTY = ("third_party/TRELLIS", "third_party/Direct3D-S2")


def qem_threshold_for(size: float) -> float:
    """The largest error TriFlow's final simplification allows for a clean mesh of `size` thousand vertices."""
    return QEM_THRESHOLD_SMALL if size < 1 else QEM_THRESHOLD


def attn_backend(compute_cap: float | None) -> str | None:
    """ATTN_BACKEND for the launcher: flash_attn needs sm_80+; below that "sdpa" (torch's own attention through
    launchers/attn_shim.py; xformers 0.0.32 refuses to import next to our flash_attn 2.8.3). None when the GPU is
    unknown: the launcher then asks torch."""
    if compute_cap is None:
        return None
    return "flash_attn" if compute_cap >= 8.0 else "sdpa"


def triflow_params(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The prop's TriFlow settings (retopo.settings triflow_*), checked -> the launcher's quad_ratio and seed.
    StageError in the artist's words."""
    values = {key: default if settings.get(key) is None else settings[key] for key, default in DEFAULTS.items()}
    ratio, seed = values["triflow_quad_ratio"], values["triflow_seed"]
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0.0 <= ratio <= 1.0:
        raise StageError("TriFlow's Quads must be a number from 0 to 1.")
    if not _whole(seed) or not 0 <= seed <= MAX_SEED:
        raise StageError(f"TriFlow's Seed must be a whole number from 0 to {MAX_SEED:,}.")
    return {"quad_ratio": float(ratio), "seed": int(seed)}


def faces_for(size: float) -> int:
    """The triangles TriFlow is asked for to make a clean mesh of `size` thousand vertices: two per vertex, from
    MIN_FACES to MAX_FACES (a size outside them is made with the nearest)."""
    return max(MIN_FACES, min(FACES_PER_VERTEX * variants.vertices(size), MAX_FACES))


def _whole(value: Any) -> bool:
    return not isinstance(value, bool) and (isinstance(value, int) or (isinstance(value, float)
                                                                       and value.is_integer()))


class TriflowAdapter(ToolAdapter):
    tool_id = paths.TRIFLOW
    display_name = "TriFlow"
    role = "retopology tool"
    stage = "retopo"
    uses_gpu = True
    ui_path = "/spellcast/health"      # no page of its own
    license = {"name": "ADPNCL 1.0 (non-commercial)", "url": LICENSE_URL, "noncommercial": True,
               "links": [{"label": "MeshLib terms", "url": MESHLIB_LICENSE_URL}]}

    @functools.cached_property
    def _gpu(self) -> gpu.GpuInfo | None:
        # nvidia-smi takes a moment and the card does not change while the hub runs
        return gpu.primary_gpu()

    def supported(self) -> tuple[bool, str]:
        return gpu.triflow_supported(self._gpu)

    # ---- process lifecycle ----------------------------------------------
    def launch_spec(self, port: int, settings: dict[str, Any]) -> LaunchSpec:
        repo = paths.TRIFLOW_DIR
        extra = {
            # TriFlow, its TRELLIS and Direct3D-S2 checkouts (as its setup.sh sets up), and the kaolin stand-in
            "PYTHONPATH": os.pathsep.join([str(repo), *(str(repo / part) for part in THIRD_PARTY), str(STUBS)]),
            # nothing may be written into TriFlow's folder: its bytecode goes into the venv
            "PYTHONPYCACHEPREFIX": str(paths.venv_dir(self.tool_id) / "pycache"),
        }
        attn = attn_backend(self._gpu.compute_cap if self._gpu else None)
        if attn:
            extra["ATTN_BACKEND"] = attn
        env = paths.clean_env(self.tool_id, extra)
        # Both sparse libraries keep their own default: TRELLIS' flow model spconv, Direct3D-S2's VAEs torchsparse.
        for name in ("SPARSE_BACKEND", "SPARSE_ATTN_BACKEND", "SPARSE_CONV_BACKEND"):
            env.pop(name, None)
        return LaunchSpec(
            argv=[str(paths.venv_python(self.tool_id)), str(paths.LAUNCHERS / "triflow_server.py"),
                  "--repo", str(repo),
                  "--port", str(port),
                  "--projects-root", str(paths.PROJECTS)],
            cwd=repo,
            env=env,
            ready_timeout=READY_TIMEOUT,
        )

    async def health(self, client: httpx.AsyncClient, base_url: str) -> bool:
        try:
            r = await client.get(f"{base_url}/spellcast/health", timeout=5.0)
            return r.status_code == 200 and r.json().get("ready") is True
        except (httpx.HTTPError, ValueError, AttributeError):
            return False

    async def on_stop(self, client: httpx.AsyncClient, base_url: str) -> None:
        # CTRL_BREAK_EVENT does not reach a process spawned with CREATE_NO_WINDOW (imesh.py)
        try:
            await client.post(f"{base_url}/spellcast/shutdown", json={}, timeout=5.0)
        except httpx.HTTPError:
            pass

    # ---- work -------------------------------------------------------------
    def check_settings(self, stage: str, settings: dict[str, Any]) -> None:
        triflow_params(settings)
        variants.sizes_of(settings)

    async def run_stage(self, ctx: StageContext, prop: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """prop.mesh.glb -> TriFlow's mesh of each size the job makes (variants.to_make), one after the other
        (retopo/<size>k/triflow.raw.obj, no UVs, in the model's own frame) -> each unwrapped by Instant Meshes into
        retopo/<size>k/triflow.obj. The prop lets go of the card at its first unwrap (the next prop goes to TriFlow)."""
        wanted = triflow_params(params)
        sizes = variants.to_make(prop, params)
        glb = (prop.get("mesh") or {}).get("glb")
        src = ctx.abs(glb)
        if src is None:
            raise StageError("Make a 3D model in the 3D stage first.")
        if not src.is_file():
            raise StageError(f"The 3D model of this prop ({glb}) is missing. Make it again in the 3D stage.")

        async def make(size: float, part: StageContext) -> tuple[str, dict[str, Any]]:
            raw = f"{variants.folder(size)}/{RAW_OBJ}"
            (ctx.prop_dir / raw).parent.mkdir(parents=True, exist_ok=True)
            body = {"mesh_path": str(src.resolve()), "out_path": str((ctx.prop_dir / raw).resolve()), **wanted,
                    "faces": faces_for(size), "qem_threshold": qem_threshold_for(size), "name": ctx.prop_id}
            made = await self._retopo(part, body)
            if not (ctx.prop_dir / raw).is_file():
                raise StageError("TriFlow reported success but no mesh file appeared. Check the TriFlow log "
                                 "(Settings > Debugging) and try again.")
            return raw, made

        made = await variants.make_each(ctx, sizes, UNWRAP_SHARE, make)
        ctx.report(UNWRAP_SHARE, "Unwrapping the UVs (xatlas, on the CPU)…")
        try:
            leniency = min(max(float(params.get("uv_leniency", projects.UV_LENIENCY)), 0.0), 1.0)
        except (TypeError, ValueError):
            leniency = projects.UV_LENIENCY
        meshes = []
        for size, (raw, result) in made:
            obj = f"{variants.folder(size)}/{OBJ}"
            unwrapped = await ctx.unwrap(raw, obj, leniency)
            _remove(ctx.prop_dir / raw)
            meshes.append({"size": size, "obj": obj, "stats": self._stats(size, wanted, result, unwrapped)})
        ctx.report(1.0, "Clean meshes saved.")
        # TriFlow can't stop in the middle of a mesh: after a cancel the one under way is still made, and saved
        ctx.completed = len(made) == len(sizes)
        return {"variants": meshes, "method": self.tool_id, "source": glb, "saved_from": "auto"}

    @staticmethod
    def _stats(size: float, wanted: dict[str, Any], made: dict[str, Any], unwrapped: dict[str, Any]) -> dict[str, Any]:
        """A clean mesh's numbers, from what TriFlow and the unwrap said of it."""
        warnings = [w for w in (made.get("warning"), unwrapped.get("warning")) if isinstance(w, str) and w]
        seconds = [s for s in (made.get("seconds"), unwrapped.get("seconds")) if isinstance(s, (int, float))]
        stats = {
            "vertices": unwrapped.get("vertices", made.get("vertices")),
            "faces": unwrapped.get("faces", made.get("faces")),
            "has_uv": unwrapped.get("has_uv"),
            "seconds": round(sum(seconds), 1) if seconds else None,
            "warning": " ".join(warnings) or None,
            "seed": made.get("seed", wanted["seed"]),
            "target_faces": faces_for(size),
            "quad_ratio": wanted["quad_ratio"],
            "watertight": made.get("watertight"),
        }
        if isinstance(made.get("note"), str) and made["note"]:
            stats["note"] = made["note"]        # what TriFlow says about the mesh, no alarm (the page: a quiet line)
        return stats

    async def _retopo(self, ctx: StageContext, body: dict[str, Any]) -> dict[str, Any]:
        """POST /spellcast/retopo and report its progress (0..1 of this mesh) until it answers."""
        ctx.report(0.0, "Sending the 3D model to TriFlow…")
        post = asyncio.create_task(ctx.client.post(
            f"{ctx.base_url}/spellcast/retopo", json=body, timeout=httpx.Timeout(RETOPO_TIMEOUT, connect=10.0)))
        try:
            while not (await asyncio.wait({post}, timeout=POLL_INTERVAL))[0]:
                await self._report_progress(ctx)
            response = post.result()
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise StageError("TriFlow is not reachable (it may have stopped; Settings > Debugging shows its log). "
                             "Make the clean mesh again: TriFlow starts by itself.") from None
        except httpx.TimeoutException:
            # It can't be told to stop and may still be at it: the next prop gets a fresh TriFlow (the job runner
            # restarts it).
            raise ToolNeedsRestart(f"TriFlow did not finish this mesh within {RETOPO_TIMEOUT / 60:.0f} minutes, so it "
                                   "was stopped (it starts again for the next prop). Ask for fewer vertices, or "
                                   "close other programs that use the graphics card, and make it again.") from None
        except httpx.HTTPError:
            raise StageError("TriFlow stopped responding while making the clean mesh (it may have crashed or run "
                             "out of memory). Check its log in Settings > Debugging, close other programs and "
                             "try again.") from None
        finally:
            post.cancel()   # no-op once answered; drops the request if the job itself was cancelled

        try:
            data = response.json()
        except ValueError:
            data = None
        if not isinstance(data, dict):
            data = {}
        if response.status_code == 200 and data.get("obj_path"):
            return data
        detail = data.get("detail")
        if data.get("kind") in ("restart", "busy"):
            # "restart": the graphics card stopped answering TriFlow after an error (the process exits); "busy": it
            # still works on a mesh the hub gave up on. Either way the next prop needs a fresh TriFlow.
            raise ToolNeedsRestart(detail if isinstance(detail, str) and detail else
                                   "TriFlow has to start again after an error. Make this clean mesh again.")
        if isinstance(detail, str) and detail:
            raise StageError(detail)
        raise StageError(f"TriFlow failed (HTTP {response.status_code}). Check the TriFlow log (Settings > "
                         "Debugging) and try again.")

    async def _report_progress(self, ctx: StageContext) -> None:
        try:
            r = await ctx.client.get(f"{ctx.base_url}/spellcast/progress", timeout=5.0)
            p = r.json()
            if p.get("phase") != "working":
                return   # not started yet, or just finished (its answer is on its way)
            fraction = float(p.get("fraction") or 0.0)
            message = str(p.get("message") or "Working")
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            return   # progress is cosmetic; the retopo request decides success
        message = ("Stopping after this mesh: TriFlow can’t stop in the middle of one, so it finishes and is saved"
                   if ctx.cancelled() else f"{message}…")
        ctx.report(fraction, message)


def _remove(path: Any) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass   # in use (an antivirus scan); the next run replaces it
