"""LATO.2 adapter: the prop's high poly -> a clean triangle mesh of a chosen vertex count (stage "retopo", GPU).

LATO.2 (MIT, weights MIT) is one of the retopology methods an artist picks per prop (retopo.settings.method,
next to Instant Meshes and TriFlow). The tool process is launchers/lato2_server.py in the lato2 venv: its models
stay loaded while the job's props are made one after another, each in one clean mesh per size it asks for
(adapters/variants.py). It writes triangles without UVs in the high poly's own frame
(retopo/<size>k/lato2.raw.obj); Instant Meshes then unwraps them on the CPU (StageContext.unwrap) into
retopo/<size>k/lato2.obj, the files the next stages use.

The prop's settings (retopo.settings, flat keys): its sizes (thousands of vertices; LATO.2 places MIN_VERTICES to
MAX_VERTICES, the latter its models' own bound, so a size outside them is made with the nearest), lato2_seed, and
the shared uv_leniency of the unwrap.

A prop that failed so that LATO.2 must start afresh (its GPU context broke, or it went past RETOPO_TIMEOUT and may
still be at it) raises ToolNeedsRestart: the job runner then starts LATO.2 again before the next prop.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Mapping

import httpx

from .. import gpu, paths
from ..projects import UV_LENIENCY
from . import variants
from .base import LaunchSpec, StageContext, StageError, ToolAdapter, ToolNeedsRestart

READY_TIMEOUT = 600.0              # ~4.8 GB of weights read before the port opens (~20 s from a warm disk)
RETOPO_TIMEOUT = 45 * 60.0         # ~50 s a mesh on an RTX 3060; float32 on a GTX 10xx is far slower
POLL_INTERVAL = 1.0
LOW_VRAM = ("auto", "on", "off")   # keep in sync with launchers/lato2_server.py
# Keep in sync with launchers/lato2_server.py (MIN_VERTICES, MAX_VERTICES; the latter is what its models were trained
# for: vflow max_vertex_num, tflow max_vertices) and rules.js RETOPO_VERTICES.
MIN_VERTICES = 200
MAX_VERTICES = 10_000
MAX_SEED = 2 ** 31 - 1             # projects.RETOPO_NUMBERS; the launcher also takes < 0 (random), never sent
RAW = "lato2.raw.obj"              # the launcher's triangles, no UVs, in its size's folder
OBJ = "lato2.obj"                  # with UVs: what the next stages read
GENERATING = 0.9                   # share of a prop's progress bar for the meshes; the unwraps take the rest

LICENSE = {"name": "MIT",
           "url": "https://github.com/IgorAherne/LATO2-stableprojectorz/blob/"
                  "fbb1f5a5755e6db8700cf6922fd506830b7cdccd/LICENSE",
           "noncommercial": False}


def lato2_params(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The prop's LATO.2 settings, checked -> {"seed", "uv_leniency"}."""
    try:
        seed = int(settings.get("lato2_seed", 0))
        leniency = float(settings.get("uv_leniency", UV_LENIENCY))
    except (TypeError, ValueError) as exc:
        raise StageError("This prop's LATO.2 settings are not valid numbers. Set its Seed again in the Retopo "
                         "stage.") from exc
    if not 0 <= seed <= MAX_SEED:
        raise StageError(f"LATO.2's Seed must be a whole number from 0 to {MAX_SEED:,}.")
    return {"seed": seed, "uv_leniency": max(0.0, min(leniency, 1.0))}


def vertices_for(size: float) -> int:
    """The vertices LATO.2 is asked for to make a clean mesh of `size` thousand vertices: from MIN_VERTICES to
    MAX_VERTICES (a size outside them is made with the nearest)."""
    return max(MIN_VERTICES, min(variants.vertices(size), MAX_VERTICES))


class Lato2Adapter(ToolAdapter):
    tool_id = paths.LATO2
    display_name = "LATO.2"
    role = "retopology tool"
    stage = "retopo"
    uses_gpu = True
    ui_path = "/spellcast/health"      # no page of its own
    license = LICENSE

    @functools.cached_property
    def _gpu(self) -> gpu.GpuInfo | None:
        return gpu.primary_gpu()       # nvidia-smi takes a moment and the card does not change while the hub runs

    def supported(self) -> tuple[bool, str]:
        return gpu.lato2_supported(self._gpu)

    # ---- process lifecycle ----------------------------------------------
    def launch_spec(self, port: int, settings: dict[str, Any]) -> LaunchSpec:
        low_vram = (settings.get("lato2") or {}).get("low_vram")
        return LaunchSpec(
            argv=[str(paths.venv_python(self.tool_id)), str(paths.LAUNCHERS / "lato2_server.py"),
                  "--repo", str(paths.LATO2_DIR),
                  "--models", str(paths.MODELS_LATO2),
                  "--port", str(port),
                  "--low-vram", low_vram if low_vram in LOW_VRAM else "auto",
                  "--projects-root", str(paths.PROJECTS)],
            cwd=paths.LATO2_DIR,
            env=paths.clean_env(self.tool_id),
            ready_timeout=READY_TIMEOUT,
        )

    async def health(self, client: httpx.AsyncClient, base_url: str) -> bool:
        try:
            r = await client.get(f"{base_url}/spellcast/health", timeout=5.0)
            return r.status_code == 200 and r.json().get("ready") is True
        except (httpx.HTTPError, ValueError, AttributeError):
            return False

    async def on_stop(self, client: httpx.AsyncClient, base_url: str) -> None:
        # CTRL_BREAK_EVENT does not reach a process spawned with CREATE_NO_WINDOW (imesh adapter)
        try:
            await client.post(f"{base_url}/spellcast/shutdown", json={}, timeout=5.0)
        except httpx.HTTPError:
            pass

    # ---- work -------------------------------------------------------------
    def check_settings(self, stage: str, settings: dict[str, Any]) -> None:
        lato2_params(settings)
        variants.sizes_of(settings)

    async def run_stage(self, ctx: StageContext, prop: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """prop.mesh.glb -> LATO.2's triangles of each size the job makes (variants.to_make), one after the other
        (retopo/<size>k/lato2.raw.obj, in the high poly's frame) -> each unwrapped by Instant Meshes into
        retopo/<size>k/lato2.obj. The prop lets go of the card at its first unwrap (the next prop goes to LATO.2)."""
        settings = lato2_params(params)
        sizes = variants.to_make(prop, params)
        glb = (prop.get("mesh") or {}).get("glb")
        src = ctx.abs(glb)
        if src is None:
            raise StageError("Make a 3D model in the 3D stage first.")
        if not src.is_file():
            raise StageError(f"The 3D model of this prop ({glb}) is missing. Make it again in the 3D stage.")

        async def make(size: float, part: StageContext) -> tuple[str, dict[str, Any]]:
            raw = f"{variants.folder(size)}/{RAW}"
            (ctx.prop_dir / raw).parent.mkdir(parents=True, exist_ok=True)
            body = {"mesh_path": str(src.resolve()), "out_path": str((ctx.prop_dir / raw).resolve()),
                    "vertices": vertices_for(size), "seed": settings["seed"]}
            part.report(0.0, "Sending the 3D model to LATO.2…")
            result = await self._retopo(part, body)
            if not (ctx.prop_dir / raw).is_file():
                raise StageError("LATO.2 reported success but no mesh file appeared. Check its log (Settings > "
                                 "Debugging) and try again.")
            return raw, result

        made = await variants.make_each(ctx, sizes, GENERATING, make)
        ctx.report(GENERATING, "Unwrapping the UVs (xatlas, on the CPU)…")
        meshes = []
        for size, (raw, result) in made:
            obj = f"{variants.folder(size)}/{OBJ}"
            unwrapped = await ctx.unwrap(raw, obj, settings["uv_leniency"])
            _remove(ctx.prop_dir / raw)        # only the unwrapped copy is kept (as TriFlow's adapter does)
            meshes.append({"size": size, "obj": obj, "stats": self._stats(size, settings, result, unwrapped)})
        ctx.report(1.0, "Clean meshes saved.")
        ctx.completed = len(made) == len(sizes)    # made in full, even when the job was cancelled meanwhile
        return {"variants": meshes, "method": self.tool_id, "source": glb, "saved_from": "auto"}

    @staticmethod
    def _stats(size: float, settings: dict[str, Any], result: dict[str, Any],
               unwrapped: dict[str, Any]) -> dict[str, Any]:
        """A clean mesh's numbers, from what LATO.2 and the unwrap said of it."""
        warnings = [w for w in (result.get("warning"), unwrapped.get("warning")) if isinstance(w, str) and w]
        seconds = [s for s in (result.get("seconds"), unwrapped.get("seconds")) if isinstance(s, (int, float))]
        stats: dict[str, Any] = {
            "vertices": _count(unwrapped.get("vertices"), _count(result.get("vertices"), 0)),
            "faces": _count(unwrapped.get("faces"), _count(result.get("faces"), 0)),
            "has_uv": unwrapped.get("has_uv") is True,
            "seconds": round(sum(seconds), 1) if seconds else None,
            "warning": " ".join(warnings) or None,
            "seed": result.get("seed", settings["seed"]),
            "target_vertices": vertices_for(size),       # what LATO.2 was asked for
        }
        for key in ("non_manifold_edges", "boundary_edges", "outward", "missing", "prepared"):
            if key in result:
                stats[key] = result[key]
        return stats

    async def _retopo(self, ctx: StageContext, body: dict[str, Any]) -> dict[str, Any]:
        """POST /spellcast/retopo and report its progress (0..1 of this mesh) until it answers; a cancelled job asks
        LATO.2 to stop (again at every poll until LATO.2 says it stops: a cancel that arrives before the mesh has
        started is not kept)."""
        post = asyncio.create_task(ctx.client.post(f"{ctx.base_url}/spellcast/retopo", json=body,
                                                   timeout=httpx.Timeout(RETOPO_TIMEOUT, connect=10.0)))
        stopping = False
        try:
            while not (await asyncio.wait({post}, timeout=POLL_INTERVAL))[0]:
                if ctx.cancelled() and not stopping:
                    stopping = await self._ask_to_stop(ctx)
                await self._report_progress(ctx)
            response = post.result()
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise StageError("LATO.2 is not reachable (it may have stopped; Settings > Debugging shows its log). "
                             "Make the clean mesh again: LATO.2 starts by itself.") from None
        except httpx.TimeoutException:
            # Its GPU thread may still be at it: the next prop gets a fresh LATO.2 (the job runner restarts it).
            await self._ask_to_stop(ctx)
            raise ToolNeedsRestart(f"LATO.2 did not finish this mesh within {RETOPO_TIMEOUT / 60:.0f} minutes, so "
                                   "it was stopped (it starts again for the next prop). Ask for fewer vertices, or "
                                   "close other programs that use the graphics card, and make it again.") from None
        except httpx.HTTPError:
            raise StageError("LATO.2 stopped responding while making the clean mesh (it may have crashed or run out "
                             "of memory). Check its log in Settings > Debugging, close other programs and try "
                             "again.") from None
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
        kind = data.get("kind")
        if kind == "cancelled":
            raise StageError("Cancelled.")
        detail = data.get("detail")
        if kind in ("restart", "busy"):
            # "restart": the graphics card stopped answering LATO.2 after an error (the process exits); "busy": it
            # still works on a mesh the hub gave up on. Either way the next prop needs a fresh LATO.2.
            raise ToolNeedsRestart(detail if isinstance(detail, str) and detail else
                                   "LATO.2 has to start again after an error. Make this clean mesh again.")
        if kind == "oom":
            raise StageError("LATO.2 ran out of GPU memory. Close other programs that use the graphics card "
                             "(games, video tools, other AI apps), ask for fewer vertices, and try again.")
        if isinstance(detail, str) and detail:
            raise StageError(detail)
        raise StageError(f"LATO.2 failed (HTTP {response.status_code}). Check its log (Settings > Debugging) and "
                         "try again.")

    @staticmethod
    async def _ask_to_stop(ctx: StageContext) -> bool:
        """POST /spellcast/cancel -> whether LATO.2 is stopping its mesh now."""
        try:
            r = await ctx.client.post(f"{ctx.base_url}/spellcast/cancel", json={}, timeout=5.0)
            return r.json().get("cancelling") is True
        except (httpx.HTTPError, ValueError, AttributeError):
            return False

    async def _report_progress(self, ctx: StageContext) -> None:
        """The tool's step in its words ("Placing the vertices (step 3 of 24)…"); the page shows the time."""
        try:
            r = await ctx.client.get(f"{ctx.base_url}/spellcast/progress", timeout=5.0)
            p = r.json()
            fraction = float(p.get("fraction") or 0.0)
            message = str(p.get("message") or "Working")
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            return   # progress is cosmetic; the retopo request decides success
        message = "Stopping…" if ctx.cancelled() else f"{message}…"
        ctx.report(fraction, message)


def _count(value: Any, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else default


def _remove(path: Any) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass   # in use (an antivirus scan); the next run replaces it
