"""The Projection stage: bake the high-poly mesh's surface onto the low poly's UVs. The high poly is the prop's 3D
model, or its Omni-Texture while the prop is chosen for one and has a fresh one (projects.high_poly).

The bake runs in the Instant Meshes tool: its launcher's task API (/spellcast/bake) starts
launchers/bake_worker.py in the imesh venv, a child process per bake. This module is the hub
side and pure Python: it checks the prop's inputs and settings, starts the bake, follows its
progress (and cancels it when the job is cancelled) and turns the result into the prop's
"projection" block. ImeshAdapter.run_stage calls run() for ctx.stage == "projection".
"""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

import httpx

from ..projects import DEFAULT_STAGE_SETTINGS, high_poly, omnitexture_used
from .base import StageContext, StageError

# Keep in sync with launchers/spellbake/settings.py (the worker checks them again).
TEXTURE_SIZES = (512, 1024, 2048, 4096)
QUALITIES = (1, 2, 3)                   # samples per texel side: Fast, Good, Best
RAY_DISTANCE_PERCENT = (0.05, 20.0)     # of the high poly's bounding-box diagonal; None = automatic
MAX_HARD_EDGE_ANGLE = 180.0             # degrees; -1 = every edge smooth
TANGENT_BASES = ("mikk", "per_vertex")
NORMAL_Y = ("opengl", "directx")
SWITCHES = {"ao": "Ambient occlusion", "height": "Height map", "repack_uvs": "Repack UVs",
            "export_obj": "Also write OBJ", "export_fbx": "Also write FBX"}

POLL_SECONDS = 0.5
WORK_SECONDS = 60 * 60.0                # a 4096 px bake of a dense mesh takes minutes, never an hour
REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

NOT_RESPONDING = ("Instant Meshes stopped answering during the bake. Restart it in Settings > Debugging "
                  "and bake again.")


def bake_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The prop's projection settings, checked, as the bake worker takes them. Raises StageError."""
    s = {**DEFAULT_STAGE_SETTINGS["projection"], **settings}
    size = _number(s["texture_size"], "Texture size")
    if size not in TEXTURE_SIZES:
        raise StageError("Texture size must be 512, 1024, 2048 or 4096 pixels.")
    quality = _number(s["quality"], "Quality")
    if quality not in QUALITIES:
        raise StageError("Quality must be Fast, Good or Best.")
    ray = s["ray_distance"]
    if ray is not None:
        ray = _number(ray, "Ray distance")
        low, high = RAY_DISTANCE_PERCENT
        if not low <= ray <= high:
            raise StageError(f"Ray distance must be Auto, or between {low:g} % and {high:g} % of the model's "
                             f"size (it is {ray:g} %).")
    hard = _number(s["hard_edge_angle"], "Hard edge angle")
    if hard != -1 and not 0 <= hard <= MAX_HARD_EDGE_ANGLE:
        raise StageError("Hard edge angle must be between 0° and 180°, or -1 to keep every edge smooth.")
    if s["tangent_basis"] not in TANGENT_BASES:
        raise StageError("Tangent basis must be MikkTSpace (Blender, Unreal, Unity HDRP) or per vertex "
                         "(three.js, Unity URP, Godot).")
    if s["normal_y"] not in NORMAL_Y:
        raise StageError("The normal map must be made for OpenGL (Blender, Unity, Godot, glTF) or DirectX "
                         "(Unreal, 3ds Max).")
    for key, label in SWITCHES.items():
        if not isinstance(s[key], bool):
            raise StageError(f'"{label}" must be on or off.')
    return {"texture_size": int(size), "quality": int(quality), "ray_distance": ray, "hard_edge_angle": hard,
            "tangent_basis": s["tangent_basis"], "normal_y": s["normal_y"], **{key: s[key] for key in SWITCHES}}


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise StageError(f"{label} must be a number.")
    try:
        number = float(value)
    except ValueError:
        raise StageError(f"{label} must be a number (it is {value!r}).") from None
    if not math.isfinite(number):
        raise StageError(f"{label} must be a number (it is {value!r}).")
    return number


async def run(ctx: StageContext, prop: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """Bake one prop -> the patch for prop["projection"] (paths relative to the prop folder)."""
    settings = bake_settings(params)
    high_rel, low_rel = _inputs(ctx, prop)
    high, low = ctx.prop_dir / high_rel, ctx.prop_dir / low_rel
    inputs_before = _signature(high, low)
    out_root = ctx.prop_dir / "projection"
    out_root.mkdir(parents=True, exist_ok=True)
    body = {"high": str(high.resolve()), "low": str(low.resolve()), "out_root": str(out_root.resolve()),
            "settings": settings, "threads": int((ctx.settings.get("imesh") or {}).get("threads", -1))}

    ctx.report(0.0, "Starting the bake…")
    started = await _post(ctx.client, f"{ctx.base_url}/spellcast/bake", body)
    task_url = f"{ctx.base_url}/spellcast/bake/{quote(str(started['task_id']), safe='')}"
    try:
        status = await _follow(ctx, task_url)
    except BaseException:
        # Cancelled, timed out, or the job itself stopped: the bake must not go on without us.
        await _cancel(ctx.client, task_url)
        raise
    if status.get("state") == "cancelled":
        raise StageError("The bake was cancelled in Instant Meshes.")
    if status.get("state") != "done":
        raise StageError(str(status.get("detail") or "The bake failed. The Instant Meshes log (in its status "
                                                      "panel) has the details."))
    result = status.get("result") or {}
    ctx.report(1.0, "Baked.")
    try:
        return {
            "glb": _rel(ctx, result["glb"]),
            "fbx": _rel(ctx, result.get("fbx")),
            "obj": _rel(ctx, result.get("obj")),
            "maps": [{"id": m["id"], "label": m["label"], "path": _rel(ctx, m["path"]),
                      "colorspace": m["colorspace"]} for m in result.get("maps") or []],
            "source_mesh": high_rel,
            "source_retopo": low_rel,
            "stats": result.get("stats") or {},
            # The mesh or the low poly may have been replaced while the bake ran.
            "stale": _signature(high, low) != inputs_before,
        }
    except (KeyError, TypeError) as e:
        raise StageError("Instant Meshes returned an incomplete bake result. Bake again; if it keeps "
                         "happening, its log has the details.") from e


def _inputs(ctx: StageContext, prop: Mapping[str, Any]) -> tuple[str, str]:
    """(high-poly GLB, low-poly OBJ) of the prop, relative to its folder, both present and usable. The high poly is
    its Omni-Texture while that is used (the colours baked are the new ones; the shape is the same)."""
    mesh, retopo = prop.get("mesh") or {}, prop.get("retopo") or {}
    high_rel, low_rel = high_poly(dict(prop)), retopo.get("obj")
    if not mesh.get("glb") or not high_rel:
        raise StageError("This prop has no 3D model yet. Make one in the 3D stage, or import your "
                         "own 3D model there.")
    if omnitexture_used(dict(prop)) and not (ctx.prop_dir / high_rel).is_file():
        raise StageError(f"The Omni-Texture of this prop ({high_rel}) is missing. Make it again in the "
                         f"Omni-Texture stage, or click the prop there to bake its 3D model's own colours.")
    if not (ctx.prop_dir / high_rel).is_file():
        raise StageError(f"The 3D model of this prop ({high_rel}) is missing. Make it again in the "
                         f"3D stage.")
    if not low_rel or not (ctx.prop_dir / low_rel).is_file():
        raise StageError("This prop has no low-poly model yet. Make its clean meshes in the Retopo stage first.")
    if (retopo.get("stats") or {}).get("has_uv") is False:
        raise StageError("The low-poly model has no UVs, so there is nothing to bake onto. Run Retopo again "
                         "(it unwraps the UVs), then bake.")
    return high_rel, low_rel


def _signature(*files: Path) -> tuple[tuple[int, int], ...]:
    out = []
    for file in files:
        try:
            stat = file.stat()
            out.append((stat.st_mtime_ns, stat.st_size))
        except OSError:
            out.append((0, -1))
    return tuple(out)


async def _follow(ctx: StageContext, task_url: str) -> dict[str, Any]:
    """Poll the bake task, reporting its progress, until it ends -> its last status (done, error or
    cancelled). Raises StageError when the job is cancelled, the bake takes too long or is lost."""
    deadline = time.monotonic() + WORK_SECONDS
    while True:
        if ctx.cancelled():
            raise StageError("The bake was cancelled.")
        if time.monotonic() > deadline:
            raise StageError(f"The bake did not finish within {WORK_SECONDS / 60:.0f} minutes. Try a smaller "
                             f"texture size or the Fast quality.")
        try:
            response = await ctx.client.get(task_url, timeout=REQUEST_TIMEOUT)
        except httpx.HTTPError as e:
            raise StageError(NOT_RESPONDING) from e
        if response.status_code == 404:
            raise StageError("Instant Meshes restarted during the bake, so the bake was lost. Bake again.")
        status = _json(response)
        if status.get("state") != "running":
            return status
        ctx.report(float(status.get("fraction") or 0.0), str(status.get("message") or "Baking…"))
        await asyncio.sleep(POLL_SECONDS)


async def _post(client: httpx.AsyncClient, url: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        response = await client.post(url, json=body, timeout=REQUEST_TIMEOUT)
    except httpx.HTTPError as e:
        raise StageError(NOT_RESPONDING) from e
    return _json(response)


def _json(response: httpx.Response) -> dict[str, Any]:
    """The tool's JSON answer; an error answer becomes a StageError with the tool's own words."""
    try:
        data = response.json()
    except ValueError:
        data = None
    if response.status_code >= 400 or not isinstance(data, dict):
        detail = data.get("detail") if isinstance(data, dict) else None
        raise StageError(detail if isinstance(detail, str) and detail else
                         f"Instant Meshes could not bake (HTTP {response.status_code}). Its log (in the tool's "
                         f"status panel) has the details.")
    return data


async def _cancel(client: httpx.AsyncClient, task_url: str) -> None:
    try:
        await asyncio.shield(client.post(f"{task_url}/cancel", json={}, timeout=REQUEST_TIMEOUT))
    except (httpx.HTTPError, asyncio.CancelledError):
        pass    # the tool is gone (then so is its bake) or the hub is closing (the tool dies with it)


def _rel(ctx: StageContext, raw: Any) -> str | None:
    """A file the worker wrote, as a path relative to the prop folder (None stays None)."""
    if raw is None:
        return None
    path = Path(str(raw))
    if not path.resolve().is_relative_to(ctx.prop_dir.resolve()) or not path.is_file():
        raise StageError(f"The bake did not write {path.name} into this prop's folder. Bake again; if it "
                         f"keeps happening, the Instant Meshes log has the details.")
    return ctx.rel(path)
