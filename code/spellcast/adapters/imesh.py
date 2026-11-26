"""Instant Meshes (stages "retopo" and "projection", CPU only).

retopo: quad retopology + UV unwrap -> OBJ, one per size the prop asks for (adapters/variants.py), or the one
the artist shapes in its interactive viewer. projection: bake the high poly's surface onto the
low poly's UVs (adapters/projection.py). It also converts the artist's own 3D model into the
canonical GLB the other stages read (import_mesh), and unwraps the UVs of the low polys LATO.2 and
TriFlow make (unwrap, through StageContext.unwrap).

The tool process is launchers/imesh_server.py inside the imesh venv: the
instant_meshes_brush viewer app plus the /spellcast/* routes this adapter calls.
Everything is exchanged as local file paths: the tool reads the prop's files and
writes its results itself, so no mesh travels over HTTP.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

import httpx

from .. import paths
from ..projects import UV_LENIENCY
from . import projection, variants
from .base import LaunchSpec, StageContext, StageError, ToolAdapter

# Keep in sync with launchers/imesh_server.py and rules.js RETOPO_VERTICES (every size fits: projects.RETOPO_MIN_SIZE
# to RETOPO_MAX_SIZE thousand).
MIN_TARGET_VERTICES = 100
MAX_TARGET_VERTICES = 100_000
OBJ = "retopo.obj"      # a clean mesh with UVs, in its size's folder (retopo/<size>k/)
MAX_SMOOTH_ITER = 10
ROSY = (2, 4, 6)        # orientation symmetry: 4 = quads, 6 = triangles, 2 = quad-dominant
POSY = (3, 4)           # position symmetry: 4 = quads, 3 = triangles
# The retopo settings the viewer's Accept stores into prop.retopo.settings (as the launcher reports them
# from the session that made the accepted mesh).
VIEWER_SETTINGS = ("target_vertices", "pure_quad", "crease_angle", "smooth_iter", "uv_leniency", "rosy", "posy",
                   "extrinsic", "align_to_boundaries")

# The solve takes seconds, and so does the UV unwrap (xatlas) since Spellcast3D's own xatlas wheel
# (install/components.py XATLAS_WHEEL): PyPI's chart merging was about cubic in the number of seed
# charts, which a low leniency and leafy shapes multiply (a 22,800-triangle tree took 164 s at
# leniency 0.5; 1.1 s now, the same atlas). The timeouts below are left generous all the same.
WORK_TIMEOUT = httpx.Timeout(60 * 60.0, connect=10.0)      # retopo, export
OPEN_TIMEOUT = httpx.Timeout(10 * 60.0, connect=10.0)      # loading + preprocessing a GLB: seconds
# The launcher gives a model conversion 10 minutes and then answers with its own message.
IMPORT_TIMEOUT = httpx.Timeout(11 * 60.0, connect=10.0)

RETOPO_TOO_SLOW = ("Try a lower target vertex count, or a higher UV leniency (fewer, bigger UV chunks "
                   "unwrap much faster).")
UNWRAP_TOO_SLOW = ("Try fewer triangles, or a higher UV leniency (fewer, bigger UV chunks unwrap much "
                   "faster).")
IMPORT_TOO_SLOW = "Try the model with fewer polygons or smaller textures."

NOT_RESPONDING = ("The retopology tool (Instant Meshes) is not responding. Restart it in Settings > "
                  "Debugging and try again.")


def _retopo_params(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The prop's retopo settings, checked and in the launcher's units."""
    try:
        target = int(settings.get("target_vertices", 5000))
        crease = float(settings.get("crease_angle", -1))
        smooth = int(settings.get("smooth_iter", 2))
        leniency = float(settings.get("uv_leniency", UV_LENIENCY))
        rosy, posy = int(settings.get("rosy", 4)), int(settings.get("posy", 4))
    except (TypeError, ValueError) as exc:
        raise StageError("This prop's Retopo settings are not valid numbers. Open it in the Retopo "
                         "stage's viewer and press Accept again.") from exc
    if not MIN_TARGET_VERTICES <= target <= MAX_TARGET_VERTICES:
        raise StageError(f"Target vertices must be between {MIN_TARGET_VERTICES:,} and "
                         f"{MAX_TARGET_VERTICES:,} (it is {target:,}).")
    return {
        "target_vertices": target,
        "pure_quad": bool(settings.get("pure_quad", False)),
        "crease_angle": -1.0 if crease < 0 else min(crease, 180.0),
        "smooth_iter": max(0, min(smooth, MAX_SMOOTH_ITER)),
        "uv_leniency": max(0.0, min(leniency, 1.0)),
        "rosy": rosy if rosy in ROSY else 4,
        "posy": posy if posy in POSY else 4,
        "extrinsic": bool(settings.get("extrinsic", True)),
        "align_to_boundaries": bool(settings.get("align_to_boundaries", False)),
        "deterministic": bool(settings.get("deterministic", False)),
    }


def vertices_for(size: float) -> int:
    """The target vertices Instant Meshes is asked for to make a clean mesh of `size` thousand vertices: from
    MIN_TARGET_VERTICES to MAX_TARGET_VERTICES, which every size fits in."""
    return max(MIN_TARGET_VERTICES, min(variants.vertices(size), MAX_TARGET_VERTICES))


def accepted_settings(raw: Any) -> dict[str, Any]:
    """The viewer settings the launcher reported with an accepted mesh, as prop.retopo.settings keeps them:
    the VIEWER_SETTINGS keys, checked and clamped like _retopo_params; a value that is missing or not
    usable is left out (the prop keeps its own), so an Accept never fails over a setting."""
    if not isinstance(raw, Mapping):
        return {}
    kept: dict[str, Any] = {}
    for key in VIEWER_SETTINGS:
        value = raw.get(key)
        if value is None or isinstance(value, (dict, list, str)):
            continue
        try:
            checked = _retopo_params({key: value})[key]
        except (StageError, TypeError, ValueError):
            continue
        if key in ("rosy", "posy") and checked != value:
            continue            # not a symmetry the viewer offers
        kept[key] = checked
    return kept


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, str) and detail:
        return detail
    return (f"Instant Meshes answered with an error (HTTP {response.status_code}). "
            f"Its log (in the tool's status panel) has the details.")


async def _post(client: httpx.AsyncClient, url: str, body: dict[str, Any],
                timeout: httpx.Timeout, too_slow: str = RETOPO_TOO_SLOW) -> dict[str, Any]:
    try:
        response = await client.post(url, json=body, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise StageError(f"Instant Meshes did not finish within {timeout.read / 60:.0f} minutes. "
                         f"{too_slow}") from exc
    except httpx.TransportError as exc:
        raise StageError(NOT_RESPONDING) from exc
    if response.status_code >= 400:
        raise StageError(_detail(response))
    return response.json()


class ImeshAdapter(ToolAdapter):
    tool_id = paths.IMESH
    display_name = "Instant Meshes"
    role = "retopology tool"
    stage = "retopo"
    extra_stages = ("projection",)
    uses_gpu = False
    ui_path = "/viewer"     # without ?session= the launcher opens a shared standalone session

    def launch_spec(self, port: int, settings: dict[str, Any]) -> LaunchSpec:
        threads = int((settings.get("imesh") or {}).get("threads", -1))
        return LaunchSpec(
            argv=[str(paths.venv_python(paths.IMESH)), str(paths.LAUNCHERS / "imesh_server.py"),
                  "--port", str(port), "--threads", str(threads), "--projects-root", str(paths.PROJECTS)],
            cwd=paths.IMESH_DIR,
            env=paths.clean_env(paths.IMESH),
            ready_timeout=90,
        )

    async def health(self, client: httpx.AsyncClient, base_url: str) -> bool:
        try:
            response = await client.get(f"{base_url}/spellcast/health", timeout=5.0)
            return response.status_code == 200 and response.json().get("ready") is True
        except (httpx.HTTPError, ValueError, AttributeError):
            return False

    async def on_stop(self, client: httpx.AsyncClient, base_url: str) -> None:
        # CTRL_BREAK_EVENT does not reach a process spawned with CREATE_NO_WINDOW (it has
        # its own console), and a hard kill leaves the sessions' temp folders behind.
        try:
            await client.post(f"{base_url}/spellcast/shutdown", json={}, timeout=5.0)
        except httpx.HTTPError:
            pass

    def check_settings(self, stage: str, settings: dict[str, Any]) -> None:
        if stage == "projection":
            projection.bake_settings(settings)
        else:
            _retopo_params(settings)
            variants.sizes_of(settings)

    async def run_stage(self, ctx: StageContext, prop: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """projection: the bake. retopo: Instant Meshes' quads of each size the job makes (variants.to_make), with
        the prop's viewer settings but the size's vertex count, one after the other: retopo/<size>k/retopo.obj."""
        if ctx.stage == "projection":
            return await projection.run(ctx, prop, params)
        glb = (prop.get("mesh") or {}).get("glb")
        src = ctx.abs(glb)
        if src is None:
            raise StageError("Make a 3D model in the 3D stage first.")
        if not src.is_file():
            raise StageError(f"The 3D model of this prop ({glb}) is missing. Make it again "
                             f"in the 3D stage.")
        settings = _retopo_params(params)

        async def make(size: float, part: StageContext) -> dict[str, Any]:
            obj = f"{variants.folder(size)}/{OBJ}"
            dst = ctx.prop_dir / obj
            dst.parent.mkdir(parents=True, exist_ok=True)
            body = {"src": str(src.resolve()), "dst": str(dst.resolve()), **settings,
                    "target_vertices": vertices_for(size)}
            result = await self._retopo(part, body)
            stats = {key: result.get(key) for key in ("vertices", "faces", "has_uv", "seconds", "warning")}
            return {"size": size, "obj": obj, "stats": stats}

        made = await variants.make_each(ctx, variants.to_make(prop, params), 1.0, make)
        ctx.report(1.0, "Clean meshes saved.")
        return {"variants": [mesh for _size, mesh in made], "method": paths.IMESH, "source": glb,
                "saved_from": "auto"}

    @staticmethod
    async def _retopo(ctx: StageContext, body: dict[str, Any]) -> dict[str, Any]:
        """One blocking request to the launcher (the solve cannot report progress), so it shows time passing."""
        message = "Retopologizing and unwrapping UVs"
        ctx.report(0.02, message + "...")
        request = asyncio.ensure_future(_post(ctx.client, f"{ctx.base_url}/spellcast/retopo",
                                              body, WORK_TIMEOUT))
        started = time.monotonic()
        try:
            while not request.done():
                await asyncio.wait({request}, timeout=1.0)
                elapsed = time.monotonic() - started
                # An estimate that approaches but never reaches the end: most meshes take 5-60 s.
                ctx.report(min(0.95, 0.02 + elapsed / (elapsed + 20.0)), f"{message}... {elapsed:.0f} s")
        finally:
            request.cancel()    # no-op once done; drops the request if this job itself is cancelled
        return request.result()

    # ---- extra capabilities used by server.py ------------------------------
    async def open_session(self, client: httpx.AsyncClient, base_url: str, src: Path,
                           settings: Mapping[str, Any] | None = None,
                           reuse: str | None = None) -> dict[str, Any]:
        """Load `src` into a viewer session -> {"session_id", "viewer_path", "reused", "replaced", "lost_strokes"}.

        settings: the prop's retopo settings, used as the viewer's starting values.
        reuse: the prop's previous session id. It is returned as-is (brush strokes
        intact) while it is alive and `src` has not changed since; otherwise it is
        closed and a new session is made. Pass it on every open so a prop keeps one
        session instead of piling up new ones. When it was not reused, "replaced" says why
        ("mesh_changed", "evicted" to make room for other props, "gone" e.g. after a restart) and
        "lost_strokes" whether unsaved guide strokes went with it.
        """
        if not src.is_file():
            raise StageError("Make a 3D model in the 3D stage first.")
        body: dict[str, Any] = {"src": str(src.resolve())}
        if settings:
            params = _retopo_params(settings)
            body["config"] = {"vertex_count": params["target_vertices"],
                              "pure_quad": params["pure_quad"],
                              "crease_angle": params["crease_angle"],
                              "smooth_iter": params["smooth_iter"],
                              "rosy": params["rosy"], "posy": params["posy"],
                              "extrinsic": params["extrinsic"],
                              "align_to_boundaries": params["align_to_boundaries"]}
        if reuse:
            body["reuse"] = reuse
        data = await _post(client, f"{base_url}/spellcast/session", body, OPEN_TIMEOUT)
        return {"session_id": data["session_id"], "viewer_path": data["viewer_path"],
                "reused": bool(data.get("reused")), "replaced": data.get("replaced"),
                "lost_strokes": bool(data.get("lost_strokes"))}

    async def save_session(self, client: httpx.AsyncClient, base_url: str, session_id: str,
                           dst: Path) -> dict[str, Any]:
        """Write the viewer's current result (brush strokes included) to `dst` as OBJ with UVs.

        Returns {"path", "vertices", "faces", "has_uv", "warning", "seconds", "settings"}: settings are
        the viewer's own (the VIEWER_SETTINGS keys, target_vertices, pure_quad...) that made this mesh,
        read from the session itself. UVs use the viewer's own UV slider. Raises StageError (with a
        message for the artist) when the session no longer exists: sessions die with the Instant
        Meshes process, after 3 idle hours, or when a 9th prop's session evicts an older one (one
        without unsaved guide strokes when there is such a session).
        """
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"{base_url}/spellcast/session/{quote(session_id, safe='')}/export"
        result = await _post(client, url, {"dst": str(dst.resolve())}, WORK_TIMEOUT)
        return {**result, "settings": accepted_settings(result.get("settings"))}

    async def import_mesh(self, client: httpx.AsyncClient, base_url: str, src: Path, dst: Path,
                          overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Convert the artist's own model `src` (FBX, OBJ, glTF...) into the canonical GLB `dst` that
        retopo, the viewers and the bake read, guessing what each texture is (base colour, normal...).

        overrides: the artist's corrections, {texture key: {"role", "flip_green", "invert"}}.
        Returns {"glb", "seconds", "vertices", "faces", "warnings",
                 "materials": [{"name", "maps": [{"key", "file", "role", "label", "confidence",
                                                  "y_convention", "y_confident", "evidence"}]}]}.
        """
        body = {"src": str(src.resolve()), "dst": str(dst.resolve()), "overrides": dict(overrides or {})}
        return await _post(client, f"{base_url}/spellcast/import", body, IMPORT_TIMEOUT, IMPORT_TOO_SLOW)

    async def unwrap(self, client: httpx.AsyncClient, base_url: str, src: Path, dst: Path,
                     uv_leniency: float = UV_LENIENCY) -> dict[str, Any]:
        """UV-unwrap the low poly `src` (an OBJ LATO.2 or TriFlow made) into `dst` (OBJ with UVs): its faces
        and positions stay as they are, nothing is remeshed. uv_leniency 0..1 as in the viewer.
        Returns {"vertices", "faces", "has_uv", "warning", "seconds"}: has_uv False, with the warning,
        when unwrapping failed (the mesh is then written without UVs)."""
        try:
            leniency = max(0.0, min(float(uv_leniency), 1.0))
        except (TypeError, ValueError):
            leniency = UV_LENIENCY
        dst.parent.mkdir(parents=True, exist_ok=True)
        body = {"src": str(src.resolve()), "dst": str(dst.resolve()), "uv_leniency": leniency}
        return await _post(client, f"{base_url}/spellcast/unwrap", body, WORK_TIMEOUT, UNWRAP_TOO_SLOW)
