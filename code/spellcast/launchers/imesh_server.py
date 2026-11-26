"""Instant Meshes server for Spellcast3D. Runs INSIDE the imesh venv (code/venvs/imesh).

    <imesh venv>\\Scripts\\python.exe code\\spellcast\\launchers\\imesh_server.py --port 7773 --threads -1
        [--projects-root <projects>]

Serves instant_meshes_brush's own FastAPI app (the /viewer brush UI and its
WebSocket, no gradio) with the exact uvicorn settings of server.run(), plus the
hub routes:

    GET  /spellcast/health                  -> {"ready": true}
    POST /spellcast/shutdown {}             graceful stop, as CTRL_BREAK (see the route)
    POST /spellcast/retopo                  headless retopo + UV unwrap of a mesh file -> OBJ
    POST /spellcast/unwrap                  UV unwrap alone of a low poly another tool made (LATO.2,
                                            TriFlow): its faces and positions kept -> OBJ with UVs
    POST /spellcast/session                viewer session with a mesh file loaded (the prop's settings
                                            start a new one; a reused one keeps its viewer's own)
                                            -> {session_id, viewer_path, reused, replaced, lost_strokes}
    POST /spellcast/session/{id}/export     the session's current result (brush strokes included) -> OBJ,
                                            and the viewer settings that made it
    POST /spellcast/bake                    start baking a high poly onto a low poly's UVs -> {task_id}
    GET  /spellcast/bake/{id}               -> {state, fraction, message, result | detail}
    POST /spellcast/bake/{id}/cancel        stop it; its unfinished folder is deleted
    POST /spellcast/import                  the artist's own model -> canonical GLB (waits for it)

Error responses carry {"detail": "<message for an artist>"}; the hub shows it verbatim.

Bakes and imports run in a child process each (bake_worker.py in this folder, same venv): a
native crash cannot take the viewer sessions (and their unsaved strokes) down, the gigabytes a
4096 px bake needs go back to Windows when it ends, and cancelling is a kill. The child shares
this process's Windows job object, so it dies with the hub; a graceful stop kills it too (the
app's shutdown does).

Viewer sessions live in memory only: they are lost when this process stops, one
nobody has looked at for SESSION_TTL_SECONDS is reaped, and when MAX_SESSIONS are
open one is closed to make room: the least recently used one without unsaved guide
strokes, and only when every session has some, the least recently used one (an open
viewer touches its session every frame, so neither ever hits a session someone is
looking at).
GET /viewer without ?session= opens one shared "standalone" session, so the
tool's bare UI URL works too.

Stopping: CTRL_BREAK_EVENT (uvicorn handles SIGBREAK) or POST /spellcast/shutdown
shuts down gracefully, which stops the solvers and deletes the sessions'
%TEMP%\\imb-* export folders.

Every request must be addressed to 127.0.0.1 / localhost (_loopback_guard.py: no DNS
rebinding), and with --projects-root the hub routes read and write only inside the
hub's projects folder.

Imports only what the imesh venv has: never the hub package.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

import instant_meshes_brush as imb
from _loopback_guard import LoopbackGuard, is_loopback   # launchers/ (this script's folder)
from instant_meshes_brush import assets, server, uv
from instant_meshes_brush.session_manager import (
    BrushSession,
    Extraction,
    SessionClosedError,
    SessionError,
    SessionLimitError,
    SessionRegistry,
    config_from_mapping,
)

LOG = logging.getLogger("spellcast.imesh")


def _join_islands(positions: Any, vmapping: Any, triangles: Any, coordinates: Any) -> tuple[Any, Any, Any] | None:
    """uv.ATLAS_REFINE: xatlas's small islands joined onto their neighbours (uv_islands); None when none joins, or
    when the join fails (it never costs an unwrap: xatlas's own atlas stays)."""
    import uv_islands                                    # launchers/ (numpy: imported where it is used)

    started = time.perf_counter()
    try:
        vmapping, triangles, coordinates, joined = uv_islands.join_small_charts(
            positions, vmapping, triangles, coordinates)
    except Exception:
        LOG.exception("the small UV islands were left as xatlas made them")
        return None
    if not joined:
        return None
    LOG.info("joined %d small UV islands onto their neighbours (%.1f s)", joined, time.perf_counter() - started)
    return vmapping, triangles, uv_islands.packing_units(coordinates)


if not hasattr(uv, "ATLAS_REFINE"):
    LOG.warning("this Instant Meshes package has no uv.ATLAS_REFINE: the small UV islands stay as xatlas makes them "
                "until Spellcast3D's installer brings the package up to date")
uv.ATLAS_REFINE = _join_islands        # every unwrap made here: Retopo's three methods and the viewer's own

# Upstream reaps sessions after 30 idle minutes, sized for throwaway browser
# tabs; the hub keeps one session per prop and an artist may leave it for a
# while, so the TTL is raised. The count stays at upstream's 8 because RAM is
# the limit: a session holding a Pixal3D mesh (~555k working vertices) measured
# ~0.5 GB resident (~0.9 GB while loading), so 8 cap it near 4 GB on a 32 GB
# machine that may also hold Pixal3D. Evicting the least recently used session
# beats refusing a new one: the artist cannot close sessions themselves.
SESSION_TTL_SECONDS = 3 * 60 * 60
MAX_SESSIONS = 8

# Keep in sync with adapters/imesh.py (which checks first, with friendlier words).
MIN_TARGET_VERTICES = 100
MAX_TARGET_VERTICES = 100_000
MAX_SMOOTH_ITER = 10

EXPORT_ATTEMPTS = 3

SESSION_GONE = ("This Instant Meshes session no longer exists: sessions are lost when "
                "Instant Meshes restarts, after 3 hours without use, or when more than "
                f"{MAX_SESSIONS} props are open in it. Select the prop in the Retopo stage again "
                "to start a new one (brush strokes that were not accepted are lost).")


UV_LENIENCY = 1.0                       # the hub's projects.UV_LENIENCY: xatlas's largest islands
uv.DEFAULT_LENIENCY = UV_LENIENCY       # a viewer session's own (its page's UV slider follows the session's)


class RetopoRequest(BaseModel):
    src: str
    dst: str
    target_vertices: int = Field(5000, ge=MIN_TARGET_VERTICES, le=MAX_TARGET_VERTICES)
    pure_quad: bool = False
    crease_angle: float = Field(-1.0, le=180.0)      # degrees; negative = no creases
    smooth_iter: int = Field(2, ge=0, le=MAX_SMOOTH_ITER)
    uv_leniency: float = Field(UV_LENIENCY, ge=0.0, le=1.0)
    rosy: Literal[2, 4, 6] = 4                       # the viewer's Output mesh: 4/4 quads, 6/3 triangles,
    posy: Literal[3, 4] = 4                          # 2/4 quad-dominant
    extrinsic: bool = True                           # the viewer's "Smooth flow"
    align_to_boundaries: bool = False                # the viewer's "Follow borders"
    deterministic: bool = False


class UnwrapRequest(BaseModel):
    src: str                                        # the low poly (OBJ, or any mesh file this tool reads)
    dst: str                                        # the .obj to write, with UVs
    uv_leniency: float = Field(UV_LENIENCY, ge=0.0, le=1.0)


class SessionRequest(BaseModel):
    src: str
    config: dict[str, Any] | None = None    # instant_meshes_brush Config keys (vertex_count, pure_quad, ...)
    reuse: str | None = None                # the prop's previous session id, if any


class ExportRequest(BaseModel):
    dst: str
    uv_leniency: float | None = Field(None, ge=0.0, le=1.0)   # None = the viewer's UV slider


# With --projects-root: the hub's projects folder, the only place the hub routes read and write.
PROJECTS_ROOT: Path | None = None


def _inside_projects(path: Path) -> None:
    if PROJECTS_ROOT is not None and not path.is_relative_to(PROJECTS_ROOT):
        raise HTTPException(400, "Spellcast3D only reads and writes meshes inside its own projects folder.")


def _source_file(raw: str) -> Path:
    path = Path(raw).resolve()
    _inside_projects(path)
    if not path.is_file():
        raise HTTPException(400, f"The mesh file {path} does not exist. Make the 3D model "
                                 f"again in the 3D stage.")
    if path.suffix.lower() not in assets.SUFFIXES:
        raise HTTPException(400, f"Instant Meshes cannot read '{path.suffix}' files. "
                                 f"Use one of: {', '.join(sorted(assets.SUFFIXES))}.")
    return path


def _target_file(raw: str) -> Path:
    path = Path(raw).resolve()
    _inside_projects(path)
    if path.suffix.lower() != ".obj":
        raise HTTPException(400, f"Instant Meshes saves OBJ files only (got '{path.name}').")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(400, f"Cannot create the folder {path.parent}: {exc}. Check that "
                                 f"the disk is not full or read-only.") from exc
    return path


def _tmp_beside(dst: Path) -> Path:
    # Same folder, so os.replace is atomic.
    return dst.with_name(f"{dst.stem}.tmp{os.getpid()}{dst.suffix}")


def _write_atomic(dst: Path, write: Any) -> None:
    """write(tmp) next to dst, then replace dst with it."""
    tmp = _tmp_beside(dst)
    try:
        write(tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _file_signature(path: Path) -> tuple[str, int, int]:
    """Identifies one version of a file: the hub regenerates meshes under the same name."""
    stat = path.stat()
    return os.path.normcase(str(path)), stat.st_mtime_ns, stat.st_size


def _unreadable(path: Path, exc: Exception) -> HTTPException:
    return HTTPException(422, f"Instant Meshes could not use {path.name}; the file may be damaged. "
                              f"Try generating the 3D mesh again. (Details: {exc})")


def _failed(what: str, exc: Exception) -> HTTPException:
    """An unexpected failure, logged with its traceback and reported in plain words."""
    LOG.exception("%s failed", what)
    if isinstance(exc, MemoryError):
        return HTTPException(500, f"Instant Meshes ran out of memory while {what}. Lower the target "
                                  f"vertex count or close other programs, then try again.")
    return HTTPException(500, f"Instant Meshes failed while {what} ({type(exc).__name__}: {exc}). "
                              f"Try a different target vertex count; the Instant Meshes log has details.")


def _faces_written(face_count: int, layout: uv.UvLayout | None) -> int:
    """Faces in the written OBJ, as a 3D program counts them: a quad that a UV seam runs through
    diagonally is written as its two triangles (uv.UvLayout.cut)."""
    return face_count + (layout.cut_count if layout is not None else 0)


def _retopo(src: Path, dst: Path, req: RetopoRequest) -> dict[str, Any]:
    """The whole headless pipeline, on ONE worker thread (a core Session is single-threaded)."""
    started = time.perf_counter()
    try:
        # Geometry only: a headless retopo never shows the maps, so none are decoded or classified.
        vertices, faces = assets.solver_mesh(assets.load_source(src, materials=False))
    except assets.AssetError as exc:
        raise _unreadable(src, exc) from exc

    core = imb.Session()
    core.set_mesh(vertices, faces)
    del vertices, faces
    core.preprocess(imb.Config(
        vertex_count=req.target_vertices, rosy=req.rosy, posy=req.posy, crease_angle=req.crease_angle,
        extrinsic=req.extrinsic, align_to_boundaries=req.align_to_boundaries,
        pure_quad=req.pure_quad, smooth_iter=req.smooth_iter, deterministic=req.deterministic,
    ))
    core.solve_all()
    error = core.status.error
    if error:
        raise HTTPException(500, f"The Instant Meshes solver failed ({error}). Try a different "
                                 f"target vertex count.")
    mesh = core.extract()

    warning = ""
    layout = None
    if uv.available():
        try:
            layout = uv.unwrap(mesh.vertices, mesh.faces, leniency=req.uv_leniency)
        except uv.UnwrapError as exc:
            warning = (f"The mesh was saved without UVs because unwrapping failed ({exc}). "
                       f"Try another UV leniency or target vertex count.")
    else:
        warning = "The mesh was saved without UVs: the xatlas package is missing. Reinstall Spellcast3D."
    # Python's writer also without UVs: it keeps the quads, writes one face per face (so the count
    # below stays true) and opens paths with any letters (the C++ writer fails on non-ASCII folders).
    _write_atomic(dst, lambda tmp: uv.write_obj(tmp, mesh.vertices, mesh.faces, layout))
    has_uv = layout is not None

    return {
        "vertices": int(mesh.vertices.shape[0]),
        "faces": _faces_written(int(mesh.faces.shape[0]), layout),
        "has_uv": has_uv,
        "warning": warning,
        "seconds": round(time.perf_counter() - started, 2),
    }


def _unwrap(src: Path, dst: Path, leniency: float) -> dict[str, Any]:
    """UV-unwrap a low poly as it is (no solve: its triangles, in its frame and scale, are the result) -> OBJ.
    Corners at the same place are welded first (a triangle soup would unwrap as one chunk per triangle)."""
    started = time.perf_counter()
    try:
        vertices, faces = assets.solver_mesh(assets.load_source(src, materials=False))
    except assets.AssetError as exc:
        raise _unreadable(src, exc) from exc
    warning = ""
    layout = None
    if uv.available():
        try:
            layout = uv.unwrap(vertices, faces, leniency=leniency)
        except uv.UnwrapError as exc:
            warning = (f"The mesh was saved without UVs because unwrapping failed ({exc}). "
                       f"Try another UV leniency.")
    else:
        warning = "The mesh was saved without UVs: the xatlas package is missing. Reinstall Spellcast3D."
    _write_atomic(dst, lambda tmp: uv.write_obj(tmp, vertices, faces, layout))
    return {
        "vertices": int(vertices.shape[0]),
        "faces": _faces_written(int(faces.shape[0]), layout),
        "has_uv": layout is not None,
        "warning": warning,
        "seconds": round(time.perf_counter() - started, 2),
    }


async def _export_current(session: BrushSession,
                          leniency: float) -> tuple[Path | None, Extraction, uv.UvLayout | None, str]:
    """(file, extraction, UV layout or None, warning) of the session's current result, as the viewer's Export.

    Extract (waiting out a running solve), cut the UV atlas (cached when the viewer
    already cut one at this leniency), write. The viewer shares the session: a stroke
    or a new resolution landing meanwhile drops the extraction, so a pass that saw
    the result change underneath it is repeated. Without UVs nothing is written here
    (file None): the caller writes the extraction itself, because the session's own
    writer is then the C++ one, which cannot open a %TEMP% with non-ASCII letters.
    """
    for _ in range(EXPORT_ATTEMPTS):
        warning = ""
        layout = None
        if not session.has_extraction:
            await session.extract()
        if uv.available():
            try:
                layout = await session.unwrap(leniency)
            except SessionError as exc:
                if not session.has_extraction:
                    continue
                warning = f"The mesh was saved without UVs because unwrapping failed ({exc})."
        else:
            warning = "The mesh was saved without UVs: the xatlas package is missing. Reinstall Spellcast3D."
        exported = await session.export_mesh("obj") if layout is not None else None
        extraction = await session.extraction()
        if extraction is not None and session.uv_layout is layout:
            return exported, extraction, layout, warning
    raise HTTPException(409, "The mesh kept changing while it was being saved. Wait until the "
                             "viewer has finished solving, then save again.")


def _viewer_settings(config: dict[str, Any], leniency: float) -> dict[str, Any]:
    """The session's settings in the hub's words (prop.retopo.settings): what its viewer panel shows,
    and so what made the mesh it exports. The hub checks them again (adapters/imesh.py accepted_settings)."""
    names = {"vertex_count": "target_vertices", "pure_quad": "pure_quad", "crease_angle": "crease_angle",
             "smooth_iter": "smooth_iter", "rosy": "rosy", "posy": "posy", "extrinsic": "extrinsic",
             "align_to_boundaries": "align_to_boundaries"}
    out: dict[str, Any] = {hub: config[key] for key, hub in names.items() if key in config}
    out["uv_leniency"] = float(leniency)
    return out


async def _strokes_of(session: BrushSession) -> tuple[Any, ...]:
    """What identifies the session's guide strokes right now (empty: none)."""
    if session.closed or not session.ready:
        return ()
    try:
        return tuple(sorted((s["id"], s["kind"], s["n_points"]) for s in await session.strokes()))
    except (SessionError, KeyError, TypeError):
        return ()


def add_spellcast_routes(app: FastAPI, registry: SessionRegistry) -> None:
    # Sessions loaded through /spellcast/session: id -> (signature of the file, session).
    opened: dict[str, tuple[tuple[str, int, int], BrushSession]] = {}
    standalone: dict[str, str] = {}      # "id" -> the session behind a bare /viewer
    saved: dict[str, tuple[Any, ...]] = {}    # session id -> its strokes when it was last exported
    evicted: dict[str, bool] = {}        # sessions closed to make room -> whether unsaved strokes were lost
    retopo_lock = asyncio.Lock()         # one headless retopo or unwrap at a time: RAM, not CPU, is scarce

    def forget(sid: str) -> None:
        opened.pop(sid, None)
        saved.pop(sid, None)

    async def new_session() -> BrushSession:
        for sid in [sid for sid, (_, s) in opened.items() if s.closed or sid not in registry]:
            forget(sid)
        if len(registry) >= registry.max_sessions:
            await registry.sweep()
        if len(registry) >= registry.max_sessions and opened:
            # Least recently used first (last_used is read directly: registry.get() would refresh it),
            # but a session whose guide strokes are saved (or that has none) goes before one whose
            # strokes exist only here: the artist cannot see or close sessions themselves.
            by_age = sorted(opened, key=lambda sid: opened[sid][1].last_used)
            unsaved = {sid: strokes for sid in by_age
                       if (strokes := await _strokes_of(opened[sid][1])) and strokes != saved.get(sid)}
            victim = next((sid for sid in by_age if sid not in unsaved), by_age[0])
            LOG.info("closing session %s to make room%s", victim,
                     " (its unsaved guide strokes are lost)" if victim in unsaved else "")
            evicted[victim] = victim in unsaved
            while len(evicted) > 200:
                del evicted[next(iter(evicted))]
            forget(victim)
            await registry.close_session(victim)
        try:
            return await registry.create()
        except SessionLimitError as exc:
            raise HTTPException(503, f"Instant Meshes already holds {registry.max_sessions} "
                                     f"retopology sessions. Restart Instant Meshes from its "
                                     f"status pill to free them (unsaved brush strokes are "
                                     f"lost).") from exc

    @app.exception_handler(RequestValidationError)
    async def readable_validation_error(request: Request, exc: RequestValidationError):
        if not request.url.path.startswith("/spellcast/"):
            return await request_validation_exception_handler(request, exc)
        problems = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', ())[1:]) or 'request'}: "
                             f"{e.get('msg', '')}" for e in exc.errors())
        return JSONResponse({"detail": f"Instant Meshes rejected the settings ({problems})."},
                            status_code=422)

    @app.middleware("http")
    async def bare_viewer(request: Request, call_next):
        """/viewer needs ?session=; without one, open the shared standalone session."""
        if request.url.path == "/viewer" and "session" not in request.query_params:
            sid = standalone.get("id")
            if sid is None or registry.get(sid) is None:
                try:
                    sid = (await new_session()).id
                except HTTPException as exc:
                    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
                standalone["id"] = sid
            return RedirectResponse(f"/viewer?session={sid}", status_code=307)
        return await call_next(request)

    @app.get("/spellcast/health")
    async def health() -> dict[str, bool]:
        return {"ready": True}

    @app.post("/spellcast/shutdown")
    async def shutdown(_: dict[str, Any] = Body(...)) -> dict[str, bool]:
        """What CTRL_BREAK does, for a parent whose console events cannot reach us.

        A process started with CREATE_NO_WINDOW has a console of its own, so the
        hub's CTRL_BREAK_EVENT never arrives. Raising the signal here runs uvicorn's
        own handler: graceful shutdown after this reply. The JSON body is required so
        a web page cannot trigger this without a CORS preflight (which is refused).
        """
        signal.raise_signal(getattr(signal, "SIGBREAK", signal.SIGTERM))
        return {"stopping": True}

    @app.post("/spellcast/retopo")
    async def retopo(req: RetopoRequest) -> dict[str, Any]:
        src = _source_file(req.src)
        dst = _target_file(req.dst)
        async with retopo_lock:
            LOG.info("retopo %s -> %s (%d vertices)", src, dst, req.target_vertices)
            try:
                result = await asyncio.to_thread(_retopo, src, dst, req)
            except HTTPException:
                raise
            except Exception as exc:
                raise _failed(f"retopologizing {src.name}", exc) from exc
        LOG.info("retopo done: %s", result)
        return result

    @app.post("/spellcast/unwrap")
    async def unwrap(req: UnwrapRequest) -> dict[str, Any]:
        src = _source_file(req.src)
        dst = _target_file(req.dst)
        async with retopo_lock:
            LOG.info("unwrap %s -> %s (leniency %.2f)", src, dst, req.uv_leniency)
            try:
                result = await asyncio.to_thread(_unwrap, src, dst, req.uv_leniency)
            except HTTPException:
                raise
            except Exception as exc:
                raise _failed(f"unwrapping the UVs of {src.name}", exc) from exc
        LOG.info("unwrap done: %s", result)
        return result

    @app.post("/spellcast/session")
    async def open_session(req: SessionRequest) -> dict[str, Any]:
        """A reused session answers reused=True, as the artist left it: its settings are the viewer
        panel's own (the prop's settings, `config`, only start a new session). Otherwise `replaced` says
        what happened to the `reuse` session: "mesh_changed", "evicted" (lost_strokes: it had unsaved
        guide strokes) or "gone" (unknown here, e.g. Instant Meshes restarted); None when no `reuse` was
        given."""
        src = _source_file(req.src)
        if req.config:
            try:
                config_from_mapping(req.config)
            except (SessionError, TypeError, ValueError) as exc:
                raise HTTPException(400, f"Instant Meshes rejected the settings ({exc}).") from exc
        signature = _file_signature(src)
        if req.reuse in opened:
            loaded_from, previous = opened.pop(req.reuse)
            if loaded_from == signature and registry.get(req.reuse) is previous and previous.ready:
                opened[req.reuse] = (loaded_from, previous)
                return {"session_id": req.reuse, "viewer_path": f"/viewer?session={req.reuse}",
                        "reused": True, "replaced": None, "lost_strokes": False}
            # The prop's mesh was regenerated (or the session died): its strokes fit a mesh that is gone.
            replaced = "mesh_changed" if loaded_from != signature else "gone"
            forget(req.reuse)
            await registry.close_session(req.reuse)
        elif req.reuse:
            replaced = "evicted" if req.reuse in evicted else "gone"
        else:
            replaced = None

        session = await new_session()
        try:
            # The same path the viewer's own upload and LOAD_MESH{path} take.
            source = await asyncio.to_thread(server.read_source, src)
            vertices, faces = await asyncio.to_thread(assets.solver_mesh, source)
            await session.load_mesh(vertices, faces, src.name, req.config, source=source)
        except Exception as exc:
            await registry.close_session(session.id)
            if isinstance(exc, (SessionError, assets.AssetError)):
                raise _unreadable(src, exc) from exc
            raise _failed(f"loading {src.name}", exc) from exc
        opened[session.id] = (signature, session)
        LOG.info("session %s opened on %s", session.id, src)
        return {"session_id": session.id, "viewer_path": f"/viewer?session={session.id}",
                "reused": False, "replaced": replaced,
                "lost_strokes": replaced == "evicted" and evicted.get(req.reuse or "", False)}

    @app.post("/spellcast/session/{session_id}/export")
    async def export_session(session_id: str, req: ExportRequest) -> dict[str, Any]:
        # Only the sessions the hub opened for its props (not ones made through the viewer's own API).
        session = registry.get(session_id) if session_id in opened else None
        if session is None:
            raise HTTPException(404, SESSION_GONE)
        if not session.ready:
            raise HTTPException(409, "This Instant Meshes session has no mesh loaded yet. Select "
                                     "the prop in the Retopo stage again.")
        dst = _target_file(req.dst)
        started = time.perf_counter()
        leniency = session.uv_leniency if req.uv_leniency is None else req.uv_leniency
        try:
            exported, extraction, layout, warning = await _export_current(session, leniency)
        except HTTPException:
            raise
        except SessionClosedError as exc:
            raise HTTPException(404, SESSION_GONE) from exc
        except SessionError as exc:
            raise HTTPException(409, f"Instant Meshes could not export this session ({exc}). "
                                     f"Wait for the viewer to finish solving and try again.") from exc
        except Exception as exc:
            raise _failed("exporting the viewer's mesh", exc) from exc
        try:
            if exported is not None:
                await asyncio.to_thread(_write_atomic, dst, lambda tmp: shutil.copyfile(exported, tmp))
            else:
                await asyncio.to_thread(_write_atomic, dst, lambda tmp: uv.write_obj(
                    tmp, extraction.vertices, extraction.faces, None))
        except OSError as exc:
            raise HTTPException(500, f"Could not write {dst}: {exc}. Check that the disk is not "
                                     f"full and the file is not open in another program.") from exc
        result = {
            "path": str(dst),
            "vertices": int(extraction.vertices.shape[0]),
            "faces": _faces_written(int(extraction.faces.shape[0]), layout),
            "has_uv": layout is not None,
            "warning": warning,
            "seconds": round(time.perf_counter() - started, 2),
            "settings": _viewer_settings(session.config, leniency),
        }
        saved[session_id] = await _strokes_of(session)
        LOG.info("session %s exported: %s", session_id, result)
        return result


# ---- bakes and model imports: bake_worker.py child processes -----------------------------------------
#
# The worker is started as  python bake_worker.py <request.json>  with, for a bake,
#   {"cmd": "bake", "high": <mesh file>, "low": <low-poly OBJ/GLB>, "out_dir": <.../bake_NNN.tmp>,
#    "final_dir": <.../bake_NNN>, "settings": {...}, "threads": N}
# (it writes into out_dir and renames it to final_dir when everything is written), and for an import
#   {"cmd": "import", "src": <model file>, "dst": <.glb to write>, "overrides": {...}}.
# It prints JSON lines on stdout: {"progress": 0..1, "message": str} while it works, then
# {"result": {...}} (exit code 0) or {"error": "<message for an artist>", "kind": "user"|"internal"}
# (exit code 1). A bake result names its files relative to the bake folder:
#   {"glb": "projection.glb", "fbx": "projection.fbx"|null, "obj": "export/projection.obj"|null,
#    "maps": [{"id": "basecolor", "label": "Base colour", "file": "maps/basecolor.png",
#              "colorspace": "srgb"|"linear"}, ...], "stats": {...}}
# Anything else it prints, and all of its stderr, goes to this tool's log.

BAKE_WORKER = Path(__file__).resolve().with_name("bake_worker.py")
KEEP_BAKES = 2                  # bake_NNN folders kept per prop: the newest, and the one before it
BAKE_HISTORY = 16               # finished bake tasks whose status can still be asked for
IMPORT_TIMEOUT_SECONDS = 10 * 60
LOW_POLY_SUFFIXES = frozenset({".obj", ".glb"})
_BAKE_DIR = re.compile(r"^bake_(\d+)(\.tmp)?$")

RUNNING, DONE, ERROR, CANCELLED = "running", "done", "error", "cancelled"


class BakeRequest(BaseModel):
    high: str                   # the prop's high-poly mesh
    low: str                    # its low poly (retopo OBJ)
    out_root: str               # the prop's projection folder; each bake gets its own bake_NNN/ in it
    settings: dict[str, Any]    # checked by the hub's adapter and again by the worker
    threads: int = Field(-1, ge=-1)     # -1 = every core


class ImportRequest(BaseModel):
    src: str                    # the model file (FBX, OBJ, glTF...) with its textures beside it
    dst: str                    # the .glb to write
    overrides: dict[str, Any] = Field(default_factory=dict)    # the artist's corrections of texture roles


@dataclass
class WorkerOutcome:
    returncode: int
    result: Any = None
    error: str | None = None
    kind: str | None = None     # "user": the input's fault; "internal": a bug or crash


def _spawn_worker(worker: Path, request_file: Path) -> subprocess.Popen[str]:
    if not worker.is_file():
        raise HTTPException(500, f"The bake program is missing ({worker}). Run tools/update.bat or download "
                                 f"Spellcast3D again.")
    # stderr is inherited: it lands in this tool's log.
    return subprocess.Popen([sys.executable, str(worker), str(request_file)], cwd=str(worker.parent),
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True, encoding="utf-8",
                            errors="replace")


def _read_worker(proc: subprocess.Popen[str], progress: Callable[[float, str], None]) -> WorkerOutcome:
    """Follow a worker's JSON lines until it exits."""
    outcome = WorkerOutcome(returncode=-1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        try:
            message = json.loads(line) if line else None
        except ValueError:
            message = None
        if not isinstance(message, dict):
            if line:
                LOG.info("bake worker: %s", line)
        elif "progress" in message:
            try:
                progress(min(1.0, max(0.0, float(message["progress"]))), str(message.get("message") or ""))
            except (TypeError, ValueError):
                LOG.info("bake worker: %s", line)
        elif "result" in message:
            outcome.result = message["result"]
        elif "error" in message:
            outcome.error, outcome.kind = str(message["error"]), str(message.get("kind") or "internal")
            LOG.log(logging.ERROR if outcome.kind == "internal" else logging.INFO, "bake worker: %s", outcome.error)
    outcome.returncode = proc.wait()
    return outcome


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    """taskkill /T /F: the worker and anything it started, then make sure it is gone."""
    if proc.poll() is None:
        taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
        subprocess.run([str(taskkill), "/T", "/F", "/PID", str(proc.pid)], stdin=subprocess.DEVNULL,
                       capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(10)


def _remove_tree(folder: Path) -> None:
    """Delete a folder a killed worker wrote; Windows releases its files a moment after it dies."""
    for _ in range(20):
        shutil.rmtree(folder, ignore_errors=True)
        if not folder.exists():
            return
        time.sleep(0.1)
    LOG.warning("could not delete %s", folder)


def _bake_numbers(out_root: Path) -> list[tuple[int, Path]]:
    if not out_root.is_dir():
        return []
    return sorted((int(m.group(1)), entry) for entry in out_root.iterdir()
                  if entry.is_dir() and (m := _BAKE_DIR.match(entry.name)))


def _bake_outputs(folder: Path, raw: Any) -> dict[str, Any]:
    """A bake result with the worker's paths (relative to the bake folder) made absolute and checked.
    Raises ValueError saying what is wrong."""
    root = folder.resolve()

    def file(rel: Any) -> str:
        path = (root / str(rel)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"{rel} is missing")
        return str(path)

    try:
        maps = [{"id": str(m["id"]), "label": str(m.get("label") or m["id"]),
                 "colorspace": str(m.get("colorspace") or "linear"), "path": file(m["file"])}
                for m in raw.get("maps") or []]
        return {"dir": str(root), "glb": file(raw["glb"]),
                "fbx": None if raw.get("fbx") is None else file(raw["fbx"]),
                "obj": None if raw.get("obj") is None else file(raw["obj"]),
                "maps": maps, "stats": raw.get("stats") if isinstance(raw.get("stats"), dict) else {}}
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("its result is incomplete") from exc


@dataclass
class BakeTask:
    id: str
    proc: subprocess.Popen[str]
    out_root: Path
    tmp_dir: Path               # where the worker writes
    final_dir: Path             # what the worker renames it to when it is done
    state: str = RUNNING
    fraction: float = 0.0
    message: str = "Starting the bake…"
    result: dict[str, Any] | None = None
    detail: str | None = None

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"state": self.state, "fraction": round(self.fraction, 4), "message": self.message}
        if self.state == DONE:
            out["result"] = self.result
        elif self.state == ERROR:
            out["detail"] = self.detail
        return out


class BakeWorkers:
    """The bake tasks of this process (one runs at a time: a bake takes gigabytes of RAM and every core)
    and the synchronous model imports. Each is a `worker` child process."""

    def __init__(self, worker: Path) -> None:
        self.worker = worker
        self.tasks: dict[str, BakeTask] = {}
        self.lock = threading.Lock()
        self.import_lock = asyncio.Lock()   # imports one at a time too (RAM)

    def start(self, high: Path, low: Path, out_root: Path, settings: dict[str, Any], threads: int) -> BakeTask:
        with self.lock:
            if any(task.state == RUNNING for task in self.tasks.values()):
                raise HTTPException(409, "Instant Meshes is already baking. Wait for that bake to finish (or "
                                         "cancel it), then try again.")
            numbers = _bake_numbers(out_root)
            for _, entry in numbers:
                if entry.suffix == ".tmp":      # a bake that never finished (Instant Meshes was killed)
                    shutil.rmtree(entry, ignore_errors=True)
            number = max((n for n, _ in numbers), default=0) + 1
            tmp_dir, final_dir = out_root / f"bake_{number:03d}.tmp", out_root / f"bake_{number:03d}"
            tmp_dir.mkdir(parents=True)
            request = {"cmd": "bake", "high": str(high), "low": str(low), "out_dir": str(tmp_dir),
                       "final_dir": str(final_dir), "settings": settings,
                       "threads": threads if threads > 0 else (os.cpu_count() or 4)}
            request_file = tmp_dir / "request.json"
            request_file.write_text(json.dumps(request, indent=1), encoding="utf-8")
            try:
                proc = _spawn_worker(self.worker, request_file)
            except BaseException:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise
            task = BakeTask(id=secrets.token_hex(6), proc=proc, out_root=out_root, tmp_dir=tmp_dir,
                            final_dir=final_dir)
            self.tasks[task.id] = task
            finished = [tid for tid, t in self.tasks.items() if t.state != RUNNING]
            for tid in finished[:max(0, len(self.tasks) - BAKE_HISTORY)]:
                del self.tasks[tid]
        LOG.info("bake %s: %s onto %s -> %s", task.id, high, low, final_dir)
        threading.Thread(target=self._follow, args=(task,), name=f"bake-{task.id}", daemon=True).start()
        return task

    def get(self, task_id: str) -> BakeTask:
        task = self.tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "This bake is not known to Instant Meshes (it restarted since). Bake again.")
        return task

    def cancel(self, task_id: str) -> BakeTask:
        with self.lock:
            task = self.get(task_id)
            if task.state != RUNNING:
                return task
            task.state, task.message = CANCELLED, "Cancelled."
        _kill_tree(task.proc)
        _remove_tree(task.tmp_dir)
        _remove_tree(task.final_dir)    # renamed just before the kill: the bake still did not finish
        LOG.info("bake %s cancelled", task.id)
        return task

    def stop_all(self) -> None:
        for task in list(self.tasks.values()):
            self.cancel(task.id)

    def _follow(self, task: BakeTask) -> None:
        def progress(fraction: float, message: str) -> None:
            if task.state == RUNNING:
                task.fraction, task.message = fraction, message or task.message

        outcome = _read_worker(task.proc, progress)
        if task.state != RUNNING:
            return                      # cancelled: cancel() cleans up
        result: dict[str, Any] | None = None
        detail = outcome.error or (f"The bake stopped unexpectedly (exit code {outcome.returncode}). The Instant "
                                   f"Meshes log has the details.")
        if outcome.returncode == 0 and outcome.error is None and outcome.result is not None:
            try:
                result = _bake_outputs(task.final_dir, outcome.result)
            except ValueError as exc:
                detail = f"The bake ended, but {exc}. Bake again; the Instant Meshes log has the details."
        # The folders are settled before the state says the bake ended.
        if result is not None:
            for _, entry in _bake_numbers(task.out_root)[:-KEEP_BAKES]:
                shutil.rmtree(entry, ignore_errors=True)
        else:
            _remove_tree(task.tmp_dir)
            _remove_tree(task.final_dir)
        with self.lock:
            if task.state != RUNNING:
                return
            if result is not None:
                task.result, task.state, task.fraction, task.message = result, DONE, 1.0, "Baked."
            else:
                task.state, task.detail = ERROR, detail
        LOG.info("bake %s %s: %s", task.id, task.state, task.final_dir if result is not None else detail)

    def run_import(self, src: Path, dst: Path, overrides: dict[str, Any]) -> dict[str, Any]:
        """Convert `src` into `dst` (written only when the conversion succeeds) -> the worker's result."""
        tmp = _tmp_beside(dst)
        request_file = dst.with_name(f"{dst.stem}.request{os.getpid()}.json")
        request_file.write_text(json.dumps({"cmd": "import", "src": str(src), "dst": str(tmp),
                                            "overrides": overrides}), encoding="utf-8")
        timed_out = threading.Event()
        try:
            proc = _spawn_worker(self.worker, request_file)

            def stop() -> None:
                timed_out.set()
                _kill_tree(proc)

            timer = threading.Timer(IMPORT_TIMEOUT_SECONDS, stop)
            timer.start()
            try:
                outcome = _read_worker(proc, lambda fraction, message: None)
            finally:
                timer.cancel()
            if timed_out.is_set():
                raise HTTPException(504, f"Converting {src.name} took longer than {IMPORT_TIMEOUT_SECONDS // 60} "
                                         f"minutes, so it was stopped. Try the model with fewer polygons or "
                                         f"smaller textures.")
            if outcome.returncode != 0 or outcome.error is not None or not isinstance(outcome.result, dict) \
                    or not tmp.is_file():
                raise HTTPException(422 if outcome.kind == "user" else 500, outcome.error or (
                    f"Converting {src.name} stopped unexpectedly (exit code {outcome.returncode}). The Instant "
                    f"Meshes log has the details."))
            os.replace(tmp, dst)
        finally:
            tmp.unlink(missing_ok=True)
            request_file.unlink(missing_ok=True)
        return {**outcome.result, "glb": str(dst)}


def _inside_file(raw: str, suffixes: frozenset[str], what: str) -> Path:
    """An existing input file inside the projects folder with one of `suffixes`."""
    path = Path(raw).resolve()
    _inside_projects(path)
    if not path.is_file():
        raise HTTPException(400, f"{what} ({path.name}) does not exist any more.")
    if path.suffix.lower() not in suffixes:
        raise HTTPException(400, f"{what} must be one of {', '.join(sorted(suffixes))} (got '{path.name}').")
    return path


def add_bake_routes(app: FastAPI, workers: BakeWorkers) -> None:
    """The bake / import routes. A running bake is killed when the app shuts down: a graceful stop
    (/spellcast/shutdown, CTRL_BREAK) reaches only this process, and uvicorn ends it by re-raising the
    signal, which exits at once, so this has to happen in the app's own shutdown."""
    app_lifespan = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(asgi_app: Any) -> AsyncIterator[Any]:
        async with app_lifespan(asgi_app) as state:
            try:
                yield state
            finally:
                await asyncio.to_thread(workers.stop_all)

    app.router.lifespan_context = lifespan

    @app.post("/spellcast/bake")
    async def bake(req: BakeRequest) -> dict[str, str]:
        high = _inside_file(req.high, assets.SUFFIXES, "The high-poly mesh")
        low = _inside_file(req.low, LOW_POLY_SUFFIXES, "The low-poly model")
        out_root = Path(req.out_root).resolve()
        _inside_projects(out_root)
        task = await asyncio.to_thread(workers.start, high, low, out_root, req.settings, req.threads)
        return {"task_id": task.id}

    @app.get("/spellcast/bake/{task_id}")
    async def bake_status(task_id: str) -> dict[str, Any]:
        return workers.get(task_id).status()

    @app.post("/spellcast/bake/{task_id}/cancel")
    async def bake_cancel(task_id: str, _: dict[str, Any] = Body(...)) -> dict[str, Any]:
        # The JSON body keeps other web pages out (a CORS preflight, which is refused), as for /shutdown.
        return (await asyncio.to_thread(workers.cancel, task_id)).status()

    @app.post("/spellcast/import")
    async def import_model(req: ImportRequest) -> dict[str, Any]:
        src = _inside_file(req.src, assets.SUFFIXES, "The model file")
        dst = Path(req.dst).resolve()
        _inside_projects(dst)
        if dst.suffix.lower() != ".glb":
            raise HTTPException(400, f"An imported model is saved as a .glb file (got '{dst.name}').")
        dst.parent.mkdir(parents=True, exist_ok=True)
        async with workers.import_lock:
            LOG.info("import %s -> %s", src, dst)
            result = await asyncio.to_thread(workers.run_import, src, dst, req.overrides)
        LOG.info("import done: %s (%s s)", dst, result.get("seconds"))
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Instant Meshes server for Spellcast3D.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--threads", type=int, default=-1, help="solver threads; -1 = all cores")
    parser.add_argument("--projects-root", default=None,
                        help="the hub's projects folder: the /spellcast routes read and write only inside it")
    args = parser.parse_args(argv)
    global PROJECTS_ROOT
    PROJECTS_ROOT = Path(args.projects_root).resolve() if args.projects_root else None

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s")
    imb.set_thread_count(args.threads)

    registry = SessionRegistry(ttl_seconds=SESSION_TTL_SECONDS, max_sessions=MAX_SESSIONS)
    app = server.build_app(registry)
    add_spellcast_routes(app, registry)
    add_bake_routes(app, BakeWorkers(BAKE_WORKER))

    print(f"Instant Meshes for Spellcast3D on http://{args.host}:{args.port}/viewer", flush=True)
    # server.run owns the uvicorn settings (WebSocket frame size, no permessage-deflate).
    server.run(args.host, args.port, app=LoopbackGuard(app) if is_loopback(args.host) else app)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
