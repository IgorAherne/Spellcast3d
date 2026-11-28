"""Pixal3D adapter: chosen image (or the prop's views) -> textured high-poly GLB (stage "mesh", GPU, GTX 10xx+).

The tool process is launchers/pixal3d_server.py running in the pixal3d venv. It serves Pixal3D's own page at "/"
plus the /spellcast/* routes used here: GET health, POST batch (all the props of a job in one request), GET
batch/{id} (where every prop is) and POST batch/{id}/cancel.

The props of a job are made together (run_batch). Pixal3D takes all of them through one step - cutting out, the
rough shape, its details, the texture, the mesh - before it reads the models of the next step from disk, so each
model is read once per job and never more than one is in memory. The launcher orders the props (those made from
several views first); a prop "rests" between its steps while the others have their turn, and each finished model
is saved, and patched into its prop, at once. The hub only asks every second where the props are.

A prop with views (projects.py images.views: parts of its pictures, each showing one side) is made from
them while its chosen picture is one of theirs (projects.effective_views): one view alone like a picture,
two or more together by Pixal3D's multi-view models, facing the front view, at the prop's quality (High, 1536, by
default; the staged engine makes it within the card's memory). Those models are an optional download
(multiview.py); without them, the front view alone. mesh.source records what the model was made from (projects.mesh_source).

Each prop's shape is made by the geometry its settings pick (projects.py mesh.settings.geometry, sent as "geometry"):
Hi3D v1.0, which refines Pixal3D's coarse shape in the same launcher, or Pixal3D's own networks; the texture is
Pixal3D's either way, and mesh.geometry records which one made the model. Hi3D's packages come with Pixal3D's install
(install/components.py), its networks only from the 3D panel's Download; a prop set to Hi3D while they are not on
this PC fails with the hub's words (HI3D_MISSING).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, NamedTuple

import httpx

from .. import gpu, paths
from ..install import models
from ..projects import (GEOMETRIES, VIEW_NAMES, VIEW_ORDER, ProjectError, effective_views, mesh_geometry, mesh_source,
                        tileable_prop)
from .base import (PROP_CANCELLED, RESTING, WAITING, WORKING, BatchContext, BatchItem, LaunchSpec, StageError,
                   ToolAdapter, ToolNeedsRestart)

log = logging.getLogger(__name__)

READY_TIMEOUT = 300.0              # it is ready once its libraries are imported: models are read when a job needs them
POLL_INTERVAL = 1.0
# One poll of the batch. The answer can take seconds: the launcher's HTTP threads share Python's lock with the
# CPU-only stretches of saving a mesh (the UV unwrap takes 16 s, each BVH build 5 s).
POLL_TIMEOUT = httpx.Timeout(10.0, connect=3.0)
START_TIMEOUT = httpx.Timeout(60.0, connect=10.0)     # POST batch: answered before any work starts
STALL_TIMEOUT = 45 * 60.0          # nothing moved in the batch for this long: Pixal3D hangs
# ... or after it took a cancel. Not while it builds or saves a mesh (the step "finish"): the decode and the UV
# unwrap of a bulky prop can keep Python busy for minutes without a word, and a mesh being saved is kept.
CANCEL_STALL = 180.0
ATTN_BACKENDS = ("flash_attn", "sdpa", "xformers")   # keep in sync with launchers/pixal3d_server.py
MAX_SEED = 2 ** 31 - 1             # Pixal3D's own (app.py MAX_SEED, the largest int32)
# In the job's work folder: the prop on Pixal3D's card and the last one it made, as it writes them itself
# (launchers/pixal3d_batch.py ON_CARD; keep in sync). Read when Pixal3D stopped without a last answer.
ON_CARD = "on_card.json"
# The camera of each view around the prop, in degrees (Pixal3D's multi-view rig: 90 looks at the prop's
# own left side, which then faces image-left).
VIEW_AZIMUTH = {"front": 0, "left": 90, "back": 180, "right": 270}
MODES = ("single", "multi")
NO_FRONT = "Mark one of the slices as Front: the 3D model is built facing it."
TILEABLE = ("This prop is a tileable texture, which gets no 3D model. To make one from its picture, untick Tileable "
            "in Images first.")
FRONT_ONLY = "Used the front view only: the multi-view models aren’t downloaded yet."
MADE_SLOWLY = "Made slowly: this prop needed more graphics memory than is free."
# The words for the failure kinds "mv_missing" and "oom" are the hub's, not the launcher's (pixal3d_mv.MV_MISSING,
# pixal3d_server.OUT_OF_GPU_MEMORY, which its own route and log use): the hub knows where the artist downloads the
# models, and the prop's quality (no "lower Quality" at the lowest one). So is "missing_file" for a prop set to Hi3D
# while Hi3D's networks are not on this PC (models.hi3d_on_disk): the 3D stage downloads them (its note under Shape).
MV_MISSING = ("Pixal3D’s multi-view models are missing, so this prop could not be made from its views. Download "
              "them again in the 3D stage, or clear the views to use one picture.")
HI3D_NOT_HERE = ("Hi3D is not part of this Spellcast3D yet, so this prop could not be made with Hi3D. Pick Pixal3D "
                 "under Shape in the 3D stage.")
HI3D_MISSING = ("Hi3D’s AI models are not on this PC yet, so this prop could not be made with Hi3D. Under Shape in the "
                "3D stage, press “Download” (or wait for the download to finish), or pick Pixal3D.")
OUT_OF_VIDEO_MEMORY = "This prop needs more video memory than the card has free."
CLOSE_GPU_PROGRAMS = "Close other programs that use the graphics card"
NO_GLB = ("Pixal3D reported success but no GLB file appeared. Check the Pixal3D log (Settings > Debugging) and "
          "try again.")


class _Request(NamedTuple):
    """What one prop asks of Pixal3D (Pixal3DAdapter._request)."""
    body: dict[str, Any]            # its entry in the batch request, without the id
    out: Path                       # mesh/mesh.glb of the prop
    partial: Path                   # where Pixal3D writes: beside `out`, which stays until the new one is complete
    warning: str | None             # what the artist should know about how it is made


def _calibration_due(info: gpu.GpuInfo | None) -> dict[str, str]:
    """Whether the next start runs the decode-in-boxes check (install/components.py pixal3d_step_due): the launcher's
    words for a prop made at less detail without it promise that only then (pixal3d_server.CALIBRATION_DUE_ENV)."""
    if info is None:
        return {}
    try:
        from ..install.components import pixal3d_step_due

        return {"SPELLCAST_PIXAL3D_CALIBRATION_DUE": "1" if pixal3d_step_due("decode_products", info) else "0"}
    except Exception:       # noqa: BLE001 - never in the way of the launch: the words then name the Settings button
        log.exception("could not tell whether the next start measures the graphics card")
        return {}


class Pixal3DAdapter(ToolAdapter):
    tool_id = paths.PIXAL3D
    display_name = "Pixal3D"
    role = "3D generator"
    stage = "mesh"
    batch_stages = ("mesh",)
    uses_gpu = True
    gpu_at_start = False    # the launcher only imports its libraries; the card is touched when a job begins
    ui_path = "/"

    @functools.cached_property
    def _gpu(self) -> gpu.GpuInfo | None:
        # nvidia-smi takes a moment and the card does not change while the hub runs
        return gpu.primary_gpu()

    def supported(self) -> tuple[bool, str]:
        return gpu.pixal3d_supported(self._gpu)

    def check_settings(self, stage: str, settings: dict[str, Any]) -> None:
        """The geometry a job asks for must be one there is (a prop's own is repaired when its project is read)."""
        if "geometry" in settings and settings["geometry"] not in GEOMETRIES:
            raise StageError("The shape must be made by Hi3D or Pixal3D. Pick it again under Shape in 3D.")

    # ---- process lifecycle ----------------------------------------------
    def launch_spec(self, port: int, settings: dict[str, Any]) -> LaunchSpec:
        attn = (settings.get("pixal3d") or {}).get("attn_backend")
        sage = (settings.get("pixal3d") or {}).get("sage", "auto")
        int8 = (settings.get("pixal3d") or {}).get("int8", "auto")
        coarse_exact = (settings.get("pixal3d") or {}).get("coarse_exact")
        return LaunchSpec(
            argv=[str(paths.venv_python(self.tool_id)), str(paths.LAUNCHERS / "pixal3d_server.py"),
                  "--repo", str(paths.PIXAL3D_DIR),
                  "--port", str(port),
                  # "auto" (or a typo): flash_attn, which the fork's precision profile replaces on a card without
                  # its kernels (xformers from sm_75, torch's SDPA below: pixal3d/modules/precision.py)
                  "--attn-backend", attn if attn in ATTN_BACKENDS else "flash_attn",
                  # SageAttention for the flows' self-attention (sm_80+, -20 % of a batch): "auto" leaves the
                  # fork's own default (on after its self-check); false turns it off (flash_attn, bit for bit)
                  "--sage", "off" if sage is False else ("on" if sage is True else "auto"),
                  # The flows' linear layers as int8 tensor-core GEMMs (sm_80+): "auto" leaves the fork's own
                  # default (on after its self-check); false keeps the bf16 GEMMs, bit for bit
                  "--int8", "off" if int8 is False else ("on" if int8 is True else "auto"),
                  # The conditioning and the coarse stages as the original computes them (sm_80+), unless the 3D
                  # panel's "Faster 3D generation" is on (pixal3d.coarse_exact false)
                  "--coarse-exact", "off" if coarse_exact is False else "on",
                  "--projects-root", str(paths.PROJECTS)],
            cwd=paths.PIXAL3D_DIR,
            env=paths.clean_env(self.tool_id, extra={**gpu.pixal3d_card_env(self._gpu), **_calibration_due(self._gpu)}),
            ready_timeout=READY_TIMEOUT,
        )

    async def health(self, client: httpx.AsyncClient, base_url: str) -> bool:
        try:
            r = await client.get(f"{base_url}/spellcast/health", timeout=5.0)
            return r.status_code == 200 and r.json().get("ready") is True
        except (httpx.HTTPError, ValueError, AttributeError):
            return False

    # ---- work -------------------------------------------------------------
    async def run_batch(self, ctx: BatchContext, items: list[BatchItem]) -> None:
        """Every prop's chosen picture, or its views while it is one of theirs (projects.effective_views) ->
        its mesh/mesh.glb, all in one request to Pixal3D. A prop the hub itself finds wrong (no picture, no
        front view, a setting that is no number) fails here and is not sent; the others do not wait for it."""
        several_ok = True
        if any(len(effective_views(item.prop.get("images") or {})) >= 2 for item in items):
            several_ok = await self._multiview_ready(ctx)
        props = {item.prop_id: item for item in items}
        requests: dict[str, _Request] = {}
        for item in items:
            try:
                requests[item.prop_id] = self._request(item.prop, item.params, item.prop_dir, several_ok)
            except StageError as e:
                ctx.fail(item.prop_id, str(e))
        if not requests:
            return
        try:
            try:
                await self._start(ctx, requests)
            except ToolNeedsRestart:
                raise
            except StageError as e:         # Pixal3D refused the request as a whole
                for prop_id in requests:
                    ctx.fail(prop_id, str(e))
                return
            await self._follow(ctx, props, requests)
        finally:
            for prop_id, request in requests.items():
                _remove(request.partial)        # of a prop that was not made: a made one's is mesh.glb by now
                _tidy_deleted(props[prop_id], request)

    async def _multiview_ready(self, ctx: BatchContext) -> bool:
        """Whether props with several views can be made from them: the multi-view models are on disk, and
        the running Pixal3D sees them too (its health says so). Else each is made from its front view."""
        from .. import multiview   # not at the top: multiview imports procman, which imports the adapters
        if not multiview.available():
            return False
        try:
            r = await ctx.client.get(f"{ctx.base_url}/spellcast/health", timeout=5.0)
            return r.json().get("multiview") is not False
        except (httpx.HTTPError, ValueError, AttributeError):
            return True     # no answer: the batch request says what is wrong with the tool

    def _request(self, prop: dict[str, Any], params: dict[str, Any], prop_dir: Path,
                 several_ok: bool = True) -> _Request:
        """What Pixal3D is asked for one prop. One view: that part of its picture alone. Two or more: one model
        from all of them, facing the front view (needed), at the prop's quality (1536 by default, the staged engine
        makes it within the card's memory); without the multi-view models (several_ok false), from the front view
        alone. The artist hears of both. A seed below
        0 asks for a random one, which is drawn here and written into `params` (the job's BatchItem.params, the
        same dict for its second attempt). Raises StageError for what the artist has to fix first."""
        if tileable_prop(prop):
            raise StageError(TILEABLE)
        images = prop.get("images") or {}
        views = sorted((v for v in effective_views(images) if v.get("view") in VIEW_AZIMUTH),
                       key=lambda v: VIEW_ORDER.index(v["view"]))     # the front first
        warning: str | None = None
        if len(views) >= 2:
            if views[0]["view"] != "front":
                raise StageError(NO_FRONT)
            if not several_ok:
                views, warning = views[:1], FRONT_ONLY
        sent = [{"image_path": str(_picture(prop_dir, v["src"], f"The picture of the {VIEW_NAMES[v['view']]} view "
                                                                f"({v['src']}) is missing from the project folder. "
                                                                f"Mark that view again in Images.")),
                 "rect": v.get("rect"), "azimuth": VIEW_AZIMUTH[v["view"]]} for v in views]
        if sent:
            image = sent[0]["image_path"]
        else:
            chosen = images.get("chosen")
            if not chosen:
                raise StageError("This prop has no image yet: import or generate one in Images first.")
            image = str(_picture(prop_dir, chosen, f"The chosen image ({chosen}) is missing from the project "
                                                   "folder. Choose another image or import it again in Images."))
        try:
            seed = int(params.get("seed", 0))
            if seed < 0:
                # A random seed is drawn here, once, and kept in the prop's params, which the job's second attempt
                # (after a restart) is given again: Pixal3D resumes a prop's sampled work only for the same seed.
                seed = params["seed"] = random.randint(0, MAX_SEED)
            body: dict[str, Any] = {
                "image_path": image,
                "seed": seed,
                "resolution": int(params.get("resolution", 1536)),
                "decimation_target": int(params.get("decimation_target", 300_000)),
                "texture_size": int(params.get("texture_size", 2048)),
                # the views Pixal3D takes, the front first; views_dir: where it keeps the framed pictures it
                # made of them (what the model saw)
                "views": sent or None,
                "views_dir": str((prop_dir / "mesh" / "views").resolve()) if sent else None,
                # whose networks make the shape: "hi3d" | "pixal3d" (Pixal3D's for settings that name none)
                "geometry": mesh_geometry(params),
            }
        except (TypeError, ValueError):
            raise StageError("A mesh setting is not a number. Check seed, quality, detail and "
                             "texture size of this prop in 3D.") from None
        out = prop_dir / "mesh" / "mesh.glb"
        out.parent.mkdir(parents=True, exist_ok=True)
        for stale in out.parent.glob("mesh.*.partial.glb"):   # left over if the hub was closed mid-run
            _remove(stale)
        # Pixal3D writes next to the final file; the old mesh stays until the new one is complete.
        partial = out.with_name(f"mesh.{uuid.uuid4().hex[:8]}.partial.glb")
        body["out_path"] = str(partial.resolve())
        return _Request(body, out, partial, warning)

    def _result(self, prop: dict[str, Any], request: _Request, result: dict[str, Any]) -> dict[str, Any]:
        """The patch of a prop Pixal3D made (`result`: its entry's "result" in the batch). Everything the
        artist should know about it is one sentence each in stats.warning: the hub's, Pixal3D's, and that a
        prop which did not fit the card's free memory took longer."""
        used = len(request.body["views"] or ()) or 1
        warnings = [w for w in (request.warning, result.get("warning"), MADE_SLOWLY if result.get("slow") else None)
                    if isinstance(w, str) and w]
        # Pixal3D counts the seconds of its own run: a prop it picked up again after a restart would show a
        # fraction of the time it really took, so it shows none.
        stats = {"seconds": None if result.get("resumed") else result.get("seconds"),
                 "seed": result.get("seed", request.body["seed"]),
                 "views": _count(result.get("views"), used),
                 "mode": result["mode"] if result.get("mode") in MODES else "multi" if used > 1 else "single",
                 "warning": " ".join(warnings) or None,
                 "ss_retries": _count(result.get("ss_retries"), 0)}
        return {"glb": f"mesh/{request.out.name}", "source": mesh_source(prop.get("images") or {}), "stats": stats,
                "geometry": request.body["geometry"]}

    async def _start(self, ctx: BatchContext, requests: dict[str, _Request]) -> None:
        """POST /spellcast/batch. Raises ToolNeedsRestart when Pixal3D cannot take it now (no answer, or it
        is still at another batch: one the hub gave up on), StageError with its words when it refuses it."""
        own = ctx.settings.get("pixal3d") or {}
        body = {"batch_id": ctx.job_id, "work_dir": str(ctx.work_dir.resolve()),
                # How many props go through the steps together; 0 (or nothing set): all of them. settings.py
                # keeps it a whole number from 0 up.
                "wave_size": own.get("wave_size") or 0,
                # after a fresh start: props whose texture was sampled are only decoded and saved again
                "resume": ctx.attempt > 1,
                # Off unless the artist allowed it: a prop that does not fit the card's free memory gets less
                # detail by itself, and fails rather than spill into shared memory (slow, and it can freeze the PC).
                "spill": own.get("allow_spill") is True,
                "items": [{"id": prop_id, **request.body} for prop_id, request in requests.items()]}
        try:
            r = await ctx.client.post(f"{ctx.base_url}/spellcast/batch", json=body, timeout=START_TIMEOUT)
        except httpx.HTTPError as e:
            raise ToolNeedsRestart(f"Pixal3D did not take the job ({type(e).__name__}).") from None
        if r.status_code in (200, 202):
            return
        data = _json(r)
        if data.get("kind") == "busy":
            raise ToolNeedsRestart("Pixal3D is still working on another job.")
        detail = data.get("detail")
        raise StageError(detail if isinstance(detail, str) and detail else
                         f"Pixal3D failed (HTTP {r.status_code}). Check the Pixal3D log (Settings > Debugging) "
                         f"and try again.")

    async def _follow(self, ctx: BatchContext, props: dict[str, BatchItem], requests: dict[str, _Request]) -> None:
        """Poll the batch until it ended, telling ctx where each prop is and finishing each one the moment
        Pixal3D has saved it. A cancel is passed on with every poll until Pixal3D says it took it.

        A poll without an answer in time says nothing (POLL_TIMEOUT); the process is taken as gone when it
        cannot be connected to twice in a row or the process manager says so, without waiting for a watchdog.
        Raises ToolNeedsRestart then, and when the batch stands still for too long (its "seq" counts every
        change, down to the bars of a mesh being saved); _stopped says which prop it blames.

        A prop the artist deletes meanwhile (its folder went to deleted/) is cancelled with the next poll: Pixal3D
        cannot be told to drop one prop, but it writes nothing more for it and fails it at its next step."""
        url = f"{ctx.base_url}/spellcast/batch/{ctx.job_id}"
        open_ids = set(requests)            # not finished, failed or cancelled yet
        current: str | None = None          # the prop on the card, by the last answer
        finishing = False                   # the last answer's step was "finish": a mesh is built or saved
        seq: Any = None
        moved = time.monotonic()            # when seq last changed
        unreachable = 0
        acknowledged = False                # Pixal3D took the cancel

        async def stopped(why: str, fresh: bool = False) -> ToolNeedsRestart:
            return await self._stopped(ctx, props, requests, open_ids, current, why, fresh)

        while True:
            if ctx.cancelled() and not acknowledged:
                await self._cancel(ctx, url)
            if not ctx.tool_alive():
                raise await stopped("Pixal3D stopped.")
            record: dict[str, Any] | None = None
            try:
                r = await ctx.client.get(url, timeout=POLL_TIMEOUT)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                unreachable += 1
                if unreachable >= 2:
                    raise await stopped("Pixal3D is not reachable any more.") from None
            except httpx.HTTPError:
                unreachable = 0             # too slow an answer, or a dropped one: the next poll tells
            else:
                unreachable = 0
                if r.status_code == 404:    # it started afresh, or never heard of this batch
                    raise await stopped("Pixal3D does not know this job any more.")
                record = _json(r) if r.status_code == 200 else None
            if record:
                now = time.monotonic()
                if record.get("seq") != seq:
                    seq, moved = record.get("seq"), now
                state = record.get("state")
                if not acknowledged and (record.get("cancelling") is True or state != "running"):
                    acknowledged, moved = True, now
                current = record["current"] if isinstance(record.get("current"), str) else None
                part, loading, told = record.get("part"), record.get("loading"), record.get("items")
                finishing = isinstance(part, dict) and part.get("key") == "finish"
                entries = {prop_id: entry for prop_id, entry in (told if isinstance(told, dict) else {}).items()
                           if prop_id in open_ids and isinstance(entry, dict)}
                # The props Pixal3D has made first: saving one takes a moment, in which the job still shows the
                # last answer. Then everything else this answer says, in one go.
                for prop_id, entry in entries.items():
                    if entry.get("state") == "done":
                        result = entry.get("result")
                        await self._complete(ctx, props[prop_id], requests[prop_id],
                                             result if isinstance(result, dict) else {})
                        open_ids.discard(prop_id)
                ctx.set_part(part if isinstance(part, dict) else None, current,
                             loading if isinstance(loading, str) and loading else None)
                self._drop_deleted(ctx, props, open_ids)
                for prop_id, entry in entries.items():
                    if prop_id in open_ids and self._tell(ctx, prop_id, requests[prop_id], entry):
                        open_ids.discard(prop_id)
                if state != "running":
                    error = record.get("error")
                    if state == "failed" and isinstance(error, dict) and error.get("kind") == "restart":
                        raise await stopped("Pixal3D has to start again.", fresh=True)
                    for prop_id in sorted(open_ids):    # the batch ended without a word about them
                        if state == "cancelled" or ctx.cancelled():
                            ctx.update(prop_id, PROP_CANCELLED, 0.0, "")
                        else:
                            ctx.fail(prop_id, _failure(error, requests[prop_id]))
                    return
            else:
                self._drop_deleted(ctx, props, open_ids)
            still = time.monotonic() - moved
            if acknowledged and still > (STALL_TIMEOUT if finishing else CANCEL_STALL):
                raise await stopped("Pixal3D did not stop after the cancel.")
            if still > STALL_TIMEOUT:
                await self._cancel(ctx, url)
                raise await stopped(f"Pixal3D made no progress for {STALL_TIMEOUT / 60:.0f} minutes.")
            await asyncio.sleep(POLL_INTERVAL)

    async def _stopped(self, ctx: BatchContext, props: dict[str, BatchItem], requests: dict[str, _Request],
                       open_ids: set[str], current: str | None, why: str, fresh: bool = False) -> ToolNeedsRestart:
        """Pixal3D stopped, or has to: the exception that says so, blaming the prop that was on the card.

        Only an answer that says so itself is `fresh` (a batch that failed with the kind "restart"). Otherwise
        the last answer can be old - no poll is answered while native code saves a mesh - and Pixal3D's own note
        in the work folder (ON_CARD) says which prop was on the card when it stopped; the last answer's `current`
        counts only when there is no note. The note also names the prop Pixal3D made last: when the hub has not
        seen it made yet and its file is there, it is saved now instead of being made again. A prop that has
        ended for the hub (its model saved, failed, cancelled) is never blamed."""
        note = None if fresh else await _on_card(ctx.work_dir)
        if note is not None:
            made = note.get("finished")
            made_id = made.get("id") if isinstance(made, dict) else None
            if made_id in open_ids and isinstance(made_id, str):
                result = made.get("result") if isinstance(made.get("result"), dict) else {}
                request = requests[made_id]
                # the note of this very request (a second attempt writes its model to another file)
                if result.get("glb_path") in (None, request.body["out_path"]) and request.partial.is_file():
                    log.info("Pixal3D stopped right after it saved prop %s; its model is kept.", made_id)
                    await self._complete(ctx, props[made_id], request, result)
                    open_ids.discard(made_id)
            current = note.get("current") if isinstance(note.get("current"), str) else None
        return ToolNeedsRestart(why, blame=current if current in open_ids else None)

    @staticmethod
    def _drop_deleted(ctx: BatchContext, props: dict[str, BatchItem], open_ids: set[str]) -> None:
        """The props the artist deleted since the last look (their folders went to deleted/) end cancelled, as a
        prop deleted while its model was saved does; what Pixal3D says of them later is not listened to."""
        for prop_id in sorted(open_ids):
            if not props[prop_id].prop_dir.is_dir():
                log.info("Prop %s was deleted while Pixal3D made it: it is cancelled.", prop_id)
                ctx.update(prop_id, PROP_CANCELLED, 0.0, "")
                open_ids.discard(prop_id)

    async def _cancel(self, ctx: BatchContext, url: str) -> None:
        try:
            await ctx.client.post(f"{url}/cancel", timeout=POLL_TIMEOUT)
        except httpx.HTTPError:
            pass    # lost: it is asked again with the next poll

    @staticmethod
    def _tell(ctx: BatchContext, prop_id: str, request: _Request, entry: dict[str, Any]) -> bool:
        """Pass the batch's entry of a prop that is not made yet on to ctx -> whether the prop has ended (it
        failed or was cancelled)."""
        state = entry.get("state")
        try:
            fraction = min(1.0, max(0.0, float(entry.get("fraction") or 0.0)))
        except (TypeError, ValueError):
            fraction = 0.0
        if state == "failed":
            ctx.fail(prop_id, _failure(entry.get("error"), request))
            return True
        if state == "cancelled":
            ctx.update(prop_id, PROP_CANCELLED, 0.0, "")
            return True
        if state in (WAITING, WORKING, RESTING):
            ctx.update(prop_id, state, fraction, str(entry.get("message") or ""))
        return False

    async def _complete(self, ctx: BatchContext, item: BatchItem, request: _Request, result: dict[str, Any]) -> None:
        """Pixal3D saved the prop's model: it becomes mesh/mesh.glb and the prop is patched. Whatever goes
        wrong here is this prop's alone."""
        try:
            if not request.partial.is_file():
                if not item.prop_dir.is_dir():
                    # The prop was deleted: its folder went to deleted/ with the half-written model in it, which
                    # is of no use to anybody there.
                    for gone in (item.prop_dir.parents[1] / "deleted").glob(f"*/mesh/{request.partial.name}"):
                        _remove(gone)
                    ctx.update(item.prop_id, PROP_CANCELLED, 0.0, "")
                    return
                raise StageError(NO_GLB)
            try:
                os.replace(request.partial, request.out)
            except OSError:
                raise StageError(f"Could not save mesh/{request.out.name}: it may be open in another program. "
                                 "Close it and try again.") from None
            if not request.body["views"]:       # made from one picture: an earlier model's views are not this one's
                shutil.rmtree(request.out.parent / "views", ignore_errors=True)
            await ctx.finish(item.prop_id, self._result(item.prop, request, result))
        except ProjectError as e:
            if e.status_code == 404 and (item.prop_dir.parents[1] / "project.json").is_file():
                # The prop was deleted while Pixal3D made it, and Pixal3D made its folder again for the model:
                # nothing of it is kept, and there is nobody to tell.
                shutil.rmtree(request.out.parent, ignore_errors=True)
                _remove_dir(item.prop_dir)
                ctx.update(item.prop_id, PROP_CANCELLED, 0.0, "")
            else:
                ctx.fail(item.prop_id, str(e))
        except (StageError, OSError) as e:
            ctx.fail(item.prop_id, str(e))
        except Exception:
            log.exception("Saving the 3D model of prop %s/%s failed", ctx.project, item.prop_id)
            ctx.fail(item.prop_id, "Something went wrong while this prop's 3D model was saved. The details are in "
                                   "code/logs/hub.log.")


def _failure(error: Any, request: _Request) -> str:
    """The artist's words for a prop Pixal3D failed: `error` is its {"detail", "kind"}. For "oom" and "mv_missing"
    the hub's own (see MV_MISSING), and for "missing_file" of a prop set to Hi3D while Hi3D's networks are not here
    (HI3D_MISSING); for every other kind Pixal3D's."""
    error = error if isinstance(error, dict) else {}
    if error.get("kind") == "oom":
        return _oom_message(request.body["resolution"])
    if error.get("kind") == "mv_missing":
        return MV_MISSING
    if error.get("kind") == "missing_file" and request.body.get("geometry") == "hi3d" and not models.hi3d_on_disk():
        return HI3D_MISSING if paths.hi3d_code_here() else HI3D_NOT_HERE
    detail = error.get("detail")
    if isinstance(detail, str) and detail:
        return detail
    return "Pixal3D could not make this prop. Check the Pixal3D log (Settings > Debugging) and try again."


def _oom_message(resolution: int) -> str:
    """A prop that did not fit: Pixal3D lowered its detail by itself and it still did not (with pixal3d.allow_spill
    off, the default, it never spills into shared memory). Lowering the quality is offered while there is a lower
    one."""
    if resolution > 1024:
        return f"{OUT_OF_VIDEO_MEMORY} {CLOSE_GPU_PROGRAMS}, or lower Quality."
    return f"{OUT_OF_VIDEO_MEMORY} {CLOSE_GPU_PROGRAMS}."


async def _on_card(work_dir: Path) -> dict[str, Any] | None:
    """Pixal3D's own note of its batch in the job's work folder (ON_CARD): {"batch_id", "current": the prop on the
    card | None, "finished": {"id", "result"} of the last prop made | None}. None when there is none (Pixal3D
    stopped before its batch began, or an older launcher) or it cannot be read."""
    path = work_dir / ON_CARD
    for attempt in range(5):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except PermissionError:     # Windows: Pixal3D replaces it this very moment
            await asyncio.sleep(0.05)
            continue
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None
    return None


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _picture(prop_dir: Path, rel: str, missing: str) -> Path:
    """The absolute path of a picture of the prop (StageError `missing` when it is not there)."""
    path = prop_dir / rel
    if not path.is_file():
        raise StageError(missing)
    return path.resolve()


def _count(value: Any, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else default


def _tidy_deleted(item: BatchItem, request: _Request) -> None:
    """What is left of a prop deleted while Pixal3D made it: its half-written model, which went to deleted/ with
    its folder, and an orphan folder in its old place (only a mesh folder - the views or a model written after the
    folder had moved, by a Pixal3D of before it checked for that). A prop folder with anything else in it is a
    prop's, and stays."""
    deleted = item.prop_dir.parents[1] / "deleted"
    if not deleted.is_dir():
        return
    for gone in deleted.glob(f"*/mesh/{request.partial.name}"):
        _remove(gone)
    try:
        orphan = (item.prop_dir.is_dir() and any(deleted.glob(f"{item.prop_id}_*"))
                  and {entry.name.lower() for entry in item.prop_dir.iterdir()} <= {"mesh"})
    except OSError:
        return
    if orphan:
        log.info("Removing what Pixal3D wrote for prop %s after it was deleted.", item.prop_id)
        shutil.rmtree(item.prop_dir / "mesh", ignore_errors=True)
        _remove_dir(item.prop_dir)


def _remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass   # Pixal3D is still writing it (the hub stopped waiting); the next run removes it


def _remove_dir(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass   # not empty: something else is in it
