"""The job queue: ONE worker runs jobs first-in first-out. A job runs one stage for a list of props, with one
tool: one of paths.STAGE_TOOLS[stage]. In the Retopo stage each prop picks its tool (retopo.settings.method),
so the page queues one job per method (TriFlow, then LATO.2, then Instant Meshes): the worker then loads each
tool once, and GPU exclusivity (procman) stops the GPU tool before it.

Most tools make a job's props one after another: per prop the worker builds an adapters.base.StageContext,
calls adapter.run_stage and merges the returned patch into prop[stage] through the ProjectStore. The props
run in the order of prop_ids (the order they were chosen, unless the tool has one of its own: prop_order), and
the job notes how each one ended (Job.ended): a prop that lets go of its tool finishes behind the next one, so
they do not always end in that order.

A stage in the adapter's batch_stages (Pixal3D's mesh) is given all the job's props at once
(adapter.run_batch, with a BatchContext): the tool takes them step by step through its models, and the
adapter reports each prop as it goes. Such a job keeps the order the artist chose, and a made prop is
patched in while the others are still worked on. The tool gets a scratch folder for the job
(projects/<project>/temp/<stage>-<job id>/), which the job removes when it ends, however it ends.

Every job says where each of its props is (Job.props: waiting, working, resting between two steps, done,
failed, cancelled, with a fraction and a message): a batch job from what the adapter reports, the others
from their counters (Job.prop_states). The page reads that, never the counters. See ARCHITECTURE.md
"Hub HTTP API".

A job whose tool is still being installed in the background waits in the queue (its message says so);
later jobs for other tools run meanwhile, unless they share a prop with a waiting job (a prop's jobs
always run in the order they were submitted: its retopo needs its mesh first). A LATO.2 or TriFlow job
also needs Instant Meshes, which unwraps its clean meshes' UVs: it waits for that installation too.

Before a job runs, its tool takes the graphics card (procman.claim_gpu): the GPU tool that holds it is stopped
then, not when the job's tool starts, so a tool that touches the card only when a job begins (Pixal3D) may already
have started beside it, early.

When the tool stops in the middle of a job (it crashed, or a prop raised ToolNeedsRestart: its GPU context
broke, or it may still be at a prop the hub gave up on), it is started afresh once per job and the job goes
on with the next prop. A batch job fails the prop that was on the card then (it is not sent a second
time: a prop that crashes the tool would do it again) and sends the others once more, into the same
scratch folder, so the tool can go on from what it had saved there. A cancelled job never starts its tool
again.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from . import paths
from .adapters import base
from .adapters.base import BatchContext, BatchItem, StageContext, StageError, ToolAdapter, ToolNeedsRestart
from .procman import READY, ProcessManager, ToolError, ToolProcess
from .projects import ProjectError, ProjectStore, check_retopo_settings, chosen_tool, now_iso
from .settings import SettingsStore

log = logging.getLogger(__name__)

JOB_HISTORY = 50
IDLE_POLL = 1.0     # seconds between looks at the queued jobs while some wait for an installation
WAITING = "Waiting for the jobs before it…"
WORK_DIR_RETRY = 2.0    # seconds before a job's scratch folder is removed a second time (a file was still open)
# What a job says from a cancel until it ends (the page shows it while the job's "stopping" is true).
CANCELLING = "Cancelling… (the current prop may take a moment to stop)"
CANCELLING_BATCH = "Stopping… (a model that is being saved is finished first)"

QUEUED, STARTING_TOOL, RUNNING, DONE, ERROR, CANCELLED = (
    "queued", "starting_tool", "running", "done", "error", "cancelled")
ACTIVE = (QUEUED, STARTING_TOOL, RUNNING)
# Once a stage's job is done, the tool of the next stage is started early (procman.start_early), so that its loading
# is over by the time the artist arrives there, in the page's order: the images stage's job starts Pixal3D (beside
# the image generator; it takes the card only at its first job), the 3D stage's hands the card to Qwen for
# Omni-Texture (its load is the longest), the retopo stage's to Instant Meshes (CPU, usually running already for the
# unwrap). After HANDOFF_SECONDS, so that one more prop sent to the same stage at once does not pay for a tool
# swap, and never while any job waits or runs. An Omni-Texture job hands the card to no one: the artist goes
# on there - masks a place and has it painted again, or Omni-Textures again - so Qwen-Image stays as its last
# picture left it, and the retopology tool loads as the artist arrives at Retopo (the page's warm wish).
NEXT_STAGE = {"images": "mesh", "mesh": "omnitexture", "retopo": "projection"}
HANDOFF_SECONDS = 20.0
UNWRAPPING = "Unwrapping the UVs (xatlas, on the CPU)…"
UNWRAP_FRACTION = 0.9       # a prop's bar while its UVs are unwrapped behind the next prop (the adapters' own share)
# The states of a prop that has ended (adapters.base.PROP_STATES).
ENDED = (base.PROP_DONE, base.PROP_FAILED, base.PROP_CANCELLED)


class JobError(Exception):
    """A user-facing problem with a job request. `status_code` becomes the HTTP status."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _prop_entry(state: str, fraction: float = 0.0, message: str = "") -> dict[str, Any]:
    """One prop's entry in Job.props. A prop that has ended gets the fraction Job.prop_states gives it too:
    made or failed, there is nothing left to do (1.0); cancelled, nothing of it was kept (0.0)."""
    try:
        fraction = min(1.0, max(0.0, float(fraction)))
    except (TypeError, ValueError):
        fraction = 0.0
    if state in ENDED:
        fraction = 0.0 if state == base.PROP_CANCELLED else 1.0
    return {"state": state, "fraction": round(fraction, 4), "message": str(message)}


@dataclass
class Job:
    id: str
    stage: str
    tool: str               # the tool that runs it (one of paths.STAGE_TOOLS[stage])
    project: str
    prop_ids: list[str]
    params: dict[str, Any]
    status: str = QUEUED
    current_prop: str | None = None
    done: int = 0
    fraction: float = 0.0
    message: str = WAITING
    errors: dict[str, str] = field(default_factory=dict)
    # prop id -> what the artist should know about its finished result (StageContext.warn)
    warnings: dict[str, str] = field(default_factory=dict)
    created: str = field(default_factory=now_iso)
    finished: str | None = None
    cancel_requested: bool = False
    cut_short: int = 0      # props the cancel stopped half-way (counted in `done`, but not finished)
    prop_fraction: float = 0.0      # how far the prop being worked on is (a job that runs prop by prop)
    # The props before current_prop that let go of the tool and finish on the CPU while current_prop is on the card
    # (JobQueue._run_props: the card never waits for them), in their order, each with its fraction and words for the
    # page: UV unwraps in Instant Meshes (one after the other; UNWRAPPING), Omni-Texture's last projection and GLB.
    behind: dict[str, tuple[float, str]] = field(default_factory=dict)
    # prop id -> how it ended (PROP_DONE, PROP_FAILED, PROP_CANCELLED: cut short by a cancel), as each prop ends
    ended: dict[str, str] = field(default_factory=dict)
    # The picture the tool is making for current_prop as it looks so far (StageContext.preview): (its bytes, the step
    # it shows, the part of the prop's picture it is of - None: all of it), kept in memory only and gone when the
    # prop lets go of the tool or ends. `previews` counts them: each one gets a URL of its own (to_dict).
    preview: tuple[bytes, int, list[float] | None] | None = None
    previews: int = 0
    # A job whose tool takes all its props at once (ToolAdapter.batch_stages), from the moment it runs: where each
    # prop is, as the adapter reports it ({prop id: {"state", "fraction", "message"}}); the step the props are taken
    # through ({"key", "label", "position", "of"}); what the tool reads from disk meanwhile, in the artist's words;
    # and the name of the job's scratch folder in projects/<project>/temp/ while it has one.
    props: dict[str, dict[str, Any]] = field(default_factory=dict)
    part: dict[str, Any] | None = None
    loading: str | None = None
    work: str | None = None

    def prop_states(self) -> dict[str, dict[str, Any]]:
        """{prop id: {"state", "fraction", "message"}}: where each prop of the job is (adapters.base.PROP_STATES).
        A batch job has it from its adapter. Any other job makes its props one after another, in the order of
        prop_ids: the ones that ended say how (`ended`; failed when they have an error), the ones finishing behind
        the prop on the card say what they do (`behind`), the next one is worked on, the rest wait; what a cancel
        cut short, or a job that ended never reached, is cancelled."""
        if self.props:
            return {prop_id: dict(entry) for prop_id, entry in self.props.items()}
        ended = self._ends()
        # The prop on the card: props start in order, and each one before it has ended or finishes behind it.
        lead = self.done + len(self.behind)
        states: dict[str, dict[str, Any]] = {}
        for index, prop_id in enumerate(self.prop_ids):
            fraction, message = 0.0, ""
            if self.status in (QUEUED, STARTING_TOOL):
                state = base.WAITING
            elif self.status == RUNNING and prop_id in self.behind:
                state, (fraction, message) = base.WORKING, self.behind[prop_id]
            elif prop_id in ended:
                state = base.PROP_FAILED if prop_id in self.errors else ended[prop_id]
                fraction = 0.0 if ended[prop_id] == base.PROP_CANCELLED else 1.0
            elif self.status == RUNNING:
                state = base.WORKING if index == lead else base.WAITING
                if state == base.WORKING:
                    fraction, message = self.prop_fraction, self.message
            elif prop_id in self.errors:        # the tool could not start: every prop says why
                state = base.PROP_FAILED
            else:
                state = base.PROP_CANCELLED
            states[prop_id] = {"state": state, "fraction": round(fraction, 4), "message": message}
        return states

    def _ends(self) -> dict[str, str]:
        """`ended`; for a job that has only its counters (made without the job runner), the first done of
        prop_ids, the last cut_short of those cut short."""
        if self.ended or not self.done:
            return self.ended
        whole = self.done - self.cut_short
        return {prop_id: base.PROP_DONE if index < whole else base.PROP_CANCELLED
                for index, prop_id in enumerate(self.prop_ids[:self.done])}

    def to_dict(self) -> dict[str, Any]:
        # progress.preview only while there is a picture: GET url answers with it, and v changes with every new one
        # (and the prop it is of: the one on the card)
        preview = {} if self.preview is None else {"preview": {
            "url": f"/api/jobs/{self.id}/preview?v={self.previews}", "step": self.preview[1],
            "prop": self.current_prop, **({"box": self.preview[2]} if self.preview[2] else {})}}
        return {
            "id": self.id, "stage": self.stage, "tool": self.tool, "project": self.project,
            "prop_ids": list(self.prop_ids),
            "status": self.status, "current_prop": self.current_prop,
            "progress": {"done": self.done, "total": len(self.prop_ids),
                         "fraction": round(self.fraction, 4), "message": self.message, **preview},
            "props": self.prop_states(), "part": dict(self.part) if self.part else None,
            # What the tool is reading from disk between two props, in the artist's words (else ""), and whether
            # a cancel is on its way (progress.message says what it waits for). The page shows both.
            "loading": self.loading or "", "stopping": self.cancel_requested and self.status in ACTIVE,
            "errors": dict(self.errors), "warnings": dict(self.warnings), "created": self.created,
            "finished": self.finished,
        }


class JobQueue:
    def __init__(self, procman: ProcessManager, store: ProjectStore, settings: SettingsStore,
                 adapters: dict[str, ToolAdapter], client: httpx.AsyncClient):
        self.procman = procman
        self.store = store
        self.settings = settings
        self.adapters = adapters
        self.client = client
        # stage -> the tools here that can run it, the default first: paths.STAGE_TOOLS, of the adapters there
        # are (and that say they run the stage).
        self._stage_tools = {stage: runs for stage, tools in paths.STAGE_TOOLS.items()
                             if (runs := tuple(t for t in tools if t in adapters and stage in adapters[t].stages))}
        self._jobs: dict[str, Job] = {}          # insertion order == creation order
        self._wake = asyncio.Event()             # a job was submitted
        self._worker: asyncio.Task[None] | None = None
        self._handoff: asyncio.Task[None] | None = None      # the next stage's tool, started after a grace

    def start(self) -> None:
        self._worker = asyncio.create_task(self._work(), name="job-worker")

    async def close(self) -> None:
        for job in self._jobs.values():
            if job.status == QUEUED:
                self._finish(job, CANCELLED, "Cancelled because Spellcast3D was closed.")
        if self._worker:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
        self._drop_handoff()

    # ---- API --------------------------------------------------------------------------------
    async def submit(self, stage: Any, project: Any, prop_ids: Any, params: Any = None,
                     tool: Any = None) -> dict[str, Any]:
        """Queue `stage` for the props with `tool` (one of paths.STAGE_TOOLS[stage]). A stage whose props pick
        their tool (Retopo: retopo.settings.method) refuses props that picked another one; without `tool`, the
        props' own pick when they share one (else 400: one job per method), or the stage's only tool."""
        tools = self._stage_tools.get(stage) if isinstance(stage, str) else None
        if not tools:
            raise JobError(f'"{stage}" is not a stage that can be run. Use one of: {", ".join(self._stage_tools)}.')
        if tool is not None and tool not in tools:
            raise JobError(f'"{tool}" cannot run the {stage} stage. Use one of: {", ".join(tools)}.')
        if not isinstance(prop_ids, list) or not prop_ids or not all(isinstance(p, str) for p in prop_ids):
            raise JobError("Choose at least one prop to run.")
        if params is not None and not isinstance(params, dict):
            raise JobError('"params" must be a JSON object.')
        if stage == "retopo" and params:
            try:
                params = check_retopo_settings(params)
            except ProjectError as e:
                raise JobError(str(e)) from e
        try:
            # Off the event loop: the read waits for the project lock, which an image upload holds
            # while it writes. The job itself is created on the loop, where the worker reads the jobs.
            data = await asyncio.to_thread(self.store.get_project, project)
        except ProjectError as e:
            raise JobError(str(e), e.status_code) from e
        props = {p["id"]: p for p in data["props"]}
        prop_ids = list(dict.fromkeys(prop_ids))
        missing = [p for p in prop_ids if p not in props]
        if missing:
            raise JobError("Some of these props no longer exist. Reload the page and try again.", 404)
        settings = {prop_id: {**((props[prop_id].get(stage) or {}).get("settings") or {}), **(params or {})}
                    for prop_id in prop_ids}
        tool_id = self._tool_for(stage, tools, tool, {prop_id: props[prop_id] for prop_id in prop_ids}, settings)
        adapter = self.adapters[tool_id]
        # The order the tool takes them in, when it has one of its own (Qwen-Image: Omni-Texture's props grouped by
        # their image model, the one its launcher holds first, so the model is switched at most once)
        prop_order = getattr(adapter, "prop_order", None)
        if prop_order is not None:
            ordered = list(prop_order(stage, [props[prop_id] for prop_id in prop_ids],
                                      running=self.procman.get(tool_id).alive()))
            if sorted(ordered) == sorted(prop_ids):
                prop_ids = ordered
        for prop_id in prop_ids:
            try:
                adapter.check_settings(stage, settings[prop_id])
            except StageError as e:
                raise JobError(f"{props[prop_id].get('name') or prop_id}: {e}") from e
        self.procman.check_available(tool_id)
        if paths.IMESH in self._needs(stage, tool_id)[1:]:
            try:
                self.procman.check_available(paths.IMESH)
            except ToolError as e:
                raise ToolError(f"{adapter.display_name}'s clean meshes get their UVs from Instant Meshes: {e}",
                                e.status_code) from e
        job = Job(id="j_" + secrets.token_hex(4), stage=stage, tool=tool_id, project=project, prop_ids=prop_ids,
                  params=dict(params or {}))
        self._jobs[job.id] = job
        self._drop_handoff()        # the job's own tool takes the card: no early start of another meanwhile
        self._wake.set()
        self._prune()
        log.info("Job %s queued: %s with %s for %d prop(s) in %s", job.id, stage, tool_id, len(prop_ids), project)
        return job.to_dict()

    # ---- the next stage's tool, started early -------------------------------------------------------------
    def _drop_handoff(self) -> None:
        if self._handoff is not None and not self._handoff.done():
            self._handoff.cancel()
        self._handoff = None

    def _hand_off(self, job: Job) -> None:
        """A job that was done: after HANDOFF_SECONDS, the tool of the next stage (NEXT_STAGE) starts early."""
        stage = NEXT_STAGE.get(job.stage) if job.status == DONE else None
        tools = self._stage_tools.get(stage) if stage else None
        if not tools:
            return
        self._drop_handoff()
        self._handoff = asyncio.create_task(self._hand_off_later(job, stage, tools), name="stage-handoff")

    async def _hand_off_later(self, job: Job, stage: str, tools: tuple[str, ...]) -> None:
        await asyncio.sleep(HANDOFF_SECONDS)
        if any(j.status in ACTIVE for j in self._jobs.values()):
            return
        tool_id = tools[0]
        try:
            started = self.procman.start_early(tool_id, wish={"stage": stage}, displace=True)
        except ToolError as e:
            log.info("%s was not started early for the %s stage: %s", self._name(tool_id), stage, e)
            return
        if not started:
            return
        log.info("Job %s done: %s started early for the %s stage", job.id, self._name(tool_id), stage)
        # Once it is ready, a tool that can warms up for the stage (its first use paid while the artist is still
        # elsewhere). A job that comes first cancels this (_drop_handoff): its own first prop pays it then.
        warm = getattr(self.adapters[tool_id], "warm_up", None)
        if warm is None:
            return
        try:
            tp = await self.procman.ensure_running(tool_id, {"stage": stage})
        except ToolError:
            return
        if not any(j.status in ACTIVE for j in self._jobs.values()):
            await warm(self.client, tp.base_url, stage)

    def _tool_for(self, stage: str, tools: tuple[str, ...], tool: str | None, props: dict[str, dict[str, Any]],
                  settings: dict[str, dict[str, Any]]) -> str:
        """The tool of a new job (see submit); JobError (400) when the props do not fit it."""
        picked = {prop_id: chosen_tool(stage, settings[prop_id]) for prop_id in props}
        if tool is not None:
            wrong = [prop_id for prop_id, own in picked.items() if own is not None and own != tool]
            if wrong:
                prop = props[wrong[0]]
                raise JobError(f"{prop.get('name') or wrong[0]} is set to be made with "
                               f"{self._name(picked[wrong[0]])}, not {self._name(tool)}. Reload the page and "
                               f"try again.")
            return tool
        own = set(picked.values())
        if own == {None}:
            return tools[0]
        if len(own) > 1:
            names = ", ".join(self._name(t) for t in (t for t in paths.STAGE_TOOLS[stage] if t in own))
            raise JobError(f"These props use different methods ({names}). Start one job per method.")
        (only,) = own
        if only not in tools:
            raise JobError(f"{self._name(only)} is not part of this Spellcast3D, so it cannot run the {stage} "
                           f"stage. Pick another method for these props.")
        return only

    def _needs(self, stage: str, tool_id: str) -> tuple[str, ...]:
        """The tools a job of `stage` on `tool_id` uses: that one, and for a retopology tool other than Instant
        Meshes also Instant Meshes (StageContext.unwrap)."""
        if stage == "retopo" and tool_id != paths.IMESH and paths.IMESH in self.adapters:
            return tool_id, paths.IMESH
        return (tool_id,)

    def _name(self, tool_id: str | None) -> str:
        adapter = self.adapters.get(tool_id or "")
        return adapter.display_name if adapter is not None else str(tool_id)

    def list(self, project: str | None = None) -> list[dict[str, Any]]:
        """Most recent first, including queued and running jobs."""
        return [j.to_dict() for j in reversed(self._jobs.values()) if project is None or j.project == project]

    def get(self, job_id: str) -> dict[str, Any]:
        return self._job(job_id).to_dict()

    def preview(self, job_id: str) -> bytes | None:
        """The picture the job's tool is making, as it looks so far (Job.preview), or None."""
        preview = self._job(job_id).preview
        return None if preview is None else preview[0]

    def active_projects(self) -> set[str]:
        """Projects with a job that is waiting or running."""
        return {job.project for job in list(self._jobs.values()) if job.status in ACTIVE}

    def active_props(self, project: str, stage: str | None = None) -> set[str]:
        """Ids of the project's props with a job (of `stage`, else of any stage) that is waiting or running."""
        return {prop_id for job in list(self._jobs.values()) if job.status in ACTIVE and job.project == project
                and stage in (None, job.stage) for prop_id in job.prop_ids}

    def work_dir_in_use(self, project: str) -> str | None:
        """The name of the scratch folder (projects/<project>/temp/<name>) a running job of the project works
        in, or None: a sweep of the project's leftovers (ProjectStore.clear_work_dirs) keeps that one."""
        return next((job.work for job in list(self._jobs.values())
                     if job.work and job.project == project and job.status in ACTIVE), None)

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self._job(job_id)
        if job.status == QUEUED:
            self._finish(job, CANCELLED, "Cancelled.")
        elif job.status in (STARTING_TOOL, RUNNING):
            job.cancel_requested = True
            adapter = self.adapters.get(job.tool)
            # A tool that takes all the props at once (Pixal3D) finishes saving a model it has begun to save; any
            # other tool stops inside its prop.
            job.message = (CANCELLING_BATCH if adapter is not None and job.stage in adapter.batch_stages
                           else CANCELLING)
        return job.to_dict()

    def cancel_tool_jobs(self, tool_id: str) -> None:
        """Ask the running job of `tool_id` to stop, e.g. because the tool is being stopped."""
        for job in self._jobs.values():
            if job.tool == tool_id and job.status in (STARTING_TOOL, RUNNING):
                job.cancel_requested = True

    # ---- worker ----------------------------------------------------------------------------
    def _job(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobError("That job is not known (the list only keeps the last 50 jobs).", 404)
        return job

    def _finish(self, job: Job, status: str, message: str) -> None:
        job.status, job.message, job.current_prop = status, message, None
        job.preview = None
        job.finished = now_iso()

    def _prune(self) -> None:
        finished = [j for j in self._jobs.values() if j.status not in ACTIVE]
        for job in finished[:max(0, len(self._jobs) - JOB_HISTORY)]:
            del self._jobs[job.id]

    def _next_job(self) -> Job | None:
        """The first queued job that can run now (see the module docstring); updates the waiting ones."""
        held: set[tuple[str, str]] = set()
        for job in list(self._jobs.values()):
            if job.status != QUEUED:
                continue
            props = {(job.project, prop_id) for prop_id in job.prop_ids}
            installing = next((why for tool in self._needs(job.stage, job.tool)
                               if (why := self.procman.install_wait(tool))), None)
            if installing is None and not props & held:
                return job
            job.message = installing or WAITING
            held |= props
        return None

    async def _work(self) -> None:
        while True:
            job = self._next_job()
            if job is None:
                self._wake.clear()
                waiting = any(j.status == QUEUED for j in self._jobs.values())
                try:
                    await asyncio.wait_for(self._wake.wait(), IDLE_POLL if waiting else None)
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                await self._run(job)
            except asyncio.CancelledError:
                self._finish(job, CANCELLED, "Cancelled because Spellcast3D was closed.")
                raise
            except Exception:
                log.exception("Job %s crashed", job.id)
                self._finish(job, ERROR, "Something went wrong inside Spellcast3D. "
                                         "The details are in code/logs/hub.log.")
            self._prune()

    async def _run(self, job: Job) -> None:
        tool_id = job.tool
        adapter = self.adapters[tool_id]
        job.status, job.message = STARTING_TOOL, f"Starting {adapter.display_name}…"
        try:
            tp = await self._start_tool(job, tool_id)
        except ToolError as e:
            for prop_id in job.prop_ids:
                job.errors[prop_id] = str(e)
            self._finish(job, ERROR, str(e))
            return
        if tp is None or job.cancel_requested:
            self._finish(job, CANCELLED, "Cancelled.")
            return
        batch = job.stage in adapter.batch_stages
        job.status = RUNNING
        self.procman.set_busy(tool_id, True)
        try:
            if batch:
                await self._run_batch(job, adapter, tp)
            else:
                await self._run_props(job, adapter, tp)
        finally:
            self.procman.set_busy(tool_id, False)
        if job.cancel_requested:
            total = len(job.prop_ids)
            finished = (sum(1 for entry in job.props.values() if entry["state"] == base.PROP_DONE) if batch
                        else job.done - job.cut_short)
            self._finish(job, CANCELLED, "Cancelled." if total == 1 else
                         f"Cancelled after {finished} of {total} props. What was already finished is kept.")
        elif job.errors:
            failed = len(job.errors)
            self._finish(job, ERROR, f"{failed} of {len(job.prop_ids)} props failed. "
                                     f"Each prop shows what went wrong.")
        else:
            self._finish(job, DONE, "Done.")
        log.info("Job %s finished: %s", job.id, job.status)
        self._hand_off(job)

    async def _run_props(self, job: Job, adapter: ToolAdapter, tp: ToolProcess) -> None:
        """The job's props one after another (ToolAdapter.run_stage). A prop that lets go of the tool (its UVs
        unwrapped by Instant Meshes on the CPU, StageContext.unwrap: TriFlow's and LATO.2's meshes; or
        StageContext.release: Omni-Texture's last projection and GLB) is left to finish behind: the next prop goes to
        the card at once, and the props behind queue up, as many as there are (the Instant Meshes server takes the
        unwraps one after the other; job.behind says which). A restart of the tool, a cancel and the end of the job
        wait for them all."""
        tool_id = adapter.tool_id
        restart = False         # the tool must start afresh before the next prop (ToolNeedsRestart)
        restarted = False       # it did once in this job: a tool that keeps failing is not started again and again
        behind: list[asyncio.Task[bool]] = []        # the props before, still finishing on the CPU
        task: asyncio.Task[bool] | None = None       # the prop on the card

        async def settle() -> bool:
            """Waits for the props behind (-> whether one left the tool needing a fresh start)."""
            waited, behind[:] = list(behind), []
            return any([await t for t in waited])

        try:
            order = list(job.prop_ids)
            for index, prop_id in enumerate(order):
                if job.cancel_requested:
                    break
                if restart or not tp.alive() or tp.state != READY:
                    restart = await settle() or restart
                    fresh = None if restarted else await self._restart_tool(job, adapter)
                    restarted, restart = True, False
                    if job.cancel_requested:
                        break
                    if fresh is None:
                        job.errors[prop_id] = (f"{adapter.display_name} stopped before this prop's turn. "
                                               f"Check its log, then run this prop again.")
                        job.ended[prop_id] = base.PROP_FAILED
                        job.done += 1
                        continue
                    tp = fresh
                released = asyncio.Event()
                following = order[index + 1] if index + 1 < len(order) else None
                task = asyncio.create_task(self._run_prop(job, adapter, tp, prop_id, released, following),
                                           name=f"prop-{prop_id}")
                freed = asyncio.ensure_future(released.wait())
                try:
                    await asyncio.wait({task, freed}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    freed.cancel()
                if task.done():
                    restart = task.result()
                else:
                    behind.append(task)             # let go of the tool: the next prop takes the card now
            restart = await settle() or restart
        except asyncio.CancelledError:          # the hub closes: no prop's task is left behind
            left = [t for t in (*behind, task) if t is not None and not t.done()]
            for t in left:
                t.cancel()
            if left:
                await asyncio.gather(*left, return_exceptions=True)
            raise
        if restart:     # the last prop left the tool needing a fresh start: the next job starts it afresh
            await self.procman.stop(tool_id)

    # ---- a job whose tool takes all the props at once -------------------------------------------
    async def _run_batch(self, job: Job, adapter: ToolAdapter, tp: ToolProcess) -> None:
        """All the job's props in one request to the tool (ToolAdapter.run_batch), which reports each of them
        as it goes (Job.props). When the tool has to start afresh (ToolNeedsRestart), the prop that was on the
        card is failed and the others are sent once more, to the same scratch folder; a second time fails
        them. After a cancel (the artist's, or the tool being stopped in Settings > Debugging) nothing is
        started or sent again. The scratch folder goes when the job ends, however it ends."""
        tool_id, name = adapter.tool_id, adapter.display_name
        job.work = f"{job.stage}-{job.id}"
        job.props = {prop_id: _prop_entry(base.WAITING) for prop_id in job.prop_ids}
        job.message = f"Sending the props to {name}…"
        closing = False
        try:
            try:
                await asyncio.to_thread(self.store.clear_work_dirs, job.project, job.work)
                work = await asyncio.to_thread(self.store.work_dir, job.project, job.work)
            except (ProjectError, OSError) as e:
                words = str(e) if isinstance(e, ProjectError) else (
                    f"Spellcast3D could not make its work folder in the project ({e}). Check that the project's "
                    f"drive has free space and can be written to, then try again.")
                for prop_id in job.prop_ids:
                    self._prop_failed(job, prop_id, words)
                return
            items = await self._batch_items(job)
            for attempt in (1, 2):
                if job.cancel_requested or not items:
                    break
                try:
                    await adapter.run_batch(self._batch_context(job, tp, work, attempt), items)
                    break
                except ToolNeedsRestart as stop:
                    fresh, items = await self._after_restart(job, adapter, stop, items, attempt)
                    if fresh is None:
                        break
                    tp = fresh
                except Exception as e:
                    log.exception("Job %s: %s failed in %s", job.id, job.stage, job.project)
                    words = self._friendly(adapter, tp, e)
                    await self.procman.stop(tool_id)    # it may still be at the props
                    self._fail_left(job, words)
                    break
            self._fail_left(job, f"{name} did not make this prop. Check its log, then run this prop again.")
        except asyncio.CancelledError:
            closing = True
            raise
        finally:
            for prop_id, entry in job.props.items():
                if entry["state"] not in ENDED:
                    job.props[prop_id] = _prop_entry(base.PROP_CANCELLED)
            job.part = job.loading = None
            self._tally(job)
            await self._drop_work_dir(job, retry=not closing)

    async def _after_restart(self, job: Job, adapter: ToolAdapter, stop: ToolNeedsRestart, items: list[BatchItem],
                             attempt: int) -> tuple[ToolProcess | None, list[BatchItem]]:
        """The tool stopped in the middle of the batch: fail the prop it was on, let the props that are left wait
        again, and start it afresh for them -> (the fresh tool, those props). The tool is left stopped and nothing
        is sent again (None) when this was already the second try, when nothing is left to make, when it could not
        start, or when the job was cancelled meanwhile."""
        name = adapter.display_name
        log.warning("Job %s: %s has to start afresh (%s)", job.id, name, stop)
        if job.cancel_requested:
            await self.procman.stop(adapter.tool_id)
            return None, []
        if stop.blame in job.props:
            self._prop_failed(job, stop.blame, f"{name} stopped while working on this prop. Check its log "
                                               f"(Settings > Debugging), then make this prop again.")
        left = [item for item in items if job.props[item.prop_id]["state"] not in ENDED]
        for item in left:
            job.props[item.prop_id] = _prop_entry(base.WAITING)
        job.part = job.loading = job.current_prop = None
        self._tally(job)
        if attempt == 1 and left:
            fresh = await self._restart_tool(job, adapter)
        else:       # it broke twice, or nothing is left to make: the next job starts it afresh
            fresh = None
            await self.procman.stop(adapter.tool_id)
        if job.cancel_requested:
            return None, []
        if fresh is None:
            for item in left:
                self._prop_failed(job, item.prop_id, f"{name} stopped before this prop's turn. Check its log, "
                                                     f"then run this prop again.")
            return None, []
        return fresh, left

    def _fail_left(self, job: Job, message: str) -> None:
        """Fail every prop of a batch job that has not ended yet; a job that is being cancelled says nothing
        (_run_batch leaves those props cancelled)."""
        if job.cancel_requested:
            return
        for prop_id in list(job.props):
            self._prop_failed(job, prop_id, message)

    async def _batch_items(self, job: Job) -> list[BatchItem]:
        """The job's props as run_batch takes them. One that cannot be read (deleted since the job was
        queued) fails at once."""
        items: list[BatchItem] = []
        for prop_id in job.prop_ids:
            try:
                prop, prop_dir, params = await self._prop_input(job, prop_id)
            except ProjectError as e:
                self._prop_failed(job, prop_id, str(e))
                continue
            items.append(BatchItem(prop_id=prop_id, prop=prop, params=params, prop_dir=prop_dir))
        return items

    async def _prop_input(self, job: Job, prop_id: str) -> tuple[dict[str, Any], Path, dict[str, Any]]:
        """What the adapter is given for one prop of the job: the prop as it stands, its folder (made if it is
        not there) and its settings for the stage with the job's own on top. Raises ProjectError."""
        prop = await asyncio.to_thread(self.store.get_prop, job.project, prop_id)
        prop_dir = self.store.prop_dir(job.project, prop_id)
        prop_dir.mkdir(parents=True, exist_ok=True)
        return prop, prop_dir, {**((prop.get(job.stage) or {}).get("settings") or {}), **job.params}

    def _batch_context(self, job: Job, tp: ToolProcess, work: Path, attempt: int) -> BatchContext:
        def update(prop_id: str, state: str, fraction: float, message: str) -> None:
            entry = job.props[prop_id]
            if entry["state"] in ENDED or state not in (base.WAITING, base.WORKING, base.RESTING,
                                                        base.PROP_CANCELLED):
                return
            job.props[prop_id] = _prop_entry(state, fraction, message)
            self._tally(job)

        async def finish(prop_id: str, patch: dict[str, Any]) -> None:
            await asyncio.to_thread(self.store.apply_stage_patch, job.project, prop_id, job.stage, patch)
            job.props[prop_id] = _prop_entry(base.PROP_DONE)
            self._tally(job)

        def set_part(part: dict[str, Any] | None, current: str | None, loading: str | None) -> None:
            job.part = {key: part.get(key) for key in ("key", "label", "position", "of")} if part else None
            job.current_prop = current if current in job.props else None
            job.loading = loading
            self._tally(job)

        return BatchContext(project=job.project, stage=job.stage, base_url=tp.base_url, client=self.client,
                            settings=self.settings.get(), job_id=job.id, work_dir=work, attempt=attempt,
                            cancelled=lambda: job.cancel_requested,
                            tool_alive=lambda: tp.alive() and tp.state == READY,
                            update=update, finish=finish,
                            fail=lambda prop_id, message: self._prop_failed(job, prop_id, message),
                            set_part=set_part)

    def _prop_failed(self, job: Job, prop_id: str, message: str) -> None:
        """A prop of a batch job failed: the artist's message is in the job at once. A prop that has already
        ended keeps what it is: a model that was saved is never turned into a failure."""
        if job.props.get(prop_id, {}).get("state") in ENDED:
            return
        job.errors[prop_id] = str(message)
        job.props[prop_id] = _prop_entry(base.PROP_FAILED)
        self._tally(job)

    def _tally(self, job: Job) -> None:
        """A batch job's counters, from its props: how many ended made or failed, how far they are on
        average (one that failed or was cancelled has nothing left to do), and what the job says: what the
        tool is reading from disk, else the words of the prop it works on. While a cancel is on its way
        the job keeps saying so."""
        entries = list(job.props.values())
        job.done = sum(1 for entry in entries if entry["state"] in (base.PROP_DONE, base.PROP_FAILED))
        job.fraction = sum(1.0 if entry["state"] in (base.PROP_FAILED, base.PROP_CANCELLED) else entry["fraction"]
                           for entry in entries) / max(1, len(entries))
        if job.cancel_requested:
            return
        working = job.props.get(job.current_prop or "")
        if job.loading:
            job.message = f"Loading the {job.loading}…"
        elif working is not None and working["state"] == base.WORKING and working["message"]:
            job.message = working["message"]

    async def _drop_work_dir(self, job: Job, retry: bool) -> None:
        """Remove the job's scratch folder; once more after a moment when a file in it was still open (in a
        tool that was just stopped). What stays is swept away later (ProjectStore.clear_work_dirs)."""
        work, job.work = job.work, None
        if not work:
            return
        for last in (False, True):
            try:
                if await asyncio.to_thread(self.store.remove_work_dir, job.project, work):
                    return
            except OSError as e:
                log.info("Job %s: could not remove its work folder (%s)", job.id, e)
                return
            if last or not retry:
                break
            await asyncio.sleep(WORK_DIR_RETRY)
        log.info("Job %s: its work folder %s is still in use; it is removed later", job.id, work)

    async def _restart_tool(self, job: Job, adapter: ToolAdapter) -> ToolProcess | None:
        """Stop the job's tool (it may be half-dead, or still at a prop the hub gave up on) and start it afresh, for
        the next prop -> it, or None when it could not start (or the job was cancelled meanwhile)."""
        tool_id = adapter.tool_id
        log.info("Job %s: starting %s again for the next prop", job.id, adapter.display_name)
        job.message = f"Starting {adapter.display_name} again…"
        try:
            await self.procman.stop(tool_id)
            return await self._start_tool(job, tool_id)
        except ToolError as e:
            log.warning("Job %s: %s could not start again: %s", job.id, adapter.display_name, e)
            return None

    async def _start_tool(self, job: Job, tool_id: str) -> ToolProcess | None:
        """ensure_running, then claim_gpu (the job is about to use the graphics card: the GPU tool that holds it
        makes way), but give up waiting (None) as soon as the job is cancelled. The tool itself keeps starting;
        the next job will find it ready. Raises ToolError when the tool cannot start or stopped meanwhile."""
        waiter = asyncio.create_task(self._ready_on_card(tool_id, job.stage))
        try:
            while not waiter.done():
                await asyncio.wait({waiter}, timeout=0.5)
                if job.cancel_requested and not waiter.done():
                    return None
            return waiter.result()
        finally:
            if not waiter.done():
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)

    async def _ready_on_card(self, tool_id: str, stage: str) -> ToolProcess:
        """The job's tool, started and ready for `stage` (the image generator in that stage's arithmetic), holding
        the graphics card when it uses one. Raises ToolError."""
        await self.procman.ensure_running(tool_id, {"stage": stage})
        return await self.procman.claim_gpu(tool_id)

    async def _run_prop(self, job: Job, adapter: ToolAdapter, tp: ToolProcess, prop_id: str,
                        released: asyncio.Event | None = None, following: str | None = None) -> bool:
        """One prop -> whether the tool must start afresh before the next one (the adapter raised
        ToolNeedsRestart). `released` is set when the prop has let go of the tool (its UV unwrap began, or the
        adapter said so: StageContext.release; the job runner may send the next prop meanwhile); from then on only
        the newer prop's reports show. `following`: the prop after this one (StageContext.upcoming)."""
        total = len(job.prop_ids)
        job.current_prop, job.prop_fraction = prop_id, 0.0

        def report(fraction: float, message: str) -> None:
            if job.current_prop != prop_id:
                return                      # the next prop is on the card: its words show, not this unwrap's
            try:
                f = min(1.0, max(0.0, float(fraction)))
            except (TypeError, ValueError):
                f = 0.0
            job.prop_fraction = f
            job.fraction = max(job.fraction, (job.done + f) / total)
            job.message = str(message)

        def preview(picture: bytes | None, step: int = 0, box: list[float] | None = None) -> None:
            if job.current_prop != prop_id:
                return                      # the next prop is on the card: only its pictures show
            if picture:
                job.previews += 1
                job.preview = (bytes(picture), int(step), box)
            else:
                job.preview = None

        finished = False
        restart = False
        try:
            prop, prop_dir, params = await self._prop_input(job, prop_id)

            def let_go(fraction: float, words: str) -> bool:
                """The prop lets go of the tool (once) -> whether it did now. It finishes behind the next prop
                until it ends (job.behind)."""
                if released is None or released.is_set():
                    return False
                job.preview = None          # its pictures are made: the next prop's viewport starts without them
                job.behind[prop_id] = (fraction, words)
                released.set()
                return True

            async def unwrapper(src: Path, dst: Path, uv_leniency: float) -> dict[str, Any]:
                let_go(UNWRAP_FRACTION, UNWRAPPING)
                return await self._unwrap(job, src, dst, uv_leniency, prop_id)

            def release(words: str) -> None:
                let_go(round(job.prop_fraction, 4), str(words))

            async def upcoming() -> tuple[dict[str, Any], Path] | None:
                if following is None:
                    return None
                try:
                    nxt, nxt_dir, _params = await self._prop_input(job, following)
                except ProjectError:
                    return None
                return nxt, nxt_dir

            ctx = StageContext(project=job.project, prop_id=prop_id, prop_dir=prop_dir, base_url=tp.base_url,
                               client=self.client, settings=self.settings.get(), report=report,
                               cancelled=lambda: job.cancel_requested, stage=job.stage, unwrapper=unwrapper,
                               warn=lambda text: job.warnings.__setitem__(prop_id, str(text)), preview=preview,
                               release=release, upcoming=upcoming)
            job.message = f"Working on {prop.get('name') or prop_id}…"
            patch = await adapter.run_stage(ctx, prop, params)
            # A tool that can't stop in the middle of a prop made it in full after a cancel (ctx.completed): it is
            # saved, so it counts as finished, not as cut short.
            finished = not job.cancel_requested or ctx.completed
            if patch:
                await asyncio.to_thread(self.store.apply_stage_patch, job.project, prop_id, job.stage, patch)
        except ToolNeedsRestart as e:
            restart = True
            if not job.cancel_requested:
                job.errors[prop_id] = str(e)
        except (StageError, ProjectError) as e:
            if not job.cancel_requested:
                job.errors[prop_id] = str(e)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("Job %s: %s failed for prop %s/%s", job.id, job.stage, job.project, prop_id)
            if not job.cancel_requested:
                job.errors[prop_id] = self._friendly(adapter, tp, e)
        finally:
            if job.current_prop == prop_id:
                job.preview = None  # done, failed or cancelled: the prop's preview picture goes with it
            job.behind.pop(prop_id, None)
        cut = job.cancel_requested and not finished
        if cut:
            job.cut_short += 1
        job.ended[prop_id] = (base.PROP_CANCELLED if cut else
                              base.PROP_FAILED if prop_id in job.errors else base.PROP_DONE)
        job.done += 1
        job.fraction = max(job.fraction, job.done / total) if job.current_prop != prop_id else job.done / total
        return restart

    async def _unwrap(self, job: Job, src: Path, dst: Path, uv_leniency: float,
                      prop_id: str | None = None) -> dict[str, Any]:
        """StageContext.unwrap: Instant Meshes UV-unwraps `src` into `dst`, started first if needed. It runs on the
        CPU, so it starts beside the job's own GPU tool without stopping it. The job's message says so (the page
        shows the time; its fraction is the adapter's) unless the next prop is on the card already."""
        imesh = self.adapters.get(paths.IMESH)
        unwrap = getattr(imesh, "unwrap", None)
        if unwrap is None:
            raise StageError("The UVs are unwrapped by Instant Meshes, which is not part of this Spellcast3D.")
        try:
            tp = await self.procman.ensure_running(paths.IMESH)
        except ToolError as e:
            raise StageError(f"The UVs are unwrapped by Instant Meshes, which could not start: {e}") from e
        self.procman.set_busy(paths.IMESH, True)
        request = asyncio.ensure_future(unwrap(self.client, tp.base_url, src, dst, uv_leniency))
        try:
            if prop_id is None or job.current_prop == prop_id:
                job.message = UNWRAPPING
            return await request
        finally:
            request.cancel()    # no-op once done; drops the request if this job itself is cancelled
            self.procman.set_busy(paths.IMESH, False)

    @staticmethod
    def _friendly(adapter: ToolAdapter, tp: ToolProcess, e: Exception) -> str:
        name = adapter.display_name
        if not tp.alive() or tp.state != READY:
            return f"{name} stopped while working on this prop. Check its log, then run this prop again."
        if isinstance(e, httpx.HTTPError):
            return (f"Spellcast3D could not talk to {name} ({type(e).__name__}). "
                    f"Try again; if it keeps failing, stop and restart {name}.")
        return (f"Something went wrong while {name} worked on this prop ({type(e).__name__}: {e}). "
                f"The details are in code/logs/hub.log.")
