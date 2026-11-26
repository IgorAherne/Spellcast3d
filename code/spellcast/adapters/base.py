"""The contract between the hub core and each wrapped tool.

The hub core (server.py / procman.py / jobs.py) knows nothing tool-specific.
Everything it needs to launch, probe, drive and stop a tool lives in a
ToolAdapter subclass: adapters/qwen.py, adapters/forge.py, adapters/pixal3d.py, adapters/imesh.py,
adapters/lato2.py, adapters/triflow.py.

Runs inside the hub venv (httpx available). Never imported by the installer.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

import httpx

from .. import paths
from ..install import state   # stdlib-only module; the install package imports nothing else here


class StageError(Exception):
    """A user-facing failure of one prop's stage (bad input, tool error...).

    The message is shown verbatim in the UI, so write it for an artist:
    "Pick an image in the Images stage first", not "KeyError: 'chosen'".
    """


class ToolNeedsRestart(StageError):
    """This prop failed, and the tool must be started afresh before the next one: its GPU context broke (the tool
    process exits), or it still works on a request the hub gave up on (a timeout). The job runner restarts it,
    once per job, and goes on with the next prop.

    From run_batch it ends the whole request: the process is gone, hangs, or still runs another batch. `blame` is
    the prop that was on the card then, when that is known: the job runner fails it and sends the others again."""

    def __init__(self, message: str = "", *, blame: str | None = None):
        super().__init__(message)
        self.blame = blame


@dataclass
class LaunchSpec:
    argv: list[str]                 # argv[0] is the venv python.exe
    cwd: Path
    env: dict[str, str]             # build with paths.clean_env(tool_id, extra)
    ready_timeout: float = 300.0    # seconds for health() to turn True after spawn


@dataclass
class StageContext:
    """Everything run_stage needs for one prop. Built by the job runner."""
    project: str
    prop_id: str
    prop_dir: Path                              # projects/<project>/props/<prop_id>/
    base_url: str                               # "http://127.0.0.1:<port>" of the running tool
    client: httpx.AsyncClient                   # shared client, no base_url set, generous timeouts
    settings: dict[str, Any]                    # merged hub settings (defaults.json + settings.json)
    report: Callable[[float, str], None]        # report(fraction 0..1 for THIS prop, message)
    cancelled: Callable[[], bool]               # poll during long work; stop early if True
    stage: str = ""                             # the stage being run; "" = the adapter's main stage
    # unwrapper(src, dst, uv_leniency): the job runner's UV unwrap in Instant Meshes (see unwrap)
    unwrapper: Callable[[Path, Path, float], Awaitable[dict[str, Any]]] | None = None
    # Set by run_stage when it made the prop in full although the job was cancelled meanwhile (a tool that can't
    # stop in the middle of a prop finishes it and it is saved): the job then counts the prop as done.
    completed: bool = False
    # warn(text): something the artist should know about this prop's finished result (e.g. "this took longer: fewer
    # example pictures are faster"). The job keeps it (Job.warnings) and the page says it when the job ends; the
    # progress message would not do, the job's last one is its own "Done.". None: nobody listens (tests).
    warn: Callable[[str], None] | None = None
    # preview(picture, step, box): the picture being made as it looks so far (a small PNG or JPEG, as bytes; step: the
    # steps it shows; box: the part of the prop's picture it is of, [left, top, right, bottom] as shares of its sides -
    # None: all of it), for the page to show while the prop is worked on; picture None: there is none any more. The
    # job keeps only the newest, in memory (Job.preview), and drops it when the prop ends. None: nobody listens.
    preview: Callable[..., None] | None = None
    # release(words): the prop has let go of its tool; what is left of it runs on the CPU (`words`: what, for the
    # page) and the job runner sends the next prop to the tool meanwhile. None: nobody listens (it keeps the tool).
    release: Callable[[str], None] | None = None
    # upcoming(): the job's next prop as its run_stage will get it, (prop, prop_dir), or None after the last one; for
    # work a prop can start for the next one while its tool is busy. None: nobody tells.
    upcoming: Callable[[], Awaitable[tuple[dict[str, Any], Path] | None]] | None = None

    def rel(self, p: Path) -> str:
        """Path relative to the prop dir, forward slashes - the form stored in project.json."""
        return p.resolve().relative_to(self.prop_dir.resolve()).as_posix()

    def abs(self, rel: str | None) -> Path | None:
        return None if not rel else (self.prop_dir / rel)

    async def unwrap(self, src_rel: str, dst_rel: str, uv_leniency: float) -> dict[str, Any]:
        """UV-unwrap the low poly src_rel (an OBJ another tool made, without UVs) into dst_rel (an OBJ with
        UVs, its faces and positions kept), both relative to the prop dir. Instant Meshes does it: it is
        started if needed, and runs on the CPU beside a GPU tool. It takes seconds to minutes and reports
        no progress, so report before calling.
        -> {"vertices", "faces", "has_uv", "warning", "seconds"}; has_uv False (with a warning) when the
        unwrap failed and the mesh was written without UVs. Raises StageError."""
        if self.unwrapper is None:
            raise StageError("Spellcast3D cannot unwrap UVs here (Instant Meshes is not available).")
        return await self.unwrapper(self.prop_dir / src_rel, self.prop_dir / dst_rel, float(uv_leniency))


@dataclass
class BatchItem:
    """One prop of a run_batch call."""
    prop_id: str
    prop: dict[str, Any]                        # read-only snapshot of the prop, as run_stage gets it
    params: dict[str, Any]                      # prop[stage]["settings"] with the job's overrides applied
    prop_dir: Path                              # projects/<project>/props/<prop_id>/ (exists)


@dataclass
class BatchContext:
    """Everything run_batch needs for all the props of a job at once. Built by the job runner, which keeps a
    state per prop (jobs.Job.props: PROP_STATES below, a fraction 0..1 and a message) and hears of each prop
    through the callbacks here, the moment something happens to it."""
    project: str
    stage: str
    base_url: str                               # "http://127.0.0.1:<port>" of the running tool
    client: httpx.AsyncClient
    settings: dict[str, Any]                    # merged hub settings
    job_id: str
    # projects/<project>/temp/<stage>-<job id>/: the tool's scratch space for this job. It exists; on the second
    # call (attempt 2) it holds what the stopped tool left, which the tool cleans up or continues from.
    work_dir: Path
    attempt: int                                # 1, or 2 after the tool was started afresh
    cancelled: Callable[[], bool]
    tool_alive: Callable[[], bool]              # the tool's process still runs and has not been asked to stop
    # update(prop_id, state, fraction, message): where the prop is now (not for "done" and "failed")
    update: Callable[[str, str, float, str], None]
    # finish(prop_id, patch): the prop is made; the patch is merged into prop[stage] NOW, so its result can be
    # used while the others are still worked on. Raises ProjectError (e.g. the prop was deleted meanwhile).
    finish: Callable[[str, dict[str, Any]], Awaitable[None]]
    fail: Callable[[str, str], None]            # fail(prop_id, message for the artist): in the job's errors at once
    # set_part(part, current, loading): the step all props are taken through now ({"key", "label", "position",
    # "of"}: prop `position` of the `of` that go through it; None between steps), the prop being worked on (None
    # between props) and what the tool is reading from disk meanwhile, in the artist's words (None: nothing).
    set_part: Callable[[dict[str, Any] | None, str | None, str | None], None]


# The states of a prop in a job (jobs.Job.props). "resting": begun, and waiting for its next step while the tool
# takes the other props through the current one.
WAITING, WORKING, RESTING, PROP_DONE, PROP_FAILED, PROP_CANCELLED = (
    "waiting", "working", "resting", "done", "failed", "cancelled")
PROP_STATES = (WAITING, WORKING, RESTING, PROP_DONE, PROP_FAILED, PROP_CANCELLED)


class ToolAdapter(abc.ABC):
    tool_id: str            # paths.QWEN | paths.FORGE | paths.PIXAL3D | paths.IMESH | paths.LATO2 | paths.TRIFLOW
    display_name: str       # "Qwen-Image 2.1", "Forge Neo", "Pixal3D", "Instant Meshes", "LATO.2", "TriFlow"
    role: str = ""          # what it is for, in the artist's words ("image generator"); empty: say the name
    stage: str              # "images" | "mesh" | "retopo": the tool's main stage
    extra_stages: tuple[str, ...] = ()   # further stages this tool runs ("projection" for Instant Meshes)
    # The stages (of self.stages) whose props the tool takes all at once (run_batch) instead of one after another
    # (run_stage): Pixal3D's "mesh", which takes every prop through one step before it loads the next step's models.
    batch_stages: tuple[str, ...] = ()
    uses_gpu: bool          # GPU tools are mutually exclusive when settings.gpu_exclusive
    # Whether the process takes the graphics card as soon as it starts (it loads its models onto it before it says it
    # is ready). False for a tool that touches the card only when a job begins (Pixal3D reads its models per job): it
    # may start early, beside the GPU tool that holds the card, and takes the card at its first job
    # (ProcessManager.claim_gpu).
    gpu_at_start: bool = True
    ui_path: str = "/"      # native UI path relative to base_url (the iframe target)
    # A license the artist must know about when picking the tool, shown by the page (/api/status tools):
    # {"name": "ADPNCL 1.0", "url": "https://...", "noncommercial": True}; "research": True flags a research licence
    # (Qwen-Image's: its pictures are still the artist's), "note" says that in a sentence. None: nothing to flag.
    license: dict[str, Any] | None = None

    @property
    def stages(self) -> tuple[str, ...]:
        return (self.stage, *self.extra_stages)

    # ---- installation ---------------------------------------------------
    def install_status(self) -> dict[str, Any]:
        """Entry from code/venvs/install-state.json, or {"status": "not_installed"}."""
        entry = state.load().get(self.tool_id)
        entry = entry if isinstance(entry, dict) else {}
        if entry.get("status") == state.INSTALLED and not paths.venv_python(self.tool_id).exists():
            return {"status": state.NOT_INSTALLED, "reason": "venv missing"}
        return entry or {"status": state.NOT_INSTALLED, "reason": ""}

    def supported(self) -> tuple[bool, str]:
        """(False, reason) when this machine cannot run the tool at all (e.g. GPU too old)."""
        return True, ""

    # ---- process lifecycle ----------------------------------------------
    @abc.abstractmethod
    def launch_spec(self, port: int, settings: dict[str, Any]) -> LaunchSpec:
        """How to start the tool server on 127.0.0.1:<port>. Must never open a browser or a public share link."""

    @abc.abstractmethod
    async def health(self, client: httpx.AsyncClient, base_url: str) -> bool:
        """True once the tool answers and is ready for run_stage and for its UI to be iframed."""

    async def on_stop(self, client: httpx.AsyncClient, base_url: str) -> None:
        """Called before the process is terminated (best effort, may be skipped)."""

    # An early start may say what the artist is about to ask of the tool (ProcessManager.start_early's `wish`; the
    # page's: {"model": ...} for the image generator), so that the start loads that and nothing else first.
    def aim(self, wish: Mapping[str, Any]) -> None:
        """Note `wish` for the next launch_spec alone ({}: none). Default: a tool has no choice to make at its start."""

    def serves(self, wish: Mapping[str, Any]) -> bool:
        """Whether the process as it runs, or loads, serves `wish` without loading something else first."""
        return True

    # ---- work -------------------------------------------------------------
    def check_settings(self, stage: str, settings: dict[str, Any]) -> None:
        """Raise StageError when a prop's `stage` settings cannot be run. Called when a job is
        submitted (so the artist hears at once) and again by run_stage. Default: no checks."""

    async def run_stage(self, ctx: StageContext, prop: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Run ctx.stage (one of self.stages, and not one of self.batch_stages) for ONE prop.

        prop   : read-only snapshot of the prop (see ARCHITECTURE.md "Prop schema").
        params : prop[stage]["settings"] with any job-level overrides applied.
        return : a patch shallow-merged into prop[stage] by the job runner, e.g.
                 {"glb": "mesh/mesh.glb", "stats": {...}}. Paths relative to the
                 prop dir with forward slashes (use ctx.rel()). Write outputs
                 under ctx.prop_dir/<stage>/. Raise StageError for user-facing errors.
        """
        raise NotImplementedError(f"{type(self).__name__} does not run the {ctx.stage} stage prop by prop")

    async def run_batch(self, ctx: BatchContext, items: list[BatchItem]) -> None:
        """Run ctx.stage (one of self.batch_stages) for ALL the props of a job in one request to the tool.

        Each prop is reported through ctx as it goes: ctx.update while it waits or is worked on, then exactly one
        of ctx.finish (made: its patch, as run_stage would return it), ctx.fail (the artist's message) or
        ctx.update(..., "cancelled", ...). Returns when every item ended in one of the three; nothing about one
        prop may fail the others. A cancel (ctx.cancelled()) is passed on to the tool, and the props it had
        finished by then are kept. Raises ToolNeedsRestart when the tool must start afresh: the job runner then
        fails the prop that exception blames, starts the tool again once and calls run_batch a second time
        (ctx.attempt 2, the same ctx.work_dir) with the props that are left; never after a cancel.
        """
        raise NotImplementedError(f"{type(self).__name__} does not run the {ctx.stage} stage for all props at once")
