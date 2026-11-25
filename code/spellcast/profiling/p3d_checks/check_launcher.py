"""The live launcher (launchers/pixal3d_server.py) as the hub uses it: started as a process, driven through HTTP,
its batches compared with the golden cases.

    python check_launcher.py --repo <fork worktree with the staged engine>                 mv1024 + sv1024 in one batch
    python check_launcher.py --repo <...> --cases sv1024 --reference                       and the reference route after
    python check_launcher.py --repo <any Pixal3D checkout> --dry                           no engine, no models, no card
    python check_launcher.py --repo <fork worktree> --fans [--case sv1024]                recentre_fans on a golden mesh
    python check_launcher.py --repo <any Pixal3D checkout> --pictures                      the picture hooks, on the CPU

The check itself (this process) only talks HTTP, as the hub does: it starts the launcher in a child process with
a scratch projects folder, waits for /spellcast/health, posts one batch with the golden cases' requests, polls
GET /spellcast/batch/{id} four times a second and reads the record. It holds the GPU lock for the child.

The child is this file again (--serve): it imports the live launcher, replaces pixal3d_server.build_engine - the
launcher's one hook point - and runs pixal3d_server.main(). Through that hook the engine gets the launcher's hooks
wrapped in a recorder, which compares on the spot what goes in and out of them with the golden tensors:

    prepare   the prepared picture (00-input/image_u8) or the framed views (00-input/view_<i>_u8, azimuths)
    camera    the three numbers of the manifest's camera, as Python floats
    bundle    the views' tensors (00-input/images_512, images_1024, camera_angle_x, camera_distance, transform_matrix)
    finish    the decoded mesh before to_glb: vertices and faces by the manifest's SHA-256 (after the hole filling
              and the launcher's own recentring), the texture voxels (11-decode/tex_coords, tex_attrs), the resolution

What is measured: seconds from the spawn to "ready" (under 40 s is asked), and how long a poll waits for its answer
while a mesh is cleaned up and saved (to_glb's native code keeps Python's lock for seconds; the hub's read timeout
is 10 s).

--reference adds what needs the reference pipeline in memory (about 25 GB of RAM): a /spellcast/generate call
after the batch, then a second batch, which must find the reference's models dropped, DINOv3's shared copy with them.

--dry runs the same driver around a stand-in engine that walks the parts without a card (no fork with a staged
engine and no model files are needed): the start, the routes on Pixal3D's real web server, cancel, and this
check's own plumbing. Nothing is compared with a golden case then.

--fans needs no server: the golden decoded mesh (11-decode vertices_raw / faces_raw) goes through Mesh.fill_holes
on the card, twice in one process (cumesh leaves the centres of the second one out of place), and the launcher's
recentre_fans must give the same vertices to the bit as the frozen launcher's (which reads every face as 64-bit
indices), and the first pass the mesh of the golden run (the manifest's vertices_sha / faces_sha).

--pictures needs neither a server nor the card nor the lock: the launcher's hooks prepare and bundle run on the CPU
for every golden case whose pictures bring their own transparency (no background removal: mv1024, mv1024b, sv1536,
big1536), and are compared as above.

--old-crop: a single picture larger than 1024 px is cut out in Pixal3D's own order (pixal3d_mv.CROP_FIRST off: the
whole picture reduced to 1024 px first), as the golden cases were captured. The launcher crops such a picture first
now, so without it sv1024, redraw (1254 px pictures) and big1536 (3772 px, 2048 once opened) cannot match their
golden input; sv1536, mv1024 and mv1024b are not touched by it.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator

import common

HUB = common.HUB
LAUNCHERS = HUB / "code" / "spellcast" / "launchers"
OUT = HUB / "code" / "logs" / "profiling" / "pixal3d-lowmem" / "wp7"
PYTHON = HUB / "code" / "venvs" / "pixal3d" / "Scripts" / "python.exe"
READY_LIMIT = 40.0          # seconds from the spawn to "ready" the specification asks for
POLL_LIMIT = 8.0            # a poll slower than this during "finish": the hub's read timeout has to grow
POLL_EVERY = 0.25
PARTS = ("cutout", "camera", "features", "ss", "cond_lr", "lr", "upsample", "cond_hr", "hr", "tex", "finish")


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo", required=True, type=lambda text: Path(text).resolve(),
                   help="the Pixal3D checkout the launcher runs (a fork worktree with the staged engine)")
    p.add_argument("--cases", default="mv1024,sv1024", help="golden cases of the batch, in the order they are sent")
    p.add_argument("--case", default="sv1024", help="--fans: the golden case whose decoded mesh is used")
    p.add_argument("--port", type=int, default=7791, help="the launcher's port (7791-7799)")
    p.add_argument("--reference", action="store_true", help="also /spellcast/generate after the batch, then a "
                                                            "second batch (loads the reference pipeline: 25 GB of RAM)")
    p.add_argument("--dry", action="store_true", help="a stand-in engine: no models, no card, no golden case")
    p.add_argument("--fans", action="store_true", help="only recentre_fans on the golden decoded mesh (GPU, no server)")
    p.add_argument("--pictures", action="store_true", help="only the hooks prepare and bundle against the golden "
                                                           "inputs (CPU; every captured case unless --cases is given)")
    p.add_argument("--max-vram-mib", type=int, default=None, help="the engine behaves as on a card of this size")
    p.add_argument("--keep", action="store_true", help="keep the scratch folder (the GLBs, the views, the log)")
    p.add_argument("--old-crop", action="store_true", help="cut a picture over 1024 px out in Pixal3D's own order, "
                                                           "as the golden cases were captured")
    p.add_argument("--serve", default=None, help=argparse.SUPPRESS)       # the child: the scratch folder
    return p.parse_args()


def result(name: str, ok: bool, detail: str = "", rule: str = "launcher") -> common.Result:
    return common.Result(name, bool(ok), rule, detail)


# =====================================================================================================================
# The child: the live launcher with a recorder around its hooks (or a stand-in engine)
# =====================================================================================================================

class Recorder:
    """What the launcher's hooks take and give, compared with the golden case of the prop at once. Each comparison
    is one JSON line in <scratch>/results.jsonl, which the driver reads when the batch is over."""

    def __init__(self, scratch: Path, cases: dict[str, str]) -> None:
        self.path = scratch / "results.jsonl"
        self.cases = {prop_id: common.golden(name) for prop_id, name in cases.items()}
        self.bundled: set[str] = set()

    def add(self, *results: common.Result) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            for one in results:
                line = {"name": one.name, "ok": one.ok, "rule": one.rule, "detail": one.detail}
                handle.write(json.dumps(line) + "\n")

    def note(self, name: str, ok: bool, detail: str = "") -> None:
        self.add(result(name, ok, detail))

    def wrap(self, hooks: Any) -> Any:
        recorder = self

        class Recording:
            """The launcher's hooks, each watched."""

            def prepare(self, req: Any, matte: Callable[[Any], Any]) -> Any:
                prepared = hooks.prepare(req, matte)
                recorder.guarded(req, "prepare", lambda case: recorder.prepared(req, case, prepared))
                return prepared

            def camera(self, req: Any, image: Any, moge: Any) -> dict[str, float]:
                found = hooks.camera(req, image, moge)
                recorder.guarded(req, "camera", lambda case: recorder.camera(req, case, found))
                return found

            def bundle(self, req: Any, images: list[Any], meta: dict[str, Any]) -> dict[str, Any]:
                views = hooks.bundle(req, images, meta)
                if req.id not in recorder.bundled:          # asked for by three parts: the same every time
                    recorder.bundled.add(req.id)
                    recorder.guarded(req, "bundle", lambda case: recorder.bundle(req, case, views))
                return views

            def prop_scope(self, req: Any, part: str) -> Any:
                return hooks.prop_scope(req, part)

            def finish(self, req: Any, mesh: Any, notes: dict[str, Any]) -> dict[str, Any]:
                recorder.guarded(req, "finish", lambda case: recorder.decoded(req, case, mesh))
                started = time.perf_counter()
                extras = hooks.finish(req, mesh, notes)
                recorder.note(f"{req.id}: finish hook", True, "to_glb and saving took "
                              f"{time.perf_counter() - started:.1f} s; notes {json.dumps(notes, default=str)}")
                return extras

        return Recording()

    def guarded(self, req: Any, what: str, compare: Callable[[common.Case], None]) -> None:
        """A comparison must never change what the launcher does: its own failure is a failed line."""
        case = self.cases.get(req.id)
        if case is None:
            return
        try:
            compare(case)
        except Exception as exc:
            self.note(f"{req.id}: {what} could not be compared", False, f"{type(exc).__name__}: {exc}")

    def prepared(self, req: Any, case: common.Case, prepared: Any) -> None:
        import numpy as np
        import torch

        def pixels(image: Any) -> Any:
            return torch.from_numpy(np.array(image))

        if not req.multi:
            self.add(self.named(req, common.compare("00-input/image_u8", pixels(prepared.images[0]),
                                                    case.tensor("00-input", "image_u8"))))
            return
        want = case.tensor("00-input", "azimuths")
        self.add(self.named(req, common.compare("00-input/azimuths", torch.tensor(prepared.meta["azimuths"],
                                                                                 dtype=want.dtype), want)))
        for n, image in enumerate(prepared.images):
            self.add(self.named(req, common.compare(f"00-input/view_{n}_u8", pixels(image),
                                                    case.tensor("00-input", f"view_{n}_u8"))))

    def camera(self, req: Any, case: common.Case, found: dict[str, Any]) -> None:
        want = case.manifest.get("camera") or {}
        same = all(type(found.get(key)) is float and found.get(key) == want.get(key) for key in want) and bool(want)
        self.note(f"{req.id}: camera", same, "equal to the golden run's: " + ", ".join(
            f"{key} {float(found[key]).hex()}" for key in sorted(found)) if same else f"got {found!r}, golden {want!r}")

    def bundle(self, req: Any, case: common.Case, views: dict[str, Any]) -> None:
        for name, got in (("images_512", views["images"][512]), ("images_1024", views["images"][1024]),
                          ("camera_angle_x", views["camera_angle_x"]), ("camera_distance", views["camera_distance"]),
                          ("transform_matrix", views["transform_matrix"])):
            self.add(self.named(req, common.compare(f"00-input/{name}", got, case.tensor("00-input", name))))
        self.note(f"{req.id}: bundle mesh_scale", views.get("mesh_scale") == case.manifest.get("mesh_scale", 1.0),
                  f"{views.get('mesh_scale')!r}")

    def decoded(self, req: Any, case: common.Case, mesh: Any) -> None:
        import torch

        manifest = case.manifest
        for what, tensor in (("vertices", mesh.vertices), ("faces", mesh.faces)):
            sha, want = common.sha256_of(tensor), manifest.get(f"{what}_sha")
            detail = f"{list(tensor.shape)} {str(tensor.dtype).replace('torch.', '')} on {tensor.device}"
            if sha == want:
                self.note(f"{req.id}: decoded {what}", True, f"{detail}: the golden run's SHA-256")
                continue
            # Not the same bytes: say how far off, on the part the golden case keeps (before the hole filling).
            raw = case.tensor("11-decode", f"{what}_raw")
            rows = min(int(raw.shape[0]), int(tensor.shape[0]))
            near = common.compare(f"11-decode/{what}_raw", tensor[:rows].cpu(), raw[:rows], case.spread)
            self.note(f"{req.id}: decoded {what}", False,
                      f"{detail}: {tensor.shape[0]} rows, the golden run {manifest.get(what)}; its first {rows} rows "
                      f"against {what}_raw: {near.detail}")
        for name, got in (("tex_coords", mesh.coords), ("tex_attrs", mesh.attrs)):
            self.add(self.named(req, common.compare(f"11-decode/{name}", got, case.tensor("11-decode", name),
                                                    case.spread)))
        on_cpu = all(t.device.type == "cpu" for t in (mesh.vertices, mesh.faces, mesh.coords, mesh.attrs))
        allocated, kept = torch.cuda.memory_allocated(), common.kept_on_card()
        empty = 0 <= allocated - kept <= common.LIBRARY_TABLES_DURING
        self.note(f"{req.id}: the decoded mesh waits on the CPU", on_cpu and empty,
                  f"{allocated} B on the card when the hook starts, {kept} B kept on purpose (cuBLAS's work area, "
                  f"the projection grids)" + ("" if empty else f"; {common.BOUNDS_HINT}"))
        self.note(f"{req.id}: resolution used", int(mesh.resolution) == int(manifest.get("res", -1)),
                  f"{mesh.resolution} (golden {manifest.get('res')})")

    @staticmethod
    def named(req: Any, compared: common.Result) -> common.Result:
        compared.name = f"{req.id}: {compared.name}"
        return compared


class DryTypes:
    """--dry: stands in for pixal3d.staged where the checkout has none (the request and event types alone)."""

    PARTS = PARTS

    class PropRequest:
        def __init__(self, **fields: Any) -> None:
            self.__dict__.update(fields)

    class Event:
        def __init__(self, kind: str, part: str = "", prop_id: str | None = None, step: int = 0, total: int = 0,
                     detail: dict[str, Any] | None = None) -> None:
            self.kind, self.part, self.prop_id, self.step, self.total = kind, part, prop_id, step, total
            self.detail = detail or {}

    class BatchBroken(Exception):
        kind = "restart"


class DryEngine:
    """--dry: walks the parts for every prop as the staged engine's runner orders them, half a second a part
    (three seconds of to_glb-like bars in "finish"), without a model or a card. It writes a placeholder at each
    prop's out_path."""

    multiview = True

    def __init__(self, hooks: Any, px: Any) -> None:
        self.hooks, self.px = hooks, px

    def sample_sparse_structure(self, cond: Any, resolution: int, num_samples: int = 1,
                                sampler_params: Any = None) -> Any:
        return [[0, 5, 5, 10], [0, 5, 5, 11]]

    def release(self) -> None:
        pass

    def close(self) -> None:
        pass

    def run(self, props: list[Any], work_dir: Path, *, on_event: Callable[[Any], None], cancelled: Callable[[], bool],
            wave_size: int | None = None, resume: bool = False) -> dict[str, Any]:
        order = [p for p in props if p.multi] + [p for p in props if not p.multi]
        results: dict[str, Any] = {}
        for part in PARTS:
            alive = [p for p in order if part != "camera" or not p.multi]
            on_event(DryTypes.Event("part_start", part, detail={"props": [p.id for p in alive]}))
            for position, prop in enumerate(alive, 1):
                if cancelled():
                    return {p.id: results.get(p.id, {"state": "cancelled"}) for p in props}
                on_event(DryTypes.Event("prop_start", part, prop.id, detail={"position": position, "of": len(alive)}))
                if part == "ss":
                    with self.hooks.prop_scope(prop, "ss"):
                        self.sample_sparse_structure({}, 32, 1, prop.sampler_params["ss"])
                for step in range(1, 5):
                    time.sleep(0.125)
                    on_event(DryTypes.Event("step", part, prop.id, step, 4))
                if part != "finish":
                    on_event(DryTypes.Event("prop_rest", part, prop.id))
                    continue
                for step in range(1, 7):            # as to_glb's bar reports through app.py
                    time.sleep(0.5)
                    self.px._update_progress("Extracting GLB", step, 6)
                Path(prop.payload.out_path).write_bytes(b"glTF placeholder of a dry run")
                extras = {"mode": "multi" if prop.multi else "single", "views": len(prop.payload.views or []) or 1,
                          "warning": None, "ss_retries": 0, "slow": [], "seconds": 5.0,
                          "resolution": prop.resolution, "resumed": False}
                on_event(DryTypes.Event("prop_done", part, prop.id, detail={"result": extras, "seconds": 5.0}))
                results[prop.id] = {"state": "done", "result": extras}
        return results


def serve(args: argparse.Namespace) -> int:
    """The child process: the live launcher's main() with build_engine replaced."""
    scratch = Path(args.serve)
    plan = json.loads((scratch / "plan.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(LAUNCHERS))
    import pixal3d_mv
    import pixal3d_server as server

    pixal3d_mv.CROP_FIRST = not args.old_crop
    recorder = Recorder(scratch, plan["cases"])
    build = server.build_engine
    share = server.share_dinov3
    kept: dict[str, Any] = {}

    def share_dinov3() -> Any:
        kept["shared"] = share()
        return kept["shared"]

    def gold_autotune_cache() -> None:
        """flex_gemm's kernel choices as the golden captures had them, in a copy of this run's own (the launcher's
        app.py points flex_gemm at the checkout's file; a size it has no choice for is tuned and written there)."""
        import flex_gemm.utils.autotuner as tuner

        source = common.golden_root() / "autotune_cache.json"
        if not source.is_file():
            recorder.note("flex_gemm's kernel choices of the golden run", False, f"{source} does not exist")
            return
        private = scratch / "autotune_cache.json"
        shutil.copyfile(source, private)
        tuner.load_autotune_cache(str(private))
        tuner.AUTOTUNE_CACHE_PATH = str(private)

    def watched(engine: Any) -> Any:
        """Before each batch: is the reference pipeline out of memory, DINOv3's shared copy included?"""
        run = engine.run
        px = sys.modules["app"]

        def checked(props: list[Any], work_dir: Path, **kwargs: Any) -> Any:
            shared = kept.get("shared")
            held = len(shared.loaded) if shared is not None else -1
            recorder.note(f"batch of {', '.join(p.id for p in props)}: the reference pipeline is not in memory",
                          px.pipeline is None and px.moge_model is None and held == 0,
                          f"app.pipeline {'None' if px.pipeline is None else 'loaded'}, {held} shared DINOv3")
            return run(props, work_dir, **kwargs)

        engine.run = checked
        return engine

    def build_engine(options: Any, hooks: Any) -> Any:
        import torch

        before = torch.cuda.is_initialized()        # the libraries' imports may have done it already
        if args.dry:
            engine: Any = DryEngine(hooks, sys.modules["app"])
        else:
            gold_autotune_cache()
            if args.max_vram_mib:
                options.max_vram_mib = args.max_vram_mib
            engine = watched(build(options, recorder.wrap(hooks)))
        after = torch.cuda.is_initialized()
        allocated = torch.cuda.memory_allocated() >> 20
        recorder.note("making the engine leaves the card alone", (before or not after) and allocated == 0,
                      f"CUDA initialised before the engine was made: {before}, after: {after}; {allocated} MiB "
                      "of tensors on the card")
        return engine

    server.build_engine = build_engine
    server.share_dinov3 = share_dinov3
    if args.dry:
        server.staged_module = lambda: DryTypes
        server.engine_options = lambda models_root, spill=False: None
    sys.argv = ["pixal3d_server.py", "--repo", str(args.repo), "--port", str(args.port),
                "--projects-root", str(scratch / "projects")]
    return server.main()


# =====================================================================================================================
# The driver: HTTP, as the hub
# =====================================================================================================================

def http(method: str, url: str, body: Any = None, timeout: float = 30.0) -> tuple[int, Any, float]:
    """(status, the answer's JSON, seconds it took). Status 0: no answer (not listening, or too slow)."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return answer.status, json.loads(answer.read() or b"null"), time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        try:
            told = json.loads(exc.read() or b"null")
        except ValueError:
            told = None
        return exc.code, told, time.perf_counter() - started
    except (OSError, ValueError):
        return 0, None, time.perf_counter() - started


class Polls:
    """What the polls of one batch saw: how long each answer took, and that the record only moves forward."""

    def __init__(self) -> None:
        self.count = self.unanswered = 0
        self.slowest = self.slowest_finishing = 0.0
        self.slowest_at = ""
        self.seq = 0
        self.fractions: dict[str, float] = {}
        self.backwards: list[str] = []
        self.states: dict[str, list[str]] = {}
        self.words: list[str] = []              # the props' messages without their step counts, in order of appearance
        self.loading: list[str] = []
        self.finishing = False

    def take(self, status: int, record: Any, seconds: float) -> None:
        self.count += 1
        if seconds > self.slowest:
            self.slowest, self.slowest_at = seconds, time.strftime("%H:%M:%S")
        if self.finishing:                      # the last answer said a mesh is being built or saved
            self.slowest_finishing = max(self.slowest_finishing, seconds)
        if status != 200 or not isinstance(record, dict):
            self.unanswered += 1
            return
        if record["seq"] < self.seq:
            self.backwards.append(f"seq {self.seq} -> {record['seq']}")
        self.seq = record["seq"]
        self.finishing = (record.get("part") or {}).get("key") == "finish" and record.get("current") is not None
        if record.get("loading") and record["loading"] not in self.loading:
            self.loading.append(record["loading"])
        for prop_id, item in record["items"].items():
            if item["fraction"] < self.fractions.get(prop_id, 0.0):
                self.backwards.append(f"{prop_id} {self.fractions[prop_id]} -> {item['fraction']}")
            self.fractions[prop_id] = item["fraction"]
            states = self.states.setdefault(prop_id, [])
            if not states or states[-1] != item["state"]:
                states.append(item["state"])
            words = item["message"].split(" (")[0]
            if words and words not in self.words:
                self.words.append(words)


def follow(base: str, batch_id: str, child: subprocess.Popen[bytes],
           limit: float) -> tuple[dict[str, Any] | None, Polls]:
    """Poll a batch until it has ended, the launcher has gone, or `limit` seconds have passed."""
    polls = Polls()
    deadline = time.monotonic() + limit
    record: dict[str, Any] | None = None
    while time.monotonic() < deadline and child.poll() is None:
        status, told, seconds = http("GET", f"{base}/spellcast/batch/{batch_id}", timeout=120.0)
        polls.take(status, told, seconds)
        if status == 200:
            record = told
            if told["state"] != "running":
                break
        time.sleep(POLL_EVERY)
    return record, polls


def scratch_request(case: common.Case, prop_id: str, projects: Path) -> dict[str, Any]:
    """A golden case's request as an item of a batch: its pictures copied into the scratch projects folder."""
    folder = projects / "check" / "props" / prop_id
    request = dict(case.request)

    def copied(relative: str) -> str:
        target = folder / "images" / Path(relative).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(case.input(relative), target)
        return str(target)

    request["id"] = prop_id
    request["image_path"] = copied(request["image_path"])
    request["out_path"] = str(folder / "mesh" / "mesh.check.partial.glb")
    if request.get("views"):
        request["views"] = [{**view, "image_path": copied(view["image_path"])} for view in request["views"]]
    request["views_dir"] = str(folder / "mesh" / "views") if request.get("views_dir") else None
    return request


def dry_request(prop_id: str, projects: Path, views: tuple[int, ...]) -> dict[str, Any]:
    folder = projects / "check" / "props" / prop_id
    (folder / "images").mkdir(parents=True, exist_ok=True)
    pictures = [folder / "images" / f"{n:04d}.png" for n in range(1, max(len(views), 1) + 1)]
    for picture in pictures:
        picture.write_bytes(b"no picture: the dry engine opens none")
    return {"id": prop_id, "image_path": str(pictures[0]), "out_path": str(folder / "mesh" / "mesh.check.partial.glb"),
            "seed": 7, "resolution": 1024, "decimation_target": 300000, "texture_size": 2048,
            "views": [{"image_path": str(p), "rect": None, "azimuth": a} for p, a in zip(pictures, views)] or None,
            "views_dir": None}


def judge_batch(name: str, record: dict[str, Any] | None, polls: Polls, items: list[dict[str, Any]],
                cases: dict[str, common.Case], several: bool = True) -> list[common.Result]:
    """The record of a finished batch against what was asked for and what the golden runs answered. `several`
    false: the launcher has no multi-view models, so a prop with two or more views must fail for that alone."""
    if record is None:
        return [result(f"{name}: a record", False, "the launcher never answered a poll with one")]
    out = [result(f"{name}: ended done", record["state"] == "done" and record["error"] is None,
                  f"state {record['state']}, error {record['error']}, {record['seconds']} s"),
           result(f"{name}: the record only moved forward", not polls.backwards,
                  "; ".join(polls.backwards[:5]) or f"{polls.count} polls, seq up to {polls.seq}")]
    for item in items:
        prop_id = item["id"]
        entry = record["items"].get(prop_id) or {}
        answer = entry.get("result") or {}
        if not several and len(item["views"] or []) >= 2:
            out.append(result(f"{prop_id}: fails for its missing multi-view models, alone",
                              entry.get("state") == "failed" and (entry.get("error") or {}).get("kind") == "mv_missing",
                              f"{entry.get('state')} {entry.get('error')}"))
            continue
        out.append(result(f"{prop_id}: done", entry.get("state") == "done",
                          f"{entry.get('state')} {entry.get('error') or ''}; states seen {polls.states.get(prop_id)}"))
        glb = Path(item["out_path"])
        size = glb.stat().st_size if glb.is_file() else 0
        out.append(result(f"{prop_id}: GLB written", size > (20 if not cases else 100_000),
                          f"{glb.name}: {size:,} bytes"))
        case = cases.get(prop_id)
        if case is None or not answer:
            continue
        golden = case.manifest.get("answer") or {}
        told = {key: answer.get(key) for key in ("mode", "views", "warning", "ss_retries")}
        out.append(result(f"{prop_id}: mode, views, warning, ss_retries", told == golden, f"{told} (golden {golden})"))
        out.append(result(f"{prop_id}: seed and resolution", answer.get("seed") == case.manifest.get("seed")
                          and answer.get("resolution") == case.manifest.get("res"),
                          f"seed {answer.get('seed')}, resolution {answer.get('resolution')}; slow "
                          f"{answer.get('slow')}; {answer.get('seconds')} s on the GPU thread; resumed "
                          f"{answer.get('resumed')}"))
        if item.get("views_dir"):
            saved = sorted(p.name for p in Path(item["views_dir"]).glob("*.png"))
            out.append(result(f"{prop_id}: views saved", len(saved) == len(item["views"] or [])
                              and saved[0].startswith("0_front"), ", ".join(saved)))
    return out


def recorded(scratch: Path) -> list[common.Result]:
    path = scratch / "results.jsonl"
    if not path.is_file():
        return []
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [common.Result(line["name"], line["ok"], line["rule"], line["detail"]) for line in lines]


@contextlib.contextmanager
def launcher(args: argparse.Namespace, scratch: Path) -> Iterator[subprocess.Popen[bytes]]:
    """The child process with the live launcher; stopped when the block is left, its tree with it."""
    command = [str(PYTHON), str(Path(__file__).resolve()), "--repo", str(args.repo), "--port", str(args.port),
               "--serve", str(scratch)] + (["--dry"] if args.dry else []) + (
                   ["--old-crop"] if args.old_crop else []) + (
                   ["--max-vram-mib", str(args.max_vram_mib)] if args.max_vram_mib else [])
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    with open(scratch / "launcher.log", "wb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                 cwd=str(args.repo))
        try:
            yield child
        finally:
            if child.poll() is None:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(child.pid)], capture_output=True, check=False)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    child.wait(30)


def drive(args: argparse.Namespace) -> int:
    scratch = OUT / f"run-{time.strftime('%H%M%S')}-{os.getpid()}"
    projects = scratch / "projects"
    projects.mkdir(parents=True)
    base = f"http://127.0.0.1:{args.port}"
    names = [] if args.dry else [name.strip() for name in args.cases.split(",") if name.strip()]
    cases = {f"p_{name}": common.golden(name) for name in names}
    if args.dry:
        items = [dry_request("p_one", projects, ()), dry_request("p_views", projects, (0, 90)),
                 dry_request("p_two", projects, ())]
    else:
        items = [scratch_request(case, prop_id, projects) for prop_id, case in cases.items()]
    (scratch / "plan.json").write_text(json.dumps({"cases": dict(zip(cases, names))}), encoding="utf-8")
    work = projects / "check" / "temp" / "mesh-check1"
    results: list[common.Result] = []
    minutes = 2 if args.dry else 6 * len(items) + (12 if args.reference else 0)
    purpose = f"check_launcher {'dry' if args.dry else ','.join(names)}{' + reference' if args.reference else ''}"

    with common.gpu_lock("WP7", purpose, expected_minutes=minutes), launcher(args, scratch) as child:
        spawned = time.monotonic()
        health: Any = None
        while child.poll() is None and time.monotonic() - spawned < 300:
            status, health, _ = http("GET", f"{base}/spellcast/health", timeout=2.0)
            if status == 200 and isinstance(health, dict) and health.get("ready") is True:
                break
            time.sleep(0.25)
        ready = time.monotonic() - spawned
        if child.poll() is not None or not isinstance(health, dict) or health.get("ready") is not True:
            results.append(result("the launcher starts", False, f"exit code {child.poll()} after {ready:.0f} s; its "
                                                                f"log: {scratch / 'launcher.log'}"))
            print((scratch / "launcher.log").read_text(encoding="utf-8", errors="replace")[-4000:], flush=True)
            return common.report(results)
        results.append(result("ready soon after the start", ready < READY_LIMIT,
                              f"{ready:.1f} s from the spawn to ready (asked: under {READY_LIMIT:.0f} s)"))
        expected = {"ready": True, "low_vram": True, "multiview": True, "batch": True}
        if args.dry:
            expected["multiview"] = bool(health.get("multiview"))
        results.append(result("health", health == expected, json.dumps(health)))

        body = {"batch_id": "check1", "work_dir": str(work), "wave_size": 0, "resume": False, "items": items}
        status, taken, seconds = http("POST", f"{base}/spellcast/batch", body)
        first = [item["id"] for item in items if len(item["views"] or []) >= 2]
        order = first + [item["id"] for item in items if item["id"] not in first]
        results.append(result("the batch is taken", status == 202 and (taken or {}).get("order") == order,
                              f"HTTP {status} in {seconds:.2f} s, order {(taken or {}).get('order')}"))
        status, busy, _ = http("POST", f"{base}/spellcast/batch", {**body, "batch_id": "check2"})
        results.append(result("a second batch meanwhile is refused",
                              status == 409 and (busy or {}).get("kind") == "busy", f"HTTP {status} {busy}"))
        record, polls = follow(base, "check1", child, 3600.0 if not args.dry else 120.0)
        results += judge_batch("batch 1", record, polls, items, cases, bool(health.get("multiview")))
        results.append(result("networks were named while they were read", bool(polls.loading) or args.dry,
                              ", ".join(polls.loading) or "none seen"))
        results.append(result("the bars of the mesh clean-up reached the record",
                              "Cleaning up the mesh and baking textures" in polls.words, "; ".join(polls.words)))
        print(f"MEASURE  polls of batch 1: {polls.count}, {polls.unanswered} without an answer; the slowest answer "
              f"took {polls.slowest:.2f} s (at {polls.slowest_at}); the slowest while a mesh was built and saved "
              f"{polls.slowest_finishing:.2f} s (the hub's read timeout is 10 s; over {POLL_LIMIT:.0f} s it has to "
              f"grow to 30 s)", flush=True)
        results.append(result("a poll is answered while a mesh is saved", polls.slowest_finishing <= POLL_LIMIT,
                              f"slowest {polls.slowest_finishing:.2f} s during finish, {polls.slowest:.2f} s overall"))

        if args.dry:        # a cancel half-way: the launcher's own plumbing, nothing of the engine's
            http("POST", f"{base}/spellcast/batch", {**body, "batch_id": "check2"})
            time.sleep(2.0)
            status, told, _ = http("POST", f"{base}/spellcast/batch/check2/cancel", {})
            record, _ = follow(base, "check2", child, 60.0)
            ends = {prop_id: item["state"] for prop_id, item in ((record or {}).get("items") or {}).items()}
            results.append(result("a cancel ends the batch", status == 200 and told == {"cancelling": True}
                                  and (record or {}).get("state") == "cancelled" and "done" not in ends.values(),
                                  f"HTTP {status} {told}; state {(record or {}).get('state')}, props {ends}"))
        if args.reference and names:
            results += reference_turns(base, child, projects, cases, names[-1], items)
        results += recorded(scratch)
        exit_code = child.poll()
        results.append(result("the launcher is still running at the end", exit_code is None, f"exit code {exit_code}"))

    if not args.dry and (scratch / "autotune_cache.json").is_file():
        golden_choices = (common.golden_root() / "autotune_cache.json").read_bytes()
        tuned = (scratch / "autotune_cache.json").read_bytes() != golden_choices
        print("NOTE  flex_gemm " + ("tuned a kernel in this run (a size its cache had no choice for): the decoders' "
                                    "sums may be in another order than in the golden run." if tuned else
                                    "used the golden runs' kernel choices throughout."), flush=True)
    log = OUT / f"{scratch.name}-launcher.log"
    shutil.copyfile(scratch / "launcher.log", log)
    print(f"the launcher's log: {log}", flush=True)
    for _ in range(20 if not args.keep else 0):     # the stopped launcher lets go of its log a moment later
        shutil.rmtree(scratch, ignore_errors=True)
        if not scratch.exists():
            break
        time.sleep(0.5)
    return common.report(results)


def reference_turns(base: str, child: subprocess.Popen[bytes], projects: Path, cases: dict[str, common.Case],
                    name: str, items: list[dict[str, Any]]) -> list[common.Result]:
    """After a batch: /spellcast/generate still makes a mesh (its pipeline is loaded now, not at the start), and
    a batch after it finds that pipeline dropped (the recorder in the child says so) and is made as the first."""
    out: list[common.Result] = []
    item = dict(next(item for item in items if item["id"] == f"p_{name}"))
    item.pop("id")
    item["out_path"] = item["out_path"].replace("mesh.check", "mesh.reference")
    status, answer, seconds = http("POST", f"{base}/spellcast/generate", item, timeout=3600.0)
    golden = cases[f"p_{name}"].manifest.get("answer") or {}
    told = {key: (answer or {}).get(key) for key in ("mode", "views", "warning", "ss_retries")}
    out.append(result("the reference route after a batch", status == 200 and told == golden
                      and Path(item["out_path"]).is_file(),
                      f"HTTP {status} after {seconds:.0f} s (its models were loaded for it); {told}"))
    again = dict(item, id=f"p_{name}", out_path=item["out_path"].replace("mesh.reference", "mesh.again"))
    body = {"batch_id": "check3", "work_dir": str(projects / "check" / "temp" / "mesh-check3"), "items": [again]}
    status, _, _ = http("POST", f"{base}/spellcast/batch", body)
    record, polls = follow(base, "check3", child, 3600.0)
    out.append(result("a batch after the reference route is taken", status == 202, f"HTTP {status}"))
    return out + judge_batch("batch after the reference", record, polls, [again], cases)


# =====================================================================================================================
# --fans: recentre_fans on a golden decoded mesh
# =====================================================================================================================

def source_function(file: Path, name: str, namespace: dict[str, Any],
                    of_class: str | None = None) -> Callable[..., Any]:
    """A function (a method of `of_class`) compiled from a source file alone, without importing its package."""
    body = ast.parse(file.read_text(encoding="utf-8")).body
    if of_class is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == of_class).body
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(file), "exec"), namespace)
    return namespace[name]


def fans(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(LAUNCHERS))
    import pixal3d_server as server

    server.configure_env(args.repo, True, "flash_attn")
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    import torch
    from pixal3d.representations.mesh.base import Mesh

    case = common.golden(args.case)
    before_fix = source_function(common.LAUNCHER / "pixal3d_server.py", "recentre_fans", {"Any": Any})  # the frozen one
    results: list[common.Result] = []
    with common.gpu_lock("WP7", f"check_launcher fans {args.case}", expected_minutes=2):
        raw_vertices = case.tensor("11-decode", "vertices_raw", "cuda")
        raw_faces = case.tensor("11-decode", "faces_raw", "cuda")
        first: Any = None
        for turn in (1, 2):         # the second mesh of a process is the one cumesh fills out of place
            mesh = Mesh(raw_vertices.clone(), raw_faces.clone())
            before = int(mesh.vertices.shape[0])
            torch.cuda.synchronize()
            mesh.fill_holes()       # cumesh's own: the launcher's repair is not installed in this process
            filled, faces = mesh.vertices, mesh.faces
            added = int(filled.shape[0]) - before
            theirs, ours = filled.clone(), filled.clone()
            with common.peak() as old:
                wrong_old = before_fix(theirs, faces, before)
            with common.peak() as new:
                wrong_new = server.recentre_fans(ours, faces, before)
            results.append(common.compare(f"turn {turn}: vertices of the two recentre_fans", ours, theirs))
            results.append(result(f"turn {turn}: centres found out of place", wrong_old == wrong_new,
                                  f"{wrong_new:,} of {added:,} added vertices (frozen launcher: {wrong_old:,}); "
                                  f"{int(faces.shape[0]):,} faces; peak above the mesh "
                                  f"{new.allocated_mib - new.start_allocated_mib} MiB, frozen "
                                  f"{old.allocated_mib - old.start_allocated_mib} MiB"))
            if turn == 1:
                first = ours
                for what, tensor in (("vertices", ours), ("faces", faces)):
                    same = common.sha256_of(tensor) == case.manifest.get(f"{what}_sha")
                    results.append(result(f"turn 1: {what} of the golden run", same,
                                          f"{int(tensor.shape[0]):,} rows (golden {case.manifest.get(what):,}), "
                                          f"SHA-256 {'equal' if same else 'DIFFERENT'}"))
            else:
                results.append(common.compare("turn 2: vertices of turn 1 (a fresh process's)", ours, first))
            del mesh, filled, faces, theirs, ours
    return common.report(results)


# =====================================================================================================================
# --pictures: the hooks prepare and bundle against the golden inputs, on the CPU
# =====================================================================================================================

def pictures(args: argparse.Namespace) -> int:
    import tempfile
    import types

    from PIL import Image

    sys.path.insert(0, str(LAUNCHERS))
    import pixal3d_mv
    import pixal3d_server as server

    pixal3d_mv.CROP_FIRST = not args.old_crop

    class PropFailed(Exception):
        def __init__(self, detail: str, kind: str = "error") -> None:
            super().__init__(detail)
            self.detail, self.kind = detail, kind

    def no_matte(picture: Any) -> Any:
        raise AssertionError("background removal was asked for: this picture has no transparency of its own")

    server.staged_module = lambda: types.SimpleNamespace(Prepared=types.SimpleNamespace, PropFailed=PropFailed)
    px = types.SimpleNamespace(_update_progress=lambda stage, step, total: None)
    hooks = server.BatchHooks(px, server.RunNotes())
    wanted = [name.strip() for name in args.cases.split(",")] if "--cases" in sys.argv else sorted(
        folder.name for folder in common.golden_root().iterdir() if (folder / "manifest.json").is_file())
    skipped: list[str] = []
    with tempfile.TemporaryDirectory() as scratch:
        recorder = Recorder(Path(scratch), {f"p_{name}": name for name in wanted})
        for prop_id, case in recorder.cases.items():
            request = dict(case.request)
            views = request.get("views") or []
            files = [view["image_path"] for view in views] or [request["image_path"]]
            opaque = []
            for file in files:
                with Image.open(case.input(file)) as picture:
                    if not ("A" in picture.getbands() and picture.getchannel("A").getextrema()[0] < 255):
                        opaque.append(file)
            if opaque:
                skipped.append(f"{case.name} ({', '.join(opaque)} needs background removal)")
                continue
            payload = server.BatchItem(
                id=prop_id, image_path=str(case.input(request["image_path"])), out_path="unused.glb",
                seed=request["seed"], resolution=request["resolution"], views_dir=None,
                views=[{**view, "image_path": str(case.input(view["image_path"]))} for view in views] or None)
            req = types.SimpleNamespace(id=prop_id, multi=len(views) >= 2, payload=payload)
            prepared = hooks.prepare(req, no_matte)
            recorder.guarded(req, "prepare", lambda case: recorder.prepared(req, case, prepared))
            if req.multi:
                bundle = hooks.bundle(req, prepared.images, prepared.meta)
                recorder.guarded(req, "bundle", lambda case: recorder.bundle(req, case, bundle))
        results = recorded(Path(scratch))
    for line in skipped:
        print(f"SKIPPED  {line}", flush=True)
    return common.report(results)


def main() -> int:
    args = parse()
    if args.serve:
        return serve(args)
    if args.fans:
        return fans(args)
    if args.pictures:
        return pictures(args)
    return drive(args)


if __name__ == "__main__":
    sys.exit(main())
