"""The profiled entry of Qwen-Image 2.1 (spellcast/profiling/qwen_profiled.py): launchers/qwen_server.py, unchanged,
with the per-request profilers attached from outside.

The attach points must exist on the CURRENT qwen_server.py (a rename there breaks a test here, not a developer's
profiling run), a request's sd.cpp log lines and progress callbacks must give the profiler exactly its phases and
steps, on the GPU thread, and the profiling must never change the server's answers. Runs in the hub venv: the profiler
is a fake (no torch, no pyinstrument), the engine a fake that plays stable-diffusion.cpp's callbacks on the real
Progress; with every switch off the real RequestProfiler is used (it imports nothing then).
"""

from __future__ import annotations

import inspect
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import pytest
from fastapi.testclient import TestClient

from spellcast import config, paths

real_request_profiler = pytest.importorskip("spellcast.profiling.request_profiler")

from spellcast.profiling import qwen_profiled as entry  # noqa: E402

import qwen_sdcpp as sdcpp  # noqa: E402  (the launchers' folder is on sys.path once the entry is imported)
import qwen_server as server  # noqa: E402

OFF = {key: False for key in config.PROFILING_KEYS}
TORCH_ON = {**OFF, "use_torch_profiler": True}


class FakeProfiler:
    """Records what the entry asks of a RequestProfiler, and on which thread."""
    made: list[FakeProfiler] = []
    slow_finish = 0.0
    fail_mark = False

    def __init__(self, switches: Mapping[str, bool], label: str, info: dict[str, Any]) -> None:
        self.switches, self.label, self.info = dict(switches), label, info
        self.active = any(switches.values())
        self.thread = threading.get_ident()
        self.calls: list[tuple[Any, ...]] = []
        self.notes: dict[str, Any] = {}
        self.threads: set[int] = set()
        self.finished_at: float | None = None

    @classmethod
    def begin(cls, switches: Mapping[str, bool], label: str, info: dict[str, Any], root: Path | None = None,
              log: Callable[[str], None] = print) -> FakeProfiler:
        profiler = cls(switches, label, info)
        cls.made.append(profiler)
        return profiler

    def mark(self, phase: str, steps: int | None = None) -> None:
        self.threads.add(threading.get_ident())
        if FakeProfiler.fail_mark:
            raise RuntimeError("a broken profiler")
        self.calls.append(("mark", phase, steps))

    def step(self, step: int, steps: int) -> None:
        self.threads.add(threading.get_ident())
        self.calls.append(("step", step, steps))

    def note(self, key: str, value: Any) -> None:
        self.notes[key] = value

    def finish(self, outcome: str = "ok") -> None:
        self.threads.add(threading.get_ident())
        time.sleep(FakeProfiler.slow_finish)
        self.calls.append(("finish", outcome))
        self.finished_at = time.monotonic()


@pytest.fixture
def switches(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    """The config file's switches as the entry reads them at every request (change the dict to switch)."""
    now = dict(TORCH_ON)
    monkeypatch.setattr(entry.config, "profiling", lambda: dict(now))
    return now


@pytest.fixture
def attached(monkeypatch: pytest.MonkeyPatch, switches: dict[str, bool]) -> SimpleNamespace:
    """The entry attached to qwen_server with a FakeProfiler; everything it wraps is put back after the test (the other
    launcher tests share the module). QwenEngine.generate is first replaced by `simulate` (an Engine's own), so the
    entry wraps what stands in for sd.cpp's generate_image."""
    for name in entry.TARGETS:
        owner: Any = server
        *parents, attr = name.split(".")
        for part in parents:
            owner = getattr(owner, part)
        monkeypatch.setattr(owner, attr, getattr(owner, attr))
    monkeypatch.setattr(server.QwenEngine, "generate", lambda self, req, seed, refs: self.simulate(req, seed, refs))
    monkeypatch.setattr(entry, "current", None)
    monkeypatch.setattr(entry, "_seen", {})
    monkeypatch.setattr(entry, "_request", {})
    monkeypatch.setattr(entry, "_failed", set())
    fake = SimpleNamespace(RequestProfiler=FakeProfiler, preloaded=[], missing=[])
    fake.preload = lambda s: fake.preloaded.append(dict(s))
    fake.missing_packages = lambda s: list(fake.missing)
    monkeypatch.setattr(entry, "request_profiler", fake)
    monkeypatch.setattr(FakeProfiler, "made", [])
    monkeypatch.setattr(FakeProfiler, "slow_finish", 0.0)
    monkeypatch.setattr(FakeProfiler, "fail_mark", False)
    entry.attach(server)
    return fake


# ---- the attach points -------------------------------------------------------------------------------------------

def test_every_attach_point_is_on_the_current_server() -> None:
    assert entry.missing(server) == []
    params = {name: list(inspect.signature(fn).parameters) for name, fn in (
        ("build_app", server.build_app), ("run", server.GpuWorker.run), ("generate", server.QwenEngine.generate),
        ("_enter", server.Progress._enter), ("on_log", server.Progress.on_log), ("on_step", server.Progress.on_step))}
    assert params["build_app"][:5] == ["engine", "gpu", "progress", "projects_root", "info"]
    assert params["run"] == ["self", "fn", "args"]
    assert params["generate"] == ["self", "req", "seed", "references", "init", "mask"]
    assert params["_enter"][:2] == ["self", "phase"]
    assert params["on_log"] == ["self", "level", "text"] and params["on_step"] == ["self", "step", "steps", "seconds"]
    # the picture request is build_app's closure "work", run on the GPU thread; sd.cpp's line after the text encoder
    source = inspect.getsource(server.build_app)
    assert f"def {entry.WORK}(" in source and f"gpu.run({entry.WORK})" in source
    assert entry.PREPARED in inspect.getsource(server.Progress.on_log)
    assert set(entry.NOTED) <= set(server.GenerateRequest.model_fields)
    assert set(entry.DRAWN) <= set(server.sampling("full", 25, 1.0, "euler", None, (8, 6)))
    gone = entry.missing(SimpleNamespace(main=print))
    assert gone[:-1] == [name for name in entry.TARGETS if name != "main"] and gone[-1].startswith("Progress() (")


def test_the_entry_starts_like_the_server() -> None:
    """The entry imports the launchers' modules as qwen_server.py does (its folder first, code/ at the end)."""
    import sys

    assert str(paths.LAUNCHERS) in sys.path and str(paths.CODE) in sys.path
    assert entry.LAUNCHERS == paths.LAUNCHERS and entry.CODE == paths.CODE and entry.HERE == paths.PACKAGE / "profiling"
    assert server.__name__ == "qwen_server" and not hasattr(server, "request_profiler")      # nothing added to it


# ---- one picture request -----------------------------------------------------------------------------------------

class FakePicture:
    transparent = False
    width = height = 1               # what the launcher's Pictures cache weighs

    def getbands(self) -> tuple[str, ...]:
        return ("R", "G", "B", "A")


class Engine(server.QwenEngine):
    """A QwenEngine whose sd.cpp is played by simulate(): the log lines and progress callbacks of a request with one
    example picture and two variations, on the real Progress (as stable-diffusion.dll calls them, on the GPU thread)."""

    def __init__(self, progress: server.Progress, fail: BaseException | None = None) -> None:   # no DLL, no models
        self.progress, self.fail = progress, fail
        self.ready, self.loading, self.has_vision, self.lib = True, False, True, None
        self.placement = {"name": "offload", "budget_mb": 9_000, "prefix_cache": "q8_0"}
        self.load_seconds = 2.5

    def cancel(self) -> None:
        self.progress.cancel.set()

    def log(self, text: str) -> None:
        self.progress.on_log(sdcpp.SD_LOG_INFO, f"[INFO ] image.cpp:1   - {text}")

    def simulate(self, req: server.GenerateRequest, seed: int, references: list[Any]) -> list[Any]:
        self.log("encode_first_stage completed, taking 1.50s")
        self.progress.on_step(5, 9, 0.2)                     # the vision encoder's own steps: not the picture's
        self.log("get_learned_condition completed, taking 4.20s")
        for image in range(1, req.batch_count + 1):
            self.log(f"generating image: {image}/{req.batch_count} - seed {seed + image - 1}")
            for step in range(req.steps + 1):
                self.progress.on_step(step, req.steps, 2.0)
            self.log("sampling completed, taking 6.00s")
        if self.fail is not None:
            raise self.fail
        self.log(f"decoding {req.batch_count} latents")
        self.progress.on_step(1, 4, 0.1)                     # the VAE's tiles: not drawing steps
        for image in range(1, req.batch_count + 1):
            self.log(f"latent {image} decoded, taking 3.00s")
        return [FakePicture() for _ in range(req.batch_count)]


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A projects folder with one example picture; pictures read and saved without Pillow."""
    projects = tmp_path / "projects"
    (projects / "p1").mkdir(parents=True)
    (projects / "p1" / "ref.png").write_bytes(b"png")

    def save(picture: Any, out_dir: Path, seed: int, transparent: bool = True) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"qwen-{seed}.png"
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr(server, "load_reference", lambda path, decoded=None: FakePicture())
    monkeypatch.setattr(server, "save_picture", save)
    monkeypatch.setattr(server, "has_transparency", lambda picture: False)
    return projects.resolve()


def _client(root: Path, engine: Engine) -> TestClient:
    info = {"gpu": "Fake RTX 3060", "vram_mb": 12288, "placement": "offload", "quality": "Q4_K_M", "sdcpp": "x"}
    return TestClient(server.build_app(engine, server.GpuWorker(), engine.progress, root, info,
                                       cuda_works=lambda: True))


def _body(root: Path, **changes: Any) -> dict[str, Any]:
    # the full model: its steps and guidance as asked (the Turbo draws its own 8 steps without guidance)
    return {"prompt": "a wooden chest", "model": "full", "width": 512, "height": 512, "steps": 3, "cfg_scale": 6.0,
            "seed": 7,
            "batch_count": 2, "ref_images": [str(root / "p1" / "ref.png")], "transparent": True,
            "out_dir": str(root / "p1" / "out"), **changes}


def test_a_request_is_profiled_phase_by_phase_on_the_gpu_thread(attached: SimpleNamespace, root: Path) -> None:
    progress = server.Progress()
    client = _client(root, Engine(progress))
    r = client.post("/spellcast/generate", json=_body(root))
    assert r.status_code == 200, r.text
    [profiler] = FakeProfiler.made
    drawing = [("step", k, 3) for k in (1, 2, 3)]
    assert profiler.calls == [("mark", "references", None), ("mark", "prompt", None), ("mark", "prepare", None),
                              ("mark", "drawing 1/2", 3), *drawing, ("mark", "drawing 2/2", 3), *drawing,
                              ("mark", "finishing", None), ("mark", "saving", None), ("finish", "ok")]
    # begun, marked and finished on the GPU worker's thread (pyinstrument and torch.profiler record the thread that
    # starts them; sd.cpp calls back on it), never on the route's
    assert profiler.threads == {profiler.thread} and profiler.thread != threading.get_ident()
    assert profiler.label == "qwen" and profiler.switches == TORCH_ON
    assert profiler.info["gpu"] == "Fake RTX 3060" and profiler.info["placement_details"]["budget_mb"] == 9_000
    assert profiler.info["load_seconds"] == 2.5
    assert {k: profiler.notes[k] for k in ("width", "height", "steps", "cfg_scale", "sampler", "seed", "batch_count",
                                           "references", "transparent", "prompt_characters")} == {
        "width": 512, "height": 512, "steps": 3, "cfg_scale": 6.0, "sampler": "euler", "seed": 7, "batch_count": 2,
        "references": 1, "transparent": True, "prompt_characters": len("a wooden chest")}
    assert profiler.notes["pictures"] == 2 and profiler.notes["seconds"] == pytest.approx(r.json()["seconds"], abs=0.1)
    assert profiler.notes["timings"] == r.json()["timings"] == {
        "references": 1.5, "prompt": 4.2, "drawing_1": 6.0, "drawing_2": 6.0, "finishing_1": 3.0, "finishing_2": 3.0}
    assert entry.current is None and progress.phase == "idle"

    # without example pictures it starts at the prompt; one picture, the next request has a profiler of its own
    r = client.post("/spellcast/generate", json=_body(root, ref_images=[], batch_count=1, steps=2))
    assert r.status_code == 200 and len(FakeProfiler.made) == 2
    assert FakeProfiler.made[1].calls == [("mark", "prompt", None), ("mark", "prepare", None),
                                          ("mark", "drawing 1/1", 2), ("step", 1, 2), ("step", 2, 2),
                                          ("mark", "finishing", None), ("mark", "saving", None), ("finish", "ok")]


def test_only_the_picture_request_is_profiled(attached: SimpleNamespace) -> None:
    gpu = server.GpuWorker()

    def works() -> bool:                    # the CUDA check after a failure, as build_app runs it
        return True

    assert gpu.run(works) is True and gpu.run(lambda x: x * 2, 4) == 8 and FakeProfiler.made == []

    def work() -> tuple[Any, ...]:
        return None, [Path("a.png")], {"timings": {"prompt": 1.0}}, 1.25

    assert gpu.run(work) == (None, [Path("a.png")], {"timings": {"prompt": 1.0}}, 1.25)
    [profiler] = FakeProfiler.made
    assert profiler.calls == [("finish", "ok")]
    assert profiler.notes == {"pictures": 1, "seconds": 1.25, "timings": {"prompt": 1.0}}


def test_the_reports_are_written_after_the_request_and_never_timed(attached: SimpleNamespace, root: Path) -> None:
    FakeProfiler.slow_finish = 0.6
    client = _client(root, Engine(server.Progress()))
    started = time.monotonic()
    r = client.post("/spellcast/generate", json=_body(root, batch_count=1, steps=2))
    assert r.status_code == 200 and time.monotonic() - started >= 0.55      # the answer waits for the reports...
    assert r.json()["seconds"] < 0.3                                         # ...whose time is not the picture's
    assert FakeProfiler.made[0].calls[-1] == ("finish", "ok")


@pytest.mark.parametrize("fail, status, outcome", [
    (server.Cancelled(), 409, "cancelled"),
    (server.OutOfMemory("CUDA error: out of memory"), 500, "oom"),
    (RuntimeError("ggml says no"), 500, "error"),
])
def test_a_failed_or_cancelled_request_still_finishes_its_profile(attached: SimpleNamespace, root: Path,
                                                                  fail: Exception, status: int, outcome: str) -> None:
    client = _client(root, Engine(server.Progress(), fail=fail))
    r = client.post("/spellcast/generate", json=_body(root, batch_count=1, steps=2))
    assert r.status_code == status and r.json()["kind"] == outcome
    [profiler] = FakeProfiler.made
    assert profiler.calls[-1] == ("finish", outcome) and profiler.notes["failure"] == r.json()["detail"]
    assert profiler.notes["pictures"] == 0 and entry.current is None


def test_a_request_that_raises_finishes_as_an_error(attached: SimpleNamespace) -> None:
    def work() -> None:
        raise ValueError("a bug in the server")

    with pytest.raises(ValueError, match="a bug"):
        server.GpuWorker().run(work)
    assert FakeProfiler.made[0].calls == [("finish", "error")] and entry.current is None


def test_a_broken_hook_is_said_once_and_never_reaches_the_server(attached: SimpleNamespace, root: Path,
                                                                 capsys: pytest.CaptureFixture[str]) -> None:
    FakeProfiler.fail_mark = True
    client = _client(root, Engine(server.Progress()))
    for _ in range(2):
        r = client.post("/spellcast/generate", json=_body(root, batch_count=1, steps=2))
        assert r.status_code == 200 and r.json()["timings"]["drawing_1"] == 6.0
    out = capsys.readouterr().out
    assert out.count("The profiling's phase hook failed (RuntimeError: a broken profiler)") == 1
    assert [p.calls[-1] for p in FakeProfiler.made] == [("finish", "ok")] * 2
    assert [c for c in FakeProfiler.made[0].calls if c[0] == "step"] == [("step", 1, 2), ("step", 2, 2)]


def test_switches_are_read_at_every_request(attached: SimpleNamespace, switches: dict[str, bool]) -> None:
    def work() -> tuple[Any, ...]:
        return None, [], {}, 0.1

    gpu = server.GpuWorker()
    gpu.run(work)
    switches.update(OFF)
    gpu.run(work)
    assert [p.active for p in FakeProfiler.made] == [True, False]


def test_with_every_switch_off_the_profiler_is_inactive(attached: SimpleNamespace, switches: dict[str, bool],
                                                        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The real RequestProfiler: inactive, no folder, nothing imported; the hooks leave it alone."""
    switches.update(OFF)
    made: list[Any] = []
    real = real_request_profiler.RequestProfiler.begin

    def begin(switches: Mapping[str, bool], label: str, info: dict[str, Any], root: Path | None = None,
              log: Callable[[str], None] = print) -> Any:
        made.append(real(switches, label, info, root=tmp_path / "reports", log=log))
        return made[-1]

    monkeypatch.setattr(attached, "RequestProfiler", SimpleNamespace(begin=begin))
    progress = server.Progress()

    def work() -> tuple[Any, ...]:
        progress.start(1, 2, references=0)
        progress.on_log(sdcpp.SD_LOG_INFO, "generating image: 1/1 - seed 3")
        progress.on_step(1, 2, 1.0)
        progress.stop()
        return None, [], {}, 0.1

    assert server.GpuWorker().run(work) == (None, [], {}, 0.1)
    [profiler] = made
    assert profiler.active is False and profiler.folder is None and not (tmp_path / "reports").exists()


# ---- main ----------------------------------------------------------------------------------------------------------

@pytest.fixture
def quiet_logger(monkeypatch: pytest.MonkeyPatch) -> logging.Logger:
    logger = logging.getLogger("spellcast.profiling")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logger.level)
    return logger


def test_main_attaches_then_runs_the_server(attached: SimpleNamespace, quiet_logger: logging.Logger,
                                            monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    attached.missing = ["torch"]
    calls: list[str] = []
    monkeypatch.setattr(entry, "attach", lambda srv=server: calls.append("attach"))
    monkeypatch.setattr(server, "main", lambda: calls.append("main") or 0)
    assert entry.main() == 0 and calls == ["attach", "main"]
    # no handler of its own: RequestProfiler routes the harness's lines while it profiles (two would print each twice)
    assert attached.preloaded == [TORCH_ON] and not quiet_logger.handlers
    out = capsys.readouterr().out
    assert "switched on in" in out and "use_torch_profiler" in out and str(paths.LOGS / "profiling") in out
    assert "torch is not installed" in out and "close Spellcast3D and start it again" in out


def test_a_renamed_attach_point_runs_the_server_unprofiled(attached: SimpleNamespace, quiet_logger: logging.Logger,
                                                           monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[str] = []
    monkeypatch.delattr(server.Progress, "on_step")
    monkeypatch.setattr(entry, "attach", lambda srv=server: calls.append("attach"))
    monkeypatch.setattr(server, "main", lambda: calls.append("main") or 0)
    assert entry.main() == 0 and calls == ["main"] and attached.preloaded == []
    out = capsys.readouterr().out
    assert "PROFILING IS OFF" in out and "no longer has Progress.on_step" in out
