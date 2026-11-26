"""LATO.2: its launcher's own logic (launchers/lato2_server.py, lato2_faces.py, lato2_prepare.py, lato2_render.py),
the flash_attn stand-in shared with TriFlow (launchers/attn_shim.py) and the hub's adapter (adapters/lato2.py).

Runs in the hub venv (fastapi + pydantic + httpx, no numpy / torch): the launcher files keep those imports inside
functions. What needs numpy, scipy and trimesh (face recovery, hole closing, the solid, the OBJ writer) or torch
(the SDPA shim against plain attention, the rasterizer) is skipped there and runs in a venv that has them, e.g.
the lato2 venv:  <venv>\\Scripts\\python.exe -m pytest -p no:cacheprovider code/tests/test_lato2_launcher.py
The models themselves run only on the GPU (the launcher's own run, not a test).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from spellcast import gpu, paths, projects
from spellcast.adapters import lato2 as adapter_module
from spellcast.adapters.base import StageContext, StageError, ToolNeedsRestart

if str(paths.LAUNCHERS) not in sys.path:     # the launchers import their siblings as top-level modules
    sys.path.insert(0, str(paths.LAUNCHERS))

import attn_shim  # noqa: E402
import lato2_server as server  # noqa: E402

HAVE_NUMPY = all(importlib.util.find_spec(m) is not None for m in ("numpy", "scipy", "trimesh"))
HAVE_TORCH = importlib.util.find_spec("torch") is not None
needs_numpy = pytest.mark.skipif(not HAVE_NUMPY, reason="needs numpy, scipy and trimesh (the lato2 venv)")
needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="needs torch (the lato2 venv)")


@pytest.fixture()
def clean_modules() -> Any:
    """Puts sys.modules' flash_attn / torch_scatter / o_voxel entries back as they were."""
    names = ("flash_attn", "flash_attn.flash_attn_interface", "torch_scatter", "o_voxel")
    saved = {name: sys.modules.get(name) for name in names}
    yield
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


# ---- the flash_attn stand-in ------------------------------------------------------------------------------------

def test_the_shim_is_importable_as_flash_attn(clean_modules: Any) -> None:
    module = attn_shim.install_flash_attn_shim()
    assert attn_shim.is_shim_installed()
    import flash_attn
    from flash_attn import flash_attn_varlen_func
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func

    assert flash_attn is module and flash_attn_varlen_func is attn_shim.flash_attn_varlen_func
    assert flash_attn_qkvpacked_func is attn_shim.flash_attn_qkvpacked_func
    for name in attn_shim.FUNCTIONS:     # every function the real package exports
        assert callable(getattr(flash_attn, name))
    assert importlib.util.find_spec("flash_attn") is not None    # transformers asks this; a None spec would raise
    assert flash_attn.__version__.endswith("+sdpa")
    assert attn_shim.install_flash_attn_shim() is module          # idempotent
    with pytest.raises(NotImplementedError):
        flash_attn.flash_attn_with_kvcache(None, None, None)
    with pytest.raises(NotImplementedError, match="softcap"):
        flash_attn.flash_attn_func(None, None, None, softcap=1.0)
    with pytest.raises(NotImplementedError, match="alibi"):
        flash_attn.flash_attn_varlen_func(None, None, None, None, None, 1, 1, alibi_slopes=[1.0])


def test_the_shim_replaces_a_flash_attn_already_imported(clean_modules: Any) -> None:
    import types

    sys.modules["flash_attn"] = types.ModuleType("flash_attn")
    sys.modules["flash_attn.flash_attn_interface"] = types.ModuleType("flash_attn.flash_attn_interface")
    attn_shim.install_flash_attn_shim()
    assert attn_shim.is_shim_installed()
    assert getattr(sys.modules["flash_attn.flash_attn_interface"], attn_shim.SHIM_ATTR)


# ---- launcher: choices, requests, answers ---------------------------------------------------------------------------

@pytest.mark.parametrize("asked,capability,flash,expected", [
    ("auto", (8, 6), True, "flash_attn"), ("auto", (8, 6), False, "sdpa"), ("auto", (7, 5), True, "sdpa"),
    ("auto", (6, 1), True, "sdpa"), ("flash_attn", (7, 5), True, "sdpa"), ("sdpa", (8, 9), True, "sdpa"),
    ("flash_attn", (12, 0), True, "flash_attn")])
def test_flash_attn_only_where_its_kernels_are(asked: str, capability: tuple[int, int], flash: bool,
                                               expected: str) -> None:
    assert server.choose_attention(asked, capability, flash) == expected


@pytest.mark.parametrize("asked,capability,expected", [
    ("auto", (8, 6), "bf16"), ("auto", (12, 0), "bf16"), ("auto", (7, 5), "fp16"), ("auto", (7, 0), "fp16"),
    ("auto", (6, 1), "fp32"), ("bf16", (7, 5), "fp16"), ("bf16", (6, 1), "fp32"), ("fp32", (8, 6), "fp32"),
    ("fp16", (8, 6), "fp16")])
def test_precision_follows_the_card(asked: str, capability: tuple[int, int], expected: str) -> None:
    assert server.choose_precision(asked, capability) == expected


def test_the_limits_agree_between_hub_and_launcher() -> None:
    """LATO.2 is asked for every size's vertices, from its least to what its models take (a size outside them is made
    with the nearest)."""
    assert (adapter_module.MIN_VERTICES, adapter_module.MAX_VERTICES) == (server.MIN_VERTICES, server.MAX_VERTICES)
    asked = [adapter_module.vertices_for(step / 100) for step in range(round(projects.RETOPO_MIN_SIZE * 100),
                                                                      projects.RETOPO_MAX_SIZE * 100 + 1)]
    assert min(asked) == server.MIN_VERTICES == 200 and max(asked) == server.MAX_VERTICES == 10_000
    assert asked[:12] == [200] * 11 + [210], "0.1k to 0.2k are made with 200, then every size's own count"
    assert adapter_module.vertices_for(projects.STAGE_DEFAULTS["retopo"]["settings"]["sizes"][0]) == \
        server.RetopoRequest.model_fields["vertices"].default
    assert set(adapter_module.LOW_VRAM) == {"auto", "on", "off"}
    args = server.parse_args(["--repo", "r", "--models", "m", "--port", "7774", "--low-vram", "on"])
    assert (args.low_vram, args.attn, args.precision) == ("on", "auto", "auto")


def _request(root: Path, **extra: Any) -> server.RetopoRequest:
    return server.RetopoRequest(mesh_path=str(root / "props" / "p1" / "mesh" / "mesh.glb"),
                                out_path=str(root / "props" / "p1" / "retopo" / "lato2.raw.obj"), **extra)


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    mesh = tmp_path / "props" / "p1" / "mesh" / "mesh.glb"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"glTF")
    return tmp_path.resolve()


def test_requests_are_checked_in_artist_words(root: Path) -> None:
    assert server.validate(_request(root), root) is None
    assert "own projects folder" in server.validate(_request(root), root / "elsewhere")
    assert "must be absolute" in server.validate(server.RetopoRequest(mesh_path="a.glb", out_path="b.obj"), None)
    gone = server.RetopoRequest(mesh_path=str(root / "gone.glb"), out_path=str(root / "x.obj"))
    assert "not found" in server.validate(gone, root)
    fbx = root / "m.fbx"
    fbx.write_bytes(b"x")
    fbx_request = server.RetopoRequest(mesh_path=str(fbx), out_path=str(root / "x.obj"))
    assert "cannot read '.fbx'" in server.validate(fbx_request, root)
    bad_out = _request(root)
    bad_out.out_path = str(root / "x.glb")
    assert ".obj" in server.validate(bad_out, root)
    assert "prepare" in server.validate(_request(root, prepare="sometimes"), root)
    with pytest.raises(ValueError):
        _request(root, vertices=server.MIN_VERTICES - 1)
    with pytest.raises(ValueError):
        _request(root, vertices=server.MAX_VERTICES + 1)


def test_failures_become_kinds_the_adapter_reads() -> None:
    def answer(exc: BaseException) -> tuple[int, dict[str, Any]]:
        r = server.failure_response(exc)
        return r.status_code, json.loads(r.body)

    assert answer(server.Cancelled()) == (409, {"detail": "Stopped: the clean mesh was cancelled.",
                                                "kind": "cancelled"})
    code, body = answer(server.InputProblem("Too flat."))
    assert (code, body) == (422, {"detail": "Too flat.", "kind": "input"})
    oom = type("OutOfMemoryError", (RuntimeError,), {})("CUDA out of memory. Tried to allocate 2 GiB")
    assert answer(oom)[1]["kind"] == "oom"
    assert answer(MemoryError())[1]["kind"] == "ram"
    assert answer(RuntimeError("DefaultCPUAllocator: not enough memory"))[1]["kind"] == "ram"
    missing = answer(FileNotFoundError(2, "missing", "C:/models/lato2/tvae.pt"))[1]
    assert missing["kind"] == "missing_file" and "tvae.pt" in missing["detail"]
    other = answer(ValueError("x" * 1000))[1]
    assert other["kind"] == "error" and len(other["detail"]) < 700 and "ValueError" in other["detail"]


def test_progress_runs_forward_and_stops_on_cancel() -> None:
    progress = server.Progress()
    assert progress.to_dict() == {"phase": "idle", "fraction": 0.0, "message": "Idle"}
    progress.start()
    seen = []
    for key in server.PHASES:
        for done in (0.0, 0.5, 1.0):
            progress.step(key, done, "step 1 of 2")
            seen.append(progress.fraction)
    assert seen == sorted(seen) and seen[-1] == 1.0
    assert progress.to_dict()["message"] == "Saving the clean mesh (step 1 of 2)"
    progress.step("read")                        # never goes back
    assert progress.fraction == 1.0
    progress.cancel.set()
    with pytest.raises(server.Cancelled):
        progress.step("faces")
    progress.start()                             # the next mesh starts afresh
    progress.step("read")
    progress.stop()
    assert progress.phase == "idle"


def test_the_weights_check_names_what_is_missing(tmp_path: Path) -> None:
    missing = server.missing_weights(tmp_path)
    assert "vflow.pt" in missing and "dinov2/checkpoints/dinov2_vitl14_reg4_pretrain.pth" in missing
    for name in missing:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    assert server.missing_weights(tmp_path) == []


def test_training_only_imports_are_stubbed(clean_modules: Any) -> None:
    sys.modules.pop("torch_scatter", None)
    real = importlib.util.find_spec("torch_scatter") is not None
    done = server.install_stubs()
    if real:
        assert "torch_scatter" not in done
    else:
        assert "torch_scatter" in done
        from torch_scatter import scatter_mean
        with pytest.raises(RuntimeError, match="training"):
            scatter_mean(None, None)
    if importlib.util.find_spec("o_voxel") is not None and "o_voxel (without its __init__)" in done:
        assert sys.modules["o_voxel"].__path__      # an empty package over the real folder


class FakeEngine:
    ready = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retopo(self, req: server.RetopoRequest, seed: int, progress: server.Progress) -> dict[str, Any]:
        self.calls.append((req.mesh_path, seed))
        progress.step("vertices", 0.5)
        if req.vertices == 999:
            raise server.InputProblem("LATO.2 placed no vertices on this model.")
        Path(req.out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(req.out_path).write_text("v 0 0 0\n")
        return {"obj_path": req.out_path, "vertices": 3, "faces": 1, "seed": seed, "warning": None}


def test_routes(root: Path) -> None:
    engine = FakeEngine()
    progress = server.Progress()
    app = server.build_app(engine, server.GpuWorker(), progress, root, {"attention": "sdpa", "precision": "fp32"})
    client = TestClient(app)
    assert client.get("/spellcast/health").json() == {"ready": True, "attention": "sdpa", "precision": "fp32"}
    assert client.get("/spellcast/progress").json()["phase"] == "idle"
    body = _request(root, seed=7).model_dump()
    r = client.post("/spellcast/retopo", json=body)
    assert r.status_code == 200 and r.json()["seed"] == 7 and isinstance(r.json()["seconds"], float)
    assert Path(body["out_path"]).is_file() and progress.phase == "idle"
    r = client.post("/spellcast/retopo", json={**body, "seed": -1})         # a random seed, reported back
    assert r.status_code == 200 and r.json()["seed"] >= 0
    r = client.post("/spellcast/retopo", json={**body, "vertices": 999})
    assert r.status_code == 422 and r.json()["kind"] == "input"
    r = client.post("/spellcast/retopo", json={**body, "vertices": 50})
    assert r.status_code == 422 and r.json()["kind"] == "input" and "vertices" in r.json()["detail"]
    r = client.post("/spellcast/retopo", json={**body, "mesh_path": str(root.parent / "x.glb")})
    assert r.status_code == 400 and "own projects folder" in r.json()["detail"]
    assert client.post("/spellcast/cancel", json={}).json() == {"cancelling": False}    # nothing running
    assert client.post("/spellcast/cancel").status_code == 422      # needs a JSON body (no CORS-free posts)
    engine.ready = False
    assert client.post("/spellcast/retopo", json=body).status_code == 503
    assert client.get("/spellcast/health").json()["ready"] is False


def test_a_second_mesh_is_refused_while_one_runs(root: Path) -> None:
    """The hub never sends two at once: a second one means it gave up on the first (a timeout), which still runs.
    It is refused ("busy", the hub then restarts LATO.2), never queued behind the first on the one GPU thread."""
    import threading

    started, release = threading.Event(), threading.Event()

    class Slow(FakeEngine):
        def retopo(self, req: server.RetopoRequest, seed: int, progress: server.Progress) -> dict[str, Any]:
            started.set()
            release.wait(10)
            return super().retopo(req, seed, progress)

    client = TestClient(server.build_app(Slow(), server.GpuWorker(), server.Progress(), root, {}))
    body = _request(root).model_dump()
    first = threading.Thread(target=lambda: client.post("/spellcast/retopo", json=body))
    first.start()
    try:
        assert started.wait(10)
        r = client.post("/spellcast/retopo", json=body)
        assert r.status_code == 409 and r.json()["kind"] == "busy"
    finally:
        release.set()
        first.join(10)
    assert client.post("/spellcast/retopo", json=body).status_code == 200      # free again


def test_broken_cuda_answers_restart_and_ends_the_process(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After an error that leaves CUDA unusable LATO.2 is no longer ready, says "restart" (the hub starts it
    afresh for the next prop) and exits; an input problem or a cancel never asks the GPU."""
    exits: list[bool] = []
    probes: list[bool] = []
    monkeypatch.setattr(server, "cuda_works", lambda: probes.append(True) or False)

    class Broken(FakeEngine):
        def retopo(self, req: server.RetopoRequest, seed: int, progress: server.Progress) -> dict[str, Any]:
            if req.vertices == 999:
                raise server.InputProblem("Too flat.")
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    engine = Broken()
    client = TestClient(server.build_app(engine, server.GpuWorker(), server.Progress(), root, {},
                                         exit_later=lambda: exits.append(True)))
    body = _request(root).model_dump()
    assert client.post("/spellcast/retopo", json={**body, "vertices": 999}).json()["kind"] == "input"
    assert probes == [] and exits == []
    r = client.post("/spellcast/retopo", json=body)
    assert r.status_code == 500 and r.json()["kind"] == "restart" and "starts again" in r.json()["detail"]
    assert exits == [True] and engine.ready is False
    assert client.get("/spellcast/health").json()["ready"] is False
    again = client.post("/spellcast/retopo", json=body)                    # until it is gone
    assert again.status_code == 503 and again.json()["kind"] == "restart"


def test_running_out_of_memory_for_the_probe_is_no_broken_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    class OutOfMemoryError(RuntimeError):
        pass

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize() -> None:
            pass

    def ones(*a: Any, **k: Any) -> Any:
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB")

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=Cuda, ones=ones))
    assert server.cuda_works() is True

    def broken(*a: Any, **k: Any) -> Any:
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=Cuda, ones=broken))
    assert server.cuda_works() is False


def test_a_stage_that_runs_out_of_memory_on_its_way_to_the_gpu_goes_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Low VRAM: when moving a stage's models to the GPU fails half-way, those moved already go back to the CPU
    (the next prop must not start with them stranded on the card)."""
    moves: list[tuple[str, str]] = []

    class Model:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name, self.fail = name, fail

        def to(self, device: Any) -> "Model":
            if str(device) == "cuda" and self.fail:
                raise RuntimeError("CUDA out of memory")
            moves.append((self.name, str(device)))
            return self

    cuda = types.SimpleNamespace(empty_cache=lambda: moves.append(("cache", "emptied")))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    engine = server.Lato2(Path("."), Path("."), "sdpa", "fp16", low_vram=True)
    engine.device = "cuda"
    stage = (Model("tflow"), Model("tvae", fail=True))
    with pytest.raises(RuntimeError, match="out of memory"):
        with engine._on_gpu(stage):
            pytest.fail("the stage must not run")
    assert moves == [("tflow", "cuda"), ("tflow", "cpu"), ("tvae", "cpu"), ("cache", "emptied")]


# ---- adapter ------------------------------------------------------------------------------------------------------

def test_adapter_settings() -> None:
    p = adapter_module.lato2_params({})
    assert p == {"seed": 0, "uv_leniency": 1.0}
    clamped = adapter_module.lato2_params({"lato2_seed": "5", "uv_leniency": 7})
    assert clamped == {"seed": 5, "uv_leniency": 1.0}
    assert adapter_module.lato2_params({"uv_leniency": -2})["uv_leniency"] == 0.0
    with pytest.raises(StageError, match="not valid numbers"):
        adapter_module.lato2_params({"lato2_seed": "x"})
    adapter = adapter_module.Lato2Adapter()
    adapter.check_settings("retopo", {"sizes": [99]})
    for bad in ({"sizes": []}, {}, {"sizes": [2], "lato2_seed": -1}):
        with pytest.raises(StageError):
            adapter.check_settings("retopo", bad)


def test_adapter_identity_and_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = adapter_module.Lato2Adapter()
    assert (adapter.tool_id, adapter.stages, adapter.uses_gpu) == (paths.LATO2, ("retopo",), True)
    assert adapter.license["name"] == "MIT" and adapter.license["noncommercial"] is False
    assert "fbb1f5a5755e6db8700cf6922fd506830b7cdccd" in adapter.license["url"]
    spec = adapter.launch_spec(7774, {"lato2": {"low_vram": "on"}})
    argv = spec.argv
    assert argv[0] == str(paths.venv_python(paths.LATO2)) and argv[1] == str(paths.LAUNCHERS / "lato2_server.py")
    assert argv[argv.index("--repo") + 1] == str(paths.LATO2_DIR)
    assert argv[argv.index("--models") + 1] == str(paths.MODELS_LATO2)
    assert argv[argv.index("--port") + 1] == "7774" and argv[argv.index("--low-vram") + 1] == "on"
    assert argv[argv.index("--projects-root") + 1] == str(paths.PROJECTS)
    assert spec.cwd == paths.LATO2_DIR and spec.ready_timeout >= 300
    assert adapter.launch_spec(7774, {"lato2": {"low_vram": "sometimes"}}).argv[-3] == "auto"
    # launch flags the launcher knows
    parsed = server.parse_args(argv[2:])
    assert parsed.port == 7774 and Path(parsed.projects_root) == paths.PROJECTS
    calls = []
    monkeypatch.setattr(gpu, "lato2_supported", lambda info: calls.append(info) or (False, "too old"))
    monkeypatch.setattr(gpu, "primary_gpu", lambda: "the card")
    assert adapter.supported() == (False, "too old") and calls == ["the card"]


def _run(adapter: adapter_module.Lato2Adapter, prop_dir: Path, handler: Any, prop: dict[str, Any] | None = None,
         cancelled: Any = lambda: False, unwrap: dict[str, Any] | None = None,
         sizes: tuple[int, ...] = (2,)) -> tuple[Any, list[Any], list[Any]]:
    reports: list[Any] = []
    unwrapped: list[Any] = []

    async def unwrapper(src: Path, dst: Path, leniency: float) -> dict[str, Any]:
        unwrapped.append((src, dst, leniency))
        dst.write_text("v 0 0 0\nvt 0 0\n")
        return unwrap or {"vertices": 2100, "faces": 4000, "has_uv": True, "warning": None, "seconds": 3.0}

    async def main() -> Any:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            ctx = StageContext(project="p", prop_id="p1", prop_dir=prop_dir, base_url="http://127.0.0.1:7774",
                               client=client, settings={}, report=lambda f, m: reports.append((f, m)),
                               cancelled=cancelled, stage="retopo", unwrapper=unwrapper)
            return await adapter.run_stage(ctx, prop if prop is not None else {"mesh": {"glb": "mesh/mesh.glb"}},
                                           {"sizes": list(sizes), "lato2_seed": 3, "uv_leniency": 0.25})
    return asyncio.run(main()), reports, unwrapped


def test_adapter_makes_the_raw_mesh_then_unwraps_it(tmp_path: Path) -> None:
    prop_dir = tmp_path / "p1"
    (prop_dir / "mesh").mkdir(parents=True)
    (prop_dir / "mesh" / "mesh.glb").write_bytes(b"glTF")
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/spellcast/progress":
            return httpx.Response(200, json={"phase": "working", "fraction": 0.5, "message": "Placing the vertices"})
        body = json.loads(request.content)
        sent.append(body)
        Path(body["out_path"]).write_text("v 0 0 0\n")
        return httpx.Response(200, json={"obj_path": body["out_path"], "vertices": 2100, "faces": 4100, "seed": 3,
                                         "seconds": 40.0, "warning": "LATO.2 placed 900 vertices.",
                                         "boundary_edges": 0.0, "non_manifold_edges": 0.0, "outward": 0.99,
                                         "prepared": "solidified"})

    patch, reports, unwrapped = _run(adapter_module.Lato2Adapter(), prop_dir, handler, sizes=(2, 22))
    raw = {size: prop_dir / "retopo" / f"{size}k" / "lato2.raw.obj" for size in (2, 22)}
    assert sent == [{"mesh_path": str((prop_dir / "mesh" / "mesh.glb").resolve()), "out_path": str(raw[size].resolve()),
                     "vertices": vertices, "seed": 3} for size, vertices in ((2, 2000), (22, 10_000))], \
        "22k is more than LATO.2 places: it is asked for its most"
    assert unwrapped == [(raw[size], prop_dir / "retopo" / f"{size}k" / "lato2.obj", 0.25) for size in (2, 22)]
    stats = {"vertices": 2100, "faces": 4000, "has_uv": True, "seconds": 43.0, "warning": "LATO.2 placed 900 vertices.",
             "seed": 3, "non_manifold_edges": 0.0, "boundary_edges": 0.0, "outward": 0.99, "prepared": "solidified"}
    assert patch == {"method": "lato2", "source": "mesh/mesh.glb", "saved_from": "auto", "variants": [
        {"size": 2, "obj": "retopo/2k/lato2.obj", "stats": {**stats, "target_vertices": 2000}},
        {"size": 22, "obj": "retopo/22k/lato2.obj", "stats": {**stats, "target_vertices": 10_000}}]}
    assert not any(path.exists() for path in raw.values())      # only the unwrapped copies are kept
    assert reports[-1] == (1.0, "Clean meshes saved.") and all(f <= 1.0 for f, _ in reports)
    assert reports[:2] == [(0.0, "2k vertices (1 of 2) · Sending the 3D model to LATO.2…"),
                           (0.45, "22k vertices (2 of 2) · Sending the 3D model to LATO.2…")]
    # an unwrap that failed keeps has_uv False and says why
    patch, _, _ = _run(adapter_module.Lato2Adapter(), prop_dir, handler,
                       unwrap={"vertices": 2100, "faces": 4100, "has_uv": False, "warning": "No UVs.", "seconds": 1})
    made = patch["variants"][0]["stats"]
    assert made["has_uv"] is False and made["warning"] == "LATO.2 placed 900 vertices. No UVs."


def test_adapter_errors(tmp_path: Path) -> None:
    prop_dir = tmp_path / "p1"
    adapter = adapter_module.Lato2Adapter()
    with pytest.raises(StageError, match="3D stage first"):
        _run(adapter, prop_dir, lambda r: httpx.Response(500), prop={})
    with pytest.raises(StageError, match="is missing"):
        _run(adapter, prop_dir, lambda r: httpx.Response(500))
    (prop_dir / "mesh").mkdir(parents=True)
    (prop_dir / "mesh" / "mesh.glb").write_bytes(b"glTF")
    for answer, words in (({"detail": "x", "kind": "oom"}, "GPU memory"), ({"detail": "Too flat.", "kind": "input"},
                                                                            "Too flat."),
                          ({"detail": "Stopped", "kind": "cancelled"}, "Cancelled"), ({}, "HTTP 500")):
        with pytest.raises(StageError, match=words):
            _run(adapter, prop_dir, lambda r, a=answer: httpx.Response(500 if a.get("kind") != "cancelled" else 409,
                                                                        json=a))
    with pytest.raises(StageError, match="no mesh file"):     # success without the file
        _run(adapter, prop_dir, lambda r: httpx.Response(200, json={"obj_path": "x"}))
    # CUDA broke, or a mesh the hub gave up on still runs: the job runner starts LATO.2 afresh for the next prop
    for answer in ({"detail": "LATO.2 stopped after an error. It starts again for the next prop.", "kind": "restart"},
                   {"detail": "LATO.2 is still working on an earlier mesh.", "kind": "busy"}):
        with pytest.raises(ToolNeedsRestart, match=answer["detail"]):
            _run(adapter, prop_dir, lambda r, a=answer: httpx.Response(500 if a["kind"] == "restart" else 409, json=a))
    asked: list[str] = []

    def slow(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        if request.url.path == "/spellcast/retopo":
            raise httpx.ReadTimeout("no answer")
        return httpx.Response(200, json={"cancelling": True})

    with pytest.raises(ToolNeedsRestart, match="did not finish this mesh within 45 minutes"):
        _run(adapter, prop_dir, slow)
    assert "/spellcast/cancel" in asked                     # and it was told to stop what it is doing

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")
    with pytest.raises(StageError, match="not reachable"):
        _run(adapter, prop_dir, down)


def test_a_cancelled_job_asks_lato2_to_stop(tmp_path: Path) -> None:
    prop_dir = tmp_path / "p1"
    (prop_dir / "mesh").mkdir(parents=True)
    (prop_dir / "mesh" / "mesh.glb").write_bytes(b"glTF")
    paths_seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        paths_seen.append(request.url.path)
        if request.url.path == "/spellcast/retopo":
            for _ in range(50):
                if "/spellcast/cancel" in paths_seen:
                    return httpx.Response(409, json={"detail": "Stopped", "kind": "cancelled"})
                await asyncio.sleep(0.1)
            return httpx.Response(500, json={})
        return httpx.Response(200, json={"cancelling": True, "phase": "working", "fraction": 0.2, "message": "x"})

    with pytest.raises(StageError, match="Cancelled"):
        _run(adapter_module.Lato2Adapter(), prop_dir, handler, cancelled=lambda: True)
    assert paths_seen.count("/spellcast/cancel") == 1


# ---- numpy: faces, holes, the solid, the OBJ ---------------------------------------------------------------------

@needs_numpy
def test_faces_are_recovered_from_an_exact_graph() -> None:
    import numpy as np
    import trimesh

    import lato2_faces as faces

    for mesh in (trimesh.creation.icosphere(subdivisions=3), trimesh.creation.torus(1.0, 0.35, 48, 24)):
        v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        n = len(v)
        edges = np.unique(np.sort(mesh.edges, axis=1), axis=0)
        logits = np.full((n, n), -20.0, dtype=np.float32)
        logits[edges[:, 0], edges[:, 1]] = logits[edges[:, 1], edges[:, 0]] = 20.0
        got = faces.build_faces(v, logits, mesh.vertex_normals)
        truth = {faces._canonical(*map(int, t)) for t in f}
        assert {faces._canonical(*map(int, t)) for t in got} == truth         # every face, wound as the mesh
        report = faces.edge_report(got)
        assert report["consistent"] and report["boundary"] == 0.0 and report["non_manifold"] == 0.0
        sparse = faces.EdgeLogits(n, edges[:, 0], edges[:, 1], np.full(len(edges), 20.0))
        assert {faces._canonical(*map(int, t)) for t in faces.build_faces(v, sparse, mesh.vertex_normals)} == truth


@needs_numpy
def test_edge_logits_answer_pairs_and_top_partners() -> None:
    import numpy as np

    import lato2_faces as faces

    store = faces.EdgeLogits(4, [0, 0, 1], [1, 2, 3], [5.0, -1.0, 2.0])
    assert store[0, 1] == 5.0 and store[1, 0] == 5.0 and store[2, 3] == -30.0 and len(store) == 3
    assert list(store[np.array([0, 3]), np.array([2, 1])]) == [-1.0, 2.0]
    rows, cols = store.candidate_pairs(1, -2.0)
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(0, 1), (1, 0), (2, 0), (3, 1)]
    assert faces.EdgeLogits(3, [], [], [])[0, 1] == -30.0


@needs_numpy
def test_holes_are_closed_and_crumbs_dropped() -> None:
    import numpy as np
    import trimesh

    import lato2_faces as faces

    sphere = trimesh.creation.icosphere(subdivisions=3)
    v, f = np.asarray(sphere.vertices), np.asarray(sphere.faces)
    around = np.where((f == 0).any(axis=1))[0]                   # the 5 faces around vertex 0: a 5-edge hole
    far = np.argsort(v[:, 0])[-1]
    around_far = np.where((f == far).any(axis=1))[0][:1]         # one more single-triangle hole
    holed = np.delete(f, np.r_[around, around_far], axis=0)
    crumb = np.array([[len(v), len(v) + 1, len(v) + 2]])
    v2 = np.concatenate([v, v[:3] * 2.0])
    kept, dropped = faces.drop_islands(np.concatenate([holed, crumb]), 16)
    assert dropped == 1 and len(kept) == len(holed)
    assert faces.edge_report(kept)["boundary"] > 0
    pts, closed_f, closed = faces.close_holes(v2, kept, 48)
    report = faces.edge_report(closed_f)
    assert closed == 2 and report["boundary"] == 0.0 and report["consistent"] and report["non_manifold"] == 0.0
    closed_mesh = trimesh.Trimesh(pts, closed_f, process=False)
    assert closed_mesh.volume > 0
    # a loop whose corners are already joined the same way elsewhere gets a fan round a new centre vertex
    ring = faces._simple_rings([1, 2, 3, 4, 2, 5])
    assert ring == [[2, 3, 4], [1, 2, 5]]


@needs_numpy
def test_the_solid_stays_where_the_surface_is_and_refuses_to_lose_a_blade() -> None:
    import numpy as np
    import trimesh

    import lato2_prepare as prepare

    sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    v, f, lost = prepare.solidify(sphere.vertices, sphere.faces, samples=1_000_000)
    pitch = 2.0 / (prepare.SOLID_RESOLUTION - 1)
    radius = np.linalg.norm(v, axis=1)
    assert lost == 0.0 and abs(radius.mean() - 1.0) < 0.25 * pitch      # not inflated, not shifted
    assert trimesh.Trimesh(v, f, process=False).volume > 0             # wound outward
    blade = trimesh.creation.box((0.06, 1.0, 0.008))
    grip = trimesh.creation.box((0.3, 0.05, 0.05))
    grip.apply_translation((0, -0.5, 0))
    sword = trimesh.util.concatenate([blade, grip])
    assert prepare.solidify(sword.vertices, sword.faces, samples=500_000)[2] > prepare.LOST_LIMIT
    assert prepare.crowding(sword.vertices, sword.faces, 2000) > prepare.crowding(v, f, 2000)


@needs_numpy
def test_a_missing_part_is_noticed() -> None:
    import numpy as np
    import trimesh

    import lato2_faces as faces

    sphere = trimesh.creation.icosphere(subdivisions=4)
    surface = faces.SurfaceSamples(sphere.vertices, sphere.faces, count=50_000)
    assert not surface.inside_out() and abs(surface.area - sphere.area) < 1e-9
    assert surface.missing_share(sphere.vertices, sphere.faces, 0.05) < 0.001
    upper = sphere.faces[sphere.triangles_center[:, 1] > 0]           # the top half only
    assert 0.4 < surface.missing_share(sphere.vertices, upper, 0.2) < 0.6
    assert surface.outward_share(sphere.vertices, sphere.faces) == 1.0
    assert surface.outward_share(sphere.vertices, sphere.faces[:, ::-1]) == 0.0
    normals = surface.normals_at(sphere.vertices)
    assert np.allclose((normals * sphere.vertices).sum(1), 1.0, atol=0.02)   # outward, radial


@needs_numpy
def test_the_crease_closing_policy() -> None:
    import trimesh

    import lato2_prepare as prepare

    sphere = trimesh.creation.icosphere(subdivisions=3)
    v, f = sphere.vertices, sphere.faces
    assert prepare.prepare(v, f, 2000, "off")[2:] == ("as is", None)
    out = prepare.prepare(v, f, 2000, "auto")
    assert out[2] == "as is" and out[3] < prepare.CROWDING_LIMIT and out[0] is v     # a clean shape keeps its detail
    assert prepare.prepare(v, f, 2000, "always")[2] == "solidified"
    blade = trimesh.creation.box((0.06, 1.0, 0.008))       # 1.5 voxels thin: the opening would delete it
    grip = trimesh.creation.box((0.3, 0.05, 0.05))
    grip.apply_translation((0, -0.5, 0))
    sword = trimesh.util.concatenate([blade, grip])
    said: list[str] = []
    sv, sf, how, crowd = prepare.prepare(sword.vertices, sword.faces, 2000, "auto", log=said.append)
    assert crowd >= prepare.CROWDING_LIMIT and how == "solidified, thin parts kept" and "would lose" in said[0]
    assert (sv.max(0) - sv.min(0))[1] > 0.95 * 1.05                   # the blade is still there


@needs_numpy
def test_quantize_positions_and_obj(tmp_path: Path) -> None:
    import numpy as np
    import trimesh

    v = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1e-6, 0.0, 0.0]])
    f = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [4, 1, 2]])
    center, extent = server.frame_of(v)
    assert np.allclose(center, [1.0, 0.5, 0.5]) and extent == 2.0
    v_int, offsets, faces = server.quantize(v, f, 1024)
    assert len(v_int) == 4 and len(faces) == 4                          # vertices 0 and 4 share a voxel
    assert np.all(np.abs(offsets) <= 1.0)
    # a voxel's centre plus its offset maps back onto the cluster mean, in the input's frame
    back = server.decode_positions(v_int, offsets, 1024) * extent + center
    assert np.abs(np.sort(back, axis=0) - np.sort(v[:4], axis=0)).max() < 1e-3
    out = tmp_path / "x.obj"
    server.write_obj(out, back, faces)
    read = trimesh.load(str(out), process=False, force="mesh")
    assert np.allclose(read.vertices, back, atol=1e-5) and (np.asarray(read.faces) == faces).all()
    assert not list(tmp_path.glob("*.tmp*"))
    pts, tris, unused = server.drop_unused(np.arange(15.0).reshape(5, 3), np.array([[0, 2, 4]]))
    assert unused == 2 and tris.tolist() == [[0, 1, 2]] and pts[1].tolist() == [6.0, 7.0, 8.0]


# ---- torch: the SDPA stand-in and the rasterizer -------------------------------------------------------------------

def _reference(q: Any, k: Any, v: Any, causal: bool = False, window: tuple[int, int] = (-1, -1)) -> Any:
    """Plain attention in float64, flash_attn's layout and mask rules (bottom-right aligned)."""
    import torch

    q, k, v = (t.double().transpose(0, 1) for t in (q, k, v))           # (heads, len, d)
    if k.shape[0] != q.shape[0]:
        k = k.repeat_interleave(q.shape[0] // k.shape[0], dim=0)
        v = v.repeat_interleave(q.shape[0] // v.shape[0], dim=0)
    scores = q @ k.transpose(1, 2) / q.shape[-1] ** 0.5
    lq, lk = q.shape[1], k.shape[1]
    pos = torch.arange(lq)[:, None] + lk - lq
    keys = torch.arange(lk)[None, :]
    left, right = window
    if causal:
        right = 0
    keep = torch.ones(lq, lk, dtype=torch.bool)
    if left >= 0:
        keep &= keys >= pos - left
    if right >= 0:
        keep &= keys <= pos + right
    scores = scores.masked_fill(~keep, float("-inf"))
    out = torch.softmax(scores, -1).nan_to_num(0.0) @ v
    return out.transpose(0, 1)


@needs_torch
@pytest.mark.parametrize("budget", [attn_shim.SCORE_BUDGET, 64])        # 64: every call sliced to the bone
def test_the_shim_matches_plain_attention(monkeypatch: pytest.MonkeyPatch, budget: int) -> None:
    import torch

    monkeypatch.setattr(attn_shim, "SCORE_BUDGET", budget)
    torch.manual_seed(0)
    lens_q, lens_k = [5, 5, 9, 1, 5, 12], [7, 7, 9, 3, 7, 4]
    cu_q = torch.tensor([0, *torch.tensor(lens_q).cumsum(0).tolist()], dtype=torch.int32)
    cu_k = torch.tensor([0, *torch.tensor(lens_k).cumsum(0).tolist()], dtype=torch.int32)
    q = torch.randn(sum(lens_q), 4, 8)
    k = torch.randn(sum(lens_k), 2, 8)       # grouped-query: 2 key/value heads for 4 query heads
    v = torch.randn(sum(lens_k), 2, 8)
    for causal, window in ((False, (-1, -1)), (True, (-1, -1)), (False, (2, 1))):
        got = attn_shim.flash_attn_varlen_func(q, k, v, cu_q, cu_k, max(lens_q), max(lens_k), causal=causal,
                                               window_size=window)
        want = torch.cat([_reference(q[a:b], k[c:d], v[c:d], causal, window) for a, b, c, d in
                          zip(cu_q[:-1].tolist(), cu_q[1:].tolist(), cu_k[:-1].tolist(), cu_k[1:].tolist())])
        assert got.dtype == q.dtype and got.shape == (sum(lens_q), 4, 8)
        assert torch.allclose(got.double(), want, atol=1e-5), (causal, window)
    qkv = torch.randn(sum(lens_q), 3, 4, 8)
    got = attn_shim.flash_attn_varlen_qkvpacked_func(qkv, cu_q, max(lens_q))
    want = torch.cat([_reference(qkv[a:b, 0], qkv[a:b, 1], qkv[a:b, 2])
                      for a, b in zip(cu_q[:-1].tolist(), cu_q[1:].tolist())])
    assert torch.allclose(got.double(), want, atol=1e-5)
    batch = torch.randn(3, 10, 3, 4, 8)
    got = attn_shim.flash_attn_qkvpacked_func(batch, softmax_scale=0.5)
    scaled = [_reference(batch[i, :, 0] * (0.5 * 8 ** 0.5), batch[i, :, 1], batch[i, :, 2]) for i in range(3)]
    assert torch.allclose(got.double(), torch.stack(scaled), atol=1e-5)
    got = attn_shim.flash_attn_kvpacked_func(batch[:, :, 0], batch[:, :, 1:])
    assert torch.allclose(got.double(), torch.stack([_reference(*batch[i].unbind(1)) for i in range(3)]), atol=1e-5)


@needs_torch
def test_the_shim_computes_in_its_own_dtype_under_autocast(monkeypatch: pytest.MonkeyPatch) -> None:
    """TriFlow samples under float16 autocast; the shim must still compute in the dtype it chose (float32 on a GTX
    10xx), as flash_attn's kernels ignore autocast: autocast cast SDPA's inputs back to float16 (10x slower)."""
    import torch
    import torch.nn.functional as F

    seen = []
    real = F.scaled_dot_product_attention

    def spy(q, k, v, **kwargs):
        seen.append((q.dtype, torch.is_autocast_enabled(q.device.type)))
        return real(q, k, v, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    torch.manual_seed(0)
    q, k, v = (torch.randn(12, 4, 8) for _ in range(3))
    cu = torch.tensor([0, 5, 12], dtype=torch.int32)
    want = attn_shim.flash_attn_varlen_func(q, k, v, cu, cu, 7, 7)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got = attn_shim.flash_attn_varlen_func(q, k, v, cu, cu, 7, 7)
    assert seen and all(dtype == torch.float32 and not autocast for dtype, autocast in seen), seen
    assert got.dtype == torch.float32 and torch.equal(got, want)


@needs_torch
def test_the_rasterizer_draws_the_nearest_triangle() -> None:
    import torch

    import lato2_render as render

    # two triangles covering the left and right halves at different depths, plus one hidden behind the left
    clip = torch.tensor([[-1, -1, 0.5, 1], [1, -1, 0.5, 1], [-1, 1, 0.5, 1],       # 0: lower-left half, z 0.5
                         [1, -1, 0.2, 1], [1, 1, 0.2, 1], [-1, 1, 0.2, 1],          # 1: upper-right half, z 0.2
                         [-1, -1, 0.9, 1], [1, -1, 0.9, 1], [-1, 1, 0.9, 1]],       # 2: under 0, farther
                        dtype=torch.float32)
    tris = torch.tensor([[0, 1, 2], [3, 5, 4], [6, 7, 8]])                         # triangle 1 wound clockwise
    tri_id, bary = render.rasterize(clip, tris, 16)
    assert set(tri_id.unique().tolist()) == {0, 1}          # 2 is hidden; both windings drawn
    assert tri_id[15, 0] == 0 and tri_id[0, 15] == 1        # rows run top to bottom: y = -1 is the last row
    assert torch.allclose(bary.sum(-1), torch.ones(16, 16))
