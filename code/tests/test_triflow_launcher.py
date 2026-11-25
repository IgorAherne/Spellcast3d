"""TriFlow: the launcher's own logic (launchers/triflow_server.py), its kaolin stand-in, the adapter
(adapters/triflow.py), and that TriFlow's checkout stays byte-identical to its commit.

Runs in the hub venv (fastapi + pydantic + httpx, no torch / numpy / trimesh / safetensors): the launcher keeps those
imports inside functions, so what needs them (reading a mesh, TriFlow itself) runs only in the triflow venv on the GPU.

The byte-identity check runs on code/triflow when its submodules are checked out (skipped otherwise), and on any other
TriFlow checkout named by SPELLCAST_TRIFLOW_REPO (a scratch copy the launcher ran from, for instance).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from spellcast import gpu, paths, projects
from spellcast.adapters import base, build_adapters
from spellcast.adapters import triflow as adapter_module
from spellcast.adapters.triflow import TriflowAdapter

if str(paths.LAUNCHERS) not in sys.path:     # the launchers import their siblings as top-level modules
    sys.path.insert(0, str(paths.LAUNCHERS))

import triflow_server as server  # noqa: E402

LICENSE_URL = "https://github.com/IgorAherne/triflow/blob/5cf7f00e4f5d5745a1cafbb21ff5d7d28bb4f878/LICENSE"
TRIFLOW_COMMIT = "5cf7f00e4f5d5745a1cafbb21ff5d7d28bb4f878"
# TriFlow's nested checkouts and their pinned commits (TriFlow's own gitlinks; FlexiCubes is TRELLIS's).
NESTED = {
    "third_party/TRELLIS": "f17fdf12d8f17a6a09225f01756d141285dc848f",
    "third_party/TRELLIS/trellis/representations/mesh/flexicubes": "815e075a2a400d06c48d94c347674344ed6ae5c5",
    "third_party/Direct3D-S2": "a1cf235b2881cff04a91900060a9546b40e7ee5d",
    "third_party/pyfqmr-triflow": "c93a7671ba4a2163dd3c7ea6ab6b52ad0167cc72",
}
# The only files that may sit in TriFlow's folder besides its own: the weights, hard-linked into checkpoints/
# (ignored by TriFlow's own .gitignore).
ALLOWED_EXTRA = {f"checkpoints/{name}" for name in server.WEIGHTS}


# ---- the grid frame -----------------------------------------------------------------------------------------

def to_grid(v: tuple[float, ...], centre: tuple[float, ...], scale: float) -> tuple[float, ...]:
    """What TriFlow's discretize_mesh does to a vertex (the forward direction the launcher undoes)."""
    return tuple((a - c) * scale + server.GRID / 2 for a, c in zip(v, centre))


def test_the_grid_frame_undoes_triflows_normalisation() -> None:
    lo, hi = (-0.36, -0.43, -0.38), (0.39, 0.49, 0.34)
    centre, scale = server.grid_frame(lo, hi)
    grid_lo, grid_hi = to_grid(lo, centre, scale), to_grid(hi, centre, scale)
    # the longest side (y, 0.92) spans the grid minus the padding on both sides, centred
    assert grid_lo[1] == pytest.approx(server.PAD) and grid_hi[1] == pytest.approx(server.GRID - server.PAD)
    assert all(pytest.approx(server.GRID / 2) == (a + b) / 2 for a, b in zip(grid_lo, grid_hi))
    for v in (lo, hi, (0.1, -0.2, 0.3)):
        back = tuple((g - server.GRID / 2) / scale + c for g, c in zip(to_grid(v, centre, scale), centre))
        assert back == pytest.approx(v, abs=1e-12)


def test_a_flat_mesh_gets_a_frame_too() -> None:
    centre, scale = server.grid_frame((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))
    assert centre == (1.0, 2.0, 3.0) and scale == 1.0


# ---- attention and the older GPUs ---------------------------------------------------------------------------

@pytest.mark.parametrize(("asked", "capability", "expected"), [
    ("auto", (8, 6), ("flash_attn", False, False)),
    (None, (12, 0), ("flash_attn", False, False)),
    ("flash_attn", (8, 0), ("flash_attn", False, False)),
    ("sdpa", (8, 6), ("flash_attn", True, False)),            # the shim's module is named flash_attn
    ("auto", (7, 5), ("flash_attn", True, False)),            # the real one has no kernels there: the shim
    ("flash_attn", (7, 5), ("flash_attn", True, False)),
    ("auto", (6, 1), ("flash_attn", True, True)),             # and torchsparse's gather-scatter below sm_75
    ("nonsense", (7, 0), ("flash_attn", True, True)),
])
def test_attention_follows_the_gpu(asked: str | None, capability: tuple[int, int],
                                   expected: tuple[str, bool, bool]) -> None:
    assert server.choose_attention(asked, capability) == expected
    assert server.choose_attention(asked, capability, xformers_installed=True) == expected


def test_xformers_only_when_asked_for_and_installed() -> None:
    assert server.choose_attention("xformers", (8, 9), xformers_installed=True) == ("xformers", False, False)
    assert server.choose_attention("xformers", (7, 5), xformers_installed=True) == ("xformers", False, False)
    assert server.choose_attention("xformers", (8, 6), xformers_installed=False) == ("flash_attn", False, False)
    assert server.choose_attention("xformers", (7, 5), xformers_installed=False) == ("flash_attn", True, False)
    assert server.choose_attention("xformers", (6, 1), xformers_installed=True) == ("flash_attn", True, True)


@pytest.mark.parametrize(("cc", "expected"), [(None, None), (12.0, "flash_attn"), (8.0, "flash_attn"),
                                              (7.5, "sdpa"), (7.0, "sdpa"), (6.1, "sdpa")])
def test_the_adapter_asks_for_what_the_launcher_would_pick(cc: float | None, expected: str | None) -> None:
    assert adapter_module.attn_backend(cc) == expected
    if cc is not None:
        backend, shim, _ = server.choose_attention(expected, (int(cc), round(cc % 1 * 10)))
        assert (backend, shim) == ({"flash_attn": ("flash_attn", False), "sdpa": ("flash_attn", True)}[expected])
        assert server.choose_attention("auto", (int(cc), round(cc % 1 * 10)))[:2] == (backend, shim)


# ---- weights: read once per process -------------------------------------------------------------------------

def test_weights_are_read_from_disk_once_and_handed_out_as_new_dicts(tmp_path: Path) -> None:
    weights = tmp_path / "flow_model.safetensors"
    weights.write_bytes(b"x" * 10)
    reads: list[tuple[str, str]] = []

    def load(filename: Any, device: str = "cpu") -> dict[str, Any]:
        reads.append((str(filename), device))
        return {"a": object(), "b": object()}

    lines: list[str] = []
    cache = server.WeightCache(load, lines.append)
    first = cache(weights, device="cpu")
    first.pop("a")                                       # the caller may change its dict
    second = cache(Path(str(weights)), device="cpu")
    assert len(reads) == 1 and cache.reads == 1 and set(second) == {"a", "b"}
    assert second["b"] is first["b"]                     # the same tensors, never copied
    assert len(lines) == 1 and "flow_model.safetensors" in lines[0]
    os.chdir(tmp_path)
    try:
        cache("flow_model.safetensors", device="cpu")    # TriFlow's relative checkpoints/ path: same file
    finally:
        os.chdir(paths.ROOT)
    assert cache.reads == 1
    weights.write_bytes(b"y" * 11)                       # replaced (a repair): read again
    cache(weights, device="cpu")
    assert cache.reads == 2


def test_the_weight_cache_replaces_safetensors_load_file_once(monkeypatch: pytest.MonkeyPatch) -> None:
    package = types.ModuleType("safetensors")
    torch_module = types.ModuleType("safetensors.torch")

    def load_file(filename: Any, device: str = "cpu") -> dict[str, Any]:
        return {}

    torch_module.load_file = load_file
    package.torch = torch_module
    monkeypatch.setitem(sys.modules, "safetensors", package)
    monkeypatch.setitem(sys.modules, "safetensors.torch", torch_module)
    cache = server.install_weight_cache(lambda line: None)
    assert torch_module.load_file is cache and isinstance(cache, server.WeightCache)
    assert server.install_weight_cache(lambda line: None) is cache


# ---- progress from what TriFlow prints ---------------------------------------------------------------------

def test_progress_follows_triflows_output() -> None:
    progress = server.Progress()
    progress.feed("(0/1) Processing C:\\x\\p_1.obj...")          # idle: ignored
    assert progress.to_dict() == {"phase": "idle", "fraction": 0.0, "message": "Idle"}
    progress.start()
    progress.step("setup")
    tee = server.ProgressTee(_Sink(), progress)
    tee.write("(0/1) Processing C:\\x\\p_1.obj...\n")
    assert progress.message.startswith("Measuring the shape")
    tee.write("Sampling:   0%|          | 0/50 [00:00<?, ?it/s]\r"
              "Sampling:  50%|#####     | 25/50 [00:09<00:07,  3.41it/s]")
    tee.write("\r")
    label, start, end = server.PHASES["sampling"]
    assert progress.fraction == pytest.approx(start + (end - start) * 0.5, abs=1e-3)
    assert progress.message == f"{label} (25/50)"
    tee.write("Sampling: 100%|##########| 50/50 [00:17<00:00,  2.89it/s]\n")
    assert progress.message.startswith("Building the clean mesh")
    before = progress.fraction
    tee.write("[flow2mesh] Watershed target constraint done: 8.28s\n")
    assert progress.fraction > before
    tee.write("Sampling:  10%|#         | 5/50 [00:01<00:12]\n")      # never goes back
    assert progress.fraction >= before
    progress.stop()
    assert progress.to_dict()["phase"] == "idle"


class _Sink:
    def __init__(self) -> None:
        self.text = ""

    def write(self, text: str) -> int:
        self.text += text
        return len(text)

    def flush(self) -> None:
        pass


def test_the_tee_passes_everything_on() -> None:
    sink = _Sink()
    tee = server.ProgressTee(sink, server.Progress())
    tee.write("a\rb\n")
    tee.write("c")
    assert sink.text == "a\rb\nc" and not tee.isatty()


# ---- requests, warnings, names -------------------------------------------------------------------------------

def request(root: Path, **extra: Any) -> server.RetopoRequest:
    return server.RetopoRequest(mesh_path=str(root / "props" / "p_1" / "mesh" / "mesh.glb"),
                                out_path=str(root / "props" / "p_1" / "retopo" / "triflow.raw.obj"), **extra)


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    mesh = tmp_path / "props" / "p_1" / "mesh" / "mesh.glb"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"glb")
    return tmp_path


def test_requests_are_checked_in_artist_words(root: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    assert server.validate(request(root), root) is None
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    assert "own projects folder" in server.validate(request(root), elsewhere)
    missing = request(root)
    missing.mesh_path = str(root / "props" / "p_2" / "mesh" / "mesh.glb")
    assert "was not found" in server.validate(missing, root)
    wrong = request(root)
    wrong.out_path = str(root / "props" / "p_1" / "retopo" / "triflow.glb")
    assert ".obj" in server.validate(wrong, root)
    relative = request(root)
    relative.mesh_path = "mesh.glb"
    assert "absolute" in server.validate(relative, root)


def test_the_settings_ranges_agree_everywhere() -> None:
    """The launcher takes the triangles of every size a prop may ask for: two per vertex of 0.25 to 99 thousand, and
    a smaller size is asked for its least, 500."""
    fields = server.RetopoRequest.model_fields
    limits = {type(m).__name__: m for m in fields["faces"].metadata}
    assert (limits["Ge"].ge, limits["Le"].le) == (server.MIN_FACES, server.MAX_FACES) == (500, 198_000)
    assert (adapter_module.MIN_FACES, adapter_module.MAX_FACES) == (server.MIN_FACES, server.MAX_FACES)
    sizes = (projects.RETOPO_MIN_SIZE, 0.25, 0.3, 1, projects.RETOPO_MAX_SIZE)
    assert [adapter_module.faces_for(size) for size in sizes] == [500, 500, 600, 2_000, 198_000]
    assert projects.RETOPO_NUMBERS["triflow_quad_ratio"][1:3] == (0.0, 1.0)
    assert projects.RETOPO_NUMBERS["triflow_seed"][2] == adapter_module.MAX_SEED
    defaults = projects.DEFAULT_STAGE_SETTINGS["retopo"]
    assert {key: defaults[key] for key in adapter_module.DEFAULTS} == adapter_module.DEFAULTS
    assert adapter_module.faces_for(defaults["sizes"][0]) == fields["faces"].default
    assert fields["quad_ratio"].default == adapter_module.DEFAULTS["triflow_quad_ratio"]
    assert fields["qem_threshold"].default == adapter_module.QEM_THRESHOLD
    # Under a thousand vertices the simplification may go as far as the launcher lets it; from there on, TriFlow's own.
    assert [adapter_module.qem_threshold_for(size) for size in (0.1, 0.25, 0.99, 1, 2, 99)] == [1000.0] * 3 + [12.0] * 3
    assert {type(m).__name__: m for m in fields["qem_threshold"].metadata}["Le"].le == adapter_module.QEM_THRESHOLD_SMALL


def test_warnings_name_what_the_artist_can_do() -> None:
    assert server.warnings_for(4100, 4000, 0, True, True, 0.002) is None
    over = server.warnings_for(16_078, 4000, 0, True, True, 0.0)
    assert "16,078" in over and "4,000" in over
    assert "2,406 open edges" in server.warnings_for(4000, 4000, 2406, True, True, 0.0)
    assert "Seed" in server.warnings_for(4000, 4000, 12, True, True, 0.0)
    # holes no more than the 3D model's own: no warning, a quiet note
    assert server.warnings_for(4000, 4000, 125, True, True, 0.0, input_open=172) is None
    assert server.note_for(125, 172) == ("Open where the 3D model is open (125 edges; the model has 172): TriFlow "
                                         "follows its surface.")
    assert server.note_for(0, 172) is None and server.note_for(12, 0) is None
    # clearly fewer triangles than asked (all TriFlow's remesh of the model has): a quiet note too, before the holes'
    short = "TriFlow made 96,552 triangles where 198,000 were asked: it has no more for this model's surface."
    assert server.note_for(0, 0, made=96_552, asked=198_000) == short
    assert server.note_for(0, 0, made=5_900, asked=6_000) is None and server.note_for(0, 0, made=6_100, asked=6_000) is None
    assert server.note_for(125, 172, made=96_552, asked=198_000) == (
        short + " Open where the 3D model is open (125 edges; the model has 172): TriFlow follows its surface.")
    more = server.warnings_for(4000, 4000, 300, True, True, 0.0, input_open=172)    # more holes than the model
    assert "3D model has some too (172)" in more and "Seed" not in more and server.note_for(300, 172) is None
    assert "inward" in server.warnings_for(4000, 4000, 0, True, False, 0.0)
    assert "line up" in server.warnings_for(4000, 4000, 0, True, True, 0.2)


def test_only_old_work_folders_are_cleared(tmp_path: Path) -> None:
    old, new, other = (tmp_path / f"{server.WORK_PREFIX}old", tmp_path / f"{server.WORK_PREFIX}new",
                       tmp_path / "someone-elses")
    for folder in (old, new, other):
        (folder / "in").mkdir(parents=True)
    os.utime(old, (0, 0))
    os.utime(other, (0, 0))
    assert server.clear_old_work(tmp_path) == 1
    assert not old.exists() and new.exists() and other.exists()


def test_names_are_safe_for_triflows_folder() -> None:
    assert server.safe_name("p_3af910") == "p_3af910"
    assert server.safe_name("Crate (old) ü") == "Crate_old"
    assert server.safe_name("") == server.safe_name("...") == "mesh"


def test_the_launcher_finds_triflows_parts_and_leaves_its_folder_alone(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    missing = server.missing_parts(tmp_path)
    assert "inference.py" in missing and "checkpoints/flow_model.safetensors" in missing
    assert "third_party/TRELLIS/trellis/representations/mesh/flexicubes/flexicubes.py" in missing
    entries = server.search_path(tmp_path)
    assert entries == [str(tmp_path), str(tmp_path / "third_party" / "TRELLIS"),
                       str(tmp_path / "third_party" / "Direct3D-S2"), str(server.STUBS)]
    monkeypatch.chdir(paths.ROOT)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "pycache_prefix", None)
    for name in ("SPARSE_BACKEND", "SPARSE_ATTN_BACKEND", "HF_HUB_OFFLINE"):
        monkeypatch.setenv(name, "spconv") if name != "HF_HUB_OFFLINE" else monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    server.configure_env(tmp_path)
    assert Path.cwd() == tmp_path
    assert sys.path[:4] == entries
    assert "SPARSE_BACKEND" not in os.environ and "SPARSE_ATTN_BACKEND" not in os.environ
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert sys.pycache_prefix == str(Path(sys.prefix) / "pycache")
    assert os.environ["NUMBA_CACHE_DIR"] == str(Path(sys.prefix) / "numba-cache")   # numba ignores the prefix


# ---- the routes ---------------------------------------------------------------------------------------------

class FakeEngine:
    def __init__(self, fail: BaseException | None = None) -> None:
        self.ready = True
        self.weights = types.SimpleNamespace(reads=3)
        self.fail = fail
        self.seen: list[tuple[server.RetopoRequest, int]] = []

    def retopo(self, req: server.RetopoRequest, seed: int, progress: server.Progress) -> dict[str, Any]:
        self.seen.append((req, seed))
        assert progress.phase == "working"
        if self.fail is not None:
            raise self.fail
        return {"obj_path": req.out_path, "vertices": 2100, "faces": 4012, "seed": seed, "warning": None}


def client(root: Path, engine: FakeEngine) -> TestClient:
    return TestClient(server.build_app(engine, server.GpuWorker(), server.Progress(), root,
                                       {"attention": "flash_attn", "gpu": "RTX"}))


def test_the_routes_answer_the_hub(root: Path) -> None:
    engine = FakeEngine()
    c = client(root, engine)
    assert c.get("/spellcast/health").json() == {"ready": True, "weights_read": 3, "attention": "flash_attn",
                                                 "gpu": "RTX"}
    assert c.get("/spellcast/progress").json() == {"phase": "idle", "fraction": 0.0, "message": "Idle"}
    body = request(root, faces=4000, quad_ratio=0.5, seed=7).model_dump()
    r = c.post("/spellcast/retopo", json=body)
    assert r.status_code == 200 and r.json()["faces"] == 4012 and "seconds" in r.json()
    assert engine.seen[0][1] == 7 and engine.seen[0][0].quad_ratio == 0.5
    r = c.post("/spellcast/retopo", json={**body, "faces": 100})
    assert r.status_code == 422 and r.json()["kind"] == "input" and "faces" in r.json()["detail"]
    r = c.post("/spellcast/retopo", json={**body, "mesh_path": str(root / "nope.glb")})
    assert r.status_code == 400 and r.json()["kind"] == "input"
    r = c.post("/spellcast/retopo", json={**body, "seed": -1})
    assert r.status_code == 200 and engine.seen[-1][1] >= 0


@pytest.mark.parametrize(("error", "kind"), [
    (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), "oom"),
    (MemoryError(), "ram"),
    (FileNotFoundError(2, "No such file", "checkpoints/sdf_vae.safetensors"), "missing_file"),
    (server.InputProblem("The 3D model mesh.glb has no faces."), "input"),
    (ValueError("boom"), "error"),
])
def test_failures_carry_a_kind(root: Path, error: BaseException, kind: str) -> None:
    r = client(root, FakeEngine(error)).post("/spellcast/retopo", json=request(root).model_dump())
    assert r.status_code >= 400 and r.json()["kind"] == kind and r.json()["detail"]


def test_broken_cuda_answers_restart_and_ends_the_process(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After an error that leaves CUDA unusable TriFlow is no longer ready (health says so), answers "restart" (the
    hub starts it afresh for the next prop), refuses what still comes, and exits a moment later."""
    exits: list[bool] = []
    monkeypatch.setattr(server, "cuda_works", lambda: False)
    engine = FakeEngine(RuntimeError("CUDA error: an illegal memory access was encountered"))
    c = TestClient(server.build_app(engine, server.GpuWorker(), server.Progress(), root, {},
                                    exit_later=lambda: exits.append(True)))
    body = request(root).model_dump()
    r = c.post("/spellcast/retopo", json=body)
    assert r.status_code == 500 and r.json()["kind"] == "restart" and "starts again" in r.json()["detail"]
    assert exits == [True] and engine.ready is False and c.get("/spellcast/health").json()["ready"] is False
    again = c.post("/spellcast/retopo", json=body)
    assert again.status_code == 503 and again.json()["kind"] == "restart" and len(engine.seen) == 1
    # an ordinary failure with CUDA fine: no restart
    monkeypatch.setattr(server, "cuda_works", lambda: True)
    r = client(root, FakeEngine(ValueError("boom"))).post("/spellcast/retopo", json=body)
    assert r.json()["kind"] == "error"


def test_a_second_mesh_is_refused_while_one_runs(root: Path) -> None:
    """TriFlow can't be stopped in the middle of a mesh: after the hub gave up on one (a timeout) the next request
    is refused ("busy": the hub restarts TriFlow), not queued behind it."""
    import threading

    started, release = threading.Event(), threading.Event()

    class Slow(FakeEngine):
        def retopo(self, req: server.RetopoRequest, seed: int, progress: server.Progress) -> dict[str, Any]:
            started.set()
            release.wait(10)
            return super().retopo(req, seed, progress)

    c = client(root, Slow())
    body = request(root).model_dump()
    first = __import__("threading").Thread(target=lambda: c.post("/spellcast/retopo", json=body))
    first.start()
    try:
        assert started.wait(10)
        r = c.post("/spellcast/retopo", json=body)
        assert r.status_code == 409 and r.json()["kind"] == "busy"
    finally:
        release.set()
        first.join(10)
    assert c.post("/spellcast/retopo", json=body).status_code == 200


def test_spconvs_products_go_to_torch_only_where_its_wheel_has_no_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """spconv_shim, for TriFlow and for LATO.2 alike: a card spconv's wheel has GEMM kernels for keeps them, one it
    has none for (a GTX 10xx) gets the products on torch.mm, `always` asks for them anyway; without spconv nothing
    is done. Both launchers ask at their start."""
    import spconv_shim

    monkeypatch.setitem(sys.modules, "spconv", None)                      # not installed
    assert spconv_shim.spconv_products_in_torch((6, 1)) is False
    built = {(7, 5), (8, 6)}
    ops = types.SimpleNamespace(indice_conv="spconv's own")
    fakes = {"spconv": types.ModuleType("spconv"), "spconv.core_cc": types.ModuleType("spconv.core_cc"),
             "spconv.core_cc.cumm": types.ModuleType("spconv.core_cc.cumm"),
             "spconv.core_cc.cumm.common": types.ModuleType("spconv.core_cc.cumm.common"),
             "spconv.pytorch": types.ModuleType("spconv.pytorch"), "torch": types.ModuleType("torch")}
    fakes["spconv.core_cc.cumm.common"].CompileInfo = types.SimpleNamespace(
        arch_is_compiled_gemm=lambda capability: tuple(capability) in built)
    fakes["spconv.pytorch"].ops = ops
    for name, module in fakes.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("SPCONV_ALGO", "")                 # the shim sets it itself: as it was again after the test
    del os.environ["SPCONV_ALGO"]
    assert [spconv_shim.torch_products_where_needed(cap) for cap in ((8, 6), (7, 5))] == [False, False]
    assert ops.indice_conv == "spconv's own" and "SPCONV_ALGO" not in os.environ
    assert spconv_shim.torch_products_where_needed((6, 1)) is True
    assert callable(ops.indice_conv) and os.environ["SPCONV_ALGO"] == "native"
    ops.indice_conv = "spconv's own"
    assert spconv_shim.torch_products_where_needed((8, 6), always=True) is True and callable(ops.indice_conv)
    import ast

    for name, forced in (("triflow_server", True), ("lato2_server", False)):         # a call, not a comment
        calls = [node for node in ast.walk(ast.parse((paths.LAUNCHERS / f"{name}.py").read_text(encoding="utf-8")))
                 if isinstance(node, ast.Call) and ast.unparse(node.func) == "spconv_shim.torch_products_where_needed"]
        assert len(calls) == 1 and ast.unparse(calls[0].args[0]) == "capability", name
        asked = [ast.unparse(k.value) for k in calls[0].keywords if k.arg == "always"]
        assert asked == (["args.spconv == 'torch'"] if forced else []), name      # TriFlow's --spconv torch


def test_the_modules_in_triflows_process_are_mit() -> None:
    """Every Spellcast3D module the TriFlow process loads says it is MIT, not AGPL (next to ADPNCL code)."""
    import ast

    tree = ast.parse((paths.LAUNCHERS / "triflow_server.py").read_text(encoding="utf-8"))
    ours = {path.stem for path in paths.LAUNCHERS.glob("*.py")}
    imported = {name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                for name in ([a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""])
                if name in ours}
    assert imported == {"_loopback_guard", "attn_shim", "spconv_shim"}
    for name in ("triflow_server", *imported):
        doc = ast.get_docstring(ast.parse((paths.LAUNCHERS / f"{name}.py").read_text(encoding="utf-8"))) or ""
        assert "LICENSE OF THIS FILE:" in doc and "MIT" in doc.split("LICENSE OF THIS FILE:")[1][:200], name
    for stub in (paths.LAUNCHERS / "triflow_stubs").rglob("*.py"):
        assert "triflow_stubs" in (paths.LAUNCHERS / "triflow_server.py").read_text(encoding="utf-8")[:1200], stub


# ---- the kaolin stand-in ------------------------------------------------------------------------------------

def test_the_kaolin_stand_in_checks_like_kaolin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(server.STUBS))
    for name in [n for n in sys.modules if n == "kaolin" or n.startswith("kaolin.")]:
        monkeypatch.delitem(sys.modules, name)
    from kaolin.utils.testing import check_tensor

    assert Path(sys.modules["kaolin"].__file__).parent.parent == server.STUBS
    t = types.SimpleNamespace(shape=(10, 3), dtype="float32", device=types.SimpleNamespace(type="cuda"))
    assert check_tensor(t, (None, 3), throw=False)
    assert check_tensor(t, (10, 3), dtype="float32", device="cuda")
    assert check_tensor(t, (10, 8), throw=False) is False
    assert check_tensor(t, (10,), throw=False) is False
    assert check_tensor(t, device="cpu", throw=False) is False
    with pytest.raises(ValueError):
        check_tensor(t, (None, 8))


# ---- the adapter --------------------------------------------------------------------------------------------

@pytest.fixture()
def adapter(monkeypatch: pytest.MonkeyPatch) -> TriflowAdapter:
    a = TriflowAdapter()
    a.__dict__["_gpu"] = gpu.GpuInfo(0, "NVIDIA GeForce RTX 3060", 8.6, 12288, 300, "596.36")
    return a


def test_the_adapter_is_a_retopology_gpu_tool_with_its_license(adapter: TriflowAdapter) -> None:
    assert (adapter.tool_id, adapter.display_name, adapter.stages, adapter.uses_gpu) == \
        ("triflow", "TriFlow", ("retopo",), True)
    assert adapter.license == {"name": "ADPNCL 1.0 (non-commercial)", "url": LICENSE_URL, "noncommercial": True,
                               "links": [{"label": "MeshLib terms", "url": adapter_module.MESHLIB_LICENSE_URL}]}
    # MeshLib's terms at the very version TriFlow's venv installs
    pinned = re.search(r"^meshlib==(\S+)", (paths.PACKAGE / "install" / "requirements-triflow.txt").read_text(
        encoding="utf-8"), re.M)
    assert pinned and adapter_module.MESHLIB_VERSION == pinned.group(1)
    assert adapter_module.MESHLIB_LICENSE_URL.endswith(f"/v{pinned.group(1)}/LICENSE")
    page = (paths.STATIC / "js" / "licenses.js").read_text(encoding="utf-8")
    assert f"'{adapter_module.MESHLIB_LICENSE_URL}'" in page             # the page's default: the same link
    assert adapter.supported() == (True, "")
    assert isinstance(build_adapters()[paths.TRIFLOW], TriflowAdapter)
    old = TriflowAdapter()
    old.__dict__["_gpu"] = gpu.GpuInfo(0, "GeForce GTX 980", 5.2, 4096, 0, "596.36")
    assert old.supported() == gpu.triflow_supported(old._gpu) and old.supported()[0] is False


def test_the_launch_spec_keeps_triflows_folder_clean(adapter: TriflowAdapter,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPARSE_BACKEND", "spconv")        # from the artist's shell: must not reach TriFlow
    monkeypatch.setenv("PYTHONPATH", r"C:\elsewhere")
    spec = adapter.launch_spec(7775, {})
    repo = paths.TRIFLOW_DIR
    assert spec.cwd == repo
    assert spec.argv[:2] == [str(paths.venv_python("triflow")), str(paths.LAUNCHERS / "triflow_server.py")]
    assert spec.argv[spec.argv.index("--repo") + 1] == str(repo)
    assert spec.argv[spec.argv.index("--port") + 1] == "7775"
    assert spec.argv[spec.argv.index("--projects-root") + 1] == str(paths.PROJECTS)
    assert spec.env["PYTHONPATH"].split(os.pathsep) == [
        str(repo), str(repo / "third_party" / "TRELLIS"), str(repo / "third_party" / "Direct3D-S2"),
        str(paths.LAUNCHERS / "triflow_stubs")]
    assert Path(spec.env["PYTHONPYCACHEPREFIX"]) == paths.venv_dir("triflow") / "pycache"
    assert spec.env["ATTN_BACKEND"] == "flash_attn"
    assert "SPARSE_BACKEND" not in spec.env
    assert spec.ready_timeout >= 300
    older = TriflowAdapter()
    older.__dict__["_gpu"] = gpu.GpuInfo(0, "GeForce RTX 2070", 7.5, 8192, 0, "596.36")
    assert older.launch_spec(7775, {}).env["ATTN_BACKEND"] == "sdpa"
    unknown = TriflowAdapter()
    unknown.__dict__["_gpu"] = None
    monkeypatch.setenv("ATTN_BACKEND", "naive")
    assert "ATTN_BACKEND" not in unknown.launch_spec(7775, {}).env    # the launcher asks torch


def test_the_adapter_checks_the_settings(adapter: TriflowAdapter) -> None:
    adapter.check_settings("retopo", {"sizes": [2]})                        # the defaults
    adapter.check_settings("retopo", {"sizes": [99, 1], "triflow_quad_ratio": 0, "triflow_seed": 5})
    for bad in ({"sizes": []}, {"sizes": [2], "triflow_quad_ratio": 1.5}, {"sizes": [2], "triflow_quad_ratio": "0.5"},
                {"sizes": [2], "triflow_seed": -1}):
        with pytest.raises(base.StageError):
            adapter.check_settings("retopo", bad)
    assert adapter_module.triflow_params({"triflow_seed": None}) == {"quad_ratio": 0.95, "seed": 0}


def run(adapter: TriflowAdapter, prop_dir: Path, handler: Any, unwrapped: dict[str, Any] | Exception,
        params: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[Any], list[tuple[float, str]]]:
    calls: list[Any] = []
    reports: list[tuple[float, str]] = []

    async def unwrapper(src: Path, dst: Path, leniency: float) -> dict[str, Any]:
        calls.append((src, dst, leniency))
        assert src.is_file()                 # TriFlow's mesh is there while Instant Meshes unwraps it
        if isinstance(unwrapped, Exception):
            raise unwrapped
        dst.write_text("obj with uvs")
        return unwrapped

    async def go() -> dict[str, Any]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            ctx = base.StageContext(project="demo", prop_id="p_1", prop_dir=prop_dir, base_url="http://tool",
                                    client=http, settings={}, report=lambda f, m: reports.append((f, m)),
                                    cancelled=lambda: False, stage="retopo", unwrapper=unwrapper)
            prop = {"id": "p_1", "mesh": {"glb": "mesh/mesh.glb"}, "retopo": {"settings": {}}}
            return await adapter.run_stage(ctx, prop, {"method": "triflow", "sizes": [2],
                                                        "triflow_quad_ratio": 0.8, "triflow_seed": 4,
                                                        **(params or {})})

    return asyncio.run(go()), calls, reports


def test_a_prop_goes_through_triflow_then_the_uv_unwrap(adapter: TriflowAdapter, root: Path) -> None:
    prop_dir = root / "props" / "p_1"
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/spellcast/progress":
            return httpx.Response(200, json={"phase": "working", "fraction": 0.5, "message": "Sampling"})
        body = json.loads(request.content)
        sent.append(body)
        Path(body["out_path"]).write_text("o")
        return httpx.Response(200, json={"obj_path": body["out_path"], "vertices": 1600, "faces": 3100,
                                         "seconds": 70.0, "seed": 4, "warning": None, "watertight": True})

    patch, calls, reports = run(adapter, prop_dir, handler,
                                {"vertices": 1900, "faces": 3100, "has_uv": True, "warning": "", "seconds": 2.5},
                                {"uv_leniency": 0.7, "sizes": [0.3, 22]})
    raw = {size: prop_dir / "retopo" / f"{size}k" / "triflow.raw.obj" for size in (0.3, 22)}
    assert sent == [{"mesh_path": str((prop_dir / "mesh" / "mesh.glb").resolve()), "out_path": str(raw[size].resolve()),
                     "faces": faces, "quad_ratio": 0.8, "seed": 4, "qem_threshold": threshold, "name": "p_1"}
                    for size, faces, threshold in ((0.3, 600, 1000.0), (22, 44_000, 12.0))], \
        "two triangles per vertex wanted; a small mesh may be simplified as far as it takes"
    assert calls == [(raw[size], prop_dir / "retopo" / f"{size}k" / "triflow.obj", 0.7) for size in (0.3, 22)]
    assert not any(path.exists() for path in raw.values())
    assert (patch["method"], patch["source"], patch["saved_from"]) == ("triflow", "mesh/mesh.glb", "auto")
    assert [(v["size"], v["obj"]) for v in patch["variants"]] == [(0.3, "retopo/0.3k/triflow.obj"),
                                                                 (22, "retopo/22k/triflow.obj")]
    stats = patch["variants"][1]["stats"]
    assert (stats["vertices"], stats["faces"], stats["has_uv"], stats["warning"]) == (1900, 3100, True, None)
    assert stats["seconds"] == 72.5 and stats["target_faces"] == 44_000 and stats["seed"] == 4
    assert reports[-1][0] == 1.0 and all(0.0 <= f <= 1.0 for f, _ in reports)
    assert [(round(f, 2), m) for f, m in reports[:2]] == [(0.0, "0.3k vertices (1 of 2) · Sending the 3D model to TriFlow…"),
                                                          (0.45, "22k vertices (2 of 2) · Sending the 3D model to TriFlow…")]


def test_triflows_own_words_reach_the_artist(adapter: TriflowAdapter, root: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "TriFlow ran out of GPU memory. Close other programs...",
                                         "kind": "oom"})

    with pytest.raises(base.StageError, match="ran out of GPU memory"):
        run(adapter, root / "props" / "p_1", handler, {})


def test_a_prop_without_a_3d_model_is_refused(adapter: TriflowAdapter, tmp_path: Path) -> None:
    with pytest.raises(base.StageError, match="3D model"):
        run(adapter, tmp_path, lambda r: httpx.Response(500), {})


# ---- TriFlow's checkout stays byte-identical -----------------------------------------------------------------

def _git() -> str | None:
    found = shutil.which("git")
    return found or (str(paths.PORTABLE_GIT) if paths.PORTABLE_GIT.exists() else None)


def _run_git(git: str, repo: Path, *args: str) -> str:
    # --no-optional-locks: reading must not even refresh the index's stat cache
    return subprocess.run([git, "--no-optional-locks", "-C", str(repo), *args], capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def tree_changes(repo: Path, git: str, allowed: set[str] = frozenset(), prefix: str = "") -> list[str]:
    """Every difference between a checkout's files and its HEAD commit, nested checkouts included: tracked files
    whose bytes differ (git hash-object --no-filters, so CRLF conversion counts), missing ones, nested checkouts
    at another commit, and files that are not TriFlow's (untracked or ignored, e.g. a __pycache__) except
    `allowed`. [] when the checkout is exactly its commit."""
    changes: list[str] = []
    listing = [line.split("\t", 1) for line in _run_git(git, repo, "ls-tree", "-r", "-z", "HEAD").split("\0") if line]
    blobs = []
    for meta, path in listing:
        mode, kind, sha = meta.split()
        if kind == "commit":
            nested = repo / path
            if not (nested / ".git").exists():
                changes.append(f"{prefix}{path}: nested checkout missing")
                continue
            head = _run_git(git, nested, "rev-parse", "HEAD").strip()
            if head != sha:
                changes.append(f"{prefix}{path}: at {head[:10]}, not {sha[:10]}")
            changes += tree_changes(nested, git, allowed, f"{prefix}{path}/")
        elif not (repo / path).is_file():
            changes.append(f"{prefix}{path}: missing")
        else:
            blobs.append((path, sha))
    if blobs:
        hashed = subprocess.run([git, "-C", str(repo), "hash-object", "--no-filters", "--stdin-paths"],
                                input="\n".join(path for path, _ in blobs) + "\n", capture_output=True,
                                text=True, encoding="utf-8", check=True).stdout.split()
        changes += [f"{prefix}{path}: bytes differ" for (path, sha), got in zip(blobs, hashed) if got != sha]
    status = _run_git(git, repo, "status", "--porcelain", "--ignored", "--untracked-files=all",
                      "--ignore-submodules=all", "-z")
    for entry in status.split("\0"):
        if entry[:2] in ("??", "!!") and f"{prefix}{entry[3:]}" not in allowed:
            changes.append(f"{prefix}{entry[3:]}: not TriFlow's file")
    return changes


def triflow_checkouts() -> list[Path]:
    found = []
    # a git checkout only: a release copy has TriFlow's files without git data (prepare_release checked those)
    if ((paths.TRIFLOW_DIR / ".git").exists()
            and (paths.TRIFLOW_DIR / "third_party" / "TRELLIS" / "trellis" / "__init__.py").is_file()):
        found.append(paths.TRIFLOW_DIR)
    extra = os.environ.get("SPELLCAST_TRIFLOW_REPO")
    if extra:
        found.append(Path(extra))
    return found


@pytest.mark.parametrize("repo", triflow_checkouts() or [pytest.param(None, marks=pytest.mark.skip(
    reason="code/triflow is not a git checkout with its submodules (and no SPELLCAST_TRIFLOW_REPO)"))])
def test_triflows_checkout_is_byte_identical_to_its_commit(repo: Path) -> None:
    git = _git()
    if git is None:
        pytest.skip("git is not installed")
    assert _run_git(git, repo, "rev-parse", "HEAD").strip() == TRIFLOW_COMMIT
    for nested, commit in NESTED.items():
        assert _run_git(git, repo / nested, "rev-parse", "HEAD").strip() == commit
    assert tree_changes(repo, git, ALLOWED_EXTRA) == []


def test_the_byte_check_notices_a_change(tmp_path: Path) -> None:
    git = _git()
    if git is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "r"
    repo.mkdir()
    for args in (("init", "-q"), ("config", "core.autocrlf", "false"), ("config", "user.email", "t@t"),
                 ("config", "user.name", "t")):
        subprocess.run([git, "-C", str(repo), *args], check=True)
    (repo / "a.py").write_bytes(b"x = 1\n")
    (repo / ".gitignore").write_bytes(b"checkpoints/*\n__pycache__/\n")
    subprocess.run([git, "-C", str(repo), "add", "."], check=True)
    subprocess.run([git, "-C", str(repo), "commit", "-q", "-m", "c"], check=True)
    assert tree_changes(repo, git) == []
    (repo / "checkpoints").mkdir()
    (repo / "checkpoints" / "w.safetensors").write_bytes(b"w")
    assert tree_changes(repo, git, {"checkpoints/w.safetensors"}) == []
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "a.cpython-311.pyc").write_bytes(b"pyc")
    (repo / "a.py").write_bytes(b"x = 1\r\n")                      # what core.autocrlf=true would do
    assert sorted(tree_changes(repo, git, {"checkpoints/w.safetensors"})) == [
        "__pycache__/a.cpython-311.pyc: not TriFlow's file", "a.py: bytes differ"]
