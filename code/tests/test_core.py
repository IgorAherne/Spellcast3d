"""Hub core tests: settings, projects, procman (real child processes), jobs and the HTTP API.

Run with the hub venv:  <venv>\\Scripts\\python.exe -m pytest code\\tests -q
Tools are faked by tests/fake_tool.py, a tiny HTTP server started with this same Python.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import json
import os
import random
import socket
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest
from fastapi.testclient import TestClient

from spellcast import paths
from spellcast import procman as procman_mod
from spellcast.adapters.base import LaunchSpec, StageContext, StageError, ToolAdapter, ToolNeedsRestart
from spellcast.gpu import GpuInfo
from spellcast.jobs import CANCELLING, Job, JobError, JobQueue
from spellcast.procman import ProcessManager, ToolError
from spellcast.projects import ProjectError, ProjectStore, safe_name, sniff_image
from spellcast.server import create_app
from spellcast.settings import SettingsError, SettingsStore, deep_merge

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Spellcast3D is Windows-only")

FAKE_TOOL = Path(__file__).resolve().parent / "fake_tool.py"
PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 64
JPG = b"\xff\xd8\xff\xe0" + b"\0" * 64
WEBP = b"RIFF\x24\0\0\0WEBPVP8 " + b"\0" * 64


# ---- helpers ------------------------------------------------------------------------------------

def free_ports(n: int) -> list[int]:
    """Free ports 10 apart, below the OS ephemeral range (outgoing connections take ports from
    that range at any moment, so a port found free there may be gone by the time a tool binds it)."""
    ports: list[int] = []
    port = random.randrange(20000, 40000)
    while len(ports) < n:
        if procman_mod.port_is_free(port):
            ports.append(port)
        port += 10
    return ports


def make_settings(tmp_path: Path, user: dict[str, Any] | None = None) -> SettingsStore:
    defaults = json.loads(paths.DEFAULTS_FILE.read_text(encoding="utf-8"))
    ids = list(dict.fromkeys([*paths.ALL_VENV_IDS, *defaults["ports"], paths.LATO2, paths.TRIFLOW]))
    defaults["ports"] = dict(zip(ids, free_ports(len(ids))))
    (tmp_path / "defaults.json").write_text(json.dumps(defaults), encoding="utf-8")
    if user is not None:
        (tmp_path / "settings.json").write_text(json.dumps(user), encoding="utf-8")
    return SettingsStore(tmp_path / "defaults.json", tmp_path / "settings.json")


_k32 = ctypes.WinDLL("kernel32", use_last_error=True) if os.name == "nt" else None
if _k32 is not None:
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]


def pid_alive(pid: int) -> bool:
    handle = _k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        _k32.GetExitCodeProcess(handle, ctypes.byref(code))
        return code.value == 259                    # STILL_ACTIVE
    finally:
        _k32.CloseHandle(handle)


def wait_until(predicate: Any, timeout: float = 20.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


async def wait_state(pm: ProcessManager, tool: str, states: set[str], timeout: float = 20.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = pm.status(tool)["state"]
        if state in states:
            return state
        await asyncio.sleep(0.05)
    raise AssertionError(f"{tool} never reached {states}: {pm.status(tool)}")


class FakeAdapter(ToolAdapter):
    """Launches fake_tool.py; run_stage behaviour is chosen by params["mode"]."""

    def __init__(self, tool_id: str, stage: str, uses_gpu: bool, cwd: Path, args: tuple[str, ...] = (),
                 ready_timeout: float = 30.0, install: str = "installed",
                 supported: tuple[bool, str] = (True, ""), gpu_at_start: bool = True):
        self.tool_id, self.stage, self.uses_gpu, self.gpu_at_start = tool_id, stage, uses_gpu, gpu_at_start
        self.display_name = f"Fake {tool_id}"
        self.cwd, self.args, self.ready_timeout = cwd, list(args), ready_timeout
        self.install, self._supported = install, supported
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.sessions: list[dict[str, Any]] = []
        self.loaded_from: dict[str, tuple[int, int]] = {}    # session id -> (mtime_ns, size) of its file
        self.on_stop_calls = 0

    def install_status(self) -> dict[str, Any]:
        return {"status": self.install, "reason": "" if self.install == "installed" else "venv missing"}

    def supported(self) -> tuple[bool, str]:
        return self._supported

    def launch_spec(self, port: int, settings: dict[str, Any]) -> LaunchSpec:
        return LaunchSpec(argv=[sys.executable, str(FAKE_TOOL), "--port", str(port), *self.args],
                          cwd=self.cwd, env=paths.clean_env(None), ready_timeout=self.ready_timeout)

    async def health(self, client: httpx.AsyncClient, base_url: str) -> bool:
        return (await client.get(f"{base_url}/health", timeout=2.0)).status_code == 200

    async def on_stop(self, client: httpx.AsyncClient, base_url: str) -> None:
        self.on_stop_calls += 1

    async def run_stage(self, ctx: StageContext, prop: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((ctx.prop_id, dict(params)))
        mode = params.get("mode", "ok")
        if mode == "stage_error":
            raise StageError(f"Pick an image for {prop['name']} first.")
        if mode == "crash":
            raise RuntimeError("boom")
        if mode == "restart_once" and len(self.calls) == 1:
            raise ToolNeedsRestart(f"{self.display_name} has to start again after an error.")
        if mode == "finish_after_cancel":      # like TriFlow: can't stop in the middle of a prop, saves it
            for i in range(50):
                ctx.report(i / 50, f"step {i}")
                await asyncio.sleep(0.02)
            ctx.completed = True
        if mode == "slow":
            for i in range(150):
                if ctx.cancelled():
                    raise StageError("Cancelled.")
                ctx.report(i / 150, f"step {i}")
                await asyncio.sleep(0.02)
        pid = (await ctx.client.get(f"{ctx.base_url}/echo")).json()["pid"]
        out = ctx.prop_dir / self.stage / "out.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("result", encoding="utf-8")
        ctx.report(0.5, "halfway")
        return {"result": ctx.rel(out), "tool_pid": pid}

    # Forge extras
    async def list_models(self, client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
        await client.get(f"{base_url}/echo")
        return {"checkpoints": [{"title": "a.safetensors [123]", "name": "a.safetensors"}],
                "current": "a.safetensors", "samplers": ["Euler"], "schedulers": ["Karras"]}

    async def refresh_models(self, client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
        return await self.list_models(client, base_url)

    # Instant Meshes extras
    async def open_session(self, client: httpx.AsyncClient, base_url: str, src: Path,
                           settings: dict[str, Any] | None = None, reuse: str | None = None) -> dict[str, Any]:
        # Like launchers/imesh_server.py: `reuse` is kept only while the file is unchanged.
        self.sessions.append({"src": src, "reuse": reuse, "settings": settings})
        signature = (src.stat().st_mtime_ns, src.stat().st_size)
        reused = reuse is not None and self.loaded_from.get(reuse) == signature
        session_id = reuse if reused else f"s{len(self.sessions)}"
        replaced = None if reused or reuse is None else "mesh_changed" if reuse in self.loaded_from else "gone"
        self.loaded_from[session_id] = signature
        return {"session_id": session_id, "viewer_path": f"/viewer?session={session_id}", "reused": reused,
                "replaced": replaced, "lost_strokes": False}

    async def save_session(self, client: httpx.AsyncClient, base_url: str, session_id: str,
                           dst: Path) -> dict[str, Any]:
        dst.write_text("v 0 0 0\n", encoding="utf-8")
        # settings: as ImeshAdapter.save_session gives them (the session's own, checked by accepted_settings).
        return {"path": str(dst), "vertices": 1, "faces": 0, "has_uv": False, "warning": "no UVs", "seconds": 0.1,
                "settings": {"target_vertices": 3390, "pure_quad": True, "rosy": 6, "posy": 3, "uv_leniency": 0.8}}


class ModelAdapter(FakeAdapter):
    """A tool that loads one of several models at its start, as the image generator does (aim / serves)."""

    wished: str | None = None
    holding = "turbo"

    def aim(self, wish: Any) -> None:
        self.wished = wish.get("model")

    def serves(self, wish: Any, settings: Any) -> bool:
        return wish.get("model") in (None, self.holding)

    def launch_spec(self, port: int, settings: dict[str, Any]) -> LaunchSpec:
        self.holding, self.wished = self.wished or "turbo", None
        self.launched = [*getattr(self, "launched", []), self.holding]
        return super().launch_spec(port, settings)


def fake_adapters(tmp_path: Path, **overrides: FakeAdapter) -> dict[str, ToolAdapter]:
    adapters: dict[str, ToolAdapter] = {
        "forge": FakeAdapter("forge", "images", True, tmp_path),
        "pixal3d": FakeAdapter("pixal3d", "mesh", True, tmp_path),
        "imesh": FakeAdapter("imesh", "retopo", False, tmp_path),
    }
    adapters.update(overrides)
    return adapters


@contextlib.asynccontextmanager
async def manager(tmp_path: Path, adapters: dict[str, ToolAdapter],
                  user_settings: dict[str, Any] | None = None) -> AsyncIterator[ProcessManager]:
    async with httpx.AsyncClient(trust_env=False) as client:
        pm = ProcessManager(adapters, make_settings(tmp_path, user_settings), client, logs_dir=tmp_path / "logs")
        await pm.startup()
        try:
            yield pm
        finally:
            await pm.stop_all()


# ---- settings -----------------------------------------------------------------------------------

def test_deep_merge() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    assert deep_merge(base, {"a": {"b": 9}, "e": 4}) == {"a": {"b": 9, "c": 2}, "d": 3, "e": 4}
    assert deep_merge(base, {"a": None}) == base
    assert deep_merge(base, {"a": {"c": None}, "d": None}, null_removes=True) == {"a": {"b": 1}}
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}


def test_settings_patch_and_validation(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    assert s.get()["gpu_exclusive"] is True and s.warnings == []
    merged = s.patch({"gpu_exclusive": False, "forge": {"extra_args": ["--xformers"]}})
    assert merged["gpu_exclusive"] is False and merged["forge"]["ckpt_dirs"] == []
    assert json.loads((tmp_path / "settings.json").read_text()) == {"gpu_exclusive": False,
                                                                    "forge": {"extra_args": ["--xformers"]}}
    with pytest.raises(SettingsError, match="ports.forge"):
        s.patch({"ports": {"forge": 80}})
    assert s.patch({"gpu_exclusive": None})["gpu_exclusive"] is True     # null = back to the default
    with pytest.raises(SettingsError):
        s.patch(["not", "a", "dict"])  # type: ignore[arg-type]


def test_settings_warn_about_missing_model_folder(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    s.patch({"forge": {"ckpt_dirs": [str(tmp_path), str(tmp_path / "typo")]}})
    assert len(s.warnings) == 1 and "typo" in s.warnings[0]
    s.patch({"forge": {"ckpt_dirs": [str(tmp_path)]}})
    assert s.warnings == []


def test_settings_broken_user_file_falls_back(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    (tmp_path / "settings.json").write_text("{broken", encoding="utf-8")
    assert s.get()["open_browser"] is True
    assert any("not valid settings" in w for w in s.warnings)
    assert (tmp_path / "settings.json.broken").read_text() == "{broken"
    time.sleep(0.05)
    (tmp_path / "settings.json").write_text('{"ports": {"forge": "x"}, "imesh": {"threads": 4}}', encoding="utf-8")
    os.utime(tmp_path / "settings.json", (time.time() + 5, time.time() + 5))
    merged = s.get()
    assert merged["imesh"]["threads"] == 4
    assert isinstance(merged["ports"]["forge"], int)                    # bad value replaced by the default
    assert any("ports.forge" in w for w in s.warnings)


def test_pixal3d_settings(tmp_path: Path) -> None:
    """pixal3d.wave_size: how many props of a job go through Pixal3D's steps together (0: all of them).
    pixal3d.low_vram is read by nothing any more, and a settings.json that holds it is still fine."""
    s = make_settings(tmp_path, {"pixal3d": {"low_vram": True}})
    assert s.get()["pixal3d"] == {"low_vram": True, "attn_backend": "auto", "sage": "auto", "int8": "auto",
                                  "coarse_exact": True, "wave_size": 0, "allow_spill": False} and s.warnings == []
    assert s.patch({"pixal3d": {"coarse_exact": False}})["pixal3d"]["coarse_exact"] is False   # Faster 3D generation
    with pytest.raises(SettingsError, match="pixal3d.coarse_exact"):
        s.patch({"pixal3d": {"coarse_exact": "off"}})
    assert s.patch({"pixal3d": {"sage": False}})["pixal3d"]["sage"] is False       # flash_attn, bit for bit
    with pytest.raises(SettingsError, match="pixal3d.sage"):
        s.patch({"pixal3d": {"sage": "on"}})
    assert s.patch({"pixal3d": {"int8": False}})["pixal3d"]["int8"] is False       # bf16 GEMMs, bit for bit
    with pytest.raises(SettingsError, match="pixal3d.int8"):
        s.patch({"pixal3d": {"int8": 8}})
    assert s.patch({"pixal3d": {"wave_size": 8, "low_vram": "auto"}})["pixal3d"]["wave_size"] == 8
    for bad in (-1, 2.5, "8", True):
        with pytest.raises(SettingsError, match="pixal3d.wave_size"):
            s.patch({"pixal3d": {"wave_size": bad}})
    assert s.get()["pixal3d"]["wave_size"] == 8
    (tmp_path / "settings.json").write_text('{"pixal3d": {"wave_size": "many"}}', encoding="utf-8")
    os.utime(tmp_path / "settings.json", (time.time() + 5, time.time() + 5))
    assert s.get()["pixal3d"]["wave_size"] == 0 and any("pixal3d.wave_size" in w for w in s.warnings)


def test_pixal3d_spill_setting(tmp_path: Path) -> None:
    """pixal3d.allow_spill: off by default, so a 3D job never spills into shared GPU memory; true or false only."""
    s = make_settings(tmp_path)
    assert s.get()["pixal3d"]["allow_spill"] is False
    assert s.patch({"pixal3d": {"allow_spill": True}})["pixal3d"]["allow_spill"] is True
    for bad in (1, 0, "true", "yes"):
        with pytest.raises(SettingsError, match="pixal3d.allow_spill"):
            s.patch({"pixal3d": {"allow_spill": bad}})
    assert s.get()["pixal3d"]["allow_spill"] is True
    assert s.patch({"pixal3d": {"allow_spill": None}})["pixal3d"]["allow_spill"] is False     # back to the default
    (tmp_path / "settings.json").write_text('{"pixal3d": {"allow_spill": "on"}}', encoding="utf-8")
    os.utime(tmp_path / "settings.json", (time.time() + 5, time.time() + 5))
    assert s.get()["pixal3d"]["allow_spill"] is False and any("pixal3d.allow_spill" in w for w in s.warnings)


# ---- projects -----------------------------------------------------------------------------------

def test_safe_name() -> None:
    assert safe_name("My Props") == "My Props"
    assert safe_name('  a<b>:c"d/e\\f|g?h*  ') == "a b c d e f g h"
    assert safe_name("..") == ""
    assert safe_name("con") == "con project"
    assert safe_name("Château ßeta") == "Château ßeta"
    assert len(safe_name("x" * 200)) == 60


def test_sniff_image() -> None:
    assert (sniff_image(PNG), sniff_image(JPG), sniff_image(WEBP)) == (".png", ".jpg", ".webp")
    assert sniff_image(b"GIF89a....") is None


def test_projects_and_props(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    project = store.create_project("  Castle: props  ")
    assert project["name"] == "Castle props"
    assert project["ui"] == {"stage": "images"}
    with pytest.raises(ProjectError) as e:
        store.create_project("castle props")                 # same folder on Windows
    assert e.value.status_code == 409
    with pytest.raises(ProjectError):
        store.create_project("???")
    with pytest.raises(ProjectError) as e:
        store.get_project("..\\..\\Windows")
    assert e.value.status_code == 404

    first = store.add_prop("Castle props", "Barrel")
    assert first["id"].startswith("p_") and len(first["id"]) == 8
    assert first["images"]["settings"]["steps"] == 25 and first["images"]["chosen"] is None
    assert first["retopo"] == {"settings": {"method": "triflow", "sizes": [2], "target_vertices": 5000,
                                            "pure_quad": False, "crease_angle": -1, "smooth_iter": 2,
                                            "uv_leniency": 1.0, "rosy": 4, "posy": 4, "extrinsic": True,
                                            "align_to_boundaries": False, "lato2_seed": 0,
                                            "triflow_quad_ratio": 0.95, "triflow_seed": 0},
                               "obj": None, "method": None, "source": None, "session_id": None, "stale": False,
                               "accepted": False, "variants": []}
    assert first["projection"]["glb"] is None and first["projection"]["settings"]["texture_size"] == 2048
    assert first["rig"] == {}
    store.patch_prop("Castle props", first["id"], {"images": {"settings": {"prompt": "barrel", "cfg_scale": 3.5,
                                                                           "checkpoint": "a.safetensors"}},
                                                   "mesh": {"settings": {"resolution": 1536}}})
    second = store.add_prop("Castle props", "")
    assert second["name"] == "Prop 2"
    assert second["images"]["settings"]["prompt"] == ""                       # the prompt is not inherited
    assert second["images"]["settings"]["cfg_scale"] == 3.5                   # model / CFG are
    assert second["images"]["settings"]["checkpoint"] == "a.safetensors"
    assert second["mesh"]["settings"]["resolution"] == 1536

    with pytest.raises(ProjectError):
        store.patch_prop("Castle props", first["id"], {"images": {"chosen": "x"}})
    with pytest.raises(ProjectError):
        store.patch_prop("Castle props", first["id"], {"id": "p_000000"})
    renamed = store.patch_prop("Castle props", first["id"], {"name": "  Big barrel "})
    assert renamed["name"] == "Big barrel"

    ui = store.patch_ui("Castle props", {"ui": {"stage": "retopo", "panel": {"open": True}}})["ui"]
    assert ui == {"stage": "retopo", "panel": {"open": True}}
    assert store.patch_ui("Castle props", {"ui": {"panel": {"wide": 1}}})["ui"]["panel"] == {"open": True, "wide": 1}
    with pytest.raises(ProjectError):
        store.patch_ui("Castle props", {"name": "other"})

    listed = store.list_projects()
    assert listed == [{"id": "Castle props", "name": "Castle props", "untitled": False, "prop_count": 2,
                       "updated": listed[0]["updated"]}]

    (store.prop_dir("Castle props", second["id"]) / "images").mkdir(parents=True)
    store.delete_prop("Castle props", second["id"])
    assert [p["id"] for p in store.get_project("Castle props")["props"]] == [first["id"]]
    assert not store.prop_dir("Castle props", second["id"]).exists()
    assert list((tmp_path / "projects" / "Castle props" / "deleted").iterdir())[0].name.startswith(second["id"])
    with pytest.raises(ProjectError) as e:
        store.get_prop("Castle props", second["id"])
    assert e.value.status_code == 404


def test_a_deleted_prop_comes_back_at_its_place(tmp_path: Path) -> None:
    """Ctrl+Z in the props column: the prop's entry, place and folder come back from deleted/; once only."""
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    first, second, third = (store.add_prop("P", n) for n in ("A", "B", "C"))
    (store.prop_dir("P", second["id"]) / "images").mkdir(parents=True)
    (store.prop_dir("P", second["id"]) / "images" / "0001.png").write_bytes(b"png")
    store.patch_prop("P", second["id"], {"images": {"settings": {"prompt": "a crate"}}})
    name = store.delete_prop("P", second["id"])
    gone = store.delete_prop("P", third["id"])          # a prop without a folder
    assert [p["id"] for p in store.get_project("P")["props"]] == [first["id"]]

    store.restore_prop("P", gone)                       # the newest first, as Ctrl+Z does
    assert [p["id"] for p in store.get_project("P")["props"]] == [first["id"], third["id"]]
    assert not (store.project_dir("P") / "deleted" / gone).exists()
    back = store.restore_prop("P", name)
    assert back["id"] == second["id"] and back["images"]["settings"]["prompt"] == "a crate"
    assert [p["id"] for p in store.get_project("P")["props"]] == [first["id"], second["id"], third["id"]]
    assert (store.prop_dir("P", second["id"]) / "images" / "0001.png").read_bytes() == b"png"
    assert not (store.prop_dir("P", second["id"]) / "deleted-prop.json").exists()

    for bad, status in ((name, 404), ("../P", 404), ("p_zz_1", 404)):
        with pytest.raises(ProjectError) as e:
            store.restore_prop("P", bad)
        assert e.value.status_code == status
    again = store.delete_prop("P", second["id"])
    assert (store.project_dir("P") / "deleted" / again / "deleted-prop.json").is_file()


def test_images_and_references(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    prop = store.add_prop("P", "Crate")
    pid = prop["id"]
    prop = store.add_image("P", pid, PNG)
    assert prop["images"]["candidates"] == ["images/upload_001.png"] and prop["images"]["chosen"] == "images/upload_001.png"
    prop = store.add_image("P", pid, JPG)
    assert prop["images"]["chosen"] == "images/upload_002.jpg"
    assert store.select_image("P", pid, "images/upload_001.png")["images"]["chosen"] == "images/upload_001.png"
    with pytest.raises(ProjectError):
        store.select_image("P", pid, "../../elsewhere.png")
    for data, status in ((b"", 400), (b"hello world", 415)):
        with pytest.raises(ProjectError) as e:
            store.add_image("P", pid, data)
        assert e.value.status_code == status
    monkeypatch.setattr("spellcast.projects.MAX_UPLOAD_BYTES", 10)
    with pytest.raises(ProjectError) as e:
        store.add_image("P", pid, PNG)
    assert e.value.status_code == 413
    monkeypatch.undo()

    prop = store.add_references("P", pid, [(WEBP, "a.webp")])
    first_ref = prop["images"]["references"][0]
    assert first_ref == "reference/reference_001.webp"
    prop = store.add_references("P", pid, [(PNG, "b.png")])
    assert prop["images"]["references"] == [first_ref, "reference/reference_002.png"]
    prop = store.remove_references("P", pid, first_ref)
    assert prop["images"]["references"] == ["reference/reference_002.png"]
    assert not (store.prop_dir("P", pid) / first_ref).exists()
    prop = store.add_references("P", pid, [(JPG, "c.jpg")])
    assert prop["images"]["references"][-1] == "reference/reference_003.jpg"    # a new name: no stale browser cache
    prop = store.remove_references("P", pid)
    assert prop["images"]["references"] == []
    assert not any((store.prop_dir("P", pid) / "reference").iterdir())


def test_resolve_file_stays_inside_project(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("A")
    store.create_project("B")
    (tmp_path / "projects" / "B" / "secret.txt").write_text("b")
    (tmp_path / "secret.txt").write_text("root")
    ok = tmp_path / "projects" / "A" / "props" / "x.glb"
    ok.write_bytes(b"glb")
    assert store.resolve_file("A", "props/x.glb") == ok.resolve()
    for bad in ("../B/secret.txt", "..\\B\\secret.txt", "../../secret.txt", str(tmp_path / "secret.txt"),
                "props/x.glb:stream", "props", "", "props/missing.glb", "C:/Windows/win.ini"):
        with pytest.raises(ProjectError) as e:
            store.resolve_file("A", bad)
        assert e.value.status_code == 404, bad
    with pytest.raises(ProjectError):
        store.resolve_file("../projects/A", "props/x.glb")


def test_concurrent_stage_patches_are_not_lost(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    ids = [store.add_prop("P", f"prop {i}")["id"] for i in range(4)]

    def worker(pid: str) -> None:
        for n in range(15):
            store.apply_stage_patch("P", pid, "mesh", {f"k{n}": n})

    threads = [threading.Thread(target=worker, args=(pid,)) for pid in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for pid in ids:
        mesh = store.get_prop("P", pid)["mesh"]
        assert all(mesh[f"k{n}"] == n for n in range(15))
    assert not list((tmp_path / "projects" / "P").glob("*.tmp"))


# ---- procman --------------------------------------------------------------------------------------

def test_last_words_quote_only_what_went_wrong_in_this_run(tmp_path: Path) -> None:
    tp = procman_mod.ToolProcess(FakeAdapter("pixal3d", "mesh", True, tmp_path))
    for line in ("==== 2026-09-23 19:00:00 starting Pixal3D on port 7772 ====", "RuntimeError: from the last run",
                 "==== 2026-09-23 19:10:00 starting Pixal3D on port 7772 ====",
                 "[spellcast] Starting Pixal3D from C:\\x (low VRAM on). Loading models...",
                 "Traceback (most recent call last):", '  File "app.py", line 199, in init_models',
                 "    ^^^^^^^^^^^^", "cv2.error: OpenCV(4.12.0) OpenEXR codec is disabled",
                 "[spellcast] Pixal3D could not start (the error is above)."):
        tp.add_line(line)
    assert tp.last_words() == "cv2.error: OpenCV(4.12.0) OpenEXR codec is disabled | Pixal3D could not start (the error is above)."
    # killed mid-run: only progress output and warnings after the start -> nothing worth quoting
    tp.add_line("==== 2026-09-23 19:20:00 starting Pixal3D on port 7772 ====")
    tp.add_line("[spellcast] Pixal3D models loaded.")
    tp.add_line("module.py:1355: UserWarning: expandable_segments not supported on this platform")
    tp.add_line("Sampling texture SLat (proj):  50%|#####     | 6/12")
    assert tp.last_words() == ""


def test_start_ready_log_and_graceful_stop(tmp_path: Path) -> None:
    async def main() -> None:
        forge = FakeAdapter("forge", "images", True, tmp_path)
        async with manager(tmp_path, {"forge": forge}) as pm:
            assert pm.status("forge")["state"] == "stopped"
            tp = await pm.ensure_running("forge")
            status = pm.status("forge")
            assert status["state"] == "ready" and status["installed"] and status["supported"]
            assert status["ui_url"] == f"http://127.0.0.1:{status['port']}/"
            assert set(status) == {"state", "message", "starting_seconds", "port", "ui_url", "display_name", "stage",
                                   "uses_gpu", "gpu_at_start", "holds_gpu", "installed", "supported", "reason",
                                   "license"}
            assert status["starting_seconds"] is None                       # said only while it starts
            assert status["license"] is None and status["gpu_at_start"] and status["holds_gpu"]
            pid = (await pm.client.get(f"{tp.base_url}/echo")).json()["pid"]
            lines = pm.log_tail("forge", 100)
            assert "loading 100%" in lines and "loading 10%" not in lines      # \r progress collapsed
            assert "stderr line" in lines and "ready \u2713" in lines
            assert pm.log_tail("forge", 2) == lines[-2:]
            pm.set_busy("forge", True)
            assert pm.status("forge")["state"] == "busy"
            pm.set_busy("forge", False)

            await pm.stop("forge")
            assert pm.status("forge")["state"] == "stopped" and pm.status("forge")["port"] is None
            assert not pid_alive(pid)
            assert forge.on_stop_calls == 1
            log_file = tmp_path / "logs" / "forge.log"
            if pm.console:
                wait_until(lambda: "got CTRL_BREAK" in log_file.read_text(encoding="utf-8"), 5, "graceful exit")
            assert "loading 10%" not in log_file.read_text(encoding="utf-8")

    asyncio.run(main())


@pytest.mark.parametrize("console", [True, False])
def test_stop_kills_a_tool_that_ignores_ctrl_break(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   console: bool) -> None:
    monkeypatch.setattr(procman_mod, "BREAK_WAIT", 1.0)
    pidfile = tmp_path / "pid.txt"

    async def main() -> None:
        forge = FakeAdapter("forge", "images", True, tmp_path, args=("--ignore-break", "--pidfile", str(pidfile)))
        async with manager(tmp_path, {"forge": forge}) as pm:
            pm.console = console and pm.console
            await pm.ensure_running("forge")
            pid = int(pidfile.read_text())
            started = time.monotonic()
            await pm.stop("forge")
            assert time.monotonic() - started < 10
            assert pm.status("forge")["state"] == "stopped"
            assert not pid_alive(pid)

    asyncio.run(main())


def test_crash_after_ready_becomes_error_then_restarts(tmp_path: Path) -> None:
    async def main() -> None:
        forge = FakeAdapter("forge", "images", True, tmp_path, args=("--crash-after", "0.5"))
        async with manager(tmp_path, {"forge": forge}) as pm:
            await pm.ensure_running("forge")
            await wait_state(pm, "forge", {"error"})
            message = pm.status("forge")["message"]
            assert "stopped unexpectedly (exit code 3)" in message and "CUDA out of memory" in message
            assert pm.status("forge")["ui_url"] is None
            forge.args = []
            await pm.ensure_running("forge")                     # Start again after an error works
            assert pm.status("forge")["state"] == "ready"

    asyncio.run(main())


def test_exit_during_start_and_ready_timeout(tmp_path: Path) -> None:
    pidfile = tmp_path / "pid.txt"

    async def main() -> None:
        forge = FakeAdapter("forge", "images", True, tmp_path, args=("--exit-now",))
        pixal = FakeAdapter("pixal3d", "mesh", True, tmp_path, ready_timeout=1.5,
                            args=("--never-healthy", "--pidfile", str(pidfile)))
        async with manager(tmp_path, {"forge": forge, "pixal3d": pixal}) as pm:
            with pytest.raises(ToolError, match="stopped while starting") as e:
                await pm.ensure_running("forge")
            assert "exit code 2" in str(e.value) and "model file missing" in str(e.value)
            assert pm.status("forge")["state"] == "error"

            with pytest.raises(ToolError, match="did not become ready"):
                await pm.ensure_running("pixal3d")
            assert pm.status("pixal3d")["state"] == "error"
            assert not pid_alive(int(pidfile.read_text()))

    asyncio.run(main())


def test_restart_while_stopping_and_stop_while_starting(tmp_path: Path) -> None:
    async def main() -> None:
        forge = FakeAdapter("forge", "images", True, tmp_path)
        async with manager(tmp_path, {"forge": forge}) as pm:
            first = await pm.ensure_running("forge")
            old_proc = first.proc
            pm.request_stop("forge")                        # the Stop button...
            assert pm.status("forge")["state"] == "stopping"
            await pm.ensure_running("forge")                # ...then Start before the stop finished
            assert pm.status("forge")["state"] == "ready" and pm.get("forge").proc is not old_proc
            assert not any("stopped while starting" in line for line in pm.log_tail("forge", 500))

            await pm.stop("forge")
            forge.args = ["--delay", "5"]
            waiter = asyncio.create_task(pm.ensure_running("forge"))
            await asyncio.sleep(0.5)
            assert pm.status("forge")["state"] == "starting"
            await pm.stop("forge")
            with pytest.raises(ToolError, match="stopped before it finished starting"):
                await waiter
            assert pm.status("forge")["state"] == "stopped"

    asyncio.run(main())


def test_gpu_exclusivity(tmp_path: Path) -> None:
    async def main(exclusive: bool) -> None:
        async with manager(tmp_path, fake_adapters(tmp_path), {"gpu_exclusive": exclusive}) as pm:
            await asyncio.gather(pm.ensure_running("forge"), pm.ensure_running("imesh"))
            forge_pid = pm.get("forge").proc.pid
            await pm.ensure_running("pixal3d")
            states = {t: pm.status(t)["state"] for t in ("forge", "pixal3d", "imesh")}
            if exclusive:
                assert states == {"forge": "stopped", "pixal3d": "ready", "imesh": "ready"}
                assert not pid_alive(forge_pid)
                pm.set_busy("pixal3d", True)
                with pytest.raises(ToolError, match="busy"):
                    pm.start("forge")
                pm.set_busy("pixal3d", False)
            else:
                assert states == {"forge": "ready", "pixal3d": "ready", "imesh": "ready"}

    asyncio.run(main(True))
    asyncio.run(main(False))


def test_a_tool_that_takes_the_card_at_its_first_job_starts_early_beside_the_one_on_it(tmp_path: Path) -> None:
    """Pixal3D (gpu_at_start False) starts beside the image generator, even a busy one, and takes the card only when a
    job claims it; starting a tool that loads its models at start stops the tool that holds the card, never one that
    is running without it."""
    async def main() -> None:
        adapters = fake_adapters(tmp_path, pixal3d=FakeAdapter("pixal3d", "mesh", True, tmp_path, gpu_at_start=False))
        async with manager(tmp_path, adapters) as pm:
            await pm.ensure_running("forge")
            forge_pid = pm.get("forge").proc.pid
            pm.set_busy("forge", True)
            assert pm.warm("pixal3d") is True                  # (a) beside the busy one: nothing is stopped
            await pm.ensure_running("pixal3d")
            assert pm.status("forge")["state"] == "busy" and pid_alive(forge_pid)
            assert pm.status("forge")["holds_gpu"] and not pm.status("pixal3d")["holds_gpu"]
            assert not pm.status("pixal3d")["gpu_at_start"] and pm.status("forge")["gpu_at_start"]
            assert pm.warm("pixal3d") is False                 # running already
            pm.set_busy("forge", False)

            await pm.claim_gpu("pixal3d")                      # (b) its job takes the card: the other tool goes
            assert pm.status("forge")["state"] == "stopped" and not pid_alive(forge_pid)
            assert pm.status("pixal3d")["holds_gpu"]
            pixal_pid = pm.get("pixal3d").proc.pid
            assert (await pm.claim_gpu("pixal3d")).proc.pid == pixal_pid and pm.status("pixal3d")["state"] == "ready"
            await pm.claim_gpu("imesh")                        # a CPU tool takes nothing
            assert not pm.status("imesh")["holds_gpu"]
            assert pm.warm("forge") is False                   # the card is taken: no early start of a tool that loads
            pm.set_busy("pixal3d", True)
            with pytest.raises(ToolError, match="busy"):
                pm.start("forge")
            pm.set_busy("pixal3d", False)

            await pm.ensure_running("forge")                   # (c) a tool that holds the card is stopped ...
            assert pm.status("pixal3d")["state"] == "stopped" and not pid_alive(pixal_pid)
            forge_pid = pm.get("forge").proc.pid
            await pm.ensure_running("pixal3d")                 # ... started again it holds nothing ...
            pixal_pid = pm.get("pixal3d").proc.pid
            assert not pm.status("pixal3d")["holds_gpu"] and pid_alive(forge_pid)
            await pm.stop("forge")
            await pm.ensure_running("forge")                   # ... so a start that loads leaves it running
            assert pm.status("pixal3d")["state"] == "ready" and pm.get("pixal3d").proc.pid == pixal_pid
            await pm.stop("forge")
            assert pm.warm("forge") is True                    # nothing on the card: the early start may load
            await wait_state(pm, "forge", {"ready"})

            pm.get("pixal3d").proc.kill()                      # a tool that stopped meanwhile cannot take the card
            await wait_state(pm, "pixal3d", {"error"})
            with pytest.raises(ToolError, match="stopped"):
                await pm.claim_gpu("pixal3d")
            assert pm.status("forge")["state"] == "ready"      # ... and the tool on the card stays
            assert pm.warm("pixal3d") is False                 # in error: a job or Settings > Debugging starts it

    async def not_exclusive() -> None:
        adapters = fake_adapters(tmp_path, pixal3d=FakeAdapter("pixal3d", "mesh", True, tmp_path, gpu_at_start=False))
        async with manager(tmp_path, adapters, {"gpu_exclusive": False}) as pm:
            await asyncio.gather(pm.ensure_running("forge"), pm.ensure_running("pixal3d"))
            await pm.claim_gpu("pixal3d")
            assert {t: pm.status(t)["state"] for t in ("forge", "pixal3d")} == {"forge": "ready", "pixal3d": "ready"}

    asyncio.run(main())
    asyncio.run(not_exclusive())


def test_concurrent_ensure_running_spawns_once_and_port_fallback(tmp_path: Path) -> None:
    async def main() -> None:
        async with manager(tmp_path, {"imesh": FakeAdapter("imesh", "retopo", False, tmp_path)}) as pm:
            wanted = pm.settings.get()["ports"]["imesh"]
            with socket.socket() as blocker:
                blocker.bind(("0.0.0.0", wanted))
                blocker.listen()
                tps = await asyncio.gather(*(pm.ensure_running("imesh") for _ in range(3)))
            assert tps[0] is tps[1] is tps[2]
            assert pm.status("imesh")["port"] != wanted
            assert sum("starting Fake imesh" in line for line in pm.log_tail("imesh", 500)) == 1

    asyncio.run(main())


def test_not_installed_and_unsupported(tmp_path: Path) -> None:
    async def main() -> None:
        adapters = fake_adapters(
            tmp_path,
            forge=FakeAdapter("forge", "images", True, tmp_path, install="not_installed"),
            pixal3d=FakeAdapter("pixal3d", "mesh", True, tmp_path, supported=(False, "GTX 1080 is too old.")))
        async with manager(tmp_path, adapters) as pm:
            forge, pixal = pm.status("forge"), pm.status("pixal3d")
            assert forge["state"] == "not_installed" and not forge["installed"] and "run-spellcast3d.bat" in forge["message"]
            assert pixal["state"] == "unsupported" and not pixal["supported"] and pixal["message"] == "GTX 1080 is too old."
            with pytest.raises(ToolError, match="not installed"):
                pm.start("forge")
            with pytest.raises(ToolError, match="too old"):
                await pm.ensure_running("pixal3d")
            with pytest.raises(ToolError) as e:
                pm.start("nope")
            assert e.value.status_code == 404

    asyncio.run(main())


HUB_SCRIPT = r'''
import os, subprocess, sys, threading, time
sys.path.insert(0, {code!r})
from spellcast.procman import spawn_in_job
from spellcast import paths
proc = spawn_in_job([sys.executable, {fake!r}, "--port", "{port}", "--pidfile", {pid!r}, "--child-pidfile", {child!r}],
                    {cwd!r}, paths.clean_env(None), subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
threading.Thread(target=proc.stdout.read, daemon=True).start()   # keep the tool's output pipe open and drained
print(os.getpid(), flush=True)
time.sleep(120)
'''


def test_job_object_kills_tools_when_the_hub_dies(tmp_path: Path) -> None:
    """Hard-kill a 'hub' process: its tool AND the tool's own child must die with it."""
    pidfile, childfile = tmp_path / "tool.pid", tmp_path / "child.pid"
    script = tmp_path / "hub.py"
    script.write_text(HUB_SCRIPT.format(code=str(FAKE_TOOL.parents[1]), fake=str(FAKE_TOOL), port=free_ports(1)[0],
                                        pid=str(pidfile), child=str(childfile), cwd=str(tmp_path)))
    hub = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        hub_pid = int(hub.stdout.readline())               # the real interpreter, not the venv launcher
        wait_until(lambda: pidfile.exists() and childfile.exists() and childfile.read_text(), 15, "tool pids")
        tool_pid, child_pid = int(pidfile.read_text()), int(childfile.read_text())
        assert pid_alive(tool_pid) and pid_alive(child_pid)
        os.kill(hub_pid, 9)                                 # TerminateProcess: no cleanup code runs
        wait_until(lambda: not pid_alive(tool_pid) and not pid_alive(child_pid), 10, "tools to die with the hub")
    finally:
        hub.kill()
        hub.wait()


def test_tools_inherit_the_quiet_hard_error_mode_and_their_exit_codes_are_read(tmp_path: Path) -> None:
    """A tool that cannot start ends with its exit code in the hub's log instead of standing behind a Windows dialog:
    spawn_in_job hands the children SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX."""
    from spellcast import procman
    proc = procman.spawn_in_job([sys.executable, "-c", "import ctypes; print(ctypes.windll.kernel32.GetErrorMode())"],
                                tmp_path, paths.clean_env(None), subprocess.CREATE_NO_WINDOW)
    out, _ = proc.communicate(timeout=30)
    assert int(out.split()[-1]) & 0x8003 == 0x8003
    assert procman._exit_code_text(3) == "3"
    assert procman._exit_code_text(-1073741819) == "0xc0000005"
    assert procman._exit_code_text(0xC0000142).startswith("0xc0000142: Windows could not initialise")
    assert procman._exit_code_text(-1073741502) == procman._exit_code_text(0xC0000142)


# ---- jobs --------------------------------------------------------------------------------------

def _project_with_props(tmp_path: Path, n: int) -> tuple[ProjectStore, list[str]]:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("Demo")
    return store, [store.add_prop("Demo", f"Prop {i}")["id"] for i in range(n)]


async def _wait_job(queue: JobQueue, job_id: str, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = queue.get(job_id)
        if job["status"] in ("done", "error", "cancelled"):
            return job
        await asyncio.sleep(0.05)
    raise AssertionError(f"job stuck: {queue.get(job_id)}")


def _states(job: dict[str, Any]) -> list[str]:
    """Where each prop of the job is (job.props), in the order of its prop_ids."""
    assert list(job["props"]) == job["prop_ids"]
    return [job["props"][prop_id]["state"] for prop_id in job["prop_ids"]]


def test_a_job_that_runs_prop_by_prop_says_where_each_prop_is() -> None:
    """Job.prop_states of a job whose tool makes one prop after another: read from the job's counters. The page
    reads these states, never the counters."""
    def job(**fields: Any) -> Job:
        return Job(id="j_1", stage="images", tool="forge", project="P", prop_ids=["a", "b", "c", "d"], params={},
                   **fields)

    def states(**fields: Any) -> list[str]:
        return _states(job(**fields).to_dict())

    assert states() == states(status="starting_tool") == ["waiting"] * 4
    assert states(status="running") == ["working", "waiting", "waiting", "waiting"]
    assert states(status="running", done=2, errors={"b": "x"}) == ["done", "failed", "working", "waiting"]
    assert states(status="done", done=4) == ["done"] * 4
    assert states(status="error", done=4, errors={"a": "x", "d": "y"}) == ["failed", "done", "done", "failed"]
    assert states(status="error", errors=dict.fromkeys("abcd", "no tool")) == ["failed"] * 4    # it could not start
    assert states(status="error", done=1) == ["done", "cancelled", "cancelled", "cancelled"]   # the hub's own crash
    assert states(status="cancelled") == ["cancelled"] * 4                  # cancelled while it waited
    # A cancel cut the second prop short (it counts in progress.done, and was not made); the rest never began.
    assert states(status="cancelled", done=2, cut_short=1, errors={"a": "x"}) == ["failed"] + ["cancelled"] * 3
    assert states(status="cancelled", done=2) == ["done", "done", "cancelled", "cancelled"]     # saved in full
    told = job(status="running", done=1, prop_fraction=0.25, message="step 3").to_dict()
    assert told["part"] is None and told["props"] == {
        "a": {"state": "done", "fraction": 1.0, "message": ""},
        "b": {"state": "working", "fraction": 0.25, "message": "step 3"},     # the job's own progress in the prop
        "c": {"state": "waiting", "fraction": 0.0, "message": ""},
        "d": {"state": "waiting", "fraction": 0.0, "message": ""}}
    # A job whose tool takes all props at once says what its adapter reported (and never "resting" otherwise).
    reported = {"a": {"state": "resting", "fraction": 0.3, "message": "Waiting for the next step"},
                "b": {"state": "working", "fraction": 0.16, "message": "Sculpting the shape (1/12)"}}
    batch = job(status="running", props=reported, part={"key": "lr", "label": "Sculpting the shape", "position": 2,
                                                        "of": 2}).to_dict()
    assert batch["props"] == reported and batch["part"]["position"] == 2
    # The page reads what a tool is loading, and whether a job is being stopped, from these two keys and never
    # from the job's words. A job that runs prop by prop loads nothing of its own.
    assert batch["loading"] == "" and batch["stopping"] is False
    assert job(status="running", message="Loading the shape model…").to_dict()["loading"] == ""
    assert job(status="running", loading="texture model").to_dict()["loading"] == "texture model"
    assert job(status="running", cancel_requested=True).to_dict()["stopping"] is True
    assert job(status="cancelled", cancel_requested=True).to_dict()["stopping"] is False


def test_job_flow_errors_and_busy(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    store, ids = _project_with_props(tmp_path, 4)
    store.patch_prop("Demo", ids[0], {"images": {"settings": {"steps": 7}}})
    store.patch_prop("Demo", ids[1], {"images": {"settings": {"mode": "stage_error"}}})
    store.patch_prop("Demo", ids[2], {"images": {"settings": {"mode": "crash"}}})

    async def main() -> None:
        forge = FakeAdapter("forge", "images", True, tmp_path)
        async with manager(tmp_path, {"forge": forge}) as pm:
            queue = JobQueue(pm, store, pm.settings, {"forge": forge}, pm.client)
            queue.start()
            job = await queue.submit("images", "Demo", [ids[0], ids[0], ids[3]], {"cfg_scale": 3})
            assert job["status"] == "queued" and job["prop_ids"] == [ids[0], ids[3]]
            done = await _wait_job(queue, job["id"])
            assert done["status"] == "done" and done["errors"] == {} and done["finished"]
            assert done["progress"] == {"done": 2, "total": 2, "fraction": 1.0, "message": "Done."}
            assert done["props"] == {p: {"state": "done", "fraction": 1.0, "message": ""} for p in (ids[0], ids[3])}
            assert done["part"] is None
            assert forge.calls[0] == (ids[0], {**store.get_prop("Demo", ids[0])["images"]["settings"], "cfg_scale": 3})
            assert forge.calls[0][1]["steps"] == 7
            prop = store.get_prop("Demo", ids[0])
            assert prop["images"]["result"] == "images/out.txt"
            assert prop["images"]["tool_pid"] == (await pm.client.get(f"{pm.get('forge').base_url}/echo")).json()["pid"]

            failed = await _wait_job(queue, (await queue.submit("images", "Demo", ids[1:4]))["id"])
            assert failed["status"] == "error"
            assert failed["errors"][ids[1]] == "Pick an image for Prop 1 first."
            assert "RuntimeError: boom" in failed["errors"][ids[2]] and "hub.log" in failed["errors"][ids[2]]
            assert ids[3] not in failed["errors"] and _states(failed) == ["failed", "failed", "done"]
            assert "boom" in caplog.text                         # full traceback logged

            slow = await queue.submit("images", "Demo", [ids[3]], {"mode": "slow"})
            await asyncio.sleep(0.5)
            running = queue.get(slow["id"])
            assert running["status"] == "running" and running["current_prop"] == ids[3]
            assert 0 < running["progress"]["fraction"] < 1 and running["progress"]["message"].startswith("step")
            assert running["props"] == {ids[3]: {"state": "working", "fraction": running["progress"]["fraction"],
                                                 "message": running["progress"]["message"]}}
            assert pm.status("forge")["state"] == "busy"
            await _wait_job(queue, slow["id"])
            assert pm.status("forge")["state"] == "ready"
            assert [j["id"] for j in queue.list("Demo")][:3] == [slow["id"], failed["id"], job["id"]]
            assert queue.list("Other") == []
            await queue.close()

    asyncio.run(main())


def test_job_cancel(tmp_path: Path) -> None:
    store, ids = _project_with_props(tmp_path, 3)

    async def main() -> None:
        forge = FakeAdapter("forge", "images", True, tmp_path)
        async with manager(tmp_path, {"forge": forge}) as pm:
            queue = JobQueue(pm, store, pm.settings, {"forge": forge}, pm.client)
            queue.start()
            running = await queue.submit("images", "Demo", ids, {"mode": "slow"})
            waiting = await queue.submit("images", "Demo", ids[:1])
            cancelled = queue.cancel(waiting["id"])
            assert cancelled["status"] == "cancelled" and cancelled["finished"] and _states(cancelled) == ["cancelled"]
            while queue.get(running["id"])["status"] != "running":
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.3)
            assert _states(queue.get(running["id"])) == ["working", "waiting", "waiting"]
            stopping = queue.cancel(running["id"])
            # A tool that makes the props one after another stops inside its prop (a 3D job's words differ:
            # it finishes a model it is saving, test_mesh_batch).
            assert stopping["stopping"] is True and stopping["progress"]["message"] == CANCELLING
            final = await _wait_job(queue, running["id"])
            assert final["status"] == "cancelled" and final["errors"] == {} and final["stopping"] is False
            assert final["progress"]["done"] == 1                  # stopped inside the first prop
            # ...which is not reported as finished
            assert final["progress"]["message"].startswith("Cancelled after 0 of 3 props.")
            assert _states(final) == ["cancelled"] * 3              # ...in the props' own states either
            assert len(forge.calls) == 1
            assert queue.cancel(running["id"])["status"] == "cancelled"
            with pytest.raises(JobError) as e:
                queue.get("j_missing")
            assert e.value.status_code == 404
            await queue.close()

    asyncio.run(main())


def test_job_when_the_tool_dies_mid_batch(tmp_path: Path) -> None:
    store, ids = _project_with_props(tmp_path, 3)

    async def main() -> None:
        forge = FakeAdapter("forge", "images", True, tmp_path, args=("--crash-after", "1"))
        async with manager(tmp_path, {"forge": forge}) as pm:
            queue = JobQueue(pm, store, pm.settings, {"forge": forge}, pm.client)
            queue.start()
            final = await _wait_job(queue, (await queue.submit("images", "Demo", ids, {"mode": "slow"}))["id"])
            assert final["status"] == "error" and final["progress"]["done"] == 3
            assert "stopped while working on this prop" in final["errors"][ids[0]]
            # started again once for the next prop (here it dies again), then not again in this job
            assert "stopped while working on this prop" in final["errors"][ids[1]]
            assert "stopped before this prop's turn" in final["errors"][ids[2]]
            assert len(forge.calls) == 2 and _states(final) == ["failed"] * 3
            assert pm.status("forge")["state"] == "error"
            await queue.close()

    asyncio.run(main())


def test_a_tool_that_must_start_afresh_is_restarted_for_the_next_prop(tmp_path: Path) -> None:
    """ToolNeedsRestart (its GPU context broke, or it still works on a prop the hub gave up on): that prop fails, the
    tool starts again, and the job goes on with the rest of its props."""
    store, ids = _project_with_props(tmp_path, 2)

    async def main() -> None:
        tool = FakeAdapter("forge", "images", True, tmp_path)
        async with manager(tmp_path, {"forge": tool}) as pm:
            queue = JobQueue(pm, store, pm.settings, {"forge": tool}, pm.client)
            queue.start()
            first_pid = (await pm.ensure_running("forge")).proc.pid
            final = await _wait_job(queue, (await queue.submit("images", "Demo", ids, {"mode": "restart_once"}))["id"])
            assert final["status"] == "error" and list(final["errors"]) == [ids[0]]
            assert final["errors"][ids[0]] == "Fake forge has to start again after an error."
            made = store.get_prop("Demo", ids[1])["images"]
            assert made["result"] == "images/out.txt" and made["tool_pid"] != first_pid     # a fresh process
            assert pm.status("forge")["state"] == "ready" and len(tool.calls) == 2
            # The last prop of a job needs the restart: the tool is stopped, the next job starts it afresh.
            tool.calls.clear()
            final = await _wait_job(queue, (await queue.submit("images", "Demo", ids[:1],
                                                               {"mode": "restart_once"}))["id"])
            assert final["status"] == "error" and pm.status("forge")["state"] == "stopped"
            final = await _wait_job(queue, (await queue.submit("images", "Demo", ids[:1]))["id"])
            assert final["status"] == "done"
            await queue.close()

    asyncio.run(main())


def test_a_prop_saved_in_full_after_a_cancel_counts_as_finished(tmp_path: Path) -> None:
    """A tool that can't stop in the middle of a prop (TriFlow) finishes and saves it after a cancel: the job
    counts it as finished, not as cut short."""
    store, ids = _project_with_props(tmp_path, 3)

    async def main() -> None:
        tool = FakeAdapter("forge", "images", True, tmp_path)
        async with manager(tmp_path, {"forge": tool}) as pm:
            queue = JobQueue(pm, store, pm.settings, {"forge": tool}, pm.client)
            queue.start()
            job = await queue.submit("images", "Demo", ids, {"mode": "finish_after_cancel"})
            while queue.get(job["id"])["status"] != "running":
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.3)
            queue.cancel(job["id"])
            final = await _wait_job(queue, job["id"])
            assert final["status"] == "cancelled" and final["progress"]["done"] == 1
            assert final["progress"]["message"].startswith("Cancelled after 1 of 3 props.")
            assert _states(final) == ["done", "cancelled", "cancelled"]
            assert store.get_prop("Demo", ids[0])["images"]["result"] == "images/out.txt"
            await queue.close()

    asyncio.run(main())


def test_job_validation_tool_failure_and_history(tmp_path: Path) -> None:
    store, ids = _project_with_props(tmp_path, 2)

    async def main() -> None:
        broken = FakeAdapter("forge", "images", True, tmp_path, args=("--exit-now",))
        adapters = {"forge": broken, "imesh": FakeAdapter("imesh", "retopo", False, tmp_path, install="failed")}
        async with manager(tmp_path, adapters) as pm:
            queue = JobQueue(pm, store, pm.settings, adapters, pm.client)
            for args, status in ((("rig", "Demo", ids), 400), (("images", "Demo", []), 400),
                                 (("images", "Nope", ids), 404), (("images", "Demo", ["p_ffffff"]), 404),
                                 (("images", "Demo", ids, "x"), 400), (("mesh", "Demo", ids), 400)):
                with pytest.raises(JobError) as e:
                    await queue.submit(*args)
                assert e.value.status_code == status, args
            with pytest.raises(ToolError, match="not installed"):
                await queue.submit("retopo", "Demo", ids, {"method": "imesh"})

            for _ in range(55):
                queue.cancel((await queue.submit("images", "Demo", ids))["id"])
            queue.start()
            job = await queue.submit("images", "Demo", ids)
            assert len(queue.list()) == 50 and queue.list()[0]["id"] == job["id"]
            final = await _wait_job(queue, job["id"])
            assert final["status"] == "error" and "stopped while starting" in final["progress"]["message"]
            assert set(final["errors"]) == set(ids) and _states(final) == ["failed", "failed"]
            await queue.close()

    asyncio.run(main())


# ---- HTTP API -------------------------------------------------------------------------------------

@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    config_file = tmp_path / "spellcast3d-config.ini"       # the template's choices, never the Spellcast3D folder's
    config_file.write_bytes(paths.CONFIG_TEMPLATE.read_bytes())
    monkeypatch.setenv("SPELLCAST_CONFIG", str(config_file))
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>Spellcast3D</h1>", encoding="utf-8")
    (static / "app.js").write_text("export {};", encoding="utf-8")
    adapters = fake_adapters(tmp_path)
    app = create_app(adapters, settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=static, logs_dir=tmp_path / "logs", prompt_library=tmp_path / "prompt-library.json",
                     gpu_probe=lambda: GpuInfo(0, "Fake RTX 3060", 8.6, 12288, 3000, "580.10"))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield app, client, adapters


def test_the_prompt_library_is_one_file_for_every_project(api: Any, tmp_path: Path) -> None:
    """GET / PUT /api/prompt-library: the whole list, checked; names cut to 10 characters; an unreadable file is set
    aside instead of being overwritten by the next save."""
    from spellcast import promptlib

    _app, client, _adapters = api
    defaults = [dict(d) for d in promptlib.DEFAULTS]
    assert [(d["name"], d["text"], d["shown"]) for d in defaults] == [
        ("4 views", "shown in an orthographic sheet of four quadrants: front, back, left, right.", True),
        ("T 4 views", "standing in a t-pose, shown in an orthographic sheet of four quadrants: front, back, left view, "
                      "right view.\nAt the top row the arms are spread to the sides.", True),
        ("toon", "a stylized hand-painted cartoon style", True)]
    assert all(len(d["name"]) <= promptlib.MAX_NAME for d in defaults), "a default's chip is not cut short"
    assert client.get("/api/prompt-library").json() == {"prompts": defaults}, "a new library starts with the defaults"
    sent = [{"id": "a1", "name": "  hand-painted  ", "text": "hand-painted style, soft light", "shown": True},
            {"id": "b2", "name": "", "text": "", "shown": "yes"}]
    saved = client.put("/api/prompt-library", json={"prompts": sent}).json()["prompts"]
    assert saved == [{"id": "a1", "name": "hand-paint", "text": "hand-painted style, soft light", "shown": True},
                     {"id": "b2", "name": "", "text": "", "shown": False}]
    assert client.get("/api/prompt-library").json()["prompts"] == saved
    offered = [d["id"] for d in defaults]
    assert json.loads((tmp_path / "prompt-library.json").read_text(encoding="utf-8")) == {
        "prompts": saved, "offered": offered, "placed": ["tpose4"]}
    assert not any(p["id"] in offered for p in saved), "a default the artist removed stays removed"
    for bad in ([{"id": "a1"}, {"id": "a1"}], [{"id": "../x"}], [{"id": "c", "text": 5}], "nope"):
        assert client.put("/api/prompt-library", json={"prompts": bad}).status_code == 400
    assert client.get("/api/prompt-library").json()["prompts"] == saved, "a refused save changes nothing"
    # A library from before the defaults (no "offered"): they are added at its end, once.
    (tmp_path / "prompt-library.json").write_text(json.dumps({"prompts": saved}), encoding="utf-8")
    assert client.get("/api/prompt-library").json()["prompts"] == [*saved, *defaults]
    assert client.put("/api/prompt-library", json={"prompts": saved}).json()["prompts"] == saved
    assert client.get("/api/prompt-library").json()["prompts"] == saved
    # A library that was offered some of them (an earlier version's): only the new ones come.
    (tmp_path / "prompt-library.json").write_text(json.dumps({"prompts": saved, "offered": ["views4"]}), encoding="utf-8")
    assert client.get("/api/prompt-library").json()["prompts"] == [*saved, *defaults[1:]]
    (tmp_path / "prompt-library.json").write_text(json.dumps({"prompts": saved, "offered": ["views4", "toon"]}),
                                                  encoding="utf-8")
    assert client.get("/api/prompt-library").json()["prompts"] == [*saved, defaults[1]], "the third reaches a library of two"
    # "T 4 views" stands right after "4 views": where it arrives in a library that has that one ...
    views4, tpose4, toon = defaults
    library = tmp_path / "prompt-library.json"
    library.write_text(json.dumps({"prompts": [saved[0], views4, toon, saved[1]], "offered": ["views4", "toon"]}),
                       encoding="utf-8")
    assert client.get("/api/prompt-library").json()["prompts"] == [saved[0], views4, tpose4, toon, saved[1]]
    # ... and where a library that got it at its end (before it had a place) has it moved to, once
    library.write_text(json.dumps({"prompts": [saved[0], views4, toon, tpose4], "offered": offered}), encoding="utf-8")
    assert client.get("/api/prompt-library").json()["prompts"] == [saved[0], views4, tpose4, toon]
    assert json.loads(library.read_text(encoding="utf-8"))["placed"] == ["tpose4"]
    library.write_text(json.dumps({"prompts": [views4, toon, tpose4], "offered": offered, "placed": ["tpose4"]}),
                       encoding="utf-8")
    assert client.get("/api/prompt-library").json()["prompts"] == [views4, toon, tpose4], "placed once: left where it is"
    library.write_text(json.dumps({"prompts": [views4, saved[0], tpose4], "offered": ["views4", "tpose4"],
                                   "placed": ["tpose4"]}), encoding="utf-8")
    assert client.get("/api/prompt-library").json()["prompts"] == [views4, saved[0], tpose4, toon], \
        "also when another default arrives later"
    (tmp_path / "prompt-library.json").write_text("{broken", encoding="utf-8")
    assert client.get("/api/prompt-library").json() == {"prompts": defaults}
    assert (tmp_path / "prompt-library.unreadable.json").read_text(encoding="utf-8") == "{broken"


def _poll(client: TestClient, url: str, key: Any, want: set[str], timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = client.get(url).json()
        if key(data) in want:
            return data
        time.sleep(0.1)
    raise AssertionError(f"{url} never reached {want}: {data}")


def test_api_status_ui_and_guards(api: Any, tmp_path: Path) -> None:
    app, client, _ = api
    status = client.get("/api/status").json()
    assert status["version"] and status["gpu"]["name"] == "Fake RTX 3060" and status["gpu"]["sm"] == "sm_86"
    assert status["settings_warnings"] == []
    assert set(status["tools"]) == {"forge", "pixal3d", "imesh"}
    assert status["tools"]["forge"]["state"] == "stopped" and status["tools"]["forge"]["stage"] == "images"
    # The tools that run each stage here (the test's own adapters), and the config file the hub started with.
    assert status["stage_tools"] == {"images": ["forge"], "mesh": ["pixal3d"], "omnitexture": [], "retopo": ["imesh"],
                                     "projection": [], "rig": []}
    assert status["config"] == {"image_backend": "qwen", "quality": "Q4_K_M", "problems": [], "changed": None}

    page = client.get("/")
    assert page.status_code == 200 and "Spellcast3D" in page.text and page.headers["cache-control"] == "no-store"
    js = client.get("/static/app.js")
    assert js.status_code == 200 and js.headers["cache-control"] == "no-store"

    assert client.get("/api/status", headers={"host": "evil.example:7770"}).status_code == 403
    assert client.get("/api/status", headers={"origin": "http://evil.example"}).status_code == 403
    assert client.get("/api/status", headers={"host": "localhost:7770", "origin": "http://localhost:7770"}).status_code == 200
    assert client.get("/api/status", headers={"host": "[::1]:7770"}).status_code == 200
    # Requests of other web pages (an <img> or <script> sends no Origin, but browsers send Sec-Fetch-Site).
    assert client.get("/api/status", headers={"sec-fetch-site": "cross-site"}).status_code == 403
    assert client.get("/api/status", headers={"sec-fetch-site": "same-site"}).status_code == 403
    assert client.get("/api/status", headers={"sec-fetch-site": "same-origin"}).status_code == 200
    assert client.get("/api/status", headers={"sec-fetch-site": "none"}).status_code == 200
    opened_from_link = {"sec-fetch-site": "cross-site", "sec-fetch-mode": "navigate", "sec-fetch-dest": "document"}
    assert client.get("/", headers=opened_from_link).status_code == 200
    assert client.get("/", headers={**opened_from_link, "sec-fetch-dest": "iframe"}).status_code == 403

    bad = client.post("/api/projects", json=[1, 2])
    assert bad.status_code == 422 and isinstance(bad.json()["detail"], str)

    empty = tmp_path / "empty_static"
    empty.mkdir()
    bare = TestClient(create_app(api[2], settings=make_settings(tmp_path), static_dir=empty,
                                 store=ProjectStore(tmp_path / "p2")), base_url="http://127.0.0.1")
    missing = bare.get("/")
    assert missing.status_code == 500 and "missing" in missing.text


def test_api_projects_files_and_uploads(api: Any) -> None:
    app, client, _ = api
    assert client.get("/api/projects").json() == []
    project = client.post("/api/projects", json={"name": "Dungeon/Props"}).json()
    assert project["name"] == "Dungeon Props"
    assert client.post("/api/projects", json={"name": "Dungeon Props"}).status_code == 409
    assert client.get("/api/projects/Nope").json()["detail"].startswith("There is no project")
    prop = client.post("/api/projects/Dungeon Props/props", json={"name": "Torch"}).json()
    pid = prop["id"]
    assert client.post("/api/projects/Dungeon Props/props").json()["name"] == "Prop 2"
    patched = client.patch(f"/api/projects/Dungeon Props/props/{pid}",
                           json={"images": {"settings": {"prompt": "a torch"}}}).json()
    assert patched["images"]["settings"]["prompt"] == "a torch" and patched["images"]["settings"]["steps"] == 25
    assert client.patch("/api/projects/Dungeon Props", json={"ui": {"stage": "mesh"}}).json()["ui"]["stage"] == "mesh"

    up = client.post(f"/api/projects/Dungeon Props/props/{pid}/images/upload",
                     files={"file": ("torch.png", PNG, "image/png")})
    assert up.status_code == 200 and up.json()["images"]["chosen"] == "images/upload_001.png"
    rejected = client.post(f"/api/projects/Dungeon Props/props/{pid}/images/upload",
                           files={"file": ("evil.png", b"MZ\x90\x00", "image/png")})
    assert rejected.status_code == 415 and "PNG, JPG or WEBP" in rejected.json()["detail"]
    ref = client.post(f"/api/projects/Dungeon Props/props/{pid}/images/references",
                      files=[("files", ("r.jpg", JPG)), ("files", ("s.webp", WEBP))])
    assert ref.json()["images"]["references"] == ["reference/reference_001.jpg", "reference/reference_002.webp"]
    assert client.delete(f"/api/projects/Dungeon Props/props/{pid}/images/references",
                         params={"path": "reference/reference_001.jpg"}).json()["images"]["references"] == \
        ["reference/reference_002.webp"]
    assert client.delete(f"/api/projects/Dungeon Props/props/{pid}/reference").json()["images"]["references"] == []
    assert client.post(f"/api/projects/Dungeon Props/props/{pid}/images/select",
                       json={"path": "images/nope.png"}).status_code == 400

    url = f"/files/Dungeon Props/props/{pid}/images/upload_001.png"
    image = client.get(url)
    assert image.status_code == 200 and image.content == PNG
    assert image.headers["content-type"] == "image/png" and image.headers["cache-control"] == "no-cache"
    assert client.get(url, headers={"if-none-match": image.headers["etag"]}).status_code == 304
    prop_dir = app.state.hub.store.prop_dir("Dungeon Props", pid)
    (prop_dir / "mesh").mkdir()
    (prop_dir / "mesh" / "mesh.glb").write_bytes(b"glTF")
    (prop_dir / "mesh" / "retopo.obj").write_text("v 0 0 0")
    assert client.get(f"/files/Dungeon Props/props/{pid}/mesh/mesh.glb").headers["content-type"] == "model/gltf-binary"
    assert client.get(f"/files/Dungeon Props/props/{pid}/mesh/retopo.obj").headers["content-type"].startswith("text/plain")
    assert client.get("/files/Dungeon Props/%2e%2e/%2e%2e/secret.txt").status_code == 404
    assert client.get("/files/Dungeon Props/..%5C..%5Csecret.txt").status_code == 404
    # UNC / rooted paths are refused before anything touches the file system: resolve() would open an SMB
    # connection to that host first (and hand it the Windows credentials).
    resolved: list[str] = []
    real_resolve = Path.resolve

    def spy(self: Path, *args: Any, **kwargs: Any) -> Path:
        resolved.append(str(self))
        return real_resolve(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "resolve", spy)
        for evil in ("//192.0.2.1/share/x.png", "%5C%5C192.0.2.1%5Cshare%5Cx.png", "/%5C192.0.2.1/share/x.png",
                     "%2F192.0.2.1/share/x.png", "/Windows/win.ini"):
            started = time.monotonic()
            assert client.get(f"/files/Dungeon Props/{evil}").status_code == 404, evil
            assert time.monotonic() - started < 2, evil
    assert not [r for r in resolved if r.startswith("\\\\") or "192.0.2.1" in r], resolved

    deleted = client.delete(f"/api/projects/Dungeon Props/props/{pid}").json()
    assert deleted == {"ok": True, "deleted": deleted["deleted"]} and deleted["deleted"].startswith(pid)
    assert client.post(f"/api/projects/Dungeon Props/deleted/{deleted['deleted']}/restore").json()["id"] == pid
    assert client.post(f"/api/projects/Dungeon Props/deleted/{deleted['deleted']}/restore").status_code == 404
    assert client.delete(f"/api/projects/Dungeon Props/props/{pid}").json()["ok"]
    assert client.get("/api/projects").json()[0]["prop_count"] == 1


def test_api_tools_jobs_and_forge(api: Any) -> None:
    app, client, adapters = api
    assert client.get("/api/forge/models").status_code == 409
    assert client.post("/api/tools/nope/start").status_code == 404
    started = client.post("/api/tools/forge/start").json()
    assert started["state"] == "starting"
    _poll(client, "/api/status", lambda s: s["tools"]["forge"]["state"], {"ready"})
    models = client.get("/api/forge/models").json()
    assert models["checkpoints"][0]["name"] == "a.safetensors" and models["current"] == "a.safetensors"
    assert client.post("/api/forge/refresh").json()["samplers"] == ["Euler"]
    assert any("fake tool starting" in line for line in client.get("/api/tools/forge/log?tail=50").json()["lines"])

    client.post("/api/projects", json={"name": "Demo"})
    pid = client.post("/api/projects/Demo/props", json={"name": "Chest"}).json()["id"]
    job = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [pid]}).json()
    assert set(job) == {"id", "stage", "tool", "project", "prop_ids", "status", "current_prop", "progress", "props",
                        "part", "loading", "stopping", "retouch", "errors", "warnings", "created", "finished"}
    assert job["tool"] == "forge" and job["part"] is None
    assert job["props"] == {pid: {"state": "waiting", "fraction": 0.0, "message": ""}}
    final = _poll(client, f"/api/jobs/{job['id']}", lambda j: j["status"], {"done", "error"})
    assert final["status"] == "done"
    assert client.get("/api/projects/Demo").json()["props"][0]["images"]["result"] == "images/out.txt"
    assert client.get("/api/jobs?project=Demo").json()[0]["id"] == job["id"]
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["status"] == "done"
    assert client.post("/api/jobs", json={"stage": "nope", "project": "Demo", "prop_ids": [pid]}).status_code == 400

    # A mesh job stops Forge first (only one GPU tool at a time).
    mesh_job = client.post("/api/jobs", json={"stage": "mesh", "project": "Demo", "prop_ids": [pid]}).json()
    _poll(client, f"/api/jobs/{mesh_job['id']}", lambda j: j["status"], {"done"})
    tools = client.get("/api/status").json()["tools"]
    assert tools["forge"]["state"] == "stopped" and tools["pixal3d"]["state"] == "ready"

    assert client.post("/api/tools/pixal3d/stop").json()["state"] in ("stopping", "stopped")
    _poll(client, "/api/status", lambda s: s["tools"]["pixal3d"]["state"], {"stopped"})


def test_api_warm_starts_early_without_displacing_and_never_refuses(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /api/tools/{tool}/warm: the stage's tool starts early when that displaces nothing; a tool that cannot
    start is answered with its status and "warmed": false. A mesh job then takes the card for the warm Pixal3D."""
    config_file = tmp_path / "spellcast3d-config.ini"
    config_file.write_bytes(paths.CONFIG_TEMPLATE.read_bytes())
    monkeypatch.setenv("SPELLCAST_CONFIG", str(config_file))
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>Spellcast3D</h1>", encoding="utf-8")
    adapters = fake_adapters(tmp_path, pixal3d=FakeAdapter("pixal3d", "mesh", True, tmp_path, gpu_at_start=False),
                             imesh=FakeAdapter("imesh", "retopo", False, tmp_path, install="not_installed"))
    app = create_app(adapters, settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=static, logs_dir=tmp_path / "logs",
                     gpu_probe=lambda: GpuInfo(0, "Fake RTX 3060", 8.6, 12288, 3000, "580.10"))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        missing = client.post("/api/tools/imesh/warm")             # (d) not installed: no error, no start
        assert missing.status_code == 200
        assert missing.json()["warmed"] is False and missing.json()["state"] == "not_installed"
        assert client.post("/api/tools/nope/warm").status_code == 404

        assert client.post("/api/tools/forge/start").status_code == 200
        _poll(client, "/api/status", lambda s: s["tools"]["forge"]["state"], {"ready"})
        warm = client.post("/api/tools/pixal3d/warm").json()
        assert warm["warmed"] is True and warm["state"] == "starting"     # answered at once, not when ready
        _poll(client, "/api/status", lambda s: s["tools"]["pixal3d"]["state"], {"ready"})
        tools = client.get("/api/status").json()["tools"]
        assert tools["forge"]["state"] == "ready" and tools["forge"]["holds_gpu"] and not tools["pixal3d"]["holds_gpu"]
        assert client.post("/api/tools/pixal3d/warm").json()["warmed"] is False      # running already

        client.post("/api/projects", json={"name": "Demo"})
        pid = client.post("/api/projects/Demo/props", json={"name": "Chest"}).json()["id"]
        job = client.post("/api/jobs", json={"stage": "mesh", "project": "Demo", "prop_ids": [pid]}).json()
        assert _poll(client, f"/api/jobs/{job['id']}", lambda j: j["status"], {"done", "error"})["status"] == "done"
        tools = client.get("/api/status").json()["tools"]
        assert tools["forge"]["state"] == "stopped" and tools["pixal3d"]["holds_gpu"]
        forge = client.post("/api/tools/forge/warm").json()
        assert forge["warmed"] is False and forge["state"] == "stopped"         # the card is taken
        assert client.post("/api/tools/forge/warm", json={"model": "full"}).json()["warmed"] is False
        taken = client.post("/api/tools/forge/warm", json={"model": "full", "displace": True}).json()
        assert taken["warmed"] is True and taken["state"] == "starting"         # (e) Images asks so: the idle tool goes
        _poll(client, "/api/status", lambda s: s["tools"]["forge"]["state"], {"ready"})
        assert client.get("/api/status").json()["tools"]["pixal3d"]["state"] == "stopped"


def test_api_imesh_sessions(api: Any) -> None:
    app, client, adapters = api
    imesh = adapters["imesh"]
    client.post("/api/projects", json={"name": "Demo"})
    pid = client.post("/api/projects/Demo/props", json={"name": "Barrel"}).json()["id"]
    no_mesh = client.post("/api/imesh/session", json={"project": "Demo", "prop_id": pid})
    assert no_mesh.status_code == 409 and "3D stage" in no_mesh.json()["detail"]

    store = app.state.hub.store
    glb = store.prop_dir("Demo", pid) / "mesh" / "mesh.glb"
    glb.parent.mkdir(parents=True)
    glb.write_bytes(b"glTF-1")
    store.apply_stage_patch("Demo", pid, "mesh", {"glb": "mesh/mesh.glb"})
    store.patch_prop("Demo", pid, {"retopo": {"settings": {"method": "imesh"}}})     # the viewer is this method's

    first = client.post("/api/imesh/session", json={"project": "Demo", "prop_id": pid},
                        headers={"origin": "http://localhost:7780"}).json()
    port = client.get("/api/status").json()["tools"]["imesh"]["port"]
    # The viewer as the Retopo stage embeds it: panel on the right, "Accept" handed to the page that asked.
    options = "panel=right&export_label=Accept&export_action=host&host_origin="
    assert first == {"session_id": "s1",
                     "viewer_url": f"http://127.0.0.1:{port}/viewer?session=s1&{options}http%3A%2F%2Flocalhost%3A7780",
                     "reused": False, "replaced": None, "lost_strokes": False}
    assert imesh.sessions[0]["reuse"] is None and imesh.sessions[0]["src"] == glb.resolve()
    assert imesh.sessions[0]["settings"]["target_vertices"] == 5000
    retopo = client.get("/api/projects/Demo").json()["props"][0]["retopo"]
    assert retopo["session_id"] == "s1" and retopo["session_source"] == "mesh/mesh.glb"

    again = client.post("/api/imesh/session", json={"project": "Demo", "prop_id": pid}).json()
    assert again["session_id"] == "s1" and imesh.sessions[1]["reuse"] == "s1" and again["reused"] is True
    # No Origin header: the hub's own address (the test app has no port of its own).
    assert again["viewer_url"].endswith(f"?session=s1&{options}http%3A%2F%2F127.0.0.1")
    ipv6 = client.post("/api/imesh/session", json={"project": "Demo", "prop_id": pid},
                       headers={"host": "[::1]:7780", "origin": "http://[::1]:7780"}).json()
    assert ipv6["viewer_url"].endswith("host_origin=http%3A%2F%2F%5B%3A%3A1%5D%3A7780")
    glb.write_bytes(b"glTF-2 regenerated")
    fresh = client.post("/api/imesh/session", json={"project": "Demo", "prop_id": pid}).json()
    # The hub always offers the old session; the tool itself refuses it because the GLB changed.
    assert fresh["session_id"] == "s4" and imesh.sessions[3]["reuse"] == "s1"
    assert fresh["reused"] is False and fresh["replaced"] == "mesh_changed" and fresh["lost_strokes"] is False

    wrong = client.post("/api/imesh/session/s1/save", json={"project": "Demo", "prop_id": pid})
    assert wrong.status_code == 409 and "does not belong to this prop" in wrong.json()["detail"]

    # Accept: the viewer's mesh is one of the prop's clean meshes, of the size its vertex count stands for (the one
    # vertex the fake tool makes: the smallest size), picked, with the viewer's settings (the body's are ignored). The
    # clean meshes of the prop's other sizes stay.
    store.patch_prop("Demo", pid, {"retopo": {"settings": {"sizes": [0.1, 3]}}})
    made = store.prop_dir("Demo", pid) / "retopo" / "3k" / "retopo.obj"
    made.parent.mkdir(parents=True)
    made.write_text("v 0 0 0\n", encoding="utf-8")
    store.apply_stage_patch("Demo", pid, "retopo", {
        "variants": [{"size": 3, "obj": "retopo/3k/retopo.obj", "stats": {"vertices": 3000}}],
        "method": "imesh", "source": "mesh/mesh.glb", "saved_from": "auto"})
    accepted = client.post("/api/imesh/session/s4/save", json={
        "project": "Demo", "prop_id": pid, "settings": {"target_vertices": 99}}).json()["retopo"]
    assert (accepted["obj"], accepted["saved_from"], accepted["method"]) == ("retopo/retopo.obj", "viewer", "imesh")
    assert (accepted["source"], accepted["accepted"], accepted["stale"]) == ("mesh/mesh.glb", True, False)
    assert accepted["stats"] == {"vertices": 1, "faces": 0, "has_uv": False, "warning": "no UVs", "seconds": 0.1}
    assert [(v["size"], v["obj"]) for v in accepted["variants"]] == [(0.1, "retopo/retopo.obj"),
                                                                     (3, "retopo/3k/retopo.obj")]
    assert accepted["settings"] == {"method": "imesh", "sizes": [0.1, 3], "target_vertices": 3390, "pure_quad": True,
                                    "crease_angle": -1, "smooth_iter": 2, "uv_leniency": 0.8, "rosy": 6, "posy": 3,
                                    "extrinsic": True, "align_to_boundaries": False, "lato2_seed": 0,
                                    "triflow_quad_ratio": 0.95, "triflow_seed": 0}
    assert (store.prop_dir("Demo", pid) / "retopo" / "retopo.obj").is_file()
    # The next session starts from them.
    client.post("/api/imesh/session", json={"project": "Demo", "prop_id": pid})
    assert imesh.sessions[-1]["settings"]["target_vertices"] == 3390


def test_api_settings_and_open_folder(api: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, client, _ = api
    assert client.get("/api/settings").json()["gpu_exclusive"] is True
    assert client.patch("/api/settings", json={"gpu_exclusive": False}).json()["gpu_exclusive"] is False
    bad = client.patch("/api/settings", json={"ports": {"forge": "abc"}})
    assert bad.status_code == 400 and "ports.forge" in bad.json()["detail"]

    opened: list[Path] = []
    monkeypatch.setattr(os, "startfile", lambda p: opened.append(Path(p)), raising=False)
    monkeypatch.delenv("SPELLCAST_NO_EXPLORER", raising=False)
    assert client.post("/api/open-folder", json={"which": "logs"}).status_code == 403     # "testclient" is not local
    local = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))
    assert local.post("/api/open-folder", json={"which": "logs"}).json() == {"ok": True, "opened": True}
    client.post("/api/projects", json={"name": "Demo"})
    assert local.post("/api/open-folder", json={"which": "project", "project": "Demo"}).json()["opened"] is True
    assert local.post("/api/open-folder", json={"which": "exports", "project": "Demo"}).json()["opened"] is True
    assert local.post("/api/open-folder", json={"which": "project", "project": "../.."}).status_code == 404
    assert local.post("/api/open-folder", json={"which": "C:/"}).status_code == 400
    monkeypatch.setattr(paths, "MODELS", tmp_path / "models")
    monkeypatch.setattr(paths, "MODELS_CHECKPOINTS", tmp_path / "models" / "checkpoints")
    for which in ("models", "forge_models"):          # "Add models...": where the artist drops them
        assert local.post("/api/open-folder", json={"which": which}).json()["ok"] is True
    assert opened == [tmp_path / "logs", tmp_path / "projects" / "Demo", tmp_path / "exports" / "Demo",
                      tmp_path / "models", tmp_path / "models" / "checkpoints"]
    assert (tmp_path / "models" / "checkpoints").is_dir()
    monkeypatch.setenv("SPELLCAST_NO_EXPLORER", "1")    # automated runs: no window on the desktop
    assert local.post("/api/open-folder", json={"which": "logs"}).json() == {"ok": True, "opened": False}
    assert len(opened) == 5


def test_an_early_start_loads_what_the_artist_picked_and_drops_a_start_for_something_else(tmp_path: Path) -> None:
    """procman.warm with a wish (the image generator's model): the start loads it first; a tool that runs without a
    job for another wish is stopped and started anew, in the middle of loading too; a wish that comes before the
    launch replaces the one before; nothing happens under a job, nor for a wish the tool serves. With `displace`
    the start takes the card from a tool that idles on it."""
    async def main() -> None:
        forge = ModelAdapter("forge", "images", True, tmp_path, args=("--delay", "1.5"))
        adapters = fake_adapters(tmp_path, forge=forge,
                                 pixal3d=FakeAdapter("pixal3d", "mesh", True, tmp_path, gpu_at_start=False))
        async with manager(tmp_path, adapters) as pm:
            assert pm.warm("forge", {"model": "full"}) is True
            assert pm.warm("forge", {"model": "flux3"}) is True          # not launched yet: only the wish changes
            await asyncio.sleep(0.3)
            assert forge.launched == ["flux3"] and pm.status("forge")["state"] == "starting"
            loading = pm.get("forge").proc.pid
            assert pm.warm("forge", {"model": "flux3"}) is False and pm.warm("forge") is False     # as it loads
            assert pm.warm("forge", {"model": "full"}) is True           # loading another: dropped for this one ...
            await asyncio.sleep(0)
            assert pm.status("forge")["state"] == "stopping"
            assert pm.warm("forge", {"model": "turbo"}) is True          # ... and while it stops, for that one
            await wait_state(pm, "forge", {"ready"})
            assert forge.launched == ["flux3", "turbo"] and not pid_alive(loading)
            ready = pm.get("forge").proc.pid
            assert pm.warm("forge", {"model": "turbo"}) is False and pm.get("forge").proc.pid == ready

            pm.set_busy("forge", True)                                   # under a job: its request switches itself
            assert pm.warm("forge", {"model": "full"}) is False
            pm.set_busy("forge", False)
            assert pm.warm("forge", {"model": "full"}) is True           # idle with another model: started anew
            await wait_state(pm, "forge", {"stopping", "stopped"})
            await wait_state(pm, "forge", {"ready"})
            assert forge.launched == ["flux3", "turbo", "full"] and not pid_alive(ready)

            await pm.ensure_running("pixal3d")                           # the 3D generator takes the card ...
            await pm.claim_gpu("pixal3d")
            pixal_pid = pm.get("pixal3d").proc.pid
            assert pm.status("forge")["state"] == "stopped"
            assert pm.warm("forge", {"model": "full"}) is False          # ... and an early start displaces nothing
            assert forge.wished is None, "a start that did not begin leaves no wish behind"
            pm.set_busy("pixal3d", True)
            assert pm.warm("forge", {"model": "full"}, displace=True) is False       # never beside a job
            pm.set_busy("pixal3d", False)
            assert pm.warm("forge", {"model": "full"}, displace=True) is True        # Images' arrival: the idle tool goes
            await wait_state(pm, "forge", {"ready"})
            assert forge.launched[-1] == "full" and not pid_alive(pixal_pid)
            await pm.stop("forge")
            await pm.ensure_running("forge")                             # a job's start: the tool's default
            assert forge.launched[-1] == "turbo"

    asyncio.run(main())


def test_a_job_has_a_tool_that_runs_for_something_else_started_anew_for_its_wish(tmp_path: Path) -> None:
    """procman.ensure_running with a wish (a job's: the image generator in its stage's arithmetic): a tool that runs
    for another wish without a job is stopped and started for this one; one that serves it is kept; a start under
    way for another is waited for, then started anew; nothing is stopped under a job."""
    async def main() -> None:
        forge = ModelAdapter("forge", "images", True, tmp_path)
        async with manager(tmp_path, fake_adapters(tmp_path, forge=forge)) as pm:
            await pm.ensure_running("forge", {"model": "full"})
            assert forge.launched == ["full"]
            kept = pm.get("forge").proc.pid
            await pm.ensure_running("forge", {"model": "full"})
            await pm.ensure_running("forge", {})
            assert forge.launched == ["full"] and pm.get("forge").proc.pid == kept, "served: kept"
            await pm.ensure_running("forge", {"model": "turbo"})
            assert forge.launched == ["full", "turbo"] and not pid_alive(kept), "another wish: started anew"
            pm.set_busy("forge", True)
            await pm.ensure_running("forge", {"model": "full"})
            assert forge.launched == ["full", "turbo"], "under a job: never stopped"
            pm.set_busy("forge", False)
            await pm.stop("forge")
            assert pm.warm("forge", {"model": "flux3"}) is True          # a start under way for another wish ...
            await pm.ensure_running("forge", {"model": "full"})
            assert forge.launched == ["full", "turbo", "flux3", "full"], "... done, then started anew for this one"

    asyncio.run(main())


def test_a_hand_off_start_may_stop_the_idle_tool_on_the_card(tmp_path: Path) -> None:
    """procman.start_early(displace=True), the job runner's hand-off between stages: the GPU tool that holds the card
    without a job on it is stopped for the next stage's tool; never beside a busy one; warm() never displaces."""
    async def main() -> None:
        adapters = fake_adapters(tmp_path, pixal3d=FakeAdapter("pixal3d", "mesh", True, tmp_path, gpu_at_start=False))
        async with manager(tmp_path, adapters) as pm:
            await pm.ensure_running("pixal3d")
            await pm.claim_gpu("pixal3d")
            pixal_pid = pm.get("pixal3d").proc.pid
            assert pm.warm("forge") is False and pm.start_early("forge") is False     # the card is taken
            pm.set_busy("pixal3d", True)
            assert pm.start_early("forge", displace=True) is False                     # busy with a job: never
            pm.set_busy("pixal3d", False)
            assert pm.start_early("forge", displace=True) is True
            await wait_state(pm, "forge", {"ready"})
            assert pm.status("pixal3d")["state"] == "stopped" and not pid_alive(pixal_pid)
            assert pm.status("forge")["holds_gpu"]
            assert pm.start_early("forge", displace=True) is False                     # running already

    asyncio.run(main())
