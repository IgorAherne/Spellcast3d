"""Qwen-Image 2.1 in the hub: the image generator the root config file picks (only it is registered, none when the
file is not valid), the Qwen adapter (its request to the launcher, its pictures saved like Forge's, cancel and
errors), a prop's example pictures (images.references, and the one "reference" of older projects), Qwen's own
images settings, its licence in /api/status, and the config file changing while the hub runs.

Runs in the hub venv. The launcher (launchers/qwen_server.py) is played by fake_tool.py: FakeQwen is the real
QwenAdapter started on it. The config file is each test's own (`root_config`, through SPELLCAST_CONFIG), never the
one in the Spellcast3D folder.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from spellcast import config, gpu, paths
from spellcast.adapters import IMAGE_ADAPTERS, ForgeAdapter, QwenAdapter, build_adapters, qwen
from spellcast.adapters.base import LaunchSpec, StageContext, StageError, ToolNeedsRestart
from spellcast.adapters.images_common import reference_files
from spellcast.gpu import GpuInfo
from spellcast.install import models
from spellcast.projects import (DEFAULT_STAGE_SETTINGS, IMAGES_NUMBERS, MAX_REFERENCES, QWEN_MODELS, ProjectError, ProjectStore,
                                _is_empty)
from spellcast.server import config_changed, create_app
from test_core import FAKE_TOOL, JPG, PNG, WEBP, FakeAdapter, _poll, make_settings

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Spellcast3D is Windows-only")

QWEN, FORGE = paths.QWEN, paths.FORGE
LICENSE = {"name": "Qwen Research License (model)",
           "url": "https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE",
           "noncommercial": False, "research": True,
           "note": "The pictures you make are yours, also for commercial work (Qwen, 21 Sep 2026).",
           # the statement the note rests on (Qwen Developers on X, 21 Sep 2026)
           "links": [{"label": "Qwen’s statement", "url": "https://x.com/QwenDevs/status/2101917379785838660"}]}
RTX_3060 = GpuInfo(0, "Fake RTX 3060", 8.6, 12288, 3000, "580.10")


class FakeQwen(QwenAdapter):
    """The real QwenAdapter, its launcher played by fake_tool.py."""

    def __init__(self, cwd: Path, *args: str):
        # its models folders are the test's (empty: the full model and FLUX.3 are not downloaded), never models\...
        super().__init__(quality="Q4_K_M", models_dir=cwd / "qwen-image", flux3_dir=cwd / "flux3-klein")
        self.cwd, self.args = cwd, list(args)

    def install_status(self) -> dict[str, Any]:
        return {"status": "installed", "reason": ""}

    def supported(self) -> tuple[bool, str]:
        return True, ""

    def launch_spec(self, port: int, settings: dict[str, Any]) -> LaunchSpec:
        return LaunchSpec(argv=[sys.executable, str(FAKE_TOOL), "--port", str(port), *self.args],
                          cwd=self.cwd, env=paths.clean_env(None), ready_timeout=30.0)


@pytest.fixture
def root_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """The root config file of a test: say(backend, quality) saves it as the template has it, with that backend line
    ("forg": a typo) and quality (a new modification time each time). It starts as the template: qwen, Q4_K_M."""
    template = paths.CONFIG_TEMPLATE.read_text(encoding="utf-8")
    assert "\nbackend = qwen " in template and "\nquality = Q4_K_M" in template
    file = tmp_path / "spellcast3d-config.ini"
    saves = [time.time()]

    def say(backend: str = QWEN, quality: str = "Q4_K_M") -> None:
        file.write_text(template.replace("\nbackend = qwen ", f"\nbackend = {backend} ")
                        .replace("\nquality = Q4_K_M", f"\nquality = {quality}"), encoding="utf-8")
        saves.append(saves[-1] + 2)
        os.utime(file, (saves[-1], saves[-1]))

    monkeypatch.setenv("SPELLCAST_CONFIG", str(file))
    say()
    return say


def _static(tmp_path: Path) -> Path:
    static = tmp_path / "static"
    static.mkdir(exist_ok=True)
    (static / "index.html").write_text("<h1>Spellcast3D</h1>", encoding="utf-8")
    return static


@pytest.fixture
def api(tmp_path: Path, root_config: Callable[..., None]) -> Any:
    adapters = {QWEN: FakeQwen(tmp_path), "pixal3d": FakeAdapter("pixal3d", "mesh", True, tmp_path),
                "imesh": FakeAdapter("imesh", "retopo", False, tmp_path)}
    app = create_app(adapters, settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=_static(tmp_path), logs_dir=tmp_path / "logs", gpu_probe=lambda: RTX_3060)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield app, client, root_config


def _prop(client: TestClient, project: str = "Demo", name: str = "Barrel") -> str:
    if client.get(f"/api/projects/{project}").status_code == 404:
        client.post("/api/projects", json={"name": project})
    return client.post(f"/api/projects/{project}/props", json={"name": name}).json()["id"]


def _run(client: TestClient, pid: str, **settings: Any) -> dict[str, Any]:
    """Set the prop's images settings, run an images job -> the finished job."""
    assert client.patch(f"/api/projects/Demo/props/{pid}", json={"images": {"settings": settings}}).status_code == 200
    job = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [pid]}).json()
    assert job["tool"] == QWEN, job
    return _poll(client, f"/api/jobs/{job['id']}", lambda j: j["status"], {"done", "error", "cancelled"})


def _images(client: TestClient, pid: str) -> dict[str, Any]:
    return next(p for p in client.get("/api/projects/Demo").json()["props"] if p["id"] == pid)["images"]


def _launcher(client: TestClient, route: str) -> Any:
    port = client.get("/api/status").json()["tools"][QWEN]["port"]
    return httpx.get(f"http://127.0.0.1:{port}{route}", timeout=10.0, trust_env=False).json()


def _color_type(png: bytes) -> int:
    """The PNG's colour type: 6 RGBA, 2 RGB."""
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and png[12:16] == b"IHDR"
    return png[25]


# ---- which image generator ----------------------------------------------------------------------------------

def test_build_adapters_registers_the_configured_image_generator(root_config: Callable[..., None]) -> None:
    every = build_adapters()
    assert [t for t in every if every[t].stage == "images"] == list(paths.STAGE_TOOLS["images"]) == [QWEN, FORGE]
    assert isinstance(every[QWEN], QwenAdapter) and isinstance(every[FORGE], ForgeAdapter)
    assert list(IMAGE_ADAPTERS) == [QWEN, FORGE]
    others = {"pixal3d", "imesh", "lato2", "triflow"}
    for backend, images in ((QWEN, {QWEN}), (FORGE, {FORGE}), ("", set()), ("forg", set())):
        adapters = build_adapters(image_backend=backend)
        assert set(adapters) == others | images, backend
    root_config(quality="Q8_0")
    assert QwenAdapter().quality == "Q8_0"                  # the model size the config picks
    assert QwenAdapter(quality="Q3_K_M").quality == "Q3_K_M"


@pytest.mark.parametrize("backend", [QWEN, FORGE, None])
def test_the_hub_registers_only_the_configured_generator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         root_config: Callable[..., None], backend: str | None) -> None:
    """create_app without adapters: the image generator spellcast3d-config.ini picks, or none when the file has a
    mistake (its problems are in /api/status, and an images job says them)."""
    root_config(backend or "forg")
    problems = config.problems()
    assert bool(problems) == (backend is None) and all('"forg"' in p for p in problems)
    monkeypatch.setattr(gpu, "primary_gpu", lambda: RTX_3060)     # no nvidia-smi
    app = create_app(settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                     static_dir=_static(tmp_path), logs_dir=tmp_path / "logs", gpu_probe=lambda: RTX_3060)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        status = client.get("/api/status").json()
        images = [backend] if backend else []
        assert [t for t, s in status["tools"].items() if s["stage"] == "images"] == images
        assert status["stage_tools"]["images"] == images
        assert status["stage_tools"]["retopo"] == ["triflow", "lato2", "imesh"] and status["stage_tools"]["rig"] == []
        # the problems without their "until this is fixed" sentence: the page's Images gate says that its own way
        assert status["config"] == {"image_backend": backend, "quality": "Q4_K_M",
                                    "problems": [config.short(p) for p in problems], "changed": None}
        assert status["settings_warnings"] == []
        if backend is None:
            pid = _prop(client)
            refused = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [pid]})
            detail = refused.json()["detail"]
            assert refused.status_code == 409 and config.short(problems[0]) in detail
            assert detail.endswith("Fix it, save the file, then close Spellcast3D and start it again.")
        if backend != FORGE:
            forge_models = client.get("/api/forge/models")
            assert forge_models.status_code == 409 and "spellcast3d-config.ini" in forge_models.json()["detail"]


def test_status_has_the_licence_the_tools_and_says_when_the_config_changed(api: Any) -> None:
    app, client, say = api
    status = client.get("/api/status").json()
    assert status["tools"][QWEN]["license"] == LICENSE == qwen.LICENSE
    assert status["tools"][QWEN]["display_name"] == "Qwen-Image 2.1" and status["tools"][QWEN]["stage"] == "images"
    assert status["stage_tools"]["images"] == [QWEN] and status["stage_tools"]["mesh"] == ["pixal3d"]
    assert status["config"] == {"image_backend": QWEN, "quality": "Q4_K_M", "problems": [], "changed": None}

    say(QWEN, "Q4_K_M")                                     # saved again, the same choice: nothing to say
    assert client.get("/api/status").json()["settings_warnings"] == []
    say(FORGE)
    status = client.get("/api/status").json()
    [said] = status["settings_warnings"]
    assert said.startswith("spellcast3d-config.ini now picks Forge Neo.") and "start it again" in said
    assert status["config"]["changed"] == said              # the page shows it on the Images stage too
    assert said == config_changed(config.parse(paths.CONFIG_TEMPLATE.read_text(encoding="utf-8")), config.load())
    assert status["config"]["image_backend"] == QWEN       # what the hub runs with until it starts again
    say(QWEN, "Q8_0")
    assert "model size Q8_0" in client.get("/api/status").json()["settings_warnings"][0]
    say(QWEN, "Q4_K_M")                                     # switched back before a restart
    status = client.get("/api/status").json()
    assert status["settings_warnings"] == [] and status["config"]["changed"] is None
    say("forg")                                             # a mistake: no image generator at the next start
    assert "has a mistake now" in client.get("/api/status").json()["settings_warnings"][0]


# ---- the adapter: the launcher's request -------------------------------------------------------------------

def test_the_request_maps_the_images_settings(tmp_path: Path) -> None:
    refs = [tmp_path / "reference" / "reference_001.png", tmp_path / "reference" / "reference_002.jpg"]
    params = {"prompt": " a wooden barrel with iron bands. ", "object_mode": True,       # "3D-ready framing": retired
              "transparent": True, "width": 1000, "height": 830, "seed": 42, "batch_count": 3,
              "qwen_steps": 24, "qwen_model": "full",
              # Forge's own ("Avoid" too), and the guidance of projects from before Qwen's own recipe: never sent
              "negative_prompt": " blurry ", "qwen_cfg": 4.5,
              "steps": 7, "cfg_scale": 2.0, "sampler": "DPM++ 2M", "checkpoint": "x.safetensors"}
    body = qwen.qwen_request(params, refs, tmp_path / "out")
    # no guidance, whatever the prop holds: CFG 1.0 (one model pass per step) and no negative prompt
    # the words as the artist typed them: nothing is added, whatever an older prop's object_mode says
    assert body == {"prompt": "a wooden barrel with iron bands.", "negative_prompt": "",
                    "width": 992, "height": 832, "model": "full", "steps": 24, "cfg_scale": 1.0, "sampler": "euler",
                    "seed": 42, "batch_count": 3, "ref_images": [str(p.resolve()) for p in refs], "transparent": True,
                    "tileable": False, "slide_back": 0.0, "keep_place": False, "first_block_cache": None,
                    "out_dir": str((tmp_path / "out").resolve())}
    # Re-Render Viewport from a picture the page showed slid: the pictures made are slid back as far. Only with
    # from_viewport, and only a number the slider can give.
    slide = lambda **more: qwen.qwen_request({**params, **more}, refs, tmp_path / "out")["slide_back"]    # noqa: E731
    assert [slide(from_viewport="viewport/viewport.png", viewport_shift=s) for s in (0.1, -0.5, 0, 0.5)] == [0.1, -0.5, 0.0, 0.5]
    assert [slide(from_viewport="images/0001.png", viewport_shift=s) for s in (0.6, -1, "0.1", True, None, float("nan"))] == [0.0] * 6
    assert slide(viewport_shift=0.1) == 0.0, "a picture drawn from the words alone lies where it is drawn"
    # ... and, tileable, are laid onto it to the pixel: for a job that says how far its picture was slid (0 too).
    place = lambda **more: qwen.qwen_request({**params, "qwen_model": "turbo", **more}, refs, tmp_path / "out")["keep_place"]   # noqa: E731
    assert [place(tileable=True, from_viewport="images/0001.png", viewport_shift=s) for s in (0, 0.1, -0.5)] == [True] * 3
    assert place(tileable=False, from_viewport="images/0001.png", viewport_shift=0) is False
    assert place(tileable=True, from_viewport="images/0001.png") is False, "no word of the page's: a picture of its own"
    assert place(tileable=True, viewport_shift=0) is False and place(tileable=True, from_viewport="images/0001.png", viewport_shift=2) is False
    assert place(tileable=True, qwen_model="flux3", from_viewport="images/0001.png", viewport_shift=0) is False
    # A Smart Mask job (kept: the picture a part of which changes, and the mask of that part): the launcher is told
    # both, and the first picture the job is shown (its tinted copy) with the mask as what the drawing may be held
    # to - whether it draws the change whole or held is its own to say. A tileable picture's job says so (the
    # launcher then holds it, not wrapped round), is slid back, and is not laid onto its picture, which lies where
    # it lay.
    kept = (tmp_path / "images" / "0001.png", tmp_path / "viewport" / "mask.png")
    for tileable, shift in ((False, None), (True, 0.1)):
        asked = {**params, "qwen_model": "turbo", "tileable": tileable, "from_viewport": "viewport/viewport.png"}
        masked = qwen.qwen_request({**asked, **({"viewport_shift": shift} if shift else {})}, refs, tmp_path / "out", kept)
        assert (masked["changed_image"], masked["changed_mask"]) == tuple(str(p.resolve()) for p in kept)
        assert (masked["init_image"], masked["mask_image"], masked["strength"]) == (str(refs[0].resolve()), str(kept[1].resolve()), 1.0)
        assert (masked["tileable"], masked["keep_place"], masked["slide_back"]) == (tileable, False, shift or 0.0)
        assert set(masked) - set(body) == {"changed_image", "changed_mask", "init_image", "mask_image", "strength"}
    assert (body["cfg_scale"], body["negative_prompt"]) == (qwen.CFG_SCALE, qwen.NEGATIVE_PROMPT)
    assert set(qwen.qwen_settings(params)) == {"model", "steps", "width", "height", "seed", "batch_count",
                                               "transparent", "tileable", "first_block_cache"}
    # Tileable: Qwen-Image's two models, on the artist's tick alone; never FLUX.3 (its positions do not wrap round)
    assert [qwen.qwen_request({**params, "tileable": tick, "qwen_model": model}, refs, tmp_path / "out")["tileable"]
            for model, tick in (("full", True), ("turbo", True), ("flux3", True), ("turbo", 1), ("turbo", False))] == [
        True, True, False, False, False]
    # the Turbo, the default model: its own 8 steps and no First-Block Cache, whatever the prop's (the full
    # model's) qwen_steps and first_block_cache say; everything else as for the full model
    turbo = qwen.qwen_request({**params, "qwen_model": "turbo", "first_block_cache": True}, refs, tmp_path / "out")
    assert turbo == {**body, "model": "turbo", "steps": qwen.TURBO_STEPS, "first_block_cache": False}
    assert qwen.TURBO_STEPS == 8 and qwen.QWEN_MODELS == QWEN_MODELS == ("turbo", "full", "flux3")
    assert qwen.DEFAULT_MODEL == "turbo"
    for older in ({}, {"qwen_model": None}):                                          # a prop from before the choice
        assert qwen.qwen_settings({**older, "qwen_steps": 40})["model"] == "turbo"
    with pytest.raises(StageError, match="Turbo, Full or FLUX.3 Klein"):
        qwen.qwen_settings({"qwen_model": "fast"})

    # transparent only says whether the launcher wraps the prompt in Qwen's RGBA words: the prompt is the same
    opaque = qwen.qwen_request({**params, "transparent": False}, [], tmp_path)
    assert opaque["prompt"] == body["prompt"] and opaque["transparent"] is False
    assert qwen.qwen_request({"prompt": "lamp"}, [], tmp_path)["prompt"] == "lamp"
    defaults = qwen.qwen_request({"prompt": "lamp"}, [], tmp_path)                       # an older prop's settings
    assert (defaults["model"], defaults["steps"], defaults["cfg_scale"], defaults["transparent"]) == (
        "turbo", 8, 1.0, False)                                                         # opaque unless asked
    full = qwen.qwen_request({"prompt": "lamp", "qwen_model": "full"}, [], tmp_path)
    assert (full["steps"], full["first_block_cache"]) == (25, None)                      # the card decides
    for tick in (True, False):                                                           # the artist's tick goes as is
        assert qwen.qwen_request({"prompt": "lamp", "qwen_model": "full", "first_block_cache": tick}, [],
                                 tmp_path)["first_block_cache"] is tick
    # the launcher's default per card: on for the GTX 10xx (6.x), off from the RTX 20xx (7.5) on; the tick wins
    if str(paths.LAUNCHERS) not in sys.path:     # the launchers import their siblings as top-level modules
        sys.path.insert(0, str(paths.LAUNCHERS))
    from spellcast.launchers.qwen_server import first_block_cache_on
    assert first_block_cache_on(None, (6, 1)) and not first_block_cache_on(None, (7, 5))
    assert not first_block_cache_on(None, (8, 6)) and not first_block_cache_on(None, (12, 0))
    assert first_block_cache_on(True, (8, 6)) and not first_block_cache_on(False, (6, 1))
    assert (defaults["width"], defaults["height"], defaults["seed"], defaults["batch_count"]) == (1024, 1024, -1, 1)
    sizes = qwen.qwen_request({"prompt": "x", "width": 100, "height": 5000, "seed": -7}, [], tmp_path)
    assert (sizes["width"], sizes["height"], sizes["seed"]) == (256, qwen.MAX_SIDE, -1)
    assert qwen.qwen_request({"prompt": "x", "width": 832, "height": "1216"}, [], tmp_path)["height"] == 1216

    # Resolution: the Shape's sides (kept at 1K in the prop) times 1, 1.5 or 2, on the engine's 32-pixel grid.
    def sized(shape: tuple[int, int], resolution: object) -> tuple[int, int]:
        request = qwen.qwen_request({"prompt": "x", "width": shape[0], "height": shape[1], "resolution": resolution},
                                    [], tmp_path)
        return request["width"], request["height"]

    assert sized((1024, 1024), 1) == (1024, 1024) and sized((1024, 1024), None) == (1024, 1024)
    assert sized((1024, 1024), 1.5) == (1536, 1536) and sized((1024, 1024), 2) == (2048, 2048)
    assert sized((832, 1216), 1.5) == (1248, 1824) and sized((1216, 832), 2) == (2432, 1664)
    assert all(side % qwen.GRID == 0 and side <= qwen.MAX_SIDE for shape in ((1024, 1024), (832, 1216), (1216, 832))
               for resolution in qwen.RESOLUTIONS for side in sized(shape, resolution))
    assert defaults.get("resolution", 1) == 1 and list(qwen.RESOLUTIONS.values()) == ["1K", "1.5K", "2K"]
    for wrong in (3, 1.25, 0, "big"):
        with pytest.raises(StageError, match="Resolution"):
            sized((1024, 1024), wrong)

    # Example pictures without a prompt: the prompt points at them.
    assert qwen.qwen_request({"object_mode": True}, refs, tmp_path)["prompt"] == "the object from the example pictures"
    assert qwen.qwen_request({}, refs[:1], tmp_path)["prompt"] == "the object from the example picture"
    with pytest.raises(StageError, match="Type a prompt"):
        qwen.qwen_request({"prompt": "  "}, [], tmp_path)
    with pytest.raises(StageError, match="at most 10"):
        qwen.qwen_request({"prompt": "x"}, refs * 6, tmp_path)
    for bad, words in (({"qwen_steps": 0}, "Steps"), ({"qwen_steps": 101}, "Steps"), ({"qwen_steps": 2.5}, "Steps"),
                       ({"batch_count": 17}, "Variations"), ({"width": "wide"}, "Width"), ({"seed": True}, "Seed")):
        with pytest.raises(StageError, match=words):
            qwen.qwen_request({"prompt": "x", **bad}, [], tmp_path)
        with pytest.raises(StageError, match=words):
            FakeQwen(tmp_path).check_settings("images", bad)       # said when the job is submitted
    # Guidance is no setting of Qwen-Image: what a prop holds for it is not even checked.
    for ignored in ({"qwen_cfg": 0.5}, {"qwen_cfg": "much"}, {"cfg_scale": "a lot"}, {"negative_prompt": 5}):
        FakeQwen(tmp_path).check_settings("images", ignored)
        sent = qwen.qwen_request({"prompt": "x", **ignored}, [], tmp_path)
        assert (sent["cfg_scale"], sent["negative_prompt"]) == (1.0, "")


def test_the_adapter_the_launcher_and_the_projects_agree() -> None:
    """The limits and defaults the adapter uses are the launcher's (launchers/qwen_server.py) and the projects'."""
    if str(paths.LAUNCHERS) not in sys.path:     # the launchers import their siblings as top-level modules
        sys.path.insert(0, str(paths.LAUNCHERS))
    import qwen_server as launcher

    assert (qwen.MAX_REFERENCES, qwen.MAX_BATCH) == (launcher.MAX_REFERENCES, launcher.MAX_BATCH) == (10, 16)
    assert (qwen.MIN_SIDE, qwen.MAX_SIDE, qwen.GRID) == (launcher.SIDE_MIN, launcher.SIDE_MAX, launcher.SIDE_STEP)
    # Qwen's own recipe in both: no guidance (CFG 1.0, no negative prompt), and the same default number of steps
    assert (qwen.DEFAULT_STEPS, qwen.CFG_SCALE) == (launcher.STEPS, launcher.CFG_SCALE) == (25, 1.0)
    assert launcher.GenerateRequest(prompt="x", out_dir="y").negative_prompt == qwen.NEGATIVE_PROMPT == ""
    assert qwen.SAMPLER == launcher.SAMPLER
    assert qwen.MAX_REFERENCES == MAX_REFERENCES
    assert IMAGES_NUMBERS["qwen_steps"][1:3] == (qwen.MIN_STEPS, qwen.MAX_STEPS)
    # the adapter's DEFAULT_STEPS is the hub's one place for the steps; a new prop's settings repeat it
    defaults = DEFAULT_STAGE_SETTINGS["images"]
    assert (defaults["qwen_steps"], defaults["transparent"], defaults["first_block_cache"]) == (qwen.DEFAULT_STEPS, False, None)
    assert "qwen_cfg" not in defaults and "qwen_cfg" not in IMAGES_NUMBERS      # guidance is no setting of Qwen-Image
    assert defaults["negative_prompt"] == ""                                    # Forge Neo's "Avoid" stays one
    # the page's one place for the number (its Steps slider when a prop has none saved) says the same
    page = (paths.STATIC / "js" / "stages" / "images.js").read_text(encoding="utf-8")
    assert re.findall(r"^export const QWEN_STEPS = (\d+);", page, re.MULTILINE) == [str(qwen.DEFAULT_STEPS)]


def test_launch_spec_and_licence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                 root_config: Callable[..., None]) -> None:
    sdcpp = tmp_path / "sdcpp"
    monkeypatch.setattr(paths, "QWEN_SDCPP", sdcpp)
    adapter = QwenAdapter(quality="Q5_K_M")
    with pytest.raises(StageError, match="stable-diffusion.cpp"):
        adapter.launch_spec(7776, {})
    sdcpp.mkdir()
    spec = adapter.launch_spec(7776, {})
    assert spec.argv[:2] == [str(paths.venv_python(QWEN)), str(paths.LAUNCHERS / "qwen_server.py")]
    args = dict(zip(spec.argv[2::2], spec.argv[3::2]))
    assert args == {"--port": "7776", "--projects-root": str(paths.PROJECTS), "--models": str(paths.MODELS_QWEN),
                    "--albedo": str(paths.MODELS_ALBEDO),
                    "--flux3-models": str(paths.MODELS_FLUX3), "--sdcpp": str(sdcpp), "--quality": "Q5_K_M",
                    "--text-encoder": config.qwen_text_encoder(), "--first-model": "turbo", "--matmul": "mmq"}
    assert spec.cwd == sdcpp and spec.env["PATH"].split(os.pathsep)[0] == str(sdcpp)    # its DLLs load
    assert spec.env["VIRTUAL_ENV"] == str(paths.venv_dir(QWEN)) and spec.ready_timeout >= 600
    assert adapter.license == LICENSE and adapter.uses_gpu and adapter.role == "image generator"
    monkeypatch.setattr(gpu, "qwen_supported", lambda info: (info is RTX_3060, "too old"))
    monkeypatch.setattr(gpu, "primary_gpu", lambda: RTX_3060)
    assert QwenAdapter(quality="Q4_K_M").supported() == (True, "too old")


def test_an_early_start_loads_the_model_the_artist_picked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ToolAdapter.aim / serves: the page's early start names the model picked; the next start alone loads it first
    (--first-model), and a launcher that holds or loads another does not serve the wish. A model that cannot draw
    now (not downloaded, an engine that cannot) is no wish."""
    from spellcast.adapters import qwen

    sdcpp = tmp_path / "sdcpp"
    sdcpp.mkdir()
    monkeypatch.setattr(paths, "QWEN_SDCPP", sdcpp)
    monkeypatch.setattr(qwen, "_holding", "turbo")
    there = {"full": False, "flux3": False, "engine": False}
    adapter = QwenAdapter(quality="Q4_K_M", text_encoder="UD-Q4_K_XL")
    monkeypatch.setattr(adapter, "full_model_on_disk", lambda: there["full"])
    monkeypatch.setattr(adapter, "flux3_on_disk", lambda: there["flux3"])
    monkeypatch.setattr(qwen.engine_pin, "engine_draws", lambda model: there["engine"])

    def first() -> str:
        argv = adapter.launch_spec(7776, {}).argv
        return dict(zip(argv[2::2], argv[3::2]))["--first-model"]

    assert all(adapter.serves(wish, {}) for wish in ({}, {"model": "turbo"}, {"displace": True}, {"model": "sdxl"}))
    assert adapter.serves({"model": "full"}, {}), "not downloaded: no wish"
    adapter.aim({"model": "full"})
    assert first() == "turbo"
    there["full"] = True
    assert not adapter.serves({"model": "full"}, {})
    adapter.aim({"model": "full"})
    assert first() == "full" and qwen._holding == "full"
    assert adapter.serves({"model": "full"}, {}) and not adapter.serves({"model": "turbo"}, {})
    assert first() == "turbo", "that start alone: a job's or a hand-off's loads the Turbo"
    adapter.aim({"model": "full"})
    adapter.aim({})
    assert first() == "turbo", "a start without a wish drops the one before"
    there["flux3"] = True
    assert adapter.serves({"model": "flux3"}, {}), "its files alone do not make it drawable"
    there["engine"] = True
    assert not adapter.serves({"model": "flux3"}, {})
    adapter.aim({"model": "flux3"})
    assert first() == "flux3" and adapter.serves({"model": "flux3"}, {}) and not adapter.serves({"model": "full"}, {})


def test_faster_image_generation_starts_the_launcher_on_the_int8_matmuls(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Each stage's "Faster image generation" (Images: qwen.int8_matmul, Omni-Texture: qwen.int8_matmul_omnitexture,
    both off by default) reaches the launcher as --matmul int8 when it is started for that stage (aim's "stage"; none:
    Images); anything else, a missing or broken entry included, keeps the model file's own pictures (--matmul mmq).
    A launcher started in one stage's arithmetic does not serve a wish of the other stage that wants the other one."""
    from spellcast.adapters import qwen

    sdcpp = tmp_path / "sdcpp"
    sdcpp.mkdir()
    monkeypatch.setattr(paths, "QWEN_SDCPP", sdcpp)
    monkeypatch.setattr(qwen, "_matmul", "mmq")
    defaults = json.loads((paths.PACKAGE / "defaults.json").read_text(encoding="utf-8"))
    assert defaults["qwen"] == {"int8_matmul": False, "int8_matmul_omnitexture": False}
    adapter = QwenAdapter()

    def matmul(settings: dict[str, Any], stage: str | None = None) -> str:
        if stage is not None:
            adapter.aim({"stage": stage})
        argv = adapter.launch_spec(7776, settings).argv
        return argv[argv.index("--matmul") + 1]

    assert matmul(defaults) == matmul({}) == matmul({"qwen": None}) == "mmq"
    assert matmul({"qwen": {"int8_matmul": "yes"}}) == "mmq"
    images_on = {**defaults, "qwen": {"int8_matmul": True, "int8_matmul_omnitexture": False}}
    assert matmul(images_on) == matmul(images_on, "images") == "int8"
    assert matmul(images_on, "omnitexture") == "mmq", "Omni-Texture's own switch is off"
    assert adapter.serves({"stage": "omnitexture"}, images_on) and not adapter.serves({"stage": "images"}, images_on)
    assert adapter.serves({}, images_on) and adapter.serves({"stage": "mesh"}, images_on), "no image stage: any"
    assert matmul(images_on) == "int8", "a start without a stage: Images' (the one before is not kept)"
    omni_on = {**defaults, "qwen": {"int8_matmul": False, "int8_matmul_omnitexture": True}}
    assert matmul(omni_on, "omnitexture") == "int8" and matmul(omni_on, "images") == "mmq"
    both = {**defaults, "qwen": {"int8_matmul": True, "int8_matmul_omnitexture": True}}
    assert matmul(both, "images") == "int8" and adapter.serves({"stage": "omnitexture"}, both), "the same arithmetic"
    launcher = (paths.LAUNCHERS / "qwen_server.py").read_text(encoding="utf-8")
    choices = re.search(r"^MATMULS = (\(.*\))$", launcher, re.MULTILINE)
    assert choices and set(ast.literal_eval(choices.group(1))) == {"mmq", "int8"}       # what the launcher takes


def test_a_profiling_switch_starts_the_profiled_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      root_config: Callable[..., None]) -> None:
    """Any [profiling] switch on when Qwen-Image starts: spellcast/profiling/qwen_profiled.py instead of the server
    (it runs the server with the profilers attached), with everything else of the launch the same."""
    sdcpp = tmp_path / "sdcpp"
    sdcpp.mkdir()
    monkeypatch.setattr(paths, "QWEN_SDCPP", sdcpp)
    file = Path(os.environ["SPELLCAST_CONFIG"])
    template = paths.CONFIG_TEMPLATE.read_text(encoding="utf-8")
    plain = QwenAdapter().launch_spec(7776, {})
    assert plain.argv[1] == str(paths.LAUNCHERS / "qwen_server.py") and not any(config.profiling().values())
    entry = paths.PACKAGE / "profiling" / "qwen_profiled.py"
    assert entry.is_file()
    for n, key in enumerate(config.PROFILING_KEYS, 1):
        text, found = re.subn(rf"(?m)^{key}\s*=\s*off\b", f"{key} = on", template)
        file.write_text(text if found else f"{template}\n[profiling]\n{key} = on\n", encoding="utf-8")
        os.utime(file, (time.time() + 10 * n, time.time() + 10 * n))
        spec = QwenAdapter().launch_spec(7776, {})
        assert config.profiling()[key] and spec.argv[1] == str(entry)
        assert spec.argv[:1] + spec.argv[2:] == plain.argv[:1] + plain.argv[2:]          # the same arguments
        assert (spec.cwd, spec.env, spec.ready_timeout) == (plain.cwd, plain.env, plain.ready_timeout)
    root_config()                                                                       # all off again
    assert QwenAdapter().launch_spec(7776, {}).argv == plain.argv


def test_reference_files_are_checked(tmp_path: Path) -> None:
    ctx = StageContext(project="p", prop_id="p_000001", prop_dir=tmp_path, base_url="http://x",
                       client=None,  # type: ignore[arg-type]
                       settings={}, report=lambda f, m: None, cancelled=lambda: False)
    (tmp_path / "reference").mkdir()
    (tmp_path / "reference" / "reference_001.png").write_bytes(PNG)
    assert reference_files(ctx, {"references": ["reference/reference_001.png"]}) == \
        [tmp_path / "reference" / "reference_001.png"]
    assert reference_files(ctx, {}) == []
    (tmp_path.parent / "elsewhere.png").write_bytes(PNG)
    for bad in ("reference/reference_002.png", "../elsewhere.png"):
        with pytest.raises(StageError, match="example pictures is missing"):
            reference_files(ctx, {"references": ["reference/reference_001.png", bad]})


# ---- the adapter against the launcher ------------------------------------------------------------------------

def test_a_qwen_job_saves_its_pictures_like_forge(api: Any) -> None:
    app, client, _ = api
    pid = _prop(client)
    added = client.post(f"/api/projects/Demo/props/{pid}/images/references",
                        files=[("files", ("front.png", PNG, "image/png")), ("files", ("side.jpg", JPG, "image/jpeg"))])
    refs = added.json()["images"]["references"]
    assert refs == ["reference/reference_001.png", "reference/reference_002.jpg"]

    job = _run(client, pid, prompt="a red barrel", batch_count=2, seed=5, transparent=True, object_mode=True)
    assert job["status"] == "done", job
    images = _images(client, pid)
    assert images["candidates"] == ["images/0001.png", "images/0002.png"] and images["chosen"] == "images/0001.png"
    assert images["last_seeds"] == [5, 6]
    prop_dir = app.state.hub.store.prop_dir("Demo", pid)
    assert all(_color_type((prop_dir / c).read_bytes()) == 6 for c in images["candidates"])     # their own alpha
    assert not (prop_dir / "images" / ".incoming").exists()
    sent = _launcher(client, "/fake/qwen")[-1]
    # what Qwen-Image was given: the launcher's record, beside each picture it made, with the number of its answer
    assert sent["given_dir"] == str((prop_dir / "images" / ".incoming" / "given").resolve())
    for answer, name in enumerate(("0001", "0002")):
        kept = prop_dir / "images" / ".given" / name
        assert json.loads((kept / "given.json").read_text(encoding="utf-8")) == {
            "words": "a red barrel", "way": "words", "pictures": [], "answers": ["answer-1.webp"] * 2, "answer": answer}
        assert (kept / "answer-1.webp").read_bytes() == b"an answer"
    assert client.get(f"/files/Demo/props/{pid}/images/.given/0002/given.json").json()["answer"] == 1, "the page reads it"
    assert sent["ref_images"] == [str((prop_dir / r).resolve()) for r in refs]
    assert sent["prompt"] == "a red barrel" and sent["transparent"] is True      # as typed: object_mode is retired
    assert "object_mode" not in images["settings"]
    # a new prop draws with the Turbo, the default model: its own 8 steps
    assert images["settings"]["qwen_model"] == "turbo"
    assert (sent["model"], sent["steps"], sent["cfg_scale"], sent["sampler"], sent["batch_count"]) == (
        "turbo", 8, 1.0, "euler", 2)
    assert sent["negative_prompt"] == "" and sent["first_block_cache"] is False

    # The next pictures join them; the first new one becomes the chosen one, also after the artist picked another.
    client.post(f"/api/projects/Demo/props/{pid}/images/select", json={"path": "images/0002.png"})
    job = _run(client, pid, transparent=False, seed=-1, batch_count=1)
    assert job["status"] == "done", job
    images = _images(client, pid)
    assert images["candidates"] == ["images/0001.png", "images/0002.png", "images/0003.png"]
    assert images["chosen"] == "images/0003.png" and images["last_seeds"] == [777]
    assert _color_type((prop_dir / "images" / "0003.png").read_bytes()) == 2
    assert sorted(p.name for p in (prop_dir / "images" / ".given").iterdir()) == ["0001", "0002", "0003"]
    assert _launcher(client, "/fake/qwen")[-1]["prompt"] == "a red barrel"       # no words about a background either


def test_qwen_errors_cancel_and_restart(api: Any) -> None:
    app, client, _ = api
    pid = _prop(client)
    job = _run(client, pid, prompt="a lamp FAKE_OOM")
    assert job["status"] == "error" and job["errors"][pid] == qwen.OUT_OF_MEMORY
    job = _run(client, pid, prompt="a lamp FAKE_ERROR:The prompt is too long")
    assert job["errors"][pid] == "The prompt is too long"
    # The graphics card broke: the tool is started afresh before the next prop (here: stopped after the job).
    job = _run(client, pid, prompt="a lamp FAKE_RESTART")
    assert job["status"] == "error" and "starts again" in job["errors"][pid]
    _poll(client, "/api/status", lambda s: s["tools"][QWEN]["state"], {"stopped"})

    # A CUDA error inside ggml ends the process at once (exit code 3): the connection drops without an answer. The
    # prop says so, and the tool is started afresh (here: stopped after the job, started by the next one).
    job = _run(client, pid, prompt="a lamp FAKE_CRASH")
    assert job["status"] == "error" and job["errors"][pid] == qwen.CRASHED
    _poll(client, "/api/status", lambda s: s["tools"][QWEN]["state"], {"stopped"})

    # stable-diffusion.cpp draws every picture first and decodes them all at the end: a cancel while drawing keeps
    # none of them.
    client.patch(f"/api/projects/Demo/props/{pid}", json={"images": {"settings": {
        "prompt": "a lamp FAKE_SLOW", "batch_count": 5}}})
    started = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [pid]}).json()
    _poll(client, f"/api/jobs/{started['id']}", lambda j: j["progress"]["message"].startswith(
        ("Drawing picture 1 of 5", "Drawing picture 2 of 5")), {True}, timeout=30)
    # the launcher's words, with the time left it reckons ("about 1 min left": the picture takes minutes)
    message = client.get(f"/api/jobs/{started['id']}").json()["progress"]["message"]
    assert " · step " in message and message.endswith(" min left"), message
    client.post(f"/api/jobs/{started['id']}/cancel")
    job = _poll(client, f"/api/jobs/{started['id']}", lambda j: j["status"], {"done", "error", "cancelled"})
    assert job["status"] == "cancelled" and job["errors"] == {}
    assert _images(client, pid)["candidates"] == []
    # A cancel while they are decoded keeps the ones decoded so far.
    client.patch(f"/api/projects/Demo/props/{pid}", json={"images": {"settings": {
        "prompt": "a lamp FAKE_LONG_DECODE", "batch_count": 4}}})
    started = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [pid]}).json()
    deadline = time.monotonic() + 30
    while not str((_launcher(client, "/spellcast/progress") or {}).get("message")).startswith(
            "Finishing picture 2 of 4"):
        assert time.monotonic() < deadline, "the fake launcher never decoded its second picture"
        time.sleep(0.05)
    client.post(f"/api/jobs/{started['id']}/cancel")        # the adapter passes it on at its next poll (1 s)
    job = _poll(client, f"/api/jobs/{started['id']}", lambda j: j["status"], {"done", "error", "cancelled"})
    assert job["status"] == "cancelled" and job["errors"] == {}
    kept = len(_images(client, pid)["candidates"])
    assert 1 <= kept < 4
    # the launcher's "Stopped after N of M pictures." reaches the page (job.warnings): the job's last words are its own
    assert job["warnings"] == {pid: f"Stopped after {kept} of 4 pictures."}


def test_what_qwen_says_about_its_pictures_reaches_the_page(api: Any) -> None:
    """A launcher warning (e.g. "this took longer: fewer example pictures are faster") is kept with the job for the
    page; the job's progress ends as "Done."."""
    app, client, _ = api
    pid = _prop(client)
    job = _run(client, pid, prompt="a lamp FAKE_WARN", batch_count=2)
    assert job["status"] == "done" and job["progress"]["message"] == "Done."
    assert job["warnings"] == {pid: "The pictures were made in parts (fake)."}
    job = _run(client, pid, prompt="a lamp")
    assert job["warnings"] == {}


def test_describe_leaves_the_example_pictures_out_and_keeps_them(api: Any) -> None:
    """The page's "Describe" mode sends "use_references": false: the pictures come from the words alone, and the
    example pictures stay with the prop for "Describe + example"."""
    app, client, _ = api
    pid = _prop(client)
    client.post(f"/api/projects/Demo/props/{pid}/images/references", files=[("files", ("front.png", PNG, "image/png"))])
    client.patch(f"/api/projects/Demo/props/{pid}", json={"images": {"settings": {"prompt": "a stool"}}})
    job = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [pid],
                                         "params": {"use_references": False}}).json()
    job = _poll(client, f"/api/jobs/{job['id']}", lambda j: j["status"], {"done", "error", "cancelled"})
    assert job["status"] == "done", job
    assert _launcher(client, "/fake/qwen")[-1]["ref_images"] == []
    assert _images(client, pid)["references"] == ["reference/reference_001.png"]
    job = _run(client, pid, prompt="a stool")               # without it: the example pictures go along
    assert job["status"] == "done" and len(_launcher(client, "/fake/qwen")[-1]["ref_images"]) == 1
    # The page saves the mode with the prop (images.settings.use_references), so Describe stays Describe after a
    # reload: a job without params follows it.
    job = _run(client, pid, prompt="a stool", use_references=False)
    assert job["status"] == "done" and _launcher(client, "/fake/qwen")[-1]["ref_images"] == []
    assert _images(client, pid)["settings"]["use_references"] is False
    job = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [pid],
                                         "params": {"use_references": True}}).json()    # the job's own word wins
    job = _poll(client, f"/api/jobs/{job['id']}", lambda j: j["status"], {"done", "error", "cancelled"})
    assert job["status"] == "done" and len(_launcher(client, "/fake/qwen")[-1]["ref_images"]) == 1
    bad = client.patch(f"/api/projects/Demo/props/{pid}", json={"images": {"settings": {"use_references": "no"}}})
    assert bad.status_code == 400


def test_re_render_viewport_shows_the_picture_first(api: Any) -> None:
    """The page's "Re-Render Viewport": the job's from_viewport - one of the prop's pictures, or the marked copy the
    page uploads (images/viewport) - is the first picture the model is shown, alone or before the example pictures
    ("use_references"), at most 10 in all; it is the job's own, never saved with the prop; another file is refused."""
    app, client, _ = api
    pid = _prop(client)
    prop_dir = app.state.hub.store.prop_dir("Demo", pid)
    client.post(f"/api/projects/Demo/props/{pid}/images/references", files=[("files", ("front.png", PNG, "image/png"))])
    assert _run(client, pid, prompt="make it brown")["status"] == "done"
    picture = _images(client, pid)["chosen"]
    path = lambda rel: str((prop_dir / rel).resolve())      # noqa: E731

    def render(**params: Any) -> dict[str, Any]:
        job = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [pid],
                                             "params": params}).json()
        return _poll(client, f"/api/jobs/{job['id']}", lambda j: j["status"], {"done", "error", "cancelled"})

    assert render(from_viewport=picture, use_references=False)["status"] == "done"
    assert _launcher(client, "/fake/qwen")[-1]["ref_images"] == [path(picture)]
    assert render(from_viewport=picture, use_references=True)["status"] == "done"
    assert _launcher(client, "/fake/qwen")[-1]["ref_images"] == [path(picture), path("reference/reference_001.png")]
    assert _images(client, pid)["chosen"] == "images/0003.png", "the new picture shows"

    # With marks: the page uploads the picture with them, and the job names that copy.
    up = client.post(f"/api/projects/Demo/props/{pid}/images/viewport", files={"file": ("viewport.png", PNG, "image/png")})
    assert up.status_code == 200 and up.json() == {"path": "viewport/viewport.png"}
    assert (prop_dir / "viewport" / "viewport.png").read_bytes() == PNG
    assert render(from_viewport="viewport/viewport.png", use_references=False)["status"] == "done"
    assert _launcher(client, "/fake/qwen")[-1]["ref_images"] == [path("viewport/viewport.png")]
    assert client.post(f"/api/projects/Demo/props/{pid}/images/viewport",
                       files={"file": ("v.jpg", JPG, "image/jpeg")}).status_code == 415
    assert client.post(f"/api/projects/Demo/props/{pid}/images/viewport",
                       files={"file": ("v.png", b"", "image/png")}).status_code == 400
    assert client.post("/api/projects/Demo/props/p_404404/images/viewport",
                       files={"file": ("v.png", PNG, "image/png")}).status_code == 404

    # Only the prop's own pictures and its marked copy: an example picture, a missing one, a way out are refused.
    for bad in ("reference/reference_001.png", "images/0099.png", "../../elsewhere.png", 7):
        job = render(from_viewport=bad)
        assert job["status"] == "error" and "re-render" in job["errors"][pid], bad
    # The job's own: a page cannot save it with the prop (it would re-render that picture every time after).
    assert client.patch(f"/api/projects/Demo/props/{pid}",
                        json={"images": {"settings": {"from_viewport": picture, "viewport_shift": 0.1}}}).status_code == 200
    assert not {"from_viewport", "viewport_shift"} & set(_images(client, pid)["settings"])
    # Shown slid by the offset slider (a tileable picture): the launcher is told to slide what it makes back.
    assert render(from_viewport=picture, use_references=False, viewport_shift=0.1)["status"] == "done"
    told = _launcher(client, "/fake/qwen")[-1]
    assert (told["slide_back"], told["keep_place"]) == (0.1, False), "not a tileable prop: slid back, laid nowhere"
    assert render(from_viewport=picture, use_references=False)["status"] == "done"
    assert _launcher(client, "/fake/qwen")[-1]["slide_back"] == 0.0
    client.patch(f"/api/projects/Demo/props/{pid}", json={"images": {"settings": {"tileable": True}}})
    assert render(from_viewport=picture, use_references=False, viewport_shift=0)["status"] == "done"
    told = _launcher(client, "/fake/qwen")[-1]
    assert (told["tileable"], told["slide_back"], told["keep_place"]) == (True, 0.0, True)
    client.patch(f"/api/projects/Demo/props/{pid}", json={"images": {"settings": {"tileable": False}}})

    # Ten example pictures and the viewport's: the first nine go along, and the page is told.
    client.post(f"/api/projects/Demo/props/{pid}/images/references",
                files=[("files", (f"{i}.png", PNG, "image/png")) for i in range(9)])
    job = render(from_viewport=picture, use_references=True)
    assert job["status"] == "done" and "left out" in job["warnings"][pid]
    sent = _launcher(client, "/fake/qwen")[-1]["ref_images"]
    assert len(sent) == MAX_REFERENCES and sent[0] == path(picture)
    assert sent[1:] == [path(r) for r in _images(client, pid)["references"][:MAX_REFERENCES - 1]]


def test_a_smart_mask_job_names_its_picture_and_keeps_a_tileable_one(api: Any) -> None:
    """The page's Smart Mask: the mask goes up with the tinted picture (images/viewport's "mask" part), and the job
    names the picture brushed over (viewport_mask_of, the job's own). The launcher is told that picture and the
    mask, and the tinted copy as what the drawing may be held to, whatever the prop: how the change is drawn is the
    launcher's to say. A mask or a picture that is not there is refused in words."""
    app, client, _ = api
    pid = _prop(client)
    prop_dir = app.state.hub.store.prop_dir("Demo", pid)
    assert _run(client, pid, prompt="grass", tileable=True)["status"] == "done"
    picture = _images(client, pid)["chosen"]
    path = lambda rel: str((prop_dir / rel).resolve())      # noqa: E731
    viewport = f"/api/projects/Demo/props/{pid}/images/viewport"
    mask = PNG + b"where"

    def render(**params: Any) -> dict[str, Any]:
        job = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [pid],
                                             "params": {"use_references": False, **params}}).json()
        return _poll(client, f"/api/jobs/{job['id']}", lambda j: j["status"], {"done", "error", "cancelled"})

    # The tinted picture alone went up, no mask with it: refused.
    assert client.post(viewport, files={"file": ("viewport.png", PNG, "image/png")}).status_code == 200
    job = render(from_viewport="viewport/viewport.png", viewport_mask_of=picture)
    assert job["status"] == "error" and "Brush over the picture again" in job["errors"][pid], "no mask went up"
    up = client.post(viewport, files={"file": ("viewport.png", PNG, "image/png"), "mask": ("mask.png", mask, "image/png")})
    assert up.status_code == 200 and up.json() == {"path": "viewport/viewport.png", "mask": "viewport/mask.png"}
    assert (prop_dir / "viewport" / "mask.png").read_bytes() == mask
    assert render(from_viewport="viewport/viewport.png", viewport_shift=0.1, viewport_mask_of=picture)["status"] == "done"
    told = _launcher(client, "/fake/qwen")[-1]
    assert (told["changed_image"], told["changed_mask"]) == (path(picture), path("viewport/mask.png"))
    assert (told["tileable"], told["keep_place"], told["slide_back"]) == (True, False, 0.1)
    assert told["ref_images"] == [path("viewport/viewport.png")], "the model is shown the tinted picture"
    assert (told["init_image"], told["mask_image"], told["strength"]) == (path("viewport/viewport.png"), path("viewport/mask.png"), 1.0)
    # The same picture re-rendered without a mask: tileable as the prop is, nothing kept.
    assert render(from_viewport="viewport/viewport.png", viewport_shift=0.1)["status"] == "done"
    told = _launcher(client, "/fake/qwen")[-1]
    assert (told["tileable"], told["keep_place"]) == (True, True) and not {"changed_image", "changed_mask", "init_image", "mask_image"} & set(told)
    # The mask belongs to the tinted copy the page sent with it: over another picture of the job's it is refused.
    job = render(from_viewport=picture, viewport_mask_of=picture)
    assert job["status"] == "error" and "Brush over the picture again" in job["errors"][pid]
    # A picture uploaded without a mask leaves the last mask for the job that names it (it may still be waiting).
    assert client.post(viewport, files={"file": ("viewport.png", PNG, "image/png")}).json() == {"path": "viewport/viewport.png"}
    assert (prop_dir / "viewport" / "mask.png").read_bytes() == mask
    for bad, status in ((JPG, 415), (b"", 400)):
        assert client.post(viewport, files={"file": ("viewport.png", PNG, "image/png"),
                                            "mask": ("mask.png", bad, "image/png")}).status_code == status
    # Only over one of the prop's own pictures: the tinted copy, a missing one, a way out are refused.
    for bad in ("viewport/viewport.png", "images/0099.png", "../../elsewhere.png", 7):
        job = render(from_viewport="viewport/viewport.png", viewport_mask_of=bad)
        assert job["status"] == "error" and "Brush over the picture again" in job["errors"][pid], bad
    # A prop that is not tileable: the same request, said not to be tileable.
    assert render(from_viewport="viewport/viewport.png", viewport_mask_of=picture, tileable=False)["status"] == "done"
    told = _launcher(client, "/fake/qwen")[-1]
    assert (told["changed_image"], told["changed_mask"], told["tileable"]) == (path(picture), path("viewport/mask.png"), False)
    assert (told["init_image"], told["mask_image"]) == (path("viewport/viewport.png"), path("viewport/mask.png"))
    # The job's own: a page cannot save it with the prop (every later picture would be a masked one).
    assert client.patch(f"/api/projects/Demo/props/{pid}",
                        json={"images": {"settings": {"viewport_mask_of": picture}}}).status_code == 200
    assert "viewport_mask_of" not in _images(client, pid)["settings"]


def test_a_submitted_job_with_bad_qwen_settings_is_refused(api: Any) -> None:
    app, client, _ = api
    pid = _prop(client)
    refused = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [pid],
                                             "params": {"prompt": "x", "batch_count": 40}})
    assert refused.status_code == 400 and "Variations" in refused.json()["detail"]
    # a seed sd.cpp's int64 would wrap (and the page could not even say exactly)
    refused = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [pid],
                                             "params": {"prompt": "x", "seed": 1e20}})
    assert refused.status_code == 400
    assert "Seed must be a whole number from 0 to 4,294,967,295" in refused.json()["detail"]


def test_a_full_model_prop_waits_for_its_download_and_is_never_drawn_with_the_turbo(
        api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A prop set to the full model while its file is not in the models folder: its job is refused at once in the
    artist's words, and a job queued before the file went away fails with them (the launcher is never asked, so
    nothing is drawn with the Turbo instead). Once the file is there (at its pinned size) it runs with "full"."""
    app, client, _ = api
    pid = _prop(client)
    patched = client.patch(f"/api/projects/Demo/props/{pid}", json={"images": {"settings": {"qwen_model": "full"}}})
    assert patched.status_code == 200 and patched.json()["images"]["settings"]["qwen_model"] == "full"
    refused = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [pid],
                                             "params": {"prompt": "a barrel"}})
    assert refused.status_code == 400 and refused.json()["detail"] == f"Barrel: {qwen.FULL_MISSING}"
    assert "not downloaded yet" in qwen.FULL_MISSING and "Turbo" in qwen.FULL_MISSING
    adapter = app.state.hub.adapters[QWEN]
    assert adapter.models_dir == app.state.hub.qwen_full_download.folder      # the download goes where it reads
    monkeypatch.setitem(models.QWEN_DENOISERS, "Q4_K_M", (5, "0" * 64))
    adapter.models_dir.mkdir(parents=True, exist_ok=True)
    full = adapter.models_dir / "qwen-image-2.1-Q4_K_M.gguf"
    full.write_bytes(b"1234")                                                  # unfinished: not its size yet
    assert not adapter.full_model_on_disk()
    full.write_bytes(b"12345")
    assert adapter.full_model_on_disk()
    job = _run(client, pid, prompt="a barrel")
    assert job["status"] == "done", job
    assert _launcher(client, "/fake/qwen")[-1]["model"] == "full"
    # the file goes away while a job waits: the job fails with the same words, the launcher is not asked
    asked = len(_launcher(client, "/fake/qwen"))
    ctx = StageContext(project="Demo", prop_id=pid, prop_dir=app.state.hub.store.prop_dir("Demo", pid),
                       base_url="http://127.0.0.1:9", client=None, settings={},      # type: ignore[arg-type]
                       report=lambda f, m: None, cancelled=lambda: False)
    full.unlink()
    with pytest.raises(StageError, match="not downloaded yet"):
        asyncio.run(adapter.run_stage(ctx, {"images": _images(client, pid)}, {"prompt": "a barrel",
                                                                              "qwen_model": "full"}))
    assert len(_launcher(client, "/fake/qwen")) == asked


def test_the_albedo_call_for_omnitexture_and_its_errors(tmp_path: Path) -> None:
    """qwen.albedo (POST /spellcast/albedo) and albedo_ready (health "albedo"), the launcher played by a transport:
    the answer's picture, and each failure as the stage sees it (a Qwen-Image to start again, out of memory, a
    problem in words, a crash) - as for its pictures, with the add-on's own words for memory."""
    health = {"albedo": True}
    answers: list[Any] = []
    posted: list[dict[str, Any]] = []

    def launcher(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/spellcast/health":
            return httpx.Response(200, json={"ready": True, **health})
        posted.append(json.loads(request.content))
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        status, body = answer
        if status == 200:
            Path(body["image"]).write_bytes(PNG)
        return httpx.Response(status, json=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(launcher))
    ctx = StageContext(project="P", prop_id="p1", prop_dir=tmp_path, base_url="http://qwen", client=client,
                       settings={}, report=lambda f, m: None, cancelled=lambda: False, stage="omnitexture")
    picture, out = tmp_path / "answer.png", tmp_path / "flat" / "albedo.png"
    out.parent.mkdir()
    assert asyncio.run(qwen.albedo_ready(ctx)) is True
    health["albedo"] = False
    assert asyncio.run(qwen.albedo_ready(ctx)) is False

    answers.append((200, {"image": str(out), "seconds": 3.8}))
    assert asyncio.run(qwen.albedo(ctx, picture, out, "turbo")) == out
    assert posted == [{"picture": str(picture.resolve()), "out": str(out.resolve()), "model": "turbo"}]
    failures: list[tuple[Any, type[StageError], str]] = [
        ((500, {"detail": "The card broke.", "kind": "restart"}), ToolNeedsRestart, "The card broke."),
        ((409, {"detail": "Still busy.", "kind": "busy"}), ToolNeedsRestart, "Still busy."),
        ((500, {"detail": "ggml_cuda: failed to allocate", "kind": "oom"}), StageError, qwen.ALBEDO_OUT_OF_MEMORY),
        ((400, {"detail": "The picture's sides must be multiples of 32 pixels.", "kind": "input"}), StageError,
         "The picture's sides must be multiples of 32 pixels."),
        ((502, "not json"), StageError, "could not take the lighting off a picture (HTTP 502)"),
        (httpx.ConnectError("refused"), StageError, qwen.UNREACHABLE),
        (httpx.ReadTimeout("slow"), ToolNeedsRestart, "did not take the lighting off a picture within 5 minutes"),
        (httpx.RemoteProtocolError("gone"), ToolNeedsRestart, qwen.CRASHED),
    ]
    for answer, error, words in failures:
        answers.append(answer)
        with pytest.raises(error) as raised:
            asyncio.run(qwen.albedo(ctx, picture, out, "turbo"))
        assert type(raised.value) is error and words in str(raised.value), answer
    # the launcher down: no add-on, not an error
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    asyncio.run(client.aclose())
    ctx.client = httpx.AsyncClient(transport=httpx.MockTransport(down))
    assert asyncio.run(qwen.albedo_ready(ctx)) is False
    asyncio.run(ctx.client.aclose())


# ---- example pictures (images.references) ------------------------------------------------------------------

def test_references_are_added_capped_and_removed(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    pid = store.add_prop("P", "Crate")["id"]
    folder = store.prop_dir("P", pid) / "reference"
    prop = store.add_references("P", pid, [(PNG, "a.png"), (JPG, "b.jpg"), (WEBP, "c.webp")])
    assert prop["images"]["references"] == ["reference/reference_001.png", "reference/reference_002.jpg",
                                            "reference/reference_003.webp"]
    with pytest.raises(ProjectError, match="at most 10 example pictures, so 7 more") as e:
        store.add_references("P", pid, [(PNG, f"{i}.png") for i in range(8)])
    assert e.value.status_code == 400 and len(list(folder.iterdir())) == 3         # nothing was stored
    with pytest.raises(ProjectError, match="b.txt: This file is not a PNG"):
        store.add_references("P", pid, [(PNG, "a.png"), (b"hello", "b.txt")])
    assert len(list(folder.iterdir())) == 3
    store.add_references("P", pid, [(PNG, f"{i}.png") for i in range(7)])
    with pytest.raises(ProjectError, match="Remove one"):
        store.add_references("P", pid, [(PNG, "one more.png")])
    assert len(store.get_prop("P", pid)["images"]["references"]) == MAX_REFERENCES == 10

    prop = store.remove_references("P", pid, "reference/reference_002.jpg")
    assert "reference/reference_002.jpg" not in prop["images"]["references"]
    assert not (folder / "reference_002.jpg").exists() and len(prop["images"]["references"]) == 9
    (store.prop_dir("P", pid) / "keep.png").write_bytes(PNG)
    for not_one in ("reference/reference_002.jpg", "keep.png", "../keep.png", "images/0001.png"):
        assert len(store.remove_references("P", pid, not_one)["images"]["references"]) == 9
    assert (store.prop_dir("P", pid) / "keep.png").exists()
    assert store.remove_references("P", pid)["images"]["references"] == [] and not list(folder.iterdir())
    # The next one gets a new name (no stale browser cache).
    assert store.add_references("P", pid, [(PNG, "a.png")])["images"]["references"] == ["reference/reference_001.png"]


def test_older_projects_and_qwen_settings(tmp_path: Path) -> None:
    """The one "reference" of an older project becomes its first example picture; Qwen's settings are filled in,
    repaired and checked."""
    store = ProjectStore(tmp_path / "projects")
    store.create_project("P")
    pid = store.add_prop("P", "Crate")["id"]
    project = json.loads((store.root / "P" / "project.json").read_text(encoding="utf-8"))
    images = project["props"][0]["images"]
    del images["references"]
    images["reference"] = "reference/reference_004.png"
    for key in ("transparent", "qwen_steps", "qwen_model"):
        del images["settings"][key]
    (store.root / "P" / "project.json").write_text(json.dumps(project), encoding="utf-8")
    images = store.get_prop("P", pid)["images"]
    assert images["references"] == ["reference/reference_004.png"] and "reference" not in images
    assert {k: images["settings"][k] for k in ("transparent", "qwen_steps")} == {"transparent": False, "qwen_steps": 25}
    assert "qwen_cfg" not in images["settings"]
    assert images["settings"]["qwen_model"] == "turbo"      # a prop from before the choice: the Turbo, the default

    project["props"][0]["images"].update(reference=None, references=[
        "reference/a.png", "reference/a.png", "../x.png", 5, "images/0001.png", "reference/../x.png", "C:/x.png"])
    project["props"][0]["images"]["settings"].update(transparent="yes", tileable=1, qwen_steps=0, qwen_cfg="x",
                                                     qwen_model="fast")
    (store.root / "P" / "project.json").write_text(json.dumps(project), encoding="utf-8")
    images = store.get_prop("P", pid)["images"]
    assert images["references"] == ["reference/a.png"]
    assert {k: images["settings"][k] for k in ("transparent", "tileable", "qwen_steps", "qwen_model")} == {
        "transparent": False, "tileable": False, "qwen_steps": 25, "qwen_model": "turbo"}
    assert "qwen_cfg" not in images["settings"]             # not a setting any more, whatever the file says

    # qwen_cfg from a page loaded before the update: left out, whatever its value (never a 400)
    prop = store.patch_prop("P", pid, {"images": {"settings": {"transparent": False, "qwen_steps": 30.0,
                                                               "qwen_cfg": 7}}})
    assert (prop["images"]["settings"]["transparent"], prop["images"]["settings"]["qwen_steps"]) == (False, 30)
    for gone in (7, 25, True, "x"):
        assert "qwen_cfg" not in store.patch_prop("P", pid, {"images": {"settings": {"qwen_cfg": gone}}})[
            "images"]["settings"]
    assert "qwen_cfg" not in (store.root / "P" / "project.json").read_text(encoding="utf-8")
    assert store.patch_prop("P", pid, {"images": {"settings": {"tileable": True}}})["images"]["settings"]["tileable"] is True
    for bad in ({"transparent": "yes"}, {"tileable": "yes"}, {"qwen_steps": 0}, {"qwen_steps": 1.5}, {"qwen_steps": True},
                {"qwen_model": "fast"}, {"qwen_model": None}):
        with pytest.raises(ProjectError) as e:
            store.patch_prop("P", pid, {"images": {"settings": bad}})
        assert e.value.status_code == 400
    with pytest.raises(ProjectError, match="Turbo, Full or FLUX.3 Klein"):
        store.patch_prop("P", pid, {"images": {"settings": {"qwen_model": "Full"}}})
    assert store.patch_prop("P", pid, {"images": {"settings": {"qwen_model": "full"}}})["images"]["settings"][
        "qwen_model"] == "full"
    # A new prop takes the last one's settings, never its example pictures.
    store.add_references("P", pid, [(PNG, "a.png")])
    second = store.add_prop("P", "Lamp")
    assert second["images"]["references"] == [] and second["images"]["settings"]["transparent"] is False
    assert second["images"]["settings"]["qwen_model"] == "full"      # the model too, like every setting
    assert DEFAULT_STAGE_SETTINGS["images"]["qwen_model"] == "turbo"
    # A draft with only example pictures is not empty: an untitled project of it is not tidied away.
    untitled = store.new_project()["id"]
    draft = store.ensure_draft(untitled)
    assert _is_empty(store.get_project(untitled))
    store.add_references(untitled, draft["id"], [(PNG, "a.png")])
    assert not _is_empty(store.get_project(untitled))


def test_a_project_from_before_qwens_own_recipe_loads_and_runs(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A project saved while Qwen-Image was sampled with guidance: every prop has qwen_cfg 6.0 and that recipe's
    default of 20 steps, and may have an "Avoid" text. It loads; qwen_cfg is no setting any more; the old default of
    20 steps follows today's default, steps the artist set stay; "Avoid" is kept (Forge Neo's) and not sent. Its props
    had no choice of model: they draw with the Turbo, and their steps are the full model's again once it is picked."""
    app, client, _ = api
    old, tuned = _prop(client, name="Barrel"), _prop(client, name="Lamp")
    file = app.state.hub.store.root / "Demo" / "project.json"
    project = json.loads(file.read_text(encoding="utf-8"))
    for prop, steps in zip(project["props"], (20, 32)):
        prop["images"]["settings"].update(prompt="a barrel", negative_prompt="blurry, text", qwen_steps=steps,
                                          qwen_cfg=6.0)
    file.write_text(json.dumps(project), encoding="utf-8")

    loaded = {p["id"]: p["images"]["settings"] for p in client.get("/api/projects/Demo").json()["props"]}
    assert all("qwen_cfg" not in s and s["negative_prompt"] == "blurry, text" for s in loaded.values())
    assert (loaded[old]["qwen_steps"], loaded[tuned]["qwen_steps"]) == (qwen.DEFAULT_STEPS, 32) == (25, 32)
    assert '"qwen_cfg"' in file.read_text(encoding="utf-8")                # reading changes nothing on disk

    assert loaded[old]["qwen_model"] == loaded[tuned]["qwen_model"] == "turbo"      # no choice then: the default

    def run_both() -> list[tuple[str, int, float, str]]:
        job = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [old, tuned]}).json()
        job = _poll(client, f"/api/jobs/{job['id']}", lambda j: j["status"], {"done", "error", "cancelled"})
        assert job["status"] == "done", job
        return [(s["model"], s["steps"], s["cfg_scale"], s["negative_prompt"])
                for s in _launcher(client, "/fake/qwen")[-2:]]

    assert run_both() == [("turbo", 8, 1.0, ""), ("turbo", 8, 1.0, "")]
    # the full model downloaded (a stand-in of its pinned size) and picked: the steps each prop has
    adapter = app.state.hub.adapters[QWEN]
    monkeypatch.setitem(models.QWEN_DENOISERS, "Q4_K_M", (5, "0" * 64))
    adapter.models_dir.mkdir(parents=True, exist_ok=True)
    (adapter.models_dir / "qwen-image-2.1-Q4_K_M.gguf").write_bytes(b"12345")
    for pid in (old, tuned):
        client.patch(f"/api/projects/Demo/props/{pid}", json={"images": {"settings": {"qwen_model": "full"}}})
    assert run_both() == [("full", 25, 1.0, ""), ("full", 32, 1.0, "")]
    saved = [p["images"]["settings"] for p in json.loads(file.read_text(encoding="utf-8"))["props"]]
    assert all("qwen_cfg" not in s and s["negative_prompt"] == "blurry, text" for s in saved)
    assert [s["qwen_steps"] for s in saved] == [25, 32]
    # A new prop takes the last one's settings, and 20 steps set now are the artist's: they stay.
    new = client.post("/api/projects/Demo/props", json={"name": "Stool"}).json()["images"]["settings"]
    assert "qwen_cfg" not in new and (new["qwen_steps"], new["negative_prompt"]) == (32, "blurry, text")
    client.patch(f"/api/projects/Demo/props/{old}", json={"images": {"settings": {"qwen_steps": 20}}})
    assert _images(client, old)["settings"]["qwen_steps"] == 20


def test_reference_routes(api: Any) -> None:
    app, client, _ = api
    pid = _prop(client)
    base = f"/api/projects/Demo/props/{pid}"
    three = [("files", (f"{i}.png", PNG, "image/png")) for i in range(3)]
    assert len(client.post(f"{base}/images/references", files=three).json()["images"]["references"]) == 3
    too_many = client.post(f"{base}/images/references", files=[("files", (f"{i}.png", PNG)) for i in range(8)])
    assert too_many.status_code == 400 and "at most 10" in too_many.json()["detail"]
    flood = client.post(f"{base}/images/references", files=[("files", (f"{i}.png", PNG)) for i in range(12)])
    assert flood.status_code == 400 and "at most 10" in flood.json()["detail"]
    wrong = client.post(f"{base}/images/references", files=[("files", ("evil.png", b"MZ\x90\x00"))])
    assert wrong.status_code == 415
    one = client.delete(f"{base}/images/references", params={"path": "reference/reference_002.png"}).json()
    assert one["images"]["references"] == ["reference/reference_001.png", "reference/reference_003.png"]
    # The routes of the one reference picture: add one, remove all.
    alias = client.post(f"{base}/reference", files={"file": ("r.webp", WEBP)}).json()
    assert alias["images"]["references"][-1] == "reference/reference_004.webp"
    assert client.delete(f"{base}/reference").json()["images"]["references"] == []
    client.post(f"{base}/images/references", files=three)
    assert client.delete(f"{base}/images/references").json()["images"]["references"] == []
    assert client.post("/api/projects/Demo/props/p_000000/images/references", files=three).status_code == 404
    url = f"/files/Demo/props/{pid}/reference/reference_001.png"
    assert client.get(url).status_code == 404                  # its file went with it


def test_the_config_file_is_one_click_away(tmp_path: Path, root_config: Callable[..., None],
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """"Show the file" (the Images gate, the licence line): File Explorer on the Spellcast3D folder with
    spellcast3d-config.ini selected, made first when it is missing."""
    from spellcast.install import state

    shown: list[tuple[Path, bool]] = []
    app = create_app({QWEN: FakeQwen(tmp_path)}, settings=make_settings(tmp_path),
                     store=ProjectStore(tmp_path / "projects"), static_dir=_static(tmp_path),
                     logs_dir=tmp_path / "logs", gpu_probe=lambda: RTX_3060,
                     show_in_explorer=lambda path, select: shown.append((path, select)) or True)
    monkeypatch.delenv("SPELLCAST_NO_EXPLORER", raising=False)
    monkeypatch.setattr(state, "load", lambda: {})
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as local:
        assert local.post("/api/open-folder", json={"which": "config"}).json() == {"ok": True, "opened": True}
        assert shown == [(config.config_file(), True)]
        config.config_file().unlink()
        assert local.post("/api/open-folder", json={"which": "config"}).json()["opened"] is True
        assert config.config_file().read_bytes() == paths.CONFIG_TEMPLATE.read_bytes() and len(shown) == 2
        monkeypatch.setenv("SPELLCAST_NO_EXPLORER", "1")
        assert local.post("/api/open-folder", json={"which": "config"}).json()["opened"] is False
    with TestClient(app, base_url="http://127.0.0.1") as remote:        # another computer: never
        assert remote.post("/api/open-folder", json={"which": "config"}).status_code == 403


def test_a_mistake_in_the_file_is_in_the_hub_log(tmp_path: Path, root_config: Callable[..., None],
                                                 monkeypatch: pytest.MonkeyPatch,
                                                 caplog: pytest.LogCaptureFixture) -> None:
    root_config("forg")
    monkeypatch.setattr(gpu, "primary_gpu", lambda: RTX_3060)
    with caplog.at_level("WARNING", logger="spellcast.server"):
        create_app(settings=make_settings(tmp_path), store=ProjectStore(tmp_path / "projects"),
                   static_dir=_static(tmp_path), logs_dir=tmp_path / "logs", gpu_probe=lambda: RTX_3060)
    assert any('"forg" is not an image generator' in r.getMessage() for r in caplog.records)


def test_the_time_left_in_words() -> None:
    """A Qwen-Image picture takes minutes: the progress says roughly how many are left, once the launcher knows."""
    assert [qwen.time_left(eta) for eta in (None, True, "5", -1.0, float("nan"))] == [""] * 5
    assert [qwen.time_left(eta) for eta in (0, 0.4, 7.6, 42, 59.4)] == ["1 s left", "1 s left", "8 s left", "42 s left",
                                                                       "59 s left"]
    assert qwen.time_left(59.6) == "about 1 min left"
    assert (qwen.time_left(60), qwen.time_left(89), qwen.time_left(91), qwen.time_left(600)) == (
        "about 1 min left", "about 1 min left", "about 2 min left", "about 10 min left")


def test_the_picture_being_drawn_is_shown_while_it_forms(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The launcher's preview picture travels with the job's progress: a URL that changes with every new picture,
    served from memory; gone when the job ends. Always asked for: there is no setting that turns it off."""
    app, client, _ = api
    # The fake draws a picture in one second and has no preview at its first step: asked once a second, it could
    # be at that step every time.
    monkeypatch.setattr(qwen, "POLL_INTERVAL", 0.25)
    pid = _prop(client)
    assert "live_preview" not in client.get("/api/settings").json()
    client.patch(f"/api/projects/Demo/props/{pid}", json={"images": {"settings": {"prompt": "a lamp FAKE_SLOW",
                                                                                 "batch_count": 5}}})
    started = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [pid]}).json()
    assert "preview" not in started["progress"]
    for _attempt in range(20):      # again when the next picture began between the two requests (no preview then)
        job = _poll(client, f"/api/jobs/{started['id']}", lambda j: "preview" in j["progress"], {True})
        preview = job["progress"]["preview"]
        picture = client.get(preview["url"])
        if picture.status_code == 200:
            break
    assert preview["url"].startswith(f"/api/jobs/{started['id']}/preview?v=") and preview["step"] >= 1
    assert "box" not in preview, "of the whole picture"
    assert picture.status_code == 200 and picture.headers["content-type"] == "image/png"
    assert picture.content.startswith(b"\x89PNG") and picture.headers["cache-control"] == "no-store"
    job = _poll(client, f"/api/jobs/{started['id']}", lambda j: j["status"], {"done", "error", "cancelled"})
    assert job["status"] == "done" and "preview" not in job["progress"]
    assert client.get(preview["url"]).status_code == 404
    assert _launcher(client, "/fake/qwen")[-1]["preview"] is True
    assert client.get("/api/jobs/nope/preview").status_code == 404

    # A window's request (Smart Mask on a large picture): its previews are of the window, and the job says where
    # in the picture that lies.
    client.patch(f"/api/projects/Demo/props/{pid}", json={"images": {"settings": {"prompt": "a lamp FAKE_SLOW FAKE_WINDOW",
                                                                                 "batch_count": 1}}})
    started = client.post("/api/jobs", json={"stage": "images", "project": "Demo", "prop_ids": [pid]}).json()
    job = _poll(client, f"/api/jobs/{started['id']}", lambda j: "preview" in j["progress"], {True})
    assert job["progress"]["preview"]["box"] == [0.5, 0.25, 1.25, 0.75]
    _poll(client, f"/api/jobs/{started['id']}", lambda j: j["status"], {"done", "error", "cancelled"})

    # A settings.json from before still says live_preview false: it is no setting any more, the preview is asked for
    assert client.patch("/api/settings", json={"live_preview": False}).status_code == 200
    job = _run(client, pid, prompt="a lamp")
    assert job["status"] == "done" and _launcher(client, "/fake/qwen")[-1]["preview"] is True
