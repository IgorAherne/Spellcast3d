"""Installing Qwen-Image 2.1 and picking the image generator: its id, port and folders, the component (no PyTorch,
its fingerprint without the model size, a new size fetched as a missing file), its pinned AI models (all sizes, linked
from the artist's HuggingFace cache under other names), stable-diffusion.cpp's runtime (Spellcast3D's own build from
its release on the author's fork, its zip kept and unpacked - whole over an install made with the official release,
whose leftovers go; NVIDIA's CUDA DLLs linked from a PyTorch venv when byte-identical, else from the official
release's cudart zip), the installer following spellcast3d-config.ini (only the configured generator, none for a
mistake), the package a switched-on profiler needs (no work of Qwen-Image's: the installer's last step, best effort,
never blocking or failing it nor taking the disk of a component after it), preflight (no PyTorch server for it;
huggingface.co and github.com when it downloads from them), and prepare_release's view of it.

Fast (no network: downloads are stand-ins; no GPU).
Run with the hub venv:  python -m pytest -p no:cacheprovider code/tests
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import zipfile
from pathlib import Path
from typing import Any, Callable

import pytest

from spellcast import config, gpu, paths
from spellcast.gpu import GpuInfo
from spellcast.install import components, models, preflight, progress, sdcpp, state, venvs
from spellcast.install import main as install_main
from spellcast.install.main import Pending
from spellcast.install.runner import Ctx, InstallError, Output

if str(paths.LAUNCHERS) not in sys.path:     # the launchers import their siblings as top-level modules
    sys.path.insert(0, str(paths.LAUNCHERS))

import qwen_sdcpp  # noqa: E402  (stdlib only: the launcher's ctypes mirror, which pins the same build)

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Spellcast3D is Windows-only")

TEMPLATE = paths.CONFIG_TEMPLATE.read_text(encoding="utf-8")
FORGE = TEMPLATE.replace("\nbackend = qwen ", "\n; backend = qwen ").replace("\n; backend = forge", "\nbackend = forge")
GIT = shutil.which("git") or (str(paths.PORTABLE_GIT) if paths.PORTABLE_GIT.exists() else None)
needs_repo = pytest.mark.skipif(GIT is None or not (paths.ROOT / ".git").exists(),
                                reason="not a git checkout of Spellcast3D (a release copy has no git data)")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _gpu(cc: float) -> GpuInfo:
    return GpuInfo(index=0, name="Test GPU", compute_cap=cc, vram_total_mb=8192, vram_used_mb=0,
                   driver_version="596.36")


def _qwen() -> components.Qwen:
    return next(c for c in components.COMPONENTS if c.id == paths.QWEN)     # type: ignore[return-value]


class _RecordingCtx(Ctx):
    """Records the commands instead of running them, and the printed lines."""

    def __init__(self, dry_run: bool = False) -> None:
        super().__init__(Output(), dry_run, Path(sys.executable))
        self.commands: list[tuple[list[str], Path]] = []
        self.lines: list[str] = []
        self.out.line = self.lines.append   # type: ignore[method-assign]

    def run(self, argv: list[str], *, cwd: Path, env: dict[str, str], fail: str,
            notes: tuple[tuple[str, str], ...] = ()) -> None:
        self.commands.append(([str(a) for a in argv], cwd))


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """spellcast3d-config.ini in tmp_path (missing until written)."""
    path = tmp_path / "spellcast3d-config.ini"
    monkeypatch.setattr(config, "CONFIG_FILE", path)
    monkeypatch.delenv("SPELLCAST_CONFIG", raising=False)
    config._cache.clear()

    def write(text: str) -> None:
        path.write_text(text, encoding="utf-8")
        config._cache.clear()

    return write


@pytest.fixture
def qwen_here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Qwen's folders in tmp_path: models/qwen-image, code/venvs (with qwen/sdcpp) and code/venvs/downloads."""
    monkeypatch.setattr(paths, "VENVS", tmp_path / "venvs")
    monkeypatch.setattr(paths, "QWEN_SDCPP", tmp_path / "venvs" / "qwen" / "sdcpp")
    monkeypatch.setattr(paths, "DOWNLOADS", tmp_path / "venvs" / "downloads")
    monkeypatch.setattr(components.Qwen, "models_dir", tmp_path / "models" / "qwen-image")
    monkeypatch.setattr(models, "user_hf_caches", lambda: [])
    monkeypatch.setattr(models, "EXTRA_SOURCES", [])
    monkeypatch.delenv("SPELLCAST_MODELS_FROM", raising=False)
    return tmp_path


def _zip(path: Path, files: dict[str, bytes], release: sdcpp.Release = sdcpp.ENGINE) -> sdcpp.Archive:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    data = path.read_bytes()
    return sdcpp.Archive(release, path.name, len(data), _sha(data), {name: len(d) for name, d in files.items()})


@pytest.fixture
def small_runtime(qwen_here: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Small stand-ins for stable-diffusion.cpp's two zips (the engine's, of Spellcast3D's release; the CUDA runtime's,
    of the official one), in code/venvs/downloads, and the CUDA DLLs' pins."""
    runtime = {"stable-diffusion.dll": b"sd" * 50, "ggml-cuda.dll": b"cuda" * 40, "ggml.txt": b"MIT"}
    cuda = {"cudart64_12.dll": b"R" * 30, "cublas64_12.dll": b"B" * 70, "cublasLt64_12.dll": b"L" * 90}
    sd = _zip(paths.DOWNLOADS / "sd-test.zip", runtime)
    cudart = _zip(paths.DOWNLOADS / "cudart-test.zip", cuda, sdcpp.OFFICIAL)
    monkeypatch.setattr(sdcpp, "SDCPP", sd)
    monkeypatch.setattr(sdcpp, "CUDART", cudart)
    monkeypatch.setattr(sdcpp, "CUDA_DLLS", {name: (len(data), _sha(data)) for name, data in cuda.items()})
    return {"runtime": runtime, "cuda": cuda, "sd": sd, "cudart": cudart}


def _torch_venv(venv: Path, files: dict[str, bytes]) -> None:
    lib = venv / "Lib" / "site-packages" / "torch" / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (lib / name).write_bytes(data)


# ---- ids, paths, ports ---------------------------------------------------------------------------------------

def test_qwen_has_its_id_port_folders_and_is_the_default_image_tool() -> None:
    assert paths.QWEN == "qwen" and paths.QWEN in paths.TOOL_IDS and paths.QWEN in paths.ALL_VENV_IDS
    assert dict(paths.STAGES)["images"] == paths.QWEN and paths.STAGE_TOOLS["images"] == ("qwen", "forge")
    assert paths.MODELS_QWEN == paths.MODELS / "qwen-image" and paths.MODELS_QWEN in models.FOLDERS
    assert paths.QWEN_SDCPP == paths.venv_dir("qwen") / "sdcpp" and paths.DOWNLOADS == paths.VENVS / "downloads"
    assert paths.QWEN not in paths.TOOL_DIRS                                      # no git submodule
    assert paths.CONFIG_FILE == paths.ROOT / "spellcast3d-config.ini" == config.CONFIG_FILE
    assert json.loads(paths.DEFAULTS_FILE.read_text(encoding="utf-8"))["ports"]["qwen"] == 7776
    assert (paths.MODELS_QWEN / ".gitkeep").is_file() or not (paths.ROOT / ".git").exists()


def test_qwen_runs_from_a_gtx_10xx_without_pytorch() -> None:
    q = _qwen()
    assert (q.title, q.uses_gpu, q.needs_torch) == ("Qwen-Image 2.1", True, False)
    assert components.Forge.needs_torch and components.Pixal3D.needs_torch and not components.Imesh.needs_torch
    for cc in (6.1, 7.5, 8.6, 12.0):
        assert q.supported(_gpu(cc)) == (True, "")
    ok, why = q.supported(_gpu(5.2))
    assert not ok and "(Test GPU) is too old for Qwen-Image 2.1" in why and "GTX 1060" in why and "sm_" not in why
    ok, why = gpu.qwen_supported(None)
    assert not ok and "No NVIDIA graphics card" in why and "installed by itself" in why
    assert "research licence" in q.about and "commercial work" in q.about and "no PyTorch" in q.about


# ---- the component ------------------------------------------------------------------------------------------------

def test_the_model_size_follows_the_config_but_not_the_fingerprint(qwen_here: Path, cfg: Callable[[str], None]) -> None:
    q = _qwen()                                                        # qwen_here: an empty models folder
    cfg(TEMPLATE)
    before = q.fingerprint()
    # the Turbo's denoiser (the default model) and the shared files; never the full model's (an optional download)
    assert set(q.manifest) == {"qwen-image-2.1-turbo-Q4_K_M.gguf", models.QWEN_VAE, models.QWEN_TEXT_ENCODER,
                               models.QWEN_VISION, *models.ALBEDO_PINS}
    assert round(q.models_gb, 2) == 12.39
    cfg(TEMPLATE.replace("quality = Q4_K_M", "quality = Q8_0"))
    assert "qwen-image-2.1-turbo-Q8_0.gguf" in q.manifest and "qwen-image-2.1-turbo-Q4_K_M.gguf" not in q.manifest
    assert not any(name in q.manifest for name in ("qwen-image-2.1-Q8_0.gguf", "qwen-image-2.1-Q4_K_M.gguf"))
    assert q.fingerprint() == before                                  # a new size: no new venv, only a new file
    assert round(q.models_gb, 2) == 15.83 and q.size_gb > q.models_gb + 2


def test_the_pinned_qwen_models_cover_every_size(tmp_path: Path) -> None:
    assert tuple(models.QWEN_DENOISERS) == tuple(models.QWEN_TURBO_DENOISERS) == config.QUALITIES
    for size, sha in [*models.QWEN_DENOISERS.values(), *models.QWEN_TURBO_DENOISERS.values(),
                      *(pin for name, pin in models.QWEN_SHARED_FILES.items() if name not in models.ALBEDO_PINS)]:
        assert size > 600_000_000 and len(sha) == 64 and int(sha, 16) >= 0
    assert models.QWEN_DENOISERS["Q4_K_M"] == (4_199_565_024,
                                               "631d532e7ca71e8d90a87c71d3699761a812039d22e3370e87498d87754660fe")
    # the Turbo: unsloth/Qwen-Image-2.1-Turbo-GGUF @ 5d1f9f7 (HuggingFace's LFS ids), 96 bytes more in every size
    assert models.QWEN_TURBO_DENOISERS == {
        "Q3_K_M": (3_168_290_624, "2fd16e6590c8153980f0de4d8965007d8cf3cc497240e74c03db7bb05e799467"),
        "Q4_K_M": (4_199_565_120, "8940e8b522af886e0c0f5afddf720e82dd223cb9a43580b5f61c4d740452ac20"),
        "Q5_K_M": (5_390_223_168, "37c51ff446f4db2e131a9d12c003f0c0e824c8b8ba4dcd8c6cd5753f13457f5f"),
        "Q6_K": (6_271_551_296, "f4560fb47ad863ffed7e32737d3c42988c44bd0c459606777d7ffe2e895d0a70"),
        "Q8_0": (7_640_860_480, "399d2ddc3636f36456e2f5bfa57187da4e38721bc046430317f42a9eab5b1712")}
    assert models.QWEN_MODELS == ("turbo", "full")
    assert models.qwen_denoiser("turbo", "Q5_K_M") == ("qwen-image-2.1-turbo-Q5_K_M.gguf",
                                                       models.QWEN_TURBO_DENOISERS["Q5_K_M"])
    assert models.qwen_denoiser("full", "Q5_K_M") == ("qwen-image-2.1-Q5_K_M.gguf", models.QWEN_DENOISERS["Q5_K_M"])
    with pytest.raises(ValueError):
        models.qwen_denoiser("fast", "Q5_K_M")
    assert models.qwen_full_manifest("Q6_K", tmp_path) == {"qwen-image-2.1-Q6_K.gguf": models.QWEN_DENOISERS["Q6_K"]}
    # every pin, both models in every size, for prepare_release.py check
    assert len(models.QWEN_PINS) == 2 * len(config.QUALITIES) + len(models.QWEN_SHARED_FILES) + \
        len(models.QWEN_TEXT_ENCODERS) - 1                  # the default text encoder is among the shared files
    assert set(models.QWEN_PINS) == set(models.QWEN_SOURCES)
    q = _qwen()
    assert q.model_url("qwen-image-2.1-Q3_K_M.gguf") == ("https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/resolve/"
                                                         "2c31ccd392b367a6637841a143813320a02dff55/"
                                                         "qwen-image-2.1-Q3_K_M.gguf")
    assert q.model_url("qwen-image-2.1-turbo-Q4_K_M.gguf") == (
        "https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF/resolve/5d1f9f759ea877ae556cda88dbec958b62ad3809/"
        "qwen-image-2.1-turbo-Q4_K_M.gguf")
    assert q.model_url(models.QWEN_VAE).endswith("/resolve/d67caebb412f98c968f9be41b3a8cea80b3e039a/vae/"
                                                 "qwen_image_2.1_vae_bf16.safetensors")
    assert q.model_url(models.QWEN_VISION).endswith("Qwen3-VL-8B-Instruct-GGUF/resolve/"
                                                    "b93a7ee713758252c555be4210c00540df954dc2/mmproj-F16.gguf")
    # our albedo add-on: its own repo at its commit, into models/qwen-image/albedo, with the files the launcher hands
    # the engine among them
    assert q.model_url("albedo/albedo_lora.safetensors") == (
        "https://huggingface.co/IgorAherne/spellcast3d-marigold-albedo-qwen-image-2.1/resolve/"
        "1867b64326fb628ac8bf6d60801169d9ee750560/albedo_lora.safetensors")
    assert paths.MODELS_ALBEDO == paths.MODELS_QWEN / "albedo"
    assert {f"albedo/{name}" for name in models.ALBEDO_FILES.values()} < set(models.ALBEDO_PINS)
    assert models.ALBEDO_PINS["albedo/albedo_lora.safetensors"] == (
        683_777_548, "720ddc255a165c7f483115a4e97dd8511a7df71c3133bca2e3d8571b3ce1378a")
    # the names the launcher loads (launchers/qwen_server.py) are these (nothing in tmp_path: the configured size)
    files = models.qwen_model_files("Q5_K_M", tmp_path)
    assert {k: v.name for k, v in files.items()} == {
        "turbo": "qwen-image-2.1-turbo-Q5_K_M.gguf", "full": "qwen-image-2.1-Q5_K_M.gguf",
        "vae": "qwen_image_2.1_vae_bf16.safetensors",
        "text_encoder": "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf", "vision": "Qwen3-VL-8B-Instruct-mmproj-F16.gguf"}
    assert all(f.parent == tmp_path for f in files.values())
    assert models.qwen_model_files("Q5_K_M")["vae"].parent == paths.MODELS_QWEN     # the real folder by default


def test_switching_the_size_downloads_only_the_new_file(cfg: Callable[[str], None], small_runtime: dict[str, Any],
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    q = _qwen()
    data = {name: name.encode() * 3 for name in ("qwen-image-2.1-turbo-Q4_K_M.gguf", "qwen-image-2.1-turbo-Q6_K.gguf",
                                                  models.QWEN_VAE, models.QWEN_TEXT_ENCODER, models.QWEN_VISION)}
    pins = {name: (len(d), _sha(d)) for name, d in data.items()}
    monkeypatch.setattr(models, "QWEN_TURBO_DENOISERS", {"Q4_K_M": pins["qwen-image-2.1-turbo-Q4_K_M.gguf"],
                                                         "Q6_K": pins["qwen-image-2.1-turbo-Q6_K.gguf"]})
    monkeypatch.setattr(models, "QWEN_SHARED_FILES", {n: pins[n] for n in list(data)[2:]})
    for name in ("qwen-image-2.1-turbo-Q4_K_M.gguf", *list(data)[2:]):    # installed with Q4_K_M
        (q.models_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (q.models_dir / name).write_bytes(data[name])
        models.record_verified(q.models_dir / name, pins[name][1])
    sdcpp.unpack_engine(paths.QWEN_SDCPP, lambda line: None)
    sdcpp.unpack(small_runtime["cudart"], paths.QWEN_SDCPP)
    cfg(TEMPLATE)
    assert q.missing_files() == [] and q.repair_work() is None
    cfg(TEMPLATE.replace("quality = Q4_K_M", "quality = Q6_K"))
    assert q.repair_work() == "some of its downloaded files are missing: qwen-image-2.1-turbo-Q6_K.gguf"
    assert q.download_hosts() == {"huggingface.co"}
    downloads: list[str] = []

    def download(url: str, dst: Path, size: int, sha: str, log: Any) -> None:
        downloads.append(url)
        Path(dst).write_bytes(data[Path(dst).name])
        models.record_verified(Path(dst), sha)

    monkeypatch.setattr(models, "download_file", download)
    args = argparse.Namespace(reinstall=(), force_pixal3d=False)
    monkeypatch.setattr(install_main.venvs, "usable", lambda _id: True)
    pending = install_main._pending(q, {"status": state.INSTALLED, "fingerprint": q.fingerprint()}, args)
    assert pending == Pending("some of its downloaded files are missing: qwen-image-2.1-turbo-Q6_K.gguf",
                              repair=True)
    ctx = _RecordingCtx()
    q.repair(ctx)
    assert downloads == [q.model_url("qwen-image-2.1-turbo-Q6_K.gguf")] and q.missing_files() == []
    assert ctx.commands == []                                          # the runtime was complete: not even checked
    assert (q.models_dir / "qwen-image-2.1-turbo-Q4_K_M.gguf").is_file()   # the old size stays (switching back)


def test_a_bigger_turbo_already_there_stands_in_for_the_configured_size(cfg: Callable[[str], None],
                                                                       small_runtime: dict[str, Any],
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """The artist's own Turbo of a bigger size than the config's is the one used (models.size_on_disk: nothing is
    downloaded), and it is theirs: never hashed (unverified), never replaced. A smaller size does not stand in - the
    configured one is downloaded -, and the configured size, once there, is the one used whatever else is there."""
    q = _qwen()
    names = {size: f"qwen-image-2.1-turbo-{size}.gguf" for size in ("Q3_K_M", "Q4_K_M", "Q6_K")}
    data = {name: name.encode() * 3 for name in (*names.values(), models.QWEN_VAE, models.QWEN_TEXT_ENCODER,
                                                  models.QWEN_VISION)}
    pins = {name: (len(d), _sha(d)) for name, d in data.items()}
    monkeypatch.setattr(models, "QWEN_TURBO_DENOISERS", {size: pins[name] for size, name in names.items()})
    monkeypatch.setattr(models, "QWEN_SHARED_FILES", {n: pins[n] for n in list(data)[3:]})
    q.models_dir.mkdir(parents=True, exist_ok=True)
    for name in list(data)[3:]:
        (q.models_dir / name).write_bytes(data[name])
        models.record_verified(q.models_dir / name, pins[name][1])
    sdcpp.unpack_engine(paths.QWEN_SDCPP, lambda line: None)
    sdcpp.unpack(small_runtime["cudart"], paths.QWEN_SDCPP)
    cfg(TEMPLATE)                                                      # quality = Q4_K_M
    own = b"the artist's own Q6_K" + bytes(len(data[names["Q6_K"]]) - 21)      # its pinned size, other bytes
    (q.models_dir / names["Q6_K"]).write_bytes(own)
    assert models.qwen_size_on_disk("turbo", "Q4_K_M", q.models_dir) == "Q6_K"
    assert names["Q6_K"] in q.manifest and names["Q4_K_M"] not in q.manifest
    assert q.missing_files() == [] and q.unverified() == [] and q.repair_work() is None
    monkeypatch.setattr(models, "download_file", lambda *a: pytest.fail("downloaded"))
    q.repair(_RecordingCtx())
    assert (q.models_dir / names["Q6_K"]).read_bytes() == own          # never hashed, never replaced
    assert models.qwen_model_files("Q4_K_M", q.models_dir)["turbo"].name == names["Q6_K"]     # the launcher's
    # a smaller size only: the configured one is downloaded
    (q.models_dir / names["Q6_K"]).unlink()
    (q.models_dir / names["Q3_K_M"]).write_bytes(data[names["Q3_K_M"]])
    assert models.qwen_size_on_disk("turbo", "Q4_K_M", q.models_dir) is None
    assert q.missing_files() == [names["Q4_K_M"]]
    downloads: list[str] = []

    def download(url: str, dst: Path, size: int, sha: str, log: Any) -> None:
        downloads.append(Path(dst).name)
        Path(dst).write_bytes(data[Path(dst).name])
        models.record_verified(Path(dst), sha)

    monkeypatch.setattr(models, "download_file", download)
    q.repair(_RecordingCtx())
    assert downloads == [names["Q4_K_M"]] and q.missing_files() == []
    # the configured size there: the one used, whatever else is
    (q.models_dir / names["Q6_K"]).write_bytes(own)
    assert models.qwen_size_on_disk("turbo", "Q4_K_M", q.models_dir) == "Q4_K_M" and names["Q4_K_M"] in q.manifest
    assert models.size_on_disk("Q4_K_M", ("Q3_K_M", "Q4_K_M", "Q6_K", "Q8_0"), lambda s: s in ("Q3_K_M", "Q6_K")) == "Q6_K"
    assert models.size_on_disk("Q9", ("Q3_K_M",), lambda s: False) is None       # an unknown size: nothing stands in


def test_an_update_fetches_the_turbo_and_leaves_the_full_model_alone(cfg: Callable[[str], None],
                                                                     small_runtime: dict[str, Any],
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """An install from before the Turbo (the full model and the shared files) updates: the Turbo's denoiser is the one
    missing file and the one download; the full model's file is no part of the install - never checked, never
    replaced, never removed, even when it is not the pinned file - and stays usable (the launcher's)."""
    q = _qwen()
    turbo, full = "qwen-image-2.1-turbo-Q4_K_M.gguf", "qwen-image-2.1-Q4_K_M.gguf"
    data = {name: name.encode() * 3 for name in (turbo, full, models.QWEN_VAE, models.QWEN_TEXT_ENCODER,
                                                  models.QWEN_VISION)}
    pins = {name: (len(d), _sha(d)) for name, d in data.items()}
    monkeypatch.setattr(models, "QWEN_TURBO_DENOISERS", {"Q4_K_M": pins[turbo]})
    monkeypatch.setattr(models, "QWEN_DENOISERS", {"Q4_K_M": pins[full]})
    monkeypatch.setattr(models, "QWEN_SHARED_FILES", {n: pins[n] for n in list(data)[2:]})
    q.models_dir.mkdir(parents=True, exist_ok=True)
    for name in list(data)[2:]:                                        # the shared files, checked when installed
        (q.models_dir / name).write_bytes(data[name])
        models.record_verified(q.models_dir / name, pins[name][1])
    old = b"another full model" + bytes(len(data[full]) - 18)        # its pinned size, other bytes, never checked
    (q.models_dir / full).write_bytes(old)
    sdcpp.unpack_engine(paths.QWEN_SDCPP, lambda line: None)
    sdcpp.unpack(small_runtime["cudart"], paths.QWEN_SDCPP)
    cfg(TEMPLATE)
    assert q.missing_files() == [turbo] and q.unverified() == []
    assert q.repair_work() == f"some of its downloaded files are missing: {turbo}"
    assert models.qwen_model_on_disk("full", "Q4_K_M", q.models_dir)          # the Images page: there, usable
    downloads: list[str] = []

    def download(url: str, dst: Path, size: int, sha: str, log: Any) -> None:
        downloads.append(url)
        Path(dst).write_bytes(data[Path(dst).name])
        models.record_verified(Path(dst), sha)

    monkeypatch.setattr(models, "download_file", download)
    q.repair(_RecordingCtx())
    assert downloads == [q.model_url(turbo)] and q.missing_files() == [] and q.repair_work() is None
    assert (q.models_dir / full).read_bytes() == old                   # left exactly as it was


def test_the_artists_huggingface_cache_is_linked_under_our_names(qwen_here: Path, cfg: Callable[[str], None],
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    q = _qwen()
    cfg(TEMPLATE)
    vision = b"vision weights"
    monkeypatch.setattr(models, "QWEN_SHARED_FILES", {**models.QWEN_SHARED_FILES,
                                                      models.QWEN_VISION: (len(vision), _sha(vision))})
    repo, revision, remote = models.QWEN_SOURCES[models.QWEN_VISION]
    cache = qwen_here / "hf" / "hub"
    blob = models.hf_repo_folder(cache, repo) / "blobs" / _sha(vision)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(vision)
    snapshot = models.hf_repo_folder(cache, repo) / "snapshots" / revision
    snapshot.mkdir(parents=True)
    os.link(blob, snapshot / remote)                       # as huggingface_hub writes it without symbolic links
    monkeypatch.setattr(models, "user_hf_caches", lambda: [cache])
    ctx = _RecordingCtx()
    q._seed_models(ctx)
    assert os.path.samefile(blob, q.models_dir / "Qwen3-VL-8B-Instruct-mmproj-F16.gguf")
    assert models.QWEN_VISION not in q.missing_files() and "linked them" in ctx.lines[-1]


def test_qwen_installs_its_packages_runtime_and_models_and_checks_the_gpu(small_runtime: dict[str, Any],
                                                                         monkeypatch: pytest.MonkeyPatch,
                                                                         tmp_path: Path) -> None:
    folder = tmp_path / "install"
    folder.mkdir()
    (folder / "requirements-qwen.txt").write_text("fastapi==0.141.1\n", encoding="utf-8")
    monkeypatch.setattr(components, "INSTALL_DIR", folder)
    steps: list[str] = []
    monkeypatch.setattr(components.Qwen, "_fetch_models", lambda self, ctx, until: steps.append("models"))
    q = _qwen()
    ctx = _RecordingCtx()
    q.prepare(ctx)
    q.install(ctx)
    pip, verify = ctx.commands[0][0], ctx.commands[-1]
    assert pip[pip.index("install") + 1:] == ["--progress-bar", "raw", "-r", str(folder / "requirements-qwen.txt")]
    assert not any("torch" in " ".join(argv) for argv, _cwd in ctx.commands)      # no PyTorch anywhere
    assert steps == ["models"]
    for name, data in {**small_runtime["runtime"], **small_runtime["cuda"]}.items():
        assert (paths.QWEN_SDCPP / name).read_bytes() == data
    assert sdcpp.missing() == [] and not list(paths.QWEN_SDCPP.glob("*.partial"))
    assert verify == ([str(paths.venv_python("qwen")), "-c", sdcpp.VERIFY_CODE], paths.QWEN_SDCPP)
    assert "sd_list_devices" in sdcpp.VERIFY_CODE and "'CUDA' not in devices" in sdcpp.VERIFY_CODE
    assert f"not '{sdcpp.COMMIT}'.startswith(commit)" in sdcpp.VERIFY_CODE     # the DLL must be the pinned build
    assert sdcpp.unpacked_from(paths.QWEN_SDCPP) == small_runtime["sd"].sha256  # the folder says which zip it holds
    # the zips stay in code/venvs/downloads: a repair unpacks again without downloading
    (paths.QWEN_SDCPP / "ggml-cuda.dll").unlink()
    assert sdcpp.missing() == ["stable-diffusion.cpp"] and "github.com" not in q.download_hosts()
    monkeypatch.setattr(models, "download_file", lambda *a: pytest.fail("downloaded"))
    ctx = _RecordingCtx()
    q.repair(ctx)
    assert sdcpp.missing() == [] and ctx.commands[-1][0][-1] == sdcpp.VERIFY_CODE
    (folder / "requirements-qwen.txt").unlink()
    with pytest.raises(InstallError, match="requirements-qwen.txt is missing, so this copy of Spellcast3D is"):
        q.prepare(_RecordingCtx())


# ---- the developers' profilers ([profiling] in spellcast3d-config.ini) ---------------------------------------------

def _on(*keys: str) -> str:
    """The template with these [profiling] switches on."""
    text = TEMPLATE
    for key in keys:
        text = text.replace(f"\n{key} = off", f"\n{key} = on", 1)
    return text


class _Disk:
    def __init__(self, free_gb: float) -> None:
        self.free = int(free_gb * 1e9)


def _pip(pin: str) -> list[str]:
    """The profiler step's pip command for one pin (code\\venvs where the test has it)."""
    torch = ["--index-url", components.TORCH_INDEX] if pin == components.TORCH[0] else []
    return [str(paths.venv_python("qwen")), "-m", "pip", "install", "--progress-bar", "raw", pin, "-c",
            str(_qwen().requirements_file()), *torch]


@pytest.fixture
def installed_qwen(qwen_here: Path, monkeypatch: pytest.MonkeyPatch) -> components.Qwen:
    """Qwen-Image installed and complete (recorded with its fingerprint, a usable venv, its models and runtime there):
    only a profiler's package can be missing. The install state, code\\tmp and code\\logs are in tmp_path."""
    monkeypatch.setattr(components.Qwen, "missing_files", lambda self: [])
    monkeypatch.setattr(components.Qwen, "unverified", lambda self: [])
    monkeypatch.setattr(sdcpp, "missing", lambda folder=None: [])
    monkeypatch.setattr(venvs, "usable", lambda _id: True)
    monkeypatch.setattr(paths, "INSTALL_STATE", qwen_here / "install-state.json")
    monkeypatch.setattr(paths, "INSTALL_TMP", qwen_here / "tmp")
    monkeypatch.setattr(paths, "LOGS", qwen_here / "logs")
    monkeypatch.setattr(install_main, "pixal3d_calibration_work", lambda *args, **kwargs: None)   # not Qwen's
    q = _qwen()
    state.record(q.id, state.INSTALLED, "", q.fingerprint())
    return q


class _Main:
    """install_main.main() / _run() around a fake pip: the commands it ran, the disk it sees, the venvs it shared."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.calls: list[list[str]] = []
        self.shared: list[str] = []
        self.disk = _Disk(100)
        self.pip_fails = False
        main = self

        def stream(ctx: Ctx, argv: list[str], cwd: Path, env: dict[str, str]) -> int:
            main.calls.append([str(a) for a in argv])
            if main.pip_fails:
                ctx._tail.append("ERROR: Could not install packages due to an OSError: [Errno 28] No space left on "
                                 "device")
                return 1
            return 0

        monkeypatch.setattr(install_main, "HUB_LOCK", tmp_path / ".hub.lock")
        monkeypatch.setattr(install_main, "INSTALL_LOCK", tmp_path / ".install.lock")
        monkeypatch.setattr(install_main.models, "ensure_folders", lambda: None)
        monkeypatch.setattr(install_main.venvs, "repair_if_moved", lambda *a: False)
        monkeypatch.setattr(install_main, "_gpu_info", lambda: None)
        monkeypatch.setattr(install_main.sdcpp, "share_cuda_dlls", lambda log: 0)
        monkeypatch.setattr(components.shutil, "disk_usage", lambda path: main.disk)    # (preflight's shutil too)
        monkeypatch.setattr(components, "share_torch_files", lambda ctx, venv_id: main.shared.append(venv_id))
        monkeypatch.setattr(Ctx, "_stream", stream)


@pytest.fixture
def fake_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Main:
    return _Main(monkeypatch, tmp_path)


def test_a_switched_on_profiler_is_no_work_of_qwen_image(installed_qwen: components.Qwen, cfg: Callable[[str], None],
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen-Image never installs, repairs or budgets a profiler's package: its fingerprint, repair work and plan are
    the same with the switches on, so preflight never sees one (it could block Qwen-Image, and so record it failed)."""
    q = installed_qwen
    cfg(TEMPLATE)
    before = (q.fingerprint(), q.disk_needed_gb())
    cfg(_on("use_python_profiler", "use_torch_profiler", "diagnostic_enable_sync_hunter"))
    assert (q.fingerprint(), q.disk_needed_gb()) == before and q.repair_work() is None
    assert install_main._plan([q], _args()) == {}
    assert components.missing_profilers([q]) == {"use_python_profiler": "pyinstrument==5.1.3",
                                                 "use_torch_profiler": components.TORCH[0],
                                                 "diagnostic_enable_sync_hunter": components.TORCH[0]}
    assert components.profilers_work(components.missing_profilers([q])) == (
        "spellcast3d-config.ini switches on a profiler for developers (use_python_profiler, use_torch_profiler, "
        "diagnostic_enable_sync_hunter): it needs pyinstrument==5.1.3 and torch==2.8.0+cu128 in Qwen-Image 2.1's "
        "Python environment")
    # pip writes the dist-info's RECORD last: without it the install was cut off
    site = paths.venv_dir("qwen") / "Lib" / "site-packages"
    for dist in ("pyinstrument-5.1.3.dist-info", components.TORCH_DIST):
        (site / dist).mkdir(parents=True)
    assert len(components.missing_profilers([q])) == 3
    (site / "pyinstrument-5.1.3.dist-info" / "RECORD").write_text("", encoding="utf-8")
    assert set(components.missing_profilers([q])) == {"use_torch_profiler", "diagnostic_enable_sync_hunter"}
    (site / components.TORCH_DIST / "RECORD").write_text("", encoding="utf-8")
    assert components.missing_profilers([q]) == {}
    # switched off: nothing to do, and the packages stay
    cfg(TEMPLATE)
    assert components.missing_profilers([q]) == {} and (site / components.TORCH_DIST / "RECORD").is_file()
    # only for a run that looks after an installed Qwen-Image: --only / --skip and the configured generator decide
    shutil.rmtree(site / "pyinstrument-5.1.3.dist-info")
    cfg(_on("use_python_profiler"))

    def needs(**kw: Any) -> dict[str, str]:
        return components.missing_profilers(install_main._selected(_args(**kw)))

    assert needs() and needs(only=("qwen",)) and needs(skip=("hub",))         # (the hub's background run)
    assert needs(skip=("qwen",)) == {} and needs(only=("hub",)) == {}          # (spellcast-internal.bat's hub run)
    cfg(_on("use_python_profiler").replace("\nbackend = qwen ", "\n; backend = qwen ")
        .replace("\n; backend = forge", "\nbackend = forge"))
    assert needs() == {} and needs(only=("qwen",))                             # Forge Neo picked: only when named
    cfg(_on("use_python_profiler"))
    for status in (state.FAILED, state.UNSUPPORTED):                           # not installed: nothing to add to
        state.record(q.id, status, "", q.fingerprint())
        assert components.missing_profilers([q]) == {}
        assert components.missing_profilers([q], dry_run=True)                # a dry run takes the run's word
    state.record(q.id, state.INSTALLED, "", q.fingerprint())
    monkeypatch.setattr(venvs, "usable", lambda _id: False)
    assert components.missing_profilers([q]) == {}


def test_the_profiler_step_adds_the_packages_and_never_raises(installed_qwen: components.Qwen,
                                                              cfg: Callable[[str], None],
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    shared: list[str] = []
    monkeypatch.setattr(components, "share_torch_files", lambda ctx, venv_id: shared.append(venv_id))
    disk = _Disk(100)
    monkeypatch.setattr(components.shutil, "disk_usage", lambda path: disk)

    class StepCtx(_RecordingCtx):
        """Records the commands, and Qwen-Image's entry in the progress file at each (the hub reads it)."""
        fail: type[Exception] | None = None

        def __init__(self) -> None:
            super().__init__()
            self.out.line = lambda text="": self.lines.append(text)       # type: ignore[method-assign]
            self.progress.begin({"qwen": (progress.DONE, "")})
            self.qwen_while_pip: list[str] = []

        def run(self, argv: list[str], *, cwd: Path, env: dict[str, str], fail: str,
                notes: tuple[tuple[str, str], ...] = ()) -> None:
            super().run(argv, cwd=cwd, env=env, fail=fail, notes=notes)
            self.qwen_while_pip.append(self.progress._data["components"]["qwen"]["status"])
            if self.fail is not None:
                raise self.fail(fail)

    def step(*keys: str) -> StepCtx:
        cfg(_on(*keys))
        ctx = StepCtx()
        assert components.install_profilers(ctx, [installed_qwen]) is bool(keys)
        assert ctx.progress._data["components"]["qwen"]["status"] == progress.DONE
        return ctx

    ctx = step("use_python_profiler")
    assert [argv for argv, _cwd in ctx.commands] == [_pip("pyinstrument==5.1.3")] and shared == []
    assert ctx.qwen_while_pip == [progress.RUNNING]          # the hub starts no Qwen-Image meanwhile
    assert ctx.lines == ["", "spellcast3d-config.ini switches on a profiler for developers (use_python_profiler): it "
                             "needs pyinstrument==5.1.3 in Qwen-Image 2.1's Python environment."]
    ctx = step("use_torch_profiler", "diagnostic_enable_sync_hunter")       # PyTorch once, then its files shared
    assert [argv for argv, _cwd in ctx.commands] == [_pip(components.TORCH[0])] and shared == ["qwen"]
    # all switched off: nothing at all
    ctx = step()
    assert ctx.commands == [] and ctx.lines == [] and ctx.qwen_while_pip == []
    # pip fails: one line each, the next pin still tried, Qwen-Image's entry ends as done
    StepCtx.fail = InstallError
    ctx = step("use_python_profiler", "use_torch_profiler")
    assert len(ctx.commands) == 2 and ctx.lines[2:] == [
        f"{pin} for the profiler was not installed (pip failed (its lines above say why)); pictures are made without "
        f"it, and the next start tries again." for pin in ("pyinstrument==5.1.3", components.TORCH[0])]
    # nor a bug, an OS error or an unreadable state: never an exception out of the step
    for error in (RuntimeError, OSError):
        StepCtx.fail = error
        step("use_python_profiler", "use_torch_profiler")
    StepCtx.fail = None

    def no_drive(path: Path) -> _Disk:
        raise OSError("no drive")

    monkeypatch.setattr(components.shutil, "disk_usage", no_drive)
    ctx = step("use_torch_profiler")
    assert ctx.commands == [] and "was not installed (no drive)" in ctx.lines[-1]
    monkeypatch.setattr(components, "missing_profilers", lambda *a: {}["broken"])
    ctx = StepCtx()
    assert components.install_profilers(ctx, [installed_qwen]) is False and ctx.commands == []
    assert ctx.lines == ["Could not tell whether a profiler needs a package ('broken'); the next start looks again."]


def test_a_profiler_switch_never_blocks_or_fails_qwen_on_a_nearly_full_drive(installed_qwen: components.Qwen,
                                                                             cfg: Callable[[str], None],
                                                                             fake_main: _Main, tmp_path: Path,
                                                                             capsys: pytest.CaptureFixture[str]
                                                                             ) -> None:
    """Less than preflight's 1 GB margin free and a switch on: Qwen-Image stays installed, nothing is recorded as
    failed, the run is no failure, and the profiler waits for the space (said)."""
    q = installed_qwen
    fingerprint = q.fingerprint()
    cfg(_on("use_python_profiler", "use_torch_profiler"))
    fake_main.disk = _Disk(0.8)
    assert preflight.DISK_MARGIN_GB > 0.8
    progress_file = tmp_path / "install-progress.json"
    # the hub's run at every start (here --only qwen): nothing to install, so only the profiler step
    assert install_main.main(["--if-needed", "--only", "qwen", "--progress-file", str(progress_file)]) == 0
    assert state.load()["qwen"] == {**state.load()["qwen"], "status": state.INSTALLED, "fingerprint": fingerprint}
    assert fake_main.calls == [_pip("pyinstrument==5.1.3")]                  # PyTorch waits for the space
    out = capsys.readouterr().out
    assert ("PyTorch for the profiler is not installed: it takes about 12 GB of free disk space while it installs, "
            "and 0.8 GB is free. Free up space and start Spellcast3D again.") in out
    written = json.loads(progress_file.read_text(encoding="utf-8"))
    assert written["error"] is None and written["components"]["qwen"]["status"] == progress.DONE
    assert "pyinstrument==5.1.3" in paths.log_file("install").read_text(encoding="utf-8")     # logged too
    # pip fails on the full disk: said, and still nothing failed
    fake_main.pip_fails = True
    fake_main.calls.clear()
    assert install_main.main(["--if-needed", "--only", "qwen"]) == 0
    assert len(fake_main.calls) == 1 and state.load()["qwen"]["status"] == state.INSTALLED
    assert "pyinstrument==5.1.3 for the profiler was not installed (A disk is full" in capsys.readouterr().out
    # a whole run (tools\update.bat's): the same, Qwen-Image up to date in the summary
    assert install_main.main(["--only", "qwen"]) == 0
    out = capsys.readouterr().out
    assert "System check:" in out and "Qwen-Image 2.1   up to date" in out
    assert state.load()["qwen"]["status"] == state.INSTALLED
    # switched off again: nothing runs, nothing is logged, Qwen-Image as it was
    cfg(TEMPLATE)
    fake_main.calls.clear()
    paths.log_file("install").unlink()
    assert install_main.main(["--if-needed", "--only", "qwen"]) == 0
    assert fake_main.calls == [] and not paths.log_file("install").exists() and capsys.readouterr().out == ""
    assert state.load()["qwen"]["status"] == state.INSTALLED and q.fingerprint() == fingerprint


def test_the_profiler_step_comes_after_every_component_and_takes_only_what_they_left(
        installed_qwen: components.Qwen, cfg: Callable[[str], None], fake_main: _Main,
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """No PyTorch in any venv, LATO.2 installed fresh in the same run with 18 GB free: preflight budgets LATO.2 alone,
    LATO.2 installs first, and PyTorch for the profiler only gets what is left (not enough here: it waits). Meanwhile
    the progress file shows Qwen-Image as waiting for that step (the hub starts none whose venv then gets PyTorch),
    and its entry ends done whatever the step does."""
    cfg(_on("use_torch_profiler"))
    events: list[str] = []
    budgeted: list[list[str]] = []
    qwen_meanwhile: list[tuple[str, str]] = []      # Qwen-Image's progress entry while LATO.2 installs
    contexts: list[Ctx] = []

    def qwen_entry(ctx: Ctx) -> tuple[str, str]:
        entry = ctx.progress._data["components"]["qwen"]
        return entry["status"], entry["message"]

    class Fresh(components.Component):
        id, title, about, size_gb = paths.LATO2, "LATO.2", "A fresh component after Qwen-Image.", 15.0

        def inputs(self) -> dict[str, str]:
            return {}

        def install(self, ctx: Ctx) -> None:
            events.append("lato2")
            qwen_meanwhile.append(qwen_entry(ctx))
            fake_main.disk.free -= int(7e9)       # what it keeps

    def preflight_run(todo: list[components.Component], gpu_info: Any, network: bool,
                      reinstall: tuple[str, ...] = ()) -> preflight.Report:
        budgeted.append([c.id for c in todo])
        return preflight.Report()

    monkeypatch.setattr(install_main.venvs, "ensure", lambda *a, **k: None)
    monkeypatch.setattr(install_main.preflight, "run", preflight_run)
    args = _args(if_needed=False, progress_file=None)

    def run(free_gb: float, **plan: Pending) -> int:
        fake_main.disk = _Disk(free_gb)
        fake_main.calls.clear()
        events.clear()
        qwen_meanwhile.clear()
        contexts.append(Ctx(Output(), False, Path(sys.executable)))
        return install_main._run(contexts[-1], [installed_qwen, Fresh()],
                                 {**plan, "lato2": Pending("not installed yet")}, args)

    waiting = (progress.PENDING, "spellcast3d-config.ini switches on a profiler for developers (use_torch_profiler): "
                                 "it needs torch==2.8.0+cu128 in Qwen-Image 2.1's Python environment")
    assert run(18) == 0
    assert budgeted[-1] == ["lato2"] and events == ["lato2"] and fake_main.calls == []
    assert qwen_meanwhile == [waiting] and qwen_entry(contexts[-1]) == (progress.DONE, "")
    assert state.load()["lato2"]["status"] == state.INSTALLED and state.load()["qwen"]["status"] == state.INSTALLED
    assert "PyTorch for the profiler is not installed" in capsys.readouterr().out
    # room for both: PyTorch after LATO.2, then its files shared with LATO.2's
    assert run(40) == 0
    assert events == ["lato2"] and fake_main.calls == [_pip(components.TORCH[0])] and fake_main.shared == ["qwen"]
    out = capsys.readouterr().out
    assert out.index("[2/2] LATO.2") < out.index("torch==2.8.0+cu128 in Qwen-Image 2.1's") < out.index("Summary")
    # LATO.2 failing does not stop it either (Qwen-Image is installed), and the run's failure is LATO.2's alone

    def broken(self: Fresh, ctx: Ctx) -> None:
        raise InstallError("no")

    works = Fresh.install
    monkeypatch.setattr(Fresh, "install", broken)
    assert run(40) == install_main.EXIT_FAILED and len(fake_main.calls) == 1
    assert state.load()["lato2"]["status"] == state.FAILED and state.load()["qwen"]["status"] == state.INSTALLED
    assert qwen_entry(contexts[-1]) == (progress.DONE, "")
    # Qwen-Image installed in the same run, before LATO.2: from the end of its own install on, it waits for the step
    monkeypatch.setattr(Fresh, "install", works)
    monkeypatch.setattr(components.Qwen, "repair", lambda self, ctx: events.append("qwen"))
    monkeypatch.setattr(install_main, "_unsupported", lambda c, args: None)     # (no GPU here)
    assert run(40, qwen=Pending("its models are missing", repair=True)) == 0
    assert events == ["qwen", "lato2"] and qwen_meanwhile == [waiting] and len(fake_main.calls) == 1
    assert qwen_entry(contexts[-1]) == (progress.DONE, "")
    # switched off while LATO.2 installs: no step, and Qwen-Image's entry still ends done (never "failed")

    def switch_off(self: Fresh, ctx: Ctx) -> None:
        works(self, ctx)
        cfg(TEMPLATE)

    monkeypatch.setattr(Fresh, "install", switch_off)
    cfg(_on("use_torch_profiler"))
    assert run(40) == 0 and qwen_meanwhile == [waiting] and fake_main.calls == []
    assert qwen_entry(contexts[-1]) == (progress.DONE, "")


def test_check_dry_run_and_skip_with_a_profiler_switched_on(installed_qwen: components.Qwen,
                                                            cfg: Callable[[str], None], fake_main: _Main,
                                                            monkeypatch: pytest.MonkeyPatch,
                                                            capsys: pytest.CaptureFixture[str]) -> None:
    cfg(_on("use_python_profiler"))
    # --check says it, installs nothing, and exits with "something needs installing"
    assert install_main.main(["--check", "--only", "qwen"]) == install_main.EXIT_PENDING
    out = capsys.readouterr().out
    assert "  qwen     installed and up to date" in out
    said = ("For developers: spellcast3d-config.ini switches on a profiler for developers (use_python_profiler): it "
            "needs pyinstrument==5.1.3 in Qwen-Image 2.1's Python environment, added at the next start.")
    assert "".join(said.split()) in "".join(out.split())                     # (wrapped at 78 characters)
    # --dry-run prints the command, runs nothing
    assert install_main.main(["--dry-run", "--only", "qwen"]) == 0
    out = capsys.readouterr().out
    assert f"  > {subprocess.list2cmdline(_pip('pyinstrument==5.1.3'))}" in out
    assert "nothing would run" not in out and not paths.log_file("install").exists()
    assert fake_main.calls == []
    # --skip qwen: not this run's business
    monkeypatch.setattr(install_main, "_plan", lambda selected, args: {})
    assert install_main.main(["--if-needed", "--skip", "qwen"]) == 0 and fake_main.calls == []
    assert install_main.main(["--check", "--skip", "qwen"]) == 0
    assert "For developers" not in capsys.readouterr().out
    # switched off: --check has nothing to say, a dry run nothing to print
    cfg(TEMPLATE)
    assert install_main.main(["--check", "--only", "qwen"]) == 0
    assert install_main.main(["--dry-run", "--only", "qwen"]) == 0
    assert "nothing would run" in capsys.readouterr().out and fake_main.calls == []


def test_the_cuda_dlls_are_linked_from_pytorch_only_when_byte_identical(small_runtime: dict[str, Any],
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    cuda = small_runtime["cuda"]
    folder = paths.QWEN_SDCPP
    _torch_venv(paths.venv_dir("pixal3d"), cuda)
    _torch_venv(paths.venv_dir("forge"), {**cuda, "cublasLt64_12.dll": b"X" * 90})     # same size, other bytes
    _torch_venv(paths.venv_dir("lato2"), {"cudart64_12.dll": b"short"})
    assert sdcpp.torch_libs("cublasLt64_12.dll") == [
        paths.venv_dir(v) / "Lib" / "site-packages" / "torch" / "lib" / "cublasLt64_12.dll"
        for v in ("forge", "pixal3d")]
    lines: list[str] = []
    assert sdcpp.link_cuda_dlls(folder, lines.append) == []
    for name, data in cuda.items():                                     # links to a byte-identical copy
        assert (folder / name).read_bytes() == data and os.stat(folder / name).st_nlink == 2
    assert os.path.samefile(folder / "cublasLt64_12.dll",
                            paths.venv_dir("pixal3d") / "Lib/site-packages/torch/lib/cublasLt64_12.dll")
    assert "are the same as PyTorch's in" in lines[-1] and "no extra disk space" in lines[-1]
    # a torch whose DLL is another build: that one comes from the cudart zip
    shutil.rmtree(folder)
    shutil.rmtree(paths.venv_dir("pixal3d"))
    assert sdcpp.link_cuda_dlls(folder, lines.append) == ["cublasLt64_12.dll"]
    downloads: list[str] = []
    monkeypatch.setattr(models, "download_file", lambda url, *a: downloads.append(url))
    components.Qwen()._install_runtime(_RecordingCtx(), until=0.1)
    assert (folder / "cublasLt64_12.dll").read_bytes() == cuda["cublasLt64_12.dll"]
    assert downloads == [] and sdcpp.missing() == []                    # the cudart zip was in the download cache


def test_an_interrupted_cuda_download_is_dropped_once_pytorch_provides_the_dlls(small_runtime: dict[str, Any],
                                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """A first install downloads the CUDA zip (no venv with PyTorch yet); cut off there, its .part would stay in
    code/venvs/downloads for good once Pixal3D's PyTorch provides the DLLs at the next start."""
    cudart = small_runtime["cudart"]
    part = cudart.path.with_name(cudart.name + ".part")
    cudart.path.rename(part)                                            # downloaded part of the way
    _torch_venv(paths.venv_dir("pixal3d"), small_runtime["cuda"])
    monkeypatch.setattr(models, "download_file", lambda *a: pytest.fail("downloaded"))
    components.Qwen()._install_runtime(_RecordingCtx(), until=0.1)
    assert sdcpp.missing() == [] and not part.exists()


def test_the_release_zips_are_checked_kept_and_downloaded_only_when_missing(small_runtime: dict[str, Any],
                                                                            monkeypatch: pytest.MonkeyPatch,
                                                                            tmp_path: Path) -> None:
    sd = small_runtime["sd"]
    lines: list[str] = []
    assert sdcpp.fetch(sd, lines.append) == sd.path and models.is_recorded(sd.path, sd.sha256)
    with monkeypatch.context() as m:
        m.setattr(models, "sha256_file", lambda p: pytest.fail("hashed again"))
        assert sdcpp.fetch(sd, lines.append) == sd.path                 # checked once, remembered
    good = sd.path.read_bytes()
    sd.path.write_bytes(b"x" * sd.size)                                 # the right size, other bytes
    # A file put there at another moment: the record (size and change time) must not match it. Written in the same
    # clock tick as the good one, it could keep the good one's change time (the test then failed now and then).
    later = os.stat(sd.path).st_mtime_ns + 10 ** 9
    os.utime(sd.path, ns=(later, later))
    downloads: list[str] = []

    def download(url: str, dst: Path, size: int, sha: str, log: Any) -> None:
        downloads.append(url)
        Path(dst).write_bytes(good)

    monkeypatch.setattr(models, "download_file", download)
    sdcpp.fetch(sd, lines.append)
    assert downloads == [sd.url] and "not the expected file" in lines[-2]
    assert sd.url == sdcpp.ENGINE.url + sd.name                         # each zip from its own release
    assert small_runtime["cudart"].url == sdcpp.OFFICIAL.url + "cudart-test.zip"
    # another copy's download folder (--models-from): linked, not downloaded
    sd.path.unlink()
    other = tmp_path / "other"
    other.mkdir()
    (other / sd.name).write_bytes(good)
    monkeypatch.setattr(models, "EXTRA_SOURCES", [other])
    downloads.clear()
    sdcpp.fetch(sd, lines.append)
    assert downloads == [] and os.path.samefile(other / sd.name, sd.path)
    # only the pinned names are unpacked, so nothing lands outside the folder
    bad = _zip(tmp_path / "evil" / "x.zip", {"../evil.dll": b"e", "stable-diffusion.dll": b"s"})
    monkeypatch.setattr(paths, "DOWNLOADS", tmp_path / "evil")
    sdcpp.unpack(sdcpp.Archive(sdcpp.ENGINE, bad.name, bad.size, bad.sha256, {"stable-diffusion.dll": 1}),
                 tmp_path / "out")
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["stable-diffusion.dll"]
    assert not (tmp_path / "evil.dll").exists()


def test_the_real_runtime_pins() -> None:
    assert sdcpp.COMMIT == "25d412700bd616e4a9118abe6c5b42498de038f4" and sdcpp.GGML_COMMIT.startswith("a7c45a7")
    assert (sdcpp.SDCPP.name, sdcpp.SDCPP.size, sdcpp.SDCPP.sha256[:8]) == (
        "sd-spellcast-perf-25d4127-bin-win-cuda12-x64.zip", 352_768_144, "85e7c7c0")
    assert (sdcpp.CUDART.name, sdcpp.CUDART.size, sdcpp.CUDART.sha256[:8]) == (
        "cudart-sd-bin-win-cu12-x64.zip", 563_452_046, "fe203668")
    files = sdcpp.SDCPP.files
    assert {"stable-diffusion.dll", "ggml.dll", "ggml-base.dll", "ggml-cuda.dll", "SPELLCAST-BUILD.txt"} <= set(files)
    assert len(files) == 24 and (files["stable-diffusion.dll"], files["ggml-cuda.dll"]) == (35_496_448, 348_747_776)
    assert {name: size for name, (size, _sha) in sdcpp.CUDA_DLLS.items()} == sdcpp.CUDART.files
    assert all("/" not in n and "\\" not in n for n in [*files, *sdcpp.CUDART.files, *sdcpp.FORMER_ONLY])   # flat
    # built without WebP / WebM: what only the official release's engine zip had, and nothing this build has
    assert sdcpp.FORMER_ONLY == {"libsharpyuv.dll": 24_576, "libwebp.dll": 373_248, "libwebpmux.dll": 41_984,
                                 "webm.dll": 328_192}
    assert not set(sdcpp.FORMER_ONLY) & {*files, *sdcpp.CUDART.files} and sdcpp.ENGINE_MARK not in files


def test_the_two_zips_come_from_their_two_releases() -> None:
    """The engine is Spellcast3D's own build, a release of the author's fork; NVIDIA's CUDA runtime is still the
    official release's. The launcher's mirror and the installer's verification pin the same build."""
    assert sdcpp.ENGINE == sdcpp.Release("IgorAherne/stable-diffusion.cpp", "spellcast-perf-25d4127")
    assert sdcpp.OFFICIAL == sdcpp.Release("leejet/stable-diffusion.cpp", "master-929-3f8527a")
    assert sdcpp.SDCPP.release is sdcpp.ENGINE and sdcpp.CUDART.release is sdcpp.OFFICIAL
    assert sdcpp.SDCPP.url == ("https://github.com/IgorAherne/stable-diffusion.cpp/releases/download/"
                               "spellcast-perf-25d4127/sd-spellcast-perf-25d4127-bin-win-cuda12-x64.zip")
    assert sdcpp.CUDART.url == ("https://github.com/leejet/stable-diffusion.cpp/releases/download/"
                                "master-929-3f8527a/cudart-sd-bin-win-cu12-x64.zip")
    assert sdcpp.ENGINE.api == ("https://api.github.com/repos/IgorAherne/stable-diffusion.cpp/releases/tags/"
                                "spellcast-perf-25d4127")
    assert sdcpp.OFFICIAL.api == ("https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/tags/"
                                  "master-929-3f8527a")
    assert sdcpp.ENGINE.tag.endswith(sdcpp.COMMIT[:7]) and sdcpp.COMMIT[:7] in sdcpp.SDCPP.name
    assert (qwen_sdcpp.PINNED_TAG, qwen_sdcpp.PINNED_COMMIT) == (sdcpp.ENGINE.tag, sdcpp.COMMIT)
    # the fingerprint names the engine's release and both zips: an install made with the official engine is changed
    assert _qwen().inputs()["runtime"] == f"spellcast-perf-25d4127 {sdcpp.SDCPP.sha256} {sdcpp.CUDART.sha256}"


def test_the_engine_pins_are_the_zip_on_this_pc() -> None:
    """With the engine's zip in code\\venvs\\downloads (an install here keeps it): the files it holds are the pinned
    ones, by name and size. (Its SHA-256 is the installer's to check, when it unpacks.)"""
    if not sdcpp.cached(sdcpp.SDCPP):
        pytest.skip("the engine's zip is not in code\\venvs\\downloads")
    with zipfile.ZipFile(sdcpp.SDCPP.path) as z:
        assert {info.filename: info.file_size for info in z.infolist()} == sdcpp.SDCPP.files


def test_an_install_made_with_the_official_engine_gets_the_own_build_whole(small_runtime: dict[str, Any],
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """An updated install: the runtime folder holds the former pin's files. Their sizes do not tell every one of
    them from this build's (ggml.dll has the same in both), so all of the engine is unpacked again; the four files
    only the official release had are removed, and nothing else in the folder is touched."""
    folder, runtime = paths.QWEN_SDCPP, small_runtime["runtime"]
    sdcpp.unpack(small_runtime["cudart"], folder)                       # NVIDIA's DLLs: the same in both
    cuda_before = {name: os.stat(folder / name).st_mtime_ns for name in small_runtime["cuda"]}
    (folder / "stable-diffusion.dll").write_bytes(b"official")          # another size
    (folder / "ggml-cuda.dll").write_bytes(b"X" * len(runtime["ggml-cuda.dll"]))    # the pinned size, other bytes
    (folder / "ggml.txt").write_bytes(runtime["ggml.txt"])              # the same file in both
    for name, size in sdcpp.FORMER_ONLY.items():
        (folder / name).write_bytes(bytes(size))
    (folder / "notes.txt").write_bytes(b"not the installer's")
    assert sdcpp.unpacked_from(folder) == "" and sdcpp.runtime_missing() == list(runtime)       # all: no mark
    assert sdcpp.missing() == ["stable-diffusion.cpp"]
    monkeypatch.setattr(models, "download_file", lambda *a: pytest.fail("downloaded"))  # the zip is in the cache
    real_unpack = sdcpp.unpack

    def switched_off(archive: sdcpp.Archive, into: Path, names: list[str] | None = None) -> None:
        real_unpack(archive, into, ["stable-diffusion.dll"])            # the PC goes off after the first file
        raise OSError("stopped")

    # stopped half-way: by their sizes the files would all pass now, but nothing marked the folder - all again
    with monkeypatch.context() as m:
        m.setattr(sdcpp, "unpack", switched_off)
        with pytest.raises(InstallError, match=r"runtime \(stable-diffusion.cpp\) failed \(stopped\)"):
            components.Qwen()._install_runtime(_RecordingCtx(), until=0.1)
    assert sdcpp._wrong_size(folder, sdcpp.SDCPP.files) == [] and sdcpp.runtime_missing() == list(runtime)
    assert (folder / "ggml-cuda.dll").read_bytes() != runtime["ggml-cuda.dll"] and (folder / "webm.dll").exists()
    ctx = _RecordingCtx()
    components.Qwen()._install_runtime(ctx, until=0.1)
    for name, data in runtime.items():
        assert (folder / name).read_bytes() == data
    assert sdcpp.missing() == [] and sdcpp.unpacked_from(folder) == small_runtime["sd"].sha256
    assert sorted(p.name for p in folder.iterdir()) == sorted([*runtime, *small_runtime["cuda"], "notes.txt",
                                                               sdcpp.ENGINE_MARK])
    assert {name: os.stat(folder / name).st_mtime_ns for name in small_runtime["cuda"]} == cuda_before
    assert ctx.lines[-1] == ("Removed 4 files of the stable-diffusion.cpp build used before, which this one does not "
                             "need (libsharpyuv.dll, libwebp.dll, libwebpmux.dll, webm.dll).")
    # complete: the next run does nothing at all
    with monkeypatch.context() as m:
        m.setattr(sdcpp, "unpack", lambda *a: pytest.fail("unpacked again"))
        ctx = _RecordingCtx()
        components.Qwen()._install_runtime(ctx, until=0.1)
        assert ctx.lines == []
    # a file of one of those names that is not the official release's (another size) is not the installer's to delete
    (folder / "libwebp.dll").write_bytes(b"somebody else's")
    (folder / sdcpp.ENGINE_MARK).unlink()
    lines: list[str] = []
    sdcpp.unpack_engine(folder, lines.append)
    assert (folder / "libwebp.dll").read_bytes() == b"somebody else's" and lines == [] and sdcpp.missing() == []


def test_the_cuda_pins_are_pytorchs_dlls_on_this_pc() -> None:
    """The pinned CUDA DLLs are byte for byte those of PyTorch 2.8.0+cu128 (checked against the release's cudart zip
    when they were pinned): on a PC with such a venv, they are what the installer links."""
    found = {name: sdcpp.torch_libs(name) for name in sdcpp.CUDA_DLLS}
    if not all(found.values()):
        pytest.skip("no venv here with PyTorch 2.8.0+cu128's CUDA DLLs")
    for name, libs in found.items():
        assert models.sha256_file(libs[0]) == sdcpp.CUDA_DLLS[name][1], libs[0]


def test_a_dry_run_only_prints_what_would_happen(qwen_here: Path, cfg: Callable[[str], None],
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    cfg(TEMPLATE)
    monkeypatch.setattr(models, "download_file", lambda *a: pytest.fail("downloaded"))
    ctx = _RecordingCtx(dry_run=True)
    q = _qwen()
    q.prepare(ctx)
    q.repair(ctx)
    text = "\n".join(ctx.lines)
    assert sdcpp.SDCPP.url in text and sdcpp.CUDART.url in text and "huggingface.co/unsloth" in text
    assert all(line.startswith("  [dry-run]") for line in ctx.lines) and len(ctx.lines) == 1 + 3 + 4 + len(models.ALBEDO_PINS)
    assert not paths.QWEN_SDCPP.exists() and not q.models_dir.exists()


def test_the_cuda_dlls_from_the_zip_become_links_once_pytorch_is_here(small_runtime: dict[str, Any],
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """On a new PC Qwen-Image installs before Pixal3D: its CUDA DLLs come from the cudart zip. Once a venv has PyTorch's
    byte-identical copies they become links to those, and the zip goes (about 1.35 GB back); a DLL in use stays."""
    cuda = small_runtime["cuda"]
    folder = paths.QWEN_SDCPP
    sdcpp.unpack(small_runtime["cudart"], folder)                          # what the first install did
    lines: list[str] = []
    assert sdcpp.share_cuda_dlls(lines.append) == 0 and lines == []       # no PyTorch here yet: nothing to do
    assert sdcpp.CUDART.path.exists()
    _torch_venv(paths.venv_dir("pixal3d"), {**cuda, "cublasLt64_12.dll": b"X" * 90})   # one of another build
    assert sdcpp.share_cuda_dlls(lines.append) == 30 + 70                 # the identical ones: links
    torch_lib = paths.venv_dir("pixal3d") / "Lib" / "site-packages" / "torch" / "lib"
    assert os.path.samefile(folder / "cudart64_12.dll", torch_lib / "cudart64_12.dll")
    assert not os.path.samefile(folder / "cublasLt64_12.dll", torch_lib / "cublasLt64_12.dll")
    assert sdcpp.CUDART.path.exists()                                      # cublasLt64_12 still needs it
    _torch_venv(paths.venv_dir("lato2"), cuda)
    real_replace = os.replace

    def in_use(src: Any, dst: Any) -> None:                                # Qwen-Image runs: its DLL is loaded
        if Path(dst).name == "cublasLt64_12.dll":
            raise PermissionError(5, "Access is denied")
        real_replace(src, dst)

    monkeypatch.setattr(sdcpp.os, "replace", in_use)
    assert sdcpp.share_cuda_dlls(lines.append) == 0 and sdcpp.CUDART.path.exists()
    assert not list(folder.glob("*.link"))
    monkeypatch.setattr(sdcpp.os, "replace", real_replace)
    zip_size = sdcpp.CUDART.size
    assert sdcpp.share_cuda_dlls(lines.append) == 90 + zip_size           # the next run: the last one, and the zip
    assert not sdcpp.CUDART.path.exists() and "given back" in lines[-1]
    for name, data in cuda.items():
        assert (folder / name).read_bytes() == data
    assert sdcpp.cuda_missing() == [] and sdcpp.share_cuda_dlls(lines.append) == 0    # done: a few stat calls


def test_the_model_size_is_taken_once_per_installer_run(cfg: Callable[[str], None]) -> None:
    """The file edited while the installer downloads must not change what it fetches (it would end with the new size
    missing, and Qwen-Image recorded as failed): install/main.py pins the size for its run."""
    q = _qwen()
    cfg(TEMPLATE)
    q.pinned_quality = "Q4_K_M"
    try:
        cfg(TEMPLATE.replace("quality = Q4_K_M", "quality = Q8_0"))
        assert q.quality == "Q4_K_M" and "qwen-image-2.1-turbo-Q4_K_M.gguf" in q.manifest
    finally:
        q.pinned_quality = None
    assert q.quality == "Q8_0"


def test_files_on_this_pc_need_no_server(qwen_here: Path, cfg: Callable[[str], None], small_runtime: dict[str, Any],
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """Preflight asks huggingface.co and github.com only for what must really be downloaded: model files in an
    --models-from folder (or the artist's caches) are linked, and so are the zips in another Spellcast3D's
    code\\venvs\\downloads (its models folder is what --models-from names)."""
    cfg(TEMPLATE)
    q = _qwen()
    small = {name: name.encode() * 2 for name in q.manifest}
    turbo = "qwen-image-2.1-turbo-Q4_K_M.gguf"
    monkeypatch.setattr(models, "QWEN_TURBO_DENOISERS", {"Q4_K_M": (len(small[turbo]), _sha(small[turbo]))})
    monkeypatch.setattr(models, "QWEN_SHARED_FILES", {n: (len(d), _sha(d)) for n, d in small.items() if n != turbo})
    other = qwen_here / "other" / "spellcast3d"
    (other / "models" / "qwen-image").mkdir(parents=True)
    for name, data in small.items():
        (other / "models" / "qwen-image" / name).parent.mkdir(exist_ok=True)       # the albedo add-on's folder
        (other / "models" / "qwen-image" / name).write_bytes(data)
    downloads = other / "code" / "venvs" / "downloads"
    downloads.mkdir(parents=True)
    for archive in (small_runtime["sd"], small_runtime["cudart"]):
        os.link(archive.path, downloads / archive.name)
        archive.path.unlink()                                       # not in our own download cache
    assert q.download_hosts() == {"huggingface.co", "github.com"}   # nothing here yet
    monkeypatch.setattr(models, "EXTRA_SOURCES", [other / "models"])
    assert q.download_hosts() == set()
    # and the installer takes them from there: links, no download
    monkeypatch.setattr(models, "download_file", lambda *a: pytest.fail("downloaded"))
    lines: list[str] = []
    for archive in (small_runtime["sd"], small_runtime["cudart"]):
        assert sdcpp.fetch(archive, lines.append) == archive.path
        assert os.path.samefile(archive.path, downloads / archive.name)
    q._seed_models(_RecordingCtx())
    assert q.missing_files() == ["stable-diffusion.cpp", *sdcpp.CUDART.files] and not models.missing_files(
        q.models_dir, q.manifest)
    # a copy of another size is no copy: that file still comes from huggingface.co
    (q.models_dir / models.QWEN_VAE).unlink()
    (other / "models" / "qwen-image" / models.QWEN_VAE).write_bytes(b"short")
    assert q.download_hosts() == {"huggingface.co"}


def test_the_retopology_tools_need_no_server_for_models_in_the_hf_cache(qwen_here: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """LATO.2 and TriFlow seed their models from the artist's caches: an unreachable huggingface.co must not stop
    them when nothing needs downloading (as before preflight asked it)."""
    triflow = next(c for c in components.COMPONENTS if c.id == paths.TRIFLOW)
    data = {rel: rel.encode() * 3 for rel in triflow.manifest}
    monkeypatch.setattr(components.Triflow, "manifest", {rel: (len(d), _sha(d)) for rel, d in data.items()})
    monkeypatch.setattr(components.Triflow, "models_dir", qwen_here / "models" / "triflow")
    monkeypatch.setattr(paths, "TRIFLOW_CHECKPOINTS", qwen_here / "no-checkpoints")
    snapshot = qwen_here / "hf" / "snap"
    for rel, d in data.items():
        (snapshot / rel).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / rel).write_bytes(d)
    assert triflow.download_hosts() == {"huggingface.co"}
    monkeypatch.setattr(models, "hf_cache_sources", lambda repo, revision: [snapshot])
    assert triflow.download_hosts() == set()


def test_the_disk_budget_counts_models_a_copy_on_this_pc_provides(qwen_here: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """Model files that --models-from or the artist's HuggingFace cache hold on this drive are hard-linked, not
    downloaded: preflight must not ask for room for them (it blocked Pixal3D and TriFlow on a 37 GB drive)."""
    triflow = next(c for c in components.COMPONENTS if c.id == paths.TRIFLOW)
    data = {rel: rel.encode() * 3 for rel in triflow.manifest}
    manifest = {rel: (len(d), _sha(d)) for rel, d in data.items()}
    total = sum(len(d) for d in data.values()) / 1e9
    monkeypatch.setattr(components.Triflow, "manifest", manifest)
    monkeypatch.setattr(components.Triflow, "models_dir", qwen_here / "models" / "triflow")
    monkeypatch.setattr(paths, "TRIFLOW_CHECKPOINTS", qwen_here / "no-checkpoints")
    assert triflow.missing_gb() == pytest.approx(total) and triflow.models_present_gb() == 0
    snapshot = qwen_here / "hf" / "snap"
    for rel, d in data.items():
        (snapshot / rel).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / rel).write_bytes(d)
    monkeypatch.setattr(models, "hf_cache_sources", lambda repo, revision: [snapshot])
    assert triflow.missing_gb() == 0 and triflow.models_present_gb() == pytest.approx(total)
    # an interrupted download of a linkable file is not counted twice
    first = next(iter(data))
    target = triflow.models_dir / (first + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data[first][:2])
    assert triflow.models_present_gb() == pytest.approx(total)
    # a copy of another size provides nothing: that file is still to come
    (snapshot / first).write_bytes(b"short")
    assert triflow.missing_gb() == pytest.approx((len(data[first]) - 2) / 1e9)
    assert triflow.models_present_gb() == pytest.approx(total - (len(data[first]) - 2) / 1e9)


# ---- the installer follows spellcast3d-config.ini ------------------------------------------------------------------

def _args(**kw: Any) -> argparse.Namespace:
    return argparse.Namespace(**{"only": (), "skip": (), "reinstall": (), "force_pixal3d": False, **kw})


def test_only_the_configured_image_generator_is_installed(cfg: Callable[[str], None]) -> None:
    def ids(**kw: Any) -> list[str]:
        return [c.id for c in install_main._selected(_args(**kw))]

    cfg(TEMPLATE)
    assert ids() == ["hub", "imesh", "qwen", "pixal3d", "lato2", "triflow"]
    cfg(FORGE)
    assert ids() == ["hub", "imesh", "forge", "pixal3d", "lato2", "triflow"]
    cfg(TEMPLATE.replace("backend = qwen ", "backend = forg "))       # a typo: no 12 GB download nobody chose
    assert ids() == ["hub", "imesh", "pixal3d", "lato2", "triflow"]
    assert install_main._config_problems(_args()) and not install_main._config_problems(_args(only=("hub",)))
    # named on the command line: installed whatever the file says
    cfg(TEMPLATE)
    assert ids(only=("forge",)) == ["forge"] and "forge" in ids(reinstall=("forge",))
    assert install_main._parse(["--reinstall", "forge"]).reinstall == ("forge",)
    assert install_main._ids("qwen,forge") == ("qwen", "forge")


def test_the_install_card_leaves_out_the_generator_not_picked(cfg: Callable[[str], None], tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "INSTALL_STATE", tmp_path / "install-state.json")
    state.record("forge", state.INSTALLED, "", "f")                     # the old generator: kept, not shown
    cfg(TEMPLATE)
    start = install_main._progress_start(install_main._selected(_args(skip=("hub",))), {"qwen": Pending("new")})
    assert list(start) == ["hub", "imesh", "qwen", "pixal3d", "lato2", "triflow"]
    assert start["qwen"] == ("pending", "new") and start["hub"] == ("skipped", "")
    cfg("backend = nope\n")
    assert "qwen" not in install_main._progress_start(install_main._selected(_args()), {})
    assert state.load()["forge"]["status"] == state.INSTALLED


def test_every_real_run_makes_the_config_file_and_a_dry_run_does_not(cfg: Callable[[str], None], tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch,
                                                                    capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(install_main, "HUB_LOCK", tmp_path / ".hub.lock")
    monkeypatch.setattr(install_main, "INSTALL_LOCK", tmp_path / ".install.lock")
    monkeypatch.setattr(paths, "INSTALL_STATE", tmp_path / "install-state.json")
    monkeypatch.setattr(install_main.models, "ensure_folders", lambda: None)
    monkeypatch.setattr(install_main.venvs, "repair_if_moved", lambda *a: False)
    monkeypatch.setattr(install_main, "_plan", lambda selected, args: {})
    monkeypatch.setattr(install_main, "_gpu_info", lambda: None)
    tidied: list[int] = []
    monkeypatch.setattr(install_main.sdcpp, "share_cuda_dlls", lambda log: tidied.append(1) or 0)
    path = tmp_path / "spellcast3d-config.ini"
    assert install_main.main(["--dry-run"]) == install_main.EXIT_OK and not path.exists()
    assert install_main.main(["--check"]) == install_main.EXIT_OK and not path.exists()
    assert not tidied                                                       # a dry run and a check change nothing
    assert install_main.main(["--only", "hub", "--if-needed"]) == install_main.EXIT_OK        # the hub-only run
    assert path.read_bytes() == paths.CONFIG_TEMPLATE.read_bytes()
    assert "Made spellcast3d-config.ini in the Spellcast3D folder" in capsys.readouterr().out
    assert tidied == [1]                    # the CUDA DLLs are looked at even when nothing needs installing
    cfg("backend = forg\n")
    assert install_main.main(["--if-needed", "--skip", "hub"]) == install_main.EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith('NOTE: spellcast3d-config.ini, line 1: "forg" is not an image generator')
    assert install_main.main(["--check"]) == install_main.EXIT_FAILED           # nothing can be installed for Images
    assert "Image generator (spellcast3d-config.ini): none" in capsys.readouterr().out


# ---- preflight --------------------------------------------------------------------------------------------------

def test_preflight_asks_no_pytorch_server_for_qwen_but_the_servers_it_downloads_from(
        qwen_here: Path, cfg: Callable[[str], None], monkeypatch: pytest.MonkeyPatch) -> None:
    cfg(TEMPLATE)
    asked: list[str] = []
    down = {"https://download.pytorch.org/whl/cu128/"}
    monkeypatch.setattr(preflight, "_reachable", lambda url: asked.append(url) or ("offline" if url in down else None))
    q, lato2 = _qwen(), next(c for c in components.COMPONENTS if c.id == paths.LATO2)
    monkeypatch.setattr(components.Lato2, "download_hosts", lambda self: set())
    r = preflight.Report()
    preflight._check_internet(r, [q, lato2])
    assert set(r.blocked) == {"lato2"} and "download.pytorch.org" in r.blocked["lato2"]
    # nothing here yet: its models from huggingface.co, stable-diffusion.cpp and the CUDA DLLs from github.com
    assert asked == ["https://pypi.org/simple/pip/", "https://download.pytorch.org/whl/cu128/",
                     "https://github.com/", "https://huggingface.co/"]
    assert r.info[-1] == "Internet: pypi.org, github.com and huggingface.co reachable"
    down.add("https://huggingface.co/")
    r = preflight.Report()
    preflight._check_internet(r, [q])
    assert "Could not reach huggingface.co, where its AI models come from (offline)" in r.blocked["qwen"]
    # with the zips in the download cache, github.com is not needed
    monkeypatch.setattr(sdcpp, "cached", lambda archive: True)
    assert q.download_hosts() == {"huggingface.co"}


def test_preflight_budgets_qwen_without_a_torch_twin(qwen_here: Path, cfg: Callable[[str], None],
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    cfg(TEMPLATE)
    q = _qwen()
    need = q.disk_needed_gb()
    assert need == pytest.approx(q.size_gb)                              # nothing here yet
    assert q.disk_kept_gb(torch_twin=True) == pytest.approx(need - sdcpp.CUDA_GB)     # CUDA DLLs: links
    pix = next(c for c in components.COMPONENTS if c.id == paths.PIXAL3D)

    class Usage:
        free = int((need + 1.5) * 1e9)

    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda p: Usage)
    monkeypatch.setattr(components.Pixal3D, "disk_needed_gb", lambda self, fresh=False: 0.2)
    r = preflight.Report()
    preflight._check_disk(r, [q, pix])
    assert r.blocked == {}
    # Qwen has no PyTorch: Pixal3D after it gets no torch twin out of it (no torch_shared_gb discount)
    seen: list[bool] = []
    monkeypatch.setattr(components.Pixal3D, "disk_kept_gb", lambda self, twin, fresh=False: seen.append(twin) or 0.2)
    preflight._check_disk(preflight.Report(), [q, pix])
    assert seen == [False]


# ---- the release -----------------------------------------------------------------------------------------------

@functools.cache
def _prepare_release() -> Any:
    sys.path.insert(0, str(paths.ROOT))
    try:
        import prepare_release
    finally:
        sys.path.remove(str(paths.ROOT))
    return prepare_release


def test_the_release_ships_the_config_file_ready_made(tmp_path: Path) -> None:
    pr = _prepare_release()
    assert pr.CONFIG_NAME == "spellcast3d-config.ini" and pr.CONFIG_NAME in pr.RELEASE_MUST_HAVE
    assert pr.CONFIG_TEMPLATE == "code/spellcast/config-template.ini" and pr.CONFIG_TEMPLATE in pr.RELEASE_MUST_HAVE
    assert "code/spellcast/install/requirements-qwen.txt" in pr.RELEASE_MUST_HAVE
    assert pr.LICENSE_TOOL_NAMES["qwen"] == "Qwen-Image 2.1"
    template = tmp_path / pr.CONFIG_TEMPLATE
    template.parent.mkdir(parents=True)
    shutil.copyfile(paths.CONFIG_TEMPLATE, template)
    pr.stage_config(tmp_path)
    assert (tmp_path / pr.CONFIG_NAME).read_bytes() == paths.CONFIG_TEMPLATE.read_bytes()


def test_the_release_check_compares_the_pinned_downloads(monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    """Each zip is looked up in its own release: the engine's in Spellcast3D's (the author's fork), the CUDA runtime's
    in the official one. A release holds only its own zip here, so asking one repository for both would fail."""
    pr = _prepare_release()
    real = {a.release.api: a for a in (sdcpp.SDCPP, sdcpp.CUDART)}
    asked: list[str] = []
    answers: dict[str, Exception] = {}  # a release's API URL -> what happens instead of GitHub answering as pinned

    def http(url: str, code: int) -> urllib.error.HTTPError:
        return urllib.error.HTTPError(url, code, "", None, None)       # type: ignore[arg-type]

    def api(url: str) -> Any:
        if "api.github.com" in url:
            asked.append(url)
            if url in answers:
                raise answers[url]
            a = real[url]
            return {"assets": [{"name": a.name, "size": a.size, "digest": f"sha256:{a.sha256}"}]}   # as GitHub answers
        repo, _, rest = url.removeprefix("https://huggingface.co/api/models/").partition("/tree/")
        revision, _, folder = rest.partition("/")
        pins = {**models.QWEN_PINS, **models.FLUX3_PINS}     # FLUX.3 Image Klein's too (the third model)
        return [{"path": remote, "size": pins[name][0], **({"lfs": {"oid": pins[name][1]}}
                                                           if remote.endswith((".gguf", ".safetensors")) else {})}
                for name, (r, rev, remote) in {**models.QWEN_SOURCES, **models.FLUX3_SOURCES}.items()
                if (r, rev, remote.rpartition("/")[0]) == (repo, revision, folder)]   # small files: plain git ones

    monkeypatch.setattr(pr, "_get_json", api)
    r = pr.Report()
    pr.check_qwen_downloads(r)
    assert (r.errors, r.warnings) == (0, 0)
    assert asked == [sdcpp.ENGINE.api, sdcpp.OFFICIAL.api] and sdcpp.ENGINE.api != sdcpp.OFFICIAL.api
    out = capsys.readouterr().out
    assert "[ OK ] IgorAherne/stable-diffusion.cpp spellcast-perf-25d4127: sd-spellcast-perf-25d4127-bin-" in out
    assert "[ OK ] leejet/stable-diffusion.cpp master-929-3f8527a: cudart-sd-bin-win-cu12-x64.zip as pinned" in out
    # FLUX.3 Image Klein's files, each repo at its commit (the VAE in a folder of its repo)
    assert "[ OK ] unsloth/flux-3-image-klein-GGUF@12ef13c: 5 file(s) as pinned" in out
    assert "[ OK ] Qwen/Qwen3-VL-4B-Instruct-GGUF@1cd86af: 1 file(s) as pinned" in out
    assert "[ OK ] Comfy-Org/Flux-2-Klein-4B@533c4cb: 1 file(s) as pinned" in out
    # the albedo add-on: its LICENSE, NOTICE and credits are plain git files there (no LFS id), checked by size
    assert "[ OK ] IgorAherne/spellcast3d-marigold-albedo-qwen-image-2.1@1867b64: 6 file(s) as pinned" in out
    # the engine's release is not published (yet): an error that says what to do; the official one is still checked
    answers[sdcpp.ENGINE.api] = http(sdcpp.ENGINE.api, 404)
    r = pr.Report()
    pr.check_qwen_downloads(r)
    out = capsys.readouterr().out
    assert (r.errors, r.warnings) == (1, 0) and "[ OK ] leejet/stable-diffusion.cpp master-929-3f8527a" in out
    assert ("[FAIL] GitHub has no release IgorAherne/stable-diffusion.cpp spellcast-perf-25d4127, where the "
            "installer downloads sd-spellcast-perf-25d4127-bin-win-cuda12-x64.zip: publish it") in out
    # offline, or GitHub's rate limit: warnings (check again online), never an error
    answers[sdcpp.ENGINE.api] = urllib.error.URLError("no route to host")
    answers[sdcpp.OFFICIAL.api] = http(sdcpp.OFFICIAL.api, 403)
    r = pr.Report()
    pr.check_qwen_downloads(r)
    assert (r.errors, r.warnings) == (0, 2) and capsys.readouterr().out.count("check again online") == 2
    answers.clear()
    # a zip that is not the pinned one any more
    monkeypatch.setattr(sdcpp, "SDCPP", sdcpp.Archive(sdcpp.ENGINE, sdcpp.SDCPP.name, 1, "0" * 64, {}))
    r = pr.Report()
    pr.check_qwen_downloads(r)
    assert r.errors == 1 and "is not the pinned file" in capsys.readouterr().out


@needs_repo
def test_the_gitignore_keeps_qwens_files_and_ignores_the_artists() -> None:
    pr = _prepare_release()
    probes = [(p, want) for p, want in pr.IGNORE_PROBES if "qwen" in p or "config" in p or "sdcpp" in p
              or "downloads" in p]
    assert len(probes) >= 12
    ignored = set(subprocess.run([GIT, "-C", str(paths.ROOT), "check-ignore", "--no-index", *(p for p, _ in probes)],
                                 capture_output=True, text=True).stdout.split("\n"))
    assert [(p, want) for p, want in probes if (p in ignored) != want] == []


def test_the_text_encoder_size_in_use_follows_the_rule_of_the_denoisers(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """The configured size when it is there, else the biggest bigger one there, else the configured one (downloaded);
    the files in use are said with that rule's words."""
    pins = {}
    for size in models.QWEN_TEXT_ENCODERS:
        data = size.encode()
        pins[size] = (len(data), _sha(data))
    monkeypatch.setattr(models, "QWEN_TEXT_ENCODERS", pins)
    put = lambda size: (tmp_path / models.qwen_text_encoder(size)[0]).write_bytes(size.encode())  # noqa: E731
    assert models.text_encoder_in_use("UD-Q6_K_XL", tmp_path) == "UD-Q6_K_XL"           # none there: downloaded
    assert models.qwen_text_encoder("UD-Q6_K_XL")[0] in models.qwen_manifest("Q4_K_M", tmp_path, "UD-Q6_K_XL")
    put("Q8_0")
    put("UD-Q5_K_XL")
    assert models.text_encoder_in_use("UD-Q6_K_XL", tmp_path) == "Q8_0"                 # the biggest bigger one
    assert models.text_encoder_in_use("UD-Q8_K_XL", tmp_path) == "UD-Q8_K_XL"           # never a smaller one
    put("UD-Q6_K_XL")
    assert models.text_encoder_in_use("UD-Q6_K_XL", tmp_path) == "UD-Q6_K_XL"           # the configured one first
    files = models.qwen_model_files("Q4_K_M", tmp_path, "UD-Q4_K_XL")
    assert files["text_encoder"].name == "Qwen3-VL-8B-Instruct-Q8_0.gguf"
    assert "text encoder Qwen3-VL-8B-Instruct-Q8_0.gguf (UD-Q4_K_XL, the configured size, is not there: the biggest " \
           "bigger one)" in models.qwen_files_words("Q4_K_M", "UD-Q4_K_XL", tmp_path)
    assert "Turbo qwen-image-2.1-turbo-Q4_K_M.gguf (not downloaded)" in models.qwen_files_words("Q4_K_M", folder=tmp_path)
    shared = models.qwen_shared_files("UD-Q6_K_XL", tmp_path)
    assert models.QWEN_TEXT_ENCODER not in shared and models.qwen_text_encoder("UD-Q6_K_XL")[0] in shared
