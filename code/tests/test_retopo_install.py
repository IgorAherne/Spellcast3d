"""Installing the retopology tools LATO.2 and TriFlow: ids, ports and paths, their components (fingerprints,
pending work, pip steps), their AI models (pinned manifests, hard links from copies on this PC, resumable
downloads, TriFlow's checkpoints links), venvs sharing identical files by hard links, TriFlow's nested submodules
and the proof that its files stay byte-identical to its commits, and prepare_release's view of all that.

Fast (no network: downloads come from a local HTTP server, git repos are made in tmp_path; no GPU).
Run with the hub venv:  python -m pytest -p no:cacheprovider code/tests
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import http.server
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any, Iterator

import pytest

from spellcast import gpu, paths
from spellcast.gpu import GpuInfo
from spellcast.install import components, models, preflight, state, submodules, venvs
from spellcast.install import main as install_main
from spellcast.install.main import Pending
from spellcast.install.runner import Ctx, InstallError, Output
from spellcast.settings import SettingsError, SettingsStore

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Spellcast3D is Windows-only")

GIT = shutil.which("git") or (str(paths.PORTABLE_GIT) if paths.PORTABLE_GIT.exists() else None)
needs_git = pytest.mark.skipif(GIT is None, reason="git is not installed")
# the git data of Spellcast3D itself (a release copy has none: its .gitignore and submodule pins are not in use)
needs_repo = pytest.mark.skipif(GIT is None or not (paths.ROOT / ".git").exists(),
                                reason="not a git checkout of Spellcast3D (a release copy has no git data)")


def _gpu(cc: float) -> GpuInfo:
    return GpuInfo(index=0, name="Test GPU", compute_cap=cc, vram_total_mb=8192, vram_used_mb=0,
                   driver_version="596.36")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(folder: Path, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(data)


def _lato2() -> components.Lato2:
    return next(c for c in components.COMPONENTS if c.id == paths.LATO2)     # type: ignore[return-value]


def _triflow() -> components.Triflow:
    return next(c for c in components.COMPONENTS if c.id == paths.TRIFLOW)   # type: ignore[return-value]


class _RecordingCtx(Ctx):
    """Records the commands instead of running them."""

    def __init__(self, dry_run: bool = False) -> None:
        super().__init__(Output(), dry_run, Path(sys.executable))
        self.commands: list[list[str]] = []
        self.lines: list[str] = []
        self.out.line = self.lines.append   # type: ignore[method-assign]

    def run(self, argv: list[str], *, cwd: Path, env: dict[str, str], fail: str,
            notes: tuple[tuple[str, str], ...] = ()) -> None:
        self.commands.append([str(a) for a in argv])


# ---- ids, paths, ports, GPU support ----------------------------------------------------------------------

def test_the_retopology_tools_have_ids_folders_and_stages() -> None:
    assert (paths.LATO2, paths.TRIFLOW) == ("lato2", "triflow")
    assert paths.TOOL_IDS == ("forge", "pixal3d", "imesh", "lato2", "triflow", "qwen")
    assert paths.ALL_VENV_IDS == ("hub", *paths.TOOL_IDS)
    assert paths.TOOL_DIRS[paths.LATO2] == paths.CODE / "lato2" == paths.LATO2_DIR
    assert paths.TOOL_DIRS[paths.TRIFLOW] == paths.CODE / "triflow" == paths.TRIFLOW_DIR
    assert (paths.MODELS_LATO2, paths.MODELS_TRIFLOW) == (paths.MODELS / "lato2", paths.MODELS / "triflow")
    assert paths.TRIFLOW_CHECKPOINTS == paths.TRIFLOW_DIR / "checkpoints"
    assert paths.venv_dir(paths.TRIFLOW) == paths.VENVS / "triflow"
    # STAGES keeps its shape (stage -> default tool); STAGE_TOOLS says which tools can run a stage.
    assert dict(paths.STAGES)["retopo"] == paths.TRIFLOW and len(paths.STAGES) == 6
    assert paths.STAGE_TOOLS["retopo"] == ("triflow", "lato2", "imesh")
    assert list(paths.STAGE_TOOLS) == [stage for stage, _tool in paths.STAGES]
    for stage, default in paths.STAGES:
        assert paths.STAGE_TOOLS[stage][:1] == ((default,) if default else ())
    assert {t for tools in paths.STAGE_TOOLS.values() for t in tools} == set(paths.TOOL_IDS)
    assert paths.MODELS_LATO2 in models.FOLDERS and paths.MODELS_TRIFLOW in models.FOLDERS


def test_every_venv_id_has_its_own_port(tmp_path: Path) -> None:
    ports = json.loads(paths.DEFAULTS_FILE.read_text(encoding="utf-8"))["ports"]
    assert ports == {"hub": 7770, "forge": 7771, "pixal3d": 7772, "imesh": 7773, "lato2": 7774, "triflow": 7775,
                     "qwen": 7776}
    assert set(ports) == set(paths.ALL_VENV_IDS) and len(set(ports.values())) == 7
    store = SettingsStore(paths.DEFAULTS_FILE, tmp_path / "settings.json")
    assert store.warnings == [] and store.get()["ports"]["triflow"] == 7775
    with pytest.raises(SettingsError, match="7 different port numbers"):      # two ports the artist set, alike
        store.patch({"ports": {"lato2": 7790, "triflow": 7790}})
    # a port the artist sets that is another tool's default: that default moves on, the artist's stays
    assert store.patch({"ports": {"triflow": 7774}})["ports"] == {**ports, "triflow": 7774, "lato2": 7775}
    store.patch({"ports": {"triflow": None}})
    with pytest.raises(SettingsError, match='"ports.lato2" must be a port number'):
        store.patch({"ports": {"lato2": "x"}})
    assert store.patch({"ports": {"lato2": 17774}})["ports"]["lato2"] == 17774
    # A settings.json from before the retopology tools keeps working: the new ports come from the defaults.
    (tmp_path / "old.json").write_text(json.dumps({"ports": {"hub": 7780}}), encoding="utf-8")
    old = SettingsStore(paths.DEFAULTS_FILE, tmp_path / "old.json")
    assert old.warnings == [] and old.get()["ports"] == {**ports, "hub": 7780}
    # ...even when the artist had moved a tool onto one of the new tools' ports: only the new default moves (the
    # hub keeps its port, and the artist's bookmark with it)
    (tmp_path / "moved.json").write_text(json.dumps({"ports": {"hub": 7774, "forge": 7775}}), encoding="utf-8")
    moved = SettingsStore(paths.DEFAULTS_FILE, tmp_path / "moved.json")
    assert moved.warnings == [] and moved.get()["ports"] == {"hub": 7774, "forge": 7775, "pixal3d": 7772,
                                                               "imesh": 7773, "lato2": 7777, "triflow": 7778,
                                                               "qwen": 7776}


@pytest.mark.parametrize("check", [gpu.lato2_supported, gpu.triflow_supported])
def test_retopology_tools_run_from_a_gtx_10xx(check: Any) -> None:
    for cc in (6.1, 7.5, 8.6, 12.0):
        assert check(_gpu(cc)) == (True, "")
    ok, why = check(_gpu(5.2))
    # plain words, no compute capability: the page shows it on the method's button
    assert not ok and "(Test GPU) is too old" in why and "GTX 1060" in why and "Instant Meshes" in why
    assert "sm_" not in why and "capability" not in why
    ok, why = check(None)
    assert not ok and "No NVIDIA GPU" in why
    assert gpu.pixal3d_supported(_gpu(7.5))[0] is True                    # round L: Pixal3D from sm_61 too
    assert gpu.pixal3d_supported(_gpu(6.1))[0] is True
    assert gpu.pixal3d_supported(_gpu(5.2))[0] is False
    assert gpu.pixal3d_card_env(_gpu(7.5)) == {"PIXAL3D_CARD_SM": "75"} and gpu.pixal3d_card_env(None) == {}
    assert gpu.pixal3d_card_env(_gpu(6.1)) == {"PIXAL3D_CARD_SM": "61", "DISABLE_ADDMM_CUDA_LT": "1"}, "a GTX 10xx"


# ---- components ------------------------------------------------------------------------------------------

def test_components_are_installed_after_pixal3d() -> None:
    assert [c.id for c in components.COMPONENTS] == ["hub", "imesh", "qwen", "forge", "pixal3d", "lato2",
                                                     "triflow"]
    lato2, triflow = _lato2(), _triflow()
    assert (lato2.title, triflow.title) == ("LATO.2", "TriFlow")
    assert lato2.uses_gpu and triflow.uses_gpu and lato2.needs_torch and triflow.needs_torch
    assert lato2.submodule == paths.LATO2_DIR and triflow.submodule == paths.TRIFLOW_DIR
    assert "NON-COMMERCIAL" in triflow.about and "MIT" in lato2.about
    assert lato2.supported(_gpu(6.1))[0] and not lato2.supported(_gpu(5.0))[0]
    assert triflow.supported(_gpu(6.1))[0] and not triflow.supported(None)[0]
    assert round(lato2.models_gb, 2) == 4.79 and round(triflow.models_gb, 2) == 1.32
    assert install_main._ids("lato2,triflow") == ("lato2", "triflow")
    assert components.TORCH_DIST == "torch-2.8.0+cu128.dist-info"
    # the wheels are named exactly (prepare_release.py checks the same names)
    assert [w.name for w in triflow.wheels] == [
        "torchsparse-2.1.0+cu128torch2.8-cp311-cp311-win_amd64.whl",
        "pyfqmr_triflow-0.5.0+triflow-cp311-cp311-win_amd64.whl",
        "flash_attn-2.8.3+cu128torch2.8-cp311-cp311-win_amd64.whl"]
    assert components.FLASH_ATTN_WHEEL in lato2.wheels and components.O_VOXEL_WHEEL in lato2.wheels


@pytest.fixture
def install_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """components.INSTALL_DIR in tmp_path, with a requirements file for both tools."""
    folder = tmp_path / "install"
    folder.mkdir()
    for tool in ("lato2", "triflow"):
        (folder / f"requirements-{tool}.txt").write_text("numpy==2.2.6\n", encoding="utf-8")
    monkeypatch.setattr(components, "INSTALL_DIR", folder)
    return folder


def test_fingerprints_follow_their_own_inputs_only(install_dir: Path, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    def prints() -> dict[str, str]:
        return {c.id: c.fingerprint() for c in components.COMPONENTS}

    before = prints()
    (install_dir / "requirements-lato2.txt").write_text("numpy==2.2.6\ntrimesh==4.10.1\n", encoding="utf-8")
    after = prints()
    assert [k for k in before if before[k] != after[k]] == ["lato2"]
    before = after
    (install_dir / "constraints-triflow.txt").write_text("scipy==1.17.1\n", encoding="utf-8")   # optional files count
    after = prints()
    assert [k for k in before if before[k] != after[k]] == ["triflow"]
    before = after
    (install_dir / "requirements-triflow-nodeps.txt").write_text("meshiki==0.0.7\n", encoding="utf-8")
    after = prints()
    assert [k for k in before if before[k] != after[k]] == ["triflow"]
    before = after
    # a rebuilt wheel (same name, other bytes) installs only the tool that uses it
    wheel = tmp_path / "torchsparse-2.1.0+cu128torch2.8-cp311-cp311-win_amd64.whl"
    wheel.write_bytes(b"one")
    monkeypatch.setattr(components.Triflow, "wheels", (wheel,))
    one = prints()
    wheel.write_bytes(b"two!")
    assert [k for k in one if one[k] != prints()[k]] == ["triflow"]
    before = prints()
    # TriFlow's nested submodules are part of its fingerprint
    monkeypatch.setattr(submodules, "triflow_heads", lambda triflow=None: {"third_party/TRELLIS": "f" * 40})
    after = prints()
    assert [k for k in before if before[k] != after[k]] == ["triflow"]
    assert "third_party/TRELLIS@" + "f" * 40 in _triflow().inputs()["nested"]
    assert _lato2().inputs()["torch"] == " ".join(components.TORCH)
    assert _triflow().inputs()["torch"] == " ".join(components.TORCH)          # no xformers (see Triflow)


def test_wheel_hashes_are_computed_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel = tmp_path / "x.whl"
    wheel.write_bytes(b"abc")
    calls: list[Path] = []
    real = state.file_sha256
    monkeypatch.setattr(components.state, "file_sha256", lambda p: calls.append(p) or real(p))
    components._sha256_of.cache_clear()
    assert components.wheel_sha256(wheel) == components.wheel_sha256(wheel) == _sha(b"abc")
    assert len(calls) == 1
    assert components.wheel_sha256(tmp_path / "gone.whl") == "missing"


@pytest.mark.parametrize("tool", ["lato2", "triflow"])
def test_pending_matrix_of_the_retopology_tools(tool: str, monkeypatch: pytest.MonkeyPatch) -> None:
    c = next(c for c in components.COMPONENTS if c.id == tool)
    args = argparse.Namespace(reinstall=(), force_pixal3d=False)
    monkeypatch.setattr(install_main.venvs, "usable", lambda _id: True)
    monkeypatch.setattr(type(c), "repair_work", lambda self: None)
    ok = {"status": state.INSTALLED, "fingerprint": c.fingerprint()}
    assert install_main._pending(c, ok, args) is None
    assert install_main._pending(c, {**ok, "fingerprint": "old"}, args) == Pending("an update changed it")
    assert install_main._pending(c, None, args) == Pending("not installed yet")
    assert install_main._pending(c, {"status": state.FAILED}, args) == Pending("the last attempt did not finish")
    monkeypatch.setattr(type(c), "repair_work", lambda self: "some of its downloaded files are missing: vflow.pt")
    assert install_main._pending(c, ok, args) == Pending("some of its downloaded files are missing: vflow.pt",
                                                          repair=True)
    monkeypatch.setattr(install_main, "_gpu_info", lambda: _gpu(5.2))
    assert install_main._pending(c, {"status": state.UNSUPPORTED}, args) is None
    monkeypatch.setattr(install_main, "_gpu_info", lambda: _gpu(6.1))
    assert install_main._pending(c, {"status": state.UNSUPPORTED}, args) == Pending("this PC can run it now")
    args.reinstall = (tool,)
    assert install_main._pending(c, ok, args) == Pending("a fresh reinstall was requested")


def test_repair_work_names_missing_models_and_triflow_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lato2, triflow = _lato2(), _triflow()
    manifest = {"a.pt": (3, _sha(b"aaa")), "sub/b.pt": (2, _sha(b"bb"))}
    monkeypatch.setattr(components.Lato2, "manifest", manifest)
    monkeypatch.setattr(components.Lato2, "models_dir", tmp_path / "lato2")
    assert lato2.missing_files() == ["a.pt", "sub/b.pt", "DINOv2's code"]
    _write(tmp_path / "lato2", {"a.pt": b"aaa", "sub/b.pt": b"b",      # b.pt has the wrong size
                                f"{models.DINOV2_CODE}/{models.DINOV2_CODE_LANDMARK}": b"#"})
    assert lato2.missing_files() == ["sub/b.pt"]
    assert "sub/b.pt" in lato2.repair_work()
    assert lato2.models_present_gb() == (3 + models.DINOV2_CODE_BYTES) / 1e9
    tri_files = {"w.safetensors": (4, _sha(b"wwww"))}
    monkeypatch.setattr(models, "TRIFLOW_FILES", tri_files)
    monkeypatch.setattr(components.Triflow, "manifest", tri_files)
    monkeypatch.setattr(components.Triflow, "models_dir", tmp_path / "triflow")
    monkeypatch.setattr(paths, "MODELS_TRIFLOW", tmp_path / "triflow")
    monkeypatch.setattr(paths, "TRIFLOW_CHECKPOINTS", tmp_path / "code" / "checkpoints")
    assert triflow.repair_work() == "some of its downloaded files are missing: w.safetensors"
    _write(tmp_path / "triflow", {"w.safetensors": b"wwww"})
    # put there by hand: checked once
    assert triflow.repair_work() == "some of its AI model files were not downloaded by Spellcast3D and are checked once"
    models.record_verified(tmp_path / "triflow" / "w.safetensors", _sha(b"wwww"))     # as a download does
    assert triflow.repair_work() is None                                  # TriFlow is not checked out here
    (tmp_path / "code" / "checkpoints").mkdir(parents=True)
    assert triflow.repair_work() == "its AI models are not linked into code\\triflow\\checkpoints yet"
    ctx = _RecordingCtx()
    triflow.repair(ctx)
    assert triflow.repair_work() is None and ctx.commands == []
    assert os.path.samefile(tmp_path / "triflow" / "w.safetensors", tmp_path / "code" / "checkpoints" / "w.safetensors")


def test_retopology_tools_install_torch_packages_and_wheels(install_dir: Path, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    ours = tmp_path / "whl" / "torchsparse-2.1.0+cu128torch2.8-cp311-cp311-win_amd64.whl"
    reused = tmp_path / "pixal3d-whl" / "flash_attn-2.8.3+cu128torch2.8-cp311-cp311-win_amd64.whl"
    monkeypatch.setattr(paths, "WHL", ours.parent)
    monkeypatch.setattr(components.Triflow, "wheels", (ours, reused))
    shared: list[str] = []
    monkeypatch.setattr(components._RetopoTool, "_share_files", lambda self, ctx, what, until: shared.append(what))
    monkeypatch.setattr(components._RetopoTool, "_fetch_models", lambda self, ctx, until: shared.append("models"))
    (install_dir / "constraints-triflow.txt").write_text("scipy==1.17.1\n", encoding="utf-8")
    (install_dir / "requirements-triflow-nodeps.txt").write_text("meshiki==0.0.7\n", encoding="utf-8")
    ctx = _RecordingCtx()
    _triflow().install(ctx)
    pips = [cmd[cmd.index("install") + 1:] for cmd in ctx.commands if "pip" in cmd]
    assert pips[0] == ["--progress-bar", "raw", *components.TORCH, "--index-url", components.TORCH_INDEX]
    assert pips[1] == ["--progress-bar", "raw", "-r", str(install_dir / "requirements-triflow.txt"),
                       "-c", str(install_dir / "constraints-triflow.txt")]
    assert pips[2][2:5] == ["--no-deps", "-r", str(install_dir / "requirements-triflow-nodeps.txt")]
    # ours and Pixal3D's alike: a wheel rebuilt under the same version must replace the installed one
    assert pips[3] == ["--progress-bar", "raw", "--no-deps", "--force-reinstall", str(ours), str(reused)]
    assert len(pips) == 4
    assert shared == ["PyTorch", "its packages", "models"]                      # torch shared before the rest
    assert ctx.commands[-1][1] == "-c" and "TriFlow packages OK" in ctx.commands[-1][2]


def test_an_incomplete_copy_is_named(install_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(components.Lato2, "wheels", (tmp_path / "gone.whl",))
    monkeypatch.setattr(components, "checkout_submodule", lambda *a, **k: None)
    with pytest.raises(InstallError, match="gone.whl is missing, so this copy of Spellcast3D is incomplete"):
        _lato2().prepare(_RecordingCtx())
    (install_dir / "requirements-lato2.txt").unlink()
    (tmp_path / "gone.whl").write_bytes(b"x")
    with pytest.raises(InstallError, match="requirements-lato2.txt is missing"):
        _lato2().prepare(_RecordingCtx())


def test_the_repo_has_the_files_the_retopology_tools_install_from() -> None:
    for c in (_lato2(), _triflow()):
        for wheel in c.wheels:
            if wheel.parent == paths.WHL or (paths.PIXAL3D_DIR / "install.py").is_file():
                assert wheel.is_file(), wheel
    for name in ("torchsparse", "pyfqmr_triflow"):
        meta = json.loads((paths.WHL / f"{name}.json").read_text(encoding="utf-8"))
        assert state.file_sha256(paths.WHL / meta["wheel"]) == meta["sha256"]
        assert "_myDrive" not in json.dumps(meta) and "scratchpad" not in json.dumps(meta)


def _requirements(name: str) -> dict[str, str]:
    """{normalised project name: its requirement line} of install/<name> (comments left out)."""
    lines = [line.split("#")[0].strip() for line in (components.INSTALL_DIR / name).read_text(
        encoding="utf-8").splitlines()]
    return {components._package_key(re.split(r"[=@ ]", line)[0]): line for line in lines if line}


def test_the_retopology_requirement_files_are_pinned_and_leave_out_what_must_not_come() -> None:
    lato2, triflow = _requirements("requirements-lato2.txt"), _requirements("requirements-triflow.txt")
    nodeps = _requirements("requirements-triflow-nodeps.txt")
    for reqs in (lato2, triflow, nodeps):
        assert all("==" in line or " @ https://" in line for line in reqs.values()), reqs
        # torch comes from the PyTorch index (components.TORCH), never from these files
        assert not {"torch", "torchvision", "torchaudio"} & set(reqs)
    # LATO.2 stays MIT: nvdiffrast's licence is non-commercial (its launcher renders with plain torch)
    assert "nvdiffrast" not in lato2 and not any("nvdiffrast" in w.name for w in _lato2().wheels)
    assert "nvdiffrast" not in _lato2().verify_code
    # xformers 0.0.32 refuses to import next to flash_attn 2.8.3: TriFlow gets none
    assert "xformers" not in triflow and not any("xformers" in x for x in _triflow().torch_extras)
    assert set(nodeps) == {"meshiki", "rootpath"} and not {"meshiki", "rootpath", "pymeshlab", "kiui"} & set(triflow)
    for module in ("meshiki", "rootpath", "torchsparse", "pyfqmr_triflow", "flash_attn", "meshlib"):
        assert module in _triflow().verify_code
    assert [w.name.split("-")[0] for w in _triflow().wheels] == ["torchsparse", "pyfqmr_triflow", "flash_attn"]
    assert [w.name.split("-")[0] for w in _lato2().wheels] == ["o_voxel", "flash_attn"]


# ---- AI models -------------------------------------------------------------------------------------------

def test_the_model_manifests_are_pinned() -> None:
    assert set(models.LATO2_FILES) == {"vflow.pt", "vvae.pt", "offset_head.pt", "tflow.pt", "tvae.pt",
                                       "voxel_encoder.pt"}
    assert set(models.TRIFLOW_FILES) == set(submodules.TRIFLOW_WEIGHT_FILES)
    for size, sha in [*models.LATO2_FILES.values(), *models.DINOV2_FILES.values(), *models.TRIFLOW_FILES.values()]:
        assert size > 1_000_000 and len(sha) == 64 and int(sha, 16) >= 0
    assert models.hf_url(models.TRIFLOW_HF_REPO, models.TRIFLOW_HF_REVISION, "sdf_vae.safetensors") == \
        "https://huggingface.co/lihcxr/TriFlow/resolve/e3403f35d0566a0487b9c8f3816bc6cdee814dca/sdf_vae.safetensors"
    assert _lato2().model_url("vvae.pt").startswith("https://huggingface.co/0x4c48/LATO.2/resolve/a91090e8")
    assert _lato2().model_url(models.DINOV2_CHECKPOINT) == models.DINOV2_URL
    record = paths.MODELS_TRIFLOW / "triflow-weights.json"     # written where the weights were first fetched
    if record.is_file():
        files = json.loads(record.read_text(encoding="utf-8"))["files"]
        assert {n: (f["size"], f["sha256"]) for n, f in files.items()} == models.TRIFLOW_FILES


def _local_dir(folder: Path, files: dict[str, bytes], revision: str, etags: dict[str, str] | None = None) -> None:
    """A folder as `hf download --local-dir` leaves it: the files plus .cache/huggingface/download/*.metadata."""
    _write(folder, files)
    for rel, data in files.items():
        meta = folder / ".cache" / "huggingface" / "download" / f"{rel}.metadata"
        meta.parent.mkdir(parents=True, exist_ok=True)
        stamp = os.stat(folder / rel).st_mtime + 0.5
        meta.write_text(f"{revision}\n{(etags or {}).get(rel, _sha(data))}\n{stamp}\n", encoding="utf-8")


def test_local_copies_are_hard_linked_after_their_sha256_is_checked(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    files = {"vflow.pt": b"v" * 10, "sub/tvae.pt": b"t" * 7, "wrong.pt": b"x" * 5}
    manifest = {rel: (len(data), _sha(data)) for rel, data in files.items()}
    manifest["wrong.pt"] = (5, _sha(b"y" * 5))                              # same size, other bytes
    src, dst = tmp_path / "Pixal3D" / "MODELS" / "lato2", tmp_path / "models" / "lato2"
    _local_dir(src, files, "rev1")
    hashed: list[Path] = []
    real = models.sha256_file
    monkeypatch.setattr(models, "sha256_file", lambda p: hashed.append(p) or real(p))
    lines: list[str] = []
    (dst / "sub").mkdir(parents=True)
    (dst / "sub" / "tvae.pt.part").write_bytes(b"t")                        # an interrupted download of it
    linked = models.link_from_local_dir(src, dst, list(manifest), manifest, lines.append, "rev1")
    assert linked == ["vflow.pt", "sub/tvae.pt"]
    assert os.path.samefile(src / "vflow.pt", dst / "vflow.pt") and not (dst / "sub" / "tvae.pt.part").exists()
    # the metadata vouched for the right ones; the file its metadata calls something else was hashed, and refused
    assert hashed == [src / "wrong.pt"] and not (dst / "wrong.pt").exists()
    assert any("not the expected file" in line for line in lines) and "linked them (0.00 GB)" in lines[-1]
    assert models.missing_files(dst, manifest) == ["wrong.pt"]
    # metadata of another revision is not trusted: the file is hashed (and fits)
    other = tmp_path / "other"
    _local_dir(other, {"vflow.pt": b"v" * 10}, "rev0")
    hashed.clear()
    assert models.link_from_local_dir(other, tmp_path / "d2", ["vflow.pt"], manifest, lines.append, "rev1") == \
        ["vflow.pt"] and hashed == [other / "vflow.pt"]
    # already there: left alone; a wrong-size file there is replaced
    before = os.stat(dst / "vflow.pt").st_mtime_ns
    assert models.link_from_local_dir(src, dst, ["vflow.pt"], manifest, lines.append) == []
    assert os.stat(dst / "vflow.pt").st_mtime_ns == before
    (dst / "sub" / "tvae.pt").unlink()
    (dst / "sub" / "tvae.pt").write_bytes(b"short")
    assert models.link_from_local_dir(src, dst, ["sub/tvae.pt"], manifest, lines.append) == ["sub/tvae.pt"]
    assert (dst / "sub" / "tvae.pt").read_bytes() == b"t" * 7
    # another drive: never copied
    monkeypatch.setattr(models, "_same_volume", lambda a, b: False)
    assert models.link_from_local_dir(src, tmp_path / "d3", ["vflow.pt"], manifest, lines.append) == []
    assert "another drive" in lines[-1] and not (tmp_path / "d3" / "vflow.pt").exists()


def test_a_huggingface_cache_blob_is_trusted_by_its_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"weights"
    blob = tmp_path / "blobs" / _sha(data)
    _write(tmp_path, {f"blobs/{_sha(data)}": data})
    monkeypatch.setattr(models, "sha256_file", lambda p: pytest.fail("hashed"))
    assert models.verified(blob, len(data), _sha(data))
    assert not models.verified(blob, len(data) + 1, _sha(data))


def test_hf_cache_sources_prefer_the_pinned_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "hub"
    folder = models.hf_repo_folder(cache, "a/b")
    (folder / "snapshots" / "pinned").mkdir(parents=True)
    (folder / "snapshots" / "main1").mkdir(parents=True)
    (folder / "refs").mkdir()
    (folder / "refs" / "main").write_text("main1", encoding="utf-8")
    monkeypatch.setattr(models, "user_hf_caches", lambda: [cache])
    assert models.hf_cache_sources("a/b", "pinned") == [folder / "snapshots" / "pinned", folder / "snapshots" / "main1"]
    assert models.hf_cache_sources("a/b", "gone") == [folder / "snapshots" / "main1"]


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves `files` with HTTP Range support; `cut` = bytes after which the first answer of a path stops."""
    files: dict[str, bytes] = {}
    cut: dict[str, int] = {}
    served: list[tuple[str, str | None]] = []

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        data = self.files.get(self.path)
        self.served.append((self.path, self.headers.get("Range")))
        if data is None:
            self.send_error(404)
            return
        start = int(self.headers["Range"].split("=")[1].rstrip("-")) if self.headers.get("Range") else 0
        body = data[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        cut = self.cut.pop(self.path, None)
        self.wfile.write(body[:cut] if cut is not None else body)
        if cut is not None:
            self.wfile.flush()
            self.connection.shutdown(2)


@pytest.fixture
def server() -> Iterator[str]:
    _RangeHandler.files, _RangeHandler.cut, _RangeHandler.served = {}, {}, []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_downloads_resume_and_are_checked(server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    data = bytes(range(256)) * 400
    _RangeHandler.files["/w.bin"] = data
    dst = tmp_path / "models" / "w.bin"
    dst.parent.mkdir(parents=True)
    dst.with_name("w.bin.part").write_bytes(data[:1000])                    # left by an earlier run
    _RangeHandler.cut["/w.bin"] = 5000                                       # and this try breaks off too
    lines: list[str] = []
    models.download_file(f"{server}/w.bin", dst, len(data), _sha(data), lines.append)
    assert dst.read_bytes() == data and not dst.with_name("w.bin.part").exists()
    assert _RangeHandler.served == [("/w.bin", "bytes=1000-"), ("/w.bin", "bytes=6000-")]
    assert "trying again" in lines[0]
    # the right size but other bytes: refused, and the partial file is dropped
    _RangeHandler.files["/bad.bin"] = b"z" * 10
    with pytest.raises(OSError, match="SHA-256 differs"):
        models.download_file(f"{server}/bad.bin", tmp_path / "bad.bin", 10, _sha(b"y" * 10), lines.append)
    assert not (tmp_path / "bad.bin").exists() and not (tmp_path / "bad.bin.part").exists()
    with pytest.raises(OSError, match="HTTP 404"):
        models.download_file(f"{server}/gone.bin", tmp_path / "gone.bin", 10, _sha(b"y" * 10), lines.append)


def test_a_continued_download_that_is_not_intact_starts_again(server: str, tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """A power cut can leave the end of a .part unwritten (zeros): the continued download then has the right size
    but not the right SHA-256. It is downloaded once more from its first byte, in the same run."""
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    data = bytes(range(256)) * 40
    _RangeHandler.files["/torn.bin"] = data
    dst = tmp_path / "torn.bin"
    dst.with_name("torn.bin.part").write_bytes(data[:1000] + b"\0" * 24)       # its last bytes never written
    lines: list[str] = []
    models.download_file(f"{server}/torn.bin", dst, len(data), _sha(data), lines.append)
    assert dst.read_bytes() == data and _RangeHandler.served == [("/torn.bin", "bytes=1024-"), ("/torn.bin", None)]
    assert any("not intact" in line for line in lines)
    assert models.is_recorded(dst, _sha(data))                              # checked: not hashed again
    # a whole download with other bytes: no second try, and the error says it arrived (not "check the internet")
    _RangeHandler.files["/other.bin"] = b"z" * 10
    with pytest.raises(models.ChecksumError):
        models.download_file(f"{server}/other.bin", tmp_path / "other.bin", 10, _sha(b"y" * 10), lines.append)
    assert [s for s in _RangeHandler.served if s[0] == "/other.bin"] == [("/other.bin", None)]


def test_models_copied_in_by_hand_are_checked_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model file that got into the models folder some other way than the installer (copied in by hand, e.g. from
    Pixal3D's MODELS\\lato2) is hashed once: the pinned one is remembered, another one is replaced."""
    manifest = {"a.pt": (3, _sha(b"aaa")), "b.pt": (3, _sha(b"bbb"))}
    folder = tmp_path / "lato2"
    _write(folder, {"a.pt": b"aaa", "b.pt": b"xxx"})                          # b.pt: the right size, other bytes
    assert models.unverified_files(folder, manifest) == ["a.pt", "b.pt"]
    lines: list[str] = []
    assert models.check_unverified(folder, manifest, lines.append) == ["b.pt"]
    assert not (folder / "b.pt").exists() and models.missing_files(folder, manifest) == ["b.pt"]
    assert models.unverified_files(folder, manifest) == [] and "is not the file" in lines[-1]
    os.utime(folder / "a.pt", ns=(1, 1))                                     # changed since: checked again
    assert models.unverified_files(folder, manifest) == ["a.pt"]
    # the component: a repair checks them before anything is linked or downloaded
    monkeypatch.setattr(components.Triflow, "manifest", manifest)
    monkeypatch.setattr(components.Triflow, "models_dir", folder)
    monkeypatch.setattr(models, "TRIFLOW_FILES", manifest)
    monkeypatch.setattr(paths, "MODELS_TRIFLOW", folder)
    monkeypatch.setattr(paths, "TRIFLOW_CHECKPOINTS", tmp_path / "no-checkout" / "checkpoints")
    monkeypatch.setattr(models, "user_hf_caches", lambda: [])
    monkeypatch.setattr(models, "EXTRA_SOURCES", [])
    downloads: list[str] = []

    def download(url: str, dst: Path, size: int, sha: str, log: Any) -> None:
        downloads.append(Path(dst).name)
        Path(dst).write_bytes(b"bbb")
        models.record_verified(Path(dst), sha)

    monkeypatch.setattr(models, "download_file", download)
    _write(folder, {"b.pt": b"xxx"})                                         # copied in by hand again
    triflow = _triflow()
    assert "checked once" in triflow.repair_work()
    triflow.repair(_RecordingCtx())
    assert downloads == ["b.pt"] and triflow.repair_work() is None and triflow.unverified() == []


def test_dinov2_code_copied_in_by_hand_is_checked_and_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code = {"hubconf.py": b"dependencies = ['torch']\n", "dinov2/__init__.py": b"__version__ = '0.0.1'\n"}
    src = tmp_path / "hub" / "facebookresearch_dinov2_main"
    _write(src, code)
    monkeypatch.setattr(models, "DINOV2_CODE_DIGEST", models.tree_digest(src))
    dst = tmp_path / "models" / "lato2" / models.DINOV2_CODE
    _write(dst, {"hubconf.py": b"# another commit\n"})
    assert not models.dinov2_code_ok(dst)
    lines: list[str] = []
    assert models.link_dinov2_code(src, dst, lines.append)                   # replaced by the pinned code
    assert (dst / "hubconf.py").read_bytes() == code["hubconf.py"] and models.dinov2_code_ok(dst)
    monkeypatch.setattr(models, "tree_digest", lambda folder: pytest.fail("hashed again"))
    assert models.dinov2_code_ok(dst)                                        # remembered
    monkeypatch.setattr(components.Lato2, "models_dir", tmp_path / "models" / "lato2")
    monkeypatch.setattr(components.Lato2, "manifest", {})
    assert _lato2().unverified() == []
    (dst / "hubconf.py").write_bytes(b"# edited\n")
    assert _lato2().unverified() == ["DINOv2's code"]


def test_dinov2_code_is_linked_or_downloaded_and_checked_file_by_file(server: str, tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    code = {"hubconf.py": b"dependencies = ['torch']\n", "dinov2/__init__.py": b"__version__ = '0.0.1'\n"}
    src = tmp_path / "hub" / "facebookresearch_dinov2_main"
    _write(src, {**code, "__pycache__/hubconf.cpython-311.pyc": b"junk"})
    digest = models.tree_digest(src)
    assert digest == models.tree_digest(tmp_path / "hub" / "facebookresearch_dinov2_main")
    monkeypatch.setattr(models, "DINOV2_CODE_DIGEST", digest)
    dst = tmp_path / "models" / "lato2" / models.DINOV2_CODE
    lines: list[str] = []
    assert models.link_dinov2_code(src, dst, lines.append)
    assert {p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file()} == set(code)   # no __pycache__
    assert os.path.samefile(src / "hubconf.py", dst / "hubconf.py") and "linked it" in lines[-1]
    shutil.rmtree(dst)
    (src / "hubconf.py").write_bytes(b"changed")                            # another commit's code: not used
    assert not models.link_dinov2_code(src, dst, lines.append) and not dst.exists()
    # the download: GitHub's archive of the pinned commit, unpacked and checked the same way
    monkeypatch.setattr(models, "DINOV2_CODE_COMMIT", "c0ffee")
    monkeypatch.setattr(models, "DINOV2_CODE_URL", f"{server}/dinov2.zip")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for rel, data in code.items():
            z.writestr(f"dinov2-c0ffee/{rel}", data)
    _RangeHandler.files["/dinov2.zip"] = buf.getvalue()
    models.download_dinov2_code(dst, lines.append)
    assert models.tree_digest(dst) == digest and not dst.with_name("dinov2-code.zip").exists()
    shutil.rmtree(dst)
    monkeypatch.setattr(models, "DINOV2_CODE_DIGEST", "0" * 64)
    with pytest.raises(OSError, match="not that commit's code"):
        models.download_dinov2_code(dst, lines.append)
    assert not dst.exists() and not dst.with_name(dst.name + ".partial").exists()


def test_models_are_linked_from_this_pc_before_anything_is_downloaded(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    files = {"vflow.pt": b"v" * 10, models.DINOV2_CHECKPOINT: b"d" * 6}
    manifest = {rel: (len(data), _sha(data)) for rel, data in files.items()}
    lato2 = _lato2()
    monkeypatch.setattr(components.Lato2, "manifest", manifest)
    monkeypatch.setattr(components.Lato2, "models_dir", tmp_path / "models" / "lato2")
    monkeypatch.setattr(models, "LATO2_FILES", {"vflow.pt": manifest["vflow.pt"]})
    monkeypatch.setattr(models, "DINOV2_FILES", {models.DINOV2_CHECKPOINT: manifest[models.DINOV2_CHECKPOINT]})
    monkeypatch.setattr(models, "user_hf_caches", lambda: [])
    hub = tmp_path / "torch" / "hub"
    _write(hub, {"checkpoints/dinov2_vitl14_reg4_pretrain.pth": b"d" * 6,
                 "facebookresearch_dinov2_main/hubconf.py": b"#"})
    monkeypatch.setattr(models, "torch_hub_dirs", lambda: [hub])
    monkeypatch.setattr(models, "DINOV2_CODE_DIGEST", models.tree_digest(hub / "facebookresearch_dinov2_main"))
    other = tmp_path / "other-spellcast3d" / "models"                        # --models-from: another copy's folder
    _write(other / "lato2", {"vflow.pt": b"v" * 10})
    monkeypatch.setattr(models, "EXTRA_SOURCES", [other])
    downloads: list[str] = []
    monkeypatch.setattr(models, "download_file", lambda url, dst, *a: downloads.append(url))
    ctx = _RecordingCtx()
    lato2.repair(ctx)
    assert lato2.missing_files() == [] and downloads == []
    assert os.path.samefile(other / "lato2" / "vflow.pt", tmp_path / "models" / "lato2" / "vflow.pt")
    assert os.path.samefile(hub / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth",
                            tmp_path / "models" / "lato2" / models.DINOV2_CHECKPOINT)
    # what is not on this PC is downloaded; a failed download is an artist-facing error
    (tmp_path / "models" / "lato2" / "vflow.pt").unlink()
    monkeypatch.setattr(models, "EXTRA_SOURCES", [])

    def fail(url: str, dst: Path, *a: Any) -> None:
        downloads.append(url)
        raise OSError("vflow.pt: timed out (0 of 10 bytes)")

    monkeypatch.setattr(models, "download_file", fail)
    with pytest.raises(InstallError, match="Downloading the AI models of LATO.2 failed .*timed out.*resumes"):
        lato2.repair(ctx)
    assert downloads == [lato2.model_url("vflow.pt")]


def test_a_dry_run_only_prints_where_the_models_come_from(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _RecordingCtx(dry_run=True)
    monkeypatch.setattr(models, "download_file", lambda *a: pytest.fail("downloaded"))
    _triflow()._fetch_models(ctx, until=1.0)
    assert len(ctx.lines) == 3 and all("[dry-run]" in line and "huggingface.co/lihcxr/TriFlow" in line
                                       for line in ctx.lines)


def test_triflow_weights_are_hard_linked_into_its_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = {"flow.safetensors": b"f" * 9, "vae.safetensors": b"v" * 4}
    monkeypatch.setattr(models, "TRIFLOW_FILES", {rel: (len(d), _sha(d)) for rel, d in files.items()})
    monkeypatch.setattr(paths, "MODELS_TRIFLOW", tmp_path / "models" / "triflow")
    monkeypatch.setattr(paths, "TRIFLOW_CHECKPOINTS", tmp_path / "code" / "triflow" / "checkpoints")
    paths.TRIFLOW_CHECKPOINTS.mkdir(parents=True)
    assert models.triflow_link_work() is None                               # nothing downloaded yet
    _write(paths.MODELS_TRIFLOW, files)
    assert models.triflow_link_work() is not None
    (paths.TRIFLOW_CHECKPOINTS / "vae.safetensors").write_bytes(b"old!")    # a copy of its own: replaced
    lines: list[str] = []
    assert models.link_triflow_checkpoints(lines.append) and models.triflow_link_work() is None
    for name in files:
        assert os.path.samefile(paths.MODELS_TRIFLOW / name, paths.TRIFLOW_CHECKPOINTS / name)
    assert not list(paths.TRIFLOW_CHECKPOINTS.glob("*.link")) and lines == []
    monkeypatch.setattr(models, "_same_volume", lambda a, b: False)        # models/triflow on another drive
    (paths.TRIFLOW_CHECKPOINTS / "flow.safetensors").unlink()
    assert models.link_triflow_checkpoints(lines.append)
    assert (paths.TRIFLOW_CHECKPOINTS / "flow.safetensors").read_bytes() == files["flow.safetensors"]
    # the copy is known as one: not copied again at every start (no endless repair)
    assert models.triflow_link_work() is None and not os.path.samefile(
        paths.MODELS_TRIFLOW / "flow.safetensors", paths.TRIFLOW_CHECKPOINTS / "flow.safetensors")
    copies: list[str] = []
    real_copy2 = models.shutil.copy2
    monkeypatch.setattr(models.shutil, "copy2", lambda a, b: copies.append(str(a)) or real_copy2(a, b))
    assert models.link_triflow_checkpoints(lines.append) and copies == []
    # a FAT or exFAT drive (or an antivirus) refuses hard links: copied instead, and known as a copy after
    monkeypatch.setattr(models, "_same_volume", lambda a, b: True)

    def no_links(src: Any, dst: Any) -> None:
        raise OSError(1, "Incorrect function")

    monkeypatch.setattr(models.os, "link", no_links)
    (paths.TRIFLOW_CHECKPOINTS / "vae.safetensors").unlink()
    assert models.link_triflow_checkpoints(lines.append)
    assert (paths.TRIFLOW_CHECKPOINTS / "vae.safetensors").read_bytes() == files["vae.safetensors"]
    assert models.triflow_link_work() is None and "copying it instead" in lines[-1]


# ---- venvs share identical files -------------------------------------------------------------------------

def _venv(root: Path, files: dict[str, bytes]) -> Path:
    _write(root / "Lib" / "site-packages", files)
    return root


def test_share_identical_files_links_only_the_new_venvs_big_identical_files(tmp_path: Path) -> None:
    big = b"B" * 2000
    old = _venv(tmp_path / "forge", {"torch/lib/torch_cuda.dll": big, "torch/lib/other.dll": b"O" * 2000,
                                     "torch/small.py": b"s", "flash_attn_2_cuda.cp311-win_amd64.pyd": big + b"f",
                                     "numpy.libs/openblas.dll": big + b"n", "gradio/big.js": big})
    new = _venv(tmp_path / "lato2", {"torch/lib/torch_cuda.dll": big, "torch/lib/other.dll": b"X" * 2000,
                                     "torch/small.py": b"s", "flash_attn_2_cuda.cp311-win_amd64.pyd": big + b"f",
                                     "numpy.libs/openblas.dll": big + b"n", "gradio/big.js": big,
                                     "torch-2.8.0+cu128.dist-info/RECORD": big,
                                     "torch/__pycache__/x.pyc": big, "torch/lib/only_here.dll": big})
    old_bytes = {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}
    assert venvs.share_identical_files(new, [old], min_bytes=1000, dry_run=True) == (3, 3 * 2000 + 2)
    assert not any(os.stat(p).st_nlink > 1 for p in new.rglob("*") if p.is_file())    # a dry run changes nothing
    count, saved = venvs.share_identical_files(new, [old], min_bytes=1000)
    assert (count, saved) == (3, 3 * 2000 + 2)
    sp_new, sp_old = new / "Lib" / "site-packages", old / "Lib" / "site-packages"
    for rel in ("torch/lib/torch_cuda.dll", "flash_attn_2_cuda.cp311-win_amd64.pyd", "numpy.libs/openblas.dll"):
        assert os.path.samefile(sp_new / rel, sp_old / rel)
    for rel in ("torch/lib/other.dll", "torch/small.py", "gradio/big.js", "torch-2.8.0+cu128.dist-info/RECORD",
                "torch/__pycache__/x.pyc", "torch/lib/only_here.dll"):
        assert not os.path.samefile(sp_new / rel, sp_old / rel) if (sp_old / rel).exists() else True
    assert {p: p.read_bytes() for p in old.rglob("*") if p.is_file()} == old_bytes    # the target venv: untouched
    assert (sp_new / "torch/lib/other.dll").read_bytes() == b"X" * 2000
    assert not list(new.rglob("*.sc3dlink"))
    assert venvs.share_identical_files(new, [old], min_bytes=1000) == (0, 0)          # linked already
    # pip replacing a package in one venv (delete + write) leaves the other venv's copy as it was
    (sp_new / "torch/lib/torch_cuda.dll").unlink()
    (sp_new / "torch/lib/torch_cuda.dll").write_bytes(b"new torch")
    assert (sp_old / "torch/lib/torch_cuda.dll").read_bytes() == big


def test_torch_twins_are_the_venvs_with_the_same_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "VENVS", tmp_path)
    _venv(tmp_path / "forge", {"torch-2.8.0+cu128.dist-info/METADATA": b"m"})
    _venv(tmp_path / "pixal3d", {"torch-2.8.0+cu128.dist-info/METADATA": b"m"})
    _venv(tmp_path / "imesh", {"numpy/x": b"n"})
    _venv(tmp_path / "triflow", {"torch-2.9.0+cu128.dist-info/METADATA": b"m"})
    assert venvs.torch_twins("lato2", components.TORCH_DIST) == [tmp_path / "forge", tmp_path / "pixal3d"]
    assert venvs.torch_twins("forge") == [tmp_path / "pixal3d"]
    assert venvs.torch_twins("triflow") == [] and venvs.torch_twins("imesh") == []


def test_a_retopology_venv_shares_its_files_and_still_imports_torch(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "VENVS", tmp_path)
    big = b"T" * 2_000_000
    _venv(tmp_path / "forge", {"torch-2.8.0+cu128.dist-info/METADATA": b"m", "torch/lib/c.dll": big})
    _venv(tmp_path / "lato2", {"torch-2.8.0+cu128.dist-info/METADATA": b"m", "torch/lib/c.dll": big})
    ctx = _RecordingCtx()
    _lato2()._share_files(ctx, "PyTorch", until=0.3)
    assert os.path.samefile(tmp_path / "forge/Lib/site-packages/torch/lib/c.dll",
                            tmp_path / "lato2/Lib/site-packages/torch/lib/c.dll")
    assert "1 big file(s) of PyTorch (0.0 GB) are the same as in code\\venvs\\forge" in ctx.lines[-1]
    assert ctx.commands[-1][-1].startswith("import torch")                  # checked right after linking


def test_forge_and_pixal3d_share_their_identical_files_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "VENVS", tmp_path)
    big = b"T" * 2_000_000
    for venv in ("forge", "pixal3d", "lato2"):
        _venv(tmp_path / venv, {"torch-2.8.0+cu128.dist-info/METADATA": b"m", "torch/lib/c.dll": big})
    ctx = _RecordingCtx()
    components.share_torch_files(ctx, "pixal3d")
    sp = "Lib/site-packages/torch/lib/c.dll"
    assert os.path.samefile(tmp_path / "pixal3d" / sp, tmp_path / "forge" / sp)
    assert "1 big file(s)" in ctx.lines[-1]
    components.share_torch_files(_RecordingCtx(dry_run=True), "lato2")        # a dry run changes nothing
    assert not os.path.samefile(tmp_path / "lato2" / sp, tmp_path / "forge" / sp)


def test_preflight_budgets_the_disk_a_shared_venv_gives_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "VENVS", tmp_path / "venvs")
    monkeypatch.setattr(components.Lato2, "models_present_gb", lambda self: self.models_gb)   # models on disk
    monkeypatch.setattr(components.Triflow, "models_present_gb", lambda self: 0.0)
    lato2, triflow = _lato2(), _triflow()
    need = lato2.disk_needed_gb()
    assert need == pytest.approx(lato2.size_gb - lato2.models_gb)
    assert lato2.disk_kept_gb(torch_twin=False) == need
    assert lato2.disk_kept_gb(torch_twin=True) == pytest.approx(need - components.SHARED_TORCH_GB)
    assert components.Forge().disk_kept_gb(torch_twin=True) == components.Forge().disk_needed_gb()
    # --reinstall of an installed venv: its files other venvs link to stay on disk, so it needs a first install's
    _write(tmp_path / "venvs" / "forge" / "Scripts", {"python.exe": b"x"})
    forge = components.Forge()
    monkeypatch.setattr(components.state, "load", lambda: {"forge": {"status": "installed"}})
    assert forge.disk_needed_gb() == pytest.approx(min(components.UPDATE_GB, forge.size_gb))
    assert forge.disk_needed_gb(fresh=True) == pytest.approx(forge.size_gb - forge.models_gb)
    monkeypatch.setattr(components.state, "load", lambda: {})

    class Usage:
        free = int((need + triflow.disk_needed_gb() - components.SHARED_TORCH_GB + 2.5) * 1e9)

    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda p: Usage)
    r = preflight.Report()
    preflight._check_disk(r, [lato2, triflow])
    assert set(r.blocked) == {"triflow"}                                    # no venv with that torch yet
    _venv(tmp_path / "venvs" / "forge", {f"{components.TORCH_DIST}/METADATA": b"m"})
    r = preflight.Report()
    preflight._check_disk(r, [lato2, triflow])
    assert r.blocked == {}                                                  # LATO.2 keeps ~7 GB less


# ---- TriFlow's nested submodules, byte-identical ------------------------------------------------------------

def test_nested_submodules_are_checked_out_level_by_level(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    triflow = tmp_path / "triflow"
    (triflow / ".git").mkdir(parents=True)
    calls: list[tuple[list[str], Path]] = []
    submodules.init_triflow("git", lambda argv, cwd: calls.append((argv, cwd)), triflow)
    commands = [" ".join(argv[1:]).replace(str(triflow), "T") for argv, _cwd in calls]
    assert commands == [
        "-C T config core.autocrlf false",
        "-c core.autocrlf=false -C T submodule update --init --depth 1 -- third_party/TRELLIS third_party/Direct3D-S2 "
        "third_party/pyfqmr-triflow",
        "-c core.autocrlf=false -C T\\third_party\\TRELLIS submodule update --init --depth 1 -- "
        "trellis/representations/mesh/flexicubes",
    ]
    assert not any("--recursive" in c for c in commands)
    # everything checked out at its pins: only the settings are made sure of
    for rel, landmark in submodules.TRIFLOW_LANDMARKS.items():
        _write(triflow / rel, {landmark: b"x", ".git": b"gitdir: x"})
    monkeypatch.setattr(submodules, "pinned_commit", lambda git, repo, sub: "a" * 40)
    monkeypatch.setattr(state, "git_head", lambda repo: "a" * 40)
    calls.clear()
    submodules.init_triflow("git", lambda argv, cwd: calls.append((argv, cwd)), triflow)
    assert [argv[-3:] for argv, _cwd in calls] == [["config", "core.autocrlf", "false"]] * 5
    # one of them is not at the commit TriFlow pins (e.g. after an update): only it is updated
    monkeypatch.setattr(state, "git_head", lambda repo: "b" * 40 if repo.name == "Direct3D-S2" else "a" * 40)
    calls.clear()
    submodules.init_triflow("git", lambda argv, cwd: calls.append((argv, cwd)), triflow)
    updates = [argv for argv, _cwd in calls if "update" in argv]
    assert len(updates) == 1 and updates[0][-2:] == ["--", "third_party/Direct3D-S2"]


def test_a_long_install_path_blocks_only_triflow_when_git_must_download_its_parts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    triflow = tmp_path / "triflow"
    assert submodules.triflow_checkout_work("git", triflow) == ["."]            # TriFlow itself not checked out
    _write(triflow, {"inference.py": b"x"})
    (triflow / ".git").mkdir()
    for rel, landmark in submodules.TRIFLOW_LANDMARKS.items():
        _write(triflow / rel, {landmark: b"x", ".git": b"gitdir: x"})
    monkeypatch.setattr(submodules, "pinned_commit", lambda git, repo, sub: "a" * 40)
    monkeypatch.setattr(state, "git_head", lambda repo: "b" * 40 if repo.name == "flexicubes" else "a" * 40)
    flexicubes = "third_party/TRELLIS/trellis/representations/mesh/flexicubes"
    assert submodules.triflow_checkout_work("git", triflow) == [flexicubes]
    assert submodules.triflow_checkout_work(None, triflow) == []                # no git: only what is missing
    # the deepest path git writes is ~165 characters below the root; Windows stops at 259
    assert submodules.TRIFLOW_MAX_ROOT_LEN == 94
    assert 259 - 1 - len(submodules.TRIFLOW_GIT_DEEPEST) == submodules.TRIFLOW_MAX_ROOT_LEN

    lato2, tri = _lato2(), _triflow()
    long_root = tmp_path / ("L" * 100)
    (long_root / ".git").mkdir(parents=True)
    work = [flexicubes]
    monkeypatch.setattr(submodules, "triflow_checkout_work", lambda git, triflow=None: work)
    monkeypatch.setattr(paths, "ROOT", long_root)
    r = preflight.Report()
    preflight._check_triflow_path(r, [lato2, tri])
    assert set(r.blocked) == {"triflow"} and f"too long ({len(str(long_root))} characters)" in r.blocked["triflow"]
    assert "at most 94 characters" in r.blocked["triflow"] and "other tools are not affected" in r.blocked["triflow"]
    for case in ("checked out already", "short root", "release copy (no git data)", "not installing TriFlow"):
        work = [] if case == "checked out already" else [flexicubes]
        monkeypatch.setattr(paths, "ROOT", Path("C:/sc3d") if case == "short root" else
                            tmp_path / ("R" * 100) if case.startswith("release") else long_root)
        r = preflight.Report()
        preflight._check_triflow_path(r, [lato2] if case == "not installing TriFlow" else [lato2, tri])
        assert r.blocked == {}, case


def _git(repo: Path, *args: str) -> str:
    return subprocess.run([GIT, "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def _repo(path: Path, files: dict[str, bytes]) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    for key, value in (("user.name", "t"), ("user.email", "t@t"), ("core.autocrlf", "false")):
        _git(path, "config", key, value)
    _write(path, files)
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "c")
    return path


@needs_git
def test_byte_identity_is_checked_file_by_file_nested_repos_included(tmp_path: Path) -> None:
    inner = _repo(tmp_path / "inner-src", {"flexi.py": b"a = 1\nb = 2\n"})
    outer = _repo(tmp_path / "outer", {"inference.py": b"import x\nprint(1)\n", ".gitignore": b"__pycache__/\n",
                                       "checkpoints/.gitignore": b"*\n!.gitignore\n"})
    subprocess.run([GIT, "-c", "protocol.file.allow=always", "-c", "core.autocrlf=false", "-C", str(outer),
                    "submodule", "add", "-q", inner.as_uri(), "third_party/inner"], check=True, capture_output=True)
    for key, value in (("user.name", "t"), ("user.email", "t@t"), ("core.autocrlf", "false")):
        _git(outer / "third_party" / "inner", "config", key, value)
    _git(outer, "commit", "-qm", "sub")
    allowed = ("checkpoints/w.safetensors",)
    assert submodules.byte_identity_problems(GIT, outer, allowed) == []
    (outer / "checkpoints" / "w.safetensors").write_bytes(b"weights")      # ignored by the tool's own .gitignore
    assert submodules.byte_identity_problems(GIT, outer, allowed) == []
    assert submodules.byte_identity_problems(GIT, outer) == [
        "checkpoints/w.safetensors: added (not part of its commit)"]
    # CRLF written by a checkout with autocrlf=true: git status may not even notice, the bytes check does
    (outer / "inference.py").write_bytes(b"import x\r\nprint(1)\r\n")
    nested_file = outer / "third_party" / "inner" / "flexi.py"
    nested_file.write_bytes(b"a = 1\nb = 3\n")                              # an edit in a nested checkout
    (outer / "third_party" / "inner" / "__pycache__").mkdir()
    (outer / "third_party" / "inner" / "__pycache__" / "flexi.pyc").write_bytes(b"c")
    crlf: list[Path] = []
    problems = submodules.byte_identity_problems(GIT, outer, allowed, prefix="code/triflow/", crlf_only=crlf)
    assert problems == ["code/triflow/inference.py: differs from its commit",
                        "code/triflow/third_party/inner/flexi.py: differs from its commit",
                        "code/triflow/third_party/inner/__pycache__/: added (not part of its commit)"]
    assert crlf == [outer / "inference.py"]                                 # only this one can be written back
    # the nested checkout moved to another commit
    nested_file.write_bytes(b"a = 1\nb = 2\n")
    shutil.rmtree(outer / "third_party" / "inner" / "__pycache__")
    _git(outer / "third_party" / "inner", "commit", "-qam", "moved", "--allow-empty")
    (outer / "inference.py").write_bytes(b"import x\nprint(1)\n")
    problems = submodules.byte_identity_problems(GIT, outer, allowed)
    assert len(problems) == 1 and "third_party/inner: at " in problems[0] and "but outer pins" in problems[0]
    (outer / "inference.py").unlink()
    assert "inference.py: missing" in submodules.byte_identity_problems(GIT, outer)[0]
    shutil.rmtree(outer / "third_party" / "inner")
    (outer / "third_party" / "inner").mkdir()
    assert "third_party/inner: not checked out" in submodules.byte_identity_problems(GIT, outer)


@needs_git
def test_triflow_prepare_writes_back_line_endings_git_changed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path / "triflow", {"inference.py": b"a\nb\n", "edited.py": b"x\n"})
    (repo / "inference.py").write_bytes(b"a\r\nb\r\n")
    (repo / "edited.py").write_bytes(b"y\n")
    monkeypatch.setattr(components.Triflow, "submodule", repo)
    monkeypatch.setattr(submodules, "triflow_nested_dirs", lambda triflow=None: [])

    class Ctx2(_RecordingCtx):
        def run(self, argv: list[str], **kw: Any) -> None:
            super().run(argv, **kw)
            subprocess.run(argv, check=True, cwd=kw["cwd"], capture_output=True)

    ctx = Ctx2()
    _triflow()._check_untouched(ctx, GIT)
    assert (repo / "inference.py").read_bytes() == b"a\nb\n"                # written as committed
    assert (repo / "edited.py").read_bytes() == b"y\n"                      # a real edit is only reported
    assert ctx.commands[0][1:4] == ["-c", "core.autocrlf=false", "-C"]
    assert ctx.commands[0][-4:] == ["checkout-index", "-f", "--", "inference.py"]
    assert "edited.py: differs" in ctx.lines[-1] and "AUDI AG" in ctx.lines[-1]


def _gitmodules() -> dict[str, dict[str, str]]:
    out = subprocess.run([GIT, "-C", str(paths.ROOT), "config", "-f", ".gitmodules", "--list"], check=True,
                         capture_output=True, text=True).stdout
    entries: dict[str, dict[str, str]] = {}
    for line in out.splitlines():
        key, _, value = line.partition("=")
        name, _, field = key[len("submodule."):].rpartition(".")
        entries.setdefault(name, {})[field] = value
    return entries


@needs_repo
def test_the_retopology_submodules_are_pinned() -> None:
    entries = _gitmodules()
    assert entries["lato2"] == {"path": "code/lato2", "url": "https://github.com/IgorAherne/LATO2-stableprojectorz.git",
                                "branch": "spellcast3d", "ignore": "untracked", "shallow": "true"}
    assert entries["triflow"] == {"path": "code/triflow", "url": "https://github.com/IgorAherne/triflow.git",
                                  "branch": "spellcast3d", "ignore": "untracked", "shallow": "true"}
    staged = _git(paths.ROOT, "ls-files", "-s", "--", "code/lato2", "code/triflow").split()
    assert staged[:2] == ["160000", "fbb1f5a5755e6db8700cf6922fd506830b7cdccd"]
    assert staged[4:6] == ["160000", "5cf7f00e4f5d5745a1cafbb21ff5d7d28bb4f878"]


@needs_git
def test_triflow_here_is_byte_identical_to_its_commits() -> None:
    """The licence rule, on this checkout: nothing changed or added in TriFlow's tree (nested ones included);
    only its weights are linked into checkpoints/ (ignored by its own .gitignore)."""
    if not (paths.TRIFLOW_DIR / ".git").exists() or submodules.triflow_missing():
        pytest.skip("code/triflow and its nested submodules are not checked out")
    assert submodules.byte_identity_problems(GIT, paths.TRIFLOW_DIR, submodules.TRIFLOW_ALLOWED_EXTRA) == []
    for repo in [paths.TRIFLOW_DIR, *submodules.triflow_nested_dirs()]:
        assert _git(repo, "config", "--local", "core.autocrlf").strip() == "false", repo
    assert _git(paths.TRIFLOW_DIR, "status", "--porcelain").strip() == ""
    heads = submodules.triflow_heads()
    assert heads["third_party/TRELLIS"].startswith("f17fdf12")
    assert heads["third_party/Direct3D-S2"].startswith("a1cf235b")
    assert heads["third_party/pyfqmr-triflow"].startswith("c93a7671")
    assert heads["third_party/TRELLIS/trellis/representations/mesh/flexicubes"].startswith("815e075a")


# ---- prepare_release -------------------------------------------------------------------------------------

@functools.cache
def _prepare_release() -> Any:
    sys.path.insert(0, str(paths.ROOT))
    try:
        import prepare_release
    finally:
        sys.path.remove(str(paths.ROOT))
    return prepare_release


def test_prepare_release_knows_the_retopology_tools() -> None:
    pr = _prepare_release()
    assert pr.TRIFLOW_WHEELS == tuple(w.name for w in _triflow().wheels if w.parent == paths.WHL)
    assert pr.wheel_sidecar(pr.TRIFLOW_WHEELS[0]) == paths.WHL / "torchsparse.json"
    assert pr.SUBMODULE_LANDMARKS["code/lato2"] == _lato2().marker
    assert pr.SUBMODULE_LANDMARKS["code/triflow/third_party/TRELLIS"] == "trellis/__init__.py"
    assert pr.TRIFLOW_REPOS[0] == "code/triflow" and len(pr.TRIFLOW_REPOS) == 5
    assert pr.triflow_init_commands() == [
        "git -c core.autocrlf=false -C code/triflow submodule update --init --depth 1 -- third_party/TRELLIS "
        "third_party/Direct3D-S2 third_party/pyfqmr-triflow",
        "git -c core.autocrlf=false -C code/triflow/third_party/TRELLIS submodule update --init --depth 1 -- "
        "trellis/representations/mesh/flexicubes"]
    for script in pr.WHEEL_BUILD_SCRIPTS.values():
        assert (paths.ROOT / script).is_file(), script
    # "README - LICENSE.txt" must name every tool a stage runs on, and what TriFlow brings, before a release
    from spellcast.adapters import build_adapters

    assert set(pr.LICENSE_TOOL_NAMES) == {t for tools in paths.STAGE_TOOLS.values() for t in tools}
    assert pr.LICENSE_TOOL_NAMES == {tool: adapter.display_name for tool, adapter in build_adapters().items()}
    assert pr.license_text_gaps("forge neo PIXAL3D Instant Meshes lato.2") == [
        "TriFlow", "Qwen-Image 2.1", "ADPNCL", "MeshLib", "TRELLIS", "DINOv2", "torchsparse", "sparsehash",
        "pyfqmr-triflow", "triflow_server.py", "triflow_stubs", "attn_shim.py", "spconv_shim.py", "_loopback_guard.py",
        "Qwen RESEARCH LICENSE AGREEMENT", "Hangzhou Tongyi Laboratory", "stable-diffusion.cpp", "Qwen3-VL",
        "Qwen-Image-2.1-Turbo", "FLUX Kommunity License", "Black Forest Labs",
        "Hi3D Open License Agreement Version 1.0", "OCEANO HK HOLDING LIMITED", "Powered by Hi3D 1.0",
        "Acceptable Use Policy", "CLIP", "dinov2-large"]
    assert pr.license_text_gaps(" ".join(pr.LICENSE_MUST_NAME).upper()) == []


@needs_repo
def test_the_gitignore_keeps_the_retopology_files() -> None:
    pr = _prepare_release()
    probes = [(p, want) for p, want in pr.IGNORE_PROBES
              if any(k in p for k in ("lato2", "triflow", "torchsparse", "pyfqmr", "wheels"))]
    assert len(probes) >= 12
    ignored = set(subprocess.run([GIT, "-C", str(paths.ROOT), "check-ignore", "--no-index", *(p for p, _ in probes)],
                                 capture_output=True, text=True).stdout.split("\n"))
    assert [(p, want) for p, want in probes if (p in ignored) != want] == []
