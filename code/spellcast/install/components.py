"""The things the installer sets up, in install order: hub, imesh, the image generator (qwen or forge, the one
spellcast3d-config.ini picks: install/main.py), pixal3d, lato2, triflow. STDLIB ONLY."""

from __future__ import annotations

import abc
import ast
import functools
import json
import os
import re
import shutil
import sys
import urllib.parse
from pathlib import Path

from .. import config, forge_config, gpu, paths
from ..forge_config import forge_version_uid
from . import layout, models, progress, sdcpp, state, submodules, venvs
from .runner import Ctx, InstallError

# Bump only when a step shared by ALL components changes (venvs.ensure, how pip is run): it reinstalls
# all of them. For one component's own install() steps, bump that class's `recipe` instead.
INSTALLER_VERSION = "1"

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
# One build with kernels for sm_61..sm_120 on Windows (GTX 1080 .. RTX 50xx). Pixal3D's
# prebuilt CUDA wheels are ABI-locked to exactly this torch.
TORCH = ("torch==2.8.0+cu128", "torchvision==0.23.0+cu128")
TORCHAUDIO = "torchaudio==2.8.0+cu128"
# The folder pip records that torch in (venvs.torch_dist): venvs with it share their identical files.
TORCH_DIST = f"torch-{TORCH[0].partition('==')[2]}.dist-info"
FORGE_GRADIO = "4.40.0"
# Pixal3D pins gradio 6.0.1, which lacks gradio.Server (added in 6.10). --no-deps keeps every
# other pin of Pixal3D; these three are the only packages that change.
PIXAL3D_GRADIO = ("gradio==6.14.0", "gradio_client==2.5.0", "hf-gradio==0.4.1")
PIXAL3D_SPACES = "spaces==0.51.3"
# Hi3D's packages in Pixal3D's venv (its launcher runs Hi3D's refiner there): spconv's sparse convolutions - the cu126
# build the other tools use with torch cu128 (spconv-cu120 crashes under numpy 2) - with cumm below 0.8 (spconv 2.3.8
# needs it) and their dependencies, and pyfqmr. The versions of the tested venv (constraints-pixal3d.txt has them too),
# installed --no-deps: nothing else in the venv moves (numpy, sympy, ninja and requests are Pixal3D's own).
HI3D_PACKAGES = ("spconv-cu126==2.3.8", "cumm-cu126==0.7.11", "pccm==0.4.16", "ccimport==0.4.4", "fire==0.7.1",
                 "pybind11==3.1.0", "lark==1.3.1", "portalocker==4.4.1", "termcolor==3.3.0", "pyfqmr==0.5.0")
HI3D_PACKAGES_GB = 0.3  # the packages above, installed
# What the installer says when Hi3D's code or packages did not arrive: Pixal3D works on (its own shapes), Hi3D's props
# fail with the hub's words until they are here (the next start tries again). Its AI models are no part of the
# install: the 3D stage downloads them when a prop is set to Hi3D (spellcast/model_downloads.py "hi3d").
HI3D_LATER = "Pixal3D works without it; props set to Hi3D wait for it. Spellcast3D tries again at its next start."
# huggingface_hub 0.36 downloads through Xet storage (parallel, retried in chunks: faster on a slow or
# flaky line) only when hf_xet is installed, and its own dependency on it never matches on Windows
# (platform_machine is "AMD64" there, the marker asks for "amd64"). One abi3 wheel, 4 MB; it needs nothing else.
# Tried with huggingface_hub 0.36.2 (its range: hf-xet >=1.1.3,<2.0.0).
HF_XET = "hf-xet==1.6.0"

HUB_REQUIREMENTS = paths.CODE / "requirements-hub.txt"
IMESH_REQUIREMENTS = Path(__file__).with_name("requirements-imesh.txt")
IMESH_WHEEL_GLOB = "instant_meshes_brush-*-cp311-*.whl"
# Versions of a tested Pixal3D venv, used as constraints by a fresh install (see the file's header).
PIXAL3D_CONSTRAINTS = Path(__file__).with_name("constraints-pixal3d.txt")
# Pixal3D's own models folder, a junction to models/pixal3d (install/models.py). HF_HOME = this folder
# (install.py's HF_HOME is <repo>/models: the same folder, NTFS ignores case).
PIXAL3D_MODELS = paths.PIXAL3D_MODELS_LINK
PIXAL3D_MODELS_BYTES = 27_500_000_000  # all of it (measured 27.497 GB at c809a01; the models are the same at the pin), for the download progress
# What the installer says when the faster-loading copies of the flow models (models.PIXAL3D_COPIES_MODULE) were
# not made: they are no download, and no part of what "missing" means.
PIXAL3D_COPIES_LATER = ("Pixal3D works without them: it reads the original files, which takes longer. They are made "
                        "the next time Pixal3D is installed or updated.")
PIXAL3D_HF_REPO = "TencentARC/Pixal3D"
MOGE_HF_REPO = "Ruicheng/moge-2-vitl"
# An interrupted download this big is continued over plain HTTPS: huggingface_hub resumes a partial file
# there, while an Xet download starts the file again from its first byte (the files are 5.5 GB).
RESUME_OVER_HTTPS_BYTES = 200_000_000
# Pixal3D's install.py ends with this hint for people who run its app by hand.
PIXAL3D_ENV_HINT = "Set these before running"
PIXAL3D_ENV_NOTE = "(Spellcast3D sets those for you - ignore the lines above)"
# Pixal3D install.py's PIXAL3D_SINGLE_VIEW_PATTERNS (unchanged since c809a01), used if it cannot be read from there.
_PIXAL3D_PATTERNS_FALLBACK = (
    "pipeline.json",
    "ckpts/ss_dec_conv3d_16l8_fp16.*",
    "ckpts/ss_flow_img_dit_1_3B_64_bf16.*",
    "ckpts/shape_dec_next_dc_f16c32_fp16.*",
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.*",
    "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.*",
    "ckpts/tex_dec_next_dc_f16c32_fp16.*",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.*",
)

# The retopology tools (LATO.2, TriFlow). Each has its own venv and its pinned packages in this folder:
# requirements-<id>.txt (pip -r), plus when present constraints-<id>.txt (pip -c) and requirements-<id>-nodeps.txt
# (installed with --no-deps: packages whose declared dependencies must not come along). Then prebuilt wheels,
# with --no-deps. Pixal3D's cp311 / torch 2.8.0+cu128 wheels are reused from code/pixal3d/whl: flash_attn (kernels
# for sm_80+; TriFlow imports it on every GPU) and o_voxel (LATO.2's surface voxels). Not nvdiffrast: its licence is
# non-commercial, and LATO.2's launcher draws its conditioning picture with plain torch (launchers/lato2_render.py).
INSTALL_DIR = Path(__file__).parent
PIXAL3D_WHL = paths.PIXAL3D_DIR / "whl"
FLASH_ATTN_WHEEL = PIXAL3D_WHL / "flash_attn-2.8.3+cu128torch2.8-cp311-cp311-win_amd64.whl"
O_VOXEL_WHEEL = PIXAL3D_WHL / "o_voxel-0.0.1-cp311-cp311-win_amd64.whl"
# Built for TriFlow from their pinned sources (code/release/wheels/*/build_*_win.bat; provenance in
# code/whl/<project>.json): torchsparse (Direct3D-S2's sparse convolutions, sm_61..sm_120 + PTX) and
# pyfqmr-triflow (TriFlow's mesh simplification, CPU). prepare_release.py checks the same file names.
TORCHSPARSE_WHEEL = paths.WHL / "torchsparse-2.1.0+cu128torch2.8-cp311-cp311-win_amd64.whl"
PYFQMR_WHEEL = paths.WHL / "pyfqmr_triflow-0.5.0+triflow-cp311-cp311-win_amd64.whl"
# xatlas (the UV unwrap of the retopology stage, in Instant Meshes' venv): Spellcast3D's own build of xatlas-python
# 0.0.11, its vendored xatlas with the chart-merging pass fixed (it scanned every chart for every chart it tested and
# started over after each merge: 164 s for a 22,800-triangle tree, 1.1 s now, the same atlas bit for bit). Built by
# code/release/wheels/xatlas/build_xatlas_win.py from this project's forks; the version tag keeps it apart from PyPI's.
XATLAS_WHEEL = paths.WHL / "xatlas-0.0.11+spellcast3d.1-cp311-cp311-win_amd64.whl"
# About what share_identical_files leaves out of a new retopology venv when another venv has the same torch
# (measured: the forge and pixal3d venvs have 7.4 GB of identical big files).
SHARED_TORCH_GB = 7.0

# Free disk (GB) wanted to update or repair a venv that already exists.
UPDATE_GB = 3.0

LOG_HINT = r"Details are in code\logs\install.log."


def _download_failed(title: str) -> str:
    return (f"Downloading or installing the packages of {title} failed. Check your internet connection and "
            f"that the drive has enough free space, then start Spellcast3D again (finished downloads are "
            f"kept). {LOG_HINT}")


def _broken(title: str) -> str:
    return (f"{title} was installed but does not start correctly. Start Spellcast3D again to retry; "
            f"if it keeps failing, see code\\logs\\install.log.")


def git_exe() -> str | None:
    return str(paths.PORTABLE_GIT) if paths.PORTABLE_GIT.exists() else shutil.which("git")


def _pixal3d_models_bytes() -> int:
    """Bytes of Pixal3D's models on disk, for its download progress: without its optional multi-view models
    (the hub downloads those on request, multiview.py) and without the copies of its flow models (made after the
    download), both of which PIXAL3D_MODELS_BYTES leaves out."""
    total = models.folder_bytes(PIXAL3D_MODELS)
    snapshot = models.hf_snapshot(PIXAL3D_MODELS / "hub", PIXAL3D_HF_REPO)
    if snapshot is not None:
        for rel in models.PIXAL3D_MULTI_VIEW_FILES:
            try:
                total -= os.stat(snapshot / rel).st_size    # a snapshot entry links to its blob: the blob's
            except OSError:
                pass
        total -= _pixal3d_copies_bytes(snapshot)
    return max(0, total)


def _pixal3d_models_env() -> dict[str, str]:
    """Where Pixal3D's child processes find its models: through the junction into models/pixal3d."""
    return {"HF_HOME": str(PIXAL3D_MODELS), "TORCH_HOME": str(PIXAL3D_MODELS / "torch")}


def _pixal3d_copies_bytes(snapshot: Path) -> int:
    """Bytes of the copies of Pixal3D's checkpoints in a snapshot folder, unfinished ones included."""
    total = 0
    for path in snapshot.glob(f"ckpts/*{models.PIXAL3D_COPY_SUFFIX}*"):
        if path.name.endswith((models.PIXAL3D_COPY_SUFFIX, f"{models.PIXAL3D_COPY_SUFFIX}.tmp")):
            try:
                total += os.lstat(path).st_size
            except OSError:
                pass
    return total


@functools.lru_cache(maxsize=None)
def _sha256_of(path: str, size: int, mtime_ns: int) -> str:
    return state.file_sha256(Path(path))


def wheel_sha256(path: Path) -> str:
    """state.file_sha256 of a wheel, hashed once per process (flash_attn's is 250 MB, and every start
    computes the fingerprints)."""
    try:
        st = os.stat(path)
    except OSError:
        return "missing"
    return _sha256_of(str(path), st.st_size, st.st_mtime_ns)


def _root_rel(path: Path) -> str:
    try:
        return str(path.relative_to(paths.ROOT))
    except ValueError:
        return str(path)


def checkout_submodule(ctx: Ctx, submodule: Path, marker: str, title: str, git_config: tuple[str, ...] = ()) -> None:
    """Check out the git submodule `submodule` when `marker` is not in it (git_config: `git -c` settings for it)."""
    if (submodule / marker).exists():
        return
    rel = submodule.relative_to(paths.ROOT)
    incomplete = (f"The folder {rel} is missing or empty, so this copy of Spellcast3D is incomplete. "
                  f"Please download Spellcast3D again.")
    git = git_exe()
    if git is None or not (paths.ROOT / ".git").exists():
        raise InstallError(incomplete)
    ctx.out.line(f"Downloading the source of {title} ({rel}) with git ...")
    ctx.step(f"Downloading the source of {title}", until=0.0)
    # Never --recursive: instant-meshes' nested nanogui/Eigen is not needed and breaks builds.
    # --depth 1: only the pinned commit, not the tool's whole history (60+ MB for Forge).
    config = [arg for setting in git_config for arg in ("-c", setting)]
    ctx.run([git, *config, "-C", str(paths.ROOT), "submodule", "update", "--init", "--depth", "1", "--",
             rel.as_posix()],
            cwd=paths.ROOT, env=ctx.env(None),
            fail=f"Downloading {rel} with git failed. Check your internet connection and start again.")
    if not ctx.dry_run and not (submodule / marker).exists():
        raise InstallError(incomplete)


def share_torch_files(ctx: Ctx, venv_id: str) -> None:
    """Hard-link the venv's big files that another venv with the same torch has byte for byte (share_identical_files:
    ~7 GB for torch alone). Forge's and Pixal3D's end with it as the retopology tools' do."""
    twins = venvs.torch_twins(venv_id)
    if ctx.dry_run or not twins:
        return
    count, saved = venvs.share_identical_files(paths.venv_dir(venv_id), twins)
    if count:
        names = ", ".join(f"code\\venvs\\{t.name}" for t in twins)
        ctx.out.line(f"{count} big file(s) ({saved / 1e9:.1f} GB) are the same as in {names}: they are hard links to "
                     f"those now, which takes no extra disk space.")


def dir_size_gb(folder: Path) -> float:
    total = 0
    for dirpath, _dirs, files in os.walk(folder):
        for name in files:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            total += st.st_size
    return total / 1e9


class Component(abc.ABC):
    id: str
    title: str
    about: str                      # what it is, how much it downloads, how long it takes
    size_gb: float                  # free disk needed for a fresh install
    submodule: Path | None = None   # git submodule it needs, and a file that proves it is checked out
    marker: str = ""
    uses_gpu: bool = False
    needs_torch: bool = False       # installs PyTorch (from download.pytorch.org, through pip's shared cache)
    models_gb: float = 0.0          # how much of size_gb its AI models are (see models_present_gb)
    # How much of a fresh install's disk it gives back by sharing another venv's identical files (the same torch)
    # through hard links (venvs.share_identical_files): what preflight need not keep free for the next components.
    torch_shared_gb: float = 0.0
    # Bump (e.g. "2") when ONLY this component's install() steps change: only it is installed again.
    recipe: str = ""

    @abc.abstractmethod
    def inputs(self) -> dict[str, str]:
        """Everything (besides the shared parts) whose change means: install again."""

    @abc.abstractmethod
    def install(self, ctx: Ctx) -> None:
        """Install into the (existing) venv and verify. Raise InstallError with an artist-facing reason."""

    def supported(self, gpu_info: gpu.GpuInfo | None) -> tuple[bool, str]:
        """(False, reason) when this PC can never run it (recorded as 'unsupported', not retried as a failure)."""
        return True, ""

    def missing_files(self) -> list[str]:
        """Short names of files it needs besides its venv (e.g. AI models) that are not on disk."""
        return []

    def repair_work(self) -> str | None:
        """Why an installed, unchanged component still needs repair() (e.g. AI models are missing), or None.

        Checked at every start even when the venv is up to date. Must be fast (a few stat calls).
        """
        missing = self.missing_files()
        return f"some of its downloaded files are missing: {', '.join(missing)}" if missing else None

    def repair(self, ctx: Ctx) -> None:
        """Do what repair_work() reports, for an installed and unchanged venv."""
        self.install(ctx)

    def models_present_gb(self) -> float:
        """How much of its AI models is on disk already (that part needs no free space)."""
        return 0.0

    def download_hosts(self) -> set[str]:
        """The servers (besides pypi.org and download.pytorch.org) it must download from now: preflight checks that
        they answer. Must be fast (a few stat calls)."""
        return set()

    def fingerprint(self) -> str:
        return state.fingerprint({
            "component": self.id,
            "installer": INSTALLER_VERSION,
            "python": f"{sys.version_info[0]}.{sys.version_info[1]}",
            "pip": venvs.PIP_VERSION,
            # only when set, so that adding the key did not change any existing install's fingerprint
            **({"recipe": self.recipe} if self.recipe else {}),
            **self.inputs(),
        })

    def disk_needed_gb(self, fresh: bool = False) -> float:
        """fresh: the venv is deleted and made again (--reinstall): its files that other venvs share stay on disk
        through their links, so it needs as much as a first install."""
        update = min(UPDATE_GB, self.size_gb)
        if fresh or not paths.venv_python(self.id).exists():
            need = self.size_gb - self.models_gb
        elif (state.load().get(self.id) or {}).get("status") == state.INSTALLED:
            need = update
        else:  # an earlier attempt stopped part-way: the rest of a fresh install
            need = max(update, self.size_gb - self.models_gb - dir_size_gb(paths.venv_dir(self.id)))
        return need + max(0.0, self.models_gb - self.models_present_gb())

    def disk_kept_gb(self, torch_twin: bool, fresh: bool = False) -> float:
        """Of disk_needed_gb(), what stays used once it is installed: a fresh venv that shares its identical files
        with another venv's (torch_twin: one has, or gets in this run, the same torch) gives torch_shared_gb back."""
        need = self.disk_needed_gb(fresh)
        if torch_twin and self.torch_shared_gb and (fresh or not paths.venv_python(self.id).exists()):
            return max(0.0, need - self.torch_shared_gb)
        return need

    def prepare(self, ctx: Ctx) -> None:
        """Make sure the files it installs from are here (checking out its git submodule if needed)."""
        if self.submodule is not None:
            checkout_submodule(ctx, self.submodule, self.marker, self.title)

    def _python(self) -> str:
        return str(paths.venv_python(self.id))

    def _verify(self, ctx: Ctx, code: str) -> None:
        ctx.run([self._python(), "-c", code], cwd=paths.venv_dir(self.id), env=ctx.env(self.id),
                fail=_broken(self.title))


class Hub(Component):
    id = paths.HUB
    title = "Spellcast3D hub"
    about = "The web page you work in. Downloads about 30 MB; takes about a minute."
    size_gb = 0.3

    def inputs(self) -> dict[str, str]:
        return {"requirements": state.file_sha256(HUB_REQUIREMENTS)}

    def install(self, ctx: Ctx) -> None:
        ctx.step("Installing its packages", until=0.9)
        ctx.pip(self.id, "-r", str(HUB_REQUIREMENTS), fail=_download_failed(self.title))
        ctx.step("Checking the installation", until=1.0)
        self._verify(ctx, "import fastapi, uvicorn, httpx, python_multipart\n"
                          "print('hub packages OK: fastapi', fastapi.__version__, '| uvicorn', uvicorn.__version__)")


class Imesh(Component):
    id = paths.IMESH
    title = "Instant Meshes"
    about = ("Retopology, UV unwrapping and texture baking (runs on the CPU). Downloads about 75 MB; "
             "takes about a minute.")
    size_gb = 0.5
    submodule = paths.IMESH_DIR
    marker = "pyproject.toml"

    @staticmethod
    def wheel() -> Path | None:
        """The newest prebuilt instant_meshes_brush wheel in code/whl."""
        def key(p: Path) -> tuple[tuple[int, ...], float]:
            version = p.name.split("-")[1]
            return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.+]", version)), p.stat().st_mtime
        return max(paths.WHL.glob(IMESH_WHEEL_GLOB), key=key, default=None)

    def inputs(self) -> dict[str, str]:
        wheel = self.wheel()
        return {"wheel": state.file_sha256(wheel) if wheel else "missing",
                "requirements": state.file_sha256(IMESH_REQUIREMENTS),
                "xatlas": state.file_sha256(XATLAS_WHEEL) if XATLAS_WHEEL.is_file() else "missing"}

    def prepare(self, ctx: Ctx) -> None:
        super().prepare(ctx)
        if self.wheel() is None:
            raise InstallError(
                "The Instant Meshes package (code\\whl\\instant_meshes_brush-*.whl) is missing, so this copy "
                "of Spellcast3D is incomplete. Please download Spellcast3D again. "
                "(Developers: python prepare_release.py build-wheel)")
        if not XATLAS_WHEEL.is_file():
            raise InstallError(
                f"code\\whl\\{XATLAS_WHEEL.name} is missing, so this copy of Spellcast3D is incomplete. Please "
                "download Spellcast3D again. (Developers: code\\release\\wheels\\xatlas\\build_xatlas_win.py)")

    def install(self, ctx: Ctx) -> None:
        ctx.step("Installing its packages", until=0.9)
        ctx.pip(self.id, "-r", str(IMESH_REQUIREMENTS), fail=_download_failed(self.title))
        # Our wheels last (xatlas: Spellcast3D's build, see XATLAS_WHEEL; the Instant Meshes package). A rebuilt
        # wheel keeps its version number, so pip must be told to replace it.
        ctx.pip(self.id, "--force-reinstall", "--no-deps", str(XATLAS_WHEEL), str(self.wheel()),
                fail=_download_failed(self.title))
        ctx.step("Checking the installation", until=1.0)
        self._verify(ctx, "import instant_meshes_brush, instant_meshes_brush.server as server\n"
                          "from instant_meshes_brush import uv\n"
                          "if not uv.available(): raise SystemExit('UV unwrapping (xatlas) is not available')\n"
                          "from embreex import rtcore_scene, mesh_construction  # the Projection stage's rays\n"
                          "server.build_app()\n"
                          "print('Instant Meshes OK:', instant_meshes_brush.__version__)")


class Forge(Component):
    id = paths.FORGE
    title = "Forge Neo"
    about = ("Image generation. Downloads about 5 GB (PyTorch alone is 3.5 GB) and uses about 12 GB of disk; "
             "takes 10-30 minutes depending on your internet speed.")
    size_gb = 12.0
    submodule = paths.FORGE_DIR
    marker = "launch.py"
    uses_gpu = True
    needs_torch = True

    def inputs(self) -> dict[str, str]:
        return {"source": state.source_version(paths.FORGE_DIR, ("requirements.txt", "modules/launch_utils.py")),
                "torch": " ".join(TORCH), "gradio": FORGE_GRADIO}

    def supported(self, gpu_info: gpu.GpuInfo | None) -> tuple[bool, str]:
        # No NVIDIA GPU is final for this PC (not a failure to retry with a pause at every start).
        # It is re-checked at each start, so installing a card or its driver later installs Forge.
        # A too-old driver is NOT this case: preflight blocks it as a fixable failure.
        if gpu_info is None:
            return False, ("No NVIDIA graphics card was found (or its driver is not installed). Forge Neo needs "
                           "one to make images; you can still bring your own images. If this PC has an NVIDIA "
                           "card, install its latest driver from https://www.nvidia.com/drivers and start "
                           "Spellcast3D again: Forge Neo is then installed by itself.")
        return True, ""

    def install(self, ctx: Ctx) -> None:
        ctx.step("Downloading PyTorch (3.5 GB)", until=0.6)
        ctx.pip(self.id, *TORCH, "--index-url", TORCH_INDEX, fail=_download_failed(self.title))
        self._seed_config(ctx)
        # Forge's own installer: requirements, gradio, built-in extensions. --exit stops it before
        # the web UI starts; SD_WEBUI_RESTARTING keeps it from opening a browser.
        ctx.step("Installing Forge Neo's packages", until=0.97)
        ctx.run([self._python(), "launch.py", "--exit", "--skip-python-version-check", "--skip-version-check"],
                cwd=paths.FORGE_DIR,
                env=ctx.env(self.id, {"SD_WEBUI_RESTARTING": "1", "WEBUI_LAUNCH_LIVE_OUTPUT": "1"}),
                fail=f"Forge Neo's own setup step failed. Check your internet connection and start Spellcast3D "
                     f"again. {LOG_HINT}")
        ctx.step("Sharing identical files with the other tools", until=0.99)
        share_torch_files(ctx, self.id)
        ctx.step("Checking the installation", until=1.0)
        self._verify(ctx, "import torch, gradio\n"
                          "print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(),"
                          " '| gradio', gradio.__version__)\n"
                          "if not torch.cuda.is_available(): raise SystemExit('PyTorch cannot use the NVIDIA GPU')\n"
                          f"if gradio.__version__ != '{FORGE_GRADIO}':"
                          f" raise SystemExit('expected gradio {FORGE_GRADIO}, found ' + gradio.__version__)")

    def _seed_config(self, ctx: Ctx) -> None:
        # Without Forge's VERSION_UID in config.json, launch.py blocks on input() (see forge_config.py).
        if ctx.dry_run:
            ctx.out.line(f"  [dry-run] set VERSION_UID={forge_version_uid()} in {forge_config.CONFIG}")
            return
        note = forge_config.ensure_forge_config()
        if note:
            ctx.out.line(note)


# Pixal3D measured once on this PC's graphics card, as the last step of its install (the fork's pixal3d/calibrate.py:
# a quiet card first, its GPU kernels compiled, the faster of SageAttention's two kernels, the 3D decoders' convolution
# tiles for every prop size), so that nothing is timed or tuned while the artist works. Its record lies beside the fork
# (code/pixal3d/calibration.json, belonging to the install). Raise PIXAL3D_CALIBRATION when a change needs every install
# to measure again (a new wheel with new kernels, a new set of keys); another card is measured again by itself, and a
# new Triton re-times the int8 tile by itself (pixal3d_calibration_work).
PIXAL3D_CALIBRATION = 3        # 3: spconv's kernel choices for Hi3D's convolutions recorded too (once its models are here)
PIXAL3D_CALIBRATION_MODULE = "pixal3d.calibrate"
PIXAL3D_CALIBRATION_FAILED = ("Warning: Pixal3D could not be calibrated for this graphics card (see the lines above). It "
                              "works all the same: it then tunes itself during its first 3D jobs. To measure the card "
                              "again, close heavy programs and press \"Measure the graphics card again\" in Settings > "
                              "Debugging.")
# The same below sm_80, said from what the record lacks (_calibration_failed_words): there the decode-in-boxes check
# and the attention trial have no stand-in at first use.
PIXAL3D_CALIBRATION_FAILED_OLD_CARD = ("Warning: Pixal3D could not finish measuring this graphics card (see the lines "
                                       "above).")
OLD_CARD_LOSSES = {"decode_products": "High (1536) 3D models may be made at Standard (1024) detail",
                   "sage": "3D models take longer"}
LOWER_DETAIL_BELOW_MB = 16 * 1024   # a card this large makes 1536 props without the decode in boxes too
MEASURED_AT_NEXT_START = ("Spellcast3D measures it again at its next start, or close heavy programs and press "
                          "\"Measure the graphics card again\" in Settings > Debugging.")
MEASURED_ON_REQUEST = ("To measure it again, close heavy programs and press \"Measure the graphics card again\" in "
                       "Settings > Debugging.")
# A step of a card below sm_80 cut short by a passing cause (another program's video memory, the PC's own memory or
# page file run short, a trial process that crashed or ran out of time) runs again alone at the next start, behind the
# same notice, at most CALIBRATION_RETRIES times for this card and calibration version. The installer counts them
# itself in install-state.json (calibrate.py's "attempts" restarts with every complete record), before each run, and
# forgets them at a whole calibration (a new record: Settings' "Measure the graphics card again", another card, a new
# version). A measured answer stands, and so does every step of an sm_80+ card (each tunes itself at its first use).
CALIBRATION_RETRIES = 2
CALIBRATION_RETRIES_KEY = "pixal3d-calibration-retries"
CALIBRATION_RETRY_WORK = {
    "decode_products": "the 3D decoders' GEMM calls are checked again (the last check did not finish)",
    "sage": "its attention kernel is tried again (the last trial did not finish)",
    "spconv_decoders": "the 3D decoders' convolution kernels are chosen again (the last choice did not finish)"}
_CUT_SHORT = re.compile(r"out[ _]of[ _]memory|outofmemory|alloc_failed|not enough memory|memoryerror|bad[ _]alloc|"
                        r"paging file|did not finish|ended with code", re.IGNORECASE)
_MEASURED = re.compile(r"differ|inexact|not exact|no kernel image|has no ", re.IGNORECASE)
# Why Pixal3D is calibrated once more when only Hi3D's step is missing (pixal3d_calibration_work, _calibrate --only):
# Hi3D's AI models came by their own download (model_downloads.py), whose own calibration run did not finish.
HI3D_CALIBRATION_WORK = "Hi3D's convolutions are measured once for this PC's graphics card"
# Why the calibration's Triton step runs alone (calibrate --only triton): a record made before that step chose the int8
# GEMMs' fc1 tile for the card (the record's triton.int8_tiles, read by the fork's int8mm.fc1_group_cfg; on sm_80+
# cards, where those GEMMs are Triton's), or chose it with another Triton than the venv's (the fork reads such a choice
# as none), or holds no race of the group GEMMs' CUDA kernel against Triton's (int8_tiles.group_cuda, made with the
# venv's pixal3d_kernels build where that build has the kernel: the fork's int8mm.group_cuda_stands). The step
# compiles its kernels again into Triton's cache (cached ones are found) and times the tiles once; the record's other
# sections stand.
INT8_TILES_CALIBRATION_WORK = "its int8 kernels' tile is chosen once for this PC's graphics card"
INT8_TILES_FROM = 8.0           # the cards whose int8 GEMMs are Triton's (sm_80+): those the tile choice serves
KERNELS_RACED_FROM = (0, 2, 0)  # the first pixal3d_kernels with the sm_80 group GEMM and the old cards' element-wise
#                                 kernels, whose races the record holds
# Why the calibration's step for the 3D decoders' spconv kernels runs alone (calibrate --only spconv_decoders): a
# record made before that step, on a card whose decoders run spconv's own GEMM kernels (Turing and Volta: spconv's
# wheel has none for Pascal, whose decoders take torch products; Ampere and newer run flex_gemm's), unless the record
# found flex_gemm serving there (its "conv_backend"). Without the choices spconv times its kernels again at the first
# prop of every process, and may choose others from one process to the next.
SPCONV_DECODERS_CALIBRATION_WORK = "the 3D decoders' convolution kernels are chosen once for this PC's graphics card"
SPCONV_DECODERS_CARDS = (7.0, 8.0)      # compute capabilities [from, below) whose decoders run spconv's GEMM kernels
# Why the calibration's race of the old cards' element-wise kernels runs alone (calibrate --only old_cards): a record
# below sm_80 without it, or with one of another pixal3d_kernels build than the venv's (the fork's modules/old_cards.py
# reads such a race as none and runs every kernel). Each kind serves only where it measured faster than the eager
# chain it stands for, on the card itself; the numbers are the same either way.
OLD_CARDS_CALIBRATION_WORK = "its element-wise kernels are timed once for this PC's graphics card"
OLD_CARDS_BELOW = 8.0
# Why the calibration's attention step runs alone (calibrate --only sage): a Turing record whose SageAttention kernel
# ran at the flows' sizes (the "sage" section's candidate "turing", candidate_ok) but holds no race of its int8 P.V
# twin ("int8pv", which the fork's trial always records since: sage.int8pv_race_lacking is the same rule). The twin
# serves only where the fork's switch asks for it and it timed 5 % faster; without the race it would be timed in the
# engine's first process.
TURING_INT8PV_CALIBRATION_WORK = "its int8 attention kernel is timed once for this PC's graphics card"
# Why the attention step runs alone when the pinned SageAttention wheel is not the record's (calibrate --only sage):
# the attention timings belong to the wheel they were made with (the fork reads a section of another wheel as none:
# sage._stored), while the other sections stand. A new wheel's new kernels are raced there - Pascal's twins of its
# attention kernel from post8 on (the fork's sage.PASCAL_TWINS) -, and the old cards' step's own rule above re-times
# pixal3d_kernels' twins of the dp4a GEMM with a new kernels build (int8mm.DP4A_TWINS, the old_cards section).
SAGE_WHEEL_CALIBRATION_WORK = "its attention kernels are timed once more (a new build of them)"
# Why the calibration's step for the decode in boxes runs alone (calibrate --only decode_products): a record of a card
# whose decoders run on spconv (below sm_80: Pascal's torch products, Turing's and Volta's spconv GEMMs; unless the
# record found flex_gemm serving there, its "conv_backend") made before that step (Spellcast3D round O). The step
# measures the smallest cuBLAS call from which a decoder row keeps its bits whatever the call's size (the fork's
# modules/sparse/conv/products_floor.py), which the decode in boxes pads its calls to. Without a section that passed
# the fork decodes no boxes on spconv (products_floor.exact) and a card under 10 GiB keeps its 1536 props at 1024,
# with a one-time warning. The step is due once, where the section is MISSING; one cut short by a passing cause (no
# room on the card, say) runs again at the next start, CALIBRATION_RETRIES times at most (CALIBRATION_RETRY_WORK); one
# that measured stands; Settings > Debugging > "Measure the graphics card again" measures again on request.
DECODE_PRODUCTS_CALIBRATION_WORK = "the 3D decoders' GEMM calls are checked once for this PC's graphics card"
DECODE_PRODUCTS_BELOW = 8.0     # the cards whose decoders run on spconv
# Why that step runs alone once more (Spellcast3D round P's review): a section that measured the card's shapes in an
# older form than the fork's (products_floor.FORM; 3 checks the launch boundary at 2^20 rows too - the real GTX 10xx
# fail there, unmeasured for the 128 -> 512 shapes before; 4 checks the whole calls past the ladder as far as the
# card holds them, so that a card with more memory decodes bulkier props in boxes). The fork reads an older section all the same (from its
# shapes); the hub drops it from the record first (_forget_decode_products: the fork's calibrate keeps a section that
# passed), so `--only decode_products` measures it. Once: the new section is of the form (a failed one: the rule above).
DECODE_PRODUCTS_AGAIN_WORK = ("the 3D decoders' GEMM calls are checked once more for this PC's graphics card (at one "
                              "more size)")
DECODE_PRODUCTS_FORM = 4
# The steps a standing record may lack, each run alone (several: in one run, calibrate --only a,b), by the reason
# pixal3d_calibration_work gives (several joined by "; ").
CALIBRATION_STEPS_ALONE = {INT8_TILES_CALIBRATION_WORK: "triton", HI3D_CALIBRATION_WORK: "spconv",
                           SPCONV_DECODERS_CALIBRATION_WORK: "spconv_decoders",
                           OLD_CARDS_CALIBRATION_WORK: "old_cards", TURING_INT8PV_CALIBRATION_WORK: "sage",
                           SAGE_WHEEL_CALIBRATION_WORK: "sage", DECODE_PRODUCTS_CALIBRATION_WORK: "decode_products",
                           DECODE_PRODUCTS_AGAIN_WORK: "decode_products",
                           **{why: step for step, why in CALIBRATION_RETRY_WORK.items()}}

# The calibration's own lines ("[CALIBRATE]  42% convolutions 1024x4096: ...", "[CALIBRATE] the graphics card is busy
# (46 % in use): waiting ..."): what it does now and, with a percentage, how far it is, for the page (the hub shows
# the step text and the bar: static/js/shell/install.js, calibration.js).
_CALIBRATION_LINE = re.compile(r"^\[CALIBRATE\]\s+(?:(\d{1,3})%\s+)?(.*)$")


def _calibration_line(ctx: Ctx, text: str) -> None:
    m = _CALIBRATION_LINE.match(text.strip())
    if m is None:
        return
    percent, what = m.group(1), m.group(2).strip()
    ctx.progress.detail(what, int(percent) / 100.0 if percent is not None else None)


@functools.cache
def _card() -> gpu.GpuInfo | None:
    return gpu.primary_gpu()


@functools.cache
def _cards() -> tuple[gpu.GpuInfo, ...]:
    return tuple(gpu.query_gpus())


def pixal3d_calibration_file() -> Path:
    return paths.PIXAL3D_DIR / "calibration.json"


def _uuid_key(text: object) -> str:
    """A card's UUID as a comparable key: nvidia-smi says GPU-xxxxxxxx-..., torch may say it without the prefix."""
    key = str(text or "").strip().lower()
    return key[4:] if key.startswith("gpu-") else key


def pixal3d_calibration_work(gpu_info: gpu.GpuInfo | None = None,
                             cards: tuple[gpu.GpuInfo, ...] | None = None) -> str | None:
    """Why Pixal3D is still to be calibrated on this PC's graphics card, or None: never on a card it does not support,
    nor with a checkout of the fork that has no calibration; a record that is missing, of an older version, or made
    on a card this PC no longer has (by UUID; by name where the record has none - any of this PC's cards, since
    torch and nvidia-smi may order them differently). A record that is there stands, complete or not, except the
    steps of a card below sm_80 cut short by a passing cause (CALIBRATION_RETRY_WORK: at the next start, behind the
    same notice, CALIBRATION_RETRIES times at most); Settings > Debugging measures again on request, through
    `drop_pixal3d_calibration`. Fast: one small file read (nvidia-smi once a process)."""
    info = gpu_info if gpu_info is not None else _card()
    if info is None or not gpu.pixal3d_supported(info)[0]:
        return None
    if (paths.PIXAL3D_DIR / "install.py").is_file() and not (paths.PIXAL3D_DIR / "pixal3d" / "calibrate.py").is_file():
        return None     # checked out at a version without it (not yet checked out: it comes with it)
    try:
        record = json.loads(pixal3d_calibration_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "it is calibrated once for this PC's graphics card"
    if not isinstance(record, dict) or record.get("version") != PIXAL3D_CALIBRATION:
        return "it is calibrated once more for this PC's graphics card (a new version of the calibration)"
    here = cards if cards is not None else (_cards() or (info,))
    uuids = {_uuid_key(c.uuid) for c in here if getattr(c, "uuid", "")}
    names = {c.name for c in here}
    recorded = _uuid_key(record.get("card_uuid"))
    if recorded and uuids:
        changed = recorded not in uuids
    else:
        changed = record.get("card") not in names
    if changed:
        return f"the graphics card changed ({record.get('card')} -> {info.name}): it is calibrated for the new one"
    pinned = pixal3d_sage_wheel_version()
    # the attention timings belong to the wheel they were made with: that step alone, the rest of the record stands
    wheel = bool(pinned) and (record.get("sage") or {}).get("wheel") not in (None, pinned)
    # Hi3D's step is left out while its models are not here (its download runs it: model_downloads.hi3d_calibration);
    # a step that ran and failed stands, as every other
    hi3d = models.hi3d_on_disk() and (record.get("spconv") is None or record["spconv"].get("skipped"))
    triton = record.get("triton") if isinstance(record.get("triton"), dict) else {}
    tiles = (info.compute_cap >= INT8_TILES_FROM and triton.get("int8") is True
             and ("int8_tiles" not in triton or _tiles_of_another_triton(triton["int8_tiles"])
                  or _group_race_lacking(triton["int8_tiles"])))
    conv = record.get("conv_backend") if isinstance(record.get("conv_backend"), dict) else {}
    flex = conv.get("ok", True) and conv.get("backend") == "flex_gemm"     # the record found flex_gemm serving
    decoders = (SPCONV_DECODERS_CARDS[0] <= info.compute_cap < SPCONV_DECODERS_CARDS[1]
                and "spconv_decoders" not in record and not flex)
    products = info.compute_cap < DECODE_PRODUCTS_BELOW and "decode_products" not in record and not flex
    again = (info.compute_cap < DECODE_PRODUCTS_BELOW and not flex
             and _decode_products_of_an_older_form(record.get("decode_products")))
    old = info.compute_cap < OLD_CARDS_BELOW and _old_cards_race_lacking(record.get("old_cards"))
    attention = record.get("sage") if isinstance(record.get("sage"), dict) else {}
    twin = (attention.get("candidate") == "turing" and bool(attention.get("candidate_ok"))
            and "int8pv" not in attention and not wheel)
    due = [why for why, lacking in ((INT8_TILES_CALIBRATION_WORK, tiles), (HI3D_CALIBRATION_WORK, hi3d),
                                    (SPCONV_DECODERS_CALIBRATION_WORK, decoders), (OLD_CARDS_CALIBRATION_WORK, old),
                                    (TURING_INT8PV_CALIBRATION_WORK, twin), (SAGE_WHEEL_CALIBRATION_WORK, wheel),
                                    (DECODE_PRODUCTS_CALIBRATION_WORK, products), (DECODE_PRODUCTS_AGAIN_WORK, again))
           if lacking]
    taken = {CALIBRATION_STEPS_ALONE[why] for why in due}
    due += [why for why in _calibration_retries(record, info) if CALIBRATION_STEPS_ALONE[why] not in taken]
    return "; ".join(due) or None


def _calibration_retries(record: dict, info: gpu.GpuInfo) -> list[str]:
    """The CALIBRATION_RETRY_WORK reasons due below sm_80: steps cut short with retries left (_retries_made)."""
    if info.compute_cap >= OLD_CARDS_BELOW:
        return []
    made = _retries_made(record)
    return [why for step, why in CALIBRATION_RETRY_WORK.items()
            if _cut_short(step, record.get(step)) and made.get(step, 0) < CALIBRATION_RETRIES]


def _passed(step: str, section: object) -> bool:
    """A section that passed: ok, and for the attention trial of a card below sm_80 (run in a child process, whose
    failure leaves the section "ok" with candidate_ok False and the trial's why) its kernel ran."""
    return (isinstance(section, dict) and section.get("ok") is True
            and (step != "sage" or not section.get("candidate") or bool(section.get("candidate_ok"))))


def _cut_short(step: str, section: object) -> bool:
    """A section that a passing cause ended before it measured (_CUT_SHORT, not a measured answer: _MEASURED)."""
    if not isinstance(section, dict) or _passed(step, section):
        return False
    text = str(section.get("error") or section.get("why") or "")
    return bool(_CUT_SHORT.search(text)) and not _MEASURED.search(text)


def _record_card(record: dict) -> str:
    return _uuid_key(record.get("card_uuid")) or str(record.get("card") or "")


def _retries_made(record: dict) -> dict[str, int]:
    """How often each step was run again for this record's card and calibration version (CALIBRATION_RETRIES_KEY)."""
    entry = state.load().get(CALIBRATION_RETRIES_KEY)
    if (not isinstance(entry, dict) or entry.get("card") != _record_card(record)
            or entry.get("version") != record.get("version") or not isinstance(entry.get("steps"), dict)):
        return {}
    return {step: n for step, n in entry["steps"].items() if isinstance(n, int)}


def _note_retries(steps: list[str]) -> None:
    """Count a run of these steps again, before it starts: one that is cut short once more counts as well."""
    try:
        record = json.loads(pixal3d_calibration_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(record, dict):
        return
    made = _retries_made(record)
    for step in steps:
        made[step] = made.get(step, 0) + 1
    state.note(CALIBRATION_RETRIES_KEY, {"card": _record_card(record), "version": record.get("version"), "steps": made})


def _forget_retries() -> None:
    if CALIBRATION_RETRIES_KEY in state.load():
        state.note(CALIBRATION_RETRIES_KEY, None)


def pixal3d_step_due(step: str, info: gpu.GpuInfo | None = None) -> bool:
    """Whether the next start runs this calibration step (pixal3d_calibration_work), alone or in a whole one."""
    why = pixal3d_calibration_work(info)
    if why is None:
        return False
    parts = why.split("; ")
    return not all(part in CALIBRATION_STEPS_ALONE for part in parts) or any(
        CALIBRATION_STEPS_ALONE[part] == step for part in parts)


def _calibration_failed_words(retried: list[str] | tuple[str, ...] = ()) -> str:
    """The install's warning after a calibration run that did not finish. Below sm_80: what the card does without the
    steps the record lacks (OLD_CARD_LOSSES), and the next start's measurement promised only where it runs them;
    nothing where the steps run again (`retried`) all measured now - calibrate.py ends a run of some steps of a record
    with a reservation with its exit code 1 even then (the earlier run's reservation stands)."""
    info = _card()
    if info is None or info.compute_cap >= OLD_CARDS_BELOW:
        return PIXAL3D_CALIBRATION_FAILED
    try:
        record = json.loads(pixal3d_calibration_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = {}
    record = record if isinstance(record, dict) else {}
    lost = [step for step in OLD_CARD_LOSSES if not _passed(step, record.get(step))
            and (step != "decode_products" or info.vram_total_mb < LOWER_DETAIL_BELOW_MB)]
    again = (all(pixal3d_step_due(step, info) for step in lost) if lost
             else pixal3d_calibration_work(info) is not None)
    if not lost and not again:
        if retried and all(_passed(step, record.get(step)) for step in retried):
            return ""
        return PIXAL3D_CALIBRATION_FAILED
    until = f" Until it does, {' and '.join(OLD_CARD_LOSSES[step] for step in lost)}." if lost else ""
    return f"{PIXAL3D_CALIBRATION_FAILED_OLD_CARD}{until} {MEASURED_AT_NEXT_START if again else MEASURED_ON_REQUEST}"


def _decode_products_of_an_older_form(section: object) -> bool:
    """A decode_products section that passed and measured shapes (not one skipped on flex_gemm's decoders, not a
    failed one: those stand) in a form older than DECODE_PRODUCTS_FORM."""
    if not isinstance(section, dict) or section.get("ok") is not True or not section.get("shapes"):
        return False
    form = section.get("form")
    return not isinstance(form, int) or form < DECODE_PRODUCTS_FORM


def _forget_decode_products() -> None:
    """Drop the decode_products section from the record (DECODE_PRODUCTS_AGAIN_WORK), the rest kept as it is."""
    path = pixal3d_calibration_file()
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(record, dict) and record.pop("decode_products", None) is not None:
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")    # as the fork writes it


_TRITON_VERSION = re.compile(r"""^__version__\s*=\s*['"]([^'"]+)['"]""", re.MULTILINE)


def pixal3d_triton_version() -> str:
    """triton.__version__ of Pixal3D's venv ("3.4.0"; not its wheel's "3.4.0.post21"), read from triton/__init__.py
    without importing it: what the fork's calibration records with its int8 tile. "" where it cannot be read."""
    venv = paths.venv_dir(paths.PIXAL3D)
    for site in [venvs.site_packages(venv), *sorted(venv.glob("lib/python3*/site-packages"))]:     # Windows, Linux
        try:
            found = _TRITON_VERSION.search((site / "triton" / "__init__.py").read_text(encoding="utf-8"))
        except OSError:
            continue
        if found:
            return found.group(1)
    return ""


def _tiles_of_another_triton(tiles: object) -> bool:
    """A tile choice that was timed (ok) with another Triton than the venv's, which int8mm.fc1_group_cfg ignores. A
    failed timing stands, and so does every choice where the venv's version cannot be read."""
    if not isinstance(tiles, dict) or tiles.get("ok") is not True:
        return False
    venv = pixal3d_triton_version()
    return bool(venv) and tiles.get("triton") != venv


_DIST_VERSION = re.compile(r"^Version:\s*(\d+(?:\.\d+)*)", re.MULTILINE)


def pixal3d_kernels_version() -> tuple[int, ...] | None:
    """The version of pixal3d_kernels in Pixal3D's venv ((0, 2, 0)), read from its dist-info's METADATA without
    importing it: what the fork's calibration records with its races of that build's kernels. None where it cannot be
    read."""
    venv = paths.venv_dir(paths.PIXAL3D)
    for site in [venvs.site_packages(venv), *sorted(venv.glob("lib/python3*/site-packages"))]:     # Windows, Linux
        for meta in sorted(site.glob("pixal3d_kernels-*.dist-info/METADATA")):
            try:
                found = _DIST_VERSION.search(meta.read_text(encoding="utf-8"))
            except OSError:
                continue
            if found:
                return tuple(int(part) for part in found.group(1).split("."))
    return None


def _kernels_text(version: tuple[int, ...]) -> str:
    return ".".join(map(str, version))


def _group_race_lacking(tiles: object) -> bool:
    """A tile choice (timed) without the race of the group GEMMs' CUDA kernel where the venv's pixal3d_kernels has
    that kernel, or with a race made with another build of it: the fork's int8mm reads neither. A race that could not
    run (no kernel then, a failure) stands, and so does every choice where the venv's build cannot be read."""
    if not isinstance(tiles, dict) or tiles.get("ok") is not True:
        return False
    kernels = pixal3d_kernels_version()
    if kernels is None or kernels < KERNELS_RACED_FROM:
        return False
    race = tiles.get("group_cuda")
    if not isinstance(race, dict):
        return True
    return race.get("ok") is True and race.get("kernels") != _kernels_text(kernels)


def _old_cards_race_lacking(section: object) -> bool:
    """No race of the old cards' element-wise kernels in the record, or one made with another pixal3d_kernels build
    than the venv's, where the venv's build has those kernels. A race that ran and failed stands, and so does every
    record where the venv's build cannot be read."""
    kernels = pixal3d_kernels_version()
    if kernels is None or kernels < KERNELS_RACED_FROM:
        return False
    if not isinstance(section, dict):
        return True
    return section.get("ok") is True and "on" in section and section.get("kernels") != _kernels_text(kernels)


def pixal3d_calibration_command(only: str = "", skip: str = "", check: bool = True) -> list[str] | None:
    """The command of Pixal3D's calibration (PIXAL3D_CALIBRATION_MODULE in its venv, run from the fork), or None where
    it cannot run (`check`): a checkout of the fork without it, or a card Pixal3D does not support. only / skip: steps
    (the module's --only / --skip; "spconv" is Hi3D's convolutions' step)."""
    if check:
        if not (paths.PIXAL3D_DIR / "pixal3d" / "calibrate.py").is_file():
            return None
        info = _card()
        if info is None or not gpu.pixal3d_supported(info)[0]:
            return None
    argv = [str(paths.venv_python(paths.PIXAL3D)), "-m", PIXAL3D_CALIBRATION_MODULE, "--version",
            str(PIXAL3D_CALIBRATION)]
    if only:
        argv += ["--only", only]
    if skip:
        argv += ["--skip", skip]
    return argv


def pixal3d_calibration_env() -> dict[str, str]:
    """The calibration's environment: Pixal3D's models (offline) and the card's profile (gpu.pixal3d_card_env)."""
    return {**_pixal3d_models_env(), **gpu.pixal3d_card_env(_card()), "HF_HUB_OFFLINE": "1"}


def pixal3d_sage_wheel_version() -> str:
    """The version of the SageAttention wheel the fork ships (whl/sageattention.json names the file:
    sageattention-<version>-cp39-abi3-win_amd64.whl), "" where there is none."""
    try:
        name = json.loads((PIXAL3D_WHL / "sageattention.json").read_text(encoding="utf-8")).get("wheel", "")
    except (OSError, ValueError):
        return ""
    parts = str(name).split("-")
    return parts[1] if len(parts) > 2 and parts[0] == "sageattention" else ""


def drop_pixal3d_calibration() -> bool:
    """Forget the record, so that the next installer run measures the card again (the hub's "Measure the graphics
    card again" in Settings > Debugging, which then starts the background installer; that whole calibration forgets
    the retries counted, CALIBRATION_RETRIES_KEY). -> whether there was one."""
    try:
        pixal3d_calibration_file().unlink()
        return True
    except FileNotFoundError:
        return False


class Pixal3D(Component):
    """Pixal3D, and with it Hi3D v1.0's code and packages (the 3D stage's second geometry, a prop's
    mesh.settings.geometry: Hi3D runs in Pixal3D's launcher and venv on Pixal3D's coarse shape, so it is no component
    of its own). Its packages (HI3D_PACKAGES) are part of Pixal3D's install and its code is the code/hi3d submodule,
    best effort: without them Pixal3D works on. Its AI models (11.4 GB) are a download of their own, which the 3D stage
    starts when a prop is set to Hi3D (spellcast/model_downloads.py "hi3d"): nobody sits through them at the install."""
    id = paths.PIXAL3D
    title = "Pixal3D"
    about = ("Image to 3D model (needs a GTX 1060 or newer; fastest on RTX 30xx and newer). Downloads about 33 GB, of "
             "which about 27 GB are AI models (not those already in the models folder), and uses about 55 GB of disk "
             "(14 GB of it faster-loading copies of four of the models). This can take an hour or more. Hi3D, the "
             "optional finer shape, is a separate 11.4 GB download the 3D stage offers.")
    # The faster-loading copies are in what a fresh install needs, but not in models_gb: they are no download, and
    # an update of an install without them needs no room for them (where there is none, they are not made).
    size_gb = 40.0 + models.PIXAL3D_COPIES_BYTES / 1e9 + HI3D_PACKAGES_GB
    models_gb = PIXAL3D_MODELS_BYTES / 1e9
    submodule = paths.PIXAL3D_DIR
    marker = "install.py"
    uses_gpu = True
    needs_torch = True

    def inputs(self) -> dict[str, str]:
        return {"source": pixal3d_install_inputs(),
                "torch": " ".join(TORCH + (TORCHAUDIO,)),
                "post": " ".join(PIXAL3D_GRADIO + (PIXAL3D_SPACES,)),
                "xet": HF_XET,
                "hi3d": " ".join(HI3D_PACKAGES)}

    def supported(self, gpu_info: gpu.GpuInfo | None) -> tuple[bool, str]:
        return gpu.pixal3d_supported(gpu_info)

    def missing_files(self) -> list[str]:
        """The AI models Pixal3D's install.py downloads and the app loads at start (offline)."""
        m = PIXAL3D_MODELS
        missing = [name for name, path in (
            ("DINOv3", m / "dinov3" / "model.safetensors"),
            ("RMBG-2.0", m / "RMBG-2.0" / "model.safetensors"),
            ("NAF", m / "torch" / "hub" / "valeoai_NAF_main" / "hubconf.py"),
            ("NAF", m / "torch" / "hub" / "checkpoints" / "naf_release.pth"),
        ) if not path.is_file()]
        for name, repo_id, files in _hf_repos():
            snapshot = models.hf_snapshot(m / "hub", repo_id)
            if snapshot is None or not all((snapshot / f).is_file() for f in files or ("model.pt",)):
                missing.append(name)
        return list(dict.fromkeys(missing))

    def repair_work(self) -> str | None:
        return models.pixal3d_link_work() or super().repair_work() or hi3d_work() or pixal3d_calibration_work()

    def models_present_gb(self) -> float:
        # models/pixal3d, and an older install's real code/pixal3d/MODELS until it is moved there (the multi-view
        # models and the copies there count for no more than Pixal3D's own)
        folders = {os.path.normcase(os.path.realpath(p)): p for p in (paths.MODELS_PIXAL3D, PIXAL3D_MODELS)
                   if p.is_dir()}
        own = sum(models.folder_bytes(p) for p in folders.values()) / 1e9
        return min(own, PIXAL3D_MODELS_BYTES / 1e9)

    def prepare(self, ctx: Ctx) -> None:
        super().prepare(ctx)
        self._checkout_hi3d(ctx)
        self._link_models(ctx)

    def repair(self, ctx: Ctx) -> None:
        self._checkout_hi3d(ctx)
        self._link_models(ctx)
        missing = [] if ctx.dry_run else self.missing_files()
        if missing or ctx.dry_run:
            ctx.out.line(f"Downloading the missing AI models: {', '.join(missing) or '-'}")
            self._link_user_hf_cache(ctx)
            self._download_models(ctx, "--models-only", until=0.9)
            self._make_copies(ctx, until=0.95)     # a model downloaded again makes its copy stale
        if ctx.dry_run or pixal3d_calibration_work():
            self._calibrate(ctx, until=1.0)

    def install(self, ctx: Ctx) -> None:
        ctx.out.line("If the download is interrupted or this window is closed, just start Spellcast3D again:")
        ctx.out.line("finished downloads are kept and the rest resumes.")
        # Pixal3D's installer runs pip with --no-cache-dir. Installing the same torch and the same pinned
        # packages first, through pip's shared cache, means its own pip calls find them already there:
        # torch (3.5 GB) is not downloaded again, and a reinstall downloads none of them again.
        ctx.step("Downloading PyTorch (3.5 GB)", until=0.12)
        ctx.pip(self.id, *TORCH, TORCHAUDIO, "--index-url", TORCH_INDEX, fail=_download_failed(self.title))
        ctx.step("Installing Pixal3D's packages", until=0.2)
        general, torch_index = _pixal3d_pip_pins()
        # the tested versions also for install.py's own pip calls (its unpinned dependencies: utils3d's moderngl and
        # glcontext came newer than the tested venv's), where they served the packages above
        constraints = self._pip_tested_versions(ctx, general) if general else None
        if torch_index:
            ctx.pip(self.id, *torch_index, "--index-url", TORCH_INDEX, fail=_download_failed(self.title))
        # install.py's own `pip install spaces` then finds it; --no-deps: spaces' unpinned gradio dependency
        # (hf_xet has none).
        ctx.pip(self.id, "--no-deps", PIXAL3D_SPACES, HF_XET, fail=_download_failed(self.title))
        self._link_user_hf_cache(ctx)
        # --no-pillow-simd: its Pillow-SIMD wheel imports jpeg62.dll, which neither the wheel nor Windows
        # has, so it never loads; install.py would then print a traceback and fall back to Pillow anyway.
        self._download_models(ctx, "--no-pillow-simd", until=0.94, constraints=constraints)
        self._make_copies(ctx, until=0.96)
        ctx.step("Checking the installation", until=0.97)
        ctx.pip(self.id, "--no-deps", *PIXAL3D_GRADIO, fail=_download_failed(self.title))
        self._hi3d_packages(ctx)
        share_torch_files(ctx, self.id)
        # below sm_80 the 3D decoders convolve on spconv (the fork's precision profile): Pixal3D's own need there, not
        # only Hi3D's, so a failed spconv install fails Pixal3D's check instead of passing as a warning
        card = _card()
        spconv = ", spconv.pytorch" if card is not None and card.compute_cap < DECODE_PRODUCTS_BELOW else ""
        self._verify(ctx, "from gradio import Server\n"
                          "import spaces, torch, hf_xet, o_voxel, flex_gemm, cumesh, nvdiffrast, natten, pixal3d_kernels"
                          f"{spconv}\n"
                          "print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())\n"
                          "if not torch.cuda.is_available(): raise SystemExit('PyTorch cannot use the NVIDIA GPU')\n"
                          "print('Pixal3D packages OK')")
        # Last, everything it measures being installed - and only when due: an update of Pixal3D's checkout runs
        # install() again, and the card measured at the first install is the same card.
        if ctx.dry_run or pixal3d_calibration_work():
            self._calibrate(ctx, until=1.0)

    def _calibrate(self, ctx: Ctx, until: float) -> None:
        """Pixal3D measured once on this PC's graphics card (PIXAL3D_CALIBRATION, above): the fork's
        `python -m pixal3d.calibrate` in Pixal3D's venv, offline. It first waits for the card to be quiet (another
        program at work on it would make every timing wrong; at most 15 minutes) and takes a few minutes more. A
        failure is a warning, never the install's (_calibration_failed_words: below sm_80 what the card does
        meanwhile; a step cut short there runs again, CALIBRATION_RETRY_WORK). Hi3D's step
        (spconv's choices for its convolutions) is left out while Hi3D's AI models are not here - their download runs
        it (model_downloads.hi3d_calibration) - and runs alone when it is all that is missing; so do the other steps
        a standing record lacks (CALIBRATION_STEPS_ALONE: the int8 tile choice and the group GEMMs' race, the
        decoders' spconv choices, the old cards' element-wise race, Turing's int8 attention race, a new attention
        wheel's timings, the spconv decoders' GEMM-call floor for the decode in boxes), several of them in one run."""
        why = None if ctx.dry_run else pixal3d_calibration_work()
        parts = (why or "").split("; ")
        alone = all(part in CALIBRATION_STEPS_ALONE for part in parts)
        only = ",".join(CALIBRATION_STEPS_ALONE[part] for part in parts) if alone else ""
        skip = "" if only or ctx.dry_run or models.hi3d_on_disk() else "spconv"
        argv = pixal3d_calibration_command(only=only, skip=skip, check=not ctx.dry_run)
        if argv is None:
            return      # a checkout of the fork without it, or (--force-pixal3d) a card it does not support
        if DECODE_PRODUCTS_AGAIN_WORK in parts:
            _forget_decode_products()       # the fork's calibrate keeps a section that passed: measured anew
        retried = [CALIBRATION_STEPS_ALONE[part] for part in parts if only and part in CALIBRATION_RETRY_WORK.values()]
        if retried:
            _note_retries(retried)
        elif not only and not ctx.dry_run:
            _forget_retries()               # a whole calibration: its own steps cut short get their own retries
        ctx.step(progress.CALIBRATING_STEP, until=until)
        ctx.out.line("Pixal3D now measures your graphics card once, so that it runs as fast as it can on it. Please "
                     "keep heavy programs (Blender, games, videos, Photoshop) closed until it is done.")
        try:
            ctx.run(argv, cwd=paths.PIXAL3D_DIR, env=ctx.env(self.id, pixal3d_calibration_env()),
                    fail="the calibration did not finish", on_line=lambda text: _calibration_line(ctx, text))
        except InstallError:
            words = _calibration_failed_words(retried)
            if words:
                ctx.out.line(words)

    def _link_models(self, ctx: Ctx) -> None:
        """code/pixal3d/MODELS -> models/pixal3d, before anything is downloaded into it (install/models.py)."""
        work = models.pixal3d_link_work()
        if ctx.dry_run:
            if work:
                ctx.out.line(f"  [dry-run] {work}: code\\pixal3d\\MODELS becomes a link to models\\pixal3d")
            return
        models.unwrap_copied_folder(ctx.out.line)
        if work:
            ctx.step("Moving its AI models into models\\pixal3d", until=0.0)
            models.link_pixal3d_models(ctx.out.line)

    def _link_user_hf_cache(self, ctx: Ctx) -> None:
        """Weights the artist already has in their own HuggingFace cache (same drive): linked, not downloaded."""
        if ctx.dry_run:
            return
        missing = set(self.missing_files())
        for name, repo_id, files in _hf_repos():
            if name in missing:
                models.link_from_hf_cache(PIXAL3D_MODELS / "hub", repo_id, files, ctx.out.line)

    def _download_models(self, ctx: Ctx, *args: str, until: float, constraints: Path | None = None) -> None:
        """Pixal3D's install.py: downloads what is missing of its AI models (without --models-only, it installs
        its packages first, at the tested versions of `constraints`). The progress is how far the models folder has
        grown."""
        downloads = ctx.dry_run or bool(self.missing_files())
        ctx.step("Downloading its AI models (about 27 GB)" if downloads else "Running Pixal3D's own setup",
                 until=until, measure=lambda: (_pixal3d_models_bytes(), PIXAL3D_MODELS_BYTES))
        env = _pixal3d_models_env()
        if constraints is not None:         # install.py adds it as -c to its pip calls (they run --isolated: no PIP_*)
            env["SPELLCAST_PIP_CONSTRAINT"] = str(constraints)
        if not ctx.dry_run and models.partial_download_bytes(PIXAL3D_MODELS) >= RESUME_OVER_HTTPS_BYTES:
            ctx.out.line("Continuing the AI model download that was interrupted last time, where it stopped.")
            env["HF_HUB_DISABLE_XET"] = "1"
        ctx.run([self._python(), "install.py", *args], cwd=paths.PIXAL3D_DIR, env=ctx.env(self.id, env),
                fail=("Pixal3D's own installer stopped with an error, most often because a download was "
                      "interrupted. Start Spellcast3D again: finished downloads are kept and the rest resumes. "
                      + LOG_HINT),
                notes=((PIXAL3D_ENV_HINT, PIXAL3D_ENV_NOTE),))
        missing = [] if ctx.dry_run else self.missing_files()
        if missing:
            raise InstallError(f"Some of Pixal3D's AI model files are still missing ({', '.join(missing)}), most "
                               f"often because a download was interrupted. Start Spellcast3D again: finished "
                               f"downloads are kept and the rest resumes. {LOG_HINT}")

    def _hi3d_packages(self, ctx: Ctx) -> None:
        """Hi3D's packages (HI3D_PACKAGES, after install.py's own pip calls; --no-deps: exact pins that move nothing
        else), checked by importing them. Best effort: a pip or import failure is a warning, never Pixal3D's."""
        try:
            ctx.pip(self.id, "--no-deps", *HI3D_PACKAGES, fail=_download_failed("Hi3D"))
            ctx.run([self._python(), "-c", "import spconv.pytorch, pyfqmr; print('Hi3D packages OK')"],
                    cwd=paths.venv_dir(self.id), env=ctx.env(self.id), fail=_broken("Hi3D"))
        except InstallError as e:
            ctx.out.line(f"Warning: {e.reason} {HI3D_LATER}")

    def _checkout_hi3d(self, ctx: Ctx) -> None:
        """Hi3D's code, the submodule code/hi3d, checked out when .gitmodules names it and it is not (an update made
        without `git submodule update`). Without it Pixal3D works on: a warning, and the next start tries again."""
        if (paths.HI3D_DIR / models.HI3D_CODE_LANDMARK).exists() or not hi3d_submodule():
            return
        try:
            checkout_submodule(ctx, paths.HI3D_DIR, models.HI3D_CODE_LANDMARK, "Hi3D")
        except InstallError as e:
            ctx.out.line(f"Warning: Hi3D's code could not be put into code\\hi3d ({e.reason}) {HI3D_LATER}")

    def _make_copies(self, ctx: Ctx, until: float) -> None:
        """The faster-loading copies of Pixal3D's flow models (models.PIXAL3D_COPIES_BYTES), made by the fork's
        `python -m pixal3d.staged.weights <snapshot>` in Pixal3D's venv: it makes those that are missing or stale
        and leaves those that are up to date alone, reading only what is on disk. A failure is a warning, never
        the install's: Pixal3D then reads the original files, which only takes longer."""
        snapshot = models.hf_snapshot(PIXAL3D_MODELS / "hub", PIXAL3D_HF_REPO)
        if snapshot is None and not ctx.dry_run:
            ctx.out.line(f"Warning: Pixal3D's models were not found in {_root_rel(PIXAL3D_MODELS)}, so their "
                         f"faster-loading copies were not made. {PIXAL3D_COPIES_LATER}")
            return
        ctx.step("Preparing its AI models for faster loading", until=until)
        folder = snapshot or PIXAL3D_MODELS / "hub" / "<the snapshot of TencentARC/Pixal3D>"
        # Cast on the CPU: with the card hidden the process takes none of its memory (the card's precision profile,
        # which decides the copies' dtypes, is told through PIXAL3D_CARD_SM). Offline: nothing is downloaded.
        env = {**_pixal3d_models_env(), **gpu.pixal3d_card_env(_card()), "HF_HUB_OFFLINE": "1",
               "CUDA_VISIBLE_DEVICES": "-1"}
        try:
            ctx.run([self._python(), "-m", models.PIXAL3D_COPIES_MODULE, str(folder)], cwd=paths.PIXAL3D_DIR,
                    env=ctx.env(self.id, env), fail="the copies were not all made")
        except InstallError:
            ctx.out.line(f"Warning: the faster-loading copies of some of Pixal3D's AI models were not made (see "
                         f"the lines above). {PIXAL3D_COPIES_LATER}")

    def _pip_tested_versions(self, ctx: Ctx, general: tuple[str, ...]) -> Path | None:
        """Install install.py's packages at the versions of a tested install (their dependencies too).

        Several of install.py's names are unpinned, so each new install would otherwise get whatever is
        newest that day. Pixal3D's own pins win; when the tested versions no longer fit them (after a
        Pixal3D update), the packages are installed with Pixal3D's pins alone. -> the constraints file where it
        served (install.py's own pip calls take it too: _download_models), else None.
        """
        pinned = {_package_key(req.split("==")[0]) for req in general if "==" in req}
        lines = []
        try:
            for line in PIXAL3D_CONSTRAINTS.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and _package_key(line.split("==")[0]) not in pinned:
                    lines.append(line)
        except OSError:
            pass
        constraints = paths.INSTALL_TMP / PIXAL3D_CONSTRAINTS.name
        if lines and not ctx.dry_run:
            constraints.parent.mkdir(parents=True, exist_ok=True)
            constraints.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if not lines:
            ctx.pip(self.id, *general, fail=_download_failed(self.title))
            return None
        try:
            ctx.pip(self.id, *general, "-c", str(constraints), fail=_download_failed(self.title))
        except InstallError:
            ctx.out.line("Installing the tested package versions did not work; installing the versions "
                         "Pixal3D asks for instead.")
            ctx.pip(self.id, *general, fail=_download_failed(self.title))
            return None
        return constraints


def _package_key(name: str) -> str:
    """A requirement's project name in normalised form ("Pillow" / "nest_asyncio" -> "pillow" / "nest-asyncio")."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def _hf_repos() -> tuple[tuple[str, str, tuple[str, ...] | None], ...]:
    """(name, HuggingFace repo, the files it needs or None = all) of what install.py downloads from there.
    MoGe-2 is downloaded whole; model.pt is the file that matters."""
    return (("Pixal3D", PIXAL3D_HF_REPO, _pixal3d_hf_files()), ("MoGe-2", MOGE_HF_REPO, None))


def pixal3d_install_inputs() -> str:
    """What of the fork's checkout its install depends on, as one hash: install.py (its pins, its downloads, the
    wheels it installs), every file of whl/ (the prebuilt wheels and the notes that name them) and the formats of the
    faster-loading copies in staged/weights.py - the model-dtypes copies' and the flows' int8 copies' (a new one has
    install() make them again: an install made before the int8 copies existed makes them once). Not the fork's
    commit: most updates change only Python code, which runs from the checkout, and those must start without a
    reinstall."""
    files = {"install.py": state.file_sha256(paths.PIXAL3D_DIR / "install.py"),
             "copies_format": pixal3d_copies_format()}
    int8_format = pixal3d_copies_format("INT8_FORMAT")
    if int8_format:         # only where the fork has them: an older checkout's install keeps its fingerprint
        files["int8_copies_format"] = int8_format
    for path in sorted(PIXAL3D_WHL.glob("*")) if PIXAL3D_WHL.is_dir() else ():
        if path.is_file():
            files["whl/" + path.name] = wheel_sha256(path)
    return state.fingerprint(files)


def pixal3d_copies_format(name: str = "FORMAT") -> str:
    """The fork's weights.FORMAT (the version of its faster-loading copies; `name` INT8_FORMAT: of its flows' int8
    copies), read without importing it; "" where the file or the name is not there."""
    try:
        text = (paths.PIXAL3D_DIR / "pixal3d" / "staged" / "weights.py").read_text(encoding="utf-8")
    except OSError:
        return ""
    found = re.search(rf"^{name}\s*=\s*(['\"])(.*?)\1", text, re.MULTILINE)
    return found.group(2) if found else ""


def _pixal3d_install_py() -> tuple[str, ast.Module | None]:
    """Source and syntax tree of Pixal3D's install.py, read without importing (running) it."""
    try:
        source = (paths.PIXAL3D_DIR / "install.py").read_text(encoding="utf-8")
        return source, ast.parse(source)
    except (OSError, SyntaxError, ValueError):
        return "", None


def _str_list_assigned(tree: ast.AST | None, name: str) -> list[str] | None:
    """The literal list of strings assigned to `name` anywhere in tree (e.g. inside a function), or None."""
    for node in ast.walk(tree) if tree is not None else ():
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                return value
    return None


def _pixal3d_pip_pins() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(general packages, packages from the PyTorch index) that Pixal3D's install.py installs by name.

    Followed from install.py (its general_deps list and its xformers pin), so a Pixal3D update that
    changes them is followed too. Empty when install.py's layout changed: it then downloads them itself.
    """
    source, tree = _pixal3d_install_py()
    general = tuple(_str_list_assigned(tree, "general_deps") or ())
    torch_index = tuple(dict.fromkeys(re.findall(r"\bxformers==[\w.+]+", source)))
    return general, torch_index


@functools.cache
def _pixal3d_hf_files() -> tuple[str, ...]:
    """Files of TencentARC/Pixal3D that Pixal3D's install.py downloads (its PIXAL3D_SINGLE_VIEW_PATTERNS).

    Read from install.py without importing it, so a Pixal3D update that changes the list is followed;
    "ckpts/x.*" means ckpts/x.json + ckpts/x.safetensors.
    """
    patterns = (_str_list_assigned(_pixal3d_install_py()[1], "PIXAL3D_SINGLE_VIEW_PATTERNS")
                or list(_PIXAL3D_PATTERNS_FALLBACK))
    files: list[str] = []
    for p in patterns:
        files += [p[:-2] + ".json", p[:-2] + ".safetensors"] if p.endswith(".*") else [p]
    return tuple(files)


# Servers preflight can check with a HEAD request (their front page answers 200), by host name.
PROBE_URLS = {"huggingface.co": "https://huggingface.co/", "github.com": "https://github.com/"}


class _PinnedModels(Component):
    """A component whose AI models the installer fetches itself (models.py): a pinned manifest of files in its models
    folder. Copies on this PC are hard-linked (model_sources), the rest downloaded (resumed, SHA-256 checked), and a
    file that got into the folder some other way is checked once. Qwen-Image 2.1 and the retopology tools."""
    models_dir: Path
    manifest: models.Manifest
    models_step = "Downloading its AI models"      # its download's step (the bar says how many GB)

    def missing_files(self) -> list[str]:
        return models.missing_files(self.models_dir, self.manifest)

    def missing_gb(self) -> float:
        """How much of its AI models is still to be downloaded (the files not there whole, less their .part, and less
        those a copy on this PC provides: _seed_models hard-links them, no download and no disk space)."""
        manifest = self.manifest
        missing = self.missing_files()
        linkable = self._linkable(missing)
        todo = {rel: manifest[rel] for rel in missing if rel in manifest and rel not in linkable}
        return (sum(size for size, _sha in todo.values()) - models.present_bytes(self.models_dir, todo)) / 1e9

    def _linkable(self, missing: list[str]) -> set[str]:
        """Of the `missing` model files, those a model_sources() folder has at their size on the models folder's drive
        (_local_copies: a few stat calls). The disk budget counts them as there: a hard link takes no space, and
        without this, --models-from or a HuggingFace cache made preflight ask for room for files never downloaded."""
        missing = [rel for rel in missing if rel in self.manifest]
        if not missing:
            return set()
        local = self._local_copies()
        return {rel for rel in missing if _key(self.models_dir / rel) in local}

    def unverified(self) -> list[str]:
        """Its AI model files that are there but were never checked against their pins on this PC (e.g. copied into
        the models folder by hand): a few stat calls."""
        return models.unverified_files(self.models_dir, self.manifest)

    def repair_work(self) -> str | None:
        work = super().repair_work()
        if work is None and self.unverified():
            work = "some of its AI model files were not downloaded by Spellcast3D and are checked once"
        return work

    def _check_present(self, ctx: Ctx) -> None:
        """Check the unverified model files once (SHA-256): a wrong one is removed, so it is fetched again."""
        models.check_unverified(self.models_dir, self.manifest, ctx.out.line)

    def models_present_gb(self) -> float:
        """On disk already, or a hard link away (_linkable: a copy on this PC; a link takes no disk space)."""
        manifest = self.manifest
        linkable = self._linkable(models.missing_files(self.models_dir, manifest))
        rest = {rel: value for rel, value in manifest.items() if rel not in linkable}
        return (models.present_bytes(self.models_dir, rest) + sum(manifest[rel][0] for rel in linkable)) / 1e9

    def download_hosts(self) -> set[str]:
        """The servers its missing AI models come from (those preflight can check), leaving out the files a copy on
        this PC provides (model_sources: linked, not downloaded)."""
        manifest = self.manifest
        missing = models.missing_files(self.models_dir, manifest)
        if missing:
            local = self._local_copies()
            missing = [rel for rel in missing if _key(self.models_dir / rel) not in local]
        hosts = {urllib.parse.urlsplit(self.model_url(rel)).hostname for rel in missing}
        return {host for host in hosts if host in PROBE_URLS}

    def _local_copies(self) -> set[str]:
        """Our files (_key of their path here) that a model_sources() folder has at the pinned size on the drive of
        the models folder, where _seed_models can hard-link them (a few stat calls; the hash is checked then)."""
        try:
            drive = os.stat(_existing(self.models_dir)).st_dev
        except OSError:
            return set()
        found: set[str] = set()
        for src, dst, manifest, *renamed in self.model_sources():
            names = renamed[0] if renamed else {}
            for rel, (size, _sha) in manifest.items():
                try:
                    st = os.stat(Path(src) / names.get(rel, rel))
                except OSError:
                    continue
                if st.st_size == size and st.st_dev == drive:
                    found.add(_key(Path(dst) / rel))
        return found

    def model_url(self, rel: str) -> str:
        raise NotImplementedError

    def model_sources(self) -> list[tuple]:
        """(folder on this PC, the folder of ours it matches, the manifest relative to both[, {our name: its name
        there} for files named otherwise there]) to link models from."""
        return []

    def _seed_models(self, ctx: Ctx) -> None:
        for src, dst, manifest, *renamed in self.model_sources():
            wanted = models.missing_files(dst, manifest)
            if wanted:
                models.link_from_local_dir(src, dst, wanted, manifest, ctx.out.line,
                                           names=renamed[0] if renamed else None)

    def _download_models(self, ctx: Ctx) -> None:
        for rel in self.missing_files():
            if rel not in self.manifest:
                continue
            size, sha = self.manifest[rel]
            ctx.out.line(f"Downloading {rel} ({size / 1e9:.2f} GB) ...")
            models.download_file(self.model_url(rel), self.models_dir / rel, size, sha, ctx.out.line)

    def _after_models(self, ctx: Ctx) -> None:
        """What needs the models in place (e.g. links to them)."""

    def _fetch_models(self, ctx: Ctx, until: float) -> None:
        """Its AI models: those on this PC already are hard-linked, the rest downloaded (resumed, SHA-256 checked)."""
        if ctx.dry_run:
            for rel in self.manifest:
                name = rel.replace("/", "\\")
                ctx.out.line(f"  [dry-run] models\\{self.models_dir.name}\\{name}: if missing, link a copy on this "
                             f"PC, else download {self.model_url(rel)}")
            return
        self._check_present(ctx)
        if self.missing_files():
            self._seed_models(ctx)
        missing = self.missing_files()
        if missing:
            todo = sum(self.manifest[rel][0] for rel in missing if rel in self.manifest)
            total = sum(size for size, _sha in self.manifest.values())
            ctx.step(f"{self.models_step} ({max(todo, 1e7) / 1e9:.1f} GB)", until=until,
                     measure=lambda: (models.present_bytes(self.models_dir, self.manifest), total))
            try:
                self._download_models(ctx)
            except models.ChecksumError as e:
                raise InstallError(f"An AI model file of {self.title} was downloaded in full, but it is not the "
                                   f"expected file ({e}). Start Spellcast3D again to download it again; if that keeps "
                                   f"happening, a proxy or an antivirus may be changing it. {LOG_HINT}") from e
            except OSError as e:
                raise InstallError(f"Downloading the AI models of {self.title} failed ({e}). Check your internet "
                                   f"connection and start Spellcast3D again: finished downloads are kept and the "
                                   f"rest resumes. {LOG_HINT}") from e
            missing = self.missing_files()
            if missing:
                raise InstallError(f"Some of {self.title}'s AI model files are still missing ({', '.join(missing)}). "
                                   f"Start Spellcast3D again to retry. {LOG_HINT}")
        self._after_models(ctx)


def hi3d_submodule() -> bool:
    """Whether .gitmodules names code/hi3d (Hi3D's code is a submodule of this checkout). A release's folder leaves
    that file to git (install/layout.py), which is asked for it then."""
    text = layout.tracked_text(".gitmodules", git_exe())
    if text is None:
        return False
    rel = re.escape(paths.HI3D_DIR.relative_to(paths.ROOT).as_posix())
    return re.search(rf"^\s*path\s*=\s*{rel}\s*$", text, re.MULTILINE) is not None


def hi3d_work() -> str | None:
    """What an installed Pixal3D still has to do for Hi3D, or None: check out its code (its AI models are the 3D
    stage's own download, never the installer's). A stat call while the code is there."""
    if not (paths.HI3D_DIR / models.HI3D_CODE_LANDMARK).exists() and hi3d_submodule():
        return "Hi3D's code is not checked out in code\\hi3d yet"
    return None


class _RetopoTool(_PinnedModels):
    """A retopology tool in its own venv: torch, its pinned requirement files, prebuilt wheels (--no-deps), the big
    files shared with another venv's by hard links, and its AI models fetched by the installer (models.py)."""
    uses_gpu = True
    needs_torch = True
    torch_shared_gb = SHARED_TORCH_GB
    source_files: tuple[str, ...] = ()      # its fingerprint in a copy without git data
    torch_extras: tuple[str, ...] = ()      # installed with torch from the PyTorch index (PyPI has other builds)
    wheels: tuple[Path, ...] = ()
    verify_code = ""

    def requirement_files(self) -> dict[str, Path]:
        return {"requirements": INSTALL_DIR / f"requirements-{self.id}.txt",
                "constraints": INSTALL_DIR / f"constraints-{self.id}.txt",
                "nodeps": INSTALL_DIR / f"requirements-{self.id}-nodeps.txt"}

    def inputs(self) -> dict[str, str]:
        files = self.requirement_files()
        return {"source": state.source_version(self.submodule, self.source_files),
                "torch": " ".join(TORCH + self.torch_extras),
                **{key: state.file_sha256(path) for key, path in files.items()
                   if key == "requirements" or path.is_file()},
                "wheels": " ".join(f"{w.name}={wheel_sha256(w)}" for w in self.wheels)}

    def prepare(self, ctx: Ctx) -> None:
        super().prepare(ctx)
        self._prepare_files(ctx)

    def _prepare_files(self, ctx: Ctx) -> None:
        """Its requirement file and wheels must be here (Pixal3D's wheels: code/pixal3d checked out)."""
        if any(w.parent == PIXAL3D_WHL for w in self.wheels):
            checkout_submodule(ctx, paths.PIXAL3D_DIR, "install.py", "Pixal3D")     # for its prebuilt wheels
        missing = [_root_rel(p) for p in (self.requirement_files()["requirements"], *self.wheels) if not p.is_file()]
        if missing and not ctx.dry_run:
            raise InstallError(f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} missing, so this copy of "
                               f"Spellcast3D is incomplete. Please download Spellcast3D again.")

    def install(self, ctx: Ctx) -> None:
        ctx.out.line("If the download is interrupted or this window is closed, just start Spellcast3D again:")
        ctx.out.line("finished downloads are kept and the rest resumes.")
        # From pip's shared cache when another tool installed the same torch: nothing is downloaded again.
        ctx.step("Installing PyTorch (3.5 GB, downloaded only once for all tools)", until=0.3)
        ctx.pip(self.id, *TORCH, *self.torch_extras, "--index-url", TORCH_INDEX, fail=_download_failed(self.title))
        self._share_files(ctx, "PyTorch", until=0.35)
        ctx.step(f"Installing {self.title}'s packages", until=0.6)
        self._pip_packages(ctx)
        self._share_files(ctx, "its packages", until=0.65)
        self._fetch_models(ctx, until=0.97)
        ctx.step("Checking the installation", until=1.0)
        self._verify(ctx, self.verify_code)

    def repair(self, ctx: Ctx) -> None:
        self._fetch_models(ctx, until=1.0)

    def _pip_packages(self, ctx: Ctx) -> None:
        files = self.requirement_files()
        extra = ["-c", str(files["constraints"])] if files["constraints"].is_file() else []
        ctx.pip(self.id, "-r", str(files["requirements"]), *extra, fail=_download_failed(self.title))
        if files["nodeps"].is_file():
            ctx.pip(self.id, "--no-deps", "-r", str(files["nodeps"]), *extra, fail=_download_failed(self.title))
        # Ours (code/whl) and Pixal3D's (code/pixal3d/whl) alike: a rebuilt wheel keeps its version number (e.g.
        # new GPU kernels), and pip skips a wheel whose version is installed, so it must be told to replace it. The
        # sharing step after this links the files back to another venv's when they are the same.
        if self.wheels:
            ctx.pip(self.id, "--no-deps", "--force-reinstall", *(str(w) for w in self.wheels),
                    fail=_download_failed(self.title))

    def _share_files(self, ctx: Ctx, what: str, until: float) -> None:
        """Hard-link the venv's big files that another venv with the same torch has byte for byte (saves ~7 GB)."""
        ctx.step("Sharing identical files with the other tools", until=until)
        twins = venvs.torch_twins(self.id, TORCH_DIST)
        names = ", ".join(f"code\\venvs\\{t.name}" for t in twins)
        if ctx.dry_run:
            ctx.out.line(f"  [dry-run] hard-link the files of {what} that are the same in: "
                         f"{names or 'no other venv yet'}")
            return
        if not twins:
            return
        count, saved = venvs.share_identical_files(paths.venv_dir(self.id), twins)
        if count:
            ctx.out.line(f"{count} big file(s) of {what} ({saved / 1e9:.1f} GB) are the same as in {names}: they "
                         f"are hard links to those now, which takes no extra disk space.")
        ctx.run([self._python(), "-c", "import torch; print('torch', torch.__version__, 'imports')"],
                cwd=paths.venv_dir(self.id), env=ctx.env(self.id), fail=_broken(self.title))


def _key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _existing(path: Path) -> Path:
    """`path`, or the nearest folder above it that exists (the drive of a folder still to be made)."""
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def _source_folders(*names: str) -> list[Path]:
    """The --models-from / SPELLCAST_MODELS_FROM folders, and in each the sub-folders `names` (e.g. another
    Spellcast3D's models folder holds lato2\\ and triflow\\)."""
    return [folder / name if name else folder for folder in models.extra_sources() for name in ("", *names)]


class Lato2(_RetopoTool):
    id = paths.LATO2
    title = "LATO.2"
    about = ("Retopology with AI (MIT licence): makes a clean low poly of the 3D model on the GPU. Downloads about "
             "1 GB of packages (PyTorch comes from the download cache when another tool has it) and 4.8 GB of AI "
             "models (not those already on this PC). Its Python environment takes about 9 GB of disk, or about 1 GB "
             "when it shares its identical files (PyTorch's among them) with another tool; takes 5-30 minutes "
             "depending on your internet speed.")
    size_gb = 15.0
    models_gb = (sum(size for size, _sha in {**models.LATO2_FILES, **models.DINOV2_FILES}.values())
                 + models.DINOV2_CODE_BYTES) / 1e9
    submodule = paths.LATO2_DIR
    marker = "scripts/e2e_inference.py"
    source_files = ("scripts/e2e_inference.py",)
    wheels = (O_VOXEL_WHEEL, FLASH_ATTN_WHEEL)
    models_dir = paths.MODELS_LATO2
    manifest = {**models.LATO2_FILES, **models.DINOV2_FILES}
    # o_voxel is only looked for: its package __init__ needs triton, flex_gemm and cumesh, which LATO.2's launcher
    # never loads (it imports o_voxel's convert module on its own).
    verify_code = ("import importlib.util, numpy, scipy, PIL, torch, spconv.pytorch, flash_attn, trimesh, fastapi, "
                   "uvicorn\n"
                   "print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())\n"
                   "if not torch.cuda.is_available(): raise SystemExit('PyTorch cannot use the NVIDIA GPU')\n"
                   "if importlib.util.find_spec('o_voxel') is None: raise SystemExit('o_voxel is missing')\n"
                   "print('LATO.2 packages OK')")

    def supported(self, gpu_info: gpu.GpuInfo | None) -> tuple[bool, str]:
        return gpu.lato2_supported(gpu_info)

    def missing_files(self) -> list[str]:
        code = self.models_dir / models.DINOV2_CODE / models.DINOV2_CODE_LANDMARK
        return super().missing_files() + ([] if code.is_file() else ["DINOv2's code"])

    def unverified(self) -> list[str]:
        code = self.models_dir / models.DINOV2_CODE
        stray = (code / models.DINOV2_CODE_LANDMARK).is_file() and not models.is_recorded(code,
                                                                                       models.DINOV2_CODE_DIGEST)
        return super().unverified() + (["DINOv2's code"] if stray else [])

    def _check_present(self, ctx: Ctx) -> None:
        super()._check_present(ctx)
        code = self.models_dir / models.DINOV2_CODE
        if (code / models.DINOV2_CODE_LANDMARK).is_file() and not models.dinov2_code_ok(code):
            ctx.out.line(f"{models.DINOV2_CODE} is not the DINOv2 code this version of Spellcast3D needs (commit "
                         f"{models.DINOV2_CODE_COMMIT[:7]}): it is replaced.")
            shutil.rmtree(code, ignore_errors=True)

    def models_present_gb(self) -> float:
        code = models.DINOV2_CODE_BYTES if "DINOv2's code" not in self.missing_files() else 0
        return super().models_present_gb() + code / 1e9

    def model_url(self, rel: str) -> str:
        if rel == models.DINOV2_CHECKPOINT:
            return models.DINOV2_URL
        return models.hf_url(models.LATO2_HF_REPO, models.LATO2_HF_REVISION, rel)

    def model_sources(self) -> list[tuple[Path, Path, models.Manifest]]:
        dinov2 = self.models_dir / "dinov2"
        in_hub = {rel.removeprefix("dinov2/"): value for rel, value in models.DINOV2_FILES.items()}
        return ([(folder, self.models_dir, self.manifest) for folder in _source_folders("lato2")]
                + [(snapshot, self.models_dir, models.LATO2_FILES)
                   for snapshot in models.hf_cache_sources(models.LATO2_HF_REPO, models.LATO2_HF_REVISION)]
                + [(hub, dinov2, in_hub) for hub in models.torch_hub_dirs()])

    def _seed_models(self, ctx: Ctx) -> None:
        super()._seed_models(ctx)
        code = self.models_dir / models.DINOV2_CODE
        for folder in [*(f / models.DINOV2_CODE for f in _source_folders("lato2")),
                       *(hub / "facebookresearch_dinov2_main" for hub in models.torch_hub_dirs())]:
            if models.link_dinov2_code(folder, code, ctx.out.line):
                break

    def _fetch_models(self, ctx: Ctx, until: float) -> None:
        super()._fetch_models(ctx, until)
        if ctx.dry_run:
            code = models.DINOV2_CODE.replace("/", "\\")
            ctx.out.line(f"  [dry-run] models\\lato2\\{code}: if missing, link a copy on this PC, else download "
                         f"{models.DINOV2_CODE_URL}")

    def _download_models(self, ctx: Ctx) -> None:
        super()._download_models(ctx)
        code = self.models_dir / models.DINOV2_CODE
        if not (code / models.DINOV2_CODE_LANDMARK).is_file():
            ctx.out.line("Downloading DINOv2's code (about 3 MB) from GitHub ...")
            models.download_dinov2_code(code, ctx.out.line)


class Triflow(_RetopoTool):
    id = paths.TRIFLOW
    title = "TriFlow"
    about = ("Retopology with AI, free for NON-COMMERCIAL use only (trial: TriFlow's ADPNCL licence and MeshLib's "
             "terms): makes a clean low poly of mostly quads on the GPU. Downloads about 1.5 GB of packages "
             "(PyTorch comes from the download cache when another tool has it) and 1.3 GB of AI models. Its Python "
             "environment takes about 11 GB of disk, or about 2 GB when it shares its identical files (PyTorch's among "
             "them) with another tool.")
    size_gb = 13.0
    models_gb = sum(size for size, _sha in models.TRIFLOW_FILES.values()) / 1e9
    submodule = paths.TRIFLOW_DIR
    marker = "inference.py"
    source_files = ("inference.py", "requirements.txt")
    # No xformers: 0.0.32 refuses to import next to flash_attn 2.8.3. Below sm_80 (no flash_attn kernels) the
    # launcher runs TRELLIS' and Direct3D-S2's attention on torch's SDPA (launchers/attn_shim.py).
    wheels = (TORCHSPARSE_WHEEL, PYFQMR_WHEEL, FLASH_ATTN_WHEEL)
    models_dir = paths.MODELS_TRIFLOW
    manifest = models.TRIFLOW_FILES
    verify_code = ("import torch, spconv.pytorch, torchsparse, pyfqmr_triflow, flash_attn, meshlib.mrmeshpy, meshiki, "
                   "rootpath, hydra, safetensors, trimesh, fastapi, uvicorn\n"
                   "print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())\n"
                   "if not torch.cuda.is_available(): raise SystemExit('PyTorch cannot use the NVIDIA GPU')\n"
                   "print('TriFlow packages OK')")

    def supported(self, gpu_info: gpu.GpuInfo | None) -> tuple[bool, str]:
        return gpu.triflow_supported(gpu_info)

    def inputs(self) -> dict[str, str]:
        nested = " ".join(f"{rel}@{head}" for rel, head in submodules.triflow_heads().items())
        return {**super().inputs(), "nested": nested}

    def repair_work(self) -> str | None:
        return super().repair_work() or models.triflow_link_work()

    def prepare(self, ctx: Ctx) -> None:
        """TriFlow's checkout and its nested ones, byte-identical to their commits (install/submodules.py)."""
        checkout_submodule(ctx, self.submodule, self.marker, self.title, git_config=("core.autocrlf=false",))
        git = git_exe()
        if git is not None and (self.submodule / ".git").exists():
            ctx.step(f"Downloading the source of {self.title}'s parts", until=0.0)
            submodules.init_triflow(
                git, lambda argv, cwd: ctx.run(argv, cwd=cwd, env=ctx.env(None),
                                               fail=f"Downloading the source of {self.title} with git failed. Check "
                                                    f"your internet connection and start again."),
                everything=ctx.dry_run)
            if not ctx.dry_run:
                self._check_untouched(ctx, git)
        missing = [] if ctx.dry_run else submodules.triflow_missing()
        if missing:
            raise InstallError(f"code\\triflow\\{missing[0]} is missing or empty, so this copy of Spellcast3D is "
                               f"incomplete. Please download Spellcast3D again.")
        self._prepare_files(ctx)
        if not ctx.dry_run:
            models.link_triflow_checkpoints(ctx.out.line)

    def _check_untouched(self, ctx: Ctx, git: str) -> None:
        """Make sure TriFlow's files are exactly its commits': files that differ only by CRLF line endings (a
        checkout with Git's autocrlf) are written again as committed; anything else is reported."""
        crlf: list[Path] = []
        problems = submodules.byte_identity_problems(git, self.submodule, submodules.TRIFLOW_ALLOWED_EXTRA,
                                                     crlf_only=crlf)
        if crlf:
            by_repo: dict[Path, list[str]] = {}
            for file in crlf:
                repo = next(r for r in [*reversed(submodules.triflow_nested_dirs()), self.submodule]
                            if file.is_relative_to(r))
                by_repo.setdefault(repo, []).append(file.relative_to(repo).as_posix())
            ctx.out.line(f"{len(crlf)} of TriFlow's files have Windows line endings; writing them as committed.")
            for repo, files in by_repo.items():
                ctx.run([git, "-c", "core.autocrlf=false", "-C", str(repo), "checkout-index", "-f", "--", *files],
                        cwd=repo, env=ctx.env(None), fail=_broken(self.title))
            problems = submodules.byte_identity_problems(git, self.submodule, submodules.TRIFLOW_ALLOWED_EXTRA)
        if problems:
            ctx.out.line(f"Note: TriFlow's files are not exactly those of its commit ({len(problems)} difference(s), "
                         f"e.g. {problems[0]}). Its licence asks that changes be documented and sent to AUDI AG; "
                         f"see code\\triflow\\LICENSE.")

    def model_url(self, rel: str) -> str:
        return models.hf_url(models.TRIFLOW_HF_REPO, models.TRIFLOW_HF_REVISION, rel)

    def model_sources(self) -> list[tuple[Path, Path, models.Manifest]]:
        return ([(folder, self.models_dir, self.manifest) for folder in _source_folders("triflow", "checkpoints")]
                + [(snapshot, self.models_dir, self.manifest)
                   for snapshot in models.hf_cache_sources(models.TRIFLOW_HF_REPO, models.TRIFLOW_HF_REVISION)]
                + [(paths.TRIFLOW_CHECKPOINTS, self.models_dir, self.manifest)])

    def _after_models(self, ctx: Ctx) -> None:
        if not models.link_triflow_checkpoints(ctx.out.line):
            raise InstallError(f"Could not put TriFlow's AI models into code\\triflow\\checkpoints. Start "
                               f"Spellcast3D again to retry. {LOG_HINT}")


class Qwen(_PinnedModels):
    """Qwen-Image 2.1, the default image generator: no PyTorch. A small venv for its launcher (fastapi, uvicorn,
    pillow: requirements-qwen.txt), Spellcast3D's own pinned build of stable-diffusion.cpp with NVIDIA's CUDA DLLs
    from the official release (sdcpp.py), and its AI models (models.py qwen_manifest): the Turbo's denoiser of the size
    spellcast3d-config.ini picks, or of a bigger size already in the folder (models.qwen_size_on_disk: the artist's
    file, never hashed or replaced), and the shared files. The size is not part of the fingerprint: a new size is a
    missing file, which repair() downloads without touching the rest. The full model's denoiser is no part of it (the
    hub downloads it on request: spellcast/model_downloads.py); one an older install downloaded stays untouched."""
    id = paths.QWEN
    title = "Qwen-Image 2.1"
    uses_gpu = True
    models_dir = paths.MODELS_QWEN
    # Not downloaded or kept when a venv here has PyTorch's identical CUDA DLLs (linked: sdcpp.link_cuda_dlls).
    torch_shared_gb = sdcpp.CUDA_GB
    VENV_GB = 0.1
    # The model size of this installer run (install/main.py pins it at the start): spellcast3d-config.ini edited while
    # it downloads must not change what it is fetching. None: what the file says now.
    pinned_quality: str | None = None
    pinned_text_encoder: str | None = None

    @property
    def quality(self) -> str:
        return self.pinned_quality or config.qwen_quality()

    @property
    def text_encoder(self) -> str:
        return self.pinned_text_encoder or config.qwen_text_encoder()

    @property
    def manifest(self) -> models.Manifest:      # type: ignore[override]
        return models.qwen_manifest(self.quality, self.models_dir, self.text_encoder)

    def _own_manifest(self) -> models.Manifest:
        """The manifest less a Turbo or a text encoder of another size than the config's: the artist's own file, used
        as it is."""
        own = {models.qwen_denoiser("turbo", self.quality)[0], models.qwen_text_encoder(self.text_encoder)[0]}
        shared = {rel for rel in models.QWEN_SHARED_FILES if rel != models.QWEN_TEXT_ENCODER}
        return {rel: pin for rel, pin in self.manifest.items() if rel in own or rel in shared}

    def unverified(self) -> list[str]:
        return models.unverified_files(self.models_dir, self._own_manifest())

    def _check_present(self, ctx: Ctx) -> None:
        models.check_unverified(self.models_dir, self._own_manifest(), ctx.out.line)

    @property
    def models_gb(self) -> float:               # type: ignore[override]
        return sum(size for size, _sha in self.manifest.values()) / 1e9

    @property
    def size_gb(self) -> float:                 # type: ignore[override]
        return self.models_gb + sdcpp.RUNTIME_GB + sdcpp.CUDA_GB + self.VENV_GB

    @property
    def about(self) -> str:                     # type: ignore[override]
        return (f"Image generation, built in (Qwen's research licence for the model; the pictures you make are yours, "
                f"also for commercial work). Downloads about {self.models_gb:.0f} GB of AI models (Qwen-Image 2.1 "
                f"Turbo; the full model is an optional download from the Images page; not those already on this PC) "
                f"and its runtime, stable-diffusion.cpp (0.3 GB; no PyTorch); takes 30-60 minutes depending on your "
                f"internet speed.")

    def requirements_file(self) -> Path:
        return INSTALL_DIR / "requirements-qwen.txt"

    def inputs(self) -> dict[str, str]:
        return {"runtime": f"{sdcpp.SDCPP.release.tag} {sdcpp.SDCPP.sha256} {sdcpp.CUDART.sha256}",
                "requirements": state.file_sha256(self.requirements_file())}

    def supported(self, gpu_info: gpu.GpuInfo | None) -> tuple[bool, str]:
        return gpu.qwen_supported(gpu_info)

    def missing_files(self) -> list[str]:
        return super().missing_files() + sdcpp.missing()

    def download_hosts(self) -> set[str]:
        hosts = super().download_hosts()
        runtime = bool(sdcpp.runtime_missing()) and not sdcpp.on_this_pc(sdcpp.SDCPP)
        cuda = [n for n in sdcpp.cuda_missing() if not sdcpp.torch_libs(n)] and not sdcpp.on_this_pc(sdcpp.CUDART)
        return hosts | ({"github.com"} if runtime or cuda else set())

    def model_url(self, rel: str) -> str:
        return models.hf_url(*models.QWEN_SOURCES[rel])

    def model_sources(self) -> list[tuple]:
        """Folders laid out like ours (--models-from: another Spellcast3D's models\\qwen-image), then the artist's
        HuggingFace caches, where the files have the repos' own names and folders."""
        manifest = self.manifest
        sources: list[tuple] = [(folder, self.models_dir, manifest) for folder in _source_folders("qwen-image")]
        for rel, (repo, revision, remote) in models.QWEN_SOURCES.items():
            if rel not in manifest:
                continue
            for snapshot in models.hf_cache_sources(repo, revision):
                sources.append((snapshot, self.models_dir, {rel: manifest[rel]}, {rel: remote}))
        return sources

    def prepare(self, ctx: Ctx) -> None:
        """Its requirement file (the launcher's packages) must be here."""
        req = self.requirements_file()
        if not req.is_file() and not ctx.dry_run:
            raise InstallError(f"{_root_rel(req)} is missing, so this copy of Spellcast3D is incomplete. Please "
                               f"download Spellcast3D again.")

    def install(self, ctx: Ctx) -> None:
        ctx.out.line("If the download is interrupted or this window is closed, just start Spellcast3D again:")
        ctx.out.line("finished downloads are kept and the rest resumes.")
        ctx.step("Installing its packages", until=0.01)
        ctx.pip(self.id, "-r", str(self.requirements_file()), fail=_download_failed(self.title))
        self._install_runtime(ctx, until=0.07)
        self._fetch_models(ctx, until=0.98)
        ctx.step("Checking the installation", until=1.0)
        self._verify_runtime(ctx)

    def repair(self, ctx: Ctx) -> None:
        runtime = ctx.dry_run or bool(sdcpp.missing())
        self._install_runtime(ctx, until=0.1)
        self._fetch_models(ctx, until=1.0)
        if runtime:
            self._verify_runtime(ctx)

    def _install_runtime(self, ctx: Ctx, until: float) -> None:
        """stable-diffusion.cpp unpacked into code\\venvs\\qwen\\sdcpp, and the CUDA DLLs next to it (sdcpp.py): two
        zips, each from its own release. The engine's goes over an install made with another build of it, whole."""
        folder = paths.QWEN_SDCPP
        if ctx.dry_run:
            sdcpp.dry_run(ctx.out.line, folder)
            return
        try:
            for archive, label in ((sdcpp.SDCPP, "stable-diffusion.cpp"), (sdcpp.CUDART, "NVIDIA's CUDA files")):
                if archive is sdcpp.SDCPP:
                    wanted = sdcpp.runtime_missing(folder)
                else:
                    wanted = sdcpp.cuda_missing(folder) and sdcpp.link_cuda_dlls(folder, ctx.out.line)
                if not wanted:
                    sdcpp.drop_partial(archive)     # (one cut off before PyTorch's copies came: not needed now)
                    continue
                if not sdcpp.on_this_pc(archive):
                    ctx.step(f"Downloading {label} ({archive.size / 1e9:.1f} GB)",
                             until=until if archive is sdcpp.CUDART else until / 2,
                             measure=lambda a=archive: (sdcpp.download_bytes(a), a.size))
                sdcpp.fetch(archive, ctx.out.line)
                ctx.out.line(f"Unpacking {label} into {_root_rel(folder)} ...")
                if archive is sdcpp.SDCPP:
                    sdcpp.unpack_engine(folder, ctx.out.line)
                else:
                    sdcpp.unpack(archive, folder, wanted)
        except models.ChecksumError as e:
            raise InstallError(f"A file of {self.title}'s runtime was downloaded in full, but it is not the expected "
                               f"file ({e}). Start Spellcast3D again to download it again; if that keeps happening, a "
                               f"proxy or an antivirus may be changing it. {LOG_HINT}") from e
        except OSError as e:
            raise InstallError(f"Downloading or unpacking {self.title}'s runtime (stable-diffusion.cpp) failed ({e}). "
                               f"Check your internet connection and that the drive has enough free space, then start "
                               f"Spellcast3D again: finished downloads are kept. {LOG_HINT}") from e

    def _verify_runtime(self, ctx: Ctx) -> None:
        """The launcher's packages import, and stable-diffusion.cpp loads and sees the NVIDIA GPU."""
        ctx.run([self._python(), "-c", sdcpp.VERIFY_CODE], cwd=paths.QWEN_SDCPP, env=ctx.env(self.id),
                fail=_broken(self.title))


# Install order. Of the image generators (paths.STAGE_TOOLS["images"]) install/main.py keeps only the configured one.
COMPONENTS: tuple[Component, ...] = (Hub(), Imesh(), Qwen(), Forge(), Pixal3D(), Lato2(), Triflow())


# ---- the developers' profilers (spellcast3d-config.ini [profiling], spellcast/profiling/) ---------------------------
# Neither a component nor part of Qwen-Image's install: install/main.py adds their packages after every component of
# a run (install_profilers), also when none needed work, so a switch turned on is served at the next start. Preflight
# never sees them, so a developer's switch can never block Qwen-Image, have it recorded as failed, or take disk that
# preflight promised to a component installed after it.

# The package each switch (config.PROFILING_KEYS) needs in Qwen-Image's venv: torch.profiler's CUPTI records
# stable-diffusion.cpp's CUDA kernels although PyTorch launches none. The other venvs' torch, so pip takes it from its
# cache and its files become hard links to theirs. Switching off leaves the package there.
PROFILER_PINS = {"use_python_profiler": "pyinstrument==5.1.3", "use_torch_profiler": TORCH[0],
                 "diagnostic_enable_sync_hunter": TORCH[0]}
# Free disk (GB) PyTorch takes while it installs (7 GB unpacked, plus pip's temporary copies of its 3.5 GB wheel),
# checked against what the components left.
TORCH_INSTALL_GB = 12.0


def missing_profilers(installed: list[Component], dry_run: bool = False) -> dict[str, str]:
    """{switch: pip pin} of the [profiling] switches on in spellcast3d-config.ini whose package Qwen-Image's venv lacks
    (pip writes a dist-info's RECORD last). Empty unless Qwen-Image is among `installed`, the components of the run
    that it leaves installed (--only / --skip and the configured generator decide: --skip qwen adds nothing), and
    recorded installed with a usable venv (a dry run, which installs nothing, takes `installed`'s word). A few stat
    calls."""
    on = config.profiling()
    if not any(on.values()) or paths.QWEN not in {c.id for c in installed}:
        return {}
    if not dry_run and ((state.load().get(paths.QWEN) or {}).get("status") != state.INSTALLED
                        or not venvs.usable(paths.QWEN)):
        return {}
    site = venvs.site_packages(paths.venv_dir(paths.QWEN))
    return {key: pin for key, pin in PROFILER_PINS.items()
            if on[key] and not (site / f"{pin.replace('==', '-')}.dist-info" / "RECORD").is_file()}


def profilers_work(missing: dict[str, str]) -> str:
    """missing_profilers() in words, for the console and install.py --check."""
    return (f"{config.config_file().name} switches on a profiler for developers ({', '.join(missing)}): it needs "
            f"{' and '.join(sorted(set(missing.values())))} in {Qwen.title}'s Python environment")


def profilers_pending(selected: list[Component]) -> str:
    """profilers_work() of what install_profilers will add at the end of a run of `selected`, "" when nothing (or when
    it cannot tell: install_profilers says why). install/main.py's _run shows Qwen-Image as waiting to be installed
    with it, from its first progress picture or from the end of Qwen-Image's own install on: the components before
    the step can take many minutes, and the hub must start no Qwen-Image meanwhile whose venv then gets PyTorch while
    it runs. Never raises."""
    try:
        missing = missing_profilers(selected)
    except Exception:           # an unreadable file, or a bug: install_profilers says it
        return ""
    return profilers_work(missing) if missing else ""


def install_profilers(ctx: Ctx, installed: list[Component], log_header: str | None = None) -> bool:
    """install/main.py's last step of a run, best effort: the packages missing_profilers(installed) names, into
    Qwen-Image's venv. It records no install state and never raises (Ctrl+C aside): a problem is one line, and the next
    start tries again. PyTorch only with TORCH_INSTALL_GB free. pip may add packages but never change the launcher's
    (requirements-qwen.txt as constraints: a conflict fails instead), so Qwen-Image's install is not touched.
    Meanwhile the progress file shows Qwen-Image as being installed, so the hub starts none on a half-installed
    PyTorch; with Qwen-Image among `installed`, its entry ends done on every path (also with no work: a switch turned
    off while _run showed it waiting for this step, see profilers_pending). log_header: when there is work,
    install.log is opened under it first (a run that had no component to install has not opened it). True when it had
    work (a dry run prints the commands)."""
    try:
        missing = missing_profilers(installed, ctx.dry_run)
    except Exception as e:      # an unreadable file, or a bug: never the run's problem
        ctx.out.line(f"Could not tell whether a profiler needs a package ({e}); the next start looks again.")
        missing = {}
    if not missing:
        if paths.QWEN in {c.id for c in installed}:
            ctx.progress.end(paths.QWEN, progress.DONE)     # (what it already shows unless profilers_pending said so)
        return False
    if log_header is not None:
        try:
            ctx.out.open_log(paths.log_file("install"), log_header)
        except OSError:
            pass                # the console still has its lines
    ctx.out.line()
    ctx.out.line(f"{profilers_work(missing)}.")
    pins = sorted(set(missing.values()))
    ctx.progress.start(paths.QWEN)
    try:
        for done, pin in enumerate(pins, 1):
            try:
                _install_profiler(ctx, pin, until=done / len(pins))
            except Exception as e:      # pip, the disk, or a bug: never Qwen-Image's failure
                ctx.out.line(f"{pin} for the profiler was not installed ({getattr(e, 'reason', e)}); pictures are "
                             f"made without it, and the next start tries again.")
    finally:
        ctx.progress.end(paths.QWEN, progress.DONE)
    return True


def _install_profiler(ctx: Ctx, pin: str, until: float) -> None:
    """One pin of PROFILER_PINS into Qwen-Image's venv; PyTorch from its index, then hard-linked to the other venvs'."""
    torch = pin == TORCH[0]
    if torch:
        free = shutil.disk_usage(_existing(paths.venv_dir(paths.QWEN))).free / 1e9
        if free < TORCH_INSTALL_GB:
            ctx.out.line(f"PyTorch for the profiler is not installed: it takes about {TORCH_INSTALL_GB:.0f} GB of free "
                         f"disk space while it installs, and {free:.1f} GB is free. Free up space and start "
                         f"Spellcast3D again.")
            return
    qwen = next(c for c in COMPONENTS if isinstance(c, Qwen))
    ctx.step(f"Installing {pin} for the profiler (for developers)", until=until)
    ctx.pip(qwen.id, pin, "-c", str(qwen.requirements_file()), *(("--index-url", TORCH_INDEX) if torch else ()),
            fail="pip failed (its lines above say why)")
    if torch:
        share_torch_files(ctx, qwen.id)
