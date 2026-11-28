"""Single source of truth for every path, port and tool id in Spellcast3D.

STDLIB ONLY. The installer imports this module with the bare portable Python,
before any venv or third-party package exists.

Layout (release zip == git repo root):

    spellcast3d/
      run-browser/run-spellcast3d.bat     user double-clicks this
      spellcast3d-config.ini              the artist's choices that decide what is installed (gitignored; made
                                          from code/spellcast/config-template.ini, see spellcast/config.py)
      prompt-library.json                 the artist's prompt library, shared by every project (gitignored;
                                          spellcast/promptlib.py)
      tools/python/                       portable CPython 3.11.9 (NuGet full layout, gitignored)
      tools/git/                          portable MinGit (gitignored)
      tools/*.bat                         environment / launcher / update scripts
      code/spellcast/                     the hub (this package)
      code/forge/                         git submodule: Forge Neo (the image generator the config can pick instead
                                          of Qwen-Image 2.1)
      code/pixal3d/                       git submodule: Pixal3D
      code/instant-meshes/                git submodule: instant-meshes-py
      code/lato2/                         git submodule: LATO.2 (retopology, MIT)
      code/triflow/                       git submodule: TriFlow (retopology, non-commercial ADPNCL licence;
                                          its files stay byte-identical: nothing is edited or added there)
      code/whl/                           prebuilt wheels we ship (instant-meshes, torchsparse, pyfqmr-triflow)
      code/venvs/<tool>/                  one venv per tool (gitignored)
      code/venvs/qwen/sdcpp/              stable-diffusion.cpp, Qwen-Image 2.1's runtime (Spellcast3D's pinned build of
                                          it, unpacked)
      code/venvs/downloads/               the installer's big downloads it keeps, so they are never fetched twice
      code/logs/                          install + tool logs (gitignored)
      code/tmp/                           the installer's temporary files (gitignored)
      code/settings.json                  user overrides of spellcast/defaults.json (gitignored)
      models/checkpoints|vae|text-encoders|loras/   the artist's image models (Forge reads them)
      models/qwen-image/                  Qwen-Image 2.1's AI models (the installer fetches them, pinned)
      models/pixal3d/                     Pixal3D's AI models; code/pixal3d/MODELS is a junction to it
      models/lato2/, models/triflow/      the retopology tools' AI models (TriFlow's three are hard-linked
                                          into code/triflow/checkpoints, which TriFlow's own .gitignore ignores)
      projects/<project>/                 user work (gitignored)
      exports/<project>/<prop>/           finished models saved for the artist, one folder per project
                                          with a folder per prop in it (gitignored)

models/ holds a tracked README.txt and a .gitkeep per folder; everything else in it is gitignored.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]
CODE: Path = ROOT / "code"
TOOLS: Path = ROOT / "tools"
PACKAGE: Path = CODE / "spellcast"

PORTABLE_PYTHON_DIR: Path = TOOLS / "python"
PORTABLE_PYTHON: Path = PORTABLE_PYTHON_DIR / "python.exe"
PORTABLE_GIT_DIR: Path = TOOLS / "git"
PORTABLE_GIT: Path = PORTABLE_GIT_DIR / "cmd" / "git.exe"

VENVS: Path = CODE / "venvs"
LOGS: Path = CODE / "logs"
# TEMP of the installer's child processes (pip, Forge's and Pixal3D's own installers): on the install
# drive, which preflight budgets, not on C: (pip keeps two temporary copies of the 3.5 GB torch wheel).
INSTALL_TMP: Path = CODE / "tmp"
WHL: Path = CODE / "whl"
PROJECTS: Path = ROOT / "projects"
# The models the artist saves from the page (a .glb/.fbx/.obj press, Export all), beside the projects folder:
# exports/<project name>/, flat, so they can be dragged into a game engine from File Explorer.
EXPORTS: Path = ROOT / "exports"
SETTINGS_FILE: Path = CODE / "settings.json"
# What the artist chooses by editing a text file (the image generator, Qwen's model size): spellcast/config.py.
CONFIG_FILE: Path = ROOT / "spellcast3d-config.ini"
PROMPT_LIBRARY: Path = ROOT / "prompt-library.json"
CONFIG_TEMPLATE: Path = PACKAGE / "config-template.ini"
DEFAULTS_FILE: Path = PACKAGE / "defaults.json"
STATIC: Path = PACKAGE / "static"
LAUNCHERS: Path = PACKAGE / "launchers"

# Install bookkeeping, written by the installer, read by the hub.
# {tool_id: {"status": "installed"|"unsupported"|"failed"|"not_installed",
#            "reason": str, "fingerprint": str, "time": iso8601},
#  "pixal3d-calibration-retries": {"card": str, "version": int, "steps": {step: int}}}   (install/components.py)
INSTALL_STATE: Path = VENVS / "install-state.json"
# What a running installer is doing (install.py --progress-file; schema in install/progress.py).
INSTALL_PROGRESS: Path = VENVS / "install-progress.json"
# The hub's word to its installer before Pixal3D's calibration: {"free": bool, "waiting_for": tool names}, true once
# none of the hub's own tools holds video memory (procman.free_card_for_calibration writes it, components.py waits).
CALIBRATION_CARD: Path = VENVS / "calibration-card.json"
# Big archives the installer downloads (stable-diffusion.cpp's release zip, ...), kept so that a reinstall or a
# repair never downloads them again; inside code/venvs, so gitignored and on the install drive.
DOWNLOADS: Path = VENVS / "downloads"

# Submodules.
FORGE_DIR: Path = CODE / "forge"
PIXAL3D_DIR: Path = CODE / "pixal3d"
IMESH_DIR: Path = CODE / "instant-meshes"
LATO2_DIR: Path = CODE / "lato2"
TRIFLOW_DIR: Path = CODE / "triflow"
# Hi3D's code (a submodule once .gitmodules names it): the 3D stage's second geometry, which Pixal3D's launcher
# imports from here (SPELLCAST_HI3D_CODE names a checkout elsewhere; launchers/pixal3d_server.py hi3d_paths).
HI3D_DIR: Path = CODE / "hi3d"
HI3D_CODE_ENV = "SPELLCAST_HI3D_CODE"


def hi3d_code_here() -> bool:
    """Whether Hi3D's code is on this PC: code/hi3d, or the checkout SPELLCAST_HI3D_CODE names (as Pixal3D's launcher
    reads it). Without it no prop can be made with Hi3D, whatever is in models/hi3d."""
    override = os.environ.get(HI3D_CODE_ENV, "").strip().strip('"')
    return ((Path(override) if override else HI3D_DIR) / "sparc3d" / "__init__.py").is_file()

# The artist's models folder. Forge reads the first four through its folder flags (adapters/forge.py),
# besides its own code/forge/models and the extra folders of the settings (forge.ckpt_dirs).
MODELS: Path = ROOT / "models"
MODELS_CHECKPOINTS: Path = MODELS / "checkpoints"
MODELS_VAE: Path = MODELS / "vae"
MODELS_TEXT_ENCODERS: Path = MODELS / "text-encoders"
MODELS_LORAS: Path = MODELS / "loras"
# Pixal3D's weights (~27 GB). Pixal3D itself uses code/pixal3d/MODELS (its install.py and app.py; HF_HOME and
# TORCH_HOME point inside it), which the installer makes a directory junction to this folder (install/models.py).
MODELS_PIXAL3D: Path = MODELS / "pixal3d"
PIXAL3D_MODELS_LINK: Path = PIXAL3D_DIR / "MODELS"
# LATO.2's weights (~4.8 GB with DINOv2); its launcher passes their paths explicitly.
MODELS_LATO2: Path = MODELS / "lato2"
# TriFlow's weights (1.3 GB). TriFlow loads them from checkpoints/ relative to its working folder (code/triflow);
# the installer hard-links them there (install/models.py).
MODELS_TRIFLOW: Path = MODELS / "triflow"
TRIFLOW_CHECKPOINTS: Path = TRIFLOW_DIR / "checkpoints"
# Qwen-Image 2.1's weights (~11 GB, flat: the Turbo's denoiser of the configured size, the full model's once the
# artist asks for it, the VAE, the Qwen3-VL text encoder and its vision part); its launcher gets their paths
# (install/models.py names them).
MODELS_QWEN: Path = MODELS / "qwen-image"
# Our albedo add-on for Qwen-Image 2.1 (1.2 GB: LoRA, decoder, fixed prompt; github.com/IgorAherne/marigold-v2),
# installed with Qwen-Image (install/models.py ALBEDO_FILES): Omni-Texture turns each painted view into its albedo
# before projecting it.
MODELS_ALBEDO: Path = MODELS_QWEN / "albedo"
# FLUX.3 Image Klein's weights (~6.8 GB at the default size, flat: its denoiser of the configured size, FLUX.2's VAE and
# the Qwen3-VL-4B text encoder): an optional third model of the Images stage, downloaded only when the artist asks for
# it on the Images page (spellcast/model_downloads.py); Qwen-Image's launcher serves it too (install/models.py).
MODELS_FLUX3: Path = MODELS / "flux3-klein"
# Hi3D v1.0's networks (11.4 GB: its own, DINOv2, CLIP and TRELLIS.2's shape encoder, each in its folder), installed
# with Pixal3D (install/models.py HI3D_FILES): the 3D stage's second geometry, which Pixal3D's launcher finds beside
# models/pixal3d.
MODELS_HI3D: Path = MODELS / "hi3d"

# Tool ids. "hub" is Spellcast3D itself; the others are the wrapped tools, each in its own venv and process.
HUB = "hub"
FORGE = "forge"
PIXAL3D = "pixal3d"
IMESH = "imesh"
LATO2 = "lato2"
TRIFLOW = "triflow"
QWEN = "qwen"
TOOL_IDS: tuple[str, ...] = (FORGE, PIXAL3D, IMESH, LATO2, TRIFLOW, QWEN)
ALL_VENV_IDS: tuple[str, ...] = (HUB, FORGE, PIXAL3D, IMESH, LATO2, TRIFLOW, QWEN)

# The tools that are a git submodule (Qwen-Image 2.1 is not: its runtime is a pinned release zip).
TOOL_DIRS: dict[str, Path] = {FORGE: FORGE_DIR, PIXAL3D: PIXAL3D_DIR, IMESH: IMESH_DIR, LATO2: LATO2_DIR,
                              TRIFLOW: TRIFLOW_DIR}
# stable-diffusion.cpp's files (stable-diffusion.dll, ggml-cuda.dll, ...) and NVIDIA's CUDA DLLs next to them.
QWEN_SDCPP: Path = VENVS / QWEN / "sdcpp"
# (The ports are set in spellcast/defaults.json "ports", the only port table.)

# Pipeline stages, in order, each with its DEFAULT tool. A stage with tool None is a placeholder (coming soon).
# One tool can run several stages: Instant Meshes does a retopo and, through its bake worker, the projection;
# Qwen-Image paints the Omni-Texture (a new colour texture for the props the artist chooses; optional).
# Retopo: TriFlow until the artist picks another method for the prop (projects.DEFAULT_RETOPO_METHOD).
# The Images stage runs on the image generator spellcast3d-config.ini picks (config.image_backend()): only that
# one is installed and registered in the hub, so with Forge Neo there is no Omni-Texture.
# A stored stage is its id, never its place here (projects.py maps the old rows' numbers itself).
STAGES: tuple[tuple[str, str | None], ...] = (
    ("images", QWEN),
    ("mesh", PIXAL3D),
    ("omnitexture", QWEN),
    ("retopo", TRIFLOW),
    ("projection", IMESH),
    ("rig", None),
)
# Every tool that can run a stage (the default first): the artist picks the retopology method per prop
# (retopo.settings.method), and the image generator in spellcast3d-config.ini. The source of truth for "which
# tools can run a stage".
STAGE_TOOLS: dict[str, tuple[str, ...]] = {
    "images": (QWEN, FORGE),
    "mesh": (PIXAL3D,),
    "omnitexture": (QWEN,),
    "retopo": (TRIFLOW, LATO2, IMESH),
    "projection": (IMESH,),
    "rig": (),
}


def venv_dir(venv_id: str) -> Path:
    return VENVS / venv_id


def venv_python(venv_id: str) -> Path:
    # Windows' venv layout; elsewhere (a Linux test pod: code/devtools/pod) the POSIX one
    return venv_dir(venv_id) / "Scripts" / "python.exe" if os.name == "nt" else venv_dir(venv_id) / "bin" / "python"


def log_file(name: str) -> Path:
    return LOGS / f"{name}.log"


def user_path(value: object) -> Path:
    """A folder the user typed into the settings: relative paths are relative to the Spellcast3D folder."""
    p = Path(str(value).strip()).expanduser()
    return p if p.is_absolute() else ROOT / p


def write_text_atomic(path: Path, text: str) -> None:
    """Write via a temp file + os.replace, so a crash never leaves a half-written file.

    The temp name is unique per process and thread (the hub writes from several threads), and
    the rename is retried: an antivirus, an indexer, an editor or the hub reading the file at that
    moment can briefly lock the target on Windows. The temp file never stays behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.1)
    finally:
        tmp.unlink(missing_ok=True)


# Variables that must never leak from the user's shell (or from one tool) into a
# tool process: they redirect caches, change pip/torch behaviour, or inject
# arguments (Forge appends COMMANDLINE_ARGS to argv; Pixal3D honours ATTN_BACKEND).
_SCRUB_EXACT = {
    "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV",
    "CONDA_PREFIX", "CONDA_PYTHON_EXE", "CONDA_DEFAULT_ENV", "CONDA_SHLVL", "CONDAPATH",
    "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE",
    "TORCH_HOME", "XDG_CACHE_HOME",
    "COMMANDLINE_ARGS", "TORCH_COMMAND", "TORCH_INDEX_URL", "INDEX_URL", "REQS_FILE",
    "XFORMERS_PACKAGE", "GRADIO_PACKAGE", "PYTORCH_VERSION",
    "ATTN_BACKEND", "SPARSE_BACKEND", "SPARSE_ATTN_BACKEND", "SPARSE_CONV_BACKEND", "LOW_VRAM",
    "GRADIO_SERVER_NAME", "GRADIO_SERVER_PORT", "GRADIO_ROOT_PATH", "GRADIO_SHARE",
    "CUDA_VISIBLE_DEVICES_OVERRIDE", "PIP_REQUIRE_VIRTUALENV", "PIP_TARGET", "PIP_PREFIX", "PIP_USER",
}
_SCRUB_PREFIXES = ("CONDA_",)


def clean_env(venv_id: str | None = None, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for a child process (installer step or tool server).

    - scrubs variables that would redirect caches or inject arguments
    - puts the venv's Scripts dir (if any), portable git and portable python first on PATH
    - sets the Python hygiene flags every tool expects
    """
    env = {k: v for k, v in os.environ.items()
           if k.upper() not in _SCRUB_EXACT and not k.upper().startswith(_SCRUB_PREFIXES)}

    path_parts: list[str] = []
    if venv_id is not None:   # the venv's programs (ninja, ...): Scripts on Windows, bin on a Linux test pod
        path_parts.append(str(venv_dir(venv_id) / ("Scripts" if os.name == "nt" else "bin")))
    if PORTABLE_GIT.exists():
        path_parts.append(str(PORTABLE_GIT.parent))
    if PORTABLE_PYTHON.exists():
        path_parts.append(str(PORTABLE_PYTHON_DIR))
    old_path = env.get("PATH") or env.get("Path") or ""
    for k in [k for k in env if k.upper() == "PATH"]:
        del env[k]
    env["PATH"] = os.pathsep.join(path_parts + ([old_path] if old_path else []))

    env.update({
        "PYTHONNOUSERSITE": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        "GRADIO_ANALYTICS_ENABLED": "False",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        # gradio probes its own local URL at startup; a system proxy must not intercept it
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
        # The Intel Fortran runtime that some numpy / scipy builds load answers the hub's CTRL_BREAK
        # (a normal stop) with a scary "forrtl: error (200): program aborting" + stack dump in the log.
        "FOR_DISABLE_CONSOLE_CTRL_HANDLER": "1",
    })
    if PORTABLE_GIT.exists():
        env["GIT_PYTHON_GIT_EXECUTABLE"] = str(PORTABLE_GIT)
        ca = PORTABLE_GIT_DIR / "mingw64" / "etc" / "ssl" / "certs" / "ca-bundle.crt"
        if ca.exists():
            env.setdefault("GIT_SSL_CAINFO", str(ca))
    if venv_id is not None:
        env["VIRTUAL_ENV"] = str(venv_dir(venv_id))
    if extra:
        env.update(extra)
    return env
