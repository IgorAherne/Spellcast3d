"""The models folder of Spellcast3D (spellcast3d/models) and the tools' AI models in it. STDLIB ONLY.

    models/checkpoints, vae, text-encoders, loras   image models; Forge Neo reads them (adapters/forge.py)
    models/qwen-image                               Qwen-Image 2.1's AI models (~12 GB with the default size: the
                                                    Turbo and our albedo add-on in albedo/; + 4.2 GB for the full
                                                    model, on the artist's request)
    models/flux3-klein                              FLUX.3 Image Klein's (6.8 GB with the default size), only on the
                                                    artist's request (made by its download, not at the start)
    models/pixal3d                                  Pixal3D's AI models (~27 GB, + 14 GB of faster-loading copies)
    models/hi3d                                     Hi3D's (11.4 GB), the 3D stage's second Shape, only on the
                                                    artist's request (made by its download, not at the start)
    models/lato2                                    LATO.2's AI models (4.8 GB with DINOv2)
    models/triflow                                  TriFlow's AI models (1.3 GB); code/triflow/checkpoints links to them

Qwen-Image 2.1's and the retopology tools' models are fetched by the installer itself (pinned revisions, SHA-256
checked, resumed after an interruption). Copies already on this PC are hard-linked instead of downloaded: from the
artist's HuggingFace caches, from folders given with install.py --models-from (or SPELLCAST_MODELS_FROM), and for
TriFlow from code/triflow/checkpoints. A hard link needs the same drive and takes no disk space; GBs are never
copied. A file that got into those folders some other way (copied in by hand) is checked once against its pin too
(SHA-256; DINOv2's code: tree_digest) before it is used; what was checked is remembered in
code/venvs/models-verified.json (by size and time), so it is not hashed again at every start.

Pixal3D keeps using its own folder code/pixal3d/MODELS: its install.py downloads there, its app loads from
there (HF_HOME and TORCH_HOME point inside it). The installer makes that folder a directory junction to
models/pixal3d - junctions need no admin rights, unlike symbolic links - so the artist can drop weights they
already have into models/pixal3d, and they stay there when the tool is updated or reinstalled.

An older install's real code/pixal3d/MODELS folder is moved into models/pixal3d first: renames on the same
drive (a second, never a copy of 27 GB), and nothing is ever deleted. Where models/pixal3d already has a file,
it keeps its own, and the older copies are set aside in models/pixal3d-old-<time> for the artist to delete.
"""

from __future__ import annotations

import ctypes
import datetime
import hashlib
import http.client
import json
import os
import shutil
import stat
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Sequence

from .. import paths

FOLDERS: tuple[Path, ...] = (paths.MODELS_CHECKPOINTS, paths.MODELS_VAE, paths.MODELS_TEXT_ENCODERS,
                             paths.MODELS_LORAS, paths.MODELS_QWEN, paths.MODELS_PIXAL3D, paths.MODELS_HI3D,
                             paths.MODELS_LATO2, paths.MODELS_TRIFLOW)
# What Pixal3D's MODELS folder holds at its top (install.py's zips and HF_HOME / TORCH_HOME).
_PIXAL3D_LANDMARKS = ("hub", "dinov3", "RMBG-2.0", "torch")
# Pixal3D's optional multi-view models in TencentARC/Pixal3D at commit b0cb2e1 and their sizes (pipeline_mv.json
# names them; the three decoders and the background remover are the single-view ones): Pixal3D install.py's
# PIXAL3D_MULTI_VIEW_PATTERNS. The installer leaves them out; the hub downloads them on request (multiview.py).
PIXAL3D_MULTI_VIEW_FILES: dict[str, int] = {
    "pipeline_mv.json": 6_036,
    "ckpts/ss_flow_img_dit_1_3B_64_bf16_mv.json": 503,
    "ckpts/ss_flow_img_dit_1_3B_64_bf16_mv.safetensors": 5_359_822_584,
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16_mv.json": 535,
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16_mv.safetensors": 5_546_764_048,
    "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16_mv.json": 535,
    "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16_mv.safetensors": 5_546_764_048,
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv.json": 535,
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv.safetensors": 5_546_960_656,
}
# Copies of Pixal3D's four flow models in the dtypes it keeps them in, which its staged engine reads in place of the
# fp32 originals (half the bytes: 11.1 GB instead of 22 GB for the four). `python -m pixal3d.staged.weights
# <snapshot>` in the Pixal3D venv makes them beside the originals in the snapshot folder,
# ckpts/<checkpoint>.model-dtypes.safetensors, after the installer's download (install/components.py) and after the
# multi-view download (multiview.py); it skips copies that are up to date. A copy that is missing or stale only means
# that Pixal3D reads the original. The two detail flows (hr, tex) also get pre-quantised int8 copies,
# ckpts/<checkpoint>.int8.model-dtypes.safetensors. Bytes: the single-view set at b0cb2e1 on an sm_80+ card, the
# largest: the model-dtypes copies (2.72 + 3 x 2.81 GB) and the int8 copies (2 x 1.63 GB); an RTX 20xx makes no int8
# copies, a GTX 10xx only the two int8 ones (2 x 2.01 GB). The multi-view set takes as much again.
PIXAL3D_COPIES_MODULE = "pixal3d.staged.weights"
PIXAL3D_COPY_SUFFIX = ".model-dtypes.safetensors"     # the fork's weights.COPY_SUFFIX (the int8 copies end so too)
PIXAL3D_COPIES_BYTES = 11_142_000_000 + 3_264_892_744
_FILE_SUPPORTS_REPARSE_POINTS = 0x80

# ---- the retopology tools' AI models: pinned, with their sizes and SHA-256 -----------------------------------
# A manifest maps a file (relative to the tool's models folder) to (bytes, sha256).
Manifest = dict[str, tuple[int, str]]

# LATO.2 (MIT): huggingface.co/0x4c48/LATO.2 at this commit. The six checkpoints its end-to-end inference loads
# (vdf_encoder.pt is for training only). The SHA-256 are HuggingFace's LFS ids at that commit.
LATO2_HF_REPO = "0x4c48/LATO.2"
LATO2_HF_REVISION = "a91090e8077b9318ab87ac08fd9eb905903d4515"
LATO2_FILES: Manifest = {
    "vflow.pt": (649_153_594, "a5c660601af9ce9909feb5157cfd456d81b0498835d7ba4129a7d4267a8927d3"),
    "vvae.pt": (1_261_420_852, "b1192e3e6301dee5ee3dac37d411f82141ee1a8764707f0ea5b7746052342d44"),
    "offset_head.pt": (4_229_424, "b9ab60321ffa1026813b4cb79eba7a9d224ccd77e01e6b2c84fd29cad301c9df"),
    "tflow.pt": (882_821_662, "5ac0033544b813ec046f00d5f12d3b488a91a98c1bfef0560948fb4c6b52c6ad"),
    "tvae.pt": (713_397_214, "b109bcf2bc42abea89a3563e3fefa9509f9cd523501d5c62d78108380a3db17c"),
    "voxel_encoder.pt": (61_373_424, "c304e36651f2a8c63d5b30d0716a4a8ab4ef505fe39311d060e04e5ab253d4ce"),
}
# LATO.2 encodes its conditioning render with DINOv2 ViT-L/14 with registers (Apache-2.0), loaded through
# torch.hub from models/lato2/dinov2: the weights from Meta's server (Last-Modified 2023-10-27) ...
DINOV2_CHECKPOINT = "dinov2/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
DINOV2_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
DINOV2_FILES: Manifest = {
    DINOV2_CHECKPOINT: (1_217_607_321, "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"),
}
# ... and the model's code: facebookresearch/dinov2 at this commit, in the folder torch.hub would unpack it into
# (LATO.2 loads it with source="local"). DINOV2_CODE_DIGEST is tree_digest() of that commit's 203 files.
DINOV2_CODE = "dinov2/facebookresearch_dinov2_main"
DINOV2_CODE_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_CODE_URL = f"https://github.com/facebookresearch/dinov2/archive/{DINOV2_CODE_COMMIT}.zip"
DINOV2_CODE_DIGEST = "52355da3fb055777d31aac17448d8e76e76d60269b1c29681487e440df00b39e"
DINOV2_CODE_BYTES = 3_956_490
DINOV2_CODE_LANDMARK = "hubconf.py"

# TriFlow (ADPNCL 1.0, non-commercial): huggingface.co/lihcxr/TriFlow at this commit. TriFlow loads them from
# checkpoints/ in its working folder (code/triflow); see link_triflow_checkpoints.
TRIFLOW_HF_REPO = "lihcxr/TriFlow"
TRIFLOW_HF_REVISION = "e3403f35d0566a0487b9c8f3816bc6cdee814dca"
TRIFLOW_FILES: Manifest = {
    "flow_model.safetensors": (754_943_468, "f724b037546d1930e39c3f20edb16f5af3ff3f24f3969ade5daf74ffbdab624b"),
    "sdf_vae.safetensors": (151_508_820, "a0ee5844b54d179b6724d332fa67d8fdc5c3a9904e103dc0b3a93b073311d94d"),
    "nvv_vae.safetensors": (412_442_376, "46c941c826c83f1e9f00d557dd655e41bc407693db33b94b422ae10313fe829d"),
}
# Qwen-Image 2.1 (the Images stage's default generator), flat in models/qwen-image under these names: a denoiser of
# the size spellcast3d-config.ini picks (config.QUALITIES), its VAE (both under Qwen's research licence), and the
# Qwen3-VL-8B-Instruct text encoder with its vision part, which reference pictures need (Apache-2.0: from unsloth's
# Qwen3-VL repo, not from Qwen-Image's own). The SHA-256 are HuggingFace's LFS ids at these commits.
# Two denoisers of the same architecture share the VAE, the text encoder and the vision part (QWEN_MODELS):
# - "turbo", Qwen-Image-2.1-Turbo, the default for every card: its own weights,
#   sampled with a FIXED 8-step schedule (the launcher's TURBO_SIGMAS), no guidance. The installer downloads it
#   (qwen_manifest).
# - "full", Qwen-Image 2.1 itself, 25 steps by default: never downloaded by the installer any more, only when the
#   artist asks for it on the Images page (spellcast/model_downloads.py, qwen_full_manifest). One an older install
#   has already downloaded stays and is used.
# Both come in the same five sizes (config.QUALITIES, the smallest first), and so does FLUX.3 Image Klein below:
# `quality` names the size of all three, and a bigger size already in the folder is used in its place (size_on_disk).
QWEN_MODELS: tuple[str, ...] = ("turbo", "full")
QWEN_DENOISER = "qwen-image-2.1-{quality}.gguf"                 # the full model
QWEN_GGUF_REPO = "unsloth/Qwen-Image-2.1-GGUF"
QWEN_GGUF_REVISION = "2c31ccd392b367a6637841a143813320a02dff55"
QWEN_DENOISERS: Manifest = {        # quality -> the full model's denoiser
    "Q3_K_M": (3_168_290_528, "f688e47c32c033aa8d059c6b7e27d668b2cef0f1dd3d667f97f544954d541238"),
    "Q4_K_M": (4_199_565_024, "631d532e7ca71e8d90a87c71d3699761a812039d22e3370e87498d87754660fe"),
    "Q5_K_M": (5_390_223_072, "4b53321654dd3bf0aa8dd6bb821fdcb1c9053ca735070e359a23cbb9f97268ce"),
    "Q6_K": (6_271_551_200, "2ead1ee665e8ebc5fec517d287a167f5c6a7e985acbf3b1b93143437a04616ac"),
    "Q8_0": (7_640_860_384, "c0ed4b2ffd56cbe9c3df1e4a4098045256484ebe94ba5a7e4338d35ec046baa5"),
}
QWEN_TURBO_DENOISER = "qwen-image-2.1-turbo-{quality}.gguf"     # the Turbo
QWEN_TURBO_GGUF_REPO = "unsloth/Qwen-Image-2.1-Turbo-GGUF"
QWEN_TURBO_GGUF_REVISION = "5d1f9f759ea877ae556cda88dbec958b62ad3809"
QWEN_TURBO_DENOISERS: Manifest = {  # quality -> the Turbo's denoiser
    "Q3_K_M": (3_168_290_624, "2fd16e6590c8153980f0de4d8965007d8cf3cc497240e74c03db7bb05e799467"),
    "Q4_K_M": (4_199_565_120, "8940e8b522af886e0c0f5afddf720e82dd223cb9a43580b5f61c4d740452ac20"),
    "Q5_K_M": (5_390_223_168, "37c51ff446f4db2e131a9d12c003f0c0e824c8b8ba4dcd8c6cd5753f13457f5f"),
    "Q6_K": (6_271_551_296, "f4560fb47ad863ffed7e32737d3c42988c44bd0c459606777d7ffe2e895d0a70"),
    "Q8_0": (7_640_860_480, "399d2ddc3636f36456e2f5bfa57187da4e38721bc046430317f42a9eab5b1712"),
}
QWEN_VAE = "qwen_image_2.1_vae_bf16.safetensors"
# The text encoder, Qwen3-VL-8B-Instruct (unsloth's quants), in its own sizes, smallest first: spellcast3d-config.ini
# text_encoder picks one (config.TEXT_ENCODERS), and the rule of the denoisers' sizes holds for it too (size_on_disk).
# It reads the words and the example pictures once a request; the denoiser draws every step with what it made.
QWEN_TEXT_ENCODER_FILE = "Qwen3-VL-8B-Instruct-{size}.gguf"
QWEN_TEXT_ENCODERS: Manifest = {    # size -> (bytes, sha256)
    "UD-Q4_K_XL": (5_148_699_488, "e3d1a6e87c5cb31e054f2c3bc0dd82ffde052f613de5eef3665b7bd33c9b703e"),
    "UD-Q5_K_XL": (5_884_767_072, "d138d047fe5736cdc678490d6ba16d2e42f690a6d648b5c8fbdac0e17789fac6"),
    "UD-Q6_K_XL": (7_490_550_624, "dc3cc983735542500f40c32c1a9dacdeff198a440fac541056dca6b550389130"),
    "Q8_0": (8_709_520_224, "cb8616bf6ed228982d9e47d7b72b42195342efa26044b0ee1873e61d9e78d3d7"),
    "UD-Q8_K_XL": (10_824_039_264, "6d15b2376bea85649a9f9d63277a80519a261af2e34ee9ef6b605d8a0bc365b6"),
}
DEFAULT_TEXT_ENCODER = "UD-Q4_K_XL"
QWEN_TEXT_ENCODER = QWEN_TEXT_ENCODER_FILE.format(size=DEFAULT_TEXT_ENCODER)
QWEN_VISION = "Qwen3-VL-8B-Instruct-mmproj-F16.gguf"
# Our albedo add-on for Qwen-Image 2.1 (Qwen's research licence: its LICENSE and NOTICE come with it), in
# models/qwen-image/albedo (paths.MODELS_ALBEDO): Omni-Texture takes the lighting off each painted view with it.
# ALBEDO_FILES are what the launcher hands the engine; the SHA-256 are HuggingFace's LFS ids at ALBEDO_HF_REVISION (the
# three small text files are plain git files there: their SHA-256 of the files downloaded from that commit).
ALBEDO_HF_REPO = "IgorAherne/spellcast3d-marigold-albedo-qwen-image-2.1"
ALBEDO_HF_REVISION = "1867b64326fb628ac8bf6d60801169d9ee750560"
ALBEDO_FILES = {"lora": "albedo_lora.safetensors", "condition": "albedo_condition.safetensors",
                "vae": "albedo_vae.safetensors"}
_ALBEDO = paths.MODELS_ALBEDO.relative_to(paths.MODELS_QWEN).as_posix()
ALBEDO_PINS: Manifest = {           # its files' paths in models/qwen-image
    f"{_ALBEDO}/albedo_lora.safetensors": (683_777_548, "720ddc255a165c7f483115a4e97dd8511a7df71c3133bca2e3d8571b3ce1378a"),
    f"{_ALBEDO}/albedo_vae.safetensors": (518_112_104, "eaacdc52f354ad28a8cdc14205c61b2e465243c46edd7221996e8324aebfb206"),
    f"{_ALBEDO}/albedo_condition.safetensors": (311_600, "eed5d6a7477d628c13cff36394707c44097cb4b8cd05ce3cd00c9ca4c001951e"),
    f"{_ALBEDO}/LICENSE": (7_831, "8dc973f024ff95966bea25866efa443fd16776dcb1001e681e3d467ea572b28d"),
    f"{_ALBEDO}/NOTICE": (475, "40dc5c1dd15f9322abd06cf68737b54a608295818e4d71fe874b8da3cfbdff7a"),
    f"{_ALBEDO}/arb_objaverse_credits.csv": (878_671, "f7df4f458fbd13e6de4b69c33ea2df4733c0707b8192724e231d83bc3d36517c"),
}
QWEN_SHARED_FILES: Manifest = {     # needed with every size (the text encoder's default size: qwen_shared_files)
    QWEN_VAE: (675_508_656, "71879ffd5321e6d10c3c87513e2b474b1252efa7f3dec2969214a9bf06a6dd5c"),
    QWEN_TEXT_ENCODER: QWEN_TEXT_ENCODERS[DEFAULT_TEXT_ENCODER],
    QWEN_VISION: (1_159_030_336, "d406d03ebabefdef86a2c86bf0c1b65f9e046f7a81c218f25de4931b46a07fc4"),
    **ALBEDO_PINS,
}
_QWEN_VL = ("unsloth/Qwen3-VL-8B-Instruct-GGUF", "b93a7ee713758252c555be4210c00540df954dc2")
# Where each file comes from: its name here -> (HuggingFace repo, commit, its path in the repo).
QWEN_SOURCES: dict[str, tuple[str, str, str]] = {
    **{QWEN_TURBO_DENOISER.format(quality=q): (QWEN_TURBO_GGUF_REPO, QWEN_TURBO_GGUF_REVISION,
                                               QWEN_TURBO_DENOISER.format(quality=q)) for q in QWEN_TURBO_DENOISERS},
    **{QWEN_DENOISER.format(quality=q): (QWEN_GGUF_REPO, QWEN_GGUF_REVISION, QWEN_DENOISER.format(quality=q))
       for q in QWEN_DENOISERS},
    QWEN_VAE: ("unsloth/Qwen-Image-2.1-FP8", "d67caebb412f98c968f9be41b3a8cea80b3e039a", f"vae/{QWEN_VAE}"),
    **{QWEN_TEXT_ENCODER_FILE.format(size=size): (*_QWEN_VL, QWEN_TEXT_ENCODER_FILE.format(size=size))
       for size in QWEN_TEXT_ENCODERS},
    QWEN_VISION: (*_QWEN_VL, "mmproj-F16.gguf"),
    **{rel: (ALBEDO_HF_REPO, ALBEDO_HF_REVISION, rel.removeprefix(f"{_ALBEDO}/")) for rel in ALBEDO_PINS},
}
# Every pinned file of Qwen-Image 2.1, both models in every size: {its name here: (bytes, sha256)}
# (prepare_release.py check compares them all with HuggingFace).
QWEN_PINS: Manifest = {
    **{QWEN_TURBO_DENOISER.format(quality=q): pin for q, pin in QWEN_TURBO_DENOISERS.items()},
    **{QWEN_DENOISER.format(quality=q): pin for q, pin in QWEN_DENOISERS.items()},
    **QWEN_SHARED_FILES,
    **{QWEN_TEXT_ENCODER_FILE.format(size=size): pin for size, pin in QWEN_TEXT_ENCODERS.items()},
}


def qwen_denoiser(model: str, quality: str) -> tuple[str, tuple[int, str]]:
    """(the file name, (bytes, sha256)) of the denoiser of `model` ("turbo" | "full") in size `quality`."""
    if model == "turbo":
        return QWEN_TURBO_DENOISER.format(quality=quality), QWEN_TURBO_DENOISERS[quality]
    if model == "full":
        return QWEN_DENOISER.format(quality=quality), QWEN_DENOISERS[quality]
    raise ValueError(f"no Qwen-Image model {model!r} (one of {', '.join(QWEN_MODELS)})")


def qwen_text_encoder(size: str) -> tuple[str, tuple[int, str]]:
    """(the file name, (bytes, sha256)) of the text encoder of size `size` (QWEN_TEXT_ENCODERS)."""
    return QWEN_TEXT_ENCODER_FILE.format(size=size), QWEN_TEXT_ENCODERS[size]


def text_encoder_in_use(size: str, folder: Path | None = None) -> str:
    """The text encoder's size the launcher reads (size_on_disk in models/qwen-image or `folder`), else `size` (the
    installer downloads it)."""
    folder = folder or paths.MODELS_QWEN
    return size_on_disk(size, tuple(QWEN_TEXT_ENCODERS),
                        lambda s: not missing_files(folder, dict([qwen_text_encoder(s)]))) or size


def qwen_shared_files(text_encoder: str = DEFAULT_TEXT_ENCODER, folder: Path | None = None) -> Manifest:
    """QWEN_SHARED_FILES with the text encoder of the size in use (text_encoder_in_use) in the default one's place."""
    size = text_encoder_in_use(text_encoder, folder)
    if size == DEFAULT_TEXT_ENCODER:
        return dict(QWEN_SHARED_FILES)
    name, pin = qwen_text_encoder(size)
    return {**{rel: p for rel, p in QWEN_SHARED_FILES.items() if rel != QWEN_TEXT_ENCODER}, name: pin}


def size_on_disk(quality: str, sizes: Sequence[str], present: Callable[[str], bool]) -> str | None:
    """The size a model runs with, among those of its denoisers on disk (present(size); `sizes`: the model's, the
    smallest first): `quality`, the configured one, when it is there; else the largest bigger one that is (a bigger
    file the artist put there is the better model, and spares the download); else None, and `quality` is downloaded
    (a smaller size does not stand in for it)."""
    if present(quality):
        return quality
    bigger = sizes[sizes.index(quality) + 1:] if quality in sizes else ()
    return next((size for size in reversed(bigger) if present(size)), None)


def qwen_size_on_disk(model: str, quality: str, folder: Path | None = None) -> str | None:
    """size_on_disk of `model` ("turbo" | "full") in models/qwen-image (or `folder`): a denoiser counts when it is
    there whole, at its pinned size (an unfinished download is <name>.part). A few stat calls, no hashing."""
    folder = folder or paths.MODELS_QWEN
    sizes = tuple(QWEN_TURBO_DENOISERS if model == "turbo" else QWEN_DENOISERS)
    return size_on_disk(quality, sizes, lambda size: not missing_files(folder, dict([qwen_denoiser(model, size)])))


def qwen_manifest(quality: str, folder: Path | None = None, text_encoder: str = DEFAULT_TEXT_ENCODER) -> Manifest:
    """The files the installer wants for Qwen-Image 2.1 with the size `quality` (config.qwen_quality()): the Turbo's
    denoiser, the default model - of the size in use (qwen_size_on_disk; `quality` when none is there: the download) -
    and the files both models share, the text encoder of the size in use (`text_encoder`: config.qwen_text_encoder()).
    The full model's denoiser is not among them (qwen_full_manifest: downloaded only on the artist's request)."""
    name, pin = qwen_denoiser("turbo", qwen_size_on_disk("turbo", quality, folder) or quality)
    return {name: pin, **qwen_shared_files(text_encoder, folder)}


def qwen_full_manifest(quality: str, folder: Path | None = None) -> Manifest:
    """The full model's denoiser - of the size in use (qwen_size_on_disk), else of size `quality`: what the Images page's
    download fetches (spellcast/model_downloads.py)."""
    name, pin = qwen_denoiser("full", qwen_size_on_disk("full", quality, folder) or quality)
    return {name: pin}


def qwen_model_on_disk(model: str, quality: str, folder: Path | None = None) -> bool:
    """Whether a denoiser of `model` that stands for size `quality` (qwen_size_on_disk) is in models/qwen-image (or
    `folder`), whole."""
    return qwen_size_on_disk(model, quality, folder) is not None


def qwen_model_files(quality: str, folder: Path | None = None, text_encoder: str = DEFAULT_TEXT_ENCODER
                     ) -> dict[str, Path]:
    """{"turbo", "full" (the two denoisers: diffusion_model_path), "vae", "text_encoder", "vision": its file} for
    stable-diffusion.cpp (vae_path, llm_path, llm_vision_path), in models/qwen-image unless `folder` is given. Each
    denoiser and the text encoder of the size in use (qwen_size_on_disk, text_encoder_in_use; the configured size
    when none is there), every path whether the file is there or not."""
    folder = folder or paths.MODELS_QWEN
    return {**{model: folder / qwen_denoiser(model, qwen_size_on_disk(model, quality, folder) or quality)[0]
               for model in QWEN_MODELS},
            "vae": folder / QWEN_VAE, "vision": folder / QWEN_VISION,
            "text_encoder": folder / qwen_text_encoder(text_encoder_in_use(text_encoder, folder))[0]}


def qwen_files_words(quality: str, text_encoder: str = DEFAULT_TEXT_ENCODER, folder: Path | None = None) -> str:
    """The model files Qwen-Image 2.1 reads, for the console: "Turbo qwen-image-2.1-turbo-Q5_K_M.gguf (Q4_K_M,
    the configured size, is not there: the biggest bigger one), full model ... (not downloaded), text encoder ..."."""
    folder = folder or paths.MODELS_QWEN
    files = qwen_model_files(quality, folder, text_encoder)
    used = {"turbo": qwen_size_on_disk("turbo", quality, folder), "full": qwen_size_on_disk("full", quality, folder),
            "text_encoder": text_encoder_in_use(text_encoder, folder)}
    wanted = {"turbo": quality, "full": quality, "text_encoder": text_encoder}
    words = []
    for key, label in (("turbo", "Turbo"), ("full", "full model"), ("text_encoder", "text encoder")):
        name = files[key].name
        if not files[key].is_file():
            words.append(f"{label} {name} (not downloaded)")
        elif used[key] != wanted[key]:
            words.append(f"{label} {name} ({wanted[key]}, the configured size, is not there: the biggest bigger one)")
        else:
            words.append(f"{label} {name}")
    return ", ".join(words)


# FLUX.3 Image Klein (Black Forest Labs, 2026-10-15; the FLUX Kommunity License: free for non-commercial use, and the
# pictures for commercial use only by creators and companies under US$5M of yearly revenue): a 5.6B distilled model,
# an optional third model of the Images stage beside Qwen-Image 2.1's two, which Qwen-Image's launcher serves with its
# own files (launchers/qwen_server.py "flux3"). Never downloaded by the installer: only when the artist asks for it on
# the Images page (spellcast/model_downloads.py, flux3_manifest), flat in models/flux3-klein under these names. Its
# denoiser in unsloth's recommended file for each of the hub's sizes (config.QUALITIES: the "UD" dynamic quants where
# unsloth has them), FLUX.2's VAE (an Apache-2.0 repo's copy, byte-identical to Black Forest Labs' own ae.safetensors)
# and the Qwen3-VL-4B-Instruct text encoder (Apache-2.0) for every size - no vision part: FLUX.3's example pictures go
# through the VAE, not through the text encoder. The SHA-256 are HuggingFace's LFS ids at these commits.
FLUX3_GGUF_REPO = "unsloth/flux-3-image-klein-GGUF"
FLUX3_GGUF_REVISION = "12ef13ce920d53f45fe805093910c899668b9e76"
FLUX3_DENOISERS: dict[str, tuple[str, tuple[int, str]]] = {     # quality -> (its file, (bytes, sha256))
    "Q3_K_M": ("flux-3-image-klein-UD-Q3_K_XL.gguf",
               (3_279_836_064, "8508bfefa5c0e62de434336a9836c5ac8a234c73f02a296507c1d637f560ac9c")),
    "Q4_K_M": ("flux-3-image-klein-UD-Q4_K_XL.gguf",
               (3_929_944_992, "0a057e07550a5b2169e344d686171a7dd3c7d82c7d8d1d1605d8d881c16e9fa7")),
    "Q5_K_M": ("flux-3-image-klein-UD-Q5_K_XL.gguf",
               (4_479_218_592, "12ed7d9f4fbd4aba881bc1741f7d35dc840435c754ffb298a04e8199df9e275b")),
    "Q6_K": ("flux-3-image-klein-Q6_K.gguf",
             (4_799_886_240, "a33abd53ee3484cb9aec0a81c4bc5227e0a0fc40a67e645834f3253195cf5ed8")),
    "Q8_0": ("flux-3-image-klein-Q8_0.gguf",
             (6_092_755_872, "af562de40a975439a0f55340221bf48c951e2fc5ad0e11df0bdd73c8091eefcf")),
}
FLUX3_TEXT_ENCODER = "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
FLUX3_VAE = "flux2-vae.safetensors"
FLUX3_SHARED_FILES: Manifest = {    # needed with every size
    FLUX3_TEXT_ENCODER: (2_497_281_664, "66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a"),
    FLUX3_VAE: (336_211_292, "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3"),
}
# Where each file comes from: its name here -> (HuggingFace repo, commit, its path in the repo).
FLUX3_SOURCES: dict[str, tuple[str, str, str]] = {
    **{name: (FLUX3_GGUF_REPO, FLUX3_GGUF_REVISION, name) for name, _pin in FLUX3_DENOISERS.values()},
    FLUX3_TEXT_ENCODER: ("Qwen/Qwen3-VL-4B-Instruct-GGUF", "1cd86afb9a95c410a6038ab3b40d8b578c892266",
                         FLUX3_TEXT_ENCODER),
    FLUX3_VAE: ("Comfy-Org/Flux-2-Klein-4B", "533c4cbc3d3b89af1743d9466c434e89810391c0",
                "split_files/vae/flux2-vae.safetensors"),
}
# Every pinned file of FLUX.3 Image Klein in every size (prepare_release.py check compares them with HuggingFace).
FLUX3_PINS: Manifest = {**{name: pin for name, pin in FLUX3_DENOISERS.values()}, **FLUX3_SHARED_FILES}

# The models that are a download of their own, on the artist's request (spellcast/model_downloads.py,
# launchers/qwen_download.py --model): the full Qwen-Image 2.1 (into models/qwen-image), FLUX.3 Image Klein (into
# models/flux3-klein) and Hi3D's AI models (into models/hi3d: the 3D stage's second Shape, picked per prop). The Turbo
# and Pixal3D's own models are the installer's.
OPTIONAL_MODELS: tuple[str, ...] = ("full", "flux3", "hi3d")


def flux3_denoiser(quality: str) -> tuple[str, tuple[int, str]]:
    """(the file name, (bytes, sha256)) of FLUX.3 Image Klein's denoiser for the hub's size `quality`."""
    return FLUX3_DENOISERS[quality]


def flux3_size_on_disk(quality: str, folder: Path | None = None) -> str | None:
    """size_on_disk of FLUX.3 Image Klein's denoiser in models/flux3-klein (or `folder`), as qwen_size_on_disk."""
    folder = folder or paths.MODELS_FLUX3
    return size_on_disk(quality, tuple(FLUX3_DENOISERS),
                        lambda size: not missing_files(folder, dict([flux3_denoiser(size)])))


def flux3_manifest(quality: str, folder: Path | None = None) -> Manifest:
    """FLUX.3 Image Klein's files for the size `quality`: its denoiser - of the size in use (flux3_size_on_disk), else
    of that size -, the text encoder and the VAE."""
    name, pin = flux3_denoiser(flux3_size_on_disk(quality, folder) or quality)
    return {name: pin, **FLUX3_SHARED_FILES}


def flux3_model_on_disk(quality: str, folder: Path | None = None) -> bool:
    """Whether all of FLUX.3 Image Klein's files for size `quality` (flux3_manifest) are in models/flux3-klein (or
    `folder`), whole."""
    return not missing_files(folder or paths.MODELS_FLUX3, flux3_manifest(quality, folder))


def flux3_model_files(quality: str, folder: Path | None = None) -> dict[str, Path]:
    """{"diffusion", "vae", "text_encoder": its file} of FLUX.3 Image Klein for stable-diffusion.cpp
    (diffusion_model_path, vae_path, llm_path), in models/flux3-klein unless `folder` is given: the denoiser of the
    size in use (flux3_size_on_disk; of size `quality` when none is there), each path whether the file is there or
    not."""
    folder = folder or paths.MODELS_FLUX3
    return {"diffusion": folder / flux3_denoiser(flux3_size_on_disk(quality, folder) or quality)[0],
            "vae": folder / FLUX3_VAE, "text_encoder": folder / FLUX3_TEXT_ENCODER}


def optional_manifest(model: str, quality: str, folder: Path | None = None) -> Manifest:
    """The files of the optional model `model` (OPTIONAL_MODELS) for size `quality` in its folder (`folder`: another
    one): what its download fetches, less what is there."""
    if model == "full":
        return qwen_full_manifest(quality, folder)
    if model == "flux3":
        return flux3_manifest(quality, folder)
    if model == "hi3d":
        return HI3D_FILES               # one size: the hub's `quality` is Qwen-Image's
    raise ValueError(f"no optional model {model!r} (one of {', '.join(OPTIONAL_MODELS)})")


def optional_sources(model: str) -> dict[str, tuple[str, str, str]]:
    """Where the files of the optional model `model` come from: {name: (repo, commit, path in the repo)}."""
    if model == "full":
        return QWEN_SOURCES
    if model == "flux3":
        return FLUX3_SOURCES
    if model == "hi3d":
        return HI3D_SOURCES
    raise ValueError(f"no optional model {model!r} (one of {', '.join(OPTIONAL_MODELS)})")


# Hi3D v1.0 (OCEANO HK HOLDING LIMITED, the Hi3D Open License Agreement 1.0): the 3D stage's second geometry, a mesh
# refiner that Pixal3D's launcher runs on Pixal3D's coarse shape (projects.py mesh.settings.geometry). A download of
# its own, which the 3D stage offers when a prop is set to Hi3D (OPTIONAL_MODELS "hi3d", spellcast/model_downloads.py;
# its code and packages come with Pixal3D: install/components.py), into models/hi3d, one folder per source: Hi3D's own networks with its LICENSE and
# notices (the licence asks that they stay beside the weights), DINOv2-large (Apache-2.0) and CLIP ViT-L/14 (MIT), its
# image encoders, and TRELLIS.2's shape encoder (MIT). The SHA-256 of the big files are HuggingFace's LFS ids at these
# commits; those of the small ones were taken from the files themselves (their git blob ids checked against the API).
HI3D_SOURCES_PINNED: dict[str, tuple[str, str]] = {      # its folder here -> (HuggingFace repo, commit)
    "Hi3Dv1.0": ("hi3d-official/Hi3Dv1.0", "d502e938cbedc6e87aba8884614c5ca6d177a4b6"),
    "dinov2-large": ("facebook/dinov2-large", "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"),
    "clip-vit-large-patch14": ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"),
    "trellis2-shape-encoder": ("microsoft/TRELLIS.2-4B", "af44b45f2e35a493886929c6d786e563ec68364d"),
}
HI3D_FILES: Manifest = {
    "Hi3Dv1.0/model_index.json": (224, "e24988612733dad44b7354fbfa0b2ef50073ba6ef8d7a677e41023dc2d651386"),
    "Hi3Dv1.0/scheduler/scheduler_config.json":
        (127, "40cdcdaaa6149dea32b86a5ff730e581b069ae619d7f580444ef503737fb0bf3"),
    "Hi3Dv1.0/vae/config.json": (131, "71c629eb188485786227277c7bb5b0fa0a1bfcf07a3d21316174da6b4d04f92f"),
    "Hi3Dv1.0/vae/model.safetensors":
        (376_629_712, "605d207c77f8db0b33c567a5861252f0090cf3c3604dfda1cfee09e8f0b53f3e"),
    "Hi3Dv1.0/transformer_512/config.json": (1_336, "809189b8183c3f44118751322f2cb6fe6fa4ca1922ea16c88cd032970d4bc937"),
    "Hi3Dv1.0/transformer_512/model.safetensors":
        (3_675_679_280, "8ba88311dccf23463c6e10dd0d3bd5e0c5c35d1608c6f466413b6428b5ad8e56"),
    "Hi3Dv1.0/transformer_1024/config.json": (1_337, "2f96d851844e159523cb6cf5e5b29ddbc1354e222a333232612c19e1eaf278c2"),
    "Hi3Dv1.0/transformer_1024/model.safetensors":
        (3_675_679_280, "a2ca5e44dde51cacecd59b42fceeb5f14e75d55e6dd047337752c4e6f281008c"),
    "Hi3Dv1.0/LICENSE": (27_470, "e7914ddbb068110d9f9e1e1283dd4bc23ec2a5223909b7fd2a2ca9a152d853fe"),
    "Hi3Dv1.0/NOTICE": (2_157, "fb56f04476d143ed5d9f44cb90dba2a7d7c67ee387d0be138937dd2586a16912"),
    "Hi3Dv1.0/AUTHORS": (713, "996bb0bcc8829a22ae9f1a57d874ce7b9bc29f5f6acfd397cfc7d1d1f6e574cc"),
    "Hi3Dv1.0/THIRD_PARTY_LICENSES.md": (24_344, "0b3316bbe649b762a4a0c14aa19bdd6e112ba6291e87fb33aa79de3c4cdb4a9a"),
    "dinov2-large/config.json": (549, "12df51c069a2dc1305e34ba71ef58bc2407ea553b75f4722a1715c1bce3bbed0"),
    "dinov2-large/model.safetensors":
        (1_217_522_888, "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23"),
    "dinov2-large/preprocessor_config.json": (436, "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"),
    "clip-vit-large-patch14/config.json": (4_519, "8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a"),
    "clip-vit-large-patch14/model.safetensors":
        (1_710_540_580, "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb"),
    "clip-vit-large-patch14/preprocessor_config.json":
        (316, "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd"),
    "trellis2-shape-encoder/ckpts/shape_enc_next_dc_f16c32_fp16.json":
        (653, "5a960a5c419d70f8d4a7a6b6e52ae689cd948c88bd7ed92346adfbf82cd12007"),
    "trellis2-shape-encoder/ckpts/shape_enc_next_dc_f16c32_fp16.safetensors":
        (708_797_208, "f37c5ff5b983b68e9946060000f09bc131f3e84318a2c8b7430a81e4b4636c41"),
}
# Where each file comes from: its name here -> (HuggingFace repo, commit, its path in the repo).
HI3D_SOURCES: dict[str, tuple[str, str, str]] = {
    rel: (*HI3D_SOURCES_PINNED[rel.split("/", 1)[0]], rel.split("/", 1)[1]) for rel in HI3D_FILES}
HI3D_BYTES = sum(size for size, _sha in HI3D_FILES.values())     # 11,364,913,260
# Hi3D's code: a file that proves it is checked out (the launcher imports the package sparc3d from there).
HI3D_CODE_LANDMARK = "sparc3d/__init__.py"


def hi3d_code() -> Path:
    """Hi3D's code: code/hi3d, or the checkout SPELLCAST_HI3D_CODE names (as Pixal3D's launcher reads it)."""
    override = os.environ.get(paths.HI3D_CODE_ENV, "").strip().strip('"')
    return Path(override) if override else paths.HI3D_DIR


def hi3d_on_disk(folder: Path | None = None) -> bool:
    """Whether Hi3D can make a shape here: its code, and every file of HI3D_FILES in models/hi3d (or `folder`) at its
    size (a few stat calls, no hashing)."""
    return paths.hi3d_code_here() and not missing_files(folder or paths.MODELS_HI3D, HI3D_FILES)


# Folders to take AI models from before downloading them (install.py --models-from, or SPELLCAST_MODELS_FROM with
# several separated by ";"): e.g. another Spellcast3D's models folder, or a folder with LATO.2's .pt files.
EXTRA_SOURCES: list[Path] = []
DOWNLOAD_ATTEMPTS = 6
DOWNLOAD_TIMEOUT = 60
# {file or folder: [size, mtime_ns, sha256 / tree digest]} of the retopology models checked on this PC (None:
# paths.VENVS / models-verified.json; the tests point it elsewhere).
VERIFIED_RECORDS: Path | None = None

Log = Callable[[str], None]


def ensure_folders() -> None:
    """Create the models folders (a copy of Spellcast3D made without git may lack the empty ones)."""
    for folder in FOLDERS:
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass   # e.g. a read-only folder: whoever needs it next reports the problem


# ---- code/pixal3d/MODELS -> models/pixal3d ---------------------------------------------------------------

def is_junction(path: Path) -> bool:
    try:
        return os.lstat(path).st_reparse_tag == stat.IO_REPARSE_TAG_MOUNT_POINT
    except OSError:
        return False


def _is_link(path: Path) -> bool:
    return os.path.islink(path) or is_junction(path)


def _same_volume(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def _same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def _supports_junctions(folder: Path) -> bool:
    """NTFS and ReFS can hold junctions; FAT32 / exFAT (e.g. a USB stick) cannot."""
    flags = ctypes.c_uint32()
    root = os.path.splitdrive(os.path.abspath(folder))[0] + "\\"
    ok = ctypes.windll.kernel32.GetVolumeInformationW(ctypes.c_wchar_p(root), None, 0, None, None,
                                                      ctypes.byref(flags), None, 0)
    return bool(ok) and bool(flags.value & _FILE_SUPPORTS_REPARSE_POINTS)


def pixal3d_link_work() -> str | None:
    """What still has to happen to code/pixal3d/MODELS, in artist words, or None when nothing can or must.

    None also when it is a link the artist made to a folder of their own, and when an older real folder
    cannot be moved without copying (models/ is on another drive): Pixal3D then keeps using that folder.
    """
    link, target = paths.PIXAL3D_MODELS_LINK, paths.MODELS_PIXAL3D
    if _is_link(link) and link.is_dir():
        return None
    if not _supports_junctions(paths.PIXAL3D_DIR):
        return None
    if link.is_dir():
        existing = next(p for p in (target, paths.MODELS, paths.ROOT) if p.exists())
        if not _same_volume(link, existing):
            return None
        return "its AI models move into the models\\pixal3d folder"
    return "the link from code\\pixal3d\\MODELS to models\\pixal3d is missing"


def link_pixal3d_models(log: Log) -> bool:
    """Make code/pixal3d/MODELS a junction to models/pixal3d, moving an older real MODELS folder there first.

    Returns False (and logs why) when it could not; then everything is as it was before, and Pixal3D keeps
    using code/pixal3d/MODELS as a normal folder. The next start tries again.
    """
    link, target = paths.PIXAL3D_MODELS_LINK, paths.MODELS_PIXAL3D
    if pixal3d_link_work() is None:
        return True
    undo: list[tuple[Path, Path]] = []      # renames done so far, (from, to)
    try:
        target.mkdir(parents=True, exist_ok=True)
        if _is_link(link):
            os.rmdir(link)                  # a link to a folder that is gone: removes only the link
        elif link.is_dir():
            _move_into(link, target, "code\\pixal3d\\MODELS", undo, log)
        _make_junction(link, target)
    except OSError as e:
        failed = _undo(undo)
        log(f"Could not move Pixal3D's AI models into models\\pixal3d ({e}). Pixal3D keeps them in "
            f"code\\pixal3d\\MODELS; Spellcast3D tries again at its next start."
            + (f" These could not be put back: {', '.join(failed)}" if failed else ""))
        return False
    log("Pixal3D's AI models are in models\\pixal3d (code\\pixal3d\\MODELS links there).")
    return True


def unwrap_copied_folder(log: Log) -> None:
    """An artist who copied a whole MODELS folder (e.g. from StableProjectorz's Pixal3D) into models/pixal3d,
    instead of what is inside it: move its contents up, where Pixal3D looks."""
    target = paths.MODELS_PIXAL3D
    if not target.is_dir() or any((target / name).exists() for name in _PIXAL3D_LANDMARKS):
        return
    nested = next((p for p in target.iterdir() if p.name.lower() == "models" and p.is_dir() and not _is_link(p)
                   and any((p / name).exists() for name in _PIXAL3D_LANDMARKS)), None)
    if nested is None:
        return
    undo: list[tuple[Path, Path]] = []
    try:
        _move_into(nested, target, f"models\\pixal3d\\{nested.name}", undo, log)
    except OSError as e:
        _undo(undo)
        log(f"Could not move the contents of models\\pixal3d\\{nested.name} up into models\\pixal3d ({e}). "
            f"Move them there yourself (Pixal3D does not look inside {nested.name}).")


def _move_into(old: Path, target: Path, name: str, undo: list[tuple[Path, Path]], log: Log) -> None:
    """Rename everything of `old` that `target` lacks into `target`, set the rest aside, and remove `old`."""
    moves: list[tuple[Path, Path]] = []
    kept: list[Path] = []
    _plan_merge(old, target, moves, kept)
    if moves:
        log(f"Moving Pixal3D's AI models from {name} into models\\pixal3d ({len(moves)} item(s); "
            f"renamed on the same drive, nothing is copied) ...")
    for src, dst in moves:
        os.rename(src, dst)
        undo.append((src, dst))
    size = sum(folder_bytes(p) if p.is_dir() and not _is_link(p) else os.lstat(p).st_size for p in kept)
    _remove_empty_folders(old)
    try:
        os.rmdir(old)                       # empty unless something was kept
        return
    except OSError:
        pass
    aside = paths.MODELS / f"pixal3d-old-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    os.rename(old, aside)
    undo.append((old, aside))
    if kept:
        log(f"{len(kept)} file(s) or folder(s) of {name} ({size / 1e9:.1f} GB) were already in models\\pixal3d, "
            f"which keeps its own. The older copies are now in {aside.relative_to(paths.ROOT)}: delete that "
            f"folder once Pixal3D works.")


def _plan_merge(src: Path, dst: Path, moves: list[tuple[Path, Path]], kept: list[Path]) -> None:
    for entry in os.scandir(src):
        s, d = Path(entry.path), dst / entry.name
        if not os.path.lexists(d):
            moves.append((s, d))
        elif entry.is_dir(follow_symlinks=False) and not _is_link(s) and d.is_dir() and not _is_link(d):
            _plan_merge(s, d, moves, kept)
        else:
            kept.append(s)


def _undo(undo: list[tuple[Path, Path]]) -> list[str]:
    failed = []
    for src, dst in reversed(undo):
        try:
            src.parent.mkdir(parents=True, exist_ok=True)   # _remove_empty_folders may have removed it
            os.rename(dst, src)
        except OSError:
            failed.append(str(dst))
    undo.clear()
    return failed


def _remove_empty_folders(root: Path) -> None:
    """Remove the folders under root (not root itself) that the moves left empty."""
    for dirpath, _dirs, _files in os.walk(root, topdown=False):
        try:
            if Path(dirpath) != root and not os.listdir(dirpath) and not _is_link(Path(dirpath)):
                os.rmdir(dirpath)
        except OSError:
            pass    # then `root` is set aside with it


def _make_junction(link: Path, target: Path) -> None:
    # mklink is built into cmd. The quotes are written here: subprocess quotes only arguments with spaces,
    # and cmd would split a folder name with & at the &. /d: no AutoRun commands from the registry.
    res = subprocess.run(f'cmd /d /c mklink /J "{link}" "{target}"', capture_output=True, text=True,
                         errors="replace", stdin=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if res.returncode != 0 or not is_junction(link):
        raise OSError(f"mklink: {(res.stdout + res.stderr).strip() or f'exit code {res.returncode}'}")


# ---- sizes ----------------------------------------------------------------------------------------------

def folder_bytes(folder: Path) -> int:
    """Bytes of the files under folder, a hard-linked file once. Symbolic links are not followed (HuggingFace's
    snapshot links point at blobs that are counted already); `folder` itself may be a junction."""
    seen: set[tuple[int, int]] = set()
    total = 0
    for dirpath, _dirs, files in os.walk(folder):
        for name in files:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode) or (st.st_dev, st.st_ino) in seen:
                continue
            if st.st_ino:
                seen.add((st.st_dev, st.st_ino))
            total += st.st_size
    return total


def partial_download_bytes(hf_home: Path) -> int:
    """Bytes of interrupted HuggingFace downloads (<blob>.incomplete) under hf_home/hub."""
    total = 0
    for dirpath, _dirs, files in os.walk(hf_home / "hub"):
        for name in files:
            if name.endswith(".incomplete"):
                try:
                    total += os.lstat(os.path.join(dirpath, name)).st_size
                except OSError:
                    pass
    return total


# ---- models the artist already has in their own HuggingFace cache ------------------------------------------

def hf_repo_folder(hub: Path, repo_id: str) -> Path:
    return hub / ("models--" + repo_id.replace("/", "--"))


def hf_snapshot(hub: Path, repo_id: str) -> Path | None:
    """The snapshot folder that refs/main points to in a HuggingFace cache (what an offline start uses)."""
    folder = hf_repo_folder(hub, repo_id)
    try:
        commit = (folder / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    snapshot = folder / "snapshots" / commit
    return snapshot if commit and snapshot.is_dir() else None


HF_CACHES_ENV = "SPELLCAST_HF_CACHES"


def user_hf_caches() -> list[Path]:
    """The artist's own HuggingFace caches: HF_HUB_CACHE, HF_HOME\\hub, then the default ~/.cache one. A child of the
    hub (the background installer, a model download) gets them in HF_CACHES_ENV (hf_caches_env), separated by ";":
    paths.clean_env scrubs those two variables from every child."""
    found: list[Path] = []
    for value in (*os.environ.get(HF_CACHES_ENV, "").split(";"), os.environ.get("HF_HUB_CACHE"),
                  os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
                  os.path.join(Path.home(), ".cache", "huggingface", "hub")):
        if value and Path(value).is_dir() and not any(_same_path(Path(value), f) for f in found):
            found.append(Path(value))
    return found


def hf_caches_env() -> dict[str, str]:
    """{HF_CACHES_ENV: the caches user_hf_caches finds here}, for a child process's environment."""
    return {HF_CACHES_ENV: ";".join(str(p) for p in user_hf_caches())}


def link_from_hf_cache(hub: Path, repo_id: str, files: tuple[str, ...] | None, log: Log) -> bool:
    """Hard-link repo_id's snapshot from the artist's own HuggingFace cache into `hub` instead of downloading it.

    files None = every file of that snapshot. Only from a cache on the same drive (a hard link takes no disk
    space; tens of GB are never copied), and only when that snapshot has every file. Returns True when the
    files are now in `hub`.
    """
    for cache in user_hf_caches():
        if hub.exists() and _same_path(cache, hub):
            continue
        source = hf_snapshot(cache, repo_id)
        if source is None:
            continue
        wanted = files if files is not None else tuple(
            p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file())
        real = {f: Path(os.path.realpath(source / f)) for f in wanted}
        if not real or not all(p.is_file() for p in real.values()):
            continue
        hub.mkdir(parents=True, exist_ok=True)
        if not all(_same_volume(p, hub) for p in real.values()):
            log(f"{repo_id} is in your HuggingFace cache ({cache}), but on another drive, so it is downloaded "
                f"(copying it would take as much disk space as a download).")
            return False
        try:
            _link_snapshot(cache, hub, repo_id, source, real)
        except OSError as e:
            log(f"Could not link {repo_id} from your HuggingFace cache ({e}); it is downloaded instead.")
            return False
        size = sum(p.stat().st_size for p in real.values())
        log(f"Found {repo_id} in your HuggingFace cache ({cache}): linked {len(real)} file(s), {size / 1e9:.1f} GB, "
            f"instead of downloading them (hard links: no extra disk space).")
        return True
    return False


def _link_snapshot(cache: Path, hub: Path, repo_id: str, source: Path, real: dict[str, Path]) -> None:
    """The layout huggingface_hub itself writes where it cannot make symbolic links: a snapshot file per file
    (a hard link to the artist's file here), its blob too when the name of that is known, and refs/main."""
    blobs = os.path.normcase(os.path.realpath(hf_repo_folder(cache, repo_id) / "blobs"))
    dst = hf_repo_folder(hub, repo_id)
    commit = source.name
    for rel, file in real.items():
        targets = [dst / "snapshots" / commit / rel]
        if os.path.normcase(str(file.parent)) == blobs:   # a cache with symbolic links: blob name = its etag
            targets.append(dst / "blobs" / file.name)
        for target in targets:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(file, target)
    (dst / "refs").mkdir(parents=True, exist_ok=True)
    (dst / "refs" / "main").write_text(commit, encoding="utf-8")


# ---- the AI models the installer fetches itself (Qwen-Image 2.1, Hi3D, LATO.2, TriFlow) ------------------------

def hf_url(repo_id: str, revision: str, name: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{name}"


def missing_files(folder: Path, manifest: Manifest) -> list[str]:
    """The manifest's files that are not in `folder` at their size (a few stat calls: no hashing)."""
    missing = []
    for rel, (size, _sha) in manifest.items():
        try:
            if os.stat(folder / rel).st_size == size:
                continue
        except OSError:
            pass
        missing.append(rel)
    return missing


def present_bytes(folder: Path, manifest: Manifest) -> int:
    """Bytes of the manifest's files that are in `folder` at their size, plus their interrupted downloads (.part)."""
    total = 0
    for rel, (size, _sha) in manifest.items():
        try:
            if os.stat(folder / rel).st_size == size:
                total += size
                continue
        except OSError:
            pass
        try:
            total += min(os.stat(folder / (rel + ".part")).st_size, size)
        except OSError:
            pass
    return total


def sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class ChecksumError(OSError):
    """A download arrived in full but is not the pinned file (its SHA-256 differs)."""


# ---- what was checked on this PC already -------------------------------------------------------------------------

def _records_file() -> Path:
    return VERIFIED_RECORDS or paths.VENVS / "models-verified.json"


def _records() -> dict[str, list]:
    try:
        data = json.loads(_records_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _record_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _signature(path: Path) -> list[int] | None:
    """[size, mtime_ns] of a file, or [files, bytes, newest mtime_ns] of a folder (its __pycache__ left out)."""
    try:
        if not path.is_dir():
            st = os.stat(path)
            return [st.st_size, st.st_mtime_ns]
        count = size = newest = 0
        for dirpath, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in files:
                st = os.stat(os.path.join(dirpath, name))
                count, size, newest = count + 1, size + st.st_size, max(newest, st.st_mtime_ns)
        return [count, size, newest]
    except OSError:
        return None


def record_verified(path: Path, digest: str) -> None:
    """Remember that `path` (a file or DINOv2's code folder) was checked against `digest` as it is now."""
    signature = _signature(path)
    if signature is None:
        return
    records = _records()
    records[_record_key(path)] = [*signature, digest]
    try:
        paths.write_text_atomic(_records_file(), json.dumps(records, indent=1) + "\n")
    except OSError:
        pass    # it is checked again next time


def is_recorded(path: Path, digest: str, records: dict[str, list] | None = None) -> bool:
    """`path` was checked against `digest` and has not changed since (size and time)."""
    signature = _signature(path)
    return signature is not None and (records if records is not None else _records()).get(
        _record_key(path)) == [*signature, digest]


def unverified_files(folder: Path, manifest: Manifest) -> list[str]:
    """The manifest's files that are in `folder` at their size but were never checked on this PC (e.g. copied in by
    hand): a few stat calls, no hashing (check_unverified does that)."""
    records = _records()
    out = []
    for rel, (size, sha) in manifest.items():
        try:
            if os.stat(folder / rel).st_size != size:
                continue            # missing_files' part
        except OSError:
            continue
        if not is_recorded(folder / rel, sha, records):
            out.append(rel)
    return out


def check_unverified(folder: Path, manifest: Manifest, log: Log) -> list[str]:
    """Hash the files unverified_files names, once: the pinned ones are remembered, any other is removed from
    `folder` (a hard link: only this name of it), so it is linked or downloaded again. Returns those removed."""
    wrong = []
    for rel in unverified_files(folder, manifest):
        size, sha = manifest[rel]
        path = folder / rel
        log(f"Checking {rel} ({size / 1e9:.2f} GB) once: it was not downloaded by Spellcast3D ...")
        try:
            ok = sha256_file(path) == sha
        except OSError as e:
            log(f"  Could not read it ({e}); it is checked again next time.")
            continue
        if ok:
            record_verified(path, sha)
            continue
        log(f"  {rel} is not the file this version of Spellcast3D needs (its SHA-256 differs): it is replaced.")
        try:
            path.unlink()
            wrong.append(rel)
        except OSError as e:
            log(f"  Could not remove it ({e}). Delete {path} yourself and start Spellcast3D again.")
    return wrong


def dinov2_code_ok(folder: Path) -> bool:
    """DINOv2's code in `folder` is the pinned commit's: remembered from an earlier check, else checked now (4 MB)."""
    if not (folder / DINOV2_CODE_LANDMARK).is_file():
        return False
    if is_recorded(folder, DINOV2_CODE_DIGEST):
        return True
    if tree_digest(folder) != DINOV2_CODE_DIGEST:
        return False
    record_verified(folder, DINOV2_CODE_DIGEST)
    return True


def extra_sources() -> list[Path]:
    """The folders of --models-from and SPELLCAST_MODELS_FROM that exist."""
    found = list(EXTRA_SOURCES)
    found += [Path(p.strip().strip('"')) for p in os.environ.get("SPELLCAST_MODELS_FROM", "").split(";") if p.strip()]
    return [p for p in dict.fromkeys(found) if p.is_dir()]


def _hf_metadata(src: Path, rel: str) -> tuple[str, str, float] | None:
    """(commit, etag, timestamp) that huggingface_hub wrote for `rel` in a --local-dir download folder."""
    try:
        lines = (src / ".cache" / "huggingface" / "download" / f"{rel}.metadata").read_text(encoding="utf-8").split()
        return lines[0], lines[1], float(lines[2])
    except (OSError, IndexError, ValueError):
        return None


def verified(file: Path, size: int, sha256: str, src: Path | None = None, rel: str = "",
             revision: str | None = None) -> bool:
    """True when `file` is that exact file. Cheap when something already vouches for it: huggingface_hub's
    metadata in a --local-dir folder (the commit and the SHA-256 etag, written after the file), or a HuggingFace
    cache blob named after its SHA-256. Otherwise the file is hashed (a few seconds per GB)."""
    try:
        st = os.stat(file)
    except OSError:
        return False
    if st.st_size != size:
        return False
    meta = _hf_metadata(src, rel) if src is not None else None
    if meta is not None and meta[1] == sha256 and (revision is None or meta[0] == revision) \
            and meta[2] >= st.st_mtime - 1:
        return True
    if Path(os.path.realpath(file)).name == sha256:
        return True
    try:
        return sha256_file(file) == sha256
    except OSError:
        return False


def link_from_local_dir(src: Path, dst: Path, files: list[str] | tuple[str, ...], expected: Manifest, log: Log,
                        revision: str | None = None, names: dict[str, str] | None = None,
                        what: str = "AI model file(s)") -> list[str]:
    """Hard-link those of `files` that `src` holds (checked against `expected`: size + SHA-256) into `dst`.

    For a folder laid out like `dst` (an `hf download --local-dir` folder, another Spellcast3D's models folder,
    a HuggingFace snapshot); `names` gives the files that have another name in `src` ({name in dst: name in src}).
    Only on the same drive: a hard link takes no disk space, GBs are never copied. A file already in `dst` at its
    size is left alone, a wrong one is replaced. Returns the files it linked.
    """
    linked: list[str] = []
    other_drive = False
    if not src.is_dir() or (dst.exists() and _same_path(src, dst)):
        return linked
    for rel in files:
        size, sha = expected[rel]
        src_rel = (names or {}).get(rel, rel)
        source, target = src / src_rel, dst / rel
        if not missing_files(dst, {rel: expected[rel]}) or not source.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if not _same_volume(source, target.parent):
            other_drive = True
            continue
        if not verified(source, size, sha, src, src_rel, revision):
            log(f"{source} is not the expected file (its size or SHA-256 differs), so it is not used.")
            continue
        tmp = target.with_name(target.name + ".link")
        try:
            tmp.unlink(missing_ok=True)
            os.link(os.path.realpath(source), tmp)
            os.replace(tmp, target)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            log(f"Could not link {source} ({e}); it is downloaded instead.")
            continue
        (dst / (rel + ".part")).unlink(missing_ok=True)
        record_verified(target, sha)
        linked.append(rel)
    if linked:
        size = sum(expected[rel][0] for rel in linked)
        log(f"Found {len(linked)} {what} in {src}: linked them ({size / 1e9:.2f} GB) instead of "
            f"downloading them (hard links: no extra disk space).")
    if other_drive:
        log(f"{src} holds AI models, but on another drive, so they are downloaded (copying them would take as "
            f"much disk space as a download).")
    return linked


def hf_cache_sources(repo_id: str, revision: str) -> list[Path]:
    """Snapshot folders of repo_id in the artist's HuggingFace caches: the pinned commit's, then refs/main's
    (their files are checked against the SHA-256 anyway)."""
    found: list[Path] = []
    for cache in user_hf_caches():
        pinned = hf_repo_folder(cache, repo_id) / "snapshots" / revision
        for snapshot in (pinned if pinned.is_dir() else None, hf_snapshot(cache, repo_id)):
            if snapshot is not None and not any(_same_path(snapshot, f) for f in found):
                found.append(snapshot)
    return found


def torch_hub_dirs() -> list[Path]:
    """The artist's torch.hub folders (TORCH_HOME\\hub, then the default ~/.cache/torch/hub)."""
    found: list[Path] = []
    for value in (os.path.join(os.environ["TORCH_HOME"], "hub") if os.environ.get("TORCH_HOME") else None,
                  os.path.join(Path.home(), ".cache", "torch", "hub")):
        if value and Path(value).is_dir() and not any(_same_path(Path(value), f) for f in found):
            found.append(Path(value))
    return found


def download_file(url: str, dst: Path, size: int, sha256: str, log: Log) -> None:
    """Download url to dst over HTTPS, continuing <dst>.part where an earlier try (or run) stopped.

    The file gets its name only once its size and SHA-256 are right. A continued download whose SHA-256 is wrong
    (e.g. a power cut left the end of the .part unwritten) is downloaded once more from its first byte. Raises
    OSError with a short reason (ChecksumError: it arrived in full, but not as the pinned file).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    resumed = _fetch(url, part, size, log, dst.name)
    if sha256_file(part) != sha256:
        part.unlink()
        if not resumed:
            raise ChecksumError(f"{dst.name}: the downloaded file is not the expected one (its SHA-256 differs)")
        log(f"  {dst.name}: the part continued from an earlier download is not intact; downloading it again from "
            f"the start ...")
        _fetch(url, part, size, log, dst.name)
        if sha256_file(part) != sha256:
            part.unlink()
            raise ChecksumError(f"{dst.name}: the downloaded file is not the expected one (its SHA-256 differs)")
    os.replace(part, dst)
    record_verified(dst, sha256)


def _fetch(url: str, part: Path, size: int, log: Log, name: str) -> bool:
    """Download url into `part` until it has `size` bytes, continuing what it holds -> whether any of it came from
    before (a continued download). Raises OSError when it does not get there."""
    problem = ""
    last: BaseException | None = None       # the error of the last try: the cause of the OSError raised below
    resumed = False
    for attempt in range(DOWNLOAD_ATTEMPTS):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if have == size:
            resumed = resumed or attempt == 0      # a .part an earlier run finished (never checked since)
            break
        if attempt:
            log(f"  {name}: {problem}; trying again ...")
            time.sleep(min(30, 5 * attempt))
        req = urllib.request.Request(url, headers={"User-Agent": "spellcast3d-installer"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
                if have and resp.status != 206:      # the server ignored the range: start again
                    have = 0
                resumed = resumed or bool(have)
                with open(part, "ab" if have else "wb") as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
        except urllib.error.HTTPError as e:
            problem, last = f"the server answered HTTP {e.code}", e
            if e.code in (401, 403, 404):
                break
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            problem, last = str(getattr(e, "reason", e)) or type(e).__name__, e
    got = part.stat().st_size if part.exists() else 0
    if got != size:
        raise OSError(f"{name}: {problem or 'the download stopped'} ({got:,} of {size:,} bytes)") from last
    return resumed


def tree_digest(folder: Path) -> str | None:
    """SHA-256 over "<git blob id> <path>" of every file under folder (sorted by path, __pycache__ left out): the
    same for a checkout, a GitHub archive or torch.hub's copy of one commit. None when folder is missing."""
    if not folder.is_dir():
        return None
    entries = []
    for dirpath, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            full = Path(dirpath) / name
            data = full.read_bytes()
            blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
            entries.append((full.relative_to(folder).as_posix(), blob))
    return hashlib.sha256("".join(f"{blob} {rel}\n" for rel, blob in sorted(entries)).encode("utf-8")).hexdigest()


def link_dinov2_code(src: Path, dst: Path, log: Log) -> bool:
    """Put DINOv2's pinned code that is on this PC already (e.g. torch.hub's facebookresearch_dinov2_main) into
    dst: hard links on the same drive, else a copy (4 MB); only when it is exactly that commit's code. A dst with
    other code (another commit, copied in by hand) is replaced. Returns True when dst has the code now."""
    if dinov2_code_ok(dst):
        return True
    if not (src / DINOV2_CODE_LANDMARK).is_file() or _same_path(src, dst):
        return False
    tmp = dst.with_name(dst.name + ".partial")
    try:
        if tree_digest(src) != DINOV2_CODE_DIGEST:
            return False
        shutil.rmtree(tmp, ignore_errors=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        link = _same_volume(src, dst.parent)
        shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("__pycache__"),
                        copy_function=os.link if link else shutil.copy2)
        _replace_folder(tmp, dst)
    except OSError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        log(f"Could not take DINOv2's code from {src} ({e}); it is downloaded instead.")
        return False
    record_verified(dst, DINOV2_CODE_DIGEST)
    log(f"Found DINOv2's code in {src}: {'linked' if link else 'copied'} it instead of downloading it.")
    return True


def download_dinov2_code(dst: Path, log: Log) -> None:
    """DINOv2's code at DINOV2_CODE_COMMIT from GitHub (a zip of about 3 MB), unpacked into dst and checked file
    by file. Raises OSError."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    archive = dst.with_name("dinov2-code.zip")
    tmp = dst.with_name(dst.name + ".partial")
    req = urllib.request.Request(DINOV2_CODE_URL, headers={"User-Agent": "spellcast3d-installer"})
    problem = ""
    for attempt in range(DOWNLOAD_ATTEMPTS):
        if attempt:
            log(f"  DINOv2's code: {problem}; trying again ...")
            time.sleep(min(30, 5 * attempt))
        try:
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp, open(archive, "wb") as f:
                shutil.copyfileobj(resp, f, 1 << 20)
            break
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            problem = str(getattr(e, "reason", e)) or type(e).__name__
    else:
        archive.unlink(missing_ok=True)
        raise OSError(f"DINOv2's code: {problem}")
    try:
        shutil.rmtree(tmp, ignore_errors=True)
        top = f"dinov2-{DINOV2_CODE_COMMIT}/"
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                name = info.filename.replace("\\", "/")
                if not name.startswith(top) or name.endswith("/") or ".." in name.split("/"):
                    continue
                target = tmp / name[len(top):]
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as zsrc, open(target, "wb") as out:
                    shutil.copyfileobj(zsrc, out)
        if tree_digest(tmp) != DINOV2_CODE_DIGEST:
            raise OSError("DINOv2's code: the download is not that commit's code")
        _replace_folder(tmp, dst)
        record_verified(dst, DINOV2_CODE_DIGEST)
    except (OSError, zipfile.BadZipFile) as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise OSError(str(e)) from e
    finally:
        archive.unlink(missing_ok=True)


def _replace_folder(new: Path, dst: Path) -> None:
    """Put the finished folder `new` at dst, replacing an incomplete dst."""
    if dst.exists():
        shutil.rmtree(dst)
    os.replace(new, dst)


def _triflow_weight_in_place(name: str) -> bool:
    """code/triflow/checkpoints/<name> is TriFlow's weight: a hard link to models/triflow's, or a copy of it that
    link_triflow_checkpoints made (remembered with its size and time: record_verified)."""
    src, dst = paths.MODELS_TRIFLOW / name, paths.TRIFLOW_CHECKPOINTS / name
    try:
        if os.path.samefile(src, dst):
            return True
    except OSError:
        return False
    return is_recorded(dst, TRIFLOW_FILES[name][1])


def triflow_link_work() -> str | None:
    """Why code/triflow/checkpoints still needs TriFlow's weights linked from models/triflow, or None."""
    if not paths.TRIFLOW_CHECKPOINTS.is_dir():
        return None                 # TriFlow is not checked out: nothing to link into yet
    for name in TRIFLOW_FILES:
        if missing_files(paths.MODELS_TRIFLOW, {name: TRIFLOW_FILES[name]}):
            continue                # not downloaded yet: that is missing_files' part
        if not _triflow_weight_in_place(name):
            return "its AI models are not linked into code\\triflow\\checkpoints yet"
    return None


def link_triflow_checkpoints(log: Log) -> bool:
    """Hard-link TriFlow's weights from models/triflow into code/triflow/checkpoints, where TriFlow loads them.

    TriFlow's own checkpoints/.gitignore ignores them, so its checkout stays exactly its commit. On another drive
    (the artist moved models/triflow elsewhere), or where a hard link cannot be made (a FAT or exFAT drive, an
    antivirus that blocks it), they are copied, and the copy is remembered (record_verified), so it is not made
    again at the next start. Returns False when a file could not be put there.
    """
    ok = True
    for name in TRIFLOW_FILES:
        src, dst = paths.MODELS_TRIFLOW / name, paths.TRIFLOW_CHECKPOINTS / name
        if missing_files(paths.MODELS_TRIFLOW, {name: TRIFLOW_FILES[name]}) or not dst.parent.is_dir():
            continue
        if _triflow_weight_in_place(name):
            continue
        tmp = dst.with_name(dst.name + ".link")
        try:
            tmp.unlink(missing_ok=True)
            linked = False
            if _same_volume(src, dst.parent):
                try:
                    os.link(src, tmp)
                    linked = True
                except OSError as e:
                    log(f"Could not hard-link {name} into code\\triflow\\checkpoints ({e}); copying it instead "
                        f"({TRIFLOW_FILES[name][0] / 1e9:.1f} GB).")
            if not linked:
                shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            if not linked:
                record_verified(dst, TRIFLOW_FILES[name][1])
        except OSError as e:
            tmp.unlink(missing_ok=True)
            log(f"Could not put {name} into code\\triflow\\checkpoints ({e}).")
            ok = False
    return ok
