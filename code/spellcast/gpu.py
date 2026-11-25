"""NVIDIA GPU detection through nvidia-smi. STDLIB ONLY (used by the installer).

Facts this module encodes (verified while designing Spellcast3D):
- torch 2.8.0+cu128 on Windows ships kernels for sm_61..sm_120, so Forge runs
  on everything from a GTX 1080 to an RTX 50xx with that single build.
- Pixal3D runs from compute capability 6.1 (GTX 10xx): on sm_80+ (RTX 30xx and newer) as it was made (bf16 flows,
  flash_attn, SageAttention, natten, Triton); below that the fork's precision profile (pixal3d/modules/precision.py,
  round L) runs the flows in fp16 (Turing: sm_70-sm_79) or fp32 (Pascal), its attention on torch's SDPA,
  its int8 GEMMs through cuBLASLt, NAF in plain torch and the decoders on spconv - every compiled piece those cards
  need (cumesh, flex_gemm's CUDA part, o_voxel, nvdiffrast, spconv, torch itself) ships with sm_61+ kernels.
  Measured on an RTX 2060 Super and a GTX 1080 (Linux pods); Windows on those cards follows the same wheels.
- The retopology tools LATO.2 and TriFlow are allowed from compute capability 6.1 (GTX 10xx):
  their compiled parts (spconv, torchsparse, o_voxel, nvdiffrast) have sm_61 kernels, and their
  launchers fall back where the attention wheels have none (flash_attn is sm_80+, xformers sm_75+):
  torch's SDPA through a flash_attn shim, xformers on 7.5, torchsparse's GatherScatter dataflow
  below 7.5, fp16 / fp32 instead of bf16. Only sm_86 was tried; below sm_80 it is untested.
- Qwen-Image 2.1 runs on Spellcast3D's own CUDA 12.8 build of stable-diffusion.cpp (no PyTorch; install/sdcpp.py),
  made for the official release's list of compute capabilities, 6.1..12.0 (sm_61;70;75;80;86;89;90;100;120, each
  with its PTX): GTX 10xx and newer. Only sm_86 (an RTX 3060) was run; the other generations compile.
- cu128 wheels need an NVIDIA driver >= 570.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

PIXAL3D_MIN_CC = 6.1
LATO2_MIN_CC = 6.1
TRIFLOW_MIN_CC = 6.1
QWEN_MIN_CC = 6.1
MIN_DRIVER = 570.0


@dataclass
class GpuInfo:
    index: int
    name: str
    compute_cap: float          # e.g. 8.6
    vram_total_mb: int
    vram_used_mb: int
    driver_version: str         # e.g. "596.36"
    uuid: str = ""              # e.g. "GPU-5b2e...": the card itself, whatever its index (empty when unknown)

    @property
    def sm(self) -> str:
        major, _, minor = f"{self.compute_cap:.1f}".partition(".")
        return f"sm_{major}{minor}"

    @property
    def driver_major(self) -> float:
        try:
            return float(".".join(self.driver_version.split(".")[:2]))
        except ValueError:
            return 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sm"] = self.sm
        return d


def _nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for candidate in (Path(r"C:\Windows\System32\nvidia-smi.exe"),
                      Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")):
        if candidate.exists():
            return str(candidate)
    return None


def query_gpus(timeout: float = 10.0) -> list[GpuInfo]:
    """All NVIDIA GPUs, or [] when there is no driver / no nvidia-smi."""
    exe = _nvidia_smi()
    if exe is None:
        return []
    fields = "index,name,compute_cap,memory.total,memory.used,driver_version,uuid"
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    gpus: list[GpuInfo] = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) not in (6, 7):
            continue
        try:
            gpus.append(GpuInfo(
                index=int(parts[0]), name=parts[1], compute_cap=float(parts[2]),
                vram_total_mb=int(float(parts[3])), vram_used_mb=int(float(parts[4])),
                driver_version=parts[5], uuid=parts[6] if len(parts) == 7 else "",
            ))
        except ValueError:
            continue
    return gpus


def primary_gpu() -> GpuInfo | None:
    """The GPU the tools will use (CUDA device 0 == nvidia-smi index 0 by default)."""
    gpus = query_gpus()
    return gpus[0] if gpus else None


def pixal3d_supported(gpu: GpuInfo | None) -> tuple[bool, str]:
    if gpu is None:
        return False, "No NVIDIA GPU was found. Pixal3D needs an NVIDIA GTX 1060 or newer."
    if gpu.compute_cap < PIXAL3D_MIN_CC:
        return False, (f"{gpu.name} ({gpu.sm}) is not supported by Pixal3D: it needs a "
                       f"GTX 1060 or newer (compute capability 6.1+).")
    return True, ""


def pixal3d_card_env(gpu: GpuInfo | None) -> dict[str, str]:
    """What Pixal3D's processes are told about the card: PIXAL3D_CARD_SM (its compute capability as two digits), so
    that a process which cannot see the card (the installer's copies of the flow models, made with the card hidden)
    still takes the card's precision profile (pixal3d/modules/precision.py); below sm_70, DISABLE_ADDMM_CUDA_LT=1,
    which keeps torch's fused linear-with-bias off cuBLASLt, whose fp32 SGEMM reads out of bounds on a GTX 1080 (an
    illegal memory access a few seconds into the first flow; the fork's precision profile sets it as well)."""
    if gpu is None:
        return {}
    env = {"PIXAL3D_CARD_SM": gpu.sm[3:]}
    if gpu.compute_cap < 7.0:
        env["DISABLE_ADDMM_CUDA_LT"] = "1"
    return env


def lato2_supported(gpu: GpuInfo | None) -> tuple[bool, str]:
    return _retopo_supported(gpu, "LATO.2", LATO2_MIN_CC)


def triflow_supported(gpu: GpuInfo | None) -> tuple[bool, str]:
    return _retopo_supported(gpu, "TriFlow", TRIFLOW_MIN_CC)


def _retopo_supported(gpu: GpuInfo | None, name: str, min_cc: float) -> tuple[bool, str]:
    if gpu is None:
        return False, (f"No NVIDIA GPU was found. {name} needs an NVIDIA GTX 1060 or newer; Instant Meshes "
                       f"(on the CPU) still makes clean meshes.")
    if gpu.compute_cap < min_cc:
        # plain words for the artist (the page shows it on the method's button); the log has gpu.sm
        return False, (f"Your graphics card ({gpu.name}) is too old for {name}: it needs a GTX 1060 or newer. "
                       f"Instant Meshes still makes clean meshes.")
    return True, ""


def qwen_supported(gpu: GpuInfo | None) -> tuple[bool, str]:
    # No NVIDIA GPU is final for this PC (recorded as unsupported, re-checked at each start), as for Forge.
    if gpu is None:
        return False, ("No NVIDIA graphics card was found (or its driver is not installed). Qwen-Image 2.1 needs "
                       "one to make images; you can still bring your own images. If this PC has an NVIDIA card, "
                       "install its latest driver from https://www.nvidia.com/drivers and start Spellcast3D again: "
                       "Qwen-Image 2.1 is then installed by itself.")
    if gpu.compute_cap < QWEN_MIN_CC:
        return False, (f"Your graphics card ({gpu.name}) is too old for Qwen-Image 2.1: it needs a GTX 1060 or newer. "
                       f"You can still bring your own images.")
    return True, ""


def driver_ok(gpu: GpuInfo | None) -> tuple[bool, str]:
    if gpu is None:
        return False, "No NVIDIA GPU / driver detected."
    if gpu.driver_major < MIN_DRIVER:
        return False, (f"NVIDIA driver {gpu.driver_version} is too old; version {MIN_DRIVER:.0f} "
                       f"or newer is required. Please update your GPU driver.")
    return True, ""


if __name__ == "__main__":
    import json
    print(json.dumps([g.to_dict() for g in query_gpus()], indent=2))
