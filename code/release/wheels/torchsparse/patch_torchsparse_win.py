"""Minimal Windows (MSVC + nvcc) patch for a CLEAN copy of mit-han-lab/torchsparse @ 385f5ce.

Apply it to an exported copy of the source (never to a git checkout you want to keep pristine):

    python patch_torchsparse_win.py <copy-of-torchsparse> [--local-version cu128torch2.8]

Every hunk is an exact, anchored text replacement; the script fails if an anchor is missing or
ambiguous, so a different upstream commit cannot be patched silently. It is idempotent-safe only in
the sense that it refuses to patch an already-patched tree (anchors no longer match).

Equivalence with Deathdadev/torchsparse (MIT, commit f1787ee "Add Windows compilation compatibility
fixes", contained in bebadc4): P1-P3 are the same edits byte for byte; P4 uses int64_t where the fork
uses `long long` (the same type under MSVC, and still correct on Linux). Two fork edits are NOT taken:
  * hash_cpu.cpp: the fork switched the CPU hash input from `int` to `int64_t`. torchsparse's Python
    side asserts int32 coords (`nn/functional/hash.py`: `assert coords.dtype == torch.int`), so
    `data_ptr<int64_t>()` would raise on every CPU hash. Upstream `int` compiles fine with MSVC.
  * hash_cpu.cpp / hash_cuda.cu: `uint64_t` -> `unsigned long long` is a no-op under MSVC
    (uint64_t is a typedef of unsigned long long there).
P5 (optional, --local-version) only changes the wheel's version label (PEP 440 local segment); the
runtime `torchsparse.__version__` stays "2.1.0".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PIN = "385f5ce8718fcae93540511b7f5832f4e71fd835"

ASM_BLOCK = ("#ifdef _MSC_VER // MSVC compiler (used by nvcc on Windows)\n"
             "#define __asm__ asm\n"
             "#define __volatile__ volatile\n"
             "#endif\n\n")

IMPLICIT_GEMM_FILES = [
    "convolution_backward_wgrad_implicit_gemm_cuda",
    "convolution_backward_wgrad_implicit_gemm_sorted_cuda",
    "convolution_forward_implicit_gemm_cuda",
    "convolution_forward_implicit_gemm_sorted_cuda",
]

# (id, relative path, old text, new text, why)
HUNKS: list[tuple[str, str, str, str, str]] = [
    ("P1", "setup.py",
     'import glob\nimport os\n\nimport torch\n',
     'import glob\nimport os\nimport platform\n\nimport torch\n',
     "setup.py: import platform (for P1b)"),
    ("P1b", "setup.py",
     'extra_compile_args = {\n'
     '    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp"],\n'
     '    "nvcc": ["-O3", "-std=c++17"],\n'
     '}\n',
     '\n# Platform-specific compiler arguments\n'
     'if platform.system() == "Windows":\n'
     '    extra_compile_args = {\n'
     '        "cxx": ["/MD", "/O2", "/EHsc"],\n'
     '        "nvcc": ["-O3", "-std=c++17"],\n'
     '    }\n'
     'else:\n'
     '    extra_compile_args = {\n'
     '        "cxx": ["-g", "-O3", "-fopenmp", "-lgomp"],\n'
     '        "nvcc": ["-O3", "-std=c++17"],\n'
     '    }\n',
     "setup.py: MSVC flags instead of the GCC-only -g/-O3/-fopenmp/-lgomp (cl would ignore them with "
     "D9002 warnings; no /openmp, so the CPU-only fallbacks run single-threaded)"),
]
for _name in IMPLICIT_GEMM_FILES:
    HUNKS.append((
        "P2", f"torchsparse/backend/convolution/{_name}.cu",
        f'#include <torch/extension.h>\n#include "{_name}.h"\n',
        ASM_BLOCK + f'#include <torch/extension.h>\n#include "{_name}.h"\n',
        "implicit-GEMM kernels: map GNU __asm__/__volatile__ to asm/volatile, which nvcc accepts in MSVC mode"))
HUNKS += [
    ("P3", "torchsparse/backend/convolution/convolution_gather_scatter_cuda.cu",
     '#include <cuda.h>\n#include <cuda_runtime.h>\n',
     '#include <cuda.h>\n#include <cuda_fp16.h>\n#include <cuda_runtime.h>\n',
     "gather/scatter: include cuda_fp16.h explicitly (half2/__hadd2 are not pulled in transitively on Windows)"),
    ("P4", "torchsparse/backend/others/downsample_cuda.cu",
     '      n_out_points, _out_coords_transformed.data_ptr<long>());\n',
     '      n_out_points, _out_coords_transformed.data_ptr<int64_t>());\n',
     "downsample: data_ptr<long> is not instantiated by libtorch on Windows (long is 32-bit); "
     "the kernel takes int64_t*"),
    ("P4", "torchsparse/backend/others/downsample_cuda.cu",
     '      num_out_points, _out_coords_transformed.data_ptr<long>(),\n',
     '      num_out_points, _out_coords_transformed.data_ptr<int64_t>(),\n',
     "downsample: same as above, second call"),
]


def local_version_hunk(label: str) -> tuple[str, str, str, str, str]:
    return ("P5", "setup.py", '    version=version,\n', f'    version=version + "+{label}",\n',
            f"wheel version label 2.1.0+{label} (metadata only; torchsparse.__version__ stays 2.1.0)")


def apply(root: Path, local_version: str | None) -> list[dict]:
    hunks = list(HUNKS) + ([local_version_hunk(local_version)] if local_version else [])
    # Validate every anchor against the pristine text first, so a failure leaves the tree untouched.
    texts: dict[str, str] = {}
    for hid, rel, old, _new, _why in hunks:
        p = root / rel
        if rel not in texts:
            if not p.is_file():
                sys.exit(f"[{hid}] {rel} not found under {root}")
            texts[rel] = p.read_bytes().decode("utf-8")
        n = texts[rel].count(old)
        if n != 1:
            sys.exit(f"[{hid}] anchor found {n}x (expected 1) in {rel}: {old[:60]!r}")
    applied = []
    for hid, rel, old, new, why in hunks:
        texts[rel] = texts[rel].replace(old, new, 1)
        applied.append({"id": hid, "file": rel, "why": why})
    for rel, t in texts.items():
        (root / rel).write_bytes(t.encode("utf-8"))
    return applied


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--local-version", default=None)
    a = ap.parse_args()
    for h in apply(a.root, a.local_version):
        print(f"[{h['id']}] {h['file']}: {h['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
