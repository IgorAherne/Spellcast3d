"""What every check of the Pixal3D low-memory work shares: the process set up as the reference path sets it up, the
golden tensors, the comparison rules, memory peaks and caps, and the GPU lock.

A check script lies in this folder (or in code/devtools/pixal3d-lowmem, with this folder put on sys.path) and is
run in the pixal3d venv:

    import common                                        # first: before torch and before anything of Pixal3D
    args = common.parser("what this check proves").parse_args()
    common.boot(args.repo)                               # the fork worktree under test
    import torch
    case = common.golden(args.case)                      # golden/<case> of the reference path
    with common.gpu_lock("WP1", f"check_x {args.case}", expected_minutes=3):
        dino = common.builders(args.repo)["dino"]("cuda")
        with common.peak() as mem:
            got = ...                                    # the piece under test, fed the case's golden inputs
        results = [common.compare("07-cond_hr/proj", got, case.tensor("07-cond_hr", "proj"), case.spread)]
        with common.capped(2048) as cap:                 # the same again under a tight cap: equal values, cap.ooms == 0
            ...
    print(f"peak {mem.allocated_mib} MiB allocated, {mem.reserved_mib} MiB reserved")
    sys.exit(common.report(results))

The rules `compare` applies (the specification's section 4.3):
1. Integers and booleans (coords, faces, subs, res): torch.equal, always.
2. Floats: torch.equal - the same bits, dtype and shape - wherever the case's spread.json says the reference path
   itself reproduced that tensor bit for bit in all its captures, and wherever spread.json has no entry for it.
3. Floats the reference does not reproduce bit for bit: max |got - want| <= the spread's max_abs; both are printed.
A tensor in another dtype or shape than the golden one fails: cast the golden tensor the way the code under test
is specified to (`want.to(torch.bfloat16)` for a stored conditioning), never the result.

The launcher every check on the golden cases imports is the FROZEN copy in golden/launcher (hub commit 4ee3e36),
never the live one in code/spellcast/launchers, which is being rewritten while checks run. GOLD (the golden cases
and that launcher, code/logs/profiling/pixal3d-lowmem/golden, git-ignored) and the worktree's MODELS junction are
needed only by the checks on the golden cases and on the card: boot(repo, golden=False) - check_store.py on the
CPU - takes the launcher's settings from the live launcher when GOLD is not there (configure_env is the same in
both), and check_engine.py --fake does not boot at all.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import dataclasses
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

HERE = Path(__file__).resolve().parent
PROF = HERE.parent
HUB = PROF.parents[2]
GOLD = HUB / "code" / "logs" / "profiling" / "pixal3d-lowmem" / "golden"
LAUNCHER = GOLD / "launcher"
LIVE_LAUNCHERS = HUB / "code" / "spellcast" / "launchers"
if str(PROF) not in sys.path:           # gpu_lock.py and toolkit.py, one module object each for every check
    sys.path.append(str(PROF))

import gpu_lock as _lock  # noqa: E402  (stdlib only)
import toolkit  # noqa: E402  (stdlib only)

MIB = 1 << 20
CHUNK = toolkit.CHUNK               # bytes hashed at a time

# What the checks on the card allow to stay there after a prop or a batch. Exact figures, measured with torch 2.8.0 /
# CUDA 12.8 on driver 596.36 (RTX 3060): they fail safe - a library that keeps more after an upgrade makes the
# checks fail, never pass something they should not - but they are not laws. When a torch, CUDA or driver upgrade
# moves them, measure again before reading the failure as a leak: cuBLAS's work area is what
# torch.cuda.memory_allocated() grows by over a process's first matrix product (check_engine.py --case sv1024 prints
# the bytes allocated at every finish hook and after the batch), and the tables are what stays after a batch with
# nothing of a prop alive; set the new numbers here with the versions they were measured with.
CUBLAS_WORKSPACE = 8_519_680        # bytes cuBLAS keeps after a process's first matrix product (torch 2.8, cu128)
# Libraries keep small tables on the card for the life of the process: DINOv3's patch-centre grids in an lru_cache
# (transformers, 40 KiB for the 512 and 1024 px pictures) and a few index tables. Measured: 43,008 B after every
# batch, one prop or four, so they do not grow with the props. While a batch runs cuBLAS also holds work area
# beyond its first 8 MiB, sized by the matrix products that ran: 0.88-1.17 MiB measured, held by no Python object
# and given back by engine.release() with the rest of cuBLAS's work area.
LIBRARY_TABLES_AFTER = 64 * 1024
LIBRARY_TABLES_DURING = 2 * 1024 * 1024
BOUNDS_HINT = ("if torch, CUDA or the driver changed since these bounds were measured (torch 2.8.0, CUDA 12.8, "
               "driver 596.36), measure them again before reading this as a leak: see CUBLAS_WORKSPACE in "
               "p3d_checks/common.py")

_booted: dict[str, Any] = {}
_peaks: list["Peak"] = []


def kept_on_card() -> int:
    """Bytes the staged engine keeps on the card between props on purpose: cuBLAS's work area and the projection
    grids the next wave's conditioning would make again. Anything a finish hook finds beyond these and the
    libraries' tables is a prop's or a decoder's."""
    from pixal3d.pipelines import proj_cond
    return CUBLAS_WORKSPACE + sum(buffer.nbytes for grid in proj_cond._grids.values() for buffer in grid.buffers())


gpu_lock = _lock.gpu_lock


def golden_root() -> Path:
    """The folder of the golden cases (P3D_GOLD overrides it: tests, a second capture to compare with)."""
    return Path(os.environ.get("P3D_GOLD") or GOLD)


def parser(description: str = "") -> argparse.ArgumentParser:
    """The arguments every check takes; a check adds its own."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--repo", required=True, type=lambda text: Path(text).resolve(),
                   help="the fork worktree under test, e.g. .claude/worktrees/p3d/wp1")
    p.add_argument("--case", default="sv1024", help="the golden case (sv1024, mv1024, sv1536, mv1024b, redraw, "
                                                    "big1536)")
    p.add_argument("--full", action="store_true", help="the whole stage instead of the quick form (all sampler "
                                                       "steps instead of the first two)")
    return p


# =====================================================================================================================
# The process: environment, working directory, imports - as the reference path's launcher makes them
# =====================================================================================================================

def boot(repo: str | Path, *, low_vram: bool = True, attn_backend: str = "flash_attn", golden: bool = True) -> None:
    """Make this process what Pixal3D's launcher makes of its own before it imports Pixal3D: the frozen launcher's
    configure_env (model caches, offline mode, memory settings), the variables app.py sets when it is imported,
    the worktree as working directory and first import path, NAF loaded from the local checkout, and flex_gemm
    pointed at a private copy of the kernel choices the golden captures ran with.

    golden=False is for the checks that need neither the golden cases nor the model files (check_store.py on the
    CPU): no MODELS junction is asked for, and without GOLD the live launcher's configure_env is used (the same
    settings as the frozen copy's).

    Call it before torch is imported: two of the settings are read when torch loads. A second call for the same
    worktree does nothing; another worktree in the same process is refused (its modules are already imported)."""
    repo = Path(repo).resolve()
    if _booted:
        if _booted["repo"] != repo:
            raise RuntimeError(f"this process was booted for {_booted['repo']}; start another one for {repo}")
        return
    if not (repo / "app.py").is_file() or not (repo / "pixal3d").is_dir():
        raise FileNotFoundError(f"{repo} is not a Pixal3D checkout (app.py, pixal3d/)")
    if golden and not (repo / "MODELS").is_dir():
        raise FileNotFoundError(f"{repo} has no MODELS junction to the hub's models/pixal3d folder")
    if "torch" in sys.modules:
        raise RuntimeError("common.boot(repo) must run before torch is imported (import common first, torch after)")
    for name in ("pixal3d", "pixal3d_server", "pixal3d_mv", "pixal3d_guard", "app"):
        if name in sys.modules:
            raise RuntimeError(f"`{name}` was imported before common.boot(repo): it would not be the frozen "
                               f"launcher's / this worktree's")
    launcher_dir = LAUNCHER
    if not (LAUNCHER / "pixal3d_server.py").is_file():
        if golden:
            raise FileNotFoundError(f"the frozen launcher is missing: {LAUNCHER} (GOLD: pixal3d_golden.py in "
                                    f"code/devtools/pixal3d-lowmem makes it)")
        launcher_dir = LIVE_LAUNCHERS
        print(f"[common] NOTE: no frozen launcher ({LAUNCHER}): its settings are taken from the live one in "
              f"{launcher_dir}", flush=True)

    sys.path.insert(0, str(launcher_dir))
    import pixal3d_server as srv

    torch_home = srv.configure_env(repo, low_vram, attn_backend)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"          # app.py:28-31
    os.environ.setdefault("ATTN_BACKEND", "flash_attn")
    # The flows' fused elementwise kernels (fork: pixal3d/modules/fused.py) are on by default and change the
    # rounding; the goldens were captured by the eager ops. Off here, unless the environment says otherwise.
    os.environ.setdefault("PIXAL3D_FUSED", "off")
    # SageAttention (fork: pixal3d/modules/sage.py) is on by default too and changes the attention's rounding.
    os.environ.setdefault("PIXAL3D_SAGE", "off")
    # The flows' linear layers as int8 GEMMs (fork: pixal3d/modules/int8.py) change every GEMM's rounding.
    os.environ.setdefault("PIXAL3D_INT8", "off")
    # DINOv3 in float16 (fork: pipelines/proj_cond.py) is on by default on sm_75+ and changes the tokens' rounding.
    os.environ.setdefault("PIXAL3D_DINO_FP16", "off")
    os.environ.setdefault("PIXAL3D_MATTE_FP16", "off")    # the matte in float16 (fork: staged/builders.py)
    os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")
    source = golden_root() / "autotune_cache.json"
    if not source.is_file():
        source = repo / "autotune_cache.json"
        print(f"[common] NOTE: {golden_root() / 'autotune_cache.json'} does not exist yet; flex_gemm's kernel "
              f"choices are taken from {source}", flush=True)
    private = repo / "tmp" / "p3d-checks" / str(os.getpid())
    private.mkdir(parents=True, exist_ok=True)
    atexit.register(shutil.rmtree, private, ignore_errors=True)
    choices = source.read_bytes() if source.is_file() else None
    if choices is not None:
        (private / "autotune_cache.json").write_bytes(choices)
    os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = str(private / "autotune_cache.json")
    # The install's calibration record (fork: pixal3d/calibration.py, written by `python -m pixal3d.calibrate`) is not
    # the goldens': with it a convolution size the private cache lacks would take the nearest calibrated size's
    # choice (modules/sparse/conv/tuning.py) instead of being tuned, and the Sage step would skip its timing. The
    # checks run as an uncalibrated install does: the record is a file that does not exist. A tool that wants the
    # install's record sets PIXAL3D_CALIBRATION_FILE again after boot, before it imports Pixal3D.
    os.environ["PIXAL3D_CALIBRATION_FILE"] = str(private / "no-calibration-record.json")

    os.chdir(repo)
    sys.path.insert(0, str(repo))
    srv.load_naf_locally(torch_home)
    srv.quiet_expandable_segments()
    _booted.update(repo=repo, launcher=srv, torch_home=torch_home, autotune_source=source,
                   autotune_choices=choices, autotune_private=private / "autotune_cache.json")


def launcher() -> Any:
    """The frozen launcher module (golden/launcher/pixal3d_server.py) boot imported."""
    if not _booted:
        raise RuntimeError("call common.boot(repo) first")
    return _booted["launcher"]


def autotune_changed() -> bool:
    """Did flex_gemm tune a kernel in this process (it then rewrote its private cache file). The kernel it picks
    decides the order of the sums inside the two sparse decoders, so their output need not match the golden run."""
    if not _booted or not _booted["autotune_private"].is_file():
        return False
    return _booted["autotune_private"].read_bytes() != _booted["autotune_choices"]


def builders(repo: str | Path) -> dict[str, Callable[[str], Any]]:
    """The four small models' builders of the worktree given: {"rembg", "moge", "dino", "naf"}, each
    `build(device) -> module on it` (pixal3d/staged/builders.py: what the engine and the launcher use too)."""
    boot(repo)
    from pixal3d.staged import builders as module

    if Path(module.__file__).resolve().parents[2] != _booted["repo"]:
        raise RuntimeError(f"pixal3d was imported from {module.__file__}, not from {_booted['repo']}")
    return module.default_builders(module.default_paths(_booted["launcher"].models_folder(_booted["repo"])))


# =====================================================================================================================
# Golden cases
# =====================================================================================================================

class Case:
    """One golden case: golden/<case>/ with manifest.json, request.json, spread.json and NN-<stage>.safetensors."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.name = self.folder.name
        manifest = self.folder / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"the golden case {self.name} is not captured yet ({manifest} does not exist)")
        self.manifest: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
        self.request: dict[str, Any] = self._json("request.json")
        self.spread: dict[str, Any] = self._json("spread.json")

    def _json(self, name: str) -> dict[str, Any]:
        path = self.folder / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def files(self) -> list[str]:
        """The stage files of the case, without the extension: ["00-input", "01-dino", ...]."""
        return sorted(path.stem for path in self.folder.glob("*.safetensors"))

    def path(self, file: str) -> Path:
        """The stage file "08-hr" (also given as "08-hr.safetensors")."""
        stem = file[:-len(".safetensors")] if file.endswith(".safetensors") else file
        for candidate in (stem, stem.replace("_", "-"), stem.replace("-", "_")):
            path = self.folder / f"{candidate}.safetensors"
            if path.is_file():
                return path
        raise FileNotFoundError(f"the golden case {self.name} has no stage file {stem!r}; it has {self.files()}")

    def names(self, file: str) -> dict[str, tuple[str, list[int]]]:
        """tensor name -> (safetensors dtype code, shape) of one stage file, read from its header alone."""
        with open(self.path(file), "rb") as handle:
            header = json.loads(handle.read(int.from_bytes(handle.read(8), "little")))
        return {name: (info["dtype"], info["shape"]) for name, info in header.items() if name != "__metadata__"}

    def tensor(self, file: str, name: str, device: str = "cpu") -> Any:
        """One golden tensor, e.g. case.tensor("08-hr", "noise", "cuda"). Only that tensor is read from the file."""
        from safetensors import safe_open

        path = self.path(file)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if name not in handle.keys():
                raise KeyError(f"{self.name}/{path.stem} has no tensor {name!r}; it has {sorted(handle.keys())}")
            value = handle.get_tensor(name)
        return value if str(device) == "cpu" else value.to(device)

    def tensors(self, file: str, device: str = "cpu") -> dict[str, Any]:
        """Every tensor of one stage file."""
        return {name: self.tensor(file, name, device) for name in self.names(file)}

    def input(self, relative: str) -> Path:
        """A path the request names (they are relative to the case folder)."""
        return self.folder / relative

    def recorded_sha(self, file: str, name: str) -> str | None:
        """The SHA-256 the manifest records for a tensor, wherever its layout keeps it; None when it has none."""
        stem = self.path(file).stem
        keys = (f"{stem}/{name}", f"{stem}.safetensors/{name}", f"{stem}:{name}")
        for table in (self.manifest.get("sha256"), self.manifest.get("tensors"), self.manifest.get("files"),
                      self.manifest.get("stages")):
            if not isinstance(table, dict):
                continue
            entries = [table.get(key) for key in keys]
            nested = table.get(stem) or table.get(f"{stem}.safetensors")
            if isinstance(nested, dict):
                entries += [nested.get(name), (nested.get("tensors") or {}).get(name)
                            if isinstance(nested.get("tensors"), dict) else None]
            for entry in entries:
                if isinstance(entry, str):
                    return entry
                if isinstance(entry, dict) and isinstance(entry.get("sha256"), str):
                    return entry["sha256"]
        return None

    def verify(self) -> list[str]:
        """Read every tensor of the case and compare it with the SHA-256 its manifest records. The problems found,
        one line each (an empty list: every recorded hash matches; tensors without a recorded hash are counted in
        the last line only when there are any)."""
        problems: list[str] = []
        unrecorded = 0
        for file in self.files():
            for name in self.names(file):
                want = self.recorded_sha(file, name)
                if want is None:
                    unrecorded += 1
                elif sha256_of(self.tensor(file, name)) != want:
                    problems.append(f"{file}/{name}: the file's tensor does not have the manifest's SHA-256")
        if unrecorded:
            problems.append(f"{unrecorded} tensors have no SHA-256 in the manifest")
        return problems


def golden(case: str, root: str | Path | None = None) -> Case:
    """The golden case of that name (sv1024, mv1024, sv1536, mv1024b, redraw, big1536)."""
    return Case(Path(root) / case if root is not None else golden_root() / case)


sha256_of = toolkit.sha256_of     # SHA-256 of a tensor's values in row-major order: the golden manifests' hash


def state_digest(state: Any) -> dict[str, str]:
    """name -> "dtype shape sha256" for every tensor of a module's state_dict (or of a dict of tensors): two
    modules hold the same weights exactly when these dicts are equal."""
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    return {name: f"{str(value.dtype).replace('torch.', '')} {list(value.shape)} {sha256_of(value)}"
            for name, value in state.items()}


# =====================================================================================================================
# Comparison
# =====================================================================================================================

@dataclasses.dataclass
class Result:
    name: str
    ok: bool
    rule: str                       # "integers", "bit-equal", "spread", "form"
    detail: str = ""
    max_abs: float | None = None    # largest |got - want| (None when the forms differ)
    allowed: float | None = None    # the spread's max_abs under rule 3
    differing: int | None = None    # how many values differ
    first: list[int] | None = None  # index of the first differing value

    def __str__(self) -> str:
        return f"{'PASS' if self.ok else 'FAIL'}  {self.name:<34} {self.rule:<10} {self.detail}"


def spread_entry(spread: Any, name: str) -> dict[str, Any] | None:
    """The entry of one tensor ("08-hr/slat") in a case's spread.json: {"equal": bool, "max_abs": float}, or None
    when the spread says nothing about it. `spread` may also be that entry itself, or a number (the max_abs)."""
    if spread is None:
        return None
    if isinstance(spread, (int, float)) and not isinstance(spread, bool):
        return {"equal": spread == 0, "max_abs": float(spread)}
    if not isinstance(spread, dict):
        raise TypeError(f"a spread is a dict or a number, not {type(spread).__name__}")
    if isinstance(spread.get("equal"), bool) or isinstance(spread.get("max_abs"), (int, float)):
        return spread
    file, _, tensor = name.partition("/")
    for table in (spread, spread.get("tensors")):
        if not isinstance(table, dict):
            continue
        for stem in (file, file + ".safetensors", file.replace("-", "_")):
            for entry in (table.get(f"{stem}/{tensor}"), table.get(f"{stem}:{tensor}"),
                          (table.get(stem) or {}).get(tensor) if isinstance(table.get(stem), dict) else None):
                if isinstance(entry, dict):
                    return entry
    return None


def _unravel(flat: int, shape: Any) -> list[int]:
    index: list[int] = []
    for size in reversed(tuple(shape)):
        flat, here = divmod(flat, size)
        index.append(here)
    return index[::-1]


def compare(name: str, got: Any, want: Any, spread: Any = None) -> Result:
    """One tensor of the code under test against the golden one, by the rules at the top of this file.
    `name` is "<stage file>/<tensor>" ("08-hr/slat"); `spread` is the case's spread (case.spread), the tensor's
    own entry, or None (bit-equality is then required)."""
    import torch

    got = torch.as_tensor(got).detach().cpu()
    want = torch.as_tensor(want).detach().cpu()
    if got.is_complex() and want.is_complex():
        got, want = torch.view_as_real(got), torch.view_as_real(want)
    form = f"{list(want.shape)} {str(want.dtype).replace('torch.', '')}"
    if got.shape != want.shape or got.dtype != want.dtype:
        return Result(name, False, "form", f"got {list(got.shape)} {str(got.dtype).replace('torch.', '')}, the golden "
                                           f"tensor is {form}")
    floats = want.is_floating_point()
    entry = spread_entry(spread, name) if floats else None
    if torch.equal(got, want):
        return Result(name, True, "bit-equal" if floats else "integers", form, max_abs=0.0, differing=0)

    unequal = got != want
    differing = int(unequal.sum())
    first = _unravel(int(unequal.reshape(-1).to(torch.uint8).argmax()), got.shape) if differing else None
    if floats:
        a, b = (got, want) if got.dtype in (torch.float32, torch.float64) else (got.float(), want.float())
        max_abs = float((a - b).abs().max()) if got.numel() else 0.0
        where = (f"{differing} of {got.numel()} values differ, max abs {max_abs:.3e}, first at {first} "
                 f"(got {float(a[tuple(first)]):.9g}, golden {float(b[tuple(first)]):.9g})")
        if entry is not None and entry.get("equal") is False:
            allowed = entry.get("max_abs")
            if not isinstance(allowed, (int, float)):
                return Result(name, False, "spread", f"{form}: {where}; spread.json has no max_abs for this tensor",
                              max_abs=max_abs, differing=differing, first=first)
            ok = max_abs <= float(allowed)                              # False for a NaN as well
            return Result(name, ok, "spread", f"{form}: max abs {max_abs:.3e} {'<=' if ok else '>'} the reference's "
                          f"own run-to-run spread {float(allowed):.3e} ({differing} of {got.numel()} values differ, "
                          f"first at {first})", max_abs=max_abs, allowed=float(allowed), differing=differing,
                          first=first)
        why = ("the reference reproduces this tensor bit for bit" if entry is not None
               else "no spread recorded for it: bit-equality is required")
        return Result(name, False, "bit-equal", f"{form}: {where}; {why}", max_abs=max_abs, differing=differing,
                      first=first)
    a, b = got.to(torch.int64), want.to(torch.int64)
    return Result(name, False, "integers", f"{form}: {differing} of {got.numel()} values differ, first at {first} "
                  f"(got {int(a[tuple(first)])}, golden {int(b[tuple(first)])})",
                  max_abs=float((a - b).abs().max()), differing=differing, first=first)


def report(results: Iterable[Result]) -> int:
    """Print every result and the verdict. The exit code: 0 when all passed, 1 otherwise (also for no results)."""
    results = list(results)
    for result in results:
        print(result, flush=True)
    failed = [result for result in results if not result.ok]
    if autotune_changed():
        print("NOTE  flex_gemm tuned a sparse-convolution kernel in this run (its cache had no entry for a shape): "
              "the decoders' sums may be in another order than in the golden run.", flush=True)
    if not results:
        print("FAILED: nothing was compared.", flush=True)
        return 1
    if failed:
        print(f"FAILED: {len(failed)} of {len(results)} comparisons; first: {failed[0].name}", flush=True)
        return 1
    print(f"PASSED: all {len(results)} comparisons.", flush=True)
    return 0


# =====================================================================================================================
# Video memory: peaks, caps
# =====================================================================================================================

@dataclasses.dataclass
class Peak:
    """PyTorch's memory on the card around a block, in MiB. Filled when the block ends."""
    allocated_mib: int = 0          # the highest allocated
    reserved_mib: int = 0           # the highest reserved (what the per-process cap limits)
    start_allocated_mib: int = 0    # at the start of the block (models already on the card, ...)
    start_reserved_mib: int = 0
    end_allocated_mib: int = 0
    end_reserved_mib: int = 0
    ooms: int = 0                   # out-of-memory events PyTorch counted inside the block
    seconds: float = 0.0
    _allocated: int = 0
    _reserved: int = 0


def _ooms() -> int:
    import torch

    return int(torch.cuda.memory_stats().get("num_ooms", 0))


@contextlib.contextmanager
def peak() -> Iterator[Peak]:
    """PyTorch's allocated AND reserved peaks around a block: `with common.peak() as mem: ...`, then
    mem.allocated_mib / mem.reserved_mib. Blocks may be nested: an inner block's peak counts for the outer one."""
    import torch

    torch.cuda.synchronize()
    for outer in _peaks:
        outer._allocated = max(outer._allocated, torch.cuda.max_memory_allocated())
        outer._reserved = max(outer._reserved, torch.cuda.max_memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    mem = Peak(start_allocated_mib=torch.cuda.memory_allocated() // MIB,
               start_reserved_mib=torch.cuda.memory_reserved() // MIB)
    ooms, began = _ooms(), time.perf_counter()
    _peaks.append(mem)
    try:
        yield mem
    finally:
        torch.cuda.synchronize()
        _peaks.pop()
        mem.seconds = time.perf_counter() - began
        mem._allocated = max(mem._allocated, torch.cuda.max_memory_allocated())
        mem._reserved = max(mem._reserved, torch.cuda.max_memory_reserved())
        for outer in _peaks:
            outer._allocated = max(outer._allocated, mem._allocated)
            outer._reserved = max(outer._reserved, mem._reserved)
        mem.allocated_mib, mem.reserved_mib = mem._allocated // MIB, mem._reserved // MIB
        mem.end_allocated_mib = torch.cuda.memory_allocated() // MIB
        mem.end_reserved_mib = torch.cuda.memory_reserved() // MIB
        mem.ooms = _ooms() - ooms


@dataclasses.dataclass
class Cap:
    mib: int                        # what PyTorch may reserve inside the block
    ooms: int = 0                   # out-of-memory events PyTorch counted inside the block (filled at its end):
    #                                 above 0 with no exception means something caught one and went on another way


@contextlib.contextmanager
def capped(mib: int) -> Iterator[Cap]:
    """Limit what PyTorch may RESERVE on the card to `mib` for the block (set_per_process_memory_fraction), and
    put the previous limit back after. PyTorch then reuses its cache before it grows and raises
    torch.OutOfMemoryError instead of spilling into shared memory. `cap.ooms` is the growth of PyTorch's own
    out-of-memory counter over the block. Memory already reserved when the block starts counts against the cap:
    call gc.collect() and torch.cuda.empty_cache() first when the block should start clean."""
    import torch

    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    before = torch.cuda.get_per_process_memory_fraction()
    fraction = min(1.0, mib * MIB / total)
    cap = Cap(mib=int(fraction * total) // MIB)
    ooms = _ooms()
    torch.cuda.set_per_process_memory_fraction(fraction)
    try:
        yield cap
    finally:
        torch.cuda.set_per_process_memory_fraction(before)
        cap.ooms = _ooms() - ooms


def lowest_cap(run: Callable[[], Any], low_mib: int, high_mib: int, step_mib: int = 256) -> int:
    """The lowest cap, in steps of `step_mib` between the two bounds, at which run() still completes (it does not
    when it raises torch.OutOfMemoryError). Bisection: run() is called about log2 of the number of steps times,
    each time inside capped(...) after the cache was emptied. Raises when it does not complete at `high_mib`."""
    import torch

    def completes(mib: int) -> bool:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            with capped(mib):
                run()
            return True
        except torch.OutOfMemoryError:
            return False
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    if not completes(high_mib):
        raise RuntimeError(f"the block does not complete under a cap of {high_mib} MiB")
    low, high = 0, max(1, (high_mib - low_mib) // step_mib)     # caps low_mib + i * step; `high` is known to do
    best = high_mib
    while low < high:
        middle = (low + high) // 2
        if completes(low_mib + middle * step_mib):
            best = low_mib + middle * step_mib
            high = middle
        else:
            low = middle + 1
    return best
