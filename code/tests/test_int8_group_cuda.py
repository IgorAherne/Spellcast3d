"""Which group-scaled int8 GEMMs take pixal3d_kernels' sm_80+ CUDA kernel (code/pixal3d/pixal3d/modules/int8mm.py
group_cuda_tile) instead of the Triton one, and when the install's race of the two is due again (group_cuda_stands,
calibrate.py's Triton step): on the CPU, the kernel's op, its build and the record replaced by stand-ins. Both kernels
give the same bits (code/devtools/pixal3d-lowmem/round-n/group-gemm/README.md); these are the rules that choose.

Runs in the pixal3d venv, with the card hidden and pytest from code/logs/profiling/pytools:
    set CUDA_VISIBLE_DEVICES=-1
    set PYTHONPATH=code\\logs\\profiling\\pytools
    code\\venvs\\pixal3d\\Scripts\\python.exe -m pytest -p no:cacheprovider code\\tests\\test_int8_group_cuda.py -q
Skipped in the hub venv (no torch).
"""

from __future__ import annotations

import sys

import pytest

from spellcast import paths

torch = pytest.importorskip("torch")
if not (paths.PIXAL3D_DIR / "pixal3d" / "modules" / "int8mm.py").is_file():
    pytest.skip("the Pixal3D checkout is not here", allow_module_level=True)

sys.path.insert(0, str(paths.PIXAL3D_DIR))
try:
    from pixal3d import calibration                                    # noqa: E402
    from pixal3d.modules import int8mm                                  # noqa: E402
finally:
    sys.path.remove(str(paths.PIXAL3D_DIR))

TILES = {"2": [64, 128, 32, 64, 64, 4, 2, 49168, 128], "4": [64, 128, 32, 64, 128, 2, 2, 49168, 128]}
CHOICE = {"4608x1536.bf16": 2, "8192x1536.gelu.int8": 4, "1536x8192.bf16": None}


class _OnCuda:
    """A tensor stand-in that says it is on a CUDA device (no kernel runs: the choice alone is looked at)."""

    def __init__(self, shape):
        self.shape = shape
        self.is_cuda = True
        self.device = "cuda:0"


@pytest.fixture
def kernel(monkeypatch):
    """The kernel's op present, build 0.2.0, the tiles of TILES; the record's entry made with them."""
    monkeypatch.delenv(int8mm.GROUP_CUDA_ENV, raising=False)
    monkeypatch.setattr(int8mm, "_group_cuda", {"op": (lambda *a: None), "checked": {}, "choice": {}, "said": True})
    monkeypatch.setattr(int8mm, "_kernels_build", lambda: "0.2.0")
    monkeypatch.setattr(int8mm, "_tile_info", lambda tile: list(TILES[str(tile)]))
    monkeypatch.setattr(int8mm, "group_cuda_check", lambda device, tile: True)
    import triton

    entry = {"ok": True, "triton": triton.__version__, "kernels": "0.2.0", "tiles": TILES, "choice": CHOICE}
    record = {"int8_tiles": {"fc1_group": [64, 128, 128, 8, 3, 4], "group_cuda": entry}}
    monkeypatch.setattr(calibration, "section", lambda name, device=None: record if name == "triton" else None)
    return entry


def _tile(N, K, gelu=False, out="bf16", resid=None):
    return int8mm.group_cuda_tile(_OnCuda((19928, K)), _OnCuda((N, K)), gelu, out, resid)


def test_the_race_s_choice_serves_its_forms_and_triton_the_others(kernel):
    assert _tile(4608, 1536) == 2
    assert _tile(8192, 1536, gelu=True, out="int8") == 4
    assert _tile(1536, 8192) is None                # Triton won fc2's race
    assert _tile(8192, 1536) is None                # a form the race did not time
    assert _tile(4608, 1536, out="fp16") is None    # fp16 out only where forced


def test_a_record_of_another_kernel_build_or_tile_table_is_not_read(kernel, monkeypatch):
    monkeypatch.setattr(int8mm, "_kernels_build", lambda: "0.3.0")
    assert _tile(4608, 1536) is None
    monkeypatch.setattr(int8mm, "_kernels_build", lambda: "0.2.0")
    int8mm._group_cuda["choice"].clear()
    monkeypatch.setattr(int8mm, "_tile_info", lambda tile: list(TILES[str(tile)])[:-1] + [256])
    assert _tile(4608, 1536) is None


def test_the_developer_switch(kernel, monkeypatch):
    monkeypatch.setenv(int8mm.GROUP_CUDA_ENV, "triton")
    assert _tile(4608, 1536) is None
    int8mm._group_cuda["choice"].clear()
    monkeypatch.setenv(int8mm.GROUP_CUDA_ENV, "cuda")
    assert _tile(1536, 8192) == int8mm.GROUP_CUDA_DEFAULT["bf16"]         # forced: the forms the race left to Triton
    assert _tile(4608, 1536, out="fp16") == int8mm.GROUP_CUDA_DEFAULT["fp16"]
    assert _tile(4608, 1536) == 2                                        # the record's tile still serves its form


def test_forms_the_kernel_lacks_keep_triton(kernel):
    gate32 = torch.zeros(4608, dtype=torch.float32)
    assert _tile(4608, 1536, resid={"x": torch.zeros(1, dtype=torch.bfloat16), "gate": gate32}) is None
    assert _tile(4608, 1536, resid={"x": torch.zeros(1, dtype=torch.bfloat16)}) == 2
    int8mm._group_cuda["choice"]["cuda:0"] = {"*": True}
    assert _tile(4600, 1536, gelu=True, out="int8") is None                # int8 out in whole groups of 128 only


def test_no_kernel_no_choice(kernel, monkeypatch):
    monkeypatch.setattr(int8mm, "_group_cuda", {"op": False, "checked": {}, "choice": {}, "said": True})
    assert _tile(4608, 1536) is None


def test_a_failed_first_use_check_keeps_triton(kernel, monkeypatch):
    monkeypatch.setattr(int8mm, "group_cuda_check", lambda device, tile: False)
    assert _tile(4608, 1536) is None


def test_when_the_install_races_again(kernel, monkeypatch):
    tiles = {"fc1_group": [64, 128, 128, 8, 3, 4]}
    assert int8mm.group_cuda_stands(tiles) is False                      # a record made before the race existed
    assert int8mm.group_cuda_stands({**tiles, "group_cuda": kernel}) is True
    assert int8mm.group_cuda_stands({**tiles, "group_cuda": {**kernel, "kernels": "0.1.0"}}) is False
    assert int8mm.group_cuda_stands({**tiles, "group_cuda": {"ok": False, "error": "x"}}) is True
    monkeypatch.setattr(int8mm, "_group_cuda", {"op": False, "checked": {}, "choice": {}, "said": True})
    assert int8mm.group_cuda_stands(tiles) is True                       # nothing to race with this build / card
