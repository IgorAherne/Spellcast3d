"""The release check for CPUs with AVX but without AVX2 (prepare_release.py check / scan-cpu): which binaries count as
built for AVX2, which of those a CPU without it never runs, where the binaries are looked for, and that one build is
disassembled once. The numbers are those of real binaries: the nvdiffrast wheel that stopped Pixal3D's server on a
Sandy Bridge CPU, its rebuild, the image engine's CPU variants, numpy.
Run with the hub venv:  python -m pytest -p no:cacheprovider code/tests/test_release_old_cpus.py
"""

from __future__ import annotations

import functools
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from spellcast import paths


@functools.cache
def _prepare_release() -> Any:
    sys.path.insert(0, str(paths.ROOT))
    try:
        import prepare_release
    finally:
        sys.path.remove(str(paths.ROOT))
    return prepare_release


def _counts(legacy: int, vex: int, bmi2: int = 0, fma: int = 0, avx2: int = 0) -> dict[str, int]:
    return {"legacy": legacy, "vex": vex, "bmi2": bmi2, "fma": fma, "avx2": avx2}


FAULTY_NVDIFFRAST = _counts(0, 221, bmi2=22, avx2=12)          # MSVC /arch:AVX2: died with "Illegal instruction"
REBUILT_NVDIFFRAST = _counts(198, 0, avx2=12)                  # no /arch: the 12 are the C runtime's, CPU-checked
GGML_HASWELL = _counts(0, 6098, bmi2=45, fma=232, avx2=4685)
GGML_SANDYBRIDGE = _counts(0, 6731, avx2=6)                    # built for AVX, which such a CPU has
NUMPY_UMATH = _counts(15939, 4703, bmi2=1124, fma=186, avx2=63646)     # SSE2 code, AVX2 paths picked at run time


def test_a_binary_is_built_for_avx2_when_its_float_code_is_vex_encoded_and_it_holds_what_only_that_target_emits() -> None:
    pr = _prepare_release()
    assert pr.needs_avx2(FAULTY_NVDIFFRAST) and pr.needs_avx2(GGML_HASWELL)
    assert not pr.needs_avx2(REBUILT_NVDIFFRAST)
    assert not pr.needs_avx2(GGML_SANDYBRIDGE), "an AVX build runs on a CPU with AVX"
    assert not pr.needs_avx2(NUMPY_UMATH), "SSE2 code with AVX2 paths beside it picks its path by the CPU"
    # One BMI2 instruction is the compiler's; FMA and AVX2 only count beyond the C runtime's own routines.
    assert pr.needs_avx2(_counts(0, 100, bmi2=1))
    assert not pr.needs_avx2(_counts(0, 100, fma=pr.RUNTIME_OWN, avx2=pr.RUNTIME_OWN))
    assert pr.needs_avx2(_counts(0, 100, fma=pr.RUNTIME_OWN + 1)) and pr.needs_avx2(_counts(0, 100, avx2=pr.RUNTIME_OWN + 1))
    assert not pr.needs_avx2(_counts(0, 0)), "a binary without float code (a launcher, a resource DLL)"


def test_the_instructions_are_counted_from_the_disassembly() -> None:
    pr = _prepare_release()
    lines = [
        "Dump of file example.pyd",
        "  0000000180001000: F3 0F 10 05 F8 1F  movss       xmm0,dword ptr [180003000h]",
        "  0000000180001008: C5 FA 10 05 F8 1F  vmovss      xmm0,dword ptr [180003000h]",
        "  0000000180001010: C5 FB 58 C1        vaddsd      xmm0,xmm0,xmm1",
        "  0000000180001014: C4 E2 71 F7 C2     shlx        eax,edx,ecx",
        "  0000000180001019: C4 E2 79 B8 CA     vfmadd231ps xmm1,xmm0,xmm2",
        "  000000018000101E: C5 FD FE C1        vpaddd      ymm0,ymm0,ymm1",
        "  0000000180001022: C5 FD 6F 00        vmovdqa     ymm0,ymmword ptr [rax]",          # AVX: a Sandy Bridge has it
        "  0000000180001026: C4 E2 7D 18 C1     vbroadcastss ymm0,xmm1",                      # AVX2: from a register
        "  000000018000102B: C4 E2 7D 18 00     vbroadcastss ymm0,dword ptr [rax]",           # AVX: from memory
        "  0000000180001030: 62 F1 7C 48 58 C1  vaddps      zmm0,zmm0,zmm1",
        "  0000000180001036: 62 F1 7E 09 10 C1  vmovss      xmm0{k1},xmm0,xmm1",              # AVX-512, not plain VEX
        "  000000018000103C: 48 89 5C 24 08     mov         qword ptr [rsp+8],rbx",
        "                    00 00",
    ]
    assert pr.count_host_code(lines) == {"legacy": 1, "vex": 2, "bmi2": 1, "fma": 1, "avx2": 4}


def test_only_what_a_cpu_check_guards_is_let_through_and_a_wheel_of_ours_never_is() -> None:
    pr = _prepare_release()
    faulty = "code/pixal3d/whl/nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl!_nvdiffrast_c.cp311-win_amd64.pyd"
    counted = {
        faulty: FAULTY_NVDIFFRAST,
        "code/pixal3d/whl/nvdiffrast-0.4.0+spellcast3d.1-cp311-cp311-win_amd64.whl!_nvdiffrast_c.cp311-win_amd64.pyd":
            REBUILT_NVDIFFRAST,
        "code/venvs/qwen/sdcpp/ggml-cpu-haswell.dll": GGML_HASWELL,
        "code/venvs/qwen/sdcpp/ggml-cpu-sandybridge.dll": GGML_SANDYBRIDGE,
        "code/venvs/pixal3d/Lib/site-packages/numpy/_core/_multiarray_umath.cp311-win_amd64.pyd": NUMPY_UMATH,
        "code/venvs/lato2/Lib/site-packages/torch/lib/fbgemm.dll": _counts(3465, 3749, bmi2=147, fma=383, avx2=22237),
        "code/pixal3d/whl/Pillow_SIMD-9.5.0.post2-cp311-cp311-win_amd64.whl!PIL/_imaging.cp311-win_amd64.pyd":
            _counts(0, 900, bmi2=30),
    }
    fails, excused = pr.avx2_only(counted)
    assert list(fails) == [faulty] and fails[faulty] is FAULTY_NVDIFFRAST
    assert set(excused) == {name for name in counted if "haswell" in name or "fbgemm" in name or "Pillow_SIMD" in name}
    assert all(why for why in excused.values())
    # The same binary anywhere else, or under another name, is no exception.
    assert pr.picked_by_cpu("code/venvs/pixal3d/Lib/site-packages/nvdiffrast/_nvdiffrast_c.cp311-win_amd64.pyd") is None
    assert pr.picked_by_cpu("C:/elsewhere/release/tools/ggml-cpu-haswell.dll"), "scan-cpu on another folder"
    assert pr.picked_by_cpu("code/venvs/pixal3d/Lib/site-packages/numpy/_core/_multiarray_umath.cp311-win_amd64.pyd") is None


def _fake_binary(path: Path, linked: int, size: int = 600) -> Path:
    """A file with a PE header (its link time `linked`), as build_of reads one."""
    head = bytearray(size)
    head[:2] = b"MZ"
    struct.pack_into("<I", head, 0x3C, 0x80)
    head[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<I", head, 0x88, linked)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(head))
    return path


def test_binaries_are_found_in_folders_and_inside_wheels_and_each_knows_its_build(tmp_path: Path) -> None:
    pr = _prepare_release()
    venv = tmp_path / "venv"
    _fake_binary(venv / "Lib" / "site-packages" / "pkg" / "_c.cp311-win_amd64.pyd", 1)
    _fake_binary(venv / "Scripts" / "python.exe", 2)
    (venv / "Lib" / "site-packages" / "pkg" / "__init__.py").write_text("")
    (venv / "Lib" / "site-packages" / "notes.dll").write_text("no Windows binary")
    (venv / "Lib" / "site-packages" / "numpy-2.2.6-cp311-cp311-win_amd64.whl").write_bytes(b"")     # numpy's marker
    wheel = tmp_path / "whl" / "pkg-1.0-cp311-cp311-win_amd64.whl"
    wheel.parent.mkdir()
    with zipfile.ZipFile(wheel, "w") as z:
        z.write(_fake_binary(tmp_path / "member.pyd", 3), "pkg/_c.cp311-win_amd64.pyd")
        z.writestr("pkg/__init__.py", "")
    found = pr.binaries_in([venv, wheel.parent])
    names = {name.split(tmp_path.name + "/", 1)[1]: binary for name, binary in found.items()}
    assert set(names) == {"venv/Lib/site-packages/pkg/_c.cp311-win_amd64.pyd", "venv/Scripts/python.exe",
                          "whl/pkg-1.0-cp311-cp311-win_amd64.whl!pkg/_c.cp311-win_amd64.pyd"}
    assert names["venv/Scripts/python.exe"] == pr.Binary("python.exe|600|00000002", venv / "Scripts" / "python.exe")
    packed = names["whl/pkg-1.0-cp311-cp311-win_amd64.whl!pkg/_c.cp311-win_amd64.pyd"]
    assert (packed.file, packed.member) == (wheel, "pkg/_c.cp311-win_amd64.pyd")
    assert packed.build.startswith("_c.cp311-win_amd64.pyd|600|crc"), "from the wheel's directory: nothing unpacked"
    # Another build under the same name is another build.
    _fake_binary(venv / "Scripts" / "python.exe", 5)
    assert pr.binaries_in([venv / "Scripts" / "python.exe"]).popitem()[1].build != names["venv/Scripts/python.exe"].build


class _InProcess:
    """Stands in for the process pool: the tests count the calls."""

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_InProcess":
        return self

    def __exit__(self, *_exc: Any) -> None:
        pass

    def map(self, fn: Any, *columns: Any) -> list[Any]:
        return [fn(*row) for row in zip(*columns)]


def test_one_build_is_disassembled_once_however_many_venvs_hold_it_and_remembered(tmp_path: Path,
                                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    pr = _prepare_release()
    calls: list[str] = []

    def counted(_dumpbin: str, binary: str) -> dict[str, int]:
        assert Path(binary).is_file(), "a wheel's binary is unpacked for the disassembler"
        calls.append(Path(binary).name)
        return _counts(5, 0)

    monkeypatch.setattr(pr, "host_code", counted)
    monkeypatch.setattr(pr.concurrent.futures, "ProcessPoolExecutor", _InProcess)
    monkeypatch.setattr(pr, "HOST_CODE_CACHE", tmp_path / "cache" / "host-code.json")
    for venv, linked in (("a", 7), ("b", 7), ("c", 8)):          # the same build twice, and another build of it
        _fake_binary(tmp_path / "venvs" / venv / "torch_cpu.dll", linked)
    wheel = tmp_path / "whl" / "pkg-1.0-cp311-cp311-win_amd64.whl"
    wheel.parent.mkdir()
    with zipfile.ZipFile(wheel, "w") as z:
        z.write(_fake_binary(tmp_path / "member.pyd", 3), "pkg/_c.pyd")
    binaries = pr.binaries_in([tmp_path / "venvs", wheel])
    result = pr.scan_host_code(Path("dumpbin.exe"), binaries)
    assert sorted(calls) == ["_c.pyd", "torch_cpu.dll", "torch_cpu.dll"]
    assert result == {name: _counts(5, 0) for name in binaries} and len(result) == 4
    # Next run: nothing is disassembled (no wheel unpacked), and a new build alone is.
    assert pr.scan_host_code(Path("dumpbin.exe"), binaries) == result and len(calls) == 3
    _fake_binary(tmp_path / "venvs" / "d" / "new.dll", 9)
    assert len(pr.scan_host_code(Path("dumpbin.exe"), pr.binaries_in([tmp_path / "venvs", wheel]))) == 5
    assert calls[3:] == ["new.dll"]


def test_python_itself_runs_without_avx2() -> None:
    """The real thing, where Visual Studio's dumpbin is there: the interpreter running this test."""
    pr = _prepare_release()
    dumpbin = pr.find_dumpbin()
    if dumpbin is None:
        pytest.skip("Visual Studio's C++ tools (dumpbin) are not installed")
    counts = pr.host_code(str(dumpbin), sys.executable)
    assert set(counts) == {"legacy", "vex", "bmi2", "fma", "avx2"} and not pr.needs_avx2(counts)
