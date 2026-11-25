"""The developers' profiling harness (spellcast/profiling/): the copies of AURA-V5's profilers with their Spellcast3D
adaptations (readable ggml kernel names, user code told apart on the full path), cuda_trace_report's three reports
on a synthetic CUPTI trace with known numbers, and RequestProfiler (inactive costs nothing; PyInstrument's reports;
marks from another thread ignored; AURA's one-at-a-time rule).

No GPU. PyTorch is not needed: what records with torch.profiler is skipped without it (it runs with the CPU activity
only where torch is installed, never touching the card). PyInstrument is in the hub's test venv.
Run with the hub venv:  python -m pytest -p no:cacheprovider code/tests/test_profiling.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import sysconfig
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from spellcast import paths
from spellcast.profiling import cuda_trace_report as report
from spellcast.profiling import profiler_python_utils as utils
from spellcast.profiling import request_profiler as rp

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_PYINSTRUMENT = importlib.util.find_spec("pyinstrument") is not None
needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="needs PyTorch (a tool venv)")
needs_pyinstrument = pytest.mark.skipif(not HAVE_PYINSTRUMENT, reason="needs pyinstrument")

ALL_OFF = {rp.PYTHON: False, rp.TORCH: False, rp.SYNC: False}
PYTHON_ON = {**ALL_OFF, rp.PYTHON: True}

# Two real kernel names from stable-diffusion.cpp's ggml-cuda.dll, as CUPTI reports them on Windows (mangled).
MMQ = "_Z9mul_mat_qIL9ggml_type12ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf5uint3iiiiiS8_S8_iiiS8_S8_iiiS8_"
QUANT = "_Z17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout1ELb0EEvPKfPKiPvxxxxxiii"
FLASH = ("_Z18flash_attn_ext_f16ILi128ELi128ELi8ELi4ELi4ELi1ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP6float2"
         "ffffjfiiiiiiiiiiiiiliiiiiiiiil")
BCAST = "_Z11k_bin_bcastIXadL_ZL6op_addffEEfffEvPKT0_PKT1_PT2_iiiiiiiiiiiiiiiiii"
# from the same DLL: nvcc's per-file namespace around a static function, and a parameter pack
BCAST_PACK = ("_Z11k_bin_bcastIXadL_ZN42_INTERNAL_d3f459f4_11_binbcast_cu_6840010b6op_addEffEE6__halfffJPKfEEvPKT0_"
              "PKT1_PT2_jjj5uint3SC_SC_SC_SC_jjjjjjjjjjjDpT3_")
FILL = "_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLm1EEEEviT0_T1_"

PID, REQ, OTHER, DEV, S1, S2 = 100, 7, 9, 0, 20, 21


# ---- a synthetic Kineto / CUPTI trace (microseconds) ---------------------------------------------------------------

def _x(cat: str, name: str, ts: float, dur: float, pid: int = PID, tid: int = REQ, **args: Any) -> dict[str, Any]:
    return {"ph": "X", "cat": cat, "name": name, "pid": pid, "tid": tid, "ts": ts, "dur": dur, "args": args}


def _ann(name: str, ts: float, dur: float) -> dict[str, Any]:
    return _x("user_annotation", "qwen:" + name, ts, dur)


def _call(name: str, ts: float, dur: float, corr: int, tid: int = REQ) -> dict[str, Any]:
    return _x("cuda_runtime", name, ts, dur, tid=tid, correlation=corr, cbid=1)


def _kernel(name: str, ts: float, dur: float, corr: int, stream: int = S1) -> dict[str, Any]:
    return _x("kernel", name, ts, dur, pid=DEV, tid=stream, correlation=corr, stream=stream)


def synthetic_events() -> list[dict[str, Any]]:
    """One request of 10 ms (1000..11000): prompt 2 ms, drawing 1/1 6 ms with two 3 ms steps, finishing 1.5 ms, then
    0.5 ms outside the phases. GPU busy 5.3 ms (an overlapping pair on two streams: 0.8 ms of kernels, 0.7 ms
    busy), and 0.3 ms waiting for host memory (a pageable copy: not busy); a kernel launched in step 1 runs in step 2
    (charged to step 1); one launched from another thread and one whose launch is missing are charged by their own
    time; Kineto's own syncs lie outside the window."""
    return [
        _call("cudaDeviceSynchronize", 500, 100, 900),                  # Kineto's own, before the window
        _ann("request", 1000, 10000),
        _ann("prompt", 1000, 2000),
        _ann("drawing 1/1", 3000, 6000),
        _ann("step 1/2", 3000, 3000),
        _ann("step 2/2", 6000, 3000),
        _ann("finishing", 9000, 1500),
        _x("user_annotation", "PyTorch Profiler (0)", 400, 11300),
        _x("gpu_user_annotation", "qwen:prompt", 1100, 1800, pid=DEV, tid=S1),
        _call("cudaLaunchKernel", 1100, 10, 1), _kernel(MMQ, 1200, 500, 1),
        _call("cudaLaunchKernel", 1150, 10, 2), _kernel(QUANT, 1600, 300, 2, stream=S2),
        _call("cudaMemcpyAsync", 2000, 400, 3),
        _x("gpu_memcpy", "Memcpy HtoD (Pageable -> Device)", 2050, 300, pid=DEV, tid=S1, correlation=3,
           bytes=3_000_000),
        _call("cudaStreamSynchronize", 2500, 200, 10),
        _call("cudaLaunchKernel", 3100, 20, 4), _kernel(FLASH, 3200, 2000, 4),
        _call("cudaStreamSynchronize", 3150, 2160, 11),
        _call("cudaEventSynchronize", 4000, 40, 12, tid=OTHER),
        _call("cudaLaunchKernel", 4100, 5, 8, tid=OTHER), _kernel("my_custom_kernel", 10200, 100, 8),
        _call("cudaLaunchKernel", 5900, 10, 5), _kernel(MMQ, 6100, 800, 5),
        _call("cudaLaunchKernel", 6150, 10, 6), _kernel(BCAST, 6920, 1000, 6),
        _call("cudaStreamSynchronize", 7000, 900, 13),
        _call("cudaFree", 8000, 300, 14),
        _call("cudaLaunchKernel", 9100, 10, 7), _kernel(MMQ, 9200, 600, 7),
        _call("cudaMalloc", 9900, 100, 15),
        _kernel(FILL, 10600, 100, 99),                                    # its launch is not in the trace
        _call("cudaDeviceSynchronize", 11500, 50, 901),                  # Kineto's own, after the window
    ]


def _write_trace(path: Path, events: list[dict[str, Any]]) -> Path:
    path.write_text(json.dumps({"traceEvents": events, "schemaVersion": 1}), encoding="utf-8")
    return path


def _lines(text: str) -> list[str]:
    return text.splitlines()


# ---- kernel names --------------------------------------------------------------------------------------------------

def test_ggml_kernel_names_are_demangled_with_their_template_arguments() -> None:
    clean = utils._clean_kernel_name
    assert clean(MMQ) == "mul_mat_q<Q4_K,128,false>"                 # AURA's buckets said "[Mul Kernel]"
    assert clean(QUANT) == "quantize_mmq_q8_1<mmq_q8_1_ds_layout=1,false>"
    assert clean(FLASH) == "flash_attn_ext_f16<128,128,8,4,4,1,false,false>"
    assert clean(BCAST) == "k_bin_bcast<op_add,float,float,float>"
    assert clean(BCAST_PACK) == "k_bin_bcast<op_add,__half,float,float,const float*>"
    assert clean("_ZN4ggml4cuda8k_kernelIfEEvPf") == "ggml::cuda::k_kernel<float>"
    # every ggml_type of the enum by name; an unknown one as its number; negative literals
    assert clean("_Z15mul_mat_vec_q_xIL9ggml_type30ELin1EEvv") == "mul_mat_vec_q_x<BF16,-1>"
    assert clean("_Z15mul_mat_vec_q_xIL9ggml_type99ELi2EEvv") == "mul_mat_vec_q_x<ggml_type=99,2>"
    # libraries keep AURA's heuristics; what does not parse to its end keeps AURA's bucket instead of a wrong name
    assert clean(FILL) == "[Elementwise: FillFunctor]"
    assert clean("_ZN7cutlass7Kernel2I52cutlass_80_tensorop_h16816gemm_128x128_32x4_nn_align8EEvNT_6ParamsE") \
        == "[GEMM Kernel]"
    assert clean("_Z8cpy_fltIXadL_ZL13cpy_1_flt_fltIffEvPKcPcEEEvS2_S3_ii") == "[CUDA Kernel]"   # bad length
    assert clean("Memcpy HtoD (Pageable -> Device)") == "Memcpy HtoD (Pageable -> Device)"
    # distinct instantiations stay distinct
    names = {clean(f"_Z9mul_mat_qIL9ggml_type{t}ELi{w}ELb{b}EEvPKc")
             for t in (12, 14) for w in (64, 128) for b in (0, 1)}
    assert len(names) == 8


def test_kernel_classes() -> None:
    def kind(name: str, cat: str = "kernel") -> str:
        return report.kernel_class({"cat": cat, "name": name})

    assert kind(MMQ) == "matmul" and kind(QUANT) == "quantize" and kind(FLASH) == "attention"
    assert kind(BCAST) == "elementwise" and kind(FILL) == "elementwise"
    assert kind("_Z12rms_norm_f32ILi1024ELb1ELb0EEvPKfPfi") == "norm"
    assert kind("_Z9rope_neoxILb1ELb0EfEvPKT1_PS0_i") == "rope"           # rope_neox, not a norm
    assert kind("_Z13convert_unaryIf6__halfEvPKvPT0_l") == "copy/convert"  # convert, not conv
    assert kind("ampere_h16816gemm_256x128_ldg8_stages_32x3_nn") == "matmul"
    assert kind("Memcpy DtoH (Device -> Pinned)", "gpu_memcpy") == "memcpy DtoH"
    assert kind("Memcpy HtoD (Pageable -> Device)", "gpu_memcpy") == "memcpy HtoD (pageable)"
    assert kind("Memset (Device)", "gpu_memset") == "memset"
    assert kind("my_custom_kernel") == "other"


# ---- cuda_trace_report ---------------------------------------------------------------------------------------------

def test_phase_report_charges_gpu_work_to_the_launching_phase_and_step() -> None:
    text = report.phase_report(synthetic_events(), "qwen:")
    lines = _lines(text)
    assert "Window: qwen:request, 10.0 ms." in lines[1]
    assert lines[3:] == [
        "prompt | 2.0 | 0.7 | 35.0 | 0.3 | 2 | 3.0 | 0.6",        # overlapping pair: 0.8 ms of kernels, 0.7 busy
        "drawing 1/1 | 6.0 | 3.8 | 63.3 | 0.0 | 3 | 0.0 | 3.4",
        "finishing | 1.5 | 0.7 | 46.7 | 0.0 | 2 | 0.0 | 0.1",     # incl. the other thread's kernel, by its own time
        "(outside the phases) | 0.5 | 0.1 | 20.0 | 0.0 | 1 | 0.0 | 0.0",
        "STEPS: 2 | wall ms min 3.0 / median 3.0 / max 3.0 | GPU busy median 1.9 ms (63.3%)",
        # incl. the kernel run in step 2
        "  first step of drawing 1/1: wall 3.0 ms, GPU busy 2.8 ms (93.3%), waiting for host memory 0.0 ms (0.0%)",
        "  the other 1 steps: wall ms min 3.0 / median 3.0 / max 3.0 | GPU busy median 1.0 ms | waiting for host "
        "memory median 0.0 ms",
        "TOTAL: wall 10.0 ms | GPU busy 5.3 ms (53.0%) | GPU idle 4.7 ms (47.0%), of it waiting for host memory "
        "(pageable copies) 0.3 ms (3.0%)",
    ]
    assert "GPU busy = the union of kernel, memset and pinned / device memcpy intervals" in lines[1]
    a = report.TraceAnalysis(synthetic_events())                       # the prefix is found
    assert a.prefix == "qwen:" and a.window == (1000.0, 11000.0) and a.thread == (PID, REQ)
    assert [p.name for p in a.phases] == ["prompt", "drawing 1/1", "finishing"]
    assert a.busy_total == 5300.0 and a.busy_between(3000, 6000) == 2000.0 and a.busy_between(1500, 1650) == 150.0
    assert a.waiting == [(2050.0, 2350.0)] and a.waiting_total == 300.0 and a.waiting_between(2000, 2100) == 50.0


def test_kernel_class_report() -> None:
    assert _lines(report.kernel_class_report(synthetic_events(), "qwen:")) == [
        "--- GPU TIME BY KERNEL CLASS (cuda_trace_report) ---",
        "GPU total 5.7 ms (sum of kernel, memcpy and memset durations; 9 events; 0.3 ms of it pageable copies, mostly "
        "the CPU staging them: see waiting for host memory in the phases)",
        "CLASS | GPU (ms) | SHARE % | CALLS",
        "attention | 2.0 | 35.1 | 1",
        " flash_attn_ext_f16<128,128,8,4,4,1,false,false> (2.0 ms, 1 call)",
        "matmul | 1.9 | 33.3 | 3",
        " mul_mat_q<Q4_K,128,false> (1.9 ms, 3 calls)",
        "elementwise | 1.1 | 19.3 | 2",
        " k_bin_bcast<op_add,float,float,float> (1.0 ms, 1 call)",
        " [Elementwise: FillFunctor] (0.1 ms, 1 call)",
        "quantize | 0.3 | 5.3 | 1",
        " quantize_mmq_q8_1<mmq_q8_1_ds_layout=1,false> (0.3 ms, 1 call)",
        "memcpy HtoD (pageable) | 0.3 | 5.3 | 1",
        " Memcpy HtoD (Pageable -> Device) (0.3 ms, 1 call)",
        "other | 0.1 | 1.8 | 1",
        " my_custom_kernel (0.1 ms, 1 call)",
        "PER PHASE (GPU ms by class, largest first)",
        " prompt: matmul 0.5 (45.5%); quantize 0.3 (27.3%); memcpy HtoD (pageable) 0.3 (27.3%)",
        " drawing 1/1: attention 2.0 (52.6%); elementwise 1.0 (26.3%); matmul 0.8 (21.1%)",
        " finishing: matmul 0.6 (85.7%); other 0.1 (14.3%)",
        " (outside the phases): elementwise 0.1 (100.0%)",
    ]


def test_kernels_that_read_alike_stay_distinct_rows() -> None:
    events = [_kernel("_ZN7cutlass7Kernel2I4gemmEEvv", 0, 10, 1), _kernel("_ZN7cutlass7Kernel2I5gemm2EEvv", 20, 5, 2)]
    lines = _lines(report.kernel_class_report(events))
    assert " [GEMM Kernel] (0.0 ms, 1 call)" in lines and " [GEMM Kernel] #2 (0.0 ms, 1 call)" in lines


def test_sync_report() -> None:
    lines = _lines(report.sync_report(synthetic_events(), "qwen:"))
    assert lines[2] == ("Window: qwen:request, 10.0 ms | GPU busy 5.3 ms (53.0%; kernels, memsets, pinned / device "
                        "copies) | GPU idle 4.7 ms, of it waiting for host memory (pageable copies) 0.3 ms (3.0%)")
    assert lines[3:24] == [
        "BLOCKING CALLS (the CPU waits for the GPU) | CALLS | TOTAL (ms) | MAX (ms)",
        "cudaStreamSynchronize | 3 | 3.3 | 2.2",
        "cudaMemcpyAsync (pageable: synchronous) | 1 | 0.4 | 0.4",
        "cudaFree | 1 | 0.3 | 0.3",
        "cudaEventSynchronize | 1 | 0.0 | 0.0",
        " per phase: prompt 2 calls 0.6 ms; drawing 1/1 4 calls 3.4 ms",
        " per step: 2.0 calls, 1.7 ms on average (2 steps)",
        " 1 of them on other threads than the request's",
        "COPIES | CALLS | MB | ms | GB/s",
        "Memcpy HtoD (Pageable -> Device) | 1 | 3.0 | 0.3 | 10.0",
        " per phase: prompt 3.0 MB 0.3 ms",
        " Pageable: from or to ordinary (not pinned) memory. Such an async copy is effectively synchronous: the CPU "
        "stages it and waits (the BLOCKING CALLS above count it). Its ms here are mostly that staging, so they are GPU "
        "idle time (waiting for host memory), not GPU busy.",
        "ALLOCATION / VMM CALLS | CALLS | TOTAL (ms)",
        "cudaFree | 1 | 0.3",
        "cudaMalloc | 1 | 0.1",
        "IDLE GAPS (no kernel, memset or pinned / device copy runs; WAITING FOR HOST MEMORY: a pageable copy is in "
        "flight meanwhile)",
        " GPU idle inside the window: 4.7 ms, of it waiting for host memory 0.3 ms; gaps over 0.1 ms: 7 (4.7 ms)",
        " per phase: prompt 1.3 ms (65.0%, 0.3 ms waiting for host memory); drawing 1/1 2.2 ms (36.7%); finishing "
        "0.8 ms (53.3%)",
        " inside the steps: 2.2 ms (1.1 ms per step on average)",
        " at the boundaries between steps: 1, the GPU idle there median 0.9 ms, max 0.9 ms, total 0.9 ms",
        " the 8 largest (WHERE: the phase and step holding most of the gap):",
    ]
    assert lines[24:28] == [
        " OFFSET (ms) | IDLE (ms) | OF IT WAITING FOR HOST MEMORY (ms) | WHERE | THE REQUEST THREAD",
        " 0.9 | 1.3 | 0.3 | prompt | in cudaMemcpyAsync for 0.4 ms of it",     # the copy no longer splits the gap
        " 6.9 | 1.3 | 0.0 | drawing 1/1, step 2/2 | in cudaFree for 0.3 ms of it",
        " 4.2 | 0.9 | 0.0 | drawing 1/1, step 1/2 | in cudaStreamSynchronize for 0.1 ms of it",
    ]
    assert " 9.3 | 0.3 | 0.0 | finishing | no CUDA call: CPU work in sd.cpp or Python" in lines   # 0.2 of 0.3 ms
    assert " 9.7 | 0.3 | 0.0 | (outside the phases) | no CUDA call: CPU work in sd.cpp or Python" in lines
    assert " 0.0 | 0.2 | 0.0 | prompt | in cudaLaunchKernel for 0.0 ms of it (2 calls)" in lines   # summed by name
    assert not any("cudaDeviceSynchronize" in line for line in lines)      # Kineto's own, outside the window


def _stall_events() -> list[dict[str, Any]]:
    """One request of 100 ms whose first step waits 44.9 ms for its weights (a pageable copy staged from disk) and
    whose prompt annotation opens 5 us after the request's: prompt 0.005..4, drawing 1/1 4..100 with step 1/2 4..60
    and step 2/2 60..100. GPU busy 49.8 ms: a kernel in the prompt (3..3.9), one in step 1 (49.3..59) and a pinned
    copy after it (59.1..59.5, busy), one in step 2 (60.2..99)."""
    return [
        _ann("request", 0, 100_000), _ann("prompt", 5, 3995), _ann("drawing 1/1", 4000, 96_000),
        _ann("step 1/2", 4000, 56_000), _ann("step 2/2", 60_000, 40_000),
        _call("cudaLaunchKernel", 3000, 10, 1), _kernel(MMQ, 3000, 900, 1),
        _call("cudaMemcpyAsync", 4100, 45_000, 2),
        _x("gpu_memcpy", "Memcpy HtoD (Pageable -> Device)", 4150, 44_900, pid=DEV, tid=S1, correlation=2,
           bytes=64_000_000),
        _call("cudaLaunchKernel", 49_200, 10, 3), _kernel(MMQ, 49_300, 9700, 3),
        _call("cudaStreamSynchronize", 49_210, 9800, 10),
        _call("cudaMemcpyAsync", 59_050, 20, 4),
        _x("gpu_memcpy", "Memcpy HtoD (Pinned -> Device)", 59_100, 400, pid=DEV, tid=S1, correlation=4, bytes=8e6),
        _call("cudaLaunchKernel", 60_100, 10, 5), _kernel(FLASH, 60_200, 38_800, 5),
    ]


def test_a_step_waiting_for_its_weights_is_not_gpu_busy_and_gaps_are_labelled_by_most_of_them() -> None:
    """Finding 3 (run 20261002-230625-qwen: 'GPU busy 97%' while step 1 waited 73 s for weights read from disk) and
    finding 4 (a gap opening a moment before the first phase was '(outside the phases)')."""
    a = report.TraceAnalysis(_stall_events())
    assert a.busy_total == 49_800.0 and a.waiting_total == 44_900.0
    lines = _lines(report.phase_report(a))
    assert lines[3:5] == ["prompt | 4.0 | 0.9 | 22.5 | 0.0 | 1 | 0.0 | 0.0",
                          "drawing 1/1 | 96.0 | 48.9 | 50.9 | 44.9 | 2 | 72.0 | 54.8"]
    assert "  first step of drawing 1/1: wall 56.0 ms, GPU busy 10.1 ms (18.0%), waiting for host memory 44.9 ms " \
           "(80.2%)" in lines
    assert lines[-1] == ("TOTAL: wall 100.0 ms | GPU busy 49.8 ms (49.8%) | GPU idle 50.2 ms (50.2%), of it waiting "
                         "for host memory (pageable copies) 44.9 ms (44.9%)")
    sync = _lines(report.sync_report(a))
    assert sync[2].endswith("| GPU idle 50.2 ms, of it waiting for host memory (pageable copies) 44.9 ms (44.9%)")
    at = sync.index(" OFFSET (ms) | IDLE (ms) | OF IT WAITING FOR HOST MEMORY (ms) | WHERE | THE REQUEST THREAD")
    assert sync[at + 1:at + 3] == [
        " 3.9 | 45.4 | 44.9 | drawing 1/1, step 1/2 | in cudaMemcpyAsync for 45.0 ms of it",   # the stall, first
        " 0.0 | 3.0 | 0.0 | prompt | no CUDA call: CPU work in sd.cpp or Python",              # was outside
    ]
    assert " inside the steps: 47.1 ms (23.6 ms per step on average), of it 44.9 ms waiting for host memory" in sync
    assert a.where_most(0, 3000) == "prompt" and a.where_most(3900, 4200) == "drawing 1/1, step 1/2"
    assert a.where_most(-5000, 1000) == report.OUTSIDE                     # mostly before the request


def test_reports_without_annotations_use_the_whole_trace() -> None:
    events = [e for e in synthetic_events() if e["cat"] != "user_annotation"]
    a = report.TraceAnalysis(events)
    assert a.prefix == "" and a.window == (500.0, 11550.0) and not a.phases
    assert "TOTAL: wall 11.1 ms" in report.phase_report(a)
    assert "cudaDeviceSynchronize | 2 |" in report.sync_report(a)


def test_load_trace_cli_and_aura_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    trace = _write_trace(tmp_path / "trace.json", synthetic_events())
    # torch's exporter can write a name with an unescaped """ (AURA repairs it the same way)
    text = trace.read_text(encoding="utf-8").replace('"my_custom_kernel"', '"my"""custom"')
    trace.write_text(text, encoding="utf-8")
    events = report.load_trace(trace)
    assert len(events) == len(synthetic_events()) and any(e["name"] == "my'''custom" for e in events)
    assert report.main([str(trace), "--report", "phases"]) == 0
    out = capsys.readouterr().out
    assert "drawing 1/1 | 6.0 | 3.8 | 63.3 | 0.0 | 3 | 0.0 | 3.4" in out and "GPU TIME BY KERNEL CLASS" not in out
    # AURA's kernel summary on the same trace: the ggml kernels by their readable names
    summary = tmp_path / "trace_summary.txt"
    utils.process_torch_trace_json(str(trace), str(summary))
    lines = _lines(summary.read_text(encoding="utf-8"))
    assert "mul_mat_q<Q4_K,128,false> | 1.90 | 1.90 | 3 | -" in lines
    assert "GPU BUSY: 5.7 ms of 11.3 ms (50.4%)  [kernel+memcpy+memset]" in lines             # AURA's, as it is
    assert ("  [SPELLCAST3D] of it pageable copies: 0.3 ms, mostly the CPU staging them, not GPU work; without them "
            "5.4 ms (47.8%). cuda_trace_report counts them as waiting for host memory, not busy.") in lines


def test_trace_module_share_charges_a_requests_trace_to_its_annotations(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch,
                                                                          capsys: pytest.CaptureFixture[str]) -> None:
    """Finding 6: request_profiler records with_stack=False; the CLI request.txt points to says so instead of AURA's
    'use a training-window trace'."""
    from spellcast.profiling import trace_module_share as share

    trace = _write_trace(tmp_path / "trace.json", synthetic_events())
    monkeypatch.setattr(sys, "argv", ["trace_module_share", str(trace)])
    share.main()
    out = capsys.readouterr().out
    assert "request_profiler records a request with_stack=False" in out and "training-window" not in out
    assert "<qwen:step 1/2>" in out and "<qwen:prompt>" in out
    _write_trace(trace, [e for e in synthetic_events() if e["cat"] != "user_annotation"])    # not a request's
    share.main()
    assert "use a training-window trace" in capsys.readouterr().out


def test_a_whole_requests_trace_is_read_in_seconds() -> None:
    """~200k events (one request's worth of kernels and launches): no quadratic step."""
    events = [_ann("request", 0, 2_000_000)]
    t, corr = 0.0, 0
    for picture_step in range(1, 21):
        events.append(_ann(f"step {picture_step}/20", t, 100_000))
        for _ in range(5000):
            corr += 1
            events.append(_call("cudaLaunchKernel", t, 2, corr))
            events.append(_kernel(MMQ if corr % 2 else BCAST, t + 5, 15, corr))
            t += 20
    events.insert(1, _ann("drawing 1/1", 0, t))
    started = time.perf_counter()
    a = report.TraceAnalysis(events, "qwen:")
    texts = [report.phase_report(a), report.kernel_class_report(a), report.sync_report(a)]
    assert time.perf_counter() - started < 20
    assert "STEPS: 20 |" in texts[0] and a.busy_total == 100_000 * 15


# ---- PyInstrument's summary ----------------------------------------------------------------------------------------

def test_profile_summary_keeps_our_code_by_its_full_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = str(paths.ROOT)
    monkeypatch.setattr(utils, "_STDLIB_PREFIX", "d:/py311/lib/")
    frame = {
        "function": "<module>", "file_path_short": "qwen_profiled.py", "time": 1.0,
        "file_path": os.path.join(root, "code", "spellcast", "profiling", "qwen_profiled.py"),
        "children": [{
            # short path "threading.py" and no "python" in the full one: AURA's filter kept it
            "function": "run", "file_path_short": "threading.py", "file_path": "D:\\py311\\Lib\\threading.py",
            "time": 0.9, "children": [{
                "function": "work", "file_path_short": "qwen_server.py", "time": 0.8,
                "file_path": os.path.join(root, "code", "spellcast", "launchers", "qwen_server.py"),
                "children": [
                    {"function": "[self]", "file_path_short": "qwen_server.py", "file_path": "", "time": 0.5},
                    {"function": "solve", "file_path_short": "fastapi/routing.py", "time": 0.3,
                     "file_path": os.path.join(root, "code", "venvs", "qwen", "Lib", "site-packages", "fastapi",
                                               "routing.py")},
                ]}]}]}
    raw, summary = tmp_path / "profile_raw.json", tmp_path / "performance_summary.txt"
    raw.write_text(json.dumps({"root_frame": frame}), encoding="utf-8")
    utils.process_profile_json(str(raw), str(summary))
    rows = _lines(summary.read_text(encoding="utf-8"))[4:]
    assert rows == ["<module> | 1.0000 | 0.1000 | 1 | qwen_profiled.py",
                    "work | 0.8000 | 0.0000 | 1 | qwen_server.py",
                    "[self] | 0.5000 | 0.5000 | 1 | qwen_server.py"]
    # the real stdlib, wherever this Python keeps it
    monkeypatch.undo()
    stdlib = sysconfig.get_paths()["stdlib"]
    frame["children"][0]["file_path"] = os.path.join(stdlib, "threading.py")
    raw.write_text(json.dumps({"root_frame": frame}), encoding="utf-8")
    utils.process_profile_json(str(raw), str(summary))
    assert not any(row.startswith("run |") for row in _lines(summary.read_text(encoding="utf-8")))


def _small_part() -> None:
    _busy(0.008)


def _profiled_work() -> None:
    time.sleep(1.0)
    _small_part()                                        # 0.8% of the profile


@needs_pyinstrument
def test_the_python_profile_keeps_every_frame_so_self_is_a_functions_own(tmp_path: Path) -> None:
    """Finding 5: pyinstrument's default JSON drops frames under 1% of the profile without giving their time back
    (run 20261002-231153-qwen: 4 frames, saving the pictures shown as work's SELF)."""
    from spellcast.profiling import profiling_harness as harness

    profiler = harness.start_python_profiler({rp.PYTHON: True})
    _profiled_work()
    harness.stop_python_profiler(profiler, str(tmp_path))
    rows = {row.split(" | ")[0]: row.split(" | ")
            for row in _lines((tmp_path / "performance_summary.txt").read_text(encoding="utf-8"))[4:]}
    assert "_small_part" in rows
    assert float(rows["_profiled_work"][2]) < 0.004      # its SELF; with the default filter ~0.008 (_small_part's)


# ---- RequestProfiler -----------------------------------------------------------------------------------------------

def test_all_switches_off_is_inactive_makes_no_folder_and_imports_nothing(tmp_path: Path) -> None:
    code = (
        "import sys\n"
        "import spellcast.profiling, spellcast.profiling.request_profiler as rp\n"
        "import spellcast.profiling.cuda_trace_report, spellcast.profiling.profiling_harness\n"
        "import spellcast.profiling.sync_hunter, spellcast.profiling.trace_module_share\n"
        "from pathlib import Path\n"
        f"p = rp.RequestProfiler.begin({ALL_OFF!r}, 'qwen', {{'gpu': 'x'}}, root=Path(r'{tmp_path}'))\n"
        "p.mark('prompt'); p.mark('drawing 1/1', steps=2); p.step(1, 2); p.note('k', 1); p.finish('ok')\n"
        "rp.preload({})\n"
        "print(p.active, p.folder, 'torch' in sys.modules, 'pyinstrument' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=paths.CODE, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "None", "False", "False"]
    assert list(tmp_path.iterdir()) == []


def test_missing_packages_and_the_one_at_a_time_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    on = {rp.PYTHON: True, rp.TORCH: True, rp.SYNC: True}
    monkeypatch.setattr(rp, "_available", lambda package: False)
    assert rp.missing_packages(on) == ["pyinstrument", "torch"]
    assert rp.missing_packages({rp.SYNC: True}) == ["torch"] and rp.missing_packages(ALL_OFF) == []
    runs, skipped = rp.plan(on)
    assert not any(runs.values()) and set(skipped) == set(rp.SWITCHES)
    monkeypatch.setattr(rp, "_available", lambda package: True)
    assert rp.missing_packages(on) == []
    runs, skipped = rp.plan(on)                       # python + torch: torch is skipped (AURA's rule); sync runs
    assert runs == {rp.PYTHON: True, rp.TORCH: False, rp.SYNC: True}
    assert "PyInstrument's sampling would slow down the request the trace times" in skipped[rp.TORCH]
    assert "stack tracer" not in skipped[rp.TORCH]           # with_stack=False: torch runs no Python tracer here
    monkeypatch.setattr(rp, "_available", lambda package: package == "torch")
    runs, skipped = rp.plan(on)                       # without pyinstrument, nothing conflicts with torch
    assert runs == {rp.PYTHON: False, rp.TORCH: True, rp.SYNC: True} and "pyinstrument" in skipped[rp.PYTHON]


def test_every_switched_on_profiler_unavailable_is_inactive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "_available", lambda package: False)
    said: list[str] = []
    p = rp.RequestProfiler.begin({**ALL_OFF, rp.TORCH: True}, "qwen", {}, root=tmp_path, log=said.append)
    p.mark("prompt")
    p.finish()
    assert not p.active and p.folder is None and list(tmp_path.iterdir()) == []
    assert any("use_torch_profiler skipped: PyTorch is not installed" in line for line in said)


# A torch that imports slowly, shaped like the real one where it matters (finding 2): its __init__ imports
# torch.autograd and uses it, and torch.autograd imports from torch. A second thread importing
# torch.autograd.profiler while __init__ still runs finds the half-made torch in sys.modules, loads torch.autograd
# itself and waits there for torch; __init__'s `import torch.autograd` then meets that half-loaded module, CPython
# accepts it to break the deadlock, and torch.autograd.grad is an AttributeError ("partially initialized module
# 'torch' has no attribute 'autograd'") - as the real torch 2.8 fails, in both threads.
FAKE_TORCH = {
    "torch/__init__.py": ("import time\ntime.sleep(0.6)\nnn = object()\nimport torch.autograd\n"
                          "grad = torch.autograd.grad\nimport torch.profiler\n"),
    "torch/autograd/__init__.py": "from torch import nn\n\n\ndef grad():\n    return nn\n",
    "torch/autograd/profiler.py": (
        "class record_function:\n"
        "    def __init__(self, name):\n        self.name = name\n"
        "    def __enter__(self):\n        return self\n"
        "    def __exit__(self, *exc):\n        return None\n"),
    "torch/profiler/__init__.py": (
        "import enum, json\n"
        "class ProfilerActivity(enum.Enum):\n    CPU = 0\n    CUDA = 1\n"
        "class profile:\n"
        "    def __init__(self, **kwargs):\n        pass\n"
        "    def start(self):\n        pass\n"
        "    def stop(self):\n        pass\n"
        "    def export_chrome_trace(self, path):\n"
        "        open(path, 'w', encoding='utf-8').write(json.dumps({'traceEvents': []}))\n"),
}


def test_a_request_during_the_preload_waits_for_it_instead_of_importing_beside_it(tmp_path: Path) -> None:
    """Finding 2: the first request came 0.02-0.3 s after preload() started importing torch, both imports failed and
    torch could not be imported again in that process. In a fresh process (a fake slow torch first on sys.path): the
    request's begin() 0.1 s into the preload waits for it, then profiles."""
    fake = tmp_path / "fake"
    for name, text in FAKE_TORCH.items():
        (fake / name).parent.mkdir(parents=True, exist_ok=True)
        (fake / name).write_text(text, encoding="utf-8")
    code = (
        "import json, sys, threading, time\n"
        f"sys.path.insert(0, r'{fake}')\n"
        "from pathlib import Path\n"
        "from spellcast.profiling import request_profiler as rp\n"
        "said, out = [], {}\n"
        "rp.preload({rp.TORCH: True}, log=said.append)\n"
        "time.sleep(0.1)\n"
        "def request():\n"
        f"    p = rp.RequestProfiler.begin({{rp.TORCH: True}}, 'qwen', {{}}, root=Path(r'{tmp_path / 'reports'}'), "
        "log=said.append)\n"
        "    out['active'] = p.active\n"
        "    p.mark('prompt')\n"
        "    p.finish('ok')\n"
        "thread = threading.Thread(target=request)\n"
        "thread.start()\n"
        "thread.join()\n"
        "try:\n"
        "    import torch.autograd.profiler\n"
        "    out['later'] = torch.__file__\n"
        "except Exception as exc:\n"
        "    out['later'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps({**out, 'said': said}))\n"
    )
    run = subprocess.run([sys.executable, "-c", code], cwd=paths.CODE, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout.strip().splitlines()[-1])
    said = result["said"]
    assert result["active"] and result["later"].startswith(str(fake)), result     # the fake, importable afterwards
    assert any("torch imported in" in line for line in said), said
    assert any(line.startswith(rp.PREFIX + "Waited ") for line in said), said
    assert not any("could not start" in line or "failed" in line.lower() for line in said), said


@needs_pyinstrument
def test_a_preload_that_never_ends_leaves_the_request_unprofiled(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    assert rp.wait_for_preload() == 0.0                       # none running
    release = threading.Event()
    hung = threading.Thread(target=release.wait, name="profiling-preload", daemon=True)
    hung.start()
    monkeypatch.setattr(rp, "_preloading", hung)
    monkeypatch.setattr(rp, "PRELOAD_WAIT", 0.2)
    said: list[str] = []
    try:
        p = rp.RequestProfiler.begin(PYTHON_ON, "qwen", {}, root=tmp_path, log=said.append)
        p.finish()
    finally:
        release.set()
        hung.join()
    assert not p.active and any("could not start (TimeoutError: the profilers' background import (preload) has "
                                "not finished" in line for line in said)
    assert rp.wait_for_preload() == 0.0                       # done now


def _busy(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        sum(range(200))


@needs_pyinstrument
def test_python_profiler_writes_its_reports_and_request_txt(tmp_path: Path) -> None:
    said: list[str] = []
    p = rp.RequestProfiler.begin(PYTHON_ON, "qwen", {"gpu": "NVIDIA GeForce RTX 3060", "placement": "offload"},
                                 root=tmp_path, log=said.append)
    assert p.active and p.folder is not None and p.folder.parent == tmp_path and p.folder.name.endswith("-qwen")
    assert "STILL RUNNING" in (p.folder / "request.txt").read_text(encoding="utf-8")
    p.note("width", 1024)
    p.mark("prompt")
    _busy(0.02)
    p.mark("drawing 1/1", steps=2)
    _busy(0.02)
    p.step(1, 2)
    _busy(0.02)
    p.step(2, 2)
    p.mark("finishing")
    p.finish("ok")
    p.finish("twice")                                   # a second finish does nothing
    files = sorted(f.name for f in p.folder.iterdir())
    assert files == ["performance_summary.txt", "profile_raw.json", "request.txt"]
    text = (p.folder / "request.txt").read_text(encoding="utf-8")
    assert "outcome: ok" in text and "STILL RUNNING" not in text and "twice" not in text
    assert "  ran: use_python_profiler" in text and "  gpu: NVIDIA GeForce RTX 3060" in text
    assert "  width: 1024" in text and "overhead" in text
    assert any(line.startswith("  drawing 1/1: ") and "2 of 2 steps" in line for line in text.splitlines())
    assert "  performance_summary.txt - " in text
    assert all(line.startswith(rp.PREFIX) for line in said)
    assert any("PROFILE SUMMARY HEAD (performance_summary.txt)" in line for line in said)
    summary = (p.folder / "performance_summary.txt").read_text(encoding="utf-8")
    assert "_busy" in summary                           # this test file's own code is shown
    # the log routing is undone
    logger = __import__("logging").getLogger(rp.LOGGER)
    assert not any(isinstance(h, rp._LogHandler) for h in logger.handlers) and logger.propagate


@needs_pyinstrument
def test_marks_steps_and_notes_from_another_thread_are_ignored(tmp_path: Path) -> None:
    p = rp.RequestProfiler.begin(PYTHON_ON, "qwen", {}, root=tmp_path, log=lambda line: None)
    elsewhere = threading.Thread(target=lambda: (p.mark("elsewhere", steps=3), p.step(1, 3),
                                                 p.note("noted_elsewhere", 1)))
    elsewhere.start()
    elsewhere.join()
    p.mark("prompt")
    p.finish()
    text = (p.folder / "request.txt").read_text(encoding="utf-8")
    assert "  prompt: " in text and "elsewhere" not in text


@needs_pyinstrument
def test_two_requests_in_one_second_get_their_own_folders(tmp_path: Path) -> None:
    first = rp.RequestProfiler.begin(PYTHON_ON, "qwen", {}, root=tmp_path, log=lambda line: None)
    first.finish()
    second = rp.RequestProfiler.begin(PYTHON_ON, "qwen", {}, root=tmp_path, log=lambda line: None)
    second.finish()
    assert first.folder != second.folder and len(list(tmp_path.iterdir())) == 2


@needs_pyinstrument
def test_a_profiler_that_cannot_start_leaves_an_unprofiled_request(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(self: Any) -> None:
        raise RuntimeError("no CUPTI")

    monkeypatch.setattr(rp.RequestProfiler, "_start", broken)
    said: list[str] = []
    p = rp.RequestProfiler.begin(PYTHON_ON, "qwen", {}, root=tmp_path, log=said.append)
    p.mark("prompt")
    p.finish()
    assert not p.active and any("could not start (RuntimeError: no CUPTI)" in line for line in said)
    text = (p.folder / "request.txt").read_text(encoding="utf-8")       # not left saying the request still runs
    assert "STILL RUNNING" not in text and "outcome: profiling could not start (RuntimeError: no CUPTI)" in text


@needs_torch
@pytest.mark.parametrize("switches", [{rp.TORCH: True, rp.SYNC: True}, {rp.SYNC: True}])
def test_torch_profiler_records_the_requests_annotations(switches: dict[str, bool], tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "TORCH_ACTIVITIES", ("CPU",))     # never the card in a test
    said: list[str] = []
    p = rp.RequestProfiler.begin({**ALL_OFF, **switches}, "qwen", {"gpu": "none"}, root=tmp_path, log=said.append)
    assert p.active
    p.mark("prompt")
    _busy(0.01)
    p.mark("drawing 1/1", steps=2)
    _busy(0.01)
    p.step(1, 2)
    _busy(0.01)
    p.step(2, 2)
    p.mark("finishing")
    p.finish("ok")
    full = switches.get(rp.TORCH, False)
    trace = p.folder / ("trace.json" if full else "sync_trace.json")
    names = {e.get("name") for e in report.load_trace(trace) if e.get("cat") == "user_annotation"}
    assert {"qwen:request", "qwen:prompt", "qwen:drawing 1/1", "qwen:step 1/2", "qwen:step 2/2",
            "qwen:finishing"} <= names
    a = report.TraceAnalysis(report.load_trace(trace), "qwen:")
    assert [s.name for s in a.phases] == ["prompt", "drawing 1/1", "finishing"]
    assert [s.name for s in a.steps] == ["step 1/2", "step 2/2"]
    assert (p.folder / "sync_report.txt").is_file()
    if full:
        summary = (p.folder / "trace_summary.txt").read_text(encoding="utf-8")
        assert "--- AURA TORCH KERNEL SUMMARY ---" in summary and "--- PHASES (cuda_trace_report) ---" in summary
        assert "--- GPU TIME BY KERNEL CLASS (cuda_trace_report) ---" in summary
    else:
        assert not (p.folder / "trace_summary.txt").exists()
    assert "outcome: ok" in (p.folder / "request.txt").read_text(encoding="utf-8")


# ---- Pixal3D's staged path: the measuring tools (pixal3d_staged_profiled, membench --batch, memreport, ram_ballast) --

from types import SimpleNamespace  # noqa: E402

from spellcast.profiling import mem_watch  # noqa: E402
from spellcast.profiling import pixal3d_memreport as memreport  # noqa: E402

MiB = 1024 * 1024


def _launchers_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(paths.LAUNCHERS))      # the launchers import their siblings as top-level modules


def _membench() -> Any:
    from spellcast.profiling import pixal3d_membench
    return pixal3d_membench


def _write_rows(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    lines = [",".join(columns)] + [",".join("" if r.get(c) is None else str(r[c]) for c in columns) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _staged_run(folder: Path, name: str = "staged") -> None:
    """A synthetic pixal3d_staged_profiled.py run with known numbers: one batch of two props. The loader's 64 MiB
    buffer is locked from the first part on; the store's two 256 MiB slots are locked from the hr part on (Windows shows
    them as 512 MiB more shared GPU memory, which is NOT a spill); in finish 300 MiB really spill. p1's hr needed
    the cap lifted after 2 s. The card idles 0.6 s between p1 and p2 in hr, and 4 s inside p1's GLB phase."""
    def pinned(total_mib: int) -> int:
        return total_mib * MiB

    events = [
        {"kind": "batch_start", "run": "batch1", "t": 10.0, "wave_size": 0, "resume": False, "pinned_bytes": 0,
         "props": [{"id": "p1", "resolution": 1024, "multi": False, "seed": 1, "views": 1},
                   {"id": "p2", "resolution": 1024, "multi": False, "seed": 2, "views": 1}]},
        {"kind": "part_start", "run": "batch1", "t": 11.0, "part": "cutout", "props": ["p1", "p2"],
         "pinned_bytes": pinned(64)},
        {"kind": "prop_part", "run": "batch1", "t": 12.0, "part": "cutout", "prop": "p1", "t0": 11.2, "t1": 12.0,
         "seconds": 0.8, "peak_alloc": 3100.0, "peak_reserved": 3400.0, "caps": [{"t": 11.2, "cap_mib": 6300}],
         "spilled": False, "pinned0": pinned(64), "pinned1": pinned(64)},
        {"kind": "prop_part", "run": "batch1", "t": 13.0, "part": "cutout", "prop": "p2", "t0": 12.1, "t1": 13.0,
         "seconds": 0.9, "peak_alloc": 3200.0, "peak_reserved": 3500.0, "caps": [{"t": 12.1, "cap_mib": 6200}],
         "spilled": False, "pinned0": pinned(64), "pinned1": pinned(64)},
        {"kind": "part_end", "run": "batch1", "t": 13.2, "part": "cutout",
         "pinned_bytes": {"loader": pinned(64), "store": 0, "spare": 0, "total": pinned(64)},
         "loader": {"pinned_bytes": pinned(64), "cap_mib": 6200}, "store": {"ram_bytes": 0, "uploads_ahead": 0}},
        {"kind": "part_start", "run": "batch1", "t": 14.0, "part": "hr", "props": ["p1", "p2"],
         "pinned_bytes": pinned(576)},
        {"kind": "net_ready", "run": "batch1", "t": 14.4, "part": "hr", "detail": {"name": "hr_flow.sv",
                                                                                   "seconds": 4.0, "bytes": 5.0e9}},
        {"kind": "prop_spilled", "run": "batch1", "t": 16.5, "part": "hr", "prop": "p1", "seconds_before_lift": 2.0},
        {"kind": "prop_part", "run": "batch1", "t": 17.0, "part": "hr", "prop": "p1", "t0": 14.5, "t1": 17.0,
         "seconds": 2.5, "peak_alloc": 6900.0, "peak_reserved": 7000.0, "caps": [{"t": 14.5, "cap_mib": 6368},
                                                                                 {"t": 16.5, "cap_mib": None}],
         "spilled": True, "seconds_before_lift": 2.0, "seconds_uncapped": 0.5,
         "pinned0": pinned(576), "pinned1": pinned(576)},
        {"kind": "prop_part", "run": "batch1", "t": 20.0, "part": "hr", "prop": "p2", "t0": 17.6, "t1": 20.0,
         "seconds": 2.4, "peak_alloc": 3500.0, "peak_reserved": 3600.0, "caps": [{"t": 17.6, "cap_mib": 6368}],
         "spilled": False, "pinned0": pinned(576), "pinned1": pinned(576)},
        {"kind": "part_end", "run": "batch1", "t": 20.2, "part": "hr",
         "pinned_bytes": {"loader": pinned(64), "store": pinned(512), "spare": 0, "total": pinned(576)},
         "loader": {"pinned_bytes": pinned(64)}, "store": {"ram_bytes": 300 * MiB, "peak_ram_bytes": 400 * MiB,
                                                          "disk_bytes": 0, "spills": 0, "uploads_ahead": 1}},
        {"kind": "part_start", "run": "batch1", "t": 21.0, "part": "finish", "props": ["p1"],
         "pinned_bytes": pinned(576)},
        {"kind": "prop_part", "run": "batch1", "t": 28.0, "part": "finish", "prop": "p1", "t0": 21.2, "t1": 28.0,
         "seconds": 6.8, "peak_alloc": 4000.0, "peak_reserved": 5000.0, "spilled": False,
         "caps": [{"t": 21.2, "cap_mib": 6068}, {"t": 23.0, "cap_mib": 3546}],
         "phases": [{"name": "decode", "t0": 21.2, "t1": 23.0, "peak_alloc": 4000.0, "peak_reserved": 5000.0,
                     "cap_mib": 6068},
                    {"name": "glb", "t0": 23.0, "t1": 28.0, "peak_alloc": 900.0, "peak_reserved": 1200.0,
                     "cap_mib": 3546}],
         "pinned0": pinned(576), "pinned1": pinned(576)},
        {"kind": "part_end", "run": "batch1", "t": 28.5, "part": "finish",
         "pinned_bytes": {"loader": pinned(64), "store": pinned(512), "spare": 0, "total": pinned(576)}},
        {"kind": "batch_end", "run": "batch1", "t": 29.0, "pinned_bytes": pinned(576),
         "results": {"p1": {"state": "done", "seconds": 10.1, "slow": ["hr"]},
                     "p2": {"state": "failed", "error": {"kind": "oom"}}}},
    ]
    (folder / f"{name}.events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    rows = []
    for i in range(50, 301):                       # 10 rows a second, t = 5.0 .. 30.0
        t = i / 10
        locked = 0 if t < 11.0 else 64 if t < 14.0 else 576
        spill = 300 if 22.0 <= t <= 24.0 else 0
        rows.append({"t": f"{t:.1f}", "run": "batch1", "stage": "", "proc_ded": 4000 + 1000, "proc_shared":
                     100 + locked + spill, "priv": 9000, "ws": 6000, "ws_private": 5000, "avail": 6000 if t > 21 else
                     8000, "commit": 20000, "commit_limit": 40000, "pinned": locked})
    columns = ("t", "run", "stage", "proc_ded", "proc_shared", "priv", "ws", "ws_private", "avail", "commit",
               "commit_limit", "pinned")
    _write_rows(folder / f"{name}.samples.csv", columns, rows)
    util = [{"t": f"{k / 5:.1f}", "util": 0 if (17.2 <= k / 5 <= 17.6 or 24.2 <= k / 5 <= 28.0) else 95}
            for k in range(25, 151)]                # one sample per 200 ms, each for the 200 ms before it
    _write_rows(folder / f"{name}.samples.util.csv", ("t", "util"), util)
    (folder / f"{name}.env.json").write_text(json.dumps({
        "tool": "pixal3d_staged_profiled", "card": {"name": "NVIDIA GeForce RTX 3060", "total_mib": 12288},
        "ram_total_mib": 31909, "commit_limit_mib": 40000, "hz": 10}), encoding="utf-8")
    (folder / f"{name}.ballast.json").write_text(json.dumps({"ballast_mib": 1000, "leave_mib": 8340, "t": 1.0}),
                                                 encoding="utf-8")
    (folder / f"{name}.ramballast.json").write_text(json.dumps({
        "pc_gib": 16, "locked_mib": 15525, "verified": True, "available_drop_mib": 15500, "partial": False,
        "before": {"ram_total_mib": 31909}}), encoding="utf-8")


def test_pinned_bytes_are_taken_off_shared_gpu_memory(tmp_path: Path) -> None:
    _staged_run(tmp_path)
    events = memreport.load_events(tmp_path / "staged.events.jsonl")
    steps = memreport.pinned_steps(events)
    assert memreport.pinned_at(steps, 5.0) == 0.0                       # before the first figure
    assert memreport.pinned_at(steps, 12.0) == 64.0
    assert memreport.pinned_at(steps, 14.0) == 576.0                    # the store's slots from the hr part on
    samples = memreport.Samples(tmp_path / "staged.samples.csv", 1000.0, 1.0)
    net = memreport.net_shared(samples, steps)
    idle = memreport.idle_level(samples, net, 11.0)
    assert idle == 100.0                                                # nothing locked before the first part
    hr = memreport.shared_rise(samples, steps, 14.0, 20.2, idle)
    assert hr["shared_peak"] == 676.0 and hr["pinned_peak"] == 576.0
    assert hr["net_peak"] == 100.0 and hr["rise"] == 0.0                # the slots are locked RAM, not a spill
    finish = memreport.shared_rise(samples, steps, 21.0, 28.5, idle)
    assert finish["net_peak"] == 400.0 and finish["rise"] == 300.0      # the real spill stays visible
    assert samples.peak("proc_ded", 14.0, 20.0) == 4000.0               # the ballast block taken off


def test_staged_report_arithmetic_on_synthetic_rows(tmp_path: Path) -> None:
    _staged_run(tmp_path)
    events = memreport.load_events(tmp_path / "staged.events.jsonl")
    prop_parts = memreport.staged_lines(events, "prop_part")
    peaks = memreport.part_peaks(prop_parts)
    assert peaks["cutout"]["alloc"] == 3200.0 and peaks["cutout"]["prop"] == "p2"
    assert peaks["cutout"]["reserved"] == 3500.0 and sorted(peaks["cutout"]["caps"]) == [6200, 6300]
    assert peaks["hr"]["spilled"] == ["p1"] and peaks["hr"]["reserved"] == 7000.0
    assert peaks["finish: decode"]["reserved"] == 5000.0 and peaks["finish: glb"]["alloc"] == 900.0
    assert "finish" not in peaks                                          # split into its two phases
    assert memreport.verdict_reserved(7000.0, 6368) == "OVER by 632"
    assert memreport.verdict_reserved(5000.0, 6068) == "inside"
    samples = memreport.Samples(tmp_path / "staged.samples.csv")
    hr = sorted((x for x in prop_parts if x["part"] == "hr"), key=lambda x: x["t0"])
    gaps = memreport.idle_gaps(samples, hr)
    assert len(gaps) == 1 and abs(gaps[0] - 0.6) < 1e-6                 # 17.0 .. 17.6
    assert abs(samples.idle_seconds(23.0, 28.0) - 4.0) < 1e-6            # the GLB phase: 24.0 .. 28.0

    text = memreport.staged_report(tmp_path, "staged")
    head = text.splitlines()[:8]
    assert any("emulated card of about 8.1 GiB" in line and "8,340 MiB" in line for line in head)
    assert any("an emulated 16 GiB PC" in line and "15,525 MiB page-locked" in line for line in head)
    assert any("Neither a real 8 GB card nor a real 16 GB PC was used" in line for line in head)
    rows = {line.split("|")[1].strip(): [c.strip() for c in line.split("|")[1:-1]]
            for line in text.splitlines() if line.startswith("| ") and ("OVER" in line or "inside" in line)}
    assert rows["hr"][7] == "OVER by 632" and rows["hr"][11] == "p1"     # reserved vs the 2.2 limit, cap lifted
    assert rows["finish: glb"][6] == "3,546" and rows["finish: glb"][7] == "inside"
    shared = [line for line in text.splitlines() if line.startswith("| hr | 6.2 |") and "idle" in line]
    assert shared and "| 676 | 576 | 100 | 0 | at idle |" in shared[0]
    finish = [line for line in text.splitlines() if line.startswith("| finish | 7.5 |") and "idle" in line]
    assert finish and "| 300 | ABOVE idle |" in finish[0]
    assert "| p1 | hr | 2.5 | 2.0 | 0.5 | 6,900 | 7,000 |" in text          # the prop that needed the cap lifted
    assert "| **launcher: private working set + page-locked, peak (held)** | 5,576 (inside 12,288) |" in text
    assert "| system: least commit headroom (limit - charge) | 20,000 |" in text
    assert "| p1 | 6.8 | 4.0 | 5.0 | 4.0 |" in text                      # finish: the GLB phase, 4 s of it idle
    assert "| 1 | 0.6 | 0.6 | 4.0 |" in text                              # hr: one boundary, its gap, the load


def test_staged_report_says_when_nothing_was_emulated(tmp_path: Path) -> None:
    _staged_run(tmp_path)
    (tmp_path / "staged.ballast.json").unlink()
    (tmp_path / "staged.ramballast.json").unlink()
    head = memreport.staged_report(tmp_path, "staged").splitlines()[:8]
    assert any("the full card, nothing emulated: NVIDIA GeForce RTX 3060 (12,288 MiB = 12.0 GiB)" in line
               for line in head)
    assert any("the PC's full RAM, nothing emulated: 31,909 MiB" in line for line in head)
    assert any("Neither a real 8 GB card nor a real 16 GB PC was used" in line for line in head)


def test_membench_batch_body_passes_the_launchers_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _launchers_on_path(monkeypatch)
    import pixal3d_server as server

    bench = _membench()
    work = tmp_path / "work"
    pictures = tmp_path / "pics"
    pictures.mkdir()
    for name in ("a.png", "b.png"):
        (pictures / name).write_bytes(b"x")
    props = bench.parse_props(["mixed4", f"1536:{pictures / 'a.png'}:7", "1024:b.png"], pictures=pictures)
    assert [p["id"] for p in props] == ["mv1024", "sv1024", "mv1024b", "sv1536", "p5-a", "p6-b"]
    assert props[4]["seed"] == 7 and props[4]["resolution"] == 1536 and props[5]["picture"] == pictures / "b.png"
    assert props[0]["seed"] == 33 and len(props[0]["views"]) == 4 and props[2]["seed"] == 44   # the golden cases
    staged = bench.stage_inputs(props, work)
    body = bench.batch_body(staged, work, "b1", wave_size=8, faces=300_000, texture=2048)

    request = server.BatchRequest.model_validate(json.loads(json.dumps(body)))     # what the route parses
    assert server.BatchRunner._refusal(SimpleNamespace(projects_root=work.resolve()), request) is None
    assert all(server.validate(item) is None for item in request.items)
    hub_keys = {"id", "image_path", "seed", "resolution", "decimation_target", "texture_size", "views", "views_dir",
                "out_path", "geometry"}                         # adapters/pixal3d.py _request + the id of _start
    # the bench names the geometry only for a prop set to Hi3D (the launcher's default is Pixal3D's own)
    assert all(set(item) | {"geometry"} == hub_keys for item in body["items"])
    assert set(server.BatchItem.model_fields) == hub_keys
    assert set(body) == set(server.BatchRequest.model_fields)
    mv = body["items"][0]
    assert [v["azimuth"] for v in mv["views"]] == [0, 90, 180, 270] and mv["image_path"] == mv["views"][0]["image_path"]
    assert mv["views_dir"] and body["items"][1]["views"] is None and body["items"][1]["views_dir"] is None
    assert body["wave_size"] == 8 and body["resume"] is False
    assert Path(body["work_dir"]).relative_to(work.resolve()).parts == ("bench", "temp", "mesh-b1")

    # A body the launcher refuses is caught here too: a work folder outside the projects root.
    bad = server.BatchRequest.model_validate({**body, "work_dir": str(tmp_path / "temp" / "mesh-b1")})
    assert server.BatchRunner._refusal(SimpleNamespace(projects_root=work.resolve()), bad) is not None
    with pytest.raises(ValueError):
        bench.parse_props(["fast"])


def test_membench_arguments_keep_the_generate_mode() -> None:
    bench = _membench()
    old = bench.parse_args(["--out", "o", "--work", "w", "--image", "x.png", "--runs", "1024,mv1024"])
    assert not old.batch and old.name == "pixal3d" and old.runs == "1024,mv1024"
    new = bench.parse_args(["--out", "o", "--work", "w", "--batch", "--props", "ten", "--wave-size", "8",
                            "--ram-pc-gib", "16", "--ballast-leave", "7360"])
    assert new.batch and new.name == "staged" and new.wave_size == 8 and new.ram_pc_gib == 16.0
    assert len(bench.parse_props(new.props)) == 10
    with pytest.raises(SystemExit):
        bench.parse_args(["--out", "o", "--work", "w", "--batch"])            # a batch needs props
    with pytest.raises(SystemExit):
        bench.parse_args(["--out", "o", "--work", "w"])                       # generate mode needs a picture


def test_ram_ballast_arguments_and_its_safe_minimum() -> None:
    from spellcast.profiling import ram_ballast as rb

    args = rb.parse_args(["--pc-gib", "16"])
    assert args.pc_gib == 16.0 and args.min_free_mib == 2048 and args.leave_mib is None
    for bad in (["--pc-gib", "16", "--leave-mib", "4000"], [], ["--pc-gib", "16", "--min-free-mib", "512"],
                ["--leave-mib", "0"], ["--pc-gib", "16", "--chunk-mib", "8"]):
        with pytest.raises(SystemExit):
            rb.parse_args(bad)
    # 31,909 MiB of RAM, 20,000 available: a 16 GiB PC is 15,525 MiB of ballast and leaves 4,475 available.
    assert rb.plan(16, None, 31909, 20000, 2048) == (15525, None)
    hold, why = rb.plan(16, None, 31909, 17000, 2048)                 # would leave 1,475: refused
    assert hold == 0 and "1,475 MiB" in why and "minimum of 2,048" in why
    hold, why = rb.plan(32, None, 31909, 30000, 2048)
    assert hold == 0 and "cannot emulate" in why
    assert rb.plan(None, 6000, 31909, 20000, 2048) == (14000, None)
    hold, why = rb.plan(None, 1500, 31909, 20000, 2048)               # leaving less than the minimum
    assert hold == 0 and "minimum" in why
    hold, why = rb.plan(None, 25000, 31909, 20000, 2048)
    assert hold == 0 and "already less" in why
    assert rb.main(["--pc-gib", "100000"]) == 2                       # refused before anything is allocated


def test_ram_ballast_locks_and_lets_go_of_a_small_block() -> None:
    from spellcast.profiling import ram_ballast as rb

    host = mem_watch.HostProbe()
    ballast = rb.Ballast()
    try:
        assert ballast.hold(32, 16) is None
        assert ballast.locked_bytes == 32 * MiB and len(ballast.chunks) == 2
        assert host.process()[1] >= 32 * MiB                         # the locked pages are in the working set
    finally:
        ballast.release()
    assert ballast.locked_bytes == 0


def test_mem_watch_adds_columns_at_the_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any) -> Any:
        raise OSError("not in this test")

    monkeypatch.setattr(mem_watch, "NvmlProbe", refuse)               # nothing asks the card in a test
    monkeypatch.setattr(mem_watch, "PdhProbe", refuse)
    watch = mem_watch.MemWatch(tmp_path / "rows.csv", extra={"pinned": lambda: 64 * MiB})
    try:
        cells = dict(zip(watch.columns, watch.row()))
    finally:
        watch.stop()
    assert watch.columns[:len(mem_watch.COLUMNS)] == mem_watch.COLUMNS and watch.columns[-1] == "pinned"
    assert mem_watch.COLUMNS[-3:] == ("late_ms", "ws_private", "commit_limit")
    assert cells["pinned"] == "64" and int(cells["commit_limit"]) >= int(cells["commit"]) > 0
    assert int(cells["ws_private"]) > 0                               # the Process V2 counter finds this process
    header = (tmp_path / "rows.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == list(watch.columns)


class _FakeLoader:
    def __init__(self) -> None:
        self.cap: int | None = None

    def set_cap(self, native_reserve_mib: int = 200, safety_mib: int | None = None) -> int:
        self.cap = 6000 - native_reserve_mib - (safety_mib or 300)
        return self.cap

    def clear_cap(self) -> None:
        self.cap = None

    def capped(self) -> bool:
        return self.cap is not None

    def stats(self) -> dict[str, Any]:
        return {"pinned_bytes": 64 * MiB, "cap_mib": self.cap, "loads": [{"name": "x"}], "prefetch": None}


class _FakeStore:
    def stats(self) -> dict[str, Any]:
        return {"ram_bytes": 10 * MiB, "pinned_bytes": 512 * MiB, "uploads_ahead": 2}


class _FakeEngine:
    """Emits what the staged engine emits for two prop-parts: cutout (capped) and finish (the cap lifted, then
    the hooks' finish), calling the loader the way the runner and parts.finish do."""

    def __init__(self, options: Any, hooks: Any) -> None:
        self.options, self.hooks, self.loader, self._store = options, hooks, _FakeLoader(), None
        self.multiview = True

    def run(self, props: list[Any], work_dir: Path, *, on_event: Any, cancelled: Any, wave_size: Any = None,
            resume: bool = False) -> dict[str, Any]:
        E = SimpleNamespace
        self._store = _FakeStore()
        on_event(E(kind="part_start", part="cutout", prop_id=None, detail={"props": ["p1"], "wave": 1, "waves": 1}))
        on_event(E(kind="net_ready", part="cutout", prop_id=None, detail={"name": "rembg", "seconds": 1.5,
                                                                          "bytes": 885 * MiB}))
        on_event(E(kind="prop_start", part="cutout", prop_id="p1", detail={"position": 1, "of": 1}))
        self.loader.set_cap(200)
        on_event(E(kind="prop_rest", part="cutout", prop_id="p1", detail={}))
        on_event(E(kind="part_end", part="cutout", prop_id=None, detail={}))
        on_event(E(kind="part_start", part="finish", prop_id=None, detail={"props": ["p1"]}))
        on_event(E(kind="prop_start", part="finish", prop_id="p1", detail={"position": 1, "of": 1}))
        self.loader.set_cap(300, safety_mib=500)
        self.loader.clear_cap()
        on_event(E(kind="prop_spilled", part="finish", prop_id="p1", detail={}))
        on_event(E(kind="step", part="finish", prop_id="p1", step=1, total=3, detail={}))
        mesh = SimpleNamespace(faces=SimpleNamespace(shape=(1000, 3)), vertices=SimpleNamespace(shape=(500, 3)),
                               coords=SimpleNamespace(shape=(4000, 3)), resolution=1024)
        result = self.hooks.finish(props[0], mesh, {})
        on_event(E(kind="prop_done", part="finish", prop_id="p1", detail={"result": result, "seconds": 2.0}))
        on_event(E(kind="part_end", part="finish", prop_id=None, detail={}))
        self._store = None
        return {"p1": {"state": "done", "result": {**result, "slow": ["finish"], "seconds": 2.0}}}


def test_staged_profiler_wraps_build_engine_from_outside(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _launchers_on_path(monkeypatch)
    from spellcast.profiling import pixal3d_staged_profiled as sp

    clock = iter(range(1000))
    watch = SimpleNamespace(now=lambda: float(next(clock)), stage="", period=0.05, segment=lambda *a: None,
                            flush=lambda: None)
    rec = sp.Recorder(tmp_path, "staged", watch)
    monkeypatch.setattr(sp, "REC", rec)
    pool = SimpleNamespace(_spare={(128 * MiB, True): [object(), object()], (64 * MiB, False): [object()]})
    monkeypatch.setitem(sys.modules, "pixal3d.staged.store", pool)
    assert sp.spare_pinned_bytes() == 256 * MiB                      # only the page-locked buffers (for the card)

    hooks = SimpleNamespace(finish=lambda req, mesh, notes: {"mode": "single"})
    srv = SimpleNamespace(build_engine=_FakeEngine, add_hub_routes=lambda *a: None, main=lambda: 0)
    assert sp.missing(srv) == []
    sp.attach_server(srv)
    engine = srv.build_engine(SimpleNamespace(max_vram_mib=None, wave_size=None), hooks)
    srv.build_engine(SimpleNamespace(), hooks)                         # a rebuild does not wrap the hooks twice
    seen: list[str] = []
    props = [SimpleNamespace(id="p1", resolution=1024, multi=False, seed=3, payload=SimpleNamespace(views=None))]
    results = engine.run(props, tmp_path / "w", on_event=lambda e: seen.append(e.kind), cancelled=lambda: False)
    rec.close()

    assert results["p1"]["state"] == "done"
    assert seen == ["part_start", "net_ready", "prop_start", "prop_rest", "part_end", "part_start", "prop_start",
                    "prop_spilled", "step", "prop_done", "part_end"]   # the launcher still gets every event
    lines = [json.loads(line) for line in (tmp_path / "staged.events.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = [line["kind"] for line in lines]
    assert kinds.count("prop_part") == 2 and "batch_start" in kinds and "batch_end" in kinds and "step" not in kinds
    cutout, finish = [line for line in lines if line["kind"] == "prop_part"]
    assert [c["cap_mib"] for c in cutout["caps"]] == [5500] and cutout["capped_at_end"] is True
    assert not cutout["spilled"]
    assert [c["cap_mib"] for c in finish["caps"]] == [5200, None] and finish["caps"][1]["lifted"]
    assert finish["spilled"] and finish["seconds_before_lift"] is not None and finish["seconds_uncapped"] is not None
    assert [p["name"] for p in finish["phases"]] == ["decode", "glb"] and finish["phases"][0]["faces"] == 1000
    assert [p["cap_mib"] for p in finish["phases"]] == [None, None]      # this try ran with the cap lifted
    decode, glb = {"cap_mib": 5500, "native_reserve_mib": 300}, {"cap_mib": 3000, "native_reserve_mib": 2360}
    assert sp.finish_caps([decode, glb]) == (5500, 3000) and sp.finish_caps([decode]) == (5500, None)
    assert sp.finish_caps([decode, {"cap_mib": None, "lifted": True}]) == (None, None)
    assert len(finish["steps"]) == 1 and finish["steps"][0][1:] == [1, 3]
    part_end = next(line for line in lines if line["kind"] == "part_end")
    assert part_end["pinned_bytes"] == {"loader": 64 * MiB, "store": 512 * MiB, "spare": 256 * MiB,
                                        "total": 832 * MiB}
    assert part_end["loader"]["loads_so_far"] == 1 and "loads" not in part_end["loader"]
    assert part_end["store"]["uploads_ahead"] == 2
    ready = next(line for line in lines if line["kind"] == "net_ready")
    assert ready["detail"] == {"name": "rembg", "seconds": 1.5, "bytes": 885 * MiB} and ready["part"] == "cutout"
    end = next(line for line in lines if line["kind"] == "batch_end")
    assert end["results"]["p1"]["slow"] == ["finish"]
    assert rec.pinned_total() == 64 * MiB + 256 * MiB                  # the store is closed: its slots are in the pool


def test_a_probe_that_fails_is_said_once_and_the_server_goes_on(capsys: pytest.CaptureFixture[str]) -> None:
    from spellcast.profiling import launcher_probes as probes

    guard = probes.Guard("probe")

    def broken(x: int) -> int:
        raise ValueError(f"bad {x}")

    assert guard("cap", broken, 1) is None and guard("cap", broken, 2) is None
    assert guard("cap", lambda x: x + 1, 1) == 2 and guard("none", None) is None
    out = capsys.readouterr().out
    assert out.count("[profiling] The cap probe failed (ValueError: bad 1)") == 1 and "bad 2" not in out
    own, rest = probes.profile_arguments("staged").parse_known_args(
        ["--port", "7791", "--profile-out", "o", "--profile-hz", "10"])
    assert own.profile_name == "staged" and own.profile_hz == 10.0 and own.profile_ballast_leave == 0
    assert rest == ["--port", "7791"]                                  # pixal3d_server.py's own arguments
    assert probes.mib(-1) is None and probes.mib(None) is None and probes.mib(3 * MiB // 2) == 1.5


def test_decoded_mesh_hashes_are_held_against_the_golden(tmp_path: Path) -> None:
    case = tmp_path / "golden" / "sv1024"
    case.mkdir(parents=True)
    (case / "manifest.json").write_text(json.dumps({
        "vertices_sha": "a", "faces_sha": "b", "tensors": {"11-decode": {"tex_coords": {"sha256": "c"}}}}),
        encoding="utf-8")
    mesh = {"vertices": {"sha256": "a"}, "faces": {"sha256": "x"}, "coords": {"sha256": "c"}, "attrs": {"sha256": "d"}}
    prop_parts = [{"part": "finish", "prop": "sv1024-2", "t0": 1.0, "phases": [{"name": "decode", "mesh": mesh}]},
                  {"part": "finish", "prop": "p5-x", "t0": 2.0, "phases": [{"name": "decode", "mesh": mesh}]},
                  {"part": "finish", "prop": "p6-y", "t0": 3.0, "phases": [{"name": "decode"}]}]
    assert memreport.golden_case("sv1024-2") == "sv1024" and memreport.golden_case("mv1024b") == "mv1024b"
    assert memreport.against_golden(prop_parts, tmp_path / "golden") == [
        ["sv1024-2", "sv1024", "equal", "DIFFERENT", "equal", "no golden"],
        ["p5-x", "(none)", "no golden", "no golden", "no golden", "no golden"]]


@needs_torch
def test_the_mesh_hash_is_the_goldens(monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib

    import torch

    _launchers_on_path(monkeypatch)
    from spellcast.profiling import pixal3d_staged_profiled as sp
    from spellcast.profiling import toolkit

    t = torch.arange(12, dtype=torch.float32).reshape(4, 3).t()          # not contiguous
    found = toolkit.tensor_record(t)
    assert found == {"shape": [3, 4], "dtype": "float32",
                     "sha256": hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()}
    assert toolkit.tensor_record(t.contiguous().numpy()) == found      # numpy's view of the same values
    mesh = SimpleNamespace(vertices=t, faces=torch.zeros(2, 3, dtype=torch.int32), coords=None, attrs=None)
    assert sp.mesh_hashes(mesh) == {"vertices": found, "faces": toolkit.tensor_record(mesh.faces)}
    low = toolkit.tensor_record(t.to(torch.bfloat16))                  # numpy has no bfloat16: its bits are hashed
    assert low["dtype"] == "bfloat16" and low["sha256"] == hashlib.sha256(
        t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
