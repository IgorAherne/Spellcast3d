"""Profiling for developers: where a tool's request spends its time, to make the Images stage much faster.

Switched on in spellcast3d-config.ini's [profiling] section (spellcast/config.py PROFILING_KEYS, config.profiling()),
with the switch names of AURA-V5's training config; enable ONE at a time. Reports: code/logs/profiling/<time>-<label>/.

Copied from AURA-V5 (commit 41007db; each file's header lists exactly what was adapted - lazy imports of torch and
pyinstrument, module loggers, these import paths, Spellcast3D's project markers, and ggml-aware kernel names):
  profiling_harness.py      start_python_profiler / stop_python_profiler (PyInstrument -> profile_raw.json +
                            performance_summary.txt), torch_profiler_context (scheduled torch.profiler for a step loop),
                            _echo_summary (a summary's head into the log). AURA's trainer-only phase_profiler is
                            left out.
  profiler_python_utils.py  process_profile_json (PyInstrument summary), process_torch_trace_json (kernel summary of a
                            Chrome trace), _clean_kernel_name - with readable_kernel_name for stable-diffusion.cpp's
                            mangled ggml kernels (mul_mat_q<Q4_K,128,false>).
  sync_hunter.py            hunt_syncs: torch's sync debug mode, for synchronisations PyTorch operations cause.
  trace_module_share.py     GPU ms per nn.Module (CLI); a launch outside any module goes to its innermost annotation.
New here:
  cuda_trace_report.py      reports over a CUPTI trace: per phase and step, per kernel class, and the CPU-GPU sync /
                            GPU idle report (CLI too).
  request_profiler.py       RequestProfiler: one request profiled from begin() to finish() with the switched-on
                            profilers, the request's phases and steps as annotations, the reports written at the end;
                            preload() and missing_packages() for the tool's start.
Owned by the Qwen entry (not part of the harness): qwen_profiled.py starts the unchanged launchers/qwen_server.py with
RequestProfiler attached from outside (the hub starts it instead of the server while a switch is on;
launchers/qwen_server.py has no profiling code at all), and qwen_bench.py runs the same request a few times.

What each switch measures for Qwen-Image 2.1 (stable-diffusion.cpp in-process through ctypes; torch launches nothing):
  use_python_profiler            the launcher's own Python around the ctypes call - the request's closure, the
                                 callbacks, loading the example pictures, saving the PNGs. Expected to be a small
                                 share: the picture itself is made inside one C call.
  use_torch_profiler             stable-diffusion.cpp's CUDA kernels, copies and runtime calls (CUPTI records them
                                 although torch launches none): kernels by time, per phase and per step, per class.
  diagnostic_enable_sync_hunter  the CUDA runtime's own blocking calls (stream / device synchronisations, synchronous
                                 and pageable copies, frees) and the GPU's idle gaps, per phase and between steps.
                                 sync_hunter.hunt_syncs is NOT used for Qwen: torch's sync debug mode only reports
                                 PyTorch's own operations, so it would always say nothing there.
For the torch-based tools only (Pixal3D, LATO.2, TriFlow; not wired to a switch yet): sync_hunter.hunt_syncs,
profiling_harness.torch_profiler_context, and trace_module_share's nn.Module attribution.

Every module above is stdlib-only at import; torch and pyinstrument are imported only by what a switched-on profiler
runs.

The 3D stage's memory tools (Pixal3D's staged engine; developers only, no switch): mem_watch.py, the launcher with
probes pixal3d_staged_profiled.py (with launcher_probes.py), pixal3d_membench.py, pixal3d_memreport.py,
ram_ballast.py, gpu_lock.py, toolkit.py and the checks in p3d_checks/ - "Measuring the 3D stage" in
code/ARCHITECTURE.md. The one-off tools of their development are in code/devtools/pixal3d-lowmem.
"""
