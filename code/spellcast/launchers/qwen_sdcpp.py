"""ctypes mirror of stable-diffusion.cpp's C API (include/stable-diffusion.h) for launchers/qwen_server.py.
STDLIB ONLY: the hub's tests import it without the DLL.

PINNED: Spellcast3D's own build of stable-diffusion.cpp, release spellcast-perf-25d4127 of
github.com/IgorAherne/stable-diffusion.cpp (its branch spellcast3d at commit
25d412700bd616e4a9118abe6c5b42498de038f4: spellcast-perf + the First-Block Cache + upstream's FLUX.3 Image Klein support, with
github.com/IgorAherne/ggml's branch spellcast3d at a7c45a7), asset
sd-spellcast-perf-25d4127-bin-win-cuda12-x64.zip: what install/sdcpp.py unpacks. It is based on
leejet/stable-diffusion.cpp's release master-929-3f8527a (commit 3f8527a46c54ecf4cb4ed6003da8e8982283c73c,
2026-09-27) and leaves the public header's structs as they are there (`git diff 3f8527a 25d412700bd616e4a9118abe6c5b42498de038f4 -- include/` adds
sd_albedo_params_t / sd_albedo, mirrored below, and changes one comment line, the VAE tiling's default tile): every
layout below is that header's, and SD_PIN_PARAMS. Every struct below is copied field by field from that header,
in order: ctypes has no way to read a C header, so a DLL from another commit can disagree with these layouts
without any error, and then every setting lands in the wrong place. So load() checks the DLL before anything is
trusted (check_layout):
- the size: the DLL's own *_params_init fills a buffer a little larger than our struct, with a guard pattern behind
  it; a larger C struct overwrites the guard, a smaller one leaves our last fields unset;
- the offsets: init's documented defaults must appear in our fields (including ones its *_to_str never prints, such
  as the photo-maker and cache blocks);
- a round trip: every field gets its own sentinel value and the DLL prints them back through sd_ctx_params_to_str /
  sd_img_gen_params_to_str; one field in the wrong place shows as a wrong value there.
When the pin moves, compare the new header with this file (`git diff 25d412700bd616e4a9118abe6c5b42498de038f4 <new> -- include/stable-diffusion.h`
in the fork) and update both, then the checks here; PINNED_TAG and PINNED_COMMIT are install/sdcpp.py's ENGINE tag
and COMMIT (a test keeps the two files together).

The Windows build loads its CUDA backend (ggml-cuda.dll) at run time from the executable's folder or the current
folder, and ggml-cuda.dll needs cudart64_12 / cublas64_12 / cublasLt64_12 next to it: load() therefore makes the
sdcpp folder the current folder, a DLL directory and the front of PATH.

LICENSE OF stable-diffusion.cpp: MIT (Copyright (c) 2023 leejet; Spellcast3D's changes to it are MIT too), see
"README - LICENSE.txt".
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (CFUNCTYPE, POINTER, Structure, c_bool, c_char_p, c_float, c_int, c_int64, c_size_t, c_ubyte,
                    c_uint8, c_uint32, c_void_p)
from pathlib import Path
from typing import Any

PINNED_TAG = "spellcast-perf-25d4127"
PINNED_COMMIT = "25d412700bd616e4a9118abe6c5b42498de038f4"       # sd_commit() answers its first 7 characters
DLL_NAME = "stable-diffusion.dll"
CUDA_DLLS = ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")
GGML_DLLS = ("ggml.dll", "ggml-base.dll", "ggml-cuda.dll")
if os.name != "nt":
    # A Linux build of the same engine (a test pod, code/devtools/pod): its shared objects in the sdcpp folder; the
    # CUDA runtime and cuBLAS come from the library path (PyTorch's copies there), not from that folder.
    DLL_NAME = "libstable-diffusion.so"
    CUDA_DLLS = ()
    GGML_DLLS = ("libggml.so", "libggml-base.so", "libggml-cuda.so")

# ---- enums (only the values Spellcast3D uses or checks) ---------------------------------------------------------
STD_DEFAULT_RNG, CUDA_RNG, CPU_RNG, RNG_TYPE_COUNT = 0, 1, 2, 3
EULER_SAMPLE_METHOD = 0
SAMPLE_METHOD_COUNT = 21
SCHEDULER_COUNT = 17
FLOW_PRED, PREDICTION_COUNT = 3, 8
SD_TYPE_Q8_0, SD_TYPE_COUNT = 8, 45
SD_LOG_DEBUG, SD_LOG_VERBOSE, SD_LOG_INFO, SD_LOG_WARN, SD_LOG_ERROR = 0, 1, 2, 3, 4
SD_VAE_FORMAT_AUTO, SD_VAE_FORMAT_WAN = -1, 3
SD_CACHE_DISABLED, SD_CACHE_EASYCACHE = 0, 1
SD_HIRES_UPSCALER_LATENT, SD_HIRES_UPSCALER_LANCZOS = 1, 7
SD_CANCEL_ALL, SD_CANCEL_NEW_LATENTS, SD_CANCEL_RESET = 0, 1, 2
LORA_APPLY_AUTO = 0
# preview_t. PROJ: the latent through a fixed linear projection (no model, no decode): a picture of the latent's size
PREVIEW_NONE, PREVIEW_PROJ = 0, 1


# ---- structs (stable-diffusion.h @ 25d412700bd616e4a9118abe6c5b42498de038f4 = @ 3f8527a, same order) ---------------------------------------------
class SdTilingParams(Structure):
    _fields_ = [("enabled", c_bool), ("temporal_tiling", c_bool),
                ("tile_size_w", c_int), ("tile_size_h", c_int),        # image pixels; 0 = the engine's default
                ("target_overlap", c_float),
                ("rel_size_w", c_float), ("rel_size_h", c_float),
                ("extra_tiling_args", c_char_p)]


class SdEmbedding(Structure):
    _fields_ = [("name", c_char_p), ("path", c_char_p)]


class SdCtxParams(Structure):
    _fields_ = [("model_path", c_char_p), ("clip_l_path", c_char_p), ("clip_g_path", c_char_p),
                ("clip_vision_path", c_char_p), ("t5xxl_path", c_char_p), ("llm_path", c_char_p),
                ("llm_vision_path", c_char_p), ("diffusion_model_path", c_char_p),
                ("high_noise_diffusion_model_path", c_char_p), ("uncond_diffusion_model_path", c_char_p),
                ("embeddings_connectors_path", c_char_p), ("vae_path", c_char_p), ("audio_vae_path", c_char_p),
                ("audio_encoder_path", c_char_p), ("taesd_path", c_char_p), ("control_net_path", c_char_p),
                ("ip_adapter_path", c_char_p), ("motion_module_path", c_char_p),
                ("embeddings", POINTER(SdEmbedding)), ("embedding_count", c_uint32),
                ("photo_maker_path", c_char_p), ("pulid_weights_path", c_char_p), ("tensor_type_rules", c_char_p),
                ("n_threads", c_int), ("wtype", c_int), ("rng_type", c_int), ("sampler_rng_type", c_int),
                ("prediction", c_int), ("lora_apply_mode", c_int),
                ("enable_mmap", c_bool), ("flash_attn", c_bool), ("diffusion_flash_attn", c_bool),
                ("tae_preview_only", c_bool), ("diffusion_conv_direct", c_bool), ("vae_conv_direct", c_bool),
                ("force_sdxl_vae_conv_scale", c_bool),
                ("vae_format", c_int),
                ("max_vram", c_char_p),               # GiB budget per device; "-1": free at start minus ~1 GiB
                ("disable_prefetch", c_bool), ("eager_load", c_bool),
                ("backend", c_char_p), ("params_backend", c_char_p), ("split_mode", c_char_p),
                ("auto_fit", c_bool),
                ("rpc_servers", c_char_p), ("model_args", c_char_p),
                ("disable_segmented_compute", c_bool),
                ("linear_scale", c_float), ("attn_scale", c_float),
                ("tokenizer", c_char_p),
                ("sage_attn", c_bool),
                ("conditioning_cache_size", c_int)]


class SdImage(Structure):
    _fields_ = [("width", c_uint32), ("height", c_uint32), ("channel", c_uint32), ("data", POINTER(c_uint8))]


class SdImagePreprocessParams(Structure):
    _fields_ = [("rules", c_char_p)]


class SdSlgParams(Structure):
    _fields_ = [("layers", POINTER(c_int)), ("layer_count", c_size_t), ("layer_start", c_float),
                ("layer_end", c_float), ("scale", c_float)]


class SdGuidanceParams(Structure):
    _fields_ = [("txt_cfg", c_float), ("img_cfg", c_float), ("distilled_guidance", c_float), ("slg", SdSlgParams)]


class SdSampleParams(Structure):
    _fields_ = [("guidance", SdGuidanceParams), ("scheduler", c_int), ("sample_method", c_int),
                ("sample_steps", c_int), ("eta", c_float), ("shifted_timestep", c_int),
                ("custom_sigmas", POINTER(c_float)), ("custom_sigmas_count", c_int), ("flow_shift", c_float),
                ("extra_sample_args", c_char_p)]


class SdPmParams(Structure):
    _fields_ = [("id_images", POINTER(SdImage)), ("id_images_count", c_int), ("id_embed_path", c_char_p),
                ("style_strength", c_float)]


class SdPulidParams(Structure):
    _fields_ = [("id_embedding_path", c_char_p), ("id_weight", c_float)]


class SdCacheParams(Structure):
    _fields_ = [("mode", c_int), ("reuse_threshold", c_float), ("start_percent", c_float),
                ("end_percent", c_float), ("error_decay_rate", c_float), ("use_relative_threshold", c_bool),
                ("reset_error_on_compute", c_bool), ("Fn_compute_blocks", c_int), ("Bn_compute_blocks", c_int),
                ("residual_diff_threshold", c_float), ("max_warmup_steps", c_int), ("max_cached_steps", c_int),
                ("max_continuous_cached_steps", c_int), ("taylorseer_n_derivatives", c_int),
                ("taylorseer_skip_interval", c_int), ("scm_mask", c_char_p), ("scm_policy_dynamic", c_bool),
                ("spectrum_w", c_float), ("spectrum_m", c_int), ("spectrum_lam", c_float),
                ("spectrum_window_size", c_int), ("spectrum_flex_window", c_float),
                ("spectrum_warmup_steps", c_int), ("spectrum_stop_percent", c_float)]


class SdLora(Structure):
    _fields_ = [("is_high_noise", c_bool), ("multiplier", c_float), ("path", c_char_p)]


class SdHiresParams(Structure):
    _fields_ = [("enabled", c_bool), ("upscaler", c_int), ("model_path", c_char_p), ("scale", c_float),
                ("target_width", c_int), ("target_height", c_int), ("steps", c_int),
                ("denoising_strength", c_float), ("upscale_tile_size", c_int),
                ("custom_sigmas", POINTER(c_float)), ("custom_sigmas_count", c_int)]


class SdImgGenParams(Structure):
    _fields_ = [("loras", POINTER(SdLora)), ("lora_count", c_uint32),
                ("prompt", c_char_p), ("negative_prompt", c_char_p), ("clip_skip", c_int),
                ("init_image", SdImage),
                ("ref_images", POINTER(SdImage)), ("ref_images_count", c_int), ("ref_image_args", c_char_p),
                ("mask_image", SdImage),
                ("width", c_int), ("height", c_int),
                ("sample_params", SdSampleParams),
                ("strength", c_float), ("seed", c_int64), ("batch_count", c_int),
                ("control_image", SdImage), ("control_strength", c_float),
                ("ip_adapter_image", SdImage), ("ip_adapter_strength", c_float),
                ("pm_params", SdPmParams), ("pulid_params", SdPulidParams),
                ("vae_tiling_params", SdTilingParams),
                ("cache", SdCacheParams),
                ("hires", SdHiresParams),
                ("qwen_image_layers", c_int),
                ("circular_x", c_bool), ("circular_y", c_bool),
                ("image_preprocess", SdImagePreprocessParams)]


class SdAlbedoParams(Structure):
    """sd_albedo's request (engine builds with our albedo add-on's pass only: has_albedo)."""
    _fields_ = [("image", SdImage), ("lora_path", c_char_p), ("lora_multiplier", c_float),
                ("condition_path", c_char_p), ("decoder_path", c_char_p),
                ("timestep", c_float), ("output_gamma", c_float)]


LOG_CALLBACK = CFUNCTYPE(None, c_int, c_char_p, c_void_p)           # (level, text, data)
PROGRESS_CALLBACK = CFUNCTYPE(None, c_int, c_int, c_float, c_void_p)  # (step, steps, seconds of that step, data)
# (step, number of frames, the frames, whether they show the noisy input, data); sd.cpp frees the frames right after
PREVIEW_CALLBACK = CFUNCTYPE(None, c_int, c_int, POINTER(SdImage), c_bool, c_void_p)


class LayoutError(RuntimeError):
    """The DLL's structs are not the ones this file mirrors (another stable-diffusion.cpp build)."""


# ---- loading ----------------------------------------------------------------------------------------------------

def missing_files(sdcpp: Path) -> list[str]:
    """What load() needs in the sdcpp folder and does not find there."""
    names = (DLL_NAME,) + GGML_DLLS + CUDA_DLLS
    return [name for name in names if not (sdcpp / name).is_file()]


def prepare_dll_search(sdcpp: Path) -> None:
    """Make stable-diffusion.dll's run-time loads (ggml-cuda.dll, the CPU variants, the CUDA runtime) find their files
    in the sdcpp folder (see the module doc)."""
    folder = str(sdcpp)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(folder)
    os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
    os.chdir(folder)


def load(sdcpp: Path) -> Any:
    """stable-diffusion.dll from the sdcpp folder with the prototypes of every function the launcher calls."""
    prepare_dll_search(sdcpp)
    lib = ctypes.CDLL(str(sdcpp / DLL_NAME))
    declare(lib)
    return lib


def declare(lib: Any) -> None:
    def proto(name: str, restype: Any, *argtypes: Any) -> None:
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = restype, list(argtypes)

    proto("sd_set_log_callback", None, LOG_CALLBACK, c_void_p)
    proto("sd_set_progress_callback", None, PROGRESS_CALLBACK, c_void_p)
    # (callback, preview_t, every Nth step, the denoised estimate, the noisy input, data)
    proto("sd_set_preview_callback", None, PREVIEW_CALLBACK, c_int, c_int, c_bool, c_bool, c_void_p)
    proto("sd_ctx_params_init", None, POINTER(SdCtxParams))
    proto("sd_ctx_params_to_str", c_void_p, POINTER(SdCtxParams))          # malloc'ed: free with crt_free
    proto("new_sd_ctx", c_void_p, POINTER(SdCtxParams))
    proto("free_sd_ctx", None, c_void_p)
    proto("sd_img_gen_params_init", None, POINTER(SdImgGenParams))
    proto("sd_img_gen_params_to_str", c_void_p, POINTER(SdImgGenParams))
    proto("generate_image", c_bool, c_void_p, POINTER(SdImgGenParams), POINTER(POINTER(SdImage)), POINTER(c_int))
    proto("free_sd_images", None, POINTER(SdImage), c_int)
    proto("sd_cancel_generation", None, c_void_p, c_int)
    proto("sd_commit", c_char_p)
    proto("sd_version", c_char_p)
    proto("sd_get_system_info", c_char_p)
    proto("sd_get_model_version_name", c_char_p, c_void_p)
    proto("sd_type_name", c_char_p, c_int)
    proto("sd_rng_type_name", c_char_p, c_int)
    proto("sd_prediction_name", c_char_p, c_int)
    proto("sd_sample_method_name", c_char_p, c_int)
    proto("str_to_sample_method", c_int, c_char_p)
    proto("sd_scheduler_name", c_char_p, c_int)
    proto("sd_hires_upscaler_name", c_char_p, c_int)
    if has_albedo(lib):
        proto("sd_albedo_params_init", None, POINTER(SdAlbedoParams))
        proto("sd_albedo", POINTER(SdImage), c_void_p, POINTER(SdAlbedoParams))     # free with free_sd_images(_, 1)


def has_albedo(lib: Any) -> bool:
    """Whether this engine build has the albedo add-on's pass (sd_albedo)."""
    return hasattr(lib, "sd_albedo") and hasattr(lib, "sd_albedo_params_init")


def crt_free(pointer: int | None) -> None:
    """free() of the C runtime stable-diffusion.dll allocates with (the Universal CRT, as Python's own)."""
    if pointer:
        _ucrt().free(c_void_p(pointer))


_UCRT: Any = None


def _ucrt() -> Any:
    global _UCRT
    if _UCRT is None:
        _UCRT = ctypes.CDLL("ucrtbase" if os.name == "nt" else "libc.so.6")
        _UCRT.free.restype, _UCRT.free.argtypes = None, [c_void_p]
    return _UCRT


def take_string(pointer: int | None) -> str:
    """A char* the DLL malloc'ed -> its text; the memory is freed."""
    if not pointer:
        return ""
    try:
        return ctypes.string_at(pointer).decode("utf-8", "replace")
    finally:
        crt_free(pointer)


def commit(lib: Any) -> str:
    return (lib.sd_commit() or b"").decode("ascii", "replace")


def version(lib: Any) -> str:
    return (lib.sd_version() or b"").decode("ascii", "replace")


def is_pinned(lib: Any) -> bool:
    found = commit(lib).strip().lower()
    return len(found) >= 7 and PINNED_COMMIT.startswith(found)


# ---- the layout check ---------------------------------------------------------------------------------------------

GUARD = 256          # bytes behind our struct that the DLL must not touch
GUARD_BYTE = 0xA5


def init_struct(init: Any, struct_type: type) -> tuple[Any, bool]:
    """struct_type filled by the DLL's init function -> (the struct, whether the DLL stayed inside it)."""
    size = ctypes.sizeof(struct_type)
    buf = (c_ubyte * (size + GUARD))()
    ctypes.memset(buf, GUARD_BYTE, size + GUARD)
    init(ctypes.cast(buf, POINTER(struct_type)))
    raw = bytes(buf)
    return struct_type.from_buffer_copy(raw[:size]), all(b == GUARD_BYTE for b in raw[size:])


def _name(fn: Any, value: int) -> str:
    return (fn(value) or b"").decode("ascii", "replace")


def check_layout(lib: Any) -> None:
    """Raise LayoutError when the DLL disagrees with the structs above (see the module doc)."""
    problems: list[str] = []
    problems += _check_ctx_params(lib)
    problems += _check_img_gen_params(lib)
    if has_albedo(lib):
        problems += _check_albedo_params(lib)
    if problems:
        raise LayoutError("; ".join(problems))


def _check_albedo_params(lib: Any) -> list[str]:
    problems: list[str] = []
    p, inside = init_struct(lib.sd_albedo_params_init, SdAlbedoParams)
    if not inside:
        problems.append("sd_albedo_params_t is larger in the DLL")
    # sd_albedo_params_init's defaults, from the first field to the last one
    _expect(problems, "init albedo lora_multiplier", p.lora_multiplier, 1.0)
    _expect(problems, "init albedo timestep", p.timestep, 0.498046875)
    _expect(problems, "init albedo output_gamma", p.output_gamma, 2.2)
    return problems


def _expect(problems: list[str], what: str, got: Any, want: Any) -> None:
    if isinstance(want, float):
        ok = isinstance(got, float) and abs(got - want) <= 1e-6 * max(1.0, abs(want))
    else:
        ok = got == want
    if not ok:
        problems.append(f"{what} is {got!r}, expected {want!r}")


def _check_ctx_params(lib: Any) -> list[str]:
    problems: list[str] = []
    p, inside = init_struct(lib.sd_ctx_params_init, SdCtxParams)
    if not inside:
        problems.append("sd_ctx_params_t is larger in the DLL")
    # sd_ctx_params_init's defaults (src/stable-diffusion.cpp @ 3f8527a, the same file in the pinned build)
    _expect(problems, "init conditioning_cache_size", p.conditioning_cache_size, 4)
    _expect(problems, "init wtype", p.wtype, SD_TYPE_COUNT)
    _expect(problems, "init rng_type", p.rng_type, CUDA_RNG)
    _expect(problems, "init sampler_rng_type", p.sampler_rng_type, RNG_TYPE_COUNT)
    _expect(problems, "init prediction", p.prediction, PREDICTION_COUNT)
    _expect(problems, "init vae_format", p.vae_format, SD_VAE_FORMAT_AUTO)
    _expect(problems, "init auto_fit", p.auto_fit, True)
    _expect(problems, "init sage_attn", p.sage_attn, False)
    _expect(problems, "init model_args", p.model_args, None)
    _expect(problems, "init tokenizer", p.tokenizer, None)
    if problems:
        return problems

    # Sentinels: every field its own value; the DLL prints them back.
    keep: list[Any] = []

    def text(value: str) -> bytes:
        data = value.encode("ascii")
        keep.append(data)
        return data

    strings = ("model_path", "clip_l_path", "clip_g_path", "clip_vision_path", "t5xxl_path", "llm_path",
               "llm_vision_path", "diffusion_model_path", "high_noise_diffusion_model_path",
               "uncond_diffusion_model_path", "embeddings_connectors_path", "vae_path", "audio_vae_path",
               "audio_encoder_path", "taesd_path", "control_net_path", "ip_adapter_path", "motion_module_path",
               "photo_maker_path", "pulid_weights_path", "tensor_type_rules", "max_vram", "backend",
               "params_backend", "split_mode", "rpc_servers", "model_args", "tokenizer")
    for i, name in enumerate(strings):
        setattr(p, name, text(f"<{name}#{i}>"))
    p.n_threads, p.conditioning_cache_size = 37, 11
    p.wtype, p.rng_type, p.sampler_rng_type, p.prediction = SD_TYPE_Q8_0, CPU_RNG, STD_DEFAULT_RNG, FLOW_PRED
    p.enable_mmap, p.flash_attn, p.diffusion_flash_attn, p.tae_preview_only = True, True, False, True
    p.diffusion_conv_direct, p.vae_conv_direct, p.force_sdxl_vae_conv_scale = False, True, False
    p.vae_format = SD_VAE_FORMAT_WAN
    p.disable_prefetch, p.eager_load, p.auto_fit, p.disable_segmented_compute = True, False, False, True
    p.linear_scale, p.attn_scale, p.sage_attn = 0.25, 1.5, True
    printed = take_string(lib.sd_ctx_params_to_str(ctypes.byref(p)))
    lines = dict(line.split(": ", 1) for line in printed.splitlines() if ": " in line)
    for i, name in enumerate(strings):
        if name in ("ip_adapter_path", "motion_module_path", "rpc_servers"):
            continue                                    # not printed (their neighbours are)
        _expect(problems, name, lines.get(name), f"<{name}#{i}>")
    wanted = {"n_threads": "37", "conditioning_cache_size": "11",
              "wtype": _name(lib.sd_type_name, SD_TYPE_Q8_0), "rng_type": _name(lib.sd_rng_type_name, CPU_RNG),
              "sampler_rng_type": _name(lib.sd_rng_type_name, STD_DEFAULT_RNG),
              "prediction": _name(lib.sd_prediction_name, FLOW_PRED),
              "disable_prefetch": "true", "disable_segmented_compute": "true", "eager_load": "false",
              "auto_fit": "false", "flash_attn": "true", "diffusion_flash_attn": "false", "sage_attn": "true",
              "linear_scale": "0.25", "attn_scale": "1.5", "vae_format": "wan"}
    for name, want in wanted.items():
        _expect(problems, name, lines.get(name), want)
    return problems


def _check_img_gen_params(lib: Any) -> list[str]:
    problems: list[str] = []
    g, inside = init_struct(lib.sd_img_gen_params_init, SdImgGenParams)
    if not inside:
        problems.append("sd_img_gen_params_t is larger in the DLL")
    # sd_img_gen_params_init's defaults, from the first field to the last one
    s = g.sample_params
    _expect(problems, "init clip_skip", g.clip_skip, -1)
    _expect(problems, "init ref_image_args", g.ref_image_args, b"")
    _expect(problems, "init width", g.width, 512)
    _expect(problems, "init txt_cfg", s.guidance.txt_cfg, 7.0)
    _expect(problems, "init distilled_guidance", s.guidance.distilled_guidance, 3.5)
    _expect(problems, "init slg.layer_start", s.guidance.slg.layer_start, 0.01)
    _expect(problems, "init scheduler", s.scheduler, SCHEDULER_COUNT)
    _expect(problems, "init sample_method", s.sample_method, SAMPLE_METHOD_COUNT)
    _expect(problems, "init sample_steps", s.sample_steps, 20)
    _expect(problems, "init strength", g.strength, 0.75)
    _expect(problems, "init seed", g.seed, -1)
    _expect(problems, "init batch_count", g.batch_count, 1)
    _expect(problems, "init control_strength", g.control_strength, 0.9)
    _expect(problems, "init ip_adapter_strength", g.ip_adapter_strength, 1.0)
    _expect(problems, "init pm style_strength", g.pm_params.style_strength, 20.0)
    _expect(problems, "init pulid id_weight", g.pulid_params.id_weight, 1.0)
    _expect(problems, "init tiling overlap", g.vae_tiling_params.target_overlap, 0.5)
    _expect(problems, "init cache start", g.cache.start_percent, 0.15)
    _expect(problems, "init cache Fn blocks", g.cache.Fn_compute_blocks, 8)
    _expect(problems, "init cache spectrum_stop", g.cache.spectrum_stop_percent, 0.9)
    _expect(problems, "init hires upscaler", g.hires.upscaler, SD_HIRES_UPSCALER_LATENT)
    _expect(problems, "init hires tile", g.hires.upscale_tile_size, 128)
    _expect(problems, "init qwen_image_layers", g.qwen_image_layers, 3)
    _expect(problems, "init image_preprocess", g.image_preprocess.rules, None)
    if problems:
        return problems

    keep: list[Any] = []

    def text(value: str) -> bytes:
        data = value.encode("ascii")
        keep.append(data)
        return data

    g.prompt, g.negative_prompt, g.clip_skip = text("<prompt>"), text("<negative>"), 3
    g.width, g.height = 1344, 736
    s.guidance.txt_cfg, s.guidance.img_cfg, s.guidance.distilled_guidance = 1.25, 2.5, 4.75
    s.guidance.slg.layer_count, s.guidance.slg.layer_start = 5, 0.15
    s.guidance.slg.layer_end, s.guidance.slg.scale = 0.35, 1.75
    s.scheduler, s.sample_method, s.sample_steps = 2, 5, 37
    s.eta, s.shifted_timestep, s.flow_shift, s.extra_sample_args = 0.65, 9, 2.25, text("<sample-args>")
    g.strength, g.seed, g.batch_count, g.qwen_image_layers = 0.85, 1234567890123, 6, 7
    g.ref_images_count, g.ref_image_args, g.control_strength = 0, text("<ref-args>"), 0.3      # 0: no array
    g.pm_params.style_strength, g.pm_params.id_images_count, g.pm_params.id_embed_path = 12.5, 0, text("<pm>")
    g.vae_tiling_params.enabled, g.vae_tiling_params.temporal_tiling = True, False
    g.vae_tiling_params.extra_tiling_args = text("<tiling>")
    g.circular_x, g.circular_y = False, True
    g.hires.enabled, g.hires.upscaler, g.hires.model_path = True, SD_HIRES_UPSCALER_LANCZOS, text("<hires>")
    g.hires.scale, g.hires.target_width, g.hires.target_height = 1.5, 1600, 900
    g.hires.steps, g.hires.denoising_strength = 11, 0.45
    g.cache.mode, g.cache.start_percent, g.cache.end_percent = SD_CACHE_EASYCACHE, 0.2, 0.8
    printed = take_string(lib.sd_img_gen_params_to_str(ctypes.byref(g)))
    scheduler = _name(lib.sd_scheduler_name, 2)
    method = _name(lib.sd_sample_method_name, 5)
    upscaler = _name(lib.sd_hires_upscaler_name, SD_HIRES_UPSCALER_LANCZOS)
    wanted = ["prompt: <prompt>\n", "negative_prompt: <negative>\n", "clip_skip: 3\n", "width: 1344\n",
              "height: 736\n",
              "sample_params: (txt_cfg: 1.25, img_cfg: 2.50, distilled_guidance: 4.75, slg.layer_count: 5, "
              "slg.layer_start: 0.15, slg.layer_end: 0.35, slg.scale: 1.75, "
              f"scheduler: {scheduler}, sample_method: {method}, sample_steps: 37, eta: 0.65, "
              "shifted_timestep: 9, flow_shift: 2.25, extra_sample_args: <sample-args>)\n",
              "strength: 0.85\n", "seed: 1234567890123\n", "batch_count: 6\n", "qwen_image_layers: 7\n",
              "ref_images_count: 0\n", "ref_image_args: <ref-args>\n", "control_strength: 0.30\n",
              "photo maker: {style_strength = 12.50, id_images_count = 0, id_embed_path = <pm>}\n",
              "VAE tiling: true (temporal=false, extra_tiling_args=<tiling>)\n",
              "circular_x: false\n", "circular_y: true\n",
              f"hires: {{enabled=true, upscaler={upscaler}, model_path=<hires>, scale=1.50, target=1600x900, "
              "steps=11, denoising_strength=0.45}\n",
              "cache: easycache (threshold="]
    for piece in wanted:
        if piece not in printed:
            problems.append(f"sd_img_gen_params_to_str has no {piece.strip()!r}")
    if "start=0.20, end=0.80)" not in printed:
        problems.append("sd_img_gen_params_to_str shows other cache start/end values")
    return problems


def describe_failure(exc: BaseException) -> str:
    """Plain English for a load() / check_layout() failure."""
    if isinstance(exc, LayoutError):
        return ("Qwen-Image's runtime (stable-diffusion.cpp in code\\venvs\\qwen\\sdcpp) is not the version this "
                f"Spellcast3D was made for ({PINNED_TAG}): {exc}. Close Spellcast3D and double-click "
                "run-spellcast3d.bat again to repair it.")
    if isinstance(exc, OSError):
        return ("Qwen-Image's runtime (stable-diffusion.cpp in code\\venvs\\qwen\\sdcpp) could not be loaded "
                f"({exc}). The Microsoft Visual C++ Redistributable or the NVIDIA driver may be missing. Close "
                "Spellcast3D and double-click run-spellcast3d.bat again to repair it.")
    return f"Qwen-Image's runtime could not be checked ({type(exc).__name__}: {exc})."


if __name__ == "__main__":       # python qwen_sdcpp.py <sdcpp folder>: the layout check alone
    folder = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    dll = load(folder)
    print(f"stable-diffusion.cpp commit {commit(dll)}, version {version(dll)} (the pinned build, {PINNED_TAG}: "
          f"{'yes' if is_pinned(dll) else 'NO'})")
    check_layout(dll)
    print(f"Layout check passed: sd_ctx_params_t {ctypes.sizeof(SdCtxParams)} B, sd_img_gen_params_t "
          f"{ctypes.sizeof(SdImgGenParams)} B.")
