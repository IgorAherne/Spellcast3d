"""A stand-in `flash_attn` package built on torch's own attention, for the GPUs flash_attn has no kernels for.

    import attn_shim
    attn_shim.install_flash_attn_shim()      # before the model code runs `import flash_attn`

The flash_attn wheel we ship (code/pixal3d/whl, 2.8.3) holds kernels for RTX 30xx and newer only (sm_80, 90,
100, 120; no PTX). LATO.2's sparse attention (and TRELLIS' / Direct3D-S2's, which TriFlow uses) accepts only
flash_attn or xformers and imports one of them when its module loads, so an RTX 20xx or GTX 10xx could not even
import the models. install_flash_attn_shim() puts a package named `flash_attn` into sys.modules whose functions
keep flash_attn's signatures, tensor layouts and results but run torch.nn.functional.scaled_dot_product_attention
(SDPA) instead: the model code runs unchanged.

What it does per call:
- Packed and variable-length ("varlen") inputs: every sequence attends only to itself, exactly like flash_attn.
  Sequences of the same length are stacked and computed in one SDPA call (sparse windows mostly share a length),
  a lone sequence is used in place.
- Any dtype: float32 too (flash_attn takes only fp16/bf16), so a GTX 10xx runs the models in fp32. bfloat16 on a
  GPU without bf16 support (below sm_80) is computed in float32 and handed back as bfloat16; so is float16 below
  sm_70 (a GTX 10xx): SDPA's math kernel in float16 took 20x longer than in float32 there (14 s against 0.7 s for
  21,500 tokens on a GTX 1080, Spellcast3D round L), and Pascal computes float16 through float32 anyway.
  SDPA runs with torch's autocast off, as flash_attn's own kernels do: autocast would cast its inputs back to
  float16 (TriFlow samples under float16 autocast: 10 s a step on a GTX 1070 instead of 0.9).
- Memory stays bounded when SDPA would fall back to its plain "math" kernel (it materialises the whole score
  matrix: 24 GB for 20,000 tokens and 16 heads in float32): the work is then cut into slices of at most
  SCORE_BUDGET scores, over the sequences and then over the queries of one long sequence (measured: 1.2 GB peak).
  When torch says one of its tiled kernels will run (flash, memory-efficient) nothing is cut.
- Grouped/multi-query attention (fewer K/V heads), softmax_scale, causal (aligned to the bottom right corner, as
  flash_attn does) and window_size are supported; dropout_p is passed on to SDPA. softcap, alibi_slopes,
  return_attn_probs and the KV-cache call are not (they raise NotImplementedError: no model here uses them).

Measured against flash_attn 2.8.3 on an RTX 3060 (sm_86), shim forced: all six functions within rounding (max abs
difference 1e-3 in float16, 8e-3 in bfloat16, varlen with 40-500 token windows, causal, local, grouped heads);
LATO.2's vertex stage (24 steps, ~19k tokens) 41-45 s instead of 38-39 s, its clean meshes as good (Chamfer
0.30% vs 0.32% of the diagonal on a Pixal3D bear).

Only the stdlib at module level (torch is imported inside the functions), so the hub's tests import this file.
Shared by launchers/lato2_server.py and launchers/triflow_server.py; it wraps a third-party library only.

LICENSE OF THIS FILE: MIT (Copyright (c) Spellcast3D's authors; the MIT terms are in "README - LICENSE.txt"),
not AGPL like most of Spellcast3D: it is loaded into the TriFlow process (launchers/triflow_server.py), next to
TriFlow's ADPNCL code.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from typing import Any

__version__ = "2.8.3+sdpa"   # what `flash_attn.__version__` reads through the shim

# The largest score matrix (sequences x heads x queries x keys) one SDPA call gets: 2^27 scores are 512 MB in
# float32 when SDPA uses its math kernel, which then needs about twice that for the softmax.
SCORE_BUDGET = 1 << 27

SHIM_ATTR = "__spellcast_sdpa_shim__"   # marks the shim modules in sys.modules
FUNCTIONS = ("flash_attn_func", "flash_attn_kvpacked_func", "flash_attn_qkvpacked_func",
             "flash_attn_varlen_func", "flash_attn_varlen_kvpacked_func", "flash_attn_varlen_qkvpacked_func",
             "flash_attn_with_kvcache")

_BF16_OK: dict[Any, bool] = {}          # device -> whether it computes in bfloat16 natively
_EFFICIENT: dict[Any, bool] = {}        # (device, dtype, head sizes, dropout, backend flags) -> tiled SDPA kernel


# ---- installing ------------------------------------------------------------------------------------------------

def install_flash_attn_shim() -> types.ModuleType:
    """Register the shim as `flash_attn` (and `flash_attn.flash_attn_interface`, where the real package defines
    the functions) in sys.modules and return it. A real flash_attn already imported is replaced; call this
    before the model code imports flash_attn. Idempotent."""
    current = sys.modules.get("flash_attn")
    if current is not None and getattr(current, SHIM_ATTR, False):
        return current
    for name in [n for n in sys.modules if n == "flash_attn" or n.startswith("flash_attn.")]:
        del sys.modules[name]
    package = _module("flash_attn", is_package=True)
    interface = _module("flash_attn.flash_attn_interface")
    this = sys.modules[__name__]
    for name in FUNCTIONS:
        setattr(package, name, getattr(this, name))
        setattr(interface, name, getattr(this, name))
    package.flash_attn_interface = interface
    package.__version__ = __version__
    sys.modules["flash_attn"] = package
    sys.modules["flash_attn.flash_attn_interface"] = interface
    return package


def is_shim_installed() -> bool:
    """True when `import flash_attn` gives this shim."""
    return bool(getattr(sys.modules.get("flash_attn"), SHIM_ATTR, False))


def _module(name: str, is_package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name, "SDPA stand-in for flash_attn (Spellcast3D launchers/attn_shim.py)")
    # importlib.util.find_spec() raises for a module in sys.modules without a spec (transformers asks it)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=is_package)
    if is_package:
        module.__path__ = []
    module.__file__ = __file__
    setattr(module, SHIM_ATTR, True)
    return module


# ---- flash_attn's functions (same signatures and layouts: tokens, heads, head size) ------------------------------

def flash_attn_func(q: Any, k: Any, v: Any, dropout_p: float = 0.0, softmax_scale: float | None = None,
                    causal: bool = False, window_size: tuple[int, int] = (-1, -1), softcap: float = 0.0,
                    alibi_slopes: Any = None, deterministic: bool = False, return_attn_probs: bool = False) -> Any:
    """q (batch, seqlen_q, heads, d), k/v (batch, seqlen_k, heads_k, d) -> (batch, seqlen_q, heads, d_v)."""
    _unsupported(softcap, alibi_slopes, return_attn_probs)
    return _attend(q, k, v, dropout_p, softmax_scale, causal, window_size)


def flash_attn_qkvpacked_func(qkv: Any, dropout_p: float = 0.0, softmax_scale: float | None = None,
                              causal: bool = False, window_size: tuple[int, int] = (-1, -1), softcap: float = 0.0,
                              alibi_slopes: Any = None, deterministic: bool = False,
                              return_attn_probs: bool = False) -> Any:
    """qkv (batch, seqlen, 3, heads, d) -> (batch, seqlen, heads, d)."""
    _unsupported(softcap, alibi_slopes, return_attn_probs)
    return _attend(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], dropout_p, softmax_scale, causal, window_size)


def flash_attn_kvpacked_func(q: Any, kv: Any, dropout_p: float = 0.0, softmax_scale: float | None = None,
                             causal: bool = False, window_size: tuple[int, int] = (-1, -1), softcap: float = 0.0,
                             alibi_slopes: Any = None, deterministic: bool = False,
                             return_attn_probs: bool = False) -> Any:
    """q (batch, seqlen_q, heads, d), kv (batch, seqlen_k, 2, heads_k, d) -> (batch, seqlen_q, heads, d)."""
    _unsupported(softcap, alibi_slopes, return_attn_probs)
    return _attend(q, kv[:, :, 0], kv[:, :, 1], dropout_p, softmax_scale, causal, window_size)


def flash_attn_varlen_func(q: Any, k: Any, v: Any, cu_seqlens_q: Any, cu_seqlens_k: Any, max_seqlen_q: int,
                           max_seqlen_k: int, dropout_p: float = 0.0, softmax_scale: float | None = None,
                           causal: bool = False, window_size: tuple[int, int] = (-1, -1), softcap: float = 0.0,
                           alibi_slopes: Any = None, deterministic: bool = False, return_attn_probs: bool = False,
                           block_table: Any = None) -> Any:
    """q (total_q, heads, d), k/v (total_k, heads_k, d); sequence i is rows cu_seqlens[i]:cu_seqlens[i+1]
    -> (total_q, heads, d_v)."""
    _unsupported(softcap, alibi_slopes, return_attn_probs)
    if block_table is not None:
        raise NotImplementedError("flash_attn shim: paged K/V (block_table) is not supported")
    return _varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, dropout_p, softmax_scale, causal, window_size)


def flash_attn_varlen_qkvpacked_func(qkv: Any, cu_seqlens: Any, max_seqlen: int, dropout_p: float = 0.0,
                                     softmax_scale: float | None = None, causal: bool = False,
                                     window_size: tuple[int, int] = (-1, -1), softcap: float = 0.0,
                                     alibi_slopes: Any = None, deterministic: bool = False,
                                     return_attn_probs: bool = False) -> Any:
    """qkv (total, 3, heads, d) -> (total, heads, d)."""
    _unsupported(softcap, alibi_slopes, return_attn_probs)
    return _varlen(qkv[:, 0], qkv[:, 1], qkv[:, 2], cu_seqlens, cu_seqlens, dropout_p, softmax_scale, causal,
                   window_size)


def flash_attn_varlen_kvpacked_func(q: Any, kv: Any, cu_seqlens_q: Any, cu_seqlens_k: Any, max_seqlen_q: int,
                                    max_seqlen_k: int, dropout_p: float = 0.0, softmax_scale: float | None = None,
                                    causal: bool = False, window_size: tuple[int, int] = (-1, -1),
                                    softcap: float = 0.0, alibi_slopes: Any = None, deterministic: bool = False,
                                    return_attn_probs: bool = False) -> Any:
    """q (total_q, heads, d), kv (total_k, 2, heads_k, d) -> (total_q, heads, d)."""
    _unsupported(softcap, alibi_slopes, return_attn_probs)
    return _varlen(q, kv[:, 0], kv[:, 1], cu_seqlens_q, cu_seqlens_k, dropout_p, softmax_scale, causal,
                   window_size)


def flash_attn_with_kvcache(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("flash_attn shim: flash_attn_with_kvcache (generation with a K/V cache) is not "
                              "supported")


def _unsupported(softcap: float, alibi_slopes: Any, return_attn_probs: bool) -> None:
    if softcap and softcap > 0.0:
        raise NotImplementedError("flash_attn shim: softcap is not supported")
    if alibi_slopes is not None:
        raise NotImplementedError("flash_attn shim: alibi_slopes are not supported")
    if return_attn_probs:
        raise NotImplementedError("flash_attn shim: return_attn_probs is not supported")


# ---- the work --------------------------------------------------------------------------------------------------

def _varlen(q: Any, k: Any, v: Any, cu_q: Any, cu_k: Any, dropout_p: float, scale: float | None, causal: bool,
            window: tuple[int, int]) -> Any:
    """Packed sequences -> SDPA per sequence, sequences of the same (query, key) lengths in one call."""
    import torch

    starts_q = [int(x) for x in cu_q.tolist()]
    starts_k = [int(x) for x in cu_k.tolist()]
    if len(starts_q) != len(starts_k):
        raise ValueError(f"flash_attn shim: {len(starts_q) - 1} query sequences but {len(starts_k) - 1} key ones")
    count = len(starts_q) - 1
    out = q.new_zeros((q.shape[0], q.shape[1], v.shape[-1]))
    if count <= 0:
        return out
    if count == 1:          # one sequence: in place, no copies
        a, b, c, d = starts_q[0], starts_q[1], starts_k[0], starts_k[1]
        if b > a and d > c:
            out[a:b] = _attend(q[a:b][None], k[c:d][None], v[c:d][None], dropout_p, scale, causal, window)[0]
        return out
    groups: dict[tuple[int, int], list[int]] = {}
    for i in range(count):
        groups.setdefault((starts_q[i + 1] - starts_q[i], starts_k[i + 1] - starts_k[i]), []).append(i)
    device = q.device
    for (len_q, len_k), members in groups.items():
        if len_q <= 0 or len_k <= 0:
            continue        # no queries, or nothing to attend to: zeros, as flash_attn gives
        first_q = torch.tensor([starts_q[i] for i in members], device=device, dtype=torch.long)
        first_k = torch.tensor([starts_k[i] for i in members], device=device, dtype=torch.long)
        rows_q = first_q[:, None] + torch.arange(len_q, device=device)[None, :]     # (sequences, len_q)
        rows_k = first_k[:, None] + torch.arange(len_k, device=device)[None, :]
        out[rows_q] = _attend(q[rows_q], k[rows_k], v[rows_k], dropout_p, scale, causal, window)
    return out


def _attend(q: Any, k: Any, v: Any, dropout_p: float, scale: float | None, causal: bool,
            window: tuple[int, int]) -> Any:
    """q (batch, len_q, heads, d), k/v (batch, len_k, heads_k, d) -> (batch, len_q, heads, d_v) in q's dtype."""
    import torch
    import torch.nn.functional as F

    batch, len_q, heads, _ = q.shape
    len_k, heads_k = k.shape[1], k.shape[2]
    out_dtype = q.dtype
    dtype = _compute_dtype(q)
    if heads_k != heads:
        if heads % heads_k:
            raise ValueError(f"flash_attn shim: {heads} query heads cannot share {heads_k} key/value heads")
        k = k.repeat_interleave(heads // heads_k, dim=2)
        v = v.repeat_interleave(heads // heads_k, dim=2)
    # SDPA's layout: (batch, heads, tokens, d)
    q, k, v = (t.to(dtype).transpose(1, 2) for t in (q, k, v))
    left, right = (int(window[0]), int(window[1])) if window is not None else (-1, -1)
    if causal:
        right = 0
    masked = left >= 0 or right >= 0

    per_sequence = heads * len_q * len_k
    if not masked and _efficient(q, k, v, dropout_p):
        step_b, step_q = batch, len_q          # a tiled kernel: no score matrix to bound
    elif per_sequence <= SCORE_BUDGET:
        step_b, step_q = max(1, SCORE_BUDGET // per_sequence), len_q
    else:
        step_b, step_q = 1, max(1, SCORE_BUDGET // (heads * len_k))
    # in the dtype chosen above: autocast would cast SDPA's inputs to its own dtype (the module doc)
    with torch.autocast(q.device.type, enabled=False):
        if step_b >= batch and step_q >= len_q and not masked:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=scale)
            return out.transpose(1, 2).to(out_dtype)

        out = q.new_empty((batch, heads, len_q, v.shape[-1]))
        for b0 in range(0, batch, step_b):
            b1 = min(batch, b0 + step_b)
            for q0 in range(0, len_q, step_q):
                q1 = min(len_q, q0 + step_q)
                mask, blind = _band(q0, q1, len_q, len_k, left, right, q.device) if masked else (None, None)
                block = F.scaled_dot_product_attention(q[b0:b1, :, q0:q1], k[b0:b1], v[b0:b1], attn_mask=mask,
                                                       dropout_p=dropout_p, scale=scale)
                if blind is not None:
                    block[:, :, blind] = 0      # queries that may see no key: zeros, as flash_attn gives
                out[b0:b1, :, q0:q1] = block
    return out.transpose(1, 2).to(out_dtype)


def _band(q0: int, q1: int, len_q: int, len_k: int, left: int, right: int, device: Any) -> tuple[Any, Any]:
    """(mask, blind) for the queries q0..q1: mask says which keys each may see (flash_attn's local/causal rule,
    aligned to the bottom right corner: query i sits at key position i + len_k - len_q; -1 = no limit on that
    side); blind indexes the queries that may see none (None when there are none). Those get one key in the mask
    so SDPA gives no NaN, and zeros afterwards."""
    import torch

    pos = torch.arange(q0, q1, device=device)[:, None] + (len_k - len_q)
    keys = torch.arange(len_k, device=device)[None, :]
    keep = torch.ones((q1 - q0, len_k), dtype=torch.bool, device=device)
    if left >= 0:
        keep &= keys >= pos - left
    if right >= 0:
        keep &= keys <= pos + right
    empty = ~keep.any(dim=1)
    if not bool(empty.any()):
        return keep, None
    blind = empty.nonzero(as_tuple=True)[0]
    keep[blind, 0] = True
    return keep, blind


def _efficient(q: Any, k: Any, v: Any, dropout_p: float) -> bool:
    """True when SDPA will run one of its tiled kernels (flash or memory-efficient) on these tensors, which never
    hold the whole score matrix; asked once per device, dtype and head sizes (what decides it, besides settings)."""
    import torch

    key = (q.device, q.dtype, q.shape[-1], v.shape[-1], dropout_p > 0, _backend_flags())
    found = _EFFICIENT.get(key)
    if found is None:
        found = False
        if q.is_cuda:
            try:
                params = torch.backends.cuda.SDPAParams(q[:1, :, :8], k[:1, :, :8], v[:1, :, :8], None,
                                                        float(dropout_p), False, False)
                found = bool(torch.backends.cuda.can_use_flash_attention(params, False)
                             or torch.backends.cuda.can_use_efficient_attention(params, False))
            except Exception:     # another torch: slice to be safe
                found = False
        _EFFICIENT[key] = found
    return found


def _backend_flags() -> tuple[bool, ...]:
    import torch

    return (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled())


def _compute_dtype(q: Any) -> Any:
    """q's dtype, or float32 for bfloat16 on a GPU that cannot compute in it (below sm_80: SDPA's efficient
    kernels refuse it there and its math kernel emulates it slowly), and for float16 below sm_70 (the module doc:
    20x slower than float32 on a GTX 1080)."""
    import torch

    if q.dtype in (torch.bfloat16, torch.float16) and q.is_cuda:
        ok = _BF16_OK.get((q.device, q.dtype))
        if ok is None:
            major = torch.cuda.get_device_capability(q.device)[0]
            ok = _BF16_OK[(q.device, q.dtype)] = major >= (8 if q.dtype == torch.bfloat16 else 7)
        if not ok:
            return torch.float32
    if q.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return q.dtype
    return torch.float32
