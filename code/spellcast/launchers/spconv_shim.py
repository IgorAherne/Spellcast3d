"""spconv's convolutions on a card its wheel has no prebuilt GEMM kernels for.

    import spconv_shim
    spconv_shim.torch_products_where_needed(capability)     # before the model code runs its first convolution

spconv-cu126 2.3.8 holds kernels for sm_60, 70, 75, 80, 86, 89 and 90; a GTX 10xx (sm_61) is not among them, nor is
an RTX 50xx (sm_120). There spconv would compile them at run time through NVRTC, which cannot work from the wheel (no
headers shipped, no include path given: "could not open source file tensorview/core/all.h" at the first
convolution; seen on a GTX 1080 Ti, not run on an RTX 50xx). On such a card spconv's Native algorithm runs with
its products on torch.mm instead, as Pixal3D's fork does for its own decoders.
For every launcher whose model runs spconv: TriFlow (TRELLIS' flow model) and LATO.2 (its V-VAE).

Only the stdlib at module level: the hub's tests import this file without torch or spconv.

LICENSE OF THIS FILE: MIT (Copyright (c) Spellcast3D's authors; the MIT terms are in "README - LICENSE.txt"), as
attn_shim.py: it is loaded into the TriFlow and LATO.2 processes, next to their own licences' code."""

from __future__ import annotations

import os


def spconv_products_in_torch(capability: tuple[int, int]) -> bool:
    """Whether spconv's GEMM kernels are not prebuilt for this card (the module doc: a GTX 10xx), so that its
    convolutions must run their products on torch.mm (install_spconv_torch_products)."""
    try:
        from spconv.core_cc.cumm.common import CompileInfo
    except Exception:       # noqa: BLE001 - no spconv: nothing to do here
        return False
    return not CompileInfo.arch_is_compiled_gemm(tuple(capability))


def install_spconv_torch_products() -> None:
    """spconv's Native convolution with the products on torch.mm, in place of its ops.indice_conv (which spconv's
    functional module looks up at every call), and SPCONV_ALGO=native so that TRELLIS' convolutions take that
    algorithm: for every kernel offset, the rows that offset pairs times the offset's [Ci, Co] slice of the KRSC
    weight, summed into the output rows - spconv's own sums, fp32 rounding order apart (the same function Pixal3D's
    fork runs for its decoders, checked against spconv's kernels there)."""
    import torch
    from spconv.pytorch import ops

    def indice_conv_torch(features, filters, indice_pairs, indice_pair_num, num_activate_out, inverse=False,
                          subm=False, algo=None, timer=None, bias=None, act_alpha=0.0, act_beta=0.0, act_type=None):
        if act_type is not None and act_type != ops.tv.gemm.Activation.None_:
            raise NotImplementedError("a fused activation")
        co, ci = filters.shape[0], filters.shape[-1]
        w = filters.reshape(co, -1, ci)                     # [Co, kv, Ci]
        kv = w.shape[1]
        centre = kv // 2
        counts = indice_pair_num.tolist()                   # the host sync spconv's own path has
        pair_in, pair_out = indice_pairs[int(inverse)], indice_pairs[int(not inverse)]
        out = torch.mm(features, w[:, centre].t()) if subm else features.new_zeros((num_activate_out, co))
        for i, n in enumerate(counts):
            if subm and i == centre:
                continue
            if subm and i > centre:                         # a subm's far half mirrors the near half's pairs
                n = counts[kv - i - 1]
            if n <= 0:
                continue
            rows = features.index_select(0, pair_in[i, :n].long())
            out.index_add_(0, pair_out[i, :n].long(), torch.mm(rows, w[:, i].t()))
        if bias is not None:
            out += bias.to(out.dtype)
        return out

    os.environ["SPCONV_ALGO"] = "native"
    ops.indice_conv = indice_conv_torch


def torch_products_where_needed(capability: tuple[int, int], always: bool = False) -> bool:
    """spconv's products on torch.mm where the card has no prebuilt kernels for them (or `always`: asked for) ->
    whether they now are."""
    needed = always or spconv_products_in_torch(capability)
    if needed:
        install_spconv_torch_products()
    return needed
