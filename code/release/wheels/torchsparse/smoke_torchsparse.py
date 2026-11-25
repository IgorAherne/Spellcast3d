"""GPU smoke test for the torchsparse wheel: sparse Conv3d vs a dense torch conv3d reference (float64),
for the default dataflow (ImplicitGEMM), ImplicitGEMM with sorted kernel maps, and GatherScatter
(the dataflow planned for GPUs below sm_75, selected through torchsparse's global conv config),
in fp32, fp16 and mixed (fp16 features + fp32 weights, what fp16 autocast feeds it).

Ops (TriFlow's Direct3D-S2 VAE uses torchsparse Conv3d with 27-tap kernels):
  subm3  : 3x3x3 submanifold conv, stride 1            (implicit-GEMM / gather-scatter forward)
  down2  : 2x2x2 conv, stride 2                          (downsample kernel map, patch P4)
  up2    : 2x2x2 transposed conv back to the fine coords (transposed kernel map)
  bwd    : fp32 autograd grads of subm3 vs dense autograd (wgrad kernels, patch P2 files)

    python smoke_torchsparse.py            (needs torch with CUDA, numpy, tqdm, rootpath+six)

Prints one JSON document; exit code 0 = every check passed.
"""
from __future__ import annotations

import copy
import json
import sys
import time

import torch
import torch.nn.functional as TF

import torchsparse
import torchsparse.nn as spnn
from torchsparse import SparseTensor
from torchsparse.nn.functional.conv import conv_config as cc
from torchsparse.nn.functional.conv.conv_config import Dataflow
from torchsparse.nn.utils import get_kernel_offsets

DEV = "cuda"
R = 40            # grid size per axis
B = 2             # batch size
TOL = {"fp32": 5e-3, "fp16": 2e-2, "mixed": 2e-2}   # max|err| / max|ref| (fp32 ImplicitGEMM runs TF32 on >= sm_80)


def make_coords(seed: int) -> torch.Tensor:
    """Surface-like voxels (two noisy spherical shells, one per batch item) + scattered points."""
    g = torch.Generator().manual_seed(seed)
    pts = []
    for b in range(B):
        n = 6000
        d = torch.randn(n, 3, generator=g)
        d = d / d.norm(dim=1, keepdim=True)
        r = 13 + 3 * b + torch.randn(n, 1, generator=g) * 0.6
        p = (d * r + R / 2).round().long().clamp(0, R - 1)
        q = torch.randint(0, R, (400, 3), generator=g)
        p = torch.cat([p, q])
        pts.append(torch.cat([torch.full((len(p), 1), b), p], 1))
    c = torch.unique(torch.cat(pts), dim=0)
    return c.int().to(DEV)


def dense(coords: torch.Tensor, feats: torch.Tensor, size: int) -> torch.Tensor:
    x = torch.zeros(B, feats.shape[1], size, size, size, dtype=torch.float64, device=DEV)
    c = coords.long()
    x[c[:, 0], :, c[:, 1], c[:, 2], c[:, 3]] = feats.double()
    return x


def gather(x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    c = coords.long()
    return x[c[:, 0], :, c[:, 1], c[:, 2], c[:, 3]]


def dense_weight(W: torch.Tensor, ksize: int, transposed: bool) -> torch.Tensor:
    """torchsparse kernel [K, Cin, Cout] -> dense kernel, tap k at offset get_kernel_offsets(k)."""
    offs = get_kernel_offsets(ksize, 1, 1, device="cpu").long()
    lo = offs.min().item()
    K, cin, cout = W.shape
    if transposed:   # conv_transpose3d weight: [Cin, Cout, kD, kH, kW]
        w = torch.zeros(cin, cout, ksize, ksize, ksize, dtype=torch.float64, device=DEV)
        for k in range(K):
            ox, oy, oz = (offs[k] - lo).tolist()
            w[:, :, ox, oy, oz] = W[k].double()
    else:            # conv3d weight: [Cout, Cin, kD, kH, kW]; out(p) = sum_k W[k]^T in(p + off_k)
        w = torch.zeros(cout, cin, ksize, ksize, ksize, dtype=torch.float64, device=DEV)
        for k in range(K):
            ox, oy, oz = (offs[k] - lo).tolist()
            w[:, :, ox, oy, oz] = W[k].double().t()
    return w


def rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.double() - ref).abs().max() / ref.abs().max().clamp_min(1e-12))


def set_dataflow(name: str) -> None:
    # get_default_conv_config() returns the module-level default object itself: copy it, do not mutate it.
    cfg = cc.AttributeDict(dict(cc.get_default_conv_config()))
    cfg.dataflow = {"ImplicitGEMM": Dataflow.ImplicitGEMM, "ImplicitGEMM_sorted": Dataflow.ImplicitGEMM,
                    "GatherScatter": Dataflow.GatherScatter}[name]
    cfg.ifsort = name == "ImplicitGEMM_sorted"
    cc.set_global_conv_config(cfg)


def run_case(dataflow: str, prec: str, coords, feats64, convs) -> dict:
    set_dataflow(dataflow)
    fdt = torch.float32 if prec == "fp32" else torch.float16
    wdt = torch.float16 if prec == "fp16" else torch.float32
    subm, down, up = (copy.deepcopy(m).to(DEV, wdt).eval() for m in convs)
    res = {}
    with torch.no_grad():
        x = SparseTensor(feats=feats64.to(fdt), coords=coords)
        t = time.perf_counter()
        y = subm(x)
        torch.cuda.synchronize()
        res["subm3_ms"] = round((time.perf_counter() - t) * 1e3, 2)
        ref = gather(TF.conv3d(dense(coords, feats64.to(fdt), R), dense_weight(subm.kernel, 3, False), padding=1), coords)
        res["subm3_same_coords"] = bool(torch.equal(y.coords, coords))
        res["subm3_rel_err"] = rel_err(y.feats, ref)
        y2 = subm(SparseTensor(feats=feats64.to(fdt), coords=coords))
        res["subm3_repeat_max_diff"] = float((y.feats.double() - y2.feats.double()).abs().max())

        z = down(x)
        oc = z.coords
        units = "fine" if bool((oc[:, 1:] % 2 == 0).all()) and int(oc[:, 1:].max()) > R // 2 else "coarse"
        oc_coarse = oc.clone()
        if units == "fine":
            oc_coarse[:, 1:] //= 2
        dref = TF.conv3d(dense(coords, feats64.to(fdt), R), dense_weight(down.kernel, 2, False), stride=2)
        expect = (dense(coords, torch.ones(len(coords), 1, device=DEV), R)
                  .reshape(B, 1, R // 2, 2, R // 2, 2, R // 2, 2).sum((3, 5, 7)) > 0).nonzero()
        res["down2_coord_units"] = units
        res["down2_coords_match"] = bool(len(oc) == len(expect) and torch.equal(
            torch.unique(oc_coarse.long(), dim=0), torch.unique(expect[:, [0, 2, 3, 4]], dim=0)))
        res["down2_rel_err"] = rel_err(z.feats, gather(dref, oc_coarse))

        u = up(z)
        uref = TF.conv_transpose3d(dense(oc_coarse, z.feats.double(), R // 2), dense_weight(up.kernel, 2, True), stride=2)
        res["up2_same_coords"] = bool(torch.equal(u.coords, coords))
        res["up2_rel_err"] = rel_err(u.feats, gather(uref, coords))
    tol = TOL[prec]
    res["ok"] = (res["subm3_same_coords"] and res["subm3_rel_err"] < tol and res["subm3_repeat_max_diff"] == 0.0
                 and res["down2_coords_match"] and res["down2_rel_err"] < tol
                 and res["up2_same_coords"] and res["up2_rel_err"] < tol)
    return res


def backward_case(dataflow: str, coords, feats64, subm0) -> dict:
    set_dataflow(dataflow)
    subm = copy.deepcopy(subm0).to(DEV, torch.float32).train()
    f = feats64.float().clone().requires_grad_(True)
    y = subm(SparseTensor(feats=f, coords=coords))
    g = torch.randn_like(y.feats)
    (y.feats * g).sum().backward()
    fd = feats64.clone().requires_grad_(True)
    wd = dense_weight(subm.kernel.detach(), 3, False).requires_grad_(True)
    yd = gather(TF.conv3d(dense(coords, fd, R), wd, padding=1), coords)
    (yd * g.double()).sum().backward()
    # dense weight grad back to torchsparse layout [K, Cin, Cout]
    offs = get_kernel_offsets(3, 1, 1, device="cpu").long() + 1
    gw = torch.stack([wd.grad[:, :, ox, oy, oz].t() for ox, oy, oz in offs.tolist()])
    r = {"dgrad_rel_err": rel_err(f.grad, fd.grad), "wgrad_rel_err": rel_err(subm.kernel.grad, gw)}
    r["ok"] = r["dgrad_rel_err"] < TOL["fp32"] and r["wgrad_rel_err"] < TOL["fp32"]
    return r


def main() -> int:
    torch.manual_seed(0)
    cap = torch.cuda.get_device_capability()
    out = {"torch": torch.__version__, "torchsparse": torchsparse.__version__,
           "torchsparse_file": torchsparse.__file__, "gpu": torch.cuda.get_device_name(),
           "capability": f"sm_{cap[0]}{cap[1]}",
           "backends": {"allow_fp16": torchsparse.backends.allow_fp16, "allow_tf32": torchsparse.backends.allow_tf32},
           "grid": R, "batch": B, "cases": {}}
    coords = make_coords(0)
    out["n_voxels"] = len(coords)
    channel_sets = [(32, 64), (16, 32)]
    all_ok = True
    for cin, cout in channel_sets:
        feats64 = torch.randn(len(coords), cin, dtype=torch.float64, device=DEV)
        convs = (spnn.Conv3d(cin, cout, 3), spnn.Conv3d(cin, cout, 2, stride=2),
                 spnn.Conv3d(cout, cin, 2, stride=2, transposed=True))
        for dataflow in ("ImplicitGEMM", "ImplicitGEMM_sorted", "GatherScatter"):
            for prec in ("fp32", "fp16", "mixed"):
                key = f"{cin}->{cout} {dataflow} {prec}"
                try:
                    r = run_case(dataflow, prec, coords, feats64, convs)
                except Exception as e:  # noqa: BLE001 - report every failure, keep going
                    r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                out["cases"][key] = r
                all_ok &= r["ok"]
        for dataflow in ("ImplicitGEMM", "GatherScatter"):
            key = f"{cin}->{cout} {dataflow} fp32 backward"
            try:
                r = backward_case(dataflow, coords, feats64, convs[0])
            except Exception as e:  # noqa: BLE001
                r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            out["cases"][key] = r
            all_ok &= r["ok"]

    # Emulate a GPU below sm_75 on this card: torchsparse.backends sets allow_fp16 = cc >= 7.5 and
    # allow_tf32 = cc >= 8.0 at import. With both False, fp16 ImplicitGEMM must refuse (upstream
    # behaviour), fp32 ImplicitGEMM falls back to the plain-CUDA f32 kernels, and GatherScatter must
    # work in every precision: the path a launcher selects for GTX 10xx / sm_70.
    saved = (torchsparse.backends.allow_fp16, torchsparse.backends.allow_tf32)
    torchsparse.backends.allow_fp16, torchsparse.backends.allow_tf32 = False, False
    cin, cout = 32, 64
    feats64 = torch.randn(len(coords), cin, dtype=torch.float64, device=DEV)
    convs = (spnn.Conv3d(cin, cout, 3), spnn.Conv3d(cin, cout, 2, stride=2),
             spnn.Conv3d(cout, cin, 2, stride=2, transposed=True))
    for dataflow, prec in (("ImplicitGEMM", "fp32"), ("GatherScatter", "fp32"), ("GatherScatter", "fp16"),
                           ("GatherScatter", "mixed")):
        key = f"pre-sm75 emulation {cin}->{cout} {dataflow} {prec}"
        try:
            r = run_case(dataflow, prec, coords, feats64, convs)
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        out["cases"][key] = r
        all_ok &= r["ok"]
    key = f"pre-sm75 emulation {cin}->{cout} ImplicitGEMM fp16 refuses"
    try:
        run_case("ImplicitGEMM", "fp16", coords, feats64, convs)
        r = {"ok": False, "error": "fp16 ImplicitGEMM ran although allow_fp16=False"}
    except RuntimeError as e:
        r = {"ok": "not supported" in str(e), "raised": str(e)}
    out["cases"][key] = r
    all_ok &= r["ok"]
    torchsparse.backends.allow_fp16, torchsparse.backends.allow_tf32 = saved

    cc.clear_global_conv_config()
    out["passed"] = bool(all_ok)
    print(json.dumps(out, indent=1))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
