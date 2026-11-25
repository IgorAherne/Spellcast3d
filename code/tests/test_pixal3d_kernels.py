"""The flows' hand-off kernels (code/pixal3d/pixal3d/modules/int8mm.py norm_quant) on the CPU: what the install's
warm step (int8mm.warm_handoffs, called by calibrate.py) compiles against what a block of the detail flows calls.
Triton keys a compiled kernel on its tensors' dtypes and its constants, so a warm with other dtypes leaves the
blocks' own kernel to be compiled at the first prop. Both sides are recorded here (norm_quant replaced by a
recorder), on tensors of the meta device: no kernel runs. Then the old cards' element-wise kernels
(modules/old_cards.py): which kinds serve, as the install's race recorded them.

Runs in the pixal3d venv, with the card hidden and pytest from code/logs/profiling/pytools:
    set CUDA_VISIBLE_DEVICES=-1
    set PYTHONPATH=code\\logs\\profiling\\pytools
    code\\venvs\\pixal3d\\Scripts\\python.exe -m pytest -p no:cacheprovider code\\tests\\test_pixal3d_kernels.py -q
Skipped in the hub venv (no torch).
"""

from __future__ import annotations

import sys
from functools import partial
from typing import Any

import pytest

from spellcast import paths

torch = pytest.importorskip("torch")
if not (paths.PIXAL3D_DIR / "pixal3d" / "modules" / "int8mm.py").is_file():
    pytest.skip("the Pixal3D checkout is not here", allow_module_level=True)

# imported while this file is collected and taken off the path again at once (as test_pixal3d_weights.py does)
sys.path.insert(0, str(paths.PIXAL3D_DIR))
try:
    from pixal3d import calibration                                     # noqa: E402
    from pixal3d.modules import fused, int8, int8mm, old_cards, precision  # noqa: E402
    from pixal3d.modules.sparse.transformer import modulated            # noqa: E402
    from pixal3d.modules.utils import convert_module_to                 # noqa: E402
finally:
    sys.path.remove(str(paths.PIXAL3D_DIR))

C = 1536            # the flows' width: warm_handoffs warms at it


def _compiled_as(x: Any, kw: dict) -> tuple:
    """What Triton compiles a norm_quant call as: its tensors' dtypes and the kernel's forms (constants, warps)."""
    def dtype(name: str) -> Any:
        return None if kw.get(name) is None else kw[name].dtype

    return (x.dtype, dtype("weight"), dtype("bias"), dtype("scale"), dtype("shift"), kw.get("batch_index") is None,
            kw.get("inv_smooth") is None, kw.get("group", 0), kw.get("rounded", True), kw.get("num_warps", 0))


def _recording(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []
    monkeypatch.setattr(int8mm, "norm_quant", lambda x, eps, **kw: calls.append(_compiled_as(x, kw)))
    return calls


def _warmed(monkeypatch: pytest.MonkeyPatch) -> list:
    """The norm_quant calls of the install's warm step, as on a card whose int8 GEMMs are Triton's."""
    calls = _recording(monkeypatch)
    monkeypatch.setattr(int8mm, "ready", lambda device: (True, ""))
    monkeypatch.setattr(int8mm, "gemm_path", lambda: "triton")
    monkeypatch.setattr(int8mm, "quantize_weight", lambda w, *args, **kwargs: (w, w))
    monkeypatch.setattr(int8mm, "int8_linear", lambda *args, **kwargs: None)
    monkeypatch.setattr(fused, "active", lambda feats: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    assert int8mm.warm_handoffs(torch.device("cpu"), rows=(8,)) > 0
    return calls


def _block_handoffs(monkeypatch: pytest.MonkeyPatch, norm: str) -> list:
    """The norm_quant calls of one hr block's fused path (modulated.py _forward_fused, one prop) at `norm`: the
    block built at the flows' width with its torso cast as the flow casts it, its grains the detail policy's."""
    with torch.device("meta"):
        block = modulated.ModulatedSparseTransformerCrossBlock(C, 1024, num_heads=12)
        mod = torch.empty((1, 6 * C), dtype=torch.bfloat16)
        feats = torch.empty((4099, C), dtype=torch.bfloat16)
    block.apply(partial(convert_module_to, dtype=torch.bfloat16))
    shift_msa, scale_msa, _, shift_mlp, scale_mlp, _ = mod.chunk(6, dim=1)
    policy = int8.POLICIES["detail"]["hr"]
    grain = {kind: int8mm.GROUP if policy[kind] == "group" else 0 for kind in ("qkv", "xq", "fc1")}
    rows = {"batch": 1, "rows_per_batch": feats.shape[0]}
    calls = _recording(monkeypatch)
    if norm == "norm2":
        modulated._norm_quant(feats, block.norm2.eps, grain["xq"], rows, weight=block.norm2.weight,
                              bias=block.norm2.bias)
    else:
        modulated._norm_quant(feats, block.norm1.eps, grain["qkv"], rows, scale=scale_msa, shift=shift_msa)
        modulated._norm_quant(feats, block.norm3.eps, grain["fc1"], rows, scale=scale_mlp, shift=shift_mlp)
    return calls


def test_the_install_warms_the_modulated_norms_hand_offs_as_the_blocks_call_them(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """norm1 -> to_qkv and norm3 -> fc1 (bf16 scale and shift, group scales): the kernels the blocks run are the
    ones the warm step compiled."""
    want = _block_handoffs(monkeypatch, "modulated")
    warmed = _warmed(monkeypatch)
    assert want and all(call in warmed for call in want), (want, warmed)


def test_the_install_warms_the_cross_attentions_norm_hand_off_with_norm2s_fp32_weight_and_bias(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """norm2 -> the cross attention's to_q (per row): LayerNorm32 keeps its weight and bias in fp32 when the flow
    casts its torso (the staged copies say F32 too), so the warm step compiles the affine form with fp32 ones."""
    want = _block_handoffs(monkeypatch, "norm2")
    assert want and want[0][1] == want[0][2] == torch.float32
    warmed = _warmed(monkeypatch)
    assert all(call in warmed for call in want), (want, warmed)


# ---- the old cards' element-wise kernels: the install's race read back (modules/old_cards.py) ----------------------

RTX_2070 = "NVIDIA GeForce RTX 2070"


@pytest.fixture
def old_card(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A process on the turing profile of an RTX 2070 with pixal3d_kernels 0.2.0 and a calibration record in
    tmp_path (written by the returned function: the old_cards section, or None for a record without it)."""
    import json

    record = tmp_path / "calibration.json"
    monkeypatch.setenv(calibration.ENV, str(record))
    monkeypatch.setenv("PIXAL3D_PRECISION", "turing")
    monkeypatch.delenv("PIXAL3D_OLD_CARD_KERNELS", raising=False)
    monkeypatch.setattr(calibration, "card_name", lambda device=None: RTX_2070)
    monkeypatch.setattr(old_cards, "_kernels_build", lambda: "0.2.0")
    monkeypatch.setattr(old_cards, "_state", {"ok": None, "why": "", "on": {}})
    precision.reset()
    calibration.forget()

    def write(section: Any, card: str = RTX_2070) -> None:
        body = {"version": 3, "complete": True, "card": card}
        if section is not None:
            body[old_cards.SECTION] = section
        record.write_text(json.dumps(body), encoding="utf-8")
        calibration.forget()

    yield write
    precision.reset()
    calibration.forget()


def test_the_old_cards_kernels_serve_as_the_installs_race_chose(old_card: Any,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """Each kind serves where the race of this card, profile and kernels build kept it; without such a race (no
    record, another card, profile or build, a failed race) every kind serves, as before the race. force: every kind,
    whatever the race said; the first-use check decides either way (ready)."""
    every = dict.fromkeys(old_cards.KINDS, True)
    raced = {"ok": True, "profile": "turing", "kernels": "0.2.0",
             "on": {"quantize_rows": True, "modulate": False, "residual_gated": True, "qk_norm_rope": False}}
    old_card(None)
    assert old_cards.chosen() == every                                  # a record without the race
    old_card(raced)
    assert old_cards.chosen() == raced["on"] and old_cards.race_holds(raced)
    for other in ({"profile": "pascal"}, {"kernels": "0.1.0"}, {"ok": False, "error": "x"}, {"on": None}):
        old_card({**raced, **other})
        assert old_cards.chosen() == every, other
    old_card(raced, card="NVIDIA GeForce GTX 1080")
    assert old_cards.chosen() == every                                  # made on another card
    old_card({**raced, "on": {"modulate": False}})
    assert old_cards.chosen() == {**every, "modulate": False}           # a kind the race did not name serves
    old_card(raced)
    monkeypatch.setenv("PIXAL3D_OLD_CARD_KERNELS", "force")
    assert old_cards.chosen() == every
    monkeypatch.setenv("PIXAL3D_OLD_CARD_KERNELS", "on")
    assert old_cards.chosen() == raced["on"]

    # the dispatch: a kind the race left out returns None (the caller's eager code), the others go on
    class OnTheCard:
        is_cuda, device = True, "cuda"

    old_cards._state.update(ok=True, on=raced["on"])
    with torch.no_grad():
        assert old_cards.quantiser_ok("cuda") and old_cards._usable(OnTheCard(), "residual_gated")
        assert not old_cards._usable(OnTheCard(), "modulate") and not old_cards._usable(OnTheCard(), "qk_norm_rope")
        old_cards._state.update(on={**raced["on"], "quantize_rows": False})
        assert not old_cards.quantiser_ok("cuda")
        old_cards._state.update(ok=False)
        assert not old_cards.quantiser_ok("cuda") and not old_cards._usable(OnTheCard(), "residual_gated")


def test_the_old_cards_race_does_nothing_on_the_ampere_profile(old_card: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ampere and newer run the flows' chains in Triton's fused kernels: the race records that it skipped, times
    nothing (no card is asked here), and a record's skipped section never holds as a race."""
    monkeypatch.setenv("PIXAL3D_PRECISION", "ampere")
    precision.reset()
    found = old_cards.calibrate("cuda")
    assert found["ok"] is True and "ampere" in found["skipped"] and not old_cards.race_holds(found)
    monkeypatch.setenv("PIXAL3D_OLD_CARD_KERNELS", "off")
    assert old_cards._mode() == "off" and not old_cards._switch()
    monkeypatch.setenv("PIXAL3D_OLD_CARD_KERNELS", "1")
    assert old_cards._mode() == "" and old_cards._switch()


# ---- Spellcast3D round O: the dp4a GEMM's twins (int8mm.DP4A_TWINS), raced at install, the shipped kernel by default

_REAL_DP4A_RACE = old_cards.dp4a_race        # (the fakes below replace it)


def _dp4a_card(monkeypatch: pytest.MonkeyPatch, capability: tuple, race: Any = None, wrong: tuple = ()) -> dict:
    """A card on fakes: its capability, the record's old_cards.dp4a, the shipped op and the twins (fakes answering
    zeros of the output's shape; the `wrong` ones ones). Returns the calls made, by kernel."""
    calls: dict = {"shipped": 0, "q4": 0, "f5": 0}
    profile = type("P", (), {"name": "pascal" if capability < (7, 0) else "turing", "capability": capability})()
    monkeypatch.setattr(precision, "profile", lambda: profile)
    monkeypatch.setattr(precision, "torso", lambda: torch.float32)
    monkeypatch.setattr(int8mm, "_dp4a", {})
    monkeypatch.setattr(old_cards, "dp4a_race", lambda device=None: race)

    def op(name: str) -> Any:
        def run(xq, xs, wq, ws, bias, gelu, out_dtype, raster=0, kernel=0):
            calls[name] += 1
            y = torch.zeros((xq.shape[0], wq.shape[0]))
            return y + 1 if name in wrong else y
        return run

    monkeypatch.setattr(int8mm, "cuda_gemm_op", lambda: op("shipped"))
    monkeypatch.setattr(int8mm, "_dp4a_twin_op", lambda name: op(name) if name in calls else None)
    monkeypatch.setattr(int8mm, "_twin_table", lambda device, grouped, M, K, N, gelu, seed: (
        torch.zeros((4, 8)), torch.ones((4, 1) if grouped else (4,)), torch.zeros((16, 8)), torch.ones(16), None, gelu))
    return calls


def test_the_dp4a_twin_serves_as_the_installs_race_chose_after_its_bits(monkeypatch: pytest.MonkeyPatch,
                                                                         capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("PIXAL3D_DP4A_TWIN", raising=False)
    cpu = torch.device("cpu")
    calls = _dp4a_card(monkeypatch, (6, 1), race=None)
    assert int8mm._dp4a_twin(cpu, True) is None and calls == {"shipped": 0, "q4": 0, "f5": 0}   # no record
    calls = _dp4a_card(monkeypatch, (6, 1), race={"ok": True, "group": "f5", "rows": "shipped", "summary": "-5 %"})
    assert int8mm._dp4a_twin(cpu, True) is not None and int8mm._dp4a_twin(cpu, False) is None
    assert calls == {"shipped": 1, "q4": 0, "f5": 1}                        # the bit check, once
    xq, xs = torch.zeros((4, 8), dtype=torch.int8), torch.ones((4, 1))
    int8mm.cuda_linear(xq, xs, torch.zeros((16, 8), dtype=torch.int8), torch.ones(16), None, out_dtype=torch.float32)
    assert calls == {"shipped": 1, "q4": 0, "f5": 2}                        # the group layer on the twin
    _dp4a_card(monkeypatch, (6, 1), race={"ok": True, "group": "f5"}, wrong=("f5",))
    assert int8mm._dp4a_twin(cpu, True) is None and "did not give the shipped kernel's bits" in capsys.readouterr().out
    calls = _dp4a_card(monkeypatch, (7, 5), race={"ok": True, "group": "f5"})
    assert int8mm._dp4a_twin(cpu, True) is None and calls["f5"] == 0       # the tensor cores: never the twin
    for switch, group, rows in (("q4", "q4", "q4"), ("group=f5", "f5", "shipped"), ("shipped", "shipped", "shipped")):
        monkeypatch.setenv("PIXAL3D_DP4A_TWIN", switch)
        _dp4a_card(monkeypatch, (6, 1), race={"ok": True, "group": "q4", "rows": "q4"})
        assert int8mm._dp4a_wanted(cpu, "group")[0] == group and int8mm._dp4a_wanted(cpu, "rows")[0] == rows


def test_the_dp4a_race_is_skipped_on_the_tensor_cores_and_read_for_its_own_build(monkeypatch: pytest.MonkeyPatch) -> None:
    _dp4a_card(monkeypatch, (7, 5))
    assert int8mm.calibrate_dp4a_twins(torch.device("cpu"))["skipped"] == "the card runs the tensor-core kernel"
    assert int8mm.DP4A_TWIN_MARGIN == 0.05     # a twin serves 5 % faster at every size, as the project's other races
    monkeypatch.setattr(precision, "profile", lambda: type("P", (), {"name": "pascal", "capability": (6, 1)})())
    monkeypatch.setattr(old_cards, "_kernels_build", lambda: "0.2.1")
    race = {"ok": True, "group": "f5", "rows": "q4"}
    for section, read in (({"ok": True, "profile": "pascal", "kernels": "0.2.1", "on": {}, "dp4a": race}, True),
                          ({"ok": True, "profile": "pascal", "kernels": "0.2.0", "on": {}, "dp4a": race}, False),
                          ({"ok": True, "profile": "turing", "kernels": "0.2.1", "on": {}, "dp4a": race}, False),
                          ({"ok": True, "profile": "pascal", "kernels": "0.2.1", "dp4a": {"ok": True, "skipped": "x"}},
                           False)):
        monkeypatch.setattr(calibration, "section", lambda name, device=None, section=section: section)
        assert (_REAL_DP4A_RACE() is not None) is read, section


# ---- Spellcast3D round O: the old cards' MLP in row slabs (int8mm.cuda_mlp), the same bits in a bounded table

def test_the_old_cards_mlp_runs_in_row_slabs_with_the_whole_mlps_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """mlp_slab_rows: the whole table on the ampere profile, where the hidden layer fits MLP_SLAB_MIB, or with the
    budget 0; else the budget's rows, a multiple of MLP_SLAB_ALIGN. cuda_mlp on slabs gives the one-call MLP's values
    (here a row-wise stand-in for the kernels, whose rows depend on their own row alone: the card checks prove it of
    the real ones), and int8_mlp's "cuda" path takes the slabs."""
    profile = {"name": "pascal"}
    monkeypatch.setattr(precision, "profile", lambda: type("P", (), dict(profile))())
    monkeypatch.delenv(int8mm.MLP_SLAB_ENV, raising=False)
    f32, f16 = torch.float32, torch.float16
    assert int8mm.MLP_SLAB_MIB == 256 and int8mm.MLP_SLAB_ALIGN == 256
    assert int8mm.mlp_slab_rows(43_529, 8192, f32) == 8192 and int8mm.mlp_slab_rows(43_529, 8192, f16) == 16384
    assert int8mm.mlp_slab_rows(8192, 8192, f32) == 8192 and int8mm.mlp_slab_rows(5078, 8192, f32) == 5078
    monkeypatch.setenv(int8mm.MLP_SLAB_ENV, "3")         # 3 MiB of fp32 hidden rows: 96 rows, aligned up to 256
    assert int8mm.mlp_slab_rows(5078, 8192, f32) == 256
    monkeypatch.setenv(int8mm.MLP_SLAB_ENV, "0")
    assert int8mm.mlp_slab_rows(43_529, 8192, f32) == 43_529
    monkeypatch.delenv(int8mm.MLP_SLAB_ENV)
    profile["name"] = "ampere"
    assert int8mm.mlp_slab_rows(43_529, 8192, f32) == 43_529       # the Triton path's MLP is never slabbed
    profile["name"] = "turing"

    calls: list = []

    def linear(xq, xs, w_q, w_s, bias, *, gelu=False, out_dtype=f32):
        """Exact integer sums (fp64 holds them), each group's scale in group order, element-wise ops after: a row's
        values depend on its own row alone, as in the kernels (an exact clamp stands in for GELU)."""
        calls.append(xq.shape[0])
        groups = xs.shape[1] if xs.dim() == 2 else 1
        width = xq.shape[1] // groups
        acc = torch.zeros((xq.shape[0], w_q.shape[0]), dtype=torch.float64)
        for i in range(groups):
            part = xq[:, i * width:(i + 1) * width].double() @ w_q[:, i * width:(i + 1) * width].double().t()
            acc = acc + part * (xs[:, i:i + 1] if xs.dim() == 2 else xs[:, None]).double()
        y = acc * w_s.double()[None, :] + bias.double()[None, :]
        return (y.clamp_min(0) if gelu else y).to(out_dtype)

    monkeypatch.setattr(int8mm, "cuda_linear", linear)
    monkeypatch.setattr(int8mm, "_state", {**int8mm._state, "gemm": "cuda", "elementwise": "torch"})
    g = torch.Generator().manual_seed(0)
    w1 = int8mm.quantize_rows_torch(torch.randn((256, 128), generator=g)) + (torch.randn(256, generator=g),)
    w2 = int8mm.quantize_rows_torch(torch.randn((64, 256), generator=g)) + (torch.randn(64, generator=g),)
    x = torch.randn((1000, 128), generator=g) * 3
    for group in (0, int8mm.GROUP):
        xq, xs = int8mm.quantize_rows_torch(x, group)
        whole = int8mm.cuda_mlp(xq, xs, w1, w2, group, None, f32, 1000)
        for step in (1, 7, 256, 999):
            assert torch.equal(int8mm.cuda_mlp(xq, xs, w1, w2, group, None, f32, step), whole), (group, step)
    calls.clear()
    monkeypatch.setenv(int8mm.MLP_SLAB_ENV, str(256 * 256 * 4 / 2 ** 20))       # 256 hidden rows a slab
    got = int8mm.int8_mlp(int8mm.quantize_rows_torch(x, int8mm.GROUP), w1, w2, path="group", out_dtype=f32)
    assert torch.equal(got, int8mm.cuda_mlp(*int8mm.quantize_rows_torch(x, int8mm.GROUP), w1, w2, int8mm.GROUP,
                                            None, f32, 1000))
    assert calls[:8] == [256, 256, 256, 256, 256, 256, 232, 232]                 # fc1 / fc2 a slab, the last short


def test_the_install_races_the_dp4a_twins_by_their_bits_and_worst_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """int8mm.calibrate_dp4a_twins on a (6, 1) card on fakes (ops answering by their layer's K and M, CUDA events
    on a clock of fixed ms per kernel, form and size): a twin whose bits differ at one layer of a form (q4 at fc2,
    K = 8,192) is left out of that form although the fastest, and still races the per-row form on its own (where it
    wins); a twin faster by 6 % at one size but 4 % at the other keeps the shipped kernel (the 5 % margin at the
    worst size); a twin that raises gives ok False and the shipped kernel for both forms."""
    _dp4a_card(monkeypatch, (6, 1))
    speed = {("shipped", True): 1.0, ("q4", True): 0.80, ("f5", True): (0.94, 0.96),
             ("shipped", False): 1.0, ("q4", False): 0.90, ("f5", False): 0.98}
    state: dict = {"last": None, "raise": ()}

    def op(name: str) -> Any:
        def run(xq, xs, wq, ws, bias, gelu, out_dtype, raster=0, kernel=0):
            if name in state["raise"]:
                raise RuntimeError("CUDA error: an illegal memory access was encountered")
            K, M, grouped = int(xq.item()), int(xs.reshape(-1)[0].item()), xs.dim() == 2
            state["last"] = (name, grouped, M)
            return torch.ones((1, 1)) if (name == "q4" and grouped and K == 8192) else torch.zeros((1, 1))
        return run

    class Event:
        def __init__(self, enable_timing: bool = False) -> None:
            self.at = None

        def record(self, stream: Any = None) -> None:
            self.at = state["last"]

        def elapsed_time(self, end: Any) -> float:
            name, grouped, M = end.at
            factor = speed[(name, grouped)]
            return 10.0 * (factor[0 if M == 8192 else 1] if isinstance(factor, tuple) else factor)

    monkeypatch.setattr(int8mm, "cuda_gemm_op", lambda: op("shipped"))
    monkeypatch.setattr(int8mm, "_dp4a_twin_op", lambda name: op(name) if name in int8mm.DP4A_TWINS else None)
    monkeypatch.setattr(int8mm, "_twin_table", lambda device, grouped, M, K, N, gelu, seed: (
        torch.full((1, 1), float(K)), torch.full((1, 1) if grouped else (1,), float(M)), torch.zeros((1, 1)),
        torch.ones(1), None, gelu))
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    cpu = torch.device("cpu")
    found = int8mm.calibrate_dp4a_twins(cpu)
    assert found["ok"] is True and found["present"] == ["q4", "f5"]
    assert found["group"] == "shipped" and "other bits: q4" in found["summary"]      # q4 out, f5 -4 % at the worst
    assert found["rows"] == "q4"                                                        # raced on its own: -10 %
    timed = {(t["form"], t["layer"], t["M"]) for t in found["timings"]}
    assert timed == {(form, layer, M) for form, layers in int8mm.DP4A_RACE.items() for layer, *_ in layers
                     for M, _ in int8mm.DP4A_RACE_ROWS}
    assert all("q4" not in t["ms"] for t in found["timings"] if t["form"] == "group" and t["layer"] == "fc2")
    speed[("f5", True)] = (0.94, 0.94)                                                  # -6 % at both: f5 serves
    assert int8mm.calibrate_dp4a_twins(cpu)["group"] == "f5"
    state["raise"] = ("f5",)
    found = int8mm.calibrate_dp4a_twins(cpu)
    assert found["ok"] is False and found["group"] == found["rows"] == "shipped" and "RuntimeError" in found["error"]


def test_the_dp4a_race_runs_in_a_child_and_a_child_that_fails_keeps_the_shipped_kernel(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """old_cards' step races the twins in a child process (int8mm.calibrate_dp4a_twins_in_child): its last JSON
    line is the result; a child that crashes (a faulting twin), prints nothing or runs out of time records ok False
    and the shipped kernel for both forms; no child where nothing is to be raced."""
    import json
    import subprocess

    _dp4a_card(monkeypatch, (7, 5))
    started: list = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: started.append(argv))
    assert int8mm.calibrate_dp4a_twins_in_child(torch.device("cpu"))["skipped"] and started == []
    calls = _dp4a_card(monkeypatch, (6, 1))
    raced = {"ok": True, "present": ["q4", "f5"], "group": "f5", "rows": "shipped", "timings": [{"M": 8192}]}
    for code, out, want in ((0, "[PRECISION] pascal\n" + json.dumps(raced) + "\n", raced),
                            (3221225477, json.dumps(raced) + "\n", None), (-11, "", None)):
        monkeypatch.setattr(subprocess, "run", lambda argv, code=code, out=out, **kw: subprocess.CompletedProcess(
            argv, code, stdout=out, stderr="Windows fatal exception: access violation"))
        found = int8mm.calibrate_dp4a_twins_in_child(torch.device("cpu"))
        if want is not None:
            assert found == want
        else:
            assert found["ok"] is False and found["group"] == found["rows"] == "shipped", found
            assert f"code {code}" in found["error"] and "access violation" in found["error"]
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: (_ for _ in ()).throw(
        subprocess.TimeoutExpired(argv, kw.get("timeout"))))
    found = int8mm.calibrate_dp4a_twins_in_child(torch.device("cpu"))
    assert found["ok"] is False and found["rows"] == "shipped" and "did not finish" in found["error"]
    assert calls == {"shipped": 0, "q4": 0, "f5": 0}                                # nothing ran in this process
