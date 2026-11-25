"""SageAttention's int8 P.V kernel per card (code/pixal3d/pixal3d/modules/sage.py) on the CPU, with fake kernels: which
P.V kernel serves after the self-check (`_setup_int8pv`) for each PIXAL3D_INT8PV and card, the blocks it may serve
(`mark_int8pv`), and the install's trial of the Turing twin (`_old_card_trial`, its child process's partial result,
the record's race that the installer looks for). No kernel of the wheel runs: the fakes answer with torch's SDPA.

Runs in the pixal3d venv, with the card hidden and pytest from code/logs/profiling/pytools:
    set CUDA_VISIBLE_DEVICES=-1
    set PYTHONPATH=code\\logs\\profiling\\pytools
    code\\venvs\\pixal3d\\Scripts\\python.exe -m pytest -p no:cacheprovider code\\tests\\test_pixal3d_sage.py -q
Skipped in the hub venv (no torch).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from spellcast import paths

torch = pytest.importorskip("torch")
if not (paths.PIXAL3D_DIR / "pixal3d" / "modules" / "sage.py").is_file():
    pytest.skip("the Pixal3D checkout is not here", allow_module_level=True)

# imported while this file is collected and taken off the path again at once (as test_pixal3d_weights.py does);
# calibrate.py sets two cache folders in the environment when imported: put back as they were
_ENV = {name: os.environ.get(name) for name in ("FLEX_GEMM_AUTOTUNE_CACHE_PATH", "TRITON_CACHE_DIR")}
sys.path.insert(0, str(paths.PIXAL3D_DIR))
try:
    from pixal3d import calibrate                           # noqa: E402
    from pixal3d.modules import precision, sage             # noqa: E402
finally:
    sys.path.remove(str(paths.PIXAL3D_DIR))
    for _name, _value in _ENV.items():
        if _value is None:
            os.environ.pop(_name, None)
        else:
            os.environ[_name] = _value

F = torch.nn.functional
CPU = torch.device("cpu")


def _exact(q: Any, k: Any, v: Any) -> Any:
    """A kernel that answers as fp32 SDPA does ([B, N, H, D] NHD)."""
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


class _Card:
    """The fakes of one card: its capability, the fp16 kernel in use, the int8 P.V candidate and what was called."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, major: int, record: dict | None = None,
                 race: bool | None = None, answer: Any = _exact) -> None:
        self.calls: list[str] = []
        self.raced = 0
        profile = SimpleNamespace(name={6: "pascal", 7: "turing"}.get(major, "ampere"), capability=(major, 5),
                                  torso=torch.float32)
        monkeypatch.setattr(precision, "profile", lambda: profile)
        monkeypatch.setattr(sage.torch.cuda, "synchronize", lambda *args: None)
        monkeypatch.setattr(sage, "_stored", lambda device: record)
        monkeypatch.setattr(sage, "_kernel", _exact)

        def twin(q, k, v):
            self.calls.append("int8pv")
            return answer(q, k, v)

        def candidate(m: int):
            assert m == major
            return twin, f"the fake int8 P.V kernel of sm_{m}x"

        def race_int8pv(device, fp16, int8, plan=sage.RACE_TOKENS):
            assert race is not None, "the race ran where the switch or the record decides"
            self.raced += 1
            return race, "[SAGE] the fake race", [(8192, 10.0, 9.0 if race else 10.0)]

        monkeypatch.setattr(sage, "_int8pv_candidate", candidate)
        monkeypatch.setattr(sage, "_race_int8pv", race_int8pv)

    def setup(self) -> bool:
        """`_setup_int8pv` on the self-check's table: whether the int8 P.V kernel serves the marked blocks."""
        q, k, vs, want = sage._check_table(CPU)
        sage._setup_int8pv(CPU, q, k, vs, want)
        return sage._kernel_int8pv is not None


@pytest.fixture(autouse=True)
def _restore(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv("PIXAL3D_INT8PV", raising=False)
    yield
    sage._kernel_int8pv, sage.KERNEL_INT8PV = None, ""


@pytest.mark.parametrize("switch", ["", "off", "0", "no"])
def test_the_turing_twin_does_not_run_unless_asked_for(switch: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """INT8PV_TURING_DEFAULT is False: an unset switch on sm_7x keeps the fp16 P.V kernel for every block - the twin
    is not even tried - whatever the install's race found."""
    assert sage.INT8PV_TURING_DEFAULT is False
    monkeypatch.setenv("PIXAL3D_INT8PV", switch)
    card = _Card(monkeypatch, 7, record={"tuned": False, "int8pv": True})
    assert not card.setup()
    assert card.calls == [] and card.raced == 0


@pytest.mark.parametrize("record, race, serves", [
    ({"tuned": False, "int8pv": True, "int8pv_summary": "-9 %"}, None, True),     # the install's race decides
    ({"tuned": False, "int8pv": False}, None, False),
    ({"tuned": False}, True, True),                                                 # no race in the record: timed
    (None, False, False),                                                           # here, at first use
])
def test_the_turing_twin_asked_for_serves_as_the_race_says(record: dict | None, race: bool | None, serves: bool,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIXAL3D_INT8PV", "on")
    card = _Card(monkeypatch, 7, record=record, race=race)
    assert card.setup() is serves
    assert card.calls == ["int8pv"]                             # the self-check's table, once
    assert card.raced == (0 if record and "int8pv" in record else 1)


def test_the_turing_twin_forced_serves_without_a_race(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIXAL3D_INT8PV", "force")
    card = _Card(monkeypatch, 7, record={"tuned": False, "int8pv": False})
    assert card.setup() and card.raced == 0
    assert sage.KERNEL_INT8PV == "the fake int8 P.V kernel of sm_7x"


def test_the_turing_default_once_on_reads_the_race(monkeypatch: pytest.MonkeyPatch) -> None:
    """The user's decision is the constant alone: with INT8PV_TURING_DEFAULT True an unset switch means "on"."""
    monkeypatch.setattr(sage, "INT8PV_TURING_DEFAULT", True)
    card = _Card(monkeypatch, 7, record={"tuned": False, "int8pv": True})
    assert card.setup() and card.raced == 0


@pytest.mark.parametrize("switch", ["on", "force"])
def test_a_twin_that_answers_wrongly_is_left_out(switch: str, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("PIXAL3D_INT8PV", switch)
    card = _Card(monkeypatch, 7, record={"tuned": False, "int8pv": True}, answer=lambda q, k, v: _exact(q, k, v) * 1.1)
    assert not card.setup()
    assert "the int8 P.V kernel is left out" in capsys.readouterr().out
    nan = _Card(monkeypatch, 7, record={"tuned": False, "int8pv": True},
                answer=lambda q, k, v: _exact(q, k, v).index_fill_(1, torch.tensor([3]), float("nan")))
    assert not nan.setup()


def test_a_twin_that_raises_is_left_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIXAL3D_INT8PV", "on")

    def fails(q, k, v):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    card = _Card(monkeypatch, 7, record={"tuned": False, "int8pv": True}, answer=fails)
    assert not card.setup() and sage.KERNEL_INT8PV == ""


@pytest.mark.parametrize("major", [8, 6])
def test_other_cards_keep_their_unset_meaning_on(major: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset is "on" on sm_8x and sm_6x as before (the race decides); "off" keeps the fp16 kernel there too."""
    card = _Card(monkeypatch, major, record={"tuned": True, "int8pv": True})
    assert card.setup()
    monkeypatch.setenv("PIXAL3D_INT8PV", "off")
    off = _Card(monkeypatch, major, record={"tuned": True, "int8pv": True})
    assert not off.setup() and off.calls == []


def test_the_switch_by_card() -> None:
    cases = {"": ("on", "off", "on"), "on": ("on", "on", "on"), "1": ("on", "on", "on"), "off": ("off",) * 3,
             "false": ("off",) * 3, "force": ("force",) * 3, "Force ": ("force",) * 3}
    for text, want in cases.items():
        os.environ["PIXAL3D_INT8PV"] = text
        assert tuple(sage.int8pv_switch(major) for major in (8, 7, 6)) == want, text
        assert sage.int8pv_switch() == want[0], text                 # the calibration's reading: no card
    del os.environ["PIXAL3D_INT8PV"]


class _Attention(torch.nn.Module):
    def __init__(self, kind: str = "self", mode: str = "full") -> None:
        super().__init__()
        self._type, self.attn_mode = kind, mode


_Attention.__name__ = "SparseMultiHeadAttention"


def test_the_marked_blocks_are_the_detail_flows_but_hr_block_0() -> None:
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList()
    for _ in range(4):
        block = torch.nn.Module()
        block.self_attn, block.cross_attn = _Attention(), _Attention("cross")
        block.windowed = _Attention(mode="windowed")
        model.blocks.append(block)
    marked = {kind: sage.mark_int8pv(model, kind) for kind in ("ss", "lr", "hr", "tex")}
    assert marked == {"ss": 0, "lr": 0, "hr": 3, "tex": 4}
    sage.mark_int8pv(model, "hr")
    assert [b.self_attn.sage_int8pv for b in model.blocks] == [False, True, True, True]
    assert not any(getattr(b.cross_attn, "sage_int8pv", False) or getattr(b.windowed, "sage_int8pv", False)
                   for b in model.blocks)


def _on_card(tokens: int = 70) -> Any:
    """What sage.usable reads of q, k and v, as a self-attention's operands on the card look (no card needed)."""
    return SimpleNamespace(is_cuda=True, device=torch.device("cuda", 0), dtype=precision.torso(),
                           shape=(1, tokens, 2, 128))


class _Asking(_Attention):
    """A self-attention module whose forward asks sage.usable, as the flows' attention calls do."""

    def forward(self, qkv: Any) -> bool:
        return sage.usable(qkv, qkv, qkv)


_Asking.__name__ = "SparseMultiHeadAttention"


@pytest.mark.parametrize("flows, left_out", [("", set()), ("all", set()), ("hr,tex", {"ss", "lr"}),
                                             ("ss, LR", {"hr", "tex"})])
def test_pixal3d_sage_flows_leaves_the_flows_it_leaves_out_to_their_backend(flows: str, left_out: set,
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """A flow PIXAL3D_SAGE_FLOWS leaves out never asks the kernel: usable() says no inside its self-attention, so the
    attention backend answers (flash_attn on the ampere profile, as with the switch off), and it is not folded."""
    monkeypatch.setenv("PIXAL3D_SAGE_FLOWS", flows)
    monkeypatch.setattr(sage, "_state", True)
    for kind in ("ss", "lr", "hr", "tex"):
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList()
        for _ in range(2):
            block = torch.nn.Module()
            block.self_attn = _Asking()
            model.blocks.append(block)
        marked = sage.mark_int8pv(model, kind)
        sage.mark_int8pv(model, kind)                  # marking twice wraps once
        with torch.no_grad():
            asked = [model.blocks[i].self_attn(_on_card()) for i in range(2)]
        assert sage.runs(kind) == (kind not in left_out)
        if kind in left_out:
            assert asked == [False, False] and marked == 0
            assert all(b.self_attn.exact_scope and not b.self_attn.sage_int8pv for b in model.blocks)
        else:
            assert asked == [True, True]
            assert marked == {"ss": 0, "lr": 0, "hr": 1, "tex": 2}[kind]
    with torch.no_grad():
        assert sage.usable(*[_on_card()] * 3)          # outside a wrapped module


class _Trial:
    """The calibration's trial of the Turing kernel on fakes: small race sizes, CUDA events that count calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, twin: Any = _exact, wins: bool = True) -> None:
        self.raced: list = []
        profile = SimpleNamespace(name="turing", capability=(7, 5), torso=torch.float32)
        monkeypatch.setattr(precision, "profile", lambda: profile)
        monkeypatch.setattr(sage.torch.cuda, "synchronize", lambda *args: None)
        monkeypatch.setattr(sage, "RACE_TOKENS", ((96, 2),))
        monkeypatch.setattr(sage.torch.cuda, "Event", _Event)
        monkeypatch.setattr(sage, "_old_card_candidate", lambda major: ("turing", _exact, "the fake Turing op"))
        monkeypatch.setattr(sage, "_turing_int8pv_op", lambda: (None if twin is None else "op"))
        monkeypatch.setattr(sage, "_turing_int8pv_call", lambda op: twin)

        def race(device, fp16, int8, plan=sage.RACE_TOKENS):
            self.raced.append(plan)
            return wins, "[SAGE] the fake race", [(21504, 78.8, 70.0 if wins else 78.0), (8192, 10.2, 9.2)]

        monkeypatch.setattr(sage, "_race_int8pv", race)


class _Event:
    def __init__(self, enable_timing: bool = False) -> None:
        pass

    def record(self, stream: Any = None) -> None:
        pass

    def elapsed_time(self, other: Any) -> float:
        return 1.0


@pytest.mark.parametrize("switch", ["", "off", "on"])
def test_the_install_times_the_turing_twin_whatever_the_switch(switch: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The record says how fast the twin is on this card; the switch decides in the engine whether it serves."""
    monkeypatch.setenv("PIXAL3D_INT8PV", switch)
    trial = _Trial(monkeypatch)
    told: list = []
    found = sage._old_card_trial(CPU, progress=told.append)
    assert found["candidate"] == "turing" and found["candidate_ok"] is True
    assert found["int8pv"] is True and found["int8pv_timings"][0] == {"tokens": 21504, "fp16_ms": 78.8,
                                                                       "int8_ms": 70.0}
    assert trial.raced == [sage.CALIBRATION_RACE]
    assert len(told) == 1 and told[0]["candidate_ok"] is True and "int8pv" not in told[0]   # before the twin ran
    assert not sage.int8pv_race_lacking(found) and sage.int8pv_race_lacking(told[0])


@pytest.mark.parametrize("twin, why", [
    (None, "the installed wheel has no int8 P.V kernel for sm_75"),
    (lambda q, k, v: _exact(q, k, v) * 1.2, "its answer differs from fp32 attention"),
    (lambda q, k, v: (_ for _ in ()).throw(RuntimeError("CUDA error: misaligned address")), "RuntimeError: CUDA error"),
])
def test_a_turing_twin_that_fails_its_trial_leaves_the_fp16_kernel_standing(twin: Any, why: str,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    trial = _Trial(monkeypatch, twin=twin)
    found = sage._old_card_trial(CPU)
    assert found["candidate_ok"] is True and found["int8pv"] is False and why in found["int8pv_why"]
    assert trial.raced == [] and not sage.int8pv_race_lacking(found)


def test_a_twin_that_takes_the_trial_process_down_leaves_the_kernel_standing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The child prints the candidate's fields before its twin runs: a crash after that keeps them."""
    partial = {"candidate": "turing", "candidate_ok": True, "why": "the fake Turing op ran at the flows' sizes",
               "timings": [], "summary": "21,504 tokens 78.78 ms"}

    def crashed(argv, **kw):
        return subprocess.CompletedProcess(argv, -11, stdout=json.dumps(partial) + "\n", stderr="Segmentation fault")

    monkeypatch.setattr(subprocess, "run", crashed)
    found = sage._old_card_trial_in_child(7, 5)
    assert found["candidate_ok"] is True and found["int8pv"] is False and "code -11" in found["int8pv_why"]

    def crashed_early(argv, **kw):
        return subprocess.CompletedProcess(argv, -11, stdout="", stderr="Segmentation fault")

    monkeypatch.setattr(subprocess, "run", crashed_early)
    assert sage._old_card_trial_in_child(7, 5)["candidate_ok"] is False


def test_a_turing_record_without_the_twins_race_is_timed_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """calibrate.py keeps a standing record's attention section only with the race (the installer's --only sage)."""
    monkeypatch.setattr(sage, "_wheel_version", lambda: "2.2.0")
    ran = {"wheel": "2.2.0", "tuned": False, "candidate": "turing", "candidate_ok": True}
    assert not calibrate._sage_stands({"sage": ran})
    assert calibrate._sage_stands({"sage": {**ran, "int8pv": False}})
    assert calibrate._sage_stands({"sage": {**ran, "candidate_ok": False}})
    assert calibrate._sage_stands({"sage": {"wheel": "2.2.0", "tuned": True, "int8pv": True}})     # sm_8x
    assert not calibrate._sage_stands({"sage": {**ran, "wheel": "2.1.0", "int8pv": True}})


def test_a_trial_asked_for_by_name_runs_on_a_complete_record(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A trial cut short leaves its section "ok" and the record complete, so a whole run keeps it; the installer's
    retry of it (--only sage) runs the trial all the same, and the record then holds the kernel that ran."""
    from pixal3d import calibration

    card = "NVIDIA GeForce GTX 1080"
    monkeypatch.setattr(sage, "_wheel_version", lambda: "2.2.0")
    monkeypatch.setattr(precision, "profile", lambda: SimpleNamespace(name="pascal", capability=(6, 1)))
    monkeypatch.setattr(calibrate, "_query", lambda fields: ["0"])
    monkeypatch.setattr(calibrate, "_shapes_lacking", lambda earlier, record: {})       # a complete record of this card
    for name, answer in (("is_available", True), ("current_device", 0), ("get_device_name", card),
                         ("get_device_capability", (6, 1))):
        monkeypatch.setattr(torch.cuda, name, lambda *args, answer=answer: answer)
    trials: list = []
    ran = {"candidate": "pascal", "candidate_ok": True, "why": "the fake Pascal op ran at the flows' sizes"}
    monkeypatch.setattr(sage, "_old_card_trial_in_child", lambda major, minor: trials.append((major, minor)) or dict(ran))
    record = tmp_path / calibration.FILE_NAME
    monkeypatch.setenv(calibration.ENV, str(record))
    monkeypatch.setattr(calibration, "_cache", {})
    cut_short = {"ok": True, "wheel": "2.2.0", "tuned": False, "candidate": "pascal", "candidate_ok": False,
                 "why": "the trial did not finish in 600 s"}
    record.write_text(json.dumps({"version": 3, "complete": True, "card": card, "sage": cut_short,
                                  "flex_gemm": {"ok": True, "skipped": "the sparse convolutions run on spconv"}}),
                      encoding="utf-8")
    assert calibrate._sage_stands({"sage": cut_short})
    assert calibrate.main(["--version", "3", "--only", "sage", "--record", str(record)]) == 0
    assert trials == [(6, 1)]
    assert json.loads(record.read_text(encoding="utf-8"))["sage"]["candidate_ok"] is True


# ---- Spellcast3D round O: the Pascal op's twins (sage.PASCAL_TWINS), raced at install, the shipped op by default

class _Pascal:
    """A GTX 10xx on fakes: the shipped Pascal op and its twins (fake calls answering as fp32 SDPA, or wrongly), the
    record, the CUDA events (each call's time from `ms`)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, record: dict | None = None, wrong: tuple = (),
                 ms: dict | None = None, precise: tuple = ()) -> None:
        self.calls: list[str] = []
        profile = SimpleNamespace(name="pascal", capability=(6, 1), torso=torch.float32)
        monkeypatch.setattr(precision, "profile", lambda: profile)
        monkeypatch.setattr(sage.torch.cuda, "synchronize", lambda *args: None)
        monkeypatch.setattr(sage.torch.cuda, "mem_get_info", lambda device=None: (1 << 34, 1 << 34))
        monkeypatch.setattr(sage.torch.cuda, "memory_reserved", lambda device=None: 0)
        monkeypatch.setattr(sage.torch.cuda, "memory_allocated", lambda device=None: 0)
        monkeypatch.setattr(sage, "_stored", lambda device: record)
        monkeypatch.setattr(sage, "_pascal_twins", {})
        monkeypatch.setattr(sage, "_precise_state", {})
        monkeypatch.setattr(sage, "CALIBRATION_RACE", ((96, 3), (64, 3)))
        self.ms = ms or {}
        self.ops = {n: SimpleNamespace(name=n) for n in ("shipped", *sage.PASCAL_TWINS, *precise)}
        monkeypatch.setattr(sage, "_pascal_op", lambda: self.ops["shipped"])
        monkeypatch.setattr(sage, "_pascal_twin_op", lambda twin: self.ops.get(twin))
        clock = self

        class Event:
            def __init__(self, enable_timing: bool = False) -> None:
                self.at = None

            def record(self, stream: Any = None) -> None:
                self.at = clock.calls[-1] if clock.calls else None

            def elapsed_time(self, other: Any) -> float:
                return clock.ms.get(other.at, 10.0)

        monkeypatch.setattr(sage.torch.cuda, "Event", Event)

        def call(op, int8pv):
            def run(q, k, v):
                self.calls.append(op.name)
                out = _exact(q, k, v)
                return out * 1.001 if op.name in wrong else out
            return run

        monkeypatch.setattr(sage, "_pascal_call", call)


def test_the_pascal_op_serves_without_a_record_and_a_recorded_twin_after_its_bits(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("PIXAL3D_PASCAL_TWIN", raising=False)
    card = _Pascal(monkeypatch, record=None)
    assert sage._pascal_twin(CPU, True) == (card.ops["shipped"], "shipped") and card.calls == []
    record = {"tuned": False, "pascal_twins": {"fp32": "u28", "int8": "s28", "int8_summary": "-6 %"}}
    card = _Pascal(monkeypatch, record=record)
    assert sage._pascal_twin(CPU, True) == (card.ops["s28"], "s28")
    assert sage._pascal_twin(CPU, False) == (card.ops["u28"], "u28")
    # the self-check's table and the tail table (990 tokens: k64's last step without its second block), once each
    assert card.calls == ["shipped", "s28", "shipped", "s28", "shipped", "u28", "shipped", "u28"]
    assert sage._pascal_twin(CPU, True)[1] == "s28" and len(card.calls) == 8       # cached
    card = _Pascal(monkeypatch, record=record, wrong=("s28",))
    assert sage._pascal_twin(CPU, True) == (card.ops["shipped"], "shipped")
    assert "did not give the shipped kernel's bits" in capsys.readouterr().out
    card = _Pascal(monkeypatch, record={"tuned": False, "pascal_twins": {"int8": "x99"}})
    assert sage._pascal_twin(CPU, True)[1] == "shipped" and card.calls == []        # not a known twin


def test_the_twin_check_covers_k64s_tail_without_its_second_block() -> None:
    """The 1000-token self-check table ends on a 64-key step with both 32-key blocks (1000 mod 64 = 40) and the races'
    sizes are multiples of 64: the tail table makes the bit check reach k64's "block B absent" path."""
    tables = sage._twin_tables(CPU)
    assert [t[0].shape[1] % 64 for t in tables] == [40, sage.TAIL_TOKENS % 64]
    assert 1 <= sage.TAIL_TOKENS % 64 <= 32
    assert all(n % 64 == 0 for n, _calls in sage.CALIBRATION_RACE)


def test_a_known_twin_missing_from_the_wheel_leaves_the_shipped_op(monkeypatch: pytest.MonkeyPatch,
                                                                  capsys: pytest.CaptureFixture[str]) -> None:
    """A record (or PIXAL3D_PASCAL_TWIN) naming k64 while the installed wheel lacks it: the shipped op serves, one
    line says so; the install's race leaves k64 out of its present list and still races the others."""
    monkeypatch.delenv("PIXAL3D_INT8PV", raising=False)
    monkeypatch.delenv("PIXAL3D_PASCAL_TWIN", raising=False)
    card = _Pascal(monkeypatch, record={"tuned": False, "pascal_twins": {"int8": "k64"}})
    del card.ops["k64"]                                              # an older wheel (post8): no k64
    assert sage._pascal_twin(CPU, True) == (card.ops["shipped"], "shipped") and card.calls == []
    assert "k64 is not in the installed wheel" in capsys.readouterr().out
    monkeypatch.setenv("PIXAL3D_PASCAL_TWIN", "int8=k64")
    card = _Pascal(monkeypatch)
    del card.ops["k64"]
    assert sage._pascal_twin(CPU, True)[1] == "shipped"
    assert "k64 is not in the installed wheel" in capsys.readouterr().out
    monkeypatch.delenv("PIXAL3D_PASCAL_TWIN")
    card = _Pascal(monkeypatch, ms={"shipped": 10.0, "s28": 9.0, "u28": 9.9, "u44": 9.9, "s24": 9.9})
    del card.ops["k64"]
    q, k, vs, _want = sage._check_table(CPU)
    found = sage._pascal_twins_trial(CPU, q, k, vs)
    assert found["present"] == [t for t in sage.PASCAL_TWINS if t != "k64"]
    assert found["int8"] == "s28" and "k64" not in {n for r in found["int8_timings"] for n in r["ms"]}


@pytest.mark.parametrize("switch, fp32, int8", [("u44", "u44", "u44"), ("int8=s24", "shipped", "s24"),
                                                 ("fp32=u28, int8=s28", "u28", "s28"), ("shipped", "shipped", "shipped"),
                                                 ("off", "shipped", "shipped")])
def test_pixal3d_pascal_twin_forces_a_twin(switch: str, fp32: str, int8: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIXAL3D_PASCAL_TWIN", switch)
    _Pascal(monkeypatch, record={"tuned": False, "pascal_twins": {"fp32": "u44", "int8": "u44"}})
    assert sage._pascal_twin(CPU, False)[1] == fp32 and sage._pascal_twin(CPU, True)[1] == int8


def test_the_install_races_the_pascal_twins_per_pv_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fastest twin by its worst size serves where it is PASCAL_TWIN_MARGIN faster; a twin with other bits never
    serves; the slice twins are not raced for the fp32 P.V; nothing faster keeps the shipped op."""
    monkeypatch.delenv("PIXAL3D_INT8PV", raising=False)
    card = _Pascal(monkeypatch, wrong=("s24",), ms={"shipped": 10.0, "u28": 9.8, "u44": 9.4, "s28": 9.0, "s24": 1.0})
    q, k, vs, _want = sage._check_table(CPU)
    found = sage._pascal_twins_trial(CPU, q, k, vs)
    assert found["present"] == list(sage.PASCAL_TWINS)
    assert found["int8"] == "s28" and "s24" in found["int8_why"]       # s24's bits differ: left out, though "fast"
    assert found["fp32"] == "u44"                                      # s28 is not raced for the fp32 P.V
    assert {r["tokens"] for r in found["int8_timings"]} == {96, 64}
    assert sage.PASCAL_TWIN_MARGIN == 0.05                             # as the project's other races
    card.ms = {"shipped": 10.0, "u28": 9.9, "u44": 9.6, "s28": 9.65}   # -4 % at best: under the 5 % margin
    assert sage._pascal_twins_trial(CPU, q, k, vs)["int8"] == "shipped"
    monkeypatch.setenv("PIXAL3D_INT8PV", "off")
    found = sage._pascal_twins_trial(CPU, q, k, vs)
    assert found["int8"] == "shipped" and "off" in found["int8_why"]


def test_the_round_p_k64_twin_races_for_the_int8_pv_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round P's k64 (its fp32 P.V is the shipped kernel's code, -1..-3 %: under the margin) is a race candidate for
    the int8 P.V alone, and wins there only past the margin at every size."""
    monkeypatch.delenv("PIXAL3D_INT8PV", raising=False)
    assert "k64" in sage.PASCAL_TWINS and "k64" not in sage.PASCAL_TWINS_FP32
    card = _Pascal(monkeypatch, ms={"shipped": 10.0, "k64": 8.9})
    q, k, vs, _want = sage._check_table(CPU)
    found = sage._pascal_twins_trial(CPU, q, k, vs)
    assert found["int8"] == "k64" and found["fp32"] == "shipped"
    assert "k64" not in {name for r in found["fp32_timings"] for name in r.get("ms", {})}
    card.ms = {"shipped": 10.0, "k64": 9.6}                             # -4 %: under the 5 % margin
    assert sage._pascal_twins_trial(CPU, q, k, vs)["int8"] == "shipped"


def test_a_pascal_twin_that_raises_in_the_trial_keeps_the_shipped_op(monkeypatch: pytest.MonkeyPatch) -> None:
    _Pascal(monkeypatch)

    def raises(*args: Any) -> Any:
        raise RuntimeError("CUDA error: an illegal memory access")

    monkeypatch.setattr(sage, "_race_pascal_twins", raises)
    q, k, vs, _want = sage._check_table(CPU)
    found = sage._pascal_twins_trial(CPU, q, k, vs)
    assert found["fp32"] == found["int8"] == "shipped" and "CUDA error" in found["error"]


def test_pascal_twins_that_take_the_trial_process_down_leave_the_races_standing(monkeypatch: pytest.MonkeyPatch) -> None:
    partial = {"candidate": "pascal", "candidate_ok": True, "why": "ran", "timings": [], "summary": "s",
               "int8pv": True, "int8pv_why": "faster"}

    def crashed(argv, **kw):
        return subprocess.CompletedProcess(argv, -11, stdout=json.dumps(partial) + "\n", stderr="Segmentation fault")

    monkeypatch.setattr(subprocess, "run", crashed)
    found = sage._old_card_trial_in_child(6, 1)
    assert found["candidate_ok"] is True and found["int8pv"] is True
    assert found["pascal_twins"]["fp32"] == found["pascal_twins"]["int8"] == "shipped"
    assert "code -11" in found["pascal_twins"]["error"]


def test_a_trial_that_printed_its_whole_answer_keeps_it_whatever_its_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A child whose last line is the finished trial (the Pascal op's twins included) but that ends with an error -
    a twin's sticky CUDA error in its teardown (0xC0000409 on Windows) - keeps the shipped op and its int8 P.V race,
    with the exit code noted and the twins left out; a child whose time ran out is salvaged the same way."""
    partial = {"candidate": "pascal", "candidate_ok": True, "why": "ran", "timings": [], "summary": "s",
               "int8pv": True, "int8pv_why": "faster"}
    whole = {**partial, "pascal_twins": {"present": ["u28"], "fp32": "u28", "int8": "shipped", "error": "RuntimeError"}}
    out = json.dumps(partial) + "\n" + json.dumps(whole) + "\n"

    def crashed(argv, **kw):
        return subprocess.CompletedProcess(argv, 3221226505, stdout=out, stderr="Fatal Python error")

    monkeypatch.setattr(subprocess, "run", crashed)
    found = sage._old_card_trial_in_child(6, 1)
    assert found["candidate_ok"] is True and found["int8pv"] is True and found["exit_code"] == 3221226505
    assert found["pascal_twins"]["fp32"] == found["pascal_twins"]["int8"] == "shipped"
    error = found["pascal_twins"]["error"]
    assert error.startswith("RuntimeError; ") and "code 3221226505" in error and "'fp32': 'u28'" in error

    def timed_out(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw.get("timeout"), output=out.encode(), stderr=b"")

    monkeypatch.setattr(subprocess, "run", timed_out)
    found = sage._old_card_trial_in_child(6, 1)
    assert found["candidate_ok"] is True and found["exit_code"] is None and "did not finish" in found["exit_note"]
    turing = {"candidate": "turing", "candidate_ok": True, "why": "ran", "int8pv": False, "int8pv_why": "slower"}
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(
        argv, 1, stdout=json.dumps(turing) + "\n", stderr="x"))
    found = sage._old_card_trial_in_child(7, 5)
    assert found["candidate_ok"] is True and found["int8pv"] is False and found["exit_code"] == 1
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: (_ for _ in ()).throw(
        subprocess.TimeoutExpired(argv, 600)))
    assert sage._old_card_trial_in_child(7, 5)["candidate_ok"] is False                 # nothing printed in time


# ---- round P sage-bias: the precise Pascal op (PASCAL_PRECISE) - not a twin (other bits by design), so it is
# accepted on its accuracy against the shipped op on the user's card, with a speed guard, never on speed alone

def test_the_precise_pascal_op_serves_both_pv_modes_where_the_record_says_ok(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("PIXAL3D_PASCAL_TWIN", raising=False)
    monkeypatch.delenv("PIXAL3D_PASCAL_PRECISE", raising=False)
    px = sage.PASCAL_PRECISE
    ok = {"tuned": False, "pascal_twins": {"fp32": "shipped", "int8": "k64"},
          "pascal_precise": {"present": True, "ok": True, "summary": "error 0.007 (shipped 0.010)"}}
    assert sage.PRECISE_DEFAULT is False and sage.precise_switch() == "off"     # unset: off until the user opts in
    monkeypatch.setattr(sage, "PRECISE_DEFAULT", True)                           # True: unset = the record decides
    assert sage.precise_switch() == "auto"
    monkeypatch.setattr(sage, "PRECISE_DEFAULT", False)                          # False: unset means off
    assert sage.precise_switch() == "off"
    card = _Pascal(monkeypatch, record=ok, precise=(px,), wrong=("shipped",))
    assert sage._pascal_twin(CPU, True)[1] != px and px not in card.calls
    monkeypatch.setattr(sage, "PRECISE_DEFAULT", True)
    monkeypatch.setenv("PIXAL3D_PASCAL_PRECISE", "on")                            # on: the record decides
    card = _Pascal(monkeypatch, record=None, precise=(px,), wrong=("shipped",))
    assert sage._pascal_twin(CPU, True) == (card.ops["shipped"], "shipped") and card.calls == []   # no record
    card = _Pascal(monkeypatch, record=ok, precise=(px,), wrong=("shipped",))     # px the more exact one here
    assert sage._pascal_twin(CPU, True) == (card.ops[px], px)
    assert sage._pascal_twin(CPU, False) == (card.ops[px], px)
    assert set(card.calls) == {"shipped", px} and "precise Pascal kernel" in capsys.readouterr().out
    calls = len(card.calls)
    assert sage._pascal_twin(CPU, True)[1] == px and len(card.calls) == calls     # checked once a process
    card = _Pascal(monkeypatch, record=ok, precise=(px,), wrong=(px,))            # less exact on this card
    assert sage._pascal_twin(CPU, True) == (card.ops["k64"], "k64")               # the raced twin, as before
    assert "left out on this card" in capsys.readouterr().out
    card = _Pascal(monkeypatch, record={**ok, "pascal_precise": {"present": True, "ok": False}}, precise=(px,))
    assert sage._pascal_twin(CPU, True)[1] == "k64" and px not in card.calls       # the record said no: not tried
    card = _Pascal(monkeypatch, record=ok)                                          # an older wheel: no px op
    assert sage._pascal_twin(CPU, True)[1] == "k64"
    monkeypatch.setenv("PIXAL3D_PASCAL_PRECISE", "off")
    card = _Pascal(monkeypatch, record=ok, precise=(px,), wrong=("shipped",))
    assert sage._pascal_twin(CPU, True)[1] == "shipped" and px not in card.calls    # k64's bits differ from the
    monkeypatch.setenv("PIXAL3D_PASCAL_PRECISE", "force")                            # wrong shipped fake: shipped
    card = _Pascal(monkeypatch, record=None, precise=(px,), wrong=("shipped",))
    assert sage._pascal_twin(CPU, False) == (card.ops[px], px)


def test_the_install_records_the_precise_op_on_its_accuracy_with_a_speed_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIXAL3D_INT8PV", raising=False)
    px = sage.PASCAL_PRECISE
    card = _Pascal(monkeypatch, precise=(px,), wrong=("shipped",), ms={"shipped": 10.0, "k64": 8.9, px: 9.1})
    q, k, vs, want = sage._check_table(CPU)
    twins = {"fp32": "shipped", "int8": "k64"}
    found = sage._pascal_precise_trial(CPU, q, k, vs, want, twins)
    assert found["present"] and found["ok"], found
    assert found["rel"][f"fp32_{px}"] < found["rel"]["fp32_shipped"]
    assert {(r["mode"], r["against"]) for r in found["timings"]} == {("fp32", "shipped"), ("int8", "k64")}
    assert {r["tokens"] for r in found["timings"]} == {96, 64} and "error" in found["summary"]
    card.ms = {"shipped": 10.0, "k64": 8.9, px: 10.0}                    # +12 % against k64: over the guard
    found = sage._pascal_precise_trial(CPU, q, k, vs, want, twins)
    assert not found["ok"] and "slower" in found["why"]
    assert sage.PRECISE_MAX_SLOWDOWN == 0.10
    card = _Pascal(monkeypatch, precise=(px,), wrong=(px,), ms={"shipped": 10.0, px: 5.0})
    found = sage._pascal_precise_trial(CPU, q, k, vs, want, twins)
    assert not found["ok"] and "not more exact" in found["why"] and "timings" not in found   # never on speed
    card = _Pascal(monkeypatch)
    found = sage._pascal_precise_trial(CPU, q, k, vs, want, twins)
    assert found == {"present": False, "ok": False, "why": "no precise op in the installed wheel"}
    card = _Pascal(monkeypatch, precise=(px,), wrong=("shipped",))

    def raises(*args: Any, **kw: Any) -> Any:
        raise RuntimeError("CUDA error: an illegal memory access")

    monkeypatch.setattr(sage, "_race_pascal_twins", raises)
    found = sage._pascal_precise_trial(CPU, q, k, vs, want, twins)
    assert not found["ok"] and "CUDA error" in found["error"]


def test_a_trial_that_dies_in_the_precise_op_keeps_the_shipped_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """The child prints the trial after the twins' race, so a precise op that takes the process down leaves the
    candidate and the int8 P.V race standing and promotes nothing (the twins and the precise op left out)."""
    partial = {"candidate": "pascal", "candidate_ok": True, "why": "ran", "timings": [], "summary": "s",
               "int8pv": True, "int8pv_why": "faster", "pascal_twins": {"present": ["k64"], "fp32": "shipped",
                                                                        "int8": "k64"}}
    whole = {**partial, "pascal_precise": {"present": True, "ok": True}}
    for out in (json.dumps(partial) + "\n", json.dumps(partial) + "\n" + json.dumps(whole) + "\n"):
        monkeypatch.setattr(subprocess, "run", lambda argv, out=out, **kw: subprocess.CompletedProcess(
            argv, -11, stdout=out, stderr="Segmentation fault"))
        found = sage._old_card_trial_in_child(6, 1)
        assert found["candidate_ok"] is True and found["int8pv"] is True
        assert found["pascal_twins"]["int8"] == "shipped"
        assert found["pascal_precise"]["ok"] is False and "code -11" in found["pascal_precise"]["error"]
