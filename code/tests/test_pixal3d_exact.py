"""The conditioning and the coarse stages computed exactly (code/pixal3d/pixal3d/modules/exact.py) on the CPU, with
fakes: on which profiles and by which switch; that the ss and lr flows, once the engine prepared them, run no
SageAttention, no fused kernels and no fold while hr and tex keep them, and that their load makes SageAttention's
self-check; that the cut-out and DINOv3's tokens are made in float32. No kernel runs: sage.usable and fused.active
are asked about operands that look as they do on the card.

Runs in the pixal3d venv, with the card hidden and pytest from code/logs/profiling/pytools:
    set CUDA_VISIBLE_DEVICES=-1
    set PYTHONPATH=code\\logs\\profiling\\pytools
    code\\venvs\\pixal3d\\Scripts\\python.exe -m pytest -p no:cacheprovider code\\tests\\test_pixal3d_exact.py -q
Skipped in the hub venv (no torch).
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from spellcast import paths

torch = pytest.importorskip("torch")
if not (paths.PIXAL3D_DIR / "pixal3d" / "modules" / "sage.py").is_file():
    pytest.skip("the Pixal3D checkout is not here", allow_module_level=True)

_ENV = {name: os.environ.get(name) for name in ("FLEX_GEMM_AUTOTUNE_CACHE_PATH", "TRITON_CACHE_DIR")}
sys.path.insert(0, str(paths.PIXAL3D_DIR))
try:
    from pixal3d.modules import exact, fused, precision, sage     # noqa: E402
    from pixal3d.modules.sparse import config as sparse_config     # noqa: E402
    from pixal3d.pipelines import proj_cond                        # noqa: E402
    from pixal3d.staged import builders, engine, parts             # noqa: E402
finally:
    sys.path.remove(str(paths.PIXAL3D_DIR))
    for _name, _value in _ENV.items():
        if _value is None:
            os.environ.pop(_name, None)
        else:
            os.environ[_name] = _value


@pytest.fixture(autouse=True)
def _profile(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Each test on the profile and switches it sets; the process's profile back afterwards."""
    for name in ("PIXAL3D_COARSE_EXACT", "PIXAL3D_PRECISION", "PIXAL3D_CARD_SM", "PIXAL3D_SAGE_FLOWS"):
        monkeypatch.delenv(name, raising=False)
    precision.reset()
    yield
    precision.reset()


def _set(monkeypatch: pytest.MonkeyPatch, profile: str = "", switch: str | None = None) -> None:
    if profile:
        monkeypatch.setenv("PIXAL3D_PRECISION", profile)
    if switch is not None:
        monkeypatch.setenv("PIXAL3D_COARSE_EXACT", switch)
    precision.reset()


@pytest.mark.parametrize("profile, switch, on", [
    ("", None, True), ("ampere", None, True), ("ampere", "on", True), ("ampere", "1", True), ("ampere", "", True),
    ("ampere", "maybe", True), ("ampere", "off", False), ("ampere", "0", False), ("ampere", " OFF ", False),
    ("turing", None, False), ("pascal", None, False), ("turing", "on", False)])
def test_the_coarse_stages_are_exact_on_the_ampere_profile_unless_switched_off(
        profile: str, switch: str | None, on: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    _set(monkeypatch, profile, switch)
    assert exact.on() is on
    assert exact.active() is False


def test_the_scope_nests_and_ends_with_its_block() -> None:
    with exact.scope():
        with exact.scope():
            assert exact.active()
        assert exact.active()
    assert not exact.active()
    with pytest.raises(RuntimeError):
        with exact.scope():
            raise RuntimeError("a failed pass")
    assert not exact.active()


def _on_card(dtype: Any) -> Any:
    """What sage.usable and fused.active read of a flow's tensors on the card."""
    return SimpleNamespace(is_cuda=True, device=torch.device("cuda", 0), dtype=dtype, shape=(1, 4096, 12, 128))


class _SelfAttention(torch.nn.Module):
    """The two layers sage.fold scales and the attributes it looks for."""

    def __init__(self) -> None:
        super().__init__()
        self._type, self.attn_mode, self.channels = "self", "full", 4
        self.to_qkv = torch.nn.Linear(4, 12)
        self.to_out = torch.nn.Linear(4, 4)
        self.sage_folded = False


_SelfAttention.__name__ = "SparseMultiHeadAttention"


class _Flow(torch.nn.Module):
    """A flow network whose pass asks what the real blocks ask: SageAttention for its self-attention, the fused
    kernels for its elementwise chains."""

    def __init__(self) -> None:
        super().__init__()
        block = torch.nn.Module()
        block.attn = _SelfAttention()
        self.blocks = torch.nn.ModuleList([block])

    def forward(self, x: Any) -> dict:
        torso = _on_card(precision.torso())
        return {"sage": sage.usable(torso, torso, torso), "fused": fused.active(_on_card(torch.bfloat16))}


@pytest.fixture
def kernels_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """SageAttention and the fused kernels as after their self-checks on the card."""
    monkeypatch.setattr(sparse_config, "SAGE", True)
    monkeypatch.setattr(sage, "_state", True)
    monkeypatch.setattr(fused, "MODE", "rounded")
    monkeypatch.setattr(fused, "_checked", True)


@pytest.mark.parametrize("switch, exact_kinds", [(None, {"ss", "lr"}), ("off", set())])
def test_the_coarse_flows_run_none_of_the_faster_paths(switch: str | None, exact_kinds: set, kernels_ready: None,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    _set(monkeypatch, "ampere", switch)
    for kind in ("ss", "lr", "hr", "tex"):
        flow = _Flow()
        weights = [p.detach().clone() for p in flow.parameters()]
        assert engine._prepare_flow(flow, kind=kind, checkpoint="x") is flow
        with torch.no_grad():
            asked = flow(None)
        folded = not all(torch.equal(a, b) for a, b in zip(weights, flow.parameters()))
        if kind in exact_kinds:
            assert asked == {"sage": False, "fused": False}, kind
            assert not folded and not flow.blocks[0].attn.sage_folded, kind
        else:
            assert asked == {"sage": True, "fused": True}, kind
            assert folded and flow.blocks[0].attn.sage_folded, kind
        assert not exact.active()


class _LoadedFlow(torch.nn.Module):
    """A flow network as the loader hands it over: its weights on the card."""

    def parameters(self, recurse: bool = True) -> Any:
        return iter([SimpleNamespace(is_cuda=True, device=torch.device("cuda", 0))])

    def forward(self, x: Any) -> Any:
        return x


@pytest.mark.parametrize("profile, switch, sage_on, checked_at_load", [
    ("ampere", None, True, ["fused", "sage"]), ("ampere", None, False, ["fused"]), ("ampere", "off", True, []),
    ("turing", None, True, []), ("pascal", None, True, [])])
def test_the_self_checks_run_when_the_first_coarse_flow_is_loaded(profile: str, switch: str | None, sage_on: bool,
                                                                  checked_at_load: list,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """The coarse flows' passes never ask for the fused kernels or SageAttention: their load makes the two
    self-checks, so that they run before the first flow's first pass (the card holding the coarse flow only), not
    inside the first detail step. Elsewhere they stay where they were: at the first call that may use them."""
    _set(monkeypatch, profile, switch)
    monkeypatch.setattr(sparse_config, "SAGE", sage_on)
    monkeypatch.setattr(sage, "_state", None)
    monkeypatch.setattr(fused, "MODE", "rounded")
    monkeypatch.setattr(fused, "_checked", False)
    checks: list = []

    def check_sage(device: Any) -> bool:
        checks.append(("sage", str(device)))
        sage._state = True
        return True

    def check_fused(device: Any) -> bool:
        checks.append(("fused", str(device)))
        fused._checked = True
        return True

    monkeypatch.setattr(sage, "_check", check_sage)
    monkeypatch.setattr(fused, "_check", check_fused)
    for kind in ("ss", "lr"):
        engine._prepare_flow(_LoadedFlow(), kind=kind, checkpoint="x")
    assert checks == [(name, "cuda:0") for name in checked_at_load]


@pytest.mark.parametrize("profile", ["turing", "pascal"])
def test_the_old_cards_flows_are_prepared_as_before(profile: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _set(monkeypatch, profile)
    monkeypatch.setattr(sparse_config, "SAGE", True)
    for kind in ("ss", "lr"):
        flow = _Flow()
        engine._prepare_flow(flow, kind=kind, checkpoint="x")
        assert not getattr(flow, "exact_scope", False) and flow.blocks[0].attn.sage_folded


def test_the_matte_and_dino_keep_float32_in_the_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    cpu = torch.device("cpu")
    monkeypatch.setattr(builders, "_matte_fp16_on", True)
    monkeypatch.setattr(proj_cond, "_dino_fp16_on", True)
    assert builders.matte_fp16(cpu) and proj_cond.dino_fp16(cpu)
    with exact.scope():
        assert not builders.matte_fp16(cpu) and not proj_cond.dino_fp16(cpu)
    assert builders.matte_fp16(cpu) and proj_cond.dino_fp16(cpu)


class _Store:
    def __init__(self) -> None:
        self.small: dict = {}

    def put_small(self, prop: str, key: str, value: Any) -> None:
        self.small[key] = value

    def get_small(self, prop: str, key: str) -> Any:
        return self.small[key]

    def put_device(self, prop: str, key: str, value: Any) -> None:
        self.small[key] = value


@pytest.mark.parametrize("switch, coarse", [(None, True), ("off", False)])
def test_the_cutout_and_dinos_tokens_are_made_in_the_scope(switch: str | None, coarse: bool,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    _set(monkeypatch, "ampere", switch)
    seen: list = []

    class Conditioning:
        SPECS = {"ss": "ss"}

        @staticmethod
        def image_tensor(source: Any, size: int, device: Any) -> int:
            return size

        @staticmethod
        def encode(dino: Any, size: int) -> dict:
            seen.append(("dino", size, exact.active()))
            return {"size": size}

        @staticmethod
        def cond_dense(spec: Any, z: Any, view: Any) -> dict:
            return {"z": z}

    def matte(image: Any) -> Any:
        seen.append(("matte", exact.active()))
        return image

    def prepare(req: Any, cut: Any) -> Any:
        return SimpleNamespace(images=[cut("picture")], meta={}, cutout=None)

    monkeypatch.setattr(parts, "_proj_cond", lambda: Conditioning)
    monkeypatch.setattr(parts, "_network", lambda ctx, name: matte if name == "rembg" else "dino")
    monkeypatch.setattr(parts, "_source", lambda ctx, prop: ("source", "view"))
    monkeypatch.setattr(parts, "_torso", lambda cond: cond)
    monkeypatch.setattr(parts, "_keep_bundle", lambda ctx, prop, source: None)
    monkeypatch.setattr(parts, "geometry", lambda prop: "pixal3d")
    ctx = SimpleNamespace(device="cpu", store=_Store(), hooks=SimpleNamespace(prepare=prepare))
    prop = SimpleNamespace(id="p", req=SimpleNamespace(), views=0)
    parts.cutout(ctx, prop, "normal")
    parts.features(ctx, prop, "normal")
    assert seen == [("matte", coarse), ("dino", 512, coarse), ("dino", 1024, coarse)]
    assert not exact.active()
