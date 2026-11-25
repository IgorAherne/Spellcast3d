"""The fork's copies of checkpoints in their models' dtypes (code/pixal3d/pixal3d/staged/weights.py) and the
network loader reading them (loader.py), on the CPU with small stand-in models: what a copy holds and how it is
cast, when a copy is stale, foreign or broken, what a pipeline folder gets, and that a load reads the copy when it
may and the original otherwise, with `bytes` saying which.

Runs in the pixal3d venv, with the card hidden and pytest from code/logs/profiling/pytools:
    set CUDA_VISIBLE_DEVICES=-1
    set PYTHONPATH=code\\logs\\profiling\\pytools
    code\\venvs\\pixal3d\\Scripts\\python.exe -m pytest -p no:cacheprovider code\\tests\\test_pixal3d_weights.py -q
Skipped in the hub venv (no torch). Whether a copy's bfloat16 is bit for bit the cast the loader makes on the card
is a check for the GPU: here both sides are the CPU's cast.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from spellcast import paths

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")             # the loader builds models without weights through it
save_file = pytest.importorskip("safetensors.torch").save_file
if not (paths.PIXAL3D_DIR / "pixal3d" / "staged" / "weights.py").is_file():
    pytest.skip("the Pixal3D checkout is not here", allow_module_level=True)

# The fork, imported while this file is collected: later, code/ on the import path (conftest.py) can make
# `pixal3d` the checkout's folder seen as a namespace package. It is taken off the path again at once, so that the
# other test files of a run do not find it there.
sys.path.insert(0, str(paths.PIXAL3D_DIR))
try:
    from pixal3d.modules import precision      # noqa: E402
    from pixal3d.staged import loader as L     # noqa: E402
    from pixal3d.staged import stfile           # noqa: E402
    from pixal3d.staged import weights as W     # noqa: E402
    from pixal3d.staged.types import Cancelled  # noqa: E402
finally:
    sys.path.remove(str(paths.PIXAL3D_DIR))

nn = torch.nn


class Block(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.mix = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)


class Flow(nn.Module):
    """Shaped like Pixal3D's flow models: a bf16 torso, fp32 ends, a complex buffer the file carries."""

    def __init__(self, width: int = 24, depth: int = 3):
        super().__init__()
        self.first = nn.Linear(8, width)
        angles = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        self.register_buffer("phases", torch.polar(torch.ones_like(angles), angles))
        self.blocks = nn.ModuleList([Block(width) for _ in range(depth)])
        for block in self.blocks:
            block.mix.to(torch.bfloat16)


class Typed(nn.Module):
    """A flow whose checkpoint names its torso's dtype, as Pixal3D's jsons do ("dtype": "bfloat16"): the card's
    precision profile makes it the torso's (precision.model_args), so the copies hold what the card runs."""
    def __init__(self, width: int = 24, depth: int = 3, dtype: str = "bfloat16"):
        super().__init__()
        self.first = nn.Linear(8, width)
        self.blocks = nn.ModuleList([Block(width) for _ in range(depth)])
        for block in self.blocks:
            block.mix.to({"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype])


class Kept(nn.Module):
    """Keeps every tensor in float32, as its file has them."""

    def __init__(self, width: int = 24, depth: int = 3):
        super().__init__()
        self.first = nn.Linear(8, width)


class Half(nn.Module):
    """A decoder whose file and model are all float16."""

    def __init__(self, width: int = 24, depth: int = 3):
        super().__init__()
        self.first = nn.Linear(8, width).to(torch.float16)


class Wide(nn.Module):
    """A decoder that keeps in float32 what its file stores in float16 (Pixal3D's sparse decoders do so)."""

    def __init__(self, width: int = 24, depth: int = 3):
        super().__init__()
        self.first = nn.Linear(8, width)


class MultiHeadAttention(nn.Module):
    """Named as the flows' attention, which modules/int8.py converts: to_qkv (group scales) and to_out (rows)."""
    def __init__(self, channels: int):
        super().__init__()
        self._type, self.attn_mode, self.channels = "self", "full", channels
        self.to_qkv = nn.Linear(channels, 3 * channels)
        self.to_out = nn.Linear(channels, channels)


class FeedForwardNet(nn.Module):
    """Named as the flows' feed-forward: fc1, GELU, fc2, converted together as one Int8MLP."""
    def __init__(self, channels: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, 2 * channels), nn.GELU(approximate="tanh"),
                                 nn.Linear(2 * channels, channels))


class Int8Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.self_attn = MultiHeadAttention(channels)
        self.mlp = FeedForwardNet(channels)


class Int8Flow(nn.Module):
    """A detail flow in small: fp32 ends, a bf16 torso of blocks whose linears the int8 policy converts."""
    def __init__(self, width: int = 128, depth: int = 2):
        super().__init__()
        self.first = nn.Linear(8, width)
        self.blocks = nn.ModuleList([Int8Block(width) for _ in range(depth)])
        for block in self.blocks:
            block.self_attn.to(torch.bfloat16)
            block.mlp.to(torch.bfloat16)


class MetaBuffered(Int8Flow):
    """An Int8Flow whose constructor makes a non-persistent buffer on its parameters' device - the meta device in a
    build without weights: no file carries it, only a build for real gives it values."""
    def __init__(self, width: int = 128, depth: int = 2):
        super().__init__(width, depth)
        self.register_buffer("phases", torch.arange(4, dtype=torch.float32, device=self.first.weight.device),
                             persistent=False)


CLASSES = {"Flow": Flow, "Typed": Typed, "Kept": Kept, "Half": Half, "Wide": Wide,
           "Int8Flow": Int8Flow, "MetaBuffered": MetaBuffered}
SIZES = {"Int8Flow": {"width": 128, "depth": 2}}     # the int8 GEMMs take multiples of 128
# float32 values around the halfway points of bfloat16 (7 bits after the point), and what rounding to the nearest,
# ties to even, gives: truncation would give 0x3F80, 0x3F81, 0x3F81, 0xBF80, 0x7F7F.
TIES = {1 + 2 ** -8: 0x3F80, 1 + 3 * 2 ** -8: 0x3F82, 1 + 2 ** -8 + 2 ** -20: 0x3F81, -(1 + 2 ** -8): 0xBF80 - 0x10000,
        3.4028234663852886e38: 0x7F80}


@pytest.fixture(autouse=True)
def classes(monkeypatch: pytest.MonkeyPatch) -> None:
    """This file's stand-in classes are the loader's models."""
    monkeypatch.setattr(L, "_model_class", CLASSES.__getitem__)


def write_checkpoint(ckpt: Path, cls: str, stored: torch.dtype = torch.float32, *, also: dict[str, torch.dtype] | None
                     = None, leave_out: tuple[str, ...] = (), seed: int = 1, args: dict | None = None) -> Path:
    """<ckpt>.json naming the class, and <ckpt>.safetensors (written by the safetensors library, as Pixal3D's
    files are) with random values for the model's state, stored in `stored` (`also`: other dtypes per key), the
    complex buffer as it is, plus a tensor the model has no place for. Flow's first bf16 weight starts with TIES
    and a NaN, an infinity and a subnormal."""
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    state = {}
    for key, value in CLASSES[cls](**SIZES.get(cls, {})).state_dict().items():
        if key in leave_out:
            continue
        if value.is_complex():
            state[key] = value
            continue
        values = torch.randn(value.shape, generator=generator) * 0.1
        if key == "blocks.0.mix.weight":
            flat = values.view(-1)
            flat[:len(TIES)] = torch.tensor(list(TIES))
            flat[len(TIES):len(TIES) + 3] = torch.tensor([float("nan"), float("inf"), 1e-40])
        state[key] = values.to((also or {}).get(key, stored))
    state["unused.weight"] = torch.randn(5, generator=generator).to(stored)
    save_file(state, f"{ckpt}.safetensors")
    sizes = SIZES.get(cls, {"width": 24, "depth": 3})
    Path(f"{ckpt}.json").write_text(json.dumps({"name": cls, "args": {**sizes, **(args or {})}}), encoding="utf-8")
    return ckpt


def raw(tensor: torch.Tensor) -> bytes:
    """A tensor's bytes, to compare bit for bit (NaN included)."""
    if tensor.is_complex():
        tensor = torch.view_as_real(tensor)
    return bytes(tensor.contiguous().view(-1).view(torch.uint8).numpy())


def tensors_of(path: Path) -> dict[str, torch.Tensor]:
    buffer = torch.empty(4 << 20, dtype=torch.uint8)
    return {name: value.clone() for name, value in stfile.read_into(path, None, buffer).items()}


def plan_of(ckpt: Path) -> Any:
    return L._plan(Path(f"{ckpt}.json"), Path(f"{ckpt}.safetensors"))


def rewrite(copy: Path, *, metadata: dict[str, str] | None = None, drop: str = "",
            dtype: tuple[str, torch.dtype] | None = None, swap: bool = False) -> None:
    """The copy written again with other metadata, without a tensor, with a tensor in another dtype, or with its
    first two tensors swapped."""
    tensors = tensors_of(copy)
    names = [name for name in stfile.read_header(copy) if name != drop]
    if swap:
        names[0], names[1] = names[1], names[0]
    if dtype is not None:
        tensors[dtype[0]] = tensors[dtype[0]].to(dtype[1])
    stfile.write_streamed(copy, [(name, tensors[name].dtype, tuple(tensors[name].shape)) for name in names],
                          ((name, tensors[name].reshape(-1)) for name in names),
                          metadata=stfile.read_metadata(copy) if metadata is None else metadata)


# ---- what a copy holds ---------------------------------------------------------------------------------------

def test_a_copy_holds_what_the_model_takes_in_its_dtypes_rounded_to_nearest_even(tmp_path: Path) -> None:
    ckpt = write_checkpoint(tmp_path / "flow", "Flow")
    source, copy = Path(f"{ckpt}.safetensors"), W.copy_of(Path(f"{ckpt}.safetensors"))
    outcome = W.make_copy(ckpt)
    assert copy.name == "flow.model-dtypes.safetensors" and copy.parent == tmp_path
    assert outcome.state == "written" and outcome.checkpoint == "flow" and not Path(f"{copy}.tmp").exists()
    assert outcome.copy_bytes == copy.stat().st_size < outcome.source_bytes == source.stat().st_size
    assert outcome.line().startswith("flow: copy written, 0.00 GB read in place of 0.00 GB (")

    want = Flow().state_dict()
    original, header = stfile.read_header(source), stfile.read_header(copy)
    assert list(header) == [name for name in original if name in want]   # the original's order, no "unused"
    stored, got = tensors_of(source), tensors_of(copy)
    for name in header:
        assert got[name].dtype == want[name].dtype and got[name].shape == want[name].shape, name
        assert raw(got[name]) == raw(stored[name].to(want[name].dtype)), name
    assert got["blocks.0.mix.weight"].dtype == torch.bfloat16 and got["first.weight"].dtype == torch.float32
    assert got["phases"].dtype == torch.complex64 and got["blocks.2.norm.bias"].dtype == torch.float32
    first = got["blocks.0.mix.weight"].view(-1).view(torch.int16)
    assert first[:len(TIES)].tolist() == list(TIES.values())
    assert torch.isnan(got["blocks.0.mix.weight"].view(-1)[len(TIES)])

    with open(source, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(0)
        digest = hashlib.sha256(handle.read(8 + length)).hexdigest()
    status = os.stat(source)
    assert stfile.read_metadata(copy) == {
        "copy_of": "flow.safetensors", "format": W.FORMAT, "source_bytes": str(status.st_size),
        "source_mtime_ns": str(status.st_mtime_ns), "source_header_sha256": digest, "torch": torch.__version__}

    before = copy.stat().st_mtime_ns
    again = W.make_copy(ckpt)
    assert again.state == "up to date" and copy.stat().st_mtime_ns == before
    assert again.line().startswith("flow: copy up to date, ")

    whole = copy.read_bytes()
    copy.unlink()
    # a buffer smaller than the bf16 weights (2,304 bytes in fp32), not a multiple of 4: those go piece by piece
    assert W.make_copy(ckpt, torch.empty(1030, dtype=torch.uint8)).state == "written"
    assert copy.read_bytes() == whole


@pytest.mark.parametrize("card, torso", [("86", torch.bfloat16), ("75", torch.float16), ("61", None)])
def test_a_copy_holds_the_cards_torso(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, card: str,
                                      torso: torch.dtype | None) -> None:
    """The installer makes the copies with the card hidden and PIXAL3D_CARD_SM naming it (round L): a flow's bf16
    torso is bf16 on sm_80+, fp16 on Turing, and fp32 on Pascal - the file's own dtype, so no copy is made."""
    monkeypatch.setenv("PIXAL3D_CARD_SM", card)
    precision.reset()
    try:
        ckpt = write_checkpoint(tmp_path / "typed", "Typed", args={"dtype": "bfloat16"})
        outcome = W.make_copy(ckpt)
        if torso is None:
            assert (outcome.state, outcome.why) == ("not needed", "its model keeps every tensor as the file has it")
            return
        assert outcome.state == "written"
        got = tensors_of(W.copy_of(Path(f"{ckpt}.safetensors")))
        assert got["blocks.0.mix.weight"].dtype == torso and got["first.weight"].dtype == torch.float32
    finally:
        precision.reset()


def test_checkpoints_that_gain_nothing_get_no_copy(tmp_path: Path) -> None:
    kept = write_checkpoint(tmp_path / "kept", "Kept")
    half = write_checkpoint(tmp_path / "half", "Half", torch.float16)
    wide = write_checkpoint(tmp_path / "wide", "Wide", also={"first.weight": torch.float16})
    W.copy_of(Path(f"{kept}.safetensors")).write_bytes(b"an old copy")
    outcomes = {ckpt.name: W.make_copy(ckpt) for ckpt in (kept, half, wide)}
    assert {name: (o.state, o.why) for name, o in outcomes.items()} == {
        "kept": ("not needed", "its model keeps every tensor as the file has it"),
        "half": ("not needed", "every tensor takes two bytes or fewer"),
        "wide": ("not needed", "in its model's dtypes it would not be smaller")}
    assert outcomes["half"].line().startswith("half: no copy needed, every tensor takes two bytes or fewer (")
    assert not list(tmp_path.glob(f"*{W.COPY_SUFFIX}*"))               # the old copy is gone too


def test_a_copy_is_made_only_with_room_and_only_from_an_original_that_stays_put(tmp_path: Path,
                                                                                 monkeypatch: pytest.MonkeyPatch
                                                                                 ) -> None:
    ckpt = write_checkpoint(tmp_path / "flow", "Flow")
    source, copy = Path(f"{ckpt}.safetensors"), W.copy_of(Path(f"{ckpt}.safetensors"))
    real_usage = W.shutil.disk_usage
    monkeypatch.setattr(W.shutil, "disk_usage", lambda path: real_usage(path)._replace(free=W.DISK_SPARE_BYTES))
    with pytest.raises(W.NoRoom, match="must stay free"):
        W.make_copy(ckpt)
    assert not copy.exists() and not Path(f"{copy}.tmp").exists()
    monkeypatch.setattr(W.shutil, "disk_usage", real_usage)

    real_cast = W._cast

    def touching(*args: Any) -> Any:              # the original is downloaded again while its copy is made
        for item in real_cast(*args):
            status = os.stat(source)
            os.utime(source, ns=(status.st_atime_ns, status.st_mtime_ns + 1000))
            yield item

    monkeypatch.setattr(W, "_cast", touching)
    with pytest.raises(RuntimeError, match="changed while its copy was made"):
        W.make_copy(ckpt)
    assert not copy.exists() and not Path(f"{copy}.tmp").exists()
    monkeypatch.setattr(W, "_cast", real_cast)
    assert W.make_copy(ckpt).state == "written"


# ---- when a copy is read -------------------------------------------------------------------------------------

def test_a_stale_foreign_or_broken_copy_is_ignored_and_made_again(tmp_path: Path) -> None:
    ckpt = write_checkpoint(tmp_path / "flow", "Flow")
    source, copy = Path(f"{ckpt}.safetensors"), W.copy_of(Path(f"{ckpt}.safetensors"))
    plan = plan_of(ckpt)

    def usable() -> bool:
        found = W.usable_copy(source, plan.keys, plan.want)
        assert found is None or found[0] == copy
        return found is not None

    assert not usable() and W.current_copy(source) is None              # none yet
    W.make_copy(ckpt)
    good = copy.read_bytes()
    assert usable() and W.current_copy(source) == copy
    status = os.stat(source)

    os.utime(source, ns=(status.st_atime_ns, status.st_mtime_ns + 1000))  # the original was written again
    assert not usable() and W.current_copy(source) is None
    os.utime(source, ns=(status.st_atime_ns, status.st_mtime_ns))
    assert usable()

    text = source.read_bytes()                                          # same size and time, another header
    assert text.count(b"unused.weight") == 1
    source.write_bytes(text.replace(b"unused.weight", b"unused.weighs"))
    os.utime(source, ns=(status.st_atime_ns, status.st_mtime_ns))
    assert os.stat(source).st_size == status.st_size and not usable()
    source.write_bytes(text)
    os.utime(source, ns=(status.st_atime_ns, status.st_mtime_ns))
    assert usable()

    metadata = stfile.read_metadata(copy)
    for key, value in (("torch", "0.0.0"), ("format", "0"), ("source_bytes", "1"), ("copy_of", "other.safetensors")):
        rewrite(copy, metadata={**metadata, key: value})                 # another torch, format, original
        assert not usable(), key
    rewrite(copy, metadata={})
    assert not usable()
    for change in ({"drop": "first.bias"}, {"dtype": ("blocks.1.mix.weight", torch.float32)}, {"swap": True}):
        copy.write_bytes(good)
        rewrite(copy, **change)
        assert W.current_copy(source) == copy and not usable(), change  # current, but not what the model takes
    for broken in (good[:-10], b"\xff" * 64, b""):
        copy.write_bytes(broken)
        assert not usable() and W.current_copy(source) is None
    copy.unlink()
    copy.mkdir()
    assert not usable()
    copy.rmdir()

    copy.write_bytes(good)
    rewrite(copy, metadata={**metadata, "torch": "0.0.0"})
    assert W.make_copy(ckpt).state == "written" and copy.read_bytes() == good


class Reads:
    """Which files stfile.iter_tensors reads, and a copy that breaks after `breaks_after` tensors."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.files: list[str] = []
        self.breaks_after: int | None = None
        real = stfile.iter_tensors

        def spy(path: Any, buffer: Any, names: Any = None) -> Any:
            self.files.append(Path(path).name)
            inner = real(path, buffer, names)
            try:
                for count, item in enumerate(inner):
                    if self.breaks_after is not None and count == self.breaks_after and W.COPY_SUFFIX in str(path):
                        raise OSError(22, "a bad sector")
                    yield item
            finally:
                inner.close()

        monkeypatch.setattr(stfile, "iter_tensors", spy)


def test_the_loader_reads_the_copy_when_it_may_and_the_original_otherwise(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = write_checkpoint(tmp_path / "flow", "Flow")
    source, copy = Path(f"{ckpt}.safetensors"), W.copy_of(Path(f"{ckpt}.safetensors"))
    events: list[dict] = []
    loader = L.NetworkLoader("cpu", prefetch="off", on_event=events.append)
    loader.register_checkpoint("flow", ckpt)
    reads = Reads(monkeypatch)
    blocks: list[int] = []                       # blocks of bytes the loader takes on its device (the cast's)
    real_empty = torch.empty

    def empty(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("dtype") == torch.uint8 and kwargs.get("device") == loader.device:
            blocks.append(int(args[0]))
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)

    def load(cancelled: Callable[[], bool] | None = None) -> tuple[dict[str, torch.Tensor], list[dict]]:
        events.clear()
        reads.files.clear()
        blocks.clear()
        module = loader.load("flow", cancelled)
        state = {name: value.clone() for name, value in module.state_dict().items()}
        del module
        loader.drop("flow")
        return state, [event for event in events if event["kind"] != "net_drop"]

    original_bytes = sum(info.nbytes for name, info in stfile.read_header(source).items() if name != "unused.weight")
    plain, told = load()
    assert reads.files == ["flow.safetensors"] and blocks and max(blocks) > 0      # cast through one block
    assert told[0] == {"kind": "net_load", "name": "flow", "bytes_read": 0, "bytes": original_bytes}
    assert told[-1]["kind"] == "net_ready" and told[-1]["bytes"] == original_bytes

    W.make_copy(ckpt)
    copy_bytes = sum(info.nbytes for info in stfile.read_header(copy).values())
    assert copy_bytes < original_bytes
    copied, told = load()
    assert reads.files == [copy.name] and blocks == []                              # nothing to cast
    assert told[0]["bytes"] == copy_bytes == told[-1]["bytes"] and told[-1]["kind"] == "net_ready"
    assert loader.stats()["loads"][-1]["bytes"] == copy_bytes
    assert list(copied) == list(plain)
    for name in plain:
        assert copied[name].dtype == plain[name].dtype and raw(copied[name]) == raw(plain[name]), name

    status = os.stat(source)
    os.utime(source, ns=(status.st_atime_ns, status.st_mtime_ns + 1000))           # stale: the original
    stale, told = load()
    assert reads.files == ["flow.safetensors"] and told[-1]["bytes"] == original_bytes
    os.utime(source, ns=(status.st_atime_ns, status.st_mtime_ns))

    reads.breaks_after = 3                                                          # the copy breaks half-way
    broke, told = load()
    assert reads.files == [copy.name, "flow.safetensors"]
    starts = [(event["kind"], event["bytes"]) for event in told
              if event["kind"] == "net_ready" or event["bytes_read"] == 0]
    assert starts == [("net_load", copy_bytes), ("net_load", original_bytes), ("net_ready", original_bytes)]
    assert all(raw(broke[name]) == raw(plain[name]) for name in plain)
    reads.breaks_after = None

    calls: list[int] = []

    def after_two() -> bool:
        calls.append(1)
        return len(calls) > 2

    with pytest.raises(Cancelled):                                                  # a cancel is no failure
        load(after_two)
    assert reads.files == [copy.name] and loader.loaded() == []
    again, told = load()
    assert reads.files == [copy.name] and told[-1]["bytes"] == copy_bytes


def test_the_copy_is_what_is_read_ahead(tmp_path: Path) -> None:
    ckpt = write_checkpoint(tmp_path / "flow", "Flow")
    source, copy = Path(f"{ckpt}.safetensors"), W.copy_of(Path(f"{ckpt}.safetensors"))

    def read_ahead() -> int:
        """The size of the file a fresh loader reads ahead for the network (known before its first read)."""
        loader = L.NetworkLoader("cpu", prefetch="cache")
        loader.register_checkpoint("flow", ckpt)
        try:
            loader.prefetch("flow")
            end = time.monotonic() + 20
            while time.monotonic() < end:
                status = loader.stats()["prefetch"]
                if status is not None and status["bytes"]:
                    return status["bytes"]
                time.sleep(0.01)
            raise AssertionError(f"nothing was read ahead: {loader.stats()['prefetch']}")
        finally:
            loader.close()

    assert read_ahead() == source.stat().st_size
    W.make_copy(ckpt)
    assert read_ahead() == copy.stat().st_size
    status = os.stat(source)
    os.utime(source, ns=(status.st_atime_ns, status.st_mtime_ns + 1000))
    assert read_ahead() == source.stat().st_size
    assert not [thread for thread in threading.enumerate() if thread.name == "p3d-net-prefetch"]


# ---- the int8 copies of the detail flows ---------------------------------------------------------------------

def int8_reference(ckpt: Path) -> dict[str, torch.Tensor]:
    """The state int8.convert's layers make of the checkpoint's weights in the model's dtypes: Int8Linear and
    Int8MLP built by hand from the model loaded on the CPU (the torch reference quantiser, the card's bits)."""
    from pixal3d.modules import int8 as I

    model = Int8Flow()
    stored = tensors_of(Path(f"{ckpt}.safetensors"))
    model.load_state_dict({key: stored[key].to(value.dtype) for key, value in model.state_dict().items()})
    with torch.no_grad():
        for block in model.blocks:
            block.self_attn.to_qkv = I.Int8Linear(block.self_attn.to_qkv, "group")
            block.self_attn.to_out = I.Int8Linear(block.self_attn.to_out, "rows")
            block.mlp.mlp = I.Int8MLP(block.mlp.mlp[0], block.mlp.mlp[2], "group", "group")
    return model.state_dict()


def test_an_int8_copy_holds_what_convert_makes_and_is_read_only_while_it_fits(tmp_path: Path) -> None:
    from pixal3d.modules import int8 as I

    ckpt = write_checkpoint(tmp_path / "hr", "Int8Flow")
    source = Path(f"{ckpt}.safetensors")
    rules = I.copy_rules("hr")
    assert rules == {"qkv": "group", "fc1": "group", "fc2": "group", "out": "rows", "xq": "rows", "xout": "rows"}
    W.make_copy(ckpt)
    outcome = W.make_int8_copy(ckpt, "hr")
    copy = W.int8_copy_of(source)
    assert copy.name == f"hr{W.INT8_SUFFIX}" and outcome.state == "written" and not Path(f"{copy}.tmp").exists()
    got, want = tensors_of(copy), int8_reference(ckpt)
    assert sorted(got) == sorted(want)
    for key in want:
        assert got[key].dtype == want[key].dtype and raw(got[key]) == raw(want[key]), key
    assert got["blocks.0.self_attn.to_qkv.weight_q"].dtype == torch.int8
    assert got["blocks.1.mlp.mlp.w2_s"].dtype == torch.float32 and got["first.weight"].dtype == torch.float32
    assert got["blocks.0.norm.weight"].dtype == torch.float32          # what the model keeps in fp32 stays fp32
    metadata = stfile.read_metadata(copy)
    assert metadata["kind"] == "hr" and W.recorded_rules(metadata) == rules and metadata["torso"] == "bfloat16"
    assert W.make_int8_copy(ckpt, "hr").state == "up to date"
    assert W.make_int8_copy(ckpt, "lr").state == "not needed" and not copy.exists()     # detail: lr stays bf16
    W.make_int8_copy(ckpt, "hr")

    plan = plan_of(ckpt)
    I.skeleton(plan.model, rules)
    shapes = {key: (value.dtype, tuple(value.shape)) for key, value in plan.model.state_dict().items()}
    assert W.usable_int8_copy(source, "hr", rules, shapes) is not None
    assert W.usable_int8_copy(source, "tex", rules, shapes) is None                     # another kind's
    assert W.usable_int8_copy(source, "hr", {**rules, "qkv": "rows"}, shapes) is None   # other rules
    assert W.usable_int8_copy(source, "hr", rules, {**shapes, "x": (torch.float32, (1,))}) is None
    status = os.stat(source)
    os.utime(source, ns=(status.st_atime_ns, status.st_mtime_ns + 1000))               # the original changed
    assert W.current_int8_copy(source) is None and W.usable_int8_copy(source, "hr", rules, shapes) is None
    assert W.make_int8_copy(ckpt, "hr").state == "written"


def test_a_load_reads_the_int8_copy_into_the_int8_layers_and_its_hook_still_runs(tmp_path: Path,
                                                                                 monkeypatch: pytest.MonkeyPatch
                                                                                 ) -> None:
    from pixal3d.modules import int8 as I

    ckpt = write_checkpoint(tmp_path / "hr", "Int8Flow")
    W.make_copy(ckpt)
    W.make_int8_copy(ckpt, "hr")
    copy, rules = W.int8_copy_of(Path(f"{ckpt}.safetensors")), I.copy_rules("hr")
    # off the card the loader never takes the int8 copy; here it is told to, as a process on the card is
    monkeypatch.setattr(L.NetworkLoader, "_int8_copy", lambda self, source: (copy, "hr", rules))
    seen: list = []
    loader = L.NetworkLoader("cpu", prefetch="off", on_event=seen.append)
    loader.register_checkpoint("hr", ckpt, convert=lambda model: seen.append(model) or model)
    model = loader.load("hr")
    assert isinstance(model.blocks[0].self_attn.to_qkv, I.Int8Linear)
    assert isinstance(model.blocks[1].mlp.mlp, I.Int8MLP)
    assert any(item is model for item in seen)                      # the convert hook ran, after the read
    assert [item for item in seen if isinstance(item, dict)][-1]["bytes"] == sum(
        info.nbytes for info in stfile.read_header(copy).values())
    want = int8_reference(ckpt)
    got = model.state_dict()
    assert sorted(got) == sorted(want) and all(raw(got[key]) == raw(want[key]) for key in want)
    loader.drop_all()

    rewrite(copy, drop="blocks.0.self_attn.to_out.scale")           # a copy that does not fit: the model-dtypes copy
    loader.register_checkpoint("hr", ckpt, convert=lambda model: model)
    model = loader.load("hr")
    assert isinstance(model.blocks[0].self_attn.to_qkv, nn.Linear)
    assert model.blocks[0].self_attn.to_qkv.weight.dtype == torch.bfloat16
    loader.drop_all()


def test_a_load_takes_over_its_read_ahead_and_a_read_ahead_waits_for_ram(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    import psutil

    ckpt = write_checkpoint(tmp_path / "flow", "Flow")
    other = write_checkpoint(tmp_path / "other", "Flow", seed=3)
    free = {"bytes": 0}
    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(available=free["bytes"]))
    monkeypatch.setattr(L, "PREFETCH_RAM_POLL_S", 0.01)
    loader = L.NetworkLoader("cpu", prefetch="cache")
    loader.register_checkpoint("flow", ckpt)
    loader.register_checkpoint("other", other)

    def status(name: str, state: str) -> dict:
        end = time.monotonic() + 20
        while time.monotonic() < end:
            found = loader.stats()["prefetch"]
            if found is not None and found["name"] == name and found["state"] == state:
                return found
            time.sleep(0.005)
        raise AssertionError(f"the read ahead never was {name} {state}: {loader.stats()['prefetch']}")

    try:
        loader.prefetch("flow", "other")
        status("flow", "waiting for RAM")                           # RAM short: it waits instead of giving up
        loader.load("flow")                                         # the load reads it: the read ahead lets it go
        status("other", "waiting for RAM")                          # and goes on with the rest of its request
        free["bytes"] = 1 << 50
        assert status("other", "done")["bytes_read"] == Path(f"{other}.safetensors").stat().st_size
        loader.drop_all()
        free["bytes"] = 0
        loader.prefetch("flow")
        status("flow", "waiting for RAM")
        free["bytes"] = 1 << 50
        assert status("flow", "done")["bytes_read"] == Path(f"{ckpt}.safetensors").stat().st_size
    finally:
        loader.close()


def test_an_fp16_torso_gets_no_int8_copy_and_reads_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On an RTX 20xx (the turing profile's fp16 torso) SageAttention's fold of v's 2^-8 turns small to_qkv weights
    into fp16 subnormals before convert quantises them, while a copy is quantised from the unfolded weights: other
    int8 codes. So no int8 copy is made there, one that is there is removed, and none is read."""
    from pixal3d.modules import int8 as I
    from pixal3d.modules import int8mm

    ckpt = write_checkpoint(tmp_path / "hr", "Int8Flow")
    copy = W.int8_copy_of(Path(f"{ckpt}.safetensors"))
    assert W.make_int8_copy(ckpt, "hr").state == "written"               # no card: the ampere profile
    monkeypatch.setattr(I, "_state", True)                              # the self-check passed (no card is asked)
    monkeypatch.setitem(int8mm._state, "gemm", "cuda")                  # pixal3d_kernels' group GEMM
    monkeypatch.delenv("PIXAL3D_INT8", raising=False)
    cuda = torch.device("cuda")
    try:
        monkeypatch.setenv("PIXAL3D_CARD_SM", "75")
        precision.reset()
        assert precision.profile().torso is torch.float16
        assert I.copy_rules("hr") is None and I.live_rules("hr", cuda) is None
        outcome = W.make_int8_copy(ckpt, "hr")
        assert outcome.state == "not needed" and "fp16" in outcome.why and not copy.exists()
        monkeypatch.setenv("PIXAL3D_CARD_SM", "86")                     # bf16: exact, the copy serves
        precision.reset()
        monkeypatch.setitem(int8mm._state, "gemm", "triton")
        assert I.live_rules("hr", cuda) == I.copy_rules("hr") is not None
    finally:
        precision.reset()


def test_the_read_ahead_takes_the_int8_copy_only_for_the_rules_a_load_reads_it_with(tmp_path: Path,
                                                                                    monkeypatch: pytest.MonkeyPatch
                                                                                    ) -> None:
    """The read-ahead holds an int8 copy against the rules the process runs - after its self-check when that ran,
    the install's prediction before - and the kind the convert hook names, as the load does; otherwise it reads
    the file the load will read. It never runs the self-check."""
    import functools

    from pixal3d.modules import int8 as I
    from pixal3d.modules import int8mm

    ckpt = write_checkpoint(tmp_path / "hr", "Int8Flow")
    W.make_copy(ckpt)
    W.make_int8_copy(ckpt, "hr")
    source = Path(f"{ckpt}.safetensors")
    int8_copy, copy = W.int8_copy_of(source), W.copy_of(source)
    loader = L.NetworkLoader("cuda", prefetch="off")        # the choice alone: nothing is read, no card is asked
    for kind in ("hr", "tex"):
        loader.register_checkpoint(kind, ckpt, convert=functools.partial(lambda model, kind: model, kind=kind))

    def ahead(name: str = "hr") -> Path:
        return loader._read_ahead_file(loader._source(name))

    real_available = I.available
    monkeypatch.delenv("PIXAL3D_INT8", raising=False)
    monkeypatch.setattr(I, "available", lambda device: pytest.fail("the read-ahead ran the self-check"))
    monkeypatch.setattr(I, "_state", None)                  # before the self-check: the install's prediction
    assert ahead() == int8_copy and ahead("tex") == copy    # tex: the copy is another kind's
    monkeypatch.setattr(I, "available", real_available)
    monkeypatch.setattr(I, "_state", True)
    monkeypatch.setitem(int8mm._state, "gemm", "triton")
    assert ahead() == int8_copy                             # the process's own rules after its self-check
    monkeypatch.setitem(int8mm._state, "gemm", "intmm")     # the group GEMM failed its check: per-row rules only
    assert ahead() == copy
    monkeypatch.setattr(I, "_state", False)                 # no int8 GEMMs on this card
    assert ahead() == copy
    monkeypatch.setattr(I, "_state", None)
    for mode in ("safe", "cs", "off"):                      # another policy, or none
        monkeypatch.setenv("PIXAL3D_INT8", mode)
        assert ahead() == copy, mode


def test_a_load_that_takes_over_its_read_ahead_keeps_a_newer_request(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """forget goes on with the rest of the request the thread runs only while no newer request (or a stop) came
    since the thread took it: the newest prefetch wins, as without the load."""
    files = {}
    for name in ("A", "X", "B"):
        files[name] = tmp_path / f"{name}.bin"
        files[name].write_bytes(b"x")
    read: list[str] = []
    reading, go_on = threading.Event(), threading.Event()

    def slow(self: Any, file: Path, serial: int, view: memoryview) -> str:
        """A read the thread is inside of (a 16 MiB read from a slow drive): it looks at the serial at its end."""
        read.append(file.stem)
        reading.set()
        go_on.wait(20)
        return "stopped" if self._serial != serial else "done"

    monkeypatch.setattr(L._Prefetcher, "_read_through", slow)

    def started(names: list[str], newer: list[str] | None) -> Any:
        """A prefetcher inside the read of A (`names` its request), `newer` asked for then (None: nothing), and the
        load of A begun."""
        reading.clear()
        go_on.clear()
        read.clear()
        prefetcher = L._Prefetcher()
        prefetcher.read([(name, files[name]) for name in names])
        assert reading.wait(20)
        if newer is not None:
            prefetcher.read([(name, files[name]) for name in newer])
        prefetcher.forget("A")
        return prefetcher

    def finished(prefetcher: Any, expected: list[str]) -> list[str]:
        go_on.set()
        end = time.monotonic() + 20
        while time.monotonic() < end and read != expected:
            time.sleep(0.005)
        time.sleep(0.1)                                     # nothing more comes
        prefetcher.close()
        return list(read)

    prefetcher = started(["A", "X"], ["B"])                 # a newer request waits: it stays
    assert prefetcher._request == [("B", files["B"])]
    assert finished(prefetcher, ["A", "B"]) == ["A", "B"]
    prefetcher = started(["A", "X"], [])                    # a stop came: nothing is revived
    assert prefetcher._request is None
    assert finished(prefetcher, ["A"]) == ["A"]
    prefetcher = started(["A", "X"], ["A"])                 # the newer request was A alone: nothing is left
    assert prefetcher._request is None
    assert finished(prefetcher, ["A"]) == ["A"]
    prefetcher = started(["A", "X"], None)                  # no newer request: the rest of this one goes on
    assert prefetcher._request == [("X", files["X"])]
    assert finished(prefetcher, ["A", "X"]) == ["A", "X"]


def test_a_model_with_a_meta_buffer_no_file_carries_is_never_read_from_an_int8_copy(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a build for real gives such a buffer values, and that build has no int8 layers: the loader reads the
    model-dtypes copy and converts, and the maker writes no int8 copy for such a model (a load from it would have
    quantised uninitialised weights)."""
    from pixal3d.modules import int8 as I

    ckpt = write_checkpoint(tmp_path / "hr", "Int8Flow")
    W.make_copy(ckpt)
    W.make_int8_copy(ckpt, "hr")
    source, rules = Path(f"{ckpt}.safetensors"), I.copy_rules("hr")
    copy = W.int8_copy_of(source)
    assert copy.is_file()
    config = Path(f"{ckpt}.json")
    config.write_text(json.dumps({"name": "MetaBuffered", "args": SIZES["Int8Flow"]}), encoding="utf-8")
    assert L._int8_plan(plan_of(ckpt), source, copy, "hr", rules) is None
    monkeypatch.setattr(L.NetworkLoader, "_int8_copy", lambda self, source: (copy, "hr", rules))
    loader = L.NetworkLoader("cpu", prefetch="off")
    loader.register_checkpoint("hr", ckpt, convert=lambda model: model)
    model = loader.load("hr")
    assert type(model.blocks[0].self_attn.to_qkv) is nn.Linear and torch.equal(model.phases, torch.arange(4.0))
    stored = tensors_of(W.copy_of(source))
    assert raw(model.blocks[0].self_attn.to_qkv.weight) == raw(stored["blocks.0.self_attn.to_qkv.weight"])
    loader.drop_all()
    outcome = W.make_int8_copy(ckpt, "hr")
    assert outcome.state == "not needed" and "meta device" in outcome.why and not copy.exists()


# ---- a pipeline folder and the command -----------------------------------------------------------------------

def pipeline(folder: Path, name: str, models: dict[str, str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps({"name": "Pipeline", "args": {"models": models}}), encoding="utf-8")


def test_a_pipeline_folder_gets_copies_of_what_gains_from_them(tmp_path: Path,
                                                               capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "snapshot"
    write_checkpoint(folder / "ckpts" / "flow", "Flow")
    write_checkpoint(folder / "ckpts" / "dec", "Kept")
    write_checkpoint(folder / "ckpts" / "half", "Half", torch.float16)
    single = {"flow": "ckpts/flow", "dec": "ckpts/dec", "half": "ckpts/half"}
    pipeline(folder, "pipeline.json", single)
    pipeline(folder, "pipeline_mv.json", {"flow": "ckpts/flow_mv", "dec": "ckpts/dec", "half": "ckpts/half"})

    def run() -> tuple[int, list[str]]:
        code = W.main([str(folder)])
        return code, capsys.readouterr().out.splitlines()

    code, lines = run()                          # the multi-view models are not downloaded: left out, silently
    assert code == 0 and [line.split(":")[0] for line in lines] == ["flow", "dec", "half"]
    assert "copy written" in lines[0] and "no copy needed" in lines[1] and "no copy needed" in lines[2]
    assert sorted(p.name for p in (folder / "ckpts").glob(f"*{W.COPY_SUFFIX}")) == [f"flow{W.COPY_SUFFIX}"]

    write_checkpoint(folder / "ckpts" / "flow_mv", "Flow", seed=2)
    code, lines = run()                          # they arrived: copied too, the rest is up to date
    assert code == 0 and [line.split(":")[0] for line in lines] == ["flow", "dec", "half", "flow_mv"]
    assert "copy up to date" in lines[0] and "copy written" in lines[3]

    write_checkpoint(folder / "ckpts" / "broken", "Flow", leave_out=("first.bias",))
    pipeline(folder, "pipeline.json", {**single, "broken": "ckpts/broken"})
    code, lines = run()                          # one that does not fit its model: told, the others still done
    assert code == 1 and len(lines) == 5
    assert lines[3].startswith("broken: no copy made (KeyError: ") and "its original is read" in lines[3]
    assert "copy up to date" in lines[4]

    pipeline(folder, "pipeline.json", {**single, "gone": "ckpts/gone"})
    code, lines = run()
    assert code == 1 and lines[3].startswith("gone: no copy made (FileNotFoundError")
    code, lines = W.main([str(tmp_path / "nowhere")]), capsys.readouterr().out.splitlines()
    assert code == 1 and "its pipeline files cannot be read" in lines[-1]


# ---- stfile's side ---------------------------------------------------------------------------------------------

def test_a_file_written_piece_by_piece(tmp_path: Path) -> None:
    path = tmp_path / "streamed.safetensors"
    a, b = torch.arange(10, dtype=torch.float32), torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    layout = [("b", torch.bfloat16, (2, 3)), ("none", torch.float32, (0, 4)), ("a", torch.float32, (10,))]
    size = stfile.write_streamed(path, layout, [("b", b.reshape(-1)[:5]), ("b", b.reshape(-1)[5:]), ("a", a[:3]),
                                                ("a", a[3:])], metadata={"note": "kept"})
    assert size == path.stat().st_size and not Path(f"{path}.tmp").exists()
    assert list(stfile.read_header(path)) == ["b", "none", "a"]                     # the layout's order
    got = tensors_of(path)
    assert torch.equal(got["a"], a) and torch.equal(got["b"], b) and got["none"].shape == (0, 4)
    assert stfile.read_metadata(path) == {"note": "kept"}
    from safetensors.torch import load_file
    assert torch.equal(load_file(str(path))["a"], a)                                # the library reads it

    for pieces, words in (([("a", a)], "still to come"), ([("b", b.reshape(-1)), ("b", b.reshape(-1)[:1])],
                                                           "more elements"),
                          ([("b", b.reshape(-1))], "ended before a"), ([("b", a[:6])], "dense CPU tensor of"),
                          ([("b", b.reshape(-1)), ("a", a), ("x", a)], "not in the layout")):
        bad = tmp_path / "bad.safetensors"
        with pytest.raises(ValueError, match=words):
            stfile.write_streamed(bad, layout, pieces)
        assert not bad.exists() and not Path(f"{bad}.tmp").exists()
    with pytest.raises(ValueError, match="cannot be a tensor's name"):
        stfile.write_streamed(tmp_path / "twice.safetensors", layout + [("a", torch.float32, (1,))], [])
    plain = tmp_path / "plain.safetensors"
    stfile.write(plain, {"a": a})
    assert stfile.read_metadata(plain) == {}


# ---- the calibration's standing record (pixal3d/calibrate.py) --------------------------------------------------

def test_the_calibration_runs_its_triton_step_again_where_the_record_lacks_the_int8_tile(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A record made before the Triton step chose the int8 GEMMs' fc1 tile does not stand on a card whose int8 GEMMs
    are Triton's: that step runs again (Spellcast3D's installer asks for it alone, --only triton); everywhere else,
    and once the record holds a choice (or a failed timing), it stands."""
    import importlib

    saved = dict(os.environ)                       # calibrate.py points Triton's cache at the venv when imported
    sys.path.insert(0, str(paths.PIXAL3D_DIR))
    try:
        calibrate = importlib.import_module("pixal3d.calibrate")
        from pixal3d.modules import int8mm
    finally:
        sys.path.remove(str(paths.PIXAL3D_DIR))
        os.environ.clear()
        os.environ.update(saved)
    stands = calibrate._triton_stands
    monkeypatch.setattr(int8mm, "capability", lambda device=None: (True, True, ""))        # sm_80+: Triton's GEMMs
    assert not stands({"triton": {"ok": True, "fused": True, "int8": True, "codes": True}}, "cuda")
    assert stands({"triton": {"int8": True, "int8_tiles": {"ok": False, "error": "RuntimeError: x"}}}, "cuda")
    assert stands({"triton": {"int8": "off: the card is sm_75"}}, "cuda") and stands({}, "cuda")
    import triton                                  # a choice made with another Triton is read as none: timed again
    timed = {"fc1_group": [64, 128, 128, 8, 3, 4], "ok": True}
    assert not stands({"triton": {"int8": True, "int8_tiles": {**timed, "triton": "0.0"}}}, "cuda")
    assert not stands({"triton": {"int8": True, "int8_tiles": timed}}, "cuda")
    assert stands({"triton": {"int8": True, "int8_tiles": {**timed, "triton": triton.__version__}}}, "cuda")
    assert stands({"triton": {"int8": True, "int8_tiles": {"ok": False, "triton": "0.0"}}}, "cuda")
    monkeypatch.setattr(int8mm, "capability", lambda device=None: (True, False, "sm_75"))  # no Triton int8 GEMM
    assert stands({"triton": {"int8": True}}, "cuda")
    assert stands({"triton": {"int8": True, "int8_tiles": {**timed, "triton": "0.0"}}}, "cuda")
