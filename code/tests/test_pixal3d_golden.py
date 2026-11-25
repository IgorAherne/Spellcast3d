"""The foundations of the Pixal3D low-memory checks (spellcast/profiling/gpu_lock.py and p3d_checks/common.py):
the GPU lock (one run on the card at a time: a live holder is never broken, a killed one is taken over by the
stale rule), the comparison rules against golden tensors (integers always equal; floats bit-equal unless the
reference's own spread says otherwise), reading a golden case, booting a check on the CPU without GOLD.

No GPU: the lock tests use a lock folder of their own and never ask about the card unless a test replaces the
answer; nothing here imports Pixal3D. The lock tests run in any venv; the others need PyTorch and safetensors and are
skipped without them.
Run:  set PYTHONPATH=code\\logs\\profiling\\pytools
      code\\venvs\\pixal3d\\Scripts\\python.exe -m pytest -p no:cacheprovider code/tests/test_pixal3d_golden.py -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from spellcast.profiling.p3d_checks import common

gpu_lock = sys.modules[common.gpu_lock.__module__]          # profiling/gpu_lock.py, as common loaded it

HAVE_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None
needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="needs PyTorch and safetensors (the pixal3d venv)")

HOLDER = """
import sys, time
from pathlib import Path
sys.path.insert(0, {folder!r})
import gpu_lock
with gpu_lock.gpu_lock("WPX", "holder", expected_minutes=1, lock=Path({lock!r}), check_card=False, heartbeat=0.2):
    print("held", flush=True)
    time.sleep(120)
"""


def quiet(text: str) -> None:
    pass


def start_holder(lock: Path) -> tuple[subprocess.Popen, dict[str, Any]]:
    """A process that holds the lock (its heartbeat every 0.2 s) until it is killed; its owner record."""
    script = HOLDER.format(folder=str(Path(gpu_lock.__file__).parent), lock=str(lock))
    child = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    assert child.stdout.readline().strip() == "held"
    return child, gpu_lock.read_owner(lock)


def kill(child: subprocess.Popen, owner: dict[str, Any]) -> None:
    """End the holder without any clean-up of its own (a venv's python.exe starts the real interpreter as a child:
    the pid in the owner record is the one that holds the lock)."""
    for pid in {int(owner["pid"]), child.pid}:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    child.wait(timeout=10)
    deadline = time.time() + 10
    while gpu_lock.process(int(owner["pid"]))[0] and time.time() < deadline:
        time.sleep(0.05)
    assert not gpu_lock.process(int(owner["pid"]))[0]


def age_heartbeat(lock: Path, seconds: float) -> None:
    """Make the lock's last heartbeat `seconds` old, as if that much time had passed since its holder wrote it."""
    owner = gpu_lock.read_owner(lock)
    owner["heartbeat"] = gpu_lock.iso(time.time() - seconds)
    (lock / gpu_lock.OWNER).write_text(json.dumps(owner), encoding="utf-8")


# ---- the GPU lock ---------------------------------------------------------------------------------------------------

def test_the_lock_is_taken_and_given_back(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    assert gpu_lock.lock_state(lock)["held"] is False
    with gpu_lock.gpu_lock("WP1", "check_cond sv1024", expected_minutes=4, lock=lock, check_card=False) as owner:
        state = gpu_lock.lock_state(lock)
        assert state["held"] and state["alive"] and not state["stale"]
        on_disk = json.loads((lock / "owner.json").read_text(encoding="utf-8"))
        assert {"wp", "pid", "started", "heartbeat", "purpose", "expected_minutes"} <= set(on_disk)
        assert (on_disk["wp"], on_disk["pid"], on_disk["purpose"], on_disk["expected_minutes"]) == (
            "WP1", os.getpid(), "check_cond sv1024", 4)
        assert gpu_lock.seconds_of(on_disk["started"]) == pytest.approx(time.time(), abs=5)
        assert owner["pid"] == os.getpid()
        with gpu_lock.gpu_lock("WP1", "a block inside the block", lock=lock, check_card=False, timeout=0.5):
            assert gpu_lock.read_owner(lock)["purpose"] == "check_cond sv1024"      # the outer block's lock
        assert lock.is_dir()
    assert not lock.exists()
    assert list(tmp_path.iterdir()) == []


def test_an_exception_inside_the_block_gives_the_lock_back(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    with pytest.raises(ZeroDivisionError):
        with gpu_lock.gpu_lock("WP1", lock=lock, check_card=False):
            1 / 0
    assert not lock.exists()


def test_a_live_holder_is_refused_whatever_its_heartbeat_says(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    child, owner = start_holder(lock)
    try:
        time.sleep(0.6)
        assert gpu_lock.read_owner(lock)["heartbeat"] >= owner["heartbeat"]         # the holder's thread rewrites it
        with pytest.raises(gpu_lock.LockTimeout, match="held by WPX"):
            gpu_lock.acquire("WP2", "intruder", lock=lock, poll=0.05, timeout=0.3, say=quiet)
        # even a rule that calls every heartbeat too old does not break a lock whose process runs
        with pytest.raises(gpu_lock.LockTimeout):
            gpu_lock.acquire("WP2", "intruder", lock=lock, poll=0.05, timeout=0.3, stale_seconds=0.0, say=quiet)
        assert gpu_lock.read_owner(lock)["wp"] == "WPX"
        assert not list(tmp_path.glob("gpu.lock.stale-*"))
    finally:
        kill(child, owner)


def test_a_killed_holder_is_taken_over_once_its_heartbeat_is_ten_minutes_old(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    child, owner = start_holder(lock)
    kill(child, owner)
    assert lock.is_dir()                                    # a killed process cleans nothing up
    state = gpu_lock.lock_state(lock)
    assert state["held"] and state["alive"] is False and not state["stale"]
    with pytest.raises(gpu_lock.LockTimeout, match="NOT running"):                  # heartbeat seconds old: wait
        gpu_lock.acquire("WP2", "too early", lock=lock, poll=0.05, timeout=0.3, say=quiet)

    age_heartbeat(lock, gpu_lock.STALE_SECONDS - 30)
    assert not gpu_lock.lock_state(lock)["stale"]
    age_heartbeat(lock, gpu_lock.STALE_SECONDS + 30)
    assert gpu_lock.lock_state(lock)["stale"]
    with gpu_lock.gpu_lock("WP2", "takeover", lock=lock, check_card=False, poll=0.05, timeout=5) as mine:
        assert gpu_lock.read_owner(lock)["wp"] == "WP2" and mine["pid"] == os.getpid()
        assert not list(tmp_path.glob("gpu.lock.stale-*"))                         # moved aside, then deleted
    assert not lock.exists()


def test_stale_locks_left_aside_by_earlier_runs_are_deleted_and_fresh_ones_kept(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    old = time.time() - gpu_lock.STALE_SECONDS - 60
    stale = tmp_path / "gpu.lock.stale-20261012-125150-64208"             # an earlier version left these
    stale.mkdir()
    (stale / "owner.json").write_text(json.dumps({"wp": "WP1", "pid": child.pid, "heartbeat": gpu_lock.iso(old)}),
                                      encoding="utf-8")
    fresh = tmp_path / "gpu.lock.stale-20261013-090000-1"                 # a live lock moved aside this instant
    fresh.mkdir()
    (fresh / "owner.json").write_text(json.dumps({"wp": "WP2", "pid": os.getpid(), "heartbeat": gpu_lock.iso()}),
                                      encoding="utf-8")
    record = gpu_lock.acquire("WP3", lock=lock, poll=0.05, timeout=5, say=quiet)
    gpu_lock.release(lock, record)
    if not gpu_lock.process(child.pid)[0]:                  # (the pid could have been given out again already)
        assert not stale.exists()
    assert fresh.is_dir()


def test_a_lock_made_by_hand_is_held_while_its_process_lives(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    lock.mkdir()
    by_hand = {"wp": "WP0", "pid": os.getpid(), "started": "2026-10-08 01:00:00", "purpose": "golden capture",
               "expected_minutes": 25}                      # no heartbeat, no thread that would write one
    (lock / "owner.json").write_text(json.dumps(by_hand), encoding="utf-8")
    state = gpu_lock.lock_state(lock)
    assert state["held"] and state["alive"] and not state["stale"] and state["heartbeat_age"] > 600
    with pytest.raises(gpu_lock.LockTimeout, match="held by WP0"):
        gpu_lock.acquire("WP2", lock=lock, poll=0.05, timeout=0.3, say=quiet)

    # a heartbeat that is missing AND a process that is gone: stale
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    time.sleep(0.2)
    (lock / "owner.json").write_text(json.dumps({**by_hand, "pid": child.pid}), encoding="utf-8")
    if not gpu_lock.process(child.pid)[0]:                  # (the pid could have been given out again already)
        assert gpu_lock.lock_state(lock)["stale"]


def test_a_pid_given_to_another_process_does_not_keep_a_lock(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    lock.mkdir()
    created = gpu_lock.process(os.getpid())[1]
    record = {"wp": "WP3", "pid": os.getpid(), "pid_created": created, "started": gpu_lock.iso(time.time() - 3600),
              "heartbeat": gpu_lock.iso(time.time() - 3600)}
    (lock / "owner.json").write_text(json.dumps(record), encoding="utf-8")
    assert gpu_lock.lock_state(lock)["alive"]
    if created is not None:                                 # Windows: the record names the process, not only the pid
        (lock / "owner.json").write_text(json.dumps({**record, "pid_created": created - 1}), encoding="utf-8")
        state = gpu_lock.lock_state(lock)
        assert state["alive"] is False and state["stale"]


def test_a_folder_without_an_owner_counts_as_held_for_ten_minutes(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    lock.mkdir()
    state = gpu_lock.lock_state(lock)
    assert state["held"] and state["owner"] is None and not state["stale"]
    old = time.time() - gpu_lock.STALE_SECONDS - 60
    os.utime(lock, (old, old))
    assert gpu_lock.lock_state(lock)["stale"]
    record = gpu_lock.acquire("WP2", lock=lock, poll=0.05, timeout=5, say=quiet)
    gpu_lock.release(lock, record)
    assert not lock.exists()


def test_release_leaves_a_lock_that_is_not_ours(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    lock = tmp_path / "gpu.lock"
    record = gpu_lock.acquire("WP1", lock=lock, say=quiet)
    other = {**record, "wp": "WP9", "started": gpu_lock.iso(time.time() + 60)}
    (lock / "owner.json").write_text(json.dumps(other), encoding="utf-8")
    gpu_lock.release(lock, record)
    assert gpu_lock.read_owner(lock)["wp"] == "WP9"
    assert "no longer this run's" in capsys.readouterr().out


def test_a_busy_card_makes_the_holder_wait_and_give_the_lock_back(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tmp_path / "gpu.lock"
    answers = [{"used_mib": 5061, "total_mib": 12288, "processes": ["49684, python.exe"]}]
    monkeypatch.setattr(gpu_lock, "card", lambda: answers[0])
    monkeypatch.setattr(gpu_lock, "own_mib", lambda: 0)     # an earlier test may have started CUDA in this process
    with pytest.raises(gpu_lock.CardBusy, match="5061 MiB"):
        with gpu_lock.gpu_lock("WP1", lock=lock, poll=0.05, card_timeout=0.2):
            pytest.fail("the block ran beside another program on the card")
    assert not lock.exists()

    answers[0] = {"used_mib": 338, "total_mib": 12288, "processes": []}             # the idle desktop
    with gpu_lock.gpu_lock("WP1", lock=lock, poll=0.05, card_timeout=0.2):
        assert lock.is_dir()
    answers[0] = {"error": "FileNotFoundError: nvidia-smi"}                         # cannot ask: do not block
    with gpu_lock.gpu_lock("WP1", lock=lock, poll=0.05, card_timeout=0.2):
        assert lock.is_dir()


def test_status_says_who_holds_the_lock(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tmp_path / "gpu.lock"
    monkeypatch.setattr(gpu_lock, "card", lambda: {"used_mib": 338, "total_mib": 12288, "processes": []})
    assert gpu_lock.main(["status", "--lock", str(lock)]) == 0
    assert "free" in capsys.readouterr().out
    with gpu_lock.gpu_lock("WP4", "check_loader", expected_minutes=2, lock=lock, check_card=False):
        assert gpu_lock.main(["status", "--lock", str(lock)]) == 3
    text = capsys.readouterr().out
    assert "held by WP4" in text and "check_loader" in text and "338 of 12288 MiB" in text


# ---- comparison with golden tensors ---------------------------------------------------------------------------------

@needs_torch
def test_integers_and_booleans_must_be_equal_always() -> None:
    import torch

    coords = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]], dtype=torch.int32)
    assert common.compare("06-upsample/coords", coords.clone(), coords, {"equal": False, "max_abs": 5.0}).ok
    moved = coords.clone()
    moved[1, 2] = 6
    result = common.compare("06-upsample/coords", moved, coords, {"equal": False, "max_abs": 5.0})
    assert not result.ok and result.rule == "integers"      # a spread never excuses an integer
    assert result.differing == 1 and result.first == [1, 2] and result.max_abs == 1.0
    assert "got 6, golden 5" in result.detail

    reordered = coords[[1, 0, 2]]                           # the same rows in another order are another tensor
    assert not common.compare("06-upsample/coords", reordered, coords).ok
    subs = torch.tensor([[True, False], [False, True]])
    assert common.compare("11-decode/sub_0", subs.clone(), subs).ok
    assert not common.compare("11-decode/sub_0", ~subs, subs).ok


@needs_torch
def test_floats_must_have_the_same_bits_where_the_reference_reproduces_them() -> None:
    import torch

    want = torch.linspace(-3, 3, 96, dtype=torch.float32).reshape(4, 24)
    got = want.clone()
    for spread in (None, {}, {"08-hr/slat": {"equal": True, "max_abs": 0.0}}, {"equal": True}):
        result = common.compare("08-hr/slat", got, want, spread)
        assert result.ok and result.rule == "bit-equal" and result.max_abs == 0.0

    got[2, 5] = torch.nextafter(got[2, 5], torch.tensor(10.0))                      # one value, one ulp
    for spread in (None, {}, {"08-hr/slat": {"equal": True, "max_abs": 0.0}}, {"equal": True, "max_abs": 1.0}):
        result = common.compare("08-hr/slat", got, want, spread)
        assert not result.ok and result.rule == "bit-equal"                         # whatever the size of it
        assert result.differing == 1 and result.first == [2, 5] and 0 < result.max_abs < 1e-6
    assert "bit for bit" in common.compare("08-hr/slat", got, want, {"equal": True}).detail
    assert "no spread recorded" in common.compare("08-hr/slat", got, want, {}).detail


@needs_torch
def test_floats_the_reference_does_not_reproduce_are_held_to_its_own_spread() -> None:
    import torch

    want = torch.linspace(0, 1, 60, dtype=torch.float32).reshape(20, 3)
    got = want.clone()
    got[7, 1] += 3e-4
    spread = {"11-decode/vertices_raw": {"equal": False, "max_abs": 5e-4},
              "11-decode/tex_attrs": {"equal": False, "max_abs": 1e-4}}
    inside = common.compare("11-decode/vertices_raw", got, want, spread)
    assert inside.ok and inside.rule == "spread" and inside.allowed == 5e-4
    assert inside.max_abs == pytest.approx(3e-4, rel=1e-2) and inside.differing == 1 and inside.first == [7, 1]
    outside = common.compare("11-decode/tex_attrs", got, want, spread)
    assert not outside.ok and outside.rule == "spread" and ">" in outside.detail
    assert common.compare("11-decode/vertices_raw", want.clone(), want, spread).rule == "bit-equal"

    assert common.compare("x/y", got, want, 5e-4).ok        # a number is the max_abs
    assert not common.compare("x/y", got, want, 0.0).ok
    assert not common.compare("x/y", got, want, {"equal": False}).ok                # no max_abs to judge by
    broken = want.clone()
    broken[0, 0] = float("nan")
    assert not common.compare("11-decode/vertices_raw", broken, want, spread).ok    # a NaN is never inside a spread


@needs_torch
def test_another_dtype_or_shape_than_the_golden_tensor_fails() -> None:
    import torch

    want = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    half = common.compare("04-cond_lr/proj", want.to(torch.bfloat16), want)
    assert not half.ok and half.rule == "form" and "bfloat16" in half.detail
    assert common.compare("04-cond_lr/proj", want.to(torch.bfloat16), want.to(torch.bfloat16)).ok
    assert common.compare("04-cond_lr/proj", want.reshape(4, 3), want).rule == "form"
    assert common.compare("04-cond_lr/proj", want[:2], want).rule == "form"
    assert not common.compare("03-ss/coords", want.to(torch.int64), want.to(torch.int32)).ok

    low = want.to(torch.bfloat16)
    other = low.clone()
    other[1, 1] = 100.0
    result = common.compare("04-cond_lr/proj", other, low)
    assert not result.ok and result.first == [1, 1] and result.max_abs == pytest.approx(95.0)


def test_the_spread_of_a_tensor_is_found_in_the_layouts_a_spread_file_may_have() -> None:
    entry = {"equal": False, "max_abs": 2.5e-7}
    assert common.spread_entry(None, "08-hr/slat") is None
    assert common.spread_entry({}, "08-hr/slat") is None
    assert common.spread_entry(entry, "08-hr/slat") is entry
    assert common.spread_entry({"08-hr/slat": entry}, "08-hr/slat") is entry
    assert common.spread_entry({"08-hr": {"slat": entry}}, "08-hr/slat") is entry
    assert common.spread_entry({"08-hr.safetensors": {"slat": entry}}, "08-hr/slat") is entry
    assert common.spread_entry({"tensors": {"08-hr/slat": entry}}, "08-hr/slat") is entry
    assert common.spread_entry({"08-hr/noise": entry}, "08-hr/slat") is None
    assert common.spread_entry(0, "x/y") == {"equal": True, "max_abs": 0.0}
    with pytest.raises(TypeError):
        common.spread_entry("loose", "x/y")


@needs_torch
def test_report_is_the_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    import torch

    want = torch.ones(3)
    good = common.compare("a/b", want.clone(), want)
    bad = common.compare("a/c", want * 2, want)
    assert common.report([good]) == 0
    assert "PASSED: all 1 comparisons" in capsys.readouterr().out
    assert common.report([good, bad]) == 1
    text = capsys.readouterr().out
    assert "FAIL  a/c" in text and "FAILED: 1 of 2" in text and "first: a/c" in text
    assert common.report([]) == 1                           # a check that compared nothing has proved nothing


# ---- reading a golden case ------------------------------------------------------------------------------------------

@needs_torch
def test_a_golden_case_is_read_tensor_by_tensor(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(5)
    noise = torch.randn(7, 32, generator=generator)
    slat = torch.randn(7, 32, generator=generator)
    coords = torch.randint(0, 64, (7, 4), dtype=torch.int32, generator=generator)
    half = torch.randn(5, 8, generator=generator).to(torch.bfloat16)
    case_dir = tmp_path / "tiny"
    case_dir.mkdir()
    save_file({"noise": noise, "slat": slat, "half": half}, str(case_dir / "08-hr.safetensors"))
    save_file({"coords": coords}, str(case_dir / "06-upsample.safetensors"))
    manifest = {"seed": 11, "resolution": 1024, "N": 7,
                "sha256": {"08-hr/noise": common.sha256_of(noise), "08-hr/slat": common.sha256_of(slat),
                           "08-hr/half": common.sha256_of(half), "06-upsample/coords": common.sha256_of(coords)}}
    (case_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (case_dir / "request.json").write_text(json.dumps({"image_path": "input/a.png", "seed": 11}), encoding="utf-8")
    (case_dir / "spread.json").write_text(json.dumps({"08-hr/slat": {"equal": True, "max_abs": 0.0}}),
                                          encoding="utf-8")

    case = common.golden("tiny", root=tmp_path)
    assert case.name == "tiny" and case.manifest["seed"] == 11 and case.request["seed"] == 11
    assert case.files() == ["06-upsample", "08-hr"]
    assert case.names("08-hr") == {"noise": ("F32", [7, 32]), "slat": ("F32", [7, 32]), "half": ("BF16", [5, 8])}
    assert torch.equal(case.tensor("08-hr", "noise"), noise)
    assert torch.equal(case.tensor("08-hr.safetensors", "slat"), slat)
    read = case.tensor("08-hr", "half")
    assert read.dtype == torch.bfloat16 and torch.equal(read, half)
    assert case.tensor("06-upsample", "coords").dtype == torch.int32
    assert set(case.tensors("08-hr")) == {"noise", "slat", "half"}
    assert case.input("input/a.png") == case_dir / "input" / "a.png"
    assert common.compare("08-hr/slat", slat.clone(), case.tensor("08-hr", "slat"), case.spread).ok
    assert case.verify() == []

    with pytest.raises(FileNotFoundError, match="has no stage file '09-tex'"):
        case.tensor("09-tex", "noise")
    with pytest.raises(KeyError, match="has no tensor 'x_steps'"):
        case.tensor("08-hr", "x_steps")
    with pytest.raises(FileNotFoundError, match="not captured yet"):
        common.golden("sv4096", root=tmp_path)

    # a tensor that is not the one the manifest recorded
    save_file({"coords": coords + 1}, str(case_dir / "06-upsample.safetensors"))
    assert common.golden("tiny", root=tmp_path).verify() == [
        "06-upsample/coords: the file's tensor does not have the manifest's SHA-256"]
    # hashes kept per file, per tensor, with the shape beside them
    manifest["sha256"] = None
    manifest["tensors"] = {"08-hr": {"noise": {"sha256": common.sha256_of(noise), "shape": [7, 32]}}}
    (case_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    again = common.golden("tiny", root=tmp_path)
    assert again.recorded_sha("08-hr", "noise") == common.sha256_of(noise)
    assert again.verify() == ["3 tensors have no SHA-256 in the manifest"]


@needs_torch
def test_hashes_are_of_the_values_alone() -> None:
    import hashlib

    import torch

    values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    plain = hashlib.sha256(values.numpy().tobytes()).hexdigest()
    assert common.sha256_of(values) == plain
    assert common.sha256_of(values.reshape(6, 4)) == plain                          # no shape in the hash
    assert common.sha256_of(values.permute(2, 0, 1)) != plain                       # but row-major order of the VIEW
    assert common.sha256_of(values.permute(2, 0, 1)) == hashlib.sha256(
        values.permute(2, 0, 1).contiguous().numpy().tobytes()).hexdigest()
    low = values.to(torch.bfloat16)
    assert common.sha256_of(low) == hashlib.sha256(low.view(torch.int16).numpy().tobytes()).hexdigest()
    assert common.sha256_of(torch.empty(0, 3)) == hashlib.sha256(b"").hexdigest()

    layer = torch.nn.Linear(3, 2)
    twin = torch.nn.Linear(3, 2)
    twin.load_state_dict(layer.state_dict())
    assert common.state_digest(layer) == common.state_digest(twin)
    assert common.state_digest(layer) != common.state_digest(twin.half())           # the dtype is part of it
    assert set(common.state_digest(layer)) == {"weight", "bias"}


def test_boot_refuses_what_is_not_a_worktree_with_models(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a Pixal3D checkout"):
        common.boot(tmp_path)
    (tmp_path / "app.py").write_text("", encoding="utf-8")
    (tmp_path / "pixal3d").mkdir()
    with pytest.raises(FileNotFoundError, match="MODELS junction"):
        common.boot(tmp_path)


@needs_torch
def test_a_check_on_the_cpu_boots_without_models_and_without_gold(tmp_path: Path) -> None:
    """boot(golden=False) (check_store.py on the CPU) asks for neither the MODELS junction nor the golden folder:
    with P3D_GOLD pointing at an empty folder and no frozen launcher, it takes the live launcher's settings."""
    repo = tmp_path / "fork"
    (repo / "pixal3d").mkdir(parents=True)
    (repo / "app.py").write_text("", encoding="utf-8")
    script = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(Path(common.__file__).parent)!r})\n"
        "import common\n"
        "common.LAUNCHER = common.LAUNCHER.with_name('no-launcher-here')\n"
        f"common.boot({str(repo)!r}, golden=False)\n"
        "print('booted', os.getcwd(), os.environ['HF_HOME'], os.environ['MIMALLOC_PURGE_DELAY'], "
        "common.launcher().__file__)\n")
    env = {key: value for key, value in os.environ.items() if key != "MIMALLOC_PURGE_DELAY"}
    env.update(P3D_GOLD=str(tmp_path / "no-gold"), CUDA_VISIBLE_DEVICES="-1")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    line = [text for text in done.stdout.splitlines() if text.startswith("booted")][0]
    assert str(repo) in line and str(repo / "models") in line and " 0 " in line
    assert str(Path(common.LIVE_LAUNCHERS) / "pixal3d_server.py") in line
    assert "no frozen launcher" in done.stdout
