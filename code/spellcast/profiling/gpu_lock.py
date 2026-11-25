"""One process on the video card at a time: the lock the GPU checks of the Pixal3D low-memory work take.

A check that runs beside another program on the card measures nothing (a failure under contention is not
evidence), so a run that touches CUDA holds the folder code/logs/profiling/gpu.lock for exactly as long as it is on
the card.

    in Python    import common                             (p3d_checks/common.py: common.gpu_lock is this one)
                 with common.gpu_lock("WP1", "check_x sv1024", expected_minutes=4):
                     ...                                   # everything that touches CUDA; take it BEFORE the first
                                                           # CUDA call of the process
    who has it   python gpu_lock.py status                 exit code 0 = free, 3 = held

How it works:
- The folder is made with os.mkdir, which either succeeds or fails as one step, so two runs can never both get it.
  Inside is owner.json: {"wp", "pid", "started", "heartbeat", "purpose", "expected_minutes"} (ISO times), rewritten
  every 60 s by a thread of the holder. A waiting run tries every second and prints who holds the lock every 20 s.
- Waiting runs queue first come, first served: each writes a ticket (its name orders them by the moment they asked)
  into gpu.lock.queue/ and only the oldest live ticket tries the mkdir, so a run cannot be starved by others that
  keep winning the race (round J: an agent waited 73 minutes). A ticket counts while its process is alive and its
  heartbeat (rewritten every 5 s by the waiting run) is under a minute old; the others are deleted on the way.
  `status` lists the queue.
- A lock is stale, and is then moved aside, deleted and taken, only when its heartbeat is older than 10 minutes
  AND its process is gone. A lock whose process is alive is never broken, whatever its heartbeat says (a lock folder
  made by hand has no heartbeat thread at all). A folder without a readable owner file counts as held for 10 minutes
  after it was made.
- With the lock in hand the helper asks nvidia-smi how much video memory is in use. More than 1.5 GiB held by
  others (a game, a 3D program, a run that did not take the lock) means: wait, and say so. The mirror rule: a
  process that asks for the lock while it still holds more than 512 MiB of live tensors is refused
  (HoldingVideoMemory), since every holder would wait for it to leave the card while it waits for the lock
  (round J's deadlock); free the model and the tables before releasing the lock, or hold it once per script. Others' share is the
  whole card's use minus what this process holds by PyTorch's count (plus its CUDA context): Windows does count
  video memory per process (mem_watch.py reads those counters), but the card's total is what also shows the
  desktop and the programs that never asked for the lock.

A holder that is killed (a tool's time limit, Task Manager) cannot clean up: its lock is taken over 10 minutes
after its last heartbeat.
"""

from __future__ import annotations

import contextlib
import ctypes
import datetime
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:          # toolkit.py lies beside this file, which p3d_checks/common.py loads by its path
    sys.path.append(str(HERE))

import toolkit  # noqa: E402  (stdlib only)

LOCK = HERE.parents[1] / "logs" / "profiling" / "gpu.lock"
OWNER = "owner.json"
POLL_SECONDS = 20.0                     # a waiting run says who holds the lock this often
TRY_SECONDS = 1.0                       # and tries to take it this often: a holder that takes it again within
#                                         seconds (one capture after another) must not starve the others
HEARTBEAT_SECONDS = 60.0
STALE_SECONDS = 600.0
QUEUE_SUFFIX = ".queue"                 # the folder of the tickets of the runs waiting for the lock (see above)
TICKET_BEAT_SECONDS = 5.0               # a waiting run rewrites its ticket's heartbeat this often
TICKET_STALE_SECONDS = 60.0             # a ticket without a heartbeat for this long is not waiting any more
OTHER_VRAM_MIB = 1536
SELF_VRAM_MIB = 512                     # live tensors a process may still hold when it asks for the lock (below)
CUDA_CONTEXT_MIB = 350                  # what a process holds on the card beside PyTorch's reserve
STILL_ACTIVE = 259
ACCESS_DENIED = 5

_depth_guard = threading.Lock()
_write_guard = threading.Lock()         # the heartbeat thread rewrites the owner record while the block runs
_depth: dict[str, int] = {}             # lock folder -> how many `with gpu_lock` blocks of this process are open

tell = toolkit.teller("gpu-lock")
card = toolkit.card


class LockTimeout(RuntimeError):
    """The lock stayed with another run for longer than the caller was willing to wait."""


class CardBusy(RuntimeError):
    """Other programs kept more video memory than allowed for longer than the caller was willing to wait."""


def iso(when: float | None = None) -> str:
    stamp = datetime.datetime.fromtimestamp(time.time() if when is None else when).astimezone()
    return stamp.isoformat(timespec="seconds")


def seconds_of(text: Any) -> float | None:
    """An ISO time as seconds since the epoch; a time without a zone is local time. None when it is not a time."""
    try:
        when = datetime.datetime.fromisoformat(str(text).strip())
    except (TypeError, ValueError):
        return None
    return when.astimezone().timestamp()


def process(pid: int) -> tuple[bool, int | None]:
    """(is this process running, the moment it was created as the system counts it or None when unknown)."""
    if pid <= 0:
        return False, None
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except PermissionError:
            return True, None
        except OSError:
            return False, None
        # Linux: a zombie has ended (a pod whose first process never reaps them keeps its pid for good); the
        # creation moment is the start time in clock ticks after boot, which tells a reused pid apart
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        except OSError:
            return True, None
        return fields[0] not in ("Z", "X"), int(fields[19])
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    k32.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = k32.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ctypes.get_last_error() == ACCESS_DENIED, None   # it exists, under another account
    try:
        code = wintypes.DWORD()
        running = bool(k32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == STILL_ACTIVE
        times = [wintypes.FILETIME() for _ in range(4)]
        created = None
        if k32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
            created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        return running, created
    finally:
        k32.CloseHandle(handle)


def read_owner(lock: Path) -> dict[str, Any] | None:
    """The lock's owner record, or None when the folder has no readable one (yet)."""
    for _ in range(3):                                      # a writer may be replacing the file this instant
        try:
            record = json.loads((lock / OWNER).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            time.sleep(0.05)
            continue
        return record if isinstance(record, dict) else None
    return None


def owner_alive(owner: dict[str, Any]) -> bool:
    """Is the process that took the lock still running. A pid that Windows has given to a newer process does not
    count: the record carries the moment its own process was created."""
    try:
        pid = int(owner.get("pid", 0))
    except (TypeError, ValueError):
        pid = 0
    running, created = process(pid)
    recorded = owner.get("pid_created")
    return running and (recorded is None or created is None or int(recorded) == created)


def lock_state(lock: Path = LOCK, stale_seconds: float = STALE_SECONDS) -> dict[str, Any]:
    """The lock as it is now: {"held", "owner", "alive", "heartbeat_age", "stale"}."""
    try:
        made = lock.stat().st_mtime
    except OSError:
        return {"held": False, "owner": None, "alive": None, "heartbeat_age": None, "stale": False}
    owner = read_owner(lock)
    now = time.time()
    if owner is None:
        if not lock.exists():
            return {"held": False, "owner": None, "alive": None, "heartbeat_age": None, "stale": False}
        age = now - made
        return {"held": True, "owner": None, "alive": None, "heartbeat_age": age, "stale": age > stale_seconds}
    beat = seconds_of(owner.get("heartbeat"))
    if beat is None:
        beat = seconds_of(owner.get("started"))
    if beat is None:
        beat = made
    age = now - beat
    alive = owner_alive(owner)
    return {"held": True, "owner": owner, "alive": alive, "heartbeat_age": age,
            "stale": age > stale_seconds and not alive}


def describe(state: dict[str, Any]) -> str:
    if not state["held"]:
        return "free"
    owner = state["owner"]
    if owner is None:
        return f"held by a folder without an owner file, made {state['heartbeat_age']:.0f} s ago"
    text = (f"held by {owner.get('wp') or '?'} (pid {owner.get('pid')}, "
            f"{'running' if state['alive'] else 'NOT running'}) since {owner.get('started', '?')}, heartbeat "
            f"{state['heartbeat_age']:.0f} s ago, expected {owner.get('expected_minutes', '?')} min")
    return text + (f": {owner['purpose']}" if owner.get("purpose") else "")


def write_owner(lock: Path, record: dict[str, Any]) -> None:
    """Replace owner.json in one step: a reader sees the old record or the new one, never half of one."""
    part = lock / f"{OWNER}.{os.getpid()}.part"
    with _write_guard:
        part.write_text(json.dumps(record), encoding="utf-8")
        for attempt in range(20):
            try:
                os.replace(part, lock / OWNER)
                return
            except PermissionError:                         # another run is reading it this instant (Windows)
                if attempt == 19:
                    raise
                time.sleep(0.05)


def break_stale(lock: Path, state: dict[str, Any]) -> bool:
    """Move a stale lock aside (gpu.lock.stale-<time>) and delete it. False when another run got there first.

    Moved first, deleted after: the rename is one step, so of two runs that both found the lock stale only one
    gets it, and the one that finds it moved a fresh lock by mistake (a new one came in between) puts it back."""
    aside = lock.with_name(f"{lock.name}.stale-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}")
    try:
        os.rename(lock, aside)
    except OSError:
        return False
    moved = read_owner(aside)
    judged = state["owner"]
    same = (moved is None and judged is None) or (
        moved is not None and judged is not None
        and (moved.get("pid"), moved.get("started")) == (judged.get("pid"), judged.get("started")))
    if not same:                                            # the stale lock went and a new one came in between
        try:
            os.rename(aside, lock)
        except OSError:
            tell(f"WARNING: moved a fresh lock aside by mistake and could not put it back: {aside}")
        return False
    shutil.rmtree(aside, ignore_errors=True)
    tell(f"A stale lock was taken away ({describe(state)}).")
    return True


def remove_stale_asides(lock: Path, stale_seconds: float = STALE_SECONDS) -> None:
    """Delete the gpu.lock.stale-* folders earlier versions of this file left beside the lock. Only those that are
    stale themselves: a fresh lock another run moved aside by mistake this instant is on its way back."""
    for aside in lock.parent.glob(f"{lock.name}.stale-*"):
        if aside.is_dir() and lock_state(aside, stale_seconds)["stale"]:
            shutil.rmtree(aside, ignore_errors=True)


def queue_dir(lock: Path) -> Path:
    return lock.with_name(lock.name + QUEUE_SUFFIX)


def _write_json(path: Path, record: dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    os.replace(tmp, path)


def take_ticket(lock: Path, wp: str, purpose: str) -> Path:
    """A ticket in the lock's queue: its name orders the waiting runs by the moment they asked."""
    queue = queue_dir(lock)
    queue.mkdir(parents=True, exist_ok=True)
    now = time.time()
    ticket = queue / f"{time.time_ns():020d}-{os.getpid()}.json"
    _write_json(ticket, {"wp": wp, "pid": os.getpid(), "pid_created": process(os.getpid())[1], "purpose": purpose,
                         "created": iso(now), "heartbeat": iso(now)})
    return ticket


def refresh_ticket(ticket: Path) -> None:
    try:
        record = json.loads(ticket.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    record["heartbeat"] = iso()
    with contextlib.suppress(OSError):
        _write_json(ticket, record)


def drop_ticket(ticket: Path) -> None:
    with contextlib.suppress(OSError):
        ticket.unlink()
    with contextlib.suppress(OSError):                      # the folder goes with the last ticket
        os.rmdir(ticket.parent)


def queue_ahead(lock: Path, mine: Path | None = None,
                stale_seconds: float = TICKET_STALE_SECONDS) -> list[dict[str, Any]]:
    """The tickets of the runs that asked for the lock before `mine` (all of them when None) and still wait: their
    process alive (the same one), their heartbeat fresh. A ticket of a run that is gone or stuck is deleted on the
    way; one being written this instant counts."""
    try:
        tickets = sorted(path for path in queue_dir(lock).iterdir() if path.suffix == ".json")
    except OSError:
        return []
    ahead: list[dict[str, Any]] = []
    for ticket in tickets:
        if mine is not None and ticket.name >= mine.name:
            break
        try:
            record = json.loads(ticket.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            try:
                young = time.time() - ticket.stat().st_mtime < 5
            except OSError:
                continue
            if young:
                ahead.append({"wp": "?", "pid": None, "purpose": "(a ticket being written)", "created": "?"})
            else:
                with contextlib.suppress(OSError):
                    ticket.unlink()
            continue
        running, created = process(int(record.get("pid") or 0))
        beat = seconds_of(record.get("heartbeat"))
        fresh = beat is not None and time.time() - beat < stale_seconds
        same = record.get("pid_created") in (None, created)
        if running and same and fresh:
            ahead.append(record)
        else:
            with contextlib.suppress(OSError):
                ticket.unlink()
    return ahead


def acquire(wp: str, purpose: str = "", expected_minutes: float = 5, *, lock: Path = LOCK,
            poll: float = POLL_SECONDS, timeout: float | None = None, stale_seconds: float = STALE_SECONDS,
            say: Callable[[str], None] = tell) -> dict[str, Any]:
    """Take the lock, waiting for as long as another run holds it (LockTimeout after `timeout` seconds, when
    given). Returns the owner record written."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    began = time.time()
    told = None
    ticket = take_ticket(lock, wp, purpose)
    beat = time.time()
    try:
        while True:
            if time.time() - beat >= TICKET_BEAT_SECONDS:
                refresh_ticket(ticket)
                beat = time.time()
            ahead = queue_ahead(lock, ticket)               # runs that asked earlier take the lock first
            if not ahead:
                try:
                    os.mkdir(lock)
                    break
                except FileExistsError:
                    pass
                except PermissionError:                     # Windows, while the previous folder is being removed
                    time.sleep(0.2)
                    continue
            state = lock_state(lock, stale_seconds)
            if not ahead and (not state["held"] or (state["stale"] and break_stale(lock, state))):
                continue
            waited = time.time() - began
            if timeout is not None and waited >= timeout:
                raise LockTimeout(f"the GPU lock is {describe(state)}; {len(ahead)} waiting ahead")
            if told is None or time.time() - told >= poll:
                told = time.time()
                queued = f"; {len(ahead)} waiting ahead of this run" if ahead else ""
                say(f"The GPU lock is {describe(state)}{queued}. Waiting ({waited:.0f} s so far).")
            pause = min(poll, TRY_SECONDS)
            time.sleep(pause if timeout is None else max(0.01, min(pause, timeout - waited)))
    finally:
        drop_ticket(ticket)
    now = time.time()
    record = {"wp": wp, "pid": os.getpid(), "started": iso(now), "heartbeat": iso(now), "purpose": purpose,
              "expected_minutes": expected_minutes, "pid_created": process(os.getpid())[1]}
    write_owner(lock, record)
    remove_stale_asides(lock, stale_seconds)
    return record


def ours(lock: Path, record: dict[str, Any]) -> bool:
    owner = read_owner(lock)
    return owner is not None and (owner.get("pid"), owner.get("started")) == (record["pid"], record["started"])


def release(lock: Path, record: dict[str, Any]) -> None:
    """Give the lock back: the folder is renamed first (the lock is free from that instant), then deleted."""
    if not lock.exists():
        return
    if not ours(lock, record):
        tell(f"WARNING: the lock folder is no longer this run's ({describe(lock_state(lock))}); it is left alone.")
        return
    gone = lock.with_name(f"{lock.name}.released-{os.getpid()}-{time.time_ns()}")
    for _ in range(40):
        try:
            os.rename(lock, gone)
            shutil.rmtree(gone, ignore_errors=True)
            return
        except FileNotFoundError:
            return
        except OSError:                                     # a waiting run is reading owner.json this instant
            time.sleep(0.05)
    shutil.rmtree(lock, ignore_errors=True)


def own_mib() -> int:
    """Video memory this process itself holds, as far as PyTorch knows (0 before its first CUDA call)."""
    torch = sys.modules.get("torch")
    try:
        if torch is None or not torch.cuda.is_initialized():
            return 0
        return int(torch.cuda.memory_reserved() >> 20) + CUDA_CONTEXT_MIB
    except Exception:
        return 0


class HoldingVideoMemory(RuntimeError):
    """The process asks for the lock while it still holds video memory: every other holder would wait for it to
    leave the card (wait_for_card) while it waits for them to release the lock - a deadlock."""


def release_own_card_memory(limit_mib: int = SELF_VRAM_MIB) -> None:
    """Before waiting for the lock: PyTorch's cached-but-free blocks given back to the card, and HoldingVideoMemory
    when the process still has more than `limit_mib` of live tensors. A process that holds the card between two
    lock holds stalls everybody (round J: a calibration script kept 3.5 GiB of tables while it waited to take the
    lock again, and the holder waited for it to leave the card). Free the model and the tables (del, gc.collect())
    before the lock is released, or hold it once for the whole script."""
    torch = sys.modules.get("torch")
    try:
        if torch is None or not torch.cuda.is_initialized():
            return
        torch.cuda.empty_cache()
        live = int(torch.cuda.memory_allocated() >> 20)
    except Exception:
        return
    if live > limit_mib:
        raise HoldingVideoMemory(
            f"this process holds {live} MiB of live tensors on the card while asking for the GPU lock; the other "
            f"holders would wait for it to leave the card (more than {OTHER_VRAM_MIB} MiB of others' memory) while "
            f"it waits for the lock: free the model and the tables before releasing the lock, or hold it once for "
            f"the whole script")


def wait_for_card(other_mib: int = OTHER_VRAM_MIB, *, poll: float = POLL_SECONDS, timeout: float | None = None,
                  say: Callable[[str], None] = tell) -> dict[str, Any]:
    """The card's state once other programs hold no more than `other_mib` of it (CardBusy after `timeout`)."""
    began = time.time()
    while True:
        now = card()
        if "error" in now:
            say(f"nvidia-smi could not be asked ({now['error']}): the card is not checked.")
            return now
        others = now["used_mib"] - own_mib()
        if others <= other_mib:
            return now
        waited = time.time() - began
        if timeout is not None and waited >= timeout:
            raise CardBusy(f"other programs hold {others} MiB of video memory ({'; '.join(now['processes'])})")
        say(f"Other programs hold {others} MiB of the card's {now['total_mib']} MiB (more than {other_mib}): "
            f"waiting ({waited:.0f} s so far). Processes on the card: {'; '.join(now['processes']) or 'none listed'}")
        time.sleep(poll)


@contextlib.contextmanager
def gpu_lock(wp: str, purpose: str = "", expected_minutes: float = 5, *, lock: Path = LOCK, check_card: bool = True,
             other_mib: int = OTHER_VRAM_MIB, poll: float = POLL_SECONDS, timeout: float | None = None,
             card_timeout: float | None = None, heartbeat: float = HEARTBEAT_SECONDS,
             stale_seconds: float = STALE_SECONDS) -> Iterator[dict[str, Any]]:
    """Hold the GPU lock for the block. Yields the owner record. `wp`: who takes it, as owner.json and `status`
    show it. A block inside another one of the same process uses the lock that is already held."""
    lock = Path(lock)
    key = os.path.normcase(os.path.abspath(lock))
    with _depth_guard:
        nested = _depth.get(key, 0) > 0
        if nested:
            _depth[key] += 1
    if nested:
        try:
            yield read_owner(lock) or {}
        finally:
            with _depth_guard:
                _depth[key] -= 1
        return

    if check_card:
        release_own_card_memory()
    record = acquire(wp, purpose, expected_minutes, lock=lock, poll=poll, timeout=timeout,
                     stale_seconds=stale_seconds)
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(heartbeat):
            if not ours(lock, record):
                return
            record["heartbeat"] = iso()
            with contextlib.suppress(OSError):
                write_owner(lock, record)

    thread = threading.Thread(target=beat, name="gpu-lock-heartbeat", daemon=True)
    thread.start()
    with _depth_guard:
        _depth[key] = 1
    try:
        if check_card:
            wait_for_card(other_mib, poll=poll, timeout=card_timeout)
        yield record
    finally:
        with _depth_guard:
            _depth[key] = 0
        stop.set()
        thread.join(timeout=5)
        release(lock, record)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="The GPU lock of the Pixal3D low-memory checks.")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("status", help="who holds the lock, and the card's memory in use")
    p.add_argument("--lock", default=str(LOCK), help="the lock folder")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    lock = Path(args.lock)
    state = lock_state(lock)
    print(f"GPU lock ({lock}): {describe(state)}")
    for record in queue_ahead(lock):
        print(f"Waiting: {record.get('wp')} (pid {record.get('pid')}) since {record.get('created')}: "
              f"{record.get('purpose')}")
    now = card()
    if "error" in now:
        print(f"Card: nvidia-smi could not be asked ({now['error']})")
    else:
        print(f"Card: {now['used_mib']} of {now['total_mib']} MiB in use; processes: "
              f"{'; '.join(now['processes']) or 'none listed'}")
    return 3 if state["held"] else 0


if __name__ == "__main__":
    sys.exit(main())
