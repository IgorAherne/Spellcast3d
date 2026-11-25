"""A RAM ballast: page-locked memory held so that this PC behaves like one with less RAM. For developers; stdlib
only, any Python (tools\\python, a venv's):

    python ram_ballast.py --pc-gib 16 | --leave-mib N  [--min-free-mib 2048] [--chunk-mib 256]
                          [--parent-pid PID] [--ready-file ready.json] [--stop-file stop] [--log ballast.csv]
                          [--keep-partial]

--pc-gib G: hold (this PC's RAM - G GiB), so what Windows and every program can use is what a G GiB PC has (a real
16 GiB PC reports a little less than 16,384 MiB as its RAM; pass 15.7 to match one). --leave-mib N: hold what is
available now minus N. Either way it refuses to start when less than --min-free-mib (never under 1,024) would stay
available: a PC that is already that full gives no measurement, only a frozen desktop.

Why page-locked. A ballast that is merely allocated and touched is paged out as soon as another process wants the
RAM, and the measurement then shows a PC with much more memory than the one emulated (the RAM twin of round A's
finding that a GPU ballast in another process is evicted from the card). So the ballast raises its own working-set
quota (SetProcessWorkingSetSizeEx: the minimum working set bounds what VirtualLock may lock) and VirtualLocks
every chunk: Windows cannot trim locked pages, and they leave "available" for good. It also commits them, so the
system's commit headroom shrinks by the same amount, as on the smaller PC (with the same page file; a default
16 GiB PC's page file is smaller than a big one set by hand: limit the page file in Windows' settings for that run).

It checks with the system's "available" counter that the RAM it took is really gone (the drop must be at least 90%
of what it locked), writes --ready-file (JSON: what it holds, available and commit before and after) once it holds
all of it, then prints available RAM, commit charge and the commit headroom once a second (and writes them to
--log: a row of five numbers a second, about 3 KB a minute). It lets go of everything on Ctrl+C, when --stop-file
appears, or when --parent-pid ends (default: the process that started it; 0: never).

Locking needs the right to raise a working set ("Increase a process working set", which Windows gives to every
user) and that much free RAM. When Windows refuses a chunk it says how much it could lock and why, lets go of it
all and exits with code 3 (a partial ballast flatters the result); --keep-partial holds the partial amount instead.
Exit codes: 0 released after a stop, 2 refused (arguments, too little RAM), 3 could not lock all of it.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mem_watch  # noqa: E402  (stdlib + ctypes)

MIB = mem_watch.MIB
MIN_FREE_FLOOR_MIB = 1024           # --min-free-mib cannot go below this
QUOTA_MARGIN_MIB = 128              # working-set room above the locked pages (the program itself, page tables)
MEM_COMMIT, MEM_RESERVE, MEM_RELEASE, PAGE_READWRITE = 0x1000, 0x2000, 0x8000, 0x04
QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE = 0x2, 0x8
DROP_SHARE = 0.9                    # the available counter must fall by at least this share of what was locked


def say(text: str) -> None:
    print(f"[ram ballast {time.strftime('%H:%M:%S')}] {text}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hold page-locked RAM so this PC behaves like one with less RAM.")
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--pc-gib", type=float, help="emulate a PC with this much RAM: hold the rest")
    what.add_argument("--leave-mib", type=int, help="hold what is available now minus this")
    p.add_argument("--min-free-mib", type=int, default=2048,
                   help=f"refuse when less would stay available (at least {MIN_FREE_FLOOR_MIB})")
    p.add_argument("--chunk-mib", type=int, default=256, help="MiB per allocation and lock")
    p.add_argument("--parent-pid", type=int, default=None, help="let go when this process ends (0: never; default: "
                                                                "the parent)")
    p.add_argument("--ready-file", default=None, help="JSON written once the ballast is held")
    p.add_argument("--stop-file", default=None, help="let go when this file appears")
    p.add_argument("--log", default=None, help="CSV, a row a second: t, available, commit, limit, locked (MiB)")
    p.add_argument("--keep-partial", action="store_true", help="hold what could be locked when not all of it can")
    args = p.parse_args(argv)
    if args.min_free_mib < MIN_FREE_FLOOR_MIB:
        p.error(f"--min-free-mib must be at least {MIN_FREE_FLOOR_MIB}: with less available Windows itself stalls")
    if not 16 <= args.chunk_mib <= 4096:
        p.error("--chunk-mib must be between 16 and 4096")
    if args.pc_gib is not None and args.pc_gib <= 0:
        p.error("--pc-gib must be above 0")
    if args.leave_mib is not None and args.leave_mib <= 0:
        p.error("--leave-mib must be above 0")
    return args


def plan(pc_gib: float | None, leave_mib: int | None, total_mib: float, available_mib: float,
         min_free_mib: int) -> tuple[int, str | None]:
    """(MiB to hold, None) or (0, why not). total_mib: the PC's RAM as Windows reports it; available_mib: now."""
    if pc_gib is not None:
        want = pc_gib * 1024
        if want >= total_mib:
            return 0, (f"This PC has {total_mib:,.0f} MiB of RAM: it cannot emulate one with {want:,.0f} MiB "
                       f"({pc_gib:g} GiB).")
        hold = int(total_mib - want)
    else:
        hold = int(available_mib - (leave_mib or 0))
        if hold <= 0:
            return 0, (f"Only {available_mib:,.0f} MiB of RAM are available: already less than the "
                       f"{leave_mib:,} MiB asked to leave.")
    left = available_mib - hold
    if left < min_free_mib:
        return 0, (f"Holding {hold:,} MiB would leave {left:,.0f} MiB of RAM available, under the minimum of "
                   f"{min_free_mib:,} MiB. Close programs (the hub, browsers) and start it again"
                   + (", or emulate a larger PC." if pc_gib is not None else ", or leave more."))
    return hold, None


class Ballast:
    """Page-locked chunks of this process, and the working-set quota that lets them be locked."""

    def __init__(self) -> None:
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32.GetCurrentProcess.restype = wintypes.HANDLE
        self._k32.VirtualAlloc.restype = ctypes.c_void_p
        self._k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
        self._k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD]
        self._k32.VirtualLock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._k32.VirtualUnlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._k32.SetProcessWorkingSetSizeEx.argtypes = [wintypes.HANDLE, ctypes.c_size_t, ctypes.c_size_t,
                                                         wintypes.DWORD]
        self._k32.SetProcessWorkingSetSize.argtypes = [wintypes.HANDLE, ctypes.c_size_t, ctypes.c_size_t]
        self._process = self._k32.GetCurrentProcess()
        self.chunks: list[tuple[int, int]] = []      # (address, bytes)

    @property
    def locked_bytes(self) -> int:
        return sum(size for _, size in self.chunks)

    def _error(self, what: str) -> str:
        code = ctypes.get_last_error()
        return f"{what} failed: error {code} ({ctypes.FormatError(code).strip()})"

    def _quota(self, locked: int) -> str | None:
        """A minimum working set big enough for `locked` bytes and the program itself."""
        low = locked + QUOTA_MARGIN_MIB * MIB
        if self._k32.SetProcessWorkingSetSizeEx(self._process, low, low + QUOTA_MARGIN_MIB * MIB,
                                                QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE):
            return None
        return self._error(f"raising the working-set quota to {low // MIB:,} MiB (SetProcessWorkingSetSizeEx)")

    def add(self, size: int) -> str | None:
        """One more chunk: quota, allocate (commit), lock. None, or why it failed (nothing of the chunk is kept)."""
        problem = self._quota(self.locked_bytes + size)
        if problem:
            return problem
        address = self._k32.VirtualAlloc(None, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
        if not address:
            return self._error(f"committing {size // MIB} MiB (VirtualAlloc; the commit limit?)")
        if not self._k32.VirtualLock(address, size):
            problem = self._error(f"locking {size // MIB} MiB (VirtualLock)")
            self._k32.VirtualFree(address, 0, MEM_RELEASE)
            return problem
        ctypes.memset(address, 0xA5, size)          # written once: the pages exist and are this process's own
        self.chunks.append((address, size))
        return None

    def hold(self, total_mib: int, chunk_mib: int, progress: Any = None) -> str | None:
        """Lock `total_mib` in chunks. None when all of it is held, else why it stopped (what was locked stays)."""
        left = total_mib * MIB
        while left > 0:
            size = min(left, chunk_mib * MIB)
            problem = self.add(size)
            if problem:
                return problem
            left -= size
            if progress is not None:
                progress(self.locked_bytes)
        return None

    def release(self) -> None:
        for address, size in reversed(self.chunks):
            self._k32.VirtualUnlock(address, size)
            self._k32.VirtualFree(address, 0, MEM_RELEASE)
        self.chunks.clear()
        self._k32.SetProcessWorkingSetSize(self._process, ctypes.c_size_t(-1), ctypes.c_size_t(-1))


def snapshot(host: mem_watch.HostProbe) -> dict[str, int]:
    available, commit, limit, total = host.system()
    return {"available_mib": available // MIB, "commit_mib": commit // MIB, "commit_limit_mib": limit // MIB,
            "commit_headroom_mib": (limit - commit) // MIB, "ram_total_mib": total // MIB}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    host = mem_watch.HostProbe()
    before = snapshot(host)
    hold, problem = plan(args.pc_gib, args.leave_mib, before["ram_total_mib"], before["available_mib"],
                         args.min_free_mib)
    if problem:
        say("REFUSED: " + problem)
        return 2
    parent_pid = os.getppid() if args.parent_pid is None else args.parent_pid
    parent = None
    if parent_pid:
        try:
            parent = mem_watch.HostProbe(parent_pid)
        except OSError:
            say(f"The parent process {parent_pid} cannot be watched; stop this one with Ctrl+C or --stop-file.")
    target = (f"a {args.pc_gib:g} GiB PC" if args.pc_gib is not None
              else f"{args.leave_mib:,} MiB of RAM available")
    say(f"RAM {before['ram_total_mib']:,} MiB, available {before['available_mib']:,} MiB, commit "
        f"{before['commit_mib']:,} of {before['commit_limit_mib']:,} MiB. Locking {hold:,} MiB to emulate {target}...")
    ballast = Ballast()
    started = time.perf_counter()
    marks = [0]

    def progress(locked: int) -> None:
        if locked >= marks[0] + 2048 * MIB:
            marks[0] = locked
            say(f"  {locked // MIB:,} MiB locked")

    stop_file = Path(args.stop_file) if args.stop_file else None
    log = None
    code = 0
    try:
        failure = ballast.hold(hold, args.chunk_mib, progress)
        locked_mib = ballast.locked_bytes // MIB
        if failure:
            say(f"COULD NOT LOCK ALL OF IT: {locked_mib:,} of {hold:,} MiB are locked; then {failure}. Locking needs "
                f"the right to raise a working set (every user has it unless a policy took it away) and that much "
                f"free RAM: close programs and try again, or emulate a larger PC.")
            if not args.keep_partial:
                return 3
            say(f"--keep-partial: holding the {locked_mib:,} MiB that are locked.")
        time.sleep(1.0)                              # Windows' counters settle
        after = snapshot(host)
        drop = before["available_mib"] - after["available_mib"]
        verified = drop >= DROP_SHARE * locked_mib
        state = {"pid": os.getpid(), "pc_gib": args.pc_gib, "leave_mib": args.leave_mib, "hold_mib": hold,
                 "locked_mib": locked_mib, "partial": bool(failure), "failure": failure, "verified": verified,
                 "available_drop_mib": drop, "before": before, "after": after,
                 "seconds": round(time.perf_counter() - started, 1), "perf_counter": time.perf_counter(),
                 "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        if verified:
            say(f"Holding {locked_mib:,} MiB page-locked: available {before['available_mib']:,} -> "
                f"{after['available_mib']:,} MiB, commit headroom {before['commit_headroom_mib']:,} -> "
                f"{after['commit_headroom_mib']:,} MiB.")
        else:
            say(f"WARNING: available RAM fell by only {drop:,} MiB while {locked_mib:,} MiB were locked: something "
                f"else let go of memory meanwhile, or Windows does not count the ballast. Look before trusting it.")
        if args.ready_file:
            Path(args.ready_file).write_text(json.dumps(state, indent=1), encoding="utf-8")
        if args.log:
            log = open(args.log, "w", encoding="utf-8", newline="\n")
            log.write("t,available,commit,commit_limit,locked\n")
        say("Ctrl+C" + (f", {stop_file}" if stop_file else "") + (f" or the end of process {parent_pid}"
                                                                    if parent is not None else "")
            + " lets go of it.")
        while True:
            now = snapshot(host)
            line = (f"available {now['available_mib']:>7,} MiB | commit {now['commit_mib']:>7,} of "
                    f"{now['commit_limit_mib']:,} MiB (headroom {now['commit_headroom_mib']:,}) | ballast "
                    f"{locked_mib:,} MiB")
            print(line, flush=True)
            if log is not None:
                log.write(f"{time.perf_counter():.1f},{now['available_mib']},{now['commit_mib']},"
                          f"{now['commit_limit_mib']},{locked_mib}\n")
                log.flush()
            if stop_file is not None and stop_file.exists():
                say("The stop file is there.")
                break
            if parent is not None and not parent.alive():
                say(f"Process {parent_pid} ended.")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        say("Ctrl+C.")
    finally:
        held = ballast.locked_bytes // MIB
        ballast.release()
        if log is not None:
            log.close()
        if held:
            say(f"Released {held:,} MiB.")
    return code


if __name__ == "__main__":
    sys.exit(main())
