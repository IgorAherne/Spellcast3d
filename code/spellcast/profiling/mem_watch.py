"""A background sampler of memory and GPU use, for the low-memory work on the torch-based tools (Pixal3D first).

    watch = MemWatch(folder / "startup.csv", hz=20)
    watch.start()
    watch.stage = "sampling"                 # any text: every row carries the label current when it was taken
    watch.segment("gen1", folder / "gen1.csv")   # the next rows go to another file, labelled run=gen1
    watch.stop()

One daemon thread wakes `hz` times a second and writes one CSV row. What a row holds (memory in MiB):

  t               seconds since the watch was made (time.perf_counter; the event records of the launchers with
                  probes use the same clock, so a stage's rows are those with t0 <= t <= t1)
  run, stage      the labels current at that moment (set by the owner; plain attribute reads)
  torch_alloc     torch.cuda.memory_allocated()  - tensors alive in PyTorch's allocator. No synchronisation.
  torch_reserved  torch.cuda.memory_reserved()   - what PyTorch's allocator holds from the driver (alive + cached)
  nvml_used       the CARD's dedicated memory in use, every process and the desktop included (NVML through ctypes
                  on nvml.dll; the number nvidia-smi prints)
  nvml_util       the card's utilisation in percent, as nvidia-smi prints it (nvmlDeviceGetUtilizationRates: measured
                  here to change only twice a second and about 0.4 s late - use the .util.csv file for idle gaps)
  proc_ded        THIS process's dedicated GPU memory  - Windows' counter \\GPU Process Memory(pid_N_*)\\Dedicated Usage
  proc_shared     THIS process's shared GPU memory     - ...\\Shared Usage: system RAM the driver uses once the
                  dedicated memory is full (the NVIDIA driver's "sysmem fallback"). The only way to see the spill:
                  neither torch nor NVML counts it.
  adapter_ded     the card's dedicated memory in use, all processes  - \\GPU Adapter Memory(luid of the card)\\...
  adapter_shared  the card's shared memory in use, all processes       (the adapter this process opened; empty
                  until CUDA starts, since only then Windows says which adapter that is)
  priv            the process's private bytes (committed memory: what counts against RAM + page file)
  ws              the process's working set (the part of it in physical RAM right now)
  avail           the system's available physical RAM
  commit          the system's commit charge (all processes; the limit is RAM + page file)
  late_ms         how late the row was taken (the thread waits for the GIL like any other; a long gap means the
                  rows around it are sparse, not that anything was missed in the peaks torch itself keeps)
  ws_private      the process's PRIVATE working set (\\Process V2(*:pid)\\Working Set - Private): the RAM only it
                  uses, without the DLLs and mapped files it shares with others
  commit_limit    the system's commit limit (RAM + page file as it is sized right now)
  + the owner's extra columns (MemWatch(extra={"pinned": fn})), each fn() -> bytes, -1 for "unknown"

The columns are only ever added at the end: readers take them by name.

Next to the rows' file, <name>.util.csv gets the driver's own utilisation samples (nvmlDeviceGetSamples: one per
200 ms, each stamped by the driver, fetched once a second): `t` on the same clock and `util` in percent of the 200 ms
before it. That is the trace to find the card's idle stretches with.

The Windows counters are read through PDH (ctypes on pdh.dll) in this thread: no child process. They are read every
`pdh_every`-th row (default: every row). A probe that fails at start is left out (its columns stay empty) and said
once on the console; nothing here ever raises into the program it watches.

Verified on the RTX 3060 / Windows 11 PC (python mem_watch.py gpu): a 1 GiB and a 2 GiB tensor raise proc_ded by
exactly 1024 and 2048 MiB within one row (50 ms) and empty_cache() takes them off again; nvml_used and adapter_ded
move by the same amounts. Windows' \\GPU Engine(pid_N_*) counters are NOT used: this driver lists no CUDA / compute
engine there (3D, Copy, VideoDecode ... only), and their running time did not follow 3 s of matrix products.

The watch from OUTSIDE. The thread needs the GIL for each row, so a native call that keeps it (cumesh's BVH build:
5.8 s) leaves a gap in the rows. `python mem_watch.py watch --pid N --out file.csv --origin T` is the same sampler as
a process of its own, watching process N: no gaps, and no torch columns (PyTorch's numbers exist only inside N;
its exact peaks per stage are in the stage records anyway). --origin is N's time.perf_counter() at its t = 0
(MemWatch.origin): Windows' performance counter is one clock for every process, so both files share `t`. It ends
when N does. The launchers with probes start it with --profile-external (launcher_probes.watch_from_outside).

Cost: one row is a few ctypes calls that release the GIL and two torch calls that take the allocator's lock for
microseconds - about 0.1 ms; `python mem_watch.py` prints it for this PC. stdlib + ctypes at import; torch is looked
up in sys.modules at every row (the watch starts before the launcher imports torch, and must not import it first).
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any, IO

MIB = 1024 * 1024
COLUMNS = ("t", "run", "stage", "torch_alloc", "torch_reserved", "nvml_used", "nvml_util", "proc_ded",
           "proc_shared", "adapter_ded", "adapter_shared", "priv", "ws", "avail", "commit", "late_ms", "ws_private",
           "commit_limit")
UTIL_COLUMNS = ("t", "util")


def say(text: str) -> None:
    print(f"[mem_watch] {text}", flush=True)


# ---- the process and the system (kernel32 / psapi) -------------------------------------------------------------------

class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


class _PerformanceInformation(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                ("HandleCount", wintypes.DWORD), ("ProcessCount", wintypes.DWORD), ("ThreadCount", wintypes.DWORD)]


class HostProbe:
    """This process's private bytes and working set, the system's available RAM and commit charge (bytes)."""

    def __init__(self, pid: int | None = None) -> None:
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self._k32.GetCurrentProcess.restype = wintypes.HANDLE
        self._k32.OpenProcess.restype = wintypes.HANDLE
        self._k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self._psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        self._psapi.GetPerformanceInfo.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        if pid is None:
            self._process = self._k32.GetCurrentProcess()
        else:                                    # PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE
            self._process = self._k32.OpenProcess(0x1000 | 0x00100000, False, pid)
            if not self._process:
                raise OSError(f"process {pid} cannot be opened")
        self._pmc = _ProcessMemoryCountersEx()
        self._pmc.cb = ctypes.sizeof(self._pmc)
        self._perf = _PerformanceInformation()
        self._perf.cb = ctypes.sizeof(self._perf)

    def process(self) -> tuple[int, int]:
        """(private bytes, working set)."""
        if not self._psapi.GetProcessMemoryInfo(self._process, ctypes.byref(self._pmc), self._pmc.cb):
            return 0, 0
        return int(self._pmc.PrivateUsage), int(self._pmc.WorkingSetSize)

    def alive(self) -> bool:
        """Whether the watched process still runs."""
        return self._k32.WaitForSingleObject(self._process, 0) == 0x102      # WAIT_TIMEOUT

    def system(self) -> tuple[int, int, int, int]:
        """(available physical RAM, commit charge, commit limit, physical RAM), bytes."""
        if not self._psapi.GetPerformanceInfo(ctypes.byref(self._perf), self._perf.cb):
            return 0, 0, 0, 0
        page = int(self._perf.PageSize)
        return (int(self._perf.PhysicalAvailable) * page, int(self._perf.CommitTotal) * page,
                int(self._perf.CommitLimit) * page, int(self._perf.PhysicalTotal) * page)


class _ProcHostProbe:
    """HostProbe on a Linux test pod, from /proc: private bytes = resident anonymous memory + swapped out, working set
    = resident; the physical RAM and what is available in the container's limit (cgroup v2) when it has one."""

    def __init__(self, pid: int | None = None) -> None:
        self._pid = os.getpid() if pid is None else pid
        if not self.alive():
            raise OSError(f"process {self._pid} cannot be opened")

    @staticmethod
    def _kib(text: str, *keys: str) -> dict[str, int]:
        found = {}
        for line in text.splitlines():
            key, _, rest = line.partition(":")
            if key in keys:
                found[key] = int(rest.split()[0]) * 1024
        return found

    def process(self) -> tuple[int, int]:
        try:
            status = self._kib(Path(f"/proc/{self._pid}/status").read_text(), "RssAnon", "VmSwap", "VmRSS")
        except OSError:
            return 0, 0
        return status.get("RssAnon", 0) + status.get("VmSwap", 0), status.get("VmRSS", 0)

    def alive(self) -> bool:
        try:
            return Path(f"/proc/{self._pid}/stat").read_text().rpartition(")")[2].split()[0] not in ("Z", "X")
        except (OSError, IndexError):
            return False

    def system(self) -> tuple[int, int, int, int]:
        """(available physical RAM, commit charge, commit limit, physical RAM); the commit figures are the host's."""
        try:
            info = self._kib(Path("/proc/meminfo").read_text(), "MemAvailable", "Committed_AS", "CommitLimit",
                             "MemTotal")
        except OSError:
            return 0, 0, 0, 0
        available, total = info.get("MemAvailable", 0), info.get("MemTotal", 0)
        cgroup = Path("/sys/fs/cgroup")
        try:        # the container's limit, less what it holds that cannot be dropped (its inactive file pages can)
            limit = int((cgroup / "memory.max").read_text())
            inactive = next(int(line.split()[1]) for line in (cgroup / "memory.stat").read_text().splitlines()
                            if line.startswith("inactive_file "))
            used = int((cgroup / "memory.current").read_text()) - inactive
            available, total = min(available, max(0, limit - used)), min(total, limit)
        except (OSError, ValueError, StopIteration):
            pass                                    # no limit ("max"), or no cgroup v2: the machine's figures
        return available, info.get("Committed_AS", 0), info.get("CommitLimit", 0), total


if os.name != "nt":
    HostProbe = _ProcHostProbe      # type: ignore[misc]  # noqa: F811


# ---- the card (NVML through ctypes) ----------------------------------------------------------------------------------

class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _NvmlValue(ctypes.Union):
    _fields_ = [("dVal", ctypes.c_double), ("uiVal", ctypes.c_uint), ("ulVal", ctypes.c_ulong),
                ("ullVal", ctypes.c_ulonglong), ("sllVal", ctypes.c_longlong)]


class _NvmlSample(ctypes.Structure):
    _fields_ = [("timeStamp", ctypes.c_ulonglong), ("sampleValue", _NvmlValue)]


_NVML_GPU_UTILIZATION_SAMPLES = 1


class NvmlProbe:
    """The card's memory in use and utilisation, as nvidia-smi shows them. index: the card's NVML index."""

    def __init__(self, index: int = 0) -> None:
        if os.name == "nt":
            system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvml.dll"
            self._nvml = ctypes.WinDLL(str(system32) if system32.is_file() else "nvml.dll")
        else:                       # a Linux test pod: the driver's own library
            self._nvml = ctypes.CDLL("libnvidia-ml.so.1")
        if self._nvml.nvmlInit_v2() != 0:
            raise OSError("nvmlInit_v2 failed")
        self._handle = ctypes.c_void_p()
        if self._nvml.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(self._handle)) != 0:
            raise OSError(f"NVML has no card {index}")
        self._memory = _NvmlMemory()
        self._utilization = _NvmlUtilization()
        self._last_stamp = 0
        self._samples = (_NvmlSample * 512)()
        self.total = self.read()[2]

    def read(self) -> tuple[int, int, int]:
        """(memory in use in bytes, utilisation in percent, total memory in bytes); -1 for what failed."""
        used, total, util = -1, -1, -1
        if self._nvml.nvmlDeviceGetMemoryInfo(self._handle, ctypes.byref(self._memory)) == 0:
            used, total = int(self._memory.used), int(self._memory.total)
        if self._nvml.nvmlDeviceGetUtilizationRates(self._handle, ctypes.byref(self._utilization)) == 0:
            util = int(self._utilization.gpu)
        return used, util, total

    def new_samples(self) -> list[tuple[float, int]]:
        """The driver's utilisation samples since the last call: (its time stamp in seconds since the epoch, percent
        of the ~200 ms before it). The driver keeps about 14 s of them."""
        kind, count = ctypes.c_uint(0), ctypes.c_uint(len(self._samples))
        status = self._nvml.nvmlDeviceGetSamples(self._handle, _NVML_GPU_UTILIZATION_SAMPLES,
                                                 ctypes.c_ulonglong(self._last_stamp), ctypes.byref(kind),
                                                 ctypes.byref(count), self._samples)
        if status != 0 or count.value == 0:
            return []
        found = [(int(self._samples[i].timeStamp), int(self._samples[i].sampleValue.uiVal))
                 for i in range(min(count.value, len(self._samples)))]
        found = [(stamp, value) for stamp, value in found if stamp > self._last_stamp]
        if found:
            self._last_stamp = max(stamp for stamp, _ in found)
        return [(stamp / 1e6, value) for stamp, value in found]


# ---- Windows' GPU counters (PDH through ctypes) ----------------------------------------------------------------------

_PDH_FMT_LARGE = 0x00000400
_PDH_MORE_DATA = 0x800007D2


class _PdhValue(ctypes.Union):
    _fields_ = [("longValue", ctypes.c_long), ("doubleValue", ctypes.c_double),
                ("largeValue", ctypes.c_longlong), ("text", ctypes.c_void_p)]


class _PdhCounterValue(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("value", _PdhValue)]


class _PdhItem(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _PdhCounterValue)]


class PdhProbe:
    """Windows' per-process and per-card GPU memory counters.

    A process's instance appears when it first opens the card (pid_N_luid_0x..._0x..._phys_0); the counters are
    added with a wildcard, which PDH expands again at every collection, so a watch made before CUDA starts sees it
    arrive. The card's own counters (\\GPU Adapter Memory) have one instance per adapter of the PC (this one has
    three); the one summed is the adapter the process opened, known from its instance's name.
    read() -> (process dedicated, process shared, card dedicated, card shared) in bytes; -1 where there is no instance.
    """

    PATHS = (r"\GPU Process Memory(pid_{pid}_*)\Dedicated Usage", r"\GPU Process Memory(pid_{pid}_*)\Shared Usage",
             r"\GPU Adapter Memory(*)\Dedicated Usage", r"\GPU Adapter Memory(*)\Shared Usage")

    def __init__(self, pid: int | None = None) -> None:
        self.pid = os.getpid() if pid is None else pid
        self._pdh = ctypes.WinDLL("pdh")
        self._query = wintypes.HANDLE()
        if self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)) != 0:
            raise OSError("PdhOpenQueryW failed")
        self._counters: list[Any] = []
        for path in self.PATHS:
            counter = wintypes.HANDLE()
            status = self._pdh.PdhAddEnglishCounterW(self._query, path.format(pid=self.pid), 0, ctypes.byref(counter))
            if status != 0:
                raise OSError(f"PDH has no counter {path} (status 0x{status & 0xFFFFFFFF:08X})")
            self._counters.append(counter)
        self._buffer = ctypes.create_string_buffer(16384)
        self.luid = ""                           # "luid_0x00000000_0x0000F129": the adapter this process opened
        self._luid_final = False
        self.names: list[str] = []               # the process's instances at the last read

    def _items(self, counter: Any) -> list[tuple[str, int]]:
        """[(instance name, value)] of a wildcard counter after a collection."""
        for _ in range(2):
            size, count = wintypes.DWORD(len(self._buffer)), wintypes.DWORD(0)
            status = self._pdh.PdhGetFormattedCounterArrayW(counter, _PDH_FMT_LARGE, ctypes.byref(size),
                                                            ctypes.byref(count), self._buffer)
            if status & 0xFFFFFFFF == _PDH_MORE_DATA:
                self._buffer = ctypes.create_string_buffer(size.value * 2)
                continue
            if status != 0:
                return []
            items = ctypes.cast(self._buffer, ctypes.POINTER(_PdhItem))
            return [(items[i].szName or "", int(items[i].FmtValue.value.largeValue)) for i in range(count.value)]
        return []

    def read(self) -> tuple[int, int, int, int]:
        if self._pdh.PdhCollectQueryData(self._query) != 0:
            return -1, -1, -1, -1
        dedicated, shared = self._items(self._counters[0]), self._items(self._counters[1])
        self.names = [name for name, _ in dedicated]
        if not self._luid_final and dedicated:
            name, most = max(dedicated, key=lambda item: item[1])       # pid_N_luid_0x..._0x..._phys_0
            start = name.find("luid_")
            if start >= 0:
                self.luid = "_".join(name[start:].split("_")[:3])
                self._luid_final = most > 0      # an instance can exist before the process uses any adapter
        card = [-1, -1]
        if self.luid:
            for n, counter in enumerate(self._counters[2:]):
                values = [value for name, value in self._items(counter) if name.startswith(self.luid)]
                card[n] = sum(values) if values else -1
        return (sum(v for _, v in dedicated) if dedicated else -1, sum(v for _, v in shared) if shared else -1,
                card[0], card[1])

    def close(self) -> None:
        self._pdh.PdhCloseQuery(self._query)


class PrivateWsProbe:
    """A process's private working set (bytes) from Windows' "Process V2" counters, whose instances are named
    <image>:<pid>, so the pid alone finds them (the older "Process" set names instances python#3 and renames
    them when another python ends). read() -> bytes, -1 when there is no instance."""

    def __init__(self, pid: int | None = None) -> None:
        self.pid = os.getpid() if pid is None else pid
        self._pdh = ctypes.WinDLL("pdh")
        self._query = wintypes.HANDLE()
        if self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)) != 0:
            raise OSError("PdhOpenQueryW failed")
        self._counter = wintypes.HANDLE()
        path = rf"\Process V2(*:{self.pid})\Working Set - Private"
        status = self._pdh.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(self._counter))
        if status != 0:
            raise OSError(f"PDH has no counter {path} (status 0x{status & 0xFFFFFFFF:08X})")
        self._buffer = ctypes.create_string_buffer(4096)

    def read(self) -> int:
        if self._pdh.PdhCollectQueryData(self._query) != 0:
            return -1
        size, count = wintypes.DWORD(len(self._buffer)), wintypes.DWORD(0)
        status = self._pdh.PdhGetFormattedCounterArrayW(self._counter, _PDH_FMT_LARGE, ctypes.byref(size),
                                                        ctypes.byref(count), self._buffer)
        if status != 0 or count.value == 0:
            return -1
        items = ctypes.cast(self._buffer, ctypes.POINTER(_PdhItem))
        return sum(int(items[i].FmtValue.value.largeValue) for i in range(count.value))


# ---- the watch -------------------------------------------------------------------------------------------------------

def _mib(value: int) -> str:
    return "" if value < 0 else f"{value / MIB:.0f}"


class MemWatch(threading.Thread):
    """The sampler thread (see the module's docstring). `origin`: the perf_counter value that is t = 0."""

    def __init__(self, path: Path, hz: float = 20.0, pdh_every: int = 1, origin: float | None = None,
                 nvml_index: int = 0, pid: int | None = None, extra: dict[str, Any] | None = None) -> None:
        super().__init__(name="mem-watch", daemon=True)
        self.extra = dict(extra or {})          # column -> fn() returning bytes (-1: unknown); read every row
        self.columns = COLUMNS + tuple(self.extra)
        self.origin = time.perf_counter() if origin is None else origin
        self.epoch = time.time() - (time.perf_counter() - self.origin)     # the wall clock at t = 0
        self.period = 1.0 / hz
        self.pdh_every = max(1, int(pdh_every))
        self.util_every = max(1, round(hz))                                 # rows between two fetches of the samples
        self.run_label = "startup"
        self.stage = ""
        self.rows = 0
        self._failed = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()            # the files: segment() swaps them under the sampler's feet
        self._file: IO[str] | None = None
        self._util_file: IO[str] | None = None
        self._open(Path(path))
        self.host: HostProbe | None = self._probe("the process and system memory", HostProbe, pid)
        self.nvml: NvmlProbe | None = self._probe("NVML (the card's memory and utilisation)", NvmlProbe, nvml_index)
        self.pdh: PdhProbe | None = self._probe("Windows' GPU counters (per-process dedicated and shared memory)",
                                                PdhProbe, pid)
        self.private_ws: PrivateWsProbe | None = self._probe("the private working set (Process V2 counters)",
                                                             PrivateWsProbe, pid)
        self._pdh_last: tuple[str, ...] = ("", "", "", "")
        self._private_last = ""

    @staticmethod
    def _probe(what: str, make: Any, *args: Any) -> Any:
        try:
            return make(*args)
        except Exception as exc:
            say(f"{what} cannot be read here ({type(exc).__name__}: {exc}); those columns stay empty.")
            return None

    def _open(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", encoding="utf-8", newline="\n", buffering=1 << 16)
        self._file.write(",".join(self.columns) + "\n")
        self._util_file = open(path.with_suffix(".util.csv"), "w", encoding="utf-8", newline="\n")
        self._util_file.write(",".join(UTIL_COLUMNS) + "\n")
        self.path = path

    def segment(self, run_label: str, path: Path | None = None) -> None:
        """The next rows carry run=run_label; with `path`, they (and the utilisation samples) go to new files."""
        with self._lock:
            self.run_label = run_label
            if path is not None and self._file is not None:
                self._write_util()
                self._file.close()
                if self._util_file is not None:
                    self._util_file.close()
                self._open(Path(path))

    def now(self) -> float:
        return time.perf_counter() - self.origin

    def row(self, late: float = 0.0) -> list[str]:
        """One row, as the CSV's cells."""
        clock = time.perf_counter()
        cells = [f"{clock - self.origin:.3f}", self.run_label, self.stage.replace(",", ";")]
        torch = sys.modules.get("torch")
        allocated = reserved = -1
        if torch is not None:
            try:
                if torch.cuda.is_initialized():
                    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            except Exception:
                pass                                      # torch half-imported, or no CUDA: empty cells
        cells += [_mib(allocated), _mib(reserved)]
        used, util = (self.nvml.read()[:2] if self.nvml is not None else (-1, -1))
        cells += [_mib(used), "" if util < 0 else str(util)]
        if self.pdh is not None and self.rows % self.pdh_every == 0:
            self._pdh_last = tuple(_mib(value) for value in self.pdh.read())
        cells += list(self._pdh_last)
        private, working = self.host.process() if self.host is not None else (-1, -1)
        available, commit, limit = self.host.system()[:3] if self.host is not None else (-1, -1, -1)
        cells += [_mib(private), _mib(working), _mib(available), _mib(commit), f"{late * 1000:.0f}"]
        if self.private_ws is not None and self.rows % self.pdh_every == 0:
            self._private_last = _mib(self.private_ws.read())
        cells += [self._private_last, _mib(limit)]
        for read in self.extra.values():
            try:
                cells.append(_mib(int(read())))
            except Exception:
                cells.append("")
        return cells

    def _write_util(self) -> None:
        """The driver's new utilisation samples into the .util.csv file (call with the lock held)."""
        if self.nvml is None or self._util_file is None:
            return
        for stamp, value in self.nvml.new_samples():
            self._util_file.write(f"{stamp - self.epoch:.3f},{value}\n")
        self._util_file.flush()

    def run(self) -> None:
        due = time.perf_counter()
        flushed = due
        while not self._stop_event.is_set():
            now = time.perf_counter()
            try:
                cells = self.row(max(0.0, now - due))
                with self._lock:
                    if self._file is not None:
                        self._file.write(",".join(cells) + "\n")
                        if now - flushed >= 0.5:           # a killed process keeps all but the last half second
                            self._file.flush()
                            flushed = now
                    if self.rows % self.util_every == 0:
                        self._write_util()
                self.rows += 1
            except Exception as exc:                       # never into the watched program
                if not self._failed:
                    self._failed = True
                    say(f"a row failed ({type(exc).__name__}: {exc}); the watch goes on.")
            due += self.period
            wait = due - time.perf_counter()
            if wait < -self.period:                        # fell behind (the GIL was held): do not catch up in a burst
                due = time.perf_counter()
                wait = 0.0
            if wait > 0:
                self._stop_event.wait(wait)

    def flush(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.flush()
            self._write_util()

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(2.0)
        with self._lock:
            self._write_util()
            for handle in (self._file, self._util_file):
                if handle is not None:
                    handle.close()
            self._file = self._util_file = None


# ---- self-test: what a row costs, and whether the counters track this process ----------------------------------------

def _self_test() -> int:
    """`python mem_watch.py [gpu]`: the cost of one row; with `gpu` (in a venv with torch), 1 GiB and 2 GiB are
    allocated on the card and released, and the per-process counter must follow them."""
    folder = Path(os.environ.get("TEMP", ".")) / f"mem_watch_test_{os.getpid()}"
    watch = MemWatch(folder / "test.csv", hz=20)
    try:
        for _ in range(3):
            watch.row()
        started = time.perf_counter()
        for _ in range(200):
            watch.row()
        per_row = (time.perf_counter() - started) / 200
        print(f"one row: {per_row * 1000:.2f} ms ({per_row * 20 * 100:.2f}% of one core at 20 Hz)")
        print(dict(zip(watch.columns, watch.row())))
        if len(sys.argv) > 1 and sys.argv[1] == "gpu":
            import torch

            def show(what: str) -> None:
                cells = dict(zip(watch.columns, watch.row()))
                print(f"{what:28s} " + " ".join(f"{key} {cells[key] or '-':>5}" for key in (
                    "torch_alloc", "torch_reserved", "proc_ded", "proc_shared", "nvml_used", "adapter_ded",
                    "adapter_shared")))

            show("before CUDA starts")
            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            show("CUDA context made")
            block = torch.empty(1024, 1024, 1024, dtype=torch.uint8, device="cuda")
            block.fill_(1)
            torch.cuda.synchronize()
            show("1 GiB tensor allocated")
            time.sleep(0.05)
            show("... 50 ms later")
            more = torch.empty(2048, 1024, 1024, dtype=torch.uint8, device="cuda")
            more.fill_(1)
            torch.cuda.synchronize()
            show("+2 GiB tensor allocated")
            time.sleep(0.05)
            show("... 50 ms later")
            del block, more
            show("tensors deleted (cached)")
            torch.cuda.empty_cache()
            time.sleep(0.05)
            show("empty_cache()")
            print("this process's instances:", watch.pdh.names if watch.pdh else None, "adapter:",
                  watch.pdh.luid if watch.pdh else None)
            time.sleep(1.0)
            print("utilisation samples of the driver:", watch.nvml.new_samples()[-8:] if watch.nvml else None)
    finally:
        watch.stop()
        for name in ("test.csv", "test.util.csv"):
            try:
                (folder / name).unlink()
            except OSError:
                pass
        try:
            folder.rmdir()
        except OSError:
            pass
    return 0


def _watch_outside(argv: list[str]) -> int:
    """`python mem_watch.py watch --pid N --out file.csv [--origin T] [--hz 20]`: rows about process N until it ends."""
    import argparse

    p = argparse.ArgumentParser(prog="mem_watch.py watch")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--origin", type=float, default=None, help="the watched process's perf_counter at its t = 0")
    p.add_argument("--hz", type=float, default=20.0)
    args = p.parse_args(argv)
    watch = MemWatch(Path(args.out), hz=args.hz, origin=args.origin, pid=args.pid)
    watch.run_label = "outside"
    if watch.host is None:
        return 2
    watch.start()
    try:
        while watch.host.alive():
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    watch.stop()
    return 0


if __name__ == "__main__":
    sys.exit(_watch_outside(sys.argv[2:]) if len(sys.argv) > 1 and sys.argv[1] == "watch" else _self_test())
