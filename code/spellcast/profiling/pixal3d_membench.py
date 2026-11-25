"""Pixal3D memory bench: a measured launcher started on its own, a few batches (--batch: the staged path the hub
uses, pixal3d_staged_profiled.py) or generations (the round-A reference path, code/devtools/pixal3d-lowmem/
pixal3d_profiled.py) sent to it, then stopped. For developers; with any Python that has the stdlib (the pixal3d
venv's, or tools\\python):

    code\\venvs\\pixal3d\\Scripts\\python.exe code\\spellcast\\profiling\\pixal3d_membench.py
        --out <folder for the measurements> --work <an empty folder: the server's projects root>
        --image <picture> [<picture 2> ...] [--runs 1024,1024,1536 | 1024,1024@2 | mv1024,mv1024,1024]
        [--views <front picture>:0 <left>:90 <back>:180 <right>:270] [--views-b ...] [--name pixal3d] [--seed 0]
        [--faces 300000] [--texture 2048]
        [--port 7791] [--low-vram on|off] [--first-job single|multi] [--attn-backend flash_attn] [--step-sync]
        [--no-external] [--pause 8] [--keep-glb] [--ballast-leave MIB]

The server is started the way the hub starts Pixal3D (adapters/pixal3d.py launch_spec: the venv's python, --repo,
--port, --low-vram, --attn-backend, --projects-root, --first-job, the working folder code\\pixal3d, the environment
paths.clean_env gives a tool), with three differences: the script is the launcher with probes with its --profile-*
arguments, --projects-root is --work (never the artist's projects), and the port is the bench's. NEVER the hub
itself, and no Job object: if the bench is killed, stop the server by hand (the PID is in <name>.bench.json).
Without --batch the probes are round A's (pixal3d_profiled.py, a developer's checkout only: code/devtools is not in
the release).

--runs: one generation per entry, each a quality (1024 or 1536), all in the same server process: the first carries
the one-off costs (kernels compiled or loaded, a cold file cache), the second is the steady state. Each request is
the hub's (adapters/pixal3d.py run_stage for a prop with one chosen picture and the default mesh settings of
projects.STAGE_DEFAULTS): image_path, seed, resolution, decimation_target, texture_size, out_path. While a request
runs, /spellcast/progress is polled once a second, as the hub does. The GLB is measured and deleted (--keep-glb
keeps it). --pause: idle seconds after each generation, so the sampler also records what stays resident.

A run "mvQUALITY" is a prop made from several views, the hub's request for it (adapters/pixal3d.py run_stage): the
--views pictures (PATH:AZIMUTH, 0 front, 90 left, 180 back, 270 right; whole pictures, rect null) as "views", the
front one as image_path, and views_dir, where the launcher saves the framed views it made (deleted after the run).
A run "mvbQUALITY" takes the --views-b pictures instead (a second prop, or the same views without transparency).
Start with --first-job multi when the first run is one, as the hub does. --ballast-leave MIB emulates a smaller
card (--profile-ballast-leave: a block held inside the server's process, launcher_probes.hold_ballast).

It writes <name>.bench.log (the server's output) and <name>.bench.json (what was started, the card before and
after, per run: the answer, the wall time, the GLB's size) next to the measurements, and prints a line per step.
Exit code 0 when every run was answered without a failure.

BATCHES (--batch): the staged path the hub uses. The server is pixal3d_staged_profiled.py instead (same launch,
its --profile-* arguments) and each batch is POST /spellcast/batch with the hub's body (adapters/pixal3d.py
run_batch: batch_id, work_dir <work>/<project>/temp/mesh-<batch id>, wave_size, resume, items with id, image_path,
out_path, seed, resolution, decimation_target, texture_size, views, views_dir), then GET /spellcast/batch/<id> once
a second until it ends, as the hub polls. --props lists the props, each one of:
  sv1024 sv1536 mv1024 mv1024b big1536   the golden cases (pixal3d_golden.py CASES: their pictures and seeds, so the
                                         meshes can be held against the golden ones)
  QUALITY:PICTURE[:SEED]                 one picture (a path, or a file name in code\\pixal3d\\assets\\images)
  <file>.json                            props listed in a JSON file: [{"id", "resolution", "seed", "picture": path}
                                         | {..., "views": [[path, azimuth], ...]}], paths relative to the file;
                                         "geometry": "hi3d" makes the shape with Hi3D
  mixed4                                 [mv1024, sv1024, mv1024b, sv1536] (design note 10, 4.3.5)
  ten                                    sv1024, mv1024 and 8 more pictures of assets\\images at 1024 (two waves of 8)
    code\\venvs\\pixal3d\\Scripts\\python.exe code\\spellcast\\profiling\\pixal3d_membench.py --batch
        --out <folder> --work <an empty folder> --props mixed4 [--batches 1] [--wave-size 8] [--name staged]
        [--ballast-leave MIB] [--ram-pc-gib 16] [--purge-cache] [--mesh-hashes] [--faces 300000]
        [--texture 2048] [--keep-glb]
--batches N sends the same batch N times to the same server (N=2: the second job on a running tool). --purge-cache
drops Pixal3D's network files from Windows' file cache before every batch (each file opened once unbuffered: no
admin rights needed), for the cold-cache rows of note 10, 2.5. --ram-pc-gib G starts ram_ballast.py --pc-gib G
(--ram-leave-mib N: ram_ballast.py --leave-mib N, so that N MiB stay available whatever this PC's other programs hold)
before the server and stops it after (the PC behaves like one with G GiB; it writes <name>.ramballast.json and
.ramballast.csv). --mesh-hashes: pixal3d_staged_profiled.py --profile-mesh-hashes (the decoded meshes'
SHA-256, for the report to hold against the golden ones). bench.json then holds, per batch: the body (paths left
out), the 202 answer, the part / loading timeline as the polls saw it (a line per change), the final record, the
wall time, the longest poll answer, the GLB sizes. Nothing large is kept: the GLBs and the views the launcher
saved are deleted unless --keep-glb. bench.json is a few KB per batch.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROUND_A = CODE / "devtools" / "pixal3d-lowmem" / "pixal3d_profiled.py"       # the generate mode's launcher with probes
if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == HERE:
    del sys.path[0]
if str(CODE) not in sys.path:
    sys.path.append(str(CODE))

from spellcast import paths  # noqa: E402  (stdlib only)
from spellcast.profiling import toolkit  # noqa: E402  (stdlib only)

READY_TIMEOUT = 1500.0          # the hub waits 1200 s; a little more, since the probes are attached
GENERATE_TIMEOUT = 45 * 60.0    # adapters/pixal3d.py GENERATE_TIMEOUT
POLL_INTERVAL = 1.0             # adapters/pixal3d.py POLL_INTERVAL
STALL_TIMEOUT = 45 * 60.0       # adapters/pixal3d.py: a batch whose seq stands still this long is given up
RAM_BALLAST_TIMEOUT = 600.0     # locking 15 GiB takes seconds; a PC that pages hard takes minutes

# The golden cases (pixal3d_golden.py CASES): the same pictures and seeds, so a mesh can be held against them.
PICTURES = paths.PIXAL3D_DIR / "assets" / "images"
MV_EXAMPLE = paths.PIXAL3D_DIR / "assets" / "mv_images" / "example"
CASES: dict[str, dict[str, Any]] = {
    "sv1024": {"resolution": 1024, "seed": 11, "picture": PICTURES / "s_15_img.png"},
    "sv1536": {"resolution": 1536, "seed": 22, "picture": PICTURES / "21_img.png"},
    "mv1024": {"resolution": 1024, "seed": 33,
               "views": [(MV_EXAMPLE / "view00_azim000.png", 0), (MV_EXAMPLE / "view01_azim090.png", 90),
                         (MV_EXAMPLE / "view02_azim180.png", 180), (MV_EXAMPLE / "view03_azim270.png", 270)]},
    "mv1024b": {"resolution": 1024, "seed": 44,
                "views": [(MV_EXAMPLE / "view02_azim180.png", 0), (MV_EXAMPLE / "view03_azim270.png", 90)]},
    "big1536": {"resolution": 1536, "seed": 55, "picture": PICTURES / "12_img.png"},
}
SETS: dict[str, list[str]] = {
    "mixed4": ["mv1024", "sv1024", "mv1024b", "sv1536"],            # design note 10, 4.3.5, in request order
    "ten": ["sv1024", "mv1024"] + [f"1024:{name}:{100 + n}" for n, name in enumerate(
        ("0_img.png", "1_img.png", "4_img.png", "6_img.png", "7_img.png", "9_img.png", "11_img.png", "17_img.png"))],
}
PROJECT = "bench"               # <work>/bench/...: the one project of the bench's projects root


say = toolkit.teller("bench")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Start the measured Pixal3D launcher, send generations or batches, "
                                            "stop it.")
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--image", nargs="+", default=[], help="one picture, or several (a run picks one with @N)")
    p.add_argument("--runs", default="1024,1024",
                   help="one generation per entry: QUALITY or QUALITY@N (the N-th --image, from 1; default the first), "
                        "or mvQUALITY (the --views)")
    p.add_argument("--views", nargs="*", default=[], help="PATH:AZIMUTH for each view of a multi-view run")
    p.add_argument("--views-b", nargs="*", default=[], help="the views of the runs written mvbQUALITY")
    p.add_argument("--batch", action="store_true", help="the staged path: POST /spellcast/batch with --props")
    p.add_argument("--props", nargs="*", default=[],
                   help="--batch: the props, each a case (sv1024 sv1536 mv1024 mv1024b big1536), a set (mixed4, ten) "
                        "or QUALITY:PICTURE[:SEED]")
    p.add_argument("--batches", type=int, default=1, help="--batch: send the batch this many times to one server")
    p.add_argument("--wave-size", type=int, default=0, help="--batch: props per wave (0: all, as the hub sends)")
    p.add_argument("--purge-cache", action="store_true",
                   help="--batch: drop Pixal3D's network files from Windows' file cache before every batch")
    p.add_argument("--mesh-hashes", action="store_true",
                   help="--batch: SHA-256 of every decoded mesh, held against the golden ones by the report")
    p.add_argument("--resume-from", type=Path, default=None,
                   help="--batch: a folder of saved checkpoints (<prop id>/final.safetensors + final.json, as the "
                        "engine writes them after the texture is sampled) copied into every batch's work folder "
                        "before it is sent with resume on: the props run only their last part (the decode and the "
                        "GLB), for measuring that part alone in a few minutes. The checkpoints must come from a "
                        "batch of the same props, pictures (their sizes and change times), seeds and settings")
    p.add_argument("--ram-pc-gib", type=float, default=0.0,
                   help="hold page-locked RAM (ram_ballast.py) so the PC behaves like one with this many GiB")
    p.add_argument("--ram-leave-mib", type=int, default=0,
                   help="hold page-locked RAM so that this many MiB stay available at the start (what a 16 GiB PC "
                        "leaves the launcher beside Windows, the hub and a browser: about 11,000); on a developer "
                        "PC whose other programs hold more than such a PC's, --ram-pc-gib emulates a fuller one")
    p.add_argument("--ballast-leave", type=int, default=0,
                   help="emulate a smaller card: MiB of dedicated memory the server's process may hold in all")
    p.add_argument("--name", default=None, help="what every file's name starts with (pixal3d; staged with --batch)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--faces", type=int, default=300_000)
    p.add_argument("--texture", type=int, default=2048)
    p.add_argument("--port", type=int, default=7791)
    p.add_argument("--low-vram", choices=("on", "off"), default="on")
    p.add_argument("--first-job", choices=("single", "multi"), default="single")
    p.add_argument("--attn-backend", default="flash_attn")
    p.add_argument("--repo", type=Path, default=None,
                   help="the Pixal3D checkout the server runs from (default: code/pixal3d; a worktree under test)")
    p.add_argument("--step-sync", action="store_true")
    p.add_argument("--no-external", action="store_true", help="without the sampler process next to the thread")
    p.add_argument("--pause", type=float, default=8.0)
    p.add_argument("--warm-seconds", type=float, default=0.0,
                   help="idle seconds between the server's ready line and the first batch (the hub's warm start)")
    p.add_argument("--keep-glb", action="store_true")
    args = p.parse_args(argv)
    if args.name is None:
        args.name = "staged" if args.batch else "pixal3d"
    if args.batch and not args.props:
        p.error("--batch needs --props")
    if not args.batch and not args.image:
        p.error("--image is needed (or --batch with --props)")
    if args.batches < 1 or args.wave_size < 0:
        p.error("--batches must be 1 or more, --wave-size 0 or more")
    return args


# ---- the props of a batch, and the hub's body for them ---------------------------------------------------------------

def parse_props(entries: list[str], pictures: Path = PICTURES) -> list[dict[str, Any]]:
    """--props -> [{"id", "resolution", "seed", "picture": Path | None, "views": [(Path, azimuth)] | None}]: the
    sets expanded, every prop with an id the launcher accepts (letters, digits, _ and -), each id once."""
    expanded: list[Any] = []
    for entry in entries:
        if entry.lower().endswith(".json"):
            expanded += props_from_file(Path(entry))
            continue
        for part in (p.strip() for p in entry.split(",")):
            if part:
                expanded += SETS.get(part, [part])
    props: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for n, entry in enumerate(expanded, 1):
        if isinstance(entry, dict):
            prop = dict(entry)
        elif entry in CASES:
            case = CASES[entry]
            prop = {"id": entry, "resolution": case["resolution"], "seed": case["seed"],
                    "picture": case.get("picture"), "views": case.get("views")}
        else:
            quality, _, rest = entry.partition(":")
            if not quality.isdigit() or not rest:
                raise ValueError(f"--props: {entry!r} is neither a case ({', '.join(CASES)}), a set "
                                 f"({', '.join(SETS)}) nor QUALITY:PICTURE[:SEED]")
            seed = 0
            head, colon, tail = rest.rpartition(":")
            if colon and len(head) > 1 and tail.lstrip("-").isdigit():     # not the colon of a drive (C:\...)
                rest, seed = head, int(tail)
            picture = Path(rest)
            if not picture.is_absolute() and not picture.parent.parts:
                picture = pictures / picture
            stem = "".join(c if c.isalnum() or c in "_-" else "-" for c in picture.stem)[:40]
            prop = {"id": f"p{n}-{stem}", "resolution": int(quality), "seed": seed, "picture": picture,
                    "views": None}
        base = prop["id"]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            prop["id"] = f"{base}-{seen[base]}"
        props.append(prop)
    return props


def props_from_file(path: Path) -> list[dict[str, Any]]:
    """A JSON file's props -> parse_props' entries: [{"id", "resolution", "seed", "picture": Path | None,
    "views": [(Path, azimuth)] | None}]. Paths are relative to the file's folder; "id" defaults to
    p<n>-<picture stem>, "resolution" to 1024, "seed" to 0."""
    folder = path.resolve().parent
    listed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(listed, list):
        raise ValueError(f"--props: {path} must hold a JSON list of props")
    props = []
    for n, entry in enumerate(listed, 1):
        views = entry.get("views")
        picture = entry.get("picture")
        if views:
            views = [(folder / str(source), int(azimuth)) for source, azimuth in views]
            picture = None
        elif picture:
            picture = folder / str(picture)
        else:
            raise ValueError(f"--props: prop {n} of {path} names neither a picture nor views")
        stem = Path(views[0][0] if views else picture).stem
        default_id = "".join(c if c.isalnum() or c in "_-" else "-" for c in f"p{n}-{stem}")[:40]
        props.append({"id": str(entry.get("id") or default_id), "resolution": int(entry.get("resolution", 1024)),
                      "seed": int(entry.get("seed", 0)), "picture": picture, "views": views,
                      "geometry": str(entry.get("geometry", "pixal3d"))})
    return props


def project_dir(work: Path) -> Path:
    return work / PROJECT


def stage_inputs(props: list[dict[str, Any]], work: Path) -> list[dict[str, Any]]:
    """The pictures copied into the bench's project (the launcher takes files from its projects root only)."""
    staged = []
    for prop in props:
        folder = project_dir(work) / "input" / prop["id"]
        folder.mkdir(parents=True, exist_ok=True)
        copy = dict(prop)
        if prop.get("views"):
            copy["views"] = []
            for source, azimuth in prop["views"]:
                target = folder / Path(source).name
                shutil.copy2(source, target)
                copy["views"].append((target, azimuth))
        else:
            target = folder / Path(prop["picture"]).name
            shutil.copy2(prop["picture"], target)
            copy["picture"] = target
        staged.append(copy)
    return staged


def restore_checkpoints(saved: Path, work_dir: Path, prop_ids: list[str]) -> int:
    """Copy the saved checkpoints (<saved>/<prop id>/final.safetensors and final.json) of these props into the
    batch's work folder, where the engine looks for them when the batch is sent with resume on. Returns how many
    props got one. The engine itself verifies each (the request's fingerprint, size and SHA-256) and runs a prop
    with a valid one from its last part only."""
    restored = 0
    for prop_id in prop_ids:
        source = saved / prop_id
        if not (source / "final.safetensors").is_file() or not (source / "final.json").is_file():
            continue
        target = work_dir / prop_id
        target.mkdir(parents=True, exist_ok=True)
        for name in ("final.safetensors", "final.json"):
            shutil.copy2(source / name, target / name)
        restored += 1
    return restored


def batch_body(props: list[dict[str, Any]], work: Path, batch_id: str, wave_size: int = 0, faces: int = 300_000,
               texture: int = 2048, resume: bool = False, spill: bool = False) -> dict[str, Any]:
    """The body the hub posts to /spellcast/batch (adapters/pixal3d.py run_batch / _request / _start) for these
    props: the front view first, its picture as image_path; a partial GLB next to each prop's mesh/mesh.glb; the
    work folder <projects root>/<project>/temp/mesh-<batch id>; `spill` as the hub's pixal3d.allow_spill (off by
    default: a prop that does not fit the card fails instead of spilling into shared memory)."""
    project = project_dir(work)
    items = []
    for prop in props:
        mesh = project / "props" / prop["id"] / "mesh"
        # The prop's own folder, as the hub's projects have it: the launcher writes into <prop>/mesh and never
        # makes a prop folder itself (one that is gone was deleted by the artist).
        mesh.parent.mkdir(parents=True, exist_ok=True)
        views = None
        if prop.get("views"):
            ordered = sorted(prop["views"], key=lambda v: (0, 90, 180, 270).index(v[1]))
            views = [{"image_path": str(Path(path).resolve()), "rect": None, "azimuth": int(azimuth)}
                     for path, azimuth in ordered]
        items.append({"id": prop["id"],
                      "image_path": views[0]["image_path"] if views else str(Path(prop["picture"]).resolve()),
                      "seed": int(prop["seed"]), "resolution": int(prop["resolution"]),
                      "decimation_target": int(faces), "texture_size": int(texture),
                      "views": views, "views_dir": str((mesh / "views").resolve()) if views else None,
                      "out_path": str((mesh / f"mesh.{batch_id}.partial.glb").resolve()),
                      **({"geometry": prop["geometry"]} if prop.get("geometry", "pixal3d") != "pixal3d" else {})})
    return {"batch_id": batch_id, "work_dir": str((project / "temp" / f"mesh-{batch_id}").resolve()),
            "wave_size": int(wave_size), "resume": bool(resume), "spill": bool(spill), "items": items}


# ---- the card, HTTP, processes ---------------------------------------------------------------------------------------

card = toolkit.card                 # memory in use, utilisation, the driver and the processes on the card


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def get_json(url: str, timeout: float = 5.0) -> Any:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))      # never through a system proxy
    with opener.open(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, body: dict[str, Any], timeout: float) -> tuple[int, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except ValueError:
            return exc.code, None


def kill_tree(pid: int) -> None:
    if os.name != "nt":     # a Linux test pod: the server is a process group of its own (start_new_session)
        import signal
        for kill in (os.killpg, os.kill):
            try:
                kill(pid, signal.SIGKILL)
                return
            except OSError:
                continue
        return
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)


def purge_file_cache(folder: Path, min_bytes: int = 1 << 20) -> tuple[int, int]:
    """Every file of `folder` (the junction to models/pixal3d followed) larger than min_bytes opened once without
    buffering, which drops its pages from Windows' file cache (no admin rights needed). -> (files, bytes).
    On a Linux test pod the kernel is asked to drop each file's clean pages (posix_fadvise, no root needed)."""
    if os.name != "nt":
        files = total = 0
        seen: set[Path] = set()         # HuggingFace's snapshot links and their blobs: each file once
        for path in folder.rglob("*"):
            try:
                real = path.resolve()
                if real in seen or not real.is_file() or real.stat().st_size < min_bytes:
                    continue
                seen.add(real)
                fd = os.open(real, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except OSError:
                continue
            files += 1
            total += real.stat().st_size
        return files, total
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                wintypes.DWORD, wintypes.HANDLE]
    files = total = 0
    for path in folder.rglob("*"):
        try:
            if not path.is_file() or path.stat().st_size < min_bytes:
                continue
        except OSError:
            continue
        handle = k32.CreateFileW(str(path), 0x80000000, 0x7, None, 3, 0x20000000, None)   # GENERIC_READ, NO_BUFFERING
        if handle not in (None, wintypes.HANDLE(-1).value):
            k32.CloseHandle(handle)
            files += 1
            total += path.stat().st_size
    return files, total


def start_ram_ballast(args: argparse.Namespace, out: Path) -> subprocess.Popen[bytes] | None:
    """ram_ballast.py --pc-gib (or --leave-mib), watching this process; returns once it holds the RAM (None: it
    refused)."""
    ready = out / f"{args.name}.ramballast.json"
    stop = out / f"{args.name}.ramballast.stop"
    for stale in (ready, stop):
        stale.unlink(missing_ok=True)
    size = (["--leave-mib", str(args.ram_leave_mib)] if args.ram_leave_mib > 0
            else ["--pc-gib", f"{args.ram_pc_gib:g}"])
    command = [sys.executable, str(HERE / "ram_ballast.py"), *size,
               "--ready-file", str(ready), "--stop-file", str(stop), "--log", str(out / f"{args.name}.ramballast.csv"),
               "--parent-pid", str(os.getpid())]
    say("Starting the RAM ballast: " + subprocess.list2cmdline(command))
    log = open(out / f"{args.name}.ramballast.log", "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    started = time.monotonic()
    while not ready.is_file():
        if proc.poll() is not None:
            say(f"The RAM ballast did not start (exit code {proc.returncode}); see {log.name}.")
            return None
        if time.monotonic() - started > RAM_BALLAST_TIMEOUT:
            say("The RAM ballast did not hold its RAM in time; stopping it.")
            stop_ram_ballast(proc, stop)
            return None
        time.sleep(0.5)
    time.sleep(0.2)
    state = json.loads(ready.read_text(encoding="utf-8"))
    say(f"RAM ballast: {state.get('locked_mib'):,} MiB page-locked, available RAM now "
        f"{state.get('after', {}).get('available_mib'):,} MiB (verified: {state.get('verified')}).")
    return proc


def stop_ram_ballast(proc: subprocess.Popen[bytes], stop: Path) -> None:
    stop.write_text("stop", encoding="utf-8")
    try:
        proc.wait(60.0)
    except subprocess.TimeoutExpired:
        kill_tree(proc.pid)
    stop.unlink(missing_ok=True)


# ---- one batch -------------------------------------------------------------------------------------------------------

def run_batch(base: str, body: dict[str, Any], summary: dict[str, Any], proc: subprocess.Popen[bytes],
              keep: bool) -> bool:
    """POST the batch, poll it once a second until it ends; what happened into summary["batches"]. True when every
    prop was made."""
    batch_id = body["batch_id"]
    entry: dict[str, Any] = {"batch_id": batch_id, "wave_size": body["wave_size"], "resume": body["resume"],
                             "props": [{k: v for k, v in item.items() if k not in ("image_path", "out_path",
                                                                                   "views_dir")}
                                       | {"views": len(item["views"] or []) or 1} for item in body["items"]]}
    summary["batches"].append(entry)
    say(f"Batch {batch_id}: {len(body['items'])} props ({', '.join(item['id'] for item in body['items'])}), "
        f"wave size {body['wave_size']}...")
    sent = time.monotonic()
    try:
        status, answer = post_json(f"{base}/spellcast/batch", body, 60.0)
    except OSError as exc:
        status, answer = 0, {"detail": f"{type(exc).__name__}: {exc}"}
    entry["status"], entry["answer"] = status, answer
    if status != 202:
        say(f"Batch {batch_id}: HTTP {status}, {answer}")
        return False
    timeline: list[list[Any]] = []
    last_key: tuple[Any, ...] = ()
    last_seq, last_change = None, time.monotonic()
    slowest = 0.0
    record: dict[str, Any] = {}
    while True:
        time.sleep(POLL_INTERVAL)
        asked = time.monotonic()
        try:
            record = get_json(f"{base}/spellcast/batch/{batch_id}", timeout=30.0)
        except (OSError, ValueError):
            if proc.poll() is not None:
                say(f"The server stopped (exit code {proc.returncode}).")
                break
            continue
        slowest = max(slowest, time.monotonic() - asked)
        part = record.get("part") or {}
        key = (record.get("state"), part.get("key"), part.get("position"), record.get("current"),
               record.get("loading"))
        if key != last_key:
            last_key = key
            timeline.append([round(time.monotonic() - sent, 1), *key])
            if part.get("key"):
                say(f"  {part.get('label')} ({part.get('key')}): prop {part.get('position')} of {part.get('of')}"
                    + (f", {record.get('current')}" if record.get("current") else "")
                    + (f", loading the {record['loading']}" if record.get("loading") else ""))
        if record.get("seq") != last_seq:
            last_seq, last_change = record.get("seq"), time.monotonic()
        if record.get("state") != "running":
            break
        if time.monotonic() - last_change > STALL_TIMEOUT:
            say(f"Batch {batch_id} stood still for {STALL_TIMEOUT / 60:.0f} minutes; giving up on it.")
            break
    entry["wall_seconds"] = round(time.monotonic() - sent, 1)
    entry["slowest_poll_seconds"] = round(slowest, 2)
    entry["timeline"] = timeline
    entry["record"] = record
    glbs = {}
    for prop_id, item in (record.get("items") or {}).items():
        path = Path(((item.get("result") or {}).get("glb_path")) or "")
        glbs[prop_id] = path.stat().st_size if path.name and path.is_file() else 0
        if not keep and path.name and path.is_file():
            path.unlink()
    entry["glb_bytes"] = glbs
    if not keep:
        for item in body["items"]:
            if item["views_dir"]:
                shutil.rmtree(item["views_dir"], ignore_errors=True)
    states = {prop_id: item.get("state") for prop_id, item in (record.get("items") or {}).items()}
    told = []
    for prop_id, state in states.items():
        result = (record["items"][prop_id].get("result") or {}) if state == "done" else {}
        slow = f", slow in {', '.join(result['slow'])}" if result.get("slow") else ""
        told.append(f"{prop_id} {state}" + (f" ({result.get('seconds')} s{slow})" if state == "done" else ""))
    say(f"Batch {batch_id}: {record.get('state')} in {entry['wall_seconds']} s; " + "; ".join(told))
    return record.get("state") == "done" and all(state == "done" for state in states.values())


# ---- main ------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out, work = Path(args.out).resolve(), Path(args.work).resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[int, int]] = []                     # (quality, index of the picture; -1: --views, -2: --views-b)
    props: list[dict[str, Any]] = []
    pictures: list[Path] = []
    view_sets: dict[int, list[dict[str, Any]]] = {}      # -1 / -2 -> the request's "views", the front one first
    if args.batch:
        try:
            props = parse_props(args.props)
        except ValueError as exc:
            say(str(exc))
            return 2
        missing = [str(p) for prop in props for p in ([prop["picture"]] if prop["picture"] else
                                                      [v[0] for v in prop["views"]]) if not Path(p).is_file()]
        if missing:
            say(f"Pictures not found: {', '.join(missing)}")
            return 2
        props = stage_inputs(props, work)
    else:
        for part in (part.strip() for part in args.runs.split(",")):
            if part.startswith("mvb"):
                runs.append((int(part[3:]), -2))
            elif part.startswith("mv"):
                runs.append((int(part[2:]), -1))
            elif part:
                quality, _, which = part.partition("@")
                runs.append((int(quality), int(which or 1) - 1))
        if any(not -2 <= which < len(args.image) for _, which in runs):
            say("--runs names a picture that --image does not give.")
            return 2
        if (any(which == -1 for _, which in runs) and len(args.views) < 2) or (
                any(which == -2 for _, which in runs) and len(args.views_b) < 2):
            say("A multi-view run needs --views (mvb: --views-b) with two pictures or more.")
            return 2
        (work / "input").mkdir(parents=True, exist_ok=True)
        for source in args.image:
            pictures.append(work / "input" / Path(source).name)
            shutil.copy2(source, pictures[-1])
        for key, folder, entries in ((-1, "views", args.views), (-2, "views-b", args.views_b)):
            found: list[dict[str, Any]] = []
            for entry in entries:
                source, _, azimuth = entry.rpartition(":")
                target = work / "input" / folder / Path(source).name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                found.append({"image_path": str(target), "rect": None, "azimuth": int(azimuth)})
            view_sets[key] = sorted(found, key=lambda v: (0, 90, 180, 270).index(v["azimuth"]))
    if port_open(args.port):
        say(f"Port {args.port} is in use: stop what listens there, or pick another --port.")
        return 2
    if not args.batch and not ROUND_A.is_file():
        say(f"The generate mode starts {ROUND_A}, which this copy does not have (code/devtools is in a developer's "
            f"checkout only). --batch measures the staged path.")
        return 2

    ballast_proc = None
    if args.ram_pc_gib > 0 or args.ram_leave_mib > 0:
        ballast_proc = start_ram_ballast(args, out)
        if ballast_proc is None:
            return 2
    before = card()
    say(f"The card before the start: {before}")

    script = HERE / "pixal3d_staged_profiled.py" if args.batch else ROUND_A
    repo = args.repo.resolve() if args.repo else paths.PIXAL3D_DIR
    argv_server = [str(paths.venv_python(paths.PIXAL3D)), str(script),
                   "--repo", str(repo), "--port", str(args.port), "--low-vram", args.low_vram,
                   "--attn-backend", args.attn_backend, "--projects-root", str(work), "--first-job", args.first_job,
                   "--profile-out", str(out), "--profile-name", args.name]
    if args.step_sync and not args.batch:
        argv_server.append("--profile-step-sync")
    if not args.no_external:
        argv_server.append("--profile-external")
    if args.ballast_leave > 0:
        argv_server += ["--profile-ballast-leave", str(args.ballast_leave)]
    if args.batch and args.mesh_hashes:
        argv_server.append("--profile-mesh-hashes")
    log_path = out / f"{args.name}.bench.log"
    summary: dict[str, Any] = {"argv": argv_server, "cwd": str(repo), "card_before": before,
                               "started": time.strftime("%Y-%m-%d %H:%M:%S"), "ram_pc_gib": args.ram_pc_gib or None,
                               "ballast_leave_mib": args.ballast_leave or None}
    if args.batch:
        summary.update(mode="batch", props=[p["id"] for p in props], batches=[], purge_cache=args.purge_cache)
    else:
        summary.update(mode="generate", pictures=list(args.image), runs=[])
    summary_path = out / f"{args.name}.bench.json"

    def save() -> None:
        summary_path.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")

    say("Starting: " + subprocess.list2cmdline(argv_server))
    failed = False
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            log.write(subprocess.list2cmdline(argv_server) + "\n")
            log.flush()
            proc = subprocess.Popen(argv_server, cwd=str(repo), env=paths.clean_env(paths.PIXAL3D),
                                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                                    start_new_session=os.name != "nt")
            summary["pid"] = proc.pid
            save()
            base = f"http://127.0.0.1:{args.port}"
            try:
                started = time.monotonic()
                while True:
                    if proc.poll() is not None:
                        say(f"The server stopped while starting (exit code {proc.returncode}); see {log_path}.")
                        return 1
                    try:
                        if get_json(f"{base}/spellcast/health").get("ready") is True:
                            break
                    except (OSError, ValueError):
                        pass
                    if time.monotonic() - started > READY_TIMEOUT:
                        say(f"The server was not ready within {READY_TIMEOUT / 60:.0f} minutes.")
                        return 1
                    time.sleep(1.0)
                summary["ready_seconds"] = round(time.monotonic() - started, 1)
                summary["card_ready"] = card()
                save()
                say(f"Ready in {summary['ready_seconds']} s. The card: {summary['card_ready']}")
                time.sleep(args.pause + args.warm_seconds)
                if args.batch:
                    models = paths.PIXAL3D_DIR / ("models" if os.name == "nt" else "MODELS")   # Linux minds the case
                    for n in range(1, args.batches + 1):
                        if args.purge_cache:
                            files, size = purge_file_cache(models)
                            say(f"File cache: {files} files of Pixal3D's models dropped ({size / 1e9:.1f} GB).")
                        body = batch_body(props, work, f"b{n}", args.wave_size, args.faces, args.texture,
                                          resume=args.resume_from is not None)
                        if args.resume_from is not None:
                            restored = restore_checkpoints(args.resume_from, Path(body["work_dir"]),
                                                           [item["id"] for item in body["items"]])
                            say(f"Checkpoints of {restored} of {len(body['items'])} props put into the work "
                                f"folder; the batch resumes them at their last part.")
                        if not run_batch(base, body, summary, proc, args.keep_glb):
                            failed = True
                        save()
                        if proc.poll() is not None:
                            break
                        time.sleep(args.pause)
                else:
                    failed = not generate_runs(args, base, work, runs, pictures, view_sets, summary, proc, save)
                summary["card_after_runs"] = card()
            finally:
                if proc.poll() is None:
                    kill_tree(proc.pid)
                    try:
                        proc.wait(20.0)
                    except subprocess.TimeoutExpired:
                        say(f"The server (PID {proc.pid}) did not stop: end it by hand.")
                time.sleep(2.0)
                summary["card_after_stop"] = card()
                summary["port_open_after_stop"] = port_open(args.port)
                summary["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
                save()
                say(f"Stopped. The card: {summary['card_after_stop']}; port {args.port} "
                    f"{'STILL OPEN' if summary['port_open_after_stop'] else 'free'}.")
    finally:
        if ballast_proc is not None:
            stop_ram_ballast(ballast_proc, out / f"{args.name}.ramballast.stop")
            say("The RAM ballast let go of its RAM.")
    if args.batch:
        return 1 if failed or len(summary["batches"]) != args.batches else 0
    return 1 if failed or len(summary["runs"]) != len(runs) else 0


def generate_runs(args: argparse.Namespace, base: str, work: Path, runs: list[tuple[int, int]],
                  pictures: list[Path], view_sets: dict[int, list[dict[str, Any]]], summary: dict[str, Any],
                  proc: subprocess.Popen[bytes], save: Any) -> bool:
    """The reference path: one POST /spellcast/generate per run. True when every run was answered with 200."""
    ok = True
    for n, (resolution, which) in enumerate(runs, 1):
        multi = which < 0
        views = view_sets[which] if multi else []
        picture = Path(views[0]["image_path"]) if multi else pictures[which]
        glb = work / f"run-{n}" / f"mesh.run{n}.partial.glb"
        glb.parent.mkdir(parents=True, exist_ok=True)
        views_dir = glb.parent / "views"
        body = {"image_path": str(picture), "seed": args.seed, "resolution": resolution,
                "decimation_target": args.faces, "texture_size": args.texture,
                "views": views if multi else None, "views_dir": str(views_dir) if multi else None,
                "out_path": str(glb)}
        what = f"{len(views)} views, the front one {picture.name}" if multi else picture.name
        say(f"Run {n} of {len(runs)}: {what}, quality {resolution}, seed {args.seed}, {args.faces} "
            f"faces, texture {args.texture}...")
        done = threading.Event()
        stages: list[str] = []

        def poll() -> None:            # as the hub: /spellcast/progress once a second
            while not done.wait(POLL_INTERVAL):
                try:
                    stage = str(get_json(f"{base}/spellcast/progress").get("stage") or "")
                except (OSError, ValueError):
                    continue
                if stage and (not stages or stages[-1] != stage):
                    stages.append(stage)
                    say(f"  run {n}: {stage}")

        poller = threading.Thread(target=poll, daemon=True)
        poller.start()
        sent = time.monotonic()
        try:
            status, answer = post_json(f"{base}/spellcast/generate", body, GENERATE_TIMEOUT)
        except OSError as exc:
            status, answer = 0, {"detail": f"{type(exc).__name__}: {exc}"}
        wall = time.monotonic() - sent
        done.set()
        poller.join(3.0)
        size = glb.stat().st_size if glb.is_file() else 0
        run = {"run": n, "resolution": resolution, "picture": picture.name,
               "views": len(views) if multi else 1, "status": status,
               "wall_seconds": round(wall, 1),
               "answer": answer, "glb_bytes": size, "request": {k: v for k, v in body.items()
                                                                if k not in ("image_path", "out_path")}}
        summary["runs"].append(run)
        save()
        say(f"Run {n}: HTTP {status}, wall {wall:.1f} s, GLB {size / 1e6:.1f} MB, answer {answer}")
        if not args.keep_glb and glb.is_file():
            glb.unlink()
        if multi and not args.keep_glb:
            shutil.rmtree(views_dir, ignore_errors=True)       # the framed views the launcher saved
        if status != 200:
            ok = False
            if proc.poll() is not None:
                say(f"The server stopped (exit code {proc.returncode}).")
                break
        time.sleep(args.pause)
    return ok


if __name__ == "__main__":
    sys.exit(main())
