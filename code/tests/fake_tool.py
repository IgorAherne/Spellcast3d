"""A stand-in tool server for the hub tests (stdlib only).

python fake_tool.py --port P [--delay S] [--crash-after S] [--exit-now] [--never-healthy]
                    [--ignore-break] [--pidfile F] [--child-pidfile F] [--bake-seconds S] [--no-multiview] [--no-hi3d]
                    [--part-seconds S] [--load-seconds S] [--batch-die-after S] [--batch-mute S]
                    [--batch-deaf-after S] [--batch-lose-cancels N]

Besides /health and /echo it answers like launchers/imesh_server.py: /spellcast/health, the bake task
API (/spellcast/bake...) and /spellcast/import, writing small placeholder files. A low-poly file or a
model file whose text contains "FAKE_ERROR:<message>" makes the bake / import fail with that message.
GET /fake/bakes and /fake/imports list what was asked for.

It also answers like launchers/pixal3d_server.py (class Batches): POST /spellcast/batch takes all the props of
a job, GET /spellcast/batch/{id} says where each one is, POST /spellcast/batch/{id}/cancel stops it. Like the
real one it takes the props step by step (PARTS; --part-seconds per prop and step, --load-seconds "reading a
model" before some steps), those with two or more views first, `wave_size` of them at a time; a prop rests
between its steps; its placeholder GLB is written to out_path in the last step ("finish", which a cancel does
not interrupt). With "resume", a prop whose texture step was done in work_dir (by a process that died) does
only the last step. With --no-multiview, /spellcast/health says "multiview": false and a prop with two or
more views fails with the kind "mv_missing" (so does one whose front picture contains "FAKE_MV_MISSING"); with
--no-hi3d a prop whose "geometry" is "hi3d" fails with the kind "missing_file", as the launcher's engine does. A
picture whose text contains "FAKE_ERROR:<message>" fails its prop with that message, "FAKE_OOM" with the kind
"oom"; "FAKE_RETRY" answers ss_retries 2 and a warning, "FAKE_SPILL" a result made slowly ("slow"),
"FAKE_NO_GLB" says done without writing the file; "FAKE_STALL" stands still at that prop for good (a cancel
is taken, not obeyed); "FAKE_BROKEN" fails the whole batch at that prop with the kind "restart"; "FAKE_DIE"
ends the process (exit code 3) as soon as a poll has shown that prop as the one worked on in the step "lr",
"FAKE_SAVE_DIE" in the last step (the others' textures are done by then); "FAKE_CRASH_AFTER_SAVE" ends it right
after that prop's model is saved and the next prop began its turn, with no poll answered in between (the hub's
last answer still shows the saved prop on the card); "FAKE_SLOW_SAVE" stands still for SLOW_SAVE seconds at the
start of its last step (a long decode, which a cancel does not interrupt). Like the launcher it keeps the prop
on the card and the last one made in work_dir/on_card.json, and a prop whose folder is gone (the artist deleted
it) fails at the next of the steps whose hooks look (CHECKED) with the kind "input", and its folder is not made
again; "FAKE_OLD_SAVE" does neither (a launcher of before that check). --batch-die-after S ends the process
S seconds into a batch, --batch-mute S answers no poll in a batch's first S seconds (they wait),
--batch-deaf-after S stops listening S seconds into a batch while the process lives on (hung: its batch stands
still), --batch-lose-cancels N
drops the first N cancel requests unanswered. GET /fake/batches lists the batch requests, GET /fake/cancels
counts the cancels {"received", "lost"}.

POST /spellcast/retopo answers like the LATO.2 / TriFlow launchers ({"mesh_path", "out_path", "seed", ...} ->
a triangle OBJ without UVs, {"obj_path", "vertices", "faces", "seconds", "warning"}; FAKE_ERROR / FAKE_OOM in the
mesh as above; each takes --retopo-seconds), or like Instant Meshes' ({"src", "dst", ...} -> an OBJ with UVs).
POST /spellcast/unwrap answers like Instant Meshes' ({"src", "dst", "uv_leniency"} -> dst: src's text plus a
"vt" line, {"vertices", "faces", "has_uv", "warning", "seconds"}); "FAKE_NO_UV" in src answers has_uv false and
a warning, FAKE_ERROR or FAKE_UNWRAP_ERROR:<message> fails with 422 (a LATO.2 / TriFlow retopo copies the mesh's
"#" lines into its OBJ, so these can come from the prop's mesh). GET /fake/retopos and /fake/unwraps list those
requests.

POST /spellcast/generate answers like launchers/qwen_server.py (Qwen-Image 2.1): it writes
batch_count small PNGs into out_dir (RGBA when "transparent") and answers {"images", "seeds", "seconds",
"warning"}; seeds count up from the body's seed (777 for -1). In the prompt, "FAKE_ERROR:<message>" fails with that
message, "FAKE_OOM" with the kind "oom", "FAKE_RESTART" with the kind "restart", "FAKE_WARN" answers a warning,
"FAKE_CRASH" ends the process at once with exit code 3 (a CUDA error inside ggml: the connection drops unanswered).
Like stable-diffusion.cpp it draws every picture first and decodes them all at the end: "FAKE_SLOW" takes a second
per picture to draw, and POST /spellcast/cancel meanwhile keeps none (the kind "cancelled"); "FAKE_LONG_DECODE" takes
1.5 seconds per picture to decode, and a cancel then keeps the pictures decoded so far (with the warning "Stopped after
N of M pictures."). GET /spellcast/progress follows it; GET /fake/qwen lists those requests.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import struct
import subprocess
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MAPS = (("basecolor", "Base colour", "srgb"), ("normal", "Normal", "linear"), ("orm", "ORM", "linear"))


def tiny_png(rgba: bool, shade: int) -> bytes:
    """A real 2x2 PNG: RGBA with a transparent corner, or RGB."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    pixel = bytes([shade % 256, 80, 160] + ([255] if rgba else []))
    clear = bytes([0, 0, 0, 0]) if rgba else pixel
    rows = b"".join(b"\0" + first + pixel for first in (clear, pixel))
    header = struct.pack(">IIBBBBB", 2, 2, 8, 6 if rgba else 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def fake_error(path: Path) -> str | None:
    text = path.read_bytes().decode("utf-8", "replace")
    return text.split("FAKE_ERROR:", 1)[1].strip() if "FAKE_ERROR:" in text else None


class Bakes:
    """The bake tasks: each one takes `seconds`, then writes its files (checked when asked for)."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.tasks: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def start(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        high, low, out_root = Path(body["high"]), Path(body["low"]), Path(body["out_root"])
        if not high.is_file() or not low.is_file():
            return 400, {"detail": "The high-poly mesh or the low-poly model does not exist."}
        with self.lock:
            if any(t["state"] == "running" for t in self.tasks.values()):
                return 409, {"detail": "Instant Meshes is already baking."}
            number = 1 + sum(1 for t in self.tasks.values() if t["out_root"] == out_root)
            task_id = f"t{len(self.tasks) + 1}"
            self.tasks[task_id] = {"id": task_id, "request": body, "out_root": out_root, "state": "running",
                                   "started": time.monotonic(), "number": number, "error": fake_error(low)}
        return 200, {"task_id": task_id}

    def status(self, task_id: str) -> tuple[int, dict[str, Any]]:
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                return 404, {"detail": "This bake is not known."}
            fraction = (time.monotonic() - task["started"]) / self.seconds
            if task["state"] == "running" and fraction >= 1:
                self._finish(task)
            if task["state"] == "running":
                return 200, {"state": "running", "fraction": round(fraction, 3),
                             "message": f"Fake baking {fraction:.0%}"}
            return 200, {"state": task["state"], "fraction": 1.0, "message": task["state"],
                         **{key: task[key] for key in ("result", "detail") if key in task}}

    def cancel(self, task_id: str) -> tuple[int, dict[str, Any]]:
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                return 404, {"detail": "This bake is not known."}
            if task["state"] == "running":
                task["state"] = "cancelled"
            return 200, {"state": task["state"], "fraction": 0.0, "message": "Cancelled."}

    def listing(self) -> list[dict[str, Any]]:
        with self.lock:
            return [{"id": t["id"], "state": t["state"], "request": t["request"]} for t in self.tasks.values()]

    @staticmethod
    def _finish(task: dict[str, Any]) -> None:
        if task["error"]:
            task["state"], task["detail"] = "error", task["error"]
            return
        folder = task["out_root"] / f"bake_{task['number']:03d}"
        settings = task["request"]["settings"]

        def write(rel: str, data: bytes) -> str:
            path = folder / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return str(path)

        maps = [{"id": mid, "label": label, "colorspace": space,
                 "path": write(f"maps/{mid}.png", b"\x89PNG\r\n\x1a\n" + mid.encode())} for mid, label, space in MAPS]
        obj = None
        if settings.get("export_obj"):
            write("export/projection.mtl", b"newmtl projection\nmap_Kd ../maps/basecolor.png\n")
            obj = write("export/projection.obj", b"mtllib projection.mtl\nv 0 0 0\n")
        task["result"] = {"dir": str(folder), "glb": write("projection.glb", b"glTF-baked"),
                          "fbx": write("projection.fbx", b"Kaydara FBX Binary") if settings.get("export_fbx") else None,
                          "obj": obj, "maps": maps,
                          "stats": {"seconds": 1.5, "resolution": settings["texture_size"], "coverage": 0.8,
                                    "miss_fraction": 0.0, "warnings": []}}
        task["state"] = "done"


# Pixal3D's steps as its launcher reports them: (key, the artist's words, where a prop's bar starts and ends).
PARTS = (("cutout", "Removing the background", 0.00, 0.03), ("camera", "Estimating the camera angle", 0.03, 0.05),
         ("features", "Reading the picture", 0.05, 0.06), ("ss", "Blocking out the shape", 0.06, 0.15),
         ("cond_lr", "Preparing the shape", 0.15, 0.16), ("lr", "Sculpting the shape", 0.16, 0.29),
         ("upsample", "Sculpting the shape", 0.29, 0.30), ("cond_hr", "Preparing the details", 0.30, 0.32),
         ("hr", "Refining shape details", 0.32, 0.55), ("tex", "Painting the texture", 0.55, 0.72),
         ("finish", "Building the 3D mesh", 0.72, 1.00))
LOADS = {"ss": "shape model", "hr": "detail model", "tex": "texture model"}    # read from disk before these steps
PART_STEPS = 4
RESTING = "Waiting for the next step"
MARKS = ("FAKE_OOM", "FAKE_RETRY", "FAKE_SPILL", "FAKE_NO_GLB", "FAKE_STALL", "FAKE_DIE", "FAKE_SAVE_DIE",
         "FAKE_BROKEN", "FAKE_CRASH_AFTER_SAVE", "FAKE_SLOW_SAVE", "FAKE_OLD_SAVE")
ON_CARD = "on_card.json"    # launchers/pixal3d_batch.py: the prop on the card and the last one made, in work_dir
CHECKED = ("cutout", "camera", "ss", "finish")      # the parts whose hooks fail a prop that was deleted
PROP_DELETED = "This prop was deleted while its 3D model was being made."
SLOW_SAVE = 2.0             # FAKE_SLOW_SAVE: seconds without a word at the start of the prop's "finish"


class Batches:
    """Pixal3D's batches: one at a time, each run by a thread of its own; the last one stays readable."""

    def __init__(self, args: argparse.Namespace, deafen: Any):
        self.args, self.deafen = args, deafen
        self.lock = threading.Lock()
        self.batch: dict[str, Any] | None = None
        self.requests: list[dict[str, Any]] = []
        self.cancels = {"received": 0, "lost": 0}
        self.die_at: str | None = None          # the process ends once a poll has shown this prop as worked on
        self.frozen = threading.Event()         # --batch-deaf-after: the batch stands still with the server

    # ---- the routes ----
    def start(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        self.requests.append(body)
        items = body.get("items") or []
        ids = [item.get("id") for item in items]
        work = Path(str(body.get("work_dir") or ""))
        if not items or len(set(ids)) != len(ids):
            return 400, {"detail": "The batch has no props, or one of them twice.", "kind": "input"}
        if work.parent.name != "temp" or not work.name.startswith("mesh-") or not work.is_dir():
            return 400, {"detail": f"This is no work folder of a job: {work}.", "kind": "input"}
        with self.lock:
            if self.batch is not None and self.batch["state"] == "running":
                return 409, {"detail": "Pixal3D is still working on another batch.", "kind": "busy",
                             "batch_id": self.batch["batch_id"]}
            entries = {item["id"]: self._entry(item, work, body.get("resume") is True) for item in items}
            order = sorted(ids, key=lambda i: len(entries[i]["request"].get("views") or []) < 2)    # views first
            self.batch = {"batch_id": body["batch_id"], "state": "running", "cancelling": False, "seq": 1,
                          "started": time.monotonic(), "part": None, "current": None, "loading": None,
                          "items": entries, "order": order, "error": None, "cancel": threading.Event(),
                          "wave": int(body.get("wave_size") or 0) or len(order), "work": work, "finished": None,
                          "marked": ()}
            batch = self.batch
            self._mark(batch)
        threading.Thread(target=self._run, args=(batch,), daemon=True).start()
        for seconds, action in ((self.args.batch_die_after, self._die), (self.args.batch_deaf_after, self._deafen)):
            if seconds is not None:
                timer = threading.Timer(seconds, action)
                timer.daemon = True
                timer.start()
        return 202, {"batch_id": batch["batch_id"], "order": order,
                     "parts": [{"key": key, "label": label, "start": start, "end": end}
                               for key, label, start, end in PARTS]}

    def status(self, batch_id: str) -> tuple[int, dict[str, Any]]:
        batch = self._known(batch_id)
        if batch is None:
            return 404, {"detail": "This batch is not known.", "kind": "input"}
        wait = batch["started"] + (self.args.batch_mute or 0.0) - time.monotonic()
        if wait > 0:
            time.sleep(wait)        # the poll that asked has given up by now
        with self.lock:
            keys = ("state", "part", "parts_done", "fraction", "message", "step", "total", "result", "error")
            return 200, {"batch_id": batch["batch_id"], "state": batch["state"], "cancelling": batch["cancelling"],
                         "seq": batch["seq"], "seconds": round(time.monotonic() - batch["started"], 1),
                         "part": dict(batch["part"]) if batch["part"] else None, "current": batch["current"],
                         "loading": batch["loading"], "error": batch["error"],
                         "items": {prop_id: {key: entry[key] for key in keys if key in entry}
                                   for prop_id, entry in batch["items"].items()}}

    def cancel(self, batch_id: str) -> tuple[int, dict[str, Any]] | None:
        """None: this cancel is one of the first --batch-lose-cancels, dropped without an answer."""
        batch = self._known(batch_id)
        if batch is None:
            return 404, {"detail": "This batch is not known.", "kind": "input"}
        with self.lock:
            self.cancels["received"] += 1
            if self.cancels["lost"] < self.args.batch_lose_cancels:
                self.cancels["lost"] += 1
                return None
            if batch["state"] == "running" and not batch["cancelling"]:
                batch["cancelling"] = True
                batch["seq"] += 1
            batch["cancel"].set()
        return 200, {"cancelling": True}

    def poll_answered(self, answer: dict[str, Any]) -> None:
        """After a poll's answer went out: FAKE_DIE ends the process once the hub has seen its prop at work."""
        if self.die_at is not None and answer.get("current") == self.die_at:
            time.sleep(0.1)         # let the answer leave
            self._die()

    # ---- the work ----
    def _known(self, batch_id: str) -> dict[str, Any] | None:
        with self.lock:
            return self.batch if self.batch is not None and self.batch["batch_id"] == batch_id else None

    def _entry(self, item: dict[str, Any], work: Path, resume: bool) -> dict[str, Any]:
        """A prop as the batch starts: waiting, or failed already for what is wrong with its request."""
        entry: dict[str, Any] = {"state": "waiting", "part": None, "parts_done": 0, "fraction": 0.0, "message": "",
                                 "step": 0, "total": 0, "request": item, "marks": set(), "fails": None,
                                 "folder": work / item["id"], "resumed": False}
        views = item.get("views") or []
        image = Path(item["image_path"])
        if not image.is_file() or not all(Path(v["image_path"]).is_file() for v in views):
            return {**entry, "state": "failed",
                    "error": {"detail": f"The input image was not found: {image}.", "kind": "input"}}
        text = image.read_bytes().decode("utf-8", "replace")
        if len(views) >= 2 and (self.args.no_multiview or "FAKE_MV_MISSING" in text):
            return {**entry, "state": "failed",
                    "error": {"detail": "The multi-view models are not downloaded.", "kind": "mv_missing"}}
        if item.get("geometry") == "hi3d" and self.args.no_hi3d:
            return {**entry, "state": "failed",
                    "error": {"detail": "Hi3D's networks are not installed.", "kind": "missing_file"}}
        entry["marks"] = {mark for mark in MARKS if mark in text}
        entry["fails"] = fake_error(image)
        entry["resumed"] = resume and (entry["folder"] / "checkpoint").is_file()
        if entry["resumed"]:
            entry.update(fraction=PARTS[-1][2], parts_done=len(PARTS) - 1)
        else:       # like the real one: what a stopped process left of the prop is cleared
            entry["folder"].mkdir(parents=True, exist_ok=True)
            (entry["folder"] / "checkpoint").unlink(missing_ok=True)
            (entry["folder"] / "state.bin").write_bytes(b"\0" * 64)
        return entry

    def _die(self) -> None:
        print("crashing now: CUDA error (fake)", flush=True)
        os._exit(3)

    def _deafen(self) -> None:
        self.frozen.set()
        self.deafen()

    def _run(self, batch: dict[str, Any]) -> None:
        items, order, cancel = batch["items"], batch["order"], batch["cancel"]

        def change(entry: dict[str, Any] | None = None, values: dict[str, Any] | None = None, **of_batch: Any) -> None:
            """One change of the record: a prop's entry, the batch's own fields, or both."""
            if self.frozen.is_set():            # a hung process: nothing moves any more
                threading.Event().wait()
            with self.lock:
                if entry is not None:
                    entry.update(values or {})
                batch.update(of_batch)
                batch["seq"] += 1
                self._mark(batch)

        def cancelled() -> bool:
            if not cancel.is_set():
                return False
            with self.lock:
                for entry in items.values():
                    if entry["state"] not in ("done", "failed"):
                        entry.update(state="cancelled", message="")
                batch.update(state="cancelled", part=None, current=None, loading=None)
                batch["seq"] += 1
            return True

        for first in range(0, len(order), batch["wave"]):
            wave = order[first:first + batch["wave"]]
            for index, (key, label, start, end) in enumerate(PARTS):
                todo = [prop_id for prop_id in wave if items[prop_id]["state"] in ("waiting", "resting")
                        and (key == "finish" or not items[prop_id]["resumed"])]
                if not todo:
                    continue
                part = {"key": key, "index": index + 1, "count": len(PARTS), "label": label, "position": 1,
                        "of": len(todo)}
                if key in LOADS and self.args.load_seconds:
                    change(part=part, current=None, loading=LOADS[key])
                    if cancel.wait(self.args.load_seconds) and cancelled():
                        return
                    change(loading=None)
                for position, prop_id in enumerate(todo, 1):
                    entry = items[prop_id]
                    if cancelled():
                        return
                    change(entry, {"state": "working", "part": key, "step": 0, "total": PART_STEPS, "fraction": start,
                                   "message": label}, part={**part, "position": position}, current=prop_id)
                    if key == ("finish" if "FAKE_SAVE_DIE" in entry["marks"] else "lr" if "FAKE_DIE" in entry["marks"]
                               else None):
                        self.die_at = prop_id
                        threading.Event().wait()
                    if key == "lr" and "FAKE_STALL" in entry["marks"]:
                        threading.Event().wait()
                    if key == "lr" and "FAKE_BROKEN" in entry["marks"]:
                        change(state="failed", error={"detail": "The graphics card stopped answering.",
                                                      "kind": "restart"})
                        return
                    if key == "finish" and "FAKE_SLOW_SAVE" in entry["marks"]:
                        time.sleep(SLOW_SAVE)       # a long decode: native code holds on, nothing changes
                    error = None
                    if key in CHECKED and self._gone(entry):
                        error = {"detail": PROP_DELETED, "kind": "input"}
                    elif key == "ss" and entry["fails"]:
                        error = {"detail": entry["fails"], "kind": "error"}
                    elif key == "hr" and "FAKE_OOM" in entry["marks"]:
                        error = {"detail": "CUDA out of memory", "kind": "oom"}
                    if error:
                        change(entry, {"state": "failed", "error": error, "message": ""}, current=None)
                        continue
                    for step in range(1, PART_STEPS + 1):
                        if key != "finish" and cancelled():     # a mesh that is being saved is saved
                            return
                        time.sleep(self.args.part_seconds / PART_STEPS)
                        change(entry, {"step": step, "fraction": round(start + (end - start) * step / PART_STEPS, 4),
                                       "message": f"{label} · step {step} of {PART_STEPS}"})
                    if key == "tex":
                        (entry["folder"] / "checkpoint").write_bytes(b"latents")
                    if key != "finish":
                        change(entry, {"state": "resting", "parts_done": index + 1, "message": RESTING}, current=None)
                        continue
                    saved = self._saved(entry)
                    made = {"id": prop_id, "result": saved["result"]} if saved["state"] == "done" else batch["finished"]
                    if "FAKE_CRASH_AFTER_SAVE" not in entry["marks"]:
                        change(entry, saved, current=None, finished=made)
                        continue
                    # Saved; the next prop takes its turn; the process dies - and no poll is answered in between,
                    # so the hub's last answer still has this prop on the card.
                    with self.lock:
                        entry.update(saved)
                        batch.update(current=None, finished=made)
                        self._mark(batch)
                        after = todo[position:]
                        if after:
                            items[after[0]].update(state="working", part=key, step=0, message=label)
                            batch.update(current=after[0])
                            self._mark(batch)
                        self._die()
        if not cancelled():
            change(state="done", part=None, current=None)

    @staticmethod
    def _gone(entry: dict[str, Any]) -> bool:
        """The prop's folder (out_path's <prop>/mesh/..) is gone: deleted by the artist. FAKE_OLD_SAVE does not look,
        as a launcher of before the check, and makes the folder again when it saves."""
        if "FAKE_OLD_SAVE" in entry["marks"]:
            return False
        return not Path(entry["request"]["out_path"]).parent.parent.is_dir()

    @staticmethod
    def _mark(batch: dict[str, Any]) -> None:
        """As the launcher does: the prop on the card and the last one made, in the work folder, whenever either
        changes (the lock is held)."""
        key = (batch["current"], (batch["finished"] or {}).get("id"))
        if key == batch["marked"]:
            return
        batch["marked"] = key
        path = batch["work"] / ON_CARD
        spare = path.with_name(path.name + ".tmp")
        note = {"batch_id": batch["batch_id"], "current": batch["current"], "finished": batch["finished"]}
        try:
            spare.write_text(json.dumps(note), encoding="utf-8")
            for attempt in range(10):
                try:
                    os.replace(spare, path)
                    return
                except PermissionError:     # the hub reads it this very moment
                    time.sleep(0.02)
        except OSError:
            pass                            # the job is over and its work folder gone

    def _saved(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Write the prop's placeholder GLB -> what its entry says from then on. Its mesh folder is made again if
        it went, never the prop's folder (but with FAKE_OLD_SAVE)."""
        request, marks = entry["request"], entry["marks"]
        views = request.get("views") or []
        if "FAKE_NO_GLB" not in marks:
            out = Path(request["out_path"])
            try:
                out.parent.mkdir(parents="FAKE_OLD_SAVE" in marks, exist_ok=True)
            except FileNotFoundError:
                return {"state": "failed", "error": {"detail": PROP_DELETED, "kind": "input"}, "message": ""}
            out.write_bytes(b"glTF-generated")
        retried = "FAKE_RETRY" in marks
        return {"state": "done", "parts_done": len(PARTS), "fraction": 1.0, "message": "", "result": {
            "glb_path": request["out_path"], "seconds": 0.1, "seed": request["seed"] if request["seed"] >= 0 else 1234,
            "mode": "multi" if len(views) >= 2 else "single", "views": max(1, len(views)),
            "warning": "The shape needed 3 tries." if retried else None, "ss_retries": 2 if retried else 0,
            "resolution": request["resolution"], "slow": ["finish"] if "FAKE_SPILL" in marks else [],
            "resumed": entry["resumed"]}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--delay", type=float, default=0.0, help="seconds before the server listens")
    p.add_argument("--crash-after", type=float, default=None, help="exit with code 3 this long after listening")
    p.add_argument("--exit-now", action="store_true", help="exit with code 2 before listening")
    p.add_argument("--never-healthy", action="store_true", help="/health always answers 503")
    p.add_argument("--ignore-break", action="store_true", help="ignore CTRL_BREAK_EVENT")
    p.add_argument("--pidfile", default=None, help="write this interpreter's pid here")
    p.add_argument("--child-pidfile", default=None, help="start a sleeping grandchild, write its pid here")
    p.add_argument("--bake-seconds", type=float, default=1.0, help="how long a fake bake takes")
    p.add_argument("--no-multiview", action="store_true", help="Pixal3D's multi-view models are missing")
    p.add_argument("--no-hi3d", action="store_true", help="Hi3D's networks are missing")
    p.add_argument("--retopo-seconds", type=float, default=0.0, help="how long a LATO.2 / TriFlow retopo takes")
    p.add_argument("--part-seconds", type=float, default=0.0, help="Pixal3D batch: one prop in one step")
    p.add_argument("--load-seconds", type=float, default=0.0, help="Pixal3D batch: reading a model before a step")
    p.add_argument("--batch-die-after", type=float, default=None, help="exit with code 3 this long into a batch")
    p.add_argument("--batch-mute", type=float, default=0.0, help="answer no poll in a batch's first seconds")
    p.add_argument("--batch-deaf-after", type=float, default=None,
                   help="stop listening this long into a batch; the process lives on")
    p.add_argument("--batch-lose-cancels", type=int, default=0, help="drop the first cancels unanswered")
    args = p.parse_args()

    if args.pidfile:
        with open(args.pidfile, "w") as f:
            f.write(str(os.getpid()))
    if args.child_pidfile:
        code = "import os,sys,time; open(sys.argv[1],'w').write(str(os.getpid())); time.sleep(300)"
        subprocess.Popen([sys.executable, "-c", code, args.child_pidfile],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def on_break(signum: int, frame: object) -> None:
        if args.ignore_break:
            print("ignoring CTRL_BREAK", flush=True)
            return
        print("got CTRL_BREAK, exiting", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGBREAK, on_break)

    print(f"fake tool starting on port {args.port}", flush=True)
    sys.stdout.write("loading 10%\rloading 50%\rloading 100%\n")
    sys.stdout.flush()
    print("stderr line", file=sys.stderr, flush=True)
    if args.exit_now:
        print("fatal: model file missing", flush=True)
        sys.exit(2)
    end = time.monotonic() + args.delay
    while time.monotonic() < end:
        time.sleep(0.05)

    bakes = Bakes(args.bake_seconds)
    imports: list[dict[str, Any]] = []

    def import_model(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        src, dst = Path(body["src"]), Path(body["dst"])
        imports.append(body)
        if not src.is_file():
            return 400, {"detail": "The model file does not exist any more."}
        error = fake_error(src)
        if error:
            return 422, {"detail": error}
        dst.write_bytes(b"glTF-imported")
        textures = sorted(p.name for p in src.parent.rglob("*.png"))
        return 200, {"glb": str(dst), "seconds": 0.2, "vertices": 8, "faces": 12, "warnings": [],
                     "materials": [{"name": "Material", "maps": [
                         {"key": name, "file": name, "role": "basecolor", "label": "Base colour", "confidence": 0.9,
                          "y_convention": None, "y_confident": False, "evidence": ["fake"]} for name in textures]}]}

    qwen_jobs: list[dict[str, Any]] = []
    qwen_state: dict[str, Any] = {"progress": None, "cancel": threading.Event()}

    def qwen_generate(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        qwen_jobs.append(body)
        out_dir, prompt = Path(body["out_dir"]), str(body.get("prompt") or "")
        if not out_dir.is_dir():
            return 400, {"detail": f"The output folder does not exist: {out_dir}.", "kind": "input"}
        missing = [ref for ref in body.get("ref_images") or [] if not Path(ref).is_file()]
        if missing:
            return 400, {"detail": f"The example picture was not found: {missing[0]}.", "kind": "input"}
        if "FAKE_ERROR:" in prompt:
            return 500, {"detail": prompt.split("FAKE_ERROR:", 1)[1].split(",")[0].strip(), "kind": "error"}
        if "FAKE_OOM" in prompt:
            return 500, {"detail": "CUDA out of memory", "kind": "oom"}
        if "FAKE_RESTART" in prompt:
            return 500, {"detail": "The graphics card stopped answering. Qwen-Image 2.1 starts again.",
                         "kind": "restart"}
        if "FAKE_CRASH" in prompt:
            print("ggml abort: CUDA error (fake)", flush=True)
            os._exit(3)         # as a CUDA error inside ggml ends the real one: no answer, the connection drops
        count, seed = int(body.get("batch_count") or 1), int(body.get("seed", -1))
        first = seed if seed >= 0 else 777
        cancel = qwen_state["cancel"]
        cancel.clear()
        # Like stable-diffusion.cpp: every picture is drawn first, and all are decoded at the end. A cancel while
        # drawing keeps none; one while decoding keeps the pictures decoded so far.
        for index in range(count):
            for step in range(10 if "FAKE_SLOW" in prompt else 0):
                qwen_state["progress"] = {"phase": "drawing", "step": step, "steps": 10,
                                          "fraction": 0.07 + 0.88 * (index + step / 10) / count,
                                          "message": f"Drawing picture {index + 1} of {count} · step {step + 1} of 10",
                                          "eta": round(((count - index) * 10 - step) * 0.1 + 75, 1)}
                if body.get("preview") is True and step:        # as the real one: none before a picture's first step
                    qwen_state["progress"]["preview"] = {"seq": index * 10 + step, "step": step, "image": index + 1}
                    if "FAKE_WINDOW" in prompt:                 # a window's request: where its previews lie
                        qwen_state["progress"]["preview"]["box"] = [0.5, 0.25, 1.25, 0.75]
                time.sleep(0.1)
                if cancel.is_set():
                    qwen_state["progress"] = None
                    return 409, {"detail": "Stopped: the pictures were cancelled.", "kind": "cancelled"}
        images: list[str] = []
        for index in range(count):
            if "FAKE_LONG_DECODE" in prompt:
                qwen_state["progress"] = {"phase": "finishing", "step": 0, "steps": 0,
                                          "fraction": 0.95 + 0.04 * index / count,
                                          "message": f"Finishing picture {index + 1} of {count}", "eta": None}
                time.sleep(1.5)
            if cancel.is_set():
                break
            path = out_dir / f"qwen_{index:02d}.png"
            path.write_bytes(tiny_png(body.get("transparent") is True, first + index))
            images.append(str(path))
        qwen_state["progress"] = None
        if not images:
            return 409, {"detail": "Stopped: the pictures were cancelled.", "kind": "cancelled"}
        if body.get("given_dir"):                               # the record of what the model was given
            given = Path(body["given_dir"])
            given.mkdir(parents=True, exist_ok=True)
            (given / "answer-1.webp").write_bytes(b"an answer")
            (given / "given.json").write_text(json.dumps({"words": prompt, "way": "words", "pictures": [],
                                                          "answers": ["answer-1.webp"] * len(images)}), encoding="utf-8")
        warnings = ["The pictures were made in parts (fake)."] if "FAKE_WARN" in prompt else []
        if len(images) < count:
            warnings.insert(0, f"Stopped after {len(images)} of {count} pictures.")
        return 200, {"images": images, "seeds": [first + i for i in range(len(images))], "seconds": 0.1,
                     "warning": " ".join(warnings) or None}

    retopos: list[dict[str, Any]] = []
    unwraps: list[dict[str, Any]] = []
    triangle = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"

    def retopo(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        retopos.append(body)
        if "src" in body:           # Instant Meshes: retopo + UVs
            if not Path(body["src"]).is_file():
                return 400, {"detail": "The mesh file does not exist. Make the 3D model again in the 3D stage."}
            Path(body["dst"]).write_text(triangle.replace("f 1 2 3", "vt 0 0\nf 1/1 2/1 3/1"), encoding="utf-8")
            return 200, {"vertices": 3, "faces": 1, "has_uv": True, "warning": "", "seconds": 0.1}
        mesh = Path(body["mesh_path"])
        if not mesh.is_file():
            return 400, {"detail": f"The 3D model was not found: {mesh}.", "kind": "input"}
        error = fake_error(mesh)
        if error:
            return 500, {"detail": error, "kind": "error"}
        if "FAKE_OOM" in mesh.read_bytes().decode("utf-8", "replace"):
            return 500, {"detail": "CUDA out of memory", "kind": "oom"}
        end = time.monotonic() + args.retopo_seconds
        while time.monotonic() < end:
            time.sleep(0.02)
        # The mesh's "#" lines travel along, so a test can steer the unwrap (FAKE_NO_UV, FAKE_UNWRAP_ERROR:<message>,
        # FAKE_UNWRAP_SECONDS:<seconds>).
        notes = [line for line in mesh.read_bytes().decode("utf-8", "replace").splitlines() if line.startswith("#")]
        Path(body["out_path"]).write_text(triangle + "".join(f"{line}\n" for line in notes), encoding="utf-8")
        return 200, {"obj_path": body["out_path"], "vertices": 3, "faces": 1, "seconds": 0.1, "warning": None,
                     "seed": body.get("seed")}

    def unwrap(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        unwraps.append(body)
        src, dst = Path(body["src"]), Path(body["dst"])
        if not src.is_file():
            return 400, {"detail": f"The mesh file {src} does not exist. Make the 3D model again in the 3D stage."}
        text = src.read_bytes().decode("utf-8", "replace")
        error = fake_error(src) or (text.split("FAKE_UNWRAP_ERROR:", 1)[1].strip()
                                    if "FAKE_UNWRAP_ERROR:" in text else None)
        if error:
            return 422, {"detail": error}
        no_uv = "FAKE_NO_UV" in text
        if "FAKE_UNWRAP_SECONDS:" in text:      # a slow unwrap (the job runner overlaps it with the next prop)
            end = time.monotonic() + float(text.split("FAKE_UNWRAP_SECONDS:", 1)[1].split()[0])
            while time.monotonic() < end:
                time.sleep(0.02)
        dst.write_text(text if no_uv else text + "vt 0 0\n", encoding="utf-8")
        lines = text.splitlines()
        return 200, {"vertices": sum(line.startswith("v ") for line in lines),
                     "faces": sum(line.startswith("f ") for line in lines), "has_uv": not no_uv,
                     "warning": "The mesh was saved without UVs because unwrapping failed (fake)." if no_uv else "",
                     "seconds": 0.1}

    deaf = threading.Event()

    def deafen() -> None:
        deaf.set()
        server.shutdown()
        server.server_close()

    batches = Batches(args, deafen)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *a: object) -> None:
            pass

        def answer(self, code: int, body: Any) -> None:
            data = json.dumps(body).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()
            except OSError:
                pass        # whoever asked has stopped waiting

        def do_GET(self) -> None:
            parts = self.path.split("/")
            if self.path == "/health":
                code = 503 if args.never_healthy else 200
                self.answer(code, {"ok": code == 200})
            elif self.path == "/spellcast/health":
                self.answer(503 if args.never_healthy else 200,
                            {"ready": not args.never_healthy, "low_vram": True, "multiview": not args.no_multiview,
                             "batch": True})
            elif self.path.startswith("/spellcast/batch/") and len(parts) == 4:
                code, body = batches.status(parts[3])
                self.answer(code, body)
                batches.poll_answered(body)
            elif self.path == "/fake/batches":
                self.answer(200, batches.requests)
            elif self.path == "/fake/cancels":
                self.answer(200, batches.cancels)
            elif self.path == "/spellcast/progress":
                self.answer(200, qwen_state["progress"] or {"stage": "Sculpting the shape", "step": 1, "total": 2,
                                                            "fraction": 0.5})
            elif self.path == "/spellcast/preview":
                preview = (qwen_state["progress"] or {}).get("preview")
                if not preview:
                    self.answer(404, {"detail": "There is no preview picture right now.", "kind": "input"})
                    return
                data = tiny_png(False, preview["seq"])
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Preview-Seq", str(preview["seq"]))
                self.send_header("X-Preview-Step", str(preview["step"]))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/fake/qwen":
                self.answer(200, qwen_jobs)
            elif self.path.startswith("/echo"):
                self.answer(200, {"path": self.path, "pid": os.getpid()})
            elif self.path == "/fake/bakes":
                self.answer(200, bakes.listing())
            elif self.path == "/fake/imports":
                self.answer(200, imports)
            elif self.path == "/fake/retopos":
                self.answer(200, retopos)
            elif self.path == "/fake/unwraps":
                self.answer(200, unwraps)
            elif self.path.startswith("/spellcast/bake/"):
                self.answer(*bakes.status(self.path.split("/")[3]))
            else:
                self.answer(404, {"detail": "not found"})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            parts = self.path.split("/")
            if self.path == "/spellcast/bake":
                self.answer(*bakes.start(body))
            elif self.path.startswith("/spellcast/bake/") and parts[-1] == "cancel":
                self.answer(*bakes.cancel(parts[3]))
            elif self.path == "/spellcast/import":
                self.answer(*import_model(body))
            elif self.path == "/spellcast/batch":
                self.answer(*batches.start(body))
            elif self.path.startswith("/spellcast/batch/") and parts[-1] == "cancel":
                answer = batches.cancel(parts[3])
                if answer is not None:      # None: lost, the connection closes without a word
                    self.answer(*answer)
            elif self.path == "/spellcast/generate":
                self.answer(*qwen_generate(body))
            elif self.path == "/spellcast/cancel":
                qwen_state["cancel"].set()
                self.answer(200, {"cancelling": qwen_state["progress"] is not None})
            elif self.path == "/spellcast/retopo":
                self.answer(*retopo(body))
            elif self.path == "/spellcast/unwrap":
                self.answer(*unwrap(body))
            else:
                self.answer(404, {"detail": "not found"})

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("ready ✓", flush=True)
    if args.crash_after is not None:
        def crash() -> None:
            print("crashing now: CUDA out of memory", flush=True)
            os._exit(3)
        threading.Timer(args.crash_after, crash).start()
    server.serve_forever(poll_interval=0.1)
    while deaf.is_set():        # --batch-deaf-after: nobody is listening any more, and the process stays
        time.sleep(1.0)


if __name__ == "__main__":
    main()
