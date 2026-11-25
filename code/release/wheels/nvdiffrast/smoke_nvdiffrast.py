"""GPU smoke test for a built nvdiffrast wheel: pip installs it into a scratch folder (--target: no venv changes), the
python of a torch 2.8.0+cu128 venv imports it from there, rasterizes one triangle and interpolates its corner colours
on the card, the venv's own nvdiffrast does the same in a second process, and the two answers must be equal bit for bit
(the same CUDA source through the same nvcc: only the host compiler's flags may differ).

    <any python 3.9+> smoke_nvdiffrast.py <wheel> [--torch-venv DIR] [--work DIR] [--wait-minutes 10]

Each answer is also checked on its own: about 32 % of the pixels covered (the triangle's share of the picture), triangle
id 1 inside and 0 outside, the interpolated colours summing to 1. The card is shared: both runs happen inside one hold
of the project's GPU lock (code/spellcast/profiling/gpu_lock.py), and when the lock or the card is not free within
--wait-minutes the GPU check is skipped and says so. Exit 0: both runs passed and agree; 1: they did not; 3: skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # the Spellcast3D folder (this script is in code/release/wheels/<wheel>/)
SIZE = 64               # the picture's width and height in pixels
COVERED = 0.32          # the triangle (-0.8, -0.8) (0.8, -0.8) (0, 0.8): 1.28 of the clip square's 4
SKIPPED = 3


def render() -> dict:
    """One triangle through RasterizeCudaContext, rasterize and interpolate, in this process (on the card)."""
    import hashlib

    import torch
    import _nvdiffrast_c
    import nvdiffrast
    import nvdiffrast.torch as dr

    pos = torch.tensor([[[-0.8, -0.8, 0.0, 1.0], [0.8, -0.8, 0.0, 1.0], [0.0, 0.8, 0.0, 1.0]]], device="cuda")
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    colour = torch.eye(3, device="cuda")[None]          # red, green, blue at the corners
    rast, _ = dr.rasterize(dr.RasterizeCudaContext(device="cuda"), pos, tri, resolution=[SIZE, SIZE])
    image, _ = dr.interpolate(colour, rast, tri)
    torch.cuda.synchronize()
    rast, image = rast.cpu(), image.cpu()
    inside = rast[..., 3] > 0
    return {
        "extension": _nvdiffrast_c.__file__,
        "package": nvdiffrast.__file__,
        "version": nvdiffrast.__version__,
        "torch": torch.__version__,
        "card": torch.cuda.get_device_name(),
        "covered_pixels": int(inside.sum()),
        "triangle_ids": sorted(rast[..., 3].unique().tolist()),
        "colour_sum_error": float((image[inside].sum(-1) - 1).abs().max()),
        "rast_sha256": hashlib.sha256(rast.numpy().tobytes()).hexdigest(),
        "image_sha256": hashlib.sha256(image.numpy().tobytes()).hexdigest(),
    }


def problems_of(answer: dict) -> list[str]:
    expected = COVERED * SIZE * SIZE
    found = []
    if abs(answer["covered_pixels"] - expected) > SIZE:        # pixel centres along the edges
        found.append(f"{answer['covered_pixels']} pixels covered, about {expected:.0f} expected")
    if answer["triangle_ids"] != [0.0, 1.0]:
        found.append(f"triangle ids {answer['triangle_ids']}, [0, 1] expected")
    if not answer["colour_sum_error"] < 1e-5:
        found.append(f"the interpolated colours sum to 1 +- {answer['colour_sum_error']}")
    return found


def child(py: Path, env: dict, cwd: Path) -> dict:
    cp = subprocess.run([str(py), "-s", str(Path(__file__)), "--render"], env=env, cwd=cwd, capture_output=True,
                        text=True)
    if cp.returncode != 0:
        return {"error": (cp.stdout + cp.stderr)[-3000:]}
    return json.loads(cp.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wheel", type=Path, nargs="?")
    ap.add_argument("--torch-venv", type=Path, default=ROOT / "code" / "venvs" / "pixal3d")
    ap.add_argument("--work", type=Path, default=ROOT / "release" / "cache" / "wheels" / "test" / "nvdr")
    ap.add_argument("--wait-minutes", type=float, default=10)
    ap.add_argument("--render", action="store_true", help=argparse.SUPPRESS)   # the child's part
    a = ap.parse_args()
    if a.render:
        print(json.dumps(render()))
        return 0
    if a.wheel is None or not a.wheel.is_file():
        ap.error("give the wheel to test")

    py = a.torch_venv / "Scripts" / "python.exe"
    shutil.rmtree(a.work, ignore_errors=True)
    target = a.work / "target"
    target.mkdir(parents=True)
    subprocess.run([str(py), "-m", "pip", "install", "--no-deps", "--no-index", "--disable-pip-version-check", "-q",
                    "--target", str(target), str(a.wheel.absolute())], check=True)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    runs = {"built": {**env, "PYTHONPATH": str(target)}, "installed": env}

    sys.path.insert(0, str(ROOT / "code" / "spellcast" / "profiling"))
    import gpu_lock  # noqa: E402  (stdlib only)
    wait = a.wait_minutes * 60
    try:
        with gpu_lock.gpu_lock("WHEELS", f"nvdiffrast wheel smoke: one triangle ({a.wheel.name})", expected_minutes=1,
                               timeout=wait, card_timeout=wait):
            answers = {name: child(py, run_env, a.work) for name, run_env in runs.items()}
    except (gpu_lock.LockTimeout, gpu_lock.CardBusy) as exc:
        print(f"GPU check skipped: {exc}")
        return SKIPPED

    problems = []
    for name, answer in answers.items():
        if "error" in answer:
            problems.append(f"{name}: the run failed: {answer['error']}")
            continue
        problems += [f"{name}: {p}" for p in problems_of(answer)]
    if not problems:
        inside_target = Path(answers["built"]["extension"]).resolve().is_relative_to(target.resolve())
        if not inside_target:
            problems.append(f"built: the extension came from {answers['built']['extension']}, not from {target}")
        if Path(answers["installed"]["extension"]).resolve().is_relative_to(target.resolve()):
            problems.append("installed: the extension came from the scratch install")
        for key in ("rast_sha256", "image_sha256"):
            if answers["built"][key] != answers["installed"][key]:
                problems.append(f"{key[:-7]}: the built wheel's answer differs from the installed one's")
    print(json.dumps({"wheel": str(a.wheel), "answers": answers, "problems": problems}, indent=1))
    print("PASSED: the same answers bit for bit" if not problems else "FAILED")
    if not problems:
        shutil.rmtree(a.work, ignore_errors=True)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
