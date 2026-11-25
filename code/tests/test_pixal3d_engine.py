"""The staged engine's logic on fakes (spellcast/profiling/p3d_checks/check_engine.py --fake), one pytest case per
test of check_engine.TESTS: the REAL engine, runner and part functions of a Pixal3D checkout (pixal3d/staged) with
everything below them stood in - loader, store, networks, samplers, conditioning, decoding, hooks. The order of the
parts, waves, the cap ladder and the failure classifier, cancel, failure isolation, resume, the random generators'
hand-over, the events: see check_engine.py's docstring.

CPU only: no card, no model file, no golden case. Needs PyTorch and a checkout with the staged engine: P3D_REPO (a
fork worktree, e.g. the main checkout's code/pixal3d when the tests run from a worktree of the hub, whose submodule
is empty), else code/pixal3d; skipped without either. About 80 s.
Run:  set PYTHONPATH=code\\logs\\profiling\\pytools
      code\\venvs\\pixal3d\\Scripts\\python.exe -m pytest -p no:cacheprovider code/tests/test_pixal3d_engine.py -q
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest

from spellcast import paths

REPO = Path(os.environ.get("P3D_REPO") or paths.PIXAL3D_DIR)
HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_ENGINE = (REPO / "pixal3d" / "staged" / "engine.py").is_file()


def _check_engine() -> Any:
    """check_engine.py, which imports `common` by that name as a script does: the package's common.py is that
    module (one object, not two)."""
    sys.modules.setdefault("common", importlib.import_module("spellcast.profiling.p3d_checks.common"))
    return importlib.import_module("spellcast.profiling.p3d_checks.check_engine")


check_engine = _check_engine()          # stdlib only at import: the tests are listed without torch or a checkout


@pytest.fixture(scope="module")
def fakes(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """The process as check_engine.py --fake makes it (the checkout as working folder and first import path, the
    stand-in modules in place), and as it was again afterwards: the checkout's pixal3d package and the stand-ins
    leave sys.modules, so no other test sees them."""
    if not HAVE_TORCH:
        pytest.skip("needs PyTorch (the pixal3d venv)")
    if not HAVE_ENGINE:
        pytest.skip(f"{REPO} has no pixal3d/staged/engine.py (P3D_REPO: a Pixal3D checkout with the staged engine)")
    cwd, path, visible = os.getcwd(), list(sys.path), os.environ.get("CUDA_VISIBLE_DEVICES")
    # Another test file's own import of the fork (test_pixal3d_weights holds its modules from collection on) is set
    # aside while the fakes import theirs, and comes back after.
    aside = {name: sys.modules.pop(name) for name in list(sys.modules) if name == "pixal3d" or name.startswith("pixal3d.")}
    try:
        check_engine.prepare_fake(REPO, tmp_path_factory.mktemp("engine-fake") / "scratch")
        yield check_engine
    finally:
        os.chdir(cwd)
        sys.path[:] = path
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
        for name in [name for name in sys.modules if name == "pixal3d" or name.startswith("pixal3d.")]:
            del sys.modules[name]
        sys.modules.update(aside)


@pytest.mark.parametrize("name", list(check_engine.TESTS))
def test_the_engine_on_fakes(fakes: Any, name: str) -> None:
    import torch

    fakes.TESTS[name]()
    assert not torch.cuda.is_initialized(), "the fake tests initialised CUDA"
