"""Tests of the baker (code/spellcast/launchers/spellbake).

They run in the Instant Meshes venv, which has numpy, trimesh, Pillow, xatlas and
embreex:

    code\\venvs\\imesh\\Scripts\\python -m pytest code/tests/bake -m "not realasset"

The hub venv has none of these, so there nothing in this folder is collected.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

LAUNCHERS = Path(__file__).resolve().parents[2] / "spellcast" / "launchers"
if str(LAUNCHERS) not in sys.path:
    sys.path.insert(0, str(LAUNCHERS))

if importlib.util.find_spec("embreex") is None:
    # The hub venv. (pytest.importorskip here would skip the folder when pytest reaches it
    # from code/tests, but abort a run that names this folder: pytest imports this file
    # before collecting anything.)
    collect_ignore_glob = ["*"]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: takes more than a few seconds")
    config.addinivalue_line("markers", "realasset: needs meshes from this PC's projects; skipped when absent")
