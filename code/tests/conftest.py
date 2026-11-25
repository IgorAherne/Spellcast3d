"""Makes `import spellcast` work when pytest is started from any folder, and keeps File Explorer shut:
a save into the export folder opens no window on the desktop (SPELLCAST_NO_EXPLORER, spellcast/exports.py);
the tests of the opener itself unset it. What the installer remembers of the AI model files it checked goes into
each test's own folder, never into code/venvs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

os.environ["SPELLCAST_NO_EXPLORER"] = "1"


def pytest_configure(config: pytest.Config) -> None:
    # starlette's TestClient nags about httpx2; the hub pins httpx on purpose.
    config.addinivalue_line("filterwarnings", "ignore:Using `httpx` with `starlette.testclient`")


@pytest.fixture(autouse=True)
def _own_model_records(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    from spellcast.install import models

    monkeypatch.setattr(models, "VERIFIED_RECORDS", tmp_path_factory.mktemp("records") / "models-verified.json")


@pytest.fixture(autouse=True)
def _own_hi3d_folder(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hi3D's models folder is each test's own, empty one: no test looks at, links or hashes the 11 GB of a real one
    (Pixal3D's install and repair fetch them, and check those that were not downloaded by Spellcast3D)."""
    from spellcast import paths

    monkeypatch.setattr(paths, "MODELS_HI3D", tmp_path_factory.mktemp("hi3d"))


@pytest.fixture(autouse=True)
def _no_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The installer's downloader reaches only a server of the test's own (127.0.0.1): a test that forgets to replace a
    step fails at once instead of downloading GBs of AI models. Tests that replace models.download_file are not
    affected."""
    from spellcast.install import models

    real = models.download_file

    def local_only(url: str, *args: object, **kwargs: object) -> None:
        if not url.startswith(("http://127.0.0.1", "http://localhost")):
            pytest.fail(f"a test tried to download {url}")
        real(url, *args, **kwargs)      # type: ignore[arg-type]

    monkeypatch.setattr(models, "download_file", local_only)
