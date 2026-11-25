"""Forge's own config.json (code/forge/config.json), which both the installer and the hub prepare. STDLIB ONLY.

Forge's launch.py runs verify_version() at every start, and that blocks on a "Press Enter"
input() whenever config.json exists without the VERSION_UID it expects (with stdin=DEVNULL the
input() crashes the start instead). So the installer seeds the value before Forge's own setup
step, and the hub checks it again before every launch.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import paths

CONFIG = paths.FORGE_DIR / "config.json"


def forge_version_uid() -> str:
    """The VERSION_UID Forge wants in its config.json (read from its modules/launch_utils.py)."""
    try:
        text = (paths.FORGE_DIR / "modules" / "launch_utils.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    m = re.search(r'^VERSION_UID\b[^=\n]*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else "PY313"


def ensure_forge_config(cfg: Path = CONFIG) -> str | None:
    """Make config.json carry Forge's VERSION_UID. Returns a message when an unreadable file was set aside.

    A new config also gets auto_launch_browser=Disable (Forge's default "Local" would open a browser
    tab; under the hub SD_WEBUI_RESTARTING=1 already prevents that). A choice the user made in Forge's
    own settings is kept. The file is rewritten only when something changed, in Forge's own format.
    Raises OSError when the file cannot be read or written.
    """
    note = None
    data: dict = {}
    if cfg.exists():
        try:
            loaded = json.loads(cfg.read_text(encoding="utf-8"))
            data = loaded if isinstance(loaded, dict) else {}
        except ValueError:   # includes UnicodeDecodeError; Forge would crash on it too
            os.replace(cfg, cfg.with_name("config.json.broken"))
            note = "Forge's config.json could not be read; it was renamed to config.json.broken."
    wanted = {**data, "VERSION_UID": forge_version_uid()}
    wanted.setdefault("auto_launch_browser", "Disable")
    if wanted != data or not cfg.exists():
        paths.write_text_atomic(cfg, json.dumps(wanted, indent=4, ensure_ascii=False))
    return note
