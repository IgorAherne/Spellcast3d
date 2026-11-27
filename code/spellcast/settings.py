"""Hub settings: spellcast/defaults.json deep-merged with the user's code/settings.json.

code/settings.json only holds what the user changed. PATCH /api/settings deep-merges
into it; a null value removes an override (back to the default).

The hub's own store also reads spellcast3d-config.ini (spellcast/config.py) once, when it is made: the image generator
the hub runs (and registers, server.create_app) is the one the file picked then. A later change of the file applies
at the next start; the hub's /api/status says so meanwhile (server.ConfigWatch).
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

from . import config, paths

log = logging.getLogger(__name__)


class SettingsError(Exception):
    """A settings change that cannot be applied. The message is shown to the user."""
    status_code = 400


def deep_merge(base: dict[str, Any], override: dict[str, Any], *, null_removes: bool = False) -> dict[str, Any]:
    """New dict: `base` with `override` merged in recursively (inputs are not modified).

    A None in `override` keeps the base value, or removes the key when `null_removes`.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if value is None:
            if null_removes:
                out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value, null_removes=null_removes)
        else:
            out[key] = copy.deepcopy(value)
    return out


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (paths.write_text_atomic), so a crash never leaves a half-written file."""
    paths.write_text_atomic(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _is_port(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and 1024 <= v <= 65535


def _is_bool(v: Any) -> bool:
    return isinstance(v, bool)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_str(v: Any) -> bool:
    return isinstance(v, str)


def _is_str_list(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


_CHECKS: list[tuple[tuple[str, ...], Callable[[Any], bool], str]] = [
    *((("ports", t), _is_port, "a port number between 1024 and 65535") for t in paths.ALL_VENV_IDS),
    (("open_browser",), _is_bool, "true or false"),
    (("gpu_exclusive",), _is_bool, "true or false"),
    (("forge", "ckpt_dirs"), _is_str_list, "a list of folder paths"),
    (("forge", "extra_args"), _is_str_list, "a list of command-line arguments"),
    # Nothing reads pixal3d.low_vram and no screen offers it: it is only kept so that an older settings.json
    # that still holds it is accepted. It goes in a later version.
    (("pixal3d", "low_vram"), lambda v: isinstance(v, bool) or v == "auto", 'true, false or "auto"'),
    (("pixal3d", "attn_backend"), _is_str, 'text such as "auto"'),
    (("pixal3d", "sage"), lambda v: isinstance(v, bool) or v == "auto", 'true, false or "auto"'),
    # The flows' linear layers as int8 GEMMs (the fork's PIXAL3D_INT8): "auto" = the fork's default, false = bf16
    (("pixal3d", "int8"), lambda v: isinstance(v, bool) or v == "auto", 'true, false or "auto"'),
    # The 3D panel's "Faster 3D generation" off: the conditioning and the coarse stages as the original computes them
    # on RTX 30xx and newer (the fork's PIXAL3D_COARSE_EXACT), so that a seed gives its model; false: the fast paths
    (("pixal3d", "coarse_exact"), _is_bool, "true or false"),
    # How many props of a job Pixal3D takes through its steps together; 0: all of them (adapters/pixal3d.py).
    (("pixal3d", "wave_size"), lambda v: _is_int(v) and v >= 0, "a whole number (0 = all the props of a job)"),
    # Whether a prop that does not fit the card's free memory may spill into shared memory (slow; it can freeze
    # the PC). Off: Pixal3D lowers its detail and fails it only when even that does not fit (adapters/pixal3d.py).
    (("pixal3d", "allow_spill"), _is_bool, "true or false"),
    (("imesh", "threads"), _is_int, "a whole number (-1 = all CPU cores)"),
    # "auto": LATO.2 moves each stage's models to the GPU only while it runs on cards below 10 GB (adapters/lato2.py)
    (("lato2", "low_vram"), lambda v: v in ("auto", "on", "off"), '"auto", "on" or "off"'),
    # "Faster image generation", Images' and Omni-Texture's own: Qwen-Image's int8 matmuls (adapters/qwen.py
    # matmul_arg)
    (("qwen", "int8_matmul"), _is_bool, "true or false"),
    (("qwen", "int8_matmul_omnitexture"), _is_bool, "true or false"),
]


def _lookup(d: dict[str, Any], keys: tuple[str, ...]) -> tuple[bool, Any]:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return False, None
        d = d[k]
    return True, d


def _free_default_ports(merged: dict[str, Any], user: dict[str, Any]) -> None:
    """A port the artist left at its default that is the same as another tool's (e.g. a new tool's default port,
    7774, which the artist had given the hub before that tool existed) moves to the next number no other tool has.
    Only defaults move: the artist's own ports stay as set (two of those alike are still refused)."""
    ports = merged.get("ports")
    if not isinstance(ports, dict):
        return
    own = user.get("ports") if isinstance(user.get("ports"), dict) else {}
    for tool in paths.ALL_VENV_IDS:
        port = ports.get(tool)
        if tool in own or not _is_port(port):
            continue
        taken = {ports.get(other) for other in paths.ALL_VENV_IDS if other != tool}
        if port not in taken:
            continue
        free = next((n for n in range(port + 1, 65536) if n not in taken), None)
        if free is not None:
            log.info("Port %s of %s is another tool's too; %s uses %s.", port, tool, tool, free)
            ports[tool] = free


def _invalid_keys(merged: dict[str, Any]) -> list[tuple[tuple[str, ...], str]]:
    """(key path, what it should be) for every known setting with a wrong type or value."""
    bad = []
    for keys, ok, expect in _CHECKS:
        found, value = _lookup(merged, keys)
        if not found or not ok(value):
            bad.append((keys, expect))
    ports = merged.get("ports")
    if isinstance(ports, dict):
        values = [ports.get(t) for t in paths.ALL_VENV_IDS]
        if len(set(values)) != len(values):
            bad.append((("ports",), f"{len(values)} different port numbers"))
    return bad


class SettingsStore:
    """Thread-safe access to the merged settings. Re-reads settings.json when it changes on disk.

    watch_config: remember what spellcast3d-config.ini picks now (.config): the image generator this hub runs, for
    the whole time it runs (Forge's model folders are only checked when it is Forge Neo). None = only for the hub's own
    code/settings.json.
    """

    def __init__(self, defaults_file: Path = paths.DEFAULTS_FILE, user_file: Path = paths.SETTINGS_FILE, *,
                 watch_config: bool | None = None):
        self.defaults_file = defaults_file
        self.user_file = user_file
        if watch_config is None:
            watch_config = Path(user_file) == paths.SETTINGS_FILE
        self.config: config.Config | None = config.load() if watch_config else None
        self._lock = threading.Lock()
        self._defaults: dict[str, Any] = json.loads(defaults_file.read_text(encoding="utf-8"))
        self._merged: dict[str, Any] = {}
        self._warnings: list[str] = []
        self._user_mtime: float | None = -1.0
        self._reload_if_changed()

    def get(self) -> dict[str, Any]:
        """A deep copy of the merged settings."""
        with self._lock:
            self._reload_if_changed()
            return copy.deepcopy(self._merged)

    @property
    def warnings(self) -> list[str]:
        with self._lock:
            self._reload_if_changed()
            return list(self._warnings)

    def patch(self, changes: dict[str, Any]) -> dict[str, Any]:
        """Deep-merge `changes` into settings.json and return the new merged settings."""
        if not isinstance(changes, dict):
            raise SettingsError("Settings must be sent as a JSON object.")
        with self._lock:
            user, _ = self._read_user()
            new_user = deep_merge(user, changes, null_removes=True)
            merged = deep_merge(self._defaults, new_user)
            _free_default_ports(merged, new_user)
            bad = _invalid_keys(merged)
            if bad:
                raise SettingsError(" ".join(f'"{".".join(k)}" must be {expect}.' for k, expect in bad))
            atomic_write_json(self.user_file, new_user)
            self._user_mtime = None
            self._reload_if_changed()
            return copy.deepcopy(self._merged)

    # ---- internals (call with the lock held) ----------------------------------------------
    def _read_user(self) -> tuple[dict[str, Any], str | None]:
        """(overrides, warning). A broken file is backed up and ignored, never fatal."""
        try:
            text = self.user_file.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return {}, None
        except OSError as e:
            return {}, f"Could not read {self.user_file} ({e}); using the default settings."
        try:
            data = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError as e:
            data, problem = None, f"error at line {e.lineno}"
        else:
            problem = None if isinstance(data, dict) else "it is not a JSON object"
        if problem is None:
            return data, None
        backup = self.user_file.with_name(self.user_file.name + ".broken")
        try:
            shutil.copyfile(self.user_file, backup)
        except OSError:
            pass
        return {}, (f"{self.user_file} is not valid settings ({problem}), so the defaults are used. "
                    f"A copy was saved as {backup.name}; fix it or delete settings.json.")

    def _reload_if_changed(self) -> None:
        try:
            mtime: float | None = self.user_file.stat().st_mtime
        except OSError:
            mtime = None
        if mtime == self._user_mtime and self._merged:
            return
        self._user_mtime = mtime
        user, warning = self._read_user()
        merged = deep_merge(self._defaults, user)
        _free_default_ports(merged, user)
        warnings = [warning] if warning else []
        for keys, expect in _invalid_keys(merged):
            _, default = _lookup(self._defaults, keys)
            target = merged
            for k in keys[:-1]:
                if not isinstance(target.get(k), dict):
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = copy.deepcopy(default)
            warnings.append(f'Setting "{".".join(keys)}" in settings.json must be {expect}; '
                            f"the default ({json.dumps(default)}) is used instead.")
        # Only when Forge can be the image generator (not for the hub that runs Qwen-Image 2.1).
        forge_runs = self.config is None or self.config.backend == paths.FORGE
        for folder in merged["forge"]["ckpt_dirs"] if forge_runs else ():
            # The same folder adapters/forge.py hands to Forge.
            if folder.strip() and not paths.user_path(folder).is_dir():
                warnings.append(f'The extra model folder "{folder}" does not exist, so Forge finds no models '
                                f"there. Fix the path in Settings (top right).")
        for w in warnings:
            if w not in self._warnings:
                log.warning(w)
        self._merged, self._warnings = merged, warnings
