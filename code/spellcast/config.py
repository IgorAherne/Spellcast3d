"""spellcast3d-config.ini: the choices the artist makes by editing a text file. STDLIB ONLY (the installer reads it).

    [image generator]
    backend = qwen | forge   the Images stage's tool. Only that one is installed (install/main.py) and registered in
                             the hub; switching installs the other one at the next start and keeps the first one's
                             files, so switching back is instant.
    quality = Q4_K_M         the size of the Images stage's models (QUALITIES): Qwen-Image 2.1's Turbo and full one,
                             and FLUX.3 Image Klein's file for it; the installer downloads the Turbo's of that size,
                             the other two only when the artist asks for them on the Images page
                             (spellcast/model_downloads.py). A bigger size already in the models folder is used in
                             its place (install/models.py size_on_disk).
    text_encoder = UD-Q4_K_XL  the size of Qwen-Image 2.1's text encoder, Qwen3-VL-8B (TEXT_ENCODERS), by the same rule:
                             the configured one when it is in the models folder, else the biggest bigger one there,
                             else the configured one is downloaded.
    [profiling]              for developers (PROFILING_KEYS, profiling()): on | off switches that measure Qwen-Image
    use_python_profiler      2.1's picture requests (spellcast/profiling/; reports in code\\logs\\profiling). One on
    use_torch_profiler       when Qwen-Image STARTS makes adapters/qwen.py start it through profiling/qwen_profiled.py
    diagnostic_enable_sync_hunter   instead of launchers/qwen_server.py; that entry reads them again at every
                             request, so switching off works from the next picture. The installer puts the package a
                             switched-on one needs into code\\venvs\\qwen after its components, never blocking one
                             (install/components.py install_profilers). Not part of changed(): the template's lines
                             say to start again, the hub does not.

The file is in the Spellcast3D folder, next to run-browser\\. Git does not track it (tools\\update.bat's
`git reset --keep` refuses to update over a tracked file the artist edited): install.py makes it from the tracked
config-template.ini when it is missing (also in the hub-only run at every start), and prepare_release.py puts it
into the release zip, so the artist can choose before the first start downloads anything.

Plain INI for Notepad, read with configparser: a line starting with ; (or #) is switched off, a ; after a value
starts a comment, a BOM, a missing [section] line, a line indented by mistake or a file Notepad saved as "Unicode"
(UTF-16) are fine, and the section names do not matter. A mistake never starts a download the artist did not choose:
an unknown backend, two backend lines switched on, every backend line switched off, a line that cannot be read, an
unknown setting while no backend line is switched on, or a file that is there but cannot be read all make
image_backend() None - no image generator is installed, and the hub registers none - and problems() says which
line to fix. A missing file, or one without any backend line, means the defaults (Qwen-Image 2.1, Q4_K_M); an
unknown quality means the default size, with a problem that says so. A profiling switch that is neither on nor off
(on/off, true/false, yes/no, 1/0), or set twice, stays off with a problem: a developer's switch never blocks the
image generator.

An installation from before Qwen-Image 2.1 (Forge Neo installed, Qwen-Image never) gets the file with Forge Neo
switched on (ensure_config_file): an update keeps the image generator the artist has, and downloads nothing new.

SPELLCAST_CONFIG names another file instead (developers and tests).
"""

from __future__ import annotations

import configparser
import os
import re
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .install import models     # stdlib only: Qwen-Image's model files and their sizes

CONFIG_FILE: Path = paths.CONFIG_FILE
TEMPLATE: Path = paths.CONFIG_TEMPLATE
SECTION = "image generator"
BACKENDS: tuple[str, ...] = paths.STAGE_TOOLS["images"]      # (qwen, forge): the default first
DEFAULT_BACKEND = BACKENDS[0]
BACKEND_NAMES = {paths.QWEN: "Qwen-Image 2.1", paths.FORGE: "Forge Neo"}
# Qwen-Image 2.1's denoiser sizes, smallest first, the same for the Turbo and the full model: the installer's manifest
# (install/models.py) names their files, for the installer, the adapter and the launcher alike.
QUALITIES: tuple[str, ...] = tuple(models.QWEN_DENOISERS)        # = tuple(models.QWEN_TURBO_DENOISERS) (a test)
DEFAULT_QUALITY = "Q4_K_M"
# Qwen-Image 2.1's text encoder's sizes, smallest first (install/models.py QWEN_TEXT_ENCODERS).
TEXT_ENCODERS: tuple[str, ...] = tuple(models.QWEN_TEXT_ENCODERS)
DEFAULT_TEXT_ENCODER = models.DEFAULT_TEXT_ENCODER
# The [profiling] switches, as AURA-V5's training config names them (spellcast/profiling/): PyInstrument on the
# launcher's Python, torch.profiler (CUPTI) on stable-diffusion.cpp's CUDA kernels, and the CPU-GPU syncs. One at a
# time is the rule (python + torch together: torch is skipped, request_profiler.py).
PROFILING_KEYS: tuple[str, ...] = ("use_python_profiler", "use_torch_profiler", "diagnostic_enable_sync_hunter")
KEYS = ("backend", "quality", "text_encoder", *PROFILING_KEYS)
# What a profiling switch may say, once lower-cased.
_SWITCH_WORDS = {"on": True, "true": True, "yes": True, "1": True, "off": False, "false": False, "no": False, "0": False}
# What may be written for a backend, once lower-cased and without spaces, dashes, dots and underscores.
_BACKEND_WORDS = {"qwen": paths.QWEN, "qwenimage": paths.QWEN, "qwenimage21": paths.QWEN,
                  "forge": paths.FORGE, "forgeneo": paths.FORGE}
_UNTIL_FIXED = ("Until this is fixed, no image generator is installed for the Images stage: fix it, save the file, "
                "then close Spellcast3D and start it again.")
_OPTION = re.compile(r"([^=:]*?)\s*[=:]")
_SWITCHED_OFF_BACKEND = re.compile(r"[;#]+\s*backend\s*[=:]", re.IGNORECASE)
# What the file made for an installation from before Qwen-Image 2.1 says above its backend lines (_forge_template).
KEPT_FORGE = ("; This Spellcast3D used Forge Neo before Qwen-Image 2.1 was built in, so Forge Neo stays switched on.\n"
              "; To try Qwen-Image 2.1, put a ; in front of the forge line and remove the one in front of the qwen "
              "line.\n")


@dataclass(frozen=True)
class Config:
    backend: str | None             # paths.QWEN or paths.FORGE; None: the file has a mistake, no image generator
    quality: str                    # one of QUALITIES
    problems: tuple[str, ...] = ()  # in artist words: the file, the line, what to do
    stamp: tuple[int, int] | None = None    # (mtime_ns, size) of the file that was read; None: there was none
    profiling: tuple[str, ...] = ()  # the PROFILING_KEYS switched on, in PROFILING_KEYS order
    text_encoder: str = DEFAULT_TEXT_ENCODER    # one of TEXT_ENCODERS


DEFAULTS = Config(DEFAULT_BACKEND, DEFAULT_QUALITY)
_cache: dict[str, Config] = {}


def config_file() -> Path:
    """The file in use: SPELLCAST_CONFIG when set, else spellcast3d-config.ini in the Spellcast3D folder."""
    override = os.environ.get("SPELLCAST_CONFIG", "").strip().strip('"')
    return Path(override) if override else CONFIG_FILE


def load() -> Config:
    """The settings in the file now: read again only when the file changed (one stat call otherwise)."""
    path = config_file()
    stamp = _stamp(path)
    key = os.path.normcase(os.path.abspath(path))
    cached = _cache.get(key)
    if cached is not None and cached.stamp == stamp:
        return cached
    cfg = read(path)
    _cache[key] = cfg
    return cfg


def image_backend() -> str | None:
    """The image generator to install and run: "qwen" or "forge", or None when the file has a mistake."""
    return load().backend


def qwen_quality() -> str:
    """The size of Qwen-Image 2.1's denoiser, e.g. "Q4_K_M" (always one of QUALITIES)."""
    return load().quality


def qwen_text_encoder() -> str:
    """The size of Qwen-Image 2.1's text encoder, e.g. "UD-Q4_K_XL" (always one of TEXT_ENCODERS)."""
    return load().text_encoder


def profiling() -> dict[str, bool]:
    """{switch: on?} for each of PROFILING_KEYS. Qwen's adapter asks when it starts the tool (any on: the profiling
    entry), the entry at every picture request (one stat call while the file is unchanged)."""
    on = load().profiling
    return {key: key in on for key in PROFILING_KEYS}


def problems() -> list[str]:
    """What is wrong in the file, in artist words (empty when nothing is)."""
    return list(load().problems)


def short(problem: str) -> str:
    """A problem without its last sentence about no image generator being installed until it is fixed: for a place
    that says that in its own words (the page's Images gate)."""
    return problem.replace(f" {_UNTIL_FIXED}", "")


def changed(since: Config) -> bool:
    """The file now picks something else than `since` did (e.g. what the running hub started with). Saving it
    unchanged, or editing only its comments, is no change; nor is a profiling switch (its lines say to start again)."""
    now = load()
    return (now.backend, now.quality, now.text_encoder) != (since.backend, since.quality, since.text_encoder)


def describe(cfg: Config | None = None) -> str:
    """What the file picks, in a few words: "Qwen-Image 2.1 (model size Q4_K_M)", "Forge Neo" or "none"."""
    cfg = cfg or load()
    if cfg.backend is None:
        return "none (the file has a mistake, see below)"
    return BACKEND_NAMES[cfg.backend] + (f" (model size {cfg.quality}, text encoder {cfg.text_encoder})"
                                         if cfg.backend == paths.QWEN else "")


def ensure_config_file() -> str | None:
    """Make the file from the template when it is missing (never over an existing one). -> A line to show when it
    made one or could not, else None. An installation from before Qwen-Image 2.1 (kept_forge) gets Forge Neo
    switched on: updating it must not swap the artist's image generator, or download Qwen-Image's 11 GB unasked."""
    path = config_file()
    if path.exists():
        return None
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    keep_forge = kept_forge()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = TEMPLATE.read_bytes()
        tmp.write_bytes(_forge_template(text) if keep_forge else text)
        try:
            os.rename(tmp, path)        # fails when another process made it meanwhile: theirs stays
        except FileExistsError:
            return None
    except OSError as e:
        return (f"Could not make {path} ({e}); Spellcast3D uses its default image generator, "
                f"{BACKEND_NAMES[DEFAULT_BACKEND]}.")
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    where = f"{path.name} in the Spellcast3D folder" if _name(path) == path.name else str(path)
    if keep_forge:
        return (f"Made {where}: it keeps Forge Neo, the image generator this Spellcast3D had so far. Qwen-Image 2.1 "
                f"is now built in: to switch to it, open the file with Notepad (it says how).")
    return f"Made {where}: open it with Notepad to choose the image generator (Qwen-Image 2.1 or Forge Neo)."


def kept_forge() -> bool:
    """This Spellcast3D is from before Qwen-Image 2.1 was built in: Forge Neo was installed here (or tried), Qwen-Image
    never. (Since then Forge Neo is only installed when the file picks it, and then the file is there.)"""
    from .install import state     # stdlib only

    entries = state.load()
    forge = entries.get(paths.FORGE)
    return isinstance(forge, dict) and forge.get("status") != state.UNSUPPORTED and paths.QWEN not in entries


def _forge_template(text: bytes) -> bytes:
    """The template with Forge Neo switched on instead of Qwen-Image 2.1, and KEPT_FORGE saying why."""
    newline = b"\r\n" if b"\r\n" in text else b"\n"
    out = text.replace(newline + b"backend = qwen ", newline + b"; backend = qwen ", 1)
    out = out.replace(newline + b"; backend = forge ", newline + b"backend = forge ", 1)
    note = KEPT_FORGE.replace("\n", newline.decode("ascii")).encode("utf-8")
    heading = newline + b"; Keep ONE backend line switched on."
    out = out.replace(heading, newline + note.rstrip(newline) + heading, 1)
    return out


def read(path: Path) -> Config:
    """The settings in `path`, without the cache. A file that is there but cannot be read is a mistake (no image
    generator: it may well pick another one than the default); its result is not kept (stamp None), so the next
    load() tries again."""
    stamp = _stamp(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return DEFAULTS
    except OSError as e:
        return Config(None, DEFAULT_QUALITY,
                      (f"Spellcast3D could not read {_name(path)} ({e}). Until it can, no image generator is "
                       f"installed for the Images stage: make sure the file is there and can be opened (for example "
                       f"that OneDrive keeps it on this PC), then close Spellcast3D and start it again.",), None)
    text = _decode(raw)
    if text is None:
        return Config(None, DEFAULT_QUALITY,
                      (f"{_name(path)}: Spellcast3D cannot read the text of this file. Save it as UTF-8 (in Notepad: "
                       f"File > Save as, Encoding: UTF-8). {_UNTIL_FIXED}",), stamp)
    return parse(text, _name(path), stamp)


def _decode(raw: bytes) -> str | None:
    """The file's text: UTF-8 (with or without a BOM, as Notepad saves it), UTF-16 (Notepad's "Unicode"), else the
    Windows code page (an old editor's ANSI). None when it is not text."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ["utf-16"]
    elif b"\x00" in raw:        # UTF-16 without a BOM: every other byte of plain letters is 0
        encodings = ["utf-16-le" if raw[1:2] == b"\x00" else "utf-16-be"]
    else:
        encodings = ["utf-8-sig", "mbcs" if os.name == "nt" else "cp1252"]
    for encoding in encodings:
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        return None if "\x00" in text else text
    return None


def parse(text: str, name: str = CONFIG_FILE.name, stamp: tuple[int, int] | None = None) -> Config:
    """The settings in `text` (the file's contents; `name` is how the problems call the file)."""
    # An indented line is a line of its own: configparser would add it to the value above (" backend = forge" under
    # "backend = qwen" would make one value of both lines), and two backend lines must be found as such.
    lines = [line.lstrip() for line in text.lstrip("\ufeff").splitlines()]
    values, unreadable = _values(lines)
    where = _option_lines(lines)
    problems: list[str] = []
    fatal = False

    def at(key: str) -> str:
        found = where.get(key, [])
        if len(found) > 1:
            return f"{name}, lines {', '.join(map(str, found[:-1]))} and {found[-1]}"
        return f"{name}, line {found[0]}" if found else name

    for lineno in unreadable:
        fatal = True
        problems.append(f'{name}, line {lineno}: Spellcast3D cannot read this line ("{lines[lineno - 1]}"). A setting '
                        f"is written as name = value, and a line starting with ; is switched off. {_UNTIL_FIXED}")
    backend = DEFAULT_BACKEND
    if len(where.get("backend", [])) > 1:
        fatal = True
        problems.append(f"{at('backend')}: more than one backend line is switched on. Keep one, and put a ; in front "
                        f"of the others. {_UNTIL_FIXED}")
    elif "backend" in values:
        word = re.sub(r"[\s_.\-]", "", _plain(values["backend"]).lower())
        if word in _BACKEND_WORDS:
            backend = _BACKEND_WORDS[word]
        else:
            fatal = True
            said = f'"{_plain(values["backend"])}" is not an image generator Spellcast3D knows' if word else \
                'no image generator is written after "backend ="'
            problems.append(f"{at('backend')}: {said} (write backend = qwen or backend = forge). {_UNTIL_FIXED}")
    else:
        # Backend lines, all switched off: one was meant to stay on (the default would download what nobody chose).
        off = [lineno for lineno, line in enumerate(lines, 1) if _SWITCHED_OFF_BACKEND.match(line)]
        if off:
            fatal = True
            where_off = f"line {off[0]}" if len(off) == 1 else \
                f"lines {', '.join(map(str, off[:-1]))} and {off[-1]}"
            problems.append(f"{name}, {where_off}: every backend line is switched off (it starts with ;). Remove the ; "
                            f"in front of the one you want, backend = qwen or backend = forge. {_UNTIL_FIXED}")
    for key in values:
        if key not in KEYS:
            # Without a backend line, this may be one with a typo: the default must not start a download then.
            ignored = "backend" in values
            fatal = fatal or not ignored
            problems.append(f'{at(key)}: "{key}" is not a setting Spellcast3D knows (it knows backend, quality and '
                            f"text_encoder, "
                            f"and for developers {', '.join(PROFILING_KEYS[:-1])} and {PROFILING_KEYS[-1]})"
                            + (", so the line is ignored." if ignored else f". {_UNTIL_FIXED}"))
    quality = DEFAULT_QUALITY
    if len(where.get("quality", [])) > 1:
        problems.append(f"{at('quality')}: more than one quality line is switched on, so {DEFAULT_QUALITY} is "
                        f"used. Keep one, and put a ; in front of the others.")
    elif "quality" in values:
        wanted = _plain(values["quality"]).upper().replace("-", "_")
        if wanted in QUALITIES:
            quality = wanted
        else:
            problems.append(f'{at("quality")}: "{_plain(values["quality"])}" is not a Qwen model size (use '
                            f"{', '.join(QUALITIES[:-1])} or {QUALITIES[-1]}), so {DEFAULT_QUALITY} is used.")
    text_encoder = DEFAULT_TEXT_ENCODER
    if len(where.get("text_encoder", [])) > 1:
        problems.append(f"{at('text_encoder')}: more than one text_encoder line is switched on, so "
                        f"{DEFAULT_TEXT_ENCODER} is used. Keep one, and put a ; in front of the others.")
    elif "text_encoder" in values:
        given = _plain(values["text_encoder"])
        wanted = _text_encoder_size(given)
        if wanted is not None:
            text_encoder = wanted
        else:
            problems.append(f'{at("text_encoder")}: "{given}" is not a size of the text encoder (use '
                            f"{', '.join(TEXT_ENCODERS[:-1])} or {TEXT_ENCODERS[-1]}), so {DEFAULT_TEXT_ENCODER} is "
                            f"used.")
    # The developers' switches: a mistake leaves that profiler off and is said, never fatal (no picture waits on it).
    profiling: list[str] = []
    for key in PROFILING_KEYS:
        word = _plain(values.get(key, "off")).lower()
        if len(where.get(key, [])) > 1:
            problems.append(f"{at(key)}: {key} is set more than once, so that profiler stays off. Keep one line, "
                            f"and put a ; in front of the others.")
        elif word not in _SWITCH_WORDS:
            said = f'"{_plain(values[key])}" is neither on nor off' if word else f'nothing is written after "{key} ="'
            problems.append(f"{at(key)}: {said}, so that profiler stays off (write {key} = on or {key} = off).")
        elif _SWITCH_WORDS[word]:
            profiling.append(key)
    return Config(None if fatal else backend, quality, tuple(problems), stamp, tuple(profiling), text_encoder)


def _text_encoder_size(given: str) -> str | None:
    """A text encoder's size as TEXT_ENCODERS names it, from what the artist wrote: the size ("UD-Q6_K_XL", any case,
    "Q6_K_XL" or "UD_Q6_K_XL" too) or the whole file name; None when it is none of them."""
    word = given.strip().upper().removeprefix("QWEN3-VL-8B-INSTRUCT-").removesuffix(".GGUF").replace("UD_", "UD-")
    for size in TEXT_ENCODERS:
        if word in (size.upper(), size.upper().removeprefix("UD-")):
            return size
    return None


def _values(lines: list[str]) -> tuple[dict[str, str], list[int]]:
    """({setting: its value}, numbers of the lines configparser cannot read). The sections are only headings for
    the artist: the settings of all of them count (a typo in a [section] line must not hide the backend)."""
    def parser() -> configparser.ConfigParser:
        # strict=False: two lines of one setting are reported by parse() (with their line numbers), not raised here.
        return configparser.ConfigParser(interpolation=None, comment_prefixes=(";", "#"),
                                         inline_comment_prefixes=(";", "#"), strict=False,
                                         empty_lines_in_values=False, default_section="\0")

    text, offset = "\n".join(lines), 0
    cp = parser()
    try:
        try:
            cp.read_string(text)
        except configparser.MissingSectionHeaderError:
            cp, offset = parser(), 1
            cp.read_string(f"[{SECTION}]\n{text}")
        unreadable: list[int] = []
    except configparser.ParsingError as e:
        unreadable = [lineno - offset for lineno, _line in e.errors]
    values: dict[str, str] = {}
    for section in cp.sections():
        for key, value in cp.items(section, raw=True):
            values[key] = value
    return values, unreadable


def _option_lines(lines: list[str]) -> dict[str, list[int]]:
    """{setting (lower case): the numbers of the lines that set it}, as configparser reads them."""
    where: dict[str, list[int]] = {}
    for lineno, line in enumerate(lines, 1):
        if not line or line[0] in ";#[":
            continue
        m = _OPTION.match(line)
        if m and m.group(1).strip():
            where.setdefault(m.group(1).strip().lower(), []).append(lineno)
    return where


def _plain(value: str) -> str:
    """A value without a comment glued to it ("qwen;x") and without quotes around it."""
    return re.split(r"[;#]", value, maxsplit=1)[0].strip().strip("\"'").strip()


def _name(path: Path) -> str:
    return path.name if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(CONFIG_FILE)) \
        else str(path)


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size
