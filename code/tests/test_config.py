"""spellcast3d-config.ini (spellcast/config.py): the template, what Notepad leaves in the file (a BOM, CRLF, a missing
[section] line, an indented line), every mistake an artist can make and what it does (never a download nobody chose),
the file made from the template, the cache, the hub's warning when the file changes while it runs, and the
developers' [profiling] switches.

Fast (no network, no GPU). Run with the hub venv:  python -m pytest -p no:cacheprovider code/tests
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import pytest
from fastapi.testclient import TestClient

from spellcast import config, paths
from spellcast.install import state
from spellcast.projects import ProjectStore
from spellcast.server import ConfigWatch, config_changed, create_app
from spellcast.settings import SettingsStore

TEMPLATE = paths.CONFIG_TEMPLATE.read_text(encoding="utf-8")
FORGE = TEMPLATE.replace("\nbackend = qwen ", "\n; backend = qwen ").replace("\n; backend = forge", "\nbackend = forge")
UNTIL_FIXED = "no image generator is installed for the Images stage"
AFTER_TEMPLATE = TEMPLATE.count("\n") + 1      # the number of a line added at the template's end
QUALITY_LINE = TEMPLATE.splitlines().index("quality = Q4_K_M") + 1
ALL_OFF = {"use_python_profiler": False, "use_torch_profiler": False, "diagnostic_enable_sync_hunter": False}


def _switch(text: str, key: str, value: str) -> str:
    """The template's line of a profiling switch, saying `value` instead of off (its comment kept)."""
    line = next(line for line in text.splitlines() if line.startswith(f"{key} = off"))
    return text.replace(line, line.replace(f"{key} = off", f"{key} = {value}", 1))


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str | bytes], Path]:
    """Writes the config file (in tmp_path; spellcast3d-config.ini there is the one in use). The installation it
    belongs to has nothing installed yet (a new one), unless a test says otherwise."""
    path = tmp_path / "spellcast3d-config.ini"
    monkeypatch.setattr(config, "CONFIG_FILE", path)
    monkeypatch.delenv("SPELLCAST_CONFIG", raising=False)
    monkeypatch.setattr(state, "load", lambda: {})
    config._cache.clear()

    def write(text: str | bytes) -> Path:
        path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
        config._cache.clear()
        return path

    return write


def test_the_template_picks_qwen_and_the_default_size() -> None:
    assert config.parse(TEMPLATE) == config.Config("qwen", "Q4_K_M")
    assert config.parse(FORGE) == config.Config("forge", "Q4_K_M")
    assert config.BACKENDS == paths.STAGE_TOOLS["images"] == ("qwen", "forge")
    assert config.QUALITIES == ("Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
    for quality in config.QUALITIES:                      # every size the template lists is one Spellcast3D knows
        assert quality in TEMPLATE
    # what the design asks the template to say, for artists
    assert "Research licence" in TEMPLATE and "also for commercial work" in TEMPLATE
    assert "Keep ONE backend line switched on." in TEMPLATE and TEMPLATE.startswith("; Spellcast3D settings.")
    # and at its end, for developers: the profilers, all switched off, each on its own line, where the reports go
    assert config.parse(TEMPLATE).profiling == () and config.parse(FORGE).profiling == ()
    profiling = TEMPLATE[TEMPLATE.index("\n[profiling]\n") + 1:]
    assert [line.split(" ")[0] for line in profiling.splitlines() if not line.startswith((";", "["))] \
        == list(config.PROFILING_KEYS)
    words = " ".join(line.lstrip("; ") for line in profiling.splitlines())
    assert "Switch ONE on, save, then close Spellcast3D and start it again" in words
    assert "Reports go to code\\logs\\profiling\\." in words and "the next picture is not measured" in words
    assert config.KEYS == ("backend", "quality", "text_encoder", "use_python_profiler", "use_torch_profiler",
                           "diagnostic_enable_sync_hunter")


@pytest.mark.parametrize("value, on", [
    ("on", True), ("On", True), ("TRUE", True), ("yes", True), ("1", True), ('"on"', True), ("on;mine", True),
    ("off", False), ("Off", False), ("false", False), ("NO", False), ("0", False),
])
def test_a_profiling_switch_is_on_or_off(value: str, on: bool) -> None:
    for key in config.PROFILING_KEYS:
        result = config.parse(_switch(TEMPLATE, key, value))
        assert (result.backend, result.problems) == ("qwen", ())          # known settings: nothing to say
        assert result.profiling == ((key,) if on else ())
    # several on: in PROFILING_KEYS order, whatever the file's (the launcher decides what runs together)
    text = TEMPLATE.replace("use_python_profiler = off", "use_python_profiler = " + value)
    text = text + f"[more]\nuse_torch_profiler = {value}\n"
    lines = config.parse(text.replace("\nuse_torch_profiler = off", "\n; use_torch_profiler = off"))
    assert lines.profiling == (("use_python_profiler", "use_torch_profiler") if on else ())


@pytest.mark.parametrize("line, words", [
    ("use_torch_profiler = maybe", '"maybe" is neither on nor off, so that profiler stays off (write '
                                   "use_torch_profiler = on or use_torch_profiler = off)."),
    ("use_torch_profiler =", 'nothing is written after "use_torch_profiler =", so that profiler stays off'),
    ("use_torch_profiler = ;on", 'nothing is written after "use_torch_profiler ="'),
])
def test_a_wrong_profiling_switch_stays_off_and_blocks_nothing(line: str, words: str) -> None:
    lines = TEMPLATE.splitlines()
    torch_line = next(n for n, text in enumerate(lines, 1) if text.startswith("use_torch_profiler"))
    python_line = next(n for n, text in enumerate(lines, 1) if text.startswith("use_python_profiler"))
    text = TEMPLATE.replace(lines[torch_line - 1], line)
    result = config.parse(text)
    assert (result.backend, result.quality, result.profiling) == ("qwen", "Q4_K_M", ())   # the image generator stays
    assert len(result.problems) == 1, result.problems
    assert result.problems[0].startswith(f"spellcast3d-config.ini, line {torch_line}: {words}"), result.problems[0]
    assert UNTIL_FIXED not in result.problems[0]
    # a mistake in one switch leaves another one on
    both = config.parse(_switch(text, "use_python_profiler", "on"))
    assert both.profiling == ("use_python_profiler",) and both.problems == result.problems
    # set twice, both on: neither line is taken
    twice = config.parse(_switch(TEMPLATE, "use_python_profiler", "on") + "use_python_profiler = on\n")
    assert (twice.backend, twice.profiling) == ("qwen", ())
    assert twice.problems == (f"spellcast3d-config.ini, lines {python_line} and {AFTER_TEMPLATE}: use_python_profiler "
                              "is set more than once, so that profiler stays off. Keep one line, and put a ; in front of "
                              "the others.",)
    # on in one line and off in another: that profiler stays off as well (no guessing which line was meant)
    assert config.parse(_switch(TEMPLATE, "use_torch_profiler", "on") + "use_torch_profiler = off\n").profiling == ()
    # a mistake in the backend line does not hide the switches (the profiling entry of a running Qwen-Image reads them)
    broken = config.parse(_switch(TEMPLATE.replace("\nbackend = qwen ", "\nbackend = forg "), "use_torch_profiler",
                                  "on"))
    assert broken.backend is None and broken.profiling == ("use_torch_profiler",)


def test_a_misspelt_switch_is_named_with_the_ones_spellcast3d_knows() -> None:
    result = config.parse(TEMPLATE + "use_python_profilr = on\n")
    assert result.backend == "qwen" and result.profiling == ()
    assert result.problems == (f'spellcast3d-config.ini, line {AFTER_TEMPLATE}: "use_python_profilr" is not a setting '
                               "Spellcast3D knows (it knows backend, quality and text_encoder, and for developers use_python_profiler, "
                               "use_torch_profiler and diagnostic_enable_sync_hunter), so the line is ignored.",)


def test_the_switches_are_read_again_when_saved_and_the_hub_warns_only_about_the_generator(
        cfg: Callable[[str | bytes], Path]) -> None:
    """profiling() follows the file (Qwen's adapter asks when it starts the tool, the profiling entry at every request);
    the hub's "start again" warning stays about the image generator (developers restart themselves)."""
    assert config.profiling() == ALL_OFF                                  # no file: all off
    path = cfg(TEMPLATE)
    start = config.load()
    assert config.profiling() == ALL_OFF and list(config.profiling()) == list(config.PROFILING_KEYS)
    watch = ConfigWatch(start)
    path.write_text(_switch(TEMPLATE, "use_torch_profiler", "on"), encoding="utf-8")
    os.utime(path, (5000, 5000))                                          # saved: read again
    assert config.profiling() == {**ALL_OFF, "use_torch_profiler": True}
    assert not config.changed(start) and watch.warnings() == []           # the hub says nothing
    path.write_text(_switch(FORGE, "diagnostic_enable_sync_hunter", "yes"), encoding="utf-8")
    os.utime(path, (6000, 6000))
    assert config.profiling() == {**ALL_OFF, "diagnostic_enable_sync_hunter": True}
    assert config.changed(start) and len(watch.warnings()) == 1           # the image generator: as before
    cfg(b"\x00\x01 not text")                                             # unreadable: all off
    assert config.profiling() == ALL_OFF


def test_what_notepad_leaves_in_the_file_is_fine(cfg: Callable[[str | bytes], Path]) -> None:
    cfg(b"\xef\xbb\xbf" + FORGE.replace("\n", "\r\n").encode("utf-8"))          # Notepad's BOM and CRLF
    assert (config.image_backend(), config.qwen_quality(), config.problems()) == ("forge", "Q4_K_M", [])
    assert config.parse("backend = forge\nquality = q8_0\n") == config.Config("forge", "Q8_0")   # no [section]
    assert config.parse("[image generater]\nbackend=Forge\n").backend == "forge"   # a typo in the heading: still read
    assert config.parse('backend = "Forge Neo"\n').backend == "forge"
    assert config.parse("backend = qwen-image 2.1;mine\n").backend == "qwen"       # a comment glued to the value
    assert config.parse("").backend == config.parse("; all switched off\n").backend == "qwen"   # defaults


@pytest.mark.parametrize("text, line, words", [
    (TEMPLATE.replace("\nbackend = qwen ", "\nbackend = forg "), "line 6", '"forg" is not an image generator'),
    (TEMPLATE.replace("\nbackend = qwen ", "\nbackend = "), "line 6", 'no image generator is written'),
    (TEMPLATE.replace("\n; backend = forge", "\nbackend = forge"), "lines 6 and 7", "more than one backend line"),
    # switched on by removing only the ;  - the space left in front must not glue it to the line above
    (TEMPLATE.replace("\n; backend = forge", "\n backend = forge"), "lines 6 and 7", "more than one backend line"),
    (TEMPLATE + "this is not a setting\n", f"line {AFTER_TEMPLATE}", 'cannot read this line ("this is not a setting")'),
    ("[image generator]\nbakend = forge\n", "line 2", '"bakend" is not a setting'),   # the backend line misspelt
    # the ; put in front of the qwen line, and not taken away from the forge line: the default must not download
    (TEMPLATE.replace("\nbackend = qwen ", "\n; backend = qwen "), "lines 6 and 7",
     "every backend line is switched off"),
    ("; backend = forge\nquality = Q4_K_M\n", "line 1", "every backend line is switched off"),
])
def test_a_mistake_installs_no_image_generator_and_names_its_line(text: str, line: str, words: str) -> None:
    result = config.parse(text)
    assert result.backend is None and len(result.problems) == 1
    problem = result.problems[0]
    assert problem.startswith(f"spellcast3d-config.ini, {line}: ") and words in problem, problem
    assert UNTIL_FIXED in problem and "start it again" in problem
    # the page's Images gate says that part in its own words: the problem without it
    assert UNTIL_FIXED not in config.short(problem)
    assert config.short(problem).startswith(f"spellcast3d-config.ini, {line}")
    assert config.short(problem).endswith(").") or config.short(problem).endswith(".")


def test_a_wrong_size_or_an_unknown_setting_only_warns() -> None:
    result = config.parse(TEMPLATE.replace("quality = Q4_K_M", "quality = Q4"))
    assert (result.backend, result.quality) == ("qwen", "Q4_K_M") and QUALITY_LINE == 13
    assert result.problems == ('spellcast3d-config.ini, line 13: "Q4" is not a Qwen model size (use Q3_K_M, Q4_K_M, '
                               "Q5_K_M, Q6_K or Q8_0), so Q4_K_M is used.",)
    result = config.parse(TEMPLATE + "quality = Q8_0\n")          # two sizes switched on: the default, not the biggest
    assert (result.backend, result.quality) == ("qwen", "Q4_K_M")
    assert f"lines 13 and {AFTER_TEMPLATE}" in result.problems[0]
    result = config.parse(FORGE + "colour = red\n")
    assert result.backend == "forge" and "the line is ignored" in result.problems[0]
    assert config.parse(TEMPLATE.replace("Q4_K_M\n", "q6_k\n")).quality == "Q6_K"


def test_notepads_unicode_and_a_file_that_cannot_be_read(cfg: Callable[[str | bytes], Path], tmp_path: Path) -> None:
    """Notepad's "Unicode" (UTF-16, with or without its BOM) is read; text that is not text, or a file that is there
    but cannot be read, is a mistake (no image generator: the file may pick another one than the default), and a file
    that could not be read is read again next time."""
    cfg(b"\xff\xfe" + FORGE.replace("\n", "\r\n").encode("utf-16-le"))
    assert (config.image_backend(), config.problems()) == ("forge", [])
    cfg(FORGE.encode("utf-16-be"))
    assert (config.image_backend(), config.problems()) == ("forge", [])
    cfg(FORGE.replace("Spellcast3D settings", "Spellcast3D réglages").encode("cp1252"))   # an old editor's ANSI
    assert config.image_backend() == "forge"
    cfg(b"\x00\x01\x02 backend = forge")
    result = config.load()
    assert result.backend is None and "Save it as UTF-8" in result.problems[0] and "\x00" not in result.problems[0]
    path = tmp_path / "spellcast3d-config.ini"
    path.unlink()
    path.mkdir()                                  # there, but not readable as a file
    config._cache.clear()
    result = config.load()
    assert result.backend is None and result.problems[0].startswith("Spellcast3D could not read")
    assert "no image generator is installed" in result.problems[0] and result.stamp is None
    path.rmdir()
    path.write_text(FORGE, encoding="utf-8")
    assert config.image_backend() == "forge"      # read again: the failure was not kept


def test_an_installation_from_before_qwen_keeps_forge(cfg: Callable[[str | bytes], Path], tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """Forge Neo installed and Qwen-Image never: the file is made with Forge Neo switched on (an update must not swap
    the artist's image generator, nor download 11 GB they did not ask for). A new installation gets Qwen-Image."""
    path = tmp_path / "spellcast3d-config.ini"
    entries: dict[str, dict] = {"hub": {"status": state.INSTALLED}, "forge": {"status": state.INSTALLED}}
    monkeypatch.setattr(state, "load", lambda: entries)
    assert config.kept_forge()
    made = config.ensure_config_file()
    assert made is not None and "keeps Forge Neo" in made and "Qwen-Image 2.1 is now built in" in made
    text = path.read_text(encoding="utf-8")
    assert config.parse(text) == config.Config("forge", "Q4_K_M")
    assert "\n; backend = qwen " in text and "\nbackend = forge " in text and config.KEPT_FORGE in text
    assert text.replace(config.KEPT_FORGE, "") == FORGE                       # the template otherwise
    lines = text.splitlines()
    assert lines.index("; Keep ONE backend line switched on.") == lines.index("[image generator]") + 3
    # the developers' switches at its end are untouched, and read as in the template
    assert text.endswith(TEMPLATE[TEMPLATE.index("[profiling]\n"):])
    assert config.parse(_switch(text, "use_python_profiler", "on")).profiling == ("use_python_profiler",)
    # switching to Qwen-Image the way the note says gives the template's choice
    switched = text.replace("\n; backend = qwen ", "\nbackend = qwen ").replace("\nbackend = forge ",
                                                                                 "\n; backend = forge ")
    assert config.parse(switched) == config.Config("qwen", "Q4_K_M")
    # CRLF, as git may check the template out
    crlf = config._forge_template(paths.CONFIG_TEMPLATE.read_bytes().replace(b"\n", b"\r\n"))
    assert b"\r\n; This Spellcast3D used Forge Neo" in crlf and b"\n;" not in crlf.replace(b"\r\n", b"")
    assert config.parse(crlf.decode("utf-8")).backend == "forge"
    for others in ({"forge": {"status": state.INSTALLED}, "qwen": {"status": state.FAILED}},   # Qwen-Image tried
                   {"forge": {"status": state.UNSUPPORTED}},                                    # Forge can't run here
                   {"hub": {"status": state.INSTALLED}}, {}):                                   # a new installation
        entries.clear()
        entries.update(others)
        assert not config.kept_forge(), others
    path.unlink()
    assert "keeps Forge" not in (config.ensure_config_file() or "")
    assert path.read_bytes() == paths.CONFIG_TEMPLATE.read_bytes()


def test_a_missing_file_is_made_from_the_template_and_never_overwritten(cfg: Callable[[str | bytes], Path],
                                                                       tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "spellcast3d-config.ini"
    assert config.load() == config.DEFAULTS and config.problems() == []            # no file: the defaults
    made = config.ensure_config_file()
    assert made is not None and "spellcast3d-config.ini in the Spellcast3D folder" in made
    assert path.read_bytes() == paths.CONFIG_TEMPLATE.read_bytes() and not list(tmp_path.glob("*.tmp"))
    assert config.image_backend() == "qwen"
    cfg(FORGE)
    assert config.ensure_config_file() is None and config.image_backend() == "forge"   # the artist's file stays
    # a place that cannot take it (its folder is a file): said, and the defaults are used
    (tmp_path / "x").write_text("a file, not a folder", encoding="utf-8")
    monkeypatch.setenv("SPELLCAST_CONFIG", str(tmp_path / "x" / "y.ini"))
    note = config.ensure_config_file()
    assert note is not None and note.startswith("Could not make") and "Qwen-Image 2.1" in note
    assert config.image_backend() == "qwen"


def test_another_file_can_be_named_and_the_cache_follows_edits(cfg: Callable[[str | bytes], Path], tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    other = tmp_path / "other.ini"
    other.write_text(FORGE, encoding="utf-8")
    monkeypatch.setenv("SPELLCAST_CONFIG", f'"{other}"')
    assert config.config_file() == other and config.image_backend() == "forge"
    first = config.load()
    assert config.load() is first                                                 # unchanged: not read again
    other.write_text(FORGE.replace("Q4_K_M\n", "Q8_0\n"), encoding="utf-8")
    os.utime(other, (1000, 1000))
    assert config.qwen_quality() == "Q8_0" and config.changed(first)
    monkeypatch.delenv("SPELLCAST_CONFIG")
    path = cfg(TEMPLATE)
    start = config.load()
    path.write_text(TEMPLATE.replace("; Change a line", ";  Change a line"), encoding="utf-8")   # comments only
    config._cache.clear()
    assert not config.changed(start)
    assert config.describe(start) == "Qwen-Image 2.1 (model size Q4_K_M, text encoder UD-Q4_K_XL)"
    assert config.describe(config.Config("forge", "Q4_K_M")) == "Forge Neo"
    assert config.describe(config.Config(None, "Q4_K_M")).startswith("none")


def test_the_running_hub_says_once_when_the_file_picks_something_else(cfg: Callable[[str | bytes], Path],
                                                                     tmp_path: Path) -> None:
    path = cfg(TEMPLATE)
    store = SettingsStore(paths.DEFAULTS_FILE, tmp_path / "settings.json", watch_config=True)
    assert store.warnings == [] and store.config is not None and store.config.backend == "qwen"
    # Forge's extra model folders are not checked for a hub that runs Qwen-Image 2.1
    store.patch({"forge": {"ckpt_dirs": [str(tmp_path / "typo")]}})
    assert store.warnings == []
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>Spellcast3D</h1>", encoding="utf-8")
    app = create_app({}, settings=store, store=ProjectStore(tmp_path / "projects"), static_dir=static,
                     logs_dir=tmp_path / "logs", gpu_probe=lambda: None)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        def warnings() -> list[str]:
            return client.get("/api/status").json()["settings_warnings"]

        assert warnings() == [] and client.get("/api/status").json()["config"]["changed"] is None
        path.write_text(FORGE, encoding="utf-8")
        os.utime(path, (2000, 2000))
        # ONE "start again" banner (the hub's ConfigWatch, from the snapshot the store took): the store itself
        # does not say it too. It names the new choice, and the page shows it on the Images stage (config.changed).
        said = config_changed(store.config, config.load())
        assert warnings() == [said] and store.warnings == []
        assert said == ("spellcast3d-config.ini now picks Forge Neo. Close Spellcast3D and start it again to switch "
                        "(it installs what that needs; Qwen-Image 2.1 runs until then).")
        status = client.get("/api/status").json()
        # what it runs with until it starts again, and what the page says meanwhile
        assert status["config"]["image_backend"] == "qwen" and status["config"]["changed"] == said
        path.write_text(TEMPLATE, encoding="utf-8")                                  # switched back: nothing to do
        os.utime(path, (3000, 3000))
        assert warnings() == []
    # a store for other files (the tests' own) does not read it, and keeps Forge's folder check
    plain = SettingsStore(paths.DEFAULTS_FILE, tmp_path / "plain.json")
    assert plain.config is None
    (tmp_path / "plain.json").write_text(json.dumps({"forge": {"ckpt_dirs": [str(tmp_path / "typo")]}}),
                                         encoding="utf-8")
    assert any("Forge finds no models" in w for w in plain.warnings)
    path.write_text(FORGE, encoding="utf-8")
    os.utime(path, (4000, 4000))
    forge_hub = SettingsStore(paths.DEFAULTS_FILE, tmp_path / "plain.json", watch_config=True)
    assert forge_hub.config is not None and forge_hub.config.backend == "forge"
    assert any("Forge finds no models" in w for w in forge_hub.warnings)


def test_what_the_hub_says_when_the_file_changed() -> None:
    qwen, forge = config.parse(TEMPLATE), config.parse(FORGE)
    big = config.parse(TEMPLATE.replace("quality = Q4_K_M", "quality = Q8_0"))
    typo = config.parse(TEMPLATE.replace("\nbackend = qwen ", "\nbackend = forg "))
    assert config_changed(forge, qwen) == ("spellcast3d-config.ini now picks Qwen-Image 2.1 (model size Q4_K_M, text "
                                           "encoder UD-Q4_K_XL). Close "
                                           "Spellcast3D and start it again to switch (it installs what that needs; "
                                           "Forge Neo runs until then).")
    assert config_changed(qwen, big) == ("spellcast3d-config.ini now picks Qwen-Image 2.1's model size Q8_0. Close "
                                         "Spellcast3D and start it again to use it (it downloads that size first; "
                                         "Q4_K_M runs until then).")
    said = config_changed(qwen, typo)
    assert said.startswith('spellcast3d-config.ini was changed and has a mistake now: spellcast3d-config.ini, line 6: '
                           '"forg" is not an image generator')
    assert said.endswith("then close Spellcast3D and start it again.")
    assert UNTIL_FIXED not in said and "  " not in said
    assert config_changed(typo, qwen).endswith("start it again to switch (it installs what that needs).")


def test_the_text_encoder_has_its_own_size() -> None:
    """text_encoder = one of TEXT_ENCODERS (any case, without "UD-", or the whole file name); a mistake means the
    default, said; a change of it is a change of the file (the hub says to start again)."""
    assert config.parse(TEMPLATE).text_encoder == config.DEFAULT_TEXT_ENCODER == "UD-Q4_K_XL"
    assert "text_encoder = UD-Q4_K_XL" in TEMPLATE
    for written, size in (("ud-q6_k_xl", "UD-Q6_K_XL"), ("Q6_K_XL", "UD-Q6_K_XL"), ("q8_0", "Q8_0"),
                          ("Qwen3-VL-8B-Instruct-UD-Q8_K_XL.gguf", "UD-Q8_K_XL")):
        cfg = config.parse(TEMPLATE.replace("text_encoder = UD-Q4_K_XL", f"text_encoder = {written}"))
        assert (cfg.text_encoder, cfg.problems, cfg.backend) == (size, (), "qwen")
    wrong = config.parse(TEMPLATE.replace("text_encoder = UD-Q4_K_XL", "text_encoder = Q9"))
    assert wrong.text_encoder == "UD-Q4_K_XL" and wrong.backend == "qwen"
    assert '"Q9" is not a size of the text encoder' in wrong.problems[0] and "UD-Q8_K_XL" in wrong.problems[0]
    assert config.TEXT_ENCODERS == ("UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0", "UD-Q8_K_XL")
