"""The Qwen-Image 2.1 bench (spellcast/profiling/qwen_bench.py): its arguments, the config copy it starts the server
with (only the [profiling] switches changed), the launch (the hub's own, but for --projects-root, the launcher's
switches it was asked to add and, with --pin, the shell's SD_PIN_PARAMS left out), its requests (the hub's, without
guidance, unless --cfg asks for guidance), its refusal when the installer has not put a profiler's package into the
qwen venv, and Ctrl+C during a run. Fakes only: no process is started.
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from spellcast import config, paths, procman
from spellcast.adapters import qwen
from spellcast.adapters.base import LaunchSpec
from spellcast.adapters.qwen import QwenAdapter
from spellcast.profiling import qwen_bench as bench

if str(paths.LAUNCHERS) not in sys.path:     # the launchers import their siblings as top-level modules
    sys.path.insert(0, str(paths.LAUNCHERS))

import qwen_server as launcher  # noqa: E402  (what the bench's requests and switches go to)

TEMPLATE = paths.CONFIG_TEMPLATE.read_text(encoding="utf-8")
ARTISTS = ("; my own notes\r\n[image generator]\r\nbackend = qwen   ; built in\r\n; backend = forge\r\n"
           "quality = Q5_K_M\r\n")


@pytest.fixture
def own_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The config file in use is the test's own (SPELLCAST_CONFIG, put back after the test: the bench sets it)."""
    file = tmp_path / "spellcast3d-config.ini"
    file.write_bytes(ARTISTS.encode("utf-8"))
    monkeypatch.setenv("SPELLCAST_CONFIG", str(file))
    return file


def test_the_arguments() -> None:
    args = bench.parse_args([])
    assert (args.profile, args.runs, args.seed, args.size, args.steps, args.cfg, args.negative, args.batch,
            args.references, args.transparent, args.port, args.pin, args.prefix_cache) == (
        "none", 2, 1234, 1024, 25, 1.0, "", 1, [], True, 0, None, None)
    # the recipe is the hub's unless told otherwise: Qwen's own, without guidance
    assert (args.steps, args.cfg) == (qwen.DEFAULT_STEPS, qwen.CFG_SCALE) == (launcher.STEPS, launcher.CFG_SCALE)
    args = bench.parse_args(["--profile", "torch+sync", "--runs", "3", "--prompt", "a lamp", "--size", "768",
                             "--references", "a.png", "b.png", "--transparent", "off", "--batch", "2"])
    assert (args.profile, args.runs, args.prompt, args.size, args.references, args.transparent, args.batch) == (
        "torch+sync", 3, "a lamp", 768, ["a.png", "b.png"], False, 2)
    args = bench.parse_args(["--steps", "20", "--cfg", "6", "--negative", "blurry", "--pin", "off", "--prefix-cache",
                             "q8_0"])                                              # the former recipe, as it ran
    assert (args.steps, args.cfg, args.negative, args.pin, args.prefix_cache) == (20, 6.0, "blurry", "off", "q8_0")
    assert (args.prompts, args.verbose, args.sdcpp) == (None, False, None)
    args = bench.parse_args(["--prompts", "a lamp", "a chair", "--verbose", "--sdcpp", "x", "--attention", "sage"])
    assert (args.prompts, args.verbose, args.prompt, args.sdcpp, args.attention) == (
        ["a lamp", "a chair"], True, bench.DEFAULT_PROMPT, "x", "sage")
    for wrong in (["--profile", "python+torch"], ["--size", "1000"], ["--runs", "0"], ["--transparent", "maybe"],
                  ["--cfg", "31"], ["--cfg", "-1"], ["--steps", "0"], ["--pin", "maybe"], ["--prefix-cache", "card"]):
        with pytest.raises(SystemExit):
            bench.parse_args(wrong)
    assert set(bench.PROFILES) == {"none", "python", "torch", "sync", "torch+sync", "python+sync"}
    assert {key for keys in bench.PROFILES.values() for key in keys} == set(config.PROFILING_KEYS)
    assert bench.switches_for("python+sync") == {"use_python_profiler": True, "use_torch_profiler": False,
                                                 "diagnostic_enable_sync_hunter": True}


def _differing(before: str, after: str) -> list[tuple[str, str]]:
    return [(a, b) for a, b in zip(before.splitlines(), after.splitlines()) if a != b]


def test_the_copy_changes_only_the_switches_of_the_template() -> None:
    text = bench.with_switches(TEMPLATE, bench.switches_for("torch"))
    assert len(text.splitlines()) == len(TEMPLATE.splitlines()) and text.endswith("\n")
    [(before, after)] = _differing(TEMPLATE, text)
    assert after == before.replace("use_torch_profiler = off", "use_torch_profiler = on", 1) != before   # value only
    assert config.parse(text).profiling == ("use_torch_profiler",)
    assert config.parse(bench.with_switches(text, bench.switches_for("none"))).profiling == ()


def test_the_copy_of_a_file_without_switches_gets_a_profiling_section() -> None:
    text = bench.with_switches(ARTISTS, bench.switches_for("python+sync"))
    assert text.startswith(ARTISTS) and "\r\n" in text and "\n" not in text.replace("\r\n", "")
    assert text[len(ARTISTS):].split("\r\n") == ["[profiling]", "use_python_profiler = on", "use_torch_profiler = off",
                                                 "diagnostic_enable_sync_hunter = on", ""]
    parsed = config.parse(text)
    assert parsed.profiling == ("use_python_profiler", "diagnostic_enable_sync_hunter")
    assert (parsed.backend, parsed.quality, parsed.problems) == ("qwen", "Q5_K_M", ())


def test_a_switch_set_twice_is_set_once() -> None:
    text = bench.with_switches("[profiling]\nuse_torch_profiler = off\n  USE_TORCH_PROFILER = on\n",
                               bench.switches_for("torch"))
    assert text.splitlines() == ["[profiling]", "use_python_profiler = off", "diagnostic_enable_sync_hunter = off",
                                 "use_torch_profiler = on", "; USE_TORCH_PROFILER = on"]
    assert config.parse(text).profiling == ("use_torch_profiler",) and not config.parse(text).problems


def test_the_config_copy_is_the_file_in_use_with_the_switches(tmp_path: Path, own_config: Path) -> None:
    folder = tmp_path / "bench"
    folder.mkdir()
    copy, seen = bench.write_config_copy(folder, bench.switches_for("sync"))
    assert copy == folder / "spellcast3d-config.ini" and copy.read_bytes().startswith(ARTISTS.encode("utf-8"))
    assert seen.profiling == ("diagnostic_enable_sync_hunter",) and seen.quality == "Q5_K_M"
    assert own_config.read_bytes() == ARTISTS.encode("utf-8")                     # the artist's file is untouched


@pytest.mark.parametrize("profile, script", [("none", paths.LAUNCHERS / "qwen_server.py"),
                                             ("torch", paths.PACKAGE / "profiling" / "qwen_profiled.py"),
                                             ("python+sync", paths.PACKAGE / "profiling" / "qwen_profiled.py")])
def test_the_launch_is_the_hubs_but_for_the_projects_folder(tmp_path: Path, own_config: Path, profile: str,
                                                            script: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "QWEN_SDCPP", tmp_path / "sdcpp")
    (tmp_path / "sdcpp").mkdir()
    folder = tmp_path / "bench"
    folder.mkdir()
    copy, _seen = bench.write_config_copy(folder, bench.switches_for(profile))
    spec = bench.server_launch(7790, {"ports": {}}, folder, copy)
    assert os.environ["SPELLCAST_CONFIG"] == str(copy) == spec.env["SPELLCAST_CONFIG"]    # the server reads the copy
    hub = QwenAdapter().launch_spec(7790, {"ports": {}})                 # as procman starts it, with that config
    assert spec.argv[1] == hub.argv[1] == str(script)
    at = hub.argv.index("--projects-root") + 1
    assert hub.argv[at] == str(paths.PROJECTS) and spec.argv[at] == str(folder)
    assert spec.argv[:at] + spec.argv[at + 1:] == hub.argv[:at] + hub.argv[at + 1:]
    assert "--quality" in spec.argv and spec.argv[spec.argv.index("--quality") + 1] == "Q5_K_M"   # the file's size
    # --sdcpp DIR: an engine folder under test instead of the installed one (the launch's --sdcpp, its working
    # folder and the head of its PATH), nothing else changed
    other = tmp_path / "other-engine"
    other.mkdir()
    test = bench.server_launch(7790, {"ports": {}}, folder, copy, (), other)
    assert test.argv[test.argv.index("--sdcpp") + 1] == str(other) and str(test.cwd) == str(other)
    assert test.env["PATH"].split(os.pathsep)[0] == str(other)
    assert test.env["PATH"].split(os.pathsep)[1:] == spec.env["PATH"].split(os.pathsep)[1:]
    assert spec.env["PATH"].split(os.pathsep)[0] == str(paths.QWEN_SDCPP) == str(spec.cwd)
    at = test.argv.index("--sdcpp") + 1
    assert test.argv[:at] + test.argv[at + 1:] == spec.argv[:at] + spec.argv[at + 1:]
    assert {k: v for k, v in test.env.items() if k != "PATH"} == {k: v for k, v in spec.env.items() if k != "PATH"}
    assert (spec.cwd, spec.env, spec.ready_timeout) == (hub.cwd, hub.env, hub.ready_timeout)


def test_the_launchers_switches_are_added_only_when_asked(tmp_path: Path, own_config: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """--pin and --prefix-cache go to the server as its own switches, at the end of the hub's launch; not given, the
    launch is the hub's (the launcher decides). The launcher knows every one of them."""
    monkeypatch.setattr(paths, "QWEN_SDCPP", tmp_path / "sdcpp")
    (tmp_path / "sdcpp").mkdir()
    folder = tmp_path / "bench"
    folder.mkdir()
    copy, _seen = bench.write_config_copy(folder, bench.switches_for("none"))
    assert bench.launcher_switches(bench.parse_args([])) == []
    plain = bench.server_launch(7790, {"ports": {}}, folder, copy)
    for given, extra in ((["--pin", "off"], ["--pin", "off"]), (["--prefix-cache", "q8_0"], ["--prefix-cache", "q8_0"]),
                         (["--prefix-cache", "f16", "--pin", "on"], ["--pin", "on", "--prefix-cache", "f16"]),
                         (["--pin", "auto", "--prefix-cache", "auto"], ["--pin", "auto", "--prefix-cache", "auto"]),
                         (["--attention", "flash"], ["--attention", "flash"]), (["--card-gb", "8"], ["--card-gb", "8"]),
                         (["--card-gb", "7.5", "--attention", "auto"], ["--attention", "auto", "--card-gb", "7.5"]),
                         (["--verbose", "--pin", "off"], ["--pin", "off", "--verbose"]),
                         (["--attention", "sage"], ["--attention", "sage"]),
                         (["--prefix-keep", "off"], ["--prefix-keep", "off"])):
        assert bench.launcher_switches(bench.parse_args(given)) == extra
        spec = bench.server_launch(7790, {"ports": {}}, folder, copy, extra)
        assert spec.argv == plain.argv + extra and (spec.cwd, spec.env) == (plain.cwd, plain.env)
        seen = launcher.parse_args(spec.argv[2:])                   # the server's own reading of its arguments
        assert [seen.pin, seen.prefix_cache, seen.attention] == [dict(zip(extra[::2], extra[1::2])).get(name, "auto")
                                                                 for name in ("--pin", "--prefix-cache", "--attention")]
        assert seen.card_gb == (float(extra[extra.index("--card-gb") + 1]) if "--card-gb" in extra else None)
        assert seen.verbose == ("--verbose" in extra)
        assert seen.prefix_keep == (extra[extra.index("--prefix-keep") + 1] if "--prefix-keep" in extra else "on")
    assert (launcher.parse_args(plain.argv[2:]).pin, launcher.parse_args(plain.argv[2:]).prefix_cache) == (
        "auto", "auto")


@pytest.mark.parametrize("theirs", ["diffusion", "all"])
def test_pin_decides_over_a_variable_of_the_benchs_own_shell(theirs: str, tmp_path: Path, own_config: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """Review finding: with SD_PIN_PARAMS set in the developer's shell the hub's launch passes it on and the launcher
    leaves a set one as it is, so --pin off and --pin on both ran pinned while the bench said "--pin off". With --pin
    the bench's launch is without the variable (the launcher then does what the switch says); without --pin the
    launch stays the hub's, the variable included."""
    assert bench.PIN_ENV == launcher.PIN_ENV == "SD_PIN_PARAMS"
    monkeypatch.setattr(paths, "QWEN_SDCPP", tmp_path / "sdcpp")
    (tmp_path / "sdcpp").mkdir()
    folder = tmp_path / "bench"
    folder.mkdir()
    copy, _seen = bench.write_config_copy(folder, bench.switches_for("none"))
    monkeypatch.setenv(bench.PIN_ENV, theirs)
    monkeypatch.setattr(launcher, "total_ram_mb", lambda: 31_909)
    plain = bench.server_launch(7790, {"ports": {}}, folder, copy)
    assert plain.env[bench.PIN_ENV] == theirs == QwenAdapter().launch_spec(7790, {"ports": {}}).env[bench.PIN_ENV]
    others = {name: value for name, value in plain.env.items() if name != bench.PIN_ENV}
    # auto on the 32 GB PC: the text encoder's weights page-locked too (PIN_TEXT_MIN_RAM_MB)
    for pin, at_load in (("off", None), ("on", launcher.PIN_PARAMS), ("auto", launcher.PIN_TEXT_PARAMS)):
        kept = dict(plain.env)                                      # what the finding was: the switch did nothing
        assert f"--pin {pin} never changes a set one" in launcher.set_pinning(pin, environ=kept)
        assert kept[bench.PIN_ENV] == theirs
        spec = bench.server_launch(7790, {"ports": {}}, folder, copy, bench.launcher_switches(
            bench.parse_args(["--pin", pin])))
        assert spec.env == others and spec.argv == plain.argv + ["--pin", pin]
        assert (spec.cwd, spec.ready_timeout) == (plain.cwd, plain.ready_timeout)
        environ = dict(spec.env)                                    # the server's own start, in that environment
        assert "already set" not in launcher.set_pinning(launcher.parse_args(spec.argv[2:]).pin, environ=environ)
        assert environ.get(bench.PIN_ENV) == at_load
    assert plain.env[bench.PIN_ENV] == theirs                       # the hub's launch itself was not touched
    # --prefix-cache alone says nothing about pinning: the hub's environment, the variable with it
    assert bench.server_launch(7790, {"ports": {}}, folder, copy, ["--prefix-cache", "q8_0"]).env == plain.env


def test_what_the_bench_says_about_the_shells_variable() -> None:
    assert bench.shell_pin_words(None, None) is None and bench.shell_pin_words("off", None) is None
    for pin in launcher.PINS:
        said = bench.shell_pin_words(pin, "diffusion")
        assert said is not None and said.startswith("SD_PIN_PARAMS='diffusion' of this shell is not passed on")
        assert f"--pin {pin} decides" in said and "\n" not in said
    kept = bench.shell_pin_words(None, "all")
    assert kept is not None and kept.startswith("SD_PIN_PARAMS='all' is set in this shell and goes to the server")
    assert "in every run the engine keeps in page-locked RAM what IT names" in kept
    assert "Give --pin auto, on or off" in kept and "\n" not in kept
    assert bench.shell_pin_words(None, "") is not None              # set but empty (not on Windows): theirs too


@pytest.mark.parametrize("given, shell, starts, says", [
    (["--pin", "off"], "diffusion", False, ["The server's switches: the hub's, and --pin off.",
                                            "SD_PIN_PARAMS='diffusion' of this shell is not passed on to the server: "
                                            "--pin off decides"]),
    (["--pin", "on", "--prefix-cache", "q8_0"], "all", False, [
        "The server's switches: the hub's, and --pin on --prefix-cache q8_0.",
        "SD_PIN_PARAMS='all' of this shell is not passed on to the server: --pin on decides"]),
    ([], "diffusion", True, ["The server's switches: the hub's (the launcher decides --pin and --prefix-cache).",
                             "WARNING: SD_PIN_PARAMS='diffusion' is set in this shell and goes to the server"]),
    (["--prefix-cache", "q8_0"], "diffusion", True, [
        "The server's switches: the hub's, and --prefix-cache q8_0.",
        "WARNING: SD_PIN_PARAMS='diffusion' is set in this shell and goes to the server"]),
    (["--pin", "off"], None, False, ["The server's switches: the hub's, and --pin off."]),
    ([], None, False, ["The server's switches: the hub's (the launcher decides --pin and --prefix-cache)."]),
])
def test_the_bench_starts_the_server_as_it_says_about_the_pin(given: list[str], shell: str | None, starts: bool,
                                                              says: list[str], tmp_path: Path, own_config: Path,
                                                              monkeypatch: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    """The real run_bench up to the server's start (a fake that keeps the environment it was given and ends the
    bench): what it prints about --pin and the shell's SD_PIN_PARAMS is what the server gets."""
    python = tmp_path / "python.exe"
    python.write_bytes(b"")
    folder = tmp_path / "bench"
    monkeypatch.setattr(paths, "venv_python", lambda venv_id: python)
    monkeypatch.setattr(paths, "QWEN_SDCPP", tmp_path / "sdcpp")
    (tmp_path / "sdcpp").mkdir()
    monkeypatch.setattr(bench, "PROFILING_ROOT", tmp_path / "profiling")
    monkeypatch.setattr(bench, "bench_folder", lambda: folder.mkdir() or folder)
    monkeypatch.setattr(bench, "SettingsStore", lambda: SimpleNamespace(get=lambda: {"ports": {paths.QWEN: 7790}}))
    monkeypatch.setattr(bench, "busy_gpu_warning", lambda: None)
    monkeypatch.setattr(bench, "hub_qwen_warning", lambda port: None)
    monkeypatch.setattr(procman, "port_is_free", lambda port: True)
    monkeypatch.setenv(bench.PIN_ENV, "theirs")          # put back as it was after the test, whatever it is now
    if shell is None:
        monkeypatch.delenv(bench.PIN_ENV)
    else:
        monkeypatch.setenv(bench.PIN_ENV, shell)
    started: list[tuple[list[str], dict[str, str]]] = []

    def spawn(argv: list[str], cwd: Path, env: dict[str, str], flags: int) -> Any:
        started.append((argv, env))
        raise bench.BenchError("the test ends here")

    monkeypatch.setattr(procman, "spawn_in_job", spawn)
    assert bench.main([*given, "--port", "7791"]) == 2
    captured = capsys.readouterr()
    assert captured.err.strip() == "qwen_bench: the test ends here"
    [(argv, env)] = started
    assert (bench.PIN_ENV in env) == starts and (not starts or env[bench.PIN_ENV] == shell)
    assert argv[argv.index("--projects-root") + 1] == str(folder)
    extra = bench.launcher_switches(bench.parse_args(given))
    assert argv[len(argv) - len(extra):] == extra
    lines = captured.out.splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("The server's switches: "))
    assert lines[at] == says[0]
    if len(says) > 1:
        assert lines[at + 1].startswith(says[1])
    else:
        assert "SD_PIN_PARAMS" not in captured.out


def test_the_requests_are_the_hubs(tmp_path: Path) -> None:
    args = bench.parse_args(["--prompt", "a lamp", "--size", "512", "--steps", "8", "--batch", "2",
                             "--transparent", "off", "--model", "full"])
    body = bench.request_body(args, 99, [], tmp_path / "run-1")
    assert {k: body[k] for k in ("width", "height", "steps", "cfg_scale", "negative_prompt", "seed", "batch_count",
                                 "transparent", "sampler")} == {
        "width": 512, "height": 512, "steps": 8, "cfg_scale": 1.0, "negative_prompt": "", "seed": 99,
        "batch_count": 2, "transparent": False, "sampler": "euler"}
    assert body["prompt"] == "a lamp"                                           # the words as typed, like the hub's
    assert body["out_dir"] == str((tmp_path / "run-1").resolve())
    assert body["model"] == "full"
    # without --cfg it is the hub's very body: Qwen's own recipe, no guidance; a new prop's model, the Turbo, with
    # its own 8 steps, and the full model at the hub's default steps. The bench keeps the transparent background on
    # (every baseline was measured with it), where a new prop has it off since 2026-10-31.
    plain = bench.request_body(bench.parse_args([]), 5, [], tmp_path / "run-1")
    params = {**bench.projects.STAGE_DEFAULTS["images"]["settings"], "prompt": bench.DEFAULT_PROMPT, "seed": 5,
              "transparent": True}
    assert plain == qwen.qwen_request(params, [], tmp_path / "run-1")
    assert (plain["model"], plain["steps"], plain["cfg_scale"], plain["negative_prompt"]) == ("turbo", 8, 1.0, "")
    assert launcher.condition_count(plain["cfg_scale"]) == 1 and not bench.guided(plain["cfg_scale"])
    full = bench.request_body(bench.parse_args(["--model", "full"]), 5, [], tmp_path / "run-1")
    assert full == qwen.qwen_request({**params, "qwen_model": "full"}, [], tmp_path / "run-1")
    assert (full["model"], full["steps"], full["cfg_scale"], full["first_block_cache"]) == ("full", 25, 1.0, None)


def test_a_recipe_with_guidance_can_still_be_measured(tmp_path: Path) -> None:
    """The hub never asks for guidance; the bench can (--cfg other than 1, then also --negative), for comparing a
    recipe with guidance - the former one: 20 steps at CFG 6 - with Qwen's own. The launcher takes such a request."""
    former = bench.request_body(bench.parse_args(["--model", "full", "--steps", "20", "--cfg", "6"]), 1, [],
                                tmp_path / "run-1")
    assert (former["steps"], former["cfg_scale"], former["negative_prompt"]) == (20, 6.0, "")
    avoid = bench.request_body(bench.parse_args(["--model", "full", "--steps", "20", "--cfg", "6", "--negative",
                                                 " blurry, text "]), 1, [], tmp_path / "run-1")
    assert (avoid["cfg_scale"], avoid["negative_prompt"]) == (6.0, "blurry, text")
    assert {k: v for k, v in avoid.items() if k != "negative_prompt"} == {
        k: v for k, v in former.items() if k != "negative_prompt"}
    # without guidance the negative prompt would not be read: not sent
    unused = bench.request_body(bench.parse_args(["--model", "full", "--negative", "blurry"]), 1, [],
                                tmp_path / "run-1")
    assert (unused["cfg_scale"], unused["negative_prompt"]) == (1.0, "")
    # the Turbo takes no guidance and draws its own 8 steps: the full model's --steps and --cfg are not sent for it
    turbo = bench.request_body(bench.parse_args(["--steps", "20", "--cfg", "6", "--negative", "blurry"]), 1, [],
                               tmp_path / "run-1")
    assert (turbo["model"], turbo["steps"], turbo["cfg_scale"], turbo["negative_prompt"]) == ("turbo", 8, 1.0, "")
    for body, conditions in ((former, 2), (avoid, 2), (unused, 1)):
        request = launcher.GenerateRequest(**body)                      # the launcher's own check of the body
        assert launcher.condition_count(request.cfg_scale) == conditions == (2 if bench.guided(body["cfg_scale"])
                                                                             else 1)
        assert request.negative_prompt == body["negative_prompt"]
    below = bench.parse_args(["--model", "full", "--cfg", "0.5", "--negative", "blurry"])   # below 1: two passes too
    assert bench.guided(below.cfg) and bench.request_body(below, 1, [], tmp_path / "x")["negative_prompt"] == "blurry"
    assert bench.guidance_words(bench.parse_args([])) == "no guidance"
    assert bench.guidance_words(bench.parse_args(["--cfg", "6"])) == "CFG 6 with no negative prompt"
    assert bench.guidance_words(bench.parse_args(["--cfg", "6", "--negative", "blurry"])) == (
        "CFG 6 with the negative prompt 'blurry'")


def _site(tmp_path: Path, *installed: str, record: bool = True) -> Path:
    site = tmp_path / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    for name in installed:
        info = site / f"{name}-1.0.dist-info"
        info.mkdir()
        if record:
            (info / "RECORD").write_text("", encoding="utf-8")
    return site


def test_what_a_profile_needs_in_the_qwen_venv(tmp_path: Path) -> None:
    site = _site(tmp_path, "pyinstrument")
    assert bench.missing_for("none", site) == [] and bench.missing_for("python", site) == []
    assert bench.missing_for("torch", site) == bench.missing_for("sync", site) == ["torch"]
    assert bench.missing_for("python+sync", site) == ["torch"]
    half = _site(tmp_path / "half", "torch", record=False)               # pip stopped before it finished
    assert bench.missing_for("torch", half) == ["torch"]
    assert bench.missing_for("torch+sync", _site(tmp_path / "all", "torch")) == []


def test_a_profile_without_its_package_is_refused_before_anything_starts(tmp_path: Path, own_config: Path,
                                                                        monkeypatch: pytest.MonkeyPatch,
                                                                        capsys: pytest.CaptureFixture[str]) -> None:
    python = tmp_path / "python.exe"
    python.write_bytes(b"")
    monkeypatch.setattr(paths, "venv_python", lambda venv_id: python)
    monkeypatch.setattr(bench, "qwen_site_packages", lambda: _site(tmp_path, "pyinstrument"))
    monkeypatch.setattr(bench, "PROFILING_ROOT", tmp_path / "profiling")

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing may start")

    monkeypatch.setattr(procman, "spawn_in_job", never)
    monkeypatch.setattr(bench, "bench_folder", never)
    assert bench.main(["--profile", "torch"]) == 2
    err = capsys.readouterr().err
    assert "--profile torch needs torch" in err and "use_torch_profiler = on" in err and str(own_config) in err
    assert "close Spellcast3D and start it again" in err and "install.py --only qwen" in err
    assert os.environ["SPELLCAST_CONFIG"] == str(own_config)          # refused before the copy was made


def test_the_newest_report_of_a_run(tmp_path: Path) -> None:
    root = tmp_path / "profiling"
    (root / "bench-20261002-100000").mkdir(parents=True)
    (root / "20261002-100001-qwen").mkdir()
    before = bench.report_folders(root)
    assert bench.new_report(root, before) is None
    (root / "20261002-100500-qwen").mkdir()
    (root / "bench-20261002-100600").mkdir()
    assert bench.new_report(root, before) == root / "20261002-100500-qwen"


class _Answer:
    def __init__(self, data: dict[str, Any]) -> None:
        self.status_code, self._data, self.text = 200, data, json.dumps(data)

    def json(self) -> dict[str, Any]:
        return self._data


@pytest.mark.parametrize("stopped_in", [1, 2])
def test_ctrl_c_during_a_run_is_a_stopped_bench_even_on_the_last_run(stopped_in: int, tmp_path: Path,
                                                                      own_config: Path,
                                                                      monkeypatch: pytest.MonkeyPatch,
                                                                      capsys: pytest.CaptureFixture[str]) -> None:
    """Finding 8: Ctrl+C while run 2 of 2 drew returned 0 with run 2 looking fine in the table. The real run_bench and
    one_run; the server, its process and httpx are fakes: Ctrl+C comes from the progress poll of run `stopped_in`."""
    python = tmp_path / "python.exe"
    python.write_bytes(b"")
    folder = tmp_path / "bench"
    monkeypatch.setattr(paths, "venv_python", lambda venv_id: python)
    monkeypatch.setattr(bench, "PROFILING_ROOT", tmp_path / "profiling")
    monkeypatch.setattr(bench, "bench_folder", lambda: folder.mkdir() or folder)
    monkeypatch.setattr(bench, "SettingsStore", lambda: SimpleNamespace(get=lambda: {"ports": {paths.QWEN: 7790}}))
    monkeypatch.setattr(bench, "busy_gpu_warning", lambda: None)
    monkeypatch.setattr(bench, "hub_qwen_warning", lambda port: None)
    monkeypatch.setattr(procman, "port_is_free", lambda port: True)
    monkeypatch.setattr(bench, "missing_models", lambda args, folder, quality, flux3=None: [])  # whatever this PC has
    monkeypatch.setattr(bench, "server_launch",
                        lambda port, settings, root, copy, extra=(), sdcpp=None, models_dir=None, flux3_dir=None: LaunchSpec(
                            argv=["python", "qwen_server.py"], cwd=tmp_path, env={}, ready_timeout=5.0))
    proc = SimpleNamespace(stdout=io.BytesIO(b"Model files read ahead in 1.0 s\n"), pid=1, returncode=None,
                           poll=lambda: None)
    monkeypatch.setattr(procman, "spawn_in_job", lambda argv, cwd, env, flags: proc)
    monkeypatch.setattr(bench, "wait_ready", lambda base_url, process, output, timeout: 2.0)
    stopped: list[str] = []
    cancelled: list[str] = []
    monkeypatch.setattr(bench, "stop_server", lambda base_url, process: stopped.append(base_url))
    monkeypatch.setattr(bench, "cancel_picture", lambda base_url: cancelled.append(base_url))
    monkeypatch.setattr(bench, "POLL_INTERVAL", 0.01)
    drawing = threading.Event()                          # released at the end: the bench's POST thread then ends
    posts: list[int] = []

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def post(self, url: str, json: dict[str, Any], timeout: Any) -> _Answer:
            posts.append(json["seed"])
            if len(posts) < stopped_in:
                return _Answer({"seconds": 12.5, "timings": {"drawing_1": 11.0}, "images": ["a.png"]})
            drawing.wait(10)                             # still drawing when Ctrl+C comes
            return _Answer({})

        def get(self, url: str, timeout: Any) -> _Answer:
            if len(posts) >= stopped_in:
                raise KeyboardInterrupt
            return _Answer({"phase": "drawing", "message": "Drawing"})

    monkeypatch.setattr(bench.httpx, "Client", Client)
    try:
        code = bench.main(["--runs", "2"])
    finally:
        drawing.set()
    out = capsys.readouterr().out
    assert code == bench.INTERRUPTED == 130 and cancelled and stopped
    rows = [line for line in out.splitlines() if line[:2] in ("1 ", "2 ")]
    assert len(rows) == stopped_in and rows[-1].endswith(bench.STOPPED)
    assert stopped_in == 1 or "12.5" in rows[0]


def test_each_run_gets_its_prompt_and_is_a_new_description_when_it_differs(tmp_path: Path) -> None:
    """--prompts: run N takes the Nth prompt, the list repeated; a run whose prompt is not the run before's is a new
    description (the text encoder runs again), a run with the same prompt a re-roll. Without --prompts every run is
    --prompt's, and only run 1 is a new description."""
    args = bench.parse_args(["--prompts", "a lamp", "a lamp", "a chair"])
    assert [bench.run_prompt(args, n) for n in range(1, 6)] == ["a lamp", "a lamp", "a chair", "a lamp", "a lamp"]
    runs = [bench.Run(n, 1233 + n, new_description=n == 1 or bench.run_prompt(args, n) != bench.run_prompt(args, n - 1))
            for n in range(1, 6)]
    assert [r.conditioning for r in runs] == ["new description", "re-roll", "new description", "new description",
                                              "re-roll"]
    assert bench.request_body(args, 7, [], tmp_path / "run-3", bench.run_prompt(args, 3))["prompt"] == "a chair"
    plain = bench.parse_args(["--prompt", "a lamp"])
    assert [bench.run_prompt(plain, n) for n in (1, 2, 3)] == ["a lamp"] * 3
    assert [bench.Run(n, n).conditioning for n in (1, 2, 3)] == ["new description", "re-roll", "re-roll"]
    assert bench.request_body(plain, 7, [], tmp_path / "run-1")["prompt"] == "a lamp"


def test_each_run_gets_its_model_and_a_switch_is_a_new_model(tmp_path: Path) -> None:
    """--model: run N takes the Nth model, the list repeated (turbo by default); a run whose model is not the one the
    launcher holds switches first (the table's "new model" and its switch seconds); a model whose file is not in the
    models folder (or --models-dir) refuses the bench before it starts anything."""
    args = bench.parse_args(["--model", "turbo", "full", "--runs", "3"])
    assert [bench.run_model(args, n) for n in range(1, 5)] == ["turbo", "full", "turbo", "full"]
    assert bench.parse_args([]).model == ["turbo"] and bench.run_model(bench.parse_args([]), 3) == "turbo"
    with pytest.raises(SystemExit):
        bench.parse_args(["--model", "fast"])
    assert bench.request_body(args, 7, [], tmp_path / "run-2", model=bench.run_model(args, 2))["model"] == "full"
    assert bench.model_words(args) == "turbo (its own 8 steps), then full (25 steps) (repeated)"
    assert bench.Run(2, 1, model="full", new_model=True).conditioning == "new model"
    runs = [bench.Run(1, 1234, model="turbo", seconds=14.0, timings={"prompt": 1.0, "drawing_1": 9.6}),
            bench.Run(2, 1235, model="full", new_model=True, seconds=36.0,
                      timings={"switch": 3.2, "prompt": 1.1, "drawing_1": 30.0})]
    text = bench.summary(runs, 2.1)
    header, first, second = text.splitlines()[1:4]
    assert header.split()[:7] == ["run", "seed", "model", "conditioning", "seconds", "wall", "switch"]
    assert first.split()[:3] == ["1", "1234", "turbo"] and second.split()[:4] == ["2", "1235", "full", "new"]
    assert "3.2" in second and "switch" in text.splitlines()[0]
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    assert bench.missing_models(args, models_dir, "Q4_K_M") == ["qwen-image-2.1-Q4_K_M.gguf",
                                                                "qwen-image-2.1-turbo-Q4_K_M.gguf"]
    (models_dir / "qwen-image-2.1-turbo-Q4_K_M.gguf").write_bytes(b"")
    assert bench.missing_models(bench.parse_args([]), models_dir, "Q4_K_M") == []
    # --models-dir: the launch's --models names that folder, nothing else changes
    spec = LaunchSpec(argv=["python", "qwen_server.py", "--projects-root", "P", "--models", "M", "--sdcpp", "S"],
                      cwd=tmp_path, env={}, ready_timeout=5.0)
    moved = bench.bench_launch(spec, tmp_path / "bench", models_dir=models_dir)
    assert moved.argv == ["python", "qwen_server.py", "--projects-root", str(tmp_path / "bench"), "--models",
                          str(models_dir), "--sdcpp", "S"]


def test_the_summary_says_which_run_is_a_reroll() -> None:
    runs = [bench.Run(1, 1234, seconds=120.4, wall=121.0, timings={"prompt": 4.2, "drawing_1": 110.0},
                      report=Path("20261002-100500-qwen")),
            bench.Run(2, 1235, failure="oom: out of memory")]
    text = bench.summary(runs, 2.1)
    assert text.startswith("Ready in 2.1 s.") and "new description" in text and "re-roll" in text
    assert "20261002-100500-qwen" in text and "oom: out of memory" in text and "110.0" in text
