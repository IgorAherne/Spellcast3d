"""Qwen-Image 2.1: its launcher (launchers/qwen_server.py) and the ctypes mirror of stable-diffusion.cpp's C API
(launchers/qwen_sdcpp.py).

Runs in the hub venv (fastapi + pydantic + httpx, no Pillow, no DLL): what reads or writes pictures needs Pillow and
is skipped there (it runs in the qwen venv:  <venv>\\Scripts\\python.exe -m pytest -p no:cacheprovider
code/tests/test_qwen_launcher.py). The layout check against the real stable-diffusion.dll runs when the pinned
build is unpacked (code/venvs/qwen/sdcpp, or the folder in SPELLCAST_SDCPP); the model itself runs only on the GPU
(the launcher's own run, not a test).
"""

from __future__ import annotations

import ctypes
import importlib.util
import io
import json
import os
import sys
import types
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from spellcast import paths

if str(paths.LAUNCHERS) not in sys.path:     # the launchers import their siblings as top-level modules
    sys.path.insert(0, str(paths.LAUNCHERS))

import qwen_sdcpp as sdcpp  # noqa: E402
import qwen_server as server  # noqa: E402

HAVE_PIL = importlib.util.find_spec("PIL") is not None
needs_pil = pytest.mark.skipif(not HAVE_PIL, reason="needs Pillow (the qwen venv)")
SDCPP = Path(os.environ.get("SPELLCAST_SDCPP") or paths.CODE / "venvs" / "qwen" / "sdcpp")
needs_dll = pytest.mark.skipif(os.name != "nt" or not (SDCPP / sdcpp.DLL_NAME).is_file(),
                               reason="needs the pinned stable-diffusion.cpp build (code/venvs/qwen/sdcpp)")
GiB = 1024


# ---- the prompt, the seeds and the placement ----------------------------------------------------------------------

def test_transparent_prompts_use_qwens_own_words() -> None:
    assert server.wrap_prompt("a wooden treasure chest", True) == (
        "This is an RGBA image with transparency. a wooden treasure chest. The image has alpha channel and the "
        "background is transparent.")
    assert server.wrap_prompt("  a red apple.  \n", True).count(".") == 3      # one full stop of its own, kept once
    wrapped = server.wrap_prompt("a lamp", True)
    assert server.wrap_prompt(wrapped, True) == wrapped                         # never twice
    assert server.wrap_prompt("a  lamp\non a desk", False) == "a lamp on a desk"


def test_variations_get_consecutive_seeds() -> None:
    assert server.batch_seeds(41, 3) == [41, 42, 43]
    assert server.batch_seeds(7, 1) == [7]


def test_omnitextures_sheets_are_as_large_as_their_painting_fits_the_budget() -> None:
    """health "sheet_side": the largest square picture Omni-Texture may ask for. On an RTX 3060 held to a budget of
    6,144 MB (an 8 GB card) a 2048x2048 sheet was painted (2026-11-16, again 2026-11-22); its lighting comes off
    in a smaller pass (albedo_size), which bounds nothing."""
    assert server.sheet_side(6_144) == server.sheet_side(10_224) == server.sheet_side(20_000) == 2048 == server.SHEET_SIDE_MAX
    assert server.sheet_side(4_500) == 1792 and server.sheet_side(3_000) == 1408
    assert server.sheet_side(2_000) == server.sheet_side(0) == 1024
    assert all(server.sheet_side(mb) % 128 == 0 for mb in range(1_000, 16_000, 250))


def test_a_picture_larger_than_a_views_goes_through_the_albedo_pass_at_three_quarters() -> None:
    """albedo_size: a view's picture as it is; a larger one at 3/4 of its sides, where the pass takes what the
    painting of the whole picture took, and never fewer pixels than a view's picture."""
    assert server.albedo_size((1024, 1024)) == (1024, 1024) and server.albedo_size((512, 2048)) == (512, 2048)
    assert [server.albedo_size((side, side))[0] for side in (1152, 1280, 1408, 1536, 1792, 2048)] == [
        1024, 1024, 1056, 1152, 1344, 1536]
    assert server.albedo_size((2048, 1024)) == (1536, 768)
    assert all(side % server.SIDE_STEP == 0 for n in range(1024, 2049, 32) for side in server.albedo_size((n, n)))


@needs_pil
def test_a_sheets_lighting_comes_off_in_one_pass_and_the_paintings_fine_detail_comes_back() -> None:
    """lighting_off: one pass over all of the picture whatever its size (a sheet's figures are judged together); a
    large picture goes through smaller, and comes back at its size with the detail the way down cost it."""
    from PIL import Image

    sizes: list[tuple[int, int]] = []

    def albedo(part: Any) -> Any:
        sizes.append(part.size)
        return Image.new("RGB", part.size, (100, 110, 120))

    view = Image.new("RGB", (1024, 1024), (200, 100, 50))
    assert server.lighting_off(albedo, view).getpixel((5, 5)) == (100, 110, 120) and sizes == [(1024, 1024)]
    sheet = Image.new("RGB", (2048, 2048), (200, 100, 50))
    for x in range(400, 464, 2):                     # lines a pixel wide: gone at 3/4 of the size
        sheet.paste((250, 150, 100), (x, 400, x + 1, 464))
    out = server.lighting_off(albedo, sheet)
    assert sizes[1:] == [(1536, 1536)] and out.size == sheet.size
    assert out.getpixel((1000, 1000)) == (100, 110, 120), "flat where the painting is: the pass's own colour"
    line, gap = out.getpixel((432, 432)), out.getpixel((433, 432))
    assert all(a - b > 25 for a, b in zip(line, gap)), "the lines are back, on the pass's colour"


@pytest.mark.parametrize("free, budget", [
    (23 * GiB, 22 * GiB),               # a 24 GB card: the copies on the card stay there, all fit
    (15 * GiB, 14 * GiB),
    (11 * GiB, 10 * GiB),               # an RTX 3060
    (12 * GiB, 11 * GiB),               # a big card that other programs fill
    (7 * GiB, 6 * GiB),                 # an 8 GB card
])
def test_the_weights_live_in_ram_and_the_prefix_cache_is_the_same_on_every_card(free: int, budget: int) -> None:
    placement = server.choose_placement(free, 10_664)
    assert placement == {"name": "offload", "params_backend": "*=cpu", "max_vram": "-1", "budget_mb": budget,
                         "prefix_cache": "auto", "prefix_keep": "on"}
    # sd.cpp's own choice of the type (f16 under flash attention) on every card: no model argument for it, only the
    # cache kept between pictures. Without guidance there is one condition, so cards of 16 GB and less no longer
    # need q8_0's half of the memory
    assert server.PREFIX_CACHE == "auto" and server.model_args(placement) == "qwen_image_2_1_prefix_cache_keep=on"
    assert not hasattr(server, "BIG_CARD_MB")


def test_the_placement_can_be_forced() -> None:
    on_card = server.choose_placement(23 * GiB, 10_664, "gpu")
    assert on_card["name"] == "gpu" and on_card["budget_mb"] == 23 * GiB - 512 - 10_664     # the rest of the card
    assert on_card["params_backend"] is None and on_card["max_vram"] is None
    forced = server.choose_placement(7 * GiB, 10_664, "gpu", max_vram="6")
    assert forced == {"name": "gpu", "params_backend": None, "max_vram": "6", "prefix_cache": "auto",
                      "prefix_keep": "on", "budget_mb": 6 * GiB - 10_664}
    assert server.choose_placement(11 * GiB, 10_664, "auto", max_vram="9")["budget_mb"] == 9 * GiB
    assert server.budget_mb(8000, None, 512) == 7488 and server.budget_mb(8000, "-2", 512) == 8000 - 2048
    assert server.budget_mb(8000, "20", 512) == 7488 and server.budget_mb(8000, "x", 512) == 7488


def test_the_prefix_cache_type_stays_an_override() -> None:
    """--prefix-cache: auto (the default) leaves the type to sd.cpp; f16 and q8_0 are said to it, and the placement
    names the type for the reports."""
    assert server.parse_args(["--sdcpp", "s", "--models", "m", "--port", "1"]).prefix_cache == "auto"
    for cache, said in (("auto", None), ("f16", "qwen_image_2_1_prefix_cache_type=f16"),
                        ("q8_0", "qwen_image_2_1_prefix_cache_type=q8_0")):
        args = server.parse_args(["--sdcpp", "s", "--models", "m", "--port", "1", "--prefix-cache", cache])
        placement = server.choose_placement(11 * GiB, 10_664, prefix_cache=args.prefix_cache, prefix_keep="auto")
        assert placement["prefix_cache"] == cache and server.model_args(placement) == said
    with pytest.raises(SystemExit):                 # "card" (by the card's size) is no choice any more
        server.parse_args(["--sdcpp", "s", "--models", "m", "--port", "1", "--prefix-cache", "card"])


def test_the_prefix_cache_is_kept_between_pictures_unless_told_otherwise() -> None:
    """--prefix-keep on (the default, PREFIX_KEEP): the engine keeps the prefix cache and the example pictures' latents
    from one picture to the next with the same conditioning (qwen_image_2_1_prefix_cache_keep=on); auto leaves the
    engine its own rule (nothing said); off says so. Said next to the cache's type when both are overrides."""
    assert server.PREFIX_KEEP == "on" and server.PREFIX_KEEPS == ("auto", "on", "off")
    assert server.parse_args(["--sdcpp", "s", "--models", "m", "--port", "1"]).prefix_keep == "on"
    for keep, said in (("on", "qwen_image_2_1_prefix_cache_keep=on"), ("off", "qwen_image_2_1_prefix_cache_keep=off"),
                       ("auto", None)):
        args = server.parse_args(["--sdcpp", "s", "--models", "m", "--port", "1", "--prefix-keep", keep])
        placement = server.choose_placement(11 * GiB, 10_664, prefix_cache=args.prefix_cache,
                                            prefix_keep=args.prefix_keep)
        assert placement["prefix_keep"] == keep and server.model_args(placement) == said
    both = server.choose_placement(11 * GiB, 10_664, prefix_cache="q8_0", prefix_keep="on")
    assert server.model_args(both) == "qwen_image_2_1_prefix_cache_type=q8_0,qwen_image_2_1_prefix_cache_keep=on"
    assert server.model_args({"prefix_cache": "auto", "prefix_keep": "auto"}) is None
    assert server.model_args({}) is None
    with pytest.raises(SystemExit):
        server.parse_args(["--sdcpp", "s", "--models", "m", "--port", "1", "--prefix-keep", "maybe"])


def test_the_vae_decode_is_tiled_where_it_would_not_fit() -> None:
    assert not server.needs_vae_tiling(1024, 1024, 10 * GiB)            # an RTX 3060 with the weights in RAM
    assert server.needs_vae_tiling(1024, 1024, 6 * GiB)                 # an 8 GB card
    assert not server.needs_vae_tiling(512, 512, 6 * GiB)
    # with direct convolutions (half the memory) the 8 GB card's decode fits in one piece; 2048x2048 needs tiles
    assert not server.needs_vae_tiling(1024, 1024, 6 * GiB, direct=True)
    assert server.needs_vae_tiling(2048, 2048, 6 * GiB, direct=True)


def test_direct_convolutions_only_where_they_save_the_tiles() -> None:
    """vae_direct: the 8 GB card (the usual decode needs tiles, the direct one fits) decodes with direct convolutions;
    the RTX 3060 (the usual one fits, ~0.4 s faster) and a card where neither fits keep the usual one."""
    assert server.vae_direct(6 * GiB)
    assert not server.vae_direct(10 * GiB) and not server.vae_direct(3 * GiB)
    assert server.vae_direct(6 * GiB, 1024, 1024) and not server.vae_direct(6 * GiB, 512, 512)
    # GTX 10xx (Pascal): always the direct convolutions, in tiles where one piece does not fit (its usual form in
    # tiles was 4x slower on a GTX 1080 held to 6 GB); Volta keeps the usual ones
    assert server.vae_direct(6 * GiB, capability=(6, 1)) and server.vae_direct(10 * GiB, capability=(6, 1))
    assert server.vae_direct(3 * GiB, capability=(6, 1)) and server.vae_direct(6 * GiB, 4096, 4096, capability=(6, 1))
    assert server.needs_vae_tiling(1024, 1024, 4 * GiB, direct=server.vae_direct(4 * GiB, capability=(6, 1)))
    assert not server.vae_direct(6 * GiB, capability=(7, 0)) and not server.vae_direct(10 * GiB, capability=(7, 0))
    assert server.vae_direct(6 * GiB, capability=(7, 5)) and server.vae_direct(6 * GiB, capability=(12, 0))


def test_the_weights_stay_on_the_card_where_the_decode_fits_beside_them() -> None:
    """keep_params: auto keeps the diffusion weights on the card between runs where the budget holds them beside the
    direct decode, the VAE's weights and the drawing's working memory (a 12 GB card), not on an 8 GB card (its decode
    needs their room every picture); on / off always / never. The variable goes into the environment once and a set
    one is never changed. With kept weights vae_direct takes the direct convolutions wherever the decode fits beside
    them in one piece (the usual decode's im2col buffers would evict the weights every picture)."""
    assert server.keep_params("auto", 10 * GiB, 4005)
    assert not server.keep_params("auto", 6 * GiB, 4005)
    assert not server.keep_params("auto", 10 * GiB, 0), "no denoiser known: nothing to keep"
    assert server.keep_params("on", 6 * GiB, 4005) and not server.keep_params("off", 10 * GiB, 4005)
    # an engine that ignores the variable (the pinned build before the release that reads it): auto stays off
    assert not server.keep_params("auto", 10 * GiB, 4005, engine_knows=False)
    assert server.keep_params("on", 10 * GiB, 4005, engine_knows=False)
    assert isinstance(server.engine_pin.ENGINE_KEEPS_PARAMS, bool)
    env: dict[str, str] = {}
    assert server.KEEP_ENV in server.set_keep(True, env) and env == {server.KEEP_ENV: "1"}
    env = {}
    assert "freed" in server.set_keep(False, env) and env == {}
    env = {server.KEEP_ENV: "0"}
    assert "already set" in server.set_keep(True, env) and env == {server.KEEP_ENV: "0"}
    # the RTX 3060's 10 GB budget: the usual decode without kept weights, the direct one beside the 4 GB kept
    assert not server.vae_direct(10 * GiB) and server.vae_direct(10 * GiB, kept_mb=4005)
    # beside them it does not fit at 5 GB: the usual rule decides (there: tiles for im2col, direct in one piece)
    assert server.vae_direct(5 * GiB, kept_mb=4005) == server.vae_direct(5 * GiB) is True
    assert server.vae_direct(16 * GiB, kept_mb=4005) == server.vae_direct(16 * GiB) is False, \
        "the usual decode fits even beside them: no change"
    assert server.parse_args(["--sdcpp", "s", "--models", "m", "--port", "1"]).keep_params == "auto"


@pytest.fixture(autouse=True)
def _matmul_variable_put_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_runtime writes --matmul's variable into this process's environment: no test leaves it to the next."""
    monkeypatch.setenv(server.MATMUL_ENV, "theirs")
    monkeypatch.delenv(server.MATMUL_ENV)


def test_the_mmq_kernels_draw_on_every_card_unless_the_int8_path_is_asked_for(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The int8 path's pictures are not the model file's (MATMULS): no card takes it by default, and the engine is
    told so before its DLL is loaded. --matmul int8 leaves the engine its choice; a variable already set stays."""
    assert server.parse_args(["--sdcpp", "s", "--models", "m", "--port", "1"]).matmul == "mmq"
    assert server.matmul_choice("mmq", None) == ("off", f"matmuls: the MMQ kernels ({server.MATMUL_ENV}=off)")
    assert server.matmul_choice("int8", None)[0] is None
    assert server.matmul_choice("mmq", "on") == (None, f"matmuls: {server.MATMUL_ENV}=on was already set, kept")
    env = {server.MATMUL_ENV: "on"}
    server.set_matmul("mmq", env)
    assert env == {server.MATMUL_ENV: "on"}, "a variable already set is never changed"
    for name in (server.PIN_ENV, server.LORA_TYPE_ENV):         # load_runtime sets these too: put back after the test
        monkeypatch.setenv(name, "theirs")
        monkeypatch.delenv(name)
    seen: list[str | None] = []
    monkeypatch.setattr(sdcpp, "load", lambda folder: seen.append(os.environ.get(server.MATMUL_ENV)) or object())
    monkeypatch.setattr(sdcpp, "commit", lambda dll: sdcpp.PINNED_COMMIT[:7])
    monkeypatch.setattr(sdcpp, "check_layout", lambda dll: None)
    cards = ((6, 1), (7, 5), (8, 6), (8, 9), (12, 0), None)
    for capability in cards:
        monkeypatch.delenv(server.MATMUL_ENV, raising=False)
        server.load_runtime(tmp_path, capability=capability)
    monkeypatch.delenv(server.MATMUL_ENV)
    server.load_runtime(tmp_path, matmul="int8", capability=(8, 6))
    assert seen == ["off"] * len(cards) + [None]


def test_sage_attention_only_with_the_pinned_engine_on_a_card_that_has_it() -> None:
    """--attention auto (the default): the engine's SageAttention with the pinned build on a card of compute capability
    8.0 or newer, flash attention with any other build (upstream's sage kernel would keep an F32 prefix cache) and on
    older cards (the engine refuses to start when sage attention is asked without a kernel for the card: the launcher
    must not ask); flash: always flash attention; sage (developers): the sage op with a build under test that is not
    the pinned one, still never on an older card."""
    assert server.parse_args(["--sdcpp", "s", "--models", "m", "--port", "1"]).attention == "auto"
    assert server.attention_choice("auto", pinned=True, capability=(8, 6)) == "sage"
    assert server.attention_choice("auto", pinned=False, capability=(8, 6)) == "flash"
    assert server.attention_choice("flash", pinned=True, capability=(8, 6)) == "flash"
    assert server.attention_choice("sage", pinned=False, capability=(8, 6)) == "sage"
    assert server.attention_choice("sage", pinned=False, capability=(7, 5)) == "flash"
    assert server.ATTENTIONS == ("auto", "flash", "sage")
    with pytest.raises(TypeError):                                   # the card's capability is never assumed
        server.attention_choice("auto", True)  # type: ignore[call-arg]
    for capability in ((8, 0), (8, 6), (8, 9), (9, 0), (12, 0), [8, 6]):      # A100, RTX 30 / 40, H100, RTX 50
        assert server.attention_choice("auto", True, capability) == "sage"
        assert server.attention_choice("flash", True, capability) == "flash"
    for capability in ((6, 1), (7, 0), (7, 5), [7, 5]):                     # GTX 10, V100, RTX 20 series
        assert server.attention_choice("auto", True, capability) == "flash"
        assert server.attention_choice("auto", False, capability) == "flash"
    assert server.SAGE_MIN_CAPABILITY == (8, 0)
    with pytest.raises(SystemExit):
        server.parse_args(["--sdcpp", "s", "--models", "m", "--port", "1", "--attention", "sage2"])


def test_a_request_without_guidance_has_one_condition() -> None:
    """sd.cpp skips the unconditional pass only at a CFG of exactly 1.0 (as the float32 it gets); any other value is
    two conditions per step, below 1.0 too."""
    assert server.condition_count(server.CFG_SCALE) == server.condition_count(1) == 1
    assert server.condition_count(1.0 + 1e-12) == 1               # 1.0 as a float32
    assert [server.condition_count(cfg) for cfg in (6.0, 1.5, 1.001, 0.5, 0.0)] == [2, 2, 2, 2, 2]


def test_several_example_pictures_are_made_smaller_to_fit_the_prefix_cache() -> None:
    full = 1024 * 1024
    # WITHOUT guidance (one condition, the product's way) on an RTX 3060 (~10 GB): two example pictures at the
    # picture's own size, three and more made smaller - their cache and the step that fills it would not fit
    assert server.reference_pixels(10 * GiB, "auto", 1, full) is None
    assert server.reference_pixels(10 * GiB, "auto", 2, full) is None
    assert server.reference_pixels(10 * GiB, "auto", 3, full) == server.reference_pixels(10 * GiB, "f16", 3, full, 1) \
        == 738_304                                                                      # 859x859
    assert server.reference_pixels(10 * GiB, "auto", 10, full) == 209_920
    assert server.reference_pixels(20 * GiB, "auto", 3, full) is None                   # a big budget holds more
    assert server.reference_pixels(20 * GiB, "auto", 5, full) == 1_030_656
    assert server.reference_pixels(2 * GiB, "auto", 10, full) == server.MIN_REFERENCE_TOKENS * 256
    assert server.reference_pixels(10 * GiB, "auto", 0, full) is None
    # A LARGER PICTURE leaves its example less room: its own tokens take theirs in every step. Measured on that
    # card: a 2048x2048 picture kept the cache of a 1024x1024 copy and lost that of a 1536x1536 one
    sides = {side: server.reference_pixels(10 * GiB, "auto", 1, side * side) for side in (1536, 2048)}
    assert sides == {1536: 1_915_648, 2048: 1_451_008}                                  # 1384 and 1205 px
    assert 1024 ** 2 < sides[2048] < 1536 ** 2
    assert server.reference_pixels(24 * GiB, "auto", 1, 2048 * 2048) is None
    # WITH guidance there are two caches: q8_0 ones take about what the one f16 cache takes, f16 ones twice that
    assert server.reference_pixels(10 * GiB, "q8_0", 2, full, 2) is None
    assert server.reference_pixels(10 * GiB, "q8_0", 3, full, 2) == 695_296             # 834x834
    assert server.reference_pixels(10 * GiB, "f16", 3, full, 2) == 427_520              # 654x654
    assert server.reference_pixels(10 * GiB, "q8_0", 3, full, 1) is None                # the override, one condition
    for cache in ("auto", "f16", "q8_0"):           # one condition never needs smaller pictures than two
        for count in (1, 2, 3, 5, 10):
            one, two = (server.reference_pixels(10 * GiB, cache, count, full, n) or full for n in (1, 2))
            assert one >= two
    # per token of the prefix AND per condition: f16 262,144 values of two bytes; q8_0 half of the former 0.544
    assert server.PREFIX_MB_PER_TOKEN == {"q8_0": 0.272, "auto": 0.5, "f16": 0.5}


def test_a_smaller_card_gives_the_example_pictures_the_copies_of_a_12_gb_card_as_far_as_it_holds_them() -> None:
    """The rule: an 8 GB card must paint as well as a 12 GB one. A 12 GB card whose desktop holds
    a gigabyte or so gives the copies an empty one does; a smaller budget gives what it holds itself, with no margin
    kept: on 6,144 MB (--card-gb 8) one example picture at the picture's own size, two at 824 px (864 px were
    measured to keep their cache there; a cautious estimate had given 464 px, and Omni-Texture's painting sat 3 px
    off the model), a 1536x1536 picture's at 1021 px (kept) and a 2048x2048 one's at 760 px (1024 px lost it)."""
    full = 1024 * 1024
    three = server.reference_pixels(10 * GiB, "auto", 3, full)
    assert server.reference_pixels(9 * GiB, "auto", 3, full) == three    # a 12 GB card, 1.5 GB of it the desktop's
    assert server.reference_pixels(6 * GiB, "auto", 1, full) is None
    assert server.reference_pixels(6 * GiB, "auto", 2, full) == 678_656  # 824 px: the engine takes 832 x 832
    assert server.reference_pixels(5 * GiB, "auto", 2, full) == 504_064
    assert server.reference_pixels(2 * GiB, "auto", 2, full) == server.MIN_REFERENCE_TOKENS * 256
    assert [server.reference_pixels(6 * GiB, "auto", 1, side * side) for side in (1536, 2048)] == [1_041_920, 577_024]
    for count in (1, 2, 3, 5, 10):                  # a larger budget never gives smaller copies
        for conditions in (1, 2):
            for side in (1024, 1536, 2048):
                sizes = [server.reference_pixels(budget, "auto", count, side * side, conditions) or side * side
                         for budget in (3 * GiB, 4 * GiB, 6 * GiB, 8 * GiB, 9 * GiB, 10 * GiB, 16 * GiB)]
                assert sizes == sorted(sizes), (count, conditions, side, sizes)
                assert sizes[4] == sizes[5]         # 9 GiB of budget: already the reference's


def test_the_estimate_of_a_request_with_example_pictures_is_what_the_engine_was_seen_to_take() -> None:
    """prefix_room against the cases measured on an RTX 3060 (reports/copy-size-2026-11-24): the cache kept or lost
    for a picture and a prefix (its one example picture's copy and the 60 tokens of its words), by the budget."""
    def fits(budget: int, side: int, copy: int) -> bool:
        return server.prefix_room(budget, side * side // 256, server.PREFIX_MB_PER_TOKEN["f16"]) >= copy * copy // 256 + 60

    kept = [(10_224, 1536, 1024), (10_224, 1536, 1376), (10_224, 2048, 608), (10_224, 2048, 1024),
            (6_144, 1536, 1024), (6_144, 2048, 608)]
    lost = [(10_224, 2048, 1536), (6_144, 2048, 1024), (6_144, 1536, 1536)]
    assert all(fits(*case) for case in kept) and not any(fits(*case) for case in lost)


class _FakeReference:
    """What generate() reads of an example picture (a PIL RGBA image; opaque_reference makes RGB of it for FLUX.3)."""
    width = height = 8
    bands = "RGBA"
    opaque = True

    def tobytes(self) -> bytes:
        return bytes(len(self.bands) * self.width * self.height)

    def getbands(self) -> tuple[str, ...]:
        return tuple(self.bands)

    def getextrema(self) -> tuple[tuple[int, int], ...]:
        return ((0, 255),) * 3 + ((255 if self.opaque else 0, 255),)

    def convert(self, mode: str) -> "_FakeReference":
        copy = _FakeReference()
        copy.bands, copy.opaque, copy.width, copy.height = mode, self.opaque, self.width, self.height
        return copy

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def resize(self, size: tuple[int, int], resample: Any = None) -> "_FakeReference":
        copy = self.convert(self.bands)
        copy.width, copy.height = size
        return copy


class _RecordingLib:
    """The DLL as generate() uses it, keeping what each generate_image call was given (and returning no picture:
    generate() then raises, after the call seen here)."""

    def __init__(self) -> None:
        self.given: list[dict[str, Any]] = []

    @staticmethod
    def str_to_sample_method(name: bytes) -> int:
        return sdcpp.EULER_SAMPLE_METHOD if name == b"euler" else 3

    @staticmethod
    def sd_img_gen_params_init(pointer: Any) -> None:
        """The DLL's sd_img_gen_params_init (src/stable-diffusion.cpp): zeros, and the sample params' defaults that
        mean "the model's own" - so a test sees whether generate() left them alone (a shift of 0 would ruin a flow
        model, and a memset alone could not tell it from untouched)."""
        ctypes.memset(pointer, 0, ctypes.sizeof(sdcpp.SdImgGenParams))
        s = pointer.contents.sample_params
        s.guidance.txt_cfg, s.guidance.img_cfg, s.guidance.distilled_guidance = 7.0, float("inf"), 3.5
        s.scheduler, s.sample_method, s.sample_steps = sdcpp.SCHEDULER_COUNT, sdcpp.SAMPLE_METHOD_COUNT, 20
        s.eta, s.flow_shift = float("inf"), float("inf")

    def generate_image(self, ctx: Any, params: Any, out: Any, count: Any) -> bool:
        g = params._obj
        s = g.sample_params
        self.given.append({"cfg": s.guidance.txt_cfg, "steps": s.sample_steps, "method": s.sample_method,
                           "negative": g.negative_prompt, "references": g.ref_images_count,
                           "ref_image_args": g.ref_image_args,
                           "sizes": [(g.ref_images[i].width, g.ref_images[i].height)
                                     for i in range(g.ref_images_count)],
                           "sigmas": [round(s.custom_sigmas[i], 6) for i in range(s.custom_sigmas_count)]
                           if s.custom_sigmas else None,
                           "circular": (g.circular_x, g.circular_y), "seed": g.seed,
                           "tiles": g.vae_tiling_params.enabled and g.vae_tiling_params.tile_size_w,
                           "cache": os.environ.get("SD_QWEN_FIRST_BLOCK_CACHE"),
                           "drop": os.environ.get(server.PREFIX_DROP_ENV)})
        return False


def _fake_engine(monkeypatch: pytest.MonkeyPatch, lib: Any, capability: tuple[int, int] = (8, 6)) -> Any:
    """A QwenEngine (no DLL, no models) that generate() can run on: an RTX 3060's budget of 10 GiB."""
    pil = type(sys)("PIL")
    # generate() needs it for the pictures that come back, sized_reference for its filter's name
    pil.Image = types.SimpleNamespace(Resampling=types.SimpleNamespace(LANCZOS="lanczos"))
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setenv("SD_QWEN_FIRST_BLOCK_CACHE", "")       # generate() sets it; as it was again after the test
    monkeypatch.setenv(server.PREFIX_DROP_ENV, "")            # the same
    engine = server.QwenEngine.__new__(server.QwenEngine)
    engine.lib, engine.ctx, engine.progress = lib, 1, server.Progress()
    engine.placement = {**server.choose_placement(11 * GiB, 10_664), "capability": capability}
    return engine


def test_the_example_pictures_size_follows_the_requests_own_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    """generate() sizes the example pictures by the conditions of THIS request: one without guidance, two for a
    request of the bench that asks for guidance. The DLL is a fake that keeps what generate_image was given."""
    lib = _RecordingLib()
    engine = _fake_engine(monkeypatch, lib)
    for changes in ({}, {"cfg_scale": 6.0, "steps": 20, "negative_prompt": " blurry "}):
        req = server.GenerateRequest(prompt="a chest", model="full", out_dir="C:/x", transparent=False, **changes)
        with pytest.raises(server.GenerationFailed):
            engine.generate(req, 7, [_FakeReference()] * 3)
    assert [{k: g[k] for k in ("cfg", "steps", "negative", "references", "sizes")} for g in lib.given] == [
        {"cfg": 1.0, "steps": 25, "negative": b"", "references": 3, "sizes": [(864, 864)] * 3},
        {"cfg": 6.0, "steps": 20, "negative": b"blurry", "references": 3, "sizes": [(640, 640)] * 3}]
    assert {g["ref_image_args"] for g in lib.given} == {b"resize_before_vae=0"}, "the engine takes them as sized here"
    assert [g["sigmas"] for g in lib.given] == [None, None]                    # the full model: the sampler's own


@needs_pil
def test_a_request_may_size_qwen_images_copy_of_each_example_picture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen-Image's own copy of each example picture is made here (both sides multiples of 32, its shape kept) and
    the engine told to take the pictures as they come - it would shrink them by nearest neighbour. ref_sides names
    each copy's longer side (Omni-Texture's depth map large, its example small); without it each copy has the
    pixels of the picture asked for, or as many as the prefix cache's budget holds. FLUX.3's are left to the
    engine, by the launcher's own cap."""
    import PIL
    from PIL import Image

    assert server.sized_reference(Image.new("RGBA", (2048, 2048)), 1152, 0).size == (1152, 1152)
    assert server.sized_reference(Image.new("RGBA", (1024, 768)), 512, 0).size == (512, 384)
    assert server.sized_reference(Image.new("RGBA", (1000, 20)), 500, 0).size == (512, 32), "to the VAE's step, never 0"
    same = Image.new("RGBA", (512, 512))
    assert server.sized_reference(same, 512, 0) is same and server.sized_reference(same, None, 512 * 512) is same
    assert server.sized_reference(Image.new("RGBA", (2048, 2048)), None, 1024 * 1024).size == (1024, 1024)
    assert server.sized_reference(Image.new("RGBA", (1024, 768)), None, 1024 * 1024).size == (1184, 896), "as many pixels, its shape"
    assert server.sized_reference(Image.new("RGBA", (1040, 1040)), None, 1040 * 1040).size == (1056, 1056), \
        "a half step rounds up, as the engine's own rule does"
    lines = Image.new("L", (64, 64))
    lines.putdata([255 if x % 2 else 0 for _y in range(64) for x in range(64)])
    assert set(server.sized_reference(lines, None, 32 * 32).tobytes()) <= set(range(96, 160)), \
        "one-pixel lines come out grey, not every other line picked"
    lib = _RecordingLib()
    engine = _fake_engine(monkeypatch, lib)
    monkeypatch.setitem(sys.modules, "PIL", PIL)
    pictures = [Image.new("RGBA", (2048, 2048)), Image.new("RGBA", (1024, 768))]
    for model, sides in (("turbo", [1152, 512]), ("turbo", None), ("flux3", [1152, 512])):
        req = server.GenerateRequest(prompt="a chest", model=model, out_dir="C:/x", transparent=False, width=2048,
                                     height=2048, ref_sides=sides)
        with pytest.raises(server.GenerationFailed):
            engine.generate(req, 7, pictures)
    asked, own, flux3 = lib.given
    assert (asked["ref_image_args"], asked["sizes"]) == (b"resize_before_vae=0", [(1152, 1152), (512, 384)])
    assert (own["ref_image_args"], own["sizes"]) == (b"resize_before_vae=0", [(832, 832), (992, 736)]), "the budget's"
    assert flux3["ref_image_args"] != b"resize_before_vae=0" and flux3["sizes"] == [(2048, 2048), (1024, 768)]


def test_a_picture_drawn_from_pictures_never_starts_from_the_noise_they_came_from(monkeypatch: pytest.MonkeyPatch) -> None:
    """noise_seed: words alone start from the request's seed; with example pictures (or a start picture) the engine's
    seed is the request's moved on by a number taken from the pictures - never the seed itself (the noise the
    pictures may have come from: a change with the seed that made its picture came out speckled all over), the same
    for the same pictures, another for other pictures, and seed + i moves with it, as a batch's pictures do."""
    one, other = _FakeReference(), _FakeReference()
    other.width = 16
    assert server.noise_seed(11, []) == 11
    moved = server.noise_seed(11, [one])
    assert moved != 11 and 0 <= moved <= server.MAX_SEED and moved == server.noise_seed(11, [_FakeReference()])
    assert server.noise_seed(11, [other]) != moved and server.noise_seed(11, [one, other]) != moved
    assert [server.noise_seed(11 + i, [one]) for i in range(3)] == [moved, moved + 1, moved + 2]
    assert all(server.noise_seed(seed, [one]) != seed for seed in (0, 1, server.MAX_SEED))
    # ... by their pixels, not their size, and by all 32 bits: 600 pictures of one size, 600 seeds
    pictures = [types.SimpleNamespace(tobytes=lambda data=bytes([i % 256, i // 256]): data) for i in range(600)]
    assert len({server.noise_seed(11, [picture]) for picture in pictures}) >= 599
    # ... and never by nothing: a digest of zero still moves the seed on
    monkeypatch.setattr(server.hashlib, "blake2b", lambda digest_size: types.SimpleNamespace(
        update=lambda data: None, digest=lambda: bytes(digest_size)))
    assert server.noise_seed(11, [one]) == 12 and server.noise_seed(server.MAX_SEED, [one]) == 0
    monkeypatch.undo()
    lib = _RecordingLib()
    engine = _fake_engine(monkeypatch, lib)
    req = server.GenerateRequest(prompt="a chest", out_dir="C:/x", transparent=False)
    for start in ({"references": []}, {"references": [one]}, {"references": [], "init": one}):
        with pytest.raises(server.GenerationFailed):
            engine.generate(req, 11, **start)
    assert [g["seed"] for g in lib.given] == [11, moved, moved]


def test_a_decode_in_one_piece_leaves_room_for_what_the_sampling_held(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request's picture is decoded in one piece only where the budget holds the decode with the direct
    convolutions' own scratch, which grows with the picture and of which the engine's check counts 512 MB: the engine
    lets a tighter decode through and the card cannot give it. An 11 GB card's 1536x1536 picture goes in tiles (a
    12 GB card's too, by a hair); its 1408x1408 one, and the 1024 and 1152 sheets of an 8 GB card with a busy
    desktop, stay whole. The tiles are the largest whose own decode fits in one piece: 1024 pixels wherever a
    1024x1024 picture is decoded whole, 512 on a card too small for that."""
    lib = _RecordingLib()
    engine = _fake_engine(monkeypatch, lib, capability=(6, 1))
    for budget, side in ((9_999, 1536), (9_999, 1408), (5_875, 1024), (5_875, 1152), (11_100, 1536), (10_224, 1536),
                         (11_100, 2048), (5_875, 2048), (4_096, 1024)):
        engine.placement = {**engine.placement, "budget_mb": budget, "vae_direct": True}
        req = server.GenerateRequest(prompt="a chest", out_dir="C:/x", transparent=False, width=side, height=side)
        with pytest.raises(server.GenerationFailed):
            engine.generate(req, 7, [])
    assert [g["tiles"] for g in lib.given] == [1024, False, False, False, False, 1024, 1024, 1024, 512]
    # the usual convolutions have no such scratch: their rule is the budget's alone
    assert not server.needs_vae_tiling(1280, 1280, 14_336) and server.needs_vae_tiling(1280, 1280, 14_000)
    assert server.vae_tile(16 * GiB) == 1024 and server.vae_tile(8 * GiB) == 512, "the usual convolutions take twice the memory"


def test_a_tileable_picture_asks_the_engine_for_both_circular_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    """tileable: sd.cpp's circular_x and circular_y for that request - Qwen-Image's two models only, off unless
    asked for, and never while a mask holds the drawing to a picture (a masked change)."""
    lib = _RecordingLib()
    engine = _fake_engine(monkeypatch, lib)
    for model, tileable in (("turbo", True), ("full", True), ("flux3", True), ("turbo", False)):
        req = server.GenerateRequest(prompt="bricks", model=model, out_dir="C:/x", transparent=False, tileable=tileable)
        with pytest.raises(server.GenerationFailed):
            engine.generate(req, 7, [])
    assert [g["circular"] for g in lib.given] == [(True, True), (True, True), (False, False), (False, False)]
    req = server.GenerateRequest(prompt="bricks", out_dir="C:/x", transparent=False, tileable=True)
    for start in ({"init": _FakeReference()}, {"init": _FakeReference(), "mask": _FakeReference()}):
        with pytest.raises(server.GenerationFailed):
            engine.generate(req, 7, [], **start)
    assert [g["circular"] for g in lib.given[-2:]] == [(True, True), (False, False)]
    assert server.GenerateRequest(prompt="bricks", out_dir="C:/x").tileable is False


def test_a_request_says_whether_the_engine_keeps_its_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """keep_prefix (default True: a re-roll reuses the prefix) reaches the engine as SD_QWEN_PREFIX_DROP for each
    request: "1" frees the run's prefix at its end (Omni-Texture's views, whose prefix never comes back)."""
    lib = _RecordingLib()
    engine = _fake_engine(monkeypatch, lib)
    assert server.GenerateRequest(prompt="a chest", out_dir="C:/x").keep_prefix is True
    for keep in (False, True, False):
        req = server.GenerateRequest(prompt="a chest", out_dir="C:/x", transparent=False, keep_prefix=keep)
        with pytest.raises(server.GenerationFailed):
            engine.generate(req, 7, [_FakeReference()])
    assert [g["drop"] for g in lib.given] == ["1", "0", "1"]


def test_the_first_step_runs_the_prefix_on_its_own_where_measured() -> None:
    """set_prefix_pass: SD_QWEN_PREFIX_PASS=1 on sm_80+ with an engine that has the pass's gates (the pinned build
    when install/sdcpp.py ENGINE_PREFIX_PASS says so, a build under test), "0" on RTX 20xx / GTX 10xx and with an
    engine without them; a variable already set is never changed."""
    assert isinstance(server.engine_pin.ENGINE_PREFIX_PASS, bool)
    for capability, knows, value in (((8, 6), True, "1"), ((12, 0), True, "1"), ((8, 6), False, "0"),
                                     ((7, 5), True, "0"), ((6, 1), True, "0")):
        env: dict[str, str] = {}
        server.set_prefix_pass(capability, knows, env)
        assert env == {server.PASS_ENV: value}, (capability, knows)
    env = {server.PASS_ENV: "0"}
    assert "already set" in server.set_prefix_pass((8, 6), True, env) and env == {server.PASS_ENV: "0"}


def test_the_turbo_draws_its_own_fixed_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Turbo request (the default model) is drawn with its 8 fixed sigmas, euler, no guidance and without the
    First-Block Cache, whatever its steps, cfg_scale, sampler and first_block_cache say; the full model as asked, its
    cache by the tick or the card."""
    assert server.TURBO_SIGMAS == (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0)
    assert server.TURBO_STEPS == 8 and server.DEFAULT_MODEL == "turbo"
    assert server.MODELS == ("turbo", "full", "flux3") and server.QWEN_MODELS == ("turbo", "full")
    assert server.GenerateRequest(prompt="a chest", out_dir="C:/x").model == "turbo"
    with pytest.raises(ValueError):
        server.GenerateRequest(prompt="a chest", out_dir="C:/x", model="fast")
    turbo = {"steps": 8, "sigmas": server.TURBO_SIGMAS, "cfg_scale": 1.0, "sampler": "euler",
             "first_block_cache": False}
    for asked in ((25, 1.0, "euler", None), (30, 6.0, "dpm++2m", True), (8, 1.0, "euler", False)):
        for card in ((6, 1), (8, 6)):
            assert server.sampling("turbo", *asked, card) == turbo
    assert server.sampling("full", 25, 1.0, "euler", None, (6, 1)) == {
        "steps": 25, "sigmas": None, "cfg_scale": 1.0, "sampler": "euler", "first_block_cache": True}
    assert server.sampling("full", 40, 6.0, "euler", None, (8, 6))["first_block_cache"] is False   # by the card
    assert server.sampling("full", 20, 1.0, "euler", True, (8, 6))["first_block_cache"] is True     # the tick

    lib = _RecordingLib()
    engine = _fake_engine(monkeypatch, lib, capability=(6, 1))          # a GTX 10xx: the full model's cache on
    sigmas = [round(v, 6) for v in server.TURBO_SIGMAS]
    for req in (server.GenerateRequest(prompt="a chest", out_dir="C:/x"),
                server.GenerateRequest(prompt="a chest", out_dir="C:/x", model="turbo", steps=30, cfg_scale=6.0,
                                       sampler="dpm++2m", negative_prompt="blurry", first_block_cache=True),
                server.GenerateRequest(prompt="a chest", out_dir="C:/x", model="full", steps=30)):
        with pytest.raises(server.GenerationFailed):
            engine.generate(req, 7, [])
    assert [{k: g[k] for k in ("cfg", "steps", "method", "negative", "sigmas", "cache")} for g in lib.given] == [
        {"cfg": 1.0, "steps": 8, "method": sdcpp.EULER_SAMPLE_METHOD, "negative": b"", "sigmas": sigmas, "cache": "0"},
        {"cfg": 1.0, "steps": 8, "method": sdcpp.EULER_SAMPLE_METHOD, "negative": b"", "sigmas": sigmas, "cache": "0"},
        {"cfg": 1.0, "steps": 30, "method": sdcpp.EULER_SAMPLE_METHOD, "negative": b"", "sigmas": None,
         "cache": server.FIRST_BLOCK_CACHE_SKIP}]


def test_the_model_files_and_what_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files, missing = server.model_files(tmp_path, "Q4_K_M")
    # without any denoiser the Turbo's is the one missing (the installer's); the full one is optional
    assert missing == ["qwen-image-2.1-turbo-Q4_K_M.gguf", server.VAE_FILE, server.LLM_FILE] and files["vision"] is None
    assert server.start_model(files) is None
    for name in missing + [server.VISION_FILE]:
        (tmp_path / name).write_bytes(b"")
    files, missing = server.model_files(tmp_path, "Q4_K_M")
    assert missing == [] and files["vision"] == tmp_path / server.VISION_FILE
    assert files["turbo"] == tmp_path / "qwen-image-2.1-turbo-Q4_K_M.gguf" and server.start_model(files) == "turbo"
    assert files["full"] == tmp_path / "qwen-image-2.1-Q4_K_M.gguf" and not files["full"].is_file()   # its path anyway
    assert server.model_files(tmp_path, "Q8_0")[1] == ["qwen-image-2.1-turbo-Q8_0.gguf"]
    (tmp_path / server.VISION_FILE).unlink()                    # without it: no example pictures, still starts
    assert server.model_files(tmp_path, "Q4_K_M") == ({**files, "vision": None}, [])
    # an install from before the Turbo, not updated since: only the full model - it starts with that one
    (tmp_path / "qwen-image-2.1-turbo-Q4_K_M.gguf").unlink()
    (tmp_path / "qwen-image-2.1-Q4_K_M.gguf").write_bytes(b"")
    files, missing = server.model_files(tmp_path, "Q4_K_M")
    assert missing == [] and server.start_model(files) == "full"
    (tmp_path / "qwen-image-2.1-turbo-Q4_K_M.gguf").write_bytes(b"")
    files = server.model_files(tmp_path, "Q4_K_M")[0]
    assert server.start_model(files) == "turbo"                                           # both: the Turbo first
    # --first-model: the one the artist picked, when all its files are there
    assert server.start_model(files, "full") == "full" and server.start_model(files, "turbo") == "turbo"
    assert server.start_model(files, "flux3") == "turbo" and server.start_model(files, "sdxl") == "turbo"
    flux3 = tmp_path.parent / server.FLUX3_FOLDER
    flux3.mkdir(exist_ok=True)
    for key in ("flux3", "flux3_vae"):
        (flux3 / files[key].name).write_bytes(b"")
    assert server.start_model(server.model_files(tmp_path, "Q4_K_M")[0], "flux3") == "turbo", "two of its three files"
    (flux3 / files["flux3_llm"].name).write_bytes(b"")
    assert server.start_model(server.model_files(tmp_path, "Q4_K_M")[0], "flux3") == "flux3"
    assert server.parse_args(["--sdcpp", "x", "--models", "y", "--port", "1"]).first_model is None
    assert server.parse_args(["--sdcpp", "x", "--models", "y", "--port", "1", "--first-model", "full"]).first_model == "full"
    assert set(server.QUALITIES) == {"Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"}
    # one source of truth: the names are the installer's manifest's (install/models.py), for every size it fetches
    from spellcast import config
    from spellcast.install import models
    assert server.QUALITIES == config.QUALITIES == tuple(models.QWEN_DENOISERS) == tuple(models.QWEN_TURBO_DENOISERS)
    for quality in server.QUALITIES:
        named = models.qwen_model_files(quality, tmp_path)
        wanted = server.model_files(tmp_path, quality)[0]
        assert [wanted["turbo"], wanted["full"], wanted["vae"], wanted["llm"]] == [
            named["turbo"], named["full"], named["vae"], named["text_encoder"]]
        assert {named[k].name for k in ("turbo", "vae", "text_encoder", "vision")} | set(models.ALBEDO_PINS) == set(
            models.qwen_manifest(quality, tmp_path))           # and the albedo add-on (--albedo, not read at start)
        assert {named["full"].name} == set(models.qwen_full_manifest(quality, tmp_path))
        assert server.diffusion_file(quality) == named["turbo"].name
        assert server.diffusion_file(quality, "full") == named["full"].name
    # a bigger Turbo the artist put there (whole, at its pinned size) stands in for the configured size
    # (install/models.py size_on_disk): the launcher loads that file as the Turbo
    monkeypatch.setitem(models.QWEN_TURBO_DENOISERS, "Q6_K", (3, "0" * 64))
    (tmp_path / "qwen-image-2.1-turbo-Q4_K_M.gguf").unlink()
    (tmp_path / "qwen-image-2.1-turbo-Q6_K.gguf").write_bytes(b"big")
    files, missing = server.model_files(tmp_path, "Q4_K_M")
    assert missing == [] and files["turbo"] == tmp_path / "qwen-image-2.1-turbo-Q6_K.gguf"
    assert server.start_model(files) == "turbo" and files["full"] == tmp_path / "qwen-image-2.1-Q4_K_M.gguf"


def test_flux3_example_pictures_share_a_few_pictures_worth_of_pixels() -> None:
    """FLUX.3 has no prefix cache: every example picture's tokens go through every block at every step, so beyond
    FLUX3_REFERENCES_FULL_SIZE of them they share that many pictures' pixels (the engine's vae_input_max_pixels)."""
    mp = 1024 * 1024
    assert server.FLUX3_REFERENCES_FULL_SIZE == 2
    assert server.flux3_reference_pixels(0, mp) is None and server.flux3_reference_pixels(2, mp) is None
    assert server.flux3_reference_pixels(4, mp) == mp // 2 and server.flux3_reference_pixels(10, mp) == 2 * mp // 10
    assert server.flux3_reference_pixels(10, 256 * 256) == server.MIN_REFERENCE_TOKENS * 256


def test_an_example_picture_for_flux3_is_laid_on_white() -> None:
    """FLUX.3 takes RGB: the engine would drop the alpha channel of a cut-out and show the model whatever colour its
    see-through pixels hold; opaque_reference lays it on white first. An opaque picture only loses its alpha."""
    Image = pytest.importorskip("PIL.Image")
    cut = Image.new("RGBA", (4, 2), (20, 0, 200, 0))            # see-through purple ...
    cut.putpixel((0, 0), (10, 20, 30, 255))                     # ... with one opaque pixel
    cut.putpixel((1, 0), (0, 0, 0, 128))                        # and a half-transparent black one
    flat = server.opaque_reference(cut)
    assert flat.mode == "RGB" and flat.size == (4, 2)
    assert flat.getpixel((0, 0)) == (10, 20, 30) and flat.getpixel((3, 1)) == (255, 255, 255)
    assert flat.getpixel((1, 0)) == (127, 127, 127)
    solid = Image.new("RGBA", (2, 2), (1, 2, 3, 255))
    assert server.opaque_reference(solid).mode == "RGB" and server.opaque_reference(solid).getpixel((1, 1)) == (1, 2, 3)


def test_the_recipe_is_qwens_own_without_guidance() -> None:
    """The request defaults: no guidance (CFG exactly 1.0, no negative prompt) and 25 steps (ComfyUI's official
    template; Qwen's own reference is 40). cfg_scale and negative_prompt are still taken, for the bench."""
    req = server.GenerateRequest(prompt="a chest", out_dir="C:/x")
    assert (req.steps, req.cfg_scale, req.sampler, req.width, req.height) == (25, 1.0, "euler", 1024, 1024)
    assert (server.STEPS, server.CFG_SCALE) == (25, 1.0) and req.negative_prompt == ""
    assert req.transparent is True and req.batch_count == 1 and req.ref_images == []
    assert server.condition_count(req.cfg_scale) == 1
    former = server.GenerateRequest(prompt="a chest", out_dir="C:/x", steps=20, cfg_scale=6.0, negative_prompt="blurry")
    assert (former.steps, former.cfg_scale, former.negative_prompt) == (20, 6.0, "blurry")
    with pytest.raises(ValueError):
        server.GenerateRequest(prompt="a chest", out_dir="C:/x", cfg_scale=31)


# ---- the diffusion weights in page-locked RAM (--pin, SD_PIN_PARAMS) ---------------------------------------------

@pytest.mark.parametrize("total_mb, pinned", [
    (16_310, False),                    # 16 GB installed (Windows counts what the hardware leaves)
    (20_400, False),                    # 20 GB
    (server.PIN_MIN_RAM_MB - 1, False),
    (server.PIN_MIN_RAM_MB, True),
    (23_717, True),                     # 24 GB installed
    (31_909, True),                     # 32 GB installed (the RTX 3060 PC)
    (None, False),                      # unknown (not Windows, or the call failed)
])
def test_pin_auto_is_for_pcs_with_24_gb_of_ram(total_mb: int | None, pinned: bool) -> None:
    value, said = server.pin_choice("auto", total_mb, None)
    # from 32 GB the text encoder's weights too (test_pin_auto_adds_the_text_encoder_on_pcs_with_32_gb_of_ram)
    expected = (server.PIN_TEXT_PARAMS if total_mb >= server.PIN_TEXT_MIN_RAM_MB else "diffusion") if pinned else None
    assert value == expected
    if pinned:
        assert said.startswith(f"SD_PIN_PARAMS={expected} (--pin auto: this PC has ") and f"{total_mb:,} MB" in said
        assert "Spellcast3D's engine build keeps the diffusion weights" in said and "in page-locked RAM" in said
        assert "official stable-diffusion.cpp release ignores it" in said
    else:
        assert said.startswith("The diffusion weights are not kept in page-locked RAM: ")
        assert ("unknown" if total_mb is None else f"{total_mb:,} MB") in said and "--pin on" in said
    assert "\n" not in said                                     # one line of the log


def test_pin_auto_adds_the_text_encoder_on_pcs_with_32_gb_of_ram() -> None:
    """From 30 GiB (32 GB installed: Windows counts 31,909 MB) --pin auto keeps the text encoder's weights page-locked
    too; below it (a 24 GB PC) the diffusion weights alone, as before; --pin on stays the diffusion weights."""
    assert server.pin_choice("auto", server.PIN_TEXT_MIN_RAM_MB - 1, None)[0] == "diffusion"
    assert server.pin_choice("auto", 24_000, None)[0] == "diffusion"
    for total_mb in (server.PIN_TEXT_MIN_RAM_MB, 31_909, 65_300):          # 32 GB, 64 GB installed
        value, said = server.pin_choice("auto", total_mb, None)
        assert value == "diffusion,text_encoder" and f"{total_mb:,} MB" in said and "32 GB or more" in said
        assert said.startswith("SD_PIN_PARAMS=diffusion,text_encoder (--pin auto: ") and "\n" not in said
        assert server.pin_choice("on", total_mb, None)[0] == "diffusion"


def test_pin_on_and_off_do_not_ask_the_pc() -> None:
    for total_mb in (8_000, 31_909, None):
        value, said = server.pin_choice("on", total_mb, None)
        assert value == "diffusion" and said.startswith("SD_PIN_PARAMS=diffusion (--pin on): ")
        assert server.pin_choice("off", total_mb, None) == (
            None, "The diffusion weights are not kept in page-locked RAM (--pin off).")
    # weights kept on the card (--placement gpu) are never copied: auto leaves it, "on" is still obeyed
    value, said = server.pin_choice("auto", 31_909, None, in_ram=False)
    assert value is None and "--placement gpu keeps them on the card" in said
    assert server.pin_choice("on", 31_909, None, in_ram=False)[0] == "diffusion"


@pytest.mark.parametrize("asked", server.PINS)
def test_a_variable_the_user_set_is_left_alone(asked: str) -> None:
    for theirs in ("all", "diffusion,vae", ""):                 # "" (not on Windows): theirs too, pinning off
        value, said = server.pin_choice(asked, 31_909, theirs)
        assert value is None and said.startswith(f"SD_PIN_PARAMS is already set ({theirs!r}): left as it is")
        assert f"--pin {asked} never changes a set one" in said and "\n" not in said
        environ = {"SD_PIN_PARAMS": theirs, "OTHER": "1"}
        assert server.set_pinning(asked, environ=environ) == said
        assert environ == {"SD_PIN_PARAMS": theirs, "OTHER": "1"}


def test_the_pin_goes_into_the_environment_given(monkeypatch: pytest.MonkeyPatch) -> None:
    assert (server.PIN_ENV, server.PIN_PARAMS, server.PINS) == ("SD_PIN_PARAMS", "diffusion", ("auto", "on", "off"))
    assert server.parse_args(["--sdcpp", "s", "--models", "m", "--port", "1"]).pin == "auto"       # the adapter's
    assert server.parse_args(["--sdcpp", "s", "--models", "m", "--port", "1", "--pin", "off"]).pin == "off"
    for total_mb, asked, wanted in ((31_909, "auto", "diffusion,text_encoder"), (24_000, "auto", "diffusion"),
                                    (16_310, "auto", None), (16_310, "on", "diffusion"),
                                    (31_909, "off", None), (31_909, "on", "diffusion")):
        monkeypatch.setattr(server, "total_ram_mb", lambda total_mb=total_mb: total_mb)
        environ: dict[str, str] = {"PATH": "x"}
        said = server.set_pinning(asked, environ=environ)
        assert environ.get("SD_PIN_PARAMS") == wanted and len(environ) == (2 if wanted else 1)
        assert said == server.pin_choice(asked, total_mb, None)[1]
    monkeypatch.setattr(server, "total_ram_mb", lambda: 31_909)
    on_card: dict[str, str] = {}
    assert "--placement gpu" in server.set_pinning("auto", in_ram=False, environ=on_card) and on_card == {}


@pytest.mark.skipif(os.name != "nt", reason="GlobalMemoryStatusEx")
def test_the_pcs_ram_is_read_from_windows() -> None:
    total, free = server.total_ram_mb(), server.free_ram_mb()
    assert isinstance(total, int) and isinstance(free, int) and 0 < free <= total and 1024 <= total < 64 * 1024 * 1024


@pytest.mark.parametrize("total_mb, asked, at_load", [(31_909, "auto", "diffusion,text_encoder"),
                                                      (24_000, "auto", "diffusion"), (16_310, "auto", None),
                                                      (16_310, "on", "diffusion"), (31_909, "off", None)])
def test_the_pin_is_in_the_environment_before_the_dll_is_loaded(total_mb: int, asked: str, at_load: str | None,
                                                               tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                               capsys: pytest.CaptureFixture[str]) -> None:
    """The engine reads SD_PIN_PARAMS through its C runtime, once: load_runtime (main's loading of the DLL) puts it
    into this process's environment first. The DLL is a fake that notes what the environment held when it was loaded."""
    monkeypatch.setenv("SD_PIN_PARAMS", "theirs")       # so that the variable is put back as it was after the test...
    monkeypatch.delenv("SD_PIN_PARAMS")                 # ...and is not set now
    monkeypatch.setattr(server, "total_ram_mb", lambda: total_mb)
    events: list[tuple[str, Any]] = []
    lib = object()

    def load(folder: Path) -> Any:
        events.append(("load", folder, os.environ.get("SD_PIN_PARAMS")))
        return lib

    monkeypatch.setattr(sdcpp, "load", load)
    monkeypatch.setattr(sdcpp, "commit", lambda dll: sdcpp.PINNED_COMMIT[:7])          # the pinned build
    monkeypatch.setattr(sdcpp, "check_layout", lambda dll: events.append(("checked", dll)))
    assert server.load_runtime(tmp_path, asked) is lib
    assert events == [("load", tmp_path, at_load), ("checked", lib)]
    assert os.environ.get("SD_PIN_PARAMS") == at_load
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "[spellcast] " + server.pin_choice(asked, total_mb, None)[1]       # said before the DLL's line
    assert out[1] == f"[spellcast] stable-diffusion.cpp commit {sdcpp.PINNED_COMMIT[:7]} (pinned {sdcpp.PINNED_TAG})"
    assert len(out) == 2


def test_the_users_own_pin_reaches_the_dll_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("SD_PIN_PARAMS", "all")
    monkeypatch.setattr(server, "total_ram_mb", lambda: 16_310)
    seen: list[str | None] = []
    monkeypatch.setattr(sdcpp, "load", lambda folder: seen.append(os.environ.get("SD_PIN_PARAMS")) or object())
    monkeypatch.setattr(sdcpp, "commit", lambda dll: "abcdef0")
    monkeypatch.setattr(sdcpp, "check_layout", lambda dll: None)
    for asked in server.PINS:
        server.load_runtime(tmp_path, asked)
    assert seen == ["all"] * 3 and os.environ["SD_PIN_PARAMS"] == "all"
    out = capsys.readouterr().out
    assert out.count("SD_PIN_PARAMS is already set ('all'): left as it is") == 3
    assert "ANOTHER BUILD, its layout is checked" in out            # not the pinned release: said, then checked


@pytest.mark.parametrize("capability, at_load", [((6, 1), "q8_0"), ((7, 5), "q8_0"), ((8, 6), None), ((12, 0), None),
                                                 (None, None)])
def test_older_cards_have_loras_as_q8_0_before_the_dll_is_loaded(capability: tuple[int, int] | None,
                                                                 at_load: str | None, tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """SD_LORA_WEIGHT_TYPE=q8_0 reaches the engine on cards without BF16 tensor cores (GTX 10xx, RTX 20xx; see
    LORA_TYPE_ENV), on no other card; a value already set is the user's and stays."""
    monkeypatch.setenv(server.LORA_TYPE_ENV, "theirs")
    monkeypatch.delenv(server.LORA_TYPE_ENV)
    monkeypatch.setenv(server.PIN_ENV, "theirs")         # load_runtime sets the pin too: put back after the test
    monkeypatch.delenv(server.PIN_ENV)                   # (it leaked into later tests' launch environments)
    seen: list[str | None] = []
    monkeypatch.setattr(sdcpp, "load", lambda folder: seen.append(os.environ.get(server.LORA_TYPE_ENV)) or object())
    monkeypatch.setattr(sdcpp, "commit", lambda dll: sdcpp.PINNED_COMMIT[:7])
    monkeypatch.setattr(sdcpp, "check_layout", lambda dll: None)
    server.load_runtime(tmp_path, capability=capability)
    monkeypatch.setenv(server.LORA_TYPE_ENV, "bf16")
    server.load_runtime(tmp_path, capability=capability)
    assert seen == [at_load, "bf16"]


def test_the_compute_device_is_the_first_card_or_the_freest_for_gpu() -> None:
    cards = [{"index": 0, "free_mb": 7000, "capability": (6, 1)}, {"index": 1, "free_mb": 11000, "capability": (8, 6)}]
    assert server.compute_device(cards, "offload")["index"] == 0 and server.compute_device(cards, "auto")["index"] == 0
    assert server.compute_device(cards, "gpu")["index"] == 1


def test_main_loads_the_dll_through_load_runtime() -> None:
    """main() itself needs the GPU; what is checked here is that it has no other way to the DLL than load_runtime
    (which sets the pin first), with --pin, the placement and the card it computes on (queried before the DLL)."""
    import inspect

    source = inspect.getsource(server.main)
    assert ('load_runtime(sdcpp_dir, args.pin, in_ram=args.placement != "gpu", matmul=args.matmul,\n'
            '                           capability=device["capability"] if device else None)') in source
    assert (source.index("sdcpp.prepare_dll_search(sdcpp_dir)") < source.index("cuda.devices()")
            < source.index("device = compute_device(devices, args.placement)") < source.index("load_runtime("))
    assert "sdcpp.load(" not in source and "sdcpp.load(" in inspect.getsource(server.load_runtime)


# ---- the checks of a request -------------------------------------------------------------------------------------

@pytest.fixture()
def root(tmp_path: Path) -> Path:
    ref = tmp_path / "p1" / "reference" / "reference_001.png"
    ref.parent.mkdir(parents=True)
    ref.write_bytes(b"png")
    return tmp_path.resolve()


def _request(root: Path, **changes: Any) -> server.GenerateRequest:
    body = {"prompt": "a wooden treasure chest", "out_dir": str(root / "p1" / "images"), **changes}
    return server.GenerateRequest(**body)


def test_requests_are_checked_in_artist_words(root: Path) -> None:
    ref = str(root / "p1" / "reference" / "reference_001.png")
    assert server.validate(_request(root), root, True) is None
    assert server.validate(_request(root, ref_images=[ref] * 10), root, True) is None
    # a negative prompt without guidance is no mistake, only unused (sd.cpp never encodes it at CFG 1.0); with it too
    assert server.validate(_request(root, negative_prompt="blurry, text"), root, True) is None
    assert server.validate(_request(root, negative_prompt="blurry, text", cfg_scale=6.0), root, True) is None
    assert "Describe" in server.validate(_request(root, prompt="  "), root, True)
    assert "multiple of 32" in server.validate(_request(root, width=1000), root, True)
    assert "multiple of 32" in server.validate(_request(root, height=4096), root, True)
    assert "own projects folder" in server.validate(_request(root, out_dir=str(root.parent / "x")), root, True)
    assert "absolute" in server.validate(_request(root, out_dir="images"), root, True)
    assert "up to 10" in server.validate(_request(root, ref_images=[ref] * 11), root, True)
    assert "vision file" in server.validate(_request(root, ref_images=[ref]), root, False)
    assert "not found" in server.validate(_request(root, ref_images=[str(root / "gone.png")]), root, True)
    outside = root.parent / "outside.png"
    assert "own projects folder" in server.validate(_request(root, ref_images=[str(outside)]), root, True)
    assert "absolute" in server.validate(_request(root, ref_images=["ref.png"]), root, True)
    # img2img: the start picture and its mask are the projects folder's own pictures, a mask only with a picture
    assert server.validate(_request(root, init_image=ref, strength=0.6, mask_image=ref), root, True) is None
    assert "needs the picture" in server.validate(_request(root, mask_image=ref), root, True)
    for given in (str(outside), "viewport.png", str(root / "gone.png")):
        assert "start picture" in server.validate(_request(root, init_image=given), root, True)
    with pytest.raises(ValueError):
        _request(root, strength=0.0)
    with pytest.raises(ValueError):
        _request(root, steps=0)
    with pytest.raises(ValueError):
        _request(root, batch_count=server.MAX_BATCH + 1)

    class Lib:     # str_to_sample_method: SAMPLE_METHOD_COUNT for a name it does not know
        @staticmethod
        def str_to_sample_method(name: bytes) -> int:
            return 0 if name == b"euler" else sdcpp.SAMPLE_METHOD_COUNT

    assert server.validate(_request(root), root, True, Lib()) is None
    assert "sampler" in server.validate(_request(root, sampler="dpm++9"), root, True, Lib())


def test_failures_become_kinds_the_adapter_reads() -> None:
    def answer(exc: BaseException) -> tuple[int, dict[str, Any]]:
        r = server.failure_response(exc)
        return r.status_code, json.loads(r.body)

    assert answer(server.Cancelled()) == (409, {"detail": "Stopped: the pictures were cancelled.",
                                                "kind": "cancelled"})
    assert answer(server.InputProblem("Bad picture."))[1] == {"detail": "Bad picture.", "kind": "input"}
    assert answer(server.OutOfMemory("cudaMalloc failed"))[1]["kind"] == "oom"
    assert answer(MemoryError())[1]["kind"] == "oom"
    other = answer(server.GenerationFailed("x" * 1000))[1]
    assert other["kind"] == "error" and len(other["detail"]) < 700


def test_memory_shortage_is_read_from_sd_cpps_lines() -> None:
    assert server.ran_out_of_memory(["[ERROR] ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2048 MiB "
                                     "on device 0: cudaMalloc failed: out of memory"])
    # its offload manager, when other programs hold the card (seen live: the text encoder could not be brought in)
    assert server.ran_out_of_memory(["model_manager.cpp:1919 - model manager cannot make enough memory available on "
                                     "CUDA0: need 789.85 MB device / 277.85 MB budget, available 585.85 MB device / "
                                     "5806.37 MB budget", "ggml_runner.cpp:899  - qwen3vl segment failed"])
    # the prefix cache's own shortage: sd.cpp retries without it, so a later failure is something else
    assert not server.ran_out_of_memory(["[WARN ] Qwen Image 2.1: insufficient memory for prefix caching; retrying "
                                         "without it for this sampling run", "[ERROR] sampling failed"])


# ---- progress ----------------------------------------------------------------------------------------------------

def _log(progress: server.Progress, text: str, level: int = sdcpp.SD_LOG_INFO) -> None:
    progress.on_log(level, f"[INFO ] image.cpp:1   - {text}")


def test_progress_follows_sd_cpps_phases_forward() -> None:
    progress = server.Progress()
    assert progress.to_dict() == {"phase": "idle", "step": 0, "steps": 0, "fraction": 0.0, "message": "Idle",
                                  "eta": None}
    progress.loading()
    progress.on_step(3, 10, 0.5)
    assert progress.to_dict()["phase"] == "loading" and progress.to_dict()["fraction"] == 0.3
    progress.stop()

    progress.start(2, 20, references=1)
    seen = [progress.fraction]
    assert progress.phase == "references"
    _log(progress, "EDIT mode")
    _log(progress, "encode_first_stage completed, taking 1.50s")
    assert progress.phase == "prompt"
    progress.on_step(5, 9, 0.2)                          # the vision encoder's own steps: not the picture's
    _log(progress, "get_learned_condition completed, taking 4.20s")
    seen.append(progress.fraction)
    for image in (1, 2):
        _log(progress, f"generating image: {image}/2 - seed {40 + image}")
        # the step being drawn, from the start: never "step 0"
        assert progress.to_dict()["message"] == f"Drawing picture {image} of 2 · step 1 of 20"
        for step in range(21):
            progress.on_step(step, 20, 2.0)
            seen.append(progress.fraction)
        _log(progress, "sampling completed, taking 40.00s")
    state = progress.to_dict()
    assert state["phase"] == "drawing" and state["message"] == "Drawing picture 2 of 2 · step 20 of 20"
    _log(progress, "decoding 2 latents")
    assert progress.phase == "finishing" and progress.to_dict()["message"] == "Finishing picture 1 of 2"
    _log(progress, "latent 1 decoded, taking 3.00s")
    assert progress.to_dict()["message"] == "Finishing picture 2 of 2"
    seen.append(progress.fraction)
    _log(progress, "latent 2 decoded, taking 3.00s")
    progress.saving()
    seen.append(progress.fraction)
    assert seen == sorted(seen) and seen[-1] == 0.99
    assert progress.timings == {"references": 1.5, "prompt": 4.2, "drawing_1": 40.0, "drawing_2": 40.0,
                                "finishing_1": 3.0, "finishing_2": 3.0}
    progress.stop()
    assert progress.to_dict()["phase"] == "idle"


def test_the_eta_counts_the_steps_and_decodes_left() -> None:
    progress = server.Progress()
    progress.start(2, 20, references=0)
    _log(progress, "generating image: 1/2 - seed 1")
    assert progress.eta() is None                         # no step timed yet
    progress.on_step(0, 20, 0.0)
    for step in range(1, 11):
        progress.on_step(step, 20, 1.5)
    # 10 steps of this picture and 20 of the next at 1.5 s, and two decodes at the first guess
    assert progress.eta() == pytest.approx(30 * 1.5 + 2 * server.DECODE_GUESS, abs=0.1)
    # between two steps it counts down, but never past the end of the step under way
    progress.moved_at -= 1.0
    assert progress.eta() == pytest.approx(30 * 1.5 + 2 * server.DECODE_GUESS - 1.0, abs=0.1)
    progress.moved_at -= 10.0
    assert progress.eta() == pytest.approx(30 * 1.5 + 2 * server.DECODE_GUESS - 0.95 * 1.5, abs=0.1)
    assert progress.to_dict()["message"] == "Drawing picture 1 of 2 · step 11 of 20"     # 10 done: the 11th is drawn
    progress.start(1, 20, references=0)
    _log(progress, "generating image: 1/1 - seed 1")
    assert progress.to_dict()["message"] == "Drawing · step 1 of 20"
    progress.on_step(20, 20, 1.5)
    assert progress.to_dict()["message"] == "Drawing · step 20 of 20"
    _log(progress, "decoding 1 latents")
    assert progress.to_dict()["message"] == "Finishing"


def test_every_phase_of_a_request_says_the_size_it_renders_at() -> None:
    """The artist is reminded of the resolution for as long as the pictures are made; loading a model has none."""
    progress = server.Progress()
    progress.start(2, 8, references=1, size=(1248, 1824))
    assert progress.to_dict()["message"] == "Looking at the example pictures · 1248 × 1824 px"
    _log(progress, "encode_first_stage completed, taking 0.4s")
    assert progress.to_dict()["message"] == "Reading the description · 1248 × 1824 px"
    _log(progress, "generating image: 2/2 - seed 1")
    assert progress.to_dict()["message"] == "Drawing picture 2 of 2 · 1248 × 1824 px · step 1 of 8"
    _log(progress, "decoding 2 latents")
    assert progress.to_dict()["message"] == "Finishing picture 1 of 2 · 1248 × 1824 px"
    progress.saving()
    assert progress.to_dict()["message"] == "Saving · 1248 × 1824 px"
    progress.start(1, 8, references=0, size=(2048, 2048))
    _log(progress, "generating image: 1/1 - seed 1")
    assert progress.to_dict()["message"] == "Drawing · 2048 × 2048 px · step 1 of 8"
    progress.loading("Loading the full Qwen-Image 2.1")
    assert progress.to_dict()["message"] == "Loading the full Qwen-Image 2.1"


def test_a_shortage_for_the_prefix_cache_is_noticed() -> None:
    progress = server.Progress()
    progress.start(1, 20, references=3)
    _log(progress, "Qwen Image 2.1: insufficient memory for prefix caching; retrying without it", sdcpp.SD_LOG_WARN)
    assert progress.cache_dropped and len(progress.lines) == 1


# ---- the routes --------------------------------------------------------------------------------------------------

class FakePicture:
    width = height = 1

    def __init__(self, transparent: bool) -> None:
        self.transparent = transparent

    def getbands(self) -> tuple[str, ...]:
        return ("R", "G", "B", "A")


class FakeEngine:
    ready = True
    loading = False
    has_vision = True
    lib = None
    load_seconds = 12.5

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self.cancels = 0

    def generate(self, req: server.GenerateRequest, seed: int, references: list[Any]) -> list[Any]:
        self.calls.append((req.prompt, seed, len(references)))
        if req.prompt == "impossible":
            raise server.InputProblem("Not like that.")
        return [FakePicture(req.transparent) for _ in range(req.batch_count)]

    def cancel(self) -> None:
        self.cancels += 1


@pytest.fixture()
def fake_pictures(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """load_reference / save_picture / has_transparency without Pillow."""
    saved: list[Path] = []

    def save(picture: FakePicture, out_dir: Path, seed: int, transparent: bool = True) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"qwen-{seed}.png"
        path.write_bytes(b"png")
        saved.append(path)
        return path

    monkeypatch.setattr(server, "load_reference", lambda path, decoded=None: FakePicture(False))
    monkeypatch.setattr(server, "save_picture", save)
    monkeypatch.setattr(server, "has_transparency", lambda picture: picture.transparent)
    return saved


def test_routes(root: Path, fake_pictures: list[Path]) -> None:
    engine = FakeEngine()
    progress = server.Progress()
    info = {"gpu": "RTX", "quality": "Q4_K_M"}
    client = TestClient(server.build_app(engine, server.GpuWorker(), progress, root, info))
    assert client.get("/spellcast/health").json() == {"ready": True, "loading": False, "references": True,
                                                      "load_seconds": 12.5, "albedo": False, "gpu": "RTX",
                                                      "quality": "Q4_K_M"}
    assert client.get("/spellcast/progress").json()["phase"] == "idle"
    body = _request(root, seed=7, batch_count=2, ref_images=[str(root / "p1" / "reference" / "reference_001.png")])
    r = client.post("/spellcast/generate", json=body.model_dump())
    assert r.status_code == 200, r.text
    answer = r.json()
    assert answer["seeds"] == [7, 8] and answer["images"] == [str(p) for p in fake_pictures]
    assert all(Path(p).is_file() for p in answer["images"]) and answer["warning"] is None
    assert isinstance(answer["seconds"], float) and engine.calls == [("a wooden treasure chest", 7, 1)]
    assert progress.phase == "idle"
    r = client.post("/spellcast/generate", json={**body.model_dump(), "seed": -1, "batch_count": 1})
    assert r.status_code == 200 and r.json()["seeds"][0] >= 0                 # a random seed, reported back
    r = client.post("/spellcast/generate", json={**body.model_dump(), "prompt": "impossible"})
    assert r.status_code == 422 and r.json() == {"detail": "Not like that.", "kind": "input"}
    r = client.post("/spellcast/generate", json={**body.model_dump(), "width": 1000})
    assert r.status_code == 400 and r.json()["kind"] == "input"
    for sides in ([1152], [1152, 512], [128], [4096]):      # a side for each example picture, of a size a copy has
        r = client.post("/spellcast/generate", json={**body.model_dump(), "ref_sides": sides})
        assert (r.status_code, r.json().get("kind")) == ((200, None) if sides == [1152] else (400, "input")), sides
    r = client.post("/spellcast/generate", json={**body.model_dump(), "steps": 0})
    assert r.status_code == 422 and r.json()["kind"] == "input" and "steps" in r.json()["detail"]
    assert client.post("/spellcast/cancel", json={}).json() == {"cancelling": False}    # nothing running
    assert engine.cancels == 0
    assert client.post("/spellcast/cancel").status_code == 422      # needs a JSON body (no CORS-free posts)
    engine.ready, engine.loading = False, True
    r = client.post("/spellcast/generate", json=body.model_dump())
    assert r.status_code == 503 and "loading" in r.json()["detail"]
    assert client.get("/spellcast/health").json()["loading"] is True


def test_a_warm_up_is_queued_on_the_gpu_thread_once_the_model_is_loaded(root: Path, fake_pictures: list[Path]) -> None:
    """POST /spellcast/warm_up: QwenEngine.warm_omnitexture on the GPU thread, never "busy" for a request that
    comes meanwhile (it waits behind it there); nothing while the model loads, nor for an engine without it."""
    class Warming(FakeEngine):
        def __init__(self) -> None:
            super().__init__()
            self.warmed: list[bool] = []

        def warm_omnitexture(self, albedo: bool = True) -> float:
            self.warmed.append(albedo)
            return 1.5

    engine, gpu = Warming(), server.GpuWorker()
    client = TestClient(server.build_app(engine, gpu, server.Progress(), root, {}))
    assert client.post("/spellcast/warm_up").status_code == 422          # a JSON body, as for cancel
    engine.ready = False
    assert client.post("/spellcast/warm_up", json={}).json() == {"queued": False}
    engine.ready = True
    assert client.post("/spellcast/warm_up", json={"albedo": True}).json() == {"queued": True}
    r = client.post("/spellcast/generate", json=_request(root, seed=3).model_dump())
    assert r.status_code == 200 and engine.calls == [("a wooden treasure chest", 3, 0)]
    gpu.run(lambda: None)
    assert engine.warmed == [True]
    plain = TestClient(server.build_app(FakeEngine(), server.GpuWorker(), server.Progress(), root, {}))
    assert plain.post("/spellcast/warm_up", json={}).json() == {"queued": False}


@pytest.mark.parametrize("cuda", [False, True])
def test_a_warm_up_that_breaks_cuda_ends_the_process_before_any_view(root: Path, fake_pictures: list[Path],
                                                                     cuda: bool) -> None:
    """A warm-up's error is checked as a request's is: CUDA that no longer works makes Qwen-Image not ready and ends
    the process (the hub starts it afresh before the job, rather than losing its first prop); with CUDA working it
    stays, and the first view pays the first use."""
    class Failing(FakeEngine):
        def warm_omnitexture(self, albedo: bool = True) -> float:
            raise server.GenerationFailed("[ERROR] CUDA error: an illegal memory access was encountered")

    exits: list[bool] = []
    engine, gpu = Failing(), server.GpuWorker()
    client = TestClient(server.build_app(engine, gpu, server.Progress(), root, {}, cuda_works=lambda: cuda,
                                         exit_later=lambda: exits.append(True)))
    assert client.post("/spellcast/warm_up", json={}).json() == {"queued": True}
    gpu.run(lambda: None)                                   # the warm-up has run
    assert client.get("/spellcast/health").json()["ready"] is cuda
    assert exits == ([] if cuda else [True])
    r = client.post("/spellcast/generate", json=_request(root).model_dump())
    if cuda:
        assert r.status_code == 200
    else:
        assert r.status_code == 503 and r.json()["kind"] == "restart"


def test_a_cancel_that_stops_the_warm_up_is_no_failure(root: Path, fake_pictures: list[Path],
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    """A request's cancel that reaches the warm-up ahead of it (the engine's one cancel flag) stops the warm-up: no
    traceback, no "failed" in the log, and Qwen-Image stays ready."""
    class Stopped(FakeEngine):
        def warm_omnitexture(self, albedo: bool = True) -> float:
            raise server.Cancelled()

    engine, gpu = Stopped(), server.GpuWorker()
    client = TestClient(server.build_app(engine, gpu, server.Progress(), root, {}, cuda_works=lambda: False,
                                         exit_later=lambda: pytest.fail("no exit")))
    assert client.post("/spellcast/warm_up", json={}).json() == {"queued": True}
    gpu.run(lambda: None)
    out = capsys.readouterr()
    assert "Traceback" not in out.out + out.err and "failed" not in out.out + out.err
    assert "warm-up stopped" in out.out and client.get("/spellcast/health").json()["ready"] is True


def test_the_warm_up_draws_a_small_view_and_its_albedo_once_before_any_request(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """QwenEngine.warm_omnitexture: one WARM_SIDE picture as an Omni-Texture view asks for one (two example pictures,
    a start picture, a mask) with the loaded model, then an albedo (only with the add-on), once per process; never
    after a request (that paid the first use), nor with FLUX.3 loaded. Pillow is a fake here."""
    class Picture:
        def __init__(self, mode: str, size: tuple[int, int], colour: Any = 0) -> None:
            self.mode, self.size = mode, size

        def paste(self, colour: Any, box: Any) -> None:
            pass

    pil = type(sys)("PIL")
    pil.Image = types.SimpleNamespace(new=Picture)
    monkeypatch.setitem(sys.modules, "PIL", pil)
    asked: list[Any] = []

    def make(model: str = "turbo", albedo: bool = True) -> Any:
        engine = server.QwenEngine.__new__(server.QwenEngine)
        engine.ctx, engine.model = 1, model
        engine.albedo_files = {"lora": Path("l")} if albedo else None
        engine.generate = lambda req, seed, refs, init=None, mask=None: asked.append(
            ("generate", req.model, req.width, req.height, seed, [r.mode for r in refs], init.mode, mask.mode))
        engine.albedo = lambda picture: asked.append(("albedo", picture.size))
        return engine

    engine = make()
    assert isinstance(engine.warm_omnitexture(), float)
    side = server.WARM_SIDE
    assert asked == [("generate", "turbo", side, side, 0, ["RGBA", "RGBA"], "RGB", "L"), ("albedo", (side, side))]
    assert engine.warm_omnitexture() is None and len(asked) == 2 and not engine.served       # once per process
    asked.clear()
    used = make("full", albedo=False)
    used.served = True                                      # a request came first
    assert used.warm_omnitexture() is None and asked == []
    assert make("full", albedo=False).warm_omnitexture() is not None and [a[0] for a in asked] == ["generate"]
    assert make("flux3").warm_omnitexture() is None


class _ContextLib:
    """The DLL's context calls as QwenEngine's load, free and use_model make them: the denoiser each new context was
    made with, and the contexts freed. A denoiser whose name is in `fail_for` makes no context (new_sd_ctx NULL)."""

    def __init__(self) -> None:
        self.made: list[tuple[int, str]] = []
        self.freed: list[int] = []
        self.fail_for: set[str] = set()
        self.handles = 100

    def sd_set_log_callback(self, callback: Any, data: Any) -> None:
        pass

    def sd_set_progress_callback(self, callback: Any, data: Any) -> None:
        pass

    @staticmethod
    def sd_ctx_params_init(pointer: Any) -> None:
        ctypes.memset(pointer, 0, ctypes.sizeof(sdcpp.SdCtxParams))

    def new_sd_ctx(self, params: Any) -> int | None:
        name = Path(params._obj.diffusion_model_path.decode()).name
        if name in self.fail_for:
            return None
        self.handles += 1
        self.made.append((self.handles, name))
        return self.handles

    def free_sd_ctx(self, ctx: int) -> None:
        self.freed.append(ctx)

    @staticmethod
    def sd_get_model_version_name(ctx: int) -> bytes:
        return b"Qwen Image 2.1"

    def sd_cancel_generation(self, ctx: int, mode: int) -> None:
        pass

    @staticmethod
    def str_to_sample_method(name: bytes) -> int:
        return sdcpp.EULER_SAMPLE_METHOD


TURBO_FILE, FULL_FILE = "qwen-image-2.1-turbo-Q4_K_M.gguf", "qwen-image-2.1-Q4_K_M.gguf"


def _two_model_engine(folder: Path, *on_disk: str) -> tuple[Any, _ContextLib]:
    """A real QwenEngine on _ContextLib, its model files (empty) in `folder`: the shared ones and `on_disk`."""
    for name in (server.VAE_FILE, server.LLM_FILE, server.VISION_FILE, *on_disk):
        (folder / name).write_bytes(b"")
    files, missing = server.model_files(folder, "Q4_K_M")
    assert missing == []
    lib = _ContextLib()
    placement = {**server.choose_placement(11 * GiB, 10_664), "mmap": False, "capability": (8, 6)}
    engine = server.QwenEngine(lib, files, placement, server.Progress(), model=server.start_model(files))
    engine.load()
    return engine, lib


class AlbedoEngine(FakeEngine):
    """An engine with the albedo add-on: its answer is the picture in grey."""

    def __init__(self, files: dict[str, Path]) -> None:
        super().__init__()
        self.albedo_files = files
        self.albedo_sizes: list[tuple[int, int]] = []

    def albedo(self, picture: Any) -> Any:
        self.albedo_sizes.append(picture.size)
        return picture.convert("L").convert("RGB")


def test_the_albedo_add_on_needs_all_its_files(tmp_path: Path) -> None:
    assert server.albedo_files(None) is None
    for name in list(server.ALBEDO_FILES.values())[:-1]:
        (tmp_path / name).write_bytes(b"")
    assert server.albedo_files(tmp_path) is None
    (tmp_path / list(server.ALBEDO_FILES.values())[-1]).write_bytes(b"")
    assert set(server.albedo_files(tmp_path)) == {"lora", "condition", "vae"}


@needs_pil
def test_pictures_made_from_a_slid_picture_are_slid_back(root: Path) -> None:
    """slide_back: the picture the request starts from was shown slid right and down by that share of its size (the
    page's offset slider for tileable pictures); what is made from it is slid back as far, to the pixel as the page
    slides (round half up of the share in thousandths times the side)."""
    from PIL import Image

    def dot(size: tuple[int, int], at: tuple[int, int]) -> Any:
        picture = Image.new("RGB", size, "black")
        picture.putpixel(at, (255, 0, 0))
        return picture

    def where(picture: Any) -> tuple[int, int]:
        box = picture.getbbox()
        return box[0], box[1]

    # Right and down, what leaves one side coming back at the other; 1024 px slid a tenth: 102 px, as on the page.
    assert where(server.slid(dot((10, 20), (0, 0)), 0.2)) == (2, 4)
    assert where(server.slid(dot((10, 20), (9, 19)), 0.2)) == (1, 3)
    assert where(server.slid(dot((10, 20), (0, 0)), -0.2)) == (8, 16)
    assert where(server.slid(dot((1024, 768), (0, 0)), 0.1)) == (102, 77)
    for size in ((1024, 768), (1328, 1328), (2048, 1152)):
        for tenth in range(-5, 6):
            there = server.slid(dot(size, (5, 7)), tenth / 10)
            assert where(server.slid(there, -tenth / 10)) == (5, 7), (size, tenth)
    assert server.slid(Image.new("RGBA", (8, 8)), 0.5).mode == "RGBA"

    class Dotted(FakeEngine):
        def generate(self, req: server.GenerateRequest, seed: int, references: list[Any]) -> list[Any]:
            return [dot((20, 10), (4, 2))]

    client = TestClient(server.build_app(Dotted(), server.GpuWorker(), server.Progress(), root, {}))
    made = {}
    for slide_back in (0.0, 0.2, -0.2):
        body = _request(root, seed=3, transparent=False, slide_back=slide_back).model_dump()
        r = client.post("/spellcast/generate", json=body)
        assert r.status_code == 200, r.text
        with Image.open(r.json()["images"][0]) as saved:
            made[slide_back] = where(saved)
    assert made == {0.0: (4, 2), 0.2: (0, 0), -0.2: (8, 4)}, "slid back by what the picture was slid by"
    assert server.GenerateRequest(prompt="x", out_dir="C:/x").slide_back == 0.0
    for bad in (0.6, -0.6):
        assert client.post("/spellcast/generate", json={**body, "slide_back": bad}).status_code == 422


@needs_pil
def test_a_tileable_picture_is_laid_onto_the_one_it_was_made_from(root: Path) -> None:
    """keep_place: a tileable picture made from the first example picture lies some pixels off it (nothing holds a
    picture without borders); it is found by how little the two differ, and laid back, whole pixels, wrapping round.
    Another picture altogether, or one further off than is searched, stays as it is."""
    import random

    from PIL import Image, ImageChops, ImageDraw, ImageFilter

    def texture(seed: int, size: tuple[int, int] = (512, 384)) -> Any:
        rng = random.Random(seed)
        picture = Image.new("RGB", size, (60, 120, 40))
        draw = ImageDraw.Draw(picture)
        for _ in range(900):
            x, y = rng.randrange(size[0]), rng.randrange(size[1])
            draw.ellipse((x, y, x + rng.randrange(4, 26), y + rng.randrange(4, 26)), fill=tuple(rng.randrange(256) for _ in range(3)))
        return picture

    source = texture(1)
    # Made anew by the model: the same picture, not its pixels (blurred a little here), and moved.
    for moved in ((-8, -13), (6, 0), (0, 0), (-10, 0), (33, -47), (1, 1)):
        made = ImageChops.offset(source.filter(ImageFilter.GaussianBlur(1.2)), *moved)
        assert server.drift(source, made) == moved
        assert ImageChops.difference(server.in_place(made, source), ImageChops.offset(made, -moved[0], -moved[1])).getbbox() is None
    assert server.drift(source.convert("RGBA").resize((1024, 768)), ImageChops.offset(source, 5, -3)) == (5, -3), "another size"
    assert server.drift(source, texture(2)) == (0, 0), "another picture: it stays where it is"
    assert server.drift(source, ImageChops.offset(source, 200, 150)) == (0, 0), "further off than is searched"
    other = texture(2)
    assert server.in_place(other, source) is other

    # The request: only a tileable one, and only when it asks; then slid back as the page had slid its picture.
    ref = root / "p1" / "reference" / "texture.png"
    source.save(ref)

    class Drifting(FakeEngine):
        def generate(self, req: server.GenerateRequest, seed: int, references: list[Any]) -> list[Any]:
            return [ImageChops.offset(source, -8, -13)]

    client = TestClient(server.build_app(Drifting(), server.GpuWorker(), server.Progress(), root, {}))

    def saved(**changes: Any) -> Any:
        body = _request(root, seed=3, transparent=False, width=512, height=384, ref_images=[str(ref)], **changes).model_dump()
        r = client.post("/spellcast/generate", json=body)
        assert r.status_code == 200, r.text
        with Image.open(r.json()["images"][0]) as picture:
            return picture.convert("RGB")

    same = lambda a, b: ImageChops.difference(a, b).getbbox() is None      # noqa: E731
    assert same(saved(tileable=True, keep_place=True), source), "laid back onto it"
    assert same(saved(tileable=True), ImageChops.offset(source, -8, -13)), "not asked: as the model made it"
    assert same(saved(keep_place=True), ImageChops.offset(source, -8, -13)), "not tileable: moving it would wrap its edges round"
    assert same(saved(tileable=True, keep_place=True, slide_back=0.25), ImageChops.offset(source, -128, -96)), "then slid back"
    assert server.GenerateRequest(prompt="x", out_dir="C:/x").keep_place is False


GOLD = (242, 212, 32)


def _brushed_picture(root: Path, size: tuple[int, int] = (512, 384),
                     box: tuple[int, int, int, int] = (224, 160, 288, 224)) -> tuple[Any, Any, Path, Path]:
    """A picture of noise and a brushed box on it, saved as a Smart Mask job's picture and mask -> (the picture, the
    brush, their two paths)."""
    from PIL import Image, ImageDraw

    base = Image.effect_noise(size, 40).convert("RGB")
    brush = Image.new("L", size, 0)
    ImageDraw.Draw(brush).rectangle(box, fill=255)
    kept, where = root / "p1" / "images" / "kept.png", root / "p1" / "viewport" / "mask.png"
    for path, picture in ((kept, base), (where, brush)):
        path.parent.mkdir(parents=True, exist_ok=True)
        picture.save(path)
    return base, brush, kept, where


def _same(a: Any, b: Any, box: tuple[int, int, int, int]) -> bool:
    from PIL import ImageChops

    return ImageChops.difference(a.convert("RGB").crop(box), b.convert("RGB").crop(box)).getbbox() is None


def _near(pixel: tuple[int, ...], colour: tuple[int, ...], by: int = 3) -> bool:
    return all(abs(p - c) <= by for p, c in zip(pixel, colour))


def _redrawn(base: Any, where: Any = None) -> Any:
    """What the model makes of `base` where `where` is white (all of it without one): 8 levels darker, and with a
    grain of its own (every other pixel 12 up, the others 12 down)."""
    from PIL import Image, ImageChops

    grain = Image.new("L", base.size)
    grain.putdata([24 if (i % base.width + i // base.width) % 2 else 0 for i in range(base.width * base.height)])
    own = ImageChops.subtract(ImageChops.add(base, Image.merge("RGB", [grain] * 3)), Image.new("RGB", base.size, (20, 20, 20)))
    if where is None:
        return own
    picture = base.copy()
    picture.paste(own, mask=where)
    return picture


def _stretched(picture: Any, dx: float, wider: float, dy: float, taller: float) -> Any:
    """`picture` as a redraw that does not lie on it: what the picture has at x, y it has at (1 + wider) x + dx,
    (1 + taller) y + dy."""
    from PIL import Image

    return picture.transform(picture.size, Image.Transform.AFFINE,
                             (1 / (1 + wider), 0, -(dx - wider / 2) / (1 + wider),
                              0, 1 / (1 + taller), -(dy - taller / 2) / (1 + taller)), Image.Resampling.BICUBIC)


class _Redrawing(FakeEngine):
    """Gives back `made` - of a request for a window of the picture, that window (`box`) -, and keeps what each
    request was shown and held to."""
    made: Any = None
    box: tuple[int, int, int, int] | None = None

    def __init__(self) -> None:
        super().__init__()
        self.starts: list[dict[str, Any]] = []
        self.shown: list[list[Any]] = []

    def generate(self, req: server.GenerateRequest, seed: int, references: list[Any], **start: Any) -> list[Any]:
        self.starts.append(start)
        self.shown.append(references)
        return [self.made.copy() if self.made.size == (req.width, req.height) else self.made.crop(self.box)]


def _whole(picture: Any, rest: Any, brush: Any) -> tuple[Any, bool]:
    """redrawn_whole with the patch the launcher takes for what was brushed."""
    return server.redrawn_whole(picture, rest, brush, server.patch_of(brush.getbbox(), rest.size), 5)


def _answer(client: TestClient, root: Path, **changes: Any) -> tuple[Any, Any]:
    """-> (the picture saved, the answer's warning) of a 4 : 3 request."""
    from PIL import Image

    body = _request(root, seed=5, transparent=False, width=1024, height=768, **changes).model_dump()
    r = client.post("/spellcast/generate", json=body)
    assert r.status_code == 200, r.text
    with Image.open(r.json()["images"][0]) as picture:
        return picture.convert("RGB"), r.json()["warning"]


@needs_pil
def test_a_masked_change_is_drawn_whole_and_brought_back_to_the_pictures_tone(root: Path) -> None:
    """changed_image / changed_mask (the page's Smart Mask). The model draws all of the picture anew and nothing of
    it is held or laid back, so what it draws around the change - a shadow to the picture's edge - stays whole.
    Its tone is brought back to the picture's, read on the ground both show alike - away from the brush's room,
    or under a wide mask on all that did not change. A result with nothing new under the brush says so."""
    from PIL import Image, ImageChops, ImageDraw

    size = (512, 384)
    base, brush, kept, where = _brushed_picture(root)
    rest = base.convert("RGBA")

    def drawn(*things: tuple[tuple[int, int, int, int], tuple[int, int, int]]) -> Any:
        picture = _redrawn(base)
        for box, colour in things:
            ImageDraw.Draw(picture).rectangle(box, fill=colour)
        return picture

    fruit = ((232, 168, 280, 216), GOLD)                        # under the brush
    shadow = ((281, 200, 511, 216), (20, 20, 20))               # from it to the picture's edge
    out, changed = _whole(drawn(fruit, shadow), rest, brush)
    assert changed and (out.size, out.mode, out.getchannel("A").getextrema()) == (size, "RGBA", (255, 255))
    assert _near(out.getpixel((256, 192)), (250, 220, 40), 5), "the new thing, in the picture's tone (some 8 levels back)"
    assert all(_near(out.getpixel((x, 208)), (30, 30, 30), 4) for x in (300, 400, 505)), "its shadow whole, to the edge"
    far = [out.getpixel((x, 40))[1] - base.getpixel((x, 40))[1] for x in range(40, 200)]
    assert {abs(d) for d in far} >= {11, 12, 13} and abs(sum(far)) <= 2 * len(far), "the rest: the model's own grain, the picture's tone"
    # Nothing clearly new where it was brushed - the picture redrawn, or only another thing changed far off: said
    # so. A thing set beside the stroke, in its room, is the change asked for.
    stray = ((40, 300, 90, 350), (200, 0, 0))
    beside = ((300, 168, 340, 216), GOLD)
    assert [_whole(drawn(*things), rest, brush)[1] for things in ((), (stray,), (stray, fruit), (beside,))] == [
        False, False, True, True]
    # A faint shadow about the new thing, too faint to count as changed, does not bend the tone: it is read away
    # from the brush and its room.
    faint = drawn()
    faint.paste(ImageChops.subtract(faint, Image.new("RGB", size, (12, 12, 12))), mask=server.with_room(brush, size))
    out = _whole(faint, rest, brush)[0]
    far = [out.getpixel((x, 40))[1] - base.getpixel((x, 40))[1] for x in range(40, 200)]
    assert abs(sum(far)) <= len(far)
    # Brushed over most of the picture, its room leaves too little to read the tone on: it is read on all that did
    # not change. With nothing left that did not - another picture altogether -: as the model made it.
    wide = Image.new("L", size, 0)
    ImageDraw.Draw(wide).rectangle((60, 60, 450, 320), fill=255)
    out = _whole(drawn(), rest, wide)[0]
    far = [out.getpixel((x, 40))[1] - base.getpixel((x, 40))[1] for x in range(40, 200)]
    assert abs(sum(far)) <= 2 * len(far), "in the picture's tone"
    other = Image.new("RGB", size, (200, 40, 40))
    assert _whole(other, rest, wide)[0].convert("RGB").tobytes() == other.tobytes()
    shred = other.copy()                            # ... or with a shred that did not, under 2 % of the picture
    shred.paste(drawn().crop((20, 20, 82, 82)), (20, 20))
    assert _whole(shred, rest, wide)[0].convert("RGB").tobytes() == shred.tobytes()
    # An answer that came back see-through around its thing, unasked (the picture is not): the picture stays where
    # the answer shows nothing - it is not what changed.
    alone = Image.new("L", size, 0)
    ImageDraw.Draw(alone).rectangle((216, 152, 296, 232), fill=255)
    cutout = drawn(fruit).convert("RGBA")
    cutout.putalpha(alone)
    out, changed = _whole(cutout, rest, brush)
    assert changed and out.getchannel("A").getextrema() == (255, 255) and _near(out.getpixel((256, 192)), GOLD, 5)
    assert _same(out, base, (0, 0, 512, 100)) and _same(out, base, (0, 0, 150, 384)), "the picture, where the answer shows none"
    # A see-through picture: the tone is read where neither picture is see-through, and the colour left under a
    # see-through part is no change.
    clear = Image.new("L", size, 255)
    ImageDraw.Draw(clear).rectangle((260, 0, 511, 383), fill=0)
    cut = rest.copy()
    cut.putalpha(clear)
    model = drawn().convert("RGBA")
    model.paste((90, 20, 120, 0), (260, 0, 512, 384))
    out, changed = _whole(model, cut, brush)
    assert changed is False and out.getchannel("A").tobytes() == clear.tobytes()
    left = [out.getpixel((x, 40))[1] - base.getpixel((x, 40))[1] for x in range(40, 200)]
    assert abs(sum(left)) <= 2 * len(left), "the tone, read on the part that shows"
    # ... and what the model made see-through in it stays so: the picture is not laid under an answer to a
    # see-through picture.
    hole = model.copy()
    hole.paste((0, 0, 0, 0), (232, 168, 280, 216))
    out, changed = _whole(hole, cut, brush)
    assert changed and out.getpixel((256, 192))[3] == 0

    engine = _Redrawing()
    engine.made = drawn(fruit, shadow).resize((1024, 768))
    client = TestClient(server.build_app(engine, server.GpuWorker(), server.Progress(), root, {}))
    whole = {"changed_image": str(kept), "changed_mask": str(where), "init_image": str(kept), "mask_image": str(where),
             "strength": 1.0, "ref_images": [str(kept)]}                                             # as the hub asks
    change = server.change_of(str(kept), str(where), (1024, 768))
    assert [(c.size, c.mode) for c in change] == [((1024, 768), "RGBA"), ((1024, 768), "L")]
    picture, warning = _answer(client, root, **whole)
    assert picture.tobytes() == _whole(engine.made, *change)[0].convert("RGB").tobytes() and warning is None
    assert picture.tobytes() != engine.made.tobytes() and engine.starts[-1] == {}, "its tone brought back; nothing held"
    assert _answer(client, root)[0].tobytes() == engine.made.tobytes(), "not asked: the model's picture as it is"
    engine.made = drawn().resize((1024, 768))
    assert "only slightly" in _answer(client, root, **whole)[1]

    class OneOfTwo(_Redrawing):         # two pictures of one request, the change in the second alone
        def generate(self, req: server.GenerateRequest, seed: int, references: list[Any], **start: Any) -> list[Any]:
            return [drawn().resize((1024, 768)), drawn(fruit).resize((1024, 768))]

    two = TestClient(server.build_app(OneOfTwo(), server.GpuWorker(), server.Progress(), root, {}))
    assert _answer(two, root, batch_count=2, **whole)[1] is None, "one of the pictures has the change: nothing to say"
    # Asked for in another shape than the picture brushed over, there is no tone to bring it back to.
    tall = root / "p1" / "images" / "tall.png"
    base.resize((384, 512)).save(tall)
    picture, warning = _answer(client, root, **{**whole, "changed_image": str(tall)})
    assert picture.tobytes() == engine.made.tobytes() and warning is None
    for half in ({"changed_image": str(kept)}, {"changed_mask": str(where)}, {**whole, "changed_mask": str(root / "p1" / "nope.png")},
                 {**whole, "changed_image": "C:/elsewhere/picture.png"},
                 {"changed_image": str(kept), "changed_mask": str(where)}):       # nothing to hold the drawing to
        r = client.post("/spellcast/generate", json=_request(root, **half).model_dump())
        assert r.status_code == 400 and r.json()["kind"] == "input", half


@needs_pil
def test_a_held_masked_change_keeps_the_rest_of_the_picture_to_the_pixel(root: Path) -> None:
    """A tileable picture's masked change is drawn HELD (redrawn whole, the picture would no longer repeat). The
    model draws in a room around what was brushed. Of that, only what it really changed and that reaches the brush
    is kept - the new thing with its shadow -, brought to the picture's tone; everywhere else, in the room too, the
    picture itself is laid back, its own pixels. Slid by the page's offset, it is kept as slid. A result with
    nothing new under the brush says so; asked for in another shape, nothing is kept and the drawing is not held
    either."""
    from PIL import Image, ImageChops, ImageDraw

    size = (512, 384)
    base, brush, kept, where = _brushed_picture(root)

    # The room: what was brushed and a share of the longer side around it (61 px here), black and white.
    room = server.with_room(brush, size)
    assert (room.mode, room.size, sorted(set(room.tobytes()))) == ("L", size, [0, 255])
    assert [room.getpixel((x, 192)) for x in (150, 170, 256, 345, 372)] == [0, 255, 255, 255, 0]
    assert [room.getpixel((256, y)) for y in (86, 106, 282, 308)] == [0, 255, 255, 0]
    assert server.with_room(Image.new("L", size, 0), size).getbbox() is None
    assert server.brushed(brush.resize((256, 192)), size).getpixel((256, 192)) == 255, "at the picture's size, whatever the mask's"
    assert server.brushed(Image.new("L", size, 100), size).getbbox() is None, "a faint mask is none"
    lines = Image.new("RGB", (256, 192))
    lines.putdata([(255, 255, 255) if x % 2 else (0, 0, 0) for _y in range(192) for x in range(256)])
    fine, coarse = root / "p1" / "images" / "lines.png", root / "p1" / "viewport" / "small-mask.png"
    lines.save(fine)
    brush.resize((256, 192)).save(coarse)
    half, _ = server.change_of(str(fine), str(coarse), size)
    assert len(set(half.getchannel("R").tobytes())) > 2, "a smaller picture is brought to the request's size smoothly"
    keep = server.Kept(base.convert("RGBA"), brush, room, (0, 0), base.convert("RGBA"), (0, 0))

    def drawn(*things: tuple[tuple[int, int, int, int], tuple[int, int, int]]) -> Any:
        """What the model made: the picture, drawn anew all over the room (_redrawn), and `things` drawn into it."""
        picture = _redrawn(base, room)
        for box, colour in things:
            ImageDraw.Draw(picture).rectangle(box, fill=colour)
        return picture

    fruit = ((232, 168, 280, 216), GOLD)                        # under the brush
    shadow = ((281, 200, 330, 216), (20, 20, 20))               # joined to it, reaching beyond the brush
    stray = ((164, 264, 184, 284), (200, 0, 0))                 # in the room too, but another thing altogether
    out, changed = server.kept_outside(drawn(fruit, shadow, stray), keep)
    assert changed and (out.size, out.mode, out.getchannel("A").getextrema()) == (size, "RGBA", (255, 255))
    assert _near(out.getpixel((256, 192)), (250, 220, 40)), "the new thing, in the picture's tone (the room's 8 levels back)"
    assert _near(out.getpixel((310, 208)), (28, 28, 28)), "its shadow beyond the brush is kept with it"
    assert _same(out, base, (160, 260, 190, 290)), "another thing that changed beside it is not the change asked for"
    assert _same(out, base, (310, 100, 350, 140)), "the room's own ground, drawn anew a little darker: the picture's again"
    for far in ((0, 0, 512, 86), (0, 308, 512, 384), (0, 0, 150, 384), (372, 0, 512, 384)):
        assert _same(out, base, far), "outside the room: the picture itself, to the pixel"
    off = {abs(out.getpixel((x, y))[1] - base.getpixel((x, y))[1]) for x in range(190, 232) for y in (176, 180, 184)}
    assert 0 in off and any(10 <= d for d in off) and any(2 <= d <= 7 for d in off), "a soft edge: the model's grain fades into the picture's"
    # Nothing clearly new under the brush - the ground drawn anew, or only another thing changed: said so, and what
    # the model drew under the brush is kept all the same (a slight change cannot be told from none), in the
    # picture's tone; beside the brush the picture is its own.
    for nothing in (drawn(), drawn(stray)):
        again, told = server.kept_outside(nothing, keep)
        assert told is False and _same(again, base, (310, 100, 350, 140)) and _same(again, base, (160, 260, 190, 290))
        under = [again.getpixel((x, 192))[1] - base.getpixel((x, 192))[1] for x in range(236, 276)]
        assert {abs(d) for d in under} >= {11, 12, 13} and abs(sum(under)) <= 2 * len(under), "the model's grain, the picture's tone"
    # ... so a slight change is not lost: a part 10 levels darker under the brush stays 10 levels darker.
    slight = drawn()
    patch = (236, 172, 276, 212)
    slight.paste(ImageChops.subtract(slight, Image.new("RGB", size, (10, 10, 10))).crop(patch), patch)
    dimmed, told = server.kept_outside(slight, keep)
    drop = [base.getpixel((x, 192))[1] - dimmed.getpixel((x, 192))[1] for x in range(240, 272)]
    assert told is False and 8 <= sum(drop) / len(drop) <= 12
    # What is kept: the brush and the part of what changed that it reaches, 2 % wider (10 px here) - never a part
    # that lies apart from it.
    spots = Image.new("L", size, 0)
    for box in ((270, 176, 330, 208), (164, 264, 184, 284)):
        ImageDraw.Draw(spots).rectangle(box, fill=255)
    part = server.reached_from(brush, spots, room)
    assert [part.getpixel((x, 192)) for x in (204, 218, 256, 320, 336, 350)] == [0, 255, 255, 255, 255, 0]
    assert part.getpixel((174, 274)) == 0 and part.getpixel((256, 140)) == 0
    over = spots.copy()
    ImageDraw.Draw(over).rectangle((300, 176, 420, 208), fill=255)                     # ... nor past the room's border
    assert ImageChops.subtract(server.reached_from(brush, over, room), room).getbbox() is None
    alone = server.reached_from(brush, Image.new("L", size, 0), room)
    assert [alone.getpixel((x, 192)) for x in (204, 218, 256, 294, 308)] == [0, 255, 255, 255, 0], "the brush itself, always"
    # ... but for one that begins close to it (6 % of the side, 31 px here): the thing set beside a thin stroke
    beside = Image.new("L", size, 0)
    for box in ((308, 176, 340, 208), (164, 264, 184, 284)):                           # 20 px and 40 px from the brush
        ImageDraw.Draw(beside).rectangle(box, fill=255)
    near = server.reached_from(brush, beside, room)
    assert near.getpixel((324, 192)) == 255 and near.getpixel((174, 274)) == 0
    apple = ((300, 232, 330, 262), GOLD)                                               # beside the brush, not under it
    put, told = server.kept_outside(drawn(apple), keep)
    assert told and _near(put.getpixel((315, 247)), (250, 220, 40), 6), "kept, and no word that nothing changed"
    twice = server.reached_from(brush, beside, room, 1024)
    assert twice.getpixel((174, 274)) == 255, "of a picture twice as large, twice as far"
    far = ((140, 288, 184, 332), (200, 0, 0))                                          # 64 px from the brush
    window = server.Kept(keep.rest, brush, server.with_room(brush, size, 1024), (0, 0),
                         base.resize((1024, 768)).convert("RGBA"), (0, 0))
    assert _same(server.kept_outside(drawn(far), keep)[0], base, far[0])
    assert server.kept_outside(drawn(far), window)[0].getpixel((162, 310))[0] > 150, "a window's lengths are its whole picture's"
    # ... but how clearly a thing changed is told at the scale of the pixels the model drew, the window's: a small
    # thing 50 levels darker is plain there, and would drown in a blur as wide as the whole picture's share.
    small, patch = drawn(), (244, 180, 268, 204)
    small.paste(ImageChops.subtract(small, Image.new("RGB", size, (50, 50, 50))).crop(patch), patch)
    assert server.kept_outside(small, window)[1] is True, "a small thing in a window of a larger picture is told"
    # A change that fills most of the room leaves too little ground to read the tone from: it stays as the model made it.
    filled, _ = server.kept_outside(drawn(((170, 106, 300, 282), GOLD)), keep)
    assert _near(filled.getpixel((256, 192)), GOLD, 1)
    # The tone alone: a picture 8 levels darker all over comes back to the picture's levels where the ground was read;
    # a new thing's own colours, darker or lighter than any ground, move as far as the nearest level read does.
    darker = ImageChops.subtract(base, Image.new("RGB", size, (8, 8, 8)))
    ImageDraw.Draw(darker).rectangle((232, 168, 256, 216), fill=(2, 2, 2))
    ImageDraw.Draw(darker).rectangle((257, 168, 280, 216), fill=(246, 246, 246))
    toned = server.in_tone(darker, base, ImageChops.subtract(room, brush))
    assert _near(toned.getpixel((300, 120)), base.getpixel((300, 120)), 1)
    assert _near(toned.getpixel((240, 192)), (10, 10, 10), 1) and _near(toned.getpixel((270, 192)), (254, 254, 254), 1)

    # A request's change: at the request's size, the picture slid as the page showed it.
    both = {"changed_image": str(kept), "changed_mask": str(where)}
    held = {**both, "init_image": str(kept), "mask_image": str(where), "strength": 1.0, "ref_images": [str(kept)]}   # as the hub asks

    def kept_of(**changes: Any) -> Any:
        req = _request(root, width=1024, height=768, **held, **changes)
        return server.kept_from(req, server.change_of(req.changed_image, req.changed_mask, (req.width, req.height)))

    asked = kept_of()                   # its window: 768 px wide, round what was brushed
    box = (129, 0, 897, 768)
    assert (asked.whole.size, asked.whole.mode, asked.moved) == ((1024, 768), "RGBA", (0, 0))
    assert (asked.at, asked.rest.size, asked.brush.size) == (box[:2], (768, 768), (768, 768))
    assert asked.rest.tobytes() == asked.whole.crop(box).tobytes()
    assert asked.room.tobytes() == ImageChops.multiply(server.with_room(asked.brush, (768, 768), 1024),
                                                       server.within_rim(box, (1024, 768))).tobytes()
    assert server.with_room(asked.brush, (768, 768), 1024).getbbox()[0] < server.with_room(asked.brush, (768, 768)).getbbox()[0], \
        "the room is the whole picture's share, not the window's"
    slid = kept_of(slide_back=0.25)
    assert slid.whole.tobytes() == ImageChops.offset(asked.whole, 256, 192).tobytes(), "as the page showed it"
    # Which way a change is drawn: held for a tileable picture alone - moved first, the brush to its middle (a pixel
    # here) -; any other whole, also on a card that shows the model a smaller copy of it
    tiled, tiled_slid = kept_of(tileable=True), kept_of(tileable=True, slide_back=0.25)
    there = (*tiled.at, tiled.at[0] + 768, tiled.at[1] + 768)
    assert tiled.moved == (1023, 767) and there == (128, 0, 896, 768)
    held = {**held, "tileable": True}

    def made(*things: Any) -> Any:              # what the model draws of the picture as it is moved
        return ImageChops.offset(drawn(*things).resize((1024, 768)), *tiled.moved)

    engine = _Redrawing()
    engine.placement = {"budget_mb": 3_000, "prefix_cache": "auto"}
    assert server.copy_cap(engine.placement, _request(root, width=1024, height=768, **held), 1) is not None
    engine.made, engine.box = made(fruit, shadow), there
    client = TestClient(server.build_app(engine, server.GpuWorker(), server.Progress(), root, {}))
    expected = server.kept_outside(engine.made.crop(there), tiled)[0].convert("RGB")
    assert _answer(client, root, **held) == (expected, None)
    held_by = engine.starts[-1]
    assert sorted(held_by) == ["init", "mask"], "the drawing itself is held to the picture ..."
    assert held_by["mask"].tobytes() == tiled.room.tobytes(), "... outside the room"
    assert _answer(client, root, **held, slide_back=0.25)[0].tobytes() == ImageChops.offset(
        server.kept_outside(engine.made.crop(there), tiled_slid)[0].convert("RGB"), -256, -192).tobytes(), "kept as shown, then slid back"
    # Nothing new under the brush: the picture is the one it was, and the answer says what to try.
    engine.made = made(stray)
    picture, warning = _answer(client, root, **held)
    assert _same(picture, asked.whole, (0, 0, 1024, 172)) and _same(picture, asked.whole, (320, 520, 380, 580)) and "only slightly" in warning
    engine.made = made(fruit)
    # Asked for in another shape than the picture brushed over: nothing of that picture fits it, so all of it is drawn
    # anew, unheld - and the answer says why. Another size of the same shape is kept (above).
    tall = root / "p1" / "images" / "tall.png"
    base.resize((384, 512)).save(tall)
    picture, warning = _answer(client, root, **{**held, "changed_image": str(tall)})
    assert picture.tobytes() == engine.made.tobytes() and "another shape" in warning and engine.starts[-1] == {}
    assert server.same_shape((1024, 1024), (1536, 1536)) and not server.same_shape((992, 832), (1472, 1248))
    # The same request of a picture that is not tileable: drawn whole, nothing held.
    assert _answer(client, root, **{**held, "tileable": False})[1] is None and engine.starts[-1] == {}
    for half in ({"changed_image": str(kept)}, {"changed_mask": str(where)}, both, {**held, "changed_mask": str(root / "p1" / "nope.png")},
                 {**held, "changed_image": "C:/elsewhere/picture.png"}):
        r = client.post("/spellcast/generate", json=_request(root, **half).model_dump())
        assert r.status_code == 400 and r.json()["kind"] == "input", half


@needs_pil
def test_a_held_change_of_a_tileable_picture_is_drawn_with_the_brush_in_its_middle(root: Path) -> None:
    """A tileable picture brushed at its edge. Picture, brush and room are moved, wrapping round, so that the brush
    lies in the middle (to_the_middle): the model is shown the picture laid that way and held to it, and what it
    drew is moved back - a thing drawn across the edge goes on at the other side, and the rest is the picture's own
    pixels. A picture that is not tileable is not moved: its room ends at its edge."""
    from PIL import Image, ImageChops, ImageDraw

    size = (1024, 768)

    def at(*boxes: tuple[int, int, int, int]) -> Any:
        mask = Image.new("L", size, 0)
        for box in boxes:
            ImageDraw.Draw(mask).rectangle(box, fill=255)
        return mask

    assert server.to_the_middle(at((448, 320, 575, 447))) == (0, 0), "in the middle already"
    assert server.to_the_middle(at((960, 320, 1023, 447))) == (544, 0), "at the right edge: 480 px left, round the edge"
    assert server.to_the_middle(at((0, 320, 31, 447), (992, 320, 1023, 447))) == (512, 0), "across the edge: one stretch"
    assert server.to_the_middle(at((448, 0, 575, 63))) == (0, 352)
    assert server.to_the_middle(at()) == (0, 0) and server.to_the_middle(at((0, 0, 1023, 767))) == (0, 0)

    base, brush, kept, where = _brushed_picture(root, size, (960, 320, 1023, 447))
    other = root / "p1" / "reference" / "other.png"
    other.parent.mkdir(parents=True, exist_ok=True)
    Image.effect_noise((256, 256), 30).save(other)

    class Drawing(_Redrawing):
        """Draws a gold bar over the middle of the room it is given, on the picture it is held to."""

        def generate(self, req: server.GenerateRequest, seed: int, references: list[Any], **start: Any) -> list[Any]:
            self.starts.append(start)
            self.shown.append(references)
            if not start:               # drawn whole: the picture it is shown, as it is
                return [references[0].convert("RGB")]
            picture = start["init"].convert("RGB")
            left, top, right, bottom = start["mask"].getbbox()
            ImageDraw.Draw(picture).rectangle(((left + right) // 2 - 70, (top + bottom) // 2 - 40,
                                               (left + right) // 2 + 70, (top + bottom) // 2 + 40), fill=GOLD)
            return [picture]

    engine = Drawing()
    client = TestClient(server.build_app(engine, server.GpuWorker(), server.Progress(), root, {}))
    held = {"changed_image": str(kept), "changed_mask": str(where), "init_image": str(kept), "mask_image": str(where),
            "strength": 1.0, "ref_images": [str(kept), str(other)]}
    picture, warning = _answer(client, root, tileable=True, **held)
    moved = ImageChops.offset(base, 544, 0).crop((128, 0, 896, 768))        # ... and its window there
    room = engine.starts[-1]["mask"].getbbox()
    assert warning is None and 0 < room[0] and room[2] < 768, "the room lies whole inside the window"
    assert _same(engine.starts[-1]["init"], moved, (0, 0, 768, 768)), "held to the picture laid that way ..."
    assert _same(engine.shown[-1][0], moved, (0, 0, 768, 768)), "... and shown it laid that way"
    assert engine.shown[-1][1].size == (256, 256) and _same(engine.shown[-1][1], Image.open(other), (0, 0, 256, 256)), \
        "another example picture is as it is"
    assert len(engine.shown[-1]) == 2, "nothing else: the whole picture beside its window, the model drew into the mask"
    tiled = _request(root, width=1024, height=768, tileable=True, **held)
    there = server.kept_from(tiled, server.change_of(tiled.changed_image, tiled.changed_mask, size))
    assert (there.moved, there.at) == ((544, 0), (128, 0))
    assert server.window_place(there) == [608, 0, 1376, 768], "as the page shows the picture: past its right side, round the edge"
    keep = server.Kept(base.convert("RGBA"), brush, brush, (544, 0), base.convert("RGBA"), (0, 0))
    small = base.resize((512, 384))
    _drawn, shown, _start, _room = server.held_to(keep, _request(root, init_image=str(kept), ref_images=[str(kept)]), [small], base)
    assert _same(shown[0], ImageChops.offset(small, 272, 0), (0, 0, 512, 384)), "a copy of half the size moves half as far"
    assert _near(picture.getpixel((1000, 384)), GOLD) and _near(picture.getpixel((20, 384)), GOLD), "across the edge, and on at the other side"
    assert _same(picture, base, (160, 0, 800, 768)) and _same(picture, base, (0, 0, 1024, 180)), "the rest: the picture's own pixels"
    # Not tileable, also on a card that shows the picture smaller: drawn whole, nothing held or moved.
    engine.placement = {"budget_mb": 3_000, "prefix_cache": "auto"}
    _answer(client, root, **held)
    assert engine.starts[-1] == {} and _same(engine.shown[-1][0], base, (0, 0, 1024, 768))


@needs_pil
def test_a_held_change_of_a_see_through_picture_stays_see_through(root: Path) -> None:
    """A cut-out brushed across its edge. The drawing starts from the picture as it is, RGBA (laid on black the
    model draws a black ground around the new thing); the colour left under a see-through part is no change, a
    thing drawn onto one is; the tone is read where neither picture is see-through; and where the two are mixed, a
    see-through pixel adds nothing of the colour under it."""
    from PIL import Image, ImageDraw

    size = (512, 384)
    base, brush, kept, where = _brushed_picture(root)
    clear = Image.new("L", size, 255)
    ImageDraw.Draw(clear).rectangle((256, 0, 511, 383), fill=0)             # the right half: see-through
    rest = base.convert("RGBA")
    rest.paste((0, 0, 0, 0), (256, 0, 512, 384))
    rest.save(kept)
    room = server.with_room(brush, size)
    keep = server.Kept(rest, brush, room, (0, 0), rest, (0, 0))

    def drawn(*things: tuple[tuple[int, int, int, int], tuple[int, int, int]]) -> Any:
        """The model's picture: the room drawn anew, see-through where the picture is - with another colour left
        under it -, and `things` drawn into it, solid."""
        picture = _redrawn(base, room).convert("RGBA")
        picture.paste((90, 20, 120, 0), (256, 0, 512, 384))
        for box, colour in things:
            ImageDraw.Draw(picture).rectangle(box, fill=(*colour, 255))
        return picture

    assert server.changed_in(room, drawn(), rest).getbbox() is None, "the colour under a see-through part is no change"
    out, changed = server.kept_outside(drawn(((240, 176, 300, 208), GOLD)), keep)
    assert changed and _near(out.getpixel((290, 192)), (*GOLD, 255), 12), "a thing drawn onto the see-through part is kept"
    assert out.getpixel((280, 216))[3] == 0 and out.getpixel((280, 166))[3] == 0, "beside it, under the brush: see-through still"
    assert out.crop((330, 0, 512, 384)).tobytes() == rest.crop((330, 0, 512, 384)).tobytes(), "outside: the picture's own, to the bit"
    wood = [out.getpixel((x, 192))[1] - base.getpixel((x, 192))[1] for x in range(226, 238)]
    assert abs(sum(wood)) <= 3 * len(wood), "the tone, read on the part that shows"
    # Mixed as far as each shows: half of a solid gold over a see-through pixel is gold, half see-through.
    where_mixed = Image.new("L", (4, 1))
    where_mixed.putdata([0, 128, 255, 64])
    mixed = server.laid_onto(Image.new("RGBA", (4, 1), (0, 0, 0, 0)), Image.new("RGBA", (4, 1), (*GOLD, 255)), where_mixed)
    assert (mixed.getpixel((0, 0)), mixed.getpixel((2, 0))) == ((0, 0, 0, 0), (*GOLD, 255)), "each one's own pixels"
    assert _near(mixed.getpixel((1, 0)), (*GOLD, 128), 3) and _near(mixed.getpixel((3, 0)), (*GOLD, 64), 4), "no dark fringe"
    # ... to the bit, the colour left under a see-through pixel and a half see-through one's too
    own = server.laid_onto(Image.new("RGBA", (4, 1), (90, 20, 120, 0)), Image.new("RGBA", (4, 1), (201, 99, 37, 77)), where_mixed)
    assert (own.getpixel((0, 0)), own.getpixel((2, 0))) == ((90, 20, 120, 0), (201, 99, 37, 77))
    # The start picture: as it is for a held masked change by a model that draws see-through pictures; a viewport's
    # picture, and any for FLUX.3, on black.
    held = {"changed_image": str(kept), "changed_mask": str(where), "init_image": str(kept), "mask_image": str(where)}
    start, mask = server.start_pictures(_request(root, width=512, height=384, **held))
    assert (start.mode, start.getpixel((400, 100))[3], mask.mode) == ("RGBA", 0, "L")
    for changes in ({"init_image": str(kept), "mask_image": str(where)}, {**held, "model": "flux3"}):
        start, _ = server.start_pictures(_request(root, width=512, height=384, **changes))
        assert (start.mode, start.getpixel((400, 100))) == ("RGB", (0, 0, 0))


def test_without_the_add_on_an_albedo_is_refused_in_words(root: Path) -> None:
    client = TestClient(server.build_app(FakeEngine(), server.GpuWorker(), server.Progress(), root, {}))
    r = client.post("/spellcast/albedo", json={"picture": str(root / "p1" / "a.png"), "out": str(root / "p1" / "b.png")})
    assert r.status_code == 400 and r.json() == {"detail": server.ALBEDO_MISSING, "kind": "input"}


@needs_pil
def test_the_albedo_route_writes_the_add_ons_answer(root: Path, tmp_path: Path) -> None:
    from PIL import Image

    engine = AlbedoEngine({key: tmp_path / name for key, name in server.ALBEDO_FILES.items()})
    client = TestClient(server.build_app(engine, server.GpuWorker(), server.Progress(), root, {}))
    assert client.get("/spellcast/health").json()["albedo"] is True
    picture, out = root / "p1" / "view.png", root / "p1" / "flat" / "albedo.png"
    Image.new("RGB", (64, 32), (200, 100, 50)).save(picture)
    r = client.post("/spellcast/albedo", json={"picture": str(picture), "out": str(out)})
    assert r.status_code == 200, r.text
    assert r.json()["image"] == str(out) and engine.albedo_sizes == [(64, 32)]
    with Image.open(out) as written:
        assert written.size == (64, 32) and len(set(written.getpixel((0, 0)))) == 1      # the fake's grey
    Image.new("RGB", (60, 32)).save(root / "p1" / "odd.png")
    for body in ({"picture": str(root / "p1" / "odd.png"), "out": str(out)},           # not multiples of 32
                 {"picture": str(picture), "out": str(tmp_path.parent / "x.png")},       # outside the projects
                 {"picture": str(picture), "out": str(root / "p1" / "x.jpg")},           # not a .png
                 {"picture": str(root / "p1" / "none.png"), "out": str(out)}):           # no such picture
        r = client.post("/spellcast/albedo", json=body)
        assert r.status_code == 400 and r.json()["kind"] == "input", body
    assert engine.albedo_sizes == [(64, 32)]
    # a sheet of several views: one pass over all of it, smaller, and back at the sheet's size
    sheet = root / "p1" / "sheet.png"
    Image.new("RGB", (1536, 1536), (200, 100, 50)).save(sheet, compress_level=1)
    assert client.post("/spellcast/albedo", json={"picture": str(sheet), "out": str(out)}).status_code == 200
    assert engine.albedo_sizes[1:] == [(1152, 1152)]
    with Image.open(out) as written:
        assert written.size == (1536, 1536) and len(set(written.getpixel((700, 700)))) == 1


@needs_pil
def test_an_albedo_failure_reads_its_own_log_lines_and_a_broken_card_restarts(root: Path, tmp_path: Path) -> None:
    """The warning lines an earlier call left (an encode that fell back to tiles, say) do not make a later failure
    an "oom"; a failure that leaves CUDA unusable answers "restart" and ends the process, as for pictures."""
    from PIL import Image

    progress = server.Progress()
    exits: list[bool] = []
    cuda = [True]

    class Failing(AlbedoEngine):
        def albedo(self, picture: Any) -> Any:              # as QwenEngine.albedo sorts a failure by the log
            lines = list(progress.lines)
            if server.ran_out_of_memory(lines):
                raise server.OutOfMemory(" ".join(lines))
            raise server.GenerationFailed("the transformer pass failed")

    engine = Failing({key: tmp_path / name for key, name in server.ALBEDO_FILES.items()})
    client = TestClient(server.build_app(engine, server.GpuWorker(), progress, root, {}, cuda_works=lambda: cuda[0],
                                         exit_later=lambda: exits.append(True)))
    picture, out = root / "p1" / "view.png", root / "p1" / "albedo.png"
    Image.new("RGB", (64, 32)).save(picture)
    body = {"picture": str(picture), "out": str(out)}
    progress.on_log(sdcpp.SD_LOG_WARN, "ggml_cuda: failed to allocate 1536 MB; encoding in tiles")
    r = client.post("/spellcast/albedo", json=body)
    assert r.status_code == 500 and r.json()["kind"] == "error" and engine.ready and exits == []
    cuda[0] = False
    r = client.post("/spellcast/albedo", json=body)
    assert r.status_code == 500 and r.json()["kind"] == "restart" and "starts again" in r.json()["detail"]
    assert exits == [True] and engine.ready is False and not out.exists()
    assert client.post("/spellcast/albedo", json=body).json()["kind"] == "restart"      # until it is gone


def test_switching_models_frees_the_one_loaded_first(tmp_path: Path) -> None:
    """use_model: nothing for the model loaded; the other one only after the loaded context is freed (never two at
    once); a model that cannot be loaded brings the one before back (its error raised all the same); when neither
    loads there is no context, and the engine is not ready."""
    engine, lib = _two_model_engine(tmp_path, TURBO_FILE, FULL_FILE)
    assert (engine.model, engine.ctx, engine.ready) == ("turbo", 101, True) and lib.made == [(101, TURBO_FILE)]
    assert engine.use_model("turbo") is None and lib.freed == []
    seconds = engine.use_model("full")
    assert isinstance(seconds, float) and (engine.model, engine.ctx, engine.ready) == ("full", 102, True)
    assert lib.freed == [101] and lib.made[-1] == (102, FULL_FILE)
    assert engine.files["diffusion"] == tmp_path / FULL_FILE and engine.files["vae"] == tmp_path / server.VAE_FILE
    assert engine.progress.phase == "idle"                     # the switch's loading words are gone again
    engine.use_model("turbo")
    assert lib.freed == [101, 102] and lib.made[-1] == (103, TURBO_FILE) and engine.model == "turbo"

    lib.fail_for = {FULL_FILE}                                 # e.g. out of memory: the Turbo comes back
    with pytest.raises(server.GenerationFailed):
        engine.use_model("full")
    assert lib.freed == [101, 102, 103] and lib.made[-1] == (104, TURBO_FILE)
    assert (engine.model, engine.ctx, engine.ready) == ("turbo", 104, True)
    lib.fail_for = {FULL_FILE, TURBO_FILE}                     # neither: no context, not ready (the route restarts)
    with pytest.raises(server.GenerationFailed):
        engine.use_model("full")
    assert engine.ctx is None and engine.ready is False and lib.freed[-1] == 104


def test_the_model_files_are_read_ahead_while_the_engine_reads_them(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """QwenEngine.load: the model's files are read ahead in the engine's own order, from the moment the engine says
    it has its page-locked memory (PIN_MARK) - not before: that memory would be taken out of the cache just filled -,
    at once when nothing is page-locked, after the load with an engine that never says so, never without the mapping;
    a load gives up the read-ahead of the model before."""
    started: list[tuple[list[str], threading.Event]] = []

    def read_ahead(paths: list[Path], stop: threading.Event) -> None:
        started.append(([path.name for path in paths], stop))

    def count() -> int:                     # (a moment for the thread just started, if any)
        time.sleep(0.05)
        return len(started)

    monkeypatch.setattr(server, "read_ahead", read_ahead)
    for name in (server.VAE_FILE, server.LLM_FILE, server.VISION_FILE, TURBO_FILE, FULL_FILE):
        (tmp_path / name).write_bytes(b"")
    files, _missing = server.model_files(tmp_path, "Q4_K_M")
    locked = "model params: 4005.00 MB in page-locked host memory (CUDA_Host)"

    class Lib(_ContextLib):
        says: list[str] = []                # the engine's log lines while it makes a context
        counts: list[int] = []              # read-aheads started: as the context begins, then after each line

        def sd_set_log_callback(self, callback: Any, data: Any) -> None:
            self.log = callback

        def new_sd_ctx(self, params: Any) -> int | None:
            self.counts = [count()]
            for line in self.says:
                self.log(sdcpp.SD_LOG_INFO, line.encode(), None)
                self.counts.append(count())
            return super().new_sd_ctx(params)

    def engine_with(**placement: Any) -> tuple[Any, Lib]:
        lib = Lib()
        placed = {**server.choose_placement(11 * GiB, 10_664), "capability": (8, 6), **placement}
        return server.QwenEngine(lib, files, placed, server.Progress(), model="turbo"), lib

    order = [TURBO_FILE, server.LLM_FILE, server.VISION_FILE, server.VAE_FILE]      # all of it read at the load
    # the order: what the engine reads as it loads first (the page-locked files, then the VAE), in its own order;
    # then what it only maps, as a first picture uses it
    names = {"diffusion": "d", "vae": "v", "llm": "l", "vision": "i"}
    ordered = lambda pinned, **gone: "".join(  # noqa: E731
        path.name for path in server.read_order({key: None if key in gone else Path(name)
                                                 for key, name in names.items()}, pinned))
    assert ordered("diffusion,text_encoder") == "dliv" and ordered("all") == "dliv"
    assert ordered("diffusion") == "dvli" and ordered("text_encoder") == "livd"
    assert ordered(None) == "vldi" and ordered("") == "vldi"
    assert ordered(None, vision=True) == "vld" and ordered("diffusion,text_encoder", vision=True) == "dlv"   # FLUX.3
    # page-locked weights: nothing until the engine has taken its memory, then once
    engine, lib = engine_with(mmap=True, pin_params="diffusion,text_encoder")
    lib.says = ["loading diffusion model", f"Conditioner {locked}", f"Diffusion {locked}"]
    engine.load()
    assert lib.counts == [0, 0, 1, 1] and [names for names, _stop in started] == [order]
    assert not started[0][1].is_set()
    engine.use_model("full")                # another model: the first read-ahead gives up, its own files are read
    assert started[0][1].is_set() and len(started) == 2 and lib.counts == [1, 1, 2, 2]
    assert started[1][0] == [FULL_FILE, *order[1:]] and not started[1][1].is_set()
    # an engine that never says so (the official build ignores SD_PIN_PARAMS): after the load, as before
    del started[:]
    engine, lib = engine_with(mmap=True, pin_params="diffusion")
    lib.says = ["loading diffusion model"]
    engine.load()
    assert lib.counts == [0, 0] and count() == 1
    assert started[0][0] == [TURBO_FILE, server.VAE_FILE, server.LLM_FILE, server.VISION_FILE]
    # nothing page-locked (a PC with less RAM): at once, the engine takes no memory first
    del started[:]
    engine, lib = engine_with(mmap=True, pin_params=None)
    lib.says = ["loading diffusion model"]
    engine.load()
    assert lib.counts == [1, 1] and count() == 1
    assert started[0][0] == [server.VAE_FILE, server.LLM_FILE, TURBO_FILE, server.VISION_FILE]
    # without the mapping the weights are read into private memory once: no read-ahead
    del started[:]
    engine, lib = engine_with(mmap=False, pin_params="diffusion")
    lib.says = [f"Diffusion {locked}"]
    engine.load()
    assert lib.counts == [0, 0] and count() == 0


def test_the_read_ahead_reads_the_files_in_order_and_gives_up_when_told(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(server, "READ_CHUNK", 4)
    first, second = tmp_path / "a.gguf", tmp_path / "b.gguf"
    first.write_bytes(b"x" * 10)
    second.write_bytes(b"y" * 6)
    opened: list[str] = []
    real_open = open

    def spy(path: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(Path(path).name)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", spy)
    server.read_ahead([first, second])
    assert opened == ["a.gguf", "b.gguf"] and "Model files read ahead (0.0 GB in 0 s)" in capsys.readouterr().out
    stop = threading.Event()
    stop.set()
    del opened[:]
    server.read_ahead([first, second], stop)
    assert opened == ["a.gguf"] and capsys.readouterr().out == "", "given up at its first chunk, without a word"
    server.read_ahead([tmp_path / "gone.gguf"])
    assert "Reading the model files ahead stopped" in capsys.readouterr().out
    server.read_ahead([second], None, "Engine libraries")
    assert "Engine libraries read ahead (0.0 GB in 0 s)" in capsys.readouterr().out


def test_the_switching_words_are_the_requests_progress() -> None:
    progress = server.Progress()
    progress.loading("Loading the full Qwen-Image 2.1")
    progress.on_step(3, 10, 0.5)
    assert progress.to_dict() == {"phase": "loading", "step": 0, "steps": 0, "fraction": 0.0,
                                  "message": "Loading the full Qwen-Image 2.1", "eta": None}
    progress.stop()
    progress.loading()                                         # the start's own loading follows sd.cpp's
    progress.on_step(3, 10, 0.5)
    assert progress.to_dict()["fraction"] == 0.3 and progress.to_dict()["message"] == "Loading Qwen-Image 2.1"


def test_a_request_switches_the_model_and_a_missing_one_is_said(root: Path, fake_pictures: list[Path]) -> None:
    """The route: a request for the model that is not loaded switches first (its seconds in the timings), the health
    says which is loaded and which are on disk, a model whose file is not there is a 400 in the artist's words (the
    full one may arrive while the launcher runs: it is looked for at every request), and a switch that leaves no
    model answers "restart"."""
    folder = root / "models"
    folder.mkdir()
    engine, lib = _two_model_engine(folder, TURBO_FILE)
    calls: list[str] = []

    def generate(req: server.GenerateRequest, seed: int, references: list[Any]) -> list[Any]:
        calls.append(f"{req.model} {engine.model}")
        return [FakePicture(req.transparent)]

    engine.generate = generate
    exits: list[int] = []
    client = TestClient(server.build_app(engine, server.GpuWorker(), engine.progress, root, {"quality": "Q4_K_M"},
                                         exit_later=lambda: exits.append(1)))
    health = client.get("/spellcast/health").json()
    assert health["model"] == "turbo" and health["models"] == {"turbo": True, "full": False, "flux3": False}
    r = client.post("/spellcast/generate", json=_request(root, model="full").model_dump())
    assert r.status_code == 400 and r.json() == {
        "detail": f"The full Qwen-Image 2.1 is not downloaded yet (models\\qwen-image\\{FULL_FILE}). Download it with "
                  "its button under Model on the Images page, or pick Turbo.", "kind": "input"}
    assert lib.made == [(101, TURBO_FILE)] and calls == []      # nothing switched, nothing drawn
    r = client.post("/spellcast/generate", json=_request(root).model_dump())
    assert r.status_code == 200 and "switch" not in r.json()["timings"] and calls == ["turbo turbo"]

    (folder / FULL_FILE).write_bytes(b"")                       # the Images page's download has finished
    assert client.get("/spellcast/health").json()["models"] == {"turbo": True, "full": True, "flux3": False}
    r = client.post("/spellcast/generate", json=_request(root, model="full").model_dump())
    assert r.status_code == 200 and isinstance(r.json()["timings"]["switch"], float)
    assert calls[-1] == "full full" and lib.freed == [101] and lib.made[-1] == (102, FULL_FILE)
    assert client.get("/spellcast/health").json()["model"] == "full"
    r = client.post("/spellcast/generate", json=_request(root, model="turbo").model_dump())
    assert r.status_code == 200 and calls[-1] == "turbo turbo" and lib.freed == [101, 102]

    (folder / TURBO_FILE).unlink()                              # the installer's Turbo gone (deleted by hand)
    r = client.post("/spellcast/generate", json=_request(root, model="turbo").model_dump())
    assert r.status_code == 400 and r.json()["detail"].startswith(
        f"Qwen-Image 2.1 Turbo's model file is missing (models\\qwen-image\\{TURBO_FILE}). Close Spellcast3D")
    (folder / TURBO_FILE).write_bytes(b"")
    lib.fail_for = {FULL_FILE, TURBO_FILE}                      # neither can be loaded any more
    r = client.post("/spellcast/generate", json=_request(root, model="full").model_dump())
    assert r.status_code == 500 and r.json()["kind"] == "restart" and exits == [1]
    assert client.get("/spellcast/health").json()["ready"] is False


def test_the_log_says_how_a_request_is_guided(root: Path, fake_pictures: list[Path],
                                              capsys: pytest.CaptureFixture[str]) -> None:
    """cfg_scale and negative_prompt are still taken (the bench compares recipes with them): with CFG 1.0 a negative
    prompt is accepted and not used, which the log says; any other CFG is guidance."""
    client = TestClient(server.build_app(FakeEngine(), server.GpuWorker(), server.Progress(), root, {}))
    for changes in ({}, {"negative_prompt": "blurry"}, {"cfg_scale": 6.0, "steps": 20, "negative_prompt": "blurry"}):
        r = client.post("/spellcast/generate", json=_request(root, seed=1, model="full", **changes).model_dump())
        assert r.status_code == 200, r.text
    # the Turbo: its own 8 steps, without guidance, whatever the request said (and the line says what it left out)
    for changes in ({}, {"cfg_scale": 6.0, "steps": 20, "negative_prompt": "blurry"}):
        r = client.post("/spellcast/generate", json=_request(root, seed=1, **changes).model_dump(exclude_unset=True))
        assert r.status_code == 200, r.text
    lines = [line for line in capsys.readouterr().out.splitlines() if "picture(s) 1024x1024" in line]
    assert len(lines) == 5 and "with the full Qwen-Image 2.1, 25 steps, no guidance, euler, seed 1, " in lines[0]
    assert ", 25 steps, no guidance (its negative prompt is not used), euler, " in lines[1]
    assert ", 20 steps, CFG 6, euler, " in lines[2] and lines[2].endswith("first-block cache off (by the card)")
    assert ("with Qwen-Image 2.1 Turbo, 8 steps (its fixed schedule), no guidance, euler, seed 1, " in lines[3]
            and lines[3].endswith("first-block cache off"))
    assert ("with Qwen-Image 2.1 Turbo, 8 steps (its fixed schedule; the request's 20 steps, CFG 6 not used), no "
            "guidance (its negative prompt is not used), euler, " in lines[4])


def test_the_prefix_cache_shortage_becomes_a_warning(root: Path, fake_pictures: list[Path]) -> None:
    progress = server.Progress()

    class Short(FakeEngine):
        def generate(self, req: server.GenerateRequest, seed: int, references: list[Any]) -> list[Any]:
            progress.on_log(sdcpp.SD_LOG_WARN, "Qwen Image 2.1: insufficient memory for prefix caching; retrying")
            return super().generate(req, seed, references)

    client = TestClient(server.build_app(Short(), server.GpuWorker(), progress, root, {}))
    r = client.post("/spellcast/generate", json=_request(root).model_dump())
    assert r.status_code == 200 and r.json()["warning"] == server.CACHE_DROPPED


def test_a_picture_that_cannot_be_saved_loses_only_itself(root: Path, fake_pictures: list[Path],
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """Each picture is saved on its own: one that cannot be (a full disk, a file another program holds) is left out
    with a warning, and the others come back; none saved is a failure."""
    real = server.save_picture

    def save(picture: FakePicture, out_dir: Path, seed: int, transparent: bool = True) -> Path:
        if seed == 4:
            raise OSError(28, "There is not enough space on the disk")
        return real(picture, out_dir, seed, transparent)

    monkeypatch.setattr(server, "save_picture", save)
    client = TestClient(server.build_app(FakeEngine(), server.GpuWorker(), server.Progress(), root, {}))
    r = client.post("/spellcast/generate", json=_request(root, seed=3, batch_count=3).model_dump())
    assert r.status_code == 200, r.text
    assert r.json()["seeds"] == [3, 5] and len(r.json()["images"]) == 2
    assert r.json()["warning"] == "1 of 3 pictures could not be saved (There is not enough space on the disk)."
    r = client.post("/spellcast/generate", json=_request(root, seed=4, batch_count=1).model_dump())
    assert r.status_code == 500 and r.json()["kind"] == "error" and "not enough space" in r.json()["detail"]


def test_saving_tries_again_while_another_program_holds_the_file(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """An antivirus or the search indexer may hold a new file for a moment: written straight to its name (nothing to
    rename in the job's own folder), and tried again on a sharing error."""
    monkeypatch.setattr(server, "has_transparency", lambda picture: True)
    monkeypatch.setattr(server.time, "sleep", lambda seconds: None)

    class Held:
        tries = 0

        def save(self, path: Path, format: str, compress_level: int = 6) -> None:
            assert format == "PNG" and compress_level == 3      # zlib level 3: ~60 ms a picture, the same size as 6
            Held.tries += 1
            if Held.tries < 3:
                raise PermissionError(32, "The process cannot access the file because it is being used")
            Path(path).write_bytes(b"png")

    path = server.save_picture(Held(), tmp_path / "out", 9)
    assert path.name == "qwen-9.png" and path.read_bytes() == b"png" and Held.tries == 3
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["qwen-9.png"]
    Held.tries = -10
    with pytest.raises(PermissionError):
        server.save_picture(Held(), tmp_path / "out", 10)


def test_seeds_are_bounded() -> None:
    """sd.cpp's seed is an int64: ctypes would wrap a bigger one silently (a negative result is random again)."""
    assert server.GenerateRequest(prompt="x", out_dir="C:/x", seed=server.MAX_SEED).seed == 2 ** 32 - 1
    for bad in (2 ** 32, 10 ** 20, -2):
        with pytest.raises(ValueError):
            server.GenerateRequest(prompt="x", out_dir="C:/x", seed=bad)


@pytest.mark.skipif(os.name != "nt", reason="Windows paths")
def test_model_paths_reach_sd_cpp_in_plain_letters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """stable-diffusion.cpp opens files with the ANSI code page in some places and as UTF-8 in others: a model path
    with other letters goes relative to the sdcpp folder (the current folder, in the same Spellcast3D folder), else as
    its 8.3 short form, else the launcher says plainly what to do."""
    plain = tmp_path / "models" / "qwen-image" / "x.gguf"
    assert server.c_path(plain) == str(plain).encode("utf-8")
    root = tmp_path / "Jos\u00e9" / "spellcast3d"
    model = root / "models" / "qwen-image" / "qwen-image-2.1-Q4_K_M.gguf"
    sdcpp_dir = root / "code" / "venvs" / "qwen" / "sdcpp"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"gguf")
    sdcpp_dir.mkdir(parents=True)
    monkeypatch.chdir(sdcpp_dir)
    rel = server.c_path(model)
    assert rel == b"..\\..\\..\\..\\models\\qwen-image\\qwen-image-2.1-Q4_K_M.gguf"
    assert Path(rel.decode("ascii")).resolve() == model.resolve()
    # on another drive than the models (no relative path): the short form, or the plain message
    monkeypatch.setattr(server.os.path, "relpath", lambda *a: (_ for _ in ()).throw(ValueError("another drive")))
    monkeypatch.setattr(server.ctypes.windll.kernel32, "GetShortPathNameW", lambda text, buf, size: 0)
    with pytest.raises(server.PathProblem) as e:
        server.c_path(model)
    assert "plain letters" in str(e.value) and server.startup_hint(e.value) == str(e.value)

    def short(text: str, buf: Any, size: int) -> int:
        buf.value = "C:\\JOSE~1\\SPELLC~1\\models\\QWEN-I~1\\QWEN-I~1.GGU"
        return len(buf.value)

    monkeypatch.setattr(server.ctypes.windll.kernel32, "GetShortPathNameW", short)
    assert server.c_path(model) == b"C:\\JOSE~1\\SPELLC~1\\models\\QWEN-I~1\\QWEN-I~1.GGU"


def test_the_other_cards_get_their_memory_back() -> None:
    """Measuring a card makes a CUDA context on it: every card but the one sd.cpp computes on is reset after."""
    calls: list[tuple[str, int]] = []

    class Runtime:
        current = 0

        def cudaSetDevice(self, index: int) -> int:
            calls.append(("set", index))
            Runtime.current = index
            return 0

        def cudaDeviceReset(self) -> int:
            calls.append(("reset", Runtime.current))
            return 0

    cuda = server.Cuda.__new__(server.Cuda)
    cuda.rt = Runtime()
    cuda.release_others(1, [{"index": 0}, {"index": 1}, {"index": 2}])
    assert calls == [("set", 0), ("reset", 0), ("set", 2), ("reset", 2), ("set", 1)]


def test_pictures_finished_before_a_cancel_are_kept(root: Path, fake_pictures: list[Path]) -> None:
    class Partly(FakeEngine):
        def generate(self, req: server.GenerateRequest, seed: int, references: list[Any]) -> list[Any]:
            return super().generate(req, seed, references)[:1]       # cancelled while finishing the second

    client = TestClient(server.build_app(Partly(), server.GpuWorker(), server.Progress(), root, {}))
    r = client.post("/spellcast/generate", json=_request(root, seed=3, batch_count=2).model_dump())
    assert r.status_code == 200 and r.json()["seeds"] == [3] and len(r.json()["images"]) == 1
    assert r.json()["warning"] == "Stopped after 1 of 2 pictures."


def test_a_running_request_can_be_cancelled_and_a_second_one_is_refused(root: Path, fake_pictures: list[Path]) -> None:
    """The hub never sends two at once: a second one means it gave up on the first (a timeout), which still runs.
    It is refused ("busy"), never queued behind the first on the one GPU thread. A cancel reaches the engine."""
    started, release = threading.Event(), threading.Event()

    class Slow(FakeEngine):
        def generate(self, req: server.GenerateRequest, seed: int, references: list[Any]) -> list[Any]:
            started.set()
            release.wait(10)
            if progress.cancel.is_set():
                raise server.Cancelled()
            return super().generate(req, seed, references)

    progress = server.Progress()
    engine = Slow()
    client = TestClient(server.build_app(engine, server.GpuWorker(), progress, root, {}))
    body = _request(root).model_dump()
    answers: list[Any] = []
    first = threading.Thread(target=lambda: answers.append(client.post("/spellcast/generate", json=body)))
    first.start()
    try:
        assert started.wait(10)
        r = client.post("/spellcast/generate", json=body)
        assert r.status_code == 409 and r.json()["kind"] == "busy"

        def cancel_like_the_engine() -> None:
            progress.cancel.set()
            engine.cancels += 1

        engine.cancel = cancel_like_the_engine          # type: ignore[method-assign]
        assert client.post("/spellcast/cancel", json={}).json() == {"cancelling": True}
    finally:
        release.set()
        first.join(10)
    assert engine.cancels == 1
    assert answers[0].status_code == 409 and answers[0].json()["kind"] == "cancelled"
    assert client.post("/spellcast/generate", json=body).status_code == 200      # free again, cancel forgotten


def test_broken_cuda_answers_restart_and_ends_the_process(root: Path, fake_pictures: list[Path]) -> None:
    """After an error that leaves CUDA unusable Qwen-Image is no longer ready, says "restart" (the hub starts it
    afresh) and exits; an input problem or a cancel never asks the GPU."""
    exits: list[bool] = []
    probes: list[bool] = []

    class Broken(FakeEngine):
        def generate(self, req: server.GenerateRequest, seed: int, references: list[Any]) -> list[Any]:
            if req.prompt == "impossible":
                raise server.InputProblem("Not like that.")
            raise server.GenerationFailed("[ERROR] CUDA error: an illegal memory access was encountered")

    engine = Broken()
    client = TestClient(server.build_app(engine, server.GpuWorker(), server.Progress(), root, {},
                                         cuda_works=lambda: probes.append(True) or False,
                                         exit_later=lambda: exits.append(True)))
    body = _request(root).model_dump()
    assert client.post("/spellcast/generate", json={**body, "prompt": "impossible"}).json()["kind"] == "input"
    assert probes == [] and exits == []
    r = client.post("/spellcast/generate", json=body)
    assert r.status_code == 500 and r.json()["kind"] == "restart" and "starts again" in r.json()["detail"]
    assert exits == [True] and engine.ready is False
    assert client.get("/spellcast/health").json()["ready"] is False
    again = client.post("/spellcast/generate", json=body)                    # until it is gone
    assert again.status_code == 503 and again.json()["kind"] == "restart"


def test_a_failure_with_working_cuda_keeps_the_tool(root: Path, fake_pictures: list[Path]) -> None:
    class Short(FakeEngine):
        def generate(self, req: server.GenerateRequest, seed: int, references: list[Any]) -> list[Any]:
            raise server.OutOfMemory("cudaMalloc failed: out of memory")

    exits: list[bool] = []
    engine = Short()
    client = TestClient(server.build_app(engine, server.GpuWorker(), server.Progress(), root, {},
                                         cuda_works=lambda: True, exit_later=lambda: exits.append(True)))
    r = client.post("/spellcast/generate", json=_request(root).model_dump())
    assert r.status_code == 500 and r.json()["kind"] == "oom" and exits == [] and engine.ready is True


def test_only_this_computer_reaches_the_tool(root: Path) -> None:
    from _loopback_guard import LoopbackGuard

    app = server.build_app(FakeEngine(), server.GpuWorker(), server.Progress(), root, {})
    client = TestClient(LoopbackGuard(app), base_url="http://127.0.0.1:7776")
    assert client.get("/spellcast/health").status_code == 200
    assert client.get("/spellcast/health", headers={"host": "evil.example:7776"}).status_code == 403
    assert client.post("/spellcast/cancel", json={}, headers={"origin": "http://evil.example"}).status_code == 403


# ---- pictures (Pillow) -------------------------------------------------------------------------------------------

@needs_pil
def test_example_pictures_keep_their_transparency(tmp_path: Path) -> None:
    from PIL import Image

    see_through = Image.new("RGBA", (64, 48), (200, 10, 10, 255))
    see_through.putpixel((0, 0), (0, 0, 0, 0))
    see_through.save(tmp_path / "a.png")
    Image.new("RGB", (40, 40), (1, 2, 3)).save(tmp_path / "b.png")      # RGB: opaque
    Image.new("RGB", (5000, 2500), (9, 9, 9)).save(tmp_path / "c.png")
    a = server.load_reference(tmp_path / "a.png")
    assert a.mode == "RGBA" and a.size == (64, 48) and a.getpixel((0, 0))[3] == 0 and server.has_transparency(a)
    b = server.load_reference(tmp_path / "b.png")
    assert b.mode == "RGBA" and b.getpixel((5, 5)) == (1, 2, 3, 255) and not server.has_transparency(b)
    c = server.load_reference(tmp_path / "c.png")
    assert c.size == (2048, 1024)                                   # shrunk, its shape kept
    (tmp_path / "d.png").write_bytes(b"not a picture")
    with pytest.raises(server.InputProblem, match="could not be read"):
        server.load_reference(tmp_path / "d.png")


@needs_pil
def test_pictures_are_saved_rgba_only_when_see_through(tmp_path: Path) -> None:
    from PIL import Image

    cut_out = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    cut_out.paste((255, 0, 0, 255), (8, 8, 24, 24))
    opaque = Image.new("RGBA", (32, 32), (10, 20, 30, 255))
    opaque.putpixel((3, 3), (10, 20, 30, 253))            # the model's alpha on an opaque picture: 253-255
    first = server.save_picture(cut_out, tmp_path / "out", 5)
    second = server.save_picture(opaque, tmp_path / "out", 5)          # the same seed again: a new name
    third = server.save_picture(cut_out, tmp_path / "out", 6, transparent=False)
    assert first.name == "qwen-5.png" and second.name == "qwen-5-1.png" and third.name == "qwen-6.png"
    with Image.open(first) as img:
        assert img.mode == "RGBA" and img.getpixel((0, 0))[3] == 0 and img.getpixel((16, 16)) == (255, 0, 0, 255)
    with Image.open(second) as img:          # no alpha at all: Pixal3D removes the background itself
        assert img.mode == "RGB" and img.getpixel((0, 0)) == (10, 20, 30)
    with Image.open(third) as img:           # no transparent background asked for: never an alpha channel
        assert img.mode == "RGB"
    assert not server.has_transparency(opaque) and server.has_transparency(cut_out)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["qwen-5-1.png", "qwen-5.png", "qwen-6.png"]


@needs_pil
def test_a_cut_out_nobody_asked_for_is_laid_on_white(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Without a transparent background asked for, the model still leaves the background see-through for some seeds,
    and what it leaves under it is a flat colour of its own (magenta on the user's sheet): dropping the alpha channel
    showed that colour. Such a picture is laid on white; an opaque one keeps the model's colours bit for bit, its
    stray half-opaque pixels included."""
    from PIL import Image

    magenta = (163, 65, 202)
    cut_out = Image.new("RGBA", (100, 100), (*magenta, 0))
    cut_out.paste((10, 200, 30, 255), (30, 30, 70, 70))
    cut_out.putpixel((30, 30), (10, 200, 30, 128))                    # an edge pixel, half through
    assert server.has_cutout(cut_out) and not server.has_cutout(cut_out.convert("RGB"))
    with Image.open(server.save_picture(cut_out, tmp_path, 2, transparent=False)) as img:
        assert img.mode == "RGB" and img.getpixel((5, 5)) == (255, 255, 255) and img.getpixel((50, 50)) == (10, 200, 30)
        assert img.getpixel((30, 30)) == (132, 227, 142), "blended with the white behind it"
    assert "laid on white" in capsys.readouterr().out
    with Image.open(server.save_picture(cut_out, tmp_path, 3, transparent=True)) as img:
        assert img.mode == "RGBA" and img.getpixel((5, 5)) == (*magenta, 0), "asked for: kept as the model drew it"
    # an opaque picture: 14 see-through pixels of a million and soft alpha elsewhere are the model's noise
    opaque = Image.new("RGBA", (1000, 1000), (40, 50, 60, 255))
    opaque.paste((40, 50, 60, 210), (0, 0, 300, 100))
    for x in range(14):
        opaque.putpixel((x, 500), (*magenta, 0))
    assert not server.has_cutout(opaque)
    with Image.open(server.save_picture(opaque, tmp_path, 4, transparent=False)) as img:
        assert img.mode == "RGB" and img.tobytes() == opaque.convert("RGB").tobytes()
        assert img.getpixel((0, 500)) == magenta and img.getpixel((5, 5)) == (40, 50, 60)
    assert "laid on white" not in capsys.readouterr().out
    opaque.paste((*magenta, 0), (0, 0, 40, 40))                      # 0.16 % of it: a part one would see
    assert server.has_cutout(opaque)
    assert server.opaque(opaque).getpixel((5, 5)) == (255, 255, 255)


# ---- the ctypes mirror -------------------------------------------------------------------------------------------

@pytest.mark.skipif(ctypes.sizeof(ctypes.c_void_p) != 8 or os.name != "nt", reason="Windows x64 layout")
def test_the_structs_have_the_pinned_sizes() -> None:
    # measured against stable-diffusion.dll of master-929-3f8527a and of Spellcast3D's builds (spellcast-perf-36f92d4
    # and later), whose public header is the same (their init functions fill exactly this much)
    assert ctypes.sizeof(sdcpp.SdCtxParams) == 320
    assert ctypes.sizeof(sdcpp.SdImgGenParams) == 552
    assert ctypes.sizeof(sdcpp.SdImage) == 24 and ctypes.sizeof(sdcpp.SdTilingParams) == 32
    assert sdcpp.SdCtxParams.conditioning_cache_size.offset == 316
    assert sdcpp.SdImgGenParams.image_preprocess.offset == 544


def test_the_dll_writing_outside_our_struct_is_caught() -> None:
    class Wide:
        """An init function of a DLL whose struct has 8 more bytes."""

        def __call__(self, pointer: Any) -> None:
            raw = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_ubyte * (ctypes.sizeof(sdcpp.SdImage) + 8)))
            ctypes.memset(raw, 0, ctypes.sizeof(sdcpp.SdImage) + 8)

    image, inside = sdcpp.init_struct(Wide(), sdcpp.SdImage)
    assert not inside and image.width == 0

    def fits(pointer: Any) -> None:
        pointer.contents.width = 7

    image, inside = sdcpp.init_struct(fits, sdcpp.SdImage)
    assert inside and image.width == 7 and image.channel == 0xA5A5A5A5    # an unset field shows the guard


@needs_dll
def test_the_pinned_dll_matches_the_mirror() -> None:
    cwd = os.getcwd()
    lib = sdcpp.load(SDCPP.resolve())        # it makes the sdcpp folder the current one
    try:
        assert sdcpp.is_pinned(lib)
        sdcpp.check_layout(lib)                     # raises LayoutError with the fields that disagree
        assert lib.str_to_sample_method(b"euler") == sdcpp.EULER_SAMPLE_METHOD
        assert lib.str_to_sample_method(b"nonsense") == sdcpp.SAMPLE_METHOD_COUNT
    finally:
        os.chdir(cwd)


@needs_dll
def test_a_shifted_field_fails_the_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check really reads the DLL's answer: with two neighbouring bools of our mirror swapped, it refuses."""
    cwd = os.getcwd()
    lib = sdcpp.load(SDCPP.resolve())
    try:
        fields = list(sdcpp.SdCtxParams._fields_)
        a = [name for name, _ in fields].index("flash_attn")
        fields[a], fields[a + 1] = fields[a + 1], fields[a]

        class Swapped(ctypes.Structure):
            _fields_ = fields

        monkeypatch.setattr(sdcpp, "SdCtxParams", Swapped)
        lib.sd_ctx_params_init.argtypes = [ctypes.c_void_p]
        lib.sd_ctx_params_to_str.argtypes = [ctypes.c_void_p]
        with pytest.raises(sdcpp.LayoutError, match="flash_attn"):
            sdcpp.check_layout(lib)
    finally:
        sdcpp.declare(lib)
        os.chdir(cwd)


def test_failures_to_load_the_runtime_are_explained() -> None:
    assert "not the version" in sdcpp.describe_failure(sdcpp.LayoutError("width is 1"))
    assert "Visual C++" in sdcpp.describe_failure(OSError("[WinError 126] The specified module could not be found"))
    assert "run-spellcast3d.bat" in server.startup_hint(sdcpp.LayoutError("x"))
    assert "out of memory" in server.startup_hint(server.OutOfMemory("x"))


def test_the_runtime_check_names_missing_dlls(tmp_path: Path) -> None:
    assert sdcpp.missing_files(tmp_path)[0] == sdcpp.DLL_NAME
    for name in sdcpp.missing_files(tmp_path):
        (tmp_path / name).write_bytes(b"")
    assert sdcpp.missing_files(tmp_path) == []


# ---- preview pictures ----------------------------------------------------------------------------------------------

def test_the_newest_preview_is_kept_while_a_picture_is_drawn(root: Path) -> None:
    progress = server.Progress()
    client = TestClient(server.build_app(FakeEngine(), server.GpuWorker(), progress, root, {}))
    progress.on_preview(1, b"early")                     # nothing is drawn: not kept
    assert "preview" not in client.get("/spellcast/progress").json()
    assert client.get("/spellcast/preview").status_code == 404
    progress.start(2, 20, references=0)
    _log(progress, "generating image: 1/2 - seed 41")
    assert "preview" not in progress.to_dict()
    progress.on_preview(1, b"one")
    progress.on_preview(2, b"two")
    assert client.get("/spellcast/progress").json()["preview"] == {"seq": 2, "step": 2, "image": 1}
    progress.box = [0.5, 0.25, 1.0, 0.75]           # a window's request: the part of the picture its previews are of
    assert client.get("/spellcast/progress").json()["preview"]["box"] == [0.5, 0.25, 1.0, 0.75]
    progress.box = None
    r = client.get("/spellcast/preview")
    assert r.status_code == 200 and r.content == b"two" and r.headers["content-type"] == "image/png"
    assert (r.headers["x-preview-seq"], r.headers["x-preview-step"]) == ("2", "2")
    _log(progress, "generating image: 2/2 - seed 42")    # the next picture: the first one's preview is not its own
    assert "preview" not in progress.to_dict()
    progress.on_preview(1, b"three")
    assert progress.to_dict()["preview"] == {"seq": 3, "step": 1, "image": 2}       # the numbers go on
    progress.stop()
    assert "preview" not in progress.to_dict() and client.get("/spellcast/preview").status_code == 404


def test_sd_cpp_previews_only_a_request_that_asks(monkeypatch: pytest.MonkeyPatch) -> None:
    """generate() gives sd.cpp the preview callback for a request with "preview" and takes it away afterwards; the
    frames it hands over are copied out and kept as the newest preview. The DLL is a fake that plays two steps."""
    registered: list[tuple[bool, int, int, bool, bool]] = []
    frames_seen: list[tuple[int, int, bool, int]] = []

    class Lib:
        callback: Any = None

        @staticmethod
        def str_to_sample_method(name: bytes) -> int:
            return sdcpp.EULER_SAMPLE_METHOD

        @staticmethod
        def sd_img_gen_params_init(pointer: Any) -> None:
            ctypes.memset(pointer, 0, ctypes.sizeof(sdcpp.SdImgGenParams))

        @staticmethod
        def sd_set_preview_callback(cb: Any, mode: int, interval: int, denoised: bool, noisy: bool, data: Any) -> None:
            registered.append((bool(cb), mode, interval, denoised, noisy))
            Lib.callback = cb if cb else None

        @staticmethod
        def generate_image(ctx: Any, params: Any, out: Any, count: Any) -> bool:
            if Lib.callback is not None:
                for step in (1, 2):
                    pixels = (ctypes.c_uint8 * (4 * 4 * 4))(*([step] * 64))
                    frame = (sdcpp.SdImage * 1)(sdcpp.SdImage(4, 4, 4, ctypes.cast(pixels, ctypes.POINTER(
                        ctypes.c_uint8))))
                    Lib.callback(step, 1, frame, False, None)
                Lib.callback(3, 1, frame, True, None)       # the noisy input: never shown
                Lib.callback(3, 0, None, False, None)       # no frame: nothing to show, and nothing raised
            return False                                    # no picture: generate() then raises

    def encode(raw: bytes, width: int, height: int, transparent: bool) -> bytes:
        frames_seen.append((width, height, transparent, raw[0]))
        return b"png-%d" % raw[0]

    pil = type(sys)("PIL")
    pil.Image = object()
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setattr(server, "encode_preview", encode)
    engine = server.QwenEngine.__new__(server.QwenEngine)
    engine.lib, engine.ctx, engine.progress = Lib(), 1, server.Progress()
    engine._preview_cb = sdcpp.PREVIEW_CALLBACK(engine._on_preview)
    engine.placement = server.choose_placement(11 * GiB, 10_664)
    engine.progress.start(1, 2, references=0)
    _log(engine.progress, "generating image: 1/1 - seed 7")
    with pytest.raises(server.GenerationFailed):
        engine.generate(server.GenerateRequest(prompt="a chest", out_dir="C:/x"), 7, [])
    assert registered == [] and engine.progress.newest_preview() is None            # not asked for: sd.cpp untouched
    with pytest.raises(server.GenerationFailed):
        engine.generate(server.GenerateRequest(prompt="a chest", out_dir="C:/x", preview=True, transparent=False),
                        7, [])
    assert registered == [(True, sdcpp.PREVIEW_PROJ, 1, True, False), (False, sdcpp.PREVIEW_NONE, 1, True, False)]
    assert frames_seen == [(4, 4, False, 1), (4, 4, False, 2)]
    assert engine.progress.newest_preview() == (2, 2, 1, b"png-2")


@needs_pil
def test_a_preview_frame_becomes_a_small_png() -> None:
    from PIL import Image

    raw = bytes([200, 10, 10, 255] * 8 * 8)
    for transparent, mode in ((True, "RGBA"), (False, "RGB")):
        png = server.encode_preview(raw, 8, 8, transparent)
        with Image.open(io.BytesIO(png)) as picture:
            assert picture.format == "PNG" and picture.mode == mode and picture.size == (8, 8)
            assert picture.getpixel((0, 0))[:3] == (200, 10, 10)
    # a see-through frame nobody asked for shows white, as the finished picture will (never the colour under it)
    through = bytes([200, 10, 10, 0] * 8 * 8)
    with Image.open(io.BytesIO(server.encode_preview(through, 8, 8, False))) as picture:
        assert picture.mode == "RGB" and picture.getpixel((0, 0)) == (255, 255, 255)
    with Image.open(io.BytesIO(server.encode_preview(through, 8, 8, True))) as picture:
        assert picture.getpixel((0, 0)) == (200, 10, 10, 0)
    with Image.open(io.BytesIO(server.encode_preview(bytes([1, 2, 3] * 4), 2, 2, True))) as picture:
        assert picture.mode == "RGB"                    # a frame without alpha stays without


WINDOWS = [     # what was brushed, the picture's size -> the window; the page pins the same (tests/web/mask_window.test.mjs)
    ((1454, 1454, 1823, 1823), (2048, 2048), (1120, 1120, 2048, 2048)),     # the room of a 2K picture round it
    ((200, 1000, 1700, 1040), (2048, 2048), (0, 764, 1952, 1276)),          # long: thinner
    ((500, 500, 1400, 1400), (2048, 2048), None),                           # too large
    ((624, 624, 1424, 1424), (2048, 2048), (512, 512, 1536, 1536)),
    ((10, 10, 60, 60), (2048, 2048), (0, 0, 768, 768)),                     # at a corner: moved inside
    ((18, 700, 1518, 740), (1536, 1536), (0, 384, 1536, 1056)),
    ((400, 400, 600, 600), (1024, 1024), (116, 116, 884, 884)),
    ((162, 162, 862, 862), (1024, 1024), (0, 0, 1024, 1024)),               # the whole picture
    ((100, 500, 400, 800), (832, 1216), (0, 266, 768, 1034)),
    ((0, 0, 2048, 300), (2048, 2048), (0, 0, 2048, 512)),
    ((0, 900, 2048, 1230), (2048, 2048), None),
    ((900, 100, 1000, 1900), (2048, 2048), (694, 0, 1206, 2048)),           # tall: narrower
    ((600, 600, 937, 937), (1536, 1536), (368, 368, 1168, 1168)),
    ((1990, 1000, 2040, 1050), (2048, 2048), (1280, 641, 2048, 1409)),
    ((300, 300, 1132, 1132), (2048, 2048), (204, 204, 1228, 1228)),         # the largest square that fits
    ((300, 300, 1133, 1133), (2048, 2048), None),
    ((700, 700, 1100, 1100), (1536, 1536), (484, 484, 1316, 1316)),
]


@needs_pil
def test_a_held_change_is_drawn_in_a_window_of_its_picture(root: Path) -> None:
    """A held masked change of a picture larger than its window (window_of): the window is cut out at the picture's
    own size and asked for as a picture of its own - the window is all the model is shown -; the drawing is held to
    the picture outside the room and along the window's sides; what comes back is laid into the picture, whose
    every other pixel is its own. The steps' previews are of the window: the progress says where it lies. A mask
    too large for a window is drawn in the whole picture, as before. And what the model was given is kept for the
    artist to look at (given_dir)."""
    from PIL import Image, ImageChops, ImageDraw

    for brushed, size, window in WINDOWS:
        assert server.window_of(brushed, size) == window, (brushed, size)
        if window:
            width, height = window[2] - window[0], window[3] - window[1]
            assert width % 32 == 0 == height % 32 and width * height <= server.WINDOW_PIXELS
            assert 0 <= window[0] <= brushed[0] - 96 or window[0] == 0, "what was brushed lies inside, with its buffer"
            assert window[2] <= size[0] and window[3] <= size[1]
    rim = server.within_rim((256, 0, 1024, 768), (2048, 768))
    assert [rim.getpixel((x, 384)) for x in (0, 31, 32, 735, 736, 767)] == [0, 0, 255, 255, 0, 0], "held along both sides ..."
    assert rim.getpixel((400, 0)) == 255 == rim.getpixel((400, 767)), "... but not where the window's side is the picture's"
    corner = server.within_rim((0, 0, 768, 768), (2048, 768))
    assert [corner.getpixel((x, 384)) for x in (0, 735, 736)] == [255, 255, 0]

    size = (2048, 2048)
    base, brush, kept, where = _brushed_picture(root, size, (1454, 1454, 1822, 1822))
    held = {"changed_image": str(kept), "changed_mask": str(where), "init_image": str(kept), "mask_image": str(where),
            "strength": 1.0, "ref_images": [str(kept)], "width": 2048, "height": 2048, "prompt": "add an apple"}
    req = _request(root, preview=True, **held)
    keep = server.kept_from(req, server.change_of(req.changed_image, req.changed_mask, size))
    box = (1120, 1120, 2048, 2048)
    assert (keep.at, keep.rest.size, keep.whole.size, keep.moved) == (box[:2], (928, 928), size, (0, 0))
    assert _same(keep.rest, base.crop(box), (0, 0, 928, 928)) and keep.brush.getbbox() == (334, 334, 703, 703)
    room = keep.room.getbbox()              # 12 % of the WHOLE picture's side round the brush: 248 px
    assert 80 <= room[0] <= 96 and 80 <= room[1] <= 96 and room[2:] == (928, 928), "to the picture's own sides, and a rim inside"

    drawn, shown, start, mask = server.held_to(keep, req, [base.convert("RGBA")], base)
    assert (drawn.width, drawn.height, drawn.preview, drawn.ref_sides, drawn.prompt) == (928, 928, True, None, "add an apple")
    assert [p.size for p in shown] == [(928, 928)] and _same(shown[0], base.crop(box), (0, 0, 928, 928)), "the window alone"
    assert _same(start, base.crop(box), (0, 0, 928, 928)) and mask.tobytes() == keep.room.tobytes()
    assert server.window_place(keep) == [1120, 1120, 2048, 2048]
    assert server.fractions([1120, 1120, 2048, 2048], size) == [0.54688, 0.54688, 1.0, 1.0]
    assert req.width == 2048 and req.prompt == "add an apple", "the request itself is as it was"
    # an example picture of half the picture's size is cut at half the window; one that is not the start picture
    # stays whole
    half = base.resize((1024, 1024)).convert("RGBA")
    assert server.held_to(keep, req, [half], base)[1][0].size == (464, 464)
    other = req.model_copy(update={"ref_images": [str(where)]})
    assert [p.size for p in server.held_to(keep, other, [half], base)[1]] == [(1024, 1024)]

    class Drawing(_Redrawing):
        """Draws a gold box under the brush of the picture it is held to, and a grain all over the room."""

        def generate(self, req: server.GenerateRequest, seed: int, references: list[Any], **start: Any) -> list[Any]:
            self.asked.append(req)
            self.starts.append(start)
            self.shown.append(references)
            self.previews.append(progress.box)
            picture = _redrawn(start["init"].convert("RGB"), start["mask"])
            left, top, right, bottom = start["mask"].getbbox()
            ImageDraw.Draw(picture).rectangle(((left + right) // 2 - 90, (top + bottom) // 2 - 90,
                                               (left + right) // 2 + 90, (top + bottom) // 2 + 90), fill=GOLD)
            return [picture]

    # Through the launcher it is a TILEABLE picture's change (any other is drawn whole): moved first, the brush to its
    # middle, so its window reaches round the picture's edge as the page shows it
    held = {**held, "tileable": True}
    there = server.kept_from(_request(root, **held), server.change_of(str(kept), str(where), size))
    place, room = server.window_place(there), there.room.getbbox()
    assert (there.rest.size, place) == ((928, 928), [1174, 1174, 2102, 2102])
    engine, progress = Drawing(), server.Progress()
    engine.asked, engine.previews = [], []
    client = TestClient(server.build_app(engine, server.GpuWorker(), progress, root, {}))
    record = root / "p1" / "images" / ".incoming" / "given"
    r = client.post("/spellcast/generate", json=_request(root, seed=5, transparent=False, given_dir=str(record),
                                                         **held).model_dump())
    assert r.status_code == 200 and r.json()["warning"] is None, r.text
    with Image.open(r.json()["images"][0]) as saved:
        picture = saved.convert("RGB")
    asked = engine.asked[-1]
    assert (asked.width, asked.height, len(engine.shown[-1])) == (928, 928, 1), "the window, and nothing beside it"
    assert engine.previews[-1] == server.fractions(place, size) == [0.57324, 0.57324, 1.02637, 1.02637], \
        "where a step's picture lies in the artist's"
    # ... and the record: the words, the window as the model got it, what the drawing was held to, its own answer
    given = json.loads((record / "given.json").read_text(encoding="utf-8"))
    assert {k: given[k] for k in ("words", "model", "seed", "size", "picture", "way", "window", "start", "strength", "mask",
                                  "answers")} == {
        "words": "add an apple", "model": "turbo", "seed": 5, "size": [928, 928], "picture": [2048, 2048], "way": "window",
        "window": place, "start": "start.webp", "strength": 1.0, "mask": "mask.png",
        "answers": ["answer-1.webp"]}
    assert given["pictures"] == [{"file": "picture-1.webp", "seen": None, "is": "window"}]
    assert given["noise_seed"] != 5 and given["steps"] == 8
    with Image.open(record / "picture-1.webp") as shown_window, Image.open(record / "mask.png") as room_kept, \
            Image.open(record / "answer-1.webp") as answer:
        assert (shown_window.size, room_kept.size, room_kept.mode, answer.size) == ((928, 928), (928, 928), "L", (928, 928))
        assert _near(answer.convert("RGB").getpixel((464, 464)), GOLD, 12), "what the model drew, before it was laid back"
    assert picture.size == size and _near(picture.getpixel((1638, 1638)), GOLD, 12), "laid back where it was cut out"
    outside = Image.new("L", size, 255)
    for across in (0, -size[0]):                    # the room, also where it comes back at the picture's other side
        for down in (0, -size[1]):
            outside.paste(0, (place[0] + room[0] + across, place[1] + room[1] + down,
                              place[0] + room[2] + across, place[1] + room[3] + down))
    assert ImageChops.multiply(ImageChops.difference(picture, base).convert("L"), outside).getbbox() is None, \
        "outside the room - the rest of the window too - the picture's own pixels"
    # the largest mask a window takes leaves its room less than it would reach: held along the window's sides
    most = Image.new("L", size, 0)
    ImageDraw.Draw(most).rectangle((300, 300, 1131, 1131), fill=255)
    tight = server.kept_from(req, (base.convert("RGBA"), most))
    assert (tight.at, tight.rest.size, tight.brush.getbbox(), tight.room.getbbox()) == (
        (204, 204), (1024, 1024), (96, 96, 928, 928), (32, 32, 992, 992))
    # a mask too large for a window: the whole picture, as a picture of its own size
    wide = root / "p1" / "viewport" / "wide.png"
    large = Image.new("L", size, 0)
    ImageDraw.Draw(large).rectangle((500, 500, 1399, 1399), fill=255)
    large.save(wide)
    whole = {**held, "changed_mask": str(wide), "mask_image": str(wide)}
    r = client.post("/spellcast/generate", json=_request(root, seed=5, transparent=False, **whole).model_dump())
    assert r.status_code == 200 and (engine.asked[-1].width, engine.asked[-1].prompt, len(engine.shown[-1])) == (2048, "add an apple", 1)
    assert engine.starts[-1]["init"].size == size and engine.previews[-1] is None, "a step's picture is the whole one's"


# (what was brushed, the picture's size, its patch): the page's mask_window.test.mjs holds the same cases
PATCHES = [
    ((900, 900, 1100, 1100), (2048, 2048), (488, 488, 1512, 1512)),         # round what was brushed, 1024 a side
    ((1454, 1454, 1823, 1823), (2048, 2048), (955, 955, 1979, 1979)),       # brushed 225 px from two sides: 69 off them
    ((1900, 1000, 2040, 1100), (2048, 2048), (1024, 538, 2048, 1562)),      # 8 px from a side: as good as on it
    ((2030, 1000, 2048, 1100), (2048, 2048), (1024, 538, 2048, 1562)),      # brushed at a side: on it
    ((20, 20, 200, 200), (2048, 2048), (1, 1, 1025, 1025)),
    ((300, 300, 500, 500), (2048, 2048), (81, 81, 1105, 1105)),
    ((180, 180, 620, 260), (2048, 1536), (59, 54, 1083, 1078)),
    ((400, 400, 1600, 1500), (2048, 2048), (154, 154, 1846, 1746)),         # a large mask: its room each way
    ((200, 300, 1900, 1700), (2048, 2048), (64, 81, 1999, 1963)),           # ... as far as stays off the sides
    ((700, 700, 836, 836), (1536, 1536), (256, 256, 1280, 1280)),
    ((1300, 40, 1500, 90), (1536, 1152), (506, 10, 1530, 1034)),
    ((100, 100, 300, 300), (1024, 1024), (0, 0, 1024, 1024)),               # up to 1024 x 1024 pixels: all of it
    ((100, 100, 300, 300), (1024, 768), (0, 0, 1024, 768)),
    ((300, 500, 500, 700), (832, 1216), (0, 0, 832, 1216)),                 # 1K, not square: seen at its own size too
    ((100, 100, 300, 300), (1216, 832), (0, 0, 1216, 832)),
    ((300, 500, 500, 700), (1248, 1824), (56, 88, 1080, 1112)),
    ((300, 200, 1500, 900), (2048, 1536), (81, 57, 1773, 1249)),            # a large mask in a picture that is not square
    ((150, 900, 1650, 1400), (1824, 1248), (49, 179, 1768, 1203)),
    ((901, 900, 1100, 1100), (2048, 2048), (488, 488, 1512, 1512)),         # its middle between two pixels: the lower
]


@needs_pil
def test_a_mask_is_spread_in_steps_and_a_line_is_laid_through_most_points() -> None:
    """spread: the square one wide rank filter gives, whatever the reach. line_through: the line most points lie
    along - one wrong point changes nothing -, level where all lie at one place."""
    from PIL import Image, ImageDraw, ImageFilter

    mask = Image.new("L", (96, 80), 0)
    ImageDraw.Draw(mask).ellipse((40, 30, 52, 44), fill=255)
    ImageDraw.Draw(mask).point((3, 70), fill=255)
    for reach in (1, 2, 5, 8, 31):
        assert server.spread(mask, reach).tobytes() == mask.filter(ImageFilter.MaxFilter(2 * reach + 1)).tobytes(), reach
    at_zero, slope = server.line_through([(0, 1.0), (100, 2.0), (200, 3.0), (300, 4.0), (50, 40.0)])
    assert abs(at_zero - 1.0) < 1e-9 and abs(slope - 0.01) < 1e-9
    assert server.line_through([(5, 2.0), (5, 4.0)]) == (3.0, 0.0) and server.line_through([(7, -1.5)]) == (-1.5, 0.0)
    assert [box for box in server.along_the_sides((100, 0, 500, 300), 100, (600, 300))] == [
        (100, 0, 200, 50), (200, 0, 300, 50), (300, 0, 400, 50), (400, 0, 500, 50),
        (100, 250, 200, 300), (200, 250, 300, 300), (300, 250, 400, 300), (400, 250, 500, 300),
        (50, 0, 150, 100), (50, 100, 150, 200), (50, 200, 150, 300), (450, 0, 550, 100), (450, 100, 550, 200), (450, 200, 550, 300)]


@needs_pil
def test_of_a_large_pictures_whole_redraw_a_patch_is_laid_onto_the_picture(root: Path) -> None:
    """A masked change of a picture that is not tileable is drawn whole at every size. A large picture's redraw lies
    some pixels off its picture and stretched a little, in another tone, its finest detail another drawing's: the
    patch round what was brushed is kept of it (patch_of) - in the picture's tone, brought onto it (fitted), joined
    along a ragged border (joined) - and everywhere else the picture's own pixels are. The patch stays off the
    picture's sides and nears one without a jump as what was brushed does. The record says which part."""
    from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, ImageStat

    for brushed, size, patch in PATCHES:
        assert server.patch_of(brushed, size) == patch, (brushed, size)
    assert server.patch_of(None, (2048, 1536)) == (0, 0, 2048, 1536), "nothing brushed: all of it"
    # A mask brought nearer the picture's side, px by px: the patch's side follows, never further in one step than
    # the mask went, never more than a third of the way from the side to the mask, and on the side only with it.
    lefts = [server.patch_of((low, 900, low + 200, 1100), (2048, 2048))[0] for low in range(700, -1, -1)]
    assert lefts[0] == 288 and lefts[-1] == 0 and all(0 <= was - now <= 1 for was, now in zip(lefts, lefts[1:]))
    assert all(3 * left <= low + 2 for left, low in zip(lefts, range(700, -1, -1)) if low < 400)
    assert [lefts[700 - low] for low in (400, 300, 200, 100, 50, 20, 5)] == [89, 81, 64, 30, 9, 1, 0]
    rights = [server.patch_of((1848 - gap, 900, 2048 - gap, 1100), (2048, 2048))[2] for gap in range(700, -1, -1)]
    assert [2048 - right for right in rights] == lefts, "the same from the far side"

    # The join: all of the patch but along its sides inside the picture, where over the band (4 % of the picture's
    # side: 82 px) each pixel is one picture's or the other's - more the patch's the further in -, the same for a seed.
    size, patch, under = (2048, 2048), (488, 488, 1512, 1512), (900, 900, 1100, 1100)
    where = server.joined(patch, under, size, 5)
    assert (where.mode, where.size) == ("L", size) and where.tobytes() == server.joined(patch, under, size, 5).tobytes()
    assert where.tobytes() != server.joined(patch, under, size, 6).tobytes(), "another seed, another line"
    assert where.crop((488 + 84, 488 + 84, 1512 - 84, 1512 - 84)).getextrema() == (255, 255), "past the band: the patch"
    outside = where.copy()
    outside.paste(0, (486, 486, 1514, 1514))
    assert outside.getbbox() is None, "outside the patch: the picture"
    thirds = {"left": lambda low: (488 + low, 600, 488 + low + 27, 1400), "right": lambda low: (1512 - low - 27, 600, 1512 - low, 1400),
              "top": lambda low: (600, 488 + low, 1400, 488 + low + 27), "bottom": lambda low: (600, 1512 - low - 27, 1400, 1512 - low)}
    for side, third in thirds.items():              # the band's outer, middle and inner third, at each side
        shares = [ImageStat.Stat(where.crop(third(low)).point(lambda v: 255 if v > 127 else 0)).mean[0] / 255 for low in (0, 27, 54)]
        assert shares[0] < 0.25 < shares[1] < 0.75 < shares[2], (side, shares)
    solid = ImageStat.Stat(where.crop((488, 600, 570, 1400)).point(lambda v: 255 if v in (0, 255) else 0)).mean[0] / 255
    assert solid > 0.8, "a threshold, not a fade: a pixel is one picture's or the other's"
    at_side = server.joined((0, 488, 1024, 1512), (40, 900, 240, 1100), size, 5)
    assert at_side.crop((0, 600, 40, 1400)).getextrema() == (255, 255), "a side that is the picture's own has no join"
    assert server.joined((0, 0, *size), under, size, 5).getextrema() == (255, 255)
    # Near a side of the picture the patch's side comes close to what was brushed: its band is no wider than the way
    # between the two, so what was brushed is all the patch's.
    close = (60, 900, 260, 1100)
    near = server.patch_of(close, size)
    tight = server.joined(near, close, size, 5)
    assert near[0] == 13 and tight.crop(close).getextrema() == (255, 255), "the new thing is not mixed with what was there"
    assert tight.crop((0, 600, 12, 1400)).getbbox() is None and tight.crop((13, 600, 59, 1400)).getextrema() == (0, 255)
    far_close = (1788, 1788, 1988, 1988)            # the same at the picture's right side and its bottom
    far_near = server.patch_of(far_close, size)
    assert far_near[2:] == (2035, 2035) and server.joined(far_near, far_close, size, 5).crop(far_close).getextrema() == (255, 255)

    # The model's picture: the picture drawn anew (its grain, 8 levels darker), a gold thing under the brush - and all
    # of it 4 px left and 1 px down of the picture.
    base, brush, kept, mask = _brushed_picture(root, size, (900, 900, 1100, 1100))
    rest = base.convert("RGBA")
    made = _redrawn(base)
    ImageDraw.Draw(made).rectangle((940, 940, 1060, 1060), fill=GOLD)
    made = ImageChops.offset(made, -4, 1)
    out, changed = server.redrawn_whole(made, rest, brush, patch, 5)
    assert changed and (out.size, out.mode) == (size, "RGBA")
    assert _near(out.getpixel((1000, 1000)), (250, 220, 40), 6), "the new thing, where it was brushed, in the picture's tone"
    for far in ((0, 0, 2048, 486), (0, 1514, 2048, 2048), (0, 0, 486, 2048), (1514, 0, 2048, 2048)):
        assert _same(out, base, far), "outside the patch: the picture itself, to the pixel"
    inside = [out.getpixel((x, 700))[1] - base.getpixel((x, 700))[1] for x in range(600, 1400)]
    assert {abs(d) for d in inside} >= {11, 12, 13} and max(abs(d) for d in inside) <= 20, "the model's grain, on its picture"
    assert abs(sum(inside)) <= 2 * len(inside), "in the picture's tone"
    band = [out.getpixel((x, 1000))[1] - base.getpixel((x, 1000))[1] for x in range(490, 566)]
    assert 0 in band and any(abs(d) >= 11 for d in band), "along the join: some pixels the picture's, some the patch's"
    # Not brought onto it, a picture of noise would lie on its picture nowhere:
    unmoved = Image.composite(made.convert("RGBA"), rest, where)
    assert max(abs(unmoved.getpixel((x, 700))[1] - base.getpixel((x, 700))[1]) for x in range(600, 1400)) > 60
    # A thing the model drew larger than its mask, past the patch's side where that lies close to the mask (near the
    # picture's own): kept whole - it is what was asked for -, and beside it the picture's own pixels as ever.
    near_side = Image.new("L", size, 0)
    ImageDraw.Draw(near_side).rectangle((60, 900, 259, 1099), fill=255)
    large = _redrawn(base)
    ImageDraw.Draw(large).rectangle((4, 940, 300, 1060), fill=GOLD)
    whole_thing = server.redrawn_whole(large, rest, near_side, near, 5)[0]
    assert _near(whole_thing.getpixel((7, 1000)), (250, 220, 40), 8), "past the patch's side, 13 px in: the thing still"
    assert _same(whole_thing, base, (0, 0, 12, 800)) and _same(whole_thing, base, (0, 1200, 12, 2048))
    # ... and so is a shadow longer than the patch, far beyond the brush's room.
    shadowed = _redrawn(base)
    ImageDraw.Draw(shadowed).rectangle((940, 940, 1060, 1060), fill=GOLD)
    ImageDraw.Draw(shadowed).rectangle((1061, 990, 1900, 1010), fill=(20, 20, 20))
    longer = server.redrawn_whole(ImageChops.offset(shadowed, -4, 1), rest, brush, patch, 5)[0]
    assert _near(longer.getpixel((1800, 1000)), (28, 28, 28), 8) and _same(longer, base, (1514, 0, 2048, 900))
    # A thing drawn to the picture's very side is the redraw's to the last column: where the redraw, brought onto its
    # picture, ends short of the side, its own last pixels stand.
    at_edge = Image.new("L", size, 0)
    ImageDraw.Draw(at_edge).rectangle((0, 900, 199, 1099), fill=255)
    to_the_side = _redrawn(base)
    ImageDraw.Draw(to_the_side).rectangle((0, 940, 160, 1060), fill=GOLD)
    edged = server.redrawn_whole(ImageChops.offset(to_the_side, -4, 1), rest, at_edge, server.patch_of(at_edge.getbbox(), size), 5)[0]
    assert all(_near(edged.getpixel((x, 1000)), (250, 220, 40), 8) for x in range(8))
    # A redraw far off its picture's tone - 30 levels, more than counts as a change: the tone is brought back before
    # what changed is told, so the ground is not taken for changed, and outside the patch the picture stays.
    dark = ImageChops.subtract(made, Image.new("RGB", size, (22, 22, 22)))
    back, changed = server.redrawn_whole(dark, rest, brush, patch, 5)
    assert changed and _near(back.getpixel((1000, 1000)), (250, 220, 40), 8) and _same(back, base, (0, 0, 2048, 486))
    # A thing that stands out little is told all the same: from as far as the model's own pixels go, not the
    # picture's - 30 levels lighter over 120 px, in a picture of 2048.
    faint = _redrawn(base)
    faint.paste(ImageChops.add(faint, Image.new("RGB", size, (30, 30, 30))).crop((940, 940, 1060, 1060)), (940, 940))
    assert server.redrawn_whole(faint, rest, brush, patch, 5)[1] is True

    # A REDRAW DOES NOT LIE ON ITS PICTURE BY ONE MOVE: it is stretched a little too. Here 2 px less far left at the
    # patch's left side than at its right one, and 1.5 px further down at its top than at its bottom.
    soft = ImageOps.autocontrast(Image.effect_noise(size, 60).filter(ImageFilter.GaussianBlur(3))).convert("RGBA")
    grey, nothing = soft.convert("L"), Image.new("L", size, 0)
    sides = {"left": (406, 900, 570, 1064), "right": (1430, 900, 1594, 1064), "top": (900, 406, 1064, 570), "bottom": (900, 1430, 1064, 1594)}

    def off(picture: Any, box: tuple[int, int, int, int]) -> tuple[float, float]:
        return server.place_in(grey, picture.convert("L"), box, nothing)

    lies = _stretched(soft, -3.4, 0.002, 1.2, -0.0015)
    assert off(lies, sides["right"])[0] - off(lies, sides["left"])[0] > 1.8
    assert off(lies, sides["top"])[1] - off(lies, sides["bottom"])[1] > 1.3
    onto = server.fitted(lies, soft, patch, nothing)
    assert all(abs(v) < 0.2 for box in sides.values() for v in off(onto, box)), "on its picture at each of the patch's sides"
    # The colour left under a see-through part comes through as it is (a tool that drops the alpha shows it; at an
    # alpha of 1, resampled as RGBA, it came out a garish magenta in a cut-out's live render).
    veiled = lies.copy()
    veiled.putalpha(1)
    under = server.fitted(veiled, soft, patch, nothing)
    assert under.getchannel("A").getextrema() == (1, 1) and under.convert("RGB").tobytes() == onto.convert("RGB").tobytes()
    # A box most of which changed tells no place: too little of it is left to tell by.
    mostly, some = Image.new("L", size, 0), Image.new("L", size, 0)
    mostly.paste(255, (406, 900, 570, 1010))
    some.paste(255, (406, 900, 570, 940))
    assert server.place_in(grey, lies.convert("L"), sides["left"], mostly) is None
    assert server.place_in(grey, lies.convert("L"), sides["left"], some) is not None
    # Nothing to join (the patch is the whole picture), or nothing left to tell its place by: as it is.
    assert server.fitted(lies, soft, (0, 0, *size), nothing) is lies and server.fitted(lies, soft, patch, Image.new("L", size, 255)) is lies
    # A ground that only shades off tells no place: a little lighter, it fits the better the further it is moved.
    shade = Image.linear_gradient("L").resize(size).rotate(90)
    assert server.place_in(shade, shade.point(lambda v: v + 4), sides["left"], nothing) is None
    # A LARGE MASK: its room leaves little of the picture, and what was brushed is another thing altogether - the
    # redraw's tone and its place are read all the same, on what did not change.
    wide = Image.new("L", size, 0)
    ImageDraw.Draw(wide).rectangle((449, 449, 1598, 1598), fill=255)
    large_patch = server.patch_of(wide.getbbox(), size)
    answer = _stretched(ImageChops.subtract(soft.convert("RGB"), Image.new("RGB", size, (8, 8, 8))).convert("RGBA"), -3.4, 0.002, 1.2, -0.0015)
    answer.paste((*GOLD, 255), wide.getbbox())
    redone, changed = server.redrawn_whole(answer, soft, wide, large_patch, 5)
    beside = (large_patch[0] + 84, 900, large_patch[0] + 234, 1050)     # past the join's band, not brushed: the redraw's
    assert changed and large_patch[0] > 150 and all(abs(v) < 0.3 for v in off(redone, beside)), "on its picture"
    assert abs(ImageStat.Stat(redone.crop(beside).convert("L")).mean[0] - ImageStat.Stat(soft.crop(beside).convert("L")).mean[0]) < 2, \
        "in its tone"

    class Whole(_Redrawing):
        def generate(self, req: server.GenerateRequest, seed: int, references: list[Any], **start: Any) -> list[Any]:
            self.asked = req
            self.previews = progress.box
            return super().generate(req, seed, references, **start)

    engine, progress = Whole(), server.Progress()
    engine.made = made
    engine.placement = {"budget_mb": 12 * GiB, "prefix_cache": "auto"}
    client = TestClient(server.build_app(engine, server.GpuWorker(), progress, root, {}))
    record = root / "p1" / "images" / ".incoming" / "given"
    asked = {"changed_image": str(kept), "changed_mask": str(mask), "init_image": str(kept), "mask_image": str(mask),
             "strength": 1.0, "ref_images": [str(kept)], "width": 2048, "height": 2048, "prompt": "add an apple"}
    r = client.post("/spellcast/generate", json=_request(root, seed=5, transparent=False, given_dir=str(record), **asked).model_dump())
    assert r.status_code == 200 and r.json()["warning"] is None, r.text
    assert (engine.asked.width, engine.asked.height, engine.starts[-1], engine.previews) == (2048, 2048, {}, None), \
        "all of the picture asked for, nothing held, a step's picture the whole one's"
    with Image.open(r.json()["images"][0]) as saved:
        assert saved.convert("RGB").tobytes() == out.convert("RGB").tobytes(), "the patch of it, laid onto the picture"
    given = json.loads((record / "given.json").read_text(encoding="utf-8"))
    assert (given["way"], given["patch"], given["size"], given["answers"]) == ("whole", [488, 488, 1512, 1512], [2048, 2048], ["answer-1.webp"])
    assert given["pictures"][0]["seen"][0] < 2048 and given["noise_seed"] == server.noise_seed(5, engine.shown[-1]), \
        "the engine's noise seed: told from the pictures as read, not from the copies the model sees"
    with Image.open(record / "answer-1.webp") as answer:
        assert answer.size == size, "the model's own picture: all of it"
    # A picture of up to 1024 x 1024 pixels: all of the redraw is kept, and the record names no part.
    small = _request(root, seed=5, transparent=False, given_dir=str(record), **{**asked, "width": 1024, "height": 1024})
    engine.made = made.resize((1024, 1024))
    assert client.post("/spellcast/generate", json=small.model_dump()).status_code == 200
    assert "patch" not in json.loads((record / "given.json").read_text(encoding="utf-8"))


@needs_pil
def test_what_the_model_was_given_is_kept_for_the_artist_to_look_at(root: Path) -> None:
    """given_dir: given.json and the pictures it names - of a picture from words alone, of one from example pictures
    (each with the size the model sees it at on this card), of a masked change drawn whole (the brushed picture,
    and the model's own answer before its tone was matched). Only inside the projects folder; a record that cannot
    be written is not the request's failure."""
    from PIL import Image

    base, _brush, kept, where = _brushed_picture(root)
    engine = _Redrawing()
    engine.made = _redrawn(base).resize((1024, 768))
    engine.placement = {"budget_mb": 10 * GiB, "prefix_cache": "auto"}
    client = TestClient(server.build_app(engine, server.GpuWorker(), server.Progress(), root, {}))

    def record_of(name: str, **changes: Any) -> dict[str, Any]:
        folder = root / "p1" / "given" / name
        body = _request(root, seed=5, transparent=False, width=1024, height=768, given_dir=str(folder), **changes).model_dump()
        assert client.post("/spellcast/generate", json=body).status_code == 200
        return json.loads((folder / "given.json").read_text(encoding="utf-8"))

    words = record_of("words", prompt="a lamp")
    assert (words["way"], words["pictures"], words["start"], words["mask"], words["answers"]) == ("words", [], None, None, [])
    assert words["words"] == "a lamp" and "noise_seed" not in words and "window" not in words and "strength" not in words
    examples = record_of("examples", ref_images=[str(kept)])
    assert examples["way"] == "examples" and examples["pictures"] == [{"file": "picture-1.webp", "seen": [1024, 768], "is": "example"}]
    with Image.open(root / "p1" / "given" / "examples" / "picture-1.webp") as seen:
        assert seen.size == (1024, 768), "as the model sees it: with the pixels of the picture asked for"
    whole = record_of("whole", changed_image=str(kept), changed_mask=str(where), init_image=str(kept), mask_image=str(where),
                      strength=1.0, ref_images=[str(kept)])
    assert (whole["way"], whole["pictures"][0]["is"], whole["start"], whole["mask"], whole["answers"]) == (
        "whole", "brushed", None, None, ["answer-1.webp"])
    viewport = record_of("start", init_image=str(kept), ref_images=[str(kept)], strength=0.6)
    assert (viewport["way"], viewport["pictures"][0]["is"], viewport["start"], viewport["strength"]) == ("start", "start", "start.webp", 0.6)
    outside = _request(root, given_dir="C:/elsewhere/given").model_dump()
    r = client.post("/spellcast/generate", json=outside)
    assert r.status_code == 400 and "record" in r.json()["detail"]
    blocked = root / "p1" / "given" / "blocked"
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_text("a file where the folder would be", encoding="utf-8")
    assert client.post("/spellcast/generate", json=_request(root, given_dir=str(blocked)).model_dump()).status_code == 200
