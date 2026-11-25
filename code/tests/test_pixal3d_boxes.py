"""The decode in boxes on spconv (Spellcast3D round O; code/pixal3d/pixal3d/models/sc_vaes/chunked.py and
modules/sparse/conv/{conv_spconv.py, products_floor.py}) on the CPU: the torch-products convolution of the cards
without spconv's GEMM kernels on a level's per-offset pair lists against spconv's Native layout through
indice_conv_torch (every order of its slots, the floor's padding, index_add_ against the atomic-free accumulation,
row subsets, the whole table's own calls unpadded), the floors read from the calibration record, the sweep's floor
rule and its shapes' roles, and which backend decodes in boxes (PIXAL3D_CHUNK_VOXELS, the record's exact flag, the
Linear windows following the floor on spconv only, the whole levels' own calls on spconv).

Most values are small integers in float32, so every sum is exact: those check which rows meet which weights. The
products are also compared on random fp32 / fp16 values, where only the same order of the additions (centre,
offsets 0..26, bias last) gives the same bits. The card's own bits are the GPU checks' business
(devtools/pixal3d-lowmem/round-o/boxes).

Runs in the pixal3d venv, with the card hidden and pytest from code/logs/profiling/pytools:
    set CUDA_VISIBLE_DEVICES=-1
    set PYTHONPATH=code\\logs\\profiling\\pytools
    code\\venvs\\pixal3d\\Scripts\\python.exe -m pytest -p no:cacheprovider code\\tests\\test_pixal3d_boxes.py -q
Skipped in the hub venv (no torch) and where spconv is not installed.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from spellcast import paths

torch = pytest.importorskip("torch")
pytest.importorskip("spconv.pytorch")
if not (paths.PIXAL3D_DIR / "pixal3d" / "modules" / "sparse" / "conv" / "products_floor.py").is_file():
    pytest.skip("the Pixal3D checkout has no decode in boxes on spconv", allow_module_level=True)

sys.path.insert(0, str(paths.PIXAL3D_DIR))
try:
    from pixal3d import calibration                                       # noqa: E402
    from pixal3d.models.sc_vaes import chunked                            # noqa: E402
    from pixal3d.models.sc_vaes import lean_encode                        # noqa: E402
    from pixal3d.models.sc_vaes import sparse_unet_vae as suv             # noqa: E402
    from pixal3d.modules import precision                                 # noqa: E402
    from pixal3d.modules import sparse as sp                              # noqa: E402
    from pixal3d.modules.sparse import config as sparse_config            # noqa: E402
    from pixal3d.modules.sparse.conv import config as conv_config         # noqa: E402
    from pixal3d.modules.sparse.conv import conv_spconv as cs             # noqa: E402
    from pixal3d.modules.sparse.conv import products_floor as pf          # noqa: E402
    from pixal3d.staged import parts                                      # noqa: E402
finally:
    sys.path.remove(str(paths.PIXAL3D_DIR))

KV = 27
DELTAS = [(a - 1, b - 1, c - 1) for a in range(3) for b in range(3) for c in range(3)]   # offset k's displacement


@pytest.fixture(autouse=True)
def _no_switches(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Every test starts from the record's and the decoder's defaults, whatever the developer's shell has set."""
    monkeypatch.delenv(pf.WINDOW_ENV, raising=False)
    monkeypatch.delenv(chunked.CHUNK_ENV, raising=False)
    monkeypatch.delenv("PIXAL3D_SPCONV_BOXES", raising=False)
    monkeypatch.setattr(cs, "BOXES", True)          # read at import: PIXAL3D_SPCONV_BOXES=off in the shell
    pf.forget()
    yield
    pf.forget()


def _level(n: int = 400, side: int = 9, seed: int = 0) -> torch.Tensor:
    """n distinct voxels [n, 3] in a side^3 grid, in scan order (as the decoders' tables are)."""
    g = torch.Generator().manual_seed(seed)
    flat = torch.randperm(side ** 3, generator=g)[:n].sort().values
    return torch.stack([flat // (side * side), flat // side % side, flat % side], 1)


def _pair_fwd(coords: torch.Tensor) -> torch.Tensor:
    """[KV, n] int32: the row of output row r's neighbour at offset k (coords[r] + DELTAS[k]), -1 where none - the
    layout of spconv's implicit-GEMM pair table."""
    where = {tuple(c): i for i, c in enumerate(coords.tolist())}
    out = torch.full((KV, coords.shape[0]), -1, dtype=torch.int32)
    for r, c in enumerate(coords.tolist()):
        for k, d in enumerate(DELTAS):
            out[k, r] = where.get((c[0] + d[0], c[1] + d[1], c[2] + d[2]), -1)
    return out


def _native(pair_fwd: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The same pairs in spconv's Native layout ([2, KV, n]: input rows, output rows; the counts of offsets
    0..KV/2 only, the far half's taken from its mirror), the slots of every offset in a random order (as the Native
    builder's atomics leave them)."""
    n = pair_fwd.shape[1]
    pairs = torch.full((2, KV, n), -1, dtype=torch.int32)
    counts = torch.zeros(KV, dtype=torch.int32)
    g = torch.Generator().manual_seed(seed)
    for k in range(KV):
        dest = (pair_fwd[k] >= 0).nonzero().squeeze(1)
        dest = dest[torch.randperm(dest.shape[0], generator=g)]
        pairs[0, k, :dest.shape[0]] = pair_fwd[k, dest]
        pairs[1, k, :dest.shape[0]] = dest.to(torch.int32)
        if k < KV // 2:
            counts[k] = dest.shape[0]
    for k in range(KV // 2 + 1, KV):                    # a subm's table is symmetric: the mirror's count is the same
        assert int((pair_fwd[k] >= 0).sum()) == int(counts[KV - k - 1])
    return pairs, counts


def _exact(shape: tuple, seed: int, span: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(-span, span + 1, shape, generator=g).to(torch.float32)


@pytest.fixture
def level() -> dict:
    coords = _level()
    fwd = _pair_fwd(coords)
    n, ci, co = coords.shape[0], 8, 5
    return {"coords": coords, "fwd": fwd, "n": n, "features": _exact((n, ci), 1),
            "filters": _exact((co, 3, 3, 3, ci), 2), "bias": _exact((co,), 3),
            "pairs": cs.LevelPairs(n, KV, offsets=cs.offset_lists(fwd))}


def test_the_products_on_the_level_pairs_are_indice_conv_torchs_sums(level: dict, monkeypatch: pytest.MonkeyPatch):
    """The per-offset lists made of the implicit-GEMM table give spconv's Native sums whatever order the Native
    builder left its slots in, with the floor's padding or without it, with index_add_ or the
    atomic-free accumulation; the centre offset has no list (its product covers every row)."""
    f, w, b = level["features"], level["filters"], level["bias"]
    assert level["pairs"].offsets[KV // 2] is None
    for k, pair in enumerate(level["pairs"].offsets):
        if pair is not None:
            dest, src = pair
            assert dest.dtype == src.dtype == torch.int32 and bool((dest[1:] > dest[:-1]).all())
    for seed in (0, 1, 2):
        native, counts = _native(level["fwd"], seed)
        want = cs.indice_conv_torch(f, w, native, counts, level["n"], subm=True, bias=b)
        for floor in (1, 64, 1000):
            for atomic_free, slab in ((True, 4096), (True, 7), (False, 4096)):
                monkeypatch.setattr(cs, "ATOMIC_FREE", atomic_free)
                monkeypatch.setattr(cs, "ACCUMULATE_MIN_ROWS", slab)
                monkeypatch.setattr(cs, "ACCUMULATE_BYTES", 1)
                got = cs.products(f, w, level["pairs"], None, b, floor=floor)
                assert torch.equal(got, want), (seed, floor, slab, atomic_free)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_the_products_add_in_the_native_order_on_random_values(level: dict, dtype: torch.dtype,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """On random values (where another order of the additions gives other bits: the bias first, the offsets
    descending) the products equal indice_conv_torch for every slot order, floor and accumulation."""
    g = torch.Generator().manual_seed(11)
    f = torch.randn((level["n"], 8), generator=g).to(dtype)
    w = (torch.randn((5, 3, 3, 3, 8), generator=g) / 4).to(dtype)
    b = torch.randn((5,), generator=g).to(dtype)
    for seed in (0, 1):
        native, counts = _native(level["fwd"], seed)
        want = cs.indice_conv_torch(f, w, native, counts, level["n"], subm=True, bias=b)
        for floor in (1, 64):
            for atomic_free in (True, False):
                monkeypatch.setattr(cs, "ATOMIC_FREE", atomic_free)
                got = cs.products(f, w, level["pairs"], None, b, floor=floor, window=0)
                assert torch.equal(got.view(torch.int16 if dtype == torch.float16 else torch.int32),
                                   want.view(torch.int16 if dtype == torch.float16 else torch.int32))
    # the order matters here: the offsets added in descending order give other bits
    reverse = cs.LevelPairs(level["n"], KV, offsets=list(reversed(level["pairs"].offsets)))
    w_reverse = w.reshape(5, KV, 8).flip(1).reshape(5, 3, 3, 3, 8).contiguous()
    other = cs.products(f, w_reverse, reverse, None, b, floor=1, window=0)
    assert not torch.equal(other, want)


def test_a_subset_of_rows_is_the_whole_tables_rows(level: dict) -> None:
    """products of some output rows only (an up-block's parents, a slab) are those rows of the whole call; the rows
    are taken as _rows_index gives them (a slice, ascending indices; the whole table as None)."""
    f, w, b, n = level["features"], level["filters"], level["bias"], level["n"]
    whole = cs.products(f, w, level["pairs"], None, b, floor=1)
    for rows in (slice(10, 200), torch.arange(0, n, 3), torch.tensor([5]), torch.arange(n - 7, n)):
        index = cs._rows_index(rows, n, f.device)
        got = cs.products(f, w, level["pairs"], index, b, floor=50)
        assert torch.equal(got, whole[index]), rows
    assert cs._rows_index(slice(0, n), n, "cpu") is None and cs._rows_index(None, n, "cpu") is None
    assert cs._rows_index(torch.arange(n), n, "cpu") is None


def test_a_box_of_its_own_gives_its_inner_rows_the_whole_tables_values(level: dict) -> None:
    """A box cut out of the level with a halo of one voxel, as its own table (its own row numbers and pairs): its
    rows inside the box equal the whole table's (chunked.distance / cut_boxes, the decode's helpers)."""
    coords, f, w, b = level["coords"], level["features"], level["filters"], level["bias"]
    whole = cs.products(f, w, level["pairs"], None, b, floor=1)
    boxes = chunked.cut_boxes(coords, 1, coords.shape[0] // 4)
    assert len(boxes) > 1
    for lo, hi in boxes:
        d = chunked.distance(coords, lo, hi)
        g = (d <= 1).nonzero().squeeze(1)
        own = (d[g] == 0).nonzero().squeeze(1)
        sub = coords[g]
        pairs = cs.LevelPairs(g.shape[0], KV, offsets=cs.offset_lists(_pair_fwd(sub)))
        got = cs.products(f[g], w, pairs, None, b, floor=32)
        assert torch.equal(got[own], whole[g[own]])


def test_the_floor_pads_small_calls_and_keeps_large_ones() -> None:
    calls: list = []
    real_mm = torch.mm

    def mm(a, b):
        calls.append(a.shape[0])
        return real_mm(a, b)

    rows, wt = _exact((10, 4), 5), _exact((4, 3), 6)
    try:
        torch.mm = mm
        small = cs._call(rows, wt, 16)
        large = cs._call(rows, wt, 8)
        gathered = cs._product(_exact((30, 4), 8), torch.tensor([3, 4, 9]), wt, 16)
        whole = cs._product(rows, None, wt, 1)
    finally:
        torch.mm = real_mm
    assert calls == [16, 10, 16, 10]
    assert small.shape == large.shape == whole.shape == (10, 3) and torch.equal(small, large)
    assert gathered.shape == (3, 3)


def test_the_floors_come_from_this_cards_record_for_this_profile(tmp_path: Any, monkeypatch: pytest.MonkeyPatch):
    """products_floor.floor / exact / problem: without this card's and profile's ok section (none, another card's,
    another profile's, a failed one) not exact - no boxes - and `problem` says why; a record's higher floor, never a
    lower one; not exact where its sweep found a stand-in shape without a floor (unless it names a window)."""
    record = tmp_path / "calibration.json"
    monkeypatch.setenv(calibration.ENV, str(record))
    monkeypatch.setenv("PIXAL3D_PRECISION", "pascal")
    monkeypatch.setattr(calibration, "card_name", lambda device=None: "NVIDIA GeForce GTX 1080")
    precision.reset()

    def write(section: Any, card: str = "NVIDIA GeForce GTX 1080") -> None:
        body = {"version": 3, "card": card}
        if section is not None:
            body[pf.SECTION] = section
        record.write_text(json.dumps(body), encoding="utf-8")
        calibration.forget()
        pf.forget()

    try:
        write(None)
        assert pf.floor() == pf.floor("linear") == pf.DEFAULT_FLOOR and not pf.exact()
        assert "no decode_products section" in pf.problem()
        record.unlink()
        calibration.forget()
        pf.forget()
        assert not pf.exact() and "no calibration record" in pf.problem()
        found = {"ok": True, "profile": "pascal", "products_floor": 32768, "linear_floor": 4096, "exact": True,
                 "verified": {"2": 1_000_000, "3": 2_500_000, "4": 12_000_000}}
        write(found)
        assert pf.floor("products") == 32768 and pf.floor("linear") == pf.DEFAULT_FLOOR and pf.exact()
        assert pf.problem() == "" and pf.verified(3) == 2_500_000 and pf.verified(0) is None
        assert pf.beyond({2: 900_000, 3: 2_600_000, 4: 12_000_000}) == [(3, 2_600_000, 2_500_000)]
        assert pf.covers_voxels(9_000_000) and not pf.covers_voxels(13_000_000)
        write({**found, "profile": "turing"})
        assert pf.floor() == pf.DEFAULT_FLOOR and not pf.exact() and "turing profile" in pf.problem()
        write(found, card="NVIDIA GeForce RTX 3060")
        assert pf.floor() == pf.DEFAULT_FLOOR and not pf.exact() and "another card" in pf.problem()
        write({**found, "linear_floor": None, "exact": False})
        assert pf.floor("linear") == pf.DEFAULT_FLOOR and not pf.exact()
        # round O's windows-everywhere record without shapes: the window is not read any more (round P), no boxes
        write({**found, "linear_floor": None, "exact": False, "window": 65536})
        assert not pf.exact() and pf.window() is None and pf.box_level() is None
        write({"ok": False, "error": "RuntimeError: CUDA out of memory"})
        assert pf.floor() == pf.DEFAULT_FLOOR and not pf.exact() and "failed" in pf.problem()
    finally:
        precision.reset()
        calibration.forget()
        pf.forget()


def test_the_tree_at_1536_is_covered_by_a_record_verified_to_4_m_level_3_rows(tmp_path: Any,
                                                                              monkeypatch: pytest.MonkeyPatch):
    """products_floor.covers_voxels on the planner's figures for the biggest test prop, the tree at 1536: 13.46 M
    foreseen fine voxels (parts.voxels_at, its margin included; it decodes 12,240,099, level 3 3,026,001 rows) give
    LEVEL_SHARE's 3.37 M level-3 rows - inside a record whose level 3 is verified at 4.0 M (the v2 sweep's ladder
    top), outside one verified at 3,355,000 (the v1 copies' 4 GiB budget), where the planner keeps it off the boxes;
    the windows-everywhere mode has no whole-level call to verify."""
    record = tmp_path / "calibration.json"
    monkeypatch.setenv(calibration.ENV, str(record))
    monkeypatch.setenv("PIXAL3D_PRECISION", "pascal")
    monkeypatch.setattr(calibration, "card_name", lambda device=None: "NVIDIA GeForce GTX 1080")
    precision.reset()
    tree = 13_460_000
    assert pf.LEVEL_SHARE[3] >= 3_026_001 / 12_240_099 and int(pf.LEVEL_SHARE[3] * tree) >= 3_026_001

    def write(level3: int, **more: Any) -> None:
        record.write_text(json.dumps({"version": 3, "card": "NVIDIA GeForce GTX 1080", pf.SECTION: {
            "ok": True, "profile": "pascal", "products_floor": 4441, "exact": True,
            "verified": {"3": level3, "4": 15_000_000}, **more}}), encoding="utf-8")
        calibration.forget()
        pf.forget()

    try:
        write(4_000_000)
        assert pf.exact() and pf.verified(3) == 4_000_000 and pf.verified(4) == 15_000_000
        assert pf.covers_voxels(tree) and pf.beyond({3: int(pf.LEVEL_SHARE[3] * tree), 4: tree}) == []
        write(3_355_000)
        assert not pf.covers_voxels(tree)
        assert pf.beyond({3: int(pf.LEVEL_SHARE[3] * tree), 4: tree}) == [(3, 3_365_000, 3_355_000)]
        assert pf.covers_voxels(int(3_355_000 / pf.LEVEL_SHARE[3]))     # the v1 records' reach: 13.42 M voxels
        monkeypatch.setenv(pf.WINDOW_ENV, "65536")          # the forced windows-everywhere mode: no limit
        assert pf.exact() and pf.covers_voxels(tree) and pf.beyond({3: 9_000_000}) == []
        monkeypatch.delenv(pf.WINDOW_ENV)
    finally:
        precision.reset()
        calibration.forget()
        pf.forget()


def test_the_sweeps_floor_rule() -> None:
    """The smallest tested M from which every tested M agreed; none when the largest did not."""
    assert pf.floor_of([(64, False), (1024, True), (2048, False), (4096, True), (8192, True)]) == 4096
    assert pf.floor_of([(64, True), (128, True)]) == 64
    assert pf.floor_of([(64, True), (128, False)]) is None
    assert pf.floor_of([]) is None
    counts = pf.row_counts(100_000, points=10, seed=3)
    assert counts == sorted(set(counts)) and counts[0] == pf.SMALLEST and counts[-1] < 100_000
    assert {64, 96, 1024, 1536, 65536} <= set(counts)
    assert pf.row_counts(100_000, points=10, seed=3) == counts
    sizes: list = []
    X = _exact((25, 3), 4)
    got = pf._windowed(lambda rows: (sizes.append(rows.shape[0]), rows * 2)[1], X, 7)
    assert sizes == [7, 7, 7, 7] and torch.equal(got, X * 2)
    shapes = pf.product_shapes()
    assert len(shapes) == 9 and (1024, 4096) in shapes and (64, 64) in shapes
    assert ("output", 64, 7, torch.float32) in pf.linear_shapes()
    roles = {(ci, co): role for ci, co, role in pf.product_roles()}
    assert {k for k, v in roles.items() if v == "stand-in"} == {(1024, 4096), (512, 2048), (256, 1024), (128, 128),
                                                                 (128, 512), (64, 64)}
    stand_in = {(name, ci) for name, ci, co, dtype, role in pf.linear_roles() if role == "stand-in"}
    assert stand_in == {("mlp1", 128), ("mlp2", 512), ("to_subdiv", 128), ("output", 64)}


def test_which_backend_decodes_in_boxes_and_the_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """chunked.backend / boxes_supported / applies: flex_gemm's masked split-K as before, spconv where conv_spconv
    has subm_rows for the card, nothing else; PIXAL3D_CHUNK_VOXELS overrides the decoder's chunk_voxels (0: none)."""
    class Decoder:
        chunk_voxels, training = 0, False

    x = type("X", (), {"shape": (1, 8)})()
    monkeypatch.delenv(chunked.CHUNK_ENV, raising=False)
    monkeypatch.setattr(sparse_config, "CONV", "flex_gemm")
    monkeypatch.setattr(conv_config, "FLEX_GEMM_ALGO", "masked_implicit_gemm_splitk")
    assert chunked.backend() == "flex_gemm" and chunked.boxes_supported()
    with torch.no_grad():
        assert not chunked.applies(Decoder(), x)
        monkeypatch.setenv(chunked.CHUNK_ENV, "1500000")
        assert chunked.chunk_voxels(Decoder()) == 1_500_000 and chunked.applies(Decoder(), x)
        monkeypatch.setenv(chunked.CHUNK_ENV, "0")
        decoder = Decoder()
        decoder.chunk_voxels = 2_000_000
        assert not chunked.applies(decoder, x)
        monkeypatch.delenv(chunked.CHUNK_ENV)
        assert chunked.applies(decoder, x)
        monkeypatch.setattr(conv_config, "FLEX_GEMM_ALGO", "implicit_gemm")
        assert chunked.backend() == "" and not chunked.applies(decoder, x)
        monkeypatch.setattr(sparse_config, "CONV", "spconv")
        monkeypatch.setattr(cs, "boxes_supported", lambda: True)
        monkeypatch.setattr(pf, "exact", lambda device=None: False)         # no usable record: no boxes
        assert chunked.backend() == "" and not chunked.applies(decoder, x)
        monkeypatch.setattr(pf, "exact", lambda device=None: True)
        assert chunked.backend() == "spconv" and chunked.applies(decoder, x)
        monkeypatch.setattr(pf, "covers_voxels", lambda voxels, device=None: voxels < 10_000_000)
        assert chunked.boxes_supported() and chunked.boxes_supported(9_000_000)
        assert not chunked.boxes_supported(11_000_000)
        monkeypatch.setattr(cs, "boxes_supported", lambda: False)
        assert chunked.backend() == "" and not chunked.boxes_supported()
        monkeypatch.setattr(sparse_config, "CONV", "torchsparse")
        assert chunked.backend() == ""
    assert not chunked.applies(decoder, x)                  # gradients on


def test_the_linear_windows_follow_the_floor_on_spconv_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """min_call_rows and box_window keep their constants whatever the record says (the boxes' own windows are
    judged by the record's exact; other callers of rowwise keep their windows); slices: flex_gemm's rule there,
    always on spconv (the under-floor offsets come from the level's own calls); whole_level_calls on spconv only;
    no boxes on spconv where the record is not exact."""
    up = type("Up", (), {"conv1": object(), "channels": 8, "out_channels": 8})()
    x = type("X", (), {"feats": torch.zeros((10, 8))})()
    monkeypatch.setattr(pf, "floor", lambda kind="products", device=None: 1 << 17)
    monkeypatch.setattr(sparse_config, "CONV", "flex_gemm")
    monkeypatch.setattr(conv_config, "FLEX_GEMM_ALGO", "masked_implicit_gemm_splitk")
    assert chunked.box_window() == chunked.BOX_WINDOW and chunked.min_call_rows() == chunked.MIN_CALL_ROWS
    monkeypatch.setattr(chunked, "reference_splitk", lambda rows, ci, co: 2)     # flex_gemm would split the sum
    assert not chunked.slices(up, x) and not chunked.whole_level_calls()
    monkeypatch.setattr(sparse_config, "CONV", "spconv")
    monkeypatch.setattr(cs, "boxes_supported", lambda: True)
    monkeypatch.setattr(pf, "exact", lambda device=None: True)
    monkeypatch.setattr(cs, "counts_at_least", lambda conv, x, floor=None: True)
    assert chunked.min_call_rows() == chunked.MIN_CALL_ROWS and chunked.box_window() == chunked.BOX_WINDOW
    assert chunked.slices(up, x)
    assert chunked.whole_level_calls()
    monkeypatch.setattr(cs, "counts_at_least", lambda conv, x, floor=None: False)
    assert chunked.slices(up, x)
    monkeypatch.setattr(pf, "floor", lambda kind="products", device=None: pf.DEFAULT_FLOOR)
    assert chunked.box_window() == chunked.BOX_WINDOW and chunked.min_call_rows() == chunked.MIN_CALL_ROWS
    monkeypatch.setattr(pf, "exact", lambda device=None: False)
    assert chunked.backend() == "" and not chunked.boxes_supported() and not chunked.whole_level_calls()


def test_counts_at_least_reads_the_level_pairs(level: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "path", lambda: "products")
    monkeypatch.setattr(cs, "level_pairs", lambda conv, x: level["pairs"])
    smallest = min(pair[0].shape[0] for pair in level["pairs"].offsets if pair is not None)
    assert cs.counts_at_least(None, None, smallest) and not cs.counts_at_least(None, None, smallest + 1)
    monkeypatch.setattr(cs, "path", lambda: "igemm")
    assert cs.counts_at_least(None, None, 1 << 30)


def test_forget_neighbors_drops_the_spconv_pair_tables() -> None:
    x = sp.SparseTensor(torch.zeros((3, 2)), torch.zeros((3, 4), dtype=torch.int32))
    x.register_spatial_cache(f"{cs.CACHE_KEY}_3x3x3_dilation(1, 1, 1)_products", object())
    x.register_spatial_cache("layout", [slice(0, 3)])
    suv.forget_neighbors(x)
    assert list(x.get_spatial_cache()) == ["layout"]


def test_the_windows_everywhere_mode(level: dict, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Where the install found a box shape without a floor, the record names a window: every product call (the
    whole table's too) and every Linear call has exactly that many rows; boxes are then exact by construction, every
    decode goes through forward_chunked, and the coarse levels' whole calls give way to the windows.
    PIXAL3D_DECODE_WINDOW forces it."""
    f, w, b = level["features"], level["filters"], level["bias"]
    native, counts = _native(level["fwd"], 0)
    want = cs.indice_conv_torch(f, w, native, counts, level["n"], subm=True, bias=b)
    for window in (7, 37, 1000):
        assert torch.equal(cs.products(f, w, level["pairs"], None, b, floor=1, window=window), want)
    calls: list = []
    real_mm = torch.mm
    try:
        torch.mm = lambda a, c: (calls.append(a.shape[0]), real_mm(a, c))[1]
        cs._call(_exact((25, 4), 7), _exact((4, 3), 6), 1, 10)
    finally:
        torch.mm = real_mm
    assert calls == [10, 10, 10]

    monkeypatch.delenv(pf.WINDOW_ENV, raising=False)
    pf.forget()
    record = tmp_path / "calibration.json"
    monkeypatch.setenv(calibration.ENV, str(record))
    monkeypatch.setenv("PIXAL3D_PRECISION", "pascal")
    monkeypatch.setattr(calibration, "card_name", lambda device=None: "NVIDIA GeForce GTX 1080")
    precision.reset()
    try:
        record.write_text(json.dumps({"card": "NVIDIA GeForce GTX 1080", pf.SECTION: {
            "ok": True, "profile": "pascal", "products_floor": 8192, "linear_floor": None, "exact": False,
            "window": 65536}}), encoding="utf-8")
        calibration.forget()
        pf.forget()
        assert pf.window() is None and not pf.exact()      # a record's window is round O's: not read (round P)
        monkeypatch.setenv(pf.WINDOW_ENV, "40")
        assert pf.window() == 40 and pf.exact() and pf.box_level() == 3
    finally:
        precision.reset()
        calibration.forget()
        pf.forget()

    class Decoder:
        chunk_voxels, training = 0, False

    x = type("X", (), {"shape": (1, 8)})()
    monkeypatch.setattr(sparse_config, "CONV", "spconv")
    monkeypatch.setattr(cs, "boxes_supported", lambda: True)
    monkeypatch.setattr(pf, "window", lambda device=None: 40)
    monkeypatch.setattr(pf, "exact", lambda device=None: True)
    with torch.no_grad():
        assert chunked.windows() == 40 and chunked.applies(Decoder(), x) and not chunked.whole_level_calls()
        sizes: list = []
        src = _exact((100, 3), 9)
        out = torch.empty_like(src)
        chunked.rowwise(lambda rows: (sizes.append(rows.shape[0]), rows * 2)[1], src, out, 1 << 16, pad=False)
        assert set(sizes) == {40} and torch.equal(out, src * 2)
        monkeypatch.setattr(cs, "BOXES", False)                 # PIXAL3D_SPCONV_BOXES=off: today's unwindowed calls
        assert cs._window() is None
        monkeypatch.setattr(cs, "BOXES", True)
        assert cs._window() == 40
        monkeypatch.setattr(pf, "window", lambda device=None: None)
        assert chunked.windows() is None and not chunked.applies(Decoder(), x) and chunked.whole_level_calls()


def test_the_install_step_skips_itself_where_the_decoders_run_on_flex_gemm(monkeypatch: pytest.MonkeyPatch) -> None:
    """products_floor.measure (the calibration's "decode_products" step): on a profile whose decoders run on
    flex_gemm (Ampere), or with flex_gemm chosen for an old card's decoders, a "skipped" section at once, no card
    touched; the profile is read first (the pascal profile turns cuBLASLt's fused linear off before torch's first
    addmm)."""
    seen: list = []

    def profile(name: str, conv: str):
        def read():
            seen.append(name)
            return type("P", (), {"name": name, "capability": (8, 6), "conv": conv})()
        return read

    def no_card(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the card was touched")

    monkeypatch.setattr(torch.cuda, "get_device_name", no_card)
    monkeypatch.setattr(pf, "_table", no_card)
    for name, conv, backend in (("ampere", "flex_gemm", "flex_gemm"), ("turing", "spconv", "flex_gemm")):
        monkeypatch.setattr(precision, "profile", profile(name, conv))
        monkeypatch.setattr(sparse_config, "CONV", backend)
        found = pf.measure("cpu", say=no_card, pause=no_card)
        assert found["ok"] and found["skipped"] and found["exact"] and found["profile"] == name, found
        assert found["products_floor"] is None and found["shapes"] == []
    assert seen == ["ampere", "turing"]


def test_counts_at_least_lets_an_offset_without_pairs_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """A one-voxel-thick plane: the offsets across it pair no rows, and make no call on either side."""
    coords = torch.stack(torch.meshgrid(torch.arange(12), torch.arange(12), torch.zeros(1, dtype=torch.long),
                                        indexing="ij"), -1).reshape(-1, 3)
    pairs = cs.LevelPairs(coords.shape[0], KV, offsets=cs.offset_lists(_pair_fwd(coords)))
    sizes = [pair[0].shape[0] for pair in pairs.offsets if pair is not None]
    assert 0 in sizes and min(n for n in sizes if n) == 121
    monkeypatch.setattr(cs, "path", lambda: "products")
    monkeypatch.setattr(cs, "_window", lambda: None)
    monkeypatch.setattr(cs, "level_pairs", lambda conv, x: pairs)
    assert cs.counts_at_least(None, None, 121) and not cs.counts_at_least(None, None, 122)


def test_sorted_masks_are_a_stable_sort_of_the_subsets_own_masks() -> None:
    g = torch.Generator().manual_seed(4)
    n = 300
    masks = torch.randint(0, 1 << 6, (n,), generator=g, dtype=torch.int32)    # few values: many ties
    values, order = torch.sort(masks, stable=True)
    pairs = cs.LevelPairs(n, KV, pair_fwd=torch.zeros((KV, n), dtype=torch.int32),
                          mask_sorted=values.reshape(-1, 1).contiguous(), argsort=order.to(torch.int32))
    rows = torch.arange(0, n, 3)
    got_values, got_order = cs._sorted_masks(pairs, rows)
    want_values, want_order = torch.sort(masks[rows], stable=True)
    assert torch.equal(got_values[:, 0], want_values) and torch.equal(got_order.long(), want_order)
    assert got_order.dtype == torch.int32 and got_values.shape == (rows.shape[0], 1)


def test_subm_rows_pads_only_the_calls_that_stand_in(level: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """reference=True (the whole table's own call) makes unpadded product calls; reference=False pads every call
    below the floor - spied on torch.mm - and both give the same rows."""
    conv = type("Conv", (), {"weight": level["filters"], "bias": level["bias"], "out_channels": 5})()
    x = type("X", (), {"feats": level["features"], "device": "cpu"})()
    monkeypatch.setattr(cs, "path", lambda: "products")
    monkeypatch.setattr(cs, "level_pairs", lambda conv_, x_: level["pairs"])
    monkeypatch.setattr(cs.spconv_choices, "restore", lambda device=None: 0)
    monkeypatch.setattr(cs, "_note_choices", lambda: None)
    monkeypatch.setattr(pf, "floor", lambda kind="products", device=None: 1000)
    monkeypatch.setattr(pf, "window", lambda device=None: None)
    calls: list = []
    real_mm = torch.mm
    try:
        torch.mm = lambda a, b: (calls.append(a.shape[0]), real_mm(a, b))[1]
        whole = cs.subm_rows(conv, x, reference=True)
        reference_calls, calls[:] = list(calls), []
        stand_in = cs.subm_rows(conv, x)
        padded_calls = list(calls)
    finally:
        torch.mm = real_mm
    sizes = [level["n"]] + [pair[0].shape[0] for pair in level["pairs"].offsets if pair is not None and pair[0].numel()]
    assert reference_calls == sizes and all(c >= 1000 for c in padded_calls) and len(padded_calls) == len(sizes)
    assert torch.equal(whole, stand_in)


def test_a_box_takes_the_under_floor_offsets_from_the_whole_level(level: dict, monkeypatch: pytest.MonkeyPatch):
    """whole_offset_products: the offsets that pair fewer rows than the floor, each with the whole level's own call
    (prep applied to the rows); a box's call with `fixed` adds those rows in at their offset's turn and gives the
    whole table's rows (here exact integers: the same sums), as a subset of the whole level does."""
    coords, f, w, b = level["coords"], level["features"], level["filters"], level["bias"]
    conv = type("Conv", (), {"weight": w, "bias": b, "out_channels": 5})()
    x = type("X", (), {"feats": f, "device": "cpu"})()
    monkeypatch.setattr(cs, "path", lambda: "products")
    monkeypatch.setattr(cs, "_window", lambda: None)
    monkeypatch.setattr(cs, "level_pairs", lambda conv_, x_: level["pairs"])
    sizes = sorted(pair[0].shape[0] for pair in level["pairs"].offsets if pair is not None)
    floor = sizes[len(sizes) // 2]
    whole = cs.whole_offset_products(conv, x, prep=lambda rows: rows * 2, floor=floor)
    assert whole and all(0 < whole[k][0].shape[0] < floor for k in whole)
    assert all(torch.equal(whole[k][1], (f[level["pairs"].offsets[k][1].long()] * 2)
                           @ w.reshape(5, KV, 8)[:, k].t()) for k in whole)
    whole = cs.whole_offset_products(conv, x, floor=floor)
    reference = cs.products(f, w, level["pairs"], None, b, floor=1)
    for lo, hi in chunked.cut_boxes(coords, 1, coords.shape[0] // 3):
        d = chunked.distance(coords, lo, hi)
        g = (d <= 1).nonzero().squeeze(1)
        own = (d[g] == 0).nonzero().squeeze(1)
        pairs = cs.LevelPairs(g.shape[0], KV, offsets=cs.offset_lists(_pair_fwd(coords[g])))
        got = cs.products(f[g], w, pairs, own, b, floor=floor, fixed=(whole, g[own]))
        assert torch.equal(got, reference[g[own]])
    monkeypatch.setattr(cs, "_window", lambda: 65536)
    assert cs.whole_offset_products(conv, x, floor=floor) == {}


def test_the_boundary_band_and_the_big_check_sizes(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    record = tmp_path / "calibration.json"
    monkeypatch.setenv(calibration.ENV, str(record))
    monkeypatch.setenv("PIXAL3D_PRECISION", "pascal")
    monkeypatch.setattr(calibration, "card_name", lambda device=None: "NVIDIA GeForce GTX 1080")
    precision.reset()
    try:
        record.write_text(json.dumps({"card": "NVIDIA GeForce GTX 1080", pf.SECTION: {
            "ok": True, "profile": "pascal", "exact": True, "boundary": [
                {"kind": "output", "ci": 64, "co": 7, "M": 2097152, "rows_differ": 64, "first": 2097088},
                {"kind": "output", "ci": 64, "co": 7, "M": 4194304, "rows_differ": 128, "first": 4194176}]}}),
            encoding="utf-8")
        calibration.forget()
        pf.forget()
        assert pf.has_boundary() and not pf.has_boundary(shape=("products", 64))
        assert pf.boundary_band(2_097_152) and pf.boundary_band(4_194_200) and pf.boundary_band(8_388_400)
        assert not pf.boundary_band(2_097_088) and not pf.boundary_band(2_500_000) and not pf.boundary_band(None)
        # round P's review: power-of-two multiples of the smallest boundary only (3 x 2,097,088 is no check's)
        assert not pf.boundary_band(6_291_300)
        # the first row of a banded call that may differ: BAND_ROWS before the band's start
        assert pf.band_from("output", 64, 7, 2_097_152) == 2_097_088 - pf.BAND_ROWS
        assert pf.band_from("output", 64, 7, 2_500_000) is None
    finally:
        precision.reset()
        calibration.forget()
        pf.forget()
    assert pf.big_sizes((600_000, 1_000_000), 256, 1024, 2, 4 << 30) == [600_000, 1_000_000]
    assert pf.big_sizes((1_200_000, 2_500_000, 4_000_000), 128, 512, 2, 4 << 30) == [1_200_000, 2_500_000, 3_355_000]
    assert pf.big_sizes((3_000_000,), 64, 64, 2, 1 << 20) == []
    # Past the ladder: the level's sizes of props of 20 to 40 M fine voxels, and the ladder sizes a capped ladder
    # budget left out, as far as the budget holds them with the check's own working memory - an 8 GB card reaches
    # one more level-4 size and no level-3 one, an 11 GB card two of each; nothing without a card.
    gib = 1 << 30
    assert pf.reach_sizes(4, 64, 64, 2, int(5.1 * gib), 17_000_000) == [20_000_000]
    assert pf.reach_sizes(3, 128, 512, 2, int(5.1 * gib), 4_000_000) == []
    assert pf.reach_sizes(3, 128, 512, 2, int(7.53 * gib), 4_000_000) == [5_000_000, 6_000_000]
    assert pf.reach_sizes(4, 64, 64, 2, int(7.53 * gib), 17_000_000) == [20_000_000, 24_000_000]
    assert pf.reach_sizes(3, 128, 512, 4, int(7.53 * gib), 2_300_000) == [2_500_000]
    assert pf.reach_sizes(4, 64, 64, 2, 0, 17_000_000) == []
    whole_call = 20_000_000 * 2 * (64 + 64)             # a 20 M-row call's input and output, no more
    assert pf.reach_sizes(4, 64, 64, 2, whole_call + (8 << 20), 17_000_000) == [], "no room for the windows' outputs"
    assert pf.reach_sizes(4, 64, 64, 2, whole_call + (17 << 20), 17_000_000) == [20_000_000]
    for voxels in pf.REACH_VOXELS:      # a prop of that many voxels is covered once every level agreed at its size
        assert all(int(pf.LEVEL_SHARE[level] * voxels) in pf.reach_sizes(level, 8, 8, 2, 64 * gib, 0) for level in range(5))
    # ... checked one after the other, up to and with the first whose whole call differs: that ends the reach.
    weight = torch.randn((4, 3))

    def another_kernel_above_150k(rows: torch.Tensor) -> torch.Tensor:
        out = rows @ weight
        return out + 1 if rows.shape[0] > 150_000 else out

    found = pf._reach_check(another_kernel_above_150k, 4, 3, torch.float32, "cpu", 0, [70_000, 140_000, 210_000, 280_000])
    assert [(c["M"], bool(c["rows_differ"])) for c in found] == [(70_000, False), (140_000, False), (210_000, True)]
    assert pf.product_level(256, 1024) == 2 and pf.product_level(64, 64) == 4 and pf.product_level(128, 512) == 3
    assert pf.linear_level("mlp2", 512, 128) == 3 and pf.linear_level("output", 64, 7) == 4


def test_a_records_linear_floor_leaves_the_encoders_windows(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pascal record whose stand-in Linear floor is 1,229,010 (the RTX 3060's: the fp32 output layer's tail rows at
    2^20) on the spconv backend: lean_encode's Linear windows (LINEAR_WINDOW, the shape encoder Hi3D runs) are cut
    exactly as without a record - the floor is the decoders' business only."""
    record = tmp_path / "calibration.json"
    monkeypatch.setenv(calibration.ENV, str(record))
    monkeypatch.setenv("PIXAL3D_PRECISION", "pascal")
    monkeypatch.setattr(calibration, "card_name", lambda device=None: "NVIDIA GeForce GTX 1080")
    precision.reset()
    layer = torch.nn.Linear(4, 2)
    rows = _exact((600_000, 4), 12)

    def cut() -> list:
        sizes: list = []
        real = torch.nn.Linear.forward
        try:
            torch.nn.Linear.forward = lambda self, r: (sizes.append(r.shape[0]), real(self, r))[1]
            lean_encode.linear_rows(layer, rows, torch.float32)
        finally:
            torch.nn.Linear.forward = real
        return sizes

    try:
        monkeypatch.setattr(sparse_config, "CONV", "flex_gemm")
        before = cut()
        record.write_text(json.dumps({"card": "NVIDIA GeForce GTX 1080", pf.SECTION: {
            "ok": True, "profile": "pascal", "products_floor": 4441, "linear_floor": 1_229_010, "exact": True}}),
            encoding="utf-8")
        calibration.forget()
        pf.forget()
        monkeypatch.setattr(sparse_config, "CONV", "spconv")
        monkeypatch.setattr(cs, "boxes_supported", lambda: True)
        assert chunked.backend() == "spconv" and pf.floor("linear") == 1_229_010
        assert chunked.min_call_rows() == chunked.MIN_CALL_ROWS
        assert cut() == before and max(before) <= lean_encode.LINEAR_WINDOW
    finally:
        precision.reset()
        calibration.forget()
        pf.forget()


# ------------------------------------------------------------------------------------------------ round P: decode v2

def _shape_entry(kind: str, ci: int, co: int, level: int, floor: Any, failed: list = (), big: list = (),
                 mmax: int = 1_900_000, window_ok: bool = True, role: str = "stand-in") -> dict:
    return {"kind": kind, "role": role, "level": level, "ci": ci, "co": co, "dtype": "float16", "mmax": mmax,
            "tested": 77, "failed": list(failed), "floor": floor, "window_ok": window_ok, "verified": 4_000_000,
            "big": [{"M": m, "rows_differ": d, "first": 0 if d else None} for m, d in big], "boundary": []}


def _sm61_shapes(flag128: bool = True, output_limit: bool = True, subdiv_point: bool = True) -> list:
    """The stand-in shapes of a decode_products sweep as the real GTX 1080 Ti / 1070 Ti recorded them (round O,
    pod2 / pod3: the 128 -> 128 product's floor 1,160,440 - its one failing row count above 6,144 is 2^20 - the
    subdivision head's 2^20 failure, and the fp32 output layer's whole calls of 12 M / 17 M rows on another kernel),
    or as an RTX 3060 under the pascal profile did (every shape box-safe, every big check equal, no failure from
    64 Ki rows on: all three False)."""
    ladder3 = [(1_200_000, 0), (2_500_000, 0), (4_000_000, 0)]
    out_big = [(3_000_000, 0), (6_000_000, 0), (12_000_000, 11_999_810 if output_limit else 0),
               (17_000_000, 16_999_725 if output_limit else 0)]
    return [
        _shape_entry("products", 1024, 4096, 0, 962, big=[(60_000, 0)], mmax=78_643),
        _shape_entry("products", 512, 2048, 1, 2048, big=[(250_000, 0)], mmax=157_286),
        _shape_entry("products", 256, 1024, 2, 1010, big=[(1_250_000, 0)], mmax=314_572),
        _shape_entry("products", 128, 128, 3, 1_160_440 if flag128 else 64,
                     failed=[64, 6144, 1_048_576] if flag128 else [], big=ladder3, mmax=1_572_864),
        _shape_entry("products", 128, 512, 3, 1884, big=ladder3, mmax=629_145),
        _shape_entry("products", 64, 64, 4, 64, big=[(17_000_000, 0)]),
        _shape_entry("mlp1", 128, 512, 3, 1780, big=ladder3, mmax=629_145),
        _shape_entry("mlp2", 512, 128, 3, 32_768, big=ladder3, mmax=629_145),
        _shape_entry("to_subdiv", 128, 8, 3, 1_232_399 if subdiv_point else 12_800,
                     failed=[12_536, 1_048_576] if subdiv_point else [12_536], big=ladder3),
        _shape_entry("output", 64, 7, 4, None if output_limit else 64, big=out_big, window_ok=not output_limit),
        _shape_entry("output", 64, 6, 4, None if output_limit else 64, big=out_big, window_ok=not output_limit),
        _shape_entry("products", 512, 512, 1, 8192, role="whole", mmax=393_216),
    ]


def test_a_round_o_record_of_a_real_gtx_10xx_decodes_in_the_level_4_form(tmp_path: Any,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """products_floor.plan_of / plan (round P) on a round O record of the real sm_61 sweep - exact False, window
    65,536, as the installs wrote it: the 128 -> 128 product is whole-only, so the boxes start at level 4 (level 3
    decoded whole); the output layer's windows reach 6 M fine rows; no windows-everywhere mode; the per-shape floors
    pad the box shapes to 8,192 rows, not to the class floor's 1,160,440. On an RTX 3060's record the boxes start at
    level 3 as in round O, with no limit to the output layer's windows."""
    plan = pf.plan_of({"shapes": _sm61_shapes()}, products=True)
    assert plan["box_level"] == 4 and plan["output_windows_to"] == 6_000_000
    assert [(w["kind"], w["ci"], w["co"]) for w in plan["whole_only"]] == [("products", 128, 128)]
    assert "products 128 -> 128 is whole-only" in plan["why_not_3"]
    clean = _sm61_shapes(flag128=False, output_limit=False, subdiv_point=False)
    assert pf.plan_of({"shapes": clean}, True) == \
        {"box_level": 3, "whole_only": [], "output_windows_to": None, "why": "", "why_not_3": ""}
    # round P's review: round O's form never makes the planner's whole output call - a reach of the output layer's
    # windows alone starts the boxes at level 4 (the measured figures, level 3's table in RAM, the parked mesh) ...
    reach_only = pf.plan_of({"shapes": _sm61_shapes(flag128=False, subdiv_point=False)}, True)
    assert reach_only["box_level"] == 4 and reach_only["whole_only"] == [] and reach_only["output_windows_to"] == 6_000_000
    assert "reach 6,000,000 fine rows" in reach_only["why_not_3"]
    # ... and so does a launch-boundary band of a level-3 stand-in round O's boxes would not keep (the subdivision
    # head's 2^20 sweep failure; mlp1's recorded 2^21 band; the level-3 conv1 128 -> 512's)
    subdiv = pf.plan_of({"shapes": _sm61_shapes(flag128=False, output_limit=False)}, True)
    assert subdiv["box_level"] == 4 and subdiv["whole_only"] == [] and "to_subdiv 128 -> 8" in subdiv["why_not_3"]
    band = [{"M": 2_097_152, "rows_differ": 16, "first": 2_097_092}]
    for kind, ci, co in (("mlp1", 128, 512), ("products", 128, 512), ("mlp2", 512, 128), ("products", 128, 128)):
        shapes = [dict(s, boundary=band) if (s["kind"], s["ci"], s["co"]) == (kind, ci, co) else s for s in clean]
        assert pf.plan_of({"shapes": shapes}, True)["box_level"] == 4, kind
        # the same band named in the section's boundary list (round O's records) instead of the shape's own
        assert pf.plan_of({"shapes": clean, "boundary": [{"kind": kind, "ci": ci, "co": co, **band[0]}]},
                          True)["box_level"] == 4, kind
    # level 2's conv1 (256 -> 1024) is the one round O's form takes from the whole level: its band leaves level 3
    shapes = [dict(s, boundary=band) if (s["kind"], s["ci"], s["co"]) == ("products", 256, 1024) else s for s in clean]
    assert pf.plan_of({"shapes": shapes}, True)["box_level"] == 3
    assert pf.band_handled(3) == ("products", 256, 1024) and pf.band_handled(4) == ("products", 128, 512)
    # the finest level's 64 -> 64 products stand in for the whole level's in both forms: a band there, no boxes
    shapes = [dict(s, boundary=band) if (s["kind"], s["ci"], s["co"]) == ("products", 64, 64) else s for s in clean]
    assert pf.plan_of({"shapes": shapes}, True)["box_level"] is None
    assert "64 -> 64 has a launch-boundary band" in pf.plan_of({"shapes": shapes}, True)["why"]
    # the implicit GEMM (turing) does not look at the products: the 3060's turing record has none; with the output
    # layer's reach (and the subdivision head's band) its boxes too start at level 4
    assert pf.plan_of({"shapes": _sm61_shapes()}, products=False)["box_level"] == 4
    assert pf.plan_of({"shapes": _sm61_shapes(output_limit=False, subdiv_point=False)}, products=False)["box_level"] == 3
    assert pf.FORM == 4 and (1 << 20) in pf.BOUNDARY_SIZES
    # a products card whose record has no products' shapes, and no level whose box shapes are safe: no boxes
    linear_only = [s for s in _sm61_shapes() if s["kind"] != "products"]
    assert pf.plan_of({"shapes": linear_only}, products=True)["box_level"] is None
    bad64 = [dict(s, floor=None) if s["kind"] == "products" and s["ci"] == 64 else s for s in _sm61_shapes()]
    assert pf.plan_of({"shapes": bad64}, products=True)["box_level"] is None
    assert pf.box_shapes(4) == [("products", 128, 512), ("products", 64, 64)]
    assert set(pf.box_shapes(3)) == {("products", 256, 1024), ("products", 128, 128), ("products", 128, 512),
                                     ("mlp1", 128, 512), ("mlp2", 512, 128), ("to_subdiv", 128, 8),
                                     ("products", 64, 64)}

    record = tmp_path / "calibration.json"
    monkeypatch.setenv(calibration.ENV, str(record))
    monkeypatch.setenv("PIXAL3D_PRECISION", "pascal")
    monkeypatch.setattr(calibration, "card_name", lambda device=None: "NVIDIA GeForce GTX 1080 Ti")
    monkeypatch.setattr(pf, "_products_here", lambda: True)
    precision.reset()
    try:
        record.write_text(json.dumps({"version": 3, "card": "NVIDIA GeForce GTX 1080 Ti", pf.SECTION: {
            "ok": True, "profile": "pascal", "products_floor": 1_160_440, "linear_floor": None, "exact": False,
            "window": 65536, "verified": {"3": 4_000_000, "4": 17_000_000}, "shapes": _sm61_shapes(),
            "boundary": [{"kind": "products", "ci": 128, "co": 512, "M": 2_097_152, "rows_differ": 28,
                          "first": 2_097_089}]}}), encoding="utf-8")
        calibration.forget()
        pf.forget()
        assert pf.exact() and pf.box_level() == 4 and pf.window() is None and pf.problem() == ""
        assert pf.floor("products") == 1_160_440                    # the class floor: information only now
        assert pf.floor_for("products", 128, 512) == pf.floor_for("products", 64, 64) == pf.DEFAULT_FLOOR
        assert pf.floor_for("mlp2", 512, 128) == 32_768
        assert not pf.box_safe("products", 128, 128) and pf.box_safe("products", 128, 512)
        assert pf.box_safe("products", 256, 1024) and pf.box_safe("mlp1", 128, 512)
        assert pf.output_may_be_whole() and pf.output_whole(10_500_000) and not pf.output_whole(3_550_000)
        assert not pf.output_whole(6_000_000) and pf.output_whole(6_000_001)
        # the products' launch-boundary band of the shape's own (kind, ci, co), and the sweep's 2^20 failure
        assert pf.in_band("products", 128, 512, 2_097_500) and not pf.in_band("products", 128, 128, 2_097_500)
        assert not pf.in_band("products", 128, 512, 2_096_000)
        assert pf.in_band("to_subdiv", 128, 8, 1_048_000) and not pf.in_band("to_subdiv", 128, 8, 1_200_000)
        assert pf.boundary_band(1_048_576, shape=("products", 128, 128))
        # PIXAL3D_DECODE_WINDOW forces round O's windows mode: boxes from level 3, every shape stood in for
        monkeypatch.setenv(pf.WINDOW_ENV, "65536")
        assert pf.window() == 65536 and pf.box_level() == 3 and pf.box_safe("products", 128, 128)
        assert not pf.output_whole(10_500_000) and not pf.output_may_be_whole()
        monkeypatch.delenv(pf.WINDOW_ENV)
        # this round's form: the fields measure() writes are read as they are (the same plan as the shapes give)
        body = json.loads(record.read_text(encoding="utf-8"))
        derived = pf.plan_of(body[pf.SECTION], True)
        body[pf.SECTION].update(form=pf.FORM, box_level=derived["box_level"], whole_only=derived["whole_only"],
                                output_windows_to=derived["output_windows_to"], exact=True, window=None,
                                shapes=[s for s in body[pf.SECTION]["shapes"] if s["kind"] != "output"])
        record.write_text(json.dumps(body), encoding="utf-8")
        calibration.forget()
        pf.forget()
        assert pf.box_level() == 4 and pf.plan()["output_windows_to"] == 6_000_000
        assert not pf.box_safe("products", 128, 128) and pf.output_whole(10_500_000)
    finally:
        precision.reset()
        calibration.forget()
        pf.forget()


def test_the_level_choice_the_slices_and_level_3s_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """chunked.first_box_level: round O's level 3 on flex_gemm and where the record says 3, the finest level where it
    says 4; slices follows the shape's box safety on spconv; _level_windows lets level 3's MLP run in the boxes'
    64 Ki-row calls only where both layers are box-safe, the level reaches their floors and is in neither's band."""
    decoder = type("D", (), {"blocks": [None] * 5})()
    monkeypatch.setattr(sparse_config, "CONV", "flex_gemm")
    monkeypatch.setattr(conv_config, "FLEX_GEMM_ALGO", "masked_implicit_gemm_splitk")
    monkeypatch.setattr(pf, "box_level", lambda device=None: 4)
    assert chunked.first_box_level(decoder) == 3
    monkeypatch.setattr(sparse_config, "CONV", "spconv")
    monkeypatch.setattr(cs, "boxes_supported", lambda: True)
    monkeypatch.setattr(pf, "exact", lambda device=None: True)
    assert chunked.first_box_level(decoder) == 4
    monkeypatch.setattr(pf, "box_level", lambda device=None: 3)
    assert chunked.first_box_level(decoder) == 3

    up = type("Up", (), {"channels": 128, "out_channels": 64})()
    monkeypatch.setattr(pf, "box_safe", lambda kind, ci, co, device=None: (kind, ci, co) != ("products", 128, 512))
    assert not chunked.slices(up, None)
    up.channels, up.out_channels = 256, 128
    assert chunked.slices(up, None)

    block = type("B", (), {"mlp": (torch.nn.Linear(128, 512), None, torch.nn.Linear(512, 128))})()
    monkeypatch.setattr(pf, "box_safe", lambda kind, ci, co, device=None: True)
    monkeypatch.setattr(pf, "floor_for", lambda kind, ci, co, device=None: 32_768 if kind == "mlp2" else 8192)
    monkeypatch.setattr(pf, "in_band", lambda kind, ci, co, rows, device=None: kind == "mlp1" and rows == 2_097_500)
    assert chunked._level_windows(block, 900_000) and not chunked._level_windows(block, 30_000)
    assert not chunked._level_windows(block, 2_097_500)
    monkeypatch.setattr(pf, "box_safe", lambda kind, ci, co, device=None: kind != "mlp2")
    assert not chunked._level_windows(block, 900_000)


def test_a_box_takes_the_band_offsets_from_the_whole_level(level: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """whole_offset_products (round P) also takes an offset whose whole-level count lands in the shape's
    launch-boundary band (products_floor.in_band), above the floor, and the shape's own floor (floor_for) where none
    is given; a subset of rows given those offsets' rows gets the whole table's values."""
    f, w, b = level["features"], level["filters"], level["bias"]
    conv = type("Conv", (), {"weight": w, "bias": b, "out_channels": 5})()
    x = type("X", (), {"feats": f, "device": "cpu"})()
    monkeypatch.setattr(cs, "path", lambda: "products")
    monkeypatch.setattr(cs, "_window", lambda: None)
    monkeypatch.setattr(cs, "level_pairs", lambda conv_, x_: level["pairs"])
    sizes = sorted({pair[0].shape[0] for pair in level["pairs"].offsets if pair is not None})
    banded = sizes[-1]
    monkeypatch.setattr(pf, "in_band", lambda kind, ci, co, rows, device=None: (kind, ci, co, rows) ==
                        ("products", 8, 5, banded))
    seen: list = []
    monkeypatch.setattr(pf, "floor_for", lambda kind, ci, co, device=None: (seen.append((kind, ci, co)), 1)[1])
    found = cs.whole_offset_products(conv, x)
    assert seen == [("products", 8, 5)]
    assert found and all(found[k][0].shape[0] == banded for k in found)
    reference = cs.products(f, w, level["pairs"], None, b, floor=1)
    rows = torch.arange(0, level["n"], 2)
    got = cs.products(f, w, level["pairs"], rows, b, floor=1, fixed=(found, rows))
    assert torch.equal(got, reference[rows])


# ------------------------------------------------------------------------------------------------ round P: the review

def _band_level(level: dict, monkeypatch: pytest.MonkeyPatch) -> tuple:
    conv = type("Conv", (), {"weight": level["filters"], "bias": level["bias"], "out_channels": 5})()
    x = type("X", (), {"feats": level["features"], "device": "cpu"})()
    monkeypatch.setattr(cs, "path", lambda: "products")
    monkeypatch.setattr(cs, "_window", lambda: None)
    monkeypatch.setattr(cs, "level_pairs", lambda conv_, x_: level["pairs"])
    cs.BAND_NOTES.clear()
    return conv, x


def test_a_band_call_that_does_not_fit_the_card_is_left_to_the_boxes(level: dict, monkeypatch: pytest.MonkeyPatch):
    """whole_offset_products (round P's review): a band's whole call is made only where its input and output fit the
    card's room (conv_spconv.fits: card_room less BAND_ROOM_MARGIN; always off the card); where they do not, the offset
    is left to the boxes' own calls and BAND_NOTES says so - the offsets under the floor are taken whatever the room
    (each under floor x Co)."""
    conv, x = _band_level(level, monkeypatch)
    sizes = sorted({pair[0].shape[0] for pair in level["pairs"].offsets if pair is not None})
    banded, floor = sizes[-1], sizes[len(sizes) // 2]
    monkeypatch.setattr(pf, "in_band", lambda kind, ci, co, rows, device=None: rows == banded)
    monkeypatch.setattr(pf, "band_from", lambda kind, ci, co, rows, device=None: None)
    assert cs.card_room("cpu") is None and cs.fits(1 << 40, "cpu")
    roomy = cs.whole_offset_products(conv, x, floor=floor)
    assert any(roomy[k][0].shape[0] == banded for k in roomy) and not cs.BAND_NOTES
    monkeypatch.setattr(cs, "card_room", lambda device: 0)
    tight = cs.whole_offset_products(conv, x, floor=floor)
    assert tight and all(0 < tight[k][0].shape[0] < floor for k in tight)          # the under-floor ones stay
    assert set(tight) == {k for k in roomy if roomy[k][0].shape[0] < floor}
    assert cs.BAND_NOTES and all("does not fit the card now" in note for note in cs.BAND_NOTES)
    cs.BAND_NOTES.clear()


def test_the_centre_of_a_level_in_a_band_comes_from_the_whole_levels_call(level: dict,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """kept[3] of round P's review: where the level's own row count is in the shape's band (products_floor.band_from),
    whole_offset_products makes the level's own centre call (prep applied) and keeps its rows from the band's first
    possibly differing row on; products() puts them in place of the call's own centre rows before any offset is
    added - a subset of rows gets the whole table's values, and the entry's values are the ones used."""
    conv, x = _band_level(level, monkeypatch)
    f, w, b, n = level["features"], level["filters"], level["bias"], level["n"]
    centre, lo = KV // 2, 100
    monkeypatch.setattr(pf, "in_band", lambda kind, ci, co, rows, device=None: False)
    monkeypatch.setattr(pf, "band_from", lambda kind, ci, co, rows, device=None:
                        lo if (kind, ci, co, rows) == ("products", 8, 5, n) else None)
    found = cs.whole_offset_products(conv, x, prep=lambda rows: rows * 2, floor=1)
    assert set(found) == {centre} and torch.equal(found[centre][0], torch.arange(lo, n))
    assert torch.equal(found[centre][1], (f * 2)[lo:] @ w.reshape(5, KV, 8)[:, centre].t())
    found = cs.whole_offset_products(conv, x, floor=1)
    reference = cs.products(f, w, level["pairs"], None, b, floor=1)
    rows = torch.arange(1, n, 3)
    got = cs.products(f, w, level["pairs"], rows, b, floor=1, fixed=(found, rows))
    assert torch.equal(got, reference[rows])
    marked = {centre: (found[centre][0], found[centre][1] + 1)}            # the entry's rows are the ones taken
    moved = cs.products(f, w, level["pairs"], rows, b, floor=1, fixed=(marked, rows)) - got
    assert torch.equal(moved, (rows >= lo).to(moved.dtype)[:, None].expand_as(moved))


def test_level_3s_mlp_in_a_band_makes_the_whole_calls_only_where_they_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """chunked._level_mlp_windows (round P's review, kept[2]): the windows where they keep the bits; the whole level's
    calls for a level whose size is in the MLP's band where its hidden table fits the card's room, else the windows
    with a note; whole calls whatever the room where a layer is whole-only (parts counts those)."""
    blocks = [type("B", (), {"mlp": (torch.nn.Linear(128, 512), None, torch.nn.Linear(512, 128))})()] * 2
    monkeypatch.setattr(pf, "box_safe", lambda kind, ci, co, device=None: True)
    monkeypatch.setattr(pf, "floor_for", lambda kind, ci, co, device=None: 8192)
    monkeypatch.setattr(pf, "in_band", lambda kind, ci, co, rows, device=None: kind == "mlp1" and rows == 2_097_500)
    assert chunked._level_mlp_windows(blocks, 900_000, "cpu") == (True, "")
    windowed, note = chunked._level_mlp_windows(blocks, 2_097_500, "cpu")
    assert not windowed and "the whole level's own calls" in note
    monkeypatch.setattr(cs, "fits", lambda need, device: need < 2_097_500 * 640 * 2)
    windowed, note = chunked._level_mlp_windows(blocks, 2_097_500, "cpu")
    assert windowed and "do not fit the card now" in note
    monkeypatch.setattr(pf, "box_safe", lambda kind, ci, co, device=None: kind != "mlp2")
    assert chunked._level_mlp_windows(blocks, 2_097_500, "cpu") == (False, "")
    assert chunked._level_mlp_windows(blocks, 900_000, "cpu") == (False, "")


def _planner(monkeypatch: pytest.MonkeyPatch, level: int = 4, reach: Any = 6_000_000, whole: bool = False,
             mlp_safe: bool = True, window: Any = None) -> None:
    """parts' figures on the products path with a plan of `level` and an output reach (no record read)."""
    monkeypatch.setattr(parts, "decode_backend", lambda: "spconv_torch")
    monkeypatch.setattr(parts, "_decodes_in_boxes", lambda voxels=None: True)
    monkeypatch.setattr(pf, "box_level", lambda device=None: level)
    monkeypatch.setattr(pf, "window", lambda device=None: window)
    monkeypatch.setattr(pf, "plan", lambda device=None: {"box_level": level, "whole_only": [], "why": "",
                                                         "output_windows_to": reach, "why_not_3": ""})
    monkeypatch.setattr(pf, "output_may_be_whole", lambda device=None: reach is not None or whole)
    monkeypatch.setattr(pf, "box_safe", lambda kind, ci, co, device=None: mlp_safe or kind not in ("mlp1", "mlp2"))


# decode-v2-REPORT section 4d, the real GTX 1080 Ti, both decoders, above them (MiB): (voxels, box size, the largest
# of level 3 whole, the boxes, the whole output call - shape or texture)
L4_PEAKS = [(3_549_866, 500_000, 1_005), (3_549_866, 2_000_000, 1_520), (10_564_297, 500_000, 3_129),
            (10_564_297, 2_000_000, 3_765), (12_238_871, 500_000, 3_596), (12_238_871, 2_000_000, 4_331)]


def test_the_level_4_forms_figures_hold_the_real_gtx_1080_tis_peaks(monkeypatch: pytest.MonkeyPatch) -> None:
    """parts.decode_figures / decode_mib / chunk_for in the level-4 form (round P's review, kept[6]): the form, the
    whole output call above the reach, decode_mib at or above every measured peak and at most 16 % over; 500 k boxes
    beside the whole output call (the ork under the pessimistic Windows 8 GB cap), 2 M below the reach; a band only
    (reach 0): the call counted for every prop; no reach: never."""
    _planner(monkeypatch)
    figures = parts.decode_figures()
    assert figures["form"] == 4 and figures["whole_out_from"] == 6_000_000 and figures["whole_out_beside_box"]
    assert figures["level3"] == parts.SPCONV_L4_LEVEL3_MIB_PER_KVOXEL["spconv_torch"]
    for voxels, chunk, peak in L4_PEAKS:
        counted = parts.decode_mib(voxels, decoders=2, chunk=chunk) - 2 * parts.DECODER_MIB
        assert peak <= counted <= 1.16 * peak, (voxels, chunk, counted, peak)
    assert parts.chunk_for(10_560_000, 5_761) == parts.SPCONV_L4_WHOLE_OUT_CHUNK_VOXELS == 500_000
    assert parts.chunk_for(5_900_000, 5_761) == parts.CHUNK_VOXELS                 # below the reach: the windows
    assert parts.chunk_for(3_550_000, 5_761) == 0                                  # its whole-table decode fits
    _planner(monkeypatch, reach=None, whole=True)                                  # a band only: any size may call
    assert parts.decode_figures()["whole_out_from"] == 0 and parts.chunk_for(5_900_000, 5_761) == 500_000
    _planner(monkeypatch, reach=None)
    assert not parts.decode_figures()["whole_out"] and parts.chunk_for(10_560_000, 5_761) == parts.CHUNK_VOXELS
    _planner(monkeypatch, mlp_safe=False)                                          # level 3's MLP whole-only
    assert parts.decode_figures()["level3"] == pytest.approx(parts.SPCONV_L4_LEVEL3_MIB_PER_KVOXEL["spconv_torch"]
                                                             + parts.SPCONV_L4_WHOLE_MLP_MIB_PER_KVOXEL)


def test_round_os_form_under_a_reach_counts_the_whole_output_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """kept[0] of round P's review: products_floor.plan_of starts such a card's boxes at level 4 (tested above); a
    record of this round's form that still pairs level 3 with a reach (by hand) has the held rows' whole call counted
    (SPCONV_HOLD_WHOLE_OUT_MIB_PER_KVOXEL above the reach, 2 M boxes, no box beside it): the 1536 tree (13.46 M
    foreseen) no longer fits one decoder under the pessimistic Windows 8 GB cap of 5,761 MiB, as the held rows' 905 +
    0.31 x 13,460 + 100 said it did."""
    _planner(monkeypatch, level=3)
    figures = parts.decode_figures()
    assert figures["form"] == 3 and figures["whole_out"] == parts.SPCONV_HOLD_WHOLE_OUT_MIB_PER_KVOXEL
    assert figures["whole_out_from"] == 6_000_000 and not figures["whole_out_beside_box"] and figures["hold"]
    tree = 13_460_000
    assert parts.chunk_for(tree, 5_761) == parts.CHUNK_VOXELS
    assert parts.decode_mib(tree, 1, parts.CHUNK_VOXELS) == pytest.approx(
        parts.DECODER_MIB + parts.SPCONV_HOLD_WHOLE_OUT_MIB_PER_KVOXEL * tree / 1000 + parts.SPCONV_L4_BASE_MIB)
    assert parts.decode_mib(tree, 1, parts.CHUNK_VOXELS) > 5_761 / 1.1
    assert parts.decode_mib(5_000_000, 1, parts.CHUNK_VOXELS) < parts.DECODER_MIB + 0.45 * 5_000   # below the reach
    _planner(monkeypatch, level=3, reach=None, whole=True)                         # a band only: round O's rule
    assert "whole_out" not in parts.decode_figures() and parts.decode_figures()["hold"]
    _planner(monkeypatch, level=3, window=65536)                                   # the forced windows: no call
    assert "whole_out" not in parts.decode_figures()


def test_park_mesh_and_the_decodes_ram(monkeypatch: pytest.MonkeyPatch) -> None:
    """parts.park_mesh moves the shape decoder's mesh to RAM for the texture decode in the level-4 form's boxes and in
    round O's form under a reach, nowhere else; decode_ram_mib counts the level-3 table of the whole output call's
    boxes (above the reach) and the parked mesh for the finish's RAM test."""
    moved: list = []

    class Table:
        is_cuda = True

        def __init__(self, name: str) -> None:
            self.name = name

        def cpu(self) -> str:
            moved.append(self.name)
            return f"{self.name} in RAM"

    def mesh() -> Any:
        return type("Mesh", (), {"vertices": Table("vertices"), "faces": Table("faces")})()

    _planner(monkeypatch)
    one = mesh()
    assert parts.park_mesh(one, 500_000) and one.vertices == "vertices in RAM" and moved == ["vertices", "faces"]
    assert not parts.park_mesh(mesh(), 0)                                           # whole tables: nothing parked
    tree = 12_238_871
    assert parts.decode_ram_mib(tree, 500_000) == pytest.approx(
        (parts.SPCONV_L4_RAM_LEVEL3_MIB_PER_KVOXEL + parts.SPCONV_L4_RAM_MESH_MIB_PER_KVOXEL) * tree / 1000)
    assert 1_200 < parts.decode_ram_mib(tree, 500_000) < 1_500                       # 0.74 GB + 0.44 GB measured
    assert parts.decode_ram_mib(3_549_866, 2_000_000) == pytest.approx(
        parts.SPCONV_L4_RAM_MESH_MIB_PER_KVOXEL * 3_549_866 / 1000)                 # below the reach: the mesh only
    assert parts.decode_ram_mib(tree, 0) == 0
    _planner(monkeypatch, level=3)
    assert parts.park_mesh(mesh(), 2_000_000)                                       # round O's form under a reach
    assert parts.decode_ram_mib(tree, 2_000_000) == pytest.approx(parts.SPCONV_L4_RAM_MESH_MIB_PER_KVOXEL * tree / 1000)
    _planner(monkeypatch, level=3, reach=None, whole=True)
    assert not parts.park_mesh(mesh(), 2_000_000) and parts.decode_ram_mib(tree, 2_000_000) == 0
    _planner(monkeypatch)
    monkeypatch.setattr(parts, "decode_backend", lambda: "flex_gemm")
    assert not parts.park_mesh(mesh(), 2_000_000) and parts.decode_ram_mib(tree, 2_000_000) == 0
