"""launchers/uv_islands.py: small UV islands laid onto a neighbouring island - flat, at its scale, whole or not at all.

The atlases are made by hand: a flat grid that is one island (its UVs its x, y times SCALE), and small islands
beside it with their split vertices of their own, placed far away, as xatlas leaves a triangle folded against its
neighbours.
"""

from __future__ import annotations

import numpy as np
import pytest

import uv_islands

SIDE = 40                                   # quads along the grid's side: one triangle is well under SMALL_SHARE
SCALE = 1.0 / SIDE                          # the grid's UVs: its x, y times this


def grid():
    """(positions, triangles) of a SIDE x SIDE grid of unit quads in z = 0, wound counter-clockwise from +z."""
    xs, ys = np.meshgrid(np.arange(SIDE + 1), np.arange(SIDE + 1), indexing="ij")
    positions = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1).astype(np.float64)
    tris = []
    for i in range(SIDE):
        for j in range(SIDE):
            a, b, c, d = i * (SIDE + 1) + j, (i + 1) * (SIDE + 1) + j, (i + 1) * (SIDE + 1) + j + 1, i * (SIDE + 1) + j + 1
            tris += [[a, b, c], [a, c, d]]
    return positions, np.asarray(tris, dtype=np.int64)


def vertex(positions, x, y):
    return int(np.flatnonzero((positions[:, 0] == x) & (positions[:, 1] == y))[0])


class Atlas:
    """An unwrapped mesh being put together: the grid as one island, then islands of their own."""

    def __init__(self):
        self.positions, tris = grid()
        self.vmap = list(range(len(self.positions)))
        self.uv = [p[:2] * SCALE for p in self.positions]
        self.tris = [list(t) for t in tris]

    def point(self, xyz) -> int:
        self.positions = np.vstack([self.positions, xyz])
        return len(self.positions) - 1

    def island(self, triangles, places):
        """Triangles over mesh vertices, with split vertices of their own at the UV places given per vertex."""
        own = {}
        for v in sorted({v for t in triangles for v in t}):
            own[v] = len(self.vmap)
            self.vmap.append(v)
            self.uv.append(np.asarray(places[v], dtype=np.float64))
        self.tris += [[own[v] for v in t] for t in triangles]

    def arrays(self):
        return self.positions, np.asarray(self.vmap), np.asarray(self.tris), np.asarray(self.uv)


def charts(tris, count):
    return int(uv_islands.chart_labels(np.asarray(tris), count).max()) + 1


def signed_area(uv, tri):
    a, b, c = uv[tri]
    return 0.5 * ((b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0])


def folded_crumb(at: Atlas):
    """A sliver on the grid's lower border, folded back over the grid: (its triangle over mesh vertices)."""
    a, b = vertex(at.positions, 0, 0), vertex(at.positions, 1, 0)
    p = at.point([0.5, 0.3, 0.05])                      # over the grid, above it: folded ~170 degrees
    tri = [b, a, p]                                      # along a-b the other way from the grid's triangle there
    at.island([tri], {a: (5.0, 5.0), b: (5.2, 5.0), p: (5.1, 5.1)})
    return tri


def test_a_folded_sliver_is_laid_flat_onto_the_island_beside_it() -> None:
    at = Atlas()
    crumb = folded_crumb(at)
    positions, vmap, tris, uv = at.arrays()
    assert charts(tris, len(vmap)) == 2

    new_vmap, new_tris, new_uv, joined = uv_islands.join_small_charts(positions, vmap, tris, uv)

    assert joined == 1 and charts(new_tris, len(new_vmap)) == 1
    assert len(new_tris) == len(tris) and np.array_equal(new_vmap[new_tris], vmap[tris])   # the same triangles
    laid = new_tris[-1]
    assert [int(new_vmap[s]) for s in laid] == crumb
    # Unfolded below the border, at the grid's scale, the triangle's own shape, wound as the island.
    b, a, p = new_uv[laid]
    assert np.allclose(a, [0, 0]) and np.allclose(b, [SCALE, 0])
    height = np.hypot(0.3, 0.05)
    assert np.allclose(p, [0.5 * SCALE, -height * SCALE])
    assert signed_area(new_uv, laid) > 0 and signed_area(new_uv, new_tris[0]) > 0
    assert len(new_vmap) == len(at.positions)            # its old split vertices dropped, one new one for p


def test_an_island_that_would_overlap_stays_as_it_was() -> None:
    """The grid's island also holds a flap (a triangle elsewhere on the model, joined to it at one corner) whose UVs
    lie where the sliver would be unfolded: the sliver keeps its own island."""
    at = Atlas()
    corner = vertex(at.positions, 0, 0)
    x1, x2 = at.point([-1.0, -1.0, -1.0]), at.point([-2.0, -1.0, -1.0])
    flap = [corner, x1, x2]
    at.vmap += [x1, x2]
    at.uv += [np.array([-1.0, -2.0]) * SCALE, np.array([10.0, -2.0]) * SCALE]
    at.tris.append([corner, len(at.vmap) - 2, len(at.vmap) - 1])
    assert signed_area(np.asarray(at.uv), np.asarray(at.tris[-1])) > 0
    folded_crumb(at)
    positions, vmap, tris, uv = at.arrays()

    new_vmap, new_tris, new_uv, joined = uv_islands.join_small_charts(positions, vmap, tris, uv)

    assert joined == 0 and charts(new_tris, len(new_vmap)) == 2
    assert np.array_equal(new_vmap[new_tris], vmap[tris]) and np.allclose(new_uv[new_tris], uv[tris])
    assert flap == [int(v) for v in vmap[tris[-2]]]


def test_a_crumb_beside_a_crumb_follows_it_in_the_next_pass() -> None:
    at = Atlas()
    b, a, p = folded_crumb(at)
    q = at.point([0.2, 0.1, 0.1])                        # a smaller sliver on the first one's edge a-p
    at.island([[a, q, p]], {a: (7.0, 7.0), q: (7.05, 7.0), p: (7.0, 7.05)})
    positions, vmap, tris, uv = at.arrays()
    assert charts(tris, len(vmap)) == 3

    new_vmap, new_tris, new_uv, joined = uv_islands.join_small_charts(positions, vmap, tris, uv)

    assert joined == 2 and charts(new_tris, len(new_vmap)) == 1
    assert all(signed_area(new_uv, t) > 0 for t in new_tris)


def test_a_flat_triangle_in_a_notch_takes_the_islands_own_corners() -> None:
    """A triangle of the grid given an island of its own: laid back, its corners are the island's - no new split
    vertex, no slit."""
    at = Atlas()
    alone = at.tris.pop(2 * (SIDE * 5 + 5))              # a triangle in the grid's middle
    corners = [at.vmap[s] for s in alone]
    at.island([corners], {v: (9.0 + i, 9.0) for i, v in enumerate(corners)})
    positions, vmap, tris, uv = at.arrays()
    assert charts(tris, len(vmap)) == 2

    new_vmap, new_tris, new_uv, joined = uv_islands.join_small_charts(positions, vmap, tris, uv)

    assert joined == 1 and charts(new_tris, len(new_vmap)) == 1
    assert len(new_vmap) == len(positions)               # every mesh vertex once: the notch is closed
    assert np.allclose(new_uv, positions[new_vmap][:, :2] * SCALE)


def test_xatlas_packs_a_joined_sliver_with_its_island_in_packing_units() -> None:
    """The real xatlas, packing again from the joined UVs, keeps a sliver too thin for FLT_EPSILON in a 0..1 atlas
    in its island when given packing_units - and charts it anew when given 0..1 (why packing_units exists)."""
    xatlas = pytest.importorskip("xatlas")
    at = Atlas()
    a, b = vertex(at.positions, 0, 0), vertex(at.positions, 1, 0)
    p = at.point([0.5, 2e-4, 1e-4])                    # its laid area: about 7e-8 of the 0..1 atlas
    at.island([[b, a, p]], {a: (5.0, 5.0), b: (5.2, 5.0), p: (5.1, 5.1)})
    positions, vmap, tris, uv = at.arrays()
    new_vmap, new_tris, new_uv, joined = uv_islands.join_small_charts(positions, vmap, tris, uv)
    assert joined == 1

    def packed_charts(given):
        atlas = xatlas.Atlas()
        atlas.add_mesh(np.ascontiguousarray(positions[new_vmap], dtype=np.float32),
                       np.ascontiguousarray(new_tris, dtype=np.uint32), uvs=np.ascontiguousarray(given, dtype=np.float32))
        options = xatlas.ChartOptions()
        options.use_input_mesh_uvs = True
        atlas.generate(chart_options=options)
        vm, idx, _ = atlas[0]
        assert np.array_equal(new_vmap[np.asarray(vm, dtype=np.int64)][np.asarray(idx, dtype=np.int64)],
                              new_vmap[new_tris])
        return charts(np.asarray(idx, dtype=np.int64), len(vm))

    scaled = uv_islands.packing_units(new_uv)
    assert np.isclose((scaled.max(axis=0) - scaled.min(axis=0)).max(), uv_islands.PACK_SPAN)
    assert packed_charts(scaled) == 1
    assert packed_charts(new_uv) == 2


def test_nothing_small_changes_nothing() -> None:
    at = Atlas()
    positions, vmap, tris, uv = at.arrays()
    new_vmap, new_tris, new_uv, joined = uv_islands.join_small_charts(positions, vmap, tris, uv)
    assert joined == 0
    assert np.array_equal(new_vmap, vmap) and np.array_equal(new_tris, tris) and np.array_equal(new_uv, uv)
