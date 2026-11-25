"""The high poly as LATO.2 is handed it, for lato2_server.py: creases closed when they would confuse it.

LATO.2's topology flow sees vertex positions and a coarse (64^3) voxel field, not the surface, so it has to guess
"neighbours along the surface" from closeness in space. A Pixal3D mesh is closed and wound outward but busy:
fur, scutes, the slot between a limb and the body fold the surface back on itself more tightly than 2,000
vertices are spaced, and across such a crease "the next vertex along the surface" and "the vertex on the far
side" look alike. Measured on the user's branches (a Pixal3D turtle, 2,000 vertices): 40.6% of the predicted
edges jumped a crease (LATO.2's own crocodile: 1.1%), leaving 18.7% of the edges on hole boundaries.

crowding() measures that: the share of the surface with an opposite-facing surface within one vertex spacing
(99.9% on a Pixal3D export at 2,000 vertices, 16.9% on the crocodile). solidify() rebuilds the surface as a solid
at about that spacing: voxelised, closed morphologically (merges near-touching sheets, seals small tunnels),
opened (drops spikes too thin to hold a vertex) and contoured with surface nets - closed, genus ~0 and wound
outward by construction. At 2,000 vertices a crease cannot be represented anyway, so closing it costs little
accuracy and stops the guessing. Measured here (RTX 3060, 2,000 vertices, seeds 0-1) on a furry Pixal3D bear
(crowding 58%): open edges 15.2% / 15.4% as it is -> 0.17% / 0.06% solidified, faces facing like the surface 90% ->
99.7%, Euler number -627 -> -5; the solid itself is 0.25% of the diagonal (Chamfer) from the original.

Two things the branch's version got wrong are fixed here: the solid was shifted by half a voxel (field sample i is
the centre of voxel i, not its corner) and inflated by another half (every voxel the surface passes through is
solid): both measured on a sphere, now +0.06 voxel. And it deleted thin parts (the opening removes anything
under ~2.6% of the prop's size: a sword's blade) and separate pieces without a word: solidify() now reports the
share of the surface it lost, and lato2_server.py keeps the mesh as it is when that is over LOST_LIMIT (a sword:
55%; the bear 0.2%, LATO.2's crocodile 0.7%).

Ported (MIT, the user's own code) from the Pixal3D-stableprojectorz branch llm-experiments-sept2026
(mesh_solidify.py). numpy / scipy / trimesh at call time only.
"""

from __future__ import annotations

from typing import Any

# Solidify only above this crowding: a clean mesh keeps its detail (the branch's threshold). A Pixal3D ogre head at
# 19% came out with 3.3% open edges as it is (closed afterwards by lato2_faces.close_holes) and 1% solidified, but
# 0.25% of the diagonal from the original instead of 0.37%: kept as it is.
CROWDING_LIMIT = 0.40
# The solid's voxel resolution, and the closing radius in voxels scales with it (resolution / 128). Measured on
# the branch: coarser loses shape (Chamfer 0.53% of the diagonal at 128, 0.40% at 192); finer lets the detail
# back in (a handle survives at 224, boundary edges back up at 256). 192 gave genus 0 on a Pixal3D export.
SOLID_RESOLUTION = 192
INFLATION_VOXELS = 0.6       # how far the solid's skin is moved back in (see solidify)
# The solid is not used when it leaves more than LOST_LIMIT of the input's surface over LOST_VOXELS voxels
# outside it: the opening deletes whatever is thinner than ~2.6% of the prop's size - a sword's blade, a staff -
# and only the largest piece is kept. Better holes in the clean mesh than a missing blade.
LOST_VOXELS = 2.0
LOST_LIMIT = 0.03


def crowding(vertices: Any, faces: Any, vert_num: int, seed: int = 0) -> float:
    """Share of the surface that has an oppositely facing surface within one vertex spacing (at `vert_num`
    vertices). About 9 samples per vertex: the 32 nearest then reach just over one spacing."""
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree

    mesh = _mesh(vertices, faces)
    n = int(np.clip(9 * max(vert_num, 1), 20_000, 400_000))
    pts, face_id = trimesh.sample.sample_surface(mesh, n, seed=seed)
    nrm = np.asarray(mesh.face_normals, dtype=np.float64)[face_id]
    spacing = float(np.sqrt(max(mesh.area, 1e-12) / max(vert_num, 1)))
    dist, idx = cKDTree(pts).query(pts, k=min(32, n))
    opposed = ((nrm[idx] * nrm[:, None, :]).sum(-1) < -0.5) & (dist < spacing)
    return float(opposed.any(1).mean())


def solidify(vertices: Any, faces: Any, resolution: int = SOLID_RESOLUTION, close_radius: int | None = None,
             open_radius: int | None = None, smooth: float = 1.2, samples: int = 4_000_000,
             seed: int = 0) -> tuple[Any, Any, float]:
    """A closed, low-genus version of the mesh at voxel `resolution` -> (vertices, faces, lost), wound outward, in
    the input's frame. close_radius is in voxels (None: resolution / 128, a fixed physical distance); open_radius
    too (None: the same, 0: no opening). lost: the share of the input's surface left more than LOST_VOXELS outside
    the solid - what the opening and keeping the largest piece took away (thin blades, spikes, separate parts);
    see LOST_LIMIT and prepare()."""
    import numpy as np
    import scipy.ndimage as ndi
    import trimesh

    mesh = _mesh(vertices, faces)
    if close_radius is None:
        close_radius = max(1, int(round(resolution / 128.0)))
    if open_radius is None:
        open_radius = close_radius
    pad = max(close_radius, open_radius) + 2
    grid, lo, pitch, pts = _occupancy(mesh, resolution, pad, samples, seed)
    if close_radius:
        grid = ndi.binary_closing(grid, _ball(close_radius))
    if open_radius:
        grid = ndi.binary_opening(grid, _ball(open_radius))
    grid = ndi.binary_fill_holes(grid)
    labels, count = ndi.label(grid)
    if count == 0:
        raise RuntimeError("solidify: nothing left after closing and opening")
    if count > 1:       # the largest solid piece only (LATO.2 would scatter vertices over crumbs)
        grid = labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1)

    # how much of the input the solid lost: samples in empty voxels far from any solid one
    outside = ndi.distance_transform_edt(~grid)
    probe = pts[:: max(1, len(pts) // 200_000)]
    cells = np.clip(np.floor((probe - lo) / pitch).astype(np.int64) + pad, 0, np.asarray(grid.shape) - 1)
    lost = float((outside[tuple(cells.T)] > LOST_VOXELS).mean())

    field = ndi.gaussian_filter(grid.astype(np.float32), smooth)
    verts, tris = surface_nets(field, 0.5)
    if len(tris) == 0:
        raise RuntimeError("solidify: the surface extraction produced no faces")
    # field sample i is the centre of voxel i - pad, i.e. lo + (i - pad + 0.5) * pitch
    out = trimesh.Trimesh((verts - pad + 0.5) * pitch + lo, tris, process=False)
    # no merge_vertices: surface nets already share vertices, and merging by position would weld the pair made
    # where the solid pinches diagonally - the only way this goes non-manifold
    out.remove_unreferenced_vertices()
    parts = out.split(only_watertight=False)
    if len(parts) > 1:
        out = max(parts, key=lambda c: len(c.faces))
    faces_out = np.asarray(out.faces)
    if out.volume < 0:      # closed and consistently wound, so the sign settles which way is out
        faces_out = faces_out[:, ::-1]
    # a new mesh, not out.invert(): after split() trimesh keeps the old face normals cached through invert()
    out = trimesh.Trimesh(np.asarray(out.vertices), np.ascontiguousarray(faces_out), process=False)
    # Every voxel the surface passes through counts as solid, so the solid's skin sits outside the surface (+0.6
    # voxel measured on a sphere): moved back in along the normals.
    moved = np.asarray(out.vertices) - np.asarray(out.vertex_normals) * (INFLATION_VOXELS * pitch)
    return moved, np.asarray(out.faces), lost


def prepare(vertices: Any, faces: Any, vert_num: int, mode: str = "auto", seed: int = 0,
            log: Any = print) -> tuple[Any, Any, str, float | None]:
    """The mesh LATO.2 should be handed -> (vertices, faces, what was done, crowding or None).

    mode "off": as it is. "auto": solidified when crowding() is at least CROWDING_LIMIT, "always": solidified. The
    solid is used only when it loses at most LOST_LIMIT of the surface; otherwise a second one is tried that closes
    twice as far and does not open (thin walls and parts stay, as slabs of a few voxels: a lantern's glass panels
    and handle - the first solid kept only its frame, 38% lost; this one 0%); when that loses too much as well, the
    mesh is used as it is."""
    if mode == "off":
        return vertices, faces, "as is", None
    crowd = crowding(vertices, faces, vert_num, seed=seed)
    if mode != "always" and crowd < CROWDING_LIMIT:
        return vertices, faces, "as is", crowd
    default = max(1, int(round(SOLID_RESOLUTION / 128.0)))
    for how, close, opening in (("solidified", default, default), ("solidified, thin parts kept", 2 * default, 0)):
        try:
            solid_v, solid_f, lost = solidify(vertices, faces, close_radius=close, open_radius=opening, seed=seed)
        except Exception as exc:      # the raw surface still works, only less cleanly
            log(f"[spellcast] Could not close the creases ({type(exc).__name__}: {exc}).")
            continue
        if lost <= LOST_LIMIT:
            return solid_v, solid_f, how, crowd
        log(f"[spellcast] Closing the creases ({how}) would lose {lost:.1%} of the surface (thin or separate "
            "parts).")
    return vertices, faces, "as is (closing the creases would lose parts)", crowd


def surface_nets(field: Any, iso: float = 0.5) -> tuple[Any, Any]:
    """Naive surface nets: one vertex per crossed cell, two triangles per crossed grid edge (manifold by
    construction: every interior edge has the four cells around it)."""
    import numpy as np

    solid = field >= iso
    cs = tuple(n - 1 for n in field.shape)
    if min(cs) < 1:
        raise ValueError("field too small to contour")

    def corner(arr: Any, d: Any) -> Any:
        return arr[d[0]:d[0] + cs[0], d[1]:d[1] + cs[1], d[2]:d[2] + cs[2]]

    count = np.zeros(cs, dtype=np.int8)
    for d in np.ndindex(2, 2, 2):
        count += corner(solid, d)
    active = (count > 0) & (count < 8)

    acc = np.zeros(cs + (3,), dtype=np.float32)
    hits = np.zeros(cs, dtype=np.float32)
    for axis in range(3):
        for off in np.ndindex(2, 2):
            a = [0, 0, 0]
            a[(axis + 1) % 3], a[(axis + 2) % 3] = off
            b = list(a)
            b[axis] = 1
            fa, fb = corner(field, a).astype(np.float32), corner(field, b).astype(np.float32)
            crossed = (fa < iso) != (fb < iso)
            denom = fb - fa
            t = np.clip(np.where(np.abs(denom) > 1e-12,
                                 (iso - fa) / np.where(np.abs(denom) > 1e-12, denom, 1.0), 0.5), 0.0, 1.0)
            for k in range(3):
                acc[..., k] += np.where(crossed, (t if k == axis else float(a[k])), 0.0)
            hits += crossed

    cell_idx = np.argwhere(active)
    vert_id = np.full(cs, -1, dtype=np.int64)
    vert_id[active] = np.arange(len(cell_idx))
    verts = cell_idx.astype(np.float64) + acc[active] / np.maximum(hits[active], 1.0)[:, None]

    quads = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[axis] = slice(1, -1)
        sl_hi[axis] = slice(2, None)
        for d in (u, v):
            sl_lo[d] = sl_hi[d] = slice(1, -1)
        a = solid[tuple(sl_lo)]
        b = solid[tuple(sl_hi)]
        crossed = a != b
        if not crossed.any():
            continue
        base = np.argwhere(crossed)       # the four cells around this grid edge, in order
        gi = base.copy()
        gi[:, axis] += 1
        gi[:, u] += 1
        gi[:, v] += 1
        ring = np.empty((len(gi), 4), dtype=np.int64)
        for n, (du, dv) in enumerate(((-1, -1), (0, -1), (0, 0), (-1, 0))):
            c = gi.copy()
            c[:, u] += du
            c[:, v] += dv
            ring[:, n] = vert_id[c[:, 0], c[:, 1], c[:, 2]]
        keep = (ring >= 0).all(1)
        ring = ring[keep]
        flip = a[tuple(base[keep].T)]      # the solid on the other side: reverse, so the winding stays outward
        ring[flip] = ring[flip][:, ::-1]
        quads.append(ring)
    if not quads:
        return verts, np.empty((0, 3), dtype=np.int64)
    quads = np.concatenate(quads, axis=0)
    return verts, np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0)


def _mesh(vertices: Any, faces: Any) -> Any:
    import numpy as np
    import trimesh

    return trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces, dtype=np.int64),
                           process=False)


def _ball(radius: int) -> Any:
    import numpy as np

    if radius <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    g = np.indices((2 * radius + 1,) * 3) - radius
    return np.linalg.norm(g, axis=0) <= radius


def _occupancy(mesh: Any, res: int, pad: int, samples: int, seed: int) -> tuple[Any, Any, float]:
    """Solid occupancy grid from a dense surface sampling; the dilate / fill / erode around the fill keeps a
    missed voxel from becoming a pinhole that empties the solid."""
    import numpy as np
    import scipy.ndimage as ndi
    import trimesh

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    lo = verts.min(0)
    pitch = float((verts.max(0) - lo).max()) / max(res - 1, 1)
    pts, _ = trimesh.sample.sample_surface(mesh, samples, seed=seed)
    idx = np.floor((pts - lo) / pitch).astype(np.int64) + pad
    shape = tuple(int(x) for x in (np.floor((verts.max(0) - lo) / pitch) + 2 * pad + 2))
    idx = np.clip(idx, 0, np.asarray(shape) - 1)
    grid = np.zeros(shape, dtype=bool)
    grid[tuple(idx.T)] = True
    one = _ball(1)
    grid = ndi.binary_dilation(grid, one)
    grid = ndi.binary_fill_holes(grid)
    grid = ndi.binary_erosion(grid, one)
    return grid, lo, pitch, pts
