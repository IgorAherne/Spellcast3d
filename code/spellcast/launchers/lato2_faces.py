"""LATO.2's faces, recovered so the clean mesh is manifold and wound outward, for lato2_server.py.

LATO.2 makes a mesh in two steps: a vertex flow places the vertices, then a topology flow scores every pair of
them (TopologyVAE's connection predictor: one logit per pair, > 0 = "an edge"). Upstream then keeps the pairs
over 0 and makes a triangle of every 3-clique of that graph, its indices sorted (utils/inference.py
edges_to_faces). A 3-clique is not a face and sorted indices are no winding: measured on a Pixal3D export at
2000 vertices, 73% of the edges were shared by more than two faces and 49.9% of the faces faced inward
(LATO.2's own crocodile: 19.5% and 50.4%). Under backface culling that reads as a mesh full of holes, and
nothing in an OBJ can repair it.

Here the faces are traced from the graph as a surface embedded in 3D (its "rotation system"):

  1. Orient: every generated vertex gets the outward normal of the high poly at that spot (nearest samples).
  2. Filter: drop candidate edges no surface edge can be (joining vertices whose normals oppose, far out of both
     tangent planes, or much longer than the typical edge).
  3. Rotate: each vertex's neighbours, sorted by angle in its tangent plane: counter-clockwise seen from outside.
  4. Propose: every vertex proposes the triangle (v, a_k, a_k+1) for each adjacent pair of neighbours - already
     wound counter-clockwise from outside; the edge (a_k, a_k+1) need not have been predicted.
  5. Vote: a true face is proposed by all three corners, a spurious one mostly by one.
  6. Assemble: faces are accepted by votes, then by the weakest edge's logit, while every DIRECTED edge is used at
     most once - the half-edge rule: each edge ends up with at most two faces running it in opposite
     directions, so the result is edge-manifold and consistently wound by construction.
  7. Close: the rotation is rebuilt without edges that won no face, then the remaining boundary is grown from
     (advancing front) and small holes are ear-clipped, all under the same rule.

Then (added here, lato2_server.py calls them): drop_islands() removes the crumbs of a few faces left beside the
surface, close_holes() closes the loops the steps above left open, and SurfaceSamples.missing_share() notices a
part of the model the clean mesh left out. Measured on an RTX 3060 at 2,000 vertices (Pixal3D bear, ogre head,
sheep; LATO.2's crocodile): no non-manifold edge, no open edge, every edge run both ways (consistent winding),
94-100% of the faces facing like the high poly next to them (the rest are folds, not flipped windings), Chamfer
distance to the high poly 0.20-0.32% of its diagonal.

Ported (MIT, the user's own code) from the Pixal3D-stableprojectorz branch llm-experiments-sept2026
(lato2_topology.py, lato2_topology_patch.py); pair_logits replaces the patched TopologyVAE.decode and scores the
pairs row block by row block instead of building torch.triu_indices (800 MB at 10,000 vertices).

numpy / scipy / trimesh at call time only (torch inside pair_logits): the hub's tests import this file without them.
"""

from __future__ import annotations

from typing import Any

# Pairs scoring below this logit are not kept: by a vertex's 15th-best partner the mean logit is already -13.6
# (measured on one asset), well under TopologyParams.min_logit, so nothing the recovery uses is lost.
KEEP_LOGIT = -8.0
MAX_KEPT_PER_VERTEX = 48
PAIRS_PER_BLOCK = {2: 1 << 17, 4: 1 << 16}   # element size of the predictor's dtype -> pairs scored per call


class TopologyParams:
    """Tunables of build_faces (the values swept on the user's branch, _topo_eval.py)."""

    # --- candidate edge set ---
    max_degree = 14          # keep at most this many neighbours per vertex, by logit
    min_logit = -6.0         # and only those scoring above this
    # --- geometric admissibility ---
    min_normal_dot = -0.2    # reject edges joining vertices whose normals oppose
    max_normal_align = 0.86  # reject edges >~60 deg out of both tangent planes
    max_len_factor = 3.0     # reject edges longer than this * median candidate length
    # --- triangle admissibility ---
    min_tri_angle_deg = 1.0  # reject slivers
    max_tri_tilt = 0.30      # reject faces >~72 deg off the local surface normal
    # --- assembly ---
    min_votes = 1            # lowest vote count still eligible
    refine_rounds = 2        # rebuild the rotation system with dead edges removed
    front_rounds = 6         # advancing-front passes closing the boundary
    fill_holes = True
    max_hole_len = 24        # ear-clip boundary loops up to this many edges


class EdgeLogits:
    """The pair logits that matter, standing in for the full N x N matrix: each vertex's strongest partners
    (pair_logits keeps them); any other pair reads as `floor`. Answers the two questions build_faces asks: the
    top-k partners of every vertex, and the logit of arbitrary pairs (vectorised)."""

    def __init__(self, n: int, rows: Any, cols: Any, vals: Any, floor: float = -30.0) -> None:
        import numpy as np

        self.n = int(n)
        self.floor = float(floor)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        vals = np.asarray(vals, dtype=np.float32)
        key = np.concatenate([rows * self.n + cols, cols * self.n + rows])
        val = np.concatenate([vals, vals])
        order = np.argsort(key, kind="stable")
        key, val = key[order], val[order]
        first = np.concatenate([[True], key[1:] != key[:-1]]) if len(key) else np.zeros(0, dtype=bool)
        self._key, self._val = key[first], val[first]

    def __len__(self) -> int:
        return len(self._key) // 2

    def __getitem__(self, idx: tuple[Any, Any]) -> Any:
        import numpy as np

        a, b = idx
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        if len(self._key) == 0:
            out = np.full(np.broadcast(a, b).shape, self.floor, dtype=np.float32)
            return out if out.ndim else float(out)
        q = a * self.n + b
        pos = np.clip(np.searchsorted(self._key, q), 0, len(self._key) - 1)
        out = np.where(self._key[pos] == q, self._val[pos], self.floor).astype(np.float32)
        return out if out.ndim else float(out)

    def candidate_pairs(self, k: int, min_logit: float) -> tuple[Any, Any]:
        """The stored pairs, trimmed to the k strongest per row above `min_logit`."""
        import numpy as np

        rows = (self._key // self.n).astype(np.int64)
        cols = (self._key % self.n).astype(np.int64)
        keep = self._val > min_logit
        rows, cols, vals = rows[keep], cols[keep], self._val[keep]
        order = np.lexsort((-vals, rows))
        rows, cols = rows[order], cols[order]
        rank = np.arange(len(rows)) - np.searchsorted(rows, rows)   # rows is sorted: position within its row
        sel = rank < k
        return rows[sel], cols[sel]


# ---- the topology flow's pair scores -----------------------------------------------------------------------

def pair_logits(tvae: Any, z: Any, verts: Any, verts_mask: Any, keep_logit: float = KEEP_LOGIT,
                max_per_vertex: int = MAX_KEPT_PER_VERTEX) -> list[EdgeLogits]:
    """TopologyVAE.decode (models/topo_autoencoder.py) up to its pair scores, keeping each pair's logit instead of
    thresholding it: the decoder and connection blocks unchanged, then the connection predictor on every pair
    u < v, scored in row blocks of about PAIRS_PER_BLOCK pairs. One EdgeLogits per batch item. Run it under the
    same autocast as upstream (bfloat16 on RTX 30xx+)."""
    import numpy as np
    import torch

    verts_feat = tvae.decoder(z=z, verts_mask=verts_mask)
    rope = tvae.connection_rope(verts.long())
    for block in tvae.connection_transformer_blocks:
        verts_feat = block(verts_feat, verts_mask, rope_phases=rope)

    out: list[EdgeLogits] = []
    for b in range(verts_mask.shape[0]):
        feats = verts_feat[b][verts_mask[b]]
        n = int(feats.shape[0])
        per_block = PAIRS_PER_BLOCK.get(feats.element_size(), 1 << 16)
        keep_u, keep_v, keep_l = [], [], []
        start = 0
        while start < n - 1:
            rows = max(1, per_block // (n - start))
            stop = min(n - 1, start + rows)
            r = torch.arange(start, stop, device=feats.device)
            c = torch.arange(start + 1, n, device=feats.device)
            u = r.repeat_interleave(len(c))
            v = c.repeat(len(r))
            upper = v > u
            u, v = u[upper], v[upper]
            logits = tvae.connection_predictor(feats[u].unsqueeze(0), feats[v].unsqueeze(0)).squeeze(0).float()
            keep = logits > keep_logit
            if bool(keep.any()):
                keep_u.append(u[keep])
                keep_v.append(v[keep])
                keep_l.append(logits[keep])
            start = stop
        if keep_u:
            ku, kv, kl = torch.cat(keep_u), torch.cat(keep_v), torch.cat(keep_l)
            cap = max_per_vertex * n
            if kl.numel() > cap:        # a pathological graph: keep the strongest
                top = torch.topk(kl, cap).indices
                ku, kv, kl = ku[top], kv[top], kl[top]
            out.append(EdgeLogits(n, ku.cpu().numpy(), kv.cpu().numpy(), kl.cpu().numpy()))
        else:
            out.append(EdgeLogits(n, np.empty(0), np.empty(0), np.empty(0)))
    return out


# ---- normals ----------------------------------------------------------------------------------------------

class SurfaceSamples:
    """Area-uniform samples of the high poly with their face normals, and a KD-tree over them: the normal of the
    surface near any point, and whether a triangle faces the same way. The mesh must be wound outward (the hub's
    GLBs are); inside_out() tells when most of it is not."""

    def __init__(self, vertices: Any, faces: Any, count: int = 400_000, seed: int = 0) -> None:
        import numpy as np
        import trimesh
        from scipy.spatial import cKDTree

        mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64),
                               faces=np.asarray(faces, dtype=np.int64), process=False)
        self.points, face_id = trimesh.sample.sample_surface(mesh, count, seed=seed)
        self.normals = np.asarray(mesh.face_normals, dtype=np.float64)[face_id]
        self.tree = cKDTree(self.points)
        self._mesh = mesh

    def inside_out(self) -> bool:
        """True when the mesh's winding points inward overall (negative signed volume)."""
        try:
            return float(self._mesh.volume) < 0.0
        except Exception:
            return False

    def flip(self) -> None:
        self.normals = -self.normals

    def normals_at(self, points: Any, k: int = 6) -> Any:
        """Outward unit normal of the surface near each point: the distance-weighted mean of the k nearest sample
        normals (smooth: it sets the plane a vertex's neighbours are sorted in), or the nearest one's where they
        cancel (both sides of a thin wall)."""
        import numpy as np

        dist, idx = self.tree.query(np.asarray(points, dtype=np.float64), k=min(k, len(self.points)))
        if idx.ndim == 1:
            idx, dist = idx[:, None], dist[:, None]
        w = 1.0 / np.maximum(dist, 1e-9)
        acc = (self.normals[idx] * w[..., None]).sum(1)
        degenerate = np.linalg.norm(acc, axis=1) < 1e-9
        acc[degenerate] = self.normals[idx[degenerate, 0]]
        return _normalize(acc)

    def outward_share(self, vertices: Any, faces: Any) -> float:
        """Share of the triangles whose normal agrees with the high poly's nearest to their centre."""
        import numpy as np

        tris = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
        if not len(tris):
            return 0.0
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        idx = self.tree.query(tris.mean(1))[1]
        return float(((n * self.normals[idx]).sum(1) > 0).mean())

    def missing_share(self, vertices: Any, faces: Any, distance: float, count: int = 100_000) -> float:
        """Share of this surface farther than `distance` from the mesh (vertices, faces): the parts the mesh left
        out. Measured at one vertex spacing on RTX 3060 runs: good clean meshes 0-1.6%; one that lost an ogre's
        neck 10%, a lantern's side 13%, a sword's blade 40-48%."""
        import numpy as np
        import trimesh
        from scipy.spatial import cKDTree

        mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64),
                               faces=np.asarray(faces, dtype=np.int64), process=False)
        if len(mesh.faces) == 0:
            return 1.0
        probe = self.points[:: max(1, len(self.points) // count)]
        covered = cKDTree(trimesh.sample.sample_surface(mesh, count, seed=0)[0])
        return float((covered.query(probe)[0] > distance).mean())

    @property
    def area(self) -> float:
        return float(self._mesh.area)


def vertex_normals_from_points(points: Any, k: int = 16) -> Any:
    """Normals without a high poly: PCA over each neighbourhood, signs made consistent over a minimum spanning tree
    (Hoppe et al.), seeded at the vertex furthest along the widest axis (whose outward direction is known)."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import depth_first_order, minimum_spanning_tree
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    n_pts = len(points)
    k = min(k, n_pts - 1)
    _, idx = cKDTree(points).query(points, k=k + 1)
    nbr = points[idx[:, 1:]] - points[:, None, :]
    cov = np.einsum("nki,nkj->nij", nbr, nbr)
    normals = np.linalg.eigh(cov)[1][:, :, 0]

    rows = np.repeat(np.arange(n_pts), k)
    cols = idx[:, 1:].ravel()
    w = 1.0 - np.abs((normals[rows] * normals[cols]).sum(1)) + 1e-6
    mst = minimum_spanning_tree(coo_matrix((w, (rows, cols)), shape=(n_pts, n_pts)))
    mst = mst + mst.T
    axis = int(np.argmax(points.max(0) - points.min(0)))
    seed = int(np.argmax(points[:, axis]))
    if normals[seed, axis] < 0:
        normals[seed] *= -1.0
    order, preds = depth_first_order(mst, seed, directed=False)
    for node in order[1:]:
        p = preds[node]
        if p >= 0 and float(normals[node] @ normals[p]) < 0.0:
            normals[node] *= -1.0
    return _normalize(normals)


# ---- face recovery -----------------------------------------------------------------------------------------

def build_faces(points: Any, logits: Any, normals: Any, params: TopologyParams | None = None,
                verbose: bool = False) -> Any:
    """(N, 3) vertex positions, their pair logits (EdgeLogits or a dense N x N array) and outward unit normals ->
    (F, 3) int32 triangles, counter-clockwise seen from outside, edge-manifold (see the module doc)."""
    import numpy as np

    p = params or TopologyParams()
    points = np.asarray(points, dtype=np.float64)
    normals = _normalize(np.asarray(normals, dtype=np.float64))
    if not isinstance(logits, EdgeLogits):
        logits = np.asarray(logits, dtype=np.float32)
    n = len(points)

    edges = _geometric_filter(points, normals, _candidate_edges(logits, p), p)
    t1, t2 = _tangent_basis(normals)

    faces = np.empty((0, 3), dtype=np.int64)
    used: set[tuple[int, int]] = set()
    for rnd in range(p.refine_rounds + 1):
        adj = _adjacency(edges, n)
        votes = _umbrella_proposals(points, adj, t1, t2)
        if not votes:
            break
        tris = np.asarray(list(votes.keys()), dtype=np.int64)
        counts = np.asarray(list(votes.values()), dtype=np.int64)
        keep = (counts >= p.min_votes) & _triangle_quality(points, normals, tris, p)
        tris, counts = tris[keep], counts[keep]
        if not len(tris):
            break
        # votes first; the weakest of the three edges' logits breaks ties
        edge_logit = np.minimum.reduce([logits[tris[:, 0], tris[:, 1]], logits[tris[:, 1], tris[:, 2]],
                                        logits[tris[:, 2], tris[:, 0]]]).astype(np.float64)
        faces, used = _assemble(tris, counts * 1000.0 + np.clip(edge_logit, -30.0, 30.0))
        if verbose:
            print(f"   round {rnd}: {len(edges)} edges, {len(tris)} proposals "
                  f"(votes {np.bincount(counts, minlength=4)[1:4].tolist()}), {len(faces)} faces", flush=True)
        if rnd == p.refine_rounds:
            break
        # edges that won no face were noise in the rotation: drop them, the neighbours close the gap next round
        live = set()
        for a, b, c in faces:
            live.update(((min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))))
        nxt = np.asarray([e for e in map(tuple, edges) if e in live], dtype=np.int64)
        if len(nxt) == len(edges) or not len(nxt):
            break
        edges = nxt.reshape(-1, 2)

    if p.fill_holes and len(faces):
        faces, grown = _advance_front(points, normals, logits, _adjacency(edges, n), faces, used, p)
        loops = _boundary_loops(used, p.max_hole_len)
        extra = _fill_loops(points, normals, logits, loops, used, p)
        if verbose:
            print(f"   advancing front +{grown}, ear-clipped {len(loops)} loops -> +{len(extra)} faces", flush=True)
        if extra:
            faces = np.concatenate([faces, np.asarray(extra, dtype=np.int64)], axis=0)
    return faces.astype(np.int32)


def close_holes(points: Any, faces: Any, max_len: int = 48) -> tuple[Any, Any, int]:
    """Close the boundary loops of up to max_len edges that build_faces left open, under the same half-edge rule
    (so the mesh stays edge-manifold and consistently wound) -> (points, faces, how many loops were closed).

    build_faces' own ear clipping (_fill_loops) takes an ear only when it faces the way the high poly does near its
    corners, which a hole across a fold or at a thin part rarely offers: measured on a Pixal3D ogre head at 2,000
    vertices, 39 loops of 4-17 edges stayed open (3.3% of the edges). Here each loop (a loop that touches itself is
    split into simple ones first) is ear-clipped in its own best-fit plane (its Newell normal) by the classic 2D
    rule - an ear is a convex corner with no other corner of the loop inside it - wound against the loop, so the
    patch faces like the faces around it. Where that cannot finish (about half those loops: their corners are
    already joined the same way elsewhere, e.g. across a fold), the loop is closed by a fan around a new vertex at
    its centre, whose edges are all new. Measured on the same head: 3.3% -> 0.0% open edges."""
    import numpy as np

    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    points = np.asarray(points, dtype=np.float64)
    used = set(map(tuple, np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).tolist()))
    new: list[tuple[int, int, int]] = []
    centres: list[Any] = []
    closed = 0
    for loop in _boundary_loops(used, max_len):
        for ring in _simple_rings(list(reversed(loop))):     # the patch runs against the faces already there
            if len(ring) < 3:
                continue
            patch = _clip_ring(points, ring, used)
            if patch is None:
                centre = len(points) + len(centres)
                centres.append(points[ring].mean(axis=0))
                patch = [(ring[i], ring[(i + 1) % len(ring)], centre) for i in range(len(ring))]
                used.update(e for a, b, c in patch for e in ((a, b), (b, c), (c, a)))
            new.extend(patch)
            closed += 1
    if centres:
        points = np.concatenate([points, np.asarray(centres)], axis=0)
    if new:
        faces = np.concatenate([faces, np.asarray(new, dtype=np.int64)], axis=0)
    return points, faces.astype(np.int32), closed


def _simple_rings(ring: list[int]) -> list[list[int]]:
    """A ring that passes a vertex more than once, cut into rings that do not."""
    out: list[list[int]] = []
    stack: list[int] = []
    seen: dict[int, int] = {}
    for vid in ring:
        if vid in seen:
            start = seen[vid]
            out.append(stack[start:])
            for gone in stack[start + 1:]:
                seen.pop(gone, None)
            del stack[start + 1:]
        else:
            seen[vid] = len(stack)
            stack.append(vid)
    if stack:
        out.append(stack)
    return out


def _clip_ring(points: Any, ring: list[int], used: set[tuple[int, int]]) -> list[tuple[int, int, int]] | None:
    """Ear-clip one ring (in patch order) in its Newell plane; None (nothing added) when it cannot be done
    completely without breaking the half-edge rule."""
    import numpy as np

    if len(set(ring)) != len(ring):
        return None                        # a loop that touches itself: leave it
    p = points[ring]
    normal = np.zeros(3)
    for i in range(len(ring)):             # Newell's method: the ring's area-weighted normal
        a, b = p[i], p[(i + 1) % len(ring)]
        normal += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]),
                            (a[0] - b[0]) * (a[1] + b[1])])
    if np.linalg.norm(normal) < 1e-18:
        return None
    normal /= np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    flat = {vid: np.array([pt @ u, pt @ v]) for vid, pt in zip(ring, p)}   # counter-clockwise about `normal`

    taken: set[tuple[int, int]] = set()
    out: list[tuple[int, int, int]] = []
    work = list(ring)
    while len(work) > 3:
        best, best_i = None, -1
        for i in range(len(work)):
            a, b, c = work[i - 1], work[i], work[(i + 1) % len(work)]
            pa, pb, pc = flat[a], flat[b], flat[c]
            cross = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
            if cross <= 1e-18:
                continue                   # a reflex (or flat) corner is no ear
            if (c, a) in used or (c, a) in taken:
                continue                   # the diagonal is already run this way: the half-edge rule
            if any(_in_triangle(flat[x], pa, pb, pc) for x in work if x not in (a, b, c)):
                continue
            # the fattest ear first: its smallest angle's sine
            e = (np.linalg.norm(pb - pa), np.linalg.norm(pc - pb), np.linalg.norm(pa - pc))
            quality = cross / max(max(e) * (sum(e) - max(e)), 1e-18)
            if best is None or quality > best:
                best, best_i = quality, i
        if best_i < 0:
            return None
        a, b, c = work[best_i - 1], work[best_i], work[(best_i + 1) % len(work)]
        out.append((a, b, c))
        taken.update(((a, b), (b, c), (c, a)))
        work.pop(best_i)
    a, b, c = work
    if (a, b) in used or (b, c) in used or (c, a) in used or {(a, b), (b, c), (c, a)} & taken:
        return None
    out.append((a, b, c))
    taken.update(((a, b), (b, c), (c, a)))
    if taken & used:
        return None
    used.update(taken)
    return out


def _in_triangle(p: Any, a: Any, b: Any, c: Any) -> bool:
    """p inside (or on) the counter-clockwise 2D triangle a, b, c."""
    def side(o: Any, s: Any, t: Any) -> float:
        return (s[0] - o[0]) * (t[1] - o[1]) - (s[1] - o[1]) * (t[0] - o[0])
    return side(a, b, p) >= 0 and side(b, c, p) >= 0 and side(c, a, p) >= 0


def drop_islands(faces: Any, min_faces: int) -> tuple[Any, int]:
    """Remove the connected pieces (sharing edges) of fewer than min_faces faces: the stray slivers the recovery
    leaves beside the surface. -> (faces, how many faces went)."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(faces) == 0:
        return faces, 0
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(faces)), 3)
    _, key = np.unique(edges, axis=0, return_inverse=True)
    key = key.reshape(-1)
    order = np.argsort(key, kind="stable")
    same = key[order][1:] == key[order][:-1]
    a, b = owner[order][1:][same], owner[order][:-1][same]
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(faces), len(faces)))
    _, label = connected_components(graph, directed=False)
    sizes = np.bincount(label)
    keep = sizes[label] >= min_faces
    return faces[keep], int((~keep).sum())


def edge_report(faces: Any) -> dict[str, float]:
    """How clean a triangle mesh is: its undirected edges, the share on more than two faces (non-manifold), on one
    face (boundary), and whether every edge shared by two faces runs in opposite directions (consistent winding)."""
    import numpy as np

    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(faces):
        return {"edges": 0, "non_manifold": 0.0, "boundary": 0.0, "consistent": True}
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    _, directed_counts = np.unique(directed, axis=0, return_counts=True)
    total = max(len(undirected), 1)
    return {"edges": int(len(undirected)),
            "non_manifold": float((counts > 2).sum() / total),
            "boundary": float((counts == 1).sum() / total),
            "consistent": bool((directed_counts == 1).all())}


def _normalize(v: Any) -> Any:
    import numpy as np

    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _tangent_basis(normals: Any) -> tuple[Any, Any]:
    """Right-handed (t1, t2, n) per vertex, so ascending atan2 is counter-clockwise seen from outside."""
    import numpy as np

    helper = np.tile(np.array([0.0, 0.0, 1.0]), (len(normals), 1))
    helper[np.abs(normals[:, 2]) > 0.9] = np.array([1.0, 0.0, 0.0])
    t1 = _normalize(np.cross(helper, normals))
    return t1, np.cross(normals, t1)


def _candidate_edges(logits: Any, p: TopologyParams) -> Any:
    """The top-`max_degree` partners of every vertex above `min_logit`, unioned symmetrically."""
    import numpy as np

    if isinstance(logits, EdgeLogits):
        n = logits.n
        rows, cols = logits.candidate_pairs(min(p.max_degree, n - 1), p.min_logit)
    else:
        n = logits.shape[0]
        k = min(p.max_degree, n - 1)
        part = np.argpartition(-logits, k, axis=1)[:, :k]
        rows = np.repeat(np.arange(n), k)
        cols = part.ravel()
        keep = logits[rows, cols] > p.min_logit
        rows, cols = rows[keep], cols[keep]
    pairs = np.stack([np.minimum(rows, cols), np.maximum(rows, cols)], axis=1)
    if not len(pairs):
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.unique(pairs, axis=0)
    return pairs[pairs[:, 0] != pairs[:, 1]]


def _geometric_filter(points: Any, normals: Any, edges: Any, p: TopologyParams) -> Any:
    """Drop edges no surface edge could be: through a thin wall, off both tangent planes, or over-long."""
    import numpy as np

    if not len(edges):
        return edges
    u, v = edges[:, 0], edges[:, 1]
    d = points[v] - points[u]
    length = np.linalg.norm(d, axis=1)
    ok = length > 1e-9
    dirn = d / np.maximum(length, 1e-12)[:, None]
    ok &= (normals[u] * normals[v]).sum(1) > p.min_normal_dot
    align_u = np.abs((dirn * normals[u]).sum(1))
    align_v = np.abs((dirn * normals[v]).sum(1))
    ok &= np.minimum(align_u, align_v) < p.max_normal_align
    med = np.median(length[ok]) if ok.any() else 1.0
    ok &= length < p.max_len_factor * med
    return edges[ok]


def _adjacency(edges: Any, n: int) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    return adj


def _umbrella_proposals(points: Any, adj: list[list[int]], t1: Any, t2: Any) -> dict[tuple[int, int, int], int]:
    """One triangle per adjacent pair in each vertex's cyclic neighbour order -> {canonical oriented triple: votes}."""
    import numpy as np

    votes: dict[tuple[int, int, int], int] = {}
    for v, nbrs in enumerate(adj):
        if len(nbrs) < 2:
            continue
        nb = np.asarray(nbrs)
        d = points[nb] - points[v]
        nb = nb[np.argsort(np.arctan2(d @ t2[v], d @ t1[v]))]
        for a, b in zip(nb, np.roll(nb, -1)):
            if a == b:
                continue
            key = _canonical(v, int(a), int(b))
            votes[key] = votes.get(key, 0) + 1
    return votes


def _canonical(a: int, b: int, c: int) -> tuple[int, int, int]:
    """Rotate an oriented triple so the smallest index leads (the winding is kept)."""
    if a <= b and a <= c:
        return (a, b, c)
    if b <= a and b <= c:
        return (b, c, a)
    return (c, a, b)


def _triangle_quality(points: Any, normals: Any, tris: Any, p: TopologyParams) -> Any:
    """Mask of the triangles that are neither slivers nor fins standing off the surface."""
    import numpy as np

    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    cross = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cross, axis=1)
    ok = area2 > 1e-16
    e0 = np.linalg.norm(b - a, axis=1)
    e1 = np.linalg.norm(c - b, axis=1)
    e2 = np.linalg.norm(a - c, axis=1)
    longest = np.maximum(np.maximum(e0, e1), e2)
    # a lower bound on sin(smallest angle): 2*area over the longest side times the sum of the other two
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_min = area2 / np.maximum(longest * np.maximum(e0 + e1 + e2 - longest, 1e-12), 1e-12)
    ok &= sin_min > np.sin(np.radians(p.min_tri_angle_deg))
    face_n = cross / np.maximum(area2, 1e-12)[:, None]
    vert_n = _normalize(normals[tris[:, 0]] + normals[tris[:, 1]] + normals[tris[:, 2]])
    ok &= (face_n * vert_n).sum(1) > p.max_tri_tilt
    return ok


def _assemble(tris: Any, scores: Any) -> tuple[Any, set[tuple[int, int]]]:
    """Accept triangles best first while no directed edge is used twice (the half-edge rule)."""
    import numpy as np

    used: set[tuple[int, int]] = set()
    accepted = []
    for i in np.argsort(-scores, kind="stable"):
        a, b, c = int(tris[i, 0]), int(tris[i, 1]), int(tris[i, 2])
        h = ((a, b), (b, c), (c, a))
        if h[0] in used or h[1] in used or h[2] in used:
            continue
        used.update(h)
        accepted.append((a, b, c))
    return np.asarray(accepted, dtype=np.int64).reshape(-1, 3), used


def _open_halfedges(used: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Directed edges whose opposite is unused: the boundary."""
    return [(a, b) for (a, b) in used if (b, a) not in used]


def _tri_score(points: Any, normals: Any, logits: Any, b: int, a: int, c: int, p: TopologyParams) -> float:
    """How good the triangle (b, a, c) is on its own (facing, shape, the model's belief in its edges); -inf when it
    is not admissible."""
    import numpy as np

    pa, pb, pc = points[a], points[b], points[c]
    fn = np.cross(pa - pb, pc - pb)
    area2 = float(np.linalg.norm(fn))
    if area2 < 1e-16:
        return -np.inf
    fn = fn / area2
    vn = normals[a] + normals[b] + normals[c]
    vn = vn / max(float(np.linalg.norm(vn)), 1e-12)
    tilt = float(fn @ vn)
    if tilt < p.max_tri_tilt:
        return -np.inf
    e = (float(np.linalg.norm(pa - pb)), float(np.linalg.norm(pc - pa)), float(np.linalg.norm(pb - pc)))
    longest = max(e)
    sin_min = area2 / max(longest * max(sum(e) - longest, 1e-12), 1e-12)
    if sin_min < np.sin(np.radians(p.min_tri_angle_deg)):
        return -np.inf
    conf = float(min(logits[b, a], logits[a, c], logits[c, b]))
    return tilt + 0.4 * sin_min + 0.02 * float(np.clip(conf, -30.0, 10.0))


def _advance_front(points: Any, normals: Any, logits: Any, adj: list[list[int]], faces: Any,
                   used: set[tuple[int, int]], p: TopologyParams) -> tuple[Any, int]:
    """Close the boundary: for each open directed edge (a, b), the best third vertex c whose face (b, a, c) is
    admissible and keeps the half-edge rule."""
    import numpy as np

    nbr = [set(a) for a in adj]
    extra = []
    for _ in range(p.front_rounds):
        added = 0
        for a, b in _open_halfedges(used):
            if (b, a) in used:
                continue
            best, best_c = -np.inf, -1
            for c in (nbr[a] | nbr[b]) - {a, b}:
                if (b, a) in used or (a, c) in used or (c, b) in used:
                    continue
                s = _tri_score(points, normals, logits, b, a, c, p)
                if s > best:
                    best, best_c = s, c
            if best_c >= 0:
                used.update(((b, a), (a, best_c), (best_c, b)))
                extra.append((b, a, best_c))
                added += 1
        if not added:
            break
    if extra:
        faces = np.concatenate([faces, np.asarray(extra, dtype=np.int64)], axis=0)
    return faces, len(extra)


def _boundary_loops(used: set[tuple[int, int]], max_len: int) -> list[list[int]]:
    """Chain the open directed edges into closed boundary loops (up to max_len edges)."""
    remaining = set(_open_halfedges(used))
    out: dict[int, list[int]] = {}
    for a, b in remaining:
        out.setdefault(a, []).append(b)
    loops = []
    while remaining:
        a0, b0 = next(iter(remaining))
        remaining.discard((a0, b0))
        loop, cur = [a0], b0
        ok = True
        while cur != a0:
            step = [x for x in out.get(cur, ()) if (cur, x) in remaining]
            if not step or len(loop) > max_len:
                ok = False
                break
            remaining.discard((cur, step[0]))
            loop.append(cur)
            cur = step[0]
        if ok and 3 <= len(loop) <= max_len:
            loops.append(loop)
    return loops


def _fill_loops(points: Any, normals: Any, logits: Any, loops: list[list[int]], used: set[tuple[int, int]],
                p: TopologyParams) -> list[tuple[int, int, int]]:
    """Ear-clip each boundary loop, reversed so the patch faces the same way as its surroundings."""
    import numpy as np

    new = []
    for loop in loops:
        ring = list(reversed(loop))
        guard = 0
        while len(ring) > 3 and guard < 4 * len(loop):
            guard += 1
            best, bi = -np.inf, -1
            for i in range(len(ring)):
                a, b, c = ring[i - 1], ring[i], ring[(i + 1) % len(ring)]
                if (a, b) in used or (b, c) in used or (c, a) in used:
                    continue
                s = _tri_score(points, normals, logits, a, b, c, p)
                if s > best:
                    best, bi = s, i
            if bi < 0:
                break
            a, b, c = ring[bi - 1], ring[bi], ring[(bi + 1) % len(ring)]
            used.update(((a, b), (b, c), (c, a)))
            new.append((a, b, c))
            ring.pop(bi)
        if len(ring) == 3:
            a, b, c = ring
            if (a, b) not in used and (b, c) not in used and (c, a) not in used:
                used.update(((a, b), (b, c), (c, a)))
                new.append((a, b, c))
    return new
