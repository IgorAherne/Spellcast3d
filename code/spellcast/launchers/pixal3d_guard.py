"""Cheap checks on what Pixal3D makes, for pixal3d_server.py.

Pure Python (streak_share alone needs numpy), so the hub's tests import it without torch. The numbers
below were measured on real props, among them a sheep (fine) and a bear statue whose front came out as
a bundle of thin fins running straight toward the camera.
"""

from __future__ import annotations

from typing import Any, Sequence

# ---- sparse structure: the first stage ("Blocking out the shape") ---------------------------------
# If it filled the silhouette as a solid block from the body to the grid face on the camera side (or the
# opposite face), every later stage could only refine inside it and the front would become fins. The bear
# statue first blamed on this had a clean blocked-out shape: its fins came from misplaced hole-fill vertices
# (pixal3d_server.exact_hole_centres). The guard stays as a cheap safety net.
# Measured on the GPU (seed 0-7 bear, seed 0-3 sheep, the 19 props of Pixal3D's assets/images at seeds 0
# and 1: 101 healthy samples at the hub's settings): edge score max 0.109, 95th percentile 0.055, median 0;
# deep share max 0.017. The failed bear (its final mesh at 32^3, a lower bound on its sparse structure):
# edge score 0.938, deep share 0.922. A draw is redrawn only when BOTH are over their limit, so a prop that
# is really deep along the camera axis (a car seen head-on) but hollow does not trip it.
GUARD_THRESHOLD = 0.3     # edge score: share of the columns that reach the camera-side (or far) grid face
DEEP_THRESHOLD = 0.3      # deep share: share of the columns solid for DEEP_RUN+ voxels from that face
DEEP_RUN = 8
GUARD_ATTEMPTS = 3        # samples in total, the first one included; the lowest edge score is kept
# Sampler settings per attempt on top of the hub's own (the attempts not listed keep them): attempt 1 is
# exactly today's draw, attempt 2 new noise (the CPU random generator has moved on), attempt 3 new noise
# with more steps and weaker guidance (about 38 s instead of 20 s for this stage).
RETRY_SAMPLER = {3: {"steps": 25, "guidance_strength": 5.0}}

# ---- remeshing: the inward-facing copy of the surface ------------------------------------------------
# to_glb's dual contouring (remesh=True) extracts the surface at a small distance eps around the decoded
# mesh: around every closed part that is one shell outside and one eps inside, facing inward. On the sheep the
# copy is a separate piece with negative signed volume (1.43M of 2.97M remeshed faces). On a furry bear it is
# joined to the visible shell wherever the prop is thinner than a few voxels: dropping pieces by volume took
# 16k of 4.4M faces and 67% of its area kept a second wall 0.004 behind it. There the copy is found face by face:
# faces whose centre is inside the decoded mesh (cuBVH's ray test), in large connected pieces - the bear's
# copy is one piece of 1.89M faces; the rest (44k faces in 13.8k pieces of at most 789) are mostly visible
# faces in tight crevices where every test ray hits fur, and stay. The averaged-normal test is no use there:
# it agreed with the ray test on 49% of the bear's faces.
# Two things the ray test cannot tell from the copy, so the joined copy is taken with two more checks
# (pixal3d_server.find_inner_copy):
# (a) A fold in the fur is "inside" too, yet visible: no face seen from outside goes (rasterized views all
#     around - cuBVH's own ray_trace returns a far face instead of the nearest for many rays on meshes of
#     millions of faces), nor SEEN_RINGS rings of faces around one. Measured on the bear's remesh (4.4M faces,
#     300 other views, simplified like to_glb after): the rule removes 42.9% of the faces, 2.3% of the kept
#     area still has a wall behind it, and ONE small pit on the shoulder, seen only through a narrow opening,
#     still loses its bottom (8 back-face patches in 300 views, as with the rule before these checks: 7-11;
#     no rule tried closes it without keeping most of the copy - 400 or 1000 views, 12 rings, filling the
#     openings). The sheep and the multi-view head: 0.
# (b) The inside wall of a hollow with a small opening (a jar through its neck, a room through a window) is
#     "inside" as well: the test only asks whether a few rays escape. A piece is kept when most of its sampled
#     points see out: jars with a 15, 10 and 8 degree opening 0.99, 0.90, 0.81 of their points (the rule before
#     removed 62-90% of their inside, seen through the opening); the bear's copy 0.46 (its fur lets rays out),
#     the multi-view head's 0.08. A misjudged piece stays: never a hole.
# How much can be the copy (inner_plan): a solid decoded surface gets one hidden wall per visible one (half the
# faces: sheep 48%, bear 43%, multi-view head 52%). But a decoded surface that is itself a thin closed shell
# around a hollow gets three: eps outside and inside each of its two sides, so up to three quarters. The head
# made from a 2x2 sheet came out like that (decoded volume 0.012, a solid head's is about 0.3): 71% of its 7.4M
# remeshed faces are hidden, by area 2.3 times what stays; in one run 43% were separate pieces and the rest
# joined (49% of what they leave), in another 3% and 70.5%. Removing all of it: 36 of 5.4M pixels of the
# remeshed surface in 300 views showed a removed face, 0 back-face patches after simplify, and the share of the
# kept area inside the decoded mesh (the copy left) fell from 70% to 0.85%. So each step is judged against the
# faces it works on, with a limit a little above three quarters.
MAX_INNER_SHARE = 0.8          # a step never removes more than this share of the faces it works on (inner_plan)
INNER_PIECE_MIN_FACES = 2000   # a piece of inside faces goes when it has at least this many faces...
INNER_PIECE_MIN_SHARE = 0.02   # ...and at least this share of the largest such piece
SEEN_DIRECTIONS = 96           # (a) directions the surface is looked at from...
SEEN_GRID = 512                # ...in pictures of SEEN_GRID x SEEN_GRID pixels
SEEN_RINGS = 2                 # faces this close to a seen one stay too
CAVITY_SAMPLES = 1024          # (b) points of a piece tested...
CAVITY_RAYS = 1024             # ...with this many rays each in random directions;
CAVITY_POINT_ESCAPE = 1 / 1024 # a point sees out when at least this share of its rays leave the decoded mesh (one);
CAVITY_SHARE = 0.7             # a piece is a hollow's wall when at least this share of its points see out

# ---- final mesh: streaks along the camera axis --------------------------------------------------------
# Area share of long thin faces lying along the camera axis: bear 0.44, sheep 0.0000.
STREAK_THRESHOLD = 0.15
STREAK_WARNING = ("This model may be stretched: the hidden side was hard to guess. Try another seed, "
                  "or add more views.")


def ss_edge_score(coords: Any, resolution: int = 32) -> float:
    """How much of a sampled sparse structure is a solid block running to the edge of the grid.

    coords: (N, 4) rows (batch, i, j, k) as sample_sparse_structure returns them (a tensor, an array
    or a list) at `resolution`. k is the depth axis: ProjGrid maps grid point (x, y, z) to Blender
    (x, -z, y) with the camera at Blender (0, -d, 0), so the camera sits at high k. Returns the share
    of the occupied (i, j) columns that reach the camera-side face (k >= resolution - 2) or the far
    face (k <= 1), whichever is larger; 0 for an empty structure. A normal prop is a thin shell well
    inside the grid and scores about 0.
    """
    columns = _columns(coords)
    near = sum(1 for ks in columns.values() if max(ks) >= resolution - 2)
    far = sum(1 for ks in columns.values() if min(ks) <= 1)
    return max(near, far) / len(columns) if columns else 0.0


def ss_deep_share(coords: Any, resolution: int = 32, run: int = DEEP_RUN) -> float:
    """How much of a sampled sparse structure is SOLID from a grid face inwards: the share of the occupied
    (i, j) columns whose voxels k = resolution-1, resolution-2, ... (camera side) are all occupied for at
    least `run` voxels, or the same from k = 0 (far side), whichever side has more. A thin surface that
    merely touches a face scores 0; the failed bear 0.92."""
    columns = _columns(coords)
    if not columns:
        return 0.0
    near = sum(1 for ks in columns.values() if all(resolution - 1 - n in ks for n in range(run)))
    far = sum(1 for ks in columns.values() if all(n in ks for n in range(run)))
    return max(near, far) / len(columns)


def ss_blocked(score: float, deep: float) -> bool:
    """True when a draw is the solid block to draw again (both measures over their limit)."""
    return score > GUARD_THRESHOLD and deep >= DEEP_THRESHOLD


def _columns(coords: Any) -> dict[tuple[int, int, int], set[int]]:
    """(batch, i, j) -> its occupied k, from (N, 4) rows (a tensor, an array or a list)."""
    rows = coords.tolist() if hasattr(coords, "tolist") else coords
    columns: dict[tuple[int, int, int], set[int]] = {}
    for b, i, j, k in rows:
        columns.setdefault((int(b), int(i), int(j)), set()).add(int(k))
    return columns


def drop_inner_copy(inner: int, total: int, max_share: float = MAX_INNER_SHARE) -> bool:
    """Whether to remove the `inner` faces found as the inward-facing copy from a remeshed surface of `total`
    faces: not when there are none, when every face would go (then the tests say nothing), or when they
    are more than `max_share` of the faces (the copy is about half of them, up to three quarters around a
    hollow)."""
    return 0 < inner < total and inner <= max_share * total


def inner_plan(whole: int, joined: int, total: int) -> str:
    """Which faces of the copy go: "both" (the separate inward-facing pieces and the copy joined to the
    visible shell), "whole" (the separate pieces alone: the joined faces are too many, and the separate
    pieces are sure) or "none" (the separate pieces alone are too many: nothing found is trusted).

    Each step is judged against the faces it works on, by drop_inner_copy: the separate pieces against all
    `total` faces, the joined copy against the faces the separate pieces leave (the joined copy lies inside
    what stays: one wall per visible one, three around a hollow). Their sum is no measure: the three hidden
    walls of a decoded shell around a hollow split between the two steps as the pieces happen to join."""
    if whole and not drop_inner_copy(whole, total):
        return "none"
    if joined and drop_inner_copy(joined, total - whole):
        return "both"
    return "whole" if whole else "none"


def is_cavity(seeing_out: float) -> bool:
    """A piece of faces "inside" the decoded mesh is the wall of a hollow with an opening (kept), not the
    hidden copy, when at least CAVITY_SHARE of its sampled points see out (`seeing_out`, 0-1)."""
    return seeing_out >= CAVITY_SHARE


def inner_pieces(sizes: Sequence[int]) -> list[bool]:
    """Which connected pieces of the faces found inside the decoded mesh (their face counts) are the
    inward-facing copy: the large ones (INNER_PIECE_MIN_FACES, INNER_PIECE_MIN_SHARE of the largest)."""
    if not sizes:
        return []
    limit = max(INNER_PIECE_MIN_FACES, INNER_PIECE_MIN_SHARE * max(sizes))
    return [n >= limit for n in sizes]


def streak_share(vertices: Any, faces: Any, axis: int = 2) -> float:
    """Share of the surface area in long thin faces lying along `axis` (the camera axis: GLB Z).

    A face counts when its normal is nearly perpendicular to the axis (|n| < 0.05 along it), it is
    long and thin (longest edge squared over twice its area > 8) and its longest edge runs within
    about 18 degrees of the axis. Needs numpy; about 0.1 s for 300k faces.
    """
    import numpy as np

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        return 0.0
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    edges = np.stack([b - a, c - b, a - c], axis=1)                    # (F, 3 edges, xyz)
    lengths = np.linalg.norm(edges, axis=2)
    cross = np.cross(b - a, c - a)
    double_area = np.linalg.norm(cross, axis=1)
    total = double_area.sum()
    ok = double_area > 1e-20                                           # degenerate faces have no shape
    if total <= 0 or not ok.any():
        return 0.0
    rows = np.nonzero(ok)[0]
    longest = lengths[rows].argmax(axis=1)
    long_edge = edges[rows, longest]
    long_len = lengths[rows, longest]
    normal_along = np.abs(cross[rows, axis]) / double_area[rows]
    edge_along = np.abs(long_edge[:, axis]) / long_len
    aspect = long_len ** 2 / double_area[rows]
    streak = (normal_along < 0.05) & (aspect > 8) & (edge_along > 0.95)
    return float(double_area[rows][streak].sum() / total)
