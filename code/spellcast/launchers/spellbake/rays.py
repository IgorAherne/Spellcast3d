"""Ray casting against the high poly with Embree (embreex 4.4.0), on a thread pool.

Embree works in float32, so the vertices are recentred on the bounding-box centre and
scaled into a unit box before the scene is built; every function here takes and returns
world units. embreex's ``run()`` releases the GIL, so Python threads scale almost
linearly (measured 12.3 Mray/s on 4 threads, 23.4 on 8, for a 292k-triangle prop), but
only once the scene is committed: the constructor commits it with one ray before any
thread uses it. The version is pinned because that behaviour was verified on 4.4.0.

Design section 4.4, step L10 (the cage ray and the automatic ray distance) lives here too.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from embreex import rtcore_scene
from embreex.mesh_construction import TriangleMesh

# Rays per embree call on one thread: big enough to amortise the call, small enough that
# a 0.5M-ray chunk still spreads over 8 threads.
_MIN_PIECE = 4096
_MAX_PIECE = 1 << 18
# The cage ray starts past other parts of the low poly (other_parts): crossings more than this
# share of d above P count (nearer ones are P's own neighbourhood), and at most this many are
# stepped through.
PART_MARGIN = 0.1
PART_CROSSINGS = 4
# A cage hit more than PART_MARGIN * d above P with another part of the low poly within this share of
# its height (claimed_by_other_parts) is that part's surface: the ray passes on, at most this many times.
CLAIM_RATIO = 0.25
CLAIM_SKIPS = 2

_Part = TypeVar("_Part")


@dataclass(frozen=True)
class Hits:
    """One entry per ray. ``u``/``v`` weight the hit triangle's vertices 1 and 2 (vertex 0
    gets 1 - u - v); ``t`` is the signed world distance from the origin along the direction
    the caller gave; ``back`` marks a hit on the back of a double-sided triangle."""

    prim: np.ndarray     # (n,) int64 triangle index, -1 = miss
    u: np.ndarray        # (n,) float64
    v: np.ndarray        # (n,) float64
    t: np.ndarray        # (n,) float64
    back: np.ndarray     # (n,) bool

    @property
    def hit(self) -> np.ndarray:
        return self.prim >= 0


@dataclass(frozen=True)
class CageHits:
    """The result of the cage ray (L10) for each sample point P."""

    prim: np.ndarray       # (n,) int64 high triangle, -1 = miss
    u: np.ndarray          # (n,) float64
    v: np.ndarray          # (n,) float64
    height: np.ndarray     # (n,) float64 hit offset from P along the cage normal (+ = outside), NaN on a miss
    back: np.ndarray       # (n,) bool: back of a double-sided triangle; flip its normals
    fallback: np.ndarray   # (n,) bool: found by the nearest-hit fallback, not the cage ray

    @property
    def hit(self) -> np.ndarray:
        return self.prim >= 0


class Scene:
    """The high poly's triangles in an Embree scene.

    ``double_sided`` (one flag per triangle) lets ``first_front_hit`` accept the back of
    triangles whose material is double-sided. Use it as a context manager, or call
    ``close()``, to stop the thread pool.
    """

    def __init__(
        self,
        positions: np.ndarray,
        faces: np.ndarray,
        *,
        double_sided: np.ndarray | None = None,
        threads: int = 1,
    ) -> None:
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if faces.size == 0:
            raise ValueError("the high poly has no triangles")
        lo, hi = positions.min(axis=0), positions.max(axis=0)
        self.centre = (lo + hi) / 2.0
        self.diag = float(np.linalg.norm(hi - lo))
        # World units per unit-box unit; a single point still gets a usable scale.
        self.scale = max(self.diag / 2.0, 1e-12)

        corners = positions[faces]
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        self.face_normals = np.divide(normals, length, out=np.zeros_like(normals), where=length > 0)

        if double_sided is not None:
            double_sided = np.asarray(double_sided, dtype=bool).reshape(-1)
            if len(double_sided) != len(faces):
                raise ValueError("double_sided needs one flag per triangle")
            if not double_sided.any():
                double_sided = None
        self.double_sided = double_sided

        self._scene = rtcore_scene.EmbreeScene()
        TriangleMesh(
            scene=self._scene,
            vertices=np.ascontiguousarray((positions - self.centre) / self.scale, dtype=np.float32),
            indices=np.ascontiguousarray(faces, dtype=np.int32),
        )
        # Commit the BVH on this thread: concurrent first calls would race to build it.
        self._scene.run(np.zeros((1, 3), np.float32), np.array([[0.0, 0.0, 1.0]], np.float32), output=1)
        self.threads = max(1, int(threads))
        self._pool = ThreadPoolExecutor(self.threads) if self.threads > 1 else None

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> Scene:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ embree calls

    def _pieces(self, count: int) -> list[slice]:
        if self._pool is None or count <= _MIN_PIECE:
            return [slice(0, count)]
        size = min(_MAX_PIECE, max(_MIN_PIECE, math.ceil(count / (2 * self.threads))))
        return [slice(start, min(start + size, count)) for start in range(0, count, size)]

    def _map(self, work: Callable[[slice], _Part], count: int) -> list[_Part]:
        pieces = self._pieces(count)
        if len(pieces) == 1:
            return [work(pieces[0])]
        return list(self._pool.map(work, pieces))

    def _unit_rays(
        self, origins: np.ndarray, directions: np.ndarray, distance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World rays -> the float32 unit-box rays embree takes (done per piece, in the pool)."""
        o = np.ascontiguousarray((origins - self.centre) / self.scale, dtype=np.float32)
        d = np.ascontiguousarray(directions, dtype=np.float32)
        dist = np.ascontiguousarray(distance / self.scale, dtype=np.float32)
        return o, d, dist

    def _intersect(
        self, origins: np.ndarray, directions: np.ndarray, distance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Closest hit within ``distance`` -> (prim, u, v, t), world units."""

        def work(piece: slice) -> tuple[np.ndarray, ...]:
            o, d, dist = self._unit_rays(origins[piece], directions[piece], distance[piece])
            hit = self._scene.run(o, d, dists=dist, output=1)
            return (hit["primID"].astype(np.int64), hit["u"].astype(np.float64),
                    hit["v"].astype(np.float64), hit["tfar"].astype(np.float64) * self.scale)

        parts = self._map(work, len(origins))
        prim, u, v, t = (np.concatenate([part[k] for part in parts]) for k in range(4))
        return prim, u, v, t

    # ------------------------------------------------------------------ queries

    def first_front_hit(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        max_distance: float | np.ndarray,
        *,
        max_skips: int = 3,
    ) -> Hits:
        """First hit whose triangle faces the ray (its normal points against the direction).

        A back-facing hit on a single-sided triangle is skipped by casting again from just
        past it, at most ``max_skips`` times. A double-sided triangle is accepted from
        either side (``back`` says which). Directions must be unit vectors.
        """
        origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        count = len(origins)
        limit = np.broadcast_to(np.asarray(max_distance, dtype=np.float64), (count,))
        eps = 1e-5 * self.diag

        prim = np.full(count, -1, dtype=np.int64)
        u = np.zeros(count)
        v = np.zeros(count)
        t = np.full(count, np.inf)
        back = np.zeros(count, dtype=bool)
        start = np.zeros(count)
        active = np.arange(count)
        for _ in range(max_skips + 1):
            active = active[limit[active] - start[active] > 0]
            if active.size == 0:
                break
            ray_o = origins[active] + directions[active] * start[active, None]
            p, hu, hv, ht = self._intersect(ray_o, directions[active], limit[active] - start[active])
            found = p >= 0
            ids, p, hu, hv = active[found], p[found], hu[found], hv[found]
            t_hit = start[ids] + ht[found]
            facing = np.einsum("ij,ij->i", self.face_normals[p], directions[ids])
            front = facing < 0
            ok = (front | self.double_sided[p]) if self.double_sided is not None else front
            take = ids[ok]
            prim[take], u[take], v[take], t[take] = p[ok], hu[ok], hv[ok], t_hit[ok]
            back[take] = ~front[ok]
            active = ids[~ok]
            start[active] = t_hit[~ok] + eps
        return Hits(prim, u, v, t, back)

    def nearest_bidirectional(
        self, origins: np.ndarray, directions: np.ndarray, max_distance: float | np.ndarray
    ) -> Hits:
        """The nearer of the closest hits along +direction and -direction, whichever way
        the triangle faces. ``t`` is negative for a hit behind the origin."""
        origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        limit = np.broadcast_to(np.asarray(max_distance, dtype=np.float64), (len(origins),))
        both_o = np.concatenate([origins, origins])
        both_d = np.concatenate([directions, -directions])
        prim, u, v, t = self._intersect(both_o, both_d, np.concatenate([limit, limit]))
        count = len(origins)
        t = np.where(prim >= 0, t, np.inf)
        use_back = t[count:] < t[:count]
        pick = np.where(use_back, np.arange(count) + count, np.arange(count))
        signed = np.where(use_back, -t[pick], t[pick])
        found = prim[pick] >= 0
        return Hits(
            np.where(found, prim[pick], -1),
            np.where(found, u[pick], 0.0),
            np.where(found, v[pick], 0.0),
            np.where(found, signed, np.inf),
            np.zeros(count, dtype=bool),
        )

    def occluded(
        self, origins: np.ndarray, directions: np.ndarray, max_distance: float | np.ndarray
    ) -> np.ndarray:
        """True where any triangle lies within ``max_distance`` along the ray."""
        origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        directions = np.asarray(directions).reshape(-1, 3)
        limit = np.broadcast_to(np.asarray(max_distance, dtype=np.float64), (len(origins),))

        def work(piece: slice) -> np.ndarray:
            o, d, dist = self._unit_rays(origins[piece], directions[piece], limit[piece])
            return self._scene.run(o, d, dists=dist, query="OCCLUDED") != -1

        return np.concatenate(self._map(work, len(origins)))


# ---------------------------------------------------------------------- L10


def other_parts(low: Scene, points: np.ndarray, cage: np.ndarray, distance: float, *,
                margin: float = PART_MARGIN, max_crossings: int = PART_CROSSINGS) -> np.ndarray:
    """How far along the cage ray (from P + d * C, towards P) to start it, so that it starts
    past every other part of the low poly: (n,) in [0, (1 - margin) d], 0 where the way is clear.

    ``low`` holds the low poly's own triangles, the retopo's say of what is one part. Where
    the cage ray crosses them more than ``margin * d`` above P, it passes through another
    part (a finger lying on the thigh, the far side of a crease, a cap's brim over the hair),
    and the high poly it would find there is that part's. The ray then starts just past the
    last such crossing, whichever way it faces: no longer inside or behind the other part,
    whatever that part's high poly does around its low poly (where the high pokes out below,
    the ray meets its underside from inside, a back face, which first_front_hit skips).
    High-poly relief on the same low-poly surface (the metal hoop over the planks) has no
    low triangles of its own above P and keeps the outermost rule. Crossings within
    ``margin * d`` of P are P's own neighbourhood (a curved or creased surface near P)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cage = np.asarray(cage, dtype=np.float64).reshape(-1, 3)
    d = float(distance)
    limit = (1.0 - margin) * d
    eps = 1e-5 * low.diag
    origins = points + d * cage
    start = np.zeros(len(points))
    active = np.arange(len(points))
    for _ in range(max_crossings):
        active = active[limit - start[active] > eps]
        if active.size == 0:
            break
        prim, _, _, t = low._intersect(origins[active] - cage[active] * start[active, None], -cage[active],
                                       limit - start[active])
        found = prim >= 0
        active = active[found]
        start[active] += t[found] + eps
    return np.minimum(start, limit)


def claimed_by_other_parts(low: Scene, scene: Scene, points: np.ndarray, cage: np.ndarray, prim: np.ndarray,
                           height: np.ndarray, *, ratio: float = CLAIM_RATIO) -> np.ndarray:
    """True where the high-poly hit on triangle ``prim``, ``height`` above P along the cage ray, lies
    nearer another part of the low poly than to P: a low triangle within ``ratio * height`` of it,
    looking along the high triangle's normal both ways.

    Such a hit is that part's own surface where its high poly bulges past its low poly (the side of a
    finger beside the thigh, of a drawstring beside the chest): the ray reaches it from P only by
    slipping past the low part, and the part's own texels already bake it, from close by. P itself,
    ``height`` away, is never that near, and high relief with no low part of its own (the hoop over
    the planks) has only P's surface under it, ``height`` away."""
    at = points + height[:, None] * cage
    return low.nearest_bidirectional(at, scene.face_normals[prim], ratio * height).hit


def cage_hits(scene: Scene, points: np.ndarray, cage: np.ndarray, distance: float, *,
              low: Scene | None = None, claims: Scene | None = None) -> CageHits:
    """Where each low-poly sample P lands on the high poly (design section 4.4, L10).

    The ray starts outside, at P + d * C, and runs along -C over [0, 2d]; the first
    triangle facing it wins. C is the welded smooth normal, a continuous field, so the
    map P -> hit is continuous across every triangle and UV seam, and among several
    crossings the outermost front-facing one is the visible one (nearest-in-either-
    direction would bake the planks under a metal hoop). With ``low`` (the low poly's own
    triangles) a ray that would start inside or behind another part of the low poly starts
    past it instead (``other_parts``): the skin of a finger lying on the thigh stays off the
    thigh. With ``claims`` (the low poly's own triangles too) a hit more than PART_MARGIN * d
    above P that another part of the low poly claims (``claimed_by_other_parts``) is passed
    over: the ray goes on from just past it to P - d, and keeps it only if it finds nothing
    else. When nothing is accepted, the nearer of the hits along +C and -C from P within
    4d is used, facing ignored.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cage = np.asarray(cage, dtype=np.float64).reshape(-1, 3)
    d = float(distance)
    # Distance skipped from P + d * C before the ray starts (0 = the full cage ray).
    skipped = other_parts(low, points, cage, d) if low is not None else np.zeros(len(points))
    first = scene.first_front_hit(points + (d - skipped)[:, None] * cage, -cage, 2.0 * d - skipped)
    prim, u, v = first.prim.copy(), first.u.copy(), first.v.copy()
    height = np.where(first.hit, d - skipped - first.t, np.nan)
    back = first.back.copy()
    fallback = np.zeros(len(points), dtype=bool)

    if claims is not None:
        eps = 1e-5 * scene.diag
        todo = np.flatnonzero(first.hit)
        for _ in range(CLAIM_SKIPS):
            todo = todo[height[todo] > PART_MARGIN * d]
            if todo.size:
                todo = todo[claimed_by_other_parts(claims, scene, points[todo], cage[todo], prim[todo], height[todo])]
            if todo.size == 0:
                break
            restart = height[todo] - eps      # just past the claimed hit, on along -C to P - d
            again = scene.first_front_hit(points[todo] + restart[:, None] * cage[todo], -cage[todo], d + restart)
            ok = again.hit
            todo = todo[ok]
            prim[todo], u[todo], v[todo], back[todo] = again.prim[ok], again.u[ok], again.v[ok], again.back[ok]
            height[todo] = restart[ok] - again.t[ok]

    missed = np.flatnonzero(~first.hit)
    if missed.size:
        near = scene.nearest_bidirectional(points[missed], cage[missed], 4.0 * d)
        rescued = missed[near.hit]
        prim[rescued] = near.prim[near.hit]
        u[rescued], v[rescued] = near.u[near.hit], near.v[near.hit]
        height[rescued] = near.t[near.hit]
        fallback[rescued] = True
    return CageHits(prim, u, v, height, back, fallback)


def auto_distance(scene: Scene, points: np.ndarray, directions: np.ndarray) -> float:
    """The cage distance d for these probe points (L10, "Auto d").

    Nearest hits along +-direction within 10% of the diagonal give the deviation delta
    between the low and the high; d = 1.25 * p99.5(delta), clamped to 0.2%..5% of the
    diagonal. Probes that find nothing are ignored; with no hit at all d is the maximum.
    """
    hits = scene.nearest_bidirectional(points, directions, 0.10 * scene.diag)
    lo, hi = 0.002 * scene.diag, 0.05 * scene.diag
    if not hits.hit.any():
        return hi
    delta = np.abs(hits.t[hits.hit])
    return float(np.clip(1.25 * np.percentile(delta, 99.5), lo, hi))
