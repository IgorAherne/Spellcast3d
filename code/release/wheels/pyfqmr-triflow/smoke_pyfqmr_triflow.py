"""Smoke test for the pyfqmr-triflow wheel, exercising simplify_mesh the way TriFlow's flow-to-mesh
step drives it (custom per-vertex point quadrics + per-vertex target-group ids, lossless=False with a
threshold cap, preserve_border=False, aggressiveness 7, 200 iterations). Written from pyfqmr's own
documented API; no TriFlow code is copied.

    python smoke_pyfqmr_triflow.py [mesh.stl|obj]      (needs numpy, scipy, trimesh; no torch)

Run it with PYTHONPATH pointing at the unpacked wheel, or in a venv that has the wheel installed.
Prints one JSON line with the numbers; exit code 0 = all checks passed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import pyfqmr_triflow

FACE_TARGET = 4000          # TriFlow's --face_count default
QEM_CAP = 12.0              # TriFlow's --qem_threshold default
POS_WEIGHT = 0.1            # weight per voter in TriFlow's run


def point_quadrics(targets: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Quadric of w*|x - t|^2 as a symmetric 4x4 [[wI, -wt], [-wt^T, w t.t]], packed in pyfqmr's
    documented order [m11, m12, m13, m14, m22, m23, m24, m33, m34, m44]."""
    q = np.zeros((len(targets), 10))
    q[:, [0, 4, 7]] = weight[:, None]
    q[:, [3, 6, 8]] = -weight[:, None] * targets
    q[:, 9] = weight * np.einsum("ij,ij->i", targets, targets)
    return q


def simplify(v, f, **kw):
    s = pyfqmr_triflow.Simplify()
    s.setMesh(v, f)
    t = time.perf_counter()
    s.simplify_mesh(target_count=FACE_TARGET, aggressiveness=7, max_iterations=200, preserve_border=False,
                    verbose=False, lossless=False, threshold_lossless=QEM_CAP, **kw)
    dt = time.perf_counter() - t
    ov, of, on = s.getMesh()
    return ov, of, on, dt


def main() -> int:
    here = Path(__file__).resolve()
    mesh_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        here.parents[4] / "release" / "cache" / "wheels" / "src" / "pyfqmr-triflow" / "example" / "Stanford_Bunny_sample.stl"
    m = trimesh.load(mesh_path, force="mesh", process=True)
    # TriFlow works in a 512-voxel frame: longest side 488 voxels, centred at 256.
    v = (m.vertices - m.bounds.mean(0)) * (488.0 / m.extents.max()) + 256.0
    f = np.asarray(m.faces)
    nv = len(v)
    checks, out = {}, {"mesh": mesh_path.name, "in_vertices": nv, "in_faces": len(f),
                       "pyfqmr_triflow": pyfqmr_triflow.__file__}

    # Target groups: ~1 root per 12 vertices, every vertex joins its nearest root (by position).
    rng = np.random.default_rng(0)
    roots = np.sort(rng.choice(nv, size=nv // 12, replace=False))
    _, nearest = cKDTree(v[roots]).query(v)
    group = roots[nearest].astype(np.int64)               # per-vertex: mesh index of its root
    targets = v[group]                                     # each vertex is pulled to its root's position
    ids, counts = np.unique(group, return_counts=True)
    weight = np.zeros(nv)
    weight[ids] = counts * POS_WEIGHT                      # only roots carry a pull, scaled by voters
    q = point_quadrics(targets, weight)

    ov0, of0, _, dt0 = simplify(v, f)                      # plain QEM (reference)
    ov1, of1, on1, dt1 = simplify(v, f, custom_quadratics=q, target_vertex_ids=group, min_edge_length=-1.0)
    ov2, of2, _, _ = simplify(v, f, custom_quadratics=q, target_vertex_ids=group, min_edge_length=-1.0)

    root_tree = cKDTree(v[ids])
    in_tree = cKDTree(v)
    d_plain = root_tree.query(ov0)[0]
    d_custom = root_tree.query(ov1)[0]
    out.update({
        "plain_faces": len(of0), "plain_seconds": round(dt0, 3),
        "custom_faces": len(of1), "custom_vertices": len(ov1), "custom_seconds": round(dt1, 3),
        "mean_dist_to_root_plain": round(float(d_plain.mean()), 4),
        "mean_dist_to_root_custom": round(float(d_custom.mean()), 4),
        "median_dist_to_root_custom": round(float(np.median(d_custom)), 4),
        "max_dist_to_input_vertices_plain": round(float(in_tree.query(ov0)[0].max()), 4),
        "max_dist_to_input_vertices_custom": round(float(in_tree.query(ov1)[0].max()), 4),
        "mean_dist_to_input_vertices_custom": round(float(in_tree.query(ov1)[0].mean()), 4),
    })
    checks["faces_reduced"] = len(of1) < len(f) // 4
    checks["face_count_near_target"] = len(of1) <= FACE_TARGET * 3   # the QEM cap may stop above target
    checks["indices_valid"] = bool(of1.min() >= 0 and of1.max() < len(ov1))
    checks["finite"] = bool(np.isfinite(ov1).all() and np.isfinite(on1).all())
    checks["dtypes"] = (of1.dtype == np.int32 and ov1.dtype == np.float64)
    # Output vertices stay near the input (distances in voxels; the input vertex spacing is ~2-4 voxels,
    # and plain QEM is the yardstick for the worst case).
    checks["stays_near_input"] = (out["mean_dist_to_input_vertices_custom"] < 2.0 and
                                  out["max_dist_to_input_vertices_custom"] <= 2 * out["max_dist_to_input_vertices_plain"] + 1)
    checks["quadrics_pull_to_roots"] = out["mean_dist_to_root_custom"] < out["mean_dist_to_root_plain"]
    checks["deterministic"] = bool(np.array_equal(ov1, ov2) and np.array_equal(of1, of2))
    try:
        s = pyfqmr_triflow.Simplify()
        s.setMesh(v, f)
        s.simplify_mesh(target_count=FACE_TARGET, verbose=False, custom_quadratics=q[:, :9])
        checks["bad_shape_rejected"] = False
    except ValueError:
        checks["bad_shape_rejected"] = True
    out["checks"] = checks
    out["passed"] = all(checks.values())
    print(json.dumps(out))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
