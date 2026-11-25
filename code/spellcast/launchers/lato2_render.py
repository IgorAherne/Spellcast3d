"""The picture LATO.2 is conditioned on, drawn with plain torch instead of Open3D, for lato2_server.py.

LATO.2 looks at one white-model view of the input (y up, camera at azimuth 45 / elevation 30, black background,
cropped to the object with 1.2 padding, 518 x 518) and encodes it with DINOv2 (dataset/voxel_dataset.py
_render_image). Upstream draws it with Open3D's OffscreenRenderer, which cannot work on Windows: Open3D's Python
binding always asks for EGL headless rendering, compiled for Linux only ("EGL Headless is not supported on this
platform", Open3D 0.17-0.19 alike).

The user's earlier Windows port (Pixal3D-stableprojectorz branch llm-experiments-sept2026, lato2_render_patch.py)
drew it with nvdiffrast. Here a small z-buffer rasterizer in torch does the same job, for two reasons: nvdiffrast's
licence (NVIDIA Source Code License) allows non-commercial use only, which would make LATO.2 - MIT, like its
weights - non-commercial in practice; and torch runs on every GPU the venv's torch supports (and on the CPU),
without another compiled wheel. It needs no antialiasing pass of its own: the picture is drawn at twice the size
and averaged down, and it is then resized to 518 px by the crop anyway.

The framing is upstream's own: the camera placement (_orbit_eye) and the crop to the object
(_crop_resize_to_object) are LATO.2's functions (dataset/mesh_render.py), so only the shading differs: a Lambertian
sun from Open3D's sun direction plus an ambient term, approximating Filament's sun + image-based light on the grey
(0.78, 0.78, 0.82) material (the branch's constants).

Needs LATO.2's repo on sys.path; numpy / torch at call time only.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

# Linear-space shading, tuned on the branch to approximate Open3D's sun_intensity=90000 / ambient_intensity=32000.
AMBIENT = 0.35
GAMMA = 2.2
# LATO.2's conditioning view (dataset/voxel_dataset.py _render_image, scripts/e2e_inference.py defaults).
IMG_RES = 518
AZIMUTH = 45.0
ELEVATION = 30.0
MESH_COLOR = (0.78, 0.78, 0.82)
BG_COLOR = (0.0, 0.0, 0.0)
CROP_PADDING = 1.2
CAMERA_DISTANCE = 1.8        # WhiteModelRenderer's defaults
FOV = 50.0                   # vertical, degrees
SUPERSAMPLE = 2              # drawn at 2x and averaged down: smooth silhouettes
# Candidate (triangle, pixel) pairs tested per pass: about 60 bytes each, so 8M keep a pass near 0.5 GB.
CANDIDATES_PER_PASS = 1 << 23


def render_condition(vertices: Any, faces: Any, device: Any = "cuda", img_res: int = IMG_RES,
                     azimuth: float = AZIMUTH, elevation: float = ELEVATION) -> Any:
    """The (img_res, img_res, 3) uint8 view LATO.2 conditions on, of the mesh (vertices in its [-0.5, 0.5] frame,
    y up): what WhiteModelRenderer(img_res, mesh_color=MESH_COLOR, bg_color=BG_COLOR, up_axis="y",
    add_ground=False, crop_to_object=True, crop_padding=1.2).render(..., azimuths=[45], elevations=[30]) gives."""
    import numpy as np
    from dataset.mesh_render import WhiteModelRenderer, _orbit_eye   # LATO.2 (MIT)

    vertices = np.asarray(vertices, dtype=np.float64)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    eye, up = _orbit_eye(center, CAMERA_DISTANCE, azimuth, elevation, "y")
    rgb, mask = shade(vertices, np.asarray(faces), eye, center, up, img_res, device)
    crop = SimpleNamespace(img_res=img_res, crop_padding=CROP_PADDING)   # all _crop_resize_to_object reads
    return np.ascontiguousarray(WhiteModelRenderer._crop_resize_to_object(crop, rgb, mask))


def shade(vertices: Any, faces: Any, eye: Any, center: Any, up: Any, img_res: int, device: Any = "cuda",
          mesh_color: tuple[float, float, float] = MESH_COLOR,
          bg_color: tuple[float, float, float] = BG_COLOR) -> tuple[Any, Any]:
    """(rgb uint8 (img_res, img_res, 3), object mask bool (img_res, img_res)), rows top to bottom."""
    import numpy as np
    import torch

    dev = torch.device(device)
    verts = torch.as_tensor(np.ascontiguousarray(vertices), dtype=torch.float32, device=dev)
    tris = torch.as_tensor(np.ascontiguousarray(faces), dtype=torch.long, device=dev).reshape(-1, 3)

    # smooth vertex normals (area weighted), like Open3D's compute_vertex_normals()
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    face_n = torch.linalg.cross(v1 - v0, v2 - v0)
    normals = torch.zeros_like(verts)
    for i in range(3):
        normals.index_add_(0, tris[:, i], face_n)
    normals = torch.nn.functional.normalize(normals, dim=1)

    mvp = perspective(FOV, 1.0, 0.05, 100.0) @ look_at(np.asarray(eye), np.asarray(center), np.asarray(up))
    clip = torch.cat([verts, torch.ones_like(verts[:, :1])], dim=1) @ torch.as_tensor(mvp, dtype=torch.float32,
                                                                                      device=dev).T
    size = img_res * SUPERSAMPLE
    tri_id, bary = rasterize(clip, tris, size)
    covered = tri_id >= 0

    # Open3D's sun for up = y shines along (0.35, -1, 0.35); light arrives from the opposite direction
    sun = np.array([0.35, -1.0, 0.35])
    light = torch.as_tensor(-sun / np.linalg.norm(sun), dtype=torch.float32, device=dev)
    color = torch.as_tensor(bg_color, dtype=torch.float32, device=dev).repeat(size, size, 1)
    if bool(covered.any()):
        corners = tris[tri_id[covered]]                                       # (P, 3)
        n = (normals[corners] * bary[covered][..., None]).sum(1)
        n = torch.nn.functional.normalize(n, dim=-1)
        ndl = torch.clamp((n * light).sum(-1, keepdim=True), 0.0, 1.0)
        color[covered] = torch.as_tensor(mesh_color, dtype=torch.float32, device=dev) * (AMBIENT + (1 - AMBIENT) * ndl)
    # average SUPERSAMPLE x SUPERSAMPLE blocks in linear space, then gamma
    color = torch.nn.functional.avg_pool2d(color.permute(2, 0, 1)[None], SUPERSAMPLE)[0].permute(1, 2, 0)
    coverage = torch.nn.functional.avg_pool2d(covered.float()[None, None], SUPERSAMPLE)[0, 0]
    srgb = torch.clamp(color, 0.0, 1.0) ** (1.0 / GAMMA)
    rgb = (srgb * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    return rgb, (coverage > 0).cpu().numpy()


def rasterize(clip: Any, tris: Any, size: int) -> tuple[Any, Any]:
    """A plain z-buffer: clip-space vertices (V, 4) and triangles (F, 3) -> for every pixel of a size x size image
    (rows top to bottom, pixel centres at half-integers) the nearest triangle covering it (-1: none) and its
    perspective-correct barycentric weights (size, size, 3). Both windings are drawn (no culling), like the
    renderers it stands in for. Work goes in passes of CANDIDATES_PER_PASS (triangle, pixel) pairs."""
    import torch

    dev = clip.device
    w = clip[:, 3]
    ok_vert = w > 1e-6
    safe_w = torch.where(ok_vert, w, torch.ones_like(w))
    x = (clip[:, 0] / safe_w * 0.5 + 0.5) * size             # pixel units, 0 at the left edge
    y = (0.5 - clip[:, 1] / safe_w * 0.5) * size             # 0 at the top edge (image rows)
    z = clip[:, 2] / safe_w                                  # NDC depth: affine in screen space

    xs, ys = x[tris], y[tris]                                # (F, 3)
    area = (xs[:, 1] - xs[:, 0]) * (ys[:, 2] - ys[:, 0]) - (xs[:, 2] - xs[:, 0]) * (ys[:, 1] - ys[:, 0])
    # pixel centres (i + 0.5) inside each triangle's box
    x0 = torch.ceil(xs.min(1).values - 0.5).clamp(min=0).long()
    x1 = torch.floor(xs.max(1).values - 0.5).clamp(max=size - 1).long()
    y0 = torch.ceil(ys.min(1).values - 0.5).clamp(min=0).long()
    y1 = torch.floor(ys.max(1).values - 0.5).clamp(max=size - 1).long()
    live = ok_vert[tris].all(1) & (area.abs() > 1e-12) & (x1 >= x0) & (y1 >= y0)
    ids = live.nonzero(as_tuple=True)[0]
    width = (x1 - x0 + 1)[ids]
    counts = width * (y1 - y0 + 1)[ids]

    empty = torch.iinfo(torch.int64).max
    best = torch.full((size * size,), empty, dtype=torch.int64, device=dev)
    depth_scale = float(1 << 30)
    start = 0
    total = len(ids)
    ends = torch.cumsum(counts, 0)
    while start < total:
        # the triangles whose candidates fit into this pass (at least one)
        base = int(ends[start - 1]) if start else 0
        stop = int(torch.searchsorted(ends, base + CANDIDATES_PER_PASS, right=True))
        stop = max(stop, start + 1)
        seg = slice(start, stop)
        tri = torch.repeat_interleave(torch.arange(start, stop, device=dev), counts[seg])
        local = torch.arange(base, base + len(tri), device=dev) - (ends[tri] - counts[tri])   # pixel in its box
        f = ids[tri]
        px = x0[f] + local % width[tri]
        py = y0[f] + torch.div(local, width[tri], rounding_mode="floor")
        b = _barycentric(xs[f], ys[f], area[f], px.float() + 0.5, py.float() + 0.5)
        inside = (b >= 0).all(1)
        depth = (b * z[tris[f]]).sum(1)
        inside &= (depth >= -1.0) & (depth <= 1.0)
        key = ((depth[inside] * 0.5 + 0.5) * depth_scale).long() * (1 << 32) + f[inside]
        best.scatter_reduce_(0, (py * size + px)[inside], key, reduce="amin")
        start = stop

    hit = best != empty
    tri_id = torch.full((size * size,), -1, dtype=torch.long, device=dev)
    tri_id[hit] = best[hit] & 0xFFFFFFFF
    bary = torch.zeros((size * size, 3), dtype=torch.float32, device=dev)
    if bool(hit.any()):
        pix = hit.nonzero(as_tuple=True)[0]
        f = tri_id[pix]
        b = _barycentric(xs[f], ys[f], area[f], (pix % size).float() + 0.5,
                         torch.div(pix, size, rounding_mode="floor").float() + 0.5).clamp(min=0)
        b = b / safe_w[tris[f]]                                  # perspective-correct weights
        bary[pix] = b / b.sum(1, keepdim=True).clamp(min=1e-12)
    return tri_id.reshape(size, size), bary.reshape(size, size, 3)


def _barycentric(xs: Any, ys: Any, area: Any, px: Any, py: Any) -> Any:
    """Screen-space barycentric weights of the points (px, py) in the triangles (xs, ys) of signed `area`
    (any winding: the weights are positive inside)."""
    import torch

    w0 = (xs[:, 1] - px) * (ys[:, 2] - py) - (xs[:, 2] - px) * (ys[:, 1] - py)
    w1 = (xs[:, 2] - px) * (ys[:, 0] - py) - (xs[:, 0] - px) * (ys[:, 2] - py)
    w2 = (xs[:, 0] - px) * (ys[:, 1] - py) - (xs[:, 1] - px) * (ys[:, 0] - py)
    return torch.stack([w0, w1, w2], dim=1) / area[:, None]


def look_at(eye: Any, center: Any, up: Any) -> Any:
    """World -> camera (OpenGL: the camera looks down -z)."""
    import numpy as np

    fwd = center - eye
    fwd = fwd / max(np.linalg.norm(fwd), 1e-12)
    right = np.cross(fwd, up)
    n = np.linalg.norm(right)
    if n < 1e-9:     # looking straight along `up`: any stable basis
        right = np.cross(fwd, np.roll(up, 1))
        n = np.linalg.norm(right)
    right = right / max(n, 1e-12)
    view = np.eye(4, dtype=np.float64)
    view[0, :3], view[1, :3], view[2, :3] = right, np.cross(right, fwd), -fwd
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def perspective(fov_deg: float, aspect: float, near: float, far: float) -> Any:
    """OpenGL projection for a vertical field of view in degrees."""
    import numpy as np

    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = 2.0 * far * near / (near - far)
    proj[3, 2] = -1.0
    return proj
