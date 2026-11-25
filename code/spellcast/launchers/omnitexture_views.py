"""The cameras Omni-Texture's worker looks at the model through, and what each sees of it.

The stage's cameras (VIEWS) see the model as its own pictures show it, since Qwen-Image draws it the way those do
whatever the depth map says: level with it and flat. They are orthographic, so the most of the surface faces a
camera squarely. A repaint looks through the page's own camera instead, a perspective one (Lens)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from omnitexture_tone import CLAY, GAMMA

FACING = 0.5                 # the cosine a surface must face a camera with to be in its mask and take its picture
MARGIN = 1.08                # the model's box fills the picture but this much
RING_ELEVATION = 0.0         # degrees the six cameras around the model look down at it
# From above and below at a slant, not straight down or up: the rest of the figure stays in sight, so Qwen-Image
# knows what it looks at (straight down, a character's cap is painted as a face).
SLANT = 65.0
# (name, side, azimuth, elevation). THE FRONT AND THE BACK STAY FIRST, in this order: they are painted with
# nothing painted in their sight, and the words they are painted with say so (adapters/omnitexture.py ALONE).
VIEWS = (
    ("front", "front", 0.0, RING_ELEVATION),
    ("back", "back", 180.0, RING_ELEVATION),
    ("front_right", "front right", 60.0, RING_ELEVATION),
    ("front_left", "front left", -60.0, RING_ELEVATION),
    ("back_right", "back right", 120.0, RING_ELEVATION),
    ("back_left", "back left", -120.0, RING_ELEVATION),
    ("top", "top", 0.0, SLANT),
    ("bottom", "bottom", 0.0, -SLANT),
)
AROUND = 6                   # the first of VIEWS are the cameras around the model
DEPTH_FAR = 0.1              # the farthest surface's grey in the depth map (the nearest is white, the background black)
# The model as the page's bare clay (CLAY), lit, which the settings' test may show in the depth map's place:
# Lambert, no shadows cast, the model's own normals, so its shapes show as light and shade.
LIGHT = (0.4, 0.3, 1.0)      # towards the light, along the camera's right / up / back: slightly to the side, above
LIT = (0.35, 2.3)            # the clay's linear colour times (ambient + this x the light's cosine)


@dataclass
class Camera:
    """Orthographic: every pixel looks along `forward`, from its own place on the picture plane through `eye`.
    Its pixels are the view's own part of the picture: all of it, or the view's window in a sheet."""
    eye: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    left: float              # that part's left and top edges on the plane, from `eye`, in the model's units
    top: float
    ppu: float               # pixels per unit of the model
    width: int               # that part's size, pixels
    height: int

    def origins(self) -> np.ndarray:
        """Where each pixel's ray starts on the picture plane, row-major (they all run along `forward`)."""
        across = self.eye + (self.left + (np.arange(self.width) + 0.5) / self.ppu)[:, None] * self.right
        down = (self.top - (np.arange(self.height) + 0.5) / self.ppu)[:, None] * self.up
        return (across[None, :, :] + down[:, None, :]).reshape(-1, 3)

    def place(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Where the points lie in the picture (x, y: a pixel's middle at its whole number) and how deep."""
        rel = points - self.eye
        return (rel @ self.right - self.left) * self.ppu - 0.5, (self.top - rel @ self.up) * self.ppu - 0.5, \
            rel @ self.forward

    def facing(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """How squarely the camera faces the surface at each point (the cosine)."""
        return normals @ -self.forward

    def step(self, depth: np.ndarray) -> float:
        """A pixel's side at that depth, in the model's units."""
        return 1.0 / self.ppu


@dataclass
class Lens:
    """A camera as the page's viewport is one: every pixel looks from `eye`. focal: the picture's pixels per unit
    of the tangent of the angle from `forward`. It looks through the middle of its square picture."""
    eye: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    focal: float
    width: int
    height: int

    @classmethod
    def looking(cls, eye: list[float], forward: list[float], up: list[float], tan: float, size: int) -> Lens | None:
        """The lens at `eye` looking along `forward`, `up` up, its square picture of `size` px spanning the half
        angle of tangent `tan`; None when forward and up are one line, or the angle is none."""
        f = np.asarray(forward, np.float64)
        f /= max(np.linalg.norm(f), 1e-12)
        r = np.cross(f, np.asarray(up, np.float64))
        if np.linalg.norm(r) < 1e-9 or not np.isfinite(tan) or tan <= 0:
            return None
        r /= np.linalg.norm(r)
        return cls(np.asarray(eye, np.float64), f, r, np.cross(r, f), size / 2.0 / float(tan), size, size)

    def rays(self) -> np.ndarray:
        """Each pixel's direction from `eye` (unit), row-major."""
        across = ((np.arange(self.width) + 0.5 - self.width / 2.0) / self.focal)[:, None] * self.right
        down = ((self.height / 2.0 - np.arange(self.height) - 0.5) / self.focal)[:, None] * self.up
        out = (self.forward + across[None, :, :] + down[:, None, :]).reshape(-1, 3)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def place(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rel = points - self.eye
        z = rel @ self.forward
        scale = self.focal / np.maximum(z, 1e-12)
        return self.width / 2.0 + (rel @ self.right) * scale - 0.5, \
            self.height / 2.0 - (rel @ self.up) * scale - 0.5, z

    def facing(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        back = self.eye - points
        return (normals * back).sum(axis=1) / np.maximum(np.linalg.norm(back, axis=1), 1e-12)

    def step(self, depth: np.ndarray) -> np.ndarray:
        return depth / self.focal


def frame(positions: np.ndarray, azimuth: float,
          elevation: float) -> tuple[tuple[np.ndarray, ...], tuple[float, float, float, float]]:
    """A view's (eye, forward, right, up) and the model's extents on its picture plane, from `eye`: (x0, x1, y0,
    y1) along right and up."""
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    centre = (lo + hi) / 2.0
    radius = max(float(np.linalg.norm(hi - lo)) / 2.0, 1e-9)
    a, e = np.radians(azimuth), np.radians(elevation)
    back = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    eye = centre + back * radius * 2.0          # outside the model: all of it lies in front of the picture plane
    forward = -back
    if abs(elevation) < 80.0:
        hint = np.array([0.0, 1.0, 0.0])
    else:                    # from above or below: the model's front at the picture's bottom (above) or top (below)
        hint = -np.sign(elevation) * np.array([np.sin(a), 0.0, np.cos(a)])
    right = np.cross(forward, hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rel = positions - eye
    x, y = rel @ right, rel @ up
    return (eye, forward, right, up), (float(x.min()), float(x.max()), float(y.min()), float(y.max()))


def camera_alone(positions: np.ndarray, azimuth: float, elevation: float, size: int = 1024) -> Camera:
    """The view alone in a square picture of `size` px: the model's box in its middle, filling it but MARGIN."""
    axes, (x0, x1, y0, y1) = frame(positions, azimuth, elevation)
    half = max(x1 - x0, y1 - y0) / 2.0 * MARGIN
    return Camera(*axes, (x0 + x1) / 2.0 - half, (y0 + y1) / 2.0 + half, size / (2.0 * half), size, size)


def lit_clay(normal: np.ndarray, cam: Camera) -> np.ndarray:
    """(n, 3) unit normals -> the clay's sRGB colour under the camera's light (LIGHT, LIT), uint8 (n, 3)."""
    light = np.array(LIGHT[0] * cam.right + LIGHT[1] * cam.up - LIGHT[2] * cam.forward)
    light /= np.linalg.norm(light)
    clay = (np.array(CLAY) / 255.0) ** GAMMA
    shade = LIT[0] + LIT[1] * np.clip(normal @ light, 0.0, None)
    return np.round(255.0 * np.clip(clay * shade[:, None], 0.0, 1.0) ** (1.0 / GAMMA)).astype(np.uint8)


def normal_map(normal: np.ndarray, cam: Camera) -> np.ndarray:
    """(n, 3) unit normals -> their colours in a normal map as the camera sees it: red towards its right, green up,
    blue towards the camera (uint8 (n, 3)). Small relief the depth map shows in a grey level or two - an eye
    socket - is a clear change of colour here."""
    seen = np.stack([normal @ cam.right, normal @ cam.up, normal @ -cam.forward], axis=1)
    return np.round(127.5 + 127.5 * np.clip(seen, -1.0, 1.0)).astype(np.uint8)


def depth_grey(depth: np.ndarray) -> np.ndarray:
    """The depth map's grey of each depth a camera sees: white the nearest, DEPTH_FAR the farthest, whatever lies
    between in a straight line (the background stays black)."""
    near, far = (float(depth.min()), float(depth.max())) if depth.size else (0.0, 0.0)
    return np.round(255.0 - (255.0 - 255.0 * DEPTH_FAR) * (depth - near) / max(far - near, 1e-9)).astype(np.uint8)


@dataclass
class Met:
    """Where a camera's rays meet the model: per pixel whether it does (`hit`), and per such pixel, in order, its
    triangle's corners and its place between them (the weights of the three)."""
    hit: np.ndarray
    corners: np.ndarray
    weights: tuple[np.ndarray, np.ndarray, np.ndarray]
    t: np.ndarray            # how far along its ray

    def between(self, per_vertex: np.ndarray) -> np.ndarray:
        """A per-vertex quantity where each ray met the model."""
        w0, w1, w2 = self.weights
        f = self.corners
        return (w0 * per_vertex[f[:, 0]] + w1 * per_vertex[f[:, 1]]) + w2 * per_vertex[f[:, 2]]

    def unit(self, per_vertex: np.ndarray) -> np.ndarray:
        """A per-vertex normal where each ray met the model, at unit length."""
        normal = self.between(per_vertex)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        return normal


def cast(scene: Any, faces: np.ndarray, origins: np.ndarray, directions: np.ndarray) -> Met:
    """The rays cast into the scene (spellbake.rays.Scene): the first surface each meets from its front."""
    hits = scene.first_front_hit(origins, directions, np.inf)
    hit = hits.prim >= 0
    u, v = hits.u[hit], hits.v[hit]
    return Met(hit, faces[hits.prim[hit]], ((1.0 - u - v)[:, None], u[:, None], v[:, None]), hits.t[hit])


@dataclass
class Sight:
    """What a camera sees of the model, whatever is painted: per pixel whether it meets the model, per such pixel
    (`ids`, in order) its texel, whether its surface faces the camera at FACING or more, its depth map grey, its
    lit clay colour and its normal map colour, and the depth each pixel sees (inf: no model)."""
    hit: np.ndarray
    ids: np.ndarray
    texel: np.ndarray
    facing: np.ndarray
    grey: np.ndarray
    clay: np.ndarray
    normal: np.ndarray
    depth: np.ndarray


def see(cam: Camera, scene: Any, mesh: Any, smooth: np.ndarray, res: int) -> Sight:
    """What `cam` sees of the model (`mesh`: its faces, uv and own normals; `smooth`: its smoothed normals, which
    the FACING rule reads) with a texture of `res` texels a side."""
    origins = cam.origins()
    met = cast(scene, mesh.faces, origins, np.broadcast_to(cam.forward, origins.shape))
    uv = met.between(mesh.uv)
    col = np.clip((uv[:, 0] * res).astype(np.int64), 0, res - 1)
    row = np.clip(((1.0 - uv[:, 1]) * res).astype(np.int64), 0, res - 1)
    depth = met.t                                # along forward, from the picture plane
    buffer = np.full(met.hit.size, np.inf)
    buffer[met.hit] = depth
    facing = met.unit(smooth) @ -cam.forward >= FACING
    own = met.unit(mesh.normals)
    return Sight(met.hit, np.flatnonzero(met.hit), row * res + col, facing, depth_grey(depth), lit_clay(own, cam),
                 normal_map(own, cam), buffer.reshape(cam.height, cam.width))
