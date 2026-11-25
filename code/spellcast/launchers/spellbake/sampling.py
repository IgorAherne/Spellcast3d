"""Reading the high poly's texture maps at hit points (design section 4.4, step L12).

Colour maps (base colour, emissive) are sRGB-encoded: they are decoded to linear light
through a lookup table *before* any filtering, so averages (mip levels, bilinear taps and
the bake's supersamples) are physically correct; their pyramids hold 16-bit linear
values. Data maps (normal, ORM, height) are filtered as stored: level 0 keeps the file's
own bytes and the smaller levels are 16-bit, so no level drifts by rounding. Alpha is
always linear.

Lookups are trilinear, GPU style: texel centres at (i + 1/2) / size, wrap modes as glTF
samplers name them, and the level chosen per sample from the ratio of the low poly's
sample size to the high poly's texel size (``mip_level``, from ``uv_jacobians``).
"""

from __future__ import annotations

import numpy as np

# glTF sampler wrap modes.
REPEAT = 10497
CLAMP_TO_EDGE = 33071
MIRRORED_REPEAT = 33648


def _srgb_eotf(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb_oetf(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


# Linear light of every 8-bit sRGB code, and the same in 16-bit.
SRGB_TO_LINEAR = _srgb_eotf(np.arange(256) / 255.0)
SRGB_TO_LINEAR16 = np.round(SRGB_TO_LINEAR * 65535.0).astype(np.uint16)


def srgb_to_linear(codes: np.ndarray) -> np.ndarray:
    """8-bit sRGB codes -> linear light in [0, 1] (float32)."""
    return SRGB_TO_LINEAR.astype(np.float32)[np.asarray(codes, dtype=np.uint8)]


def linear_to_srgb8(linear: np.ndarray) -> np.ndarray:
    """Linear light -> the nearest 8-bit sRGB code (the inverse of ``srgb_to_linear`` on
    all 256 codes)."""
    x = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    return np.round(_srgb_oetf(x) * 255.0).astype(np.uint8)


def unorm8(values: np.ndarray) -> np.ndarray:
    """[0, 1] -> round(255 * v) for data maps."""
    return np.round(np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0) * 255.0).astype(np.uint8)


def _half(level: np.ndarray, scale: float) -> np.ndarray:
    """The next mip level, uint16 in [0, 65535]: the 2 x 2 box average of ``level`` (whose
    values times ``scale`` are in [0, 1]), an odd last row or column repeated."""
    h, w = level.shape[:2]
    if h > 1 and h % 2:
        level = np.concatenate([level, level[-1:]], axis=0)
    if w > 1 and w % 2:
        level = np.concatenate([level, level[:, -1:]], axis=1)
    h, w = level.shape[:2]
    rows, cols = max(1, h // 2), max(1, w // 2)
    fy, fx = h // rows, w // cols
    out = np.empty((rows, cols, level.shape[2]), dtype=np.uint16)
    to16 = np.float32(scale * 65535.0)
    # Row strips bound the float32 temporary of a 4096^2 level.
    step = max(1, (1 << 22) // max(1, cols * fx * fy * level.shape[2]))
    for r in range(0, rows, step):
        block = level[r * fy:(r + step) * fy].astype(np.float32) * to16
        block = block.reshape(-1, fy, cols, fx, level.shape[2]).mean(axis=(1, 3))
        out[r:r + step] = np.round(block).astype(np.uint16)
    return out


def _wrap(index: np.ndarray, size: int, mode: int) -> np.ndarray:
    if mode == CLAMP_TO_EDGE:
        return np.clip(index, 0, size - 1)
    if mode == MIRRORED_REPEAT:
        period = np.mod(index, 2 * size)
        return np.where(period < size, period, 2 * size - 1 - period)
    return np.mod(index, size)


class MipPyramid:
    """A texture map and its box-filtered mip chain down to 1 x 1.

    ``image`` is (H, W) or (H, W, C), uint8 or uint16. With ``srgb_channels`` = k > 0 the
    first k channels of an 8-bit image are sRGB colour, and every level is kept as
    16-bit linear light. ``levels[i] * scales[i]`` is in [0, 1]; ``sample`` returns linear
    values in [0, 1].
    """

    def __init__(self, image: np.ndarray, *, srgb_channels: int = 0) -> None:
        image = np.asarray(image)
        if image.ndim == 2:
            image = image[..., None]
        if image.dtype not in (np.uint8, np.uint16):
            raise ValueError(f"texture maps are uint8 or uint16, not {image.dtype}")
        if srgb_channels:
            if image.dtype != np.uint8:
                raise ValueError("sRGB maps are 8-bit")
            base = image.astype(np.uint16) * np.uint16(257)
            base[..., :srgb_channels] = SRGB_TO_LINEAR16[image[..., :srgb_channels]]
        else:
            base = np.ascontiguousarray(image)
        self.levels = [base]
        self.scales = [1.0 / np.iinfo(base.dtype).max]
        while max(self.levels[-1].shape[:2]) > 1:
            self.levels.append(_half(self.levels[-1], self.scales[-1]))
            self.scales.append(1.0 / 65535.0)

    @property
    def width(self) -> int:
        return self.levels[0].shape[1]

    @property
    def height(self) -> int:
        return self.levels[0].shape[0]

    @property
    def channels(self) -> int:
        return self.levels[0].shape[2]

    def bilinear(self, level: int, u: np.ndarray, v: np.ndarray, wrap_s: int, wrap_t: int) -> np.ndarray:
        """(n, C) float32 at v-up coordinates (u, v) of one level."""
        image = self.levels[level]
        h, w, c = image.shape
        x = np.nan_to_num(np.asarray(u, dtype=np.float64)) * w - 0.5
        y = (1.0 - np.nan_to_num(np.asarray(v, dtype=np.float64))) * h - 0.5
        x0 = np.floor(x)
        y0 = np.floor(y)
        fx = (x - x0).astype(np.float32)[:, None]
        fy = (y - y0).astype(np.float32)[:, None]
        x0 = x0.astype(np.int64)
        y0 = y0.astype(np.int64)
        xa, xb = _wrap(x0, w, wrap_s), _wrap(x0 + 1, w, wrap_s)
        ya, yb = _wrap(y0, h, wrap_t) * w, _wrap(y0 + 1, h, wrap_t) * w
        flat = image.reshape(-1, c)
        top = flat[ya + xa] * (1 - fx) + flat[ya + xb] * fx
        bottom = flat[yb + xa] * (1 - fx) + flat[yb + xb] * fx
        return ((top * (1 - fy) + bottom * fy) * np.float32(self.scales[level])).astype(np.float32)

    def sample(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lod: np.ndarray | float = 0.0,
        wrap_s: int = REPEAT,
        wrap_t: int = REPEAT,
    ) -> np.ndarray:
        """(n, C) float32 trilinear lookups; ``lod`` is the (fractional) mip level."""
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        top = len(self.levels) - 1
        lod = np.clip(np.nan_to_num(np.broadcast_to(np.asarray(lod, dtype=np.float64), u.shape)), 0.0, top)
        base = np.floor(lod).astype(np.int64)
        frac = (lod - base).astype(np.float32)[:, None]
        out = np.empty((len(u), self.channels), dtype=np.float32)
        for level in np.unique(base):
            rows = np.flatnonzero(base == level)
            value = self.bilinear(int(level), u[rows], v[rows], wrap_s, wrap_t)
            if level < top:
                blend = rows[frac[rows, 0] > 0]
                if blend.size:
                    part = np.searchsorted(rows, blend)
                    upper = self.bilinear(int(level) + 1, u[blend], v[blend], wrap_s, wrap_t)
                    value[part] += (upper - value[part]) * frac[blend]
            out[rows] = value
        return out


def uv_jacobians(positions: np.ndarray, uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(nF, 2, 2) d(u, v) / d(s, t) of every triangle: UV units per world unit along an
    orthonormal frame (s, t) of its plane (zeros when the triangle has no 3D area)."""
    p = np.asarray(positions, dtype=np.float64)[faces]
    t = np.asarray(uv, dtype=np.float64)[faces]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    s_axis = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-300)
    normal = np.cross(e1, e2)
    t_axis = np.cross(normal, s_axis)
    t_axis /= np.maximum(np.linalg.norm(t_axis, axis=1, keepdims=True), 1e-300)
    # World edges in (s, t), and the UV edges they map to: J @ planar = uv_edges.
    planar = np.stack([np.stack([np.einsum("ij,ij->i", e, s_axis) for e in (e1, e2)], axis=1),
                       np.stack([np.einsum("ij,ij->i", e, t_axis) for e in (e1, e2)], axis=1)], axis=1)
    uv_edges = np.stack([t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]], axis=2)
    det = planar[:, 0, 0] * planar[:, 1, 1] - planar[:, 0, 1] * planar[:, 1, 0]
    ok = np.abs(det) > 0
    inverse = np.zeros_like(planar)
    inverse[ok, 0, 0], inverse[ok, 1, 1] = planar[ok, 1, 1] / det[ok], planar[ok, 0, 0] / det[ok]
    inverse[ok, 0, 1], inverse[ok, 1, 0] = -planar[ok, 0, 1] / det[ok], -planar[ok, 1, 0] / det[ok]
    return uv_edges @ inverse


def mip_level(footprint: np.ndarray, jacobian: np.ndarray, width: int, height: int) -> np.ndarray:
    """lambda = max(0, log2(rho_low * tau_h)) per sample.

    rho_low is the world size of one bake sample; tau_h the high map's texels per world
    unit on the hit triangle, taken along the direction where the map is *least* dense
    (the smaller singular value of diag(W, H) @ J). On an evenly unwrapped triangle that
    is sqrt(W * H * A_uv / A_3D); on a stretched one (the pole of a latitude-longitude map)
    the geometric mean would blur the short direction by many texels."""
    m = np.asarray(jacobian, dtype=np.float64) * np.array([[width], [height]], dtype=np.float64)
    frobenius = np.einsum("nij,nij->n", m, m)
    det = m[:, 0, 0] * m[:, 1, 1] - m[:, 0, 1] * m[:, 1, 0]
    smallest = np.sqrt(np.maximum(frobenius - np.sqrt(np.maximum(frobenius ** 2 - 4 * det ** 2, 0.0)), 0.0) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        level = np.log2(np.asarray(footprint, dtype=np.float64) * smallest)
    return np.nan_to_num(np.maximum(level, 0.0), nan=0.0, posinf=64.0).astype(np.float32)
