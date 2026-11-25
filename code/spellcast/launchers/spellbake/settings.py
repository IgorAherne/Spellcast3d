"""The settings of one bake, checked (design section 5.4).

The hub's adapter (adapters/projection.py) checks the artist's settings before it asks for a
bake; the worker checks them again here, because it also runs from tests and by hand. Keep the
ranges in sync with that module and with projects.STAGE_DEFAULTS["projection"].
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from . import BakeError

TEXTURE_SIZES = (512, 1024, 2048, 4096)
QUALITIES = (1, 2, 3)                       # samples per texel side: Fast, Good, Best
RAY_DISTANCE_PERCENT = (0.05, 20.0)         # of the high poly's bounding-box diagonal
AO_DISTANCE_PERCENT = (0.1, 1000.0)         # of the diagonal; None = DEFAULT_AO_DISTANCE_PERCENT
DEFAULT_AO_DISTANCE_PERCENT = 10.0
AO_RAYS_STEP = 32                           # ambient-occlusion rays per texel come in sets of 32
MAX_AO_RAYS = 256
MAX_HARD_EDGE_ANGLE = 180.0                 # degrees; -1 = every edge smooth
MAX_RING = 32.0                             # texels
TANGENT_BASES = ("mikk", "per_vertex")
NORMAL_Y = ("opengl", "directx")


@dataclass(frozen=True)
class BakeSettings:
    """What to bake, and how.

    ``ray_distance`` and ``ao_distance`` are percentages of the high poly's bounding-box
    diagonal; None means automatic (L10 for the ray distance, 10% for AO). ``ring`` is the
    extrapolation ring around UV islands in texels; None lets the low poly decide (4 when
    the islands were repacked, 8 on the retopo's own layout).
    """

    texture_size: int = 2048
    quality: int = 2
    ray_distance: float | None = None
    ao: bool = True
    ao_rays: int = 32
    ao_distance: float | None = None
    height: bool = False
    hard_edge_angle: float = 60.0
    tangent_basis: str = "mikk"
    normal_y: str = "opengl"
    repack_uvs: bool = True
    export_obj: bool = True
    export_fbx: bool = True
    ring: float | None = None

    def __post_init__(self) -> None:
        for name in ("ao", "height", "repack_uvs", "export_obj", "export_fbx"):
            if not isinstance(getattr(self, name), bool):
                raise BakeError(f'The bake setting "{name}" must be on or off (true or false).')
        size = _integer(self.texture_size, "Texture size")
        if size not in TEXTURE_SIZES:
            raise BakeError("Texture size must be 512, 1024, 2048 or 4096 pixels.")
        quality = _integer(self.quality, "Quality")
        if quality not in QUALITIES:
            raise BakeError("Quality must be 1 (Fast), 2 (Good) or 3 (Best).")
        rays = _integer(self.ao_rays, "Ambient occlusion rays")
        if rays % AO_RAYS_STEP or not AO_RAYS_STEP <= rays <= MAX_AO_RAYS:
            raise BakeError(f"Ambient occlusion rays must be a multiple of {AO_RAYS_STEP} "
                            f"from {AO_RAYS_STEP} to {MAX_AO_RAYS}.")
        ray = _percent(self.ray_distance, "Ray distance", RAY_DISTANCE_PERCENT)
        ao_distance = _percent(self.ao_distance, "Ambient occlusion distance", AO_DISTANCE_PERCENT)
        hard = _number(self.hard_edge_angle, "Hard edge angle")
        if hard != -1 and not 0 <= hard <= MAX_HARD_EDGE_ANGLE:
            raise BakeError("Hard edge angle must be between 0° and 180°, or -1 to keep every edge smooth.")
        ring = None if self.ring is None else _number(self.ring, "Ring")
        if ring is not None and not 0 <= ring <= MAX_RING:
            raise BakeError(f"The ring around UV islands must be between 0 and {MAX_RING:g} texels.")
        if self.tangent_basis not in TANGENT_BASES:
            raise BakeError("Tangent basis must be MikkTSpace ('mikk': Blender, Unreal, Unity HDRP) or "
                            "per vertex ('per_vertex': three.js, Unity URP, Godot).")
        if self.normal_y not in NORMAL_Y:
            raise BakeError("The normal map must be made for OpenGL ('opengl': Blender, Unity, Godot, glTF) "
                            "or DirectX ('directx': Unreal, 3ds Max).")
        # Settings arrive as JSON: store 2048.0 as 2048 and 60 as 60.0.
        for name, value in (("texture_size", size), ("quality", quality), ("ao_rays", rays),
                            ("ray_distance", ray), ("ao_distance", ao_distance),
                            ("hard_edge_angle", hard), ("ring", ring)):
            object.__setattr__(self, name, value)

    @property
    def supersample(self) -> int:
        """S: each texel is sampled S x S times."""
        return self.quality

    @classmethod
    def from_json(cls, raw: Any) -> BakeSettings:
        """Settings from a request's JSON object; missing keys take the defaults."""
        if not isinstance(raw, Mapping):
            raise BakeError("The bake settings must be a JSON object.")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise BakeError(f"Unknown bake setting(s): {', '.join(map(str, unknown))}. "
                            f"Update Spellcast3D so the hub and the baker agree.")
        return cls(**raw)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BakeError(f"{label} must be a number (it is {value!r}).")
    return float(value)


def _integer(value: Any, label: str) -> int:
    number = _number(value, label)
    if number != int(number):
        raise BakeError(f"{label} must be a whole number (it is {value!r}).")
    return int(number)


def _percent(value: Any, label: str, bounds: tuple[float, float]) -> float | None:
    if value is None:
        return None
    number = _number(value, label)
    low, high = bounds
    if not low <= number <= high:
        raise BakeError(f"{label} must be automatic, or between {low:g} % and {high:g} % of the model's "
                        f"size (it is {number:g} %).")
    return number
