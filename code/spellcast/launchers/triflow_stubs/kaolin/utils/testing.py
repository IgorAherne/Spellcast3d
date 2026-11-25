"""check_tensor, the one kaolin function FlexiCubes imports (see ../__init__.py). Spellcast3D's own
small version of it: TriFlow's inference never calls it, but if something ever does, it checks."""

from __future__ import annotations

from typing import Any, Sequence


def check_tensor(tensor: Any, shape: Sequence[int | None] | None = None, dtype: Any = None,
                 device: Any = None, throw: bool = True) -> bool:
    """True when `tensor` has `shape` (None: any size on that axis), `dtype` and `device` (a type such as
    "cuda", or a torch.device), each only when given. Otherwise raises ValueError, or returns False when
    `throw` is False."""
    problem = None
    if shape is not None:
        size = tuple(tensor.shape)
        if len(size) != len(shape) or any(want is not None and got != want for got, want in zip(size, shape)):
            problem = f"tensor shape is {size}, should be {tuple(shape)}"
    if problem is None and dtype is not None and tensor.dtype != dtype:
        problem = f"tensor dtype is {tensor.dtype}, should be {dtype}"
    if problem is None and device is not None:
        want = getattr(device, "type", device)
        if tensor.device.type != want:
            problem = f"tensor device is {tensor.device}, should be {device}"
    if problem is None:
        return True
    if throw:
        raise ValueError(problem)
    return False
