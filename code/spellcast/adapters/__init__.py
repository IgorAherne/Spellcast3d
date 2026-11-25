"""Tool adapters, keyed by tool id. See base.py for the contract."""

from __future__ import annotations

from .. import paths
from .base import LaunchSpec, StageContext, StageError, ToolAdapter
from .forge import ForgeAdapter
from .imesh import ImeshAdapter
from .lato2 import Lato2Adapter
from .pixal3d import Pixal3DAdapter
from .qwen import QwenAdapter
from .triflow import TriflowAdapter

# The image generators, one of which the root config file picks (spellcast/config.py image_backend), in the order
# of paths.STAGE_TOOLS["images"] (the default first).
IMAGE_ADAPTERS: dict[str, type[ToolAdapter]] = {paths.QWEN: QwenAdapter, paths.FORGE: ForgeAdapter}


def build_adapters(image_backend: str | None = None) -> dict[str, ToolAdapter]:
    """Every tool's adapter, with ONE image generator: `image_backend` ("qwen" | "forge"), the one the root config
    file picks. Any other text (config.image_backend() gives None when the file is not valid: "") registers no image
    generator at all. None registers both (tests, prepare_release)."""
    return {
        **{tool: adapter() for tool, adapter in IMAGE_ADAPTERS.items() if image_backend in (None, tool)},
        paths.PIXAL3D: Pixal3DAdapter(),
        paths.IMESH: ImeshAdapter(),
        paths.LATO2: Lato2Adapter(),
        paths.TRIFLOW: TriflowAdapter(),
    }


__all__ = ["LaunchSpec", "StageContext", "StageError", "ToolAdapter", "ForgeAdapter", "ImeshAdapter", "Lato2Adapter",
           "Pixal3DAdapter", "QwenAdapter", "TriflowAdapter", "IMAGE_ADAPTERS", "build_adapters"]
