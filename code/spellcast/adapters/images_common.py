"""What the image generators share (adapters/qwen.py, adapters/forge.py): the prop's example pictures they read, and
how their pictures are saved.

A generation's pictures become new candidates images/NNNN.png (numbered after every picture the prop has, imported
ones included). The first of them becomes the chosen picture, also when the prop had one: the page shows what
Generate has just made (the store keeps pictures imported meanwhile among the candidates). A generation that made
none leaves the chosen picture as it is. last_seeds are the seeds they were made with.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

from ..projects import VIEWPORT_FILE, VIEWPORT_MASK
from .base import StageContext, StageError

RENAME_TRIES = 5


def reference_files(ctx: StageContext, images: dict[str, Any], params: dict[str, Any] | None = None) -> list[Path]:
    """The prop's example pictures (images.references, the "Describe + example" mode), each checked to be there.
    None of them when the job's params say "use_references": false (the page's "Describe" mode: from the words alone;
    the example pictures stay for later). The params are the prop's images settings with the job's own over them, so
    the mode the page saved with the prop (images.settings.use_references) counts too.
    With "from_viewport" (the page's "Re-Render Viewport": one of the prop's pictures, or VIEWPORT_FILE when the
    artist painted marks over it or on the empty viewport) that picture comes first, before the example pictures (none
    with "use_references": false): Qwen-Image is shown it beside the words, Forge Neo starts from it."""
    lead = [viewport_file(ctx, images, params["from_viewport"])] \
        if params is not None and params.get("from_viewport") is not None else []
    if params is not None and params.get("use_references") is False:
        return lead
    files: list[Path] = []
    for rel in images.get("references") or []:
        path = _prop_file(ctx, rel)
        if path is None:
            raise StageError(f"One of this prop's example pictures is missing ({rel}). Remove it and add it again.")
        files.append(path)
    return lead + files


def _prop_file(ctx: StageContext, rel: Any) -> Path | None:
    """`rel` (a path relative to the prop) as a file inside the prop's folder; None when it is none."""
    path = ctx.abs(rel) if isinstance(rel, str) else None
    return path if path is not None and path.resolve().is_relative_to(ctx.prop_dir.resolve()) and path.is_file() else None


def viewport_file(ctx: StageContext, images: dict[str, Any], rel: Any) -> Path:
    """The picture "Re-Render Viewport" re-renders: one of the prop's pictures (images.candidates) or the marked copy
    the page uploaded (VIEWPORT_FILE), checked to be there."""
    ok = isinstance(rel, str) and (rel == VIEWPORT_FILE or rel in (images.get("candidates") or []))
    path = _prop_file(ctx, rel) if ok else None
    if path is None:
        raise StageError("The picture to re-render is gone. Pick a picture under the viewport and press Re-Render "
                         "Viewport again.")
    return path


def masked_change(ctx: StageContext, images: dict[str, Any], params: dict[str, Any]) -> tuple[Path, Path] | None:
    """A Smart Mask job - its "viewport_mask_of" names the prop's picture the artist brushed over - -> that picture,
    as it is, and the mask the page uploaded with its copy of it under the mask's magenta (VIEWPORT_MASK), both
    checked to be there; that copy is the job's viewport picture (from_viewport). None for any other job."""
    rel = params.get("viewport_mask_of")
    if rel is None:
        return None
    picture = _prop_file(ctx, rel) if rel in (images.get("candidates") or []) else None
    mask = _prop_file(ctx, VIEWPORT_MASK)
    if picture is None or mask is None or params.get("from_viewport") != VIEWPORT_FILE:
        raise StageError("The mask, or the picture under it, is gone. Brush over the picture again and press Render.")
    return picture, mask


def save_candidates(ctx: StageContext, images: dict[str, Any], pictures: Iterable[bytes]) -> list[str]:
    """Write the pictures as images/NNNN.png (.jpg / .webp by their bytes) -> their paths relative to the prop."""
    out_dir = ctx.prop_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    number = next_number(out_dir, images.get("candidates") or [])
    saved: list[str] = []
    for raw in pictures:
        target = out_dir / f"{number:04d}{image_extension(raw)}"
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            tmp.write_bytes(raw)
            for attempt in range(RENAME_TRIES):
                try:
                    tmp.replace(target)
                    break
                except PermissionError:     # an antivirus or the search indexer holds the new file for a moment
                    if attempt == RENAME_TRIES - 1:
                        raise
                    time.sleep(0.2 * (attempt + 1))
        finally:
            tmp.unlink(missing_ok=True)
        saved.append(ctx.rel(target))
        number += 1
    return saved


def results_patch(images: dict[str, Any], new_rel: list[str], seeds: list[int]) -> dict[str, Any]:
    """The images patch of a generation that made `new_rel` (see the module docstring). One that made none says
    nothing about the chosen picture: `images` is the prop as the job found it, and a picture imported (and
    chosen) while it ran stays the chosen one."""
    patch: dict[str, Any] = {"candidates": list(images.get("candidates") or []) + new_rel}
    if new_rel:
        patch["chosen"] = new_rel[0]
        patch["last_seeds"] = seeds
    return patch


def next_number(folder: Path, candidates: list[str]) -> int:
    stems = [p.stem for p in folder.iterdir()] + [Path(c).stem for c in candidates]
    return max((int(s) for s in stems if s.isdigit()), default=0) + 1


def image_extension(raw: bytes) -> str:
    """Forge encodes results in its "samples_format" option (png unless the user changed it); Qwen writes PNG."""
    if raw.startswith(b"\xff\xd8"):
        return ".jpg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return ".webp"
    return ".png"
