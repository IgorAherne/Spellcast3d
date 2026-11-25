"""What the retopology adapters share (TriFlow, LATO.2 and Instant Meshes): a job makes each of its props one clean mesh
per size the prop asks for (retopo.settings.sizes, in thousands of vertices: 0.25 is 250), its variants, one after the
other, each in a folder of its own (retopo/<size>k/: retopo/0.25k/, retopo/2k/), with progress words that name the size.
The hub keeps them as the prop's retopo.variants (projects._variants_patch) and the artist picks one.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Awaitable, Callable, Mapping

from .. import projects
from .base import StageContext, StageError, ToolNeedsRestart

NO_SIZES = ("This prop asks for no clean mesh: type how many thousand vertices it should have in a box on its "
            "tile in the Retopo stage.")


def sizes_of(settings: Mapping[str, Any]) -> list[float]:
    """The sizes a prop's settings ask for (its boxes, in their order). StageError while they ask for none."""
    sizes = projects.retopo_sizes(dict(settings))
    if not sizes:
        raise StageError(NO_SIZES)
    return sizes


def to_make(prop: dict[str, Any], settings: Mapping[str, Any]) -> list[float]:
    """The sizes a job makes for the prop: those it has no up-to-date clean mesh of yet, else all of them again (its
    "Make it again")."""
    sizes = sizes_of(settings)
    return projects.retopo_missing_sizes(prop, sizes) or sizes


def vertices(size: float) -> int:
    """The vertices a size stands for (0.25 -> 250): what each tool is asked for, within its own bounds."""
    return round(1000 * size)


def folder(size: float) -> str:
    """The folder of the clean mesh of `size`, relative to the prop's (retopo/4k, retopo/0.25k)."""
    return f"retopo/{projects.retopo_size_name(size)}"


def label(sizes: list[float], index: int) -> str:
    """"22k vertices (2 of 3)": the clean mesh being made, in the progress words ("2k vertices" alone for one)."""
    count = f" ({index + 1} of {len(sizes)})" if len(sizes) > 1 else ""
    return f"{projects.retopo_size_name(sizes[index])} vertices{count}"


def part(ctx: StageContext, sizes: list[float], index: int, share: float) -> StageContext:
    """ctx for making the clean mesh of sizes[index]: what it reports (0..1 of its own work) is its turn of the first
    `share` of the prop's bar, and its words come after its label ("22k vertices (2 of 3) · Measuring the shape…")."""
    def report(fraction: float, message: str) -> None:
        within = min(max(float(fraction), 0.0), 1.0)
        ctx.report(share * (index + within) / len(sizes), f"{label(sizes, index)} · {message}")

    return dataclasses.replace(ctx, report=report)


async def make_each(ctx: StageContext, sizes: list[float], share: float,
                    make: Callable[[float, StageContext], Awaitable[Any]]) -> list[tuple[float, Any]]:
    """make(size, its part of ctx) for each size, one after the other -> [(size, what make returned)]. A size that
    fails is told with the result (ctx.warn) and the others are made all the same; when none is made, the first
    failure is raised. A cancel stops before the next size, and keeps what was made; ToolNeedsRestart (the tool must
    start afresh) stops at once."""
    made: list[tuple[float, Any]] = []
    failed: list[tuple[float, StageError]] = []
    for index, size in enumerate(sizes):
        if index and ctx.cancelled():
            break
        try:
            made.append((size, await make(size, part(ctx, sizes, index, share))))
        except ToolNeedsRestart:
            raise
        except StageError as e:
            if ctx.cancelled() and made:
                break
            failed.append((size, e))
    if not made:
        raise failed[0][1]
    if failed and ctx.warn is not None:
        ctx.warn(" ".join(f"{projects.retopo_size_name(size)} vertices: {e}" for size, e in failed))
    return made
