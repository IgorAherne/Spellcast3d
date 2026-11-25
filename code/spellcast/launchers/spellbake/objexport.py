"""OBJ + MTL export of the baked low poly (design section 6.3), and of a 3D model saved as OBJ
(write_obj_materials, spellbake.convert).

Triangles only, because the baked tangent space assumes exactly these triangles (MikkTSpace
warns that re-triangulating a quad differently breaks it). Positions are welded (``v`` once per
point, so UV seams stay connected in a DCC tool) and every corner carries its ``vt`` and its
``vn``: the export normals the bake used, since normals recomputed on import cost up to 2.8 deg
at p99. The MTL names the maps with paths relative to itself (``../maps/...`` when the OBJ sits
in the bake folder's export/), in the keywords Blender, 3ds Max and most OBJ readers know;
Godot and trimesh read only part of them.

The lines are formatted a block of rows at a time (one str.format per block, no Python loop over
the rows), which writes a million-triangle model in about a second; the text is what np.savetxt
wrote.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

MATERIAL = "projection"
COMMENT = "Spellcast3D projection. Metal/rough PBR maps; the normal map is tangent space (MikkTSpace)."
_BLOCK_ROWS = 65_536


@dataclass(frozen=True)
class MtlMaps:
    """Map files, relative to the MTL; None where there is no such map."""

    basecolor: str | None = None
    normal: str | None = None
    roughness: str | None = None
    metallic: str | None = None
    emissive: str | None = None
    opacity: str | None = None
    emissive_strength: float = 1.0      # Ke: MTL multiplies map_Ke by it (the GLB's emissive strength)


@dataclass(frozen=True)
class MtlMaterial:
    """One ``newmtl`` of write_obj_materials: its maps, and the values that stand in for a missing map."""

    name: str                                       # no spaces (an MTL name is one word)
    maps: MtlMaps
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)     # Kd (linear); a base colour map replaces it
    roughness: float | None = None                  # Pr without a roughness map
    metallic: float | None = None                   # Pm without a metallic map
    emissive: tuple[float, float, float] | None = None      # Ke without an emissive map
    opacity: float | None = None                    # d without an opacity map


def _unique_rows(values: np.ndarray, decimals: int) -> tuple[np.ndarray, np.ndarray]:
    """(distinct rows as written, index of each input row) after rounding to ``decimals``."""
    rounded = np.round(values, decimals) + 0.0     # + 0.0 folds -0.0 into 0.0
    return np.unique(rounded, axis=0, return_inverse=True)


def _write_rows(out: TextIO, rows: np.ndarray, line: str) -> None:
    """``line`` (a str.format pattern ending in a newline) filled with every row of ``rows``."""
    for top in range(0, len(rows), _BLOCK_ROWS):
        block = rows[top:top + _BLOCK_ROWS]
        out.write((line * len(block)).format(*block.reshape(-1).tolist()))


def write_obj(obj_path: str | Path, *, positions: np.ndarray, position_id: np.ndarray, uv: np.ndarray,
              normals: np.ndarray, faces: np.ndarray, maps: MtlMaps) -> Path:
    """Write ``obj_path`` and the MTL beside it (same name, .mtl); returns the MTL's path.

    ``positions``, ``uv`` (v up) and ``normals`` are per render vertex, ``position_id`` welds the
    render vertices that share a point, ``faces`` are triangles of render vertices.
    """
    return write_obj_materials(obj_path, positions=positions, position_id=position_id, uv=uv, normals=normals,
                               faces=faces, materials=[MtlMaterial(MATERIAL, maps)])


def write_obj_materials(obj_path: str | Path, *, positions: np.ndarray, position_id: np.ndarray, uv: np.ndarray,
                        normals: np.ndarray, faces: np.ndarray, materials: Sequence[MtlMaterial],
                        face_materials: np.ndarray | None = None, title: str = "Spellcast3D projection",
                        comment: str = COMMENT) -> Path:
    """write_obj with one or more materials: ``face_materials`` (m,) gives each triangle's index into
    ``materials`` (None: all the first); each run of triangles gets its ``usemtl``. ``title`` heads the
    OBJ, ``comment`` the MTL."""
    obj_path = Path(obj_path)
    mtl_path = obj_path.with_suffix(".mtl")
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

    points, first = np.unique(np.asarray(position_id), return_index=True)
    point_of = np.searchsorted(points, position_id)
    texcoords, texcoord_of = _unique_rows(np.asarray(uv, dtype=np.float64), 7)
    directions, direction_of = _unique_rows(np.asarray(normals, dtype=np.float64), 6)
    texcoord_of, direction_of = texcoord_of.reshape(-1), direction_of.reshape(-1)

    corner = (np.stack([point_of[faces], texcoord_of[faces], direction_of[faces]], axis=-1) + 1).reshape(-1, 9)
    if face_materials is None or len(materials) == 1:
        runs = [(0, 0, len(faces))]
    else:
        order = np.argsort(np.asarray(face_materials), kind="stable")
        corner, owner = corner[order], np.asarray(face_materials)[order]
        edges = np.flatnonzero(np.diff(owner)) + 1
        runs = [(int(owner[start]), int(start), int(end)) for start, end in
                zip(np.r_[0, edges], np.r_[edges, len(owner)])]
    out = io.StringIO()
    out.write(f"# {title}: {len(faces)} triangles\n")
    out.write(f"mtllib {mtl_path.name}\n")
    _write_rows(out, np.asarray(positions, dtype=np.float64)[first], "v {:.9g} {:.9g} {:.9g}\n")
    _write_rows(out, texcoords, "vt {:.7f} {:.7f}\n")
    _write_rows(out, directions, "vn {:.6f} {:.6f} {:.6f}\n")
    for number, (material, start, end) in enumerate(runs):
        out.write(f"usemtl {materials[material].name}\n" + ("s off\n" if number == 0 else ""))
        _write_rows(out, corner[start:end], "f {}/{}/{} {}/{}/{} {}/{}/{}\n")
    obj_path.write_text(out.getvalue(), encoding="utf-8", newline="\n")

    lines = [f"# {comment}"]
    for material in materials:
        lines += _mtl_lines(material)
    mtl_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return mtl_path


def _number(value: float) -> str:
    return f"{value:.6g}"


def _mtl_lines(material: MtlMaterial) -> list[str]:
    maps = material.maps
    lines = [f"newmtl {material.name}", "Kd " + " ".join(map(_number, material.color)), "Ks 0 0 0", "illum 2"]
    if maps.basecolor is not None:
        lines.append(f"map_Kd {maps.basecolor}")
    if maps.normal is not None:
        lines += [f"map_Bump -bm 1 {maps.normal}", f"norm {maps.normal}"]
    if maps.roughness is not None:
        lines.append(f"map_Pr {maps.roughness}")
    elif material.roughness is not None:
        lines.append(f"Pr {_number(material.roughness)}")
    if maps.metallic is not None:
        lines.append(f"map_Pm {maps.metallic}")
    elif material.metallic is not None:
        lines.append(f"Pm {_number(material.metallic)}")
    if maps.emissive is not None:
        strength = f"{maps.emissive_strength:.6g}"
        lines += [f"Ke {strength} {strength} {strength}", f"map_Ke {maps.emissive}"]
    elif material.emissive is not None and any(material.emissive):
        lines.append("Ke " + " ".join(map(_number, material.emissive)))
    if maps.opacity is not None:
        lines.append(f"map_d {maps.opacity}")
    elif material.opacity is not None:
        lines.append(f"d {_number(material.opacity)}")
    return lines
