"""Dev-only check of a Spellcast3D FBX with Blender's own importer (never shipped, never imported by the hub).

    blender --background --factory-startup --python check_fbx.py -- <model.fbx> <source.npz> <report.json>

source.npz holds what the FBX was written from: positions (n, 3) metres in the glTF frame, normals (n, 3),
uvs (n, 2) v-up, indices (m, 3). The script imports the FBX with the default settings, measures what Blender
made of it, re-exports it, and writes report.json for test_fbx.py to judge.
"""

import json
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np

fbx_path, source_path, report_path = sys.argv[sys.argv.index("--") + 1:]
source = np.load(source_path)
corners = source["indices"].reshape(-1)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx_path)
meshes = [ob for ob in bpy.context.scene.objects if ob.type == "MESH"]
ob = meshes[0]
me = ob.data
loops = len(me.loops)


def loop_array(collection, attribute, width):
    values = np.empty(loops * width, np.float64)
    collection.foreach_get(attribute, values)
    return values.reshape(-1, width)


vertex_of_loop = np.empty(loops, np.int64)
me.loops.foreach_get("vertex_index", vertex_of_loop)
local = np.empty(len(me.vertices) * 3, np.float64)
me.vertices.foreach_get("co", local)
local = local.reshape(-1, 3)
matrix = np.array(ob.matrix_world)
world = local @ matrix[:3, :3].T + matrix[:3, 3]
# Blender's glTF importer maps glTF (x, y, z) to Blender (x, -z, y); an FBX in the glTF frame must land the same way.
expected_world = source["positions"][corners][:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])
normals = loop_array(me.corner_normals, "vector", 3)
lengths = np.linalg.norm(source["normals"][corners], axis=1, keepdims=True)
directed = lengths[:, 0] > 1e-6                      # a zero source normal has no direction to compare
cosines = np.clip(np.sum(normals[directed] * source["normals"][corners][directed] / lengths[directed], axis=1),
                  -1.0, 1.0)
sharp = me.attributes.get("sharp_edge")
sharp_count = 0
if sharp is not None:
    flags = np.zeros(len(me.edges), bool)
    sharp.data.foreach_get("value", flags)
    sharp_count = int(flags.sum())


def image_behind(socket):
    """The image feeding a Principled BSDF input (through a Normal Map node for Normal)."""
    if not socket.is_linked:
        return None
    node = socket.links[0].from_node
    if node.type == "NORMAL_MAP":
        return image_behind(node.inputs["Color"])
    if node.type != "TEX_IMAGE" or node.image is None:
        return None
    image = node.image
    return {"name": image.name, "size": list(image.size), "packed": image.packed_file is not None,
            "colorspace": image.colorspace_settings.name}


material = ob.active_material
principled = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
inputs = ("Base Color", "Normal", "Roughness", "Metallic", "Emission Color", "Alpha")

report = {
    "mesh_objects": len(meshes),
    "vertices": len(me.vertices),
    "polygons": len(me.polygons),
    "loops": loops,
    "dimensions": list(ob.dimensions),
    "scale": list(ob.scale),
    "rotation_deg": [float(np.degrees(angle)) for angle in ob.rotation_euler],
    "world_position_error": float(np.abs(world[vertex_of_loop] - expected_world).max()),
    "world_min": world.min(0).tolist(),
    "world_max": world.max(0).tolist(),
    "uv_error": float(np.abs(loop_array(me.uv_layers.active.data, "uv", 2) - source["uvs"][corners]).max()),
    "normal_error_deg": float(np.degrees(np.arccos(cosines)).max()),
    "sharp_edges": sharp_count,
    "material": material.name,
    "images": {name: image_behind(principled.inputs[name]) for name in inputs},
    "values": {name: principled.inputs[name].default_value for name in ("Roughness", "Metallic", "Alpha")},
}
with tempfile.TemporaryDirectory() as folder:
    report["reexport"] = sorted(bpy.ops.export_scene.fbx(filepath=str(Path(folder) / "again.fbx"),
                                                         path_mode="COPY", embed_textures=True))
Path(report_path).write_text(json.dumps(report, indent=1), encoding="utf-8")
