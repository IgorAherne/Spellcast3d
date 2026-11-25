"""Dev-only check of a model Spellcast3D saved as FBX or OBJ, with Blender's own importers (never shipped,
never imported by the hub).

    blender --background --factory-startup --python check_export.py -- <model.fbx|.obj> <source.glb> <report.json>

It imports the source GLB and the saved model into empty scenes and writes report.json for
test_export.py to judge: per model its mesh objects, triangle count, world-space bounding box (so the
size and axes can be compared with the GLB's), UV layers, and per material the images behind the
Principled BSDF inputs.
"""

import json
import sys
from pathlib import Path

import bpy
import numpy as np

model_path, source_path, report_path = sys.argv[sys.argv.index("--") + 1:]


def image_behind(socket):
    """The image feeding a Principled BSDF input (through a Normal Map node for Normal)."""
    if not socket.is_linked:
        return None
    node = socket.links[0].from_node
    if node.type in ("NORMAL_MAP", "BUMP"):
        return image_behind(node.inputs["Color"] if node.type == "NORMAL_MAP" else node.inputs["Height"])
    if node.type == "SEPARATE_COLOR":
        return image_behind(node.inputs["Color"])
    if node.type != "TEX_IMAGE" or node.image is None:
        return None
    image = node.image
    return {"name": image.name, "size": list(image.size), "packed": image.packed_file is not None,
            "colorspace": image.colorspace_settings.name, "has_data": bool(image.has_data or image.size[0])}


def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    suffix = Path(path).suffix.lower()
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        bpy.ops.import_scene.gltf(filepath=path)
    meshes = [ob for ob in bpy.context.scene.objects if ob.type == "MESH"]
    corners, triangles, uv_layers, materials = [], 0, [], {}
    for ob in meshes:
        me = ob.data
        me.calc_loop_triangles()
        triangles += len(me.loop_triangles)
        local = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", local)
        matrix = np.array(ob.matrix_world)
        corners.append(local.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3])
        uv_layers.append(len(me.uv_layers))
        for slot in ob.material_slots:
            material = slot.material
            if material is None or not material.use_nodes:
                continue
            principled = next((n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if principled is None:
                continue
            materials[material.name] = {name: image_behind(principled.inputs[name]) for name in
                                        ("Base Color", "Normal", "Roughness", "Metallic", "Emission Color", "Alpha")}
    points = np.concatenate(corners) if corners else np.zeros((1, 3))
    return {"mesh_objects": len(meshes), "triangles": triangles, "uv_layers": uv_layers,
            "min": points.min(0).tolist(), "max": points.max(0).tolist(), "materials": materials}


report = {"source": load(source_path), "model": load(model_path)}
Path(report_path).write_text(json.dumps(report, indent=1), encoding="utf-8")
