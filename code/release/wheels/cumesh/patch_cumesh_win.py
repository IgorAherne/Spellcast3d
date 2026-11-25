"""Spellcast3D's edits to CuMesh (JeffreyXiang/CuMesh, Microsoft TRELLIS.2's mesh library), as the user's fork carries
them, and a check that the fork's tree is exactly upstream plus these edits.

    python patch_cumesh_win.py check [--src <CuMesh clone>] [--rev <fork commit>] [--cubvh-src <cubvh clone>]
    python patch_cumesh_win.py apply <folder> [--only <id> ...]      (developers: the edits onto a base export)

Since cumesh 0.0.1.post4 the wheel is built from the fork's tree as committed (build_cumesh_win.py exports PINS["fork"]
with `git archive`; nothing is patched at build time). The fork, https://github.com/IgorAherne/CuMesh branch
spellcast3d, is upstream main (PINS["upstream"], merged into the fork's main) with cubvh vendored as plain files in
third_party/cubvh (PINS["cubvh"], the commit upstream's submodule pins; upstream's submodule is not ours to change,
and three of our edits are cubvh's), eigen as a submodule of the fork at third_party/cubvh/third_party/eigen
(PINS["eigen"], cubvh's own submodule, unchanged), and the edits below as commits. `check` rebuilds that tree from
the pins - upstream's tree, cubvh's tree in third_party/cubvh, every HUNK applied - and holds it to the fork's
commit file by file, byte for byte (README.md aside: the fork's build notes), plus the one submodule link (eigen at
its pin). So this file names every change the wheel's sources carry, and the fork cannot drift from it unnoticed.
The hunk ids are those of the wheels 0.0.1.post1-post3 (built from b84410e + cubvh ff89002 + these edits); hunks
upstream made unnecessary are listed in DROPPED.

Eight kinds of edit:

1. Building on Windows with MSVC 2022 + CUDA 12.8: upstream's own Windows build (setup.py since f149d59: /O2
   /std:c++17 /EHsc /permissive- /Zc:__cplusplus, -allow-unsupported-compiler, --extended-lambda; atlas.cu's
   `reduce_op` outside the macro; cubvh's `py::ssize_t` since a3d8d25) builds as it is, so the only edit left is
   pyproject's `torch>=2.4.0` runtime pin, which would let pip replace the cu128 torch with PyPI's (the installer
   installs torch itself), and the wheel's version: 0.0.1.post4+cu128torch2.8 (a post-release: pip installs a wheel
   over an installed one only when its version is greater, and Pixal3D's install.py installs the wheels of its whl
   folder on every run).

2. cuBVH's build lets go of Python's lock. `create_cuBVH` builds the tree on the CPU for seconds (5-7 s for a
   9 M-face mesh); it reads only the Eigen matrices pybind converted before the call and touches no Python object,
   so `py::call_guard<py::gil_scoped_release>()` is safe, and Spellcast3D's 3D stage builds one prop's BVH on a
   worker thread while the GPU thread decodes the next prop (code/spellcast/launchers/pixal3d_glb.py).

3. Pixal3D's memory edits of cumesh/remeshing.py (the wheel TencentARC's Pixal3D shipped had them): the remesh's BVH
   queries in chunks of two million points, and intermediates freed as soon as they are read. Unsigned distances are
   the same point by point whatever the chunk, so the values are upstream's. (The fork's main also expanded the
   sparse grid 500,000 parents at a time, b9ab99b; measured on the golden meshes it gives the same remesh, no lower
   peak - the dual contouring after the grid is 4-5x wider - and 0-5 % more time: not taken. Spellcast3D's
   code/logs/profiling/pixal3d-lowmem/cumesh-fork/.)

4. cuBVH's card copy on demand. Upstream, the constructor builds the tree on the CPU and uploads the nodes and the
   triangles, and the card copy lives as long as the object. Spellcast3D builds the tree on a worker thread while
   the GPU thread decodes the next prop (no card memory may be taken there), uploads it in a later GPU step, frees
   the card copy at the end of that step and uploads it again for the texture bake instead of building it again.
   So `create_cuBVH(vertices, triangles, upload=True)` and `cumesh.cuBVH(..., upload=True)` take a flag (False: no
   CUDA call at all), and the object gets `upload()`, `release_device()` and `on_device`; the queries upload first
   when the copy is not there. The tree's construction is the same, step for step (only the upload at the end of
   TriangleBvh::build moved out), and the host vectors are kept, so every query answers as upstream's.

5. fill_holes' chain in less card memory (0.0.1.post2). Pixal3D closes the small holes of every decoded mesh with
   CuMesh's chain init, get_edges, get_boundary_info, get_vertex_edge_adjacency, get_vertex_boundary_adjacency,
   get_manifold_boundary_adjacency, get_boundary_connected_components, get_boundary_loops, fill_holes, read; at 1536
   that is the widest moment of a prop's finish. Upstream's peak: 141-144 bytes a face of native tables; with these
   edits (and Pixal3D's Mesh.fill_holes without its clones) 66-67, get_edges' sort of the 3F edge keys. The tables
   are the same, bit for bit; only how long they live and how large they are allocated changes:
   - get_edges sorts the 3F edge keys in a cub DoubleBuffer (two buffers, where the out-of-place sort takes a third
     inside its scratch), frees the half that does not hold the result, counts the unique edges before it encodes
     them so that `edges` and `edge2face_cnt` are E long instead of 3F, and frees the sorted keys and cub's scratch
     before it returns. Sorting keys alone has one result whatever the method; the encoding reads the same keys.
   - get_boundary_info and get_manifold_boundary_adjacency select from a counting iterator instead of an arange
     table, and count first so that their outputs are as long as what they select (the selection keeps the input
     order; the values are those of the arange).
   - fill_holes frees the connectivity tables it no longer reads (all but the edges) before it extends the vertex
     and face buffers (upstream frees them with clear_cache() right after), so the extensions do not stack on them.
   Rebased onto upstream's stream handling (4843631): every launch, CUB call and copy on the current stream as
   upstream's own code around them, and the stream synchronised before a buffer that queued work reads is freed.
   Left as upstream has it: the z of the hole centres fill_holes adds is not a function of the input (its segmented
   sum most likely starts from an unset value); Spellcast3D's launcher recomputes every centre (recentre_fans).

6. The ray-stab inside test with an any-hit traversal (0.0.1.post3). cuBVH's signed_distance(mode="raystab") calls a
   point inside when all 64 stab rays hit the mesh, and asks each ray only "hit or no hit"; upstream answers that
   with ray_intersect, the closest-hit traversal. ray_intersect_any returns at the first hit, with ray_intersect's
   own tests (a triangle counts when Triangle::ray_intersect gives t < MAX_DIST; a child is left out when its box's
   entry distance is not below MAX_DIST; a stack overflow falls back to the escape-link traversal with the same
   tests), so the verdicts and the signed distances are the same bit for bit, in whatever order the boxes are
   visited; children whose box lies wholly behind the ray's origin are visited last, for speed (the stab in 28-48 %
   of its card time on the golden meshes). ray_intersect itself, ray_trace and the other queries are untouched.

7. cuBVH's tree built on several threads (0.0.1.post5): the same nodes and triangle order as the one-thread build.

8. The ray-stab's inside test for any subset of a call's points (0.0.1.post6). `cuBVH.raystab_inside(positions,
   rng_index, first_ray, last_ray, closest)` answers signed_distance(mode="raystab")[0] < 0 per point without the
   distances: each point's stab directions are seeded from rng_index, its index in the signed_distance call it
   stands for (the raystab kernel seeds them from the index in its call, so a subset would otherwise get other
   rays), a range of the 32 directions can be asked at a time (each ray's hit or miss depends on the point, the
   direction and the tree alone, so the points whose rays all hit in one range can be compacted and asked the
   next), and the nearest-triangle search, which only tells a point on the surface (-0 is not inside), runs only
   for the points whose rays all hit. A kernel of its own beside signed_distance's, which is untouched. And
   `cuBVH.ray_hits(rays_o, rays_d)`: whether each ray hits the mesh at all, ray_trace(...)[1] >= 0 bit for bit, by the
   any-hit traversal of 6 (ray_trace, the closest-hit query, is untouched).

Measurements and checks: code/logs/profiling/pixal3d-lowmem/round-g/cumesh/REPORT.md (5), round-h/anyhit/REPORT.md
(6), cumesh-fork/ (the move to the fork, 0.0.1.post4), round-n/geometry/ (8) in Spellcast3D.
"""
from __future__ import annotations

import argparse
import filecmp
import io
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

# The sources: (repository, commit). "fork" is what the wheel is built from; the others are what `check` rebuilds it
# from. upstream = "Merge pull request #29 from cuzelac/fix/blackwell-stream-sync-upstream" (2026-03), the head of
# JeffreyXiang/CuMesh main that the fork's branch merges; cubvh = "wrap cubvh API with cumesh namespace to avoid
# symbol conflicts", what that commit's third_party/cubvh points at; eigen = cubvh's submodule (headers only, MPL-2.0).
PINS = {
    "fork": ("https://github.com/IgorAherne/CuMesh.git", "6e9875f3c5a49f68ced42a612609465d8f655de2"),
    "upstream": ("https://github.com/JeffreyXiang/CuMesh.git", "12289e1062f0603f2f0d0771b02e1395d247f26f"),
    "cubvh": ("https://github.com/JeffreyXiang/cubvh.git", "ce92267a24ef6ad7d2c8ccbc2ae2c021a6597e70"),
    "eigen": ("https://gitlab.com/libeigen/eigen", "e63d9f6ccb7f6f29f31241b87c542f3f0ab3112b"),
}
FORK_BRANCH = "spellcast3d"
WHEEL_VERSION = "0.0.1.post6+cu128torch2.8"
CUBVH_DIR = "third_party/cubvh"
EIGEN_DIR = "third_party/cubvh/third_party/eigen"
DOCS = ("README.md",)           # the fork's own build notes: not checked

# Changes that are not text edits of upstream's files (documentation; `check` builds the base this way).
STRUCTURE = [
    {"id": "cubvh-vendored", "file": f"{CUBVH_DIR}/ (cubvh {PINS['cubvh'][1][:7]}'s tree as plain files)",
     "commit": "cubvh: vendored at ce92267 instead of a submodule"},
    {"id": "eigen-submodule", "file": f"{EIGEN_DIR} (submodule link at eigen {PINS['eigen'][1][:9]})",
     "commit": "cubvh: vendored at ce92267 instead of a submodule"},
]

# Hunks of 0.0.1.post1-post3 that the new base made unnecessary (kept for the record).
DROPPED = {
    "setup-msvc-flags": "upstream's setup.py has its own Windows flags since f149d59 (/O2 /std:c++17 and more)",
    "setup-ext-C-flags": "as setup-msvc-flags",
    "setup-ext-cubvh-flags": "as setup-msvc-flags; the cubvh include of src/ was for the ssize_t header only",
    "setup-ext-xatlas-flags": "as setup-msvc-flags",
    "ext-ssize_t": "CuMesh's own sources never use ssize_t; the need was cubvh's api_cpu.h, fixed in cubvh a3d8d25",
    "cubvh-bindings-ssize_t": "cubvh a3d8d25 uses py::ssize_t",
    "msvc-compat-header": "as ext-ssize_t",
    "atlas-cub-sum-for-cuda-12.8": "upstream f149d59/063a714 moved the #if out of the macro's arguments",
    "connectivity-free-manifold-vert-idx": "upstream 4843631 frees the list (after a stream sync)",
    "remeshing-memory-edits": "the whole-file copy of remeshing_pixal3d.py is now the three remeshing-* hunks",
}

HUNKS: list[dict] = [
    # 1. The Windows wheel.
    {"id": "pyproject-no-torch-pin", "file": "pyproject.toml", "commit": "the fork's a04452a (kept as it is)",
     "old": '    "torch>=2.4.0",\n',
     "new": ('    # we are manually installing torch, so comment this one out, else it tries to overwrite with another version\n'
             '    #"torch>=2.4.0",\n')},
    {"id": "local-version", "file": "pyproject.toml", "commit": "cubvh: the ray-stab's inside test for any subset of a call's points (post6; before it: the tree built on several threads, post5)",
     "old": 'version = "0.0.1"\n',
     "new": f'version = "{WHEEL_VERSION}"\n'},
    {"id": "gitignore-vs", "file": ".gitignore", "commit": "the fork's b9ab99b (kept as it is)",
     "old": "__marimo__/\n",
     "new": "__marimo__/\n/.vs\n"},
    {"id": "gitmodules-eigen", "file": ".gitmodules", "commit": "cubvh: vendored at ce92267 instead of a submodule",
     "old": ('[submodule "third_party/cubvh"]\n'
             '\tpath = third_party/cubvh\n'
             '\turl = https://github.com/JeffreyXiang/cubvh.git\n'
             '\tbranch = trellis.2\n'),
     "new": ('[submodule "third_party/cubvh/third_party/eigen"]\n'
             '\tpath = third_party/cubvh/third_party/eigen\n'
             '\turl = https://gitlab.com/libeigen/eigen\n')},

    # 2. cuBVH's build without the GIL.
    {"id": "cubvh-build-without-the-gil", "file": "third_party/cubvh/src/bindings.cpp",
     "commit": "cubvh: build the tree without Python's lock",
     "old": 'm.def("create_cuBVH", &create_cuBVH);\n',
     "new": ('// The tree is built on the CPU for seconds and touches no Python object (its arguments are Eigen matrices\n'
             '// pybind converted before the call): without the GIL, other Python threads run meanwhile (Spellcast3D\n'
             '// builds a prop\'s BVH on a worker thread while its GPU thread decodes the next prop).\n'
             'm.def("create_cuBVH", &create_cuBVH, py::call_guard<py::gil_scoped_release>());\n')},

    # 4. cuBVH's card copy on demand. The tree's construction is not touched: build() keeps every line but the upload
    # at its end, and the constructor keeps its loop and its build call.
    {"id": "cubvh-tree-upload-release", "file": "third_party/cubvh/include/gpu/bvh.cuh",
     "commit": "cubvh: the card copy of the tree on demand",
     "old": ('    TriangleBvhNode* nodes_gpu() const {\n'
             '        return m_nodes_gpu.data();\n'
             '    }\n'),
     "new": ('    TriangleBvhNode* nodes_gpu() const {\n'
             '        return m_nodes_gpu.data();\n'
             '    }\n'
             '\n'
             '    // Spellcast3D: build() fills the host vector m_nodes only. upload() copies it to the card and\n'
             '    // release_device() frees the card copy but keeps m_nodes, so the tree can go back to the card later\n'
             '    // without being built again.\n'
             '    virtual void upload() {\n'
             '        m_nodes_gpu.resize_and_copy_from_host(m_nodes);\n'
             '    }\n'
             '\n'
             '    // resize(0) rather than free_memory(): free_memory() keeps the element count, and a later resize to the\n'
             '    // same count would then skip the allocation and copy into a null pointer.\n'
             '    virtual void release_device() {\n'
             '        m_nodes_gpu.resize(0);\n'
             '    }\n')},
    {"id": "cubvh-tree-build-without-upload", "file": "third_party/cubvh/src/bvh.cu",
     "commit": "cubvh: the card copy of the tree on demand",
     "old": ('        if (!m_nodes.empty()) {\n'
             '            thread_bvh(0, -1);\n'
             '        }\n'
             '\n'
             '        m_nodes_gpu.resize_and_copy_from_host(m_nodes);\n'),
     "new": ('        if (!m_nodes.empty()) {\n'
             '            thread_bvh(0, -1);\n'
             '        }\n'
             '\n'
             '        // Spellcast3D: no upload here, so that the build makes no CUDA call and a worker thread can run it\n'
             '        // while the GPU is busy; the owner calls upload() when it wants the nodes on the card.\n')},
    {"id": "cubvh-api-upload-release", "file": "third_party/cubvh/include/gpu/api_gpu.h",
     "commit": "cubvh: the card copy of the tree on demand",
     "old": ('    virtual void signed_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw, uint32_t mode) = 0;\n'
             '};\n'
             '\n'
             '// function to create an implementation of cuBVH\n'
             'cuBVH* create_cuBVH(Ref<const Verts> vertices, Ref<const Trigs> triangles);\n'),
     "new": ('    virtual void signed_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw, uint32_t mode) = 0;\n'
             '\n'
             '    // Spellcast3D: the card copy of the tree (nodes and triangles) on demand. upload() makes it when it is\n'
             '    // not there; release_device() frees it and keeps the host copy, so a later upload needs no rebuild; the\n'
             '    // three queries above upload first when the copy is not there.\n'
             '    virtual void upload() = 0;\n'
             '    virtual void release_device() = 0;\n'
             '    virtual bool is_on_device() = 0;\n'
             '};\n'
             '\n'
             '// function to create an implementation of cuBVH. Spellcast3D: with upload false the tree is built on the\n'
             '// CPU and no CUDA call is made at all, so a worker thread can build it while the GPU thread works.\n'
             'cuBVH* create_cuBVH(Ref<const Verts> vertices, Ref<const Trigs> triangles, bool upload = true);\n')},
    {"id": "cubvh-impl-mutex-include", "file": "third_party/cubvh/src/api_gpu.cu",
     "commit": "cubvh: the card copy of the tree on demand",
     "old": '#include <Eigen/Dense>\n\nusing namespace Eigen;\n',
     "new": '#include <Eigen/Dense>\n\n#include <mutex>\n\nusing namespace Eigen;\n'},
    {"id": "cubvh-impl-upload-flag", "file": "third_party/cubvh/src/api_gpu.cu",
     "commit": "cubvh: the card copy of the tree on demand",
     "old": '    cuBVHImpl(Ref<const Verts> vertices, Ref<const Trigs> triangles) : cuBVH() {\n',
     "new": '    cuBVHImpl(Ref<const Verts> vertices, Ref<const Trigs> triangles, bool upload_now) : cuBVH() {\n'},
    {"id": "cubvh-impl-build-without-upload", "file": "third_party/cubvh/src/api_gpu.cu",
     "commit": "cubvh: the card copy of the tree on demand",
     "old": '        triangles_gpu.resize_and_copy_from_host(triangles_cpu);\n',
     "new": ('        // Spellcast3D: the build above runs on the CPU only (the nodes and the reordered triangles stay in\n'
             '        // host vectors); the card copy is made here when asked for, otherwise by upload() or the first query.\n'
             '        if (upload_now) {\n'
             '            ensure_on_device();\n'
             '        }\n')},
    {"id": "cubvh-impl-upload-release", "file": "third_party/cubvh/src/api_gpu.cu",
     "commit": "cubvh: the card copy of the tree on demand",
     "old": '    void ray_trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor positions, at::Tensor face_id, at::Tensor depth) {\n',
     "new": ('    // Spellcast3D: the card copy is made from the host vectors, which stay as they are, so it can be dropped\n'
             '    // and made again without building the tree again. The copies are plain cudaMalloc + cudaMemcpy on the\n'
             '    // calling thread\'s current device (CUDA\'s primary context), with no state kept per thread, so a thread\n'
             '    // other than the one that built the tree may upload it, query it and release it.\n'
             '    void upload() override {\n'
             '        std::lock_guard<std::mutex> lock(device_mutex);\n'
             '        ensure_on_device();\n'
             '    }\n'
             '\n'
             '    // cudaFree waits for the queued kernels that may still read the copy, as the destructor always did.\n'
             '    void release_device() override {\n'
             '        std::lock_guard<std::mutex> lock(device_mutex);\n'
             '        release_device_locked();\n'
             '    }\n'
             '\n'
             '    bool is_on_device() override {\n'
             '        std::lock_guard<std::mutex> lock(device_mutex);\n'
             '        return on_device;\n'
             '    }\n'
             '\n'
             '    // Callers hold device_mutex (or own the object alone, as the constructor does). The nodes go first and\n'
             '    // the triangles second, the order of the upstream constructor. The sync at the end waits for the copies\n'
             '    // to land: a copy from pageable memory may return before its DMA has finished, and a query on a stream\n'
             '    // that does not wait for the default stream would then read a half-written tree. A failed upload (out\n'
             '    // of card memory, typically) gives back what it had taken before it rethrows.\n'
             '    void ensure_on_device() {\n'
             '        if (on_device) {\n'
             '            return;\n'
             '        }\n'
             '        try {\n'
             '            triangle_bvh->upload();\n'
             '            triangles_gpu.resize_and_copy_from_host(triangles_cpu);\n'
             '            CUDA_CHECK_THROW(cudaStreamSynchronize(0));\n'
             '        } catch (...) {\n'
             '            try {\n'
             '                release_device_locked();\n'
             '            } catch (...) {\n'
             '            }\n'
             '            throw;\n'
             '        }\n'
             '        on_device = true;\n'
             '    }\n'
             '\n'
             '    // resize(0) rather than free_memory(), for the reason given in TriangleBvh::release_device().\n'
             '    void release_device_locked() {\n'
             '        on_device = false;\n'
             '        triangle_bvh->release_device();\n'
             '        triangles_gpu.resize(0);\n'
             '    }\n'
             '\n'
             '    void ray_trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor positions, at::Tensor face_id, at::Tensor depth) {\n')},
    {"id": "cubvh-impl-lazy-upload", "file": "third_party/cubvh/src/api_gpu.cu", "count": 3,
     "commit": "cubvh: the card copy of the tree on demand",
     "old": ('        cudaStream_t stream = at::cuda::getCurrentCUDAStream();\n'
             '\n'
             '        triangle_bvh->'),
     "new": ('        cudaStream_t stream = at::cuda::getCurrentCUDAStream();\n'
             '\n'
             '        std::lock_guard<std::mutex> lock(device_mutex);\n'
             '        ensure_on_device();\n'
             '        triangle_bvh->')},
    {"id": "cubvh-impl-state-and-factory", "file": "third_party/cubvh/src/api_gpu.cu",
     "commit": "cubvh: the card copy of the tree on demand",
     "old": ('    std::vector<Triangle> triangles_cpu;\n'
             '    GPUMemory<Triangle> triangles_gpu;\n'
             '    std::shared_ptr<TriangleBvh> triangle_bvh;\n'
             '};\n'
             '    \n'
             'cuBVH* create_cuBVH(Ref<const Verts> vertices, Ref<const Trigs> triangles) {\n'
             '    return new cuBVHImpl{vertices, triangles};\n'
             '}\n'),
     "new": ('    std::vector<Triangle> triangles_cpu;\n'
             '    GPUMemory<Triangle> triangles_gpu;\n'
             '    std::shared_ptr<TriangleBvh> triangle_bvh;\n'
             '    // Spellcast3D: whether the card holds the tree now, and the lock that keeps upload(), release_device()\n'
             '    // and the queries from interleaving when Python threads call them on one object (upload and release\n'
             '    // run without the GIL). Nothing done under the lock needs the GIL, so the two locks cannot deadlock.\n'
             '    bool on_device = false;\n'
             '    std::mutex device_mutex;\n'
             '};\n'
             '    \n'
             'cuBVH* create_cuBVH(Ref<const Verts> vertices, Ref<const Trigs> triangles, bool upload) {\n'
             '    return new cuBVHImpl{vertices, triangles, upload};\n'
             '}\n')},
    {"id": "cubvh-bindings-upload-release", "file": "third_party/cubvh/src/bindings.cpp",
     "commit": "cubvh: the card copy of the tree on demand",
     "old": '    .def("signed_distance", &cuBVH::signed_distance);\n',
     "new": ('    .def("signed_distance", &cuBVH::signed_distance)\n'
             '    // Spellcast3D: the card copy on demand. upload and release_device touch no Python object; without the\n'
             '    // GIL other threads run while the copy is made or while cudaFree waits for queued kernels.\n'
             '    .def("upload", &cuBVH::upload, py::call_guard<py::gil_scoped_release>())\n'
             '    .def("release_device", &cuBVH::release_device, py::call_guard<py::gil_scoped_release>())\n'
             '    .def_property_readonly("on_device", &cuBVH::is_on_device);\n')},
    # Builds on the text cubvh-build-without-the-gil wrote (hunks are applied in this order).
    {"id": "cubvh-bindings-create-upload-arg", "file": "third_party/cubvh/src/bindings.cpp",
     "commit": "cubvh: the card copy of the tree on demand",
     "old": 'm.def("create_cuBVH", &create_cuBVH, py::call_guard<py::gil_scoped_release>());\n',
     "new": ('// upload=False builds the tree on the CPU only and leaves the card alone (upload() or the first query copies it).\n'
             'm.def("create_cuBVH", &create_cuBVH, py::arg("vertices"), py::arg("triangles"), py::arg("upload") = true,\n'
             '      py::call_guard<py::gil_scoped_release>());\n')},
    {"id": "bvh-py-upload-arg", "file": "cumesh/bvh.py", "commit": "cubvh: the card copy of the tree on demand",
     "old": ('    def __init__(self, vertices, triangles):\n'
             '        # vertices: np.ndarray, [N, 3]\n'
             '        # triangles: np.ndarray, [M, 3]\n'),
     "new": ('    def __init__(self, vertices, triangles, upload=True):\n'
             '        # vertices: np.ndarray, [N, 3]\n'
             '        # triangles: np.ndarray, [M, 3]\n'
             '        # upload: copy the tree to the card now, as upstream does. Spellcast3D passes False to build it on\n'
             '        # the CPU only, on a worker thread, without taking card memory; upload() or the first query copies it.\n')},
    {"id": "bvh-py-upload-release", "file": "cumesh/bvh.py", "commit": "cubvh: the card copy of the tree on demand",
     "old": '        self.impl = _backend.create_cuBVH(vertices, triangles)\n',
     "new": ('        self.impl = _backend.create_cuBVH(vertices, triangles, bool(upload))\n'
             '\n'
             '    def upload(self):\n'
             '        # Copies the tree (nodes and triangles) to the card when it is not there, without Python\'s lock so\n'
             '        # other threads run meanwhile. The queries do it themselves when needed (holding the lock).\n'
             '        self.impl.upload()\n'
             '\n'
             '    def release_device(self):\n'
             '        # Frees the card copy and keeps the host copy, so a later upload() or query needs no rebuild.\n'
             '        self.impl.release_device()\n'
             '\n'
             '    @property\n'
             '    def on_device(self):\n'
             '        return self.impl.on_device\n')},

    # 6. The ray-stab's inside test with an any-hit traversal: two functions beside ray_intersect, which stays as it is
    # for ray_trace, and the stab loop asking them instead.
    {"id": "cubvh-ray-intersect-any", "file": "third_party/cubvh/src/bvh.cu",
     "commit": "cubvh: the ray-stab's inside test with an any-hit traversal",
     "old": ('    __host__ __device__ static std::pair<int, float> signed_distance_raystab(const Vector3f& point, '
             'const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, '
             'float max_distance_sq = MAX_DIST_SQ, pcg32 rng={}) {\n'),
     "new": ('    // Spellcast3D: whether a ray hits the mesh at all - all that signed_distance_raystab asks of a stab ray,\n'
             '    // where ray_intersect (the closest-hit traversal above) goes on searching for the nearest hit after its\n'
             '    // first one. The answer is ray_intersect(ro, rd, ...).first >= 0 for every ray, by construction:\n'
             '    //  - a triangle counts when Triangle::ray_intersect gives t < MAX_DIST, which is how ray_intersect takes\n'
             '    //    its first hit (t < mint, and mint is MAX_DIST until then): the same function of the same arguments\n'
             '    //    and the same comparison;\n'
             "    //  - a child is left out when its box's entry distance (BoundingBox::ray_intersect(...).x()) is not below\n"
             '    //    MAX_DIST, which is what ray_intersect leaves out until its first hit (it pushes a child when that\n'
             '    //    distance is below mint); only after a first hit, when its answer is settled, does it leave out more;\n'
             '    //  - so both answer "some leaf whose box has an entry distance below MAX_DIST holds a triangle with t below\n'
             '    //    MAX_DIST", whatever the order of the visit: a box holds its descendants\' boxes (min and max over the\n'
             '    //    same vertex floats) and the slab test is monotone in the box (IEEE subtraction and division, no\n'
             '    //    contraction possible), so every ancestor of such a leaf, the root included, has an entry distance no\n'
             "    //    larger than the leaf's. The same holds for the escape-link traversals that both fall back to after a\n"
             '    //    stack overflow (they test the root too, the stack traversals do not).\n'
             '    // Only the order of the visit changes, for speed: children whose box lies wholly behind the ray\'s origin\n'
             '    // (exit distance below 0) come after their siblings, which come by entry distance as in ray_intersect, so\n'
             '    // that a ray from inside the mesh meets the surface ahead of it before it walks the boxes behind it.\n'
             '    __host__ __device__ static bool ray_intersect_any_stackless(Ref<const Vector3f> ro, Ref<const Vector3f> rd, '
             'const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles) {\n'
             '        int idx = 0;\n'
             '        while (idx != -1) {\n'
             '            const TriangleBvhNode& node = bvhnodes[idx];\n'
             '\n'
             '            float tbb = node.bb.ray_intersect(ro, rd).x();\n'
             '            if (tbb >= MAX_DIST) {\n'
             '                idx = node.escape_idx;\n'
             '                continue;\n'
             '            }\n'
             '\n'
             '            if (node.left_idx < 0) {\n'
             '                int end = -node.right_idx-1;\n'
             '                for (int i = -node.left_idx-1; i < end; ++i) {\n'
             '                    float t = triangles[i].ray_intersect(ro, rd);\n'
             '                    if (t < MAX_DIST) {\n'
             '                        return true;\n'
             '                    }\n'
             '                }\n'
             '                idx = node.escape_idx;\n'
             '            } else {\n'
             '                idx = node.left_idx;\n'
             '            }\n'
             '        }\n'
             '\n'
             '        return false;\n'
             '    }\n'
             '\n'
             '    __host__ __device__ static bool ray_intersect_any(Ref<const Vector3f> ro, Ref<const Vector3f> rd, '
             'const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles) {\n'
             '        FixedIntStack query_stack;\n'
             '        query_stack.push(0);\n'
             '\n'
             '        while (!query_stack.empty()) {\n'
             '            int idx = query_stack.pop();\n'
             '\n'
             '            const TriangleBvhNode& node = bvhnodes[idx];\n'
             '\n'
             '            if (node.left_idx < 0) {\n'
             '                int end = -node.right_idx-1;\n'
             '                for (int i = -node.left_idx-1; i < end; ++i) {\n'
             '                    float t = triangles[i].ray_intersect(ro, rd);\n'
             '                    if (t < MAX_DIST) {\n'
             '                        return true;\n'
             '                    }\n'
             '                }\n'
             '            } else {\n'
             '                DistAndIdx children[BRANCHING_FACTOR];\n'
             '\n'
             '                uint32_t first_child = node.left_idx;\n'
             '\n'
             '                #pragma unroll\n'
             '                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {\n'
             '                    Vector2f span = bvhnodes[i+first_child].bb.ray_intersect(ro, rd);\n'
             '                    // The sort key: the entry distance; MAX_DIST (after every other child) for a box wholly\n'
             '                    // behind the origin; infinity (never pushed) for a box ray_intersect leaves out before\n'
             '                    // its first hit. So a child is pushed exactly when its entry distance is below MAX_DIST.\n'
             '                    float key = span.x();\n'
             '                    if (!(span.x() < MAX_DIST)) {\n'
             '                        key = std::numeric_limits<float>::infinity();\n'
             '                    } else if (span.y() < 0.0f) {\n'
             '                        key = MAX_DIST;\n'
             '                    }\n'
             '                    children[i] = {key, i+first_child};\n'
             '                }\n'
             '\n'
             '                sorting_network<BRANCHING_FACTOR>(children);\n'
             '\n'
             '                #pragma unroll\n'
             '                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {\n'
             '                    if (children[i].dist <= MAX_DIST) {\n'
             '                        query_stack.push(children[i].idx);\n'
             '                    }\n'
             '                }\n'
             '            }\n'
             '\n'
             '            if (query_stack.overflowed()) {\n'
             '                // Fallback to stackless traversal to guarantee correctness\n'
             '                return ray_intersect_any_stackless(ro, rd, bvhnodes, triangles);\n'
             '            }\n'
             '        }\n'
             '\n'
             '        return false;\n'
             '    }\n'
             '\n'
             '    __host__ __device__ static std::pair<int, float> signed_distance_raystab(const Vector3f& point, '
             'const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, '
             'float max_distance_sq = MAX_DIST_SQ, pcg32 rng={}) {\n')},
    {"id": "cubvh-raystab-any-hit", "file": "third_party/cubvh/src/bvh.cu",
     "commit": "cubvh: the ray-stab's inside test with an any-hit traversal",
     "old": ('            // If any of the stab rays goes outside the mesh, the SDF is positive.\n'
             '            if (ray_intersect(point, -d, bvhnodes, triangles).first < 0 || '
             'ray_intersect(point, d, bvhnodes, triangles).first < 0) {\n'),
     "new": ('            // If any of the stab rays goes outside the mesh, the SDF is positive.\n'
             '            // Spellcast3D: only hit or no hit counts here, so ray_intersect_any, which stops at the first hit\n'
             '            // (the same answer as ray_intersect(...).first >= 0, see above).\n'
             '            if (!ray_intersect_any(point, -d, bvhnodes, triangles) || '
             '!ray_intersect_any(point, d, bvhnodes, triangles)) {\n')},

    # 5. fill_holes' chain in less card memory (see the module docstring, point 5). The values are upstream's, bit
    # for bit; only the lifetimes and sizes of the tables change. On upstream's stream handling (4843631).
    {"id": "connectivity-get-edges-lean", "file": "src/connectivity.cu",
     "commit": "fill_holes' chain in less card memory",
     "old": ('void CuMesh::get_edges() {\n'
             '    cudaStream_t stream = current_stream();\n'
             '    size_t F = this->faces.size;\n'
             '    this->edges.resize(F * 3);\n'
             '    expand_edges_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(this->faces.ptr, F, this->edges.ptr);\n'
             '    CUDA_CHECK(cudaGetLastError());\n'
             '\n'
             '    // sort edges\n'
             '    this->temp_storage.resize(F * 3 * sizeof(uint64_t));\n'
             '    size_t temp_storage_bytes = 0;\n'
             '    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(\n'
             '        nullptr, temp_storage_bytes,\n'
             '        this->edges.ptr,\n'
             '        reinterpret_cast<uint64_t*>(this->temp_storage.ptr),\n'
             '        F * 3, 0, sizeof(uint64_t) * 8, stream\n'
             '    ));\n'
             '    this->cub_temp_storage.resize(temp_storage_bytes);\n'
             '    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(\n'
             '        this->cub_temp_storage.ptr, temp_storage_bytes,\n'
             '        this->edges.ptr,\n'
             '        reinterpret_cast<uint64_t*>(this->temp_storage.ptr),\n'
             '        F * 3, 0, sizeof(uint64_t) * 8, stream\n'
             '    ));\n'
             '\n'
             '    // unique edges\n'
             '    int* num_edges;\n'
             '    CUDA_CHECK(cudaMalloc(&num_edges, sizeof(int)));\n'
             '    this->edge2face_cnt.resize(F * 3);\n'
             '    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(\n'
             '        nullptr, temp_storage_bytes,\n'
             '        reinterpret_cast<uint64_t*>(this->temp_storage.ptr), this->edges.ptr, this->edge2face_cnt.ptr, num_edges,\n'
             '        F * 3, stream\n'
             '    ));\n'
             '    this->cub_temp_storage.resize(temp_storage_bytes);\n'
             '    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(\n'
             '        this->cub_temp_storage.ptr, temp_storage_bytes,\n'
             '        reinterpret_cast<uint64_t*>(this->temp_storage.ptr), this->edges.ptr, this->edge2face_cnt.ptr, num_edges,\n'
             '        F * 3, stream\n'
             '    ));\n'
             '    CUDA_CHECK(cudaMemcpyAsync(&this->edges.size, num_edges, sizeof(int), cudaMemcpyDeviceToHost, stream));\n'
             '    CUDA_CHECK(cudaStreamSynchronize(stream));\n'
             '    this->edge2face_cnt.size = this->edges.size;\n'
             '    CUDA_CHECK(cudaFree(num_edges));\n'
             '}\n'),
     "new": ('void CuMesh::get_edges() {\n'
             '    cudaStream_t stream = current_stream();\n'
             '    size_t F = this->faces.size;\n'
             '\n'
             '    // Spellcast3D: the same edges and counts as upstream, bit for bit, in far less card memory. Upstream\n'
             "    // expands the 3F edge keys into `edges`, sorts them into `temp_storage` with cub's out-of-place sort\n"
             '    // (which takes a third buffer of 3F keys inside cub_temp_storage), encodes the runs back into `edges` and\n'
             '    // their lengths into a 3F-int `edge2face_cnt`, and keeps all four at their 3F sizes until clear_cache():\n'
             '    // 72 bytes a face during the sort, 84 held after it. Here the keys are sorted in a DoubleBuffer of two 3F\n'
             '    // buffers (cub sorts back and forth between them instead of taking a third), the one that does not hold\n'
             '    // the result is freed at once, the runs are counted first (the same encoding with its outputs discarded)\n'
             "    // so that `edges` and `edge2face_cnt` get exactly E entries, and the sorted keys and cub's scratch are\n"
             '    // freed before the return: 48 bytes a face during the sort, 12 E (about 18 a face) held after it. A sort\n'
             '    // of keys alone has one result whatever the sort (the keys in ascending order), and the encoding reads\n'
             '    // the same sorted keys. The stream is synchronised before a buffer that queued work reads is freed.\n'
             '    uint64_t *cu_keys, *cu_keys_alt;\n'
             '    CUDA_CHECK(cudaMalloc(&cu_keys, F * 3 * sizeof(uint64_t)));\n'
             '    expand_edges_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(this->faces.ptr, F, cu_keys);\n'
             '    CUDA_CHECK(cudaGetLastError());\n'
             '    CUDA_CHECK(cudaMalloc(&cu_keys_alt, F * 3 * sizeof(uint64_t)));\n'
             '\n'
             '    // sort edges\n'
             '    cub::DoubleBuffer<uint64_t> keys(cu_keys, cu_keys_alt);\n'
             '    size_t temp_storage_bytes = 0;\n'
             '    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(\n'
             '        nullptr, temp_storage_bytes,\n'
             '        keys,\n'
             '        F * 3, 0, sizeof(uint64_t) * 8, stream\n'
             '    ));\n'
             '    this->cub_temp_storage.resize(temp_storage_bytes);\n'
             '    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(\n'
             '        this->cub_temp_storage.ptr, temp_storage_bytes,\n'
             '        keys,\n'
             '        F * 3, 0, sizeof(uint64_t) * 8, stream\n'
             '    ));\n'
             '    uint64_t* cu_sorted = keys.Current();\n'
             '    CUDA_CHECK(cudaStreamSynchronize(stream));\n'
             '    CUDA_CHECK(cudaFree(keys.Alternate()));\n'
             '\n'
             '    // unique edges: counted first, then encoded into tables of exactly that length\n'
             '    int* num_edges;\n'
             '    CUDA_CHECK(cudaMalloc(&num_edges, sizeof(int)));\n'
             '    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(\n'
             '        nullptr, temp_storage_bytes,\n'
             '        cu_sorted, cub::DiscardOutputIterator<>(), cub::DiscardOutputIterator<>(), num_edges,\n'
             '        F * 3, stream\n'
             '    ));\n'
             '    this->cub_temp_storage.resize(temp_storage_bytes);\n'
             '    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(\n'
             '        this->cub_temp_storage.ptr, temp_storage_bytes,\n'
             '        cu_sorted, cub::DiscardOutputIterator<>(), cub::DiscardOutputIterator<>(), num_edges,\n'
             '        F * 3, stream\n'
             '    ));\n'
             '    int E;\n'
             '    CUDA_CHECK(cudaMemcpyAsync(&E, num_edges, sizeof(int), cudaMemcpyDeviceToHost, stream));\n'
             '    CUDA_CHECK(cudaStreamSynchronize(stream));\n'
             '    this->edges.free();\n'
             '    this->edges.resize(E);\n'
             '    this->edge2face_cnt.free();\n'
             '    this->edge2face_cnt.resize(E);\n'
             '    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(\n'
             '        nullptr, temp_storage_bytes,\n'
             '        cu_sorted, this->edges.ptr, this->edge2face_cnt.ptr, num_edges,\n'
             '        F * 3, stream\n'
             '    ));\n'
             '    this->cub_temp_storage.resize(temp_storage_bytes);\n'
             '    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(\n'
             '        this->cub_temp_storage.ptr, temp_storage_bytes,\n'
             '        cu_sorted, this->edges.ptr, this->edge2face_cnt.ptr, num_edges,\n'
             '        F * 3, stream\n'
             '    ));\n'
             '    CUDA_CHECK(cudaStreamSynchronize(stream));\n'
             '    CUDA_CHECK(cudaFree(num_edges));\n'
             '    CUDA_CHECK(cudaFree(cu_sorted));\n'
             '    this->cub_temp_storage.free();\n'
             '}\n')},
    {"id": "connectivity-boundaries-exact", "file": "src/connectivity.cu",
     "commit": "fill_holes' chain in less card memory",
     "old": ('    // Select boundary edges\n'
             '    size_t temp_storage_bytes = 0;\n'
             '    int *cu_num_boundary, *cu_edge_idx;\n'
             '    CUDA_CHECK(cudaMalloc(&cu_num_boundary, sizeof(int)));\n'
             '    CUDA_CHECK(cudaMalloc(&cu_edge_idx, E * sizeof(int)));\n'
             '    this->boundaries.resize(E);\n'
             '    arange_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(cu_edge_idx, E);\n'
             '    CUDA_CHECK(cudaGetLastError());\n'
             '    CUDA_CHECK(cub::DeviceSelect::If(\n'
             '        nullptr, temp_storage_bytes,\n'
             '        cu_edge_idx, this->boundaries.ptr, cu_num_boundary,\n'
             '        E,\n'
             '        is_boundary_edge{this->edge2face_cnt.ptr},\n'
             '        stream\n'
             '    ));\n'
             '    this->cub_temp_storage.resize(temp_storage_bytes);\n'
             '    CUDA_CHECK(cub::DeviceSelect::If(\n'
             '        this->cub_temp_storage.ptr, temp_storage_bytes,\n'
             '        cu_edge_idx, this->boundaries.ptr, cu_num_boundary,\n'
             '        E,\n'
             '        is_boundary_edge{this->edge2face_cnt.ptr},\n'
             '        stream\n'
             '    ));\n'
             '    CUDA_CHECK(cudaMemcpyAsync(&this->boundaries.size, cu_num_boundary, sizeof(int), cudaMemcpyDeviceToHost, stream));\n'
             '    CUDA_CHECK(cudaStreamSynchronize(stream));\n'
             '    CUDA_CHECK(cudaFree(cu_num_boundary));\n'
             '    CUDA_CHECK(cudaFree(cu_edge_idx));\n'),
     "new": ('    // Select boundary edges\n'
             '    // Spellcast3D: the edge indices come from a counting iterator instead of an arange table of E ints, and the\n'
             '    // boundary edges are counted first (the same selection with its output discarded) so that `boundaries`\n'
             '    // gets exactly B entries instead of E: the same indices in the same order (the selection keeps the input\n'
             '    // order), 8 bytes an edge less on the card during the call and 4 an edge less held after it.\n'
             '    size_t temp_storage_bytes = 0;\n'
             '    int *cu_num_boundary;\n'
             '    CUDA_CHECK(cudaMalloc(&cu_num_boundary, sizeof(int)));\n'
             '    cub::CountingInputIterator<int> cu_edge_idx(0);\n'
             '    CUDA_CHECK(cub::DeviceSelect::If(\n'
             '        nullptr, temp_storage_bytes,\n'
             '        cu_edge_idx, cub::DiscardOutputIterator<>(), cu_num_boundary,\n'
             '        E,\n'
             '        is_boundary_edge{this->edge2face_cnt.ptr},\n'
             '        stream\n'
             '    ));\n'
             '    this->cub_temp_storage.resize(temp_storage_bytes);\n'
             '    CUDA_CHECK(cub::DeviceSelect::If(\n'
             '        this->cub_temp_storage.ptr, temp_storage_bytes,\n'
             '        cu_edge_idx, cub::DiscardOutputIterator<>(), cu_num_boundary,\n'
             '        E,\n'
             '        is_boundary_edge{this->edge2face_cnt.ptr},\n'
             '        stream\n'
             '    ));\n'
             '    int num_boundaries;\n'
             '    CUDA_CHECK(cudaMemcpyAsync(&num_boundaries, cu_num_boundary, sizeof(int), cudaMemcpyDeviceToHost, stream));\n'
             '    CUDA_CHECK(cudaStreamSynchronize(stream));\n'
             '    this->boundaries.free();\n'
             '    this->boundaries.resize(num_boundaries);\n'
             '    CUDA_CHECK(cub::DeviceSelect::If(\n'
             '        nullptr, temp_storage_bytes,\n'
             '        cu_edge_idx, this->boundaries.ptr, cu_num_boundary,\n'
             '        E,\n'
             '        is_boundary_edge{this->edge2face_cnt.ptr},\n'
             '        stream\n'
             '    ));\n'
             '    this->cub_temp_storage.resize(temp_storage_bytes);\n'
             '    CUDA_CHECK(cub::DeviceSelect::If(\n'
             '        this->cub_temp_storage.ptr, temp_storage_bytes,\n'
             '        cu_edge_idx, this->boundaries.ptr, cu_num_boundary,\n'
             '        E,\n'
             '        is_boundary_edge{this->edge2face_cnt.ptr},\n'
             '        stream\n'
             '    ));\n'
             '    CUDA_CHECK(cudaStreamSynchronize(stream));\n'
             '    CUDA_CHECK(cudaFree(cu_num_boundary));\n')},
    {"id": "connectivity-manifold-boundary-verts-exact", "file": "src/connectivity.cu",
     "commit": "fill_holes' chain in less card memory",
     "old": ('    // Select manifold boundary vertices\n'
             '    size_t temp_storage_bytes = 0;\n'
             '    int *cu_num_manifold_boundary_verts, *cu_vert_idx, *cu_manifold_vert_idx;\n'
             '    CUDA_CHECK(cudaMalloc(&cu_num_manifold_boundary_verts, sizeof(int)));\n'
             '    CUDA_CHECK(cudaMalloc(&cu_vert_idx, V * sizeof(int)));\n'
             '    CUDA_CHECK(cudaMalloc(&cu_manifold_vert_idx, V * sizeof(int)));\n'
             '    arange_kernel<<<(V+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(cu_vert_idx, V);\n'
             '    CUDA_CHECK(cudaGetLastError());\n'
             '    CUDA_CHECK(cub::DeviceSelect::If(\n'
             '        nullptr, temp_storage_bytes,\n'
             '        cu_vert_idx, cu_manifold_vert_idx, cu_num_manifold_boundary_verts,\n'
             '        V,\n'
             '        is_manifold_boundary_vertex{this->vert_is_manifold.ptr, this->vert_is_boundary.ptr},\n'
             '        stream\n'
             '    ));\n'
             '    this->cub_temp_storage.resize(temp_storage_bytes);\n'
             '    CUDA_CHECK(cub::DeviceSelect::If(\n'
             '        this->cub_temp_storage.ptr, temp_storage_bytes,\n'
             '        cu_vert_idx, cu_manifold_vert_idx, cu_num_manifold_boundary_verts,\n'
             '        V,\n'
             '        is_manifold_boundary_vertex{this->vert_is_manifold.ptr, this->vert_is_boundary.ptr},\n'
             '        stream\n'
             '    ));\n'
             '    int manifold_boundary_vert_count;\n'
             '    CUDA_CHECK(cudaMemcpyAsync(&manifold_boundary_vert_count, cu_num_manifold_boundary_verts, sizeof(int), cudaMemcpyDeviceToHost, stream));\n'
             '    CUDA_CHECK(cudaStreamSynchronize(stream));\n'
             '    CUDA_CHECK(cudaFree(cu_num_manifold_boundary_verts));\n'
             '    CUDA_CHECK(cudaFree(cu_vert_idx));\n'),
     "new": ('    // Select manifold boundary vertices\n'
             '    // Spellcast3D: the vertex indices come from a counting iterator instead of an arange table of V ints, and\n'
             '    // the selected vertices are counted first (the same selection with its output discarded) so that their\n'
             '    // list gets exactly as many entries as there are instead of V: the same indices in the same order (the\n'
             '    // selection keeps the input order), 8 bytes a vertex less on the card during the call.\n'
             '    size_t temp_storage_bytes = 0;\n'
             '    int *cu_num_manifold_boundary_verts, *cu_manifold_vert_idx = nullptr;\n'
             '    CUDA_CHECK(cudaMalloc(&cu_num_manifold_boundary_verts, sizeof(int)));\n'
             '    cub::CountingInputIterator<int> cu_vert_idx(0);\n'
             '    CUDA_CHECK(cub::DeviceSelect::If(\n'
             '        nullptr, temp_storage_bytes,\n'
             '        cu_vert_idx, cub::DiscardOutputIterator<>(), cu_num_manifold_boundary_verts,\n'
             '        V,\n'
             '        is_manifold_boundary_vertex{this->vert_is_manifold.ptr, this->vert_is_boundary.ptr},\n'
             '        stream\n'
             '    ));\n'
             '    this->cub_temp_storage.resize(temp_storage_bytes);\n'
             '    CUDA_CHECK(cub::DeviceSelect::If(\n'
             '        this->cub_temp_storage.ptr, temp_storage_bytes,\n'
             '        cu_vert_idx, cub::DiscardOutputIterator<>(), cu_num_manifold_boundary_verts,\n'
             '        V,\n'
             '        is_manifold_boundary_vertex{this->vert_is_manifold.ptr, this->vert_is_boundary.ptr},\n'
             '        stream\n'
             '    ));\n'
             '    int manifold_boundary_vert_count;\n'
             '    CUDA_CHECK(cudaMemcpyAsync(&manifold_boundary_vert_count, cu_num_manifold_boundary_verts, sizeof(int), cudaMemcpyDeviceToHost, stream));\n'
             '    CUDA_CHECK(cudaStreamSynchronize(stream));\n'
             '    if (manifold_boundary_vert_count > 0) {\n'
             '        CUDA_CHECK(cudaMalloc(&cu_manifold_vert_idx, manifold_boundary_vert_count * sizeof(int)));\n'
             '        CUDA_CHECK(cub::DeviceSelect::If(\n'
             '            nullptr, temp_storage_bytes,\n'
             '            cu_vert_idx, cu_manifold_vert_idx, cu_num_manifold_boundary_verts,\n'
             '            V,\n'
             '            is_manifold_boundary_vertex{this->vert_is_manifold.ptr, this->vert_is_boundary.ptr},\n'
             '            stream\n'
             '        ));\n'
             '        this->cub_temp_storage.resize(temp_storage_bytes);\n'
             '        CUDA_CHECK(cub::DeviceSelect::If(\n'
             '            this->cub_temp_storage.ptr, temp_storage_bytes,\n'
             '            cu_vert_idx, cu_manifold_vert_idx, cu_num_manifold_boundary_verts,\n'
             '            V,\n'
             '            is_manifold_boundary_vertex{this->vert_is_manifold.ptr, this->vert_is_boundary.ptr},\n'
             '            stream\n'
             '        ));\n'
             '        CUDA_CHECK(cudaStreamSynchronize(stream));\n'
             '    }\n'
             '    CUDA_CHECK(cudaFree(cu_num_manifold_boundary_verts));\n')},
    {"id": "clean-up-fill-holes-free-tables-first", "file": "src/clean_up.cu",
     "commit": "fill_holes' chain in less card memory",
     "old": ('    // Update mesh\n'
             '    this->vertices.extend(new_num_bound_loops);\n'
             '    this->faces.extend(new_num_loop_boundaries);\n'),
     "new": ('    // Spellcast3D: the connectivity tables (edge counts, the vertex-edge and vertex-boundary lists, the flags,\n'
             "    // the boundary components and loops) and cub's scratch are not read again here - the update below needs\n"
             "    // the edges and this function's own lists only - and upstream frees all of them with clear_cache() right\n"
             '    // after it. They are freed before the vertex and face buffers are extended (an extension holds the old and\n'
             "    // the new buffer for a moment), so the update's widest moment is the mesh and the edges alone (each free\n"
             '    // synchronises the stream first). The edges go with clear_cache() below, as upstream: the object ends as\n'
             '    // upstream leaves it.\n'
             '    {\n'
             '        Buffer<uint64_t> kept_edges = this->edges;\n'
             '        this->edges = Buffer<uint64_t>();\n'
             '        this->clear_cache();\n'
             '        this->edges = kept_edges;\n'
             '    }\n'
             '\n'
             '    // Update mesh\n'
             '    this->vertices.extend(new_num_bound_loops);\n'
             '    this->faces.extend(new_num_loop_boundaries);\n')},

    # 3. Pixal3D's memory edits of the remesh (values unchanged: unsigned distances point by point, frees only).
    {"id": "remeshing-grid-bvh-chunks", "file": "cumesh/remeshing.py",
     "commit": "remeshing: Pixal3D's memory edits, without the chunked grid expansion",
     "old": ('        # Calculate UDF for current voxels\n'
             '        cell_size = scale / base_resolution\n'
             '        pts = ((coords.float() + 0.5) / base_resolution - 0.5) * scale + center\n'
             '        \n'
             '        distances = bvh.unsigned_distance(pts)[0]\n'
             '        distances -= eps\n'
             '        distances = torch.abs(distances) # Keep positive for refinement check\n'
             '        \n'
             '        # Determine which voxels are near the surface (Narrow Band)\n'
             '        # 0.87 is approx sqrt(3)/2, covering the diagonal radius of a voxel\n'
             '        subdiv_mask = distances < 0.87 * cell_size\n'
             '        coords = coords[subdiv_mask]\n'),
     "new": ('        # Calculate UDF for current voxels - chunk BVH queries to limit peak allocation\n'
             '        cell_size = scale / base_resolution\n'
             '        pts = ((coords.float() + 0.5) / base_resolution - 0.5) * scale + center\n'
             '        \n'
             '        BVH_CHUNK = 2_000_000  # Process 2M points at a time\n'
             '        if pts.shape[0] <= BVH_CHUNK:\n'
             '            distances = bvh.unsigned_distance(pts)[0]\n'
             '        else:\n'
             '            distances = torch.empty(pts.shape[0], device=device, dtype=torch.float32)\n'
             '            for i in range(0, pts.shape[0], BVH_CHUNK):\n'
             '                distances[i:i+BVH_CHUNK] = bvh.unsigned_distance(pts[i:i+BVH_CHUNK])[0]\n'
             '        del pts\n'
             '        distances -= eps\n'
             '        distances = torch.abs(distances) # Keep positive for refinement check\n'
             '        \n'
             '        # Determine which voxels are near the surface (Narrow Band)\n'
             '        # 0.87 is approx sqrt(3)/2, covering the diagonal radius of a voxel\n'
             '        subdiv_mask = distances < 0.87 * cell_size\n'
             '        del distances\n'
             '        coords = coords[subdiv_mask]\n'
             '        del subdiv_mask\n')},
    {"id": "remeshing-vertex-bvh-chunks", "file": "cumesh/remeshing.py",
     "commit": "remeshing: Pixal3D's memory edits, without the chunked grid expansion",
     "old": '    distances_vert = bvh.unsigned_distance(pts_vert)[0]\n',
     "new": ('    BVH_CHUNK = 2_000_000\n'
             '    if pts_vert.shape[0] <= BVH_CHUNK:\n'
             '        distances_vert = bvh.unsigned_distance(pts_vert)[0]\n'
             '    else:\n'
             '        distances_vert = torch.empty(pts_vert.shape[0], device=device, dtype=torch.float32)\n'
             '        for i in range(0, pts_vert.shape[0], BVH_CHUNK):\n'
             '            distances_vert[i:i+BVH_CHUNK] = bvh.unsigned_distance(pts_vert[i:i+BVH_CHUNK])[0]\n'
             '    del pts_vert\n')},
    {"id": "remeshing-free-after-dual-contouring", "file": "cumesh/remeshing.py",
     "commit": "remeshing: Pixal3D's memory edits, without the chunked grid expansion",
     "old": ('        *hashmap_vert, coords, distances_vert, resolution + 1, resolution + 1, resolution + 1\n'
             '    )\n'
             '    \n'),
     "new": ('        *hashmap_vert, coords, distances_vert, resolution + 1, resolution + 1, resolution + 1\n'
             '    )\n'
             '    \n'
             '    # Free data only needed for dual contouring\n'
             '    del hashmap_vert, distances_vert, grid_verts\n'
             '    torch.cuda.empty_cache()\n'
             '    \n')},
    {"id": "remeshing-free-topology-a", "file": "cumesh/remeshing.py",
     "commit": "remeshing: Pixal3D's memory edits, without the chunked grid expansion",
     "old": ('    connected_voxel = edge_neighbor_voxel[intersected != 0]                           # (M, 4, 3)\n'),
     "new": ('    connected_voxel = edge_neighbor_voxel[intersected != 0]                           # (M, 4, 3)\n'
             '    del edge_neighbor_voxel\n')},
    {"id": "remeshing-free-topology-b", "file": "cumesh/remeshing.py",
     "commit": "remeshing: Pixal3D's memory edits, without the chunked grid expansion",
     "old": ('        connected_voxel.reshape(-1, 3)\n'
             '    ], dim=1)\n'),
     "new": ('        connected_voxel.reshape(-1, 3)\n'
             '    ], dim=1)\n'
             '    del connected_voxel\n')},
    {"id": "remeshing-free-topology-c", "file": "cumesh/remeshing.py",
     "commit": "remeshing: Pixal3D's memory edits, without the chunked grid expansion",
     "old": ('resolution, resolution, resolution).reshape(M, 4).int()\n'),
     "new": ('resolution, resolution, resolution).reshape(M, 4).int()\n'
             '    del connected_voxel_hash_key, hashmap_vox\n')},
    {"id": "remeshing-free-topology-d", "file": "cumesh/remeshing.py",
     "commit": "remeshing: Pixal3D's memory edits, without the chunked grid expansion",
     "old": ('    intersected_dir = intersected[connected_voxel_valid].int()\n'),
     "new": ('    intersected_dir = intersected[connected_voxel_valid].int()\n'
             '    del connected_voxel_indices, connected_voxel_valid\n')},

    # 7. cuBVH's tree built on several threads (2026-11-04): the same nodes and triangle order as the one-thread
    # build, which stays reachable (CUBVH_BUILD_THREADS=1) and is compared against (CUBVH_BUILD_CHECK=1). A 32 M-
    # triangle Pixal3D mesh took 20 s on one core with the card idle.
    {"id": "cubvh-build-threads-includes", "file": "third_party/cubvh/src/bvh.cu",
     "commit": 'cubvh: the tree built on several threads, the same tree byte for byte; version 0.0.1.post5+cu128torch2.8',
     "old": '#include <stack>\n#include <iostream>\n#include <cstdio>\n',
     "new": '#include <stack>\n#include <iostream>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <thread>\n#include <unordered_map>\n'},
    {"id": "cubvh-build-threads", "file": "third_party/cubvh/src/bvh.cu",
     "commit": 'cubvh: the tree built on several threads, the same tree byte for byte; version 0.0.1.post5+cu128torch2.8',
     "old": '    void build(std::vector<Triangle>& triangles, uint32_t n_primitives_per_leaf) override {\n        m_nodes.clear();\n\n        // Root\n        m_nodes.emplace_back();\n        m_nodes.front().bb = BoundingBox(std::begin(triangles), std::end(triangles));\n\n        struct BuildNode {\n            int node_idx;\n            std::vector<Triangle>::iterator begin;\n            std::vector<Triangle>::iterator end;\n        };\n\n        std::stack<BuildNode> build_stack;\n        build_stack.push({0, std::begin(triangles), std::end(triangles)});\n\n    while (!build_stack.empty()) {\n            const BuildNode& curr = build_stack.top();\n            size_t node_idx = curr.node_idx;\n\n            std::array<BuildNode, BRANCHING_FACTOR> children;\n            children[0].begin = curr.begin;\n            children[0].end = curr.end;\n\n            build_stack.pop();\n\n            // Partition the triangles into the children\n            int n_children = 1;\n            while (n_children < BRANCHING_FACTOR) {\n                for (int i = n_children - 1; i >= 0; --i) {\n                    auto& child = children[i];\n\n                    // Choose axis with maximum standard deviation\n                    Vector3f mean = Vector3f::Zero();\n                    for (auto it = child.begin; it != child.end; ++it) {\n                        mean += it->centroid();\n                    }\n                    mean /= (float)std::distance(child.begin, child.end);\n\n                    Vector3f var = Vector3f::Zero();\n                    for (auto it = child.begin; it != child.end; ++it) {\n                        Vector3f diff = it->centroid() - mean;\n                        var += diff.cwiseProduct(diff);\n                    }\n                    var /= (float)std::distance(child.begin, child.end);\n\n                    Vector3f::Index axis;\n                    var.maxCoeff(&axis);\n\n                    auto m = child.begin + std::distance(child.begin, child.end)/2;\n                    std::nth_element(child.begin, m, child.end, [&](const Triangle& tri1, const Triangle& tri2) { return tri1.centroid(axis) < tri2.centroid(axis); });\n\n                    children[i*2].begin = children[i].begin;\n                    children[i*2+1].end = children[i].end;\n                    children[i*2].end = children[i*2+1].begin = m;\n                }\n\n                n_children *= 2;\n            }\n\n            // Create next build nodes\n            m_nodes[node_idx].left_idx = (int)m_nodes.size();\n            for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {\n                auto& child = children[i];\n                assert(child.begin != child.end);\n                child.node_idx = (int)m_nodes.size();\n\n                m_nodes.emplace_back();\n                m_nodes.back().bb = BoundingBox(child.begin, child.end);\n\n                if (std::distance(child.begin, child.end) <= n_primitives_per_leaf) {\n                    m_nodes.back().left_idx = -(int)std::distance(std::begin(triangles), child.begin)-1;\n                    m_nodes.back().right_idx = -(int)std::distance(std::begin(triangles), child.end)-1;\n                } else {\n                    build_stack.push(child);\n                }\n            }\n            m_nodes[node_idx].right_idx = (int)m_nodes.size();\n        }\n\n        // Thread the BVH with escape links for stackless traversal.\n        // Initialize all escape indices to -1.\n        for (auto& n : m_nodes) n.escape_idx = -1;\n\n        // Recursive lambda to assign escape links in pre-order (child order 0..BRANCHING_FACTOR-1)\n        std::function<void(int,int)> thread_bvh = [&](int node_idx, int escape_idx) {\n            TriangleBvhNode& node = m_nodes[node_idx];\n            node.escape_idx = escape_idx;\n            if (node.left_idx < 0) return; // leaf\n            int first_child = node.left_idx;\n            int end_child = node.right_idx; // exclusive\n            for (int c = first_child; c < end_child; ++c) {\n                int next_escape = (c+1 < end_child) ? (c+1) : escape_idx;\n                thread_bvh(c, next_escape);\n            }\n        };\n\n        if (!m_nodes.empty()) {\n            thread_bvh(0, -1);\n        }\n\n        // Spellcast3D: no upload here, so that the build makes no CUDA call and a worker thread can run it\n        // while the GPU is busy; the owner calls upload() when it wants the nodes on the card.\n\n        // std::cout << "[INFO] Built TriangleBvh: nodes=" << m_nodes.size() << std::endl;\n    }\n',
     "new": '    // Spellcast3D: the tree built on several threads, the same tree the one-thread build makes, byte for byte.\n    //\n    // The one-thread build (build_range below, the original algorithm) takes a node off a stack, partitions its\n    // triangles into BRANCHING_FACTOR children with nth_element (in place, within the node\'s own range), gives the\n    // children the next node indices, and pushes the internal ones, so the last child\'s whole subtree is built\n    // before the one before it. Two facts make the parallel build exact: a subtree only ever touches its own\n    // triangle range (nth_element calls on disjoint ranges are the same calls whatever the thread), and the\n    // number of nodes a subtree gets is a function of its triangle count alone (every split is at n/2), so every\n    // node index is known before the subtree is built. build_subtree partitions a node as the one-thread build\n    // would, computes each child\'s node count, hands each child its index range in the one-thread order (the\n    // last child\'s descendants first) and builds the children\'s subtrees on threads of their own, each writing\n    // its nodes into its own slice of m_nodes and reordering its own slice of the triangles. A 32 M-triangle\n    // mesh (a Pixal3D prop at 1536) took 20 s on one core of a Ryzen 9, the card idle meanwhile.\n    //\n    // CUBVH_BUILD_THREADS=N picks the thread count (1 = the one-thread build; unset: the machine\'s threads, at\n    // most 64). CUBVH_BUILD_CHECK=1 (developers) builds the tree both ways and compares nodes and triangles.\n\n    using TriIt = std::vector<Triangle>::iterator;\n\n    struct BuildRange {\n        int node_idx;\n        TriIt begin;\n        TriIt end;\n    };\n\n    // The partition of a node\'s triangles into its children: the original code, unchanged.\n    static void split_range(const BuildRange& curr, std::array<BuildRange, BRANCHING_FACTOR>& children) {\n        children[0].begin = curr.begin;\n        children[0].end = curr.end;\n        int n_children = 1;\n        while (n_children < BRANCHING_FACTOR) {\n            for (int i = n_children - 1; i >= 0; --i) {\n                auto& child = children[i];\n\n                // Choose axis with maximum standard deviation\n                Vector3f mean = Vector3f::Zero();\n                for (auto it = child.begin; it != child.end; ++it) {\n                    mean += it->centroid();\n                }\n                mean /= (float)std::distance(child.begin, child.end);\n\n                Vector3f var = Vector3f::Zero();\n                for (auto it = child.begin; it != child.end; ++it) {\n                    Vector3f diff = it->centroid() - mean;\n                    var += diff.cwiseProduct(diff);\n                }\n                var /= (float)std::distance(child.begin, child.end);\n\n                Vector3f::Index axis;\n                var.maxCoeff(&axis);\n\n                auto m = child.begin + std::distance(child.begin, child.end)/2;\n                std::nth_element(child.begin, m, child.end, [&](const Triangle& tri1, const Triangle& tri2) { return tri1.centroid(axis) < tri2.centroid(axis); });\n\n                children[i*2].begin = children[i].begin;\n                children[i*2+1].end = children[i].end;\n                children[i*2].end = children[i*2+1].begin = m;\n            }\n\n            n_children *= 2;\n        }\n    }\n\n    // The nodes of the subtree under an internal node with n triangles, that node included: the children\'s\n    // sizes are what split_range makes them (n/2 and n - n/2, each halved again), a leaf has at most\n    // n_primitives_per_leaf triangles.\n    static size_t subtree_nodes(size_t n, uint32_t n_primitives_per_leaf, std::unordered_map<size_t, size_t>& memo) {\n        auto found = memo.find(n);\n        if (found != memo.end()) return found->second;\n        const size_t h = n / 2, r = n - h;\n        const size_t sizes[4] = {h / 2, h - h / 2, r / 2, r - r / 2};\n        size_t total = 1;\n        for (size_t s : sizes) {\n            total += (s <= n_primitives_per_leaf) ? 1 : subtree_nodes(s, n_primitives_per_leaf, memo);\n        }\n        memo[n] = total;\n        return total;\n    }\n\n    // The children of `curr` get the node indices [next, next + BRANCHING_FACTOR); each child\'s bounding box,\n    // and the leaves\' triangle ranges, as the one-thread build sets them. Returns the internal children.\n    static int make_children(std::vector<TriangleBvhNode>& nodes, std::vector<Triangle>& triangles,\n                             uint32_t n_primitives_per_leaf, const BuildRange& curr, size_t& next,\n                             std::array<BuildRange, BRANCHING_FACTOR>& children, bool* internal) {\n        nodes[curr.node_idx].left_idx = (int)next;\n        int n_internal = 0;\n        for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {\n            auto& child = children[i];\n            assert(child.begin != child.end);\n            child.node_idx = (int)next;\n            TriangleBvhNode& node = nodes[next++];\n            node.bb = BoundingBox(child.begin, child.end);\n            if (std::distance(child.begin, child.end) <= n_primitives_per_leaf) {\n                node.left_idx = -(int)std::distance(triangles.begin(), child.begin)-1;\n                node.right_idx = -(int)std::distance(triangles.begin(), child.end)-1;\n                internal[i] = false;\n            } else {\n                internal[i] = true;\n                ++n_internal;\n            }\n        }\n        nodes[curr.node_idx].right_idx = (int)next;\n        return n_internal;\n    }\n\n    // The one-thread build of everything under the ranges on the stack, its nodes written from index `next` on.\n    static void build_range(std::vector<TriangleBvhNode>& nodes, std::vector<Triangle>& triangles,\n                            uint32_t n_primitives_per_leaf, std::stack<BuildRange>& build_stack, size_t next) {\n        while (!build_stack.empty()) {\n            const BuildRange curr = build_stack.top();\n            build_stack.pop();\n            std::array<BuildRange, BRANCHING_FACTOR> children;\n            split_range(curr, children);\n            bool internal[BRANCHING_FACTOR];\n            make_children(nodes, triangles, n_primitives_per_leaf, curr, next, children, internal);\n            for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {\n                if (internal[i]) build_stack.push(children[i]);\n            }\n        }\n    }\n\n    // The subtree under `curr` (an internal node whose slot is set already), its nodes from `next` on, with up\n    // to `threads` threads: the node\'s children are built on threads of their own, each child\'s descendants in\n    // the index range the one-thread build would give them.\n    static void build_subtree(std::vector<TriangleBvhNode>& nodes, std::vector<Triangle>& triangles,\n                              uint32_t n_primitives_per_leaf, const BuildRange& curr, size_t next, int threads) {\n        const size_t n = (size_t)std::distance(curr.begin, curr.end);\n        if (threads <= 1 || n < 4 * MIN_TRIANGLES_PER_THREAD) {\n            std::stack<BuildRange> build_stack;\n            build_stack.push(curr);\n            build_range(nodes, triangles, n_primitives_per_leaf, build_stack, next);\n            return;\n        }\n        std::array<BuildRange, BRANCHING_FACTOR> children;\n        split_range(curr, children);\n        bool internal[BRANCHING_FACTOR];\n        make_children(nodes, triangles, n_primitives_per_leaf, curr, next, children, internal);\n\n        // the one-thread build takes the last child off its stack first, so its descendants come first\n        std::unordered_map<size_t, size_t> memo;\n        size_t descendants[BRANCHING_FACTOR];\n        size_t total = 0;\n        for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {\n            const size_t count = (size_t)std::distance(children[i].begin, children[i].end);\n            descendants[i] = internal[i] ? subtree_nodes(count, n_primitives_per_leaf, memo) - 1 : 0;\n            total += descendants[i];\n        }\n        size_t offset[BRANCHING_FACTOR];\n        size_t at = next;\n        for (int i = (int)BRANCHING_FACTOR - 1; i >= 0; --i) {\n            offset[i] = at;\n            at += descendants[i];\n        }\n\n        std::vector<std::thread> workers;\n        for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {\n            if (!internal[i]) continue;\n            // the threads shared by the children in proportion to their work, one at least\n            int share = (int)((threads * descendants[i] + total / 2) / (total ? total : 1));\n            if (share < 1) share = 1;\n            const BuildRange child = children[i];\n            workers.emplace_back([&nodes, &triangles, n_primitives_per_leaf, child, off = offset[i], share]() {\n                build_subtree(nodes, triangles, n_primitives_per_leaf, child, off, share);\n            });\n        }\n        for (auto& worker : workers) worker.join();\n    }\n\n    static constexpr size_t MIN_TRIANGLES_PER_THREAD = 65536;\n\n    static int build_threads() {\n        const char* env = std::getenv("CUBVH_BUILD_THREADS");\n        if (env && *env) {\n            const int asked = std::atoi(env);\n            if (asked >= 1) return asked > 256 ? 256 : asked;\n        }\n        const unsigned hardware = std::thread::hardware_concurrency();\n        return hardware == 0 ? 1 : (hardware > 64 ? 64 : (int)hardware);\n    }\n\n    // The nodes of the tree over `triangles` (reordered in place), with `threads` threads.\n    static void build_tree(std::vector<TriangleBvhNode>& nodes, std::vector<Triangle>& triangles,\n                           uint32_t n_primitives_per_leaf, int threads) {\n        std::unordered_map<size_t, size_t> memo;\n        nodes.clear();\n        nodes.resize(subtree_nodes(triangles.size(), n_primitives_per_leaf, memo));\n        nodes.front().bb = BoundingBox(std::begin(triangles), std::end(triangles));\n        BuildRange root{0, std::begin(triangles), std::end(triangles)};\n        build_subtree(nodes, triangles, n_primitives_per_leaf, root, 1, threads);\n    }\n\n    void build(std::vector<Triangle>& triangles, uint32_t n_primitives_per_leaf) override {\n        const int threads = build_threads();\n        const char* check = std::getenv("CUBVH_BUILD_CHECK");\n        if (check && *check && *check != \'0\' && threads > 1) {\n            // developers: the one-thread tree, then the parallel one, compared node by node and triangle by triangle\n            std::vector<Triangle> copy = triangles;\n            std::vector<TriangleBvhNode> single;\n            build_tree(single, copy, n_primitives_per_leaf, 1);\n            build_tree(m_nodes, triangles, n_primitives_per_leaf, threads);\n            const bool same_nodes = single.size() == m_nodes.size() &&\n                std::memcmp(single.data(), m_nodes.data(), single.size() * sizeof(TriangleBvhNode)) == 0;\n            // field by field: a Triangle has padding bytes, which a copy leaves undefined\n            bool same_triangles = copy.size() == triangles.size();\n            for (size_t i = 0; same_triangles && i < copy.size(); ++i) {\n                same_triangles = copy[i].id == triangles[i].id && copy[i].a == triangles[i].a &&\n                                 copy[i].b == triangles[i].b && copy[i].c == triangles[i].c;\n            }\n            std::cout << "[cubvh] build check: " << threads << " threads against 1: nodes "\n                      << (same_nodes ? "identical" : "DIFFERENT") << ", triangles "\n                      << (same_triangles ? "identical" : "DIFFERENT") << " (" << m_nodes.size() << " nodes, "\n                      << triangles.size() << " triangles)" << std::endl;\n        } else {\n            build_tree(m_nodes, triangles, n_primitives_per_leaf, threads);\n        }\n\n        // Thread the BVH with escape links for stackless traversal.\n        // Initialize all escape indices to -1.\n        for (auto& n : m_nodes) n.escape_idx = -1;\n\n        // Recursive lambda to assign escape links in pre-order (child order 0..BRANCHING_FACTOR-1)\n        std::function<void(int,int)> thread_bvh = [&](int node_idx, int escape_idx) {\n            TriangleBvhNode& node = m_nodes[node_idx];\n            node.escape_idx = escape_idx;\n            if (node.left_idx < 0) return; // leaf\n            int first_child = node.left_idx;\n            int end_child = node.right_idx; // exclusive\n            for (int c = first_child; c < end_child; ++c) {\n                int next_escape = (c+1 < end_child) ? (c+1) : escape_idx;\n                thread_bvh(c, next_escape);\n            }\n        };\n\n        if (!m_nodes.empty()) {\n            thread_bvh(0, -1);\n        }\n\n        // Spellcast3D: no upload here, so that the build makes no CUDA call and a worker thread can run it\n        // while the GPU is busy; the owner calls upload() when it wants the nodes on the card.\n\n        // std::cout << "[INFO] Built TriangleBvh: nodes=" << m_nodes.size() << std::endl;\n    }\n'},

    # 8. The ray-stab's inside test for any subset of a call's points (2026-11-08): cuBVH.raystab_inside, a kernel of its
    # own beside signed_distance's (which stays as it is) that seeds each point's stab directions from its index in the
    # signed_distance call it stands for, asks a range of the directions, and searches the nearest triangle only for
    # the points whose rays all hit. Spellcast3D's inner-copy search asks only the faces whose answer it uses.
    {"id": 'cubvh-raystab-hits-declaration', "file": 'third_party/cubvh/src/bvh.cu',
     "commit": "cubvh: the ray-stab's inside test for any subset of a call's points and any range of its directions, the same verdicts bit for bit; version 0.0.1.post6+cu128torch2.8",
     "old": '__global__ void raytrace_kernel(uint32_t n_elements, const Vector3f* __restrict__ rays_o, const Vector3f* __restrict__ rays_d, Vector3f* __restrict__ positions, int64_t* __restrict__ face_id, float* __restrict__ depth, const TriangleBvhNode* __restrict__ nodes, const Triangle* __restrict__ triangles);\n',
     "new": '__global__ void raytrace_kernel(uint32_t n_elements, const Vector3f* __restrict__ rays_o, const Vector3f* __restrict__ rays_d, Vector3f* __restrict__ positions, int64_t* __restrict__ face_id, float* __restrict__ depth, const TriangleBvhNode* __restrict__ nodes, const Triangle* __restrict__ triangles);\n__global__ void raystab_hits_kernel(uint32_t n_elements, const Vector3f* __restrict__ positions, const int32_t* __restrict__ rng_index, uint32_t first_ray, uint32_t last_ray, bool closest, bool* __restrict__ hits, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles);\n'},
    {"id": 'cubvh-raystab-hits', "file": 'third_party/cubvh/src/bvh.cu',
     "commit": "cubvh: the ray-stab's inside test for any subset of a call's points and any range of its directions, the same verdicts bit for bit; version 0.0.1.post6+cu128torch2.8",
     "old": '        return {res.first, -res.second};\n    }\n\n\n    void signed_distance_gpu(',
     "new": "        return {res.first, -res.second};\n    }\n\n    // Spellcast3D: signed_distance_raystab's inside test alone, for any subset of the points of a call and any\n    // range [first_ray, last_ray) of its stab directions. `rng` is seeded as the raystab kernel seeds it (from the\n    // point's index in the call it stands for), so direction i is the same direction, and each ray's hit or miss\n    // is a function of the point, the direction and the tree alone (ray_intersect_any, see above): whether some\n    // ray of the range escapes is the answer signed_distance_raystab's loop gives over those directions, in\n    // whatever range or order they are asked. With `closest`, a point whose rays all hit is then tested as\n    // signed_distance_raystab tests it, by its distance to the nearest triangle (-distance < 0: a point on the\n    // surface is not inside), so the whole range answers signed_distance_raystab(...).second < 0 bit for bit and\n    // the nearest-triangle search runs only for the points whose rays all hit. True for a point whose rays of\n    // the range all hit (and, with closest, that lies off the surface).\n    __host__ __device__ static bool raystab_hits(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, uint32_t first_ray, uint32_t last_ray, bool closest, float max_distance_sq = MAX_DIST_SQ, pcg32 rng={}) {\n        Vector2f offset = {rng.next_float(), rng.next_float()};\n\n        static constexpr uint32_t N_STAB_RAYS = 32;     // signed_distance_raystab's\n        if (last_ray > N_STAB_RAYS) {\n            last_ray = N_STAB_RAYS;\n        }\n        for (uint32_t i = first_ray; i < last_ray; ++i) {\n            Vector3f d = fibonacci_dir<N_STAB_RAYS>(i, offset);\n            if (!ray_intersect_any(point, -d, bvhnodes, triangles) || !ray_intersect_any(point, d, bvhnodes, triangles)) {\n                return false;\n            }\n        }\n\n        if (closest) {\n            auto res = closest_triangle(point, bvhnodes, triangles, max_distance_sq);\n            return -res.second < 0.0f;\n        }\n        return true;\n    }\n\n    void raystab_hits_gpu(uint32_t n_elements, const float* positions, const int32_t* rng_index, uint32_t first_ray, uint32_t last_ray, bool closest, bool* hits, const Triangle* gpu_triangles, cudaStream_t stream) override {\n        linear_kernel(raystab_hits_kernel, 0u, stream,\n            n_elements,\n            (const Vector3f*)positions,\n            rng_index,\n            first_ray,\n            last_ray,\n            closest,\n            hits,\n            m_nodes_gpu.data(),\n            gpu_triangles\n        );\n    }\n\n\n    void signed_distance_gpu("},
    {"id": 'cubvh-raystab-hits-kernel', "file": 'third_party/cubvh/src/bvh.cu',
     "commit": "cubvh: the ray-stab's inside test for any subset of a call's points and any range of its directions, the same verdicts bit for bit; version 0.0.1.post6+cu128torch2.8",
     "old": '__global__ void unsigned_distance_kernel(\n',
     "new": "// Spellcast3D: TriangleBvh4::raystab_hits for each point. rng_index (may be null: the point's index here) is the\n// point's index in the signed_distance call it stands for, which seeds its stab directions' random offset there\n// as signed_distance_raystab_kernel seeds it (rng.advance(i * 2) on a uint32 index), and the nearest-triangle\n// search has that kernel's bound (max_distance*max_distance, max_distance = MAX_DIST).\n__global__ void raystab_hits_kernel(\n    uint32_t n_elements, const Vector3f* __restrict__ positions, const int32_t* __restrict__ rng_index,\n    uint32_t first_ray, uint32_t last_ray, bool closest, bool* __restrict__ hits,\n    const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles\n) {\n    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i >= n_elements) return;\n\n    uint32_t seed_index = rng_index ? (uint32_t)rng_index[i] : i;\n    float max_distance = MAX_DIST;\n    pcg32 rng;\n    rng.advance(seed_index * 2);\n\n    Vector3f point = positions[i];\n\n    hits[i] = TriangleBvh4::raystab_hits(point, bvhnodes, triangles, first_ray, last_ray, closest, max_distance*max_distance, rng);\n}\n\n__global__ void unsigned_distance_kernel(\n"},
    {"id": 'cubvh-tree-raystab-hits', "file": 'third_party/cubvh/include/gpu/bvh.cuh',
     "commit": "cubvh: the ray-stab's inside test for any subset of a call's points and any range of its directions, the same verdicts bit for bit; version 0.0.1.post6+cu128torch2.8",
     "old": '    virtual void ray_trace_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d, float* positions, int64_t* face_id, float* depth, const Triangle* gpu_triangles, cudaStream_t stream) = 0;\n',
     "new": "    virtual void ray_trace_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d, float* positions, int64_t* face_id, float* depth, const Triangle* gpu_triangles, cudaStream_t stream) = 0;\n    // Spellcast3D: the raystab signed distance's inside test for a subset of a call's points (bvh.cu, raystab_hits)\n    virtual void raystab_hits_gpu(uint32_t n_elements, const float* positions, const int32_t* rng_index, uint32_t first_ray, uint32_t last_ray, bool closest, bool* hits, const Triangle* gpu_triangles, cudaStream_t stream) = 0;\n"},
    {"id": 'cubvh-api-raystab-hits', "file": 'third_party/cubvh/include/gpu/api_gpu.h',
     "commit": "cubvh: the ray-stab's inside test for any subset of a call's points and any range of its directions, the same verdicts bit for bit; version 0.0.1.post6+cu128torch2.8",
     "old": '    virtual void signed_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw, uint32_t mode) = 0;\n',
     "new": "    virtual void signed_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw, uint32_t mode) = 0;\n    // Spellcast3D: signed_distance(mode raystab)'s inside test alone, per point (hits: bool), over the stab\n    // directions [first_ray, last_ray), each point's directions seeded from rng_index (int32; none: its index\n    // here); with closest, a point whose rays all hit is also held off the surface (bvh.cu, raystab_hits).\n    virtual void raystab_hits(at::Tensor positions, at::optional<at::Tensor> rng_index, at::Tensor hits, uint32_t first_ray, uint32_t last_ray, bool closest) = 0;\n"},
    {"id": 'cubvh-impl-raystab-hits', "file": 'third_party/cubvh/src/api_gpu.cu',
     "commit": "cubvh: the ray-stab's inside test for any subset of a call's points and any range of its directions, the same verdicts bit for bit; version 0.0.1.post6+cu128torch2.8",
     "old": '    std::vector<Triangle> triangles_cpu;\n    GPUMemory<Triangle> triangles_gpu;\n',
     "new": '    void raystab_hits(at::Tensor positions, at::optional<at::Tensor> rng_index, at::Tensor hits, uint32_t first_ray, uint32_t last_ray, bool closest) {\n\n        const uint32_t n_elements = positions.size(0);\n        cudaStream_t stream = at::cuda::getCurrentCUDAStream();\n\n        std::lock_guard<std::mutex> lock(device_mutex);\n        ensure_on_device();\n        triangle_bvh->raystab_hits_gpu(n_elements, positions.data_ptr<float>(), rng_index.has_value() ? rng_index.value().data_ptr<int32_t>() : nullptr, first_ray, last_ray, closest, hits.data_ptr<bool>(), triangles_gpu.data(), stream);\n    }\n\n    std::vector<Triangle> triangles_cpu;\n    GPUMemory<Triangle> triangles_gpu;\n'},
    {"id": 'cubvh-bindings-raystab-hits', "file": 'third_party/cubvh/src/bindings.cpp',
     "commit": "cubvh: the ray-stab's inside test for any subset of a call's points and any range of its directions, the same verdicts bit for bit; version 0.0.1.post6+cu128torch2.8",
     "old": '    .def("signed_distance", &cuBVH::signed_distance)\n',
     "new": '    .def("signed_distance", &cuBVH::signed_distance)\n    .def("raystab_hits", &cuBVH::raystab_hits)\n'},
    {"id": 'bvh-py-raystab-directions', "file": 'cumesh/bvh.py',
     "commit": "cubvh: the ray-stab's inside test for any subset of a call's points and any range of its directions, the same verdicts bit for bit; version 0.0.1.post6+cu128torch2.8",
     "old": "_sdf_mode_to_id = {\n    'watertight': 0,\n    'raystab': 1,\n}\n",
     "new": "_sdf_mode_to_id = {\n    'watertight': 0,\n    'raystab': 1,\n}\n\n# Spellcast3D: the stab directions of signed_distance(mode='raystab'); each is stabbed both ways (bvh.cu N_STAB_RAYS)\nRAYSTAB_DIRECTIONS = 32\n"},
    {"id": 'bvh-py-raystab-inside', "file": 'cumesh/bvh.py',
     "commit": "cubvh: the ray-stab's inside test for any subset of a call's points and any range of its directions, the same verdicts bit for bit; version 0.0.1.post6+cu128torch2.8",
     "old": "    def signed_distance(self, positions, return_uvw=False, mode='watertight'):\n",
     "new": '    def raystab_inside(self, positions, rng_index=None, first_ray=0, last_ray=RAYSTAB_DIRECTIONS, closest=True):\n        # Spellcast3D: signed_distance(mode=\'raystab\')\'s inside test alone, as a bool per point: whether every\n        # stab ray of the directions [first_ray, last_ray) hits the mesh (both ways), and with `closest` also that\n        # the point lies off the surface. Over all the directions with closest it is exactly\n        # signed_distance(positions, mode=\'raystab\')[0] < 0, without the nearest-triangle search for the points\n        # a ray leaves. rng_index (int32, per point): the point\'s index in the signed_distance call it stands for,\n        # which seeds its directions there; default its index in this call. So a subset of a call\'s points, or\n        # its directions asked in parts (the points whose rays all hit in one range asked again for the next),\n        # gets every point the answer the whole call gives it.\n        # positions: torch.Tensor, float, [N, 3]\n\n        positions = positions.float().contiguous()\n\n        if not positions.is_cuda: positions = positions.cuda()\n\n        prefix = positions.shape[:-1]\n        positions = positions.view(-1, 3)\n\n        N = positions.shape[0]\n\n        if rng_index is not None:\n            rng_index = rng_index.to(device=positions.device, dtype=torch.int32).contiguous().view(-1)\n            assert rng_index.shape[0] == N, "one rng_index per point"\n\n        hits = torch.empty(N, dtype=torch.bool, device=positions.device)\n        if N > 0:\n            self.impl.raystab_hits(positions, rng_index, hits, int(first_ray), int(last_ray), bool(closest))\n\n        return hits.view(*prefix)\n\n    def signed_distance(self, positions, return_uvw=False, mode=\'watertight\'):\n'},
    # The escape test's rays (Spellcast3D's cavity check asks only whether each one leaves the mesh): cuBVH.ray_hits, the
    # any-hit traversal above for any rays, beside ray_trace (which stays as it is).
    {"id": 'bvh-py-ray-hits', "file": 'cumesh/bvh.py',
     "commit": "cubvh: an any-hit ray query, ray_trace's hit or miss bit for bit without the search for the nearest hit",
     "old": '\n    def unsigned_distance(self, positions, return_uvw=False):\n',
     "new": '\n    def ray_hits(self, rays_o, rays_d):\n        # Spellcast3D: whether each ray hits the mesh at all, as a bool: ray_trace(rays_o, rays_d)[1] >= 0 for the\n        # same rays, bit for bit, by a traversal that stops at the first hit instead of searching for the nearest.\n        # rays_o: torch.Tensor, float, [N, 3]\n        # rays_d: torch.Tensor, float, [N, 3]\n\n        rays_o = rays_o.float().contiguous()\n        rays_d = rays_d.float().contiguous()\n\n        if not rays_o.is_cuda: rays_o = rays_o.cuda()\n        if not rays_d.is_cuda: rays_d = rays_d.cuda()\n\n        prefix = rays_o.shape[:-1]\n        rays_o = rays_o.view(-1, 3)\n        rays_d = rays_d.view(-1, 3)\n\n        N = rays_o.shape[0]\n\n        hits = torch.empty(N, dtype=torch.bool, device=rays_o.device)\n        if N > 0:\n            self.impl.ray_hits(rays_o, rays_d, hits)\n\n        return hits.view(*prefix)\n\n    def unsigned_distance(self, positions, return_uvw=False):\n'},
    {"id": 'cubvh-api-ray-hits', "file": 'third_party/cubvh/include/gpu/api_gpu.h',
     "commit": "cubvh: an any-hit ray query, ray_trace's hit or miss bit for bit without the search for the nearest hit",
     "old": '    virtual void raystab_hits(at::Tensor positions, at::optional<at::Tensor> rng_index, at::Tensor hits, uint32_t first_ray, uint32_t last_ray, bool closest) = 0;\n',
     "new": "    // Spellcast3D: whether each ray hits the mesh at all (hits: bool) - ray_trace's face_id >= 0 for the same\n    // ray, without the search for the nearest hit after the first.\n    virtual void ray_hits(at::Tensor rays_o, at::Tensor rays_d, at::Tensor hits) = 0;\n    virtual void raystab_hits(at::Tensor positions, at::optional<at::Tensor> rng_index, at::Tensor hits, uint32_t first_ray, uint32_t last_ray, bool closest) = 0;\n"},
    {"id": 'cubvh-tree-ray-hits', "file": 'third_party/cubvh/include/gpu/bvh.cuh',
     "commit": "cubvh: an any-hit ray query, ray_trace's hit or miss bit for bit without the search for the nearest hit",
     "old": '    virtual void raystab_hits_gpu(uint32_t n_elements, const float* positions, const int32_t* rng_index, uint32_t first_ray, uint32_t last_ray, bool closest, bool* hits, const Triangle* gpu_triangles, cudaStream_t stream) = 0;\n',
     "new": "    // Spellcast3D: whether each ray hits the mesh at all (ray_trace's face_id >= 0), by an any-hit traversal\n    virtual void ray_hits_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d, bool* hits, const Triangle* gpu_triangles, cudaStream_t stream) = 0;\n    virtual void raystab_hits_gpu(uint32_t n_elements, const float* positions, const int32_t* rng_index, uint32_t first_ray, uint32_t last_ray, bool closest, bool* hits, const Triangle* gpu_triangles, cudaStream_t stream) = 0;\n"},
    {"id": 'cubvh-impl-ray-hits', "file": 'third_party/cubvh/src/api_gpu.cu',
     "commit": "cubvh: an any-hit ray query, ray_trace's hit or miss bit for bit without the search for the nearest hit",
     "old": '    }\n\n    void raystab_hits(at::Tensor positions, at::optional<at::Tensor> rng_index, at::Tensor hits, uint32_t first_ray, uint32_t last_ray, bool closest) {\n',
     "new": '    }\n\n    void ray_hits(at::Tensor rays_o, at::Tensor rays_d, at::Tensor hits) {\n\n        const uint32_t n_elements = rays_o.size(0);\n        cudaStream_t stream = at::cuda::getCurrentCUDAStream();\n\n        std::lock_guard<std::mutex> lock(device_mutex);\n        ensure_on_device();\n        triangle_bvh->ray_hits_gpu(n_elements, rays_o.data_ptr<float>(), rays_d.data_ptr<float>(), hits.data_ptr<bool>(), triangles_gpu.data(), stream);\n    }\n\n    void raystab_hits(at::Tensor positions, at::optional<at::Tensor> rng_index, at::Tensor hits, uint32_t first_ray, uint32_t last_ray, bool closest) {\n'},
    {"id": 'cubvh-bindings-ray-hits', "file": 'third_party/cubvh/src/bindings.cpp',
     "commit": "cubvh: an any-hit ray query, ray_trace's hit or miss bit for bit without the search for the nearest hit",
     "old": '    // Spellcast3D: the card copy on demand. upload and release_device touch no Python object; without the\n',
     "new": '    .def("ray_hits", &cuBVH::ray_hits)\n    // Spellcast3D: the card copy on demand. upload and release_device touch no Python object; without the\n'},
    {"id": 'cubvh-ray-hits-declaration', "file": 'third_party/cubvh/src/bvh.cu',
     "commit": "cubvh: an any-hit ray query, ray_trace's hit or miss bit for bit without the search for the nearest hit",
     "old": '\nstruct DistAndIdx {\n',
     "new": '__global__ void ray_hits_kernel(uint32_t n_elements, const Vector3f* __restrict__ rays_o, const Vector3f* __restrict__ rays_d, bool* __restrict__ hits, const TriangleBvhNode* __restrict__ nodes, const Triangle* __restrict__ triangles);\n\nstruct DistAndIdx {\n'},
    {"id": 'cubvh-ray-hits', "file": 'third_party/cubvh/src/bvh.cu',
     "commit": "cubvh: an any-hit ray query, ray_trace's hit or miss bit for bit without the search for the nearest hit",
     "old": '    void signed_distance_gpu(uint32_t n_elements, uint32_t mode, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, cudaStream_t stream) override {\n',
     "new": '    void ray_hits_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d, bool* hits, const Triangle* gpu_triangles, cudaStream_t stream) override {\n        linear_kernel(ray_hits_kernel, 0u, stream,\n            n_elements,\n            (const Vector3f*)rays_o,\n            (const Vector3f*)rays_d,\n            hits,\n            m_nodes_gpu.data(),\n            gpu_triangles\n        );\n    }\n\n    void signed_distance_gpu(uint32_t n_elements, uint32_t mode, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, cudaStream_t stream) override {\n'},
    {"id": 'cubvh-ray-hits-kernel', "file": 'third_party/cubvh/src/bvh.cu',
     "commit": "cubvh: an any-hit ray query, ray_trace's hit or miss bit for bit without the search for the nearest hit",
     "old": '__global__ void raytrace_kernel(\n',
     "new": "// Spellcast3D: whether each ray hits the mesh at all - ray_trace's face_id >= 0 for the same ray, bit for bit\n// (ray_intersect_any: the same triangle and box tests as ray_intersect, see above), without the search for the\n// nearest hit after the first.\n__global__ void ray_hits_kernel(\n    uint32_t n_elements, const Vector3f* __restrict__ rays_o, const Vector3f* __restrict__ rays_d,\n    bool* __restrict__ hits, const TriangleBvhNode* __restrict__ nodes, const Triangle* __restrict__ triangles\n) {\n    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i >= n_elements) return;\n\n    Vector3f ro = rays_o[i];\n    Vector3f rd = rays_d[i];\n\n    hits[i] = TriangleBvh4::ray_intersect_any(ro, rd, nodes, triangles);\n}\n\n__global__ void raytrace_kernel(\n"},
]


def apply(src: Path, only: list[str] | None = None) -> list[dict]:
    """Apply the hunks (all, or the ids in `only`) to the tree at `src`, which holds upstream's tree with cubvh's in
    third_party/cubvh; each `old` must occur exactly `count` times (1 unless said). Files are read and written as
    bytes (UTF-8, line endings as they are). Returns the hunks applied."""
    applied = []
    for hunk in HUNKS:
        if only is not None and hunk["id"] not in only:
            continue
        path = src / hunk["file"]
        text = path.read_bytes().decode("utf-8")
        count = text.count(hunk["old"])
        if count != hunk.get("count", 1):
            raise SystemExit(f"patch {hunk['id']}: {hunk['file']} has {count} matches of the text to replace, "
                             f"not {hunk.get('count', 1)}")
        path.write_bytes(text.replace(hunk["old"], hunk["new"]).encode("utf-8"))
        applied.append({"id": hunk["id"], "file": hunk["file"]})
    return applied


def git(repo: Path, *args: str, text: bool = True):
    cp = subprocess.run(["git", "-C", str(repo), "-c", "core.autocrlf=false", *args], capture_output=True, text=text)
    if cp.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} in {repo} failed: {cp.stderr if text else cp.stderr.decode()}")
    return cp.stdout


def has_commit(repo: Path, rev: str) -> bool:
    return subprocess.run(["git", "-C", str(repo), "cat-file", "-e", rev + "^{commit}"],
                          capture_output=True).returncode == 0


def ensure_commit(repo: Path, url: str, rev: str, what: str) -> str:
    """A git repository at `repo` (a clone, a bare one or a submodule's) that has commit `rev` - made with depth 1 when
    the folder is not a repository, fetched from `url` when it lacks the commit -> the commit's full id. Nothing else
    in the repository changes: the trees are taken with `git archive`."""
    if subprocess.run(["git", "-C", str(repo), "rev-parse", "--git-dir"], capture_output=True).returncode != 0 \
            or not (repo / ".git").exists() and not (repo / "HEAD").exists():
        repo.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", "--bare", str(repo)], check=True)
    if not has_commit(repo, rev):
        print(f"{what}: fetching {rev} from {url}", flush=True)
        subprocess.run(["git", "-C", str(repo), "fetch", "-q", "--depth", "1", url, rev], check=True)
    full = git(repo, "rev-parse", rev + "^{commit}").strip()
    if all(ch in "0123456789abcdef" for ch in rev) and not full.startswith(rev):
        raise SystemExit(f"{what}: {repo} resolves {rev} to {full}")
    return full


def export_tree(repo: Path, rev: str, dest: Path) -> None:
    """`git archive` of a commit into dest (submodule links are not in an archive: their folders stay empty)."""
    data = git(repo, "archive", "--format=tar", rev, text=False)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        t.extractall(dest)


def gitlinks(repo: Path, rev: str) -> dict[str, str]:
    """The submodule links of a commit's tree: {path: commit}."""
    links = {}
    for line in git(repo, "ls-tree", "-r", rev).splitlines():
        meta, path = line.split("\t", 1)
        mode, kind, obj = meta.split()
        if mode == "160000":
            links[path] = obj
    return links


def base_tree(src: Path, cubvh_src: Path, dest: Path) -> None:
    """Upstream's tree with cubvh's in third_party/cubvh, at the pins (what the hunks apply to)."""
    upstream = ensure_commit(src, PINS["upstream"][0], PINS["upstream"][1], "CuMesh upstream")
    cubvh = ensure_commit(cubvh_src, PINS["cubvh"][0], PINS["cubvh"][1], "cubvh")
    if gitlinks(src, upstream).get(CUBVH_DIR) != cubvh:
        raise SystemExit(f"upstream {upstream[:9]} does not point {CUBVH_DIR} at cubvh {cubvh[:9]}")
    export_tree(src, upstream, dest)
    export_tree(cubvh_src, cubvh, dest / CUBVH_DIR)


def files_of(root: Path) -> dict[str, Path]:
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}


def check(src: Path, rev: str, cubvh_src: Path, work: Path | None = None) -> int:
    """The fork's commit `rev` against upstream + cubvh + HUNKS: 0 when every file is equal (DOCS aside) and the only
    submodule link is eigen at its pin."""
    rev = ensure_commit(src, PINS["fork"][0], rev, "the fork")
    with tempfile.TemporaryDirectory(prefix="cumesh-check-", dir=work) as tmp:
        tmp = Path(tmp)
        base_tree(src, cubvh_src, tmp / "expected")
        applied = apply(tmp / "expected")
        export_tree(src, rev, tmp / "fork")
        want, have = files_of(tmp / "expected"), files_of(tmp / "fork")
        problems = []
        for name in sorted(set(want) | set(have)):
            if name in DOCS:
                continue
            if name not in have:
                problems.append(f"missing in the fork: {name}")
            elif name not in want:
                problems.append(f"only in the fork: {name}")
            elif not filecmp.cmp(want[name], have[name], shallow=False):
                problems.append(f"differs: {name}")
        links = gitlinks(src, rev)
        if links != {EIGEN_DIR: PINS["eigen"][1]}:
            problems.append(f"submodule links {links}, expected only {EIGEN_DIR} at {PINS['eigen'][1][:9]}")
        print(f"fork {rev[:9]} against upstream {PINS['upstream'][1][:9]} + cubvh {PINS['cubvh'][1][:9]} + "
              f"{len(applied)} hunks: {len(have)} files, {len(problems)} problem(s)"
              + (f" (not compared: {', '.join(DOCS)})" if DOCS else ""))
        for p in problems:
            print("  " + p)
        return 1 if problems else 0


def main() -> int:
    scratch = Path(__file__).resolve().parents[4] / "release" / "cache" / "wheels" / "src"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="the fork's commit against upstream + cubvh + the hunks")
    c.add_argument("--src", type=Path, default=scratch / "cumesh", help="a CuMesh clone (the fork's)")
    c.add_argument("--rev", default=PINS["fork"][1], help="the fork's commit (default: the wheel's pin)")
    c.add_argument("--cubvh-src", type=Path, default=scratch / "cubvh", help="a cubvh repository (made if missing)")
    p = sub.add_parser("apply", help="apply the hunks to a base tree (upstream + cubvh in third_party/cubvh)")
    p.add_argument("folder", type=Path)
    p.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.cmd == "check":
        return check(a.src.resolve(), a.rev, a.cubvh_src.resolve())
    for h in apply(a.folder, a.only):
        print(f"patched [{h['id']}] {h['file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
