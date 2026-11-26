"""launchers/pixal3d_glb.py: the GLB of a decoded prop in five steps (the staged engine's FinishSteps), checked on the
CPU against o_voxel's own to_glb.

The card is played by the CPU: a tensor "on the card" is a Card (a CPU tensor whose .numpy() is refused and that
cannot be mixed with CPU tensors, as on CUDA), made by .cuda() and by device="cuda" under CardMode. cumesh's
native part, xatlas, cuBVH, nvdiffrast and flex_gemm's grid_sample are small deterministic stand-ins that log
their calls; cumesh's own Python (CuMesh, its uv_unwrap) and o_voxel's postprocess.py are the installed files,
read from the pixal3d venv and run over the stand-ins. So to_glb itself and the five steps run over the same
stand-ins, and must call them alike and write the same GLB, byte for byte. OpenCV, trimesh and PIL are the real
ones.

Runs in the pixal3d venv, with the card hidden and pytest from code/logs/profiling/pytools:
    set CUDA_VISIBLE_DEVICES=-1
    set PYTHONPATH=code\\logs\\profiling\\pytools
    code\\venvs\\pixal3d\\Scripts\\python.exe -m pytest -p no:cacheprovider code\\tests\\test_pixal3d_glb.py -q
Skipped in the hub venv (no torch); that the module imports without torch is checked here with torch hidden, and
in the hub venv by test_pixal3d_launcher.py once pixal3d_server imports it. Whether the steps make to_glb's GLB on
the card is a check for the GPU (glb_equivalence.py).
"""

from __future__ import annotations

import contextlib
import copy
import gc
import importlib.util
import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import Any, Iterator

import pytest

from spellcast import paths

if str(paths.LAUNCHERS) not in sys.path:     # the launchers import their siblings as top-level modules
    sys.path.insert(0, str(paths.LAUNCHERS))


def test_the_module_imports_without_torch_numpy_or_pil() -> None:
    """pixal3d_server imports it, and the hub's tests import pixal3d_server without them."""
    code = ("import sys\n"
            "for name in ('torch', 'numpy', 'PIL', 'cv2', 'trimesh', 'cumesh', 'nvdiffrast', 'flex_gemm'):\n"
            "    sys.modules[name] = None\n"
            f"sys.path.insert(0, {str(paths.LAUNCHERS)!r})\n"
            "import pixal3d_glb\n"
            "print(pixal3d_glb.BARS[0])\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "Extracting GLB"


torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("trimesh")
pytest.importorskip("PIL")

from torch.overrides import TorchFunctionMode  # noqa: E402
from torch.utils._pytree import tree_flatten  # noqa: E402

import pixal3d_glb as glbmod  # noqa: E402
import pixal3d_guard as guard  # noqa: E402
import pixal3d_mv as mv  # noqa: E402


def _installed(package: str, file: str) -> Path | None:
    """A file of an installed package, found without importing the package (cumesh's imports its CUDA code)."""
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(list(spec.submodule_search_locations)[0]) / file
    return path if path.is_file() else None


CUMESH_PY = _installed("cumesh", "cumesh.py")
POSTPROCESS_PY = _installed("o_voxel", "postprocess.py")
TYPES_PY = paths.PIXAL3D_DIR / "pixal3d" / "staged" / "types.py"
ROTATION = ((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1))      # pixal3d_server.GLB_ROTATION
LAYOUT = {"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)}
RESOLUTION = 1024
DECIMATION = 40
TEXTURE = 16

needs_sources = pytest.mark.skipif(CUMESH_PY is None or POSTPROCESS_PY is None or not TYPES_PY.is_file(),
                                   reason="needs the pixal3d venv's cumesh and o_voxel, and the fork's types.py")


# ---- the card, played by the CPU -----------------------------------------------------------------------------------

class Card(torch.Tensor):
    """A tensor on the card. .cpu() gives a plain copy; .numpy() is refused; mixing it with a CPU tensor of one or
    more dimensions fails (indexing a Card with CPU indices is allowed, as on CUDA)."""

    @property
    def is_cuda(self) -> bool:  # type: ignore[override]
        return True

    @classmethod
    def __torch_function__(cls, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        if func is torch.Tensor.cpu:
            return _plain(args[0])
        if func is torch.Tensor.numpy:
            raise TypeError("can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() first.")
        _same_device(func, args, kwargs)
        return super().__torch_function__(func, types, args, kwargs)


def _plain(tensor: Any) -> Any:
    with torch._C.DisableTorchFunctionSubclass():
        return tensor.as_subclass(torch.Tensor).clone()


def on_card(tensor: Any) -> Card:
    with torch._C.DisableTorchFunctionSubclass():
        return tensor.as_subclass(torch.Tensor).clone().as_subclass(Card)


def _same_device(func: Any, args: Any, kwargs: Any) -> None:
    if func in (torch.Tensor.__getitem__, torch.Tensor.__setitem__) and isinstance(args[0], Card):
        values = [args[2]] if func is torch.Tensor.__setitem__ else []
    else:
        values = tree_flatten((args, kwargs))[0]
    if any(isinstance(v, torch.Tensor) and not isinstance(v, Card) and v.dim() > 0 for v in values):
        raise RuntimeError(f"{getattr(func, '__name__', func)}: Expected all tensors to be on the same device, "
                           "but found at least two devices, cuda:0 and cpu!")


def _wants_card(args: Any, kwargs: dict) -> bool:
    device = kwargs.get("device")
    return device is not None and torch.device(device).type == "cuda"


class CardMode(TorchFunctionMode):
    """.cuda() and factories with device="cuda" make Cards."""

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        kwargs = dict(kwargs or {})
        if func is torch.Tensor.cuda:
            return on_card(args[0])
        if func is torch.Tensor.to and (_wants_card(args, kwargs) or any(
                isinstance(a, (str, torch.device)) and torch.device(a).type == "cuda" for a in args[1:])):
            return on_card(args[0])
        if _wants_card(args, kwargs):
            kwargs["device"] = "cpu"
            return on_card(func(*args, **kwargs))
        return func(*args, **kwargs)


class NoCardMode(TorchFunctionMode):
    """A CPU step: nothing may go to the card or come from it."""

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        kwargs = dict(kwargs or {})
        if func in (torch.Tensor.cuda, torch.Tensor.to) or _wants_card(args, kwargs) \
                or any(isinstance(v, Card) for v in tree_flatten((args, kwargs))[0]):
            raise AssertionError(f"a CPU step touched the card: {getattr(func, '__name__', func)}")
        return func(*args, **kwargs)


CPU_ONLY = [False]
LOG: list[tuple] = []


class CardObject:
    """Something native that holds memory on the card: a CuMesh, a BVH, nvdiffrast's context."""

    def __init__(self) -> None:
        if CPU_ONLY[0]:
            raise AssertionError(f"a CPU step made a {type(self).__name__}")


def on_card_now() -> list[Any]:
    """Everything on the card that is still alive."""
    gc.collect()
    return [o for o in gc.get_objects() if type(o) is Card or issubclass(type(o), CardObject)]


def card_things(value: Any) -> list[Any]:
    """The Cards and card objects inside a state value."""
    if isinstance(value, dict):
        return [x for v in value.values() for x in card_things(v)]
    if isinstance(value, (list, tuple)):
        return [x for v in value for x in card_things(v)]
    return [value] if isinstance(value, (Card, CardObject)) else []


# ---- stand-ins for the native code ----------------------------------------------------------------------------------

class FakeMeshBackend(CardObject):
    """cumesh._C.CuMesh: small, deterministic, and every call logged."""

    def __init__(self) -> None:
        super().__init__()
        self.v: Any = None
        self.f: Any = None
        self.normals: Any = None
        self.chart_of: Any = None

    def init(self, v: Any, f: Any) -> None:
        assert isinstance(v, Card) and isinstance(f, Card) and v.dtype == torch.float32 and f.dtype == torch.int32
        LOG.append(("mesh.init", tuple(v.shape), tuple(f.shape)))
        self.v, self.f = v.clone(), f.clone()

    def read(self) -> tuple[Any, Any]:
        LOG.append(("mesh.read",))
        return self.v.clone(), self.f.clone()

    def num_vertices(self) -> int:
        return int(self.v.shape[0])

    def num_faces(self) -> int:
        return int(self.f.shape[0])

    def simplify_step(self, lambda_edge_length: float, lambda_skinny: float, thresh: float, timing: bool
                      ) -> tuple[int, int]:
        LOG.append(("mesh.simplify_step", lambda_edge_length, lambda_skinny, thresh, timing))
        self.f = self.f[:max(1, self.num_faces() * 3 // 4)].clone()
        return self.num_vertices(), self.num_faces()

    def remove_duplicate_faces(self) -> None:
        LOG.append(("mesh.remove_duplicate_faces",))
        seen, keep = set(), []
        for i, row in enumerate(self.f.tolist()):
            if tuple(sorted(row)) not in seen:
                seen.add(tuple(sorted(row)))
                keep.append(i)
        self.f = self.f[torch.tensor(keep, dtype=torch.long, device="cuda")].clone()

    def repair_non_manifold_edges(self) -> None:
        LOG.append(("mesh.repair_non_manifold_edges",))

    def remove_small_connected_components(self, min_area: float) -> None:
        LOG.append(("mesh.remove_small_connected_components", min_area))

    def unify_face_orientations(self) -> None:
        LOG.append(("mesh.unify_face_orientations",))
        f = self.f.clone()
        f[::3] = self.f[::3][:, [0, 2, 1]]
        self.f = f

    def fill_holes(self, max_hole_perimeter: float = 0.03) -> None:
        LOG.append(("mesh.fill_holes", max_hole_perimeter))             # closes nothing: the fake has no holes

    def remove_degenerate_faces(self, abs_thresh: float, rel_thresh: float) -> None:
        LOG.append(("mesh.remove_degenerate_faces", abs_thresh, rel_thresh))
        f = self.f
        good = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        self.f = f[good].clone()

    def compute_charts(self, *options: Any) -> None:
        LOG.append(("mesh.compute_charts", *[float(o) for o in options]))
        self.chart_of = (torch.arange(self.num_faces(), device="cuda") % 3).int()

    def read_atlas_charts(self) -> tuple:
        LOG.append(("mesh.read_atlas_charts",))
        vmaps, faces, voff, foff = [], [], [0], [0]
        for chart in range(3):
            chart_faces = self.f[self.chart_of == chart].long()
            used = torch.unique(chart_faces.reshape(-1))
            local = torch.searchsorted(used, chart_faces)
            vmaps.append(used.int())
            faces.append((local + voff[-1]).int())      # indices into the charts' vertex list (uv_unwrap subtracts)
            voff.append(voff[-1] + used.shape[0])
            foff.append(foff[-1] + chart_faces.shape[0])
        return (3, self.chart_of.clone(), torch.cat(vmaps), torch.cat(faces),
                torch.tensor(voff, dtype=torch.int32, device="cuda"),
                torch.tensor(foff, dtype=torch.int32, device="cuda"))

    def compute_vertex_normals(self) -> None:
        LOG.append(("mesh.compute_vertex_normals",))
        corners = self.v[self.f.long()]
        face_n = torch.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=1)
        normals = torch.zeros_like(self.v)
        for k in range(3):
            normals.index_add_(0, self.f[:, k].long(), face_n)
        self.normals = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def read_vertex_normals(self) -> Any:
        LOG.append(("mesh.read_vertex_normals",))
        return self.normals.clone()


class FakeAtlas:
    """cumesh.xatlas.Atlas: CPU only, as the real one; each chart unfolded flat into its own column."""

    def __init__(self) -> None:
        LOG.append(("xatlas.Atlas",))
        self.meshes: list[tuple[Any, Any]] = []

    def add_mesh(self, vertices: Any, faces: Any, normals: Any = None, uvs: Any = None) -> None:
        for tensor, dtype in ((vertices, torch.float32), (faces, torch.int32)):
            assert not isinstance(tensor, Card) and tensor.device.type == "cpu", "xatlas takes CPU tensors"
            assert tensor.dtype == dtype and tensor.is_contiguous()
        LOG.append(("xatlas.add_mesh", tuple(vertices.shape), tuple(faces.shape)))
        self.meshes.append((vertices.clone(), faces.clone(), None if uvs is None else uvs.clone()))
        self.given_uvs = False

    def compute_charts(self, **options: Any) -> None:
        LOG.append(("xatlas.compute_charts", tuple(sorted(options.items()))))
        self.given_uvs = bool(options.get("use_input_mesh_uvs"))

    def pack_charts(self, **options: Any) -> None:
        LOG.append(("xatlas.pack_charts", tuple(sorted(options.items()))))

    def get_mesh(self, index: int) -> tuple[Any, Any, Any]:
        vertices, faces, uvs = self.meshes[index]
        if self.given_uvs and uvs is not None:                         # packed again: the charts as given, moved
            return torch.arange(vertices.shape[0], dtype=torch.int32), faces.clone(), (uvs * 0.5 + 0.25).contiguous()
        n = vertices.shape[0]
        xref = torch.arange(n - 1, -1, -1, dtype=torch.int32)          # new vertex j is the chart's n-1-j
        new_faces = (n - 1 - faces.long()).int().contiguous()
        moved = vertices[xref.long()]
        uv = torch.stack([(index + 0.1 + 0.8 * (moved[:, 0] + 1) / 2) / len(self.meshes),
                          0.1 + 0.8 * (moved[:, 1] + 1) / 2], dim=1).float().contiguous()
        return xref, new_faces, uv


class FakeBVH(CardObject):
    """cumesh.cuBVH: built from CPU copies (as the real one, which uploads at the end), queried on the card."""

    def __init__(self, vertices: Any, triangles: Any) -> None:
        super().__init__()
        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(triangles):
            triangles = triangles.detach().cpu().numpy()
        LOG.append(("cuBVH", vertices.shape, triangles.shape))
        self.v = on_card(torch.from_numpy(vertices.copy()))
        self.f = on_card(torch.from_numpy(triangles.copy()).long())

    def unsigned_distance(self, positions: Any, return_uvw: bool = False) -> tuple[Any, Any, Any]:
        positions = positions.float().contiguous()
        if not positions.is_cuda:
            positions = positions.cuda()
        LOG.append(("cuBVH.unsigned_distance", tuple(positions.shape), return_uvw))
        distances, face_id = torch.cdist(positions, self.v[self.f].mean(dim=1)).min(dim=1)
        near = 1 / (1e-3 + (self.v[self.f[face_id]] - positions[:, None]).norm(dim=2))
        return distances, face_id, (near / near.sum(dim=1, keepdim=True) if return_uvw else None)


class FakeHostBVH:
    """cumesh.cuBVH of Spellcast3D's own wheel: the tree built in RAM alone (upload=False), uploaded and let go on
    demand, and uploaded by a query when it is not on the card. Not a CardObject: it holds memory on the card only
    between upload() and release_device(), as Card tensors, so on_card_now() sees exactly that."""

    def __init__(self, vertices: Any, triangles: Any, upload: bool = True) -> None:
        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(triangles):
            triangles = triangles.detach().cpu().numpy()
        LOG.append(("cuBVH", vertices.shape, triangles.shape))
        self.host_v, self.host_f = vertices.copy(), triangles.copy()
        self.v: Any = None
        self.f: Any = None
        if upload:
            self.upload()

    @property
    def on_device(self) -> bool:
        return self.v is not None

    def upload(self) -> None:
        assert not CPU_ONLY[0], "a CPU step uploaded a BVH"
        if self.v is None:
            LOG.append(("cuBVH.upload",))
            self.v = on_card(torch.from_numpy(self.host_v))
            self.f = on_card(torch.from_numpy(self.host_f).long())

    def release_device(self) -> None:
        if self.v is not None:
            LOG.append(("cuBVH.release_device",))
        self.v = self.f = None

    def unsigned_distance(self, positions: Any, return_uvw: bool = False) -> tuple[Any, Any, Any]:
        self.upload()
        return FakeBVH.unsigned_distance(self, positions, return_uvw)       # the same answers from the same tree


@contextlib.contextmanager
def host_trees(fakes: "Fakes") -> Iterator[None]:
    """The steps as they run with Spellcast3D's cumesh: cuBVH builds in RAM, and the module knows it."""
    before = glbmod._host_trees
    fakes.cumesh.cuBVH = FakeHostBVH
    glbmod._host_trees = True
    try:
        yield
    finally:
        fakes.cumesh.cuBVH = FakeBVH
        glbmod._host_trees = before


def fake_remesh(vertices: Any, faces: Any, center: Any, scale: float, resolution: int, band: float = 1,
                project_back: float = 0, verbose: bool = False, bvh: Any = None) -> tuple[Any, Any]:
    """cumesh.remeshing.remesh_narrow_band_dc: the surface moved a little, with a degenerate face and two
    duplicates in front (simplify keeps the front)."""
    assert isinstance(vertices, Card) and isinstance(faces, Card) and isinstance(center, Card)
    assert isinstance(bvh, (FakeBVH, FakeHostBVH))
    assert not isinstance(bvh, FakeHostBVH) or bvh.on_device, "the remesh got a tree that is not on the card"
    LOG.append(("remesh", tuple(vertices.shape), tuple(faces.shape), tuple(center.tolist()), scale, resolution,
                band, project_back, verbose))
    extra = torch.tensor([[0, 0, 1]], dtype=torch.int32, device="cuda")
    return (vertices * 0.98 + center * 0.01).contiguous(), torch.cat([extra, faces[:2], faces]).contiguous()


class FakeRasterizeContext(CardObject):
    def __init__(self) -> None:
        super().__init__()
        LOG.append(("RasterizeCudaContext",))


def fake_rasterize(ctx: Any, pos: Any, tri: Any, resolution: Any, ranges: Any = None, grad_db: bool = True
                   ) -> tuple[Any, Any]:
    """Each triangle covers the texel at its centre (the later one wins), with barycentrics (0.2, 0.3)."""
    assert isinstance(ctx, FakeRasterizeContext) and isinstance(pos, Card) and isinstance(tri, Card)
    assert tri.dtype == torch.int32 and pos.shape[0] == 1 and pos.shape[2] == 4
    height, width = resolution
    LOG.append(("rasterize", tuple(pos.shape), tuple(tri.shape), tuple(resolution)))
    rast = torch.zeros((1, height, width, 4), device="cuda", dtype=torch.float32)
    centre = pos[0][tri.long()][..., :2].mean(dim=1)
    xs = ((centre[:, 0] + 1) / 2 * width).long().clamp(0, width - 1).tolist()
    ys = ((centre[:, 1] + 1) / 2 * height).long().clamp(0, height - 1).tolist()
    for t, (x, y) in enumerate(zip(xs, ys)):
        rast[0, y, x] = torch.tensor([0.2, 0.3, 0.5, t + 1], device="cuda")
    return rast, None


def fake_interpolate(attr: Any, rast: Any, tri: Any, rast_db: Any = None, diff_attrs: Any = None) -> tuple[Any, Any]:
    assert isinstance(attr, Card) and isinstance(rast, Card) and isinstance(tri, Card)
    LOG.append(("interpolate", tuple(attr.shape), tuple(rast.shape), tuple(tri.shape)))
    ids = rast[0, ..., 3].long()
    covered = ids > 0
    corners = attr[0][tri[ids[covered] - 1].long()]
    u, v = rast[0][covered][:, 0:1], rast[0][covered][:, 1:2]
    out = torch.zeros((1, *rast.shape[1:3], attr.shape[-1]), device="cuda", dtype=torch.float32)
    out[0][covered] = corners[:, 0] * u + corners[:, 1] * v + corners[:, 2] * (1 - u - v)
    return out, None


def fake_grid_sample_3d(feats: Any, coords: Any, shape: Any, grid: Any, mode: str = "trilinear") -> Any:
    """The nearest voxel's features, scaled a little by where the point is: [1, L, C]."""
    assert isinstance(feats, Card) and isinstance(coords, Card) and isinstance(grid, Card)
    LOG.append(("grid_sample_3d", tuple(feats.shape), tuple(coords.shape), tuple(shape), tuple(grid.shape), mode))
    points = grid[0]
    nearest = torch.cdist(points, coords[:, 1:].float()).argmin(dim=1)
    return (feats[nearest] * (1 + 0.001 * points[:, :1])).unsqueeze(0)


class Fakes(types.SimpleNamespace):
    pass


@pytest.fixture()
def fakes(monkeypatch: pytest.MonkeyPatch) -> Iterator[Fakes]:
    """The stand-ins installed as cumesh, nvdiffrast.torch and flex_gemm.ops.grid_sample; cumesh's CuMesh is the
    installed cumesh.py over the stand-in native part."""
    if CUMESH_PY is None or POSTPROCESS_PY is None:
        pytest.skip("needs the pixal3d venv's cumesh and o_voxel")
    source = CUMESH_PY.read_text(encoding="utf-8")
    for line in ("from .xatlas import Atlas", "from . import _C"):
        assert line in source, f"cumesh.py changed: {line!r} is gone"
        source = source.replace(line, "")
    wrapper: dict[str, Any] = {"__name__": "cumesh_cumesh_under_test", "Atlas": FakeAtlas,
                               "_C": types.SimpleNamespace(CuMesh=FakeMeshBackend)}
    exec(compile(source, str(CUMESH_PY), "exec"), wrapper)

    cumesh = types.ModuleType("cumesh")
    remeshing = types.ModuleType("cumesh.remeshing")
    remeshing.remesh_narrow_band_dc = fake_remesh
    cumesh.remeshing, cumesh.CuMesh, cumesh.cuBVH, cumesh.Atlas = remeshing, wrapper["CuMesh"], FakeBVH, FakeAtlas
    nvdiffrast = types.ModuleType("nvdiffrast")
    dr = types.ModuleType("nvdiffrast.torch")
    dr.RasterizeCudaContext, dr.rasterize, dr.interpolate = FakeRasterizeContext, fake_rasterize, fake_interpolate
    nvdiffrast.torch = dr
    flex_gemm, ops = types.ModuleType("flex_gemm"), types.ModuleType("flex_gemm.ops")
    grid_sample = types.ModuleType("flex_gemm.ops.grid_sample")
    grid_sample.grid_sample_3d = fake_grid_sample_3d
    flex_gemm.ops, ops.grid_sample = ops, grid_sample
    for name, module in (("cumesh", cumesh), ("cumesh.remeshing", remeshing), ("nvdiffrast", nvdiffrast),
                         ("nvdiffrast.torch", dr), ("flex_gemm", flex_gemm), ("flex_gemm.ops", ops),
                         ("flex_gemm.ops.grid_sample", grid_sample)):
        monkeypatch.setitem(sys.modules, name, module)
    # to_glb has no joined islands, so the steps are compared with it without them (the join has tests of its own:
    # test_the_unwrap_joins_small_islands_and_packs_the_atlas_again, and tests/bake/test_uv_islands.py).
    monkeypatch.setenv("PIXAL3D_JOIN_ISLANDS", "off")
    LOG.clear()
    CPU_ONLY[0] = False
    yield Fakes(cumesh=cumesh, remeshing=remeshing)
    LOG.clear()
    CPU_ONLY[0] = False


@pytest.fixture(scope="module")
def staged_types() -> Any:
    """The fork's types.py (FinishSteps, PropFailed), read from its file: importing pixal3d.staged imports
    torch and the whole fork."""
    if not TYPES_PY.is_file():
        pytest.skip("the fork's pixal3d/staged/types.py is missing")
    spec = importlib.util.spec_from_file_location("staged_types_under_test", TYPES_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses look their module up
    spec.loader.exec_module(module)
    if not hasattr(module, "FinishSteps"):
        pytest.skip("the fork's types.py has no FinishSteps yet")
    return module


def postprocess_module() -> Any:
    """o_voxel's postprocess.py, loaded over the stand-ins (the fakes fixture must be active)."""
    spec = importlib.util.spec_from_file_location("o_voxel_postprocess_under_test", POSTPROCESS_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- a decoded prop -------------------------------------------------------------------------------------------------

def decoded() -> types.SimpleNamespace:
    """A decoded prop as the engine hands it over (DecodedMesh): a closed sphere-like mesh and its texture voxels,
    all on the CPU."""
    rings, segments, radius = 5, 8, 0.4
    vertices = [[0.0, 0.0, radius]]
    for r in range(1, rings):
        theta = np.pi * r / rings
        for s in range(segments):
            phi = 2 * np.pi * s / segments
            vertices.append([radius * np.sin(theta) * np.cos(phi) + 0.01 * s, radius * np.sin(theta) * np.sin(phi),
                             radius * np.cos(theta)])
    vertices.append([0.0, 0.0, -radius])
    faces = []
    ring = lambda r, s: 1 + (r - 1) * segments + s % segments  # noqa: E731
    for s in range(segments):
        faces.append([0, ring(1, s), ring(1, s + 1)])
        for r in range(1, rings - 1):
            faces += [[ring(r, s), ring(r + 1, s), ring(r + 1, s + 1)],
                      [ring(r, s), ring(r + 1, s + 1), ring(r, s + 1)]]
        faces.append([len(vertices) - 1, ring(rings - 1, s + 1), ring(rings - 1, s)])
    generator = torch.Generator().manual_seed(5)
    coords = torch.randint(100, 924, (64, 3), generator=generator, dtype=torch.int32)
    attrs = torch.rand((64, 6), generator=generator, dtype=torch.float32)
    return types.SimpleNamespace(vertices=torch.tensor(vertices, dtype=torch.float32),
                                 faces=torch.tensor(faces, dtype=torch.int32), coords=coords, attrs=attrs,
                                 resolution=RESOLUTION, attr_layout=LAYOUT)


class Px:
    """What finish_steps needs of app.py: the progress function (route_progress replaces it on the module)."""

    def __init__(self) -> None:
        self.reports: list[tuple[str, int, int]] = []

    def _update_progress(self, stage: str, step: int, total: int) -> None:
        self.reports.append((stage, step, total))


def streak(answer: str | None = None) -> Any:
    def check(glb: Any) -> str | None:
        check.seen.append(np.asarray(glb.vertices).copy())     # type: ignore[attr-defined]
        return answer
    check.seen = []     # type: ignore[attr-defined]
    return check


def make_steps(staged_types: Any, out: Path, *, px: Px | None = None, inline: bool = False, multi: bool = False,
               views: int = 1, blocked: bool = False, answer: str | None = None, ss_retries: int = 0,
               work_folder: Path | None = None) -> Any:
    return glbmod.finish_steps(px or Px(), decoded(), LAYOUT, RESOLUTION, DECIMATION, TEXTURE, str(out), multi,
                               ss_retries, blocked, ROTATION, streak(answer), views=views, inline=inline,
                               types=staged_types, work_folder=work_folder)


@contextlib.contextmanager
def cpu_only() -> Iterator[None]:
    CPU_ONLY[0] = True
    try:
        with NoCardMode():
            yield
    finally:
        CPU_ONLY[0] = False


@contextlib.contextmanager
def cpu_step_holding() -> Iterator[None]:
    """A CPU step the engine gave room on the card: it may make a card object (its hold) but move no tensor."""
    with NoCardMode():
        yield


def run_step(kind: str, fn: Any, state: dict) -> None:
    """One step where the engine would run it: a GPU step with the card, a CPU step with no card at all."""
    if kind == "gpu":
        with CardMode():
            fn(state)
    else:
        with cpu_only():
            fn(state)


def cloned(state: dict) -> dict:
    """A copy of a state to hold a later one against; the texture volume and the BVH (objects the steps only read)
    are the same objects."""
    return {key: value.clone() if torch.is_tensor(value)
            else value if isinstance(value, (glbmod.TextureVoxels, FakeHostBVH, FakeBVH)) else copy.deepcopy(value)
            for key, value in state.items()}


def same(a: Any, b: Any) -> bool:
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    return a == b


# ---- the stand-ins themselves ----------------------------------------------------------------------------------------

def test_the_card_stand_in_behaves_as_cuda_where_it_matters() -> None:
    with CardMode():
        on = torch.ones(3, 2).cuda()
        made = torch.zeros(2, device="cuda")
        assert type(on) is Card and type(made) is Card and type(on * 2 + 1) is Card and on.is_cuda
        assert type(on.cpu()) is torch.Tensor and type(on[torch.tensor([0, 2])]) is Card
        with pytest.raises(TypeError):
            on.numpy()
        with pytest.raises(RuntimeError, match="same device"):
            on + torch.ones(3, 2)
        with pytest.raises(AssertionError, match="CPU step"):
            with cpu_only():
                torch.ones(2).cuda()


# ---- the steps ------------------------------------------------------------------------------------------------------

@needs_sources
def test_five_steps_alternate_between_cpu_and_card(fakes: Fakes, staged_types: Any, tmp_path: Path) -> None:
    steps = make_steps(staged_types, tmp_path / "prop" / "mesh" / "mesh.glb")
    assert isinstance(steps, staged_types.FinishSteps)
    assert [kind for kind, _ in steps.steps] == ["cpu", "gpu", "cpu", "gpu", "cpu"]
    assert [fn.__name__ for _, fn in steps.steps] == ["bvh", "geometry", "unwrap", "bake", "export"]
    assert steps.native_mib == 0 and steps.safety_mib is None          # the engine's own figures (glb_reserve)
    assert steps.holds_mib == 0 and steps.release is not None          # a few faces: no MiB to speak of
    assert sorted(steps.state) == sorted(glbmod.DECODED + glbmod.VOXELS)
    assert not card_things(steps.state) and all(steps.state[k].device.type == "cpu" for k in glbmod.DECODED)
    voxels = steps.state["voxels"]                                      # no work folder (a check): kept in RAM
    assert isinstance(voxels, glbmod.TextureVoxels) and voxels.path is None and not voxels.on_disk


@needs_sources
def test_each_step_reads_and_writes_the_state_it_says(fakes: Fakes, staged_types: Any, tmp_path: Path) -> None:
    (tmp_path / "prop").mkdir()
    steps = make_steps(staged_types, tmp_path / "prop" / "mesh" / "mesh.glb")
    state = steps.state
    expected = [
        (glbmod.DECODED, glbmod.DECODED + glbmod.VOXELS),             # bvh without room: nothing
        (glbmod.DECODED + glbmod.VOXELS, glbmod.VOXELS + glbmod.SOURCE + glbmod.CHARTS),
        (glbmod.VOXELS + glbmod.SOURCE + glbmod.CHARTS, glbmod.VOXELS + glbmod.SOURCE + glbmod.CHARTS
         + glbmod.UNWRAPPED),
        (glbmod.VOXELS + glbmod.SOURCE + glbmod.UNWRAPPED, glbmod.VOXELS + glbmod.SOURCE + glbmod.UNWRAPPED
         + glbmod.BAKED),
        (glbmod.UNWRAPPED + glbmod.BAKED, glbmod.UNWRAPPED + glbmod.BAKED + ("result",)),
    ]
    for (kind, fn), (reads, after) in zip(steps.steps, expected):
        assert set(reads) <= set(state), f"{fn.__name__} would miss {set(reads) - set(state)}"
        if fn.__name__ == "geometry":
            after = glbmod.DECODED + after          # a GPU step keeps its inputs; unwrap throws them away
        run_step(kind, fn, state)
        assert sorted(state) == sorted(set(after)), fn.__name__
        assert not card_things(state), f"{fn.__name__} left card tensors in the state"
    assert state["out_faces"].dtype == torch.int32 and state["out_uvs"].dtype == torch.float32
    assert state["mask"].dtype == np.bool_ and state["mask"].shape == (TEXTURE, TEXTURE)
    assert state["texels"].shape == (TEXTURE, TEXTURE, 6) and 0 < state["mask"].sum() < TEXTURE * TEXTURE
    assert (tmp_path / "prop" / "mesh" / "mesh.glb").is_file()


@needs_sources
def test_gpu_steps_leave_nothing_on_the_card(fakes: Fakes, staged_types: Any, tmp_path: Path) -> None:
    steps = make_steps(staged_types, tmp_path / "out" / "mesh.glb")
    assert not on_card_now()
    for kind, fn in steps.steps:
        run_step(kind, fn, steps.state)
        assert not on_card_now(), f"after {fn.__name__}: {[type(o).__name__ for o in on_card_now()][:5]}"


@needs_sources
def test_geometry_keeps_its_bvh_for_the_bake_when_the_engine_gives_it_room(fakes: Fakes, staged_types: Any,
                                                                         tmp_path: Path) -> None:
    """The engine writes state["room_mib"] before every step. With room, geometry hands its BVH of the decoded
    mesh on to the bake (HELD: the object and the MiB it takes on the card) instead of the bake building the same
    one again; the unwrap leaves it alone, the bake uses it up, release() lets go of it at once. Without room, as
    before: nothing stays on the card between the steps."""
    def builds() -> int:
        return len([entry for entry in LOG if entry[0] == "cuBVH"])

    out = tmp_path / "out" / "mesh.glb"
    steps = make_steps(staged_types, out)
    assert steps.release is not None
    state = steps.state
    state["room_mib"] = 1 << 20
    with cpu_step_holding():
        steps.steps[0][1](state)                            # bvh, on the worker: builds and holds
    assert isinstance(state.get("bvh"), FakeBVH) and isinstance(state.get("card_mib"), int)
    assert state["card_mib"] >= 0                           # the fake mesh is a few faces: no MiB to speak of
    assert state["bvh"] in on_card_now()
    made = builds()
    run_step(*steps.steps[1], state)                        # geometry: the bvh step's, kept for the bake
    assert isinstance(state.get("bvh"), FakeBVH) and "card_mib" in state and builds() == made
    run_step(*steps.steps[2], state)                        # unwrap: a CPU step, the hold untouched
    assert "bvh" in state and "card_mib" in state
    state["room_mib"] = 0                                   # what the engine says now does not matter to the bake
    run_step(*steps.steps[3], state)                        # bake: the held BVH, used up
    assert "bvh" not in state and "card_mib" not in state
    assert builds() == made, "the bake built the BVH again"
    assert not on_card_now()
    run_step(*steps.steps[4], state)
    assert out.is_file()

    steps = make_steps(staged_types, tmp_path / "again" / "mesh.glb")
    state = steps.state
    state["room_mib"] = 1 << 20
    with cpu_step_holding():
        steps.steps[0][1](state)
    steps.release(state)                                    # the engine: a decode ran out of memory
    assert "bvh" not in state and "card_mib" not in state and not on_card_now()
    run_step(*steps.steps[1], state)                        # geometry builds its own, and keeps it (room)
    assert isinstance(state.get("bvh"), FakeBVH) and builds() == made + 2
    run_step(*steps.steps[2], state)
    run_step(*steps.steps[3], state)
    assert builds() == made + 2 and not on_card_now()

    steps = make_steps(staged_types, tmp_path / "small" / "mesh.glb")
    state = steps.state
    state["room_mib"] = 0                                   # too little room: nothing is held
    run_step(*steps.steps[0], state)
    assert "bvh" not in state and "card_mib" not in state and not on_card_now()
    run_step(*steps.steps[1], state)
    assert "bvh" not in state and "card_mib" not in state and not on_card_now()
    run_step(*steps.steps[2], state)
    run_step(*steps.steps[3], state)                        # builds the BVH again, as to_glb does
    assert builds() == made + 4 and not on_card_now()


@needs_sources
def test_with_host_trees_the_bvh_is_built_once_in_ram_and_uploaded_by_each_step(fakes: Fakes, staged_types: Any,
                                                                               tmp_path: Path) -> None:
    """With Spellcast3D's cumesh the bvh step builds the tree on the worker without the card, whatever room the
    engine offers (nothing is held between the steps: holds_mib is 0); geometry and bake upload it for their own
    queries and let the card copy go when they return; the bake uses the tree up; a step run again (after an
    out-of-memory) uploads it again and gives the same state; release() leaves the tree in RAM."""
    def count(what: str) -> int:
        return len([entry for entry in LOG if entry[0] == what])

    with host_trees(fakes):
        out = tmp_path / "out" / "mesh.glb"
        steps = make_steps(staged_types, out)
        assert steps.holds_mib == 0 and steps.release is not None
        state = steps.state
        state["room_mib"] = 0                                   # no room at all: the worker still builds it
        run_step(*steps.steps[0], state)                        # bvh, a CPU step: in RAM alone
        tree = state.get("bvh")
        assert isinstance(tree, FakeHostBVH) and not tree.on_device and "card_mib" not in state
        assert not on_card_now() and count("cuBVH") == 1 and count("cuBVH.upload") == 0
        steps.release(state)                                    # the engine: a decode ran out of memory
        assert state.get("bvh") is tree and not on_card_now()   # nothing on the card to let go of; the tree stays
        run_step(*steps.steps[1], state)                        # geometry: uploads, remeshes, lets the card copy go
        assert state.get("bvh") is tree and not tree.on_device and not on_card_now()
        assert count("cuBVH") == 1 and count("cuBVH.upload") == 1 and count("cuBVH.release_device") == 1
        first = cloned({k: v for k, v in state.items() if k != "bvh"})
        run_step(*steps.steps[1], state)                        # again, as after an out-of-memory
        assert state.get("bvh") is tree and not tree.on_device and not on_card_now()
        assert all(same(state[k], first[k]) for k in first) and count("cuBVH") == 1 and count("cuBVH.upload") == 2
        run_step(*steps.steps[2], state)                        # unwrap, a CPU step: the tree untouched
        assert state.get("bvh") is tree and not tree.on_device
        run_step(*steps.steps[3], state)                        # bake: uploads the same tree, uses it up
        assert "bvh" not in state and "card_mib" not in state and not on_card_now()
        assert count("cuBVH") == 1 and count("cuBVH.upload") == 3 and count("cuBVH.release_device") == 3
        run_step(*steps.steps[4], state)
        assert out.is_file()


def test_a_bake_that_runs_out_of_memory_keeps_its_host_tree_for_the_retry(fakes: Fakes, staged_types: Any,
                                                                          tmp_path: Path) -> None:
    """With host trees, a bake attempt that runs out of memory (the engine then runs the step again, with what it
    kept on the card gone) leaves the tree in RAM in the state: the retry uploads the same tree instead of building
    a new one on the GPU thread (9-13 s for a 15-21 M-face mesh), and the bake's state is that of a bake that never
    failed; only a bake that succeeded uses the tree up."""
    def count(what: str) -> int:
        return len([entry for entry in LOG if entry[0] == what])

    with host_trees(fakes):
        states = []
        for fail in (False, True):
            steps = make_steps(staged_types, tmp_path / ("fail" if fail else "plain") / "mesh.glb")
            state = steps.state
            state["room_mib"] = 0
            for kind, fn in steps.steps[:3]:
                run_step(kind, fn, state)
            tree = state["bvh"]
            built = count("cuBVH")
            if fail:
                sample = sys.modules["flex_gemm.ops.grid_sample"].grid_sample_3d

                def once(*args: Any, **kwargs: Any) -> Any:
                    grid_sample.grid_sample_3d = sample
                    raise torch.OutOfMemoryError("CUDA out of memory (a stand-in)")

                grid_sample = sys.modules["flex_gemm.ops.grid_sample"]
                grid_sample.grid_sample_3d = once
                with pytest.raises(torch.OutOfMemoryError):
                    run_step(*steps.steps[3], state)
                assert state.get("bvh") is tree and not tree.on_device and not on_card_now()
            run_step(*steps.steps[3], state)                    # the bake (again, after the failure)
            assert count("cuBVH") == built                      # no tree built anew
            assert "bvh" not in state and "card_mib" not in state and not on_card_now()
            states.append({key: state[key] for key in ("mask", "texels")})
        plain, failed = states
        assert all(same(plain[key], failed[key]) for key in plain)


def test_the_bake_reuses_flex_gemms_tuning_for_a_new_texel_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """flex_gemm keys its grid_sample tuning by the exact number of query points (a prop's valid texels): a new
    count gets the config tuned for the nearest count of the same table size (log2 of the rows), channels and
    corners, so the kernel does not tune 12 configs again (any config gives the same bytes); a count it knows,
    or one with nothing of its shape tuned, is left alone, and so is a flex_gemm laid out otherwise."""
    tail = ("torch.float32", "torch.uint32", "torch.float32", "torch.float32")
    near, far, other = object(), object(), object()
    kernel = types.SimpleNamespace(cache={str((21, 2_500_000, 6, 8, *tail)): near,
                                          str((21, 1_000_000, 6, 8, *tail)): far,
                                          str((22, 2_520_000, 6, 8, *tail)): other})
    module = types.ModuleType(glbmod.GRID_SAMPLE_KERNEL)
    module.indice_weighed_sum_fwd_kernel = kernel
    monkeypatch.setitem(sys.modules, glbmod.GRID_SAMPLE_KERNEL, module)
    rows = 3_000_000                                            # log2 -> 21
    assert glbmod.reuse_grid_sample_tuning(rows, 6, 2_510_000)
    assert kernel.cache[str((21, 2_510_000, 6, 8, *tail))] is near
    assert glbmod.reuse_grid_sample_tuning(rows, 6, 2_510_000) and len(kernel.cache) == 4    # known now
    assert not glbmod.reuse_grid_sample_tuning(rows, 4, 2_510_000)                          # no 4-channel tuning
    assert not glbmod.reuse_grid_sample_tuning(1 << 23, 6, 2_510_000) and len(kernel.cache) == 4
    monkeypatch.setitem(sys.modules, glbmod.GRID_SAMPLE_KERNEL, types.ModuleType("empty"))
    assert not glbmod.reuse_grid_sample_tuning(rows, 6, 2_600_000)


@needs_sources
def test_host_trees_make_the_same_file(fakes: Fakes, staged_types: Any, tmp_path: Path) -> None:
    """The file, the answer and the bar are those of the steps with a cuBVH that uploads at construction; only the
    tree is built once instead of twice."""
    plain = steps_run(staged_types, tmp_path / "plain")
    with host_trees(fakes):
        host = steps_run(staged_types, tmp_path / "host")
    assert host.file == plain.file and host.result == plain.result and host.bar == plain.bar
    assert [entry[0] for entry in plain.log].count("cuBVH") == 2
    assert [entry[0] for entry in host.log].count("cuBVH") == 1
    assert [entry[0] for entry in host.log].count("cuBVH.upload") == 2
    kept = [entry for entry in host.log if entry[0] not in ("cuBVH", "cuBVH.upload", "cuBVH.release_device")]
    assert kept == [entry for entry in plain.log if entry[0] != "cuBVH"]


def test_the_texture_volume_waits_in_the_work_folder_and_comes_back_the_same(tmp_path: Path) -> None:
    """TextureVoxels with the prop's work folder: a thread of its own writes the volume there and lets go of it in RAM;
    take() reads back the same bytes (dtype, shape, values; bfloat16 too), on another thread when prefetched, and
    keeps them for a second take; the file goes with the object. Without a folder, or with one that is gone, or
    when the file cannot be written, it stays in RAM."""
    generator = torch.Generator().manual_seed(3)
    coords = torch.randint(0, 1536, (1000, 3), generator=generator, dtype=torch.int32)
    attrs = torch.rand((1000, 6), generator=generator)
    folder = tmp_path / "work" / "p_a"
    folder.mkdir(parents=True)
    voxels = glbmod.TextureVoxels(coords.clone(), attrs.clone(), folder)
    voxels._written.result()
    path = folder / glbmod.VOXEL_FILE
    assert voxels.on_disk and voxels._tensors is None and path.is_file()
    assert sorted(p.name for p in folder.iterdir()) == [glbmod.VOXEL_FILE]      # the .tmp became the file
    voxels.prefetch()
    assert voxels._read is not None
    back_coords, back_attrs = voxels.take()
    assert same(back_coords, coords) and same(back_attrs, attrs)
    assert voxels.take()[0] is back_coords                  # kept for a bake run again
    from safetensors.torch import load_file                 # a safetensors file any tool reads
    assert same(load_file(str(path))["attrs"], attrs)
    del voxels, back_coords, back_attrs
    gc.collect()
    assert not path.exists()                                # the file goes with the object

    half = torch.rand((10, 4), generator=generator).to(torch.bfloat16)
    glbmod.write_tensors(tmp_path / "b.safetensors", {"h": half, "c": coords})
    found = glbmod.read_tensors(tmp_path / "b.safetensors")
    assert same(found["h"], half) and same(found["c"], coords)

    kept = glbmod.TextureVoxels(coords, attrs)               # no folder: the reference path, a check
    assert kept.path is None and not kept.on_disk and kept.take() == (coords, attrs)
    gone = glbmod.TextureVoxels(coords, attrs, tmp_path / "gone")
    assert gone.path is None and gone.take()[0] is coords
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / (glbmod.VOXEL_FILE + ".tmp")).mkdir()         # the file cannot be written
    failed = glbmod.TextureVoxels(coords, attrs, blocked)
    failed._written.result()
    assert not failed.on_disk and failed.take()[1] is attrs


@needs_sources
def test_steps_with_a_work_folder_make_the_same_file_and_leave_nothing_there(fakes: Fakes, staged_types: Any,
                                                                             tmp_path: Path) -> None:
    """The five steps with the prop's work folder: the volume waits there from the hand-over to the bake (written
    and read on threads of their own, never on the steps' own threads), the GLB is the one made with it in RAM, and the
    file is gone once the export let go of it - or once the engine cleared the state of a prop that failed."""
    plain = tmp_path / "plain" / "mesh.glb"
    with CardMode():
        glbmod.run_all(make_steps(staged_types, plain))
    folder = tmp_path / "work" / "p_a"
    folder.mkdir(parents=True)
    out = tmp_path / "spilled" / "mesh.glb"
    steps = make_steps(staged_types, out, work_folder=folder)
    voxels = steps.state["voxels"]
    voxels._written.result()
    assert voxels.on_disk and (folder / glbmod.VOXEL_FILE).is_file()
    seen: list[str] = []
    real_read = glbmod.read_tensors

    def read(path: Path) -> Any:
        seen.append(threading.current_thread().name)
        return real_read(path)

    glbmod.read_tensors, saved = read, glbmod.read_tensors
    try:
        for kind, fn in steps.steps:
            run_step(kind, fn, steps.state)
    finally:
        glbmod.read_tensors = saved
    assert seen and all(name.startswith("spellcast-glb-read") for name in seen)
    del voxels
    gc.collect()
    assert out.read_bytes() == plain.read_bytes()
    assert list(folder.iterdir()) == []

    steps = make_steps(staged_types, tmp_path / "failed" / "mesh.glb", work_folder=folder)
    steps.state["voxels"]._written.result()
    assert (folder / glbmod.VOXEL_FILE).is_file()
    steps.state.clear()                                      # the engine, after a step failed
    gc.collect()
    assert list(folder.iterdir()) == []


@needs_sources
def test_a_gpu_step_run_again_gives_the_same_state(fakes: Fakes, staged_types: Any, tmp_path: Path) -> None:
    """After an out-of-memory the engine runs a GPU step again on the same state."""
    steps = make_steps(staged_types, tmp_path / "out" / "mesh.glb")
    state = steps.state
    for kind, fn in steps.steps:
        run_step(kind, fn, state)
        if kind != "gpu":
            continue
        first, log = cloned(state), len(LOG)
        run_step(kind, fn, state)
        assert sorted(state) == sorted(first) and all(same(state[k], first[k]) for k in first), fn.__name__
        assert len(LOG) > log                       # it did run again


def to_glb_reference(folder: Path) -> types.SimpleNamespace:
    """o_voxel's to_glb with the launcher's arguments, then turned, checked and saved as decoded_to_glb and
    save_glb do (a multi-view prop from 3 views, 2 redraws, a streak found): the file, the calls into the native
    code, the bar (through app.py's tqdm interceptor) and save_glb's answer with the resolution."""
    postprocess = postprocess_module()
    bar: list[tuple[str, int, int]] = []

    class Bar(postprocess.tqdm):        # app.py's _TqdmProgressInterceptor
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._stage = kwargs.get("desc", "Processing")
            super().__init__(*args, **kwargs)

        def set_description(self, desc: Any = None, refresh: bool = True) -> None:
            self._stage = desc or "Processing"
            super().set_description(desc, refresh)

        def update(self, n: int = 1) -> None:
            super().update(n)
            bar.append((self._stage, self.n, self.total or 0))

    postprocess.tqdm = Bar
    LOG.clear()
    mesh = decoded()
    out = folder / "reference" / "mesh" / "mesh.glb"
    out.parent.parent.mkdir(parents=True)
    check = streak("streaked")
    with CardMode():
        glb = postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords, attr_layout=LAYOUT,
            grid_size=torch.tensor([RESOLUTION] * 3, dtype=torch.int32, device="cuda"),
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=DECIMATION, texture_size=TEXTURE,
            remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True)
        del mesh
    glb.apply_transform(np.array(ROTATION, dtype=np.float64))
    warning = check(glb) or None
    bar.append((glbmod.SAVING, 0, 1))
    mv.make_folder(out.parent)
    glb.export(str(out), extension_webp=True)
    log = list(LOG)
    LOG.clear()
    return types.SimpleNamespace(file=out.read_bytes(), log=log, bar=bar, seen=check.seen,
                                 result={"mode": "multi", "views": 3, "warning": warning, "ss_retries": 2,
                                         "resolution": RESOLUTION})


def steps_run(staged_types: Any, folder: Path) -> types.SimpleNamespace:
    """The same prop through run_all of the five steps."""
    LOG.clear()
    out = folder / "steps" / "mesh" / "mesh.glb"
    out.parent.parent.mkdir(parents=True)
    px = Px()
    steps = make_steps(staged_types, out, px=px, inline=True, multi=True, views=3, answer="streaked", ss_retries=2)
    with CardMode():
        result = glbmod.run_all(steps)
    log = list(LOG)
    LOG.clear()
    return types.SimpleNamespace(file=out.read_bytes(), log=log, bar=px.reports, result=result, state=steps.state,
                                 seen=steps.steps[3][1].__self__.streak_warning.seen)


@needs_sources
def test_the_steps_make_to_glbs_file_byte_for_byte(fakes: Fakes, staged_types: Any, tmp_path: Path) -> None:
    """to_glb and its launcher around it against run_all of the steps: the same file, the same calls into the
    native code in the same order (the one move: the vertex normals are read before xatlas runs instead of after;
    the one addition: the simplified mesh's holes filled once more, fill_after_remesh), the same bar, the same
    answer."""
    reference = to_glb_reference(tmp_path)
    got = steps_run(staged_types, tmp_path)

    assert got.result == reference.result
    assert got.state == {}                                               # cleared, as the engine clears it
    assert got.file == reference.file
    normals = [("mesh.compute_vertex_normals",), ("mesh.read_vertex_normals",)]
    at = reference.log.index(normals[0])
    assert reference.log[at:at + 2] == normals and reference.log.index(("xatlas.Atlas",)) < at
    moved = reference.log[:at] + reference.log[at + 2:]
    after_charts = moved.index(("mesh.read_atlas_charts",)) + 1
    moved[after_charts:after_charts] = normals
    after_unify = moved.index(("mesh.unify_face_orientations",)) + 1
    moved[after_unify:after_unify] = [("mesh.fill_holes", glbmod.FILL_AFTER_REMESH_PERIMETER)]
    assert got.log == moved
    assert [entry[0] for entry in got.log].count("cuBVH") == 2
    remesh = [entry for entry in got.log if entry[0] == "remesh"]
    assert len(remesh) == 1 and remesh[0][3:] == ((0.0, 0.0, 0.0), (512 + 3 * 1) / 512 * 1.0, 512, 1, 0, False)
    assert ("mesh.compute_charts", np.radians(90.0), 0.0, 1.0, 1.0, 0.1, 0.0001) in got.log
    assert ("xatlas.compute_charts", (("verbose", False),)) in got.log
    assert got.bar == reference.bar and [stage for stage, _, _ in got.bar] == [*glbmod.BARS, glbmod.SAVING]
    assert len(got.seen) == len(reference.seen) == 1 and np.array_equal(got.seen[0], reference.seen[0])


class _HoleMesh:
    """cumesh.CuMesh where it matters to fill_after_remesh: a square of two faces with a hole beside it, which
    fill_holes closes with a fan around a new vertex put anywhere (as cumesh leaves it), wound against the square."""

    def __init__(self, log: list) -> None:
        self.log = log
        self.vertices = self.faces = None

    def init(self, vertices: Any, faces: Any) -> None:
        self.log.append("init")
        self.vertices, self.faces = vertices.clone(), faces.clone()

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    def read(self) -> tuple[Any, Any]:
        return self.vertices.clone(), self.faces.clone()

    def fill_holes(self, max_hole_perimeter: float) -> None:
        self.log.append("fill_holes")
        if self.num_vertices == 5:
            self.vertices = torch.cat([self.vertices, torch.tensor([[9.0, 9.0, 9.0]])])
            self.faces = torch.cat([self.faces, torch.tensor([[1, 4, 5], [4, 2, 5], [2, 1, 5]], dtype=torch.int32)])

    def unify_face_orientations(self) -> None:
        self.log.append("unify_face_orientations")


def test_the_holes_closed_after_the_remesh_are_oriented_with_the_rest() -> None:
    """The fans fill_holes adds after the remesh's own unify_face_orientations come in its winding, often against
    their neighbours: the mesh made of the filled one goes through unify_face_orientations once more. A mesh without
    holes to close is left as the remesh made it."""
    log: list = []
    cumesh = types.SimpleNamespace(CuMesh=lambda: _HoleMesh(log))
    square = _HoleMesh(log)
    square.init(torch.tensor([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0.5, 0]]),
                torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32))
    log.clear()
    filled = glbmod.fill_after_remesh(cumesh, square)
    assert filled is not square and log == ["fill_holes", "init", "unify_face_orientations"]
    assert torch.allclose(filled.vertices[5], (filled.vertices[1] + filled.vertices[2] + filled.vertices[4]) / 3)
    log.clear()
    assert glbmod.fill_after_remesh(cumesh, filled) is filled and log == ["fill_holes"]


def _atlas_with_a_crumb() -> tuple[Any, Any, Any, Any]:
    """(positions, vmap, faces, uvs) as the unwrap's xatlas leaves them: a flat 40 x 40 grid that is one island, and
    a sliver folded back over its border that is an island of its own, far away in the atlas."""
    side = 40
    xs, ys = np.meshgrid(np.arange(side + 1), np.arange(side + 1), indexing="ij")
    positions = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    faces = []
    for i in range(side):
        for j in range(side):
            a, b, d = i * (side + 1) + j, (i + 1) * (side + 1) + j, i * (side + 1) + j + 1
            faces += [[a, b, b + 1], [a, b + 1, d]]
    a, b = 0, side + 1                                      # the vertices at (0, 0) and (1, 0)
    positions = np.vstack([positions, [[0.5, 0.3, 0.05]]])
    vmap = np.concatenate([np.arange(len(positions) - 1), [b, a, len(positions) - 1]])
    uvs = np.vstack([positions[:-1, :2] / side, [[5.0, 5.0], [5.2, 5.0], [5.1, 5.1]]])
    faces.append([len(vmap) - 3, len(vmap) - 2, len(vmap) - 1])
    return (torch.tensor(positions, dtype=torch.float32), torch.tensor(vmap, dtype=torch.int32),
            torch.tensor(faces, dtype=torch.int32), torch.tensor(uvs, dtype=torch.float32))


def test_the_unwrap_joins_small_islands_and_packs_the_atlas_again(fakes: Fakes,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """join_islands: the sliver laid onto the grid's island (uv_islands), and the atlas packed again by xatlas from
    those UVs; the same triangles, in the unwrap's dtypes. Off (PIXAL3D_JOIN_ISLANDS), xatlas's atlas as it was."""
    import uv_islands

    positions, vmap, faces, uvs = _atlas_with_a_crumb()
    assert int(uv_islands.chart_labels(faces.numpy(), len(vmap)).max()) + 1 == 2
    same = glbmod.join_islands(fakes.cumesh, positions, vmap, faces, uvs)                  # off (the fixture's)
    assert all(got is given for got, given in zip(same, (vmap, faces, uvs))) and LOG == []

    monkeypatch.setenv("PIXAL3D_JOIN_ISLANDS", "on")
    got_vmap, got_faces, got_uvs = glbmod.join_islands(fakes.cumesh, positions, vmap, faces, uvs)
    assert LOG[-2:] == [("xatlas.compute_charts", (("use_input_mesh_uvs", True),)), ("xatlas.pack_charts", ())]
    assert (got_vmap.dtype, got_faces.dtype, got_uvs.dtype) == (torch.int32, torch.int32, torch.float32)
    assert torch.equal(got_vmap[got_faces.long()], vmap[faces.long()])
    assert int(uv_islands.chart_labels(got_faces.numpy(), len(got_vmap)).max()) + 1 == 1
    assert len(got_vmap) == len(positions)                  # the sliver's corners: two of the grid's, one new

    # Two islands after the join: xatlas gets a mesh per set of islands (uv_islands.chart_groups) and the
    # triangles come back in their order.
    far = torch.tensor([[60.0, 60.0, 0.0], [61.0, 60.0, 0.0], [60.0, 61.0, 0.0]])
    more_positions = torch.cat([positions, far])
    more_vmap = torch.cat([vmap, torch.arange(len(positions), len(positions) + 3, dtype=torch.int32)])
    more_faces = torch.cat([faces[:5], torch.tensor([[len(vmap), len(vmap) + 1, len(vmap) + 2]], dtype=torch.int32),
                            faces[5:]])
    more_uvs = torch.cat([uvs, torch.tensor([[8.0, 8.0], [9.0, 8.0], [8.0, 9.0]])])
    LOG.clear()
    got_vmap, got_faces, got_uvs = glbmod.join_islands(fakes.cumesh, more_positions, more_vmap, more_faces, more_uvs)
    added = [entry for entry in LOG if entry[0] == "xatlas.add_mesh"]
    assert [entry[2] for entry in added] == [(len(faces), 3), (1, 3)]         # the grid with its sliver; the far one
    assert (got_vmap.dtype, got_faces.dtype, got_uvs.dtype) == (torch.int32, torch.int32, torch.float32)
    assert torch.equal(got_vmap[got_faces.long()], more_vmap[more_faces.long()])
    assert int(uv_islands.chart_labels(got_faces.numpy(), len(got_vmap)).max()) + 1 == 2
    assert got_uvs.shape == (len(got_vmap), 2) and len(got_vmap) == len(more_positions)

    class Refuses(FakeAtlas):
        def pack_charts(self, **options: Any) -> None:
            raise RuntimeError("xatlas: no room")

    monkeypatch.setattr(fakes.cumesh, "Atlas", Refuses)     # the join never costs the mesh: xatlas's atlas stays
    kept = glbmod.join_islands(fakes.cumesh, positions, vmap, faces, uvs)
    assert all(got is given for got, given in zip(kept, (vmap, faces, uvs)))


@needs_sources
def test_the_export_encodes_its_textures_ahead_on_its_threads_into_the_same_file(
        fakes: Fakes, staged_types: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The export fills the base colour channel by channel on its threads (or in one call where the PC has fewer
    than three: EXPORT_THREADS) and encodes both WebPs there while it builds the mesh; trimesh's export writes those
    bytes (encoded_as_webp) instead of encoding again. The file is the same either way (and to_glb's: the test
    above)."""
    made: list[str] = []
    real_webp = glbmod._webp

    def webp(pixels: Any) -> bytes:
        made.append(threading.current_thread().name)
        return real_webp(pixels)

    monkeypatch.setattr(glbmod, "_webp", webp)
    many = steps_run(staged_types, tmp_path / "many")
    assert len(made) == 2 and all(name.startswith("spellcast-glb-export") for name in made)
    monkeypatch.setattr(glbmod, "EXPORT_THREADS", 1)
    one = steps_run(staged_types, tmp_path / "one")
    assert one.file == many.file and len(made) == 4

    from PIL import Image
    import concurrent.futures
    import io
    done: concurrent.futures.Future[bytes] = concurrent.futures.Future()
    done.set_result(b"made ahead")
    picture = glbmod.encoded_as_webp(Image.new("RGB", (4, 4), (9, 8, 7)), done)
    buffer = io.BytesIO()
    picture.save(buffer, format="WEBP")
    assert buffer.getvalue() == b"made ahead"
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")                      # anything else: Pillow's own
    assert buffer.getvalue().startswith(b"\x89PNG")


@needs_sources
def test_texels_rasterized_in_chunks_keep_their_face(fakes: Fakes, staged_types: Any, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """to_glb rasterizes 100,000 faces at a time and adds the chunk's first face to the ids it gets back; with a
    chunk of 7 faces the stand-in rasterizer covers each texel as in one pass, so the file must not change."""
    reference = to_glb_reference(tmp_path)
    monkeypatch.setattr(glbmod, "RASTER_CHUNK", 7)
    got = steps_run(staged_types, tmp_path)
    assert [entry[0] for entry in got.log].count("rasterize") > 1
    assert got.file == reference.file


@needs_sources
def test_the_remesh_is_called_through_the_module_attribute(fakes: Fakes, staged_types: Any, tmp_path: Path) -> None:
    """The launcher's drop_inner_shells replaces cumesh.remeshing.remesh_narrow_band_dc; the steps must meet it."""
    calls = []

    def wrapper(vertices: Any, faces: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("bvh"))
        return fake_remesh(vertices, faces, *args, **kwargs)

    fakes.remeshing.remesh_narrow_band_dc = wrapper
    steps = make_steps(staged_types, tmp_path / "out" / "mesh.glb")
    run_step(*steps.steps[1], steps.state)                 # geometry (the bvh step before it holds nothing here)
    assert len(calls) == 1 and isinstance(calls[0], FakeBVH)


@needs_sources
def test_progress_under_the_engine_comes_from_the_first_step_only(fakes: Fakes, staged_types: Any,
                                                                  tmp_path: Path) -> None:
    """The later steps run while other props are on the card, and the launcher gives a report to the prop on the
    card: only geometry, which runs right after this prop's decode, reports."""
    px = Px()
    steps = make_steps(staged_types, tmp_path / "out" / "mesh.glb", px=px)
    for kind, fn in steps.steps:
        run_step(kind, fn, steps.state)
        if fn.__name__ == "geometry":
            assert px.reports == [(glbmod.BARS[0], 1, 6), (glbmod.BARS[1], 2, 6), (glbmod.BARS[2], 3, 6)]
    assert len(px.reports) == 3


@needs_sources
@pytest.mark.parametrize("multi, views, blocked, answer, warning", [
    (False, 1, False, None, None),
    (False, 1, True, None, guard.STREAK_WARNING),
    (False, 1, True, "streaked", "streaked"),
    (True, 4, False, None, None),
])
def test_the_result_is_what_save_glb_and_the_hook_answer(fakes: Fakes, staged_types: Any, tmp_path: Path,
                                                         multi: bool, views: int, blocked: bool, answer: str | None,
                                                         warning: str | None) -> None:
    out = tmp_path / "prop" / "mesh" / "mesh.glb"
    (tmp_path / "prop").mkdir()
    steps = make_steps(staged_types, out, multi=multi, views=views, blocked=blocked, answer=answer, ss_retries=3)
    check = steps.steps[3][1].__self__.streak_warning
    with CardMode():
        result = glbmod.run_all(steps)
    assert result == {"mode": "multi" if multi else "single", "views": views if multi else 1, "warning": warning,
                      "ss_retries": 3, "resolution": RESOLUTION}
    assert out.is_file() and len(check.seen) == 1
    # the streak check sees the mesh as it is saved: to_glb's axis swap and the launcher's turn cancel out
    import trimesh
    saved = trimesh.load(str(out), force="mesh", process=False)
    assert np.allclose(np.asarray(saved.vertices), check.seen[0], atol=1e-6)


@needs_sources
def test_a_prop_deleted_meanwhile_fails_with_kind_input(fakes: Fakes, staged_types: Any, tmp_path: Path) -> None:
    out = tmp_path / "gone" / "mesh" / "mesh.glb"          # <prop>/mesh/ with <prop> gone
    steps = make_steps(staged_types, out)
    with CardMode(), pytest.raises(staged_types.PropFailed) as failed:
        glbmod.run_all(steps)
    assert failed.value.kind == "input" and failed.value.detail == mv.PROP_DELETED
    assert not (tmp_path / "gone").exists() and steps.state == {}


class StabBVH:
    """cuBVH's two ray-stab queries over a made-up mesh: point p's stab ray r, seeded from s (its index in a
    signed_distance call, or the rng_index given), hits unless a hash of (p, s, r) says otherwise, and a point
    whose coordinate is a whole number lies on the surface (distance 0: never inside)."""

    DIRECTIONS = 32

    @staticmethod
    def _hits(point: Any, seed: int, ray: int) -> bool:
        return hash((tuple(round(float(x), 6) for x in point), int(seed), ray)) % 97 != 0

    def signed_distance(self, positions: Any, mode: str = "watertight") -> tuple[Any, Any, Any]:
        assert mode == "raystab"
        out = []
        for i, p in enumerate(positions):
            distance = 0.0 if float(p[0]).is_integer() else 1.0
            inside = all(self._hits(p, i, r) for r in range(self.DIRECTIONS))
            out.append(-distance if inside else distance)
        return torch.tensor(out), None, None

    def raystab_inside(self, positions: Any, rng_index: Any = None, first_ray: int = 0, last_ray: int = 32,
                       closest: bool = True) -> Any:
        seeds = range(positions.shape[0]) if rng_index is None else rng_index.tolist()
        out = []
        for p, s in zip(positions, seeds):
            hit = all(self._hits(p, s, r) for r in range(first_ray, min(last_ray, self.DIRECTIONS)))
            if hit and closest:
                hit = -(0.0 if float(p[0]).is_integer() else 1.0) < 0
            out.append(hit)
        return torch.tensor(out, dtype=torch.bool)


def test_the_inner_copy_stab_asks_a_subset_with_the_seeds_of_the_whole_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """stab_inside gives each point it is asked about the verdict the old query of every point gives it (queries of
    STAB_QUERY points, each point's rays seeded from its index within its query)."""
    monkeypatch.setattr(glbmod, "STAB_QUERY", 7)
    generator = torch.Generator().manual_seed(3)
    centres = torch.rand(120, 3, generator=generator)
    centres[::11, 0] = 1.0                                          # on the surface
    rest = torch.randperm(120, generator=generator)[:90].sort().values
    asked = torch.randperm(90, generator=generator)[:50].sort().values
    bvh = StabBVH()
    old = torch.cat([bvh.signed_distance(centres[rest[i:i + 7]], mode="raystab")[0] < 0 for i in range(0, 90, 7)])
    assert 0 < int(old.sum()) < 90                                   # both verdicts occur
    assert glbmod.stab_by_index(bvh) and not glbmod.stab_by_index(types.SimpleNamespace(signed_distance=None))
    assert not glbmod.rays_by_hit(bvh) and glbmod.rays_by_hit(types.SimpleNamespace(ray_hits=lambda o, d: None))
    assert torch.equal(glbmod.stab_inside(bvh, centres, rest, asked), old[asked])
    # stab_until: the verdicts of a prefix, slices of 2, 8, 32 ... points, until enough are inside or all are asked
    monkeypatch.setattr(glbmod, "STAB_PROBE", 2)
    for enough in (1, 5, int(old.sum()), int(old.sum()) + 1):
        probed = glbmod.stab_until(bvh, centres, rest, enough)
        assert probed.dtype == torch.bool and torch.equal(probed, old[:probed.shape[0]])
        before = {2: 0, 10: 2, 42: 10, 90: 42}[int(probed.shape[0])]  # whole slices (the last one cut at the end)
        assert int(old[:before].sum()) < enough                       # one slice fewer was not enough...
        assert int(probed.sum()) >= enough or probed.shape[0] == 90   # ...and it stops once it is
    assert glbmod.stab_until(bvh, centres, rest[:0], 1).shape == (0,)


# ---- the inner-copy search over the two kinds of cuBVH (pixal3d_server.find_inner_copy) ----------------------------

def _point_keys(points: Any) -> Any:
    """An integer per point from its coordinates (quantised): what the stand-ins' rays are drawn from."""
    q = (points.double() * 1e5).round().long()
    return q[:, 0] * 73_856_093 + q[:, 1] * 19_349_663 + q[:, 2] * 83_492_791


class OldRaysBVH:
    """A cuBVH without the subset queries: signed_distance(mode="raystab") seeds point i's stab rays from i, its index
    within the call (cuBVH's own rule); a face centre with x == 0.5 lies on the surface (distance 0: never inside).
    ray_trace: a ray hits unless a hash of its origin and direction says otherwise; rays from x > 0.75 mostly escape
    (the wall of a hollow there)."""

    RAYS = 32

    def __init__(self) -> None:
        self.asked = 0          # points stabbed

    def _all_hit(self, points: Any, seeds: Any, first: int, last: int) -> Any:
        keys = _point_keys(points) * 31 + seeds.long() * 1_000_003
        hit = torch.ones(points.shape[0], dtype=torch.bool)
        for ray in range(first, min(last, self.RAYS)):
            hit &= (keys + ray * 7919) % 53 != 0
        return hit

    def signed_distance(self, positions: Any, return_uvw: bool = False, mode: str = "watertight") -> tuple[Any, ...]:
        assert mode == "raystab"
        self.asked += int(positions.shape[0])
        hit = self._all_hit(positions, torch.arange(positions.shape[0]), 0, self.RAYS)
        distance = (positions[:, 0] != 0.5).float()
        return torch.where(hit, -distance, distance), None, None

    def ray_trace(self, origins: Any, directions: Any) -> tuple[Any, Any, Any]:
        keys = _point_keys(origins) + _point_keys(directions) * 5
        hit = torch.where(origins[:, 0] > 0.75, keys % 3 != 0, keys % 61 != 0)
        return None, torch.where(hit, 0, -1), None


class NewRaysBVH(OldRaysBVH):
    """cumesh 0.0.1.post6's cuBVH: raystab_inside for any subset of a query's points, each with the seed given
    (rng_index, int32), and the any-hit ray_hits."""

    def raystab_inside(self, positions: Any, rng_index: Any = None, first_ray: int = 0, last_ray: int = 32,
                       closest: bool = True) -> Any:
        assert rng_index is not None and rng_index.dtype == torch.int32 and rng_index.shape == positions.shape[:1]
        self.asked += int(positions.shape[0])
        hit = self._all_hit(positions, rng_index, first_ray, last_ray)
        return hit & (positions[:, 0] != 0.5) if closest else hit

    def ray_hits(self, origins: Any, directions: Any) -> Any:
        return self.ray_trace(origins, directions)[1] >= 0


def _pieces_by_corners(cumesh: Any, vertices: Any, faces: Any) -> tuple[int, Any]:
    """pixal3d_server._pieces on the CPU: the faces that share a corner, joined."""
    parent = list(range(int(vertices.shape[0])))

    def root(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for face in faces.tolist():
        for corner in face[1:]:
            a, b = root(face[0]), root(corner)
            if a != b:
                parent[b] = a
    roots: dict[int, int] = {}
    piece = [roots.setdefault(root(face[0]), len(roots)) for face in faces.tolist()]
    return len(roots), torch.tensor(piece, dtype=torch.long)


def _remesh_like(seed: int) -> tuple[Any, Any]:
    """A made-up remesh: 1-6 pieces of random triangles side by side along x (either winding, so some pieces have a
    negative volume); one piece in five lies in the plane x == 0.5 (its centres on the surface)."""
    import random

    rnd = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    count = rnd.randint(1, 6)
    vertices: list[Any] = []
    faces: list[list[int]] = []
    for n in range(count):
        corners = torch.rand(rnd.randint(8, 60), 3, generator=generator)
        corners[:, 0] = (n + 0.8 * corners[:, 0]) / count
        if rnd.random() < 0.2:
            corners[:, 0] = 0.5
        first = sum(int(v.shape[0]) for v in vertices)
        faces += [[first + i for i in rnd.sample(range(corners.shape[0]), 3)] for _ in range(rnd.randint(10, 300))]
        vertices.append(corners)
    return torch.cat(vertices), torch.tensor(faces, dtype=torch.int32)


def test_the_inner_copy_search_finds_the_same_copy_with_either_cuBVH(monkeypatch: pytest.MonkeyPatch) -> None:
    """find_inner_copy with a cuBVH that stabs subsets (it stabs slices of `rest` until enough faces are inside, looks
    for the faces seen from outside, then stabs only the faces left that are away from them) removes the faces the
    old cuBVH's path removes (every face of `rest` stabbed in queries of STAB_QUERY points), and looks for the faces
    seen from outside exactly when that path does: never when fewer faces are inside than a piece of the copy has.
    Also when the look fails: the separate pieces go."""
    import random

    import pixal3d_server as server

    monkeypatch.setitem(sys.modules, "cumesh", types.SimpleNamespace(cuBVH=None))
    monkeypatch.setattr(server, "_pieces", _pieces_by_corners)
    monkeypatch.setattr(glbmod, "STAB_QUERY", 7)
    monkeypatch.setattr(glbmod, "STAB_PROBE", 3)
    monkeypatch.setattr(guard, "CAVITY_SAMPLES", 16)
    monkeypatch.setattr(guard, "CAVITY_RAYS", 8)
    look = {"share": 0.0, "fails": False, "calls": 0}

    def seen_from_outside(vertices: Any, faces: Any) -> Any:
        look["calls"] += 1
        if look["fails"]:
            raise RuntimeError("out of memory\nmore")
        return torch.rand(faces.shape[0], generator=torch.Generator().manual_seed(int(faces.shape[0]))) < look["share"]

    monkeypatch.setattr(server, "seen_from_outside", seen_from_outside)
    keys = ("whole", "joined", "plan", "pieces", "hollows", "seeing_out", "seen", "joined_error")
    tally = dict.fromkeys(("skipped", "joined_over_queries", "hollows", "failed", "probe_stopped", "fewer_stabbed"), 0)
    for seed in range(40):
        vertices, faces = _remesh_like(seed)
        monkeypatch.setattr(guard, "INNER_PIECE_MIN_FACES", random.Random(seed).choice([1, 3, 10, 40, 200]))
        for share, fails in ((0.0, False), (0.3, False), (0.9, False), (0.5, True)):
            look.update(share=share, fails=fails)
            results = []
            for kind in (OldRaysBVH, NewRaysBVH):
                bvh = kind()
                look["calls"] = 0
                drop, info = server.find_inner_copy(vertices, faces, (vertices, faces), bvh)
                results.append((drop, info, look["calls"], bvh.asked))
            (old, old_info, old_calls, old_asked), (new, new_info, new_calls, new_asked) = results
            assert torch.equal(old, new), (seed, share, fails, old_info, new_info)
            assert {k: old_info.get(k) for k in keys} == {k: new_info.get(k) for k in keys}, (seed, share, fails)
            assert new_calls == old_calls
            rest = old_info["faces"] - old_info["whole"]
            if old_info["inside"] < guard.INNER_PIECE_MIN_FACES:        # the old path's _joined_copy returns at once
                assert new_calls == 0 and new_info["inside"] == old_info["inside"] and new_info["stabbed"] == rest
                tally["skipped"] += 1
            else:
                assert new_calls == 1 and "inside" not in new_info
            if fails and old_calls:
                assert old_info["joined_error"] == "RuntimeError: out of memory" and old_info["joined"] == 0
                tally["failed"] += 1
            tally["joined_over_queries"] += old_info["joined"] > 0 and rest > glbmod.STAB_QUERY
            tally["hollows"] += bool(old_info.get("hollows"))
            tally["probe_stopped"] += old_calls and new_info["stabbed"] < rest
            tally["fewer_stabbed"] += new_asked < old_asked
    assert all(tally.values()), tally                                   # every branch was taken


def test_the_share_of_rays_that_escape_is_the_same_with_either_cuBVH() -> None:
    import pixal3d_server as server

    points = torch.rand(40, 3, generator=torch.Generator().manual_seed(5))
    old = server.escape_share(OldRaysBVH(), points, 64, torch.Generator().manual_seed(1))
    new = server.escape_share(NewRaysBVH(), points, 64, torch.Generator().manual_seed(1))
    assert torch.equal(old, new) and 0 < float(old.max()) and float(old.min()) < 1
