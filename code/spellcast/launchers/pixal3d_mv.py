"""Multi-view generation for pixal3d_server.py: several labelled pictures of one prop -> one 3D model.

Pixal3D's multi-view pipeline (pixal3d/pipelines/pixal3d_mv_image_to_3d.py) is the single-view cascade
with other image conditions: every view is projected into the same 3D grid through its own camera and
the features are averaged. It needs square views framed like its training renders and one camera pose
per view; inference_mv.py shows the input it expects. That file is not imported (at import it sets
app.py's environment variables again): the few lines the hub needs are rebuilt here.

The hub's batches run on the fork's staged engine, which loads each network when a part needs it; of this
file they use the views (crop, cut out, frame, bundle, save), the single-view cut-out (cut_out_single) and the
check of the weights on disk. The views the model saw are written by a thread of their own (save_views), so the
GPU thread does not wait for the PNG encoder.

ModelModes is for the reference path alone (POST /spellcast/generate, Pixal3D's own page), which keeps
every model in RAM. Its four multi-view denoisers are fine-tunes of the single-view ones with identical keys
and shapes. Both sets do not fit in RAM together (the single-view pipeline alone keeps ~25 GB), so ModelModes
loads one set's weights into the other's modules, one tensor at a time. A load for a prop made from several
views reads the multi-view set right away instead (ModelModes.loading_first), so nothing is read only to be
replaced.

Only the stdlib at module level: the hub's tests import this file without torch, numpy or PIL.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import math
import os
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

# Azimuth of the camera around the prop (degrees, counter-clockwise seen from above) -> the hub's view
# kind (project.json images.views). "left" shows the prop's own left side: it faces image-left.
VIEW_NAMES = {0: "front", 90: "left", 180: "back", 270: "right"}
VIEW_LABELS = {0: "Front", 90: "← Left", 180: "Back", 270: "Right →"}   # as the page names them
VIEW_ORDER = (0, 90, 180, 270)

FRONT_REQUIRED = "Mark one of the slices as Front: the 3D model is built facing it."
# For this launcher's own route and log. The hub shows its own words for this kind (adapters/pixal3d.py MV_MISSING):
# it knows where the artist downloads the models.
MV_MISSING = ("Making one 3D model from several views needs Pixal3D's multi-view models, which are not "
              "downloaded yet. Download them, or use one picture.")
NO_PROP = ("No prop was found in the picture: use a picture with a plain background, or one whose background is "
           "already transparent.")
PROP_DELETED = "This prop was deleted while its 3D model was being made."

# The camera rig of assets/mv_images/example/transforms.json (the one the README vouches for): a 20 degree
# field of view at 1.1 x distance_from_fov, so the unit cube fills 1/1.1 of the frame at the centre.
MV_CAMERA_ANGLE_X = 0.3490658503988659
MV_CAMERA_DISTANCE = 3.1192049980163574
# A picture or slice smaller than this on its largest side is enlarged to it before it is cut out (to_input_size):
# the size background removal works at and the 1024 stages read.
INPUT_SIZE = 1024
MAX_SLICE = 2048          # bigger ones are shrunk first: the framed view is 1024 px anyway
# A single picture larger than INPUT_SIZE is cut out and cropped at its own size, and only the prop's square is
# reduced (cut_out_single). False: Pixal3D's own order, the whole picture reduced to INPUT_SIZE first - only for
# checks against results made that way (the golden cases sv1024, redraw and big1536 have larger pictures).
CROP_FIRST = True
VIEW_SIZE = 1024          # the framed views: square, the size of the 1024 stages' input
FRAME_MARGIN = 1.1        # the largest extent over all views fills 1/1.1 of VIEW_SIZE (as in preprocess_image)
OBJECT_ALPHA = 204        # alpha > 0.8 * 255 counts as the prop (preprocess_image's bounding box)
# A picture's own transparency cuts the prop out (no background removal) only when its BACKGROUND is transparent:
# pixels under OBJECT_ALPHA make up at least BACKGROUND_SHARE of the picture and at least BACKGROUND_EDGE_SHARE of
# its outermost ring (EDGE_RING of its larger side, 2 px at least). A picture with a speck of transparency (a few
# anti-aliased pixels, a stray hole: one of the user's test pictures had 0.05 % of them) is an opaque one whose
# background is removed. Pixal3D's preprocess_image takes any alpha below 255 as a cut-out; that took such a
# picture's whole frame as the prop.
BACKGROUND_SHARE = 0.02
BACKGROUND_EDGE_SHARE = 0.25
EDGE_RING = 0.02
# A separate opaque part of a slice is cleared (a bit of the neighbouring view, a caption) when it touches the
# slice's edge, or when it is under STRAY_AREA of the prop's area and away from it (STRAY_GAP of the slice's
# size around the prop's box). Anything else is part of the prop: a floating orb, a sword held apart.
STRAY_AREA = 0.03
STRAY_GAP = 0.05
COND_SIZES = (512, 1024)  # SS and shape-512 stages read 512 px views, the 1024 stages 1024 px

# The four denoisers the two modes differ in (pipeline.json vs pipeline_mv.json), and the
# conditioning models the multi-view pipeline replaces (app.IMAGE_COND_CONFIGS key -> pipeline attribute).
FLOW_MODELS = ("sparse_structure_flow_model", "shape_slat_flow_model_512",
               "shape_slat_flow_model_1024", "tex_slat_flow_model_1024")
EXTRACTORS = {"ss": "image_cond_model_ss", "shape_512": "image_cond_model_shape_512",
              "shape_1024": "image_cond_model_shape_1024", "tex_1024": "image_cond_model_tex_1024"}
CONFIGS = {"sv": "pipeline.json", "mv": "pipeline_mv.json"}
REPO_CACHE = Path("hub") / "models--TencentARC--Pixal3D"

# Progress stages written while a mode switch loads weights (pixal3d_server.PHASES labels them). A switch
# happens only when this Pixal3D last held the other set: say so, it is not a second load of the same thing.
SWITCHING = {"sv": "Swapping in the single-view models", "mv": "Swapping in the multi-view models"}

SAVED_VIEW = re.compile(r"\d+_(front|left|back|right)\.png")

# The thread that writes the views the model saw (save_views): one for the process, made when first needed. Pillow's
# PNG encoder lets go of Python's lock, so the GPU thread goes on with the next view meanwhile. It is not a daemon
# thread: when the process ends normally, concurrent.futures finishes the writes already queued before the process
# goes; a process that is killed loses the ones still queued (the last fraction of a second), as it would lose a
# write under way.
_writer: concurrent.futures.ThreadPoolExecutor | None = None
_writing: dict[str, concurrent.futures.Future[None]] = {}     # a views folder -> the last write queued for it
_writing_lock = threading.Lock()


class ViewProblem(Exception):
    """A view the artist has to fix (the message says how)."""


class MultiviewMissing(Exception):
    """The multi-view weights are not (or no longer) on disk."""


class PropDeleted(ViewProblem):
    """The folder of the prop a file is for is gone: the artist deleted the prop meanwhile (the hub moves its folder
    to deleted/). Nothing is written for it and its folder is not made again; like a view to fix, it fails that
    prop alone (kind "input")."""


def prop_gone(out_path: str | Path) -> bool:
    """Whether the prop a model file is for was deleted: the file lies in <prop>/mesh/ (what the hub sends) and
    <prop> is gone. A file anywhere else (a check's scratch space) belongs to no prop."""
    mesh = Path(out_path).absolute().parent
    return mesh.name.lower() == "mesh" and not mesh.parent.is_dir()


def make_folder(folder: str | Path) -> None:
    """Make the folder a file is written to. In a prop's mesh folder - <prop>/mesh, <prop>/mesh/views, what the hub
    sends - the prop's own folder is never made: each level is made without its parents, so a prop deleted
    meanwhile raises PropDeleted instead of coming back as a folder the hub would have to clear away. Any other
    folder (a check's scratch space) is made with its parents."""
    folder = Path(folder).absolute()
    if folder.is_dir():
        return
    if "mesh" not in (folder.name.lower(), folder.parent.name.lower()):
        folder.mkdir(parents=True, exist_ok=True)
        return
    if folder.parent.name.lower() == "mesh":     # <prop>/mesh/views: its mesh folder first
        make_folder(folder.parent)
    try:
        folder.mkdir(exist_ok=True)
    except FileNotFoundError:
        raise PropDeleted(PROP_DELETED) from None


# ---- views: pure Python ----------------------------------------------------------------------------

def camera_to_world(azimuth: int, distance: float = MV_CAMERA_DISTANCE) -> list[list[float]]:
    """The 4x4 camera-to-world matrix (Blender/NeRF convention, Z up, the camera looks down its -Z) of a
    camera `distance` from the centre at `azimuth` degrees and elevation 0, as transforms.json has them:
    position d(sin a, -cos a, 0), columns right (cos a, sin a, 0), up (0, 0, 1), back (sin a, -cos a, 0).
    0 is the front view (camera at -Y), 90 looks at the prop's left side from +X."""
    s, c = {0: (0.0, 1.0), 90: (1.0, 0.0), 180: (0.0, -1.0), 270: (-1.0, 0.0)}.get(
        azimuth % 360, (math.sin(math.radians(azimuth)), math.cos(math.radians(azimuth))))
    return [[c, 0.0, s, distance * s],
            [s, 0.0, 0.0 - c, 0.0 - distance * c],      # 0.0 - x: no -0.0 entries
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]


def ordered(views: Sequence[Any]) -> list[Any]:
    """Views in VIEW_ORDER: the front one first (the model is posed facing view 0); the order of the
    others does not matter to the model (their features are averaged)."""
    return sorted(views, key=lambda v: VIEW_ORDER.index(v.azimuth) if v.azimuth in VIEW_ORDER else 4)


def label(azimuth: int) -> str:
    return VIEW_LABELS.get(azimuth, f"{azimuth}°")


def validate_views(views: Sequence[Any]) -> str | None:
    """What is wrong with the views of a request (objects with image_path, rect, azimuth), or None."""
    seen: set[int] = set()
    for view in views:
        if view.azimuth not in VIEW_NAMES:
            return f"Unknown view direction {view.azimuth}: use 0 (front), 90, 180 or 270."
        if view.azimuth in seen:
            return f"Two slices are marked {label(view.azimuth)}: each direction can be used once."
        seen.add(view.azimuth)
        rect = view.rect
        if rect is not None and not (len(rect) == 4 and all(isinstance(n, int) and not isinstance(n, bool)
                                                             for n in rect)
                                     and rect[0] >= 0 and rect[1] >= 0 and rect[2] >= 1 and rect[3] >= 1):
            return f"The {label(view.azimuth)} slice has an invalid rectangle ({rect})."
        if not Path(view.image_path).is_file():
            return (f"The picture of the {label(view.azimuth)} slice was not found: {view.image_path}. "
                    "Slice it again and retry.")
    if len(views) >= 2 and 0 not in seen:
        return FRONT_REQUIRED
    return None


# ---- the weights on disk: pure Python --------------------------------------------------------------

def snapshot_dir(models_root: Path) -> Path | None:
    """The Pixal3D snapshot folder the offline loader uses (the commit in refs/main), or None."""
    cache = Path(models_root) / REPO_CACHE
    try:
        commit = (cache / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    folder = cache / "snapshots" / commit
    return folder if commit and folder.is_dir() else None


def flow_checkpoints(models_root: Path, mode: str) -> dict[str, Path] | None:
    """FLOW_MODELS name -> checkpoint path without its extension (<snapshot>/ckpts/...), as the mode's
    pipeline config lists them; None when the config or one of the .json / .safetensors files is missing
    (a snapshot entry appears only once its download is complete)."""
    snapshot = snapshot_dir(models_root)
    if snapshot is None:
        return None
    try:
        models = json.loads((snapshot / CONFIGS[mode]).read_text(encoding="utf-8"))["args"]["models"]
        paths = {name: snapshot / models[name] for name in FLOW_MODELS}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not all(Path(f"{p}.json").is_file() and Path(f"{p}.safetensors").is_file() for p in paths.values()):
        return None
    return paths


def multiview_available(models_root: Path) -> bool:
    return flow_checkpoints(models_root, "mv") is not None


def startup_checkpoints(models_root: Path) -> dict[str, Path] | None:
    """For a start that loads the multi-view denoisers right away: the name of each single-view flow
    checkpoint (the last part of its path in pipeline.json) -> the multi-view checkpoint to load in its
    place. None unless both sets are complete on disk and each pair is the same model (identical .json
    config): then the modules are the ones init_models() would have built anyway."""
    sv, target = flow_checkpoints(models_root, "sv"), flow_checkpoints(models_root, "mv")
    if sv is None or target is None:
        return None
    try:
        same = all(json.loads(Path(f"{target[name]}.json").read_text(encoding="utf-8"))
                   == json.loads(Path(f"{sv[name]}.json").read_text(encoding="utf-8")) for name in FLOW_MODELS)
    except (OSError, ValueError):
        return None
    return {sv[name].name: target[name] for name in FLOW_MODELS} if same else None


# ---- one view: crop, cut out, frame (numpy / PIL / OpenCV, imported inside) ------------------------

def to_input_size(image: Any) -> tuple[Any, float]:
    """`image` (RGB or RGBA) at the size it is cut out at, its aspect kept -> (the image, its pixels per pixel
    of `image`). One smaller than INPUT_SIZE px on its largest side is enlarged to it, by bilinear
    interpolation: background removal, the stray bits and the prop's box then work at the size the 1024 px
    stages read however small the slice was drawn (a 512 px slice's matte is otherwise made at 512 px and
    enlarged afterwards). One up to MAX_SLICE px stays as it is - reducing it here would only lose detail that
    the framing (frame_views) and the single-view crop (cut_out_single: the prop's square is reduced, not the
    picture) still use - and a bigger one is shrunk to MAX_SLICE px. Pillow resizes RGBA premultiplied by its
    alpha: the colour under transparent pixels does not bleed into the prop's rim, and an opaque picture stays
    opaque."""
    from PIL import Image

    largest = max(image.size)
    if INPUT_SIZE <= largest <= MAX_SLICE:
        return image, 1.0
    unit, resample = ((INPUT_SIZE / largest, Image.Resampling.BILINEAR) if largest < INPUT_SIZE
                      else (MAX_SLICE / largest, Image.Resampling.LANCZOS))
    size = (max(1, round(image.width * unit)), max(1, round(image.height * unit)))
    return image.resize(size, resample), unit


def open_crop(path: str, rect: Sequence[int] | None, azimuth: int = 0) -> Any:
    """The slice `rect` ([x, y, w, h] in source pixels, clamped to the picture; None = all of it) of a
    picture, as the page shows it (EXIF rotation applied), as RGB or, when it has transparency, RGBA,
    enlarged to INPUT_SIZE px on its largest side when it is smaller (to_input_size). Its
    info["spellcast_unit"]: its pixels per pixel of the picture, so slices of one sheet keep their
    proportions (frame_views)."""
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        picture = ImageOps.exif_transpose(source)
        # Palette, grey, 16-bit: the resize and the matte need RGB(A). A colour key (PNG tRNS on RGB) becomes
        # an alpha here: resized, its pixels blend with their neighbours and no longer match the key.
        transparent = "A" in picture.getbands() or "transparency" in picture.info
        if picture.mode != ("RGBA" if transparent else "RGB"):
            picture = picture.convert("RGBA" if transparent else "RGB")
        width, height = picture.size
        box = (0, 0, width, height)
        if rect is not None:
            x, y, w, h = (int(n) for n in rect)
            box = (min(max(x, 0), width), min(max(y, 0), height),
                   min(max(x + w, 0), width), min(max(y + h, 0), height))
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                raise ViewProblem(f"The {label(azimuth)} slice lies outside its picture: slice it again.")
        image = picture.crop(box)
    drawn = image.size
    image, unit = to_input_size(image)
    image.info["spellcast_unit"] = unit
    if image.size != drawn:
        print(f"[spellcast] {VIEW_NAMES.get(azimuth, azimuth)} view: {'slice' if rect is not None else 'picture'} "
              f"of {drawn[0]}x{drawn[1]} px brought to {image.width}x{image.height} px", flush=True)
    return image


def transparent_background(image: Any) -> bool:
    """Whether the picture comes with its background transparent (BACKGROUND_SHARE, BACKGROUND_EDGE_SHARE above),
    so that its own alpha cuts the prop out and no background is removed. False for a picture without an alpha
    band or a transparency key, and for one that is opaque but for a few pixels. A few milliseconds: one pass
    over the alpha."""
    mode = getattr(image, "mode", "")
    if "A" not in mode and "transparency" not in getattr(image, "info", {}):
        return False
    import numpy as np

    alpha = np.asarray(image.getchannel("A") if "A" in mode else image.convert("RGBA").getchannel("A"))
    clear = alpha < OBJECT_ALPHA
    if clear.mean() < BACKGROUND_SHARE:
        return False
    height, width = clear.shape
    ring = max(2, round(EDGE_RING * max(width, height)))
    edge = np.ones(clear.shape, dtype=bool)
    edge[ring:height - ring, ring:width - ring] = False
    return bool(clear[edge].mean() >= BACKGROUND_EDGE_SHARE)


def real_alpha(image: Any) -> Any:
    """The image as RGBA when its background is transparent (transparent_background), else None."""
    return image.convert("RGBA") if transparent_background(image) else None


def matte(pipeline: Any, image: Any) -> Any:
    """Background removal with the pipeline's own model (BiRefNet / RMBG-2.0), as inference_mv.make_rembg:
    in low-VRAM mode it visits the GPU only for this call. In a batch `pipeline` is pixal3d_server.matte_shim
    around the staged engine's model, which is on the card already."""
    rgb = image.convert("RGB")
    if pipeline.low_vram:
        pipeline.rembg_model.to(pipeline.device)
    try:
        out = pipeline.rembg_model(rgb)
    finally:
        if pipeline.low_vram:
            pipeline.rembg_model.cpu()
    return out.convert("RGBA")


def cut_out_single(pipeline: Any, image: Any, crop_first: bool = True, keep: list[Any] | None = None) -> Any:
    """The picture of a prop made from one picture (or one slice) as the models take it: cut out, cropped to the
    prop and laid on black - what Pixal3D's pipeline.preprocess_image makes (pixal3d_image_to_3d.py), step for
    step: the picture's own alpha when its background is transparent (transparent_background; preprocess_image takes
    any alpha below 255), else background removal (matte); the box of the pixels with alpha > 0.8 * 255; a square of
    int(1.1 x its larger side) around its centre (preprocess_image's integer arithmetic, and Pillow's rounding of the
    crop box); composited on black in float32.

    preprocess_image first reduces a picture larger than INPUT_SIZE px to it (LANCZOS), and the 512 / 1024 px
    stages then enlarge the prop's crop again: detail lost. With crop_first such a picture is cut out and cropped
    at its own size (the matte resizes to 1024 x 1024 inside and gives its mask at the picture's size), and only
    the square is reduced, to INPUT_SIZE px when it is larger (LANCZOS, as preprocess_image reduces; RGBA, so
    premultiplied); a smaller square stays as it is, as preprocess_image's own squares do. A picture up to
    INPUT_SIZE px gives exactly what preprocess_image gives. crop_first False: preprocess_image's order for every
    picture. `pipeline`: what matte() takes. `keep`: a list the RGBA square is appended to (Hi3D frames the prop by
    its alpha). Raises ViewProblem when nothing in the picture is opaque enough."""
    import numpy as np
    from PIL import Image

    own = transparent_background(image)                                            # told before any resize
    largest = max(image.size)
    if largest > INPUT_SIZE and not crop_first:
        scale = INPUT_SIZE / largest
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    output = image if own else matte(pipeline, image)
    found = np.argwhere(np.array(output)[:, :, 3] > 0.8 * 255)
    if len(found) == 0:
        raise ViewProblem(NO_PROP)
    bbox = np.min(found[:, 1]), np.min(found[:, 0]), np.max(found[:, 1]), np.max(found[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.1)
    square = output.crop((center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2))
    if largest > INPUT_SIZE and crop_first and max(square.size) > INPUT_SIZE:
        drawn = square.size
        unit = INPUT_SIZE / max(drawn)
        square = square.resize((max(1, round(drawn[0] * unit)), max(1, round(drawn[1] * unit))),
                               Image.Resampling.LANCZOS)
        print(f"[spellcast] Cut out at {image.width}x{image.height} px; the prop's square of {drawn[0]} px "
              f"brought to {square.width} px", flush=True)
    if keep is not None:
        keep.append(square)
    rgba = np.array(square).astype(np.float32) / 255
    rgb, alpha = rgba[:, :, :3], rgba[:, :, 3:4]
    black = np.array((0, 0, 0), dtype=np.float32) / 255.0
    return Image.fromarray((np.clip(rgb * alpha + black * (1.0 - alpha), 0, 1) * 255).astype(np.uint8))


def object_box(image: Any) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1), end exclusive, of the pixels with alpha > OBJECT_ALPHA; None when there are none."""
    import numpy as np

    ys, xs = np.nonzero(np.asarray(image.getchannel("A")) > OBJECT_ALPHA)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def strays(stats: Sequence[Sequence[int]], width: int, height: int) -> list[int]:
    """Which opaque parts of a slice are strays, not the prop: stats[i] = (x, y, w, h, area) of part i (as
    OpenCV's connectedComponentsWithStats gives them, without the background) in a width x height slice.
    The largest part is the prop. Another part is a stray when it touches the slice's edge (a bit of the
    neighbouring view the slice cut into), or when it is small (under STRAY_AREA of the prop's area) and
    away from the prop's box (a caption, a speck). Everything else stays: a floating orb, a sword held
    apart, a chain the matte broke. Returns the indices of the strays."""
    if len(stats) <= 1:
        return []
    largest = max(range(len(stats)), key=lambda i: stats[i][4])
    px, py, pw, ph, parea = stats[largest]
    gap = STRAY_GAP * max(width, height)
    found = []
    for i, (x, y, w, h, area) in enumerate(stats):
        if i == largest:
            continue
        edge = x <= 0 or y <= 0 or x + w >= width or y + h >= height
        near = x < px + pw + gap and x + w > px - gap and y < py + ph + gap and y + h > py - gap
        if edge or (area < STRAY_AREA * parea and not near):
            found.append(i)
    return found


def clear_strays(image: Any) -> tuple[int, int]:
    """Clear the alpha of the stray opaque parts of a slice (strays: bits of a neighbouring view, a caption),
    each with its soft edge. Returns (how many parts there were, how many were cleared)."""
    import cv2
    import numpy as np
    from PIL import Image

    alpha = np.asarray(image.getchannel("A"))
    count, labels, stats, _ = cv2.connectedComponentsWithStats((alpha > 127).astype(np.uint8), connectivity=8)
    parts = count - 1
    gone = strays([tuple(int(n) for n in row[:5]) for row in stats[1:]], image.width, image.height)
    if not gone:
        return parts, 0
    drop = np.isin(labels, [1 + i for i in gone]).astype(np.uint8)
    rest = ((labels > 0) & (drop == 0)).astype(np.uint8)
    # Each stray goes with its anti-aliased rim, but never a pixel of what stays.
    clear = cv2.dilate(drop, np.ones((7, 7), np.uint8)) & (1 - cv2.dilate(rest, np.ones((3, 3), np.uint8)))
    image.putalpha(Image.fromarray(alpha * (1 - clear)))
    return parts, len(gone)


def cut_out(pipeline: Any, image: Any, azimuth: int = 0) -> Any:
    """One slice -> RGBA with only the prop opaque: its own alpha when its background is transparent, else a matte."""
    rgba = real_alpha(image)
    matted = rgba is None
    if matted:
        rgba = matte(pipeline, image)
    parts, cleared = clear_strays(rgba)
    if parts < 1 or object_box(rgba) is None:
        raise ViewProblem(f"No prop was found in the {label(azimuth)} slice: draw the slice around the prop, "
                          "or use a picture with a plain background.")
    print(f"[spellcast] {VIEW_NAMES.get(azimuth, azimuth)} view: {rgba.width}x{rgba.height} px, "
          f"{'background removed' if matted else 'own transparency'}"
          + (f", cleared {cleared} stray bit{'s' if cleared > 1 else ''} of {parts} parts" if cleared else ""),
          flush=True)
    return rgba


def frame_scales(sizes: Sequence[tuple[float, float]], groups: Sequence[Any]) -> list[float]:
    """How much to scale each view's prop, in pixels of its framed square per pixel of its source picture:
    sizes[i] = (width, height) of view i's prop in its source picture's pixels; groups[i] = the picture it
    was cut from.

    Views cut from one picture (a turnaround sheet) share one scale: the sheet's own proportions are exact
    (a back view drawn a little taller stays a little taller). Views from different pictures (photos from
    each side, at any distance) cannot be compared pixel for pixel: at elevation 0 the prop is equally tall
    from every side, so each picture's views are scaled to one common height - a picture's height being the
    mean of its views'. Then everything is scaled together so the largest extent over all views (height or
    width) fills 1/FRAME_MARGIN of the square."""
    heights: dict[Any, list[float]] = {}
    for (_, h), group in zip(sizes, groups):
        heights.setdefault(group, []).append(h)
    ref = {group: sum(hs) / len(hs) for group, hs in heights.items()}
    ratio = max(max(w, h) / ref[group] for (w, h), group in zip(sizes, groups))
    target = VIEW_SIZE / FRAME_MARGIN / ratio          # the common height (a group's mean height) when framed
    return [target / ref[group] for group in groups]


def frame_views(images: Sequence[Any], azimuths: Sequence[int], sources: Sequence[Any] | None = None,
                units: Sequence[float] | None = None) -> list[Any]:
    """Frame the cut-out views like the training renders: VIEW_SIZE px squares, each centred on its
    prop's bounding box, scaled by frame_scales: views of one source picture (sources[i]; None: each view
    its own) together, different pictures to one common silhouette height. units[i]: pixels of image i
    per pixel of its source picture (open_crop's info["spellcast_unit"]; 1.0 when not given)."""
    boxes = [object_box(im) for im in images]
    if any(box is None for box in boxes):
        raise ViewProblem("One of the slices shows no prop.")
    units = list(units) if units is not None else [1.0] * len(images)
    groups = list(sources) if sources is not None else list(range(len(images)))
    sizes = [((x1 - x0) / unit, (y1 - y0) / unit) for (x0, y0, x1, y1), unit in zip(boxes, units)]   # type: ignore[misc]
    scales = frame_scales(sizes, groups)
    shared = len(set(groups)) < len(groups)
    framed = []
    for image, (x0, y0, x1, y1), azimuth, unit, scale in zip(images, boxes, azimuths, units, scales):   # type: ignore[misc]
        scale = scale / unit                             # per pixel of the cut-out image
        framed.append(place(image, scale, (x0 + x1) / 2, (y0 + y1) / 2))
        print(f"[spellcast] {VIEW_NAMES.get(azimuth, azimuth)} view framed: prop {x1 - x0}x{y1 - y0} px scaled "
              f"{scale:.2f}x to {(x1 - x0) * scale:.0f}x{(y1 - y0) * scale:.0f} of {VIEW_SIZE} px", flush=True)
    if shared:
        print("[spellcast] Views cut from one picture keep its proportions (one scale for them).", flush=True)
    return framed


def place(image: Any, scale: float, cx: float, cy: float) -> Any:
    """A VIEW_SIZE square of the RGBA `image` scaled by `scale` with source point (cx, cy) at its centre
    (continuous pixel coordinates); transparent where the image does not reach."""
    from PIL import Image

    half = VIEW_SIZE / 2 / scale                                  # half the square, in source pixels
    box = [cx - half, cy - half, cx + half, cy + half]
    left, top = max(0, math.ceil(-box[0])), max(0, math.ceil(-box[1]))
    right, bottom = max(0, math.ceil(box[2] - image.width)), max(0, math.ceil(box[3] - image.height))
    if left or top or right or bottom:                            # resize(box=) must stay inside the image
        padded = Image.new("RGBA", (image.width + left + right, image.height + top + bottom), (0, 0, 0, 0))
        padded.paste(image, (left, top))
        image = padded
        box = [box[0] + left, box[1] + top, box[2] + left, box[3] + top]
    # Pillow resizes RGBA premultiplied, so no dark or coloured fringe bleeds in from transparent pixels.
    return image.resize((VIEW_SIZE, VIEW_SIZE), Image.Resampling.LANCZOS, box=tuple(box))


def view_bundle(framed: Sequence[Any], azimuths: Sequence[int]) -> dict[str, Any]:
    """The input of Pixal3DMVImageTo3DPipeline.run_mv, built like inference_mv.load_views: each view at
    512 and 1024 px (LANCZOS), premultiplied by its alpha (black background); one c2w matrix, the rig's
    field of view and the matrix's distance per view; view 0 must be the front."""
    import numpy as np
    import torch
    from PIL import Image

    def cond(image: Any, size: int) -> Any:   # inference_mv.to_cond_tensor
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        alpha = torch.tensor(np.array(image.getchannel(3))).float() / 255.0
        rgb = torch.tensor(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 255.0
        return rgb * alpha.unsqueeze(0)

    transform = torch.tensor([camera_to_world(a) for a in azimuths], dtype=torch.float32)[None]   # [1, V, 4, 4]
    return {
        "images": {size: torch.stack([cond(im, size) for im in framed], dim=0)[None]           # [1, V, 3, S, S]
                   for size in COND_SIZES},
        "camera_angle_x": torch.tensor([MV_CAMERA_ANGLE_X] * len(framed), dtype=torch.float32)[None],   # [1, V]
        "camera_distance": torch.norm(transform[:, :, :3, 3], dim=-1),                           # [1, V]
        "transform_matrix": transform,
        "mesh_scale": 1.0,
        "view_names": [VIEW_NAMES.get(a, str(a)) for a in azimuths],
    }


def save_views(views_dir: str | None, named: Sequence[tuple[str, Any]]) -> concurrent.futures.Future[None] | None:
    """Save what the model saw as <n>_<view>.png in `views_dir` (the page can show it), replacing the
    views of an earlier run. Fails the generation only when the prop was deleted meanwhile (PropDeleted:
    make_folder does not make its folder again); a file that cannot be written is only noted.

    The folder is made and the earlier views are deleted here, before this returns, so that PropDeleted still
    reaches the caller; the pictures are encoded and written by the writer thread, so the GPU thread does not wait
    for them (the hub shows them only once the prop is made, minutes later). They are copied first: the caller goes
    on with its own (the camera estimate saves the single-view picture again, and Pillow's save sets attributes on
    the picture it writes). Returns the write (wait_for_views waits for every one), None when nothing is written."""
    if not views_dir:
        return None
    folder = Path(views_dir)
    key = os.path.normcase(str(folder.absolute()))
    try:
        make_folder(folder)
        with _writing_lock:
            earlier = _writing.get(key)
        if earlier is not None:
            earlier.result()        # the folder's earlier views are written before they are deleted, never after
        for old in folder.iterdir():
            if SAVED_VIEW.fullmatch(old.name):
                old.unlink(missing_ok=True)
    except OSError as exc:
        print(f"[spellcast] Note: could not save the views the model saw in {folder} ({exc}).", flush=True)
        return None
    files = [(folder / f"{n}_{name}.png", image.copy()) for n, (name, image) in enumerate(named)]
    global _writer
    with _writing_lock:
        if _writer is None:
            _writer = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="spellcast-views")
        for done in [folder_key for folder_key, write in _writing.items() if write.done()]:
            del _writing[done]
        write = _writing[key] = _writer.submit(_write_views, folder, files)
    return write


def _write_views(folder: Path, files: list[tuple[Path, Any]]) -> None:
    """The writer thread's half of save_views. Nothing is raised: nobody waits for the result to hear of it."""
    try:
        for path, image in files:
            image.save(path)
    except OSError as exc:
        print(f"[spellcast] Note: could not save the views the model saw in {folder} ({exc}).", flush=True)
    except Exception:
        traceback.print_exc()
        print(f"[spellcast] Note: could not save the views the model saw in {folder}.", flush=True)


def wait_for_views(timeout: float | None = None) -> None:
    """Wait until every view save_views has queued is written (checks and tests that read the files at once)."""
    with _writing_lock:
        writes = list(_writing.values())
    concurrent.futures.wait(writes, timeout)


# ---- mode switch: which denoisers the flow models hold -----------------------------------------------

def check_swappable(models: dict[str, Any], sv: dict[str, Path], target: dict[str, Path]) -> None:
    """Refuse before anything changes unless every target checkpoint fits the loaded modules: the same
    model config as the single-view one they were built from, the same shape for every shared key."""
    from safetensors import safe_open

    for name in FLOW_MODELS:
        config = json.loads(Path(f"{target[name]}.json").read_text(encoding="utf-8"))
        if config != json.loads(Path(f"{sv[name]}.json").read_text(encoding="utf-8")):
            raise RuntimeError(f"{target[name].name} is a different model than {sv[name].name}; "
                               "it cannot be loaded in its place.")
        state = models[name].state_dict()
        with safe_open(f"{target[name]}.safetensors", framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                if key in state and tuple(f.get_slice(key).get_shape()) != tuple(state[key].shape):
                    raise RuntimeError(f"{target[name].name}: {key} has another shape than the loaded model's.")
        missing, unexpected = len(set(state) - set(keys)), len(set(keys) - set(state))
        if missing or unexpected:   # load_state_dict(strict=False) at startup tolerated the same
            print(f"[spellcast] Note: {target[name].name} leaves {missing} tensors of the model as they are "
                  f"and has {unexpected} it does not use.", flush=True)


def shared_state(pipeline: Any) -> dict[str, Any]:
    """A pipeline's instance attributes minus those that shadow a method of its class: the launcher's
    wrappers (trim_cuda_cache, label_decoding, the sparse-structure guard) wrap the single-view methods."""
    owner = type(pipeline)
    return {k: v for k, v in vars(pipeline).items() if not callable(getattr(owner, k, None))}


def copy_weights(module: Any, path: Path) -> int:
    """Load a .safetensors checkpoint into `module`'s existing tensors, one tensor at a time.

    Same result as pixal3d.models.from_pretrained's load_state_dict(load_file(...), strict=False), which
    holds the whole file (5.5 GB in fp32) in RAM at once: copy_ converts the fp32 on disk to the bf16 the
    model keeps its transformer blocks in, like load_state_dict does. Returns the bytes read.
    """
    import torch
    from safetensors import safe_open

    state = module.state_dict()        # detached views of the parameters: copying into them updates the model
    read = 0
    with safe_open(str(path), framework="pt", device="cpu") as f, torch.no_grad():
        for key in f.keys():
            target = state.get(key)
            if target is None:
                continue
            tensor = f.get_tensor(key)
            target.copy_(tensor)
            read += tensor.numel() * tensor.element_size()
            del tensor
    return read


class ModelModes:
    """Which denoisers Pixal3D's four flow models hold, and the multi-view pipeline that uses them.

    "sv": the single-view weights init_models() loads by itself; px.pipeline (and Pixal3D's own page) needs
    them. "mv": the multi-view fine-tunes, loaded into the SAME modules (identical keys and shapes), so no
    second 11 GB set is kept in RAM; mv_pipeline runs them. None: a switch failed half-way, and the
    next ensure() loads all four again. Call everything on the GPU worker thread.

    The set a start loads is the one its first job needs (loading_first): a start that loaded the
    single-view set for a multi-view prop read 22 GB only to replace them at once. After that the set
    changes only when a job (or Pixal3D's own page) needs the other one, and stays between jobs of one kind.

    mv_pipeline is a Pixal3DMVImageTo3DPipeline built once from px.pipeline's attributes: it shares the
    models dict (flow models and decoders), background removal, samplers and settings, and has its own
    four multi-view conditioning models that reuse the loaded DINOv3 and NAF. Only its class differs,
    for the methods it overrides (run_mv, and what a condition is built from). Binding those
    methods onto px.pipeline instead would change the pipeline Pixal3D's own page uses. The launcher's
    wrappers live in instance attributes that shadow methods (trim_cuda_cache, label_decoding, the
    sparse-structure guard): they are NOT copied - they would bring back the single-view condition
    builders - and `prepare` installs them again on the new pipeline.
    """

    def __init__(self, px: Any, models_root: Path, prepare: Callable[[Any], None]) -> None:
        self.px = px
        self.models_root = Path(models_root)
        self.prepare = prepare
        self.mode: str | None = "sv"
        self.mv_pipeline: Any = None

    @contextlib.contextmanager
    def loading_first(self, mode: str, loader: Any) -> Iterator[None]:
        """Run init_models() inside this to make it load `mode`'s denoisers. "sv" is what it does by itself.
        For "mv" (the hub starts Pixal3D for a prop made from several views) `loader`, the pixal3d.models
        module, is handed the multi-view checkpoint in place of each single-view one: the pipeline calls
        loader.from_pretrained through the module attribute, so replacing it while init_models() runs is
        enough. Everything else it loads is shared by both modes. Without a complete multi-view set
        (startup_checkpoints) the single-view one is loaded as usual, and the job says what is missing."""
        swap = startup_checkpoints(self.models_root) if mode == "mv" else None
        if swap is None:
            yield
            return
        original = loader.from_pretrained
        loaded: set[str] = set()

        def from_pretrained(path: Any, *args: Any, **kwargs: Any) -> Any:
            other = swap.get(Path(str(path)).name)
            if other is None:
                return original(path, *args, **kwargs)
            model = original(str(other), *args, **kwargs)
            loaded.add(other.name)
            return model

        loader.from_pretrained = from_pretrained
        try:
            yield
        finally:
            loader.from_pretrained = original
        if len(loaded) == len(FLOW_MODELS):
            self.mode = "mv"
            print("[spellcast] Loaded the multi-view models right away: the prop that waits is made from "
                  "several views. The single-view ones take their place when a prop made from one picture "
                  "comes.", flush=True)
        elif loaded:     # Pixal3D loads its models another way now: the first job loads the set it needs
            self.mode = None
            print(f"[spellcast] Note: only {len(loaded)} of the {len(FLOW_MODELS)} multi-view models were "
                  "loaded at the start; the first 3D model loads its set again.", flush=True)

    def reset(self) -> None:
        """The reference path's models were dropped (pixal3d_server.Reference.unload): forget the multi-view
        pipeline built from them. The next load starts from the single-view set, as init_models() loads it."""
        self.mode = "sv"
        self.mv_pipeline = None

    def extractors(self) -> list[Any]:
        """The multi-view conditioning models, once built (release_gpu puts them back on the CPU)."""
        if self.mv_pipeline is None:
            return []
        return [getattr(self.mv_pipeline, attr, None) for attr in EXTRACTORS.values()]

    def pipeline(self) -> Any:
        return self.mv_pipeline if self.mode == "mv" else self.px.pipeline

    def ensure(self, mode: str) -> None:
        if mode not in CONFIGS:
            raise ValueError(f"unknown mode {mode!r}")
        if mode == self.mode:
            if mode == "mv" and self.mv_pipeline is None:
                self.mv_pipeline = self._build_mv_pipeline()
            return
        target = flow_checkpoints(self.models_root, mode)
        sv = flow_checkpoints(self.models_root, "sv")
        if target is None or sv is None:
            if mode == "mv":
                raise MultiviewMissing(MV_MISSING)
            raise FileNotFoundError(f"Pixal3D's single-view model files in {self.models_root / REPO_CACHE}")
        models = self.px.pipeline.models
        check_swappable(models, sv, target)
        print(f"[spellcast] Switching Pixal3D from its {self._name(self.mode)} to its {self._name(mode)} "
              "models (the two sets do not fit in memory together, so one replaces the other)...", flush=True)
        started = time.monotonic()
        self.mode = None
        read = 0
        for n, name in enumerate(FLOW_MODELS):
            self.px._update_progress(SWITCHING[mode], n, len(FLOW_MODELS))
            read += copy_weights(models[name], Path(f"{target[name]}.safetensors"))
        self.px._update_progress(SWITCHING[mode], len(FLOW_MODELS), len(FLOW_MODELS))
        self.mode = mode
        print(f"[spellcast] Switched to the {self._name(mode)} models in {time.monotonic() - started:.0f} s "
              f"({read / 1e9:.1f} GB read).", flush=True)
        if mode == "mv" and self.mv_pipeline is None:
            self.mv_pipeline = self._build_mv_pipeline()

    @staticmethod
    def _name(mode: str | None) -> str:
        return {"sv": "single-view", "mv": "multi-view"}.get(mode or "", "half-loaded")

    def _build_mv_pipeline(self) -> Any:
        from pixal3d.pipelines import Pixal3DMVImageTo3DPipeline
        from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
            DinoV3ProjMultiViewFeatureExtractor,
        )

        started = time.monotonic()
        sv = self.px.pipeline
        mv = Pixal3DMVImageTo3DPipeline()        # models=None: an empty shell, filled from the single-view one
        mv.__dict__.update(shared_state(sv))
        for key, attr in EXTRACTORS.items():
            single = getattr(sv, attr)
            # inference_mv.IMAGE_COND_CONFIGS: app.py's configs plus the fusion the fine-tunes were trained with
            extractor = DinoV3ProjMultiViewFeatureExtractor(**self.px.IMAGE_COND_CONFIGS[key],
                                                            multiview_fusion="average")
            extractor.eval()
            if extractor.model is not single.model:    # share_dinov3 normally hands back the same object
                extractor.model = single.model
            if getattr(single, "naf_model", None) is not None:
                extractor.naf_model = single.naf_model
            if not sv.low_vram:
                extractor.cuda()                         # its projection grid; DINOv3 and NAF are there already
            setattr(mv, attr, extractor)
        self.prepare(mv)
        print(f"[spellcast] Multi-view pipeline ready in {time.monotonic() - started:.1f} s "
              "(sharing DINOv3, NAF, the decoders and background removal).", flush=True)
        return mv
