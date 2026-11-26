"""A repaint of what the artist masked on the OmniTextured model, through the page's own camera, and the fade of
its edge: the worker's "look", "left", "repaint" and "fades" (omnitexture_worker's doc has the commands).

The picture Qwen-Image made of the masked surface goes onto the masked texels the camera can paint. Its edge is
blended in 3D as a seam between two views is, `fade` times as far. Two kinds of edge fade apart: where the camera
can paint beyond the mask, outwards (the picture holds what is around the mask too: drawn anew in a rim, the view
itself beyond it); where the mask
meets the end of what the camera can paint - the surface turns away from it or hides behind something -, inwards:
there is nothing beyond to fade into, so the repaint itself thins out towards that edge."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image
from spellbake import BakeError
from omnitexture_blend import blend, within_reach
from omnitexture_masks import SNAP, SNAP_REACH, SPECK, clean, grow, mask_left, shrink
from omnitexture_pixels import between
from omnitexture_tone import on_clay, to_bytes
from omnitexture_views import SMOOTH_SIDE, PageLens, cast, depth_grey, faced, smooth_depth

if TYPE_CHECKING:
    from omnitexture_worker import Prop

MASK_COLOUR = (255, 0, 255)  # what is masked in the picture Qwen-Image restores: magenta (its words name it)
# px around the masked pixels that are drawn anew with them: where the two meet. No wider: with 32 px the model
# filled the whole band with a pale blur.
HELD_ROOM = 8
MASK_GUTTER = 2              # texels of the gutter beside masked texels that count as masked in the picture
# The page's slider (stages/rules.js) runs from 1, a seam's own reach, to MAX_FADE in steps of FADE_STEP, and the
# hub lets no other fade through: these are the fades worked out ahead.
MAX_FADE = 8.0
FADE_STEP = 0.5
FADE_STEPS = tuple(1.0 + FADE_STEP * n for n in range(round((MAX_FADE - 1.0) / FADE_STEP) + 1))
FADE_MARGIN = 4.0            # reaches around the repainted texels within which the blend is worked out
KEPT_SPARE = 2               # repaints "fades" keeps beyond those asked: one taken off comes back at once (Ctrl+Z)
FULL_SHARE = 90              # the percentile of the mask's blurred share that a fade raises to the whole picture
NOTHING_MASKED = "Nothing is masked on the model. Paint over what should be made again first."
MODEL_OUT_OF_SIGHT = "The model is not in sight. Turn the view to what you masked, then render again."
MASK_OUT_OF_SIGHT = ("Nothing of what you masked is in sight from here. Turn the view so that the masked part "
                     "faces you, then render again.")


@dataclass
class Look:
    """What a repaint's camera sees of the model: the texels its picture may go onto (`chosen`: indices into the
    islands' texels) and their places in it, the texels the artist masked (per texel of the islands), and in its
    picture where the mask shows and where the model is."""
    lens: PageLens
    chosen: np.ndarray
    px: np.ndarray
    py: np.ndarray
    masked: np.ndarray
    shown: np.ndarray
    model: np.ndarray

    def can(self) -> np.ndarray:
        """Per texel of the islands, whether the camera can paint it."""
        able = np.zeros(len(self.masked), dtype=bool)
        able[self.chosen] = True
        return able


def clamped_fade(fade: Any) -> float:
    """A fade as asked, within 1 (a seam's own reach) and MAX_FADE."""
    return min(max(float(fade or 1.0), 1.0), MAX_FADE)


def _made_from(asked: dict[str, Any], lens: PageLens) -> tuple:
    """What a repaint's look, picture and weights are made from: its lens, and its mask's and its picture's
    files as they are (size and time of writing: the hub's link or copy of one in another folder is the same)."""
    files = tuple((stat.st_size, stat.st_mtime_ns) for stat in (Path(asked[key]).stat() for key in ("mask", "picture")))
    return (type(lens), *(float(value) for part in astuple(lens) for value in np.ravel(part)), *files)


class Repaints:
    """One prop's repaints: the look through the page's camera, the picture laid onto what it sees of the mask,
    and - for the page's edge fade - what a change of a fade does not change, kept from call to call."""

    def __init__(self, prop: Prop) -> None:
        self.prop = prop
        self.at: Look | None = None                  # the last look
        self.encoders = ThreadPoolExecutor(4)        # a fade's models and layers, written side by side
        self.ahead = ThreadPoolExecutor(1)           # a repaint's other fades, worked out while idle
        self._stop_ahead = threading.Event()         # of the job working ahead now (each job has its own)
        self._lock = threading.Lock()
        # fades: per repaint (_made_from) its look, picture and weights per fade; the one used last stands last
        self.kept: dict[tuple, dict[str, Any]] = {}
        self._loaded: tuple | None = None            # fades: the model as loaded (Prop.snapshot)

    def close(self) -> None:
        self._stop_ahead.set()
        self.ahead.shutdown(wait=True)
        self.encoders.shutdown(wait=True)

    def loaded_anew(self) -> None:
        """The prop holds another texture (Prop.retexture): "fades" starts from that one. What is kept of the
        repaints stays - it is made from the surface, the masks and the cameras, never from the texture."""
        self._loaded = None

    # ---- the look --------------------------------------------------------------------------------------------------
    def look(self, lens: PageLens, mask: Path, pictures: bool = True, smooth: bool = False) -> dict[str, Any]:
        """The worker's "look" through `lens`: mask is a picture of the texture's size, white where the artist
        masked. Without `pictures` only where a picture made through this camera before would go. `smooth`: the
        depth map without the model's facets (the page's "No Facets": smooth_depth)."""
        prop, size = self.prop, lens.width
        marked = self._marked(mask)
        masked = marked.reshape(-1)[prop.texel]
        if not masked.any():
            raise BakeError(NOTHING_MASKED)
        rays = lens.rays()
        along = rays @ lens.forward              # (a ray's depth, from the lens's own plane, is this times its length)
        met = cast(prop.scene, prop.mesh.faces, lens.origins(), rays, lens.reach / along)
        if not met.hit.any():
            raise BakeError(MODEL_OUT_OF_SIGHT)
        uv = met.between(prop.mesh.uv)
        depth = np.full(size * size, np.inf)
        depth[met.hit] = met.t * along[met.hit]
        over = self._masked_pixels(uv, marked)
        shown = np.zeros(size * size, dtype=bool)
        shown[met.hit] = over
        # Where its picture may go: what faces it, without specks; then the texels of that, away from snaps. An
        # inverse lens ends the model where its reach does, and lets a slanted surface take the picture right up
        # to the model's outline: no texel reads within SNAP_REACH of that outline either, where Qwen-Image's
        # painting may be the background's (a texel's four pixels lie up to one from its own).
        facing = np.zeros(size * size, dtype=bool)
        facing[met.hit] = faced(lens, (met.unit(prop.smooth) * -rays[met.hit]).sum(axis=1), depth[met.hit])
        model = met.hit.reshape(size, size)
        may = clean(facing.reshape(size, size), SPECK) & model
        if lens.reach < np.inf:
            may &= shrink(model, SNAP_REACH + 1)
        chosen, px, py, _ = prop.targets(lens, may, depth.reshape(size, size))
        look = Look(lens, chosen, px, py, masked, shown.reshape(size, size), model)
        seen = int((look.can() & masked).sum())
        if not seen:
            raise BakeError(MASK_OUT_OF_SIGHT)
        self.at = look
        answer = {"masked": int(masked.sum()), "seen": seen}
        if pictures:
            viewport = to_bytes(self._textured(uv))
            viewport[over] = MASK_COLOUR
            redrawn = grow(look.shown, HELD_ROOM).reshape(-1)[met.hit]
            if smooth:
                depth = smooth_depth(depth.reshape(size, size), SMOOTH_SIDE * size, SNAP * prop.scene.diag).reshape(-1)
            answer = {**self._pictures(lens, met.hit, depth_grey(depth[met.hit]), viewport, redrawn), **answer}
        return answer

    def _marked(self, mask: Path) -> np.ndarray:
        """The texels the artist masked, (res, res)."""
        res = self.prop.res
        with Image.open(mask) as opened:
            return np.asarray(opened.convert("L").resize((res, res), Image.Resampling.NEAREST)) > 127

    def _textured(self, uv: np.ndarray, values: np.ndarray | None = None) -> np.ndarray:
        """`values` (res x res x c, kept per texel; the texture as it is now when none) at the places `uv`,
        between the four texels around each: the page's camera may be close, a pixel then smaller than a texel."""
        res = self.prop.res
        flat = (self.prop.paint if values is None else values).reshape(res * res, -1)
        return between(flat, res, res, uv[:, 0] * res - 0.5, (1.0 - uv[:, 1]) * res - 0.5)

    def _masked_pixels(self, uv: np.ndarray, marked: np.ndarray) -> np.ndarray:
        """Which of the pixels that read the texture at `uv` show the mask: most of what a pixel reads is masked,
        the gutter beside masked texels counted (no unmasked lines along the UV seams), and some of it is masked
        surface itself, a texel of an island: the gutter alone - the page's brush marks it too - would speck the
        islands that lie beside a masked one in the texture."""
        res = self.prop.res
        inside = self.prop.inside.reshape(res, res)
        grown = marked | (grow(marked, MASK_GUTTER) & ~inside)
        return ((self._textured(uv, grown.reshape(res, res, 1).astype(np.float32))[:, 0] > 0.5)
                & (self._textured(uv, (marked & inside).reshape(res, res, 1).astype(np.float32))[:, 0] > 0.0))

    def _pictures(self, lens: PageLens, hit: np.ndarray, grey: np.ndarray, viewport: np.ndarray,
                  redrawn: np.ndarray) -> dict[str, Any]:
        """A look's pictures written, each the lens's own square: the depth map, and the model as it is
        textured now with the mask in MASK_COLOUR (never the browser's lit picture: its shading would go onto
        the model). Nothing else of the model's texture: a picture of it without the mask - this view, or a
        camera's near it - shows Qwen-Image what is under the mask, and it paints that back. And, not for its
        eyes, where the view is `redrawn` (white): the drawing is held to the view everywhere else."""
        prop, size = self.prop, lens.width
        out = {key: prop.work / f"look_{key}.png" for key in ("depth", "viewport", "redrawn")}
        writes = []
        for key, colours in (("depth", grey[:, None]), ("viewport", viewport), ("redrawn", redrawn[:, None] * np.uint8(255))):
            picture = np.zeros((size * size, 3), np.uint8)
            picture[hit] = colours
            writes.append(prop.writer.submit(Image.fromarray(picture.reshape(size, size, 3)).save, out[key],
                                             compress_level=1))
        for done in writes:
            done.result()
        return {key: str(path) for key, path in out.items()}

    def left(self, look: Look, picture: Path) -> dict[str, Any]:
        """How much of the mask's shape Qwen-Image's picture of what the look showed it still has (mask_left)."""
        return {"left": round(mask_left(self.prop.picture(picture, look.lens.width), look.shown, look.model), 1)}

    # ---- a picture onto the look's texels --------------------------------------------------------------------------
    def repaint(self, look: Look, picture: Path, keep: Path | None = None, layer: Path | None = None,
                fade: float = 1.0) -> dict[str, Any]:
        """The worker's "repaint": Qwen-Image's picture of what the look showed it onto the masked texels, as it was
        painted. Its lighting is not taken off and its tone is not matched: it restores a view of the textured
        model, which is lit and toned as the model is already (a second lighting pass made a pale model's repaint
        nearly twice as bright)."""
        prop, size = self.prop, look.lens.width
        whole = prop.picture(picture, size)
        if keep is not None:
            Image.fromarray(whole).save(keep)
        to, sampled, a = self._onto(look, whole, self.weight(look, fade))
        prop.mix_in(to, sampled, a)
        if layer is not None:
            prop.write_layer(layer, to, on_clay(sampled, a))
        prop.resettle()
        return {"painted": len(to)}

    def weight(self, look: Look, fade: float) -> np.ndarray:
        """How much of a picture each texel takes (0 to 1; nothing outside the camera's reach): the masked texels
        the camera can paint take it, their edge blended in 3D as a seam between two views is (the texture as it
        is plays the model's own), `fade` times as far (the module's doc). Only the texels near them are looked
        at. It depends on the model's shape, the mask and the camera alone, never on the texture."""
        prop = self.prop
        can = look.can()
        mine = can & look.masked
        fade = clamped_fade(fade)
        reach = prop.seam_reach()
        margin = FADE_MARGIN * reach * fade
        lo, hi = prop.point[mine].min(axis=0) - margin, prop.point[mine].max(axis=0) + margin
        near = np.flatnonzero(((prop.point >= lo) & (prop.point <= hi)).all(axis=1))
        grain = 2.0 * reach / prop.seam_texels()                  # two texels' sides

        def blended(owned: np.ndarray, able: np.ndarray, radius: float) -> np.ndarray:
            """blend's weight of the one view that owns `owned` and can paint `able`, per texel (0 away from them)."""
            out = np.zeros(len(prop.texel), np.float32)
            out[near] = blend(prop.point[near], prop.normal[near], np.where(owned[near], 0, -1).astype(np.int16),
                              [able[near]], radius, grain=grain)[0][0]
            return out

        if fade == 1.0:
            return blended(mine, can, reach)
        # Each edge alone, as blend weighs a view that can paint everywhere: the mask's share, blurred, raised
        # so that it is whole at the mask's edge (fully at a long fade); times the share of what the camera can
        # paint, blurred, which is nothing at that end and whole further in. (blend's own weight lets that end
        # show only where the old texture still has a share: raised, it made the inner edge steeper.)
        radius = reach * fade
        everywhere = np.ones(len(prop.texel), dtype=bool)
        spread = blended(mine, everywhere, radius)
        inside = within_reach(blended(can, everywhere, radius))
        full = float(np.percentile(spread[mine], FULL_SHARE))    # (less in a small mask: spread thin)
        gain = 1.0 + (2.0 / max(full, 1e-3) - 1.0) * (1.0 - 1.0 / fade)
        return np.minimum(spread * gain, 1.0) * inside

    def _onto(self, look: Look, whole: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A picture through the look's camera (`whole`, its bytes) at the texels with any `weight` -> (their
        places in the texture, the picture's colour there, between its pixels; the weight, (n, 1))."""
        size = look.lens.width
        some = weight[look.chosen] > 0
        a = weight[look.chosen][some].astype(np.float64)[:, None]
        sampled = between(whole.reshape(size * size, 3), size, size, look.px[some], look.py[some])
        return self.prop.texel[look.chosen][some], sampled, a

    # ---- the repaints put on again, each with its fade -------------------------------------------------------------
    def fades(self, repaints: list[dict[str, Any]], lenses: list[PageLens], outs: list[str] | None,
              ahead: int | None = None) -> dict[str, Any]:
        """The worker's "fades": the repaints (each through its lens) onto the model as loaded, one after the
        other, each with what was painted by hand on its step over it (Prop.step_models). A "hidden" one puts
        nothing on: its step is the model before it with that step's hand paint. Kept from call to call: the
        model as loaded, each repaint's look and picture, and its weights per fade - so a change of one repaint
        costs that one's blend, laying the pictures on, and writing the models; of the repaints no longer asked,
        the KEPT_SPARE used last (each holds tens of megabytes). `ahead`: the repaint whose
        weights for the slider's other fades (FADE_STEPS) are worked out meanwhile, on a thread, nearest first, so
        that the next change finds its own ready."""
        prop = self.prop
        self._stop_ahead.set()                       # (the thread lets go after the fade it is at)
        if self._loaded is None:
            self._loaded = prop.snapshot()
        prop.restore(self._loaded)
        writes: list[Future[None]] = []
        painted: list[int] = []
        kept: list[dict[str, Any] | None] = []
        try:
            for asked, lens, out in zip(repaints, lenses, outs or [None] * len(repaints)):
                shown = not asked.get("hidden")
                layer = asked.get("layer") if outs is not None else None
                kept.append(self._kept_repaint(asked, lens) if shown or layer else None)
                laid = self._kept_fade(kept[-1], clamped_fade(asked.get("fade"))) if kept[-1] else None
                painted.append(len(laid[0]) if shown else 0)
                if outs is None:
                    continue
                if shown:
                    prop.mix_in(*laid)
                if layer:                            # (where it lies, or would: the page shows it either way)
                    writes.append(self.encoders.submit(prop.write_layer, Path(layer), laid[0], on_clay(*laid[1:])))
                for path, model in prop.step_models(out, asked.get("hand_paint"), asked.get("unpainted")):
                    writes.append(self.encoders.submit(prop.write_model, path, model, True))
        finally:
            wait(writes)                             # (nothing is still being written when the answer goes out)
            for key in list(self.kept)[:-(len(repaints) + KEPT_SPARE)]:
                del self.kept[key]
            if ahead is not None and ahead < len(kept) and kept[ahead]:
                self._stop_ahead = stop = threading.Event()
                self.ahead.submit(self._fades_ahead, kept[ahead], float(repaints[ahead].get("fade") or 1.0), stop)
        for written in writes:
            written.result()
        return {"painted": painted}

    def _kept_repaint(self, asked: dict[str, Any], lens: PageLens) -> dict[str, Any]:
        """What `fades` keeps of a repaint: its look (without pictures), its picture, its weights per fade - by
        what they are made from (_made_from), so wherever the repaint stands among those asked."""
        key = _made_from(asked, lens)
        kept = self.kept.pop(key, None)
        if kept is None:
            self.look(lens, Path(asked["mask"]), pictures=False)
            kept = {"look": self.at, "whole": self.prop.picture(Path(asked["picture"]), lens.width), "fades": {}}
        self.kept[key] = kept
        return kept

    def _kept_fade(self, kept: dict[str, Any], fade: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A kept repaint's texels, colours and weights at `fade` (_onto's): kept, or being worked out ahead (it is
        waited for), or worked out here."""
        done, mine = self._claim(kept, fade)
        if mine:
            self._work_out(kept, fade, done)
        return done.result()

    def _claim(self, kept: dict[str, Any], fade: float) -> tuple[Future, bool]:
        """The Future of a kept repaint's `fade`, and whether the caller is the one to work it out."""
        with self._lock:
            done = kept["fades"].get(fade)
            if done is not None:
                return done, False
            done = kept["fades"][fade] = Future()
            return done, True

    def _work_out(self, kept: dict[str, Any], fade: float, done: Future) -> None:
        try:
            done.set_result(self._onto(kept["look"], kept["whole"], self.weight(kept["look"], fade)))
        except BaseException as exc:  # noqa: BLE001 - whoever waits for this fade hears of it
            with self._lock:
                if kept["fades"].get(fade) is done:  # (not kept: it is worked out anew the next time)
                    del kept["fades"][fade]
            done.set_exception(exc)

    def _fades_ahead(self, kept: dict[str, Any], now: float, stop: threading.Event) -> None:
        """The slider's other fades of a kept repaint worked out, nearest to `now` first, until `stop` is set (the
        next "fades" sets it)."""
        for fade in sorted(FADE_STEPS, key=lambda value: abs(value - now)):
            if stop.is_set():
                return
            done, mine = self._claim(kept, fade)
            if mine:
                self._work_out(kept, fade, done)
