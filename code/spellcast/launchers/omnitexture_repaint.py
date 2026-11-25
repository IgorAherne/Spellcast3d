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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image
from spellbake import BakeError
from omnitexture_blend import blend, within_reach
from omnitexture_masks import SPECK, clean, grow, mask_left
from omnitexture_pixels import between, nearest
from omnitexture_tone import on_clay, recolour, to_bytes, tone_table
from omnitexture_views import FACING, Lens, cast, depth_grey

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
FADE_PNG = 3                 # a fade's models' texture: zlib level 3 (a third of level 6's time, 6 % larger)
FADE_MARGIN = 4.0            # reaches around the repainted texels within which the blend is worked out
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
    lens: Lens
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
        self.kept: dict[int, dict[str, Any]] = {}    # fades: per repaint its camera, look, picture, weights per fade
        self._loaded: tuple | None = None            # fades: the model as loaded (Prop.snapshot)

    def close(self) -> None:
        self._stop_ahead.set()
        self.ahead.shutdown(wait=True)
        self.encoders.shutdown(wait=True)

    # ---- the look --------------------------------------------------------------------------------------------------
    def look(self, lens: Lens, mask: Path, pictures: bool = True) -> dict[str, Any]:
        """The worker's "look" through `lens`: mask is a picture of the texture's size, white where the artist
        masked. Without `pictures` only where a picture made through this camera before would go."""
        prop, size = self.prop, lens.width
        marked = self._marked(mask)
        masked = marked.reshape(-1)[prop.texel]
        if not masked.any():
            raise BakeError(NOTHING_MASKED)
        rays = lens.rays()
        met = cast(prop.scene, prop.mesh.faces, np.broadcast_to(lens.eye, rays.shape), rays)
        if not met.hit.any():
            raise BakeError(MODEL_OUT_OF_SIGHT)
        uv = met.between(prop.mesh.uv)
        depth = np.full(size * size, np.inf)
        depth[met.hit] = met.t * (rays[met.hit] @ lens.forward)
        over = self._masked_pixels(uv, marked)
        shown = np.zeros(size * size, dtype=bool)
        shown[met.hit] = over
        # Where its picture may go: what faces it, without specks; then the texels of that, away from snaps.
        facing = np.zeros(size * size, dtype=bool)
        facing[met.hit] = (met.unit(prop.smooth) * -rays[met.hit]).sum(axis=1) >= FACING
        model = met.hit.reshape(size, size)
        chosen, px, py, _ = prop.targets(lens, clean(facing.reshape(size, size), SPECK) & model,
                                         depth.reshape(size, size))
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
        surface itself (the gutter alone would speck the islands that lie beside a masked one in the texture)."""
        res = self.prop.res
        grown = marked | (grow(marked, MASK_GUTTER) & ~self.prop.inside.reshape(res, res))
        return ((self._textured(uv, grown.reshape(res, res, 1).astype(np.float32))[:, 0] > 0.5)
                & (self._textured(uv, marked.reshape(res, res, 1).astype(np.float32))[:, 0] > 0.0))

    def _pictures(self, lens: Lens, hit: np.ndarray, grey: np.ndarray, viewport: np.ndarray,
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
    def repaint(self, look: Look, picture: Path, colour: Path | None = None, keep: Path | None = None,
                layer: Path | None = None, fade: float = 1.0, tone: bool = True) -> dict[str, Any]:
        """The worker's "repaint": Qwen-Image's picture of what the look showed it onto the masked texels."""
        prop, size = self.prop, look.lens.width
        whole = to_bytes(prop.to_project(picture, colour, size)[0])
        matched = 0
        if prop.agree and tone:
            whole, matched = self._in_tone(look, whole)
        if keep is not None:
            Image.fromarray(whole).save(keep)
        to, sampled, a = self._onto(look, whole, self.weight(look, fade))
        prop.mix_in(to, sampled, a)
        if layer is not None:
            prop.write_layer(layer, to, on_clay(sampled, a))
        prop.resettle()
        return {"painted": len(to), "matched": matched}

    def _in_tone(self, look: Look, whole: np.ndarray) -> tuple[np.ndarray, int]:
        """The picture in the tone of the texture around the mask: read where it shows surface that is not masked,
        as a later picture's tone is read against the earlier ones' -> the picture, and the texels the tone was
        read from (0: left as it came). All of that surface, the part held to the view too: taking the lighting
        off changes the tone of the whole picture alike, far more than the model's own drawing beside the mask."""
        prop, size = self.prop, look.lens.width
        col, row = nearest(look.px, look.py, size, size)
        beside = ~look.masked[look.chosen]
        table = tone_table(whole.reshape(-1, 3)[(row * size + col)[beside]],
                           prop.paint[prop.texel[look.chosen][beside]])
        if table is None:
            return whole, 0
        return recolour(whole.reshape(-1, 3), table).reshape(size, size, 3), int(beside.sum())

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
    def fades(self, repaints: list[dict[str, Any]], lenses: list[Lens], outs: list[str] | None,
              ahead: int | None = None) -> dict[str, Any]:
        """The worker's "fades": the repaints (each through its lens) onto the model as loaded, one after the
        other, each with what was painted by hand on its step over it (Prop.step_models). Kept from call to call:
        the model as loaded, each repaint's look and picture, and its weights per fade - so a change of one
        repaint's fade costs that one's blend, laying the pictures on, and writing the models. `ahead`: the
        repaint whose weights for the slider's other fades (FADE_STEPS) are worked out meanwhile, on a thread,
        nearest first, so that the next change finds its own ready."""
        prop = self.prop
        self._stop_ahead.set()                       # (the thread lets go after the fade it is at)
        if self._loaded is None:
            self._loaded = prop.snapshot()
        prop.restore(self._loaded)
        writes: list[Future[None]] = []
        painted: list[int] = []
        try:
            for i, (asked, lens) in enumerate(zip(repaints, lenses)):
                kept = self._kept_repaint(i, asked, lens)
                to, sampled, a = self._kept_fade(kept, clamped_fade(asked.get("fade")))
                painted.append(len(to))
                if outs is None:
                    continue
                prop.mix_in(to, sampled, a)
                if asked.get("layer"):
                    writes.append(self.encoders.submit(prop.write_layer, Path(asked["layer"]), to,
                                                       on_clay(sampled, a)))
                for out, model in prop.step_models(outs[i], asked.get("hand_paint"), asked.get("unpainted")):
                    writes.append(self.encoders.submit(prop.write_model, out, model, FADE_PNG))
        finally:
            wait(writes)                             # (nothing is still being written when the answer goes out)
            if ahead is not None and ahead in self.kept:
                self._stop_ahead = stop = threading.Event()
                self.ahead.submit(self._fades_ahead, self.kept[ahead], float(repaints[ahead].get("fade") or 1.0),
                                  stop)
        for written in writes:
            written.result()
        return {"painted": painted}

    def _kept_repaint(self, i: int, asked: dict[str, Any], lens: Lens) -> dict[str, Any]:
        """What `fades` keeps of its i-th repaint: its camera, its look (without pictures), its picture, its
        weights per fade. Made anew when the camera asked is another one."""
        camera = tuple(tuple(float(v) for v in asked[key]) for key in ("eye", "forward", "up")) + (float(asked["tan"]),)
        kept = self.kept.get(i)
        if kept is None or kept["camera"] != camera:
            self.look(lens, Path(asked["mask"]), pictures=False)
            kept = self.kept[i] = {"camera": camera, "look": self.at,
                                   "whole": self.prop.picture(Path(asked["picture"]), lens.width), "fades": {}}
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
