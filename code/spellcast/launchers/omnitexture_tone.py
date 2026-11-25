"""Colour and tone for Omni-Texture's worker; what is mixed, is mixed in linear light.

keep_colour: a painting's colour at its albedo's brightness. tone_table / recolour: a picture brought to the tone
of another, per colour, read from the surface both show. match_tone: the model's own texture, where no camera
painted, retinted to the tone of the paintings near it on the model."""

from __future__ import annotations

import numpy as np

GAMMA = 2.2                  # the albedo add-on writes linear albedo with this gamma
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)      # Rec. 709 luminance of linear RGB
DARK = (0.002, 0.02)         # linear luminance: a pixel this dark has no colour to keep or compare
CLAY = (126, 133, 145)       # sRGB, the page's bare clay: viewer3d.js CLAY
LAYER_GREY = CLAY            # a view's layer where the view painted nothing
# match_tone. Where no camera painted, the model's own texture stays, lighter and blander than the paintings
# around it: it takes their tone, read from patches of the surface, the smallest half the blend's reach across,
# each twice the size of the one before.
TONE_FULL = 0.25             # of a patch's texels: painted this much, the patch's own change is taken whole; less,
TONE_TEXELS = 8              # (or fewer texels than this) the larger patch's around it is mixed in
TONE_LIMIT = (0.4, 2.5)      # the most a colour is dimmed or brightened (per channel)
# tone_table. Pictures painted apart differ in tone, and they meet on the model. The difference is one per
# colour, not one curve per channel, and is read from the surface both pictures show: never from a picture's
# empty background, whatever share of it that is.
AGREE_BINS = 8               # per channel: the cells of a picture's colours its change is read in
AGREE_PAIRS = 2000           # texels two pictures must share, alike in both, to be matched at all
AGREE_SUPPORT = 300.0        # texels of a colour at which what was read for it counts half (fewer: the change of all)
AGREE_ALIKE = 1.6            # a texel brighter by more than this in one picture than in the other, in any channel
                             # (past the change of all): not the same thing in both (an edge, a detail), left out
AGREE_LIMIT = (0.6, 1.6)     # the most a colour is changed; pictures apart by more overall are not matched
AGREE_FLOOR = 0.003          # added to compare channels by (a channel near nothing has no ratio)

_LINEAR = (np.arange(256, dtype=np.float32) / 255.0) ** GAMMA      # every byte's (v / 255) ** GAMMA, float32


def _linear(rgb: np.ndarray) -> np.ndarray:
    """(rgb / 255) ** GAMMA: looked up for bytes (float32, the values float32 bytes give), computed for floats."""
    return _LINEAR[rgb] if rgb.dtype == np.uint8 else (rgb / 255.0) ** GAMMA


def _levels(linear: np.ndarray) -> np.ndarray:
    """Linear light back to levels (0..255, not rounded)."""
    return np.clip(linear, 0.0, 1.0) ** (1.0 / GAMMA) * 255.0


def to_bytes(levels: np.ndarray) -> np.ndarray:
    """Levels rounded to bytes (bytes are given back as they are)."""
    return levels if levels.dtype == np.uint8 else np.clip(np.round(levels), 0, 255).astype(np.uint8)


def laid_over(under: np.ndarray, over: np.ndarray, share: np.ndarray) -> np.ndarray:
    """`over` on `under` at `share` ((n, 1), 0..1) of it: a plain blend of the sRGB levels, as bytes."""
    return to_bytes(under * (1.0 - share) + over * share)


def on_clay(colours: np.ndarray, share: np.ndarray) -> np.ndarray:
    """A layer's colours: a picture's `share` of its `colours` over the page's bare clay."""
    return laid_over(np.array(LAYER_GREY), colours, share)


def keep_colour(albedo: np.ndarray, painting: np.ndarray) -> np.ndarray:
    """The painting's colour at the albedo's brightness (both RGB 0..255, one size), per pixel:
    painting * Y(albedo) / Y(painting). The albedo add-on takes the shading off, but also a share of an object's
    own colour that varies per view (it cannot tell coloured paint from coloured light), while Qwen-Image's views
    agree in colour. Where the painting is nearly black its colour says nothing, and the albedo's own is used."""
    a, p = _linear(albedo), _linear(painting)
    ya, yp = a @ LUMA, p @ LUMA
    mixed = p * (ya / np.maximum(yp, 1e-6))[..., None]
    trust = np.clip((yp - DARK[0]) / (DARK[1] - DARK[0]), 0.0, 1.0)[..., None]
    out = trust * mixed + (1.0 - trust) * a
    return _levels(out)


# ---- the model's own texture in the tone of the paintings near it --------------------------------------------------

def _cell_keys(cells: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """One number per cell of a grid (n, 3: from -1 to dims - 2 along each axis), rising with the cells' order."""
    return ((cells[:, 0] + 1) * dims[1] + cells[:, 1] + 1) * dims[2] + cells[:, 2] + 1


def _cells(keys: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """The cells of _cell_keys' numbers."""
    return np.stack([keys // (dims[1] * dims[2]) - 1, keys // dims[2] % dims[1] - 1, keys % dims[2] - 1], axis=1)


def _between(keys: np.ndarray, dims: np.ndarray, values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """`values` (a row per cell there is of a grid, the cells named by their sorted `keys`) at the places `at`
    (n, 3, in cells, a cell's middle at its whole number): between the middles of the up to 8 cells around each
    place, those there are (one at the least: the cell the place lies in)."""
    base = np.floor(at).astype(np.int64)
    t = at - base
    total = np.zeros(len(at))
    out = np.zeros((len(at), values.shape[1]))
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                want = _cell_keys(base + (a, b, c), dims)
                found = np.minimum(np.searchsorted(keys, want), len(keys) - 1)
                weight = ((t[:, 0] if a else 1.0 - t[:, 0]) * (t[:, 1] if b else 1.0 - t[:, 1])
                          * (t[:, 2] if c else 1.0 - t[:, 2]) * (keys[found] == want))
                total += weight
                out += weight[:, None] * values[found]
    return out / total[:, None]


def _paintings_alone(own: np.ndarray, paint: np.ndarray, share: np.ndarray) -> np.ndarray:
    """The paintings' own colours (linear): the paint without the own texture's share of it, where they mix."""
    now = _linear(paint)
    mixed = np.flatnonzero((share > 0.0) & (share < 1.0))
    if mixed.size:
        s = share[mixed, None].astype(np.float64)
        now[mixed] = _linear(np.clip((paint[mixed] - s * own[mixed]) / (1.0 - s), 0.0, 255.0))
    return now


def _patches(points: np.ndarray, lo: np.ndarray, grain: float, painted: np.ndarray, now: np.ndarray,
             was: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray]:
    """The surface in patches: the texels in cubes `grain` across, then in cubes twice as large, and so on up to
    one patch, the whole model -> per size (the patches' keys, their grid's dims, their sums: the paintings' and
    the own texture's under them (3 + 3), the texels painted, the texels), and each texel's smallest patch."""
    cells = np.floor((points - lo) / grain).astype(np.int64)
    dims = cells.max(axis=0) + 3
    keys, patch = np.unique(_cell_keys(cells, dims), return_inverse=True)
    sums = np.stack([np.bincount(patch, painted * values[:, c], minlength=len(keys))
                     for values in (now, was) for c in range(3)]
                    + [np.bincount(patch, painted, minlength=len(keys)),
                       np.bincount(patch, minlength=len(keys)).astype(np.float64)], axis=1)
    levels = [(keys, dims, sums)]
    smallest = patch
    cells = _cells(keys, dims)
    while cells.max() > 0:
        cells = cells >> 1
        dims = cells.max(axis=0) + 3
        keys, patch = np.unique(_cell_keys(cells, dims), return_inverse=True)
        sums = np.stack([np.bincount(patch, sums[:, f], minlength=len(keys)) for f in range(sums.shape[1])], axis=1)
        levels.append((keys, dims, sums))
        cells = _cells(keys, dims)
    return levels, smallest


def _gains(levels: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each smallest patch's change (the paintings' sum over the own texture's, per channel), worked down from the
    whole model: a patch painted enough has its own, one painted less mixes in the larger patch's around it,
    between that one's neighbours -> (the smallest patches' keys, their grid's dims, their gains)."""
    gain = above = None
    for keys, dims, sums in reversed(levels):
        mine = sums[:, :3] / np.maximum(sums[:, 3:6], 1e-12)
        if above is not None:
            sure = np.clip(sums[:, 6] / np.maximum(TONE_FULL * sums[:, 7], TONE_TEXELS), 0.0, 1.0)[:, None]
            mine = sure * mine + (1.0 - sure) * _between(*above, gain, (_cells(keys, dims) + 0.5) / 2.0 - 0.5)
        gain, above = np.clip(mine, *TONE_LIMIT), (keys, dims)
    return (*above, gain)


def match_tone(points: np.ndarray, own: np.ndarray, paint: np.ndarray, share: np.ndarray, grain: float) -> np.ndarray:
    """The paint with the model's own texture's share of it retinted to the tone of the paintings near it on the
    model. points (n, 3); own, paint: (n, 3) uint8, the model's own colours and what the texture holds now; share
    (n,): how much of `paint` is still `own` (0: a painting whole, 1: nothing painted there).
    The change the paintings made is read from patches of the surface (_patches, _gains): the own texture next to
    a painting takes the change that painting made right there - the two meet in one tone -, deeper into an
    unpainted part the change of ever more of the surroundings, and its own detail stays. Not read per colour of
    the own texture over the whole model: a shadow baked into it would take what the paintings made of that
    colour elsewhere."""
    shows = np.flatnonzero(share > 0.0)
    painted = np.clip(1.0 - share, 0.0, 1.0).astype(np.float64)
    if not shows.size or not painted.any():
        return paint
    lo = points.min(axis=0)
    levels, patch = _patches(points, lo, grain, painted, _paintings_alone(own, paint, share), _linear(own))
    shows = shows[np.argsort(patch[shows], kind="stable")]       # (patch by patch: the look-ups below in order)
    mine = _between(*_gains(levels), (points[shows] - lo) / grain - 0.5)
    retinted = _levels(_linear(own[shows]) * mine)
    out = paint.copy()
    out[shows] = np.clip(np.round(paint[shows] + share[shows, None] * (retinted - own[shows])), 0, 255)
    return out


# ---- a picture in the tone of another ------------------------------------------------------------------------------

def _colour_cells(colours: np.ndarray, bins: int) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Where each colour (n, 3; bytes) lies among `bins` cells a channel -> the cell below it per channel (n, 3) and
    per channel its weights for that cell and the next (a cell's middle is its colour)."""
    at = colours.astype(np.float32) * (bins / 256.0) - 0.5
    base = np.clip(np.floor(at).astype(np.int64), 0, bins - 2)
    t = np.clip(at - base, 0.0, 1.0)
    return base, [(1.0 - t[:, i], t[:, i]) for i in range(3)]


def _around(base: np.ndarray, along: list[tuple[np.ndarray, np.ndarray]], bins: int):
    """The 8 cells around each colour (_colour_cells): each cell's number and the colour's weight for it."""
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                yield (((base[:, 0] + i) * bins + base[:, 1] + j) * bins + base[:, 2] + k,
                       along[0][i] * along[1][j] * along[2][k])


def _same_things(ours: np.ndarray, theirs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The pairs that show one thing in both pictures -> (ours, theirs: linear, and which of the pairs given they
    are), or None: too few (AGREE_PAIRS, or under half of those lit), or the pictures differ by more than a tone
    (AGREE_LIMIT). Left out: pairs too dark to compare, and those whose two colours are not one thing
    (AGREE_ALIKE)."""
    a, b = _linear(ours).astype(np.float64), _linear(theirs).astype(np.float64)
    lit = ((a @ LUMA) > DARK[1]) & ((b @ LUMA) > DARK[1])
    if np.count_nonzero(lit) < AGREE_PAIRS:
        return None
    apart = np.log((a + AGREE_FLOOR) / (b + AGREE_FLOOR))
    overall = np.median(apart[lit], axis=0)
    if (np.abs(overall) > np.log(AGREE_LIMIT[1])).any():
        return None
    alike = lit & (np.abs(apart - overall) < np.log(AGREE_ALIKE)).all(axis=1)
    if np.count_nonzero(alike) < max(AGREE_PAIRS, 0.5 * np.count_nonzero(lit)):
        return None
    return a[alike], b[alike], alike


def tone_table(theirs: np.ndarray, ours: np.ndarray) -> np.ndarray | None:
    """What brings a picture's colours to the tone of another's: (AGREE_BINS, ^, ^, 3) gains, per colour of the
    picture, read from pairs of colours of the same texels - theirs (n, 3; bytes): the picture's, ours: the
    other's. Per cell of their colour the sum of ours over the sum of theirs, each pair counted in the cells
    around its colour and the cells smoothed with their neighbours, so the gain changes evenly from colour to
    colour; a colour met in few pairs takes the change of all of them. None when the pictures cannot be matched
    (_same_things)."""
    pairs = _same_things(ours, theirs)
    if pairs is None:
        return None
    a, b, alike = pairs
    bins = AGREE_BINS
    sums = np.zeros((bins ** 3, 7))                                 # ours (3), theirs (3), the pairs
    for cell, weight in _around(*_colour_cells(theirs[alike], bins), bins):
        for f, values in enumerate((*a.T, *b.T)):
            sums[:, f] += np.bincount(cell, weight * values, minlength=bins ** 3)
        sums[:, 6] += np.bincount(cell, weight, minlength=bins ** 3)
    sums = sums.reshape(bins, bins, bins, 7)
    for axis in range(3):                                           # neighbouring colours share what they met
        padded = np.pad(sums, [(1, 1) if i == axis else (0, 0) for i in range(4)])
        sums = sum(part * np.take(padded, np.arange(i, i + bins), axis=axis) for i, part in enumerate((0.25, 0.5, 0.25)))
    ours, theirs = a.sum(axis=0), b.sum(axis=0)
    everywhere = np.divide(ours, theirs, out=np.ones(3), where=theirs > 0)   # (no light in a channel: no change)
    sure = sums[..., 6:] / (sums[..., 6:] + AGREE_SUPPORT)
    gain = sure * sums[..., :3] / np.maximum(sums[..., 3:6], 1e-12) + (1.0 - sure) * everywhere
    return np.clip(gain, *AGREE_LIMIT).astype(np.float32)


def recolour(colours: np.ndarray, table: np.ndarray) -> np.ndarray:
    """The colours (n, 3; bytes) in the tone tone_table read: each times the gain of its colour, between the
    table's cells around it."""
    bins = table.shape[0]
    flat = table.reshape(-1, 3)
    gain = np.zeros(colours.shape, np.float32)
    for cell, weight in _around(*_colour_cells(colours, bins), bins):
        gain += weight.astype(np.float32)[:, None] * flat[cell]
    return to_bytes(_levels(_linear(colours) * gain))
